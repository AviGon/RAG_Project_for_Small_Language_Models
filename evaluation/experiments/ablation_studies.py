"""
Ablation studies for RAG system.
Tests impact of different configurations:
- Top-k values (number of retrieved contexts)
- Chunk sizes (requires regenerating embeddings)
- Temperature settings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from typing import Dict, List
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from metrics.latency_metrics import LatencyProfile, LatencyTracker, compute_latency_stats
from metrics.generation_metrics import compute_all_metrics, aggregate_metrics


# ==============================
# CONFIG
# ==============================
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EVALUATION_DATASET = "evaluation/dataset/evaluation_dataset.json"
MAX_NEW_TOKENS = 300


class AblationStudy:
    """Run ablation studies on RAG system"""
    
    def __init__(self, use_cuda: bool = None):
        """Initialize models and indices"""
        print("Initializing Ablation Study Framework...")
        
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        self.device = "cuda" if use_cuda else "cpu"
        print(f"Using device: {self.device}")
        
        # Load components
        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_FILE)
        
        print("Loading metadata...")
        with open(METADATA_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        print("Loading LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None
        )
        
        if not use_cuda:
            self.model = self.model.to(self.device)
        
        print("Initialization complete!\n")
    
    def retrieve_context(self, query: str, k: int) -> List[str]:
        """Retrieve top-k contexts"""
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        
        return retrieved_chunks
    
    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """Build RAG prompt"""
        context_text = "\n\n".join(contexts)
        prompt = f"""You are an assistant answering questions strictly using the provided context.
If the answer is not present in the context, respond with:
"I don't know based on the provided document."

Context:
{context_text}

Question:
{query}

Answer:"""
        return prompt.strip()
    
    def generate_answer(self, prompt: str, temperature: float = 0.2, do_sample: bool = False) -> str:
        """Generate answer with configurable parameters"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            do_sample=do_sample
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text.strip()
        
        return answer
    
    def evaluate_with_config(self, questions: List[Dict], top_k: int, 
                            temperature: float = 0.2) -> Dict:
        """Evaluate with specific configuration"""
        results = []
        latencies = []
        
        for q in tqdm(questions, desc=f"k={top_k}, temp={temperature}"):
            profile = LatencyProfile()
            
            # Retrieve
            with LatencyTracker() as timer:
                contexts = self.retrieve_context(q['question'], top_k)
            profile.retrieval_time = timer.get_elapsed()
            
            # Build prompt
            with LatencyTracker() as timer:
                prompt = self.build_prompt(q['question'], contexts)
            profile.prompt_construction_time = timer.get_elapsed()
            
            # Generate
            with LatencyTracker() as timer:
                answer = self.generate_answer(prompt, temperature)
            profile.generation_time = timer.get_elapsed()
            
            profile.total_time = profile.retrieval_time + profile.prompt_construction_time + profile.generation_time
            
            # Compute metrics
            metrics = compute_all_metrics(answer, q['reference_answer'], contexts)
            metrics['question_id'] = q['id']
            metrics['in_document'] = q['in_document']
            
            results.append(metrics)
            latencies.append(profile)
        
        # Aggregate
        aggregated = aggregate_metrics(results)
        latency_stats = compute_latency_stats(latencies)
        
        return {
            "metrics": aggregated,
            "latency": latency_stats.means.to_dict(),
            "raw_results": results
        }
    
    def run_top_k_ablation(self, dataset_path: str, k_values: List[int], 
                          output_dir: str):
        """Test different top-k values"""
        print("="*60)
        print("Top-K Ablation Study")
        print("="*60)
        print(f"Testing k values: {k_values}\n")
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Filter to in-document questions only
        questions = [q for q in data['questions'] if q['in_document']]
        print(f"Using {len(questions)} in-document questions\n")
        
        results = {}
        
        for k in k_values:
            print(f"\nEvaluating with top-k = {k}")
            result = self.evaluate_with_config(questions, k)
            results[f"k_{k}"] = result
            
            # Print summary
            print(f"  ROUGE-L: {result['metrics'].get('rouge_l_mean', 0):.4f}")
            print(f"  F1: {result['metrics'].get('f1_score_mean', 0):.4f}")
            print(f"  Latency: {result['latency']['total_ms']:.1f}ms")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/top_k_ablation.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate comparison report
        self._print_top_k_comparison(results, k_values)
        
        print(f"\n✓ Results saved to {output_dir}/top_k_ablation.json")
    
    def run_temperature_ablation(self, dataset_path: str, 
                                 temperatures: List[float], output_dir: str):
        """Test different temperature settings"""
        print("="*60)
        print("Temperature Ablation Study")
        print("="*60)
        print(f"Testing temperatures: {temperatures}\n")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        questions = [q for q in data['questions'] if q['in_document']]
        print(f"Using {len(questions)} in-document questions\n")
        
        results = {}
        
        for temp in temperatures:
            print(f"\nEvaluating with temperature = {temp}")
            # Use fixed k=5 for temperature study
            result = self.evaluate_with_config(questions, top_k=5, temperature=temp)
            results[f"temp_{temp}"] = result
            
            print(f"  ROUGE-L: {result['metrics'].get('rouge_l_mean', 0):.4f}")
            print(f"  F1: {result['metrics'].get('f1_score_mean', 0):.4f}")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/temperature_ablation.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {output_dir}/temperature_ablation.json")
    
    def _print_top_k_comparison(self, results: Dict, k_values: List[int]):
        """Print formatted comparison table"""
        print("\n" + "="*80)
        print("Top-K Comparison Summary")
        print("="*80)
        print(f"{'k':<5} {'ROUGE-L':>10} {'F1':>10} {'Exact Match':>13} {'Latency (ms)':>15} {'Grounding':>12}")
        print("-"*80)
        
        for k in k_values:
            key = f"k_{k}"
            metrics = results[key]['metrics']
            latency = results[key]['latency']
            
            rouge = metrics.get('rouge_l_mean', 0)
            f1 = metrics.get('f1_score_mean', 0)
            em = metrics.get('exact_match_mean', 0)
            lat = latency['total_ms']
            grounding = metrics.get('grounding_score_mean', 0)
            
            print(f"{k:<5} {rouge:>10.4f} {f1:>10.4f} {em:>13.4f} {lat:>15.1f} {grounding:>12.4f}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies on RAG system")
    parser.add_argument("--study", type=str, choices=["top_k", "temperature", "all"],
                        default="all", help="Which ablation study to run")
    parser.add_argument("--dataset", type=str, default=EVALUATION_DATASET,
                        help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, default="evaluation/results/ablation",
                        help="Output directory")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Top-k values to test")
    parser.add_argument("--temperatures", type=float, nargs="+", 
                        default=[0.0, 0.2, 0.5, 0.7],
                        help="Temperature values to test")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA")
    
    args = parser.parse_args()
    
    ablation = AblationStudy(use_cuda=not args.no_cuda)
    
    if args.study in ["top_k", "all"]:
        ablation.run_top_k_ablation(args.dataset, args.k_values, args.output)
    
    if args.study in ["temperature", "all"]:
        ablation.run_temperature_ablation(args.dataset, args.temperatures, args.output)


if __name__ == "__main__":
    main()
