"""
Main evaluation script for comparing RAG vs No-RAG performance.
Runs both modes on the evaluation dataset and computes metrics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
from typing import Dict, List, Tuple
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Import evaluation metrics
from metrics.latency_metrics import (
    LatencyProfile, LatencyTracker, compute_latency_stats,
    format_latency_report, compare_latencies, save_latency_results
)
from metrics.generation_metrics import (
    compute_all_metrics, aggregate_metrics,
    format_metrics_report, compare_metrics
)


# ==============================
# CONFIG
# ==============================
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
EVALUATION_DATASET = "evaluation/dataset/evaluation_dataset.json"

TOP_K = 5
MAX_NEW_TOKENS = 300


class RAGEvaluator:
    """Main evaluation class for RAG system"""
    
    def __init__(self, use_cuda: bool = None):
        """Initialize models and indices"""
        print("Initializing RAG Evaluator...")
        
        # Auto-detect CUDA if not specified
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        
        self.device = "cuda" if use_cuda else "cpu"
        print(f"Using device: {self.device}")
        
        # Load FAISS index
        print("Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_FILE)
        
        # Load metadata (chunks)
        print("Loading metadata...")
        with open(METADATA_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        
        # Load embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        # Load LLM
        print("Loading LLM (Phi-3)...")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None
        )
        
        if not use_cuda:
            self.model = self.model.to(self.device)
        
        print("Initialization complete!\n")
    
    def retrieve_context(self, query: str, k: int = TOP_K) -> Tuple[List[str], float]:
        """
        Retrieve top-k relevant contexts for a query.
        Returns: (list of contexts, retrieval time in seconds)
        """
        with LatencyTracker() as timer:
            query_embedding = self.embed_model.encode(
                [query],
                normalize_embeddings=True
            )
            query_embedding = np.array(query_embedding)
            
            scores, indices = self.index.search(query_embedding, k)
            retrieved_chunks = [self.chunks[i] for i in indices[0]]
        
        return retrieved_chunks, timer.get_elapsed()
    
    def build_prompt(self, query: str, contexts: List[str] = None) -> Tuple[str, float]:
        """
        Build prompt for LLM (with or without context).
        Returns: (prompt, construction time in seconds)
        """
        with LatencyTracker() as timer:
            if contexts:
                # RAG mode: include context
                context_text = "\n\n".join(contexts)
                prompt = f"""You are an assistant answering questions strictly using the provided context.
If the answer is not present in the context, respond with:
"I don't know based on the provided document."

Context:
{context_text}

Question:
{query}

Answer:"""
            else:
                # No-RAG mode: direct question
                prompt = f"""Answer the following question concisely.

Question:
{query}

Answer:"""
        
        return prompt.strip(), timer.get_elapsed()
    
    def generate_answer(self, prompt: str) -> Tuple[str, float]:
        """
        Generate answer using LLM.
        Returns: (generated answer, generation time in seconds)
        """
        with LatencyTracker() as timer:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.2,
                do_sample=False
            )
            
            # Decode and extract only the new tokens (not the prompt)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to extract just the answer part
            if "Answer:" in full_text:
                answer = full_text.split("Answer:")[-1].strip()
            else:
                answer = full_text.strip()
        
        return answer, timer.get_elapsed()
    
    def evaluate_question_rag(self, question: str) -> Tuple[str, List[str], LatencyProfile]:
        """
        Evaluate a single question using RAG mode.
        Returns: (answer, retrieved_contexts, latency_profile)
        """
        profile = LatencyProfile()
        start_total = time.perf_counter()
        
        # Step 1: Encode query
        with LatencyTracker() as timer:
            query_embedding = self.embed_model.encode([question], normalize_embeddings=True)
        profile.query_encoding_time = timer.get_elapsed()
        
        # Step 2: Retrieve contexts
        contexts, retrieval_time = self.retrieve_context(question)
        profile.retrieval_time = retrieval_time
        
        # Step 3: Build prompt
        prompt, prompt_time = self.build_prompt(question, contexts)
        profile.prompt_construction_time = prompt_time
        
        # Step 4: Generate answer
        answer, gen_time = self.generate_answer(prompt)
        profile.generation_time = gen_time
        
        profile.total_time = time.perf_counter() - start_total
        
        return answer, contexts, profile
    
    def evaluate_question_no_rag(self, question: str) -> Tuple[str, LatencyProfile]:
        """
        Evaluate a single question without RAG (baseline).
        Returns: (answer, latency_profile)
        """
        profile = LatencyProfile()
        start_total = time.perf_counter()
        
        # Build prompt (no context)
        prompt, prompt_time = self.build_prompt(question, contexts=None)
        profile.prompt_construction_time = prompt_time
        
        # Generate answer
        answer, gen_time = self.generate_answer(prompt)
        profile.generation_time = gen_time
        
        profile.total_time = time.perf_counter() - start_total
        
        return answer, profile
    
    def run_evaluation(self, dataset_path: str, output_dir: str = "evaluation/results"):
        """
        Run complete evaluation on the dataset.
        Compares RAG vs No-RAG performance.
        """
        # Load dataset
        print(f"Loading evaluation dataset from {dataset_path}...")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        questions = data['questions']
        print(f"Loaded {len(questions)} questions\n")
        
        # Storage for results
        rag_results = []
        no_rag_results = []
        rag_latencies = []
        no_rag_latencies = []
        
        # Evaluate each question
        print("="*60)
        print("Running Evaluation...")
        print("="*60)
        
        for i, q in enumerate(tqdm(questions, desc="Evaluating")):
            question_text = q['question']
            reference_answer = q['reference_answer']
            in_document = q['in_document']
            
            # RAG mode
            rag_answer, contexts, rag_latency = self.evaluate_question_rag(question_text)
            rag_metrics = compute_all_metrics(rag_answer, reference_answer, contexts)
            rag_metrics['question_id'] = q['id']
            rag_metrics['in_document'] = in_document
            rag_results.append(rag_metrics)
            rag_latencies.append(rag_latency)
            
            # No-RAG mode
            no_rag_answer, no_rag_latency = self.evaluate_question_no_rag(question_text)
            no_rag_metrics = compute_all_metrics(no_rag_answer, reference_answer, contexts=None)
            no_rag_metrics['question_id'] = q['id']
            no_rag_metrics['in_document'] = in_document
            no_rag_results.append(no_rag_metrics)
            no_rag_latencies.append(no_rag_latency)
            
            # Store detailed results for this question
            detailed_result = {
                'question_id': q['id'],
                'question': question_text,
                'reference_answer': reference_answer,
                'rag_answer': rag_answer,
                'no_rag_answer': no_rag_answer,
                'in_document': in_document,
                'category': q['category'],
                'rag_metrics': rag_metrics,
                'no_rag_metrics': no_rag_metrics
            }
            
            # Save individual results
            os.makedirs(f"{output_dir}/generation_quality", exist_ok=True)
            with open(f"{output_dir}/generation_quality/question_{q['id']}.json", 'w') as f:
                json.dump(detailed_result, f, indent=2)
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60 + "\n")
        
        # Compute aggregate statistics
        self._compute_and_save_results(
            rag_results, no_rag_results,
            rag_latencies, no_rag_latencies,
            output_dir
        )
    
    def _compute_and_save_results(self, rag_results, no_rag_results,
                                    rag_latencies, no_rag_latencies, output_dir):
        """Compute aggregate statistics and save results"""
        
        # Compute aggregate metrics
        rag_metrics_agg = aggregate_metrics(rag_results)
        no_rag_metrics_agg = aggregate_metrics(no_rag_results)
        
        # Separate in-document vs out-of-document
        rag_in_doc = [r for r in rag_results if r['in_document']]
        rag_out_doc = [r for r in rag_results if not r['in_document']]
        no_rag_in_doc = [r for r in no_rag_results if r['in_document']]
        no_rag_out_doc = [r for r in no_rag_results if not r['in_document']]
        
        rag_in_doc_metrics = aggregate_metrics(rag_in_doc)
        rag_out_doc_metrics = aggregate_metrics(rag_out_doc)
        no_rag_in_doc_metrics = aggregate_metrics(no_rag_in_doc)
        no_rag_out_doc_metrics = aggregate_metrics(no_rag_out_doc)
        
        # Compute latency statistics
        rag_latency_stats = compute_latency_stats(rag_latencies)
        no_rag_latency_stats = compute_latency_stats(no_rag_latencies)
        
        # Print reports
        print(format_metrics_report(rag_metrics_agg, "RAG"))
        print(format_metrics_report(no_rag_metrics_agg, "No-RAG"))
        print(compare_metrics(rag_metrics_agg, no_rag_metrics_agg))
        
        print(format_latency_report(rag_latency_stats, "RAG"))
        print(format_latency_report(no_rag_latency_stats, "No-RAG"))
        print(compare_latencies(rag_latency_stats, no_rag_latency_stats))
        
        # Save aggregate results
        os.makedirs(output_dir, exist_ok=True)
        
        summary = {
            "overall": {
                "rag": rag_metrics_agg,
                "no_rag": no_rag_metrics_agg
            },
            "in_document": {
                "rag": rag_in_doc_metrics,
                "no_rag": no_rag_in_doc_metrics
            },
            "out_of_document": {
                "rag": rag_out_doc_metrics,
                "no_rag": no_rag_out_doc_metrics
            },
            "latency": {
                "rag": rag_latency_stats.means.to_dict(),
                "no_rag": no_rag_latency_stats.means.to_dict()
            }
        }
        
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save latency details
        os.makedirs(f"{output_dir}/latency_profiles", exist_ok=True)
        save_latency_results(rag_latency_stats, f"{output_dir}/latency_profiles/rag_latency.json", "RAG")
        save_latency_results(no_rag_latency_stats, f"{output_dir}/latency_profiles/no_rag_latency.json", "No-RAG")
        
        print(f"\n✓ Results saved to {output_dir}/")
        print(f"  - Summary: {output_dir}/summary.json")
        print(f"  - Latency profiles: {output_dir}/latency_profiles/")
        print(f"  - Per-question results: {output_dir}/generation_quality/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG vs No-RAG performance")
    parser.add_argument("--dataset", type=str, default=EVALUATION_DATASET,
                        help="Path to evaluation dataset JSON")
    parser.add_argument("--output", type=str, default="evaluation/results",
                        help="Output directory for results")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = RAGEvaluator(use_cuda=not args.no_cuda)
    
    # Run evaluation
    evaluator.run_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()
