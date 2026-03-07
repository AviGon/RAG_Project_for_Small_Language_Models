"""
Latency-first RAG evaluation.
Profiles end-to-end latency and component latency for:
- Dense retrieval: FAISS and/or ChromaDB
- Sparse retrieval: BM25
- No-RAG baseline
Across configurable document counts and device settings.
"""

import argparse
import importlib
import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_DATASET = "evaluation/dataset/evaluation_dataset.json"
DEFAULT_OUTPUT = "evaluation/results/latency"
DEFAULT_CHROMA_DIR = "chroma_db"
DEFAULT_LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class LatencyProfile:
    query_encoding_ms: float = 0.0
    ann_search_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


def parse_int_csv(value: str) -> List[int]:
    parsed = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise ValueError("At least one integer value is required.")
    if any(item <= 0 for item in parsed):
        raise ValueError("All values must be > 0")
    return sorted(set(parsed))


class LatencyEvaluator:
    def __init__(
        self,
        use_cuda: bool,
        max_new_tokens: int,
        dense_backends: List[str],
        doc_counts: List[int],
        top_k: int,
        chroma_dir: str,
        llm_model_name: str,
        embed_model_name: str,
        warmup_runs: int,
    ):
        self.torch = importlib.import_module("torch")
        self.device = "cuda" if use_cuda and self.torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.dense_backends = dense_backends
        self.requested_doc_counts = doc_counts
        self.chroma_dir = chroma_dir
        self.llm_model_name = llm_model_name
        self.embed_model_name = embed_model_name
        self.warmup_runs = max(0, warmup_runs)

        self._load_assets()
        self._run_sanity_checks()
        self._prepare_indices()
        self._load_models()

    def _run_sanity_checks(self):
        print("\nSanity checks")
        print("-" * 40)
        print(f"NumPy version: {np.__version__}")
        print(f"Torch version: {self.torch.__version__}")

        numpy_major = int(np.__version__.split(".")[0])
        if numpy_major >= 2:
            print(
                "Warning: NumPy >=2 detected. If you hit Torch/FAISS import/runtime issues, "
                "pin NumPy to <2 in requirements and reinstall."
            )

        if self.device == "cpu":
            try:
                import psutil

                available_gb = psutil.virtual_memory().available / (1024 ** 3)
                print(f"Available RAM: {available_gb:.2f} GB")
                if available_gb < 8:
                    print(
                        "Warning: <8GB free RAM on CPU mode. Phi-3 may be unstable or very slow. "
                        "Consider --max-questions, lower --max-new-tokens, or GPU."
                    )
            except Exception:
                print("Note: install psutil for RAM pre-checks.")
        else:
            print("CUDA mode enabled.")

        if "chroma" in self.dense_backends and not CHROMA_AVAILABLE:
            raise ImportError("Chroma backend requested but chromadb is not installed.")

        if not BM25_AVAILABLE:
            raise ImportError(
                "BM25 sparse retrieval is enabled by default but rank-bm25 is not installed. "
                "Install dependencies with: python3 -m pip install -r requirements.txt"
            )

    def _load_assets(self):
        with open("chunks.pkl", "rb") as handle:
            self.chunks = pickle.load(handle)
        self.embeddings = np.load("embeddings.npy").astype("float32")

        if len(self.chunks) != self.embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.chunks)} chunks vs {self.embeddings.shape[0]} embeddings"
            )

        self.max_docs = len(self.chunks)
        self.doc_counts = [min(value, self.max_docs) for value in self.requested_doc_counts]
        self.doc_counts = sorted(set(self.doc_counts))

        if len(self.doc_counts) == 0:
            raise ValueError("No valid doc counts after bounds check.")

        print(f"Loaded corpus: {self.max_docs} chunks")
        print(f"Doc-count sweep: {self.doc_counts}")

    def _prepare_indices(self):
        self.faiss_indices: Dict[int, faiss.Index] = {}
        self.bm25_indices: Dict[int, Optional[BM25Okapi]] = {}
        self.chroma_collections = {}

        if "faiss" in self.dense_backends:
            for docs in self.doc_counts:
                index = faiss.IndexFlatIP(self.embeddings.shape[1])
                index.add(self.embeddings[:docs])
                self.faiss_indices[docs] = index

        if "chroma" in self.dense_backends:
            if not CHROMA_AVAILABLE:
                raise ImportError("chromadb is required but not installed")
            self._prepare_chroma_collections()

        for docs in self.doc_counts:
            tokenized = [chunk.lower().split() for chunk in self.chunks[:docs]]
            self.bm25_indices[docs] = BM25Okapi(tokenized)

    def _prepare_chroma_collections(self):
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)

        existing = {c.name for c in self.chroma_client.list_collections()}
        for docs in self.doc_counts:
            name = f"chunks_{docs}"
            if name in existing:
                self.chroma_client.delete_collection(name)

            collection = self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

            batch_size = 512
            for start in range(0, docs, batch_size):
                end = min(start + batch_size, docs)
                ids = [str(i) for i in range(start, end)]
                documents = self.chunks[start:end]
                embeddings = self.embeddings[start:end].tolist()
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=[{"idx": i} for i in range(start, end)],
                )

            self.chroma_collections[docs] = collection

    def _load_models(self):
        print(f"Using device: {self.device.upper()}")
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(self.embed_model_name)
        if self.device == "cuda":
            self.embed_model = self.embed_model.to(self.device)

        print("Loading LLM...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                    torch_dtype=self.torch.float16 if self.device == "cuda" else self.torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to load LLM. Try: "
                "1) numpy<2, 2) --no-cuda (or enable CUDA), "
                "3) reduce model size via --llm-model, "
                "4) reduce --max-new-tokens. "
                f"Original error: {error}"
            ) from error

        if self.warmup_runs > 0:
            self._warmup()

    def _warmup(self):
        print(f"Running warmup ({self.warmup_runs} run(s))...")
        for _ in range(self.warmup_runs):
            _ = self._generate("Question: warmup\nAnswer:")

    def _encode_query(self, query: str) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        embedding = self.embed_model.encode([query], normalize_embeddings=True)
        elapsed = (time.perf_counter() - start) * 1000
        return embedding.astype("float32"), elapsed

    def _retrieve_faiss(self, query_embedding: np.ndarray, docs: int) -> Tuple[List[str], float]:
        start = time.perf_counter()
        _, indices = self.faiss_indices[docs].search(query_embedding, min(self.top_k, docs))
        elapsed = (time.perf_counter() - start) * 1000
        contexts = [self.chunks[i] for i in indices[0]]
        return contexts, elapsed

    def _retrieve_chroma(self, query_embedding: np.ndarray, docs: int) -> Tuple[List[str], float]:
        start = time.perf_counter()
        result = self.chroma_collections[docs].query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=min(self.top_k, docs),
            include=["documents"],
        )
        elapsed = (time.perf_counter() - start) * 1000
        contexts = result.get("documents", [[]])[0]
        return contexts, elapsed

    def _retrieve_bm25(self, query: str, docs: int) -> Tuple[List[str], float]:
        if docs not in self.bm25_indices or self.bm25_indices[docs] is None:
            return [], 0.0

        start = time.perf_counter()
        tokens = query.lower().split()
        scores = self.bm25_indices[docs].get_scores(tokens)
        top_indices = np.argsort(scores)[-min(self.top_k, docs):][::-1]
        elapsed = (time.perf_counter() - start) * 1000
        contexts = [self.chunks[i] for i in top_indices]
        return contexts, elapsed

    def _build_prompt(self, query: str, contexts: Optional[List[str]]) -> Tuple[str, float]:
        start = time.perf_counter()
        if contexts:
            context_text = "\n\n".join(contexts)
            prompt = (
                "You are an assistant. Answer using only the context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
        else:
            prompt = f"Question: {query}\nAnswer:"
        elapsed = (time.perf_counter() - start) * 1000
        return prompt, elapsed

    def _generate(self, prompt: str) -> float:
        start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed

    def _run_single(self, question: str, method: str, docs: Optional[int]) -> LatencyProfile:
        profile = LatencyProfile()
        total_start = time.perf_counter()

        contexts = None

        if method == "no_rag":
            prompt, prompt_ms = self._build_prompt(question, None)
            profile.prompt_construction_ms = prompt_ms
            profile.generation_ms = self._generate(prompt)

        elif method == "dense_faiss":
            query_embedding, enc_ms = self._encode_query(question)
            contexts, ann_ms = self._retrieve_faiss(query_embedding, docs)
            profile.query_encoding_ms = enc_ms
            profile.ann_search_ms = ann_ms
            prompt, prompt_ms = self._build_prompt(question, contexts)
            profile.prompt_construction_ms = prompt_ms
            profile.generation_ms = self._generate(prompt)

        elif method == "dense_chroma":
            query_embedding, enc_ms = self._encode_query(question)
            contexts, ann_ms = self._retrieve_chroma(query_embedding, docs)
            profile.query_encoding_ms = enc_ms
            profile.ann_search_ms = ann_ms
            prompt, prompt_ms = self._build_prompt(question, contexts)
            profile.prompt_construction_ms = prompt_ms
            profile.generation_ms = self._generate(prompt)

        elif method == "sparse_bm25":
            contexts, ann_ms = self._retrieve_bm25(question, docs)
            profile.ann_search_ms = ann_ms
            prompt, prompt_ms = self._build_prompt(question, contexts)
            profile.prompt_construction_ms = prompt_ms
            profile.generation_ms = self._generate(prompt)

        else:
            raise ValueError(f"Unknown method: {method}")

        profile.total_ms = (time.perf_counter() - total_start) * 1000
        return profile

    @staticmethod
    def _stats(profiles: List[LatencyProfile]) -> Dict:
        if not profiles:
            return {}

        def values(field: str) -> List[float]:
            return [getattr(profile, field) for profile in profiles]

        components = [
            "query_encoding_ms",
            "ann_search_ms",
            "prompt_construction_ms",
            "generation_ms",
            "total_ms",
        ]

        summary = {
            "mean": {component: float(np.mean(values(component))) for component in components},
            "std": {component: float(np.std(values(component))) for component in components},
            "min": {component: float(np.min(values(component))) for component in components},
            "max": {component: float(np.max(values(component))) for component in components},
            "percentiles": {
                "p50": {component: float(np.percentile(values(component), 50)) for component in components},
                "p95": {component: float(np.percentile(values(component), 95)) for component in components},
                "p99": {component: float(np.percentile(values(component), 99)) for component in components},
            },
            "n_samples": len(profiles),
        }

        total = summary["mean"]["total_ms"]
        percentages = {
            "query_encoding_pct": (summary["mean"]["query_encoding_ms"] / total * 100) if total else 0.0,
            "ann_search_pct": (summary["mean"]["ann_search_ms"] / total * 100) if total else 0.0,
            "prompt_construction_pct": (summary["mean"]["prompt_construction_ms"] / total * 100) if total else 0.0,
            "generation_pct": (summary["mean"]["generation_ms"] / total * 100) if total else 0.0,
        }

        dominant_component = max(percentages, key=percentages.get)
        summary["percentages"] = percentages
        summary["insight"] = {
            "bottleneck": dominant_component.replace("_pct", "_ms"),
            "bottleneck_pct": percentages[dominant_component],
        }
        return summary

    def run(self, dataset_path: str, output_dir: str, max_questions: Optional[int]):
        with open(dataset_path, "r") as handle:
            dataset = json.load(handle)

        questions = [item["question"] for item in dataset]
        if max_questions is not None:
            questions = questions[:max_questions]

        methods_to_run = []
        if "faiss" in self.dense_backends:
            methods_to_run.append("dense_faiss")
        if "chroma" in self.dense_backends:
            methods_to_run.append("dense_chroma")
        methods_to_run.append("sparse_bm25")
        methods_to_run.append("no_rag")

        raw_profiles: Dict[str, List[Dict]] = {}
        summary_methods: Dict[str, Dict] = {}

        for method in methods_to_run:
            if method == "no_rag":
                key = "no_rag"
                profiles = []
                for question in tqdm(questions, desc=f"{method}"):
                    profile = self._run_single(question, method, None)
                    profiles.append(profile)
                raw_profiles[key] = [profile.to_dict() for profile in profiles]
                summary_methods[key] = self._stats(profiles)
                continue

            for docs in self.doc_counts:
                key = f"{method}_docs_{docs}"
                profiles = []
                for question in tqdm(questions, desc=key):
                    profile = self._run_single(question, method, docs)
                    profiles.append(profile)
                raw_profiles[key] = [profile.to_dict() for profile in profiles]
                summary_methods[key] = self._stats(profiles)

        comparisons = self._build_comparisons(summary_methods)

        output = {
            "device": self.device,
            "config": {
                "dataset": dataset_path,
                "n_questions": len(questions),
                "doc_counts": self.doc_counts,
                "top_k": self.top_k,
                "dense_backends": self.dense_backends,
                "max_new_tokens": self.max_new_tokens,
            },
            "methods": summary_methods,
            "comparisons": comparisons,
        }

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "latency_results.json", "w") as handle:
            json.dump(output, handle, indent=2)
        with open(Path(output_dir) / "latency_raw_profiles.json", "w") as handle:
            json.dump(raw_profiles, handle, indent=2)

        print("\nLatency run complete")
        print(f"Saved: {Path(output_dir) / 'latency_results.json'}")
        print(f"Saved: {Path(output_dir) / 'latency_raw_profiles.json'}")

    def _build_comparisons(self, methods: Dict[str, Dict]) -> Dict:
        comparisons = {
            "rag_vs_no_rag_overhead_pct": {},
            "doc_count_scaling_pct": {},
            "dense_vs_sparse_pct": {},
            "bottlenecks": {},
        }

        no_rag = methods.get("no_rag", {}).get("mean", {}).get("total_ms", None)

        for key, stats in methods.items():
            mean_total = stats.get("mean", {}).get("total_ms", 0.0)
            if no_rag and key != "no_rag":
                comparisons["rag_vs_no_rag_overhead_pct"][key] = ((mean_total / no_rag) - 1.0) * 100.0

            insight = stats.get("insight", {})
            comparisons["bottlenecks"][key] = insight

        families = ["dense_faiss", "dense_chroma", "sparse_bm25"]
        for family in families:
            series = []
            for key, stats in methods.items():
                if key.startswith(family + "_docs_"):
                    docs = int(key.split("_docs_")[-1])
                    total = stats.get("mean", {}).get("total_ms", 0.0)
                    series.append((docs, total))

            series.sort(key=lambda item: item[0])
            if len(series) >= 2 and series[0][1] > 0:
                start_docs, start_total = series[0]
                end_docs, end_total = series[-1]
                comparisons["doc_count_scaling_pct"][family] = {
                    "from_docs": start_docs,
                    "to_docs": end_docs,
                    "latency_change_pct": ((end_total / start_total) - 1.0) * 100.0,
                }

        for docs in self.doc_counts:
            dense_key = f"dense_faiss_docs_{docs}"
            sparse_key = f"sparse_bm25_docs_{docs}"
            if dense_key in methods and sparse_key in methods:
                dense_total = methods[dense_key]["mean"]["total_ms"]
                sparse_total = methods[sparse_key]["mean"]["total_ms"]
                if sparse_total > 0:
                    comparisons["dense_vs_sparse_pct"][str(docs)] = ((dense_total / sparse_total) - 1.0) * 100.0

        return comparisons


def main():
    parser = argparse.ArgumentParser(description="Latency-first RAG evaluation")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc-counts", type=str, default="50,200")
    parser.add_argument("--dense-backends", type=str, default="faiss,chroma")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--chroma-dir", type=str, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true")

    args = parser.parse_args()

    doc_counts = parse_int_csv(args.doc_counts)
    dense_backends = [item.strip().lower() for item in args.dense_backends.split(",") if item.strip()]

    allowed = {"faiss", "chroma"}
    invalid = [item for item in dense_backends if item not in allowed]
    if invalid:
        raise ValueError(f"Invalid dense backend(s): {invalid}. Allowed: faiss,chroma")

    evaluator = LatencyEvaluator(
        use_cuda=not args.no_cuda,
        max_new_tokens=args.max_new_tokens,
        dense_backends=dense_backends,
        doc_counts=doc_counts,
        top_k=args.top_k,
        chroma_dir=args.chroma_dir,
        llm_model_name=args.llm_model,
        embed_model_name=args.embed_model,
        warmup_runs=args.warmup_runs,
    )
    evaluator.run(args.dataset, args.output, args.max_questions)


if __name__ == "__main__":
    main()
