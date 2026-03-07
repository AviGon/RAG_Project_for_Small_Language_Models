#!/usr/bin/env python3
"""
Latency-focused full pipeline for local RAG benchmarking.

Pipeline:
1) Extract and chunk data
2) Generate embeddings
3) Build FAISS and Chroma indices
4) Run latency matrix (dense/sparse, doc-count sweep, RAG vs No-RAG)
5) Compare CPU vs GPU and generate charts
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str, critical: bool = True) -> bool:
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print(f"$ {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        if critical:
            print(f"❌ Failed: {description}")
            sys.exit(1)
        print(f"⚠️  Non-critical failure: {description}")
        return False

    print(f"✅ Done: {description}")
    return True


def check_prerequisites() -> bool:
    required = [
        "data/handbook.pdf",
        "extract_and_chunk.py",
        "generate_embeddings.py",
        "store_faiss_index.py",
        "store_chroma_index.py",
        "evaluation/experiments/latency_eval.py",
        "evaluation/experiments/compare_latency.py",
        "evaluation/visualizations/plot_latency.py",
    ]

    missing = [file for file in required if not Path(file).exists()]
    if missing:
        print("Missing required files:")
        for file in missing:
            print(f"  - {file}")
        return False
    return True


def setup_phase(force: bool):
    outputs = ["chunks.pkl", "embeddings.npy", "faiss_index.bin", "metadata.pkl", "chroma_db"]
    existing = [file for file in outputs if Path(file).exists()]

    if existing and not force:
        print("Existing setup artifacts found:")
        for item in existing:
            print(f"  - {item}")
        response = input("Rebuild setup artifacts? (y/N): ").strip().lower()
        if response != "y":
            print("Skipping setup; reusing existing artifacts.")
            return

    run_command("python3 extract_and_chunk.py", "Step 1: Extract + chunk PDF")
    run_command("python3 generate_embeddings.py", "Step 2: Generate embeddings")
    run_command("python3 store_faiss_index.py", "Step 3: Build FAISS index")
    run_command("python3 store_chroma_index.py", "Step 4: Build Chroma index")


def evaluation_phase(args):
    max_q_flag = f" --max-questions {args.max_questions}" if args.max_questions is not None else ""
    cpu_skip = " --skip-cpu" if args.skip_cpu else ""
    gpu_skip = " --skip-gpu" if args.skip_gpu else ""

    cmd = (
        "python3 evaluation/experiments/compare_latency.py "
        f"--dataset {args.dataset} "
        f"--output {args.output} "
        f"--doc-counts {args.doc_counts} "
        f"--dense-backends {args.dense_backends} "
        f"--top-k {args.top_k} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--chroma-dir {args.chroma_dir} "
        f"--llm-model {args.llm_model} "
        f"{'--llm-models ' + args.llm_models + ' ' if args.llm_models else ''}"
        f"--embed-model {args.embed_model} "
        f"--warmup-runs {args.warmup_runs} "
        f"{'--use-existing ' if args.use_existing else ''}"
        f"{'--visualize ' if not args.skip_visualization else ''}"
        f"{cpu_skip}{gpu_skip}{max_q_flag}"
    )
    run_command(cmd, "Step 5: Run latency matrix + CPU/GPU comparison")


def print_summary(output_dir: str):
    root = Path(output_dir)
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nLatency JSON outputs (per model):")
    print(f"  - {root / '<model_slug>' / 'cpu' / 'latency_results.json'}")
    print(f"  - {root / '<model_slug>' / 'gpu' / 'latency_results.json'}")
    print(f"  - {root / '<model_slug>' / 'cpu_gpu_comparison.json'}")
    print(f"  - {root / 'multi_model_summary.json'} (when --llm-models is used)")
    print("\nCharts:")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_stepwise_latency.png")
    print("  - evaluation/visualizations/latency/<model_slug>/gpu_stepwise_latency.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_doc_scaling.png")
    print("  - evaluation/visualizations/latency/<model_slug>/gpu_doc_scaling.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_rag_vs_no_rag_overhead.png")
    print("  - evaluation/visualizations/latency/<model_slug>/gpu_rag_vs_no_rag_overhead.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_total_latency_percentiles.png")
    print("  - evaluation/visualizations/latency/<model_slug>/gpu_total_latency_percentiles.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_generation_latency_percentiles.png")
    print("  - evaluation/visualizations/latency/<model_slug>/gpu_generation_latency_percentiles.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_gpu_total_speedup.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_gpu_component_speedup_heatmap.png")
    print("  - evaluation/visualizations/latency/<model_slug>/cpu_gpu_p95_total_speedup.png")


def main():
    parser = argparse.ArgumentParser(description="Latency-focused RAG full pipeline")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--dataset", type=str, default="evaluation/dataset/evaluation_dataset.json")
    parser.add_argument("--output", type=str, default="evaluation/results/latency")
    parser.add_argument("--doc-counts", type=str, default="50,200")
    parser.add_argument("--dense-backends", type=str, default="faiss,chroma")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--chroma-dir", type=str, default="chroma_db")
    parser.add_argument("--llm-model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        help="Comma-separated LLM models. Overrides --llm-model when provided.",
    )
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--warmup-runs", type=int, default=1)

    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")

    args = parser.parse_args()

    if not check_prerequisites():
        sys.exit(1)

    if not args.skip_setup:
        setup_phase(args.force)

    if args.setup_only:
        print("Setup-only mode complete.")
        return

    evaluation_phase(args)
    print_summary(args.output)


if __name__ == "__main__":
    main()
