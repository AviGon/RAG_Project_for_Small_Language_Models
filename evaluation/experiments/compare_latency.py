"""
CPU/GPU latency comparison for latency-first RAG evaluation.
Supports single or multiple LLMs in one run.
"""

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_LLM_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct,"
    "microsoft/Phi-3-mini-4k-instruct,"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0,"
    "mistralai/Mistral-7B-Instruct-v0.2"
)


def run_command(cmd: str) -> bool:
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def gpu_available() -> bool:
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return True
        mps_backend = getattr(torch.backends, "mps", None)
        return bool(mps_backend and mps_backend.is_available())
    except Exception:
        return False


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip())
    return slug.strip("_") or "model"


def parse_models(llm_model: str, llm_models: Optional[str]) -> List[str]:
    if llm_models and llm_models.strip():
        models = [item.strip() for item in llm_models.split(",") if item.strip()]
        if not models:
            raise ValueError("--llm-models was provided but no valid model names were parsed")
        return models
    return [llm_model]


def run_device(
    device: str,
    dataset: str,
    output_dir: str,
    doc_counts: str,
    dense_backends: str,
    top_k: int,
    max_new_tokens: int,
    max_questions,
    chroma_dir: str,
    llm_model: str,
    embed_model: str,
    warmup_runs: int,
) -> bool:
    cuda_flag = "" if device == "gpu" else "--no-cuda"
    max_q_flag = f" --max-questions {max_questions}" if max_questions is not None else ""

    cmd = (
        "python3 evaluation/experiments/latency_eval.py "
        f"{cuda_flag} "
        f"--dataset {dataset} "
        f"--output {output_dir}/{device} "
        f"--doc-counts {doc_counts} "
        f"--dense-backends {dense_backends} "
        f"--top-k {top_k} "
        f"--max-new-tokens {max_new_tokens} "
        f"--chroma-dir {chroma_dir}/{device}"
        f" --llm-model {llm_model}"
        f" --embed-model {embed_model}"
        f" --warmup-runs {warmup_runs}"
        f"{max_q_flag}"
    )
    return run_command(cmd)


def load_results(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def build_cpu_gpu_comparison(cpu: dict, gpu: dict) -> dict:
    cpu_methods = cpu.get("methods", {})
    gpu_methods = gpu.get("methods", {})

    common_methods = sorted(set(cpu_methods.keys()).intersection(set(gpu_methods.keys())))

    per_method = {}
    for method in common_methods:
        cpu_mean = cpu_methods[method].get("mean", {})
        gpu_mean = gpu_methods[method].get("mean", {})
        components = [
            "query_encoding_ms",
            "ann_search_ms",
            "rerank_ms",
            "tool_routing_ms",
            "prompt_construction_ms",
            "generation_ms",
            "total_ms",
        ]

        component_speedups = {}
        for component in components:
            cpu_val = cpu_mean.get(component, 0.0)
            gpu_val = gpu_mean.get(component, 0.0)
            component_speedups[component.replace("_ms", "_speedup")] = (
                (cpu_val / gpu_val) if gpu_val > 0 else None
            )

        per_method[method] = {
            "cpu_total_ms": cpu_mean.get("total_ms", 0.0),
            "gpu_total_ms": gpu_mean.get("total_ms", 0.0),
            "total_speedup": component_speedups.get("total_speedup"),
            "component_speedups": component_speedups,
            "cpu_bottleneck": cpu_methods[method].get("insight", {}).get("bottleneck"),
            "gpu_bottleneck": gpu_methods[method].get("insight", {}).get("bottleneck"),
        }

    fastest = sorted(
        [(method, values.get("total_speedup", 0) or 0) for method, values in per_method.items()],
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "common_methods": common_methods,
        "per_method": per_method,
        "overall": {
            "top_speedups": fastest[:5],
        },
    }


def select_primary_method(methods: Dict[str, Dict]) -> Optional[str]:
    if not methods:
        return None

    candidates = [name for name in methods.keys() if name.startswith("dense_faiss_docs_")]
    if candidates:
        return sorted(candidates, key=lambda item: int(item.split("_docs_")[-1]))[-1]

    if "no_rag" in methods:
        return "no_rag"

    return sorted(methods.keys())[0]


def build_multi_model_summary(all_model_results: Dict[str, Dict]) -> Dict:
    rows = []
    for model_name, bundle in all_model_results.items():
        cpu = bundle.get("cpu")
        gpu = bundle.get("gpu")

        method_name = None
        cpu_ms = None
        gpu_ms = None
        speedup = None

        if cpu and gpu:
            method_name = select_primary_method(cpu.get("methods", {}))
            if method_name and method_name in gpu.get("methods", {}):
                cpu_ms = cpu["methods"][method_name]["mean"]["total_ms"]
                gpu_ms = gpu["methods"][method_name]["mean"]["total_ms"]
                speedup = (cpu_ms / gpu_ms) if gpu_ms and gpu_ms > 0 else None

        rows.append(
            {
                "model": model_name,
                "primary_method": method_name,
                "cpu_total_ms": cpu_ms,
                "gpu_total_ms": gpu_ms,
                "cpu_gpu_speedup": speedup,
            }
        )

    ranked = sorted(
        rows,
        key=lambda item: (item["cpu_gpu_speedup"] if item["cpu_gpu_speedup"] is not None else -1),
        reverse=True,
    )

    return {
        "model_count": len(rows),
        "rows": rows,
        "ranked_by_speedup": ranked,
    }


def run_single_model(args, llm_model: str, gpu_allowed: bool) -> Dict:
    slug = model_slug(llm_model)
    model_output = Path(args.output) / slug
    model_chroma = Path(args.chroma_dir) / slug

    print("\n" + "=" * 70)
    print(f"Model: {llm_model}")
    print(f"Output root: {model_output}")
    print("=" * 70)

    if not args.use_existing:
        if not args.skip_cpu:
            ok = run_device(
                "cpu",
                args.dataset,
                str(model_output),
                args.doc_counts,
                args.dense_backends,
                args.top_k,
                args.max_new_tokens,
                args.max_questions,
                str(model_chroma),
                llm_model,
                args.embed_model,
                args.warmup_runs,
            )
            if not ok:
                print("CPU run failed")
                sys.exit(1)

        if not args.skip_gpu and gpu_allowed:
            ok = run_device(
                "gpu",
                args.dataset,
                str(model_output),
                args.doc_counts,
                args.dense_backends,
                args.top_k,
                args.max_new_tokens,
                args.max_questions,
                str(model_chroma),
                llm_model,
                args.embed_model,
                args.warmup_runs,
            )
            if not ok and args.skip_cpu:
                print("GPU run failed")
                sys.exit(1)

    cpu_path = model_output / "cpu" / "latency_results.json"
    gpu_path = model_output / "gpu" / "latency_results.json"

    cpu = None if args.skip_cpu else load_results(cpu_path)
    gpu = None if (args.skip_gpu or not gpu_allowed) else load_results(gpu_path)

    if cpu is None and gpu is None:
        print(f"No results found for model {llm_model}")
        sys.exit(1)

    comparison = None
    if cpu and gpu:
        comparison = build_cpu_gpu_comparison(cpu, gpu)
        out = model_output / "cpu_gpu_comparison.json"
        with open(out, "w") as handle:
            json.dump(comparison, handle, indent=2)
        print(f"Saved comparison: {out}")

    if args.visualize:
        vis_out = Path("evaluation/visualizations/latency") / slug
        cmd = (
            "python3 evaluation/visualizations/plot_latency.py "
            f"--output-dir {vis_out} "
            f"--cpu-results {cpu_path} "
            f"--gpu-results {gpu_path}"
        )
        if run_command(cmd):
            print(f"Saved latency charts: {vis_out}")

    if comparison:
        top = comparison.get("overall", {}).get("top_speedups", [])
        print(f"\nTop speedups (CPU/GPU) for {llm_model}:")
        for method, speedup in top:
            print(f"  - {method}: {speedup:.2f}x")

    return {
        "slug": slug,
        "cpu": cpu,
        "gpu": gpu,
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CPU vs GPU latency")
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
        default=DEFAULT_LLM_MODELS,
        help=(
            "Comma-separated LLM models. Overrides --llm-model when provided. "
            "Defaults to the 4-model suite including Mistral."
        ),
    )
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--warmup-runs", type=int, default=1)

    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    models = parse_models(args.llm_model, args.llm_models)

    gpu_allowed = True
    if not args.skip_gpu and not gpu_available():
        print("No GPU backend available (CUDA/MPS). Skipping GPU run.")
        gpu_allowed = False

    all_results = {}
    for model in models:
        all_results[model] = run_single_model(args, model, gpu_allowed)

    if len(models) > 1:
        aggregate = build_multi_model_summary(all_results)
        output_path = Path(args.output) / "multi_model_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as handle:
            json.dump(aggregate, handle, indent=2)
        print(f"\nSaved multi-model summary: {output_path}")

        print("\nModel speedup ranking (primary method):")
        for row in aggregate["ranked_by_speedup"]:
            speed = row["cpu_gpu_speedup"]
            speed_text = f"{speed:.2f}x" if speed is not None else "N/A"
            print(f"  - {row['model']}: {speed_text}")


if __name__ == "__main__":
    main()
