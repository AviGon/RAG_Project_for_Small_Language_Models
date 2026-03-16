"""
Latency visualization utilities.
Creates key charts for:
- Stepwise latency per method
- Dense vs sparse and doc-count scaling
- RAG vs No-RAG overhead
- CPU vs GPU speedups
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = [
    "query_encoding_ms",
    "ann_search_ms",
    "rerank_ms",
    "tool_routing_ms",
    "prompt_construction_ms",
    "generation_ms",
]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def method_order(method_name: str):
    if method_name == "no_rag":
        return (9, 0)
    if method_name.startswith("dense_faiss"):
        docs = int(method_name.split("_docs_")[-1])
        return (1, docs)
    if method_name.startswith("dense_faiss_rerank"):
        docs = int(method_name.split("_docs_")[-1])
        return (2, docs)
    if method_name.startswith("tool_rag_faiss"):
        docs = int(method_name.split("_docs_")[-1])
        return (3, docs)
    if method_name.startswith("dense_chroma"):
        docs = int(method_name.split("_docs_")[-1])
        return (4, docs)
    if method_name.startswith("dense_chroma_rerank"):
        docs = int(method_name.split("_docs_")[-1])
        return (5, docs)
    if method_name.startswith("sparse_bm25"):
        docs = int(method_name.split("_docs_")[-1])
        return (6, docs)
    return (99, 0)


def pretty_name(method_name: str) -> str:
    if method_name == "no_rag":
        return "No-RAG"

    family, docs = method_name.split("_docs_")
    docs = docs.strip()
    family_label = {
        "dense_faiss": "Dense-FAISS",
        "dense_faiss_rerank": "Dense-FAISS+Rerank",
        "tool_rag_faiss": "Tool-Routed FAISS",
        "dense_chroma": "Dense-Chroma",
        "dense_chroma_rerank": "Dense-Chroma+Rerank",
        "sparse_bm25": "Sparse-BM25",
    }.get(family, family)
    return f"{family_label}\n(docs={docs})"


def plot_stepwise_latency(single_results: dict, output_path: Path):
    methods = sorted(single_results["methods"].keys(), key=method_order)

    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))

    plt.figure(figsize=(16, 7))
    for component in COMPONENTS:
        values = [single_results["methods"][method]["mean"].get(component, 0.0) for method in methods]
        plt.bar(x, values, bottom=bottoms, label=component.replace("_ms", ""))
        bottoms = bottoms + np.array(values)

    for idx, total in enumerate(bottoms):
        plt.text(idx, total, f"{total:.0f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(x, [pretty_name(method) for method in methods], rotation=0)
    plt.ylabel("Latency (ms)")
    plt.title(f"Stepwise Latency Breakdown ({single_results['device'].upper()})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_doc_scaling(single_results: dict, output_path: Path):
    methods = single_results["methods"]

    families = [
        "dense_faiss",
        "dense_faiss_rerank",
        "tool_rag_faiss",
        "dense_chroma",
        "dense_chroma_rerank",
        "sparse_bm25",
    ]
    plt.figure(figsize=(10, 6))

    for family in families:
        points = []
        for method, stats in methods.items():
            if method.startswith(family + "_docs_"):
                docs = int(method.split("_docs_")[-1])
                total = stats["mean"]["total_ms"]
                points.append((docs, total))
        points.sort(key=lambda item: item[0])
        if points:
            xs = [item[0] for item in points]
            ys = [item[1] for item in points]
            plt.plot(xs, ys, marker="o", label=family)

    if "no_rag" in methods:
        baseline = methods["no_rag"]["mean"]["total_ms"]
        plt.axhline(y=baseline, linestyle="--", label="no_rag")

    plt.xlabel("Number of retrieved documents in index")
    plt.ylabel("Total latency (ms)")
    plt.title(f"Latency vs Document Count ({single_results['device'].upper()})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rag_overhead(single_results: dict, output_path: Path):
    overhead = single_results.get("comparisons", {}).get("rag_vs_no_rag_overhead_pct", {})
    if not overhead:
        return

    methods = sorted(overhead.keys(), key=method_order)
    values = [overhead[method] for method in methods]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(np.arange(len(methods)), values)
    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(methods)), [pretty_name(method) for method in methods])
    plt.ylabel("Overhead vs No-RAG (%)")
    plt.title(f"RAG Overhead vs No-RAG ({single_results['device'].upper()})")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:+.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cpu_gpu_speedup(cpu_results: dict, gpu_results: dict, output_path: Path):
    cpu_methods = cpu_results["methods"]
    gpu_methods = gpu_results["methods"]
    common_methods = sorted(set(cpu_methods.keys()).intersection(set(gpu_methods.keys())), key=method_order)

    speedups = []
    for method in common_methods:
        cpu_total = cpu_methods[method]["mean"]["total_ms"]
        gpu_total = gpu_methods[method]["mean"]["total_ms"]
        speedups.append((cpu_total / gpu_total) if gpu_total > 0 else 0.0)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(np.arange(len(common_methods)), speedups)
    plt.axhline(y=1.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(common_methods)), [pretty_name(method) for method in common_methods])
    plt.ylabel("Speedup (CPU / GPU)")
    plt.title("CPU vs GPU Total Latency Speedup")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}x", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_component_speedup_heatmap(cpu_results: dict, gpu_results: dict, output_path: Path):
    cpu_methods = cpu_results["methods"]
    gpu_methods = gpu_results["methods"]
    common_methods = sorted(set(cpu_methods.keys()).intersection(set(gpu_methods.keys())), key=method_order)

    matrix = []
    for method in common_methods:
        row = []
        for component in COMPONENTS:
            cpu_val = cpu_methods[method]["mean"].get(component, 0.0)
            gpu_val = gpu_methods[method]["mean"].get(component, 0.0)
            row.append((cpu_val / gpu_val) if gpu_val > 0 else 0.0)
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(11, 8))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label="Speedup (CPU/GPU)")

    plt.xticks(np.arange(len(COMPONENTS)), [item.replace("_ms", "") for item in COMPONENTS], rotation=30, ha="right")
    plt.yticks(np.arange(len(common_methods)), [pretty_name(method) for method in common_methods])
    plt.title("CPU/GPU Speedup Heatmap by Component")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_percentiles(single_results: dict, output_path: Path, metric: str = "total_ms"):
    methods = sorted(single_results["methods"].keys(), key=method_order)

    p50_values = []
    p95_values = []
    p99_values = []
    for method in methods:
        percentiles = single_results["methods"][method].get("percentiles", {})
        p50_values.append(percentiles.get("p50", {}).get(metric, 0.0))
        p95_values.append(percentiles.get("p95", {}).get(metric, 0.0))
        p99_values.append(percentiles.get("p99", {}).get(metric, 0.0))

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(16, 7))
    bars1 = plt.bar(x - width, p50_values, width, label="p50")
    bars2 = plt.bar(x, p95_values, width, label="p95")
    bars3 = plt.bar(x + width, p99_values, width, label="p99")

    plt.xticks(x, [pretty_name(method) for method in methods])
    plt.ylabel("Latency (ms)")
    title_metric = metric.replace("_ms", "").replace("_", " ").title()
    plt.title(f"{title_metric} Percentiles ({single_results['device'].upper()})")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.0f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cpu_gpu_p95_speedup(cpu_results: dict, gpu_results: dict, output_path: Path):
    cpu_methods = cpu_results["methods"]
    gpu_methods = gpu_results["methods"]
    common_methods = sorted(set(cpu_methods.keys()).intersection(set(gpu_methods.keys())), key=method_order)

    speedups = []
    for method in common_methods:
        cpu_p95 = cpu_methods[method].get("percentiles", {}).get("p95", {}).get("total_ms", 0.0)
        gpu_p95 = gpu_methods[method].get("percentiles", {}).get("p95", {}).get("total_ms", 0.0)
        speedups.append((cpu_p95 / gpu_p95) if gpu_p95 > 0 else 0.0)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(np.arange(len(common_methods)), speedups)
    plt.axhline(y=1.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(common_methods)), [pretty_name(method) for method in common_methods])
    plt.ylabel("Speedup (CPU p95 / GPU p95)")
    plt.title("CPU vs GPU p95 Total Latency Speedup")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}x", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot latency-focused charts")
    parser.add_argument("--cpu-results", type=str, default="evaluation/results/latency/cpu/latency_results.json")
    parser.add_argument("--gpu-results", type=str, default="evaluation/results/latency/gpu/latency_results.json")
    parser.add_argument("--output-dir", type=str, default="evaluation/visualizations/latency")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu = load_json(Path(args.cpu_results))
    gpu = load_json(Path(args.gpu_results))

    if cpu:
        plot_stepwise_latency(cpu, output_dir / "cpu_stepwise_latency.png")
        plot_doc_scaling(cpu, output_dir / "cpu_doc_scaling.png")
        plot_rag_overhead(cpu, output_dir / "cpu_rag_vs_no_rag_overhead.png")
        plot_percentiles(cpu, output_dir / "cpu_total_latency_percentiles.png", metric="total_ms")
        plot_percentiles(cpu, output_dir / "cpu_generation_latency_percentiles.png", metric="generation_ms")

    if gpu:
        plot_stepwise_latency(gpu, output_dir / "gpu_stepwise_latency.png")
        plot_doc_scaling(gpu, output_dir / "gpu_doc_scaling.png")
        plot_rag_overhead(gpu, output_dir / "gpu_rag_vs_no_rag_overhead.png")
        plot_percentiles(gpu, output_dir / "gpu_total_latency_percentiles.png", metric="total_ms")
        plot_percentiles(gpu, output_dir / "gpu_generation_latency_percentiles.png", metric="generation_ms")

    if cpu and gpu:
        plot_cpu_gpu_speedup(cpu, gpu, output_dir / "cpu_gpu_total_speedup.png")
        plot_component_speedup_heatmap(cpu, gpu, output_dir / "cpu_gpu_component_speedup_heatmap.png")
        plot_cpu_gpu_p95_speedup(cpu, gpu, output_dir / "cpu_gpu_p95_total_speedup.png")

    print(f"Saved charts to {output_dir}")


if __name__ == "__main__":
    main()
