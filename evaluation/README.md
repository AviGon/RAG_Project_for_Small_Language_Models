# Latency Evaluation System

This evaluation package is now latency-only.

It profiles end-to-end latency and stepwise latency for:
- Dense retrieval (FAISS, ChromaDB)
- Sparse retrieval (BM25)
- No-RAG baseline
- Low vs high document counts
- CPU vs GPU
- Single or multiple LLMs

## Structure

evaluation/
- dataset/
  - evaluation_dataset.json
- experiments/
  - latency_eval.py
  - compare_latency.py
- visualizations/
  - plot_latency.py
- results/
  - latency/

## Main Dimensions Covered

1. Retrieval strategy
- dense_faiss
- dense_chroma
- sparse_bm25
- no_rag

2. Corpus size sweep
- configurable via --doc-counts (example: 50,200)

3. Device comparison
- CPU and GPU (if CUDA available)

4. Model comparison
- configurable single model via --llm-model
- multi-model via --llm-models

## Stepwise Latency Metrics

For each method, the pipeline records:
- query_encoding_ms
- ann_search_ms
- prompt_construction_ms
- generation_ms
- total_ms

For each metric, outputs include:
- mean
- std
- min
- max
- percentiles: p50, p95, p99

Derived insights include:
- bottleneck component
- bottleneck percentage
- RAG vs No-RAG overhead percentage
- doc-count scaling percentage
- dense-vs-sparse percentage

## Run

From project root:

python3 run_full_pipeline.py

Skip setup and only run evaluation:

python3 run_full_pipeline.py --skip-setup

Single model example:

python3 evaluation/experiments/compare_latency.py --llm-model microsoft/Phi-3-mini-4k-instruct --visualize

Multi-model example:

python3 evaluation/experiments/compare_latency.py --llm-models Qwen/Qwen2.5-1.5B-Instruct,microsoft/Phi-3-mini-4k-instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0 --visualize

## Key Output Files

Per model:
- evaluation/results/latency/<model_slug>/cpu/latency_results.json
- evaluation/results/latency/<model_slug>/gpu/latency_results.json
- evaluation/results/latency/<model_slug>/cpu_gpu_comparison.json
- evaluation/results/latency/<model_slug>/cpu/latency_raw_profiles.json
- evaluation/results/latency/<model_slug>/gpu/latency_raw_profiles.json

Cross-model:
- evaluation/results/latency/multi_model_summary.json

## Charts

Per model charts are saved in:
- evaluation/visualizations/latency/<model_slug>/

Generated charts:
- cpu_stepwise_latency.png
- gpu_stepwise_latency.png
- cpu_doc_scaling.png
- gpu_doc_scaling.png
- cpu_rag_vs_no_rag_overhead.png
- gpu_rag_vs_no_rag_overhead.png
- cpu_total_latency_percentiles.png
- gpu_total_latency_percentiles.png
- cpu_generation_latency_percentiles.png
- gpu_generation_latency_percentiles.png
- cpu_gpu_total_speedup.png
- cpu_gpu_component_speedup_heatmap.png
- cpu_gpu_p95_total_speedup.png

## Notes

- ChromaDB and BM25 are optional dependencies used when selected.
- If CUDA is unavailable, GPU runs are skipped automatically.
- Warmup runs are supported via --warmup-runs to reduce cold-start bias.
