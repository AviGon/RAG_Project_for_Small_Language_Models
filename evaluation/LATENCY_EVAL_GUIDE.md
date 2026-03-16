# Latency Evaluation Guide

This framework is latency-only and covers:

- Dense vs sparse retrieval
  - Dense: FAISS, ChromaDB
  - Sparse: BM25
- Less vs more indexed docs (doc-count sweep)
- RAG vs No-RAG
- CPU vs GPU

## 1) Run full pipeline

```bash
python3 run_full_pipeline.py
```

With custom matrix:

```bash
python3 run_full_pipeline.py \
  --skip-setup \
  --doc-counts 50,200 \
  --dense-backends faiss,chroma \
  --top-k 5 \
  --max-new-tokens 150
```

## 2) Run latency matrix only (single device)

CPU:

```bash
python3 evaluation/experiments/latency_eval.py \
  --no-cuda \
  --doc-counts 50,200 \
  --dense-backends faiss,chroma
```

GPU:

```bash
python3 evaluation/experiments/latency_eval.py \
  --doc-counts 50,200 \
  --dense-backends faiss,chroma
```

## 3) Run CPU/GPU comparison + charts

```bash
python3 evaluation/experiments/compare_latency.py --visualize
```

## Outputs

JSON:

- `evaluation/results/latency/cpu/latency_results.json`
- `evaluation/results/latency/cpu/latency_raw_profiles.json`
- `evaluation/results/latency/gpu/latency_results.json`
- `evaluation/results/latency/gpu/latency_raw_profiles.json`
- `evaluation/results/latency/cpu_gpu_comparison.json`

Charts:

- `evaluation/visualizations/latency/cpu_stepwise_latency.png`
- `evaluation/visualizations/latency/gpu_stepwise_latency.png`
- `evaluation/visualizations/latency/cpu_doc_scaling.png`
- `evaluation/visualizations/latency/gpu_doc_scaling.png`
- `evaluation/visualizations/latency/cpu_rag_vs_no_rag_overhead.png`
- `evaluation/visualizations/latency/gpu_rag_vs_no_rag_overhead.png`
- `evaluation/visualizations/latency/cpu_gpu_total_speedup.png`
- `evaluation/visualizations/latency/cpu_gpu_component_speedup_heatmap.png`

## Metrics in JSON

Each method includes:

- mean/std/min/max for:
  - `query_encoding_ms`
  - `ann_search_ms`
  - `prompt_construction_ms`
  - `generation_ms`
  - `total_ms`
- component percentages
- bottleneck component and bottleneck percentage

Methods are named like:

- `dense_faiss_docs_50`
- `dense_chroma_docs_200`
- `sparse_bm25_docs_50`
- `no_rag`

## Notes

- ChromaDB is included as a dense backend.
- `store_chroma_index.py` builds the full Chroma index in setup.
- Framework is latency-only; quality metrics are intentionally excluded.
