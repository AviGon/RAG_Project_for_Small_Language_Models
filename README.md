# RAG Project for Small Language Models (Latency Focus)

This project benchmarks local RAG latency end-to-end with a strict profiling pipeline:

- Query Encoding
- ANN / Retrieval Search
- Prompt Construction
- Generation

It supports these comparisons:

- Dense vs Sparse retrieval
	- Dense: FAISS and ChromaDB
	- Sparse: BM25
- Fewer vs more indexed documents (doc-count sweeps)
- RAG vs No-RAG
- CPU vs GPU

## Core Models

- LLM: `microsoft/Phi-3-mini-4k-instruct`
- Embedding model: `BAAI/bge-small-en-v1.5`

## Setup Artifacts

- `chunks.pkl` (chunked corpus)
- `embeddings.npy` (dense embeddings)
- `faiss_index.bin` (FAISS dense index)
- `chroma_db/` (Chroma dense index)

## Run Full Latency Pipeline

```bash
python3 run_full_pipeline.py
```

Useful options:

```bash
python3 run_full_pipeline.py \
	--skip-setup \
	--doc-counts 50,200 \
	--dense-backends faiss,chroma \
	--top-k 5 \
	--max-new-tokens 150
```

## Run CPU/GPU Latency Comparison Directly

```bash
python3 evaluation/experiments/compare_latency.py --visualize
```

## Latency Outputs

JSON:

- `evaluation/results/latency/cpu/latency_results.json`
- `evaluation/results/latency/gpu/latency_results.json`
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

## Data

- Open Education Handbook PDF used as source corpus.
