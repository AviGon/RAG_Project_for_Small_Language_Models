# Quickstart (Latency-Only)

This quickstart runs the current latency-first framework.

## 1) Install dependencies

python3 -m pip install -r requirements.txt

## 2) First run (includes setup)

python3 run_full_pipeline.py

This will:
- extract and chunk
- build embeddings
- build FAISS and Chroma indices
- run latency evaluation matrix
- compare CPU/GPU
- generate charts

## 3) Re-run evaluation only

python3 run_full_pipeline.py --skip-setup

## 4) Fast sanity run (CPU only)

python3 run_full_pipeline.py --skip-setup --skip-gpu --max-questions 3 --max-new-tokens 64

## 5) Multi-LLM run

python3 run_full_pipeline.py --skip-setup --llm-models Qwen/Qwen2.5-1.5B-Instruct,microsoft/Phi-3-mini-4k-instruct,TinyLlama/TinyLlama-1.1B-Chat-v1.0

## 6) Customize matrix

python3 run_full_pipeline.py --skip-setup --doc-counts 50,200 --dense-backends faiss,chroma --top-k 5

## Outputs

Per model results:
- evaluation/results/latency/<model_slug>/cpu/latency_results.json
- evaluation/results/latency/<model_slug>/gpu/latency_results.json
- evaluation/results/latency/<model_slug>/cpu_gpu_comparison.json

Cross-model summary:
- evaluation/results/latency/multi_model_summary.json

Per model charts:
- evaluation/visualizations/latency/<model_slug>/

Important charts include:
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

## Sanity checks

Check setup and pipeline state:

python3 check_status.py

## Troubleshooting

If you hit NumPy/Torch compatibility errors, keep NumPy below 2:

python3 -m pip install "numpy<2" --upgrade
