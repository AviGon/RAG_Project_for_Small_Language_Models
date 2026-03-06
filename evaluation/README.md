# RAG Evaluation System

Comprehensive evaluation framework for comparing **RAG vs No-RAG** performance on the Open Education Handbook dataset.

## 📋 Overview

This evaluation system provides:
- **Baseline Evaluation**: RAG vs No-RAG comparison
- **Latency Profiling**: Breakdown of pipeline components
- **Generation Metrics**: ROUGE-L, BLEU, F1, Exact Match
- **RAG-Specific Metrics**: Context grounding, hallucination detection
- **Ablation Studies**: Top-k values, temperature settings
- **Visualizations**: Comprehensive plots and charts

## 🏗️ Structure

```
evaluation/
├── dataset/
│   └── evaluation_dataset.json          # 20 evaluation questions (15 in-doc, 5 out-of-doc)
│
├── metrics/
│   ├── latency_metrics.py               # Timing utilities
│   └── generation_metrics.py            # ROUGE, BLEU, F1, grounding
│
├── experiments/
│   ├── run_baseline_eval.py             # Main RAG vs No-RAG evaluation
│   └── ablation_studies.py              # Parameter sensitivity tests
│
├── results/                             # Generated evaluation results
│   ├── summary.json
│   ├── latency_profiles/
│   └── generation_quality/
│
└── visualizations/
    ├── plot_results.py                  # Visualization generator
    └── *.png                            # Generated plots
```

## 🚀 Quick Start

### 1. Run Baseline Evaluation

Compare RAG vs No-RAG on all questions:

```bash
# From project root directory
python evaluation/experiments/run_baseline_eval.py
```

**Output:**
- Console reports with metrics and latency breakdowns
- `evaluation/results/summary.json` - Aggregate statistics
- `evaluation/results/generation_quality/` - Per-question results
- `evaluation/results/latency_profiles/` - Timing data

**Expected Runtime:** ~15-30 minutes (20 questions × 2 modes)

### 2. Generate Visualizations

Create plots from evaluation results:

```bash
python evaluation/visualizations/plot_results.py
```

**Generated Plots:**
- `latency_breakdown.png` - Pie chart + bar comparison
- `metrics_comparison.png` - RAG vs No-RAG quality metrics
- `in_vs_out_document.png` - Performance by question type
- `accuracy_latency_tradeoff.png` - Efficiency analysis

### 3. Run Ablation Studies

Test different top-k values:

```bash
# Test k = 1, 3, 5, 10
python evaluation/experiments/ablation_studies.py --study top_k

# Test different temperatures
python3 evaluation/experiments/ablation_studies.py --study temperature

# Run all ablation studies
python3 evaluation/experiments/ablation_studies.py --study all
```

### 4. Device Comparison (GPU vs CPU)

Compare performance and latency across different compute devices:

```bash
# Run evaluation on both GPU and CPU, then compare
python3 evaluation/experiments/compare_devices.py

# Use existing results (skip re-running evaluations)
python3 evaluation/experiments/compare_devices.py --use-existing

# Run only one device
python3 evaluation/experiments/compare_devices.py --skip-gpu  # CPU only
python3 evaluation/experiments/compare_devices.py --skip-cpu  # GPU only
```

**What's Analyzed:**

1. **Component-Wise Latency Breakdown**
   - Query encoding time (GPU vs CPU)
   - FAISS retrieval time
   - Prompt construction time
   - LLM generation time

2. **End-to-End Latency Comparison**
   - Total RAG pipeline time on each device
   - No-RAG baseline on each device
   - RAG overhead per device

3. **Speedup Analysis**
   - Speedup factor for each component
   - Overall system speedup
   - Identifies which components benefit most from GPU

4. **Throughput Metrics**
   - Queries per second on GPU
   - Queries per second on CPU
   - Time savings quantification

5. **Accuracy Verification**
   - Confirms generation quality is consistent across devices
   - Detects any numerical precision differences

**Output:**
- `device_comparison.json` - Detailed comparison data
- `device_comparison.png` - 6-panel visualization with:
  - Component latency bars
  - Total e2e comparison
  - Speedup factors
  - Percentage breakdown
  - Throughput comparison
  - Time savings pie chart

---

## 📊 Evaluation Dataset

**Location:** `evaluation/dataset/evaluation_dataset.json`

**Breakdown:**
- **15 In-Document Questions** (answerable from handbook)
  - 5 Definitional (What is OER?)
  - 5 Factual (What are the 5 Rs?)
  - 3 Procedural (How to evaluate OER quality?)
  - 2 Comparison (OER vs Open Access)
  
- **5 Out-of-Document Questions** (test hallucination control)
  - General knowledge questions not in handbook

**Adding Your Own Questions:**

Edit `evaluation_dataset.json`:
```json
{
  "id": 21,
  "question": "Your question here?",
  "reference_answer": "Expected answer",
  "category": "factual",
  "difficulty": "medium",
  "in_document": true,
  "requires_retrieval": true
}
```

## 📈 Metrics Explained

### Generation Quality Metrics

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **Exact Match** | Binary exact string match | 0-1 | Strict correctness |
| **F1 Score** | Token overlap F1 | 0-1 | Partial correctness |
| **ROUGE-L** | Longest common subsequence | 0-1 | Sequence similarity |
| **BLEU** | N-gram precision | 0-1 | Translation quality |

### RAG-Specific Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Grounding Score** | % of answer tokens in context | >0.8 |
| **Context Coverage** | % of context used in answer | 0.2-0.5 |
| **IDK Response Rate** | % of "I don't know" responses | Varies by question type |

### Latency Breakdown

- **Query Encoding**: Embed user question (~45ms)
- **FAISS Search**: k-NN retrieval (~12ms)
- **Prompt Construction**: String formatting (~5ms)
- **LLM Generation**: Token generation (~480ms)

**Expected Total:**
- RAG: ~540ms
- No-RAG: ~480ms
- **Overhead: ~60ms (12%)**

## 🔬 Advanced Usage

### Custom Evaluation Script

```python
from evaluation.experiments.run_baseline_eval import RAGEvaluator
from evaluation.metrics.generation_metrics import compute_all_metrics

evaluator = RAGEvaluator(use_cuda=True)

# Evaluate single question
question = "What does OER stand for?"
answer, contexts, latency = evaluator.evaluate_question_rag(question)

print(f"Answer: {answer}")
print(f"Latency: {latency.total_time * 1000:.1f}ms")
```

### Run Evaluation on Custom Dataset

```bash
python evaluation/experiments/run_baseline_eval.py \
  --dataset path/to/your_dataset.json \
  --output path/to/results
```

### GPU/CPU Selection

```bash
# Force CPU (slower but more compatible)
python evaluation/experiments/run_baseline_eval.py --no-cuda

# Use GPU if available (default)
python evaluation/experiments/run_baseline_eval.py
```

### Custom Top-k Values

```bash
python evaluation/experiments/ablation_studies.py \
  --study top_k \
  --k-values 1 2 5 10 15 20
```

## 📚 Understanding Results

### Summary JSON Structure

```json
{
  "overall": {
    "rag": {
      "rouge_l_mean": 0.52,
      "f1_score_mean": 0.48,
      "exact_match_mean": 0.44,
      "grounding_score_mean": 0.89
    },
    "no_rag": {
      "rouge_l_mean": 0.18,
      "f1_score_mean": 0.15,
      "exact_match_mean": 0.06
    }
  },
  "latency": {
    "rag": {"total_ms": 542},
    "no_rag": {"total_ms": 480}
  }
}
```

### Interpreting Results

**Good RAG Performance:**
- ROUGE-L > 0.4 on in-document questions
- Grounding score > 0.8
- Low hallucination on out-of-document questions

**Performance Issues:**
- Low grounding score → Model ignoring context
- High latency → Try lower top-k values
- Poor out-of-domain behavior → Adjust prompt template

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size or use CPU
python evaluation/experiments/run_baseline_eval.py --no-cuda
```

### Slow Evaluation

- **Expected:** ~1-2 minutes per question on CPU
- **Speed up:** Use GPU, reduce top-k, or test on subset
- **Subset testing:** Edit dataset JSON to include fewer questions

### Import Errors

```bash
# Ensure you're running from project root
cd /path/to/RAG_Project_for_Small_Language_Models
python evaluation/experiments/run_baseline_eval.py
```

## 📊 Expected Results

Based on similar RAG systems:

### Generation Quality (In-Document Questions)
| Metric | RAG | No-RAG | Improvement |
|--------|-----|--------|-------------|
| ROUGE-L | 0.45-0.55 | 0.15-0.25 | +150-200% |
| F1 | 0.40-0.50 | 0.10-0.20 | +200-300% |
| Exact Match | 0.35-0.45 | 0.05-0.10 | +400-600% |

### Latency (CPU)
- **RAG:** 500-600ms per query
- **No-RAG:** 450-500ms per query
- **Overhead:** 10-15%

## 🎯 Next Steps

1. **Run baseline evaluation** to get initial results
2. **Generate visualizations** to understand patterns
3. **Run ablation studies** to optimize parameters
4. **Analyze per-question results** in `results/generation_quality/`
5. **Add custom questions** to test specific scenarios
6. **Experiment with prompts** in baseline eval script

## 📝 Citation

If using this evaluation framework, please acknowledge:
- **Dataset:** Open Education Handbook (2017)
- **Embedding Model:** BAAI/bge-small-en-v1.5
- **LLM:** microsoft/Phi-3-mini-4k-instruct
- **Vector DB:** FAISS (IndexFlatIP)

## 💡 Tips for Your Research

- **Latency Analysis:** Focus on the generation component (~90% of time)
- **Context Injection:** Check grounding scores to verify attention
- **Hallucination:** Compare IDK rates on out-of-document questions
- **Retrieval Quality:** Add manual relevance labels for precision@k analysis
- **Sparse vs Dense:** Compare current FAISS with BM25 retrieval

---

**Questions?** Check individual script help:
```bash
python evaluation/experiments/run_baseline_eval.py --help
python evaluation/experiments/ablation_studies.py --help
python evaluation/visualizations/plot_results.py --help
```
