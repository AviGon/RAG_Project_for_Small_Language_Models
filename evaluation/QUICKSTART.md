# 🚀 Quick Start Guide: RAG Evaluation System

## What Was Implemented

A complete evaluation framework to assess your RAG system with:

✅ **20-question evaluation dataset** (15 in-document, 5 out-of-domain)  
✅ **RAG vs No-RAG comparison** (accuracy + latency)  
✅ **Comprehensive metrics** (ROUGE, BLEU, F1, Exact Match, context grounding)  
✅ **Latency profiling** (breakdown by pipeline component)  
✅ **Ablation studies** (top-k and temperature testing)  
✅ **Visualization tools** (automated chart generation)  
✅ **Interactive question adding** (expand your dataset easily)

---

## ⚠️ IMPORTANT: First-Time Setup

**If starting from scratch**, you need to build the RAG system first!

### Option A: Automated (Recommended)

Run the complete pipeline automatically:

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything: setup + evaluation + visualization
python run_full_pipeline.py
```

This handles everything in the correct order (30-45 minutes).

---

### Option B: Manual Setup

If the RAG system isn't set up yet, run these first:

```bash
# 1. Extract and chunk PDF
python extract_and_chunk.py

# 2. Generate embeddings
python generate_embeddings.py

# 3. Build FAISS index
python store_faiss_index.py
```

**Then** proceed with the evaluation steps below.

---

## 3-Step Quick Start (Assuming RAG is Set Up)

### Step 1: Run Main Evaluation (15-30 minutes)

```bash
# From project root directory
python3 evaluation/experiments/run_baseline_eval.py
```

**What it does:**
- Tests all 20 questions in both RAG and No-RAG modes
- Computes metrics and latency for each
- Saves detailed results to `evaluation/results/`
- Prints comparison reports to console

**Expected Output:**
```
RAG Generation Quality Metrics
============================================================
  Exact Match..................... 0.4400
  F1 Score........................ 0.4750
  ROUGE-L......................... 0.5200
  BLEU............................ 0.3800

RAG Latency Profile (n=20 runs)
============================================================
  Query Encoding.................. 45.2ms ± 3.1ms   (8.3%)
  FAISS Retrieval................. 12.5ms ± 1.8ms   (2.3%)
  Prompt Construction............. 5.1ms ± 0.4ms    (0.9%)
  LLM Generation.................. 479.3ms ± 12.5ms (88.5%)
  TOTAL........................... 542.1ms ± 13.2ms (100.0%)
```

---

### Step 2: Generate Visualizations (< 1 minute)

```bash
python3 evaluation/visualizations/plot_results.py
```

**Generated Plots:**
- `evaluation/visualizations/latency_breakdown.png`
- `evaluation/visualizations/metrics_comparison.png`
- `evaluation/visualizations/in_vs_out_document.png`
- `evaluation/visualizations/accuracy_latency_tradeoff.png`

---

### Step 3: Run Ablation Studies (Optional, 20-40 minutes)

Test different top-k values to find optimal configuration:

```bash
python3 evaluation/experiments/ablation_studies.py --study top_k
```

---

### Step 4: GPU vs CPU Comparison (Optional, 30-60 minutes)

Compare latency and performance across devices:

```bash
# Automatically runs evaluation on both GPU and CPU, then compares
python3 evaluation/experiments/compare_devices.py
```

**What this does:**
- Runs RAG evaluation on GPU
- Runs same evaluation on CPU
- Generates detailed comparison:
  - Component-wise latency (query encoding, retrieval, generation)
  - End-to-end latency comparison
  - Speedup factors for each stage
  - Throughput analysis
  - Time savings quantification
- Creates comprehensive visualization

**Generated output:**
- `evaluation/results/device_comparison/device_comparison.json`
- `evaluation/results/device_comparison/device_comparison.png`

**If you want to use existing results:**
```bash
python3 evaluation/experiments/compare_devices.py --use-existing
```

---

## Key Files to Check

### Results
- **`evaluation/results/summary.json`** - All aggregate metrics
- **`evaluation/results/generation_quality/question_*.json`** - Per-question details
- **`evaluation/results/latency_profiles/`** - Timing breakdowns

### Visualizations
- Check `evaluation/visualizations/*.png` after generating plots

### Dataset
- **`evaluation/dataset/evaluation_dataset.json`** - Current questions
- Add more with: `python evaluation/dataset/add_questions.py`

---

## Understanding Your Results

### What to Look For:

#### ✅ **Good RAG Performance:**
- **ROUGE-L > 0.4** on in-document questions
- **Grounding score > 0.8** (answer uses context)
- **Low hallucination** on out-of-document questions
- **RAG overhead < 15%** of total latency

#### ⚠️ **Potential Issues:**
- **Low grounding score** → Model ignoring retrieved context
- **High latency** → Try reducing top-k value
- **Poor accuracy** → Check chunk quality or try different retrieval

---

## Common Commands

```bash
# Basic evaluation (CPU)
python3 evaluation/experiments/run_baseline_eval.py --no-cuda

# Custom output directory
python3 evaluation/experiments/run_baseline_eval.py --output my_results/

# Compare GPU vs CPU performance
python3 evaluation/experiments/compare_devices.py

# Use existing device comparison results
python3 evaluation/experiments/compare_devices.py --use-existing

# List all questions in dataset
python3 evaluation/dataset/add_questions.py --list

# Add question interactively
python3 evaluation/dataset/add_questions.py

# Test specific k values
python3 evaluation/experiments/ablation_studies.py --study top_k --k-values 3 7 15

# Generate plots from custom results
python3 evaluation/visualizations/plot_results.py --results-dir my_results/

# Check pipeline status
python3 check_status.py
```

---

## Customizing the Evaluation

### Add Your Own Questions

**Interactive mode:**
```bash
python3 evaluation/dataset/add_questions.py
```

**Command-line mode:**
```bash
python3 evaluation/dataset/add_questions.py \
  --question "What is open pedagogy?" \
  --answer "Teaching approach using OER for collaborative learning" \
  --category definitional \
  --in-document
```

### Modify Evaluation Parameters

Edit values in the evaluation scripts:

**File:** `evaluation/experiments/run_baseline_eval.py`
```python
TOP_K = 5              # Number of retrieved chunks
MAX_NEW_TOKENS = 300   # Max answer length
```

---

## Interpreting Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **ROUGE-L** | Sequence similarity with reference | > 0.4 |
| **F1 Score** | Token-level overlap | > 0.4 |
| **Exact Match** | Perfect string match | > 0.3 |
| **BLEU** | N-gram precision | > 0.3 |
| **Grounding Score** | % answer from context | > 0.8 |

---

## Troubleshooting

### "Out of memory" error
```bash
# Use CPU instead of GPU
python evaluation/experiments/run_baseline_eval.py --no-cuda
```

### "FAISS index not found"
```bash
# Make sure you've run the indexing pipeline first
python store_faiss_index.py
```

### Evaluation is too slow
- **Expected:** 1-2 min/question on CPU, 30-60s on GPU
- **Speed up:** Use GPU or test on fewer questions (edit dataset JSON)

### Import errors
```bash
# Ensure you're in the project root directory
cd /path/to/RAG_Project_for_Small_Language_Models
python evaluation/experiments/run_baseline_eval.py
```

---

## Next Steps for Your Research

1. **Run baseline evaluation** → Get initial metrics
2. **Analyze per-question results** → Identify failure cases
3. **Run ablation studies** → Find optimal parameters
4. **Add domain-specific questions** → Test on your use cases
5. **Compare retrieval methods** → Try BM25 vs FAISS
6. **Experiment with prompts** → Improve instruction following

---

## Expected Results (Typical RAG Systems)

### Generation Quality

| Metric | RAG | No-RAG | Improvement |
|--------|-----|--------|-------------|
| ROUGE-L | 0.50 | 0.20 | **+150%** |
| F1 Score | 0.45 | 0.15 | **+200%** |
| Exact Match | 0.40 | 0.08 | **+400%** |

### Latency

- **RAG Total:** ~540ms
- **No-RAG Total:** ~480ms
- **Overhead:** ~60ms (12%)

**Conclusion:** RAG adds minimal latency (~12%) for massive accuracy gains (+150-400%)

---

## Questions?

- Check detailed docs: `evaluation/README.md`
- Get help on scripts: `python <script>.py --help`
- Review example results: After running Step 1-2 above

---

**Ready to start? Run Step 1!** 🚀

```bash
python3 evaluation/experiments/run_baseline_eval.py
```
