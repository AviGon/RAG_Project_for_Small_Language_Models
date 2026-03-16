# RAG Pipeline: Order of Operations

## 🔄 Current Project Status

✅ **RAG System Setup Complete**
- ✓ PDF extracted and chunked (chunks.pkl)
- ✓ Embeddings generated (embeddings.npy)
- ✓ FAISS index built (faiss_index.bin)
- ✓ Metadata stored (metadata.pkl)

⏳ **Ready to Run**
- → Evaluation (RAG vs No-RAG comparison)
- → Visualizations
- → Ablation studies

---

## 📊 Complete Workflow (From Scratch → Full Evaluation)

### Phase 1: RAG System Setup ✅ (Already Done for You)

```bash
# Step 1: Extract text from PDF and create chunks
python extract_and_chunk.py
# Output: chunks.pkl (254 KB)

# Step 2: Generate embeddings for all chunks  
python generate_embeddings.py
# Output: embeddings.npy (761 KB)

# Step 3: Build FAISS search index
python store_faiss_index.py
# Output: faiss_index.bin (761 KB), metadata.pkl
```

**Status:** ✅ Complete (files exist)

---

### Phase 2: Run Evaluation ⏳ (Next Step)

```bash
# Main evaluation: RAG vs No-RAG comparison
python evaluation/experiments/run_baseline_eval.py

# Expected runtime: 15-30 minutes
# Output: evaluation/results/
```

**What this does:**
- Tests 20 questions in both RAG and No-RAG modes
- Computes ROUGE, BLEU, F1, Exact Match
- Profiles latency for each pipeline component
- Checks context grounding and hallucination rates

---

### Phase 3: Generate Visualizations ⏳

```bash
# Create charts and plots
python evaluation/visualizations/plot_results.py

# Output: evaluation/visualizations/*.png
```

**Generated plots:**
- Latency breakdown (pie chart + comparison)
- Metrics comparison (RAG vs No-RAG)
- In-document vs out-of-document performance
- Accuracy-latency tradeoff

---

### Phase 4: Ablation Studies ⏳ (Optional)

```bash
# Test different top-k values
python evaluation/experiments/ablation_studies.py --study top_k

# Expected runtime: 20-40 minutes
```

---

## 🚀 Quick Commands for Your Current State

Since your RAG system is **already set up**, you can:

### Option 1: Run Everything Automatically

```bash
# Skip setup (already done), run evaluation + visualization
python run_full_pipeline.py --skip-setup
```

### Option 2: Run Step-by-Step

```bash
# 1. Run evaluation
python evaluation/experiments/run_baseline_eval.py

# 2. Generate visualizations  
python evaluation/visualizations/plot_results.py

# 3. (Optional) Ablation studies
python evaluation/experiments/ablation_studies.py --study top_k
```

### Option 3: Just Test Interactively

```bash
# Interactive Q&A (no evaluation metrics)
python query_and_generate.py
```

---

## 🔍 Starting From Scratch? (New Users)

If someone starts fresh without any generated files:

### Automated (Recommended)

```bash
python run_full_pipeline.py
```

Handles everything: setup → evaluation → visualization

### Manual (Full Control)

```bash
# Phase 1: Setup
python extract_and_chunk.py
python generate_embeddings.py  
python store_faiss_index.py

# Phase 2: Evaluate
python evaluation/experiments/run_baseline_eval.py

# Phase 3: Visualize
python evaluation/visualizations/plot_results.py
```

---

## 📋 Pipeline Dependencies

```
Phase 1 (Setup):
  extract_and_chunk.py
    ↓ requires: data/handbook.pdf
    ↓ produces: chunks.pkl
    
  generate_embeddings.py
    ↓ requires: chunks.pkl
    ↓ produces: embeddings.npy
    
  store_faiss_index.py
    ↓ requires: chunks.pkl, embeddings.npy
    ↓ produces: faiss_index.bin, metadata.pkl

Phase 2 (Evaluation):
  run_baseline_eval.py
    ↓ requires: faiss_index.bin, metadata.pkl, evaluation_dataset.json
    ↓ produces: results/summary.json, results/generation_quality/

Phase 3 (Visualization):
  plot_results.py
    ↓ requires: results/summary.json
    ↓ produces: visualizations/*.png
```

---

## 🛠️ Utility Commands

```bash
# Check what's been completed
python check_status.py

# List evaluation questions
python evaluation/dataset/add_questions.py --list

# Add new questions interactively
python evaluation/dataset/add_questions.py

# Run with CPU only (no GPU)
python evaluation/experiments/run_baseline_eval.py --no-cuda

# Force rebuild everything from scratch
python run_full_pipeline.py --force
```

---

## ⚡ Quick Decision Tree

**Q: Do you have chunks.pkl, embeddings.npy, faiss_index.bin?**

- ✅ **YES** → Run evaluation: `python evaluation/experiments/run_baseline_eval.py`
  
- ❌ **NO** → Run setup first:
  - Automated: `python run_full_pipeline.py`
  - Manual: `python extract_and_chunk.py` → `python generate_embeddings.py` → `python store_faiss_index.py`

**Q: Do you have evaluation/results/summary.json?**

- ✅ **YES** → Generate visualizations: `python evaluation/visualizations/plot_results.py`
  
- ❌ **NO** → Run evaluation first: `python evaluation/experiments/run_baseline_eval.py`

---

## 🎯 For Your Current Situation

**Status:** RAG setup ✅ complete | Evaluation ⏳ pending

**Recommended next step:**

```bash
# Run the automated pipeline (skip setup since it's done)
python run_full_pipeline.py --skip-setup
```

This will:
1. ✓ Skip setup (already complete)
2. → Run evaluation (15-30 min)
3. → Generate visualizations (1 min)
4. → (Optional) Ask about ablation studies

**Alternative (manual control):**

```bash
python evaluation/experiments/run_baseline_eval.py
```

Then after completion:

```bash
python evaluation/visualizations/plot_results.py
```

---

## 💡 Pro Tips

1. **Check status anytime:** `python check_status.py`
2. **Test before full eval:** Use `python query_and_generate.py` to verify system works
3. **CPU vs GPU:** Add `--no-cuda` flag if GPU issues occur
4. **Customize questions:** Edit `evaluation/dataset/evaluation_dataset.json` before running eval
5. **Incremental work:** Each phase can be run independently - results are saved

---

**Ready to proceed? Run:** `python run_full_pipeline.py --skip-setup`
