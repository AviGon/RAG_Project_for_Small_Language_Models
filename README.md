# RAG Project for Small Language Models

**Retrieval-Augmented Generation (RAG) system using Phi-3 and FAISS for on-device inference.**

This project demonstrates how small language models can leverage external knowledge through RAG, enabling accurate answers without requiring all knowledge to be stored in model weights.

## 🎯 Project Goals

- Build a local RAG pipeline using small models (Phi-3-mini)
- Evaluate RAG vs No-RAG performance comprehensively
- Profile latency breakdown across pipeline components
- Demonstrate that tiny models + retrieval >> larger models alone

## 🏗️ System Architecture

**Model:** microsoft/Phi-3-mini-4k-instruct (3.8B parameters)  
**Embedding Model:** BAAI/bge-small-en-v1.5 (33M parameters, 384-dim)  
**Vector DB:** FAISS (IndexFlatIP for cosine similarity)  
**Dataset:** Open Education Handbook (PDF)

### Pipeline Components

1. **Document Processing** - Extract and chunk PDF text
2. **Embedding Generation** - Convert chunks to dense vectors
3. **Index Building** - Create FAISS index for fast retrieval
4. **Query & Generation** - Retrieve context + generate answers
5. **Evaluation** - Compare RAG vs No-RAG with comprehensive metrics

## 🚀 Quick Start (From Scratch)

### Option 1: Automated Full Pipeline

Run everything with one command:

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (setup + evaluation + visualization)
python run_full_pipeline.py
```

This will:
1. Extract text from PDF and create chunks
2. Generate embeddings for all chunks
3. Build FAISS index
4. Run RAG vs No-RAG evaluation (20 questions)
5. Generate visualization plots

**Expected time:** ~30-45 minutes on CPU

---

### Option 2: Manual Step-by-Step

If you prefer to run each step manually:

#### Step 1: Setup RAG System

```bash
# 1. Extract and chunk PDF
python extract_and_chunk.py

# 2. Generate embeddings
python generate_embeddings.py

# 3. Build FAISS index
python store_faiss_index.py
```

**Output files:**
- `chunks.pkl` - Text chunks (500 char with 100 overlap)
- `embeddings.npy` - Vector embeddings (N × 384)
- `faiss_index.bin` - FAISS search index
- `metadata.pkl` - Chunk metadata

#### Step 2: Test Interactive Query (Optional)

```bash
# Interactive Q&A with the RAG system
python query_and_generate.py
```

#### Step 3: Run Evaluation

```bash
# Comprehensive RAG vs No-RAG evaluation
python evaluation/experiments/run_baseline_eval.py

# Generate visualizations
python evaluation/visualizations/plot_results.py
```

## 📊 Evaluation System

Comprehensive evaluation framework for RAG performance analysis.

### What's Measured

- **Generation Quality:** ROUGE-L, BLEU, F1, Exact Match
- **RAG-Specific:** Context grounding, hallucination detection
- **Latency:** Component-wise breakdown (encoding, retrieval, generation)
- **Ablation Studies:** Top-k values, temperature settings

### Quick Evaluation Commands

```bash
# Basic evaluation (RAG vs No-RAG)
python evaluation/experiments/run_baseline_eval.py

# Run on CPU only
python evaluation/experiments/run_baseline_eval.py --no-cuda

# Ablation studies (test different top-k values)
python evaluation/experiments/ablation_studies.py --study top_k

# Generate all visualizations
python evaluation/visualizations/plot_results.py
```

### Evaluation Documentation

- **Quick Start:** [evaluation/QUICKSTART.md](evaluation/QUICKSTART.md)
- **Detailed Guide:** [evaluation/README.md](evaluation/README.md)

## 📁 Project Structure

```
RAG_Project_for_Small_Language_Models/
├── run_full_pipeline.py              # Master script (run everything)
│
├── extract_and_chunk.py              # Step 1: PDF → text chunks
├── generate_embeddings.py            # Step 2: Chunks → embeddings
├── store_faiss_index.py              # Step 3: Build FAISS index
├── query_and_generate.py             # Interactive testing
│
├── data/
│   └── handbook.pdf                  # Source document
│
├── evaluation/                       # Evaluation framework
│   ├── QUICKSTART.md                 # Quick start guide
│   ├── README.md                     # Detailed documentation
│   │
│   ├── dataset/
│   │   ├── evaluation_dataset.json   # 20 test questions
│   │   └── add_questions.py          # Add more questions
│   │
│   ├── metrics/
│   │   ├── latency_metrics.py        # Timing utilities
│   │   └── generation_metrics.py     # Quality metrics
│   │
│   ├── experiments/
│   │   ├── run_baseline_eval.py      # Main evaluation
│   │   └── ablation_studies.py       # Parameter testing
│   │
│   ├── visualizations/
│   │   └── plot_results.py           # Generate charts
│   │
│   └── results/                      # Generated results
│       ├── summary.json
│       ├── latency_profiles/
│       └── generation_quality/
│
├── requirements.txt
└── README.md

**Generated Files (after running pipeline):**
- chunks.pkl
- embeddings.npy  
- faiss_index.bin
- metadata.pkl
```

## 📦 Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- torch
- transformers
- sentence-transformers
- faiss-cpu
- matplotlib
- numpy
- pymupdf

## 🎓 Data Source

**Open Education Handbook**  
URL: https://oerpolicy.eu/wp-content/uploads/sites/4/2017/07/Open-Education-Handbook.pdf

Topics covered: Open Educational Resources (OER), Creative Commons, MOOCs, open pedagogy, etc.

## 🔬 Expected Results

### Generation Quality (In-Document Questions)

| Metric | RAG | No-RAG | Improvement |
|--------|-----|--------|-------------|
| ROUGE-L | ~0.50 | ~0.20 | **+150%** |
| F1 Score | ~0.45 | ~0.15 | **+200%** |
| Exact Match | ~0.40 | ~0.08 | **+400%** |

### Latency Profile (CPU)

| Component | Time | % of Total |
|-----------|------|------------|
| Query Encoding | ~45ms | 8% |
| FAISS Search | ~12ms | 2% |
| Prompt Build | ~5ms | 1% |
| LLM Generation | ~480ms | 89% |
| **Total** | **~540ms** | **100%** |

**Key Insight:** RAG adds only ~60ms (12%) overhead for massive accuracy gains.

## 🛠️ Customization

### Change Chunk Size

Edit `extract_and_chunk.py`:
```python
CHUNK_SIZE = 500  # Change to 300, 700, 1000, etc.
OVERLAP = 100     # Overlap between chunks
```

Then rebuild: `python run_full_pipeline.py --force`

### Change Top-K Retrieval

Edit evaluation scripts or run ablation:
```python
TOP_K = 5  # Number of chunks to retrieve
```

### Use Different Model

Edit `query_and_generate.py` or evaluation scripts:
```python
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Change to other model
```

### Add Your Own Questions

```bash
# Interactive mode
python evaluation/dataset/add_questions.py

# Or edit evaluation_dataset.json directly
```

## 📈 Useful Commands

```bash
# Full pipeline from scratch
python run_full_pipeline.py

# Setup only (skip evaluation)
python run_full_pipeline.py --setup-only

# Evaluation only (skip setup if already done)
python run_full_pipeline.py --skip-setup

# Force rebuild even if files exist
python run_full_pipeline.py --force

# List all evaluation questions
python evaluation/dataset/add_questions.py --list

# Interactive Q&A testing
python query_and_generate.py
```

## 🐛 Troubleshooting

### "FAISS index not found"
Run the setup steps first: `python run_full_pipeline.py --setup-only`

### "Out of memory"
Use CPU mode: `python evaluation/experiments/run_baseline_eval.py --no-cuda`

### Slow performance
- Expected: 1-2 min/question on CPU
- Speed up: Use GPU or reduce dataset size

### Import errors
Make sure you're in the project root directory when running scripts.

## 📚 Further Reading

- **Evaluation Guide:** [evaluation/README.md](evaluation/README.md)
- **Quick Start:** [evaluation/QUICKSTART.md](evaluation/QUICKSTART.md)
- **Research Context:** See project goals in initial prompt

## 🤝 Contributing

To extend this project:
1. Add more evaluation questions to test specific scenarios
2. Implement sparse retrieval (BM25) for comparison
3. Test different embedding models
4. Experiment with RAG-specific fine-tuning (RAFT)
5. Add retrieval quality metrics (Precision@k, Recall@k)

## 📄 License

This is a research project. Please cite appropriately if used in academic work.
