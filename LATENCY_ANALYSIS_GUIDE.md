# Latency Analysis Guide

## Your Current Results Summary

### Bottleneck Identified: LLM Generation (99.5%)

From your CPU evaluation results:

| Pipeline Component | Time (ms) | % of Total | Optimization Potential |
|-------------------|-----------|------------|----------------------|
| Query Encoding | 44.27 | 0.4% | Low priority |
| FAISS Search | 7.74 | 0.1% | Low priority |
| Prompt Construction | 0.01 | 0.0% | Low priority |
| **Phi-3 Generation** | **10,667** | **99.5%** | **🎯 HIGH PRIORITY** |
| **Total** | **10,719** | **100%** | |

### Key Insights:
- ✅ **Retrieval is fast**: Only 52ms overhead for RAG vs No-RAG
- ⚠️ **Generation is the bottleneck**: 10.6 seconds per question
- 💡 **GPU will help most with generation**, not retrieval

---

## What You Need to Analyze

### 1. Visualize Current Results

```bash
cd /Users/bhavya/Desktop
python3 evaluation/visualizations/plot_results.py \
  --results-dir evaluation/results \
  --output-dir evaluation/visualizations
```

**Output:**
- `latency_breakdown.png` - Shows 99.5% generation dominance
- `metrics_comparison.png` - RAG vs No-RAG quality
- `in_vs_out_document.png` - Performance on different question types
- `accuracy_latency_tradeoff.png` - Speed vs quality comparison

---

### 2. GPU vs CPU Comparison (Critical!)

This is what you really need for bottleneck analysis.

#### Step 1: Check if you have GPU

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Step 2: Run Device Comparison

**If you have GPU:**
```bash
cd /Users/bhavya/Desktop/ms_projects/RAG_Project_for_Small_Language_Models

# Run evaluation on both devices
python3 evaluation/experiments/compare_devices.py
```

**Expected speedup analysis:**
```
Component Speedup Factors:
- Query Encoding: ~2-3x faster on GPU
- FAISS Search: ~1.1x (already fast)
- Generation: ~10-50x faster on GPU  ← THIS IS KEY!
- End-to-end: ~10-30x faster overall
```

#### Step 3: Analyze Output

**Generated files:**
- `evaluation/results/device_comparison/device_comparison.png`
  - 6 panels showing:
    1. Component-wise latency comparison
    2. Speedup factors per component
    3. End-to-end time comparison
    4. Throughput (queries/sec)
    5. Time savings projection
    6. Accuracy comparison

- `evaluation/results/device_comparison/device_comparison.json`
  - Detailed numerical comparison

---

### 3. Component-Level Analysis

#### Query Encoding (44ms on CPU)
- **Model:** BAAI/bge-small-en-v1.5 (33M params, 384-dim)
- **Current:** 0.4% of total time
- **GPU Speedup Expected:** 2-3x → ~15-20ms
- **Impact:** Minimal (saves ~25ms per query)

#### FAISS Search (7.74ms on CPU)
- **Index:** IndexFlatIP (brute-force cosine similarity)
- **Current:** 0.1% of total time
- **GPU Speedup Expected:** ~1.1x (already near-optimal)
- **Impact:** Negligible

#### Prompt Construction (0.01ms)
- **Current:** Pure Python string manipulation
- **Optimization:** Not worth it - already instant

#### **Phi-3 Generation (10,667ms on CPU)** ⚠️
- **Model:** microsoft/Phi-3-mini-4k-instruct (3.8B params)
- **Current:** 99.5% of total time (10.7 seconds!)
- **Settings:** max_new_tokens=300, temperature=0.2
- **GPU Speedup Expected:** 10-50x → **200-1000ms**
- **Impact:** MASSIVE - this is your bottleneck!

**Why so slow on CPU?**
- 3.8B parameters = ~7.6GB model size (FP16)
- CPU memory bandwidth: ~40 GB/s
- GPU memory bandwidth: ~300-900 GB/s (7-20x faster)
- CPU FLOPS: ~100 GFLOPS
- GPU FLOPS: ~10-50 TFLOPS (100-500x faster)

---

## 4. Detailed Breakdown Analysis

### What to Look For:

#### A. Variance Analysis
From your results:
```json
"std": {
  "query_encoding_ms": 155.88,  // High variance! Why?
  "retrieval_ms": 0.37,          // Consistent
  "generation_ms": 67.75         // Relatively stable
}
```

**Query encoding variance (155ms std):** 
- Min: 8.23ms, Max: 706ms
- Possible causes:
  - First query loads model into cache
  - Cold start penalty
  - Memory swapping if RAM is tight

#### B. Outlier Detection
```json
"max": {
  "query_encoding_ms": 706.5,   // ⚠️ Outlier! 16x slower than min
  "total_ms": 11631.07          // ~1 second slower than average
}
```

**Actions:**
- Check if first question has cold-start penalty
- Look at `generation_quality/question_*.json` to find which question took 11.6 seconds

#### C. Consistency Metrics
```json
"min": {
  "generation_ms": 10587.04,
  "total_ms": 10604.0
},
"max": {
  "generation_ms": 10915.41,
  "total_ms": 11631.07
}
```

Generation time range: 10.6 - 10.9 seconds (fairly consistent)

---

## 5. Expected GPU Results

When you run on GPU, expect these approximate values:

### CPU Results (Current):
```
Query Encoding:    44ms
FAISS Search:      8ms
Prompt Build:      0ms
Generation:        10,667ms  ← Bottleneck
─────────────────────────
Total:             10,719ms (10.7 seconds/question)
Throughput:        0.09 questions/sec
```

### GPU Results (Expected):
```
Query Encoding:    ~15ms      (3x speedup)
FAISS Search:      ~7ms       (1.1x speedup)
Prompt Build:      0ms        (same)
Generation:        ~300-600ms (20-35x speedup) ← Major improvement!
─────────────────────────
Total:             ~320-620ms (0.3-0.6 seconds/question)
Throughput:        1.6-3.1 questions/sec
Overall Speedup:   17-33x
```

---

## 6. Optimization Recommendations

Based on bottleneck analysis:

### High Impact (Generation Bottleneck):
1. ✅ **Use GPU** - 20-35x speedup expected
2. ⚡ **Reduce max_new_tokens** - Currently 300, try 150
3. 🔧 **Use model quantization** - INT8/INT4 for faster inference
4. 💾 **Enable KV-cache** - Faster generation with caching

### Medium Impact:
5. 📉 **Reduce top_k** - Fewer chunks = shorter prompts = faster generation
6. 🎯 **Batch queries** - Process multiple questions together on GPU

### Low Impact (Not Worth It):
- Optimizing query encoding (0.4% of time)
- Optimizing FAISS search (0.1% of time)
- Optimizing prompt construction (negligible)

---

## 7. Commands to Run Analysis

### Copy results to project directory (if not already)
```bash
cp -r /Users/bhavya/Desktop/evaluation/results \
      /Users/bhavya/Desktop/ms_projects/RAG_Project_for_Small_Language_Models/evaluation/
```

### Generate visualizations
```bash
cd /Users/bhavya/Desktop/ms_projects/RAG_Project_for_Small_Language_Models

python3 evaluation/visualizations/plot_results.py \
  --results-dir evaluation/results \
  --output-dir evaluation/visualizations
```

### Run GPU comparison (if available)
```bash
python3 evaluation/experiments/compare_devices.py
```

### View results
```bash
open evaluation/visualizations/latency_breakdown.png
open evaluation/results/device_comparison/device_comparison.png
```

---

## 8. Analysis Outputs

After running the commands above, you'll have:

### Numerical Analysis:
- `evaluation/results/summary.json` - ✅ Already have
- `evaluation/results/latency_profiles/rag_latency.json` - ✅ Already have
- `evaluation/results/device_comparison/device_comparison.json` - Need GPU run

### Visualizations:
- `latency_breakdown.png` - Pie chart showing 99.5% generation
- `accuracy_latency_tradeoff.png` - Quality vs speed
- `device_comparison.png` - 6-panel GPU vs CPU analysis

### Key Metrics to Report:
1. **End-to-end latency:** 10.7s (CPU) vs ~0.5s (GPU expected)
2. **Bottleneck:** Generation takes 99.5% of time
3. **Retrieval overhead:** Only 52ms (negligible)
4. **Speedup potential:** 20-35x with GPU
5. **Throughput:** 0.09 q/s (CPU) vs ~2 q/s (GPU expected)

---

## 9. What This Tells You About Small Language Models

### Findings:
- ✅ **Embeddings are fast:** BGE-small (33M params) takes <50ms
- ✅ **Vector search is fast:** FAISS takes ~8ms for 254 chunks
- ⚠️ **Generation is slow:** Phi-3 (3.8B params) takes 10+ seconds on CPU
- 💡 **RAG overhead is minimal:** Only 0.5% slower than No-RAG

### Implications:
- For **CPU deployment**: Generation will always be the bottleneck
- For **GPU deployment**: Even 3.8B models can be fast (~300-600ms)
- **Retrieval is not the bottleneck** - no need to optimize FAISS further
- **Small models benefit greatly from GPU** - worth the hardware investment

---

## 10. Next Steps

1. ✅ **You have:** Complete CPU latency breakdown
2. 🔄 **You need:** GPU evaluation for comparison
3. 📊 **Then generate:** Comparative visualizations
4. 📝 **Finally analyze:** Speedup factors and bottleneck resolution

**If you have GPU access, run:**
```bash
python3 evaluation/experiments/compare_devices.py
```

**If you don't have GPU, you can still:**
- Visualize current CPU results
- Report that generation is the bottleneck (99.5%)
- Recommend GPU for 20-35x expected speedup
- Show theoretical speedup calculations
