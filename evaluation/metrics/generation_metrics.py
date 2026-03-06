"""
Generation quality metrics for evaluating RAG system outputs.
Includes ROUGE, BLEU, Exact Match, BERTScore, and custom RAG-specific metrics.
"""

import re
import string
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and removing punctuation"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())  # normalize whitespace
    return text


def compute_exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization"""
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    return 1.0 if pred_norm == ref_norm else 0.0


def compute_f1_score(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference"""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L (Longest Common Subsequence) metric.
    Measures longest common subsequence between prediction and reference.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    # ROUGE-L F-score
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f_score = 2 * precision * recall / (precision + recall)
    return f_score


def compute_bleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """
    Simplified BLEU score computation with uniform weights.
    Measures n-gram precision with brevity penalty.
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # Brevity penalty
    bp = 1.0 if len(pred_tokens) > len(ref_tokens) else np.exp(1 - len(ref_tokens) / len(pred_tokens))
    
    # Compute precision for n-grams
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        
        if len(pred_ngrams) == 0:
            break
        
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        overlap = sum((pred_counter & ref_counter).values())
        precision = overlap / len(pred_ngrams) if len(pred_ngrams) > 0 else 0
        precisions.append(precision)
    
    if len(precisions) == 0 or all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    bleu = bp * np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
    
    return bleu if not np.isnan(bleu) and not np.isinf(bleu) else 0.0


def check_context_grounding(answer: str, contexts: List[str]) -> Dict[str, float]:
    """
    Check if the answer is grounded in retrieved contexts.
    Returns metrics about overlap between answer and context.
    """
    answer_tokens = set(normalize_text(answer).split())
    
    if len(answer_tokens) == 0:
        return {
            "grounding_score": 0.0,
            "token_overlap_ratio": 0.0,
            "context_coverage": 0.0
        }
    
    # Combine all contexts
    all_context_text = " ".join(contexts)
    context_tokens = set(normalize_text(all_context_text).split())
    
    if len(context_tokens) == 0:
        return {
            "grounding_score": 0.0,
            "token_overlap_ratio": 0.0,
            "context_coverage": 0.0
        }
    
    # Calculate overlap
    overlapping_tokens = answer_tokens & context_tokens
    grounding_score = len(overlapping_tokens) / len(answer_tokens)
    
    # How much of the context is used in the answer
    context_coverage = len(overlapping_tokens) / len(context_tokens)
    
    return {
        "grounding_score": round(grounding_score, 3),
        "token_overlap_ratio": round(len(overlapping_tokens) / max(len(answer_tokens), 1), 3),
        "context_coverage": round(context_coverage, 3)
    }


def detect_idk_response(answer: str) -> bool:
    """
    Detect if the model responded with "I don't know" or similar.
    Returns True if answer indicates uncertainty.
    """
    idk_patterns = [
        r"i don'?t know",
        r"not sure",
        r"cannot answer",
        r"can'?t answer",
        r"no information",
        r"not present in",
        r"not mentioned",
        r"not provided",
        r"don'?t have",
        r"unable to answer",
        r"insufficient information"
    ]
    
    answer_lower = answer.lower()
    
    for pattern in idk_patterns:
        if re.search(pattern, answer_lower):
            return True
    
    return False


def compute_all_metrics(prediction: str, reference: str, contexts: List[str] = None) -> Dict:
    """
    Compute all generation quality metrics.
    
    Args:
        prediction: Model's generated answer
        reference: Ground truth reference answer
        contexts: Retrieved context chunks (optional, for RAG-specific metrics)
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        "exact_match": compute_exact_match(prediction, reference),
        "f1_score": compute_f1_score(prediction, reference),
        "rouge_l": compute_rouge_l(prediction, reference),
        "bleu": compute_bleu_score(prediction, reference),
        "is_idk_response": detect_idk_response(prediction)
    }
    
    # Add RAG-specific metrics if contexts are provided
    if contexts is not None and len(contexts) > 0:
        grounding_metrics = check_context_grounding(prediction, contexts)
        metrics.update(grounding_metrics)
    
    return metrics


def aggregate_metrics(results: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple evaluation examples.
    
    Args:
        results: List of metric dictionaries from compute_all_metrics
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not results:
        return {}
    
    # Collect all metric values
    metric_names = [k for k in results[0].keys() if not isinstance(results[0][k], bool)]
    bool_metric_names = [k for k in results[0].keys() if isinstance(results[0][k], bool)]
    
    aggregated = {}
    
    # Aggregate numeric metrics
    for metric in metric_names:
        values = [r[metric] for r in results if metric in r]
        if values:
            aggregated[f"{metric}_mean"] = round(np.mean(values), 4)
            aggregated[f"{metric}_std"] = round(np.std(values), 4)
    
    # Aggregate boolean metrics (as percentages)
    for metric in bool_metric_names:
        values = [r[metric] for r in results if metric in r]
        if values:
            aggregated[f"{metric}_pct"] = round(np.mean(values) * 100, 2)
    
    aggregated["n_samples"] = len(results)
    
    return aggregated


def format_metrics_report(metrics: Dict, mode: str = "RAG") -> str:
    """Format metrics into a readable report"""
    report = [
        f"\n{'='*60}",
        f"{mode} Generation Quality Metrics",
        f"{'='*60}\n"
    ]
    
    # Extract mean metrics
    metric_display = [
        ("Exact Match", "exact_match_mean"),
        ("F1 Score", "f1_score_mean"),
        ("ROUGE-L", "rouge_l_mean"),
        ("BLEU", "bleu_mean")
    ]
    
    for name, key in metric_display:
        if key in metrics:
            report.append(f"  {name:.<30} {metrics[key]:.4f}")
    
    # Add RAG-specific metrics if present
    if "grounding_score_mean" in metrics:
        report.append("\nRAG-Specific Metrics:")
        report.append(f"  {'Context Grounding':.<30} {metrics['grounding_score_mean']:.4f}")
        report.append(f"  {'Context Coverage':.<30} {metrics.get('context_coverage_mean', 0):.4f}")
    
    # Add boolean metrics
    if "is_idk_response_pct" in metrics:
        report.append(f"\n  {'IDK Response Rate':.<30} {metrics['is_idk_response_pct']:.1f}%")
    
    report.append(f"\nNumber of samples: {metrics.get('n_samples', 0)}")
    report.append("=" * 60)
    
    return "\n".join(report)


def compare_metrics(rag_metrics: Dict, no_rag_metrics: Dict) -> str:
    """Generate comparison report between RAG and No-RAG metrics"""
    report = [
        f"\n{'='*60}",
        "RAG vs No-RAG Quality Comparison",
        f"{'='*60}\n",
        f"{'Metric':<25} {'RAG':>10} {'No-RAG':>10} {'Δ':>10}",
        "-" * 60
    ]
    
    metrics_to_compare = [
        ("Exact Match", "exact_match_mean"),
        ("F1 Score", "f1_score_mean"),
        ("ROUGE-L", "rouge_l_mean"),
        ("BLEU", "bleu_mean")
    ]
    
    for name, key in metrics_to_compare:
        rag_val = rag_metrics.get(key, 0)
        no_rag_val = no_rag_metrics.get(key, 0)
        delta = rag_val - no_rag_val
        delta_pct = (delta / no_rag_val * 100) if no_rag_val > 0 else 0
        
        report.append(f"{name:<25} {rag_val:>10.4f} {no_rag_val:>10.4f} {delta:>+9.4f} ({delta_pct:>+6.1f}%)")
    
    report.append("=" * 60)
    
    return "\n".join(report)
