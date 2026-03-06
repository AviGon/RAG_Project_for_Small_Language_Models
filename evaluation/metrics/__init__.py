"""
Evaluation metrics package for RAG system.
"""

from .latency_metrics import (
    LatencyProfile,
    LatencyTracker,
    compute_latency_stats,
    format_latency_report,
    compare_latencies
)

from .generation_metrics import (
    compute_all_metrics,
    aggregate_metrics,
    format_metrics_report,
    compare_metrics,
    compute_exact_match,
    compute_f1_score,
    compute_rouge_l,
    compute_bleu_score,
    check_context_grounding,
    detect_idk_response
)

__all__ = [
    'LatencyProfile',
    'LatencyTracker',
    'compute_latency_stats',
    'format_latency_report',
    'compare_latencies',
    'compute_all_metrics',
    'aggregate_metrics',
    'format_metrics_report',
    'compare_metrics',
    'compute_exact_match',
    'compute_f1_score',
    'compute_rouge_l',
    'compute_bleu_score',
    'check_context_grounding',
    'detect_idk_response'
]
