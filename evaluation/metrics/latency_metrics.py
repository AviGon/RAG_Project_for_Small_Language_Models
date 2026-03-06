"""
Latency profiling utilities for RAG pipeline evaluation.
Tracks timing for each component: query encoding, retrieval, prompt construction, and generation.
"""

import time
import statistics
from typing import Dict, List, Callable
from dataclasses import dataclass, field
import json


@dataclass
class LatencyProfile:
    """Stores timing information for different pipeline stages"""
    query_encoding_time: float = 0.0
    retrieval_time: float = 0.0
    prompt_construction_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "query_encoding_ms": round(self.query_encoding_time * 1000, 2),
            "retrieval_ms": round(self.retrieval_time * 1000, 2),
            "prompt_construction_ms": round(self.prompt_construction_time * 1000, 2),
            "generation_ms": round(self.generation_time * 1000, 2),
            "total_ms": round(self.total_time * 1000, 2)
        }
    
    def get_percentages(self) -> Dict:
        """Calculate percentage breakdown of each component"""
        if self.total_time == 0:
            return {}
        
        return {
            "query_encoding_pct": round((self.query_encoding_time / self.total_time) * 100, 1),
            "retrieval_pct": round((self.retrieval_time / self.total_time) * 100, 1),
            "prompt_construction_pct": round((self.prompt_construction_time / self.total_time) * 100, 1),
            "generation_pct": round((self.generation_time / self.total_time) * 100, 1)
        }


@dataclass
class LatencyStats:
    """Aggregate statistics across multiple runs"""
    means: LatencyProfile = field(default_factory=LatencyProfile)
    stds: LatencyProfile = field(default_factory=LatencyProfile)
    mins: LatencyProfile = field(default_factory=LatencyProfile)
    maxs: LatencyProfile = field(default_factory=LatencyProfile)
    n_samples: int = 0


class LatencyTracker:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.elapsed_time = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed_time = time.perf_counter() - self.start_time
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed_time


def compute_latency_stats(profiles: List[LatencyProfile]) -> LatencyStats:
    """Compute aggregate statistics from multiple latency profiles"""
    if not profiles:
        return LatencyStats()
    
    # Extract lists for each metric
    query_times = [p.query_encoding_time for p in profiles]
    retrieval_times = [p.retrieval_time for p in profiles]
    prompt_times = [p.prompt_construction_time for p in profiles]
    generation_times = [p.generation_time for p in profiles]
    total_times = [p.total_time for p in profiles]
    
    stats = LatencyStats(
        means=LatencyProfile(
            query_encoding_time=statistics.mean(query_times),
            retrieval_time=statistics.mean(retrieval_times),
            prompt_construction_time=statistics.mean(prompt_times),
            generation_time=statistics.mean(generation_times),
            total_time=statistics.mean(total_times)
        ),
        stds=LatencyProfile(
            query_encoding_time=statistics.stdev(query_times) if len(query_times) > 1 else 0.0,
            retrieval_time=statistics.stdev(retrieval_times) if len(retrieval_times) > 1 else 0.0,
            prompt_construction_time=statistics.stdev(prompt_times) if len(prompt_times) > 1 else 0.0,
            generation_time=statistics.stdev(generation_times) if len(generation_times) > 1 else 0.0,
            total_time=statistics.stdev(total_times) if len(total_times) > 1 else 0.0
        ),
        mins=LatencyProfile(
            query_encoding_time=min(query_times),
            retrieval_time=min(retrieval_times),
            prompt_construction_time=min(prompt_times),
            generation_time=min(generation_times),
            total_time=min(total_times)
        ),
        maxs=LatencyProfile(
            query_encoding_time=max(query_times),
            retrieval_time=max(retrieval_times),
            prompt_construction_time=max(prompt_times),
            generation_time=max(generation_times),
            total_time=max(total_times)
        ),
        n_samples=len(profiles)
    )
    
    return stats


def format_latency_report(stats: LatencyStats, mode: str = "RAG") -> str:
    """Generate a formatted text report of latency statistics"""
    means_dict = stats.means.to_dict()
    percentages = stats.means.get_percentages()
    stds_dict = stats.stds.to_dict()
    
    report = [
        f"\n{'='*60}",
        f"{mode} Latency Profile (n={stats.n_samples} runs)",
        f"{'='*60}\n",
        "Component Breakdown:",
        "-" * 60
    ]
    
    if mode == "RAG":
        components = [
            ("Query Encoding", "query_encoding_ms", "query_encoding_pct"),
            ("FAISS Retrieval", "retrieval_ms", "retrieval_pct"),
            ("Prompt Construction", "prompt_construction_ms", "prompt_construction_pct"),
            ("LLM Generation", "generation_ms", "generation_pct")
        ]
    else:
        components = [
            ("LLM Generation", "generation_ms", "generation_pct")
        ]
    
    for name, time_key, pct_key in components:
        mean_time = means_dict.get(time_key, 0)
        std_time = stds_dict.get(time_key, 0)
        pct = percentages.get(pct_key, 0)
        
        if mean_time > 0:
            report.append(f"  {name:.<30} {mean_time:>7.1f}ms ± {std_time:>5.1f}ms  ({pct:>5.1f}%)")
    
    report.extend([
        "-" * 60,
        f"  {'TOTAL':.<30} {means_dict['total_ms']:>7.1f}ms ± {stds_dict['total_ms']:>5.1f}ms  (100.0%)",
        "=" * 60
    ])
    
    return "\n".join(report)


def save_latency_results(stats: LatencyStats, filepath: str, mode: str = "RAG"):
    """Save latency statistics to JSON file"""
    result = {
        "mode": mode,
        "n_samples": stats.n_samples,
        "statistics": {
            "mean": stats.means.to_dict(),
            "std": stats.stds.to_dict(),
            "min": stats.mins.to_dict(),
            "max": stats.maxs.to_dict()
        },
        "percentages": stats.means.get_percentages()
    }
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)


def compare_latencies(rag_stats: LatencyStats, no_rag_stats: LatencyStats) -> str:
    """Generate comparison report between RAG and No-RAG latencies"""
    rag_total = rag_stats.means.total_time * 1000
    no_rag_total = no_rag_stats.means.total_time * 1000
    overhead = rag_total - no_rag_total
    overhead_pct = (overhead / no_rag_total) * 100 if no_rag_total > 0 else 0
    
    report = [
        f"\n{'='*60}",
        "RAG vs No-RAG Latency Comparison",
        f"{'='*60}\n",
        f"RAG Total:        {rag_total:>7.1f}ms",
        f"No-RAG Total:     {no_rag_total:>7.1f}ms",
        f"Overhead:         {overhead:>7.1f}ms ({overhead_pct:>5.1f}% increase)",
        "\nBreakdown of RAG Overhead:",
        "-" * 60,
        f"  Query Encoding:  {rag_stats.means.query_encoding_time * 1000:>7.1f}ms",
        f"  FAISS Search:    {rag_stats.means.retrieval_time * 1000:>7.1f}ms",
        f"  Prompt Build:    {rag_stats.means.prompt_construction_time * 1000:>7.1f}ms",
        "=" * 60
    ]
    
    return "\n".join(report)
