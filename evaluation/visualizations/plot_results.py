"""
Visualization utilities for RAG evaluation results.
Creates plots for latency breakdown, metric comparisons, and ablation studies.
"""

import json
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_latency_breakdown(rag_latency: Dict, no_rag_latency: Dict, output_path: str):
    """Create pie chart showing latency breakdown for RAG"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RAG latency breakdown
    rag_components = {
        'Query Encoding': rag_latency.get('query_encoding_ms', 0),
        'FAISS Search': rag_latency.get('retrieval_ms', 0),
        'Prompt Build': rag_latency.get('prompt_construction_ms', 0),
        'LLM Generation': rag_latency.get('generation_ms', 0)
    }
    
    # Filter out zero values
    rag_components = {k: v for k, v in rag_components.items() if v > 0}
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    ax1.pie(rag_components.values(), labels=rag_components.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title(f'RAG Pipeline Latency Breakdown\nTotal: {sum(rag_components.values()):.1f}ms')
    
    # Comparison bar chart
    categories = ['RAG Total', 'No-RAG Total', 'RAG Overhead']
    values = [
        rag_latency['total_ms'],
        no_rag_latency['total_ms'],
        rag_latency['total_ms'] - no_rag_latency['total_ms']
    ]
    
    bars = ax2.bar(categories, values, color=['#ff9999', '#66b3ff', '#ffcc99'])
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('RAG vs No-RAG Latency Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}ms',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved latency breakdown to {output_path}")
    plt.close()


def plot_metrics_comparison(rag_metrics: Dict, no_rag_metrics: Dict, output_path: str):
    """Create bar chart comparing RAG vs No-RAG metrics"""
    metrics_to_plot = [
        ('exact_match_mean', 'Exact Match'),
        ('f1_score_mean', 'F1 Score'),
        ('rouge_l_mean', 'ROUGE-L'),
        ('bleu_mean', 'BLEU')
    ]
    
    metric_names = [name for _, name in metrics_to_plot]
    rag_values = [rag_metrics.get(key, 0) for key, _ in metrics_to_plot]
    no_rag_values = [no_rag_metrics.get(key, 0) for key, _ in metrics_to_plot]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, rag_values, width, label='RAG', color='#ff9999')
    bars2 = ax.bar(x + width/2, no_rag_values, width, label='No-RAG', color='#66b3ff')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('RAG vs No-RAG: Generation Quality Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics comparison to {output_path}")
    plt.close()


def plot_in_vs_out_document(summary: Dict, output_path: str):
    """Compare performance on in-document vs out-of-document questions"""
    categories = ['In-Document', 'Out-of-Document']
    
    # Extract metrics
    rag_rouge_in = summary['in_document']['rag'].get('rouge_l_mean', 0)
    rag_rouge_out = summary['out_of_document']['rag'].get('rouge_l_mean', 0)
    no_rag_rouge_in = summary['in_document']['no_rag'].get('rouge_l_mean', 0)
    no_rag_rouge_out = summary['out_of_document']['no_rag'].get('rouge_l_mean', 0)
    
    rag_f1_in = summary['in_document']['rag'].get('f1_score_mean', 0)
    rag_f1_out = summary['out_of_document']['rag'].get('f1_score_mean', 0)
    no_rag_f1_in = summary['in_document']['no_rag'].get('f1_score_mean', 0)
    no_rag_f1_out = summary['out_of_document']['no_rag'].get('f1_score_mean', 0)
    
    x = np.arange(len(categories))
    width = 0.2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROUGE-L comparison
    ax1.bar(x - width*1.5, [rag_rouge_in, rag_rouge_out], width, 
            label='RAG', color='#ff9999')
    ax1.bar(x - width*0.5, [no_rag_rouge_in, no_rag_rouge_out], width, 
            label='No-RAG', color='#66b3ff')
    ax1.set_ylabel('ROUGE-L Score')
    ax1.set_title('ROUGE-L: In-Document vs Out-of-Document')
    ax1.set_xticks(x - width)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 0.8)
    
    # F1 comparison
    ax2.bar(x - width*1.5, [rag_f1_in, rag_f1_out], width, 
            label='RAG', color='#ff9999')
    ax2.bar(x - width*0.5, [no_rag_f1_in, no_rag_f1_out], width, 
            label='No-RAG', color='#66b3ff')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1: In-Document vs Out-of-Document')
    ax2.set_xticks(x - width)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved in/out document comparison to {output_path}")
    plt.close()


def plot_top_k_ablation(ablation_data: Dict, output_path: str):
    """Plot results of top-k ablation study"""
    k_values = []
    rouge_scores = []
    f1_scores = []
    latencies = []
    grounding_scores = []
    
    for key in sorted(ablation_data.keys(), key=lambda x: int(x.split('_')[1])):
        k = int(key.split('_')[1])
        k_values.append(k)
        
        metrics = ablation_data[key]['metrics']
        rouge_scores.append(metrics.get('rouge_l_mean', 0))
        f1_scores.append(metrics.get('f1_score_mean', 0))
        grounding_scores.append(metrics.get('grounding_score_mean', 0))
        
        latency = ablation_data[key]['latency']
        latencies.append(latency['total_ms'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # ROUGE-L vs k
    ax1.plot(k_values, rouge_scores, marker='o', linewidth=2, markersize=8, color='#ff9999')
    ax1.set_xlabel('Top-k Value')
    ax1.set_ylabel('ROUGE-L Score')
    ax1.set_title('ROUGE-L vs Top-k')
    ax1.grid(alpha=0.3)
    ax1.set_xticks(k_values)
    
    # F1 vs k
    ax2.plot(k_values, f1_scores, marker='s', linewidth=2, markersize=8, color='#66b3ff')
    ax2.set_xlabel('Top-k Value')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Top-k')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(k_values)
    
    # Latency vs k
    ax3.plot(k_values, latencies, marker='^', linewidth=2, markersize=8, color='#99ff99')
    ax3.set_xlabel('Top-k Value')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Latency vs Top-k')
    ax3.grid(alpha=0.3)
    ax3.set_xticks(k_values)
    
    # Grounding score vs k
    ax4.plot(k_values, grounding_scores, marker='D', linewidth=2, markersize=8, color='#ffcc99')
    ax4.set_xlabel('Top-k Value')
    ax4.set_ylabel('Grounding Score')
    ax4.set_title('Context Grounding vs Top-k')
    ax4.grid(alpha=0.3)
    ax4.set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved top-k ablation plot to {output_path}")
    plt.close()


def plot_accuracy_vs_latency_tradeoff(rag_metrics: Dict, no_rag_metrics: Dict,
                                       rag_latency: Dict, no_rag_latency: Dict,
                                       output_path: str):
    """Plot accuracy-latency tradeoff"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # RAG point
    rag_accuracy = rag_metrics.get('rouge_l_mean', 0)
    rag_lat = rag_latency['total_ms']
    
    # No-RAG point
    no_rag_accuracy = no_rag_metrics.get('rouge_l_mean', 0)
    no_rag_lat = no_rag_latency['total_ms']
    
    ax.scatter(rag_lat, rag_accuracy, s=300, c='#ff9999', marker='o', 
               label='RAG', edgecolors='black', linewidths=2)
    ax.scatter(no_rag_lat, no_rag_accuracy, s=300, c='#66b3ff', marker='s', 
               label='No-RAG', edgecolors='black', linewidths=2)
    
    # Add annotations
    ax.annotate('RAG', (rag_lat, rag_accuracy), 
                xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold')
    ax.annotate('No-RAG', (no_rag_lat, no_rag_accuracy), 
                xytext=(10, -20), textcoords='offset points', fontsize=12, fontweight='bold')
    
    # Arrow showing improvement
    ax.annotate('', xy=(rag_lat, rag_accuracy), xytext=(no_rag_lat, no_rag_accuracy),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', linestyle='--'))
    
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('ROUGE-L Score', fontsize=12)
    ax.set_title('Accuracy vs Latency Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add quadrant labels
    mid_x = (rag_lat + no_rag_lat) / 2
    mid_y = (rag_accuracy + no_rag_accuracy) / 2
    
    ax.axhline(y=mid_y, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=mid_x, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved accuracy-latency tradeoff to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots for evaluation results")
    parser.add_argument("--results-dir", type=str, default="evaluation/results",
                        help="Directory containing evaluation results")
    parser.add_argument("--output-dir", type=str, default="evaluation/visualizations",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary results
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"Error: Summary file not found at {summary_path}")
        print("Please run the baseline evaluation first.")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("Generating visualizations...")
    print("="*60)
    
    # Plot latency breakdown
    plot_latency_breakdown(
        summary['latency']['rag'],
        summary['latency']['no_rag'],
        str(output_dir / "latency_breakdown.png")
    )
    
    # Plot metrics comparison
    plot_metrics_comparison(
        summary['overall']['rag'],
        summary['overall']['no_rag'],
        str(output_dir / "metrics_comparison.png")
    )
    
    # Plot in-document vs out-of-document
    plot_in_vs_out_document(
        summary,
        str(output_dir / "in_vs_out_document.png")
    )
    
    # Plot accuracy-latency tradeoff
    plot_accuracy_vs_latency_tradeoff(
        summary['overall']['rag'],
        summary['overall']['no_rag'],
        summary['latency']['rag'],
        summary['latency']['no_rag'],
        str(output_dir / "accuracy_latency_tradeoff.png")
    )
    
    # Plot ablation studies if available
    ablation_path = results_dir / "ablation" / "top_k_ablation.json"
    if ablation_path.exists():
        with open(ablation_path, 'r') as f:
            ablation_data = json.load(f)
        
        plot_top_k_ablation(
            ablation_data,
            str(output_dir / "top_k_ablation.png")
        )
    
    print("="*60)
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
