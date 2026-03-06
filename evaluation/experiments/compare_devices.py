"""
Device comparison script: Compare latency between GPU and CPU.
Runs evaluation on both devices and generates comparative analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def run_evaluation_on_device(device: str, dataset: str, output_dir: str):
    """Run evaluation on specified device"""
    print(f"\n{'='*70}")
    print(f"Running Evaluation on {device.upper()}")
    print(f"{'='*70}\n")
    
    cuda_flag = "" if device == "gpu" else "--no-cuda"
    device_output = f"{output_dir}/{device}"
    
    cmd = f"python3 evaluation/experiments/run_baseline_eval.py {cuda_flag} --dataset {dataset} --output {device_output}"
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Evaluation failed on {device.upper()}")
        return False
    
    print(f"\n✓ {device.upper()} evaluation complete!")
    return True


def load_device_results(device: str, output_dir: str):
    """Load evaluation results for a device"""
    summary_path = Path(f"{output_dir}/{device}/summary.json")
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def compare_latencies(gpu_results, cpu_results, output_dir):
    """Generate detailed latency comparison"""
    print("\n" + "="*70)
    print("DEVICE LATENCY COMPARISON")
    print("="*70)
    
    # Extract latency data
    gpu_rag = gpu_results['latency']['rag']
    cpu_rag = cpu_results['latency']['rag']
    gpu_no_rag = gpu_results['latency']['no_rag']
    cpu_no_rag = cpu_results['latency']['no_rag']
    
    # Component-wise comparison for RAG
    print("\n🔍 RAG Mode - Component Breakdown:")
    print(f"{'Component':<25} {'GPU (ms)':>12} {'CPU (ms)':>12} {'Speedup':>12}")
    print("-" * 70)
    
    components = [
        ('Query Encoding', 'query_encoding_ms'),
        ('FAISS Retrieval', 'retrieval_ms'),
        ('Prompt Construction', 'prompt_construction_ms'),
        ('LLM Generation', 'generation_ms'),
        ('TOTAL (End-to-End)', 'total_ms')
    ]
    
    speedups = {}
    for name, key in components:
        gpu_val = gpu_rag.get(key, 0)
        cpu_val = cpu_rag.get(key, 0)
        speedup = cpu_val / gpu_val if gpu_val > 0 else 0
        speedups[name] = speedup
        
        print(f"{name:<25} {gpu_val:>12.1f} {cpu_val:>12.1f} {speedup:>11.2f}x")
    
    # No-RAG comparison
    print("\n🔍 No-RAG Mode - Generation Time:")
    print(f"{'Mode':<25} {'GPU (ms)':>12} {'CPU (ms)':>12} {'Speedup':>12}")
    print("-" * 70)
    
    gpu_gen = gpu_no_rag.get('generation_ms', 0)
    cpu_gen = cpu_no_rag.get('generation_ms', 0)
    gen_speedup = cpu_gen / gpu_gen if gpu_gen > 0 else 0
    
    print(f"{'Generation Only':<25} {gpu_gen:>12.1f} {cpu_gen:>12.1f} {gen_speedup:>11.2f}x")
    
    # Summary
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    total_speedup = speedups['TOTAL (End-to-End)']
    gen_component_speedup = speedups['LLM Generation']
    
    print(f"\n• Overall RAG Speedup (GPU vs CPU): {total_speedup:.2f}x")
    print(f"• LLM Generation Speedup: {gen_component_speedup:.2f}x")
    print(f"• Query Encoding Speedup: {speedups['Query Encoding']:.2f}x")
    print(f"• FAISS Retrieval Speedup: {speedups['FAISS Retrieval']:.2f}x")
    
    # Identify bottleneck
    bottleneck_gpu = max([(k, v) for k, v in gpu_rag.items() if k.endswith('_ms')], key=lambda x: x[1])
    bottleneck_cpu = max([(k, v) for k, v in cpu_rag.items() if k.endswith('_ms')], key=lambda x: x[1])
    
    print(f"\n• Bottleneck on GPU: {bottleneck_gpu[0].replace('_ms', '')} ({bottleneck_gpu[1]:.1f}ms)")
    print(f"• Bottleneck on CPU: {bottleneck_cpu[0].replace('_ms', '')} ({bottleneck_cpu[1]:.1f}ms)")
    
    # Calculate cost/benefit
    gpu_total = gpu_rag['total_ms']
    cpu_total = cpu_rag['total_ms']
    time_saved = cpu_total - gpu_total
    
    print(f"\n• Time Saved per Query (GPU): {time_saved:.1f}ms ({(time_saved/cpu_total)*100:.1f}%)")
    
    queries_per_sec_gpu = 1000 / gpu_total
    queries_per_sec_cpu = 1000 / cpu_total
    
    print(f"• Throughput (GPU): {queries_per_sec_gpu:.2f} queries/sec")
    print(f"• Throughput (CPU): {queries_per_sec_cpu:.2f} queries/sec")
    
    print("="*70)
    
    # Save comparison data
    comparison_data = {
        "gpu_rag": gpu_rag,
        "cpu_rag": cpu_rag,
        "gpu_no_rag": gpu_no_rag,
        "cpu_no_rag": cpu_no_rag,
        "speedups": speedups,
        "metrics": {
            "overall_speedup": total_speedup,
            "generation_speedup": gen_component_speedup,
            "time_saved_per_query_ms": time_saved,
            "gpu_throughput_qps": queries_per_sec_gpu,
            "cpu_throughput_qps": queries_per_sec_cpu
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/device_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    return comparison_data


def plot_device_comparison(comparison_data, output_dir):
    """Generate visualization comparing GPU and CPU"""
    print("\n📊 Generating device comparison visualizations...")
    
    gpu_rag = comparison_data['gpu_rag']
    cpu_rag = comparison_data['cpu_rag']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Component-wise latency comparison (bar chart)
    ax1 = plt.subplot(2, 3, 1)
    components = ['query_encoding_ms', 'retrieval_ms', 'prompt_construction_ms', 'generation_ms']
    component_labels = ['Query\nEncoding', 'FAISS\nRetrieval', 'Prompt\nBuild', 'LLM\nGeneration']
    
    gpu_vals = [gpu_rag.get(c, 0) for c in components]
    cpu_vals = [cpu_rag.get(c, 0) for c in components]
    
    x = np.arange(len(component_labels))
    width = 0.35
    
    ax1.bar(x - width/2, gpu_vals, width, label='GPU', color='#4CAF50')
    ax1.bar(x + width/2, cpu_vals, width, label='CPU', color='#FF9800')
    
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Component-Wise Latency (RAG Mode)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(component_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total e2e latency comparison
    ax2 = plt.subplot(2, 3, 2)
    modes = ['RAG\nGPU', 'RAG\nCPU', 'No-RAG\nGPU', 'No-RAG\nCPU']
    totals = [
        gpu_rag['total_ms'],
        cpu_rag['total_ms'],
        comparison_data['gpu_no_rag']['total_ms'],
        comparison_data['cpu_no_rag']['total_ms']
    ]
    colors = ['#4CAF50', '#FF9800', '#2196F3', '#FFC107']
    
    bars = ax2.bar(modes, totals, color=colors)
    ax2.set_ylabel('Total Latency (ms)')
    ax2.set_title('End-to-End Latency Comparison')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}ms',
                ha='center', va='bottom')
    
    # 3. Speedup factors
    ax3 = plt.subplot(2, 3, 3)
    speedup_components = ['Query\nEncoding', 'FAISS\nRetrieval', 'Prompt\nBuild', 'LLM\nGeneration', 'Total']
    speedup_vals = [
        comparison_data['speedups']['Query Encoding'],
        comparison_data['speedups']['FAISS Retrieval'],
        comparison_data['speedups']['Prompt Construction'],
        comparison_data['speedups']['LLM Generation'],
        comparison_data['speedups']['TOTAL (End-to-End)']
    ]
    
    bars = ax3.barh(speedup_components, speedup_vals, color='#9C27B0')
    ax3.set_xlabel('Speedup Factor (CPU/GPU)')
    ax3.set_title('GPU Speedup by Component')
    ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, speedup_vals)):
        ax3.text(val + 0.1, bar.get_y() + bar.get_height()/2.,
                f'{val:.2f}x',
                ha='left', va='center')
    
    # 4. Percentage breakdown (stacked bar)
    ax4 = plt.subplot(2, 3, 4)
    
    gpu_percentages = [
        (gpu_rag['query_encoding_ms'] / gpu_rag['total_ms']) * 100,
        (gpu_rag['retrieval_ms'] / gpu_rag['total_ms']) * 100,
        (gpu_rag['prompt_construction_ms'] / gpu_rag['total_ms']) * 100,
        (gpu_rag['generation_ms'] / gpu_rag['total_ms']) * 100
    ]
    
    cpu_percentages = [
        (cpu_rag['query_encoding_ms'] / cpu_rag['total_ms']) * 100,
        (cpu_rag['retrieval_ms'] / cpu_rag['total_ms']) * 100,
        (cpu_rag['prompt_construction_ms'] / cpu_rag['total_ms']) * 100,
        (cpu_rag['generation_ms'] / cpu_rag['total_ms']) * 100
    ]
    
    labels = ['Query Encoding', 'Retrieval', 'Prompt', 'Generation']
    colors_stack = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    x_pos = [0, 1]
    bottom_gpu = 0
    bottom_cpu = 0
    
    for i, label in enumerate(labels):
        ax4.bar(0, gpu_percentages[i], bottom=bottom_gpu, label=label if i == 0 else "", 
               color=colors_stack[i], width=0.5)
        bottom_gpu += gpu_percentages[i]
        
        ax4.bar(1, cpu_percentages[i], bottom=bottom_cpu, 
               color=colors_stack[i], width=0.5)
        bottom_cpu += cpu_percentages[i]
    
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['GPU', 'CPU'])
    ax4.set_ylabel('Percentage of Total Time (%)')
    ax4.set_title('Time Distribution by Component')
    ax4.legend(labels, loc='upper right', fontsize=8)
    ax4.set_ylim(0, 100)
    
    # 5. Throughput comparison
    ax5 = plt.subplot(2, 3, 5)
    devices = ['GPU', 'CPU']
    throughputs = [
        comparison_data['metrics']['gpu_throughput_qps'],
        comparison_data['metrics']['cpu_throughput_qps']
    ]
    
    bars = ax5.bar(devices, throughputs, color=['#4CAF50', '#FF9800'])
    ax5.set_ylabel('Queries per Second')
    ax5.set_title('System Throughput')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}\nq/s',
                ha='center', va='bottom')
    
    # 6. Time saved visualization
    ax6 = plt.subplot(2, 3, 6)
    time_saved = comparison_data['metrics']['time_saved_per_query_ms']
    
    data = [cpu_rag['total_ms'] - time_saved, time_saved]
    labels_pie = ['GPU Time', 'Time Saved\nby GPU']
    colors_pie = ['#4CAF50', '#CDDC39']
    
    wedges, texts, autotexts = ax6.pie(data, labels=labels_pie, colors=colors_pie,
                                         autopct='%1.1f%%', startangle=90)
    ax6.set_title(f'Time Savings per Query\n({time_saved:.1f}ms saved)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/device_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/device_comparison.png")
    plt.close()


def compare_accuracy(gpu_results, cpu_results):
    """Compare generation quality metrics between devices"""
    print("\n" + "="*70)
    print("GENERATION QUALITY COMPARISON (GPU vs CPU)")
    print("="*70)
    
    gpu_metrics = gpu_results['overall']['rag']
    cpu_metrics = cpu_results['overall']['rag']
    
    print(f"\n{'Metric':<20} {'GPU':>15} {'CPU':>15} {'Difference':>15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('ROUGE-L', 'rouge_l_mean'),
        ('F1 Score', 'f1_score_mean'),
        ('Exact Match', 'exact_match_mean'),
        ('BLEU', 'bleu_mean')
    ]
    
    for name, key in metrics_to_compare:
        gpu_val = gpu_metrics.get(key, 0)
        cpu_val = cpu_metrics.get(key, 0)
        diff = gpu_val - cpu_val
        
        print(f"{name:<20} {gpu_val:>15.4f} {cpu_val:>15.4f} {diff:>+15.4f}")
    
    print("\n💡 Note: Quality metrics should be nearly identical between devices.")
    print("   Any significant differences may indicate numerical precision issues.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAG system performance between GPU and CPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on both devices and compare
  python3 evaluation/experiments/compare_devices.py
  
  # Use existing results (skip re-running)
  python3 evaluation/experiments/compare_devices.py --use-existing
        """
    )
    
    parser.add_argument("--dataset", type=str, 
                        default="evaluation/dataset/evaluation_dataset.json",
                        help="Path to evaluation dataset")
    parser.add_argument("--output", type=str, 
                        default="evaluation/results/device_comparison",
                        help="Output directory")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing results instead of re-running")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="Skip GPU evaluation (only run CPU)")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU evaluation (only run GPU)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DEVICE COMPARISON EVALUATION")
    print("="*70)
    print("\nThis script compares RAG performance between GPU and CPU")
    print("Evaluates: latency, throughput, and accuracy")
    
    # Run evaluations if not using existing
    if not args.use_existing:
        if not args.skip_gpu:
            success = run_evaluation_on_device("gpu", args.dataset, args.output)
            if not success:
                print("\n⚠️  GPU evaluation failed. Continuing with CPU only...")
                args.skip_gpu = True
        
        if not args.skip_cpu:
            success = run_evaluation_on_device("cpu", args.dataset, args.output)
            if not success:
                print("\n❌ CPU evaluation failed.")
                sys.exit(1)
    
    # Load results
    print("\n" + "="*70)
    print("Loading Results...")
    print("="*70)
    
    gpu_results = load_device_results("gpu", args.output) if not args.skip_gpu else None
    cpu_results = load_device_results("cpu", args.output) if not args.skip_cpu else None
    
    if gpu_results is None and cpu_results is None:
        print("\n❌ No results found. Run without --use-existing flag first.")
        sys.exit(1)
    
    if gpu_results is None or cpu_results is None:
        print("\n⚠️  Only one device result available. Need both for comparison.")
        if gpu_results is None:
            print("Missing: GPU results")
        if cpu_results is None:
            print("Missing: CPU results")
        sys.exit(1)
    
    print("✓ Loaded GPU results")
    print("✓ Loaded CPU results")
    
    # Generate comparisons
    comparison_data = compare_latencies(gpu_results, cpu_results, args.output)
    compare_accuracy(gpu_results, cpu_results)
    plot_device_comparison(comparison_data, args.output)
    
    print("\n" + "="*70)
    print("🎉 DEVICE COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n📊 Results saved to: {args.output}/")
    print(f"  - Comparison data: {args.output}/device_comparison.json")
    print(f"  - Visualization: {args.output}/device_comparison.png")
    print(f"  - GPU details: {args.output}/gpu/")
    print(f"  - CPU details: {args.output}/cpu/")


if __name__ == "__main__":
    main()
