#!/usr/bin/env python3
"""
Check the status of the RAG pipeline - what's been completed and what's missing.
"""

from pathlib import Path
import json


def check_file(filepath, description):
    """Check if a file exists and print status"""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    color = "\033[92m" if exists else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {description:.<50} {filepath}")
    return exists


def get_file_size(filepath):
    """Get human-readable file size"""
    path = Path(filepath)
    if not path.exists():
        return "N/A"
    
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def main():
    print("\n" + "="*70)
    print("RAG PIPELINE STATUS CHECK")
    print("="*70)
    
    # Check source data
    print("\n📄 Source Data:")
    pdf_exists = check_file("data/handbook.pdf", "PDF document")
    
    # Check pipeline scripts
    print("\n🔧 Pipeline Scripts:")
    extract_exists = check_file("extract_and_chunk.py", "Extract & chunk script")
    embed_exists = check_file("generate_embeddings.py", "Embedding generator")
    index_exists = check_file("store_faiss_index.py", "FAISS indexer")
    query_exists = check_file("query_and_generate.py", "Query system")
    
    # Check generated files
    print("\n📦 Generated Files (RAG Setup):")
    chunks_exists = check_file("chunks.pkl", "Text chunks")
    embeddings_exists = check_file("embeddings.npy", "Embeddings")
    faiss_exists = check_file("faiss_index.bin", "FAISS index")
    metadata_exists = check_file("metadata.pkl", "Metadata")
    
    # Check evaluation components
    print("\n🎯 Evaluation System:")
    eval_dataset = check_file("evaluation/dataset/evaluation_dataset.json", "Evaluation dataset")
    eval_script = check_file("evaluation/experiments/run_baseline_eval.py", "Evaluation script")
    viz_script = check_file("evaluation/visualizations/plot_results.py", "Visualization script")
    
    # Check if evaluation has been run
    print("\n📊 Evaluation Results:")
    summary_exists = check_file("evaluation/results/summary.json", "Summary results")
    
    if summary_exists:
        latency_exists = check_file("evaluation/results/latency_profiles/rag_latency.json", "Latency profiles")
        viz_exists = check_file("evaluation/visualizations/latency_breakdown.png", "Visualizations")
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    setup_complete = all([chunks_exists, embeddings_exists, faiss_exists, metadata_exists])
    eval_complete = summary_exists
    
    if not pdf_exists:
        print("\n❌ Missing PDF document!")
        print("   → Place your PDF in: data/handbook.pdf")
    
    elif not setup_complete:
        print("\n⚠️  RAG setup incomplete. Need to run pipeline setup.")
        print("\n📋 Next steps:")
        print("   Option 1 (Automated):")
        print("     python run_full_pipeline.py")
        print("\n   Option 2 (Manual):")
        print("     python extract_and_chunk.py")
        print("     python generate_embeddings.py")
        print("     python store_faiss_index.py")
        
        # Show what's missing
        if not chunks_exists:
            print("\n   ✗ Need to run: extract_and_chunk.py")
        if not embeddings_exists:
            print("   ✗ Need to run: generate_embeddings.py")
        if not faiss_exists:
            print("   ✗ Need to run: store_faiss_index.py")
    
    elif not eval_complete:
        print("\n✓ RAG setup complete!")
        print("\n📋 Next steps - Run evaluation:")
        print("   python evaluation/experiments/run_baseline_eval.py")
        print("\n   Or run full pipeline:")
        print("   python run_full_pipeline.py --skip-setup")
    
    else:
        print("\n✅ Everything complete!")
        print("\n📊 Your results are ready:")
        print("   - Summary: evaluation/results/summary.json")
        print("   - Details: evaluation/results/generation_quality/")
        print("   - Plots: evaluation/visualizations/")
        
        # Show summary metrics if available
        try:
            with open("evaluation/results/summary.json", 'r') as f:
                summary = json.load(f)
            
            rag_rouge = summary['overall']['rag'].get('rouge_l_mean', 0)
            no_rag_rouge = summary['overall']['no_rag'].get('rouge_l_mean', 0)
            improvement = ((rag_rouge - no_rag_rouge) / no_rag_rouge * 100) if no_rag_rouge > 0 else 0
            
            print("\n📈 Quick Results:")
            print(f"   RAG ROUGE-L:    {rag_rouge:.4f}")
            print(f"   No-RAG ROUGE-L: {no_rag_rouge:.4f}")
            print(f"   Improvement:    +{improvement:.1f}%")
        except:
            pass
        
        print("\n💡 What's next:")
        print("   - Add more questions: python evaluation/dataset/add_questions.py")
        print("   - Run ablation studies: python evaluation/experiments/ablation_studies.py")
        print("   - Interactive testing: python query_and_generate.py")
    
    # File sizes
    if chunks_exists or embeddings_exists or faiss_exists:
        print("\n" + "="*70)
        print("FILE SIZES")
        print("="*70)
        if chunks_exists:
            print(f"  chunks.pkl:       {get_file_size('chunks.pkl')}")
        if embeddings_exists:
            print(f"  embeddings.npy:   {get_file_size('embeddings.npy')}")
        if faiss_exists:
            print(f"  faiss_index.bin:  {get_file_size('faiss_index.bin')}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
