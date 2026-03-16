#!/usr/bin/env python3
"""
Check the status of the latency-focused RAG pipeline.
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
    chroma_index_exists = check_file("store_chroma_index.py", "Chroma indexer")
    query_exists = check_file("query_and_generate.py", "Interactive query system")
    
    # Check generated files
    print("\n📦 Generated Files (RAG Setup):")
    chunks_exists = check_file("chunks.pkl", "Text chunks")
    embeddings_exists = check_file("embeddings.npy", "Embeddings")
    faiss_exists = check_file("faiss_index.bin", "FAISS index")
    metadata_exists = check_file("metadata.pkl", "Metadata")
    
    # Check latency evaluation components
    print("\n🎯 Latency Evaluation System:")
    eval_dataset = check_file("evaluation/dataset/evaluation_dataset.json", "Evaluation dataset")
    eval_script = check_file("evaluation/experiments/latency_eval.py", "Latency evaluator")
    compare_script = check_file("evaluation/experiments/compare_latency.py", "CPU/GPU comparison")
    viz_script = check_file("evaluation/visualizations/plot_latency.py", "Latency visualization")
    
    # Check if evaluation has been run
    print("\n📊 Evaluation Results:")
    summary_exists = check_file("evaluation/results/latency", "Latency results folder")
    
    if summary_exists:
        cpu_exists = check_file("evaluation/results/latency", "Per-model latency outputs")
        viz_exists = check_file("evaluation/visualizations/latency", "Latency visualizations")
    
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
        print("     python store_chroma_index.py")
        
        # Show what's missing
        if not chunks_exists:
            print("\n   ✗ Need to run: extract_and_chunk.py")
        if not embeddings_exists:
            print("   ✗ Need to run: generate_embeddings.py")
        if not faiss_exists:
            print("   ✗ Need to run: store_faiss_index.py")
    
    elif not eval_complete:
        print("\n✓ RAG setup complete!")
        print("\n📋 Next steps - Run latency evaluation:")
        print("   python evaluation/experiments/compare_latency.py --visualize")
        print("\n   Or run full pipeline:")
        print("   python run_full_pipeline.py --skip-setup")
    
    else:
        print("\n✅ Everything complete!")
        print("\n📊 Your latency results are ready:")
        print("   - Per-model JSON: evaluation/results/latency/<model_slug>/")
        print("   - Multi-model summary: evaluation/results/latency/multi_model_summary.json")
        print("   - Plots: evaluation/visualizations/latency/<model_slug>/")
        
        print("\n💡 What's next:")
        print("   - Add more questions: python evaluation/dataset/add_questions.py")
        print("   - Re-run with more models: python run_full_pipeline.py --skip-setup --llm-models ...")
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
