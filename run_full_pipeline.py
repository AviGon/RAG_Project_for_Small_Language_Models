#!/usr/bin/env python3
"""
Master pipeline script to set up and run RAG system evaluation from scratch.

Order of operations:
1. Extract and chunk PDF
2. Generate embeddings
3. Build FAISS index
4. Run evaluation (RAG vs No-RAG)
5. Generate visualizations
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_command(cmd, description, critical=True):
    """Run a shell command and handle errors"""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        if critical:
            print(f"\n❌ ERROR: {description} failed!")
            print(f"Exit code: {result.returncode}")
            sys.exit(1)
        else:
            print(f"\n⚠️  WARNING: {description} failed but continuing...")
    else:
        print(f"\n✓ {description} completed successfully!")
    
    return result.returncode == 0


def check_prerequisites():
    """Check if required files exist"""
    print("\n" + "="*70)
    print("Checking Prerequisites")
    print("="*70)
    
    required_files = [
        "data/handbook.pdf",
        "extract_and_chunk.py",
        "generate_embeddings.py",
        "store_faiss_index.py",
        "evaluation/experiments/run_baseline_eval.py"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"❌ Missing: {file}")
        else:
            print(f"✓ Found: {file}")
    
    if missing:
        print(f"\n❌ ERROR: Missing required files!")
        print("Please ensure you have:")
        print("  - data/handbook.pdf (source PDF document)")
        print("  - All pipeline scripts in project root")
        return False
    
    print("\n✓ All prerequisites satisfied!")
    return True


def check_outputs_exist():
    """Check if pipeline outputs already exist"""
    outputs = {
        "chunks.pkl": "Text chunks",
        "embeddings.npy": "Embeddings",
        "faiss_index.bin": "FAISS index",
        "metadata.pkl": "Metadata"
    }
    
    existing = []
    for file, desc in outputs.items():
        if Path(file).exists():
            existing.append((file, desc))
    
    return existing


def main():
    parser = argparse.ArgumentParser(
        description="Run complete RAG pipeline from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_full_pipeline.py
  
  # Skip setup, only run evaluation
  python run_full_pipeline.py --skip-setup
  
  # Run setup only, skip evaluation
  python run_full_pipeline.py --setup-only
  
  # Force rebuild even if outputs exist
  python run_full_pipeline.py --force
        """
    )
    
    parser.add_argument("--skip-setup", action="store_true",
                        help="Skip data processing and indexing (assumes already done)")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only run setup steps, skip evaluation")
    parser.add_argument("--skip-visualization", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation studies")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if outputs exist")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA for evaluation")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("RAG SYSTEM FULL PIPELINE")
    print("="*70)
    print("\nThis script will:")
    print("  1. Extract and chunk PDF document")
    print("  2. Generate embeddings for all chunks")
    print("  3. Build FAISS index for retrieval")
    print("  4. Run RAG vs No-RAG evaluation")
    print("  5. Generate visualizations")
    print("  6. (Optional) Run ablation studies")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check if outputs already exist
    if not args.force and not args.skip_setup:
        existing = check_outputs_exist()
        if existing:
            print("\n" + "="*70)
            print("EXISTING OUTPUTS DETECTED")
            print("="*70)
            for file, desc in existing:
                print(f"  ✓ {file} ({desc})")
            
            print("\nOptions:")
            print("  - Use --force to rebuild everything")
            print("  - Use --skip-setup to use existing outputs and run evaluation only")
            
            response = input("\nRebuild from scratch? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping setup phase. Using existing outputs.")
                args.skip_setup = True
    
    success = True
    
    # SETUP PHASE
    if not args.skip_setup:
        print("\n" + "="*70)
        print("PHASE 1: DATA PREPARATION & INDEXING")
        print("="*70)
        
        # Step 1: Extract and chunk
        success = run_command(
            "python3 extract_and_chunk.py",
            "Extract text from PDF and create chunks"
        )
        if not success:
            sys.exit(1)
        
        # Step 2: Generate embeddings
        success = run_command(
            "python3 generate_embeddings.py",
            "Generate embeddings for all chunks"
        )
        if not success:
            sys.exit(1)
        
        # Step 3: Build FAISS index
        success = run_command(
            "python3 store_faiss_index.py",
            "Build and store FAISS index"
        )
        if not success:
            sys.exit(1)
        
        print("\n" + "="*70)
        print("✓ SETUP PHASE COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print("  ✓ chunks.pkl - Text chunks")
        print("  ✓ embeddings.npy - Vector embeddings")
        print("  ✓ faiss_index.bin - FAISS search index")
        print("  ✓ metadata.pkl - Chunk metadata")
    
    if args.setup_only:
        print("\n✓ Setup complete! You can now run evaluations.")
        return
    
    # EVALUATION PHASE
    print("\n" + "="*70)
    print("PHASE 2: EVALUATION")
    print("="*70)
    
    # Step 4: Run baseline evaluation
    cuda_flag = "--no-cuda" if args.no_cuda else ""
    success = run_command(
        f"python3 evaluation/experiments/run_baseline_eval.py {cuda_flag}",
        "Run RAG vs No-RAG evaluation"
    )
    
    if not success:
        print("\n⚠️  Evaluation failed. Check error messages above.")
        print("You may need to:")
        print("  - Install required packages: pip install -r requirements.txt")
        print("  - Use --no-cuda flag if GPU issues occur")
        sys.exit(1)
    
    # VISUALIZATION PHASE
    if not args.skip_visualization:
        print("\n" + "="*70)
        print("PHASE 3: VISUALIZATION")
        print("="*70)
        
        success = run_command(
            "python3 evaluation/visualizations/plot_results.py",
            "Generate visualization plots",
            critical=False
        )
        
        if success:
            print("\n✓ Visualizations saved to: evaluation/visualizations/")
    
    # ABLATION STUDIES (OPTIONAL)
    if not args.skip_ablation:
        print("\n" + "="*70)
        print("PHASE 4: ABLATION STUDIES (Optional)")
        print("="*70)
        
        response = input("\nRun ablation studies? This will take 20-40 minutes. (y/N): ").strip().lower()
        
        if response == 'y':
            success = run_command(
                f"python3 evaluation/experiments/ablation_studies.py --study top_k {cuda_flag}",
                "Run top-k ablation study",
                critical=False
            )
            
            if success:
                # Regenerate visualizations to include ablation plots
                run_command(
                    "python3 evaluation/visualizations/plot_results.py",
                    "Update visualizations with ablation results",
                    critical=False
                )
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print("\n📊 Check your results:")
    print("  - Summary metrics: evaluation/results/summary.json")
    print("  - Per-question details: evaluation/results/generation_quality/")
    print("  - Latency profiles: evaluation/results/latency_profiles/")
    print("  - Visualizations: evaluation/visualizations/*.png")
    print("\n📖 Documentation:")
    print("  - Quick start: evaluation/QUICKSTART.md")
    print("  - Detailed guide: evaluation/README.md")
    print("\n✨ Next steps:")
    print("  - Review the generated plots")
    print("  - Check summary.json for aggregate metrics")
    print("  - Add more questions: python evaluation/dataset/add_questions.py")
    print("  - Experiment with different parameters in the evaluation scripts")
    

if __name__ == "__main__":
    main()
