from rank_bm25 import BM25Okapi
import pickle
import numpy as np

CHUNKS_FILE = "chunks.pkl"
BM25_INDEX_FILE = "bm25_index.pkl"


def tokenize(text):
    """Simple tokenization - split by whitespace and lowercase"""
    return text.lower().split()


def create_bm25_index():
    """Create and save BM25 index from chunks"""
    print("Loading chunks")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    
    print("Tokenizing chunks")
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    
    print("Creating BM25 index")
    bm25 = BM25Okapi(tokenized_chunks)
    
    print("Saving BM25 index")
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump({
            'bm25': bm25,
            'chunks': chunks
        }, f)
    
    print(f"BM25 index created with {len(chunks)} chunks")
    print(f"Saved to {BM25_INDEX_FILE}")


def search_bm25(query, top_k=5):
    """Search using BM25"""
    print("Loading BM25 index")
    with open(BM25_INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    
    bm25 = data['bm25']
    chunks = data['chunks']
    
    # Tokenize query
    tokenized_query = tokenize(query)
    
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Return chunks and scores
    results = [(chunks[i], scores[i]) for i in top_indices]
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        # Search mode
        print("\nBM25 Sparse Retrieval - Search Mode")
        print("="*60)
        
        while True:
            query = input("\nEnter your question (or 'exit' to quit):\n> ")
            
            if query.lower() in ['exit', 'quit']:
                break
            
            print("\nSearching with BM25\n")
            results = search_bm25(query, top_k=5)
            
            for i, (chunk, score) in enumerate(results, 1):
                print(f"--- Result {i} (Score: {score:.4f}) ---")
                print(chunk[:300])
                print()
    else:
        # Index creation mode
        print("Creating BM25 index")
        create_bm25_index()
        print("\nTo search, run: python sparse_bm25.py search")