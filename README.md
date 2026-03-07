# RAG_Project_for_Small_Language_Models

# File order for FAISS (Both dense and sparse) (4 files + 1 sparse):

Step 1: Extract and chunk from PDF
python extract_and_chunk.py
Output: chunks.pkl

Step 2: Generate embeddings
python generate_embeddings.py
Output: embeddings.npy

Step 3: Create FAISS index
python store_faiss_index.py
Output: faiss_index.bin, metadata.pkl

Step 4: Create BM25 index (SPARSE)
python sparse_bm25.py
Output: bm25_index.pkl

Step 5: Query with dense retrieval (FAISS)
python query_and_generate.py

OR Step 5: Query with sparse retrieval (BM25)
python sparse_bm25.py search



# File order for Chroma db based (Both dense and sparse) (3 files + 1 sparse):

Step 1: Extract and chunk from PDF
python extract_and_chunk.py
Output: chunks.pkl

Step 2: Setup ChromaDB (embeddings + index)
python setup_chromadb.py
Output: chroma_db/ folder

Step 3: Create BM25 index (SPARSE)
python sparse_bm25.py
Output: bm25_index.pkl

Step 4: Query with dense retrieval (ChromaDB)
python query_and_generate_chroma.py

OR Step 4: Query with sparse retrieval (BM25)
python sparse_bm25.py search

Model: microsoft/Phi-3-mini-4k-instruct

Embedding model: BAAI/bge-small-en-v1.5

FAISS file: faiss_index.bin
Chromadb file: chromadb/

Context stored: chunks.pkl

Embeddings file: embeddings.npy

Data used: https://oerpolicy.eu/wp-content/uploads/sites/4/2017/07/Open-Education-Handbook.pdf? (Open education handbook)
