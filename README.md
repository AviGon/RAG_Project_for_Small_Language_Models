# RAG_Project_for_Small_Language_Models

## RAG Pipeline

The RAG pipeline consists of the following stages:

- Document Extraction & Chunking  
- Embedding Generation  
- Vector Index Creation  
- Sparse Index Creation (BM25)  
- Query & Response Generation  

---

## File order for FAISS (Both dense and sparse, and reranking) (4 files + 1 sparse)

### Step 1: Extract and chunk from PDF

```bash
python extract_and_chunk.py
```

Output:
```
chunks.pkl
```

---

### Step 2: Generate embeddings

```bash
python generate_embeddings.py
```

Output:
```
embeddings.npy
```

---

### Step 3: Create FAISS index

```bash
python store_faiss_index.py
```

Output:
```
faiss_index.bin
metadata.pkl
```

---

### Step 4: Create BM25 index (SPARSE)

```bash
python sparse_bm25.py
```

Output:
```
bm25_index.pkl
```

---

### Step 5: Query with dense retrieval (FAISS)

```bash
python query_and_generate.py
```

OR

### Step 5: Query with sparse retrieval (BM25)

```bash
python sparse_bm25.py search
```

OR

### Step 5: Query with dense retrieval and have reranking optimization

```bash
python query_and_generate_with_reranking.py
```

OR

### Step 5: Query with tool-augmented RAG (FAISS + Tool Router)

```bash
python query_and_generate_with_tools.py
```

Supports:
- `rag_search` (default): retrieves top-k chunks and answers from context
- `calculator`: safe arithmetic evaluation
- `current_datetime`: local date/time lookup
- `corpus_stats`: chunk/index stats
- `web_fetch`: fetch a web page by URL and answer from its content
- `web_search`: search the web and answer from top results

Optional explicit commands:
- `/tool calc 21*19`
- `/tool time`
- `/tool stats`
- `/tool rag <your question>`
- `/tool web <url> <question>`
- `/tool websearch <query>`

---

## File order for Chroma db based (Both dense and sparse, and reranking) (3 files + 1 sparse)

### Step 1: Extract and chunk from PDF

```bash
python extract_and_chunk.py
```

Output:
```
chunks.pkl
```

---

### Step 2: Setup ChromaDB (embeddings + index)

```bash
python setup_chromadb.py
```

Output:
```
chroma_db/ folder
```

---

### Step 3: Create BM25 index (SPARSE)

```bash
python sparse_bm25.py
```

Output:
```
bm25_index.pkl
```

---

### Step 4: Query with dense retrieval (ChromaDB)

```bash
python query_and_generate_chromadb.py
```

OR

### Step 4: Query with sparse retrieval (BM25)

```bash
python sparse_bm25.py search
```

OR

### Step 4: Query with dense retrieval with reranking optimization

```bash
python query_and_generate_chroma_with_reranking.py
```

---

## Models and Storage

```
LLM Model: microsoft/Phi-3-mini-4k-instruct
Embedding Model: BAAI/bge-small-en-v1.5

Vector Storage:
    FAISS Index      : faiss_index.bin
    ChromaDB Folder  : chroma_db/

Data Files:
    Context Chunks   : chunks.pkl
    Embeddings       : embeddings.npy
```

---

## Files

```
Context stored: chunks.pkl

Embeddings file: embeddings.npy
```

---

## Data used

1) Open Education Handbook (50 pages)

https://oerpolicy.eu/wp-content/uploads/sites/4/2017/07/Open-Education-Handbook.pdf

2) US Department of Housing and Urban affairs (1800 pages)

https://www.hud.gov/sites/dfiles/OCHCO/documents/40001-hsgh-update15-052024.pdf

3) Science and Technology Peer Review Council (250 pages)

https://www.epa.gov/sites/default/files/2020-08/documents/epa_peer_review_handbook_4th_edition.pdf

4) Lease Implementation Guidelines (13 pages)

https://files.fasab.gov/pdffiles/handbook_tr_22.pdf
To change dataset, in the file: extract_and_chunk.py, change PDF_PATH, CHUNK_SIZE and OVERLAP. Try to keep OVERLAP 20% of CHUNK_SIZE
