import chromadb
from chromadb.utils import embedding_functions
import pickle

CHUNKS_FILE = "chunks.pkl"
COLLECTION_NAME = "handbook"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

if __name__ == "__main__":
    print("Initializing ChromaDB...")
    
    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Set up embedding function (same model you were using!)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    print("Loading chunks")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"Adding {len(chunks)} chunks to ChromaDB")
    print("This will generate embeddings automatically")
    
    # Add documents - ChromaDB handles embedding generation!
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"chunk_index": i} for i in range(len(chunks))]
    )
    
    print(f"Collection now has {collection.count()} documents")