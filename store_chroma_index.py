import pickle
from pathlib import Path

import chromadb
import numpy as np


EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.pkl"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "chunks_full"


def main():
    print("Loading chunks and embeddings...")
    with open(CHUNKS_FILE, "rb") as handle:
        chunks = pickle.load(handle)
    embeddings = np.load(EMBEDDINGS_FILE)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
        )

    print(f"Loaded {len(chunks)} chunks, dim={embeddings.shape[1]}")

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    print("Adding vectors to ChromaDB...")
    batch_size = 512
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        ids = [str(i) for i in range(start, end)]
        docs = chunks[start:end]
        embs = embeddings[start:end].tolist()
        metas = [{"idx": i} for i in range(start, end)]
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    print(f"✅ Stored {collection.count()} vectors in {CHROMA_DIR}/{COLLECTION_NAME}")


if __name__ == "__main__":
    main()
