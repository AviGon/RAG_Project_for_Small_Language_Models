import faiss
import numpy as np
import pickle

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.pkl"

FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"


if __name__ == "__main__":
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_FILE)

    dimension = embeddings.shape[1]
    print(f"Vector dimension: {dimension}")

    print("Creating FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(dimension)

    print("Adding embeddings to index...")
    index.add(embeddings)

    print(f"Total vectors indexed: {index.ntotal}")

    print("Saving FAISS index...")
    faiss.write_index(index, FAISS_INDEX_FILE)

    print("Saving metadata mapping...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print("Done and stored!")