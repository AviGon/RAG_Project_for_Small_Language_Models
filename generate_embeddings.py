from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from tqdm import tqdm

CHUNKS_FILE = "chunks.pkl"
EMBEDDINGS_FILE = "embeddings.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


if __name__ == "__main__":
    print("Loading chunks...")
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings)

    print(f"Embedding shape: {embeddings.shape}")

    np.save(EMBEDDINGS_FILE, embeddings)

    print("Embeddings saved to embeddings.npy")