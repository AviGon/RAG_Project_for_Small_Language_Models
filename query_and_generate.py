import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==============================
# CONFIG
# ==============================
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TOP_K = 5
MAX_NEW_TOKENS = 300


# ==============================
# Load FAISS + Metadata
# ==============================
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)

print("Loading metadata...")
with open(METADATA_FILE, "rb") as f:
    chunks = pickle.load(f)


# ==============================
# Load Embedding Model
# ==============================
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# ==============================
# Load Phi-3
# ==============================
print("Loading Phi-3 model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


# ==============================
# Retrieval
# ==============================
def retrieve_context(query, k=TOP_K):
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding)

    scores, indices = index.search(query_embedding, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]

    return retrieved_chunks


# ==============================
# Prompt Builder
# ==============================
def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an assistant answering questions strictly using the provided context.
If the answer is not present in the context, respond with:
"I don't know based on the provided document."

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt.strip()


# ==============================
# Generation
# ==============================
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==============================
# MAIN LOOP
# ==============================
if __name__ == "__main__":
    print("\nRAG System Ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question:\n> ")

        if user_query.lower() in ["exit", "quit"]:
            break

        # Step 1: Retrieve context
        contexts = retrieve_context(user_query)

        print("\nRetrieved Context Chunks:\n")
        for i, chunk in enumerate(contexts):
            print(f"--- Chunk {i+1} ---")
            print(chunk[:500])  # show first 500 chars
            print("\n")

        # Step 2: Build prompt
        prompt = build_prompt(user_query, contexts)

        print("\nConstructed Prompt:\n")
        print(prompt[:1500])  # show preview
        print("\n")

        # Optional manual override
        use_custom = input("Do you want to modify the prompt? (y/n): ")

        if use_custom.lower() == "y":
            print("Enter your custom prompt below:")
            prompt = input()

        # Step 3: Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(prompt)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")