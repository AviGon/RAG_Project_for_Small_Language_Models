"""
Query system with Cross-Encoder Reranking.
Two-stage retrieval:
  1. Dense retrieval: Get 20 candidates (FAISS)
  2. Reranking: Rerank to best 5 (Cross-Encoder)

NO TRAINING NEEDED - Works immediately!
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==============================
# CONFIG
# ==============================
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# ============================================
# RERANKING SETTINGS
# ============================================
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_K = 20  # Retrieve more candidates from FAISS
FINAL_K = 5     # Rerank to best 5
# ============================================

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


# ============================================
# Load Cross-Encoder for Reranking
# ============================================
print("Loading cross-encoder for reranking...")
reranker = CrossEncoder(RERANKER_MODEL)
print("✓ Reranker loaded")
# ============================================


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
# Retrieval with Reranking
# ==============================
def retrieve_context(query, initial_k=INITIAL_K, final_k=FINAL_K):
    """
    Two-stage retrieval:
    1. Dense retrieval: Get initial_k candidates from FAISS
    2. Reranking: Rerank using cross-encoder to get best final_k
    """
    
    # Stage 1: Dense retrieval
    print(f"\n[Stage 1] Dense retrieval: Getting top {initial_k} candidates...")
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding)
    
    scores, indices = index.search(query_embedding, initial_k)
    
    candidate_chunks = [chunks[i] for i in indices[0]]
    
    # Stage 2: Reranking
    print(f"[Stage 2] Reranking: Selecting best {final_k}...")
    
    # Create query-document pairs for cross-encoder
    pairs = [[query, chunk] for chunk in candidate_chunks]
    
    # Get reranking scores
    rerank_scores = reranker.predict(pairs)
    
    # Sort by reranking score (descending)
    ranked_indices = np.argsort(rerank_scores)[::-1]
    
    # Get top final_k
    top_chunks = [candidate_chunks[i] for i in ranked_indices[:final_k]]
    top_scores = [rerank_scores[i] for i in ranked_indices[:final_k]]
    
    print(f"✓ Reranking complete\n")
    
    return top_chunks, top_scores


# ==============================
# Prompt Builder
# ==============================
def build_prompt(query, contexts):
    # Number the contexts for better reference
    context_text = "\n\n".join([
        f"[Document {i+1}]\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])

    # IMPROVED PROMPT - More strict instructions
    prompt = f"""<|system|>
You are a precise assistant that answers questions using ONLY the provided documents.

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in the documents below
2. If the answer requires information NOT in the documents, you MUST respond: "I cannot answer this based on the provided documents."
3. Do NOT use any outside knowledge or make assumptions
4. Quote or reference document numbers when answering
5. If you're uncertain, say so - never guess

REMEMBER: It's better to say "I don't know" than to provide incorrect information.<|end|>
<|user|>
Documents:
{context_text}

Question: {query}

Answer based ONLY on the documents above:<|end|>
<|assistant|>
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
    print("\n" + "="*60)
    print("RAG System with Cross-Encoder Reranking")
    print(f"Stage 1: Retrieve top {INITIAL_K} candidates")
    print(f"Stage 2: Rerank to best {FINAL_K}")
    print("="*60)
    print("Type 'exit' to quit.\n")

    while True:
        user_query = input("Enter your question:\n> ")

        if user_query.lower() in ["exit", "quit"]:
            break

        # Retrieve with reranking
        contexts, scores = retrieve_context(user_query)

        print("Top Retrieved Chunks (After Reranking):\n")
        for i, (chunk, score) in enumerate(zip(contexts, scores), 1):
            print(f"--- Chunk {i} (Rerank Score: {score:.4f}) ---")
            print(chunk[:300])
            print()

        # Build prompt
        prompt = build_prompt(user_query, contexts)

        print("\nConstructed Prompt:\n")
        print(prompt[:1500])
        print("\n")

        # Optional manual override
        use_custom = input("Do you want to modify the prompt? (y/n): ")

        if use_custom.lower() == "y":
            print("Enter your custom prompt below:")
            prompt = input()

        # Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(prompt)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")