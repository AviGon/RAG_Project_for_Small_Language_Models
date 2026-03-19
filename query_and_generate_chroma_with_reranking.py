import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

COLLECTION_NAME = "handbook"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_K = 20
FINAL_K = 5

MAX_NEW_TOKENS = 200

RELEVANCE_THRESHOLD = -5.0

client = chromadb.PersistentClient(path="./chroma_db")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

print(f"Collection loaded with {collection.count()} documents")

reranker = CrossEncoder(RERANKER_MODEL)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def retrieve_context(query, initial_k=INITIAL_K, final_k=FINAL_K):
    # Stage 1: Get candidates from ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=initial_k
    )
    
    candidate_chunks = results['documents'][0]
    
    # Stage 2: Rerank
    pairs = [[query, chunk] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(pairs)
    ranked_indices = np.argsort(rerank_scores)[::-1]
    
    top_chunks = [candidate_chunks[i] for i in ranked_indices[:final_k]]
    top_scores = [rerank_scores[i] for i in ranked_indices[:final_k]]
    
    return top_chunks, top_scores

def build_prompt(query, contexts):
    context_text = "\n\n".join(contexts)
    
    prompt = f"""Use the context below to answer the question. If the answer is not in the context, say "I don't have information about this in the documents."

Context:
{context_text}

Question: {query}

Answer:"""
    
    return prompt

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the answer part
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return answer

if __name__ == "__main__":
    print("="*60)
    print("ChromaDB RAG with Reranking (Simple Version)")
    print("="*60)
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Question: ")
        
        if query.lower() in ["exit", "quit"]:
            break
        
        # Retrieve
        print("\nRetrieving and reranking...")
        contexts, scores = retrieve_context(query)
        
        best_score = scores[0] if scores else -999
        
        if best_score < RELEVANCE_THRESHOLD:
            print(f"\nNo relevant information found (best score: {best_score:.2f})")
            print("The question doesn't seem to relate to the document content.\n")
            
            print("="*60)
            print("ANSWER:")
            print("="*60)
            print("I don't have information about this in the documents.")
            print("This question is outside the scope of the handbook.")
            print("\n" + "="*60 + "\n")
            continue
        
        # Show what was retrieved
        print("\nTop chunks:")
        for i, (chunk, score) in enumerate(zip(contexts, scores), 1):
            print(f"\n[{i}] Score: {score:.3f}")
            print(chunk[:150] + "...")
        
        # Generate answer
        prompt = build_prompt(query, contexts)
        print("\nGenerating answer\n")
        answer = generate_answer(prompt)
        
        # Show answer
        print("="*60)
        print("ANSWER:")
        print("="*60)
        print(answer)
        print("\n" + "="*60 + "\n")