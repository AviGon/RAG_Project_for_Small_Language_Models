import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

COLLECTION_NAME = "handbook"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TOP_K = 5
MAX_NEW_TOKENS = 300

client = chromadb.PersistentClient(path="./chroma_db")

# Set up the same embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL_NAME
)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

print(f"Collection loaded with {collection.count()} documents")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def retrieve_context(query, k=TOP_K):
    # ChromaDB does embedding + search in one step!
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results['documents'][0]

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

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
            print(f"Chunk {i+1}")
            print(chunk[:500])  # show first 500 chars
            print("\n")

        # Step 2: Build prompt
        prompt = build_prompt(user_query, contexts)

        print("\nConstructed Prompt:\n")
        print(prompt[:1500])  # show preview
        print("\n")

        # Step 3: Generate answer
        print("\nGenerating answer...\n")
        answer = generate_answer(prompt)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")