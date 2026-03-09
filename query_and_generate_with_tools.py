"""
Tool-augmented RAG pipeline for small language models.

This script demonstrates a lightweight "function calling" style flow:
1) Route user query to a tool (calculator, datetime, corpus stats) when appropriate
2) Otherwise use document retrieval tool (FAISS top-k search)
3) Compose final response with the SLM
"""

import ast
import datetime as dt
import operator as op
import pickle
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================
# CONFIG
# ==============================
FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TOP_K = 5
MAX_NEW_TOKENS = 220


# ==============================
# Models + Index
# ==============================
print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)

print("Loading metadata...")
with open(METADATA_FILE, "rb") as f:
    chunks = pickle.load(f)

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)


# ==============================
# Tooling
# ==============================
@dataclass
class ToolDecision:
    name: str
    args: Dict[str, str]
    reason: str


def retrieve_context(query: str, k: int = TOP_K) -> List[str]:
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding)
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


def corpus_stats() -> str:
    num_chunks = len(chunks)
    avg_len = sum(len(c) for c in chunks) / max(num_chunks, 1)
    return (
        f"Total chunks: {num_chunks}\n"
        f"Average chunk length (characters): {avg_len:.1f}\n"
        f"FAISS vectors indexed: {index.ntotal}"
    )


_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARY_OPS = {ast.USub: op.neg, ast.UAdd: op.pos}


def _safe_eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # py<3.8 compatibility
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_safe_eval_ast(node.operand))
    raise ValueError("Unsupported expression.")


def calculator(expression: str) -> str:
    sanitized = expression.replace("^", "**").strip()
    tree = ast.parse(sanitized, mode="eval")
    result = _safe_eval_ast(tree.body)
    return f"{expression.strip()} = {result:g}"


def current_datetime() -> str:
    now = dt.datetime.now()
    return now.strftime("%A, %B %d, %Y %H:%M:%S")


def parse_explicit_tool(query: str) -> ToolDecision:
    """
    Explicit commands:
      /tool calc <expression>
      /tool time
      /tool stats
      /tool rag <question>
    """
    q = query.strip()
    if not q.lower().startswith("/tool "):
        return ToolDecision("auto", {}, "No explicit tool command.")

    remainder = q[6:].strip()
    if remainder.lower().startswith("calc "):
        return ToolDecision(
            "calculator",
            {"expression": remainder[5:].strip()},
            "Explicit calculator request.",
        )
    if remainder.lower() == "time":
        return ToolDecision("current_datetime", {}, "Explicit time request.")
    if remainder.lower() == "stats":
        return ToolDecision("corpus_stats", {}, "Explicit corpus stats request.")
    if remainder.lower().startswith("rag "):
        return ToolDecision(
            "rag_search",
            {"question": remainder[4:].strip()},
            "Explicit RAG request.",
        )
    return ToolDecision("auto", {}, "Unknown explicit tool command; falling back to auto.")


def select_tool(query: str) -> ToolDecision:
    explicit = parse_explicit_tool(query)
    if explicit.name != "auto":
        return explicit

    q = query.lower()

    if any(kw in q for kw in ["current time", "what time", "today date", "current date"]):
        return ToolDecision("current_datetime", {}, "Time/date query detected.")

    if any(kw in q for kw in ["how many chunks", "index size", "corpus stats", "database stats"]):
        return ToolDecision("corpus_stats", {}, "Corpus/index metadata query detected.")

    math_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.\^%]+$")
    if q.startswith(("calculate ", "compute ", "solve ")):
        expr = query.split(" ", 1)[1].strip()
        return ToolDecision("calculator", {"expression": expr}, "Math command detected.")
    if math_pattern.match(q.strip()) and any(ch.isdigit() for ch in q):
        return ToolDecision("calculator", {"expression": query.strip()}, "Math expression detected.")

    return ToolDecision("rag_search", {"question": query}, "Default to document retrieval.")


def run_tool(decision: ToolDecision) -> Tuple[str, List[str], str]:
    """
    Returns:
      tool_output: textual output from non-RAG tools
      contexts: retrieved contexts for RAG
      effective_question: question to answer
    """
    if decision.name == "current_datetime":
        return current_datetime(), [], "What is the current date and time?"
    if decision.name == "corpus_stats":
        return corpus_stats(), [], "Summarize corpus/index stats."
    if decision.name == "calculator":
        expr = decision.args.get("expression", "").strip()
        if not expr:
            return "No expression provided.", [], "Compute the requested expression."
        try:
            return calculator(expr), [], f"Compute: {expr}"
        except Exception as exc:
            return f"Calculation error: {exc}", [], f"Compute: {expr}"
    if decision.name == "rag_search":
        question = decision.args.get("question", "").strip()
        contexts = retrieve_context(question, TOP_K)
        return "", contexts, question

    # Fallback: RAG
    contexts = retrieve_context(decision.args.get("question", ""), TOP_K)
    return "", contexts, decision.args.get("question", "")


def build_final_prompt(
    user_query: str, tool_decision: ToolDecision, tool_output: str, contexts: List[str], effective_question: str
) -> str:
    if tool_decision.name == "rag_search":
        context_text = "\n\n".join(contexts)
        return f"""You are an assistant answering questions strictly from the provided context.
If the answer is not present, say: "I don't know based on the provided document."

Context:
{context_text}

Question:
{effective_question}

Answer:"""

    return f"""You are an assistant. Use ONLY the tool output below to answer the user.
If tool output indicates an error or missing data, say that clearly.

User Query:
{user_query}

Tool Used:
{tool_decision.name}

Tool Output:
{tool_output}

Final Answer:"""


def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Final Answer:" in decoded:
        return decoded.split("Final Answer:")[-1].strip()
    if "Answer:" in decoded:
        return decoded.split("Answer:")[-1].strip()
    return decoded


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("Tool-Augmented RAG (FAISS + Phi-3)")
    print("Tools: rag_search (default), calculator, current_datetime, corpus_stats")
    print("Optional explicit commands:")
    print("  /tool calc 21*19")
    print("  /tool time")
    print("  /tool stats")
    print("  /tool rag <your question>")
    print("=" * 72 + "\n")

    while True:
        user_query = input("Enter your question (or 'exit'):\n> ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break

        decision = select_tool(user_query)
        tool_output, contexts, effective_question = run_tool(decision)

        print(f"\n[Tool Router] Selected: {decision.name}")
        print(f"[Reason] {decision.reason}")

        if contexts:
            print("\nTop retrieved chunks:")
            for i, chunk in enumerate(contexts, 1):
                print(f"--- Chunk {i} ---")
                print(chunk[:250] + "...\n")
        elif tool_output:
            print("\nTool output:")
            print(tool_output)

        prompt = build_final_prompt(user_query, decision, tool_output, contexts, effective_question)
        answer = generate_answer(prompt)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "=" * 72 + "\n")
