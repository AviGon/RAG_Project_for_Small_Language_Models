import ast
import datetime as dt
import operator as op
import pickle
import re
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import faiss
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

FAISS_INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

TOP_K = 5
MAX_NEW_TOKENS = 220

WEB_SEARCH_RESULTS = 5
WEB_FETCH_TOP_N = 3
WEB_MAX_CHARS = 120000
WEB_CHUNK_SIZE = 900
WEB_CHUNK_OVERLAP = 150
WEB_TIMEOUT_SECS = 12
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"

index = faiss.read_index(FAISS_INDEX_FILE)

with open(METADATA_FILE, "rb") as f:
    chunks = pickle.load(f)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

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


def chunk_text(text: str, chunk_size: int = WEB_CHUNK_SIZE, overlap: int = WEB_CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    chunks_out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks_out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks_out


def retrieve_context_from_text(query: str, text: str, k: int = TOP_K) -> List[str]:
    raw_chunks = chunk_text(text)
    if not raw_chunks:
        return []
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    chunk_embeddings = embed_model.encode(raw_chunks, normalize_embeddings=True)
    scores = np.dot(chunk_embeddings, query_embedding[0])
    top_idx = np.argsort(scores)[::-1][:k]
    return [raw_chunks[i] for i in top_idx]


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
    if isinstance(node, ast.Num):
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


def _clean_ddg_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("/l/"):
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return urllib.parse.unquote(qs["uddg"][0])
    return url


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_url_text(url: str) -> Tuple[str, str, str]:
    if not url:
        return "", "", "Missing URL."
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        url = "https://" + url

    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=WEB_TIMEOUT_SECS)
        resp.raise_for_status()
    except Exception as exc:
        return "", "", f"Request failed: {exc}"

    content_type = resp.headers.get("Content-Type", "")
    if "text" not in content_type and "html" not in content_type:
        return "", "", f"Unsupported content type: {content_type}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled"
    text = soup.get_text(" ", strip=True)
    text = _normalize_whitespace(text)
    if not text:
        return title, "", "No extractable text found."
    if len(text) > WEB_MAX_CHARS:
        text = text[:WEB_MAX_CHARS]
    return title, text, ""


def duckduckgo_search(query: str, num_results: int = WEB_SEARCH_RESULTS) -> List[Dict[str, str]]:
    if not query:
        return []
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=WEB_TIMEOUT_SECS,
        )
        resp.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    seen = set()
    for result in soup.select("div.result"):
        link = result.select_one("a.result__a")
        if not link:
            continue
        url = _clean_ddg_url(link.get("href", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        title = link.get_text(" ", strip=True)
        snippet_el = result.select_one("a.result__snippet") or result.select_one("div.result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= num_results:
            break
    return results


def build_source_contexts(query: str, title: str, url: str, text: str, k: int = TOP_K) -> List[str]:
    chunks_out = retrieve_context_from_text(query, text, k)
    if not chunks_out:
        return []
    source_header = f"Source: {title} ({url})\n"
    return [source_header + chunk for chunk in chunks_out]


def parse_explicit_tool(query: str) -> ToolDecision:
    """
    Explicit commands:
      /tool calc <expression>
      /tool time
      /tool stats
      /tool rag <question>
      /tool web <url> <question>
      /tool websearch <query>
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
    if remainder.lower().startswith("websearch "):
        return ToolDecision(
            "web_search",
            {"query": remainder[10:].strip()},
            "Explicit web search request.",
        )
    if remainder.lower().startswith("web "):
        args = remainder[4:].strip()
        if not args:
            return ToolDecision("web_fetch", {"url": "", "question": ""}, "Missing URL.")
        parts = args.split()
        url = parts[0]
        question = " ".join(parts[1:]).strip()
        return ToolDecision(
            "web_fetch",
            {"url": url, "question": question},
            "Explicit web fetch request.",
        )
    return ToolDecision("auto", {}, "Unknown explicit tool command; falling back to auto.")


def select_tool(query: str) -> ToolDecision:
    explicit = parse_explicit_tool(query)
    if explicit.name != "auto":
        return explicit

    q = query.lower()

    url_match = re.search(r"https?://\S+", query)
    if url_match:
        url = url_match.group(0).rstrip(").,]")
        question = query.replace(url, "").strip()
        return ToolDecision(
            "web_fetch",
            {"url": url, "question": question},
            "URL detected in query.",
        )

    if any(kw in q for kw in ["from the web", "search the web", "on the internet", "online", "latest", "news"]):
        return ToolDecision("web_search", {"query": query}, "Web search intent detected.")

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
    if decision.name == "web_fetch":
        url = decision.args.get("url", "").strip()
        question = decision.args.get("question", "").strip() or "Summarize the page."
        title, text, error = fetch_url_text(url)
        if error:
            return f"Web fetch error: {error}", [], question
        contexts = build_source_contexts(question, title, url, text, TOP_K)
        if not contexts:
            return "No relevant content extracted from the page.", [], question
        return f"Fetched: {title} ({url})", contexts, question
    if decision.name == "web_search":
        query = decision.args.get("query", "").strip()
        results = duckduckgo_search(query, WEB_SEARCH_RESULTS)
        if not results:
            return "No search results found.", [], query
        contexts = []
        used_sources = []
        for result in results[:WEB_FETCH_TOP_N]:
            title, text, error = fetch_url_text(result["url"])
            if error:
                continue
            used_sources.append(f"- {title} ({result['url']})")
            contexts.extend(build_source_contexts(query, title, result["url"], text, TOP_K))
        if not contexts:
            return "No usable text extracted from web results.", [], query
        tool_output = "Sources:\n" + "\n".join(used_sources)
        return tool_output, contexts, query

    # Fallback: RAG
    contexts = retrieve_context(decision.args.get("question", ""), TOP_K)
    return "", contexts, decision.args.get("question", "")


def build_final_prompt(
    user_query: str, tool_decision: ToolDecision, tool_output: str, contexts: List[str], effective_question: str
) -> str:
    if tool_decision.name in {"rag_search", "web_fetch", "web_search"}:
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
    print("Tools: rag_search (default), calculator, current_datetime, corpus_stats, web_fetch, web_search")
    print("Optional explicit commands:")
    print("  /tool calc 21*19")
    print("  /tool time")
    print("  /tool stats")
    print("  /tool rag <your question>")
    print("  /tool web <url> <question>")
    print("  /tool websearch <query>")
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
        if tool_output:
            print("\nTool output:")
            print(tool_output)

        prompt = build_final_prompt(user_query, decision, tool_output, contexts, effective_question)
        answer = generate_answer(prompt)

        print("\nFinal Answer:\n")
        print(answer)
        print("\n" + "=" * 72 + "\n")
