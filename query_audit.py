import json
import pickle
import re
from collections import Counter


def tokenize(text: str):
    return [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if len(w) > 2]


def coverage(answer_tokens, chunk_token_set):
    if not answer_tokens:
        return 0.0
    answer_set = set(answer_tokens)
    return len(answer_set & chunk_token_set) / len(answer_set)


with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("/Users/bhavya/Desktop/evaluation/dataset/evaluation_dataset.json", "r") as f:
    data = json.load(f)

questions = data["questions"] if isinstance(data, dict) else data
chunk_token_sets = [set(tokenize(c)) for c in chunks]

rows = []
for item in questions:
    qid = item.get("id")
    question = item.get("question", "")
    answer = item.get("reference_answer", "")

    ans_tokens = tokenize(answer)
    best_score = 0.0
    best_idx = -1
    for idx, token_set in enumerate(chunk_token_sets):
        score = coverage(ans_tokens, token_set)
        if score > best_score:
            best_score = score
            best_idx = idx

    answer_lower = answer.lower().strip()
    exact_substring_found = bool(answer_lower) and any(answer_lower in c.lower() for c in chunks)

    rows.append(
        {
            "id": qid,
            "question": question,
            "answer_terms": len(set(ans_tokens)),
            "best_chunk": best_idx,
            "token_coverage": best_score,
            "exact_substring_found": exact_substring_found,
        }
    )

rows_sorted = sorted(rows, key=lambda r: r["token_coverage"])

buckets = Counter()
for row in rows:
    s = row["token_coverage"]
    if s >= 0.75:
        buckets["high (>=0.75)"] += 1
    elif s >= 0.50:
        buckets["medium (0.50-0.74)"] += 1
    elif s >= 0.30:
        buckets["low (0.30-0.49)"] += 1
    else:
        buckets["very_low (<0.30)"] += 1

print("=== Query Dataset Audit ===")
print(f"Total questions: {len(rows)}")
print(f"Total chunks: {len(chunks)}")
print(f"Coverage buckets: {dict(buckets)}")
print()
print("Lowest coverage questions (potentially weak/noisy references):")
for row in rows_sorted[:10]:
    print(
        f"id={row['id']:>2} | coverage={row['token_coverage']:.3f} | "
        f"answer_terms={row['answer_terms']:>2} | exact_match={row['exact_substring_found']} | "
        f"q={row['question'][:90]}"
    )

exact_count = sum(1 for r in rows if r["exact_substring_found"])
print()
print(f"Exact reference-answer substring appears in corpus for {exact_count}/{len(rows)} questions")
