"""Extension (runnable): retrieval-augmented generation over this lab's glossary.

    .venv/bin/python scripts/extensions/rag_glossary.py

* Corpus: docs/GLOSSARY.md, one chunk per bullet.
* Retriever: BM25 written in ~20 lines of Python (no extra packages).
* Generator: LFM2.5-350M, greedy, with and without the top-3 retrieved chunks.
* Check: each question has required keywords; an answer counts as correct if it
  contains all of them (a crude but deterministic check; read the raw answers too).
"""
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401

from lab.common import OUTPUTS, ROOT, write_json  # noqa: E402
from lab.models import load_text_model  # noqa: E402
from lab.text_eval import chat_prompt, generate  # noqa: E402

QUESTIONS = [  # (question, keywords that a correct answer must contain)
    ("In this lab's glossary, how many query heads and KV heads does LFM2.5-350M have?", ["16", "8"]),
    ("What label value makes PyTorch ignore a position in the loss?", ["-100"]),
    ("Into how many pixels is a tile split for LFM2.5-VL tiling?", ["512"]),
    ("Why does LoRA's B matrix start at zero?", ["original"]),
    ("What does an equal-reward group contribute to the GRPO gradient?", ["zero"]),
    ("Which padding side is used for generation?", ["left"]),
    ("What vocabulary size does LFM2.5 have?", ["65,536"]),
    ("What does pixel unshuffle do to the image-token count?", ["4"]),
]


def tokenize(s):
    return re.findall(r"[a-z0-9\-]+", s.lower())


def glossary_chunks(text):
    """One chunk per bullet, INCLUDING its wrapped continuation lines."""
    chunks, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("- **"):
            if cur:
                chunks.append(cur)
            cur = ln[2:].strip()
        elif cur is not None and ln.startswith("  ") and ln.strip():
            cur += " " + ln.strip()
        else:
            if cur:
                chunks.append(cur)
            cur = None
    if cur:
        chunks.append(cur)
    return chunks


chunks = glossary_chunks((ROOT / "docs/GLOSSARY.md").read_text())
docs = [tokenize(c) for c in chunks]
df = Counter(t for d in docs for t in set(d))
avgdl = sum(map(len, docs)) / len(docs)


def bm25(query, k1=1.5, b=0.75):
    q = tokenize(query)
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return sorted(range(len(docs)), key=lambda i: -scores[i])[:3]


model, tok = load_text_model("text_instruct")
model.eval()
plain, rag, retrieved = [], [], []
for q, _ in QUESTIONS:
    plain.append(chat_prompt(tok, [{"role": "user", "content": q + " Answer in one sentence."}]))
    top = [chunks[i] for i in bm25(q)]
    retrieved.append(top)
    ctx = "\n".join(f"[{j+1}] {c}" for j, c in enumerate(top))
    rag.append(chat_prompt(tok, [{"role": "system", "content": "Answer using only the context.\n" + ctx},
                                 {"role": "user", "content": q + " Answer in one sentence."}]))
out_p = generate(model, tok, plain, max_new_tokens=60)["texts"]
out_r = generate(model, tok, rag, max_new_tokens=60)["texts"]
rows, score = [], Counter()
for (q, kws), a, b, top in zip(QUESTIONS, out_p, out_r, retrieved):
    ok_p = all(k.lower() in a[0].lower() for k in kws)
    ok_r = all(k.lower() in b[0].lower() for k in kws)
    score["plain"] += ok_p
    score["rag"] += ok_r
    rows.append({"question": q, "keywords": kws, "retrieved": top, "plain": a[0], "rag": b[0],
                 "plain_ok": ok_p, "rag_ok": ok_r})
    print(f"\nQ: {q}\n  plain [{ok_p}]: {a[0][:120]}\n  rag   [{ok_r}]: {b[0][:120]}")
summary = {"n": len(QUESTIONS), "plain_keyword_accuracy": score["plain"] / len(QUESTIONS),
           "rag_keyword_accuracy": score["rag"] / len(QUESTIONS)}
print("\n", summary)
write_json(OUTPUTS / "runs" / "ext_rag" / "summary.json", {"summary": summary, "rows": rows})
