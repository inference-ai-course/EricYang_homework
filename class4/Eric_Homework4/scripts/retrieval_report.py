# scripts/retrieval_report.py
import json
from pathlib import Path
import datetime as dt
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX = Path("artifacts/index/chunks.index")
CHUNKS = Path("artifacts/metadata/chunks.jsonl")
OUT    = Path("artifacts/metadata/retrieval_report.md")
MODEL  = "all-MiniLM-L6-v2"

QUERIES = [
    "What is the role of attention in transformer models?",
    "How do instruction-tuned LLMs differ from base models?",
    "Techniques for mitigating hallucinations in RAG systems",
    "What datasets are used for multilingual NLU benchmarks?",
    "Contrastive learning methods for sentence embeddings",
]

def norm(x): return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype("float32")

def main():
    model = SentenceTransformer(MODEL)
    index = faiss.read_index(str(INDEX))
    chunks = [json.loads(l) for l in open(CHUNKS, "r", encoding="utf-8")]

    lines = [f"# Retrieval Report ({dt.date.today().isoformat()})", ""]
    for q in QUERIES:
        qv = norm(np.asarray(model.encode([q])))
        D, I = index.search(qv, 3)
        lines.append(f"## Query: {q}")
        for rank, idx in enumerate(I[0], start=1):
            rec = chunks[idx]
            lines.append(f"**[{rank}] {rec['title']} (chunk {rec['chunk_idx']}) — dist={D[0][rank-1]:.4f}**")
            snippet = rec["text"].replace("\n", " ")
            lines.append(f"> {snippet[:800]} ...")
            lines.append("")
        lines.append("---\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {OUT}")

if __name__ == "__main__":
    main()
