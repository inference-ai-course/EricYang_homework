# scripts/demo_search.py

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX = Path("artifacts/index/chunks.index")
CHUNKS = Path("artifacts/metadata/chunks.jsonl")
MODEL = "all-MiniLM-L6-v2"

def norm(x): return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype("float32")

model = SentenceTransformer(MODEL)
index = faiss.read_index(str(INDEX))
chunks = [json.loads(l) for l in open(CHUNKS, "r", encoding="utf-8")]

query = "How do large language models perform in zero-shot classification?"
q = norm(np.asarray(model.encode([query])))
D, I = index.search(q, 3)

print("Query:", query)
for r, idx in enumerate(I[0], start=1):
    rec = chunks[idx]
    print(f"\n[{r}] {rec['title']} (chunk {rec['chunk_idx']})  dist={D[0][r-1]:.4f}")
    print(rec["text"][:700], "...")