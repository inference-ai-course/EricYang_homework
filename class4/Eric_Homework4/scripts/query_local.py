#scripts/query_local.py

import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("artifacts/index/chunks.index")
CHUNKS_PATH = Path("artifacts/metadata/chunks.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype("float32")

def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    print("[INFO] Loading model & index...")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    chunks = load_chunks()

    while True:
        q = input("\nEnter query (or blank to quit): ").strip()
        if not q:
            break
        q_vec = model.encode([q])
        q_vec = l2_normalize(np.asarray(q_vec))
        k = 3
        D, I = index.search(q_vec, k)
        print("\nTop passages:")
        for rank, idx in enumerate(I[0], start=1):
            rec = chunks[idx]
            print(f"\n[{rank}] {rec['title']} (chunk {rec['chunk_idx']}) - dist={D[0][rank-1]:.4f}")
            print(rec["text"][:600], "...")
    print("\nDone.")

if __name__ == "__main__":
    main()