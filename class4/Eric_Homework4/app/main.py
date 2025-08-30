# app/main.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("artifacts/index/chunks.index")
CHUNKS_PATH = Path("artifacts/metadata/chunks.jsonl")
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI()
model = SentenceTransformer(MODEL_NAME)
faiss_index = faiss.read_index(str(INDEX_PATH))
chunks = [json.loads(line) for line in open(CHUNKS_PATH, "r", encoding="utf-8")]

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype("float32")

class Passage(BaseModel):
    paper_id: str
    title: str
    chunk_idx: int
    text: str
    distance: float

class SearchResponse(BaseModel):
    query: str
    results: List[Passage]

@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., description="Your question"), k: int = Query(3, ge=1, le=10)):
    """
    Receive a query 'q', embed it, retrieve top-k passages, and return them.
    """
    query_vector = model.encode([q])
    query_vector = l2_normalize(np.asarray(query_vector))
    distances, indices = faiss_index.search(query_vector, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        rec = chunks[idx]
        results.append(Passage(
            paper_id=rec["paper_id"],
            title=rec["title"],
            chunk_idx=rec["chunk_idx"],
            text=rec["text"],
            distance=float(distances[0][rank]),
        ))
    return SearchResponse(query=q, results=results)