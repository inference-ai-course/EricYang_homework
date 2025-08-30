from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from app.hybrid_search import hybrid_search

app = FastAPI(title="Week5 Hybrid Retrieval API")

class SearchResult(BaseModel):
    chunk_id: int
    doc_id: int
    content: str
    hybrid: float

@app.get("/hybrid_search", response_model=List[SearchResult])
async def do_hybrid_search(
    query: str = Query(..., description="User query"),
    k: int = Query(3, ge=1, le=20),
    method: str = Query("rrf", pattern="^(rrf|weighted)$"),
    alpha: float = Query(0.6, ge=0.0, le=1.0)
):
    rows = hybrid_search(query, k=k, method=method, alpha=alpha)
    # return only fields defined in the pydantic model (includes 'hybrid')
    output = []
    for r in rows:
        output.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "content": r["content"],
            "hybrid": r.get("hybrid", r.get("score", 0.0)),
        })
    return output
