import os, argparse, sqlite3
from typing import List, Dict, Tuple
import numpy as np
import faiss
from .common import get_conn, load_faiss, load_encoder, IDMAP_PATH, FAISS_PATH, normalize_scores

def idmap_load() -> np.ndarray:
    return np.load(IDMAP_PATH)

def faiss_topk(query: str, k: int = 5) -> List[Dict]:
    conn = get_conn()
    encoder = load_encoder()
    index = load_faiss(FAISS_PATH)
    id_map = idmap_load()
    q_emb = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k)
    res = []
    for rank, (score, vid) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
        if vid == -1:
            continue
        chunk_id = int(id_map[vid])
        row = conn.execute("SELECT chunk_id, doc_id, content FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row:
            res.append({"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "content": row["content"], "score": float(score), "rank": rank})
    return res

def keyword_topk(query: str, k: int = 5) -> List[Dict]:
    # FTS5 query with BM25 ranking (lower is better) -> invert to higher-is-better
    conn = get_conn()
    sql = """            SELECT c.chunk_id, c.doc_id, c.content, bm25(doc_chunks) AS bm25
        FROM doc_chunks
        JOIN chunks c ON doc_chunks.rowid = c.chunk_id
        WHERE doc_chunks MATCH ?
        ORDER BY bm25(doc_chunks) ASC
        LIMIT ?
    """
    rows = conn.execute(sql, (query, k)).fetchall()
    res = []
    # convert bm25 (lower better) to positive score (higher better)
    for rank, r in enumerate(rows, start=1):
        bm25_val = float(r["bm25"]) if r["bm25"] is not None else 1000.0
        score = 1.0 / (1.0 + bm25_val)
        res.append({"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "content": r["content"], "score": score, "rank": rank})
    return res

def reciprocal_rank_fusion(list_a: List[Dict], list_b: List[Dict], k_val: int = 60, top_k: int = 3) -> List[Dict]:
    # RRF: sum 1/(k + rank)
    scores: Dict[int, float] = {}
    meta: Dict[int, Dict] = {}
    for lst in (list_a, list_b):
        for item in lst:
            rid = item["chunk_id"]
            rr = 1.0 / (k_val + item["rank"])  # ranks start at 1
            scores[rid] = scores.get(rid, 0.0) + rr
            if rid not in meta:
                meta[rid] = item
    merged = [{**meta[cid], "hybrid": sc} for cid, sc in scores.items()]
    merged.sort(key=lambda x: x["hybrid"], reverse=True)
    return merged[:top_k]

def weighted_sum_merge(list_a: List[Dict], list_b: List[Dict], alpha: float = 0.6, top_k: int = 3) -> List[Dict]:
    # Both lists contain 'score' but on different scales -> normalize within each list first
    a_scores = normalize_scores([x["score"] for x in list_a])
    for i, s in enumerate(a_scores):
        list_a[i]["norm"] = s
    b_scores = normalize_scores([x["score"] for x in list_b])
    for i, s in enumerate(b_scores):
        list_b[i]["norm"] = s
    # merge by chunk_id
    by_id: Dict[int, Dict] = {}
    for x in list_a:
        by_id[x["chunk_id"]] = {**x, "vec_norm": x.get("norm", 0.0), "key_norm": 0.0}
    for x in list_b:
        if x["chunk_id"] in by_id:
            by_id[x["chunk_id"]]["key_norm"] = x.get("norm", 0.0)
        else:
            by_id[x["chunk_id"]] = {**x, "vec_norm": 0.0, "key_norm": x.get("norm", 0.0)}
    merged = []
    for cid, it in by_id.items():
        hybrid = alpha * it.get("vec_norm", 0.0) + (1.0 - alpha) * it.get("key_norm", 0.0)
        it["hybrid"] = hybrid
        merged.append(it)
    merged.sort(key=lambda x: x["hybrid"], reverse=True)
    return merged[:top_k]

def hybrid_search(query: str, k: int = 3, method: str = "rrf", alpha: float = 0.6):
    vec = faiss_topk(query, k)
    key = keyword_topk(query, k)
    if method == "rrf":
        return reciprocal_rank_fusion(vec, key, top_k=k)
    else:
        return weighted_sum_merge(vec, key, alpha=alpha, top_k=k)

def pretty(rows: List[Dict]) -> str:
    lines = []
    for i, r in enumerate(rows, start=1):
        lines.append(f"{i}. [doc_id={r['doc_id']}, chunk_id={r['chunk_id']}] score={r.get('hybrid', r.get('score')):.4f}\n    {r['content'][:160].replace('\n',' ')}...")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--method", type=str, default="rrf", choices=["rrf", "weighted"])
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()
    rows = hybrid_search(args.query, k=args.k, method=args.method, alpha=args.alpha)
    print(pretty(rows))

if __name__ == "__main__":
    main()
