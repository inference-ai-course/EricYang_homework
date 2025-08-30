import os, argparse, json, sqlite3
from typing import List, Dict
from app.hybrid_search import faiss_topk, keyword_topk, hybrid_search
from app.common import get_conn

def title_of_chunk(conn, chunk_id: int) -> str:
    sql = """        SELECT d.title
    FROM chunks c
    JOIN documents d ON c.doc_id = d.doc_id
    WHERE c.chunk_id=?
    """
    row = conn.execute(sql, (chunk_id,)).fetchone()
    return row[0] if row else None

def hitrate_at_k(results: List[Dict], relevant_titles: List[str], conn) -> float:
    top_titles = []
    for r in results:
        t = title_of_chunk(conn, r["chunk_id"])            
        if t: top_titles.append(t)
    return 1.0 if any(t in relevant_titles for t in top_titles) else 0.0

def evaluate(k: int, method: str):
    conn = get_conn()
    payload = json.load(open(os.path.join(os.path.dirname(__file__), "queries.json"), "r", encoding="utf-8"))
    qs = payload["queries"]
    v_hits = 0.0
    k_hits = 0.0
    h_hits = 0.0
    for item in qs:
        q = item["q"]
        rel = item["relevant_titles"]
        v = faiss_topk(q, k=k)
        s = keyword_topk(q, k=k)
        h = hybrid_search(q, k=k, method=method)
        v_hits += hitrate_at_k(v, rel, conn)
        k_hits += hitrate_at_k(s, rel, conn)
        h_hits += hitrate_at_k(h, rel, conn)
    n = len(qs)
    return {
        "n": n,
        f"vector_only@{k}": v_hits / n,
        f"keyword_only@{k}": k_hits / n,
        f"hybrid({method})@{k}": h_hits / n,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--method", type=str, default="rrf", choices=["rrf", "weighted"])        
    args = ap.parse_args()
    metrics = evaluate(k=args.k, method=args.method)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
