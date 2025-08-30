import os, json, argparse, glob, sqlite3
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from .common import get_conn, ensure_schema, load_encoder, embed_texts, build_faiss, save_faiss, DB_PATH, FAISS_PATH, IDMAP_PATH

def chunk_text(text: str, max_chars: int = 600) -> List[str]:
    # Simple character-based chunking to keep the example minimal
    text = text.strip().replace("\r\n", "\n")
    chunks = []
    buf = []
    count = 0
    for line in text.split("\n"):
        if not line.strip():
            line = "\n"
        if count + len(line) > max_chars:
            chunks.append(" ".join(buf).strip())
            buf, count = [line], len(line)
        else:
            buf.append(line)
            count += len(line)
    if buf:
        chunks.append(" ".join(buf).strip())
    # filter empties
    return [c for c in chunks if c.strip()]

def ingest_from_folder(conn: sqlite3.Connection, raw_folder: str) -> List[Dict]:
    rows = []
    doc_id_counter = 1
    for path in sorted(glob.glob(os.path.join(raw_folder, "*.txt"))):
        title = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        # insert documents row
        conn.execute("INSERT INTO documents(doc_id, title, author, year, keywords) VALUES (?, ?, ?, ?, ?)",
                     (doc_id_counter, title, None, None, None))
        for ch in chunks:
            cur = conn.execute("INSERT INTO chunks(doc_id, content) VALUES (?, ?)", (doc_id_counter, ch))
            chunk_id = cur.lastrowid
            conn.execute("INSERT INTO doc_chunks(rowid, content) VALUES (?, ?)", (chunk_id, ch))
            rows.append({"chunk_id": chunk_id, "doc_id": doc_id_counter, "content": ch})
        doc_id_counter += 1
    conn.commit()
    return rows

def ingest_from_week4_json(conn: sqlite3.Connection, json_path: str) -> List[Dict]:
    payload = json.load(open(json_path, "r", encoding="utf-8"))
    docs_meta = payload.get("docs_meta", [])
    chunks = payload.get("chunks", [])
    # Insert documents
    for d in docs_meta:
        conn.execute("INSERT INTO documents(doc_id, title, author, year, keywords) VALUES (?, ?, ?, ?, ?)",
                     (d["doc_id"], d.get("title"), d.get("author"), d.get("year"), d.get("keywords")))
    # Insert chunks
    rows = []
    for c in chunks:
        cur = conn.execute("INSERT INTO chunks(doc_id, content) VALUES (?, ?)", (c["doc_id"], c["content"]))
        chunk_id = cur.lastrowid
        conn.execute("INSERT INTO doc_chunks(rowid, content) VALUES (?, ?)", (chunk_id, c["content"]))        
        rows.append({"chunk_id": chunk_id, "doc_id": c["doc_id"], "content": c["content"]})
    conn.commit()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="drop existing index files before build")
    ap.add_argument("--from-week4", type=str, default=None, help="optional JSON export with docs_meta and chunks")
    ap.add_argument("--raw-folder", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    args = ap.parse_args()

    if args.rebuild:
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
        if os.path.exists(FAISS_PATH): os.remove(FAISS_PATH)
        if os.path.exists(IDMAP_PATH): os.remove(IDMAP_PATH)

    conn = get_conn()
    ensure_schema(conn)

    # Ingest rows -> list of dicts with chunk_id and text
    if args.from_week4 and os.path.exists(args.from_week4):
        rows = ingest_from_week4_json(conn, args.from_week4)
    else:
        rows = ingest_from_folder(conn, args.raw_folder)

    texts = [r["content"] for r in rows]
    model = load_encoder()
    embs = embed_texts(model, texts)  # (N, dim) normalized

    index = build_faiss(embs.shape[1])
    index.add(embs)  # vector IDs are [0..N-1]

    # map vector_id -> chunk_id
    id_map = np.array([r["chunk_id"] for r in rows], dtype=np.int64)
    np.save(IDMAP_PATH, id_map)

    # persist FAISS
    save_faiss(index, FAISS_PATH)
    print(f"Built FAISS with {index.ntotal} vectors.\nDB: {DB_PATH}\nFAISS: {FAISS_PATH}\nIDMAP: {IDMAP_PATH}")

if __name__ == "__main__":
    main()
