import os
import sqlite3
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "index", "mydata.db"))
FAISS_PATH = os.environ.get("FAISS_PATH", os.path.join(os.path.dirname(__file__), "..", "index", "faiss.index"))
IDMAP_PATH = os.environ.get("IDMAP_PATH", os.path.join(os.path.dirname(__file__), "..", "index", "id_map.npy"))

MODEL_NAME = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_schema(conn: sqlite3.Connection):
    # documents table as in teacher snippet
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id    INTEGER PRIMARY KEY,
            title     TEXT,
            author    TEXT,
            year      INTEGER,
            keywords  TEXT
        );
    """)
    # allow many chunks per document
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id  INTEGER PRIMARY KEY,
            doc_id    INTEGER,
            content   TEXT
        );
    """)
    # FTS5 index over chunks.content; external content to keep a single source of truth
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
            content,
            content='chunks',
            content_rowid='chunk_id'
        );
    """)
    conn.commit()

def load_encoder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def build_faiss(dim: int) -> faiss.Index:
    # cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)
    return index

def save_faiss(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_faiss(path: str) -> faiss.Index:
    return faiss.read_index(path)

def normalize_scores(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return [0.0 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]
