# scripts/build_index.py

import json
import math
import os
from pathlib import Path
from typing import List, Dict

import fitz
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/raw_pdfs")
META_DIR = Path("artifacts/metadata")
INDEX_DIR = Path("artifacts/index")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all text as a single string.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        page_text = page.get_text()  # get raw text from page
        # (Optional) clean page_text here (remove headers/footers)
        pages.append(page_text)
    full_text = "\n".join(pages)
    return full_text


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_tokens]
        chunks.append(" ".join(chunk))
    return chunks

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return (mat/norms).astype("float32")

def main(max_tokens=512, overlap=50, model_name="all-MiniLM-L6-v2"):
    docs_meta: List[Dict] = []
    all_chunks: List[str] = []
    all_chunks_meta: List[Dict] = []

    pdf_files = sorted([p for p in DATA_DIR.glob("*.pdf")])
    print(f"[INFO] Found {len(pdf_files)} PDFs")

    for pi, pdf in enumerate(tqdm(pdf_files, desc="Extracting + Chunking")):
        try:
            text = extract_text_from_pdf(str(pdf))
        except Exception as e:
            print(f"[WARN] Failed to read {pdf}: {e}")
            continue
        chunks = chunk_text(text, max_tokens=max_tokens, overlap=overlap)
        paper_id = pdf.stem.split("__")[0]
        title = "__".join(pdf.stem.split("__")[1:]) if "__" in pdf.stem else pdf.stem

        docs_meta.append({"paper_id": paper_id, "title": title, "path": str(pdf), "num_chunks": len(chunks)})

        for ci, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_chunks_meta.append({
                "paper_id": paper_id,
                "title": title,
                "chunk_idx": ci,
                "source": str(pdf),
            })

    print(f"[INFO] Total chunks: {len(all_chunks)}")

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    embeds = model.encode(all_chunks, batch_size=64, show_progress_bar=True)
    embeds = l2_normalize(np.asarray(embeds))

    dim = embeds.shape[1] # 384
    index = faiss.IndexFlatL2(dim)
    index.add(embeds)

    faiss.write_index(index, str(INDEX_DIR / "chunks.index"))
    with open(META_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for meta, text in zip(all_chunks_meta, all_chunks):
            rec = {**meta, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(META_DIR / "docs_meta.json", "w", encoding="utf-8") as f:
        json.dump(docs_meta, f, ensure_ascii=False, indent=2)

    print("[OK] Index + metadata written.")

if __name__ == "__main__":
    main()