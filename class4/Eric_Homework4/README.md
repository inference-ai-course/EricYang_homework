# Scientific RAG Pipeline (Class 4)

Semantic search over arXiv cs.CL papers using Sentence-Transformers + FAISS. Includes both LangChain-based and custom FAISS implementations.

---

# 1. Clone & setup
git clone <your-repo-url>
cd <your-repo-folder>

py -3.12 -m venv .venv312
.\.venv312\Scripts\activate      # or source .venv312/bin/activate on Mac/Linux
pip install --upgrade pip

# Core dependencies
pip install faiss-cpu sentence-transformers PyMuPDF fastapi "uvicorn[standard]" tqdm

# Optional LangChain version
pip install langchain langchain-community langchain-core

---

# 2. Download PDFs
python scripts/download_arxiv.py
# → Downloads 50 cs.CL papers from arXiv into /data/raw_pdfs

---

# 3. Build the index
python scripts/build_index.py
# → Extracts, chunks, embeds, and indexes all documents using FAISS

---

# 4. Run the FastAPI search server
uvicorn app.main:app --reload --reload-exclude ".venv312*"
# → Visit http://127.0.0.1:8000/search?q=What+is+contrastive+learning

---

# 5. Try it out
- `/search?q=...&k=3` returns the top-k semantic matches
- You can query manually or integrate the endpoint into a frontend
- Also try `scripts/demo_search.py` or `query_local.py` for CLI test

---

# 6. Deliverables
- LangChain-based pipeline: `class_4_lecture.ipynb`
- Custom FAISS pipeline: `Class 4 Homework.ipynb`
- API: `/search` FastAPI route
- Retrieval report: `artifacts/metadata/retrieval_report.md`

