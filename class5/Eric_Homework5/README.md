# 1. Clone & setup
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate      # or .\.venv\Scripts\activate.ps1 on Windows
pip install -r requirements.txt

# 2. Build index (reads docs in data/raw/)
python -m app.build_index --rebuild

# 3. Try search
python -m app.hybrid_search --query "hybrid retrieval" --k 3 --method rrf

# 4. Run API
uvicorn api.main:app --reload --port 8000
# → open http://127.0.0.1:8000/hybrid_search?query=example
