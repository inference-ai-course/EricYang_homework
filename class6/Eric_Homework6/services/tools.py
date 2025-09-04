# services/tools.py
from __future__ import annotations
import os

def search_arxiv(query: str) -> str:
    """
    Simulate or perform an arXiv search and return a short passage.
    If env ARXIV_LIVE=1 and the 'arxiv' package is installed, it will fetch the top result.
    """
    try:
        if os.getenv("ARXIV_LIVE", "0") == "1":
            import arxiv  # pip install arxiv
            search = arxiv.Search(
                query=query,
                max_results=1,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            for r in search.results():
                title = (r.title or "").strip()
                summary = (r.summary or "").strip().replace("\n", " ")
                return f"{title}: {summary}"
        # Fallback dummy text
        return f"[arXiv snippet related to '{query}']"
    except Exception as e:
        return f"Error: {e}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result string."""
    try:
        from sympy import sympify
        res = sympify(expression)
        return str(res)
    except Exception as e:
        return f"Error: {e}"
