import os
import re
from pathlib import Path

import download_arxiv
from arxiv import Client, Search, SortCriterion, SortOrder
from tqdm import tqdm

RAW_DIR = Path("data/raw_pdfs")
# RAW_DIR.mkdir(parents=True, exists_ok=True)

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def main(max_results: int = 50):
    client = Client(page_size=25, delay_seconds=3, num_retries=3)
    search = Search(
        query = "cat:cs.CL",
        max_results=max_results,
        sort_by=SortCriterion.SubmittedDate,
        sort_order=SortOrder.Descending,
    )

    for result in tqdm(client.results(search), total=max_results, desc="Downloading PDFS"):
        paper_id = result.get_short_id()
        title = safe_filename(result.title)[:120] or paper_id
        fname = f"{paper_id}__{title}.pdf"
        out_path = RAW_DIR / fname
        if out_path.exists():
            continue
        result.download_pdf(dirpath=str(RAW_DIR), filename=fname)

if __name__=="__main__":
    main()