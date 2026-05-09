
# ingest/fetch_papers.py
import json
import arxiv
import os
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────
PDF_DIR = Path("data/pdfs")
MAX_PAPERS = 50
SEARCH_QUERY = "cat:cs.AI"          # cs.AI category
DAYS_BACK = 90                      # last 90 days
# ────────────────────────────────────────────────────────


def fetch_papers(max_papers: int = MAX_PAPERS) -> list[dict]:
    """
    Search arXiv for recent cs.AI papers and download their PDFs.
    Returns a list of metadata dicts for each downloaded paper.
    """
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[fetch] Searching arXiv for '{SEARCH_QUERY}'...")

    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,     # be polite to arXiv servers
        num_retries=3
    )

    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending   # newest first
    )

    papers_metadata = []

    for result in tqdm(client.results(search), total=max_papers, desc="Downloading"):
        paper_id = result.entry_id.split("/")[-1]   # e.g. "2401.12345v1"
        pdf_path = PDF_DIR / f"{paper_id}.pdf"

        # skip if already downloaded
        if pdf_path.exists():
            print(f"  [skip] {paper_id} already exists")
        else:
            try:
                result.download_pdf(dirpath=str(PDF_DIR), filename=f"{paper_id}.pdf")
                time.sleep(1)   # don't hammer the server
            except Exception as e:
                print(f"  [error] Failed to download {paper_id}: {e}")
                continue

        # store metadata — this becomes searchable later
        papers_metadata.append({
            "paper_id":  paper_id,
            "title":     result.title,
            "authors":   ", ".join(a.name for a in result.authors[:5]),
            "abstract":  result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "pdf_path":  str(pdf_path),
            "url":       result.entry_id,
            "categories": ", ".join(result.categories)
        })

    print(f"\n[fetch] Done. {len(papers_metadata)} papers downloaded to {PDF_DIR}/\n")
    return papers_metadata


if __name__ == "__main__":

    papers = fetch_papers()

    # save metadata to disk
    Path("data").mkdir(exist_ok=True)

    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print("[fetch] Metadata saved to data/metadata.json")

    # print quick summary
    for p in papers[:5]:
        print(f"  • {p['published']}  {p['title'][:70]}")

    print(f"  ... and {len(papers) - 5} more")