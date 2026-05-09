# ingest/parse_pdfs.py

import fitz   # PyMuPDF
import re
from pathlib import Path
from tqdm import tqdm

PDF_DIR  = Path("data/pdfs")
TEXT_DIR = Path("data/texts")


def clean_text(raw: str) -> str:
    """
    Clean raw PDF text:
    - Remove excessive whitespace and newlines
    - Remove page numbers (lone digits on a line)
    - Remove URLs
    - Normalize unicode dashes and quotes
    """
    # remove lone page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", raw)

    # collapse multiple newlines into two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # normalize dashes and quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str | None:
    """
    Extract all text from a PDF file using PyMuPDF.
    Returns None if extraction fails or text is too short.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages_text = []

        for page in doc:
            # "text" mode preserves reading order better than raw
            page_text = page.get_text("text")
            pages_text.append(page_text)

        doc.close()
        full_text = "\n\n".join(pages_text)
        cleaned   = clean_text(full_text)

        # sanity check — skip papers with almost no text (scanned images, etc.)
        if len(cleaned.split()) < 500:
            print(f"  [skip] {pdf_path.name} — too little text (possibly scanned)")
            return None

        return cleaned

    except Exception as e:
        print(f"  [error] {pdf_path.name} — {e}")
        return None


def parse_all_pdfs() -> list[dict]:
    """
    Parse every PDF in PDF_DIR, save extracted text to TEXT_DIR.
    Returns list of dicts with paper_id and text.
    """
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"\n[parse] Found {len(pdf_files)} PDFs to parse...\n")

    results = []

    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        paper_id  = pdf_path.stem
        text_path = TEXT_DIR / f"{paper_id}.txt"

        # skip if already parsed
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
            results.append({"paper_id": paper_id, "text": text})
            continue

        text = extract_text_from_pdf(pdf_path)
        if text is None:
            continue

        # save to disk so we don't re-parse next time
        text_path.write_text(text, encoding="utf-8")
        results.append({"paper_id": paper_id, "text": text})

    print(f"\n[parse] Done. {len(results)} papers parsed successfully.\n")
    return results


if __name__ == "__main__":
    parsed = parse_all_pdfs()
    # show a sample
    if parsed:
        sample = parsed[0]
        print(f"Sample — {sample['paper_id']}:")
        print(sample["text"][:500])
        print("...")