# ingest/chunk_texts.py

import json
import tiktoken
from pathlib import Path
from tqdm import tqdm

TEXT_DIR   = Path("data/texts")
CHUNKS_DIR = Path("data/chunks")

# ── Chunking config ─────────────────────────────────────
CHUNK_SIZE    = 512    # tokens per chunk
OVERLAP       = 50     # ~10% overlap between chunks
ENCODING_NAME = "cl100k_base"   # same tokenizer as GPT-4 / Gemini-compatible
# ────────────────────────────────────────────────────────


def chunk_text(text: str, paper_id: str, metadata: dict) -> list[dict]:
    """
    Split text into overlapping token-based chunks.
    Each chunk carries its parent paper's metadata.
    """
    enc    = tiktoken.get_encoding(ENCODING_NAME)
    tokens = enc.encode(text)

    chunks = []
    start  = 0
    idx    = 0

    while start < len(tokens):
        end        = min(start + CHUNK_SIZE, len(tokens))
        chunk_tok  = tokens[start:end]
        chunk_text = enc.decode(chunk_tok)

        chunks.append({
            # unique ID for this chunk
            "chunk_id":   f"{paper_id}_chunk_{idx}",
            "paper_id":   paper_id,
            "text":       chunk_text,
            "chunk_index": idx,
            "token_count": len(chunk_tok),

            # carry paper metadata into every chunk
            # so we can show citations later
            **metadata
        })

        idx   += 1
        start += CHUNK_SIZE - OVERLAP   # slide window with overlap

    return chunks


def load_metadata() -> dict[str, dict]:
    """
    Load paper metadata saved by fetch_papers.py.
    Falls back to empty dict if file doesn't exist yet.
    """
    meta_path = Path("data/metadata.json")
    if not meta_path.exists():
        print("  [warn] data/metadata.json not found — metadata will be minimal")
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["paper_id"]: r for r in records}


def chunk_all_texts() -> list[dict]:
    """
    Chunk every parsed text file. Save all chunks to data/chunks/.
    Returns the full list of chunk dicts.
    """
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_map = load_metadata()

    text_files  = list(TEXT_DIR.glob("*.txt"))
    all_chunks  = []

    print(f"\n[chunk] Chunking {len(text_files)} text files...\n")

    for text_path in tqdm(text_files, desc="Chunking"):
        paper_id   = text_path.stem
        chunk_path = CHUNKS_DIR / f"{paper_id}.json"

        # skip if already chunked
        if chunk_path.exists():
            with open(chunk_path, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
            continue

        text     = text_path.read_text(encoding="utf-8")
        metadata = metadata_map.get(paper_id, {
            "title":     paper_id,
            "authors":   "unknown",
            "published": "unknown",
            "url":       ""
        })

        chunks = chunk_text(text, paper_id, metadata)

        # save per-paper chunks
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        all_chunks.extend(chunks)

    print(f"\n[chunk] Done. {len(all_chunks)} total chunks from {len(text_files)} papers.")
    print(f"        Average chunks per paper: {len(all_chunks) // max(len(text_files), 1)}\n")
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_texts()
    # show a sample chunk
    if chunks:
        c = chunks[0]
        print(f"Sample chunk: {c['chunk_id']}")
        print(f"  Paper : {c.get('title', 'N/A')[:60]}")
        print(f"  Tokens: {c['token_count']}")
        print(f"  Text  : {c['text'][:200]}...")