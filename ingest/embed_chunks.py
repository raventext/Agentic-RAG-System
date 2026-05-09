# ingest/embed_chunks.py

import json
import chromadb
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────
CHUNKS_DIR  = Path("data/chunks")
CHROMA_DIR  = Path("data/chroma")
COLLECTION  = "arxiv_papers"
BATCH_SIZE  = 64       # SentenceTransformers handles larger batches fine
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def load_all_chunks() -> list[dict]:
    all_chunks = []
    for chunk_file in CHUNKS_DIR.glob("*.json"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
    return all_chunks


def build_vector_store():
    print("\n[embed] Loading embedding model (downloads once ~80MB)...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"[embed] Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    collection = get_chroma_collection()
    all_chunks = load_all_chunks()

    if not all_chunks:
        print("[embed] No chunks found. Run chunk_texts.py first.")
        return

    # find chunks not yet embedded
    existing_ids = set(collection.get()["ids"])
    new_chunks   = [c for c in all_chunks if c["chunk_id"] not in existing_ids]

    print(f"\n[embed] Total chunks      : {len(all_chunks)}")
    print(f"[embed] Already embedded  : {len(existing_ids)}")
    print(f"[embed] To embed now      : {len(new_chunks)}\n")

    if not new_chunks:
        print("[embed] Nothing to do — all chunks already embedded.")
        verify_store(model, collection)
        return

    # process in batches
    for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding"):
        batch = new_chunks[i : i + BATCH_SIZE]

        texts     = [c["text"]     for c in batch]
        ids       = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "paper_id":    c.get("paper_id",    ""),
                "title":       c.get("title",       "")[:200],
                "authors":     c.get("authors",     "")[:200],
                "published":   c.get("published",   ""),
                "url":         c.get("url",         ""),
                "chunk_index": c.get("chunk_index", 0),
                "token_count": c.get("token_count", 0),
            }
            for c in batch
        ]

        # embed locally — no API call
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        collection.add(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas
        )

    print(f"\n[embed] Done. {collection.count()} chunks in ChromaDB.\n")
    verify_store(model, collection)


def verify_store(model, collection):
    print("[verify] Running test query...\n")

    query_vector = model.encode("transformer attention mechanism").tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    print("Test query: 'transformer attention mechanism'\n")
    print("Top 3 results:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n  [{i+1}] {meta['title'][:60]}")
        print(f"       Published : {meta['published']}")
        print(f"       Excerpt   : {doc[:150]}...")


if __name__ == "__main__":
    build_vector_store()