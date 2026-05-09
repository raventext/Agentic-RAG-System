# agent/retriever.py

import json
import chromadb
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Config ───────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma")
CHUNKS_DIR      = Path("data/chunks")
COLLECTION      = "arxiv_papers"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SEMANTIC_TOP_K  = 20
BM25_TOP_K      = 20
FINAL_TOP_K     = 5
# ─────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    chunk_id:         str
    text:             str
    score:            float
    title:            str
    authors:          str
    published:        str
    url:              str
    chunk_index:      int
    retrieval_method: str


class HybridRetriever:

    def __init__(self):
        print("[retriever] Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        print("[retriever] Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection    = self.chroma_client.get_collection(COLLECTION)

        self._build_bm25_index()
        print("[retriever] Ready.\n")


    def _build_bm25_index(self):
        print("[retriever] Building BM25 index...")
        self.all_chunks: list[dict] = []

        for chunk_file in CHUNKS_DIR.glob("*.json"):
            with open(chunk_file, "r", encoding="utf-8") as f:
                self.all_chunks.extend(json.load(f))

        tokenized_corpus = [c["text"].lower().split() for c in self.all_chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

        self.chunk_id_to_idx = {
            c["chunk_id"]: i for i, c in enumerate(self.all_chunks)
        }

        print(f"[retriever] BM25 index built over {len(self.all_chunks)} chunks.")


    def _semantic_search(self, query: str, k: int) -> list[RetrievedChunk]:

        # local embeddings — NO API call
        query_vector = self.embed_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )

        chunks = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            score = 1 - (dist / 2)

            chunks.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    score=score,
                    title=meta.get("title", ""),
                    authors=meta.get("authors", ""),
                    published=meta.get("published", ""),
                    url=meta.get("url", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    retrieval_method="semantic"
                )
            )

        return chunks


    def _bm25_search(self, query: str, k: int) -> list[RetrievedChunk]:

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        chunks = []

        for idx in top_indices:

            if scores[idx] == 0:
                continue

            c = self.all_chunks[idx]

            chunks.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    score=float(scores[idx]),
                    title=c.get("title", ""),
                    authors=c.get("authors", ""),
                    published=c.get("published", ""),
                    url=c.get("url", ""),
                    chunk_index=c.get("chunk_index", 0),
                    retrieval_method="bm25"
                )
            )

        return chunks


    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        k: int = 60
    ) -> list[RetrievedChunk]:

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(semantic_results):

            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0)
                + 1 / (k + rank + 1)
            )

            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_results):

            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0)
                + 1 / (k + rank + 1)
            )

            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        sorted_ids = sorted(
            rrf_scores,
            key=lambda x: rrf_scores[x],
            reverse=True
        )

        merged = []

        for cid in sorted_ids:

            chunk = chunk_map[cid]
            chunk.score = rrf_scores[cid]
            chunk.retrieval_method = "hybrid"

            merged.append(chunk)

        return merged


    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:

        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]

        scores = self.reranker.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk.score = float(score)

        return sorted(
            candidates,
            key=lambda c: c.score,
            reverse=True
        )[:top_k]


    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        log: bool = True
    ) -> list[RetrievedChunk]:

        if log:
            print(f"\n[retriever] Query: '{query}'")

        semantic = self._semantic_search(query, SEMANTIC_TOP_K)
        bm25     = self._bm25_search(query, BM25_TOP_K)

        if log:
            print(f"  Semantic hits : {len(semantic)}")
            print(f"  BM25 hits     : {len(bm25)}")

        merged = self._reciprocal_rank_fusion(semantic, bm25)

        if log:
            print(f"  After RRF     : {len(merged)} candidates")

        final = self._rerank(query, merged, top_k)

        if log:
            print(f"  After rerank  : {len(final)} final chunks")

            for i, c in enumerate(final):
                print(f"    [{i+1}] score={c.score:.3f}  {c.title[:50]}")

        return final


    def query_rewrite(
        self,
        original_query: str,
        conversation_context: str = ""
    ) -> str:
        """
        Uses OpenRouter GPT-4o-mini for query rewriting.
        """

        prompt = f"""You are a search query optimizer for a database of AI research papers.

Rewrite the following user question into an optimal search query.

Rules:
- Use technical terms and keywords
- Expand vague pronouns using the conversation context
- Keep it under 15 words
- Return ONLY the rewritten query, nothing else

Conversation context:
{conversation_context or "None"}

User question:
{original_query}

Rewritten query:
"""

        try:

            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You rewrite AI research queries for retrieval systems."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            print(f"[retriever] Query rewrite failed: {e}")

            return original_query


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":

    retriever = HybridRetriever()

    print("\n" + "="*50)
    print("TEST 1 — Basic hybrid retrieval")
    print("="*50)

    results = retriever.retrieve("attention mechanism transformers")

    print(f"\nReturned {len(results)} chunks")

    print("\n" + "="*50)
    print("TEST 2 — Query rewriting")
    print("="*50)

    vague = "how do they stop it from forgetting things"

    rewritten = retriever.query_rewrite(
        vague,
        "discussing continual learning"
    )

    print(f"Original : {vague}")
    print(f"Rewritten: {rewritten}")