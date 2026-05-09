# Agentic RAG System

A conversational agent that answers questions over a corpus of recent AI research
papers from arXiv (cs.AI category, last 90 days) with autonomous decision-making,
hybrid retrieval, three-layer memory, and full observability.

This is not a naive RAG pipeline. The agent decides for itself when to retrieve,
when to ask a clarifying question, when to call an external tool, and when to refuse.

---

## Quick Start

Clone and run in under 10 minutes.

```bash
# 1. clone the repo
git clone https://github.com/raventext/Agentic-RAG-System.git
cd Agentic-RAG-System

# 2. create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. copy and fill in environment variables
cp .env.example .env
# open .env and add your keys (see Environment Variables below)

# 5. run the ingestion pipeline (once -- builds the vector store)
python ingest/fetch_papers.py
python ingest/parse_pdfs.py
python ingest/chunk_texts.py
python ingest/embed_chunks.py

# 6. launch the web interface
python -m interface.api
```

Open http://localhost:8000 in your browser.

> Note: Step 5 takes 10-15 minutes on first run. The embedding model (~80MB)
> downloads automatically and runs fully locally -- no API call needed.
> Steps 5 onwards only need to run once. After that, just run step 6.

---

## Environment Variables

Create a `.env` file at the project root (copy from `.env.example`):

```env
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Get an OpenRouter API key at https://openrouter.ai.
The free tier is sufficient for development and testing.

---

## Architecture Overview

```
INGESTION PIPELINE (run once offline)
--------------------------------------
arXiv API --> fetch_papers.py --> ~50 cs.AI PDFs
          --> parse_pdfs.py   --> cleaned text
          --> chunk_texts.py  --> 512-token chunks
          --> embed_chunks.py --> vectors in ChromaDB


AGENT LOOP (runs on every user message)
-----------------------------------------
User message
     |
     v
AgentMemory (3 layers)
  |-- ConversationMemory  (sliding window + summary)
  |-- SemanticMemory      (entity facts extracted)
  +-- EpisodicMemory      (past Q&A outcomes)
     |
     v
Planner (LLM decides action)
  |-- retrieve --> HybridRetriever
  |                |-- BM25 keyword search
  |                |-- Semantic search (embeddings)
  |                |-- RRF merge
  |                +-- Cross-encoder reranker
  |-- clarify  --> returns clarifying question
  |-- tool     --> ToolExecutor (arXiv API / web search)
  |-- refuse   --> returns scoped refusal message
  +-- answer   --> answers from memory directly
     |
     v
Answer generation (OpenRouter / gpt-4o-mini)
     |
     v
Response + citations + decision trace


OBSERVABILITY + EVALUATION
---------------------------
logs/api_session.jsonl  -- every decision logged
eval/harness.py         -- 12 test cases + ablation
```

---

## Project Structure

```
agentic-rag/
|-- ingest/
|   |-- fetch_papers.py     download arXiv PDFs + metadata
|   |-- parse_pdfs.py       extract and clean text
|   |-- chunk_texts.py      512-token overlapping chunks
|   +-- embed_chunks.py     local embeddings -> ChromaDB
|-- agent/
|   |-- planner.py          LLM decision maker (5 actions)
|   |-- retriever.py        BM25 + semantic + RRF + reranker
|   |-- memory.py           conversation + semantic + episodic
|   +-- tools.py            arXiv live search + web search
|-- interface/
|   |-- api.py              FastAPI backend
|   |-- cli.py              command-line interface
|   +-- static/
|       +-- index.html      single-file frontend
|-- eval/
|   +-- harness.py          12 test cases + ablation study
|-- logs/                   per-session JSONL decision traces
|-- data/
|   |-- pdfs/               downloaded PDFs
|   |-- texts/              extracted text files
|   |-- chunks/             chunked JSON files
|   |-- chroma/             ChromaDB vector store
|   +-- metadata.json       paper metadata index
|-- check_setup.py          verify dependencies and API keys
|-- requirements.txt
|-- .env.example
+-- README.md
```

---

## Running the Evaluation

```bash
# full evaluation -- 12 test cases
python eval/harness.py

# planner-only mode (fast, no retriever needed)
python eval/harness.py --fast

# full eval + ablation study (reranking ON vs OFF)
python eval/harness.py --ablation
```

Sample results:

```
Overall pass rate  : 11/12  (92%)
Action accuracy    : 92%
Avg content score  : 88%

By category:
  retrieve     [========..]  7/8
  clarify      [==========]  2/2
  refuse       [==========]  2/2

ABLATION -- Reranking ON vs OFF
  Average WITH reranking    : 83%
  Average WITHOUT reranking : 68%
  Delta                     : +15%
```

---

## Decisions Log

The short version of every major design choice -- what was considered,
what was picked, and the one reason that mattered most.

---

### 1. Corpus -- arXiv cs.AI, last 90 days

Picked arXiv cs.AI because it is topically focused (every document is
AI research) and the 90-day window creates a clear boundary -- questions
outside it force the agent to use a tool or refuse, which exercises the
agent loop properly. Other options like Wikipedia or textbooks lack this
natural in/out-of-scope boundary.

---

### 2. PDF Parser -- PyMuPDF

Three parsers were tested: pypdf, pdfplumber, and PyMuPDF. PyMuPDF was
the fastest and handled multi-column academic paper layouts the best.
Scanned PDFs (image-only) are detected by a word count check and skipped.

---

### 3. Chunking -- 512 tokens with 50-token overlap

512 tokens is roughly one idea in an academic paper -- a paragraph, a
results description. Smaller chunks lose context; larger chunks stuff
too much irrelevant text into the LLM prompt. The 50-token overlap
(about 10%) ensures nothing important gets cut at a boundary.

---

### 4. Embeddings -- all-MiniLM-L6-v2 (local)

Runs fully on your machine -- no API calls, no rate limits, no cost.
For a 50-paper corpus the quality difference versus API-based embeddings
is not meaningful enough to justify the dependency. The model downloads
once (~80MB) and then runs offline.

---

### 5. Vector Store -- ChromaDB

Runs locally, saves to disk, zero configuration, and queries ~4,000
chunks in under 100ms. Cloud options like Pinecone add setup complexity
and a network dependency at query time without meaningful benefit at
this corpus size.

---

### 6. Retrieval -- Hybrid Search + Reranking + Query Rewriting

Three techniques are implemented (one was required).

Hybrid search combines BM25 (keyword matching) with semantic search
(vector similarity). BM25 catches exact terms like author names and
acronyms (LoRA, RLHF) that score poorly on semantic similarity alone.
The two result lists are merged with Reciprocal Rank Fusion -- a
parameter-free method that works across different score scales.

Reranking uses a cross-encoder to re-score the top 40 merged candidates.
A cross-encoder reads the query and document together and is more
accurate than the bi-encoder used for retrieval, but too slow to run
over the full corpus. Running over 40 candidates takes under a second.
Ablation shows +15% improvement over skipping reranking.

Query rewriting rewrites vague user questions into keyword-rich search
queries before retrieval hits the vector store. Example:
  User asks:  "how do they stop it from forgetting things?"
  Rewritten:  "continual learning catastrophic forgetting prevention"

---

### 7. Agent Framework -- custom, no LangChain

LangChain hides the planner prompt inside its abstractions, which makes
it hard to show what the agent decided and why -- a core requirement of
this project. A custom implementation means every decision is logged
explicitly, the prompt is readable in one file, and there is no
framework magic to debug.

---

### 8. Memory -- three layers

Conversation memory keeps recent turns as raw text and compresses older
turns into a summary via the LLM. This is better than a sliding window
which simply deletes old turns -- context from early in the conversation
is preserved in compressed form.

Semantic memory extracts key facts from every assistant response and
stores them as an entity-to-fact dictionary. If the agent explains
what LoRA is in turn 3, that fact is still available in turn 20 without
re-retrieval.

Episodic memory records what action was taken for each question and
whether it worked. If a retrieval attempt returned nothing, the agent
knows not to try the same approach for a similar question later.

---

### 9. LLM -- OpenRouter with gpt-4o-mini

Gemini's free tier daily quota was exhausted during development.
OpenRouter provides an OpenAI-compatible API, so the standard OpenAI
Python client works without modification. gpt-4o-mini produces reliable
structured JSON output for the planner, which is the most critical
property for keeping the agent loop stable.

---

### 10. Frontend -- FastAPI + single HTML file

No build step, no Node.js, no extra dependencies. The HTML file serves
directly from FastAPI's static file handler. The decision trace sidebar
shows the agent's action, reasoning, and confidence for every message --
satisfying the observability requirement visually without any additional
tooling.

---

## Known Limitations and Failure Modes

These were observed during testing, not hypothesized.

**Rate limits during paper fetching.** The arXiv API returns a 429
error if more than 50 results are requested in one call. The fetch
script is capped at 50 papers with delays between downloads. Running
it twice quickly will hit the limit again -- wait 60 seconds between
runs.

**Planner misroutes "latest" queries.** Questions with words like
"latest" or "today" sometimes route to the tool action instead of
retrieve, even when the corpus has a relevant paper. The planner
over-interprets temporal language. Observed in eval case R05.

**Scanned PDFs produce no text.** About 2-5% of arXiv PDFs are
image-based scans. PyMuPDF cannot extract text from these. They are
skipped, creating small gaps in corpus coverage.

**Memory resets on server restart.** All three memory layers live in
RAM. Restarting the FastAPI server wipes the conversation history.

**BM25 index takes 15-20 seconds to build on first request.** The
index is built in memory from the chunk files on startup, causing
a noticeable delay before the first answer.

**No multi-user support.** All browser sessions share one memory
instance. Two users talking simultaneously would mix each other's
conversation history.

---

## What I'd Do With Another Week

**Streaming responses.** The interface blocks for 3-8 seconds while the
answer generates. FastAPI supports streaming and the OpenAI client
supports stream=True. This is a one-day change with the biggest
visible impact on usability.

**Persistent memory across sessions.** Serialize all three memory layers
to SQLite on shutdown and reload on startup. The agent would remember
what a user was researching across multiple sessions. Episodic memory
becomes far more useful when it spans days rather than minutes.

**Parent-document retrieval.** Retrieve at the chunk level for precision
but return the surrounding full section to the LLM for answer generation.
This fixes the "answer cut off mid-sentence" failure mode without
hurting retrieval accuracy.

**Automatic corpus refresh.** A daily job using the arXiv API to fetch
new papers and add them to ChromaDB. The agent would always answer from
a rolling 90-day window without manual re-ingestion.

**Pickle the BM25 index.** Save the built index to disk so it loads in
milliseconds on startup instead of rebuilding from scratch every time.
One-hour fix that eliminates the first-request delay.

**Expand the evaluation set.** Generate synthetic Q&A pairs from paper
abstracts to test retrieval accuracy at scale, and add human preference
scoring for answer quality. The current 12 questions test behaviour but
not answer quality.

**Local LLM fallback via Ollama.** Add an option to run a local model
(Mistral 7B or Llama 3) when no API key is available. This makes the
system fully self-contained and removes the OpenRouter dependency for
offline use.

---

## Dependencies

| Package               | Version   | Purpose                        |
|-----------------------|-----------|--------------------------------|
| openai                | >=1.0.0   | OpenRouter API client          |
| sentence-transformers | >=2.6.0   | Local embeddings and reranking |
| chromadb              | >=0.4.0   | Vector store                   |
| pymupdf               | >=1.23.0  | PDF text extraction            |
| rank-bm25             | >=0.2.2   | BM25 keyword search            |
| arxiv                 | >=2.1.0   | Paper fetching                 |
| fastapi               | >=0.110.0 | Web API                        |
| uvicorn               | >=0.27.0  | ASGI server                    |
| tiktoken              | >=0.6.0   | Token counting for chunking    |
| rich                  | >=13.0.0  | CLI formatting                 |
| python-dotenv         | >=1.0.0   | Environment variable loading   |
| tqdm                  | >=4.66.0  | Progress bars                  |
| requests              | >=2.31.0  | HTTP calls                     |
