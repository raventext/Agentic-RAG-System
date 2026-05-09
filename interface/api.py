# interface/api.py

import sys
import os
import time
import json

from pathlib import Path
from datetime import datetime

# add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# import all agent modules
from agent.memory    import AgentMemory
from agent.planner   import Planner
from agent.retriever import HybridRetriever, RetrievedChunk
from agent.tools     import ToolExecutor

# ── OpenRouter config ─────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found in .env"
    )

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openai/gpt-4o-mini"
)

llm_client = OpenAI(
    api_key  = OPENROUTER_API_KEY,
    base_url = "https://openrouter.ai/api/v1"
)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
LOGS_DIR   = BASE_DIR / "logs"

STATIC_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "api_session.jsonl"

# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title       = "Agentic RAG",
    version     = "1.0",
    description = "AI Research Assistant"
)

# serve static frontend
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)

# ── Global agent state ────────────────────────────────────
memory    = AgentMemory()
planner   = Planner()
tools     = ToolExecutor()
retriever = None


# ── Retriever loader ──────────────────────────────────────
def get_retriever() -> HybridRetriever:
    global retriever

    if retriever is None:
        print("\n[api] Loading retriever...")
        retriever = HybridRetriever()
        print("[api] Retriever ready.\n")

    return retriever


# ── Request / Response models ─────────────────────────────
class ChatRequest(BaseModel):
    message: str


class ChunkInfo(BaseModel):
    title:     str
    authors:   str
    published: str
    score:     float
    excerpt:   str


class ChatResponse(BaseModel):
    reply:           str
    action:          str
    reasoning:       str
    confidence:      float
    sources:         list[ChunkInfo]
    latency_ms:      int
    rewritten_query: str


# ── LLM SYSTEM PROMPT ─────────────────────────────────────
ANSWER_SYSTEM = """
You are a helpful research assistant specializing in AI research papers.

Rules:
- Use ONLY the provided context
- Cite paper titles when referencing claims
- If context is insufficient, say so clearly
- Be concise but informative
- Never invent papers or results
"""


# ── LLM helper ────────────────────────────────────────────
def call_llm(
    prompt: str,
    system: str = ANSWER_SYSTEM
) -> str:
    """
    Safe OpenRouter call.
    Never crashes API.
    """

    try:

        response = llm_client.chat.completions.create(
            model = OPENROUTER_MODEL,

            messages = [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature = 0.3,

            extra_headers = {
                "HTTP-Referer": "http://localhost:8000",
                "X-Title":      "Agentic RAG"
            }
        )

        content = response.choices[0].message.content

        if not content:
            return "The model returned an empty response."

        return content.strip()

    except Exception as e:

        print(f"[api] LLM ERROR: {e}")

        return (
            "I encountered an error generating a response. "
            f"({type(e).__name__})"
        )


# ── Answer generators ─────────────────────────────────────
def generate_answer(
    query: str,
    chunks: list[RetrievedChunk]
) -> str:
    """
    Generate grounded answer from retrieved chunks.
    """

    context_parts = []

    for i, c in enumerate(chunks):

        context_parts.append(
            f"[Source {i+1}] {c.title} ({c.published})\n"
            f"Authors: {c.authors}\n\n"
            f"{c.text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
Context from research papers:

{context}

User question:
{query}

Answer using ONLY the context above.
"""

    return call_llm(prompt)


def generate_direct_answer(query: str) -> str:
    """
    Answer from conversation memory only.
    """

    conv_context = memory.conversation.get_context()

    prompt = f"""
Conversation so far:

{conv_context}

User question:
{query}

Answer:
"""

    return call_llm(prompt)


# ── Routes ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():

    html_file = STATIC_DIR / "index.html"

    if not html_file.exists():

        return HTMLResponse(
            """
            <h2>Frontend not found</h2>
            <p>Create:</p>
            <pre>interface/static/index.html</pre>
            """,
            status_code=404
        )

    return FileResponse(str(html_file))


@app.get("/health")
async def health():

    return {
        "status":        "ok",
        "model":         OPENROUTER_MODEL,
        "retriever":     retriever is not None,
        "timestamp":     datetime.now().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    start_time = time.time()

    query = req.message.strip()

    chunks: list[RetrievedChunk] = []

    reply = ""

    # ── Empty input ───────────────────────────────────────
    if not query:

        return ChatResponse(
            reply           = "Please enter a question.",
            action          = "refuse",
            reasoning       = "Empty input",
            confidence      = 1.0,
            sources         = [],
            latency_ms      = 0,
            rewritten_query = ""
        )

    print(f"\n[api] User: {query}")

    # ── Memory ────────────────────────────────────────────
    memory.add_user_turn(query)

    # ── Planner ───────────────────────────────────────────
    decision = planner.decide(
        user_query = query,
        memory     = memory,
        iteration  = 0
    )

    print(f"[api] Action: {decision.action}")

    # ── Execute ───────────────────────────────────────────
    if decision.action == "retrieve":

        search_query = (
            decision.rewritten_query
            or query
        )

        r = get_retriever()

        chunks = r.retrieve(
            search_query,
            log=False
        )

        if chunks:

            reply = generate_answer(
                query,
                chunks
            )

        else:

            reply = (
                "I searched the corpus but couldn't "
                "find relevant papers on this topic."
            )

    elif decision.action == "clarify":

        reply = (
            decision.clarifying_question
            or "Could you clarify your question?"
        )

    elif decision.action == "tool":

        tool_result = tools.execute(
            decision.tool_name,
            decision.tool_input
        )

        if tool_result.success:

            reply = tools.summarize_tool_result(
                tool_result,
                query
            )

        else:

            print(
                f"[api] Tool failed: {tool_result.error}"
            )

            # fallback retrieval
            try:

                chunks = get_retriever().retrieve(
                    query,
                    log=False
                )

                if chunks:
                    reply = generate_answer(
                        query,
                        chunks
                    )
                else:
                    reply = (
                        "The tool failed and I could not "
                        "find relevant papers."
                    )

            except Exception as e:

                print(f"[api] Fallback ERROR: {e}")

                reply = (
                    "The external tool failed."
                )

    elif decision.action == "refuse":

        reply = (
            f"{decision.refusal_reason}\n\n"
            f"I'm specialized in AI research papers."
        )

    elif decision.action == "answer":

        reply = generate_direct_answer(query)

    else:

        reply = (
            "Unknown planner action."
        )

    # ── Finalize ──────────────────────────────────────────
    latency_ms = int(
        (time.time() - start_time) * 1000
    )

    # save assistant turn
    memory.add_assistant_turn(
        reply,
        action=decision.action
    )

    memory.record_episode(
        question     = query,
        action_taken = decision.action,
        papers_cited = [c.title for c in chunks],
        was_answered = decision.action not in [
            "refuse",
            "clarify"
        ]
    )

    # ── Logging ───────────────────────────────────────────
    try:

        log_entry = {
            "timestamp": datetime.now().isoformat(),

            "query": query,

            "action": decision.action,

            "reasoning": decision.reasoning,

            "rewritten_query":
                decision.rewritten_query or "",

            "confidence":
                decision.confidence,

            "chunks_retrieved":
                len(chunks),

            "latency_ms":
                latency_ms,

            "reply":
                reply[:1000]
        }

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(log_entry)
                + "\n"
            )

    except Exception as e:

        print(f"[api] Log write failed: {e}")

    # ── Response ──────────────────────────────────────────
    return ChatResponse(

        reply      = reply,

        action     = decision.action,

        reasoning  = decision.reasoning,

        confidence = decision.confidence,

        latency_ms = latency_ms,

        rewritten_query =
            decision.rewritten_query or "",

        sources = [

            ChunkInfo(
                title     = c.title[:100],
                authors   = c.authors[:100],
                published = c.published,
                score     = round(c.score, 3),
                excerpt   = c.text[:250]
            )

            for c in chunks[:3]
        ]
    )


@app.post("/reset")
async def reset():

    global memory

    memory = AgentMemory()

    print("[api] Memory cleared.")

    return {
        "status": "memory cleared"
    }


# ── Startup ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():

    print("\n" + "=" * 60)
    print("  Agentic RAG API")
    print("=" * 60)

    print(f"Model : {OPENROUTER_MODEL}")
    print(f"Logs  : {LOG_FILE}")
    print(f"Open  : http://localhost:8000")

    print("=" * 60 + "\n")


# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "interface.api:app",

        host   = "0.0.0.0",

        port   = 8000,

        reload = False
    )