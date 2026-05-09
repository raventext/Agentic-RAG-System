# interface/cli.py

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# import all modules we built
from agent.memory    import AgentMemory
from agent.planner   import Planner, PlannerDecision
from agent.retriever import HybridRetriever, RetrievedChunk
from agent.tools     import ToolExecutor

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Config ────────────────────────────────────────────────
LOGS_DIR     = Path("logs")
LOG_FILE     = LOGS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
# ─────────────────────────────────────────────────────────

console = Console()


# ── Answer generator ─────────────────────────────────────
ANSWER_SYSTEM_PROMPT = """You are a helpful research assistant specializing in AI research papers.
Answer the user's question using ONLY the provided context chunks from research papers.

Rules:
- Cite papers by title and authors when referencing specific claims
- If context is insufficient, say so clearly rather than guessing
- Be concise but complete — 2-5 paragraphs is usually right
- If chunks contradict each other, acknowledge the disagreement
- Never make up paper titles, authors, or results"""


def generate_answer(
    query:    str,
    chunks:   list[RetrievedChunk],
    memory:   AgentMemory
) -> str:
    """
    Generate a grounded answer from retrieved chunks.
    This is the final LLM call after retrieval.
    """

    # format chunks as context
    context_parts = []

    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}] {chunk.title} ({chunk.published})\n"
            f"Authors: {chunk.authors}\n"
            f"{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # get recent conversation for continuity
    conv_context = memory.conversation.get_context()

    prompt = f"""Conversation so far:
{conv_context}

Retrieved context from papers:
{context}

User question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": ANSWER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


def generate_direct_answer(query: str, memory: AgentMemory) -> str:
    """
    Answer directly from conversation memory — no retrieval needed.
    Used when planner decides action == 'answer'.
    """

    conv_context = memory.conversation.get_context()

    prompt = f"""Conversation so far:
{conv_context}

User question: {query}

Answer based on the conversation above:"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# ── Logger ────────────────────────────────────────────────
class SessionLogger:
    """
    Logs every turn with full observability data.
    Each log entry shows what the agent decided and why —
    satisfying the observability requirement.
    """

    def __init__(self):
        LOGS_DIR.mkdir(exist_ok=True)
        self.entries: list[dict] = []

    def log(
        self,
        query:      str,
        decision:   PlannerDecision,
        chunks:     list[RetrievedChunk],
        response:   str,
        latency_ms: int
    ):
        entry = {
            "timestamp":  datetime.now().isoformat(),
            "query":      query,
            "decision": {
                "action":              decision.action,
                "reasoning":           decision.reasoning,
                "rewritten_query":     decision.rewritten_query,
                "clarifying_question": decision.clarifying_question,
                "refusal_reason":      decision.refusal_reason,
                "tool_name":           decision.tool_name,
                "tool_input":          decision.tool_input,
                "confidence":          decision.confidence,
            },
            "retrieved_chunks": [
                {
                    "title":     c.title,
                    "score":     round(c.score, 4),
                    "published": c.published,
                    "excerpt":   c.text[:200]
                }
                for c in chunks
            ],
            "response":   response,
            "latency_ms": latency_ms
        }

        self.entries.append(entry)

        # write to disk after every turn
        with open(LOG_FILE, "w") as f:
            json.dump(self.entries, f, indent=2)

    def print_last_trace(self):
        """Print the last decision trace for the user to inspect."""

        if not self.entries:
            return

        e = self.entries[-1]

        console.print(f"\n[dim]📋 Trace saved → {LOG_FILE}[/dim]")


# ── Main agent loop ───────────────────────────────────────
class AgentCLI:
    """
    Main conversational interface.
    Wires all modules together into one conversation loop.
    """

    def __init__(self):

        console.print(Panel(
            "[bold cyan]Agentic RAG — AI Research Assistant[/bold cyan]\n"
            "[dim]Corpus: ~100 recent cs.AI papers from arXiv[/dim]\n"
            "[dim]Type 'quit' to exit | 'trace' to see last decision log[/dim]",
            expand=False
        ))

        console.print("\n[yellow]Loading components...[/yellow]")

        self.memory   = AgentMemory()
        self.planner  = Planner()
        self.tools    = ToolExecutor()
        self.logger   = SessionLogger()

        console.print("[yellow]Loading retriever (this takes ~20 seconds)...[/yellow]")

        self.retriever = HybridRetriever()

        console.print("\n[green]✓ Ready. Ask me anything about recent AI research.\n[/green]")


    def run(self):
        """Main conversation loop."""

        while True:

            try:
                # get user input
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                console.print("[dim]Goodbye.[/dim]")
                break

            if user_input.lower() == "trace":
                self._show_trace()
                continue

            if user_input.lower() == "memory":
                self._show_memory()
                continue

            self._handle_turn(user_input)


    def _handle_turn(self, user_input: str):
        """
        Process one full conversation turn.
        This is the agent loop — plan → execute → respond.
        """

        start_time = time.time()

        chunks: list[RetrievedChunk] = []

        response = ""

        # add user turn to memory
        self.memory.add_user_turn(user_input)

        # ── PLAN ─────────────────────────────────────────
        decision = self.planner.decide(
            user_query = user_input,
            memory     = self.memory,
            iteration  = 0
        )

        # ── EXECUTE ──────────────────────────────────────

        if decision.action == "retrieve":

            # use rewritten query if planner provided one
            search_query = decision.rewritten_query or user_input

            chunks = self.retriever.retrieve(search_query)

            if chunks:
                response = generate_answer(user_input, chunks, self.memory)

            else:
                # nothing found — loop back and refuse
                response = (
                    "I searched the corpus but couldn't find relevant papers "
                    "on this topic. The corpus covers cs.AI papers from the "
                    "last 90 days — this topic may not be well represented."
                )

        elif decision.action == "clarify":

            response = decision.clarifying_question

        elif decision.action == "tool":

            tool_result = self.tools.execute(
                decision.tool_name,
                decision.tool_input
            )

            if tool_result.success:

                response = self.tools.summarize_tool_result(
                    tool_result,
                    user_input
                )

            else:
                # tool failed — fall back to retrieval
                console.print(
                    "[yellow]  Tool failed — falling back to corpus retrieval[/yellow]"
                )

                chunks = self.retriever.retrieve(user_input)

                response = generate_answer(user_input, chunks, self.memory) \
                    if chunks else "I couldn't find relevant information."

        elif decision.action == "refuse":

            response = (
                f"I can't help with that. {decision.refusal_reason}\n\n"
                f"I'm specialized in answering questions about recent AI "
                f"research papers. Try asking about machine learning methods, "
                f"specific papers, or AI research topics."
            )

        elif decision.action == "answer":

            response = generate_direct_answer(user_input, self.memory)

        # ── RESPOND ──────────────────────────────────────

        latency_ms = int((time.time() - start_time) * 1000)

        # print the response
        console.print(f"\n[bold green]Agent:[/bold green]")

        console.print(Panel(response, expand=False))

        # print citations if we retrieved
        if chunks:

            console.print("[dim]Sources:[/dim]")

            for i, c in enumerate(chunks[:3]):

                console.print(
                    f"  [dim][{i+1}] {c.title[:60]} "
                    f"({c.published}) — score: {c.score:.3f}[/dim]"
                )

        console.print(
            f"[dim]  ⏱ {latency_ms}ms | action: {decision.action}[/dim]\n"
        )

        # ── SAVE ─────────────────────────────────────────

        self.memory.add_assistant_turn(
            response,
            action=decision.action
        )

        self.memory.record_episode(
            question     = user_input,
            action_taken = decision.action,
            papers_cited = [c.title for c in chunks],
            was_answered = decision.action not in ["refuse", "clarify"]
        )

        self.logger.log(
            user_input,
            decision,
            chunks,
            response,
            latency_ms
        )


    def _show_trace(self):
        """Show the full decision trace of the last turn."""

        if not self.logger.entries:
            console.print("[dim]No turns yet.[/dim]")
            return

        last = self.logger.entries[-1]

        console.print(Rule("Last Decision Trace"))

        console.print(f"[bold]Query:[/bold] {last['query']}")

        console.print(f"[bold]Action:[/bold] {last['decision']['action']}")

        console.print(
            f"[bold]Reasoning:[/bold] "
            f"{last['decision']['reasoning']}"
        )

        console.print(
            f"[bold]Confidence:[/bold] "
            f"{last['decision']['confidence']:.0%}"
        )

        if last["retrieved_chunks"]:

            console.print(
                f"\n[bold]Retrieved "
                f"{len(last['retrieved_chunks'])} chunks:[/bold]"
            )

            for c in last["retrieved_chunks"]:

                console.print(
                    f"  • {c['title'][:55]} "
                    f"(score: {c['score']:.3f})"
                )

        console.print(
            f"\n[bold]Latency:[/bold] {last['latency_ms']}ms"
        )

        console.print(Rule())


    def _show_memory(self):
        """Show current memory state."""

        console.print(Rule("Memory State"))

        console.print(self.memory.get_full_context(""))

        console.print(Rule())


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":

    agent = AgentCLI()

    agent.run()