# agent/memory.py

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Config ────────────────────────────────────────────────
MEMORY_DIR          = Path("data/memory")
CONV_WINDOW_SIZE    = 8
SUMMARY_THRESHOLD   = 6
MODEL_NAME          = "openai/gpt-4o-mini"
# ─────────────────────────────────────────────────────────


@dataclass
class Turn:
    """A single conversation turn."""
    role:      str
    content:   str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    action:    str = ""


@dataclass
class EpisodicEntry:
    """
    One past interaction worth remembering.
    Stores the question, what the agent did, and whether it was useful.
    """
    question:        str
    action_taken:    str
    papers_cited:    list[str]
    was_answered:    bool
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat())


class ConversationMemory:
    """
    Sliding window of recent turns + a running summary of older turns.
    """

    def __init__(self):
        self.turns:   list[Turn] = []
        self.summary: str = ""

    def add_turn(self, role: str, content: str, action: str = ""):
        self.turns.append(Turn(role=role, content=content, action=action))

        if len(self.turns) >= CONV_WINDOW_SIZE:
            self._compress()

    def _compress(self):
        """
        Summarize older turns into compact memory.
        """
        cutoff     = len(self.turns) // 2
        old_turns  = self.turns[:cutoff]
        self.turns = self.turns[cutoff:]

        old_text = "\n".join(
            f"{t.role.upper()}: {t.content}"
            for t in old_turns
        )

        prompt = f"""Summarize this conversation history into 3-5 bullet points.
Focus on:
- topics discussed
- papers mentioned
- questions asked
- what was resolved

Be concise.

Previous summary:
{self.summary or "None"}

New turns:
{old_text}

Updated summary:"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            self.summary = response.choices[0].message.content.strip()

        except Exception:
            self.summary += f"\n[Earlier: {old_text[:300]}]"

    def get_context(self) -> str:
        """
        Return formatted memory context.
        """
        parts = []

        if self.summary:
            parts.append(f"[Conversation summary so far]\n{self.summary}")

        if self.turns:
            recent = "\n".join(
                f"{t.role.upper()}: {t.content}"
                for t in self.turns
            )
            parts.append(f"[Recent turns]\n{recent}")

        return "\n\n".join(parts) if parts else "No prior conversation."

    def get_recent_user_turns(self, n: int = 3) -> list[str]:
        """
        Return recent user messages.
        """
        user_turns = [t.content for t in self.turns if t.role == "user"]
        return user_turns[-n:]


class SemanticMemory:
    """
    Stores extracted technical facts.
    """

    def __init__(self):
        self.facts: dict[str, str] = {}

    def extract_and_store(self, text: str):
        """
        Extract entity-fact pairs using OpenRouter.
        """
        prompt = f"""Extract key technical facts from this text about AI research.

Return ONLY valid JSON.

Format:
{{
  "Entity": "One sentence fact"
}}

Return {{}} if nothing useful exists.

Text:
{text[:1000]}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            raw = response.choices[0].message.content.strip()

            raw = raw.replace("```json", "").replace("```", "").strip()

            new_facts = json.loads(raw)

            if isinstance(new_facts, dict):
                self.facts.update(new_facts)

        except Exception:
            pass

    def get_relevant_facts(self, query: str) -> str:
        """
        Return relevant facts.
        """
        if not self.facts:
            return ""

        query_lower = query.lower()

        relevant = {
            k: v for k, v in self.facts.items()
            if k.lower() in query_lower
        }

        if not relevant:
            relevant = self.facts

        lines = [f"- {k}: {v}" for k, v in list(relevant.items())[:10]]

        return "Known facts:\n" + "\n".join(lines)


class EpisodicMemory:
    """
    Records past interactions.
    """

    def __init__(self):
        self.episodes: list[EpisodicEntry] = []

    def record(
        self,
        question: str,
        action_taken: str,
        papers_cited: list[str],
        was_answered: bool
    ):
        self.episodes.append(EpisodicEntry(
            question     = question,
            action_taken = action_taken,
            papers_cited = papers_cited,
            was_answered = was_answered
        ))

    def find_similar(self, query: str) -> EpisodicEntry | None:
        """
        Find similar previous question.
        """
        query_words = set(query.lower().split())

        for episode in reversed(self.episodes):
            ep_words = set(episode.question.lower().split())

            overlap = len(query_words & ep_words)
            union   = len(query_words | ep_words)

            jaccard = overlap / union if union > 0 else 0

            if jaccard > 0.6:
                return episode

        return None

    def get_summary(self) -> str:
        """
        Summarize recent episodes.
        """
        if not self.episodes:
            return ""

        lines = []

        for ep in self.episodes[-5:]:
            status = "answered" if ep.was_answered else "unanswered"

            lines.append(
                f"- Asked: '{ep.question[:60]}' "
                f"→ {ep.action_taken} ({status})"
            )

        return "Recent interactions:\n" + "\n".join(lines)


class AgentMemory:
    """
    Unified memory interface.
    """

    def __init__(self):
        self.conversation = ConversationMemory()
        self.semantic     = SemanticMemory()
        self.episodic     = EpisodicMemory()

    def add_user_turn(self, message: str):
        self.conversation.add_turn("user", message)

    def add_assistant_turn(self, message: str, action: str = ""):
        self.conversation.add_turn(
            "assistant",
            message,
            action=action
        )

        self.semantic.extract_and_store(message)

    def record_episode(
        self,
        question: str,
        action_taken: str,
        papers_cited: list[str],
        was_answered: bool
    ):
        self.episodic.record(
            question,
            action_taken,
            papers_cited,
            was_answered
        )

    def get_full_context(self, query: str) -> str:
        """
        Build complete memory context.
        """
        parts = []

        conv = self.conversation.get_context()

        if conv:
            parts.append(conv)

        facts = self.semantic.get_relevant_facts(query)

        if facts:
            parts.append(facts)

        episodes = self.episodic.get_summary()

        if episodes:
            parts.append(episodes)

        return "\n\n".join(parts)

    def get_query_context(self) -> str:
        """
        Short context for query rewriting.
        """
        recent = self.conversation.get_recent_user_turns(3)

        return " | ".join(recent)


# ── Quick test ───────────────────────────────────────────
if __name__ == "__main__":
    print("Testing AgentMemory...\n")

    mem = AgentMemory()

    mem.add_user_turn(
        "What is LoRA and how does it help with fine-tuning?"
    )

    mem.add_assistant_turn(
        "LoRA (Low-Rank Adaptation) is a fine-tuning technique that inserts "
        "trainable low-rank matrices into transformer layers, reducing the "
        "number of trainable parameters by up to 10,000x.",
        action="retrieve"
    )

    mem.add_user_turn(
        "Does it work for vision models too?"
    )

    mem.add_assistant_turn(
        "Yes, LoRA has been applied to vision transformers (ViT) as well.",
        action="retrieve"
    )

    mem.record_episode(
        question     = "What is LoRA?",
        action_taken = "retrieve",
        papers_cited = ["LoRA: Low-Rank Adaptation of Large Language Models"],
        was_answered = True
    )

    print("=== Full memory context ===\n")
    print(mem.get_full_context("Does LoRA work for vision?"))

    print("\n=== Semantic facts extracted ===\n")

    for k, v in mem.semantic.facts.items():
        print(f"  {k}: {v}")

    print("\n=== Episodic check (similar question) ===\n")

    similar = mem.episodic.find_similar(
        "How does LoRA work for fine-tuning?"
    )

    if similar:
        print(
            f"  Found similar: '{similar.question}' "
            f"→ was answered: {similar.was_answered}"
        )