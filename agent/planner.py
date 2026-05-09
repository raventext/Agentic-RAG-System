# agent/planner.py

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from agent.memory import AgentMemory

load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# ── The 5 possible actions ────────────────────────────────
ACTIONS = {
    "retrieve":  "Search the corpus of AI research papers for relevant information",
    "clarify":   "Ask the user a clarifying question before proceeding",
    "tool":      "Call an external tool (web search or arXiv API)",
    "refuse":    "Decline to answer — question is out of scope or unanswerable",
    "answer":    "Answer directly from memory/context without retrieving"
}
# ─────────────────────────────────────────────────────────


@dataclass
class PlannerDecision:
    """
    Structured output from the planner.
    Every field is used downstream by cli.py.
    """
    action:              str
    reasoning:           str
    rewritten_query:     str  = ""
    clarifying_question: str  = ""
    refusal_reason:      str  = ""
    tool_name:           str  = ""
    tool_input:          str  = ""
    confidence:          float = 0.0


# ── System prompt for the planner ────────────────────────
PLANNER_SYSTEM_PROMPT = """You are the planning module of an agentic RAG system 
built over a corpus of recent AI research papers (cs.AI, last 90 days).

Your ONLY job is to decide what action to take next. You do NOT answer the question yourself.

Available actions:
1. retrieve   — search the paper corpus. Use when the question is about AI research topics, 
                specific papers, methods, authors, or results likely in the corpus.
2. clarify    — ask the user a clarifying question. Use when the question is too vague,
                uses unclear pronouns, or could mean multiple things.
3. tool       — call an external tool. Use when the question needs live/current info 
                beyond the corpus, or real-time arXiv search.
4. refuse     — decline to answer. Use when the question is completely outside AI research
                (cooking, sports, personal advice, etc.) OR when the corpus clearly 
                won't contain the answer.
5. answer     — answer directly from context/memory. Use when the answer is already 
                in the conversation, or the question is a simple follow-up.

Decision rules:
- DEFAULT to retrieve for any AI research question
- Use clarify when there are unresolved pronouns ("it", "they", "this") with no clear referent
- Use refuse for non-AI topics — do not try to answer them
- Use answer for follow-ups like "can you summarize that?" or "what did you just say?"
- Use tool when user explicitly asks for latest/recent info or asks about a paper not in corpus

You must respond with ONLY a valid JSON object — no explanation, no markdown, no preamble.

JSON schema:
{
  "action": "<one of: retrieve | clarify | tool | refuse | answer>",
  "reasoning": "<1-2 sentences explaining your choice>",
  "rewritten_query": "<optimized search query — REQUIRED if action is retrieve, else empty>",
  "clarifying_question": "<your question to the user — REQUIRED if action is clarify, else empty>",
  "refusal_reason": "<why you are refusing — REQUIRED if action is refuse, else empty>",
  "tool_name": "<web_search or arxiv_search — REQUIRED if action is tool, else empty>",
  "tool_input": "<the search query for the tool — REQUIRED if action is tool, else empty>",
  "confidence": <float between 0.0 and 1.0>
}
"""


class Planner:

    def __init__(self):
        self.decision_history: list[PlannerDecision] = []

    def decide(
        self,
        user_query: str,
        memory: AgentMemory,
        iteration: int = 0
    ) -> PlannerDecision:

        # safety valve
        if iteration >= 3:
            return PlannerDecision(
                action="answer",
                reasoning="Max iterations reached — answering with available context",
                confidence=0.5
            )

        memory_context = memory.get_full_context(user_query)

        # episodic memory
        similar_episode = memory.episodic.find_similar(user_query)
        episodic_hint = ""

        if similar_episode:
            episodic_hint = (
                f"\nNote: Similar question was asked before. "
                f"Action taken: {similar_episode.action_taken}. "
                f"Was answered: {similar_episode.was_answered}."
            )

        prompt = f"""Current user question: {user_query}

Memory context:
{memory_context}
{episodic_hint}

Loop iteration: {iteration} (0 = first attempt)

Decide what action to take. Return ONLY valid JSON."""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": PLANNER_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            raw = response.choices[0].message.content.strip()

            # remove markdown fences
            raw = raw.replace("```json", "").replace("```", "").strip()

            data = json.loads(raw)

            decision = PlannerDecision(
                action=data.get("action", "retrieve"),
                reasoning=data.get("reasoning", ""),
                rewritten_query=data.get("rewritten_query", ""),
                clarifying_question=data.get("clarifying_question", ""),
                refusal_reason=data.get("refusal_reason", ""),
                tool_name=data.get("tool_name", ""),
                tool_input=data.get("tool_input", ""),
                confidence=float(data.get("confidence", 0.7))
            )

        except Exception as e:
            print(f"  [planner] Parse error: {e} — defaulting to retrieve")

            decision = PlannerDecision(
                action="retrieve",
                reasoning="Fallback: planner output could not be parsed",
                rewritten_query=user_query,
                confidence=0.5
            )

        # validate
        if decision.action not in ACTIONS:
            decision.action = "retrieve"
            decision.reasoning = (
                f"Unknown action '{decision.action}' — defaulting to retrieve"
            )

        # log
        self._log_decision(user_query, decision)
        self.decision_history.append(decision)

        return decision

    def _log_decision(self, query: str, decision: PlannerDecision):

        action_colors = {
            "retrieve": "🔍",
            "clarify":  "❓",
            "tool":     "🛠️",
            "refuse":   "🚫",
            "answer":   "✅"
        }

        icon = action_colors.get(decision.action, "•")

        print(f"\n{'─'*50}")
        print(f"  {icon}  PLANNER DECISION: {decision.action.upper()}")
        print(f"  Reasoning  : {decision.reasoning}")
        print(f"  Confidence : {decision.confidence:.0%}")

        if decision.rewritten_query:
            print(f"  Query      : {decision.rewritten_query}")

        if decision.clarifying_question:
            print(f"  Clarify    : {decision.clarifying_question}")

        if decision.refusal_reason:
            print(f"  Refusal    : {decision.refusal_reason}")

        if decision.tool_name:
            print(f"  Tool       : {decision.tool_name} → '{decision.tool_input}'")

        print(f"{'─'*50}\n")


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":

    from agent.memory import AgentMemory

    planner = Planner()
    memory  = AgentMemory()

    test_cases = [
        "What techniques do recent papers use to reduce hallucinations in LLMs?",
        "What did they find about it?",
        "What is the best recipe for chocolate cake?",
        "Can you summarize what you just told me?",
        "Find me the latest paper published today on diffusion models",
    ]

    print("\n" + "="*50)
    print("  PLANNER TEST — 5 decision cases")
    print("="*50)

    for i, query in enumerate(test_cases):

        print(f"\nTest {i+1}: \"{query}\"")

        decision = planner.decide(query, memory)

        memory.add_user_turn(query)

        memory.add_assistant_turn(
            f"[test response {i+1}]",
            action=decision.action
        )