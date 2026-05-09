
# eval/harness.py

import json
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

from agent.memory    import AgentMemory
from agent.planner   import Planner
from agent.retriever import HybridRetriever, RetrievedChunk
from agent.tools     import ToolExecutor

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

EVAL_LOGS_DIR = Path("logs/eval")


# ── Test case definition ──────────────────────────────────
@dataclass
class EvalCase:
    """
    A single evaluation test case.

    expected_action : what the planner SHOULD decide
    must_contain    : keywords the response MUST include
    must_not_contain: keywords that would indicate a wrong answer
    description     : why this test case exists
    """
    id:               str
    question:         str
    expected_action:  str
    must_contain:     list[str]
    must_not_contain: list[str]
    description:      str
    category:         str   # retrieve / clarify / refuse / edge_case


@dataclass
class EvalResult:
    """Result of running one test case."""
    case_id:          str
    question:         str
    expected_action:  str
    actual_action:    str
    action_correct:   bool
    response:         str
    contains_checks:  dict[str, bool]
    missing_checks:   dict[str, bool]
    content_score:    float
    overall_pass:     bool
    latency_ms:       int
    reasoning:        str
    notes:            str = ""


# ── The 12 test cases ─────────────────────────────────────
EVAL_CASES: list[EvalCase] = [

    EvalCase(
        id              = "R01",
        question        = "What techniques have been proposed in recent papers to reduce hallucinations in large language models?",
        expected_action = "retrieve",
        must_contain    = ["hallucin", "retriev", "grounding"],
        must_not_contain= ["chocolate", "recipe", "I don't know anything"],
        description     = "Core AI research question — clear retrieve case",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R02",
        question        = "How do mixture of experts models differ from dense transformer models in terms of efficiency?",
        expected_action = "retrieve",
        must_contain    = ["expert", "sparse", "parameter"],
        must_not_contain= ["cannot answer", "outside my scope"],
        description     = "Technical comparison question — should retrieve and compare",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R03",
        question        = "What are the main findings of recent papers on reinforcement learning from human feedback?",
        expected_action = "retrieve",
        must_contain    = ["RLHF", "reward", "human"],
        must_not_contain= ["I have no information", "not in corpus"],
        description     = "RLHF is a hot topic — should definitely be in corpus",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R04",
        question        = "Explain the concept of chain of thought prompting based on recent research",
        expected_action = "retrieve",
        must_contain    = ["chain", "reasoning", "prompt"],
        must_not_contain= ["I cannot", "outside domain"],
        description     = "Well known AI technique — expect strong retrieval",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R05",
        question        = "What methods do recent papers propose for improving LLM inference speed?",
        expected_action = "retrieve",
        must_contain    = ["inference", "latency", "speed"],
        must_not_contain= ["no papers found", "I don't know"],
        description     = "Practical efficiency question — good corpus coverage expected",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R06",
        question        = "How has the concept of attention been modified or improved in recent transformer papers?",
        expected_action = "retrieve",
        must_contain    = ["attention", "transformer"],
        must_not_contain= ["cannot", "outside"],
        description     = "Core architecture question — very likely in corpus",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R07",
        question        = "What evaluation benchmarks are most commonly used for testing LLM reasoning abilities?",
        expected_action = "retrieve",
        must_contain    = ["benchmark", "evaluat", "reasoning"],
        must_not_contain= ["I don't have", "cannot help"],
        description     = "Meta question about evaluation — should appear in methods sections",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "R08",
        question        = "What does recent literature say about the relationship between model size and performance?",
        expected_action = "retrieve",
        must_contain    = ["scale", "parameter", "performance"],
        must_not_contain= ["unrelated", "not applicable"],
        description     = "Scaling laws question — foundational topic in cs.AI",
        category        = "retrieve"
    ),

    EvalCase(
        id              = "C01",
        question        = "What did they find about it in the paper?",
        expected_action = "clarify",
        must_contain    = ["clarif", "which", "?"],
        must_not_contain= ["hallucin", "transformer", "attention"],
        description     = "Classic ambiguous question",
        category        = "clarify"
    ),

    EvalCase(
        id              = "C02",
        question        = "Is the new method better?",
        expected_action = "clarify",
        must_contain    = ["which", "method", "clarif", "?"],
        must_not_contain= ["yes it is", "no it is not"],
        description     = "Vague comparative",
        category        = "clarify"
    ),

    EvalCase(
        id              = "RF01",
        question        = "What is the best Python web framework for building a REST API?",
        expected_action = "refuse",
        must_contain    = ["outside", "AI research", "speciali"],
        must_not_contain= ["Django", "FastAPI", "Flask", "here is how"],
        description     = "Software engineering question",
        category        = "refuse"
    ),

    EvalCase(
        id              = "RF02",
        question        = "Who won the football World Cup in 2022?",
        expected_action = "refuse",
        must_contain    = ["outside", "research", "AI"],
        must_not_contain= ["Argentina", "France", "Mbappe", "Messi"],
        description     = "Sports question",
        category        = "refuse"
    ),
]


# ── Evaluator ─────────────────────────────────────────────
class EvalHarness:

    def __init__(self, skip_retriever: bool = False):

        self.memory    = AgentMemory()
        self.planner   = Planner()
        self.tools     = ToolExecutor()
        self.results:  list[EvalResult] = []

        if not skip_retriever:
            print("[eval] Loading retriever...")
            self.retriever = HybridRetriever()
        else:
            self.retriever = None
            print("[eval] Retriever skipped (planner-only mode)")


    def _run_case(self, case: EvalCase) -> EvalResult:

        start = time.time()

        memory = AgentMemory()
        memory.add_user_turn(case.question)

        decision = self.planner.decide(case.question, memory)
        action   = decision.action

        response = ""
        chunks: list[RetrievedChunk] = []

        if action == "retrieve" and self.retriever:

            query  = decision.rewritten_query or case.question
            chunks = self.retriever.retrieve(query, log=False)

            if chunks:
                response = self._generate_answer(case.question, chunks)
            else:
                response = "No relevant papers found in the corpus."

        elif action == "clarify":

            response = decision.clarifying_question

        elif action == "refuse":

            response = (
                f"{decision.refusal_reason} "
                f"I specialize in AI research papers."
            )

        elif action == "answer":

            response = self._direct_answer(case.question, memory)

        elif action == "tool":

            result = self.tools.execute(
                decision.tool_name,
                decision.tool_input
            )

            response = (
                self.tools.summarize_tool_result(result, case.question)
                if result.success
                else "Tool call failed."
            )

        # ── Score ─────────────────────────────────────────

        action_correct = (action == case.expected_action)

        response_lower = response.lower()

        contains_checks = {
            kw: kw.lower() in response_lower
            for kw in case.must_contain
        }

        missing_checks = {
            kw: kw.lower() not in response_lower
            for kw in case.must_not_contain
        }

        all_checks = (
            list(contains_checks.values())
            + list(missing_checks.values())
        )

        content_score = (
            sum(all_checks) / len(all_checks)
            if all_checks else 1.0
        )

        overall_pass = (
            action_correct and content_score >= 0.5
        )

        latency_ms = int((time.time() - start) * 1000)

        return EvalResult(
            case_id          = case.id,
            question         = case.question,
            expected_action  = case.expected_action,
            actual_action    = action,
            action_correct   = action_correct,
            response         = response,
            contains_checks  = contains_checks,
            missing_checks   = missing_checks,
            content_score    = content_score,
            overall_pass     = overall_pass,
            latency_ms       = latency_ms,
            reasoning        = decision.reasoning
        )


    def _generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> str:

        context = "\n\n".join(
            f"[{c.title}]\n{c.text}"
            for c in chunks
        )

        prompt = f"""
Answer this question using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        try:

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

        except Exception as e:

            return f"Generation failed: {str(e)}"


    def _direct_answer(
        self,
        query: str,
        memory: AgentMemory
    ) -> str:

        prompt = f"Answer briefly: {query}"

        try:

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

        except Exception as e:

            return f"Direct answer failed: {str(e)}"


    def run_all(self) -> dict:

        EVAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  EVALUATION HARNESS — {len(EVAL_CASES)} test cases")
        print(f"{'='*60}\n")

        for i, case in enumerate(EVAL_CASES):

            print(
                f"[{i+1:02d}/{len(EVAL_CASES)}] "
                f"{case.id} — {case.description[:50]}"
            )

            result = self._run_case(case)

            self.results.append(result)

            status = (
                "✓ PASS"
                if result.overall_pass
                else "✗ FAIL"
            )

            print(
                f"         {status}  |  "
                f"action: {result.expected_action} → "
                f"{result.actual_action}  |  "
                f"content: {result.content_score:.0%}  |  "
                f"{result.latency_ms}ms\n"
            )

        return self._print_report()


    def _print_report(self) -> dict:

        total       = len(self.results)
        passed      = sum(
            1 for r in self.results if r.overall_pass
        )

        action_acc  = (
            sum(1 for r in self.results if r.action_correct)
            / total
        )

        avg_content = (
            sum(r.content_score for r in self.results)
            / total
        )

        categories = {}

        for r, c in zip(self.results, EVAL_CASES):

            cat = c.category

            if cat not in categories:
                categories[cat] = {
                    "pass": 0,
                    "total": 0
                }

            categories[cat]["total"] += 1

            if r.overall_pass:
                categories[cat]["pass"] += 1

        print(f"\n{'='*60}")
        print(f"  RESULTS SUMMARY")
        print(f"{'='*60}")

        print(
            f"  Overall pass rate  : "
            f"{passed}/{total} ({passed/total:.0%})"
        )

        print(
            f"  Action accuracy    : "
            f"{action_acc:.0%}"
        )

        print(
            f"  Avg content score  : "
            f"{avg_content:.0%}"
        )

        print(f"\n  By category:")

        for cat, scores in categories.items():

            pct = scores["pass"] / scores["total"]

            bar = (
                "█" * int(pct * 10)
                + "░" * (10 - int(pct * 10))
            )

            print(
                f"    {cat:<12} "
                f"{bar} "
                f"{scores['pass']}/{scores['total']}"
            )

        failures = [
            r for r in self.results
            if not r.overall_pass
        ]

        if failures:

            print(f"\n  Failed cases:")

            for r in failures:

                print(
                    f"    ✗ {r.case_id} — "
                    f"expected {r.expected_action}, "
                    f"got {r.actual_action}"
                )

                print(
                    f"      Reasoning: "
                    f"{r.reasoning[:80]}"
                )

        print(f"{'='*60}\n")

        report = {
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "action_accuracy": action_acc,
            "avg_content_score": avg_content,
            "categories": categories,
            "results": [
                {
                    "id": r.case_id,
                    "question": r.question,
                    "expected": r.expected_action,
                    "actual": r.actual_action,
                    "pass": r.overall_pass,
                    "content_score": r.content_score,
                    "latency_ms": r.latency_ms,
                    "reasoning": r.reasoning,
                    "response": r.response[:300]
                }
                for r in self.results
            ]
        }

        log_path = (
            EVAL_LOGS_DIR
            / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(log_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"  Full results saved → {log_path}")

        return report


    def run_ablation(self):

        if not self.retriever:
            print("[ablation] Retriever not loaded — skip")
            return

        retrieve_cases = [
            c for c in EVAL_CASES
            if c.category == "retrieve"
        ]

        print(f"\n{'='*60}")
        print(f"  ABLATION — Reranking ON vs OFF")
        print(f"  ({len(retrieve_cases)} retrieve cases)")
        print(f"{'='*60}\n")

        with_rerank = []
        without_rerank = []

        for case in retrieve_cases:

            query = case.question

            chunks_with = self.retriever.retrieve(
                query,
                log=False
            )

            semantic = self.retriever._semantic_search(query, 20)

            bm25 = self.retriever._bm25_search(query, 20)

            merged = self.retriever._reciprocal_rank_fusion(
                semantic,
                bm25
            )

            chunks_without = merged[:5]

            def keyword_hit(chunks, keywords):

                if not chunks:
                    return 0.0

                combined = " ".join(
                    c.text.lower()
                    for c in chunks
                )

                hits = sum(
                    1 for kw in keywords
                    if kw.lower() in combined
                )

                return (
                    hits / len(keywords)
                    if keywords else 0.0
                )

            with_rerank.append(
                keyword_hit(
                    chunks_with,
                    case.must_contain
                )
            )

            without_rerank.append(
                keyword_hit(
                    chunks_without,
                    case.must_contain
                )
            )

            print(
                f"  {case.id}  "
                f"with rerank: {with_rerank[-1]:.0%}  "
                f"without: {without_rerank[-1]:.0%}"
            )

        avg_with = (
            sum(with_rerank)
            / len(with_rerank)
        )

        avg_without = (
            sum(without_rerank)
            / len(without_rerank)
        )

        delta = avg_with - avg_without

        print(f"\n  Average WITH reranking    : {avg_with:.0%}")
        print(f"  Average WITHOUT reranking : {avg_without:.0%}")
        print(f"  Delta                     : +{delta:.0%}")

        print(
            f"\n  "
            f"{'Reranking helps ✓' if delta > 0 else 'No difference detected'}"
        )

        print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip retriever — test planner decisions only"
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study (rerank ON vs OFF)"
    )

    args = parser.parse_args()

    harness = EvalHarness(
        skip_retriever=args.fast
    )

    harness.run_all()

    if args.ablation:
        harness.run_ablation()