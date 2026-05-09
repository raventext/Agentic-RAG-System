import os
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from urllib.error import HTTPError

from openai import OpenAI

from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Config ────────────────────────────────────────────────
ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_ARXIV_RESULTS = 3
# ─────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Structured output from any tool call."""
    tool_name: str
    query: str
    success: bool
    content: str
    sources: list[str]
    error: str = ""


class ToolExecutor:
    """
    Executes external tool calls requested by the planner.

    Tools available:
      - arxiv_search
      - web_search
    """

    def __init__(self):
        pass

    def execute(self, tool_name: str, tool_input: str) -> ToolResult:
        """
        Route to the correct tool and return a ToolResult.
        Always returns a ToolResult — never raises.
        """

        print(f"\n[tools] Executing: {tool_name}('{tool_input}')")

        if tool_name == "arxiv_search":
            return self._arxiv_search(tool_input)

        elif tool_name == "web_search":
            return self._web_search(tool_input)

        else:
            return ToolResult(
                tool_name=tool_name,
                query=tool_input,
                success=False,
                content=f"Unknown tool '{tool_name}'.",
                sources=[],
                error=f"Tool '{tool_name}' is not registered."
            )

    def _arxiv_search(self, query: str) -> ToolResult:
        """
        Search arXiv live using their public API.
        """

        try:

            params = urllib.parse.urlencode({
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": MAX_ARXIV_RESULTS,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            })

            url = f"{ARXIV_API_URL}?{params}"

            print(f"  [tools] Fetching: {url}")

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "agentic-rag-research-bot/1.0 "
                        "(mailto:your_email@example.com)"
                    )
                }
            )

            retries = 5
            xml_data = None

            for attempt in range(retries):

                try:

                    wait_time = 5 + (attempt * 5)

                    print(
                        f"  [tools] Waiting {wait_time}s before request..."
                    )

                    time.sleep(wait_time)

                    with urllib.request.urlopen(
                        request,
                        timeout=60
                    ) as resp:

                        xml_data = resp.read().decode("utf-8")

                    break

                except HTTPError as e:

                    if e.code == 429 and attempt < retries - 1:

                        backoff = 10 * (attempt + 1)

                        print(
                            f"  [tools] Rate limited by arXiv. "
                            f"Retrying in {backoff}s..."
                        )

                        time.sleep(backoff)

                    else:
                        raise

                except Exception as e:

                    if attempt < retries - 1:

                        backoff = 5 * (attempt + 1)

                        print(
                            f"  [tools] Temporary error: {e}. "
                            f"Retrying in {backoff}s..."
                        )

                        time.sleep(backoff)

                    else:
                        raise

            if not xml_data:
                raise Exception("No response received from arXiv.")

            root = ET.fromstring(xml_data)

            ns = {
                "atom": "http://www.w3.org/2005/Atom"
            }

            entries = root.findall("atom:entry", ns)

            if not entries:
                return ToolResult(
                    tool_name="arxiv_search",
                    query=query,
                    success=False,
                    content="No papers found on arXiv for this query.",
                    sources=[]
                )

            results = []
            sources = []

            for entry in entries:

                title = entry.find(
                    "atom:title",
                    ns
                ).text.strip()

                summary = entry.find(
                    "atom:summary",
                    ns
                ).text.strip()

                published = entry.find(
                    "atom:published",
                    ns
                ).text[:10]

                link = entry.find(
                    "atom:id",
                    ns
                ).text.strip()

                authors = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                ][:3]

                results.append(
                    f"Title: {title}\n"
                    f"Authors: {', '.join(authors)}\n"
                    f"Published: {published}\n"
                    f"Abstract: {summary[:300]}...\n"
                    f"URL: {link}"
                )

                sources.append(link)

            content = (
                f"Found {len(results)} papers on arXiv:\n\n"
                + "\n\n---\n\n".join(results)
            )

            print(f"  [tools] arXiv returned {len(results)} papers")

            return ToolResult(
                tool_name="arxiv_search",
                query=query,
                success=True,
                content=content,
                sources=sources
            )

        except Exception as e:

            print(f"  [tools] arXiv ERROR: {e}")

            return ToolResult(
                tool_name="arxiv_search",
                query=query,
                success=False,
                content=f"arXiv search failed: {str(e)}",
                sources=[],
                error=str(e)
            )

    def _web_search(self, query: str) -> ToolResult:
        """
        Web search fallback using OpenRouter.
        """

        try:

            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
Answer this question accurately and concisely.

Question:
{query}

If relevant, mention recent developments,
papers, or known sources.
"""
                    }
                ],
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            print("  [tools] OpenRouter response generated")

            return ToolResult(
                tool_name="web_search",
                query=query,
                success=True,
                content=content,
                sources=[]
            )

        except Exception as e:

            print(f"  [tools] OpenRouter ERROR: {e}")

            return ToolResult(
                tool_name="web_search",
                query=query,
                success=False,
                content="Web search failed.",
                sources=[],
                error=str(e)
            )

    def summarize_tool_result(
        self,
        result: ToolResult,
        original_query: str
    ) -> str:
        """
        Summarize raw tool output into a concise answer.
        """

        prompt = f"""
A user asked:

"{original_query}"

You used the tool '{result.tool_name}'
and got this result:

{result.content[:2000]}

Summarize the relevant parts into a clear
concise answer in 3-5 sentences.

Include paper titles or source names
where relevant.

If the result doesn't answer the question,
say so clearly.
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

        except Exception:

            return result.content[:500]


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":

    executor = ToolExecutor()

    print("\n" + "=" * 50)
    print("TEST 1 — arXiv search")
    print("=" * 50)

    result = executor.execute(
        "arxiv_search",
        "mixture of experts"
    )

    print(f"\nSuccess: {result.success}")
    print(f"Content preview:\n{result.content[:400]}")
    print(f"Sources: {result.sources[:2]}")

    print("\n" + "=" * 50)
    print("TEST 2 — Summarize tool result")
    print("=" * 50)

    summary = executor.summarize_tool_result(
        result,
        "What are the latest papers on mixture of experts?"
    )

    print(f"\nSummary:\n{summary}")