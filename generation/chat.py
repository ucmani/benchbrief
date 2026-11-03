"""
Follow-up Q&A over the corpus: retrieve (+ rerank) precedent for the latest
question and answer from those sources only, with the chat history for context.

Same dependency-injection pattern as pipeline.py: the LLM is passed in.
"""

import logging
import os

from observability.metrics import REQUEST_COUNT, REQUEST_LATENCY
from retrieval.hybrid_search import search

logger = logging.getLogger(__name__)

USE_RERANK = os.environ.get("USE_RERANK", "1") == "1"
USE_BM25 = os.environ.get("USE_BM25", "0") == "1"   # off until the eval shows it helps
MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_CHARS = 3000
MAX_SOURCE_CHARS = 1500

SYSTEM = """You are BenchBrief, a research assistant for a judge.
Answer ONLY from the numbered SOURCES (N.D. Cal. federal opinions) and the earlier conversation.
- Start with a direct answer to the question: yes, no, or "the retrieved sources do not say".
  Then give the support. Never present tangential material as if it answered the question.
- Cite every claim like [Opinion 123, source 2].
- If the retrieved sources do not contain the answer, say so plainly (the corpus may hold it; you only see these passages). Do not use outside knowledge.
- Keep separate holdings separate. Do not combine reasoning from one issue into another;
  say which issue each point is about.
- When a source quotes or cites another authority, name that authority (case, court, year)
  next to the claim, and say which court it is from.
- Quote legal standards in quotation marks exactly as written. Never paraphrase a quoted rule.
- Never say how the judge should rule. Lay out the competing considerations and the
  reasoning paths the precedent supports, and leave the decision to the judge."""


def _timed(stage: str, fn, *args, **kwargs):
    """Run fn with the same latency/count metrics the main pipeline records."""
    try:
        with REQUEST_LATENCY.labels(stage=stage).time():
            out = fn(*args, **kwargs)
        REQUEST_COUNT.labels(stage=stage, status="success").inc()
        return out
    except Exception:
        REQUEST_COUNT.labels(stage=stage, status="error").inc()
        raise


def retrieval_query(messages: list[dict], max_chars_per_turn: int = 500) -> str:
    """
    A follow-up like "does the same standard apply to X?" is meaningless on its own.
    Search with the last two user turns joined, each trimmed (an earlier turn may be
    a whole pasted case).
    """
    user_turns = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(t[:max_chars_per_turn] for t in user_turns[-2:])


def _retrieve(question: str, top_k: int) -> list[dict]:
    search_fn = search
    if USE_BM25:   # dense + keyword search merged with Reciprocal Rank Fusion
        from retrieval.bm25_search import fused_search
        search_fn = fused_search
    if USE_RERANK:
        from retrieval.reranker import search_with_rerank
        return search_with_rerank(search_fn, question, top_k=top_k, fetch_k=25)
    return search_fn(question, top_k=top_k)


def _label(r: dict) -> str:
    return r.get("case_name") or f"Opinion {r.get('opinion_id')}"


def build_chat_prompt(messages: list[dict], sources: list[dict]) -> str:
    question = messages[-1]["content"]
    history = "\n".join(
        f"{'Judge' if m['role'] == 'user' else 'Assistant'}: {m['content'][:MAX_MESSAGE_CHARS]}"
        for m in messages[:-1][-MAX_HISTORY_MESSAGES:])
    blocks = "\n\n".join(
        f"[source {i}] {_label(r)} ({r.get('section_label', '')})\n{r.get('text', '')[:MAX_SOURCE_CHARS]}"
        for i, r in enumerate(sources, start=1))
    return (f"{SYSTEM}\n\nCONVERSATION SO FAR:\n{history or '(none)'}\n\n"
            f"SOURCES:\n{blocks}\n\nQUESTION: {question}\n\nANSWER:")


def answer_question(messages: list[dict], llm_call_fn, top_k: int = 8) -> dict:
    question = messages[-1]["content"]
    logger.info(f"chat started, history={len(messages) - 1}, top_k={top_k}")
    sources = _timed("chat_retrieval", _retrieve, retrieval_query(messages), top_k)
    answer = _timed("chat_generate", llm_call_fn, build_chat_prompt(messages, sources))
    logger.info("chat answer generated")
    return {"answer": answer,
            # exactly what the model was shown (sources below are trimmed for display);
            # the RAGAS eval judges the answer against THIS, not the trimmed copy
            "contexts": [r.get("text", "")[:MAX_SOURCE_CHARS] for r in sources],
            "sources": [{"opinion_id": r.get("opinion_id"), "section_label": r.get("section_label"),
                         "case_name": r.get("case_name"), "text": r.get("text", "")[:500]}
                        for r in sources]}
