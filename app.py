"""
Single-process BenchBrief for hosting (for example a Hugging Face Space): the FastAPI
endpoints AND a Gradio chat page, served together on one port.

  /            Gradio chat UI (Ask a question / Analyze a case)
  /analyze, /chat, /info, /metrics   the same API as before

Run locally:  python3 app.py        -> http://127.0.0.1:7860
Streamlit (chat_app.py) still works as before; this file does not replace it.

Environment:
  GEN_LLM               provider chain, e.g. "gemini,openrouter" (as for the API)
  DEMO_LIMIT_PER_HOUR   max requests per visitor per hour; 0 = unlimited (default 0)
  PORT                  default 7860
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque

import gradio as gr
import uvicorn

from api.main import app as api_app, get_llm, llm_specs
from generation.chat import answer_question
from generation.pipeline import run_pipeline
from utils.llm import describe_chain

logger = logging.getLogger(__name__)

ASK, ANALYZE = "Ask a question", "Analyze a case"
SOURCES_MARK = "\n\n---\n**Precedent used**"      # everything after this is display-only
LIMIT = int(os.environ.get("DEMO_LIMIT_PER_HOUR", "0"))
_visits: dict[str, deque] = defaultdict(deque)


def _over_limit(request: gr.Request | None) -> bool:
    """Simple per-visitor limit so a public demo can't drain the free-tier LLM quota."""
    if not LIMIT:
        return False
    who = (request.client.host if request and request.client else "unknown")
    now, window = time.time(), _visits[who]
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= LIMIT:
        return True
    window.append(now)
    return False


def _text(content) -> str:
    """Chat history content is a string, or a list of parts in multimodal chats."""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def _history_for_model(history: list[dict], message: str) -> list[dict]:
    turns = [{"role": m["role"], "content": _text(m["content"]).split(SOURCES_MARK)[0]}
             for m in history]
    return turns + [{"role": "user", "content": message}]


def _sources_block(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = []
    for i, s in enumerate(sources, start=1):
        label = s.get("case_name") or f"Opinion {s.get('opinion_id')}"
        snippet = (s.get("text") or "").replace("\n", " ")[:260]
        lines.append(f"{i}. **{label}**: {snippet}...")
    return SOURCES_MARK + " (" + str(len(sources)) + ")\n" + "\n".join(lines)


def respond(message: str, history: list[dict], mode: str, top_k: int,
            request: gr.Request = None) -> str:
    if _over_limit(request):
        return (f"Demo limit reached ({LIMIT} requests per hour per visitor). "
                "This keeps the free-tier model quota available for everyone. Please try later.")
    try:
        if mode == ANALYZE:
            result = run_pipeline(current_case_text=message, llm_call_fn=get_llm(), top_k=int(top_k))
            text = (f"### Case brief\n{result['case_brief']}\n\n"
                    f"### Reasoning paths for your consideration\n{result['recommendation']}")
            return text + _sources_block(result["precedent_chunks"])
        result = answer_question(_history_for_model(history, message), get_llm(), top_k=int(top_k))
        return result["answer"] + _sources_block(result["sources"])
    except Exception as e:                      # show a readable error instead of a stack trace
        logger.exception("request failed")
        return f"Sorry, that request failed: {e}"


def _warm_up() -> None:
    """Load the embedding model and reranker once at startup so the first visitor isn't slow."""
    try:
        from retrieval.hybrid_search import search
        from retrieval.reranker import search_with_rerank
        search_with_rerank(search, "warm up", top_k=1, fetch_k=2)
        logger.info("models warmed up")
    except Exception:
        logger.exception("warm-up failed (the first request will load the models instead)")


with gr.Blocks(title="BenchBrief") as demo:
    gr.Markdown(
        "# BenchBrief\n"
        "Decision support for judges: summary, cited precedent and reasoning paths over a small "
        "set of N.D. California opinions. **It never gives a verdict, and it is a research aid, "
        "not legal advice.**")
    gr.ChatInterface(
        respond,
        additional_inputs=[
            gr.Radio([ASK, ANALYZE], value=ASK, label="Mode",
                     info="Ask: questions about precedent. Analyze: paste a case to get a brief."),
            gr.Slider(1, 12, value=8, step=1, label="Precedent passages (top_k)"),
        ],
        fill_height=False,
    )
    gr.Markdown(f"Model: {' → '.join(describe_chain(llm_specs()))}")

demo.queue(default_concurrency_limit=2, max_size=20)
app = gr.mount_gradio_app(api_app, demo, path="/")     # API routes registered earlier keep priority

if __name__ == "__main__":
    threading.Thread(target=_warm_up, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
