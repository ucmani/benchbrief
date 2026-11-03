from retrieval.hybrid_search import search
from generation.prompts.summarize import build_summarize_prompt
from generation.prompts.recommend import build_recommend_prompt
import logging
import time

from observability.metrics import REQUEST_LATENCY, REQUEST_COUNT

logger = logging.getLogger(__name__)

def run_pipeline(current_case_text: str, llm_call_fn, top_k: int = 5, judge: str | None = None) -> dict:
    logger.info(f"pipeline started, top_k={top_k}, judge={judge}")

    try:
        with REQUEST_LATENCY.labels(stage="retrieval").time():
            precedent_results = search(current_case_text, top_k=top_k, judge=judge)
        REQUEST_COUNT.labels(stage="retrieval", status="success").inc()
    except Exception:
        REQUEST_COUNT.labels(stage="retrieval", status="error").inc()
        raise
    logger.info(f"retrieved {len(precedent_results)} precedent chunks")

    precedent_chunks = [
        {
            "opinion_id": r["opinion_id"],
            "section_label": r["section_label"],
            "text": r.get("text", ""),
        }
        for r in precedent_results
    ]

    summarize_prompt = build_summarize_prompt(current_case_text, precedent_chunks)
    try:
        with REQUEST_LATENCY.labels(stage="summarize").time():
            case_brief = llm_call_fn(summarize_prompt)
        REQUEST_COUNT.labels(stage="summarize", status="success").inc()
    except Exception:
        REQUEST_COUNT.labels(stage="summarize", status="error").inc()
        raise
    logger.info("case brief generated")

    recommend_prompt = build_recommend_prompt(case_brief)
    try:
        with REQUEST_LATENCY.labels(stage="recommend").time():
            recommendation = llm_call_fn(recommend_prompt)
        REQUEST_COUNT.labels(stage="recommend", status="success").inc()
    except Exception:
        REQUEST_COUNT.labels(stage="recommend", status="error").inc()
        raise
    logger.info("recommendation generated")

    return {
        "case_brief": case_brief,
        "recommendation": recommendation,
        "precedent_chunks": precedent_chunks,
    }