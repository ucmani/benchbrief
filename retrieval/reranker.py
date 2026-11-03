"""
Cross-encoder reranker. Retrieve wide (bi-encoder, fast, coarse), then
rescore each (query, chunk) pair jointly and keep the best few.

Lazy singleton model, same pattern as indexing/embed.py.
Install: pip install sentence-transformers  (already present for embeddings)
"""

from typing import Optional

RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL_NAME)
    return _model


def _text_of(candidate: dict, text_key: str) -> str:
    """Chunk text may sit at the top level or inside a nested 'metadata' dict."""
    value = candidate.get(text_key) or (candidate.get("metadata") or {}).get(text_key)
    if not value:
        raise KeyError(f"No '{text_key}' in result. Keys: {list(candidate)}")
    return value


def rerank(query: str, candidates: list[dict], top_k: int = 5,
           text_key: str = "text", max_chars: int = 2000) -> list[dict]:
    """
    Return the top_k candidates sorted by cross-encoder score.
    Each returned dict is a copy with an added `rerank_score`; the original
    retrieval score is left untouched so both can be logged and compared.
    """
    if not candidates:
        return []
    pairs = [(query, _text_of(c, text_key)[:max_chars]) for c in candidates]
    scores = _get_model().predict(pairs)
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]


def search_with_rerank(search_fn, query: str, top_k: int = 5,
                       fetch_k: int = 25, use_rerank: bool = True,
                       text_key: str = "text", **search_kwargs) -> list[dict]:
    """
    Wrap any search(query, top_k=..., **kw) -> list[dict].
    use_rerank=False gives the baseline, so hit@k can be compared on/off.
    """
    if not use_rerank:
        return search_fn(query, top_k=top_k, **search_kwargs)
    candidates = search_fn(query, top_k=fetch_k, **search_kwargs)
    return rerank(query, candidates, top_k=top_k, text_key=text_key)