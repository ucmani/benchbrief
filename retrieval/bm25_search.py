"""
Keyword (BM25) search and Reciprocal Rank Fusion with the dense (embedding) search.

Why: embeddings capture meaning but blur exact tokens such as "February 18, 2011",
"28 U.S.C. 1927", a statute number or a party name. BM25 scores exactly those tokens.
Fusing the two rankings catches questions either one alone would miss.

Pieces:
  tokenize        text -> lowercase alphanumeric tokens (stopwords removed)
  BM25Index       inverted index + the BM25 formula, in memory (fine for thousands of chunks)
  bm25_search     keyword-only search, same return shape as the dense search
  fused_search    dense + BM25, merged with Reciprocal Rank Fusion; drop-in for `search`

At much larger scale replace BM25Index with Postgres full-text search (you already run
Supabase), Elasticsearch/OpenSearch, or Pinecone sparse vectors. The fusion code stays the same.
"""

import json
import math
import re
from collections import Counter, defaultdict

from config import PROCESSED_DIR

CHUNKS_PATH = PROCESSED_DIR / "chunks_fixed_size.jsonl"
STRATEGY = "fixed_size"

K1 = 1.5       # how fast repeated occurrences of a term stop adding score (term-frequency saturation)
B = 0.75       # how strongly long chunks are penalized (length normalization)
RRF_K = 60     # RRF damping constant; 60 is the value from the original paper
POOL = 30      # candidates taken from EACH retriever before fusing

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her his if in into is it its "
    "of on or she that the their them there these they this to was were which who will with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def result_key(result: dict) -> str:
    """Chunk key of a search result (the Pinecone vector id is the chunk_key)."""
    meta = result.get("metadata") or {}
    for source in (result, meta):
        for name in ("chunk_key", "id"):
            if source.get(name):
                return str(source[name])
    opinion_id = result.get("opinion_id", meta.get("opinion_id"))
    index = result.get("chunk_index", meta.get("chunk_index"))
    if opinion_id is not None and index is not None:
        return f"{opinion_id}:{STRATEGY}:{index}"
    raise KeyError(f"Cannot identify chunk in result. Keys: {list(result)}")


class BM25Index:
    """
    score(query, chunk) = sum over query terms t of
        idf(t) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * chunk_len / avg_len))
    idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)):  rare terms count for more.
    """

    def __init__(self, docs: list[dict], k1: float = K1, b: float = B):
        self.docs, self.k1, self.b, self.N = docs, k1, b, len(docs)
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)   # term -> {doc index: count}
        for i, doc in enumerate(docs):
            tokens = tokenize(doc["text"])
            self.doc_len.append(len(tokens))
            for term, count in Counter(tokens).items():
                self.postings[term][i] = count
        self.avg_len = (sum(self.doc_len) / self.N) if self.N else 1.0

    def idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for i, tf in posting.items():
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{**self.docs[i], "bm25_score": score} for i, score in ranked]


_index: BM25Index | None = None


def get_index() -> BM25Index:
    """Build the index once per process from the same chunk file that was loaded into Pinecone/Supabase."""
    global _index
    if _index is None:
        docs = []
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                meta = record.get("metadata") or {}
                opinion_id = meta.get("opinion_id", record.get("opinion_id"))
                docs.append({
                    "chunk_key": f"{opinion_id}:{STRATEGY}:{record.get('chunk_index')}",
                    "text": record["text"],
                    "opinion_id": opinion_id,
                    "chunk_index": record.get("chunk_index"),
                    "case_name": meta.get("case_name"),
                    "section_label": record.get("section_label"),
                })
        _index = BM25Index(docs)
    return _index


def bm25_search(query: str, top_k: int = 10, **_ignored) -> list[dict]:
    """Keyword-only search. Metadata filters (judge, section_label) are not supported here."""
    return get_index().search(query, top_k=top_k)


def fused_search(query: str, top_k: int = 5, dense_fn=None, **dense_kwargs) -> list[dict]:
    """
    Dense + BM25, merged with Reciprocal Rank Fusion: every list gives a chunk
    1 / (RRF_K + rank), and the scores add up. A chunk ranked well by BOTH retrievers wins;
    a chunk ranked #1 by only one still gets in. No score calibration needed, because RRF
    uses ranks, not the (incomparable) cosine and BM25 scores.

    Same call shape as `search`, so `search_with_rerank(fused_search, ...)` works unchanged.
    If filters are passed (judge, section_label) they apply to the dense side only, so we
    skip fusion rather than return keyword hits that ignore the filter.
    """
    if dense_fn is None:
        from retrieval.hybrid_search import search as dense_fn
    pool = max(POOL, top_k)
    dense = dense_fn(query, top_k=pool, **dense_kwargs)
    if dense_kwargs:
        return dense[:top_k]
    sparse = get_index().search(query, top_k=pool)

    fused: dict[str, float] = defaultdict(float)
    by_key: dict[str, dict] = {}
    for results in (dense, sparse):               # dense first, so its richer dict is kept
        for rank, result in enumerate(results, start=1):
            key = result_key(result)
            fused[key] += 1.0 / (RRF_K + rank)
            by_key.setdefault(key, result)
    ordered = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [{**by_key[key], "rrf_score": fused[key]} for key in ordered]
