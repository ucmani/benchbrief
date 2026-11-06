"""
Retrieval evaluation against the golden set. No LLM involved.

For each golden question we ask: did a chunk listed in `reference_chunk_keys`
come back in the top-k? Metrics:
  hit@k          : fraction of questions with a correct chunk in the top k (k = 1, 3, 5, 8)
  opinion_hit@5  : a chunk from the RIGHT CASE is in the top 5 (right opinion, any chunk)
  MRR            : mean of 1/rank of the first correct chunk, counting ranks <= 5 (0 otherwise)

Modes (pick any; default is baseline + rerank):
  baseline       dense (embedding) search only
  rerank         dense, then cross-encoder rerank of the top 25
  bm25           keyword search only (no network, instant)
  hybrid         dense + BM25 merged with Reciprocal Rank Fusion
  hybrid_rerank  hybrid candidates, then cross-encoder rerank
After the table, each other mode is compared with the FIRST mode listed on a paired basis
(McNemar exact test): how many questions did it rescue vs. break?

Run: python3 -m evaluation.retrieval_eval                      # baseline rerank
     python3 -m evaluation.retrieval_eval baseline hybrid hybrid_rerank
"""

import csv
import sys
import time
from math import comb

from evaluation.golden_set import GOLDEN_PATH
from retrieval.bm25_search import bm25_search, fused_search, result_key as _result_key
from retrieval.hybrid_search import search
from retrieval.reranker import search_with_rerank

K = 5              # the headline k
K_MAX = 8          # chat sends 8 chunks to the LLM, so hit@8 is the generation-relevant recall
FETCH_K = 25
MISSES_PATH = GOLDEN_PATH.with_name("retrieval_misses.csv")
RANKS_PATH = GOLDEN_PATH.with_name("retrieval_ranks.csv")

# mode name -> (search function, rerank the candidates?)
MODES = {
    "baseline": (search, False),
    "rerank": (search, True),
    "bm25": (bm25_search, False),
    "hybrid": (fused_search, False),
    "hybrid_rerank": (fused_search, True),
}


def _opinion_of(chunk_key: str) -> str:
    return str(chunk_key).split(":")[0]       # keys look like "{opinion_id}:{strategy}:{index}"


def _first_hit_rank(results: list[dict], reference_keys: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if _result_key(result) in reference_keys:
            return rank
    return None


def evaluate(mode: str, rows: list[dict]) -> tuple[dict, list[dict], list[int | None]]:
    search_fn, use_rerank = MODES[mode]
    ranks, opinion_hits, misses, elapsed = [], [], [], 0.0
    for row in rows:
        reference = {k.strip() for k in row["reference_chunk_keys"].split(",") if k.strip()}
        reference_opinions = {_opinion_of(k) for k in reference}
        start = time.perf_counter()
        results = search_with_rerank(search_fn, row["question"], top_k=K_MAX,
                                     fetch_k=FETCH_K, use_rerank=use_rerank)
        elapsed += time.perf_counter() - start
        rank = _first_hit_rank(results, reference)
        ranks.append(rank)
        opinion_hits.append(any(_opinion_of(_result_key(r)) in reference_opinions
                                for r in results[:K]))
        if rank is None or rank > K:
            misses.append({"mode": mode, "question": row["question"],
                           "expected": row["reference_chunk_keys"],
                           "got": " | ".join(_result_key(r) for r in results[:K])})
    n = len(rows)
    within = lambda k: sum(r is not None and r <= k for r in ranks) / n
    metrics = {
        "hit@1": within(1),
        "hit@3": within(3),
        f"hit@{K}": within(K),
        f"hit@{K_MAX}": within(K_MAX),
        f"opinion_hit@{K}": sum(opinion_hits) / n,
        "MRR": sum(1 / r for r in ranks if r and r <= K) / n,
        "avg_latency_s": elapsed / n,
    }
    return metrics, misses, ranks


def mcnemar(base_ranks, other_ranks, k: int) -> tuple[int, int, float]:
    """
    Paired comparison at cutoff k. rescued = missed by the base mode, hit by the other;
    broken = the opposite. Exact two-sided p-value under "rescues and breaks are equally likely".
    """
    hit = lambda r: r is not None and r <= k
    rescued = sum(not hit(b) and hit(o) for b, o in zip(base_ranks, other_ranks))
    broken = sum(hit(b) and not hit(o) for b, o in zip(base_ranks, other_ranks))
    n = rescued + broken
    if n == 0:
        return rescued, broken, 1.0
    tail = sum(comb(n, i) for i in range(min(rescued, broken) + 1)) / 2 ** n
    return rescued, broken, min(1.0, 2 * tail)


def run(modes: list[str] | None = None) -> None:
    modes = modes or ["baseline", "rerank"]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"Unknown mode(s) {unknown}. Choose from {list(MODES)}")
    with open(GOLDEN_PATH, newline="", encoding="utf-8") as f:
        # unanswerable questions have no reference chunk: they can't be scored here
        rows = [r for r in csv.DictReader(f) if r["reference_chunk_keys"].strip()]
    print(f"Evaluating {len(rows)} questions, modes {modes} (k={K}, rerank fetches {FETCH_K})")

    # Load the embedding model, the reranker and the BM25 index BEFORE timing, so the first
    # question doesn't pay the loading cost in the latency average.
    warm_rerank = any(MODES[m][1] for m in modes)
    for m in modes:
        search_with_rerank(MODES[m][0], rows[0]["question"], top_k=K, fetch_k=FETCH_K,
                           use_rerank=warm_rerank)
    print("models warmed up\n")

    all_misses, results, all_ranks = [], {}, {}
    for mode in modes:
        results[mode], misses, all_ranks[mode] = evaluate(mode, rows)
        all_misses.extend(misses)

    print(f"{'metric':<16}" + "".join(f"{m:>15}" for m in modes))
    for metric in results[modes[0]]:
        print(f"{metric:<16}" + "".join(f"{results[m][metric]:>15.3f}" for m in modes))

    if len(modes) > 1:
        print(f"\nPaired comparison vs {modes[0]} (McNemar exact test):")
        for other in modes[1:]:
            for k in (1, K):
                rescued, broken, p = mcnemar(all_ranks[modes[0]], all_ranks[other], k)
                print(f"  {other:<14} hit@{k}: rescued {rescued:>3}, broken {broken:>3}, p = {p:.4f}")

    from evaluation.pushgateway_client import push_metrics   # no-op unless PUSHGATEWAY_URL is set
    push_metrics("benchbrief_retrieval_eval",
                 [{"labels": {"mode": name}, "metrics": m} for name, m in results.items()])

    with open(RANKS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "expected"] + [f"{m}_rank" for m in modes])
        for i, row in enumerate(rows):
            writer.writerow([row["question"], row["reference_chunk_keys"]]
                            + [all_ranks[m][i] or "" for m in modes])
    print(f"\nper-question ranks -> {RANKS_PATH}  (blank = not in top {K_MAX})")

    if all_misses:
        with open(MISSES_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["mode", "question", "expected", "got"])
            writer.writeheader()
            writer.writerows(all_misses)
        print(f"{len(all_misses)} misses written to {MISSES_PATH}")


if __name__ == "__main__":
    run(sys.argv[1:])
