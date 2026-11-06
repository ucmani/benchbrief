"""
Golden-set generator (Langfuse "RAG Dataset Generation" pattern).

Loop over chunks -> ask an LLM to draft ONE question + answer + verbatim
evidence per chunk -> auto-filter junk -> write a candidate CSV -> a human
flips `verified` -> finalize() writes the golden CSV the RAGAS eval loads.

Every row carries `reference_chunk_keys`, so retrieval can be scored
(did the right chunk come back?) with no LLM judge at all.
"""

import csv
import json
import random
import re
from pathlib import Path
from typing import Callable

from config import PROCESSED_DIR, PROJECT_ROOT
from utils.jsonl import read_jsonl

CHUNKS_PATH = PROCESSED_DIR / "chunks_fixed_size.jsonl"
DATASETS_DIR = PROJECT_ROOT / "evaluation" / "test_data" / "datasets"
CANDIDATES_PATH = DATASETS_DIR / "golden_candidates.csv"
GOLDEN_PATH = DATASETS_DIR / "golden_v2.csv"            # v1 was the 18-question pilot
REMOVED_PATH = DATASETS_DIR / "golden_removed.csv"      # rows dropped for quality; never re-added
ATTEMPTED_PATH = DATASETS_DIR / "golden_attempted.txt"  # chunk keys already sent to the drafter

MIN_CHUNK_CHARS = 600
PER_OPINION_CAP = 15      # max chunks drafted per opinion, so one long opinion can't dominate
FIELDS = ["question", "reference_answer", "evidence", "reference_chunk_keys",
          "opinion_id", "kind", "verified", "judge_scores", "judge_reason"]

# Rotated per chunk so 150 questions aren't 150 copies of "What did the court hold...?"
STYLES = [
    "the court's holding or ruling on a specific legal issue",
    "the legal standard or test the court applies",
    "a specific fact, date, or procedural event, naming the parties or subject matter",
    "why the court reached its conclusion (its reasoning)",
    "what relief, remedy, or order the court granted or denied, and on what basis",
]

PROMPT = """You write exam questions for a legal research assistant.
Read the court-opinion excerpt and write ONE question that this excerpt alone answers.
Preferred question type: {style}. If the excerpt does not fit that type, write the best question it does answer.
Rules:
- The question must be answerable by someone who has NOT seen the excerpt. Name the specific legal issue plus at
  least one identifying detail (a party, statute, technology, or procedural posture) so it could only come from
  this kind of dispute. Never say "the excerpt", "the passage" or "this case", and never refer to "the plaintiffs",
  "the court" or "the accused" without saying who or what they are.
- Do NOT ask about dates, counts, or events that have no identifying context.
- The answer must be 1-3 sentences, using only facts in the excerpt.
- "evidence" must be a sentence copied VERBATIM from the excerpt that supports the answer.
Return ONLY JSON: {{"question": "...", "answer": "...", "evidence": "..."}}

EXCERPT:
{chunk}
"""


def _get(chunk: dict, *names):
    """First non-empty value among top-level keys, then the nested 'metadata' dict."""
    for source in (chunk, chunk.get("metadata") or {}):
        for name in names:
            if source.get(name) not in (None, ""):
                return source[name]
    return None


def normalize(chunk: dict) -> dict:
    """Return the chunk with guaranteed `opinion_id` and `chunk_key`."""
    opinion_id = _get(chunk, "opinion_id", "source_opinion_id", "opinion", "doc_id")
    index = _get(chunk, "chunk_index", "index", "position")
    strategy = _get(chunk, "strategy", "strategy_name", "chunk_strategy") or "fixed_size"
    key = _get(chunk, "chunk_key", "id", "chunk_id")
    if key is None and opinion_id is not None and index is not None:
        key = f"{opinion_id}:{strategy}:{index}"  # same recipe as _chunk_row
    if opinion_id is None and key:
        opinion_id = str(key).split(":")[0]       # the key starts with the opinion id
    if opinion_id is None or key is None:
        raise KeyError(f"Cannot find opinion_id/chunk_key. Keys present: {list(chunk)}")
    return {**chunk, "opinion_id": opinion_id, "chunk_key": key,
            "text": _get(chunk, "text", "chunk_text", "content") or ""}


def _load_attempted() -> set[str]:
    """Chunks already tried: the attempts log, plus any chunk that already has a candidate row."""
    tried = set(ATTEMPTED_PATH.read_text().split()) if ATTEMPTED_PATH.exists() else set()
    return tried | {r["reference_chunk_keys"] for r in _read_csv(CANDIDATES_PATH)}


def sample_chunks(per_opinion: int = PER_OPINION_CAP, seed: int = 7) -> list[dict]:
    """
    Long, not-yet-attempted chunks, capped per opinion and ORDERED ROUND-ROBIN across
    opinions (op A, op B, op C, op A, ...). Ordering matters: if the run stops early
    (daily quota), whatever was drafted so far is still spread over all opinions.
    """
    rng = random.Random(seed)
    attempted = _load_attempted()
    by_opinion: dict = {}
    for chunk in map(normalize, read_jsonl(CHUNKS_PATH)):
        if len(chunk["text"]) >= MIN_CHUNK_CHARS and chunk["chunk_key"] not in attempted:
            by_opinion.setdefault(chunk["opinion_id"], []).append(chunk)
    queues = []
    for chunks in by_opinion.values():
        rng.shuffle(chunks)
        queues.append(chunks[:per_opinion])
    ordered = []
    for rank in range(max((len(q) for q in queues), default=0)):
        ordered.extend(q[rank] for q in queues if rank < len(q))
    return ordered


def _norm(s: str) -> str:
    """Compare on letters and digits only, so curly vs straight quotes,
    dashes, ellipses and spacing differences don't cause false rejections."""
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def draft_item(chunk: dict, llm_call_fn: Callable[[str], str]) -> tuple[dict | None, str]:
    """Ask the LLM for one Q/A. Returns (row, "ok") or (None, reason it was rejected)."""
    style = STYLES[sum(map(ord, chunk["chunk_key"])) % len(STYLES)]  # deterministic per chunk
    raw = llm_call_fn(PROMPT.format(style=style, chunk=chunk["text"]))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, f"no JSON in output: {raw[:80]!r}"
    try:
        item = json.loads(match.group(0))
        question, answer, evidence = item["question"], item["answer"], item["evidence"]
    except (json.JSONDecodeError, KeyError) as e:
        return None, f"bad JSON ({type(e).__name__}): {match.group(0)[:80]!r}"
    if _norm(evidence) not in _norm(chunk["text"]):
        return None, f"evidence not in chunk: {evidence[:80]!r}"
    if re.search(r"\b(excerpt|passage|this case|the text)\b", question, re.I):
        return None, f"question not standalone: {question[:80]!r}"
    return {"question": question, "reference_answer": answer, "evidence": evidence,
            "reference_chunk_keys": chunk["chunk_key"], "opinion_id": chunk["opinion_id"],
            "kind": "single_chunk", "verified": "", "judge_scores": "", "judge_reason": ""}, "ok"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_candidates(llm_call_fn: Callable[[str], str], target: int = 200,
                        per_opinion: int = PER_OPINION_CAP) -> int:
    """
    Draft until the candidate file holds `target` rows. APPENDS to the candidate CSV and
    records every chunk it tried, so it is safe to stop (Ctrl-C, daily quota) and re-run.
    Candidates the judge already scored are untouched.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(CANDIDATES_PATH)
    seen_q = {_norm(r["question"]) for r in rows}
    chunks = sample_chunks(per_opinion)
    print(f"{len(rows)} candidates on disk, {len(chunks)} untried chunks, target {target}")
    for i, chunk in enumerate(chunks, start=1):
        if len(rows) >= target:
            break
        try:
            row, reason = draft_item(chunk, llm_call_fn)
        except RuntimeError as e:
            if "daily quota" in str(e):      # no point trying the remaining chunks today
                print(f"  STOP: {e}. Re-run tomorrow (or switch provider); progress is saved.")
                break
            print(f"  [{i}/{len(chunks)}] skipped {chunk['chunk_key']}: {e}")  # retry next run
            continue
        with open(ATTEMPTED_PATH, "a") as f:      # tried (kept OR rejected): don't pay twice
            f.write(chunk["chunk_key"] + "\n")
        if row and _norm(row["question"]) in seen_q:
            row, reason = None, "duplicate question"
        print(f"  [{i}/{len(chunks)}] {'kept' if row else 'REJECTED'} {chunk['chunk_key']}"
              f"{'' if row else ' -> ' + reason}   (total {len(rows) + bool(row)})")
        if row:
            rows.append(row)
            seen_q.add(_norm(row["question"]))
            _save(rows)                            # save every row; a crash loses nothing
    print(f"{len(rows)} candidates -> {CANDIDATES_PATH}")
    return len(rows)


def finalize() -> int:
    """Write golden_v2.csv = judge-passed (verified=y) rows, minus anything in golden_removed.csv."""
    removed = {_norm(r["question"]) for r in _read_csv(REMOVED_PATH)}
    kept = [r for r in _read_csv(CANDIDATES_PATH)
            if r["verified"].strip().lower() == "y" and _norm(r["question"]) not in removed]
    with open(GOLDEN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    per_op: dict = {}
    for r in kept:
        per_op[r["opinion_id"]] = per_op.get(r["opinion_id"], 0) + 1
    print(f"{len(kept)} verified rows -> {GOLDEN_PATH}")
    print("per opinion:", dict(sorted(per_op.items(), key=lambda kv: -kv[1])))
    return len(kept)


def audit(n: int = 15, seed: int = 1) -> None:
    """Print a random sample of judge-passed rows for YOU to read: the human check on the judge."""
    rows = [r for r in _read_csv(CANDIDATES_PATH) if r["verified"] == "y"]
    for r in random.Random(seed).sample(rows, min(n, len(rows))):
        print(f"\nQ: {r['question']}\nA: {r['reference_answer']}\nE: {r['evidence']}\n"
              f"   [{r['reference_chunk_keys']}] scores={r['judge_scores']}")


JUDGE_PROMPT = """You are a strict reviewer of a question-answer dataset built from court opinions.
Given an EXCERPT, a QUESTION and a proposed ANSWER, score each criterion 1-5:
- answerable: the excerpt contains what is needed to answer the question
- correct: the answer is fully supported by the excerpt, with nothing invented
- standalone: the question is understandable without the excerpt (names the legal issue)
- specific: the question has one clear answer and is not trivial or vague
Return ONLY JSON: {{"answerable": n, "correct": n, "standalone": n, "specific": n, "reason": "one sentence"}}

EXCERPT:
{chunk}

QUESTION: {question}
ANSWER: {answer}
"""

PASS_MIN = 4  # every criterion must score at least this


def judge_item(row: dict, chunk_text: str, judge_fn: Callable[[str], str]) -> dict:
    """Score one candidate with an LLM judge; return scores plus a pass/fail flag."""
    raw = judge_fn(JUDGE_PROMPT.format(chunk=chunk_text, question=row["question"],
                                       answer=row["reference_answer"]))
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        d = json.loads(match.group(0))
        scores = {k: int(d[k]) for k in ("answerable", "correct", "standalone", "specific")}
    except (AttributeError, json.JSONDecodeError, KeyError, ValueError):
        return {"passed": False, "scores": {}, "reason": "unparseable judge output"}
    return {"passed": min(scores.values()) >= PASS_MIN, "scores": scores,
            "reason": str(d.get("reason", ""))}


BATCH_JUDGE_PROMPT = """You are a strict reviewer of a question-answer dataset built from court opinions.
Below are numbered items. Each has an EXCERPT, a QUESTION and a proposed ANSWER.
For EACH item, score 1-5:
- answerable: the excerpt contains what is needed to answer the question
- correct: the answer is fully supported by the excerpt, with nothing invented
- standalone: score 1-2 if the question names no party, statute, technology or specific legal issue,
  so that it could be asking about hundreds of different cases (e.g. "Which claims were dismissed on a given date?")
- specific: the question has one clear answer and is not trivial or vague
Return ONLY a JSON array, one object per item, in this form:
[{{"id": 1, "answerable": n, "correct": n, "standalone": n, "specific": n, "reason": "one sentence"}}, ...]

{items}
"""


def judge_batch(rows: list[dict], texts: list[str], judge_fn: Callable[[str], str]) -> list[dict]:
    """Judge several candidates in ONE call (free tiers allow very few requests/day)."""
    items = "\n\n".join(
        f"### ITEM {i}\nEXCERPT: {t}\nQUESTION: {r['question']}\nANSWER: {r['reference_answer']}"
        for i, (r, t) in enumerate(zip(rows, texts), start=1))
    raw = judge_fn(BATCH_JUDGE_PROMPT.format(items=items))
    fail = {"passed": False, "scores": {}, "reason": "judge gave no usable verdict"}
    results = [dict(fail) for _ in rows]
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return results
    try:
        verdicts = json.loads(match.group(0))
    except json.JSONDecodeError:
        return results
    for v in verdicts:
        try:
            idx = int(v["id"]) - 1
            scores = {k: int(v[k]) for k in ("answerable", "correct", "standalone", "specific")}
            results[idx] = {"passed": min(scores.values()) >= PASS_MIN, "scores": scores,
                            "reason": str(v.get("reason", ""))}
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    return results


def _save(rows: list[dict]) -> None:
    with open(CANDIDATES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def judge_candidates(judge_fn: Callable[[str], str], batch_size: int = 8) -> int:
    """
    Fill `verified` (y/n), `judge_scores` and `judge_reason` for every candidate,
    `batch_size` rows per LLM call. The judge should be a DIFFERENT model from
    the drafter, or it will agree with its own mistakes.
    """
    chunk_text = {c["chunk_key"]: c["text"] for c in map(normalize, read_jsonl(CHUNKS_PATH))}
    with open(CANDIDATES_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def _done(r):  # a real verdict exists (not a failed/unusable judge call)
        return r["verified"] in ("y", "n") and not r["judge_reason"].startswith(
            ("judge call failed", "judge gave no usable verdict"))

    todo = [r for r in rows if not _done(r)]
    print(f"{len(rows) - len(todo)} already judged, {len(todo)} to judge")
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        texts = [chunk_text.get(r["reference_chunk_keys"], "") for r in batch]
        try:
            results = judge_batch(batch, texts, judge_fn)
        except RuntimeError as e:
            if "daily quota" in str(e):   # every remaining batch would fail too
                print(f"  STOP: {e}. Re-run later; judged rows are saved.")
                _save(rows)
                break
            results = [{"passed": False, "scores": {}, "reason": f"judge call failed: {e}"}
                       for _ in batch]
        for row, result in zip(batch, results):
            row["verified"] = "y" if result["passed"] else "n"
            row["judge_scores"] = json.dumps(result["scores"])
            row["judge_reason"] = result["reason"]
        print(f"  judged {min(start + batch_size, len(todo))}/{len(todo)}")
        _save(rows)  # save after every batch so progress survives a crash
    passed = sum(r["verified"] == "y" for r in rows)
    print(f"{passed}/{len(rows)} passed the judge -> {CANDIDATES_PATH}")
    return passed


if __name__ == "__main__":
    # python3 -m evaluation.golden_set draft openrouter [target=200] [per_opinion=15]
    # python3 -m evaluation.golden_set judge gemini
    # python3 -m evaluation.golden_set audit
    # python3 -m evaluation.golden_set finalize
    import sys
    from utils.llm import make_llm_call, make_llm_chain
    step = sys.argv[1]
    if step == "draft":
        target = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        cap = int(sys.argv[4]) if len(sys.argv) > 4 else PER_OPINION_CAP
        generate_candidates(make_llm_call(sys.argv[2]), target, cap)
    elif step == "judge":
        judge_candidates(make_llm_chain(sys.argv[2], temperature=0.0))
    elif step == "audit":
        audit()
    elif step == "finalize":
        finalize()
