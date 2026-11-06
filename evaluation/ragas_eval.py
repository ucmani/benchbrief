"""
RAGAS evaluation of the REAL chat path (retrieve + rerank + generate) on the golden set.

For each sampled golden question:
  1. run generation.chat.answer_question with the real generator (GEN_LLM chain)
  2. judge the answer with RAGAS metrics, using a DIFFERENT model (JUDGE_LLM chain):
       faithfulness : is every claim in the answer supported by the retrieved passages?
       correctness  : does the answer agree with the golden reference answer? (FactualCorrectness)
  3. record context_has_evidence: did the retrieved passages contain the golden evidence
     sentence at all? (no LLM; separates "retrieval failed" from "generation failed")

Results append to ragas_results.csv after every question, so a quota stop loses nothing.
Re-running skips questions already scored. A bigger --n extends the same sample.

Run:  GEN_LLM=openrouter JUDGE_LLM=gemini python3 -m evaluation.ragas_eval --n 20
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
from typing import Callable

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import FactualCorrectness, Faithfulness

from evaluation.golden_set import GOLDEN_PATH, _norm
from evaluation.pushgateway_client import push_metrics

RESULTS_PATH = GOLDEN_PATH.with_name("ragas_results.csv")
FIELDS = ["question", "reference_chunk_keys", "answer", "context_has_evidence",
          "faithfulness", "correctness", "note"]


class AdapterJudge(InstructorBaseRagasLLM):
    """
    RAGAS wants an "instructor" LLM that returns a validated pydantic object. Instead of
    relying on each provider's structured-output support (patchy on free models), this
    wraps our own `llm_call_fn(prompt) -> str`: ask for JSON matching the schema, parse,
    validate, retry once. Bonus: it inherits the adapter's call spacing, retries and
    provider fallback, so RAGAS respects free-tier limits.
    """

    def __init__(self, call_fn: Callable[[str], str], retries: int = 1):
        self.call_fn = call_fn
        self.retries = retries

    def generate(self, prompt: str, response_model):
        schema = json.dumps(response_model.model_json_schema())
        full = (f"{prompt}\n\nReturn ONLY one JSON object that matches this JSON schema. "
                f"No prose, no code fences.\nSCHEMA: {schema}")
        error = None
        for _ in range(self.retries + 1):
            hint = f"\n\nYour previous reply was invalid ({error}). Return valid JSON only." if error else ""
            raw = self.call_fn(full + hint)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                if not match:
                    raise ValueError(f"no JSON in reply: {raw[:80]!r}")
                return response_model.model_validate_json(match.group(0))
            except ValueError as e:        # pydantic's ValidationError is a ValueError
                error = str(e)[:200]
        raise ValueError(f"judge returned invalid JSON: {error}")

    async def agenerate(self, prompt: str, response_model):
        return await asyncio.to_thread(self.generate, prompt, response_model)


def _value(result) -> float:
    v = getattr(result, "value", result)
    return float("nan") if v is None else float(v)


async def score_question(row: dict, answer_fn, faithfulness, correctness) -> dict:
    out = {"question": row["question"], "reference_chunk_keys": row["reference_chunk_keys"],
           "answer": "", "context_has_evidence": "", "faithfulness": "", "correctness": "", "note": ""}
    result = answer_fn([{"role": "user", "content": row["question"]}])
    answer, contexts = result["answer"], result["contexts"]
    out["answer"] = answer.replace("\n", " ")[:1500]
    out["context_has_evidence"] = int(_norm(row["evidence"]) in _norm(" ".join(contexts)))
    if not answer.strip():
        out["note"] = "empty answer"
        return out
    notes = []
    for name, metric, kwargs in (
            ("faithfulness", faithfulness, dict(user_input=row["question"], response=answer,
                                                retrieved_contexts=contexts)),
            ("correctness", correctness, dict(response=answer, reference=row["reference_answer"]))):
        if metric is None:
            continue
        try:
            out[name] = _value(await metric.ascore(**kwargs))
        except Exception as e:
            if "daily quota" in str(e):
                raise                      # stop the whole run; nothing else will work today
            out[name] = float("nan")
            notes.append(f"{name} failed: {str(e)[:80]}")
    out["note"] = "; ".join(notes)
    return out


def _read_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _scored(r: dict) -> bool:
    return r["faithfulness"] not in ("", "nan")


def _append(row: dict) -> None:
    new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _mean(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows if r[key] not in ("", "nan") and not math.isnan(float(r[key]))]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def summarize(sample_questions: set[str]) -> dict:
    rows = [r for r in _read_results() if r["question"] in sample_questions and _scored(r)]
    metrics = {}
    print(f"\n=== RAGAS summary over {len(rows)} scored questions ===")
    for key in ("faithfulness", "correctness"):
        mean, n = _mean(rows, key)
        if mean is not None:
            metrics[key] = mean
            print(f"{key:<22}{mean:.3f}   (n={n})")
    if rows:
        metrics["context_has_evidence_rate"] = sum(int(r["context_has_evidence"]) for r in rows) / len(rows)
        print(f"{'context_has_evidence':<22}{metrics['context_has_evidence_rate']:.3f}   "
              f"(retrieved passages contained the golden evidence)")
        for flag, label in ((1, "evidence retrieved"), (0, "evidence NOT retrieved")):
            part = [r for r in rows if int(r["context_has_evidence"]) == flag]
            for key in ("faithfulness", "correctness"):
                mean, n = _mean(part, key)
                if mean is not None:
                    print(f"   {label:<24}{key:<14}{mean:.3f}   (n={n})")
    metrics["n_scored"] = len(rows)
    return metrics


def run(n: int = 20, with_correctness: bool = True, answer_fn=None, judge_call=None) -> None:
    from utils.llm import describe_chain, make_llm_chain
    gen_spec = os.environ.get("GEN_LLM", "gemini,openrouter")
    judge_spec = os.environ.get("JUDGE_LLM", "gemini")
    if answer_fn is None:
        from generation.chat import answer_question
        generator = make_llm_chain(gen_spec)
        answer_fn = lambda messages: answer_question(messages, generator, top_k=8)
        print("generator:", describe_chain(gen_spec))
    if judge_call is None:
        judge_call = make_llm_chain(judge_spec, temperature=0.0)
        print("judge:    ", describe_chain(judge_spec))
        if gen_spec == judge_spec:
            print("WARNING: generator and judge are the same model; scores will be optimistic.")
    judge = AdapterJudge(judge_call)
    faithfulness = Faithfulness(llm=judge)
    correctness = FactualCorrectness(llm=judge) if with_correctness else None

    with open(GOLDEN_PATH, newline="", encoding="utf-8") as f:
        golden = [r for r in csv.DictReader(f) if r["reference_chunk_keys"].strip()]
    random.Random(7).shuffle(golden)          # fixed order: a bigger --n extends the same sample
    sample = golden[:n]
    done = {r["question"] for r in _read_results() if _scored(r)}
    todo = [r for r in sample if r["question"] not in done]
    print(f"{len(sample)} sampled, {len(sample) - len(todo)} already scored, {len(todo)} to run\n")

    async def loop():
        for i, row in enumerate(todo, start=1):
            try:
                out = await score_question(row, answer_fn, faithfulness, correctness)
            except RuntimeError as e:
                print(f"  STOP: {e}. Re-run later; scored rows are saved.")
                return
            _append(out)
            print(f"  [{i}/{len(todo)}] faith={out['faithfulness'] if out['faithfulness'] == '' else round(out['faithfulness'], 2)} "
                  f"correct={out['correctness'] if out['correctness'] == '' else round(out['correctness'], 2)} "
                  f"evidence_in_ctx={out['context_has_evidence']} {out['note']}")

    asyncio.run(loop())
    metrics = summarize({r["question"] for r in sample})
    if metrics.get("n_scored"):
        push_metrics("benchbrief_ragas_eval", [{"labels": {"mode": "chat"}, "metrics": metrics}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="questions to evaluate (LLM calls scale with this)")
    parser.add_argument("--no-correctness", action="store_true", help="faithfulness only (fewer judge calls)")
    args = parser.parse_args()
    run(args.n, with_correctness=not args.no_correctness)