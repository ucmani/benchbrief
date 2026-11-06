"""
Push evaluation results to a Prometheus Pushgateway so Grafana can chart quality.

Why a Pushgateway: Prometheus normally SCRAPES long-running services. An eval is a batch
job that finishes in minutes, so Prometheus would never catch it. The job pushes its final
numbers to the Pushgateway; the gateway keeps them; Prometheus scrapes the gateway.

Set PUSHGATEWAY_URL (e.g. http://localhost:9091) to turn pushing on. Unset = skipped quietly.
"""

import os
import re

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def _metric_name(name: str) -> str:
    """'hit@5' -> 'benchbrief_eval_hit_at_5' (Prometheus names allow only [a-zA-Z0-9_:])."""
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", name.replace("@", "_at_")).strip("_")
    return f"benchbrief_eval_{clean}"


def push_metrics(job: str, series: list[dict], url: str | None = None) -> bool:
    """
    series = [{"labels": {"mode": "rerank"}, "metrics": {"hit@5": 0.85, "MRR": 0.72}}, ...]
    One push replaces everything previously pushed under the same `job`.
    Returns True if pushed. Never raises: a monitoring hiccup must not fail an eval run.
    """
    url = url or os.environ.get("PUSHGATEWAY_URL")
    if not url:
        print("PUSHGATEWAY_URL not set: skipping push to Grafana")
        return False
    registry = CollectorRegistry()
    labelnames = sorted({k for s in series for k in s.get("labels", {})})
    gauges: dict[str, Gauge] = {}
    for s in series:
        labels = {k: str(s.get("labels", {}).get(k, "")) for k in labelnames}
        for name, value in s["metrics"].items():
            if value is None or value != value:      # skip None and NaN
                continue
            metric = _metric_name(name)
            if metric not in gauges:
                gauges[metric] = Gauge(metric, f"BenchBrief eval metric {name}",
                                       labelnames, registry=registry)
            gauge = gauges[metric]
            (gauge.labels(**labels) if labelnames else gauge).set(float(value))
    try:
        push_to_gateway(url, job=job, registry=registry)
        print(f"pushed {len(gauges)} metrics to {url} (job={job})")
        return True
    except Exception as e:  # network down, gateway not running, ...
        print(f"WARNING: could not push to {url}: {e}")
        return False