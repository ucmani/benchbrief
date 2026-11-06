from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "benchbrief_requests_total",
    "Total number of requests",
    ["stage", "status"]
)

REQUEST_LATENCY = Histogram(
    "benchbrief_request_latency_seconds",
    "Latency of pipeline stages in seconds",
    ["stage"]
)