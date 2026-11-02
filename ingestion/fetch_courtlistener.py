import json
import os
import time
from pathlib import Path
from typing import Iterator
from utils.jsonl import append_jsonl
import requests

from config import (
    CASE_TYPE_KEYWORDS,
    CL_DOCKETS_ENDPOINT,
    CL_CLUSTERS_ENDPOINT,
    CL_OPINIONS_ENDPOINT,
    CL_REQUEST_TIMEOUT_SECONDS,
    COURT_ID,
    RAW_CLUSTERS_PATH,
    RAW_DOCKETS_PATH,
    RAW_OPINIONS_PATH,
    RAW_DIR,
)

REQUEST_DELAY_SECONDS = 13


def _make_session() -> dict:
    headers = {"User-Agent": "benchbrief-portfolio-project/0.1"}
    token = os.environ.get("COURTLISTENER_API_TOKEN")
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def _paginated_get(headers: dict, url: str, params: dict | None = None) -> Iterator[dict]:
    next_url = url
    next_params = params

    while next_url:
        response = requests.get(next_url, params=next_params, headers=headers, timeout=CL_REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 429:
            wait_seconds = int(response.headers.get("Retry-After", 60))
            print("Rate limited, sleeping in 60 seconds...")
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        payload = response.json()

        for result in payload.get("results", []):
            yield result

        next_url = payload.get("next")
        next_params = None

        time.sleep(REQUEST_DELAY_SECONDS)

def fetch_dockets(headers: dict, max_dockets: int = 200) -> list[dict]:
    dockets = []
    params = {
        "court": COURT_ID,
    }

    for docket in _paginated_get(headers, CL_DOCKETS_ENDPOINT, params=params):
        dockets.append(docket)
        if len(dockets) >= max_dockets:
            break

    return dockets

def fetch_clusters_for_docket(headers: dict, docket_id: int) -> list[dict]:
    params= {"docket": docket_id}
    return list(_paginated_get(headers, CL_CLUSTERS_ENDPOINT, params=params))

def fetch_opinions_for_cluster(headers: dict, cluster_id: int) -> list[dict]:
    params= {"cluster": cluster_id}
    return list(_paginated_get(headers, CL_OPINIONS_ENDPOINT, params=params))

def fetch_opinions(headers: dict, max_opinions: int = 200) -> list[dict]:
    opinions = []
    params = {
        "cluster__docket__court": COURT_ID,
    }

    for opinion in _paginated_get(headers, CL_OPINIONS_ENDPOINT, params=params):
        opinions.append(opinion)
        if len(opinions) >= max_opinions:
            break

    return opinions

def fetch_by_url(headers: dict, url: str) -> dict:
    response = requests.get(url, headers=headers, timeout=CL_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()

def run(max_opinions: int = 200) -> None:
    headers = _make_session()

    print(f"Fetching opinions for court={COURT_ID}...")
    opinions = fetch_opinions(headers, max_opinions=max_opinions)
    print(f"  -> {len(opinions)} opinions")

    seen_cluster_urls = set()
    seen_docket_urls = set()

    for i, opinion in enumerate(opinions):
        append_jsonl(RAW_OPINIONS_PATH, opinion)

        cluster_url = opinion.get("cluster")
        if not cluster_url or cluster_url in seen_cluster_urls:
            continue
        seen_cluster_urls.add(cluster_url)

        cluster = fetch_by_url(headers, cluster_url)
        append_jsonl(RAW_CLUSTERS_PATH, cluster)

        docket_url = cluster.get("docket")
        if not docket_url or docket_url in seen_docket_urls:
            continue
        seen_docket_urls.add(docket_url)

        docket = fetch_by_url(headers, docket_url)
        append_jsonl(RAW_DOCKETS_PATH, docket)

        print(f"  [{i + 1}/{len(opinions)}] opinion {opinion['id']}: "
              f"cluster {cluster.get('id')}, docket {docket.get('id')}")

    print("Done.")

CURATED_CLUSTER_IDS = [
    2478126, 8725019, 7326492, 7333159, 7325588,
    8727687, 1431057, 7326041, 1737559, 2416225,
]


def fetch_curated_cases(headers: dict) -> None:
    for cluster_id in CURATED_CLUSTER_IDS:
        cluster_url = f"{CL_CLUSTERS_ENDPOINT}{cluster_id}/"
        cluster = fetch_by_url(headers, cluster_url)
        append_jsonl(RAW_CLUSTERS_PATH, cluster)
        time.sleep(REQUEST_DELAY_SECONDS)

        docket_url = cluster.get("docket")
        if docket_url:
            docket = fetch_by_url(headers, docket_url)
            append_jsonl(RAW_DOCKETS_PATH, docket)
            time.sleep(REQUEST_DELAY_SECONDS)

        opinions = fetch_opinions_for_cluster(headers, cluster_id)
        for opinion in opinions:
            append_jsonl(RAW_OPINIONS_PATH, opinion)
        time.sleep(REQUEST_DELAY_SECONDS)

        print(f"  cluster {cluster_id}: docket + {len(opinions)} opinion(s) fetched")

if __name__ == "__main__":
    run(max_opinions=2)
