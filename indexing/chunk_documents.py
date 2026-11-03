import json
import re
import html

from config import RAW_DOCKETS_PATH, RAW_OPINIONS_PATH, RAW_CLUSTERS_PATH, PROCESSED_DIR
from utils.text_clean import strip_html

def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, run ingestion:")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records

def _build_lookups() -> tuple[dict[int, dict], dict[int, dict]]:
    dockets = _read_jsonl(RAW_DOCKETS_PATH)
    clusters = _read_jsonl(RAW_CLUSTERS_PATH)

    dockets_by_id = {d["id"]: d for d in dockets}
    clusters_by_id = {c["id"]: c for c in clusters}

    return dockets_by_id, clusters_by_id

def _docket_id_from_cluster(cluster: dict):
    """Opinions only link to a cluster; the cluster links to the docket."""
    if cluster.get("docket_id"):
        return cluster["docket_id"]
    url = cluster.get("docket") or ""          # e.g. ".../dockets/12345/"
    match = re.search(r"/dockets/(\d+)/", url)
    return int(match.group(1)) if match else None

def _build_chunk_metadata(opinion: dict, clusters_by_id: dict, dockets_by_id: dict) -> dict:
    cluster = clusters_by_id.get(opinion.get("cluster_id"), {})
    docket = dockets_by_id.get(_docket_id_from_cluster(cluster), {})

    # RECAP orders often have a blank cluster case_name, so fall back.
    case_name = (cluster.get("case_name") or docket.get("case_name")
                 or cluster.get("case_name_short") or cluster.get("case_name_full") or "")

    return {
        "opinion_id": opinion.get("id"),
        "cluster_id": cluster.get("id"),
        "docket_id": docket.get("id"),
        "case_name": case_name,
        "court": docket.get("court_id"),
        "judge": docket.get("assigned_to_str"),
        "nature_of_suit": docket.get("nature_of_suit"),
        "date_filled": cluster.get("date_filed") or docket.get("date_filed"),
    }

def _strip_line_numbers(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"^\s*\d+\s+", "", line) for line in lines]
    return "\n".join(cleaned_lines)

def _decode_html_entities(text: str) -> str:
    return html.unescape(text)

def run(strategy_name: str = "heuristic", **strategy_kwargs) -> None:
    from indexing.chunking.registry import get_strategy
    from utils.jsonl import read_jsonl, append_jsonl
    dockets_by_id, clusters_by_id = _build_lookups()
    opinions = read_jsonl(RAW_OPINIONS_PATH)

    strategy = get_strategy(strategy_name, **strategy_kwargs)
    print(f"Chunking started using {strategy}")

    output_path = PROCESSED_DIR/ f"chunks_{strategy.name}.jsonl"
    # append_jsonl appends, so a re-run would duplicate every chunk: start clean.
    output_path.unlink(missing_ok=True)
    total_chunks = 0

    for opinion in opinions:
        text = opinion.get("html_with_citations") or opinion.get("plain_text") or ""
        if not text:
            continue

        # Order matters: tags first (they hide line numbers at line starts),
        # then entities are decoded inside strip_html, then line numbers.
        text = strip_html(text)
        text = _strip_line_numbers(text)
        metadata = _build_chunk_metadata(opinion, clusters_by_id, dockets_by_id)
        chunks = strategy.chunk(text, metadata)

        for chunk in chunks:
            record = {
                "text": chunk.text,
                "section_label": chunk.section_label,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            append_jsonl(output_path, record)
            total_chunks += 1
    print(f"Done. Wrote {total_chunks} chunks to {output_path}")