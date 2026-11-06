from config import RAW_CLUSTERS_PATH, RAW_DOCKETS_PATH, RAW_OPINIONS_PATH, PROCESSED_DIR
from db.supabase_client import get_client
from utils.jsonl import read_jsonl
from indexing.pinecone_client import _make_chunk_key

def _docket_row(docket: dict) -> dict:
    return {
        "id": docket["id"],
        "case_name": docket.get("case_name"),
        "court_id": docket.get("court_id"),
        "assigned_to_str": docket.get("assigned_to_str"),
        "nature_of_suit": docket.get("nature_of_suit"),
        "date_filed": docket.get("date_filed"),
    }

def _cluster_row(cluster: dict) -> dict:
    return {
        "id": cluster["id"],
        "docket_id": cluster.get("docket_id"),
        "case_name": cluster.get("case_name"),
        "nature_of_suit": cluster.get("nature_of_suit"),
        "judges": cluster.get("judges"),
        "date_filed": cluster.get("date_filed"),
    }

def _opinion_row(opinion: dict) -> dict:
    return {
        "id": opinion["id"],
        "cluster_id": opinion.get("cluster_id"),
        "text_html": opinion.get("html_with_citations"),
        "text_plain": opinion.get("plain_text"),
        "extracted_by_ocr": opinion.get("extracted_by_ocr"),
    }

def _chunk_row(chunk_dict: dict, strategy_name: str) -> dict:
    opinion_id = chunk_dict["metadata"]["opinion_id"]
    chunk_index = chunk_dict["chunk_index"]

    return {
        "opinion_id": opinion_id,
        "strategy_name": strategy_name,
        "chunk_index": chunk_index,
        "section_label": chunk_dict.get("section_label"),
        "text": chunk_dict["text"],
        "metadata": chunk_dict.get("metadata"),
        "chunk_key": _make_chunk_key(opinion_id, strategy_name, chunk_index),
    }

def run(strategy_name: str = "heuristic" ) -> None:
    client = get_client()

    print("Loading dockets...")
    dockets = read_jsonl(RAW_DOCKETS_PATH)
    docket_rows = [_docket_row(d) for d in dockets]
    client.table("dockets").upsert(docket_rows).execute()
    print(f"Done. Wrote {len(docket_rows)} dockets")

    print("Loading clusters...")
    clusters = read_jsonl(RAW_CLUSTERS_PATH)
    cluster_rows = [_cluster_row(c) for c in clusters]
    loaded_clusters = 0
    for row in cluster_rows:
        try:
            client.table("clusters").upsert(row).execute()
            loaded_clusters += 1
        except Exception as e:
            print(f"Failed to load cluster: {row['id']}: {e}")
    print(f"{loaded_clusters}/{len(cluster_rows)} clusters loaded")

    print("Loading opinions...")
    opinions = read_jsonl(RAW_OPINIONS_PATH)
    opinion_rows = [_opinion_row(o) for o in opinions]
    loaded_opinions = 0
    for row in opinion_rows:
        try:
            client.table("opinions").upsert(row).execute()
            loaded_opinions += 1
        except Exception as e:
            print(f"Failed to load opinion: {row['id']}: {e}")

    print(f" Loading chunks (strategy={strategy_name})")
    chunks_path = PROCESSED_DIR / f"chunks_{strategy_name}.jsonl"
    chunk_dicts = read_jsonl(chunks_path)
    loaded_chunks = 0
    for chunk in chunk_dicts:
        try:
            row = _chunk_row(chunk, strategy_name)
            client.table("chunks").upsert(row).execute()
            loaded_chunks += 1
        except Exception as e:
            print(f"Failed to load chunk: {e}")
    print(f"Done. Wrote {loaded_chunks}/{len(chunk_dicts)} chunks")

