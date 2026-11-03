from indexing.embed import embed_texts
from indexing.pinecone_client import _get_pinecone_client, PINECONE_INDEX_NAME
from db.supabase_client import get_client
from indexing.pinecone_client import _make_chunk_key
def _get_candidate_opinion_ids(judge: str | None= None, section_label: str | None = None) -> list[int] | None:
    if not judge and not section_label:
        return None

    client = get_client()
    opinion_ids = None
    if judge:
        docket_result = (
            client.table("dockets")
            .select("id")
            .ilike("assigned_to_str", f"%{judge}%")
            .execute()
        )

        docket_ids = [row["id"] for row in docket_result.data]

        if not docket_ids:
            return []

        cluster_result = (
            client.table("clusters")
            .select("id")
            .in_("docket_id", docket_ids)
            .execute()
        )
        cluster_ids = [row["id"] for row in cluster_result.data]

        if not cluster_ids:
            return []

        opinion_result = (
            client.table("opinions")
            .select("id")
            .in_("cluster_id", cluster_ids)
            .execute()
        )
        opinion_ids =  [row["id"] for row in opinion_result.data]

    if section_label:
        query = client.table("chunks").select("opinion_id").eq("section_label", section_label)
        if opinion_ids is not None:
            query = query.in_("opinion_id", opinion_ids)
        chunk_result = query.execute()
        opinion_ids = list({row["opinion_id"] for row in chunk_result.data})

    return opinion_ids if opinion_ids is not None else []

def _semantic_search(query: str, top_k: int = 5, opinion_id_filter: list[int] | None = None) -> list[dict]:
    query_vector = embed_texts([query])[0]

    pc = _get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)

    pinecone_filter = None
    if opinion_id_filter is not None:
        pinecone_filter = {"opinion_id": {"$in": opinion_id_filter}}

    results = index.query(
        vector = query_vector,
        top_k = top_k,
        include_metadata= True,
        filter = pinecone_filter,
    )

    matches = results["matches"]
    if not matches:
        return []

    chunk_keys = [match["id"] for match in matches]

    client = get_client()
    text_result = (
        client.table("chunks")
        .select("chunk_key, text")
        .in_("chunk_key", chunk_keys)
        .execute()
    )
    text_by_key = {row["chunk_key"]: row["text"] for row in text_result.data}

    return [
        {
            "chunk_key": match["id"],
            "score": match["score"],
            "opinion_id": match["metadata"].get("opinion_id"),
            "section_label": match["metadata"].get("section_label"),
            "text": text_by_key.get(match["id"], ""),
        }
        for match in results["matches"]
    ]

def search(query: str, top_k: int = 5, judge: str | None = None, section_label: str | None = None) -> list[dict]:
    candidate_opinion_ids = _get_candidate_opinion_ids(judge=judge, section_label=section_label)

    if candidate_opinion_ids is None:
        return _semantic_search(query, top_k=top_k)

    if not candidate_opinion_ids:
        return []

    return _semantic_search(query, top_k=top_k, opinion_id_filter=candidate_opinion_ids)