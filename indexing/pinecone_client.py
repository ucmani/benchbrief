import os

from pinecone import Pinecone, ServerlessSpec
from indexing.embed import EMBEDDING_DIMENSION, embed_texts


PINECONE_INDEX_NAME = "benchbrief"
BATCH_SIZE = 50

def _get_pinecone_client() -> Pinecone:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")
    return Pinecone(api_key=api_key)

def _make_chunk_key(opinion_id: int, strategy_name: str, chunk_index: int) -> str:
    return f"{opinion_id}:{strategy_name}:{chunk_index}"

def ensure_index_exists() -> None:
    pc = _get_pinecone_client()
    existing_names = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def upsert_chunks(chunks: list[dict], strategy_name: str) -> int:
    pc = _get_pinecone_client()
    index = pc.Index(PINECONE_INDEX_NAME)

    total_upserted = 0

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        print(f"batch_start={batch_start}, batch size={len(batch)}")

        texts = [c["text"] for c in batch]
        vectors = embed_texts(texts)
        print(f"embedded {len(vectors)} vectors")

        pinecone_vectors = []
        for chunk, vector in zip(batch, vectors):
            opinion_id = chunk["metadata"]["opinion_id"]
            chunk_index = chunk["chunk_index"]
            key = _make_chunk_key(opinion_id, strategy_name, chunk_index)

            pinecone_vectors.append({
                "id": key,
                "values": vector,
                "metadata": {
                    "opinion_id": opinion_id,
                    "strategy_name": strategy_name,
                    "chunk_index": chunk_index,
                    "section_label": chunk.get("section_label", "unknown"),
                },
            })

        print(f"about to upsert {len(pinecone_vectors)} vectors")
        result = index.upsert(vectors=pinecone_vectors)
        print(f"upsert result: {result}")
        total_upserted += len(pinecone_vectors)

    return total_upserted

def run(strategy_name: str = "heuristic") -> None:
    print("Run started")
    from config import PROCESSED_DIR
    from utils.jsonl import read_jsonl

    ensure_index_exists()

    chunks_path = PROCESSED_DIR / f"chunks_{strategy_name}.jsonl"
    chunks = read_jsonl(chunks_path)

    print(f"Embedding and upserting {len(chunks)} chunks(strategy={strategy_name}")
    total_upserted = upsert_chunks(chunks, strategy_name)

if __name__ == "__main__":
    run("fixed_size")
