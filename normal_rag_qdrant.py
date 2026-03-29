import os
import json
import asyncio
import logging
from time import time
import numpy as np

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from triplet_graphrag import (
    chunk_document,
    embed_texts,
    LLAMA_SERVER_BASE_URL,
    MODEL,
    INPUT_FILE,
    CACHE_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("normal-rag")

COLLECTION_NAME = "dickens_book"
QDRANT_URL = "http://localhost:6333"

_client = AsyncOpenAI(
    base_url=LLAMA_SERVER_BASE_URL,
    api_key="not-needed",
    timeout=1800.0,
    max_retries=1,
)

def init_qdrant_and_upsert_chunks(client: QdrantClient, chunks: list[dict]):
    log.info(f"Connecting to Qdrant at {QDRANT_URL}...")
    
    try:
        client.get_collection(COLLECTION_NAME)
        log.info(f"Collection '{COLLECTION_NAME}' already exists.")
        collection_info = client.get_collection(COLLECTION_NAME)
        if collection_info.points_count > 0:
            log.info(f"Collection '{COLLECTION_NAME}' has {collection_info.points_count} points. Skipping upsert.")
            return
    except UnexpectedResponse as e:
        if "Not found" in str(e) or e.status_code == 404:
            log.info(f"Creating collection '{COLLECTION_NAME}'...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        else:
            raise e

    # Embed chunks
    texts = [c["content"] for c in chunks]
    log.info(f"Embedding {len(texts)} chunks using local SentenceTransformers...")
    embeddings = embed_texts(texts)

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content"),
                    "chunk_order_index": chunk.get("chunk_order_index")
                }
            )
        )
    
    log.info(f"Upserting {len(points)} points to Qdrant collection '{COLLECTION_NAME}'...")
    # Upsert in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=batch
        )
    log.info("Finished embedding and upserting chunks.")


async def _query_qdrant(client: QdrantClient, query: str, top_k: int = 5):
    log.info(f"Querying Qdrant for: '{query}'")
    q_emb = embed_texts([query])[0]
    
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb.tolist(),
        limit=top_k
    )
    return search_result.points

async def answer_question(client: QdrantClient, question: str, top_k: int = 5):
    search_result = await _query_qdrant(client, question, top_k)
    
    context_texts = []
    for i, hit in enumerate(search_result):
        context_texts.append(f"--- Document {i+1} (Relevance Score: {hit.score:.4f}) ---\n{hit.payload.get('content')}")
        
    context = "\n\n".join(context_texts)
    
    prompt = f"""\
You are an AI assistant answering questions based on the provided document context.
Use ONLY the context below to answer the question. Be specific and detailed.
If you cannot answer the question from the context, say so.

## Context Documents
{context}

## Question
{question}

## Answer
"""
    messages = [{"role": "user", "content": prompt}]
    log.info("Sending prompt to local LLM...")
    response = await _client.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content


async def run_pipeline():
    # Load chunks (either from cache or generate them)
    chunks_path = os.path.join(CACHE_DIR, "chunks.json")
    if os.path.exists(chunks_path):
        log.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        log.info("Chunks not found. Generating them using triplet_graphrag chunking logic...")
        chunks = chunk_document(INPUT_FILE)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Note: Using grpc port (6334) if standard is 6333, wait standard HTTP is 6333
    qc = QdrantClient(url=QDRANT_URL)
    
    init_qdrant_and_upsert_chunks(qc, chunks)

    questions = [
        "What are the top themes in this story?",
        "Who is Ebenezer Scrooge and what is his character arc?"
    ]

    with open("ans_qdrant.txt", "w", encoding="utf-8") as out:
        for q in questions:
            log.info("=" * 60)
            log.info(f"NORMAL RAG QUERY: {q}")
            t0 = time()
            answer = await answer_question(qc, q, top_k=3)
            t1 = time()
            
            output_str = f"============================================================\n" \
                         f"NORMAL RAG QUERY: {q}\n" \
                         f"============================================================\n" \
                         f"{answer}\n\n(Time taken: {t1-t0:.2f}s)\n\n"
            
            print(output_str)
            out.write(output_str)
            out.flush()
            
            log.info("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("  Normal RAG with Qdrant Pipeline")
    print("=" * 60)
    asyncio.run(run_pipeline())
