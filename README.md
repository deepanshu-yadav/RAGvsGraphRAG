# RAG vs GraphRAG Comparison Project

This project provides a robust, local, cost-effective evaluation of two distinct approaches for retrieving and reasoning over documents:
1. **Normal RAG** (Retrieval-Augmented Generation) using dense vector embeddings stored in **Qdrant**.
2. **GraphRAG** using a knowledge graph extracted via `triplet-extract` (Stanford OpenIE, 0 LLM calls for extraction) and stored in **Neo4j**, enriched with communities and summaries.

Both approaches leverage a containerized `llama.cpp` server running the `granite-4.0-micro-Q5_0.gguf` model for both embedding generation and answering queries.

## Architecture

*   **Vector Database (Normal RAG)**: Qdrant (runs in Docker)
*   **Graph Database (GraphRAG)**: Neo4j (runs in Docker)
*   **LLM & Embedding Engine**: `llama.cpp` server (runs in Docker on CPU for compatibility, or natively for GPU)
*   **Information Extraction (GraphRAG)**: `triplet-extract` leverages standard NLP (spaCy + OpenIE) instead of computing-heavy LLM prompts to extract Entities and Relations.

## Prerequisites

1.  **Docker Desktop** (with WSL2 integration enabled on Windows if you run Windows).
2.  **Python 3.10+**.
3. LLama cpp server 

## Quick Setup

### 1. Start the Infrastructure (Docker)

The provided `docker-compose.yml` spins up Neo4j and Qdrant

```bash
docker compose up -d
```

*This will start:*
*   **Neo4j Browser**: `http://localhost:7474` (Credentials: `neo4j` / `carol123`)
*   **Qdrant UI**: `http://localhost:6333/dashboard`


### 2. Install Python Dependencies

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

and then download spacy models

```bash
python -m spacy download en_core_web_sm
```

*(Optional but recommended: Create a virtual environment first).*

## Running the LLama cpp server


Download llama cpp server from [Windows here](https://github.com/ggml-org/llama.cpp/releases/download/b8574/llama-b8574-bin-win-cpu-x64.zip)  or [Linux  here](https://github.com/ggml-org/llama.cpp/releases/download/b8574/llama-b8574-bin-ubuntu-x64.tar.gz)
and granite model from [here](https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF/blob/main/granite-4.0-micro-Q5_0.gguf). Granite will work for this simple comparison. 

And then run 
llama-server.exe -m "<path_to_your_model>granite-4.0-micro-Q5_0.gguf" --host 0.0.0.0 --port 8000 --n-gpu-layers -1 --embeddings  --pooling mean -c 32768

or in linux
llama-server -m "<path_to_your_model>granite-4.0-micro-Q5_0.gguf" --host 0.0.0.0 --port 8000 --n-gpu-layers -1 --embeddings  --pooling mean -c 32768

## Running Normal RAG (Qdrant)

The Normal RAG script chunks text, embeds the document chunks using local SentenceTransformers, and upserts them into Qdrant. During querying, it retrieves the most similar chunks and prompts the local LLM.

```bash
python normal_rag_qdrant.py
```

**What it does:**
1.  Chunks `book.txt` into fragments.
2.  Embeds fragments locally using SentenceTransformers.
3.  Upserts points to Qdrant collection `dickens_book`.
4.  Queries the Qdrant DB for context, then queries the local LLM server for the final answer. Output is saved to `ans_qdrant.txt`.

## Running GraphRAG Pipeline (Neo4j)

The GraphRAG script builds a semantic knowledge graph from the document. You can control which parts of the pipeline to run using the `--mode` (or `-m`) flag.

```bash
# Example: Run the full pipeline from scratch
python triplet_graphrag.py --mode full

# Example: Just run queries on existing graph
python triplet_graphrag.py --mode query
```

### Modes of Operation

| Mode | Description | When to use |
| :--- | :--- | :--- |
| `full` | **End-to-End Pipeline**: Chunking -> Extraction -> Normalization -> Neo4j Storage -> Community Detection -> Report Generation -> Embedding. | Use this for the first run on a new document. |
| `continue` | **Resume from Cluster**: Skips extraction and storage, jumps to community detection, report generation, and embedding. | Use if you've already extracted triplets but want to re-run the clustering logic. |
| `reports_only` | **Regenerate Reports**: Re-runs Leiden clustering and generates fresh community summaries and embeddings. | Use if you want to refresh the high-level community descriptions. |
| `rebuild_embeddings` | **Re-embed Only**: Only recalculates vector embeddings for entities and reports. | Use if you've changed your embedding model or parameters. |
| `query` | **Direct Query**: Skips all processing steps and goes straight to the interactive query engine. | Use for daily interaction once the graph is built. |

**Workflow Highlights:**
1.  **Extract**: Uses `triplet-extract` (CPU-bound) for extracting Subject-Predicate-Object relations rapidly. No LLM prompts needed.
2.  **Store**: Data is persistently saved as a heavily connected semantic graph inside Neo4j.
3.  **Cluster**: Igraph / Leiden algorithm detects isolated communities of related knowledge inside the graph.
4.  **Answer**: Executes Global (over community summaries) or Local (over specific semantic neighborhoods) queries using `llama-server`.

## Troubleshooting

