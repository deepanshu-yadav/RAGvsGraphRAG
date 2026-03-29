"""
Triplet-Extract + Neo4j GraphRAG Pipeline
==========================================

Fast, cost-effective GraphRAG using:
  - triplet-extract (Stanford OpenIE) for entity/relation extraction (NO LLM)
  - Neo4j for persistent graph storage
  - igraph Leiden for community detection
  - Local llama-server (granite-4.0-micro) for reports + queries only

MODE controls the pipeline:
  "full"              → full pipeline from scratch
  "reports_only"      → re-run community detection + report generation
  "rebuild_embeddings"→ re-embed entities/reports with current model
  "query"             → skip all processing, go straight to querying
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
import argparse
from collections import defaultdict
from time import time

import numpy as np
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("triplet-graphrag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# MODE is now handled via argparse in the main block.
# Default configurations:

# Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "carol123"
NEO4J_DATABASE = "neo4j"  # default database

# LLM / Embedding (local llama-server)
LLAMA_SERVER_BASE_URL = "http://localhost:8000/v1"
MODEL = "granite-4.0-micro-Q5_0.gguf"
EMBEDDING_MODEL = MODEL
EMBEDDING_DIM = 2560
EMBEDDING_MAX_TOKENS = 32768

# Chunking
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP = 100
TIKTOKEN_MODEL = "gpt-4o"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR,  "book.txt")
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

# Triplet extraction
TRIPLET_MIN_CONFIDENCE = 0.3
TRIPLET_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Ensure cache directory exists
# ---------------------------------------------------------------------------
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Async OpenAI client
# ---------------------------------------------------------------------------
_llm_semaphore = asyncio.Semaphore(1)
_embed_semaphore = asyncio.Semaphore(2)

_client = AsyncOpenAI(
    base_url=LLAMA_SERVER_BASE_URL,
    api_key="not-needed",
    timeout=1800.0,
    max_retries=1,
)


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Chunking
# ═══════════════════════════════════════════════════════════════════════════
def md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def chunk_document(filepath: str) -> list[dict]:
    """Split document into token-sized chunks."""
    import tiktoken

    log.info(f"Reading: {filepath}")
    with open(filepath, encoding="utf-8-sig") as f:
        text = f.read()
    log.info(f"  → {len(text):,} characters")

    encoder = tiktoken.encoding_for_model(TIKTOKEN_MODEL)
    tokens = encoder.encode(text)
    log.info(f"  → {len(tokens):,} tokens")

    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKEN_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append({
            "chunk_id": f"chunk-{md5_hash(chunk_text)[:12]}",
            "content": chunk_text,
            "tokens": len(chunk_tokens),
            "chunk_order_index": chunk_idx,
        })
        start += CHUNK_TOKEN_SIZE - CHUNK_OVERLAP
        chunk_idx += 1

    log.info(f"  → {len(chunks)} chunks produced")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Triplet Extraction (NO LLM — uses Stanford OpenIE via spaCy)
# ═══════════════════════════════════════════════════════════════════════════
def extract_triplets_from_chunks(chunks: list[dict]) -> list[dict]:
    """Extract (subject, relation, object) triplets from all chunks.
    
    Uses triplet-extract (CPU mode). Zero LLM calls.
    Returns list of dicts with subject, relation, object, confidence, source_chunk.
    """
    from triplet_extract import OpenIEExtractor

    log.info("Initializing triplet extractor (CPU balanced mode)...")
    extractor = OpenIEExtractor(
        enable_clause_split=True,
        enable_entailment=True,
        min_confidence=TRIPLET_MIN_CONFIDENCE,
    )

    texts = [c["content"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    log.info(f"Extracting triplets from {len(texts)} chunks (batch_size={TRIPLET_BATCH_SIZE})...")
    t0 = time()
    batch_results = extractor.extract_batch(texts, batch_size=TRIPLET_BATCH_SIZE, progress=True)
    elapsed = time() - t0

    all_triplets = []
    for chunk_id, triplets in zip(chunk_ids, batch_results):
        for t in triplets:
            all_triplets.append({
                "subject": t.subject.strip(),
                "relation": t.relation.strip(),
                "object": t.object.strip(),
                "confidence": getattr(t, "confidence", 1.0),
                "source_chunk": chunk_id,
            })

    log.info(f"  → {len(all_triplets)} triplets extracted in {elapsed:.1f}s")
    if all_triplets:
        sample = all_triplets[0]
        log.info(f"  Sample: ({sample['subject']}, {sample['relation']}, {sample['object']})")

    return all_triplets


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Entity Normalization & Deduplication
# ═══════════════════════════════════════════════════════════════════════════
def normalize_entities(triplets: list[dict]) -> tuple[dict, list[dict]]:
    """Normalize entity names (case-insensitive dedup) and deduplicate relations.
    
    Returns:
        entities: dict[canonical_name] → {name, mention_count, source_chunks}
        relations: deduplicated list of {subject, relation, object, weight, source_chunks}
    """
    # --- Entity normalization: track all surface forms ---
    entity_forms = defaultdict(lambda: defaultdict(int))  # lower → {surface_form → count}
    entity_chunks = defaultdict(set)

    for t in triplets:
        for role in ("subject", "object"):
            name = t[role]
            if len(name) < 2:
                continue
            key = name.lower().strip()
            entity_forms[key][name] += 1
            entity_chunks[key].add(t["source_chunk"])

    # Pick the most common surface form as canonical
    entities = {}
    for key, forms in entity_forms.items():
        canonical = max(forms, key=forms.get)
        entities[key] = {
            "name": canonical,
            "mention_count": sum(forms.values()),
            "source_chunks": list(entity_chunks[key]),
        }

    # --- Relation deduplication ---
    rel_key_map = defaultdict(lambda: {"weight": 0, "source_chunks": set(), "relation": ""})
    for t in triplets:
        subj_key = t["subject"].lower().strip()
        obj_key = t["object"].lower().strip()
        if subj_key not in entities or obj_key not in entities:
            continue
        rk = (entities[subj_key]["name"], t["relation"].lower().strip(), entities[obj_key]["name"])
        entry = rel_key_map[rk]
        entry["weight"] += t.get("confidence", 1.0)
        entry["source_chunks"].add(t["source_chunk"])
        entry["relation"] = t["relation"]

    relations = []
    for (subj, rel_lower, obj), data in rel_key_map.items():
        relations.append({
            "subject": subj,
            "relation": data["relation"],
            "object": obj,
            "weight": round(data["weight"], 3),
            "source_chunks": list(data["source_chunks"]),
        })

    log.info(f"  → {len(entities)} unique entities, {len(relations)} unique relations")
    return entities, relations


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: Neo4j Storage
# ═══════════════════════════════════════════════════════════════════════════
class Neo4jGraphStore:
    """Store and query the knowledge graph in Neo4j."""

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD,
                 database=NEO4J_DATABASE):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        log.info(f"Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()

    def clear_graph(self):
        """Remove all nodes and relationships."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        log.info("  Neo4j graph cleared")

    def store_entities(self, entities: dict):
        """Create Entity nodes."""
        with self.driver.session(database=self.database) as session:
            for key, ent in entities.items():
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.mention_count = $mention_count,
                        e.source_chunks = $source_chunks
                    """,
                    id=key,
                    name=ent["name"],
                    mention_count=ent["mention_count"],
                    source_chunks=ent["source_chunks"],
                )
        log.info(f"  Stored {len(entities)} entities in Neo4j")

    def store_relations(self, relations: list[dict], entities: dict):
        """Create RELATES_TO edges between entities."""
        with self.driver.session(database=self.database) as session:
            for rel in relations:
                subj_id = rel["subject"].lower().strip()
                obj_id = rel["object"].lower().strip()
                session.run(
                    """
                    MATCH (a:Entity {id: $subj_id})
                    MATCH (b:Entity {id: $obj_id})
                    MERGE (a)-[r:RELATES_TO {relation: $relation}]->(b)
                    SET r.weight = $weight,
                        r.source_chunks = $source_chunks
                    """,
                    subj_id=subj_id,
                    obj_id=obj_id,
                    relation=rel["relation"],
                    weight=rel["weight"],
                    source_chunks=rel["source_chunks"],
                )
        log.info(f"  Stored {len(relations)} relations in Neo4j")

    def export_to_networkx(self):
        """Export Neo4j graph → NetworkX for community detection."""
        import networkx as nx
        G = nx.Graph()

        with self.driver.session(database=self.database) as session:
            # Nodes
            result = session.run("MATCH (e:Entity) RETURN e")
            for record in result:
                node = record["e"]
                G.add_node(
                    node["id"],
                    name=node.get("name", node["id"]),
                    mention_count=node.get("mention_count", 1),
                )

            # Edges
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                RETURN a.id AS src, b.id AS dst, r.relation AS relation, r.weight AS weight
                """
            )
            for record in result:
                G.add_edge(
                    record["src"], record["dst"],
                    relation=record["relation"],
                    weight=record["weight"] or 1.0,
                )

        log.info(f"  Exported NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def write_communities(self, membership: dict):
        """Write community IDs back to Neo4j nodes."""
        with self.driver.session(database=self.database) as session:
            for node_id, community_id in membership.items():
                session.run(
                    "MATCH (e:Entity {id: $id}) SET e.community = $community",
                    id=node_id,
                    community=community_id,
                )
        n_communities = len(set(membership.values()))
        log.info(f"  Wrote {n_communities} community IDs to {len(membership)} nodes")

    def get_community_data(self) -> dict:
        """Get all communities with their entities and relations."""
        communities = defaultdict(lambda: {"entities": [], "relations": []})

        with self.driver.session(database=self.database) as session:
            # Get entities per community
            result = session.run(
                "MATCH (e:Entity) WHERE e.community IS NOT NULL RETURN e"
            )
            for record in result:
                node = record["e"]
                cid = str(node["community"])
                communities[cid]["entities"].append({
                    "name": node.get("name", node["id"]),
                    "mention_count": node.get("mention_count", 1),
                })

            # Get intra-community relations
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE a.community = b.community AND a.community IS NOT NULL
                RETURN a.name AS src, r.relation AS relation, b.name AS dst,
                       r.weight AS weight, a.community AS community
                """
            )
            for record in result:
                cid = str(record["community"])
                communities[cid]["relations"].append({
                    "subject": record["src"],
                    "relation": record["relation"],
                    "object": record["dst"],
                    "weight": record["weight"],
                })

        return dict(communities)

    def get_entity_context(self, entity_names: list[str], max_hops: int = 1) -> dict:
        """Get subgraph around specific entities for local queries."""
        context = {"entities": [], "relations": []}
        with self.driver.session(database=self.database) as session:
            for name in entity_names:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = toLower($name)
                    OPTIONAL MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
                    RETURN e, collect(DISTINCT {
                        relation: r.relation,
                        neighbor_name: neighbor.name,
                        weight: r.weight
                    }) AS rels
                    """,
                    name=name,
                )
                for record in result:
                    node = record["e"]
                    context["entities"].append({
                        "name": node.get("name", ""),
                        "mention_count": node.get("mention_count", 1),
                        "community": node.get("community"),
                    })
                    for rel in record["rels"]:
                        if rel["relation"]:
                            context["relations"].append({
                                "entity": node.get("name", ""),
                                "relation": rel["relation"],
                                "neighbor": rel["neighbor_name"],
                                "weight": rel["weight"],
                            })
        return context


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: Community Detection (igraph Leiden)
# ═══════════════════════════════════════════════════════════════════════════
def detect_communities(G) -> dict:
    """Run Leiden community detection on a NetworkX graph.
    
    Returns: dict[node_id] → community_id
    """
    import igraph as ig

    nodes = list(G.nodes())
    if not nodes:
        log.warning("Empty graph — no communities to detect")
        return {}

    node_index = {n: i for i, n in enumerate(nodes)}
    edges = [
        (node_index[u], node_index[v])
        for u, v in G.edges()
        if u in node_index and v in node_index
    ]
    weights = [
        float(data.get("weight", 1.0))
        for u, v, data in G.edges(data=True)
    ]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights if weights else [1.0] * len(edges)

    partition = ig_graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        n_iterations=10,
        resolution=1.0,
    )

    membership = {}
    for node_id, cluster_id in zip(nodes, partition.membership):
        membership[node_id] = cluster_id

    n_communities = len(set(membership.values()))
    log.info(f"  Leiden: {n_communities} communities from {len(nodes)} nodes")
    return membership


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6: Community Report Generation (LLM)
def generate_community_reports(community_data: dict) -> dict:

    """Generate a structured, statistical summary report for each community.
    Outputs text instead of calling an LLM, making it fast for thousands of communities.
    """
    reports = {}
    total = len(community_data)
    log.info(f"Generating reports for {total} communities...")

    for i, (cid, data) in enumerate(community_data.items(), 1):
        entities = data["entities"]
        relations = data["relations"]

        # Skip tiny communities
        if len(entities) < 2:
            continue

        # Sort entities and relations
        sorted_entities = sorted(entities, key=lambda x: -x["mention_count"])
        sorted_relations = sorted(relations, key=lambda x: -x["weight"])

        # Determine the primary subject of the community
        main_entity = sorted_entities[0]["name"] if entities else f"Unknown Entities"
        title = f"Community around {main_entity}"

        # Generate structural summary text
        content = [
            f"# {title}",
            f"\nThis community contains **{len(entities)} entities** and **{len(relations)} internal relationships**.",
            "\n### Key Entities:",
        ]
        
        for e in sorted_entities[:10]:
            content.append(f"- **{e['name']}** (mentioned {e['mention_count']} times)")
            
        if relations:
            content.append("\n### Primary Relationships:")
            for r in sorted_relations[:15]:
                content.append(f"- {r['subject']} → *{r['relation']}* → {r['object']} (weight: {r['weight']:.1f})")
        else:
            content.append("\n### Primary Relationships:")
            content.append("- No internal relationships discovered.")

        reports[cid] = {
            "title": title,
            "content": "\n".join(content),
            "entity_count": len(entities),
            "relation_count": len(relations),
        }

    log.info(f"  → {len(reports)} community reports generated instantly")
    return reports


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 7: Embedding (Sentence Transformers)
# ═══════════════════════════════════════════════════════════════════════════
_st_model = None

def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts using fast local sentence-transformers."""
    if not texts:
        return np.array([])
    model = get_st_model()
    # model.encode automatically batches and shows a progress bar if we want, but we'll do our own logging
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def build_embedding_index(entities: dict, reports: dict) -> dict:
    """Embed all entities and community reports.
    
    Returns index dict ready for similarity search.
    """
    log.info("Building embedding index...")

    # --- Entity embeddings ---
    entity_items = list(entities.items())
    entity_texts = [ent["name"] for _, ent in entity_items]
    entity_ids = [key for key, _ in entity_items]

    log.info(f"  Embedding {len(entity_texts)} entities... (fast local encoding)")
    entity_embeddings = embed_texts(entity_texts)

    # --- Report embeddings ---
    report_items = list(reports.items())
    report_texts = [f"{r['title']}\n{r['content'][:2000]}" for _, r in report_items]
    report_ids = [cid for cid, _ in report_items]

    log.info(f"  Embedding {len(report_texts)} community reports... (fast local encoding)")
    report_embeddings = embed_texts(report_texts)

    index = {
        "entities": {
            "ids": entity_ids,
            "names": entity_texts,
            "embeddings": entity_embeddings.tolist(),
        },
        "reports": {
            "ids": report_ids,
            "texts": [r["title"] for _, r in report_items],
            "embeddings": report_embeddings.tolist(),
        },
    }

    log.info(f"  → Index built: {len(entity_ids)} entities + {len(report_ids)} reports")
    return index


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 8: Query Engine
# ═══════════════════════════════════════════════════════════════════════════
def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all rows in matrix."""
    if matrix.size == 0:
        return np.array([])
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


class QueryEngine:
    """Query the knowledge graph using embeddings + Neo4j + LLM."""

    def __init__(self, graph_store: Neo4jGraphStore, index: dict, reports: dict):
        self.graph_store = graph_store
        self.index = index
        self.reports = reports
        self.entity_embeddings = np.array(index["entities"]["embeddings"])
        self.report_embeddings = np.array(index["reports"]["embeddings"])

    async def local_query(self, question: str, top_k: int = 10) -> str:
        """Local query: find similar entities → get subgraph context → LLM answer."""
        log.info(f"[Local Query] {question}")

        # Embed the question
        q_emb = embed_texts([question])
        q_vec = q_emb[0]

        # Find top-K most similar entities
        sims = cosine_similarity(q_vec, self.entity_embeddings)
        if sims.size == 0:
            return "No entities in the index."
        top_indices = np.argsort(sims)[-top_k:][::-1]

        entity_names = [self.index["entities"]["names"][i] for i in top_indices]
        entity_scores = [float(sims[i]) for i in top_indices]

        log.info(f"  Top entities: {entity_names[:5]}")

        # Get subgraph context from Neo4j
        context = self.graph_store.get_entity_context(entity_names)

        # Build context text
        entities_text = "\n".join(
            f"- **{e['name']}** (mentions: {e['mention_count']}, community: {e.get('community', '?')})"
            for e in context["entities"]
        )
        relations_text = "\n".join(
            f"- {r['entity']} → *{r['relation']}* → {r['neighbor']} (weight: {r['weight']:.1f})"
            for r in context["relations"][:50]
        )

        prompt = f"""\
Use the following knowledge graph context to answer the question. Be specific and detailed.
Always provide a substantive answer based on the context provided.

## Relevant Entities
{entities_text}

## Relationships
{relations_text}

## Question
{question}

## Answer
"""
        messages = [{"role": "user", "content": prompt}]
        async with _llm_semaphore:
            response = await _client.chat.completions.create(model=MODEL, messages=messages)
        return response.choices[0].message.content

    async def global_query(self, question: str, top_k: int = 5) -> str:
        """Global query: use community reports to answer high-level questions."""
        log.info(f"[Global Query] {question}")

        # Embed the question
        q_emb = embed_texts([question])
        q_vec = q_emb[0]

        # Find top-K most relevant community reports
        sims = cosine_similarity(q_vec, self.report_embeddings)
        if sims.size == 0:
            return "No community reports available."
        top_indices = np.argsort(sims)[-top_k:][::-1]

        # Build report context
        report_context = []
        for idx in top_indices:
            cid = self.index["reports"]["ids"][idx]
            report = self.reports.get(cid, {})
            score = float(sims[idx])
            report_context.append(
                f"### {report.get('title', f'Community {cid}')} (relevance: {score:.3f})\n"
                f"{report.get('content', 'No content.')}\n"
            )

        reports_text = "\n---\n".join(report_context)

        prompt = f"""\
You are a knowledge analyst synthesizing insights from multiple community reports in a knowledge graph.

## Community Reports
{reports_text}

## Instructions
Using ONLY the community reports above, write a comprehensive answer to the user's question.
Always provide a substantive answer — the reports contain sufficient information.
Organize your response with markdown headers and bullet points where appropriate.
Do not say you cannot answer — synthesize what is in the reports.

## Question
{question}

## Answer
"""
        messages = [{"role": "user", "content": prompt}]
        async with _llm_semaphore:
            response = await _client.chat.completions.create(model=MODEL, messages=messages)
        return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════════════
#  Cache I/O helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_json(data, filename):
    filepath = os.path.join(CACHE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved: {filepath}")


def load_json(filename):
    filepath = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNERS
# ═══════════════════════════════════════════════════════════════════════════
def run_full_pipeline():
    """Full pipeline: chunk → extract → store → community → reports → embed."""
    log.info("=" * 60)
    log.info("  FULL PIPELINE")
    log.info("=" * 60)

    t_start = time()

    # 1. Chunk
    log.info("\n[1/7] Chunking document...")
    chunks = chunk_document(INPUT_FILE)
    save_json(chunks, "chunks.json")

    # 2. Extract triplets (NO LLM!)
    log.info("\n[2/7] Extracting triplets (Stanford OpenIE — zero LLM calls)...")
    triplets = extract_triplets_from_chunks(chunks)
    save_json(triplets, "triplets.json")

    # 3. Normalize entities
    log.info("\n[3/7] Normalizing entities...")
    entities, relations = normalize_entities(triplets)
    save_json({"entities": {k: v for k, v in entities.items()}, "relations": relations},
              "normalized.json")

    # 4. Store in Neo4j
    log.info("\n[4/7] Storing in Neo4j...")
    graph_store = Neo4jGraphStore()
    graph_store.clear_graph()
    graph_store.store_entities(entities)
    graph_store.store_relations(relations, entities)

    # 5. Community detection
    log.info("\n[5/7] Running community detection...")
    G = graph_store.export_to_networkx()
    membership = detect_communities(G)
    graph_store.write_communities(membership)

    # 6. Generate reports (Statistical — zero LLM calls)
    log.info("\n[6/7] Generating community reports (Statistical)...")
    community_data = graph_store.get_community_data()
    reports = generate_community_reports(community_data)
    save_json(reports, "community_reports.json")

    # 7. Build embedding index
    log.info("\n[7/7] Building embedding index...")
    index = build_embedding_index(entities, reports)
    save_json(index, "embedding_index.json")

    graph_store.close()

    elapsed = time() - t_start
    log.info(f"\n{'=' * 60}")
    log.info(f"  FULL PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info(f"  Entities: {len(entities)} | Relations: {len(relations)}")
    log.info(f"  Communities: {len(set(membership.values()))} | Reports: {len(reports)}")
    log.info(f"{'=' * 60}")


def run_reports_only():
    """Re-run community detection + report generation."""
    log.info("=" * 60)
    log.info("  REPORTS ONLY (re-run community detection + reports)")
    log.info("=" * 60)

    graph_store = Neo4jGraphStore()
    G = graph_store.export_to_networkx()
    membership = detect_communities(G)
    graph_store.write_communities(membership)

    community_data = graph_store.get_community_data()
    reports = generate_community_reports(community_data)
    save_json(reports, "community_reports.json")

    # Re-embed reports
    entities_data = load_json("normalized.json")
    if entities_data:
        entities = entities_data["entities"]
        index = build_embedding_index(entities, reports)
        save_json(index, "embedding_index.json")

    graph_store.close()
    log.info("  Reports regenerated ✓")


def run_continue_pipeline():
    """Continue from step 5: community detection → reports → embed.
    
    Use when steps 1-4 (chunk, extract, normalize, Neo4j store) are already done.
    """
    log.info("=" * 60)
    log.info("  CONTINUE PIPELINE (from step 5)")
    log.info("=" * 60)

    t_start = time()

    # Load cached entities for embedding later
    entities_data = load_json("normalized.json")
    if not entities_data:
        log.error("  Missing normalized.json. Run with MODE='full' first.")
        return
    entities = entities_data["entities"]

    # 5. Community detection
    log.info("\n[5/7] Running community detection...")
    graph_store = Neo4jGraphStore()
    G = graph_store.export_to_networkx()
    membership = detect_communities(G)
    graph_store.write_communities(membership)

    # 6. Generate reports
    log.info("\n[6/7] Generating community reports...")
    community_data = graph_store.get_community_data()
    reports = generate_community_reports(community_data)
    save_json(reports, "community_reports.json")

    # 7. Build embedding index
    log.info("\n[7/7] Building embedding index...")
    index = build_embedding_index(entities, reports)
    save_json(index, "embedding_index.json")

    graph_store.close()

    elapsed = time() - t_start
    n_communities = len(set(membership.values())) if membership else 0
    log.info(f"\n{'=' * 60}")
    log.info(f"  CONTINUE PIPELINE COMPLETE in {elapsed:.1f}s")
    log.info(f"  Entities: {len(entities)} | Communities: {n_communities} | Reports: {len(reports)}")
    log.info(f"{'=' * 60}")


def run_rebuild_embeddings():
    """Re-embed entities and reports with the current model."""
    log.info("=" * 60)
    log.info("  REBUILD EMBEDDINGS")
    log.info("=" * 60)

    entities_data = load_json("normalized.json")
    reports = load_json("community_reports.json")

    if not entities_data or not reports:
        log.error("  Missing normalized.json or community_reports.json. Run full pipeline first.")
        return

    entities = entities_data["entities"]
    index = build_embedding_index(entities, reports)
    save_json(index, "embedding_index.json")
    log.info("  Embeddings rebuilt ✓")


def run_queries():
    """Interactive query mode."""
    log.info("=" * 60)
    log.info("  QUERY MODE")
    log.info("=" * 60)

    # Load required data
    index = load_json("embedding_index.json")
    reports = load_json("community_reports.json")

    if not index or not reports:
        log.error("  Missing embedding_index.json or community_reports.json.")
        log.error("  Run with MODE='full' first.")
        return

    graph_store = Neo4jGraphStore()
    engine = QueryEngine(graph_store, index, reports)

    # --- Demo queries ---
    print("\n" + "=" * 60)
    print("GLOBAL QUERY: What are the top themes in this story?")
    print("=" * 60)
    result = asyncio.run(engine.global_query("What are the top themes in this story?"))
    print(result)

    print("\n" + "=" * 60)
    print("LOCAL QUERY: Who is Ebenezer Scrooge and what is his character arc?")
    print("=" * 60)
    result = asyncio.run(engine.local_query("Who is Ebenezer Scrooge and what is his character arc?"))
    print(result)

    graph_store.close()


def debug_state():
    """Print the state of all cached files and Neo4j."""
    print("\n" + "=" * 60)
    print("  DEBUG: Pipeline State")
    print("=" * 60)

    for fname, label in [
        ("chunks.json", "Chunks"),
        ("triplets.json", "Triplets"),
        ("normalized.json", "Entities/Relations"),
        ("community_reports.json", "Community Reports"),
        ("embedding_index.json", "Embedding Index"),
    ]:
        data = load_json(fname)
        if data is None:
            print(f"  [{label:20s}] NOT FOUND")
        elif isinstance(data, list):
            print(f"  [{label:20s}] {len(data)} entries")
        elif isinstance(data, dict):
            if "entities" in data and "relations" in data and "reports" not in data and "embeddings" not in data:
                print(f"  [{label:20s}] {len(data['entities'])} entities, {len(data['relations'])} relations")
            elif "entities" in data and "reports" in data:
                print(f"  [{label:20s}] {len(data['entities']['ids'])} entity + {len(data['reports']['ids'])} report embeddings")
            else:
                print(f"  [{label:20s}] {len(data)} entries")

    # Check Neo4j
    try:
        graph_store = Neo4jGraphStore()
        with graph_store.driver.session(database=graph_store.database) as session:
            n_nodes = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            n_rels = session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS c").single()["c"]
            n_communities = session.run(
                "MATCH (e:Entity) WHERE e.community IS NOT NULL RETURN count(DISTINCT e.community) AS c"
            ).single()["c"]
            print(f"  [{'Neo4j':20s}] {n_nodes} entities, {n_rels} relations, {n_communities} communities")
        graph_store.close()
    except Exception as e:
        print(f"  [{'Neo4j':20s}] Connection failed: {e}")

    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triplet-Extract + Neo4j GraphRAG Pipeline")
    parser.add_argument(
        "--mode", "-m",
        choices=["full", "continue", "reports_only", "rebuild_embeddings", "query"],
        default="continue",
        help="Pipeline operation mode: 'full' (start over), 'continue' (from community detection), "
             "'reports_only' (regenerate clustering and reports), 'rebuild_embeddings' (only update vectors), "
             "or 'query' (interactive mode using existing cache)."
    )
    args = parser.parse_args()
    
    MODE = args.mode

    print("=" * 60)
    print("  Triplet-Extract + Neo4j GraphRAG Pipeline")
    print("=" * 60)
    print(f"  Mode     : {MODE}")
    print(f"  Neo4j    : {NEO4J_URI} (user: {NEO4J_USER})")
    print(f"  LLM      : {LLAMA_SERVER_BASE_URL} ({MODEL})")
    print(f"  Input    : {INPUT_FILE}")
    print(f"  Cache    : {CACHE_DIR}")
    print()

    if MODE == "full":
        run_full_pipeline()
    elif MODE == "continue":
        run_continue_pipeline()
    elif MODE == "reports_only":
        run_reports_only()
    elif MODE == "rebuild_embeddings":
        run_rebuild_embeddings()
    elif MODE == "query":
        pass  # just query below

    debug_state()

    # Run queries if we have the data
    reports = load_json("community_reports.json")
    index = load_json("embedding_index.json")
    if reports and index:
        run_queries()
    elif MODE != "full":
        log.warning("No reports/index found. Run with MODE='full' first.")
