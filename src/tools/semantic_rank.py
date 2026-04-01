"""Semantic ranking tool: rank code entities by relevance to a query."""

import json
import logging
from typing import List, Tuple

import src.state as state
from src.utils.embeddings import semantic_rank_by_embedding
from src.utils.bm25 import bm25_rank_query

logger = logging.getLogger("context_engineering.tools.semantic_rank")


def tool_semantic_rank(query: str) -> str:
    """Rank code entities by semantic similarity to the query.

    Uses BM25 or embedding+FAISS based on config.
    """
    top_k = state.config.embed_top_k

    if state.config.use_bm25_ranking:
        if not state.bm25_candidate_ids:
            return "BM25 index not built. No candidates available."
        results = bm25_rank_query(
            query, state.bm25_candidate_ids, state.bm25_index, top_k=top_k
        )
    else:
        if state.faiss_index is None:
            return "Embedding index not built. No candidates available."
        results = semantic_rank_by_embedding(
            query, state.embedder, state.faiss_index,
            state.embedding_node_ids, top_k=top_k
        )

    if not results:
        return "No results found."

    lines = [f"Top-{len(results)} ranked code entities:"]
    for i, (node_id, score) in enumerate(results, 1):
        node = state.nodes_by_id.get(node_id, {})
        ntype = node.get("type", "unknown")
        path = node.get("path", "")
        lines.append(f"  {i}. [{ntype}] {node_id} (score={score:.4f}) — {path}")

    return "\n".join(lines)
