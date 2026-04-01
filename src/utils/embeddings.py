"""Embedding index building and semantic ranking utilities."""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.embeddings import OpenAIEmbeddings

from src.utils.tokens import count_tokens, split_texts_into_token_batches

logger = logging.getLogger("context_engineering.utils.embeddings")

# FAISS optional
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    import math


def embed_documents_token_safe(embedder, texts: list, max_tokens: int = 100000,
                                char_trunc: int = 30000) -> list:
    """Embed documents in token-safe batches."""
    batches = split_texts_into_token_batches(texts, max_tokens, char_trunc)
    all_embeddings = []
    for batch in batches:
        try:
            embs = embedder.embed_documents(batch)
            all_embeddings.extend(embs)
        except Exception as e:
            logger.warning(f"Embedding batch failed: {e}")
            all_embeddings.extend([[0.0] * 3072] * len(batch))
    return all_embeddings


def build_embedding_index(
    nodes_by_id: Dict[str, Dict[str, Any]],
    embed_model: str,
    embeddings_cache_dir: str,
    repo_slug: str,
    commit: str,
    exclude_tests: bool = True,
    exclude_dirs: list = None,
):
    """Build a FAISS embedding index for non-test production nodes.

    Returns (faiss_index, node_ids_list, embedder) or (None, [], None) on failure.
    """
    from src.graph.filters import path_contains_excluded_dir, is_test_node

    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available — semantic ranking disabled")
        return None, [], None

    exclude_dirs = exclude_dirs or []

    # Cache key
    cache_key = f"{repo_slug}_{commit}"
    cache_dir = os.path.join(embeddings_cache_dir, cache_key)
    index_path = os.path.join(cache_dir, "faiss.index")
    ids_path = os.path.join(cache_dir, "node_ids.json")

    embedder = OpenAIEmbeddings(model=embed_model)

    # Try loading from cache
    if os.path.exists(index_path) and os.path.exists(ids_path):
        try:
            index = faiss.read_index(index_path)
            with open(ids_path, "r") as f:
                node_ids = json.load(f)
            logger.info(f"Loaded embedding cache: {len(node_ids)} nodes from {cache_dir}")
            return index, node_ids, embedder
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

    # Build new index
    candidate_ids = []
    candidate_texts = []

    for nid, node in nodes_by_id.items():
        ntype = node.get("type", "")
        if ntype not in ("class", "function"):
            continue
        path = node.get("path", "") or nid
        if exclude_tests and is_test_node(nid, path):
            continue
        if path_contains_excluded_dir(path, exclude_dirs):
            continue

        code = node.get("code", "") or ""
        text = f"{nid}\n{code[:2000]}" if code else nid
        candidate_ids.append(nid)
        candidate_texts.append(text)

    if not candidate_ids:
        logger.warning("No candidates for embedding index")
        return None, [], None

    logger.info(f"Building embedding index for {len(candidate_ids)} nodes...")
    embeddings = embed_documents_token_safe(embedder, candidate_texts)

    if not embeddings:
        return None, [], None

    dim = len(embeddings[0])
    np_embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(np_embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(np_embeddings)

    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    faiss.write_index(index, index_path)
    with open(ids_path, "w") as f:
        json.dump(candidate_ids, f)
    logger.info(f"Saved embedding cache: {len(candidate_ids)} nodes to {cache_dir}")

    return index, candidate_ids, embedder


def semantic_rank_by_embedding(query: str, embedder, faiss_index, node_ids: list,
                                top_k: int = 5) -> List[Tuple[str, float]]:
    """Rank nodes by semantic similarity to query using FAISS index."""
    if not FAISS_AVAILABLE or faiss_index is None:
        return []

    try:
        q_emb = embedder.embed_query(query)
        q_np = np.array([q_emb], dtype="float32")
        faiss.normalize_L2(q_np)
        scores, indices = faiss_index.search(q_np, min(top_k, len(node_ids)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(node_ids):
                results.append((node_ids[idx], float(score)))
        return results
    except Exception as e:
        logger.warning(f"Semantic ranking failed: {e}")
        return []
