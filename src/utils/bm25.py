"""BM25 ranking utilities for code entity matching."""

import math
import re
from typing import Any, Dict, List, Tuple

from src.graph.filters import path_contains_excluded_dir, is_test_node


def _simple_tokenize_for_bm25(s: str):
    """Simple tokenization for BM25 (lowercased alpha-numeric + underscores)."""
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9_]+", s)]


def bm25_prepare_candidates(nodes_by_id: Dict[str, Dict[str, Any]],
                             exclude_tests_flag: bool = True,
                             exclude_dirs: list = None) -> Tuple[list, list, dict]:
    """Build candidate node list + BM25 index.

    Returns (candidate_ids, docs_texts, bm25_index).
    """
    exclude_dirs = exclude_dirs or []
    candidate_ids = []
    docs_texts = []

    for nid, node in nodes_by_id.items():
        ntype = node.get("type", "")
        if ntype not in ("class", "function"):
            continue
        path = node.get("path", "") or nid
        if exclude_tests_flag and is_test_node(nid, path):
            continue
        if path_contains_excluded_dir(path, exclude_dirs):
            continue

        code = node.get("code", "") or ""
        text = f"{nid} {code[:1000]}"
        candidate_ids.append(nid)
        docs_texts.append(text)

    # Build IDF
    N = len(docs_texts)
    df = {}
    tokenized_docs = []
    doc_lengths = []

    for text in docs_texts:
        tokens = _simple_tokenize_for_bm25(text)
        tokenized_docs.append(tokens)
        doc_lengths.append(len(tokens))
        seen = set()
        for t in tokens:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)

    avgdl = sum(doc_lengths) / max(N, 1)
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    bm25_index = {
        "tokenized_docs": tokenized_docs,
        "doc_lengths": doc_lengths,
        "idf": idf,
        "avgdl": avgdl,
        "N": N,
    }

    return candidate_ids, docs_texts, bm25_index


def bm25_rank_query(query: str, candidate_ids: list, bm25_index: dict,
                     top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
    """Rank candidates by BM25 score for the given query."""
    query_tokens = _simple_tokenize_for_bm25(query)
    idf = bm25_index["idf"]
    tokenized_docs = bm25_index["tokenized_docs"]
    doc_lengths = bm25_index["doc_lengths"]
    avgdl = bm25_index["avgdl"]

    scores = []
    for i, doc_tokens in enumerate(tokenized_docs):
        tf_map = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        dl = doc_lengths[i]
        for qt in query_tokens:
            if qt not in idf:
                continue
            tf = tf_map.get(qt, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf[qt] * numerator / max(denominator, 1e-9)

        scores.append((candidate_ids[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
