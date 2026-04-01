"""Code graph loading and adjacency building."""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("context_engineering.graph.loader")


def load_codegraph(path: str) -> Optional[Dict[str, Any]]:
    """Load a codegraph JSON file and return the parsed dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded codegraph: {path} ({len(data.get('nodes', []))} nodes)")
        return data
    except Exception as e:
        logger.error(f"Failed to load codegraph {path}: {e}")
        return None


def build_edge_adjacency(edges: list) -> Dict[str, List[Tuple[str, str]]]:
    """Build an adjacency list from edges.

    Returns dict mapping node_id -> list of (neighbor_id, edge_type).
    """
    adj = defaultdict(list)
    for edge in edges:
        src = edge.get("source", "")
        tgt = edge.get("target", "")
        etype = edge.get("type", "")
        adj[src].append((tgt, etype))
        adj[tgt].append((src, etype))
    return dict(adj)
