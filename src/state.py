"""Mutable session state shared across tools and agents.

Each instance processing cycle resets this state via `reset_instance_state()`.
The tools (classify, get_code, etc.) read from this module directly.
"""

from typing import Any, Dict, List, Optional, Set

# --- Code graph state (set per-instance) ---
nodes_by_id: Dict[str, Dict[str, Any]] = {}
edges_list: list = []
adjacency: Dict[str, list] = {}

# --- Embedding state (set per-repo+commit) ---
faiss_index = None
embedding_node_ids: List[str] = []
embedder = None

# --- BM25 state ---
bm25_candidate_ids: list = []
bm25_docs_texts: list = []
bm25_index: dict = {}

# --- Per-instance working state ---
method_cache_global: Set[str] = set()     # IDs of methods fetched
method_cache: Dict[str, str] = {}          # node_id -> source code
classification_stats: Dict[str, dict] = {}

# --- Current registry entry (for tools to access repo path, commit, etc.) ---
current_reg_entry: Optional[Dict[str, Any]] = None

# --- Chat history tracing ---
active_chat_history: Optional[List[str]] = None
trace_include_observations: bool = True

# --- Config reference (set at startup) ---
config = None  # will be set to Config instance
llm = None     # will be set to ChatOpenAI instance


def reset_instance_state():
    """Reset per-instance mutable state before processing a new instance."""
    global method_cache_global, method_cache, current_reg_entry
    global active_chat_history, trace_include_observations
    method_cache_global = set()
    method_cache = {}
    current_reg_entry = None
    active_chat_history = None
    trace_include_observations = True


def reset_graph_state():
    """Reset code graph state (called when switching repo+commit)."""
    global nodes_by_id, edges_list, adjacency
    global faiss_index, embedding_node_ids, embedder
    global bm25_candidate_ids, bm25_docs_texts, bm25_index
    nodes_by_id = {}
    edges_list = []
    adjacency = {}
    faiss_index = None
    embedding_node_ids = []
    embedder = None
    bm25_candidate_ids = []
    bm25_docs_texts = []
    bm25_index = {}
