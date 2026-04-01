"""Code navigation tools: subgraph, get_code, file_context, search."""

import json
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import src.state as state
from src.utils.io import run_cmd

logger = logging.getLogger("context_engineering.tools.code_navigation")


# ---------- subgraph ----------

def get_subgraph_internal(node_id: str, hops: int = 2, max_nodes: int = 40) -> dict:
    """BFS traversal from node_id in the adjacency graph."""
    if node_id not in state.adjacency and node_id not in state.nodes_by_id:
        # Try fuzzy match
        for nid in state.nodes_by_id:
            if nid.endswith(node_id) or node_id in nid:
                node_id = nid
                break
        else:
            return {"error": f"Node '{node_id}' not found in graph"}

    visited = set()
    queue = deque([(node_id, 0)])
    result_nodes = []
    result_edges = []

    while queue and len(visited) < max_nodes:
        current, depth = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        nd = state.nodes_by_id.get(current, {})
        result_nodes.append({
            "id": current,
            "type": nd.get("type", "unknown"),
            "path": nd.get("path", ""),
        })

        if depth < hops:
            for neighbor, etype in state.adjacency.get(current, []):
                if neighbor not in visited:
                    result_edges.append({
                        "source": current, "target": neighbor, "type": etype
                    })
                    queue.append((neighbor, depth + 1))

    return {"center": node_id, "nodes": result_nodes, "edges": result_edges}


def tool_get_subgraph(input_str: str) -> str:
    """Explore call graph relationships around a node."""
    try:
        inp = json.loads(input_str)
        node_id = inp.get("node") or inp.get("node_id") or input_str
    except (json.JSONDecodeError, TypeError):
        node_id = input_str.strip()

    hops = state.config.subgraph_hops
    max_nodes = state.config.subgraph_max_nodes
    result = get_subgraph_internal(node_id, hops, max_nodes)

    if "error" in result:
        return result["error"]

    lines = [f"Subgraph around '{result['center']}' ({len(result['nodes'])} nodes, {len(result['edges'])} edges):"]
    lines.append("\nNodes:")
    for n in result["nodes"][:30]:
        lines.append(f"  [{n['type']}] {n['id']}")
    lines.append("\nEdges:")
    for e in result["edges"][:30]:
        lines.append(f"  {e['source']} --{e['type']}--> {e['target']}")
    return "\n".join(lines)


# ---------- get_code ----------

def _fuzzy_find_node(query: str) -> Optional[str]:
    """Find a node ID by exact match, suffix match, or substring match."""
    if query in state.nodes_by_id:
        return query

    # Suffix match (e.g., "MyClass.method" matches "file.py::MyClass.method")
    for nid in state.nodes_by_id:
        if nid.endswith(query) or nid.endswith("::" + query):
            return nid

    # Substring match
    query_lower = query.lower()
    for nid in state.nodes_by_id:
        if query_lower in nid.lower():
            return nid

    # Try matching just the last segment
    query_parts = query.replace("::", ".").split(".")
    query_last = query_parts[-1].lower() if query_parts else ""
    if query_last:
        for nid in state.nodes_by_id:
            nid_parts = nid.replace("::", ".").split(".")
            if nid_parts and nid_parts[-1].lower() == query_last:
                return nid

    return None


def tool_get_code(input_str: str) -> str:
    """Fetch source code for a method/class/file from the code graph."""
    try:
        inp = json.loads(input_str)
        query = inp.get("node") or inp.get("node_id") or inp.get("name") or input_str
    except (json.JSONDecodeError, TypeError):
        query = input_str.strip()

    node_id = _fuzzy_find_node(query)
    if not node_id:
        return f"Node '{query}' not found in code graph. Try get_subgraph or search_codebase to discover available nodes."

    node = state.nodes_by_id[node_id]
    code = node.get("code", "")
    ntype = node.get("type", "unknown")
    path = node.get("path", "")

    state.method_cache_global.add(node_id)
    state.method_cache[node_id] = code

    if not code:
        return f"[{ntype}] {node_id} ({path}) — no source code available in graph"

    header = f"[{ntype}] {node_id}\nFile: {path}\n{'='*60}\n"
    return header + code


# ---------- file_context ----------

def tool_get_file_context(input_str: str) -> str:
    """Read a window of lines from a source file at the correct commit."""
    try:
        inp = json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        return "Invalid input. Expected JSON: {\"file\": \"path/to/file.py\", \"start_line\": 1, \"end_line\": 500}"

    file_path = inp.get("file", "")
    start_line = inp.get("start_line", 1)
    end_line = inp.get("end_line", 500)

    if not file_path:
        return "Missing 'file' field in input."

    reg = state.current_reg_entry or {}
    commit = reg.get("base_commit", "")
    repo_path = state.config.repo_local_path

    if not commit or not repo_path:
        return "No commit or repo path configured."

    rc, content, err = run_cmd(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=repo_path, timeout=30
    )
    if rc != 0:
        return f"Could not read {file_path} at commit {commit[:8]}: {err}"

    lines = content.splitlines()
    selected = lines[max(0, start_line - 1):end_line]
    numbered = [f"{i + start_line}: {line}" for i, line in enumerate(selected)]
    header = f"File: {file_path} (commit {commit[:8]}) lines {start_line}-{min(end_line, len(lines))}/{len(lines)}\n"
    return header + "\n".join(numbered)


# ---------- search_codebase ----------

def tool_search_codebase(input_str: str) -> str:
    """Grep the repository for a pattern."""
    try:
        inp = json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        inp = {"pattern": input_str.strip()}

    pattern = inp.get("pattern", "")
    include = inp.get("include", "*.py")

    if not pattern:
        return "Missing 'pattern' field."

    reg = state.current_reg_entry or {}
    commit = reg.get("base_commit", "")
    repo_path = state.config.repo_local_path

    if not repo_path:
        return "No repo path configured."

    # Use git grep if commit is available, otherwise regular grep
    if commit:
        cmd = ["git", "grep", "-n", "--max-count=50", pattern, commit, "--", include]
    else:
        cmd = ["grep", "-rn", "--max-count=50", f"--include={include}", pattern, "."]

    rc, out, err = run_cmd(cmd, cwd=repo_path, timeout=60)

    if rc != 0 and not out:
        return f"No matches found for pattern '{pattern}' in {include}"

    lines = out.strip().splitlines()
    if len(lines) > 50:
        lines = lines[:50]
        lines.append(f"[... truncated, showing first 50 of many matches]")

    return f"Search results for '{pattern}' in {include}:\n" + "\n".join(lines)
