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
    """Find a node ID by exact match, suffix match, dotted-path mapping, or substring match.

    Node IDs in the codegraph use single-colon format:
        astropy/modeling/separable.py:separability_matrix
    But agents may query with dotted module paths:
        astropy.modeling.separable.separability_matrix
    """
    query = query.strip().replace("'", "").replace('"', '').replace("`", "")

    if not query or not state.nodes_by_id:
        return None

    # 0. Exact match
    if query in state.nodes_by_id:
        return query

    # 1. Check method_cache first (already fetched nodes)
    for cached_node in state.method_cache:
        if cached_node.endswith(query) or cached_node.split(':')[-1].endswith(query):
            return cached_node

    # 2. Colon-suffix match: entity_part after colon matches query
    #    e.g. query="separability_matrix" matches "astropy/modeling/separable.py:separability_matrix"
    matches = []
    for nid in state.nodes_by_id:
        # Exclude test nodes
        if "/tests/" in nid or "test_" in nid or "tests/" in nid:
            continue
        if ':' in nid:
            entity_part = nid.split(':', 1)[1]
            if entity_part == query or entity_part.endswith("." + query):
                matches.append(nid)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Return None to trigger ambiguous response
        return None

    # 3. Dotted module path → file path mapping
    #    e.g. astropy.modeling.separable.separability_matrix
    #       → astropy/modeling/separable.py:separability_matrix
    if "." in query:
        parts = query.split(".")
        if len(parts) >= 2:
            func_or_class = parts[-1]
            module_path = "/".join(parts[:-1]) + ".py"
            dotted_suffix = module_path + ":" + func_or_class
            for nid in state.nodes_by_id:
                if "/tests/" in nid or "test_" in nid or "tests/" in nid:
                    continue
                if nid.endswith(dotted_suffix):
                    matches.append(nid)

            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None

    # 4. General suffix match (e.g., "MyClass.method" or "::MyClass.method")
    for nid in state.nodes_by_id:
        if nid.endswith(query) or nid.endswith(":" + query) or nid.endswith("::" + query):
            return nid

    # 5. Substring match (case-insensitive)
    query_lower = query.lower()
    for nid in state.nodes_by_id:
        if query_lower in nid.lower():
            return nid

    # 6. Last segment match
    query_parts = query.replace("::", ".").replace(":", ".").split(".")
    query_last = query_parts[-1].lower() if query_parts else ""
    if query_last:
        for nid in state.nodes_by_id:
            nid_parts = nid.replace("::", ".").replace(":", ".").split(".")
            if nid_parts and nid_parts[-1].lower() == query_last:
                return nid

    return None


def _find_ambiguous_matches(query: str) -> List[str]:
    """Return multiple matching node IDs for disambiguation."""
    query = query.strip().replace("'", "").replace('"', '').replace("`", "")
    matches = []
    for nid in state.nodes_by_id:
        if "/tests/" in nid or "test_" in nid or "tests/" in nid:
            continue
        if ':' in nid:
            entity_part = nid.split(':', 1)[1]
            if entity_part == query or entity_part.endswith("." + query):
                matches.append(nid)
    # Also check dotted path
    if not matches and "." in query:
        parts = query.split(".")
        if len(parts) >= 2:
            func_or_class = parts[-1]
            module_path = "/".join(parts[:-1]) + ".py"
            dotted_suffix = module_path + ":" + func_or_class
            for nid in state.nodes_by_id:
                if "/tests/" in nid or "test_" in nid or "tests/" in nid:
                    continue
                if nid.endswith(dotted_suffix):
                    matches.append(nid)
    return matches


def tool_get_code(input_str: str) -> str:
    """Fetch source code for a method/class/file from the code graph."""
    try:
        inp = json.loads(input_str)
        query = inp.get("node") or inp.get("node_id") or inp.get("name") or input_str
    except (json.JSONDecodeError, TypeError):
        query = input_str.strip()

    query = query.strip().replace("'", "").replace('"', '').replace("`", "")

    node_id = _fuzzy_find_node(query)
    if not node_id:
        # Check if there are ambiguous matches
        ambiguous = _find_ambiguous_matches(query)
        if ambiguous:
            return json.dumps({
                "error": "ambiguous",
                "node": query,
                "candidates": ambiguous,
                "hint": "Multiple matches found. Please use the full node ID from the candidates list."
            })
        return json.dumps({
            "error": "node_not_found",
            "node": query,
            "hint": "Try search_codebase to find the correct node ID, or use get_file_context to read the file directly."
        })

    node = state.nodes_by_id[node_id]
    code = node.get("code", "")
    ntype = node.get("type", "unknown")
    path = node.get("path", "")

    state.method_cache_global.add(node_id)
    state.method_cache[node_id] = code

    if not code:
        return json.dumps({"node": node_id, "code": "", "type": ntype, "info": "No source code available in graph"})

    return json.dumps({"node": node_id, "code": code, "type": ntype})


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

    # Cache the file content so it's available for report generation,
    # same as get_code does for codegraph nodes.
    cache_key = file_path
    read_content = "\n".join(selected)
    state.method_cache[cache_key] = read_content
    state.method_cache_global.add(cache_key)

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
