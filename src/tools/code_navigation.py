"""Code navigation tools: subgraph, get_code, file_context, search, get_method."""

import json
import logging
import re
import textwrap
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


# ---------- class skeleton extraction ----------

def _read_file_content(file_path: str) -> Optional[str]:
    """Read full file content from git at the current commit."""
    reg = state.current_reg_entry or {}
    commit = reg.get("base_commit", "")
    repo_path = state.config.repo_local_path
    if not commit or not repo_path:
        return None
    rc, content, err = run_cmd(
        ["git", "show", f"{commit}:{file_path}"],
        cwd=repo_path, timeout=30,
    )
    return content if rc == 0 else None


def _find_class_for_method(file_content: str, method_name: str) -> Optional[str]:
    """Find which class a method belongs to in a file.

    Returns the class name, or None if the method is a module-level function.
    """
    current_class = None
    for line in file_content.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Top-level class definition (indent 0)
        m = re.match(r'^class\s+(\w+)', stripped)
        if m and indent == 0:
            current_class = m.group(1)
            continue

        # Method definition inside a class (indent > 0)
        m = re.match(r'(?:def|async\s+def)\s+(\w+)\s*\(', stripped)
        if m and m.group(1) == method_name and indent > 0 and current_class:
            return current_class

    return None


def build_class_skeleton(file_content: str, class_name: str) -> str:
    """Build a skeleton for a class: imports, class declaration, variables,
    decorators, and method signatures (without method bodies).

    This gives the agent the full structural picture of the class while
    keeping token count very low.
    """
    lines = file_content.splitlines()

    # Phase 1: Collect module-level imports (top of file, before any class)
    imports = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
        elif stripped.startswith("class ") or (stripped.startswith("def ") and not stripped.startswith("def _")):
            # Stop at first class or public function
            break
        elif stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''"):
            # Module-level assignment or constant — could be relevant
            if "=" in stripped and not stripped.startswith("("):
                imports.append(line)

    # Phase 2: Find the target class and extract its skeleton
    class_start = None
    class_indent = None
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)class\s+' + re.escape(class_name) + r'\b', line)
        if m:
            class_start = i
            class_indent = len(m.group(1))
            break

    if class_start is None:
        return f"# Class '{class_name}' not found in file"

    skeleton_lines = []
    i = class_start
    in_method_body = False
    method_body_indent = None

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Past the end of the class (back to class indent level or less, non-empty)
        if i > class_start and stripped and indent <= class_indent and not stripped.startswith("#"):
            # Check if it's another class or top-level code
            break

        if not stripped:
            # Blank line — include if we're not inside a method body
            if not in_method_body:
                skeleton_lines.append("")
            i += 1
            continue

        # Detect method/function definitions inside the class
        is_def = bool(re.match(r'(?:def|async\s+def)\s+\w+\s*\(', stripped))
        is_decorator = stripped.startswith("@")

        if is_def and indent > class_indent:
            # This is a method definition
            in_method_body = True
            method_body_indent = indent

            # Include the full signature (may span multiple lines)
            sig_lines = [line]
            # Handle multi-line signatures
            while i + 1 < len(lines) and not lines[i].rstrip().endswith(":"):
                i += 1
                sig_lines.append(lines[i])
            skeleton_lines.extend(sig_lines)

            # Add a placeholder for the body
            body_indent = " " * (indent + 4)
            skeleton_lines.append(f"{body_indent}...")

            i += 1
            continue

        if in_method_body:
            # Skip lines that are part of the method body
            if indent > method_body_indent:
                i += 1
                continue
            else:
                # Exited the method body
                in_method_body = False
                method_body_indent = None

        # Class-level content: class line, docstrings, class vars, decorators
        if is_decorator:
            skeleton_lines.append(line)
        elif stripped.startswith("class "):
            skeleton_lines.append(line)
        elif indent > class_indent and not in_method_body:
            # Class-level content (variables, docstrings, etc.)
            # Include docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                skeleton_lines.append(line)
                quote = stripped[:3]
                # If docstring doesn't close on same line, consume until close
                if stripped.count(quote) == 1:
                    i += 1
                    while i < len(lines):
                        skeleton_lines.append(lines[i])
                        if quote in lines[i]:
                            break
                        i += 1
            else:
                # Class variable or assignment
                skeleton_lines.append(line)

        i += 1

    # Assemble
    parts = []
    if imports:
        parts.append("# === Module imports ===")
        parts.extend(imports)
        parts.append("")
    parts.append(f"# === Class skeleton for {class_name} (method bodies replaced with '...') ===")
    parts.extend(skeleton_lines)

    return "\n".join(parts)


def _extract_method_code(file_content: str, class_name: str, method_name: str) -> Optional[str]:
    """Extract the full source code of a specific method from a class."""
    lines = file_content.splitlines()

    # Find the class first
    class_start = None
    class_indent = None
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)class\s+' + re.escape(class_name) + r'\b', line)
        if m:
            class_start = i
            class_indent = len(m.group(1))
            break

    if class_start is None:
        return None

    # Find the method inside the class
    method_start = None
    method_indent = None
    for i in range(class_start + 1, len(lines)):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Past end of class
        if stripped and indent <= class_indent and i > class_start:
            break

        m = re.match(r'(?:def|async\s+def)\s+' + re.escape(method_name) + r'\s*\(', stripped)
        if m and indent > class_indent:
            # Check for decorators above (including multi-line decorators)
            dec_start = i
            while dec_start > class_start + 1:
                prev = lines[dec_start - 1].lstrip()
                if prev.startswith("@"):
                    dec_start -= 1
                elif prev and not prev.startswith("def ") and not prev.startswith("async def "):
                    # Could be continuation of a multi-line decorator —
                    # keep walking back to see if we hit an '@' line
                    peek = dec_start - 1
                    found_decorator = False
                    while peek > class_start:
                        peek_line = lines[peek - 1].lstrip()
                        if peek_line.startswith("@"):
                            found_decorator = True
                            break
                        elif not peek_line or peek_line.startswith("def ") or peek_line.startswith("async def ") or peek_line.startswith("class "):
                            break
                        peek -= 1
                    if found_decorator:
                        dec_start = peek
                    else:
                        break
                else:
                    break
            method_start = dec_start
            method_indent = indent
            break

    if method_start is None:
        return None

    # Collect the method body (everything at indent > method_indent, plus the def line)
    method_lines = []
    i = method_start
    started_body = False
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if i == method_start:
            method_lines.append(line)
            i += 1
            continue

        # Handle multi-line signature
        if not started_body and not lines[i - 1].rstrip().endswith(":"):
            method_lines.append(line)
            i += 1
            continue

        started_body = True

        # Past end of method
        if stripped and indent <= method_indent:
            break

        method_lines.append(line)
        i += 1

    # Add line numbers
    numbered = []
    for j, ml in enumerate(method_lines):
        numbered.append(f"{method_start + j + 1}: {ml}")

    return "\n".join(numbered)


# ---------- get_method tool ----------

def tool_get_method(input_str: str) -> str:
    """Retrieve a specific method's full source code along with its class skeleton.

    Returns:
      1. The full source of the requested method (with line numbers)
      2. The class skeleton: imports, class declaration, class variables,
         and all method signatures (bodies replaced with '...')

    This gives you the exact code you need plus the full structural context
    of the class it belongs to.
    """
    try:
        inp = json.loads(input_str)
    except (json.JSONDecodeError, TypeError):
        return ("Invalid input. Expected JSON: "
                "{\"file\": \"path/to/file.py\", \"class_name\": \"MyClass\", \"method_name\": \"my_method\"}")

    file_path = inp.get("file", "")
    method_name = inp.get("method_name", "")
    class_name = inp.get("class_name", "")

    if not file_path or not method_name:
        return "Missing required fields. Need 'file' and 'method_name'."

    # Read full file
    file_content = _read_file_content(file_path)
    if file_content is None:
        reg = state.current_reg_entry or {}
        commit = reg.get("base_commit", "")[:8]
        return f"Could not read {file_path} at commit {commit}"

    # Auto-detect class if not provided
    if not class_name:
        class_name = _find_class_for_method(file_content, method_name)
        if not class_name:
            # It may be a module-level function — fall back to get_file_context
            return (f"Method '{method_name}' does not appear to be inside a class in {file_path}. "
                    f"Use get_file_context to read it directly, or provide 'class_name'.")

    # Extract the method code
    method_code = _extract_method_code(file_content, class_name, method_name)
    if method_code is None:
        return (f"Could not find method '{method_name}' in class '{class_name}' in {file_path}. "
                f"Use search_codebase to verify the method name and file.")

    # Build the class skeleton
    skeleton = build_class_skeleton(file_content, class_name)

    # Cache the method code
    cache_key = f"{file_path}:{class_name}.{method_name}"
    state.method_cache[cache_key] = method_code
    state.method_cache_global.add(cache_key)

    # Also cache the full file for later use
    state.method_cache[file_path] = file_content
    state.method_cache_global.add(file_path)

    # Build response
    reg = state.current_reg_entry or {}
    commit = reg.get("base_commit", "")[:8]

    parts = [
        f"=== Method: {class_name}.{method_name} ===",
        f"File: {file_path} (commit {commit})",
        "",
        method_code,
        "",
        f"=== Class skeleton for {class_name} (all method signatures, no bodies) ===",
        f"Use this skeleton to understand the full class structure.",
        f"To read any other method's full code, call get_method again with that method name.",
        "",
        skeleton,
    ]

    return "\n".join(parts)
