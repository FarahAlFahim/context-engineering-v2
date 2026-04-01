#!/usr/bin/env python3
"""
build_code_and_call_graphs.py

Single-file tool to:
 - read SWE-Bench-Lite per-repo JSON (list of instances with base_commit)
 - for each unique commit: checkout local repo to that commit (worktree or inplace)
 - build a heterogeneous code graph (directory/file/class/function) with edges:
     contains, imports, invokes, inherits
 - derive a call graph (method/class nodes only)
 - write per-commit artifacts and a repo-level index (registry)

Outputs:
 data/code_graph/<repo_slug>/<commit>/codegraph.json
 data/code_graph/<repo_slug>/<commit>/callgraph.json
 data/code_graph/<repo_slug>.json  <-- index mapping instance_id -> paths

Notes:
 - Edit the top config variables for each repo and run the script.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# -------------------- USER CONFIGURE HERE --------------------
input_file = "data/by_repo/sympy__sympy.json"

# base dir where repo-level index(s) & per-commit artifact folders will be written
code_graph_base_dir = "data/code_graph"

# local clone path for repository 
# astropy__astropy.json 
repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/astropy"
git_branch = "main"

# for django__django.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/django"
# git_branch = "main"

# for matplotlib__matplotlib.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/matplotlib"
# git_branch = "main"

# for mwaskom__seaborn.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/seaborn"
# git_branch = "master"

# for pallets__flask.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/flask"
# git_branch = "main"

# for psf__requests.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/requests"
# git_branch = "main"

# for pydata__xarray.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/xarray"
# git_branch = "main"

# for pylint-dev__pylint.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/pylint"
# git_branch = "main"

# for pytest-dev__pytest.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/pytest"
# git_branch = "main"

# for scikit-learn__scikit-learn.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/scikit-learn"
# git_branch = "main"

# for sphinx-doc__sphinx.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/sphinx"
# git_branch = "master"

# for sympy__sympy.json
# repo_local_path = "/Users/fahim/Desktop/PhD/swe_bench/sympy"
# git_branch = "master"




# enable worktree (safe) or use in-place checkout (you said no local changes)
USE_WORKTREE = False

# where to place temporary worktrees (if using worktree)
worktrees_cache_parent = os.path.join(os.path.dirname(repo_local_path), ".worktrees_cache")

# If True remove worktree after building commit artifacts (saves disk)
CLEAN_WORKTREE_AFTER = True

# git commands timeout (seconds)
GIT_CMD_TIMEOUT = 10 * 60  # 10 minutes by default

# maximum recursion depth for inner-node exploration (safety)
sys.setrecursionlimit(30000)

# Debug small-run options:
DEBUG_MAX_COMMITS = None  # set to 1 to limit to first commit for debugging
# ------------------------------------------------------------

# ---------- constants ----------
NODE_TYPE_DIRECTORY = 'directory'
NODE_TYPE_FILE = 'file'
NODE_TYPE_CLASS = 'class'
NODE_TYPE_FUNCTION = 'function'

EDGE_TYPE_CONTAINS = 'contains'
EDGE_TYPE_IMPORTS = 'imports'
EDGE_TYPE_INVOKES = 'invokes'
EDGE_TYPE_INHERITS = 'inherits'

SKIP_DIRS = {'.git', '.github', '__pycache__'}

# ---------------- utils ----------------
def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = GIT_CMD_TIMEOUT):
    """Run a subprocess command with stdout/stderr capture and timeout; raise on error."""
    start = time.time()
    cmd_str = " ".join(cmd)
    try:
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT] cmd='{cmd_str}' cwd={cwd} after {timeout}s")
        raise
    except subprocess.CalledProcessError as e:
        out = (e.stdout or b"").decode("utf-8", errors="ignore")[:400]
        err = (e.stderr or b"").decode("utf-8", errors="ignore")[:400]
        print(f"[CMD ERR] cmd='{cmd_str}' cwd={cwd} rc={e.returncode}")
        print(" stdout:", out)
        print(" stderr:", err)
        raise
    else:
        took = time.time() - start
        print(f"[OK] cmd='{cmd_str}' cwd={cwd} took={took:.1f}s")
        return proc

def is_skip_dir(dirname: str) -> bool:
    for skip in SKIP_DIRS:
        if skip in dirname:
            return True
    return False

def save_json_atomic(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------------- git helpers ----------------
def ensure_worktree_for_commit(repo_path: str, commit: str, cache_dir: str) -> str:
    """
    Create or reuse a git worktree for `commit`. Returns the worktree path.
    Uses run_cmd (with timeout) to avoid hangs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, commit)
    if os.path.isdir(target):
        # quick check: attempt a checkout (no-op if already at commit)
        try:
            run_cmd(["git", "-C", target, "checkout", commit], cwd=None, timeout=120)
            return target
        except Exception:
            # remove and recreate
            try:
                shutil.rmtree(target)
            except Exception:
                pass

    # fetch in origin repo (ensures commit present)
    run_cmd(["git", "-C", repo_path, "fetch", "--all", "--prune"], cwd=None, timeout=GIT_CMD_TIMEOUT)
    # add worktree
    run_cmd(["git", "-C", repo_path, "worktree", "add", "--detach", target, commit], cwd=None, timeout=GIT_CMD_TIMEOUT)
    return target

# Checkout to the specific commit version
def checkout_to_commit(commit_version, repo_path, git_branch):
    # Reset any local changes
    subprocess.run('git reset --hard', shell=True, cwd=repo_path)
    # Ensure a clean stash
    subprocess.run('git stash push --include-untracked', shell=True, cwd=repo_path)
    # Switch back to the main branch before checking out the commit
    subprocess.run(f'git checkout {git_branch}', shell=True, cwd=repo_path)
    # Ensure branch is up-to-date
    subprocess.run('git pull', shell=True, cwd=repo_path)
    # Checkout to the required commit
    subprocess.run(f'git checkout {commit_version}', shell=True, cwd=repo_path)
    # Drop the stash if it's no longer needed
    subprocess.run('git stash drop', shell=True, cwd=repo_path)

# ---------------- AST helpers (optimized) ----------------
def find_imports(filepath: str, repo_path: str, tree: Optional[ast.AST] = None):
    """Return list describing imports in a file or AST subtree."""
    if tree is None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            tree = ast.parse(text, filename=filepath)
        except Exception:
            raise
        candidates = ast.walk(tree)
    else:
        candidates = ast.iter_child_nodes(tree)

    imports = []
    for node in candidates:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "module": alias.name, "alias": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            entities = []
            for alias in node.names:
                if alias.name == '*':
                    entities = [{"name": "*", "alias": None}]
                    break
                entities.append({"name": alias.name, "alias": alias.asname})
            if node.level == 0:
                module_name = node.module
            else:
                rel_path = os.path.relpath(filepath, repo_path)
                package_parts = rel_path.split(os.sep)
                if len(package_parts) >= node.level:
                    package_parts = package_parts[:-node.level]
                else:
                    package_parts = []
                if node.module:
                    module_name = '.'.join(package_parts + [node.module])
                else:
                    module_name = '.'.join(package_parts)
            imports.append({"type": "from", "module": module_name, "entities": entities})
    return imports

class CodeAnalyzer(ast.NodeVisitor):
    """
    Node visitor that stores class/function nodes with source segments.
    Accepts source_text to avoid reopening files repeatedly.
    """
    def __init__(self, filename: str, source_text: Optional[str] = None):
        self.filename = filename
        self._source_text = source_text
        self.nodes = []
        self.name_stack = []
        self.type_stack = []

    def _get_segment(self, node):
        try:
            if self._source_text is not None:
                return ast.get_source_segment(self._source_text, node)
            else:
                with open(self.filename, "r", encoding="utf-8") as f:
                    text = f.read()
                return ast.get_source_segment(text, node)
        except Exception:
            return None

    def visit_ClassDef(self, node: ast.ClassDef):
        class_name = node.name
        full = '.'.join(self.name_stack + [class_name]) if self.name_stack else class_name
        src = self._get_segment(node)
        self.nodes.append({
            "name": full,
            "type": NODE_TYPE_CLASS,
            "code": src,
            "start_line": getattr(node, "lineno", None),
            "end_line": getattr(node, "end_lineno", None)
        })
        self.name_stack.append(class_name)
        self.type_stack.append(NODE_TYPE_CLASS)
        self.generic_visit(node)
        self.name_stack.pop()
        self.type_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # follow original: skip __init__ when parent is class
        if self.type_stack and self.type_stack[-1] == NODE_TYPE_CLASS and node.name == '__init__':
            return
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_func(node)

    def _visit_func(self, node):
        func_name = node.name
        full = '.'.join(self.name_stack + [func_name]) if self.name_stack else func_name
        src = self._get_segment(node)
        self.nodes.append({
            "name": full,
            "type": NODE_TYPE_FUNCTION,
            "parent_type": self.type_stack[-1] if self.type_stack else None,
            "code": src,
            "start_line": getattr(node, "lineno", None),
            "end_line": getattr(node, "end_lineno", None)
        })
        self.name_stack.append(func_name)
        self.type_stack.append(NODE_TYPE_FUNCTION)
        self.generic_visit(node)
        self.name_stack.pop()
        self.type_stack.pop()

def analyze_file_from_code(filename: str, code_text: str):
    """Parse code_text and return class/function nodes with code segments."""
    try:
        tree = ast.parse(code_text, filename=filename)
    except Exception:
        raise
    analyzer = CodeAnalyzer(filename, source_text=code_text)
    try:
        analyzer.visit(tree)
    except RecursionError:
        pass
    return analyzer.nodes

def resolve_module(module_name: str, repo_path: str) -> Optional[str]:
    if not module_name:
        return None
    candidate = os.path.join(repo_path, module_name.replace('.', os.sep) + '.py')
    if os.path.isfile(candidate):
        return candidate
    candidate2 = os.path.join(repo_path, module_name.replace('.', os.sep), '__init__.py')
    if os.path.isfile(candidate2):
        return candidate2
    return None

def add_imports(root_node_id: str, imports, graph_nodes: Dict, graph_edges: List, repo_path: str):
    for imp in imports:
        if imp['type'] == 'import':
            module_name = imp['module']
            path = resolve_module(module_name, repo_path)
            if path:
                rel = os.path.relpath(path, repo_path)
                if rel in graph_nodes:
                    graph_edges.append({"src": root_node_id, "dst": rel, "type": EDGE_TYPE_IMPORTS, "alias": imp.get('alias')})
        elif imp['type'] == 'from':
            module_name = imp['module']
            entities = imp['entities']
            if len(entities) == 1 and entities[0]['name'] == '*':
                path = resolve_module(module_name, repo_path)
                if path:
                    rel = os.path.relpath(path, repo_path)
                    if rel in graph_nodes:
                        graph_edges.append({"src": root_node_id, "dst": rel, "type": EDGE_TYPE_IMPORTS, "alias": None})
                continue
            for ent in entities:
                name = ent['name']
                alias = ent.get('alias')
                fullmod = f"{module_name}.{name}" if module_name else name
                path_ent = resolve_module(fullmod, repo_path)
                if path_ent:
                    rel = os.path.relpath(path_ent, repo_path)
                    if rel in graph_nodes:
                        graph_edges.append({"src": root_node_id, "dst": rel, "type": EDGE_TYPE_IMPORTS, "alias": alias})
                else:
                    path_module = resolve_module(module_name, repo_path)
                    if path_module:
                        relm = os.path.relpath(path_module, repo_path)
                        node_entity = f"{relm}:{name}"
                        if node_entity in graph_nodes:
                            graph_edges.append({"src": root_node_id, "dst": node_entity, "type": EDGE_TYPE_IMPORTS, "alias": alias})
                        elif relm in graph_nodes:
                            graph_edges.append({"src": root_node_id, "dst": relm, "type": EDGE_TYPE_IMPORTS, "alias": alias})

# ---------------- graph builder (optimized) ----------------
def build_code_graph(repo_workdir: str, fuzzy_search: bool = True, global_import: bool = True):
    """
    Build graph_nodes (dict) and graph_edges (list). Also return outgoing/incoming maps
    for fast neighbor queries.
    """
    repo_path = os.path.abspath(repo_workdir)
    graph_nodes: Dict[str, Dict] = {}
    graph_edges: List[Dict] = []

    graph_nodes['/'] = {'id': '/', 'type': NODE_TYPE_DIRECTORY}
    file_nodes = {}  # rel_file -> absolute path
    file_code_map = {}  # rel_file -> str (content)

    t0 = time.time()
    # Walk files and populate nodes & file content once
    file_count = 0
    for root, dirs, files in os.walk(repo_path):
        rel_dir = os.path.relpath(root, repo_path)
        if rel_dir == '.':
            rel_dir = '/'
        elif is_skip_dir(rel_dir):
            continue
        else:
            if rel_dir not in graph_nodes:
                graph_nodes[rel_dir] = {'id': rel_dir, 'type': NODE_TYPE_DIRECTORY}
                parent = os.path.dirname(rel_dir)
                if parent == '':
                    parent = '/'
                graph_edges.append({'src': parent, 'dst': rel_dir, 'type': EDGE_TYPE_CONTAINS})

        for fname in files:
            if not fname.endswith('.py'):
                continue
            file_abs = os.path.join(root, fname)
            if os.path.islink(file_abs):
                continue
            rel_file = os.path.relpath(file_abs, repo_path)
            try:
                with open(file_abs, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            file_count += 1
            file_nodes[rel_file] = file_abs
            file_code_map[rel_file] = content
            graph_nodes[rel_file] = {'id': rel_file, 'type': NODE_TYPE_FILE, 'code': content}

    parse_time = time.time() - t0
    print(f"  Files collected: {file_count}  (walk+read took {parse_time:.1f}s)")

    # Analyze each file once for entities
    t1 = time.time()
    total_entities = 0
    for rel_file, code_text in file_code_map.items():
        try:
            entities = analyze_file_from_code(rel_file, code_text)
        except Exception:
            entities = []
        for ent in entities:
            node_id = f"{rel_file}:{ent['name']}"
            graph_nodes[node_id] = {
                'id': node_id,
                'type': ent['type'],
                'code': ent.get('code'),
                'start_line': ent.get('start_line'),
                'end_line': ent.get('end_line')
            }
            total_entities += 1
        # contains edge file -> entities
        graph_edges.append({'src': os.path.dirname(rel_file) if os.path.dirname(rel_file) != '' else '/', 'dst': rel_file, 'type': EDGE_TYPE_CONTAINS})
        for ent in entities:
            node_id = f"{rel_file}:{ent['name']}"
            name_parts = ent['name'].split('.')
            if len(name_parts) == 1:
                graph_edges.append({'src': rel_file, 'dst': node_id, 'type': EDGE_TYPE_CONTAINS})
            else:
                parent_name = '.'.join(name_parts[:-1])
                parent_node = f"{rel_file}:{parent_name}"
                graph_edges.append({'src': parent_node, 'dst': node_id, 'type': EDGE_TYPE_CONTAINS})

    analyze_time = time.time() - t1
    print(f"  Parsed and analyzed code for entities: {total_entities} entities (took {analyze_time:.1f}s)")

    # imports edges
    t2 = time.time()
    imported_edges = 0
    for rel_file, abs_path in file_nodes.items():
        try:
            imports = find_imports(abs_path, repo_path)
        except Exception:
            imports = []
        prev_edge_count = len(graph_edges)
        add_imports(rel_file, imports, graph_nodes, graph_edges, repo_path)
        imported_edges += (len(graph_edges) - prev_edge_count)
    print(f"  Added import edges: {imported_edges} (took {time.time()-t2:.1f}s)")

    # optional global name map
    global_name_map = defaultdict(list)
    if global_import:
        for nid in graph_nodes:
            if nid.endswith('.py'):
                fname = os.path.basename(nid)
                global_name_map[fname].append(nid)
                modname = os.path.splitext(fname)[0]
                global_name_map[modname].append(nid)
            elif ':' in nid:
                s = nid.split(':')[-1].split('.')[-1]
                global_name_map[s].append(nid)

    # build outgoing/incoming maps for efficient neighbor queries
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for e in graph_edges:
        outgoing[e['src']].append(e)
        incoming[e['dst']].append(e)

    # collect entity nodes
    entity_nodes = [nid for nid, nd in graph_nodes.items() if nd['type'] in (NODE_TYPE_CLASS, NODE_TYPE_FUNCTION)]

    # helpers using adjacency maps (much faster than scanning edge list)
    def get_inner_nodes(query_node: str, src_node: str):
        res = []
        for e in outgoing.get(src_node, []):
            if e['type'] == EDGE_TYPE_CONTAINS and e['dst'] != query_node:
                res.append(e['dst'])
                if graph_nodes.get(e['dst'], {}).get('type') == NODE_TYPE_CLASS:
                    res.extend(get_inner_nodes(query_node, e['dst']))
        return res

    def find_parent(node_name: str):
        for e in incoming.get(node_name, []):
            if e['type'] == EDGE_TYPE_CONTAINS:
                return e['src']
        return None

    def find_all_possible_callee(node_id: str) -> Tuple[List[str], Dict[str, str]]:
        callee_nodes = []
        callee_alias = {}
        cur = node_id
        prev = node_id

        while True:
            callee_nodes.extend(get_inner_nodes(prev, cur))
            if graph_nodes[cur]['type'] == NODE_TYPE_FILE:
                # imported files reachable
                file_list = []
                file_stack = [cur]
                seen = set([cur])
                while file_stack:
                    f0 = file_stack.pop()
                    for e in outgoing.get(f0, []):
                        if e['type'] == EDGE_TYPE_IMPORTS and e['dst'] not in seen:
                            seen.add(e['dst'])
                            if graph_nodes.get(e['dst'], {}).get('type') == NODE_TYPE_FILE and e['dst'].endswith('__init__.py'):
                                file_list.append(e['dst'])
                                file_stack.append(e['dst'])
                for f in file_list:
                    callee_nodes.extend(get_inner_nodes(cur, f))
                    for e in outgoing.get(f, []):
                        if e['type'] == EDGE_TYPE_IMPORTS:
                            alias = e.get('alias')
                            if alias:
                                callee_alias[alias] = e['dst']
                            if graph_nodes.get(e['dst'], {}).get('type') in (NODE_TYPE_FILE, NODE_TYPE_CLASS):
                                callee_nodes.extend(get_inner_nodes(f, e['dst']))
                            if graph_nodes.get(e['dst'], {}).get('type') in (NODE_TYPE_FUNCTION, NODE_TYPE_CLASS):
                                callee_nodes.append(e['dst'])
                # direct imports from cur
                for e in outgoing.get(cur, []):
                    if e['type'] == EDGE_TYPE_IMPORTS:
                        alias = e.get('alias')
                        if alias:
                            callee_alias[alias] = e['dst']
                        if graph_nodes.get(e['dst'], {}).get('type') in (NODE_TYPE_FILE, NODE_TYPE_CLASS):
                            callee_nodes.extend(get_inner_nodes(cur, e['dst']))
                        if graph_nodes.get(e['dst'], {}).get('type') in (NODE_TYPE_FUNCTION, NODE_TYPE_CLASS):
                            callee_nodes.append(e['dst'])
                break
            prev = cur
            cur = find_parent(cur)
            if cur is None:
                break
        return callee_nodes, callee_alias

    # analyze invokes and inherits for each entity
    t3 = time.time()
    invokes_edges_added = 0
    inherits_edges_added = 0

    for node_id in entity_nodes:
        node_meta = graph_nodes[node_id]
        code = node_meta.get('code')
        if not code:
            continue
        try:
            caller_tree = ast.parse(code)
        except Exception:
            continue

        callee_nodes, callee_alias = find_all_possible_callee(node_id)

        if fuzzy_search:
            callee_name_dict = defaultdict(list)
            for c in set(callee_nodes):
                name = c.split(':')[-1].split('.')[-1]
                callee_name_dict[name].append(c)
            for a, c in callee_alias.items():
                callee_name_dict[a].append(c)
        else:
            callee_name_dict = {c.split(':')[-1].split('.')[-1]: c for c in callee_nodes[::-1]}
            callee_name_dict.update(callee_alias)

        # extract invocations and inheritances
        invocations = []
        inheritances = []
        if node_meta['type'] == NODE_TYPE_CLASS:
            caller_name = node_id.split(':')[-1].split('.')[-1]
            for ast_node in ast.walk(caller_tree):
                if isinstance(ast_node, ast.ClassDef) and ast_node.name == caller_name:
                    # imports inside class or init
                    try:
                        imports = find_imports(os.path.join(repo_path, node_id.split(':')[0]), repo_path, tree=ast_node)
                        add_imports(node_id, imports, graph_nodes, graph_edges, repo_path)
                    except Exception:
                        pass
                    for base in ast_node.bases:
                        if isinstance(base, ast.Name):
                            inheritances.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            inheritances.append(base.attr)
                    for deco in ast_node.decorator_list:
                        if isinstance(deco, ast.Name):
                            invocations.append(deco.id)
                        elif isinstance(deco, ast.Attribute):
                            invocations.append(deco.attr)
                    # inspect __init__
                    for body_item in ast_node.body:
                        if isinstance(body_item, ast.FunctionDef) and body_item.name == '__init__':
                            try:
                                imports = find_imports(os.path.join(repo_path, node_id.split(':')[0]), repo_path, tree=body_item)
                                add_imports(node_id, imports, graph_nodes, graph_edges, repo_path)
                            except Exception:
                                pass
                            for sub in ast.walk(body_item):
                                if isinstance(sub, ast.Call):
                                    if isinstance(sub.func, ast.Name):
                                        invocations.append(sub.func.id)
                                    elif isinstance(sub.func, ast.Attribute):
                                        invocations.append(sub.func.attr)
                            break
                    break
        else:
            caller_name = node_id.split(':')[-1].split('.')[-1]
            for ast_node in ast.walk(caller_tree):
                if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast_node.name == caller_name:
                    try:
                        imports = find_imports(os.path.join(repo_path, node_id.split(':')[0]), repo_path, tree=ast_node)
                        add_imports(node_id, imports, graph_nodes, graph_edges, repo_path)
                    except Exception:
                        pass
                    for deco in ast_node.decorator_list:
                        if isinstance(deco, ast.Name):
                            invocations.append(deco.id)
                        elif isinstance(deco, ast.Attribute):
                            invocations.append(deco.attr)

                    def traverse_call(n):
                        for child in ast.iter_child_nodes(n):
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                continue
                            elif isinstance(child, ast.Call):
                                if isinstance(child.func, ast.Name):
                                    invocations.append(child.func.id)
                                elif isinstance(child.func, ast.Attribute):
                                    invocations.append(child.func.attr)
                            traverse_call(child)
                    traverse_call(ast_node)
                    break

        # add invokes edges
        for callee_name in set(invocations):
            cand = callee_name_dict.get(callee_name)
            if cand:
                if isinstance(cand, list):
                    for c in cand:
                        graph_edges.append({'src': node_id, 'dst': c, 'type': EDGE_TYPE_INVOKES})
                        invokes_edges_added += 1
                else:
                    graph_edges.append({'src': node_id, 'dst': cand, 'type': EDGE_TYPE_INVOKES})
                    invokes_edges_added += 1
            elif global_import:
                for global_candidate in global_name_map.get(callee_name, []):
                    graph_edges.append({'src': node_id, 'dst': global_candidate, 'type': EDGE_TYPE_INVOKES})
                    invokes_edges_added += 1

        for inh in set(inheritances):
            cand = callee_name_dict.get(inh)
            if cand:
                if isinstance(cand, list):
                    for c in cand:
                        graph_edges.append({'src': node_id, 'dst': c, 'type': EDGE_TYPE_INHERITS})
                        inherits_edges_added += 1
                else:
                    graph_edges.append({'src': node_id, 'dst': cand, 'type': EDGE_TYPE_INHERITS})
                    inherits_edges_added += 1
            elif global_import:
                for global_candidate in global_name_map.get(inh, []):
                    graph_edges.append({'src': node_id, 'dst': global_candidate, 'type': EDGE_TYPE_INHERITS})
                    inherits_edges_added += 1

    print(f"  Added {invokes_edges_added} invoke edges and {inherits_edges_added} inherit edges (analysis took {time.time()-t3:.1f}s)")

    # Rebuild adjacency maps to include new edges introduced by analysis
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for e in graph_edges:
        outgoing[e['src']].append(e)
        incoming[e['dst']].append(e)

    return graph_nodes, graph_edges, outgoing, incoming

# ---------------- derive call graph ----------------
def derive_call_graph(graph_nodes: Dict, graph_edges: List[Dict]):
    method_nodes = [nid for nid, nd in graph_nodes.items() if nd['type'] in (NODE_TYPE_CLASS, NODE_TYPE_FUNCTION)]
    adj = {m: [] for m in method_nodes}
    for e in graph_edges:
        if e['type'] == EDGE_TYPE_INVOKES:
            src = e['src']; dst = e['dst']
            if src in adj:
                adj[src].append(dst)
    return method_nodes, adj

# ---------------- orchestration ----------------
def build_repo_graphs(input_json_path: str,
                      code_graph_base_dir: str,
                      repo_local_path: str,
                      worktrees_cache_parent: str,
                      use_worktree: bool = True,
                      clean_worktree_after: bool = True):
    with open(input_json_path, "r", encoding="utf-8") as f:
        instances = json.load(f)

    if not instances:
        raise RuntimeError("No instances in " + input_json_path)

    repo_str = instances[0].get("repo", "")
    repo_slug = repo_str.replace("/", "__")
    out_repo_dir = os.path.join(code_graph_base_dir, repo_slug)
    os.makedirs(out_repo_dir, exist_ok=True)

    # repo-level index file (registry)
    index_path = os.path.join(code_graph_base_dir, f"{repo_slug}.json")
    # prepare empty index if not exists
    if not os.path.exists(index_path):
        save_json_atomic([], index_path)

    # group by commit
    commit_to_instances = defaultdict(list)
    for inst in instances:
        commit_to_instances[inst['base_commit']].append(inst)

    # optional debug limit
    commits_iter = list(commit_to_instances.items())
    if DEBUG_MAX_COMMITS is not None:
        commits_iter = commits_iter[:DEBUG_MAX_COMMITS]

    # cache for commits built in this run
    built_commits = {}

    for commit, inst_list in commits_iter:
        print(f"\n=== Processing commit {commit} (#instances={len(inst_list)}) ===")
        commit_dir = os.path.join(out_repo_dir, commit)
        codegraph_path = os.path.join(commit_dir, "codegraph.json")
        callgraph_path = os.path.join(commit_dir, "callgraph.json")

        # quick skip if output exists
        if os.path.exists(codegraph_path) and os.path.exists(callgraph_path):
            print("  Found existing artifacts, skipping build.")
            # update index with references for these instance_ids (if not present)
            with open(index_path, "r", encoding="utf-8") as fx:
                registry = json.load(fx)
            for inst in inst_list:
                # if entry exists, skip
                found = False
                for e in registry:
                    if e.get("instance_id") == inst["instance_id"]:
                        found = True; break
                if not found:
                    registry.append({
                        "repo": repo_str,
                        "instance_id": inst["instance_id"],
                        "base_commit": commit,
                        "code_graph_path": codegraph_path,
                        "call_graph_path": callgraph_path
                    })
            save_json_atomic(registry, index_path)
            continue

        # prepare checkout/workdir
        if use_worktree:
            worktree_cache_dir = os.path.join(worktrees_cache_parent, repo_slug)
            os.makedirs(worktree_cache_dir, exist_ok=True)
            try:
                workdir = ensure_worktree_for_commit(repo_local_path, commit, worktree_cache_dir)
            except Exception as e:
                print("  Error creating worktree:", e)
                # store failures in index as nulls
                with open(index_path, "r", encoding="utf-8") as fx:
                    registry = json.load(fx)
                for inst in inst_list:
                    registry.append({
                        "repo": repo_str,
                        "instance_id": inst["instance_id"],
                        "base_commit": commit,
                        "code_graph_path": None,
                        "call_graph_path": None,
                        "error": str(e)
                    })
                save_json_atomic(registry, index_path)
                continue
        else:
            # in-place checkout
            try:
                checkout_to_commit(commit, repo_local_path, git_branch)
                workdir = repo_local_path
            except Exception as e:
                print("  Error checkout in-place:", e)
                with open(index_path, "r", encoding="utf-8") as fx:
                    registry = json.load(fx)
                for inst in inst_list:
                    registry.append({
                        "repo": repo_str,
                        "instance_id": inst["instance_id"],
                        "base_commit": commit,
                        "code_graph_path": None,
                        "call_graph_path": None,
                        "error": str(e)
                    })
                save_json_atomic(registry, index_path)
                continue

        # build graph (only once per unique commit)
        t_start = time.time()
        try:
            graph_nodes, graph_edges, outgoing, incoming = build_code_graph(workdir, fuzzy_search=True, global_import=True)
        except Exception as e:
            print("  Error building graph:", e)
            # write failures
            with open(index_path, "r", encoding="utf-8") as fx:
                registry = json.load(fx)
            for inst in inst_list:
                registry.append({
                    "repo": repo_str,
                    "instance_id": inst["instance_id"],
                    "base_commit": commit,
                    "code_graph_path": None,
                    "call_graph_path": None,
                    "error": str(e)
                })
            save_json_atomic(registry, index_path)
            # cleanup worktree
            if use_worktree and clean_worktree_after:
                try:
                    run_cmd(["git", "-C", repo_local_path, "worktree", "remove", workdir, "--force"], cwd=None, timeout=60)
                except Exception:
                    pass
                try:
                    shutil.rmtree(workdir, ignore_errors=True)
                except Exception:
                    pass
            continue

        build_time = time.time() - t_start
        print(f"  Built code graph: nodes={len(graph_nodes)} edges={len(graph_edges)} time={build_time:.1f}s")

        # derive call graph
        try:
            methods, adjacency = derive_call_graph(graph_nodes, graph_edges)
        except Exception as e:
            print("  Error deriving call graph:", e)
            methods, adjacency = [], {}

        # prepare per-commit output directory and write artifacts
        os.makedirs(commit_dir, exist_ok=True)
        # nodes list: convert dict->list for serialization, but we will include id inside node dicts
        nodes_list = list(graph_nodes.values())
        edges_list = graph_edges

        save_json_atomic({"nodes": nodes_list, "edges": edges_list}, codegraph_path)
        save_json_atomic({"methods": methods, "adjacency": adjacency}, callgraph_path)
        print("  Wrote codegraph:", codegraph_path)
        print("  Wrote callgraph:", callgraph_path)

        # update repo-level index (append entries for each instance_id)
        with open(index_path, "r", encoding="utf-8") as fx:
            registry = json.load(fx)
        for inst in inst_list:
            registry.append({
                "repo": repo_str,
                "instance_id": inst["instance_id"],
                "base_commit": commit,
                "code_graph_path": codegraph_path,
                "call_graph_path": callgraph_path
            })
        save_json_atomic(registry, index_path)
        print("  Updated registry:", index_path)

        # optionally clean worktree
        if use_worktree and clean_worktree_after:
            try:
                run_cmd(["git", "-C", repo_local_path, "worktree", "remove", workdir, "--force"], cwd=None, timeout=60)
            except Exception:
                pass
            if os.path.exists(workdir):
                try:
                    shutil.rmtree(workdir, ignore_errors=True)
                except Exception:
                    pass

    print("\nAll done. Repo-level index at:", index_path)

if __name__ == "__main__":
    build_repo_graphs(input_file, code_graph_base_dir, repo_local_path, worktrees_cache_parent, USE_WORKTREE, CLEAN_WORKTREE_AFTER)
