#!/usr/bin/env python3
"""
agent_full_repo_controller.py

Single-file controller that:
 - loads SWE-Bench per-repo instances JSON (data/by_repo/<repo>.json)
 - for each instance maps to prebuilt codegraph/callgraph artifacts
 - builds an embedding index of non-test production classes (cache per repo+commit)
 - exposes tools to a LangChain agent: classify_report (LLM), semantic_rank (embedding),
   get_subgraph, get_code, get_file_context, search_codebase
 - chooses starting nodes: exact symbol -> stacktrace -> test->prod mapping -> semantic_rank top-K
 - runs the agent and saves per-instance results into a single repository-level JSON file
"""

import os
import sys
import json
import time
import re
import math
import shutil
import subprocess
import tempfile
import difflib
from collections import defaultdict, deque
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

# token / embedding config
import tiktoken

# LangChain & OpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# FAISS optional
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    import math

# --- LangGraph (tool-calling) support ---
try:
    # langgraph==0.2.x
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# ---------------- CONFIGURE (edit these) ----------------
REPO_INSTANCES_JSON = "data/by_repo/astropy__astropy.json"
# OUT_SUMMARY_FILE = "swe_results/multiagent_enhanced/matplotlib__matplotlib.json"
# OUT_SINGLE_ENHANCED_FILE = "swe_results/enhanced/matplotlib__matplotlib.json"
# REPO_INSTANCES_JSON = "test.json"
OUT_SUMMARY_FILE = "test_output.json"
OUT_SINGLE_ENHANCED_FILE = "test1.json"

REPO_CODEGRAPH_INDEX = "data/code_graph/astropy__astropy.json"
REPO_LOCAL_PATH = "/Users/fahim/Desktop/PhD/swe_bench/astropy"

# Optional filter: if non-empty, only run these instance_ids from REPO_INSTANCES_JSON.
# Leave empty [] to process all instances (default behavior).
# INSTANCE_ID_FILTER: List[str] = ["pylint-dev__pylint-7080", "pylint-dev__pylint-7114"]
INSTANCE_ID_FILTER: List[str] = ["astropy__astropy-14182"]


# OPENAI_MODEL = "gpt-4o-mini"           # LLM for classification + final report
OPENAI_MODEL = "gpt-5-mini-2025-08-07"
EMBED_MODEL = "text-embedding-3-large" # Embedding model
# EMBED_MODEL = "text-embedding-3-small" # Embedding model
LLM_TEMPERATURE = 1 # 1 for gpt-5-mini, 0.0 for gpt-4o-mini
# LLM is initialized after model-specific helper functions are defined.
LLM = None

# Toggle this at module level (default True -> use LLM)
USE_LLM_CLASSIFIER = True    # True to use LLM, False to use regex

# For Semantic Ranking, toggle True/False
USE_BM25_RANKING = False    # True to use BM25, False to use embeddings+FAISS

EMBED_TOP_K = 5          # top-K to return to agent (5 or 10 as you prefer)
EMBED_MAX_TOKENS = 120000   # safe default: tune up to model limit
EMBED_BATCH_CHAR_TRUNC = 30000  # fallback truncate per text if extremely long

EXCLUDE_TESTS = True     # exclude tests from semantic index
EXCLUDE_DIRS = ["tests", "test", "third_party", "vendor", "dist", "build", "__pycache__"]

AGENT_MAX_ITERATIONS = 20
SUBGRAPH_HOPS = 2
SUBGRAPH_MAX_NODES = 40
TOKEN_EST_PER_LINE = 3
EMBEDDING_CACHE_DIR = "embeddings_cache"

# -------------------------------------------------------

# ---------------- LLM helpers (model-specific quirks) ----------------

def _is_gpt5_model(model_name: str) -> bool:
    m = (model_name or "").lower().strip()
    return m.startswith("gpt-5")


def _normalize_temperature(model_name: str, temperature: float | int | None):
    """GPT-5-mini: enforce temperature==1. Other models: pass through."""
    if _is_gpt5_model(model_name):
        return 1
    # default fallback for None
    return 0 if temperature is None else temperature


def _wrap_drop_stop_for_gpt5(llm):
    """Return an LLM wrapper so GPT-5 calls never receive `stop`.

    LangChain v0.2+ validates that `llm` passed into LLMChain is a `Runnable`.
    A plain proxy object fails that validation. This wrapper subclasses Runnable
    and delegates to the underlying LLM while stripping stop-related kwargs.
    """

    from langchain_core.runnables import Runnable

    class _DropStopRunnable(Runnable):
        def __init__(self, inner):
            super().__init__()
            self._llm = inner

        def __getattr__(self, name):
            return getattr(self._llm, name)

        @staticmethod
        def _strip_stop(kwargs):
            if kwargs:
                kwargs.pop("stop", None)
                kwargs.pop("stop_sequences", None)
            return kwargs

        # Runnable interface
        def invoke(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return self._llm.invoke(input, config=config, **kwargs)

        async def ainvoke(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return await self._llm.ainvoke(input, config=config, **kwargs)

        def batch(self, inputs, config=None, **kwargs):
            self._strip_stop(kwargs)
            return self._llm.batch(inputs, config=config, **kwargs)

        async def abatch(self, inputs, config=None, **kwargs):
            self._strip_stop(kwargs)
            return await self._llm.abatch(inputs, config=config, **kwargs)

        def stream(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return self._llm.stream(input, config=config, **kwargs)

        async def astream(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return await self._llm.astream(input, config=config, **kwargs)

        # Back-compat convenience
        def generate(self, messages, stop=None, **kwargs):
            stop = None
            self._strip_stop(kwargs)
            return self._llm.generate(messages, stop=stop, **kwargs)

        async def agenerate(self, messages, stop=None, **kwargs):
            stop = None
            self._strip_stop(kwargs)
            return await self._llm.agenerate(messages, stop=stop, **kwargs)

        def __call__(self, *args, **kwargs):
            self._strip_stop(kwargs)
            return self._llm(*args, **kwargs)

    # Avoid double wrapping
    if isinstance(llm, _DropStopRunnable):
        return llm
    return _DropStopRunnable(llm)


def make_chat_llm(model_name: str, temperature: float | int | None = None):
    """Create a ChatOpenAI instance with model-specific compatibility handling.

    - GPT-5*: force temperature=1 and strip `stop` injected by agents.
    - Others (e.g., gpt-4o-mini): keep default behavior.
    """
    temp = _normalize_temperature(model_name, temperature)
    base = ChatOpenAI(model=model_name, temperature=temp)
    return _wrap_drop_stop_for_gpt5(base) if _is_gpt5_model(model_name) else base


# Initialize the shared LLM (used by classify_report / final report generation)
LLM = make_chat_llm(OPENAI_MODEL, LLM_TEMPERATURE)

# ---------- simple utilities ----------
def mkdirp(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def save_json_atomic(obj: Any, path: str):
    mkdirp(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def load_json_safe(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            if os.path.getsize(path) == 0:
                return []
            return json.load(f)
    except Exception:
        # move corrupt aside and return empty list
        try:
            shutil.move(path, path + ".corrupt")
        except Exception:
            pass
        return []

def run_cmd(cmd: List[str], cwd: Optional[str]=None, timeout: int=300) -> Tuple[int,str,str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"TIMEOUT after {timeout}s"
    except Exception as e:
        return 1, "", str(e)
    


# ------------------ BM25 helpers ------------------

def _simple_tokenize_for_bm25(s: str):
    """Simple tokenization used for BM25 (lowercased alpha-numeric + underscores)."""
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9_]+", s)]

def bm25_prepare_candidates(nodes_by_id: Dict[str, Dict[str,Any]], exclude_tests_flag: bool = True):
    """
    Build candidate node list + representative text per candidate for BM25 ranking.
    Returns (candidates, docs_texts, bm25_index)
    - candidates: list of node ids (strings)
    - docs_texts: list of strings (same order as candidates) used to rank
    - bm25_index: dict with precomputed stats used by bm25_rank_query()
    """
    # preference order: classes -> functions -> files (mirrors embedding index)
    candidates = []
    # 1) classes
    for nid, nd in nodes_by_id.items():
        if exclude_tests_flag and is_test_node(nid):
            continue
        if nd.get("type") == "class":
            candidates.append(nid)
    # 2) functions only if no classes
    if not candidates:
        for nid, nd in nodes_by_id.items():
            if exclude_tests_flag and is_test_node(nid):
                continue
            if nd.get("type") == "function":
                candidates.append(nid)
    # 3) files fallback
    if not candidates:
        for nid, nd in nodes_by_id.items():
            if exclude_tests_flag and is_test_node(nid):
                continue
            if nd.get("type") == "file":
                candidates.append(nid)

    # build representative text for each candidate (first N lines)
    docs_texts = []
    for nid in candidates:
        nd = nodes_by_id.get(nid, {})
        code = nd.get("code", "") or ""
        # choose a snippet: first ~200 lines (tuneable)
        snippet = "\n".join(code.splitlines()[:200]).strip()
        # fallback to node id if snippet empty
        docs_texts.append(snippet if snippet else nid)

    # Tokenize documents
    docs_tokens = [_simple_tokenize_for_bm25(t) for t in docs_texts]
    N = len(docs_tokens)
    # doc term frequencies and doc lengths
    doc_counters = [Counter(d) for d in docs_tokens]
    doc_lens = [sum(c.values()) for c in doc_counters]
    avgdl = (sum(doc_lens) / N) if N > 0 else 0.0

    # document frequencies
    df = Counter()
    for tokset in docs_tokens:
        for tok in set(tokset):
            df[tok] += 1

    # build index dict (store minimal necessary items)
    bm25_index = {
        "N": N,
        "df": df,
        "doc_counters": doc_counters,
        "doc_lens": doc_lens,
        "avgdl": avgdl
    }
    return candidates, docs_texts, bm25_index

def bm25_rank_query(query: str, candidates: List[str], docs_texts: List[str], bm25_index: Dict[str,Any], top_k: int = 5):
    """
    Rank candidates for `query` using Okapi BM25 (k1=1.5, b=0.75).
    Returns list of {"node_id": ..., "score": ...} ordered by score desc.
    """
    if not candidates or not bm25_index or not docs_texts:
        return []

    N = bm25_index["N"]
    df = bm25_index["df"]
    doc_counters = bm25_index["doc_counters"]
    doc_lens = bm25_index["doc_lens"]
    avgdl = bm25_index["avgdl"]

    # BM25 params
    k1 = 1.5
    b = 0.75

    # tokenize query
    q_tokens = _simple_tokenize_for_bm25(query or "")

    # idf function with small smoothing
    def _idf(tok):
        # standard BM25-ish idf smoothing
        docfreq = df.get(tok, 0)
        return math.log(1 + (N - docfreq + 0.5) / (docfreq + 0.5))

    scores = []
    # precompute unique query tokens
    q_unique = set(q_tokens)
    for i, node_id in enumerate(candidates):
        score = 0.0
        dl = doc_lens[i] if i < len(doc_lens) else 0
        cnts = doc_counters[i] if i < len(doc_counters) else Counter()
        if dl == 0:
            scores.append((0.0, i))
            continue
        for qt in q_unique:
            f = cnts.get(qt, 0)
            if f == 0:
                continue
            term_idf = _idf(qt)
            denom = f + k1 * (1 - b + b * (dl / avgdl)) if avgdl > 0 else f + k1
            score += term_idf * (f * (k1 + 1)) / max(1e-9, denom)
        scores.append((score, i))

    # sort and return top_k
    scores.sort(reverse=True, key=lambda x: x[0])
    if top_k <= 0:
        top_k = len(scores)
    out = []
    for sc, idx in scores[:min(top_k, len(scores))]:
        out.append({"node_id": candidates[idx], "score": float(sc)})
    return out


# ---------- codegraph loader helpers ----------
def load_codegraph(path: str):
    """
    expects codegraph.json structure: {"nodes":[...], "edges":[...]}
    node must have 'id' and 'type' and optional 'code'
    """
    data = {}
    try:
        data = load_json_safe(path)
    except Exception:
        return {}, []
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    nodes_by_id = {n["id"]: n for n in nodes}
    return nodes_by_id, edges

def build_edge_adjacency(edges: List[Dict[str,Any]]):
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    for e in edges:
        outgoing[e["src"]].append(e)
        incoming[e["dst"]].append(e)
    return outgoing, incoming

# ---------- exclude / test detection functions ----------
def path_contains_excluded_dir(path: str) -> bool:
    low = path.replace("\\","/").lower()
    for d in EXCLUDE_DIRS:
        if f"/{d}/" in low or low.startswith(f"{d}/") or low.endswith(f"/{d}"):
            return True
    return False

def is_test_node(nid: str) -> bool:
    """
    Heuristics:
     - file path contains '/tests/' or '/test/'
     - filename starts with 'test' (test_foo.py)
     - node_id includes 'tests' segments
    """
    low = nid.lower()
    if "/tests/" in low or "/test/" in low or "\\tests\\" in low or "\\test\\" in low:
        return True
    file_part = nid.split(":")[0]
    base = os.path.basename(file_part).lower()
    if base.startswith("test") or base.startswith("tests"):
        return True
    return False

def test_name_to_prod_candidates(test_name: str, simple_map: Dict[str,List[str]]):
    """
    Given e.g., 'test_implement_feature' or 'testImplementFeature', produce possible production names:
    - strip leading 'test_' or 'test' prefix
    - try to map snake->camel variants, etc.
    Return matching node ids from simple_map
    """
    candidates = []
    t = os.path.basename(test_name)
    t_low = t.lower()
    # common patterns
    if t_low.startswith("test_"):
        core = t[5:]
    elif t_low.startswith("test"):
        core = t[4:]
    else:
        core = t
    # remove common suffixes
    core = re.sub(r"(_test|_case|_spec)$", "", core, flags=re.I)
    # try exact name in simple_map (class/method names)
    if core in simple_map:
        candidates.extend(simple_map[core])
    # try camel variant (TestMy -> my)
    core_lower = core.lower()
    if core_lower in simple_map:
        candidates.extend(simple_map[core_lower])
    # also search for filenames that contain core
    for k in simple_map:
        if core_lower in k.lower():
            candidates.extend(simple_map[k])
    # dedupe
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


# ---------------- token helpers & batched embedding ----------------
def count_tokens(text: str, model_name: str = "gpt-5-mini-2025-08-07") -> int:
    """Return token count for `text` with tiktoken for given model."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# This model's maximum context length is 128000 tokens
def split_into_chunks(text, max_tokens=250000, model="gpt-5-mini-2025-08-07"):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def split_texts_into_token_batches(texts: List[str], max_tokens: int, model_name: str):
    """
    Group the list of `texts` into batches so that sum(tokens) in a batch <= max_tokens.
    Returns list of batches, each batch is list of texts.
    """
    batches = []
    cur_batch = []
    cur_tokens = 0
    for t in texts:
        # safety truncate extremely long text to avoid insane single item sizes
        if len(t) > EMBED_BATCH_CHAR_TRUNC * 2:
            t = t[:EMBED_BATCH_CHAR_TRUNC] + "\n\n# ...(truncated)...\n\n" + t[-EMBED_BATCH_CHAR_TRUNC:]
        tok = count_tokens(t, model_name)
        # if single text exceeds max_tokens, we must still embed it alone after truncation
        if tok > max_tokens:
            # aggressively truncate by characters, then recompute tokens
            t_short = t[:EMBED_BATCH_CHAR_TRUNC]
            tok = count_tokens(t_short, model_name)
            t = t_short
        # start new batch if adding would overflow
        if cur_tokens + tok > max_tokens and cur_batch:
            batches.append(cur_batch)
            cur_batch = [t]
            cur_tokens = tok
        else:
            cur_batch.append(t)
            cur_tokens += tok
    if cur_batch:
        batches.append(cur_batch)
    return batches

def embed_documents_token_safe(emb_obj, texts: List[str], model_name: str = EMBED_MODEL, max_tokens_per_request: int = EMBED_MAX_TOKENS):
    """
    Token-aware batched embedding. Returns list of vectors aligned with `texts`.
    """
    # early exit
    if not texts:
        return []
    # build batches by token budget
    batches = split_texts_into_token_batches(texts, max_tokens_per_request, model_name)
    all_vecs = []
    for batch in batches:
        vecs = emb_obj.embed_documents(batch)
        all_vecs.extend(vecs)
    return all_vecs


# ---------- EMBEDDING INDEX (classes preferred) ----------
def build_embedding_index(nodes_by_id: Dict[str,Dict[str,Any]], repo_slug: str, commit_hash: str, exclude_tests_flag: bool = True):
    """
    Build and cache embeddings for candidate nodes (prefer classes).
    Returns (candidates_list, vectors_list, faiss_index_or_None, emb_obj)
    """
    mkdirp(EMBEDDING_CACHE_DIR)
    cache_file = os.path.join(EMBEDDING_CACHE_DIR, f"{repo_slug}__{commit_hash}.json")
    # try load cache
    if os.path.exists(cache_file):
        try:
            j = load_json_safe(cache_file)
            candidates = j["candidates"]
            vectors = j["vectors"]
            # vectors stored as lists
            emb_obj = OpenAIEmbeddings(model=EMBED_MODEL)
            faiss_index = None
            if FAISS_AVAILABLE and len(vectors) > 0:
                dim = len(vectors[0])
                vs = np.array(vectors, dtype='float32')
                # normalize
                norms = np.linalg.norm(vs, axis=1, keepdims=True)
                norms[norms==0] = 1.0
                vs_norm = vs / norms
                idx = faiss.IndexFlatIP(dim)
                idx.add(vs_norm)
                faiss_index = (idx, vs_norm)
            return candidates, vectors, faiss_index, emb_obj
        except Exception:
            pass

    # Build fresh candidate list (prefer classes)
    candidates = []
    for nid, nd in nodes_by_id.items():
        if exclude_tests_flag and is_test_node(nid):
            continue
        if nd.get("type") == "class":
            candidates.append(nid)
    if not candidates:
        for nid, nd in nodes_by_id.items():
            if exclude_tests_flag and is_test_node(nid):
                continue
            if nd.get("type") == "function":
                candidates.append(nid)
    if not candidates:
        for nid, nd in nodes_by_id.items():
            if exclude_tests_flag and is_test_node(nid):
                continue
            if nd.get("type") == "file":
                candidates.append(nid)

    # build texts to embed
    texts = []
    for nid in candidates:
        nd = nodes_by_id[nid]
        code = nd.get("code") or ""
        snippet = "\n".join(code.splitlines()[:80])
        texts.append(f"{nid}\n{snippet}")

    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    # vectors = emb.embed_documents(texts) if len(texts)>0 else []
    vectors = embed_documents_token_safe(emb, texts, model_name=EMBED_MODEL, max_tokens_per_request=EMBED_MAX_TOKENS)
    faiss_index = None
    if FAISS_AVAILABLE and len(vectors)>0:
        dim = len(vectors[0])
        vs = np.array(vectors, dtype='float32')
        norms = np.linalg.norm(vs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        vs_norm = vs / norms
        idx = faiss.IndexFlatIP(dim)
        idx.add(vs_norm)
        faiss_index = (idx, vs_norm)

    # write cache (vectors as lists)
    try:
        save_json_atomic({"candidates": candidates, "vectors": vectors}, cache_file)
    except Exception:
        pass

    return candidates, vectors, faiss_index, emb

def semantic_rank_by_embedding(query: str, candidates: List[str], vectors: List[List[float]], faiss_index, emb_obj, top_k: int = 5):
    """
    Rank candidates by cosine similarity between query embedding and candidate vectors.
    Returns list of {node_id, score}
    """
    qv = emb_obj.embed_query(query)
    if FAISS_AVAILABLE and faiss_index is not None and len(candidates) == len(faiss_index[1]):
        idx, vs = faiss_index
        import numpy as np
        qv = np.array(qv, dtype='float32')
        qv_norm = qv / max(1e-12, np.linalg.norm(qv))
        D, I = idx.search(np.array([qv_norm]), min(len(vs), top_k))
        results = []
        for dist, i in zip(D[0], I[0]):
            results.append({"node_id": candidates[int(i)], "score": float(dist)})
        return results
    else:
        # brute-force
        results = []
        import math
        # normalize qv
        try:
            import numpy as np
            qv_np = np.array(qv, dtype='float32')
            qv_norm = qv_np / max(1e-12, np.linalg.norm(qv_np))
            for nid, v in zip(candidates, vectors):
                vnp = np.array(v, dtype='float32')
                vnorm = vnp / max(1e-12, np.linalg.norm(vnp))
                sc = float(np.dot(qv_norm, vnorm))
                results.append((sc, nid))
            results.sort(reverse=True)
            out = [{"node_id": nid, "score": float(sc)} for sc, nid in results[:top_k]]
            return out
        except Exception:
            # fallback rough scoring by token overlap
            qtoks = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
            scores = []
            for nid in candidates:
                s = len(qtoks & set(re.findall(r"[A-Za-z0-9_]+", nid.lower())))
                if s > 0:
                    scores.append((s, nid))
            scores.sort(reverse=True)
            return [{"node_id": n, "score": float(s)} for s,n in scores[:top_k]]

# ---------- Tools (classify_report via LLM, semantic_rank, get_subgraph, get_code, get_file_context, search_codebase) ----------
def tool_classify_report(input_str: str) -> str:
    """
    LLM-driven classifier with optional regex fallback controlled by USE_LLM_CLASSIFIER.

    Input:
      - JSON string or plain text (same behavior as before).
    Output:
      - If explicit entities found -> returns a readable starting-point string (same as earlier).
      - If no explicit entities -> triggers semantic fallback and returns a fallback starting-point string.
    """
    global nodes_index_global, nodes_by_id_global, current_reg_entry_global, classification_stats_global

    # 1) parse input
    repo_slug = None
    commit = None
    fail_to_pass = ""
    problem = ""
    # try:
    #     parsed = json.loads(input_str)
    #     if isinstance(parsed, dict):
    #         print("--------- tool_classify_report (dictionary) ---------")
    #         problem = parsed.get("problem","") or ""
    #         repo_slug = parsed.get("repo_slug") or parsed.get("repo") or parsed.get("repo_slug")
    #         commit = parsed.get("commit")
    #         fail_to_pass = parsed.get("fail_to_pass", parsed.get("FAIL_TO_PASS",""))
    # except Exception:
    #     print("--------- tool_classify_report (not dictionary) ---------")
    #     problem = input_str or ""

    # 2) fallback from current_reg_entry_global
    if (not repo_slug or not commit) and current_reg_entry_global:
        repo_slug = repo_slug or current_reg_entry_global.get("repo","").replace("/","__")
        commit = commit or current_reg_entry_global.get("base_commit")
        # print("--------- tool_classify_report (fallback) ---------")
        if not fail_to_pass:
            # print("--------- tool_classify_report (fallback fail to pass) ---------")
            # some registry entries keep FAIL_TO_PASS as string; keep as-is
            fail_to_pass = current_reg_entry_global.get("FAIL_TO_PASS", "") or current_reg_entry_global.get("fail_to_pass","")

    # ensure problem present
    if not problem and current_reg_entry_global:
        problem = current_reg_entry_global.get("problem_statement","") or current_reg_entry_global.get("problem","") or ""
        # print("--------- tool_classify_report (fallback problem statement) ---------")
        # print(current_reg_entry_global)
        # print("--------- tool_classify_report (fallback problem statement end) ---------")

    # 3) prompt LLM to extract explicit programming entities
    response_dict = None
    if USE_LLM_CLASSIFIER:
        template = '''
        Extract explicit programming signals from a bug report.

        You are a software engineer tasked with analyzing a single bug report (free text) and extracting any explicit programming signals it contains: method names, class names, stack traces, code snippets, or small patches. If none of these programming entities exist, explicitly mark that they are absent.

        # Guidelines
        - **Do not hallucinate.** Only extract entities that appear verbatim in the bug report text.
        - **Programming entities to detect:**
        - Method names (e.g., `parse()`, `com.example.Foo.bar`)
        - Class names (e.g., `Parser`, `com.example.Parser`)
        - Stack traces (lines beginning with an exception or `Caused by:` and subsequent `at ...` lines)
        - Code snippets or inline code blocks (any contiguous block of source-like text)
        - Small patches or diffs (lines starting with `+`/`-` or marked as patch)
        - If an entity is ambiguous (e.g., a word that could be a method but lacks parentheses or context), extract it only if it clearly appears as code or a programming identifier in the report.
        - Preserve the exact text (including whitespace/indentation) for any extracted stack trace or code snippet.
        - If the report contains multiple occurrences of the same programming entity, include all occurrences (deduplicating is optional but allowed if explicitly noted).

        # Steps
        1. Read the entire bug report input.
        2. Detect and extract stack traces:
        - A stack trace typically starts with an exception line (e.g., `Exception`, `Error`, `Caused by:`) followed by one or more lines starting with `at`.
        - Capture each complete stack trace block as it appears, preserving newlines and indentation.
        3. Detect and extract explicit method and class names:
        - Method names are most reliably extracted when followed by `()` or shown in code context. Also detect fully-qualified names if present.
        - Class names may appear as `ClassName` or fully-qualified `package.ClassName`.
        4. Detect code snippets and small patches:
        - Any indented or fenced block of code, or inline code markers (backticks), should be captured verbatim.
        - Patches or diffs (lines beginning with `+` or `-` in context) should be captured.
        5. Build the JSON output with clear, typed fields listed below.
        6. If **no** programming entities are found (no methods, classes, code snippets, patches, or stack traces), set `absent_programming_entities` to `true` and leave arrays empty or null as specified.

        # Output Format
        Return a single **valid JSON object** with these fields (use `null` where noted and preserve original text for code-related fields):

        ```json
        {{
            'absent_programming_entities': boolean — `true` if no programming entities were detected, otherwise `false`,
            'methods': array of strings or `[]` — explicit method names found (include any surrounding qualified name if present). Example entries: `"parse(String)"`, `"rollSecret()"`, `"com.example.Parser.parse"`,
            'classes': array of strings or `[]` — explicit class names or fully-qualified class names found,
            'stack_traces': array of strings — each element is one full stack trace block exactly as it appears (preserve newlines and indentation). If none, return `[]`,
            'code_snippets': array of strings — each element is a verbatim code snippet or patch block from the report (preserve formatting). If none, return `[]`,
            'other_programming_mentions': array of strings — any other programming-related tokens (file names, variable names, error constants) that are explicitly present but not classified above. If none, return `[]`
        }}
        ```
        **Important formatting rules:**
        - Output must be valid JSON (no extra commentary or text).
        - Preserve original line breaks inside string values using `\n` or literal multiline JSON strings depending on your environment — ensure JSON remains valid.

    
        # Inputs:
        You are given the bug report below:
        {problem}
        '''
        
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=LLM, prompt=prompt)
        
        try:
            print("--------- tool_classify_report (start) ---------")
            response = chain.run({'problem': problem})
            print("response:", response)
            response_dict = json.loads(response.replace("```json\n", "").replace("\n```", ""))
            print("text:", response_dict)
            for k, v in response_dict.items():
                print(f'{k}: {v}')
            print("--------- tool_classify_report (end) ---------")
        except Exception as e:
            response_dict = {
                "absent_programming_entities": True,
                "methods": [],
                "classes": [],
                "stack_traces": [],
                "code_snippets": [],
                "other_programming_mentions": []
            }

    # If LLM not chosen or failed -> regex extractor
    if response_dict is None:
        # Conservative regex extraction
        response_dict = _regex_extract_programming_entities(problem)
        # put empty for other_programming_mentions if not found by regex
        if "other_programming_mentions" not in response_dict:
            response_dict["other_programming_mentions"] = []
        print("--------- tool_classify_report (regex: start) ---------")
        print("regex dict:", response_dict)
        for k, v in response_dict.items():
            print(f'{k}: {v}')
        print("--------- tool_classify_report (regex: end) ---------")

    # --- Determine absent_programming_entities more reliably ---
    # Even if LLM returns a boolean, recompute it to be safe
    absent_programming_entities = not (
        response_dict.get("methods") or
        response_dict.get("classes") or
        response_dict.get("stack_traces") or
        response_dict.get("code_snippets") or
        response_dict.get("other_programming_mentions")
    )
    response_dict["absent_programming_entities"] = absent_programming_entities

    # --- Update global classification stats ---
    inst_id = None
    if current_reg_entry_global:
        inst_id = current_reg_entry_global.get("instance_id") or current_reg_entry_global.get("instance")
    if inst_id:
        classification_stats_global.setdefault(inst_id, {})
        s = classification_stats_global[inst_id]
        s["has_method_or_class"] = bool(response_dict.get("methods") or response_dict.get("classes"))
        s["has_stack_trace"] = bool(response_dict.get("stack_traces"))
        s["has_patch_text"] = bool(current_reg_entry_global.get("patch")) if current_reg_entry_global else False
        s["absent_programming_entities"] = absent_programming_entities
        classification_stats_global[inst_id] = s

    # --- Branch 1: if programming entities are absent, do semantic ranking ---
    if absent_programming_entities:
        print("No explicit entities found -> performing semantic ranking.")
        # ensure repo_slug & commit available (try fallback)
        if not repo_slug or not commit:
            if current_reg_entry_global:
                repo_slug = repo_slug or current_reg_entry_global.get("repo","").replace("/","__")
                commit = commit or current_reg_entry_global.get("base_commit")
                if not fail_to_pass:
                    fail_to_pass = current_reg_entry_global.get("FAIL_TO_PASS","")

        if not repo_slug or not commit:
            response_dict["semantic_fallback"] = []
            return json.dumps(response_dict)
        
        # --- OPTION A: BM25 ranking (local, token-based) ---
        if USE_BM25_RANKING:
            candidates, docs_texts, bm25_index = bm25_prepare_candidates(nodes_by_id_global, exclude_tests_flag=EXCLUDE_TESTS)
            ranked = bm25_rank_query(problem + ("\nFailed tests: " + str(fail_to_pass) if fail_to_pass else ""), candidates, docs_texts, bm25_index, top_k=EMBED_TOP_K)
            top_ranked = [r["node_id"] for r in ranked]
            response_dict["semantic_fallback"] = top_ranked

        # --- OPTION B: embeddings + FAISS ---
        else:
            # Build/load nodes_index_global lazily
            if nodes_index_global is None:
                try:
                    nodes_index = build_embedding_index(nodes_by_id_global, repo_slug, commit, exclude_tests_flag=EXCLUDE_TESTS)
                    nodes_index_global = nodes_index
                except Exception as e:
                    response_dict["semantic_fallback_error"] = str(e)
                    response_dict["semantic_fallback"] = []
                    return json.dumps(response_dict)

            # Build query using problem + failing tests if present
            query = (problem or "")
            if fail_to_pass:
                query += "\nFailed tests: " + str(fail_to_pass)
            try:
                candidates, vectors, faiss_index, emb_obj = nodes_index_global
                ranked = semantic_rank_by_embedding(query, candidates, vectors, faiss_index, emb_obj, top_k=EMBED_TOP_K)
                top_ranked = [r["node_id"] for r in ranked]
                response_dict["semantic_fallback"] = top_ranked
            except Exception as e:
                response_dict["semantic_fallback_error"] = str(e)
                response_dict["semantic_fallback"] = []
        
        fallback_starting_point = f'''
        You can consider the following items to start your navigation.
        Analyze their names and read the initial bug report again to think about where to start.
        Then ask for anyone of them. I will provide the full implemention accordingly.

        Here are the probable buggy classes:
        {response_dict["semantic_fallback"]}


        This is the full original bug report for your reference:
        {problem}
        '''
        return fallback_starting_point

    # --- Branch 2: if entities are present, build a simple prompt for the agent ---
    probable_starting_point = (
        "You can consider the following items to start your navigation. "
        "They were mentioned in the bug report. Analyze their names and read the initial bug report again to think about where to start. "
        "Then ask for a method or class. I will provide the method body or class body accordingly.\n"
    )

    if response_dict.get("methods"):
        probable_starting_point += f"\nThese method(s) were mentioned:\n{response_dict['methods']}"
    if response_dict.get("classes"):
        probable_starting_point += f"\nThese class(es) were mentioned:\n{response_dict['classes']}"
    if response_dict.get("stack_traces"):
        probable_starting_point += f"\nStack trace(s):\n{response_dict['stack_traces']}"
    if response_dict.get("code_snippets"):
        probable_starting_point += f"\nCode snippet(s):\n{response_dict['code_snippets']}"
    if response_dict.get("other_programming_mentions"):
        probable_starting_point += f"\nOther programming mentions:\n{response_dict['other_programming_mentions']}"

    probable_starting_point += f"\n\nThis is the full original bug report for your reference:\n{problem}"

    return probable_starting_point

def _regex_extract_programming_entities(text: str):
    """
    Conservative regex-based extraction. Returns dict with keys:
      methods, classes, stack_traces, code_snippets, other_programming_mentions
    Note: this is intentionally conservative to avoid hallucination.
    """
    methods = []
    classes = []
    stack_traces = []
    code_snippets = []
    other = []

    if not text:
        return {"methods": [], "classes": [], "stack_traces": [], "code_snippets": [], "other_programming_mentions": []}

    # 1) fenced code blocks (```...```) and indented blocks
    # capture fenced first
    fenced = re.findall(r"```(?:[\w+-]*)\n(.*?)```", text, re.S)
    for block in fenced:
        snippet = block.strip()
        if snippet:
            code_snippets.append(snippet)

    # capture contiguous indented blocks (>=2 lines starting with 4+ spaces or tab)
    indented_blocks = re.findall(r"(?:\n(?: {4}|\t).+(?:\n(?: {4}|\t).+)+)", text)
    for blk in indented_blocks:
        candidate = "\n".join([line[4:] if line.startswith("    ") else line for line in blk.splitlines()])
        code_snippets.append(candidate.strip())

    # 2) stack traces: simple heuristic - lines with Exception/Error or "Traceback" and following 'at ' or 'File '
    # capture blocks that start with 'Traceback' or a line containing 'Exception' or 'Error' and include following '  File'/'at ' lines.
    stack_matches = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if re.search(r"\bTraceback\b", ln) or re.search(r"\bException\b|\bError\b", ln):
            # collect until blank line or no more '  File'/'at' pattern lines (conservative)
            j = i
            block = []
            while j < len(lines) and (lines[j].strip() != "" or len(block) == 0):
                block.append(lines[j])
                j += 1
            st = "\n".join(block).strip()
            if len(st.splitlines()) >= 1:
                stack_traces.append(st)
            i = j
            continue
        i += 1

    # 3) explicit method names heuristics: tokens with parentheses or qualified names with dots and parentheses
    # e.g., separability_matrix(), foo.bar(), Class.method()
    method_pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*\s*\([^)]*\))")
    for m in method_pattern.findall(text):
        m_clean = re.sub(r"\s+", "", m)  # normalize whitespace inside parentheses
        if m_clean not in methods:
            methods.append(m_clean)

    # 4) class names heuristics: CamelCase-like tokens (conservative) and fully-qualified names without parentheses
    class_pattern = re.compile(r"\b([A-Z][A-Za-z0-9_]+(?:\.[A-Z][A-Za-z0-9_]+)*)\b")
    for c in class_pattern.findall(text):
        # filter common words that are not classes (very conservative)
        if len(c) > 2 and not re.match(r"^(Exception|Error|Traceback|File)$", c):
            if c not in classes:
                classes.append(c)

    # 5) other programming mentions: filenames or .py occurrences
    filename_matches = re.findall(r"([A-Za-z0-9_\-/\\]+\.py)", text)
    for fn in filename_matches:
        if fn not in other:
            other.append(fn)

    # Deduplicate and filter code_snippets by length
    code_snippets = [cs for i, cs in enumerate(code_snippets) if cs and cs not in code_snippets[:i]]
    methods = list(dict.fromkeys(methods))
    classes = list(dict.fromkeys(classes))
    stack_traces = list(dict.fromkeys(stack_traces))
    other = list(dict.fromkeys(other))

    return {
        "methods": methods,
        "classes": classes,
        "stack_traces": stack_traces,
        "code_snippets": code_snippets,
        "other_programming_mentions": other
    }


def tool_semantic_rank(input_str: str) -> str:
    """
    Input JSON: {"query": "...", "top_k": int (optional), "exclude_tests": bool (optional)}
    Output JSON: {"ranked":[{"node_id":..., "score":...},...], "error": optional}
    This tool will build the embeddings lazily if needed (uses current_reg_entry_global to obtain repo/commit).
    """
    global nodes_index_global, current_reg_entry_global

    try:
        args = json.loads(input_str)
    except Exception:
        return json.dumps({"ranked": [], "error": "invalid_input_json"})

    query = args.get("query", "")
    top_k = int(args.get("top_k", EMBED_TOP_K))
    excl = bool(args.get("exclude_tests", EXCLUDE_TESTS))

    # ensure nodes_index_global is prepared (lazy)
    if nodes_index_global is None:
        # try to get repo_slug & commit from args or current_reg_entry_global
        repo_slug = args.get("repo_slug")
        commit = args.get("commit")
        if current_reg_entry_global and (not repo_slug or not commit):
            repo_slug = repo_slug or current_reg_entry_global.get("repo","").replace("/","__")
            commit = commit or current_reg_entry_global.get("base_commit")
        if not repo_slug or not commit:
            return json.dumps({"ranked": [], "error": "missing_repo_or_commit_for_embeddings"})
        try:
            nodes_index_global = build_embedding_index(nodes_by_id_global, repo_slug, commit, exclude_tests_flag=EXCLUDE_TESTS)
        except Exception as e:
            return json.dumps({"ranked": [], "error": "embedding_build_failed: " + str(e)})

    # unpack
    candidates, vectors, faiss_index, emb_obj = nodes_index_global

    # optionally filter out test candidates
    if excl:
        filtered_candidates = []
        filtered_vectors = []
        for nid, v in zip(candidates, vectors):
            if not is_test_node(nid):
                filtered_candidates.append(nid)
                filtered_vectors.append(v)
    else:
        filtered_candidates = candidates
        filtered_vectors = vectors

    # run ranking (semantic_rank_by_embedding already supports FAISS fallback)
    try:
        ranked = semantic_rank_by_embedding(query, filtered_candidates, filtered_vectors, faiss_index, emb_obj, top_k=top_k)
    except Exception as e:
        return json.dumps({"ranked": [], "error": str(e)})

    return json.dumps({"ranked": ranked})

# Subgraph extraction tool (truncated)
# def get_subgraph_internal(start_nodes: List[str], hops:int=SUBGRAPH_HOPS, max_nodes:int=SUBGRAPH_MAX_NODES, token_budget:int=12000):
#     included = set()
#     result_nodes = {}
#     result_edges = []
#     q = deque()
#     for s in start_nodes:
#         if s in nodes_by_id_global and s not in included:
#             included.add(s)
#             q.append((s,0))
#             result_nodes[s] = nodes_by_id_global[s]
#     while q and len(result_nodes) < max_nodes:
#         nid, level = q.popleft()
#         if level >= hops:
#             continue
#         for e in outgoing_global.get(nid, []):
#             dst = e["dst"]
#             result_edges.append(e)
#             if dst not in included:
#                 included.add(dst)
#                 if dst in nodes_by_id_global:
#                     result_nodes[dst] = nodes_by_id_global[dst]
#                     q.append((dst, level+1))
#         for e in incoming_global.get(nid, []):
#             src = e["src"]
#             result_edges.append(e)
#             if src not in included:
#                 included.add(src)
#                 if src in nodes_by_id_global:
#                     result_nodes[src] = nodes_by_id_global[src]
#                     q.append((src, level+1))
#         # token budget guard
#         approx_tokens = 0
#         for n in result_nodes.values():
#             code = n.get("code","") or ""
#             approx_tokens += max(1, len(code.splitlines()))*TOKEN_EST_PER_LINE
#         if approx_tokens > token_budget:
#             break
#     # label map & textual formatting
#     label_map = {}
#     parts = []
#     i = 0
#     for nid, nd in result_nodes.items():
#         label = f"N{i}"; i += 1
#         label_map[label] = nid
#         parts.append(f"{label} | {nid} | {nd.get('type')}")
#         code = nd.get("code","") or ""
#         snippet = "\n".join(code.splitlines()[:40])
#         if snippet:
#             parts.append("```")
#             parts.append(snippet)
#             parts.append("```")
#     parts.append("EDGES:")
#     for e in result_edges:
#         s = label_map.get(e["src"], e["src"])
#         d = label_map.get(e["dst"], e["dst"])
#         parts.append(f"{s} -> {d} ({e.get('type')})")
#     subgraph_text = "\n".join(parts)
#     return {"nodes": result_nodes, "edges": result_edges, "label_map": label_map, "subgraph_text": subgraph_text}

def get_subgraph_internal(start_nodes: List[str], hops:int=SUBGRAPH_HOPS, max_nodes:int=SUBGRAPH_MAX_NODES, token_budget:int=12000):
    included = set()
    result_nodes = {}
    result_edges = []

    q = deque()
    for s in start_nodes:
        if s in nodes_by_id_global and s not in included:
            included.add(s)
            q.append((s, 0))
            result_nodes[s] = nodes_by_id_global[s]

    while q and len(result_nodes) < max_nodes:
        nid, level = q.popleft()
        if level >= hops:
            continue

        # --- LIMIT: only top 2 callees ---
        out_edges = outgoing_global.get(nid, [])[:2]
        for e in out_edges:
            dst = e["dst"]
            result_edges.append(e)
            if dst not in included and dst in nodes_by_id_global:
                included.add(dst)
                result_nodes[dst] = nodes_by_id_global[dst]
                q.append((dst, level + 1))

        # --- LIMIT: only top 2 callers ---
        in_edges = incoming_global.get(nid, [])[:2]
        for e in in_edges:
            src = e["src"]
            result_edges.append(e)
            if src not in included and src in nodes_by_id_global:
                included.add(src)
                result_nodes[src] = nodes_by_id_global[src]
                q.append((src, level + 1))

        # token budget guard (keep same as before)
        approx_tokens = 0
        for n in result_nodes.values():
            code = n.get("code", "") or ""
            approx_tokens += max(1, len(code.splitlines())) * TOKEN_EST_PER_LINE
        if approx_tokens > token_budget:
            break

        # Hard stop at 5 nodes (start + 2 callers + 2 callees)
        if len(result_nodes) >= 5:
            break

    # label map & textual formatting (unchanged)
    label_map = {}
    parts = []
    i = 0
    for nid, nd in result_nodes.items():
        label = f"N{i}"; i += 1
        label_map[label] = nid
        parts.append(f"{label} | {nid} | {nd.get('type')}")
        code = nd.get("code", "") or ""
        snippet = "\n".join(code.splitlines()[:40])
        if snippet:
            parts.append("```")
            parts.append(snippet)
            parts.append("```")

    parts.append("EDGES:")
    for e in result_edges:
        s = label_map.get(e["src"], e["src"])
        d = label_map.get(e["dst"], e["dst"])
        parts.append(f"{s} -> {d} ({e.get('type')})")

    subgraph_text = "\n".join(parts)
    return {"nodes": result_nodes, "edges": result_edges, "label_map": label_map, "subgraph_text": subgraph_text}




# def tool_get_subgraph(input_str: str) -> str:
#     print(" -------- tool_get_subgraph (start) -------- ")
#     print(input_str)
#     print(" -------- tool_get_subgraph (end) -------- ")
#     args = json.loads(input_str)
#     start_nodes = args.get("start_nodes", [])
#     hops = int(args.get("hops", SUBGRAPH_HOPS))
#     max_nodes = int(args.get("max_nodes", SUBGRAPH_MAX_NODES))
#     out = get_subgraph_internal(start_nodes, hops=hops, max_nodes=max_nodes)
#     # truncate codes in nodes for safety
#     nodes_small = {}
#     for nid, nd in out["nodes"].items():
#         ndc = dict(nd)
#         if "code" in ndc and ndc["code"]:
#             ndc["code"] = "\n".join(ndc["code"].splitlines()[:200]) + ("\n# ...(truncated)" if len(ndc["code"].splitlines())>200 else "")
#         nodes_small[nid] = ndc
#     return json.dumps({"nodes": nodes_small, "edges": out["edges"], "label_map": out["label_map"], "subgraph_text": out["subgraph_text"]})


def tool_get_subgraph(input_str: str) -> str:
    """
    Accepts either:
      - JSON: {"nodes": [...], "hops": N}
      - OR a raw string (e.g., "ClassName" or "file.py::test")
    """
    global nodes_by_id_global, edges_global

    # 1) Try JSON
    try:
        payload = json.loads(input_str)
        if isinstance(payload, dict):
            nodes = payload.get("nodes", [])
            hops = payload.get("hops", 2)
    except Exception:
        # 2) Non-JSON input → treat as a single starting node request
        raw = input_str.strip().replace('"',"").replace("'", "")
        
        # if it looks like test path e.g. "tests/...::test_foo"
        # strip test part & reduce to filename
        if "::" in raw:
            raw = raw.split("::")[0]

        raw = os.path.basename(raw)  # only take last filename part
        
        # now try to match nodes ending with that part
        nodes = []
        for nid in nodes_by_id_global:
            if nid.endswith(raw):
                nodes.append(nid)

        # fallback: zero matches = safe default
        if not nodes:
            return json.dumps({
                "error": "node_not_found",
                "input": input_str
            })

        hops = 2  # default hop

    # # Test exclusion: skip subgraphs for test-related nodes
    # joined = " ".join(nodes) if nodes else input_str
    # if "tests/" in joined or "/tests/" in joined:
    #     return json.dumps({
    #         "error": "test_node_requested",
    #         "message": f"Skipping subgraph for test-related code: {input_str}",
    #         "advice": "Please select a main source module instead of test files."
    #     })

    # Prefer non-test nodes when there are mixed matches.
    if nodes:
        non_test_nodes = [nid for nid in nodes if not is_test_node(nid)]
        if non_test_nodes:
            nodes = non_test_nodes

    # Test exclusion: skip subgraphs only if all matches are test-related
    if nodes and all(is_test_node(nid) for nid in nodes):
        return json.dumps({
            "error": "test_node_requested",
            "message": f"Skipping subgraph for test-related code: {input_str}",
            "advice": "Please select a main source module instead of test files."
        })

    # Continue with normal logic…
    subg = get_subgraph_internal(nodes, hops=hops, max_nodes=SUBGRAPH_MAX_NODES)
    return json.dumps(subg)


# def tool_get_code(node_id: str) -> str:
#     nid = node_id.strip().replace("'", "").replace('"', '').replace("`", "")
#     print(" -------- tool_get_code (start) -------- ")
#     print(f'#{node_id}#')
#     print(f'#{nid}#')
#     print(" -------- tool_get_code (start) -------- ")
#     if nid not in nodes_by_id_global:
#         return json.dumps({"error":"node_not_found", "node": nid})
#     nd = nodes_by_id_global[nid]
#     # log method cache
#     method_cache_global.add(nid)
#     return json.dumps({"id": nid, "code": nd.get("code",""), "start_line": nd.get("start_line"), "end_line": nd.get("end_line")})

# def tool_get_code(name: str) -> str:
#     """
#     Return full code for a node whose ID endswith the requested name.
#     Examples:
#       - "separability_matrix"
#       - "Linear1D"
#       - "Linear1D.evaluate"
#       - "evaluate"
#     """
#     print(" -------- tool_get_code (start) -------- ")
#     print(f'#{name}#')
#     name = name.strip().replace("'", "").replace('"', '').replace("`", "")
#     print(f'#{name}#')
#     print(" -------- tool_get_code (start) -------- ")
#     global nodes_by_id_global

#     if not name or not nodes_by_id_global:
#         return json.dumps({"error": "invalid_request", "node": name})

#     # 1. exact match
#     if name in nodes_by_id_global:
#         nd = nodes_by_id_global[name]
#         return json.dumps({"node": name, "code": nd.get("code", "")})

#     # 2. endswith match
#     matches = []
#     suffix = ":" + name
#     for nid in nodes_by_id_global:
#         if nid.endswith(suffix):
#             matches.append(nid)

#     if not matches:
#         return json.dumps({"error": "node_not_found", "node": name})

#     # 3. if multiple matches → ask agent to disambiguate
#     if len(matches) > 1:
#         return json.dumps({
#             "error": "ambiguous",
#             "node": name,
#             "candidates": matches
#         })

#     # 4. single match → return code
#     nid = matches[0]
#     nd = nodes_by_id_global[nid]
#     return json.dumps({
#         "node": nid,
#         "code": nd.get("code", ""),
#         "type": nd.get("type", "")
#     })

def tool_get_code(name: str) -> str:
    """
    Return full code for a node whose ID matches or endswith the requested name.
    Examples:
      - "separability_matrix"
      - "Linear1D"
      - "Linear1D.evaluate"
      - "_calculate_separability_matrix"
    """

    global nodes_by_id_global, method_cache, method_cache_global

    print(" -------- tool_get_code (start) -------- ")
    print(f'Requested node: #{name}#')
    name = name.strip().replace("'", "").replace('"', '').replace("`", "")
    print(f'Normalized node name: #{name}#')
    print(" -------- tool_get_code -------- ")

    if not name or not nodes_by_id_global:
        return json.dumps({"error": "invalid_request", "node": name})

    # 0. Check method cache first
    for cached_node, cached_code in method_cache.items():
        if cached_node.endswith(name) or cached_node.split(':')[-1].endswith(name):
            # ensure method_cache_global is consistent with method_cache
            try:
                method_cache_global.add(cached_node)
            except Exception:
                pass
            return json.dumps({
                "node": cached_node,
                "code": cached_code,
                "info": f"You have already accessed this. Please avoid requesting it again."
            })

    # 1. Exact match
    if name in nodes_by_id_global:
        nd = nodes_by_id_global[name]
        code = nd.get("code", "")
        method_cache[name] = code
        try:
            method_cache_global.add(name)
        except Exception:
            pass
        return json.dumps({"node": name, "code": code, "type": nd.get("type", "")})

    # 2. Endswith match (covering simple :name and :Class._name)
    matches = []
    for nid in nodes_by_id_global:
        # exclude test nodes
        if "/tests/" in nid or "test_" in nid or "tests/" in nid:
            continue
        if ':' in nid:
            entity_part = nid.split(':', 1)[1]  # part after colon
            if entity_part == name or entity_part.endswith("." + name):
                matches.append(nid)

    # 2b. If still no match, map dotted module path to file path.
    # Example: astropy.modeling.separable.separability_matrix
    # -> astropy/modeling/separable.py:separability_matrix
    if not matches and "." in name:
        parts = name.split(".")
        if len(parts) >= 2:
            func_or_class = parts[-1]
            module_path = "/".join(parts[:-1]) + ".py"
            dotted_suffix = module_path + ":" + func_or_class
            for nid in nodes_by_id_global:
                if "/tests/" in nid or "test_" in nid or "tests/" in nid:
                    continue
                if nid.endswith(dotted_suffix):
                    matches.append(nid)

    if not matches:
        return json.dumps({"error": "node_not_found", "node": name})

    # 3. Multiple matches → disambiguation needed
    if len(matches) > 1:
        return json.dumps({
            "error": "ambiguous",
            "node": name,
            "candidates": matches
        })

    # 4. Single match → return code & update method cache
    nid = matches[0]
    nd = nodes_by_id_global[nid]
    code = nd.get("code", "")
    method_cache[nid] = code
    try:
        method_cache_global.add(nid)
    except Exception:
        pass

    return json.dumps({
        "node": nid,
        "code": code,
        "type": nd.get("type", "")
    })


def tool_get_file_context(input_str: str) -> str:
    """Read a window of lines from a source file at the instance's base commit.

    Input: JSON with:
      - "file" (required): repo-relative path, e.g. "sklearn/utils/validation.py"
      - "start_line" (optional, default 1): first line to return (1-based)
      - "end_line" (optional, default start+100): last line to return (1-based, inclusive)

    Reads the file via `git show <base_commit>:<file>` so it always reflects the
    correct commit state without needing to check out the branch. Returns numbered
    lines for easy reference when constructing patches.
    """
    global current_reg_entry_global

    # --- Parse input ---
    if isinstance(input_str, dict):
        args = input_str
    else:
        try:
            args = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            # Treat plain string as file path
            args = {"file": str(input_str).strip()}

    file_path = args.get("file", "").strip()
    if not file_path:
        return json.dumps({"error": "missing_file", "desc": "Provide 'file' key with a repo-relative path."})

    start_line = int(args.get("start_line", 1))
    end_line = int(args.get("end_line", start_line + 100))
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line + 100

    # --- Get base_commit from current instance ---
    base_commit = ""
    if current_reg_entry_global:
        base_commit = current_reg_entry_global.get("base_commit", "")

    content = _read_file_at_commit(file_path, base_commit)
    if content is None:
        # Fallback: try reading from working tree
        full_path = os.path.join(REPO_LOCAL_PATH, file_path)
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                pass

    if content is None:
        return json.dumps({"error": "file_not_found", "file": file_path,
                           "desc": f"Could not read '{file_path}' at commit {base_commit[:10] if base_commit else '(none)'}."})

    all_lines = content.splitlines()
    total_lines = len(all_lines)

    # Clamp to actual file bounds
    start_line = max(1, min(start_line, total_lines))
    end_line = max(start_line, min(end_line, total_lines))

    # Build numbered output
    selected = all_lines[start_line - 1 : end_line]
    numbered = []
    for i, ln in enumerate(selected, start=start_line):
        numbered.append(f"{i:>6} | {ln}")

    return json.dumps({
        "file": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "content": "\n".join(numbered),
    }, ensure_ascii=False)


def tool_search_codebase(input_str: str) -> str:
    """Grep the repository for a pattern at the instance's base commit.

    Input: JSON with:
      - "pattern" (required): search string or regex pattern
      - "include" (optional): file glob, e.g. "*.py" (default: all files)

    Uses `git grep` against the base commit so results reflect the correct revision.
    Returns up to 30 matching lines formatted as file:line_number:content.
    """
    global current_reg_entry_global

    # --- Parse input ---
    if isinstance(input_str, dict):
        args = input_str
    else:
        try:
            args = json.loads(input_str)
        except (json.JSONDecodeError, TypeError):
            args = {"pattern": str(input_str).strip()}

    pattern = args.get("pattern", "").strip()
    if not pattern:
        return json.dumps({"error": "missing_pattern", "desc": "Provide 'pattern' key with a search string."})

    include_glob = args.get("include", "").strip()

    # --- Get base_commit from current instance ---
    base_commit = ""
    if current_reg_entry_global:
        base_commit = current_reg_entry_global.get("base_commit", "")

    # Build git grep command
    # `git grep -n` gives line numbers; `--no-color` for clean parsing
    cmd = ["git", "grep", "-n", "--no-color", "-I"]  # -I = skip binary files

    # Try extended regex first; if pattern looks simple, use fixed string for speed
    if any(c in pattern for c in r".*+?[](){}|\\^$"):
        cmd += ["-E", pattern]
    else:
        cmd += ["-F", pattern]

    # Scope to a specific commit (if available) so results are at the right revision
    if base_commit:
        cmd.append(base_commit)

    # File include filter (e.g., "*.py")
    if include_glob:
        cmd += ["--", include_glob]

    rc, out, err = run_cmd(cmd, cwd=REPO_LOCAL_PATH, timeout=30)

    if rc != 0 and not out:
        if "no matches" in err.lower() or rc == 1:
            return json.dumps({"pattern": pattern, "matches": [], "total": 0,
                               "desc": "No matches found."})
        return json.dumps({"error": "grep_failed", "rc": rc, "stderr": err[:500]})

    # Parse output lines. When searching a commit, git grep prefixes with "<commit>:"
    raw_lines = out.splitlines()
    matches = []
    commit_prefix = f"{base_commit}:" if base_commit else ""
    for raw in raw_lines:
        line = raw
        # Strip commit hash prefix if present
        if commit_prefix and line.startswith(commit_prefix):
            line = line[len(commit_prefix):]

        # Format: file:line_number:content
        parts = line.split(":", 2)
        if len(parts) >= 3:
            matches.append({
                "file": parts[0],
                "line": int(parts[1]) if parts[1].isdigit() else parts[1],
                "content": parts[2].strip()[:200],
            })
        elif len(parts) == 2:
            matches.append({"file": parts[0], "line": 0, "content": parts[1].strip()[:200]})

        if len(matches) >= 30:
            break

    # Exclude test files from results to keep focus on production code
    prod_matches = [m for m in matches if "/tests/" not in m["file"] and "test_" not in m["file"]]
    # If filtering removed everything, show all matches
    if not prod_matches and matches:
        prod_matches = matches

    return json.dumps({
        "pattern": pattern,
        "matches": prod_matches,
        "total": len(prod_matches),
        "truncated": len(raw_lines) > 30,
    }, ensure_ascii=False)


# Tool tracing state
_ACTIVE_CHAT_HISTORY: Optional[List[str]] = None
# _TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY = False  # keep chat_history compact; rely on method_cache for code
_TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY = True  # keep chat_history compact; rely on method_cache for code


def _trace_tool_call(tool_name: str, tool_input: Any, observation: Any):
    """Append tool call + observation into the active instance chat history (and print to terminal)."""
    global _ACTIVE_CHAT_HISTORY
    if _ACTIVE_CHAT_HISTORY is None:
        return

    def _clip(x: Any, n: int = 2000) -> str:
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
        return s if len(s) <= n else (s[:n] + "\n...[truncated]...")

    entry_in = f"Action: {tool_name}\nAction Input: {_clip(tool_input)}"
    # Always record the action first for readability.
    _ACTIVE_CHAT_HISTORY.append(entry_in)

    # Observations are usually large (code/json) and are already present in method_cache.
    # Keep them out of chat_history to reduce prompt bloat unless explicitly enabled.
    entry_out = f"Observation: {_clip(observation)}"
    if _TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY:
        _ACTIVE_CHAT_HISTORY.append(entry_out)

    # also print to terminal for visibility
    print("\n===== AGENT TOOL TRACE =====")
    print(entry_in)
    print(entry_out)
    print("===== /AGENT TOOL TRACE =====\n")


def _wrap_tool_for_tracing(tool_name: str, fn):
    def _wrapped(arg):
        try:
            out = fn(arg)
        except Exception as e:
            out = json.dumps({"error": str(e)})
        _trace_tool_call(tool_name, arg, out)
        return out
    return _wrapped


TOOLS = [
    Tool(
        name="classify_report",
        func=_wrap_tool_for_tracing("classify_report", tool_classify_report),
        description="LLM classifier that extracts method/class names or stack traces from a bug report.",
    ),
    Tool(
        name="get_subgraph",
        func=_wrap_tool_for_tracing("get_subgraph", tool_get_subgraph),
        description="Extract a small subgraph text & label map for start nodes.",
    ),
    Tool(
        name="get_code",
        func=_wrap_tool_for_tracing("get_code", tool_get_code),
        description="Return full code and metadata for a node id.",
    ),
    Tool(
        name="get_file_context",
        func=_wrap_tool_for_tracing("get_file_context", tool_get_file_context),
        description=(
            "Read a window of lines from a source file at the correct git commit. "
            "Input: JSON with 'file' (required), 'start_line' (default 1), 'end_line' (default start+100). "
            "Example: {\"file\": \"sklearn/utils/validation.py\", \"start_line\": 1, \"end_line\": 50}  "
            "Use this to see imports, module-level constants, surrounding code, or sibling methods "
            "that are NOT in the method cache."
        ),
    ),
    Tool(
        name="search_codebase",
        func=_wrap_tool_for_tracing("search_codebase", tool_search_codebase),
        description=(
            "Search across the repository for lines matching a pattern (grep). "
            "Input: JSON with 'pattern' (required string or regex), 'include' (optional glob like '*.py'). "
            "Example: {\"pattern\": \"def separability_matrix\", \"include\": \"*.py\"}  "
            "Returns matching file:line:content entries (max 30). "
            "Use this to find all usages/callers of a function, or to locate related code across the repo."
        ),
    ),
]

TOOLS_REVIEWER = [
    Tool(
        name="get_subgraph",
        func=_wrap_tool_for_tracing("get_subgraph", tool_get_subgraph),
        description="Extract a small subgraph text & label map for start nodes.",
    ),
    Tool(
        name="get_code",
        func=_wrap_tool_for_tracing("get_code", tool_get_code),
        description="Return full code and metadata for a node id.",
    ),
    Tool(
        name="get_file_context",
        func=_wrap_tool_for_tracing("get_file_context", tool_get_file_context),
        description=(
            "Read a window of lines from a source file at the correct git commit. "
            "Input: JSON with 'file' (required), 'start_line' (default 1), 'end_line' (default start+100). "
            "Example: {\"file\": \"sklearn/utils/validation.py\", \"start_line\": 1, \"end_line\": 50}  "
            "Use this to see imports, module-level constants, surrounding code, or sibling methods "
            "that are NOT in the method cache."
        ),
    ),
    Tool(
        name="search_codebase",
        func=_wrap_tool_for_tracing("search_codebase", tool_search_codebase),
        description=(
            "Search across the repository for lines matching a pattern (grep). "
            "Input: JSON with 'pattern' (required string or regex), 'include' (optional glob like '*.py'). "
            "Example: {\"pattern\": \"def separability_matrix\", \"include\": \"*.py\"}  "
            "Returns matching file:line:content entries (max 30). "
            "Use this to find all usages/callers of a function, or to locate related code across the repo."
        ),
    ),
]

# ---------- Globals that will be populated per instance ----------
nodes_by_id_global: Dict[str,Dict[str,Any]] = {}
edges_global: List[Dict[str,Any]] = []
outgoing_global = defaultdict(list)
incoming_global = defaultdict(list)
nodes_index_global = None    # (candidates, vectors, faiss_index, emb_obj)
meta_simple_map = {}         # simple_name->list[node_id]
method_cache_global = set()  # node ids we've fetched code for during this instance
method_cache = {}  # global cache: {node_id: code}
current_reg_entry_global = None         # reg entry dict for the instance currently prepared
classification_stats_global = {}        # instance_id -> stats dict


# ---------- orchestrator per-instance ----------
def prepare_instance_state(reg_entry: Dict[str,Any], build_embeddings: bool = False):
    """
    Load codegraph & adjacency, build simple-name map (meta_simple_map), and optionally build embeddings.

    - reg_entry: registry entry containing at least 'code_graph_path', 'repo', 'base_commit'
    - build_embeddings: if True, also call build_embedding_index and populate nodes_index_global
    """
    global nodes_by_id_global, edges_global, outgoing_global, incoming_global
    global nodes_index_global, meta_simple_map, method_cache_global, method_cache, classification_stats_global

    # Reset globals
    nodes_by_id_global = {}
    edges_global = []
    outgoing_global = defaultdict(list)
    incoming_global = defaultdict(list)
    nodes_index_global = None
    meta_simple_map = {}
    method_cache_global = set()
    method_cache = {}  # global cache: {node_id: code}

    # Load codegraph (nodes_by_id, edges)
    code_graph_path = reg_entry.get("code_graph_path") or reg_entry.get("codegraph") or reg_entry.get("codegraph_path")
    if not code_graph_path or not os.path.exists(code_graph_path):
        # leave empty — caller must handle missing artifacts
        print("prepare_instance_state: code_graph not found:", code_graph_path)
        return

    nodes_by_id_global, edges_global = load_codegraph(code_graph_path)
    outgoing_global, incoming_global = build_edge_adjacency(edges_global)

    # Build simple name map (filename/module/entity -> [node_ids])
    simple_map = defaultdict(list)
    for nid, nd in nodes_by_id_global.items():
        try:
            if nid.endswith(".py"):
                fname = os.path.basename(nid)
                mod = os.path.splitext(fname)[0]
                simple_map[fname].append(nid)
                simple_map[fname.lower()].append(nid)
                simple_map[mod].append(nid)
                simple_map[mod.lower()].append(nid)
            elif ":" in nid:
                short = nid.split(":")[-1].split(".")[-1]
                simple_map[short].append(nid)
                simple_map[short.lower()].append(nid)
        except Exception:
            continue
    meta_simple_map = dict(simple_map)

    # initialize classification stats tracking for this instance if available
    inst_id = reg_entry.get("instance_id") or reg_entry.get("instance")
    if inst_id:
        if 'classification_stats_global' not in globals():
            classification_stats_global = {}
        classification_stats_global.setdefault(inst_id, {})

    # Lazy: build embedding index only if requested
    nodes_index_global = None
    if build_embeddings:
        repo_slug = reg_entry.get("repo","").replace("/","__")
        commit = reg_entry.get("base_commit", reg_entry.get("commit","unknown"))
        try:
            nodes_index_global = build_embedding_index(nodes_by_id_global, repo_slug, commit, exclude_tests_flag=EXCLUDE_TESTS)
        except Exception as e:
            print("prepare_instance_state: embedding build failed (continuing without embeddings):", e)
            nodes_index_global = None

    # method_cache_global already reset above
    return


# --- LangGraph tool-calling agent (preferred for GPT-5 robustness) ---
def _render_lg_stream_chunk(chunk: Any, max_chars: int = 2000) -> List[str]:
    """Render a single LangGraph stream chunk into human-readable history lines."""
    lines: List[str] = []
    if not isinstance(chunk, dict):
        return lines

    msgs = chunk.get("messages")
    if not msgs:
        # Handle node-keyed chunk format from LangGraph stream_mode="updates"
        for key in ("agent", "tools", "action"):
            inner = chunk.get(key)
            if isinstance(inner, dict):
                msgs = inner.get("messages")
                if msgs:
                    break
    if not msgs:
        return lines

    def clip(s: str) -> str:
        s = s or ""
        return s if len(s) <= max_chars else (s[:max_chars] + "\n...[truncated]...")

    for m in msgs:
        try:
            mtype = getattr(m, "type", None) or m.__class__.__name__
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", None)
            addk = getattr(m, "additional_kwargs", None) or {}

            # Tool observations
            if isinstance(m, ToolMessage) or str(mtype).lower().startswith("tool"):
                tool_name = getattr(m, "name", None) or addk.get("name") or "tool"
                if content:
                    lines.append(f"Observation ({tool_name}): " + clip(str(content)))
                continue

            if tool_calls:
                lines.append("Action: tool_calls\nAction Input: " + clip(json.dumps(tool_calls, ensure_ascii=False)))

            # tool call list can also be in additional_kwargs
            if not tool_calls and addk.get("tool_calls"):
                lines.append("Action: tool_calls\nAction Input: " + clip(json.dumps(addk.get("tool_calls"), ensure_ascii=False)))

            # role-like rendering
            if content:
                text = str(content).strip()
                if str(mtype).lower().startswith("system"):
                    lines.append("SYSTEM: " + clip(text))
                elif str(mtype).lower().startswith("human"):
                    lines.append("HUMAN: " + clip(text))
                elif str(mtype).lower().startswith("ai"):
                    # Strip model-echoed "Thought:" prefix to avoid duplication
                    if text.lower().startswith("thought:"):
                        text = text[len("thought:"):].strip()
                    lines.append("Thought: " + text)  # never truncate Thoughts
                else:
                    lines.append(f"{mtype}: " + clip(text))
        except Exception:
            continue

    return lines


def _append_langgraph_ai_messages_only(chat_history: List[str], chunk: Any, max_chars: int = 0):
    """Append only non-empty conversational messages (AI/Human/System) from a LangGraph stream chunk.

    Handles both flat ({"messages": [...]}) and node-keyed ({"agent": {"messages": [...]}}) chunk formats.
    AI reasoning content is labelled as 'Thought:' in the chat history and printed to terminal.
    Tool results are handled separately by the tracing wrapper.
    """
    if not isinstance(chunk, dict):
        return

    # --- Extract messages from either chunk format ---
    msgs = chunk.get("messages")
    if not msgs:
        # LangGraph stream_mode="updates" (default) nests messages under node keys
        for key in ("agent", "tools", "action"):
            inner = chunk.get(key)
            if isinstance(inner, dict):
                msgs = inner.get("messages")
                if msgs:
                    break
    if not msgs:
        return

    def clip(s: str) -> str:
        s = s or ""
        if max_chars <= 0:
            return s  # no truncation
        return s if len(s) <= max_chars else (s[:max_chars] + "\n...[truncated]...")

    for m in msgs:
        try:
            mtype = (getattr(m, "type", None) or m.__class__.__name__ or "").lower()
            # Skip tool messages entirely here
            if isinstance(m, ToolMessage) or mtype.startswith("tool"):
                continue
            content = getattr(m, "content", "")
            if not content:
                continue
            # AI messages → label as Thought (the model's reasoning before/after tool calls)
            if mtype.startswith("ai"):
                text = str(content).strip()
                # Strip model-echoed "Thought:" prefix to avoid "Thought: Thought: ..."
                if text.lower().startswith("thought:"):
                    text = text[len("thought:"):].strip()
                chat_history.append("Thought: " + text)
                print("\n===== AGENT THOUGHT =====")
                print(text)
                print("===== /AGENT THOUGHT =====\n")
            elif mtype.startswith("system"):
                chat_history.append("SYSTEM: " + clip(str(content)))
            elif mtype.startswith("human"):
                chat_history.append("HUMAN: " + clip(str(content)))
        except Exception:
            continue


def _parse_json_best_effort(text: str, preferred_keys: Optional[list] = None):
    """Parse a JSON object from *text* robustly.

    Instead of the old greedy ``{.*}`` regex (which could match across
    unrelated JSON blocks), this uses a balanced-brace scanner to extract
    every top-level ``{…}`` candidate and tries ``json.loads`` on each.

    When *preferred_keys* is given (e.g. ``["revised_report"]``), any
    candidate containing one of those keys is returned immediately.
    Otherwise the **largest** valid JSON candidate is returned.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    # Fast path: entire string is valid JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Strip markdown code fences that wrap JSON
    stripped = text
    for fence in ("```json\n", "```JSON\n", "```\n"):
        if fence in stripped:
            stripped = stripped.replace(fence, "").replace("\n```", "")
    if stripped != text:
        try:
            return json.loads(stripped)
        except Exception:
            pass

    # Balanced-brace scanner: find every top-level {…} substring
    candidates: list[dict] = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < len(text):
                ch = text[j]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate_str = text[i:j+1]
                            try:
                                obj = json.loads(candidate_str)
                                if isinstance(obj, dict):
                                    # Immediate return if it has a preferred key
                                    if preferred_keys:
                                        for pk in preferred_keys:
                                            if pk in obj:
                                                return obj
                                    candidates.append(obj)
                            except Exception:
                                pass
                            break
                j += 1
            i = j + 1 if j < len(text) else i + 1
        else:
            i += 1

    if not candidates:
        return None

    # Return the largest candidate (most keys) — heuristic for
    # "the real output" vs. small tool-output fragments.
    return max(candidates, key=lambda c: len(c))


def _filter_chat_history_for_method_cache(chat_history: List[str]) -> List[str]:
    """Filter chat_history: remove get_code Observation entries (code is in method_cache).

    Keeps all other entries intact including observations from classify_report,
    get_subgraph, get_file_context, search_codebase, etc. which may contain non-code insights.
    """
    filtered: List[str] = []
    skip_next_observation = False
    for line in chat_history:
        if line.startswith("Action: get_code"):
            skip_next_observation = True
            filtered.append(line)
        elif line.startswith("Observation:") and skip_next_observation:
            skip_next_observation = False
            filtered.append("Observation: [code available in method cache]")
        else:
            skip_next_observation = False
            filtered.append(line)
    return filtered


def run_reviewer_agent(draft_report: Any, problem: str, method_cache: Dict[str, str], chat_history: List[str]):
    """Tool-enabled reviewer that validates and improves the draft report."""
    print("\n=== Running reviewer agent to validate and improve the draft report ===")
    reviewer_history: List[str] = []
    global _ACTIVE_CHAT_HISTORY
    _ACTIVE_CHAT_HISTORY = reviewer_history
    global _TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY
    prev_trace_obs = _TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY
    _TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY = True

    draft_json = draft_report
    if not isinstance(draft_json, str):
        try:
            draft_json = json.dumps(draft_report, ensure_ascii=False)
        except Exception:
            draft_json = str(draft_report)

    instruction = (
        "You are a reviewer agent for bug reports. Your job is to carefully review and validate "
        "the draft enhanced bug report against the original bug report, the method cache (source code), "
        "and the full chat history from the previous agent's investigation. "
        "You must verify that the draft report's claims (root cause, problem location, fix suggestions) "
        "are consistent with the actual source code and the findings from the investigation. "
        "If you cannot verify a claim, soften it or remove it. "
        "\n\n"
        "You are provided with:\n"
        "1. The original bug report (problem statement).\n"
        "2. The draft enhanced report (JSON) produced by the previous agent.\n"
        "3. A method cache containing the full source code of all methods/classes the previous agent fetched.\n"
        "4. The full chat history from the previous agent's investigation (with get_code observations removed "
        "since that code is already in the method cache; observations from other tools are preserved).\n"
        "\n"
        "IMPORTANT — You MUST follow this two-phase process:\n"
        "\n"
        "Phase 1: VERIFICATION AND COMPLETENESS CHECK (write your reasoning as plain text)\n"
        "Before producing any JSON, you must write out a detailed analysis covering:\n"
        "- For each method listed in the draft report's problem_location, look up its code in the method cache "
        "and verify whether the draft report's claims about that method are accurate.\n"
        "- Verify the RootCause: does the actual code support the stated root cause? Quote specific lines or patterns.\n"
        "- Verify the possible_fix and possible_fix_code: is the proposed fix correct and complete? "
        "Does it address the root cause without introducing new issues? Critically ask: if this fix were applied, "
        "would any other code in the same class or file break because it depends on the behavior being changed?\n"
        "- Check for any claims in the draft that are not supported by code evidence. Flag them.\n"
        "- Note any important findings from the chat history that the draft report missed.\n"
        "- COMPLETENESS CHECK: Review the chat history for any secondary findings or related issues "
        "the agent discovered that were NOT included in the draft report. If the agent found that the same "
        "root cause pattern exists in multiple locations (e.g., case-sensitive comparisons in several places, "
        "missing parameter forwarding in multiple methods), ALL such locations MUST be included in the revised "
        "report's problem_location, possible_fix, and Suggestions fields. Do NOT dismiss any agent finding as "
        "'out of scope' — if it shares the same root cause, it belongs in the report.\n"
        "- If you need to verify additional locations, use `get_file_context` or `search_codebase` to confirm.\n"
        "\n"
        "Available tools (use ONLY when needed):\n"
        "- `get_code`: Retrieve source code for a method/class NOT already in the method cache.\n"
        "- `get_subgraph`: Explore call graph relationships around a node.\n"
        "- `get_file_context`: Read a window of lines from a source file at the correct commit. "
        "Input JSON: {\"file\": \"path/to/file.py\", \"start_line\": 1, \"end_line\": 500}\n"
        "- `search_codebase`: Grep the repository for a pattern. "
        "Input JSON: {\"pattern\": \"function_name\", \"include\": \"*.py\"}\n"
        "\n"
        "Do NOT call tools for methods whose code is already in the method cache — just read it from there. "
        "Before any tool call, write your reasoning explaining why the tool is needed and what you expect to learn. "
        "After receiving a tool result, analyze what was returned before proceeding. "
        "Do NOT prefix your text with 'Thought:'. "
        "\n"
        "Phase 2: FINAL OUTPUT\n"
        "Only AFTER completing your verification and completeness check, produce the final JSON object. "
        "Write your analysis text first, then on a new line output the JSON.\n"
        "The JSON must have EXACTLY these three top-level keys:\n"
        "{\n"
        "  \"revised_report\": { <the full bug report with Title, Description, RootCause, StepsToReproduce, "
        "ExpectedBehavior, ObservedBehavior, FixSteps> },\n"
        "  \"changes\": [ <list of brief edits you made> ],\n"
        "  \"evidence\": [ <list of supporting references from code/investigation> ]\n"
        "}\n"
        "CRITICAL: The 'changes' and 'evidence' keys must be OUTSIDE the revised_report, NOT inside it. "
        "The revised_report dict must contain ONLY the bug report fields listed above. "
        "Do NOT nest changes or evidence inside the revised_report.\n"
        "\n"
        "CRITICAL — SUGGESTIONS MUST BE SPECIFIC AND COMPLETE:\n"
        "The Suggestions field in the revised report is the most important field for downstream repair. "
        "It MUST be specific and actionable: name the exact methods that need changes, describe exactly what "
        "each change should do, and cover ALL affected locations (not just the primary one). "
        "Vague suggestions like 'modify the function to handle this case' are insufficient — instead write "
        "'In _line_type(), compile the regex with re.IGNORECASE; in _get_tables_from_qdp_file(), change "
        "v == \"NO\" to v.upper() == \"NO\"'. Every fix the agent identified must appear in Suggestions.\n"
        "\n"
        "IMPORTANT — MAINTAIN REPORT STRUCTURE:\n"
        "The revised_report MUST preserve the exact same field names/structure as the draft enhanced report. "
        "Do NOT add any new fields that are not already present in the draft report. "
        "You may update the values of existing fields (fix errors, add missing locations, improve descriptions), "
        "but the set of top-level keys in the revised_report must match the draft report's keys exactly.\n"
    )

    # --- Build reviewer context ---
    # Use full chat_history, but remove get_code Observation entries
    # (since that code is provided in method_cache). Keep observations from other tools.
    filtered_history = _filter_chat_history_for_method_cache(chat_history)

    # Build method cache text with full source code
    method_cache_text_parts = []
    for node_id, code in method_cache.items():
        method_cache_text_parts.append(f"--- {node_id} ---\n{code}")
    method_cache_text = "\n\n".join(method_cache_text_parts) if method_cache_text_parts else "(empty)"

    user_text = (
        "Original bug report:\n" + (problem or "") +
        "\n\n=== Draft report JSON ===\n" + draft_json +
        "\n\n=== Method cache (full source code of fetched methods) ===\n" + method_cache_text +
        "\n\n=== Full chat history from previous agent investigation ===\n"
        + "\n".join(filtered_history)
    )

    agent_events: List[Any] = []
    reviewer_llm = make_chat_llm(OPENAI_MODEL, LLM_TEMPERATURE)

    # --- Preferred: LangGraph tool-calling reviewer (robust for GPT-5) ---
    if LANGGRAPH_AVAILABLE:
        reviewer_history.append("[reviewer] Using LangGraph tool-calling agent")
        lg_reviewer = lg_create_react_agent(reviewer_llm, TOOLS_REVIEWER)
        inputs = {"messages": [SystemMessage(content=instruction), HumanMessage(content=user_text)]}
        try:
            for chunk in lg_reviewer.stream(inputs, config={"recursion_limit": 60}):
                agent_events.append(chunk)
                _append_langgraph_ai_messages_only(reviewer_history, chunk)
        except Exception as e:
            reviewer_history.append(f"Reviewer agent runtime error: {str(e)}")

    # --- Fallback: legacy LangChain reviewer ---
    else:
        reviewer_history.append("[reviewer] Using LangChain legacy ReAct agent (fallback)")
        reviewer_agent = initialize_agent(
            TOOLS_REVIEWER,
            reviewer_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
        )
        try:
            agent_result = reviewer_agent.invoke({"input": instruction + "\n\n" + user_text})
            if isinstance(agent_result, dict) and "output" in agent_result:
                reviewer_history.append(str(agent_result.get("output")))
            else:
                reviewer_history.append(str(agent_result))
        except Exception as e:
            reviewer_history.append(f"Reviewer agent runtime error: {str(e)}")

    _ACTIVE_CHAT_HISTORY = None
    _TRACE_INCLUDE_OBSERVATIONS_IN_CHAT_HISTORY = prev_trace_obs

    # NOTE: _append_langgraph_ai_messages_only already captures all AI messages
    # into reviewer_history during streaming. No need for a separate "pull last AI
    # message" pass — that was redundant and had a node-keyed chunk parsing bug.

    # best-effort parse from last non-empty reviewer history line
    # Prefer JSON that contains 'revised_report' key (skip tool-output JSON fragments)
    _TOOL_OBS_KEYS = {"node", "pattern", "error"}
    _REPORT_KEYS = {"Title", "Description", "RootCause", "Suggestions", "problem_location",
                     "revised_report", "changes", "evidence"}

    def _is_tool_observation(obj: dict) -> bool:
        """Reject JSON objects that are tool observations (get_code, search_codebase, etc.)."""
        if obj.keys() & _TOOL_OBS_KEYS:
            return True
        if "file" in obj and ("content" in obj or "start_line" in obj):
            return True
        return False

    parsed = None
    fallback_parsed = None  # first valid JSON (in case nothing has 'revised_report')
    for line in reversed(reviewer_history):
        candidate = _parse_json_best_effort(line, preferred_keys=["revised_report"])
        if candidate:
            if "revised_report" in candidate:
                parsed = candidate
                break
            elif fallback_parsed is None and not _is_tool_observation(candidate):
                if candidate.keys() & _REPORT_KEYS:
                    fallback_parsed = candidate

    if not parsed:
        parsed = fallback_parsed

    if not parsed and agent_events:
        # try to parse from any string content in events (check node-keyed chunks)
        for ev in reversed(agent_events):
            if isinstance(ev, dict):
                # Handle node-keyed chunks from LangGraph stream_mode="updates"
                for key in ("agent", "tools", "action"):
                    inner = ev.get(key)
                    if isinstance(inner, dict):
                        msgs = inner.get("messages", [])
                        for m in msgs:
                            content = getattr(m, "content", "")
                            if content:
                                candidate = _parse_json_best_effort(
                                    str(content), preferred_keys=["revised_report"]
                                )
                                if candidate:
                                    if "revised_report" in candidate:
                                        parsed = candidate
                                        break
                                    elif fallback_parsed is None and not _is_tool_observation(candidate):
                                        if candidate.keys() & _REPORT_KEYS:
                                            fallback_parsed = candidate
                    if parsed and "revised_report" in parsed:
                        break
            if parsed and "revised_report" in parsed:
                break
        if not parsed:
            parsed = fallback_parsed

    if not parsed:
        return {
            "revised_report": draft_report,
            "changes": ["Reviewer produced no parsable JSON; kept draft."],
            "evidence": [],
            "reviewer_history": reviewer_history,
        }

    revised_report = parsed.get("revised_report", parsed)
    changes = parsed.get("changes", [])
    evidence = parsed.get("evidence", [])

    # Guard: if the LLM embedded changes/evidence INSIDE the revised_report
    # (instead of as sibling keys), extract them and strip from the report.
    _VALID_BUG_REPORT_KEYS = {
        "Title", "Description", "RootCause", "StepsToReproduce",
        "ExpectedBehavior", "ObservedBehavior", "Suggestions",
        "problem_location", "possible_fix", "possible_fix_code",
    }
    if isinstance(revised_report, dict):
        if "changes" in revised_report:
            if not changes:
                changes = revised_report.pop("changes")
            else:
                revised_report.pop("changes")
        if "evidence" in revised_report:
            if not evidence:
                evidence = revised_report.pop("evidence")
            else:
                revised_report.pop("evidence")
        for extra_key in list(revised_report.keys()):
            if extra_key not in _VALID_BUG_REPORT_KEYS:
                revised_report.pop(extra_key)

    return {
        "revised_report": revised_report,
        "changes": changes,
        "evidence": evidence,
        "reviewer_history": reviewer_history,
    }


def _read_file_at_commit(file_path: str, base_commit: str) -> Optional[str]:
    """Read file content from the git repo at a specific commit using `git show`.

    This avoids needing to checkout the commit — works on any branch state.
    Returns the file content as string, or None if the file/commit doesn't exist.
    """
    if not base_commit or not file_path:
        return None
    rc, out, err = run_cmd(
        ["git", "show", f"{base_commit}:{file_path}"],
        cwd=REPO_LOCAL_PATH, timeout=30
    )
    if rc == 0:
        return out
    return None


def run_for_instance(instance: Dict[str,Any], reg_entry: Dict[str,Any], out_summary_file: str):
    """
    Orchestrate a single instance with full agent autonomy:
    """
    global nodes_index_global, method_cache_global, method_cache, current_reg_entry_global, classification_stats_global

    instance_id = instance.get("instance_id")
    print(f"\n=== Processing instance: {instance_id} ===")

    current_reg_entry_global = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    method_cache_global = set()
    method_cache = {}

    chat_history: List[str] = []
    # Enable per-instance tool tracing into chat_history
    global _ACTIVE_CHAT_HISTORY
    _ACTIVE_CHAT_HISTORY = chat_history

    classification_stats_global.setdefault(instance_id, {})

    problem = instance.get("problem_statement", "") or ""
    fail_to_pass_raw = instance.get("FAIL_TO_PASS", "") or instance.get("fail_to_pass", "")

    instruction = (
        "You are an autonomous code-reasoning agent. "
        "IMPORTANT: Before EVERY tool call, you MUST first write out your reasoning and thought process "
        "as plain text explaining WHY you are about to call that tool and what you expect to learn. "
        "This reasoning MUST appear as message content (not inside the tool call). "
        "Additionally, after receiving each tool result, you MUST analyze what was returned before proceeding: "
        "summarize the key findings (reference specific variable names, function signatures, control flow, "
        "and anything suspicious), state what you have learned so far, and explain what to investigate next and why. "
        "Do NOT skip analysis — never jump from one tool result to the next tool call without first discussing what you found. "
        "Do NOT prefix your text with 'Thought:'. "
        "BEFORE doing any code inspection, you MUST CALL the tool `classify_report` with the bug problem text. "
        "After you receive the classify_report result: "
        "(1) If it contains any method or class candidates, you MUST CALL `get_code` on at least TWO distinct candidates "
        "to ground your analysis in actual source. "
        "(2) If only file/module candidates are available (or get_code is ambiguous), you MUST CALL `get_subgraph` for at least one candidate to locate the right symbol, then CALL `get_code`. "
        "Only AFTER completing the above tool calls should you write conclusions or a final answer. "
        "Always prefer analyzing real code returned by tools over speculation. "
        "Remember: always include your reasoning as text content alongside every tool call.\n\n"
        "MULTI-LOCATION AWARENESS:\n"
        "After identifying the primary buggy method, check whether the same flawed pattern or assumption "
        "exists in closely related functions within the same file or module. Use `get_file_context` to see "
        "surrounding code if needed, and `search_codebase` to check for the same pattern in related files. "
        "However, stay focused on the specific root cause — do NOT audit every unrelated function in the file. "
        "Only investigate functions that handle the same data, share the same logic pattern, or depend on "
        "the same assumption as the primary bug.\n\n"
        "CRITICAL — REPORT ALL FINDINGS:\n"
        "Your final analysis MUST include ALL issues you discovered, both primary and secondary. "
        "Do NOT dismiss any finding as 'out of scope' or 'separate concern' — if a secondary issue shares "
        "the same root cause pattern (e.g., case sensitivity in one place implies case sensitivity elsewhere "
        "in the same module), it MUST be reported. A complete diagnosis includes every location and every "
        "manifestation of the root cause, even if some seem minor.\n\n"
        "COMPLETENESS OF YOUR PROPOSED FIX:\n"
        "After formulating your proposed fix, critically ask yourself: 'If this fix is applied, "
        "what else in the same class or file would break or need updating?' Review the methods you "
        "have already inspected for any hardcoded values, constants, or logic that silently depends on "
        "the behavior you are changing. If applying your fix would make other code incorrect, those "
        "locations are part of the complete fix and MUST be included in your analysis.\n\n"
        "Available tools:\n"
        "- `classify_report`: Extract programming entities from the bug report.\n"
        "- `get_code`: Fetch source code for a method/class/file from the code graph.\n"
        "- `get_subgraph`: Explore call graph relationships around a node.\n"
        "- `get_file_context`: Read a window of lines from a source file at the correct commit. "
        "Input JSON: {\"file\": \"path/to/file.py\", \"start_line\": 1, \"end_line\": 500}\n"
        "- `search_codebase`: Grep the repository for a pattern. "
        "Input JSON: {\"pattern\": \"function_name\", \"include\": \"*.py\"}\n"
    )

    user_text = f"Problem: {problem}\n"
    # user_text = f"Problem: {problem}\n" + (f"\nFailing tests:\n{fail_to_pass_raw}\n" if fail_to_pass_raw else "")

    agent_events: List[Any] = []

    # --- Preferred: LangGraph tool-calling agent (robust for GPT-5) ---
    if LANGGRAPH_AVAILABLE:
        chat_history.append("[agent] Using LangGraph tool-calling agent")
        agent_llm = make_chat_llm(OPENAI_MODEL, LLM_TEMPERATURE)
        lg_agent = lg_create_react_agent(agent_llm, TOOLS)

        inputs = {"messages": [SystemMessage(content=instruction), HumanMessage(content=user_text)]}

        try:
            # Stream to capture full tool traces.
            # Increase recursion_limit to reduce false stops when the agent needs more tool steps.
            for chunk in lg_agent.stream(inputs, config={"recursion_limit": 100}):
                agent_events.append(chunk)
                # Store any conversational AI messages if present (no forcing)
                _append_langgraph_ai_messages_only(chat_history, chunk)
        except Exception as e:
            agent_events = [{"output": f"LangGraph agent runtime error: {str(e)}"}]
            chat_history.append(agent_events[0]["output"])

    # --- Fallback: legacy LangChain agent ---
    else:
        chat_history.append("[agent] Using LangChain legacy ReAct agent (fallback)")
        agent_llm = make_chat_llm(OPENAI_MODEL, LLM_TEMPERATURE)
        agent = initialize_agent(
            TOOLS,
            agent_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
        )
        try:
            agent_result = agent.invoke({"input": instruction + "\n\n" + user_text})
        except Exception as e:
            agent_result = {"output": f"Agent runtime error: {str(e)}"}

        if isinstance(agent_result, dict) and "output" in agent_result:
            agent_events = [agent_result]
            chat_history.append(str(agent_result.get("output")))
        else:
            agent_events = [{"output": str(agent_result)}]
            chat_history.append(str(agent_result))

    # 6) Interaction summary
    interaction_entry = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "agent_raw_output": agent_events,
        "method_cache_global": list(method_cache_global)
    }

    # Disable per-instance tool tracing
    _ACTIVE_CHAT_HISTORY = None

    # final_report = generate_final_bug_report(method_cache, problem, chat_history)
    # Filter chat_history: remove get_code observations (code is in method_cache)
    filtered_chat_history = _filter_chat_history_for_method_cache(chat_history)

    # Build a focused agent analysis from ALL reasoning entries (Thoughts).
    # Each Thought captures the agent's analysis after a tool call — together they
    # form the complete reasoning chain without the verbose tool observations.
    # This is much more compact than the full filtered_chat_history, yet preserves
    # all analytical insights that the single last-Thought approach was losing.
    agent_thoughts = [entry for entry in chat_history if entry.startswith("Thought:")]
    agent_analysis = "\n\n".join(agent_thoughts) if agent_thoughts else ""
    print("####################################")
    print(f"agent_analysis: {len(agent_thoughts)} thought entries, {len(agent_analysis)} chars")
    print("####################################")

    # --- Phase A: Generate the bug report (LLM call) ---
    # If this fails entirely (network error, API error, etc.), we skip this instance.
    final_bug_report = None
    try:
        final_bug_report = generate_final_bug_report(method_cache, problem, agent_analysis)
        print("####################################")
        print("Final bug report called manually:", final_bug_report)
        print("####################################")
    except Exception as e:
        print(f"FATAL: generate_final_bug_report failed for {instance_id}: {e}")
        print("Skipping this instance — will retry later.")
        # Save a minimal error record so we know which instance failed
        error_summary = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "error": f"generate_final_bug_report failed: {e}",
            "chat_history": chat_history,
            "method_cache": method_cache,
            "model_patch": "",
            "bug_report": {},
        }
        all_entries = load_json_safe(out_summary_file)
        replaced = False
        for i, entry in enumerate(all_entries):
            if entry.get("instance_id") == instance_id:
                all_entries[i] = error_summary
                replaced = True
                break
        if not replaced:
            all_entries.append(error_summary)
        save_json_atomic(all_entries, out_summary_file)
        current_reg_entry_global = None
        return error_summary

    # --- Phase B: Parse the LLM output ---
    # If parsing fails, pass the raw LLM output to the reviewer agent —
    # it can often fix formatting/JSON issues on its own.
    final_report = {}
    raw_report_for_reviewer = None  # will hold raw text if parsing fails
    parsed_report = _parse_json_best_effort(final_bug_report, preferred_keys=["revised_report", "title", "bug_location"])
    if parsed_report:
        final_report = parsed_report
    else:
        print(f"WARNING: Could not parse final bug report JSON for {instance_id}. "
              "Passing raw LLM output to reviewer agent for recovery.")
        # Keep the raw text — reviewer will receive it and can extract/fix the report
        raw_report_for_reviewer = final_bug_report
        # Use a minimal dict so downstream code doesn't crash,
        # but the reviewer will get the full raw text to work with
        final_report = {"_raw_unparsed_report": final_bug_report}

    # Reviewer agent pass (tool-enabled)
    # If we couldn't parse the report, pass the raw LLM output so the reviewer can recover it
    # --- Save single-agent output (before reviewer) to OUT_SINGLE_ENHANCED_FILE ---
    single_agent_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": classification_stats_global.get(instance_id, {}),
        "method_cache": method_cache,
        "class_skeleton_cache": {},
        "chat_history": chat_history,
        "bug_report": final_report
    }
    all_single = load_json_safe(OUT_SINGLE_ENHANCED_FILE)
    replaced_single = False
    for i, e in enumerate(all_single):
        if e.get("instance_id") == instance_id:
            all_single[i] = single_agent_summary
            replaced_single = True
            break
    if not replaced_single:
        all_single.append(single_agent_summary)
    save_json_atomic(all_single, OUT_SINGLE_ENHANCED_FILE)
    print(f"Wrote/updated single-agent output for {instance_id} into {OUT_SINGLE_ENHANCED_FILE}")

    reviewer_draft = final_report
    if raw_report_for_reviewer:
        # Wrap raw text in a dict the reviewer can understand and fix
        reviewer_draft = {
            "_parsing_failed": True,
            "_raw_llm_output": raw_report_for_reviewer,
            "_instruction": "The bug report LLM produced output that could not be parsed as JSON. "
                            "Please extract the structured report from the raw text below and "
                            "produce a proper revised_report JSON.",
        }
    reviewer_result = run_reviewer_agent(reviewer_draft, problem, method_cache, chat_history)
    final_report = reviewer_result.get("revised_report", final_report)

    # 7) Build instance summary object and save incrementally
    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": classification_stats_global.get(instance_id, {}),
        "method_cache": method_cache,
        "class_skeleton_cache": {},
        "chat_history": chat_history,
        "reviewer_changes": reviewer_result.get("changes", []),
        "reviewer_evidence": reviewer_result.get("evidence", []),
        "reviewer_history": reviewer_result.get("reviewer_history", []),
        "bug_report": final_report
    }

    # fill analyzed_methods & skeletons
    for nid in list(method_cache_global):
        nd = nodes_by_id_global.get(nid)
        if not nd:
            continue
        code = nd.get("code","")
        # instance_summary["analyzed_methods"][nid] = code
        if nd.get("type") == "class":
            # build skeleton: signature lines and method names (top-level)
            lines = code.splitlines()
            skeleton = []
            for ln in lines[:120]:
                if ln.strip().startswith("def ") or ln.strip().startswith("class ") or ln.strip().startswith("async def "):
                    skeleton.append(ln)
                if len(skeleton) > 80:
                    break
            instance_summary["class_skeleton_cache"][nid] = "\n".join(skeleton)

    # method_extracted_successfully = len(instance_summary["analyzed_methods"]) > 0

    # Append or update in output file
    all_entries = load_json_safe(out_summary_file)
    replaced = False
    for i, e in enumerate(all_entries):
        if e.get("instance_id") == instance_id:
            all_entries[i] = instance_summary
            replaced = True
            break
    if not replaced:
        all_entries.append(instance_summary)
    save_json_atomic(all_entries, out_summary_file)
    print(f"Wrote/updated instance {instance_id} into {out_summary_file}")

    # cleanup current_reg_entry_global for safety
    current_reg_entry_global = None
    return instance_summary

# ---------- helper to generate final enhanced report (LLM-driven) ----------
# def generate_enhanced_report(problem: str, instance_id: str, interaction_entry: Dict[str,Any], method_cache: List[str], agent_raw_output: str):
#     schema_template = {
#         "Title": "<Bug title>",
#         "Description": "<Improved description based on analysis>",
#         "RootCause": "<Identified root cause>",
#         "StepsToReproduce": None,
#         "ExpectedBehavior": "<Correct system behavior>",
#         "ObservedBehavior": "<Actual faulty behavior>",
#         "Suggestions": "<Possible fixes or mitigation steps>",
#         "problem_location": {"files": [], "classes": [], "methods": []},
#         "possible_fix": "<Suggested resolution>",
#         "possible_fix_code": {}
#     }
#     template = """
#     You are a bug report writer. 

#     INPUT:
#     problem_statement:
#     {problem}


#     Fill the schema fields with best-effort content. For possible_fix_code include full fixed method implementation(s) if agent suggested any; otherwise leave empty.
#     """
#     prompt = PromptTemplate.from_template(template)
#     chain = LLMChain(llm=LLM, prompt=prompt)
#     print("--------- generate_enhanced_report (start) ---------")
#     response = chain.run({'problem': problem})
#     print("response:", response)
#     text = response
#     # print('Method_cache:', method_cache)
#     print("--------- generate_enhanced_report (end) ---------")

#     # resp = LLM([{"role":"user","content":prompt}])
#     # text = resp.content.strip()
#     try:
#         out = json.loads(text)
#     except Exception:
#         m = re.search(r"(\{.*\})", text, re.S)
#         if m:
#             try:
#                 out = json.loads(m.group(1))
#             except Exception:
#                 out = {"error":"could_not_parse_llm_output", "raw": text}
#         else:
#             out = {"error":"no_json_from_llm", "raw": text}
#     return out

# ---------- helper to generate final enhanced report (LLM-driven) ----------
def generate_final_bug_report(method_cache, dev_written_bug_report, chat_history):
    """Generate the final bug report based on analyzed methods."""
    

    template = '''
    You are a **professional bug report assistant** responsible for analyzing and improving bug reports to diagnose the root cause effectively.  

    ### **Task**  
    Given three input sources, enhance an existing developer-written bug report by establishing connections between the provided information and improving key fields.  

    ### **Inputs**  
    1. **Original Bug Report**: Developer-written report describing the issue context.  
    2. **Agent Analysis**: The reasoning chain from an autonomous agent that investigated the bug by analyzing source code, call graphs, and file context. Each entry is the agent's analytical reasoning after examining tool results.  
    3. **Source Code Methods**: Methods analyzed by the agent that are related to the issue or appear in the code dependency graph.  

    ### **Guidelines for Enhancement**  
    - **Analyze the Input Data**:  
    - Examine the original bug report to understand the reported behavior and context.  
    - Carefully read the full agent analysis chain — it contains the step-by-step reasoning that led to root cause identification, including any secondary findings about related issues.  
    - Cross-reference the agent's findings with the actual source code methods provided.  

    - **Determine the Root Cause**:  
    - Establish logical connections between the problem statement, analyzed methods, and agent analysis.  
    - Focus on the most relevant methods contributing to the issue.  
    - If the agent identified MULTIPLE manifestations of the same root cause (e.g., the same flawed assumption appearing in several functions), ALL of them must be captured.  

    - **Enhance Each Bug Report Field**:  
    - **Description**: Provide a more detailed, structured, and coherent summary.  
    - **RootCause**: Clearly explain the main issue or defect identified from the combined analysis.  
    - **StepsToReproduce**: Refine or generate accurate reproduction steps if sufficient context is available.  
    - **ExpectedBehavior**: Describe the intended correct behavior.  
    - **ObservedBehavior**: Specify the actual behavior observed as per the problem statement. 
    - **FixSteps**: give me the exact fixing steps that is needed to pass the issue. It has to be the exact steps that are needed to write the patch

    - **Do Not Hallucinate**:  
    - Base conclusions only on information explicitly present in the provided inputs.  
    - Avoid adding assumptions not supported by evidence.

    ### **Output Format**  
    The enhanced bug report must be structured as **valid JSON** following this schema:

    ```json
    {{
        "Title": "<Bug title>",
        "Description": "<Improved description based on analysis>",
        "RootCause": "<Identified root cause>",
        "StepsToReproduce": ["<Step-by-step reproduction guide>"] or null,
        "ExpectedBehavior": "<Correct system behavior>",
        "ObservedBehavior": "<Actual faulty behavior>",
        "FixSteps": "<Exact steps needed to fix the issue>"
    }}
    ```

    ---

    ### **Notes**  
    - If certain fields cannot be enhanced due to insufficient context, retain the original content.  
    - Keep all responses **concise, factual, and evidence-based**.  
    - The final output must **strictly** follow the JSON format above.

    ---

    ### **Input Data Provided**

    **Original Bug Report:**
    {bug_report}

    **Agent Analysis (reasoning chain from code investigation):**
    {chat_history}

    **Source Code Methods:**
    {analyzed_methods}

    '''
    # Handle Max Token limit exceed cases
    # Format the full prompt
    full_prompt = template.format(
        bug_report=dev_written_bug_report, 
        chat_history=chat_history, 
        analyzed_methods=method_cache
    )

    # Check token count
    token_count = count_tokens(full_prompt)

    if token_count > 250000:
        print(f"Token count ({token_count}) exceeds limit! Splitting into chunks...")
        prompt_chunks = split_into_chunks(full_prompt, max_tokens=250000)

        responses = []

        for chunk in prompt_chunks:
            response = LLM.invoke(chunk)
            responses.append(response)

        return "\n".join(responses)
    else:
        # Run normally when within token limit
        prompt = PromptTemplate.from_template(template)

        chain = LLMChain(llm=LLM, prompt=prompt)
        return chain.run({'bug_report': dev_written_bug_report, 'chat_history': chat_history, 'analyzed_methods': method_cache})




# ---------- main driver ----------
def main():
    mkdirp(os.path.dirname(OUT_SUMMARY_FILE) or ".")
    mkdirp(os.path.dirname(OUT_SINGLE_ENHANCED_FILE) or ".")
    instances = load_json_safe(REPO_INSTANCES_JSON)
    registry = load_json_safe(REPO_CODEGRAPH_INDEX)
    if not instances:
        print("No instances found:", REPO_INSTANCES_JSON); return
    if not registry:
        print("No registry found:", REPO_CODEGRAPH_INDEX); return
    
    # Optional: filter instances by instance_id (for reruns)
    if INSTANCE_ID_FILTER:
        want = set(str(x) for x in INSTANCE_ID_FILTER)
        before_n = len(instances)
        instances = [inst for inst in instances if str(inst.get("instance_id")) in want]
        after_n = len(instances)
        print(f"Instance filter enabled: {after_n}/{before_n} instances selected")

    # process sequentially
    for inst in instances:
        instance_id = inst.get("instance_id")
        print(f"\n=== Handling instance {instance_id} ===")

        # find registry entry for this instance by instance_id or base_commit
        reg_entry = None
        for r in registry:
            if r.get("instance_id") == instance_id:
                reg_entry = r
                break
        if reg_entry is None:
            # fallback to commit match
            commit = inst.get("base_commit")
            for r in registry:
                if r.get("base_commit") == commit:
                    reg_entry = r
                    break

        if reg_entry is None:
            print("No registry artifact found for instance", instance_id, "skipping")
            continue

        # Merge the instance-level fields into the registry entry so tools have access to problem_statement,
        # FAIL_TO_PASS, test lists, patch text, created_at, version, etc.
        # Important: instance fields override the registry fields for the same keys.
        merged_entry = dict(reg_entry)   # shallow copy of registry info
        # copy all keys from inst (instance-level), overriding duplicates
        for k, v in inst.items():
            merged_entry[k] = v

        try:
            # pass the merged entry to the per-instance runner
            run_for_instance(inst, merged_entry, OUT_SUMMARY_FILE)
        except Exception as e:
            print("Error processing", instance_id, "->", str(e))
            # write error entry
            all_entries = load_json_safe(OUT_SUMMARY_FILE)
            err_entry = {
                "instance_id": instance_id,
                "repo": inst.get("repo"),
                "base_commit": inst.get("base_commit"),
                "error": str(e)
            }
            all_entries.append(err_entry)
            save_json_atomic(all_entries, OUT_SUMMARY_FILE)


if __name__ == "__main__":
    main()
