"""Common agent utilities shared across all pipelines."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import src.state as state
from src.utils.io import load_json_safe, save_json_atomic, mkdirp, run_cmd
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.tokens import count_tokens, split_into_chunks
from src.utils.json_parser import parse_json_best_effort
from src.graph.loader import load_codegraph, build_edge_adjacency
from src.utils.embeddings import build_embedding_index
from src.utils.bm25 import bm25_prepare_candidates

# LangGraph support (optional)
try:
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger("context_engineering.agents.common")


def prepare_instance_state(reg_entry: dict, build_embeddings: bool = False):
    """Load codegraph, build indices, and set state for a single instance."""
    state.reset_graph_state()

    cg_path = reg_entry.get("code_graph_path", "")
    if not cg_path:
        logger.warning("No code_graph_path in registry entry")
        return

    codegraph = load_codegraph(cg_path)
    if not codegraph:
        logger.warning(f"Failed to load codegraph from {cg_path}")
        return

    nodes = codegraph.get("nodes", [])
    edges = codegraph.get("edges", [])

    state.nodes_by_id = {n["id"]: n for n in nodes if "id" in n}
    state.edges_list = edges
    state.adjacency = build_edge_adjacency(edges)

    logger.info(f"Loaded graph: {len(state.nodes_by_id)} nodes, {len(edges)} edges")

    if build_embeddings:
        repo_slug = reg_entry.get("repo", "").replace("/", "__")
        commit = reg_entry.get("base_commit", "")[:12]

        if state.config.use_bm25_ranking:
            state.bm25_candidate_ids, state.bm25_docs_texts, state.bm25_index = \
                bm25_prepare_candidates(
                    state.nodes_by_id,
                    exclude_tests_flag=state.config.exclude_tests,
                    exclude_dirs=state.config.exclude_dirs,
                )
        else:
            state.faiss_index, state.embedding_node_ids, state.embedder = \
                build_embedding_index(
                    state.nodes_by_id,
                    embed_model=state.config.embed_model,
                    embeddings_cache_dir=state.config.embeddings_cache_dir,
                    repo_slug=repo_slug,
                    commit=commit,
                    exclude_tests=state.config.exclude_tests,
                    exclude_dirs=state.config.exclude_dirs,
                )


def append_langgraph_ai_messages_only(chat_history: List[str], chunk: Any):
    """Append AI reasoning messages from a LangGraph stream chunk."""
    if not isinstance(chunk, dict):
        return

    for key in ("agent", "tools", "action"):
        inner = chunk.get(key)
        if not isinstance(inner, dict):
            continue
        for msg in inner.get("messages", []):
            content = getattr(msg, "content", "")
            role = getattr(msg, "type", "")
            if role == "ai" and content and content.strip():
                chat_history.append(f"Thought: {content.strip()}")
            elif role == "tool" and content:
                # Tool results are already captured by tracing
                pass


def filter_chat_history_for_method_cache(chat_history: List[str]) -> List[str]:
    """Remove get_code Observation entries (code is in method_cache)."""
    filtered = []
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


def read_file_at_commit(file_path: str, base_commit: str,
                         repo_local_path: str) -> Optional[str]:
    """Read file content from git repo at a specific commit."""
    if not base_commit or not file_path:
        return None
    rc, out, err = run_cmd(
        ["git", "show", f"{base_commit}:{file_path}"],
        cwd=repo_local_path, timeout=30
    )
    return out if rc == 0 else None


def generate_final_bug_report(method_cache: dict, bug_report: str,
                               chat_history: str,
                               prompt_name: str = "final_report_multi_agent.txt") -> str:
    """Generate the final enhanced bug report using LLM."""
    template = load_prompt(prompt_name, state.config.prompts_dir)

    analyzed_methods = "\n\n".join(
        f"--- {nid} ---\n{code}" for nid, code in method_cache.items()
    ) if method_cache else "(none)"

    full_prompt = template.format(
        bug_report=bug_report,
        chat_history=chat_history,
        analyzed_methods=analyzed_methods,
    )

    token_count = count_tokens(full_prompt)
    logger.info(f"Final report prompt: {token_count} tokens")

    if token_count > 250000:
        logger.warning(f"Token count ({token_count}) exceeds limit, splitting into chunks")
        chunks = split_into_chunks(full_prompt, max_tokens=250000)
        responses = []
        for chunk in chunks:
            response = state.llm.invoke(chunk)
            responses.append(str(response.content if hasattr(response, 'content') else response))
        return "\n".join(responses)

    prompt = PromptTemplate.from_template(template)
    chain = prompt | state.llm | StrOutputParser()
    return chain.invoke({
        "bug_report": bug_report,
        "chat_history": chat_history,
        "analyzed_methods": analyzed_methods,
    })


def parse_reviewer_output(reviewer_history: List[str], agent_events: List[Any],
                           draft_report: Any) -> dict:
    """Parse reviewer agent output to extract revised_report."""
    _TOOL_OBS_KEYS = {"node", "pattern", "error"}
    _REPORT_KEYS = {"Title", "Description", "RootCause", "Suggestions", "problem_location",
                     "revised_report", "changes", "evidence"}

    def _is_tool_observation(obj: dict) -> bool:
        if obj.keys() & _TOOL_OBS_KEYS:
            return True
        if "file" in obj and ("content" in obj or "start_line" in obj):
            return True
        return False

    parsed = None
    fallback_parsed = None

    for line in reversed(reviewer_history):
        candidate = parse_json_best_effort(line, preferred_keys=["revised_report"])
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
        for ev in reversed(agent_events):
            if isinstance(ev, dict):
                for key in ("agent", "tools", "action"):
                    inner = ev.get(key)
                    if isinstance(inner, dict):
                        for m in inner.get("messages", []):
                            content = getattr(m, "content", "")
                            if content:
                                candidate = parse_json_best_effort(
                                    str(content), preferred_keys=["revised_report"])
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

    # Clean up nested changes/evidence
    _VALID_KEYS = {
        "Title", "Description", "RootCause", "StepsToReproduce",
        "ExpectedBehavior", "ObservedBehavior", "Suggestions",
        "problem_location", "possible_fix", "possible_fix_code", "FixSteps",
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
            if extra_key not in _VALID_KEYS:
                revised_report.pop(extra_key)

    return {
        "revised_report": revised_report,
        "changes": changes,
        "evidence": evidence,
        "reviewer_history": reviewer_history,
    }


def build_class_skeleton_cache() -> dict:
    """Build class skeletons from classes in method_cache_global."""
    out = {}
    for nid in list(state.method_cache_global):
        nd = state.nodes_by_id.get(nid)
        if not nd or nd.get("type") != "class":
            continue
        code = nd.get("code", "") or ""
        skeleton = []
        for ln in code.splitlines()[:120]:
            s = ln.strip()
            if s.startswith("def ") or s.startswith("class ") or s.startswith("async def "):
                skeleton.append(ln)
            if len(skeleton) > 80:
                break
        out[nid] = "\n".join(skeleton)
    return out


def save_instance_result(instance_summary: dict, out_file: str):
    """Upsert instance result into the output JSON file."""
    instance_id = instance_summary.get("instance_id")
    all_entries = load_json_safe(out_file)
    replaced = False
    for i, e in enumerate(all_entries):
        if e.get("instance_id") == instance_id:
            all_entries[i] = instance_summary
            replaced = True
            break
    if not replaced:
        all_entries.append(instance_summary)
    save_json_atomic(all_entries, out_file)
    logger.info(f"Saved instance {instance_id} to {out_file}")


def run_agent_with_tools(instruction: str, user_text: str, tools: list,
                          chat_history: List[str],
                          recursion_limit: int = 100) -> List[Any]:
    """Run an agent (LangGraph or LangChain fallback) and return events."""
    agent_events = []
    agent_llm = make_chat_llm(state.config.openai_model, state.config.llm_temperature)

    if LANGGRAPH_AVAILABLE:
        chat_history.append("[agent] Using LangGraph tool-calling agent")
        lg_agent = lg_create_react_agent(agent_llm, tools)
        inputs = {"messages": [SystemMessage(content=instruction), HumanMessage(content=user_text)]}
        try:
            for chunk in lg_agent.stream(inputs, config={"recursion_limit": recursion_limit}):
                agent_events.append(chunk)
                append_langgraph_ai_messages_only(chat_history, chunk)
        except Exception as e:
            agent_events = [{"output": f"LangGraph agent runtime error: {str(e)}"}]
            chat_history.append(agent_events[0]["output"])
    else:
        from langchain_classic.agents import initialize_agent, AgentType
        chat_history.append("[agent] Using LangChain legacy ReAct agent (fallback)")
        agent = initialize_agent(
            tools, agent_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=state.config.agent_max_iterations,
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

    return agent_events
