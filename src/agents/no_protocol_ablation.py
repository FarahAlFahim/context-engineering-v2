"""Ablation study: no protocol-guided exploration.

Phase: 'no_protocol_ablation'
Input: Original SWE-Bench instances + code graphs
Output: Enhanced bug reports produced WITHOUT protocol-guided checkpoints

This measures the impact of the 5-checkpoint protocol by replacing it with
a simple unstructured exploration prompt. The explorer agent runs once
(no continuation loop), then the rest of the pipeline (compression, final
report generation, reviewer) stays identical to multi_agent.py.
"""

import json
import logging
import os
from typing import Any, Dict, List

import src.state as state
from src.agents.common import (
    prepare_instance_state,
    compress_chat_history, generate_final_bug_report, parse_reviewer_output,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.no_protocol_ablation")


# ---------------------------------------------------------------------------
# Explorer agent — single run, no checkpoints, no continuation
# ---------------------------------------------------------------------------

def run_explorer_agent(problem: str, chat_history: List[str]) -> List[Any]:
    """Run the explorer agent with a simple unstructured prompt.

    Unlike the protocol-guided version, this runs ONCE with no checkpoint
    detection and no continuation loop.
    """
    logger.info("Running explorer agent (no-protocol ablation)")

    state.active_chat_history = chat_history
    state.trace_include_observations = True

    instruction = load_prompt("agent_instruction_no_protocol.txt", state.config.prompts_dir)
    user_text = f"Problem: {problem}\n"

    tools = build_tools(for_reviewer=False)

    agent_events = run_agent_with_tools(
        instruction, user_text, tools, chat_history,
        recursion_limit=100, max_retries=2,
    )

    state.active_chat_history = None
    return agent_events


# ---------------------------------------------------------------------------
# Reviewer agent (identical to multi_agent.run_reviewer_agent)
# ---------------------------------------------------------------------------

def run_reviewer_agent(draft_report: Any, problem: str,
                       compressed_history: str) -> dict:
    """Tool-enabled reviewer that validates and improves the draft report."""
    logger.info("Running reviewer agent to validate and improve the draft report")
    reviewer_history: List[str] = []

    state.active_chat_history = reviewer_history
    prev_trace_obs = state.trace_include_observations
    state.trace_include_observations = True

    draft_json = draft_report
    if not isinstance(draft_json, str):
        try:
            draft_json = json.dumps(draft_report, ensure_ascii=False)
        except Exception:
            draft_json = str(draft_report)

    instruction = load_prompt("reviewer_multi_agent.txt", state.config.prompts_dir)

    user_text = (
        "Original bug report:\n" + (problem or "") +
        "\n\n=== Draft report JSON ===\n" + draft_json +
        "\n\n=== Investigation analysis (compressed) ===\n" + compressed_history
    )

    reviewer_tools = build_tools(for_reviewer=True)
    agent_events = run_agent_with_tools(
        instruction, user_text, reviewer_tools, reviewer_history, recursion_limit=60
    )

    state.active_chat_history = None
    state.trace_include_observations = prev_trace_obs

    return parse_reviewer_output(reviewer_history, agent_events, draft_report)


# ---------------------------------------------------------------------------
# Instance orchestration
# ---------------------------------------------------------------------------

def run_for_instance(instance: Dict[str, Any], reg_entry: Dict[str, Any],
                     out_summary_file: str) -> dict:
    """Orchestrate a single instance: explorer -> compress -> report -> reviewer."""
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})
    problem = instance.get("problem_statement", "") or ""

    # =====================================================================
    # Explorer agent — single run, no protocol, no continuations
    # =====================================================================
    chat_history: List[str] = []
    run_explorer_agent(problem, chat_history)

    agent_thoughts = [e for e in chat_history if e.startswith("Thought:")]
    logger.info(f"Explorer: {len(agent_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods cached")

    # =====================================================================
    # Compress chat history (same as multi_agent.py)
    # =====================================================================
    compressed_analysis = compress_chat_history(chat_history, problem)
    logger.info("Compressed chat history into 3-level memory for report generation")

    # =====================================================================
    # Generate final bug report
    # =====================================================================
    final_report = {}
    raw_report_for_reviewer = None
    try:
        final_bug_report = generate_final_bug_report(
            problem, compressed_analysis,
            prompt_name="final_report.txt",
        )
        parsed_report = parse_json_best_effort(
            final_bug_report, preferred_keys=["revised_report", "title", "bug_location"]
        )
        if parsed_report:
            final_report = parsed_report
        else:
            logger.warning(f"Could not parse final bug report JSON for {instance_id}")
            raw_report_for_reviewer = final_bug_report
            final_report = {"_raw_unparsed_report": final_bug_report}
    except Exception as e:
        logger.error(f"generate_final_bug_report failed for {instance_id}: {e}")
        error_summary = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "error": str(e),
            "chat_history": chat_history,
            "method_cache": state.method_cache,
            "bug_report": {},
        }
        save_instance_result(error_summary, out_summary_file)
        state.current_reg_entry = None
        return error_summary

    # =====================================================================
    # Reviewer agent
    # =====================================================================
    reviewer_draft = final_report
    if raw_report_for_reviewer:
        reviewer_draft = {
            "_parsing_failed": True,
            "_raw_llm_output": raw_report_for_reviewer,
            "_instruction": "Extract structured report from raw text and produce revised_report JSON.",
        }

    reviewer_result = run_reviewer_agent(
        reviewer_draft, problem, compressed_analysis,
    )
    final_report = reviewer_result.get("revised_report", final_report)

    # =====================================================================
    # Build and save result (only final output, no pre-reviewer save)
    # =====================================================================
    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": state.method_cache,
        "class_skeleton_cache": build_class_skeleton_cache(),
        "chat_history": chat_history,
        "compressed_analysis": compressed_analysis,
        "reviewer_changes": reviewer_result.get("changes", []),
        "reviewer_evidence": reviewer_result.get("evidence", []),
        "reviewer_history": reviewer_result.get("reviewer_history", []),
        "bug_report": final_report,
    }

    save_instance_result(instance_summary, out_summary_file)
    state.current_reg_entry = None
    return instance_summary


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg):
    """Run the no-protocol ablation pipeline."""
    state.config = cfg
    state.llm = make_chat_llm(cfg.openai_model, cfg.llm_temperature, cfg.openai_api_base, cfg.openai_api_key_env)

    mkdirp(cfg.output_dir)
    out_file = cfg.output_file
    if not out_file:
        logger.error("No output file specified (--output)")
        return

    mkdirp(os.path.dirname(out_file) or ".")
    logger.info(f"Output: {out_file}")

    instances = load_json_safe(cfg.repo_instances_json)
    registry = load_json_safe(cfg.repo_codegraph_index)
    if not instances:
        logger.error(f"No instances found: {cfg.repo_instances_json}")
        return
    if not registry:
        logger.error(f"No registry found: {cfg.repo_codegraph_index}")
        return

    if cfg.instance_id_filter:
        want = set(str(x) for x in cfg.instance_id_filter)
        before_n = len(instances)
        instances = [i for i in instances if str(i.get("instance_id")) in want]
        logger.info(f"Instance filter: {len(instances)}/{before_n} selected")

    for inst in instances:
        instance_id = inst.get("instance_id")
        logger.info(f"\n{'='*60}\nHandling instance {instance_id}\n{'='*60}")

        reg_entry = None
        for r in registry:
            if r.get("instance_id") == instance_id:
                reg_entry = r
                break
        if reg_entry is None:
            commit = inst.get("base_commit")
            for r in registry:
                if r.get("base_commit") == commit:
                    reg_entry = r
                    break

        if reg_entry is None:
            logger.warning(f"No registry artifact for {instance_id}, skipping")
            continue

        merged_entry = dict(reg_entry)
        for k, v in inst.items():
            merged_entry[k] = v

        try:
            run_for_instance(inst, merged_entry, out_file)
        except Exception as e:
            logger.error(f"Error processing {instance_id}: {e}")
            error_entry = {
                "instance_id": instance_id,
                "repo": inst.get("repo"),
                "base_commit": inst.get("base_commit"),
                "error": str(e),
            }
            all_entries = load_json_safe(out_file)
            all_entries.append(error_entry)
            save_json_atomic(all_entries, out_file)
