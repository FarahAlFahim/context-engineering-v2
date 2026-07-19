"""Ablation study: raw agent Thoughts instead of compressed 3-level analysis.

Phase: 'raw_traj_ablation'
Input: Enhanced reports (with chat_history) + original instances + codegraph index
Output: Reports generated using raw agent Thoughts instead of compressed_analysis

This measures the impact of the 3-level compression step by replacing it with
the raw concatenated Thought entries from the explorer agent's chat_history.
The rest of the pipeline (final report generation + reviewer) stays identical.
"""

import json
import logging
import os
from typing import Any, Dict, List

import src.state as state
from src.agents.common import (
    prepare_instance_state,
    generate_final_bug_report, parse_reviewer_output,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.raw_trajectory_ablation")


# ---------------------------------------------------------------------------
# Reviewer agent (identical to multi_agent.run_reviewer_agent but uses
# agent_analysis — raw Thoughts — instead of compressed_history)
# ---------------------------------------------------------------------------

def run_reviewer_agent(draft_report: Any, problem: str,
                       agent_analysis: str) -> dict:
    """Tool-enabled reviewer that validates and improves the draft report.

    Receives agent_analysis (concatenated raw Thoughts from the explorer)
    instead of the compressed 3-level analysis.
    """
    logger.info("Running reviewer agent with raw agent Thoughts")
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
        "\n\n=== Investigation analysis (agent reasoning trajectory) ===\n" + agent_analysis
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

def run_for_instance(enhanced_entry: Dict[str, Any],
                     original_instance: Dict[str, Any],
                     reg_entry: Dict[str, Any],
                     out_file: str) -> dict:
    """Process a single instance: extract Thoughts -> report -> reviewer."""
    instance_id = enhanced_entry.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})

    # Populate method_cache from the enhanced report
    input_method_cache = enhanced_entry.get("method_cache", {}) or {}
    if input_method_cache:
        state.method_cache.update(input_method_cache)

    # Get problem_statement from original instance
    problem = original_instance.get("problem_statement", "") or ""

    # Get chat_history from the enhanced report
    chat_history = enhanced_entry.get("chat_history", []) or []

    # =====================================================================
    # Extract raw agent Thoughts (ablation: skip compress_chat_history)
    # =====================================================================
    agent_thoughts = [entry for entry in chat_history if entry.startswith("Thought:")]
    agent_analysis = "\n\n".join(agent_thoughts) if agent_thoughts else ""
    logger.info(f"Extracted {len(agent_thoughts)} Thought entries "
                f"({len(agent_analysis)} chars) for {instance_id}")

    # =====================================================================
    # Generate final bug report using raw Thoughts
    # =====================================================================
    final_report = {}
    raw_report_for_reviewer = None
    try:
        final_bug_report = generate_final_bug_report(
            problem, agent_analysis,
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
            "repo": enhanced_entry.get("repo"),
            "base_commit": enhanced_entry.get("base_commit"),
            "error": str(e),
            "chat_history": chat_history,
            "method_cache": state.method_cache,
            "bug_report": {},
        }
        save_instance_result(error_summary, out_file)
        state.current_reg_entry = None
        return error_summary

    # =====================================================================
    # Reviewer agent — uses raw agent Thoughts instead of compressed analysis
    # =====================================================================
    reviewer_draft = final_report
    if raw_report_for_reviewer:
        reviewer_draft = {
            "_parsing_failed": True,
            "_raw_llm_output": raw_report_for_reviewer,
            "_instruction": "Extract structured report from raw text and produce revised_report JSON.",
        }

    reviewer_result = run_reviewer_agent(
        reviewer_draft, problem, agent_analysis,
    )
    final_report = reviewer_result.get("revised_report", final_report)

    # =====================================================================
    # Build and save result (no compressed_analysis field)
    # =====================================================================
    instance_summary = {
        "instance_id": instance_id,
        "repo": enhanced_entry.get("repo"),
        "base_commit": enhanced_entry.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": state.method_cache,
        "class_skeleton_cache": build_class_skeleton_cache(),
        "chat_history": chat_history,
        "agent_analysis_stats": {
            "thought_count": len(agent_thoughts),
            "char_count": len(agent_analysis),
        },
        "reviewer_changes": reviewer_result.get("changes", []),
        "reviewer_evidence": reviewer_result.get("evidence", []),
        "reviewer_history": reviewer_result.get("reviewer_history", []),
        "bug_report": final_report,
    }

    save_instance_result(instance_summary, out_file)
    state.current_reg_entry = None
    return instance_summary


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg):
    """Run the raw trajectory ablation pipeline.

    Reads enhanced reports (--repo-instances) which contain chat_history,
    and original instances (--original-instances) for problem_statement.
    """
    state.config = cfg
    state.llm = make_chat_llm(cfg.openai_model, cfg.llm_temperature, cfg.openai_api_base, cfg.openai_api_key_env)

    out_file = cfg.output_file
    if not out_file:
        logger.error("No output file specified (--output)")
        return

    mkdirp(os.path.dirname(out_file) or ".")

    # Enhanced reports (contain chat_history from explorer agent)
    enhanced_entries = load_json_safe(cfg.repo_instances_json)
    if not enhanced_entries:
        logger.error(f"No enhanced entries found: {cfg.repo_instances_json}")
        return

    # Original instances (contain problem_statement)
    original_entries = load_json_safe(cfg.original_instances_json)
    if not original_entries:
        logger.error(f"No original instances found: {cfg.original_instances_json}")
        return
    original_by_id = {
        e.get("instance_id"): e
        for e in original_entries if isinstance(e, dict)
    }

    # Codegraph registry (for reviewer agent tools)
    registry = load_json_safe(cfg.repo_codegraph_index)
    if not registry:
        logger.error(f"No registry found: {cfg.repo_codegraph_index}")
        return

    logger.info(f"Loaded {len(enhanced_entries)} enhanced entries, "
                f"{len(original_entries)} original instances, "
                f"{len(registry)} registry entries")
    logger.info(f"Output: {out_file}")

    if cfg.instance_id_filter:
        want = set(str(x) for x in cfg.instance_id_filter)
        before_n = len(enhanced_entries)
        enhanced_entries = [e for e in enhanced_entries
                           if str(e.get("instance_id")) in want]
        logger.info(f"Instance filter: {len(enhanced_entries)}/{before_n} selected")

    for entry in enhanced_entries:
        instance_id = entry.get("instance_id")
        logger.info(f"\n{'='*60}\nHandling instance {instance_id}\n{'='*60}")

        # Skip entries that had errors
        if entry.get("error"):
            logger.warning(f"Skipping {instance_id}: previous error — {entry.get('error')}")
            continue

        # Skip entries with no chat_history
        chat_history = entry.get("chat_history", [])
        if not chat_history:
            logger.warning(f"Skipping {instance_id}: no chat_history in enhanced entry")
            continue

        # Lookup original instance for problem_statement
        original_instance = original_by_id.get(instance_id)
        if not original_instance:
            logger.warning(f"No original instance for {instance_id}, skipping")
            continue

        # Lookup codegraph registry entry
        reg_entry = None
        for r in registry:
            if r.get("instance_id") == instance_id:
                reg_entry = r
                break
        if reg_entry is None:
            commit = entry.get("base_commit")
            for r in registry:
                if r.get("base_commit") == commit:
                    reg_entry = r
                    break

        if reg_entry is None:
            logger.warning(f"No registry artifact for {instance_id}, skipping")
            continue

        merged_entry = dict(reg_entry)
        for k, v in entry.items():
            merged_entry[k] = v

        try:
            run_for_instance(entry, original_instance, merged_entry, out_file)
        except Exception as e:
            logger.error(f"Error processing {instance_id}: {e}")
            error_entry = {
                "instance_id": instance_id,
                "repo": entry.get("repo"),
                "base_commit": entry.get("base_commit"),
                "error": str(e),
            }
            all_entries = load_json_safe(out_file)
            all_entries.append(error_entry)
            save_json_atomic(all_entries, out_file)
