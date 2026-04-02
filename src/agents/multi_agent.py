"""Multi-agent pipeline: exploration agent + reviewer agent.

Phase: 'enhance'
Input: Original SWE-Bench instances + code graphs
Output: Enhanced bug reports with root cause analysis and fix suggestions
"""

import json
import logging
from typing import Any, Dict, List

import src.state as state
from src.agents.common import (
    LANGGRAPH_AVAILABLE,
    prepare_instance_state, filter_chat_history_for_method_cache,
    generate_final_bug_report, parse_reviewer_output,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.multi_agent")


def run_reviewer_agent(draft_report: Any, problem: str,
                        method_cache: Dict[str, str],
                        chat_history: List[str]) -> dict:
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

    # Build reviewer context
    filtered_history = filter_chat_history_for_method_cache(chat_history)
    method_cache_text_parts = [f"--- {nid} ---\n{code}" for nid, code in method_cache.items()]
    method_cache_text = "\n\n".join(method_cache_text_parts) if method_cache_text_parts else "(empty)"

    user_text = (
        "Original bug report:\n" + (problem or "") +
        "\n\n=== Draft report JSON ===\n" + draft_json +
        "\n\n=== Method cache (full source code of fetched methods) ===\n" + method_cache_text +
        "\n\n=== Full chat history from previous agent investigation ===\n"
        + "\n".join(filtered_history)
    )

    reviewer_tools = build_tools(for_reviewer=True)
    agent_events = run_agent_with_tools(
        instruction, user_text, reviewer_tools, reviewer_history, recursion_limit=60
    )

    state.active_chat_history = None
    state.trace_include_observations = prev_trace_obs

    return parse_reviewer_output(reviewer_history, agent_events, draft_report)


def run_for_instance(instance: Dict[str, Any], reg_entry: Dict[str, Any],
                      out_summary_file: str, single_enhanced_file: str = "") -> dict:
    """Orchestrate a single instance: exploration agent -> reviewer."""
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})
    problem = instance.get("problem_statement", "") or ""

    # --- Run exploration agent ---
    chat_history: List[str] = []
    state.active_chat_history = chat_history
    state.trace_include_observations = True

    instruction = load_prompt("agent_instruction_multi_agent.txt", state.config.prompts_dir)
    user_text = f"Problem: {problem}\n"

    tools = build_tools(for_reviewer=False)
    agent_events = run_agent_with_tools(instruction, user_text, tools, chat_history)

    state.active_chat_history = None

    # Extract agent analysis (all Thought entries)
    agent_thoughts = [e for e in chat_history if e.startswith("Thought:")]
    agent_analysis = "\n\n".join(agent_thoughts) if agent_thoughts else ""
    logger.info(f"Agent analysis: {len(agent_thoughts)} thoughts, {len(agent_analysis)} chars")

    # --- Phase A: Generate bug report ---
    final_report = {}
    raw_report_for_reviewer = None
    try:
        final_bug_report = generate_final_bug_report(
            state.method_cache, problem, agent_analysis,
            prompt_name="final_report_multi_agent.txt"
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

    # --- Save single-agent output (before reviewer) ---
    if single_enhanced_file:
        single_summary = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "classification_stats": state.classification_stats.get(instance_id, {}),
            "method_cache": state.method_cache,
            "class_skeleton_cache": {},
            "chat_history": chat_history,
            "bug_report": final_report,
        }
        save_instance_result(single_summary, single_enhanced_file)
        logger.info(f"Saved single-agent output for {instance_id}")

    # --- Phase B: Reviewer agent ---
    reviewer_draft = final_report
    if raw_report_for_reviewer:
        reviewer_draft = {
            "_parsing_failed": True,
            "_raw_llm_output": raw_report_for_reviewer,
            "_instruction": "Extract structured report from raw text and produce revised_report JSON.",
        }

    reviewer_result = run_reviewer_agent(
        reviewer_draft, problem, state.method_cache, chat_history
    )
    final_report = reviewer_result.get("revised_report", final_report)

    # --- Build and save result ---
    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": state.method_cache,
        "class_skeleton_cache": build_class_skeleton_cache(),
        "chat_history": chat_history,
        "reviewer_changes": reviewer_result.get("changes", []),
        "reviewer_evidence": reviewer_result.get("evidence", []),
        "reviewer_history": reviewer_result.get("reviewer_history", []),
        "bug_report": final_report,
    }

    save_instance_result(instance_summary, out_summary_file)
    state.current_reg_entry = None
    return instance_summary


def run_pipeline(cfg):
    """Run the multi-agent enhancement pipeline."""
    state.config = cfg
    state.llm = make_chat_llm(cfg.openai_model, cfg.llm_temperature)

    mkdirp(cfg.output_dir)
    out_file = cfg.output_file
    if not out_file:
        logger.error("No output file specified (--output)")
        return

    mkdirp(os.path.dirname(out_file) or ".")

    # Auto-derive single-enhanced (before reviewer) output path if not set
    single_enhanced_file = cfg.single_enhanced_file
    if not single_enhanced_file:
        enhanced_dir = os.path.join(cfg.output_dir, "enhanced")
        single_enhanced_file = os.path.join(enhanced_dir, os.path.basename(out_file))
    mkdirp(os.path.dirname(single_enhanced_file) or ".")

    logger.info(f"Output (before reviewer): {single_enhanced_file}")
    logger.info(f"Output (after reviewer):  {out_file}")

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
            run_for_instance(inst, merged_entry, out_file, single_enhanced_file)
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


import os  # noqa: E402 (needed for os.path in run_pipeline)
