"""Vanilla agent baseline: no protocol, no compression, no reviewer.

Phase: 'vanilla_baseline'
Input: Original SWE-Bench instances + code graphs
Output: Enhanced bug reports from a basic tool-augmented LLM agent

This is the combined baseline that strips all three methodological
contributions from the full pipeline:
  1. No protocol-guided exploration (simple prompt, single run)
  2. No 3-level compressed analysis (uses raw Thoughts only)
  3. No reviewer agent (saves draft report directly)

It represents what a vanilla ReAct agent would produce: explore with
tools, then format findings into a structured bug report.
"""

import json
import logging
import os
from typing import Any, Dict, List

import src.state as state
from src.agents.common import (
    prepare_instance_state,
    generate_final_bug_report,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.vanilla_baseline")


# ---------------------------------------------------------------------------
# Explorer agent — single run, no checkpoints, no continuation
# ---------------------------------------------------------------------------

def run_explorer_agent(problem: str, chat_history: List[str]) -> List[Any]:
    """Run the explorer agent with a simple unstructured prompt.

    Single run, no checkpoint detection, no continuation loop.
    """
    logger.info("Running explorer agent (vanilla baseline)")

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
# Instance orchestration
# ---------------------------------------------------------------------------

def run_for_instance(instance: Dict[str, Any], reg_entry: Dict[str, Any],
                     out_summary_file: str) -> dict:
    """Orchestrate a single instance: explorer -> raw Thoughts -> report -> save."""
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

    # =====================================================================
    # Extract raw Thoughts (no 3-level compression)
    # =====================================================================
    agent_thoughts = [e for e in chat_history if e.startswith("Thought:")]
    agent_analysis = "\n\n".join(agent_thoughts) if agent_thoughts else ""
    logger.info(f"Explorer: {len(agent_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods cached, "
                f"{len(agent_analysis)} chars analysis")

    # =====================================================================
    # Generate final bug report using raw Thoughts (no compression)
    # =====================================================================
    final_report = {}
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
    # Save directly — NO reviewer agent
    # =====================================================================
    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": state.method_cache,
        "class_skeleton_cache": build_class_skeleton_cache(),
        "chat_history": chat_history,
        "agent_analysis_stats": {
            "thought_count": len(agent_thoughts),
            "char_count": len(agent_analysis),
        },
        "bug_report": final_report,
    }

    save_instance_result(instance_summary, out_summary_file)
    state.current_reg_entry = None
    return instance_summary


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg):
    """Run the vanilla baseline pipeline."""
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
