"""Trajectory insights pipeline: enhance reports using external trajectory summaries.

Phase: 'trajectory_enhance'
Input: Multi-agent enhanced reports + trajectory summary file
Output: Further enhanced reports integrating verified trajectory insights
"""

import json
import logging
import os
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import src.state as state
from src.agents.common import (
    prepare_instance_state, parse_reviewer_output,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.trajectory_insights")


def reviewer_agent(enhanced_report: Any, problem: str,
                    trajectory_summary: str) -> dict:
    """Reviewer that selectively integrates trajectory insights."""
    logger.info("Running trajectory reviewer agent")
    reviewer_history: List[str] = []

    state.active_chat_history = reviewer_history
    prev_trace_obs = state.trace_include_observations
    state.trace_include_observations = True

    draft_json = enhanced_report
    if not isinstance(draft_json, str):
        try:
            draft_json = json.dumps(enhanced_report, ensure_ascii=False)
        except Exception:
            draft_json = str(enhanced_report)

    instruction = load_prompt("reviewer_trajectory.txt", state.config.prompts_dir)

    user_text = (
        "Original bug report (problem statement):\n" + (problem or "(not available)") +
        "\n\n=== Existing enhanced bug report (already reviewed — high quality baseline) ===\n"
        + draft_json +
        "\n\n=== Trajectory summary (additional agent insights to evaluate) ===\n"
        + trajectory_summary
    )

    reviewer_tools = build_tools(for_reviewer=True)
    agent_events = run_agent_with_tools(
        instruction, user_text, reviewer_tools, reviewer_history, recursion_limit=60
    )

    state.active_chat_history = None
    state.trace_include_observations = prev_trace_obs

    return parse_reviewer_output(reviewer_history, agent_events, enhanced_report)


def run_for_instance(instance: Dict[str, Any], reg_entry: Dict[str, Any],
                      out_summary_file: str) -> dict:
    """Orchestrate trajectory-based enhancement for a single instance."""
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})

    chat_history: List[str] = []
    state.active_chat_history = chat_history

    # Load bug_report from instance
    problem_obj = instance.get("bug_report")
    input_method_cache = instance.get("method_cache", {}) or {}
    input_class_skeleton_cache = instance.get("class_skeleton_cache", {}) or {}

    if not problem_obj:
        problem_obj = instance.get("problem_statement", "")

    problem = (
        json.dumps(problem_obj, ensure_ascii=False, indent=2)
        if isinstance(problem_obj, (dict, list))
        else (problem_obj or "")
    )

    if not problem:
        problem = instance.get("problem_statement", "") or ""

    # Load trajectory summary
    traj_entries = load_json_safe(state.config.trajectory_summary_file)
    traj_by_id = {
        (e.get("instance_id") or e.get("instance")): e
        for e in traj_entries if isinstance(e, dict)
    }
    trajectory_summary_obj = (traj_by_id.get(instance_id) or {}).get("trajectory_summary")

    # Check trajectory label
    traj_label = None
    if isinstance(trajectory_summary_obj, dict):
        traj_label = (
            (trajectory_summary_obj.get("trajectory_evaluation") or {}).get("label") or ""
        ).strip().lower()

    # Extract usable trajectory payload
    trajectory_summary_payload = trajectory_summary_obj
    if isinstance(trajectory_summary_obj, dict) and "trajectory_summary" in trajectory_summary_obj:
        trajectory_summary_payload = trajectory_summary_obj.get("trajectory_summary")
    trajectory_summary = (
        json.dumps(trajectory_summary_payload, ensure_ascii=False, indent=2)
        if isinstance(trajectory_summary_payload, (dict, list))
        else (trajectory_summary_payload or "")
    )

    # CASE 1: Skip non-transferable/misleading trajectories
    if traj_label in {"not_transferable", "misleading"}:
        logger.info(f"Skipping {instance_id}: trajectory label={traj_label}")
        chat_history.append(f"[skip] trajectory label={traj_label}")
        summary = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "method_cache": input_method_cache,
            "class_skeleton_cache": input_class_skeleton_cache,
            "chat_history": chat_history,
            "further_enhanced": False,
            "bug_report": problem_obj if isinstance(problem_obj, (dict, list)) else {"raw": problem},
        }
        state.active_chat_history = None
        save_instance_result(summary, out_summary_file)
        state.current_reg_entry = None
        return summary

    # CASE 2: Check redundancy for transferable trajectories
    further_enhancement_reason = ""
    try:
        if problem and trajectory_summary:
            compare_template = load_prompt("redundancy_comparison.txt", state.config.prompts_dir)
            compare_prompt = PromptTemplate.from_template(compare_template)
            compare_chain = LLMChain(llm=state.llm, prompt=compare_prompt)
            cmp_raw = compare_chain.run({"problem": problem, "traj": trajectory_summary})
            cmp_json_str = cmp_raw.replace("```json\n", "").replace("\n```", "").strip()
            cmp = json.loads(cmp_json_str)
            further_enhancement_reason = str(cmp.get("reason", "") or "")
            logger.info(f"Redundancy check: similar={cmp.get('similar')}")

            if bool(cmp.get("similar")):
                logger.info(f"Skipping {instance_id}: trajectory is redundant")
                chat_history.append(f"[skip] redundant: {cmp.get('reason','')}")
                summary = {
                    "instance_id": instance_id,
                    "repo": instance.get("repo"),
                    "base_commit": instance.get("base_commit"),
                    "method_cache": input_method_cache,
                    "class_skeleton_cache": input_class_skeleton_cache,
                    "chat_history": chat_history,
                    "further_enhanced": False,
                    "bug_report": problem_obj if isinstance(problem_obj, (dict, list)) else {"raw": problem},
                }
                state.active_chat_history = None
                save_instance_result(summary, out_summary_file)
                state.current_reg_entry = None
                return summary
    except Exception:
        pass

    # CASE 3: Further enhancement via trajectory reviewer
    logger.info(f"Enhancing {instance_id} with trajectory insights")
    if input_method_cache:
        state.method_cache.update(input_method_cache)

    # Load original problem statement
    original_entries = load_json_safe(state.config.original_instances_json)
    original_by_id = {
        (e.get("instance_id") or e.get("instance")): e
        for e in original_entries if isinstance(e, dict)
    }
    original_problem = (original_by_id.get(instance_id) or {}).get("problem_statement", "")

    reviewer_result = reviewer_agent(
        enhanced_report=problem_obj,
        problem=original_problem,
        trajectory_summary=trajectory_summary,
    )
    final_report = reviewer_result.get("revised_report", problem_obj)
    chat_history.extend(reviewer_result.get("reviewer_history", []))

    state.active_chat_history = None

    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": state.method_cache,
        "class_skeleton_cache": input_class_skeleton_cache,
        "chat_history": chat_history,
        "further_enhancement_reason": further_enhancement_reason,
        "further_enhanced": True,
        "reviewer_changes": reviewer_result.get("changes", []),
        "reviewer_evidence": reviewer_result.get("evidence", []),
        "bug_report": final_report,
    }

    # Update skeletons
    for nid in list(state.method_cache_global):
        nd = state.nodes_by_id.get(nid)
        if nd and nd.get("type") == "class":
            code = nd.get("code", "")
            skeleton = [ln for ln in code.splitlines()[:120]
                        if ln.strip().startswith(("def ", "class ", "async def "))]
            instance_summary["class_skeleton_cache"][nid] = "\n".join(skeleton[:80])

    save_instance_result(instance_summary, out_summary_file)
    state.current_reg_entry = None
    return instance_summary


def run_pipeline(cfg):
    """Run the trajectory insights enhancement pipeline."""
    state.config = cfg
    state.llm = make_chat_llm(cfg.openai_model, cfg.llm_temperature)

    out_file = cfg.output_file
    if not out_file:
        logger.error("No output file specified (--output)")
        return

    mkdirp(os.path.dirname(out_file) or ".")

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
