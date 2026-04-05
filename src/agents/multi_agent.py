"""Multi-agent pipeline: three-phase exploration + reviewer.

Phase: 'enhance'
Input: Original SWE-Bench instances + code graphs
Output: Enhanced bug reports with root cause analysis and fix suggestions

Pipeline structure:
  1. Explorer agent — finds the primary bug and proposes a fix direction
  2. Architecture mapping agent — systematically gathers all related code
  3. Impact simulation (LLM call) — simulates the fix and identifies gaps
     └─ Gap-filling loop: if gaps found, explore them, re-simulate (max N iterations)
  4. Final report generation (LLM call) — produces structured bug report
  5. Reviewer agent — validates and refines the report
"""

import json
import logging
import os
from typing import Any, Dict, List

import src.state as state
from src.agents.common import (
    LANGGRAPH_AVAILABLE,
    prepare_instance_state, filter_chat_history_for_method_cache,
    generate_final_bug_report, parse_reviewer_output,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools, run_impact_simulation,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.multi_agent")

# Maximum iterations for the gap-filling loop
MAX_GAP_FILL_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Phase 1: Explorer agent
# ---------------------------------------------------------------------------

def run_explorer_agent(problem: str, chat_history: List[str]) -> List[Any]:
    """Run the explorer agent to find the primary bug."""
    logger.info("Phase 1: Running explorer agent")

    state.active_chat_history = chat_history
    state.trace_include_observations = True

    instruction = load_prompt("agent_instruction_multi_agent.txt", state.config.prompts_dir)
    user_text = f"Problem: {problem}\n"

    tools = build_tools(for_reviewer=False)
    agent_events = run_agent_with_tools(instruction, user_text, tools, chat_history)

    state.active_chat_history = None
    return agent_events


# ---------------------------------------------------------------------------
# Phase 2: Architecture mapping agent
# ---------------------------------------------------------------------------

def run_architecture_mapping(problem: str, explorer_analysis: str,
                              method_cache: Dict[str, str],
                              chat_history: List[str]) -> str:
    """Run architecture mapping agent to gather all related code."""
    logger.info("Phase 2: Running architecture mapping agent")
    mapping_history: List[str] = []

    state.active_chat_history = mapping_history
    state.trace_include_observations = True

    instruction = load_prompt("architecture_mapping.txt", state.config.prompts_dir)

    # Build context from explorer's work
    method_cache_text = "\n\n".join(
        f"--- {nid} ---\n{code}" for nid, code in method_cache.items()
    ) if method_cache else "(empty)"

    user_text = (
        "=== Original Bug Report ===\n" + problem +
        "\n\n=== Explorer's Analysis ===\n" + explorer_analysis +
        "\n\n=== Method Cache (code already retrieved) ===\n" + method_cache_text
    )

    tools = build_tools(for_reviewer=True)  # Same tools as reviewer (no classify/rank)
    agent_events = run_agent_with_tools(
        instruction, user_text, tools, mapping_history, recursion_limit=80
    )

    state.active_chat_history = None

    # Extract the architecture map from the agent's output
    # The map is the agent's full reasoning + any structured summary it produced
    map_thoughts = [e for e in mapping_history if e.startswith("Thought:")]
    map_summary = "\n\n".join(map_thoughts) if map_thoughts else ""

    # Also include all observations (tool results) since they contain the actual code
    map_observations = [e for e in mapping_history if e.startswith("Observation:")]
    full_map = map_summary + "\n\n=== Tool Results ===\n" + "\n\n".join(map_observations)

    # Append mapping history to main chat history for record-keeping
    chat_history.append("\n--- Architecture Mapping Phase ---")
    chat_history.extend(mapping_history)

    logger.info(f"Architecture map: {len(map_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods in cache")

    return full_map


# ---------------------------------------------------------------------------
# Phase 3: Impact simulation + gap-filling loop
# ---------------------------------------------------------------------------

def run_gap_filling_agent(gaps: List[dict], problem: str,
                           explorer_analysis: str,
                           method_cache: Dict[str, str],
                           chat_history: List[str]) -> str:
    """Run a tool-calling agent to explore specific gaps identified by impact simulation."""
    logger.info(f"Gap-filling: exploring {len(gaps)} gaps")
    gap_history: List[str] = []

    state.active_chat_history = gap_history
    state.trace_include_observations = True

    instruction = load_prompt("architecture_mapping.txt", state.config.prompts_dir)

    # Build a focused prompt listing the specific gaps to investigate
    gaps_text = "\n".join(
        f"- {g.get('location', 'unknown')}: {g.get('issue', '')} → {g.get('needed_change', '')}"
        for g in gaps
    )

    method_cache_text = "\n\n".join(
        f"--- {nid} ---\n{code}" for nid, code in method_cache.items()
    ) if method_cache else "(empty)"

    user_text = (
        "=== Original Bug Report ===\n" + problem +
        "\n\n=== Previous Analysis ===\n" + explorer_analysis +
        "\n\n=== GAPS TO INVESTIGATE ===\n"
        "The impact simulation identified these specific gaps — locations where code "
        "still assumes the old behavior and would break under the proposed fix. "
        "Your job is to retrieve and analyze the code at each of these locations.\n\n"
        + gaps_text +
        "\n\n=== Method Cache (code already retrieved) ===\n" + method_cache_text
    )

    tools = build_tools(for_reviewer=True)
    agent_events = run_agent_with_tools(
        instruction, user_text, tools, gap_history, recursion_limit=60
    )

    state.active_chat_history = None

    # Extract findings
    gap_thoughts = [e for e in gap_history if e.startswith("Thought:")]
    gap_observations = [e for e in gap_history if e.startswith("Observation:")]
    full_findings = "\n\n".join(gap_thoughts)
    full_findings += "\n\n=== Tool Results ===\n" + "\n\n".join(gap_observations)

    # Append to main chat history
    chat_history.append("\n--- Gap-Filling Phase ---")
    chat_history.extend(gap_history)

    logger.info(f"Gap-filling: {len(gap_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods in cache")

    return full_findings


def run_simulation_and_gap_filling(problem: str, explorer_analysis: str,
                                    architecture_map: str,
                                    chat_history: List[str]) -> dict:
    """Run impact simulation, then iteratively fill gaps until none remain."""
    all_simulation_results = []

    for iteration in range(MAX_GAP_FILL_ITERATIONS + 1):
        # Run impact simulation
        logger.info(f"Impact simulation iteration {iteration + 1}")
        print(f"\n===== IMPACT SIMULATION (iteration {iteration + 1}) =====")

        sim_result = run_impact_simulation(
            problem=problem,
            explorer_analysis=explorer_analysis,
            architecture_map=architecture_map,
            method_cache=state.method_cache,
        )
        all_simulation_results.append(sim_result)

        gaps = sim_result.get("gaps", [])
        assumption = sim_result.get("assumption_changed", "")
        proposed_fix = sim_result.get("proposed_fix", [])

        logger.info(f"Simulation result: {len(proposed_fix)} fix locations, {len(gaps)} gaps")
        print(f"  Assumption changed: {assumption}")
        print(f"  Proposed fix locations: {len(proposed_fix)}")
        print(f"  Gaps found: {len(gaps)}")

        if not gaps:
            logger.info("No gaps found — fix is complete")
            print("  → No gaps remaining, fix is complete")
            print("===== /IMPACT SIMULATION =====\n")
            break

        if iteration >= MAX_GAP_FILL_ITERATIONS:
            logger.warning(f"Max gap-fill iterations ({MAX_GAP_FILL_ITERATIONS}) reached, "
                          f"proceeding with {len(gaps)} remaining gaps")
            print(f"  → Max iterations reached, {len(gaps)} gaps remain")
            print("===== /IMPACT SIMULATION =====\n")
            break

        print(f"  → Filling {len(gaps)} gaps...")
        print("===== /IMPACT SIMULATION =====\n")

        # Run gap-filling agent
        gap_findings = run_gap_filling_agent(
            gaps, problem, explorer_analysis, state.method_cache, chat_history
        )

        # Update the architecture map with new findings for next simulation
        architecture_map += "\n\n=== Gap-Filling Findings (iteration " + str(iteration + 1) + ") ===\n"
        architecture_map += gap_findings

        # Update explorer analysis with simulation results for next iteration
        explorer_analysis += "\n\nPrevious simulation found these gaps:\n"
        for g in gaps:
            explorer_analysis += f"- {g.get('location', '')}: {g.get('issue', '')}\n"

    # Merge all simulation results
    final_result = {
        "assumption_changed": all_simulation_results[-1].get("assumption_changed", ""),
        "proposed_fix": [],
        "gaps": all_simulation_results[-1].get("gaps", []),
        "verified_safe": [],
    }
    # Collect all unique fix locations across iterations
    seen_locations = set()
    for sim in all_simulation_results:
        for fix in sim.get("proposed_fix", []):
            loc = fix.get("location", "")
            if loc not in seen_locations:
                seen_locations.add(loc)
                final_result["proposed_fix"].append(fix)
        for safe in sim.get("verified_safe", []):
            final_result["verified_safe"].append(safe)

    return final_result


# ---------------------------------------------------------------------------
# Reviewer agent (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Instance orchestration: three-phase pipeline
# ---------------------------------------------------------------------------

def run_for_instance(instance: Dict[str, Any], reg_entry: Dict[str, Any],
                      out_summary_file: str, single_enhanced_file: str = "") -> dict:
    """Orchestrate a single instance through the three-phase pipeline.

    Phase 1: Explorer agent — find the primary bug
    Phase 2: Architecture mapping — gather all related code
    Phase 3: Impact simulation + gap-filling loop
    Then: Final report generation + reviewer
    """
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})
    problem = instance.get("problem_statement", "") or ""

    # =====================================================================
    # Phase 1: Explorer agent — find the primary bug
    # =====================================================================
    chat_history: List[str] = []
    run_explorer_agent(problem, chat_history)

    # Extract explorer analysis
    explorer_thoughts = [e for e in chat_history if e.startswith("Thought:")]
    explorer_analysis = "\n\n".join(explorer_thoughts) if explorer_thoughts else ""
    logger.info(f"Explorer: {len(explorer_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods cached")

    # =====================================================================
    # Phase 2: Architecture mapping — gather all related code
    # =====================================================================
    architecture_map = run_architecture_mapping(
        problem, explorer_analysis, dict(state.method_cache), chat_history
    )

    # =====================================================================
    # Phase 3: Impact simulation + gap-filling loop
    # =====================================================================
    simulation_result = run_simulation_and_gap_filling(
        problem, explorer_analysis, architecture_map, chat_history
    )

    # Build enhanced analysis that includes simulation findings
    enhanced_analysis = explorer_analysis
    if simulation_result.get("assumption_changed"):
        enhanced_analysis += f"\n\nCore assumption being changed: {simulation_result['assumption_changed']}"
    for fix in simulation_result.get("proposed_fix", []):
        enhanced_analysis += f"\nFix: {fix.get('location', '')} — {fix.get('change', '')}"
    for gap in simulation_result.get("gaps", []):
        enhanced_analysis += f"\nRemaining gap: {gap.get('location', '')} — {gap.get('issue', '')}"

    # =====================================================================
    # Generate final bug report
    # =====================================================================
    final_report = {}
    raw_report_for_reviewer = None
    try:
        final_bug_report = generate_final_bug_report(
            state.method_cache, problem, enhanced_analysis,
            prompt_name="final_report_with_suggestions.txt",
            simulation_result=simulation_result,
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
    # Save single-agent output (before reviewer)
    # =====================================================================
    if single_enhanced_file:
        single_summary = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "classification_stats": state.classification_stats.get(instance_id, {}),
            "method_cache": state.method_cache,
            "class_skeleton_cache": {},
            "chat_history": chat_history,
            "simulation_result": simulation_result,
            "bug_report": final_report,
        }
        save_instance_result(single_summary, single_enhanced_file)
        logger.info(f"Saved single-agent output for {instance_id}")

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
        reviewer_draft, problem, state.method_cache, chat_history
    )
    final_report = reviewer_result.get("revised_report", final_report)

    # =====================================================================
    # Build and save result
    # =====================================================================
    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": state.method_cache,
        "class_skeleton_cache": build_class_skeleton_cache(),
        "chat_history": chat_history,
        "simulation_result": simulation_result,
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
