"""Multi-agent pipeline: single comprehensive explorer + reviewer.

Phase: 'enhance'
Input: Original SWE-Bench instances + code graphs
Output: Enhanced bug reports with root cause analysis and fix suggestions

Pipeline structure:
  1. Explorer agent — single agent with mandatory protocol that finds the bug,
     maps the full architecture, and proposes a complete fix (all in one context)
  2. Final report generation (LLM call) — formats findings into structured JSON
  3. Reviewer agent — validates and refines the report
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

logger = logging.getLogger("context_engineering.agents.multi_agent")

# ---------------------------------------------------------------------------
# Checkpoint completion detection
# ---------------------------------------------------------------------------

_CHECKPOINT_MARKERS = [
    ("checkpoint 1", "identify the bug"),
    ("checkpoint 2", "read the full file"),
    ("checkpoint 3", "map the architecture"),
    ("checkpoint 4", "propose a fix"),
    ("checkpoint 5", "final report"),
]


def _detect_completed_checkpoints(chat_history: List[str]) -> List[int]:
    """Detect which checkpoints the agent completed based on chat history.

    Returns list of checkpoint numbers (1-5) that appear to be completed.
    """
    full_text = "\n".join(chat_history).lower()
    completed = []
    for i, (marker, alt_marker) in enumerate(_CHECKPOINT_MARKERS, 1):
        if marker in full_text or alt_marker in full_text:
            completed.append(i)
    return completed


def _build_continuation_prompt(completed: List[int], chat_history: List[str]) -> str:
    """Build a continuation prompt telling the agent which checkpoints remain."""
    max_completed = max(completed) if completed else 0
    remaining = [i for i in range(1, 6) if i > max_completed]

    # Gather the agent's findings so far
    thoughts = [e for e in chat_history if e.startswith("Thought:")]
    recent_thoughts = thoughts[-5:] if thoughts else []

    prompt = (
        "You stopped your investigation early. You completed up to "
        f"Checkpoint {max_completed} but the protocol requires all 5 checkpoints.\n\n"
        "Here is what you found so far:\n\n"
        + "\n".join(recent_thoughts) + "\n\n"
        f"You MUST now continue with Checkpoint{'s' if len(remaining) > 1 else ''} "
        f"{', '.join(str(r) for r in remaining)}. "
        "Do NOT repeat work you already did. Do NOT re-read code you already read. "
        "Use the tools to continue your investigation from where you left off.\n\n"
        "Reminder of what remains:\n"
    )

    checkpoint_descriptions = {
        2: "Checkpoint 2: Read the ENTIRE file containing the primary class. "
           "List every class, their parent classes, all class-level attributes, "
           "and all methods. Use get_file_context to read the full file.",
        3: "Checkpoint 3: Map the architecture. Read component classes, "
           "parent class with get_file_context. Search for sibling classes. Use get_subgraph.",
        4: "Checkpoint 4: Propose a fix, then challenge it with questions 4a-4e. "
           "You MUST use get_file_context to read inherited methods during 4c and 4e. "
           "Trace through each method with concrete values. "
           "Do NOT conclude 'no new methods needed' without reading the code first.",
        5: "Checkpoint 5: Write your complete analysis covering root cause, "
           "every location that needs to change, hardcoded values, and "
           "methods to add or override.",
    }

    for r in remaining:
        if r in checkpoint_descriptions:
            prompt += f"\n- {checkpoint_descriptions[r]}"

    return prompt


# ---------------------------------------------------------------------------
# Explorer agent — single comprehensive investigation
# ---------------------------------------------------------------------------

def run_explorer_agent(problem: str, chat_history: List[str],
                       max_continuations: int = 3) -> List[Any]:
    """Run the explorer agent with the full mandatory protocol.

    This single agent finds the bug, maps the architecture, and proposes a
    complete fix — all within one context window. No lossy handoffs.

    If the agent stops before completing all 5 checkpoints, it is re-invoked
    with a continuation prompt up to max_continuations times.
    """
    logger.info("Running explorer agent (single-agent protocol)")

    state.active_chat_history = chat_history
    state.trace_include_observations = True

    instruction = load_prompt("agent_instruction_multi_agent.txt", state.config.prompts_dir)
    user_text = f"Problem: {problem}\n"

    tools = build_tools(for_reviewer=False)

    all_events = []

    # Initial run
    agent_events = run_agent_with_tools(
        instruction, user_text, tools, chat_history,
        recursion_limit=100, max_retries=2,
    )
    all_events.extend(agent_events)

    # Check if the agent completed all checkpoints; if not, continue
    for continuation in range(max_continuations):
        completed = _detect_completed_checkpoints(chat_history)
        max_completed = max(completed) if completed else 0
        logger.info(f"Checkpoints completed: {completed} (max: {max_completed})")

        if max_completed >= 4:
            # Checkpoint 5 is just writing up findings, so 4+ is sufficient
            logger.info("Agent completed through Checkpoint 4+, investigation sufficient")
            break

        logger.info(f"Agent stopped at Checkpoint {max_completed}, "
                     f"sending continuation {continuation + 1}/{max_continuations}")

        continuation_prompt = _build_continuation_prompt(completed, chat_history)
        continuation_events = run_agent_with_tools(
            instruction, continuation_prompt, tools, chat_history,
            recursion_limit=100, max_retries=1,
        )
        all_events.extend(continuation_events)

    # Final check
    completed = _detect_completed_checkpoints(chat_history)
    logger.info(f"Final checkpoints completed: {completed}")

    state.active_chat_history = None
    return all_events


# ---------------------------------------------------------------------------
# Reviewer agent
# ---------------------------------------------------------------------------

def run_reviewer_agent(draft_report: Any, problem: str,
                        compressed_history: str) -> dict:
    """Tool-enabled reviewer that validates and improves the draft report.

    The reviewer receives the compressed 3-level analysis instead of raw
    method_cache. It has tools (get_file_context, get_code, search_codebase)
    to look up any code it needs to verify.
    """
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
                      out_summary_file: str, single_enhanced_file: str = "") -> dict:
    """Orchestrate a single instance: explorer -> report -> reviewer."""
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})
    problem = instance.get("problem_statement", "") or ""

    # =====================================================================
    # Single explorer agent — full protocol in one context
    # =====================================================================
    chat_history: List[str] = []
    run_explorer_agent(problem, chat_history)

    # Extract analysis from the agent's reasoning
    agent_thoughts = [e for e in chat_history if e.startswith("Thought:")]
    logger.info(f"Explorer: {len(agent_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods cached")

    # =====================================================================
    # Compress chat history into 3-level structured memory
    # (HIGH_LEVEL / MID_LEVEL / LOW_LEVEL) to avoid information loss
    # =====================================================================
    compressed_analysis = compress_chat_history(chat_history, problem, state.method_cache)
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
            "compressed_analysis": compressed_analysis,
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
        reviewer_draft, problem, compressed_analysis,
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
