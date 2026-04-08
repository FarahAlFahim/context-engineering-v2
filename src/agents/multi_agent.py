"""Multi-agent pipeline: Discovery → Challenge → Report → Reviewer.

Phase: 'enhance'
Input: Original SWE-Bench instances + code graphs
Output: Enhanced bug reports with root cause analysis and fix suggestions

Pipeline structure (inspired by claw-code's role-based agent separation):
  1. Discovery agent — finds the bug, catalogs code, maps architecture.
     Outputs structured JSON handoff (not free-form prose).
     Tools: all (classify_report, semantic_rank, navigation tools).
  2. Challenge agent — receives structured handoff and systematically
     verifies EVERY inherited method and hardcoded value with concrete
     value traces. Does not choose what to check — checks everything.
     Tools: navigation tools only (get_file_context, get_code, etc).
  3. Final report generation (LLM call) — formats findings into JSON.
  4. Reviewer agent — validates and refines the report.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

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
# Checkpoint completion detection (for Discovery agent)
# ---------------------------------------------------------------------------

_DISCOVERY_CHECKPOINT_MARKERS = [
    ("checkpoint 1", "identify the bug"),
    ("checkpoint 2", "read the full file"),
    ("checkpoint 3", "map the architecture"),
    ("checkpoint 4", "structured handoff"),
]

_CHALLENGE_STEP_MARKERS = [
    ("step 1", "understand the handoff"),
    ("step 2", "propose a fix"),
    ("step 3", "challenge every hardcoded"),
    ("step 4", "challenge every inherited"),
    ("step 5", "check existing overridden"),
    ("step 6", "final diagnosis"),
]


def _detect_completed_markers(chat_history: List[str],
                               markers: list) -> List[int]:
    """Detect which checkpoints/steps the agent completed."""
    full_text = "\n".join(chat_history).lower()
    completed = []
    for i, (marker, alt_marker) in enumerate(markers, 1):
        if marker in full_text or alt_marker in full_text:
            completed.append(i)
    return completed


def _build_discovery_continuation(completed: List[int],
                                   chat_history: List[str]) -> str:
    """Build continuation prompt for Discovery agent."""
    max_completed = max(completed) if completed else 0
    remaining = [i for i in range(1, 5) if i > max_completed]

    thoughts = [e for e in chat_history if e.startswith("Thought:")]
    recent_thoughts = thoughts[-5:] if thoughts else []

    prompt = (
        "You stopped your investigation early. You completed up to "
        f"Checkpoint {max_completed} but the protocol requires all 4 checkpoints.\n\n"
        "Here is what you found so far:\n\n"
        + "\n".join(recent_thoughts) + "\n\n"
        f"You MUST now continue with Checkpoint{'s' if len(remaining) > 1 else ''} "
        f"{', '.join(str(r) for r in remaining)}. "
        "Do NOT repeat work you already did. Do NOT re-read code you already read. "
        "Continue from where you left off.\n\n"
        "Reminder of what remains:\n"
    )

    descriptions = {
        2: "Checkpoint 2: Read the ENTIRE file containing the primary class. "
           "List every class, their parent classes, all class-level attributes, "
           "and all methods. Use get_file_context to read the full file.",
        3: "Checkpoint 3: Map the architecture. Read component classes, "
           "parent class with get_file_context. Search for sibling classes. "
           "Use get_subgraph.",
        4: "Checkpoint 4: Produce the structured JSON handoff. This MUST "
           "include all inherited methods, all hardcoded values, and all "
           "component classes. Output the JSON in a ```json code block.",
    }

    for r in remaining:
        if r in descriptions:
            prompt += f"\n- {descriptions[r]}"

    return prompt


def _build_challenge_continuation(completed: List[int],
                                   chat_history: List[str]) -> str:
    """Build continuation prompt for Challenge agent."""
    max_completed = max(completed) if completed else 0
    remaining = [i for i in range(1, 7) if i > max_completed]

    thoughts = [e for e in chat_history if e.startswith("Thought:")]
    recent_thoughts = thoughts[-5:] if thoughts else []

    prompt = (
        "You stopped your verification early. You completed up to "
        f"Step {max_completed} but the protocol requires all 6 steps.\n\n"
        "Here is what you found so far:\n\n"
        + "\n".join(recent_thoughts) + "\n\n"
        f"You MUST now continue with Step{'s' if len(remaining) > 1 else ''} "
        f"{', '.join(str(r) for r in remaining)}. "
        "Do NOT repeat work you already did. Continue from where you left off.\n\n"
        "Reminder of what remains:\n"
    )

    descriptions = {
        3: "Step 3: Challenge every hardcoded value from the handoff.",
        4: "Step 4: Challenge every inherited method — read each one with "
           "get_file_context and write a concrete value trace.",
        5: "Step 5: Check existing overridden methods still work under the fix.",
        6: "Step 6: Write the final diagnosis with all locations, formulas, "
           "and methods that need to change.",
    }

    for r in remaining:
        if r in descriptions:
            prompt += f"\n- {descriptions[r]}"

    return prompt


# ---------------------------------------------------------------------------
# Handoff parsing: extract structured JSON from Discovery agent output
# ---------------------------------------------------------------------------

def _extract_handoff_json(chat_history: List[str]) -> Optional[dict]:
    """Extract the structured handoff JSON from the Discovery agent's output.

    Looks for a JSON block in the agent's thoughts, searching from the end
    (most recent output) backwards.
    """
    # Join all chat history and search for ```json ... ``` blocks
    full_text = "\n".join(chat_history)

    # Find all ```json blocks
    json_blocks = re.findall(r'```json\s*\n(.*?)```', full_text, re.DOTALL)

    # Try blocks from last to first (most recent is most likely the handoff)
    for block in reversed(json_blocks):
        try:
            parsed = json.loads(block.strip())
            # Validate it looks like a handoff (has expected keys)
            if isinstance(parsed, dict) and "primary_class" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try parse_json_best_effort on each thought line (reversed)
    for line in reversed(chat_history):
        if "primary_class" in line and "parent_class_methods" in line:
            parsed = parse_json_best_effort(
                line, preferred_keys=["primary_class"]
            )
            if parsed and "primary_class" in parsed:
                return parsed

    return None


def _format_handoff_for_challenge(handoff: dict) -> str:
    """Format the structured handoff JSON into the Challenge agent's input."""
    return (
        "=== DISCOVERY AGENT HANDOFF ===\n\n"
        "The Discovery agent has completed its investigation. Below is the "
        "structured output. You MUST check EVERY method in "
        "`inherited_without_override` and EVERY value in "
        "`hardcoded_values`.\n\n"
        "```json\n"
        + json.dumps(handoff, indent=2, ensure_ascii=False)
        + "\n```"
    )


def _build_fallback_handoff(chat_history: List[str], problem: str) -> str:
    """Build a fallback handoff when structured JSON extraction fails.

    Uses the Discovery agent's raw reasoning as context for the Challenge agent.
    """
    thoughts = [e for e in chat_history if e.startswith("Thought:")]
    actions = [e for e in chat_history
               if e.startswith("Action:") or e.startswith("Observation:")]

    # Take the last N thoughts and actions as context
    recent = thoughts[-10:] + actions[-10:]

    return (
        "=== DISCOVERY AGENT FINDINGS (unstructured fallback) ===\n\n"
        "The Discovery agent did not produce a valid structured JSON handoff. "
        "Below are its most recent findings. You must extract the relevant "
        "information and proceed with your verification protocol.\n\n"
        "Original bug report:\n" + problem + "\n\n"
        "Discovery agent findings:\n" + "\n".join(recent)
    )


# ---------------------------------------------------------------------------
# Discovery agent — broad search and architecture mapping
# ---------------------------------------------------------------------------

def run_discovery_agent(problem: str, chat_history: List[str],
                        max_continuations: int = 3) -> List[Any]:
    """Run the Discovery agent (Checkpoints 1-3 + structured handoff).

    This agent finds the bug, catalogs all code, and maps the architecture.
    It outputs a structured JSON handoff for the Challenge agent.

    Tools: all (classify_report, semantic_rank, + navigation tools).
    """
    logger.info("Running Discovery agent (broad search + architecture mapping)")

    state.active_chat_history = chat_history
    state.trace_include_observations = True

    instruction = load_prompt("discovery_agent.txt", state.config.prompts_dir)
    user_text = f"Problem: {problem}\n"

    tools = build_tools(role="discovery")

    all_events = []

    # Initial run
    agent_events = run_agent_with_tools(
        instruction, user_text, tools, chat_history,
        recursion_limit=100, max_retries=2,
    )
    all_events.extend(agent_events)

    # Check if the agent completed all checkpoints; if not, continue
    for continuation in range(max_continuations):
        completed = _detect_completed_markers(
            chat_history, _DISCOVERY_CHECKPOINT_MARKERS
        )
        max_completed = max(completed) if completed else 0
        logger.info(f"Discovery checkpoints completed: {completed} "
                    f"(max: {max_completed})")

        if max_completed >= 3:
            # Checkpoint 4 is the handoff — check if JSON was produced
            handoff = _extract_handoff_json(chat_history)
            if handoff or max_completed >= 4:
                logger.info("Discovery agent completed, handoff available")
                break

        logger.info(f"Discovery agent stopped at Checkpoint {max_completed}, "
                    f"sending continuation {continuation + 1}/{max_continuations}")

        continuation_prompt = _build_discovery_continuation(
            completed, chat_history
        )
        continuation_events = run_agent_with_tools(
            instruction, continuation_prompt, tools, chat_history,
            recursion_limit=100, max_retries=1,
        )
        all_events.extend(continuation_events)

    # Final check
    completed = _detect_completed_markers(
        chat_history, _DISCOVERY_CHECKPOINT_MARKERS
    )
    logger.info(f"Discovery final checkpoints: {completed}")

    state.active_chat_history = None
    return all_events


# ---------------------------------------------------------------------------
# Challenge agent — targeted verification of every method and value
# ---------------------------------------------------------------------------

def run_challenge_agent(problem: str, handoff_text: str,
                        chat_history: List[str],
                        max_continuations: int = 2) -> List[Any]:
    """Run the Challenge agent (Steps 1-6).

    This agent receives the structured handoff from Discovery and
    systematically verifies EVERY inherited method and hardcoded value.
    It does not choose what to check — it checks everything in the handoff.

    Tools: navigation only (get_file_context, get_code, search_codebase,
    get_subgraph). No classify_report or semantic_rank — it doesn't need
    broad search.
    """
    logger.info("Running Challenge agent (targeted verification)")

    state.active_chat_history = chat_history
    state.trace_include_observations = True

    instruction = load_prompt("challenge_agent.txt", state.config.prompts_dir)
    user_text = f"Original bug report:\n{problem}\n\n{handoff_text}\n"

    tools = build_tools(role="challenge")

    all_events = []

    # Initial run
    agent_events = run_agent_with_tools(
        instruction, user_text, tools, chat_history,
        recursion_limit=100, max_retries=2,
    )
    all_events.extend(agent_events)

    # Check if the agent completed all steps; if not, continue
    for continuation in range(max_continuations):
        completed = _detect_completed_markers(
            chat_history, _CHALLENGE_STEP_MARKERS
        )
        max_completed = max(completed) if completed else 0
        logger.info(f"Challenge steps completed: {completed} "
                    f"(max: {max_completed})")

        if max_completed >= 5:
            # Step 6 is final diagnosis — 5+ is sufficient
            logger.info("Challenge agent completed through Step 5+")
            break

        logger.info(f"Challenge agent stopped at Step {max_completed}, "
                    f"sending continuation {continuation + 1}/{max_continuations}")

        continuation_prompt = _build_challenge_continuation(
            completed, chat_history
        )
        continuation_events = run_agent_with_tools(
            instruction, continuation_prompt, tools, chat_history,
            recursion_limit=100, max_retries=1,
        )
        all_events.extend(continuation_events)

    # Final check
    completed = _detect_completed_markers(
        chat_history, _CHALLENGE_STEP_MARKERS
    )
    logger.info(f"Challenge final steps: {completed}")

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

    reviewer_tools = build_tools(role="reviewer")
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
    """Orchestrate: Discovery → Challenge → Report → Reviewer."""
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})
    problem = instance.get("problem_statement", "") or ""

    # =====================================================================
    # Phase 1: Discovery agent — broad search + architecture mapping
    # =====================================================================
    discovery_history: List[str] = []
    run_discovery_agent(problem, discovery_history)

    discovery_thoughts = [e for e in discovery_history if e.startswith("Thought:")]
    logger.info(f"Discovery: {len(discovery_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods cached")

    # =====================================================================
    # Extract structured handoff from Discovery agent
    # =====================================================================
    handoff = _extract_handoff_json(discovery_history)
    if handoff:
        handoff_text = _format_handoff_for_challenge(handoff)
        logger.info(f"Structured handoff extracted: "
                    f"{len(handoff.get('parent_class_methods', {}).get('inherited_without_override', []))} "
                    f"inherited methods, "
                    f"{sum(len(c.get('hardcoded_values', {})) for c in handoff.get('component_classes', []))} "
                    f"hardcoded values")
    else:
        logger.warning("No structured handoff JSON found — using fallback")
        handoff_text = _build_fallback_handoff(discovery_history, problem)

    # =====================================================================
    # Phase 2: Challenge agent — targeted verification
    # =====================================================================
    challenge_history: List[str] = []
    run_challenge_agent(problem, handoff_text, challenge_history)

    challenge_thoughts = [e for e in challenge_history if e.startswith("Thought:")]
    logger.info(f"Challenge: {len(challenge_thoughts)} thoughts, "
                f"{len(state.method_cache)} methods cached")

    # =====================================================================
    # Combine histories for compression
    # The combined history gives the compressor the full investigation
    # trajectory: discovery findings + challenge verifications
    # =====================================================================
    combined_history = (
        ["=== DISCOVERY PHASE ==="] + discovery_history +
        ["=== CHALLENGE PHASE ==="] + challenge_history
    )

    # =====================================================================
    # Compress combined history into 3-level structured memory
    # =====================================================================
    compressed_analysis = compress_chat_history(combined_history, problem)
    logger.info("Compressed combined history into 3-level memory")

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
            "discovery_history": discovery_history,
            "challenge_history": challenge_history,
            "method_cache": state.method_cache,
            "bug_report": {},
        }
        save_instance_result(error_summary, out_summary_file)
        state.current_reg_entry = None
        return error_summary

    # =====================================================================
    # Save pre-reviewer output
    # =====================================================================
    if single_enhanced_file:
        single_summary = {
            "instance_id": instance_id,
            "repo": instance.get("repo"),
            "base_commit": instance.get("base_commit"),
            "classification_stats": state.classification_stats.get(instance_id, {}),
            "method_cache": state.method_cache,
            "class_skeleton_cache": {},
            "discovery_history": discovery_history,
            "challenge_history": challenge_history,
            "handoff": handoff,
            "compressed_analysis": compressed_analysis,
            "bug_report": final_report,
        }
        save_instance_result(single_summary, single_enhanced_file)
        logger.info(f"Saved pre-reviewer output for {instance_id}")

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
        "discovery_history": discovery_history,
        "challenge_history": challenge_history,
        "handoff": handoff,
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
