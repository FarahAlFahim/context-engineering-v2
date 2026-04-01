"""Dynamic iterative enhancement with mini-sweagent feedback loop.

Phase: 'dynamic_enhance'
Input: Multi-agent enhanced reports + trajectory folder + mini-sweagent integration
Output: Iteratively refined reports using dynamic trajectory feedback
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import src.state as state
from src.agents.common import (
    prepare_instance_state, filter_chat_history_for_method_cache,
    generate_final_bug_report, parse_reviewer_output,
    build_class_skeleton_cache, save_instance_result,
    run_agent_with_tools, append_langgraph_ai_messages_only,
)
from src.tools.registry import build_tools
from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.tokens import count_tokens
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.agents.dynamic_insights")


# ---------- trajectory extraction helpers ----------

_HEREDOC_RE = re.compile(r"<<\s*['\"]?(\w+)['\"]?\n", re.MULTILINE)
_RETURNCODE_RE = re.compile(r"<returncode>(\d+)</returncode>")
_OUTPUT_BODY_RE = re.compile(r"<output>\n?([\s\S]*?)\n?</output>")


def _compact_tool_output(content: str, max_body_lines: int = 20) -> str:
    rc_m = _RETURNCODE_RE.search(content)
    rc = rc_m.group(1) if rc_m else "?"
    body_m = _OUTPUT_BODY_RE.search(content)
    if not body_m:
        return f"[returncode={rc}] (no output)"
    body = body_m.group(1)
    lines = body.splitlines()
    if len(lines) <= max_body_lines:
        return f"[returncode={rc}]\n{body}"
    preview = "\n".join(lines[:5])
    return f"[returncode={rc}]\n{preview}\n[TRUNCATED: {len(lines) - 5} more lines]"


def extract_compact_trajectory(traj_text: str) -> str:
    """Return a compact text summary of a mini-sweagent trajectory."""
    obj = json.loads(traj_text)
    msgs = obj.get("messages", [])
    parts = []
    step = 0

    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            parts.append("[SYSTEM PROMPT]")
        elif role == "user":
            parts.append(f"USER:\n{content}")
        elif role == "assistant":
            step += 1
            thought = content.strip() if content.strip() else None
            cmd = None
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments", "")
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
                        cmd = parsed["command"]
                except Exception:
                    pass
            line = f"STEP {step} — ASSISTANT:"
            if thought:
                line += f"\n  {thought}"
            if cmd:
                line += f"\n  COMMAND: {cmd}"
            parts.append(line)
        elif role == "tool":
            parts.append(f"  RESULT: {_compact_tool_output(content)}")
        elif role == "exit":
            exit_status = (msg.get("extra") or {}).get("exit_status", "unknown")
            parts.append(f"EXIT: {exit_status}")
            if content.strip():
                parts.append(f"SUBMISSION:\n{content.strip()}")

    return "\n\n".join(parts)


def find_traj_path(traj_root: Path, instance_id: str) -> Optional[Path]:
    """Find trajectory file for an instance."""
    candidates = [
        traj_root / instance_id / f"{instance_id}.traj.json",
        traj_root / instance_id / f"{instance_id}.traj",
        traj_root / f"{instance_id}.traj.json",
        traj_root / f"{instance_id}.traj",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def split_extracted_trajectory_phases(extracted_trajectory: str) -> Dict[str, str]:
    """Split trajectory into localization and repair phases using LLM."""
    if not extracted_trajectory.strip():
        return {"localization": "", "repair": ""}

    template = load_prompt("trajectory_phase_split.txt", state.config.prompts_dir)
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=state.llm, prompt=prompt)
    try:
        raw = chain.run({"trajectory": extracted_trajectory})
        txt = raw.replace("```json\n", "").replace("\n```", "").strip()
        out = json.loads(txt)
        return {
            "localization": str(out.get("localization", "") or ""),
            "repair": str(out.get("repair", "") or ""),
        }
    except Exception as e:
        logger.warning(f"split_extracted_trajectory_phases failed: {e}")
        return {"localization": "", "repair": extracted_trajectory}


def _select_phase_for_enhancement(
    problem: str, trajectory_phases: Dict[str, str],
    default_trajectory: str, phase_order: Tuple[str, str],
    chat_history: List[str],
) -> Tuple[bool, Dict[str, bool], str, str, str]:
    """Compare phases with bug report and select the non-redundant phase."""
    phase_usage = {"localization": False, "repair": False}
    selected_phase = "full"
    trajectory_for_enhancement = default_trajectory
    further_enhancement_reason = ""

    if not problem:
        return False, phase_usage, selected_phase, trajectory_for_enhancement, further_enhancement_reason

    compare_template = load_prompt("phase_comparison.txt", state.config.prompts_dir)
    compare_prompt = PromptTemplate.from_template(compare_template)
    compare_chain = LLMChain(llm=state.llm, prompt=compare_prompt)

    all_phases_redundant = True
    for next_phase in phase_order:
        phase_usage[next_phase] = True
        traj = (trajectory_phases.get(next_phase) or "").strip()
        if not traj:
            chat_history.append(f"[phase] {next_phase}: empty trajectory")
            continue

        cmp_raw = compare_chain.run({"problem": problem, "traj": traj})
        cmp_json = cmp_raw.replace("```json\n", "").replace("\n```", "").strip()
        cmp = json.loads(cmp_json)
        logger.info(f"Phase comparison ({next_phase}): similar={cmp.get('similar')}")

        if bool(cmp.get("similar")):
            chat_history.append(f"[phase] {next_phase} redundant: {cmp.get('reason','')}")
            continue

        all_phases_redundant = False
        further_enhancement_reason = str(cmp.get("reason", "") or "")
        trajectory_for_enhancement = traj
        selected_phase = next_phase
        chat_history.append(f"[phase] {next_phase} selected: {further_enhancement_reason}")
        break

    return all_phases_redundant, phase_usage, selected_phase, trajectory_for_enhancement, further_enhancement_reason


def _problem_obj_to_text(problem_obj: Any) -> str:
    if isinstance(problem_obj, (dict, list)):
        return json.dumps(problem_obj, ensure_ascii=False, indent=2)
    return str(problem_obj or "")


def _reviewer_agent(enhanced_report: Any, problem: str,
                     trajectory_summary: str) -> dict:
    """Reviewer agent that integrates trajectory insights."""
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
        "\n\n=== Existing enhanced bug report ===\n" + draft_json +
        "\n\n=== Trajectory summary ===\n" + trajectory_summary
    )

    reviewer_tools = build_tools(for_reviewer=True)
    agent_events = run_agent_with_tools(
        instruction, user_text, reviewer_tools, reviewer_history, recursion_limit=60
    )

    state.active_chat_history = None
    state.trace_include_observations = prev_trace_obs
    return parse_reviewer_output(reviewer_history, agent_events, enhanced_report)


def _run_minisweagent_wrapper(instance: dict, reg_entry: dict,
                               enhanced_problem_obj: Any, instance_id: str,
                               iteration_index: int) -> Tuple[str, Optional[Path]]:
    """Run mini-sweagent wrapper subprocess for one iteration."""
    cfg = state.config
    if not os.path.exists(cfg.minisweagent_python):
        logger.warning(f"Wrapper python not found: {cfg.minisweagent_python}")
        return "", None
    if not os.path.exists(cfg.minisweagent_wrapper_script):
        logger.warning(f"Wrapper script not found: {cfg.minisweagent_wrapper_script}")
        return "", None

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"iterative_{instance_id}_"))
    try:
        dataset_path = tmp_dir / "dataset.json"
        run_name = cfg.minisweagent_run_name

        _KEYS = ("Title", "Description", "RootCause", "StepsToReproduce",
                 "ExpectedBehavior", "ObservedBehavior", "Suggestions")
        if isinstance(enhanced_problem_obj, dict):
            filtered = {k: enhanced_problem_obj[k] for k in _KEYS if k in enhanced_problem_obj}
        else:
            filtered = enhanced_problem_obj

        temp_data = {
            "instance_id": instance_id,
            "repo": instance.get("repo") or reg_entry.get("repo"),
            "base_commit": instance.get("base_commit") or reg_entry.get("base_commit"),
            "problem_statement": _problem_obj_to_text(filtered),
        }
        save_json_atomic([temp_data], str(dataset_path))

        wrapper_cfg = {
            "VARIANT": cfg.minisweagent_variant,
            "DATASET_JSON": str(dataset_path),
            "RUN_NAME": run_name,
            "RESULTS_ROOT": cfg.minisweagent_results_root,
            "INSTANCES_FILTER": [instance_id],
            "INSTANCES_SLICE": "",
            "INSTANCES_SHUFFLE": False,
            "PROBLEM_FIELD": "problem_statement",
            "RUN_MODE": "per_instance",
            "REDO_EXISTING": True,
            "DRY_RUN": False,
        }

        py_code = (
            "import importlib.util, json, sys; "
            "spec=importlib.util.spec_from_file_location('msw_wrap', sys.argv[1]); "
            "mod=importlib.util.module_from_spec(spec); "
            "sys.modules['msw_wrap'] = mod; "
            "spec.loader.exec_module(mod); "
            "mod.CONFIG.update(json.loads(sys.argv[2])); "
            "mod.run()"
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [cfg.minisweagent_python, "-c", py_code, cfg.minisweagent_wrapper_script, json.dumps(wrapper_cfg)],
            cwd=cfg.minisweagent_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        out_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_lines.append(line)
        rc = proc.wait()

        if rc != 0:
            logger.warning(f"Wrapper failed for {instance_id}, iter={iteration_index}, rc={rc}")
            return "", None

        rep_dir = Path(cfg.minisweagent_results_root) / run_name / cfg.minisweagent_variant / "repair"
        new_traj_path = find_traj_path(rep_dir, str(instance_id))
        if new_traj_path is None:
            logger.warning(f"No trajectory found in {rep_dir} for {instance_id}")
            return "", None

        traj_raw = new_traj_path.read_text(encoding="utf-8", errors="replace")
        return extract_compact_trajectory(traj_raw), new_traj_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_for_instance(instance: Dict[str, Any], reg_entry: Dict[str, Any],
                      out_summary_file: str) -> dict:
    """Orchestrate iterative dynamic enhancement for a single instance."""
    cfg = state.config
    instance_id = instance.get("instance_id")
    logger.info(f"Processing instance: {instance_id}")

    state.reset_instance_state()
    state.current_reg_entry = reg_entry
    prepare_instance_state(reg_entry, build_embeddings=False)

    state.classification_stats.setdefault(instance_id, {})

    run_chat_history: List[str] = []
    iteration_chat_histories: List[List[str]] = []
    iteration_method_caches: List[Dict[str, str]] = []
    iteration_class_skeleton_caches: List[Dict[str, str]] = []
    iteration_selected_phases: List[str] = []

    state.active_chat_history = None

    # Load bug report
    problem_obj = instance.get("bug_report")
    input_method_cache = instance.get("method_cache", {}) or {}
    input_class_skeleton_cache = instance.get("class_skeleton_cache", {}) or {}

    if not problem_obj:
        problem_obj = instance.get("problem_statement", "")

    problem = _problem_obj_to_text(problem_obj) if isinstance(problem_obj, (dict, list)) else (problem_obj or "")
    if not problem:
        problem = instance.get("problem_statement", "") or ""

    # Load and extract trajectory
    extracted_trajectory = ""
    traj_path = find_traj_path(Path(cfg.trajectory_folder), str(instance_id or ""))
    if traj_path is None:
        logger.warning(f"Trajectory not found for {instance_id} under {cfg.trajectory_folder}")
    else:
        traj_raw = traj_path.read_text(encoding="utf-8", errors="replace")
        extracted_trajectory = extract_compact_trajectory(traj_raw)

    trajectory_phases = split_extracted_trajectory_phases(extracted_trajectory)
    logger.info(f"Trajectory phases: localization={len(trajectory_phases.get('localization',''))} chars, "
                f"repair={len(trajectory_phases.get('repair',''))} chars")

    phase_order = ("localization", "repair")
    phase_usage = {"localization": False, "repair": False}
    trajectory_for_enhancement = extracted_trajectory
    selected_phase = "full"
    latest_trajectory_path = str(traj_path) if traj_path else None
    latest_trajectory_phases = trajectory_phases
    latest_phase_usage = phase_usage
    latest_selected_phase = selected_phase
    final_report = problem_obj if isinstance(problem_obj, (dict, list)) else {"raw": problem}

    if cfg.dry_run:
        logger.info(f"DRY RUN — {instance_id}, trajectory tokens: {count_tokens(extracted_trajectory)}")
        state.current_reg_entry = None
        return {"instance_id": instance_id, "dry_run": True}

    # Phase comparison
    further_enhancement_reason = ""
    try:
        (all_redundant, latest_phase_usage, latest_selected_phase,
         trajectory_for_enhancement, further_enhancement_reason) = _select_phase_for_enhancement(
            problem=problem, trajectory_phases=latest_trajectory_phases,
            default_trajectory=extracted_trajectory, phase_order=phase_order,
            chat_history=run_chat_history,
        )
        if all_redundant:
            logger.info(f"Skipping {instance_id}: both phases redundant")
            summary = {
                "instance_id": instance_id,
                "repo": instance.get("repo"),
                "base_commit": instance.get("base_commit"),
                "method_cache": input_method_cache,
                "class_skeleton_cache": input_class_skeleton_cache,
                "chat_history": run_chat_history,
                "further_enhanced": False,
                "trajectory_phases": latest_trajectory_phases,
                "phase_usage": latest_phase_usage,
                "bug_report": problem_obj if isinstance(problem_obj, (dict, list)) else {"raw": problem},
            }
            save_instance_result(summary, out_summary_file)
            state.current_reg_entry = None
            return summary
    except Exception:
        pass

    # Load original problem statement
    original_entries = load_json_safe(cfg.original_instances_json)
    original_by_id = {(e.get("instance_id") or e.get("instance")): e
                       for e in original_entries if isinstance(e, dict)}
    original_problem = (original_by_id.get(instance_id) or {}).get("problem_statement", "")

    # Iterative refinement loop
    iterative_rounds = 0
    while iterative_rounds < cfg.max_iterative_refinement_rounds:
        iterative_rounds += 1
        logger.info(f"Enhancement round {iterative_rounds} for {instance_id}, phase: {latest_selected_phase}")
        iteration_selected_phases.append(latest_selected_phase)

        iteration_chat_history: List[str] = []
        state.method_cache_global = set()
        state.method_cache = {}
        if input_method_cache:
            state.method_cache.update(input_method_cache)

        reviewer_result = _reviewer_agent(
            enhanced_report=problem_obj,
            problem=original_problem,
            trajectory_summary=trajectory_for_enhancement,
        )
        final_report = reviewer_result.get("revised_report", problem_obj)
        iteration_chat_history.extend(reviewer_result.get("reviewer_history", []))

        iteration_chat_histories.append(iteration_chat_history)
        iteration_method_caches.append(dict(state.method_cache))
        iteration_class_skeleton_caches.append(build_class_skeleton_cache())
        state.active_chat_history = None

        problem_obj = final_report
        problem = _problem_obj_to_text(final_report)

        # Run mini-sweagent for next iteration
        new_traj, new_traj_path = _run_minisweagent_wrapper(
            instance=instance, reg_entry=reg_entry,
            enhanced_problem_obj=final_report,
            instance_id=str(instance_id),
            iteration_index=iterative_rounds,
        )
        if not new_traj:
            break

        latest_trajectory_path = str(new_traj_path) if new_traj_path else latest_trajectory_path
        latest_trajectory_phases = split_extracted_trajectory_phases(new_traj)

        try:
            (all_redundant, latest_phase_usage, latest_selected_phase,
             trajectory_for_enhancement, latest_reason) = _select_phase_for_enhancement(
                problem=problem, trajectory_phases=latest_trajectory_phases,
                default_trajectory=new_traj, phase_order=phase_order,
                chat_history=run_chat_history,
            )
            if latest_reason:
                further_enhancement_reason = latest_reason
            if all_redundant:
                logger.info(f"Stopping iterative loop: trajectory is redundant")
                break
        except Exception:
            break

    state.active_chat_history = None

    latest_method_cache = iteration_method_caches[-1] if iteration_method_caches else {}
    latest_skeleton_cache = iteration_class_skeleton_caches[-1] if iteration_class_skeleton_caches else {}

    instance_summary = {
        "instance_id": instance_id,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
        "classification_stats": state.classification_stats.get(instance_id, {}),
        "method_cache": latest_method_cache,
        "class_skeleton_cache": latest_skeleton_cache,
        "chat_history": (iteration_chat_histories[-1] if iteration_chat_histories else run_chat_history),
        "run_chat_history": run_chat_history,
        "iteration_chat_histories": iteration_chat_histories,
        "iteration_selected_phases": iteration_selected_phases,
        "further_enhancement_reason": further_enhancement_reason,
        "further_enhanced": True,
        "trajectory_path": latest_trajectory_path,
        "trajectory_phases": latest_trajectory_phases,
        "phase_usage": latest_phase_usage,
        "selected_phase_for_enhancement": latest_selected_phase,
        "iterative_rounds": iterative_rounds,
        "bug_report": final_report,
    }

    save_instance_result(instance_summary, out_summary_file)
    state.current_reg_entry = None
    return instance_summary


def run_pipeline(cfg):
    """Run the dynamic iterative enhancement pipeline."""
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
                "instance_id": instance_id, "repo": inst.get("repo"),
                "base_commit": inst.get("base_commit"), "error": str(e),
            }
            all_entries = load_json_safe(out_file)
            all_entries.append(error_entry)
            save_json_atomic(all_entries, out_file)
