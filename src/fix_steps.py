"""Generate problem_location and fix_steps from existing multiagent_enhanced reports.

Standalone script that reads a multiagent_enhanced JSON file (which already
contains reviewer-validated bug_report and compressed_analysis), and produces
a new output file with added problem_location and fix_steps fields before
bug_report.

Usage:
    python run.py generate_fix_steps \
        --fix-steps-input data/output/multiagent_enhanced/astropy__astropy.json \
        --fix-steps-output data/output/gpt-5.4_fix/multiagent_enhanced/astropy__astropy.json \
        --instance-ids astropy__astropy-7746
"""

import json
import logging
import os
from typing import List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.io import load_json_safe, save_json_atomic, mkdirp
from src.utils.llm import make_chat_llm, load_prompt
from src.utils.tokens import count_tokens
from src.utils.json_parser import parse_json_best_effort

logger = logging.getLogger("context_engineering.fix_steps")


def _generate_for_instance(
    bug_report: dict,
    compressed_analysis: str,
    llm,
    prompts_dir: str,
) -> dict:
    """Generate problem_location and fix_steps for a single instance.

    Args:
        bug_report: The reviewer-validated bug report dict.
        compressed_analysis: The 3-level compressed analysis string.
        llm: The LangChain LLM instance.
        prompts_dir: Directory containing prompt templates.

    Returns:
        Dict with "problem_location" and "fix_steps" keys.
    """
    template = load_prompt("fix_steps.txt", prompts_dir)

    # Format bug report as readable text for the prompt
    if isinstance(bug_report, dict):
        bug_report_text = json.dumps(bug_report, indent=2, ensure_ascii=False)
    else:
        bug_report_text = str(bug_report)

    full_prompt = template.format(
        bug_report=bug_report_text,
        chat_history=compressed_analysis,
    )

    token_count = count_tokens(full_prompt)
    logger.info(f"Fix steps prompt: {token_count} tokens")

    if token_count > 250000:
        logger.warning(f"Prompt too large ({token_count} tokens), truncating analysis")
        lines = compressed_analysis.split("\n")
        mid = len(lines) // 2
        keep = mid // 2
        truncated = (lines[:keep] +
                     [f"\n... [{len(lines) - 2*keep} lines omitted] ...\n"] +
                     lines[-keep:])
        compressed_analysis = "\n".join(truncated)

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke({
        "bug_report": bug_report_text,
        "chat_history": compressed_analysis,
    })

    logger.info(f"LLM response: {count_tokens(raw_result)} tokens")

    # Parse the JSON response
    parsed = parse_json_best_effort(
        raw_result, preferred_keys=["problem_location", "fix_steps"]
    )

    if parsed and "problem_location" in parsed:
        return {
            "problem_location": parsed.get("problem_location", {}),
            "fix_steps": parsed.get("fix_steps", ""),
        }

    # Fallback: return raw text as fix_steps, empty problem_location
    logger.warning("Could not parse JSON response, using raw text as fix_steps")
    return {
        "problem_location": {},
        "fix_steps": raw_result,
    }


def run_fix_steps(cfg):
    """Run problem_location + fix_steps generation on an existing multiagent_enhanced file."""
    input_file = cfg.fix_steps_input
    output_file = cfg.fix_steps_output

    if not input_file:
        logger.error("No input file specified (--fix-steps-input)")
        return
    if not output_file:
        logger.error("No output file specified (--fix-steps-output)")
        return

    mkdirp(os.path.dirname(output_file) or ".")

    llm = make_chat_llm(cfg.openai_model, cfg.llm_temperature, cfg.openai_api_base, cfg.openai_api_key_env)

    entries = load_json_safe(input_file)
    if not entries:
        logger.error(f"No entries found in {input_file}")
        return

    # Filter instances if specified
    if cfg.instance_id_filter:
        want = set(str(x) for x in cfg.instance_id_filter)
        entries_to_process = [e for e in entries if str(e.get("instance_id")) in want]
        logger.info(f"Instance filter: {len(entries_to_process)}/{len(entries)} selected")
    else:
        entries_to_process = entries

    # Load existing output to preserve already-processed instances
    output_entries = load_json_safe(output_file)
    output_by_id = {e.get("instance_id"): e for e in output_entries}

    for entry in entries_to_process:
        instance_id = entry.get("instance_id")
        bug_report = entry.get("bug_report", {})
        compressed_analysis = entry.get("compressed_analysis", "")

        if not bug_report:
            logger.warning(f"No bug_report for {instance_id}, skipping")
            continue
        if not compressed_analysis:
            logger.warning(f"No compressed_analysis for {instance_id}, skipping")
            continue

        logger.info(f"\n{'='*60}\nGenerating problem_location + fix_steps for {instance_id}\n{'='*60}")

        try:
            result = _generate_for_instance(
                bug_report, compressed_analysis, llm, cfg.prompts_dir
            )
            problem_location = result["problem_location"]
            fix_steps = result["fix_steps"]
        except Exception as e:
            logger.error(f"Generation failed for {instance_id}: {e}")
            problem_location = {}
            fix_steps = None

        # Build output entry: all original fields, with new fields before bug_report
        out_entry = {}
        for key, value in entry.items():
            if key == "bug_report":
                # Insert problem_location and fix_steps right before bug_report
                out_entry["problem_location"] = problem_location
                out_entry["fix_steps"] = fix_steps
            out_entry[key] = value

        # If bug_report wasn't in the original (shouldn't happen), add at end
        if "problem_location" not in out_entry:
            out_entry["problem_location"] = problem_location
            out_entry["fix_steps"] = fix_steps

        output_by_id[instance_id] = out_entry

        # Save after each instance (atomic)
        all_output = list(output_by_id.values())
        save_json_atomic(all_output, output_file)
        logger.info(f"Saved problem_location + fix_steps for {instance_id} to {output_file}")

    logger.info(f"Done. Output: {output_file}")
