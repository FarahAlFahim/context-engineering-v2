"""Generate problem_location from existing multiagent_enhanced reports.

Reads a multiagent_enhanced JSON file (which contains reviewer-validated
bug_report), extracts problem_location using only the bug_report field
(no compressed_analysis), and writes a new output file with the
problem_location field added.

Usage:
    python run.py generate_problem_location \
        --problem-location-input data/output/gpt_5_mini/multiagent_enhanced/astropy__astropy.json \
        --problem-location-output data/output/problem_location/gpt_5_mini/astropy__astropy.json

    python run.py generate_problem_location \
        --problem-location-input data/output/gpt_5_mini/multiagent_enhanced/mwaskom__seaborn.json \
        --problem-location-output data/output/problem_location/gpt_5_mini/mwaskom__seaborn.json \
        --instance-ids astropy__astropy-14182
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

logger = logging.getLogger("context_engineering.problem_location")


def _generate_for_instance(bug_report: dict, llm, prompts_dir: str) -> dict:
    """Generate problem_location for a single instance from its bug_report.

    Args:
        bug_report: The reviewer-validated bug report dict.
        llm: The LangChain LLM instance.
        prompts_dir: Directory containing prompt templates.

    Returns:
        Dict with "problem_location" key.
    """
    template = load_prompt("problem_location.txt", prompts_dir)

    if isinstance(bug_report, dict):
        bug_report_text = json.dumps(bug_report, indent=2, ensure_ascii=False)
    else:
        bug_report_text = str(bug_report)

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke({"bug_report": bug_report_text})

    token_count = count_tokens(raw_result)
    logger.info(f"LLM response: {token_count} tokens")

    parsed = parse_json_best_effort(
        raw_result, preferred_keys=["problem_location"]
    )

    if parsed and "problem_location" in parsed:
        return parsed["problem_location"]

    # Fallback: return empty
    logger.warning("Could not parse JSON response for problem_location")
    return {}


def run_problem_location(cfg):
    """Run problem_location generation on an existing multiagent_enhanced file."""
    input_file = cfg.problem_location_input
    output_file = cfg.problem_location_output

    if not input_file:
        logger.error("No input file specified (--problem-location-input)")
        return
    if not output_file:
        logger.error("No output file specified (--problem-location-output)")
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

        if not bug_report:
            logger.warning(f"No bug_report for {instance_id}, skipping")
            continue

        logger.info(f"\n{'='*60}\nGenerating problem_location for {instance_id}\n{'='*60}")

        try:
            problem_location = _generate_for_instance(
                bug_report, llm, cfg.prompts_dir
            )
        except Exception as e:
            logger.error(f"Generation failed for {instance_id}: {e}")
            problem_location = {}

        # Build output entry: all original fields, with problem_location before bug_report
        out_entry = {}
        for key, value in entry.items():
            if key == "bug_report":
                out_entry["problem_location"] = problem_location
            out_entry[key] = value

        if "problem_location" not in out_entry:
            out_entry["problem_location"] = problem_location

        output_by_id[instance_id] = out_entry

        # Save after each instance (atomic)
        all_output = list(output_by_id.values())
        save_json_atomic(all_output, output_file)
        logger.info(f"Saved problem_location for {instance_id} to {output_file}")

    logger.info(f"Done. Output: {output_file}")
