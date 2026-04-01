"""Evaluate method prediction accuracy against ground truth."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils.io import mkdirp

logger = logging.getLogger("context_engineering.evaluation.method_matcher")


def is_method_in_ground_truth(predicted_method: str, ground_truth_methods: list) -> bool:
    """Check if a predicted method matches any ground truth method (suffix match)."""
    if '.' in predicted_method:
        predicted_method = predicted_method.split(".")[-1]
    for gt_method in ground_truth_methods:
        if gt_method.endswith(predicted_method):
            return True
    return False


def process_repository(bug_report_path: str, ground_truth_path: str) -> Tuple[Optional[list], int, int, list]:
    """Process a single repository and compute method match metrics."""
    if not os.path.exists(bug_report_path) or not os.path.exists(ground_truth_path):
        logger.warning(f"Skipping {bug_report_path} or {ground_truth_path} — file not found")
        return None, 0, 0, []

    with open(bug_report_path, 'r') as f:
        bug_reports = json.load(f)
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    results = []
    total_bug_reports = 0
    total_matched = 0
    missing_list = []

    for report in bug_reports:
        instance_id = report.get("instance_id", "")
        if "bug_report" not in report or "problem_location" not in report.get("bug_report", {}):
            missing_list.append(instance_id)
            continue

        predicted_methods = report["bug_report"]["problem_location"].get("methods", [])
        ground_truth_methods = ground_truth.get(instance_id, [])

        total_bug_reports += 1
        matched = [m for m in predicted_methods if is_method_in_ground_truth(m, ground_truth_methods)]
        if matched:
            total_matched += 1

        results.append({
            "instance_id": instance_id,
            "total_predicted": len(predicted_methods),
            "total_ground_truth": len(ground_truth_methods),
            "matched_methods": matched,
            "match_percentage": len(matched) / len(predicted_methods) if predicted_methods else 0,
        })

    return results, total_bug_reports, total_matched, missing_list


def run_evaluation(bug_reports_path: str, ground_truth_path: str, output_path: str):
    """Run method match evaluation and save results."""
    results, total, matched, missing = process_repository(bug_reports_path, ground_truth_path)

    if results is None:
        logger.error("Evaluation failed — input files not found")
        return

    mkdirp(os.path.dirname(output_path) or ".")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total processed: {total}, Matched: {matched}, Missing problem_location: {len(missing)}")
