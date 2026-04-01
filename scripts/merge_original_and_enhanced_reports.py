"""Merge original SWE-bench style reports with enhanced reports.

Given:
- original report JSON: list[dict] with fields including `instance_id` and `problem_statement`
- enhanced report JSON: list[dict] with fields including `instance_id` and `bug_report`

The merge keeps all original entries and fields, but if an enhanced entry exists for the
same `instance_id`, it replaces the original `problem_statement` value with the enhanced
`bug_report` (while keeping the field name `problem_statement`).

One-click usage:
  - Edit ORIGINAL_PATH / ENHANCED_PATH / OUTPUT_PATH below, then run the script.

CLI usage (overrides defaults):
  python scripts/merge_original_and_enhanced_reports.py \
    --original data/by_repo/astropy__astropy.json \
    --enhanced swe_results/gpt_5_mini/astropy__astropy.json \
    --output swe_results/gpt_5_mini/merged_reports/astropy__astropy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# ----------------------
# One-click configuration
# ----------------------
# Set these to the files you want to merge.
ORIGINAL_PATH = Path("data/by_repo/scikit-learn__scikit-learn.json")
# ENHANCED_PATH = Path("swe_results/enhanced/matplotlib__matplotlib.json")
# OUTPUT_PATH = Path("swe_results/enhanced/merged_reports/matplotlib__matplotlib.json")
# ENHANCED_PATH = Path("swe_results/gpt_5_mini_agentless/astropy__astropy.json")
# OUTPUT_PATH = Path("swe_results/gpt_5_mini_agentless/merged_reports/astropy__astropy.json")
# ENHANCED_PATH = Path("swe_results/sweagent_enhanced/astropy__astropy.json")
# OUTPUT_PATH = Path("swe_results/sweagent_enhanced/merged_reports/astropy__astropy.json")
ENHANCED_PATH = Path("swe_results/multiagent_enhanced/scikit-learn__scikit-learn.json")
OUTPUT_PATH = Path("swe_results/multiagent_enhanced/without_suggestions/scikit-learn__scikit-learn.json")
# ENHANCED_PATH = Path("swe_results/our_plus_mini_sweagent_enhanced/astropy__astropy.json")
# OUTPUT_PATH = Path("swe_results/our_plus_mini_sweagent_enhanced/merged_reports/astropy__astropy.json")


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list in {path}, got {type(data).__name__}")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(
                f"Expected list elements to be objects in {path}; element {i} is {type(item).__name__}"
            )
    return data


def merge_reports(
    original_reports: List[Dict[str, Any]],
    enhanced_reports: List[Dict[str, Any]],
    *,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Return merged reports.

    If `strict` is True, missing `instance_id` or missing `bug_report` triggers an error.
    Otherwise, problematic enhanced entries are skipped.
    """

    allowed_bug_report_keys = (
        "Title",
        "Description",
        "RootCause",
        "StepsToReproduce",
        "ExpectedBehavior",
        "ObservedBehavior",
        # "Suggestions",
    )

    enhanced_by_instance: Dict[str, Any] = {}
    for item in enhanced_reports:
        iid = item.get("instance_id")
        if not isinstance(iid, str) or not iid:
            if strict:
                raise SystemExit("Enhanced report is missing a valid 'instance_id'.")
            continue

        if "bug_report" not in item:
            if strict:
                raise SystemExit(f"Enhanced report {iid} is missing 'bug_report'.")
            continue

        bug_report = item["bug_report"]
        if not isinstance(bug_report, dict):
            if strict:
                raise SystemExit(f"Enhanced report {iid} has non-object 'bug_report'.")
            continue

        filtered_bug_report = {k: bug_report.get(k) for k in allowed_bug_report_keys if k in bug_report}
        enhanced_by_instance[iid] = filtered_bug_report

    merged: List[Dict[str, Any]] = []
    for orig in original_reports:
        iid = orig.get("instance_id")
        if not isinstance(iid, str) or not iid:
            if strict:
                raise SystemExit("Original report is missing a valid 'instance_id'.")
            merged.append(orig)
            continue

        if iid in enhanced_by_instance:
            # Preserve everything else; only replace `problem_statement`.
            new_obj = dict(orig)
            new_obj["problem_statement"] = enhanced_by_instance[iid]
            merged.append(new_obj)
        else:
            merged.append(orig)

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge an original report JSON with an enhanced report JSON by instance_id. "
            "Replaces original 'problem_statement' with enhanced 'bug_report'."
        )
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=ORIGINAL_PATH,
        help=f"Path to original report JSON (default: {ORIGINAL_PATH})",
    )
    parser.add_argument(
        "--enhanced",
        type=Path,
        default=ENHANCED_PATH,
        help=f"Path to enhanced report JSON (default: {ENHANCED_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Path to write merged report JSON (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing instance_id/bug_report instead of skipping",
    )

    args = parser.parse_args()

    original_reports = _load_json_list(args.original)
    enhanced_reports = _load_json_list(args.enhanced)

    merged = merge_reports(original_reports, enhanced_reports, strict=args.strict)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Merged output has been saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
