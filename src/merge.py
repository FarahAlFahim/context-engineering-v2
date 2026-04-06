"""Merge original SWE-bench reports with enhanced reports."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import mkdirp

logger = logging.getLogger("context_engineering.merge")

ALLOWED_BUG_REPORT_KEYS = (
    "Title", "Description", "RootCause", "StepsToReproduce",
    "ExpectedBehavior", "ObservedBehavior",
)


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")
    return data


def merge_reports(original_reports: List[Dict[str, Any]],
                   enhanced_reports: List[Dict[str, Any]],
                   strict: bool = False) -> List[Dict[str, Any]]:
    """Merge original reports with enhanced bug_report fields."""
    enhanced_by_instance: Dict[str, Any] = {}
    for item in enhanced_reports:
        iid = item.get("instance_id")
        if not isinstance(iid, str) or not iid:
            if strict:
                raise ValueError("Enhanced report missing valid 'instance_id'")
            continue
        if "bug_report" not in item:
            if strict:
                raise ValueError(f"Enhanced report {iid} missing 'bug_report'")
            continue

        bug_report = item["bug_report"]
        if not isinstance(bug_report, dict):
            if strict:
                raise ValueError(f"Enhanced report {iid} has non-object 'bug_report'")
            continue

        filtered = {k: bug_report.get(k) for k in ALLOWED_BUG_REPORT_KEYS if k in bug_report}
        enhanced_by_instance[iid] = filtered

    merged = []
    for orig in original_reports:
        iid = orig.get("instance_id")
        if isinstance(iid, str) and iid in enhanced_by_instance:
            new_obj = dict(orig)
            new_obj["problem_statement"] = enhanced_by_instance[iid]
            merged.append(new_obj)
        else:
            merged.append(orig)

    return merged


def run_merge(original_path: str, enhanced_path: str, output_path: str,
               strict: bool = False):
    """Run the merge pipeline."""
    original = _load_json_list(Path(original_path))
    enhanced = _load_json_list(Path(enhanced_path))

    merged = merge_reports(original, enhanced, strict=strict)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info(f"Merged {len(merged)} reports -> {output_path}")
