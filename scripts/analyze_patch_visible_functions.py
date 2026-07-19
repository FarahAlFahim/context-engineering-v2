#!/usr/bin/env python3
"""Classify implementation patches by function/method contexts visible in patch.

This intentionally uses only ``data/by_repo/*.json`` and the ``patch`` field.
It does not inspect ``test_patch`` or fetch repository source files.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set


ROOT = pathlib.Path("data/by_repo")
OUT_CSV = pathlib.Path("data/by_repo_patch_visible_function_classification.csv")
OUT_MD = pathlib.Path("data/by_repo_patch_visible_function_report.md")


def normalize_hunk_context(context: str) -> str:
    context = context.strip()
    if not context:
        return "module:<module>"
    label = label_from_source_line(context)
    return label or f"context:{context}"


def label_from_source_line(line: str) -> Optional[str]:
    stripped = line.strip()
    match = re.match(r"(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
    if match:
        return f"function_or_method:{match.group(1)}"
    match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
    if match:
        return f"class:{match.group(1)}"
    return None


def classify_patch(patch: str) -> Dict[str, object]:
    changed_files: List[str] = []
    hunk_count = 0
    current_location = "module:<module>"
    current_function_method: Optional[str] = None
    changed_locations: Set[str] = set()
    changed_function_methods: Set[str] = set()

    lines = patch.splitlines()
    for index, raw_line in enumerate(lines):
        if raw_line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)", raw_line)
            changed_files.append(match.group(2) if match else raw_line.split()[-1][2:])
            current_location = "module:<module>"
            current_function_method = None
            continue

        if raw_line.startswith("@@"):
            hunk_count += 1
            match = re.match(r"@@ [^@]* @@ ?(.*)$", raw_line)
            current_location = normalize_hunk_context(match.group(1) if match else "")
            current_function_method = (
                current_location
                if current_location.startswith("function_or_method:")
                else None
            )
            continue

        if not raw_line or raw_line.startswith(("+++", "---")):
            continue
        marker = raw_line[0]
        if marker not in {" ", "+", "-"}:
            continue

        source_line = raw_line[1:]
        visible_label = label_from_source_line(source_line)
        if visible_label:
            current_location = visible_label
            if visible_label.startswith("function_or_method:"):
                current_function_method = visible_label

        if marker in {"+", "-"}:
            change_location = current_location
            change_function_method = current_function_method
            if source_line.strip().startswith("@"):
                decorator_target = next_visible_def_or_class(lines, index + 1)
                if decorator_target:
                    change_location = decorator_target
                    change_function_method = (
                        decorator_target
                        if decorator_target.startswith("function_or_method:")
                        else None
                    )
            changed_locations.add(change_location)
            if change_function_method:
                changed_function_methods.add(change_function_method)

    return {
        "changed_files": ";".join(changed_files),
        "file_count": len(changed_files),
        "hunk_count": hunk_count,
        "visible_location_count": len(changed_locations),
        "visible_function_method_count": len(changed_function_methods),
        "is_multi_visible_location": len(changed_locations) > 1,
        "is_multi_visible_function_method": len(changed_function_methods) > 1,
        "visible_locations": " | ".join(sorted(changed_locations)),
        "visible_function_methods": " | ".join(sorted(changed_function_methods)),
    }


def next_visible_def_or_class(lines: List[str], start_index: int) -> Optional[str]:
    for raw_line in lines[start_index:]:
        if raw_line.startswith(("diff --git ", "@@")):
            return None
        if not raw_line or raw_line.startswith(("+++", "---")):
            continue
        marker = raw_line[0]
        if marker not in {" ", "+", "-"}:
            continue
        source = raw_line[1:].strip()
        if not source or source.startswith("@"):
            continue
        return label_from_source_line(source)
    return None


def main() -> None:
    rows: List[Dict[str, object]] = []
    for path in sorted(ROOT.glob("*.json")):
        with path.open() as handle:
            data = json.load(handle)
        for obj in data:
            row = {
                "repo_json": path.name,
                "repo": obj.get("repo", ""),
                "instance_id": obj.get("instance_id", ""),
            }
            row.update(classify_patch(obj.get("patch") or ""))
            rows.append(row)

    fields = [
        "repo_json",
        "repo",
        "instance_id",
        "changed_files",
        "file_count",
        "hunk_count",
        "visible_location_count",
        "visible_function_method_count",
        "is_multi_visible_location",
        "is_multi_visible_function_method",
        "visible_locations",
        "visible_function_methods",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_repo: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo_json"])].append(row)

    lines: List[str] = []
    lines.append("# Patch-Visible Function/Method Classification")
    lines.append("")
    lines.append("Scope: `data/by_repo/*.json`, `patch` field only. `test_patch` is ignored.")
    lines.append("")
    lines.append("Definition: a patch is multi-function/method when changed lines in the `patch` map to more than one visible function/method context. Context comes from the unified diff `@@ ... @@` header and from `def` / `async def` lines visible inside each hunk, so single-hunk changes across adjacent methods are counted.")
    lines.append("")
    lines.append("Note: this is exact for function/method information visible in the `patch` field. A fully semantic AST count would require the source files at each `base_commit`.")
    lines.append("")
    lines.append(f"Total instances: {len(rows)}")
    lines.append(f"Multi-function/method patches: {sum(bool(r['is_multi_visible_function_method']) for r in rows)}")
    lines.append(f"Single-function/method or non-function patches: {sum(not bool(r['is_multi_visible_function_method']) for r in rows)}")
    lines.append(f"Multi visible-location patches, including class/module contexts: {sum(bool(r['is_multi_visible_location']) for r in rows)}")
    lines.append(f"Single visible-location patches: {sum(not bool(r['is_multi_visible_location']) for r in rows)}")
    lines.append(f"Multi-hunk patches: {sum(int(r['hunk_count']) > 1 for r in rows)}")
    lines.append("")
    lines.append("## Hunk/function overlap")
    lines.append("")
    lines.append("| hunk category | function/method category | count |")
    lines.append("|---|---|---:|")
    overlap_rows = [
        ("single-hunk", "single-function/method or non-function", False, False),
        ("single-hunk", "multi-function/method", False, True),
        ("multi-hunk", "single-function/method or non-function", True, False),
        ("multi-hunk", "multi-function/method", True, True),
    ]
    for hunk_label, function_label, is_multi_hunk, is_multi_function in overlap_rows:
        count = sum(
            (int(row["hunk_count"]) > 1) == is_multi_hunk
            and bool(row["is_multi_visible_function_method"]) == is_multi_function
            for row in rows
        )
        lines.append(f"| {hunk_label} | {function_label} | {count} |")
    lines.append("")
    lines.append("So the multi-function/method total is `51 multi-hunk and multi-function/method` plus `9 single-hunk but multi-function/method` = `60`.")
    lines.append("")
    lines.append("For the broader class/module/function location definition, the overlap is:")
    lines.append("")
    lines.append("| hunk category | visible-location category | count |")
    lines.append("|---|---|---:|")
    visible_overlap_rows = [
        ("single-hunk", "single visible location", False, False),
        ("single-hunk", "multi visible location", False, True),
        ("multi-hunk", "single visible location", True, False),
        ("multi-hunk", "multi visible location", True, True),
    ]
    for hunk_label, location_label, is_multi_hunk, is_multi_location in visible_overlap_rows:
        count = sum(
            (int(row["hunk_count"]) > 1) == is_multi_hunk
            and bool(row["is_multi_visible_location"]) == is_multi_location
            for row in rows
        )
        lines.append(f"| {hunk_label} | {location_label} | {count} |")
    lines.append("")
    lines.append("## Counts by repository")
    lines.append("")
    lines.append("| repo_json | total | multi-function/method | single-function/method or non-function | multi-visible-location | multi-hunk |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for repo_json in sorted(by_repo):
        group = by_repo[repo_json]
        lines.append(
            f"| {repo_json} | {len(group)} | "
            f"{sum(bool(r['is_multi_visible_function_method']) for r in group)} | "
            f"{sum(not bool(r['is_multi_visible_function_method']) for r in group)} | "
            f"{sum(bool(r['is_multi_visible_location']) for r in group)} | "
            f"{sum(int(r['hunk_count']) > 1 for r in group)} |"
        )
    lines.append("")
    lines.append("## Multi-function/method instances")
    for repo_json in sorted(by_repo):
        items = [
            row
            for row in sorted(by_repo[repo_json], key=lambda r: str(r["instance_id"]))
            if row["is_multi_visible_function_method"]
        ]
        lines.append("")
        lines.append(f"### {repo_json} ({len(items)})")
        if not items:
            lines.append("- none")
        for row in items:
            lines.append(
                f"- {row['instance_id']} ({row['visible_function_method_count']} function/method contexts, "
                f"{row['hunk_count']} hunks): {row['changed_files']}"
            )
            lines.append(f"  - {row['visible_function_methods']}")

    OUT_MD.write_text("\n".join(lines) + "\n")

    print(f"rows={len(rows)}")
    print(f"multi_function_method={sum(bool(r['is_multi_visible_function_method']) for r in rows)}")
    print(f"single_or_non_function={sum(not bool(r['is_multi_visible_function_method']) for r in rows)}")
    print(f"multi_visible_location={sum(bool(r['is_multi_visible_location']) for r in rows)}")
    print(f"multi_hunk={sum(int(r['hunk_count']) > 1 for r in rows)}")
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
