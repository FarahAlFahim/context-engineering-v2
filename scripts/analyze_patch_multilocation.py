#!/usr/bin/env python3
"""Classify implementation patches as single-location or multi-location.

This script uses only data/by_repo/*.json and only the ``patch`` field.

Operational definition:
- A location is a contiguous changed-line cluster in the implementation diff.
  Consecutive +/- lines are one cluster; an unchanged/context line starts a new
  cluster.
- A patch is multi-location if it has more than one changed-line cluster.
- A single cluster can still be multi-location when manual audit shows the
  contiguous block introduces/changes multiple semantic code units.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List


ROOT = pathlib.Path("data/by_repo")
OUT_CSV = pathlib.Path("data/by_repo_patch_manual_multilocation_audit.csv")
OUT_MD = pathlib.Path("data/by_repo_patch_manual_multilocation_audit.md")


SINGLE_CLUSTER_SEMANTIC_MULTI: Dict[str, str] = {
    "django__django-15400": (
        "Single contiguous block adds class-level __add__ method proxy and "
        "__radd__ method; broader method-behavior multi-location."
    ),
    "matplotlib__matplotlib-25332": (
        "Single contiguous block adds two methods: __getstate__ and __setstate__."
    ),
    "sympy__sympy-11400": (
        "Single contiguous block adds two printer methods: _print_Relational "
        "and _print_sinc."
    ),
    "sympy__sympy-16106": (
        "Single contiguous block adds three printer methods: _print_tuple, "
        "_print_IndexedBase, and _print_Indexed."
    ),
}


def edit_cluster_count(patch: str) -> int:
    clusters = 0
    in_cluster = False
    for line in patch.splitlines():
        is_change = line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        if is_change:
            in_cluster = True
        elif in_cluster:
            clusters += 1
            in_cluster = False
    if in_cluster:
        clusters += 1
    return clusters


def hunk_count(patch: str) -> int:
    return sum(1 for line in patch.splitlines() if line.startswith("@@"))


def changed_files(patch: str) -> List[str]:
    files: List[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.*?) b/(.*)", line)
            files.append(match.group(2) if match else line.split()[-1][2:])
    return files


def classify_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(ROOT.glob("*.json")):
        with path.open() as handle:
            data = json.load(handle)
        for obj in data:
            patch = obj.get("patch") or ""
            instance_id = obj["instance_id"]
            clusters = edit_cluster_count(patch)
            multi = clusters > 1 or instance_id in SINGLE_CLUSTER_SEMANTIC_MULTI
            if clusters > 1:
                reason = f"{clusters} separated edit clusters in patch"
            elif instance_id in SINGLE_CLUSTER_SEMANTIC_MULTI:
                reason = SINGLE_CLUSTER_SEMANTIC_MULTI[instance_id]
            else:
                reason = "single contiguous edit region affecting one semantic code location"
            rows.append(
                {
                    "repo_json": path.name,
                    "repo": obj.get("repo", ""),
                    "instance_id": instance_id,
                    "changed_files": ";".join(changed_files(patch)),
                    "hunk_count": hunk_count(patch),
                    "edit_cluster_count": clusters,
                    "manual_multi_location": multi,
                    "manual_location_reason": reason,
                }
            )
    return rows


def write_csv(rows: List[Dict[str, object]]) -> None:
    fields = [
        "repo_json",
        "repo",
        "instance_id",
        "changed_files",
        "hunk_count",
        "edit_cluster_count",
        "manual_multi_location",
        "manual_location_reason",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, object]]) -> None:
    by_repo: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_repo[str(row["repo_json"])].append(row)

    multi_count = sum(bool(row["manual_multi_location"]) for row in rows)
    cluster_multi = sum(int(row["edit_cluster_count"]) > 1 for row in rows)

    lines: List[str] = []
    lines.append("# Manual Patch Multi-Location Audit")
    lines.append("")
    lines.append("Scope: `data/by_repo/*.json`, `patch` field only. `test_patch` is ignored.")
    lines.append("")
    lines.append("Definition: a patch is multi-location if the fix changes more than one distinct place in the implementation code. A distinct place can be a separate edit cluster, a separate hunk, another method/function, class-level state plus method code, module-level code plus another code region, or multiple semantic code units inside one contiguous edit block.")
    lines.append("")
    lines.append("Counting rule used for the audit:")
    lines.append("- Multiple separated edit clusters in the unified diff are multi-location, even if they occur inside the same method/function.")
    lines.append("- A single contiguous edit cluster is still multi-location when it visibly changes/adds multiple semantic code units, such as multiple methods or a method-proxy assignment plus a method.")
    lines.append("- A large contiguous rewrite inside one method/function is counted as single-location unless the contiguous block itself introduces/changes multiple semantic code units.")
    lines.append("")
    lines.append(f"Total instances: {len(rows)}")
    lines.append(f"Manual multi-location patches: {multi_count}")
    lines.append(f"Manual single-location patches: {len(rows) - multi_count}")
    lines.append("")
    lines.append("## Count Decomposition")
    lines.append("")
    lines.append(f"- Patches with more than one separated edit cluster: {cluster_multi}")
    lines.append(f"- Single-cluster patches manually counted as multi-location due to multiple semantic code units: {len(SINGLE_CLUSTER_SEMANTIC_MULTI)}")
    lines.append(f"- Final multi-location total: {cluster_multi} + {len(SINGLE_CLUSTER_SEMANTIC_MULTI)} = {multi_count}")
    lines.append("")
    lines.append("## Counts by repository")
    lines.append("")
    lines.append("| repo_json | total | multi-location | single-location |")
    lines.append("|---|---:|---:|---:|")
    for repo_json in sorted(by_repo):
        group = by_repo[repo_json]
        repo_multi = sum(bool(row["manual_multi_location"]) for row in group)
        lines.append(f"| {repo_json} | {len(group)} | {repo_multi} | {len(group) - repo_multi} |")
    lines.append("")
    lines.append("## Single-Cluster Semantic Multi-Location Cases")
    lines.append("")
    for instance_id, note in SINGLE_CLUSTER_SEMANTIC_MULTI.items():
        lines.append(f"- {instance_id}: {note}")
    lines.append("")
    lines.append("## Multi-Location Instances")
    for repo_json in sorted(by_repo):
        items = [
            row
            for row in sorted(by_repo[repo_json], key=lambda item: str(item["instance_id"]))
            if row["manual_multi_location"]
        ]
        lines.append("")
        lines.append(f"### {repo_json} ({len(items)})")
        if not items:
            lines.append("- none")
        for row in items:
            lines.append(
                f"- {row['instance_id']} ({row['hunk_count']} hunks, "
                f"{row['edit_cluster_count']} edit clusters): {row['changed_files']}"
            )
            lines.append(f"  - {row['manual_location_reason']}")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = classify_rows()
    write_csv(rows)
    write_markdown(rows)
    print(f"rows={len(rows)}")
    print(f"multi_location={sum(bool(row['manual_multi_location']) for row in rows)}")
    print(f"single_location={sum(not bool(row['manual_multi_location']) for row in rows)}")
    print(f"edit_cluster_multi={sum(int(row['edit_cluster_count']) > 1 for row in rows)}")
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
