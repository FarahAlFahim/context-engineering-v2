#!/usr/bin/env python3
"""
Localization Accuracy Evaluation Script

Compares predicted problem_location (classes, methods) from bug reports
against ground-truth locations extracted from patches.

Granularities: class-level, method-level.

Metrics:
  - Hit@1     : 1 if the first predicted item matches any GT item, else 0.
  - Precision : 1 if ALL ground-truth items are found in predictions, else 0.

Usage:
    python src/evaluate_localization.py \
        --ground_truth data/by_repo/astropy__astropy.json \
        --predictions data/output/gpt-5-mini_fix/multiagent_enhanced/astropy__astropy.json

    python src/evaluate_localization.py \
        --ground_truth_dir data/by_repo \
        --predictions_dir data/output/gpt-5-mini_fix/multiagent_enhanced
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Ground-truth extraction from unified diff patches
# ---------------------------------------------------------------------------

def parse_patch_locations(patch_text: str) -> dict:
    """Extract class and method names that were actually changed in a patch.

    Strategy:
      1. The @@ hunk context line tells us the enclosing scope (the nearest
         class or def *above* the change).  If it's a class, the change is
         inside that class body.  If it's a def, the change is inside that
         function.
      2. Added/removed lines that themselves contain 'def ...' or 'class ...'
         are directly changed definitions — extract those too.
    """
    classes = []
    methods = []
    seen_classes = set()
    seen_methods = set()

    def _add_class(name):
        if name not in seen_classes:
            seen_classes.add(name)
            classes.append(name)

    def _add_method(name):
        if name not in seen_methods:
            seen_methods.add(name)
            methods.append(name)

    # Track enclosing scope.  Within a hunk, context lines (no +/-)
    # can redefine the scope if they contain class/def definitions.
    enclosing_class = None
    enclosing_method = None

    for line in patch_text.splitlines():
        # New file
        if line.startswith('diff --git'):
            enclosing_class = None
            enclosing_method = None
            continue

        # Hunk header — sets initial enclosing scope for this hunk
        hunk_match = re.match(
            r'^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@ ?(.*)', line
        )
        if hunk_match:
            ctx = hunk_match.group(1).strip()
            enclosing_method = None
            enclosing_class = None
            if ctx:
                m = re.match(r'(?:def|async def)\s+(\w+)\s*\(', ctx)
                if m:
                    enclosing_method = m.group(1)
                m = re.match(r'class\s+(\w+)', ctx)
                if m:
                    enclosing_class = m.group(1)
            continue

        if line.startswith('+++') or line.startswith('---'):
            continue

        content = line[1:] if (line.startswith('+') or line.startswith('-')) else line
        is_changed = line.startswith('+') or line.startswith('-')

        # Update enclosing scope from ANY line (context or changed) that
        # defines a class or method.  This handles cases where a context
        # line like " class SimpleRSTData(FixedWidthData):" appears between
        # the hunk header and the actual change.
        m = re.match(r'\s*class\s+(\w+)', content)
        if m:
            enclosing_class = m.group(1)
            enclosing_method = None
            if is_changed:
                _add_class(m.group(1))
            continue

        m = re.match(r'\s*(?:def|async def)\s+(\w+)\s*\(', content)
        if m:
            enclosing_method = m.group(1)
            if is_changed:
                _add_method(m.group(1))
            continue

        if not is_changed:
            continue

        # This is a changed line (not a def/class itself) inside a scope
        if enclosing_method:
            _add_method(enclosing_method)
        elif enclosing_class:
            _add_class(enclosing_class)

    return {'classes': classes, 'methods': methods}


# ---------------------------------------------------------------------------
# Prediction extraction from problem_location field
# ---------------------------------------------------------------------------

def extract_pred_locations(pred_item: dict) -> dict:
    """Extract class and method lists from the problem_location field.

    Normalises dotted names to bare names (last component).
    """
    pl = pred_item.get('problem_location', {})
    if not pl or not isinstance(pl, dict):
        return {'classes': [], 'methods': []}

    raw_classes = pl.get('classes', [])
    raw_methods = pl.get('methods', [])

    classes = _deduplicate_bare(raw_classes)
    methods = _deduplicate_bare(raw_methods)

    return {'classes': classes, 'methods': methods}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare(name: str) -> str:
    """Last dotted component, lowercased."""
    return name.rsplit('.', 1)[-1].lower()


def _deduplicate_bare(items: list) -> list:
    """Keep first occurrence of each bare name, preserving order."""
    seen = set()
    out = []
    for item in items:
        b = _bare(item)
        if b not in seen:
            seen.add(b)
            out.append(item)
    return out


def names_match(predicted: str, ground_truth: str) -> bool:
    return _bare(predicted) == _bare(ground_truth)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_hit1(pred_list: list, gt_list: list) -> int:
    """1 if the first prediction matches any GT. None if GT is empty."""
    if not gt_list:
        return None
    if not pred_list:
        return 0
    for gt in gt_list:
        if names_match(pred_list[0], gt):
            return 1
    return 0


def compute_precision(pred_list: list, gt_list: list) -> int:
    """1 if ALL GT items are found in predictions. None if GT is empty."""
    if not gt_list:
        return None
    for gt in gt_list:
        if not any(names_match(p, gt) for p in pred_list):
            return 0
    return 1


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

LEVELS = ['class', 'method']


def evaluate_instance(gt_item: dict, pred_item: dict) -> dict:
    gt = parse_patch_locations(gt_item['patch'])
    pred = extract_pred_locations(pred_item)

    gt_classes = _deduplicate_bare(gt['classes'])
    gt_methods = _deduplicate_bare(gt['methods'])

    results = {
        'class_hit@1': compute_hit1(pred['classes'], gt_classes),
        'class_precision': compute_precision(pred['classes'], gt_classes),
        'method_hit@1': compute_hit1(pred['methods'], gt_methods),
        'method_precision': compute_precision(pred['methods'], gt_methods),
    }

    counts = {
        'class_gt': len(gt_classes),
        'class_pred': len(pred['classes']),
        'method_gt': len(gt_methods),
        'method_pred': len(pred['methods']),
    }

    return {
        'instance_id': gt_item['instance_id'],
        'results': results,
        'counts': counts,
        'gt': {'classes': gt_classes, 'methods': gt_methods},
        'pred': {'classes': pred['classes'], 'methods': pred['methods']},
    }


# ---------------------------------------------------------------------------
# Aggregation & display
# ---------------------------------------------------------------------------

def evaluate_file_pair(gt_path: str, pred_path: str, verbose: bool = False) -> list:
    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    gt_by_id = {item['instance_id']: item for item in gt_data}
    pred_by_id = {item['instance_id']: item for item in pred_data}
    matched_ids = sorted(set(gt_by_id) & set(pred_by_id))

    results = []
    for iid in matched_ids:
        res = evaluate_instance(gt_by_id[iid], pred_by_id[iid])
        results.append(res)

        if verbose:
            r = res['results']
            c = res['counts']
            print(f"\n{'='*80}")
            print(f"Instance: {iid}")
            print(f"  GT  classes: {res['gt']['classes']}")
            print(f"  Pred classes: {res['pred']['classes']}")
            print(f"  GT  methods: {res['gt']['methods']}")
            print(f"  Pred methods: {res['pred']['methods']}")
            for level in LEVELS:
                h = r[f'{level}_hit@1']
                p = r[f'{level}_precision']
                h_str = "Y" if h == 1 else ("-" if h == 0 else "skip")
                p_str = "Y" if p == 1 else ("-" if p == 0 else "skip")
                gt_n = c[f'{level}_gt']
                pr_n = c[f'{level}_pred']
                print(f"    {level:<8}  hit@1={h_str}  precision={p_str}"
                      f"  (GT={gt_n}, Pred={pr_n})")

    return results


def aggregate_results(all_results: list) -> dict:
    if not all_results:
        return {}

    summary = {}
    for level in LEVELS:
        for metric_key in [f'{level}_hit@1', f'{level}_precision']:
            values = [r['results'][metric_key] for r in all_results
                      if r['results'][metric_key] is not None]
            n = len(values)
            s = sum(values) if values else 0
            summary[metric_key] = {
                'sum': s, 'total': n,
                'avg': (s / n * 100) if n > 0 else 0.0,
            }

        gt_counts = [r['counts'][f'{level}_gt'] for r in all_results]
        pred_counts = [r['counts'][f'{level}_pred'] for r in all_results]
        summary[f'{level}_counts'] = {
            'total_gt': sum(gt_counts),
            'total_pred': sum(pred_counts),
            'avg_gt': sum(gt_counts) / len(gt_counts),
            'avg_pred': sum(pred_counts) / len(pred_counts),
        }

    return summary


def print_summary(summary: dict, label: str = ""):
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

    print()
    print(f"  {'Level':<10} {'Hit@1':>12} {'Precision':>12}")
    print(f"  {'-'*36}")
    for level in LEVELS:
        h = summary.get(f'{level}_hit@1', {})
        p = summary.get(f'{level}_precision', {})
        h_pct, h_s, h_n = h.get('avg', 0.0), int(h.get('sum', 0)), h.get('total', 0)
        p_pct, p_s, p_n = p.get('avg', 0.0), int(p.get('sum', 0)), p.get('total', 0)
        print(f"  {level:<10} {h_pct:>5.1f}% ({h_s}/{h_n}) {p_pct:>5.1f}% ({p_s}/{p_n})")

    print()
    print(f"  {'Level':<10} {'Avg GT':>8} {'Avg Pred':>10} {'Total GT':>10} {'Total Pred':>12}")
    print(f"  {'-'*52}")
    for level in LEVELS:
        c = summary.get(f'{level}_counts', {})
        print(f"  {level:<10} {c.get('avg_gt',0):>8.1f} {c.get('avg_pred',0):>10.1f}"
              f" {c.get('total_gt',0):>10} {c.get('total_pred',0):>12}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate bug localization accuracy')
    parser.add_argument('--ground_truth', type=str,
                        help='Path to ground truth JSON (single repo)')
    parser.add_argument('--predictions', type=str,
                        help='Path to predictions JSON (single repo)')
    parser.add_argument('--ground_truth_dir', type=str,
                        help='Directory of ground truth JSONs (by_repo/)')
    parser.add_argument('--predictions_dir', type=str,
                        help='Directory of prediction JSONs')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print per-instance details')
    parser.add_argument('--output', '-o', type=str,
                        help='Write JSON results to file')

    args = parser.parse_args()

    all_results = []
    per_repo_summaries = {}

    if args.ground_truth and args.predictions:
        results = evaluate_file_pair(args.ground_truth, args.predictions, args.verbose)
        all_results.extend(results)
        repo_name = Path(args.ground_truth).stem
        summary = aggregate_results(results)
        per_repo_summaries[repo_name] = summary
        print_summary(summary, repo_name)

    elif args.ground_truth_dir and args.predictions_dir:
        gt_dir = Path(args.ground_truth_dir)
        pred_dir = Path(args.predictions_dir)

        for gt_file in sorted(gt_dir.glob('*.json')):
            pred_file = pred_dir / gt_file.name
            if pred_file.exists():
                repo_name = gt_file.stem
                results = evaluate_file_pair(str(gt_file), str(pred_file), args.verbose)
                if results:
                    all_results.extend(results)
                    summary = aggregate_results(results)
                    per_repo_summaries[repo_name] = summary
                    print_summary(summary, repo_name)
            else:
                print(f"  [SKIP] No prediction file for {gt_file.name}")

    else:
        parser.error('Provide either --ground_truth + --predictions, '
                      'or --ground_truth_dir + --predictions_dir')
        sys.exit(1)

    if len(per_repo_summaries) > 1:
        overall = aggregate_results(all_results)
        print_summary(overall, "OVERALL")

    if args.output:
        output_data = {
            'per_repo': {},
            'per_instance': [],
        }
        for repo, summary in per_repo_summaries.items():
            repo_out = {}
            for k, v in summary.items():
                if k.endswith('_counts'):
                    repo_out[k] = v
                else:
                    repo_out[k] = {'avg': round(v['avg'], 2),
                                   'count': int(v['sum']),
                                   'total': v['total']}
            output_data['per_repo'][repo] = repo_out
        for res in all_results:
            output_data['per_instance'].append({
                'instance_id': res['instance_id'],
                'results': res['results'],
                'counts': res['counts'],
                'ground_truth': res['gt'],
                'predicted': res['pred'],
            })
        if len(per_repo_summaries) > 1:
            overall = aggregate_results(all_results)
            overall_out = {}
            for k, v in overall.items():
                if k.endswith('_counts'):
                    overall_out[k] = v
                else:
                    overall_out[k] = {'avg': round(v['avg'], 2),
                                      'count': int(v['sum']),
                                      'total': v['total']}
            output_data['overall'] = overall_out

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results written to {args.output}")


if __name__ == '__main__':
    main()
