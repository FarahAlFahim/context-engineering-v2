#!/usr/bin/env python3
"""
Localization Accuracy Evaluation Script

Compares predicted problem_location (classes, methods) from bug reports
against ground-truth locations extracted from patches.

Granularities: class-level, method-level.

Metrics:
  - Hit       : 1 if ANY predicted item matches any GT item, else 0.
  - Coverage  : 1 if ALL ground-truth items are found in predictions, else 0
                (all-or-nothing).  Extra predictions beyond GT count are fine.
  - Precision : TP / |Pred|  — fraction of predictions that are correct.
  - Recall    : TP / |GT|    — fraction of ground-truth items found.
  - F1        : harmonic mean of Precision and Recall.

Usage (single repo):
    python src/evaluate_localization.py \
        --ground_truth data/by_repo/astropy__astropy.json \
        --predictions data/output/gpt-5-mini_fix/multiagent_enhanced/astropy__astropy.json

Usage (single repo, custom output path):
    python src/evaluate_localization.py \
        --ground_truth data/by_repo/astropy__astropy.json \
        --predictions data/output/gpt-5-mini_fix/multiagent_enhanced/astropy__astropy.json \
        --output data/output/localization/gpt-5-mini_fix/multiagent_enhanced/astropy__astropy.json

Usage (all repos in a directory):
    python src/evaluate_localization.py \
        --ground_truth_dir data/by_repo \
        --predictions_dir data/output/problem_location/minimax2.5 \
        --output data/output/localization/problem_location/minimax2.5/overall.json
    
    python src/evaluate_localization.py \
        --ground_truth_dir data/by_repo \
        --predictions_dir data/output/problem_location/gpt_5_mini \
        --output data/output/localization/gpt_5_mini/overall.json

Output:
    JSON results are always written to a file. If --output is not specified,
    the path is auto-derived from the predictions path:
        data/output/<variant>/<repo>.json
        -> data/output/localization/<variant>/<repo>.json
"""

import argparse
import json
import re
import sys
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

def compute_hit(pred_list: list, gt_list: list) -> int:
    """1 if ANY prediction matches any GT item. None if GT is empty."""
    if not gt_list:
        return None
    if not pred_list:
        return 0
    for pred in pred_list:
        for gt in gt_list:
            if names_match(pred, gt):
                return 1
    return 0


def compute_coverage(pred_list: list, gt_list: list) -> int:
    """1 if ALL GT items are found in predictions. None if GT is empty."""
    if not gt_list:
        return None
    for gt in gt_list:
        if not any(names_match(p, gt) for p in pred_list):
            return 0
    return 1


def _count_tp(pred_list: list, gt_list: list) -> int:
    """Count true positives (GT items matched by at least one prediction)."""
    return sum(1 for gt in gt_list
               if any(names_match(p, gt) for p in pred_list))


def compute_set_precision(pred_list: list, gt_list: list) -> float | None:
    """TP / |Pred|. None if GT is empty."""
    if not gt_list:
        return None
    if not pred_list:
        return 0.0
    tp = _count_tp(pred_list, gt_list)
    return tp / len(pred_list)


def compute_recall(pred_list: list, gt_list: list) -> float | None:
    """TP / |GT|. None if GT is empty."""
    if not gt_list:
        return None
    tp = _count_tp(pred_list, gt_list)
    return tp / len(gt_list)


def compute_f1(pred_list: list, gt_list: list) -> float | None:
    """Harmonic mean of precision and recall. None if GT is empty."""
    p = compute_set_precision(pred_list, gt_list)
    r = compute_recall(pred_list, gt_list)
    if p is None:
        return None
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


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
        'class_hit': compute_hit(pred['classes'], gt_classes),
        'class_coverage': compute_coverage(pred['classes'], gt_classes),
        'class_precision': compute_set_precision(pred['classes'], gt_classes),
        'class_recall': compute_recall(pred['classes'], gt_classes),
        'class_f1': compute_f1(pred['classes'], gt_classes),
        'method_hit': compute_hit(pred['methods'], gt_methods),
        'method_coverage': compute_coverage(pred['methods'], gt_methods),
        'method_precision': compute_set_precision(pred['methods'], gt_methods),
        'method_recall': compute_recall(pred['methods'], gt_methods),
        'method_f1': compute_f1(pred['methods'], gt_methods),
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
                h = r[f'{level}_hit']
                cov = r[f'{level}_coverage']
                prec = r[f'{level}_precision']
                rec = r[f'{level}_recall']
                f1 = r[f'{level}_f1']
                gt_n = c[f'{level}_gt']
                pr_n = c[f'{level}_pred']
                if h is None:
                    print(f"    {level:<8}  (no GT)")
                else:
                    h_str = "Y" if h == 1 else "-"
                    cov_str = "Y" if cov == 1 else "-"
                    print(f"    {level:<8}  hit={h_str}  cov={cov_str}"
                          f"  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}"
                          f"  (GT={gt_n}, Pred={pr_n})")

    return results


METRIC_NAMES = ['hit', 'coverage', 'precision', 'recall', 'f1']


def aggregate_results(all_results: list) -> dict:
    if not all_results:
        return {}

    summary = {}
    for level in LEVELS:
        for metric in METRIC_NAMES:
            metric_key = f'{level}_{metric}'
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


def print_instance_details(all_results: list):
    """Print per-instance GT vs Pred comparison (method-level focus, class fallback)."""
    for res in all_results:
        iid = res['instance_id']
        r = res['results']
        gt = res['gt']
        pred = res['pred']

        print(f"\n  {iid}")
        for level in LEVELS:
            key = f'{level}es' if level == 'class' else f'{level}s'
            gt_items = gt.get(key, gt.get(level + 'es', gt.get(level + 's', [])))
            pred_items = pred.get(key, pred.get(level + 'es', pred.get(level + 's', [])))

            h = r[f'{level}_hit']
            f1 = r[f'{level}_f1']
            cov = r[f'{level}_coverage']

            if h is None:
                status = "skip (no GT)"
            elif cov == 1:
                status = f"PASS (F1={f1:.2f})"
            else:
                prec = r[f'{level}_precision']
                rec = r[f'{level}_recall']
                status = f"P={prec:.2f} R={rec:.2f} F1={f1:.2f}"

            gt_str = ", ".join(gt_items) if gt_items else "(none)"
            pred_str = ", ".join(pred_items) if pred_items else "(none)"
            print(f"    {level:<8}  [{status}]")
            print(f"      GT:   {gt_str}")
            print(f"      Pred: {pred_str}")

            if f1 is not None and f1 < 1.0 and gt_items:
                missed = [g for g in gt_items
                          if not any(names_match(pp, g) for pp in pred_items)]
                if missed:
                    print(f"      Missing: {', '.join(missed)}")


def _collect_empty_instances(all_results: list) -> dict:
    """Find instances with empty GT methods or empty predicted methods."""
    empty_gt = []
    empty_pred = []
    for res in all_results:
        iid = res['instance_id']
        if not res['gt']['methods']:
            empty_gt.append(iid)
        if not res['pred']['methods']:
            empty_pred.append(iid)
    return {'empty_gt_methods': empty_gt, 'empty_pred_methods': empty_pred}


def print_summary(summary: dict, all_results: list = None, label: str = "",
                  show_empty: bool = True):
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

    print()
    header = (f"  {'Level':<10} {'Hit':>12} {'Coverage':>12}"
              f" {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(header)
    print(f"  {'-'*72}")
    for level in LEVELS:
        h = summary.get(f'{level}_hit', {})
        cov = summary.get(f'{level}_coverage', {})
        prec = summary.get(f'{level}_precision', {})
        rec = summary.get(f'{level}_recall', {})
        f1 = summary.get(f'{level}_f1', {})
        def _fmt(m):
            pct = m.get('avg', 0.0)
            s = int(m.get('sum', 0)) if isinstance(m.get('sum', 0), int) else m.get('sum', 0)
            n = m.get('total', 0)
            if isinstance(s, float):
                return f"{pct:>5.1f}% ({n})"
            return f"{pct:>5.1f}% ({s}/{n})"
        print(f"  {level:<10} {_fmt(h)} {_fmt(cov)}"
              f" {_fmt(prec)} {_fmt(rec)} {_fmt(f1)}")

    print()
    print(f"  {'Level':<10} {'Avg GT':>8} {'Avg Pred':>10} {'Total GT':>10} {'Total Pred':>12}")
    print(f"  {'-'*52}")
    for level in LEVELS:
        c = summary.get(f'{level}_counts', {})
        print(f"  {level:<10} {c.get('avg_gt',0):>8.1f} {c.get('avg_pred',0):>10.1f}"
              f" {c.get('total_gt',0):>10} {c.get('total_pred',0):>12}")

    if all_results:
        print(f"\n  --- Per-instance details ---")
        print_instance_details(all_results)

        if show_empty:
            empty = _collect_empty_instances(all_results)
            if empty['empty_gt_methods']:
                print(f"\n  --- Instances with NO ground-truth methods ({len(empty['empty_gt_methods'])}) ---")
                for iid in empty['empty_gt_methods']:
                    print(f"    {iid}")
            if empty['empty_pred_methods']:
                print(f"\n  --- Instances with NO predicted methods ({len(empty['empty_pred_methods'])}) ---")
                for iid in empty['empty_pred_methods']:
                    print(f"    {iid}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _derive_output_path(predictions_path: str) -> str:
    """Auto-derive output path from predictions path.

    data/output/<variant>/<sub>/<repo>.json
    -> data/output/localization/<variant>/<sub>/<repo>.json
    """
    parts = Path(predictions_path).parts
    # Find 'output' in the path and insert 'localization' after it
    try:
        idx = parts.index('output')
        new_parts = parts[:idx + 1] + ('localization',) + parts[idx + 1:]
        return str(Path(*new_parts))
    except ValueError:
        # Fallback: put next to predictions
        p = Path(predictions_path)
        return str(p.parent / 'localization' / p.name)


def _build_output_data(per_repo_summaries: dict, all_results: list,
                       repo_results_map: dict) -> dict:
    output_data = {
        'per_repo': {},
        'per_instance': [],
    }
    def _summarise(summary):
        out = {}
        for k, v in summary.items():
            if k.endswith('_counts'):
                out[k] = v
            else:
                s = v['sum']
                out[k] = {
                    'avg': round(v['avg'], 2),
                    'sum': int(s) if isinstance(s, int) else round(s, 4),
                    'total': v['total'],
                }
        return out

    for repo, summary in per_repo_summaries.items():
        output_data['per_repo'][repo] = _summarise(summary)
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
        output_data['overall'] = _summarise(overall)

    # Build per-repo lists of instances with coverage=1 and hit=1
    coverage_1 = {}
    hit_1 = {}
    for repo, results in repo_results_map.items():
        c1 = []
        h1 = []
        for res in results:
            iid = res['instance_id']
            r = res['results']
            level_coverages = [r[f'{lv}_coverage'] for lv in LEVELS
                               if r[f'{lv}_coverage'] is not None]
            level_hits = [r[f'{lv}_hit'] for lv in LEVELS
                          if r[f'{lv}_hit'] is not None]
            if level_coverages and all(c == 1 for c in level_coverages):
                c1.append(iid)
            if level_hits and all(h == 1 for h in level_hits):
                h1.append(iid)
        coverage_1[repo] = c1
        hit_1[repo] = h1

    output_data['coverage_1_instances'] = coverage_1
    output_data['hit_1_instances'] = hit_1

    # Empty GT / empty prediction instance lists
    empty = _collect_empty_instances(all_results)
    output_data['empty_gt_methods'] = empty['empty_gt_methods']
    output_data['empty_pred_methods'] = empty['empty_pred_methods']

    return output_data


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
                        help='Write JSON results to file (auto-derived if omitted)')

    args = parser.parse_args()

    all_results = []
    per_repo_summaries = {}
    repo_results_map = {}  # repo_name -> list of per-instance results
    output_paths = []  # for auto-deriving per-repo output

    if args.ground_truth and args.predictions:
        results = evaluate_file_pair(args.ground_truth, args.predictions, args.verbose)
        all_results.extend(results)
        repo_name = Path(args.ground_truth).stem
        summary = aggregate_results(results)
        per_repo_summaries[repo_name] = summary
        repo_results_map[repo_name] = results
        print_summary(summary, results, repo_name)
        output_paths.append(args.predictions)

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
                    repo_results_map[repo_name] = results
                    print_summary(summary, results, repo_name)
                    output_paths.append(str(pred_file))
            else:
                print(f"  [SKIP] No prediction file for {gt_file.name}")

    else:
        parser.error('Provide either --ground_truth + --predictions, '
                      'or --ground_truth_dir + --predictions_dir')
        sys.exit(1)

    if len(per_repo_summaries) > 1:
        overall = aggregate_results(all_results)
        print_summary(overall, all_results, "OVERALL")

    # Write output JSON — auto-derive path if not specified
    output_data = _build_output_data(per_repo_summaries, all_results, repo_results_map)

    if args.output:
        out_path = args.output
    elif output_paths:
        out_path = _derive_output_path(output_paths[0])
    else:
        out_path = None

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results written to {out_path}")


if __name__ == '__main__':
    main()
