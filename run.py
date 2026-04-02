#!/usr/bin/env python3
"""Main CLI entry point for the Context Engineering pipeline.

Usage:
    python run.py <phase> [options]

Phases:
    build_graphs          Build code and call graphs from a local repo
    enhance               Generate enhanced bug reports (multi-agent + reviewer)
    trajectory_enhance    Further enhance using trajectory insights
    dynamic_enhance       Iterative enhancement with mini-sweagent feedback
    merge                 Merge original + enhanced reports
    evaluate              Run method-match evaluation
    generate_patches      Run mini-swe-agent to generate patches from reports
    eval_patches          Evaluate generated patches against SWE-bench tests

Examples:
    # Build code graphs
    python run.py build_graphs \\
        --repo-instances data/by_repo/django__django.json \\
        --repo-local-path /path/to/django \\
        --git-branch main

    # Generate enhanced reports (saves both pre-reviewer and post-reviewer)
    # Pre-reviewer output auto-saved to: data/output/enhanced/astropy__astropy.json
    # Post-reviewer output saved to:     --output path
    python run.py enhance \\
        --repo-instances data/by_repo/astropy__astropy.json \\
        --repo-codegraph-index data/code_graph/astropy__astropy.json \\
        --repo-local-path /path/to/astropy \\
        --output data/output/multiagent_enhanced/astropy__astropy.json

    # Override the pre-reviewer output path explicitly:
    python run.py enhance \\
        --repo-instances data/by_repo/astropy__astropy.json \\
        --repo-codegraph-index data/code_graph/astropy__astropy.json \\
        --repo-local-path /path/to/astropy \\
        --output data/output/multiagent_enhanced/astropy__astropy.json \\
        --single-enhanced-file data/output/enhanced/custom_name.json

    # Further enhance with trajectory insights
    python run.py trajectory_enhance \\
        --repo-instances data/output/multiagent_enhanced/matplotlib__matplotlib.json \\
        --repo-codegraph-index data/code_graph/matplotlib__matplotlib.json \\
        --repo-local-path /path/to/matplotlib \\
        --original-instances data/by_repo/matplotlib__matplotlib.json \\
        --trajectory-summary /path/to/traj_summary.json \\
        --output data/output/trajectory_enhanced/matplotlib__matplotlib.json

    # Dynamic iterative enhancement
    python run.py dynamic_enhance \\
        --repo-instances data/output/multiagent_enhanced/matplotlib__matplotlib.json \\
        --repo-codegraph-index data/code_graph/matplotlib__matplotlib.json \\
        --repo-local-path /path/to/matplotlib \\
        --original-instances data/by_repo/matplotlib__matplotlib.json \\
        --trajectory-folder /path/to/trajectories \\
        --minisweagent-root /path/to/mini-swe-agent \\
        --output data/output/dynamic_enhanced/matplotlib__matplotlib.json

    # Merge reports
    python run.py merge \\
        --merge-original data/by_repo/django__django.json \\
        --merge-enhanced data/output/multiagent_enhanced/django__django.json \\
        --merge-output data/output/merged/django__django.json

    # Evaluate method matching
    python run.py evaluate \\
        --eval-bug-reports data/output/multiagent_enhanced/astropy__astropy.json \\
        --eval-ground-truth data/ground_truth/astropy__astropy.json \\
        --eval-output data/output/evaluation/astropy__astropy.json

    # Generate patches from merged reports
    python run.py generate_patches \\
        --minisweagent-root /path/to/mini-swe-agent \\
        --patch-dataset data/output/merged/django__django.json \\
        --patch-variant multiagent_enhanced \\
        --patch-run-name django__django_multiagent \\
        --instance-ids django__django-12345

    # Evaluate patches against SWE-bench tests
    python run.py eval_patches \\
        --minisweagent-root /path/to/mini-swe-agent \\
        --eval-targets results/astropy__astropy_multiagent \\
        --eval-variants original multiagent_enhanced \\
        --instance-ids astropy__astropy-12907

    # Filter to specific instances
    python run.py enhance \\
        --repo-instances data/by_repo/astropy__astropy.json \\
        --repo-codegraph-index data/code_graph/astropy__astropy.json \\
        --repo-local-path /path/to/astropy \\
        --output data/output/multiagent_enhanced/astropy__astropy.json \\
        --instance-ids astropy__astropy-14182
"""

import sys

from src.config import load_config
from src.log import setup_logging


def main():
    cfg = load_config()
    logger = setup_logging(log_dir=cfg.log_dir, phase=cfg.phase)

    logger.info(f"Phase: {cfg.phase}")
    logger.info(f"Model: {cfg.openai_model}, Temperature: {cfg.llm_temperature}")

    if cfg.phase == "build_graphs":
        from src.graph.builder import run_graph_builder
        run_graph_builder(cfg)

    elif cfg.phase == "enhance":
        from src.agents.multi_agent import run_pipeline
        run_pipeline(cfg)

    elif cfg.phase == "trajectory_enhance":
        from src.agents.trajectory_insights import run_pipeline
        run_pipeline(cfg)

    elif cfg.phase == "dynamic_enhance":
        from src.agents.dynamic_insights import run_pipeline
        run_pipeline(cfg)

    elif cfg.phase == "merge":
        from src.merge import run_merge
        if not cfg.merge_original or not cfg.merge_enhanced or not cfg.merge_output:
            logger.error("merge requires --merge-original, --merge-enhanced, --merge-output")
            sys.exit(1)
        run_merge(cfg.merge_original, cfg.merge_enhanced, cfg.merge_output, cfg.merge_strict)

    elif cfg.phase == "evaluate":
        from src.evaluation.method_matcher import run_evaluation
        if not cfg.eval_bug_reports or not cfg.eval_ground_truth or not cfg.eval_output:
            logger.error("evaluate requires --eval-bug-reports, --eval-ground-truth, --eval-output")
            sys.exit(1)
        run_evaluation(cfg.eval_bug_reports, cfg.eval_ground_truth, cfg.eval_output)

    elif cfg.phase == "generate_patches":
        from src.patch_generation import run_patch_generation
        run_patch_generation(cfg)

    elif cfg.phase == "eval_patches":
        from src.eval_patches import run_eval_patches
        run_eval_patches(cfg)

    else:
        logger.error(f"Unknown phase: {cfg.phase}")
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
