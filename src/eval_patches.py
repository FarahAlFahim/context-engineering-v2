"""Patch evaluation: run SWE-bench Docker evaluation on generated patches.

Wraps the external eval_minisweagent_resolved.py script, configuring it
via CLI args and running it under mini-swe-agent's virtualenv Python.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap

logger = logging.getLogger("context_engineering.eval_patches")


def run_eval_patches(cfg):
    """Run SWE-bench patch evaluation via subprocess."""
    minisweagent_root = cfg.minisweagent_root
    if not minisweagent_root:
        logger.error("--minisweagent-root is required for eval_patches")
        sys.exit(1)

    eval_script = os.path.join(minisweagent_root, "eval_minisweagent_resolved.py")
    if not os.path.isfile(eval_script):
        logger.error(f"Eval script not found: {eval_script}")
        sys.exit(1)

    python_exe = cfg.minisweagent_python
    if not os.path.isfile(python_exe):
        logger.error(f"mini-swe-agent Python not found: {python_exe}")
        logger.info("Expected at: <minisweagent-root>/mini_sweagent-env/bin/python")
        sys.exit(1)

    if not cfg.eval_targets:
        logger.error("--eval-targets is required for eval_patches (one or more result dirs)")
        sys.exit(1)

    # Resolve targets to absolute paths
    targets = [os.path.abspath(t) for t in cfg.eval_targets]
    variants = cfg.eval_variants or ["multiagent_enhanced"]
    output_root = os.path.abspath(cfg.eval_output_root) if cfg.eval_output_root else ""
    output_summary = os.path.abspath(cfg.eval_output_summary) if cfg.eval_output_summary else ""

    # Auto-derive output paths if not set
    if not output_root and targets:
        output_root = targets[0] + "_tests"
    if not output_summary and output_root:
        output_summary = os.path.join(output_root, "combined_eval_summary.json")

    config_override = {
        "TARGETS": targets,
        "VARIANTS": variants,
        "OUTPUT_ROOT": output_root,
        "OUTPUT_SUMMARY_JSON": output_summary,
        "KEEP_LOGS": cfg.eval_keep_logs,
        "PROGRESS_WRITE_EVERY": cfg.eval_progress_every,
        "LIMIT": cfg.eval_limit,
        "INSTANCE_IDS": cfg.instance_id_filter or [],
        "EVALUATE_ANY_REPAIR": cfg.eval_any_repair,
        "INSTANCE_LIST": cfg.eval_instance_list,
        "META_SOURCE": cfg.eval_meta_source,
        "EVAL_STYLE": cfg.eval_style,
        "DATASET_NAME": cfg.eval_dataset_name,
        "SPLIT": cfg.eval_split,
        "PULL_IMAGES": cfg.eval_pull_images,
        "DELETE_IMAGES_AFTER_EVAL": cfg.eval_delete_images,
        "TIMEOUT": cfg.eval_timeout,
    }

    logger.info(f"Targets: {targets}")
    logger.info(f"Variants: {variants}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Output summary: {output_summary}")
    logger.info(f"Eval style: {cfg.eval_style}")
    if cfg.instance_id_filter:
        logger.info(f"Instance filter: {cfg.instance_id_filter}")

    # Write launcher script that patches CONFIG before running
    launcher_code = textwrap.dedent(f"""\
        import json, sys, os
        sys.path.insert(0, {minisweagent_root!r})
        os.chdir({minisweagent_root!r})

        config_override = json.loads({json.dumps(json.dumps(config_override))})

        import eval_minisweagent_resolved as evaluator
        evaluator.CONFIG.update(config_override)
        raise SystemExit(evaluator.main())
    """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="eval_patches_", delete=False
    ) as f:
        f.write(launcher_code)
        launcher_path = f.name

    try:
        logger.info(f"Launching eval subprocess: {python_exe}")
        result = subprocess.run(
            [python_exe, launcher_path],
            cwd=minisweagent_root,
        )
        if result.returncode != 0:
            logger.error(f"Patch evaluation failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        os.unlink(launcher_path)

    logger.info(f"Evaluation complete. Summary: {output_summary}")
