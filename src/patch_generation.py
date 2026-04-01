"""Patch generation: run mini-swe-agent on enhanced reports to produce patches.

Generates a temporary config-patched wrapper script and runs it under
mini-swe-agent's own virtualenv Python, so all its dependencies resolve.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap

logger = logging.getLogger("context_engineering.patch_generation")


def run_patch_generation(cfg):
    """Run mini-swe-agent patch generation via subprocess."""
    minisweagent_root = cfg.minisweagent_root
    if not minisweagent_root:
        logger.error("--minisweagent-root is required for generate_patches")
        sys.exit(1)

    wrapper_script = cfg.minisweagent_wrapper_script
    if not os.path.isfile(wrapper_script):
        logger.error(f"Wrapper script not found: {wrapper_script}")
        sys.exit(1)

    if not cfg.patch_dataset_json:
        logger.error("--patch-dataset is required for generate_patches")
        sys.exit(1)

    python_exe = cfg.minisweagent_python
    if not os.path.isfile(python_exe):
        logger.error(f"mini-swe-agent Python not found: {python_exe}")
        logger.info("Expected at: <minisweagent-root>/mini_sweagent-env/bin/python")
        sys.exit(1)

    run_name = cfg.patch_run_name or _derive_run_name(cfg.patch_dataset_json, cfg.patch_variant)
    results_root = os.path.abspath(cfg.patch_results_root or os.path.join(minisweagent_root, "results"))
    model_name = cfg.patch_model_name or f"openai/{cfg.openai_model}"

    config_spec = cfg.patch_config_spec or [
        os.path.join(minisweagent_root, "src", "minisweagent", "config", "benchmarks", "swebench.yaml"),
    ]

    # Build the CONFIG dict to inject
    config_override = {
        "VARIANT": cfg.patch_variant,
        "DATASET_JSON": os.path.abspath(cfg.patch_dataset_json),
        "RUN_NAME": run_name,
        "RESULTS_ROOT": results_root,
        "INSTANCES_FILTER": cfg.instance_id_filter or [],
        "INSTANCES_SLICE": cfg.patch_instances_slice,
        "INSTANCES_SHUFFLE": cfg.patch_instances_shuffle,
        "PROBLEM_FIELD": cfg.patch_problem_field,
        "REFERENCE_JSON": os.path.abspath(cfg.patch_reference_json) if cfg.patch_reference_json else None,
        "PROJECT_PREFIX": cfg.patch_project_prefix or None,
        "LIMIT_INSTANCES": cfg.patch_limit_instances,
        "MODEL_NAME": model_name,
        "MODEL_CLASS": cfg.patch_model_class or None,
        "ENVIRONMENT_CLASS": cfg.patch_environment_class,
        "MODEL_REQUEST_TIMEOUT_SECONDS": cfg.patch_model_timeout,
        "PER_INSTANCE_CALL_LIMIT": cfg.patch_per_instance_call_limit,
        "PER_INSTANCE_COST_LIMIT": cfg.patch_per_instance_cost_limit,
        "GLOBAL_CALL_LIMIT": cfg.patch_global_call_limit,
        "GLOBAL_COST_LIMIT": cfg.patch_global_cost_limit,
        "CONFIG_SPEC": config_spec,
        "WORKERS": cfg.patch_workers,
        "PROGRESS_BAR": cfg.patch_progress_bar,
        "RUN_MODE": cfg.patch_run_mode,
        "MAX_MINUTES_PER_INSTANCE": cfg.patch_max_minutes,
        "PULL_DOCKER_IMAGES_BEFORE_RUN": cfg.patch_pull_images,
        "DELETE_DOCKER_IMAGE_AFTER_EACH_INSTANCE": cfg.patch_delete_image_after_each,
        "REDO_EXISTING": cfg.patch_redo_existing,
        "RAISE_EXCEPTIONS": cfg.patch_raise_exceptions,
        "DELETE_DOCKER_IMAGES_AFTER_RUN": cfg.patch_delete_images_after_run,
        "FORCE_DELETE_DOCKER_IMAGES": cfg.patch_force_delete_images,
        "DRY_RUN": cfg.dry_run,
        "STREAM_STEP_ACTIONS": cfg.patch_stream_actions,
        "WATCHDOG_TICK_SECONDS": cfg.patch_watchdog_tick,
    }

    logger.info(f"Variant: {cfg.patch_variant}")
    logger.info(f"Dataset: {cfg.patch_dataset_json}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Results root: {results_root}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Run mode: {cfg.patch_run_mode}, workers: {cfg.patch_workers}")
    if cfg.instance_id_filter:
        logger.info(f"Instance filter: {cfg.instance_id_filter}")

    # Write a launcher script that patches CONFIG before importing/running
    launcher_code = textwrap.dedent(f"""\
        import json, sys, os
        sys.path.insert(0, {minisweagent_root!r})
        os.chdir({minisweagent_root!r})

        config_override = json.loads({json.dumps(json.dumps(config_override))})

        import run_minisweagent_variant_pipeline as wrapper
        wrapper.CONFIG.update(config_override)
        wrapper.run()
    """)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="patch_gen_", delete=False
    ) as f:
        f.write(launcher_code)
        launcher_path = f.name

    try:
        logger.info(f"Launching mini-swe-agent subprocess: {python_exe}")
        result = subprocess.run(
            [python_exe, launcher_path],
            cwd=minisweagent_root,
        )
        if result.returncode != 0:
            logger.error(f"Patch generation failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        os.unlink(launcher_path)

    logger.info("Patch generation complete")


def _derive_run_name(dataset_json: str, variant: str) -> str:
    """Derive a run name from the dataset filename and variant."""
    base = os.path.splitext(os.path.basename(dataset_json))[0]
    return f"{base}_{variant}"
