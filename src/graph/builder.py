"""Graph builder wrapper.

The original build_code_and_call_graphs.py (scripts/) contains complex AST
parsing logic that is kept as-is. This module wraps it for CLI integration.

For the full graph building pipeline, use:
    python run.py build_graphs --repo-instances <path> --repo-local-path <path>
"""

import logging
import os
import sys

logger = logging.getLogger("context_engineering.graph.builder")


def run_graph_builder(cfg):
    """Run the graph builder by delegating to the original script with config."""
    # The original script uses module-level globals. We patch them before importing.
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                                "build_code_and_call_graphs.py")
    script_path = os.path.abspath(script_path)

    if not os.path.exists(script_path):
        logger.error(f"Graph builder script not found: {script_path}")
        return

    logger.info(f"Running graph builder from {script_path}")
    logger.info(f"  repo_instances: {cfg.repo_instances_json}")
    logger.info(f"  repo_local_path: {cfg.repo_local_path}")
    logger.info(f"  git_branch: {cfg.git_branch}")
    logger.info(f"  code_graph_base_dir: {cfg.code_graph_base_dir}")

    # Import and patch the script's globals
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_graphs", script_path)
    mod = importlib.util.module_from_spec(spec)

    # Patch config before execution
    mod.input_file = cfg.repo_instances_json
    mod.code_graph_base_dir = cfg.code_graph_base_dir
    mod.repo_local_path = cfg.repo_local_path
    mod.git_branch = cfg.git_branch
    mod.USE_WORKTREE = cfg.use_worktree
    mod.CLEAN_WORKTREE_AFTER = cfg.clean_worktree_after
    mod.GIT_CMD_TIMEOUT = cfg.git_cmd_timeout
    mod.DEBUG_MAX_COMMITS = cfg.debug_max_commits

    spec.loader.exec_module(mod)

    logger.info("Graph building complete")
