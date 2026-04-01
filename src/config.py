"""Centralized configuration for the context engineering pipeline.

All hardcoded paths and model settings from the original scripts are now
CLI-configurable. The config is loaded once at startup and shared across modules.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # --- Phase selection ---
    phase: str = "enhance"  # build_graphs | enhance | trajectory_enhance | dynamic_enhance | merge | evaluate | generate_patches | eval_patches

    # --- Paths ---
    repo_instances_json: str = ""
    repo_codegraph_index: str = ""
    repo_local_path: str = ""
    original_instances_json: str = ""
    output_file: str = ""
    single_enhanced_file: str = ""  # for multi-agent: single-agent output before reviewer

    # --- Data dirs ---
    data_dir: str = "data"
    output_dir: str = "data/output"
    code_graph_base_dir: str = "data/code_graph"
    prompts_dir: str = "prompts"
    log_dir: str = "logs"
    embeddings_cache_dir: str = "embeddings_cache"

    # --- Graph builder ---
    git_branch: str = "main"
    use_worktree: bool = False
    clean_worktree_after: bool = True
    git_cmd_timeout: int = 600

    # --- Model ---
    openai_model: str = "gpt-5-mini-2025-08-07"
    embed_model: str = "text-embedding-3-large"
    llm_temperature: float = 1.0

    # --- Features ---
    use_llm_classifier: bool = True
    use_bm25_ranking: bool = False

    # --- Limits ---
    embed_top_k: int = 5
    embed_max_tokens: int = 120000
    embed_batch_char_trunc: int = 30000
    agent_max_iterations: int = 20
    subgraph_hops: int = 2
    subgraph_max_nodes: int = 40
    token_est_per_line: int = 3

    # --- Filtering ---
    instance_id_filter: List[str] = field(default_factory=list)
    exclude_tests: bool = True
    exclude_dirs: List[str] = field(default_factory=lambda: [
        "tests", "test", "third_party", "vendor", "dist", "build", "__pycache__"
    ])

    # --- Trajectory / dynamic enhancement ---
    trajectory_summary_file: str = ""
    trajectory_folder: str = ""
    minisweagent_root: str = ""
    minisweagent_wrapper_script: str = ""
    minisweagent_python: str = ""
    minisweagent_results_root: str = ""
    minisweagent_variant: str = "dynamic_minisweagent_enhanced"
    minisweagent_run_name: str = ""
    max_iterative_refinement_rounds: int = 8
    dry_run: bool = False

    # --- Merge ---
    merge_original: str = ""
    merge_enhanced: str = ""
    merge_output: str = ""
    merge_strict: bool = False

    # --- Evaluate ---
    eval_bug_reports: str = ""
    eval_ground_truth: str = ""
    eval_output: str = ""

    # --- Patch generation (mini-swe-agent) ---
    patch_dataset_json: str = ""
    patch_variant: str = "multiagent_enhanced"
    patch_run_name: str = ""
    patch_results_root: str = ""
    patch_problem_field: str = "problem_statement"
    patch_reference_json: str = ""
    patch_project_prefix: str = ""
    patch_limit_instances: Optional[int] = None
    patch_instances_slice: str = ""
    patch_instances_shuffle: bool = False
    patch_model_name: str = ""
    patch_model_class: str = ""
    patch_environment_class: str = "docker"
    patch_model_timeout: int = 180
    patch_per_instance_call_limit: int = 50
    patch_per_instance_cost_limit: float = 0.50
    patch_global_call_limit: int = 0
    patch_global_cost_limit: float = 0.0
    patch_config_spec: List[str] = field(default_factory=list)
    patch_workers: int = 1
    patch_progress_bar: bool = True
    patch_run_mode: str = "per_instance"
    patch_max_minutes: int = 20
    patch_pull_images: bool = False
    patch_delete_image_after_each: bool = True
    patch_redo_existing: bool = False
    patch_raise_exceptions: bool = False
    patch_delete_images_after_run: bool = True
    patch_force_delete_images: bool = True
    patch_stream_actions: bool = True
    patch_watchdog_tick: int = 10

    # --- Patch evaluation (SWE-bench Docker eval) ---
    eval_targets: List[str] = field(default_factory=list)
    eval_variants: List[str] = field(default_factory=lambda: ["multiagent_enhanced"])
    eval_output_root: str = ""
    eval_output_summary: str = ""
    eval_keep_logs: bool = True
    eval_progress_every: int = 1
    eval_limit: Optional[int] = None
    eval_any_repair: bool = True
    eval_instance_list: str = "all"
    eval_meta_source: str = "auto"
    eval_style: str = "harness"
    eval_dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    eval_split: str = "test"
    eval_pull_images: bool = True
    eval_delete_images: bool = True
    eval_timeout: int = 1800

    # --- Debug ---
    debug_max_commits: Optional[int] = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Context Engineering Pipeline for Bug Report Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  build_graphs          Build code and call graphs from local repo
  enhance               Generate enhanced reports (multi-agent with reviewer)
  trajectory_enhance    Further enhance using trajectory insights
  dynamic_enhance       Iterative enhancement with mini-sweagent feedback
  merge                 Merge original + enhanced reports
  evaluate              Run method-match evaluation
  generate_patches      Run mini-swe-agent to generate patches from reports
  eval_patches          Evaluate generated patches against SWE-bench tests

Examples:
  python run.py enhance --repo-instances data/by_repo/django__django.json \\
      --repo-codegraph-index data/code_graph/django__django.json \\
      --repo-local-path /path/to/django \\
      --output data/output/multiagent_enhanced/django__django.json

  python run.py build_graphs --repo-instances data/by_repo/astropy__astropy.json \\
      --repo-local-path /path/to/astropy

  python run.py merge --merge-original data/by_repo/django__django.json \\
      --merge-enhanced data/output/multiagent_enhanced/django__django.json \\
      --merge-output data/output/merged/django__django.json
"""
    )

    p.add_argument("phase", nargs="?", default="enhance",
                   choices=["build_graphs", "enhance", "trajectory_enhance",
                            "dynamic_enhance", "merge", "evaluate",
                            "generate_patches", "eval_patches"],
                   help="Pipeline phase to run (default: enhance)")

    # --- Paths ---
    p.add_argument("--repo-instances", dest="repo_instances_json", default="",
                   help="Path to SWE-Bench per-repo JSON instances")
    p.add_argument("--repo-codegraph-index", dest="repo_codegraph_index", default="",
                   help="Path to codegraph index JSON")
    p.add_argument("--repo-local-path", dest="repo_local_path", default="",
                   help="Path to local git clone of the repo")
    p.add_argument("--original-instances", dest="original_instances_json", default="",
                   help="Path to original SWE-Bench instances (for trajectory phases)")
    p.add_argument("--output", dest="output_file", default="",
                   help="Output file path")
    p.add_argument("--single-enhanced-file", dest="single_enhanced_file", default="",
                   help="Single-agent output file (before reviewer, for enhance phase)")

    # --- Data dirs ---
    p.add_argument("--data-dir", dest="data_dir", default="data")
    p.add_argument("--output-dir", dest="output_dir", default="data/output")
    p.add_argument("--prompts-dir", dest="prompts_dir", default="prompts")
    p.add_argument("--log-dir", dest="log_dir", default="logs")

    # --- Graph builder ---
    p.add_argument("--git-branch", dest="git_branch", default="main")
    p.add_argument("--use-worktree", dest="use_worktree", action="store_true")

    # --- Model ---
    p.add_argument("--model", dest="openai_model", default="gpt-5-mini-2025-08-07")
    p.add_argument("--embed-model", dest="embed_model", default="text-embedding-3-large")
    p.add_argument("--temperature", dest="llm_temperature", type=float, default=1.0)

    # --- Features ---
    p.add_argument("--use-regex-classifier", dest="use_llm_classifier",
                   action="store_false", help="Use regex instead of LLM for classification")
    p.add_argument("--use-bm25", dest="use_bm25_ranking", action="store_true",
                   help="Use BM25 ranking instead of embeddings+FAISS")

    # --- Limits ---
    p.add_argument("--embed-top-k", dest="embed_top_k", type=int, default=5)
    p.add_argument("--agent-max-iter", dest="agent_max_iterations", type=int, default=20)
    p.add_argument("--subgraph-hops", dest="subgraph_hops", type=int, default=2)

    # --- Filtering ---
    p.add_argument("--instance-ids", dest="instance_id_filter", nargs="*", default=[],
                   help="Only process these instance IDs")

    # --- Trajectory / dynamic ---
    p.add_argument("--trajectory-summary", dest="trajectory_summary_file", default="")
    p.add_argument("--trajectory-folder", dest="trajectory_folder", default="")
    p.add_argument("--minisweagent-root", dest="minisweagent_root", default="")
    p.add_argument("--minisweagent-run-name", dest="minisweagent_run_name", default="")
    p.add_argument("--max-refinement-rounds", dest="max_iterative_refinement_rounds",
                   type=int, default=8)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")

    # --- Merge ---
    p.add_argument("--merge-original", dest="merge_original", default="")
    p.add_argument("--merge-enhanced", dest="merge_enhanced", default="")
    p.add_argument("--merge-output", dest="merge_output", default="")
    p.add_argument("--merge-strict", dest="merge_strict", action="store_true")

    # --- Evaluate ---
    p.add_argument("--eval-bug-reports", dest="eval_bug_reports", default="")
    p.add_argument("--eval-ground-truth", dest="eval_ground_truth", default="")
    p.add_argument("--eval-output", dest="eval_output", default="")

    # --- Patch generation ---
    p.add_argument("--patch-dataset", dest="patch_dataset_json", default="",
                   help="Input JSON dataset for patch generation (merged/enhanced reports)")
    p.add_argument("--patch-variant", dest="patch_variant", default="multiagent_enhanced",
                   help="Variant name (default: multiagent_enhanced)")
    p.add_argument("--patch-run-name", dest="patch_run_name", default="",
                   help="Run name for results directory (auto-derived if empty)")
    p.add_argument("--patch-results-root", dest="patch_results_root", default="",
                   help="Root directory for patch results (default: <minisweagent-root>/results)")
    p.add_argument("--patch-problem-field", dest="patch_problem_field", default="problem_statement",
                   help="Field name to use as problem statement (default: problem_statement)")
    p.add_argument("--patch-reference-json", dest="patch_reference_json", default="",
                   help="Reference JSON for SWE-bench fields (repo, base_commit, etc.)")
    p.add_argument("--patch-model", dest="patch_model_name", default="",
                   help="Model name for mini-swe-agent (default: openai/<model>)")
    p.add_argument("--patch-model-class", dest="patch_model_class", default="")
    p.add_argument("--patch-env-class", dest="patch_environment_class", default="docker",
                   help="Environment class: docker (default) or local")
    p.add_argument("--patch-timeout", dest="patch_model_timeout", type=int, default=180,
                   help="Model request timeout in seconds (default: 180)")
    p.add_argument("--patch-call-limit", dest="patch_per_instance_call_limit", type=int, default=50,
                   help="Max API calls per instance (default: 50)")
    p.add_argument("--patch-cost-limit", dest="patch_per_instance_cost_limit", type=float, default=0.50,
                   help="Max cost per instance in $ (default: 0.50)")
    p.add_argument("--patch-workers", dest="patch_workers", type=int, default=1,
                   help="Number of parallel workers (default: 1)")
    p.add_argument("--patch-run-mode", dest="patch_run_mode", default="per_instance",
                   choices=["batch", "per_instance"],
                   help="Execution mode (default: per_instance)")
    p.add_argument("--patch-max-minutes", dest="patch_max_minutes", type=int, default=20,
                   help="Watchdog timeout per instance in minutes (default: 20)")
    p.add_argument("--patch-redo-existing", dest="patch_redo_existing", action="store_true",
                   help="Re-run instances that already have results")
    p.add_argument("--patch-no-delete-images", dest="patch_delete_images_after_run",
                   action="store_false", help="Keep docker images after run")
    p.add_argument("--patch-config-spec", dest="patch_config_spec", nargs="*", default=[],
                   help="mini-swe-agent config spec YAML paths")

    # --- Patch evaluation ---
    p.add_argument("--eval-targets", dest="eval_targets", nargs="+", default=[],
                   help="Result directories to evaluate (e.g. results/astropy__astropy)")
    p.add_argument("--eval-variants", dest="eval_variants", nargs="+",
                   default=["multiagent_enhanced"],
                   help="Variant subfolders to evaluate (default: multiagent_enhanced)")
    p.add_argument("--eval-output-root", dest="eval_output_root", default="",
                   help="Output directory for eval results (default: <target>_tests)")
    p.add_argument("--eval-output-summary", dest="eval_output_summary", default="",
                   help="Path for combined eval summary JSON")
    p.add_argument("--eval-style", dest="eval_style", default="harness",
                   choices=["harness", "pytest"],
                   help="Evaluation style (default: harness)")
    p.add_argument("--eval-dataset", dest="eval_dataset_name",
                   default="princeton-nlp/SWE-bench_Lite",
                   help="SWE-bench dataset name for harness eval")
    p.add_argument("--eval-split", dest="eval_split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--eval-timeout", dest="eval_timeout", type=int, default=1800,
                   help="Docker timeout per evaluation in seconds (default: 1800)")
    p.add_argument("--eval-no-pull", dest="eval_pull_images", action="store_false",
                   help="Don't pull missing Docker images")
    p.add_argument("--eval-keep-images", dest="eval_delete_images", action="store_false",
                   help="Keep Docker images after evaluation")
    p.add_argument("--eval-no-logs", dest="eval_keep_logs", action="store_false",
                   help="Don't keep per-instance Docker output logs")
    p.add_argument("--eval-limit", dest="eval_limit", type=int, default=None,
                   help="Limit number of instances to evaluate")

    # --- Debug ---
    p.add_argument("--debug-max-commits", dest="debug_max_commits", type=int, default=None)

    return p


def load_config(argv=None) -> Config:
    """Parse CLI args and return a Config dataclass."""
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    # Auto-derive paths if not set
    if cfg.minisweagent_root and not cfg.minisweagent_wrapper_script:
        cfg.minisweagent_wrapper_script = os.path.join(
            cfg.minisweagent_root, "run_minisweagent_variant_pipeline.py")
        cfg.minisweagent_python = os.path.join(
            cfg.minisweagent_root, "mini_sweagent-env", "bin", "python")
        cfg.minisweagent_results_root = os.path.join(
            cfg.minisweagent_root, "results")

    return cfg
