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
    phase: str = "enhance"  # build_graphs | enhance | trajectory_enhance | dynamic_enhance | merge | evaluate

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
                            "dynamic_enhance", "merge", "evaluate"],
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
