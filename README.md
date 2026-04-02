# Context Engineering for Bug Report Enhancement

A modular pipeline for enhancing bug reports with source code context. Given original bug reports from [SWE-Bench](https://www.swebench.com/), the system analyzes code graphs, navigates source code with LLM-powered agents, and produces enhanced reports with root cause analysis and fix suggestions.

## Project Structure

```
.
├── run.py                          # CLI entry point (all phases)
├── requirements.txt
│
├── src/                            # Source code
│   ├── config.py                   # CLI args + Config dataclass
│   ├── state.py                    # Shared session state
│   ├── log.py                      # Logging (file + console)
│   ├── utils/                      # Shared utilities
│   │   ├── io.py                   #   File I/O, JSON, subprocess
│   │   ├── tokens.py               #   Token counting, chunking
│   │   ├── llm.py                  #   LLM init, prompt loading
│   │   ├── json_parser.py          #   Robust JSON extraction
│   │   ├── embeddings.py           #   FAISS embedding index
│   │   └── bm25.py                 #   BM25 ranking
│   ├── graph/                      # Code graph handling
│   │   ├── loader.py               #   Load codegraph, build adjacency
│   │   ├── filters.py              #   Test/path filtering
│   │   └── builder.py              #   Graph builder wrapper
│   ├── tools/                      # LangChain tools for agents
│   │   ├── classify.py             #   Bug report classification
│   │   ├── semantic_rank.py        #   Semantic ranking
│   │   ├── code_navigation.py      #   get_code, get_subgraph, search
│   │   ├── tracing.py              #   Tool call tracing
│   │   └── registry.py             #   Tool list builder
│   ├── agents/                     # Pipeline orchestrators
│   │   ├── common.py               #   Shared agent utilities
│   │   ├── multi_agent.py          #   Explorer + reviewer agents
│   │   ├── trajectory_insights.py  #   Trajectory-based enhancement
│   │   └── dynamic_insights.py     #   Iterative enhancement loop
│   ├── evaluation/                 # Evaluation tools
│   │   ├── method_matcher.py       #   Method prediction accuracy
│   │   └── enhancement_checker.py  #   Enhancement flag checker
│   ├── merge.py                    # Merge original + enhanced reports
│   ├── patch_generation.py         # mini-swe-agent patch generation wrapper
│   └── eval_patches.py             # SWE-bench Docker patch evaluation wrapper
│
├── prompts/                        # Prompt templates (editable)
├── data/                           # Input data (not tracked in git)
│   ├── by_repo/                    #   SWE-Bench instances per repo
│   ├── code_graph/                 #   Pre-built code/call graphs
│   ├── patches/                    #   Ground truth patches
│   └── test_patches/               #   Test patches
├── logs/                           # Runtime logs (auto-created)
├── embeddings_cache/               # Embedding cache (auto-created)
└── scripts/                        # Original monolithic scripts (reference)
```

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Pipeline Phases

All phases are run via `python run.py <phase> [options]`. Use `python run.py --help` for full option list.

### 1. Build Code Graphs

Parse a local repository at each SWE-Bench commit and produce code graph + call graph artifacts.

```bash
python run.py build_graphs \
    --repo-instances data/by_repo/astropy__astropy.json \
    --repo-local-path /path/to/astropy \
    --git-branch main
```

**Output:** `data/code_graph/<repo_slug>/<commit>/codegraph.json` and `callgraph.json`

### 2. Enhance (Multi-Agent)

Run an exploration agent to analyze the bug using code graphs, then a reviewer agent to validate and improve the report. Both pre-reviewer and post-reviewer reports are saved automatically.

```bash
python run.py enhance \
    --repo-instances data/by_repo/astropy__astropy.json \
    --repo-codegraph-index data/code_graph/astropy__astropy.json \
    --repo-local-path /path/to/astropy \
    --output data/output/multiagent_enhanced/astropy__astropy.json
```

**Outputs:**
- **Pre-reviewer (single-agent):** `data/output/enhanced/astropy__astropy.json` (auto-derived from `--output` filename)
- **Post-reviewer (multi-agent):** `data/output/multiagent_enhanced/astropy__astropy.json` (the `--output` path)

To override the pre-reviewer output path, use `--single-enhanced-file`.

### 3. Trajectory Enhance

Further enhance reports using insights from an external repair agent's trajectory summary.

```bash
python run.py trajectory_enhance \
    --repo-instances data/output/multiagent_enhanced/matplotlib__matplotlib.json \
    --repo-codegraph-index data/code_graph/matplotlib__matplotlib.json \
    --repo-local-path /path/to/matplotlib \
    --original-instances data/by_repo/matplotlib__matplotlib.json \
    --trajectory-summary /path/to/traj_summary.json \
    --output data/output/trajectory_enhanced/matplotlib__matplotlib.json
```

### 4. Dynamic Enhance

Iteratively refine reports using a mini-sweagent feedback loop (enhance → run agent → compare → enhance again).

```bash
python run.py dynamic_enhance \
    --repo-instances data/output/multiagent_enhanced/matplotlib__matplotlib.json \
    --repo-codegraph-index data/code_graph/matplotlib__matplotlib.json \
    --repo-local-path /path/to/matplotlib \
    --original-instances data/by_repo/matplotlib__matplotlib.json \
    --trajectory-folder /path/to/trajectories \
    --minisweagent-root /path/to/mini-swe-agent \
    --output data/output/dynamic_enhanced/matplotlib__matplotlib.json
```

### 5. Merge

Merge enhanced bug reports back into the original SWE-Bench format.

```bash
python run.py merge \
    --merge-original data/by_repo/django__django.json \
    --merge-enhanced data/output/multiagent_enhanced/django__django.json \
    --merge-output data/output/merged/django__django.json
```

### 6. Generate Patches

Run mini-swe-agent on merged/enhanced reports to generate code patches. Requires a local mini-swe-agent installation with Docker.

```bash
python run.py generate_patches \
    --minisweagent-root /path/to/mini-swe-agent \
    --patch-dataset data/output/merged/django__django.json \
    --patch-variant multiagent_enhanced \
    --patch-run-name django__django_multiagent \
    --instance-ids django__django-12345
```

**Key options:**

| Option | Description |
|---|---|
| `--patch-dataset PATH` | Input JSON (merged/enhanced reports) |
| `--patch-variant NAME` | Variant name (default: `multiagent_enhanced`) |
| `--patch-run-name NAME` | Run name for results dir (auto-derived if empty) |
| `--patch-results-root DIR` | Results root (default: `<minisweagent-root>/results`) |
| `--patch-model MODEL` | Model for mini-swe-agent (default: `openai/<model>`) |
| `--patch-call-limit N` | Max API calls per instance (default: `50`) |
| `--patch-cost-limit $` | Max cost per instance (default: `0.50`) |
| `--patch-workers N` | Parallel workers (default: `1`) |
| `--patch-max-minutes N` | Timeout per instance in minutes (default: `20`) |
| `--patch-redo-existing` | Re-run instances with existing results |
| `--patch-reference-json PATH` | Reference JSON for SWE-bench fields |

**Output:** `<results-root>/<run-name>/<variant>/repair/<instance_id>/`

### 7. Evaluate Patches (SWE-bench)

Run Docker-based SWE-bench evaluation on generated patches. Evaluates multiple variants per instance, reusing Docker images for efficiency.

```bash
python run.py eval_patches \
    --minisweagent-root /path/to/mini-swe-agent \
    --eval-targets results/astropy__astropy_multiagent \
    --eval-variants original multiagent_enhanced \
    --instance-ids astropy__astropy-12907
```

**Key options:**

| Option | Description |
|---|---|
| `--eval-targets DIR ...` | Result directories to evaluate |
| `--eval-variants NAME ...` | Variant subfolders (default: `multiagent_enhanced`) |
| `--eval-output-root DIR` | Output dir for eval results (default: `<target>_tests`) |
| `--eval-style {harness,pytest}` | Evaluation mode (default: `harness`) |
| `--eval-dataset NAME` | SWE-bench dataset (default: `princeton-nlp/SWE-bench_Lite`) |
| `--eval-timeout SECS` | Docker timeout per eval (default: `1800`) |
| `--eval-no-pull` | Don't pull missing Docker images |
| `--eval-keep-images` | Keep Docker images after evaluation |
| `--eval-limit N` | Limit number of instances to evaluate |

**Output:** `<output-root>/<variant>/eval_summary.json` and `combined_eval_summary.json`

### 8. Evaluate Method Matching

Evaluate method prediction accuracy against ground truth.

```bash
python run.py evaluate \
    --eval-bug-reports data/output/multiagent_enhanced/astropy__astropy.json \
    --eval-ground-truth data/ground_truth/astropy__astropy.json \
    --eval-output data/output/evaluation/astropy__astropy.json
```

## Common Options

| Option | Description |
|---|---|
| `--instance-ids ID1 ID2 ...` | Only process specific instance IDs |
| `--model MODEL` | LLM model name (default: `gpt-5-mini-2025-08-07`) |
| `--temperature T` | LLM temperature (default: `1.0`) |
| `--use-regex-classifier` | Use regex instead of LLM for entity extraction |
| `--use-bm25` | Use BM25 ranking instead of FAISS embeddings |
| `--dry-run` | Print trajectory info without running agents |
| `--log-dir DIR` | Log directory (default: `logs/`) |

## Data Flow

```
data/by_repo/*.json (original SWE-Bench instances)
        │
        ▼
  [build_graphs] ──► data/code_graph/<repo>/<commit>/{codegraph,callgraph}.json
        │
        ▼
    [enhance] ─────► data/output/enhanced/<repo>.json  (pre-reviewer)
               └──► data/output/multiagent_enhanced/<repo>.json  (post-reviewer)
        │
        ├──► [trajectory_enhance] ──► data/output/trajectory_enhanced/<repo>.json
        │
        └──► [dynamic_enhance] ────► data/output/dynamic_enhanced/<repo>.json
                                            │
                                            ▼
                                    [merge] ──► data/output/merged/<repo>.json
                                            │
                                            ▼
                                [generate_patches] ──► results/<run>/<variant>/repair/
                                            │
                                            ▼
                                  [eval_patches] ──► <output-root>/<variant>/eval_summary.json
```

## Prompts

All prompt templates are in `prompts/` and can be edited without modifying code:

| File | Purpose |
|---|---|
| `classification.txt` | Extract programming entities from bug reports |
| `agent_instruction_multi_agent.txt` | Exploration agent system prompt |
| `agent_instruction_dynamic.txt` | Dynamic exploration agent system prompt |
| `final_report_multi_agent.txt` | Bug report generation (FixSteps variant) |
| `final_report_with_suggestions.txt` | Bug report generation (Suggestions variant) |
| `reviewer_multi_agent.txt` | Reviewer agent for multi-agent pipeline |
| `reviewer_trajectory.txt` | Reviewer agent for trajectory integration |
| `redundancy_comparison.txt` | Check if trajectory adds new value |
| `phase_comparison.txt` | Compare trajectory phases with bug report |
| `trajectory_phase_split.txt` | Split trajectory into localization/repair |

## Logging

Each run creates a timestamped log file in `logs/`:

```
logs/enhance_20260401_143022.log
```

- **Console**: INFO-level (progress updates)
- **Log file**: DEBUG-level (full tool traces, LLM calls)

## Supported Repositories

The pipeline supports any SWE-Bench repository. Pre-configured repos include:
`astropy`, `django`, `flask`, `matplotlib`, `pylint`, `pytest`, `requests`, `scikit-learn`, `seaborn`, `sphinx`, `sympy`, `xarray`
