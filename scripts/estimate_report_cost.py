#!/usr/bin/env python3
"""Estimate token usage and cost for report generation (vanilla_baseline & multiagent_enhanced).

The pipeline did not track actual token usage, so we reconstruct the LLM call
structure from the saved output data and estimate tokens with tiktoken.

vanilla_baseline pipeline:
  1. Explorer agent — multi-turn (single run, no continuations)
  2. generate_final_bug_report — single call (raw Thoughts as input)
  (No compression step, no reviewer)

multiagent_enhanced pipeline:
  1. Explorer agent — multi-turn (with continuation loop)
  2. compress_chat_history — single call
  3. generate_final_bug_report — single call
  4. Reviewer agent — multi-turn
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import tiktoken

WORKSPACE = Path(__file__).parent
PROMPTS_DIR = WORKSPACE / "prompts"

# Pricing per 1M tokens
PRICING = {
    "gpt_5_mini": {"input": 0.25, "output": 2.00},
    "minimax2.5":  {"input": 0.15, "output": 0.90},
}

REPOS = [
    "astropy__astropy", "django__django", "matplotlib__matplotlib",
    "mwaskom__seaborn", "pallets__flask", "psf__requests",
    "pydata__xarray", "pylint-dev__pylint", "pytest-dev__pytest",
    "scikit-learn__scikit-learn", "sphinx-doc__sphinx", "sympy__sympy",
]


def get_encoder():
    try:
        return tiktoken.encoding_for_model("gpt-5-mini-2025-08-07")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_turns_from_history(chat_history: list) -> list:
    """Parse chat_history into turns for token estimation."""
    turns = []
    accumulated = []

    for i, entry in enumerate(chat_history):
        if entry.startswith("[agent]"):
            continue
        if entry.startswith("Thought:"):
            completion_parts = [entry]
            j = i + 1
            while j < len(chat_history) and chat_history[j].startswith("Action:"):
                completion_parts.append(chat_history[j])
                j += 1
            turns.append((list(accumulated), completion_parts))
            accumulated.extend(completion_parts)
        elif entry.startswith("Observation:") or entry.startswith("Action:"):
            if entry not in accumulated:
                accumulated.append(entry)
        elif entry.startswith("LangGraph agent error"):
            accumulated.append(entry)

    return turns


def estimate_agent_tokens(chat_history, system_text, user_text, enc):
    """Estimate prompt+completion tokens for a multi-turn agent."""
    system_tokens = count_tokens(system_text, enc)
    user_tokens = count_tokens(user_text, enc)
    turns = parse_turns_from_history(chat_history)

    total_prompt = 0
    total_completion = 0

    for accumulated, completion_parts in turns:
        ctx_tokens = count_tokens("\n".join(accumulated), enc)
        total_prompt += system_tokens + user_tokens + ctx_tokens
        total_completion += sum(count_tokens(c, enc) for c in completion_parts)

    return total_prompt, total_completion, len(turns)


def estimate_vanilla_instance(inst, enc, explorer_system, report_template):
    """Estimate tokens for one vanilla_baseline instance."""
    total_prompt = 0
    total_completion = 0

    problem = inst.get("problem_statement", "") or ""
    chat_history = inst.get("chat_history", [])
    bug_report = inst.get("bug_report", {})
    bug_report_str = json.dumps(bug_report, ensure_ascii=False) if isinstance(bug_report, dict) else str(bug_report)

    # 1. Explorer agent
    explorer_user = f"Problem: {problem}\n"
    ep, ec, e_turns = estimate_agent_tokens(chat_history, explorer_system, explorer_user, enc)
    total_prompt += ep
    total_completion += ec

    # 2. Final report (raw Thoughts as input, no compression)
    agent_thoughts = [e for e in chat_history if e.startswith("Thought:")]
    agent_analysis = "\n\n".join(agent_thoughts) if agent_thoughts else ""
    report_input = report_template.format(bug_report=problem, chat_history=agent_analysis)
    total_prompt += count_tokens(report_input, enc)
    total_completion += count_tokens(bug_report_str, enc)

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "explorer_turns": e_turns,
        "reviewer_turns": 0,
    }


def estimate_multiagent_instance(inst, enc, explorer_system, compress_template,
                                  report_template, reviewer_system):
    """Estimate tokens for one multiagent_enhanced instance."""
    total_prompt = 0
    total_completion = 0

    problem = inst.get("problem_statement", "") or ""
    chat_history = inst.get("chat_history", [])
    compressed_analysis = inst.get("compressed_analysis", "")
    reviewer_history = inst.get("reviewer_history", [])
    bug_report = inst.get("bug_report", {})
    bug_report_str = json.dumps(bug_report, ensure_ascii=False) if isinstance(bug_report, dict) else str(bug_report)

    # 1. Explorer agent
    explorer_user = f"Problem: {problem}\n"
    ep, ec, e_turns = estimate_agent_tokens(chat_history, explorer_system, explorer_user, enc)
    total_prompt += ep
    total_completion += ec

    # 2. Compress chat history
    chat_text = "\n".join(chat_history) if chat_history else "(empty)"
    compress_input = compress_template.format(bug_report=problem, chat_history=chat_text)
    total_prompt += count_tokens(compress_input, enc)
    total_completion += count_tokens(compressed_analysis, enc)

    # 3. Final report
    report_input = report_template.format(bug_report=problem, chat_history=compressed_analysis)
    total_prompt += count_tokens(report_input, enc)
    total_completion += count_tokens(bug_report_str, enc)

    # 4. Reviewer agent
    draft_json = bug_report_str
    reviewer_user = (
        "Original bug report:\n" + problem +
        "\n\n=== Draft report JSON ===\n" + draft_json +
        "\n\n=== Investigation analysis (compressed) ===\n" + compressed_analysis
    )
    rp, rc, r_turns = estimate_agent_tokens(reviewer_history, reviewer_system, reviewer_user, enc)
    total_prompt += rp
    total_completion += rc

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "explorer_turns": e_turns,
        "reviewer_turns": r_turns,
    }


def process_report_generation(model: str, variant: str, enc):
    """Process report generation estimation for one model+variant."""
    data_dir = WORKSPACE / "data" / "output" / model / variant

    explorer_system_ma = load_prompt("agent_instruction_multi_agent.txt")
    explorer_system_vb = load_prompt("agent_instruction_no_protocol.txt")
    compress_template = load_prompt("compress_chat_history.txt")
    report_template = load_prompt("final_report.txt")
    reviewer_system = load_prompt("reviewer_multi_agent.txt")

    grand_totals = defaultdict(int)
    grand_count = 0

    for repo in REPOS:
        json_path = data_dir / f"{repo}.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            instances = json.load(f)

        for inst in instances:
            try:
                if variant == "vanilla_baseline":
                    m = estimate_vanilla_instance(inst, enc, explorer_system_vb, report_template)
                else:
                    m = estimate_multiagent_instance(
                        inst, enc, explorer_system_ma, compress_template,
                        report_template, reviewer_system)
                for k, v in m.items():
                    grand_totals[k] += v
                grand_count += 1
            except Exception:
                pass

    return dict(grand_totals), grand_count


def main():
    enc = get_encoder()

    all_report = {}
    models = ["gpt_5_mini", "minimax2.5"]
    variants = ["vanilla_baseline", "multiagent_enhanced"]

    for model in models:
        prices = PRICING[model]
        print(f"\n{'='*110}")
        print(f"  REPORT GENERATION (estimated) — Model: {model}")
        print(f"  Pricing: ${prices['input']}/1M input, ${prices['output']}/1M output")
        print(f"{'='*110}")

        for variant in variants:
            totals, count = process_report_generation(model, variant, enc)
            cost = (totals.get("prompt_tokens", 0) / 1e6 * prices["input"] +
                    totals.get("completion_tokens", 0) / 1e6 * prices["output"])
            totals["cost"] = cost
            all_report[(model, variant)] = {"totals": totals, "count": count}

            avg_cost = cost / count if count else 0
            avg_total = totals.get("total_tokens", 0) / count if count else 0
            avg_prompt = totals.get("prompt_tokens", 0) / count if count else 0
            avg_compl = totals.get("completion_tokens", 0) / count if count else 0

            print(f"\n  [{variant}] ({count} instances)")
            print(f"    Est. total cost:      ${cost:.4f}")
            print(f"    Est. avg cost/inst:   ${avg_cost:.4f}")
            print(f"    Est. total tokens:    {totals.get('total_tokens', 0):,}")
            print(f"    Est. avg tokens/inst: {avg_total:,.0f}")
            print(f"      Avg prompt:         {avg_prompt:,.0f}")
            print(f"      Avg completion:     {avg_compl:,.0f}")
            print(f"    Avg explorer turns:   {totals.get('explorer_turns',0)/count:.1f}" if count else "")
            print(f"    Avg reviewer turns:   {totals.get('reviewer_turns',0)/count:.1f}" if count else "")

    # Summary table
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY TABLE — Report Generation (Estimated)")
    print(f"{'='*110}")

    header = f"{'Model':<15} {'Variant':<25} {'Inst':>5} {'Est Cost':>12} {'Avg Cost':>10} {'Avg Tokens':>12} {'Avg Prompt':>12} {'Avg Compl':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for model in models:
        for variant in variants:
            d = all_report.get((model, variant), {"totals": {}, "count": 0})
            t = d["totals"]
            n = d["count"]
            avg_cost = t.get("cost", 0) / n if n else 0
            avg_tok = t.get("total_tokens", 0) / n if n else 0
            avg_p = t.get("prompt_tokens", 0) / n if n else 0
            avg_c = t.get("completion_tokens", 0) / n if n else 0
            print(
                f"{model:<15} {variant:<25} {n:>5} "
                f"${t.get('cost', 0):>10.4f} "
                f"${avg_cost:>8.4f} "
                f"{avg_tok:>12,.0f} "
                f"{avg_p:>12,.0f} "
                f"{avg_c:>10,.0f}"
            )

    # Combined total table (report gen + patch gen would need to import patch data)
    print(f"\n\nNote: 'original' has no report generation cost (uses raw problem statements).")
    print(f"Note: Token counts are ESTIMATES reconstructed via tiktoken. Not from logs.")
    print(f"Note: For MiniMax, tiktoken is approximate (different tokenizer).")


if __name__ == "__main__":
    main()
