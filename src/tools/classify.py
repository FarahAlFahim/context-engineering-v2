"""Bug report classification tool: extract programming entities via LLM or regex."""

import json
import logging
import re
from typing import Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import src.state as state
from src.utils.llm import load_prompt

logger = logging.getLogger("context_engineering.tools.classify")


def _regex_extract_programming_entities(text: str) -> dict:
    """Regex-based fallback for extracting programming entities from text."""
    methods = list(set(re.findall(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", text)))
    classes = list(set(re.findall(r"\bclass\s+([A-Z]\w*)", text)))

    # Stack traces
    stack_traces = []
    lines = text.splitlines()
    current_trace = []
    in_trace = False
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped for kw in ["Error:", "Exception:", "Traceback", "Caused by:"]):
            if current_trace:
                stack_traces.append("\n".join(current_trace))
            current_trace = [line]
            in_trace = True
        elif in_trace and (stripped.startswith("at ") or stripped.startswith("File ")):
            current_trace.append(line)
        elif in_trace:
            if current_trace:
                stack_traces.append("\n".join(current_trace))
            current_trace = []
            in_trace = False
    if current_trace:
        stack_traces.append("\n".join(current_trace))

    # Code snippets (fenced blocks)
    code_snippets = re.findall(r"```[\w]*\n(.*?)```", text, re.DOTALL)

    # Other programming mentions (file paths, constants)
    other = list(set(re.findall(r"\b[\w/]+\.py\b", text)))

    return {
        "absent_programming_entities": not (methods or classes or stack_traces or code_snippets),
        "methods": methods,
        "classes": classes,
        "stack_traces": stack_traces,
        "code_snippets": code_snippets,
        "other_programming_mentions": other,
    }


def _update_classification_stats(instance_id: str, result: dict, method: str):
    """Update classification_stats with structured info matching original script format."""
    stats = state.classification_stats.setdefault(instance_id, {})
    stats["method"] = method
    stats["has_method_or_class"] = bool(result.get("methods") or result.get("classes"))
    stats["has_stack_trace"] = bool(result.get("stack_traces"))
    stats["has_patch_text"] = bool(
        (state.current_reg_entry or {}).get("patch")
    ) if state.current_reg_entry else False
    stats["absent_programming_entities"] = bool(result.get("absent_programming_entities"))


def _build_navigation_prompt(result: dict, problem: str) -> str:
    """Build a human-readable navigation prompt from classification results,
    matching the original script's output format."""
    absent = result.get("absent_programming_entities", False)

    if absent:
        # No entities found — provide semantic ranking fallback
        fallback_items = result.get("semantic_fallback", [])
        return (
            "You can consider the following items to start your navigation.\n"
            "Analyze their names and read the initial bug report again to think about where to start.\n"
            "Then ask for anyone of them. I will provide the full implementation accordingly.\n\n"
            f"Here are the probable buggy classes:\n{fallback_items}\n\n"
            f"This is the full original bug report for your reference:\n{problem}"
        )

    # Entities found — build structured prompt
    parts = [
        "You can consider the following items to start your navigation. "
        "They were mentioned in the bug report. Analyze their names and read the initial bug report again to think about where to start. "
        "Then ask for a method or class. I will provide the method body or class body accordingly."
    ]

    if result.get("methods"):
        parts.append(f"\nThese method(s) were mentioned:\n{result['methods']}")
    if result.get("classes"):
        parts.append(f"\nThese class(es) were mentioned:\n{result['classes']}")
    if result.get("stack_traces"):
        parts.append(f"\nStack trace(s):\n{result['stack_traces']}")
    if result.get("code_snippets"):
        parts.append(f"\nCode snippet(s):\n{result['code_snippets']}")
    if result.get("other_programming_mentions"):
        parts.append(f"\nOther programming mentions:\n{result['other_programming_mentions']}")

    parts.append(f"\n\nThis is the full original bug report for your reference:\n{problem}")

    return "\n".join(parts)


def tool_classify_report(problem: str) -> str:
    """Extract programming entities from a bug report using LLM or regex.

    This is the callable function for the LangChain Tool.
    """
    instance_id = (state.current_reg_entry or {}).get("instance_id", "unknown")

    if not state.config.use_llm_classifier:
        result = _regex_extract_programming_entities(problem)
        _update_classification_stats(instance_id, result, "regex")
        return _build_navigation_prompt(result, problem)

    try:
        template = load_prompt("classification.txt", state.config.prompts_dir)
        prompt = PromptTemplate.from_template(template)
        chain = prompt | state.llm | StrOutputParser()
        raw = chain.invoke({"problem": problem})

        # Try to parse JSON from response
        cleaned = raw.strip()
        for fence in ("```json\n", "```JSON\n", "```\n"):
            if fence in cleaned:
                cleaned = cleaned.replace(fence, "").replace("\n```", "")

        result = json.loads(cleaned)

        # Recompute absent_programming_entities reliably
        result["absent_programming_entities"] = not (
            result.get("methods") or
            result.get("classes") or
            result.get("stack_traces") or
            result.get("code_snippets") or
            result.get("other_programming_mentions")
        )

        _update_classification_stats(instance_id, result, "llm")
        logger.info(f"LLM classification for {instance_id}: "
                     f"{len(result.get('methods', []))} methods, "
                     f"{len(result.get('classes', []))} classes")
        return _build_navigation_prompt(result, problem)

    except Exception as e:
        logger.warning(f"LLM classification failed for {instance_id}, falling back to regex: {e}")
        result = _regex_extract_programming_entities(problem)
        _update_classification_stats(instance_id, result, "regex_fallback")
        return _build_navigation_prompt(result, problem)
