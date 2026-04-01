"""Bug report classification tool: extract programming entities via LLM or regex."""

import json
import logging
import re
from typing import Dict

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

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


def tool_classify_report(problem: str) -> str:
    """Extract programming entities from a bug report using LLM or regex.

    This is the callable function for the LangChain Tool.
    """
    instance_id = (state.current_reg_entry or {}).get("instance_id", "unknown")

    if not state.config.use_llm_classifier:
        result = _regex_extract_programming_entities(problem)
        state.classification_stats.setdefault(instance_id, {})["method"] = "regex"
        return json.dumps(result, indent=2)

    try:
        template = load_prompt("classification.txt", state.config.prompts_dir)
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=state.llm, prompt=prompt)
        raw = chain.run({"problem": problem})

        # Try to parse JSON from response
        cleaned = raw.strip()
        for fence in ("```json\n", "```JSON\n", "```\n"):
            if fence in cleaned:
                cleaned = cleaned.replace(fence, "").replace("\n```", "")

        result = json.loads(cleaned)
        state.classification_stats.setdefault(instance_id, {})["method"] = "llm"
        logger.info(f"LLM classification for {instance_id}: "
                     f"{len(result.get('methods', []))} methods, "
                     f"{len(result.get('classes', []))} classes")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.warning(f"LLM classification failed for {instance_id}, falling back to regex: {e}")
        result = _regex_extract_programming_entities(problem)
        state.classification_stats.setdefault(instance_id, {})["method"] = "regex_fallback"
        return json.dumps(result, indent=2)
