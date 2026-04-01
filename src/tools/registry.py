"""Tool registry: builds LangChain Tool lists for agents."""

from typing import List

from langchain.tools import Tool

from src.tools.classify import tool_classify_report
from src.tools.semantic_rank import tool_semantic_rank
from src.tools.code_navigation import (
    tool_get_subgraph, tool_get_code, tool_get_file_context, tool_search_codebase
)
from src.tools.tracing import wrap_tool_for_tracing


def build_tools(for_reviewer: bool = False) -> List[Tool]:
    """Build the list of LangChain Tools for agents.

    Args:
        for_reviewer: If True, excludes classify_report and semantic_rank
                      (reviewer doesn't need discovery tools).
    """
    tools = []

    if not for_reviewer:
        tools.append(Tool(
            name="classify_report",
            description="Extract programming entities (methods, classes, stack traces, code snippets) "
                        "from a bug report. Input: the bug report text.",
            func=tool_classify_report,
        ))
        tools.append(Tool(
            name="semantic_rank",
            description="Rank code entities by semantic similarity to a query. "
                        "Input: a natural language query describing the bug or feature.",
            func=tool_semantic_rank,
        ))

    tools.extend([
        Tool(
            name="get_subgraph",
            description="Explore call graph relationships around a code entity. "
                        "Input: node ID (e.g., 'sklearn/tree/_classes.py::DecisionTreeClassifier').",
            func=tool_get_subgraph,
        ),
        Tool(
            name="get_code",
            description="Retrieve source code for a method, class, or file from the code graph. "
                        "Input: node ID or name (e.g., 'MyClass.my_method' or full path).",
            func=tool_get_code,
        ),
        Tool(
            name="get_file_context",
            description="Read a window of lines from a source file at the correct commit. "
                        'Input JSON: {"file": "path/to/file.py", "start_line": 1, "end_line": 500}',
            func=tool_get_file_context,
        ),
        Tool(
            name="search_codebase",
            description="Grep the repository for a pattern. "
                        'Input JSON: {"pattern": "function_name", "include": "*.py"}',
            func=tool_search_codebase,
        ),
    ])

    # Wrap all tools with tracing
    return [wrap_tool_for_tracing(t) for t in tools]
