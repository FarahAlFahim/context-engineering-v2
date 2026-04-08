"""Tool registry: builds LangChain StructuredTools for agents."""

from typing import List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools.classify import tool_classify_report
from src.tools.semantic_rank import tool_semantic_rank
from src.tools.code_navigation import (
    tool_get_subgraph, tool_get_code, tool_get_file_context, tool_search_codebase
)
from src.tools.tracing import wrap_tool_for_tracing


# ---------- Input schemas ----------

class ClassifyReportInput(BaseModel):
    problem: str = Field(description="The full text of the bug report to classify.")

class SemanticRankInput(BaseModel):
    query: str = Field(description="A natural language query describing the bug or feature to rank code entities against.")

class GetSubgraphInput(BaseModel):
    node: str = Field(description="Node ID to explore, e.g. 'sklearn/tree/_classes.py::DecisionTreeClassifier'.")

class GetCodeInput(BaseModel):
    node: str = Field(description="Node ID or name to retrieve source code for, e.g. 'MyClass.my_method' or a full qualified path.")

class GetFileContextInput(BaseModel):
    file: str = Field(description="Path to the source file relative to the repo root, e.g. 'astropy/modeling/separable.py'.")
    start_line: int = Field(default=1, description="First line number to read (1-based).")
    end_line: int = Field(default=500, description="Last line number to read.")

class SearchCodebaseInput(BaseModel):
    pattern: str = Field(description="Grep pattern to search for in the repository.")
    include: str = Field(default="*.py", description="File glob to restrict search, e.g. '*.py'.")


# ---------- Wrapper functions that accept structured args ----------

def _classify_report(problem: str) -> str:
    return tool_classify_report(problem)

def _semantic_rank(query: str) -> str:
    return tool_semantic_rank(query)

def _get_subgraph(node: str) -> str:
    return tool_get_subgraph(node)

def _get_code(node: str) -> str:
    return tool_get_code(node)

def _get_file_context(file: str, start_line: int = 1, end_line: int = 500) -> str:
    import json
    return tool_get_file_context(json.dumps({"file": file, "start_line": start_line, "end_line": end_line}))

def _search_codebase(pattern: str, include: str = "*.py") -> str:
    import json
    return tool_search_codebase(json.dumps({"pattern": pattern, "include": include}))


# ---------- Role-based tool sets ----------
# Mirrors claw-code's allowed_tools_for_subagent pattern:
# each agent role gets only the tools it needs.

_NAVIGATION_TOOLS = None  # lazy-built below
_DISCOVERY_TOOLS = None
_CHALLENGE_TOOLS = None


def _navigation_tools() -> List[StructuredTool]:
    """Core navigation tools shared by all agent roles."""
    return [
        StructuredTool.from_function(
            func=_get_subgraph,
            name="get_subgraph",
            description="Explore call graph relationships around a code entity.",
            args_schema=GetSubgraphInput,
        ),
        StructuredTool.from_function(
            func=_get_code,
            name="get_code",
            description="Retrieve source code for a method, class, or file from the code graph.",
            args_schema=GetCodeInput,
        ),
        StructuredTool.from_function(
            func=_get_file_context,
            name="get_file_context",
            description="Read a window of lines from a source file at the correct commit.",
            args_schema=GetFileContextInput,
        ),
        StructuredTool.from_function(
            func=_search_codebase,
            name="search_codebase",
            description="Grep the repository for a pattern.",
            args_schema=SearchCodebaseInput,
        ),
    ]


def _discovery_only_tools() -> List[StructuredTool]:
    """Tools exclusive to the Discovery agent (broad search/classification)."""
    return [
        StructuredTool.from_function(
            func=_classify_report,
            name="classify_report",
            description="Extract programming entities (methods, classes, stack traces, code snippets) "
                        "from a bug report.",
            args_schema=ClassifyReportInput,
        ),
        StructuredTool.from_function(
            func=_semantic_rank,
            name="semantic_rank",
            description="Rank code entities by semantic similarity to a query.",
            args_schema=SemanticRankInput,
        ),
    ]


def build_tools(role: str = "discovery") -> List[StructuredTool]:
    """Build LangChain StructuredTools for a specific agent role.

    Roles (similar to claw-code's subagent types):
        - "discovery": All tools including classify_report and semantic_rank.
                       Used by the Discovery agent (broad search, architecture mapping).
        - "challenge": Navigation tools only (get_file_context, get_code,
                       search_codebase, get_subgraph). Used by the Challenge agent
                       (targeted verification of specific methods/values).
        - "reviewer":  Same as challenge. Used by the Reviewer agent.
    """
    if role == "discovery":
        tools = _discovery_only_tools() + _navigation_tools()
    elif role in ("challenge", "reviewer"):
        tools = _navigation_tools()
    else:
        # Fallback: all tools
        tools = _discovery_only_tools() + _navigation_tools()

    return [wrap_tool_for_tracing(t) for t in tools]
