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


def build_tools(for_reviewer: bool = False) -> List[StructuredTool]:
    """Build the list of LangChain StructuredTools for agents.

    Args:
        for_reviewer: If True, excludes classify_report and semantic_rank
                      (reviewer doesn't need discovery tools).
    """
    tools = []

    if not for_reviewer:
        tools.append(StructuredTool.from_function(
            func=_classify_report,
            name="classify_report",
            description="Extract programming entities (methods, classes, stack traces, code snippets) "
                        "from a bug report.",
            args_schema=ClassifyReportInput,
        ))
        tools.append(StructuredTool.from_function(
            func=_semantic_rank,
            name="semantic_rank",
            description="Rank code entities by semantic similarity to a query.",
            args_schema=SemanticRankInput,
        ))

    tools.extend([
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
    ])

    # Wrap all tools with tracing
    return [wrap_tool_for_tracing(t) for t in tools]
