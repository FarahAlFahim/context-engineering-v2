"""Tool call tracing: records agent tool calls into chat history and prints to terminal."""

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool

import src.state as state

logger = logging.getLogger("context_engineering.tools.tracing")

CLIP_LENGTH = 2000


def _clip(x: Any, n: int = CLIP_LENGTH) -> str:
    """Clip a value to n characters for display."""
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s if len(s) <= n else (s[:n] + "\n...[truncated]...")


def trace_tool_call(tool_name: str, tool_input: Any, tool_output: str):
    """Append Action/Observation entries to the active chat history and print to terminal."""
    entry_in = f"Action: {tool_name}\nAction Input: {_clip(tool_input)}"
    entry_out = f"Observation: {_clip(tool_output)}"

    # Always print to terminal for visibility
    print("\n===== AGENT TOOL TRACE =====")
    print(entry_in)
    print(entry_out)
    print("===== /AGENT TOOL TRACE =====\n")

    # Also log at DEBUG level
    logger.debug(entry_in)
    logger.debug(entry_out)

    if state.active_chat_history is None:
        return

    # Always record the action
    state.active_chat_history.append(entry_in)

    # Record observation if enabled (default: True, matching original script)
    if state.trace_include_observations:
        state.active_chat_history.append(entry_out)


def wrap_tool_for_tracing(tool: StructuredTool) -> StructuredTool:
    """Wrap a LangChain StructuredTool so calls are traced to chat history."""
    original_func = tool.func

    def traced_func(**kwargs) -> str:
        result = original_func(**kwargs)
        trace_tool_call(tool.name, kwargs, result)
        return result

    return StructuredTool.from_function(
        func=traced_func,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )
