"""Tool call tracing: records agent tool calls into chat history."""

import logging
from typing import Any

from langchain.tools import Tool

import src.state as state

logger = logging.getLogger("context_engineering.tools.tracing")


def trace_tool_call(tool_name: str, tool_input: str, tool_output: str):
    """Append Action/Observation entries to the active chat history."""
    if state.active_chat_history is None:
        return

    state.active_chat_history.append(f"Action: {tool_name}")
    state.active_chat_history.append(f"Action Input: {tool_input[:500]}")

    if state.trace_include_observations:
        # Truncate long observations
        obs = tool_output[:3000] if len(tool_output) > 3000 else tool_output
        state.active_chat_history.append(f"Observation: {obs}")
    else:
        state.active_chat_history.append(f"Observation: [output length: {len(tool_output)} chars]")


def wrap_tool_for_tracing(tool: Tool) -> Tool:
    """Wrap a LangChain Tool so calls are traced to chat history."""
    original_func = tool.func

    def traced_func(input_str: str) -> str:
        result = original_func(input_str)
        trace_tool_call(tool.name, input_str, result)
        return result

    return Tool(
        name=tool.name,
        description=tool.description,
        func=traced_func,
    )
