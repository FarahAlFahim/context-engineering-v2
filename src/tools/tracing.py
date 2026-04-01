"""Tool call tracing: records agent tool calls into chat history."""

import logging
from typing import Any

from langchain_core.tools import StructuredTool

import src.state as state

logger = logging.getLogger("context_engineering.tools.tracing")


def trace_tool_call(tool_name: str, tool_input: Any, tool_output: str):
    """Append Action/Observation entries to the active chat history."""
    if state.active_chat_history is None:
        return

    state.active_chat_history.append(f"Action: {tool_name}")
    input_str = str(tool_input)[:500]
    state.active_chat_history.append(f"Action Input: {input_str}")

    if state.trace_include_observations:
        obs = tool_output[:3000] if len(tool_output) > 3000 else tool_output
        state.active_chat_history.append(f"Observation: {obs}")
    else:
        state.active_chat_history.append(f"Observation: [output length: {len(tool_output)} chars]")


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
