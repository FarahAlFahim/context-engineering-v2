"""LLM initialization helpers with model-specific quirks."""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI


# ---------- prompt loading ----------

def load_prompt(name: str, prompts_dir: str = "prompts") -> str:
    """Load a prompt template from the prompts/ directory."""
    path = Path(prompts_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------- model helpers ----------

def _is_gpt5_model(model_name: str) -> bool:
    m = (model_name or "").lower().strip()
    return m.startswith("gpt-5")


def _normalize_temperature(model_name: str, temperature):
    """GPT-5-mini: enforce temperature==1. Other models: pass through."""
    if _is_gpt5_model(model_name):
        return 1
    return 0 if temperature is None else temperature


def _wrap_drop_stop_for_gpt5(llm):
    """Return an LLM wrapper so GPT-5 calls never receive `stop`.

    LangChain v0.2+ validates that `llm` is a `Runnable`. This wrapper
    subclasses Runnable and delegates while stripping stop-related kwargs.
    """
    from langchain_core.runnables import Runnable

    class _DropStopRunnable(Runnable):
        def __init__(self, inner):
            super().__init__()
            self._llm = inner

        def __getattr__(self, name):
            return getattr(self._llm, name)

        @staticmethod
        def _strip_stop(kwargs):
            if kwargs:
                kwargs.pop("stop", None)
                kwargs.pop("stop_sequences", None)
            return kwargs

        def invoke(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return self._llm.invoke(input, config=config, **kwargs)

        async def ainvoke(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return await self._llm.ainvoke(input, config=config, **kwargs)

        def batch(self, inputs, config=None, **kwargs):
            self._strip_stop(kwargs)
            return self._llm.batch(inputs, config=config, **kwargs)

        async def abatch(self, inputs, config=None, **kwargs):
            self._strip_stop(kwargs)
            return await self._llm.abatch(inputs, config=config, **kwargs)

        def stream(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return self._llm.stream(input, config=config, **kwargs)

        async def astream(self, input, config=None, **kwargs):
            self._strip_stop(kwargs)
            return await self._llm.astream(input, config=config, **kwargs)

        def generate(self, messages, stop=None, **kwargs):
            stop = None
            self._strip_stop(kwargs)
            return self._llm.generate(messages, stop=stop, **kwargs)

        async def agenerate(self, messages, stop=None, **kwargs):
            stop = None
            self._strip_stop(kwargs)
            return await self._llm.agenerate(messages, stop=stop, **kwargs)

        def __call__(self, *args, **kwargs):
            self._strip_stop(kwargs)
            return self._llm(*args, **kwargs)

    if isinstance(llm, _DropStopRunnable):
        return llm
    return _DropStopRunnable(llm)


def make_chat_llm(model_name: str, temperature=None):
    """Create a ChatOpenAI instance with model-specific compatibility handling."""
    temp = _normalize_temperature(model_name, temperature)
    base = ChatOpenAI(model=model_name, temperature=temp)
    return _wrap_drop_stop_for_gpt5(base) if _is_gpt5_model(model_name) else base
