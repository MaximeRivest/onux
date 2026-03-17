from __future__ import annotations

from functools import partial
from typing import Callable

from .signatures import Signature


def predict(sig: Signature, inputs: dict, *, lm, adapter) -> dict:
    """Run one LLM call for a signature."""
    prompt = adapter.render(sig, inputs)
    response = lm(prompt)
    return adapter.parse(sig, response)


def chain_of_thought(
    sig: Signature,
    inputs: dict,
    *,
    lm,
    adapter,
    desc: str = "Think step by step",
) -> dict:
    """Add one hidden reasoning field, call once, then discard it."""
    enriched = sig.via("reasoning", desc=desc)
    result = predict(enriched, inputs, lm=lm, adapter=adapter)
    return {name: value for name, value in result.items() if name in sig.output_fields}


def react(
    sig: Signature,
    inputs: dict,
    *,
    lm,
    adapter,
    tools: list[Callable],
    max_iter: int = 5,
) -> dict:
    """Simple think/act/observe loop with local tools."""
    state = inputs.copy()
    state.setdefault("observation", "")

    for _ in range(max_iter):
        step_sig = sig.via("thought", desc="What to do next").via("action", desc="Tool to call, or 'finish'")
        result = predict(step_sig, state, lm=lm, adapter=adapter)
        action = str(result.get("action", "")).strip()
        if action.lower() == "finish":
            return {name: value for name, value in result.items() if name in sig.output_fields}
        tool = _find_tool(tools, action)
        if tool is None:
            state["observation"] = f"Unknown tool: {action}"
        else:
            state["observation"] = str(tool(result.get("action_input", action)))

    return predict(sig, inputs, lm=lm, adapter=adapter)


def refine(
    sig: Signature,
    inputs: dict,
    *,
    lm,
    adapter,
    check: Callable[[dict], str | None],
    max_retries: int = 3,
) -> dict:
    """Generate, validate, and feed feedback back into the next attempt."""
    state = inputs.copy()
    result: dict = {}

    for attempt in range(max_retries):
        result = predict(sig, state, lm=lm, adapter=adapter)
        feedback = check(result)
        if feedback is None:
            return result
        state["feedback"] = f"Attempt {attempt + 1} failed: {feedback}"

    return result


def code_exec(
    sig: Signature,
    inputs: dict,
    *,
    lm,
    adapter,
    lint: Callable[[str], str | None] | None = None,
    max_fixes: int = 3,
) -> dict:
    """Generate Python code, optionally lint it, then execute it."""
    code_sig = sig.via("code", desc="Python code that solves the problem")
    state = inputs.copy()
    result: dict = {}

    for _ in range(max_fixes):
        result = predict(code_sig, state, lm=lm, adapter=adapter)
        code = str(result.get("code", ""))
        if lint is not None:
            lint_error = lint(code)
            if lint_error:
                state["lint_error"] = lint_error
                continue
        try:
            namespace: dict = {}
            exec(code, namespace)
            return {name: namespace.get(name, result.get(name)) for name in sig.output_fields}
        except Exception as exc:  # noqa: BLE001
            state["execution_error"] = str(exc)

    return {name: result.get(name) for name in sig.output_fields}


def pipe(*steps: tuple[Signature, Callable]):
    """Compose signature/module steps sequentially."""

    def module(sig: Signature, inputs: dict, *, lm, adapter) -> dict:
        state = inputs.copy()
        for step_sig, step_fn in steps:
            result = step_fn(step_sig, state, lm=lm, adapter=adapter)
            state.update(result)
        return {name: value for name, value in state.items() if name in sig.output_fields}

    module.__name__ = "pipe"
    module.__doc__ = " -> ".join(module_name(fn) for _, fn in steps)
    return module


def fallback(*module_fns: Callable):
    """Try modules in order and return the first successful result."""

    def module(sig: Signature, inputs: dict, *, lm, adapter) -> dict:
        last_error: Exception | None = None
        for fn in module_fns:
            try:
                return fn(sig, inputs, lm=lm, adapter=adapter)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is None:
            raise RuntimeError("No fallback modules provided.")
        raise last_error

    module.__name__ = "fallback"
    return module


def ensemble(*module_fns: Callable, pick: Callable[[list[dict]], dict] | None = None):
    """Run several modules and pick one result."""

    def module(sig: Signature, inputs: dict, *, lm, adapter) -> dict:
        results = [fn(sig, inputs, lm=lm, adapter=adapter) for fn in module_fns]
        return pick(results) if pick is not None else results[0]

    module.__name__ = "ensemble"
    return module


def module_name(fn: Callable) -> str:
    """Return a readable name for a module function or partial."""
    if isinstance(fn, partial):
        args = ", ".join(f"{key}={value!r}" for key, value in (fn.keywords or {}).items())
        return f"{fn.func.__name__}({args})" if args else fn.func.__name__
    return getattr(fn, "__name__", type(fn).__name__)


__all__ = [
    "chain_of_thought",
    "code_exec",
    "ensemble",
    "fallback",
    "module_name",
    "pipe",
    "predict",
    "react",
    "refine",
]


def _find_tool(tools: list[Callable], action: str) -> Callable | None:
    for tool in tools:
        if tool.__name__.lower() in action.lower():
            return tool
    return None
