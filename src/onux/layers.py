from __future__ import annotations

from typing import Any

from .core import FieldSpec, Layer, _normalize_inputs


class Generate(Layer):
    """One-shot generation layer."""

    module_name = "predict"


class ChainOfThought(Layer):
    """Generation layer with hidden reasoning."""

    module_name = "chain_of_thought"
    hidden_outputs = (
        FieldSpec(name="reasoning", annotation=str, desc="Step-by-step reasoning", kind="hidden"),
    )


class ReAct(Layer):
    """Tool-using reasoning layer with hidden reasoning and trace output."""

    module_name = "react"
    hidden_outputs = (
        FieldSpec(name="reasoning", annotation=str, desc="Deliberation before acting", kind="hidden"),
    )
    trace_outputs = (
        FieldSpec(name="trajectory", annotation=list[str], desc="Tool-use trace", kind="trace"),
    )


class Retrieve(Layer):
    """Retriever layer returning a list of context strings."""

    module_name = "retrieve"

    def default_output_decl(self):
        return ("context", list[str])


class ExecuteSQL(Layer):
    """Local execution layer for SQL."""

    module_name = "execute_sql"

    def default_output_decl(self):
        return ("rows", list[dict[str, Any]])


class Map(Layer):
    """Higher-order layer that maps an inner layer or model over a list input."""

    module_name = "map"

    def __init__(self, inner: Any, output: Any | None = None, **config: Any) -> None:
        self.inner = inner
        if output is None:
            raw_outputs = self._default_mapped_outputs()
        else:
            raw_outputs = output
        super().__init__(raw_outputs, **config)

    def _default_mapped_outputs(self):
        if hasattr(self.inner, "output_specs"):
            specs = []
            for spec in self.inner.output_specs:
                specs.append((spec.name, list[spec.annotation]))
            return tuple(specs)
        if hasattr(self.inner, "outputs"):
            specs = []
            for symbol in self.inner.outputs:
                specs.append((symbol.name, list[symbol.annotation]))
            return tuple(specs)
        return ("mapped", list[str])

    def __call__(self, inputs):
        parents = _normalize_inputs(inputs)
        if len(parents) != 1:
            raise ValueError("Map expects exactly one symbolic input.")
        return super().__call__(parents)


__all__ = [
    "ChainOfThought",
    "ExecuteSQL",
    "Generate",
    "Map",
    "ReAct",
    "Retrieve",
]
