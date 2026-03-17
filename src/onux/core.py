from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any, Iterable, Sequence

from .signatures import Desc, ExamplesTable, Field, Signature, _normalize_examples, describe_type

_CallId = count(1)


@dataclass(slots=True)
class FieldSpec:
    """Specification for one symbolic field in a layer.

    Attributes:
        name: Field name.
        annotation: Python type annotation.
        desc: Human-readable description.
        kind: One of ``"public"``, ``"hidden"``, or ``"trace"``.
    """

    name: str
    annotation: Any = str
    desc: str = ""
    kind: str = "public"


@dataclass(slots=True)
class LayerCall:
    """One concrete application of a layer inside a graph."""

    id: int
    layer_name: str
    layer_type: str
    inputs: tuple["Symbol", ...]
    config: dict[str, Any] = field(default_factory=dict)
    layer: Any | None = None
    outputs: tuple["Symbol", ...] = ()

    @property
    def signature(self) -> Signature | None:
        if self.layer is None:
            return None
        if hasattr(self.layer, "build_signature"):
            return self.layer.build_signature(self.inputs)
        return None


@dataclass(slots=True)
class Symbol:
    """A symbolic node in an Onux graph.

    Symbols are semantic, not tensor-shaped.  They carry a field name,
    a Python type, and optional description metadata.
    """

    name: str
    annotation: Any = str
    desc: str = ""
    producer: LayerCall | None = None
    role: str = "public"

    @property
    def is_input(self) -> bool:
        return self.producer is None

    def short_type(self) -> str:
        return describe_type(self.annotation)

    def __repr__(self) -> str:
        return f"Symbol(name={self.name!r}, type={self.short_type()}, role={self.role!r})"


OutputDecl = str | tuple[str, Any]
OutputDecls = OutputDecl | Sequence[OutputDecl]
SymbolLike = Symbol | Sequence[Symbol]


def Input(name: str, type: Any = str, desc: str = "") -> Symbol:
    """Create a symbolic semantic input node.

    Examples:
        >>> question = Input("question")
        >>> context = Input("context", type=list[str])
    """
    return Symbol(name=name, annotation=type, desc=desc, role="input")


class Layer:
    """Base class for Keras-style symbolic compute layers.

    Subclasses declare default hidden or trace outputs by setting the
    class attributes ``hidden_outputs`` and ``trace_outputs``.
    """

    hidden_outputs: tuple[FieldSpec, ...] = ()
    trace_outputs: tuple[FieldSpec, ...] = ()
    module_name: str = "predict"

    def __init__(
        self,
        output: OutputDecl | None = None,
        *,
        outputs: OutputDecls | None = None,
        expose: Sequence[str] = (),
        instructions: str | None = None,
        desc: str | dict[str, str] | None = None,
        **config: Any,
    ) -> None:
        if output is not None and outputs is not None:
            raise TypeError("Pass either `output` or `outputs`, not both.")

        raw_outputs = outputs if outputs is not None else output
        if raw_outputs is None:
            raw_outputs = self.default_output_decl()

        self.output_specs = _normalize_output_specs(raw_outputs, desc=desc)
        self.expose = tuple(expose)
        self.instructions = instructions or ""
        self.config = dict(config)

    def default_output_decl(self) -> OutputDecls:
        return self.default_output_name()

    def default_output_name(self) -> str:
        return self.__class__.__name__.lower()

    def semantic_hidden_outputs(self) -> tuple[FieldSpec, ...]:
        return self.hidden_outputs

    def all_internal_outputs(self) -> tuple[FieldSpec, ...]:
        return tuple([*self.hidden_outputs, *self.trace_outputs])

    def build_signature(self, inputs: Sequence[Symbol]) -> Signature:
        """Build the inner signature for one layer call."""
        fields: list[Field] = []
        for symbol in inputs:
            fields.append(Field(symbol.name, "input", _symbol_type(symbol)))
        for spec in self.semantic_hidden_outputs():
            fields.append(Field(spec.name, "hidden", _spec_type(spec)))
        for spec in self.output_specs:
            fields.append(Field(spec.name, "output", _spec_type(spec)))
        return Signature(
            _fields=tuple(fields),
            instructions=self.instructions or self.default_instructions(inputs),
        )

    def default_instructions(self, inputs: Sequence[Symbol]) -> str:
        inputs_text = ", ".join(f"`{symbol.name}`" for symbol in inputs)
        outputs_text = ", ".join(f"`{spec.name}`" for spec in self.output_specs)
        return f"Given {inputs_text}, produce {outputs_text}."

    def __call__(self, inputs: SymbolLike) -> Symbol | tuple[Symbol, ...]:
        parents = _normalize_inputs(inputs)
        call = LayerCall(
            id=next(_CallId),
            layer_name=self.display_name,
            layer_type=self.__class__.__name__,
            inputs=parents,
            config={**self.config, "module": self.module_name},
            layer=self,
        )

        internal_specs = {spec.name: spec for spec in self.all_internal_outputs()}
        returned_specs: list[FieldSpec] = []
        for name in self.expose:
            if name not in internal_specs:
                available = ", ".join(sorted(internal_specs)) or "<none>"
                raise ValueError(f"Unknown exposed output {name!r}. Available: {available}.")
            returned_specs.append(internal_specs[name])
        returned_specs.extend(self.output_specs)

        symbols = tuple(
            Symbol(
                name=spec.name,
                annotation=spec.annotation,
                desc=spec.desc,
                producer=call,
                role=spec.kind,
            )
            for spec in returned_specs
        )
        call.outputs = symbols
        return _unwrap(symbols)

    @property
    def display_name(self) -> str:
        return ", ".join(spec.name for spec in self.output_specs)

    def get_config(self) -> dict[str, Any]:
        return {**self.config, "instructions": self.instructions, "expose": self.expose}

    def __repr__(self) -> str:
        names = [spec.name for spec in self.output_specs]
        return f"{self.__class__.__name__}(outputs={names!r})"


class Model:
    """A closed symbolic graph with named inputs and outputs.

    Models are reusable graph modules: a compiled model can be called on
    new symbolic inputs and used inside a larger graph.
    """

    def __init__(
        self,
        *,
        inputs: SymbolLike,
        outputs: SymbolLike,
        name: str | None = None,
    ) -> None:
        self.inputs = _normalize_inputs(inputs)
        self.outputs = _normalize_inputs(outputs)
        self.name = name or "model"
        self.compile_config: dict[str, Any] | None = None
        self.training_examples: ExamplesTable | None = None

    def __call__(self, inputs: SymbolLike) -> Symbol | tuple[Symbol, ...]:
        parents = _normalize_inputs(inputs)
        if len(parents) != len(self.inputs):
            raise ValueError(f"Model {self.name!r} expects {len(self.inputs)} inputs, got {len(parents)}.")
        call = LayerCall(
            id=next(_CallId),
            layer_name=self.name,
            layer_type="Model",
            inputs=parents,
            config={"kind": "subgraph"},
            layer=self,
        )
        symbols = tuple(
            Symbol(
                name=output.name,
                annotation=output.annotation,
                desc=output.desc,
                producer=call,
                role=output.role,
            )
            for output in self.outputs
        )
        call.outputs = symbols
        return _unwrap(symbols)

    @property
    def signature(self) -> Signature:
        fields: list[Field] = []
        for symbol in self.inputs:
            fields.append(Field(symbol.name, "input", _symbol_type(symbol)))
        for symbol in self.outputs:
            fields.append(Field(symbol.name, "output", _symbol_type(symbol)))
        return Signature(_fields=tuple(fields), instructions=f"Run the model {self.name!r}.")

    def build_signature(self, inputs: Sequence[Symbol]) -> Signature:
        fields: list[Field] = []
        for symbol in inputs:
            fields.append(Field(symbol.name, "input", _symbol_type(symbol)))
        for output in self.outputs:
            fields.append(Field(output.name, "output", _symbol_type(output)))
        return Signature(_fields=tuple(fields), instructions=f"Run the sub-model {self.name!r}.")

    def compile(self, **config: Any) -> Model:
        self.compile_config = dict(config)
        return self

    def fit(self, examples: Any) -> Model:
        """Attach dataframe-like training examples to the model.

        Accepts pandas, polars, duckdb-style tables, or an iterable of
        dict records.
        """
        table = _normalize_examples(examples)
        if table is None:
            raise ValueError("Examples cannot be None.")
        allowed = {symbol.name for symbol in self.inputs} | {symbol.name for symbol in self.outputs}
        unknown = sorted(set(table.columns) - allowed)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Examples contain unknown fields: {names}")
        self.training_examples = table
        return self

    def calls(self) -> list[LayerCall]:
        return _collect_calls(self.outputs)

    def layer_signatures(self) -> list[tuple[LayerCall, Signature]]:
        pairs: list[tuple[LayerCall, Signature]] = []
        for call in self.calls():
            if call.signature is not None:
                pairs.append((call, call.signature))
        return pairs

    def summary(self) -> str:
        lines = [f'Model: "{self.name}"', "Inputs:"]
        for symbol in self.inputs:
            lines.append(f"  - {symbol.name}: {symbol.short_type()}")
        lines.append("Outputs:")
        for symbol in self.outputs:
            lines.append(f"  - {symbol.name}: {symbol.short_type()} ({symbol.role})")
        lines.append("Graph:")
        for call in self.calls():
            parents = ", ".join(parent.name for parent in call.inputs)
            outputs = ", ".join(output.name for output in call.outputs)
            module = call.config.get("module")
            suffix = f", module={module}" if module else ""
            lines.append(f"  - [{call.id}] {call.layer_type}({outputs}) <- {parents}{suffix}")
        if self.compile_config is not None:
            lines.append(f"Compile: {self.compile_config}")
        if self.training_examples is not None:
            lines.append(f"Examples: {len(self.training_examples)} rows")
        summary = "\n".join(lines)
        print(summary)
        return summary

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"


def _normalize_output_specs(
    value: OutputDecls,
    *,
    desc: str | dict[str, str] | None = None,
) -> tuple[FieldSpec, ...]:
    if isinstance(value, str):
        text = desc if isinstance(desc, str) else ""
        return (FieldSpec(name=value, desc=text),)
    if _is_typed_output(value):
        name, annotation = value
        text = desc if isinstance(desc, str) else ""
        return (FieldSpec(name=name, annotation=annotation, desc=text),)

    specs: list[FieldSpec] = []
    desc_map = desc if isinstance(desc, dict) else {}
    for item in value:
        if isinstance(item, str):
            specs.append(FieldSpec(name=item, desc=desc_map.get(item, "")))
        elif _is_typed_output(item):
            name, annotation = item
            specs.append(FieldSpec(name=name, annotation=annotation, desc=desc_map.get(name, "")))
        else:
            raise TypeError(f"Unsupported output declaration: {item!r}")
    if not specs:
        raise ValueError("At least one output must be declared.")
    return tuple(specs)


def _normalize_inputs(value: SymbolLike) -> tuple[Symbol, ...]:
    if isinstance(value, Symbol):
        return (value,)
    if isinstance(value, Sequence):
        symbols: list[Symbol] = []
        for item in value:
            if not isinstance(item, Symbol):
                raise TypeError(f"Expected Symbol inputs, got {type(item).__name__}.")
            symbols.append(item)
        if not symbols:
            raise ValueError("At least one Symbol input is required.")
        return tuple(symbols)
    raise TypeError(f"Expected Symbol or sequence of Symbol, got {type(value).__name__}.")


def _collect_calls(outputs: Sequence[Symbol]) -> list[LayerCall]:
    ordered: list[LayerCall] = []
    seen_symbols: set[int] = set()
    seen_calls: set[int] = set()

    def visit(symbol: Symbol) -> None:
        symbol_id = id(symbol)
        if symbol_id in seen_symbols:
            return
        seen_symbols.add(symbol_id)
        if symbol.producer is None:
            return
        for parent in symbol.producer.inputs:
            visit(parent)
        if symbol.producer.id not in seen_calls:
            seen_calls.add(symbol.producer.id)
            ordered.append(symbol.producer)

    for output in outputs:
        visit(output)
    return ordered


def _unwrap(symbols: tuple[Symbol, ...]) -> Symbol | tuple[Symbol, ...]:
    if len(symbols) == 1:
        return symbols[0]
    return symbols


def _is_typed_output(value: Any) -> bool:
    """Return True for typed output declarations like ``("score", float)``.

    A plain 2-tuple of strings like ``("notes", "sources")`` should be
    treated as *two* outputs, not one typed output.
    """
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and not isinstance(value[1], str)
    )


def _spec_type(spec: FieldSpec) -> type:
    if spec.desc:
        return _add_desc(spec.annotation, spec.desc)
    return spec.annotation


def _symbol_type(symbol: Symbol) -> type:
    if symbol.desc:
        return _add_desc(symbol.annotation, symbol.desc)
    return symbol.annotation


def _add_desc(annotation: type, desc: str) -> type:
    from typing import Annotated

    if desc:
        return Annotated[annotation, Desc(desc)]
    return annotation


__all__ = ["FieldSpec", "Input", "Layer", "LayerCall", "Model", "Symbol"]
