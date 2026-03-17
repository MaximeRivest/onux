from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin

from .examples import ExamplesTable, infer_type as _infer_type, normalize_examples as _normalize_examples


@dataclass(frozen=True)
class Field:
    """One field in a signature.

    Attributes:
        name: Field name.
        role: One of ``"input"``, ``"hidden"``, or ``"output"``.
        type_: Python type.
        note: Optional human-readable field note.
    """

    name: str
    role: Literal["input", "hidden", "output"]
    type_: type = str
    note: str | None = None

    @property
    def base_type(self) -> type:
        if get_origin(self.type_) is Annotated:
            return get_args(self.type_)[0]
        return self.type_

    @property
    def annotations(self) -> tuple[Any, ...]:
        if get_origin(self.type_) is Annotated:
            return get_args(self.type_)[1:]
        return ()


_ROLE_ORDER = {"input": 0, "hidden": 1, "output": 2}
_BUILTIN_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
}
_ENUM_RE = re.compile(r"^\{([^}]+)\}$")


class Signature:
    """Declarative input/output contract for one semantic task.

    The formula is a short pipeline:

    - ``inputs -> outputs``
    - ``inputs -> hidden -> outputs``

    There are at most two arrows. Use ``.via()`` to add hidden fields.

    Examples:
        >>> Signature("question -> answer").formula
        'question -> answer'
        >>> Signature("question -> reasoning -> answer").formula
        'question -> reasoning -> answer'
    """

    __slots__ = ("_fields", "_hint", "_examples")

    def __init__(
        self,
        formula: str | None = None,
        *,
        data: Any | None = None,
        hint: str | None = None,
        types: dict[str, type] | None = None,
        _fields: tuple[Field, ...] | None = None,
        _examples: ExamplesTable | None = None,
    ):
        if _fields is not None:
            object.__setattr__(self, "_fields", _fields)
            object.__setattr__(self, "_hint", hint or "")
            object.__setattr__(self, "_examples", _examples)
            return

        if formula is None:
            raise ValueError("Provide a formula like 'question -> answer'.")

        examples = _normalize_examples(data)
        fields = _parse_formula(formula, examples, types)

        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_examples", examples)

        if hint is None:
            ins = ", ".join(f"`{f.name}`" for f in fields if f.role == "input")
            outs = ", ".join(f"`{f.name}`" for f in fields if f.role == "output")
            hint = f"Given {ins}, produce {outs}."
        object.__setattr__(self, "_hint", hint)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Signature is immutable. Use .hint(), .note(), .via(), .retype(), etc.")

    @property
    def formula(self) -> str:
        groups: list[list[str]] = [[], [], []]
        for field in self._fields:
            groups[_ROLE_ORDER[field.role]].append(field.name)
        return " -> ".join(", ".join(group) for group in groups if group)

    @property
    def examples(self) -> Any | None:
        return None if self._examples is None else self._examples.raw

    @property
    def n_examples(self) -> int:
        return 0 if self._examples is None else len(self._examples)

    @property
    def fields(self) -> OrderedDict[str, Field]:
        return OrderedDict((field.name, field) for field in self._fields)

    @property
    def input_fields(self) -> OrderedDict[str, Field]:
        return OrderedDict((field.name, field) for field in self._fields if field.role == "input")

    @property
    def hidden_fields(self) -> OrderedDict[str, Field]:
        return OrderedDict((field.name, field) for field in self._fields if field.role == "hidden")

    @property
    def output_fields(self) -> OrderedDict[str, Field]:
        return OrderedDict((field.name, field) for field in self._fields if field.role == "output")

    def hint(self, text: str) -> Signature:
        """Return a new signature with different task guidance."""
        return Signature(_fields=self._fields, hint=text, _examples=self._examples)

    def note(self, **notes: str) -> Signature:
        """Return a new signature with updated field notes."""
        new_fields = tuple(
            Field(field.name, field.role, field.type_, notes.get(field.name, field.note))
            for field in self._fields
        )
        return Signature(_fields=new_fields, hint=self._hint, _examples=self._examples)

    def retype(self, **types: type) -> Signature:
        """Return a new signature with updated field types."""
        new_fields = tuple(
            Field(field.name, field.role, types.get(field.name, field.type_), field.note)
            for field in self._fields
        )
        return Signature(_fields=new_fields, hint=self._hint, _examples=self._examples)

    def via(
        self,
        name: str,
        type_: type = str,
        *,
        note: str | None = None,
    ) -> Signature:
        """Return a new signature factored through one more hidden field."""
        fields = list(self._fields)
        insert_at = len(fields)
        for i, field in enumerate(fields):
            if field.role == "output":
                insert_at = i
                break
        fields.insert(insert_at, Field(name, "hidden", type_, note))
        return Signature(_fields=tuple(fields), hint=self._hint, _examples=self._examples)

    def add(
        self,
        name: str,
        type_: type = str,
        *,
        note: str | None = None,
    ) -> Signature:
        """Return a new signature with one more output field."""
        return Signature(
            _fields=tuple([*self._fields, Field(name, "output", type_, note)]),
            hint=self._hint,
            _examples=self._examples,
        )

    def remove(self, name: str) -> Signature:
        """Return a new signature without the named field."""
        return Signature(
            _fields=tuple(field for field in self._fields if field.name != name),
            hint=self._hint,
            _examples=self._examples,
        )

    def with_examples(self, data: Any) -> Signature:
        """Return a new signature with different examples.

        Accepts dataframe-like tabular data (pandas, polars, duckdb-style)
        or an iterable of dict records.
        """
        return Signature(_fields=self._fields, hint=self._hint, _examples=_normalize_examples(data))

    def dump_state(self) -> dict[str, Any]:
        """Serialize prompt-facing state."""
        return {
            "formula": self.formula,
            "hint": self._hint,
            "fields": [
                {
                    "name": field.name,
                    "role": field.role,
                    "type": field.base_type.__name__,
                    "note": field.note,
                }
                for field in self._fields
            ],
            "examples": self._examples.to_records() if self._examples is not None else None,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Signature:
        """Restore a signature from ``dump_state``."""
        fields = [
            Field(item["name"], item["role"], _BUILTIN_TYPES.get(item["type"], str), item.get("note"))
            for item in state["fields"]
        ]
        examples = _normalize_examples(state["examples"]) if state.get("examples") is not None else None
        return cls(_fields=tuple(fields), hint=state["hint"], _examples=examples)

    def __repr__(self) -> str:
        def format_type(tp: Any) -> str:
            origin = get_origin(tp)

            if origin is Literal:
                return "one of: " + ", ".join(repr(v) for v in get_args(tp))

            if origin is Annotated:
                base, *_ = get_args(tp)
                return format_type(base)

            if origin is not None:
                args = get_args(tp)
                name = getattr(origin, "__name__", str(origin))
                if args:
                    return f"{name}[{', '.join(format_type(arg) for arg in args)}]"
                return name

            if isinstance(tp, type):
                return tp.__name__
            return str(tp)

        width = max((len(field.name) for field in self._fields), default=0)
        lines = [self.formula, f"  {self._hint!r}"]
        for field in self._fields:
            marker = {"input": "→", "hidden": "·", "output": "←"}[field.role]
            text = format_type(field.type_)
            if field.note:
                text = f"{field.note}: {text}"
            lines.append(f"  {marker} {field.name:<{width}}  {text}")
        if self._examples is not None:
            lines.append(f"  ({len(self._examples)} examples)")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Signature):
            return self._fields == other._fields and self._hint == other._hint
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._fields, self._hint))


__all__ = ["Field", "Signature"]


def _parse_formula(
    formula: str,
    data: ExamplesTable | None = None,
    custom_types: dict[str, type] | None = None,
) -> tuple[Field, ...]:
    groups = [group.strip() for group in formula.split("->")]
    if len(groups) < 2:
        raise ValueError(f"Formula {formula!r} must contain at least one '->'.")
    if len(groups) > 3:
        raise ValueError(
            f"Formula {formula!r} has {len(groups) - 1} arrows. "
            "At most two '->' allowed (inputs -> hidden -> outputs). Use .via() for additional hidden stages."
        )

    input_group = groups[0]
    hidden_group = groups[1] if len(groups) == 3 else ""
    output_group = groups[-1]

    fields: list[Field] = []
    for name, type_ in _parse_field_list(input_group, custom_types):
        fields.append(Field(name, "input", type_ or _infer_type(data, name)))

    if len(fields) == 1 and fields[0].name == ".":
        if data is None:
            raise ValueError("Dot shorthand '.' requires dataframe-like example data.")
        excluded = set()
        for group in (hidden_group, output_group):
            for name, _ in _parse_field_list(group, custom_types):
                excluded.add(name)
        fields = [Field(column, "input", _infer_type(data, column)) for column in data.columns if column not in excluded]

    if hidden_group:
        for name, type_ in _parse_field_list(hidden_group, custom_types):
            fields.append(Field(name, "hidden", type_ or _infer_type(data, name)))

    for name, type_ in _parse_field_list(output_group, custom_types):
        fields.append(Field(name, "output", type_ or _infer_type(data, name)))

    return tuple(fields)



def _parse_field_list(s: str, custom_types: dict[str, type] | None = None) -> list[tuple[str, type | None]]:
    if not s:
        return []
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))

    fields: list[tuple[str, type | None]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, type_str = part.split(":", 1)
            fields.append((name.strip(), _resolve_type_str(type_str.strip(), custom_types)))
        else:
            fields.append((part, None))
    return fields



def _resolve_type_str(s: str, custom_types: dict[str, type] | None = None) -> type:
    match = _ENUM_RE.match(s)
    if match:
        values = tuple(v.strip() for v in match.group(1).split(","))
        return Literal[values]  # type: ignore[valid-type]

    if "[" in s and "]" in s and ":" in s:
        raise ValueError(
            "Type constraints like 'float[0:1]' are no longer supported. "
            "Use plain types and reflect constraints in your optimizer metric instead."
        )

    return _resolve_simple_type(s, custom_types)



def _resolve_simple_type(name: str, custom_types: dict[str, type] | None = None) -> type:
    if custom_types and name in custom_types:
        return custom_types[name]
    if name in _BUILTIN_TYPES:
        return _BUILTIN_TYPES[name]
    raise ValueError(f"Unknown type {name!r}.")
