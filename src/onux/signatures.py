from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, Any, Iterable, Literal, Mapping, get_args, get_origin

import annotated_types as at


@dataclass(frozen=True)
class Desc(at.BaseMetadata):
    """Human-readable description attached to a type.

    Examples:
        >>> Rating = Annotated[float, at.Ge(0), at.Le(5), Desc("star rating")]
    """

    text: str

    def __describe__(self) -> str:
        return self.text


_ANNOTATION_DESCRIPTIONS: dict[type, Any] = {
    at.Ge: lambda a: f"≥ {a.ge}",
    at.Le: lambda a: f"≤ {a.le}",
    at.Gt: lambda a: f"> {a.gt}",
    at.Lt: lambda a: f"< {a.lt}",
    at.MultipleOf: lambda a: f"multiple of {a.multiple_of}",
    at.Len: lambda a: (
        f"length {a.min_length}–{a.max_length}"
        if a.max_length is not None
        else f"length ≥ {a.min_length}"
    ),
}


@dataclass(frozen=True)
class Field:
    """One field in a signature.

    Attributes:
        name: Field name.
        role: One of ``"input"``, ``"hidden"``, or ``"output"``.
        type_: Python type, possibly ``Annotated`` with constraints.
    """

    name: str
    role: Literal["input", "hidden", "output"]
    type_: type = str

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

    @property
    def constraints(self) -> tuple[Any, ...]:
        return tuple(a for a in self.annotations if not isinstance(a, Desc))

    @property
    def desc(self) -> str | None:
        for a in self.annotations:
            if isinstance(a, Desc):
                return a.text
        return None

    @property
    def prefix(self) -> str:
        s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", self.name)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        return " ".join(w if w.isupper() else w.capitalize() for w in s.split("_"))

    def describe(self) -> str:
        return describe_type(self.type_)


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
_INTERVAL_RE = re.compile(r"^(\w+)\[([^:]*):([^:]*)\]$")
_ENUM_RE = re.compile(r"^\{([^}]+)\}$")


def describe_type(tp: Any) -> str:
    """Render a type as human-readable text.

    Examples:
        >>> describe_type(Annotated[float, at.Ge(0), at.Le(5)])
        'float (≥ 0, ≤ 5)'
        >>> describe_type(Literal["pos", "neg"])
        "one of: 'pos', 'neg'"
    """
    origin = get_origin(tp)

    if origin is Literal:
        return "one of: " + ", ".join(repr(v) for v in get_args(tp))

    if origin is Annotated:
        base, *annotations = get_args(tp)
        descs: list[str] = []
        constraints: list[str] = []
        for annotation in annotations:
            text = _describe_annotation(annotation)
            if text is None:
                continue
            if isinstance(annotation, Desc):
                descs.append(text)
            else:
                constraints.append(text)
        result = describe_type(base)
        if constraints:
            result += " (" + ", ".join(constraints) + ")"
        if descs:
            result = ", ".join(descs) + ": " + result
        return result

    if origin is not None:
        args = get_args(tp)
        name = getattr(origin, "__name__", str(origin))
        if args:
            return f"{name}[{', '.join(describe_type(arg) for arg in args)}]"
        return name

    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def _describe_annotation(annotation: Any) -> str | None:
    if hasattr(annotation, "__describe__"):
        return annotation.__describe__()
    for cls, fn in _ANNOTATION_DESCRIPTIONS.items():
        if isinstance(annotation, cls):
            return fn(annotation)
    return None


class ExamplesTable:
    """Duck-typed wrapper around tabular training examples.

    The goal is not to define a strict dataframe protocol, but to accept
    the common shapes users already have:

    - pandas DataFrame
    - polars DataFrame
    - duckdb relation
    - list/iterable of dict records

    Internally we only need four things:

    - column names
    - row count
    - record serialization
    - coarse type inference per column
    """

    __slots__ = ("raw",)

    def __init__(self, raw: Any):
        self.raw = raw

    @property
    def columns(self) -> list[str]:
        raw = self.raw
        if hasattr(raw, "columns"):
            return [str(column) for column in raw.columns]
        if hasattr(raw, "schema"):
            schema = raw.schema
            if isinstance(schema, Mapping):
                return [str(column) for column in schema.keys()]
            if hasattr(schema, "names"):
                return [str(column) for column in schema.names]
        records = self.to_records()
        return list(records[0].keys()) if records else []

    def __len__(self) -> int:
        raw = self.raw
        if hasattr(raw, "height"):
            return int(raw.height)
        try:
            return len(raw)
        except TypeError:
            return len(self.to_records())

    def infer_type(self, name: str) -> type:
        schema_type = _infer_type_from_schema(self.raw, name)
        if schema_type is not None:
            return schema_type

        for record in self.to_records():
            if name not in record:
                continue
            value = record[name]
            if value is None:
                continue
            return _infer_python_value_type(value)
        return str

    def to_records(self) -> list[dict[str, Any]]:
        raw = self.raw

        if hasattr(raw, "to_dicts"):
            records = raw.to_dicts()
        elif hasattr(raw, "to_dict"):
            try:
                records = raw.to_dict("records")
            except TypeError:
                records = None
            if records is None:
                raise TypeError("Unsupported tabular object: .to_dict() exists but does not support record output.")
        elif hasattr(raw, "arrow"):
            table = raw.arrow()
            if hasattr(table, "to_pylist"):
                records = table.to_pylist()
            else:
                raise TypeError("Unsupported tabular object: .arrow() result cannot be converted to records.")
        elif hasattr(raw, "to_arrow"):
            table = raw.to_arrow()
            if hasattr(table, "to_pylist"):
                records = table.to_pylist()
            else:
                raise TypeError("Unsupported tabular object: .to_arrow() result cannot be converted to records.")
        elif hasattr(raw, "df"):
            df = raw.df()
            if hasattr(df, "to_dict"):
                records = df.to_dict("records")
            else:
                raise TypeError("Unsupported tabular object: .df() result cannot be converted to records.")
        elif isinstance(raw, Iterable):
            records = list(raw)
        else:
            raise TypeError(
                "Examples must be dataframe-like (pandas/polars/duckdb-style) or an iterable of dict records."
            )

        normalized: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("Example rows must be mapping-like records.")
            normalized.append(dict(record))
        return normalized


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

    __slots__ = ("_fields", "_instructions", "_examples")

    def __init__(
        self,
        formula: str | None = None,
        *,
        data: Any | None = None,
        instructions: str | None = None,
        types: dict[str, type] | None = None,
        _fields: tuple[Field, ...] | None = None,
        _examples: ExamplesTable | None = None,
    ):
        if _fields is not None:
            object.__setattr__(self, "_fields", _fields)
            object.__setattr__(self, "_instructions", instructions or "")
            object.__setattr__(self, "_examples", _examples)
            return

        if formula is None:
            raise ValueError("Provide a formula like 'question -> answer'.")

        examples = _normalize_examples(data)
        fields = _parse_formula(formula, examples, types)

        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_examples", examples)

        if instructions is None:
            ins = ", ".join(f"`{f.name}`" for f in fields if f.role == "input")
            outs = ", ".join(f"`{f.name}`" for f in fields if f.role == "output")
            instructions = f"Given {ins}, produce {outs}."
        object.__setattr__(self, "_instructions", instructions)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Signature is immutable. Use .instruct(), .via(), .retype(), etc.")

    @property
    def formula(self) -> str:
        groups: list[list[str]] = [[], [], []]
        for field in self._fields:
            groups[_ROLE_ORDER[field.role]].append(field.name)
        return " -> ".join(", ".join(group) for group in groups if group)

    @property
    def instructions(self) -> str:
        return self._instructions

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

    def instruct(self, instructions: str) -> Signature:
        """Return a new signature with different instructions."""
        return Signature(_fields=self._fields, instructions=instructions, _examples=self._examples)

    def describe(self, **descriptions: str) -> Signature:
        """Return a new signature with updated field descriptions."""
        new_fields = []
        for field in self._fields:
            if field.name in descriptions:
                new_fields.append(Field(field.name, field.role, _add_annotation(field.type_, Desc(descriptions[field.name]))))
            else:
                new_fields.append(field)
        return Signature(_fields=tuple(new_fields), instructions=self._instructions, _examples=self._examples)

    def retype(self, **types: type) -> Signature:
        """Return a new signature with updated field types."""
        new_fields = tuple(
            Field(field.name, field.role, types[field.name]) if field.name in types else field
            for field in self._fields
        )
        return Signature(_fields=new_fields, instructions=self._instructions, _examples=self._examples)

    def via(self, name: str, type_: type = str, *, desc: str | None = None) -> Signature:
        """Return a new signature factored through one more hidden field."""
        if desc is not None:
            type_ = _add_annotation(type_, Desc(desc))
        fields = list(self._fields)
        insert_at = len(fields)
        for i, field in enumerate(fields):
            if field.role == "output":
                insert_at = i
                break
        fields.insert(insert_at, Field(name, "hidden", type_))
        return Signature(_fields=tuple(fields), instructions=self._instructions, _examples=self._examples)

    def add(self, name: str, type_: type = str, *, desc: str | None = None) -> Signature:
        """Return a new signature with one more output field."""
        if desc is not None:
            type_ = _add_annotation(type_, Desc(desc))
        return Signature(
            _fields=tuple([*self._fields, Field(name, "output", type_)]),
            instructions=self._instructions,
            _examples=self._examples,
        )

    def remove(self, name: str) -> Signature:
        """Return a new signature without the named field."""
        return Signature(
            _fields=tuple(field for field in self._fields if field.name != name),
            instructions=self._instructions,
            _examples=self._examples,
        )

    def with_examples(self, data: Any) -> Signature:
        """Return a new signature with different examples.

        Accepts dataframe-like tabular data (pandas, polars, duckdb-style)
        or an iterable of dict records.
        """
        return Signature(_fields=self._fields, instructions=self._instructions, _examples=_normalize_examples(data))

    def dump_state(self) -> dict[str, Any]:
        """Serialize prompt-facing state."""
        return {
            "formula": self.formula,
            "instructions": self._instructions,
            "fields": [
                {
                    "name": field.name,
                    "role": field.role,
                    "type": field.base_type.__name__,
                    "desc": field.desc,
                }
                for field in self._fields
            ],
            "examples": self._examples.to_records() if self._examples is not None else None,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Signature:
        """Restore a signature from ``dump_state``."""
        fields = []
        for item in state["fields"]:
            type_ = _BUILTIN_TYPES.get(item["type"], str)
            if item.get("desc"):
                type_ = _add_annotation(type_, Desc(item["desc"]))
            fields.append(Field(item["name"], item["role"], type_))
        examples = _normalize_examples(state["examples"]) if state.get("examples") is not None else None
        return cls(_fields=tuple(fields), instructions=state["instructions"], _examples=examples)

    def __repr__(self) -> str:
        width = max((len(field.name) for field in self._fields), default=0)
        lines = [self.formula, f"  {self._instructions!r}"]
        for field in self._fields:
            marker = {"input": "→", "hidden": "·", "output": "←"}[field.role]
            lines.append(f"  {marker} {field.name:<{width}}  {field.describe()}")
        if self._examples is not None:
            lines.append(f"  ({len(self._examples)} examples)")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Signature):
            return self._fields == other._fields and self._instructions == other._instructions
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._fields, self._instructions))


__all__ = ["Desc", "Field", "Signature", "describe_type"]


def _normalize_examples(data: Any | None) -> ExamplesTable | None:
    if data is None:
        return None
    if isinstance(data, ExamplesTable):
        return data
    return ExamplesTable(data)


def _add_annotation(tp: type, annotation: Any) -> type:
    if get_origin(tp) is Annotated:
        base, *existing = get_args(tp)
        filtered = [item for item in existing if type(item) is not type(annotation)]
        return Annotated[tuple([base, *filtered, annotation])]
    return Annotated[tp, annotation]


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

    match = _INTERVAL_RE.match(s)
    if match:
        base_name, low, high = match.group(1), match.group(2).strip(), match.group(3).strip()
        base = _resolve_simple_type(base_name, custom_types)
        annotations: list[Any] = []
        if base is str:
            annotations.append(at.Len(int(low) if low else 0, int(high) if high else None))
        else:
            if low:
                annotations.append(at.Ge(float(low) if base is float else int(low)))
            if high:
                annotations.append(at.Le(float(high) if base is float else int(high)))
        return Annotated[tuple([base, *annotations])]  # type: ignore[return-value]

    return _resolve_simple_type(s, custom_types)


def _resolve_simple_type(name: str, custom_types: dict[str, type] | None = None) -> type:
    if custom_types and name in custom_types:
        return custom_types[name]
    if name in _BUILTIN_TYPES:
        return _BUILTIN_TYPES[name]
    raise ValueError(f"Unknown type {name!r}.")


def _infer_type(data: ExamplesTable | None, name: str) -> type:
    if data is None or name not in data.columns:
        return str
    return data.infer_type(name)


def _infer_type_from_schema(raw: Any, name: str) -> type | None:
    # pandas-like: df[col].dtype.kind
    if hasattr(raw, "__getitem__") and hasattr(raw, "columns") and name in getattr(raw, "columns"):
        try:
            dtype = raw[name].dtype
            kind = getattr(dtype, "kind", None)
            if kind is not None:
                return {"f": float, "i": int, "u": int, "b": bool}.get(kind, str)
        except Exception:  # noqa: BLE001
            pass

    # polars-like: schema is a mapping from column -> dtype
    if hasattr(raw, "schema"):
        schema = raw.schema
        if isinstance(schema, Mapping) and name in schema:
            inferred = _infer_type_from_dtype_name(str(schema[name]))
            if inferred is not None:
                return inferred
        if hasattr(schema, "field"):
            try:
                inferred = _infer_type_from_dtype_name(str(schema.field(name).type))
                if inferred is not None:
                    return inferred
            except Exception:  # noqa: BLE001
                pass

    # duckdb-like: relation.columns + relation.types
    if hasattr(raw, "columns") and hasattr(raw, "types"):
        try:
            columns = list(raw.columns)
            index = columns.index(name)
            inferred = _infer_type_from_dtype_name(str(raw.types[index]))
            if inferred is not None:
                return inferred
        except Exception:  # noqa: BLE001
            pass

    return None


def _infer_type_from_dtype_name(dtype_name: str) -> type | None:
    text = dtype_name.upper()
    if any(token in text for token in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC")):
        return float
    if any(token in text for token in ("INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "UBIGINT", "UINTEGER")):
        return int
    if "BOOL" in text:
        return bool
    if any(token in text for token in ("LIST", "ARRAY")):
        return list
    if any(token in text for token in ("STRUCT", "MAP", "DICT")):
        return dict
    if any(token in text for token in ("STR", "TEXT", "VARCHAR", "UTF8", "STRING")):
        return str
    return None


def _infer_python_value_type(value: Any) -> type:
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, str):
        return str
    if isinstance(value, (list, tuple, set)):
        return list
    if isinstance(value, Mapping):
        return dict
    return type(value)
