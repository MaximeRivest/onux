from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin

from .examples import ExamplesTable, infer_type as _infer_type, normalize_examples as _normalize_examples


@dataclass(frozen=True)
class Field:
    """One field in a signature.

    Attributes
    ----------
    name : str
        Field name.
    role : {"input", "hidden", "output"}
        Field role within the signature.
    type_ : type, default=str
        Python type associated with the field. This may be wrapped in
        ``typing.Annotated``.
    note : str | None, default=None
        Optional human-readable field note.

    Examples
    --------
    >>> field = Field("score", "output", float, "Normalized score")
    >>> field.name
    'score'
    >>> field.role
    'output'
    >>> field.base_type is float
    True
    >>> field.note
    'Normalized score'
    """

    name: str
    role: Literal["input", "hidden", "output"]
    type_: type = str
    note: str | None = None

    @property
    def base_type(self) -> type:
        """Return the underlying field type.

        Returns
        -------
        type
            The field type with any ``typing.Annotated`` wrapper removed.

        Examples
        --------
        >>> field = Field("rating", "output", Annotated[int, "0-5 stars"])
        >>> field.base_type is int
        True
        """
        if get_origin(self.type_) is Annotated:
            return get_args(self.type_)[0]
        return self.type_

    @property
    def annotations(self) -> tuple[Any, ...]:
        """Return metadata attached through ``typing.Annotated``.

        Returns
        -------
        tuple[Any, ...]
            Annotation metadata for the field type. Returns an empty tuple when
            the type is not annotated.

        Examples
        --------
        >>> field = Field("rating", "output", Annotated[int, "0-5", "stars"])
        >>> field.annotations
        ('0-5', 'stars')
        """
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

    A signature is defined by a compact pipeline formula:

    - ``inputs -> outputs``
    - ``inputs -> hidden -> outputs``

    At most two arrows are allowed. Use :meth:`via` to insert additional hidden
    fields after construction.

    Examples
    --------
    Start with a minimal task contract:

    >>> basic = Signature("question -> answer")
    >>> basic.formula
    'question -> answer'

    Add an explicit hidden reasoning field:

    >>> hidden = Signature("question -> reasoning -> answer")
    >>> list(hidden.hidden_fields)
    ['reasoning']

    Add type information directly in the formula:

    >>> typed = Signature("question:str -> label:{yes, no}")
    >>> get_args(typed.output_fields["label"].type_)
    ('yes', 'no')

    Infer types from example data and expand inputs with ``.`` shorthand:

    >>> rows = [{"question": "2 + 2?", "context": ["math"], "answer": "4"}]
    >>> inferred = Signature(". -> answer", data=rows)
    >>> list(inferred.input_fields)
    ['question', 'context']
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
        """Create a signature from a formula or prebuilt fields.

        Parameters
        ----------
        formula : str | None, optional
            Formula describing the task, such as ``"question -> answer"``.
        data : Any | None, optional
            Tabular example data or an iterable of record mappings used to
            infer field types and attach examples.
        hint : str | None, optional
            Human-readable task guidance. When omitted, a default hint is
            derived from input and output field names.
        types : dict[str, type] | None, optional
            Custom type aliases keyed by the type names used inside the
            formula.
        _fields : tuple[Field, ...] | None, optional
            Internal field tuple used when cloning an existing signature.
        _examples : ExamplesTable | None, optional
            Internal normalized examples table used when cloning an existing
            signature.

        Raises
        ------
        ValueError
            If neither ``formula`` nor ``_fields`` is provided.

        Examples
        --------
        Construct the simplest possible signature:

        >>> sig = Signature("question -> answer")
        >>> sig.dump_state()["hint"]
        'Given `question`, produce `answer`.'

        Override the default task guidance:

        >>> guided = Signature("question -> answer", hint="Answer in one sentence.")
        >>> guided.dump_state()["hint"]
        'Answer in one sentence.'

        Register a custom type alias and use it in the formula:

        >>> typed = Signature("question -> answer:Score", types={"Score": float})
        >>> typed.output_fields["answer"].base_type is float
        True

        Mix formula types and example-driven inference:

        >>> rows = [{"question": "How many legs does a spider have?", "confidence": 0.9}]
        >>> inferred = Signature("question -> confidence", data=rows)
        >>> inferred.output_fields["confidence"].base_type is float
        True
        """
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
        raise AttributeError("Signature is immutable. Use .hint(), .note(), .via(), .type(), etc.")

    @property
    def formula(self) -> str:
        """Return the signature formula.

        Returns
        -------
        str
            Comma-separated fields grouped as inputs, hidden fields, and
            outputs, joined by ``->``.

        Examples
        --------
        >>> Signature("question -> reasoning -> answer").formula
        'question -> reasoning -> answer'
        >>> Signature("question -> answer").add("confidence", float).formula
        'question -> answer, confidence'
        """
        groups: list[list[str]] = [[], [], []]
        for field in self._fields:
            groups[_ROLE_ORDER[field.role]].append(field.name)
        return " -> ".join(", ".join(group) for group in groups if group)

    @property
    def examples(self) -> Any | None:
        """Return the raw examples attached to the signature.

        Returns
        -------
        Any | None
            The original example object passed to the signature, or ``None``.

        Examples
        --------
        >>> rows = [{"question": "Q", "answer": "A"}]
        >>> Signature("question -> answer", data=rows).examples == rows
        True
        >>> Signature("question -> answer").examples is None
        True
        """
        return None if self._examples is None else self._examples.raw

    @property
    def n_examples(self) -> int:
        """Return the number of attached examples.

        Returns
        -------
        int
            Number of example rows attached to the signature.

        Examples
        --------
        >>> rows = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        >>> Signature("question -> answer", data=rows).n_examples
        2
        """
        return 0 if self._examples is None else len(self._examples)

    @property
    def fields(self) -> OrderedDict[str, Field]:
        """Return all fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Ordered mapping of field names to :class:`Field` objects.

        Examples
        --------
        >>> list(Signature("question -> reasoning -> answer").fields)
        ['question', 'reasoning', 'answer']
        """
        return OrderedDict((field.name, field) for field in self._fields)

    @property
    def input_fields(self) -> OrderedDict[str, Field]:
        """Return input fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Ordered mapping containing only input fields.

        Examples
        --------
        >>> list(Signature("question, context -> answer").input_fields)
        ['question', 'context']
        """
        return OrderedDict((field.name, field) for field in self._fields if field.role == "input")

    @property
    def hidden_fields(self) -> OrderedDict[str, Field]:
        """Return hidden fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Ordered mapping containing only hidden fields.

        Examples
        --------
        >>> list(Signature("question -> reasoning, critique -> answer").hidden_fields)
        ['reasoning', 'critique']
        """
        return OrderedDict((field.name, field) for field in self._fields if field.role == "hidden")

    @property
    def output_fields(self) -> OrderedDict[str, Field]:
        """Return output fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Ordered mapping containing only output fields.

        Examples
        --------
        >>> list(Signature("question -> answer, confidence").output_fields)
        ['answer', 'confidence']
        """
        return OrderedDict((field.name, field) for field in self._fields if field.role == "output")

    def hint(self, text: str) -> Signature:
        """Return a copy with updated task guidance.

        Parameters
        ----------
        text : str
            New hint text.

        Returns
        -------
        Signature
            A new signature with the same fields and examples.

        Examples
        --------
        >>> sig = Signature("question -> answer").hint("Be concise.")
        >>> sig.dump_state()["hint"]
        'Be concise.'
        """
        return Signature(_fields=self._fields, hint=text, _examples=self._examples)

    def note(self, **notes: str) -> Signature:
        """Return a copy with updated field notes.

        Parameters
        ----------
        **notes : str
            Keyword arguments mapping field names to note text.

        Returns
        -------
        Signature
            A new signature with updated notes.

        Examples
        --------
        >>> sig = Signature("question -> answer").note(question="A factual question", answer="A short answer")
        >>> sig.fields["question"].note
        'A factual question'
        >>> sig.fields["answer"].note
        'A short answer'
        """
        new_fields = tuple(
            Field(field.name, field.role, field.type_, notes.get(field.name, field.note))
            for field in self._fields
        )
        return Signature(_fields=new_fields, hint=self._hint, _examples=self._examples)

    def retype(self, **types: type) -> Signature:
        """Return a copy with updated field types.

        Parameters
        ----------
        **types : type
            Keyword arguments mapping field names to replacement types.

        Returns
        -------
        Signature
            A new signature with updated field types.

        Examples
        --------
        >>> sig = Signature("question -> answer").retype(answer=float)
        >>> sig.output_fields["answer"].base_type is float
        True
        >>> Signature("question -> answer").output_fields["answer"].base_type is str
        True
        """
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
        """Return a copy with an additional hidden field.

        Parameters
        ----------
        name : str
            Name of the hidden field to insert.
        type_ : type, default=str
            Python type for the hidden field.
        note : str | None, optional
            Optional note describing the hidden field.

        Returns
        -------
        Signature
            A new signature with the hidden field inserted before outputs.

        Examples
        --------
        Insert a single hidden step:

        >>> sig = Signature("question -> answer").via("reasoning", note="Think step by step")
        >>> sig.formula
        'question -> reasoning -> answer'
        >>> sig.hidden_fields["reasoning"].note
        'Think step by step'

        Chain additional hidden steps progressively:

        >>> layered = Signature("question -> answer").via("draft").via("critique")
        >>> layered.formula
        'question -> draft, critique -> answer'
        """
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
        """Return a copy with an additional output field.

        Parameters
        ----------
        name : str
            Name of the output field to append.
        type_ : type, default=str
            Python type for the output field.
        note : str | None, optional
            Optional note describing the output field.

        Returns
        -------
        Signature
            A new signature with the extra output field.

        Examples
        --------
        >>> sig = Signature("question -> answer").add("confidence", float, note="0 to 1")
        >>> sig.formula
        'question -> answer, confidence'
        >>> sig.output_fields["confidence"].base_type is float
        True
        >>> sig.output_fields["confidence"].note
        '0 to 1'
        """
        return Signature(
            _fields=tuple([*self._fields, Field(name, "output", type_, note)]),
            hint=self._hint,
            _examples=self._examples,
        )

    def remove(self, name: str) -> Signature:
        """Return a copy without one field.

        Parameters
        ----------
        name : str
            Name of the field to remove.

        Returns
        -------
        Signature
            A new signature without the named field.

        Examples
        --------
        >>> sig = Signature("question -> reasoning -> answer").remove("reasoning")
        >>> sig.formula
        'question -> answer'
        >>> list(sig.hidden_fields)
        []
        """
        return Signature(
            _fields=tuple(field for field in self._fields if field.name != name),
            hint=self._hint,
            _examples=self._examples,
        )

    def with_examples(self, data: Any) -> Signature:
        """Return a copy with different examples.

        Parameters
        ----------
        data : Any
            Dataframe-like tabular data (for example pandas, polars, or
            duckdb-style objects) or an iterable of record dictionaries.

        Returns
        -------
        Signature
            A new signature with normalized examples attached.

        Examples
        --------
        Attach a small training set:

        >>> rows = [{"question": "Q", "answer": "A"}]
        >>> sig = Signature("question -> answer").with_examples(rows)
        >>> sig.n_examples
        1
        >>> sig.examples == rows
        True

        Replace examples while keeping the same field structure:

        >>> updated = sig.with_examples([{"question": "Q2", "answer": "A2"}, {"question": "Q3", "answer": "A3"}])
        >>> updated.formula
        'question -> answer'
        >>> updated.n_examples
        2
        """
        return Signature(_fields=self._fields, hint=self._hint, _examples=_normalize_examples(data))

    def dump_state(self) -> dict[str, Any]:
        """Serialize prompt-facing state.

        Returns
        -------
        dict[str, Any]
            JSON-serializable state that can be restored with
            :meth:`load_state`.

        Examples
        --------
        >>> state = Signature("question -> answer").note(answer="Short answer").dump_state()
        >>> state["formula"]
        'question -> answer'
        >>> state["fields"][1]["note"]
        'Short answer'
        """
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
        """Restore a signature from serialized state.

        Parameters
        ----------
        state : dict[str, Any]
            State produced by :meth:`dump_state`.

        Returns
        -------
        Signature
            Restored signature instance.

        Examples
        --------
        Round-trip a basic signature:

        >>> original = Signature("question -> answer").note(answer="Short answer")
        >>> restored = Signature.load_state(original.dump_state())
        >>> restored == original
        True

        Preserve attached examples across serialization:

        >>> sig = Signature("question -> answer", data=[{"question": "Q", "answer": "A"}])
        >>> Signature.load_state(sig.dump_state()).n_examples
        1
        """
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
