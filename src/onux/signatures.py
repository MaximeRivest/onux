from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

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
        Python type associated with the field.
    note : str | None, default=None
        Optional human-readable field note.

    Examples
    --------
    >>> field = Field("score", "output", float, "Normalized score")
    >>> field.name
    'score'
    >>> field.role
    'output'
    >>> field.type_ is float
    True
    >>> field.note
    'Normalized score'
    """

    name: str
    role: Literal["input", "hidden", "output"]
    type_: type = str
    note: str | None = None


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


class Signature:
    """Declarative input/output contract for one semantic task.

    A signature is defined by a compact pipeline formula containing field
    names only:

    - ``inputs -> outputs``
    - ``inputs -> hidden -> outputs``

    At most two arrows are allowed. Build the formula from field names, then
    refine the signature with methods such as :meth:`type`, :meth:`via`, and
    :meth:`add`.

    Examples
    --------
    Start with a minimal task contract:

    >>> basic = Signature("question -> answer")
    >>> basic.formula
    'question -> answer'

    Add types after construction with :meth:`type`:

    >>> typed = Signature("question -> answer").type(answer=float)
    >>> typed.output_fields["answer"].type_ is float
    True

    Add an explicit hidden reasoning field with :meth:`via`:

    >>> hidden = Signature("question -> answer").via("reasoning", type_=str)
    >>> hidden.formula
    'question -> reasoning -> answer'

    Add more outputs with commas in the formula or with :meth:`add`:

    >>> multi = Signature("question, context -> answer").add("confidence", float)
    >>> list(multi.output_fields)
    ['answer', 'confidence']

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
        _fields: tuple[Field, ...] | None = None,
        _examples: ExamplesTable | None = None,
    ):
        """Create a signature from a formula or prebuilt fields.

        Parameters
        ----------
        formula : str | None, optional
            Formula describing the task as comma-separated field names, such as
            ``"question -> answer"`` or ``"question -> reasoning -> answer"``.
        data : Any | None, optional
            Tabular example data or an iterable of record mappings used to
            infer field types and attach examples.
        hint : str | None, optional
            Human-readable task guidance. When omitted, a default hint is
            derived from input and output field names.
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

        Keep the formula limited to names and add types afterward:

        >>> typed = Signature("question -> answer").type(question=str, answer=float)
        >>> typed.input_fields["question"].type_ is str
        True
        >>> typed.output_fields["answer"].type_ is float
        True

        Add typed hidden and output fields with :meth:`via` and :meth:`add`:

        >>> richer = Signature("question -> answer").via("reasoning", type_=list[str]).add("confidence", float)
        >>> richer.hidden_fields["reasoning"].type_
        list[str]
        >>> richer.output_fields["confidence"].type_ is float
        True

        Infer types from examples when you do not want to call :meth:`type`:

        >>> rows = [{"question": "How many legs does a spider have?", "confidence": 0.9}]
        >>> inferred = Signature("question -> confidence", data=rows)
        >>> inferred.output_fields["confidence"].type_ is float
        True

        Combine structure, notes, and types progressively:

        >>> rich = (
        ...     Signature("question -> answer")
        ...     .via("reasoning", note="Intermediate reasoning")
        ...     .add("confidence", float, note="0 to 1")
        ...     .type(answer=str)
        ... )
        >>> rich.formula
        'question -> reasoning -> answer, confidence'
        >>> rich.output_fields["confidence"].note
        '0 to 1'
        """
        if _fields is not None:
            object.__setattr__(self, "_fields", _fields)
            object.__setattr__(self, "_hint", hint or "")
            object.__setattr__(self, "_examples", _examples)
            return

        if formula is None:
            raise ValueError("Provide a formula like 'question -> answer'.")

        examples = _normalize_examples(data)
        fields = _parse_formula(formula, examples)

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

    def type(self, **types: type) -> Signature:
        """Return a copy with updated field types.

        Use this method to assign or update field types while keeping the
        formula focused on field names and structure.

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
        Start with an untyped formula and set one field type:

        >>> sig = Signature("question -> answer").type(answer=float)
        >>> sig.output_fields["answer"].type_ is float
        True

        Update several fields at once:

        >>> scored = Signature("question, context -> answer, confidence").type(context=list[str], confidence=float)
        >>> scored.input_fields["context"].type_
        list[str]
        >>> scored.output_fields["confidence"].type_ is float
        True

        The original signature remains unchanged:

        >>> Signature("question -> answer").output_fields["answer"].type_ is str
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

        Notes
        -----
        This method is useful for introducing intermediate hidden fields while
        keeping the main formula focused on the task structure.

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

        Notes
        -----
        This method is useful for extending an existing signature with
        additional outputs after the initial task structure is in place.

        Examples
        --------
        >>> sig = Signature("question -> answer").add("confidence", float, note="0 to 1")
        >>> sig.formula
        'question -> answer, confidence'
        >>> sig.output_fields["confidence"].type_ is float
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
                    "type": _type_name(field.type_),
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


def _type_name(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is not None:
        return getattr(origin, "__name__", str(origin))
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)


def _parse_formula(
    formula: str,
    data: ExamplesTable | None = None,
) -> tuple[Field, ...]:
    groups = [group.strip() for group in formula.split("->")]
    if len(groups) < 2:
        raise ValueError(
            f"Invalid signature formula {formula!r}. "
            "Use 'input1, input2 -> output1, output2' or 'input1 -> hidden1, hidden2 -> output1'."
        )
    if len(groups) > 3:
        raise ValueError(
            f"Invalid signature formula {formula!r}. "
            "Use at most two '->': 'input1, input2 -> output1, output2' or "
            "'input1 -> hidden1, hidden2 -> output1'."
        )

    input_group = groups[0]
    hidden_group = groups[1] if len(groups) == 3 else ""
    output_group = groups[-1]

    fields: list[Field] = []
    for name in _parse_field_list(input_group):
        fields.append(Field(name, "input", _infer_type(data, name)))

    if len(fields) == 1 and fields[0].name == ".":
        if data is None:
            raise ValueError("Dot shorthand '.' requires dataframe-like example data.")
        excluded = set()
        for group in (hidden_group, output_group):
            for name in _parse_field_list(group):
                excluded.add(name)
        fields = [Field(column, "input", _infer_type(data, column)) for column in data.columns if column not in excluded]

    if hidden_group:
        for name in _parse_field_list(hidden_group):
            fields.append(Field(name, "hidden", _infer_type(data, name)))

    for name in _parse_field_list(output_group):
        fields.append(Field(name, "output", _infer_type(data, name)))

    return tuple(fields)



def _parse_field_list(s: str) -> list[str]:
    if not s:
        return []

    fields: list[str] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if not _is_valid_field_name(part):
            raise ValueError(
                f"Invalid field entry {part!r}. "
                "Signature formulas accept field names only, for example 'question, context -> answer' or "
                "'question -> reasoning -> answer'. To set types, use .type(answer=float); "
                "to add typed hidden/output fields, use .via('reasoning', type_=str) or "
                ".add('confidence', float); or provide example data for inference."
            )
        fields.append(part)
    return fields



def _is_valid_field_name(name: str) -> bool:
    if name == ".":
        return True
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name)
