from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, get_args, get_origin

from .examples import ExamplesTable, infer_type as _infer_type, normalize_examples as _normalize_examples


@dataclass(frozen=True)
class Field:
    """One field in a signature.

    Parameters
    ----------
    name : str
        Field name.
    role : {"input", "hidden", "output"}
        Field role within the signature.
    type_ : type, default=str
        Python type associated with the field.
    note : str | None, default=None
        Optional human-readable field note.
    """

    name: str
    role: Literal["input", "hidden", "output"]
    type_: type = str
    note: str | None = None


@dataclass(frozen=True)
class ObjectiveTerm:
    """One numeric scoring term in a signature objective.

    A term is either:

    - a rubric string, scored numerically by an external judge
    - a Python callable that returns a numeric score (or score-like object)

    Parameters
    ----------
    kind : {"rubric", "callable"}
        Objective term kind.
    spec : str | Callable[..., Any]
        Rubric text or scoring callable.
    weight : float, default=1.0
        Relative weight for this term when aggregating multiple terms.
    name : str | None, default=None
        Optional display name.
    """

    kind: Literal["rubric", "callable"]
    spec: str | Callable[..., Any]
    weight: float = 1.0
    name: str | None = None

    @classmethod
    def rubric(cls, text: str, *, weight: float = 1.0, name: str | None = None) -> ObjectiveTerm:
        """Construct a rubric-based objective term."""
        return cls("rubric", text, weight, name)

    @classmethod
    def scorer(cls, fn: Callable[..., Any], *, weight: float = 1.0, name: str | None = None) -> ObjectiveTerm:
        """Construct a callable objective term."""
        return cls("callable", fn, weight, name)


@dataclass(frozen=True)
class Objective:
    """A numeric objective attached to a signature.

    Parameters
    ----------
    terms : tuple[ObjectiveTerm, ...]
        Objective terms to aggregate.
    reduce : {"weighted_mean"}, default="weighted_mean"
        Aggregation rule for combining multiple term scores.
    """

    terms: tuple[ObjectiveTerm, ...]
    reduce: Literal["weighted_mean"] = "weighted_mean"


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

    A signature formula contains field names only:

    - ``inputs -> outputs``
    - ``inputs -> hidden -> outputs``

    Add types, notes, examples, and objectives after construction with the
    fluent methods on this class.

    Examples
    --------
    >>> sig = Signature("question -> answer")
    >>> sig.formula
    'question -> answer'

    >>> typed = Signature("question -> answer").type(answer=float)
    >>> typed.output_fields["answer"].type_ is float
    True

    >>> rich = (
    ...     Signature("question -> answer")
    ...     .hint("Answer briefly.")
    ...     .note(answer="Short factual answer")
    ...     .examples([{"question": "2+2?", "answer": "4"}])
    ...     .objective("Correct and concise.")
    ... )
    >>> rich.n_examples
    1
    >>> rich.objective_spec is not None
    True
    """

    __slots__ = ("_fields", "_hint", "_examples", "_objective")

    def __init__(
        self,
        formula: str | None = None,
        *,
        examples: Any | None = None,
        hint: str | None = None,
        _fields: tuple[Field, ...] | None = None,
        _examples: ExamplesTable | None = None,
        _objective: Objective | None = None,
    ):
        """Create a signature from a formula or prebuilt internal state.

        Parameters
        ----------
        formula : str | None, optional
            Formula describing the task structure, for example
            ``"question -> answer"`` or ``"question -> reasoning -> answer"``.
        examples : Any | None, optional
            A few canonical examples attached to the signature. These examples
            may also be used to infer coarse field types.
        hint : str | None, optional
            Human-readable task guidance. When omitted, a default hint is
            derived from input and output field names.
        _fields : tuple[Field, ...] | None, optional
            Internal field tuple used when cloning an existing signature.
        _examples : ExamplesTable | None, optional
            Internal normalized examples table used when cloning an existing
            signature.
        _objective : Objective | None, optional
            Internal normalized objective used when cloning an existing
            signature.
        """
        if _fields is not None:
            object.__setattr__(self, "_fields", _fields)
            object.__setattr__(self, "_hint", hint or "")
            object.__setattr__(self, "_examples", _examples)
            object.__setattr__(self, "_objective", _objective)
            return

        if formula is None:
            raise ValueError("Provide a formula like 'question -> answer'.")

        normalized_examples = _normalize_examples(examples)
        fields = _parse_formula(formula, normalized_examples)

        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_examples", normalized_examples)
        object.__setattr__(self, "_objective", None)

        if hint is None:
            ins = ", ".join(f"`{f.name}`" for f in fields if f.role == "input")
            outs = ", ".join(f"`{f.name}`" for f in fields if f.role == "output")
            hint = f"Given {ins}, produce {outs}."
        object.__setattr__(self, "_hint", hint)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Signature is immutable. Use .hint(), .note(), .type(), .examples(), .objective(), etc.")

    @property
    def formula(self) -> str:
        """Return the signature formula."""
        groups: list[list[str]] = [[], [], []]
        for field in self._fields:
            groups[_ROLE_ORDER[field.role]].append(field.name)
        return " -> ".join(", ".join(group) for group in groups if group)

    @property
    def example_data(self) -> Any | None:
        """Return the raw examples attached to the signature, if any."""
        return None if self._examples is None else self._examples.raw

    @property
    def n_examples(self) -> int:
        """Return the number of attached examples."""
        return 0 if self._examples is None else len(self._examples)

    @property
    def objective_spec(self) -> Objective | None:
        """Return the normalized objective attached to the signature."""
        return self._objective

    @property
    def fields(self) -> OrderedDict[str, Field]:
        """Return all fields keyed by name."""
        return OrderedDict((field.name, field) for field in self._fields)

    @property
    def input_fields(self) -> OrderedDict[str, Field]:
        """Return input fields keyed by name."""
        return OrderedDict((field.name, field) for field in self._fields if field.role == "input")

    @property
    def hidden_fields(self) -> OrderedDict[str, Field]:
        """Return hidden fields keyed by name."""
        return OrderedDict((field.name, field) for field in self._fields if field.role == "hidden")

    @property
    def output_fields(self) -> OrderedDict[str, Field]:
        """Return output fields keyed by name."""
        return OrderedDict((field.name, field) for field in self._fields if field.role == "output")

    def hint(self, text: str) -> Signature:
        """Return a copy with updated task guidance.

        Parameters
        ----------
        text : str
            New hint text.
        """
        return Signature(
            _fields=self._fields,
            hint=text,
            _examples=self._examples,
            _objective=self._objective,
        )

    def note(self, **notes: str) -> Signature:
        """Return a copy with updated field notes.

        Parameters
        ----------
        **notes : str
            Keyword arguments mapping field names to note text.
        """
        new_fields = tuple(
            Field(field.name, field.role, field.type_, notes.get(field.name, field.note))
            for field in self._fields
        )
        return Signature(
            _fields=new_fields,
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
        )

    def type(self, **types: type) -> Signature:
        """Return a copy with updated field types.

        Parameters
        ----------
        **types : type
            Keyword arguments mapping field names to replacement types.
        """
        new_fields = tuple(
            Field(field.name, field.role, types.get(field.name, field.type_), field.note)
            for field in self._fields
        )
        return Signature(
            _fields=new_fields,
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
        )

    def examples(self, rows: Any) -> Signature:
        """Return a copy with attached canonical examples.

        Parameters
        ----------
        rows : Any
            Dataframe-like tabular examples or an iterable of record mappings.

        Notes
        -----
        Examples are part of the signature intent specification. They may be a
        few canonical rows rather than a full training dataset.
        """
        return Signature(
            _fields=self._fields,
            hint=self._hint,
            _examples=_normalize_examples(rows),
            _objective=self._objective,
        )

    def objective(
        self,
        *terms: str | Callable[..., Any] | ObjectiveTerm | Objective,
        weights: tuple[float, ...] | None = None,
        reduce: Literal["weighted_mean"] = "weighted_mean",
    ) -> Signature:
        """Return a copy with a numeric objective.

        Each term is either:

        - a rubric string, to be judged numerically by an external judge
        - a Python scoring callable
        - an :class:`ObjectiveTerm`
        - a fully constructed :class:`Objective`

        Parameters
        ----------
        *terms : str | Callable[..., Any] | ObjectiveTerm | Objective
            Objective terms to attach.
        weights : tuple[float, ...] | None, optional
            Optional weights aligned with ``terms``.
        reduce : {"weighted_mean"}, default="weighted_mean"
            Aggregation rule used when combining term scores.
        """
        objective = _normalize_objective_terms(terms, weights=weights, reduce=reduce)
        return Signature(
            _fields=self._fields,
            hint=self._hint,
            _examples=self._examples,
            _objective=objective,
        )

    def rubric(self, text: str, *, weight: float = 1.0, name: str | None = None) -> Signature:
        """Return a copy with a single rubric objective term.

        Parameters
        ----------
        text : str
            Rubric text to judge numerically.
        weight : float, default=1.0
            Weight for the rubric term.
        name : str | None, optional
            Optional display name.
        """
        return self.objective(ObjectiveTerm.rubric(text, weight=weight, name=name))

    def via(
        self,
        name: str,
        type_: type = str,
        *,
        note: str | None = None,
    ) -> Signature:
        """Return a copy with an additional hidden field."""
        fields = list(self._fields)
        insert_at = len(fields)
        for i, field in enumerate(fields):
            if field.role == "output":
                insert_at = i
                break
        fields.insert(insert_at, Field(name, "hidden", type_, note))
        return Signature(
            _fields=tuple(fields),
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
        )

    def add(
        self,
        name: str,
        type_: type = str,
        *,
        note: str | None = None,
    ) -> Signature:
        """Return a copy with an additional output field."""
        return Signature(
            _fields=tuple([*self._fields, Field(name, "output", type_, note)]),
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
        )

    def remove(self, name: str) -> Signature:
        """Return a copy without one field."""
        return Signature(
            _fields=tuple(field for field in self._fields if field.name != name),
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
        )

    def dump_state(self) -> dict[str, Any]:
        """Serialize prompt-facing state.

        Notes
        -----
        Callable objective terms are serialized by name only. Their executable
        Python bodies are not reconstructed by :meth:`load_state`.
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
            "objective": None if self._objective is None else {
                "reduce": self._objective.reduce,
                "terms": [
                    {
                        "kind": term.kind,
                        "spec": term.spec if term.kind == "rubric" else _callable_name(term.spec),
                        "weight": term.weight,
                        "name": term.name,
                    }
                    for term in self._objective.terms
                ],
            },
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Signature:
        """Restore a signature from serialized state."""
        fields = [
            Field(item["name"], item["role"], _BUILTIN_TYPES.get(item["type"], str), item.get("note"))
            for item in state["fields"]
        ]
        examples = _normalize_examples(state["examples"]) if state.get("examples") is not None else None

        objective_state = state.get("objective")
        objective = None
        if objective_state is not None:
            objective = Objective(
                terms=tuple(
                    ObjectiveTerm(term["kind"], term["spec"], term.get("weight", 1.0), term.get("name"))
                    for term in objective_state.get("terms", [])
                ),
                reduce=objective_state.get("reduce", "weighted_mean"),
            )

        return cls(_fields=tuple(fields), hint=state["hint"], _examples=examples, _objective=objective)

    def __repr__(self) -> str:
        max_literal_items = 6
        max_literal_chars = 24
        max_type_chars = 96
        max_field_chars = 120
        max_objective_chars = 100

        def shorten(text: str, limit: int) -> str:
            return text if len(text) <= limit else text[: limit - 1] + "…"

        def format_type(tp: Any) -> str:
            origin = get_origin(tp)

            if origin is Literal:
                values = []
                args = get_args(tp)
                for value in args[:max_literal_items]:
                    values.append(shorten(repr(value), max_literal_chars))
                if len(args) > max_literal_items:
                    values.append(f"… (+{len(args) - max_literal_items} more)")
                return shorten("one of: " + ", ".join(values), max_type_chars)

            if origin is not None:
                args = get_args(tp)
                name = getattr(origin, "__name__", str(origin))
                if args:
                    rendered = f"{name}[{', '.join(format_type(arg) for arg in args)}]"
                    return shorten(rendered, max_type_chars)
                return shorten(name, max_type_chars)

            if isinstance(tp, type):
                return shorten(tp.__name__, max_type_chars)
            return shorten(str(tp), max_type_chars)

        width = max((len(field.name) for field in self._fields), default=0)
        lines = [self.formula, f"  {self._hint!r}"]

        if self._objective is not None:
            lines.append("  objective:")
            for term in self._objective.terms:
                if term.kind == "rubric":
                    label = shorten(term.name or str(term.spec), max_objective_chars)
                else:
                    name = term.name or _callable_name(term.spec)
                    label = shorten(name, max_objective_chars)
                if len(self._objective.terms) > 1 or term.weight != 1.0:
                    label = f"{label} [weight={term.weight:g}]"
                lines.append(f"    - {label}")

        for field in self._fields:
            marker = {"input": "→", "hidden": "·", "output": "←"}[field.role]
            text = format_type(field.type_)
            if field.note:
                text = shorten(f"{field.note}: {text}", max_field_chars)
            lines.append(f"  {marker} {field.name:<{width}}  {text}")

        if self._examples is not None:
            lines.append(f"  ({len(self._examples)} examples)")
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Signature):
            return (
                self._fields == other._fields
                and self._hint == other._hint
                and self._objective == other._objective
                and _freeze_examples(self._examples) == _freeze_examples(other._examples)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._fields, self._hint, self._objective, _freeze_examples(self._examples)))


__all__ = ["Field", "ObjectiveTerm", "Objective", "Signature"]


def _callable_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return qualname
    return str(obj)



def _type_name(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is not None:
        return getattr(origin, "__name__", str(origin))
    if isinstance(tp, type):
        return tp.__name__
    return str(tp)



def _normalize_objective_terms(
    terms: tuple[str | Callable[..., Any] | ObjectiveTerm | Objective, ...],
    *,
    weights: tuple[float, ...] | None = None,
    reduce: Literal["weighted_mean"] = "weighted_mean",
) -> Objective:
    if len(terms) == 1 and isinstance(terms[0], Objective):
        if weights is not None:
            raise ValueError("Do not pass 'weights' when providing a prebuilt Objective.")
        return terms[0]

    if not terms:
        raise ValueError("Provide at least one objective term.")

    if weights is not None and len(weights) != len(terms):
        raise ValueError("'weights' must have the same length as the number of objective terms.")

    normalized: list[ObjectiveTerm] = []
    for i, term in enumerate(terms):
        weight = 1.0 if weights is None else float(weights[i])
        if isinstance(term, ObjectiveTerm):
            if weights is not None and term.weight != 1.0:
                normalized.append(ObjectiveTerm(term.kind, term.spec, weight, term.name))
            elif weights is not None:
                normalized.append(ObjectiveTerm(term.kind, term.spec, weight, term.name))
            else:
                normalized.append(term)
        elif isinstance(term, str):
            normalized.append(ObjectiveTerm.rubric(term, weight=weight))
        elif callable(term):
            normalized.append(ObjectiveTerm.scorer(term, weight=weight))
        else:
            raise TypeError(
                "Objective terms must be rubric strings, scoring callables, ObjectiveTerm objects, or a prebuilt Objective."
            )

    return Objective(tuple(normalized), reduce=reduce)



def _freeze_examples(examples: ExamplesTable | None) -> Any:
    if examples is None:
        return None
    return tuple(_freeze_value(record) for record in examples.to_records())



def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _freeze_value(val)) for key, val in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)



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
