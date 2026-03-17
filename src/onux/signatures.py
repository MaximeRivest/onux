from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, get_args, get_origin

from .examples import ExamplesTable, infer_type as _infer_type, normalize_examples as _normalize_examples


@dataclass(frozen=True)
class Field:
    """Describe one named field inside a `Signature`.

    A field is the smallest unit in a signature. It records four things: the
    field name, where that field sits in the task pipeline, its coarse Python
    type, and an optional note for humans.

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

    Examples
    --------
    Create a plain input field.

    >>> field = Field("question", "input")
    >>> field.name
    'question'
    >>> field.role
    'input'
    >>> field.type_ is str
    True

    Add a type and a short note when you want the field to carry more intent.

    >>> score = Field("score", "output", float, "Probability from 0 to 1")
    >>> score.type_ is float
    True
    >>> score.note
    'Probability from 0 to 1'
    """

    name: str
    role: Literal["input", "hidden", "output"]
    type_: type = str
    note: str | None = None


@dataclass(frozen=True)
class _ObjectiveCriterion:
    """Internal normalized objective criterion."""

    kind: Literal["rubric", "metric"]
    spec: str | Callable[..., Any]


@dataclass(frozen=True)
class _Objective:
    """Internal normalized objective metadata."""

    criteria: tuple[_ObjectiveCriterion, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Objective must contain at least one rubric or metric.")

        normalized_weights = tuple(float(weight) for weight in self.weights)
        if len(normalized_weights) != len(self.criteria):
            raise ValueError("'weights' must have the same length as the number of objective items.")
        if any(weight <= 0.0 for weight in normalized_weights):
            raise ValueError("Objective weights must all be positive.")
        object.__setattr__(self, "weights", normalized_weights)


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
    """Declare the input/output contract for one semantic task.

    A signature starts with a small formula made of field names only:

    - `inputs -> outputs`
    - `inputs -> hidden -> outputs`

    From there you can layer in richer intent: types, notes, examples, hidden
    intermediate fields, and numeric objectives (as metrics and rubrics). The 
    object is immutable, so each modifier returns a new signature. This makes 
    it easy to start simple, then add detail only when you need it.

    Examples
    --------
    Start with the smallest useful signature.

    >>> sig = Signature("question -> answer")
    >>> sig.formula
    'question -> answer'
    >>> list(sig.input_fields)
    ['question']
    >>> list(sig.output_fields)
    ['answer']

    Add types and notes when you want the contract to say a bit more.

    >>> typed = (
    ...     Signature("question -> answer, confidence")
    ...     .type(answer=str, confidence=float)
    ...     .note(answer="One short answer", confidence="Number from 0 to 1")
    ... )
    >>> typed.output_fields["confidence"].type_ is float
    True
    >>> typed.output_fields["confidence"].note
    'Number from 0 to 1'

    Add a hidden field when the task has an explicit intermediate step.

    >>> reasoned = Signature("question -> answer").via("reasoning", note="Work shown for inspection")
    >>> reasoned.formula
    'question -> reasoning -> answer'
    >>> list(reasoned.hidden_fields)
    ['reasoning']

    Then attach examples and an objective to make the intended behavior more
    concrete. This is often where a signature starts to feel like a tidy little
    specification rather than just a schema. Metric callables typically accept
    the signature fields they need as named parameters and return a numeric
    score.

    >>> def single_sentence(answer: str) -> float:
    ...     sentence_marks = answer.count(".") + answer.count("!") + answer.count("?")
    ...     return float("\\n" not in answer and sentence_marks <= 1)
    >>> rich = (
    ...     Signature("question -> answer")
    ...     .hint("Answer in one sentence.")
    ...     .examples([
    ...         {"question": "2 + 2", "answer": "4"},
    ...         {"question": "Capital of France", "answer": "Paris"},
    ...     ])
    ...     .objective(
    ...         "Assign a score from 0 to 1 for factual correctness and lack of ambiguity, where 1 means fully correct and clear and 0 means incorrect or misleading.",
    ...         single_sentence,
    ...         (2.0, 1.0),
    ...     )
    ... )
    >>> rich.n_examples
    2
    >>> [criterion.kind for criterion in rich.objective_spec.criteria]
    ['rubric', 'metric']
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
        _objective: _Objective | None = None,
    ):
        """Create a signature from a formula or prebuilt internal state.

        Parameters
        ----------
        formula : str | None, optional
            Formula describing the task structure, for example
            `"question -> answer"` or `"question -> reasoning -> answer"`.
        examples : Any | None, optional
            Canonical examples attached to the signature. These examples may
            also be used to infer coarse field types.
        hint : str | None, optional
            Human-readable task guidance. When omitted, a default hint is
            derived from input and output field names.
        _fields : tuple[Field, ...] | None, optional
            Internal field tuple used when cloning an existing signature.
        _examples : ExamplesTable | None, optional
            Internal normalized examples table used when cloning an existing
            signature.
        _objective : _Objective | None, optional
            Internal normalized objective used when cloning an existing
            signature.

        Examples
        --------
        Construct a signature from a formula.

        >>> sig = Signature("question, context -> answer")
        >>> sig.formula
        'question, context -> answer'

        Example data can infer coarse types for you.

        >>> sig = Signature(
        ...     "question -> answer, confidence",
        ...     examples=[{"question": "2 + 2", "answer": "4", "confidence": 0.99}],
        ... )
        >>> sig.output_fields["answer"].type_ is str
        True
        >>> sig.output_fields["confidence"].type_ is float
        True

        When you do not provide a hint, a plain-English default is created.

        >>> Signature("question -> answer")._hint
        'Given `question`, produce `answer`.'
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
        """Return the normalized signature formula.

        Returns
        -------
        str
            Formula reconstructed from the current field layout.

        Examples
        --------
        >>> Signature("question -> answer").formula
        'question -> answer'

        Hidden fields are included when present.

        >>> Signature("question -> answer").via("reasoning").formula
        'question -> reasoning -> answer'
        """
        groups: list[list[str]] = [[], [], []]
        for field in self._fields:
            groups[_ROLE_ORDER[field.role]].append(field.name)
        return " -> ".join(", ".join(group) for group in groups if group)

    @property
    def example_data(self) -> Any | None:
        """Return the raw example object attached to the signature.

        Returns
        -------
        Any | None
            The original example payload, or `None` when no examples are
            attached.

        Examples
        --------
        >>> rows = [{"question": "2 + 2", "answer": "4"}]
        >>> sig = Signature("question -> answer", examples=rows)
        >>> sig.example_data == rows
        True

        >>> Signature("question -> answer").example_data is None
        True
        """
        return None if self._examples is None else self._examples.raw

    @property
    def n_examples(self) -> int:
        """Return the number of attached examples.

        Returns
        -------
        int
            Number of canonical example rows.

        Examples
        --------
        >>> Signature("question -> answer").n_examples
        0
        >>> Signature("question -> answer", examples=[{"question": "Q", "answer": "A"}]).n_examples
        1
        """
        return 0 if self._examples is None else len(self._examples)

    @property
    def objective_spec(self) -> Any | None:
        """Return the normalized objective attached to the signature.

        Returns
        -------
        Any | None
            Internal normalized objective metadata, or `None` when the
            signature has no objective yet.

        Examples
        --------
        >>> plain = Signature("question -> answer")
        >>> plain.objective_spec is None
        True

        >>> scored = plain.objective(
        ...     "Assign a score from 0 to 1 for factual correctness, where 1 means fully correct and 0 means clearly incorrect."
        ... )
        >>> len(scored.objective_spec.criteria)
        1
        >>> scored.objective_spec.criteria[0].kind
        'rubric'
        >>> scored.objective_spec.weights
        (1.0,)
        """
        return self._objective

    @property
    def fields(self) -> OrderedDict[str, Field]:
        """Return all fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            All fields in formula order.

        Examples
        --------
        >>> sig = Signature("question -> reasoning -> answer")
        >>> list(sig.fields)
        ['question', 'reasoning', 'answer']
        >>> sig.fields['answer'].role
        'output'
        """
        return OrderedDict((field.name, field) for field in self._fields)

    @property
    def input_fields(self) -> OrderedDict[str, Field]:
        """Return input fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Input fields in formula order.

        Examples
        --------
        >>> sig = Signature("question, context -> answer")
        >>> list(sig.input_fields)
        ['question', 'context']
        """
        return OrderedDict((field.name, field) for field in self._fields if field.role == "input")

    @property
    def hidden_fields(self) -> OrderedDict[str, Field]:
        """Return hidden fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Hidden fields in formula order.

        Examples
        --------
        >>> sig = Signature("question -> answer").via("reasoning").via("evidence")
        >>> list(sig.hidden_fields)
        ['reasoning', 'evidence']
        """
        return OrderedDict((field.name, field) for field in self._fields if field.role == "hidden")

    @property
    def output_fields(self) -> OrderedDict[str, Field]:
        """Return output fields keyed by name.

        Returns
        -------
        OrderedDict[str, Field]
            Output fields in formula order.

        Examples
        --------
        >>> sig = Signature("question -> answer").add("confidence", float)
        >>> list(sig.output_fields)
        ['answer', 'confidence']
        >>> sig.output_fields['confidence'].type_ is float
        True
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
            A new signature with the same fields and metadata, except for the
            updated hint.

        Examples
        --------
        >>> sig = Signature("question -> answer")
        >>> updated = sig.hint("Answer in one sentence.")
        >>> updated is sig
        False
        >>> updated._hint
        'Answer in one sentence.'
        >>> sig._hint
        'Given `question`, produce `answer`.'
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

        Returns
        -------
        Signature
            A new signature with updated notes for the named fields.

        Examples
        --------
        Add one note.

        >>> sig = Signature("question -> answer").note(answer="A short factual response")
        >>> sig.output_fields['answer'].note
        'A short factual response'

        Or annotate several fields at once.

        >>> richer = Signature("question, context -> answer").note(
        ...     question="User's request",
        ...     context="Supporting material",
        ...     answer="One grounded answer",
        ... )
        >>> richer.input_fields['context'].note
        'Supporting material'
        >>> richer.output_fields['answer'].note
        'One grounded answer'
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

        Returns
        -------
        Signature
            A new signature with updated field types.

        Examples
        --------
        Set one output type explicitly.

        >>> sig = Signature("review -> sentiment").type(sentiment=float)
        >>> sig.output_fields['sentiment'].type_ is float
        True

        Or update several fields together.

        >>> richer = Signature("question -> answer, confidence").type(answer=str, confidence=float)
        >>> richer.output_fields['answer'].type_ is str
        True
        >>> richer.output_fields['confidence'].type_ is float
        True
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

        Returns
        -------
        Signature
            A new signature with normalized examples attached.

        Notes
        -----
        Examples are part of the signature intent specification. They are often
        best when small and representative rather than large and exhaustive.

        Examples
        --------
        Add a couple of canonical rows.

        >>> sig = Signature("question -> answer").examples([
        ...     {"question": "2 + 2", "answer": "4"},
        ...     {"question": "Capital of France", "answer": "Paris"},
        ... ])
        >>> sig.n_examples
        2
        >>> sig.example_data[0]['question']
        '2 + 2'

        Because examples are normalized, they can also be used to infer types.

        >>> typed = Signature(
        ...     "ticket -> label, priority",
        ...     examples=[{"ticket": "Printer is jammed", "label": "support", "priority": 2}],
        ... )
        >>> typed.output_fields['priority'].type_ is int
        True
        """
        return Signature(
            _fields=self._fields,
            hint=self._hint,
            _examples=_normalize_examples(rows),
            _objective=self._objective,
        )

    def objective(
        self,
        *items: str | Callable[..., Any] | tuple[float, ...],
    ) -> Signature:
        """Return a copy with a numeric objective.

        Each positional item is interpreted as follows:

        - a string means a rubric
        - a callable means a metric
        - a final tuple of floats, when present, gives weights

        When no final weight tuple is provided, all objective items receive
        equal weight.

        Parameters
        ----------
        *items : str | Callable[..., Any] | tuple[float, ...]
            Rubrics, metrics, and optionally one final weight tuple.

        Returns
        -------
        Signature
            A new signature with a normalized objective attached.

        Notes
        -----
        Rubric strings are instructions for a judge. A good rubric does not
        merely name a quality such as "correctness" or "clarity". Instead, it
        tells the judge what number to assign and what high and low scores mean.

        In practice, rubric strings work best when they:

        - specify the score range, usually 0 to 1
        - state what a high score means
        - state what a low score means
        - mention the concrete behavior being judged

        For example, prefer a rubric like:

        >>> rubric = (
        ...     "Assign a score from 0 to 1 for factual correctness, where 1 means fully correct "
        ...     "and 0 means clearly incorrect."
        ... )

        over a label-like string such as ``"correctness"``.

        Weights are relative parts, not normalized probabilities. They do not
        need to sum to 1. For example, ``(4.0, 2.0, 1.0)`` means the first
        objective item counts twice as much as the second and four times as
        much as the third.

        Metric callables should accept named parameters corresponding to the
        signature fields they need. They do not need to accept every field.

        In other words, metric parameters should be chosen by field name, not
        by position. A metric may use only outputs, or a mix of inputs,
        hidden fields, and outputs.

        Common shapes look like:

        >>> def brief(answer: str) -> float:
        ...     return float(len(answer.split()) <= 20)
        >>> def grounded(question: str, context: str, answer: str) -> float:
        ...     return 1.0

        Metrics should return a numeric score, ideally a float from 0 to 1,
        where higher is better. Returning `bool` is also fine when the metric
        is naturally pass/fail.

        Examples
        --------
        Start with a single rubric.

        >>> sig = Signature("question -> answer").objective(
        ...     "Assign a score from 0 to 1 for answers that are both factually correct and concise, where higher scores indicate better performance on both dimensions."
        ... )
        >>> len(sig.objective_spec.criteria)
        1
        >>> sig.objective_spec.criteria[0].kind
        'rubric'
        >>> sig.objective_spec.weights
        (1.0,)

        Add several rubrics and provide a final weight tuple.

        >>> weighted = Signature("question -> answer").objective(
        ...     "Assign a score from 0 to 1 for factual correctness, where 1 means fully correct and 0 means clearly incorrect.",
        ...     "Assign a score from 0 to 1 for clarity and structure, where 1 means easy to read and well organized and 0 means confusing or poorly structured.",
        ...     (3.0, 1.0),
        ... )
        >>> weighted.objective_spec.weights
        (3.0, 1.0)

        Mix rubrics with Python metrics.

        >>> def single_sentence(answer: str) -> float:
        ...     sentence_marks = answer.count(".") + answer.count("!") + answer.count("?")
        ...     return float("\\n" not in answer and sentence_marks <= 1)
        >>> mixed = Signature("question -> answer").objective(
        ...     "Assign a score from 0 to 1 for how directly the answer addresses the user's question, where 1 means fully direct and 0 means off-topic or evasive.",
        ...     single_sentence,
        ...     (2.0, 1.0),
        ... )
        >>> [criterion.kind for criterion in mixed.objective_spec.criteria]
        ['rubric', 'metric']
        >>> mixed.objective_spec.criteria[1].spec is single_sentence
        True

        You can also build a more realistic five-axis objective that mixes two
        judge-written rubrics with three programmatic metrics.

        >>> def cheap(price_usd: float) -> float:
        ...     return max(0.0, min(1.0, 1.0 - price_usd / 10.0))
        >>> def low_latency(latency_ms: float) -> float:
        ...     return max(0.0, min(1.0, 1.0 - latency_ms / 2000.0))
        >>> def fast_total_time(total_time_s: float) -> float:
        ...     return max(0.0, min(1.0, 1.0 - total_time_s / 30.0))
        >>> compound = Signature(
        ...     "question -> answer, price_usd, latency_ms, total_time_s"
        ... ).objective(
        ...     "Assign a score from 0 to 1 for answer accuracy, where 1 means fully correct and 0 means clearly incorrect.",
        ...     "Assign a score from 0 to 1 for overall usefulness, where 1 means the answer is helpful, well explained, and appropriate for the user, and 0 means it is unhelpful or poorly framed.",
        ...     cheap,
        ...     low_latency,
        ...     fast_total_time,
        ...     (4.0, 2.0, 1.0, 1.0, 1.0),
        ... )
        >>> [criterion.kind for criterion in compound.objective_spec.criteria]
        ['rubric', 'rubric', 'metric', 'metric', 'metric']
        >>> compound.objective_spec.weights
        (4.0, 2.0, 1.0, 1.0, 1.0)
        """
        objective = _normalize_objective_items(items)
        return Signature(
            _fields=self._fields,
            hint=self._hint,
            _examples=self._examples,
            _objective=objective,
        )
    def via(
        self,
        name: str,
        type_: type = str,
        *,
        note: str | None = None,
    ) -> Signature:
        """Return a copy with an additional hidden field.

        Hidden fields sit between inputs and outputs. They are useful when you
        want the task contract to expose an intermediate artifact, such as
        reasoning, retrieved evidence, or a draft.

        Parameters
        ----------
        name : str
            Hidden field name.
        type_ : type, default=str
            Hidden field type.
        note : str | None, optional
            Optional note describing the hidden field.

        Returns
        -------
        Signature
            A new signature with the hidden field inserted before outputs.

        Examples
        --------
        Add one hidden field.

        >>> sig = Signature("question -> answer").via("reasoning")
        >>> sig.formula
        'question -> reasoning -> answer'
        >>> sig.hidden_fields['reasoning'].role
        'hidden'

        Hidden fields can also carry types and notes.

        >>> inspected = Signature("question -> answer").via(
        ...     "reasoning",
        ...     str,
        ...     note="Work shown for auditing",
        ... )
        >>> inspected.hidden_fields['reasoning'].note
        'Work shown for auditing'

        You can keep layering hidden fields when the task has several stages.

        >>> staged = (
        ...     Signature("question -> answer")
        ...     .via("retrieved_context", list, note="Evidence snippets")
        ...     .via("draft_answer", str, note="Initial answer before revision")
        ... )
        >>> list(staged.hidden_fields)
        ['retrieved_context', 'draft_answer']
        >>> staged.hidden_fields['retrieved_context'].type_ is list
        True
        """
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
        """Return a copy with an additional output field.

        Parameters
        ----------
        name : str
            Output field name.
        type_ : type, default=str
            Output field type.
        note : str | None, optional
            Optional note describing the new field.

        Returns
        -------
        Signature
            A new signature with the output appended to the end.

        Examples
        --------
        >>> sig = Signature("question -> answer").add("confidence", float, note="Number from 0 to 1")
        >>> sig.formula
        'question -> answer, confidence'
        >>> sig.output_fields['confidence'].type_ is float
        True
        >>> sig.output_fields['confidence'].note
        'Number from 0 to 1'
        """
        return Signature(
            _fields=tuple([*self._fields, Field(name, "output", type_, note)]),
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
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
            A new signature with the named field removed.

        Examples
        --------
        Remove an output field you no longer need.

        >>> sig = Signature("question -> answer, confidence").remove("confidence")
        >>> sig.formula
        'question -> answer'

        The same works for hidden fields.

        >>> staged = Signature("question -> answer").via("reasoning")
        >>> stripped = staged.remove("reasoning")
        >>> stripped.formula
        'question -> answer'
        """
        return Signature(
            _fields=tuple(field for field in self._fields if field.name != name),
            hint=self._hint,
            _examples=self._examples,
            _objective=self._objective,
        )

    def dump_state(self) -> dict[str, Any]:
        """Serialize prompt-facing state.

        Returns
        -------
        dict[str, Any]
            Plain Python data describing the signature.

        Notes
        -----
        Callable metrics are serialized by name only. Their executable Python
        bodies are not reconstructed by `load_state`.

        Examples
        --------
        Serialize a small signature.

        >>> state = (
        ...     Signature("question -> answer")
        ...     .hint("Answer briefly.")
        ...     .examples([{"question": "2 + 2", "answer": "4"}])
        ...     .dump_state()
        ... )
        >>> state['formula']
        'question -> answer'
        >>> state['hint']
        'Answer briefly.'
        >>> state['examples'][0]['answer']
        '4'
        >>> state['objective']['criteria'][0]['kind']
        'rubric'

        This is especially handy for round-tripping through JSON-like storage.

        >>> restored = Signature.load_state(state)
        >>> restored == Signature.load_state(state)
        True
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
                "weights": list(self._objective.weights),
                "criteria": [
                    {
                        "kind": criterion.kind,
                        "spec": criterion.spec if criterion.kind == "rubric" else _callable_name(criterion.spec),
                    }
                    for criterion in self._objective.criteria
                ],
            },
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Signature:
        """Restore a signature from serialized state.

        Parameters
        ----------
        state : dict[str, Any]
            State previously produced by `dump_state`.

        Returns
        -------
        Signature
            Reconstructed signature.

        Notes
        -----
        Callable metrics are restored by their serialized names, not by
        executable Python function objects.

        Examples
        --------
        Round-trip a signature through serialized state.

        >>> original = (
        ...     Signature("question -> answer, confidence")
        ...     .hint("Answer and include confidence.")
        ...     .type(confidence=float)
        ...     .examples([{"question": "2 + 2", "answer": "4", "confidence": 0.99}])
        ... )
        >>> restored = Signature.load_state(original.dump_state())
        >>> restored.formula
        'question -> answer, confidence'
        >>> restored.output_fields['confidence'].type_ is float
        True
        >>> restored.n_examples
        1
        >>> restored.objective_spec.criteria[0].kind
        'rubric'
        """
        fields = [
            Field(item["name"], item["role"], _BUILTIN_TYPES.get(item["type"], str), item.get("note"))
            for item in state["fields"]
        ]
        examples = _normalize_examples(state["examples"]) if state.get("examples") is not None else None

        objective_state = state.get("objective")
        objective = None
        if objective_state is not None:
            criterion_states = objective_state.get("criteria")
            if criterion_states is None:
                criterion_states = objective_state.get("terms", [])

            weights_state = objective_state.get("weights")
            if weights_state is None:
                weights_state = [criterion.get("weight", 1.0) for criterion in criterion_states]

            objective = _Objective(
                criteria=tuple(
                    _ObjectiveCriterion(
                        "metric" if criterion["kind"] == "callable" else criterion["kind"],
                        criterion["spec"],
                    )
                    for criterion in criterion_states
                ),
                weights=_coerce_weights(weights_state),
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
            for criterion, weight in zip(self._objective.criteria, self._objective.weights):
                if criterion.kind == "rubric":
                    label = shorten(str(criterion.spec), max_objective_chars)
                else:
                    label = shorten(_metric_name(criterion.spec), max_objective_chars)
                if len(self._objective.criteria) > 1 or weight != 1.0:
                    label = f"{label} [weight={weight:g}]"
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


__all__ = ["Field", "Signature"]


def _callable_name(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    if qualname:
        return qualname
    return str(obj)



def _metric_name(obj: Any) -> str:
    name = getattr(obj, "__name__", None)
    if name:
        return name
    qualname = getattr(obj, "__qualname__", None)
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



def _equal_weights(count: int) -> tuple[float, ...]:
    return tuple(1.0 for _ in range(count))



def _coerce_weights(weights: Any) -> tuple[float, ...]:
    return tuple(float(weight) for weight in weights)



def _normalize_objective_items(
    items: tuple[str | Callable[..., Any] | tuple[float, ...], ...],
) -> _Objective:
    if not items:
        raise ValueError("Provide at least one rubric or metric.")

    weights: tuple[float, ...] | None = None
    criteria_items = items
    if isinstance(items[-1], tuple):
        weights = tuple(float(weight) for weight in items[-1])
        criteria_items = items[:-1]

    if not criteria_items:
        raise ValueError("Provide at least one rubric or metric before the final weight tuple.")

    criteria: list[_ObjectiveCriterion] = []
    for item in criteria_items:
        if isinstance(item, tuple):
            raise TypeError("Only the final positional item may be a weight tuple.")
        if isinstance(item, str):
            criteria.append(_ObjectiveCriterion("rubric", item))
        elif callable(item):
            criteria.append(_ObjectiveCriterion("metric", item))
        else:
            raise TypeError("Objective items must be rubric strings, metric callables, and optionally one final weight tuple.")

    if weights is None:
        weights = _equal_weights(len(criteria))
    elif len(weights) != len(criteria):
        raise ValueError("The final weight tuple must have the same length as the number of rubrics and metrics.")

    return _Objective(tuple(criteria), weights=weights)



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
