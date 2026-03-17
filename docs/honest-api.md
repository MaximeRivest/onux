# Honest API: intent, program, objective, evaluator

This note proposes a cleaner and more honest mental model for Onux.

The short version is:

- we are **not** only building an intent API
- we are also building a **program / protocol API**
- those two concerns should be named separately

The biggest clue is hidden fields. They are useful, but they are already a
choice about decomposition, supervision, and execution strategy. That means
hidden fields are not pure intent.

## The problem we ran into

We started with a simple story:

- a signature says what goes in and what comes out
- examples, rubrics, and metrics express intended behavior

But then hidden fields appeared:

- `question -> reasoning -> answer`
- `question -> evidence -> answer`
- `question -> plan -> answer`

These are not just descriptions of the same task. They are already claims
about **how the task should be carried out**.

That is useful, but it is no longer only "intent".

Then evaluation and runtime concerns started leaking in too:

- metric execution
- rubric judging
- runtime telemetry like cost and latency
- judge LMs and adapters
- evaluation aggregation

At that point the old story becomes muddy.

## The honest mental model

Onux should distinguish four layers:

1. **Intent** — the behavioral contract we care about
2. **Program** — one declared protocol / decomposition for implementing that intent
3. **Objective** — how we score behavior
4. **Evaluator** — how scoring is actually executed

This gives us a much cleaner way to explain the system.

---

## 1. Intent

`Intent` is the true high-level behavioral target.

It should contain the things that ought to remain stable across many possible
implementations:

- inputs
- outputs
- types and notes
- examples
- metrics
- rubrics
- external artifact being imitated or replaced

A key property of `Intent` is that it should survive changes in execution
strategy.

This is also where `Artifact` belongs. Here, an artifact is not an internal
program artifact like `reasoning` or `draft_answer`. It is a **reference
artifact**: an external system, binary, service, workflow, or model whose
behavior helps define the target we care about. That makes it part of intent,
not part of execution strategy. A reference artifact can anchor examples,
serve as an example generator, and provide a concrete behavioral baseline for
training, evaluation, migration, or replacement.

These should all be different ways to satisfy the **same** intent:

- answer directly
- retrieve then answer
- reason then answer
- plan then execute
- use tools in a loop

### Proposed API

```python
from typing import Literal
from onux import Artifact, Intent


Severity = Literal["low", "medium", "high", "critical"]
Team = Literal["billing", "bug", "feature", "account"]


def valid_triage(severity: Severity, team: Team, customer_visible: bool) -> float:
    return float(
        severity in {"low", "medium", "high", "critical"}
        and team in {"billing", "bug", "feature", "account"}
        and isinstance(customer_visible, bool)
    )


intent = (
    Intent("support_request -> severity, team, customer_visible")
    .type(
        severity=Severity,
        team=Team,
        customer_visible=bool,
    )
    .note(
        severity="Operational urgency of the request.",
        team="Owning team that should handle the request.",
        customer_visible="Whether the issue is visible to the customer right now.",
    )
    .examples([
        {
            "support_request": "I was charged twice for my subscription this month.",
            "severity": "high",
            "team": "billing",
            "customer_visible": True,
        },
        {
            "support_request": "The export button is misaligned in dark mode.",
            "severity": "low",
            "team": "bug",
            "customer_visible": True,
        },
    ])
    .artifact(Artifact.service("legacy-support-triage"))
    .objective(
        "Assign a score from 0 to 1 for routing accuracy, where 1 means the request is assigned to the correct team and severity and 0 means the routing is clearly wrong.",
        "Assign a score from 0 to 1 for operational usefulness, where 1 means the triage output would be immediately useful to a support organization and 0 means it would cause confusion or delays.",
        valid_triage,
        (4.0, 2.0, 1.0),
    )
)
```

The important point is that this says **what good behavior looks like**, not
how the system internally gets there. It includes stable behavioral structure
(inputs, outputs, types, notes), reference examples, scoring criteria
(rubrics and metrics), and even an external artifact whose behavior we may be
trying to imitate or replace.

This example also shows why typed outputs are often genuinely useful: they
make parsing and downstream use part of the behavioral contract. Here the
program is not just producing prose; it must emit structured values that other
systems can reliably consume. Typed inputs are often less interesting for raw user text.

---

## 2. Program

`Program` is the protocol-level object.

It is where we admit that the user may want to declare intermediate artifacts,
inspection points, decomposition boundaries, and execution-facing structure.

This is where hidden fields really belong.

### Program is not pure intent

If a user writes:

```python
question -> reasoning -> answer
```

that is not just a behavioral claim. It is also a program-structure claim.
So it belongs in `Program`, not in the pure intent object.

### Proposed API

```python
from onux import Program

program = (
    Program(intent)
    .artifact("retrieved_context", list[str], note="Retrieved passages")
    .artifact("draft_answer", str, note="Initial answer before revision")
    .artifact("reasoning", str, note="Inspectable reasoning trace")
)
```

Here `artifact(...)` means an intermediate declared artifact of the program.

These artifacts are useful because if we remove them entirely, users will just
hack them into outputs. So the feature is legitimate; it just belongs to the
program layer, not the pure intent layer.

### What counts as a program artifact?

Good examples:

- `reasoning`
- `evidence`
- `plan`
- `retrieved_context`
- `draft_answer`
- `chosen_tool`

These are all intermediate artifacts of one chosen strategy.

### What does **not** count as a program artifact?

Things like:

- `latency_ms`
- `cost_usd`
- `prompt_tokens`
- `completion_tokens`
- `retry_count`

Those are **runtime telemetry**, not declared task or program artifacts.

---

## 3. Objective

`Objective` is the scoring specification.

It is not a signature, and it is not an evaluator.
It only says **how success should be judged**.

We can keep the lightweight API shape we converged on:

- `str` = rubric
- `callable` = metric
- final `tuple` = relative weights

### Proposed API

```python
from onux import Objective


def single_sentence(answer: str) -> float:
    sentence_marks = answer.count(".") + answer.count("!") + answer.count("?")
    return float("\n" not in answer and sentence_marks <= 1)


objective = Objective(
    "Assign a score from 0 to 1 for factual correctness, where 1 means fully correct and 0 means clearly incorrect.",
    "Assign a score from 0 to 1 for directness, where 1 means the answer directly addresses the user's question and 0 means it is evasive or off-topic.",
    single_sentence,
    (4.0, 2.0, 1.0),
)
```

### Rubrics

A rubric is not just a label like `"correctness"`.
A rubric should tell the judge:

- what score range to use
- what high scores mean
- what low scores mean
- what concrete behavior is being judged

So rubric strings should read like scoring instructions.

### Metrics

Metrics are executable Python functions.

They bind by **parameter name**, not by position.
They may ask for:

- any named intent or program field they need
- the reserved parameter `runtime` for execution telemetry

Examples:

```python
def brief(answer: str) -> float:
    return float(len(answer.split()) <= 20)


def grounded(question: str, retrieved_context: list[str], answer: str) -> float:
    return 1.0


def cheap(runtime) -> float:
    return max(0.0, min(1.0, 1.0 - runtime.cost_usd / 0.05))
```

### Weights

Weights are relative parts, not normalized probabilities.

So all of these mean the same relative tradeoff:

- `(2.0, 1.0)`
- `(20.0, 10.0)`
- `(0.2, 0.1)`

The user should think of weights as:

> how much each criterion matters relative to the others

---

## 4. Evaluator

`Evaluator` is where scoring actually happens.

This is where runtime, judges, adapters, rendering, and aggregation belong.
That keeps them out of `Intent` and `Program`.

### Proposed API

```python
from onux import Evaluator


evaluator = Evaluator(meta_lm=judge_lm)

result = evaluator.evaluate(
    intent=intent,
    program=program,
    objective=objective,
    values={
        "question": "Capital of France",
        "answer": "Paris",
        "retrieved_context": ["Paris is the capital of France."],
        "draft_answer": "Paris.",
        "reasoning": "The capital city of France is Paris.",
    },
    runtime=runtime,
)
```

The evaluator should:

- execute metrics directly
- evaluate rubrics through a judge
- expose runtime telemetry through `runtime`
- aggregate all criterion scores with the objective weights

---

## Judge: the recursive part

Rubrics are executable only through a judge.

Conceptually, a judge is itself another semantic program. So yes, rubric
judging is recursive.

But that recursion should stay behind a very small public interface.

### Public judge protocol

```python
judge(rubric: str, values: dict, *, runtime=None, intent=None, program=None) -> float
```

Internally, that judge may itself be implemented with:

- an LM
- an adapter
- prompt rendering
- another signature/program
- another evaluator

That is fine. The key is that users should not need to think about that
recursion unless they are customizing the judge.

### Default judge

A default evaluator can simply require a meta/judge LM:

```python
evaluator = Evaluator(meta_lm=judge_lm)
```

Then rubric strings are automatically turned into judge calls.

Users can also provide their own judge implementation if they want custom
behavior.

---

## Artifact: imitating or replacing a black box

There is one more important piece of true intent that is not captured by input,
output, examples, rubrics, and metrics alone.

Sometimes the real goal is:

> imitate, replace, or stand in for an existing artifact

Examples:

- a binary
- a legacy service
- a human workflow
- a prompt pipeline
- an old model

This deserves a first-class place in the intent layer.

An `Artifact` is not an intermediate program artifact like `reasoning` or
`draft_answer`. It is a **reference artifact**: an external thing whose
behavior helps define the target we are trying to match, replace, or exceed.
That makes it part of behavioral intent, not part of execution strategy. In
practice, an artifact can guide dataset construction, evaluation, migration,
and system acceptance even when it is never called at runtime.

### Proposed API

```python
from onux import Artifact

intent = (
    Intent("stdin -> stdout")
    .artifact(Artifact.binary("legacy_qa", path="/usr/local/bin/qa_tool"))
)
```

Or more abstractly:

```python
intent = (
    Intent("request -> response")
    .artifact(Artifact.service("legacy-payments-api"))
)
```

This says that the intended behavior is anchored to some external reference
artifact, not only to a few examples.

That is a real kind of intent and should be modeled honestly.

---

## Putting it all together

A realistic end-to-end example:

```python
from onux import Intent, Program, Objective, Evaluator


def single_sentence(answer: str) -> float:
    sentence_marks = answer.count(".") + answer.count("!") + answer.count("?")
    return float("\n" not in answer and sentence_marks <= 1)


def cheap(runtime) -> float:
    return max(0.0, min(1.0, 1.0 - runtime.cost_usd / 0.05))


intent = (
    Intent("question -> answer")
    .hint("Answer in one sentence.")
    .examples([
        {"question": "2 + 2", "answer": "4"},
        {"question": "Capital of France", "answer": "Paris"},
    ])
)

program = (
    Program(intent)
    .artifact("retrieved_context", list[str], note="Retrieved passages")
    .artifact("draft_answer", str, note="Initial answer before revision")
)

objective = Objective(
    "Assign a score from 0 to 1 for factual correctness, where 1 means fully correct and 0 means clearly incorrect.",
    "Assign a score from 0 to 1 for overall usefulness, where 1 means the answer is helpful and appropriately framed for the user and 0 means it is unhelpful or poorly framed.",
    single_sentence,
    cheap,
    (4.0, 2.0, 1.0, 1.0),
)

result = Evaluator(meta_lm=judge_lm).evaluate(
    intent=intent,
    program=program,
    objective=objective,
    values={
        "question": "Capital of France",
        "answer": "Paris",
        "retrieved_context": ["Paris is the capital of France."],
        "draft_answer": "Paris is the capital of France.",
    },
    runtime=runtime,
)
```

This example is honest about all four layers:

- `Intent` says what behavior we care about
- `Program` says what intermediate artifacts we expose
- `Objective` says how success is judged
- `Evaluator` says how judging is carried out

---

## Recommended naming direction

If we adopt this model, the old name `Signature` is no longer the best name for
the richer object.

Two reasonable paths are:

### Path A: keep `Signature`, but admit it is program-facing

Then the docs should say:

> A `Signature` is a semantic program interface, not only a pure intent spec.

This is acceptable, but less clean.

### Path B: split the concepts explicitly

- `Intent` for pure behavioral intent
- `Program` for protocol / decomposition / artifacts

This is the cleaner long-term direction.

---

## Practical guidance for users

If we adopt this mental model, users should ask themselves:

### Is this part of the stable behavioral target?
Put it in **Intent**.

Examples:

- inputs
- outputs
- examples
- rubrics
- metrics
- external artifact being imitated

### Is this part of one chosen decomposition or protocol?
Put it in **Program**.

Examples:

- reasoning
- evidence
- plan
- draft answer
- retrieved context

### Is this part of how scoring is executed?
Put it in **Evaluator**.

Examples:

- judge LM
- adapters
- prompt rendering
- runtime telemetry
- score aggregation

### Is this part of success criteria?
Put it in **Objective**.

Examples:

- factual correctness rubric
- directness rubric
- latency metric
- cost metric
- relative weights

---

## Proposed minimal next step

The smallest honest next step is:

1. keep current lightweight `Objective` syntax
2. introduce `Intent`
3. move hidden fields / intermediate artifacts to `Program`
4. move `.evaluate(...)` off the current signature object and onto `Evaluator`
5. allow a default LM-backed judge, plus custom judges

This would preserve most of the good ergonomic work while giving users a much
clearer mental model.

---

## One-sentence summary

Onux should treat **behavioral intent**, **program structure**, **scoring
criteria**, and **evaluation execution** as separate concepts, because users
need all four, and pretending they are one thing makes the API muddy.
