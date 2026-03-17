# Honest API: intent, strategy, composition

This note clarifies the design of Onux by being honest about what each
layer actually does and where each concern belongs.

---

## The optimizer test

The cleanest way to decide whether something is intent or execution
strategy is to ask:

> Would an optimizer change this?

If yes, it is execution strategy. If no, it is intent.

- **Input and output fields**: no, these define the task. Intent.
- **Types**: no, these constrain the behavioral contract. Intent.
- **Examples**: no, these anchor the target behavior. Intent.
- **Objective (rubrics and metrics)**: no, these define success. Intent.
- **Reference artifacts**: no, these anchor the behavioral target. Intent.
- **Hint** ("Answer in one sentence."): yes, an optimizer would rephrase
  or rewrite it. Execution strategy (prompt engineering).
- **Notes** ("Short factual answer"): yes, an optimizer would rephrase
  per-field descriptions. Execution strategy (prompt engineering).
- **Hidden fields (`.via("reasoning")`)**: yes, an optimizer might add,
  remove, rename, or reorder them. Execution strategy.
- **Module choice (predict, chain_of_thought, react)**: yes, an optimizer
  might swap one for another. Execution strategy.
- **Runner (LM, regex, ML model, code, vision model, etc.)**: yes.
  Execution strategy.
- **Adapter (prompt rendering, output parsing)**: yes, can even be
  auto-generated. Execution strategy.

This gives us a clean separation:

- **Signature = intent.** What the task is, what good looks like. Elements
  in the signature are not parameters to optimize over — they are the
  fixed target the optimizer holds constant while it searches.
- **Module = execution strategy.** How to fulfill the signature, including
  what runner to use (LM, regex, ML model, code, vision model, etc.),
  what adapter to use, what intermediate fields to generate, what hint
  and field notes to use, and how to structure the execution.

---

## Level 1: Intent (Signature)

A Signature is the behavioral target. It contains only things the optimizer
holds fixed. No `.via()`. No `.hint()`. No `.note()`. No hidden fields. No
prompt engineering. Just: what goes in, what comes out, what good looks
like.

### Example: support ticket triage

```python
from typing import Literal
from onux import Signature

Severity = Literal["low", "medium", "high", "critical"]
Team = Literal["billing", "bug", "feature", "account"]


def valid_structured_output(severity: str, team: str, customer_visible: bool) -> float:
    """1.0 if all outputs are well-formed, 0.0 otherwise."""
    return float(
        severity in ("low", "medium", "high", "critical")
        and team in ("billing", "bug", "feature", "account")
        and isinstance(customer_visible, bool)
    )


triage = (
    Signature("support_request -> severity, team, customer_visible")
    .type(severity=Severity, team=Team, customer_visible=bool)
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
        {
            "support_request": "Can you add CSV export to the reporting dashboard?",
            "severity": "low",
            "team": "feature",
            "customer_visible": False,
        },
    ])
    .objective(
        "Assign a score from 0 to 1 for routing accuracy, where 1 means the "
        "request is assigned to the correct team and severity and 0 means the "
        "routing is clearly wrong.",
        valid_structured_output,
        (3.0, 1.0),
    )
)
```

This signature says everything about what the task IS and what good looks
like. It says nothing about how to execute it — no LM, no prompt, no
reasoning steps.

### How little can you specify?

The Signature API should let the user specify only what they know, and
infer the rest.

**Scored examples only.** The most minimal useful intent. Fields, types,
and objective can all be inferred from the data and scores.

```python
triage = Signature(
    examples=[
        {
            "support_request": "I was charged twice for my subscription.",
            "severity": "high",
            "team": "billing",
            "customer_visible": True,
            "_score": 1.0,
        },
        {
            "support_request": "The export button is misaligned in dark mode.",
            "severity": "critical",
            "team": "feature",
            "customer_visible": False,
            "_score": 0.0,
        },
    ]
)
```

**Formula + examples.** The user names the fields, types are inferred.

```python
triage = Signature(
    "support_request -> severity, team, customer_visible",
    examples=[
        {
            "support_request": "I was charged twice for my subscription.",
            "severity": "high",
            "team": "billing",
            "customer_visible": True,
        },
    ],
)
```

**Fully explicit.** The user specifies everything. Nothing left to infer.

```python
triage = (
    Signature("support_request -> severity, team, customer_visible")
    .type(severity=Severity, team=Team, customer_visible=bool)
    .examples([...])
    .objective("rubric...", valid_structured_output, (3.0, 1.0))
)
```

At every level, the user specifies what they can, and the system fills in
the rest. The result is always a complete behavioral target that can be
evaluated against.

### Reference artifacts

Sometimes the real goal is to imitate or replace an existing system. A
reference artifact anchors the behavioral target to something concrete:

```python
triage = (
    Signature("support_request -> severity, team, customer_visible")
    .type(severity=Severity, team=Team, customer_visible=bool)
    .artifact(Artifact.service(
        "legacy-triage-api",
        endpoint="https://internal.example.com/triage/v2",
    ))
    .examples([...])
    .objective("rubric...", valid_structured_output, (3.0, 1.0))
)
```

The artifact says: "the behavior we want is whatever this existing system
does, plus any improvements we can measure through the objective." It can
serve as an example generator, a behavioral baseline for evaluation, or a
concrete definition of "what good looks like." It is intent, not execution
strategy, because the optimizer holds it fixed.

---

## Level 2: Execution strategy (Modules)

A module takes any signature and fulfills it. The contract is:

```python
(sig, inputs) -> dict
```

That is all. The module decides HOW to produce the outputs — and HOW could
be anything: an LM call, a regex, a Python function, an sklearn model, a
PyTorch model, a vision model, a speech-to-speech pipeline, a small LM, a
big LM, many LMs, or any combination.

### Modules are built, not just chosen

Modules are not a fixed menu of presets. They are *built* — by humans or
by optimizers. A module specification is a recipe that describes:

- **Runner**: what executes (LM, regex, code, ML model, vision model, etc.)
- **Adapter**: how to transform between signature fields and the runner's
  native format (can be auto-generated)
- **Generation strategy**: what hidden fields to add, what hint and notes
  to set (prompt engineering)
- **Control flow**: loops, retries, tool calls, streaming, voting
- **Parameters**: temperature, max iterations, model name, etc.

This recipe must be readable and writable by both humans and AI, because
the optimizer needs to produce, mutate, and evaluate candidate strategies.

### Example: three modules for the same triage intent

The same triage signature from Level 1 can be fulfilled by completely
different modules. Each one is a different answer to "how do we produce
severity, team, and customer_visible from a support_request?"

**Direct LM call.** The simplest module — hand the signature to an LM.

```python
simple = Module(
    runner=LM("gpt-4o-mini", temperature=0.0),
)

result = simple(triage, {"support_request": "I was charged twice."})
# => {"severity": "high", "team": "billing", "customer_visible": True}
```

The module auto-generates an adapter: it renders the signature's fields
and types into a prompt, calls the LM, and parses the structured response
back into a dict.

**LM with reasoning.** Add a hidden reasoning field so the LM thinks
before classifying.

```python
with_reasoning = Module(
    runner=LM("gpt-4o", temperature=0.0),
    via=[("reasoning", {"note": "Analyze the request before classifying."})],
    hint="Read the support request carefully. Consider what the customer "
         "is experiencing and which team can resolve it.",
)

result = with_reasoning(triage, {"support_request": "I was charged twice."})
# => {"severity": "high", "team": "billing", "customer_visible": True}
```

The reasoning field is produced by the LM but not returned to the caller —
it is an internal artifact of this execution strategy. A different module
might not use reasoning at all.

**Sklearn model.** No LM, no prompt, no reasoning. A trained classifier
that maps text to labels.

```python
from_sklearn = Module(
    runner=SklearnModel("triage_classifier.joblib"),
)

result = from_sklearn(triage, {"support_request": "I was charged twice."})
# => {"severity": "high", "team": "billing", "customer_visible": True}
```

The module knows how to extract features from the input text and map the
model's output back to the signature's typed fields. No adapter needed —
the runner already speaks in dicts.

All three modules fulfill the same signature with the same contract. The
optimizer can evaluate all three against the same examples and objective,
then pick the one with the best score-cost tradeoff.

### Building from presets

Presets are named starting points. Each preset already knows its own
generation strategy — `chain_of_thought` adds reasoning internally,
`react` adds reasoning and a tool loop. The user specifies the runner
and parameters, not the preset's internal decomposition:

```python
# chain_of_thought internally adds via("reasoning") and predicts
module = chain_of_thought(runner=LM("gpt-4o"))

# refine wraps another module in a validation loop
module = refine(
    chain_of_thought(runner=LM("gpt-4o")),
    check=lambda result: (
        None if result["severity"] in ("low", "medium", "high", "critical")
        else "Invalid severity value"
    ),
    max_retries=3,
)

# ensemble runs multiple modules and votes
module = ensemble(
    chain_of_thought(runner=LM("gpt-4o")),
    chain_of_thought(runner=LM("claude-3-5-sonnet")),
    n=3,
)
```

Presets are convenient but not special. They are just named recipes that
configure the same underlying building blocks. If you want to control
which hidden fields are added, build from scratch with `Module(...)`.

### What an optimizer produces

An optimizer searches over module specifications. Each candidate is a
concrete recipe evaluated against the fixed Signature:

```python
# The optimizer tries these candidates for the same triage intent:

candidates = [
    # Cheap and fast — no reasoning, small model
    Module(runner=LM("gpt-4o-mini", temperature=0.0)),

    # Careful reasoning, bigger model
    Module(
        runner=LM("gpt-4o", temperature=0.0),
        via=[("reasoning", {"note": "Analyze the customer's situation."})],
        hint="Consider urgency, affected system, and customer impact.",
    ),

    # Multi-step: classify severity first, then route
    Module(
        runner=LM("claude-3-5-sonnet", temperature=0.0),
        via=[
            ("urgency_analysis", {"note": "How urgent is this?"}),
            ("system_analysis", {"note": "What system is affected?"}),
        ],
        hint="Analyze urgency and affected system separately, then decide.",
    ),

    # Skip the LM entirely
    Module(runner=SklearnModel("triage_v3.joblib")),

    # Ensemble: vote across two models
    ensemble(
        Module(runner=LM("gpt-4o", temperature=0.0)),
        Module(runner=LM("claude-3-5-sonnet", temperature=0.0)),
        n=3,
    ),
]

# Every candidate is evaluated against the same intent:
for candidate in candidates:
    score = evaluate(triage, candidate, test_set)
    print(f"{candidate}: {score}")
```

Because module specs are data (not opaque Python functions), the optimizer
can serialize and log every candidate, mutate specs (add a via field,
change the runner, tweak temperature), crossover specs, or generate
entirely new ones.

### The strategy spectrum

Module strategies range from simple to exotic:

**Pure generation**: augment the signature with hidden fields, then
predict. Chain-of-thought is here — it just adds `.via("reasoning")`.

**Pure procedural**: wrap a runner in control flow without changing the
signature. Ensemble (majority voting), fallback, and refine are here.

**Mixed**: augment the signature AND run code between runner calls. ReAct
(tool loop) and code_exec (generate-lint-fix loop) are here.

**Exotic**: deeply interleave code with runner execution. Streaming token
detection, assistant prefill manipulation, multi-model orchestration.

The spectrum is smooth. At every level, the contract is the same:
`(sig, inputs) -> dict`.

---

## Level 3: Composition (Graphs)

Wire multiple steps together when the task needs several module calls,
retrieval, SQL execution, or reusable subgraphs. Each node in the graph
has an inner signature and a module. The graph handles data flow.

### Example: triage with context lookup

The triage task might benefit from looking up the customer's history
before classifying. That requires two steps — retrieval, then triage —
which is a graph:

```python
from onux import Input, Model
from onux.layers import Generate, Retrieve

support_request = Input("support_request")

# Step 1: retrieve past tickets for this customer
past_tickets = Retrieve(
    runner=VectorDB("tickets_index"),
    top_k=5,
)(support_request)

# Step 2: triage with context — uses the triage signature internally
severity, team, customer_visible = Generate(
    outputs=(
        ("severity", Severity),
        ("team", Team),
        ("customer_visible", bool),
    ),
    runner=LM("gpt-4o", temperature=0.0),
    via=[("reasoning", {"note": "Consider past tickets and current request."})],
)([support_request, past_tickets])

triage_pipeline = Model(
    inputs=support_request,
    outputs=[severity, team, customer_visible],
    name="triage_with_context",
)

triage_pipeline.summary()
# Model: "triage_with_context"
# Inputs:
#   - support_request: str
# Outputs:
#   - severity: Severity (public)
#   - team: Team (public)
#   - customer_visible: bool (public)
# Graph:
#   - [1] Retrieve(past_tickets) <- support_request
#   - [2] Generate(severity, team, customer_visible) <- support_request, past_tickets
```

### Example: research pipeline

A more complex graph with multiple LM calls, retrieval, and critique:

```python
question = Input("question")
constraints = Input("constraints", type=list[str])

sources = Retrieve(runner=VectorDB("docs_index"), top_k=10)(question)

draft = Generate(
    "draft",
    runner=LM("gpt-4o"),
    via=[("reasoning", {"note": "Work through the problem."})],
)([question, constraints, sources])

critique = Generate(
    "critique",
    runner=LM("claude-3-5-sonnet", temperature=0.0),
    hint="Find factual errors, unsupported claims, and gaps.",
)([question, draft, sources])

final_answer, confidence = Generate(
    outputs=(("final_answer", str), ("confidence", float)),
    runner=LM("gpt-4o"),
    hint="Revise the draft based on the critique.",
)([question, constraints, sources, draft, critique])

research = Model(
    inputs=[question, constraints],
    outputs=[final_answer, confidence],
    name="research_pipeline",
)
```

Each node is independently configurable — different runners, different
strategies, different prompt engineering. The graph handles data flow
between them.

---

## What belongs where

### On Signature (intent)

Only what the optimizer holds fixed:

- Input and output field names
- Types
- Examples
- Objective (rubrics and metrics)
- Reference artifacts

### On the module (execution strategy)

Everything the optimizer searches over:

- Runner choice (LM, regex, ML model, code, vision model, etc.)
- Adapter (rendering inputs for the runner, parsing outputs back to
  fields — can be auto-generated)
- Prompt engineering (hint text, field notes)
- Decomposition (hidden fields via `.via()`)
- Control flow (loops, retries, tool calls, streaming)
- Runner parameters (temperature, model name, etc.)

### Teaching order

1. **Specify intent.** Write a Signature with inputs, outputs, types,
   examples, and objective.

2. **Build a module.** Start from a preset or from scratch. The module
   specifies the runner, adapter, hidden fields, prompt engineering,
   control flow — everything about how to fulfill the intent. Or let
   the optimizer build and search over candidates for you.

3. **Wire a graph** if the task needs multiple steps.

`.via()`, `.hint()`, and `.note()` stay on Signature as building blocks
that modules use internally. But the primary user path is: specify intent,
build module (or let the optimizer build one).

### Why this matters for optimization

If intent and execution strategy are cleanly separated, an optimizer can:

- Hold the Signature fixed (inputs, outputs, types, examples, objective)
- Search over modules (predict, chain_of_thought, react, custom, or
  non-LM strategies like regex, code, ML models)
- Search over module internals (which runner, which adapter, which hidden
  fields, hint text, field notes, how many retries, temperature)
- Evaluate every candidate against the same fixed objective

This only works if the Signature does not contain execution strategy
decisions. The moment `.via("reasoning")` or `.hint("Answer briefly.")`
is part of the user's intent specification, the optimizer cannot freely
change it without conflicting with what the user asked for.

### Runtime telemetry

Values like `latency_ms`, `cost_usd`, `prompt_tokens`, `retry_count` are
not task fields and not execution strategy. They are execution telemetry.
Metrics can ask for a `runtime` parameter, and `.evaluate()` accepts a
`runtime` keyword.

### Rubric judging

How rubrics get scored is an evaluation execution concern. The signature
says *what* to score. A future `Evaluator` helper could automate rubric
judging, but that does not need to change the Signature.

---

## The Objective stays on the Signature

The objective is intent, not execution strategy. A rubric like "score
factual correctness" defines what good looks like — it does not change
based on which module you use. So it belongs on the Signature:

- Metric parameter names are validated against the signature's fields.
- Serialization via `dump_state()` / `load_state()` naturally includes it.
- One object to build, pass around, and inspect.

---

## Summary

| Concept | Where it lives | Why |
|---|---|---|
| Inputs, outputs, types | Signature | Behavioral intent — optimizer holds fixed |
| Examples | Signature | Behavioral specification |
| Rubrics and metrics | Signature | Scoring criteria — define success |
| Reference artifacts | Signature | Behavioral anchor |
| Hint text | Module / adapter (internally) | Prompt engineering — optimizer searches over |
| Field notes | Module / adapter (internally) | Prompt engineering — optimizer searches over |
| Hidden intermediate fields | Module (internally) | Execution strategy — optimizer searches over |
| Module choice | Module | Execution strategy |
| Runner (LM, regex, ML, code, etc.) | Module (internally) | Execution strategy |
| Adapter (prompt rendering, parsing) | Module (internally) | Execution strategy — can be auto-generated |
| Multi-step data flow | Graph (Layer, Model) | Composition of modules |
| Runtime telemetry | `runtime` parameter | Not a task field |
| Prompt rendering / adaptation | Module (internally) | Presentation — can be auto-generated |

The honest truth is:

- **A Signature says WHAT to produce.** Inputs, outputs, types, examples,
  objective. It is the behavioral target the optimizer holds fixed.
- **A module says HOW to produce it.** Modules are built, not just chosen
  from a menu. A module spec is a recipe: runner, adapter, hidden fields,
  prompt engineering, control flow. Both humans and optimizers build them.
  Presets like `chain_of_thought` and `react` are named starting points.
  The runner could be an LM, a regex, code, an ML model, a vision model,
  or any combination. The spectrum from simple to exotic is smooth.
- **A graph wires multiple modules together.** Each node has a signature
  and a module. The graph handles data flow.
- **The optimizer test keeps the boundary clean.** If an optimizer would
  change it, it is execution strategy. If not, it is intent.
