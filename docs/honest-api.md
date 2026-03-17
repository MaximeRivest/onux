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
- **Module = strategy pattern.** The execution shape: control flow,
  decomposition, runner type. Chain-of-thought, ReAct, ensemble, etc.
- **Program = concrete candidate.** A module with every parameter pinned
  down: specific runner, adapter, hint, via fields, temperature. What
  the optimizer actually produces, evaluates, and saves.

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

## Level 2: Execution strategy (Modules and Programs)

There are two concepts at this level:

- **Module** — a general strategy pattern. It defines the *shape* of the
  execution: what kind of control flow, what kind of decomposition, what
  kind of runner. But it leaves parameters open.

- **Program** — a specification that pins down some or all of a module's
  parameters. A fully pinned program can be executed, scored, serialized,
  and compared. A partially pinned program defines a search space the
  optimizer explores.

Both fulfill the same contract:

```python
(sig, inputs) -> dict
```

### Every axis is independently open, seeded, or pinned

A program is not "general" or "specific" as a whole. Each axis has one
of three states:

- **Open** — no initial value. The optimizer searches freely.
- **Seeded** — an initial value as a starting point. The optimizer can
  explore from it.
- **Pinned** — a fixed value. The optimizer does not touch it.

| Axis | Open | Seeded | Pinned |
|---|---|---|---|
| Runner | "find a runner" | start with `LM("gpt-4o")` | must be `LM("ft:gpt-4o:triage-v3")` |
| Adapter | auto-generate | start with this template | must use this template |
| Via fields | "find a decomposition" | start with `[("reasoning",)]` | must use exactly these |
| Hint | "generate one" | start with this text | must use this text |
| Temperature | "find a value" | start with 0.3 | must be 0.0 |

A program handed to an optimizer is naturally a seed — every value is a
starting point, and the optimizer explores from there. The user then
selectively pins axes they don't want changed:

```python
# Everything here is a seed — optimizer starts from this program
# but can change any axis
seed = Program(
    module=chain_of_thought,
    runner=LM("gpt-4o", temperature=0.0),
    hint="Read the support request carefully. Consider what the customer "
         "is experiencing and which team can resolve it.",
)

# Search from the seed, but pin the runner — only explore hint,
# via fields, temperature, adapter
results = optimizer.search(
    intent=triage,
    seed=seed,
    pin=["runner"],
)

# Or pin nothing — let the optimizer change everything, even the
# module and runner type
results = optimizer.search(
    intent=triage,
    seed=seed,
)

# Or start from scratch — no seed, everything open
results = optimizer.search(
    intent=triage,
)
```

A checkpoint is a fully concrete program produced by the optimizer —
every axis has a specific value, and it can be executed and reproduced.

### Modules: strategy patterns

A module defines the execution shape. Built-in modules include:

- **predict** — single runner call, no hidden fields
- **chain_of_thought** — adds `.via("reasoning")`, then predicts
- **react** — adds reasoning, enters a tool loop
- **refine** — wraps another program in a validation-retry loop
- **ensemble** — runs multiple programs, votes on the result
- **fallback** — tries one program, falls back to another on failure
- **code_exec** — generates code, runs it, fixes errors in a loop

The module is independent of the runner type. `predict` works with an LM,
an sklearn model, a regex, or a custom Python function. `ensemble` works
with any combination of inner programs — one could be an LM, another a
finetuned model, another a rule-based classifier.

### Programs: concrete or seed

A program specifies concrete values for a module's axes. It can be
executed directly, or handed to an optimizer as a seed.

**Simplest program.** Predict with a specific runner, everything else
auto-generated.

```python
simple = Program(
    module=predict,
    runner=LM("gpt-4o-mini", temperature=0.0),
)

result = simple(triage, {"support_request": "I was charged twice."})
# => {"severity": "high", "team": "billing", "customer_visible": True}
```

**Chain-of-thought with prompt engineering.** The module adds reasoning
internally. The program specifies the runner and hint.

```python
with_reasoning = Program(
    module=chain_of_thought,
    runner=LM("gpt-4o", temperature=0.0),
    hint="Read the support request carefully. Consider what the customer "
         "is experiencing and which team can resolve it.",
)

result = with_reasoning(triage, {"support_request": "I was charged twice."})
# => {"severity": "high", "team": "billing", "customer_visible": True}

# This program works as-is. But it also works as a seed:
# "start here, find something better"
results = optimizer.search(intent=triage, seed=with_reasoning)
```

**Finetuned model.** The model already knows the task — no hint, no
reasoning, no prompt engineering needed.

```python
finetuned = Program(
    module=predict,
    runner=LM("ft:gpt-4o:triage-v3"),
)
```

**Non-LM runner.** An sklearn classifier. No adapter, no prompt. The
runner already speaks in dicts.

```python
from_sklearn = Program(
    module=predict,
    runner=SklearnModel("triage_classifier.joblib"),
)
```

**Composite programs.** Ensemble and refine compose inner programs, each
of which can mix different runner types and specificity levels.

```python
# Ensemble: vote across a general LM, a finetuned LM, and sklearn
voted = Program(
    module=ensemble,
    programs=[
        Program(module=chain_of_thought, runner=LM("gpt-4o", temperature=0.0)),
        Program(module=predict, runner=LM("ft:gpt-4o:triage-v3")),
        Program(module=predict, runner=SklearnModel("triage_v2.joblib")),
    ],
    n=3,
)

# Refine: validation loop around a chain-of-thought program
checked = Program(
    module=refine,
    inner=Program(module=chain_of_thought, runner=LM("gpt-4o")),
    check=lambda result: (
        None if result["severity"] in ("low", "medium", "high", "critical")
        else "Invalid severity value"
    ),
    max_retries=3,
)
```

All programs fulfill the same signature with the same contract.

### What an optimizer produces

An optimizer searches over programs by varying open axes. Each candidate
is evaluated against the fixed Signature:

```python
candidates = [
    # Cheap and fast — small model, no reasoning
    Program(module=predict, runner=LM("gpt-4o-mini", temperature=0.0)),

    # Reasoning with a general model
    Program(
        module=chain_of_thought,
        runner=LM("gpt-4o", temperature=0.0),
        hint="Consider urgency, affected system, and customer impact.",
    ),

    # Custom decomposition — optimizer-generated via fields and hint
    Program(
        module=predict,
        runner=LM("claude-3-5-sonnet", temperature=0.0),
        via=[
            ("urgency_analysis", {"note": "How urgent is this?"}),
            ("system_analysis", {"note": "What system is affected?"}),
        ],
        hint="Analyze urgency and affected system separately, then decide.",
    ),

    # Finetuned model — no prompt engineering needed
    Program(module=predict, runner=LM("ft:gpt-4o:triage-v3")),

    # Trained classifier — skip the LM entirely
    Program(module=predict, runner=SklearnModel("triage_v3.joblib")),

    # Ensemble: general LM + finetuned + sklearn
    Program(
        module=ensemble,
        programs=[
            Program(module=chain_of_thought, runner=LM("gpt-4o")),
            Program(module=predict, runner=LM("ft:gpt-4o:triage-v3")),
            Program(module=predict, runner=SklearnModel("triage_v3.joblib")),
        ],
        n=3,
    ),
]

for candidate in candidates:
    score = evaluate(triage, candidate, test_set)
    print(f"{candidate}: {score}")
```

Every candidate gets the same Signature, examples, and objective.

Because programs are data (not opaque Python functions), the optimizer
can:

- serialize, log, and reproduce every candidate
- mutate programs (add a via field, swap the runner, tweak temperature)
- crossover programs (combine the via fields of one with the runner of
  another)
- generate entirely new programs from scratch
- save winning programs as checkpoints

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

### On the module (strategy pattern)

The general execution shape:

- Control flow pattern (single call, loop, retry, vote, tool loop)
- Decomposition pattern (which hidden fields to add)

### On the program (each axis: open, seeded, or pinned)

- Module choice
- Runner (from "any LM" to a specific finetuned model to sklearn)
- Adapter (auto-generated or hand-written)
- Prompt engineering (hint text, field notes)
- Runner parameters (temperature, model name, etc.)
- Control flow parameters (max retries, tool list, etc.)

Each axis is independently open, seeded, or pinned. A program can be
executed directly (if concrete enough) or handed to an optimizer as a
seed. A checkpoint is a fully concrete program — every axis has a
specific value.

### Teaching order

1. **Specify intent.** Write a Signature with inputs, outputs, types,
   examples, and objective.

2. **Build a program.** Pick a module, specify values for the axes you
   know. Run it directly, or hand it to the optimizer as a seed. Pin
   axes you don't want changed. Leave the rest open or seeded.

3. **Wire a graph** if the task needs multiple steps.

`.via()`, `.hint()`, and `.note()` stay on Signature as building blocks
that modules and programs use internally. But the primary user path is:
specify intent, build program (or let the optimizer build one).

### Why this matters for optimization

If intent and execution strategy are cleanly separated, an optimizer can:

- Hold the Signature fixed (inputs, outputs, types, examples, objective)
- Search over modules (predict, chain_of_thought, react, custom)
- Search over open and seeded axes (runner, adapter, via fields,
  hint, notes, temperature, retries — whatever isn't pinned)
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
| Strategy pattern | Module | Execution shape — control flow, decomposition |
| Runner | Program (axis) | "any LM" → `LM("gpt-4o")` → `LM("ft:gpt-4o:triage-v3")` → `SklearnModel(...)` |
| Adapter | Program (axis) | Auto-generated or hand-written |
| Hint text, field notes | Program (axis) | Auto-generated, optimizer-searched, or pinned |
| Hidden fields | Program (axis) | Module default or custom |
| Control flow params | Program (axis) | Retries, tool list, temperature, etc. |
| Multi-step data flow | Graph (Layer, Model) | Composition of modules |
| Runtime telemetry | `runtime` parameter | Not a task field |

The honest truth is:

- **A Signature says WHAT to produce.** Inputs, outputs, types, examples,
  objective. It is the behavioral target the optimizer holds fixed.
- **A module defines the strategy pattern.** Chain-of-thought, ReAct,
  ensemble, refine — these are execution shapes. They define control
  flow and decomposition without pinning down specific parameters.
- **A program specifies values for a module's axes.** Each axis (runner,
  adapter, hint, via fields, temperature, etc.) is independently open,
  seeded, or pinned. A program can be executed directly or handed to an
  optimizer as a seed. A checkpoint is a fully concrete program. The
  optimizer produces, evaluates, mutates, and saves programs.
- **A graph wires multiple modules together.** Each node has a signature
  and a module. The graph handles data flow.
- **The optimizer test keeps the boundary clean.** If an optimizer would
  change it, it is execution strategy. If not, it is intent.
