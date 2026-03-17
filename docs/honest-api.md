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
holds fixed:

```python
sig = (
    Signature("question -> answer")
    .type(answer=str)
    .examples([
        {"question": "2 + 2", "answer": "4"},
        {"question": "Capital of France", "answer": "Paris"},
    ])
    .objective(
        "Assign a score from 0 to 1 for factual correctness.",
        single_sentence,
        (2.0, 1.0),
    )
)
```

No `.via()`. No `.hint()`. No `.note()`. No hidden fields. No prompt
engineering. Just: what goes in, what comes out, what good looks like.

### How little can you specify?

The Signature API should let the user specify only what they know, and
infer the rest. The progression from minimal to fully explicit:

**Scored examples only.** The most minimal useful intent. Input columns,
output columns, and types can all be inferred from the data. A sota model
can generate rubrics from the examples and scores.

```python
sig = Signature(
    examples=[
        {"question": "2 + 2", "answer": "4", "_score": 1.0},
        {"question": "Capital of France", "answer": "Paris", "_score": 1.0},
        {"question": "Largest planet", "answer": "Earth", "_score": 0.0},
    ]
)
```

**Formula + examples.** The user knows the field names but lets types be
inferred.

```python
sig = Signature(
    "question -> answer",
    examples=[
        {"question": "2 + 2", "answer": "4"},
        {"question": "Capital of France", "answer": "Paris"},
    ],
)
```

**Formula + types + objective.** Fully explicit intent. Nothing left to
infer.

```python
sig = (
    Signature("question -> answer")
    .type(answer=str)
    .examples([...])
    .objective("rubric...", metric_fn, (2.0, 1.0))
)
```

At every level, the user specifies what they can, and the system fills in
the rest. The result is always a complete behavioral target that can be
evaluated against.

Hint text, field notes, and hidden fields are NOT part of intent. They are
prompt engineering and decomposition choices — execution strategy that the
optimizer searches over. A module or adapter provides these when it
fulfills the signature.

### Reference artifacts

Sometimes the real goal is to imitate or replace an existing system. A
reference artifact is part of intent: it anchors the target behavior to
something concrete.

```python
sig = (
    Signature("support_request -> severity, team")
    .artifact(legacy_triage_service)
)
```

An artifact can serve as an example generator, a behavioral baseline for
evaluation, or a concrete definition of "what good looks like" beyond
what a few examples can capture. It is intent, not execution strategy,
because the optimizer holds it fixed.

---

## Level 2: Execution strategy (Modules)

A module takes any signature and fulfills it. The contract is:

```python
(sig, inputs) -> dict
```

That is all. The module decides HOW to produce the outputs — and HOW could
be anything:

- an LM call with a prompt adapter
- a regex
- a Python function
- an sklearn model
- a PyTorch model
- a vision model (SAM, image-to-text)
- a speech-to-speech pipeline
- a small LM, a big LM, or many LMs
- any combination of the above

The module decides what resources it needs internally. An LM-based module
needs an LM and knows how to render the signature into a prompt and parse
the response back into fields. A regex-based module needs a pattern. An ML
module needs a model. A code module needs nothing external.

The adapter concept — transforming between the signature's semantic fields
and whatever the runner natively understands — is part of the module's
internal strategy. It can be hand-written, auto-generated, or unnecessary
depending on the runner. It is not a fixed part of the contract.

This is also where hidden fields, hint text, and field notes live — the
module adds them internally as part of its strategy. Prompt engineering
belongs here, not in the intent.

### Pure generation strategies

These modules augment the signature with hidden fields, then predict.

**chain_of_thought**: Takes the signature, internally adds
`.via("reasoning")`, calls predict. The LM sees the reasoning field and
fills it before filling the outputs. The user never writes `.via()` — the
module handles it.

Any module that works by adding hidden fields is in this category. You
could write `plan_first` (adds `.via("plan")`), `evidence_then_answer`
(adds `.via("evidence")`), etc. They all augment the signature and predict.

### Pure procedural strategies

These modules wrap predict in control flow without changing the signature.

**ensemble**: Calls predict N times, votes across the results. The
signature and each LM call are unchanged — only the surrounding code
differs.

**fallback**: Tries one module, catches failure, falls back to another.
Pure error handling.

**refine**: Calls predict, runs a validator, retries on failure. The
validation and retry loop are procedural. The signature is not augmented.

### Mixed strategies

These modules augment the signature AND run code between LM calls.

**react**: Adds `.via("reasoning")`, then enters a tool loop. Calls the
LM, parses a tool action, executes the tool, injects the observation,
repeats. Both generation strategy (hidden fields) and procedural code
(tool loop).

**code_exec**: Adds a code field, then runs a generate-lint-fix loop.
Both generation strategy and procedural code.

### Exotic strategies

At the far end, strategies that deeply interleave code with LM execution.

**Streaming token detection**: The LM streams tokens. Code monitors the
stream, detects a pattern, pauses generation, runs external code, injects
results, resumes via assistant prefill.

**Multi-model orchestration**: One LM generates a plan. Code dispatches
subtasks to different LMs. Results are collected and merged.

### The spectrum is smooth

A user can move along this spectrum incrementally:

1. Start with `predict`. The signature's output fields are filled by one
   LM call.

2. Switch to `chain_of_thought` when you want the module to add a
   reasoning step. Still one predict call, but the signature is augmented
   internally.

3. Wrap in `ensemble` or `refine` when you want reliability through
   repetition. Pure procedural — the predict call is unchanged.

4. Switch to `react` when you need tool calls. Now the module both
   augments the signature and runs procedural code between LM calls.

5. Write a custom module when you need streaming detection, prefill
   tricks, or any other exotic strategy.

At every level, the contract is the same. The signature says WHAT to
produce. The module says HOW to produce it. The optimizer can swap
modules freely because the intent is fixed.

---

## Level 3: Composition (Graphs)

Wire multiple steps together when the task needs several module calls,
retrieval, SQL execution, or reusable subgraphs.

```python
question = Input("question")
context = Retrieve()(question)
answer = Generate("answer")([question, context])
model = Model(inputs=question, outputs=answer)
```

Each node in the graph has an inner signature and a module. The graph
handles data flow between nodes.

---

## What this means for `.via()`, `.hint()`, and `.note()`

These methods stay on Signature — modules need them internally to augment
signatures as part of their strategy. But users should not need to call
them when specifying intent.

- `.via()` adds hidden fields. Modules call it internally.
- `.hint()` sets task-level prompt text. Modules or adapters set it.
- `.note()` sets per-field prompt text. Modules or adapters set it.

The teaching order is:

1. **Specify intent.** Write a Signature with inputs, outputs, types,
   examples, and objective. No `.via()`, `.hint()`, or `.note()`.

2. **Choose a module.** The module adds whatever hidden fields, hints,
   and notes it needs. `chain_of_thought` adds reasoning. `react` adds
   reasoning and a tool loop. The user picks the strategy, not the
   prompt details.

3. **Wire a graph** if the task needs multiple steps.

If a power user wants to manually call `.via()`, `.hint()`, or `.note()`
on a signature, they can. But the primary path should be: specify intent,
choose module, let the module handle prompt engineering and decomposition.

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

---

## What does not belong in a Signature

### Runtime telemetry

Values like `latency_ms`, `cost_usd`, `prompt_tokens`, `retry_count` are
not task fields. They are execution telemetry. Metrics can ask for a
`runtime` parameter, and `.evaluate()` accepts a `runtime` keyword.

### Judge LMs and adapters

How rubrics get scored is an evaluation execution concern. The signature
says *what* to score. A future `Evaluator` helper could automate rubric
judging with a judge LM, but that does not need to change the Signature.

### Prompt rendering and adaptation

How the signature is rendered into a prompt (or transformed into whatever
the runner natively understands) is the module's job. The signature
describes the semantic structure. The module and its adapter handle
presentation. The adapter itself can be auto-generated.

### Hidden fields, hints, and notes

Hidden fields, hint text, and field notes are execution strategy. They
belong in modules, not in the user's intent specification. `.via()`,
`.hint()`, and `.note()` exist on Signature as building blocks for modules
and adapters, not as primary user-facing APIs for intent.

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
- **A module says HOW to produce it.** The runner could be an LM, a
  regex, a Python function, an ML model, a vision model, or any
  combination. The strategy could be pure generation (chain-of-thought),
  pure procedural (ensemble, refine), mixed (ReAct, code_exec), or
  exotic (streaming detection, prefill manipulation). The adapter, prompt
  engineering, hidden fields, and runner choice all live here. The
  spectrum is smooth.
- **A graph wires multiple modules together.** Each node has a signature
  and a module. The graph handles data flow.
- **The optimizer test keeps the boundary clean.** If an optimizer would
  change it, it is execution strategy. If not, it is intent.
