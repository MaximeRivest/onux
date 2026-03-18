# Honest API: intent, execution, composition, optimization

Onux is for building, specifying, sharing, evaluating, extending, and
automatically optimizing **compound AI systems**.

The core idea is simple:

> A compound AI system should have a clean description of **what it is
> trying to do**, a clean description of **how each part is executed**,
> and a clean description of **how the parts are wired together**.

That gives us four primary concepts:

- **Signature** — the intent
- **Preset** — a reusable execution strategy, not yet bound to a task
- **Program** — the execution of one task (a Preset bound to a Signature)
- **Graph** — the composition of many tasks

Everything else supports those four.

---

## 1. The design test

The most useful design question is:

> **Would an optimizer change this?**

If yes, it is part of execution strategy.
If no, it is part of intent.

This is not just a philosophical test. It is what keeps the API honest.
If a thing belongs to the fixed target, the optimizer must hold it fixed.
If a thing belongs to strategy, the optimizer must be free to change it.

### Intent

These define the task and success criteria:
- input fields
- output fields
- types
- examples
- objective
- optional reference artifacts or baselines

### Strategy

These are legitimate search dimensions:
- module choice
- runner choice
- adapter choice
- prompt wording (hint)
- hidden decomposition fields (via)
- field notes
- tool use
- retry policies
- ensembling strategies
- temperature, model, max_tokens, ...
- cost / latency tradeoffs

This leads to the main architectural split.

---

## 2. Signature is pure intent

A `Signature` is the behavioral target.
It says:
- what goes in
- what comes out
- what kinds of values are expected
- what good performance looks like

It does **not** say how the task is executed.

That means a Signature contains only:
- **input fields**
- **output fields**
- **types**
- **examples**
- **objective**
- optionally later: **reference artifacts**

It does **not** contain:
- prompt hints
- notes meant for prompting
- hidden reasoning fields
- execution traces
- runner choice
- decomposition strategy

### The point

A Signature is the thing the optimizer holds fixed.
It is the invariant contract.

### Example

```python
from typing import Literal
from onux import Signature

Severity = Literal["low", "medium", "high", "critical"]
Team = Literal["billing", "bug", "feature", "account"]


def valid_output(severity: str, team: str, customer_visible: bool) -> float:
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
            "support_request": "I was charged twice this month.",
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
    .objective(
        """Assign a score from 0 to 1 for routing accuracy, where 1 means the
           request is sent to the correct team with the correct severity and 0
           means it is clearly misrouted.""",
        valid_output,
        (3.0, 1.0),
    )
)
```

This says what the task is.
It does not say whether to use GPT-4o, a fine-tuned model, a sklearn
classifier, a rules engine, or a hybrid system.

### Formula discipline

A Signature formula has only two sides:

```python
"input1, input2 -> output1, output2"
```

No hidden middle stage in the formula.
If the user wants an intermediate artifact as part of the task, it is an
output:

```python
Signature("question -> reasoning, answer")
```

That means the user truly wants both `reasoning` and `answer`.
That is intent.

If reasoning is only internal scaffolding, it belongs to execution
strategy (see Modules below), not Signature. Generally, if you don't
plan to evaluate or optimize over this output, it should not be in the
Signature.

---

## 3. Preset is a reusable execution strategy

A `Preset` pairs a module with a runner. It holds all their axes in one
object. It is the reusable strategy template that can be applied to many
Signatures.

A Preset is **not callable**. It has no Signature. It does not know what
task it will run. It is a configured execution strategy waiting to be
bound.

### Creating a Preset

```python
from onux import Preset
from onux.modules import chain_of_thought, predict
from onux.runners import OpenAILM, AnthropicLM

# Minimal: just a module and a runner
base = Preset(predict, OpenAILM("gpt-4o"))

# Configured: module and runner with their axes set
reasoned = Preset(
    chain_of_thought.hint("Think step by step before answering."),
    OpenAILM("gpt-4o", temperature=0.0),
)
```

### Presets are reusable across Signatures

The whole point of a Preset is that the same strategy can serve many
tasks:

```python
# One strategy
strategy = Preset(
    chain_of_thought,
    OpenAILM("gpt-4o", temperature=0.0),
)

# Many tasks
triage_program = strategy.bind(triage)
qa_program = strategy.bind(question_answering)
summary_program = strategy.bind(summarization)
```

### Presets cannot run, cannot evaluate, cannot optimize

A Preset has no Signature, so it has no inputs to validate, no outputs
to check, no objective to evaluate against, and no call signature to
expose. It is an inert strategy template.

To become useful, a Preset must be bound to a Signature. That produces
a Program.

---

## 4. Program is how one task runs

A `Program` is a Preset bound to a Signature.

It is **callable**, **evaluable**, **optimizable**, and
**serializable**. It is the central executable object in Onux.

### Creating a Program

There are two ways:

```python
from onux import Program, Preset

# Way 1: bind a Preset to a Signature
preset = Preset(chain_of_thought, OpenAILM("gpt-4o"))
program = preset.bind(triage)

# Way 2: construct directly (Signature first)
program = Program(triage, chain_of_thought, OpenAILM("gpt-4o"))
```

Both produce the same thing: an executable Program that knows its
intent and its strategy.

### Calling a Program

A Program's Python call signature is generated from its Signature's
input fields. You call it with the input field values directly:

```python
# Signature: "support_request -> severity, team, customer_visible"
result = program(support_request="I was charged twice this month.")

# Positional args follow the Signature's input field order
result = program("I was charged twice this month.")

# Multiple inputs
qa = Program(qa_sig, predict, OpenAILM("gpt-4o"))
result = qa(question="What is 2+2?", context="A math textbook.")
result = qa("What is 2+2?", "A math textbook.")
```

The Program validates inputs against its Signature, executes the
module's strategy through the runner, validates outputs, and returns a
`Result`.

This is not `program(sig, inputs)`. The Signature is already bound.
The call site is clean.

### Programs surface all axes

A Program proxies all axes from its module and runner. Every tweak
returns a new Program (a fork):

```python
program = Program(triage, chain_of_thought, OpenAILM("gpt-4o"))

# Module axes — forwarded to the module
c1 = program.hint("Determine team first, then severity.")
c2 = program.hint("Separate issue type from urgency.").via("evidence")

# Runner axes — forwarded to the runner
c3 = program.temperature(0.3)
c4 = program.model("gpt-4o-mini")
c5 = program.max_tokens(2048)

# Mix both
c6 = (
    program
    .hint("Determine team from the user's underlying issue.")
    .via("evidence", note="Key facts supporting the team label")
    .temperature(0.0)
)
```

Each call returns a new immutable Program. The original is unchanged.
These forks are **checkpoints** — particular versions of the Program
with specific axis values.

### Checkpoints

A checkpoint is a Program with all axes pinned to concrete values. It
is the unit you evaluate, compare, save, and deploy.

Program and checkpoint are the **same type**. A checkpoint is just a
Program where `is_complete()` returns `True` — every axis has a value.

```python
# Fork checkpoints from a base program
base = Program(triage, chain_of_thought, OpenAILM("gpt-4o"))

c1 = base.hint("Determine team first.").temperature(0.0)
c2 = base.hint("Separate issue type from urgency.").temperature(0.3)
c3 = base.via("evidence").hint("Gather evidence, then classify.").temperature(0.0)

# Evaluate checkpoints
for c in [c1, c2, c3]:
    result = c(support_request="I was charged twice.")
    score = c.evaluate(result)
    print(f"{score}")

# Save the best checkpoint
best = c1
best.save("triage-v3")

# Load and run in production
production = Program.load("triage-v3")
result = production(support_request="New request here...")
```

### Evaluating a Program

Because a Program knows its Signature, it can evaluate results against
the Signature's objective directly:

```python
result = program(support_request="I was charged twice.")
score = program.evaluate(result)
# Uses the triage Signature's objective: rubrics, metrics, weights
```

This is a convenience over calling `triage.evaluate(result.outputs)`
manually. The Program already has the Signature, so it does the wiring.

### The optimizer takes a Program

Because a Program already carries its Signature, the optimizer needs
only one argument:

```python
best = optimizer.search(program)
```

The optimizer reads the Program's Signature (the fixed intent), reads
the Program's axes (the search space), explores checkpoints, evaluates
them, and returns the best one.

```python
base = Program(triage, chain_of_thought, OpenAILM("gpt-4o"))
best = optimizer.search(base)

# best is a checkpoint (a fully-pinned Program)
best.save("triage-v3")
```

### Why Program takes Signature first

The constructor is `Program(sig, module, runner)`, not
`Program(module, runner, sig)`.

This is deliberate:
- A Program **is** the execution of a specific Signature.
- The Signature defines the Program's call signature (its input args).
- Without a Signature, you have a Preset, not a Program.
- The Signature is the most important argument. It comes first.

### Preset vs Program summary

| | Preset | Program |
|---|---|---|
| Has Signature | No | Yes |
| Callable | No | Yes |
| Evaluable | No | Yes |
| Optimizable | No | Yes |
| Serializable | Yes (as template) | Yes (as checkpoint) |
| Surfaces axes | Yes | Yes |
| Forkable | Yes (returns Preset) | Yes (returns Program) |

A Preset becomes a Program when you give it a Signature.
A Program becomes a checkpoint when all its axes are pinned.

---

## 5. Module is the strategy pattern

A Module is the general execution pattern. It owns the control flow,
the decomposition, and the strategy-level axes.

### Module hierarchy

`predict` is the base module for LM runners. Its axes include `hint`,
`via`, and `field_note`. Other LM modules extend it:

```
predict
  axes: hint, via, field_note
  execution: single runner call

chain_of_thought (extends predict)
  axes: hint, via, field_note
  defaults: via("reasoning") pre-applied
  execution: reasoning first, then outputs

react (extends chain_of_thought)
  axes: hint, via, field_note, tools, max_iter
  defaults: via("reasoning"), max_iter=5
  execution: reasoning → tool loop → outputs
```

`hint` and `via` are **module axes**, not Signature concerns, not
Program-level config. They belong to `predict` and its descendants
because they are part of the execution strategy — the optimizer may
change them.

### Module types

Each module type is an immutable builder that knows its own axes:

```python
# predict: single call, base axes
predict
predict.hint("Return only the label and confidence.")
predict.field_note("confidence", "Probability from 0 to 1")

# chain_of_thought: predict + built-in reasoning
chain_of_thought
chain_of_thought.hint(
    "Work through the tax calculation carefully."
)

# react: chain_of_thought + tool loop
react.tools([search_docs, lookup_pricing])
react.tools([search_docs]).max_iter(5).hint(
    "Look up the exact API behavior in the docs first."
)

# refine: validation loop (wraps any inner module)
refine.check(my_validator).max_retries(3)

# ensemble: multi-program vote
ensemble.n(5)
```

Each builder method returns a new immutable module spec. Only the
methods that make sense for that module type are available.

### What a module is under the hood

A module has two faces:

1. **A builder / spec** — the immutable configuration object that users
   construct. It declares its axes for the optimizer and serializer.

2. **A callable** — the execution implementation. When a Program runs,
   it calls the module's execution function with the configured
   parameters:

```python
def chain_of_thought_fn(
    sig, inputs, *, runner, adapter, hint=None, via=None
) -> Result:
    ...
```

Users interact with face 1 (the builder). The framework uses face 2
(the callable) at execution time.

### Modules are broad

Modules are not LLM-specific. Domain-specific modules follow the same
pattern:

```python
# Vision
tile_and_merge.tile_size(640).overlap(0.2)
multi_scale.scales([0.5, 1.0, 2.0])

# ML
calibrate.method("isotonic")
stacking.meta_model(LogisticRegression())

# Audio
chunk_and_transcribe.chunk_seconds(30).overlap(5)
```

Each module type exposes its own axes as builder methods. The pattern is
always the same.

---

## 6. Runners are broad by design, not fake-universal

Onux must support a real climb up and down the hill of execution:
- from a general foundation model
- to a specialized fine-tuned model
- to deep learning
- to classical ML
- to deterministic code, feature engineering, regexes, and algorithms

### The correct abstraction

There is a tiny universal runner contract for the framework, and richer
domain-specific protocols for execution.

### Universal runner responsibilities

Every runner should be able to describe:
- its axes / tunable parameters
- its current config
- how it is serialized and reconstructed

```python
class Runner(Protocol):
    family: str
    def axes(self) -> dict[str, Axis]: ...
    def config(self) -> dict[str, Any]: ...
    def dump_state(self) -> dict[str, Any]: ...
```

### Domain-specific execution protocols

Actual execution differs by domain. That is normal.

```python
class LMRunner(Runner, Protocol):
    family: Literal["lm"]
    def complete(self, request: LMRequest) -> LMResponse: ...

class VisionRunner(Runner, Protocol):
    family: Literal["vision"]
    def predict(self, image, **kwargs) -> VisionResult: ...

class MLRunner(Runner, Protocol):
    family: Literal["ml"]
    def predict(self, features, **kwargs) -> MLResult: ...

class DeterministicRunner(Runner, Protocol):
    family: Literal["code"]
    def run(self, inputs, **kwargs) -> Any: ...
```

Modules know what runner family they need. The optimizer and serializer
care about the universal part.

### Runner config

Runners are their own immutable builders:

```python
OpenAILM("gpt-4o", temperature=0.0)
OpenAILM("gpt-4o").temperature(0.0).max_tokens(4096)
AnthropicLM("claude-3-5-sonnet", temperature=0.3)

# Adapter is a runner concern
OpenAILM("gpt-4o").adapter(XMLAdapter())

# Non-LM runners
YOLORunner("yolov8x.pt").conf_thresh(0.25).device("cuda")
SklearnRunner("classifier.joblib")
PythonRunner(my_function)
```

---

## 7. Adapters are real, but mostly advanced

An `Adapter` turns Signature fields and runtime values into the native
input/output format a runner expects.

### Adapter contract

```python
class Adapter(Protocol):
    def encode(self, sig: Signature, inputs: dict[str, Any]) -> Any: ...
    def decode(self, sig: Signature, raw_output: Any) -> dict[str, Any]: ...
```

### User experience

Most users should not have to think about adapters. The default is
`adapter="auto"` — the framework generates or chooses a sensible
adapter from the Signature's fields and types.

Advanced users can swap or author adapters explicitly. Adapter is a
runner concern, configured on the runner:

```python
OpenAILM("gpt-4o").adapter(XMLAdapter())
OpenAILM("gpt-4o").adapter(StructuredOutputAdapter())
```

Non-LM runners may not need an adapter at all.

---

## 8. Execution lifecycle

When you call a Program:

```python
result = program(support_request="I was charged twice.")
```

it does the following:

1. **Validate inputs against the Signature**
   - required input fields are present
   - values are type-checked or coerced

2. **Resolve execution components**
   - module: the Program's configured module (default: `predict`)
   - runner: must be present (otherwise the Program is not callable)
   - adapter: if the runner family requires one, use the configured
     adapter or resolve `adapter="auto"` from the Signature

3. **Check compatibility**
   - the module must support the runner family
   - otherwise: `ProgramCompatibilityError`

4. **Execute the module**
   - the module owns the strategy
   - it may call the runner once or many times
   - it collects hidden decomposition artifacts and usage

5. **Validate outputs against the Signature**
   - all declared output fields are present
   - output values satisfy the Signature contract

6. **Return a Result**
   - `outputs`: public Signature outputs
   - `trace`: hidden artifacts and operational evidence
   - `usage`: telemetry aggregated across the full run

The **module orchestrates execution**, the **runner performs native
calls**, and the **adapter translates I/O when needed**.

---

## 9. Result is richer than a dict

Programs and Graphs return a `Result`.

```python
@dataclass
class Result:
    outputs: dict[str, Any]
    trace: dict[str, Any] | None = None
    usage: Usage | None = None
```

### Outputs
The public fields promised by the Signature.

### Trace
Execution artifacts: reasoning, tool trajectories, retries, hidden
decomposition outputs, intermediate drafts, branch choices.

### Usage
Runtime telemetry: latency, token usage, cost, retry count.

---

## 10. Hidden decomposition belongs to the module, not the Signature

There are three different things:

### 1. User-desired intermediate artifacts

If the user wants reasoning as part of the task output, it is an output
field on the Signature:

```python
Signature("question -> reasoning, answer")
```

### 2. Strategy-level decomposition

If the module needs internal scaffolding, the module owns it.
`chain_of_thought` already introduces internal reasoning. `react`
already introduces a tool-using trajectory. The user may ask the module
to carry additional hidden artifacts:

```python
program = Program(
    triage,
    chain_of_thought.hint(
        "Compare urgency and issue type before classifying."
    ).via(
        "issue_evidence",
        note="Key facts supporting the team label"
    ),
    OpenAILM("gpt-4o"),
)
```

None of these hidden artifacts are part of the Signature. The optimizer
may change them. They live in `Result.trace` unless explicitly exposed.

### 3. Operational trace

Tool logs, retries, trajectories, and search branches belong in
`Result.trace`.

This keeps intent, strategy, and runtime evidence separate.

---

## 11. Graph is composition

A `Graph` is how many Signatures and Programs become one compound
system.

A Graph is a directed graph of nodes. Each node has:
- an inner Signature
- a bound Program
- symbolic inputs and outputs

### Build phase

The graph exists symbolically. You can inspect wiring, signatures, and
structure.

### Compile phase

Programs are bound to nodes. The system checks executability.

### Run phase

The graph executes in dependency order. Each node returns a `Result`.
The graph returns a combined `Result`.

### Example

```python
from onux import Graph, Input
from onux.layers import Generate, Retrieve, ReAct

question = Input("question")
constraints = Input("constraints", type=list[str])

sources = Retrieve(top_k=10)(question)
notes = ReAct("notes", tools=[search_tool])([question, constraints, sources])
answer = Generate("answer")([question, constraints, notes, sources])

research = Graph(
    inputs=[question, constraints],
    outputs=answer,
    name="research_pipeline",
)
```

A Graph can be serialized, evaluated, optimized, nested inside larger
Graphs, and shared as a reusable subgraph.

---

## 12. Evaluation is core, not optional decoration

### Objective on Signature

The `Signature.objective(...)` defines success:
- rubric criteria
- metric callables
- weights

### Evaluation on Program

Because a Program knows its Signature, it can evaluate its own results:

```python
result = program(support_request="I was charged twice.")
score = program.evaluate(result)
```

### Judges

Rubric scoring uses a pluggable judge interface:

```python
judge = Judge(runner=OpenAILM("gpt-4o"))
score = program.evaluate(result, judge=judge)
```

A Judge may be an LM-based rubric evaluator, a deterministic evaluator,
a human label interface, or a custom domain scorer.

Without evaluation, optimization is just aspiration. Evaluation is core.

---

## 13. Optimization and discovery are native goals

The optimizer takes a Program and searches its axes:

```python
base = Program(triage, chain_of_thought, OpenAILM("gpt-4o"))
best = optimizer.search(base)
best.save("triage-v3")
```

The optimizer reads the Program's Signature (the fixed intent), reads
the Program's axes (the search space), explores checkpoints (forks with
different axis values), evaluates them, and returns the best checkpoint.

### What the optimizer searches

- module choice and module axes (hint, via, tools, ...)
- runner choice and runner axes (model, temperature, ...)
- adapter choice

### What the optimizer holds fixed

- the Signature: input fields, output fields, types, examples, objective

### The climb up and down the hill

The same Signature can move across execution regimes:
- from foundation LM to fine-tuned LM
- from fine-tuned LM to classical ML
- from classical ML to deterministic code
- or back up when flexibility is needed

The Signature stays fixed. The Programs change. The objective judges
improvement.

---

## 14. Sharing and serialization

### Signature

Portable, lossless, JSON-like state: fields, types, examples, objective.

### Preset

Serializable strategy template: module name and config, runner type and
config.

### Program / Checkpoint

Serializable bound execution unit: Signature reference, module config,
runner config, all axis values. A checkpoint is a fully-pinned Program.
The best checkpoint is what you save to disk and run at scale.

```python
# Save
best.save("triage-v3")

# Load
production = Program.load("triage-v3")
result = production(support_request="...")
```

### Graph

Serializable graph structure: nodes, edges, symbol names, per-node
Program bindings.

---

## 15. Progressive disclosure

### Ring 1: one task

```python
sig = Signature("question -> answer")
program = Program(sig, predict, OpenAILM("gpt-4o"))
result = program(question="What is 2+2?")
```

### Ring 2: sharper intent

```python
sig = (
    Signature("question -> answer, confidence")
    .type(confidence=float)
    .examples([...])
    .objective("Score correctness and calibration from 0 to 1.")
)
```

### Ring 3: better execution

```python
program = Program(
    sig,
    chain_of_thought.hint("Think step by step."),
    OpenAILM("gpt-4o", temperature=0.0),
)
```

### Ring 4: fork and compare checkpoints

```python
c1 = program.hint("Think step by step.").temperature(0.0)
c2 = program.hint("Work backwards from the answer.").temperature(0.3)

for c in [c1, c2]:
    result = c(question="What is 17 * 24?")
    print(c.evaluate(result))
```

### Ring 5: optimize automatically

```python
best = optimizer.search(program)
best.save("qa-v1")
```

### Ring 6: compose a system

```python
graph = Graph(...)
```

Each ring adds power without invalidating the ring before it.

---

## 16. Public backbone

A user should leave the first page remembering four nouns:

### Signature

What you want. Pure intent. The invariant the optimizer holds fixed.

### Preset

A reusable execution strategy. Module + runner + configured axes. Not
yet bound to a task.

### Program

How one task runs. A Preset bound to a Signature. Callable, evaluable,
optimizable. Its checkpoints (fully-pinned forks) are what you save and
deploy.

### Graph

How many tasks compose. A directed graph of Programs.

Supporting concepts:
- **Module** — strategy pattern (predict, chain_of_thought, react, ...)
- **Runner** — execution engine (OpenAILM, SklearnRunner, ...)
- **Adapter** — I/O bridge between Signature fields and runner format
- **Result** — what execution returns (outputs + trace + usage)
- **Judge** — how rubrics are scored
- **Optimizer** — how Programs are searched
- **Checkpoint** — a fully-pinned Program, saved and deployed

---

## 17. The full lifecycle

```python
from onux import Signature, Preset, Program
from onux.modules import chain_of_thought
from onux.runners import OpenAILM

# 1. Define intent
triage = (
    Signature("support_request -> severity, team, customer_visible")
    .type(severity=Severity, team=Team, customer_visible=bool)
    .examples([...])
    .objective("routing accuracy rubric...", valid_output, (3.0, 1.0))
)

# 2. Create a reusable strategy
strategy = Preset(chain_of_thought, OpenAILM("gpt-4o"))

# 3. Bind to the task
program = strategy.bind(triage)
# or: program = Program(triage, chain_of_thought, OpenAILM("gpt-4o"))

# 4. Fork checkpoints
c1 = program.hint("Determine team first.").temperature(0.0)
c2 = program.hint("Separate issue type from urgency.").temperature(0.3)

# 5. Evaluate
for c in [c1, c2]:
    result = c(support_request="I was charged twice.")
    print(c.evaluate(result))

# 6. Or let the optimizer search
best = optimizer.search(program)

# 7. Save the best checkpoint
best.save("triage-v3")

# 8. Run at scale
production = Program.load("triage-v3")
result = production(support_request="New request here...")
```

That is the honest API:
- **Signature** is the intent.
- **Preset** is the reusable strategy.
- **Program** is the executable candidate.
- **Checkpoint** is the best version, saved and deployed.
- **Evaluation** decides what is better.
- **Optimization** searches the space.
