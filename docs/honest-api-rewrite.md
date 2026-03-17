# Honest API: intent, execution, composition, optimization

This is the document `docs/honest-api.md` should have been.

It is not a philosophy essay and not a speculative kitchen sink. It is a
clear architectural statement of what Onux is for and how its main
objects fit together.

Onux is for building, specifying, sharing, evaluating, extending, and
automatically optimizing **compound AI systems**.

That includes systems built from:
- foundation LLMs
- fine-tuned LLMs
- deep learning models
- classical ML models
- deterministic code
- regexes and boolean rules
- feature engineering pipelines
- retrieval systems
- search tools
- databases and algorithms
- and any decomposition of a larger system into components run by any of the above

The core idea is simple:

> A compound AI system should have a clean description of **what it is
> trying to do**, a clean description of **how each part is executed**,
> and a clean description of **how the parts are wired together**.

That gives us three primary concepts:
- **Signature** — the intent
- **Program** — the execution of one task
- **Model** — the composition of many tasks

Everything else supports those three.

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
- prompt wording
- hidden decomposition fields
- tool use
- retry policies
- ensembling strategies
- search plans
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

If reasoning is only internal scaffolding, it belongs to execution (see below), not
Signature. Generally, if you don't plan to evaluate or optimize over this output, it should not be in the Signature.

---

## 3. Program is how one task runs

If Signature is the **what**, Program is the **how**.

A `Program` is the concrete, executable, serializable, optimizable unit
for one Signature.

### Core shape

A Program composes a configured module and a configured runner:

```python
program = Program(
    chain_of_thought.hint(
        "Distinguish billing issues from product bugs or feature requests. "
        "Use urgency only to choose severity, not team."
    ),
    OpenAILM("gpt-4o", temperature=0.0),
)

result = program(triage, {"support_request": "I was charged twice."})
```

That is it. The constructor takes two arguments:
- a **module** (possibly configured with builder methods)
- a **runner** (possibly configured with builder methods or constructor
  args)

### Every component is its own builder

There are three kinds of configuration in an execution strategy. Each
belongs to the object it configures:

**Module config** — the strategy pattern's own parameters. Different
module types expose different builder methods:

```python
# chain_of_thought knows about: hint, and optionally extra hidden decomposition
# It already owns its internal reasoning decomposition.
chain_of_thought.hint(
    "Extract every monetary amount on the receipt, identify subtotal, tax, "
    "and total, then compute before_tax and after_tax exactly. Do not invent values."
).via(
    "number_list",
    List[float],
    note="All numeric values visible on the receipt, preserving currency symbols and nearby labels"
)

# Another meaningful decomposition: separate evidence gathering from the final answer.
chain_of_thought.hint(
    "Decide whether the support issue belongs to billing, bug, feature, or account before assigning severity."
).via(
    "issue_category_evidence",
    note="Short evidence for the most likely team label"
)

# react knows about: tools, max_iter, hint, and can expose extra working state
react.tools([search_docs, lookup_pricing]).max_iter(5).hint(
    "Check the docs for the exact API version first, then answer only from retrieved facts."
).via(
    "retrieved_facts",
    note="Facts gathered from tools that may be cited in the final answer"
)

# refine knows about: check, max_retries
refine.check(sql_is_safe).max_retries(3)

# predict knows about: hint and prompt-side field notes
predict.hint("Return only the final label and confidence.") \
       .field_note("confidence", "Probability from 0 to 1")
```

**Runner config** — the execution engine's own parameters. Different
runner types expose different builder methods or constructor args:

```python
# LM runners know about: model, temperature, max_tokens, adapter, ...
OpenAILM("gpt-4o", temperature=0.0)
OpenAILM("gpt-4o").temperature(0.0).max_tokens(4096)
AnthropicLM("claude-3-5-sonnet", temperature=0.3)

# LM runners also know about adapter — it is their I/O bridge
OpenAILM("gpt-4o").adapter(XMLAdapter())

# Vision runners know about: weights, conf_thresh, device, ...
YOLORunner("yolov8x.pt").conf_thresh(0.25).device("cuda")

# ML runners know about: model_path, device, batch_size, ...
SklearnRunner("classifier.joblib")

# Code runners know about: timeout, sandbox, ...
PythonRunner(my_function)
```

**Adapter is a runner concern**, not a Program concern. An LM runner
needs an adapter to bridge Signature fields to messages. A sklearn
runner does not need an adapter at all — it already speaks in dicts. So
`.adapter()` is a method on LM runners, not on Program:

```python
# Adapter configured on the runner
OpenAILM("gpt-4o").adapter(XMLAdapter())
OpenAILM("gpt-4o").adapter(StructuredOutputAdapter())

# Default: auto-generated from signature fields and types
OpenAILM("gpt-4o")  # adapter="auto" is the default

# Non-LM runners have no adapter concept
SklearnRunner("model.joblib")  # no .adapter() method exists
```

### Why this is right

Each object only exposes the methods that make sense for its type:

| You type...                   | IDE shows...                    |
|-------------------------------|---------------------------------|
| `chain_of_thought.`           | `hint`, `via` |
| `react.`                      | `hint`, `via`, `tools`, `max_iter` |
| `refine.`                     | `check`, `max_retries`          |
| `predict.`                    | `hint`, `field_note`            |
| `OpenAILM("gpt-4o").`        | `temperature`, `max_tokens`, `adapter`, ... |
| `YOLORunner("yolov8x.pt").`  | `conf_thresh`, `iou_thresh`, `device`, ... |
| `SklearnRunner("model.joblib").` | (nothing to configure)       |

No method appears where it does not belong. The type system enforces
ownership. The IDE guides you.

### Program is thin

Because config lives on the components, Program itself is thin — it
just composes a module and a runner:

```python
Program(module, runner)
```

Program adds a small number of its own methods for things that are
genuinely about the *binding* (not about any one component):

- `.expose(*names)` — surface trace fields as graph symbols
- `.module(m)` — swap the module (returns new Program)
- `.runner(r)` — swap the runner (returns new Program)

That is all. Everything else lives where it belongs.

### Composition and forking

Because modules and runners are immutable builders, you can fork and
compose freely:

```python
# Shared strategy, different runners
cot = chain_of_thought.hint(
    "Decide team from the user's underlying issue, then decide severity from urgency and business impact."
)
fast = Program(cot, OpenAILM("gpt-4o-mini"))
accurate = Program(cot, OpenAILM("o1"))

# Shared runner, different strategies
lm = OpenAILM("gpt-4o", temperature=0.0)
simple = Program(predict, lm)
reasoned = Program(
    chain_of_thought.hint(
        "Separate issue type, urgency, and customer visibility before emitting the final fields."
    ),
    lm,
)
agent = Program(react.tools([search_docs, calculator]), lm)

# Progressive refinement of a module
v1 = predict.hint("Return only the final classification fields.")
v2 = v1.field_note("confidence", "Probability from 0 to 1")
v3 = v2.hint("Return only the final fields. Do not explain your reasoning.")

# Compose into Programs
prog_v1 = Program(v1, lm)
prog_v3 = Program(v3, lm)
```

This mirrors how Signature authoring works:

```python
# Intent builder
sig = Signature("question -> answer").type(answer=str).examples([...])

# Strategy builder
mod = chain_of_thought.hint(
    "Work through the arithmetic before giving the final answer."
)
run = OpenAILM("gpt-4o", temperature=0.0)

# Compose
program = Program(mod, run)
result = program(sig, {"question": "What is 17 * 24?"})
```

Three immutable builders — Signature, module, runner — each configured
with its own type-aware methods. Program composes two of them.

### Partial Programs

A Program without a runner is a **partial Program** — a reusable
binding of module-to-intent that becomes executable when a runner is
bound:

```python
strategy = Program(
    chain_of_thought.hint(
        "Identify the issue category first, then map urgency to severity."
    )
)

# Bind to different runners
for_triage = strategy.runner(OpenAILM("gpt-4o", temperature=0.0))
for_research = strategy.runner(AnthropicLM("claude-3-5-sonnet"))
```

A Program without a module defaults to `predict`.

### Why Program exists as a first-class object

`functools.partial` is not enough.
A Program earns its place because it is:
- **thin** — just module + runner, not a bag of unrelated config
- **executable** — `program(sig, inputs) -> Result`
- **serializable** — `dump_state()` / `load_state()` for checkpoints
- **introspectable** — `.axes()` merges module and runner axes
- **searchable** — the optimizer produces and compares Programs
- **composable** — swap module or runner, fork from a base

The optimizer produces Programs.
Humans author Programs.
Models are built from Programs.

### One clean story

> I write a **Signature** for what I want.
> I configure a **module** and a **runner** for how to get it.
> I compose them into a **Program** and run it.

That is the basic unit of Onux.

---

## 4. Module is the strategy pattern

A Module is the general execution pattern, and it is its own immutable
builder.

### Module types

Each module type is a builder that knows its own configurable axes.
The module owns its default internal decomposition; users configure the
module's behavior, not its implementation details.

```python
# LM strategy modules
predict                                           # single call
predict.hint("Return only the label and confidence.")
predict.field_note("confidence", "Probability from 0 to 1")

chain_of_thought                                  # reasoning is built in
chain_of_thought.hint(
    "Work through the tax calculation carefully and only then produce the final amounts."
)

react                                             # tool loop is built in
react.tools([search_docs, lookup_pricing])
react.tools([search_docs]).max_iter(5)
react.tools([search_docs]).hint(
    "Look up the exact API behavior in the docs before answering."
)

refine                                            # validation loop is built in
refine.check(my_validator)
refine.check(my_validator).max_retries(3)

ensemble                                          # multi-program vote
ensemble.n(5)
```

Each builder method returns a new immutable module spec. Only the
methods that make sense for that module type are available.

### What a module is under the hood

A module has two faces:

1. **A builder / spec** — the immutable configuration object that users
   construct and that Programs hold. It declares its axes for the
   optimizer and serializer.

2. **A callable** — the execution implementation. When a Program runs,
   it calls the module's execution function with the configured
   parameters:

```python
def chain_of_thought_fn(sig, inputs, *, runner, adapter, hint=None, via=None) -> Result:
    ...
```

Users interact with face 1 (the builder). The framework uses face 2
(the callable) at execution time. The builder carries the config; the
callable receives it.

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

## 5. Runners are broad by design, not fake-universal

Onux must support a real climb up and down the hill of execution:
- from a general foundation model
- to a specialized fine-tuned model
- to deep learning
- to classical ML
- to deterministic code, feature engineering, regexes, and algorithms

That means Onux cannot be only an LLM wrapper.
But it also should not pretend every domain has the exact same runtime
shape.

### The correct abstraction
There is a tiny universal runner contract for the framework, and richer
domain-specific protocols for execution.

### Universal runner responsibilities
Every runner should be able to describe:
- its axes / tunable parameters
- its current config
- how it is serialized and reconstructed

For example:

```python
class Runner(Protocol):
    def axes(self) -> dict[str, Axis]: ...
    def config(self) -> dict[str, Any]: ...
```

That is enough for introspection, sharing, and optimization.

### Domain-specific execution protocols
Actual execution differs by domain.
That is normal.

```python
class LMRunner(Runner, Protocol):
    def complete(self, request, **kwargs) -> LMResponse: ...

class VisionRunner(Runner, Protocol):
    def predict(self, image, **kwargs) -> VisionResult: ...

class MLRunner(Runner, Protocol):
    def predict(self, features, **kwargs) -> MLResult: ...

class DeterministicRunner(Runner, Protocol):
    def run(self, inputs, **kwargs) -> Any: ...
```

Modules know what runner family they need.
The optimizer and serializer care about the universal part.

### Why this matters
This avoids two bad designs:
- pretending everything is `run(input) -> output`
- pretending LLMs are the only serious case

Onux should be broad by design, while remaining honest that some domains
will mature sooner than others.

---

## 6. Adapters are real, but mostly advanced

An `Adapter` turns Signature fields and runtime values into the native
input/output format a runner expects.

For an LM, that may mean:
- rendering messages
- defining output schema expectations
- parsing model responses into structured outputs

For a vision system, that may mean:
- mapping Signature fields to image tensors
- formatting annotations
- decoding model output into boxes/masks/labels

For an sklearn pipeline, it may mean:
- feature extraction
- label mapping
- a near-passthrough bridge

### Adapter contract

```python
class Adapter(Protocol):
    def encode(self, sig: Signature, inputs: dict[str, Any]) -> Any: ...
    def decode(self, sig: Signature, raw_output: Any) -> dict[str, Any]: ...
```

The exact `Any` is domain-specific.
That is fine.

### User experience
Most users should not have to think about adapters immediately.
Programs default to:

```python
adapter="auto"
```

and the framework generates or chooses a sensible adapter.

Advanced users can swap or author adapters explicitly.

### Important non-goal
The base design should not commit too early to a giant canonical
cross-provider intermediate language unless and until real implementation
pressure proves it necessary.

Adapters are a public extension point, but not the emotional center of
the library.

---

## 7. Result is richer than a dict

A bare dict is too weak.
Compound systems need outputs, trace, and telemetry.

So Programs and Models return a `Result`.

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
Execution artifacts such as:
- reasoning
- tool trajectories
- retries
- hidden decomposition outputs
- intermediate drafts
- branch choices

### Usage
Runtime telemetry such as:
- latency
- token usage
- cost
- retry count
- hardware/runtime metadata

This lets Onux support:
- debugging
- introspection
- evaluation
- cost-aware optimization
- sharing and reproducibility

without polluting the actual task outputs.

---

## 8. Hidden decomposition belongs to Program, not Signature

One of the original document’s biggest confusions was hidden fields.
That should be resolved cleanly.

There are three different things:

### 1. User-desired intermediate artifacts
If the user wants reasoning, evidence, or a draft as part of the task,
those are just output fields on the Signature.

```python
Signature("question -> reasoning, answer")
```

### 2. Strategy-level decomposition
If the module needs internal scaffolding, the module owns it. For
example, `chain_of_thought` already introduces internal reasoning, and
`react` already introduces a tool-using trajectory. But the user may
still ask the module to carry additional hidden working artifacts when
that decomposition is useful:

```python
Program(
    chain_of_thought.hint(
        "Compare the line items, subtotal, tax, and total before producing the answer."
    ).via(
        "number_list",
        note="All numeric values extracted from the receipt with nearby labels"
    ),
    OpenAILM("gpt-4o"),
)

Program(
    react.tools([search_docs]).hint(
        "Answer from the docs only after you have gathered the exact facts you need."
    ).via(
        "retrieved_facts",
        note="Facts collected from tool calls that support the final answer"
    ),
    OpenAILM("gpt-4o"),
)
```

The important distinction is:
- the built-in hidden structure of a module belongs to the module
- extra decomposition like `number_list` or `retrieved_facts` is a
  strategy choice layered on top

None of these hidden artifacts are part of the Signature.
The optimizer may change them.
They normally live in `Result.trace` unless explicitly exposed.

### 3. Operational trace
Things like tool logs, retries, trajectories, search branches, and
intermediate artifacts belong in `Result.trace`.

This keeps intent, strategy, and runtime evidence separate.

---

## 9. Model is composition

A `Model` is how many Signatures and Programs become one compound system.

A Model is a graph of nodes.
Each node has:
- an inner Signature
- a bound Program or a way to resolve one
- symbolic inputs and outputs

### Model is both symbolic and executable
That dual nature is important.

#### Build phase
The graph exists symbolically.
You can inspect wiring, signatures, and structure.

#### Compile phase
Programs are bound to nodes.
The system checks executability.

#### Run phase
The graph executes and returns a Result.

### Example

```python
from onux import Input, Model
from onux.layers import Generate, Retrieve, ReAct

question = Input("question")
constraints = Input("constraints", type=list[str])

sources = Retrieve(top_k=10)(question)
notes = ReAct("notes", tools=[search_tool])([question, constraints, sources])
answer = Generate("answer")([question, constraints, notes, sources])

research = Model(
    inputs=[question, constraints],
    outputs=answer,
    name="research_pipeline",
)
```

### Model as program-of-programs
A Model is not separate from the execution story.
It is a higher-order executable object built from Programs.

That means a Model can also be:
- serialized
- evaluated
- optimized
- nested inside larger Models
- shared as a reusable subgraph

This is how Onux handles decomposition.

---

## 10. Evaluation is core, not optional decoration

Onux is not just for running systems.
It is for improving them.
That means evaluation must be first-class.

The `Signature.objective(...)` belongs on Signature because it defines
success.
But objective definition alone is not enough.
Onux also needs a real execution model for evaluation.

### Objective components
A Signature objective may contain:
- rubric criteria
- metric callables
- weights

### Runtime-aware metrics
Metrics may use:
- inputs
- outputs
- public intermediate outputs
- runtime telemetry through `runtime`

### Judges
Rubric scoring should have a pluggable judge interface.

```python
judge = Judge(runner=OpenAILM("gpt-4o"))
score = triage.evaluate(values=prediction, judge=judge, runtime=result.usage)
```

A Judge may be:
- an LM-based rubric evaluator
- a deterministic evaluator
- a human label interface
- a custom domain scorer

Without evaluation, optimization is just aspiration.
So evaluation is part of the core architecture, not a future afterthought.

---

## 11. Optimization and discovery are native goals

Onux is not only for hand-authoring systems.
It is for searching over them.

That means the architecture should support:
- trying multiple runners for the same Signature
- upgrading from a foundation model to a fine-tuned one
- replacing an LLM step with deterministic code when the task hardens
- replacing a prompt-based classifier with classical ML
- replacing classical ML with deep learning when scale demands it
- decomposing one task into several smaller tasks
- recomposing many tasks into one reusable Model

### The climb up and down the hill
This is one of the central product goals.
A user — or another model — should be able to climb the hill from:
- foundation LM
- to prompt-specialized foundation LM
- to fine-tuned LM
- to deep learning model
- to classical ML pipeline
- to feature engineering + algorithmic logic
- to boolean / regex / deterministic core

or go the other direction when flexibility is needed.

The Signature stays fixed.
The Programs change.
The Model may change.
The objective judges improvement.

That is what makes the system evolvable.

### Search unit
The optimizer searches over Programs and Models.
It does not search over Signatures.
Signatures are the target.
Programs and Models are candidates.

### Axes
Runners, adapters, and modules expose axes.
Programs materialize concrete values.
The optimizer uses those axes to search, mutate, and compare.

Pin/seed/bound semantics belong to the optimizer layer, not to the core
mental model of execution.

---

## 12. Sharing and serialization matter

Onux is meant to support building and sharing systems, not just running
them in one process.

That means the main objects must serialize clearly.

### Signature
Portable, lossless, JSON-like state:
- fields
- types
- examples
- objective rubrics
- metric references by qualified name if needed

### Program
Serializable as structured config:
- module name and config
- runner type and config
- adapter type and config
- strategy configuration

### Model
Serializable graph structure:
- nodes
- edges
- layer types
- symbol names
- node-level program bindings

### Checkpoint
A fully pinned, evaluated Program or Model plus scores and metadata.

This is what makes systems:
- reproducible
- auditable
- shareable
- optimizable across sessions
- loadable by humans or other models

---

## 13. Progressive disclosure: the right teaching order

A good API lets both a human and a language model discover complexity
incrementally.

### Ring 1: one task

```python
sig = Signature("question -> answer")
program = Program(predict, OpenAILM("gpt-4o"))
result = program(sig, {"question": "What is 2+2?"})
```

### Ring 2: make intent sharper

```python
sig = (
    Signature("question -> answer, confidence")
    .type(confidence=float)
    .examples([...])
    .objective("Score correctness and calibration from 0 to 1.")
)
```

### Ring 3: improve execution

```python
program = Program(
    chain_of_thought.hint(
        "Classify the request by separating issue type from urgency: team depends on issue type; severity depends on urgency."
    ),
    OpenAILM("gpt-4o", temperature=0.0),
)
```

### Ring 4: decompose the system

```python
model = Model(...)
```

### Ring 5: optimize automatically

```python
best = optimizer.search(intent=sig, seed=program)
```

Each ring adds power without invalidating the ring before it.
That is the right kind of hill-climbing interface.

---

## 14. Public backbone

A user should leave the first page remembering three nouns:

### Signature
What you want.

### Program
How one task runs.

### Model
How many tasks compose.

Supporting concepts:
- **Module** — strategy pattern used by Program
- **Runner** — execution engine used by Program
- **Adapter** — I/O bridge used by Program
- **Result** — what execution returns
- **Judge** — how rubrics are scored
- **Optimizer** — how Programs and Models are searched
- **Checkpoint** — what gets saved and shared

That is enough structure to be powerful without being muddy.

---

## 15. What Onux is ultimately for

Onux is for the full lifecycle of compound AI systems:
- **building** them
- **specifying** them precisely
- **sharing** them as portable artifacts
- **extending** them with custom modules, runners, and graphs
- **evaluating** them against explicit objectives
- **optimizing** them automatically
- **discovering** new decompositions and implementations

The same task should be able to move across execution regimes without
rewriting its intent:
- from LLM to fine-tuned LLM
- from fine-tuned LLM to ML
- from ML to deterministic code
- from one monolithic step to many explicit components
- from many components to a reusable higher-level Model

That is the honest API:
- **Signature** is the intent.
- **Program** is the candidate execution.
- **Model** is the composition.
- **Evaluation** decides what is better.
- **Optimization** searches the space.

That is the right foundation for compound AI systems.
