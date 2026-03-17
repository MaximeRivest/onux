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

It binds together:
- a **module** — the strategy pattern
- a **runner** — the execution engine
- an **adapter** — the I/O bridge
- module configuration
- runner configuration
- any strategy-level decomposition or tools

### Core shape

```python
program = Program(
    module=chain_of_thought,
    runner=OpenAILM("gpt-4o", temperature=0.0),
    adapter="auto",
    hint="Read carefully and reason step by step.",
    via=[("reasoning", {"note": "Internal analysis"})],
)

result = program(triage, {"support_request": "I was charged twice."})
```

### Why Program exists as a first-class object
`functools.partial` is not enough.
A Program earns its place because it is:
- executable
- serializable
- introspectable
- searchable
- checkpointable
- comparable

The optimizer produces Programs.
Humans author Programs.
Models are built from Programs.

### One clean story
A user should be able to say:

> I write a **Signature** for what I want.
> I write a **Program** for how to get it.
> I run the Program.

That is the basic unit of Onux.

---

## 4. Module is the strategy pattern

A `Module` is not the concrete executable object.
It is the general execution pattern.

Examples:
- `predict`
- `chain_of_thought`
- `react`
- `refine`
- `ensemble`
- `fallback`
- `pipe`
- domain-specific modules like tiling, calibration, or cascades

A Module becomes executable when bound inside a Program.

### Module contract
A module is a callable with a minimal, explicit contract:

```python
def module(sig, inputs, *, runner, adapter, **config) -> Result:
    ...
```

Where:
- `sig` is the pure intent Signature
- `inputs` is a mapping of input values
- `runner` is the execution engine
- `adapter` bridges Signature fields to runner-native I/O
- `config` carries strategy parameters like `hint`, `via`, `tools`, `max_iter`, etc.

### Modules are functions first
Onux should prefer plain callables over base classes.
A module may later grow metadata like `.axes()`, but its conceptual core
is a callable strategy pattern.

### Strategy examples

#### Predict
One call, no extra control flow.

#### Chain of thought
Add internal reasoning scaffolding, then call the runner.

#### ReAct
Maintain a reasoning/tool loop around the runner.

#### Refine
Validate outputs and retry if necessary.

#### Ensemble
Run several Programs and aggregate them.

The important thing is that a Module defines **execution shape**, not
intent.

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
If the Program wants to introduce internal scaffolding, it does so with
strategy configuration:

```python
Program(
    module=chain_of_thought,
    runner=OpenAILM("gpt-4o"),
    via=[("reasoning", {"note": "Internal reasoning"})],
)
```

These are not part of the Signature.
The optimizer may change them.
They normally live in `Result.trace`.

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
program = Program(module=predict, runner=OpenAILM("gpt-4o"))
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
    module=chain_of_thought,
    runner=OpenAILM("gpt-4o", temperature=0.0),
    hint="Reason carefully before answering.",
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
