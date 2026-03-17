# Concrete design decisions to make for Onux

I'm interpreting your request as: **do option 2** from the previous message - a concrete list of design decisions to make next.

This document turns the critique of `docs/honest-api.md` into an actionable architecture checklist.

---

## Decision 1: What is a `Signature`, exactly?

### The question
Is `Signature`:
1. a **pure intent object**, or
2. a **pragmatic authoring object** that mixes intent with execution annotations?

### Option A - Pure intent
`Signature` contains only:
- inputs
- outputs
- types
- examples
- objective
- maybe reference artifacts

`hint`, `note`, and `via` move elsewhere.

#### Pros
- clean conceptual model
- easier optimizer story
- easier equality/serialization semantics
- clearer boundary between task and strategy

#### Cons
- less ergonomic authoring
- requires one or more new objects for prompt/execution annotations
- bigger migration from the current code/doc style

### Option B - Mixed authoring object
`Signature` stays as the main builder surface and may include:
- semantic fields
- examples/objective
- execution-ish annotations like `hint`, `note`, `via`

#### Pros
- simpler user story
- builder pattern remains elegant
- closer to current docs and code direction

#### Cons
- ontology is less clean
- optimizer boundary gets blurrier
- harder to say what counts as "same intent"

### Recommendation
If Onux is primarily an **optimizer-native system design library**, choose **Option A**.
If Onux is primarily a **human-authored library first**, choose **Option B**.

My default recommendation: **Option B in v1, documented honestly**, unless optimization is the central product thesis.

> ### Answer: Option A - Pure intent
>
> Signature is the behavioral target the optimizer holds fixed. It
> contains **only** what defines the task and what defines success:
>
> - **Input fields** and **output fields** (with names)
> - **Types** (constraining the behavioral contract)
> - **Examples** (anchoring the target behavior)
> - **Objective** (rubrics and metrics defining success)
>
> `.hint()`, `.note()`, and `.via()` **move off Signature** and onto
> Program. They are execution strategy - the optimizer test says so, and
> we follow through.
>
> **What this means concretely:**
>
> - The formula is inputs and outputs only: `"question -> answer"`,
>   `"question, context -> answer, confidence"`. No hidden arrow. The
>   three-segment formula `"question -> reasoning -> answer"` is no
>   longer a Signature-level construct.
> - `.type()`, `.examples()`, `.objective()`, `.add()`, `.remove()`
>   stay on Signature. `.hint()`, `.note()`, `.via()` move to Program.
> - Field *descriptions* that clarify what a field means (e.g., "star
>   rating from 1 to 5") are a borderline case. The clean answer: push
>   users toward the type system. `Literal["1","2","3","4","5"]` or
>   `Annotated[float, Ge(1), Le(5)]` express constraints precisely. If
>   prose descriptions are needed later, they can be added as a separate
>   concept (`field_description`) distinct from prompt-engineering notes.
>   For v1, types carry the semantic load.
> - If a user wants "reasoning" as a *public* artifact (for auditing,
>   for downstream use), it goes in the output fields:
>   `"question -> reasoning, answer"`. It is intent - the user wants
>   that output. If "reasoning" is internal scaffolding that only exists
>   to improve answer quality, that is strategy and belongs on Program
>   as a `via` configuration.
>
> **Why pure, not pragmatic:**
>
> Onux is an optimizer-native system design library. The optimizer is
> not a bolt-on; it is the architectural backbone. If Signature mixes
> intent with strategy, the optimizer cannot freely search the strategy
> space without risking collision with user-specified annotations. Purity
> here is not aesthetic - it is mechanical. The optimizer needs a fixed
> target, and Signature is that target.
>
> **The cost:**
>
> Authoring becomes slightly less fluid. You cannot write a single
> builder chain that goes from task definition to prompt engineering.
> Instead you write a Signature (what), then a Program (how). This is a
> real ergonomic cost, and it is worth paying for the clarity it buys in
> every downstream system: optimization, serialization, equality
> semantics, and conceptual teachability.
>
> **Migration from current code:**
>
> The existing `signatures.py` must be refactored. `.hint()`, `.note()`,
> `.via()`, the `_hint` slot, the hidden-field parsing in
> `_parse_formula`, and the `"hidden"` role in `Field` all move out.
> `Field.role` becomes `Literal["input", "output"]` only. The hidden
> role reappears on Program's execution signature, not on the user's
> intent Signature.

---

## Decision 2: Is `Program` a first-class public object in v1?

### The question
Should users directly construct and reason about `Program`, or is it an optimizer/runtime abstraction that can remain secondary for now?

### Option A - `Program` is core public API
Users write things like:
```python
Program(module=chain_of_thought, runner=OpenAILM("gpt-4o"), ...)
```

#### Pros
- execution model becomes explicit
- optimizer has a clear unit of search/checkpointing
- unifies module + runner + adapter + config

#### Cons
- introduces a second center of gravity besides `Signature`/`Model`
- increases conceptual load
- may make the API feel more systems-heavy than necessary

### Option B - `Program` is deferred or secondary
Users mainly work with:
- `Signature`
- module functions
- `Layer`
- `Model`

Optimizer/runtime may use programs internally.

#### Pros
- simpler user-facing API
- easier to teach
- fewer top-level abstractions in v1

#### Cons
- execution/search architecture may feel less explicit
- later introduction of `Program` may be a breaking conceptual shift

### Recommendation
Decide this early. Don't let `Program` be "half-core".

My recommendation:
- **If optimizer is not shipping early, keep `Program` secondary in v1**.
- **If optimizer is central, make `Program` explicitly first-class now**.

> ### Answer: Option A — Program is first-class, composing typed builders
>
> Program is the second noun users learn, right after Signature. It is
> the concrete, executable, serializable, optimizable binding of
> strategy to intent.
>
> **The key insight: every component is its own immutable builder.**
>
> Module config lives on the module. Runner config lives on the runner.
> Adapter config lives on the runner (adapters are a runner concern).
> Program just composes them:
>
> ```python
> program = Program(
>     chain_of_thought.hint("Read carefully.").via("reasoning"),
>     OpenAILM("gpt-4o", temperature=0.0),
> )
> result = program(signature, inputs)
> ```
>
> **Each component type exposes only its own methods:**
>
> - `chain_of_thought.` → `hint`, `via`
> - `react.` → `hint`, `via`, `tools`, `max_iter`
> - `refine.` → `check`, `max_retries`
> - `OpenAILM("gpt-4o").` → `temperature`, `max_tokens`, `adapter`, …
> - `YOLORunner("yolov8x.pt").` → `conf_thresh`, `device`, …
> - `SklearnRunner("model.joblib").` → (nothing)
>
> No method appears where it does not belong. The IDE guides you.
>
> **Program itself is thin:**
>
> ```python
> Program(module, runner)
> ```
>
> Program's own methods are only for the binding:
> - `.module(m)` — swap the module (returns new Program)
> - `.runner(r)` — swap the runner (returns new Program)
> - `.expose(*names)` — surface trace fields as graph symbols
>
> **Composition and forking:**
>
> ```python
> # Shared strategy, different runners
> cot = chain_of_thought.hint("Think step by step.").via("reasoning")
> fast = Program(cot, OpenAILM("gpt-4o-mini"))
> accurate = Program(cot, OpenAILM("o1"))
>
> # Shared runner, different strategies
> lm = OpenAILM("gpt-4o", temperature=0.0)
> simple = Program(predict, lm)
> reasoned = Program(chain_of_thought.hint("Think carefully."), lm)
> agent = Program(react.tools([search, calculator]), lm)
>
> # Progressive refinement of a module
> v1 = chain_of_thought.hint("Think step by step.")
> v2 = v1.via("evidence", note="Supporting facts")
> v3 = v2.hint("Cite evidence, then answer.")
> ```
>
> Three immutable builders — Signature, module, runner — each with
> type-aware methods. Program composes two of them.
>
> **Adapter is a runner concern:**
>
> An LM runner needs an adapter. An sklearn runner does not. So
> `.adapter()` is a method on LM runners, not on Program:
>
> ```python
> OpenAILM("gpt-4o").adapter(XMLAdapter())  # LM runner has .adapter()
> SklearnRunner("model.joblib")             # no .adapter() — not needed
> ```
>
> **Partial Programs:**
>
> ```python
> strategy = Program(chain_of_thought.hint("Think carefully.").via("reasoning"))
> for_triage = strategy.runner(OpenAILM("gpt-4o", temperature=0.0))
> for_research = strategy.runner(AnthropicLM("claude-3-5-sonnet"))
> ```
>
> **What Program is:**
>
> - Thin — just module + runner, not a bag of unrelated config
> - Callable: `program(sig, inputs) -> Result`
> - Serializable: `dump_state()` / `load_state()` for checkpoints
> - Introspectable: `.axes()` merges module and runner axes
> - The unit of search for the optimizer
> - The unit of execution for a single task
>
> **A fully-pinned Program is a Checkpoint:**
>
> When every axis has a concrete value, a Program is fully determined.
> The optimizer saves winning Programs as Checkpoints. A Checkpoint is
> just a Program plus evaluation scores and metadata.

---

## Decision 3: What is the one primary execution unit?

### The question
What is the main thing that "runs" in Onux?

Candidates:
- module function
- program
- model
- all of the above equally

### Why this matters
If too many things are independently executable, the API becomes conceptually muddy.
Users won't know what the canonical runtime surface is.

### Recommendation
Pick a hierarchy like this:

### Recommended hierarchy
- **Module** = reusable strategy pattern, not usually run naked by end users
- **Program** = concrete executable binding of strategy + runner/config
- **Model** = symbolic graph that either executes directly or compiles to programs

Or, if you do not want `Program` in v1:
- **Module** = executable function for a single task
- **Model** = executable graph abstraction
- **Program** = future optimizer/runtime layer

But define one primary runtime story.

> ### Answer: Program for one step, Model for many steps
>
> The execution hierarchy is:
>
> | Concept | What it executes | Granularity |
> |---|---|---|
> | **Module** | Nothing by itself | Strategy *pattern*, not directly run |
> | **Program** | One Signature | Single task, fully bound |
> | **Model** | A graph of tasks | Multi-step pipeline |
>
> A **Module** (e.g., `chain_of_thought`) is a strategy pattern - a
> recipe for how to execute a signature. It is not directly executable
> because it has open parameters (which runner? which adapter? what
> hint?). It becomes executable when bound into a Program.
>
> A **Program** is the primary execution unit for a single task. It
> binds a module with a runner, adapter, and configuration. You call it:
> `program(sig, inputs) -> Result`.
>
> A **Model** is the primary execution unit for a multi-step pipeline.
> It is a DAG of nodes, where each node resolves to a Program. You call
> it: `model(inputs) -> Result`. The Model handles data flow between
> nodes.
>
> **Users run Programs and Models. They do not run modules directly.**
>
> This is the one canonical runtime story. If you have one task, you
> need a Signature and a Program. If you have many tasks wired together,
> you need a Model.

---

## Decision 4: What is the relationship between `Model` and `Program`?

### The question
Are they:
1. separate abstractions,
2. one compiled into the other, or
3. one a subtype of the other?

### Viable choices

#### Option A - `Model` is symbolic, `Program` is executable
A `Model` is graph structure.
Execution happens by compiling the graph into one or more programs.

#### Option B - `Model` is itself executable and separate from `Program`
`Model` handles graph execution directly.
`Program` only matters for single-node execution or optimizer work.

#### Option C - `Model` is a kind of program
The graph itself is just a higher-order program.

### Recommendation
My recommendation is **Option A or B**, not C.

C risks collapsing symbolic composition and concrete execution into one overloaded concept.

Most likely best choice:
- **`Model` is symbolic and executable**
- **`Program` is the execution/search object for node-level or whole-model strategies**

But whichever you choose, specify it plainly.

> ### Answer: Option C - Model is a program of programs
>
> A Model is a higher-order Program. It fulfills a Signature (the
> Model's overall input/output contract) by orchestrating a graph of
> inner Programs.
>
> **Concretely:**
>
> - Each node (Layer) in the graph has an inner Signature (its local
>   input/output contract) and is assigned a Program (its local
>   execution strategy).
> - The Model's own Signature is derived from the graph's external
>   inputs and outputs.
> - Execution: the Model runs each node's Program in topological order,
>   piping outputs to downstream inputs.
> - A Model can appear as a node inside a larger Model (sub-model
>   composition). From the outer graph's perspective, it is just another
>   node with a Signature and an execution strategy.
>
> **Why "program of programs" and not separate abstractions:**
>
> The optimizer needs to search over Models the same way it searches
> over Programs. A Model has axes (the combined axes of all its inner
> Programs, plus graph-level choices like which nodes exist). A Model
> can be checkpointed, scored, and compared. If Model and Program are
> fundamentally different kinds of objects, the optimizer needs two
> separate search mechanisms. If a Model *is* a (higher-order) Program,
> the optimizer has one unified search space.
>
> **The dual nature:**
>
> A Model is simultaneously:
>
> 1. **Symbolic** - a graph structure that can be inspected, summarized,
>    and analyzed before any execution happens. This is the Keras
>    analogy: symbolic tensors, layer wiring, `.summary()`.
> 2. **Executable** - when every node has a bound Program, the Model
>    can be called with inputs and produces outputs.
>
> A Model without bound Programs is valid for inspection but not for
> execution. You can print its summary, extract layer signatures, and
> analyze data flow. Binding Programs (explicitly or via defaults) makes
> it executable.
>
> **Data flow:**
>
> Symbols (the named wires between nodes) carry values at runtime.
> The Model matches symbols to signature fields **by name**. When
> `Generate("answer")([question, context])` is called, the symbols
> named `question` and `context` become input fields for that node's
> inner Signature. Naming is the binding mechanism - not position.
>
> **Sub-model composition:**
>
> A Model used inside a larger graph exposes its external Signature.
> Its internal structure is opaque to the outer graph - the outer graph
> sees one node with inputs and outputs, not the sub-model's internal
> nodes. This is the same as Keras's model-in-model pattern.

---

## Decision 5: How much of the optimizer belongs in the base design?

### The question
Is the optimizer:
- a future extension,
- a major subsystem, or
- the hidden backbone of the whole design?

### Why this matters
A lot of the current design is optimizer-shaped:
- intent purity
- axes
- seeds/pins/bounds
- checkpoints
- programs as search candidates

If optimizer is out of scope for a long time, those choices may overcomplicate the base library.

### Recommendation
Pick one of these product/design stances:

#### Stance A - Optimizer-native library
The base architecture is explicitly built around optimization.

#### Stance B - Build systems first, optimize later
Keep optimizer compatibility, but don't let it dominate v1.

My recommendation: **Stance B unless the optimizer is the actual near-term differentiator**.

> ### Answer: Major subsystem, mostly runtime internals
>
> The optimizer is the hidden backbone of the design. It is the *reason*
> Signature is pure intent, the *reason* Program has typed axes, the
> *reason* runners and modules are self-describing. The entire
> architecture is optimizer-native.
>
> But the optimizer is not the first thing users see. It is a major
> subsystem that advanced users and framework developers engage with
> deeply - sometimes. Regular users write Signatures and Programs by
> hand, run them, iterate, and may never touch the optimizer directly.
>
> **What this means for the base design:**
>
> - **Axes are always present.** Every runner, module, and adapter
>   exposes `.axes()`. This is part of their protocol, not an
>   optimizer-only feature. It also serves documentation, debugging,
>   and tooling.
> - **Programs are always serializable and introspectable.** This is
>   part of Program's identity. The optimizer uses it for search and
>   checkpointing. Users use it for logging and reproducibility.
> - **The optimizer API is a separate import.** `from onux.optim import
>   Optimizer` or similar. It is not required for basic execution.
> - **Pin/seed/bound semantics live on the optimizer**, not on Program.
>   A Program is a concrete object. The optimizer wraps Programs in
>   search configurations (what to pin, what to explore, what bounds to
>   apply). Users who don't optimize never see pin/seed/bound.
>
> **The teaching order:**
>
> 1. Signature - what you want
> 2. Program - how to get it
> 3. Model - multi-step composition
> 4. Optimizer - find the best Program automatically (advanced)
>
> **The design order:**
>
> The optimizer's needs are considered at every design decision, but the
> optimizer itself ships after the core execution path works. The
> architecture is built *for* optimization from the start, even if the
> optimizer is not the first thing implemented.

---

## Decision 6: Are adapters public, or mostly runtime internals?

### The question
Should users think in terms of adapters as a major concept, or should adapters mostly be hidden behind runners and modules?

### Option A - Public concept
Users can author, configure, swap, and inspect adapters.

#### Pros
- powerful
- explicit
- good for advanced users and provider portability

#### Cons
- adds abstraction weight
- may expose too much machinery too early

### Option B - Mostly internal in v1
Adapters exist, but most users interact with:
- structured outputs
- built-in renderers/parsers
- runner defaults

#### Pros
- simpler API
- easier onboarding
- less premature commitment to canonical schemas

#### Cons
- advanced control may feel hidden
- harder to explain portability story cleanly

### Recommendation
**Keep adapters real in the architecture, but optional in the user story for v1.**

That means:
- public enough to extend
- not so central that every example needs them

> ### Answer: Mostly runtime internals, with deep access for advanced users
>
> Adapters are real architectural objects - public, documented, and
> extensible. But most users never think about them. The basic story is:
> you pick a runner, the adapter is auto-generated. You only reach for
> an adapter when you need something non-default.
>
> **Three tiers of engagement:**
>
> | User level | Adapter experience |
> |---|---|
> | **Basic** | Never mentioned. Program uses `adapter="auto"`. It works. |
> | **Intermediate** | Swaps to a named adapter: `adapter=XMLAdapter()` or `adapter=StructuredOutputAdapter()`. Understands the choice affects prompt rendering and output parsing. |
> | **Advanced** | Writes custom adapters for non-LM runners, novel prompt formats, or domain-specific I/O. Engages deeply with the encode/decode contract. |
>
> **Auto-generation:**
>
> For LM runners, the default adapter is generated from the Signature's
> field names and types. It produces a prompt that describes the inputs,
> asks for the outputs, and parses the response. This covers 80% of
> cases. Users who want XML tags, JSON schemas, or structured output
> switch adapters explicitly.
>
> **For non-LM runners:**
>
> Non-LM runners typically bring their own adapters or use
> `PassthroughAdapter`. An sklearn runner that already speaks in dicts
> needs no adapter. A vision runner that needs image loading and
> annotation formatting needs a domain-specific adapter. These are
> written by the runner author or the user, not auto-generated.
>
> **Design implication:**
>
> The adapter protocol is part of the public API and is documented. But
> tutorials and quickstarts do not mention adapters until the user needs
> to customize prompt rendering. The `adapter=` kwarg on Program
> defaults to `"auto"` and most examples omit it.

---

## Decision 7: How broad is v1 domain support really?

### The question
Is Onux in v1 truly for:
- LLMs + graphs + prompts + tools + retrieval + ML + vision + audio

or is it really:
- LLM systems first,
- with a runner abstraction intended to generalize later?

### Why this matters
There's a difference between:
- designing for extension, and
- claiming equal maturity across domains

### Recommendation
Be explicit.

Suggested stance:
- **LLM systems are first-class in v1**
- **non-LLM runners are supported by design, but may be less mature initially**

This is not a weakness. It makes the design more believable.

> ### Answer: Broad by design, maturing domain by domain
>
> The architecture is genuinely cross-domain. It must work for LLMs,
> vision, classical ML, audio, code, SQL, and HTTP services - not as a
> future aspiration but as a present design constraint. The abstractions
> (Signature, Program, Module, Runner, Adapter, Model) must not
> special-case LLMs at the expense of other domains.
>
> **Why broad:**
>
> Generality is the guarantee that the abstractions are right. If
> Signature only works for `"question -> answer"` and breaks for
> `"image -> boxes, labels, scores"`, the abstraction is wrong. If
> Runner only works for LLMs and needs a completely different contract
> for vision, the abstraction is wrong. Breadth is the stress test.
>
> **What broad means in practice:**
>
> - The Signature formula, field types, examples, and objective must
>   accommodate non-text I/O (images, tensors, structured data).
> - The Runner contract must be minimal enough that LM runners, vision
>   runners, ML runners, and code runners all implement it naturally,
>   without distortion.
> - The Module contract must accommodate LM-specific patterns
>   (chain_of_thought, react) and domain-specific patterns
>   (tile_and_merge, test_time_augmentation) equally.
> - The Adapter contract must work for prompt rendering (LMs) and for
>   non-prompt I/O bridges (vision, ML) without forcing a
>   canonical-request layer onto domains that don't need it.
>
> **What broad does NOT mean:**
>
> - It does not mean all domains ship equally mature in v1. LLMs will
>   have more built-in runners, adapters, and modules. Vision and ML
>   will have the extension points and example implementations but
>   fewer batteries-included.
> - It does not mean the docs pretend all domains are equally worked
>   out. The docs can be LLM-heavy while the architecture is
>   domain-neutral.
> - It does not mean every domain has a built-in runner. Community and
>   users build domain-specific runners using the public protocols.
>
> **The test:**
>
> Before finalizing any core contract (Runner, Module, Adapter), write
> a concrete example for at least three domains: LM, vision, and
> classical ML. If the contract feels natural in all three, it is
> general enough. If it distorts any of them, it is too LLM-specific.

---

## Decision 8: What is the minimal runner contract?

### The question
What must every runner guarantee?

Without this, cross-domain extensibility is hand-wavy.

### Recommendation
Define a tiny minimal runner contract, for example:
- how inputs are received
- how outputs are returned
- sync vs async expectations
- error semantics
- whether metadata/runtime telemetry is exposed
- serialization requirements, if any

### Important design choice
Keep this contract small.
The runner abstraction should be strong enough to unify execution, but not so detailed that every domain becomes awkward.

> ### Answer: Nail the shape, defer the details
>
> The runner contract should be the smallest thing that lets the
> framework (and optimizer) treat runners uniformly, while leaving
> domain-specific execution to domain-specific protocols.
>
> **What we know the contract needs:**
>
> 1. **Introspection.** The optimizer needs to know what a runner's
>    configurable axes are, what their types and ranges are, and what
>    the current values are. Every runner exposes `.axes()`.
> 2. **Execution.** The module needs to call the runner. But the I/O
>    types are domain-specific: messages for LMs, tensors for vision,
>    feature dicts for ML. There is no useful universal `run(input) ->
>    output` - the type parameters differ.
> 3. **Telemetry.** Metrics need usage data (tokens, cost, latency).
>    Every runner returns or exposes a `Usage` object alongside its
>    native result.
> 4. **Serializability.** The optimizer needs to reconstruct runners
>    from config. Every runner is describable as `(type_name, config_dict)`.
>
> **The minimal universal contract:**
>
> ```python
> class Runner(Protocol):
>     def axes(self) -> dict[str, Axis]: ...
>     def config(self) -> dict[str, Any]: ...
> ```
>
> That's it at the universal level. Execution is domain-specific:
>
> ```python
> class LMRunner(Runner, Protocol):
>     def complete(self, messages: list[Message], **kwargs) -> LMResponse: ...
>
> class VisionRunner(Runner, Protocol):
>     def predict(self, image: Any, **kwargs) -> VisionResult: ...
>
> class MLRunner(Runner, Protocol):
>     def predict(self, features: dict, **kwargs) -> MLResult: ...
> ```
>
> Modules know what kind of runner they need. An LM module expects an
> `LMRunner`. A vision module expects a `VisionRunner`. The universal
> `Runner` protocol exists for the optimizer and the framework
> (introspection, serialization), not for execution dispatch.
>
> **What we defer:**
>
> - Exact method signatures for domain-specific protocols (finalize
>   when implementing real runners)
> - Sync vs async (see D9 - let it be what it must be)
> - Error semantics (let real implementations reveal what's needed)
> - Streaming (important for LMs, irrelevant for sklearn - handle per
>   domain protocol)
>
> **The principle:**
>
> The universal contract is for the *framework*. The domain contract is
> for the *module*. Keep the universal part tiny. Let the domain parts
> be as rich as they need to be.

---

## Decision 9: What is the minimal module contract?

### The question
What shape does a module take?

The current design implies multiple possibilities:
- raw function
- callable object
- strategy metadata object
- program-bound execution template

### Recommendation
Choose one canonical representation, even if others are supported.

For example:
```python
module(signature, inputs, *, context) -> dict
```

Where `context` contains runtime services.

Or:
```python
program(signature, inputs) -> dict
```

Where the program has already bound the strategy and runtime.

### Key point
Do not allow the conceptual contract to vary casually across docs.
Even if the implementation is flexible, the design needs one story.

> ### Answer: What it must be, but not more
>
> Start with the simplest thing that works. Grow the contract only when
> real implementation pressure demands it.
>
> **The starting contract:**
>
> A module is a callable:
>
> ```python
> def module(sig, inputs, *, runner, adapter, **config) -> Result
> ```
>
> - `sig` - the Signature (pure intent: fields, types, examples,
>   objective)
> - `inputs` - a dict of input field values
> - `runner` - the execution engine (domain-specific protocol)
> - `adapter` - the I/O bridge
> - `**config` - module-specific parameters (hint, via, tools,
>   max_iter, etc.)
> - Returns a `Result` (output dict + trace + usage)
>
> This is a plain function. No class required. `functools.partial` works
> for binding config. A module that needs no special config is just:
>
> ```python
> def predict(sig, inputs, *, runner, adapter) -> Result:
>     prompt = adapter.render(sig, inputs)
>     response = runner.complete(prompt)
>     outputs = adapter.parse(sig, response)
>     return Result(outputs=outputs, usage=response.usage)
> ```
>
> **When does it become more than a function?**
>
> A module needs to be more than a bare function when:
>
> 1. **The optimizer needs axis metadata.** The optimizer must know that
>    `chain_of_thought` has a `hint` axis (type=str, default=None) and
>    a `via` axis (type=list, default=[("reasoning",)]). This metadata
>    can be attached via a decorator, a registry, or by making the
>    module a callable object with an `.axes()` method. Prefer the
>    lightest mechanism that works.
> 2. **The module manages internal state across iterations.** A ReAct
>    loop maintains trajectory state. A refine loop maintains retry
>    state. This state is internal to the function body and returned in
>    `Result.trace`. The function signature does not change.
> 3. **The module composes inner modules/programs.** `ensemble` runs N
>    programs. `fallback` tries programs in sequence. `pipe` chains
>    programs. These are higher-order modules - functions that take
>    other functions (or Programs) as parameters.
>
> **What we explicitly do NOT require:**
>
> - No base class. Modules are functions or callable objects, not
>   subclasses of `Module`.
> - No registration. Modules do not need to be registered with a
>   framework-level registry to work. (A registry may exist for
>   discovery and the optimizer, but it is not required for execution.)
> - No lifecycle hooks (setup, teardown). If these become necessary
>   later, they grow in, but we do not design them in up front.
>
> **Result:**
>
> ```python
> @dataclass
> class Result:
>     outputs: dict[str, Any]       # the output fields
>     trace: dict[str, Any] | None  # reasoning, trajectory, intermediate steps
>     usage: Usage | None           # tokens, cost, latency
> ```
>
> Every module returns a `Result`, not a bare dict. This is a small
> upgrade from the current code that pays for itself immediately:
> trace and usage data have a home without polluting the output dict.
>
> **The principle:**
>
> If a raw function suffices, it is a raw function. If it needs
> metadata, it is a function with attributes. If it needs state, it
> manages it internally. The contract grows from need, not from
> anticipation.

---

## Decision 10: What role do hidden fields really play?

### The question
Are hidden fields:
- semantic intermediates,
- execution scaffolding,
- debug surfaces,
- optimizer-generated decompositions,
- or all of the above?

### Why this matters
Hidden fields are one of the most interesting parts of Onux, but also one of the most overloaded.

They can mean:
- reasoning
- evidence
- draft answers
- search plans
- execution traces
- internal decomposition chosen by search

### Recommendation
Separate at least two concepts clearly:

#### A. Semantic hidden outputs
Intermediate artifacts that conceptually belong to the task.

#### B. Operational/strategy hidden outputs
Intermediate artifacts added for execution, optimization, or debugging.

You can still represent both similarly in code - but the design should distinguish them.

> ### Answer: Two distinct concepts, cleanly separated
>
> Because Signature is pure intent (Decision 1), hidden fields do not
> exist on Signature. This resolves the overloading problem directly.
>
> **Concept 1: User-desired intermediate outputs**
>
> If the user wants to see reasoning, evidence, a draft, or any
> intermediate artifact, they make it an **output field** on the
> Signature:
>
> ```python
> sig = Signature("question -> reasoning, answer")
> ```
>
> This is intent. The user is saying: "I want both reasoning and answer
> as outputs of this task." The Program must produce both. The optimizer
> must preserve both. There is nothing hidden about it.
>
> **Concept 2: Strategy-level scaffolding**
>
> If a module needs intermediate steps to do its job - chain-of-thought
> adds a reasoning step, ReAct adds a trajectory - those are configured
> on the **Program** via `via` fields:
>
> ```python
> program = Program(
>     module=chain_of_thought,
>     runner=LM("gpt-4o"),
>     via=[("reasoning", {"note": "Think step by step"})],
> )
> ```
>
> These via fields are execution scaffolding. The optimizer can add,
> remove, rename, and reorder them. They appear in `Result.trace`, not
> in `Result.outputs` (unless the user explicitly requests them via an
> `expose` mechanism).
>
> **Concept 3: Operational trace**
>
> Execution metadata that is neither an output nor scaffolding:
> trajectories, tool call logs, retry history, intermediate drafts.
> These live in `Result.trace` and are always available for debugging.
>
> **The clean picture:**
>
> | What | Where it lives | Who controls it |
> |---|---|---|
> | User-desired intermediate outputs | Signature output fields | The user (intent) |
> | Strategy scaffolding (via fields) | Program config | The module/optimizer (strategy) |
> | Operational trace | Result.trace | The module (runtime) |
>
> **Expose mechanism:**
>
> At the Layer/Model level, `expose=("reasoning",)` surfaces
> strategy-level scaffolding as graph-level symbols for downstream
> wiring or debugging. This is a graph-level convenience, not a
> Signature-level concept. The inner Signature still says
> `"question -> answer"`. The Program adds `via=["reasoning"]`. The
> Layer exposes `reasoning` as an additional symbol.

---

## Decision 11: What counts as stable, portable state?

### The question
Which objects should serialize losslessly and portably?

Candidates:
- `Signature`
- `Model`
- `Program`
- objective callables
- runners
- adapters

### Why this matters
Serialization is not just an implementation detail here. It affects:
- reproducibility
- optimizer checkpoints
- cross-process execution
- interchange formats

### Recommendation
Make a portability matrix.
For each major object, decide whether serialization is:
- **portable and stable**
- **Python-only / best-effort**
- **not guaranteed**

Example:
- Signature fields/examples/objectives-with-rubrics: portable
- Python metric callables: Python-only / by-name only
- arbitrary runtime objects: not portable

> ### Answer: Portability matrix
>
> | Object | Serialization | Portability |
> |---|---|---|
> | **Signature** (fields, types, examples, objective rubrics) | Lossless JSON | Fully portable across languages, processes, and versions |
> | **Objective metrics** (Python callables) | By qualified name only (`module.qualname`) | Python-only. Reconstruct by import. Not portable cross-language. |
> | **Program** (module, runner type, adapter type, config) | Structured JSON config: `{module: "chain_of_thought", runner: {type: "OpenAILM", model: "gpt-4o", temperature: 0.0}, adapter: "auto", hint: "...", via: [...]}` | Portable if all components are named. Reconstructible from config. |
> | **Model** (graph structure, layer types, wiring) | Lossless JSON/YAML | Fully portable. The graph is pure structure. |
> | **Model with bound Programs** | Graph structure + per-node Program configs | Same portability as Program |
> | **Checkpoint** (fully pinned Program + evaluation scores) | Full structured JSON | Portable. This is the artifact the optimizer saves and loads. |
> | **Runner instances** | By type name + constructor config | Reconstructible: `{type: "OpenAILM", config: {model: "gpt-4o", temperature: 0.0}}` |
> | **Adapter instances** | By type name + constructor config | Reconstructible. |
> | **Result** (outputs, trace, usage) | JSON | Fully portable. |
>
> **Principles:**
>
> - Everything that defines *what* (Signature, graph structure)
>   serializes losslessly and portably.
> - Everything that defines *how* (Program, runner, adapter) serializes
>   as reconstructible config - type name plus constructor arguments.
> - Python callables (metrics, check functions) serialize by qualified
>   name. Reconstruction requires the same Python environment.
> - A Checkpoint is the most important serialization target. It must be
>   fully reconstructible: load the checkpoint, import the metric
>   functions, and you have an identical executable Program.
>
> **`dump_state()` / `load_state()` pattern:**
>
> Every major object implements `dump_state() -> dict` and
> `classmethod load_state(dict) -> Self`. The dict is plain Python data
> that serializes to JSON. This pattern is already established on
> Signature and extends to Program, Model, and Checkpoint.

---

## Decision 12: How much symbolic purity do graphs need?

### The question
Are `Layer` and `Model`:
- purely symbolic build-time objects,
- symbolic but directly executable,
- or execution-bearing objects from the start?

### Recommendation
This should be stated very clearly because it affects almost everything else.

My recommendation:
- **Layers are symbolic nodes**
- **Models are symbolic graphs with an execution path**
- symbolic structure should remain inspectable regardless of execution backend

That gives the graph API a clear identity.

> ### Answer: Symbolic and executable, with a clear phase distinction
>
> Layers are symbolic nodes. Models are symbolic graphs. Both are
> inspectable at build time. Models become executable when their nodes
> have bound Programs.
>
> **Three phases:**
>
> | Phase | What exists | What you can do |
> |---|---|---|
> | **Build** | Layers, Symbols, Model graph structure | Inspect, summarize, extract signatures, validate wiring |
> | **Compile** | Model + Program bindings for each node | Everything above, plus: check that every node is executable |
> | **Run** | Compiled Model + actual inputs | Execute end-to-end, get Results |
>
> **Build phase (symbolic):**
>
> ```python
> question = Input("question")
> context = Retrieve()(question)
> answer = ChainOfThought("answer")([question, context])
> model = Model(inputs=question, outputs=answer, name="qa")
> model.summary()  # works - purely symbolic
> ```
>
> At this point no runner, adapter, or module config is bound. The graph
> is pure structure: which layers exist, how they wire together, what
> their inner signatures are.
>
> **Compile phase (bind Programs):**
>
> ```python
> model.compile(
>     programs={
>         "retrieve": Program(runner=VectorDB("index"), module=retrieve),
>         "answer": Program(runner=LM("gpt-4o"), module=chain_of_thought),
>     }
> )
> ```
>
> Or with defaults: `model.compile(default_runner=LM("gpt-4o"))` which
> auto-generates Programs for each node.
>
> **Run phase (execute):**
>
> ```python
> result = model({"question": "What is 2+2?"})
> ```
>
> **Why this matters:**
>
> The phase distinction keeps the symbolic graph useful even without a
> runtime. You can build, inspect, and share Models as pure graph
> descriptions. The optimizer can analyze Model structure before
> choosing Programs for each node. And the clear compile step makes it
> obvious what must be bound before execution.

---

## Decision 13: How much should user-facing API optimize for teaching simplicity?

### The question
Do you want the user story to be:
- minimal and elegant, with complexity hidden,
or
- explicit and systems-oriented, with more concepts exposed?

### Why this matters
The current design wants both.
That is often where library identity gets blurry.

### Recommendation
Pick a primary audience for v1:

#### If the audience is model/system hackers
Expose more of:
- Program
- runner
- adapter
- axes
- execution config

#### If the audience is general Python users building AI pipelines
Keep the top layer simpler:
- Signature
- built-in modules
- layers/models
- runner config only when needed

> ### Answer: Both - progressive disclosure
>
> The primary audience is system builders who want both power and
> clarity. The API is organized in concentric rings: a simple core that
> works immediately, with depth available on demand.
>
> **Ring 1 - First 30 seconds:**
>
> ```python
> from onux import Signature, Program, LM, predict
>
> sig = Signature("question -> answer")
> program = Program(module=predict, runner=LM("gpt-4o"))
> result = program(sig, {"question": "What is 2+2?"})
> ```
>
> Three objects. One import line. One result.
>
> **Ring 2 - First 5 minutes:**
>
> Add types, examples, a different module:
>
> ```python
> sig = (
>     Signature("question -> answer, confidence")
>     .type(confidence=float)
>     .examples([...])
> )
> program = Program(
>     module=chain_of_thought,
>     runner=LM("gpt-4o", temperature=0.0),
>     hint="Think step by step.",
> )
> ```
>
> **Ring 3 - First 30 minutes:**
>
> Multi-step pipelines, custom layers, sub-models:
>
> ```python
> question = Input("question")
> context = Retrieve()(question)
> answer = ChainOfThought("answer")([question, context])
> model = Model(inputs=question, outputs=answer)
> ```
>
> **Ring 4 - When you need it:**
>
> Optimizer, custom runners, custom adapters, axis introspection,
> checkpoints, custom modules, serialization.
>
> **The principle:**
>
> No ring requires understanding the rings beyond it. A user in Ring 1
> never sees adapters, axes, or the optimizer. A user in Ring 3 knows
> what a Program is but may never write a custom runner. Complexity is
> always available, never forced.

---

## Decision 14: What should be the public backbone of Onux?

### The question
When someone learns Onux, what are the 3-5 nouns they should retain?

Right now candidates include:
- Signature
- Module
- Program
- Runner
- Adapter
- Layer
- Model
- Objective

That is too many to all be equally central.

### Recommendation
Pick a backbone.

My suggested backbone is:
- **Signature** - task contract
- **Module** - execution strategy
- **Model** - graph composition
- **Runner** - concrete execution engine

Then:
- **Program** = advanced execution/search object
- **Adapter** = advanced portability/runtime object

That would make the design easier to teach and less top-heavy.

> ### Answer: Three nouns, with supporting cast
>
> **The backbone (what every user learns):**
>
> 1. **Signature** - what you want (the task contract)
> 2. **Program** - how to get it for one step (strategy + runner + config)
> 3. **Model** - how to get it for many steps (a graph of Programs)
>
> These three are the public identity of Onux. Every tutorial starts
> with them. Every example uses them. They appear in the first import
> line.
>
> **Supporting nouns (important, learned on demand):**
>
> 4. **Module** - the strategy pattern a Program uses (predict,
>    chain_of_thought, react, …). Users encounter it as the `module=`
>    parameter of Program.
> 5. **Runner** - the execution engine a Program uses (LM, sklearn,
>    vision model, …). Users encounter it as the `runner=` parameter
>    of Program.
>
> **Advanced nouns (real, documented, not front-page):**
>
> 6. **Adapter** - I/O bridge between Signature fields and runner
>    native format
> 7. **Optimizer** - automated search over Programs
> 8. **Checkpoint** - a fully pinned Program with evaluation scores
> 9. **Result** - output + trace + usage from execution
>
> **The mental model:**
>
> > You write a **Signature** (what you want), wrap it in a **Program**
> > (how to get it), and optionally compose Programs into a **Model**
> > (how to get it in many steps).
>
> That sentence should be enough to orient any new user.

---

## Resolved decisions summary

> | # | Decision | Answer |
> |---|---|---|
> | 1 | What is Signature? | **Pure intent.** Fields, types, examples, objective. No hint, no note, no via. |
> | 2 | Is Program first-class? | **Yes.** Core public API. The executable, serializable, optimizable unit. |
> | 3 | Primary execution unit? | **Program** (one step), **Model** (many steps). Modules are not run directly. |
> | 4 | Model and Program relationship? | **Model is a program of programs.** A DAG where each node resolves to a Program. |
> | 5 | Optimizer role? | **Major subsystem, mostly runtime internals.** Architecture is optimizer-native; user-facing API does not require the optimizer. |
> | 6 | Adapters? | **Mostly internal.** Public and extensible, but auto-generated by default. Advanced users engage deeply. |
> | 7 | Scope? | **Broad by design.** Architecture is genuinely cross-domain. Implementation matures domain by domain. LLMs first in maturity, not in architecture. |
> | 8 | Runner contract? | **Minimal universal (axes + config) plus domain-specific protocols.** LMRunner, VisionRunner, etc. Defer exact signatures until implementing real runners. |
> | 9 | Module contract? | **A callable: `(sig, inputs, *, runner, adapter, **config) -> Result`.** Start as a function. Grow only when need demands it. |
> | 10 | Hidden fields? | **Two concepts.** User-desired outputs live on Signature. Strategy scaffolding lives on Program. Operational trace lives on Result. |
> | 11 | Serialization? | **Portability matrix.** Signature and graph structure are portable JSON. Programs and runners are reconstructible config. Callables by qualified name. |
> | 12 | Graph purity? | **Symbolic and executable.** Build → Compile → Run phases. Inspection works without a runtime. Execution requires bound Programs. |
> | 13 | Teaching simplicity? | **Progressive disclosure.** Ring 1 (30 sec) through Ring 4 (when needed). No ring requires understanding deeper rings. |
> | 14 | Public backbone? | **Three nouns: Signature, Program, Model.** Module and Runner are supporting. Adapter, Optimizer, Checkpoint are advanced. |

---

## Questions answered in order

> 1. **Is Signature pure or pragmatic?**
>    Pure. Fields, types, examples, objective. hint/note/via move to Program.
>
> 2. **Is Program core in v1 or not?**
>    Core. First-class public API. The second noun users learn.
>
> 3. **What is the primary executable unit?**
>    Program for one task. Model for a pipeline. Modules are strategy
>    patterns, not directly executed.
>
> 4. **What is the relationship between Model and execution?**
>    A Model is a program of programs — a DAG where each node resolves
>    to a Program. The Model is both symbolic (for inspection) and
>    executable (when compiled with bound Programs).
>
> 5. **Are adapters public-first or advanced/internal?**
>    Advanced/internal. Public protocol, auto-generated by default,
>    omitted from tutorials until needed.
>
> 6. **Is v1 truly cross-domain or LLM-first with extensibility?**
>    Cross-domain by design. The architecture must not special-case
>    LLMs. Implementation ships LLMs first in maturity.
>
> 7. **What minimal contracts do runners and modules satisfy?**
>    Runner: `axes()` + `config()` universally, plus domain-specific
>    execution protocols (LMRunner.complete, VisionRunner.predict, etc.).
>    Module: `(sig, inputs, *, runner, adapter, **config) -> Result`.
>    Start minimal, grow from need.
>
> 8. **What state must serialize portably?**
>    Signature and Model graph structure: lossless portable JSON.
>    Program and runner configs: reconstructible structured JSON.
>    Checkpoints: fully portable. Callables: by qualified name
>    (Python-only reconstruction).
