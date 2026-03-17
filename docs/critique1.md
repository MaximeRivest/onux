# honest-api.md — Design review, decisions, and proposed revision

---

## Part 1: Design review (bullet points)

### What works

- **The optimizer test** ("would an optimizer change this?") is a
  genuinely good heuristic for separating intent from strategy. It
  should survive into any revision.
- **Signature-as-intent** is the right core idea. Defining what you want
  apart from how you get it is the conceptual moat.
- **Immutable builder API** on Signature (`.type()`, `.note()`, etc.) is
  ergonomic and Pythonic. The implementation is solid.
- **Objective on Signature** is correct — success criteria are part of
  the task definition, not the execution strategy.
- **Module functions as plain functions** is a good instinct. Functions
  compose, test, and replace easily.

### What doesn't

- **Signature says it's pure intent, then holds strategy.** `.hint()`,
  `.via()`, `.note()` are execution strategy by the doc's own test, but
  they live on Signature. The design contradicts its own principle.
- **Module vs Program is a false split.** Both are callable with `(sig,
  inputs) -> dict`. The only difference is whether parameters are "open"
  or "pinned," which is what `functools.partial` does. Program adds
  conceptual weight without clear payoff.
- **Runner is a false universal.** `run(input) -> output` unifies LMs,
  YOLO, sklearn, and regex in name only. No generic code can be written
  against it. The abstraction provides no polymorphism.
- **The Adapter two-layer split solves the wrong problem.** API format
  differences are already handled by LiteLLM-style libraries. The real
  problems — prompt construction and output parsing — are inherently
  model-dependent and can't be runner-independent.
- **The axes/pin/seed system is an optimizer DSL** baked into the core.
  Most users never optimize. The execution API shouldn't need to know
  about axes.
- **The Keras graph doesn't earn its weight.** LLM pipelines are 3–10
  hand-designed steps. Symbolic tensors, producer tracking, and graph
  compilation are overkill for calling functions in order.
- **Scope is too wide.** Designing for LMs + vision + ML + audio + SQL +
  HTTP forces premature abstractions that weaken the LLM path without
  making any other path real.
- **Signature does too many jobs.** Schema + examples + objective + hints
  + hidden fields + artifacts + serialization in one object. Hard to
  reuse any piece independently.
- **No theory of state.** Every contract is `(sig, inputs) -> dict`.
  Conversations, sessions, tool state, and resumability are unaddressed.
- **Evaluation is deferred but load-bearing.** The optimizer needs
  automated rubric judging to function. Without it, the entire
  intent/strategy split loses its motivation.
- **No iteration story.** Every improvement (add CoT, swap model, add
  retry, add a second step) is a conceptual level change, not an
  incremental tweak.

---

## Part 2: Design decisions to make

### D1. What is Signature?

**Option A: Pure intent.** Signature has fields, types, examples, and
objective. No hint, no notes, no hidden fields. Modules and programs
build their own internal execution signatures by decorating the user's
intent.

**Option B: Task description (intent + soft guidance).** Signature has
everything it has now, but the doc stops claiming it's pure intent. Hint
and notes are "default guidance that an optimizer may override." Via
fields are "suggested scaffolding." The user understands they're
starting points, not constraints.

**Option C: Two objects.** A `Task` (or `Intent`) holds the fixed
contract: fields, types, examples, objective. A `Signature` (or
`PromptSpec`) extends a Task with hint, notes, and hidden fields.
Modules receive a Task and produce their own Signature internally.

*Recommendation: B is the most practical. It matches the existing code,
is easy to explain, and works with or without an optimizer. Just be
honest about what's fixed and what's guidance.*

### D2. Does Program exist?

**Option A: Yes, as a first-class concept.** Program is a serializable
config object: `{module, runner, adapter, hint, via, ...}`. It's
callable with `(sig, inputs) -> dict`. The optimizer produces and
searches over Programs.

**Option B: No. Use module functions + config dicts.** A "program" is
just `partial(chain_of_thought, lm=my_lm, hint="...")`. Config is a
plain dict for serialization. The optimizer works with dicts.

**Option C: Program exists but only for the optimizer.** The execution
API uses module functions directly. The optimizer wraps them in a
Program internally for search/serialization.

*Recommendation: A, but keep it thin. Program is a dataclass with a
`__call__` method, not a class hierarchy. It earns its existence through
serialization and introspection, which `partial` doesn't provide.*

### D3. What is a Runner?

**Option A: Universal interface.** All runners implement `run(input) ->
output`. Adapters bridge signatures to runner-native I/O.

**Option B: No universal interface.** A runner is whatever a module
needs. LM modules take an LM runner. The framework provides LM runner
abstractions. Non-LM runners are just Python objects that modules call
directly.

**Option C: LM runner interface only.** The framework defines an LM
runner contract (`complete(messages) -> response`). Everything else is
user code.

*Recommendation: C. The framework is an LLM-program library. Define the
LM contract well. Let non-LM code be plain Python — it doesn't need a
framework abstraction.*

### D4. What is the Adapter?

**Option A: Two-layer (adapter → canonical → runner).** As described in
the doc.

**Option B: Single-layer (signature → prompt/parse).** One object that
knows how to render a signature into a prompt and parse a response into
a dict. Model-awareness is allowed.

**Option C: No separate adapter.** Prompt rendering and output parsing
are responsibilities of the module function, using utility helpers.

*Recommendation: B. A single `Adapter` with `render(sig, inputs) ->
prompt` and `parse(sig, response) -> dict` is the right granularity.
It can be model-aware. The two-layer split adds complexity for
runner-independence that isn't achievable in practice.*

### D5. Does the Keras graph exist?

**Option A: Yes, symbolic DAG.** Input, Layer, Symbol, Model, summary,
sub-models — the full Keras analogy.

**Option B: Lightweight composition.** `pipe()`, `parallel()`,
`branch()` combinators over module functions. No symbolic tensors, no
graph objects. Introspection via function metadata.

**Option C: Both.** Module functions for simple flows. Graph API for
complex DAGs. They interoperate.

*Recommendation: C, but sequence matters. Ship module functions and
composition first. Add the graph API later if users need it. Don't
design the module system around the graph.*

### D6. How does evaluation work?

**Option A: Manual.** `.evaluate()` takes pre-computed rubric scores.
The user handles rubric judging externally.

**Option B: Built-in judge.** An `Evaluator` or `Judge` object takes a
signature, a set of predictions, and an LM, and returns scores. Rubric
judging is a first-class feature.

**Option C: Pluggable.** `.evaluate()` accepts a `judge=` parameter. The
framework provides default judges but lets users supply their own.

*Recommendation: C. Evaluation is the core value proposition. If you
defer it, the intent/strategy split is just philosophy. Ship a default
LLM judge with a clear interface, and let users replace it.*

### D7. Scope: LLM-first or universal?

**Option A: Universal framework.** Design every abstraction to work with
LMs, vision, ML, audio, code.

**Option B: LLM-first with extension points.** Design for LLMs. Make
sure the extension points (custom module functions, custom layers) are
clean enough for non-LM use, but don't design the core abstractions
around non-LM runners.

*Recommendation: B. The vision/ML/audio examples in the doc are
aspirational. They're pushing the LLM abstractions into worse shapes
for zero concrete payoff. Build the best possible LLM-program library
and let the community figure out vision.*

### D8. How does state work?

**Option A: Stateless.** `(sig, inputs) -> dict`. All state is the
user's problem.

**Option B: Explicit context.** Module functions receive a `Context`
object alongside inputs. Context carries conversation history, tool
state, and execution metadata. It's inspectable and serializable.

**Option C: Return richer results.** Module functions return a `Result`
object that includes the output dict, trace, conversation history, cost,
and latency. The caller decides what to keep.

*Recommendation: C for the module contract, B for multi-step graphs.
Single-call modules return a `Result`. Graph execution maintains a
`Context` that accumulates across nodes.*

### D9. Sync or async?

**Option A: Sync only.** Simple to start. Async via `asyncio.run()` or
user wrappers.

**Option B: Async first.** All contracts are `async`. Sync is a wrapper.

**Option C: Dual.** Each function has a sync and async variant.
`predict()` and `apredict()`.

*Recommendation: A to ship, evolve toward C. Async matters for
production but sync is better for prototyping and teaching. Don't block
the initial design on async.*

### D10. What about errors?

**Option A: Exceptions.** Modules raise, callers catch.

**Option B: Result types.** Modules return `Result | Error`. The graph
handles error propagation.

**Option C: Retry is a module concern.** `refine` and `fallback` handle
retries. Raw modules raise. The user composes retry logic.

*Recommendation: C for modules, A for everything else. Retry/fallback
are already modules in the design. Let them handle it. Don't build error
propagation into the graph until it's needed.*

---

## Part 3: Proposed revised architecture

### Three concepts, not four

| Concept | What it is | What it holds |
|---|---|---|
| **Signature** | Task description | Fields, types, examples, objective, hint, notes, hidden fields |
| **Module** | Execution strategy | A callable: `(sig, inputs, *, lm, adapter) -> Result` |
| **Pipeline** | Multi-step composition | A sequence/DAG of modules wired by data flow |

Program is gone. A "program" is just a module with its parameters bound
via `partial`, a config dict, or a thin wrapper. The optimizer (when it
exists) searches over config dicts, not a separate Program type.

### Signature: honest about what it holds

```python
sig = (
    Signature("question -> answer")
    .hint("Answer in one sentence.")           # guidance, optimizer may change
    .note(answer="Short factual response")     # guidance, optimizer may change
    .type(answer=str)                          # intent, fixed
    .examples([...])                           # intent, fixed
    .objective("rubric...", my_metric, (2, 1)) # intent, fixed
)
```

The doc should say: **fields, types, examples, and objective are intent
— the optimizer holds them fixed. Hint, notes, and hidden fields are
guidance — starting points the optimizer may change.** This is the
honest version of the current design.

### Module: a function, not a class

```python
def predict(sig, inputs, *, lm, adapter) -> Result: ...
def chain_of_thought(sig, inputs, *, lm, adapter, hint=None) -> Result: ...
def react(sig, inputs, *, lm, adapter, tools, max_iter=5) -> Result: ...
def refine(sig, inputs, *, lm, adapter, check, max_retries=3) -> Result: ...
```

`Result` is a thin wrapper:

```python
@dataclass
class Result:
    outputs: dict[str, Any]        # the actual output fields
    trace: dict[str, Any] | None   # reasoning, trajectory, intermediate steps
    usage: Usage | None             # tokens, cost, latency
```

Modules compose with `partial`:

```python
careful_cot = partial(chain_of_thought, hint="Think step by step")
safe_predict = partial(refine, check=my_validator, max_retries=3)
```

### LM: a simple contract

```python
class LM(Protocol):
    def complete(self, messages: list[Message], **kwargs) -> Response: ...
```

One protocol. Provider-specific classes implement it. No universal
Runner. No canonical request/response layer. If you need LiteLLM, wrap
it in an `LM` implementation.

### Adapter: one layer

```python
class Adapter(Protocol):
    def render(self, sig: Signature, inputs: dict) -> list[Message]: ...
    def parse(self, sig: Signature, response: Response) -> dict: ...
```

Adapters can be model-aware. A `StructuredOutputAdapter` uses
`response_format`. A `XMLAdapter` uses XML tags. A `ChatAdapter` uses
plain chat messages. The choice of adapter is a strategy decision.

### Pipeline: lightweight composition

For simple sequences:

```python
from onux import pipe, parallel

# Sequential: output of step 1 feeds step 2
pipeline = pipe(
    (Signature("question -> draft"), chain_of_thought),
    (Signature("question, draft -> answer"), predict),
)
result = pipeline(input_sig, inputs, lm=lm, adapter=adapter)
```

For fan-out:

```python
# Parallel: same inputs, different modules, merge outputs
ensemble_pipeline = parallel(
    predict,
    chain_of_thought,
    partial(react, tools=[search]),
    merge=majority_vote,
)
```

For complex DAGs (later, if needed):

```python
from onux import Input, Model
from onux.layers import Generate, ChainOfThought, Retrieve

question = Input("question")
context = Retrieve()(question)
answer = ChainOfThought("answer")([question, context])
model = Model(inputs=question, outputs=answer)
```

The graph API is an optional layer on top. It is not required for simple
or medium-complexity flows.

### Evaluation: built in, not deferred

```python
from onux import Judge

judge = Judge(lm=my_lm)  # or Judge(lm="gpt-4o")

# Score a single prediction
result = sig.evaluate(
    values={"question": "2+2", "answer": "4"},
    judge=judge,       # scores rubrics automatically
    runtime=my_runtime, # optional telemetry for metrics
)

# Score a batch
scores = sig.evaluate_batch(predictions, judge=judge)
```

The `Judge` is an LM-backed rubric scorer with a clear interface. Users
can replace it. The framework provides a default. The optimizer uses it.

### Teaching order

1. **Run something.** `predict(sig, inputs, lm=lm, adapter=adapter)`
2. **Evaluate it.** `sig.evaluate(values, judge=judge)`
3. **Improve it.** Swap to `chain_of_thought`, add examples, tune the hint.
4. **Compose it.** `pipe(step1, step2)` for multi-step flows.
5. **Optimize it.** (When the optimizer exists.) Search over module
   configs for the best objective score.
6. **Graph it.** (If you need it.) Symbolic DAG for complex pipelines.

Each step is an incremental addition, not a level change.

### What this drops

- **Program as a separate concept.** Replaced by `partial` + config
  dicts.
- **Universal Runner.** Replaced by an LM-specific protocol.
- **Two-layer adapter.** Replaced by a single adapter protocol.
- **Axes/pin/seed DSL.** Deferred to the optimizer layer, not baked
  into the core.
- **Vision/ML/Audio modules.** Dropped from the core design. Extension
  points exist for them, but the core doesn't pretend to support them.
- **Reference artifacts.** Dropped until there's a concrete use case and
  implementation plan.
- **`_score` inference magic.** Dropped until it can be properly
  specified.

### What this keeps

- **Signature** with fields, types, examples, objective, hint, notes,
  hidden fields. Honest about which parts are intent and which are
  guidance.
- **Module functions** as the core execution unit.
- **Immutable builder API** on Signature.
- **Objective on Signature** with rubrics and metrics.
- **The optimizer test** as a design heuristic (not a strict partition).
- **The graph API** as an optional advanced layer.
- **Result objects** with trace and usage, replacing flat dicts.
