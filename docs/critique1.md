# Design critique of `honest-api.md`

---

## 1. The intent/strategy split is the right instinct, but the design flinches at the hard part

The optimizer test — "would an optimizer change this?" — is a genuinely
good design principle. It gives you a single question that decides where
something goes. The problem is that the design doesn't follow through.

The doc says: Signature is pure intent. No `.hint()`. No `.via()`. No
`.note()`. No hidden fields. Then, two pages later: "`.via()`, `.hint()`,
and `.note()` stay on Signature as building blocks that modules and
programs use internally."

This isn't a pragmatic concession. It's the design contradicting itself.
Either Signature is a task contract that the optimizer holds fixed, or
it's a mutable builder that accumulates both intent and strategy and
trusts the user to know the difference. Both are defensible positions,
but the doc claims the first and implements the second.

The real question the design avoids: **who builds the Signature that a
module or program uses internally?** If `chain_of_thought` needs to add
`.via("reasoning")` to a Signature before passing it to the adapter,
does it mutate the user's Signature? Clone it? Build a new internal one?
The design punts on this by keeping `.via()` on Signature "as a building
block" without saying who owns what.

A cleaner design would probably have two distinct objects: an Intent
(or TaskSpec) that is truly fixed and a PromptSpec (or ExecutionSignature)
that a module constructs by combining the intent with strategy-layer
choices. But that's a harder design to explain, so the doc merges them
into one Signature and hopes the ambiguity won't hurt. It will.

---

## 2. Four concepts in three layers is one concept too many

The design has Signature, Module, Program, and Graph. But Module and
Program are doing a strange dance. A module is "a general strategy
pattern" and a program is "a module with parameters pinned down." In
practice this means:

- `chain_of_thought` is a module.
- `Program(module=chain_of_thought, runner=LM("gpt-4o"), hint="...")` is
  a program.

But what *is* a module, concretely? The doc says it's a function with
the contract `(sig, inputs) -> dict`. But a Program is *also* callable
with `(sig, inputs) -> dict`. And the existing code has module functions
with the contract `(sig, inputs, *, lm, adapter) -> dict`.

So a module is a function, a program is a callable config object that
wraps a module, and both have the same external contract. The distinction
is that a module has "open parameters" and a program has "pinned
parameters." But a `partial(chain_of_thought, lm=my_lm)` also pins
parameters on a module function. And a Program with unpinned axes is
"partially pinned," which makes it functionally a module with some
defaults.

The design is trying to distinguish "strategy shape" from "strategy
instance" but the boundary is fuzzy in both directions. `functools.partial`
already bridges them in Python. The question is whether Program earns its
existence as a separate concept, or whether it's just a module function
with its arguments bundled into a serializable config. If it's the latter,
the design would be simpler as: module functions + a Config object, rather
than modules + Programs.

---

## 3. The Runner abstraction is a false universal

The doc lists LM runners, vision runners, ML runners, code runners, and
HTTP runners, and says they all share:

```python
class Runner:
    def run(self, input) -> output
```

This is not an abstraction. It's a type signature that says nothing. An
LM runner takes a CanonicalRequest and returns a CanonicalResponse. A
YOLO runner takes a tensor and returns bounding boxes. A regex runner
takes a string and returns match groups. Calling them all `Runner` gives
you nominal type conformance and zero actual polymorphism. No code can
be written against `Runner` generically, because the input and output
types are entirely runner-dependent.

The design would be more honest if it said: **there is no universal runner
interface.** Modules know what kind of runner they need. An LM module
calls an LM runner. A vision module calls a vision runner. The only
shared contract is at the Program level: `(sig, inputs) -> dict`. The
runner is an internal implementation detail of the module, not a
framework-level abstraction.

This matters because the false universal is load-bearing: the Adapter
contract, the axes system, and the introspection API all assume "runner"
is a single polymorphic concept. If it isn't, all three need rethinking.

---

## 4. The Adapter two-layer split solves the wrong problem

The Adapter is designed to solve: "different LLM APIs format the same
logical content differently." The solution is a canonical intermediate
representation (CanonicalRequest/CanonicalResponse) that sits between
the adapter and the runner.

But in practice, the hard problem is not formatting differences between
LLM APIs. Libraries like LiteLLM already solve that at the HTTP level.
The hard problems are:

- **Prompt construction**: how to render signature fields, examples,
  and task instructions into an effective prompt. This is deeply
  model-dependent (Claude responds differently to XML tags than GPT does
  to JSON schemas), which means the adapter cannot actually be
  runner-independent.

- **Output parsing**: how to extract structured fields from LLM output.
  This depends on whether you're using native structured output
  (response_format), tool-call-based extraction, or free-text parsing.
  All three are runner-dependent.

- **Multi-turn conversation state**: tool calls, tool results, and
  retries require conversation history. The canonical format would need
  to represent the full multi-turn state, not just one request/response.

The design introduces a two-layer architecture (adapter → canonical →
runner) to solve a problem that is mostly already solved (API format
differences), while the problems that actually need solving (prompt
engineering, output parsing, conversation management) cut across both
layers.

For non-LM runners, the doc admits the adapter is usually
`PassthroughAdapter()` — meaning the entire two-layer scheme exists to
serve LM runners, where it doesn't actually achieve
runner-independence.

---

## 5. The Axes/Pin/Seed system is an optimizer DSL hiding inside a library

The per-axis search control section introduces:

- Three axis states: pinned, seeded, bounded
- Four control mechanisms: `pin=`, `unpin=`, `space=`, defaults
- Implicit rules: "axes the seed does not specify are open"
- Namespacing: `runner.temperature` vs `hint`
- Introspection: `.axes()` returning `Axis(type=..., range=..., choices=...)`

This is a small domain-specific language for hyperparameter search
configuration. It's well-designed as a search DSL, but it's baked into
the core library rather than being a layer on top.

The problem: most users won't use an optimizer. They'll write a program,
run it, and iterate manually. For them, the axes/pin/seed system is
invisible complexity that they never interact with. But it still shapes
the core abstractions — Program exists largely to give the optimizer
something to search over, and the runner/adapter/module factoring is
driven by "what should be independently optimizable."

A design that serves both users better would separate the two concerns:
a simple execution API (`run this signature with this config`) and an
optimization API (`search over these configs for this objective`). The
optimization API can represent configs as axis maps, but the execution
API shouldn't need to know about axes, pins, or seeds.

---

## 6. The Graph layer does too little work for its complexity

The Keras-style graph API (Input, Layer, Symbol, Model) is the most
complex part of the design surface. It introduces symbolic tensors,
producer tracking, topological ordering, and sub-model composition. But
what does it actually *do*?

In Keras, the graph compiles to an optimized execution plan, handles
automatic differentiation, manages GPU memory, and runs on multiple
backends. The graph earns its complexity because it enables things you
couldn't do without it.

In Onux, the graph is a dependency tracker that calls modules in
topological order. Each node is a function call. The graph adds:

- Pretty printing (`.summary()`)
- Symbolic wiring (ensuring outputs feed inputs)
- Sub-model reuse

But it doesn't do execution (the runtime is deferred). It doesn't do
optimization (that's the optimizer's job). It doesn't do error handling
or retries (unspecified). It doesn't do streaming or async (unspecified).

The question is whether a Keras-style symbolic graph is the right
abstraction for "call these functions in order, piping outputs to
inputs." The alternative is just... functions. A `pipe()` over module
calls, or a simple sequential/parallel composition API, would give you
the same dependency resolution with a fraction of the conceptual weight.

The Keras analogy works when you want users to think in terms of
reusable layers and composable architectures. But LLM pipelines are
typically 3-10 steps, hand-designed for a specific task, and rarely
reused as sub-graphs. The complexity of symbolic tensors and graph
compilation may not pay for itself.

---

## 7. The design is optimized for the optimizer, not the user

The entire architecture is shaped by one core belief: an optimizer will
search over programs to find good solutions. This belief drives:

- Signature being pure intent (so the optimizer has a fixed target)
- Modules being strategy patterns (so the optimizer can swap them)
- Programs having typed axes (so the optimizer knows what to search)
- Runners being self-describing (so the optimizer knows the valid ranges)
- The pin/seed/bounded system (so the user controls what's searched)

But most LLM systems today are not built by optimization. They're built
by a developer who picks a model, writes a prompt, tests it, and
iterates. The optimizer is the aspirational future; the developer loop
is the present reality.

The design should work well for both, and it could — but the doc
presents the optimizer as the *reason* for every design choice. "Why
separate intent from strategy? Because the optimizer needs a fixed
target." "Why typed axes? So the optimizer knows what to search."

A design that starts from the developer experience would ask different
questions: What's the simplest way to go from task description to working
result? How do I debug when the LLM gives bad output? How do I swap
models? How do I add a retry loop? The answers to these questions might
produce the same architecture, but they'd produce different defaults,
different documentation, and a different entry point.

---

## 8. The scope is too wide for one library

The doc envisions a framework that handles:

- LLM prompting and parsing
- Classical ML inference (sklearn, ONNX)
- Deep learning (PyTorch, JAX)
- Computer vision (YOLO, SAM, DINOv2)
- Audio processing (Whisper)
- Code execution and linting
- SQL execution
- HTTP service calls
- Vector database retrieval
- Hyperparameter optimization
- Prompt optimization
- Multi-model ensembles
- Symbolic graph compilation

No library can do all of these well. The risk isn't just scope creep —
it's that designing *for* this breadth forces premature abstractions
(like the universal Runner) that make the LLM path worse without making
the vision/ML path real.

The most successful frameworks start narrow and grow. Keras was just
neural nets. DSPy is just LLM programs. scikit-learn is just classical
ML. Each is excellent at its core use case precisely because it doesn't
try to abstract over all the others.

Onux would be stronger as an opinionated LLM-program framework that
*happens* to have extension points clean enough for vision/ML, rather
than a universal compute-graph framework that treats LLMs as just another
runner type.

---

## 9. Signature does too many jobs

Signature currently holds:

- Task schema (input/output fields and types)
- Behavioral specification (examples)
- Success criteria (objective with rubrics and metrics)
- Prompt engineering (hint, notes)
- Execution scaffolding (hidden fields via `.via()`)
- Reference systems (artifacts)
- Serialization (dump_state/load_state)

This is a lot of responsibilities for one object. The design argument is
that it's all "intent" — but as noted in point 1, hint, notes, and
hidden fields are execution strategy by the doc's own criteria.

Even setting that aside, "task schema + examples + objective" is already
three concerns. A Signature is simultaneously:

- A type (schema): what fields exist and what types they have
- A dataset reference: what examples anchor the behavior
- A loss function: what metric/rubric combination defines success
- A prompt template: what hints and notes shape the output

In ML, these are typically separate: the model architecture, the training
data, the loss function, and the training configuration are different
objects. Bundling them into one feels convenient at first but makes it
hard to reuse any one piece independently. You can't share a schema
across tasks with different objectives, or use the same objective across
different schemas, without duplicating the Signature and stripping parts
out.

---

## 10. The design doesn't have a theory of state

LLM systems are often stateful: conversation history, retrieved context
that persists across turns, user preferences accumulated over a session,
tool state (database connections, file handles). The design has no model
for this.

Every contract is a pure function: `(sig, inputs) -> dict`. There is no
session, no conversation, no memory. This is fine for single-shot
classification tasks like the triage example, but the doc also envisions
research pipelines, ReAct agents with tool loops, and multi-hop RAG —
all of which are inherently stateful within an execution.

The module contract buries state inside the function body (the ReAct loop
maintains a trajectory internally), but there's no way for the framework
to inspect, serialize, resume, or debug that state. If a ReAct agent
fails on iteration 4 of 5, there's no way to resume from iteration 3.
If you want to inspect what tools were called and what they returned,
you need the module to expose it — but the contract only returns a
flat `dict`.

The trace outputs (trajectory, reasoning) are a partial answer, but
they're after-the-fact artifacts, not live state that the framework
manages.

---

## 11. Evaluation is the hardest part and it's mostly deferred

The design puts the objective on Signature and provides `.evaluate()`,
but:

- Rubric scoring requires an LLM judge, which is itself an LLM call
  with its own cost, latency, and failure modes. The design says this
  is "an evaluation execution concern" and defers it.

- The optimizer loop — `evaluate(triage, candidate, test_set)` — needs
  to score every candidate, which means running rubric judges at scale.
  This is the most expensive part of any LLM optimization system and the
  design doesn't address it.

- Rubric quality is itself a problem. A rubric that says "score factual
  correctness from 0 to 1" will be scored differently by different judge
  models, at different temperatures, with different system prompts. The
  objective is only as stable as the judge, but the design treats the
  objective as a fixed anchor.

- For metrics, the design assumes they're cheap Python functions. But
  real evaluation metrics often require external calls (reference model
  comparison, human eval aggregation, retrieval quality assessment).
  The sync `(values, runtime) -> float` contract doesn't accommodate
  this.

Evaluation is arguably the core value proposition of the intent/strategy
split — "define what good looks like, then let the system find the best
strategy." If evaluation is half-baked, the whole architecture loses its
motivation.

---

## 12. The design has no story for iteration

The teaching order is: specify intent → build program → wire graph. But
real development is iterative:

1. Try a simple prompt, see what happens
2. Notice it's bad at X, add an example for X
3. Notice it hallucinates, add chain-of-thought
4. Notice it's slow, try a smaller model
5. Notice edge cases, add a refinement loop
6. Ship it, get user feedback, go back to step 2

Every step in this loop touches a different layer of the design
(Signature, Module, Program, runner). The design gives you clean
separation between layers, but it doesn't make the *transitions* easy.
Going from "simple prompt" to "chain-of-thought" means going from
`Program(module=predict, ...)` to `Program(module=chain_of_thought, ...)`.
Going from "one model" to "ensemble" means restructuring the Program.
Going from "single call" to "multi-step pipeline" means rewriting
everything as a Graph.

Each transition is a conceptual level change, not an incremental tweak.
The design optimizes for clean separation at the cost of smooth
iteration. Compare this to a system where you just have functions: adding
chain-of-thought is adding a line, adding a retry is wrapping a call,
adding a second step is calling two functions. The transitions are
trivial because there are no level changes.

---

## Summary

The design's core insight — separate what you want from how you get
it — is right. The optimizer test is a genuinely useful heuristic. The
Signature-as-intent idea is good.

But the design overbuilds around an optimization future that doesn't
exist yet, while underserving the developer present. It introduces four
layers of abstraction (Signature, Module, Program, Graph) where two
would suffice (task description + execution). It claims a universal
runner/adapter architecture that only really works for LLMs. It bundles
too many concerns into Signature, splits Module and Program along a
boundary that `functools.partial` already bridges, and introduces a
Keras-style graph that doesn't earn its weight for LLM workflows.

The strongest version of this design would:

1. Commit to LLMs first and let non-LM extension happen naturally
2. Keep Signature as schema + examples + objective, strip out hint/via/note
3. Merge Module and Program into one concept: a callable config
4. Replace the symbolic graph with lightweight function composition
5. Build evaluation — especially rubric judging — as a first-class
   concern rather than deferring it
6. Design the developer iteration loop before the optimizer loop
