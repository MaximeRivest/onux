# Concrete design decisions to make for Onux

I’m interpreting your request as: **do option 2** from the previous message — a concrete list of design decisions to make next.

This document turns the critique of `docs/honest-api.md` into an actionable architecture checklist.

---

## Decision 1: What is a `Signature`, exactly?

### The question
Is `Signature`:
1. a **pure intent object**, or
2. a **pragmatic authoring object** that mixes intent with execution annotations?

### Option A — Pure intent
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

### Option B — Mixed authoring object
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
- harder to say what counts as “same intent”

### Recommendation
If Onux is primarily an **optimizer-native system design library**, choose **Option A**.
If Onux is primarily a **human-authored library first**, choose **Option B**.

My default recommendation: **Option B in v1, documented honestly**, unless optimization is the central product thesis.

---

## Decision 2: Is `Program` a first-class public object in v1?

### The question
Should users directly construct and reason about `Program`, or is it an optimizer/runtime abstraction that can remain secondary for now?

### Option A — `Program` is core public API
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

### Option B — `Program` is deferred or secondary
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
Decide this early. Don’t let `Program` be “half-core”.

My recommendation:
- **If optimizer is not shipping early, keep `Program` secondary in v1**.
- **If optimizer is central, make `Program` explicitly first-class now**.

---

## Decision 3: What is the one primary execution unit?

### The question
What is the main thing that “runs” in Onux?

Candidates:
- module function
- program
- model
- all of the above equally

### Why this matters
If too many things are independently executable, the API becomes conceptually muddy.
Users won’t know what the canonical runtime surface is.

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

---

## Decision 4: What is the relationship between `Model` and `Program`?

### The question
Are they:
1. separate abstractions,
2. one compiled into the other, or
3. one a subtype of the other?

### Viable choices

#### Option A — `Model` is symbolic, `Program` is executable
A `Model` is graph structure.
Execution happens by compiling the graph into one or more programs.

#### Option B — `Model` is itself executable and separate from `Program`
`Model` handles graph execution directly.
`Program` only matters for single-node execution or optimizer work.

#### Option C — `Model` is a kind of program
The graph itself is just a higher-order program.

### Recommendation
My recommendation is **Option A or B**, not C.

C risks collapsing symbolic composition and concrete execution into one overloaded concept.

Most likely best choice:
- **`Model` is symbolic and executable**
- **`Program` is the execution/search object for node-level or whole-model strategies**

But whichever you choose, specify it plainly.

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

#### Stance A — Optimizer-native library
The base architecture is explicitly built around optimization.

#### Stance B — Build systems first, optimize later
Keep optimizer compatibility, but don’t let it dominate v1.

My recommendation: **Stance B unless the optimizer is the actual near-term differentiator**.

---

## Decision 6: Are adapters public, or mostly runtime internals?

### The question
Should users think in terms of adapters as a major concept, or should adapters mostly be hidden behind runners and modules?

### Option A — Public concept
Users can author, configure, swap, and inspect adapters.

#### Pros
- powerful
- explicit
- good for advanced users and provider portability

#### Cons
- adds abstraction weight
- may expose too much machinery too early

### Option B — Mostly internal in v1
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

---

## Decision 7: How broad is v1 domain support really?

### The question
Is Onux in v1 truly for:
- LLMs + graphs + prompts + tools + retrieval + ML + vision + audio

or is it really:
- LLM systems first,
- with a runner abstraction intended to generalize later?

### Why this matters
There’s a difference between:
- designing for extension, and
- claiming equal maturity across domains

### Recommendation
Be explicit.

Suggested stance:
- **LLM systems are first-class in v1**
- **non-LLM runners are supported by design, but may be less mature initially**

This is not a weakness. It makes the design more believable.

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

You can still represent both similarly in code — but the design should distinguish them.

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

---

## Decision 14: What should be the public backbone of Onux?

### The question
When someone learns Onux, what are the 3–5 nouns they should retain?

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
- **Signature** — task contract
- **Module** — execution strategy
- **Model** — graph composition
- **Runner** — concrete execution engine

Then:
- **Program** = advanced execution/search object
- **Adapter** = advanced portability/runtime object

That would make the design easier to teach and less top-heavy.

---

## Recommended decisions if you want a pragmatic v1

If I had to choose a concrete path now, I would recommend:

1. **Keep `Signature` pragmatic rather than philosophically pure**
   - allow `hint`, `note`, `via`
   - but document that they are optimization-sensitive annotations

2. **Keep `Program` real but secondary in v1**
   - design for it
   - don’t make it the emotional center of the API yet

3. **Make `Model` the main composition abstraction**
   - symbolic graphs are a genuine differentiator

4. **Treat runners as first-class, adapters as advanced**
   - avoid over-committing to canonical adapter IR too early

5. **Be explicit that LLMs are the most mature initial domain**
   - preserve extensibility without pretending all domains are equally worked out

6. **Define one canonical execution story**
   - even if multiple layers are callable in implementation

7. **Separate semantic hidden outputs from operational/debug traces conceptually**

8. **Add a clear portability policy for serialized state**

---

## Questions to answer in order

If you want to resolve this efficiently, answer these in sequence:

1. Is `Signature` pure or pragmatic?
2. Is `Program` core in v1 or not?
3. What is the primary executable unit?
4. What is the relationship between `Model` and execution?
5. Are adapters public-first or advanced/internal?
6. Is v1 truly cross-domain or LLM-first with extensibility?
7. What minimal contracts do runners and modules satisfy?
8. What state must serialize portably?

That order should force the rest of the architecture to become much clearer.
