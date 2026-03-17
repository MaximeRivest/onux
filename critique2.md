# Critique of `docs/honest-api.md` as a design

`docs/honest-api.md` is much stronger as a **design essay** than as a spec, so critiquing it as a design is the right frame.

As a design, it is ambitious, intellectually coherent in parts, and directionally promising. But it also has a few deep tensions that are not just editorial problems — they are structural design problems. The biggest one is that it wants three things at once:

1. a very clean philosophical separation between **intent** and **execution**
2. a very ergonomic user-facing API
3. a very optimizer-friendly internal representation

Those goals are individually good, but the document has not fully resolved where they conflict.

---

## What is strong about the design

## 1. The central decomposition is good
The strongest part of the design is the three-level split:
- **Signature** = what the task is
- **Module / Program** = how the task is carried out
- **Graph** = how multiple tasks are composed

That is a useful decomposition because it separates three concerns that are often muddled together in LLM frameworks:
- semantic contract
- execution strategy
- system topology

This is one of the best ideas in the document, and it feels like a real foundation rather than a slogan.

---

## 2. The “optimizer test” is a powerful design heuristic
The question “Would an optimizer change this?” is a genuinely useful way to think about API boundaries.

It is especially helpful because it cuts through vague language like:
- “prompting”
- “reasoning”
- “metadata”
- “instructions”

and asks a more operational question: is this fixed intent, or is it part of the search space?

That heuristic gives the design a real spine.

---

## 3. The design avoids making LLMs the only first-class thing
Another strong aspect is that it tries not to collapse the whole system into “prompt + model call”.

The runner/module/adapter/program framing leaves room for:
- LMs
- local code
- sklearn-style predictors
- HTTP services
- retrieval systems
- vision/audio systems

That is important if Onux is supposed to describe compound AI systems rather than just prompt pipelines.

---

## 4. It treats graphs as a real abstraction, not just chaining
The document is right to distinguish reusable symbolic graphs from one-off function composition.
That is a meaningful design move.

A lot of libraries either:
- stop at single-call abstractions, or
- jump directly to heavy workflow engines

This design is trying to occupy the middle ground: symbolic, reusable DAGs with a semantics-aware layer beneath them. That is a good ambition.

---

## 5. The design is trying to be honest about where optimization fits
The document is not pretending that prompt text, hidden fields, model choice, and tool use are sacred declarations of user intent. It is explicitly acknowledging that many of these are candidates for search and transformation.

That is a healthy design instinct. It makes the system feel more like a programmable search space and less like a thin wrapper around hand-authored prompting.

---

## The main design problem

The core design problem is this:

> the document has one conceptual center, but two practical centers.

Conceptually, the center is:
- **intent vs execution vs composition**

Practically, the document keeps shifting between:
- **Signature-centric ergonomics**, and
- **Program-centric optimizer architecture**

That creates instability. It is not yet obvious what the true heart of the library is supposed to be.

---

## The biggest design tension: purity vs ergonomics

This is the most important issue in the whole document.

The design argues that:
- `Signature` should represent only intent
- things like `hint`, `note`, and `via` are execution strategy
- therefore they should not really belong to pure intent

But the design also wants a pleasant, builder-style user experience where `Signature` is the place people naturally enrich a task.

So the document ends up wanting both:
- a **clean ontology**, and
- a **convenient authoring surface**

These are not automatically compatible.

### Why this matters
This is not a wording issue. It affects the entire identity of the system.

If `Signature` is truly pure intent, then:
- hidden fields should not live there
- prompt hints should not live there
- notes should probably not live there either
- modules/programs need their own authoring surface

If `Signature` keeps those things for usability, then the design is really saying:
- `Signature` is a mixed declarative object
- some parts are semantic
- some parts are optimization/search annotations

That is a valid design too. But it is a different design.

### My view
The current document is strongest when it argues for **purity**, but strongest in usability when it slips back toward **mixed pragmatism**.

It needs to choose more deliberately.

---

## The second big tension: is `Program` the real core or not?

The document elevates `Program` to a very important role:
- a program pins down module/runner/adapter axes
- a program is what gets optimized
- a program is what becomes a checkpoint
- a program sounds like the real executable unit

That is a strong design move. But it introduces a problem:

If `Program` is the real executable abstraction, then the system is no longer mainly centered on:
- Signatures
- module functions
- layers/models

It is centered on a search/execution object that binds them together.

### Why this is a design issue
The document currently presents `Program` as both:
- central enough to organize the optimizer and execution model around, and
- lightweight enough to almost disappear behind examples

That is unstable.

A design usually has one of two shapes:

### Shape A: Programs are the real runtime object
Then everything else should clearly feed into `Program`.
- modules are strategies
- runners are resources
- adapters are bridges
- graphs maybe compile to programs

### Shape B: Programs are an optimizer-layer abstraction
Then they should stay secondary.
- users work mostly with signatures, layers, models
- optimizer may internally synthesize/search over programs

Right now the document gestures toward Shape A but still narrates much of the system like Shape B.

---

## The third big tension: universal architecture vs realistic scope

The document is trying to support a very broad universe:
- LM runners
- vision runners
- ML runners
- code runners
- HTTP runners
- domain-specific modules
- universal modules
- adapters
- graphs
- optimizers

As a design, this breadth is admirable. But it risks over-generalizing too early.

### The danger
There is a type of design that feels elegant because it can describe everything, but becomes weak because it does not give enough resistance from concrete cases.

The runner/module/adapter/program decomposition may indeed generalize across domains — but the document has not yet shown that the abstraction is equally natural for:
- an LLM with tool calls
- a YOLO detector
- a sklearn classifier
- an HTTP service
- a regex rule

These may all fit the same *diagram* while wanting different *actual abstractions*.

### My view
The design is probably right to aim broad, but it should treat cross-domain generality as something to earn, not something to assume.

---

## The adapter design is elegant, but maybe too abstract too early

The two-layer adapter architecture — canonical request/response plus runner-specific translation — is one of the most intellectually elegant parts of the document.

It solves a real problem:
- logical content structure should not be tied to SDK quirks
- providers all encode multimodal/tool interactions differently

That is all true.

But as a design critique: this part may be arriving too early relative to the rest of the system.

### Why
Canonical schemas are expensive design commitments.
Once you introduce a canonical request/response model, you are no longer just designing an API. You are designing an intermediate language.

That raises hard design questions:
- what are the canonical content primitives?
- what is the semantics of tool calls?
- how do streaming and partial results work?
- how are structured outputs represented?
- what is lossless vs lossy across providers?

Those are not minor follow-ups. They become a major subsystem.

### Design critique
This is likely a good future architecture, but it currently feels one level more abstract than the rest of the design has earned.

---

## The optimizer is the hidden author of the whole design

One thing I like about the document is that it is explicit about optimization.
But one thing I worry about is that optimization is silently shaping too many design choices.

A lot of the document’s moves make more sense from the optimizer’s perspective than from the ordinary user’s perspective:
- intent must be cleanly separated because the optimizer needs a fixed target
- prompt annotations become mutable axes because the optimizer wants search dimensions
- programs become central because the optimizer wants concrete candidates
- axis introspection exists because the optimizer wants machine-readable knobs

All of that is coherent.
But it raises a design question:

> Is Onux primarily a user library for building systems, or an optimizer-native substrate for searching system designs?

It can be both, but if optimizer needs dominate too early, user ergonomics may become oddly indirect.

### My view
The optimizer should shape the architecture, but it should not become the only “true audience” of the design.

---

## The graph story is good, but under-integrated with the rest of the design

The graph section is directionally strong, but it feels more attached than integrated.

The document says:
- signatures express intent
- programs/modules express execution
- graphs compose nodes

But it does not fully resolve how these relate.

### Important unresolved relationships
- Is a graph node backed by a `Program`, or directly by a module + runner config?
- Is `Model` just symbolic composition, or also executable deployment state?
- Can a graph be optimized node-by-node using programs?
- Are hidden fields in graphs semantic artifacts, execution scaffolding, or both?

The graph design itself is promising. What is missing is a tighter unification with the execution model.

---

## The design is cleanest at the conceptual level and weakest at the seams

This is probably the fairest one-sentence critique.

At the concept level, the design is attractive:
- intent
- strategy
- composition

At the seam level, it gets fuzzy:
- `Signature` is pure intent, except when it is an ergonomic builder
- `Program` is central, except when it is just implied
- adapters are abstractly universal, but not yet operationally grounded
- graphs matter, but are not fully connected to the program story

This often happens in early architecture documents: the top-level ontology is better than the object boundaries.

---

## Specific design risks

## 1. Ontological over-cleaning
The design may be trying too hard to make every concept live in exactly one box.

Real systems often need a little impurity to stay usable. For example:
- field notes may partly be human semantics and partly prompt guidance
- hidden fields may be both semantic intermediates and execution scaffolding
- examples may define intent, but also serve as optimization data

A design can acknowledge that some things are mixed without collapsing into confusion.

The risk is that the document sometimes treats the ontology as cleaner than the reality.

---

## 2. Premature universalism
The document wants the same framework to elegantly cover many AI system types.
That may eventually work, but premature universal abstractions can become brittle.

The biggest risk is building for hypothetical future domains in ways that make the initial library heavier and less legible.

---

## 3. Too many “important” abstractions at once
The document contains a lot of major concepts:
- Signature
- Module
- Program
- Runner
- Adapter
- Graph
- Layer
- Model
- Objective
- Artifact
- Axis
- Checkpoint
- Optimizer

This is not necessarily too many overall, but it is too many to all feel equally foundational.

Good designs usually have:
- one or two concepts that feel primary
- a small number of secondary concepts
- the rest clearly treated as implementation or extension layers

This design has not fully established that hierarchy yet.

---

## 4. Possible mismatch between teaching order and true architecture
The document’s teaching order is sensible:
1. specify intent
2. build a program
3. wire a graph if needed

But the actual mental load may be higher because the design underneath is doing a lot more.

If the real system depends heavily on:
- program axes
- runtime binding
- adapter machinery
- optimizer-driven mutability

then the simple teaching story may not hold as naturally in practice.

---

## What I think the design should decide more explicitly

## 1. Is purity more important than ergonomics?
This is the first and most fundamental choice.

If yes:
- make `Signature` truly pure
- move prompt-like and hidden-step authoring elsewhere
- accept a more layered user API

If no:
- admit that `Signature` is a mixed authoring object
- define semantic vs optimization-affecting parts clearly
- stop pretending the object is philosophically pure

Either answer is defensible. The current in-between state is not.

---

## 2. Is `Program` the user-facing center or the optimizer-facing center?
If `Program` is central, the whole design should be more obviously shaped around it.
If it is secondary, the document should stop letting it dominate the architecture narrative.

---

## 3. Is cross-domain generality a first-order goal or a proving ground?
If it is first-order, the extension contracts need to be much tighter.
If it is a proving ground, then the design should be more comfortable saying:
- LLMs are the first-class case right now
- other runners are intended, but not equally mature yet

That would actually make the design feel more credible, not less.

---

## 4. Is the optimizer a feature of the design or the design’s hidden master?
The document should decide how optimizer-centric it wants to be.
Right now, the optimizer is conceptually downstream but architecturally upstream.

That may be correct — but it should be explicit.

---

## My recommended design adjustments

## 1. Keep the core decomposition
Do not throw away the main split:
- intent
- execution
- composition

That is the strongest part of the design.

---

## 2. Be less absolutist about purity unless you are willing to pay for it
If `Signature` is going to be pure, make it pure.
If not, let it be a pragmatic authoring object and document its mixed nature honestly.

My instinct: unless optimization is the overwhelming center of the product, some pragmatic mixing is probably worth it.

---

## 3. Demote one major abstraction
Right now there are too many near-core concepts.
I would demote one of these in the narrative:
- `Program`, or
- `Adapter`

Not remove it — just stop making all major ideas feel equally primary.

My preference would be:
- keep `Signature`, `Module`, `Model`/`Graph` as the visible backbone
- treat `Program` as the execution/search carrier
- treat `Adapter` as an advanced runtime concern unless/until it becomes unavoidable

---

## 4. Earn the universal architecture through one or two domains first
The broad vision is good, but the design would be stronger if it proved itself on:
- LLM systems
- maybe one non-LLM case, like sklearn or HTTP

before fully claiming universal runner generality.

---

## 5. Tighten the graph/execution relationship
This is where the design currently feels least unified.
The document should more concretely answer:
- how nodes are executed
- what object binds execution strategy inside a graph
- whether models compile to executable programs or remain a separate abstraction

---

## Bottom line

As a **design**, `docs/honest-api.md` is good — in some places very good.
It has a real idea at its center, not just a pile of features.

Its strongest design qualities are:
- a meaningful separation of semantic intent from execution strategy
- a healthy awareness that optimization changes API boundaries
- a graph-oriented view of compound AI systems
- a desire to generalize beyond pure prompt orchestration

Its main design weaknesses are:
- unresolved tension between conceptual purity and authoring ergonomics
- unclear true center of gravity between `Signature` and `Program`
- premature confidence in broad universal abstractions
- under-integrated relationship between graphs and execution objects

## My overall judgment
This is a **promising architecture direction with real originality**.
But it is not yet a settled design. The hardest questions are not implementation details — they are about what kind of library Onux fundamentally wants to be.

If those choices are made cleanly, the design could become unusually strong.
If not, it risks becoming a smart document wrapped around a slightly confused object model.
