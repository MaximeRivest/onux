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

### Runner contract

A runner is anything that takes an input and produces an output in its
own native format:

```python
class Runner:
    def run(self, input) -> output
```

Different runner types have completely different native I/O and their
own parameters:

```python
# LM runners — different SDKs, same concept
OpenAILM("gpt-4o", temperature=0.0)
OpenAILM("ft:gpt-4o:triage-v3")          # finetuned
AnthropicLM("claude-3-5-sonnet", temperature=0.3)
LiteLLM("gpt-4o", provider="azure")      # same model, different provider
PydanticAIRunner("gpt-4o")               # different SDK
LangChainLLM("gpt-4o")                   # another SDK

# Vision runners
SAMRunner(checkpoint="sam_vit_h.pth", points_per_side=32, pred_iou_thresh=0.88)
YOLORunner(weights="yolov8x.pt", conf_thresh=0.25, iou_thresh=0.45)
DINOv2Runner(model="dinov2_vitl14", device="cuda")

# ML runners
SklearnRunner("triage_classifier.joblib")
TorchRunner("model.pt", device="cuda", batch_size=32)
ONNXRunner("model.onnx")

# Code runners
RegexRunner(r"severity:\s*(low|medium|high|critical)")
PythonRunner(my_triage_function)
HTTPRunner("https://internal.example.com/triage/v2", timeout=5.0)
```

The runner just does its native thing. It knows nothing about signatures.

### Adapter contract

An adapter bridges between signature fields and a runner's native I/O:

```python
class Adapter:
    def encode(self, sig, inputs) -> runner_input
    def decode(self, sig, output) -> dict
```

Different runner types need different adapters:

```python
# LM: render fields into a prompt, parse response back
XMLAdapter()          # XML-tagged prompt format
JSONAdapter()         # JSON structured output
ChatAdapter()         # chat messages format

# sklearn: extract features, map predictions
SklearnAdapter(vectorizer=tfidf, label_map=LABELS)

# Vision: load image, format annotations, convert outputs
SAMAdapter(input_key="image", output_format="masks")
YOLOAdapter(input_key="image", output_format="boxes")

# HTTP: serialize/deserialize JSON
HTTPJSONAdapter()

# Trivial: runner already speaks in dicts, no transformation
PassthroughAdapter()
```

For LM runners, adapters can often be **auto-generated** from the
signature's fields and types. For other runners, adapters are typically
written by hand or provided by the runner library.

### Axes are runner-type-dependent

Program axes are not universal. There are three kinds:

**Shared axes** — apply to all programs regardless of runner:

| Axis | What it controls |
|---|---|
| `module` | Strategy pattern |
| `runner` | What executes |
| `adapter` | Encode/decode bridge |

**Runner-specific axes** — each runner type has its own parameters:

| LM axes | Vision axes | ML axes |
|---|---|---|
| `model` | `checkpoint` / `weights` | `model_path` |
| `provider` / `sdk` | `device` | `device` |
| `temperature` | `conf_thresh` | `batch_size` |
| `max_tokens` | `iou_thresh` | `feature_extractor` |
| | `points_per_side` | |

**Module-specific axes** — each module adds its own parameters:

| LM module axes | Vision module axes | Universal module axes |
|---|---|---|
| `hint` | `scales` (multi_scale) | `max_retries` (refine) |
| `notes` | `tile_size` (tile_and_merge) | `n` (ensemble) |
| `via` fields | `augmentations` (TTA) | `check` (refine) |
| `tools` (react) | `prompt_points` (SAM) | `timeout` (fallback) |

### Per-axis search control

Each axis can be independently:

- **Pinned** — fixed value, optimizer does not touch it.
- **Seeded** — starting value, optimizer explores from it.
- **Bounded** — a defined search space: a list of options, a range,
  or a constraint.

`pin="all"` pins every axis the seed specifies. Axes the seed does not
specify are open — there is nothing to pin. Use `unpin=[...]` to pin
all specified axes except a few:

```python
seed = Program(
    module=chain_of_thought,
    runner=OpenAILM("gpt-4o", temperature=0.0),
    adapter=XMLAdapter(),
    hint="Read the support request carefully.",
)
# seed specifies: module, runner, adapter, hint
# seed does NOT specify: via, notes
# → via and notes are open regardless of pin setting

# Pin everything specified, only via and notes are open (unspecified)
results = optimizer.search(intent=triage, seed=seed)

# Pin everything specified EXCEPT hint — now hint, via, notes are open
results = optimizer.search(intent=triage, seed=seed, unpin=["hint"])

# Pin nothing — every specified axis is a seed (starting point),
# unspecified axes are open
results = optimizer.search(intent=triage, seed=seed, pin="none")

# No seed — everything is open
results = optimizer.search(intent=triage)
```

To execute a program directly (not search), the only required axis is
`runner`. Everything else has sensible defaults or is auto-generated:

```python
# Minimum executable program
Program(module=predict, runner=OpenAILM("gpt-4o"))

# module defaults to predict if omitted
Program(runner=OpenAILM("gpt-4o"))

# adapter auto-generated from signature fields and types
# hint defaults to none
# via defaults to none (module may add its own)
```

### Introspection: runners, modules, and adapters are self-describing

Runners, modules, and adapters report their own axes, defaults, and
valid ranges. This is how users discover what is configurable and how
the optimizer knows what to search over.

```python
>>> OpenAILM.axes()
{
    "model":       Axis(type=str, default="gpt-4o", choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", ...]),
    "temperature": Axis(type=float, default=1.0, range=(0.0, 2.0)),
    "max_tokens":  Axis(type=int, default=4096, range=(1, 128000)),
    "top_p":       Axis(type=float, default=1.0, range=(0.0, 1.0)),
}

>>> AnthropicLM.axes()
{
    "model":       Axis(type=str, default="claude-3-5-sonnet", choices=["claude-3-5-sonnet", "claude-3-5-haiku", ...]),
    "temperature": Axis(type=float, default=1.0, range=(0.0, 1.0)),
    "max_tokens":  Axis(type=int, default=4096, range=(1, 200000)),
}

>>> YOLORunner.axes()
{
    "weights":     Axis(type=str, default="yolov8x.pt", choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"]),
    "conf_thresh": Axis(type=float, default=0.25, range=(0.0, 1.0)),
    "iou_thresh":  Axis(type=float, default=0.45, range=(0.0, 1.0)),
    "device":      Axis(type=str, default="cuda", choices=["cpu", "cuda"]),
}

>>> SklearnRunner.axes()
{
    "model_path":  Axis(type=str, required=True),
}
```

Modules do the same:

```python
>>> chain_of_thought.axes()
{
    "hint": Axis(type=str, default=None),
    "via":  Axis(type=list, default=[("reasoning",)]),
}

>>> react.axes()
{
    "hint":      Axis(type=str, default=None),
    "via":       Axis(type=list, default=[("reasoning",)]),
    "tools":     Axis(type=list, required=True),
    "max_iter":  Axis(type=int, default=5, range=(1, 20)),
}

>>> ensemble.axes()
{
    "n": Axis(type=int, default=3, range=(1, 20)),
}

>>> tile_and_merge.axes()
{
    "tile_size":  Axis(type=int, default=640, range=(320, 1280)),
    "overlap":    Axis(type=float, default=0.2, range=(0.0, 0.5)),
}
```

A program combines the axes of its runner and module, and shows current
values against defaults:

```python
>>> program = Program(
...     module=chain_of_thought,
...     runner=OpenAILM("gpt-4o", temperature=0.0),
... )
>>> program.axes()
{
    # Module axes
    "hint": Axis(value=None, default=None),
    "via":  Axis(value=[("reasoning",)], default=[("reasoning",)]),
    # Runner axes
    "runner.model":       Axis(value="gpt-4o", default="gpt-4o"),
    "runner.temperature": Axis(value=0.0, default=1.0),
    "runner.max_tokens":  Axis(value=4096, default=4096),
    "runner.top_p":       Axis(value=1.0, default=1.0),
    # Adapter axes
    "adapter": Axis(value="auto", default="auto"),
}
```

The optimizer uses these axis descriptions to know what it can search,
what the valid ranges are, and where the defaults sit.

**Use case: compare LM SDKs and providers.** Same model, different
backends, to evaluate latency, quantization, and cost.

```python
seed = Program(
    module=chain_of_thought,
    runner=OpenAILM("gpt-4o", temperature=0.0),
    adapter=XMLAdapter(),
    hint="Read the support request carefully.",
)

results = optimizer.search(
    intent=triage,
    seed=seed,
    space={
        "runner": [
            OpenAILM("gpt-4o"),
            LiteLLM("gpt-4o", provider="azure"),
            LiteLLM("gpt-4o", provider="anyscale"),
            LiteLLM("gpt-4o", provider="fireworks"),
        ],
    },
    pin=["adapter", "module", "hint", "via"],
)
```

**Use case: find the best model.** Keep the cheap adapter and strategy,
search over models.

```python
results = optimizer.search(
    intent=triage,
    seed=seed,
    space={
        "runner": [
            OpenAILM("gpt-4o-mini"), OpenAILM("gpt-4o"),
            AnthropicLM("claude-3-5-haiku"), AnthropicLM("claude-3-5-sonnet"),
            OpenAILM("ft:gpt-4o:triage-v3"),
        ],
    },
    pin=["adapter", "module", "hint", "via"],
)
```

**Use case: LM vs sklearn vs finetuned.** Completely different runner
types for the same intent.

```python
results = optimizer.search(
    intent=triage,
    space={
        "program": [
            Program(module=chain_of_thought, runner=OpenAILM("gpt-4o")),
            Program(module=predict, runner=OpenAILM("ft:gpt-4o:triage-v3")),
            Program(module=predict, runner=SklearnRunner("triage_v3.joblib")),
        ],
    },
)
```

**Use case: vision — find the best YOLO config.** Search over
runner-specific axes.

```python
seg_intent = Signature(
    "image -> boxes, labels, scores",
    examples=[...],
    objective=mean_average_precision,
)

results = optimizer.search(
    intent=seg_intent,
    seed=Program(
        module=predict,
        runner=YOLORunner(weights="yolov8x.pt", conf_thresh=0.25),
    ),
    space={
        "runner.conf_thresh": (0.1, 0.9),
        "runner.iou_thresh": (0.3, 0.7),
        "runner.weights": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"],
    },
)
```

**Use case: search everything.** No seed, no pins.

```python
results = optimizer.search(intent=triage)
```

A checkpoint is a fully concrete program produced by the optimizer —
every axis has a specific value, and it can be executed and reproduced.

### Modules: strategy patterns

A module defines the execution shape: what control flow wraps the runner,
what transformations happen around it. Some modules are universal, some
are domain-specific.

**Universal modules** — work with any runner type:

- **predict** — single runner call
- **ensemble** — run N programs, vote on results
- **fallback** — try one program, fall back on failure
- **refine** — validate output, retry on failure
- **cascade** — fast/cheap runner first, expensive runner only if needed

**LM-specific modules:**

- **chain_of_thought** — adds `.via("reasoning")`, then predicts
- **react** — adds reasoning, enters a tool-calling loop
- **code_exec** — generates code, runs it, fixes errors in a loop
- **self_consistency** — sample N reasoning paths, vote on final answer

**Vision-specific modules:**

- **tile_and_merge** — tile a large image, run on each tile, merge
  results (useful for SAM, YOLO on high-res images)
- **multi_scale** — run at multiple resolutions, aggregate detections
  (YOLO, DINOv2)
- **test_time_augmentation** — run with flips/rotations/crops, merge
  predictions (any vision model)
- **prompt_ensemble** — try different point/box prompts, merge masks
  (SAM-specific)

**ML-specific modules:**

- **calibrate** — predict, then calibrate probabilities
- **stacking** — multiple models feed a meta-model

**Audio-specific modules:**

- **chunk_and_transcribe** — split audio into segments, transcribe each,
  merge (Whisper)

The framework provides universal modules and LM modules as built-ins.
Domain-specific modules are written by users or communities — a module
is just a function with the contract `(sig, inputs) -> dict`.

The pattern is always the same: a module wraps one or more runner calls
in a control flow strategy. What varies is which domain the strategy
serves.

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

Module strategies range from simple to exotic, across all domains:

**Single call**: one runner call, no extra logic. `predict` is here —
it works with any runner type.

**Input transformation**: modify or augment the input before calling the
runner. LM: `chain_of_thought` (adds reasoning field). Vision:
`multi_scale` (resize to multiple resolutions). Audio:
`chunk_and_transcribe` (split into segments).

**Output aggregation**: run multiple times, aggregate results. `ensemble`
and `test_time_augmentation` are here — both universal patterns applied
to different domains.

**Validation loop**: run, check, retry. `refine` is here — works with
any runner.

**Interleaved code**: run code between runner calls. LM: `react` (tool
loop), `code_exec` (generate-lint-fix). Vision: `tile_and_merge` (run
per tile, stitch results). ML: `cascade` (fast model filters, detailed
model refines).

The spectrum is smooth, and the contract is always the same:
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

### On the program (each axis: pinned, seeded, or bounded)

Axes include: module, runner (model, provider, temperature), adapter,
via fields, hint, notes, and control flow params (max retries, tools, n).

Each axis is independently pinned, seeded with a starting value, or
bounded to a search space. A program can be executed directly or handed
to an optimizer as a seed. A checkpoint is a fully concrete program.

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
- Search over seeded and bounded axes (runner, provider, model, adapter,
  via fields, hint, notes, temperature — whatever isn't pinned)
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
| Strategy pattern | Module | Universal, LM-specific, or domain-specific |
| Runner | Program (axis) | LM, vision, ML, code, HTTP — each with own params |
| Adapter | Program (axis) | Bridges signature fields to runner's native I/O |
| Runner-specific params | Program (axis) | Temperature, conf_thresh, device, etc. |
| Module-specific params | Program (axis) | Hint, via, tools, scales, tile_size, etc. |
| Multi-step data flow | Graph (Layer, Model) | Composition of modules |
| Runtime telemetry | `runtime` parameter | Not a task field |

The honest truth is:

- **A Signature says WHAT to produce.** Inputs, outputs, types, examples,
  objective. It is the behavioral target the optimizer holds fixed.
- **A module defines the strategy pattern.** Some are universal (predict,
  ensemble, refine, fallback). Some are LM-specific (chain_of_thought,
  react). Some are domain-specific (tile_and_merge, multi_scale, TTA).
  They define control flow without pinning down parameters.
- **A program specifies values for a module's axes.** Axes are
  runner-type-dependent (LM: temperature, provider; vision: conf_thresh,
  iou_thresh; ML: model_path, device) and module-dependent (hint, via,
  tools, scales). Each axis is independently pinned, seeded, or bounded.
  A checkpoint is a fully concrete program.
- **A graph wires multiple modules together.** Each node has a signature
  and a module. The graph handles data flow.
- **The optimizer test keeps the boundary clean.** If an optimizer would
  change it, it is execution strategy. If not, it is intent.
