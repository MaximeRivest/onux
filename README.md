# onux: Compound AI Systems for Humans

Keras-style symbolic DAGs for LLM programs.

`onux` lets you build language-model systems in four layers, from the
smallest semantic contract to a full reusable graph:

1. **Signature** — the semantic task contract
2. **Module function** — one execution strategy for one signature
3. **Layer** — a Keras-style symbolic node
4. **Model** — a reusable DAG of layers and sub-models

The key idea is simple:

- a **signature** says **what** goes in and what comes out
- a **module** says **how** to run that signature
- a **graph** wires many such steps together

---

## Install

```bash
uv add onux
```

For local development:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

---

## Level 1: Signatures

A signature is the inner building block of Onux.

```python
from onux import Signature

sig = Signature("question -> answer")
print(sig)
```

Signatures can include one hidden stage in the formula:

```python
sig = Signature("question -> reasoning -> answer")
print(sig)
```

For more hidden scaffolding, factor through extra fields with `.via()`:

```python
sig = Signature("receipt_image -> before_tax: float, after_tax: float")
sig = sig.via("extracted_numbers", desc="All monetary values on the receipt")
sig = sig.via("reasoning", desc="Step-by-step tax calculation")
print(sig)
```

### Descriptions live on fields

```python
from typing import Literal
from onux import Signature

Sentiment = Literal["positive", "negative", "neutral"]

review = Signature(
    "text -> sentiment: Sentiment, rating: float, summary",
    types={"Sentiment": Sentiment},
).describe(
    rating="star rating",
    summary="brief summary",
)
print(review)
```

### Data is dataframe-like, not pandas-only

`Signature(..., data=...)` accepts tabular examples in any duck-typed
dataframe-like form, including:

- pandas DataFrame
- polars DataFrame
- duckdb relation
- pyarrow Table / RecordBatch
- pyspark DataFrame
- numpy structured array / recarray
- list / iterable of dict records

```python
sig = Signature(
    "question -> answer",
    data=[{"question": "What is 2+2?", "answer": "4"}],
)
print(sig.n_examples)
```

---

## Level 2: Module functions

Module functions are plain Python functions that implement one execution
strategy for one signature.

```python
(sig, inputs, *, lm, adapter) -> dict
```

Built-ins:

- `predict`
- `chain_of_thought`
- `react`
- `refine`
- `code_exec`
- `pipe`
- `fallback`
- `ensemble`

Example:

```python
from functools import partial
from onux import Signature, chain_of_thought, refine

sig = Signature("question -> answer")
careful_cot = partial(chain_of_thought, desc="Break the problem into small steps")
strict_refine = partial(refine, check=lambda result: None, max_retries=3)
```

These are the clean middle layer between semantic contracts and full DAGs.

---

## Level 3: Layers

Layers are the Keras-style symbolic API.
They wrap signatures and module strategies in graph nodes.

Built-in layers:

- `Generate`
- `ChainOfThought`
- `ReAct`
- `Retrieve`
- `ExecuteSQL`
- `Map`

### Quick layer example

```python
from onux import Input, Model
from onux.layers import ChainOfThought, Generate

question = Input("question")
context = Input("context", type=list[str])

answer = ChainOfThought("answer")([question, context])
score = Generate(("score", float))([question, answer])

model = Model(
    inputs=[question, context],
    outputs=score,
    name="qa_pipeline",
)

model.compile(optimizer="auto_prompt", meta_lm="gpt-4o")
model.fit(
    [
        {
            "question": "What's the capital of France?",
            "context": ["Paris is the capital of France."],
            "score": 1.0,
        }
    ]
)

model.summary()
```

### Hidden and trace outputs

Layers may internally produce more than their public outputs.

- `ChainOfThought` has hidden `reasoning`
- `ReAct` has hidden `reasoning` and trace `trajectory`

Expose them explicitly when you want them:

```python
from onux import Input
from onux.layers import ChainOfThought, ReAct

question = Input("question")

reasoning, answer = ChainOfThought("answer", expose=("reasoning",))(question)
trajectory, answer = ReAct("answer", tools=[print], expose=("trajectory",))(question)
```

---

## Level 4: Models and graphs

A model closes the symbolic graph.

```python
from onux import Input, Model
from onux.layers import Generate, ReAct

question = Input("question")
constraints = Input("constraints", type=list[str])

notes, sources = ReAct(
    outputs=("notes", "sources"),
    tools=[print],
    max_iters=3,
    lm="claude-3-5-sonnet",
)([question, constraints])

final_answer, confidence = Generate(
    outputs=(("final_answer", str), ("confidence", float)),
    lm="gpt-4o",
)([question, constraints, notes, sources])

research_model = Model(
    inputs=[question, constraints],
    outputs=[final_answer, confidence, sources],
    name="research_answering_pipeline",
)

research_model.summary()
```

Graphs may mix:

- LLM nodes
- tools and local code
- retrieval
- SQL execution
- map/reduce patterns
- reusable sub-models

### Models are reusable graph modules

A `Model` can itself be called inside a larger graph.

```python
from onux import Input, Model
from onux.layers import Generate, Retrieve

question = Input("question")
context = Input("context")

draft = Generate("draft")([question, context])
final_answer = Generate("final_answer")([question, context, draft])

Refiner = Model(inputs=[question, context], outputs=final_answer, name="Refiner")

query = Input("query")
retrieved_context = Retrieve()(query)
answer = Refiner([query, retrieved_context])

pipeline = Model(inputs=query, outputs=answer, name="retrieval_refine_pipeline")
pipeline.summary()
```

### Inspect the inner signatures

Every layer in a graph still has an inner semantic signature.

```python
for call, sig in research_model.layer_signatures():
    print(call.layer_type, "=>", sig.formula)
```

---

## Extend Onux

You can extend Onux at the right level:

- write a **module function** if you want a new execution strategy
- subclass **Layer** if you want a new symbolic node
- build a **Model** if you want a reusable subgraph

This supports custom:

- LLM modules
- ML modules
- deep learning modules
- local code / tool / database modules

See the custom-extension vignette below.

---

## Docs

Start here:

- docs index: `docs/index.md`
- API reference: `docs/api-reference.md`

Vignettes:

- `docs/vignettes/01-signatures.md` — signatures
- `docs/vignettes/02-modules.md` — module functions
- `docs/vignettes/03-layers-and-models.md` — layers and models
- `docs/vignettes/04-hidden-outputs-and-debug.md` — hidden and trace outputs
- `docs/vignettes/05-graph-patterns.md` — larger graph patterns
- `docs/vignettes/06-custom-modules.md` — create your own modules and layers

---

## Public API

Top-level imports:

```python
from onux import (
    Signature,
    describe_type,
    Input,
    Symbol,
    FieldSpec,
    Layer,
    LayerCall,
    Model,
    Generate,
    ChainOfThought,
    ReAct,
    Retrieve,
    ExecuteSQL,
    Map,
    predict,
    chain_of_thought,
    react,
    refine,
    code_exec,
    pipe,
    fallback,
    ensemble,
    module_name,
)
```

---

## License

MIT
