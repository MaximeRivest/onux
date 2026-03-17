# onux

Keras-style symbolic DAGs for LLM programs.

`onux` has four layers of abstraction:

1. **Signature** — the semantic contract
   - `question -> answer`
   - `question -> reasoning -> answer`
2. **Module function** — one execution strategy for one signature
   - `predict`, `chain_of_thought`, `react`, `refine`, `code_exec`
3. **Layer** — a Keras-style symbolic node that wraps a signature + module
   - `Generate`, `ChainOfThought`, `ReAct`, `Retrieve`, `ExecuteSQL`, `Map`
4. **Model** — a reusable DAG of layers and sub-models

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


## Quick example

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

## The inner building block: signatures

```python
from onux import Signature

sig = Signature("receipt_image -> before_tax: float, after_tax: float")
sig = sig.via("extracted_numbers", desc="All monetary values on the receipt")
sig = sig.via("reasoning", desc="Step-by-step tax calculation")

print(sig)
```

## Execution strategies are just functions

```python
from functools import partial
from onux import Signature, predict, chain_of_thought, refine

sig = Signature("question -> answer")

strict_refine = partial(refine, check=lambda result: None, max_retries=3)
```

## Graphs may mix LLM and non-LLM layers

```python
from onux import Input, Model
from onux.layers import ChainOfThought, ExecuteSQL, Generate

user_query = Input("user_query")
db_schema = Input("db_schema")

sql_query, rationale = ChainOfThought(
    outputs=("sql_query", "rationale"),
    expose=("reasoning",),
)([user_query, db_schema])

rows = ExecuteSQL("rows")(sql_query)
answer = Generate("answer")([user_query, rows, rationale])

model = Model(inputs=[user_query, db_schema], outputs=answer, name="text_to_sql")
```

## Docs

- API reference: `docs/api-reference.md`
- Vignettes:
  - `docs/vignettes/01-signatures.md`
  - `docs/vignettes/02-modules.md`
  - `docs/vignettes/03-layers-and-models.md`
  - `docs/vignettes/04-hidden-outputs-and-debug.md`
  - `docs/vignettes/05-graph-patterns.md`
  - `docs/vignettes/06-custom-modules.md`

## License

MIT
