---
title: "Layers And Models"
---

# Vignette 3: Layers and models

Layers are the Keras-style API of Onux.
They wrap signatures and module functions in symbolic nodes.

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

## Inputs and simple generation

```python
from onux import Input, Model
from onux.layers import Generate, ChainOfThought

question = Input("question")
context = Input("context", type=list[str])

answer = ChainOfThought("answer")([question, context])
score = Generate(("score", float))([question, answer])

model = Model(inputs=[question, context], outputs=score, name="qa_pipeline")
model.summary()
```

```output:exec-1773708800695-7cuc7
Model: "qa_pipeline"
Inputs:
  - question: str
  - context: list[str]
Outputs:
  - score: float (public)
Graph:
  - [1] ChainOfThought(answer) <- question, context, module=chain_of_thought
  - [2] Generate(score) <- question, answer, module=predict
Out[1]: 'Model: "qa_pipeline"\nInputs:\n  - question: str\n  - context: list[str]\nOutputs:\n  - score: float (public)\nGraph:\n  - [1] ChainOfThought(answer) <- question, context, module=chain_of_thought\n  - [2] Generate(score) <- question, answer, module=predict'
```

## Compile and fit

```python
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

```output:exec-1773708811119-7eoki
Model: "qa_pipeline"
Inputs:
  - question: str
  - context: list[str]
Outputs:
  - score: float (public)
Graph:
  - [1] ChainOfThought(answer) <- question, context, module=chain_of_thought
  - [2] Generate(score) <- question, answer, module=predict
Compile: {'optimizer': 'auto_prompt', 'meta_lm': 'gpt-4o'}
Examples: 1 rows
Out[2]: 'Model: "qa_pipeline"\nInputs:\n  - question: str\n  - context: list[str]\nOutputs:\n  - score: float (public)\nGraph:\n  - [1] ChainOfThought(answer) <- question, context, module=chain_of_thought\n  - [2] Generate(score) <- question, answer, module=predict\nCompile: {\'optimizer\': \'auto_prompt\', \'meta_lm\': \'gpt-4o\'}\nExamples: 1 rows'
```

## Built-in non-LLM layers

```python
from onux.layers import Retrieve, ExecuteSQL

query = Input("query")
search_context = Retrieve(top_k=5)(query)
rows = ExecuteSQL()(Input("sql_query"))

print(search_context)
print(rows)
```

```output:exec-1773708819024-8gf0t
Symbol(name='context', type=list[str], role='public')
Symbol(name='rows', type=list[dict[str, Any]], role='public')
```

## A Keras-style DAG

```python
from onux.layers import ReAct

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

```output:exec-1773708953603-6tsio
Model: "research_answering_pipeline"
Inputs:
  - question: str
  - constraints: list[str]
Outputs:
  - final_answer: str (public)
  - confidence: float (public)
  - sources: str (public)
Graph:
  - [12] ReAct(notes, sources) <- question, constraints, module=react
  - [13] Generate(final_answer, confidence) <- question, constraints, notes, sources, module=predict
Out[7]: 'Model: "research_answering_pipeline"\nInputs:\n  - question: str\n  - constraints: list[str]\nOutputs:\n  - final_answer: str (public)\n  - confidence: float (public)\n  - sources: str (public)\nGraph:\n  - [12] ReAct(notes, sources) <- question, constraints, module=react\n  - [13] Generate(final_answer, confidence) <- question, constraints, notes, sources, module=predict'
```

## Models are reusable graph modules

```python
# Build a reusable subgraph.
q = Input("question")
c = Input("context")
draft = Generate("draft")([q, c])
critique = Generate("critique")([q, draft])
final = Generate("final_answer")([q, c, draft, critique])

Refiner = Model(inputs=[q, c], outputs=final, name="Refiner")

# Use that subgraph inside a bigger graph.
query = Input("query")
retrieved_context = Retrieve()(query)
answer = Refiner([query, retrieved_context])
score = Generate(("score", float))([query, answer])

pipeline = Model(inputs=query, outputs=[answer, score], name="retrieval_refine_pipeline")
pipeline.summary()
```

```output:exec-1773708865195-cw1ce
Model: "retrieval_refine_pipeline"
Inputs:
  - query: str
Outputs:
  - final_answer: str (public)
  - score: float (public)
Graph:
  - [9] Retrieve(context) <- query, module=retrieve
  - [10] Model(final_answer) <- query, context
  - [11] Generate(score) <- query, final_answer, module=predict
Out[5]: 'Model: "retrieval_refine_pipeline"\nInputs:\n  - query: str\nOutputs:\n  - final_answer: str (public)\n  - score: float (public)\nGraph:\n  - [9] Retrieve(context) <- query, module=retrieve\n  - [10] Model(final_answer) <- query, context\n  - [11] Generate(score) <- query, final_answer, module=predict'
```

## Inspect inner signatures

```python
for call, sig in pipeline.layer_signatures():
    print(call.layer_type, "=>", sig.formula)
```

```output:exec-1773708869614-tpke4
Retrieve => query -> context
Model => query, context -> final_answer
Generate => query, final_answer -> score
```

## Takeaway

Use layers and models when you want:

- a symbolic DAG
- multiple LLM calls
- mixed LLM and non-LLM nodes
- reusable subgraphs
