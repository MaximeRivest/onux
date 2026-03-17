---
title: "Hidden Outputs And Debug"
---

# Vignette 4: Hidden outputs and debug graphs

Onux distinguishes three output kinds:

- **public** — normal graph outputs
- **hidden** — semantic scaffolding such as reasoning
- **trace** — operational diagnostics such as a ReAct trajectory

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

## Hidden outputs in `ChainOfThought`

```python
from onux import Input, Model
from onux.layers import ChainOfThought, ReAct, Generate

question = Input("question")

answer = ChainOfThought("answer")(question)
print(answer)
```

```output:exec-1773708883228-1ewl3
Symbol(name='answer', type=str, role='public')
```

By default, only the public output is returned.

```python
reasoning, answer = ChainOfThought("answer", expose=("reasoning",))(question)
print(reasoning)
print(answer)
```

```output:exec-1773708886047-bkrli
Symbol(name='reasoning', type=str, role='hidden')
Symbol(name='answer', type=str, role='public')
```

## Trace outputs in `ReAct`

```python
trajectory, answer = ReAct(
    "answer",
    tools=[print],
    expose=("trajectory",),
)(question)
print(trajectory)
print(answer)
```

```output:exec-1773708888056-i03s2
Symbol(name='trajectory', type=list[str], role='trace')
Symbol(name='answer', type=str, role='public')
```

You can expose more than one internal output.

```python
reasoning, trajectory, answer = ReAct(
    "answer",
    tools=[print],
    expose=("reasoning", "trajectory"),
)(question)
print(reasoning, trajectory, answer)
```

```output:exec-1773708890710-stiqz
Symbol(name='reasoning', type=str, role='hidden') Symbol(name='trajectory', type=list[str], role='trace') Symbol(name='answer', type=str, role='public')
```

## Production model vs debug model

```python
question = Input("question")
context = Input("context")
reasoning, answer = ChainOfThought("answer", expose=("reasoning",))([question, context])

ResearchAnswerer = Model(
    inputs=[question, context],
    outputs=answer,
    name="ResearchAnswerer",
)

ResearchAnswererDebug = Model(
    inputs=[question, context],
    outputs=[reasoning, answer],
    name="ResearchAnswererDebug",
)

ResearchAnswerer.summary()
ResearchAnswererDebug.summary()
```

```output:exec-1773708893145-n57pg
Model: "ResearchAnswerer"
Inputs:
  - question: str
  - context: str
Outputs:
  - answer: str (public)
Graph:
  - [5] ChainOfThought(reasoning, answer) <- question, context, module=chain_of_thought
Model: "ResearchAnswererDebug"
Inputs:
  - question: str
  - context: str
Outputs:
  - reasoning: str (hidden)
  - answer: str (public)
Graph:
  - [5] ChainOfThought(reasoning, answer) <- question, context, module=chain_of_thought
Out[5]: 'Model: "ResearchAnswererDebug"\nInputs:\n  - question: str\n  - context: str\nOutputs:\n  - reasoning: str (hidden)\n  - answer: str (public)\nGraph:\n  - [5] ChainOfThought(reasoning, answer) <- question, context, module=chain_of_thought'
```

## Inner semantic signatures stay clean

Even if a layer exposes a hidden or trace output for debugging,
its inner semantic signature stays focused on the task.

```python
for call, sig in ResearchAnswererDebug.layer_signatures():
    print(call.layer_type, sig.formula)
```

```output:exec-1773708900984-4zmhq
ChainOfThought question, context -> reasoning -> answer
```

## Takeaway

This pattern keeps the public API clean while preserving:

- reasoning for debugging
- trajectories for inspection
- symbolic nodes for downstream wiring when you explicitly ask for them
