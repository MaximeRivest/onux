---
title: "Modules"
---

# Vignette 2: Module functions

Module functions are the second layer.
They say **how** to execute one signature.

They are plain functions:

```python
(sig, inputs, *, lm, adapter) -> dict
```

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

```output:exec-1773708580582-7tq2x
```

## A tiny fake runtime

```python
from onux import Signature
from onux import predict, chain_of_thought, react, refine, code_exec
from onux import pipe, fallback, ensemble, module_name

class EchoLM:
    def __call__(self, prompt):
        return prompt

class DemoAdapter:
    def render(self, sig, inputs):
        return f"{sig.formula} :: {inputs}"

    def parse(self, sig, response):
        return {name: f"<{name}>" for name in sig.output_fields}

lm = EchoLM()
adapter = DemoAdapter()
qa = Signature("question -> answer")
inputs = {"question": "What is 2+2?"}
```

```output:exec-1773708583867-mshvj
```

## `predict`

```python
print(predict(qa, inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708586329-472wm
{'answer': '<answer>'}
```

## `chain_of_thought`

```python
print(chain_of_thought(qa, inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708606834-e4ojv
{'answer': '<answer>'}
```

## `react`

```python
def search(query):
    return f"search({query})"

print(react(qa, inputs, lm=lm, adapter=adapter, tools=[search], max_iter=1))
```

```output:exec-1773708609639-hc3dc
{'answer': '<answer>'}
```

## `refine`

```python
validator = lambda result: None
print(refine(qa, inputs, lm=lm, adapter=adapter, check=validator, max_retries=2))
```

```output:exec-1773708611135-14fu4
{'answer': '<answer>'}
```

## `code_exec`

```python
# This fake adapter never returns real code, so this example just shows the call shape.
print(code_exec(qa, inputs, lm=lm, adapter=adapter, lint=None, max_fixes=1))
```

```output:exec-1773708612932-vlqed
{'answer': '<answer>'}
```

## Parameterize with `partial`

```python
from functools import partial

careful_cot = partial(chain_of_thought, note="Break the problem into small steps")
strict_refine = partial(refine, check=lambda result: None, max_retries=3)

print(module_name(careful_cot))
print(module_name(strict_refine))
```

```output:exec-1773708613981-71j4g
chain_of_thought(note='Break the problem into small steps')
refine(check=<function <lambda> at 0x74129c130220>, max_retries=3)
```

## Compose modules with `pipe`

```python
step1 = Signature("question -> draft")
step2 = Signature("draft -> answer")
pipeline = pipe((step1, predict), (step2, chain_of_thought))

print(module_name(pipeline))
print(pipeline(Signature("question -> answer"), inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708616306-eixph
pipe
{'answer': '<answer>'}
```

## `fallback`

```python
safe = fallback(careful_cot, predict)
print(module_name(safe))
print(safe(qa, inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708618368-gqyyk
fallback
{'answer': '<answer>'}
```

## `ensemble`

```python
vote = ensemble(predict, careful_cot)
print(module_name(vote))
print(vote(qa, inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708618952-z3cn2
ensemble
{'answer': '<answer>'}
```

## Takeaway

Module functions are a clean middle layer:

- simple enough to read in one screen
- powerful enough to express validation loops and tool use
- fully swappable by an optimizer
