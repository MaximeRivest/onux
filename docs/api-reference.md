---
title: "Api Reference"
---

# Onux API reference

This document shows the full public API of `onux`, with one small example
for each object.

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

```output:exec-1773708246437-2rqcy
```

## Top-level imports

```python
from onux import (
    ChainOfThought,
    ExecuteSQL,
    Field,
    FieldSpec,
    Generate,
    Input,
    Layer,
    LayerCall,
    Map,
    Model,
    ReAct,
    Retrieve,
    Signature,
    Symbol,
    chain_of_thought,
    code_exec,
    ensemble,
    fallback,
    module_name,
    pipe,
    predict,
    react,
    refine,
)
```

```output:exec-1773708252451-2b712
```

## Signatures

### `Signature`

```python
sig = Signature("question -> answer")
print(sig.formula)
```

```output:exec-1773708255247-9azc1
question -> answer
```

### `Field`

```python
field = Field("rating", "output", float, "star rating")
print(field.name, field.role, field.base_type.__name__, field.note)
```

```output:exec-1773708257153-ajvqd
rating output float star rating
```

## Core symbolic graph API

### `Input`

```python
question = Input("question", type=str, note="User question")
print(question)
```

```output:exec-1773708263262-ytuuo
Symbol(name='question', type=str, role='input')
```

### `Symbol`

```python
symbol = Input("context", type=list[str])
print(symbol.name, symbol.short_type(), symbol.is_input)
```

```output:exec-1773708266382-gkdmb
context list[str] True
```

### `FieldSpec`

```python
spec = FieldSpec("reasoning", str, "Step-by-step reasoning", kind="hidden")
print(spec)
```

```output:exec-1773708269478-gzvvj
FieldSpec(name='reasoning', annotation=<class 'str'>, note='Step-by-step reasoning', kind='hidden')
```

### `Layer`

`Layer` is the base class for custom layers.

```python
class Translate(Layer):
    pass

text = Input("text")
translation = Translate("translation", hint="Translate to French.")(text)
print(translation)
```

```output:exec-1773708273894-abhtq
Symbol(name='translation', type=str, role='public')
```

### `LayerCall`

```python
call = translation.producer
print(call.layer_type, call.layer_name)
print(call.signature.formula)
```

```output:exec-1773708276551-ggzp1
Translate translation
text -> translation
```

### `Model`

```python
model = Model(inputs=text, outputs=translation, name="translator")
_ = model.summary()
```

```output:exec-1773708355793-ymooz
Model: "translator"
Inputs:
  - text: str
Outputs:
  - translation: str (public)
Graph:
  - [1] Translate(translation) <- text, module=predict
```

## Built-in layers

### `Generate`

```python
question = Input("question")
answer = Generate("answer")(question)
print(answer)
```

```output:exec-1773708359789-n50w5
Symbol(name='answer', type=str, role='public')
```

### `ChainOfThought`

```python
reasoning, answer = ChainOfThought("answer", expose=("reasoning",))(question)
print(reasoning, answer)
```

```output:exec-1773708361984-qjhpf
Symbol(name='reasoning', type=str, role='hidden') Symbol(name='answer', type=str, role='public')
```

### `ReAct`

```python
trajectory, answer = ReAct("answer", expose=("trajectory",))(question)
print(trajectory, answer)
```

```output:exec-1773708367619-f47hj
Symbol(name='trajectory', type=list[str], role='trace') Symbol(name='answer', type=str, role='public')
```

### `Retrieve`

```python
context = Retrieve()(question)
print(context)
```

```output:exec-1773708370016-ceidg
Symbol(name='context', type=list[str], role='public')
```

### `ExecuteSQL`

```python
sql_query = Input("sql_query")
rows = ExecuteSQL()(sql_query)
print(rows)
```

```output:exec-1773708370694-evc06
Symbol(name='rows', type=list[dict[str, Any]], role='public')
```

### `Map`

```python
chunks = Input("chunks", type=list[str])
summaries = Map(Generate("summary"))(chunks)
print(summaries)
```

```output:exec-1773708372811-45q3s
Symbol(name='summary', type=list[str], role='public')
```

## Module functions

These operate on signatures, not symbolic graphs.

```python
class MockLM:
    def __call__(self, prompt):
        return prompt

class MockAdapter:
    def render(self, sig, inputs):
        return f"{sig.formula} :: {inputs}"
    def parse(self, sig, response):
        return {name: f"<{name}>" for name in sig.output_fields}

lm = MockLM()
adapter = MockAdapter()
sig = Signature("question -> answer")
inputs = {"question": "What is 2+2?"}
```

```output:exec-1773708376069-t4acx
```

### `predict`

```python
print(predict(sig, inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708378361-e2ayx
{'answer': '<answer>'}
```

### `chain_of_thought`

```python
print(chain_of_thought(sig, inputs, lm=lm, adapter=adapter))
```

```output:exec-1773708385951-26nud
{'answer': '<answer>'}
```

### `react`

```python
def search(query):
    return f"search({query})"

print(react(sig, inputs, lm=lm, adapter=adapter, tools=[search], max_iter=1))
```

```output:exec-1773708388035-a9ci1
{'answer': '<answer>'}
```

### `refine`

```python
print(refine(sig, inputs, lm=lm, adapter=adapter, check=lambda result: None))
```

```output:exec-1773708391051-j8srw
{'answer': '<answer>'}
```

### `code_exec`

```python
code_sig = Signature("question -> answer")
result = code_exec(code_sig, inputs, lm=lm, adapter=adapter, lint=None, max_fixes=1)
print(result)
```

```output:exec-1773708392548-afhdj
{'answer': '<answer>'}
```

### `pipe`

```python
step1 = Signature("question -> draft")
step2 = Signature("draft -> answer")
pipeline = pipe((step1, predict), (step2, predict))
print(module_name(pipeline))
```

```output:exec-1773708395400-oj39y
pipe
```

### `fallback`

```python
safe = fallback(chain_of_thought, predict)
print(module_name(safe))
```

```output:exec-1773708396523-nfebc
fallback
```

### `ensemble`

```python
vote = ensemble(predict, chain_of_thought)
print(module_name(vote))
```

```output:exec-1773708397450-30np1
ensemble
```

### `module_name`

```python
from functools import partial
print(module_name(partial(refine, check=lambda result: None, max_retries=2)))
```

```output:exec-1773708399516-3sevv
refine(check=<function <lambda> at 0x7b80d46214e0>, max_retries=2)
```

## Signature methods

### `.hint()`

```python
print(Signature("question -> answer").hint("Be concise."))
```

```output:exec-1773708400246-hr2q7
question -> answer
  'Be concise.'
  → question  str
  ← answer    str
```

### `.note()`

```python
print(Signature("question -> answer").note(question="A factual question"))
```

```output:exec-1773708407595-g0ss1
question -> answer
  'Given `question`, produce `answer`.'
  → question  A factual question: str
  ← answer    str
```

### `.retype()`

```python
print(Signature("question -> answer").retype(answer=float))
```

```output:exec-1773708413128-q7jyf
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    float
```

### `.via()`

```python
print(Signature("question -> answer").via("reasoning", note="Think first"))
```

```output:exec-1773708414188-y0kst
question -> reasoning -> answer
  'Given `question`, produce `answer`.'
  → question   str
  · reasoning  Think first: str
  ← answer     str
```

### `.add()`

```python
print(Signature("question -> answer").add("confidence", float))
```

```output:exec-1773708418323-yvxj3
question -> answer, confidence
  'Given `question`, produce `answer`.'
  → question    str
  ← answer      str
  ← confidence  float
```

### `.remove()`

```python
print(Signature("question -> answer").via("reasoning").remove("reasoning"))
```

```output:exec-1773708420684-je6uw
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    str
```

### `.with_examples()`

```python
print(Signature("question -> answer").with_examples([{"question": "Q", "answer": "A"}]))
```

```output:exec-1773708421993-74oz8
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    str
  (1 examples)
```

### `.dump_state()` / `.load_state()`

```python
sig2 = Signature("question -> answer").note(answer="Short answer")
state = sig2.dump_state()
print(Signature.load_state(state))
```

```output:exec-1773708426037-jc5jv
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    Short answer: str
```

## Model methods

### `.compile()`

```python
model = Model(inputs=question, outputs=answer).compile(optimizer="auto_prompt", meta_lm="gpt-4.1")
print(model.compile_config)
```

```output:exec-1773708435118-s34vr
{'optimizer': 'auto_prompt', 'meta_lm': 'gpt-4.1'}
```

### `.fit()`

```python
model.fit([{"question": "Q", "answer": "A"}])
print(model.training_examples)
```

```output:exec-1773708440720-jk021
  question answer
0        Q      A
```

### `.calls()`

```python
print(model.calls())
```

```output:exec-1773708441895-cbzp0
[LayerCall(id=4, layer_name='answer', layer_type='ReAct', inputs=(Symbol(name='question', type=str, role='input'),), config={'module': 'react'}, layer=ReAct(outputs=['answer']), outputs=(Symbol(name='trajectory', type=list[str], role='trace'), Symbol(name='answer', type=str, role='public')))]
```

### `.layer_signatures()`

```python
for call, layer_sig in model.layer_signatures():
    print(call.layer_type, "=>", layer_sig.formula)
```

```output:exec-1773708452616-6d5qh
ReAct => question -> reasoning -> answer
```

### `.summary()`

```python
model.summary()
```

```output:exec-1773708453802-bb9va
Model: "model"
Inputs:
  - question: str
Outputs:
  - answer: str (public)
Graph:
  - [4] ReAct(trajectory, answer) <- question, module=react
Compile: {'optimizer': 'auto_prompt', 'meta_lm': 'gpt-4.1'}
Examples: 1 rows
Out[49]: 'Model: "model"\nInputs:\n  - question: str\nOutputs:\n  - answer: str (public)\nGraph:\n  - [4] ReAct(trajectory, answer) <- question, module=react\nCompile: {\'optimizer\': \'auto_prompt\', \'meta_lm\': \'gpt-4.1\'}\nExamples: 1 rows'
```

