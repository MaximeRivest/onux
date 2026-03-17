---
title: "Custom Modules"
---

# Vignette 6: Create your own modules

This vignette shows how to extend Onux at the right level.

There are two main extension points:

1. **Module functions** — custom execution strategies for one signature
2. **Layers** — custom symbolic graph nodes for Keras-style DAGs

A good rule of thumb:

- if you want to change **how one signature runs**, write a **module function**
- if you want a reusable **graph node**, write a **layer**

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

## 1. Custom module function: LLM self-consistency

A module function is just:

```python
(sig, inputs, *, lm, adapter) -> dict
```

Here is a custom module that samples several answers, then picks one.

```python
from collections import Counter
from functools import partial
from onux import Signature, predict, module_name

def self_consistency(sig, inputs, *, lm, adapter, n=5):
    candidates = []
    for _ in range(n):
        candidates.append(predict(sig, inputs, lm=lm, adapter=adapter))

    # Tiny demo voting rule: pick the most frequent serialized output.
    keys = [tuple(sorted(item.items())) for item in candidates]
    winner, _ = Counter(keys).most_common(1)[0]
    return dict(winner)

vote5 = partial(self_consistency, n=5)
print(module_name(vote5))
```

```output:exec-1773709262278-2igxu
self_consistency(n=5)
```

This is a **custom LLM module**:

- same signature in
- same outputs out
- different execution strategy inside

## 2. Custom module function: generate → lint → fix

This one mixes LLM generation with local code.

```python
from onux import Signature, predict

def lint_and_fix(sig, inputs, *, lm, adapter, lint, max_rounds=3):
    code_sig = sig.via("code", note="Python code that solves the problem")
    state = dict(inputs)
    result = {}

    for _ in range(max_rounds):
        result = predict(code_sig, state, lm=lm, adapter=adapter)
        code = result.get("code", "")
        errors = lint(code)
        if not errors:
            return {k: result.get(k) for k in sig.output_fields}
        state["lint_errors"] = errors

    return {k: result.get(k) for k in sig.output_fields}
```

```output:exec-1773709346361-5e1mw
```

This is a **custom code module**.

## 3. Custom layer: one-shot LLM layer

If you want a reusable symbolic node in a graph, subclass `Layer`.

```python
from onux import Input, Layer, Model

class Translate(Layer):
    def default_output_name(self):
        return "translation"

text = Input("text")
translation = Translate(
    hint="Translate the input to French.",
    lm="gpt-4o",
)(text)

translator = Model(inputs=text, outputs=translation, name="translator")
translator.summary()
```

```output:exec-1773709387284-ptrwl
Model: "translator"
Inputs:
  - text: str
Outputs:
  - translation: str (public)
Graph:
  - [1] Translate(translation) <- text, module=predict
Out[3]: 'Model: "translator"\nInputs:\n  - text: str\nOutputs:\n  - translation: str (public)\nGraph:\n  - [1] Translate(translation) <- text, module=predict'
```

This creates a new symbolic layer that behaves like built-ins such as `Generate`.

## 4. Custom layer: local code / rule-based layer

A local code layer is still just a symbolic node in the graph.

```python
from onux import FieldSpec, Layer, Input, Model

class ExtractNumbers(Layer):
    module_name = "extract_numbers"

    def default_output_decl(self):
        return ("numbers", list[float])

receipt_text = Input("receipt_text")
numbers = ExtractNumbers()(receipt_text)
number_model = Model(inputs=receipt_text, outputs=numbers, name="number_extractor")
number_model.summary()
```

```output:exec-1773709408049-gj29t
Model: "number_extractor"
Inputs:
  - receipt_text: str
Outputs:
  - numbers: list[float] (public)
Graph:
  - [2] ExtractNumbers(numbers) <- receipt_text, module=extract_numbers
Out[4]: 'Model: "number_extractor"\nInputs:\n  - receipt_text: str\nOutputs:\n  - numbers: list[float] (public)\nGraph:\n  - [2] ExtractNumbers(numbers) <- receipt_text, module=extract_numbers'
```

This is a **custom code layer**. At compile/runtime, your executor can map
`module_name="extract_numbers"` to a real Python function.

## 5. Custom layer: classical ML model

You can also represent a local ML model as a symbolic layer.

```python
from onux import Input, Layer, Model

class SklearnClassifier(Layer):
    module_name = "sklearn_predict"

    def default_output_decl(self):
        return (("label", str), ("confidence", float))

text = Input("text")
label, confidence = SklearnClassifier(model_path="spam.joblib")(text)
classifier = Model(inputs=text, outputs=[label, confidence], name="spam_classifier")
classifier.summary()
```

```output:exec-1773709428788-mdokc
Model: "spam_classifier"
Inputs:
  - text: str
Outputs:
  - label: str (public)
  - confidence: float (public)
Graph:
  - [3] SklearnClassifier(label, confidence) <- text, module=sklearn_predict
Out[5]: 'Model: "spam_classifier"\nInputs:\n  - text: str\nOutputs:\n  - label: str (public)\n  - confidence: float (public)\nGraph:\n  - [3] SklearnClassifier(label, confidence) <- text, module=sklearn_predict'
```

This is a **custom ML layer**.

The graph only needs to know:

- input fields
- output fields
- config (`model_path`, etc.)

The runtime can later load the actual sklearn object.

## 6. Custom layer: deep learning encoder

Same idea for deep learning.

```python
from onux import Input, Layer, Model

class Encoder(Layer):
    module_name = "torch_encoder"

    def default_output_decl(self):
        return ("embedding", list[float])

text = Input("text")
embedding = Encoder(model_name="bge-small")(text)
embedder = Model(inputs=text, outputs=embedding, name="embedder")
embedder.summary()
```

```output:exec-1773709432162-0h1jd
Model: "embedder"
Inputs:
  - text: str
Outputs:
  - embedding: list[float] (public)
Graph:
  - [4] Encoder(embedding) <- text, module=torch_encoder
Out[6]: 'Model: "embedder"\nInputs:\n  - text: str\nOutputs:\n  - embedding: list[float] (public)\nGraph:\n  - [4] Encoder(embedding) <- text, module=torch_encoder'
```

This is a **custom deep learning layer**.

The important point: Onux graphs stay symbolic. The runtime decides whether
`torch_encoder` means PyTorch, JAX, ONNX, or something else.

## 7. Custom layer with hidden outputs

You can define your own hidden and trace outputs too.

```python
from onux import FieldSpec, Input, Layer, Model

class CritiqueAndAnswer(Layer):
    module_name = "critique_and_answer"
    hidden_outputs = (
        FieldSpec("critique", str, "Internal critique", kind="hidden"),
    )
    trace_outputs = (
        FieldSpec("drafts", list[str], "All draft attempts", kind="trace"),
    )

    def default_output_decl(self):
        return "answer"

question = Input("question")
critique, drafts, answer = CritiqueAndAnswer(
    expose=("critique", "drafts"),
)(question)

debug_model = Model(inputs=question, outputs=[critique, drafts, answer], name="debug_model")
debug_model.summary()
```

```output:exec-1773709437150-slpfg
Model: "debug_model"
Inputs:
  - question: str
Outputs:
  - critique: str (hidden)
  - drafts: list[str] (trace)
  - answer: str (public)
Graph:
  - [5] CritiqueAndAnswer(critique, drafts, answer) <- question, module=critique_and_answer
Out[7]: 'Model: "debug_model"\nInputs:\n  - question: str\nOutputs:\n  - critique: str (hidden)\n  - drafts: list[str] (trace)\n  - answer: str (public)\nGraph:\n  - [5] CritiqueAndAnswer(critique, drafts, answer) <- question, module=critique_and_answer'
```

## 8. Build a richer DAG from custom layers

Here is a graph that mixes custom LLM, ML, deep learning, and code layers.

```python
question = Input("question")

translation = Translate(lm="gpt-4o")(question)
label, confidence = SklearnClassifier(model_path="intent.joblib")(translation)
embedding = Encoder(model_name="bge-small")(translation)
numbers = ExtractNumbers()(translation)

final_answer = Translate(
    output="final_answer",
    hint="Answer using the translation, label, embedding, and numbers.",
)([translation, label, confidence, embedding, numbers])

hybrid_model = Model(
    inputs=question,
    outputs=[final_answer, label, confidence],
    name="hybrid_pipeline",
)
hybrid_model.summary()
```

```output:exec-1773709470284-l6mp5
Model: "hybrid_pipeline"
Inputs:
  - question: str
Outputs:
  - final_answer: str (public)
  - label: str (public)
  - confidence: float (public)
Graph:
  - [6] Translate(translation) <- question, module=predict
  - [7] SklearnClassifier(label, confidence) <- translation, module=sklearn_predict
  - [8] Encoder(embedding) <- translation, module=torch_encoder
  - [9] ExtractNumbers(numbers) <- translation, module=extract_numbers
  - [10] Translate(final_answer) <- translation, label, confidence, embedding, numbers, module=predict
Out[8]: 'Model: "hybrid_pipeline"\nInputs:\n  - question: str\nOutputs:\n  - final_answer: str (public)\n  - label: str (public)\n  - confidence: float (public)\nGraph:\n  - [6] Translate(translation) <- question, module=predict\n  - [7] SklearnClassifier(label, confidence) <- translation, module=sklearn_predict\n  - [8] Encoder(embedding) <- translation, module=torch_encoder\n  - [9] ExtractNumbers(numbers) <- translation, module=extract_numbers\n  - [10] Translate(final_answer) <- translation, label, confidence, embedding, numbers, module=predict'
```

## 9. Inspect the inner signatures

Every custom layer still builds an inner signature.

```python
for call, sig in hybrid_model.layer_signatures():
    print(call.layer_type, "=>", sig.formula)
```

```output:exec-1773709473778-e8kxl
Translate => question -> translation
SklearnClassifier => translation -> label, confidence
Encoder => translation -> embedding
ExtractNumbers => translation -> numbers
Translate => translation, label, confidence, embedding, numbers -> final_answer
```

```python

```

## 10. Design advice

**Use a module function when:**

- you wrap one signature in retries, voting, tool use, linting, or validation
- you do not need a symbolic graph node
- you want the cleanest possible extension point

**Use a layer when:**

- you want a reusable DAG node
- you want Keras-style graph wiring
- you want to mix the component with other layers in a `Model`

**Use a model when:**

- your custom behavior is really a reusable subgraph
- you want to package several layers as one bigger building block

## Takeaway

Onux is extensible in a very direct way:

- **LLM** → custom module function or custom `Layer`
- **ML** → custom `Layer`
- **deep learning** → custom `Layer`
- **code** → custom module function or custom `Layer`

You do not need a plugin system to get started.
A small function or a small subclass is enough.
