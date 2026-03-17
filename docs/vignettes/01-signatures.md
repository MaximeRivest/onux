---
title: "Signatures"
---

# Vignette 1: Signatures

A signature is the innermost building block in Onux.
It says what goes in, what comes out, and optionally what the model
should generate first as hidden scaffolding.

The formula itself describes the field names and flow. Types, notes, and
additional hidden or output fields are added with methods like `.type()`,
`.note()`, `.via()`, and `.add()`.

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

## The smallest signature

```python
from onux import Signature

sig = Signature("question -> answer")
print(sig)
```

```output:exec-1773708470195-usvsz
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    str
```

## A fuller builder-style signature

In practice, signatures are often built in a few steps: start from the
formula, then add hint text, field notes, and examples.

```python
sig = (
    Signature("question -> answer")
    .hint("Answer briefly and directly.")
    .note(
        question="User question",
        answer="Short factual answer",
    )
    .examples([
        {"question": "Capital of France?", "answer": "Paris"},
        {"question": "2+2?", "answer": "4"},
    ])
)
print(sig)
```

```output
question -> answer
  'Answer briefly and directly.'
  → question  User question: str
  ← answer    Short factual answer: str
  (2 examples)
```

## The same pattern with multiple fields

The same builder style works for multiple inputs, hidden fields, and outputs.

```python
sig = (
    Signature("question, context -> reasoning, evidence -> answer, confidence")
    .hint("Answer briefly, cite the strongest support, and include a confidence score.")
    .type(context=list[str], evidence=list[str], confidence=float)
    .note(
        question="User question",
        context="Relevant background passages",
        reasoning="Internal reasoning steps",
        evidence="Most relevant supporting snippets",
        answer="Short factual answer",
        confidence="Confidence score from 0 to 1",
    )
    .examples([
        {
            "question": "Capital of France?",
            "context": ["Paris is the capital of France.", "France is in Europe."],
            "reasoning": "The context explicitly names Paris as the capital.",
            "evidence": ["Paris is the capital of France."],
            "answer": "Paris",
            "confidence": 0.99,
        },
        {
            "question": "2+2?",
            "context": ["Basic arithmetic: 2 + 2 = 4."],
            "reasoning": "The arithmetic fact is directly stated.",
            "evidence": ["Basic arithmetic: 2 + 2 = 4."],
            "answer": "4",
            "confidence": 1.0,
        },
    ])
)
print(sig)
```

```output
question, context -> reasoning, evidence -> answer, confidence
  'Answer briefly, cite the strongest support, and include a confidence score.'
  → question    User question: str
  → context     Relevant background passages: list[str]
  · reasoning   Internal reasoning steps: str
  · evidence    Most relevant supporting snippets: list[str]
  ← answer      Short factual answer: str
  ← confidence  Confidence score from 0 to 1: float
  (2 examples)
```

## Add one hidden field

The formula can include one hidden stage directly, or you can add hidden
fields incrementally with `.via()`.

```python
sig = Signature("question -> answer").via("reasoning", note="Think step by step")
print(sig)
```

```output:exec-1773708477712-498po
question -> reasoning -> answer
  'Given `question`, produce `answer`.'
  → question   str
  · reasoning  Think step by step: str
  ← answer     str
```

## DataFrame-first examples

```python
import pandas as pd

train = pd.DataFrame(
    {
        "question": ["What is 2+2?", "Capital of France?"],
        "answer": ["4", "Paris"],
    }
)

sig = Signature("question -> answer", examples=train)
print(sig)
print(sig.example_data)
```

```output:exec-1773708492022-vmndn
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    str
  (2 examples)
             question answer
0        What is 2+2?      4
1  Capital of France?  Paris
```

## Notes live on fields

```python
from typing import Literal
from onux import Signature

Sentiment = Literal["positive", "negative", "neutral"]

review = (
    Signature("text -> sentiment, rating, summary")
    .type(sentiment=Sentiment, rating=float)
    .note(
        rating="star rating",
    )
)
print(review)
```

```output:exec-1773708510833-ug1h3
text -> sentiment, rating, summary
  'Given `text`, produce `sentiment`, `rating`, `summary`.'
  → text       str
  ← sentiment  one of: 'positive', 'negative', 'neutral'
  ← rating     star rating: float
  ← summary    str
```

## Update notes and types after the fact

```python
review = review.note(text="Product review text", summary="Brief summary")
review = review.type(summary=str)
print(review)
```

```output:exec-1773708521081-itqnf
text -> sentiment, rating, summary
  'Given `text`, produce `sentiment`, `rating`, `summary`.'
  → text       Product review text: str
  ← sentiment  one of: 'positive', 'negative', 'neutral'
  ← rating     star rating: float
  ← summary    Brief summary: str
```

## Add or remove outputs

```python
sig2 = Signature("question -> answer")
sig2 = sig2.add("confidence", float, note="Confidence score")
print(sig2)

sig3 = sig2.remove("confidence")
print(sig3)
```

```output:exec-1773708524513-up1x2
question -> answer, confidence
  'Given `question`, produce `answer`.'
  → question    str
  ← answer      str
  ← confidence  Confidence score: float
question -> answer
  'Given `question`, produce `answer`.'
  → question  str
  ← answer    str
```

## Objectives can mix rubrics and scoring functions

```python
def exact_match(example, prediction, *, signature=None):
    return float(example["answer"] == prediction["answer"])

sig3 = Signature("question -> answer").objective(
    exact_match,
    "Correct, concise, and grounded in the provided facts.",
    weights=(0.8, 0.2),
)
print(sig3)
print(sig3.objective_spec)
```

```output
question -> answer
  'Given `question`, produce `answer`.'
  objective:
    - __main__.exact_match [weight=0.8]
    - Correct, concise, and grounded in the provided facts. [weight=0.2]
  → question  str
  ← answer    str
Objective(terms=(ObjectiveTerm(kind='callable', spec=<function exact_match at ...>, weight=0.8, name=None), ObjectiveTerm(kind='rubric', spec='Correct, concise, and grounded in the provided facts.', weight=0.2, name=None)), reduce='weighted_mean')
```

## Dot shorthand

```python
df = pd.DataFrame({
    "text": ["Great!"],
    "category": ["electronics"],
    "sentiment": ["positive"],
})

sig4 = Signature(". -> sentiment", examples=df)
print(sig4)
```

```output:exec-1773708529935-62buy
text, category -> sentiment
  'Given `text`, `category`, produce `sentiment`.'
  → text       str
  → category   str
  ← sentiment  str
  (1 examples)
```

## Prompt-state round trip

```python
state = sig4.dump_state()
restored = Signature.load_state(state)
print(restored)
print(restored == sig4)
```

```output:exec-1773708539514-2dp2f
text, category -> sentiment
  'Given `text`, `category`, produce `sentiment`.'
  → text       str
  → category   str
  ← sentiment  str
  (1 examples)
True
```

## Takeaway

Use `Signature` when you want to define a task cleanly and independently of:

- execution strategy
- prompt renderer
- model choice
- optimization
