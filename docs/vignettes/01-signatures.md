---
title: "Signatures"
---

# Vignette 1: Signatures

A signature is the innermost building block in Onux.
It says what goes in, what comes out, and optionally what the model
should generate first as hidden scaffolding.

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

## Add one hidden field

```python
sig = Signature("question -> answer").via("reasoning", desc="Think step by step")
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

sig = Signature("question -> answer", data=train)
print(sig)
print(sig.examples)
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

## Descriptions live in the type

```python
from typing import Annotated, Literal
from onux import Description, describe_type

Rating = Annotated[float, Description("star rating")]
Sentiment = Literal["positive", "negative", "neutral"]

review = Signature("text -> sentiment: Sentiment, rating: Rating, summary", types={
    "Sentiment": Sentiment,
    "Rating": Rating,
})
print(review)
print(describe_type(Rating))
```

```output:exec-1773708510833-ug1h3
text -> sentiment, rating, summary
  'Given `text`, produce `sentiment`, `rating`, `summary`.'
  → text       str
  ← sentiment  one of: 'positive', 'negative', 'neutral'
  ← rating     star rating: float
  ← summary    str
star rating: float
```

## Update descriptions and types after the fact

```python
review = review.describe(text="Product review text", summary="One-sentence summary")
review = review.retype(summary=Annotated[str, Description("Brief summary")])
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
sig2 = sig2.add("confidence", float, desc="Confidence score")
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

## Dot shorthand

```python
df = pd.DataFrame({
    "text": ["Great!"],
    "category": ["electronics"],
    "sentiment": ["positive"],
})

sig4 = Signature(". -> sentiment", data=df)
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
