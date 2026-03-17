---
title: "Graph Patterns"
---

# Vignette 5: Graph patterns

This vignette shows larger Keras-style DAGs.

```python
import os, sys
sys.path.insert(0, os.path.abspath("src"))
```

## ReAct agent with tools

```python
from onux import Input, Model
from onux.layers import ReAct

question = Input("question")
answer = ReAct(
    "answer",
    tools=[print],
    max_iters=5,
    lm="claude-3-5-sonnet",
)(question)

agent_model = Model(inputs=question, outputs=answer, name="react_agent")
agent_model.summary()
```

```output:exec-1773708910049-m3f3h
Model: "react_agent"
Inputs:
  - question: str
Outputs:
  - answer: str (public)
Graph:
  - [1] ReAct(answer) <- question, module=react
Out[1]: 'Model: "react_agent"\nInputs:\n  - question: str\nOutputs:\n  - answer: str (public)\nGraph:\n  - [1] ReAct(answer) <- question, module=react'
```

## Critique and refine

```python
from onux.layers import Generate

question = Input("question")
context = Input("context")

draft = Generate("draft", lm="gpt-4o-mini")([question, context])
critique = Generate("critique", lm="gpt-4o", temperature=0.0)([question, draft])
final_answer = Generate("final_answer", lm="o1", thinking=True)([question, context, draft, critique])

refinement_model = Model(inputs=[question, context], outputs=final_answer, name="refinement_model")
refinement_model.summary()
```

```output:exec-1773708917042-rwjz8
Model: "refinement_model"
Inputs:
  - question: str
  - context: str
Outputs:
  - final_answer: str (public)
Graph:
  - [2] Generate(draft) <- question, context, module=predict
  - [3] Generate(critique) <- question, draft, module=predict
  - [4] Generate(final_answer) <- question, context, draft, critique, module=predict
Out[2]: 'Model: "refinement_model"\nInputs:\n  - question: str\n  - context: str\nOutputs:\n  - final_answer: str (public)\nGraph:\n  - [2] Generate(draft) <- question, context, module=predict\n  - [3] Generate(critique) <- question, draft, module=predict\n  - [4] Generate(final_answer) <- question, context, draft, critique, module=predict'
```

## Multi-hop RAG

```python
question = Input("question")
initial_context = ReAct("initial_context", tools=[print], max_iters=2)(question)
missing_info_query = Generate("search_query")([question, initial_context])
deep_context = Generate(("deep_context", list[str]))(missing_info_query)
answer = Generate("answer")([question, initial_context, deep_context])

multi_hop_rag = Model(inputs=question, outputs=answer, name="multi_hop_rag")
multi_hop_rag.summary()
```

```output:exec-1773708919063-olihn
Model: "multi_hop_rag"
Inputs:
  - question: str
Outputs:
  - answer: str (public)
Graph:
  - [5] ReAct(initial_context) <- question, module=react
  - [6] Generate(search_query) <- question, initial_context, module=predict
  - [7] Generate(deep_context) <- search_query, module=predict
  - [8] Generate(answer) <- question, initial_context, deep_context, module=predict
Out[3]: 'Model: "multi_hop_rag"\nInputs:\n  - question: str\nOutputs:\n  - answer: str (public)\nGraph:\n  - [5] ReAct(initial_context) <- question, module=react\n  - [6] Generate(search_query) <- question, initial_context, module=predict\n  - [7] Generate(deep_context) <- search_query, module=predict\n  - [8] Generate(answer) <- question, initial_context, deep_context, module=predict'
```

## Mixture of experts

```python
task = Input("task")

anthropic_take = Generate("expert_a", lm="claude-3-opus")(task)
openai_take = Generate("expert_b", lm="gpt-4o")(task)
google_take = Generate("expert_c", lm="gemini-1.5-pro")(task)

consensus = Generate("consensus_solution", temperature=0.2)([
    task,
    anthropic_take,
    openai_take,
    google_take,
])

moe_model = Model(inputs=task, outputs=consensus, name="moe_model")
moe_model.summary()
```

```output:exec-1773708921388-f3pvu
Model: "moe_model"
Inputs:
  - task: str
Outputs:
  - consensus_solution: str (public)
Graph:
  - [9] Generate(expert_a) <- task, module=predict
  - [10] Generate(expert_b) <- task, module=predict
  - [11] Generate(expert_c) <- task, module=predict
  - [12] Generate(consensus_solution) <- task, expert_a, expert_b, expert_c, module=predict
Out[4]: 'Model: "moe_model"\nInputs:\n  - task: str\nOutputs:\n  - consensus_solution: str (public)\nGraph:\n  - [9] Generate(expert_a) <- task, module=predict\n  - [10] Generate(expert_b) <- task, module=predict\n  - [11] Generate(expert_c) <- task, module=predict\n  - [12] Generate(consensus_solution) <- task, expert_a, expert_b, expert_c, module=predict'
```

## Map-reduce over document chunks

```python
from onux.layers import Map

long_document = Input("long_document")
chunks = Generate(("chunks", list[str]))(long_document)
chunk_summaries = Map(Generate("summary"))(chunks)
executive_summary = Generate("executive_summary")([long_document, chunk_summaries])

map_reduce_model = Model(inputs=long_document, outputs=executive_summary, name="map_reduce_model")
map_reduce_model.summary()
```

```output:exec-1773708923879-wcaoq
Model: "map_reduce_model"
Inputs:
  - long_document: str
Outputs:
  - executive_summary: str (public)
Graph:
  - [13] Generate(chunks) <- long_document, module=predict
  - [14] Map(summary) <- chunks, module=map
  - [15] Generate(executive_summary) <- long_document, summary, module=predict
Out[5]: 'Model: "map_reduce_model"\nInputs:\n  - long_document: str\nOutputs:\n  - executive_summary: str (public)\nGraph:\n  - [13] Generate(chunks) <- long_document, module=predict\n  - [14] Map(summary) <- chunks, module=map\n  - [15] Generate(executive_summary) <- long_document, summary, module=predict'
```

## Research pipeline

```python
question = Input("question", type=str)
constraints = Input("constraints", type=list[str])

search_plan, search_terms = Generate(
    outputs=(("search_plan", str), ("search_terms", list[str])),
    lm="gpt-4o",
)([question, constraints])

notes, sources = ReAct(
    outputs=(("notes", str), ("sources", list[str])),
    tools=[print],
    max_iters=6,
    lm="claude-3-5-sonnet",
)([question, constraints, search_terms])

first_draft = Generate("first_draft", thinking=True, lm="gpt-4o")(
    [question, constraints, search_plan, notes, sources]
)

critique = Generate("critique", lm="gpt-4o", temperature=0.0)(
    [question, constraints, first_draft, sources]
)

final_answer, confidence = Generate(
    outputs=(("final_answer", str), ("confidence", float)),
    lm="o1",
    thinking=True,
)([question, constraints, notes, first_draft, critique, sources])

research_model = Model(
    inputs=[question, constraints],
    outputs=[final_answer, confidence, sources],
    name="research_answering_pipeline",
)
research_model.summary()
```

```output:exec-1773708925913-4ocjs
Model: "research_answering_pipeline"
Inputs:
  - question: str
  - constraints: list[str]
Outputs:
  - final_answer: str (public)
  - confidence: float (public)
  - sources: list[str] (public)
Graph:
  - [16] Generate(search_plan, search_terms) <- question, constraints, module=predict
  - [17] ReAct(notes, sources) <- question, constraints, search_terms, module=react
  - [18] Generate(first_draft) <- question, constraints, search_plan, notes, sources, module=predict
  - [19] Generate(critique) <- question, constraints, first_draft, sources, module=predict
  - [20] Generate(final_answer, confidence) <- question, constraints, notes, first_draft, critique, sources, module=predict
Out[6]: 'Model: "research_answering_pipeline"\nInputs:\n  - question: str\n  - constraints: list[str]\nOutputs:\n  - final_answer: str (public)\n  - confidence: float (public)\n  - sources: list[str] (public)\nGraph:\n  - [16] Generate(search_plan, search_terms) <- question, constraints, module=predict\n  - [17] ReAct(notes, sources) <- question, constraints, search_terms, module=react\n  - [18] Generate(first_draft) <- question, constraints, search_plan, notes, sources, module=predict\n  - [19] Generate(critique) <- question, constraints, first_draft, sources, module=predict\n  - [20] Generate(final_answer, confidence) <- question, constraints, notes, first_draft, critique, sources, module=predict'
```

## Text-to-SQL with audit branch

```python
from onux.layers import ChainOfThought, ExecuteSQL

user_query = Input("user_query", type=str)
db_schema = Input("db_schema", type=str)
dialect = Input("dialect", type=str)

sql_query, sql_rationale = ChainOfThought(
    outputs=(("sql_query", str), ("sql_rationale", str)),
    lm="gpt-4o",
)([user_query, db_schema, dialect])

rows = ExecuteSQL("rows")(sql_query)
answer, confidence = Generate(
    outputs=(("answer", str), ("confidence", float)),
    lm="claude-3-5-sonnet",
    temperature=0.0,
)([user_query, rows, sql_rationale])

audit_report = ReAct(
    "audit_report",
    tools=[print],
    max_iters=3,
    lm="gpt-4o",
)([db_schema, dialect, sql_query, rows])

safe_answer = Generate("safe_answer", lm="o1-mini", thinking=True)(
    [user_query, answer, confidence, audit_report]
)

text_to_sql_model = Model(
    inputs=[user_query, db_schema, dialect],
    outputs=[safe_answer, sql_query, audit_report],
    name="text_to_sql_pipeline",
)
text_to_sql_model.summary()
```

```output:exec-1773708930930-txfms
Model: "text_to_sql_pipeline"
Inputs:
  - user_query: str
  - db_schema: str
  - dialect: str
Outputs:
  - safe_answer: str (public)
  - sql_query: str (public)
  - audit_report: str (public)
Graph:
  - [21] ChainOfThought(sql_query, sql_rationale) <- user_query, db_schema, dialect, module=chain_of_thought
  - [22] ExecuteSQL(rows) <- sql_query, module=execute_sql
  - [23] Generate(answer, confidence) <- user_query, rows, sql_rationale, module=predict
  - [24] ReAct(audit_report) <- db_schema, dialect, sql_query, rows, module=react
  - [25] Generate(safe_answer) <- user_query, answer, confidence, audit_report, module=predict
Out[7]: 'Model: "text_to_sql_pipeline"\nInputs:\n  - user_query: str\n  - db_schema: str\n  - dialect: str\nOutputs:\n  - safe_answer: str (public)\n  - sql_query: str (public)\n  - audit_report: str (public)\nGraph:\n  - [21] ChainOfThought(sql_query, sql_rationale) <- user_query, db_schema, dialect, module=chain_of_thought\n  - [22] ExecuteSQL(rows) <- sql_query, module=execute_sql\n  - [23] Generate(answer, confidence) <- user_query, rows, sql_rationale, module=predict\n  - [24] ReAct(audit_report) <- db_schema, dialect, sql_query, rows, module=react\n  - [25] Generate(safe_answer) <- user_query, answer, confidence, audit_report, module=predict'
```
