# Guided lessons

Run commands from the repository root with `uv run python
src/agentic_ai_tutorials/<lesson-file>.py`. Each program has a deterministic
offline path, so you can focus on the framework before configuring a model.

## 1. LangChain and LangGraph foundations

**File:** [`lesson_01_foundations.py`](../src/agentic_ai_tutorials/lesson_01_foundations.py)

Start with LangChain Expression Language (LCEL): a prompt, a runnable model
step, and an output parser compose into one runnable pipeline:

```text
input -> PromptTemplate -> model/runnable -> StrOutputParser -> text
```

In the offline demo, a local runnable stands in for the model. The same chain
accepts a live ChatOpenAI model when run with `--live`. Next, inspect the graph:
`make_plan` and `write_answer` are nodes; edges define the order; `LessonState`
is the shared data passed between them.

```powershell
uv run python src/agentic_ai_tutorials/lesson_01_foundations.py
uv run python src/agentic_ai_tutorials/lesson_01_foundations.py --live
```

**Try it:** change the topic, add a state field, or insert a graph node that
revises the plan. Keep each node focused on one state update.

## 2. Tools and agents

**File:** [`lesson_02_tools_agent.py`](../src/agentic_ai_tutorials/lesson_02_tools_agent.py)

The `@tool` decorator exposes a typed function and its docstring as a tool
schema. First call the local tool directly. With `--live`, LangChain's
`create_agent` lets the model decide whether to call that tool and incorporates
the tool result into its response. Do not treat model-generated tool arguments
as trusted input in a real application.

**Try it:** add a product, validate unknown product names, then ask for a
comparison. Live mode sends prompts to the configured model provider and may
incur usage charges.

## 3. Conditional routing

**File:** [`lesson_03_routing.py`](../src/agentic_ai_tutorials/lesson_03_routing.py)

The `classify` node writes a route into state. `add_conditional_edges` uses that
value to select the technical or general response node. The router explicitly
rejects unknown route values instead of silently choosing a fallback.

**Try it:** add a billing branch, update the state type, and include a test for
the new branch.

## 4. Bounded review loop

**File:** [`lesson_04_bounded_loop.py`](../src/agentic_ai_tutorials/lesson_04_bounded_loop.py)

The graph drafts, reviews, and conditionally loops back for a revision. The
attempt counter and `MAX_ATTEMPTS` cap ensure the workflow can terminate even
when a reviewer never approves. The invocation also sets a graph recursion
limit as a second guard.

**Try it:** change the review rule so it approves only the third attempt. Then
make it never approve and verify the graph still stops at the configured cap.

## 5. Checkpointing

**File:** [`lesson_05_checkpointing.py`](../src/agentic_ai_tutorials/lesson_05_checkpointing.py)

An `InMemorySaver` stores graph state by `thread_id`. The second invocation
uses the same thread and continues from the saved counter. This checkpointer
is for a single process only: its state disappears when the process exits.

**Try it:** invoke with a different `thread_id` and compare the counters. For
durable or multi-instance workloads, use a persistent checkpointer and manage
database access and retention deliberately.

## 6. Human approval

**File:** [`lesson_06_human_approval.py`](../src/agentic_ai_tutorials/lesson_06_human_approval.py)

`interrupt()` pauses after creating a draft. The command-line lesson asks for
approval and resumes the same checkpointed thread with `Command(resume=...)`.
Approval continues to the send-status node; rejection marks it cancelled.
Interrupts require a checkpointer and stable `thread_id`. Do not swallow
LangGraph's interrupt exception with a broad `except`.

**Try it:** inspect the paused result, then change the approval payload to
include a reason or an edited draft.

## 7. Supervisor and specialists

**File:** [`lesson_07_supervisor.py`](../src/agentic_ai_tutorials/lesson_07_supervisor.py)

A supervisor routes shared state through researcher, writer, and fact-checker
nodes. The `notes` field uses `operator.add` so each specialist appends to the
workflow log instead of replacing it. The review count bounds the workflow.
These are deterministic stand-ins; replace node bodies with real model calls
only after defining tool permissions, validation, and stopping conditions.

**Try it:** add an editor specialist and route to it only when the fact-checker
requests a revision.

## Next topics

After these examples, extend the same graphs with streaming, structured model
outputs, retrieval, tracing/evaluation, and a durable store for data shared
across threads. Live model examples prefer OpenAI and fall back to Groq when
both providers are configured; configure `OPENAI_API_KEY` and `OPENAI_MODEL`,
and optionally `GROQ_API_KEY` and `GROQ_MODEL`, in `.env`. Keep provider
credentials in ignored local environment files; never paste live credentials
into source code or tutorial fixtures.
