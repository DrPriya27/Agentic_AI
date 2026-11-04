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

In the offline demo, a local runnable stands in for the model. With `--live`,
the chain uses the configured OpenAI model first and retries with Groq if that
request fails and Groq is configured. Next, inspect the graph: `make_plan` and
`write_answer` are nodes; edges define the order; `LessonState` is the shared
data passed between them.

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

## How a harness sits on LangGraph

The [product split in the LangChain docs](https://docs.langchain.com/oss/python/concepts/products)
separates frameworks, runtimes, and harnesses:

```text
Harness: Deep Agents
    Framework: LangChain agents
        Runtime: LangGraph
            State, checkpoints, streaming, interrupts, and threads
```

LangGraph is the runtime underneath an agent. It executes graphs and provides
primitives for state, checkpointing, streaming, pausing, and resuming. By
itself, it does not give a model a filesystem, planner, or subagents.

[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) is a
harness built on LangGraph. `create_deep_agent()` assembles an agent loop and
opinionated tools and middleware, such as a virtual filesystem, context
management, subagent task delegation, optional todo planning, and configurable
human approval for tools. The harness uses LangGraph interrupts and
checkpointers for pause/resume and persistence; its subagents are themselves
graphs. You still invoke or stream the returned agent. You do not need to draw
its internal graph for the common case.

The tutorials in this repository do **not** load Deep Agents. In particular,
Lesson 7's supervisor, researcher, writer, and fact-checker are hand-written
LangGraph nodes, not harness-provided subagents. Choose LangGraph when you need
to design and own the workflow graph; choose a harness when you want an
opinionated agent with common tools and middleware already wired up.

## LangGraph coverage and next topics

The following checklist distinguishes implemented examples from topics still
to explore. The source of truth for current coverage is the linked lesson;
unchecked items are learning gaps, not claims that this project implements
them.

### LangGraph overview topics

- [x] State, nodes, edges, compile, and invoke: [Lesson 1](../src/agentic_ai_tutorials/lesson_01_foundations.py).
- [x] Conditional branches: [Lesson 3](../src/agentic_ai_tutorials/lesson_03_routing.py).
- [x] A bounded loop and recursion-limit guard: [Lesson 4](../src/agentic_ai_tutorials/lesson_04_bounded_loop.py).
- [x] Checkpointed thread state: [Lesson 5](../src/agentic_ai_tutorials/lesson_05_checkpointing.py).
- [x] Human pause and resume with `interrupt()` and `Command(resume=...)`: [Lesson 6](../src/agentic_ai_tutorials/lesson_06_human_approval.py).
- [x] A hand-built supervisor workflow and an append reducer: [Lesson 7](../src/agentic_ai_tutorials/lesson_07_supervisor.py).
- [ ] Stream model tokens and graph state updates with `stream` and
  `stream_events`; current lessons use `invoke`.
- [ ] Contrast thread-scoped short-term state with long-term memory in a
  store shared across different `thread_id`s.
- [ ] Combine deterministic and real model-backed graph nodes. The current
  graph nodes are deterministic; `--live` currently demonstrates a LangChain
  chain or agent.
- [ ] Call a chat model and execute its tool from inside an explicit
  `StateGraph` node.
- [ ] Enable LangSmith tracing with `LANGSMITH_TRACING` and a user-provided
  key, then inspect an actual trace.
- [ ] Explain deployment options such as LangSmith Studio and Agent Server.
  Link to current official guidance rather than simulating a deployment.
- [x] Compare LangGraph to LangChain `create_agent` and Deep Agents: see
  [Lesson 2](../src/agentic_ai_tutorials/lesson_02_tools_agent.py) and
  [How a harness sits on LangGraph](#how-a-harness-sits-on-langgraph).

### Further LangGraph guides

- [ ] Build a hello-world graph with `MessagesState`.
- [ ] Explore message reducers such as `add_messages`; Lesson 7 only uses
  `operator.add` for an append-only notes list.
- [x] See a model select and call a tool, then use its result: the prebuilt
  agent in [Lesson 2](../src/agentic_ai_tutorials/lesson_02_tools_agent.py).
- [ ] Compare that prebuilt agent with a hand-built `StateGraph` tool-calling
  loop.
- [ ] Explore checkpoint time travel, replay, and forking beyond the
  save-and-resume example in Lesson 5.
- [ ] Demonstrate durable execution by failing and retrying a node with a
  persistent checkpointer.
- [ ] Run the Postgres checkpointer against a real `POSTGRES_URI`; the current
  checkpoint example is in-memory only.
- [ ] Build and invoke a subgraph.
- [ ] Run parallel branches with `Send` and combine their results using a
  map-reduce pattern.
- [ ] Demonstrate the Functional API with `@entrypoint` and `@task`.
- [ ] Show the recursion-limit exception for a graph whose loop has no stop,
  and contrast it with Lesson 4's explicit stop condition.
- [x] Demonstrate an interrupt with a checkpointer and resume: [Lesson 6](../src/agentic_ai_tutorials/lesson_06_human_approval.py).
- [ ] Expand interrupt coverage: keep interrupt calls in a stable order, avoid
  catching the interrupt exception broadly, and put non-idempotent side
  effects after the resume point.
- [ ] Show a larger application layout with separate state, node, and graph
  builder modules.

For official, version-current explanations, begin with the
[LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
and the linked guides there.

## Provider configuration

Live model examples prefer OpenAI and fall back to Groq when both providers
are configured. Set `OPENAI_API_KEY` and `OPENAI_MODEL`, and optionally
`GROQ_API_KEY` and `GROQ_MODEL`, in `.env`. Keep provider credentials in
ignored local environment files; never paste live credentials into source code
or tutorial fixtures.
