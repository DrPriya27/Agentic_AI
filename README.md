# Agentic AI: LangChain and LangGraph tutorials

The examples emphasize explicit state, inspectable workflows, and bounded
execution. Lesson 1 and the graph lessons run locally without an API key.

## Set up the environment

Install [uv](https://docs.astral.sh/uv/), then run these commands from the
repository root in PowerShell. `uv` will use or download the pinned Python 3.13:

```powershell
uv sync
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

The `.env` file is optional for offline lessons. Live examples try OpenAI first
when `OPENAI_API_KEY` and `OPENAI_MODEL` are configured; if that request fails,
they retry with Groq when `GROQ_API_KEY` and `GROQ_MODEL` are configured. If
only Groq is configured, it is used directly. Never commit `.env` or share
provider keys. Python 3.13 is pinned for broad Windows wheel support.

## Run a lesson

```powershell
uv run python src/agentic_ai_tutorials/lesson_01_foundations.py
uv run python src/agentic_ai_tutorials/lesson_01_foundations.py --live
uv run python src/agentic_ai_tutorials/lesson_02_tools_agent.py
uv run python src/agentic_ai_tutorials/lesson_02_tools_agent.py --live
uv run python src/agentic_ai_tutorials/lesson_03_routing.py
uv run python src/agentic_ai_tutorials/lesson_04_bounded_loop.py
uv run python src/agentic_ai_tutorials/lesson_05_checkpointing.py
uv run python src/agentic_ai_tutorials/lesson_06_human_approval.py
uv run python src/agentic_ai_tutorials/lesson_07_supervisor.py
```

Live model requests can incur provider charges and send prompts to the
configured provider. The default Lesson 2 run only demonstrates the local tool.

## Tutorial path

| Lesson | Topic | What you will build |
|---|---|---|
| 1 | LangChain and LangGraph foundations | An LCEL chain and a two-node graph |
| 2 | Tools and agents | A typed local tool and an optional `create_agent` loop |
| 3 | Conditional routing | A message router with distinct graph branches |
| 4 | Bounded loops | A review/retry cycle with an explicit attempt limit |
| 5 | Checkpointing | A thread whose state is saved between graph invocations |
| 6 | Human approval | A graph that pauses, then resumes or cancels |
| 7 | Supervisor workflow | A supervisor coordinating researcher, writer, and reviewer nodes |

Use LangChain for straightforward model pipelines and its prebuilt agent
abstraction. Use LangGraph when you need explicit state, branching, cycles,
human pauses, or checkpointed execution. The examples use deterministic local
nodes so graph behavior is testable without model calls; Lesson 1 and Lesson
2 contact a model provider only when run with `--live`.

For the guided walkthrough, concepts, and exercises for all seven lessons, see
[`tutorials/README.md`](tutorials/README.md).

## Run the tests

```powershell
uv run pytest
```