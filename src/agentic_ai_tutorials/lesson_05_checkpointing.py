"""Lesson 5: persist graph state for a thread across invocations."""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    count: int
    increment: int


def build_graph():
    workflow = StateGraph(CounterState)
    workflow.add_node(
        "increment",
        lambda state: {"count": state["count"] + state["increment"]},
    )
    workflow.add_edge(START, "increment")
    workflow.add_edge("increment", END)
    return workflow.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = build_graph()
    config = {"configurable": {"thread_id": "lesson-05-demo"}}
    first = app.invoke({"count": 0, "increment": 1}, config)
    second = app.invoke({"increment": 1}, config)
    print(f"Count after first run: {first['count']}")
    print(f"Count after resuming the same thread: {second['count']}")


if __name__ == "__main__":
    main()
