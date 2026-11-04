"""Lesson 6: pause at an interrupt and resume with a human decision."""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    draft: str
    approved: bool
    status: str


def build_graph():
    workflow = StateGraph(ApprovalState)
    workflow.add_node("draft", lambda _state: {"draft": "Send the project update."})

    def request_approval(state: ApprovalState) -> dict[str, object]:
        approved = interrupt({"draft": state["draft"], "question": "Approve this message?"})
        return {"approved": bool(approved)}

    workflow.add_node("request_approval", request_approval)
    workflow.add_node(
        "finish",
        lambda state: {
            "status": "sent" if state["approved"] else "cancelled"
        },
    )
    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "request_approval")
    workflow.add_edge("request_approval", "finish")
    workflow.add_edge("finish", END)
    return workflow.compile(checkpointer=InMemorySaver())


def run_approval(approved: bool) -> dict[str, object]:
    app = build_graph()
    config = {"configurable": {"thread_id": "lesson-06-approval"}}
    paused = app.invoke(
        {"draft": "", "approved": False, "status": "waiting"},
        config,
    )
    if not paused.get("__interrupt__"):
        raise RuntimeError("The graph did not pause for approval.")
    return app.invoke(Command(resume=approved), config)


def main() -> None:
    app = build_graph()
    config = {"configurable": {"thread_id": "lesson-06-demo"}}
    paused = app.invoke(
        {"draft": "", "approved": False, "status": "waiting"},
        config,
    )
    if not paused.get("__interrupt__"):
        raise RuntimeError("The graph did not pause for approval.")
    answer = input("Approve the draft? [y/N] ").strip().casefold() == "y"
    result = app.invoke(Command(resume=answer), config)
    print(f"Final status: {result['status']}")


if __name__ == "__main__":
    main()
