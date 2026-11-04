"""Lesson 7: coordinate specialists with a simple supervisor graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

MAX_REVIEW_ATTEMPTS = 2


class CrewState(TypedDict):
    task: str
    research: str
    draft: str
    approved: bool
    review_count: int
    next_agent: str
    notes: Annotated[list[str], operator.add]


def build_graph():
    workflow = StateGraph(CrewState)

    def supervisor(state: CrewState) -> dict[str, object]:
        if not state["research"]:
            next_agent = "researcher"
        elif state["approved"] or state["review_count"] >= MAX_REVIEW_ATTEMPTS:
            next_agent = "done"
        elif not state["draft"] or not state["approved"]:
            next_agent = "writer"
        else:
            next_agent = "done"
        return {"next_agent": next_agent, "notes": [f"Supervisor -> {next_agent}"]}

    def select_next(state: CrewState) -> str:
        return state["next_agent"]

    def researcher(state: CrewState) -> dict[str, object]:
        return {
            "research": f"Key points about {state['task']}.",
            "notes": ["Researcher gathered key points."],
        }

    def writer(state: CrewState) -> dict[str, object]:
        return {
            "draft": f"Draft based on: {state['research']}",
            "notes": ["Writer prepared a draft."],
        }

    def fact_checker(state: CrewState) -> dict[str, object]:
        attempts = state["review_count"] + 1
        approved = attempts >= MAX_REVIEW_ATTEMPTS
        return {
            "review_count": attempts,
            "approved": approved,
            "notes": [
                "Fact checker approved."
                if approved
                else "Fact checker requested revision."
            ],
        }

    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)
    workflow.add_node("fact_checker", fact_checker)
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        select_next,
        {
            "researcher": "researcher",
            "writer": "writer",
            "done": END,
        },
    )
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "supervisor")
    return workflow.compile()


def main() -> None:
    result = build_graph().invoke(
        {
            "task": "LangGraph production patterns",
            "research": "",
            "draft": "",
            "approved": False,
            "review_count": 0,
            "next_agent": "",
            "notes": [],
        },
        {"recursion_limit": 20},
    )
    print(result["draft"])
    print("\nWorkflow log:")
    print("\n".join(result["notes"]))


if __name__ == "__main__":
    main()
