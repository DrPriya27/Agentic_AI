"""Lesson 4: retry a review loop, with a hard upper bound."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

MAX_ATTEMPTS = 3


class ReviewState(TypedDict):
    attempts: int
    draft: str
    approved: bool


def build_graph():
    workflow = StateGraph(ReviewState)

    def draft(state: ReviewState) -> dict[str, object]:
        attempt = state["attempts"] + 1
        return {"attempts": attempt, "draft": f"Draft revision {attempt}."}

    def review(state: ReviewState) -> dict[str, bool]:
        return {"approved": state["attempts"] >= 2}

    def next_step(state: ReviewState) -> str:
        if state["approved"]:
            return "done"
        if state["attempts"] >= MAX_ATTEMPTS:
            return "done"
        return "draft"

    workflow.add_node("draft", draft)
    workflow.add_node("review", review)
    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "review")
    workflow.add_conditional_edges(
        "review", next_step, {"draft": "draft", "done": END}
    )
    return workflow.compile()


def main() -> None:
    result = build_graph().invoke(
        {"attempts": 0, "draft": "", "approved": False},
        {"recursion_limit": MAX_ATTEMPTS * 2 + 2},
    )
    print(
        f"Approved={result['approved']} after {result['attempts']} attempt(s): "
        f"{result['draft']}"
    )


if __name__ == "__main__":
    main()
