"""Lesson 3: route a request to one of two graph branches."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class SupportState(TypedDict):
    message: str
    route: str
    response: str


def classify(state: SupportState) -> dict[str, str]:
    text = state["message"].casefold()
    route = "technical" if any(word in text for word in ("bug", "error")) else "general"
    return {"route": route}


def route_message(state: SupportState) -> Literal["technical", "general"]:
    route = state["route"]
    if route == "technical":
        return "technical"
    if route == "general":
        return "general"
    raise ValueError(f"Unknown support route: {route}")


def build_graph():
    workflow = StateGraph(SupportState)
    workflow.add_node("classify", classify)
    workflow.add_node(
        "technical",
        lambda _state: {"response": "A technical support specialist will investigate."},
    )
    workflow.add_node(
        "general",
        lambda _state: {"response": "A support specialist will answer your question."},
    )
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_message,
        {"technical": "technical", "general": "general"},
    )
    workflow.add_edge("technical", END)
    workflow.add_edge("general", END)
    return workflow.compile()


def main() -> None:
    app = build_graph()
    for message in ("I found a bug in checkout", "What are your business hours?"):
        result = app.invoke({"message": message, "route": "", "response": ""})
        print(f"{result['route']}: {result['response']}")


if __name__ == "__main__":
    main()
