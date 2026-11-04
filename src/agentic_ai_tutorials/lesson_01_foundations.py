"""Lesson 1: a LangChain pipeline and a basic LangGraph workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TypedDict

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, START, StateGraph

from agentic_ai_tutorials.models import build_fallback_runnable, load_chat_models


class LessonState(TypedDict):
    topic: str
    plan: str
    answer: str


def build_chain(model: Runnable | None = None) -> Runnable:
    """Build a runnable prompt pipeline; use a local stub unless given a model."""
    prompt = PromptTemplate.from_template(
        "Explain {topic} in one concise sentence for a new developer."
    )
    if model is None:
        model = RunnableLambda(
            lambda rendered: (
                "Begin by identifying the core idea in: " + rendered.to_string()
            )
        )
    return prompt | model | StrOutputParser()


def build_graph() -> Runnable:
    """Build a small stateful workflow with two sequential nodes."""
    workflow = StateGraph(LessonState)

    def make_plan(state: LessonState) -> dict[str, str]:
        return {"plan": f"Define the key idea, then give an example of {state['topic']}."}

    def write_answer(state: LessonState) -> dict[str, str]:
        return {"answer": f"{state['plan']} Topic: {state['topic']}."}

    workflow.add_node("make_plan", make_plan)
    workflow.add_node("write_answer", write_answer)
    workflow.add_edge(START, "make_plan")
    workflow.add_edge("make_plan", "write_answer")
    workflow.add_edge("write_answer", END)
    return workflow.compile()


def _live_model() -> Runnable:
    return build_fallback_runnable(load_chat_models())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the configured model instead of the offline demonstration",
    )
    args = parser.parse_args()

    chain = build_chain(_live_model() if args.live else None)
    topic = "the difference between a chain and a graph"
    print("LangChain output:")
    print(chain.invoke({"topic": topic}))
    print("\nLangGraph state:")
    print(build_graph().invoke({"topic": topic, "plan": "", "answer": ""}))


if __name__ == "__main__":
    main()
