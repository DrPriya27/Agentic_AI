from agentic_ai_tutorials.lesson_01_foundations import build_chain, build_graph
from agentic_ai_tutorials.lesson_02_tools_agent import lookup_product_price
from agentic_ai_tutorials.lesson_03_routing import build_graph as build_routing_graph
from agentic_ai_tutorials.lesson_04_bounded_loop import (
    MAX_ATTEMPTS,
    build_graph as build_review_graph,
)
from agentic_ai_tutorials.lesson_05_checkpointing import (
    build_graph as build_checkpointed_graph,
)
from agentic_ai_tutorials.lesson_06_human_approval import run_approval
from agentic_ai_tutorials.lesson_07_supervisor import build_graph as build_supervisor_graph


def test_lesson_01_chain_and_graph_run_offline():
    chain_result = build_chain().invoke({"topic": "LangGraph"})
    assert "LangGraph" in chain_result

    graph_result = build_graph().invoke(
        {"topic": "LangGraph", "plan": "", "answer": ""}
    )
    assert "LangGraph" in graph_result["answer"]


def test_lesson_02_local_tool():
    assert lookup_product_price.invoke({"product": "keyboard"}) == "keyboard: $89.50"


def test_lesson_03_routes_technical_and_general_messages():
    app = build_routing_graph()
    technical = app.invoke({"message": "There is a bug", "route": "", "response": ""})
    general = app.invoke({"message": "Hello", "route": "", "response": ""})

    assert technical["route"] == "technical"
    assert general["route"] == "general"


def test_lesson_04_review_loop_is_bounded_and_approves():
    result = build_review_graph().invoke(
        {"attempts": 0, "draft": "", "approved": False},
        {"recursion_limit": MAX_ATTEMPTS * 2 + 2},
    )

    assert result["approved"] is True
    assert result["attempts"] <= MAX_ATTEMPTS


def test_lesson_05_checkpoint_keeps_thread_state():
    app = build_checkpointed_graph()
    config = {"configurable": {"thread_id": "test-thread"}}

    first = app.invoke({"count": 0, "increment": 1}, config)
    second = app.invoke({"increment": 1}, config)

    assert first["count"] == 1
    assert second["count"] == 2


def test_lesson_06_resumes_after_human_approval_or_rejection():
    assert run_approval(True)["status"] == "sent"
    assert run_approval(False)["status"] == "cancelled"


def test_lesson_07_supervisor_finishes_after_review():
    result = build_supervisor_graph().invoke(
        {
            "task": "agent workflows",
            "research": "",
            "draft": "",
            "approved": False,
            "review_count": 0,
            "next_agent": "",
            "notes": [],
        },
        {"recursion_limit": 20},
    )

    assert result["approved"] is True
    assert result["review_count"] == 2
    assert any("Researcher" in note for note in result["notes"])
