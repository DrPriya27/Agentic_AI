"""Lesson 2: define a tool and optionally give it to a LangChain agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents.middleware import wrap_model_call
from langchain.tools import tool

from agentic_ai_tutorials.models import load_chat_models, run_with_fallback


@tool
def lookup_product_price(product: str) -> str:
    """Look up a product price in the sample catalog."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    price = prices.get(product.casefold())
    if price is None:
        return f"{product} is not in the sample catalog."
    return f"{product}: ${price:.2f}"


def build_agent():
    """Create an agent that prefers OpenAI and falls back to Groq when configured."""
    models = load_chat_models()

    from langchain.agents import create_agent

    middleware = []
    if models.fallback is not None:
        fallback_model = models.fallback

        @wrap_model_call
        def fallback_on_provider_error(request, handler):
            return run_with_fallback(
                lambda: handler(request),
                lambda: handler(request.override(model=fallback_model)),
                models.primary_name,
                models.fallback_name,
            )

        middleware.append(fallback_on_provider_error)

    return create_agent(models.primary, tools=[lookup_product_price], middleware=middleware)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="let a configured chat model choose and call the tool",
    )
    args = parser.parse_args()

    if args.live:
        result = build_agent().invoke(
            {"messages": [{"role": "user", "content": "What does a keyboard cost?"}]}
        )
        print(result["messages"][-1].content)
    else:
        print("Offline tool call:")
        print(lookup_product_price.invoke({"product": "keyboard"}))
        print("\nPass --live to let a model decide when to call this tool.")


if __name__ == "__main__":
    main()
