"""Lesson 2: define a tool and optionally give it to a LangChain agent."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain.tools import tool


@tool
def lookup_product_price(product: str) -> str:
    """Look up a product price in the sample catalog."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    price = prices.get(product.casefold())
    if price is None:
        return f"{product} is not in the sample catalog."
    return f"{product}: ${price:.2f}"


def build_agent():
    """Create a model-backed agent; this requires configured provider credentials."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not model_name:
        raise RuntimeError(
            "Live mode needs OPENAI_API_KEY and OPENAI_MODEL in the environment or .env."
        )

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    options: dict[str, str] = {"model": model_name, "api_key": api_key}
    if base_url:
        options["base_url"] = base_url
    return create_agent(ChatOpenAI(**options), tools=[lookup_product_price])


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
