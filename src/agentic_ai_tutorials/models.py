"""Provider selection and explicit OpenAI-to-Groq request fallback."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from dotenv import load_dotenv
from groq import GroqError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import ModelError
from langchain_core.runnables import Runnable, RunnableLambda
from openai import OpenAIError

T = TypeVar("T")
PROVIDER_ERRORS = (ModelError, OpenAIError, GroqError)


@dataclass(frozen=True)
class ChatModels:
    primary: BaseChatModel
    primary_name: str
    fallback: BaseChatModel | None
    fallback_name: str | None


def load_chat_models() -> ChatModels:
    """Prefer configured OpenAI; use Groq when it is the only provider configured."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    groq_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")

    has_openai = bool(openai_key and openai_model)
    has_groq = bool(groq_key and groq_model)
    if not has_openai and not has_groq:
        raise RuntimeError(
            "Live mode needs OPENAI_API_KEY and OPENAI_MODEL, or "
            "GROQ_API_KEY and GROQ_MODEL, in the environment or .env."
        )

    groq = None
    if has_groq:
        from langchain_groq import ChatGroq

        groq = ChatGroq(model=groq_model, api_key=groq_key)

    if has_openai:
        from langchain_openai import ChatOpenAI

        options: dict[str, str] = {"model": openai_model, "api_key": openai_key}
        if openai_base_url:
            options["base_url"] = openai_base_url
        return ChatModels(
            primary=ChatOpenAI(**options),
            primary_name="OpenAI",
            fallback=groq,
            fallback_name="Groq" if groq else None,
        )

    if groq is None:
        raise RuntimeError("Groq configuration was present but its model could not be created.")
    return ChatModels(
        primary=groq,
        primary_name="Groq",
        fallback=None,
        fallback_name=None,
    )


def run_with_fallback(
    primary_call: Callable[[], T],
    fallback_call: Callable[[], T] | None,
    primary_name: str,
    fallback_name: str | None,
) -> T:
    """Run the primary provider, reporting and trying its configured fallback."""
    try:
        return primary_call()
    except PROVIDER_ERRORS as primary_error:
        if fallback_call is None or fallback_name is None:
            raise

        print(
            f"{primary_name} request failed ({type(primary_error).__name__}); "
            f"retrying with {fallback_name}.",
            file=sys.stderr,
        )
        try:
            return fallback_call()
        except PROVIDER_ERRORS as fallback_error:
            raise RuntimeError(
                f"{primary_name} request failed ({type(primary_error).__name__}) "
                f"and {fallback_name} fallback failed "
                f"({type(fallback_error).__name__}). Check provider credentials, "
                "model names, quota, and connectivity."
            ) from fallback_error


def build_fallback_runnable(models: ChatModels) -> Runnable:
    """Wrap model invocation so provider errors can fall back without hiding failures."""
    if models.fallback is None:
        return models.primary

    def invoke(prompt, config):
        fallback_model = models.fallback
        if fallback_model is None:
            return models.primary.invoke(prompt, config)
        return run_with_fallback(
            lambda: models.primary.invoke(prompt, config),
            lambda: fallback_model.invoke(prompt, config),
            models.primary_name,
            models.fallback_name,
        )

    return RunnableLambda(invoke)
