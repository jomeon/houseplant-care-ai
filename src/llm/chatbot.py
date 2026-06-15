from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from src.llm.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def answer_with_context(question: str, context: str) -> str:
    llm = get_llm()
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=build_rag_user_prompt(context, question)),
    ]
    response = llm.invoke(messages)
    return response.content
