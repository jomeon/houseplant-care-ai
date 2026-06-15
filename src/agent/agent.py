from __future__ import annotations

import logging

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.agent.tools import SessionContext, build_tools
from src.llm.chatbot import get_llm
from src.llm.prompts import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class PlantCareAgent:
    def __init__(self, context: SessionContext) -> None:
        self.context = context
        self.tools = build_tools(context)
        self.llm = get_llm()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AGENT_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self._executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            return_intermediate_steps=True,
        )

        self._history = InMemoryChatMessageHistory()
        self.runnable = RunnableWithMessageHistory(
            self._executor,
            lambda _session_id: self._history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def chat(self, user_input: str) -> dict:
        result = self.runnable.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}},
        )
        steps = [
            (action.tool, str(observation))
            for action, observation in result.get("intermediate_steps", [])
        ]
        return {"output": result["output"], "steps": steps}
