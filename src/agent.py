"""Agent construction module.

Assembles the SQL agent from its constituent parts: an LLM, a set of
database tools, and a system prompt. Returns a compiled LangGraph agent
ready for streaming or invocation.
"""

from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from src.prompts import build_system_prompt


def build_agent(llm: ChatOpenAI, tools: list, db: SQLDatabase):
    """Create and return a ReAct SQL agent.

    Combines the language model, SQL tools, and a dialect-aware system
    prompt into a runnable LangGraph agent via create_agent.

    Args:
        llm: The chat model that will reason and generate SQL queries.
        tools: The list of SQL tools returned by get_tools().
        db: The connected SQLDatabase instance, used to derive the dialect.

    Returns:
        A compiled LangGraph agent that accepts message inputs and streams
        reasoning steps and tool calls.
    """
    system_prompt = build_system_prompt(dialect=db.dialect)

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )

    return agent
