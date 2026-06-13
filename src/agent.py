"""Agent construction module.

Assembles the SQL agent from its constituent parts: an LLM, a set of
database tools, and a system prompt. Returns a compiled LangGraph agent
ready for streaming or invocation.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

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


def build_chat_agent(llm: ChatOpenAI, tools: list, db: SQLDatabase):
    """Create and return a ReAct SQL agent with memory for multi-turn chat.

    Same as build_agent() but adds an InMemorySaver checkpointer so the
    agent retains conversation history across turns within a session.

    Args:
        llm: The chat model that will reason and generate SQL queries.
        tools: The list of SQL tools returned by get_tools().
        db: The connected SQLDatabase instance, used to derive the dialect.

    Returns:
        A compiled LangGraph agent that maintains message history per thread.
    """
    system_prompt = build_system_prompt(dialect=db.dialect)

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )

    return agent


def build_hitl_agent(llm: ChatOpenAI, tools: list, db: SQLDatabase):
    """Create and return a ReAct SQL agent with human-in-the-loop review.

    Identical to build_agent() but adds HumanInTheLoopMiddleware so the
    agent pauses for approval before executing sql_db_query, and an
    InMemorySaver checkpointer so execution can be resumed after the pause.

    Args:
        llm: The chat model that will reason and generate SQL queries.
        tools: The list of SQL tools returned by get_tools().
        db: The connected SQLDatabase instance, used to derive the dialect.

    Returns:
        A compiled LangGraph agent that interrupts before sql_db_query and
        can be resumed via Command(resume=...).
    """
    system_prompt = build_system_prompt(dialect=db.dialect)

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"sql_db_query": True},
                description_prefix="Tool execution pending approval",
            ),
        ],
        checkpointer=InMemorySaver(),
    )

    return agent
