"""Custom LangGraph SQL agent.

Implements the SQL agent using explicit LangGraph nodes and edges rather than
the high-level create_agent() abstraction. Each step of the SQL workflow runs
in a dedicated node, giving fine-grained control over tool-calling behaviour.

Base graph topology (build_graph):
    START → list_tables → call_get_schema → get_schema → generate_query
                                                               ↓ tool call
                                                          check_query → run_query
                                                               ↓ no tool call
                                                              END

HITL graph topology (build_hitl_graph):
    Replaces check_query + run_query with an interrupt-gated run_query node.
    Execution pauses before every SQL query for human review.
"""

from typing import Literal

from langchain.tools import tool as tool_decorator
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from src.prompts import build_check_query_prompt, build_generate_query_prompt


def build_graph(llm: ChatOpenAI, tools: list, db: SQLDatabase):
    """Compile and return the custom LangGraph SQL agent (no HITL).

    Enforces a fixed execution order: list tables → get schema → generate
    query → check query → run query, with a loop back to generate_query after
    each result until the LLM produces a final answer with no tool calls.

    Args:
        llm: The chat model used for all reasoning steps.
        tools: SQL toolkit tools returned by get_tools().
        db: Connected SQLDatabase instance; dialect is used for prompts.

    Returns:
        A compiled LangGraph StateGraph ready for streaming.
    """
    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
    get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
    run_query_tool = next(t for t in tools if t.name == "sql_db_query")

    get_schema_node = ToolNode([get_schema_tool], name="get_schema")
    run_query_node = ToolNode([run_query_tool], name="run_query")

    generate_query_prompt = build_generate_query_prompt(dialect=db.dialect)
    check_query_prompt = build_check_query_prompt(dialect=db.dialect)

    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "list_tables_call",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState):
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(state: MessagesState):
        system_message = {"role": "system", "content": generate_query_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def check_query(state: MessagesState):
        system_message = {"role": "system", "content": check_query_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        # Replace the ID so LangGraph updates the existing message in state
        # rather than appending a duplicate tool-call message.
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        return "check_query"

    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    return builder.compile()


def build_hitl_graph(llm: ChatOpenAI, tools: list, db: SQLDatabase):
    """Compile and return the custom LangGraph SQL agent with HITL review.

    Identical structure to build_graph() but replaces check_query + run_query
    with a single interrupt-gated run_query node. Execution pauses before
    every sql_db_query call so the user can approve, edit the query, or send
    feedback back to the LLM.

    Must be invoked with a thread-aware config so the InMemorySaver
    checkpointer can pause and resume execution across the interrupt.

    Args:
        llm: The chat model used for all reasoning steps.
        tools: SQL toolkit tools returned by get_tools().
        db: Connected SQLDatabase instance; dialect is used for prompts.

    Returns:
        A compiled LangGraph StateGraph with InMemorySaver checkpointer.
    """
    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
    get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")
    run_query_tool = next(t for t in tools if t.name == "sql_db_query")

    get_schema_node = ToolNode([get_schema_tool], name="get_schema")

    generate_query_prompt = build_generate_query_prompt(dialect=db.dialect)

    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "list_tables_call",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState):
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(state: MessagesState):
        system_message = {"role": "system", "content": generate_query_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    @tool_decorator(
        run_query_tool.name,
        description=run_query_tool.description,
        args_schema=run_query_tool.args_schema,
    )
    def run_query_with_interrupt(config: RunnableConfig, **tool_input):
        request = {
            "action": run_query_tool.name,
            "args": tool_input,
            "description": "Please review the tool call",
        }
        response = interrupt([request])
        if response["type"] == "accept":
            return run_query_tool.invoke(tool_input, config)
        elif response["type"] == "edit":
            return run_query_tool.invoke(response["args"]["args"], config)
        elif response["type"] == "response":
            return response["args"]
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

    run_query_node = ToolNode([run_query_with_interrupt], name="run_query")

    def should_continue(state: MessagesState) -> Literal["run_query", "__end__"]:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        return "run_query"

    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("run_query", "generate_query")

    return builder.compile(checkpointer=InMemorySaver())
