"""Prompt templates module.

Centralises all prompt strings used by the SQL agent so they can be
maintained independently of agent wiring logic.
"""

import os

from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT_TEMPLATE = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables."""

# Used by the generate_query node in the custom LangGraph agent (src/graph.py).
# Intentionally omits the "double check" and "list tables first" instructions
# because those are enforced structurally by dedicated graph nodes.
_GENERATE_QUERY_PROMPT_TEMPLATE = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""

# Used by the check_query node in the custom LangGraph agent (src/graph.py).
_CHECK_QUERY_PROMPT_TEMPLATE = """You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""


def build_system_prompt(dialect: str, top_k: int | None = None) -> str:
    """Render the SQL agent system prompt for a given database dialect.

    Args:
        dialect: The SQL dialect of the connected database (e.g. "sqlite").
        top_k: Maximum number of rows the agent should return per query.
               Defaults to the TOP_K environment variable, or 5 if unset.

    Returns:
        A fully rendered system prompt string ready to pass to the agent.
    """
    if top_k is None:
        top_k = int(os.getenv("TOP_K", "5"))

    return _SYSTEM_PROMPT_TEMPLATE.format(dialect=dialect, top_k=top_k)


def build_generate_query_prompt(dialect: str, top_k: int | None = None) -> str:
    """Render the generate_query node prompt for the custom LangGraph agent.

    Args:
        dialect: The SQL dialect of the connected database (e.g. "sqlite").
        top_k: Maximum rows per query. Defaults to TOP_K env var or 5.

    Returns:
        Rendered system prompt string for the generate_query node.
    """
    if top_k is None:
        top_k = int(os.getenv("TOP_K", "5"))

    return _GENERATE_QUERY_PROMPT_TEMPLATE.format(dialect=dialect, top_k=top_k)


def build_check_query_prompt(dialect: str) -> str:
    """Render the check_query node prompt for the custom LangGraph agent.

    Args:
        dialect: The SQL dialect of the connected database (e.g. "sqlite").

    Returns:
        Rendered system prompt string for the check_query node.
    """
    return _CHECK_QUERY_PROMPT_TEMPLATE.format(dialect=dialect)
