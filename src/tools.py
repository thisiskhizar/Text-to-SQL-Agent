"""SQL agent tools module.

Builds the set of database interaction tools from a SQLDatabaseToolkit,
which provides the agent with the ability to list tables, inspect schemas,
check queries, and execute queries against the database.
"""

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI


def get_tools(db: SQLDatabase, llm: ChatOpenAI) -> list:
    """Create and return the SQL database tools for the agent.

    Uses SQLDatabaseToolkit to produce four tools:
      - sql_db_list_tables: lists available tables
      - sql_db_schema: returns schema and sample rows for given tables
      - sql_db_query_checker: validates a query before execution
      - sql_db_query: executes a SQL query and returns results

    Args:
        db: A configured SQLDatabase instance (see database.py).
        llm: The chat model used internally by the toolkit.

    Returns:
        A list of LangChain BaseTool instances ready for agent use.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return tools
