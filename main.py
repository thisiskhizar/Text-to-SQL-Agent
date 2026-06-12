"""Entry point for the SQL agent.

Assembles the LLM, database, tools, and agent, then streams the agent's
reasoning and tool calls for a sample question against the Chinook database.
"""

from src.agent import build_agent
from src.database import get_db
from src.llm import get_llm
from src.tools import get_tools

QUESTION = "Which genre on average has the longest tracks?"


def main() -> None:
    """Run the SQL agent on a sample question and stream its output.

    Each step in the agent's reasoning chain is printed as it arrives,
    including tool calls and the final answer.
    """
    llm = get_llm()
    db = get_db()
    tools = get_tools(db, llm)
    agent = build_agent(llm, tools, db)

    print(f"\nQuestion: {QUESTION}\n{'=' * 60}")

    for step in agent.stream(
        {"messages": [{"role": "user", "content": QUESTION}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
