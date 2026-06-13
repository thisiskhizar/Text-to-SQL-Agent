"""Entry point for the custom LangGraph SQL agent.

Uses the manually constructed StateGraph from src/graph.py rather than the
high-level create_agent() abstraction. Each step of the SQL workflow runs in
a dedicated node: list tables, get schema, generate query, check query,
run query.
"""

from src.database import get_db
from src.graph import build_graph
from src.llm import get_llm
from src.tools import get_tools

QUESTION = "Which genre on average has the longest tracks?"


def main() -> None:
    """Run the custom LangGraph SQL agent on a sample question and stream output."""
    llm = get_llm()
    db = get_db()
    tools = get_tools(db, llm)
    graph = build_graph(llm, tools, db)

    print(f"\nQuestion: {QUESTION}\n{'=' * 60}")

    for step in graph.stream(
        {"messages": [{"role": "user", "content": QUESTION}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
