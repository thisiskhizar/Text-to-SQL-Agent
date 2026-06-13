"""Interactive chat loop for the SQL agent.

Starts a REPL that accepts natural language questions, streams the agent's
reasoning and tool calls, and retains conversation history across turns.
Type 'quit', 'exit', or Ctrl-C to stop.
"""

from src.agent import build_chat_agent
from src.database import get_db
from src.llm import get_llm
from src.tools import get_tools

_CONFIG = {"configurable": {"thread_id": "chat"}}


def main() -> None:
    """Run the interactive SQL agent chat loop."""
    llm = get_llm()
    db = get_db()
    tools = get_tools(db, llm)
    agent = build_chat_agent(llm, tools, db)

    print("\nSQL Agent ready. Ask anything about the database, or type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        print()
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            _CONFIG,
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
        print()


if __name__ == "__main__":
    main()
