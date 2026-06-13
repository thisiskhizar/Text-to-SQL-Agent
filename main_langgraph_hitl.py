"""Entry point for the custom LangGraph SQL agent with human-in-the-loop review.

Uses the interrupt()-gated graph from src/graph.py. Execution pauses before
each sql_db_query call so the user can approve the query, edit it, or send
feedback back to the model.

Resume options at each interrupt:
  [a] accept  — run the query as-is
  [e] edit    — provide a revised query string
  [r] respond — send a feedback message to the LLM instead of running a query
  [q] quit    — abort execution
"""

import json

from langgraph.types import Command

from src.database import get_db
from src.graph import build_hitl_graph
from src.llm import get_llm
from src.tools import get_tools

QUESTION = "Which genre on average has the longest tracks?"


def _print_interrupt(step: dict) -> None:
    action = step["__interrupt__"][0]
    print("\nINTERRUPTED — pending query requires review:")
    for request in action.value:
        print(json.dumps(request, indent=2))


def _ask_decision() -> dict | None:
    """Prompt the user for a decision and return the resume payload, or None to abort."""
    print("\nOptions: [a]ccept  [e]dit  [r]espond  [q]uit")
    choice = input("Choice: ").strip().lower()
    if choice == "a":
        return {"type": "accept"}
    elif choice == "e":
        new_query = input("Enter revised query: ").strip()
        return {"type": "edit", "args": {"args": {"query": new_query}}}
    elif choice == "r":
        feedback = input("Feedback for the model: ").strip()
        return {"type": "response", "args": feedback}
    else:
        return None


def _stream(graph, input_, config) -> bool:
    """Stream graph steps and return True if execution was interrupted."""
    interrupted = False
    for step in graph.stream(input_, config, stream_mode="values"):
        if "messages" in step:
            step["messages"][-1].pretty_print()
        elif "__interrupt__" in step:
            _print_interrupt(step)
            interrupted = True
    return interrupted


def main() -> None:
    """Run the HITL LangGraph SQL agent, pausing for review before each query."""
    llm = get_llm()
    db = get_db()
    tools = get_tools(db, llm)
    graph = build_hitl_graph(llm, tools, db)

    config = {"configurable": {"thread_id": "1"}}
    print(f"\nQuestion: {QUESTION}\n{'=' * 60}")

    current_input = {"messages": [{"role": "user", "content": QUESTION}]}

    while True:
        interrupted = _stream(graph, current_input, config)

        if not interrupted:
            break

        decision = _ask_decision()
        if decision is None:
            print("Aborted.")
            break

        print(f"\nResuming...\n{'=' * 60}")
        current_input = Command(resume=decision)


if __name__ == "__main__":
    main()
