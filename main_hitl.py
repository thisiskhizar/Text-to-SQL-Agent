"""Entry point for the SQL agent with human-in-the-loop query review.

Streams the agent's reasoning and pauses before each sql_db_query execution
so the user can inspect and approve or reject the generated SQL.
"""

from langgraph.types import Command

from src.agent import build_hitl_agent
from src.database import get_db
from src.llm import get_llm
from src.tools import get_tools

QUESTION = "Which genre on average has the longest tracks?"


def _print_interrupt(step: dict) -> None:
    """Print the pending tool call details from an interrupt step.

    Args:
        step: A streaming step dict that contains an '__interrupt__' key.
    """
    interrupt = step["__interrupt__"][0]
    print("\nINTERRUPTED — pending query requires approval:")
    for request in interrupt.value["action_requests"]:
        print(f"  {request['description']}")


def _ask_approval() -> bool:
    """Prompt the user to approve or reject the pending SQL query.

    Returns:
        True if the user approves, False otherwise.
    """
    answer = input("\nApprove query? [y/n]: ").strip().lower()
    return answer == "y"


def main() -> None:
    """Run the HITL SQL agent, pausing for approval before each query execution.

    Streams reasoning steps, interrupts on sql_db_query, prompts the user,
    and either resumes with approval or stops on rejection.
    """
    llm = get_llm()
    db = get_db()
    tools = get_tools(db, llm)
    agent = build_hitl_agent(llm, tools, db)

    config = {"configurable": {"thread_id": "1"}}

    print(f"\nQuestion: {QUESTION}\n{'=' * 60}")

    for step in agent.stream(
        {"messages": [{"role": "user", "content": QUESTION}]},
        config,
        stream_mode="values",
    ):
        if "__interrupt__" in step:
            _print_interrupt(step)
        elif "messages" in step:
            step["messages"][-1].pretty_print()

    if not _ask_approval():
        print("Query rejected. Stopping.")
        return

    print(f"\nResuming...\n{'=' * 60}")

    for step in agent.stream(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
        stream_mode="values",
    ):
        if "__interrupt__" in step:
            _print_interrupt(step)
        elif "messages" in step:
            step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
