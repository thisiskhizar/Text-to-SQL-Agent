"""LLM initialization module.

Loads configuration from environment variables and returns a chat model
instance ready for tool-calling.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm() -> ChatOpenAI:
    """Create and return a configured OpenAI chat model.

    Reads OPENAI_API_KEY from the environment (via .env).
    The model is configured for tool-calling use with the SQL agent.

    Returns:
        A ChatOpenAI instance ready for use with LangChain agents.

    Raises:
        ValueError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-...":
        raise ValueError(
            "OPENAI_API_KEY is not set. Please add your key to the .env file."
        )

    return ChatOpenAI(model="gpt-4o", api_key=api_key)
