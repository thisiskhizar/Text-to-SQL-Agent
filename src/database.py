"""Database configuration module.

Handles downloading the Chinook SQLite database and creating a
LangChain SQLDatabase wrapper instance for agent use.
"""

import os
import pathlib

import requests
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase

load_dotenv()

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DB_PATH = pathlib.Path(os.getenv("DB_PATH", "Chinook.db"))


def download_db(url: str = CHINOOK_URL, local_path: pathlib.Path = DB_PATH) -> pathlib.Path:
    """Download the Chinook database file if it does not already exist locally.

    Args:
        url: Public URL of the Chinook.db file.
        local_path: Destination path to save the database file.

    Returns:
        The path to the local database file.

    Raises:
        RuntimeError: If the download request returns a non-200 status code.
    """
    if local_path.exists():
        print(f"{local_path} already exists, skipping download.")
        return local_path

    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"Downloaded database to {local_path}")
    else:
        raise RuntimeError(
            f"Failed to download Chinook.db. HTTP status: {response.status_code}"
        )

    return local_path


def get_db(local_path: pathlib.Path = DB_PATH) -> SQLDatabase:
    """Create and return a SQLDatabase wrapper connected to the Chinook database.

    Downloads the database file first if it is not already present.

    Args:
        local_path: Path to the local SQLite database file.

    Returns:
        A LangChain SQLDatabase instance ready for agent tool use.
    """
    download_db(local_path=local_path)
    db = SQLDatabase.from_uri(f"sqlite:///{local_path}")
    print(f"Dialect      : {db.dialect}")
    print(f"Tables       : {db.get_usable_table_names()}")
    print(f"Sample rows  : {db.run('SELECT * FROM Artist LIMIT 5;')}")
    return db
