"""Shared pytest fixtures for tests against the bundled database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "src" / "indicio" / "data" / "refractiveindex.db"


@pytest.fixture(scope="session")
def db_path() -> Path:
    if not DB_PATH.is_file():
        raise pytest.UsageError(
            f"bundled database not found at {DB_PATH} — "
            "run `uv run python scripts/build_database.py` first"
        )
    return DB_PATH


@pytest.fixture(scope="session")
def conn(db_path: Path):
    uri = f"file:{db_path}?mode=ro"
    c = sqlite3.connect(uri, uri=True)
    yield c
    c.close()
