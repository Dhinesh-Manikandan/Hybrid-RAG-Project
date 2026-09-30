"""
io.py -- Safe I/O helpers for JSON and pickle files.
"""

import json
import pickle
from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Save data to a JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: Path) -> None:
    """Save data to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """Load data from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
