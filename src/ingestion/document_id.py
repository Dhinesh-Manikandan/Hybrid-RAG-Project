"""
document_id.py -- Stable, deterministic document identity generation.

Uses SHA-256 hash of the normalized filename stem so that:
- Same file always produces the same document_id
- Different files always produce different document_ids
- No manual ID assignment is needed
"""

import hashlib
from pathlib import Path


def generate_document_id(source_path: Path) -> str:
    """
    Generate a stable, globally unique document_id from a file path.

    Strategy: deterministic SHA-256 hash of normalized filename stem.
    This is stable across sessions, OS-independent, and collision-resistant.

    Args:
        source_path: Path to the raw document file.

    Returns:
        document_id string like "doc_4a8f2c91e3b1"
    """
    stem = source_path.stem.strip().lower()
    hash_hex = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:12]
    return f"doc_{hash_hex}"
