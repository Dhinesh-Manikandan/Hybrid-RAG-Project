"""
Generation module for Hybrid-RAG project.
Provides context building and grounded answer generation using Qwen3-8B.
"""

from src.generation.context_builder import (
    build_context,
    build_context_string,
    extract_sources,
    format_source_entry,
)
from src.generation.llm_generator import (
    NO_CONTEXT_FALLBACK_ANSWER,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    QwenGenerator,
    generate_answer,
    get_generator,
)

__all__ = [
    "build_context",
    "build_context_string",
    "extract_sources",
    "format_source_entry",
    "QwenGenerator",
    "generate_answer",
    "get_generator",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "NO_CONTEXT_FALLBACK_ANSWER",
]
