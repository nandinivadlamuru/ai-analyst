"""Detect underspecified questions before retrieval (general, no corpus-specific words)."""

from __future__ import annotations

import re

# Interrogatives and common task verbs for QA — not tied to any dataset.
_INTENT_TOKENS = frozenset(
    {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
        "whose",
        "whom",
        "list",
        "show",
        "tell",
        "give",
        "find",
        "summarize",
        "summarise",
        "explain",
        "describe",
        "compare",
        "calculate",
        "compute",
        "identify",
        "name",
        "define",
        "state",
        "report",
        "extract",
        "is",
        "are",
        "was",
        "were",
        "does",
        "did",
        "do",
        "can",
        "could",
        "should",
        "would",
        "has",
        "have",
        "had",
    }
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\-]+", text.lower())


def is_vague_question(question: str) -> bool:
    """
    Return True if the question is too short or lacks clear interrogative / task intent.

    Matches the spec: len(tokens) < 4 OR no intent keywords.
    """
    stripped = (question or "").strip()
    if not stripped:
        return True
    tokens = _tokens(stripped)
    if len(tokens) < 4:
        return True
    if not _INTENT_TOKENS.intersection(tokens):
        return True
    return False


CLARIFICATION_MESSAGE = (
    "Your question is unclear. Please specify what you want to know "
    "(e.g., metrics, dates, or categories mentioned in your documents)."
)
