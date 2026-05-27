"""Classify a JavaScript interpreter's stderr into validity classes.

Shared between the runtime mutator (``mutator.post_run``) and offline tools
(e.g. seed selection). Kept dependency-free (stdlib only).

The three classes mirror CovRL-Fuzz's reward partition: ``valid`` (empty
stderr), ``syntax`` (parser refused the input), ``semantic`` (program parsed
but raised a runtime error).
"""

from __future__ import annotations

from typing import Literal

ValidityClass = Literal["valid", "syntax", "semantic"]

_SYNTAX_MARKERS = ("SyntaxError",)
_SEMANTIC_MARKERS = (
    "ReferenceError",
    "TypeError",
    "RangeError",
    "URIError",
    "EvalError",
)


def classify_stderr(text: str) -> ValidityClass:
    """Map a JerryScript stderr string to one of the three classes.

    Empty / whitespace-only stderr → ``valid``. Any ``SyntaxError`` →
    ``syntax``. Any of the standard runtime error names → ``semantic``.
    Anything else non-empty falls back to ``semantic`` (conservative).
    """
    if any(m in text for m in _SYNTAX_MARKERS):
        return "syntax"
    if any(m in text for m in _SEMANTIC_MARKERS):
        return "semantic"
    if not text.strip():
        return "valid"
    return "semantic"
