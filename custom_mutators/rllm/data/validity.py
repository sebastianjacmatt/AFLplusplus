"""Classify a JavaScript interpreter's stderr into validity classes.

Shared between the runtime mutator (``mutator.post_run``) and offline tools
(e.g. seed selection). Kept dependency-free (stdlib only).

The classes mirror CovRL-Fuzz's reward partition: ``valid`` (empty stderr),
``syntax`` (parser refused the input), ``semantic`` (program parsed but raised a
runtime error), plus ``timeout`` (the run hung and AFL killed it). A hang yields
no stderr, so it can't be told from a clean ``valid`` run by text — instead
``exit_hook.so`` catches AFL's timeout-kill (a catchable ``AFL_KILL_SIGNAL``) and
appends ``_TIMEOUT_MARK`` to the stderr file; its presence here = ``timeout``.
"""

from __future__ import annotations

from typing import Literal

ValidityClass = Literal["valid", "syntax", "semantic", "timeout"]

# NUL-wrapped sentinel written by exit_hook.so's SIGUSR1 handler on a timeout.
# Must byte-match exit_hook.c:TMOUT_MARK. NULs can't collide with jerry's text.
_TIMEOUT_MARK = "\x00RLM_TMOUT\x00"

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

    ``_TIMEOUT_MARK`` present → ``timeout`` (AFL killed a hang; checked first
    since a hang's stderr is otherwise empty). Empty / whitespace-only stderr →
    ``valid``. Any ``SyntaxError`` → ``syntax``. Any of the standard runtime
    error names → ``semantic``. Anything else non-empty → ``semantic``.
    """
    if _TIMEOUT_MARK in text:
        return "timeout"
    if any(m in text for m in _SYNTAX_MARKERS):
        return "syntax"
    if any(m in text for m in _SEMANTIC_MARKERS):
        return "semantic"
    if not text.strip():
        return "valid"
    return "semantic"
