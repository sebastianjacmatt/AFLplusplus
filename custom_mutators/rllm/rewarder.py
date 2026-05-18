"""Reward computation.

Three classes per docs/design.md §3.3:

* ``TFIDFCoverageRewarder`` — CovRL coverage reward over the AFL++ bitmap
  (Eqs. 3-6 of the CovRL-Fuzz paper). Binary TF, EMA-smoothed lagged IDF.
* ``StderrValidityRewarder`` — 3-way classification (syntax/semantic/valid)
  from the target's stderr, captured by ``exit_hook.so``.
* ``Rewarder`` — composer. Owns the 3-tier reward rule (CovRL Eq. 2):
  syntax → −1.0, semantic → −0.5, valid → `b + (1-b)·R_cov`.

SHM attach handles both SysV builds (numeric ``__AFL_SHM_ID``) and POSIX /
``USEMMAP`` builds (string path → ``shm_open`` + ``mmap``). The env var is
read via ``libc.getenv`` because AFL++ exports ``__AFL_SHM_ID`` *after*
``Py_Initialize``, so Python's ``os.environ`` snapshot misses it.

The CovRL cycle hooks from the reference design (``observe_saved_seed``,
``update_cycle``) are folded together here: every ``score()`` observes
the bitmap into DF as an approximate corpus member, and ``update_cycle``
is exposed for the trainer to invoke at finetune-cycle boundaries.
"""

import ctypes
import math
import mmap as py_mmap
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from config import Config


# ---------------------------------------------------------------------------
# Live env var lookup (bypasses Python's os.environ snapshot)
# ---------------------------------------------------------------------------

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.getenv.restype = ctypes.c_char_p
_LIBC.getenv.argtypes = [ctypes.c_char_p]


def _live_getenv(name: str) -> Optional[str]:
    raw = _LIBC.getenv(name.encode())
    if raw:
        return raw.decode()
    return os.environ.get(name) or None


def _looks_like_int(value: str) -> bool:
    value = value.strip()
    return value.isdigit() or (value.startswith("-") and value[1:].isdigit())


# ---------------------------------------------------------------------------
# SHM attach — SysV and POSIX (USEMMAP) variants
# ---------------------------------------------------------------------------

def _attach_sysv(shm_id: int, size: int) -> np.ndarray:
    shmat = _LIBC.shmat
    shmat.restype = ctypes.c_void_p
    shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    SHM_RDONLY = 0o10000
    ptr = shmat(shm_id, None, SHM_RDONLY)
    if ptr is None or ptr == ctypes.c_void_p(-1).value:
        errno = ctypes.get_errno()
        raise OSError(errno, f"shmat({shm_id}) failed: errno={errno}")
    buf = (ctypes.c_uint8 * size).from_address(ptr)
    return np.frombuffer(buf, dtype=np.uint8)


def _attach_posix(shm_name: str, size: int) -> np.ndarray:
    shm_open = _LIBC.shm_open
    shm_open.restype = ctypes.c_int
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]

    mmap_fn = _LIBC.mmap
    mmap_fn.restype = ctypes.c_void_p
    mmap_fn.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_long,
    ]

    close_fn = _LIBC.close
    close_fn.restype = ctypes.c_int
    close_fn.argtypes = [ctypes.c_int]

    fd = shm_open(shm_name.encode(), os.O_RDONLY, 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"shm_open({shm_name}) failed: errno={errno}")

    try:
        ptr = mmap_fn(None, size, py_mmap.PROT_READ, py_mmap.MAP_SHARED, fd, 0)
    finally:
        close_fn(fd)

    if ptr is None or ptr == ctypes.c_void_p(-1).value:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mmap({shm_name}) failed: errno={errno}")

    buf = (ctypes.c_uint8 * size).from_address(ptr)
    return np.frombuffer(buf, dtype=np.uint8)


def _attach_trace_bits(size: int) -> np.ndarray:
    shm_ref = _live_getenv("__AFL_SHM_ID")
    if not shm_ref:
        raise RuntimeError(
            "__AFL_SHM_ID not present; is afl-fuzz running with instrumentation?"
        )
    if _looks_like_int(shm_ref):
        return _attach_sysv(int(shm_ref), size)
    return _attach_posix(shm_ref, size)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Coverage rewarder — CovRL-Fuzz Eqs. 3-6
# ---------------------------------------------------------------------------

class TFIDFCoverageRewarder:
    """TF·IDF coverage reward, CovRL-Fuzz Eq. 5: ``R_cov = σ(log Σ tf_i · idf_i)``.

    Each AFL++ bitmap index is a *term*, each execution an approximate
    *document* (the reference design attaches ``observe_saved_seed`` to
    AFL's queue_new_entry hook; rllm doesn't expose that callback, so we
    fold every execution into DF instead). IDF is the lagged snapshot —
    ``score()`` reads it as it was at the last ``update_cycle()`` call.
    """

    FLOOR = 0.5  # cold-start / no-touch return value, per CovRL

    def __init__(self, bitmap_size: int, alpha: float) -> None:
        self.bitmap_size = bitmap_size
        self.alpha = alpha
        self._map_scale = math.sqrt(bitmap_size)
        self._df  = np.zeros(bitmap_size, dtype=np.uint32)
        self._idf = np.zeros(bitmap_size, dtype=np.float32)
        self._n   = 0
        self._trace_bits: Optional[np.ndarray] = None

    def score(self) -> float:
        """Score the just-executed bitmap and fold it into DF."""
        if self._trace_bits is None:
            self._trace_bits = _attach_trace_bits(self.bitmap_size)
        bitmap = self._trace_bits.copy()
        tf = (bitmap > 0).astype(np.float32)               # CovRL Eq. 3 (binary)

        weighted = float(np.dot(tf, self._idf))            # CovRL Eq. 5 inner

        # Approximate observe_saved_seed: every execution contributes to DF.
        self._df += tf.astype(np.uint32)
        self._n += 1

        if weighted <= 0.0:
            return self.FLOOR
        return round(_sigmoid(math.log(weighted)), 4)

    def update_cycle(self) -> None:
        """Refresh IDF from accumulated DF and blend via momentum (Eqs. 4 + 6)."""
        if self._n == 0:
            return
        # IDF[i] = (1/sqrt(M)) · log(N / (1 + DF[i]))         — Eq. 4
        new_idf = (
            np.log(self._n / (1.0 + self._df.astype(np.float32))) / self._map_scale
        ).astype(np.float32)
        # IDF_t = α · IDF_{t-1} + (1 - α) · IDF_new           — Eq. 6
        self._idf = (
            self.alpha * self._idf + (1.0 - self.alpha) * new_idf
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Validity rewarder — CovRL Eq. 2 stderr classifier
# ---------------------------------------------------------------------------

class StderrValidityRewarder:
    """3-way classification of the target's stderr text.

    Returns ``"syntax"`` / ``"semantic"`` / ``"valid"`` on a readable
    stderr, or ``None`` when the file is missing (LD_PRELOAD didn't fire
    or the env var wasn't set). Markers are the canonical JS error class
    names — Jerry / V8 / SpiderMonkey all emit these on the error branch.
    """

    SYNTAX_MARKERS   = ("SyntaxError",)
    SEMANTIC_MARKERS = (
        "ReferenceError", "TypeError", "RangeError", "URIError", "EvalError",
    )

    def __init__(self, stderr_path: str, max_bytes: int = 4096) -> None:
        self.stderr_path = stderr_path
        self.max_bytes = max_bytes

    def classify(self) -> Optional[str]:
        try:
            with open(self.stderr_path, "rb") as fh:
                data = fh.read(self.max_bytes)
        except OSError:
            return None
        finally:
            self.clear()
        text = data.decode("utf-8", errors="replace")
        if any(m in text for m in self.SYNTAX_MARKERS):
            return "syntax"
        if any(m in text for m in self.SEMANTIC_MARKERS):
            return "semantic"
        if text.strip() == "":
            return "valid"
        # Non-empty stderr without a known marker — conservative bucket.
        return "semantic"

    def clear(self) -> None:
        """Truncate (don't unlink) — the exit_hook's fd is bound to the inode.

        Unlinking the path orphans the next child's writes in a ghost
        inode that Python can no longer read.
        """
        try:
            os.truncate(self.stderr_path, 0)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Composite — CovRL Eq. 2
# ---------------------------------------------------------------------------

class Rewarder:
    """Compose coverage + validity into one scalar reward (CovRL Eq. 2).

    Always snapshots and folds the bitmap into DF, even on invalid runs —
    coverage on the parser/error path is still information. Only the
    final reward branches on validity.
    """

    SYNTAX_REWARD   = -1.0
    SEMANTIC_REWARD = -0.5

    def __init__(
        self,
        cfg: "Config",
        tf_idf: TFIDFCoverageRewarder,
        validity: StderrValidityRewarder,
    ) -> None:
        self.cfg = cfg
        self.tf_idf = tf_idf
        self.validity = validity

    def score(self) -> float:
        verdict = self.validity.classify()
        # Always observe coverage — bitmap from invalid runs still informs DF.
        r_cov = self.tf_idf.score()
        if verdict == "syntax":
            return self.SYNTAX_REWARD
        if verdict == "semantic":
            return self.SEMANTIC_REWARD
        b = self.cfg.validity_bonus
        return round(b + (1.0 - b) * r_cov, 4)

    def update_cycle(self) -> None:
        """Refresh IDF — call at finetune-cycle boundaries."""
        self.tf_idf.update_cycle()