"""CovRL-Fuzz TF·IDF coverage reward over the AFL++ bitmap.

Ported from the rllm2 branch (`custom_mutators/rllm/rewarder.py`). Attaches
once to AFL++'s ``trace_bits`` shared-memory segment and computes the
coverage reward ``R_cov = σ(log Σ tf_i · idf_i)`` (CovRL-Fuzz Eqs. 3-6) inside
``post_run`` — zero re-executions, zero subprocesses.

The validity 3-way (syntax/semantic/valid) is handled by the mutator's existing
``classify_stderr``; this module owns only the coverage half. The composite
(CovRL Eq. 2) lives in ``Mutator._reward``:
    syntax → −1.0,  semantic → −0.5,  valid → b + (1−b)·R_cov.

SHM attach handles both SysV builds (numeric ``__AFL_SHM_ID``) and POSIX /
``USEMMAP`` builds (string path → ``shm_open`` + ``mmap``). The env var is read
via ``libc.getenv`` because AFL++ exports ``__AFL_SHM_ID`` *after* the Python
interpreter starts, so ``os.environ`` misses it.
"""

import ctypes
import math
import mmap as py_mmap
import os
from typing import Optional

import numpy as np


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
    *document* folded into DF. IDF is the lagged snapshot — ``score()`` reads
    it as it was at the last ``update_cycle()`` call (IDF frozen within a
    finetune cycle, momentum-updated at the boundary, per CovRL).

    Repeatedly hitting the same edges (e.g. no-op mutations) drives those
    edges' DF up → their IDF decays → their reward decays: the IDF momentum is
    what makes a no-op self-defeating over time.

    With ``delta=True`` (B1, ``cfg.delta_coverage``) the reward credits only edges the
    child hit that its **parent seed** did not. Each queue entry's edge set is cached
    when AFL creates the entry (``cache_parent`` from the mutator's ``queue_new_entry``,
    reusing the SHM read ``score`` already did); the seed about to be fuzzed selects its
    cached parent via ``set_parent``, and ``score`` zeroes those edges before the σ(log·)
    transform. This strips not just the shared interpreter "floor" (IDF already ~zeroes
    it) but the parent's own rare edges that every infill inherits — the constant-in-group
    mass that pins absolute Σ tf·idf high-and-flat — moving the coverage variance *inside*
    the mask group where GRPO can use it (measured ~800× the within-group signal; see
    ``docs/coverage_signal.md`` §3). Pure SHM read, no subprocess.
    """

    FLOOR = 0.5  # cold-start / no-touch return value, per CovRL

    def __init__(self, bitmap_size: int, alpha: float, delta: bool = False) -> None:
        self.bitmap_size = bitmap_size
        self.alpha = alpha
        self.delta = delta                 # B1: credit only edges new vs the parent seed
        self._map_scale = math.sqrt(bitmap_size)
        self._df  = np.zeros(bitmap_size, dtype=np.uint32)
        self._idf = np.zeros(bitmap_size, dtype=np.float32)
        self._n   = 0
        self._trace_bits: Optional[np.ndarray] = None
        # Delta-vs-parent state. `_parent_idx` = the current parent seed's edge indices
        # (zeroed out in score); `_parent_cache` maps a queue entry's basename → its edge
        # indices, captured from the SHM the moment AFL creates the entry; `_last_idx` =
        # the most-recently-scored child's edges (the value cached when that child becomes
        # a new queue entry). uint16 index arrays (~3 KB/entry) rather than full vectors.
        self._last_idx:   Optional[np.ndarray] = None
        self._parent_idx: Optional[np.ndarray] = None
        self._parent_cache: dict[str, np.ndarray] = {}
        self._cache_cap = 100_000                       # FIFO bound on the cache

    def score(self) -> float:
        """Score the just-executed bitmap and fold it into DF.

        Attaches on first call (after AFL has exported ``__AFL_SHM_ID``); the
        caller is expected to guard against attach failure and degrade to a
        validity-only reward.
        """
        if self._trace_bits is None:
            self._trace_bits = _attach_trace_bits(self.bitmap_size)
        bitmap = self._trace_bits.copy()                   # live view → snapshot
        tf = (bitmap > 0).astype(np.float32)               # CovRL Eq. 3 (binary)
        self._last_idx = np.nonzero(tf)[0].astype(np.uint16)   # child's edges → cache as a future parent

        # Every execution contributes to DF (approx. observe_saved_seed). IDF is a
        # global edge-rarity weight, so DF always folds the FULL child trace, even in
        # delta mode — the delta only changes which edges we *credit*, not the stats.
        self._df += tf.astype(np.uint32)
        self._n += 1

        # B1 delta-vs-parent: credit only edges the child hit that its parent SEED did
        # not (zero the parent's edges). This strips not just the shared interpreter floor
        # (IDF already ~zeroes it) but the parent's own rare edges that every infill
        # inherits — the constant-in-group mass that pins absolute Σ tf·idf high-and-flat.
        # Uncaptured parent (delta off / initial -i seed) ⇒ absolute CovRL reward.
        scored = tf
        if self.delta and self._parent_idx is not None:
            scored = tf.copy()
            scored[self._parent_idx] = 0.0

        weighted = float(np.dot(scored, self._idf))        # CovRL Eq. 5 inner
        if weighted <= 0.0:
            # Absolute: cold-start / empty map → neutral FLOOR. Delta: a genuine "no new
            # edges vs parent" → 0.0 (low end of the valid range) so new coverage scores
            # strictly above none (the FLOOR override would rank no-new ABOVE small-new).
            return 0.0 if (self.delta and self._parent_idx is not None) else self.FLOOR
        return round(_sigmoid(math.log(weighted)), 4)

    def cache_parent(self, name: str) -> None:
        """Cache the most-recently-scored child's edge set under ``name`` for later use
        as a parent baseline. The mutator calls this from ``queue_new_entry`` — fired
        right after the mutation AFL just saved as queue entry ``name`` — so ``_last_idx``
        is exactly that entry's edge set, captured from the SHM read we already did in
        ``score()`` (no extra execution, no subprocess). Reused on every re-selection,
        which is why this is correct in the mutations-on-mutations regime where AFL does
        not re-run the parent (so the live SHM holds the *previous* seed's trace)."""
        if not self.delta or self._last_idx is None:
            return
        if name not in self._parent_cache and len(self._parent_cache) >= self._cache_cap:
            self._parent_cache.pop(next(iter(self._parent_cache)))   # FIFO-evict oldest
        self._parent_cache[name] = self._last_idx

    def set_parent(self, name: str) -> None:
        """Select the cached parent edge set for the seed about to be fuzzed (mutator
        ``fuzz_count``). Uncached (an initial ``-i`` seed, never created via
        ``queue_new_entry``) ⇒ no parent ⇒ ``score`` falls back to the absolute reward."""
        self._parent_idx = self._parent_cache.get(name) if self.delta else None

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
