"""Rewarding for the Reinforcement Learning Language Model (RLLM) Mutator.

Reward computation is decomposed into composable :class:`AbstractRewarder`
components, each owning the state and IO it needs to score one mutation.
The :class:`Rewarder` aggregator only combines their outputs into a
:class:`RewardResult`. It does not own SHM, files, or per-execution
buffers — those live on the components that consume them.

Components:

    - :class:`TFIDFCoverageRewarder`
        CovRL-Fuzz Eq. 4-6 — TF-IDF weighted coverage reward against a
        per-edge IDF snapshot. Owns the AFL trace_bits SHM attachment and
        the cached most-recent bitmap; document frequency is accumulated
        by ``observe_saved_seed`` and the snapshot is refreshed at
        cycle boundaries (``update_cycle``).

    - :class:`ExitCodeRewarder`
        Coarse binary signal from the process exit status written by
        ``exit_hook.so``. Owns the exit-code file path, ``read()``, and
        ``clear()``.

    - :class:`ValidityRewarder`
        Granular ``-1.0`` (syntax) / ``-0.5`` (semantic) discrimination per
        CovRL-Fuzz Eq. 2. Stub for now — discriminating syntax vs semantic
        errors needs the engine's textual stderr (e.g. via afl-showmap),
        which is not yet captured by rlm_mutator.

    - :class:`CovRLRewarder`
        CovRL-Fuzz Eq. 2 composite — applies the validity penalty if the
        program is invalid, otherwise returns the TF-IDF coverage reward.
        Intended for the offline pass that patches rollout rewards once
        afl-showmap output is available.

:class:`Rewarder.compute` accepts an optional
:class:`ExecutionObservation`. When omitted (online fuzzing path) the
sub-rewarders read live SHM/exit-file state; when supplied (rollout
dataset path with afl-showmap output) it scores the pre-captured
observation instead.
"""
from __future__ import annotations

import ctypes
import math
import os
import mmap as py_mmap
from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result + per-execution observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardResult:
    """Structured reward output.

    ``reward`` is the scalar consumed by RL during fuzzing. The remaining
    fields are component-level diagnostics so an offline pass can patch the
    scalar with the full CovRL signal once afl-showmap discriminates
    syntax/semantic errors.
    """

    reward: float
    coverage_reward: float | None = None      # raw TF-IDF, ungated by validity
    exit_code: int | None = None
    valid: bool = False
    reward_reason: str | None = None
    novelty_score: float | None = None
    crash: bool | None = None
    timeout: bool | None = None
    validity: str | None = None               # 'syntax' | 'semantic' | 'valid' | None

    def as_record_fields(self) -> dict:
        return asdict(self)

    @classmethod
    def diagnostic_field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls) if field.name != "reward")


@dataclass
class ExecutionObservation:
    """Per-execution data passed to :meth:`AbstractRewarder.result`.

    Each rewarder reads only the fields it needs; unused fields stay ``None``.
    """

    bitmap: np.ndarray | None = None
    exit_code: int | None = None
    validity: str | None = None    # 'syntax' | 'semantic' | 'valid' | None


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------

class AbstractRewarder:
    """Compute one rewarder's contribution for a single execution.

    The cycle hooks (``observe_saved_seed``, ``update_cycle``) default to
    no-ops; stateful rewarders override them. Composites forward the calls
    to their children.
    """

    def result(self, obs: ExecutionObservation) -> float:
        raise NotImplementedError

    def observe_saved_seed(self, bitmap: np.ndarray) -> None:
        """Account for one corpus seed in any internal corpus statistics."""
        return None

    def update_cycle(self) -> None:
        """Snapshot any per-cycle state at a finetune-cycle boundary."""
        return None


class ExitCodeRewarder(AbstractRewarder):
    """Coarse execution-outcome reward and the exit-code IO it owns.

    Returns ``valid_reward`` on a clean exit (status 0) and ``invalid_reward``
    on any non-zero or missing status. ``exit_hook.so`` writes the status to
    ``exit_code_path`` per execution; ``read()`` consumes it and ``clear()``
    removes any stale value.

    @param exit_code_path: File written by ``exit_hook.so`` per execution.
    @param valid_reward:   Reward when the target exited 0.
    @param invalid_reward: Reward when the exit was non-zero or missing.
    """

    def __init__(
        self,
        exit_code_path: str,
        valid_reward: float = 0.0,
        invalid_reward: float = -1.0,
    ) -> None:
        self.exit_code_path = exit_code_path
        self.valid_reward   = valid_reward
        self.invalid_reward = invalid_reward

    def result(self, obs: ExecutionObservation) -> float:
        return self.valid_reward if obs.exit_code == 0 else self.invalid_reward

    def read(self) -> int | None:
        """Read and clear the exit code written by ``exit_hook.so``.

        Returns None if the file is missing or unparseable; the file is
        always cleared so the next execution starts from a clean slate.
        """
        try:
            with open(self.exit_code_path) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None
        finally:
            self.clear()

    def clear(self) -> None:
        """Remove any stale exit-code output before/after a child execution."""
        try:
            os.remove(self.exit_code_path)
        except FileNotFoundError:
            pass


class StderrValidityRewarder(AbstractRewarder):
    """3-way validity classification from the target's stderr text.

    CovRL Eq. 2 splits the validity branch by error kind:

        r(W*) = -1.0    if W* triggered a syntax error
              = -0.5    if W* triggered a semantic error (reference, type, range, URI)
              = +R_cov  if W* passed cleanly

    JS engines like Jerry print the error class to stderr ("Unhandled exception:
    SyntaxError", "Unhandled exception: ReferenceError", ...). ``exit_hook.so``
    redirects fd 2 to ``stderr_path`` per forked execution, so reading that file
    after the run gives us the engine's textual classification.

    @param stderr_path:     File written by exit_hook.so's stderr redirect.
    @param syntax_reward:   Reward when stderr contains a syntax-error marker.
    @param semantic_reward: Reward when stderr contains a runtime-error marker.
    @param valid_reward:    Reward when stderr is empty (program ran cleanly).
                            For a composite use, this is just a sentinel — the
                            composite usually swaps in R_cov instead.
    @param max_bytes:       Cap reads at this many bytes; we only need to
                            substring-match, no engine prints more for an error.
    """

    SYNTAX_MARKERS:   tuple[str, ...] = ("SyntaxError",)
    SEMANTIC_MARKERS: tuple[str, ...] = (
        "ReferenceError", "TypeError", "RangeError", "URIError", "EvalError",
    )

    def __init__(
        self,
        stderr_path:     str,
        syntax_reward:   float = -1.0,
        semantic_reward: float = -0.5,
        valid_reward:    float = 0.0,
        max_bytes:       int   = 4096,
    ) -> None:
        self.stderr_path     = stderr_path
        self.syntax_reward   = syntax_reward
        self.semantic_reward = semantic_reward
        self.valid_reward    = valid_reward
        self.max_bytes       = max_bytes

    @classmethod
    def classify(cls, stderr_text: str | None) -> str | None:
        """Map a stderr buffer to a validity class.

        Returns 'syntax' / 'semantic' / 'valid' on a definitive read, or
        ``None`` when the stderr file was missing/unreadable so the caller
        can fall back to exit-code semantics.
        """
        if stderr_text is None:
            return None
        if any(m in stderr_text for m in cls.SYNTAX_MARKERS):
            return "syntax"
        if any(m in stderr_text for m in cls.SEMANTIC_MARKERS):
            return "semantic"
        if stderr_text.strip() == "":
            return "valid"
        # Non-empty stderr without a known marker — could be debug prints,
        # warnings, or unrecognised errors. Conservative: treat as semantic.
        return "semantic"

    def result(self, obs: ExecutionObservation) -> float:
        cls_ = obs.validity
        if cls_ == "syntax":
            return self.syntax_reward
        if cls_ == "semantic":
            return self.semantic_reward
        if cls_ == "valid":
            return self.valid_reward
        # Unknown — caller should already have fallen back; return semantic
        # as a safe non-zero penalty.
        return self.semantic_reward

    def read(self) -> str | None:
        """Read and clear the stderr file written by exit_hook.so's redirect.

        Returns the stderr text (capped at ``max_bytes``), or ``None`` when
        the file does not exist (LD_PRELOAD didn't fire, or the env var was
        unset). The file is always cleared so the next execution starts fresh.
        """
        try:
            with open(self.stderr_path, "rb") as fh:
                data = fh.read(self.max_bytes)
        except OSError:
            return None
        finally:
            self.clear()
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def clear(self) -> None:
        """Truncate (don't unlink) so the forkserver-side fd keeps writing
        to the same inode. Unlinking the path leaves the next child's
        writes orphaned in a ghost inode that Python can no longer read."""
        try:
            os.truncate(self.stderr_path, 0)
        except FileNotFoundError:
            pass


class TFIDFCoverageRewarder(AbstractRewarder):
    """TF-IDF weighted coverage reward (CovRL-Fuzz Eq. 4-6).

    Treats each AFL++ bitmap index as a *term* and each saved corpus seed
    as a *document*. Document frequency ``DF_cov[i]`` is accumulated by
    ``observe_saved_seed`` and folded into a fresh IDF map at every
    ``update_cycle`` call; ``result`` scores executions against the most
    recent snapshot, matching CovRL's lagged ``idf_{t-1}`` semantics.

    Three hooks:

    - ``result(obs)`` — score one execution; pure, no state mutation.
    - ``observe_saved_seed(bitmap)`` — increment ``DF_cov`` for the edges the
      seed touched and grow ``N`` by one. Call when AFL adds the input to
      the corpus (or, as an approximation, on every execution).
    - ``update_cycle()`` — recompute ``IDF_cov`` from the accumulated
      ``DF_cov`` and blend with the previous snapshot via momentum
      ``alpha`` (Eq. 6). Call at finetune-cycle boundaries.

    @param bitmap_size: ``M``, the AFL++ coverage bitmap size in bytes.
    @param alpha:       Momentum rate ``alpha in [0, 1]``. CovRL uses 0.6.
    @param floor:       Reward returned when no IDF mass is touched (cold
                        start, no-coverage execution, or net non-positive
                        weighted sum). Defaults to 0.5 per CovRL.
    """

    def __init__(
        self,
        bitmap_size: int,
        alpha: float = 0.6,
        floor: float = 0.5,
    ) -> None:
        self.bitmap_size  = bitmap_size
        self._alpha       = alpha
        self._floor       = floor
        self._map_scale   = math.sqrt(bitmap_size)                   # sqrt(M) in Eq. 4
        self._df_cov      = np.zeros(bitmap_size, dtype=np.uint32)   # DF_cov(i)
        self._idf_prev    = np.zeros(bitmap_size, dtype=np.float32)  # IDF_{t-1}
        self._n_seeds     = 0                                        # N = corpus size
        self._trace_bits_view: Optional[np.ndarray] = None
        self._last_bitmap:     Optional[np.ndarray] = None

    def snapshot_bitmap(self) -> np.ndarray:
        """Snapshot the AFL trace_bits SHM and cache the copy.

        Lazily attaches to the SHM segment the first time it is called
        (AFL sets ``__AFL_SHM_ID`` after Python startup). Subsequent calls
        return a fresh per-execution copy; the cached copy is reused by
        :meth:`observe_last_seed` so callers don't need to re-snapshot to
        fold the same execution into DF.
        """
        if self._trace_bits_view is None:
            self._trace_bits_view = attach_trace_bits(self.bitmap_size)
        bitmap = self._trace_bits_view.copy()
        self._last_bitmap = bitmap
        return bitmap

    def observe_last_seed(self) -> None:
        """Fold the most recently snapshotted bitmap into DF as a corpus seed.

        Convenience wrapper for the online path where the same bitmap drives
        both ``result(obs)`` and DF accumulation in the same execution.
        """
        if self._last_bitmap is not None:
            self.observe_saved_seed(self._last_bitmap)

    def result(self, obs: ExecutionObservation) -> float:
        """``R_cov = sigma(log sum_i tf_i * idf_{i,t-1})``  — Eq. 5.

        TF is binary: ``tf_i = 1`` iff edge ``i`` fired this execution. CovRL
        deliberately drops AFL's bucket information here (paper §3.2 Eq. 3).
        """
        if obs.bitmap is None:
            return self._floor
        tf_cov = (obs.bitmap > 0).astype(np.float32)              # Eq. 3
        weighted = float(np.dot(tf_cov, self._idf_prev))
        if weighted <= 0.0:
            return self._floor
        return round(_sigmoid(math.log(weighted)), 4)

    def observe_saved_seed(self, bitmap: np.ndarray) -> None:
        """Add one document's binary edge-presence vector to ``DF_cov``."""
        self._df_cov += (bitmap > 0).astype(np.uint32)
        self._n_seeds += 1

    def update_cycle(self) -> None:
        """Recompute IDF from accumulated DF and blend via momentum (Eq. 4 + 6)."""
        if self._n_seeds == 0:
            return
        # IDF_cov[i] = (1/sqrt(M)) * log(N / (1 + DF_cov[i]))   — Eq. 4
        new_idf = (
            np.log(self._n_seeds / (1.0 + self._df_cov.astype(np.float32)))
            / self._map_scale
        ).astype(np.float32)
        # IDF_t = alpha * IDF_{t-1} + (1 - alpha) * IDF_t^new   — Eq. 6
        self._idf_prev = (
            self._alpha * self._idf_prev + (1.0 - self._alpha) * new_idf
        ).astype(np.float32)

    @property
    def n_seeds(self) -> int:
        return self._n_seeds

    @property
    def idf_snapshot(self) -> np.ndarray:
        """Read-only view of the IDF snapshot used by ``result``."""
        return self._idf_prev


class ValidityRewarder(AbstractRewarder):
    """Granular validity reward (CovRL-Fuzz Eq. 2 prefix).

    Returns ``-1.0`` on a syntax error, ``-0.5`` on a semantic error, and
    ``0.0`` on a valid program. Discriminating syntax vs semantic errors
    requires inspecting the engine's stderr text (typically captured via
    afl-showmap), which rlm_mutator does not yet do. Stub for now —
    callers that wrap this with :class:`CovRLRewarder` will fall back to
    pure coverage scoring.
    """

    SYNTAX_PENALTY: float = -1.0
    SEMANTIC_PENALTY: float = -0.5
    VALID: float = 0.0

    def result(self, obs: ExecutionObservation) -> float:
        raise NotImplementedError(
            "ValidityRewarder requires afl-showmap-based syntax/semantic capture"
        )


class CovRLRewarder(AbstractRewarder):
    """CovRL-Fuzz composite reward (Eq. 2).

    ``r(W*) = -1.0`` (syntax error) | ``-0.5`` (semantic error) | ``+R_cov`` (passed)

    Validity gating is delegated to :class:`ValidityRewarder` and coverage
    scoring to :class:`TFIDFCoverageRewarder`. Intended for the offline
    pass that patches rollout rewards once afl-showmap classifies each
    saved input. While ValidityRewarder remains a stub, this composite
    falls back to coverage-only scoring so the trainer still receives a
    meaningful signal if it is wired in directly.
    """

    def __init__(
        self,
        validity: ValidityRewarder,
        tf_idf:   TFIDFCoverageRewarder,
    ) -> None:
        self.validity = validity
        self.tf_idf   = tf_idf

    def result(self, obs: ExecutionObservation) -> float:
        try:
            penalty = self.validity.result(obs)
        except NotImplementedError:
            return self.tf_idf.result(obs)
        if penalty < 0.0:
            return penalty
        return self.tf_idf.result(obs)

    def observe_saved_seed(self, bitmap: np.ndarray) -> None:
        self.tf_idf.observe_saved_seed(bitmap)

    def update_cycle(self) -> None:
        self.tf_idf.update_cycle()


# ---------------------------------------------------------------------------
# Aggregator — owns SHM + exit-code plumbing and yields RewardResult
# ---------------------------------------------------------------------------

class Rewarder:
    """Per-execution aggregator that combines sub-rewarder outputs.

    Holds no IO and no per-execution buffers — those live on the
    sub-rewarders that own them (``TFIDFCoverageRewarder`` for the SHM
    bitmap, ``ExitCodeRewarder`` for the exit-code file,
    ``StderrValidityRewarder`` for the stderr-text file). This class only
    encodes CovRL Eq. 2:

        reward = -1.0    if SyntaxError    (parse failure)
               = -0.5    if semantic error (ReferenceError / TypeError / ...)
               = +R_cov  if program ran cleanly to completion

    Validity is decided from the engine's stderr text (Jerry / V8 / Spider-
    Monkey all print "SyntaxError" / "ReferenceError" / etc. on the error
    branch). When the stderr file is unavailable (LD_PRELOAD didn't fire,
    env var unset) the aggregator falls back to the binary exit-code path
    so older configurations remain operational.

    :meth:`compute` is the single entry point. With ``obs is None`` it
    reads live state from each sub-rewarder. With ``obs`` supplied it
    scores a pre-captured observation, leaving room for an offline pass.

    @param tf_idf:    Coverage rewarder; produces ``coverage_reward`` (R_cov).
    @param exit_code: Owns the exit-code file. Used both as a fallback when
                      stderr classification is unavailable, and as the
                      ``valid`` cross-check (a clean run must also exit 0).
    @param validity:  Optional ``StderrValidityRewarder``. When wired,
                      drives the 3-way Eq. 2 split; when ``None``, the
                      aggregator collapses to the binary exit-code form.
    """

    def __init__(
        self,
        tf_idf:    TFIDFCoverageRewarder,
        exit_code: ExitCodeRewarder,
        validity:  "StderrValidityRewarder | None" = None,
    ) -> None:
        self.tf_idf    = tf_idf
        self.exit_code = exit_code
        self.validity  = validity

    def compute(self, obs: ExecutionObservation | None = None) -> RewardResult:
        """Score one execution; reads live state when ``obs`` is omitted."""
        if obs is None:
            stderr_text = self.validity.read() if self.validity is not None else None
            obs = ExecutionObservation(
                bitmap    = self.tf_idf.snapshot_bitmap(),
                exit_code = self.exit_code.read(),
                validity  = StderrValidityRewarder.classify(stderr_text),
            )

        cov_reward = self.tf_idf.result(obs)

        # Decide reward + reason from stderr if we have it; fall back to
        # exit-code semantics otherwise. The "valid" branch always requires
        # exit_code == 0 as a sanity cross-check.
        if obs.validity == "syntax":
            reward, reason, is_valid = -1.0, "syntax_error", False
        elif obs.validity == "semantic":
            reward, reason, is_valid = -0.5, "semantic_error", False
        elif obs.validity == "valid" and obs.exit_code == 0:
            reward, reason, is_valid = cov_reward, "valid", True
        else:
            # No stderr file (None) or stderr-says-valid-but-exit-non-zero
            # (defensive). Fall back to the binary exit-code path.
            if obs.exit_code == 0:
                reward, reason, is_valid = cov_reward, "valid", True
            else:
                reward = self.exit_code.invalid_reward
                reason = "missing_exit_code" if obs.exit_code is None else "nonzero_exit"
                is_valid = False

        return RewardResult(
            reward          = reward,
            coverage_reward = cov_reward,
            exit_code       = obs.exit_code,
            valid           = is_valid,
            reward_reason   = reason,
            novelty_score   = cov_reward,
            crash           = obs.exit_code is None,
            timeout         = None,
            validity        = obs.validity,
        )

    def clear_exit_code(self) -> None:
        """Pass-through for the AFL fuzz() pre-execution clear."""
        self.exit_code.clear()
        if self.validity is not None:
            self.validity.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# SHM attachment
# ---------------------------------------------------------------------------

def attach_trace_bits(bitmap_size: int) -> np.ndarray:
    """Attach to AFL++'s trace_bits SHM segment and return a numpy view.

    Must be called after AFL++ has set __AFL_SHM_ID in the process
    environment. AFL++ may update this from C after Python startup, so we
    consult libc.getenv() rather than relying only on os.environ.

    On older / SysV builds, __AFL_SHM_ID is an integer shmid and we attach
    with shmat(). On newer USEMMAP builds, it is a POSIX shm name (for
    example "/afl_<pid>_<rand>") and we attach with shm_open() + mmap().
    The returned array is a live view — copy it before the next execution
    overwrites the segment.

    @param bitmap_size: Coverage bitmap size in bytes (e.g. 65536 for 2**16).
    @return: uint8 numpy array of length bitmap_size backed by the SHM.
    """
    shm_ref = _get_process_env("__AFL_SHM_ID")
    if not shm_ref:
        raise RuntimeError(
            "__AFL_SHM_ID not set — is AFL++ running with instrumentation enabled?"
        )

    if _looks_like_int(shm_ref):
        return _attach_sysv_trace_bits(int(shm_ref), bitmap_size)
    return _attach_posix_trace_bits(shm_ref, bitmap_size)


def _get_process_env(name: str) -> str:
    """Read the live process environment via libc.getenv().

    AFL++ mutates the environment from C after the Python interpreter has
    already started, so Python's os.environ mapping may be stale here.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    getenv = libc.getenv
    getenv.restype = ctypes.c_char_p
    getenv.argtypes = [ctypes.c_char_p]

    value = getenv(name.encode())
    if value:
        return value.decode()
    return os.environ.get(name, "")


def _looks_like_int(value: str) -> bool:
    value = value.strip()
    return value.isdigit() or (value.startswith("-") and value[1:].isdigit())


def _attach_sysv_trace_bits(shm_id: int, bitmap_size: int) -> np.ndarray:
    """Attach to a SysV SHM segment identified by integer shmid."""

    libc  = ctypes.CDLL(None, use_errno=True)
    shmat = libc.shmat
    shmat.restype = ctypes.c_void_p
    shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    SHM_RDONLY = 0o10000   # read-only; AFL++ owns the segment
    ptr = shmat(shm_id, None, SHM_RDONLY)
    if ptr is None or ptr == ctypes.c_void_p(-1).value:
        errno = ctypes.get_errno()
        raise OSError(errno, f"shmat failed for SHM id {shm_id}: errno={errno}")

    ArrayType = ctypes.c_uint8 * bitmap_size
    buf = ArrayType.from_address(ptr)
    return np.frombuffer(buf, dtype=np.uint8)


def _attach_posix_trace_bits(shm_name: str, bitmap_size: int) -> np.ndarray:
    """Attach to a POSIX shared-memory object exported by USEMMAP builds."""
    libc = ctypes.CDLL(None, use_errno=True)

    shm_open = libc.shm_open
    shm_open.restype = ctypes.c_int
    shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]

    mmap_fn = libc.mmap
    mmap_fn.restype = ctypes.c_void_p
    mmap_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]

    close = libc.close
    close.restype = ctypes.c_int
    close.argtypes = [ctypes.c_int]

    fd = shm_open(shm_name.encode(), os.O_RDONLY, 0)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"shm_open failed for {shm_name}: errno={errno}")

    try:
        ptr = mmap_fn(
            None,
            bitmap_size,
            py_mmap.PROT_READ,
            py_mmap.MAP_SHARED,
            fd,
            0,
        )
    finally:
        close(fd)

    if ptr is None or ptr == ctypes.c_void_p(-1).value:
        errno = ctypes.get_errno()
        raise OSError(errno, f"mmap failed for {shm_name}: errno={errno}")

    ArrayType = ctypes.c_uint8 * bitmap_size
    buf = ArrayType.from_address(ptr)
    return np.frombuffer(buf, dtype=np.uint8)
