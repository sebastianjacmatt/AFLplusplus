"""Reward observation and shaping for rlm_mutator.

CoverageRewarder owns the execution-side observations needed to score one
mutation: it lazily attaches AFL++'s trace_bits shared-memory bitmap, reads
and clears the exit-code file written by exit_hook.so, updates online IDF
state, and returns a structured RewardResult.

RewardResult.reward is the scalar consumed by PPO/GRPO. The remaining fields
are diagnostics carried into rollout CSV/logging so reward experiments can add
or compare components without changing the trainer contract.

Coverage reward follows the CovRL-style online TF-IDF signal:
  near-zero/old coverage -> 0.5 floor
  novel weighted coverage -> sigmoid(log(TF-IDF)) in (0.5, 1.0]
  repeated coverage -> approaches the floor as IDF momentum adapts

The IDF vector updates after every reward computation. The reward for the
current sample is computed from the previous IDF vector, then the vector is
updated for the next sample.
"""
import ctypes
import math
import os
import mmap as py_mmap
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    """Structured reward output.

    ``reward`` is the scalar consumed by RL. The remaining fields explain how
    that scalar was produced and are carried into rollout logging.
    """

    reward: float
    coverage_reward: float | None = None
    exit_code: int | None = None
    valid: bool = False
    reward_reason: str | None = None
    novelty_score: float | None = None
    crash: bool | None = None
    timeout: bool | None = None

    def as_record_fields(self) -> dict:
        return asdict(self)

    @classmethod
    def diagnostic_field_names(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls) if field.name != "reward")


class CoverageRewarder:
    """Combine online IDF coverage reward with execution outcome policy."""

    def __init__(
        self,
        idf: "OnlineIDF",
        bitmap_size: int,
        exit_code_path: str,
        invalid_coverage_scale: float,
        invalid_exit_penalty: float,
        missing_exit_penalty: float,
    ):
        self.idf = idf
        self.bitmap_size = bitmap_size
        self.exit_code_path = exit_code_path
        self.invalid_coverage_scale = invalid_coverage_scale
        self.invalid_exit_penalty = invalid_exit_penalty
        self.missing_exit_penalty = missing_exit_penalty
        self._trace_bits_view = None

    def compute(self) -> RewardResult:
        bitmap = self._trace_bitmap()
        exit_code = self._read_exit_code()
        coverage_reward = self.idf.reward(bitmap)
        valid = exit_code == 0
        if valid:
            reward = coverage_reward
            reason = "valid"
        else:
            penalty = (
                self.missing_exit_penalty
                if exit_code is None
                else self.invalid_exit_penalty
            )
            reward = self.invalid_coverage_scale * coverage_reward - penalty
            reason = "missing_exit_code" if exit_code is None else "nonzero_exit"

        return RewardResult(
            reward=reward,
            coverage_reward=coverage_reward,
            exit_code=exit_code,
            valid=valid,
            reward_reason=reason,
            novelty_score=coverage_reward,
            crash=exit_code is None,
            timeout=None,
        )

    def _trace_bitmap(self) -> np.ndarray:
        if self._trace_bits_view is None:
            self._trace_bits_view = attach_trace_bits(self.bitmap_size)
        return self._trace_bits_view.copy()

    def clear_exit_code(self) -> None:
        """Remove stale exit-code output before/after a child execution."""
        try:
            os.remove(self.exit_code_path)
        except FileNotFoundError:
            pass

    def _read_exit_code(self) -> int | None:
        """Read and clear the exit code written by exit_hook.so."""
        try:
            with open(self.exit_code_path) as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None
        finally:
            self.clear_exit_code()


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


# ---------------------------------------------------------------------------
# Online IDF state
# ---------------------------------------------------------------------------

class OnlineIDF:
    """Maintains a momentum-smoothed IDF weight vector updated per execution.

    Follows CovRL Eq. 4 (IDF definition) and Eq. 7 (momentum update).

    @param bitmap_size: Number of edges in the coverage bitmap.
    @param alpha:       Momentum rate α ∈ [0, 1].  CovRL uses 0.6.
    """

    def __init__(self, bitmap_size: int, alpha: float = 0.6):
        self._bitmap_size  = bitmap_size
        self._alpha        = alpha
        self._map_scale    = math.sqrt(bitmap_size)   # √M in Eq. 4
        self._total_seen   = 0
        self._idf_prev     = np.zeros(bitmap_size, dtype=np.float32)  # IDF_{t-1}
        self._idf_cur      = np.zeros(bitmap_size, dtype=np.float32)  # IDF_t (working)

    def reward(self, bitmap: np.ndarray) -> float:
        """Compute TF-IDF reward for one execution and update IDF state.

        Uses the *previous* IDF vector to score this sample (Eq. 5), then
        updates the IDF with momentum (Eq. 7) so the next call uses the
        updated weights.

        @param bitmap: uint8 array — snapshot of trace_bits for this execution.
        @return: Scalar reward in [0.5, 1.0].  0.5 is the floor for zero/low coverage.
        """
        # TF_cov: unique coverage map — binary presence per edge (Eq. 3)
        tf_cov = (bitmap > 0).astype(np.float32)

        # R_TFIDF = log(Σ tf_i,t · idf_i,t-1)  — Eq. 5
        tfidf = float(np.dot(tf_cov, self._idf_prev))
        if tfidf > 0.0:
            r_tfidf = math.log(tfidf)
            # R_cov = σ(R_TFIDF)  — Eq. 6
            reward = 1.0 / (1.0 + math.exp(-r_tfidf))
        else:
            reward = 0.5   # floor: no new coverage signal

        self._update(tf_cov)
        return round(reward, 4)

    def _update(self, tf_cov: np.ndarray) -> None:
        """Update IDF vector with exponential momentum — Eq. 7."""
        self._total_seen += 1
        n = float(self._total_seen)

        # IDF_t = (1/√M) · log(N / (1 + DF_cov))   where DF_cov = tf_cov for a single sample
        # This is an online approximation: treat this one bitmap as a document.
        # Over many samples the accumulated effect converges to the batch formula.
        new_idf = np.log(n / (1.0 + tf_cov)) / self._map_scale

        # Momentum blend: IDF_{t} ← α·IDF_{t-1} + (1-α)·new_IDF_t  — Eq. 7
        self._idf_cur   = self._alpha * self._idf_prev + (1.0 - self._alpha) * new_idf
        self._idf_prev  = self._idf_cur.copy()
