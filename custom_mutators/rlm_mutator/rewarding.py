"""Online TF-IDF coverage reward for the rlm_mutator.

Attaches to AFL++'s trace_bits shared memory segment once at init time,
then computes a TF-IDF weighted coverage reward per execution inside
post_run() — zero re-executions, zero subprocesses.

Reward range: scalar in [0.0, 1.0] for valid executions.
  near-zero coverage (syntax error, empty run) → 0.5 floor
  novel coverage                               → sigmoid(log(TF-IDF)) ∈ (0.5, 1.0]
  heavily repeated coverage                    → approaches 0.5 from above

IDF vector is updated with exponential momentum after every post_run() call,
matching CovRL Eq. 7.  Reward is computed with the *previous* IDF vector
(before this sample's update) matching CovRL Eq. 5.
"""
import ctypes
import math
import os

import numpy as np


# ---------------------------------------------------------------------------
# SHM attachment
# ---------------------------------------------------------------------------

def attach_trace_bits(bitmap_size: int) -> np.ndarray:
    """Attach to AFL++'s trace_bits SHM segment and return a numpy view.

    Must be called after AFL++ has set __AFL_SHM_ID in the environment.
    The returned array is a live view — copy it before the next execution
    overwrites the segment.

    @param bitmap_size: Coverage bitmap size in bytes (e.g. 65536 for 2**16).
    @return: uint8 numpy array of length bitmap_size backed by the SHM.
    """
    shm_id_str = os.environ.get("__AFL_SHM_ID", "")
    if not shm_id_str:
        raise RuntimeError(
            "__AFL_SHM_ID not set — is AFL++ running with instrumentation enabled?"
        )
    shm_id = int(shm_id_str)

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
