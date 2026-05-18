"""Reward computation.

Three classes:

* ``TFIDFCoverageRewarder`` — coverage reward over the AFL++ bitmap
  (CovRL Eq. 7: EMA-smoothed TF·IDF).
* ``StderrValidityRewarder`` — classifies a target execution as
  valid / semantic-error / syntax-error from its stderr.
* ``Rewarder`` — composes the two into one scalar via the 3-tier reward
  rule. The combination rule lives here, not in the sub-rewarders
  (docs/design.md §3.3: "Sub-rewarders free of policy").
"""

import ctypes
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config import Config


class _SharedMemoryBitmap:
    """Attach AFL++'s SysV shared coverage bitmap via ``__AFL_SHM_ID``."""

    def __init__(self, size: int) -> None:
        shm_id_str = os.environ.get("__AFL_SHM_ID")
        if not shm_id_str:
            raise RuntimeError(
                "__AFL_SHM_ID not set; the rewarder must run under afl-fuzz."
            )
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.shmat.restype = ctypes.c_void_p
        addr = libc.shmat(int(shm_id_str), None, 0)
        if addr is None or addr == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "shmat failed")
        self._buf = (ctypes.c_ubyte * size).from_address(addr)

    def read(self) -> np.ndarray:
        return np.frombuffer(self._buf, dtype=np.uint8).copy()


class TFIDFCoverageRewarder:
    """TF·IDF coverage reward with EMA-smoothed IDF (CovRL Eq. 7)."""

    def __init__(self, bitmap_size: int, alpha: float) -> None:
        self.bitmap_size = bitmap_size
        self.alpha = alpha
        self._bitmap = _SharedMemoryBitmap(bitmap_size)
        self._df = np.zeros(bitmap_size, dtype=np.float32)
        self._idf = np.zeros(bitmap_size, dtype=np.float32)
        self._n = 0

    def score(self) -> float:
        bits = self._bitmap.read()
        hits = (bits > 0).astype(np.float32)
        self._n += 1
        self._df += hits

        new_idf = np.log((self._n + 1.0) / (self._df + 1.0))
        self._idf = self.alpha * self._idf + (1.0 - self.alpha) * new_idf

        tf = np.log1p(bits.astype(np.float32))
        reward = float((tf * self._idf * hits).sum())
        normaliser = float(self._idf.sum()) + 1e-8
        return reward / normaliser


class StderrValidityRewarder:
    """Classify the last execution's stderr as valid / semantic / syntax."""

    SYNTAX_MARKERS = ("SyntaxError", "ParseError", "syntax error")
    SEMANTIC_MARKERS = ("Error", "Exception", "Traceback")

    def __init__(self, stderr_path: str) -> None:
        self.stderr_path = stderr_path

    def classify(self) -> str:
        try:
            with open(self.stderr_path, "rb") as fh:
                err = fh.read()
        except FileNotFoundError:
            return "valid"
        if not err:
            return "valid"
        text = err.decode("utf-8", errors="replace")
        if any(m in text for m in self.SYNTAX_MARKERS):
            return "syntax"
        if any(m in text for m in self.SEMANTIC_MARKERS):
            return "semantic"
        return "valid"


class Rewarder:
    """Compose coverage + validity into one scalar reward."""

    SYNTAX_REWARD = -1.0
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
        if verdict == "syntax":
            return self.SYNTAX_REWARD
        if verdict == "semantic":
            return self.SEMANTIC_REWARD
        r_cov = self.tf_idf.score()
        b = self.cfg.validity_bonus
        return b + (1.0 - b) * r_cov
