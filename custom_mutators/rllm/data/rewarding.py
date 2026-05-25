"""Coverage-weighted reward computation via afl-showmap + TF-IDF.

Mirrors CovRL-Fuzz/covrl/models/rewarding.py adapted for rllm's u16 queue
format and existing classify_stderr() validity interface.
"""

from __future__ import annotations

import math
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from data.validity import classify_stderr

BITMAP_SIZE = 131072  # afl-showmap default edge map size


class Rewarding:
    """Runs afl-showmap on generated JS, scores via TF-IDF over coverage bitmap.

    IDF is updated with an EMA after each rollout batch and persisted to
    idf_path so it survives across finetune cycles.
    """

    def __init__(
        self,
        afl_showmap: Path,
        target_bin: Path,
        tmp_dir: Path,
        bitmap_size: int = BITMAP_SIZE,
        alpha: float = 0.6,
        idf_path: Path | None = None,
    ):
        self.afl_showmap = afl_showmap
        self.target_bin = target_bin
        self.tmp_dir = tmp_dir
        self.bitmap_size = bitmap_size
        self.map_size_pow2 = math.sqrt(bitmap_size)
        self.alpha = alpha
        self.idf_path = idf_path
        self.idf = np.zeros(bitmap_size)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        if idf_path and idf_path.exists():
            with open(idf_path, "rb") as f:
                self.idf = pickle.load(f)

    def run(self, js_bytes: bytes, name: str) -> tuple[str, np.ndarray | None]:
        """Execute afl-showmap on js_bytes and return (validity, bitmap).

        bitmap is None when the run crashed or timed out without producing a
        coverage file (validity penalties still apply in that case).
        """
        js_path = self.tmp_dir / f"{name}.js"
        cov_path = self.tmp_dir / f"{name}.cov"
        js_path.write_bytes(js_bytes)

        cmd = [
            str(self.afl_showmap),
            "-o", str(cov_path),
            "-m", "none",
            "-t", "5000",
            "--",
            str(self.target_bin),
            str(js_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=20)
            stderr = result.stderr.decode("utf-8", errors="replace")
            stdout = result.stdout.decode("utf-8", errors="replace")
        except (subprocess.TimeoutExpired, OSError):
            return "semantic", None

        # Hard crash check before stderr classification (mirrors rewarding.py:99)
        if "SEGV" in stderr or "assertion" in stdout:
            return "semantic", None

        validity = classify_stderr(stderr)

        if validity != "valid" or not cov_path.exists():
            return validity, None

        bitmap = self._parse_coverage(cov_path)
        return validity, bitmap

    def _parse_coverage(self, cov_path: Path) -> np.ndarray:
        bitmap = np.zeros(self.bitmap_size, dtype=np.int32)
        try:
            for line in cov_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                edge = int(line.split(":")[0])
                if 0 <= edge < self.bitmap_size:
                    bitmap[edge] += 1
        except (OSError, ValueError):
            pass
        return bitmap

    def update_idf(self, bitmaps: np.ndarray) -> None:
        """EMA update of IDF over a batch of coverage bitmaps.

        Exact formula from CovRL rewarding.py:66-70.
        Only called with valid (non-None) bitmaps stacked as (N, bitmap_size).
        """
        df = np.sum(bitmaps > 0, axis=0)
        total_docs = bitmaps.shape[0]
        new_idf = (np.log(total_docs / (1 + df)) / self.map_size_pow2) * (1 - self.alpha)
        self.idf = self.alpha * self.idf + new_idf
        if self.idf_path:
            with open(self.idf_path, "wb") as f:
                pickle.dump(self.idf, f)

    def score(self, validity: str, bitmap: np.ndarray | None) -> float:
        """Map (validity, bitmap) to scalar reward.

        Validity penalties match CovRL's reward partition:
          syntax   → -1.0
          semantic → -0.5
          valid    → TF-IDF score in [0, 1] via log + sigmoid
        """
        if validity == "syntax":
            return -1.0
        if validity == "semantic":
            return -0.5
        if bitmap is None:
            # Valid run but no coverage file produced; sigmoid(0) = 0.5
            return 0.5
        s = float(np.dot(bitmap, self.idf))
        log_s = math.log(s) if s > 0.0 else 0.0
        return round(1.0 / (1.0 + math.exp(-log_s)), 2)
