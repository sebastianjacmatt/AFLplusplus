"""Coverage-weighted reward pipeline for AFL++ mutations.

Rewards (scalar in [-1.0, 1.0]):
  syntax error   → -1.0
  semantic error → -0.5
  crash / SEGV   → -0.5
  valid          → sigmoid(log(TF-IDF))  CovRL Eq. 5; 0.5 when TF-IDF == 0

compute() reads the AFL++ mutation queue directly via load_mutation_corpus(),
runs afl-showmap only on entries not yet processed, and returns all rewarded
mutations accumulated so far. The trainer handles orig-corpus mixing separately.
"""
import logging
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_utils import load_mutation_corpus

log = logging.getLogger(__name__)

NUM_LABELS   = 8
BUCKET_WIDTH = 2.0 / NUM_LABELS


def score_to_label(score):
    """Map a scalar reward in [-1.0, 1.0] to a bucket index in [0, NUM_LABELS-1]."""
    return min(NUM_LABELS - 1, int((float(score) + 1.0) / BUCKET_WIDTH))


def label_to_score(label):
    """Map a bucket index back to its midpoint scalar."""
    return -1.0 + (label + 0.5) * BUCKET_WIDTH


_ERROR_MAPS = {
    "v8": {
        "SyntaxError:":         "syntaxError",
        "ReferenceError:":      "semanticError",
        "TypeError:":           "semanticError",
        "RangeError:":          "semanticError",
        "URIError:":            "semanticError",
        "Error loading file":   "semanticError",
        "Error executing file": "semanticError",
    },
    "jsc": {
        "Exception: SyntaxError:":    "syntaxError",
        "Exception: ReferenceError:": "semanticError",
        "Exception: TypeError:":      "semanticError",
        "Exception: RangeError:":     "semanticError",
        "Exception: URIError:":       "semanticError",
        "Could not open file:":       "semanticError",
    },
    "chakra": {
        "SyntaxError:":           "syntaxError",
        "ReferenceError:":        "semanticError",
        "TypeError:":             "semanticError",
        "RangeError:":            "semanticError",
        "URIError:":              "semanticError",
        "Error in opening file:": "semanticError",
    },
    "jerry": {
        "SyntaxError":         "syntaxError",
        "ReferenceError":      "semanticError",
        "TypeError":           "semanticError",
        "RangeError":          "semanticError",
        "URIError":            "semanticError",
        "Unhandled exception": "semanticError",
    },
}


def _classify_error(stderr_text, error_map):
    for pattern, error_type in error_map.items():
        if pattern in stderr_text:
            return error_type
    return None


def _showmap_worker(args):
    """
    Run afl-showmap for a single file. Module-level for ThreadPoolExecutor.

    Returns dict with keys: file_id, bitmap (ndarray or None), reward (float or None).
    reward=None means valid — TF-IDF reward is assigned after the IDF update.
    """
    afl_showmap_path, interpreter_path, tmp_dir, bitmap_size, error_map, file_id, content = args

    input_path  = os.path.join(tmp_dir, f"{file_id}.js")
    showmap_out = os.path.join(tmp_dir, f"cov_{file_id}")

    try:
        with open(input_path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(content)
    except OSError as exc:
        raise OSError(f"Failed to write input for {file_id}: {exc}")

    cmd = [
        afl_showmap_path, "-o", showmap_out, "-m", "none", "-t", "5000",
        "--", interpreter_path, input_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=20)
    except Exception as exc:
        raise RuntimeError(f"afl-showmap failed for {file_id}: {exc}")

    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    stdout_text = proc.stdout.decode("utf-8", errors="replace")

    # Interpreter crash — check before error-string classification
    if "SEGV" in stderr_text or "assertion" in stdout_text:
        return {"file_id": file_id, "bitmap": None, "reward": -0.5}

    error_type = _classify_error(stderr_text, error_map)

    # Always try to read the bitmap regardless of error type — error-entry
    # bitmaps (partial execution before the error) are included in IDF.
    bitmap          = np.zeros(bitmap_size, dtype=np.int32)
    bitmap_readable = False
    try:
        with open(showmap_out, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                edge_str, _, count_str = line.partition(":")
                edge = int(edge_str)
                if 0 <= edge < bitmap_size:
                    bitmap[edge] += int(count_str) if count_str else 1
        bitmap_readable = True
    except OSError:
        bitmap = None   # expected for error entries; raises below for valid ones

    if error_type == "syntaxError":
        return {"file_id": file_id, "bitmap": bitmap, "reward": -1.0}
    if error_type is not None:
        return {"file_id": file_id, "bitmap": bitmap, "reward": -0.5}

    if not bitmap_readable:
        raise OSError(f"Failed to read afl-showmap output for {file_id}")

    return {"file_id": file_id, "bitmap": bitmap, "reward": None}


class Rewarder:
    """
    Reads the AFL++ mutation queue, runs afl-showmap on unseen entries,
    and maintains persistent IDF state for TF-IDF reward computation.

    compute() is called once per finetune cycle with no arguments. It loads
    the current queue, processes only new entries (is_proc=False), and returns
    all rewarded mutations accumulated so far.

    Override _compute_valid_rewards() to change the reward strategy (e.g. GRPO).
    """

    def __init__(self, tmp_dir, bitmap_size=131072, idf_alpha=0.6, n_workers=1):
        """
        @param tmp_dir:     Directory for temporary showmap input/output files.
        @param bitmap_size: AFL++ coverage map size (default 2^17 = 131072).
        @param idf_alpha:   EMA smoothing factor for IDF update (CovRL: 0.6).
        @param n_workers:   Number of parallel showmap threads (ThreadPoolExecutor).
        """
        self._afl_showmap_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../../afl-showmap")
        )
        if not os.path.isfile(self._afl_showmap_path):
            raise FileNotFoundError(f"afl-showmap not found: {self._afl_showmap_path}")

        self._interpreter_path = os.path.expanduser(
            "~/Documents/data_store/engines/jerryscript/build/bin/jerry"
        )
        if not os.path.isfile(self._interpreter_path):
            raise FileNotFoundError(f"interpreter not found: {self._interpreter_path}")

        self._error_map     = _ERROR_MAPS.get("jerry", {})  # TODO: from config
        self._tmp_dir       = tmp_dir
        self._bitmap_size   = bitmap_size
        self._idf_alpha     = idf_alpha
        self._map_size_pow2 = math.sqrt(bitmap_size)
        self._n_workers     = n_workers

        self._idf_vector     = np.zeros(bitmap_size, dtype=float)
        self._bitmap_history = []
        self._total_seen     = 0

        # Persistent record of all mutations seen across cycles.
        # is_proc=True means afl-showmap has been run for that entry.
        self._dataset = pd.DataFrame(
            columns=["file_id", "data", "group_id", "is_proc", "bitmap", "reward"]
        )

    def compute(self):
        """
        Load the current AFL++ mutation queue, process new entries, and return
        all rewarded mutations seen so far.

        New entries (file_ids not yet in self._dataset) are run through
        afl-showmap in parallel, IDF is updated from the full bitmap history,
        and rewards are assigned. Already-processed entries are not rerun.

        @return: DataFrame with columns ["file_id", "data", "group_id", "reward"].
        """
        os.makedirs(self._tmp_dir, exist_ok=True)

        current = load_mutation_corpus()
        if current.empty:
            log.info("[rewarder] queue is empty")
            return current

        # Add file_ids not yet tracked
        seen_ids = set(self._dataset["file_id"].astype(str))
        new_rows = current[~current["file_id"].astype(str).isin(seen_ids)].copy()
        if not new_rows.empty:
            new_rows["is_proc"] = False
            new_rows["bitmap"]  = None
            new_rows["reward"]  = 0.0
            self._dataset = pd.concat([self._dataset, new_rows], ignore_index=True)

        unprocessed = self._dataset[~self._dataset["is_proc"].astype(bool)]
        if unprocessed.empty:
            log.info("[rewarder] no new entries to process")
            return self._dataset[["file_id", "data", "group_id", "reward"]].copy()

        worker_args = [
            (
                self._afl_showmap_path, self._interpreter_path,
                self._tmp_dir, self._bitmap_size, self._error_map,
                str(row["file_id"]), row["data"],
            )
            for _, row in unprocessed.iterrows()
        ]

        log.info("[rewarder] afl-showmap on %d entries (workers=%d)",
                 len(worker_args), self._n_workers)
        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            raw_results = list(tqdm(
                executor.map(_showmap_worker, worker_args),
                total=len(worker_args), desc="showmap", unit="file",
            ))

        result_map = {r["file_id"]: r for r in raw_results}
        self._bitmap_history.extend(
            r["bitmap"] for r in raw_results if r["bitmap"] is not None
        )
        self._total_seen += len(raw_results)
        self._update_idf()

        valid_results = [r for r in raw_results if r["reward"] is None]
        if valid_results:
            valid_rewards = self._compute_valid_rewards([r["bitmap"] for r in valid_results])
            for r, reward in zip(valid_results, valid_rewards):
                r["reward"] = reward

        for idx in unprocessed.index:
            file_id = str(self._dataset.at[idx, "file_id"])
            r = result_map[file_id]
            self._dataset.at[idx, "is_proc"] = True
            self._dataset.at[idx, "bitmap"]  = r["bitmap"]
            self._dataset.at[idx, "reward"]  = r["reward"]

        return self._dataset[["file_id", "data", "group_id", "reward"]].copy()

    def _update_idf(self):
        """Recompute IDF from the full accumulated bitmap history (CovRL Eq. 5)."""
        if not self._bitmap_history:
            return
        bitmap_matrix    = np.vstack(self._bitmap_history)
        df_counts        = np.sum(bitmap_matrix > 0, axis=0).astype(float)
        new_idf          = (
            np.log(float(self._total_seen) / (1.0 + df_counts)) / self._map_size_pow2
        ) * (1.0 - self._idf_alpha)
        self._idf_vector = self._idf_alpha * self._idf_vector + new_idf

    def _compute_valid_rewards(self, bitmaps):
        """
        Compute rewards for valid (non-error) bitmaps.
        Override in subclasses for alternative strategies (e.g. GRPO).

        Default: sigmoid(log(TF-IDF)), 0.5 when TF-IDF == 0 (CovRL Eq. 5).
        For GRPO: access self._dataset for group_id metadata when normalising.

        @param bitmaps: list[np.ndarray] in the same order as unprocessed rows.
        @return:        list[float], same length.
        """
        rewards = []
        for bitmap in bitmaps:
            tfidf = float(np.dot(bitmap, self._idf_vector))
            if tfidf > 0.0:
                reward = round(1.0 / (1.0 + math.exp(-math.log(tfidf))), 2)
            else:
                reward = 0.5
            rewards.append(reward)
        return rewards
