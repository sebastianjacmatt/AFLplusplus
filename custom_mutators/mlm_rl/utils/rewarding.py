"""Reward bucketing, label conversion, and coverage-weighted reward computation.

Rewards in CovRL are scalar values in [-1.0, 1.0]:
  syntax_error   → -1.0
  semantic_error → -0.5
  valid          → sigmoid(log(Σ tf_i · idf_i))   (CWR, paper Eq. 5)

These are discretized into NUM_LABELS equal buckets for the critic's
classification loss.  label_to_score converts a predicted bucket index back
to the bucket midpoint for use as a scalar signal in the actor's PPO
advantage computation.

Rewarder maintains a persistent DataFrame across calls (keyed by the is_orig
flag) so that afl-showmap is run only on newly submitted entries — matching
CovRL's fit() / is_orig semantics.  Parallel execution uses a multiprocessing
Pool (module-level _showmap_worker for pickling compatibility).

Override _compute_valid_rewards() in a subclass to plug in an alternative
reward strategy (e.g. GRPO group-relative normalisation) without touching
the showmap pipeline or IDF machinery.
"""
import logging
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

NUM_LABELS   = 8
BUCKET_WIDTH = 2.0 / NUM_LABELS   # 0.25 over [-1, 1]


def score_to_label(score):
    """Map a scalar reward in [-1.0, 1.0] to a class index in [0, NUM_LABELS-1]."""
    return min(NUM_LABELS - 1, int((float(score) + 1.0) / BUCKET_WIDTH))


def label_to_score(label):
    """Map a class index back to its bucket midpoint."""
    return -1.0 + (label + 0.5) * BUCKET_WIDTH


# ---------------------------------------------------------------------------
# Error classification — keyed by interpreter target name
# ---------------------------------------------------------------------------

_ERROR_MAPS = {
    "v8": {
        "SyntaxError:":               "syntaxError",
        "ReferenceError:":            "semanticError",
        "TypeError:":                 "semanticError",
        "RangeError:":                "semanticError",
        "URIError:":                  "semanticError",
        "Error loading file":         "semanticError",
        "Error executing file":       "semanticError",
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
        "SyntaxError:":               "syntaxError",
        "ReferenceError:":            "semanticError",
        "TypeError:":                 "semanticError",
        "RangeError:":                "semanticError",
        "URIError:":                  "semanticError",
        "Error in opening file:":     "semanticError",
    },
    "jerry": {
        "SyntaxError":                "syntaxError",
        "ReferenceError":             "semanticError",
        "TypeError":                  "semanticError",
        "RangeError":                 "semanticError",
        "URIError":                   "semanticError",
        "Unhandled exception":        "semanticError",
    },
}


def _classify_error(stderr_text, error_map):
    for pattern, error_type in error_map.items():
        if pattern in stderr_text:
            return error_type
    return None


# ---------------------------------------------------------------------------
# Pool worker — module-level so multiprocessing can pickle it
# ---------------------------------------------------------------------------

def _showmap_worker(args):
    """
    Run afl-showmap for a single file and parse the coverage bitmap.

    Returns a dict with keys:
        file_id (str)
        bitmap  (np.ndarray[int32] | None) — None when an error reward is set
        reward  (float | None)             — None when TF-IDF must be computed
    """
    afl_showmap_path, interpreter_path, tmp_dir, bitmap_size, error_map, file_id, content = args

    input_path  = os.path.join(tmp_dir, f"{file_id}.js")
    showmap_out = os.path.join(tmp_dir, f"cov_{file_id}")

    try:
        with open(input_path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(content)
    except OSError as exc:
        raise OSError(f"Failed to write input file for {file_id}: {exc}")

    cmd = [
        afl_showmap_path, "-o", showmap_out,
        "-m", "none", "-t", "5000",
        "--", interpreter_path, input_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=20)
    except Exception as exc:
        raise RuntimeError(f"afl-showmap subprocess failed for {file_id}: {exc}")

    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    stdout_text = proc.stdout.decode("utf-8", errors="replace")

    # Discrepancy 3: interpreter crash — detected before error-string lookup,
    # matching CovRL's SEGV/assertion guard.  Bitmap may be partial or absent.
    if "SEGV" in stderr_text or "assertion" in stdout_text:
        return {"file_id": file_id, "bitmap": None, "reward": -0.5}

    error_type = _classify_error(stderr_text, error_map)

    # Discrepancy 2: always attempt to read the bitmap regardless of error type,
    # matching CovRL which includes error-entry bitmaps in IDF computation.
    bitmap         = np.zeros(bitmap_size, dtype=np.int32)
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
        # For error entries this is expected (interpreter may not have produced
        # any coverage output).  For valid entries we raise below.
        bitmap = None

    if error_type == "syntaxError":
        return {"file_id": file_id, "bitmap": bitmap, "reward": -1.0}
    if error_type is not None:
        return {"file_id": file_id, "bitmap": bitmap, "reward": -0.5}

    # Valid entry — showmap output must be present.
    if not bitmap_readable:
        raise OSError(f"Failed to read afl-showmap output for {file_id}")

    return {"file_id": file_id, "bitmap": bitmap, "reward": None}


# ---------------------------------------------------------------------------
# Rewarder
# ---------------------------------------------------------------------------

class Rewarder:
    """
    afl-showmap reward pipeline with persistent dataset tracking and pooled execution.

    The persistent DataFrame (flagged with is_orig) ensures afl-showmap is run only
    on entries not previously submitted — mirroring CovRL's fit() / is_orig semantics.
    Parallel afl-showmap calls are dispatched through a ThreadPoolExecutor: subprocess.run
    releases the GIL during os.waitpid, giving true I/O parallelism with no process-spawn
    overhead and no AFL++ fork-server conflicts.

    IDF weighting follows CovRL's update_idf (paper Eq. 5):
        df_counts = per-edge document-frequency over all valid bitmaps ever seen
        new_idf   = (log(N / (1 + df_counts)) / sqrt(bitmap_size)) * (1 - alpha)
        idf       = alpha * old_idf + new_idf

    where N is the running count of all entries ever submitted (including failures).

    Subclass and override _compute_valid_rewards() to implement an alternative
    reward strategy without touching the showmap pipeline or IDF state.
    For GRPO: override to receive bitmaps, compute raw TF-IDF scores, then apply
    group-relative normalisation using group metadata stored in self._dataset.
    """

    def __init__(
        self,
        tmp_dir,
        bitmap_size=131072,
        idf_alpha=0.6,
        n_workers=1,
    ):
        """
        @param tmp_dir:     Directory for temporary input files and showmap output.
        @param bitmap_size: AFL++ coverage map size (default 2^17 = 131072).
        @param idf_alpha:   EMA smoothing factor for IDF update (CovRL: 0.6).
        @param n_workers:   Number of parallel Pool workers for afl-showmap.
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
        self._error_map     = _ERROR_MAPS.get("jerry", {})  # TODO: get from config
        self._tmp_dir       = tmp_dir
        self._bitmap_size   = bitmap_size
        self._idf_alpha     = idf_alpha
        self._map_size_pow2 = math.sqrt(bitmap_size)
        self._n_workers     = n_workers

        # IDF vector — updated from the full bitmap history on every cycle.
        self._idf_vector     = np.zeros(bitmap_size, dtype=float)
        # All bitmaps accumulated across compute() calls (valid and error entries), used for IDF.
        self._bitmap_history = []   # list[np.ndarray shape (bitmap_size,), int32]
        self._total_seen     = 0    # total entries ever submitted (including failures)

        # Persistent dataset with is_orig flag, mirroring CovRL's dataset.
        # Columns: file_id, data, is_orig, bitmap, reward
        # Extra columns from the caller's DataFrame (e.g. group_id for GRPO)
        # are preserved automatically by pd.concat.
        self._dataset = pd.DataFrame(
            columns=["file_id", "data", "is_orig", "bitmap", "reward"]
        )

    @property
    def idf_vector(self):
        """Read-only view of the current IDF weight vector."""
        return self._idf_vector

    @property
    def dataset(self):
        """Read-only view of the full persistent dataset."""
        return self._dataset

    def compute(self, new_mutations_df):
        """
        Merge new mutations into the persistent dataset, run afl-showmap on
        unprocessed rows only, update the IDF from the full bitmap history,
        and assign scalar rewards.

        Rows are marked is_orig=False on entry and is_orig=True after processing;
        subsequent calls will never re-submit already-processed entries to showmap.
        This matches CovRL's fit() / update_idf / is_orig semantics.

        Extra columns in new_mutations_df (e.g. group_id) are preserved in
        self._dataset and are accessible to _compute_valid_rewards() overrides.

        @param new_mutations_df: DataFrame with at least ["file_id", "data"].
        @return: new_mutations_df with "reward" column added.
        """
        os.makedirs(self._tmp_dir, exist_ok=True)

        # Merge new rows into the persistent dataset with is_orig=False.
        new_rows            = new_mutations_df.copy()
        new_rows["is_orig"] = False
        new_rows["bitmap"]  = None
        new_rows["reward"]  = 0.0
        self._dataset = pd.concat([self._dataset, new_rows], ignore_index=True)

        # Select only unprocessed entries — mirrors CovRL's dataset[~dataset["is_orig"]].
        # Cast to bool explicitly: pd.concat with an empty DataFrame produces object
        # dtype for is_orig, and ~object_bool yields -1/-2 (bitwise NOT) instead of
        # a proper boolean mask, causing pandas to treat it as column labels.
        unprocessed_mask = ~self._dataset["is_orig"].astype(bool)
        unprocessed      = self._dataset[unprocessed_mask]

        if unprocessed.empty:
            log.info("[rewarder] no new entries to process")
            return new_mutations_df.assign(reward=0.0)

        # Build pool worker arguments (one tuple per unprocessed entry).
        worker_args = [
            (
                self._afl_showmap_path,
                self._interpreter_path,
                self._tmp_dir,
                self._bitmap_size,
                self._error_map,
                str(row["file_id"]),
                row["data"],
            )
            for _, row in unprocessed.iterrows()
        ]

        log.info(
            "[rewarder] afl-showmap on %d new entries (workers=%d)",
            len(worker_args), self._n_workers,
        )
        # ThreadPoolExecutor runs _showmap_worker calls in parallel threads.
        # subprocess.run releases the GIL while waiting for each afl-showmap
        # child to exit (os.waitpid is a blocking syscall), so threads achieve
        # true parallelism with no process-spawn overhead and no AFL++ fork-server
        # conflicts (no Python process is forked; only afl-showmap subprocesses
        # are created, which are independent of AFL++'s fork-server state).
        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            raw_results = list(
                tqdm(
                    executor.map(_showmap_worker, worker_args),
                    total=len(worker_args),
                    desc="showmap",
                    unit="file",
                )
            )

        # result_map keyed by file_id for O(1) write-back.
        result_map = {r["file_id"]: r for r in raw_results}

        # Accumulate valid bitmaps and update the total seen count.
        self._bitmap_history.extend(
            r["bitmap"] for r in raw_results if r["bitmap"] is not None
        )
        self._total_seen += len(raw_results)

        # Recompute IDF from the full accumulated bitmap history (CovRL: update_idf).
        self._update_idf()

        # Assign TF-IDF rewards to valid entries using the freshly updated IDF.
        # reward=None marks valid entries; errors already carry -1.0 / -0.5.
        valid_results = [r for r in raw_results if r["reward"] is None]
        if valid_results:
            valid_bitmaps = [r["bitmap"] for r in valid_results]
            valid_rewards = self._compute_valid_rewards(valid_bitmaps)
            for r, reward in zip(valid_results, valid_rewards):
                r["reward"] = reward

        # Write results back into the persistent dataset and mark is_orig=True.
        for idx in unprocessed.index:
            file_id = str(self._dataset.at[idx, "file_id"])
            r = result_map[file_id]
            self._dataset.at[idx, "is_orig"] = True
            self._dataset.at[idx, "bitmap"]  = r["bitmap"]
            self._dataset.at[idx, "reward"]  = r["reward"]

        # Return the caller's DataFrame with rewards attached.
        fid_to_reward = {
            str(self._dataset.at[idx, "file_id"]): self._dataset.at[idx, "reward"]
            for idx in unprocessed.index
        }
        result_df           = new_mutations_df.copy()
        result_df["reward"] = result_df["file_id"].astype(str).map(fid_to_reward)
        return result_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_idf(self):
        """
        Recompute the IDF vector from the full accumulated bitmap history.
        Matches CovRL's update_idf (paper Eq. 5).
        """
        if not self._bitmap_history:
            return
        bitmap_matrix    = np.vstack(self._bitmap_history)
        df_counts        = np.sum(bitmap_matrix > 0, axis=0).astype(float)
        total_docs       = float(self._total_seen)
        new_idf          = (
            np.log(total_docs / (1.0 + df_counts)) / self._map_size_pow2
        ) * (1.0 - self._idf_alpha)
        self._idf_vector = self._idf_alpha * self._idf_vector + new_idf

    def _compute_valid_rewards(self, bitmaps):
        """
        Compute scalar rewards for a list of valid (non-error) bitmaps.

        Override in a subclass to implement an alternative reward strategy.
        The default applies per-entry TF-IDF scoring (CovRL Eq. 5) against the
        current IDF vector, then sigmoid-compresses to (0, 1].

        For GRPO: override to compute raw TF-IDF scores across the batch, then
        apply group-relative normalisation.  Group metadata (e.g. group_id) is
        accessible via self._dataset if it was present in the input DataFrame.

        @param bitmaps: list[np.ndarray] of shape (bitmap_size,), one per valid entry,
                        in the same order as the unprocessed rows submitted to compute().
        @return:        list[float] of scalar rewards, same length as bitmaps.
        """
        rewards = []
        for bitmap in bitmaps:
            tfidf = float(np.dot(bitmap, self._idf_vector))
            if tfidf > 0.0:
                log_score = math.log(tfidf)
                reward    = round(1.0 / (1.0 + math.exp(-log_score)), 2)
            else:
                # Discrepancy 1: match CovRL's log(0)→0 substitution so that
                # sigmoid(0) = 0.5 is returned, not 0.0.
                reward = 0.5
            rewards.append(reward)
        return rewards
