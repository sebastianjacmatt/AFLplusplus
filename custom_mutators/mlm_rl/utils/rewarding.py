"""Reward bucketing, label conversion, and coverage-weighted reward computation.

Rewards in CovRL are scalar values in [-1.0, 1.0]:
  syntax_error   → -1.0
  semantic_error → -0.5
  valid          → sigmoid(log(Σ tf_i · idf_i))   (CWR, paper Eq. 5)

These are discretized into NUM_LABELS equal buckets for the critic's
classification loss.  label_to_score converts a predicted bucket index back
to the bucket midpoint for use as a scalar signal in the actor's PPO
advantage computation.

Rewarder encapsulates the afl-showmap pipeline and all persistent IDF state.
The IDF vector is recomputed each cycle from the full accumulated bitmap
history (all valid bitmaps ever seen), matching CovRL's update_idf which
operates on all processed mutations rather than just the current batch.
"""
import math
import os
import subprocess

import numpy as np

NUM_LABELS   = 8
BUCKET_WIDTH = 2.0 / NUM_LABELS   # width 0.25 over [-1, 1]


def score_to_label(score):
    """Map a scalar reward in [-1.0, 1.0] to a class index in [0, NUM_LABELS-1]."""
    return min(NUM_LABELS - 1, int((float(score) + 1.0) / BUCKET_WIDTH))


def label_to_score(label):
    """Map a class index back to its bucket midpoint."""
    return -1.0 + (label + 0.5) * BUCKET_WIDTH


# ---------------------------------------------------------------------------
# Error classification — keyed by interpreter target name
# ---------------------------------------------------------------------------

# Each inner dict maps a stderr substring to either "syntaxError" or
# "semanticError".  First match wins (order matters only within each dict).
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
    """
    Return the error type string for the first matching pattern, or None.

    @type  stderr_text: str
    @param stderr_text: Decoded stderr from the interpreter run.

    @type  error_map:   dict
    @param error_map:   Pattern → error-type mapping for the target interpreter.

    @rtype:  str or None
    @return: "syntaxError", "semanticError", or None (no error detected).
    """
    for pattern, error_type in error_map.items():
        if pattern in stderr_text:
            return error_type
    return None


# ---------------------------------------------------------------------------
# Rewarder
# ---------------------------------------------------------------------------

class Rewarder:
    """
    afl-showmap reward pipeline with persistent IDF state.

    Stable configuration (binary paths, interpreter target, temp directory)
    is bound at construction.  compute(new_mutations_df) is called once per
    finetune cycle with the batch of new mutation entries.

    IDF weighting follows CovRL's update_idf (paper Eq. 5):

      df_counts = per-edge document-frequency over all valid bitmaps ever seen
      new_idf   = (log(N / (1 + df_counts)) / sqrt(bitmap_size)) * (1 - alpha)
      idf       = alpha * old_idf + new_idf

    where N is the running count of all entries ever submitted to compute()
    (including failed and error entries), matching CovRL's use of
    total_docs = dataset.shape[0].  The IDF is updated from the full
    accumulated bitmap history rather than only the current batch, which
    ensures that each new cycle's reward computation benefits from all
    coverage signal seen so far.

    Bitmaps are kept in memory across cycles as int32 arrays.  At AFL++
    map_size=2^17 each entry uses 512 KB; for runs accumulating thousands
    of mutations the total will grow proportionally.
    """

    def __init__(
        self,
        tmp_dir,
        bitmap_size=131072,
        idf_alpha=0.6,
    ):
        """
        @type  tmp_dir:     str
        @param tmp_dir:     Directory for temporary input files and showmap output
                            files.  Created on first compute() call.

        @type  bitmap_size: int
        @param bitmap_size: AFL++ coverage map size (default 2^17 = 131072).

        @type  idf_alpha:   float
        @param idf_alpha:   EMA smoothing factor for IDF update (CovRL: 0.6).
        """
        self._afl_showmap_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../../../afl-showmap")
        )
        self._interpreter_path = os.path.expanduser(
            "~/Documents/data_store/engines/jerryscript/build/bin/jerry"
        )
        self._error_map        = _ERROR_MAPS.get("jerry", {})
        self._tmp_dir          = tmp_dir
        self._bitmap_size      = bitmap_size
        self._idf_alpha        = idf_alpha
        self._map_size_pow2    = math.sqrt(bitmap_size)

        # IDF vector — updated from the full bitmap history on every cycle.
        self._idf_vector     = np.zeros(bitmap_size, dtype=float)
        # All valid bitmaps accumulated across cycles, used to recompute IDF
        # from the full history each call (CovRL update_idf alignment).
        self._bitmap_history = []   # list[np.ndarray shape (bitmap_size,), int32]
        self._total_seen     = 0    # total entries ever submitted, including failures

    @property
    def idf_vector(self):
        """Read-only view of the current IDF weight vector."""
        return self._idf_vector

    def compute(self, new_mutations_df):
        """
        Run afl-showmap on each row in new_mutations_df, update the IDF vector
        from the full accumulated bitmap history, and assign scalar rewards.

        Reward assignment:
          SyntaxError in stderr  → -1.0
          other runtime error    → -0.5
          showmap I/O failure    → 0.0
          valid (no error)       → sigmoid(log(TF-IDF)); 0.0 when TF-IDF == 0

        The IDF is updated before TF-IDF rewards are computed so that entries
        in the current batch benefit from the full accumulated coverage signal.

        @type  new_mutations_df: pd.DataFrame
        @param new_mutations_df: Rows with at least ["file_id", "data"] columns.
                                 Must contain only entries not previously submitted.

        @rtype:  pd.DataFrame
        @return: Copy of new_mutations_df with "reward" column added.
        """
        os.makedirs(self._tmp_dir, exist_ok=True)

        df      = new_mutations_df.copy()
        results = []   # {"reward": float or None, "bitmap": np.ndarray or None}
        # reward=None marks a valid entry whose TF-IDF reward is computed after
        # the IDF update below.

        for _, row in df.iterrows():
            file_id     = str(row["file_id"])
            content     = row["data"]
            input_path  = os.path.join(self._tmp_dir, f"{file_id}.js")
            showmap_out = os.path.join(self._tmp_dir, f"cov_{file_id}")

            try:
                with open(input_path, "w", encoding="utf-8", errors="replace") as fh:
                    fh.write(content)
            except OSError:
                results.append({"reward": 0.0, "bitmap": None})
                continue

            cmd = [
                self._afl_showmap_path, "-o", showmap_out,
                "-m", "none", "-t", "5000",
                "--", self._interpreter_path, input_path,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=20)
            except Exception:
                results.append({"reward": 0.0, "bitmap": None})
                continue

            stderr_text = proc.stderr.decode("utf-8", errors="replace")
            error_type  = _classify_error(stderr_text, self._error_map)
            if error_type == "syntaxError":
                results.append({"reward": -1.0, "bitmap": None})
                continue
            if error_type is not None:
                results.append({"reward": -0.5, "bitmap": None})
                continue

            # Parse edge:count lines written by afl-showmap to showmap_out
            bitmap = np.zeros(self._bitmap_size, dtype=np.int32)
            try:
                with open(showmap_out, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        edge_str, _, count_str = line.partition(":")
                        edge = int(edge_str)
                        if 0 <= edge < self._bitmap_size:
                            bitmap[edge] += int(count_str) if count_str else 1
            except OSError:
                results.append({"reward": 0.0, "bitmap": None})
                continue

            results.append({"reward": None, "bitmap": bitmap})

        # Accumulate new valid bitmaps and update total seen count
        self._bitmap_history.extend(
            r["bitmap"] for r in results if r["bitmap"] is not None
        )
        self._total_seen += len(results)

        # Recompute IDF from the full accumulated bitmap history.
        # df_counts is the per-edge document frequency over all valid bitmaps
        # ever seen; total_docs includes all entries (failures count as docs).
        if self._bitmap_history:
            bitmap_matrix    = np.vstack(self._bitmap_history)
            df_counts        = np.sum(bitmap_matrix > 0, axis=0).astype(float)
            total_docs       = float(self._total_seen)
            new_idf          = (
                np.log(total_docs / (1.0 + df_counts)) / self._map_size_pow2
            ) * (1.0 - self._idf_alpha)
            self._idf_vector = self._idf_alpha * self._idf_vector + new_idf

        # Assign TF-IDF rewards to valid entries using the freshly updated IDF
        for r in results:
            if r["reward"] is None:
                tfidf = float(np.dot(r["bitmap"], self._idf_vector))
                if tfidf > 0.0:
                    log_score  = math.log(tfidf)
                    r["reward"] = round(1.0 / (1.0 + math.exp(-log_score)), 2)
                else:
                    r["reward"] = 0.0

        df["reward"] = [r["reward"] for r in results]
        return df