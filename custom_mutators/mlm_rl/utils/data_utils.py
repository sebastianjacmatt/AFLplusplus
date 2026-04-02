"""AFL++ queue file loading helpers.

load_queue_files()     — read all queue entries (orig + mutation) from the AFL++ queue.
load_orig_corpus()     — load only entries whose filename contains "orig:".
load_mutation_corpus() — load only non-orig entries.

The AFL++ queue directory is set once via set_queue_dir(), called from mlm_rl.py
when the path first becomes available (e.g. os.path.dirname(filename) in queue_get).
All loaders read from that directory directly; no paths are passed by callers.

AFL++ filenames encode whether an entry is an initial seed ("orig:") or a
coverage-increasing mutation.  These functions are pure readers: they load and
classify by filename only.  All cycle logic is the caller's responsibility.

Data is returned as bytes decoded to text (UTF-8, errors replaced).
Tokenization is NOT performed here; callers handle that in their Dataset paths.
"""
import os

import pandas as pd


_queue_dir = None


def set_queue_dir(path):
    """
    Set the AFL++ queue directory.  Called once from mlm_rl.py when the path
    first becomes available (e.g. os.path.dirname(filename) in queue_get).

    @type  path: str
    @param path: Absolute path to the AFL++ output queue directory.
    """
    global _queue_dir
    _queue_dir = path


def load_queue_files():
    """
    Read all AFL++ queue entries from the configured queue directory.

    AFL++ queue filenames follow the pattern:
      id:NNNNNN,src:MMMMMM,op:<op>,rep:<n>   (mutated entries)
      id:NNNNNN,orig:<seedname>               (initial seeds)

    @rtype:  pd.DataFrame
    @return: DataFrame with columns ["is_orig", "file_id", "data"].
             Returns an empty DataFrame when the queue directory is not set.
    """
    if _queue_dir is None:
        return pd.DataFrame([], columns=["is_orig", "file_id", "data"])

    records = []

    for filename in sorted(os.listdir(_queue_dir)):
        if not filename.startswith("id:"):
            continue

        parts = {
            k: v
            for part in filename.split(",")
            if ":" in part
            for k, v in [part.split(":", 1)]
        }
        file_id = parts.get("id", filename)
        is_orig = "orig" in parts
        filepath = os.path.join(_queue_dir, filename)

        try:
            with open(filepath, "rb") as fh:
                data = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue

        if data:
            records.append({"is_orig": is_orig, "file_id": file_id, "data": data})

    return pd.DataFrame(records, columns=["is_orig", "file_id", "data"])


def load_orig_corpus():
    """
    Load AFL++ queue entries whose filename contains "orig:".

    These are the initial seeds AFL++ copies from -i into the queue at startup.
    They serve as the clean reference data for the 4:1 mixing step in training.

    @rtype:  pd.DataFrame
    @return: DataFrame with columns ["is_orig", "file_id", "data"].
             All rows have is_orig == True.
    """
    df = load_queue_files()
    return df[df["is_orig"]].reset_index(drop=True)


def load_mutation_corpus():
    """
    Load AFL++ queue entries that are NOT initial seeds (non-"orig:").

    Every non-orig entry in the AFL++ queue increased coverage by definition —
    AFL++ only adds entries to the queue when they do.

    @rtype:  pd.DataFrame
    @return: DataFrame with columns ["is_orig", "file_id", "data"].
             All rows have is_orig == False.
    """
    df = load_queue_files()
    return df[~df["is_orig"]].reset_index(drop=True)
