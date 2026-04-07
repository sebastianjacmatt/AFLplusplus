"""AFL++ queue file loading helpers.

set_queue_dir()        — called once from mlm_rl.py when the queue path is known
load_queue_files()     — read all queue entries (orig + mutations)
load_orig_corpus()     — initial seeds only (filename contains "orig:")
load_mutation_corpus() — coverage-increasing mutations only (non-orig)

group_id is extracted from the AFL++ filename's "src:" field and identifies
which seed a mutation was generated from. Used for GRPO group-relative rewards.
"""
import os

import pandas as pd


_queue_dir = None


def set_queue_dir(path):
    """Set the AFL++ queue directory. Called once from mlm_rl.py."""
    global _queue_dir
    _queue_dir = path


def load_queue_files():
    """
    Read all AFL++ queue entries from the configured directory.

    AFL++ queue filenames follow the pattern:
      id:NNNNNN,src:MMMMMM,op:<op>,rep:<n>   (mutations)
      id:NNNNNN,orig:<seedname>               (initial seeds)

    @return: DataFrame with columns ["is_orig", "file_id", "data", "group_id"].
             group_id is the src: field for mutations, None for orig entries.
             Returns an empty DataFrame when the queue directory is not set.
    """
    if _queue_dir is None:
        return pd.DataFrame(columns=["is_orig", "file_id", "data", "group_id"])

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
        file_id  = parts.get("id", filename)
        is_orig  = "orig" in parts
        group_id = parts.get("src", None)
        filepath = os.path.join(_queue_dir, filename)

        try:
            with open(filepath, "rb") as fh:
                data = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue

        if data:
            records.append({
                "is_orig":  is_orig,
                "file_id":  file_id,
                "data":     data,
                "group_id": group_id,
            })

    return pd.DataFrame(records, columns=["is_orig", "file_id", "data", "group_id"])


def load_orig_corpus():
    """Load AFL++ initial seed entries (filename contains "orig:")."""
    df = load_queue_files()
    return df[df["is_orig"]].reset_index(drop=True)


def load_mutation_corpus():
    """Load AFL++ mutation entries (coverage-increasing, non-orig)."""
    df = load_queue_files()
    return df[~df["is_orig"]].reset_index(drop=True)
