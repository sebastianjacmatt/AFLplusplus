"""AFL++ queue loading and deduplication helpers.

load_queue_files()      — read all queue entries (orig + mutation) from a dir.
load_orig_corpus()      — load only entries whose filename contains "orig:".
load_mutation_corpus()  — load only non-orig entries, with optional dedup.

Data is returned as bytes decoded to text (UTF-8, errors replaced).
Tokenization is NOT performed here; callers handle that in their Dataset paths.
Deduplication against previously seen entries is the caller's responsibility
(pass the current known_ids set from the accumulated mutation dataset).
"""
import os

import pandas as pd


def load_queue_files(corpus_dir, known_ids=None):
    """
    Read AFL++ queue entries from corpus_dir into a DataFrame, skipping
    any entry whose file_id is already in known_ids.

    AFL++ queue filenames follow the pattern:
      id:NNNNNN,src:MMMMMM,op:<op>,rep:<n>   (mutated entries)
      id:NNNNNN,orig:<seedname>               (initial seeds)

    @type  corpus_dir: str or None
    @param corpus_dir: AFL++ output queue directory.
                       Returns an empty DataFrame immediately when None.

    @type  known_ids:  set or None
    @param known_ids:  File IDs already loaded in previous cycles.
                       Pass set(mutation_dataset["file_id"]) to deduplicate.
                       Deduplication is skipped when None.

    @rtype:  pd.DataFrame
    @return: DataFrame with columns ["is_orig", "file_id", "data"].
    """
    if corpus_dir is None:
        return pd.DataFrame([], columns=["is_orig", "file_id", "data"])

    if known_ids is None:
        known_ids = set()

    records = []

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.startswith("id:"):
            continue

        parts = {
            k: v
            for part in filename.split(",")
            if ":" in part
            for k, v in [part.split(":", 1)]
        }
        file_id = parts.get("id", filename)
        if file_id in known_ids:
            continue

        is_orig  = "orig" in parts
        filepath = os.path.join(corpus_dir, filename)
        try:
            with open(filepath, "rb") as fh:
                data = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue

        if data:
            records.append({"is_orig": is_orig, "file_id": file_id, "data": data})

    return pd.DataFrame(records, columns=["is_orig", "file_id", "data"])


def load_orig_corpus(corpus_dir):
    """
    Load all AFL++ queue entries whose filename contains "orig:".

    These are the initial seeds written by AFL++ when it copies the -i corpus
    into the queue.  They serve as the clean reference data for the 4:1 mixing
    step in _make_critic_dataset.

    Data is read as raw bytes and decoded to text (UTF-8, errors replaced).
    No tokenization is performed.

    @type  corpus_dir: str or None
    @param corpus_dir: AFL++ output queue directory.
                       Returns an empty DataFrame when None.

    @rtype:  pd.DataFrame
    @return: DataFrame with columns ["is_orig", "file_id", "data"].
             All rows have is_orig == True.
    """
    df = load_queue_files(corpus_dir)
    return df[df["is_orig"]].reset_index(drop=True)


def load_mutation_corpus(corpus_dir, known_ids=None):
    """
    Load AFL++ queue entries that are NOT initial seeds (non-"orig:"), with
    optional incremental deduplication against previously seen file_ids.

    These are the AFL++-generated mutations that accumulate across fuzzing
    cycles.  Only entries whose file_id is absent from known_ids are returned
    so that each finetune cycle processes only new queue additions.

    Data is read as raw bytes and decoded to text (UTF-8, errors replaced).
    No tokenization is performed.

    @type  corpus_dir: str or None
    @param corpus_dir: AFL++ output queue directory.
                       Returns an empty DataFrame when None.

    @type  known_ids:  set or None
    @param known_ids:  file_id values already loaded in previous cycles.
                       Pass set(mutation_dataset["file_id"]) to deduplicate.
                       Deduplication is skipped when None.

    @rtype:  pd.DataFrame
    @return: DataFrame with columns ["is_orig", "file_id", "data"].
             All rows have is_orig == False.
    """
    df = load_queue_files(corpus_dir, known_ids=known_ids)
    return df[~df["is_orig"]].reset_index(drop=True)


def load_train_dataset(train_dataset_path):
    """
    Load the clean offline training corpus used for CovRL style mixing.

    Expected input:
      JSON file readable by pandas.read_json

    Returns columns:
      is_orig, file_id, data
    """
    if train_dataset_path is None:
        return pd.DataFrame([], columns=["is_orig", "file_id", "data"])

    dataset = pd.read_json(train_dataset_path)
    dataset = dataset.dropna(subset=["data"]).reset_index(drop=True)

    if "is_orig" not in dataset.columns:
        dataset["is_orig"] = True

    if "file_id" not in dataset.columns:
        dataset["file_id"] = -1

    return dataset[["is_orig", "file_id", "data"]]


def sample_train_dataset(train_dataset, n):
    """
    Sample up to n rows from the clean training corpus.
    """
    if train_dataset is None or train_dataset.empty or n <= 0:
        return pd.DataFrame([], columns=["is_orig", "file_id", "data"])

    n = min(n, len(train_dataset))
    return train_dataset.sample(n=n, ignore_index=True)