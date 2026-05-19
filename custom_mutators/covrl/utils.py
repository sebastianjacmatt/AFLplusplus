"""TF-IDF, corpus mixing, and fuzz-path utility functions.

Public:
    calc_reward(I_val, cov)  — design-doc reward function.
    read_coverage()          — AFL shmem bitmap reader (stub).
    validity(out)            — target-specific validity oracle (stub).
    mix_corpus(rollout, cfg) — D_T ∪ 4·|D_T| corpus samples, as HF Dataset.
"""

import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

from .config import Config
from .rollout import RolloutBuffer


# ── TF-IDF ───────────────────────────────────────────────────────────────────


class TFIDF:
    """Document-frequency tracker over integer terms (coverage edges or token IDs)."""

    def __init__(self):
        self._df: Counter = Counter()
        self._n_docs: int = 0

    @property
    def n_docs(self) -> int:
        return self._n_docs

    def update(self, terms: Iterable[int]) -> None:
        self._n_docs += 1
        for t in set(terms):
            self._df[t] += 1

    def score(self, terms: Iterable[int]) -> float:
        if self._n_docs == 0:
            return 0.0
        tf = Counter(terms)
        total = 0.0
        for term, count in tf.items():
            idf = math.log((self._n_docs + 1) / (self._df.get(term, 0) + 1))
            total += count * idf
        return total


# Coverage-based IDF: updated by every valid run; used by calc_reward.
_COV_TFIDF = TFIDF()

# Token-based IDF over the JS corpus: built once at first mix_corpus call, then
# frozen — corpus samples carry a score under this frozen IDF (design doc).
_CORPUS_TFIDF: Optional[TFIDF] = None
_CORPUS_TOKENS: Optional[List[List[int]]] = None
_MASKING = None  # cached Masking instance reused across mix_corpus calls


# ── Fuzz-path utilities ──────────────────────────────────────────────────────


def calc_reward(I_val: str, cov: bytes) -> float:
    """Per design doc:
        syntax_error    → -1.0
        semantic_error  → -0.5
        otherwise       → TF-IDF coverage reward squashed into (0, 1).

    The coverage IDF updates only on valid runs.
    """
    if I_val == "syntax_error":
        return -1.0
    if I_val == "semantic_error":
        return -0.5
    edges = _coverage_edges(cov)
    raw = _COV_TFIDF.score(edges)
    _COV_TFIDF.update(edges)
    return _squash(raw)


def read_coverage() -> bytes:
    """Read the AFL shared-memory bitmap.

    Per project memory: AFL env vars (__AFL_SHM_ID) must come from libc.getenv —
    os.environ is snapshotted at Py_Initialize and will not reflect AFL's runtime
    settings. Stub returns an empty bitmap until that wiring is in place.
    """
    return b""


def validity(out: bytes) -> str:
    """Target-specific validity classifier returning one of
    {'valid', 'syntax_error', 'semantic_error'}. Stub assumes valid until a real
    JS oracle (e.g. a node/qjs harness) is wired in.
    """
    return "valid"


def _coverage_edges(cov: bytes) -> List[int]:
    return [i for i, b in enumerate(cov) if b]


def _squash(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ── Corpus mixing (MixCorpus) ────────────────────────────────────────────────


def mix_corpus(rollout: RolloutBuffer, cfg: Config):
    """MixCorpus from the design doc:
        sample 4·|D_T| programs from C, score each with the frozen corpus IDF,
        union with D_T.

    The 4:1 corpus:rollout ratio matches CovRL-Fuzz. Corpus rows carry
    group_id = -1 so GRPO assigns them advantage 0 (per the policy-loss spec).
    Returns a HF Dataset of {input_ids, labels, R, group_id} rows.
    """
    from datasets import Dataset

    _ensure_corpus(cfg)
    masking = _MASKING

    rollout_rows = [
        {"input_ids": e.x, "labels": e.y, "R": e.R, "group_id": e.group_id}
        for e in rollout
    ]

    n_corpus = 4 * len(rollout)
    if n_corpus == 0:
        return Dataset.from_list(rollout_rows)

    n_corpus = min(n_corpus, len(_CORPUS_TOKENS))
    indices = random.sample(range(len(_CORPUS_TOKENS)), n_corpus)

    corpus_rows = []
    for idx in indices:
        tokens = _CORPUS_TOKENS[idx]
        mask = masking.sample_mask(tokens)
        corpus_rows.append({
            "input_ids": masking.encode_input(mask),
            "labels": _t5_label_sequence(masking, mask),
            "R": _squash(_CORPUS_TFIDF.score(tokens)),
            "group_id": -1,
        })

    return Dataset.from_list(rollout_rows + corpus_rows)


def _t5_label_sequence(masking, mask) -> List[int]:
    """T5 target format: <s_0> span_0 <s_1> span_1 … <s_N> [EOS]."""
    out: List[int] = []
    for i, (s, e) in enumerate(mask.spans):
        out.append(masking._sentinel_ids[i])
        out.extend(mask.tokens[s:e])
    out.append(masking._sentinel_ids[len(mask.spans)])
    if masking.tokenizer.eos_token_id is not None:
        out.append(masking.tokenizer.eos_token_id)
    return out


def _ensure_corpus(cfg: Config) -> None:
    """Tokenise the JS corpus once, build the frozen corpus IDF, cache both."""
    global _CORPUS_TFIDF, _CORPUS_TOKENS, _MASKING
    if _CORPUS_TOKENS is not None:
        return

    if not cfg.corpus_dir:
        raise RuntimeError("cfg.corpus_dir is empty; mix_corpus needs a corpus path")

    root = Path(cfg.corpus_dir)
    paths = sorted(p for p in root.rglob("*.js") if p.is_file())
    if not paths:
        raise RuntimeError(f"no .js files found under {root}")

    from .masking import Masking

    masking = Masking(cfg)
    tokens_list: List[List[int]] = []
    tfidf = TFIDF()

    for path in paths:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        tokens = masking.tokenize(data)
        if not tokens:
            continue
        tokens_list.append(tokens)
        tfidf.update(tokens)

    _CORPUS_TOKENS = tokens_list
    _CORPUS_TFIDF = tfidf
    _MASKING = masking