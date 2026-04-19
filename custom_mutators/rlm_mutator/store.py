"""In-memory RL rollout store for rlm_mutator.

Records every fuzz() output with a two-phase write pattern:

  fuzz()      → log(sample_id, group_id, x_t, y_t, log_prob)   reward=None
  post_run()  → patch_reward(sample_id, reward)                 fills reward

close_group() returns all records with a populated reward field and clears
the in-memory store.  Records missing a reward (edge case: post_run() was
never called for a sample) are silently dropped.

Schema (one dict per sample):

    {
        "sample_id":    str,    # zero-padded "s00000000"
        "group_id":     int,    # group index within the current seed
        "x_t":          list,   # masked encoder input token IDs
        "y_t":          list,   # generated decoder token IDs
        "log_prob":     float,  # mean per-token log-prob under current actor
        "reward":       float,  # TF-IDF coverage reward from post_run()
        "ref_log_prob": float | None,  # reference model log-prob (PPO KL)
        "value_pred":   float | None,  # critic value estimate (PPO only)
    }
"""


class RLStore:
    """Single-threaded in-memory rollout store.

    AFL++ drives the Python mutator from one thread — no locking needed.
    """

    def __init__(self):
        self._records: dict[str, dict] = {}
        self._counter: int = 0

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def new_sample_id(self) -> str:
        sid = f"s{self._counter:08d}"
        self._counter += 1
        return sid

    def log(
        self,
        sample_id: str,
        group_id: int,
        x_t: list,
        y_t: list,
        log_prob: float,
        ref_log_prob: float | None = None,
        value_pred: float | None = None,
    ) -> None:
        """Record a new mutant immediately after generation.

        reward is left as None until post_run() fills it.

        @param sample_id:    Unique ID for this mutant (from new_sample_id()).
        @param group_id:     Group index within the current seed.
        @param x_t:          Masked encoder input token IDs.
        @param y_t:          Decoder output token IDs produced by actor.
        @param log_prob:     Mean per-token log-prob of y_t under current actor.
        @param ref_log_prob: Log-prob under reference model (optional, for PPO KL).
        @param value_pred:   Critic value estimate (optional, PPO only).
        """
        self._records[sample_id] = {
            "sample_id":    sample_id,
            "group_id":     group_id,
            "x_t":          x_t,
            "y_t":          y_t,
            "log_prob":     log_prob,
            "reward":       None,
            "ref_log_prob": ref_log_prob,
            "value_pred":   value_pred,
        }

    def patch_reward(self, sample_id: str, reward: float) -> None:
        """Fill in the reward computed by post_run().

        @param sample_id: Must match a previously logged sample.
        @param reward:    TF-IDF coverage reward scalar.
        """
        rec = self._records.get(sample_id)
        if rec is not None:
            rec["reward"] = reward

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def close_group(self) -> list[dict]:
        """Return all complete records and clear the store.

        Complete means reward is not None.  Incomplete records (post_run()
        missed — should not happen in normal operation) are dropped silently.

        @return: List of record dicts, one per mutant.
        """
        complete = [r for r in self._records.values() if r["reward"] is not None]
        self._records.clear()
        return complete

    def __len__(self) -> int:
        return len(self._records)
