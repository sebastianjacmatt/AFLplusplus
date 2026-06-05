# mutator.py
import difflib
import os
import sys
import time
from collections import deque

from data.rewarding import TFIDFCoverageRewarder
from data.rollout import RolloutBuffer
from data.validity import classify_stderr
from logging_tsv import TsvLogger
from training.grpo import _trim_target


class _Stats:
    """In-process mutator counters with rate-limited surfacing.

    Writes a one-line snapshot to ``<out>/rllm_stats.txt`` every flush and,
    when AFL's UI is off (``AFL_NO_UI=1``), also prints to stderr so the line
    shows up in ``debug.log`` under ``run_rllm.sh -d``. In UI mode stderr is
    suppressed — AFL repaints would garble it — so the file is the source of
    truth (``tail -f <out>/rllm_stats.txt`` from another terminal).
    """

    def __init__(self, flush_every_s: float = 2.0):
        self.flush_every_s = flush_every_s
        self.start = time.monotonic()
        self.last_flush = 0.0
        # generation counters
        self.seeds = 0
        self.mutations = 0
        self.empty_tokenize = 0
        # timing
        self.tokenize_s = 0.0
        self.mask_s = 0.0
        self.generate_s = 0.0
        self.reconstruct_s = 0.0
        self.finetune_s = 0.0
        # AFL corpus finds
        self.queue_finds = 0
        # validity counters (from post_run via exit_hook.so stderr redirect)
        self.run_total = 0
        self.run_valid = 0
        self.run_syntax = 0
        self.run_semantic = 0
        self.run_timeout = 0   # hangs AFL killed (exit_hook SIGUSR1 marker)
        # coverage reward (mean R_cov over scored mutations; 0 if not wired)
        self.cov_sum = 0.0
        self.cov_n = 0
        # state
        self.out_dir: str | None = os.environ.get("AFL_CUSTOM_INFO_OUT")
        self.stderr_path: str | None = os.environ.get("RLM_STDERR_FILE")
        self.ui_off = os.environ.get("AFL_NO_UI") == "1"
        # Append-only time-series log; rllm_stats.txt keeps only the latest
        # snapshot for tail-f monitoring, this file keeps the full history
        # for post-hoc evaluation (eval/plot.py). Held-open via TsvLogger.
        self._hist_log = TsvLogger(self.out_dir, "rllm_history.tsv", self.start,
                                   float_fmt=".3f")

    def maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_flush < self.flush_every_s:
            return
        self.last_flush = now
        elapsed = max(now - self.start, 1e-6)
        seeds = max(self.seeds, 1)
        muts = max(self.mutations, 1)
        rt = max(self.run_total, 1)
        find_rate   = self.queue_finds / muts * 100
        valid_rate  = self.run_valid   / rt   * 100
        syntax_rate = self.run_syntax  / rt   * 100
        sem_rate    = self.run_semantic / rt   * 100
        tmout_rate  = self.run_timeout / rt   * 100
        line = (
            f"[rllm] t={elapsed:7.1f}s "
            f"seeds={self.seeds:5d} "
            f"muts={self.mutations:6d} "
            f"finds={self.queue_finds:5d}({find_rate:5.1f}%) "
            f"valid={valid_rate:5.1f}% "
            f"syntax={syntax_rate:5.1f}% "
            f"semantic={sem_rate:5.1f}% "
            f"tmout={tmout_rate:5.1f}% "
            f"runs={self.run_total:6d} "
            f"gen_ms={self.generate_s*1000/seeds:7.1f} "
            f"mask_ms={self.mask_s*1000/seeds:6.1f} "
            f"ft_ms={self.finetune_s*1000/seeds:6.1f} "
            f"rcov={self.cov_sum/max(self.cov_n,1):5.3f} "
            f"muts/s={self.mutations/elapsed:6.1f}"
        )
        if self.out_dir:
            try:
                # Truncate-and-write keeps the same inode so `tail -f` (which
                # follows the fd, not the path) keeps tracking across flushes.
                path = os.path.join(self.out_dir, "rllm_stats.txt")
                with open(path, "w") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        # Same columns/order as before (consumed by eval/plot.py + eval/runner.py);
        # the logger injects the epoch_s/elapsed_s prefix and formats by type.
        self._hist_log.write({
            "seeds": self.seeds,
            "muts": self.mutations,
            "finds": self.queue_finds,
            "run_total": self.run_total,
            "run_valid": self.run_valid,
            "run_syntax": self.run_syntax,
            "run_semantic": self.run_semantic,
            "run_timeout": self.run_timeout,
            "valid_pct": valid_rate,
            "syntax_pct": syntax_rate,
            "semantic_pct": sem_rate,
            "timeout_pct": tmout_rate,
            "gen_ms_avg": self.generate_s * 1000 / seeds,
            "muts_per_s": self.mutations / elapsed,
        })
        if self.ui_off:
            print(line, file=sys.stderr, flush=True)

    def close(self) -> None:
        self._hist_log.close()


class Mutator:
    """Token-level mutator over a u16 binary queue (CovRL/TLAFL-style).

    Queue files are sequences of little-endian uint16 token IDs.
    ``AFL_POST_PROCESS_KEEP_ORIGINAL=1`` keeps the mutator's `fuzz()` output
    intact in the queue; ``post_process`` decodes tokens → JS source bytes
    only at execution time. Bytes from the target's perspective never re-enter
    the queue, so the queue-as-tokens invariant is structural rather than
    contractual.
    """

    def __init__(self, cfg, masking, model, trainer=None):
        self.cfg = cfg
        self.masking = masking
        self.model = model
        self.trainer = trainer  # may be None for pure-mutation
        # Coverage reward (CovRL R_cov over the AFL++ bitmap). Lazy-attaches to
        # __AFL_SHM_ID on first score; _score_coverage degrades to validity-only
        # if the attach ever fails. Only needed when a trainer consumes rewards.
        self._coverage = (
            TFIDFCoverageRewarder(cfg.bitmap_size, cfg.idf_alpha, delta=cfg.delta_coverage)
            if cfg.coverage_reward and trainer is not None
            else None
        )
        # GRPO rollout capture (only when a trainer is attached): the M×G masks
        # come from mask_batch; infills + rewards are attached as they execute
        # (post_run); _maybe_finetune trains on the buffer at the finetune_every
        # boundary, grouped by seed → mask. See data/rollout.py.
        self._rollouts = RolloutBuffer() if trainer is not None else None
        self._pending_masks:   deque = deque()
        self._pending_outputs: deque = deque()
        self._finetune_pending = False
        self._stats = _Stats()
        # Parent's u16 tokens cached in fuzz_count; consumed by post_process
        # to write a `.cur_input.diff` sidecar highlighting what this
        # mutation changed vs the parent. Set to None when no mutation is
        # active so calibration / trim post_process calls don't write stale
        # diffs.
        self._parent_tokens: list[int] | None = None
        # Set in fuzz(), consumed in post_run(): lets us distinguish runs that
        # executed a fresh LLM mutation from runs AFL initiated on its own
        # (calibration, trim, sync). Without this the validity stat is diluted
        # by ~56 calibration replays per saved queue entry.
        self._last_run_was_mutation = False
        self._last_run_class: str = "valid"
        # Basename of the seed currently being fuzzed (set in queue_get, read in
        # fuzz_count) — the lookup key for its cached delta-vs-parent edge set.
        self._cur_seed_name: str = ""
        # Cached in post_process(), consumed in post_run() to write the diff
        # after validity is known. None when no mutation is pending.
        # Tuple: (parent_tokens, mut_tokens, masked_positions)
        # masked_positions = span.start indices so identity mutations can be
        # highlighted in dim-green even when the output equals the input.
        self._last_diff_tokens: tuple | None = None
        self._last_masked_positions: list[int] = []
        # Runtime timelines (all held-open via TsvLogger, sharing _stats.start so
        # epoch_s/elapsed_s align across files). rllm_history.tsv is owned by
        # _Stats; the rest are owned here.
        st = self._stats
        self._seeds_log = TsvLogger(st.out_dir, "rllm_seeds.tsv", st.start, float_fmt=".3f")
        self._samples_log = TsvLogger(st.out_dir, "rllm_samples.txt", st.start)
        self._train_log = TsvLogger(st.out_dir, "rllm_train.tsv", st.start)
        # Finetune-cycle id tagged onto each sample so rllm_samples.txt rows join to
        # rllm_train.tsv by cycle; incremented when a cycle is scheduled (fuzz_count).
        self._rollout_bucket = 0

    def queue_get(self, filename):
        # Belt-and-suspenders: clear the mutation flag at the start of every
        # queue cycle so any calibration of this entry (which runs between
        # queue_get and fuzz_count) never inherits a stale True from a prior
        # fuzz() whose post_run was skipped (e.g., post_process returned 0).
        self._last_run_was_mutation = False
        # Append seed-selection event for eval/plot.py to correlate metric
        # drops with the specific seed being fuzzed at that moment.
        name = filename if isinstance(filename, str) else \
            filename.decode("utf-8", "replace")
        self._cur_seed_name = os.path.basename(name)   # delta: parent-cache lookup key (fuzz_count)
        self._seeds_log.write({"filename": self._cur_seed_name})
        return True

    def fuzz_count(self, buf):
        self._maybe_finetune()
        self._stats.seeds += 1
        # Finetune cadence: every `finetune_every` SEEDS fuzzed (CovRL SYNC_INTERVAL=100),
        # NOT per queue_get — AFL calls queue_get ~1.2x/seed for entries it then skips or
        # calibrates, which fired the finetune early (~82 seeds). Flag here; the actual
        # train runs at the next fuzz_count's _maybe_finetune so this seed's rollout finishes.
        if self.cfg.finetune_every > 0 and self._stats.seeds % self.cfg.finetune_every == 0:
            self._finetune_pending = True

        # B1 delta-vs-parent: select this seed's cached parent edge set (captured when
        # the seed was created as a queue find — see queue_new_entry). Reusing the cache
        # is what makes delta correct in the mutations-on-mutations regime, where AFL does
        # not re-run the parent so the live SHM holds the previous seed's trace instead.
        if self._coverage is not None:
            self._coverage.set_parent(self._cur_seed_name)

        t0 = time.monotonic()
        tokens = self.model.tokenizer.parse_u16(buf)
        if len(tokens) > self.cfg.max_seq_len:
            tokens = tokens[: self.cfg.max_seq_len]   # bound O(seq²) attention / VRAM (CovRL: 768)
        self._stats.tokenize_s += time.monotonic() - t0
        # Cache for the .cur_input.diff sidecar in post_process.
        self._parent_tokens = tokens

        if not tokens:
            self._stats.empty_tokenize += 1
            self._pending_masks.clear()
            self._pending_outputs.clear()
            self._stats.maybe_flush()
            return 0

        t0 = time.monotonic()
        self._pending_masks = deque(
            self.masking.mask_batch(tokens, self.cfg.fuzz_count)
        )
        self._stats.mask_s += time.monotonic() - t0
        self._pending_outputs.clear()
        if self._rollouts is not None:
            self._rollouts.begin_seed(tokens)

        self._stats.maybe_flush()
        return self.cfg.fuzz_count

    def fuzz(self, buf, add_buf, max_size):
        # MUST return bytearray, not bytes. AFL++'s Python binding takes the
        # bytes path through py_bytes() in afl-fuzz-python.c which then crashes
        # inside memcpy() with a corrupted source pointer (verified via gdb at
        # afl-fuzz-python.c:138). The bytearray path works correctly. The
        # official example mutator (custom_mutators/examples/example.py) also
        # uses bytearray; the FATAL message says "bytearray or bytes" but bytes
        # is effectively broken on the current AFL++ tree.
        self._last_run_was_mutation = True
        if not self._pending_masks and not self._pending_outputs:
            return bytearray()

        if not self._pending_outputs:
            n = min(self.cfg.inference_batch_size, len(self._pending_masks))
            batch = [self._pending_masks.popleft() for _ in range(n)]

            t0 = time.monotonic()
            outputs = self.model.batch_generate(
                [mp.input_ids for mp in batch], n_samples=1,
            )
            self._stats.generate_s += time.monotonic() - t0

            t0 = time.monotonic()
            collect = self.trainer is not None
            tok = self.model.tokenizer
            for mp, y in zip(batch, outputs):
                u16 = tok.encode_u16(tok.reconstruct_tokens(mp, y))
                positions = [span.start for span in mp.spans]
                if collect:
                    target = _trim_target(y, tok.pad_token_id, tok.eos_token_id)
                    self._pending_outputs.append((u16, positions, mp.input_ids, target))
                else:
                    self._pending_outputs.append((u16, positions, None, None))
            self._stats.reconstruct_s += time.monotonic() - t0
            self._stats.mutations += n

        out_bytes, self._last_masked_positions, enc, target = self._pending_outputs.popleft()
        if self._rollouts is not None:
            self._rollouts.set_pending(enc, target, self._last_masked_positions)
        return bytearray(out_bytes)

    def post_process(self, buf) -> bytes:
        """Decode the u16 token buffer to JS source bytes for the target.

        Runs once per execution; the decoded bytes go to ``.cur_input`` and
        never enter the queue (``AFL_POST_PROCESS_KEEP_ORIGINAL=1`` keeps
        the queue holding the original u16). U+FFFD bytes from partial
        multi-byte BPE tokens are stripped in ``detokenize`` for cleanliness
        but are not load-bearing — no feedback loop forms because re-reads
        of the queue see u16 tokens, not these bytes.
        """
        tokens = self.model.tokenizer.parse_u16(buf)
        source = self.model.tokenizer.detokenize(tokens)
        if self.cfg.write_diff_sidecar and self._last_run_was_mutation and self._parent_tokens:
            self._last_diff_tokens = (self._parent_tokens, tokens, self._last_masked_positions)
        return source

    def _write_diff_sidecar(self, parent_tokens, mut_tokens, masked_positions, cls: str) -> None:
        """Annotate the decoded mutation source with ANSI-colored markers.

        Four colors based on outcome and whether the prediction changed anything:
          bright green  \x1b[1;32m — infill changed the token, result was valid
          dim green     \x1b[2;32m — infill reproduced the original token (identity)
          bright red    \x1b[1;31m — infill changed the token, syntax error
          bright yellow \x1b[1;33m — infill changed the token, semantic/runtime error

        Written to ``<out>/.cur_input.diff`` so ``watch -n 2 -c cat
        .cur_input.diff`` shows the live mutation highlighted.
        """
        RESET = b"\x1b[0m"
        if cls == "valid":
            mut_color = b"\x1b[1;32m"
        elif cls == "syntax":
            mut_color = b"\x1b[1;31m"
        else:
            mut_color = b"\x1b[1;33m"
        same_color = b"\x1b[2;32m"  # dim green — identity prediction

        tok = self.model.tokenizer
        matcher = difflib.SequenceMatcher(None, parent_tokens, mut_tokens)
        parts: list[bytes] = []
        has_change = False

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                # Within equal regions, highlight any token whose position was
                # masked but predicted identically — dim green.
                seg = parent_tokens[i1:i2]
                masked_in_seg = {p - i1 for p in masked_positions if i1 <= p < i2}
                if masked_in_seg:
                    for k, token in enumerate(seg):
                        if k in masked_in_seg:
                            parts.append(same_color)
                            parts.append(tok.detokenize([token]))
                            parts.append(RESET)
                        else:
                            parts.append(tok.detokenize([token]))
                    has_change = True
                else:
                    parts.append(tok.detokenize(seg))
            elif op == "delete":
                parts.append(mut_color + b"[--]" + RESET)
                has_change = True
            else:  # replace or insert
                parts.append(mut_color)
                parts.append(tok.detokenize(mut_tokens[j1:j2]))
                parts.append(RESET)
                has_change = True

        if not has_change:
            return
        path = os.path.join(self._stats.out_dir, ".cur_input.diff")
        with open(path, "wb") as f:
            f.write(b"".join(parts))

    def post_run(self):
        path = self._stats.stderr_path
        # Always drain the stderr file so calibration/trim output from the
        # previous run doesn't bleed into the next mutation's classification.
        text = ""
        if path:
            try:
                with open(path, "rb") as fh:
                    data = fh.read(8192)            # head: jerry's error markers
                    if fh.seek(0, 2) > 8192:        # large output (e.g. a spammy hang):
                        fh.seek(-256, 2)            # also grab the tail for the EOF
                        data += fh.read(256)        # timeout marker exit_hook appended
                text = data.decode("utf-8", errors="replace")
            except OSError:
                pass
            try:
                os.truncate(path, 0)
            except OSError:
                pass

        # Only attribute validity to runs that executed a fresh LLM mutation.
        # AFL fires post_run for calibration/trim/sync replays too; those drown
        # out mutation outcomes (~56 replays per saved queue entry) and skew
        # the rate toward the queue's validity instead of the model's.
        if not self._last_run_was_mutation:
            self._last_diff_tokens = None
            return
        self._last_run_was_mutation = False

        cls = classify_stderr(text)
        self._last_run_class = cls

        # Attach this mutation's reward to its mask-group in the rollout buffer.
        r_cov = reward = None
        if self._rollouts is not None and self._rollouts.has_pending():
            if cls != "timeout":
                r_cov = self._score_coverage()   # reads this run's bitmap, folds DF
            reward = self._reward(cls, r_cov)    # timeout ignores r_cov → timeout_reward
            self._rollouts.commit(reward)
        self._log_sample(cls, r_cov, reward)   # per-infill row → rllm_samples.txt

        if self._last_diff_tokens is not None and self._stats.out_dir:
            try:
                self._write_diff_sidecar(*self._last_diff_tokens, cls)
            except OSError:
                pass
            self._last_diff_tokens = None

        self._stats.run_total += 1
        if cls == "valid":
            self._stats.run_valid += 1
        elif cls == "syntax":
            self._stats.run_syntax += 1
        elif cls == "timeout":
            self._stats.run_timeout += 1
        else:
            self._stats.run_semantic += 1
        self._stats.maybe_flush()

    def queue_new_entry(self, new, orig):
        self._stats.queue_finds += 1
        # B1 delta-vs-parent: AFL just saved this find as queue entry `new`; the SHM read
        # in its finds-producing mutation's post_run (rewarder._last_idx) is exactly its
        # edge set. Cache it by basename so set_parent uses it when `new` is later fuzzed.
        if self._coverage is not None:
            nm = new if isinstance(new, str) else new.decode("utf-8", "replace")
            self._coverage.cache_parent(os.path.basename(nm))
        self._stats.maybe_flush()
        return False

    def deinit(self):
        self._stats.maybe_flush(force=True)
        self._stats.close()
        self._seeds_log.close()
        self._samples_log.close()
        self._train_log.close()
        self.model.save_checkpoint()

    def _log_sample(self, cls: str, r_cov, reward) -> None:
        """One executed mutation → ``<out>/rllm_samples.txt``: validity class (from
        the exit-hook stderr), coverage ``R_cov`` (the just-run bitmap's TF-IDF
        score), the total scalar reward, and the ``(cycle, seed, positions)`` that
        identify its GRPO mask-group (joins to rllm_train.tsv by ``cycle``). All
        values are already computed for the reward, so this is one held-open write
        per mutation. ``r_cov``/``reward`` are blank when no trainer is attached.
        Disable with ``log_samples=false`` on pure-speed runs."""
        if not self.cfg.log_samples:
            return
        self._samples_log.write({
            "cycle": self._rollout_bucket,
            "seed": self._stats.seeds,
            "positions": "-".join(map(str, self._last_masked_positions)),
            "class": cls,
            "r_cov": "" if r_cov is None else float(r_cov),
            "reward": "" if reward is None else float(reward),
        })

    def _score_coverage(self):
        """``R_cov`` for the just-executed bitmap, or ``None`` when coverage
        reward is off / the SHM attach failed. Folds the bitmap into DF as a
        side effect (so invalid runs still inform IDF). On the first attach
        failure it disables coverage for the rest of the run and falls back to
        the validity-only reward, rather than killing the fuzzer."""
        if self._coverage is None:
            return None
        try:
            r = self._coverage.score()
        except Exception as e:   # attach/read failure → degrade, don't crash
            print(f"[rllm] coverage reward disabled (SHM attach failed): {e}",
                  file=sys.stderr, flush=True)
            self._coverage = None
            return None
        self._stats.cov_sum += r
        self._stats.cov_n += 1
        return r

    def _reward(self, cls: str, r_cov: float | None = None) -> float:
        """CovRL reward partition (Eq. 2): −1 syntax, −0.5 semantic, and
        ``b + (1−b)·R_cov`` for valid. ``r_cov is None`` ⇒ validity-only
        fallback (+1) when the coverage bitmap is unavailable."""
        if cls == "syntax":
            return -1.0
        if cls == "semantic":
            return -0.5
        if cls == "timeout":
            return self.cfg.timeout_reward   # hang: a non-terminating no-op (default -1)
        if r_cov is None:
            return 1.0   # coverage reward unavailable → validity-only
        b = self.cfg.validity_bonus
        return round(b + (1.0 - b) * r_cov, 4)

    def _maybe_finetune(self):
        if not self._finetune_pending:
            return
        self._finetune_pending = False
        if self._coverage is not None:
            self._coverage.update_cycle()   # refresh lagged IDF at the cycle boundary
        if self._rollouts is None:
            return
        t0 = time.monotonic()
        info = self.trainer.finetune(self._rollouts)   # builds dataset + trains
        if info:
            # Tag with the bucket id whose infills this cycle trained on, so
            # rllm_train.tsv joins to rllm_samples.txt by `cycle`.
            self._log_finetune({"cycle": self._rollout_bucket, **info})
        self._rollouts.clear()
        self._rollout_bucket += 1            # next rollout fills the next cycle's bucket
        self._stats.finetune_s += time.monotonic() - t0

    def _log_finetune(self, info: dict) -> None:
        """One per-cycle GRPO row → ``<out>/rllm_train.tsv``. Field-agnostic: the
        trainer's return-dict keys (loss, group health, policy diagnostics) become
        the columns, so a new GRPO variant's metric needs no plumbing here."""
        if info:
            self._train_log.write(info)
