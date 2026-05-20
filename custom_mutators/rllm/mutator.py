# mutator.py
import os
import sys
import time

_SYNTAX_MARKERS   = ("SyntaxError",)
_SEMANTIC_MARKERS = ("ReferenceError", "TypeError", "RangeError", "URIError", "EvalError")


def _classify_stderr(text: str) -> str:
    """Map JerryScript stderr text to 'syntax' / 'semantic' / 'valid'."""
    if any(m in text for m in _SYNTAX_MARKERS):
        return "syntax"
    if any(m in text for m in _SEMANTIC_MARKERS):
        return "semantic"
    if not text.strip():
        return "valid"
    return "semantic"  # unknown non-empty stderr — conservative


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
        # AFL corpus finds
        self.queue_finds = 0
        # validity counters (from post_run via exit_hook.so stderr redirect)
        self.run_total = 0
        self.run_valid = 0
        self.run_syntax = 0
        self.run_semantic = 0
        # state
        self.out_dir: str | None = os.environ.get("AFL_CUSTOM_INFO_OUT")
        self.stderr_path: str | None = os.environ.get("RLM_STDERR_FILE")
        self.ui_off = os.environ.get("AFL_NO_UI") == "1"

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
        line = (
            f"[rllm] t={elapsed:7.1f}s "
            f"seeds={self.seeds:5d} "
            f"muts={self.mutations:6d} "
            f"finds={self.queue_finds:5d}({find_rate:5.1f}%) "
            f"valid={valid_rate:5.1f}% "
            f"syntax={syntax_rate:5.1f}% "
            f"semantic={sem_rate:5.1f}% "
            f"runs={self.run_total:6d} "
            f"gen_ms={self.generate_s*1000/seeds:7.1f} "
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
        if self.ui_off:
            print(line, file=sys.stderr, flush=True)


class Mutator:
    def __init__(self, cfg, masking, model, trainer=None):
        self.cfg = cfg
        self.masking = masking
        self.model = model
        self.trainer = trainer  # may be None for pure-mutation
        self._pending_outputs: list[bytearray] = []
        self._finetune_pending = False
        self._queue_get_count = 0
        self._stats = _Stats()
        # Set in fuzz(), consumed in post_run(): lets us distinguish runs that
        # executed a fresh LLM mutation from runs AFL initiated on its own
        # (calibration, trim, sync). Without this the validity stat is diluted
        # by ~56 calibration replays per saved queue entry.
        self._last_run_was_mutation = False
        self._last_run_class: str = "valid"
        # Filenames of queue entries whose original execution produced a
        # syntax/semantic error. AFL doesn't expose a veto hook on queue
        # insertion (add_to_queue always commits before queue_new_entry
        # fires), so we let invalid entries sit in the queue and skip them
        # at queue_get time. Mutating already-broken JS is near-deterministically
        # wasted compute — the LLM's local span infill cannot repair global
        # syntax breakage. See docs/queue_validity_collapse.md.
        self._invalid_filenames: set = set()

    def queue_get(self, filename):
        # Belt-and-suspenders: clear the mutation flag at the start of every
        # queue cycle so any calibration of this entry (which runs between
        # queue_get and fuzz_count) never inherits a stale True from a prior
        # fuzz() whose post_run was skipped (e.g., post_process returned 0).
        self._last_run_was_mutation = False
        self._queue_get_count += 1
        if self._queue_get_count % self.cfg.finetune_every == 0:
            self._finetune_pending = True
        return filename not in self._invalid_filenames

    def fuzz_count(self, buf):
        self._maybe_finetune()
        self._stats.seeds += 1

        t0 = time.monotonic()
        tokens = self.model.tokenizer.tokenize(buf)
        self._stats.tokenize_s += time.monotonic() - t0

        # TODO(alignment): clamp source length to CodeT5's pretraining ceiling
        # of 512 tokens (CodeT5 §4.5: "maximum source and target sequence
        # lengths to be 512 and 256"). Over-length seeds either skip (return 0)
        # or truncate `tokens` here. Add `max_source_length: int = 512` to
        # MutatorConfig when wiring this in.
        if not tokens:
            self._stats.empty_tokenize += 1
            self._pending_outputs = []
            self._stats.maybe_flush()
            return 0

        t0 = time.monotonic()
        masks = [self.masking.mask(tokens) for _ in range(self.cfg.fuzz_count)]
        self._stats.mask_s += time.monotonic() - t0

        # TODO(alignment): dynamic per-call max_new_tokens via
        # `max(self.masking.generation_budget(mp, max_per_span=20) for mp in masks)`,
        # passed to batch_generate's `max_new_tokens` override. Keeps per-span
        # budget constant regardless of how many spans the masker sampled and
        # avoids truncating trailing spans on multi-span seeds. Mirrors
        # rlm_mutator's `masked_span_prediction_batch` budget calc.
        t0 = time.monotonic()
        outputs = self.model.batch_generate(
            [mp.input_ids for mp in masks], n_samples=1,
        )
        self._stats.generate_s += time.monotonic() - t0

        t0 = time.monotonic()
        self._pending_outputs = [
            bytearray(self.model.tokenizer.reconstruct(mp, y))
            for mp, y in zip(masks, outputs)
        ]
        self._stats.reconstruct_s += time.monotonic() - t0

        self._stats.mutations += len(self._pending_outputs)
        self._stats.maybe_flush()
        return len(self._pending_outputs)

    def fuzz(self, buf, add_buf, max_size):
        self._last_run_was_mutation = True
        return self._pending_outputs.pop(0)

    def post_process(self, buf) -> bytes:
        """Sanitize inputs to valid UTF-8 before the target sees them."""
        raw = bytes(buf)
        try:
            raw.decode("utf-8", errors="strict")
            return raw
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace").encode("utf-8")

    def post_run(self):
        path = self._stats.stderr_path
        # Always drain the stderr file so calibration/trim output from the
        # previous run doesn't bleed into the next mutation's classification.
        text = ""
        if path:
            try:
                with open(path, "rb") as fh:
                    data = fh.read(4096)
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
            return
        self._last_run_was_mutation = False

        cls = _classify_stderr(text)
        self._last_run_class = cls
        self._stats.run_total += 1
        if cls == "valid":
            self._stats.run_valid += 1
        elif cls == "syntax":
            self._stats.run_syntax += 1
        else:
            self._stats.run_semantic += 1
        self._stats.maybe_flush()

    def queue_new_entry(self, new, orig):
        self._stats.queue_finds += 1
        self._stats.maybe_flush()
        # Record invalid finds so queue_get can skip them later. The hook's
        # return value is NOT a veto on queue insertion (the entry is already
        # committed by add_to_queue before this fires); it only signals to
        # AFL whether we modified the file on disk, which we did not.
        if self._last_run_class != "valid":
            self._invalid_filenames.add(new)
        return False

    def deinit(self):
        self._stats.maybe_flush(force=True)
        self.model.save_checkpoint()

    def _maybe_finetune(self):
        if not self._finetune_pending:
            return
        self._finetune_pending = False
        if self.trainer is not None:
            self.trainer.finetune()
