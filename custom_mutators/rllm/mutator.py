# mutator.py
import difflib
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
        # Append-only time-series log; rllm_stats.txt keeps only the latest
        # snapshot for tail-f monitoring, this file keeps the full history
        # for post-hoc evaluation (eval/plot.py).
        self._history_path = (
            os.path.join(self.out_dir, "rllm_history.tsv") if self.out_dir else None
        )
        self._history_header_written = False

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
        if self._history_path:
            try:
                new_file = not self._history_header_written
                with open(self._history_path, "a") as f:
                    if new_file:
                        f.write(
                            "epoch_s\telapsed_s\tseeds\tmuts\tfinds\t"
                            "run_total\trun_valid\trun_syntax\trun_semantic\t"
                            "valid_pct\tsyntax_pct\tsemantic_pct\t"
                            "gen_ms_avg\tmuts_per_s\n"
                        )
                        self._history_header_written = True
                    f.write(
                        f"{time.time():.3f}\t{elapsed:.3f}\t"
                        f"{self.seeds}\t{self.mutations}\t{self.queue_finds}\t"
                        f"{self.run_total}\t{self.run_valid}\t"
                        f"{self.run_syntax}\t{self.run_semantic}\t"
                        f"{valid_rate:.3f}\t{syntax_rate:.3f}\t{sem_rate:.3f}\t"
                        f"{self.generate_s*1000/seeds:.3f}\t"
                        f"{self.mutations/elapsed:.3f}\n"
                    )
            except OSError:
                pass
        if self.ui_off:
            print(line, file=sys.stderr, flush=True)


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
        self._pending_outputs: list[bytes] = []
        self._finetune_pending = False
        self._queue_get_count = 0
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

    def queue_get(self, filename):
        # Belt-and-suspenders: clear the mutation flag at the start of every
        # queue cycle so any calibration of this entry (which runs between
        # queue_get and fuzz_count) never inherits a stale True from a prior
        # fuzz() whose post_run was skipped (e.g., post_process returned 0).
        self._last_run_was_mutation = False
        self._queue_get_count += 1
        if self._queue_get_count % self.cfg.finetune_every == 0:
            self._finetune_pending = True
        # Append seed-selection event for eval/plot.py to correlate metric
        # drops with the specific seed being fuzzed at that moment.
        if self._stats.out_dir:
            try:
                path = os.path.join(self._stats.out_dir, "rllm_seeds.tsv")
                new_file = not os.path.exists(path)
                with open(path, "a") as f:
                    if new_file:
                        f.write("epoch_s\telapsed_s\tfilename\n")
                    elapsed = time.monotonic() - self._stats.start
                    name = filename if isinstance(filename, str) else \
                        filename.decode("utf-8", "replace")
                    base = os.path.basename(name)
                    f.write(f"{time.time():.3f}\t{elapsed:.3f}\t{base}\n")
            except OSError:
                pass
        return True

    def fuzz_count(self, buf):
        self._maybe_finetune()
        self._stats.seeds += 1

        t0 = time.monotonic()
        tokens = self.model.tokenizer.parse_u16(buf)
        self._stats.tokenize_s += time.monotonic() - t0
        # Cache for the .cur_input.diff sidecar in post_process.
        self._parent_tokens = tokens

        if not tokens:
            self._stats.empty_tokenize += 1
            self._pending_outputs = []
            self._stats.maybe_flush()
            return 0

        t0 = time.monotonic()
        masks = [self.masking.mask(tokens) for _ in range(self.cfg.fuzz_count)]
        self._stats.mask_s += time.monotonic() - t0

        t0 = time.monotonic()
        outputs = self.model.batch_generate(
            [mp.input_ids for mp in masks], n_samples=1,
        )
        self._stats.generate_s += time.monotonic() - t0

        t0 = time.monotonic()
        self._pending_outputs = [
            self.model.tokenizer.encode_u16(
                self.model.tokenizer.reconstruct_tokens(mp, y)
            )
            for mp, y in zip(masks, outputs)
        ]
        self._stats.reconstruct_s += time.monotonic() - t0

        self._stats.mutations += len(self._pending_outputs)
        self._stats.maybe_flush()
        return len(self._pending_outputs)

    def fuzz(self, buf, add_buf, max_size):
        # MUST return bytearray, not bytes. AFL++'s Python binding takes the
        # bytes path through py_bytes() in afl-fuzz-python.c which then crashes
        # inside memcpy() with a corrupted source pointer (verified via gdb at
        # afl-fuzz-python.c:138). The bytearray path works correctly. The
        # official example mutator (custom_mutators/examples/example.py) also
        # uses bytearray; the FATAL message says "bytearray or bytes" but bytes
        # is effectively broken on the current AFL++ tree.
        self._last_run_was_mutation = True
        if not self._pending_outputs:
            return bytearray()
        return bytearray(self._pending_outputs.pop(0))

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
        if self._last_run_was_mutation and self._parent_tokens \
                and self._stats.out_dir:
            try:
                self._write_diff_sidecar(self._parent_tokens, tokens)
            except OSError:
                pass
        return source

    def _write_diff_sidecar(self, parent_tokens, mut_tokens) -> None:
        """Annotate the decoded mutation source with ANSI-colored markers
        wrapping any region whose tokens differ from the parent queue
        entry. Written to ``<out>/.cur_input.diff`` next to AFL's
        ``.cur_input`` so a `watch -n 2 -c cat .cur_input.diff` shows the
        live mutation with changes highlighted.

        Uses ``difflib.SequenceMatcher`` on the token sequences (not the
        decoded bytes) so the highlight respects token boundaries; per-
        segment ``detokenize`` calls handle the BPE → bytes conversion.
        Cost is ~1–3 ms per mutation; cheap relative to the LLM call.
        """
        tok = self.model.tokenizer
        matcher = difflib.SequenceMatcher(None, parent_tokens, mut_tokens)
        parts: list[bytes] = []
        for op, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                parts.append(tok.detokenize(mut_tokens[j1:j2]))
            elif op == "delete":
                # Parent had tokens here that the mutation removed; the
                # mutation buffer has no replacement bytes to color. Mark
                # the position with a dim cross so the gap is visible.
                parts.append(b"\x1b[2;31m[--]\x1b[0m")
            else:  # replace or insert
                parts.append(b"\x1b[1;33m")
                parts.append(tok.detokenize(mut_tokens[j1:j2]))
                parts.append(b"\x1b[0m")
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
