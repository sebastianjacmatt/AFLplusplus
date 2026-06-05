"""One reusable append-writer for the rllm runtime timelines.

Every per-time log the mutator emits (rollout history, seed selections, per-infill
samples, per-finetune training metrics) has the same shape: an ``epoch_s`` +
``elapsed_s`` prefix followed by a flat dict of fields. ``TsvLogger`` is that shape,
once:

* **Held open** — the handle is opened lazily on the first ``write`` and reused, so
  there is no per-row ``open``/``close`` syscall (cheaper than the old writers).
* **Field-agnostic** — the header is the dict keys, written once. A new metric for a
  new GRPO variant is just a new key in the dict the caller passes; no header or
  plumbing change. This is what lets ``rllm_train.tsv`` grow columns for free.
* **Durable** — ``flush_each`` (default on) flushes every row, matching the old
  open/close durability so a killed run keeps its last rows. The cost is one write
  syscall per row (~µs), negligible even on the per-mutation path.
* **Silent without an out dir** — ``out_dir is None`` ⇒ no-op, preserving the
  "no AFL_CUSTOM_INFO_OUT means no files" behaviour.

Non-timeline outputs (``rllm_stats.txt`` truncate snapshot, ``.cur_input.diff`` debug
dump, the one-shot ``rllm_config.json`` mirror) are deliberately NOT routed through
this — they are different shapes, not append-timelines.

Values are formatted by type so the existing column contracts are preserved exactly:
``int`` → bare integer, ``float`` → ``float_fmt`` (history ``.3f``; train/samples
``.4f``), anything else → ``str``. ``epoch_s``/``elapsed_s`` are always ``.3f``.
"""

from __future__ import annotations

import os
import time


class TsvLogger:
    """Append rows of ``{field: value}`` to one TSV, with a shared time prefix."""

    def __init__(self, out_dir, filename, start, *,
                 float_fmt: str = ".4f", flush_each: bool = True) -> None:
        self._path = os.path.join(out_dir, filename) if out_dir else None
        self._start = start            # shared monotonic zero (all timelines align)
        self._fmt = float_fmt
        self._flush_each = flush_each
        self._fh = None

    def _fmt_value(self, v) -> str:
        # bool is an int subclass — emit 0/1 explicitly so it never formats as a float.
        if isinstance(v, bool):
            return str(int(v))
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return format(v, self._fmt)
        return str(v)

    def write(self, fields: dict) -> None:
        """Append one row. On the first write to a *new* file, emit the header
        (``epoch_s`` + ``elapsed_s`` + the dict keys). A pre-existing file (resumed
        run) is appended to without a duplicate header."""
        if self._path is None:
            return
        try:
            if self._fh is None:
                new = not os.path.exists(self._path)
                self._fh = open(self._path, "a")
                if new:
                    self._fh.write(
                        "epoch_s\telapsed_s\t" + "\t".join(fields.keys()) + "\n")
            elapsed = time.monotonic() - self._start
            row = [f"{time.time():.3f}", f"{elapsed:.3f}"]
            row.extend(self._fmt_value(v) for v in fields.values())
            self._fh.write("\t".join(row) + "\n")
            if self._flush_each:
                self._fh.flush()
        except OSError:
            pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
