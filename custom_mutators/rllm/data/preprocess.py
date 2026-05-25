"""Corpus preprocessing for the u16 token-level queue.

Walks a directory of JS files, normalizes each through UglifyJS ``-m -b``
(identifier mangle + beautify), tokenizes the result, and writes the
token IDs as little-endian uint16 binary. The result is the format AFL
ingests as ``-i`` under the TLAFL/CovRL Option B architecture (see
``docs/aligning_with_covrl.md``).

This produces the **training corpus** — the full deduped tokenized set
intended for CovRL's 4:1 train-mix during future PPO finetuning. The
**fuzz corpus** (100 validity-filtered seeds for ``afl-fuzz -i``) is
produced by a separate step, ``data.sample_seeds``, which consumes the
output of this script.

Parallelized via ``multiprocessing.Pool`` (uglifyjs is a clean subprocess
boundary). Dedup hashes the uglified bytes — matches what fuzzing
actually sees, and catches test262 templated cases that uglify
normalizes to identical bytes.

A sidecar marker file ``<output>.tokenizer.json`` lets the runner
detect "already preprocessed" and skip the step on subsequent sessions.
The marker lives *alongside* the dir, not inside it — AFL ingests every
file under ``-i`` (including dotfiles), so an in-dir marker would feed
``fuzz_count`` a JSON blob parsed as garbage u16 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from model.tokenizer import Tokenizer

DEFAULT_TOKENIZER = "Salesforce/codet5p-220m"
DEFAULT_MAX_TOKENS = 1024  # CodeT5+ pretraining source-sequence ceiling


def _require_uglifyjs() -> str:
    """Locate uglifyjs on PATH or exit with an install hint."""
    path = shutil.which("uglifyjs")
    if path:
        return path
    print(
        "[preprocess] error: 'uglifyjs' not found on PATH.\n"
        "  Install with: npm install -g uglify-js",
        file=sys.stderr,
    )
    sys.exit(2)


def _uglify(uglifyjs: str, src: bytes) -> bytes | None:
    """Run uglifyjs -m -b on ``src``; return stdout or None on failure."""
    try:
        result = subprocess.run(
            [uglifyjs, "--mangle", "--beautify"],
            input=src,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


# Per-worker state. Populated by _init_worker in each pool child; read by
# _process_file. Workers are forked, so re-importing is free, but the
# Tokenizer + uglifyjs path are heavy to look up once per file.
_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_UGLIFYJS: str | None = None
_WORKER_MAX_TOKENS: int = 0


def _init_worker(tokenizer_name: str, uglifyjs_path: str, max_tokens: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_UGLIFYJS, _WORKER_MAX_TOKENS
    _WORKER_TOKENIZER = Tokenizer(tokenizer_name)
    _WORKER_UGLIFYJS = uglifyjs_path
    _WORKER_MAX_TOKENS = max_tokens


@dataclass
class _FileResult:
    name: str
    status: str  # "kept" | "skipped_uglify" | "skipped_empty" | "read_error"
    u16: bytes | None = None
    uglified_hash: str | None = None
    truncated: bool = False
    error: str | None = None


def _process_file(src_path_str: str) -> _FileResult:
    src_path = Path(src_path_str)
    try:
        src = src_path.read_bytes()
    except OSError as exc:
        return _FileResult(name=src_path.name, status="read_error", error=str(exc))

    uglified = _uglify(_WORKER_UGLIFYJS, src)
    if uglified is None or not uglified.strip():
        return _FileResult(name=src_path.name, status="skipped_uglify")

    token_ids = _WORKER_TOKENIZER.tokenize(uglified)
    if not token_ids:
        return _FileResult(name=src_path.name, status="skipped_empty")

    truncated = False
    if len(token_ids) > _WORKER_MAX_TOKENS:
        token_ids = token_ids[: _WORKER_MAX_TOKENS]
        truncated = True

    return _FileResult(
        name=src_path.name,
        status="kept",
        u16=_WORKER_TOKENIZER.encode_u16(token_ids),
        uglified_hash=hashlib.sha256(uglified).hexdigest(),
        truncated=truncated,
    )


def preprocess_dir(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    workers: int | None = None,
    dedup: bool = True,
) -> dict:
    """Preprocess every file under ``input_dir`` into u16 binary under ``output_dir``.

    Returns a summary dict (counts + tokenizer metadata) for the caller
    to log/inspect.
    """
    uglifyjs = _require_uglifyjs()
    # Tokenizer instantiated only to read vocab_size for the marker; workers
    # load their own copies. Not load-bearing but cheap.
    vocab_size = Tokenizer(tokenizer_name).vocab_size

    output_dir.mkdir(parents=True, exist_ok=True)

    src_paths = [
        str(p) for p in sorted(input_dir.iterdir()) if p.is_file()
    ]
    total_files = len(src_paths)
    if total_files == 0:
        print(f"[preprocess] error: no files in {input_dir}", file=sys.stderr)
        sys.exit(2)

    workers = workers or os.cpu_count() or 1
    workers = max(1, min(workers, total_files))

    print(
        f"[preprocess] {total_files} files, {workers} workers, "
        f"tokenizer={tokenizer_name}, dedup={dedup}",
        file=sys.stderr,
    )

    seen_hashes: set[str] = set()
    total = 0
    kept = 0
    skipped_uglify = 0
    skipped_empty = 0
    skipped_dup = 0
    truncated = 0
    read_errors = 0

    progress_step = max(1, total_files // 50)
    t0 = time.monotonic()

    with Pool(
        workers,
        initializer=_init_worker,
        initargs=(tokenizer_name, uglifyjs, max_tokens),
    ) as pool:
        for result in pool.imap_unordered(_process_file, src_paths, chunksize=8):
            total += 1
            if result.status == "kept":
                if dedup and result.uglified_hash in seen_hashes:
                    skipped_dup += 1
                elif result.u16 is not None:
                    if dedup:
                        seen_hashes.add(result.uglified_hash)
                    (output_dir / result.name).write_bytes(result.u16)
                    kept += 1
                    if result.truncated:
                        truncated += 1
            elif result.status == "skipped_uglify":
                skipped_uglify += 1
            elif result.status == "skipped_empty":
                skipped_empty += 1
            elif result.status == "read_error":
                read_errors += 1
                print(
                    f"[preprocess] read failed: {result.name}: {result.error}",
                    file=sys.stderr,
                )

            if total % progress_step == 0 or total == total_files:
                elapsed = time.monotonic() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"[preprocess] {total}/{total_files} ({rate:.1f} files/s, "
                    f"kept={kept} dup={skipped_dup} uglify_fail={skipped_uglify})",
                    file=sys.stderr,
                )

    marker = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "max_tokens": max_tokens,
        "preprocessor": "uglifyjs -m -b",
        "dedup": dedup,
        "dedup_hash": "sha256(uglified)" if dedup else None,
        "workers": workers,
        "stats": {
            "total": total,
            "kept": kept,
            "skipped_uglify": skipped_uglify,
            "skipped_empty": skipped_empty,
            "skipped_dup": skipped_dup,
            "read_errors": read_errors,
            "truncated": truncated,
        },
    }
    marker_path = output_dir.parent / f"{output_dir.name}.tokenizer.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a JS seed directory into u16 binary token files.",
    )
    parser.add_argument("--input", required=True, help="Input directory of .js files.")
    parser.add_argument("--output", required=True, help="Output directory for u16 binary files.")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer name or local path (default: {DEFAULT_TOKENIZER}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Truncate tokenized sequences to this length (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable hash-based dedup of uglified bytes (kept by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"[preprocess] error: input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(2)

    summary = preprocess_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        workers=args.workers,
        dedup=not args.no_dedup,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
