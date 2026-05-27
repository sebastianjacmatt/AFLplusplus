"""Seed preprocessing for the u16 token-level queue.

Mirrors ``CovRL-Fuzz/covrl/utils/preprocess.py``: walks a directory of JS
files, normalizes each through UglifyJS ``-m -b`` (identifier mangle +
beautify), tokenizes the result, and writes the token IDs as little-endian
uint16 binary. The result is the format AFL's ``-i`` directory must be in
for the TLAFL/CovRL Option B architecture in ``docs/aligning_with_covrl.md``.

Run once per seed corpus; the marker file ``.rllm_tokenizer`` lets the
runner skip the step on subsequent fuzzing sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
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


def preprocess_dir(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Preprocess every file under ``input_dir`` into u16 binary under ``output_dir``.

    Returns a small summary dict (counts) for the caller to log/inspect.
    """
    uglifyjs = _require_uglifyjs()
    tokenizer = Tokenizer(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    skipped_uglify = 0
    skipped_empty = 0
    skipped_dup = 0
    truncated = 0
    seen_hashes: set[str] = set()

    for src_path in sorted(input_dir.iterdir()):
        if not src_path.is_file():
            continue
        total += 1
        try:
            src = src_path.read_bytes()
        except OSError as exc:
            print(f"[preprocess] read failed: {src_path}: {exc}", file=sys.stderr)
            continue

        uglified = _uglify(uglifyjs, src)
        if uglified is None or not uglified.strip():
            skipped_uglify += 1
            continue

        h = hashlib.sha256(uglified).hexdigest()
        if h in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(h)

        token_ids = tokenizer.tokenize(uglified)
        if not token_ids:
            skipped_empty += 1
            continue

        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            truncated += 1

        out_path = output_dir / src_path.name
        out_path.write_bytes(tokenizer.encode_u16(token_ids))
        kept += 1

    marker = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "max_tokens": max_tokens,
        "preprocessor": "uglifyjs -m -b",
        "stats": {
            "total": total,
            "kept": kept,
            "skipped_uglify": skipped_uglify,
            "skipped_empty": skipped_empty,
            "skipped_dup": skipped_dup,
            "truncated": truncated,
        },
    }
    # The marker lives *alongside* the seed dir, not inside it. AFL++
    # ingests every file under -i as a seed (including dotfiles), so
    # placing the marker in the dir would feed it to fuzz() as a JSON
    # blob parsed as garbage u16 tokens. We write it next to the dir
    # with a `.tokenizer.json` suffix on the dir name.
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
