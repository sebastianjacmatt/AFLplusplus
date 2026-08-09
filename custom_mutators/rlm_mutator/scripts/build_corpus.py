"""Clean, deduplicate, and split a raw JS dataset for SFT warmup.

Pipeline:
  1. Filter  — .js only, 50 B – 100 KB, valid UTF-8
  2. Syntax  — UglifyJS (-m -b): nonzero exit = invalid/unsupported JS, skip.
               The minified+beautified output is used for the rest of the pipeline
               so content is normalised before dedup.
  3. Dedup   — SHA-1 of the normalised bytes; one file per hash.
  4. Tokens  — drop files with <MIN_TOKENS tokens; truncate files >MAX_TOKENS.
  5. Split   — deterministic 95/5 train/val split.

Output layout:
  <out>/train/<N>.js
  <out>/val/<N>.js
  <out>/stats.json

Usage:
  python scripts/build_corpus.py \\
      --input  /path/to/raw-dataset \\
      --out    /path/to/corpus \\
      --uglify /path/to/uglifyjs   # binary, not `npx uglify-js`
      --model  Salesforce/codet5p-220m
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MIN_BYTES  = 50
MAX_BYTES  = 100 * 1024   # 100 KB
MIN_TOKENS = 20
MAX_TOKENS = 512
TIMEOUT_S  = 15           # per-file UglifyJS timeout


def _uglify(src: Path, uglify_bin: str) -> bytes | None:
    """Return normalised bytes for src, or None if syntax is invalid."""
    size = src.stat().st_size
    if size < MIN_BYTES or size > MAX_BYTES:
        return None
    try:
        raw = src.read_bytes()
        raw.decode("utf-8")          # drop non-UTF-8
    except (OSError, UnicodeDecodeError):
        return None

    try:
        result = subprocess.run(
            [uglify_bin, str(src), "-m", "-b"],
            capture_output=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def main() -> None:
    p = argparse.ArgumentParser(description="Clean JS corpus for SFT warmup.")
    p.add_argument("--input",   required=True, help="Raw .js dataset directory.")
    p.add_argument("--out",     required=True, help="Output root (train/ and val/ created here).")
    p.add_argument("--uglify",  default="/home/sebastian/.npm/_npx/1c2fd02a13a2b774/node_modules/.bin/uglifyjs",
                   help="Absolute path to the uglifyjs binary.")
    p.add_argument("--model",   default="Salesforce/codet5p-220m",
                   help="HF model id for the tokenizer.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                   help="Parallel UglifyJS workers.")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--val-frac",type=float, default=0.05)
    p.add_argument("--min-val", type=int, default=500)
    args = p.parse_args()

    if not os.path.isfile(args.uglify):
        sys.exit(f"uglifyjs binary not found: {args.uglify}\n"
                 "Pass --uglify /absolute/path/to/uglifyjs")

    out       = Path(args.out)
    train_dir = out / "train"
    val_dir   = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 1 — discover .js files                                        #
    # ------------------------------------------------------------------ #
    print("[1/5] Scanning for .js files …")
    all_js = [p for p in Path(args.input).rglob("*.js") if p.is_file()]
    print(f"      {len(all_js):,} .js files found")

    # ------------------------------------------------------------------ #
    # Stage 2 — per-file: size/UTF-8 filter + UglifyJS syntax check      #
    # ------------------------------------------------------------------ #
    print(f"[2/5] Filtering and syntax-checking ({args.workers} workers) …")
    passed:    list[bytes] = []
    n_dropped: int         = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_uglify, f, args.uglify): f for f in all_js}
        for done, fut in enumerate(as_completed(futures), 1):
            if done % 5_000 == 0:
                print(f"      {done:,}/{len(all_js):,} …  kept so far: {len(passed):,}")
            content = fut.result()
            if content is not None:
                passed.append(content)
            else:
                n_dropped += 1

    print(f"      kept {len(passed):,}  |  dropped {n_dropped:,} "
          f"(invalid syntax / too small / too large / binary)")

    # ------------------------------------------------------------------ #
    # Stage 3 — SHA-1 deduplication on normalised content                 #
    # ------------------------------------------------------------------ #
    print("[3/5] Deduplicating by SHA-1 …")
    seen:   set[bytes]   = set()
    unique: list[bytes]  = []
    for content in passed:
        h = hashlib.sha1(content).digest()
        if h not in seen:
            seen.add(h)
            unique.append(content)
    n_dups = len(passed) - len(unique)
    print(f"      {len(unique):,} unique  |  {n_dups:,} duplicates removed "
          f"({100 * n_dups / max(len(passed), 1):.1f}%)")

    # ------------------------------------------------------------------ #
    # Stage 4 — tokenise, filter short, truncate long                     #
    # ------------------------------------------------------------------ #
    print(f"[4/5] Tokenising with '{args.model}' …")
    from transformers import AutoTokenizer   # import late so --help is fast
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    records:     list[bytes] = []
    n_too_short: int         = 0
    n_truncated: int         = 0

    for content in unique:
        text = content.decode("utf-8", errors="replace")
        ids  = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < MIN_TOKENS:
            n_too_short += 1
            continue
        if len(ids) > MAX_TOKENS:
            text = tokenizer.decode(ids[:MAX_TOKENS], skip_special_tokens=True)
            n_truncated += 1
        records.append(text.encode("utf-8"))

    print(f"      {len(records):,} records  |  "
          f"{n_too_short:,} too short  |  {n_truncated:,} truncated to {MAX_TOKENS} tokens")

    if len(records) < args.min_val + 1:
        sys.exit(f"Too few records ({len(records)}) to split — check your input.")

    # ------------------------------------------------------------------ #
    # Stage 5 — deterministic train / val split                           #
    # ------------------------------------------------------------------ #
    print("[5/5] Splitting …")
    records_sorted = sorted(records)       # deterministic before shuffle
    random.seed(args.seed)
    random.shuffle(records_sorted)

    n_val = max(args.min_val, int(len(records_sorted) * args.val_frac))
    n_val = min(n_val, len(records_sorted) - 1)

    val_set   = records_sorted[:n_val]
    train_set = records_sorted[n_val:]

    def _write(dest: Path, items: list[bytes], label: str) -> None:
        w = len(str(len(items) - 1))
        for i, content in enumerate(items):
            (dest / f"{i:0{w}d}.js").write_bytes(content)
        print(f"      {label:5s}: {len(items):,} files → {dest}")

    _write(train_dir, train_set, "train")
    _write(val_dir,   val_set,   "val")

    stats = {
        "raw_js_files":        len(all_js),
        "after_syntax_filter": len(passed),
        "after_dedup":         len(unique),
        "after_token_filter":  len(records),
        "train":               len(train_set),
        "val":                 len(val_set),
    }
    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nDone.  stats → {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
