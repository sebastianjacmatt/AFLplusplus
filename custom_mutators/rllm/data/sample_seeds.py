"""Sample N validity-filtered seeds from a preprocessed u16 corpus.

Builds the **fuzz corpus** that AFL ingests as ``-i``. The input is the
full **training corpus** produced by ``data.preprocess`` (u16 binary
files); the output is a small directory of u16 files (default 100)
selected to be:

  1. **Valid on the target.** Each candidate is decoded back to JS and
     executed; only those with empty stderr and exit code 0 survive. This
     also filters out files that need a test-suite harness (test262's
     ``assert.js``, V8's ``mjsunit.js``, Chakra's ``WScript`` etc.) —
     they fail with ``ReferenceError`` and are excluded automatically.
  2. **Coverage-distinct.** ``afl-cmin`` reduces the validity-filtered
     set to one file per unique coverage signature (AFL++
     ``docs/fuzzing_in_depth.md`` §2b "highly recommended").
  3. **Sampled deterministically.** Reservoir sample with a fixed seed.

The script does not modify the u16 files — it copies the chosen entries
from ``--input`` to ``--output`` by filename. A sidecar marker
``<output>.tokenizer.json`` records the selection metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from data.validity import classify_stderr
from model.tokenizer import Tokenizer

DEFAULT_TOKENIZER = "Salesforce/codet5p-220m"
DEFAULT_TIMEOUT_S = 5
DEFAULT_NUM_SEEDS = 100

# Sources we recognize for the per-source breakdown in the marker. Filenames
# from the raw-dataset-dec22 layout encode source in the leading token before
# `__` (e.g. `test262__language__...`, `v8__mjsunit__...`).
_KNOWN_SOURCES = (
    "test262",
    "javascriptcore",
    "v8",
    "chakra",
    "jerry",
    "js-vuln-db",
)


def _source_of(name: str) -> str:
    head = name.split("__", 1)[0]
    return head if head in _KNOWN_SOURCES else "other"


# Per-worker state, initialized once per pool child.
_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_TARGET: str | None = None
_WORKER_TIMEOUT: float = DEFAULT_TIMEOUT_S
_WORKER_JS_DIR: str | None = None


def _init_worker(
    tokenizer_name: str, target_bin: str, timeout_s: float, js_dir: str
) -> None:
    global _WORKER_TOKENIZER, _WORKER_TARGET, _WORKER_TIMEOUT, _WORKER_JS_DIR
    _WORKER_TOKENIZER = Tokenizer(tokenizer_name)
    _WORKER_TARGET = target_bin
    _WORKER_TIMEOUT = timeout_s
    _WORKER_JS_DIR = js_dir


@dataclass
class _ValidityResult:
    name: str
    cls: str  # "valid" | "syntax" | "semantic" | "timeout" | "read_error"
    exit_code: int = -1


def _check_validity(u16_path_str: str) -> _ValidityResult:
    """Worker task: decode → run target → classify.

    On 'valid', leaves the decoded JS file in ``_WORKER_JS_DIR`` so the
    main process can hand the directory to ``afl-cmin``. On non-valid,
    cleans the JS file up to keep the tmp dir small.
    """
    u16_path = Path(u16_path_str)
    name = u16_path.name
    js_path = Path(_WORKER_JS_DIR) / name

    try:
        u16 = u16_path.read_bytes()
    except OSError:
        return _ValidityResult(name=name, cls="read_error")

    token_ids = _WORKER_TOKENIZER.parse_u16(u16)
    if not token_ids:
        return _ValidityResult(name=name, cls="read_error")

    js_bytes = _WORKER_TOKENIZER.detokenize(token_ids)
    js_path.write_bytes(js_bytes)

    try:
        result = subprocess.run(
            [_WORKER_TARGET, str(js_path)],
            capture_output=True,
            timeout=_WORKER_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        js_path.unlink(missing_ok=True)
        return _ValidityResult(name=name, cls="timeout")

    stderr = result.stderr.decode("utf-8", errors="replace")
    cls = classify_stderr(stderr)
    if not (cls == "valid" and result.returncode == 0):
        # Bias 'valid'-classified with nonzero exit toward 'semantic' to
        # match how mutator.post_run treats failures.
        if cls == "valid":
            cls = "semantic"
        js_path.unlink(missing_ok=True)

    return _ValidityResult(name=name, cls=cls, exit_code=result.returncode)


def _aflpp_root() -> Path:
    """The AFL++ tree this mutator is bundled in.

    ``data/sample_seeds.py`` → ``rllm`` → ``custom_mutators`` → AFL++ root.
    """
    return Path(__file__).resolve().parents[3]


def _require_afl_cmin() -> Path:
    """Hard-coded path: ``<AFL++ root>/afl-cmin``. Errors out if missing."""
    path = _aflpp_root() / "afl-cmin"
    if not path.is_file():
        print(
            f"[sample_seeds] error: afl-cmin not found at {path}.\n"
            "  Build AFL++ first: cd to the AFL++ tree and run `make`.",
            file=sys.stderr,
        )
        sys.exit(2)
    return path


def _run_afl_cmin(
    afl_cmin: str, target_bin: str, in_dir: Path, out_dir: Path
) -> None:
    """Invoke afl-cmin on a JS corpus. Raises on failure.

    Inherits the caller's environment. If afl-showmap (called by cmin
    internally) errors out on a missing ``libpython3.X.so``, the user
    needs to set ``LD_LIBRARY_PATH`` to include their conda env's lib
    dir — this is documented in the README prerequisites.
    """
    cmd = [afl_cmin, "-i", str(in_dir), "-o", str(out_dir), "--", target_bin, "@@"]
    print(f"[sample_seeds] running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"afl-cmin failed (exit {proc.returncode})")


def sample_seeds(
    input_dir: Path,
    output_dir: Path,
    target_bin: Path,
    afl_cmin: Path,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    workers: int | None = None,
    seed: int = 42,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    candidates: int | None = None,
    do_cmin: bool = True,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> dict:
    """Run the full seed-selection pipeline. Returns a summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = sorted(p for p in input_dir.iterdir() if p.is_file())
    if candidates is not None:
        candidate_paths = candidate_paths[:candidates]
    if not candidate_paths:
        print(f"[sample_seeds] error: no u16 files in {input_dir}", file=sys.stderr)
        sys.exit(2)

    workers = workers or os.cpu_count() or 1
    workers = max(1, min(workers, len(candidate_paths)))
    print(
        f"[sample_seeds] validity-checking {len(candidate_paths)} candidates "
        f"with {workers} workers (target={target_bin.name})",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory(prefix="rllm-seedsel-") as tmp_root:
        tmp_root_path = Path(tmp_root)
        valid_js_dir = tmp_root_path / "valid_js"
        valid_js_dir.mkdir()

        # Phase 1: validity filter (parallel).
        class_counts: Counter[str] = Counter()
        progress_step = max(1, len(candidate_paths) // 50)
        t0 = time.monotonic()

        with Pool(
            workers,
            initializer=_init_worker,
            initargs=(tokenizer_name, str(target_bin), timeout_s, str(valid_js_dir)),
        ) as pool:
            iterator = pool.imap_unordered(
                _check_validity,
                [str(p) for p in candidate_paths],
                chunksize=4,
            )
            for i, result in enumerate(iterator, start=1):
                class_counts[result.cls] += 1
                if i % progress_step == 0 or i == len(candidate_paths):
                    elapsed = time.monotonic() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    print(
                        f"[sample_seeds] {i}/{len(candidate_paths)} "
                        f"({rate:.1f}/s, valid={class_counts['valid']})",
                        file=sys.stderr,
                    )

        num_valid = class_counts["valid"]
        if num_valid == 0:
            print(
                "[sample_seeds] error: no valid candidates found. "
                "Check the target binary and timeout.",
                file=sys.stderr,
            )
            sys.exit(2)
        if num_valid < num_seeds:
            print(
                f"[sample_seeds] warning: only {num_valid} valid candidates "
                f"(< requested {num_seeds}); using all of them.",
                file=sys.stderr,
            )

        # Phase 2: afl-cmin on the validity-filtered set.
        if do_cmin:
            cmin_out = tmp_root_path / "cmin_js"
            _run_afl_cmin(str(afl_cmin), str(target_bin), valid_js_dir, cmin_out)
            survivor_paths = sorted(p for p in cmin_out.iterdir() if p.is_file())
            print(
                f"[sample_seeds] cmin survivors: {len(survivor_paths)}",
                file=sys.stderr,
            )
        else:
            survivor_paths = sorted(p for p in valid_js_dir.iterdir() if p.is_file())

        num_after_cmin = len(survivor_paths)
        if num_after_cmin == 0:
            print("[sample_seeds] error: cmin output is empty", file=sys.stderr)
            sys.exit(2)

        # Phase 3: reservoir-sample N (uniform from survivors).
        rng = random.Random(seed)
        n_to_pick = min(num_seeds, num_after_cmin)
        chosen_js = rng.sample(survivor_paths, n_to_pick)

        # Phase 4: copy chosen u16 files from --input to --output by name.
        # No afl-tmin: it does byte-level deletions that mangle JS structure
        # (deleted brackets/commas, identifiers gibberish-renamed); the
        # resulting bytes wouldn't be canonical uglify output and re-
        # tokenization would yield a degenerate u16 sequence.
        for js_path in chosen_js:
            shutil.copy(input_dir / js_path.name, output_dir / js_path.name)

    per_source = Counter(_source_of(p.name) for p in chosen_js)

    marker = {
        "input_corpus": str(input_dir),
        "target": str(target_bin),
        "num_seeds_requested": num_seeds,
        "num_chosen": len(chosen_js),
        "sampling_seed": seed,
        "do_cmin": do_cmin,
        "stats": {
            "num_candidates": len(candidate_paths),
            "num_valid": num_valid,
            "num_after_cmin": num_after_cmin,
            "class_counts": dict(class_counts),
            "per_source": dict(per_source),
        },
    }
    marker_path = output_dir.parent / f"{output_dir.name}.tokenizer.json"
    marker_path.write_text(json.dumps(marker, indent=2) + "\n")
    return marker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample N validity-filtered seeds from a u16 corpus.",
    )
    parser.add_argument("--input", required=True, help="u16 corpus (output of data.preprocess).")
    parser.add_argument("--output", required=True, help="Output dir for sampled u16 seeds.")
    parser.add_argument("--target", required=True, help="Target interpreter (e.g. path to jerry).")
    parser.add_argument(
        "--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
        help=f"Seeds to sample (default: {DEFAULT_NUM_SEEDS}).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Validity-check worker count (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sampling (default: 42).",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_S,
        help=f"Per-file target-run timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--candidates", type=int, default=None,
        help="Cap validity-check at the first N u16 files (default: all).",
    )
    parser.add_argument(
        "--no-cmin", action="store_true",
        help="Skip afl-cmin; sample directly from the validity-filtered set.",
    )
    parser.add_argument(
        "--tokenizer", default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer (default: {DEFAULT_TOKENIZER}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    target_bin = Path(args.target).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"[sample_seeds] error: input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(2)
    if not target_bin.is_file() or not os.access(target_bin, os.X_OK):
        print(f"[sample_seeds] error: target not executable: {target_bin}", file=sys.stderr)
        sys.exit(2)

    afl_cmin = _require_afl_cmin() if not args.no_cmin else Path("/dev/null")

    summary = sample_seeds(
        input_dir=input_dir,
        output_dir=output_dir,
        target_bin=target_bin,
        afl_cmin=afl_cmin,
        num_seeds=args.num_seeds,
        workers=args.workers,
        seed=args.seed,
        timeout_s=args.timeout,
        candidates=args.candidates,
        do_cmin=not args.no_cmin,
        tokenizer_name=args.tokenizer,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
