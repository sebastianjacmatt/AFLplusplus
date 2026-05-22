"""Evaluation runner — fixed-duration fuzzing campaigns with summary stats.

Wraps ``run_rllm.sh`` so you can launch one (or several) duration-bounded
fuzzing runs and get back a JSON summary of headline metrics — validity
rate, edges found, exec rate, crashes — alongside the published CovRL
Table 7 reference values for JerryScript. The plot script
(``eval/plot.py``) consumes the per-run artifacts the fuzzer itself
writes (``plot_data``, ``rllm_history.tsv``, ``rllm_seeds.tsv``); this
runner exists to drive the run and aggregate the final numbers.

Usage:
    python -m eval.runner --name eval_1h --duration 3600
    python -m eval.runner --name eval_var --duration 3600 --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RLLM_DIR = Path(__file__).resolve().parent.parent
OUT_BASE = Path.home() / "Documents" / "data_store" / "out"

# CovRL Table 7, "LLM w/o CovRL" row, JerryScript (5h, 1 CPU, 100 seeds, 5 runs).
COVRL_REF = {
    "error_pct": 78.48,
    "valid_pct": 21.52,         # 100 - error_pct
    "valid_coverage": 12833,
    "total_coverage": 14068,
    "duration_h": 5,
    "cores": 1,
    "source": "CovRL-Fuzz Table 7, ISSTA '24",
}


def _read_fuzzer_stats(path: Path) -> dict:
    """Parse AFL's fuzzer_stats key:value file."""
    out: dict = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _read_history_tail(path: Path) -> dict | None:
    """Read the last row of rllm_history.tsv as a dict."""
    if not path.exists():
        return None
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    if len(lines) < 2:
        return None
    header = lines[0].rstrip("\n").split("\t")
    last = lines[-1].rstrip("\n").split("\t")
    return dict(zip(header, last))


def _execs_per_sec_max(plot_data: Path) -> float:
    """Pull the maximum execs_per_sec seen across the run, from plot_data."""
    if not plot_data.exists():
        return 0.0
    peak = 0.0
    header = None
    with open(plot_data) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = [c.strip() for c in line.lstrip("#").split(",")]
                continue
            if header is None:
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) != len(header):
                continue
            try:
                idx = header.index("execs_per_sec")
                v = float(cols[idx])
                if v > peak:
                    peak = v
            except (ValueError, IndexError):
                continue
    return peak


def _per_run_summary(name: str, out_dir: Path) -> dict:
    """Build a summary dict for a single run from its output dir."""
    default = out_dir / "default"
    fs = _read_fuzzer_stats(default / "fuzzer_stats")
    history = _read_history_tail(default / "rllm_history.tsv")
    plot_data = default / "plot_data"

    def fnum(d: dict | None, key: str, default_val=0.0) -> float:
        if not d or key not in d:
            return float(default_val)
        try:
            return float(d[key].rstrip("%"))
        except (ValueError, AttributeError):
            return float(default_val)

    return {
        "name": name,
        "out_dir": str(out_dir),
        "run_time_s": int(fnum(fs, "run_time")),
        "execs_done": int(fnum(fs, "execs_done")),
        "execs_per_sec": fnum(fs, "execs_per_sec"),
        "execs_per_sec_max": _execs_per_sec_max(plot_data),
        "corpus_count": int(fnum(fs, "corpus_count")),
        "edges_found": int(fnum(fs, "edges_found")),
        "bitmap_cvg_pct": fnum(fs, "bitmap_cvg"),
        "saved_crashes": int(fnum(fs, "saved_crashes")),
        "saved_hangs": int(fnum(fs, "saved_hangs")),
        "muts": int(fnum(history, "muts")),
        "run_total": int(fnum(history, "run_total")),
        "valid_pct": fnum(history, "valid_pct"),
        "syntax_pct": fnum(history, "syntax_pct"),
        "semantic_pct": fnum(history, "semantic_pct"),
        "muts_per_s": fnum(history, "muts_per_s"),
    }


def _aggregate(runs: list[dict]) -> dict:
    """Mean and (when N>1) stdev for each numeric metric across runs."""
    if not runs:
        return {}
    keys = [
        "valid_pct", "syntax_pct", "semantic_pct", "edges_found",
        "corpus_count", "execs_per_sec", "execs_per_sec_max", "muts",
        "muts_per_s", "saved_crashes",
    ]
    agg: dict = {}
    for k in keys:
        vals = [r.get(k, 0) for r in runs]
        try:
            agg[f"{k}_mean"] = round(statistics.mean(vals), 3)
            if len(vals) > 1:
                agg[f"{k}_std"] = round(statistics.stdev(vals), 3)
        except statistics.StatisticsError:
            pass
    return agg


def _run_one(args: argparse.Namespace, idx: int) -> Path:
    """Launch run_rllm.sh once. Returns the run's output directory."""
    run_name = f"{args.name}-{idx}" if args.runs > 1 else args.name
    cmd = [
        "timeout", str(args.duration),
        str(RLLM_DIR / "run_rllm.sh"),
        "-i", args.dataset,
        "-o", run_name,
        "-c", args.config,
    ]
    print(f"[eval] launching: {' '.join(cmd)}", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    try:
        subprocess.run(cmd, cwd=str(RLLM_DIR), check=False)
    except KeyboardInterrupt:
        print("[eval] interrupted", file=sys.stderr)
        raise
    elapsed = time.monotonic() - t0
    print(f"[eval] run {idx} finished in {elapsed:.1f}s "
          f"(requested {args.duration}s)", file=sys.stderr, flush=True)
    return OUT_BASE / run_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-duration eval runs producing summary stats.",
    )
    parser.add_argument("--name", required=True,
                        help="Run name; output goes to <data_store>/out/<name>[-i]/")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Per-run wall-clock seconds (default: 3600 = 1h)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent runs to average (default: 1)")
    parser.add_argument("--dataset", default="final-dataset-dec22",
                        help="Seed dataset name (default: final-dataset-dec22)")
    parser.add_argument("--config", default="configs/default.json",
                        help="Mutator config (default: configs/default.json)")
    parser.add_argument("--summary-path", default=None,
                        help="Where to write eval_summary.json "
                             "(default: <data_store>/out/<name>_summary.json)")
    args = parser.parse_args()

    per_run = []
    for i in range(1, args.runs + 1):
        out_dir = _run_one(args, i)
        summary = _per_run_summary(out_dir.name, out_dir)
        per_run.append(summary)
        print(f"[eval] run {i} summary: valid={summary['valid_pct']:.1f}% "
              f"edges={summary['edges_found']} crashes={summary['saved_crashes']}",
              file=sys.stderr, flush=True)

    aggregate = _aggregate(per_run)
    out = {
        "duration_s": args.duration,
        "runs": per_run,
        "aggregate": aggregate,
        "covrl_table7_jerry_llm_w_o_covrl": COVRL_REF,
    }
    summary_path = (
        Path(args.summary_path) if args.summary_path
        else OUT_BASE / f"{args.name}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] summary written: {summary_path}", file=sys.stderr)

    # One-line comparison vs CovRL for quick eyeball.
    if aggregate:
        v = aggregate.get("valid_pct_mean", 0)
        e = aggregate.get("edges_found_mean", 0)
        ratio_v = v / COVRL_REF["valid_pct"] if COVRL_REF["valid_pct"] else 0
        ratio_e = e / COVRL_REF["total_coverage"] if COVRL_REF["total_coverage"] else 0
        print(
            f"\n[eval] ours/CovRL ratio  valid_pct: {ratio_v:.2f}x  "
            f"edges_found: {ratio_e:.2f}x  (CovRL ran 5h on 1 core)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
