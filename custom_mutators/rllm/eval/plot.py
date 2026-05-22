"""Generate the evaluation figure from a single run's artifacts.

Reads three files from a finished fuzzing run:

  - ``plot_data``         — AFL's standard CSV (relative_time, execs_per_sec,
                            edges_found, cur_item, corpus_count, ...)
  - ``rllm_history.tsv``  — our append-only flush history (validity %)
  - ``rllm_seeds.tsv``    — our seed-selection log (which seed was active)

and produces a four-panel figure stacked on a shared time axis:

  Row 1  — exec_rate (execs/sec)
  Row 2  — validity_rate (% valid, cumulative + windowed-N)
  Row 3  — coverage (edges_found)
  Row 4  — active seed strip (queue index over time)

Vertical dashed lines mark seed switches where windowed validity dropped
>5 percentage points from the cumulative average — surfaces "this seed
degrades validity" at a glance.

Usage:
    python -m eval.plot --out-dir <out_dir>/default [--output plot.png] [--window 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# CovRL Table 7 reference for the comparison annotation.
COVRL_REF = {
    "error_pct": 78.48,
    "valid_pct": 21.52,
    "valid_coverage": 12833,
    "total_coverage": 14068,
    "duration_h": 5,
    "cores": 1,
}


def _read_plot_data(path: Path) -> pd.DataFrame:
    """Parse AFL's plot_data into a DataFrame. Header lines start with #."""
    if not path.exists():
        return pd.DataFrame()
    header = None
    rows = []
    with open(path) as f:
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
            rows.append(cols)
    if not rows or header is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    # Numeric coercion where it makes sense; leave bitmap_cvg as-is (has %)
    for c in ("relative_time", "cycles_done", "cur_item", "corpus_count",
              "pending_total", "pending_favs", "saved_crashes", "saved_hangs",
              "max_depth", "execs_per_sec", "total_execs", "edges_found",
              "total_crashes", "servers_count"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time_min"] = df["relative_time"] / 60.0
    return df


def _read_history(path: Path) -> pd.DataFrame:
    """Parse rllm_history.tsv."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return df
    df["time_min"] = df["elapsed_s"] / 60.0
    return df


def _read_seeds(path: Path) -> pd.DataFrame:
    """Parse rllm_seeds.tsv."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return df
    df["time_min"] = df["elapsed_s"] / 60.0
    return df


def _windowed_validity(history: pd.DataFrame, window_muts: int) -> pd.Series:
    """Per-row validity over the last `window_muts` mutations.

    Approximates a fixed-mutation window using the history's per-row
    run_valid / run_total deltas. The window is best-effort; if too
    few rows back have accumulated `window_muts` mutations, we fall
    back to using all available history (i.e., cumulative).
    """
    if history.empty:
        return pd.Series([], dtype=float)
    n = len(history)
    out = []
    for i in range(n):
        cur_total = history.iloc[i]["run_total"]
        cur_valid = history.iloc[i]["run_valid"]
        # find the earliest j such that cur_total - history[j].run_total >= window_muts
        j = i
        while j > 0 and cur_total - history.iloc[j]["run_total"] < window_muts:
            j -= 1
        d_total = cur_total - history.iloc[j]["run_total"]
        d_valid = cur_valid - history.iloc[j]["run_valid"]
        if d_total > 0:
            out.append(d_valid / d_total * 100.0)
        else:
            # No data inside the window — fall back to cumulative valid%
            out.append(
                cur_valid / max(cur_total, 1) * 100.0
            )
    return pd.Series(out, index=history.index)


def _seed_index_map(seeds: pd.DataFrame) -> tuple[dict, list[str]]:
    """Map seed filename -> stable integer index, in order of first appearance."""
    if seeds.empty:
        return {}, []
    mapping: dict = {}
    order: list[str] = []
    for name in seeds["filename"]:
        if name not in mapping:
            mapping[name] = len(mapping)
            order.append(name)
    return mapping, order


def _annotation_text(history: pd.DataFrame, afl: pd.DataFrame) -> str:
    """One-paragraph CovRL comparison shown on the figure."""
    if history.empty or afl.empty:
        return "(no data yet)"
    final_valid = history.iloc[-1]["valid_pct"]
    final_edges = int(afl.iloc[-1]["edges_found"])
    final_min = history.iloc[-1]["time_min"]
    ratio_v = final_valid / COVRL_REF["valid_pct"] if COVRL_REF["valid_pct"] else 0
    ratio_e = final_edges / COVRL_REF["total_coverage"] if COVRL_REF["total_coverage"] else 0
    return (
        f"final ({final_min:.1f} min):\n"
        f"  valid_pct  {final_valid:5.1f}%  "
        f"(CovRL 5h: {COVRL_REF['valid_pct']:.1f}%, ratio {ratio_v:.2f}x)\n"
        f"  edges      {final_edges:5d}    "
        f"(CovRL 5h: {COVRL_REF['total_coverage']:,d}, ratio {ratio_e:.2f}x)"
    )


def build_figure(out_dir: Path, window: int) -> plt.Figure:
    afl = _read_plot_data(out_dir / "plot_data")
    hist = _read_history(out_dir / "rllm_history.tsv")
    seeds = _read_seeds(out_dir / "rllm_seeds.tsv")

    seed_map, seed_order = _seed_index_map(seeds)
    if not seeds.empty:
        seeds = seeds.copy()
        seeds["seed_idx"] = seeds["filename"].map(seed_map)

    if not hist.empty:
        hist = hist.copy()
        hist["valid_pct_windowed"] = _windowed_validity(hist, window)

    fig, axes = plt.subplots(
        4, 1, sharex=True, figsize=(12, 10),
        gridspec_kw={"height_ratios": [2, 2, 2, 1]},
    )
    ax_exec, ax_valid, ax_cov, ax_seed = axes

    # Row 1 — exec_rate
    if not afl.empty:
        ax_exec.plot(afl["time_min"], afl["execs_per_sec"],
                     color="tab:blue", lw=1.5, label="execs/sec")
    ax_exec.set_ylabel("execs/sec")
    ax_exec.grid(True, alpha=0.3)
    ax_exec.legend(loc="upper right", fontsize=9)

    # Row 2 — validity (cumulative + windowed)
    if not hist.empty:
        ax_valid.plot(hist["time_min"], hist["valid_pct"],
                      color="tab:green", lw=1.8, label="valid_pct (cumulative)")
        ax_valid.plot(hist["time_min"], hist["valid_pct_windowed"],
                      color="tab:green", lw=1.0, ls="--",
                      label=f"valid_pct (last {window} muts)")
        ax_valid.axhline(COVRL_REF["valid_pct"], color="grey", lw=0.8,
                         ls=":", alpha=0.7,
                         label=f"CovRL ref {COVRL_REF['valid_pct']:.1f}%")
    ax_valid.set_ylabel("valid_pct (%)")
    ax_valid.set_ylim(0, 100)
    ax_valid.grid(True, alpha=0.3)
    ax_valid.legend(loc="upper right", fontsize=9)

    # Row 3 — coverage (edges_found)
    if not afl.empty:
        ax_cov.plot(afl["time_min"], afl["edges_found"],
                    color="tab:red", lw=1.5, label="edges_found")
        ax_cov.axhline(COVRL_REF["total_coverage"], color="grey", lw=0.8,
                       ls=":", alpha=0.7,
                       label=f"CovRL ref {COVRL_REF['total_coverage']:,d}")
    ax_cov.set_ylabel("edges found")
    ax_cov.grid(True, alpha=0.3)
    ax_cov.legend(loc="lower right", fontsize=9)

    # Row 4 — seed strip
    if not seeds.empty:
        ax_seed.scatter(seeds["time_min"], seeds["seed_idx"],
                        s=4, c="tab:purple", alpha=0.6)
        # Annotate the top-K most fuzzed seeds on the right edge.
        top_k = 8
        counts = seeds["filename"].value_counts().head(top_k)
        for name, _ in counts.items():
            idx = seed_map[name]
            short = name[-32:] if len(name) > 32 else name
            ax_seed.text(
                seeds["time_min"].max() * 1.005, idx, short,
                fontsize=7, va="center", ha="left", alpha=0.8,
            )
    ax_seed.set_ylabel("active seed")
    ax_seed.set_xlabel("time (minutes since init)")
    ax_seed.grid(True, alpha=0.3)

    # Cross-subplot vertical lines at validity drops > 5pp.
    if not hist.empty:
        cum_mean = hist["valid_pct"].mean()
        drops = hist[hist["valid_pct_windowed"] < cum_mean - 5.0]
        if not drops.empty:
            # cluster: only mark events that are >30s apart so we don't litter
            last_t = -1e9
            for _, row in drops.iterrows():
                t = row["time_min"]
                if t - last_t < 0.5:
                    continue
                last_t = t
                for ax in axes:
                    ax.axvline(t, color="grey", lw=0.5, ls="--", alpha=0.4)

    # Title + comparison annotation.
    fig.suptitle(
        f"rllm — TLAFL on JerryScript  (out: {out_dir.parent.name})",
        fontsize=12, fontweight="bold", y=0.995,
    )
    text = _annotation_text(hist, afl)
    fig.text(0.99, 0.965, text, ha="right", va="top",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))

    fig.tight_layout(rect=[0, 0, 0.96, 0.96])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the evaluation figure for a finished run.",
    )
    parser.add_argument("--out-dir", required=True,
                        help="Path to the run's <name>/default directory.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <out-dir>/eval_plot.png).")
    parser.add_argument("--window", type=int, default=100,
                        help="Window size (in mutations) for the windowed "
                             "validity line (default: 100).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if not out_dir.is_dir():
        print(f"[eval.plot] not a directory: {out_dir}", file=sys.stderr)
        sys.exit(2)

    output = Path(args.output) if args.output else out_dir / "eval_plot.png"

    fig = build_figure(out_dir, args.window)
    fig.savefig(output, dpi=130)
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[eval.plot] wrote {output} and {pdf_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
