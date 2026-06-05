#!/usr/bin/env python3
"""Offline probe for the GRPO coverage signal (backs docs/coverage_signal.md §3).

Measures, on **real jerry coverage**, how the levers move the signal and verifies the
decomposition identity from §1:

    Var_g(r) = V_validity(g) + V_coverage(g),
    V_coverage(g) = p_valid(g) * (1-b)^2 * Var(R_cov | valid, g)

Mirrors the live mutator path exactly:
  parse_u16 -> Masking.mask -> Model.batch_generate -> reconstruct_tokens -> detokenize
  -> jerry (validity, stderr) + afl-showmap (edge set) -> TF*IDF reward.

It contrasts two knobs:
  * A1 (span size): current single-token masks vs the `span~2.5` sweet spot — `meanJacc`
    (scale-free edge diversity) and `p_valid`.
  * B1 (delta-vs-parent reward, `σ(log Σ_new idf)`, new = edges ∉ the parent SEED — a
    *stationary* per-context baseline, NOT the global frontier) vs the absolute reward
    `σ(log Σ_all idf)`, and a B2 (de-saturate) diagnostic.

Findings (see the doc): the identity holds to ~1e-16; B1 ≈ 800x the signal magnitude while
B2 alone is negligible (jerry hits ~1300 shared startup edges every run, so absolute
coverage variance is between-seed = cancelled by the within-group baseline; delta moves it
in-group); magnitude (S_cov) and share (η) are orthogonal — η is bottlenecked by the
all-valid-group rate, not the reward.

Run (needs the conda env that has torch/transformers + an AFL-instrumented jerry):
    python eval/signal_probe.py [N_SEEDS] [GROUPS] [G]
Paths are overridable via RLLM_JERRY / RLLM_SHOWMAP / RLLM_SEEDS. afl-showmap is linked
against libpython (AFL built with the Python mutator), so $CONDA/lib is put on
LD_LIBRARY_PATH for the subprocess calls.
"""
import os, sys, math, subprocess, random, tempfile
import numpy as np

RLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # eval/ -> rllm/
sys.path.insert(0, RLLM_DIR)

# afl-showmap is linked against libpython (AFL built with the Python mutator);
# make the interpreter's lib discoverable for the subprocess calls.
os.environ["LD_LIBRARY_PATH"] = sys.prefix + "/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from data.masking import Masking
from data.validity import classify_stderr
from model.llm import Model

_AFLPP  = os.path.dirname(os.path.dirname(RLLM_DIR))                     # -> AFLplusplus/
JERRY   = os.environ.get("RLLM_JERRY",
                         os.path.expanduser("~/Documents/data_store/engines/jerryscript/build/bin/jerry"))
SHOWMAP = os.environ.get("RLLM_SHOWMAP", os.path.join(_AFLPP, "afl-showmap"))
SEEDDIR = os.environ.get("RLLM_SEEDS",
                         os.path.expanduser("~/Documents/data_store/dataset/dataset-dec22-u16-seeds"))
BITMAP_SIZE = 18313        # real jerry map (cfg.bitmap_size)
MAP_SCALE   = math.sqrt(BITMAP_SIZE)
B = 0.5                    # cfg.validity_bonus

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GROUPS  = int(sys.argv[2]) if len(sys.argv) > 2 else 6
G       = int(sys.argv[3]) if len(sys.argv) > 3 else 12

# (name, Masking-kwargs, max_new_tokens) — current single-token vs the span~2.5 sweet spot
CONFIGS = [
    ("tok x1-3", dict(max_masks=3, corruption_rate=0.03, mean_span_length=1.0, min_span_length=1, max_span_length=1), 24),
    ("span~2.5", dict(max_masks=0, corruption_rate=0.07, mean_span_length=2.5, min_span_length=1, max_span_length=5), 28),
]

_tmp = tempfile.mkdtemp(prefix="sigprobe_")
_js  = os.path.join(_tmp, "in.js")
_cov = os.path.join(_tmp, "cov.txt")


def run_validity(src: bytes) -> str:
    with open(_js, "wb") as f:
        f.write(src)
    try:
        p = subprocess.run([JERRY, _js], capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "timeout"
    return classify_stderr(p.stderr.decode("utf-8", "replace"))


def run_edges(src: bytes):
    with open(_js, "wb") as f:
        f.write(src)
    try:
        subprocess.run([SHOWMAP, "-o", _cov, "-m", "none", "-t", "5000", "-q", "--",
                        JERRY, _js], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        return set()
    edges = set()
    try:
        with open(_cov) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = int(line.split(":")[0])
                    if 0 <= e < BITMAP_SIZE:
                        edges.add(e)
    except OSError:
        pass
    return edges


def main():
    rng = random.Random(1234)
    print("loading model (codet5p-220m)...", flush=True)
    model = Model("Salesforce/codet5p-220m",
                  gen_kwargs=dict(do_sample=True, temperature=1.0, top_p=0.95,
                                  top_k=50, no_repeat_ngram_size=3),
                  max_new_tokens=24, device="auto")
    tok = model.tokenizer

    # --- load valid parent seeds -------------------------------------------------
    seeds = []                       # (name, tokens, parent_edges)
    df = np.zeros(BITMAP_SIZE, dtype=np.int64)
    n_docs = 0
    files = sorted(os.listdir(SEEDDIR))
    rng.shuffle(files)
    for fn in files:
        if len(seeds) >= N_SEEDS:
            break
        with open(os.path.join(SEEDDIR, fn), "rb") as f:
            toks = tok.parse_u16(f.read())
        if not (30 <= len(toks) <= 700):
            continue
        toks = toks[:768]
        src = tok.detokenize(toks)
        if run_validity(src) != "valid":
            continue
        pe = run_edges(src)
        if not pe:
            continue
        df[list(pe)] += 1; n_docs += 1
        seeds.append((fn, toks, pe))
        print(f"  seed {len(seeds)}: {fn[:48]:48s} len={len(toks):4d} |E_parent|={len(pe)}", flush=True)

    # --- sweep: per config, per seed, per group, G infills -----------------------
    recs = {}                        # (cfg, seed_idx, group) -> [(cls, edges), ...]
    for cname, mkw, mnt in CONFIGS:
        masking = Masking(sentinel_ids=tok.sentinel_ids, word_starts_fn=None,
                          g=G, rng=random.Random(7), **mkw)
        for si, (fn, toks, pe) in enumerate(seeds):
            for grp in range(GROUPS):
                mp = masking.mask(toks)
                ys = model.batch_generate([mp.input_ids], n_samples=G, max_new_tokens=mnt)
                lst = []
                for y in ys:
                    src = tok.reconstruct(mp, y)
                    cls = run_validity(src)
                    edges = run_edges(src)
                    df[list(edges)] += 1; n_docs += 1
                    lst.append((cls, edges))
                recs[(cname, si, grp)] = lst
            v = sum(1 for g in range(GROUPS) for c, _ in recs[(cname, si, g)] if c == "valid")
            print(f"  [{cname}] seed {si+1}/{len(seeds)} valid={v}/{GROUPS*G}", flush=True)

    # --- IDF over all executions (CovRL Eq.4): idf = log(N/(1+df)) / sqrt(M) ------
    idf = (np.log(n_docs / (1.0 + df)) / MAP_SCALE).astype(np.float64)

    def wmass(edges):
        return float(idf[list(edges)].sum()) if edges else 0.0

    sig = lambda x: 1.0 / (1.0 + math.exp(-x))

    # Two reward functions, both natural [0,1] (same σ transform) ⇒ directly comparable:
    #   cur = σ(log Σ_all idf) ; B1/delta = σ(log Σ_new idf), new = edges ∉ parent SEED.
    def cur_R(edges, pe):
        w  = wmass(edges);       return sig(math.log(w))  if w  > 0 else 0.5
    def delta_R(edges, pe):
        wn = wmass(edges - pe);  return sig(math.log(wn)) if wn > 0 else 0.5

    def jacc(es):                                    # mean pairwise Jaccard distance
        if len(es) < 2: return float("nan")
        ds = []
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                u = len(es[i] | es[j])
                ds.append(1.0 - (len(es[i] & es[j]) / u if u else 1.0))
        return float(np.mean(ds))

    pe_by_seed = {si: seeds[si][2] for si in range(len(seeds))}

    def decompose(gkeys, Rf):
        """Mean over groups of Var(Rcov|valid), S_cov(=V_cov), η, %groups-with-cov-var,
        and the max law-of-total-variance residual |Var_g(r)-(V_val+V_cov)|."""
        vcovs, scovs, etas, resids, covsig = [], [], [], [], []
        for k in gkeys:
            _, si, _ = k; pe = pe_by_seed[si]
            rvalid, r_all = [], []
            for c, e in recs[k]:
                if   c == "valid":    rc = Rf(e, pe); rvalid.append(rc); r_all.append(B + (1 - B) * rc)
                elif c == "semantic": r_all.append(-0.5)
                else:                 r_all.append(-1.0)        # syntax / timeout
            r_all = np.array(r_all)
            if len(r_all) == 0: continue
            p_valid  = len(rvalid) / len(r_all)
            var_rcov = float(np.var(rvalid)) if len(rvalid) >= 2 else 0.0
            v_cov    = p_valid * (1 - B) ** 2 * var_rcov
            grand    = r_all.mean()
            v_val, classes = 0.0, []
            if rvalid: classes.append([B + (1 - B) * rc for rc in rvalid])
            for vv in (-1.0, -0.5):
                sub = [x for x in r_all.tolist() if x == vv]
                if sub: classes.append(sub)
            for gr in classes:
                gr = np.array(gr); v_val += (len(gr) / len(r_all)) * (gr.mean() - grand) ** 2
            var_r = float(np.var(r_all))
            resids.append(abs(var_r - (v_cov + v_val)))
            vcovs.append(var_rcov); scovs.append(v_cov)
            covsig.append(1.0 if var_rcov > 1e-9 else 0.0)
            if var_r > 1e-12: etas.append(v_cov / var_r)
        return (np.mean(vcovs), np.mean(scovs),
                (np.mean(etas) if etas else float("nan")),
                100 * np.mean(covsig), max(resids))

    print("\n" + "=" * 82)
    print(f"N_seeds={len(seeds)}  groups/seed={GROUPS}  G={G}  b={B}  "
          f"executions={n_docs}  edges_seen={(df > 0).sum()}")
    print("=" * 82)

    for cname, _, _ in CONFIGS:
        gkeys = [(cname, si, grp) for si in range(len(seeds)) for grp in range(GROUPS)]
        pv, jaccs, logall, lognew = [], [], [], []
        for k in gkeys:
            _, si, _ = k; pe = pe_by_seed[si]
            val = [e for c, e in recs[k] if c == "valid"]
            pv.append(len(val) / len(recs[k])); jaccs.append(jacc(val))
            for e in val:
                w = wmass(e); wn = wmass(e - pe)
                if w  > 0: logall.append(math.log(w))
                if wn > 0: lognew.append(math.log(wn))
        jv = [x for x in jaccs if not math.isnan(x)]
        print(f"\n### {cname}:  p_valid={np.mean(pv):.3f}   "
              f"meanJacc(valid)={np.mean(jv) if jv else float('nan'):.4f}   [A1: source edge diversity]")
        print(f"    {'reward':15s} {'Var(Rcov|valid)':>16s} {'S_cov':>10s} "
              f"{'η':>8s} {'%grp_cov':>9s} {'max|resid|':>11s}")
        for rname, Rf in [("cur σ(logΣall)", cur_R), ("B1  σ(logΣnew)", delta_R)]:
            vc, sc, eta, pg, mr = decompose(gkeys, Rf)
            print(f"    {rname:15s} {vc:16.6f} {sc:10.6f} {eta:8.4f} {pg:8.1f}% {mr:11.2e}")
        print(f"    B2 diag (valid pool): std(logΣall)={np.std(logall):.4f}  "
              f"std(logΣnew)={np.std(lognew) if lognew else float('nan'):.4f}   "
              f"(de-sat acts on logΣall; flat ⇒ B1/delta is the lever, not B2)")

    print("\nVar(Rcov|valid)=mean within-group variance of the valid-only coverage reward (the term "
          "in S_cov).\nS_cov=mean p_valid·(1-b)²·Var(Rcov|valid).  η=coverage's share of the GRPO group "
          "gradient.\n%grp_cov=groups with any within-valid coverage variance.  max|resid|=|Var_g(r)-"
          "(V_val+V_cov)| (≈0 ⇒ identity holds).")


if __name__ == "__main__":
    main()
