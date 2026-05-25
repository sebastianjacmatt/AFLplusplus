# rllm — Token-Level LLM Mutation Fuzzing
An AFL++ Python custom mutator that fuzzes JavaScript interpreters with
LLM-driven mask-fill mutation over a token-level queue. Implements the
CovRL-Fuzz data + model architecture (Eom et al., ISSTA '24) on stock
AFL++ with CodeT5+ 220M as the backbone.

Architecture and design rationale: [`docs/overview.md`](docs/overview.md),
[`docs/aligning_with_covrl.md`](docs/aligning_with_covrl.md).

---

## Repository layout

```
rllm/
├── data/
│   ├── preprocess.py   # JS → u16 (parallel uglify+dedup+tokenize)
│   ├── sample_seeds.py # u16 train corpus → 100 validity-filtered fuzz seeds
│   ├── validity.py     # stderr → valid/syntax/semantic classifier (shared)
│   ├── decode.py       # inspection helper: u16 binary → JS source
│   ├── masking.py      # span / single-token masking (fuzz + train)
│   ├── rewarding.py    # stub — reward subsystem (not yet implemented)
│   └── rollout.py      # stub — PPO loop (not yet implemented)
├── model/
│   ├── tokenizer.py    # HF tokenizer wrapper (parse_u16 / encode_u16)
│   └── llm.py          # CodeT5+ batched generation wrapper
├── configs/
│   ├── default.json    # standard fuzz config (contrastive, single-token mask)
│   ├── conservative.json
│   └── validity.json
├── docs/               # architecture and design docs
├── eval/               # evaluation harness (runner + plot)
├── rllm.py             # AFL++ Python mutator shim (hooks → Mutator)
├── mutator.py          # Mutator: hot-path fuzz / post_process / stats
├── config.py           # config dataclass
├── exit_hook.so        # LD_PRELOAD shim: redirects target stderr per run
├── exit_hook.c         # source for the above
├── Makefile
└── run_rllm.sh         # launch script
```

---

## Dependencies

```bash
# Python environment (conda recommended)
conda create -n rllm python=3.10
conda activate rllm
pip install torch transformers

# UglifyJS (required for seed preprocessing)
npm install -g uglify-js

# AFL++ (build from source or use a system install)
# JerryScript target binary at ~/Documents/data_store/engines/jerryscript/build/bin/jerry
```

The tokenizer and model weights (`Salesforce/codet5p-220m`) are
downloaded automatically by HuggingFace on first use.

---

## Preprocessing — two stages

The preprocessing pipeline produces two corpora:

| Corpus | Path (example) | Size | Used for |
|---|---|---|---|
| **Training** | `dataset-dec22-u16/` | ~50k u16 files | CovRL's 4:1 train mix during future PPO finetuning |
| **Fuzz seeds** | `dataset-dec22-u16-seeds/` | 100 u16 files | AFL's `-i` (validity-filtered, coverage-distinct via `afl-cmin`) |

Both must be prepared by the user **before** running `run_rllm.sh`. The
runner doesn't auto-bootstrap — preprocessing and corpus setup are user
responsibilities.

### Why two corpora?

The seed corpus drives AFL: each file must parse and execute cleanly on
the target (jerry), otherwise the queue starts already invalid and the
validity-collapse dynamic (`docs/queue_validity_collapse.md`) takes
over from minute zero. The training corpus is the wider distribution
PPO learns from; CovRL's reference impl tokenizes the full corpus
without validity filtering, so we do the same.

### Stage 1 — training corpus (`data/preprocess.py`)

Walks a JS directory in parallel:

1. `uglifyjs --mangle --beautify` per file — normalizes identifiers
   and whitespace, mirrors CovRL paper §4. Files that fail uglify are
   skipped.
2. Dedup by SHA-256 hash of the **uglified** bytes — catches test262
   templated cases that differ only in identifier names. On by default;
   pass `--no-dedup` to disable.
3. Tokenize with `Salesforce/codet5p-220m`'s BPE tokenizer.
4. Truncate to `--max-tokens` (default 1024 — CodeT5+'s pretraining
   source-sequence ceiling).
5. Write little-endian `uint16` binary to `<output>/<basename>`.

A sidecar marker `<output>.tokenizer.json` records tokenizer, vocab
size, dedup config, per-class counters (kept / skipped_uglify /
skipped_dup / truncated). The marker lives *alongside* the dir, not
inside it — AFL ingests every file under `-i`, so an in-dir marker
would feed garbage to `fuzz_count`.

```bash
cd custom_mutators/rllm

python -m data.preprocess \
    --input  ~/Documents/data_store/dataset/raw-dataset-dec22 \
    --output ~/Documents/data_store/dataset/dataset-dec22-u16 \
    --workers 16
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--workers` | `os.cpu_count()` | Worker processes. |
| `--no-dedup` | (off) | Disable uglified-bytes dedup. |
| `--tokenizer` | `Salesforce/codet5p-220m` | HF tokenizer name or local path. |
| `--max-tokens` | `1024` | Truncate sequences to this length. |

Cost on the 64k-file `raw-dataset-dec22`: ~6 min @ 16 workers
(~180 files/sec aggregate; uglify dominates).

### Stage 2 — fuzz seeds (`data/sample_seeds.py`)

Selects 100 validity-filtered, coverage-distinct seeds from the
training corpus. Pipeline per AFL++ docs `fuzzing_in_depth.md` §2:

1. **Validity filter** (parallel). Decode each u16 file → JS, run the
   target with a 5 s timeout, classify stderr via `data/validity.py`
   (`valid` / `syntax` / `semantic` / `timeout`). Keep only `valid`
   with exit code 0. *This step doubles as a harness filter:* test262
   cases needing `assert.js`, V8 cases needing `mjsunit.js`, Chakra
   cases needing `WScript`, etc., all fail with `ReferenceError` and
   are excluded automatically — see Limitations below.
2. **`afl-cmin`** on the validity-filtered set, dropping files that
   don't add new coverage. AFL++ docs call this "highly recommended".
3. **Reservoir-sample** `--num-seeds` (default 100) uniformly with
   `--seed` (default 42) for determinism.
4. Copy the chosen u16 files (by filename) from `--input` to `--output`.

```bash
cd custom_mutators/rllm

python -m data.sample_seeds \
    --input  ~/Documents/data_store/dataset/dataset-dec22-u16 \
    --output ~/Documents/data_store/dataset/dataset-dec22-u16-seeds \
    --target ~/Documents/data_store/engines/jerryscript/build/bin/jerry \
    --num-seeds 100 \
    --workers 16
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--num-seeds` | `100` | Seeds to sample. |
| `--workers` | `os.cpu_count()` | Validity-check worker count. |
| `--seed` | `42` | RNG seed for sampling. |
| `--timeout` | `5` | Per-file target-run timeout (seconds). |
| `--candidates` | (all) | Cap validity-check at the first N u16 files. |
| `--no-cmin` | (off) | Skip `afl-cmin`; sample directly from validity-filtered set. |

`afl-cmin` is taken from the bundled AFL++ tree at `../../afl-cmin`;
build it (`make` in the AFL++ root) before running stage 2 or the
script will error out with an explicit hint.

The sidecar marker `<output>.tokenizer.json` records:
`num_candidates`, `num_valid`, `num_after_cmin`, `num_chosen`,
per-class counters, and a per-source breakdown of the chosen 100.

### Prerequisites

Before running either stage:

- Conda env active (`conda activate rlm-grpo`) — provides
  `transformers`, `tokenizers`, `torch`.
- `LD_LIBRARY_PATH` must include `$CONDA_PREFIX/lib` for `afl-cmin`'s
  internal `afl-showmap` to find `libpython` (AFL++ is built against
  Python custom-mutator support):
  ```bash
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  ```
- AFL++ binaries built (`make` in the repository root).
- Target interpreter built with afl-clang (the same instrumented
  binary used for fuzzing).

### Limitations — harness inlining (future work)

Most of `raw-dataset-dec22` (especially the ~47k test262 cases) needs
engine-specific harness files to execute:

- **test262**: `assert.js`, `sta.js`, `propertyHelper.js`, etc. — see
  YAML frontmatter `includes:` declarations.
- **V8 mjsunit**: `assertEquals`, `assertTrue` from `mjsunit.js`.
- **ChakraCore**: `WScript.Echo` from a Windows-only host.
- **JSC**: various jsc test helpers.

The validity filter throws these away automatically. The result is a
seed corpus biased toward self-contained cases (jerry's own regression
tests, js-vuln-db CVE repros, and the subset of test262/v8/jsc/chakra
that doesn't use the harness — often "negative" syntax-error tests).

A minimal harness polyfill (≈50 helper functions defining `assert`,
`assert.sameValue`, `$ERROR`, `Test262Error`, `$262`, a stub
`WScript`, etc.) would reclaim several thousand cases. Deferred — see
`docs/aligning_with_covrl.md` §5 for the proposal.

---

## Running CovRL-Fuzz — `run_rllm.sh`

### Basic usage

```bash
cd custom_mutators/rllm
./run_rllm.sh -o <run_name>
```

Output lands in `~/Documents/data_store/out/<run_name>/`.

### Flags

| Flag | Default | Description |
|---|---|---|
| `-o <name>` | — | **Required.** Names the output dir. |
| `-i <dataset>` | `final-dataset-dec22` | Raw JS dataset dir name under `~/Documents/data_store/dataset/`. |
| `-c <config>` | `configs/default.json` | Mutator config file. Relative paths resolve from the `rllm/` dir; absolute paths are used as-is. |
| `-d` | off | Debug mode. See below. |

### What the script does

1. Checks for the preprocessed u16 seed dir (by looking for
   `<dataset>-u16.tokenizer.json`). If missing, runs `python -m data.preprocess`
   automatically.
2. Sets the AFL++ and rllm environment variables (see table below).
3. Invokes `afl-fuzz -i <u16-seeds> -o <out> -- jerry @@`.

### Environment variables set

| Variable | Value | Purpose |
|---|---|---|
| `AFL_PYTHON_MODULE` | `rllm` | Selects our Python mutator. |
| `AFL_CUSTOM_MUTATOR_ONLY` | `1` | Disables AFL's havoc / deterministic stages. |
| `AFL_DISABLE_TRIM` | `1` | Skips `trim_case` (would corrupt u16 buffers). |
| `AFL_NO_FASTRESUME` | `1` | Forces full queue re-ingestion on restart. |
| `AFL_POST_PROCESS_KEEP_ORIGINAL` | `1` | **Load-bearing.** Queue keeps u16 from `fuzz()`; only `.cur_input` gets decoded JS. Without this the queue invariant collapses. See [`docs/option_b_viability.md`](docs/option_b_viability.md). |
| `AFL_FRAMESHIFT_DISABLE` | `1` | Disables FrameShift byte-level transform (not u16-aware). |
| `AFL_PRELOAD` | `exit_hook.so` | LD_PRELOADs the stderr-capture shim into jerryscript. |
| `RLM_STDERR_FILE` | `<out>/rllm_stderr.txt` | Where `exit_hook.so` writes target stderr per run. |
| `RLLM_CONFIG` | `configs/default.json` | Selects the mutator config. |

### Debug mode (`-d`)

```bash
./run_rllm.sh -o my_debug_run -d
```

Adds:

| Variable | Effect |
|---|---|
| `AFL_DEBUG=1` | AFL internal debug logging. |
| `AFL_NO_UI=1` | Disables the ncurses UI; all output goes to stdout/stderr. |
| `AFL_DEBUG_CHILD=1` | Propagates target stderr to the terminal. |
| `PYTHONUNBUFFERED=1` | Flushes Python output immediately. |
| `PYTHONFAULTHANDLER=1` | Dumps Python tracebacks on signals (SIGSEGV etc). |

All output is tee'd to `<out>/debug.log` so the run is auditable after
the fact.

### Example runs

```bash
# Standard run
./run_rllm.sh -o run_01

# Custom config (nucleus sampling)
./run_rllm.sh -o run_nucleus -c configs/validity.json

# Debug mode (verbose, no TUI)
./run_rllm.sh -o run_debug -d

# Custom dataset
./run_rllm.sh -i my-js-corpus -o run_custom
```

---

## Viewing output

All output files live under `~/Documents/data_store/out/<run_name>/`.
AFL++ creates a `default/` subdirectory for the main fuzzer instance,
so most paths are `<out>/default/`.

### Live stats snapshot

```bash
cat ~/Documents/data_store/out/<run_name>/default/rllm_stats.txt
```

Single line, labels like:
```
t=142s seeds=28 muts=448 finds=3 valid=31.2% syntax=58.0% semantic=10.7% muts/s=3.2 gen_ms=4821
```

Updated every ~2 seconds. In debug mode (`AFL_NO_UI=1`) also written to
stderr in real time.

### Live stats history (TSV with pinned header)

```bash
( head -1 ~/Documents/data_store/out/<run_name>/default/rllm_history.tsv
  tail -f -n 0 ~/Documents/data_store/out/<run_name>/default/rllm_history.tsv \
) | column -t -s $'\t'
```

Append-only, one row per 2-second flush. Columns:

| Column | Meaning |
|---|---|
| `elapsed_s` | Seconds since mutator init. |
| `seeds` | `fuzz_count` calls (one per queue cycle per entry). |
| `muts` | Total LLM-generated mutations. |
| `finds` | Queue entries AFL accepted (new coverage). |
| `run_total` | Mutations executed and classified. |
| `run_valid` | Runs with empty stderr (valid JS). |
| `run_syntax` | Runs with a `SyntaxError`. |
| `run_semantic` | Runs with other JS errors (Reference/Type/Range/…). |
| `valid_pct` | Cumulative `run_valid / run_total × 100`. |
| `gen_ms_avg` | Avg ms per `fuzz_count` batch (model generation time). |
| `muts_per_s` | Overall mutation throughput. |

### Watching a mutation diff — `.cur_input.diff`

Every mutation writes two sidecar files:

- `.cur_input` — decoded JS source being passed to jerryscript right now.
- `.cur_input.diff` — same source, with **ANSI-yellow highlights** around
  token regions the LLM changed, and dim-red `[--]` markers where tokens
  were removed.

```bash
watch -c -n 2 cat ~/Documents/data_store/out/<run_name>/default/.cur_input.diff
```

The `-c` flag on `watch` interprets ANSI color codes. The diff is
computed in `post_process` via `difflib.SequenceMatcher` on the parent
vs mutation token sequences (~1–3 ms per mutation). Calibration and trim
replays leave the sidecar untouched — you only see fresh LLM mutations.

To view `.cur_input` as plain JS (no diff markup):

```bash
watch -n 2 cat ~/Documents/data_store/out/<run_name>/default/.cur_input
```

### Decoding queue / crash / hang files

Queue, crash, and hang files are all u16 binary. Use `data/decode.py`
to convert them to JS source:

```bash
cd custom_mutators/rllm

# Decode to stdout
python -m data.decode ~/Documents/data_store/out/<run_name>/default/queue/id:000001,*

# Decode to a file
python -m data.decode ~/Documents/data_store/out/<run_name>/default/crashes/id:000000,* \
    -o /tmp/crash.js
```

The decoded output is the same JS source AFL wrote to `.cur_input` when
that input was first executed.

### Standard AFL output files

| File | Description |
|---|---|
| `default/fuzzer_stats` | Key:value run stats (execs, speed, coverage). Read with `cat`. |
| `default/plot_data` | CSV time series (execs/sec, paths found, coverage). |
| `default/queue/` | u16 binary queue entries. Decode with `data/decode.py`. |
| `default/crashes/` | u16 binary crash-triggering inputs. |
| `default/hangs/` | u16 binary timeout-triggering inputs. |
| `rllm_stderr.txt` | Last run's target stderr (overwritten each execution). |
| `debug.log` | Full debug output (only in `-d` mode). |

---

## Evaluation harness — `eval/`

For structured benchmark runs (duration-limited, multi-run variance,
CovRL Table 7 comparison), see [`eval/README.md`](eval/README.md).

Quick smoke test:

```bash
cd custom_mutators/rllm
python -m eval.runner --name eval_smoke --duration 300
python -m eval.plot \
    --out-dir ~/Documents/data_store/out/eval_smoke/default \
    --output  ~/Documents/data_store/out/eval_smoke/default/eval_plot.png
```

---

## Configuration

Three presets live in `configs/`:

| File | Sampling | Masking | Use |
|---|---|---|---|
| `default.json` | contrastive (`penalty_alpha=0.6`, `top_k=4`) | single-token overwrite (`corruption_rate=0.03`, `min=max=mean_span=1`) | Standard run. Matches CovRL's `RANDOM_OVERWRITE`. |
| `conservative.json` | contrastive | single-token overwrite | Same as default (historical preset, validity filter removed). |
| `validity.json` | contrastive | single-token overwrite | Same masking; use with custom sampling kwargs. |

Pass a config with `-c`:

```bash
./run_rllm.sh -o my_run -c configs/default.json
```

Key config fields:

| Field | Default | Description |
|---|---|---|
| `fuzz_count` | `16` | Mutations generated per queue entry per cycle. |
| `sampling_method` | `contrastive` | `contrastive` or `nucleus`. |
| `max_new_tokens` | `64` | Max tokens CodeT5+ generates per masked span. |
| `corruption_rate` | `0.03` | Fraction of tokens to mask (~1–3 per 100-token program). |
| `min/max/mean_span_length` | `1/1/1.0` | Span shape. All-1 = CovRL single-token overwrite. |
| `model_name_or_path` | `Salesforce/codet5p-220m` | HF model name or local checkpoint path. |

---

## Further reading

- [`docs/overview.md`](docs/overview.md) — data flow, queue invariant, AFL hooks.
- [`docs/aligning_with_covrl.md`](docs/aligning_with_covrl.md) — why u16 queue, masking shape, phased plan.
- [`docs/option_b_viability.md`](docs/option_b_viability.md) — how `AFL_POST_PROCESS_KEEP_ORIGINAL` makes the u16 queue work on stock AFL++.
- [`docs/queue_validity_collapse.md`](docs/queue_validity_collapse.md) — why validity drifts over a run (expected behavior, not a regression).
- [`eval/README.md`](eval/README.md) — evaluation harness, plot interpretation, CovRL Table 7 comparison.
