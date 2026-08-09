# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
make all             # Core AFL++ binaries + LLVM/GCC instrumentation
make source-only     # Source-instrumented fuzzing tools only
make binary-only     # Binary-only tools (FRIDA, QEMU, Unicorn, etc.)
make distrib         # Everything

make clean           # Remove compiled files
make deepclean       # Remove compiled files AND downloads

make code-format     # Auto-format code (required before PRs)
```

Build flags:
- `DEBUG=1` — debug build with `-ggdb3 -O0 -Werror`
- `PERFORMANCE=1` — performance optimizations (not for macOS)
- `ASAN_BUILD=1` / `UBSAN_BUILD=1` — sanitizer builds
- `LLVM_CONFIG=llvm-config-VERSION` — override LLVM version

## Testing

```bash
make tests             # Full test suite (runs test/test-all.sh)
make unit              # Unit tests via cmocka
make performance-test  # Performance benchmarks
```

Individual test scripts in `test/`: `test-basic.sh`, `test-llvm.sh`, `test-qemu-mode.sh`, `test-frida-mode.sh`, `test-cmplog-*.sh`, etc.

## Code Formatting

All new/modified C files must be formatted before committing:
```bash
make code-format                    # Format all changed files
./.custom-format.py -i <file.c>    # Format a specific file
```

## Architecture Overview

AFL++ is a coverage-guided fuzzer built around a fork-server model. The key subsystems:

### Core Fuzzer (`src/`)

The fuzzer is split into focused modules, all sharing state via `afl_state_t` (defined in `include/afl-fuzz.h`):

| File | Role |
|------|------|
| `afl-fuzz.c` | Main entry point, event loop |
| `afl-fuzz-one.c` | Per-input mutation & execution (largest file, ~175KB) |
| `afl-fuzz-init.c` | Startup, config, corpus loading |
| `afl-fuzz-queue.c` | Seed queue management |
| `afl-fuzz-run.c` | Fork server execution, result classification |
| `afl-fuzz-redqueen.c` | Input-to-state correlation (REDQUEEN/QSYM) |
| `afl-fuzz-mutators.c` | Mutator scheduling (custom mutator API) |
| `afl-fuzz-stats.c` | Statistics, UI, plot file |
| `afl-fuzz-bitmap.c` | Coverage bitmap operations |
| `afl-fuzz-cmplog.c` | CmpLog instrumentation handling |
| `afl-cc.c` | Compiler wrapper (dispatches to LLVM/GCC modes) |
| `afl-forkserver.c` | Fork server protocol implementation |
| `afl-common.c` | Shared utilities |

### Instrumentation (`instrumentation/`)

Source-level LLVM passes and a GCC plugin. Key modes:
- **PCGuard** (`SanitizerCoveragePCGUARD.so.cc`) — default, lightweight edge coverage
- **LTO** (`SanitizerCoverageLTO.so.cc`) — link-time instrumentation, best coverage
- **CmpLog** (`afl-llvm-cmplog.cc`) — logs comparison operands for REDQUEEN
- **LAF-Intel** (`compare-transform-pass.so.cc`) — splits multi-byte comparisons
- **GCC plugin** (`afl-gcc-pass.so.cc`) — for targets that can't use Clang

The compiler wrapper `afl-cc` selects which pass to use based on env vars (`AFL_LLVM_INSTRUMENT`, `AFL_USE_ASAN`, etc.) and the invoked binary name (`afl-clang-fast`, `afl-clang-lto`, `afl-gcc-fast`).

### Binary-Only Modes

For targets without source code:
- **FRIDA mode** (`frida_mode/`) — dynamic instrumentation via Frida, supports persistent mode
- **QEMU mode** (`qemu_mode/`) — full QEMU user-space emulation with AFL patches
- **Unicorn mode** (`unicorn_mode/`) — CPU emulation via Unicorn engine
- **CoreSight mode** (`coresight_mode/`) — ARM64 hardware tracing (Linux only)
- **Nyx mode** (`nyx_mode/`) — snapshot-based fuzzing for speed

### Custom Mutators (`custom_mutators/`)

Pluggable mutation strategies implementing the AFL++ custom mutator C/Python API. Notable ones: `aflpp/` (built-in strategies), `symcc/`, `honggfuzz/`, `radamsa/`, `nautilus/` (grammar-based), `libfuzzer/`.

### Utilities (`utils/`)

30+ standalone tools including `libdislocator` (heap hardening), `libtokencap` (token extraction), `aflpp_driver` (libFuzzer-compatible harness driver), `afl_network_proxy` (network-based fuzzing), and `persistent_mode` examples.

## Key Headers

- `include/afl-fuzz.h` — The central `afl_state_t` struct; understanding this is essential for any fuzzer core work
- `include/afl-mutations.h` — All mutation primitives (large inline header, ~79KB)
- `include/config.h` — Tunable constants (map size, timeouts, queue thresholds)
- `include/types.h` — Core typedefs
- `include/forkserver.h` — Fork server wire protocol

## Branch Model

- `stable` — current release (4.35c)
- `dev` — active development (4.36a); **PRs must target `dev`**

## Platform Notes

- macOS: GCC plugin and `PERFORMANCE=1` are unavailable; LLVM mode works
- Linux: Full feature set; LTO mode requires gold linker or lld
- LLVM ≥14 required; ≥18 recommended for all features
