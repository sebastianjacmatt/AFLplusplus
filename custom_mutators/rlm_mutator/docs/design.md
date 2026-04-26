# design

The following is a design document for reinforcement learning langauge model mutator for fuzzing (ref;rlm.py). It implements *Group Relative Policy Optimization*(ref;GRPO) along with *Low-Rank Adapters*(ref;LoRA).

## MAB
The problem of an afl++ mutators(ref;docs/custom_mutator.md) can be formulated as a Multi Arm Bandit problem of mutations done on a program. The program outputs rewards, the objective of the agent is therefore to maximize reward for an action taken.(ref;MAB objective formula for maximizing reward. use \pi to denote policy)

## Metrics
we are interested in the following metrics
- coverage
- error rate

## Mutator 
The reinforcement learning langauge model(rllm) mutator works by takning a program $z$ representing it as a set of tokens $w$ and masking a given span(ref;codet5) of those tokens. The bandit then infills this span with a given policy $\pi_{\theta}(y|x)$ where $y$ is some infilled sequence whilst $x$ is some masked program. rllm then encodes the program back into byte representation $z'$ and executes on the target.

## Reward

After program $z'$ is executed on target, coverage is recorded as weighted by tf-idf(ref;covrl) we also record the validity information of an executed program $z'$

The reward function is a combination of both validity and coverage on the executed program
$$
R(z') =
\begin{cases}
validity \\
tf-idf(coverage)
\end{cases}
$$

### Validity signal: exit-code-as-bug-oracle

The `validity` component of $R(z')$ is derived from the engine subprocess's
exit code. This is not an arbitrary choice — it is the established
cross-fuzzer convention, and we anchor our methodology in it explicitly.

#### In-repo precedent

There are three places in the AFL++ repository that consume exit codes as a
bug oracle, all using the same convention.

**1. AFL++ itself — the canonical site.** The actual oracle that decides
"this run is a crash" is a 4-way disjunction in
[src/afl-forkserver.c:2391-2415](../../../src/afl-forkserver.c):

```c
/* Did we crash?
 In a normal case, (abort) WIFSIGNALED(child_status) will be set.
 MSAN & LSAN in uses_asan mode use special exit codes as they doesn't support
 abort_on_error. On top, a user may specify a custom AFL_CRASH_EXITCODE.
 Handle all four cases here. */

if (unlikely(
        (WIFSIGNALED(fsrv->child_status) ...) ||
        ((fsrv->uses_asan & 4) && WEXITSTATUS(...) == MSAN_ERROR) ||
        ((fsrv->uses_asan & 2) && WEXITSTATUS(...) == LSAN_ERROR) ||
        (fsrv->uses_crash_exitcode && WEXITSTATUS(...) == fsrv->crash_exitcode)))
```

The two "magic" sanitizer exit codes are hard-coded constants in
[include/config.h:460-466](../../../include/config.h):

```c
#define MSAN_ERROR 86
#define LSAN_ERROR 23
```

User-facing description of `AFL_CRASH_EXITCODE` is at
[docs/env_variables.md:405-409](../../../docs/env_variables.md); the
rationale for MSAN=86 / LSAN=23 is at
[docs/fuzzing_in_depth.md:254-258](../../../docs/fuzzing_in_depth.md) and
[docs/env_variables.md:1001-1022](../../../docs/env_variables.md). The same
recogniser is duplicated in `afl-tmin`, `afl-showmap` and `afl-fuzz` (see
[src/afl-tmin.c:1465](../../../src/afl-tmin.c),
[src/afl-showmap.c:1710](../../../src/afl-showmap.c),
[src/afl-fuzz.c:387](../../../src/afl-fuzz.c)) — i.e. the convention is
repo-wide, not just `afl-fuzz`. The introducing changelog entry is at
[docs/Changelog.md:944](../../../docs/Changelog.md)
(`added AFL_CRASH_EXITCODE env variable to treat a child exitcode as crash`)
— that line is the primary-source citation for *when and why* the knob
exists.

**2. libFuzzer (vendored inside AFL++) — second fuzzer using
exit-code-as-oracle.** This is the cleanest "tied to a paper" example. AFL++
ships the libFuzzer source under
[custom_mutators/libfuzzer/](../../libfuzzer/), and libFuzzer publishes its
exit-code convention in
[FuzzerFlags.def:52-55](../../libfuzzer/FuzzerFlags.def):

```c
FUZZER_FLAG_INT(error_exitcode, 77, "When libFuzzer itself reports a bug "
  "this exit code will be used.")
FUZZER_FLAG_INT(timeout_exitcode, 70, "When libFuzzer reports a timeout "
  "this exit code will be used.")
```

Wired into the driver at
[FuzzerDriver.cpp:843-844](../../libfuzzer/FuzzerDriver.cpp). These
constants (77 for an error, 70 for a timeout) are the de-facto convention
that LLVM `compiler-rt` sanitizers (`UBSAN_OPTIONS=halt_on_error=1:exitcode=...`,
`ASAN_OPTIONS=...`) inherit. The primary source we cite for them is the
LLVM libFuzzer manual (https://llvm.org/docs/LibFuzzer.html) and the
SecDev'16 paper:

> Kostya Serebryany. *Continuous Fuzzing with libFuzzer and AddressSanitizer.*
> IEEE Cybersecurity Development (SecDev), 2016.

**3. afl-network-server — repo-internal reinforcement.**
[utils/afl_network_proxy/afl-network-server.c:220-303](../../../utils/afl_network_proxy/afl-network-server.c)
refuses to start unless `MSAN_OPTIONS` contains `exit_code=86`. Useful as
in-tree evidence the convention is *enforced*, not merely accepted.

#### Citation stack

We cite the validity signal in three layers:

| Claim | Citation |
|---|---|
| "Non-zero exit = error" is the C/UNIX convention | GNU C Library Manual, *Process Completion Status* / *Exit Status* (https://www.gnu.org/software/libc/manual/html_node/Exit-Status.html); IEEE Std 1003.1-2017 (POSIX), `wait(2)` — defines `WIFEXITED`, `WIFSIGNALED`, `WEXITSTATUS`, `WTERMSIG`. https://sourceware.org/glibc/manual/2.43/pdf/libc.pdf https://sourceware.org/glibc/manual/latest/html_mono/libc.html#Exit-Status-1 |
| Bug oracle = function on the run outcome | Manès et al., *The Art, Science, and Engineering of Fuzzing: A Survey*, IEEE TSE 2019. |
| Sanitizer exit-code constants (86/23) | Serebryany, Bruening, Potapenko, Vyukov, *AddressSanitizer: A Fast Address Sanity Checker*, USENIX ATC 2012; Stepanov & Serebryany, *MemorySanitizer: fast detector of uninitialized memory use in C++*, CGO 2015. The constants themselves are documented at [docs/env_variables.md:1001-1022](../../../docs/env_variables.md) with `exit_code=86 (required for legacy reasons)`. |
| Exit-code-as-oracle in libFuzzer | Serebryany, *Continuous Fuzzing with libFuzzer and AddressSanitizer*, IEEE SecDev 2016 + LLVM libFuzzer docs; constants in [FuzzerFlags.def:52-55](../../libfuzzer/FuzzerFlags.def). |
| AFL++'s configurable bug oracle | Fioraldi, Maier, Eißfeldt, Heuse, *AFL++: Combining Incremental Steps of Fuzzing Research*, USENIX WOOT 2020. Implementation at [src/afl-forkserver.c:2391-2415](../../../src/afl-forkserver.c); user-facing semantics at [docs/env_variables.md:405-409](../../../docs/env_variables.md); historical introduction at [docs/Changelog.md:944](../../../docs/Changelog.md). |

#### Argument

A coverage-guided fuzzer's bug oracle is a function on the wait-status of
the target subprocess (POSIX 1003.1). AFL++ defines this function as
$\mathrm{WIFSIGNALED}(\mathrm{status}) \lor \mathrm{WEXITSTATUS}(\mathrm{status}) \in \{\mathrm{MSAN\_ERROR}=86,\ \mathrm{LSAN\_ERROR}=23,\ \mathrm{AFL\_CRASH\_EXITCODE}\}$
([afl-forkserver.c:2391-2415](../../../src/afl-forkserver.c); Fioraldi et
al., WOOT'20). Two of the constants are LLVM compiler-rt sanitizer
conventions inherited from the AddressSanitizer family (Serebryany et al.,
ATC'12; Stepanov & Serebryany, CGO'15); the third is user-configurable,
which is exactly the hook that lets a fuzzer treat a higher-level engine
error (parser failure, assertion, type error) as a "bug". The same
exit-code-as-oracle convention is used by libFuzzer with
`error_exitcode=77` / `timeout_exitcode=70` (Serebryany, SecDev'16;
[FuzzerFlags.def:52-55](../../libfuzzer/FuzzerFlags.def)).

Our methodology therefore stands on an established cross-fuzzer convention:
we extend it, via [`exit_hook.so`](../exit_hook.c), to recover the *full*
exit-code value (not the AFL-collapsed crash bit), so that the JavaScript
engine's parser/semantic/runtime exit codes can be used as a learning
signal in addition to the crash oracle. Concretely, $R(z')$ assigns
$-1.0$ for any non-zero exit code (a syntax/semantic/runtime failure or a
sanitizer/AFL-recognised crash) and $\mathrm{tf\text{-}idf}(\mathrm{coverage})$
for $z'$ with exit code $0$.

--todo; we should initialize exit code to recive neutral rewards on specific exit codes denoted by sanatizers-- 

## Finetuning

During fuzzing we collect a rollout dataset containing $D=\{ x{_t},y{_t},\pi_{\theta}(y|x),z', \pi_{\theta_{ref}}(y|x),r \}$ along with identifiers for which group a sample belongs to.

### GRPO 
GRPO is implemented as a `Trainer` with a custom batchloader(ref;rollout.py:GroupedBatchLoader) for grouped batches.
The trainers loss is calculated using the applied formula from GRPO(ref;grpo) 

$$
\begin{align*}
J_{\mathrm{GRPO}}(\theta)
= \mathbb{E}_{x \sim Y(X),\,\{y_i\} \sim \pi_{\theta_{\mathrm{old}}}}
\Bigg[
  \frac{1}{G}\sum_{i=1}^{G}
  \Bigg(
  &\min\!\left(
    \frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)} A_i,\;
    \operatorname{clip}\!\!\left(
      \frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i\mid x)},
      1-\varepsilon,1+\varepsilon
    \right)A_i
  \right) \\
  &- \beta\, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
  \Bigg)
\Bigg],
\end{align*}
$$

advantage $A_i$:
$$
A_i
= \frac{r_i - \operatorname{mean}\!\left(\{r_1,\ldots,r_G\}\right)}
       {\operatorname{std}\!\left(\{r_1,\ldots,r_G\}\right)}.
$$

policy $\pi_{\mathrm{ref}}$:

$$
D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
= \frac{\pi_{\mathrm{ref}}(y_i \mid x)}{\pi_\theta(y_i \mid x)}
  - \log\frac{\pi_{\mathrm{ref}}(y_i \mid x)}{\pi_\theta(y_i \mid x)} - 1.
$$

