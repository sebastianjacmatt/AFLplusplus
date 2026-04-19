# design document

The following is a design document for the reinforcement learning language models for fuzzing (ref;rlm.py). It implements two different policy gradient methods: *Proximal Policy Optimization*(ref;PPO) and *Group Relative Policy Optimization*(ref;GRPO) along with *Low-Rank Adapters*(ref;LoRA) as a regularization mitigation to retain model infilling ability.

There are some challanges of AFL++ in regards to reinforcement learning:

We regard our instrumented binary target $T$ as static, whilst our state $O_t$ is an observation at timestep $t$. The dynamics of the problem can be described as $P(o_{t+1}|o_t,a_t)$ changes with $t$ i.e. the dynamics of the observed rl problem is non stationary.

1. The fuzzing context $o_t$ is non stationary because the queue, seed selection and accumulated coverage evolve as new interesting executions are discovered. 

2. The evolving queue $Z$ only adds coverage increasing or *interesting*(ref;afl++) programs $z'$ this in turn biases the $Z$ to only retain positive rewarding programs. 
$$
Z_{t+1} = \begin{cases} Z_t \cup \{z_t'\}, & \text{if } R(z_t', o_t)>0 \\ Z_t, & \text{otherwise.} \end{cases}
$$

Our goal is to implement a new fuzzer mitigating the second claim where all state-action pairs and record their reward $\pi_{\theta}(y|x)$ during fuzzing.
This gives reward that is an unbiased dataset $D_T$, this essentially allows for online and offline fuzzing implementations.

## Metrics
We are interested in the following metrics:
- unique bugs found (Best is deduplicated unique bugs)
    - de-duplicated bug count
- coverage gain (best proxy since bugs are too sparse)
    - tf-idf weighted coverage reward
- Validity rate or error rate (If validity collapses, coverage and bug signals become harder to trust)
    - course grained exit codes 0 or 1
    - semantic and syntactic validity from executed child process
- Crash quality signals (These are weaker than confirmed unique bugs, but stronger than pure training metrics.)
    - Timeouts
    - sanitizer crashes
    - abnormal exits
- Coverage efficiency (This matters because a slower method can look good in absolute terms but be poor operationally.)
    - Coverage gained per unit time
    - per execution
    - or per finetune cycle
- Reward statistics (These help explain behavior, but they are not end goals)
    - Mean reward
    - reward variance
    - fraction of zero reward samples 
    - retention rate
- Training stability diagnostics (These are useful for debugging PPO or GRPO behavior, but they do not validate the fuzzer by themselves.)
    - Actor loss
    - critic loss
    - KL, clip fraction
    - gradient norm.

## RLM Mutator

What follows is detailed description of the methodology of our fuzzing approach we first begin by contextualizing our problem.

## Notation
We denote the binary target as a static instrumented binary $T$, and our fuzzing engine as $O_t$, where $t$ indexes the timestep within a fuzzing campaign. Throughout a campaign, $O_t$ maintains a queue $Z_t$ of programs waiting to be mutated. At each timestep, a seed is drawn from $Z_t$, mutated, and executed on $T$; resulting coverage $b_t$ updates global bitmap $B_t$.

The AFL++ API exposes methods that we use to replace AFL++'s native mutation strategy with our own: `fuzz_count()` and `fuzz()`. We will use the former to prepare a seed for a predetermined number $N$ of `fuzz()` mutations, effectivly N becomes our batch size.

## Mutation
Our general mutation algorithm in `fuzz_count` and `fuzz()` then proceeds as follows:

first we take program $z_t \in Z_t$ and decode it from a `bytearray` into a plain `.js` program called $p_t$, we then tokenize this program into a sequence of token id's $w_t=tokenize(p_t)$. A tokenized program $w_t=\{id_1,id_2,id_3,...,id_n\}$ we then apply masks randomly to the sequence of tokens giving us $x_t=mask(w_t)$. We then sample a corresponding sequence of tokens from our policy to infill $x$, we denote this sequence as $y_t$ and the result is a mutated tokenizsed sequence  $w'_t=\pi_\theta (y_t|x_t)$. We then decode the tokenized sequence with the same tokenizer, and encode it again into a `bytearray`. $z'_t=encode(p'_t=de-tokenize(w'_t))$.

The final mutated program represented as a `bytearray` is then executed on the target $T$ and the reward is gathered from the fuzzing engine $r_t=R(z'_t,o_t)$. The reward depends on the novelty of previously recorded coverage, not from $T$ alone.

### Mutation algorithm

The mutated program $z'_t$ is added to the queue $Z_{t+1}$ if it produced new coverage.
Our mutation process utilizes the queue of programs $Z_t$ by extracting the top most $z_t$ and decoding the 
We denote our mutation fuzzing approach in terms of AFL as the following


Function queue_get():
    seed_count += 1
    if seed_count > 0 and seed_count % finetune_interval == 0:
        finetune_pending = True        # deferred — model updated before next seed

Function fuzz_count(buf):
    global w_t, group_id, sample_id

    if finetune_pending:
        finetune(df)                   # weights fresh before tokenising new seed

    p_t = decode(buf)
    w_t = tokenize(p_t)

    group_id = 0
    sample_id = 0

    return num_groups * group_size

Function fuzz():
    global x_t, group_id, sample_id

    if sample_id % group_size == 0:
        x_t = mask(w_t)
        group_id = group_id + 1

    y_t  ~ pi_theta( . | x_t)
    w_t' = merge(x_t, y_t)
    p_t' = de_tokenize(w_t')
    z_t' = encode(p_t')

    df.append(sample_id, group_id, x_t, y_t, log_pi_theta(y_t | x_t), reward=None)

    sample_id = sample_id + 1
    return z_t'

Function post_run():
    e_t  = read_exit_code()            # written by exit_hook.so via LD_PRELOAD
    b_t  = read_shm_bitmap()          # snapshot of AFL++ trace_bits SHM
    r_t  = -1.0 if e_t != 0 else R_cov(b_t)
    df.append(r_t)


after the mutation process we execute the now mutated program $z'_t$ on the target binary $T$ under the fuzzing context $o_t$ at timestep $t$ and record coverage rewards $r_t=R(z'_t,o_t)$. The reward of $r_t$ is not the same as $r_{t+1}$ as the global bitmap $b_t$ continually update each \texttt{fuzz()}


## Rewarding

The reward $r_t = R(z_t', o_t)$ is a scalar computed in `post_run()` after each target execution. It is composed of two signals: a TF-IDF weighted coverage reward derived from AFL++'s shared memory bitmap, and a coarse validity penalty derived from the child process exit code.

#### Coverage signal — reading AFL++ shared memory

AFL++ maintains a coverage bitmap of size $M$ bytes in a POSIX shared memory segment identified by the environment variable `__AFL_SHM_ID`. Each byte $b_{t,j}$ counts how many times edge $j$ was hit during the most recent execution; the byte saturates at 255. The bitmap is zeroed by AFL++ immediately before each fork and populated by the instrumented target during execution.

AFL++'s Python mutator API calls `post_run()` synchronously after `waitpid()` confirms the child has exited, and before the bitmap is cleared for the next run. We attach to the SHM segment once at `init()` time using `shmat()` via ctypes, obtaining a live numpy `uint8` view. Inside `post_run()` we snapshot it with a single `memcpy` (`numpy.ndarray.copy()`), which completes in under 50 µs for $M = 65536$ or $M = 131072$, well within the window between child exit and the next execution.

#### TF-IDF coverage reward

We discard hit-count granularity and work with binary presence. The unique-coverage vector for execution $t$ is

$$
\mathrm{TF}^{\mathrm{cov}}_{t,j} = \mathbf{1}[b_{t,j} > 0], \quad j = 1,\dots,M.
$$

We maintain an online inverse-document-frequency weight vector $W_t \in \mathbb{R}^M$. At timestep $t$, the weighted coverage score is

$$
S_t = \sum_{j=1}^{M} \mathrm{TF}^{\mathrm{cov}}_{t,j}\, W_{t-1,j}.
$$

Edges that have been hit often carry low weight; rarely-hit edges carry high weight. The coverage reward is then passed through a log-sigmoid to produce a value in $(0.5, 1]$:

$$
R_{\mathrm{cov}}(z_t', o_t) =
\begin{cases}
\sigma\!\left(\log S_t\right), & S_t > 0,\\
0.5, & S_t = 0,
\end{cases}
\qquad \sigma(x) = \frac{1}{1+e^{-x}}.
$$

The floor of 0.5 is assigned when the execution hits no previously weighted edges (e.g. near-zero coverage runs). After computing $R_{\mathrm{cov}}$, the IDF vector is updated with exponential momentum so that frequently observed edges are progressively down-weighted:

$$
W_t = \alpha\, W_{t-1} + (1-\alpha)\,\widetilde{W}_t,
\qquad
\widetilde{W}_{t,j} = \frac{1}{\sqrt{M}}\log\!\frac{N_t}{1 + \mathrm{TF}^{\mathrm{cov}}_{t,j}},
$$

where $N_t$ is the total number of executions seen so far and $\alpha \in [0,1]$ is a momentum hyperparameter (default 0.6). The IDF update is applied unconditionally after every execution, including error runs, so that $W_t$ reflects the full coverage distribution and not only valid executions.

#### Validity signal — obtaining the exit code

AFL++'s Python `post_run()` hook receives zero arguments. Inspecting `src/afl-fuzz-python.c` confirms this: the binding calls `PyTuple_New(0)` and passes no run-result to Python. AFL++'s forkserver captures the child's `waitpid()` status in `fsrv->child_status` and classifies it as `FSRV_RUN_OK`, `FSRV_RUN_CRASH`, or `FSRV_RUN_TMOUT`, but none of this is forwarded to the Python layer.

We recover the raw exit code via a small LD_PRELOAD library (`exit_hook.so`) that overrides `exit()` in every forked child. Before delegating to `_exit()`, the hook writes the exit code as a decimal integer to the path stored in `RLM_EXIT_FILE`:

```c
void exit(int code) {
    const char *path = getenv("RLM_EXIT_FILE");
    if (path) { FILE *f = fopen(path, "w"); fprintf(f, "%d\n", code); fclose(f); }
    _exit(code);
}
```

`RLM_EXIT_FILE` is set to `/tmp/rlm_exit_<PID>` in `rlm.py::init()`. Because `setup_custom_mutators` executes at line 2359 of `src/afl-fuzz.c` and `afl_fsrv_start` at line 2780, the environment variable is set before the forkserver starts and is therefore inherited by every child. `post_run()` reads the file to obtain $e_t$; since AFL++ is single-threaded the child has fully exited before `post_run()` runs, so there is no race. The write and read together cost roughly 10–20 µs on tmpfs — negligible for interpreter-class targets.

Only `exit()` is intercepted. Targets killed by a signal (SIGSEGV, SIGABRT) do not call `exit()` and leave no file; `post_run()` treats a missing file identically to a non-zero exit code and assigns $r_t = -1.0$. Rewarding crashes with coverage signal would incentivise the policy to target parser edge cases and crash-inducing paths rather than semantically valid, broadly covering inputs — exactly the bias we want to avoid.

#### Combined reward

Coverage reward is granted only on clean exit ($e_t = 0$). Every other outcome — non-zero exit code, signal-based crash, or any other abnormal termination — receives a fixed penalty:

$$
r_t = R(z_t', o_t) =
\begin{cases}
-1.0, & e_t \neq 0 \text{ or exit code absent (signal crash)},\\[4pt]
\sigma\!\left(\log S_t\right), & e_t = 0 \text{ and } S_t > 0,\\[4pt]
0.5, & e_t = 0 \text{ and } S_t = 0,
\end{cases}
$$

giving us a final scalar reward $r_t$ appended to the dataframe after each execution.


## Finetuning

Finetuning is initiated by the queue_