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
```txt

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
```

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

We now describe the policy update applied to the collected rollout dataset
$$
D_t = \{(x_i, y_i, \log \pi_{\theta_{\mathrm{old}}}(y_i \mid x_i), r_i)\}_{i=1}^{|D_t|},
$$


where each sample consists of a masked input $x_i$, a sampled infill $y_i$, the behaviour policy log-probability under the previous actor $\pi_{\theta_{\mathrm{old}}}$, and the scalar reward $r_i$ obtained after execution.

The goal of finetuning is to update the mutator policy $\pi_\theta(y \mid x)$ so that it assigns higher probability to infills that yield higher reward, while preventing excessively large policy updates. We consider two alternatives: Proximal Policy Optimization PPO and Group Relative Policy Optimization GRPO.

### PPO

PPO updates the actor using importance sampling with clipping. For each sample $i$, define the probability ratio
$$
r_t(\theta) = \frac{\pi_\theta(y_i \mid x_i)}{\pi_{\theta_{\mathrm{old}}}(y_i \mid x_i)}.
$$

Let $\hat{A}_i$ denote the advantage estimate for sample $i$. In PPO, this is typically obtained from a critic or value function $V_\psi$, for example through
$$
\hat{A}_i = r_i - V_\psi(x_i),
$$
or through a more general return-minus-baseline estimator if multi-step returns are used.

The clipped PPO objective is then
$$
J_{\mathrm{PPO}}(\theta)
=
\mathbb{E}_{(x_i,y_i)\sim D_t}
\left[
\min
\left(
r_i(\theta)\hat{A}_i,\,
\mathrm{clip}\!\left(r_i(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_i
\right)
\right].
$$

The clipping parameter $\epsilon > 0$ prevents the new policy from moving too far from the behaviour policy that generated the data. In practice, the actor is updated by maximizing $J_{\mathrm{PPO}}(\theta)$ over several minibatch epochs.

If KL regularization against a frozen reference model $\pi_{\mathrm{ref}}$ is used, we instead optimize
$$
J_{\mathrm{PPO\text{-}KL}}(\theta)
=
\mathbb{E}_{(x_i,y_i)\sim D_t}
\left[
\min
\left(
\rho_i(\theta)\hat{A}_i,\,
\mathrm{clip}\!\left(\rho_i(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_i
\right)
-
\beta\, D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x_i)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x_i)\right)
\right],
$$
where $\beta \ge 0$ controls the strength of the KL penalty.

Methodologically, PPO fits naturally when each rollout sample is treated independently. The grouping structure used during mutation is then ignored during optimization, and the policy is updated directly from the per-sample rewards and critic-derived advantages.

How advantage is calculated remains and outstanding issue?

### GRPO

GRPO removes the critic and instead computes a relative baseline from rewards within each group. This is particularly natural in our setting because one masked state $x$ is reused for multiple sampled infills inside each group.

Let a group $g$ contain $G$ sampled infills
$$
\{(x_g, y_{g,1}, r_{g,1}), \dots, (x_g, y_{g,G}, r_{g,G})\},
$$
where all samples in the group share the same masked input $x_g$ and differ only in the sampled infill.

For each sample in the group, define the policy ratio
$$
r_{g,j}(\theta) = \frac{\pi_\theta(y_{g,j} \mid x_g)}{\pi_{\theta_{\mathrm{old}}}(y_{g,j} \mid x_g)}.
$$

Instead of using a critic, GRPO computes a group-relative advantage. A simple form is
$$
\hat{A}_{g,j}
=
r_{g,j} - \frac{1}{G}\sum_{k=1}^{G} r_{g,k},
$$
and a normalized version is
$$
\hat{A}_{g,j}
=
\frac{r_{g,j} - \mathrm{mean}(r_g)}
{\mathrm{std}(r_g) + \delta},
$$
where $\delta > 0$ is a small constant for numerical stability.

The clipped GRPO objective is
$$
J_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}_{g}
\left[
\frac{1}{G}\sum_{j=1}^{G}
\min
\left(
r_{g,j}(\theta)\hat{A}_{g,j},\,
\mathrm{clip}\!\left(\rho_{g,j}(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_{g,j}
\right)
\right].
$$

If KL regularization is used, the objective becomes
$$
J_{\mathrm{GRPO\text{-}KL}}(\theta)
=
\mathbb{E}_{g}
\left[
\frac{1}{G}\sum_{j=1}^{G}
\left(
\min
\left(
\rho_{g,j}(\theta)\hat{A}_{g,j},\,
\mathrm{clip}\!\left(\rho_{g,j}(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_{g,j}
\right)
-
\beta\, D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x_g)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x_g)\right)
\right)
\right].
$$

Methodologically, GRPO is well aligned with our grouped mutation process. Each group corresponds to repeated infilling of the same masked state $x_g$, so the reward signal becomes comparative: samples are not judged only by their absolute reward, but by whether they outperform other candidate infills for the same state.

### KL
We follow in the steps of grpo (ref;grpo) and retain a reference policy $\pi_{\theta_{ref}}$ updated at a certain finetuning interval. We add on KL divergence 
$$
-
\beta\, D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)
$$

PPO and GRPO propose this as an additinal regularization method on policy gradients(ref;ppo,grpo), although the notion of $\pi_{ref}$ periodically updated is specifically taken from GRPO (ref;grpo)

### LoRA

To reduce the memory and optimization cost of policy finetuning, we parameterize the actor update using Low-Rank Adaptation LoRA rather than full dense finetuning. In LoRA, the pretrained weight matrices of the base model are frozen, and only a low-rank update is trained. For a pretrained weight matrix
$$
W_0 \in \mathbb{R}^{d \times k},
$$
LoRA replaces the full update
$$
W = W_0 + \Delta W
$$
with a low-rank factorization
$$
\Delta W = BA,
$$
where
$$
B \in \mathbb{R}^{d \times r}, \qquad
A \in \mathbb{R}^{r \times k}, \qquad
r \ll \min(d,k).
$$
Thus the adapted layer is
$$
W = W_0 + BA.
$$
Equivalently, for an input activation $h$, the modified forward pass is
$$
Wh = W_0 h + BAh.
$$
Only the adapter parameters $A$ and $B$ are trained, while the pretrained parameters in $W_0$ remain fixed. This is the central idea of LoRA and is motivated by the observation that downstream adaptation updates often lie in a low-rank subspace.(ref;LoRA)

Following the LoRA parameterization, the trainable policy parameters are no longer the full dense Transformer weights. Instead, if $\theta_0$ denotes the frozen pretrained policy parameters and $\phi$ denotes the collection of LoRA parameters across all adapted layers, then the policy is
$$
\pi_{\theta}(y \mid x)
\equiv
\pi_{\theta_0,\phi}(y \mid x),
$$
with
$$
\theta = \theta_0 + \Delta\theta(\phi),
$$
where $\Delta\theta(\phi)$ is induced only through the low-rank matrices $A$ and $B$. Hence, RL finetuning with PPO or GRPO updates only $\phi$, not the full parameter set $\theta_0$.

In the Transformer architecture, LoRA can in principle be applied to any dense projection matrix. In practice, it is most commonly attached to the self-attention projections, for example
$$
W_q,\; W_k,\; W_v,\; W_o,
$$
and often only a subset of these is adapted. The original LoRA paper studies this design explicitly and reports strong performance when adapting attention projections while freezing the rest of the network.(ref;LoRA)

Therefore, if an adapted attention projection is originally
$$
W_q \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}},
$$
we instead use
$$
W_q = W_{q,0} + B_q A_q,
$$
and analogously for other selected projections such as $W_v$. The Hugging Face PEFT implementation provides this mechanism directly by wrapping the chosen linear layers with LoRA modules.

The main benefit in our setting is that PPO or GRPO only needs to optimize the LoRA parameters
$$
\phi = \{A_\ell, B_\ell\}_{\ell \in \mathcal{L}},
$$
for the chosen set of adapted layers $\mathcal{L}$. Since
$$
|\phi| \ll |\theta_0|,
$$
this substantially reduces optimizer state, gradient memory, and checkpoint size during finetuning. 

The RL update is performed over LoRA parameters only:
$$
\phi_{t+1}
=
\arg\max_{\phi} J_{\mathrm{PPO}}(\theta_0,\phi)
\qquad \text{or} \qquad
\phi_{t+1}
=
\arg\max_{\phi} J_{\mathrm{GRPO}}(\theta_0,\phi),
$$
depending on the selected finetuning algorithm, while the pretrained base parameters $\theta_0$ remain frozen throughout.

LoRA may also preserves the original pretrained model as a fixed base policy, which is desirable when we want to maintain the model's general infilling ability while steering it toward higher-reward fuzzing mutations. (ref;LoRA)
For both PPO and GRPO we evaluate LoRA's effect on catastrophic forgetting by observing model infilling ability(in practice error rate $\bar{e_t}$). 

### Specific implementation details

`Trainer` owns the model. `Mutator` references the same model through the trainer via a @property. `Trainer` stores the model as `self.model`, manages the optimizer against it, and owns checkpointing. The model object persists across finetune cycles.

`Mutator`'s responsibility is mutation — tokenize, mask, infill, encode. It accesses the live actor via `self.trainer.model`, calls `trainer.ref_logprob()` for KL computation, and calls `maybe_finetune()` to trigger a training cycle when the rollout buffer is ready.

```py
class Mutator:
    def __init__(self, trainer, tokenizer, buffer, cfg):
        self.trainer   = trainer      # Trainer OWNS the model and reference snapshot
        self.tokenizer = tokenizer
        self.buffer    = buffer
        self.cfg       = cfg

    @property
    def model(self):
        return self.trainer.model     # read-only convenience accessor

    def infill(self, x_t):
        # generates y_t ~ pi_theta(. | x_t) using self.model.generate(...)
        pass

    def maybe_finetune(self):
        records = self.buffer.flush()         # drain completed rollouts
        self.trainer.set_rollout_dataset(records)
        self.trainer.train()                  # HF Trainer.train() — rebuilds optimizer each call; Adam momentum does not persist across finetune cycles
        self.trainer.snapshot_ref()           # re-anchor pi_ref after weights updated
```

`BaseTrainer` subclasses `Trainer`, owns the model by construction. LoRA is applied conditionally inside `__init__` based on `cfg`. The model presents interface to all callers, so `snapshot_ref`, `ref_logprob`, and `compute_loss` require no LoRA-specifics. `compute_loss()` is the standard HF override point — PPO and GRPO each override it with their respective clipped-ratio objectives:

```py
class BaseTrainer(Trainer):
    def __init__(self, model, args, data_collator, tokenizer, cfg, **kwargs):
        # LoRA wrapping is fully encapsulated here. After get_peft_model(), the model
        # presents an unchanged interface — callers never need to distinguish LoRA vs full.
        if cfg.lora_r > 0:
            lora_cfg = LoraConfig(
                r              = cfg.lora_r,
                lora_alpha     = cfg.lora_alpha,
                lora_dropout   = cfg.lora_dropout,
                target_modules = cfg.lora_target_modules_list,
                task_type      = TaskType.SEQ_2_SEQ_LM,
            )
            model = get_peft_model(model, lora_cfg)
        super().__init__(model=model, args=args, data_collator=data_collator, **kwargs)
        self.tokenizer  = tokenizer
        self.cfg        = cfg
        self._ref_model = None
        self.snapshot_ref()   # initialise reference to the pretrained weights before any training

    def snapshot_ref(self):
        if self._ref_model is None:
            self._ref_model = copy.deepcopy(self.model)   # full copy once at init
        elif self.cfg.lora_r > 0:
            _copy_lora_weights(self.model, self._ref_model)   # update adapter weights only; base stays from init copy
        else:
            self._ref_model = copy.deepcopy(self.model)

    def ref_logprob(self, x_t, y_t):
        return self.sequence_logprob(self._ref_model, x_t, y_t)

    def set_rollout_dataset(self, records):
        self.train_dataset = RolloutDataset(records)

    def sequence_logprob(self, model, x_t, y_t):
        pass   # shared utility: log pi_theta(y_t | x_t) under given model

    def kl_divergence(self, logprob, ref_logprob):
        pass   # KL penalty term shared by both PPO-KL and GRPO-KL

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raise NotImplementedError   # PPO and GRPO override this

def _copy_lora_weights(src, dst):
    # module-level helper; copies only lora_ keys from src into dst — O(|phi|) not O(|theta|)
    dst.load_state_dict({k: v for k, v in src.state_dict().items() if "lora_" in k}, strict=False)

class PPOTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pass   # clipped PPO objective + optional KL penalty

class GRPOTrainer(BaseTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pass   # group-relative advantage + clipped ratio + optional KL penalty
```

#### LoRA integration

LoRA wrapping happens once, inside `BaseTrainer.__init__`, immediately before handing `model` to HF `Trainer`. The `LoraConfig` is built entirely from `cfg` fields (`lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules_list`), so no LoRA configuration escapes the trainer boundary. When `cfg.lora_r == 0` the wrapping is skipped and full fine-tuning proceeds instead.

After `get_peft_model()`, the base weights are frozen and only the adapter matrices $\{A_\ell, B_\ell\}$ appear in the optimizer's parameter groups. The model still behaves as a standard PyTorch module — `generate()`, `forward()`, `state_dict()`, and `deepcopy` all work without modification. `ref_logprob()` is identical for both LoRA and full fine-tune: it always runs a forward pass through `_ref_model`. Only `snapshot_ref()` distinguishes the two cases: on the first call it does a full `deepcopy` regardless; on subsequent calls with LoRA active it calls `_copy_lora_weights()` which copies only the adapter keys into the existing `_ref_model`, leaving the frozen base weights untouched.

The full class inventory is:

```py
# rollout data pipeline
class RolloutBuffer:   pass   # accumulates (x_t, y_t, log_prob, reward) records per execution
class RolloutDataset:  pass   # torch Dataset wrapping flushed records; injected via set_rollout_dataset()
class RolloutCollator: pass   # pads and stacks records into batched tensors for Trainer.train()

# mutation
class Mutator:         pass   # references trainer.model; calls trainer.ref_logprob(); drives maybe_finetune()

# trainer hierarchy — Trainer owns model and reference snapshot; PPO/GRPO override compute_loss()
class BaseTrainer(Trainer):    pass
class PPOTrainer(BaseTrainer):  pass
class GRPOTrainer(BaseTrainer): pass
```

