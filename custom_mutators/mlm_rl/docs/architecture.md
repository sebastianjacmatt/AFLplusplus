# mlm_rl Mutator Architecture

## What this is

`mlm_rl` is an AFL++ Python custom mutator that implements the CovRL staged fuzzing scheme.
Instead of AFL++'s byte-level mutations, it applies token-level mask-and-infill mutations
using a pre-trained sequence-to-sequence language model (CodeT5+). Over time the model is
finetuned on coverage-increasing inputs found by AFL++, steering it toward producing inputs
that trigger new coverage.

---

## AFL++ hook lifecycle

AFL++ drives the mutator through five hooks. In order of call sequence per fuzzing cycle:

```
init()
  └─ load tokenizer + actor model
  └─ create PPOTrainer

queue_get(filename)                  ← called once per seed selection
  └─ set_queue_dir(os.path.dirname(filename))   ← data_utils learns the queue path
  └─ increment _queue_get_count
  └─ set _finetune_pending = True every FINETUNE_INTERVAL seeds

fuzz_count(buf)                      ← called once per seed selection, after queue_get
  └─ if _finetune_pending → _finetune()
  └─ tokenize buf → _current_seed_token_ids
  └─ return FUZZ_COUNT (number of fuzz() calls to schedule)

fuzz(buf, add_buf, max_size)         ← called FUZZ_COUNT times per seed
  └─ copy _current_seed_token_ids
  └─ apply random mask mutation (_random_mask)
  └─ run actor model inference (_actor_infill)
  └─ decode back to bytes
  └─ return mutated bytearray
```

`queue_new_entry` is not used. AFL++ notifies via the queue directory itself, which is read
directly at finetune time.

---

## Queue as the sole data source

Every file in the AFL++ queue increased coverage — that is AFL++'s invariant for adding
anything to the queue. The filenames encode whether an entry is an initial seed or a mutation:

```
id:000000,orig:seedname          ← initial seed copied from -i at startup
id:000001,src:000000,op:havoc    ← coverage-increasing mutation
```

`data_utils` reads the queue directory directly. The directory is set once via
`set_queue_dir()`, called from `queue_get` on every iteration (idempotent — same path each
time). The loaders have no parameters:

```
load_queue_files()     → all entries, classified by filename
load_orig_corpus()     → is_orig == True  (initial seeds, anti-forgetting reference)
load_mutation_corpus() → is_orig == False (coverage-increasing mutations, training signal)
```

There is no separate corpus directory, no `known_ids` deduplication, and no incremental
accumulation in `data_utils`. The queue is scanned fresh on each finetune call.

---

## CovRL finetune cycle

`_finetune()` in `mlm_rl.py` is called from `fuzz_count()` when `_finetune_pending` is True.
It delegates entirely to `PPOTrainer.finetune()` then hot-swaps the global ACTOR.

### PPOTrainer.finetune()

```
finetune()
  └─ _prepare_data()
        ├─ load_mutation_corpus()  → mutations DataFrame
        ├─ compute rewards (Rewarder, or 0.0 placeholder)
        ├─ load_orig_corpus()      → orig DataFrame
        └─ mix: mutations + sample(orig, 4× mutations) → mixed_df

  └─ _make_critic_dataset(mixed_df) → CriticDataset
  └─ _train_critic(critic_dataset)

  └─ if cycle > 0:
        └─ _snapshot_actor()           ← freeze π_{t-1}
        └─ _make_actor_dataset(mixed_df) → ActorDataset
        └─ _finetune_actor_with_ppo_like_loss(actor, critic, previous_actor)

  └─ _finetune_cycle_index += 1
```

### Cycle 0 vs cycle N

- **Cycle 0**: critic warmup only. The critic has not yet produced meaningful value estimates,
  so the actor is not updated.
- **Cycle N > 0**: critic is updated first, then the actor is updated using the PPO-like
  objective (Eq. 7 + Eq. 8 from the CovRL paper) with the just-trained critic as the baseline.

### Mixed dataset (4:1 ratio)

Each training cycle builds one mixed DataFrame consumed by both critic and actor training,
mirroring CovRL's `FineTuner.preprocess()`:

```
mixed = mutations + sample(orig, n=min(4 × len(mutations), len(orig)))
```

Orig entries receive `reward = 0.0`. For the actor this is immaterial — `ActorTrainer`
derives `r(W*)` dynamically from the frozen critic. For the critic, orig entries contribute
CE anti-forgetting signal, training toward `score_to_label(0.0)`.

### Rewards

`Rewarder` runs each mutation through `afl-showmap` to obtain a coverage bitmap, then scores
it using TF-IDF over the full accumulated bitmap history. This is the coverage-novelty reward
signal. When `Rewarder` is not configured (no `afl_showmap_path` / `interpreter_path`),
all mutations receive `reward = 0.0` as a placeholder.

---

## Actor and Critic roles

**Actor** (`AutoModelForSeq2SeqLM`, CodeT5+): the mutation model. Given a token sequence with
mask sentinels inserted, it predicts replacement tokens for each masked span. The actor is
what AFL++ calls during `fuzz()` to generate each mutated input.

**Critic** (`CriticModel`): a value estimator built on the actor's T5 encoder. Given a masked
input sequence, it predicts a coverage-quality score (5 discrete labels). Used as the PPO
baseline to reduce variance in the actor update. The critic is only active during training
and is never called during fuzzing.

**Previous actor** (`_previous_actor`): a frozen deep-copy of the actor taken immediately
before each PPO update (π_{t-1}). Used to compute the importance-sampling ratio in the
clipped PPO objective, preventing large policy updates.

---

## Module boundaries

```
mlm_rl.py              AFL++ hook entry points; owns tokenizer, actor, finetune trigger
covrl/trainer.py       PPOTrainer; owns critic, finetune cycle, training loop
covrl/actor.py         ActorDataset, ActorDataCollator, ActorTrainer (PPO loss)
covrl/critic.py        CriticModel, CriticDataset, CriticDataCollator
utils/data_utils.py    Queue file loading; owns _queue_dir module state
utils/rewarding.py     Rewarder; afl-showmap pipeline + TF-IDF scoring
utils/masking.py       Mask insertion helpers used by Dataset classes
abstract_trainer.py    Trainer ABC (finetune / get_actor interface)
```

---

## Key deviations from original CovRL

| Feature | Original CovRL | This port |
|---|---|---|
| Architecture | TCP between afl-fuzz and Python server | Single AFL++ Python custom mutator |
| Splice mode | Implemented | Omitted (deferred) |
| Adaptive energy scheduling | stage_max doubles on new finds | Not implementable in fuzz_count(); fixed FUZZ_COUNT |
| Finetune gate | Requires `-S` sync mode | Always active; triggered every FINETUNE_INTERVAL seeds |
| Data source | corpus_dir passed explicitly | Queue directory read directly via data_utils |
| Incremental loading | known_ids dedup across cycles | Full queue scan each cycle; no accumulation in data_utils |
