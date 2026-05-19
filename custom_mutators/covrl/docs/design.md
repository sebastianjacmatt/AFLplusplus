# CovRL-Fuzz implemented in AFL++

We implement the paper [CovRL-Fuzz]() as an AFL++ custom mutator

We allow for alternative collection strategies through two different custom_mutators `rllm.py` and `covrl.py`. The difference in algorithms are defined in 

## Different Algorithms


### Algorithms: Fuzzing with CovRL (Eom et al. 2024) and our adaptations

**State:**
- $\mathcal{R}_{prev}, \mathcal{R}_{cur}$: previous and current LLM-based rewarders
- $\mathcal{M}_{prev}, \mathcal{M}_{cur}$: previous and current LLM-based mutators

**Input:** finetuning dataset $\mathcal{D}_T$, seed queue $Q$, corpus $\mathcal{C}$  
**Hyperparameters:** `iter_cycle`, `fuzz_count`, `corpus_ratio` (default 4)

---

**Function** $\text{CollectInteresting}(Q, \text{Finetune})$:

1.  **for** $i = 1$ **to** `iter_cycle` **do**
2.  $\quad seed \leftarrow \text{SelectSeed}(Q)$
3.  $\quad$ **for** $j = 1$ **to** `fuzz_count` **do**
4.  $\quad\quad T \leftarrow \text{Mutate}(\mathcal{M}_{cur}, seed)$
5.  $\quad\quad I_{val}, cov \leftarrow \text{Execute}(T)$
6.  $\quad\quad$ **if** $\text{FoundNewCoverage}(T)$ **then**
7.  $\quad\quad\quad Q.\text{append}(T)$
8.  $\quad\quad\quad R_{cov} \leftarrow \text{CalcReward}(I_{val}, cov)$
9.  $\quad\quad\quad \mathcal{D}_T.\text{append}((T, R_{cov}))$
10. $\text{Finetune}(\mathcal{D}_T, \mathcal{C})$

---

**Function** $\text{CollectAll}(Q, \text{Finetune})$:

1.  **for** $i = 1$ **to** `iter_cycle` **do**
2.  $\quad seed \leftarrow \text{SelectSeed}(Q)$
3.  $\quad$ **for** $j = 1$ **to** `fuzz_count` **do**
4.  $\quad\quad T \leftarrow \text{Mutate}(\mathcal{M}_{cur}, seed)$
5.  $\quad\quad I_{val}, cov \leftarrow \text{Execute}(T)$
6.  $\quad\quad R_{cov} \leftarrow \text{CalcReward}(I_{val}, cov)$
7.  $\quad\quad \mathcal{D}_T.\text{append}((T, R_{cov}))$
8.  $\quad\quad$ **if** $\text{FoundNewCoverage}(T)$ **then** $Q.\text{append}(T)$
9.  $\text{Finetune}(\mathcal{D}_T, \mathcal{C})$

---

**Function** $\text{MixCorpus}(\mathcal{D}_T, \mathcal{C}, \rho)$:

1. **return** $\mathcal{D}_T \cup \mathcal{C}_{sample}$

Corpus samples carry no reward signal — they appear only as $(x, y)$ pairs used by the cross-entropy auxiliary loss term during mutator finetuning.

---

**Function** $\text{FinetuneCovRL}(\mathcal{D}_T, \mathcal{C})$:

1.  $\mathcal{D}_{mix} \leftarrow \text{MixCorpus}(\mathcal{D}_T, \mathcal{C}, \text{corpus\_ratio})$
2.  $\mathcal{R}_{prev}, \mathcal{M}_{prev} \leftarrow \mathcal{R}_{cur}, \mathcal{M}_{cur}$
3.  $\mathcal{R}_{cur} \leftarrow \text{FinetuneRewarder}(\mathcal{R}_{prev}, \mathcal{D}_{mix})$
4.  $\mathcal{M}_{cur} \leftarrow \text{FinetuneMutator}(\mathcal{M}_{prev}, \mathcal{R}_{cur}, \mathcal{D}_{mix})$

The rewarder and mutator is trained on the mixed dataset $\mathcal{D}_{mix}$: rollout samples contribute to both the policy gradient (via predicted rewards from $\mathcal{R}_{cur}$) and the cross-entropy term, while corpus samples contribute only to the cross-entropy term.

---

### Combinations

| Variant | Collection | Finetuning |
|---|---|---|
| CovRL (baseline) | $\text{CollectInteresting}$ | $\text{FinetuneCovRL}$ |
| CovRL-All | $\text{CollectAll}$ | $\text{FinetuneCovRL}$ |
| RLLM-Interesting | $\text{CollectInteresting}$ | $\text{FinetuneRLLM}$ |
| RLLM-All | $\text{CollectAll}$ | $\text{FinetuneRLLM}$ |


