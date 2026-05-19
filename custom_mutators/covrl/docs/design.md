# Algorithms

## Notation and state

**State carried between cycles:**
- $\mathcal{R}_{cur}$: current LLM-based rewarder (CovRL only)
- $\mathcal{M}_{cur}$: current LLM-based mutator
- $t$: cycle counter, starts at 0

**Inputs:** seed queue $Q$, finetuning dataset $\mathcal{D}_T$, corpus $\mathcal{C}$  
**Hyperparameters:** `iter_cycle`, `fuzz_count`

---

## Reward computation

**Function** $\text{CalcReward}(I_{val}, cov)$:

1.  **if** $I_{val} = \text{syntax\_error}$ **return** $-1.0$
2.  **if** $I_{val} = \text{semantic\_error}$ **return** $-0.5$
3.  **return** TF-IDF-weighted coverage reward $\in (0, 1)$

---

## Collection procedures

**Function** $\text{CollectInteresting}(Q, \text{Finetune})$:

1.  **for** $i = 1$ **to** `iter_cycle` **do**
2.  $\quad seed \leftarrow \text{SelectSeed}(Q)$
3.  $\quad$ **for** $j = 1$ **to** `fuzz_count` **do**
4.  $\quad\quad T \leftarrow \text{Mutate}(\mathcal{M}_{cur}, seed)$
5.  $\quad\quad I_{val}, cov \leftarrow \text{Execute}(T)$
6.  $\quad\quad$ **if** $\text{FoundNewCoverage}(T)$ **then**
7.  $\quad\quad\quad Q.\text{append}(T)$
8.  $\quad\quad\quad R \leftarrow \text{CalcReward}(I_{val}, cov)$
9.  $\quad\quad\quad \mathcal{D}_T.\text{append}((T, R))$
10. $\text{Finetune}(\mathcal{D}_T, \mathcal{C}, t)$
11. $t \leftarrow t + 1$

---

**Function** $\text{CollectAll}(Q, \text{Finetune})$:

1.  **for** $i = 1$ **to** `iter_cycle` **do**
2.  $\quad seed \leftarrow \text{SelectSeed}(Q)$
3.  $\quad$ **for** $j = 1$ **to** `fuzz_count` **do**
4.  $\quad\quad T \leftarrow \text{Mutate}(\mathcal{M}_{cur}, seed)$
5.  $\quad\quad I_{val}, cov \leftarrow \text{Execute}(T)$
6.  $\quad\quad R \leftarrow \text{CalcReward}(I_{val}, cov)$
7.  $\quad\quad \mathcal{D}_T.\text{append}((T, R))$
8.  $\quad\quad$ **if** $\text{FoundNewCoverage}(T)$ **then** $Q.\text{append}(T)$
9.  $\text{Finetune}(\mathcal{D}_T, \mathcal{C}, t)$
10. $t \leftarrow t + 1$

---

## Corpus mixing

**Function** $\text{MixCorpus}(\mathcal{D}_T, \mathcal{C})$:

1.  Sample $4 \cdot |\mathcal{D}_T|$ entries from $\mathcal{C}$, score each with TF-IDF using the current IDF (without updating it)
2.  **return** $\mathcal{D}_T \cup \mathcal{C}_{sample}$

Corpus samples are scored using the frozen IDF — they do not contribute to IDF updates — and are interleaved with rollouts in every training batch.

---

## Mutator training

**Function** $\text{FinetuneMutator}(\mathcal{M}_{prev}, \mathcal{R}, \mathcal{D}_{mix})$:

1.  **for each** batch $B \subseteq \mathcal{D}_{mix}$ **do**
2.  $\quad$ **for each** $(x, y, R) \in B$ **do**
3.  $\quad\quad \hat{R}(x, y) \leftarrow \mathcal{R}(x, y, R)$
4.  $\quad \mathcal{L}_{policy} \leftarrow \mathbb{E}_B\!\left[\text{PolicyLoss}(\pi_\theta, \pi_{prev}, x, y, \hat{R})\right]$
5.  $\quad \mathcal{L}_{CE} \leftarrow \mathbb{E}_B[-\log \pi_\theta(y \mid x)]$
6.  $\quad \mathcal{L} \leftarrow \mathcal{L}_{policy} + \mathcal{L}_{CE}$
7.  $\quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$
8.  **return** $\mathcal{M}_\theta$

Both $\mathcal{L}_{policy}$ and $\mathcal{L}_{CE}$ are taken over the full batch including corpus samples. The reward function $\mathcal{R}$ and the policy loss $\text{PolicyLoss}$ are abstract — concrete instances are given below.

---

## Reward functions

**$\mathcal{R}_{learned}(x, y, R)$:** query the trained rewarder on $(x, y)$ and map its predicted class to a scalar. Applied to all samples (rollout and corpus alike).

**$\mathcal{R}_{direct}(x, y, R)$:** return the stored $R$. Applied to all samples; corpus samples carry a TF-IDF score computed under the frozen IDF.

---

## Policy loss functions

**$\text{PolicyLoss}_{PPO}(\pi_\theta, \pi_{prev}, x, y, \hat{R})$:**

$$
-\min\!\left(\rho \cdot \hat{R},\ \text{clip}(\rho, 1 \pm \epsilon) \cdot \hat{R}\right) \quad \text{where } \rho = \pi_\theta(y \mid x) / \pi_{prev}(y \mid x)
$$

**$\text{PolicyLoss}_{GRPO}(\pi_\theta, \pi_{prev}, x, y, \hat{R})$:**

Compute group-relative advantage $A = (\hat{R} - \mu_g) / (\sigma_g + \epsilon_{std})$ where $\mu_g, \sigma_g$ are the mean and standard deviation of $\hat{R}$ within the rollout group $g$ containing $(x, y)$. Corpus samples have $A = 0$. Then:

$$
-\min\!\left(\rho \cdot A,\ \text{clip}(\rho, 1 \pm \epsilon) \cdot A\right)
$$

---

## Finetune procedures

**Function** $\text{FinetuneCovRL}(\mathcal{D}_T, \mathcal{C}, t)$:

1.  $\mathcal{D}_{mix} \leftarrow \text{MixCorpus}(\mathcal{D}_T, \mathcal{C})$
2.  $\mathcal{R}_{cur} \leftarrow \text{FinetuneRewarder}(\mathcal{R}_{cur}, \mathcal{D}_T)$
3.  **if** $t > 0$ **then** $\mathcal{M}_{cur} \leftarrow \text{FinetuneMutator}(\mathcal{M}_{cur}, \mathcal{R}_{learned}, \mathcal{D}_{mix})$

The rewarder is trained on unmixed $\mathcal{D}_T$ as an 8-class classifier over discretized rewards. The mutator update is skipped on the first cycle.

---

**Function** $\text{FinetuneRLLM}(\mathcal{D}_T, \mathcal{C}, t)$:

1.  $\mathcal{D}_{mix} \leftarrow \text{MixCorpus}(\mathcal{D}_T, \mathcal{C})$
2.  $\mathcal{M}_{cur} \leftarrow \text{FinetuneMutator}(\mathcal{M}_{cur}, \mathcal{R}_{direct}, \mathcal{D}_{mix})$

No rewarder is trained or queried. The mutator update runs on every cycle.

---

## Combinations

| Variant | Collection | Reward source | Policy loss |
|---|---|---|---|
| CovRL | $\text{CollectInteresting}$ | $\mathcal{R}_{learned}$ | $\text{PolicyLoss}_{PPO}$ |
| CovRL-All | $\text{CollectAll}$ | $\mathcal{R}_{learned}$ | $\text{PolicyLoss}_{PPO}$ |
| RLLM-PPO | $\text{CollectAll}$ | $\mathcal{R}_{direct}$ | $\text{PolicyLoss}_{PPO}$ |
| RLLM-GRPO | $\text{CollectAll}$ | $\mathcal{R}_{direct}$ | $\text{PolicyLoss}_{GRPO}$ |

Other combinations are valid; the four above are the primary experimental conditions.