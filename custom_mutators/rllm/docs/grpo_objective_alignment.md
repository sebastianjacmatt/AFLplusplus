# Objective Alignment in CovRL-Fuzz vs. GRPO Online Rollout

## 1. CovRL-Fuzz Actor Objective

CovRL-Fuzz trains the actor with a combined PPO and cross-entropy loss over a mixed
batch `D = D_mut ∪ D_corp` where `|D_corp| = 4|D_mut|` (4:1 corpus mixing):

$$L_\text{CovRL}(\theta; D) = \frac{1}{|D|} \sum_{(x,y) \in D} \Bigl[ L_\text{PPO}(\theta; x, y) + L_\text{CE}(\theta; x, y) \Bigr]$$

where the PPO term with clipped ratio is:

$$L_\text{PPO}(\theta; x, y) = -\min\!\Bigl( r(\theta)\, A,\; \mathrm{clip}(r(\theta),\, 1{-}\varepsilon,\, 1{+}\varepsilon)\, A \Bigr)$$

$$r(\theta) = \frac{\pi_\theta(y \mid x)}{\pi_{\theta_\text{old}}(y \mid x)}, \qquad A = \hat{V}_\text{critic}(x, y) - b$$

and the cross-entropy (masked span prediction) term is:

$$L_\text{CE}(\theta; x, y) = -\log \pi_\theta(y \mid x)$$

The reward $R(x, y)$ used to train the critic, and indirectly the advantage $A$, is:

$$R(x, y) = \begin{cases} -1.0 & \text{syntax error} \\ -0.5 & \text{semantic error} \\ \sigma\!\bigl(\log \mathbf{b}(x,y)^\top \mathbf{w}_\text{IDF}\bigr) & \text{valid} \end{cases}$$

where $\mathbf{b}(x,y)$ is the AFL++ edge-hit bitmap from afl-showmap and
$\mathbf{w}_\text{IDF}$ is the EMA-updated inverse document frequency vector.

---

## 2. Objective Alignment in CovRL-Fuzz

For any corpus sample $(x_c, y_c) \in D_\text{corp}$, both terms in $L_\text{CovRL}$
are computed over the same token sequence $y_c$. Since corpus programs are valid JS
with real coverage, $R(x_c, y_c) > 0$, giving $A > 0$. Therefore:

$$\nabla_\theta L_\text{PPO}(\theta;\, x_c, y_c) \propto +\nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

$$\nabla_\theta L_\text{CE}(\theta;\, x_c, y_c) = +\nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

Both gradients point in the **same direction** in parameter space — both push the
policy toward assigning higher probability to valid corpus completions $y_c$. The
4:1 ratio ensures corpus samples dominate the batch, so validity is the majority
gradient signal at every training step, independent of mutation quality.

---

## 3. Proposed GRPO Online Rollout Objective

In the proposed design, mutations are captured online during fuzzing. The GRPO
objective over a set of groups $\mathcal{G}$ (each group $g$ contains $G$ samples
from the same masked parent input) is:

$$L_\text{GRPO}(\theta;\, D_\text{online}) = \frac{1}{|\mathcal{G}|} \sum_{g \in \mathcal{G}}\; \frac{1}{G} \sum_{i \in g} \Bigl[ -\min\!\bigl( r_i(\theta)\, \hat{A}_i^g,\; \mathrm{clip}(r_i(\theta),\, 1{-}\varepsilon,\, 1{+}\varepsilon)\, \hat{A}_i^g \bigr) + \beta\,(r_i - \log r_i - 1) \Bigr]$$

where the group-relative advantage is:

$$\hat{A}_i^g = \frac{R_i - \mu_g}{\sigma_g + \varepsilon_\text{adv}}, \qquad \mu_g = \frac{1}{G}\sum_{i \in g} R_i, \qquad \sigma_g = \sqrt{\frac{1}{G}\sum_{i \in g}(R_i - \mu_g)^2}$$

and the KL penalty uses the generation-time reference
$r_i = \pi_\theta(y_i \mid x_i) / \pi_{\theta_\text{old}}(y_i \mid x_i)$.

To recover CovRL's validity anchor, a corpus CE term over a separate sample set
$D_\text{corp}$ is added:

$$L_\text{total}(\theta) = L_\text{GRPO}(\theta;\, D_\text{online}) + \lambda\, L_\text{CE}(\theta;\, D_\text{corp})$$

---

## 4. The Misalignment

The two terms are computed over **disjoint datasets**: $D_\text{online}$ (live AFL++
mutations) and $D_\text{corp}$ (static broader corpus). The combined gradient is:

$$\nabla_\theta L_\text{total} = \underbrace{\nabla_\theta L_\text{GRPO}(\theta;\, D_\text{online})}_{\text{over mutations}} + \lambda\; \underbrace{\nabla_\theta L_\text{CE}(\theta;\, D_\text{corp})}_{\text{over corpus}}$$

For a corpus sample $(x_c, y_c) \in D_\text{corp}$, the CE gradient is:

$$\nabla_\theta L_\text{CE}(\theta;\, x_c, y_c) = +\nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

but there is **no corresponding GRPO term** for $(x_c, y_c)$. The corpus sample does
not participate in the RL objective. Conversely, for a mutation $(x_m, y_m) \in
D_\text{online}$, GRPO computes a reward-weighted gradient but the CE term provides
no gradient for it.

Alignment cannot be guaranteed because the inner product of the two gradient
components:

$$\nabla_\theta L_\text{GRPO}(\theta;\, D_\text{online}) \cdot \nabla_\theta L_\text{CE}(\theta;\, D_\text{corp})$$

has indeterminate sign. For high-coverage mutations that use token patterns absent
from the corpus, the GRPO gradient (toward those patterns) and the corpus CE gradient
(away from them, toward corpus patterns) are directly opposed. At $\lambda = 1$ —
the scaling CovRL requires for the signal to be effective — neither objective
dominates and the update direction is ambiguous.

---

## 5. The Group Normalisation Failure Mode

GRPO's group-relative advantage introduces an additional failure mode absent from
CovRL's PPO. When validity collapses and all $G$ samples in a group are invalid,
the normalised advantage rewards **less-invalid** outputs:

$$R_i \in \{-1.0,\,-0.5\} \;\Rightarrow\; \hat{A}_i^g > 0 \text{ for semantic errors}$$

GRPO actively pushes the policy toward semantic errors in this regime. The corpus CE
term on a separate dataset cannot counteract this because its gradient is orthogonal
to the group normalisation signal. No external valid-JS signal enters the RL gradient.

---

## 6. The Alignment Condition

Objective alignment is restored if and only if corpus samples participate in **both**
the RL objective and the CE objective over the **same** token sequences. Formally,
for the corpus gradient components to be aligned:

$$\nabla_\theta L_\text{RL}(\theta;\, D_\text{corp}') \cdot \nabla_\theta L_\text{CE}(\theta;\, D_\text{corp}') \geq 0$$

requires that corpus samples $D_\text{corp}'$ appear in $D_\text{RL}$ with positive
expected advantage — i.e., corpus completions generated by the current policy must be
scored with real rewards and included in GRPO groups. This requires executing corpus
completions through the target binary at training time, reintroducing the subprocess
cost that online rollout eliminates for mutations.

---

## 7. Summary of the Constraint

| Design | Corpus in RL objective | Corpus in CE objective | Aligned? |
|---|---|---|---|
| CovRL PPO + CE (4:1 mixing) | Yes (positive reward) | Yes (same batch) | **Yes** |
| GRPO online + corpus CE (separate) | No | Yes | **No** |
| GRPO online + corpus GRPO groups (scored) | Yes | Yes (same batch) | **Yes** |

The fundamental constraint is: **online rollout accumulation from AFL++ mutations
alone cannot provide the aligned validity gradient that CovRL's 4:1 corpus mixing
provides, regardless of the RL algorithm used.** Any design that omits corpus samples
from the RL objective decouples validity anchoring from coverage optimisation and
risks validity collapse as the AFL++ queue shifts toward mutations-of-mutations.
