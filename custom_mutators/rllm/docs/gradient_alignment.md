# Gradient Alignment: Why CovRL's 4:1 Mixing Works and Separate CE Does Not

## 1. Gradients as Vectors in Parameter Space

The model has parameters $\theta \in \mathbb{R}^d$. Every loss term produces a gradient
$\nabla_\theta L \in \mathbb{R}^d$, and a gradient step moves $\theta$ by:

$$\theta \leftarrow \theta - \alpha \cdot \nabla_\theta L$$

Two gradient terms **cooperate** when their inner product $g_1 \cdot g_2 > 0$ — they push
$\theta$ in the same general direction. They **conflict** when $g_1 \cdot g_2 < 0$ — one
partially undoes the other. A scaling factor $\lambda$ stretches a gradient vector but
cannot rotate it. If $g_1$ and $g_2$ point in opposing directions, no $\lambda$ resolves
the conflict.

---

## 2. CovRL's Same-Sample Guarantee

CovRL trains over a mixed batch $D = D_\text{mut} \cup D_\text{corp}$ with $|D_\text{corp}| = 4|D_\text{mut}|$.
For any corpus sample $(x_c, y_c) \in D_\text{corp}$, both the PPO and CE terms involve the
**same token sequence** $y_c$:

$$\nabla_\theta L_\text{PPO}(\theta;\, x_c, y_c) = -A_c \cdot \nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

$$\nabla_\theta L_\text{CE}(\theta;\, x_c, y_c) = -\nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

Both are scalar multiples of the **same vector** $\nabla_\theta \log \pi_\theta(y_c \mid x_c)$.
Their inner product is:

$$g_\text{PPO} \cdot g_\text{CE} = A_c \cdot \|\nabla_\theta \log \pi_\theta(y_c \mid x_c)\|^2$$

Since $y_c$ is valid JS, $R(x_c, y_c) > 0 \Rightarrow A_c > 0$, so this is **strictly positive
by construction**. The two gradients cannot oppose each other on the same sample — they are
co-linear, both pushing $\theta$ in the direction that increases $\pi_\theta(y_c \mid x_c)$.

The total batch gradient is:

$$\nabla_\theta L_\text{CovRL} = \frac{1}{|D|}\sum_{(x,y) \in D} (A + 1) \cdot \nabla_\theta \log \pi_\theta(y \mid x)$$

For corpus samples the weight $(A_c + 1) > 1$. For invalid mutations $A < 0$ and the weight
is small or negative. With 80% of the batch from corpus, the gradient direction is dominated
by valid JS reinforcement at every step. This is not a consequence of the 4:1 ratio alone —
it is a consequence of the 4:1 ratio **combined with the same-sample alignment guarantee**.

---

## 3. Why GRPO + Separate Corpus CE Cannot Be Fixed with $\lambda$

Now suppose GRPO runs on $D_\text{online} = \{(x_m, y_m)\}$ and a separate CE term runs on
$D_\text{corp} = \{(x_c, y_c)\}$ where $D_\text{online} \cap D_\text{corp} = \emptyset$.
The combined gradient is:

$$\nabla_\theta L_\text{total} = \underbrace{\sum_{m} \hat{A}_m \cdot \nabla_\theta \log \pi_\theta(y_m \mid x_m)}_{g_\text{RL}} + \lambda \underbrace{\sum_{c} \nabla_\theta \log \pi_\theta(y_c \mid x_c)}_{g_\text{CE}}$$

The two components involve **different token sequences**: $\{y_m\}$ are AFL++ mutations
(potentially invalid, unusual patterns), $\{y_c\}$ are corpus programs (standard valid JS).
These are different points in token space, so:

$$\nabla_\theta \log \pi_\theta(y_m \mid x_m) \quad \text{and} \quad \nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

are unrelated vectors in $\mathbb{R}^d$. Their inner product:

$$g_\text{RL} \cdot g_\text{CE} = \sum_{m,c} \hat{A}_m \cdot \nabla_\theta \log \pi_\theta(y_m \mid x_m) \cdot \nabla_\theta \log \pi_\theta(y_c \mid x_c)$$

has **indeterminate sign**. A high-coverage mutation uses token patterns absent from the
corpus; its GRPO gradient pushes $\theta$ toward those patterns while the corpus CE gradient
pulls $\theta$ away. Scaling by $\lambda$ only stretches $g_\text{CE}$ — it does not rotate it:

$$\lambda = 0 \implies \text{coverage signal only, no validity}$$
$$\lambda \to \infty \implies \text{validity signal only, no coverage}$$
$$\lambda = 1 \implies g_\text{RL} + g_\text{CE}, \text{ direction indeterminate}$$

There is no $\lambda$ that guarantees $g_\text{RL} \cdot g_\text{CE} \geq 0$. The directions of
the gradient vectors are determined by which samples each objective sees. A hyperparameter
cannot change which data the gradient was computed from.

---

## 4. Geometric Interpretation

Reduce parameter space to two dimensions for clarity:

- $\theta_1$: weight toward valid JS syntax
- $\theta_2$: weight toward unusual coverage-increasing token patterns

A high-coverage mutation $y_m$ rewards unusual patterns and may break syntax. Its GRPO
gradient (positive advantage) points toward $(+\Delta\theta_2,\, -\Delta\theta_1)$.

A corpus program $y_c$ is valid JS using standard syntax. Its CE gradient points toward
$(-\Delta\theta_2,\, +\Delta\theta_1)$.

Their inner product:

$$g_\text{RL} \cdot g_\text{CE} = (+\Delta\theta_2)(-\Delta\theta_2) + (-\Delta\theta_1)(+\Delta\theta_1) = -\|\Delta\theta_2\|^2 - \|\Delta\theta_1\|^2 < 0$$

The objectives are directly opposed. In CovRL, the corpus sample $y_c$ also appears in the
PPO objective. Its PPO gradient also points toward $(-\Delta\theta_2,\, +\Delta\theta_1)$. Both
terms add; they do not fight.

---

## 5. The Alignment Condition

Gradient alignment between two loss terms is guaranteed if and only if both terms involve
$\nabla_\theta \log \pi_\theta(y \mid x)$ for the **same $y$** at the **same $\theta$**:

$$g_1 \cdot g_2 = A \cdot \|\nabla_\theta \log \pi_\theta(y \mid x)\|^2 \geq 0 \iff A \geq 0$$

This is satisfied for corpus samples in CovRL's mixed batch because:

1. Corpus completions $y_c$ are valid JS → $R(x_c, y_c) > 0$ → $A_c > 0$
2. Both PPO and CE operate on the same $y_c$ → gradients are co-linear
3. The 4:1 ratio ensures 80% of the step gradient has this alignment property

Any architecture where the RL gradient and the CE gradient see different token sequences
cannot guarantee alignment regardless of mixing ratio, loss weighting, or training schedule.
This is a structural property of how gradients compose, not a hyperparameter choice.

---

## 6. Consequence for GRPO with Online Rollout

Online GRPO accumulates mutations from AFL++ into the RL buffer. If corpus samples are
added as a separate CE term (disjoint from the GRPO buffer), the §3 misalignment applies
at $\lambda \approx 1$ — neither objective dominates and the update direction is ambiguous.

The only architecture that recovers the same-sample guarantee is:

1. Sample $K$ corpus programs and apply masking to get $x_c$
2. Generate $G$ completions $\{y_c^{(j)}\}$ from the **current** $\pi_\theta$ (on-policy, $r_i \approx 1$)
3. Score each completion with a validity check → $R \in \{-1.0,\,-0.5,\,+0.5\}$
4. Form GRPO groups from these completions; valid completions receive $\hat{A} > 0$
5. Mix corpus groups 4:1 with mutation groups in the same GRPO batch

The gradient for a valid corpus completion $y_c^{(j)}$ is then:

$$\hat{A}_j^g \cdot \nabla_\theta \log \pi_\theta(y_c^{(j)} \mid x_c), \qquad \hat{A}_j^g > 0$$

This is the same active RL signal CovRL provides for corpus samples. The CE term in GRPO
(via the KL penalty) is also evaluated on the same $y_c^{(j)}$, preserving co-linearity.
The corpus samples are first-class participants in the RL objective, not a separate term
with a conflicting gradient direction.
