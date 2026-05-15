# Implementing EDR for Your Robot — Complete Guide

---

## Part 1 — The Core Idea, Stated Simply

Your current physics model is:

$$\tau_{\text{phys}} = \underbrace{M(q)\ddot{q}}_{\tau_M} + \underbrace{C(q,\dot{q})\dot{q}}_{\tau_C} + \underbrace{g(q)}_{\tau_g} + \underbrace{\tau_f(\dot{q})}_{\text{friction}}$$

All four terms come from Pinocchio/your friction identification. It works reasonably well but not perfectly.

**The EDR idea:** each of these four terms is slightly wrong. Instead of learning one big correction $\tau_{\text{res}}$ on top, learn *four small corrections*, one per physics term:

$$\hat{\tau} = \bigl[M(q) + \delta M(q)\bigr]\ddot{q} + \bigl[C(q,\dot{q}) + \delta C(q,\dot{q})\bigr]\dot{q} + \bigl[g(q) + \delta g(q)\bigr] + \bigl[\tau_f(\dot{q}) + \delta\tau_f(\dot{q})\bigr]$$

Each $\delta$-network:
- Is **small** (few hundred parameters).
- Preserves the **structural property** of its component.
- Is initialized to output ~0 (so training starts exactly at your current physics model).

This is categorically different from "RNEA + MLP residual" because each correction is *constrained* to look like what it corrects.

---

## Part 2 — The Four Structural Properties 

Each term in the dynamics has a specific mathematical property. Your corrections must preserve them.

### 2.1 Inertia $M(q)$ is symmetric positive definite

Real physical inertia satisfies $M(q) = M(q)^\top$ and $x^\top M(q) x > 0$ for all $x \neq 0$. The correction $\delta M$ must preserve symmetry (otherwise you break conservation of momentum in the model).

### 2.2 Gravity depends only on $q$

Gravity comes from potential energy $V(q)$. It cannot depend on velocity or acceleration. $\delta g$ must also depend only on $q$.

### 2.3 Coriolis is linear in $\dot{q}$

The Coriolis torque has the form $C(q,\dot{q})\dot{q}$ — it's always proportional to $\dot{q}$. When $\dot{q} = 0$, Coriolis is zero. Your correction $\delta C$ must preserve this.

### 2.4 Friction is odd in $\dot{q}$

Real friction satisfies $\tau_f(-\dot{q}) = -\tau_f(\dot{q})$. Moving in the opposite direction flips the friction sign. Your correction $\delta\tau_f$ must preserve this symmetry.

---

## Part 3 — The Four Correction Networks, One by One

### 3.1 Gravity correction $\delta g(q)$ — simplest

**Input:** $q \in \mathbb{R}^5$
**Output:** $\delta g \in \mathbb{R}^5$
**Structure:** small MLP, nothing fancy.

$$\delta g(q) = \text{MLP}_g(q;\,\theta_g)$$

- 2 hidden layers of 32 units, tanh activations.
- **Last layer zero-initialized** so $\delta g(q) \approx 0$ at the start.
- About 400 parameters.

**Physical meaning:** your URDF's masses and centers-of-mass are slightly wrong. $\delta g$ learns a configuration-dependent bias that adjusts gravity.

### 3.2 Inertia correction $\delta M(q)$ — symmetric by construction

**Input:** $q \in \mathbb{R}^5$
**Output:** a symmetric $5\times5$ matrix $\delta M(q)$.

**The trick:** instead of outputting $25$ numbers, output a *lower-triangular* matrix $L_\theta(q)$ with $\frac{5 \cdot 6}{2} = 15$ entries, then form:

$$\delta M(q) = L_\theta(q)\,L_\theta(q)^\top$$

The product of any matrix with its transpose is guaranteed symmetric. Done.

**Implementation:**
- MLP takes $q$, outputs 15 numbers.
- Arrange these into lower-triangular positions of a $5\times5$ matrix $L_\theta$.
- Compute $\delta M = L_\theta L_\theta^\top$.
- **Last layer zero-initialized** → $L_\theta \approx 0$ → $\delta M \approx 0$.

**Contribution to torque:** $\delta M(q)\ddot{q}$ — the correction times the acceleration vector.

**About 500 parameters.**

*(Note: $L_\theta L_\theta^\top$ also enforces positive-semidefiniteness. If you want full control — the correction could make the effective inertia non-SPD if large — that's handled by initialization and regularization, discussed later.)*

### 3.3 Coriolis correction $\delta C(q,\dot{q})$ — velocity-product structure

**Input:** $q, \dot{q} \in \mathbb{R}^5$
**Output:** corrected Coriolis torque $\delta C(q,\dot{q})\dot{q} \in \mathbb{R}^5$.

**The trick:** Coriolis has the special form $C(q,\dot{q})\dot{q}$ where $C$ is a matrix *linear in $\dot{q}$*. The contribution to torque is therefore *quadratic in $\dot{q}$*.

Instead of learning the full matrix $C$, directly learn the quadratic function:

$$\delta C(q,\dot{q})\dot{q} = \sum_{j,k} \Gamma_{ijk}(q)\,\dot{q}_j\,\dot{q}_k$$

where $\Gamma_{ijk}$ are learned Christoffel-symbol-like coefficients. For each joint $i$, this is a quadratic form in $\dot{q}$ with coefficients depending on $q$.

**Cleaner implementation:** let an MLP with input $q$ output the coefficients of a quadratic form:

- $\text{MLP}_C(q) \to$ a tensor of shape $(5, 5, 5)$, which is $\Gamma_{ijk}(q)$
- Then compute $\delta C \dot{q}$ as the quadratic form: $(\delta C\dot{q})_i = \sum_{jk}\Gamma_{ijk}\,\dot{q}_j\dot{q}_k$

Output has **125 numbers** per sample. Instead of all 125, use the fact that we only care about the torque output — one simpler route:

$$\delta C(q,\dot{q})\dot{q} = \text{MLP}_C(q,\dot{q}) \odot (\dot{q}\odot\dot{q})_{\text{broadcast}}$$

The factor $(\dot{q}\odot\dot{q})$ (elementwise square of velocity) forces quadratic scaling. This is an approximation — not fully general — but captures the essential property that Coriolis vanishes at zero velocity.

**Even simpler (recommended for first version):**

$$\delta C(q,\dot{q})\dot{q} = \|\dot{q}\|_2\cdot\text{MLP}_C(q,\dot{q}/\|\dot{q}\|)\cdot\|\dot{q}\|_2$$

which is quadratic in $\|\dot{q}\|$ and normalized in direction. Small MLP (2 layers × 32 units, input dim 10, output dim 5).

About **500 parameters.**

### 3.4 Friction correction $\delta \tau_f(\dot{q})$ — odd function trick

**Input:** $\dot{q} \in \mathbb{R}^5$
**Output:** $\delta\tau_f \in \mathbb{R}^5$, with $\delta\tau_f(-\dot{q}) = -\delta\tau_f(\dot{q})$.

**The trick:** write $\delta\tau_f$ as a **product** — velocity times an even function of velocity:

$$\delta\tau_f(\dot{q}) = \dot{q}\odot h_\phi(|\dot{q}|)$$

- $h_\phi(|\dot{q}|)$ is an MLP that takes the *absolute value* of velocity as input.
- Because $|\dot{q}|$ is even, $h_\phi(|\dot{q}|)$ is even.
- Multiplying by $\dot{q}$ (which is odd) gives an odd output. 

This is a clean mathematical way to enforce the odd-function property. Exactly analogous to how $\sin(x) = x \cdot \frac{\sin(x)}{x}$ — the factor of $x$ carries the odd symmetry, the rest is even.

**MLP $h_\phi$:** 2 hidden layers of 16 units, input dim 5, output dim 5. About 400 parameters.

---

## Part 4 — Full Forward Pass, Summarized

Given $(q, \dot{q}, \ddot{q})$:

**Step 1.** Pinocchio computes nominal components:

$$M(q) \in \mathbb{R}^{5\times5}, \quad \tau_C = C(q,\dot{q})\dot{q}, \quad g(q), \quad \tau_f(\dot{q})$$

These are **fixed** — not learned.

**Step 2.** Each correction network computes its output:

$$\delta g(q), \quad \delta M(q) = L_\theta(q)L_\theta(q)^\top, \quad \delta C(q,\dot{q})\dot{q}, \quad \delta\tau_f(\dot{q}) = \dot{q}\odot h_\phi(|\dot{q}|)$$

**Step 3.** Assemble the corrected prediction:

$$\boxed{\hat{\tau} \;=\; \bigl[M(q) + \delta M(q)\bigr]\ddot{q} \;+\; \tau_C + \delta C(q,\dot{q})\dot{q} \;+\; g(q) + \delta g(q) \;+\; \tau_f(\dot{q}) + \delta\tau_f(\dot{q})}$$

Equivalently, if you prefer working with the residual:

$$\hat{\tau} = \tau_{\text{phys}} + \underbrace{\delta M(q)\ddot{q} + \delta C(q,\dot{q})\dot{q} + \delta g(q) + \delta\tau_f(\dot{q})}_{\text{structured residual}}$$

At initialization, every $\delta$ is zero, so $\hat{\tau} = \tau_{\text{phys}}$ — you start exactly at your current physics model.

---

## Part 5 — Architecture Diagram

```
               Inputs: (q, q̇, q̈)
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
  Pinocchio       q   q̇   q̈    Correction networks
  Physics
  (fixed)         
   │              ┌─── δg_net(q) ────→ δg ∈ ℝ⁵
   │              │
   │ M(q)         ├─── L_θ(q) ──┐
   │              │             │
   │              │             └──→ δM = L·Lᵀ ∈ ℝ^(5×5)
   │              │
   │ τ_C          ├─── δC_net(q, q̇) ─→ δC·q̇ ∈ ℝ⁵
   │              │
   │ g(q)         │
   │              │
   │ τ_f          └─── h_φ(|q̇|) ────→ δτ_f = q̇ ⊙ h_φ ∈ ℝ⁵
   │
   ▼
  Assemble:
  τ̂ = (M + δM)q̈ + (τ_C + δC·q̇) + (g + δg) + (τ_f + δτ_f)
                      │
                      ▼
              compare to τ_meas
```

---

## Part 6 — The Loss Function

$$\mathcal{L}_{\text{EDR}} = \mathcal{L}_{\text{data}} + \lambda_1\,\mathcal{L}_{\text{correction}} + \lambda_2\,\mathcal{L}_{\text{passivity}}$$

### 6.1 Data loss (per-joint normalized)

$$\mathcal{L}_{\text{data}} = \frac{1}{N\,n}\sum_{k=1}^N\sum_{i=1}^n\frac{(\hat{\tau}_i^{(k)} - \tau_{\text{meas},i}^{(k)})^2}{\sigma_i^2}$$

with $\sigma_i^2$ the per-joint variance of the current physics residual (computed once on training data, then fixed).

### 6.2 Correction magnitude regularization (Occam's razor)

Keep all corrections small unless data forces otherwise:

$$\mathcal{L}_{\text{correction}} = \|\delta M\|_F^2 + \|\delta C\dot{q}\|^2 + \|\delta g\|^2 + \|\delta\tau_f\|^2$$

This is the key regularizer. It says: "prefer the physics model over any correction; only correct when the data strongly disagrees."

### 6.3 Passivity constraint (skew-symmetry)

The matrix $\dot{M} - 2C$ is skew-symmetric for the true physics. When you add corrections, the *effective* inertia and Coriolis are:

$$\tilde{M}(q) = M(q) + \delta M(q),\qquad \tilde{C}(q,\dot{q}) = C(q,\dot{q}) + \delta C(q,\dot{q})$$

Enforce $\dot{\tilde{M}} - 2\tilde{C}$ skew-symmetric:

$$\mathcal{L}_{\text{passivity}} = \bigl\|\dot{\tilde{M}} - 2\tilde{C} + (\dot{\tilde{M}} - 2\tilde{C})^\top\bigr\|_F^2$$

This ensures your corrected dynamics remain passive — crucial if you ever want to use the model for control with stability guarantees.

**Computing $\dot{\tilde{M}}$:** use autograd — $\dot{\tilde{M}} = \sum_i \frac{\partial \tilde{M}}{\partial q_i}\dot{q}_i$. This is a Jacobian computation; PyTorch handles it cleanly.

### 6.4 Recommended weights

- $\lambda_1 = 10^{-3}$: gentle — don't over-suppress corrections.
- $\lambda_2 = 10^{-2}$: moderate — passivity is important but shouldn't dominate.

---

## Part 7 — Training Strategy

### 7.1 Initialization — start at the physics solution

All $\delta$-networks must output zero initially:

- $\delta g$, $\delta C$, $\delta \tau_f$: zero the last layer's weights and biases.
- $\delta M$: zero the last layer of the $L_\theta$ network. Then $L_\theta \approx 0$ and $\delta M \approx 0$.

This way, at epoch 0, $\hat{\tau} = \tau_{\text{phys}}$, and training can only improve from there.

### 7.2 Two-phase curriculum

**Phase 1 — Only $\delta g$ and $\delta\tau_f$ (epochs 1-15).** Gravity and friction are the biggest residuals on your data. Unfreeze these first and let them absorb the easy corrections. The $q$-dependent corrections often give the biggest wins early.

**Phase 2 — All four corrections (epoch 16+).** Unfreeze $\delta M$ and $\delta C$. The inertia/Coriolis corrections are subtler and need the gravity/friction residuals cleared out first.

This curriculum prevents $\delta M$ from absorbing gravity errors (it would — gravity errors are large, and $\delta M \cdot \ddot{q}$ has enough degrees of freedom to partially explain anything).

### 7.3 Optimizer

- AdamW, lr = $3\times 10^{-4}$, weight decay = $10^{-5}$
- Cosine annealing schedule
- Gradient clipping at norm 1.0
- Batch size ~256
- Float32 (the passivity loss requires Jacobian computation, which is fragile in float16)

---

## Part 8 — Parameter Count

| Network | Purpose | Parameters |
|---|---|---|
| $\delta g(q)$: MLP (2×32), in=5, out=5 | Gravity correction | ~400 |
| $L_\theta(q)$: MLP (2×32), in=5, out=15 | Inertia correction | ~500 |
| $\delta C(q,\dot{q})$: MLP (2×32), in=10, out=5 | Coriolis correction | ~500 |
| $h_\phi(\|\dot{q}\|)$: MLP (2×16), in=5, out=5 | Friction correction | ~400 |
| **Total** | | **~1,800** |

Absurdly small. But every parameter is structurally constrained, so each one pulls its weight.

---

## Part 9 — Why EDR Fits Your Data

Looking at your plot again:

- **J1, J2 shape is right, scale/bias wrong** → $\delta g(q)$ fixes this directly. Gravity correction learns per-configuration bias.
- **Friction is overshooting** → $\delta\tau_f$ can be *negative* (reducing over-aggressive friction). Your current friction is fixed — EDR's $\delta\tau_f$ can counteract it.
- **Coupling between joints not captured** → $\delta M(q)\ddot{q}$ and $\delta C(q,\dot{q})\dot{q}$ learn off-diagonal inertia and Coriolis terms your URDF might miss.
- **J6 unmodeled** → $\delta g_6(q) + \delta\tau_{f,6}(\dot{q}_6)$ can pick up everything from zero.

Every residual pattern in your plot maps to a correction network.

---

## Part 10 — What to Implement First

Build and test in this order:

**Step 1** (30 min): Write each correction network as an `nn.Module`. Test individually — verify output shapes, verify zero-initialization.

**Step 2** (1 hour): Test symmetry property. Sample 1000 random $q$, compute $\delta M(q)$, check $\|\delta M - \delta M^\top\|$ is exactly zero (machine epsilon).

**Step 3** (1 hour): Test odd-function property. Compute $\delta\tau_f(\dot{q})$ and $\delta\tau_f(-\dot{q})$, check their sum is zero to machine epsilon.

**Step 4** (2 hours): Assemble full model. Verify at init, $\hat{\tau} = \tau_{\text{phys}}$ within machine epsilon.

**Step 5** (overnight): Train with phase-1 curriculum. Plot $\tau_{\text{meas}}$, $\tau_{\text{phys}}$, $\hat{\tau}_{\text{EDR}}$ on a held-out trajectory — same style as your current plot.

Compare against three baselines:
1. RNEA + your current friction (the current state).
2. RNEA + MLP residual (200k params).
3. EDR (1,800 params).

If EDR beats (2) despite having 100× fewer parameters, you have a paper.

---

## Part 11 — One Subtle Point About the Passivity Loss

The skew-symmetry constraint is *expensive* to compute — it requires the Jacobian of $\tilde{M}$ with respect to $q$. For an implementation shortcut: compute it only on a random subset of each minibatch (say, 10%) and average. This keeps training fast while still enforcing the constraint on enough samples to matter.

If you find the passivity loss too slow or unstable, you can **omit it entirely** for the first paper. The architectural constraints (symmetric $\delta M$, odd $\delta\tau_f$) already give most of the benefit. Passivity enforcement is the cherry on top, not the foundation.

---

## Summary

EDR for your robot is:

$$\hat{\tau} = \bigl[M(q) + L_\theta(q)L_\theta(q)^\top\bigr]\ddot{q} + \bigl[C(q,\dot{q})\dot{q} + \delta C(q,\dot{q})\dot{q}\bigr] + \bigl[g(q) + \delta g(q)\bigr] + \bigl[\tau_f(\dot{q}) + \dot{q}\odot h_\phi(|\dot{q}|)\bigr]$$

Four small networks, each learning a *structured correction* to a physics component. Initialized to start at your current physics model. Only 1,800 parameters total. Symmetry, odd-ness, and passivity guaranteed by construction.

Start there. Want me to walk through the pseudocode for any single correction network in more detail — particularly the symmetric $\delta M$ construction or the skew-symmetry passivity loss?