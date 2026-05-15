# Model IV: Physics-Residual Correction FNN (ResidualCorrectionFNN)

The ResidualCorrectionFNN instantiates the Physics-Informed Neural Network (PINN) paradigm for inverse dynamics by **hard-wiring** the analytical torque prediction into the output equation: the network outputs a bounded correction $\Delta\tau$ that is added to $\tau_{phys}$ to form the final prediction. This architecture enforces three inductive biases simultaneously that are absent from Models II and III: (1) at initialization, the prediction exactly equals the analytical model; (2) the correction is bounded in $\ell^\infty$-norm by a fixed scalar $\lambda$, so the network cannot deviate arbitrarily far from the physics baseline; (3) an L2 penalty on $\Delta\tau$ in the loss encourages the network to use the smallest correction that is consistent with the measured data. Together, these properties embed the physics model as a **structural prior** rather than a soft training signal.

---

## Notation

Symbols from Models II and III ($J, q, \dot{q}, \ddot{q}, \tau^*, \hat{\tau}, \tilde{z}, \mu_{\tau,j}, \sigma_{\tau,j}, x_k, x_p, x, \tau_g, \tau_M, \tau_C, \tau_f, \tau_{phys}, \tilde{\tau}^*, \tilde{\tau}_{phys}, \theta, W^{(l)}, b^{(l)}, h^{(l)}, N, w, \sigma(\cdot), \text{LN}, p$) carry identical definitions. The following additional symbols are specific to this model.

| Symbol | Dimension | Description |
|---|---|---|
| $\Delta\tau$ | $\mathbb{R}^J$ | Learned residual correction (N·m in normalized space) |
| $\lambda$ | — | Correction scale bound; fixed non-learnable constant, $\lambda = 0.5$ |
| $\text{MLP}_\theta$ | $\mathbb{R}^{7J} \to \mathbb{R}^J$ | Internal unconstrained feedforward network |
| $\alpha_{reg}$ | — | Correction magnitude regularization weight; $\alpha_{reg} = 0.05$ |
| $\mathcal{L}_{data}$ | — | Data fidelity loss (Eq. 13, Model II) |
| $\mathcal{L}_{reg}$ | — | Correction magnitude penalty |
| $\rho$ | — | Correction-to-physics ratio (monitoring diagnostic) |

---

## 1. Motivation: Hard vs. Soft Physics Coupling

In Model III, the physics prediction $\tau_{phys}$ acts as a soft regularization target during training: once training ends, the network operates as a plain MLP and there is no guarantee that predictions remain physically plausible on out-of-distribution inputs. In contrast, Model IV embeds $\tau_{phys}$ directly into the output equation, making the physics coupling **permanent** and architecture-level rather than loss-level.

The core idea is a decomposition of the prediction into two additive parts:
$$
\hat{\tau} = \underbrace{\tau_{phys}}_{\text{physics baseline}} + \underbrace{\Delta\tau}_{\text{learned correction}} \tag{1}
$$
The physics baseline $\tau_{phys}$ is pre-computed and fixed for each input; the neural network learns only the residual $\Delta\tau$ needed to correct the RNEA model's systematic errors (unmodeled compliance, calibration offsets, nonlinear friction). The advantage over learning the full torque directly (as in Models II and III) is that the network starts from a meaningful operating point (the RNEA prediction) and needs to learn only the error signal, which has considerably smaller magnitude and spatial variation than the full inverse dynamics.

---

## 2. Output Decomposition

### 2.1 Normalized Physics Baseline

All quantities in Eq. (1) operate in the normalized torque space defined in Eq. (5) of Model II. The normalized total analytical torque is:
$$
\tilde{\tau}_{phys} = \sum_{k \in \{g, M, C, f\}} \tilde{\tau}_k \in \mathbb{R}^J \tag{2}
$$
where each $\tilde{\tau}_k$ is normalized by the sum-consistent scheme of Eq. (1) in Model III. By construction (Theorem, Model III), $\tilde{\tau}_{phys} = (\tau_{phys} - \mu_\tau)/\sigma_\tau$, so it lives in the same space as the prediction $\hat{\tau}_{norm}$ and the ground-truth $\tilde{\tau}^*$.

### 2.2 Bounded Neural Residual

The correction $\Delta\tau$ is produced by passing the output of an unconstrained MLP through a component-wise scaled hyperbolic tangent:
$$
\Delta\tau = \lambda \cdot \tanh\!\bigl(\text{MLP}_\theta(x)\bigr) \in \mathbb{R}^J \tag{3}
$$
where $\text{MLP}_\theta: \mathbb{R}^{7J} \to \mathbb{R}^J$ is an unconstrained feedforward network (defined in Section 5), $x \in \mathbb{R}^{7J}$ is the augmented input (Eq. 3, Model III), and $\lambda = 0.5$ is a fixed scalar registered as a non-differentiable buffer — it is stored with the model state but excluded from the parameter set $\theta$ and never updated by the optimizer.

The full prediction is therefore:
$$
\hat{\tau}_{norm} = \tilde{\tau}_{phys} + \lambda \cdot \tanh\!\bigl(\text{MLP}_\theta(x)\bigr) \tag{4}
$$

---

## 3. Hard Structural Bound on the Correction

### 3.1 $\ell^\infty$ Bound via Hyperbolic Tangent

Since $\tanh(y) \in (-1, 1)$ for all $y \in \mathbb{R}$, the correction magnitude is bounded by $\lambda$ in $\ell^\infty$-norm for every input $x$ and every weight configuration $\theta$:
$$
\|\Delta\tau\|_\infty = \lambda\, \|\tanh(\text{MLP}_\theta(x))\|_\infty < \lambda \tag{5}
$$
Equivalently:
$$
|\hat{\tau}_{norm,j} - \tilde{\tau}_{phys,j}| < \lambda, \qquad \forall\, j \in \{1,\ldots,J\},\quad \forall\, (x, \theta) \tag{6}
$$
This is a **hard architectural constraint**: no matter how large the MLP output grows (due to linear extrapolation through deep layers), the tanh saturates at $\pm 1$ and the correction is clamped. In de-normalized units, Eq. (6) translates to:
$$
|\hat{\tau}_j - \tau_{phys,j}| < \lambda \cdot \sigma_{\tau,j} \quad \text{(N·m)} \tag{7}
$$
With $\lambda = 0.5$ and $\sigma_{\tau,j} \approx 0.1$–$0.3$ N·m (joint-dependent), this bounds the correction to below approximately 0.05–0.15 N·m per joint — small relative to the full torque range but sufficient to correct RNEA errors.

### 3.2 Out-of-Distribution Safety

Models II and III contain no mechanism to prevent catastrophically large predictions on states outside the training distribution: the MLP's affine layers can extrapolate arbitrarily. In contrast, the tanh saturates as the MLP output grows:
$$
\lim_{\|\text{MLP}_\theta(x)\|_\infty \to \infty} \tanh(\text{MLP}_\theta(x)) = \pm \mathbf{1}
\implies \Delta\tau \to \pm\lambda\,\mathbf{1} \tag{8}
$$
At the limit of extrapolation, the prediction asymptotes to $\tilde{\tau}_{phys} \pm \lambda$ — it never escapes the corridor $[\tilde{\tau}_{phys} - \lambda,\, \tilde{\tau}_{phys} + \lambda]$ regardless of how far the input departs from the training manifold. The physics model therefore provides a **guaranteed fallback**: if the network has low confidence (which manifests as large, saturated outputs), the prediction reverts to $\tau_{phys}$.

---

## 4. Warm-Start Initialization

### 4.1 Near-Zero Output Layer

All hidden layers use Xavier normal initialization (Eq. 11, Model II). The output layer alone uses a near-zero scale:
$$
W^{(4)} \leftarrow 0.01 \cdot W_{Xavier}, \qquad b^{(4)} \leftarrow \mathbf{0} \tag{9}
$$
This is a deliberate choice of initialization point in parameter space, not an architectural modification.

### 4.2 Initial Prediction Equals Analytical Baseline

With the near-zero output layer, $\text{MLP}_\theta(x) \approx \mathbf{0}$ for any input at epoch $e = 0$. Using the Taylor expansion $\tanh(z) = z + O(z^3)$ for small $|z|$:
$$
\hat{\tau}_{norm}\big|_{\theta_0} = \tilde{\tau}_{phys} + \lambda \cdot \tanh(\underbrace{\text{MLP}_{\theta_0}(x)}_{\approx\, \mathbf{0}}) \approx \tilde{\tau}_{phys} \tag{10}
$$
The initial prediction therefore exactly reproduces the analytical torque. The training loss at epoch zero equals the squared RNEA prediction error:
$$
\mathcal{L}(\theta_0) = \frac{1}{N}\sum_{i,j} w_j\,(\tilde{\tau}_{phys,ij} - \tilde{\tau}^*_{ij})^2 = \mathcal{L}_{analytical} \tag{11}
$$
Gradient descent adopts neural corrections only when they strictly reduce this quantity. **This is a mathematical guarantee**: the optimization trajectory begins at the physics solution and moves monotonically toward data-consistent corrections. In contrast, Models II and III start from a random initialization and spend early epochs learning coarse-scale structure that the RNEA already provides.

---

## 5. Network Architecture

The internal MLP $\text{MLP}_\theta: \mathbb{R}^{35} \to \mathbb{R}^J$ follows the same three-hidden-layer structure as Models II and III:

**Hidden layers** ($l = 1, 2, 3$):
$$
h^{(l)} = \text{Dropout}_p\!\left(\sigma\!\left(\text{LN}\!\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)\right)\right), \quad h^{(0)} = x \tag{12}
$$

**Output of MLP** (raw, unbounded):
$$
\text{MLP}_\theta(x) = W^{(4)} h^{(3)} + b^{(4)} \in \mathbb{R}^J \tag{13}
$$

**Final prediction** (combining Eqs. 2, 3, and 13):
$$
\hat{\tau}_{norm} = \tilde{\tau}_{phys} + \lambda \cdot \tanh\!\bigl(W^{(4)} h^{(3)} + b^{(4)}\bigr) \tag{14}
$$

The layer dimensions are $[d_0, d_1, d_2, d_3, d_4] = [35, 256, 512, 256, 5]$ — identical to Model III. The dropout rate is $p = 0.1$, matching Model III; the physics features and structural constraint together regularize the solution space sufficiently to make aggressive dropout unnecessary.

**Parameter count:**
$$
|\theta| = (35{\cdot}256+256) + (256{\cdot}512+512) + (512{\cdot}256+256) + (256{\cdot}5+5) + 2(256+512+256) \approx 275{,}000
$$

---

## 6. Loss Function

### 6.1 Data Fidelity with Correction Magnitude Regularization

The training objective penalizes both prediction error and the magnitude of the correction:
$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{J} w_j\,(\hat{\tau}_{norm,ij} - \tilde{\tau}^*_{ij})^2}_{\mathcal{L}_{data}} + \underbrace{\frac{\alpha_{reg}}{N \cdot J}\sum_{i=1}^{N}\|\Delta\tau_i\|^2}_{\mathcal{L}_{reg}} \tag{15}
$$
where $w = [1.0, 2.5, 1.0, 1.0, 1.0]^\top$, $\alpha_{reg} = 0.05$, and $\Delta\tau_i = \lambda \cdot \tanh(\text{MLP}_\theta(x_i)) \in \mathbb{R}^J$.

Since $\Delta\tau_i = \hat{\tau}_{norm,i} - \tilde{\tau}_{phys,i}$, the regularization term can equivalently be written as:
$$
\mathcal{L}_{reg} = \frac{\alpha_{reg}}{N \cdot J}\sum_{i=1}^{N}\|\hat{\tau}_{norm,i} - \tilde{\tau}_{phys,i}\|^2 \tag{16}
$$
This penalizes isotropic deviations from the physics baseline across all joints (without the joint weighting of $\mathcal{L}_{data}$, which focuses extra capacity on J2). The interplay between the two terms implements a **minimal correction principle**: the optimizer searches for the correction of smallest L2 norm that reduces the data prediction error. This is structurally identical to Tikhonov regularization where $\|\Delta\tau\|^2$ is the regularizer centered at zero (i.e., at the physics prediction).

### 6.2 Correction-to-Physics Ratio

A dimensionless diagnostic is tracked during training to monitor whether corrections are growing relative to the physics baseline:
$$
\rho = \frac{\mathbb{E}_i\!\left[\,\|\Delta\tau_i\|_1 / J\,\right]}{\mathbb{E}_i\!\left[\,\|\tilde{\tau}_{phys,i}\|_1 / J\,\right] + 10^{-12}} \tag{17}
$$
where $\|\cdot\|_1$ is the $\ell^1$-norm and expectations are over the training batch. A value $\rho \ll 1$ indicates that the network is making modest corrections within the physics baseline's corridor; $\rho \sim 1$ would indicate that the network is attempting to override the physics model entirely, which $\mathcal{L}_{reg}$ and the $\tanh$ bound jointly prevent.

---

## 7. Training Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| Hidden layers $[d_1, d_2, d_3]$ | $[256,\, 512,\, 256]$ | Same structure as Models II and III |
| Input dimension $d_0$ | $35$ | $7J$ augmented features (identical to Model III) |
| Correction scale $\lambda$ | $0.5$ | Non-learnable buffer; hard bound on $\|\Delta\tau\|_\infty$ |
| Activation $\sigma$ | SiLU | |
| Normalization | LayerNorm | |
| Dropout rate $p$ | $0.1$ | |
| Batch size | $512$ | |
| Optimizer | AdamW | $\beta_1{=}0.9,\,\beta_2{=}0.999$ |
| Learning rate $\eta_0$ | $3 \times 10^{-4}$ | |
| Weight decay | $5 \times 10^{-3}$ | |
| LR schedule | Warm-up cosine | $e_{warm} = \lfloor E/20 \rfloor$ |
| Gradient clip norm | $5.0$ | |
| Feature noise std $\sigma_n$ | $0.02$ | Applied to full input $x$ |
| Correction penalty $\alpha_{reg}$ | $0.05$ | L2 weight on $\|\Delta\tau\|^2$ |
| Early stopping patience $P$ | $60$ epochs | Monitors val RMSE (N·m) |
| Min improvement $\delta_{min}$ | $10^{-4}$ N·m | |
| Max epochs $E$ | $100$ | |
| Training samples $N$ | $272{,}465$ | Identical dataset to Models II and III |

---

## 8. Theoretical Properties

**Hard vs. soft physics coupling.** In Model III, setting $\alpha_0 = 0$ in the loss reduces the model exactly to Model II: the physics influence is entirely eliminated. In Model IV, removing the physics penalty ($\alpha_{reg} = 0$) does not remove the physics from the model — $\tilde{\tau}_{phys}$ remains in the output equation (Eq. 4) regardless of the loss configuration. The physics coupling in Model IV is architectural, not loss-level.

**Minimal correction principle.** The combination of warm-start initialization ($\Delta\tau = 0$ at epoch 0) and L2 regularization on $\Delta\tau$ implements Occam's razor over corrections: gradient descent finds the correction of minimum squared norm that reduces the prediction error below the RNEA baseline. Formally, the saddle-point condition for Eq. (15) at convergence satisfies:
$$
\nabla_{\Delta\tau} \mathcal{L}_{data} + \frac{2\alpha_{reg}}{J}\,\Delta\tau = 0 \implies \Delta\tau^* = -\frac{J}{2\alpha_{reg}}\, \nabla_{\Delta\tau} \mathcal{L}_{data}
$$
The regularization weight $\alpha_{reg}$ directly controls the scale of the learned correction: larger $\alpha_{reg}$ enforces smaller corrections and keeps the prediction closer to the physics model.

**Convergence benefit.** Starting at $\hat{\tau}_{norm,0} = \tilde{\tau}_{phys}$ (Eq. 10) rather than a random initialization means the early training epochs refine corrections from a meaningful starting point instead of learning the coarse-scale gravity and inertia structure from scratch. Empirically, Model IV converges in fewer epochs than Model III (82 vs. 100 epochs to the best validation checkpoint) despite sharing the same architecture, consistent with the warm-start reducing the effective optimization distance.

**Out-of-distribution guarantee.** By Eq. (5), the prediction satisfies $\hat{\tau}_{norm,j} \in [\tilde{\tau}_{phys,j} - \lambda,\, \tilde{\tau}_{phys,j} + \lambda]$ for all inputs and all parameters. If the physics model is globally accurate to within $\lambda$ (in normalized units), the PINN prediction is never worse than $2\lambda$ from the true torque on any input. Models II and III offer no such guarantee and can extrapolate arbitrarily on unseen inputs.

**Experimental results.** The ResidualCorrectionFNN achieves a mean test RMSE of $\overline{\text{RMSE}} = 0.0905$ N·m and mean per-joint $R^2 = 0.861$. Per-joint results and comparisons to Models II and III are below.

| Joint | Role | Test RMSE (N·m) | $R^2$ | $\Delta$RMSE vs. II | $\Delta$RMSE vs. III |
|---|---|---|---|---|---|
| J1 | Yaw (base) | $0.0766$ | $0.866$ | $+10.2\%$ worse | $+5.6\%$ worse |
| J2 | Shoulder pitch | $0.1535$ | $0.834$ | $-11.4\%$ better | $-1.6\%$ better |
| J3 | Elbow pitch | $0.0945$ | $0.907$ | $-8.4\%$ better | $+3.8\%$ worse |
| J4 | Wrist pitch | $0.0389$ | $0.879$ | $0.0\%$ equal | $+4.0\%$ worse |
| J5 | Wrist roll | $0.0891$ | $0.821$ | $-0.1\%$ better | $+5.1\%$ worse |
| **Mean** | — | $\mathbf{0.0905}$ | $\mathbf{0.861}$ | $\mathbf{-4.5\%}$ | $\mathbf{+2.5\%}$ |

Model IV outperforms the BlackBoxFNN on aggregate (mean RMSE reduced by 4.5%) and achieves the best result on J2 (shoulder pitch), where the structural warm-start and the RNEA gravity correction provide the largest benefit. However, it is marginally outperformed by the PhysicsRegularizedFNN on J3–J5, suggesting that the soft physics penalty in Model III better captures joint-specific correction patterns for these lighter joints. The key advantage of Model IV over Model III is not peak accuracy on a single dataset, but the **guaranteed out-of-distribution behavior** (Eq. 5), which is critical for deployment in closed-loop control where the robot may encounter configurations outside the training distribution.
