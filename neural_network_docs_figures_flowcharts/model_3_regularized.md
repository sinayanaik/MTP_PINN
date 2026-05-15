# Model III: Physics-Regularized Feedforward Neural Network (PhysicsRegularizedFNN)

The PhysicsRegularizedFNN extends the baseline architecture (Model II) by introducing physics knowledge at two levels simultaneously: the analytical RNEA decomposition is concatenated to the kinematic features as an additional input (so the network has direct access to the physics-derived torque components), and a penalty term in the training loss penalizes deviation of the network prediction from the analytical prediction. This creates a **soft coupling** to the physics model: the gradient is biased toward predictions that are consistent with rigid-body mechanics, but the network retains the freedom to override the physics wherever systematic model errors exist. The result is a prior-regularized regression rather than a hard constraint.

---

## Notation

Symbols from Model II ($J, q, \dot{q}, \ddot{q}, \tau^*, \hat{\tau}, \tilde{z}, \mu_{z,j}, \sigma_{z,j}, \tilde{x}_k, \theta, W^{(l)}, b^{(l)}, h^{(l)}, N, w, \sigma(\cdot), \text{LN}, p$) carry identical definitions here. The following additional symbols are introduced.

| Symbol | Dimension | Description |
|---|---|---|
| $\tau_g$ | $\mathbb{R}^J$ | Gravitational torque from RNEA: $\tau_g = \text{RNEA}(q, \mathbf{0}, \mathbf{0})$ |
| $\tau_M$ | $\mathbb{R}^J$ | Inertial torque: $\tau_M = \text{RNEA}(q, \mathbf{0}, \ddot{q}) - \tau_g = M(q)\ddot{q}$ |
| $\tau_C$ | $\mathbb{R}^J$ | Coriolis–centripetal torque: $\tau_C = \text{RNEA}(q, \dot{q}, \mathbf{0}) - \tau_g = C(q,\dot{q})\dot{q}$ |
| $\tau_f$ | $\mathbb{R}^J$ | Friction torque: $\tau_f = c \odot \tanh(\dot{q}/\varepsilon_f) + v \odot \dot{q}$ |
| $\tau_{phys}$ | $\mathbb{R}^J$ | Total analytical torque: $\tau_{phys} = \tau_g + \tau_M + \tau_C + \tau_f$ |
| $\tilde{\tau}_k$ | $\mathbb{R}^J$ | Normalized $k$-th physics component, $k \in \{g, M, C, f\}$ |
| $x_p$ | $\mathbb{R}^{4J}$ | Normalized physics feature vector |
| $x$ | $\mathbb{R}^{7J}$ | Augmented feature vector: $x = [x_k^\top,\, x_p^\top]^\top$ |
| $\tilde{\tau}_{phys}$ | $\mathbb{R}^J$ | Normalized total analytical torque (equals sum of $\tilde{\tau}_k$) |
| $\alpha_0$ | — | Target physics regularization weight; default $\alpha_0 = 0.5$ |
| $\alpha_{eff}(e)$ | — | Effective (time-varying) physics weight at epoch $e$ |
| $f_{warm}$ | — | Physics warm-up fraction; $f_{warm} = 0.05$ |
| $e_{warm}^{phys}$ | — | Physics warm-up duration in epochs: $e_{warm}^{phys} = \max(1, \lfloor f_{warm} E \rfloor)$ |
| $\mathcal{L}_{data}$ | — | Data fidelity loss (Eq. 13 from Model II) |
| $\mathcal{L}_{phys}$ | — | Physics consistency loss (deviation from $\tilde{\tau}_{phys}$) |

---

## 1. Motivation: Physics as a Soft Training Prior

The BlackBoxFNN must independently rediscover the complete structure of inverse dynamics — gravity, inertia, Coriolis, and friction — from kinematic measurements alone. For a 5-DOF manipulator operating across diverse trajectories, this requires a large training set. The key observation motivating Model III is that the analytical RNEA model already captures the dominant physics structure, even if it is imperfect: its predictions lie within 0.20–0.25 N·m of the measured torques on dynamic trajectories (see Model I). Rather than ignoring this partial knowledge, Model III uses it in two complementary ways:

1. **As additional input features**: the four RNEA torque components $(\tau_g, \tau_M, \tau_C, \tau_f)$ are appended to the kinematic features, giving the network explicit access to the physics-derived torque breakdown. The network can then learn residual corrections to each component rather than reconstructing all of them from kinematics.

2. **As a regularization target**: a physics penalty loss penalizes the network's prediction from deviating from the analytical prediction $\tau_{phys}$. In the low-data regime, where training samples are insufficient to determine the full inverse dynamics, this penalty acts as a prior that anchors the prediction near the physics baseline.

---

## 2. Physics Feature Construction

### 2.1 Sum-Consistent Normalization

The four RNEA components $(\tau_g, \tau_M, \tau_C, \tau_f)$ and the measured torques $\tau^*$ must live in the same normalized space for the physics regularization loss to be meaningful. A naive approach — normalizing each component independently — would destroy the additive identity $\tau_{phys} = \tau_g + \tau_M + \tau_C + \tau_f$ after normalization. The following scheme preserves it.

For each joint $j$ and each component $k \in \{g, M, C, f\}$, define the normalized physics component as:
$$
\tilde{\tau}_{k,j} = \frac{\tau_{k,j} - \mu_{\tau,j}/4}{\sigma_{\tau,j}} \tag{1}
$$
where $\mu_{\tau,j}$ and $\sigma_{\tau,j}$ are the **same** statistics used to normalize the measured torque (Eq. 5, Model II). The mean offset $\mu_{\tau,j}/4$ distributes the total torque mean equally across the four components.

**Claim (Sum Consistency):** The sum of the four normalized components equals the normalized total analytical torque:
$$
\sum_{k \in \{g, M, C, f\}} \tilde{\tau}_{k,j} = \frac{(\tau_{g,j} + \tau_{M,j} + \tau_{C,j} + \tau_{f,j}) - \mu_{\tau,j}}{\sigma_{\tau,j}} = \frac{\tau_{phys,j} - \mu_{\tau,j}}{\sigma_{\tau,j}} \triangleq \tilde{\tau}_{phys,j} \tag{2}
$$

*Proof.* Summing Eq. (1) over $k$:
$$
\sum_k \tilde{\tau}_{k,j} = \frac{1}{\sigma_{\tau,j}} \sum_k \left(\tau_{k,j} - \frac{\mu_{\tau,j}}{4}\right) = \frac{\tau_{phys,j} - \mu_{\tau,j}}{\sigma_{\tau,j}}
$$
since $\sum_{k=1}^4 \mu_{\tau,j}/4 = \mu_{\tau,j}$ and $\sum_k \tau_{k,j} = \tau_{phys,j}$ by the rigid-body identity. $\square$

This identity ensures that the physics regularization target $\tilde{\tau}_{phys}$ is expressed on the same scale as the prediction $\hat{\tau}_{norm}$ and ground-truth $\tilde{\tau}^*$. The loss terms $\mathcal{L}_{data}$ and $\mathcal{L}_{phys}$ are therefore commensurable and can be meaningfully weighted against each other.

### 2.2 Augmented Feature Vector

The normalized kinematic and physics features are concatenated into a single 35-dimensional input:
$$
x = \biggl[\underbrace{\tilde{q}^\top,\;\tilde{\dot{q}}^\top,\;\tilde{\ddot{q}}^\top}_{x_k \in \mathbb{R}^{3J}},\;\underbrace{\tilde{\tau}_g^\top,\;\tilde{\tau}_M^\top,\;\tilde{\tau}_C^\top,\;\tilde{\tau}_f^\top}_{x_p \in \mathbb{R}^{4J}}\biggr]^\top \in \mathbb{R}^{7J} \tag{3}
$$
For $J = 5$, the augmented input dimension is $7J = 35$.

---

## 3. Network Architecture

The architecture follows the same MLP structure as Model II but with a wider input layer to accommodate the augmented features. Setting $h^{(0)} = x$, each hidden layer applies:
$$
h^{(l)} = \text{Dropout}_p\!\left(\sigma\!\left(\text{LN}\!\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)\right)\right), \quad l = 1, 2, 3 \tag{4}
$$
and the output layer:
$$
\hat{\tau}_{norm} = W^{(4)} h^{(3)} + b^{(4)} \in \mathbb{R}^J \tag{5}
$$
The layer dimensions are $[d_0, d_1, d_2, d_3, d_4] = [35, 256, 512, 256, 5]$. The dropout rate is $p = 0.1$ (lower than the BlackBoxFNN's $p = 0.4$) because the physics features $x_p$ provide structured prior information that reduces the effective hypothesis space, thereby reducing the need for aggressive regularization. All other architectural components — SiLU activation, Layer Normalization, Xavier initialization — are identical to Model II.

**Structural interpretation.** The first affine transformation $z^{(1)} = W^{(1)} x + b^{(1)}$ can be written by partitioning the weight matrix conformally with the input split $[x_k; x_p]$:
$$
z^{(1)} = \underbrace{W^{(1)}_{kin}\, x_k}_{\text{residual pathway}} + \underbrace{W^{(1)}_{phys}\, x_p}_{\text{physics trust projection}} + b^{(1)} \tag{6}
$$
where $W^{(1)}_{kin} \in \mathbb{R}^{d_1 \times 3J}$ and $W^{(1)}_{phys} \in \mathbb{R}^{d_1 \times 4J}$. The sub-matrix $W^{(1)}_{phys}$ effectively learns a per-joint weighting over the four physics components. During training, if the gravity prediction $\tilde{\tau}_g$ has high predictive value for the shoulder joint (J2), the gradient reinforces large entries in the columns of $W^{(1)}_{phys}$ corresponding to $\tilde{\tau}_{g}$. Conversely, $\tilde{\tau}_M$ — which depends on numerically differentiated accelerations and therefore carries more noise — typically receives smaller, noisier weights. This is analogous to a soft attention mechanism over physics channels: the network learns which RNEA components to trust joint-by-joint, without explicit supervisory signal on component reliability.

**Parameter count.** The total number of trainable parameters is:
$$
|\theta| = (35{\cdot}256+256) + (256{\cdot}512+512) + (512{\cdot}256+256) + (256{\cdot}5+5) + 2(256+512+256) \approx 275{,}000
$$

---

## 4. Physics-Regularized Loss Function

### 4.1 Tikhonov-Form Objective

The total training loss is the sum of two terms: a data fidelity term $\mathcal{L}_{data}$ and a physics consistency term $\mathcal{L}_{phys}$, weighted by the time-varying coefficient $\alpha_{eff}(e)$:
$$
\mathcal{L}(\theta; e) = \mathcal{L}_{data}(\theta) + \alpha_{eff}(e)\; \mathcal{L}_{phys}(\theta) \tag{7}
$$

Both terms use the same joint-weighted MSE structure. Using $\tilde{\tau}_{phys}$ as defined in Eq. (2) as the physics reference:
$$
\mathcal{L}_{data}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{J} w_j\,\bigl(\hat{\tau}_{norm,ij} - \tilde{\tau}^*_{ij}\bigr)^2 \tag{8}
$$
$$
\mathcal{L}_{phys}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{J} w_j\,\bigl(\hat{\tau}_{norm,ij} - \tilde{\tau}_{phys,ij}\bigr)^2 \tag{9}
$$

Here $w = [1.0, 2.5, 1.0, 1.0, 1.0]^\top$ and $\tilde{\tau}_{phys,ij}$ is the normalized total analytical torque for sample $i$, joint $j$.

**Interpretation.** The physics term $\mathcal{L}_{phys}$ measures how far the network prediction deviates from the RNEA baseline. Minimizing $\mathcal{L}_{phys}$ alone would yield $\hat{\tau}_{norm} = \tilde{\tau}_{phys}$ — the pure analytical model. The combined objective (Eq. 7) therefore finds a solution on the Pareto frontier between two competing objectives: fitting the measured torques (via $\mathcal{L}_{data}$) and agreeing with the physics model (via $\mathcal{L}_{phys}$). The weight $\alpha_{eff}$ controls the operating point on this frontier. Importantly, $\mathcal{L}_{data}$ is always fully active; $\alpha_{eff} \mathcal{L}_{phys}$ acts purely as a regularizer without ever reducing the emphasis on fitting observed data.

This structure is identical to Tikhonov regularization in classical estimation theory, where $\mathcal{L}_{phys}$ is the regularizer and $\tilde{\tau}_{phys}$ plays the role of the regularization center (the prior estimate). The network learns the minimum-energy correction to the physics model that is consistent with the measured data.

### 4.2 Physics Warm-Up Schedule

At the beginning of training, network weights are random and $\mathcal{L}_{phys}$ generates gradient signals that are unrelated to the true task. Applying the full physics penalty immediately would interfere with the network first organizing its representation around the data-fidelity objective. To avoid this, $\alpha_{eff}$ is linearly ramped from zero to the target value $\alpha_0$ over the first $e_{warm}^{phys}$ epochs:
$$
\alpha_{eff}(e) = \alpha_0 \cdot \min\!\left(1,\; \frac{e}{e_{warm}^{phys}}\right), \quad e_{warm}^{phys} = \max\!\bigl(1,\, \lfloor f_{warm} \cdot E \rfloor\bigr) \tag{10}
$$
with $f_{warm} = 0.05$ and $\alpha_0 = 0.5$. For $E = 100$ training epochs, $e_{warm}^{phys} = 5$.

During the ramp ($e < e_{warm}^{phys}$), the gradient is:
$$
\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{data} + \frac{e}{e_{warm}^{phys}}\, \alpha_0\; \nabla_\theta \mathcal{L}_{phys}
$$
The first term dominates early in training (when the model's fit to data is far from converged), and the second term grows as the network enters the fine-tuning regime where data-consistent corrections should also be physics-consistent.

---

## 5. Training Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| Hidden layers $[d_1, d_2, d_3]$ | $[256,\, 512,\, 256]$ | Same expansion–compression structure as Model II |
| Input dimension $d_0$ | $35$ | $7J = 7 \times 5$ augmented features |
| Activation $\sigma$ | SiLU | Identical to Model II |
| Normalization | LayerNorm | Identical to Model II |
| Dropout rate $p$ | $0.1$ | Lower than BlackBoxFNN due to physics prior |
| Batch size | $512$ | |
| Optimizer | AdamW | $\beta_1{=}0.9,\,\beta_2{=}0.999$ |
| Learning rate $\eta_0$ | $3 \times 10^{-4}$ | |
| Weight decay | $5 \times 10^{-3}$ | |
| LR schedule | Warm-up cosine | $e_{warm} = \lfloor E/20 \rfloor$ |
| Gradient clip norm | $5.0$ | |
| Feature noise std $\sigma_n$ | $0.02$ | Applied to full input $x$ |
| Physics weight $\alpha_0$ | $0.5$ | Target physics regularization strength |
| Physics warm-up fraction $f_{warm}$ | $0.05$ | 5% of $E$ epochs for ramp |
| Early stopping patience $P$ | $80$ epochs | Monitors val RMSE (N·m) |
| Min improvement $\delta_{min}$ | $10^{-4}$ N·m | |
| Max epochs $E$ | $100$ | |
| Training samples $N$ | $272{,}465$ | Identical dataset to Models II and IV |

---

## 6. Theoretical Properties

**Physics prior as Bayesian regularizer.** From a probabilistic perspective, the physics regularization term $\mathcal{L}_{phys}$ is proportional to the negative log-likelihood of a Gaussian prior centered at $\tilde{\tau}_{phys}$ with isotropic variance $\sigma^2_{phys} = 1/(2\alpha_{eff})$. The combined loss Eq. (7) corresponds to maximum a posteriori (MAP) estimation:
$$
\theta^* = \arg\min_\theta \bigl[-\log p(\tilde{\tau}^* \mid \theta) - \log p(\theta)\bigr]
$$
where $p(\theta)$ encodes the belief that the network prediction is close to $\tilde{\tau}_{phys}$. As $\alpha_0 \to 0$, the prior becomes uninformative and the model reduces to Model II. As $\alpha_0 \to \infty$, the posterior collapses to the prior: $\hat{\tau}_{norm} \to \tilde{\tau}_{phys}$ (the pure analytical model).

**Sample efficiency in low-data regimes.** When training data is scarce, $\mathcal{L}_{phys}$ counteracts overfitting by preventing the network from fitting noise: the physics gradient pulls predictions toward $\tilde{\tau}_{phys}$ when the data gradient would otherwise pull them toward spurious local patterns. This acts as physics-informed weight decay in prediction space — the network's predictions are regularized toward the physics baseline regardless of the parameter norm.

**No hard architectural constraint.** Unlike Model IV, the physics prediction $\tau_{phys}$ does not appear directly in the output equation. The network is free to produce any $\hat{\tau}_{norm} \in \mathbb{R}^J$ during inference; physical consistency is enforced only through gradient signals during training. Consequently, removing the physics penalty at inference time (equivalently, evaluating the trained model on a new dataset) provides no guarantee that the outputs will remain physically plausible.

**Gradient identity.** The total training gradient decomposes linearly:
$$
\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{data} + \alpha_{eff}\, \nabla_\theta \mathcal{L}_{phys} = \nabla_\theta \text{MSE}(\hat{\tau},\tilde{\tau}^*) + \alpha_{eff}\, \nabla_\theta \text{MSE}(\hat{\tau}, \tilde{\tau}_{phys}) \tag{11}
$$
This shows that the optimizer is simultaneously pulling the prediction toward the measured torques and toward the analytical prediction. When both signals agree (i.e., where RNEA is accurate), the two gradient components reinforce each other. Where they disagree (systematic model errors), the data gradient eventually dominates as the network learns to correct the physics baseline.

**Experimental results.** With physics weight $\alpha_0 = 0.5$ and the training configuration above, the PhysicsRegularizedFNN achieves a mean test RMSE of $\overline{\text{RMSE}} = 0.0883$ N·m and mean per-joint $R^2 = 0.870$, surpassing the BlackBoxFNN by 6.9\% in RMSE. Per-joint results are shown below.

| Joint | Role | Test RMSE (N·m) | $R^2$ | $\Delta$RMSE vs. Model II |
|---|---|---|---|---|
| J1 | Yaw (base) | $0.0725$ | $0.880$ | $+4.3\%$ worse |
| J2 | Shoulder pitch | $0.1560$ | $0.829$ | $-9.9\%$ better |
| J3 | Elbow pitch | $0.0910$ | $0.914$ | $-11.8\%$ better |
| J4 | Wrist pitch | $0.0374$ | $0.888$ | $-3.9\%$ better |
| J5 | Wrist roll | $0.0848$ | $0.838$ | $-5.0\%$ better |
| **Mean** | — | $\mathbf{0.0883}$ | $\mathbf{0.870}$ | $\mathbf{-6.9\%}$ |

The physics regularization most benefits J3 and J2 — the joints with the largest gravitational and inertial contributions — where the RNEA gravity prediction is most accurate. J1 shows a marginal degradation, which may indicate that the joint-weighted physics penalty is slightly over-regularizing the yaw joint where friction dominates and RNEA is less accurate.
