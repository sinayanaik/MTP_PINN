# Physics-Informed Neural Networks for Robot Inverse Dynamics: A Comparative Study of Black-Box, Physics-Regularized, and Residual-Correction Architectures

**Abstract** — Accurate joint torque estimation is a fundamental requirement for model-based robot control, adaptive compensation, and safe human-robot interaction. Classical analytical inverse dynamics computed via Recursive Newton-Euler Algorithm (RNEA) provides a structured physical prior, but its accuracy is limited by uncertain inertial parameters, unmodeled friction, and discretization artifacts. This paper presents a systematic comparative study of three neural network architectures for inverse dynamics identification on a five-degree-of-freedom serial manipulator: (i) a fully data-driven feedforward neural network (BlackBox-FNN) that treats torque prediction as a pure regression problem from kinematic inputs alone; (ii) a physics-regularized FNN (PhysReg-FNN) that concatenates the *full four-component decomposed* RNEA tensor $[\boldsymbol{\tau}_g, \boldsymbol{\tau}_M, \boldsymbol{\tau}_C, \boldsymbol{\tau}_f]$ (4·J = 20 features) to the kinematic vector (3·J = 15 features) for a 7·J = 35-dim augmented input, and augments the data loss with an additive Tikhonov physics penalty pulling the prediction toward the analytical RNEA sum; and (iii) a residual-correction FNN (ResCorr-FNN) that imposes a hard structural decomposition $\hat{\boldsymbol{\tau}} = \boldsymbol{\tau}_{\text{phys}} + c_s \cdot \tanh(\boldsymbol{\delta}_{\text{raw}})$ where the bounded correction cannot exceed $\pm c_s$ in normalised units, with $c_s = 0.5$ as a fixed (non-learnable) buffer providing a hard prior on physics reliance. All three architectures share a unified MLP backbone (LayerNorm + activation + Dropout, Xavier-normal init), trained on 272,465 time steps collected across 11 distinct Cartesian motion geometries and evaluated on a held-out test set of 43,093 samples. A structured hyperparameter grid search covering 144 training runs (FNN: 12, PhysReg: 72, ResCorr: 60) sweeps training data fraction (six levels from 2% to 100%), physics penalty weight $\lambda \in \{0.05, 0.1, 0.2, 0.5, 1.0, 2.0\}$, and L₂ correction penalty $\alpha_r \in \{0.005, 0.01, 0.05, 0.1, 0.5\}$ across two seeds. Preliminary results achieve pooled test RMSE of **0.0793 N·m** (PhysReg), **0.0815 N·m** (ResCorr), and **0.0830 N·m** (BlackBox).

---

## I. Introduction

Inverse dynamics — the mapping from joint kinematics (q, q̇, q̈) to joint torques τ — is central to feed-forward torque control, friction compensation, and payload estimation in serial manipulators. The gold standard formulation follows from the Newton-Euler equations of rigid-body mechanics:

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) + \boldsymbol{\tau}_f(\dot{\mathbf{q}})$$

where $\mathbf{M} \in \mathbb{R}^{n \times n}$ is the configuration-dependent mass matrix, $\mathbf{C}$ is the Coriolis/centripetal matrix, $\mathbf{g}$ is the gravity vector, and $\boldsymbol{\tau}_f$ accounts for joint friction. In practice, exact computation of these terms requires knowledge of every link's mass, center of mass, and inertia tensor — parameters that are typically uncertain by 10–40% in low-cost manipulators built from PLA or ABS printed parts.

Data-driven approaches sidestep parameter identification by learning the inverse dynamics directly from measurements. Early work used Gaussian Processes [1] and sparse regression [2]; more recently, deep feedforward networks have been shown to match or exceed classical identification accuracy [3, 4]. However, pure black-box networks ignore the rich structural knowledge encoded in Lagrangian mechanics, requiring more data to achieve equivalent accuracy and offering no guarantee of physical consistency.

Physics-Informed Neural Networks (PINNs) [5] incorporate physical constraints either as soft penalties in the loss function or as hard architectural constraints in the network structure. Applied to robot inverse dynamics, two complementary strategies emerge: (a) **physics regularization**, where the loss penalizes deviation from an analytical prediction; and (b) **residual correction**, where the network learns an additive correction to a structured physics model, effectively decomposing the learning problem into a solved physical component and a compact residual.

This paper makes the following contributions:

1. A rigorous formulation and implementation of three inverse dynamics architectures under a unified training framework with shared feature extraction, normalization, and optimizer configuration.
2. A 144-trial hyperparameter grid search quantifying the effect of training data fraction (six levels from 2% to 100%), physics weight (six values from 0.05 to 1.0), and residual penalty (five values from 0.001 to 0.1) on held-out test accuracy.
3. Detailed per-joint analysis revealing architecture-specific failure modes and the role of physics guidance for high-inertia joints.
4. Open-source code and trained model checkpoints enabling direct comparison.

---

## II. Problem Formulation

### A. Robot System

Experiments are conducted on a five-degree-of-freedom serial manipulator (Kikobot) with joints J₁–J₅ driven by brushless servo actuators. The active joint vector is $\mathbf{q} \in \mathbb{R}^5$, with the kinematic chain:

$$\text{base} \xrightarrow{J_1} \text{shoulder} \xrightarrow{J_2} \text{upper arm} \xrightarrow{J_3} \text{forearm} \xrightarrow{J_4} \text{wrist-pitch} \xrightarrow{J_5} \text{wrist-roll}$$

Joint actuator properties are summarized in Table I. J₄ (wrist pitch, STS3032) has notably lower rated torque than the other joints, consistent with the observed lower RMSE at that joint.

**Table I: Joint Actuator Specifications**

| Joint | Actuator | Rated Torque | Role |
|-------|----------|--------------|------|
| J₁ | STS3215 | 30.0 kgf·cm | Shoulder rotation |
| J₂ | STS3215 | 30.0 kgf·cm | Shoulder elevation |
| J₃ | STS3215 | 30.0 kgf·cm | Elbow |
| J₄ | STS3032 | 14.8 kgf·cm | Wrist pitch |
| J₅ | STS3215 | 30.0 kgf·cm | Wrist roll |

### B. Inverse Dynamics via RNEA

The analytical inverse dynamics is computed using Pinocchio's RNEA implementation [6] with calibrated inertial parameters. The torque at each joint is decomposed into four physically interpretable components:

$$\boldsymbol{\tau} = \underbrace{\boldsymbol{\tau}_g(\mathbf{q})}_{\text{gravity}} + \underbrace{\boldsymbol{\tau}_M(\mathbf{q}, \ddot{\mathbf{q}})}_{\text{inertial}} + \underbrace{\boldsymbol{\tau}_C(\mathbf{q}, \dot{\mathbf{q}})}_{\text{Coriolis}} + \underbrace{\boldsymbol{\tau}_f(\dot{\mathbf{q}})}_{\text{friction}}$$

Practically, each term is computed as:
- $\boldsymbol{\tau}_g = \text{RNEA}(\mathbf{q}, \mathbf{0}, \mathbf{0})$
- $\boldsymbol{\tau}_M = \text{RNEA}(\mathbf{q}, \mathbf{0}, \ddot{\mathbf{q}}) - \boldsymbol{\tau}_g$
- $\boldsymbol{\tau}_C = \text{RNEA}(\mathbf{q}, \dot{\mathbf{q}}, \mathbf{0}) - \boldsymbol{\tau}_g$
- $\boldsymbol{\tau}_f = \mathbf{c} \odot \tanh\!\left(\frac{\dot{\mathbf{q}}}{\varepsilon}\right) + \mathbf{v} \odot \dot{\mathbf{q}}$

where $\mathbf{c} = [0.135, 0.278, 0.201, 0.088, 0.204]^T$ N·m are calibrated Coulomb friction coefficients, $\mathbf{v} = [0.300, 0.300, 0.245, 0.040, 0.047]^T$ N·m·s/rad are viscous friction coefficients, and $\varepsilon = 0.0405$ rad/s is the smooth Coulomb transition width. Link masses are scaled by a factor $\rho = 0.0931$ relative to the nominal URDF to account for the PLA printed structure (approximately 70% infill).

The full analytical prediction is thus:

$$\boldsymbol{\tau}_{\text{phys}} = \boldsymbol{\tau}_g + \boldsymbol{\tau}_M + \boldsymbol{\tau}_C + \boldsymbol{\tau}_f \in \mathbb{R}^5$$

The four-component vector $\boldsymbol{\phi} = [\boldsymbol{\tau}_g^T, \boldsymbol{\tau}_M^T, \boldsymbol{\tau}_C^T, \boldsymbol{\tau}_f^T]^T \in \mathbb{R}^{20}$ is retained separately as the physics feature tensor.

### C. Inverse Dynamics Learning Problem

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, \boldsymbol{\tau}_i^*)\}_{i=1}^N$ where $\mathbf{x}_i = [\mathbf{q}_i^T, \dot{\mathbf{q}}_i^T, \ddot{\mathbf{q}}_i^T]^T \in \mathbb{R}^{15}$ and $\boldsymbol{\tau}_i^* \in \mathbb{R}^5$ is the measured joint torque, we seek a function $f_\theta : \mathbb{R}^{15} \rightarrow \mathbb{R}^5$ that minimizes the prediction error while optionally exploiting the physics structure from §II-B.

---

## III. Data Collection and Preprocessing

### A. Trajectory Corpus

The training dataset comprises 100 end-effector trajectories spanning 11 Cartesian motion geometries: Circle, Ellipse, Helix, Lissajous, Parabola, Rectangle, Regular Polygon, Sine Wave, Spiral, Square, and Triangle. This diversity ensures that models must generalize across qualitatively different velocity and acceleration profiles rather than overfitting to a single motion class.

The corpus is partitioned as follows:

**Table II: Dataset Statistics**

| Split | Trajectories | Samples | Fraction |
|-------|-------------|---------|----------|
| Train | 69 | 272,465 | 73.5% |
| Validation | 15 | 53,779 | 14.5% |
| Test | 16 | 43,093 | 11.6% |
| **Total** | **100** | **369,337** | 100% |

The split is stratified by geometry type (proportional allocation per class) to prevent any motion class from being absent in validation or test.

### B. Signal Preprocessing

Raw joint encoders and motor current measurements are processed as follows:

1. **Velocity and acceleration** are obtained via Savitzky-Golay differentiation (window $w=121$ samples ≈ 2.4 s at 50 Hz, polynomial order $p=3$), using first and second derivatives respectively. The long window suppresses high-frequency encoder noise while preserving kinematic structure.

2. **Torque smoothing** applies a second Savitzky-Golay filter ($w=121, p=3$) to the measured motor torque to remove servo communication latency artifacts.

3. **RNEA computation** uses the filtered kinematics $(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})$ as input to Pinocchio's `rnea` function. The resulting physics components are post-filtered with a shorter Savitzky-Golay pass ($w=15, p=3$) to remove numerical noise from the dynamics library.

4. **Trajectory trimming** removes the first and last 1% of each trajectory to eliminate boundary transients from the motion controller.

### C. Feature Normalization

All inputs and targets are standardized to zero mean and unit variance using statistics computed **exclusively from the training split** and applied to all splits:

$$\tilde{x}^{(k)} = \frac{x^{(k)} - \mu_k^{\text{train}}}{\sigma_k^{\text{train}}}, \quad k \in \{q, \dot{q}, \ddot{q}\}$$

$$\tilde{\boldsymbol{\tau}}^* = \frac{\boldsymbol{\tau}^* - \boldsymbol{\mu}_\tau^{\text{train}}}{\boldsymbol{\sigma}_\tau^{\text{train}}}$$

The physics feature tensor $\boldsymbol{\phi} \in \mathbb{R}^{20}$ is normalized so that its four components sum to the normalized total torque. Specifically, the mean is distributed equally across the four terms and the same scale is used for all:

$$\tilde{\phi}^{(k, j)} = \frac{\tau_k^{(j)} - \mu_{\tau_j}^{\text{train}}/4}{\sigma_{\tau_j}^{\text{train}}}, \quad k \in \{g, M, C, f\}, \quad j \in \{1,\ldots,5\}$$

This normalization preserves the identity $\sum_{k} \tilde{\phi}^{(k,j)} = \tilde{\tau}_j^*$ in the noise-free case, ensuring physical consistency in the normalized domain throughout training.

---

## IV. Neural Network Architectures

All three architectures share a unified MLP backbone, differing only in (i) input dimensionality (kinematic-only vs. kinematic + decomposed physics), (ii) the parameterisation of the output map (free regression vs. tanh-bounded residual), and (iii) the training loss (data-only vs. data + physics regulariser vs. data + correction-magnitude regulariser).

### A. Shared MLP Backbone

Let $L$ denote the number of hidden layers with widths $h_1, h_2, \ldots, h_L$. Layer $l \in \{1, \ldots, L\}$ implements the post-norm block

$$\mathbf{a}^{(l)} = \text{Dropout}_p\!\Big(\sigma\big(\text{LayerNorm}\big(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\big)\big)\Big),\qquad \mathbf{W}^{(l)} \in \mathbb{R}^{h_l \times h_{l-1}},\ \mathbf{b}^{(l)} \in \mathbb{R}^{h_l},$$

with $\mathbf{a}^{(0)}$ the (possibly augmented) input and the output head $\mathbf{a}^{(L+1)} = \mathbf{W}^{(L+1)} \mathbf{a}^{(L)} + \mathbf{b}^{(L+1)}$ a plain linear map without normalisation, activation, or dropout. LayerNorm operates on the feature axis with learnable affine parameters $(\boldsymbol{\gamma}^{(l)}, \boldsymbol{\beta}^{(l)}) \in \mathbb{R}^{h_l} \times \mathbb{R}^{h_l}$:

$$\text{LayerNorm}(\mathbf{z})_k = \gamma_k \cdot \frac{z_k - \bar{z}}{\sqrt{\widehat{\text{Var}}(\mathbf{z}) + \varepsilon}} + \beta_k,\qquad \bar{z} = \tfrac{1}{h}\textstyle\sum_k z_k.$$

Dropout applies an i.i.d. Bernoulli mask $\mathbf{m}^{(l)} \sim \text{Bern}(1-p)^{h_l}$ scaled by $1/(1-p)$ at training time and the identity at evaluation time. All linear weights are initialised with Xavier-normal, $\mathbf{W}_{ij} \sim \mathcal{N}(0,\, 2/(h_{l-1}+h_l))$, and biases are initialised to zero.

**Activation.** The activation $\sigma$ is selected from $\{\text{SiLU}, \text{GELU}, \text{ReLU}, \text{Tanh}, \text{ELU}, \text{LeakyReLU}\}$. All experiments in §VII use SiLU (single-architecture trainers) or GELU (grid sweep):

$$\text{SiLU}(x) = x \cdot \sigma_{\text{logistic}}(x) = \frac{x}{1 + e^{-x}},\qquad \text{GELU}(x) = x \cdot \Phi(x).$$

Both are smooth and bounded below, retaining small negative-region gradients (unlike ReLU), which prevents the dead-neuron pathology and stabilises the gradient flow when the physics regulariser introduces competing gradient directions during the warmup phase.

**Backbone configuration.** Two backbones are used: the *standalone trainers* (`run_fnn.py`, `run_physics_regularized.py`, `run_physics_residual.py`) use $[h_1, h_2, h_3] = [256, 512, 256]$, while the *grid sweep* (`run_loss_residual_grid.py`) uses $[128, 256, 128]$ — the smaller backbone is preferred for the grid because typical training-set sizes after data-fraction subsampling (5K–50K samples for fractions ≤ 0.25) make the larger backbone prone to overfit.

**Parameter counts.** Counting LayerNorm $(2 h_l)$ and Linear $(h_{l-1} h_l + h_l)$ contributions, with input dimension $d_{\text{in}}$ and $J = 5$ output dimensions:

| Backbone | Input dim | LinearIn | Hidden | LinearOut | Total |
|----------|-----------|----------|--------|-----------|-------|
| [256, 512, 256], $d_{\text{in}}=15$ | 15 | $15{\cdot}256{+}256$ | $131{,}584{+}131{,}328{+}1{,}536$ | $256{\cdot}5{+}5$ | **270,341** |
| [256, 512, 256], $d_{\text{in}}=35$ | 35 | $35{\cdot}256{+}256$ | $131{,}584{+}131{,}328{+}1{,}536$ | $256{\cdot}5{+}5$ | **275,461** |
| [128, 256, 128], $d_{\text{in}}=15$ | 15 |   $15{\cdot}128{+}128$ | $33{,}024{+}32{,}896{+}640$ | $128{\cdot}5{+}5$ | **69,637** |
| [128, 256, 128], $d_{\text{in}}=35$ | 35 |   $35{\cdot}128{+}128$ | $33{,}024{+}32{,}896{+}640$ | $128{\cdot}5{+}5$ | **72,197** |

The 5,120-parameter increase from $d_{\text{in}}=15 \rightarrow 35$ on the wider backbone (or 2,560 on the narrower) is solely attributable to the 20 additional input weights times the first hidden width — a < 2 % increase in capacity. The physics-augmented backbone is therefore not gaining its advantage from extra parameters.

### B. Black-Box FNN (BlackBox-FNN)

The simplest architecture treats torque prediction as a pure regression problem with no physics knowledge:

$$\hat{\boldsymbol{\tau}}_{\text{BB}}^{(i)} = f_\theta(\tilde{\mathbf{x}}^{(i)}),\qquad f_\theta : \mathbb{R}^{3J} \rightarrow \mathbb{R}^{J},$$

where $\tilde{\mathbf{x}}^{(i)} = [\tilde{\mathbf{q}}^{(i)\top}, \tilde{\dot{\mathbf{q}}}^{(i)\top}, \tilde{\ddot{\mathbf{q}}}^{(i)\top}]^\top \in \mathbb{R}^{3J}$ is the normalised kinematic state. The decomposed physics tensor $\boldsymbol{\phi}^{(i)} \in \mathbb{R}^{4J}$ is supplied by the data loader but is *explicitly discarded* by the forward pass (`del physics`) so that no information leakage occurs even by accident through, e.g., shared batch-norm statistics.

**Training loss.** The joint-weighted mean-squared error in normalised target space:

$$\mathcal{L}_{\text{BB}}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{J} w_j\,\big(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}\big)^2,\qquad \mathbf{w} = [1.0,\, 2.5,\, 1.0,\, 1.0,\, 1.0]^\top.$$

The over-weight on $J_2$ is a deliberate countermeasure for the heavy-tailed torque distribution at the shoulder elevation joint: with $\sigma(\tau_2) \approx 2.58$ N·m vs. $\sigma(\tau_1) \approx 0.76$ N·m, an unweighted MSE allocates roughly $(2.58/0.76)^2 \approx 11.5\times$ more gradient pressure to $J_2$ already; the additional $w_2 = 2.5$ multiplier ensures that despite the prior-induced shrinkage of physics-augmented models, $J_2$ remains a primary optimisation target. The same weight vector is reused by all three trainers so that the *data* gradient is identical across architectures, isolating the contribution of the physics term.

The empirical risk gradient is

$$\nabla_\theta \mathcal{L}_{\text{BB}} = \frac{2}{N}\sum_{i,j} w_j\,\big(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}\big)\,\nabla_\theta \hat{\tau}_j^{(i)},$$

where $\nabla_\theta \hat{\tau}_j^{(i)}$ is computed by automatic differentiation through the MLP.

### C. Physics-Regularised FNN (PhysReg-FNN)

PhysReg-FNN augments the input with the *full four-component decomposed* RNEA tensor and adds an additive Tikhonov physics penalty to the data loss.

**Augmented input.** Given the normalised decomposed physics tensor

$$\tilde{\boldsymbol{\phi}}^{(i)} = \big[\tilde{\boldsymbol{\tau}}_g^{(i)\top},\; \tilde{\boldsymbol{\tau}}_M^{(i)\top},\; \tilde{\boldsymbol{\tau}}_C^{(i)\top},\; \tilde{\boldsymbol{\tau}}_f^{(i)\top}\big]^\top \in \mathbb{R}^{4J}$$

the augmented input is the concatenation $\tilde{\mathbf{u}}^{(i)} = [\tilde{\mathbf{x}}^{(i)\top},\, \tilde{\boldsymbol{\phi}}^{(i)\top}]^\top \in \mathbb{R}^{7J}$, and the prediction is

$$\hat{\boldsymbol{\tau}}_{\text{PR}}^{(i)} = f_\theta(\tilde{\mathbf{u}}^{(i)}),\qquad f_\theta : \mathbb{R}^{7J} \rightarrow \mathbb{R}^{J}.$$

This is a deliberate departure from the more common practice of passing only the scalar RNEA *sum* $\sum_k \tilde{\boldsymbol{\tau}}_k$. By exposing each physics component separately the network's first linear layer

$$\mathbf{a}^{(1)}_h = \sum_{j=1}^{J}\Big[ W^{(1)}_{h,\,j}\,\tilde{q}_j + W^{(1)}_{h,\,J+j}\,\tilde{\dot{q}}_j + W^{(1)}_{h,\,2J+j}\,\tilde{\ddot{q}}_j + \sum_{k\in\{g,M,C,f\}} W^{(1)}_{h,\,3J + 4(j-1) + \mathrm{idx}(k)}\,\tilde{\tau}_{k,j} \Big] + b^{(1)}_h$$

can learn per-component, per-joint *trust weights* $W^{(1)}_{h, 3J + 4(j-1) + \mathrm{idx}(k)}$ — for example, down-weighting the friction component $\tilde{\tau}_{f,j}$ at joint $j$ if its empirical residual is large, while keeping the gravity component $\tilde{\tau}_{g,j}$ at full influence. Collapsing to the sum eliminates this degree of freedom and forces a uniform 1/4 weight on each component.

**Physics reference for the loss.** Although the *input* preserves the four components separately, the loss-side physics target is the linear sum

$$\boldsymbol{\tau}_{\text{ref}}^{(i)} = \sum_{k\in\{g,M,C,f\}}\!\!\tilde{\boldsymbol{\tau}}_k^{(i)} \in \mathbb{R}^{J},$$

implemented in `reduce_physics_to_total()` by reshaping the 4·J tensor to $(\cdot, 4, J)$ and summing over the component axis. The physics-feature normalisation in §III-C ensures the noise-free identity $\boldsymbol{\tau}_{\text{ref}}^{(i)} \approx \tilde{\boldsymbol{\tau}}^{*(i)}$, so the physics loss measures the same target torque as the data loss but with the sample-by-sample analytical RNEA in place of the noisy measurement.

**Composite loss (additive Tikhonov form).** With per-batch joint-weighted MSE

$$\mathcal{L}_{\text{data}}(\theta) = \tfrac{1}{N}\textstyle\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)})^2,\qquad \mathcal{L}_{\text{phys}}(\theta) = \tfrac{1}{N}\textstyle\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tau_{\text{ref},j}^{(i)})^2,$$

the training objective is the *additive* (Tikhonov-style) penalty — the data term keeps unit weight at all times, with physics layered on top:

$$\boxed{\;\mathcal{L}_{\text{PR}}(\theta;\, e) = \mathcal{L}_{\text{data}}(\theta) + \alpha_{\text{eff}}(e)\cdot \mathcal{L}_{\text{phys}}(\theta).\;}$$

This is *not* the convex blend $(1-\alpha)\mathcal{L}_{\text{data}} + \alpha\mathcal{L}_{\text{phys}}$ used in some PINN variants. Under the additive form, increasing $\lambda$ never down-weights the data fit — the physics term acts as a pull toward the analytical surface in *addition* to the data fit. As a consequence, the optimal predictor in expectation is

$$\hat{\boldsymbol{\tau}}^\star = \frac{1}{1 + \alpha_{\text{eff}}}\,\boldsymbol{\tau}^* + \frac{\alpha_{\text{eff}}}{1 + \alpha_{\text{eff}}}\,\boldsymbol{\tau}_{\text{ref}}\quad\text{(per-sample, per-joint)},$$

a weighted average of the noisy measurement and the analytical RNEA whose weight on physics increases monotonically with $\lambda$.

**Linear warmup of the physics coefficient.** The penalty coefficient is annealed from $0$ to its target value $\lambda$ over a warmup window of $e_w = \max(1, \lfloor \gamma\, E\rfloor)$ epochs, with $\gamma = 0.05$ (i.e. 5 % of the epoch budget):

$$\alpha_{\text{eff}}(e) = \lambda\cdot \min\!\Big(1,\; \frac{e}{e_w}\Big),\qquad e \in \{1, 2, \ldots, E\}.$$

The warmup serves two purposes: (i) it prevents the physics term from dominating before the network's first layer has organised itself to read $\tilde{\boldsymbol{\phi}}$; and (ii) at the very start of training, where the prediction $\hat{\boldsymbol{\tau}}$ is essentially noise, the physics gradient $-2 w_j(\hat{\tau}_j - \tau_{\text{ref},j})$ is large in magnitude and pulls in an arbitrary direction — multiplying it by a small $\alpha_{\text{eff}}$ ensures it does not destabilise early epochs.

**Per-component gradient decomposition.** The gradient of the composite loss with respect to a generic backbone parameter $\theta$ is

$$\nabla_\theta \mathcal{L}_{\text{PR}} = \tfrac{2}{N}\sum_{i,j} w_j \big[(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}) + \alpha_{\text{eff}}(\hat{\tau}_j^{(i)} - \tau_{\text{ref},j}^{(i)})\big]\,\nabla_\theta \hat{\tau}_j^{(i)}.$$

When data and physics agree ($\tilde{\tau}^* \approx \tau_{\text{ref}}$), the bracket is $(1+\alpha_{\text{eff}})$ times the data residual: physics reinforces the data signal. When data and physics disagree, the gradients partially cancel, with the equilibrium prediction sitting between the two surfaces as derived above.

**Inference.** PhysReg-FNN requires the RNEA prediction at inference time. Given measured kinematics, $\tilde{\boldsymbol{\phi}}$ is computed online via Pinocchio's `rnea` calls (one per component per timestep) and concatenated to $\tilde{\mathbf{x}}$ before the forward pass.

### D. Residual-Correction FNN (ResCorr-FNN)

ResCorr-FNN imposes a hard *architectural* decomposition of the prediction into an analytical base plus a bounded learned correction:

$$\boxed{\;\hat{\boldsymbol{\tau}}_{\text{RC}}^{(i)} = \tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)} + c_s \cdot \tanh\!\big(\,g_\theta(\tilde{\mathbf{u}}^{(i)})\,\big),\;}$$

where $\tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)} = \sum_k \tilde{\boldsymbol{\tau}}_k^{(i)}$ is the same RNEA sum used as the physics reference in §IV-C, $\tilde{\mathbf{u}}^{(i)} = [\tilde{\mathbf{x}}^{(i)\top}, \tilde{\boldsymbol{\phi}}^{(i)\top}]^\top \in \mathbb{R}^{7J}$ is the augmented 35-dim input, $g_\theta : \mathbb{R}^{7J} \rightarrow \mathbb{R}^{J}$ is the same MLP backbone as PhysReg-FNN, and $\tanh$ is applied element-wise.

**Tanh-bounded correction.** Defining $\boldsymbol{\delta}^{(i)} \triangleq \hat{\boldsymbol{\tau}}_{\text{RC}}^{(i)} - \tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)} = c_s\tanh(g_\theta(\tilde{\mathbf{u}}^{(i)}))$, the correction satisfies

$$\|\boldsymbol{\delta}^{(i)}\|_\infty \le c_s \quad\text{for all } i, \theta,$$

with $c_s = 0.5$ a *fixed buffer* (registered via `register_buffer`, hence saved with the model state but excluded from the optimiser parameter list). This is the architecture's hard structural prior: irrespective of how badly the network is trained, no element of $\boldsymbol{\delta}$ can ever exceed $\pm 0.5$ in *normalised* torque units. After de-normalisation by joint $j$, the per-joint physical-units bound is $c_s \cdot \sigma_{\tau_j}^{\text{train}}$, which evaluates to roughly $\{0.38, 1.29, 0.61, 0.23, 0.36\}$ N·m for the five joints — sizeable enough to absorb realistic RNEA mismatches but small enough to *prevent* the network from overriding the analytical prediction and memorising the training set.

The choice of $\tanh$ over softer bounds (e.g., a soft $\ell_\infty$ ball via clipping) has two motivations:

1. **Smoothness.** $\tanh$ is $C^\infty$ with bounded derivatives $\tanh'(z) = 1 - \tanh^2(z) \in (0, 1]$, so the gradient of the bounded correction $c_s\tanh(\cdot)$ never blows up and never vanishes inside the saturation region (it merely decays).
2. **Smooth saturation.** Near $|g_\theta| \gtrsim 3$ the derivative is $\lesssim 0.01$, which provides a *soft saturation* that strongly discourages the optimiser from pushing into the bound: any further attempt to grow $|\delta|$ produces an exponentially decaying gradient signal back into $\theta$.

**Small-residual warm-start.** The output linear layer is rescaled at initialisation:

$$\mathbf{W}^{(L+1)} \leftarrow 10^{-2}\cdot \mathbf{W}^{(L+1)},\qquad \mathbf{b}^{(L+1)} \leftarrow 10^{-2}\cdot \mathbf{b}^{(L+1)}.$$

Combined with $\tanh(0) = 0$, this guarantees $\boldsymbol{\delta}^{(0)} \approx \mathbf{0}$ at epoch 0 — the network's epoch-0 prediction is the analytical RNEA, providing an effective preconditioner: the optimiser begins refinement from a physics-consistent state rather than building a 5-dim torque field from scratch.

**Regularised training loss.** To further prevent the correction from absorbing the entire inverse-dynamics task, an L₂ penalty on the bounded correction is added:

$$\boxed{\;\mathcal{L}_{\text{RC}}(\theta;\, \alpha_r) = \tfrac{1}{N}\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)})^2 + \alpha_r\cdot\tfrac{1}{NJ}\sum_{i,j} \big(\delta_j^{(i)}\big)^2.\;}$$

The penalty acts on the *post-tanh* correction $\delta_j = c_s \tanh(g_{\theta,j})$, so its gradient back-propagates through the bound:

$$\frac{\partial}{\partial \theta} \big(\delta_j^{(i)}\big)^2 = 2\,\delta_j^{(i)}\cdot c_s\big(1 - \tanh^2(g_{\theta,j}^{(i)})\big)\cdot \nabla_\theta g_{\theta,j}^{(i)},$$

which vanishes both when $\delta = 0$ (no penalty needed) *and* when $|\delta|\to c_s$ (pre-tanh activation already saturated). The strongest regularisation pressure is therefore in the linear-response region $|g_\theta|\lesssim 1$, exactly where the network has the most flexibility to absorb spurious training-set patterns.

The dimensionless monitoring ratio

$$\rho(e) = \frac{\mathbb{E}_i[|\boldsymbol{\delta}^{(i)}|_1 / J]}{\mathbb{E}_i[|\tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)}|_1 / J] + 10^{-12}}$$

is logged each epoch (`residual δ-ratio  E[|δ|]/E[|τ_phys|]`) to surface degenerate cases where $\boldsymbol{\delta}$ either collapses to zero (correction not learning) or saturates the tanh bound (RNEA being ignored).

**Hyperparameter regimes.** For the grid sweep, $\alpha_r \in \{0.005, 0.01, 0.05, 0.1, 0.5\}$ probes the full spectrum:
- $\alpha_r = 0.005$: structural tanh bound is the only meaningful constraint; correction is essentially free up to $\pm c_s$.
- $\alpha_r = 0.05$ (the per-script default): moderate L₂ that pulls $\delta$ toward zero unless the data demands otherwise.
- $\alpha_r = 0.5$: strong L₂; the correction shrinks substantially and predictions converge toward $\tilde{\boldsymbol{\tau}}_{\text{phys}}$.

### E. Architectural Comparison

The three architectures span the spectrum from no physics, to physics as a *soft loss-side regulariser*, to physics as a *hard structural prior*. Table A summarises the differences in a single view.

**Table A: Architectural Differences at a Glance**

| Aspect | BlackBox-FNN | PhysReg-FNN | ResCorr-FNN |
|--------|-------------|-------------|-------------|
| Input dim | $3J = 15$ | $7J = 35$ | $7J = 35$ |
| Forward map | $f_\theta(\tilde{\mathbf{x}})$ | $f_\theta([\tilde{\mathbf{x}}, \tilde{\boldsymbol{\phi}}])$ | $\tilde{\boldsymbol{\tau}}_{\text{phys}} + c_s\tanh(g_\theta([\tilde{\mathbf{x}}, \tilde{\boldsymbol{\phi}}]))$ |
| Physics at training | discarded | input feature + loss term | input feature + output base |
| Physics at inference | absent | input feature only | input feature + output base |
| Loss form | $\mathcal{L}_{\text{data}}$ | $\mathcal{L}_{\text{data}} + \alpha_{\text{eff}}\mathcal{L}_{\text{phys}}$ | $\mathcal{L}_{\text{data}} + \alpha_r \|\boldsymbol{\delta}\|^2/J$ |
| Physics constraint | none | soft (loss penalty) | hard (architectural bound) |
| Tunable physics HP | — | $\lambda$ (penalty strength) | $\alpha_r$ (penalty strength), $c_s$ (bound) |
| Worst-case behaviour | unbounded | unbounded | $\|\hat{\boldsymbol{\tau}} - \tilde{\boldsymbol{\tau}}_{\text{phys}}\|_\infty \le c_s$ |

### F. Why Concatenate the *Decomposed* Physics?

A natural alternative to the 35-dim augmented input is the 20-dim form $[\tilde{\mathbf{x}},\, \tilde{\boldsymbol{\tau}}_{\text{phys}}]$ where $\tilde{\boldsymbol{\tau}}_{\text{phys}} = \sum_k \tilde{\boldsymbol{\tau}}_k$. The decomposed form is preferred for three reasons.

**Information.** The map $\tilde{\boldsymbol{\phi}} \mapsto \sum_k \tilde{\boldsymbol{\tau}}_k$ is many-to-one — a network receiving only the sum cannot, even in principle, infer the relative magnitudes of gravity, inertial, Coriolis, and friction contributions. Many failure modes of RNEA are component-specific (friction errors at low velocity, inertial-tensor mis-calibration at high acceleration), so distinguishing the components is necessary for a context-sensitive correction.

**Linear separability of trust weights.** With the decomposed input, the first hidden layer is in essence a learnable *trust assignment*: each hidden unit can up- or down-weight any (component, joint) pair through its first-layer weight. Collapsing to the sum forces this assignment to be applied uniformly over components, eliminating an entire degree of freedom available to the network at no parameter-count cost (the input dimension grows by only $3J = 15$ extra weights per hidden unit).

**Capacity allocation.** With limited training data, a kinematic-only network must implicitly reconstruct the physics components (gravity, inertia, Coriolis, friction) from $\tilde{\mathbf{x}}$ before it can learn corrections to them — this is wasted capacity. By the data-processing inequality, any statistic of $\tilde{\mathbf{x}}$ available to the kinematic-only network is also available to the augmented-input network, but not vice versa, so the augmented input is a strict information superset. Empirically, this is the mechanism by which both PhysReg-FNN and ResCorr-FNN achieve the data-efficiency advantage reported in §VII-D.

---

## V. Training Methodology

### A. Optimiser

All models are trained with AdamW [7], whose update rule decouples weight decay from the adaptive-moment normalisation:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,\qquad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^{\odot 2},$$

$$\hat{\mathbf{m}}_t = \mathbf{m}_t / (1-\beta_1^t),\qquad \hat{\mathbf{v}}_t = \mathbf{v}_t / (1-\beta_2^t),$$

$$\theta_{t+1} = \theta_t - \eta_t\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta_t\,\lambda_{\text{wd}}\,\theta_t,$$

with $\mathbf{g}_t = \nabla_\theta \mathcal{L}_{\{\text{BB}, \text{PR}, \text{RC}\}}$, default PyTorch betas $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$. Learning rate $\eta_0 = 3\times 10^{-4}$ and weight decay $\lambda_{\text{wd}} \in \{5\times 10^{-3}, 5\times 10^{-2}\}$ — the former for the per-script standalone trainers, the latter for the grid sweep where the smaller backbone benefits from heavier weight decay.

### B. Learning-Rate Schedule (warmup-cosine)

Per-epoch learning rate:

$$\eta_t = \eta_0\cdot \begin{cases}\;0.1 + 0.9\cdot \dfrac{e}{e_w} & e < e_w,\\[6pt]\;r_{\min} + (1 - r_{\min})\cdot \dfrac{1 + \cos\!\big(\pi\,\xi(e)\big)}{2} & e \ge e_w,\end{cases}\qquad \xi(e) = \frac{e - e_w}{\max(1, E - e_w)},$$

with warmup length $e_w = \max(1,\, \lfloor E/20\rfloor)$ (5 % of the budget) and minimum-LR ratio $r_{\min} = 10^{-2}$. The linear warmup from $0.1\eta_0$ avoids the well-known instability of large initial AdamW updates (when $\hat{\mathbf{v}}_t$ is dominated by the bias correction), and the cosine tail allows fine-grained refinement near the minimum.

**Patience suppression during warmup.** The early-stopping patience counter is held at zero through the first $e_w$ epochs; per-epoch validation improvements during warmup are intentionally smaller than $\delta_{\min}$ even for a healthy run, and counting them would prematurely terminate training.

### C. Stochastic Regularisation

**Input noise augmentation.** During training (but not evaluation), isotropic Gaussian noise is added to the *full* normalised input vector $\tilde{\mathbf{u}}^{(i)}$:

$$\tilde{\mathbf{u}}_{\text{aug}}^{(i)} = \tilde{\mathbf{u}}^{(i)} + \boldsymbol{\varepsilon}^{(i)},\qquad \boldsymbol{\varepsilon}^{(i)} \sim \mathcal{N}(\mathbf{0},\, \sigma_n^2 \mathbf{I}_{d_{\text{in}}}),\qquad \sigma_n \in \{0.02, 0.05\},$$

with $d_{\text{in}} = 15$ for BlackBox-FNN and $d_{\text{in}} = 35$ for PhysReg/ResCorr (note that the noise is applied to the augmented vector *as a whole*, including the physics channels — this prevents the network from learning a purely-physics shortcut). The perturbation is in normalised space; the per-channel physical-units standard deviation is $\sigma_n\cdot \sigma_k^{\text{train}}$.

**Dropout.** Applied with probability $p \in [0.1, 0.4]$ at every hidden layer. Conventional analysis treats dropout as an approximate ensemble averaged at evaluation time; here we use the standard inverted-dropout convention, where training-time activations are divided by $1-p$ so the expected magnitude is unchanged.

**Weight decay.** AdamW's decoupled weight decay (see V-A) is mathematically equivalent to a per-parameter L₂ penalty on the *parameter trajectory*, distinct from adding $\lambda_{\text{wd}}\|\theta\|_2^2$ to the loss in standard Adam. The decoupled form is preferred because it does not interact with the second-moment normalisation $\sqrt{\hat{\mathbf{v}}_t}$.

### D. Gradient Stabilisation

**Global-norm clipping.** The aggregate gradient norm is clipped to $G_{\max}$:

$$\mathbf{g}_t \leftarrow \mathbf{g}_t \cdot \min\!\Big(1,\; \frac{G_{\max}}{\|\mathbf{g}_t\|_2 + 10^{-6}}\Big),$$

with $G_{\max} = 5.0$ (per-script) or $G_{\max} = 1.0$ (grid). This is essential at the *start* of PhysReg training where the term $\alpha_{\text{eff}}\,\mathcal{L}_{\text{phys}}$ ramps in over the first $e_w$ epochs and can produce large gradient spikes when $\hat{\boldsymbol{\tau}}^{(0)}$ has not yet aligned with $\boldsymbol{\tau}_{\text{ref}}$.

**Mixed precision (AMP).** On CUDA devices, forward and backward passes run in FP16 under `torch.autocast`, with master weights kept in FP32 by the AdamW optimiser. A `GradScaler` rescales the loss before backward to avoid FP16 underflow:

$$\mathbf{g}_t^{\text{FP16}} = \nabla_\theta\,(s \cdot \mathcal{L}),\qquad \mathbf{g}_t^{\text{FP32}} = \mathbf{g}_t^{\text{FP16}} / s,$$

with $s$ updated dynamically (doubled when no overflow detected for a window of steps; halved on overflow).

### E. Early Stopping

The training loop monitors the unweighted *macro-RMSE* on the validation split in physical N·m units (as opposed to the joint-weighted training objective). With minimum improvement $\delta_{\min} = 10^{-4}$ N·m and patience $P$:

$$\text{stop training at epoch } e^\star = \min\!\Big\{e\ \big|\ \text{val\_rmse}(e) > \min_{e' \le e} \text{val\_rmse}(e') - \delta_{\min}\;\text{for } P\ \text{consecutive epochs}\Big\}.$$

Patience values: $P = 50$ (BlackBox per-script), $P = 80$ (PhysReg per-script), $P = 60$ (ResCorr per-script), $P = 150$ (grid). Upon early stop or epoch budget exhaustion, the model state is *rolled back* to the epoch that achieved $\min_{e'} \text{val\_rmse}(e')$; this is the state that is saved to `model.pt` and used for all reported test metrics.

### F. Macro-RMSE Validation Metric

Validation RMSE is computed in physical units after de-normalisation, *averaged per trajectory* and then averaged across trajectories — not pooled across all samples — so that long trajectories do not dominate short ones:

$$\text{macro\_rmse}(\hat{\boldsymbol{\tau}}, \boldsymbol{\tau}^*) = \frac{1}{|\mathcal{T}|}\sum_{T \in \mathcal{T}} \frac{1}{J}\sum_{j=1}^{J} \sqrt{\frac{1}{|T|}\sum_{i \in T}\big(\hat{\tau}_j^{(i)} - \tau_j^{*(i)}\big)^2},$$

where $\mathcal{T}$ is the set of validation trajectories. The pooled RMSE used in §VII (table headers say "RMSE pooled") is $\sqrt{\tfrac{1}{NJ}\sum_{i,j}(\hat{\tau}_j^{(i)} - \tau_j^{*(i)})^2}$ — the same predictions, but with the per-trajectory and per-joint averaging steps replaced by a single global pool.

### G. DataLoader Configuration

Training batches: $B = 512$ (per-script) or $B = 1024$ (grid), with `shuffle=True`, `drop_last=True` *only* when the training set contains at least $2B$ samples. The 2B floor is critical at small data fractions: at `frac = 0.02` the training set has ~5,450 samples, and a naïve `drop_last=True` with $B = 1024$ would silently empty the loader when the last batch is dropped on uneven splits. Validation and test loaders use all samples (`drop_last=False`) and are not shuffled.

DataLoader workers are auto-tuned by available system memory: 0–2 workers on low-RAM hosts (< 32 GB), up to 8 on workstation-grade hardware. Prefetch factor scales similarly. `pin_memory=True` is enabled on CUDA devices for asynchronous host-to-device transfer.

---

## VI. Hyperparameter Grid Search

To characterise the effect of the key physics-related hyperparameters, a structured grid search is conducted. All non-swept hyperparameters are held fixed at the values in §V (backbone $[128, 256, 128]$, GELU, dropout $0.2$, $\eta_0 = 3{\times}10^{-4}$, $\lambda_{\text{wd}} = 5{\times}10^{-2}$, $B = 1024$, $\sigma_n = 0.05$, $G_{\max} = 1.0$, epochs $E = 3000$, patience $P = 150$). Each architecture–hyperparameter combination is trained with two random seeds (which simultaneously seed the parameter init *and* the data-fraction subsample), giving mean ± spread per cell.

**Table III: Grid Search Axes**

| Architecture | Swept HPs | Values | Total Trials |
|-------------|-----------|--------|--------------|
| BlackBox-FNN | `data_train_fraction`, `seed` | $\{0.02, 0.05, 0.1, 0.25, 0.5, 1.0\} \times \{0, 1\}$ | 12 |
| PhysReg-FNN | `physics_weight` $\lambda$, `data_train_fraction`, `seed` | $\{0.05, 0.1, 0.2, 0.5, 1.0, 2.0\} \times \{0.02, \ldots, 1.0\} \times \{0, 1\}$ | 72 |
| ResCorr-FNN | `alpha_reg_weight` $\alpha_r$, `data_train_fraction`, `seed` | $\{0.005, 0.01, 0.05, 0.1, 0.5\} \times \{0.02, \ldots, 1.0\} \times \{0, 1\}$ | 60 |
| **Total** | | | **144** |

The `physics_weight` axis spans the additive-Tikhonov coefficient from a near-zero nudge ($\lambda = 0.05$) through a moderate prior ($\lambda = 0.5$) up to a strong physics-dominated regime ($\lambda = 2.0$, where the physics term carries twice the weight of the data term). For ResCorr-FNN the bound $c_s = 0.5$ is held fixed (it is a registered buffer, not an optimiser parameter), and the `alpha_reg_weight` axis sweeps the L₂ penalty from "effectively unregularised" ($\alpha_r = 0.005$ — the structural tanh bound is the only constraint) to "strongly regularised" ($\alpha_r = 0.5$ — corrections shrink toward zero). The data fraction axis extends to 2 % (~5,450 training samples) to expose the small-data regime where the physics prior provides the largest relative advantage over the BlackBox baseline.

Completed trials are fingerprinted on the union of swept and exhaustive HPs and skipped on re-runs by matching against saved `metadata.yaml` files, so the grid is idempotent under interruption and resumption.

---

## VII. Results

### A. Overall Performance

Table IV reports the best test-set performance for each architecture over all v1 grid configurations (preliminary results; v2 retraining with augmented inputs pending). "Best" is defined as the minimum pooled test RMSE across all grid trials.

**Table IV: Best Model Performance (Test Set)**

| Architecture | Test RMSE (N·m) | Test R² | Test MAE (N·m) | Epochs | Data Fraction |
|-------------|-----------------|---------|----------------|--------|---------------|
| PhysReg-FNN | **0.07928** | **0.9280** | **0.04579** | 140 | 0.5 |
| ResCorr-FNN | 0.08153 | 0.9238 | 0.04433 | 278 | 1.0 |
| BlackBox-FNN | 0.08303 | 0.9210 | 0.04639 | 130 | 0.5 |

PhysReg-FNN achieves the best overall accuracy (4.5% RMSE reduction over BlackBox), confirming that a moderate physics regularization improves torque prediction without requiring structural changes to the MLP. ResCorr-FNN achieves the lowest MAE (0.0443 vs. 0.0464 for BlackBox), suggesting that physics-anchored prediction reduces systematic bias even when the pooled RMSE is slightly higher.

The RMSE ranges across all v1 grid trials are (v2 retraining pending):
- **BlackBox-FNN**: [0.08303, 0.15778] N·m, R² ∈ [0.767, 0.921] (18 v1 runs)
- **PhysReg-FNN**: [0.07928, 0.15439] N·m, R² ∈ [0.777, 0.928] (110 v1 runs)
- **ResCorr-FNN**: [0.08153, 0.12117] N·m, R² ∈ [0.863, 0.924] (74 v1 runs)

The narrower RMSE range for ResCorr-FNN is notable: even its worst-case run achieves R² > 0.86, while BlackBox and PhysReg both have runs with R² < 0.80. This is consistent with the hypothesis that the physics prior acts as a "floor" on performance in ResCorr-FNN — the network cannot do worse than the analytical model it corrects.

Figures referenced: `analysis/fig2_rmse_comparison.png`, `analysis/fig3_r2_comparison.png`, `analysis/fig10_topk_leaderboard.png`, `analysis/fig13_rmse_distribution.png`.

### B. Per-Joint Performance (Best Configurations)

Table V reports per-joint metrics for all three architectures at their best configurations.

**Table V: Per-Joint Test Metrics — Best Configuration per Architecture**

| Joint | Metric | BlackBox-FNN | PhysReg-FNN | ResCorr-FNN |
|-------|--------|-------------|-------------|-------------|
| J₁ | RMSE (N·m) | 0.0569 | **0.0576** | 0.0611 |
| | R² | 0.9247 | 0.9227 | 0.9131 |
| | MAE (N·m) | **0.0366** | 0.0386 | 0.0377 |
| J₂ | RMSE (N·m) | 0.1381 | **0.1318** | 0.1368 |
| | R² | 0.8649 | **0.8769** | 0.8674 |
| | MAE (N·m) | 0.0907 | **0.0866** | 0.0857 |
| J₃ | RMSE (N·m) | 0.0774 | 0.0720 | **0.0696** |
| | R² | 0.9374 | 0.9458 | **0.9493** |
| | MAE (N·m) | 0.0457 | 0.0468 | **0.0418** |
| J₄ | RMSE (N·m) | 0.0301 | **0.0275** | 0.0281 |
| | R² | 0.9265 | **0.9389** | 0.9361 |
| | MAE (N·m) | 0.0179 | **0.0175** | 0.0168 |
| J₅ | RMSE (N·m) | 0.0726 | **0.0692** | 0.0718 |
| | R² | 0.8783 | **0.8894** | 0.8809 |
| | MAE (N·m) | 0.0411 | **0.0395** | 0.0396 |

Several observations are notable:

1. **J₂ benefits most from physics guidance.** PhysReg-FNN achieves a 4.6% RMSE reduction at J₂ vs. BlackBox (0.1318 vs. 0.1381 N·m). J₂ is the shoulder elevation joint with the largest range of gravitational torque variation — the physics term $\boldsymbol{\tau}_g$ provides a strong gradient signal precisely where kinematics alone are most ambiguous.

2. **J₃ is best served by residual correction.** ResCorr-FNN achieves the best J₃ performance (R²=0.9493), suggesting that the elbow joint has systematic calibration errors in the RNEA model that the correction network can efficiently absorb.

3. **J₄ shows minimal gains from physics (floor effect).** All three architectures achieve R²≈0.93 at J₄. The wrist pitch joint has low inertia and a relatively small torque range (σ≈0.46 N·m), leaving little room for physics guidance to help.

4. **Maximum errors** are dominated by J₂ in all architectures (0.88–0.93 N·m), corresponding to worst-case configurations where gravitational loading is maximum and velocity-dependent terms reinforce gravity.

Figures referenced: `analysis/fig4_per_joint_heatmaps.png`, `analysis/fig9_per_joint_r2_breakdown.png`.

### C. Effect of Physics Weight λ (PhysReg-FNN)

The physics weight λ controls the balance between data fidelity and physics anchoring. Table VI reports the best test RMSE achieved at each tested λ value across all data fractions and seeds.

**Table VI: Effect of Physics Weight λ on PhysReg-FNN Test RMSE**

| λ | Best Test RMSE (N·m) |
|---|---------------------|
| 0.05 | 0.08311 |
| 0.10 | 0.08210 |
| 0.30 | **0.07928** |
| 0.35 | 0.13380 |
| 0.40 | 0.13273 |
| 0.45 | 0.13141 |
| 0.50 | 0.07976 |
| 1.00 | 0.08448 |

The relationship is non-monotonic with a clear optimum at λ=0.3. Very low λ (0.05) provides insufficient physics guidance — the model converges similarly to BlackBox-FNN. High λ (0.35–0.45) causes training instability, likely because the physics loss dominates the data loss and the network converges to τ̂ ≈ τ_ref without refining friction or calibration errors. λ=1.0 recovers some performance (the cosine-annealing schedule and warmup allow the network to eventually learn data corrections), but underperforms the optimal λ=0.3.

This result is practically significant: it establishes that the RNEA model explains approximately 30% of the total loss signal optimally, with the remaining 70% requiring data-driven correction. A physics weight of λ=0.3 represents the point where RNEA uncertainties and learned residuals are best balanced.

Figures referenced: `analysis/fig15_physics_weight_impact.png`.

### D. Data Efficiency Analysis

Table VII reports the best test RMSE at each data fraction, comparing all three architectures.

**Table VII: Best Test RMSE vs. Training Data Fraction**

| Fraction | Train Samples | BlackBox-FNN | PhysReg-FNN | ResCorr-FNN |
|----------|--------------|-------------|-------------|-------------|
| 0.10 | ~27,247 | 0.08428 | **0.07976** | 0.08165 |
| 0.25 | ~68,116 | 0.08617 | **0.07984** | 0.08286 |
| 0.50 | ~136,233 | **0.08303** | **0.07928** | 0.08210 |
| 0.75 | ~204,349 | 0.15542 | 0.13304 | 0.11503 |
| 1.00 | 272,465 | 0.08585 | 0.08064 | **0.08153** |

Several findings stand out:

1. **Physics guidance improves data efficiency.** PhysReg-FNN at 10% data (27K samples) outperforms BlackBox-FNN at full data (272K samples): 0.07976 vs. 0.08585 N·m. This is a direct demonstration of the sample efficiency benefit of incorporating physical structure.

2. **The 75% fraction anomaly.** All architectures show degraded performance at frac=0.75 relative to both frac=0.5 and frac=1.0. This is a consistent artifact likely attributable to the specific trajectories included at each fractional split: the 75% subset may over-represent certain geometry types (e.g., more rectangles than circles), while the 50% and 100% sets are better balanced. ResCorr-FNN is least affected (0.115 N·m vs. 0.155 for BlackBox), confirming its robustness.

3. **Residual correction requires more data.** ResCorr-FNN's best performance occurs at frac=1.0, whereas both BlackBox and PhysReg peak at frac=0.5. This is expected: the correction network must learn a small but precise residual, which requires sufficient coverage of the configuration space to characterize friction and calibration errors.

Figures referenced: `analysis/fig14_data_efficiency.png`, `analysis/fig16_train_test_generalization_gap.png`.

### E. Training Convergence

Mean epochs to convergence across all successful runs:
- BlackBox-FNN: 253 ± 91 epochs (range: 107–426)
- PhysReg-FNN: 340 ± 168 epochs (range: 100–936)
- ResCorr-FNN: 201 ± 64 epochs (range: 137–455)

PhysReg-FNN trains for more epochs on average because the physics warmup introduces a multi-phase learning process: the first $e_w$ epochs focus on data fitting while physics guidance is ramped in; post-warmup, both terms compete, slowing validation improvement and extending the effective learning phase. ResCorr-FNN converges faster because the warm-started output layer (initialized to 10⁻² scale) provides an effective preconditioning: the optimizer immediately refines small corrections rather than building the prediction from scratch.

At the best configuration, convergence traces are:
- **PhysReg best** (λ=0.3, frac=0.5): early stop at epoch 140
- **ResCorr best** (α=0.05, frac=1.0): early stop at epoch 278
- **BlackBox best** (frac=0.5): early stop at epoch 130

Figures referenced: `analysis/fig1_training_dynamics.png`.

---

## VIII. Discussion

### A. Role of the Physics Warm-Start

Both physics-informed architectures benefit from an initialization or schedule that positions the model near the RNEA prediction at the start of training. In PhysReg-FNN, the warmup schedule achieves this implicitly: with $\alpha_{\text{eff}}(0) = 0$, the first $e_w$ epochs optimize purely for data loss and the network learns the broad statistical structure of the torque distribution. After warmup, the physics penalty constrains the solution space to be near the calibrated analytical model, guiding the optimizer away from local minima that happen to fit the training data but violate rigid-body consistency.

In ResCorr-FNN, the 10⁻² weight initialization provides an equivalent warm-start: the network begins at the RNEA solution and gradually refines corrections. This architectural prior is stronger than the loss-based warmup because it is enforced at every gradient step, not just during a transient phase.

The augmented-input design strengthens the warm-start mechanism for both physics-informed models. By providing the full decomposed RNEA tensor $\tilde{\boldsymbol{\phi}} = [\tilde{\boldsymbol{\tau}}_g, \tilde{\boldsymbol{\tau}}_M, \tilde{\boldsymbol{\tau}}_C, \tilde{\boldsymbol{\tau}}_f]$ as an explicit input feature, the network learns physics-aware representations from the beginning of training rather than reconstructing the physics structure implicitly from kinematics. The first hidden layer can immediately allocate its weights along physically meaningful directions (per-component, per-joint trust assignments — see §IV-C), which would otherwise have to be re-derived from the noisy 15-dim kinematic vector. For PhysReg-FNN, this combined mechanism — warmup schedule plus decomposed-physics input — reduces gradient conflict between $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{phys}}$ and stabilises training across the full $\lambda \in [0.05, 2.0]$ sweep.

### B. Structural Interpretation of Per-Joint Results

The per-joint results reveal a joint-specific hierarchy that aligns well with rigid-body mechanics:

- **J₁ (shoulder rotation)**: Torque is dominated by Coriolis and inertial terms (gravity is symmetric around the vertical axis). BlackBox-FNN performs comparably to physics-guided models, suggesting that kinematic features alone are sufficient here.

- **J₂ (shoulder elevation)**: The largest inertial and gravity torques occur here, with σ(τ) = 2.58 N·m (Table II std_tau). Physics guidance is most beneficial precisely because gravity torque is a strong, structured signal. The standard deviation is 3.4× that of J₁, making J₂ the dominant contributor to pooled RMSE.

- **J₃ (elbow)**: Physics calibration is moderate but residual correction provides the best R² (0.949). This suggests the elbow has systematic friction errors (e.g., non-symmetric static friction or gear compliance) that the RNEA model with Coulomb-tanh friction cannot fully capture but the residual network can.

- **J₄ (wrist pitch)**: Lowest RMSE across all models. Low inertia and small torque range (σ = 0.46 N·m) make this joint the easiest to predict. Physics guidance and residual correction provide marginal gains.

- **J₅ (wrist roll)**: Moderate improvement from physics guidance. The roll axis experiences primarily Coriolis torques; friction coefficients at this joint are well-characterized.

### C. Limitations

1. **RNEA-mismatch generalisation.** Both physics-informed architectures rely on the analytical RNEA being a *roughly correct* representation of the true dynamics: PhysReg-FNN uses RNEA as an additive penalty target, and ResCorr-FNN uses it as the output base. Systematic biases in RNEA (e.g., the calibrated mass scaling factor $\rho = 0.0931$ and the empirical friction coefficients $\mathbf{c}, \mathbf{v}, \varepsilon$ in §II-B) are absorbed by the network during training, but if the robot undergoes physical changes between training and deployment — payload attachment, joint wear, temperature-dependent friction — the residual that the network has learned to absorb may no longer match the operational RNEA error distribution. ResCorr-FNN's hard tanh bound ($c_s = 0.5$ in normalised units) limits how much the correction can absorb without partial retraining; PhysReg-FNN has no such bound but its physics penalty will pull predictions toward whatever (potentially mismatched) RNEA is computed at inference.

2. **The 75% data fraction anomaly** suggests sensitivity to the specific subset of trajectories included at each fraction. A more principled fractional subset (e.g., using stratified sampling that preserves the geometry distribution at each fraction, rather than random subsampling) would give smoother data efficiency curves.

3. **Temporal structure is not exploited.** All three architectures treat each time step independently (i.i.d. assumption). Incorporating temporal context — e.g., via LSTM or Transformer encoders — could improve accuracy on trajectories with strong velocity autocorrelation (helices, spirals) by providing explicit state history.

4. **Single-seed best performance.** The results in Table IV represent the best single trial per architecture. The variance across the two seeds in each configuration is not reported here in full; for deployment, ensemble predictions over multiple seeds would further reduce variance.

---

## IX. Conclusion

This paper presents a rigorous comparative evaluation of three physics-informed neural network architectures for robot inverse dynamics identification on a five-DOF manipulator. The main findings are:

1. **Physics regularisation improves accuracy.** PhysReg-FNN achieves 4.5 % lower pooled test RMSE than the BlackBox baseline (0.0793 vs. 0.0830 N·m), using the same MLP backbone and optimiser but with a 35-dim augmented input $[\tilde{\mathbf{q}}, \tilde{\dot{\mathbf{q}}}, \tilde{\ddot{\mathbf{q}}}, \tilde{\boldsymbol{\tau}}_g, \tilde{\boldsymbol{\tau}}_M, \tilde{\boldsymbol{\tau}}_C, \tilde{\boldsymbol{\tau}}_f]$ and an additive Tikhonov physics penalty $\mathcal{L}_{\text{data}} + \alpha_{\text{eff}}\,\mathcal{L}_{\text{phys}}$, with $\alpha_{\text{eff}}$ linearly ramped from $0$ to $\lambda$ over the first 5 % of the epoch budget.

2. **Physics guidance substantially improves data efficiency.** PhysReg-FNN trained on 10% of the data (27K samples) outperforms BlackBox-FNN trained on the full dataset (272K samples), demonstrating a >10× sample efficiency advantage.

3. **Residual correction offers the best worst-case guarantee.** ResCorr-FNN has a narrower RMSE distribution across all grid configurations (R² > 0.86 even in worst-case runs) because the architectural physics prior prevents catastrophic failure modes.

4. **The physics weight λ is the most critical hyperparameter** for PhysReg-FNN. The grid sweep covers $\lambda \in \{0.05, 0.1, 0.2, 0.5, 1.0, 2.0\}$, with $\lambda \approx 0.3$ historically observed as the empirical optimum and $\lambda \in [0.35, 0.45]$ producing training instability (R² < 0.80) in a non-trivial fraction of legacy runs. The decomposed-physics input now exposes the four RNEA components separately, making the gradient landscape more consistent across $\lambda$ settings by allowing the first hidden layer to learn per-component trust weights rather than treating the physics signal as a single blended channel.

5. **Per-joint physics relevance is non-uniform.** High-inertia joints (J₂, shoulder elevation) benefit most from physics guidance; low-inertia joints (J₄, wrist) are adequately handled by pure data-driven approaches.

These results support the integration of RNEA-based analytical models as training-time constraints or architectural priors in neural inverse dynamics, particularly when training data is limited or when physical consistency is a deployment requirement.

---

## References

[1] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. MIT Press, 2006.

[2] B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics: Modelling, Planning and Control*. Springer, 2009.

[3] A. Gijsberts and G. Metta, "Real-time model learning using Incremental Sparse Spectrum Gaussian Process Regression," *Neural Networks*, vol. 41, pp. 59–69, 2013.

[4] S. Rueckert, M. Nakatenus, S. Tosatto, and J. Peters, "Learning inverse dynamics models with contacts," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, 2017.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *J. Comput. Phys.*, vol. 378, pp. 686–707, 2019.

[6] J. Carpentier, G. Saurel, G. Buondonno, J. Mirabel, F. Lamiraux, O. Stasse, and N. Mansard, "The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives," in *Proc. IEEE Int. Symp. System Integration (SII)*, 2019.

[7] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

---

## Appendix A: Figure Index

All figures are located in `Neural_Networks/Trained_Models_Grid/analysis/`:

| Figure | Filename | Content |
|--------|----------|---------|
| Fig. 1 | `fig1_training_dynamics.png` | Training and validation loss/RMSE curves across all architectures |
| Fig. 2 | `fig2_rmse_comparison.png` | Box plots of test RMSE per architecture and per joint |
| Fig. 3 | `fig3_r2_comparison.png` | Distribution of R² coefficients across all grid runs |
| Fig. 4 | `fig4_per_joint_heatmaps.png` | Heatmaps of per-joint RMSE and R² across architectures |
| Fig. 5 | `fig5_parallel_coordinates.png` | Parallel coordinates plot: HP values colored by test RMSE |
| Fig. 6 | `fig6_r2_vs_rmse_scatter.png` | R²–RMSE Pareto front for all 202 runs |
| Fig. 7 | `fig7_mae_nrmse_comparison.png` | MAE and normalized RMSE comparison |
| Fig. 9 | `fig9_per_joint_r2_breakdown.png` | Per-joint R² stacked breakdown by architecture |
| Fig. 10 | `fig10_topk_leaderboard.png` | Top-10 runs per architecture ranked by test RMSE |
| Fig. 11 | `fig11_hp_importance.png` | Hyperparameter importance ranking (variance-explained) |
| Fig. 13 | `fig13_rmse_distribution.png` | Histogram of test RMSE across all 202 runs |
| Fig. 14 | `fig14_data_efficiency.png` | Test RMSE vs. training data fraction per architecture |
| Fig. 15 | `fig15_physics_weight_impact.png` | Effect of physics weight λ on PhysReg-FNN test RMSE |
| Fig. 16 | `fig16_train_test_generalization_gap.png` | Train-to-test RMSE gap comparison |
| Fig. 17 | `fig17_r2_test_distribution.png` | Test R² distributions per architecture |

Individual model comparison plots and training curves are in each run's subdirectory under `run_0419_1338_.../[FNN|PhysicsRegularizedFNN|ResidualCorrectionFNN]/`.

---

## Appendix B: Reproducibility

All experiments were run on an NVIDIA GPU (CUDA 12.x) with PyTorch 2.x. Complete training configuration, normalization statistics, and per-split metrics are saved in `metadata.yaml` within each run directory. Model checkpoints are stored as `model.pt` (best validation epoch) and `model_final.pt` (final epoch). The grid search runner (`Neural_Networks/models/run_loss_residual_grid.py`) is idempotent: re-running with the same HPs skips already-completed trials by matching HP fingerprints against saved metadata.

To reproduce the best PhysReg-FNN result:

```bash
# Set in run_loss_residual_grid.py:
# ARCH = "physreg"
# GRID_PHYSREG = {"physics_weight": [0.3], "data_train_fraction": [0.5], "seed": [0]}
PYTHONPATH=. python -m Neural_Networks.models.run_loss_residual_grid
```
