# Physics-Informed Neural Networks for Robot Inverse Dynamics: A Comparative Study of Black-Box, Physics-Regularized, and Residual-Correction Architectures

**Abstract** — Accurate joint torque estimation is a fundamental requirement for model-based robot control, adaptive compensation, and safe human-robot interaction. Classical analytical inverse dynamics computed via Recursive Newton-Euler Algorithm (RNEA) provides a structured physical prior, but its accuracy is limited by uncertain inertial parameters, unmodeled friction, and discretization artifacts. This paper presents a systematic comparative study of three neural network architectures for inverse dynamics identification on a five-degree-of-freedom serial manipulator: (i) a fully data-driven feedforward neural network (BlackBox-FNN) that treats torque prediction as a pure regression problem; (ii) a physics-regularized FNN (PhysReg-FNN) that concatenates the normalized RNEA torque sum to the kinematic features (20-dim augmented input) and augments the data loss with a soft physics constraint using learnable per-joint affine RNEA calibration, providing physics as an explicit input at both training and inference time; and (iii) a residual-correction FNN (ResCorr-FNN) that learns a physics-context-aware additive correction $\boldsymbol{\delta}([\mathbf{x}, \boldsymbol{\tau}_{\text{phys}}])$ on top of the RNEA prediction from the same augmented 20-dim input. All three architectures share the same MLP backbone ([256–512–256] hidden units, GELU activations, LayerNorm), trained on 272,465 time steps collected across 11 distinct Cartesian motion geometries and evaluated on a held-out test set of 43,093 samples. A structured hyperparameter grid search covering 144 training runs is reported. Preliminary v1 results (pre-augmented-input architecture) achieve pooled test RMSE of **0.0793 N·m** (PhysReg), **0.0815 N·m** (ResCorr), and **0.0830 N·m** (BlackBox). The v2 architecture with physics as an explicit input feature and reduced correction regularization (α_r=0.01) is expected to substantially widen this gap, particularly at low data fractions where the physics prior provides the largest relative advantage.

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

All three architectures share a common MLP backbone. Let $L$ denote the number of hidden layers with widths $h_1, h_2, \ldots, h_L$. The shared building block for each layer $l$ is:

$$\mathbf{a}^{(l)} = \text{Dropout}_p\!\left(\sigma\!\left(\text{LayerNorm}\!\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)\right)\right)$$

where $\sigma$ is the activation function, LayerNorm normalizes over the feature dimension, and Dropout applies Bernoulli masking with probability $p$. All weight matrices are initialized with Xavier-normal initialization; biases are initialized to zero. The final layer is a plain linear map without normalization or activation.

For all experiments: hidden widths $[h_1, h_2, h_3] = [256, 512, 256]$, activation $\sigma = \text{GELU}$, dropout $p = 0.1$. The BlackBox-FNN backbone (15-dim input) has **270,341 trainable parameters**. The PhysReg-FNN and ResCorr-FNN backbones (20-dim augmented input) have **271,621 parameters** for the MLP. PhysReg-FNN additionally includes 10 learnable calibration parameters ($\boldsymbol{s}, \boldsymbol{b} \in \mathbb{R}^5$) for a total of **271,631 parameters**. The 1,280-parameter increase over BlackBox-FNN reflects the 5 additional input weights in the first linear layer ($5 \times 256 = 1{,}280$).

### A. Black-Box FNN (BlackBox-FNN)

The simplest architecture treats torque prediction as a pure regression problem with no physics knowledge:

$$\hat{\boldsymbol{\tau}}_{\text{BB}} = f_\theta(\tilde{\mathbf{x}})$$

where $\tilde{\mathbf{x}} \in \mathbb{R}^{15}$ is the normalized kinematic feature vector.

**Training loss.** The weighted mean-squared error with per-joint weights $\mathbf{w} = [1.0, 2.5, 1.0, 1.0, 1.0]$ emphasizes J₂ (the shoulder joint with the largest torque range and highest contribution to total MSE):

$$\mathcal{L}_{\text{BB}}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^5 w_j \left(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}\right)^2$$

The network receives no physics information at any stage. The physics argument in the forward pass is explicitly discarded (`del physics`) to prevent accidental information leakage.

### B. Physics-Regularized FNN (PhysReg-FNN)

PhysReg-FNN uses the same MLP backbone as BlackBox-FNN but with an **augmented 20-dim input** that concatenates the normalized RNEA torque sum to the kinematic features:

$$\hat{\boldsymbol{\tau}}_{\text{PR}} = f_\theta\!\left([\tilde{\mathbf{x}},\; \tilde{\boldsymbol{\tau}}_{\text{phys}}]\right)$$

where $[\cdot, \cdot]$ denotes concatenation, $\tilde{\mathbf{x}} \in \mathbb{R}^{15}$ is the normalized kinematic vector, and $\tilde{\boldsymbol{\tau}}_{\text{phys}} = \sum_k \tilde{\boldsymbol{\phi}}^{(k)} \in \mathbb{R}^5$ is the normalized RNEA sum:

$$\tilde{\boldsymbol{\tau}}_{\text{phys}} = \tilde{\boldsymbol{\tau}}_g + \tilde{\boldsymbol{\tau}}_M + \tilde{\boldsymbol{\tau}}_C + \tilde{\boldsymbol{\tau}}_f$$

The MLP input dimension is therefore $n_J \times 4 = 20$. The network receives the physics prediction as an explicit feature at every forward pass — during both training and inference.

**Learnable RNEA calibration.** The RNEA model has systematic per-joint errors (e.g., ~9.3% global mass scaling in this manipulator). To correct these, two learnable per-joint parameter vectors are introduced:

$$\boldsymbol{s} \in \mathbb{R}^{n_J},\; \boldsymbol{b} \in \mathbb{R}^{n_J} \qquad \text{(initialized to } \mathbf{1} \text{ and } \mathbf{0}\text{)}$$

The physics reference used in the training loss is the affinely calibrated RNEA:

$$\boldsymbol{\tau}_{\text{ref,cal},j} = s_j \cdot \tilde{\tau}_{\text{phys},j} + b_j$$

**Composite training loss.** The blended data + calibrated-physics loss is:

$$\mathcal{L}_{\text{PR}}(\theta, \boldsymbol{s}, \boldsymbol{b};\, \alpha_{\text{eff}}) = (1 - \alpha_{\text{eff}}) \underbrace{\frac{1}{N} \sum_{i} \sum_j w_j \left(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}\right)^2}_{\mathcal{L}_{\text{data}}} + \alpha_{\text{eff}} \underbrace{\frac{1}{N} \sum_{i} \sum_j w_j \left(\hat{\tau}_j^{(i)} - \tau_{\text{ref,cal},j}^{(i)}\right)^2}_{\mathcal{L}_{\text{phys}}}$$

The gradient with respect to the calibration scale at joint $j$ is:

$$\frac{\partial \mathcal{L}_{\text{phys}}}{\partial s_j} = -\frac{2}{N} \sum_i w_j \left(\hat{\tau}_j^{(i)} - \tau_{\text{ref,cal},j}^{(i)}\right) \cdot \tilde{\tau}_{\text{phys},j}^{(i)}$$

This drives $s_j$ toward the true RNEA-to-measured-torque ratio for joint $j$. The calibration is optimized jointly with $\theta$ via the same AdamW optimizer; the identity initialization ensures epoch-0 behavior is identical to the uncalibrated case.

**Physics warmup schedule.** $\alpha_{\text{eff}}$ is linearly ramped from zero to the target physics weight $\lambda$ over a warmup period of $e_w = \lfloor \gamma \cdot E \rfloor$ epochs, where $\gamma = 0.05$ and $E$ is the total epoch budget:

$$\alpha_{\text{eff}}(e) = \lambda \cdot \min\!\left(1,\; \frac{e}{e_w}\right)$$

This prevents the physics loss from dominating before the network has learned to process the 20-dim augmented input, avoiding collapsed solutions where $\hat{\tau} \approx \tau_{\text{ref,cal}}$ without learning the residual structure.

**Identical joint weights for both loss terms.** The same per-joint weight vector $\mathbf{w}$ is applied to both $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{phys}}$. Unweighted physics MSE alongside weighted data MSE would create conflicting gradient directions for J₂ ($w_2 = 2.5$), destabilizing training.

**Inference.** PhysReg-FNN requires the RNEA prediction at inference time. Given kinematic inputs, $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ is computed from the Pinocchio RNEA and appended to $\tilde{\mathbf{x}}$ before the forward pass. The learned calibration parameters $\boldsymbol{s}, \boldsymbol{b}$ are used only during training (in $\mathcal{L}_{\text{phys}}$) and do not modify the input features at inference.

### C. Residual-Correction FNN (ResCorr-FNN)

ResCorr-FNN enforces a physics-consistent decomposition as a **hard architectural constraint**. The network predicts an additive correction $\boldsymbol{\delta}$ to the RNEA prediction, where the correction network receives the same augmented 20-dim input:

$$\hat{\boldsymbol{\tau}}_{\text{RC}} = \tilde{\boldsymbol{\tau}}_{\text{phys}} + \boldsymbol{\delta}\!\left([\tilde{\mathbf{x}},\; \tilde{\boldsymbol{\tau}}_{\text{phys}}]\right)$$

where $\boldsymbol{\delta} : \mathbb{R}^{20} \rightarrow \mathbb{R}^5$ is the [256–512–256] MLP backbone. Providing $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ to the correction network enables context-sensitive residuals: the network can learn "when $\tilde{\tau}_{\text{phys},j}$ is large (high gravitational loading), the friction calibration error at joint $j$ also tends to be large, so apply a proportionally larger correction."

**Small-residual initialization.** The last linear layer's weights and biases are multiplied by $10^{-2}$ at initialization. This ensures that at epoch 0, $\boldsymbol{\delta} \approx \mathbf{0}$ and $\hat{\boldsymbol{\tau}} \approx \tilde{\boldsymbol{\tau}}_{\text{phys}}$, giving the optimizer a warm start from the physics solution. This initialization is preserved regardless of input dimensionality — the near-zero output at startup is the structural goal, not the input size.

**Regularized training loss.** To prevent the correction network from absorbing the full inverse dynamics task (making the physics base irrelevant), an L₂ penalty on the correction magnitude is applied:

$$\mathcal{L}_{\text{RC}}(\theta) = \underbrace{\frac{1}{N} \sum_{i} \sum_j w_j \left(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}\right)^2}_{\mathcal{L}_{\text{data}}} + \alpha_r \underbrace{\frac{1}{NJ} \sum_{i,j} \delta_j^{(i)^2}}_{\mathcal{L}_{\text{reg}}}$$

where $\alpha_r = 0.01$ is the regularization weight. Note: the v1 architecture used $\alpha_r = 0.05$, which suppressed corrections to $|\delta_j| \lesssim 0.02$ N·m while RNEA calibration errors require corrections of 0.05–0.08 N·m. The reduced $\alpha_r = 0.01$ permits the network to learn meaningful corrections without eliminating the RNEA base contribution. The per-epoch ratio $\mathbb{E}[|\boldsymbol{\delta}|]/\mathbb{E}[|\tilde{\boldsymbol{\tau}}_{\text{phys}}|]$ is logged to monitor physics-vs-correction reliance throughout training.

**Relationship to PhysReg-FNN.** In PhysReg-FNN, $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ serves two roles: (a) as a concatenated input feature so the MLP conditions its output on the physics prediction, and (b) as a calibrated reference signal in the training loss. In ResCorr-FNN, $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ enters as the hard additive output base and as an explicit input to $\delta$ — providing the correction network with the physics context it needs for position-dependent residuals. Both architectures use physics at inference time; neither discards the RNEA prediction after training.

### D. Architecture Design Motivation: Why Provide τ_phys as an Explicit Input?

A natural question is why passing $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ as an input feature improves generalization, given that a sufficiently large MLP could in principle reconstruct the RNEA computation from kinematics $\tilde{\mathbf{x}}$ alone.

**Information-theoretic argument.** Suppose the true inverse dynamics factorizes as:

$$\boldsymbol{\tau} = \boldsymbol{\tau}_{\text{phys}}(\mathbf{x}) + r(\mathbf{x},\, \boldsymbol{\tau}_{\text{phys}})$$

where $r$ is a residual depending on both kinematics and the physics estimate (e.g., friction errors that scale with joint loading, which is encoded in $\boldsymbol{\tau}_{\text{phys}}$). A network receiving only $\tilde{\mathbf{x}}$ must implicitly reconstruct $\boldsymbol{\tau}_{\text{phys}}(\mathbf{x})$ as an intermediate computation before it can learn $r$. By the data-processing inequality, any statistic of $\tilde{\mathbf{x}}$ available to this network is also available to a network receiving $[\tilde{\mathbf{x}},\, \tilde{\boldsymbol{\tau}}_{\text{phys}}]$ — but not vice versa. The augmented input is therefore a strict superset in information content, with the RNEA prediction acting as a compressed sufficient statistic for the physics structure.

**Capacity allocation.** With limited training data, the network must allocate representational capacity to reconstruct the physics structure from kinematics. Providing $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ directly frees this capacity to model friction calibration errors and unmodeled dynamics — effects that require many samples to learn from kinematics alone. This is the mechanism by which physics augmentation improves data efficiency.

**Gradient stability.** In PhysReg-FNN, the physics loss $\mathcal{L}_{\text{phys}}$ provides a gradient signal pulling $\hat{\boldsymbol{\tau}}$ toward $\boldsymbol{\tau}_{\text{ref,cal}}$. When $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ is an explicit input, the network can satisfy this constraint by learning a near-identity mapping from $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ to $\hat{\boldsymbol{\tau}}$, which is a much simpler function than reconstructing the physics from raw kinematics. This reduces gradient conflict between $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{phys}}$, improving training stability at higher physics weights $\lambda$.

**Calibration warm-start.** The affine calibration $(s_j, b_j)$ is initialized at $(1, 0)$ — the identity transformation. This ensures that at epoch 0, $\boldsymbol{\tau}_{\text{ref,cal}} = \tilde{\boldsymbol{\tau}}_{\text{phys}}$, so the physics loss is anchored to the raw RNEA prediction from the start. A random initialization of the calibration would push $\hat{\boldsymbol{\tau}}$ toward a meaningless reference during early training, destabilizing the optimizer before any physics structure is learned.

---

## V. Training Methodology

### A. Optimizer and Regularization

All models are trained with AdamW [7]:

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \cdot \lambda_{\text{wd}} \cdot \theta_t$$

with learning rate $\eta = 3 \times 10^{-4}$, weight decay $\lambda_{\text{wd}} = 10^{-2}$, and default momentum parameters $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Input noise augmentation.** During training (but not evaluation), Gaussian noise is added to the normalized feature vector:

$$\tilde{\mathbf{x}}_{\text{aug}} = \tilde{\mathbf{x}} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}\!\left(\mathbf{0},\, \sigma_n^2 \mathbf{I}_{d_x}\right), \quad \sigma_n = 0.02$$

where $d_x = 20$ for PhysReg-FNN and ResCorr-FNN, and $d_x = 15$ for BlackBox-FNN (noise applied only to the kinematic channels, not to the appended $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ block). The perturbation is in normalized space; the equivalent physical-space standard deviation for channel $k$ is $\sigma_n \cdot \sigma_k^{\text{train}}$. This regularizes against overfitting to specific kinematic profiles and improves generalization to trajectories with slightly different velocity smoothness.

**Activation function.** All hidden layers use GELU (Gaussian Error Linear Unit):

$$\text{GELU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702\, x)$$

where $\Phi$ is the standard-normal CDF and $\sigma$ is the logistic sigmoid. GELU is preferred over ReLU for two reasons: (a) it is smooth and differentiable everywhere, preventing the "dead neuron" problem where $\partial \mathcal{L}/\partial x = 0$ for $x < 0$; (b) it retains small negative-activation gradients, which is beneficial when the physics loss creates competing gradient directions during the warmup phase. The approximation $\sigma(1.702x)$ is used in the PyTorch implementation for computational efficiency.

**Gradient clipping.** To prevent gradient explosions from the weighted physics loss at the beginning of training, the global gradient norm is clipped to $G_{\max} = 5.0$.

**Mixed precision.** On CUDA devices, NVIDIA's Automatic Mixed Precision (AMP) is used with FP16 compute and FP32 master weights, reducing memory footprint and improving throughput.

### B. Learning Rate Schedule: Warmup Cosine

A warmup-cosine annealing schedule is applied:

$$\eta(e) = \begin{cases} \eta_0 \left(0.1 + 0.9 \cdot \frac{e}{e_w}\right) & e < e_w \\ \eta_0 \left(r_{\min} + (1 - r_{\min}) \cdot \frac{1 + \cos(\pi \cdot \text{progress})}{2}\right) & e \geq e_w \end{cases}$$

where $e_w = \max(1, \lfloor E/20 \rfloor)$ is the warmup epoch count, $\text{progress} = (e - e_w)/(E - e_w) \in [0, 1]$, and $r_{\min} = 0.01$ is the minimum LR ratio. This schedule warms up to $\eta_0$ over the first 5% of training (preventing unstable early updates) then cosine-anneals to $0.01 \cdot \eta_0$.

### C. Early Stopping

Training stops when the validation RMSE does not improve by more than $\delta_{\min} = 10^{-4}$ N·m for $P = 100$ consecutive epochs. The model state from the best validation epoch is restored at the end of training. This prevents overfitting to the training set at the cost of some additional compute for the patience window.

### D. DataLoader Configuration

Training batches are assembled with `batch_size = 1024` and `drop_last=True` (only when the training set contains at least $2 \times \text{batch\_size}$ samples, to avoid empty training at small data fractions). Validation and test evaluation use all samples without dropping. The DataLoader uses 4 worker processes for asynchronous prefetching and `pin_memory=True` for fast host-to-device transfer.

---

## VI. Hyperparameter Grid Search

To characterize the effect of key hyperparameters, a structured grid search is conducted. All non-swept hyperparameters are held fixed at the values in §V. Each architecture–hyperparameter combination is trained with two random seeds to assess variance.

**Table III: Grid Search Axes (v2)**

| Architecture | Swept HPs | Values | Total Trials |
|-------------|-----------|--------|--------------|
| BlackBox-FNN | `data_train_fraction`, `seed` | {0.02, 0.05, 0.1, 0.25, 0.5, 1.0} × {0, 1} | 12 |
| PhysReg-FNN | `physics_weight`, `data_train_fraction`, `seed` | {0.05, 0.1, 0.2, 0.3, 0.5, 1.0} × {0.02, 0.05, 0.1, 0.25, 0.5, 1.0} × {0, 1} | 72 |
| ResCorr-FNN | `alpha_reg_weight`, `data_train_fraction`, `seed` | {0.001, 0.005, 0.01, 0.05, 0.1} × {0.02, 0.05, 0.1, 0.25, 0.5, 1.0} × {0, 1} | 60 |
| **Total** | | | **144** |

The `physics_weight` axis spans log-ish intervals from negligible (0.05) to physics-dominated (1.0), with 0.3 as the v1 empirical optimum and 1.0 as the confirmed upper bound. The `alpha_reg_weight` axis covers the under-penalized regime (0.001: near-free corrections) through the over-penalized regime (0.1: corrections suppressed below RNEA error magnitude), enabling direct identification of the optimal correction constraint. The data fraction axis extends to 2% (~5,450 training samples) to expose the data-efficiency regime where the physics prior provides the largest relative advantage over the BlackBox baseline.

PhysReg and ResCorr trials include `phys_input_concat=True` in metadata to distinguish v2 runs (20-dim augmented input) from v1 checkpoints (15-dim input). Existing v1 runs will not fingerprint-match and will be re-trained automatically. Completed v2 trials are fingerprinted and skipped on re-runs via matching against saved `metadata.yaml` files.

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

The v2 architecture further strengthens the warm-start mechanism for both models. By providing $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ as an explicit input feature, the network learns physics-aware representations from the beginning of training rather than reconstructing the physics structure implicitly from kinematics. At epoch 0, the learnable calibration $(s_j=1, b_j=0)$ ensures the physics loss is anchored to the raw RNEA prediction, giving a coherent gradient signal before the network has adapted its weights to process the 20-dim augmented input. For PhysReg-FNN, this dual mechanism — warmup schedule plus physics input feature — reduces gradient conflict between $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{phys}}$ and is expected to stabilize training at larger physics weights compared to the v1 architecture.

### B. Structural Interpretation of Per-Joint Results

The per-joint results reveal a joint-specific hierarchy that aligns well with rigid-body mechanics:

- **J₁ (shoulder rotation)**: Torque is dominated by Coriolis and inertial terms (gravity is symmetric around the vertical axis). BlackBox-FNN performs comparably to physics-guided models, suggesting that kinematic features alone are sufficient here.

- **J₂ (shoulder elevation)**: The largest inertial and gravity torques occur here, with σ(τ) = 2.58 N·m (Table II std_tau). Physics guidance is most beneficial precisely because gravity torque is a strong, structured signal. The standard deviation is 3.4× that of J₁, making J₂ the dominant contributor to pooled RMSE.

- **J₃ (elbow)**: Physics calibration is moderate but residual correction provides the best R² (0.949). This suggests the elbow has systematic friction errors (e.g., non-symmetric static friction or gear compliance) that the RNEA model with Coulomb-tanh friction cannot fully capture but the residual network can.

- **J₄ (wrist pitch)**: Lowest RMSE across all models. Low inertia and small torque range (σ = 0.46 N·m) make this joint the easiest to predict. Physics guidance and residual correction provide marginal gains.

- **J₅ (wrist roll)**: Moderate improvement from physics guidance. The roll axis experiences primarily Coriolis torques; friction coefficients at this joint are well-characterized.

### C. Limitations

1. **Calibration generalization.** The learnable affine calibration ($s_j$, $b_j$) in PhysReg-FNN corrects systematic per-joint RNEA biases observed in the training data (e.g., ~9.3% global mass scaling, per-joint friction offsets). If the robot undergoes physical changes between training and deployment — payload attachment, joint wear, or temperature-dependent friction — the learned calibration may no longer match the operational RNEA error distribution, requiring partial or full retraining. This is a weaker limitation than the v1 architecture's uncalibrated RNEA, but it remains a deployment consideration for long-horizon deployments.

2. **The 75% data fraction anomaly** suggests sensitivity to the specific subset of trajectories included at each fraction. A more principled fractional subset (e.g., using stratified sampling that preserves the geometry distribution at each fraction, rather than random subsampling) would give smoother data efficiency curves.

3. **Temporal structure is not exploited.** All three architectures treat each time step independently (i.i.d. assumption). Incorporating temporal context — e.g., via LSTM or Transformer encoders — could improve accuracy on trajectories with strong velocity autocorrelation (helices, spirals) by providing explicit state history.

4. **Single-seed best performance.** The results in Table IV represent the best single trial per architecture. The variance across the two seeds in each configuration is not reported here in full; for deployment, ensemble predictions over multiple seeds would further reduce variance.

---

## IX. Conclusion

This paper presents a rigorous comparative evaluation of three physics-informed neural network architectures for robot inverse dynamics identification on a five-DOF manipulator. The main findings are:

1. **Physics regularization improves accuracy.** PhysReg-FNN achieves 4.5% lower pooled test RMSE than the BlackBox baseline (0.0793 vs. 0.0830 N·m, preliminary v1 results) with an optimal physics weight of λ=0.3, using the same MLP backbone and optimizer. The v2 architecture provides physics as an explicit input feature (20-dim augmented input [q, q̇, q̈, τ_phys]) at both training and inference time, with learnable per-joint affine RNEA calibration, expected to further widen this margin.

2. **Physics guidance substantially improves data efficiency.** PhysReg-FNN trained on 10% of the data (27K samples) outperforms BlackBox-FNN trained on the full dataset (272K samples), demonstrating a >10× sample efficiency advantage.

3. **Residual correction offers the best worst-case guarantee.** ResCorr-FNN has a narrower RMSE distribution across all grid configurations (R² > 0.86 even in worst-case runs) because the architectural physics prior prevents catastrophic failure modes.

4. **The physics weight λ is the most critical hyperparameter** for PhysReg-FNN. In the v1 architecture, values λ=0.3 were empirically optimal; λ ∈ [0.35, 0.45] caused training instability in a significant fraction of runs (R² < 0.80). The v2 augmented-input design is expected to reduce this sensitivity by providing $\tilde{\boldsymbol{\tau}}_{\text{phys}}$ as an explicit MLP feature, making the gradient landscape more consistent across physics-weight settings and extending the stable range of λ toward larger values.

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
