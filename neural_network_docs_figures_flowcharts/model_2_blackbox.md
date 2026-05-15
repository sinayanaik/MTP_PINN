# Model II: Data-Driven Feedforward Neural Network (BlackBoxFNN)

The BlackBoxFNN is a purely data-driven Multi-Layer Perceptron (MLP) that learns the robot inverse dynamics mapping $(q, \dot{q}, \ddot{q}) \mapsto \tau^*$ end-to-end from measured joint trajectories. It imposes no analytical structure from rigid-body mechanics and therefore serves as the **data-only baseline**: any performance gain achieved by the physics-informed models (Models III and IV) is measured against this architecture. Its simplicity makes it useful for bounding the information content of the kinematic state alone.

---

## Notation

The following symbols are used throughout this section. Vectors are in $\mathbb{R}^J$ unless otherwise stated; element-wise operations apply over the joint index $j = 1, \ldots, J$.

| Symbol | Dimension | Description |
|---|---|---|
| $J = 5$ | — | Number of active joints |
| $q$ | $\mathbb{R}^J$ | Joint position vector (rad) |
| $\dot{q}$ | $\mathbb{R}^J$ | Joint velocity vector (rad/s) |
| $\ddot{q}$ | $\mathbb{R}^J$ | Joint acceleration vector (rad/s²) |
| $\tau^*$ | $\mathbb{R}^J$ | Measured (ground-truth) joint torque vector (N·m) |
| $\hat{\tau}$ | $\mathbb{R}^J$ | Predicted joint torque vector (N·m) |
| $\tilde{z}$ | — | Z-score normalized version of signal $z$ |
| $\mu_{z,j},\, \sigma_{z,j}$ | — | Training-split mean and standard deviation of the $j$-th component of signal $z$ |
| $\tilde{x}_k$ | $\mathbb{R}^{3J}$ | Normalized kinematic feature vector |
| $\theta$ | — | Set of all learnable network parameters $\{W^{(l)}, b^{(l)}\}_{l=1}^{4}$ |
| $W^{(l)}$ | $\mathbb{R}^{d_l \times d_{l-1}}$ | Weight matrix of layer $l$ |
| $b^{(l)}$ | $\mathbb{R}^{d_l}$ | Bias vector of layer $l$ |
| $h^{(l)}$ | $\mathbb{R}^{d_l}$ | Hidden activation vector of layer $l$ |
| $[d_0,d_1,d_2,d_3,d_4]$ | — | Layer dimensions: $[15, 256, 512, 256, 5]$ |
| $N$ | — | Number of training samples |
| $w$ | $\mathbb{R}^J$ | Per-joint loss weights: $[1.0,\, 2.5,\, 1.0,\, 1.0,\, 1.0]^\top$ |
| $\sigma(\cdot)$ | — | SiLU activation function |
| $\text{LN}(\cdot)$ | — | Layer Normalization |
| $p$ | — | Dropout rate |
| $\varepsilon$ | — | Small constant for numerical stability ($10^{-8}$) |

---

## 1. Problem Formulation

For a serial-chain manipulator, the Newton-Euler equations of motion in joint space take the form
$$
\tau^* = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) + \tau_f(\dot{q}) \tag{1}
$$
where $M(q) \in \mathbb{R}^{J \times J}$ is the symmetric positive-definite inertia matrix, $C(q,\dot{q})\dot{q} \in \mathbb{R}^J$ collects Coriolis and centrifugal effects, $g(q) \in \mathbb{R}^J$ is the gravitational generalized force, and $\tau_f(\dot{q}) \in \mathbb{R}^J$ is joint friction. **Inverse dynamics** is the problem of computing $\tau^*$ given the motion $(q, \dot{q}, \ddot{q})$.

In the presence of model uncertainties — 3D-printed link geometries that deviate from the nominal URDF, unmodeled joint compliance, and imperfect friction calibration — the analytical prediction from Eq. (1) carries systematic errors on the order of 0.20–0.25 N·m (see Model I). The BlackBoxFNN addresses this by treating Eq. (1) purely as a regression problem: rather than specifying the functional form, we learn the mapping $f: \mathbb{R}^{3J} \to \mathbb{R}^J$ directly from paired samples $\{(\tilde{x}_{k,i}, \tilde{\tau}^*_i)\}_{i=1}^N$.

---

## 2. Input Representation

### 2.1 Kinematic Feature Vector

At each timestep, the kinematic state is assembled into a single feature vector by concatenating the three joint-space signals:
$$
x_k = \bigl[q^\top,\;\dot{q}^\top,\;\ddot{q}^\top\bigr]^\top \in \mathbb{R}^{3J} \tag{2}
$$
For $J = 5$ active joints, this yields a raw input dimension of $3J = 15$. The feature vector contains all kinematic information relevant to inverse dynamics: positions determine the gravitational and configuration-dependent inertial terms; velocities determine Coriolis–centrifugal and friction contributions; accelerations determine the inertial $M(q)\ddot{q}$ term.

### 2.2 Z-Score Normalization

Raw kinematic signals span disparate physical magnitudes: joint positions are $\mathcal{O}(1)$ rad, velocities are $\mathcal{O}(10)$ rad/s, and accelerations are $\mathcal{O}(10^2)$ rad/s². Without normalization, the gradient of the loss with respect to acceleration features would dominate the weight updates, leaving position features — which encode the gravitational term — comparatively starved. Each signal component is therefore independently standardized using training-split statistics:
$$
\tilde{z}_j = \frac{z_j - \mu_{z,j}}{\max(\sigma_{z,j},\, \varepsilon)}, \qquad z \in \{q,\, \dot{q},\, \ddot{q}\},\quad j = 1,\ldots,J \tag{3}
$$
The $\max(\cdot, \varepsilon)$ guard prevents division by zero for channels that are constant over the entire training split (e.g., a locked joint). All statistics $\{\mu_{z,j}, \sigma_{z,j}\}$ are computed once from the training set and frozen; no validation or test data enters the normalization, ensuring that evaluation metrics are computed on truly unseen distributions. The normalized kinematic feature vector is:
$$
\tilde{x}_k = \bigl[\tilde{q}^\top,\;\tilde{\dot{q}}^\top,\;\tilde{\ddot{q}}^\top\bigr]^\top \in \mathbb{R}^{3J} \tag{4}
$$

The ground-truth torque is normalized analogously:
$$
\tilde{\tau}^*_j = \frac{\tau^*_j - \mu_{\tau,j}}{\sigma_{\tau,j}}, \qquad j = 1,\ldots,J \tag{5}
$$
where $\mu_{\tau,j}$ and $\sigma_{\tau,j}$ are training-split statistics for the $j$-th measured torque. All loss computations and gradient updates are performed in this normalized space; predictions are de-normalized by inverting Eq. (5) to recover physical units (N·m) only at evaluation time.

---

## 3. Network Architecture

### 3.1 Formal Layer Definition

The network implements the mapping $f_\theta: \mathbb{R}^{3J} \to \mathbb{R}^J$ through $L = 3$ hidden layers followed by a linear output head. Setting $h^{(0)} = \tilde{x}_k$, each hidden layer applies the following four operations in sequence:

**Hidden layers** ($l = 1, 2, 3$):
$$
h^{(l)} = \text{Dropout}_p\!\left(\sigma\!\left(\text{LN}\!\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)\right)\right) \tag{6}
$$

**Output layer**:
$$
\hat{\tau}_{norm} = W^{(4)} h^{(3)} + b^{(4)} \in \mathbb{R}^J \tag{7}
$$

The layer width progression is $[d_0, d_1, d_2, d_3, d_4] = [15, 256, 512, 256, 5]$. The intermediate expansion to 512 units allows the second layer to encode cross-joint interactions (e.g., Coriolis coupling between joints $j$ and $k$ involves products $\dot{q}_j \dot{q}_k$) before compressing back to 256 for the final hidden representation.

### 3.2 Layer Normalization

Layer Normalization (LN) standardizes the pre-activation vector $z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)} \in \mathbb{R}^{d_l}$ over its feature dimension at each forward pass:
$$
\text{LN}(z^{(l)}) = \gamma^{(l)} \odot \frac{z^{(l)} - \hat{\mu}_{z^{(l)}}}{\sqrt{\hat{\sigma}^2_{z^{(l)}} + \varepsilon}} + \beta^{(l)} \tag{8}
$$
where $\hat{\mu}_{z^{(l)}} = \frac{1}{d_l}\sum_{k=1}^{d_l} z^{(l)}_k$ and $\hat{\sigma}^2_{z^{(l)}} = \frac{1}{d_l}\sum_{k=1}^{d_l} (z^{(l)}_k - \hat{\mu}_{z^{(l)}})^2$ are the per-sample mean and variance of the pre-activation, and $\gamma^{(l)}, \beta^{(l)} \in \mathbb{R}^{d_l}$ are learnable affine correction parameters initialized to $\mathbf{1}$ and $\mathbf{0}$ respectively. The symbol $\odot$ denotes the element-wise (Hadamard) product.

Unlike Batch Normalization, LN normalizes over the feature dimension for each sample independently, requiring no running statistics and remaining stable across variable batch sizes. This is important for robot dynamics datasets where trajectory length and motion patterns vary: Batch Normalization statistics computed over a mixed batch would conflate the distributions of high-speed and quasi-static trajectories.

### 3.3 SiLU Activation

The Sigmoid-weighted Linear Unit (SiLU), also known as Swish, is defined as:
$$
\sigma(x) = x \cdot \text{sigmoid}(x) = \frac{x}{1 + e^{-x}} \tag{9}
$$
Its derivative is:
$$
\sigma'(x) = \text{sigmoid}(x) + x\, \text{sigmoid}(x)\,(1 - \text{sigmoid}(x)) \tag{10}
$$
Three properties make SiLU suitable for torque regression: (i) it is infinitely differentiable ($C^\infty$), avoiding the gradient discontinuity at the origin that characterizes ReLU; (ii) it is non-monotone near $x \approx -1.28$ where it attains its minimum of approximately $-0.278$, giving the network capacity to model sign changes; (iii) it satisfies $\sigma(x) \approx x$ for large positive $x$, preserving linear information in saturated units. Since the inverse dynamics function $\tau^*(q, \dot{q}, \ddot{q})$ is an analytic function of its arguments, a smooth approximator is preferable to piecewise-linear ones.

### 3.4 Dropout Regularization

Bernoulli dropout with rate $p = 0.4$ is applied after SiLU in each hidden layer. During a training forward pass, each element of $h^{(l)}$ is independently set to zero with probability $p$ and scaled by $1/(1-p)$ to preserve the expected magnitude. At inference time, dropout is disabled and all units are active. The higher dropout rate ($p = 0.4$) compared to the physics-informed models ($p = 0.1$) reflects that the BlackBoxFNN must reconstruct the full inverse dynamics from kinematics alone, with no structural prior to constrain the solution space; stronger regularization compensates for the absence of physics guidance.

### 3.5 Weight Initialization

All weight matrices are initialized with Xavier normal initialization:
$$
W^{(l)}_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{l-1} + d_l}\right) \tag{11}
$$
This scheme maintains approximately unit variance of the pre-activations across all layers at initialization: if the input to layer $l$ has unit variance and entries $W^{(l)}_{ij}$ are i.i.d. with variance $2/(d_{l-1}+d_l)$, then the output variance $\text{Var}(W^{(l)}x) = d_{l-1} \cdot \frac{2}{d_{l-1}+d_l} \approx 1$. All bias vectors are initialized to zero.

### 3.6 Parameter Count

The total number of trainable parameters is:
$$
|\theta| = \underbrace{\sum_{l=1}^{4} (d_{l-1} \cdot d_l + d_l)}_{\text{weight matrices and biases}} + \underbrace{\sum_{l=1}^{3} 2d_l}_{\text{LayerNorm scale and shift}} \tag{12}
$$
$$
= (15{\cdot}256+256) + (256{\cdot}512+512) + (512{\cdot}256+256) + (256{\cdot}5+5) + 2(256+512+256)
\approx 270{,}000
$$

---

## 4. Training Objective

### 4.1 Joint-Weighted Mean Squared Error

The network is trained to minimize a joint-weighted mean squared error (MSE) in normalized torque space:
$$
\mathcal{L}_{data}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{J} w_j \left(\hat{\tau}_{norm,ij} - \tilde{\tau}^*_{ij}\right)^2 \tag{13}
$$
where $\hat{\tau}_{norm,ij}$ is the $j$-th component of the network output for the $i$-th sample, $\tilde{\tau}^*_{ij}$ is the corresponding normalized ground-truth torque, and $w = [1.0,\, 2.5,\, 1.0,\, 1.0,\, 1.0]^\top$ are the per-joint weights. The 2.5× upweighting on Joint~2 (shoulder pitch, J2) reflects that this joint bears the dominant gravitational and inertial load; it consistently contributes the largest fraction of the global prediction error (approximately 40% of the total squared error on this dataset), so additional gradient pressure is allocated to J2 without discarding the other joints.

### 4.2 Evaluation Metrics

At validation and test time, the loss is evaluated without joint weighting to obtain a bias-free aggregate measure:
$$
\mathcal{L}_{eval} = \frac{1}{N_{eval} \cdot J} \sum_{i=1}^{N_{eval}} \sum_{j=1}^{J} \left(\hat{\tau}_{norm,ij} - \tilde{\tau}^*_{ij}\right)^2 \tag{14}
$$
The per-joint root mean squared error in physical units (N·m) is:
$$
\text{RMSE}_j = \sigma_{\tau,j} \cdot \sqrt{\frac{1}{N_{eval}} \sum_{i=1}^{N_{eval}} \left(\hat{\tau}_{norm,ij} - \tilde{\tau}^*_{ij}\right)^2} \tag{15}
$$
and the macro-averaged test RMSE is $\overline{\text{RMSE}} = \frac{1}{J}\sum_{j=1}^{J} \text{RMSE}_j$. The coefficient of determination for joint $j$ is:
$$
R^2_j = 1 - \frac{\sum_{i}(\tau^*_{ij} - \hat{\tau}_{ij})^2}{\sum_{i}(\tau^*_{ij} - \bar{\tau}^*_j)^2} \tag{16}
$$
where $\bar{\tau}^*_j$ is the mean measured torque for joint $j$ and $\hat{\tau}_{ij} = \sigma_{\tau,j}\, \hat{\tau}_{norm,ij} + \mu_{\tau,j}$ is the de-normalized prediction.

---

## 5. Training Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| Hidden layers $[d_1, d_2, d_3]$ | $[256,\, 512,\, 256]$ | Width progression (see Section 3.1) |
| Input dimension $d_0$ | $15$ | $3J = 3 \times 5$ kinematic features |
| Activation $\sigma$ | SiLU | Infinitely differentiable (Eq. 9) |
| Normalization | LayerNorm | Per-sample, per-layer (Eq. 8) |
| Dropout rate $p$ | $0.4$ | Applied after SiLU in each hidden layer |
| Batch size | $512$ | Stochastic mini-batch gradient descent |
| Optimizer | AdamW | $\beta_1{=}0.9,\,\beta_2{=}0.999,\,\hat{\varepsilon}{=}10^{-8}$ |
| Learning rate $\eta_0$ | $3 \times 10^{-4}$ | Peak value after warm-up |
| Weight decay | $5 \times 10^{-3}$ | Decoupled from gradient update (AdamW) |
| LR schedule | Warm-up cosine | $e_{warm} = \lfloor E/20 \rfloor = 50$ epochs |
| Gradient clip norm | $5.0$ | Global L2 norm of all parameter gradients |
| Feature noise std $\sigma_n$ | $0.02$ | Added to $\tilde{x}_k$ during training only |
| Early stopping patience $P$ | $50$ epochs | Monitors val RMSE (N·m) |
| Min improvement $\delta_{min}$ | $10^{-4}$ N·m | Minimum relative improvement threshold |
| Max epochs $E$ | $1000$ | Upper bound; early stopping usually applies |
| Training samples $N$ | $272{,}465$ | 22 trajectories, 11 motion geometries |

**Learning rate schedule.** The learning rate follows a linear warm-up for the first $e_{warm}$ epochs, then decays via cosine annealing:
$$
\eta(e) = \begin{cases} \dfrac{e}{e_{warm}}\,\eta_0 & 0 \leq e < e_{warm} \\[6pt] \dfrac{\eta_0}{2}\left(1 + \cos\!\left(\pi\,\dfrac{e - e_{warm}}{E - e_{warm}}\right)\right) & e \geq e_{warm} \end{cases} \tag{17}
$$
The warm-up phase prevents instability during the initial epochs when the loss surface is poorly characterized and gradients are large relative to the learning signal.

---

## 6. Theoretical Properties and Limitations

**Universal approximation.** A feedforward network with SiLU activations and three hidden layers of widths $[256, 512, 256]$ is a universal approximator: for any continuous target function $f: \mathbb{R}^{15} \to \mathbb{R}^J$ and any $\varepsilon > 0$, there exist parameters $\theta$ such that $\sup_x \|f_\theta(x) - f(x)\| < \varepsilon$. This theorem guarantees existence of a good approximation, but provides no bound on the required sample size $N$ or the convergence rate of gradient descent.

**No physics inductive bias.** The network must independently rediscover the full structure of Eq. (1) from data. The inertia matrix $M(q)$ contains trigonometric cross-terms $\sin(q_j)\cos(q_k)$; the Coriolis term involves velocity cross-products $\dot{q}_j \dot{q}_k$; the gravity vector involves $\sin(q_j)$ with amplitudes proportional to individual link masses and moment arms. The network approximates all of these through the same learned MLP weights, providing no guarantee of physical consistency (e.g., energy conservation or positive-definite inertial terms) and potentially extrapolating poorly to states not represented in training.

**Data requirements.** Without a physics prior to constrain the solution space, the model requires dense data coverage of the joint-space trajectory $(q, \dot{q}, \ddot{q})$ to generalize reliably. In low-data regimes, the network overfits to trajectory-specific features and fails to capture the underlying physical relationships. This motivates the physics-informed designs in Models III and IV, which embed the analytical structure of Eq. (1) directly into the architecture to reduce the effective hypothesis space.

**Experimental results.** Trained on $N = 272{,}465$ samples across 22 trajectories (11 motion geometries) on the Kikobot 5-DOF manipulator, BlackBoxFNN achieves a mean test RMSE of $\overline{\text{RMSE}} = 0.0948$ N·m with a mean per-joint $R^2 = 0.854$. Per-joint test RMSE values and $R^2$ are summarized below.

| Joint | Role | Test RMSE (N·m) | $R^2$ |
|---|---|---|---|
| J1 | Yaw (base) | $0.0695$ | $0.890$ |
| J2 | Shoulder pitch | $0.1732$ | $0.789$ |
| J3 | Elbow pitch | $0.1032$ | $0.890$ |
| J4 | Wrist pitch | $0.0389$ | $0.879$ |
| J5 | Wrist roll | $0.0892$ | $0.820$ |
| **Mean** | — | $\mathbf{0.0948}$ | $\mathbf{0.854}$ |

Joint J2 exhibits the largest absolute error (0.173 N·m) because it carries the maximum gravitational moment arm and experiences large inertial coupling from J3. Joint J4 achieves the smallest error (0.039 N·m), reflecting its lighter mechanical load (rated stall torque 14.8 kgf·cm vs. 30.0 kgf·cm for J1–J3, J5). These per-joint differences motivate the joint-weighting strategy of Eq. (13) and provide the per-joint improvement target for physics-informed models.
