# End-to-End Inverse Dynamics Estimation Workflow

This document describes the complete pipeline from raw hardware logs to trained torque-prediction models for the Kikobot 6-DOF serial manipulator. The pipeline comprises four stages: signal acquisition and repair, physics-aware preprocessing, dataset construction, and supervised neural-network training evaluated against a rigid-body analytical baseline.

---

## 1. Robot Platform and Problem Statement

The Kikobot is a 6-DOF serial manipulator with $J = 5$ independently actuated joints (J1–J5; J6 is a passive tool link). Each joint is driven by a Feetech smart servo: J1–J3 and J5 use the STS3215 model (rated stall torque $\bar{\tau}_1 = 30.0\ \text{kgf·cm}$); J4 uses the smaller STS3032 model ($\bar{\tau}_4 = 14.8\ \text{kgf·cm}$). The robot structure is 3D-printed in PLA at approximately 70% infill.

**Inverse dynamics problem.** Given the joint-space trajectory $\bigl(q(t),\,\dot{q}(t),\,\ddot{q}(t)\bigr) \in \mathbb{R}^J \times \mathbb{R}^J \times \mathbb{R}^J$, predict the generalized joint torques $\tau(t) \in \mathbb{R}^J$ required to produce that motion. The rigid-body equation of motion is:
$$
\tau(t) = M\!\bigl(q(t)\bigr)\,\ddot{q}(t) + C\!\bigl(q(t),\dot{q}(t)\bigr)\,\dot{q}(t) + g\!\bigl(q(t)\bigr) + \tau_f\!\bigl(\dot{q}(t)\bigr)
$$
where $M \in \mathbb{R}^{J\times J}$ is the joint-space inertia matrix, $C \in \mathbb{R}^{J\times J}$ is the Coriolis–centripetal matrix, $g \in \mathbb{R}^J$ is the gravity load vector, and $\tau_f \in \mathbb{R}^J$ is a friction contribution. The analytical RNEA baseline computes the right-hand side from first principles; the three neural architectures learn corrections or full mappings from data.

---

## 2. Data Acquisition and Format

Each hardware execution run is logged as a structured JSON file (schema `hwrl_execution_log_v4`). The log records $N$ timestamped entries at the servo control/feedback rate (typically 240 Hz):

| Field | Symbol | Units |
|---|---|---|
| Timestamp | $t_k$ | s |
| Actual joint positions (encoder ticks) | $p_k \in \mathbb{Z}^6$ | ticks |
| Load register | $L_k \in \mathbb{R}^6$ | $\times 0.1\%$ of stall torque |
| Supply voltage | $V_k \in \mathbb{R}^6$ | V |
| Commanded position/velocity/acceleration | $q_k^{cmd}, \dot{q}_k^{cmd}, \ddot{q}_k^{cmd}$ | ticks, ticks/s, ticks/s² |

Raw encoder ticks are converted to joint angles in URDF frame via:
$$
q_i = d_i \cdot \left(p_i - p_i^{0}\right) \cdot c_{tr}
$$
where $d_i \in \{-1, +1\}$ is the URDF-aligned direction flag, $p_i^{0}$ is the zero-position tick count for joint $i$, and $c_{tr}$ is the ticks-to-radians conversion factor read from the log metadata.

---

## 3. Signal Processing and Preprocessing

### 3.1 Timestamp Repair

Raw timestamps $\{t_k\}_{k=0}^{N-1}$ can be non-monotonic or contain jitter from scheduler latency. A three-stage repair is applied (`robot_physics.fix_timestamps`):

1. **Monotonicity repair.** Any index $k$ where $\Delta t_k = t_k - t_{k-1} \leq 0$ is flagged. The defective sample is interpolated over the surrounding good indices via `numpy.interp`.

2. **Outlier-$\Delta t$ repair.** Steps where $\Delta t_k < \bar{t}/f_{out}$ or $\Delta t_k > \bar{t} \cdot f_{out}$ (with median $\bar{t}$ and outlier factor $f_{out} = 3$) are re-interpolated.

3. **Uniform resampling.** If the coefficient of variation $\text{CV}(\Delta t) = \sigma_{\Delta t}/\mu_{\Delta t}$ still exceeds $0.05$ after the two interpolation passes, the timeline is resampled uniformly at the median sample rate via `numpy.linspace`.

The repaired timeline provides a uniform step $h = \text{median}(\Delta t)$ used as the SG differentiation delta.

### 3.2 Savitzky-Golay Differentiation

Joint positions $q \in \mathbb{R}^{N \times J}$ (converted from ticks) are differentiated by fitting a local polynomial of degree $p = 3$ over a sliding window of length $w = 25$ samples. Both velocity and acceleration come from the **same** polynomial fit:
$$
\dot{q}_k \approx \left.\frac{d}{dt}\hat{P}(t)\right|_{t=t_k}, \qquad
\ddot{q}_k \approx \left.\frac{d^2}{dt^2}\hat{P}(t)\right|_{t=t_k}
$$
ensuring that $\ddot{q}$ is the exact time derivative of $\dot{q}$ within the polynomial approximation. Using the same pass eliminates the phase lag that would accumulate from two independent filters. The SG window $w$ is validated to satisfy:
$$
w \equiv 1 \pmod{2}, \quad w > p, \quad w \leq N
$$
Outputs are clipped to safety bounds: $|\dot{q}_{ij}| \leq 100\ \text{rad/s}$, $|\ddot{q}_{ij}| \leq 1000\ \text{rad/s}^2$.

### 3.3 Measured Torque Computation

The servo load register $L_k \in [-1000, +1000]$ (signed, unit = 0.1% of stall) is converted to physical torque in the URDF frame:
$$
\tau_k^{load} = -d_i \cdot \frac{L_{k,i}}{1000} \cdot \bar{\tau}_i \cdot \frac{V_{k,i}}{V_{nom}} \cdot c_{Nm}
$$
where $\bar{\tau}_i$ is the per-joint rated stall torque (kgf·cm), $V_{nom} = 12.0\ \text{V}$, and $c_{Nm} = 0.09807\ \text{N·m/(kgf·cm)}$. The direction factor $-d_i$ aligns the signed load register with the URDF convention.

### 3.4 RNEA Torque Decomposition

The Pinocchio rigid-body library (`pinocchio.rnea`) is used to evaluate the Newton-Euler recursive algorithm at $O(n)$ complexity. The total rigid-body torque and its additive components are isolated through three RNEA calls per timestep:
$$
\tau_g = \text{RNEA}(q,\, \mathbf{0},\, \mathbf{0}), \qquad [\text{gravity only}]
$$
$$
\tau_M = \text{RNEA}(q,\, \mathbf{0},\, \ddot{q}) - \tau_g, \qquad [\text{inertia only}]
$$
$$
\tau_C = \text{RNEA}(q,\, \dot{q},\, \mathbf{0}) - \tau_g, \qquad [\text{Coriolis/centripetal only}]
$$
$$
\tau_{rnea} = \tau_g + \tau_M + \tau_C \qquad [\text{total rigid-body}]
$$
This decomposition is exact: by linearity of the manipulator equation in $\ddot{q}$, setting $\dot{q} = \mathbf{0}$ when computing $\tau_M$ and $\ddot{q} = \mathbf{0}$ when computing $\tau_C$ isolates each additive component without approximation.

The URDF mass model is calibrated by a global density scale $\alpha$ (accounting for PLA infill vs. nominal solid density) before passing to Pinocchio:
$$
m_i \leftarrow \alpha \cdot m_i, \quad I_i \leftarrow \alpha \cdot I_i, \quad \forall i \in \{1,\ldots,J\}
$$
with calibrated value $\alpha = 0.0931$ (Neural_Networks module) / $\alpha = 0.1119$ (Torque_Analysis module, bulk WLS calibration over 470k samples).

### 3.5 Friction Torque

Joint friction is modelled as smooth Coulomb plus viscous:
$$
\tau_f(\dot{q}) = c \odot \tanh\!\left(\frac{\dot{q}}{\epsilon}\right) + v \odot \dot{q}
$$
with calibrated per-joint Coulomb vector $c \in \mathbb{R}^J$, viscous vector $v \in \mathbb{R}^J$, and stiction half-width $\epsilon = 0.0405\ \text{rad/s}$. The $\tanh$ replaces the discontinuous $\text{sgn}(\dot{q})$ of the classical Coulomb model, ensuring $C^\infty$ continuity throughout the velocity range. The total analytical torque is:
$$
\tau_{phys} = \tau_g + \tau_M + \tau_C + \tau_f
$$
The 20-dimensional physics tensor $[\tau_g^\top, \tau_M^\top, \tau_C^\top, \tau_f^\top]^\top \in \mathbb{R}^{4J}$ is stored in `filtered_tau_decomposed.csv` for use by physics-informed models.

---

## 4. Dataset Construction and Normalization

### 4.1 Train/Validation/Test Split

Preprocessed trajectories are assigned chronologically to three non-overlapping splits with ratios $70\% / 15\% / 15\%$ (train/val/test). The full dataset contains approximately 369,000 samples:
$$
N_{train} = 272{,}465, \quad N_{val} = 53{,}779, \quad N_{test} = 43{,}093
$$

### 4.2 Feature Normalization

All input and physics signals are independently Z-score normalized using statistics computed **only on the training split** and stored in `metadata.json`:
$$
\tilde{z} = \frac{z - \mu_z}{\max(\sigma_z,\,10^{-8})}, \quad z \in \{q,\, \dot{q},\, \ddot{q}\}
$$
The concatenated kinematic feature vector is:
$$
\tilde{x} = \left[\tilde{q}^\top,\, \tilde{\dot{q}}^\top,\, \tilde{\ddot{q}}^\top\right]^\top \in \mathbb{R}^{3J}
$$

### 4.3 Sum-Consistent Physics Normalization

The four physics components must remain sum-consistent after normalization: the sum of normalized components should equal the normalized total torque. Let $\mu_\tau, \sigma_\tau$ be the training-set mean and std of the measured torque. Each physics component is normalized by:
$$
\tilde{\tau}_k = \frac{\tau_k - \mu_\tau / 4}{\sigma_\tau}, \quad k \in \{g, M, C, f\}
$$
This distributes the total mean equally across the four components and uses the same scale factor, so:
$$
\sum_{k \in \{g,M,C,f\}} \tilde{\tau}_k = \frac{(\tau_g + \tau_M + \tau_C + \tau_f) - \mu_\tau}{\sigma_\tau} = \frac{\tau_{phys} - \mu_\tau}{\sigma_\tau} \approx \tilde{\tau}_{target}
$$
This invariant is essential for physics-informed loss terms that operate in normalized space.

---

## 5. Neural Network Training Methodology

### 5.1 Optimizer and Learning Rate Schedule

All models share the same optimizer and schedule. Parameters are updated by AdamW:
$$
\theta_{t+1} = \theta_t - \eta_t\, \frac{\hat{m}_t}{\hat{v}_t^{1/2} + \epsilon} - \eta_t \lambda \theta_t
$$
with learning rate $\eta_0 = 3\times10^{-4}$, weight decay $\lambda \in \{0.005,\, 0.002\}$ depending on the model, $\beta_1 = 0.9$, $\beta_2 = 0.999$.

The learning rate follows a **warm-up cosine** schedule over $E$ total epochs:
$$
\eta(e) = \begin{cases}
\eta_0\!\left[0.1 + 0.9\,\frac{e}{e_{warm}}\right] & e < e_{warm} \\[4pt]
\eta_0\!\left[\eta_{min} + (1 - \eta_{min})\cdot\tfrac{1 + \cos(\pi\,p(e))}{2}\right] & e \geq e_{warm}
\end{cases}
$$
where $e_{warm} = \lfloor E/20 \rfloor$, $\eta_{min} = 0.01$, and $p(e) = (e - e_{warm})/(E - e_{warm}) \in [0,1]$ is the fractional progress through the cosine decay phase.

### 5.2 Joint-Weighted MSE Loss

The primary data fidelity loss uses per-joint weighting to handle the wide dynamic range across joints:
$$
\mathcal{L}_{data}(\hat{\tau}, \tau^*) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^J w_j \left(\hat{\tau}_{ij} - \tau^*_{ij}\right)^2
$$
with weights $w = [1.0,\ 2.5,\ 1.0,\ 1.0,\ 1.0]$ for J1–J5. J2 (shoulder) receives $2.5\times$ weight because it carries the highest gravitational and inertial load and dominates the macro RMSE.

### 5.3 Feature Noise Regularization

During training, Gaussian noise $\xi \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 I)$ with $\sigma_n = 0.02$ is added to the normalized feature vector:
$$
\tilde{x}_{noisy} = \tilde{x} + \xi
$$
This acts as a stochastic Tikhonov regularizer, penalizing sharp input-output gradients and improving generalization to unseen encoder noise patterns.

### 5.4 Gradient Clipping and Early Stopping

Parameter gradients are clipped by global L2 norm before each update:
$$
g \leftarrow g \cdot \min\!\left(1,\, \frac{G_{clip}}{\|g\|_2}\right), \quad G_{clip} = 5.0
$$
Training halts early when the physical-units validation RMSE (N·m) does not improve by more than $\delta_{min} = 10^{-4}$ for $P$ consecutive epochs. The patience counter is held at zero during the warm-up phase to prevent premature termination while the LR is intentionally suppressed.

### 5.5 Evaluation Metrics

Performance is reported in physical units (N·m) on the held-out test set. Key metrics:
$$
\text{RMSE}_j = \sqrt{\frac{1}{N_{test}}\sum_{i=1}^{N_{test}} \left(\hat{\tau}_{ij} - \tau^*_{ij}\right)^2}, \quad
\text{RMSE}_{macro} = \frac{1}{J}\sum_{j=1}^J \text{RMSE}_j
$$
$$
R^2_j = 1 - \frac{\sum_i (\hat{\tau}_{ij} - \tau^*_{ij})^2}{\sum_i (\tau^*_{ij} - \bar{\tau}^*_j)^2}, \quad
\text{NRMSE}_j = \frac{\text{RMSE}_j}{\max_i(\tau^*_{ij}) - \min_i(\tau^*_{ij})}
$$
The macro RMSE is averaged across trajectory segments (not pooled over all samples) to weight each trajectory equally regardless of length.

---

## 6. Model Hierarchy Overview

| Model | Input dim | Architecture | Physics coupling | Loss |
|---|---|---|---|---|
| Analytical (RNEA) | — | Rigid-body + friction | Full (first principles) | — |
| BlackBoxFNN | $3J = 15$ | MLP [256, 512, 256] | None | $\mathcal{L}_{data}$ |
| PhysicsRegularizedFNN | $7J = 35$ | MLP [256, 512, 256] | Soft (input augmentation + penalty) | $\mathcal{L}_{data} + \alpha\,\mathcal{L}_{phys}$ |
| ResidualCorrectionFNN | $7J = 35$ | MLP [128, 256, 128] | Hard (additive structure + tanh bound) | $\mathcal{L}_{data} + \alpha_{reg}\,\|\Delta\tau\|^2$ |
