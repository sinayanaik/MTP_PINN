# Friction Calibration — Detailed Technical README

## Table of Contents

1. [Overview](#1-overview)
2. [Symbol Table & Notation](#2-symbol-table--notation)
3. [Physical Background](#3-physical-background)
4. [The Friction Signal — What We Fit](#4-the-friction-signal--what-we-fit)
5. [The Smooth Friction Model](#5-the-smooth-friction-model)
6. [Data Pipeline](#6-data-pipeline)
7. [Diagnostics](#7-diagnostics)
8. [Method A — Regime Analysis](#8-method-a--regime-analysis)
9. [Method B — Asymmetry Analysis](#9-method-b--asymmetry-analysis)
10. [Method C — Bias-Aware ε Sweep](#10-method-c--bias-aware-ε-sweep)
11. [Method D — Nonlinear Per-Joint ε](#11-method-d--nonlinear-per-joint-ε)
12. [Per-Trajectory Consistency](#12-per-trajectory-consistency)
13. [Synthesis — Cross-Method Weighted Average](#13-synthesis--cross-method-weighted-average)
14. [Validation & Output](#14-validation--output)
15. [Visualization Suite](#15-visualization-suite)
16. [Design History & Lessons Learned](#16-design-history--lessons-learned)
17. [Usage](#17-usage)
18. [Complete Function Reference](#18-complete-function-reference)

---

## 1. Overview

`calibrate_friction.py` identifies three scalar-per-joint parameters — **Coulomb friction** $c_j$, **viscous friction** $v_j$, and a shared **transition width** $\varepsilon$ — that define how much torque each servo joint loses to internal mechanical friction. It does this by comparing measured motor torques against a rigid-body dynamics model (RNEA via Pinocchio), attributing the residual to friction, and fitting a smooth parametric curve through that residual.

Four independent estimation methods are run, cross-compared, and fused into a single recommendation via trust-weighted averaging. The output is a drop-in replacement for the friction arrays in `config.py`.

**Design philosophy**: Every method has a weakness. Regime analysis is noisy. The ε sweep absorbs gravity error into a bias but requires a global ε. The nonlinear fit matches the deployment model exactly but can overfit. Asymmetry analysis is robust to gravity error but needs velocity diversity. Running all four and comparing them provides **robustness through redundancy**.

---

## 2. Symbol Table & Notation

### 2.1 Primary Physical Quantities

| Symbol | Units | Dimensions | Description |
|--------|-------|------------|-------------|
| $\mathbf{q}(t)$ | rad | $\mathbb{R}^{n_q}$ | Joint position vector at time $t$ |
| $\dot{\mathbf{q}}(t)$ | rad/s | $\mathbb{R}^{n_q}$ | Joint velocity vector (numerical derivative of $\mathbf{q}$) |
| $\ddot{\mathbf{q}}(t)$ | rad/s² | $\mathbb{R}^{n_q}$ | Joint acceleration vector (numerical derivative of $\dot{\mathbf{q}}$) |
| $\boldsymbol{\tau}_{\text{load}}(t)$ | N·m | $\mathbb{R}^{n_q}$ | Measured motor torque (from current sensing) |
| $\boldsymbol{\tau}_{\text{RNEA}}(t)$ | N·m | $\mathbb{R}^{n_q}$ | Predicted torque from rigid-body dynamics |
| $\boldsymbol{\tau}_f^{\text{signal}}(t)$ | N·m | $\mathbb{R}^{n_q}$ | Friction signal: $\boldsymbol{\tau}_{\text{load}} - \boldsymbol{\tau}_{\text{RNEA}}$ |
| $n_q$ | — | scalar | Number of joints in the kinematic chain |
| $t$ | s | scalar | Continuous time index |

### 2.2 Friction Model Parameters (Per Joint $j$)

| Symbol | Units | Bounds | Description |
|--------|-------|--------|-------------|
| $c_j$ | N·m | $[0,\; 0.50]$ | Coulomb (dry) friction coefficient |
| $v_j$ | N·m·s/rad | $[0,\; 0.30]$ | Viscous (velocity-proportional) damping |
| $\varepsilon$ | rad/s | $[0.02,\; 0.50]$ | Transition width of the smooth sign function |
| $b_j$ | N·m | unbounded | Bias term (absorbs mean gravity model error) |

### 2.3 Fitting-Specific Notation

| Symbol | Context | Description |
|--------|---------|-------------|
| $\boldsymbol{\Phi}_j(\varepsilon)$ | Method C | Design matrix $\in \mathbb{R}^{N \times 3}$ for joint $j$ |
| $\mathbf{x}_j$ | Method C | Parameter vector $[b_j,\; c_j,\; v_j]^T$ |
| $\mathbf{y}_j$ | Method C | Observation vector $[\tau_{f,j,1}^{\text{signal}}, \ldots, \tau_{f,j,N}^{\text{signal}}]^T$ |
| $a^+_j,\; a^-_j$ | Method B | Intercepts from positive/negative velocity half-fits |
| $v^+_j,\; v^-_j$ | Method B | Slopes from positive/negative velocity half-fits |
| $\text{RMS}_j(\varepsilon)$ | Method C | Root-mean-square residual for joint $j$ at transition width $\varepsilon$ |
| $\text{CV}_j$ | Per-trajectory | Coefficient of variation: $100 \times \sigma / \mu$ (%) |
| $w_m$ | Synthesis | Trust weight for method $m$ |

### 2.4 Operators and Notation Conventions

| Notation | Meaning |
|----------|---------|
| $\odot$ | Hadamard (element-wise) product |
| $\text{clip}(x, a, b)$ | $\max(a, \min(x, b))$ — clamping to interval $[a, b]$ |
| $\text{sign}(x)$ | Classical signum: $+1$ if $x>0$, $-1$ if $x<0$, $0$ if $x=0$ |
| $\tanh(x/\varepsilon)$ | Smooth approximation to $\text{sign}(x)$ |
| $\|\cdot\|$ | Euclidean norm unless stated otherwise |
| subscript $j$ | Joint index ($j = 0, 1, \ldots, n_q - 1$) |
| subscript $k$ | Sample (time-step) index |
| superscript $(k)$ | Trajectory file index in per-trajectory analysis |

---

## 3. Physical Background

### 3.1 Friction in Servo Gearboxes

Every revolute joint in a robotic arm dissipates energy through two dominant mechanisms:

**Coulomb (dry) friction** arises from solid-to-solid contact between gear teeth. It produces a resistive torque of approximately constant magnitude that always opposes the direction of motion:

$$\tau_{\text{Coulomb},j} = c_j \cdot \text{sign}(\dot{q}_j)$$

Physically, $c_j$ depends on gear preload, lubrication condition, and contact geometry. It is roughly constant for a given joint at a given temperature.

**Viscous friction** arises from shearing of lubricant films in bearings and gear meshes. It produces a torque proportional to velocity:

$$\tau_{\text{viscous},j} = v_j \cdot \dot{q}_j$$

The coefficient $v_j$ depends on lubricant viscosity, bearing geometry, and temperature.

**Combined ideal model:**

$$\tau_{f,j}^{\text{ideal}}(\dot{q}_j) = c_j \cdot \text{sign}(\dot{q}_j) + v_j \cdot \dot{q}_j$$

This is the classical Coulomb + viscous friction model used throughout robotics (Siciliano et al., *Robotics*, Springer, 2009, Ch. 7.4).

### 3.2 Why This Model Is Sufficient

More complex friction models exist (Stribeck effect, LuGre dynamic friction, etc.), but for **calibrating a PINN residual compensator**, the Coulomb+viscous model captures >90% of the friction torque. The PINN will learn whatever structure remains in the residual:

$$\boldsymbol{\tau}_{\text{residual}} = \boldsymbol{\tau}_{\text{load}} - \boldsymbol{\tau}_{\text{RNEA}} - \boldsymbol{\tau}_f$$

A more complex friction model would reduce $\boldsymbol{\tau}_{\text{residual}}$ but add calibration complexity and risk of overfitting.

---

## 4. The Friction Signal — What We Fit

### 4.1 Signal Extraction

The input to all calibration methods is the **difference between measured and modeled torque**:

$$\boldsymbol{\tau}_f^{\text{signal}}(t) = \boldsymbol{\tau}_{\text{load}}(t) - \boldsymbol{\tau}_{\text{RNEA}}(t)$$

**Code reference:** `load_friction_data()`, lines:
```python
tau_fric_signal = tau_load[:, :nq] - tau_urdf
```

where:
- `tau_load` comes from `torque_from_load()` — converts motor current/voltage readings to torque in N·m
- `tau_urdf` comes from `torque_from_urdf()` — runs Pinocchio's RNEA (Recursive Newton-Euler Algorithm) on the URDF model with numerically differentiated $\dot{\mathbf{q}}$ and $\ddot{\mathbf{q}}$

### 4.2 Signal Contamination

This signal is **not pure friction**. It contains:

$$\tau_{f,j}^{\text{signal}}(t) = \underbrace{c_j \tanh\!\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j \dot{q}_j}_{\text{true friction}} + \underbrace{\Delta g_j(\mathbf{q}(t))}_{\text{gravity model error}} + \underbrace{\eta_j(t)}_{\text{noise, unmodeled effects}}$$

The gravity model error $\Delta g_j(\mathbf{q})$ arises because:
- URDF mass/inertia parameters are approximate (from CAD, not measured)
- Mass scaling (`MASS_SCALE` in `config.py`) and extra masses (`EXTRA_MASSES`) are estimates
- Link center-of-mass positions in the URDF may be inaccurate

**This contamination is the central challenge.** All four methods handle it differently:

| Method | How it handles $\Delta g_j$ |
|--------|---------------------------|
| A (Regime) | Ignores it — vulnerable |
| B (Asymmetry) | Exploits antisymmetry to cancel it — robust |
| C (Bias-Aware Sweep) | Absorbs mean $\Delta g$ into bias $b_j$ — partially robust |
| D (Nonlinear) | Ignores it — fits config.py model directly |

### 4.3 Data Loading Paths

**Single file mode** (default):
```python
qd_cat, tau_f_cat, q_cat, rid = load_friction_data(C.LOG_JSON, model)
```
Loads one JSON log file specified in `config.py`. Fast but potentially unrepresentative.

**Bulk mode** (`--bulk` flag):
```python
json_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
```
**Code reference:** `load_all_data()` — iterates over all JSON files in `raw_samples/`, concatenates vertically.

Bulk mode is critical because:
- Trajectory diversity averages out configuration-dependent gravity error
- More velocity samples → better conditioning of the regression
- Per-trajectory consistency analysis requires multiple files

The function returns both concatenated arrays (for global fitting) and per-file arrays (for consistency analysis):
```python
return qd_cat, tau_f_cat, q_cat, per_file
```

---

## 5. The Smooth Friction Model

### 5.1 The Discontinuity Problem

The ideal model uses $\text{sign}(\dot{q})$, which is discontinuous at $\dot{q} = 0$:

$$\text{sign}(x) = \begin{cases} +1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}$$

This is problematic for three reasons:
1. **Numerical chatter**: near zero velocity, sensor noise causes rapid sign flips
2. **Undefined gradient**: $d/dx\;\text{sign}(x)$ does not exist at $x=0$ — breaks backpropagation in PINN training
3. **Poor conditioning**: least-squares fitting with a discontinuous basis function is ill-conditioned

### 5.2 Smooth Approximation

Replace $\text{sign}(\dot{q})$ with the hyperbolic tangent:

$$\text{smooth\_sign}(\dot{q}_j) = \tanh\!\left(\frac{\dot{q}_j}{\varepsilon}\right)$$

**Properties of $\tanh(x/\varepsilon)$:**

| Regime | Behavior |
|--------|----------|
| $\|x\| \gg \varepsilon$ | $\tanh(x/\varepsilon) \approx \text{sign}(x)$ — recovers ideal model |
| $\|x\| \ll \varepsilon$ | $\tanh(x/\varepsilon) \approx x/\varepsilon$ — linear (smooth at zero) |
| $\varepsilon \to 0$ | $\tanh(x/\varepsilon) \to \text{sign}(x)$ — exact recovery |

The derivative is smooth and bounded everywhere:

$$\frac{d}{dx}\tanh\!\left(\frac{x}{\varepsilon}\right) = \frac{1}{\varepsilon}\,\text{sech}^2\!\left(\frac{x}{\varepsilon}\right)$$

This is essential for PINN training where gradients must flow through the friction model.

### 5.3 The Deployment Model

The model that goes into `config.py` and is used at runtime in `torque.py`:

$$\boxed{\tau_{f,j}(\dot{q}_j) = c_j \cdot \tanh\!\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j \cdot \dot{q}_j}$$

In vector form for all joints:

$$\boldsymbol{\tau}_f(\dot{\mathbf{q}}) = \mathbf{c} \odot \tanh\!\left(\frac{\dot{\mathbf{q}}}{\varepsilon}\right) + \mathbf{v} \odot \dot{\mathbf{q}}$$

**Parameters to calibrate:**

| Parameter | Array | Shape | Bounds | Example (J1) |
|-----------|-------|-------|--------|---------------|
| $c_j$ | `COULOMB_NM` | $(n_q,)$ | $[0, 0.50]$ N·m | 0.2118 |
| $v_j$ | `VISCOUS_NM` | $(n_q,)$ | $[0, 0.30]$ N·m·s/rad | 0.0239 |
| $\varepsilon$ | `FRICTION_EPS` | scalar | $[0.02, 0.50]$ rad/s | 0.0757 |

**Code reference — bounds defined at module level:**
```python
COULOMB_BOUNDS = (0.0, 0.50)       # N·m
VISCOUS_BOUNDS = (0.0, 0.30)       # N·m·s/rad
EPS_BOUNDS     = (0.02, 0.50)      # rad/s
```

### 5.4 Physical Interpretation of $\varepsilon$

$\varepsilon$ controls the **width of the dead zone** around $\dot{q} = 0$ where friction transitions smoothly from negative to positive.

- **Small $\varepsilon$** (e.g., 0.02 rad/s): sharp transition, close to ideal $\text{sign}(\cdot)$. Better for joints with high stiction.
- **Large $\varepsilon$** (e.g., 0.50 rad/s): wide transition, friction is nearly linear near zero. Better for joints with soft engagement.

At $|\dot{q}| = 3\varepsilon$, $\tanh(3) = 0.995 \approx 1$, so the Coulomb plateau is effectively reached at three times $\varepsilon$.

### 5.5 Shape of the Friction Curve

```
  τ_f
   ↑
   |         ╱ slope = v_j (viscous)
   |       ╱
   c_j ───•─╱─────────────── (Coulomb plateau)
   |     ╱
   |   ╱  ← tanh transition (width ~2ε)
   | ╱
───┼──────────────────────→ q̇
   |╲
   |  ╲
  -c_j──•─╲──────────────── (negative plateau)
   |      ╲
   |        ╲ slope = v_j
```

The curve is an odd function (antisymmetric): $\tau_f(-\dot{q}) = -\tau_f(\dot{q})$.

---

## 6. Data Pipeline

### 6.1 End-to-End Flow

```
JSON log file
    │
    ▼
load_log() ──→ raw arrays: act_pos (ticks), load (raw), voltage, timestamps
    │
    ▼
ticks_to_radians() ──→ q (rad), shape (N, DOF)
    │
    ├──→ torque_from_load() ──→ τ_load (N·m), shape (N, DOF)
    │
    ├──→ torque_from_urdf() ──→ τ_RNEA (N·m), q̇ (rad/s), q̈ (rad/s²)
    │          │
    │          └── Pinocchio RNEA: M(q)q̈ + C(q,q̇)q̇ + g(q)
    │
    ▼
τ_f_signal = τ_load − τ_RNEA ──→ shape (N, nq)
    │
    ▼
[Method A] [Method B] [Method C] [Method D]
    │          │          │          │
    └──────────┴──────────┴──────────┘
                    │
                    ▼
            synthesize_recommendation()
                    │
                    ▼
        final c_j, v_j, ε → config.py
```

### 6.2 Key Data Shapes

For bulk mode with $K$ files and $N_k$ samples per file:

| Variable | Shape | Description |
|----------|-------|-------------|
| `qd` (concatenated) | $(N_{\text{total}}, n_q)$ | All velocities stacked |
| `tau_f` (concatenated) | $(N_{\text{total}}, n_q)$ | All friction signals stacked |
| `q` (concatenated) | $(N_{\text{total}}, n_q)$ | All positions stacked |
| `per_file[k]["qd"]` | $(N_k, n_q)$ | Velocities from file $k$ |

where $N_{\text{total}} = \sum_{k=1}^{K} N_k$ (typically ~470,000 for 124 files).

---

## 7. Diagnostics

Two diagnostic functions run **before** any fitting, to characterize the data.

### 7.1 Velocity Statistics

**Code reference:** `print_velocity_stats()`

Computes percentiles $P_{50}, P_{90}, P_{95}, P_{99}$, and max of $|\dot{q}_j|$ for each joint. This reveals:
- Whether there is sufficient velocity diversity for fitting
- Whether high-speed data exists (needed for viscous estimation)
- Whether any joints are mostly stationary

### 7.2 Correlation Diagnostic

**Code reference:** `correlation_diagnostic()`

Computes two Pearson correlation coefficients per joint:

$$r_{\text{vel},j} = \text{corr}(\tau_{f,j}^{\text{signal}},\; \dot{q}_j)$$

$$r_{\text{pos},j} = \text{corr}(\tau_{f,j}^{\text{signal}},\; q_j)$$

**Interpretation:**
- High $|r_{\text{vel}}|$ is **expected** — friction depends on velocity
- High $|r_{\text{pos}}|$ is **concerning** — it means gravity model error (which depends on position) is leaking into the friction signal
- Flag raised if $|r_{\text{pos}}| > 0.5$

**Mitigation**: Use `--bulk` for trajectory diversity, which decorrelates $q$ from any single trajectory's configuration sequence.

---

## 8. Method A — Regime Analysis

### 8.1 Concept

Exploit the physical structure of the friction model in two velocity regimes.

**Code reference:** `regime_analysis()`

### 8.2 Low-Speed Regime: Coulomb Estimation

For samples where $|\dot{q}_j| \in (0.01, 0.05)$ rad/s:

At these speeds, $\tanh(\dot{q}_j / \varepsilon) \approx \pm 1$ (for $\varepsilon < 0.05$), and the viscous term $v_j \dot{q}_j$ is negligibly small. Therefore:

$$\tau_{f,j}^{\text{signal}} \approx \pm c_j + \Delta g_j(\mathbf{q}) + \eta_j$$

Taking the median of absolute values absorbs the sign and is robust to outliers:

$$\hat{c}_j^{\text{regime}} = \text{median}\!\left(|\tau_{f,j}^{\text{signal}}| \;\Big|\; |\dot{q}_j| \in (0.01, 0.05)\right)$$

**Code:**
```python
low = (np.abs(qd[:, j]) > 0.01) & (np.abs(qd[:, j]) < LOW)
c_est = np.median(np.abs(tau_f[low, j]))
```

The lower bound of 0.01 rad/s excludes truly stationary samples where the friction signal is dominated by static friction (not modeled).

**Fallback:** If fewer than 20 low-speed samples exist, $\hat{c}_j = 0.05$ N·m.

### 8.3 High-Speed Regime: Viscous Estimation

For samples where $|\dot{q}_j| > 0.10$ rad/s:

Subtract the Coulomb component, then fit a line through the origin:

$$\tau_{f,j}^{\text{signal}} - \hat{c}_j \cdot \text{sign}(\dot{q}_j) \approx v_j \cdot \dot{q}_j$$

This is a zero-intercept ordinary least squares (OLS) problem. The single-regressor OLS solution is:

$$\hat{v}_j^{\text{regime}} = \frac{\displaystyle\sum_{k \in \text{high}} \dot{q}_{j,k} \cdot \left(\tau_{f,j,k}^{\text{signal}} - \hat{c}_j \cdot \text{sign}(\dot{q}_{j,k})\right)}{\displaystyle\sum_{k \in \text{high}} \dot{q}_{j,k}^2}$$

In matrix form with $\mathbf{A} = [\dot{q}_{j,k}]_{k \in \text{high}} \in \mathbb{R}^{n_{\text{high}} \times 1}$ and $\mathbf{r} = [\tau_{f,j,k}^{\text{signal}} - \hat{c}_j \text{sign}(\dot{q}_{j,k})]$:

$$\hat{v}_j = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{r}$$

**Code:**
```python
resid = tau_f[high, j] - c_est * np.sign(qd[high, j])
A = qd[high, j].reshape(-1, 1)
v_est = float(np.linalg.lstsq(A, resid, rcond=None)[0][0])
v_est = np.clip(v_est, 0.0, VISCOUS_BOUNDS[1])
```

**Post-hoc clamping** enforces $\hat{v}_j \in [0, 0.30]$ — negative viscous friction is unphysical, and values above 0.30 N·m·s/rad are implausible for small servo gearboxes.

**Fallback:** If fewer than 50 high-speed samples, $\hat{v}_j = 0.01$ N·m·s/rad.

### 8.4 Weakness

Gravity error $\Delta g_j(\mathbf{q})$ contaminates both estimates directly. This method works well only when the URDF model is accurate or when trajectory diversity averages out $\Delta g$.

---

## 9. Method B — Asymmetry Analysis

### 9.1 Key Physical Insight

**Code reference:** `asymmetry_analysis()`

Friction is an **antisymmetric** (odd) function of velocity:

$$\tau_f(-\dot{q}) = -\tau_f(\dot{q})$$

Gravity error is **independent of velocity sign** — it depends on configuration $\mathbf{q}$, not the direction of motion:

$$\Delta g_j(\mathbf{q}) \text{ is the same whether the joint moves forward or backward}$$

This difference can be exploited to **algebraically separate friction from gravity error**.

### 9.2 Procedure

**Step 1: Split data by velocity sign.**

Define threshold $\varepsilon_{\text{thresh}} = 0.05$ rad/s (avoids the noisy transition zone):

$$\mathcal{S}_j^+ = \{k : \dot{q}_{j,k} > \varepsilon_{\text{thresh}}\}$$
$$\mathcal{S}_j^- = \{k : \dot{q}_{j,k} < -\varepsilon_{\text{thresh}}\}$$

**Code:**
```python
pos = qd[:, j] > threshold
neg = qd[:, j] < -threshold
```

**Step 2: Fit an affine model in each half.**

For the positive-velocity subset:

$$\tau_{f,j,k}^{\text{signal}} = a_j^+ + v_j^+ \dot{q}_{j,k} + \text{noise}, \quad k \in \mathcal{S}_j^+$$

For the negative-velocity subset:

$$\tau_{f,j,k}^{\text{signal}} = a_j^- + v_j^- \dot{q}_{j,k} + \text{noise}, \quad k \in \mathcal{S}_j^-$$

Each is a standard 2-parameter OLS problem:

$$\begin{bmatrix} 1 & \dot{q}_{j,1} \\ 1 & \dot{q}_{j,2} \\ \vdots & \vdots \end{bmatrix} \begin{bmatrix} a \\ v \end{bmatrix} = \begin{bmatrix} \tau_{f,j,1}^{\text{signal}} \\ \tau_{f,j,2}^{\text{signal}} \\ \vdots \end{bmatrix}$$

**Code:**
```python
A_pos = np.column_stack([np.ones(pos.sum()), qd[pos, j]])
x_pos = np.linalg.lstsq(A_pos, tau_f[pos, j], rcond=None)[0]
```

### 9.3 Decomposition of Intercepts

The intercepts from each half contain both friction and gravity bias:

$$a_j^+ = +c_j + b_j$$
$$a_j^- = -c_j + b_j$$

where $c_j$ is the Coulomb friction (antisymmetric part) and $b_j$ is the mean gravity error (symmetric part).

**Solving this 2×2 linear system:**

$$\boxed{\hat{c}_j^{\text{asym}} = \frac{a_j^+ - a_j^-}{2}}$$

$$\boxed{\hat{b}_j^{\text{asym}} = \frac{a_j^+ + a_j^-}{2}}$$

$$\boxed{\hat{v}_j^{\text{asym}} = \frac{v_j^+ + v_j^-}{2}}$$

**Code:**
```python
coulomb = (x_pos[0] - x_neg[0]) / 2
bias    = (x_pos[0] + x_neg[0]) / 2
viscous = (x_pos[1] + x_neg[1]) / 2
```

### 9.4 Why This Works — Geometric Interpretation

In the $(\dot{q}_j, \tau_f^{\text{signal}})$ plane:

```
  τ_f^signal
      ↑
      |          ╱ positive half: intercept a⁺ = +c + b
      |        ╱
  a⁺ ─ ─ ─ •
      |     ╱
  b ─ ─ ─ ╳ ─ ─ ─ ─ ─ ─ → q̇    ← gravity bias shifts both halves
      |      ╲                    equally (symmetric)
  a⁻ ─ ─ ─ ─ •
      |         ╲
      |           ╲ negative half: intercept a⁻ = −c + b
```

- The **midpoint** $(a^+ + a^-)/2 = b$ is the vertical shift from gravity error
- The **half-gap** $(a^+ - a^-)/2 = c$ is the friction-induced separation
- Gravity shifts both lines **equally**, so the gap between them is **pure friction**

### 9.5 Post-Processing

```python
coulomb = max(0.0, coulomb)                          # non-negative
viscous = np.clip(viscous, 0.0, VISCOUS_BOUNDS[1])   # physical bounds
```

### 9.6 Interpretation Metric

The bias-to-Coulomb ratio quantifies signal quality:

$$R_j = \frac{|b_j|}{c_j}$$

| $R_j$ | Interpretation | Implication |
|--------|----------------|-------------|
| $< 0.3$ | Clean friction signal | URDF model is accurate for this joint |
| $0.3 – 1.0$ | Significant gravity bias | Results still usable; PINN will handle residual |
| $> 1.0$ | Gravity-dominated | Friction estimate may be unreliable |

**Code:**
```python
ratio = abs(bias / coulomb) if coulomb > 0.01 else float('inf')
interp = ("gravity-dominated" if ratio > 1.0
           else "significant bias" if ratio > 0.3
           else "clean friction")
```

### 9.7 Minimum Data Requirements

The method requires at least 50 samples in each half:
```python
if pos.sum() < 50 or neg.sum() < 50:
    # skip this joint
```

---

## 10. Method C — Bias-Aware ε Sweep

### 10.1 Model Formulation

**Code reference:** `fit_bias_aware()` and `sweep_eps_bias()`

Extend the friction model with an explicit bias term to absorb mean gravity error:

$$\tau_{f,j}^{\text{signal}}(t) = b_j + c_j \cdot \tanh\!\left(\frac{\dot{q}_j(t)}{\varepsilon}\right) + v_j \cdot \dot{q}_j(t)$$

### 10.2 Key Observation: Linear in Parameters for Fixed ε

For a **fixed** value of $\varepsilon$, the function $\tanh(\dot{q}_j / \varepsilon)$ is a known nonlinear transform of the data. The model becomes **linear** in the three unknowns $(b_j, c_j, v_j)$:

$$\tau_{f,j}^{\text{signal}} = b_j \cdot 1 + c_j \cdot \tanh(\dot{q}_j / \varepsilon) + v_j \cdot \dot{q}_j$$

This is a standard multiple linear regression.

### 10.3 Design Matrix Construction

For joint $j$, stack all $N$ timesteps into a matrix equation:

$$\underbrace{\begin{bmatrix} 1 & \tanh(\dot{q}_{j,1}/\varepsilon) & \dot{q}_{j,1} \\ 1 & \tanh(\dot{q}_{j,2}/\varepsilon) & \dot{q}_{j,2} \\ \vdots & \vdots & \vdots \\ 1 & \tanh(\dot{q}_{j,N}/\varepsilon) & \dot{q}_{j,N} \end{bmatrix}}_{\boldsymbol{\Phi}_j(\varepsilon) \;\in\; \mathbb{R}^{N \times 3}} \underbrace{\begin{bmatrix} b_j \\ c_j \\ v_j \end{bmatrix}}_{\mathbf{x}_j} = \underbrace{\begin{bmatrix} \tau_{f,j,1}^{\text{signal}} \\ \tau_{f,j,2}^{\text{signal}} \\ \vdots \\ \tau_{f,j,N}^{\text{signal}} \end{bmatrix}}_{\mathbf{y}_j}$$

**Code:**
```python
phi = np.column_stack([
    np.ones(len(qd_j)),
    np.tanh(qd_j / eps),
    qd_j,
])
```

### 10.4 Analytical Least-Squares Solution

The minimizer of $\|\boldsymbol{\Phi}_j \mathbf{x}_j - \mathbf{y}_j\|^2$ is:

$$\mathbf{x}_j^*(\varepsilon) = \left(\boldsymbol{\Phi}_j^T \boldsymbol{\Phi}_j\right)^{-1} \boldsymbol{\Phi}_j^T \mathbf{y}_j$$

This is computed via `numpy.linalg.lstsq`, which uses SVD decomposition internally (numerically stable even for ill-conditioned $\boldsymbol{\Phi}$):

```python
x, _, _, _ = np.linalg.lstsq(phi, tau_f_j, rcond=None)
b = x[0]
c = np.clip(x[1], *COULOMB_BOUNDS)
v = np.clip(x[2], *VISCOUS_BOUNDS)
```

**Computational cost**: For each $\varepsilon$ and each joint, this is a $3 \times 3$ normal equation solve — $O(3^2 N)$ for forming $\boldsymbol{\Phi}^T \boldsymbol{\Phi}$ and $O(3^3)$ for inverting. With $N = 470{,}000$, this takes milliseconds. **No subsampling is needed.**

### 10.5 Per-Joint RMS Residual

After solving, compute the prediction and its RMS error:

$$\text{RMS}_j(\varepsilon) = \sqrt{\frac{1}{N}\sum_{k=1}^{N}\left(\tau_{f,j,k}^{\text{signal}} - b_j - c_j \tanh\!\left(\frac{\dot{q}_{j,k}}{\varepsilon}\right) - v_j \dot{q}_{j,k}\right)^2}$$

**Code:**
```python
pred = b + c * np.tanh(qd_j / eps) + v * qd_j
rms_val = np.sqrt(np.mean((tau_f_j - pred) ** 2))
```

### 10.6 Grid Search Over ε

Define an aggregate error across all active joints:

$$\text{RMS}_{\text{total}}(\varepsilon) = \sqrt{\frac{1}{n_{\text{active}}} \sum_{j=1}^{n_{\text{active}}} \text{RMS}_j^2(\varepsilon)}$$

Search for the optimal $\varepsilon$:

$$\varepsilon^* = \arg\min_{\varepsilon \in \{0.02,\; 0.0248,\; \ldots,\; 0.50\}} \text{RMS}_{\text{total}}(\varepsilon)$$

The grid has 100 linearly-spaced points:

**Code:**
```python
eps_range = np.linspace(*EPS_SWEEP_RANGE, EPS_SWEEP_POINTS)
# EPS_SWEEP_RANGE = (0.02, 0.50), EPS_SWEEP_POINTS = 100

for eps in eps_range:
    # ... fit all joints, compute total_rms
    if total_rms < best_rms:
        best_rms = total_rms
        best_eps = eps
```

Total sweep cost: $100 \times n_{\text{active}}$ least-squares solves, each $O(N)$ → completes in seconds even for $N = 470{,}000$.

### 10.7 Why the Bias Term Matters

Without bias (2-column design matrix):

$$\hat{c}_j^{\text{no bias}} = c_j^{\text{true}} + \frac{\text{cov}\!\left(\tanh(\dot{q}_j/\varepsilon),\; \Delta g_j\right)}{\text{var}\!\left(\tanh(\dot{q}_j/\varepsilon)\right)}$$

The gravity error $\Delta g_j(\mathbf{q})$ has nonzero mean and is partially correlated with $\tanh(\dot{q}_j/\varepsilon)$ through the trajectory. The leakage term contaminates $\hat{c}_j$.

With the bias column $\mathbf{1}$, the least-squares solution **first removes the mean** of $\mathbf{y}_j$, decorrelating the constant component of $\Delta g_j$ from the friction regressors. This is equivalent to fitting in a **de-meaned** space.

---

## 11. Method D — Nonlinear Per-Joint ε

### 11.1 Motivation

**Code reference:** `fit_nonlinear_joint()`

Methods A–C either use a global ε (sweep) or don't use ε at all (regime, asymmetry). Method D fits a **per-joint** $\varepsilon_j$ alongside $c_j$ and $v_j$, using the **exact model** that goes into `config.py` (no bias term):

$$\tau_{f,j}(\dot{q}_j) = c_j \cdot \tanh\!\left(\frac{\dot{q}_j}{\varepsilon_j}\right) + v_j \cdot \dot{q}_j$$

### 11.2 Optimization Problem

Since $\varepsilon_j$ appears inside $\tanh(\cdot)$, the model is **nonlinear in parameters**. We minimize the mean squared error:

$$\min_{c_j,\; v_j,\; \varepsilon_j} \;\; \frac{1}{N}\sum_{k=1}^{N} \left(\tau_{f,j,k}^{\text{signal}} - c_j \tanh\!\left(\frac{\dot{q}_{j,k}}{\varepsilon_j}\right) - v_j \dot{q}_{j,k}\right)^2$$

subject to box constraints:

$$c_j \in [0, 0.50], \qquad v_j \in [0, 0.30], \qquad \varepsilon_j \in [0.02, 0.50]$$

### 11.3 Solver

**L-BFGS-B** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bound constraints), a quasi-Newton method that:
- Handles box constraints natively
- Approximates the Hessian from gradient history (no explicit Hessian needed)
- Converges quadratically near the solution

**Code:**
```python
def cost(p):
    return np.mean(
        (tf_s - p[0] * np.tanh(qd_s / p[2]) - p[1] * qd_s) ** 2
    )

res = minimize(cost, x0=[c0, v0, eps0],
               bounds=[COULOMB_BOUNDS, VISCOUS_BOUNDS, EPS_BOUNDS],
               method='L-BFGS-B')
c, v, eps = res.x
```

### 11.4 Subsampling for Speed

Unlike Method C (which is analytical and instant), the iterative optimizer evaluates the cost function many times. For large datasets:

```python
NL_MAX_SAMPLES = 50000

if len(qd_j) > NL_MAX_SAMPLES:
    idx = np.random.choice(len(qd_j), NL_MAX_SAMPLES, replace=False)
    qd_s, tf_s = qd_j[idx], tau_f_j[idx]
```

50,000 samples is sufficient to capture the joint's full velocity distribution while keeping each `minimize()` call under ~1 second.

### 11.5 Warm-Starting from Method C

If the sweep was run first, its results provide excellent initial guesses:

```python
if sweep_params_dict:
    c0 = sweep_params_dict[j]["c"]
    v0 = sweep_params_dict[j]["v"]
    eps0 = best_sweep_eps
```

This dramatically reduces the number of L-BFGS-B iterations (typically converges in <20 iterations vs. >100 from a cold start).

### 11.6 Reducing Per-Joint ε to Global ε

Since `config.py` uses a single `FRICTION_EPS`, the per-joint estimates are aggregated:

$$\varepsilon_{\text{final}} = \text{median}\!\left(\left\{\varepsilon_j \;:\; \varepsilon_j \notin \{\varepsilon_{\min}, \varepsilon_{\max}\}\right\}\right)$$

Values at the bounds are excluded because they indicate the optimizer hit a constraint (uninformative about the true optimum):

```python
interior = [e for e in eps_vals
            if abs(e - EPS_BOUNDS[0]) > 0.005
            and abs(e - EPS_BOUNDS[1]) > 0.005]
if interior:
    final_eps = float(np.median(interior))
```

### 11.7 Diagnostic Flags

The code flags potential issues for each joint:

| Flag | Condition | Interpretation |
|------|-----------|---------------|
| `v_cap` | $v_j \approx 0.30$ | Viscous hit upper bound — may need more data |
| `ε_floor` | $\varepsilon_j \approx 0.02$ | Very sharp transition — check for overfitting |
| `ε_ceil` | $\varepsilon_j \approx 0.50$ | Very soft transition — possibly insufficient data |

---

## 12. Per-Trajectory Consistency

### 12.1 Concept

**Code reference:** `per_trajectory_consistency()`

If the friction parameters are physical (i.e., properties of the hardware, not artifacts of a particular trajectory), they should be **stable across trajectories**.

### 12.2 Procedure

For each of $K$ trajectory files, fit the bias-aware model at the globally optimal $\varepsilon^*$:

$$\forall k \in \{1, \ldots, K\}, \;\; \forall j: \quad \text{solve } \boldsymbol{\Phi}_j^{(k)}(\varepsilon^*) \mathbf{x}_j^{(k)} = \mathbf{y}_j^{(k)}$$

This produces $K$ estimates of $(b_j^{(k)}, c_j^{(k)}, v_j^{(k)})$ per joint.

**Code:**
```python
for pf in per_file:
    for j in range(min(active, nq_file)):
        b, c, v, _ = fit_bias_aware(
            pf["qd"][:, j], pf["tau_f"][:, j], eps
        )
        all_params[j]["c"].append(c)
        # ...
```

### 12.3 Stability Metric — Coefficient of Variation

$$\text{CV}_j = \frac{\sigma\!\left(c_j^{(1)}, \ldots, c_j^{(K)}\right)}{\mu\!\left(c_j^{(1)}, \ldots, c_j^{(K)}\right)} \times 100\%$$

**Code:**
```python
c_cv = 100 * cs.std() / max(cs.mean(), 1e-6)
stable = "✓" if c_cv < 30 and vs.std() < 0.05 else "~"
```

### 12.4 Interpretation

| $\text{CV}_j$ | Confidence | Meaning |
|----------------|------------|---------|
| $< 20\%$ | **HIGH** | Parameter is a hardware property |
| $20\% – 40\%$ | **MEDIUM** | Some trajectory dependence (gravity leak) |
| $> 40\%$ | **LOW** | Estimate is unreliable |

The combined stability condition requires **both** $\text{CV}(c_j) < 30\%$ **and** $\sigma(v_j) < 0.05$ N·m·s/rad.

---

## 13. Synthesis — Cross-Method Weighted Average

### 13.1 Concept

**Code reference:** `synthesize_recommendation()`

Each method has different strengths. Rather than picking one, the code computes a **trust-weighted average** of all available estimates.

### 13.2 Coulomb Friction Synthesis

$$c_j^{\text{final}} = \frac{\displaystyle\sum_{m \in \text{methods}} w_m^{(c)} \cdot \hat{c}_j^{(m)}}{\displaystyle\sum_{m \in \text{methods}} w_m^{(c)}}$$

| Method $m$ | Symbol | Weight $w_m^{(c)}$ | Justification |
|------------|--------|---------------------|---------------|
| B (Asymmetry) | `asym` | 3.0 | Most robust to gravity error |
| C (Sweep) | `sweep` | 2.0 | Full dataset, analytical |
| D (Nonlinear) | `nl` | 1.5 | Per-joint flexibility |
| A (Regime) | `regime` | 1.0 | No optimization, but noisy |

**Code:**
```python
candidates_c = []
if asym_est and j in asym_est and not np.isnan(ac) and ac > 0.01:
    candidates_c.append(("asym", ac, 3.0))
if sweep_params:
    candidates_c.append(("sweep", sc, 2.0))
if nl_params:
    candidates_c.append(("nl", nc, 1.5))
if j in regime_est and not np.isnan(rc):
    candidates_c.append(("regime", rc, 1.0))

total_w = sum(w for _, _, w in candidates_c)
final_c[j] = sum(val * w for _, val, w in candidates_c) / total_w
```

### 13.3 Viscous Friction Synthesis

$$v_j^{\text{final}} = \frac{\displaystyle\sum_{m \in \text{methods}} w_m^{(v)} \cdot \hat{v}_j^{(m)}}{\displaystyle\sum_{m \in \text{methods}} w_m^{(v)}}$$

| Method $m$ | Weight $w_m^{(v)}$ | Justification |
|------------|---------------------|---------------|
| C (Sweep) | 3.0 | Full data, analytical, bias-absorbed |
| D (Nonlinear) | 2.0 | Config-compatible model |
| B (Asymmetry) | 2.0 | Gravity-robust |
| A (Regime) | 1.0 | Noisy |

### 13.4 Transition Width Synthesis

Priority chain:
1. If Method D ran → median of interior per-joint $\varepsilon_j$ values
2. Else if Method C ran → $\varepsilon^*$ from the sweep
3. Else → default 0.03 rad/s

### 13.5 Final Clamping

All synthesized values are clamped to physical bounds:

$$c_j^{\text{final}} \leftarrow \text{clip}(c_j^{\text{final}},\; 0,\; 0.50)$$
$$v_j^{\text{final}} \leftarrow \text{clip}(v_j^{\text{final}},\; 0,\; 0.30)$$
$$\varepsilon^{\text{final}} \leftarrow \text{clip}(\varepsilon^{\text{final}},\; 0.02,\; 0.50)$$

### 13.6 Fallback Logic

If a method didn't run or has insufficient data for a joint, it is simply excluded from the weighted average. If **no** method produced an estimate:
- $c_j = 0.10$ N·m (conservative default)
- $v_j = 0.01$ N·m·s/rad (conservative default)

---

## 14. Validation & Output

### 14.1 RMS Improvement Check

For each joint, compare old (current `config.py`) and new parameters:

$$\text{RMS}_j^{\text{old}} = \sqrt{\frac{1}{N}\sum_{k=1}^{N}\left(\tau_{f,j,k}^{\text{signal}} - c_j^{\text{old}} \tanh\!\left(\frac{\dot{q}_{j,k}}{\varepsilon^{\text{old}}}\right) - v_j^{\text{old}} \dot{q}_{j,k}\right)^2}$$

$$\text{RMS}_j^{\text{new}} = \sqrt{\frac{1}{N}\sum_{k=1}^{N}\left(\tau_{f,j,k}^{\text{signal}} - c_j^{\text{new}} \tanh\!\left(\frac{\dot{q}_{j,k}}{\varepsilon^{\text{new}}}\right) - v_j^{\text{new}} \dot{q}_{j,k}\right)^2}$$

$$\Delta\%_j = \frac{\text{RMS}_j^{\text{new}} - \text{RMS}_j^{\text{old}}}{\text{RMS}_j^{\text{old}}} \times 100$$

Negative $\Delta\%$ means improvement.

**Code:**
```python
tau_old_j = (C.COULOMB_NM[j] * np.tanh(qd[:, j] / C.FRICTION_EPS)
             + C.VISCOUS_NM[j] * qd[:, j])
tau_new_j = (final_c[j] * np.tanh(qd[:, j] / final_eps)
             + final_v[j] * qd[:, j])
old_r = rms(tau_f[:, j] - tau_old_j)
new_r = rms(tau_f[:, j] - tau_new_j)
pct = (new_r - old_r) / old_r * 100
```

### 14.2 Final Output Format

The script prints values ready to paste into `config.py`:

```
COULOMB_NM   = np.array([0.2118, 0.1542, 0.0893, 0.0654, 0.0321, 0.0198])
VISCOUS_NM   = np.array([0.0239, 0.0187, 0.0312, 0.0145, 0.0098, 0.0056])
FRICTION_EPS = 0.0757
```

### 14.3 Warning System

| Warning | Condition | Recommendation |
|---------|-----------|----------------|
| Gravity bias | $|b_j| > 0.05$ N·m | PINN will learn the residual |
| Viscous cap | $v_j \approx 0.30$ | Run `--bulk` for more data |

---

## 15. Visualization Suite

### 15.1 ε Sweep Curve

**Code reference:** `plot_eps_sweep()`

Plots $\text{RMS}_{\text{total}}(\varepsilon)$ vs. $\varepsilon$, showing:
- The minimum (red dashed line) = optimal $\varepsilon^*$
- Current `config.py` value (orange dashed line)

### 15.2 Friction Curves per Joint

**Code reference:** `plot_friction_curves()`

For each joint, a 2D scatter plot of $(\dot{q}_j, \tau_f^{\text{signal}})$ with overlaid curves:
- **Gray scatter**: raw friction signal data (subsampled to 30K points for rendering)
- **Green curve**: new synthesized fit: $c_j^{\text{new}} \tanh(\dot{q}/\varepsilon^{\text{new}}) + v_j^{\text{new}} \dot{q}$
- **Red dashed**: old `config.py` values
- **Purple dotted**: asymmetry estimate (piece-wise linear with bias)

### 15.3 Bias Magnitude Bar Chart

**Code reference:** `plot_bias_magnitude()`

Side-by-side bars of $c_j$ vs. $|b_j|$ from Method C, showing how much of the friction signal is actual friction vs. gravity error.

### 15.4 Per-Joint ε Bar Chart

**Code reference:** `plot_per_joint_eps()`

Bar chart of $\varepsilon_j$ from Method D, with reference lines for the current value, minimum bound, and median.

### 15.5 Per-Trajectory Spread Box Plots

**Code reference:** `plot_per_trajectory_spread()`

Box plots showing the distribution of $c_j^{(k)}$ and $v_j^{(k)}$ across $K$ trajectories. Tight boxes = stable parameters; wide boxes = trajectory-dependent estimates.

---

## 16. Design History & Lessons Learned

The docstring at the top of the file documents the evolution:

| Version | Approach | Problem |
|---------|----------|---------|
| **v1** | Unconstrained bounds | Parameters explode: $\varepsilon \to 0$, $v \to 1.75$ |
| **v2** | Physical bounds + bulk diversity | Clean results but slow (iterative fitting on all data) |
| **v3** | Velocity filtering (exclude $|\dot{q}| < \theta$) | Collinearity between $\tanh(\dot{q}/\varepsilon)$ and $\dot{q}$ → viscous hits cap |
| **v4 (current)** | Bulk + unfiltered + bounded + bias-aware | Best: analytical speed, gravity robustness, physical bounds |

**Key insights:**
1. **Bounds are essential**: Without physical constraints, the optimizer finds mathematically optimal but physically absurd parameters
2. **Velocity filtering hurts**: Removing low-velocity samples creates collinearity between $\tanh(\dot{q}/\varepsilon)$ and $\dot{q}$ (both approximately linear for large $\dot{q}$), making $c$ and $v$ unidentifiable
3. **Trajectory diversity is key**: A single trajectory visits limited configurations → $\Delta g$ is nearly constant → bias is large. Many trajectories → $\Delta g$ averages toward zero
4. **Bias absorption works**: The bias column in Method C provably decorrelates the constant gravity error from the friction regressors

---

## 17. Usage

### 17.1 Quick Check (Single File)

```bash
cd ~/Desktop/MTP_PINN/Torque_Analysis
python3 calibrate_friction.py
```

Uses the file specified in `config.py` → `C.LOG_JSON`.

### 17.2 Recommended (All Files)

```bash
python3 calibrate_friction.py --bulk
```

Loads all JSON files from `raw_samples/`. Runs all four methods + per-trajectory consistency.

### 17.3 Fast Mode (Sweep Only)

```bash
python3 calibrate_friction.py --bulk --method sweep
```

Skips the nonlinear fit (Method D). The sweep is analytical and instant.

### 17.4 Nonlinear Only

```bash
python3 calibrate_friction.py --bulk --method nonlinear
```

Runs only Method D (plus A and B which always run).

### 17.5 Execution Pipeline

```
main()
  ├── build_pinocchio_model()           # URDF → Pinocchio model
  ├── load_all_data()                   # JSON → (q̇, τ_f, q)
  ├── print_velocity_stats()            # Diagnostic
  ├── correlation_diagnostic()          # Diagnostic
  ├── regime_analysis()                 # Method A (always runs)
  ├── asymmetry_analysis()              # Method B (always runs)
  ├── [baseline RMS printout]
  ├── sweep_eps_bias()                  # Method C (if --method sweep|both)
  │   ├── plot_eps_sweep()
  │   └── plot_bias_magnitude()
  ├── fit_nonlinear_joint() × nq       # Method D (if --method nonlinear|both)
  │   └── plot_per_joint_eps()
  ├── per_trajectory_consistency()      # If --bulk and sweep ran
  │   └── plot_per_trajectory_spread()
  ├── synthesize_recommendation()       # Weighted fusion of A+B+C+D
  ├── [RMS validation]
  ├── plot_friction_curves()
  └── [final config.py printout]
```

---

## 18. Complete Function Reference

| Function | Lines | Input | Output | Method |
|----------|-------|-------|--------|--------|
| `load_friction_data()` | Data loading | JSON path, Pinocchio model | `(qd, tau_f, q, run_id)` | — |
| `load_all_data()` | Data loading | Model, bulk flag | Concatenated + per-file arrays | — |
| `print_velocity_stats()` | Diagnostic | `qd`, active count | Console output | — |
| `correlation_diagnostic()` | Diagnostic | `qd`, `tau_f`, `q`, active | Console output | — |
| `regime_analysis()` | **Method A** | `qd`, `tau_f`, active | `{j: {coulomb, viscous}}` | Median + OLS |
| `asymmetry_analysis()` | **Method B** | `qd`, `tau_f`, active, threshold | `{j: {coulomb, viscous, bias}}` | Split-half OLS |
| `fit_bias_aware()` | **Method C** (inner) | `qd_j`, `tau_f_j`, ε | `(b, c, v, rms)` | Analytical LS |
| `sweep_eps_bias()` | **Method C** (outer) | `qd`, `tau_f`, active | `(best_eps, params, rms, all)` | Grid search + LS |
| `fit_nonlinear_joint()` | **Method D** | `qd_j`, `tau_f_j`, initial guess | `(c, v, eps, rms)` | L-BFGS-B |
| `per_trajectory_consistency()` | Stability check | per-file list, active, ε | `{j: {c_mean, c_std, ...}}` | Per-file LS |
| `synthesize_recommendation()` | **Fusion** | All method outputs | `(final_c, final_v, final_eps)` | Weighted average |
| `plot_eps_sweep()` | Visualization | Sweep results | Figure | — |
| `plot_friction_curves()` | Visualization | Data + parameters | Figure | — |
| `plot_bias_magnitude()` | Visualization | Sweep params | Figure | — |
| `plot_per_joint_eps()` | Visualization | NL eps array | Figure | — |
| `plot_per_trajectory_spread()` | Visualization | Trajectory stats | Figure | — |
| `rms()` | Utility | Array | Scalar | $\sqrt{\text{mean}(x^2)}$ |

---

## Summary of Mathematical Methods

| Method | Model | Solver | Handles $\Delta g$? | Fits ε? |
|--------|-------|--------|---------------------|---------|
| **A: Regime** | $c \cdot \text{sign}(\dot{q}) + v \cdot \dot{q}$ | Median + OLS | ✗ | ✗ |
| **B: Asymmetry** | $a + v \cdot \dot{q}$ (split by sign) | OLS × 2 + algebra | **✓** (cancelled) | ✗ |
| **C: Bias-Aware Sweep** | $b + c \cdot \tanh(\dot{q}/\varepsilon) + v \cdot \dot{q}$ | Analytical LS (grid on ε) | **✓** (absorbed) | ✓ (global) |
| **D: Nonlinear** | $c \cdot \tanh(\dot{q}/\varepsilon_j) + v \cdot \dot{q}$ | L-BFGS-B | ✗ | ✓ (per-joint) |
| **Synthesis** | Weighted average of A–D | Arithmetic | Inherits from B, C | From D or C |