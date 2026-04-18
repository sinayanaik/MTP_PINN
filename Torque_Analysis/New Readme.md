# Torque Analysis — Comprehensive Technical Guide

> **What this pipeline does in one sentence:** Given a robot arm's recorded joint
> positions and servo readings, it computes three independent torque estimates per joint,
> fits calibrated mass/friction models, identifies what the physics model cannot explain,
> and generates statistical summaries across 124 experimental trajectories — all as
> structured training data for a Physics-Informed Neural Network (PINN).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Configuration Space & Encoder Conversion](#configuration-space--encoder-conversion)
4. [Kinematic Tree & URDF Model](#kinematic-tree--urdf-model)
5. [Equations of Motion](#equations-of-motion)
6. [Recursive Newton–Euler Algorithm (RNEA)](#recursive-newtoneuler-algorithm-rnea)
7. [Mass Calibration — The α Problem](#mass-calibration--the-α-problem)
8. [Measured Torque from Servo Registers](#measured-torque-from-servo-registers)
9. [Numerical Differentiation Pipeline](#numerical-differentiation-pipeline)
10. [Why the Physics Model Alone Fails](#why-the-physics-model-alone-fails)
11. [Friction Physics & Smooth Model](#friction-physics--smooth-model)
12. [Friction Calibration — Four Methods Explained](#friction-calibration--four-methods-explained)
13. [The Complete Model & What the PINN Must Learn](#the-complete-model--what-the-pinn-must-learn)
14. [Model Validation Metrics](#model-validation-metrics)
15. [Per-Joint Ratio Interpretation & Troubleshooting](#per-joint-ratio-interpretation--troubleshooting)
16. [Bulk Analysis & Statistical Summaries](#bulk-analysis--statistical-summaries)
17. [Error Histogram Diagnostics](#error-histogram-diagnostics)
18. [Output Layout](#output-layout)
19. [Running the Pipeline](#running-the-pipeline)
20. [Configuration Reference](#configuration-reference)
21. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Quick Start

```bash
cd ~/Desktop/MTP_PINN

# 1. Calibrate the mass model — bulk, servo-adjusted (run once after hardware assembly)
python3 -m Torque_Analysis.calibrate_mass

# 2. Calibrate friction parameters — bulk over all trajectories
python3 -m Torque_Analysis.calibrate_friction

# 3. Verify on a single file
python3 Torque_Analysis/main.py

# 4. Run full batch analysis → infer_torque/
python3 Torque_Analysis/bulk_analyze.py
```

Both calibration scripts always use all files in `raw_samples/` (bulk mode) and save
results automatically to `Torque_Analysis/calibration_params.json` — no flags needed
and no manual edits to `config.py` required.

---

## System Overview

### Hardware

| Component | Specification |
| --------- | ------------- |
| Robot | Kikobot — 6-DOF serial arm |
| Servos J1–J5 | Feetech STS3215 — 30 kgf·cm stall, 12 V |
| Joint J6 | Passive tool joint (no actuation) |
| Structure | PLA 3D-printed, ~70% infill (ρ ≈ 875 kg/m³) |
| Control rate | ~303 Hz |
| Encoder | 4096 ticks/revolution |

### Three Torque Signals

At every timestep for every joint, three independent estimates are computed and compared:

| Signal | Symbol | What it represents |
| ------ | ------ | ------------------ |
| **Measured** | τ\_load | What the servo actually exerted — from the load register |
| **Physics model** | τ\_RNEA | What rigid-body physics predicts the arm needed |
| **Friction model** | τ\_fric | Calibrated Coulomb + viscous friction estimate |

The **combined model** is:

$$\hat{\tau}_{\text{model}} = \tau_{\text{RNEA}} + \tau_{\text{fric}}$$

The gap between measured and modelled — the **residual** — is what the PINN must learn:

$$\tau_{\text{residual}} = \tau_{\text{load}} - \hat{\tau}_{\text{model}}$$

A small residual means the physics model is accurate. A large residual means there are
effects the physics model is missing — things like unmodelled masses, gear backlash,
cable forces, or structural compliance.

### Module Map

```text
Torque_Analysis/
├── config.py               ← single source of truth for all constants
├── data_loader.py          ← JSON log → NumPy arrays
├── utils.py                ← timestamp repair, Savitzky-Golay differentiation
├── torque.py               ← RNEA, friction, load-register conversion
├── diagnostics.py          ← 9 automated data-quality checks
├── plots.py                ← per-run plots (10 types)
├── plots_global.py         ← cross-run summary + histogram plots
├── plots_bulk.py           ← headless wrapper for batch rendering
├── main.py                 ← single-file entry point
├── bulk_analyze.py         ← batch processor → infer_torque/
├── calibrate_mass.py       ← fits global density scale α
└── calibrate_friction.py   ← four-method friction calibration
```

### Data Flow

```text
JSON log  →  fix_timestamps  →  ticks_to_radians  →  q [rad]
                                                       │
                             Savitzky-Golay deriv  →  q̇, q̈
                                                       │
                         ┌─────────────────────────────┤
                         │                             │
                    torque_from_load              build_pinocchio_model
                    τ_load [N·m]                      + RNEA
                         │                        τ_RNEA [N·m]
                         │                             │
                         │                    torque_friction
                         │                        τ_fric [N·m]
                         │                             │
                         └──── τ_residual = τ_load − (τ_RNEA + τ_fric)
```

---

## Configuration Space & Encoder Conversion

### What is Configuration Space?

The robot's "pose" at any instant is fully described by the 6 joint angles:

$$\mathbf{q} = [q_1,\; q_2,\; q_3,\; q_4,\; q_5,\; q_6]^T \in \mathbb{R}^6 \quad \text{(radians)}$$

Joints 1–5 are motor-driven; joint 6 is a passive tool holder that rotates freely.

### From Servo Ticks to Radians

Servos speak in **encoder ticks** — an integer counter with 4096 ticks per revolution.
The conversion to radians in the URDF coordinate frame is:

$$q_j = d_j \cdot \left(p_j^{\text{ticks}} - p_j^{\text{center}}\right) \cdot k_{\text{t2r}}$$

| Symbol | Meaning | Where it comes from |
| ------ | ------- | ------------------- |
| $d_j \in \{-1, +1\}$ | Which direction is "positive" in URDF | `joint_map[j]["direction"]` in JSON |
| $p_j^{\text{center}}$ | Tick count at URDF zero angle | `joint_map[j]["ticks_center"]` in JSON |
| $k_{\text{t2r}} = 2\pi/4096$ | Radians per tick | `M["ticks_to_rad"]` |

The direction sign $d_j$ is critical — a servo physically rotating clockwise may correspond
to a positive or negative URDF joint angle depending on how the link was assembled.
This was determined experimentally by `calibrate_mass.py`.

> **Code**: `utils.py → ticks_to_radians()`

---

## Kinematic Tree & URDF Model

### The Kinematic Chain

The robot is a **serial chain** — each link connects to exactly one parent and one child:

$$\text{world} \xrightarrow{\text{fixed}} \text{base} \xrightarrow{q_1} \text{link}_1 \xrightarrow{q_2} \text{link}_2 \xrightarrow{q_3} \text{link}_3 \xrightarrow{q_4} \text{link}_4 \xrightarrow{q_5} \text{link}_5 \xrightarrow{q_6} \text{tool}$$

Each arrow is a **revolute joint** — a single-axis rotation. The URDF file defines,
for each link: its shape/geometry, its mass, its centre of mass position, and its inertia tensor.

### Homogeneous Transforms

The pose of link frame $i$ relative to link frame $i-1$ is:

$${}^{i-1}\mathbf{T}_i(q_i) = {}^{i-1}\mathbf{T}_i^{\text{fixed}} \cdot \begin{bmatrix} \cos q_i & -\sin q_i & 0 & 0 \\ \sin q_i & \cos q_i & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

The static part ${}^{i-1}\mathbf{T}_i^{\text{fixed}}$ encodes the geometric offset between
consecutive joints (comes from the URDF). The rotation matrix (bottom-right 3×3) encodes
the joint rotation. World-frame pose of link $i$ is the product of all transforms along the chain:

$${}^{0}\mathbf{T}_i(\mathbf{q}) = {}^{0}\mathbf{T}_1(q_1) \cdot {}^{1}\mathbf{T}_2(q_2) \cdots {}^{i-1}\mathbf{T}_i(q_i)$$

### Spatial Inertia (6×6 Matrix)

Each link $i$ has three inertial properties: mass $m_i$, centre of mass position
$\mathbf{c}_i$, and a 3×3 rotational inertia tensor $\mathbf{I}_i$. Pinocchio combines
these into a single **6×6 spatial inertia matrix**:

$$\mathcal{I}_i = \begin{bmatrix} \mathbf{I}_i + m_i [\mathbf{c}_i]_\times^T [\mathbf{c}_i]_\times & m_i [\mathbf{c}_i]_\times \\ m_i [\mathbf{c}_i]_\times^T & m_i \mathbf{I}_3 \end{bmatrix}$$

where $[\mathbf{c}]_\times$ is the skew-symmetric (cross-product) matrix of $\mathbf{c}$.
This compact form lets Pinocchio treat linear and angular momentum together — which is
why RNEA is so efficient.

> **Code**: `torque.py → build_pinocchio_model()` calls `pin.buildModelFromXML(urdf_xml)`

---

## Equations of Motion

The torque required to move a rigid-body arm follows the **Lagrangian equation of motion**:

$$\boxed{\mathbf{M}(\mathbf{q})\,\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\,\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}_{\text{applied}}}$$

Think of this as Newton's second law ($F = ma$) generalised to a rotating, multi-link chain:

| Term | Analogy | Mathematical definition |
| ---- | ------- | ----------------------- |
| $\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}$ | "mass × acceleration" | Inertia × joint accelerations |
| $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}$ | Gyroscopic effects | Velocity-dependent coupling forces |
| $\mathbf{g}(\mathbf{q})$ | Weight of the arm | Torque needed to hold arm against gravity |
| $\boldsymbol{\tau}_{\text{applied}}$ | Net input | Motor torque (what RNEA solves for) |

### Inertia Matrix $\mathbf{M}(\mathbf{q})$

$\mathbf{M}$ is a symmetric positive-definite $n \times n$ matrix. It is **configuration-dependent**
because the arm's effective inertia about each joint changes as the arm extends or contracts.

$$M_{ij}(\mathbf{q}) = \sum_{k=\max(i,j)}^{n} \left[ m_k \, \mathbf{J}_{v_k,i}^T \mathbf{J}_{v_k,j} + \mathbf{J}_{\omega_k,i}^T \, {}^{0}\mathbf{R}_k \, \mathbf{I}_k \, {}^{0}\mathbf{R}_k^T \, \mathbf{J}_{\omega_k,j} \right]$$

$M_{ij}$ represents how much torque at joint $i$ is needed per unit acceleration at joint $j$,
summing over all links distal to both joints.

### Coriolis & Centrifugal — Christoffel Symbols

The Coriolis/centrifugal term arises because $\mathbf{M}(\mathbf{q})$ is
configuration-dependent — its entries change as the arm moves, creating
velocity-dependent coupling:

$$\left[\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\,\dot{\mathbf{q}}\right]_i = \sum_{j,k} c_{ijk}\, \dot{q}_j\, \dot{q}_k, \qquad c_{ijk} = \frac{1}{2}\left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right)$$

These $c_{ijk}$ are the **Christoffel symbols** of the first kind — the same objects that
appear in geodesics on curved manifolds. When two joints move simultaneously ($j \neq k$),
Coriolis forces couple them. When a single joint spins ($j = k$), centrifugal forces arise.

### Gravity $\mathbf{g}(\mathbf{q})$

$$g_i(\mathbf{q}) = -\sum_{k=i}^{n} m_k \, \mathbf{g}_0^T \, \frac{\partial \mathbf{p}_{c_k}(\mathbf{q})}{\partial q_i}$$

$g_i$ is the torque joint $i$ must exert just to hold the arm stationary. It depends only
on configuration (not on velocity or acceleration), making it the most reliable and
strongest signal for mass calibration.

---

## Recursive Newton–Euler Algorithm (RNEA)

### Why Not Just Compute M, C, g Separately?

Constructing the full matrices $\mathbf{M}$, $\mathbf{C}$, $\mathbf{g}$ explicitly costs
$O(n^3)$ per timestep. For a 6-DOF arm at 303 Hz over 124 trajectories (~470K timesteps),
that is prohibitive. RNEA computes the identical result in **$O(n)$ time** using two
sweeps over the kinematic chain.

### Forward Sweep: Base → Tip (Propagate Motion)

The forward pass pushes kinematic quantities (velocities, accelerations) outward from the
fixed base toward the free tip. **Base initialisation:**

$$\boldsymbol{\omega}_0 = \mathbf{0}, \quad \dot{\boldsymbol{\omega}}_0 = \mathbf{0}, \quad \mathbf{a}_0 = -\mathbf{g}_0 = [0, 0, +9.81]^T$$

> **Gravity trick**: instead of adding gravity forces to every link individually, we start
> the base with a fictitious upward acceleration. This single initialisation automatically
> produces correct gravity torques at every joint — a key algorithmic elegance.

**For each joint $i = 1 \ldots n$:**

$$\boldsymbol{\omega}_i = {}^{i}\mathbf{R}_{i-1} \, \boldsymbol{\omega}_{i-1} + \dot{q}_i \, \hat{\mathbf{z}}_i \qquad \text{(angular velocity)}$$

$$\dot{\boldsymbol{\omega}}_i = {}^{i}\mathbf{R}_{i-1} \, \dot{\boldsymbol{\omega}}_{i-1} + \ddot{q}_i \, \hat{\mathbf{z}}_i + \left({}^{i}\mathbf{R}_{i-1} \, \boldsymbol{\omega}_{i-1}\right) \times \dot{q}_i \, \hat{\mathbf{z}}_i \qquad \text{(angular acceleration)}$$

$$\mathbf{a}_i = {}^{i}\mathbf{R}_{i-1} \left( \mathbf{a}_{i-1} + \dot{\boldsymbol{\omega}}_{i-1} \times \mathbf{r}_{i-1,i} + \boldsymbol{\omega}_{i-1} \times (\boldsymbol{\omega}_{i-1} \times \mathbf{r}_{i-1,i}) \right) \qquad \text{(linear accel of joint origin)}$$

$$\mathbf{a}_{c_i} = \mathbf{a}_i + \dot{\boldsymbol{\omega}}_i \times \mathbf{c}_i + \boldsymbol{\omega}_i \times (\boldsymbol{\omega}_i \times \mathbf{c}_i) \qquad \text{(linear accel of CoM)}$$

| Symbol | Meaning |
| ------ | ------- |
| ${}^{i}\mathbf{R}_{i-1}$ | Rotation from frame $i-1$ to frame $i$ |
| $\hat{\mathbf{z}}_i$ | Joint axis unit vector in frame $i$ |
| $\mathbf{r}_{i-1,i}$ | Vector from origin of frame $i-1$ to frame $i$ |
| $\mathbf{c}_i$ | CoM position of link $i$ in frame $i$ |

### Backward Sweep: Tip → Base (Propagate Forces)

The backward pass pushes force and moment quantities inward from the free tip toward the
base. At each link, Newton's and Euler's laws give the required force and moment, then
child-link contributions are propagated inward. **Tip initialisation:** $\mathbf{f}_{n+1} = \mathbf{0}$, $\boldsymbol{\mu}_{n+1} = \mathbf{0}$.

**For each link $i = n \ldots 1$:**

$$\mathbf{F}_i = m_i \, \mathbf{a}_{c_i} \qquad \text{(Newton's 2nd law for CoM)}$$

$$\mathbf{N}_i = \mathbf{I}_i \, \dot{\boldsymbol{\omega}}_i + \boldsymbol{\omega}_i \times (\mathbf{I}_i \, \boldsymbol{\omega}_i) \qquad \text{(Euler's equation; 2nd term = gyroscopic)}$$

$$\mathbf{f}_i = {}^{i}\mathbf{R}_{i+1} \, \mathbf{f}_{i+1} + \mathbf{F}_i \qquad \text{(total force through joint } i\text{)}$$

$$\boldsymbol{\mu}_i = {}^{i}\mathbf{R}_{i+1} \left( \boldsymbol{\mu}_{i+1} + \mathbf{r}_{i,i+1} \times \mathbf{f}_{i+1} \right) + \mathbf{c}_i \times \mathbf{F}_i + \mathbf{N}_i$$

$$\boxed{\tau_i = \boldsymbol{\mu}_i^T \, \hat{\mathbf{z}}_i} \qquad \text{(joint torque = moment projected onto axis)}$$

The result is exactly:
$\boldsymbol{\tau}_{\text{RNEA}} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$,
computed without ever forming those matrices.

### Special Cases

**Gravity-only** (used for mass calibration — set velocities and accelerations to zero):

$$\boldsymbol{\tau}_{\text{gravity}} = \text{RNEA}(\mathbf{q}, \mathbf{0}, \mathbf{0}) = \mathbf{g}(\mathbf{q})$$

> **Code**: `torque.py → torque_from_urdf()` (full), `torque_gravity_only()` (gravity only)

---

## Mass Calibration — The α Problem

### Why the URDF Masses are Wrong

The URDF was exported from SolidWorks assuming solid **steel** (ρ = 7800 kg/m³).
The robot is actually 3D-printed **PLA at ~70% infill** (ρ ≈ 875 kg/m³) — about 9× less dense.

If you run RNEA with the uncorrected URDF, it predicts torques 9× too large.

### The Fix: A Single Scale Factor α

Because all links are PLA with roughly the same infill, a **single scalar** α corrects every
link simultaneously:

$$m_i^{\text{actual}} = \alpha \cdot m_i^{\text{URDF}}, \qquad \mathbf{I}_i^{\text{actual}} = \alpha \cdot \mathbf{I}_i^{\text{URDF}}, \qquad \alpha = \frac{875}{7800} \approx 0.112$$

Because RNEA is **linear** in the inertial parameters, this means:

$$\boldsymbol{\tau}_{\text{RNEA}}^{(\alpha)} = \alpha \cdot \boldsymbol{\tau}_{\text{RNEA}}^{(\alpha=1)}$$

We only need to fit one number to fix the entire model.

### Fitting α via Gravity Matching

At any pose, the servo load register measures what the motor actually exerts, which should
match the scaled gravity torque for static/quasi-static motions:

$$\tau_{\text{load},j}(t) \approx \alpha \cdot \tau_{\text{gravity},j}^{(\alpha=1)}(t)$$

This is a linear regression through the origin. The **weighted least-squares** solution is:

$$\alpha^* = \arg\min_\alpha \sum_{t,j \in \{1,2\}} \left(\tau_{\text{load},j}(t) - \alpha \cdot \tau_{\text{gravity},j}^{(1)}(t)\right)^2$$

Taking the derivative and setting to zero gives the closed-form solution:

$$\boxed{\alpha^* = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}}} \qquad \mathbf{a} = \text{vec}(\tau_{\text{grav}}^{(1)}), \quad \mathbf{b} = \text{vec}(\tau_{\text{load}})$$

Only joints 1 and 2 (shoulder and elbow) are used because they have the strongest gravity
signal. Joint 0 (yaw) sees near-zero gravity; joints 3–4 have servo bodies whose metal mass
violates the uniform-density assumption.

**Calibrated result:**

$$\alpha^* \approx 0.1119 \implies \rho_{\text{implied}} = 0.1119 \times 7800 \approx 873 \text{ kg/m}^3 \quad \checkmark \text{ (matches PLA 70\% infill)}$$

> **Code**: `calibrate_mass.py → fit_scale()`
> **Update**: set `MASS_SCALE = 0.111893` in `config.py`

---

## Measured Torque from Servo Registers

### The Load Register

Each servo has an internal register that reports how hard it is working as a percentage
of its maximum (stall) torque, in 0.1% units. Converting to N·m:

$$\tau_j^{\text{servo}}(t) = \underbrace{\frac{\ell_j(t) \times 0.1}{100}}_{\substack{\text{fraction of}\\\text{stall torque}}} \times \underbrace{\frac{V_j(t)}{V_{\text{nom}}}}_{\substack{\text{voltage}\\\text{correction}}} \times \underbrace{\tau_{\text{stall}}(j)}_{\substack{\text{per-joint}\\\text{stall torque}}} \times \underbrace{k_{\text{conv}}}_{0.09807 \text{ N·m/kgf·cm}}$$

**Voltage correction** ($V/V_{\text{nom}}$) matters because servo stall torque scales
linearly with supply voltage — a battery at 11.5 V delivers ~4% less torque than at 12 V.

**Per-joint stall torques** (`STALL_TORQUE_PER_JOINT` in `config.py`) — if distal joints
use smaller servo models (e.g., STS3032 at 14.8 kgf·cm vs STS3215 at 30 kgf·cm), they
must be set independently, otherwise the load torque will be overestimated by ~2×.

### Frame Correction

The servo's internal coordinate system may be opposite to the URDF's joint direction.
The corrected torque in the URDF frame is:

$$\boxed{\tau_{\text{load},j}^{\text{URDF}} = -d_j \cdot \tau_j^{\text{servo}}}$$

The $-d_j$ convention was selected by `calibrate_mass.py` after testing all three
possible sign mappings (raw, $+d_j$, $-d_j$) and choosing the one that gives
positive, consistent α values across calibration joints.

> **Code**: `torque.py → torque_from_load_raw()` + `torque_from_load()`

---

## Numerical Differentiation Pipeline

### The Problem

The servo logs record **position** $\mathbf{q}(t)$ but RNEA requires **velocity**
$\dot{\mathbf{q}}(t)$ and **acceleration** $\ddot{\mathbf{q}}(t)$. These must be
computed numerically — a noisy process that requires careful handling.

### Step 1: Timestamp Repair

Control-loop logs occasionally contain **duplicate or retrograde timestamps** — two
consecutive samples with the same (or decreasing) timestamp. If not fixed, division
by zero occurs when differentiating.

**Two-stage repair:**

1. Find all "bad" indices where $\Delta t \le 0$
2. Replace them with linearly interpolated values from surrounding good timestamps:
   $t_{\text{fixed}} = \texttt{np.interp}(\text{all\_idx},\;\text{good\_idx},\;t[\text{good\_idx}])$
3. Final safety pass: add 1 ns to any remaining duplicate

```python
_EPS_T = 1e-9  # 1 nanosecond
for k in range(1, N):
    if t_fixed[k] <= t_fixed[k-1]:
        t_fixed[k] = t_fixed[k-1] + _EPS_T
```

The 1 ns gap is completely invisible to the physics (forces don't change in 1 ns)
but prevents divide-by-zero in `np.gradient`.

> **Code**: `utils.py → fix_timestamps()`

### Step 2: Savitzky–Golay Differentiation (Default)

Classical approach: differentiate with `np.gradient`, then smooth with a moving average.
Problem: `np.gradient` divides by per-sample spacing, so any tiny residual timestamp
jitter can cause spikes.

**New approach** (`DIFF_METHOD = "savgol"`): the Savitzky–Golay filter differentiates
and smooths **simultaneously** by fitting a polynomial over a sliding window, then
evaluating its derivative analytically.

```text
               ┌─ SG window (w=11 samples) ─┐
  …  q(k-5)  q(k-4)  …  q(k)  …  q(k+4)  q(k+5)  …
               └─── fit cubic polynomial ───┘
                           │
                   evaluate p'(t_k)
                           │
                        q̇(t_k)    ← smooth + differentiated in one pass
```

The critical difference from `np.gradient`:

$$\dot{\mathbf{q}} = \texttt{savgol\_filter}\!\left(\mathbf{q},\;w=11,\;p=3,\;\text{deriv}=1,\;\delta=\overline{\Delta t}\right)$$

Using $\overline{\Delta t}$ (mean timestep) instead of per-sample spacing means a single
corrupt timestamp cannot spike the derivative. This eliminates the divide-by-zero errors
that caused the `numpy RuntimeWarning: invalid value encountered` messages.

**Why SG preserves peaks better than diff + MA:** Moving average blurs the signal before
differentiation. SG fits a polynomial that can follow the local shape before differentiating
— it achieves similar noise suppression with less distortion of the true dynamics.

**Fallback** (`DIFF_METHOD = "gradient"`): `np.gradient` + uniform moving average.
Available for comparison; more sensitive to timestamp jitter.

### Step 3: Safety Clipping

Both methods clip extreme values before returning:

| Quantity | Clip bound | Physical justification |
| -------- | ---------- | ---------------------- |
| $\dot{q}_j$ | ±100 rad/s | Servo no-load speed is ~5 rad/s; 100 is a generous safety margin |
| $\ddot{q}_j$ | ±1000 rad/s² | Any differentiation spike exceeding this is a numerical artefact |

NaN and Inf are replaced with 0 by `np.nan_to_num`.

> **Code**: `utils.py → numerical_velocity()`, `numerical_acceleration()`
> **Config**: `DIFF_METHOD`, `SAVGOL_POLYORDER`, `SMOOTH_WINDOW` in `config.py`

---

## Why the Physics Model Alone Fails

Running RNEA without friction gives:

$$\hat{\tau}_{\text{simple}} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

This consistently under-predicts $\tau_{\text{load}}$. Four reasons:

### Missing Friction

Real servo gearboxes are NOT frictionless. The motor must overcome gear friction in addition
to the rigid-body dynamics:

$$\tau_{\text{motor}} = \underbrace{\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g}}_{\text{RNEA (modeled)}} + \underbrace{\tau_{\text{friction}}}_{\text{missing}}$$

The load register measures the left side; RNEA only computes the first three terms on the right.

### Non-Uniform Mass Model

The global α assumes all links have the same PLA/steel density ratio. But distal joints have
metal servo bodies that weigh much more than PLA:

$$\alpha_{\text{effective},j} = \frac{m_{\text{PLA},j} \cdot \alpha_{\text{PLA}} + m_{\text{servo},j} \cdot 1.0}{m_{\text{URDF},j}}$$

For wrist joints (J3–J4), the servo body often dominates the link mass, so
$\alpha_{\text{eff},j} \gg \alpha_{\text{PLA}}$ — the model underestimates inertia there.

### Differentiation Noise Amplified by Inertia

The inertia matrix $\mathbf{M}$ amplifies any error in the numerically computed $\ddot{\mathbf{q}}$:

$$\mathbf{M} \cdot (\ddot{\mathbf{q}}_{\text{true}} + \boldsymbol{\epsilon}) = \mathbf{M}\ddot{\mathbf{q}}_{\text{true}} + \underbrace{\mathbf{M}\boldsymbol{\epsilon}}_{\text{amplified noise}}$$

The shoulder joint (J2) has a large diagonal $M_{22}$ because it carries the weight of the
entire distal chain — meaning small acceleration errors create large torque errors.

### Unmodeled Physical Effects

| Effect | When it appears | Magnitude |
| ------ | --------------- | --------- |
| Stiction (break-away torque) | Direction reversals | Transient spike |
| Gear backlash | Near zero velocity | Dead-zone artefact |
| Cable routing forces | Configuration-dependent | Bias |
| Link flexibility | Fast motions | High-frequency oscillation |

---

## Friction Physics & Smooth Model

### The Intuition

Imagine pushing a heavy drawer:

- No matter how slowly you push it, you need to overcome a minimum force before it moves
  → this is **Coulomb (dry) friction**
- The faster you push, the more resistance you feel from the sliding surfaces
  → this is **viscous friction**

In a servo gearbox, the same two effects appear at every joint:

$$\tau_{f,j}^{\text{ideal}} = \underbrace{c_j \cdot \text{sign}(\dot{q}_j)}_{\text{Coulomb: constant, opposes motion}} + \underbrace{v_j \cdot \dot{q}_j}_{\text{viscous: grows with speed}}$$

### The Problem with sign(x)

The $\text{sign}$ function jumps discontinuously at zero velocity:

```text
   τ_friction
      ↑
   +c ─ ─ ─ ─ ─ •─────────────
      |          •
  ────┼──────────•──────────── q̇
      |          •
   -c ─ ─ ─ ─ ─ •─────────────
                 ↑
           discontinuity at q̇=0
```

This discontinuity causes three practical problems:

1. **Chattering** — simulations oscillate wildly near zero velocity
2. **No gradient** — the PINN cannot backpropagate through an undefined derivative at $\dot{q}=0$
3. **Noise sensitivity** — tiny velocity measurement noise causes sign flips → torque spikes

### The Smooth Fix: tanh

Replace $\text{sign}(\dot{q})$ with $\tanh(\dot{q}/\varepsilon)$:

```text
   τ_friction
      ↑
   +c ─ ─ ─ ─ ─ ────────────
      |       ╱
      |      ╱  ← smooth transition
      |     ╱     width ≈ ε
  ────┼────╳────────────────── q̇
      |     ╲
      |      ╲
   -c ─ ─ ─ ─ ─ ────────────
                 ↑
        smooth, gradient everywhere
```

The calibrated transition width $\varepsilon = 0.0628$ rad/s means the jump from $-c$ to $+c$
happens within ±0.2 rad/s (about ±2°/s) — well below typical joint speeds. From the outside,
this is indistinguishable from the ideal sign function.

$$\boxed{\tau_{f,j}(\dot{q}_j) = c_j \cdot \tanh\!\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j \cdot \dot{q}_j}$$

Key properties of tanh:

- $\tanh(x/\varepsilon) \to +1$ when $x \gg \varepsilon$ (fast positive motion — full Coulomb)
- $\tanh(x/\varepsilon) \approx x/\varepsilon$ when $|x| \ll \varepsilon$ (near-zero velocity — linear)
- $\tanh(x/\varepsilon) \to -1$ when $x \ll -\varepsilon$ (fast negative motion — full Coulomb)
- Derivative $= \frac{1}{\varepsilon}\text{sech}^2(x/\varepsilon)$ — smooth, bounded, everywhere defined

> **Code**: `torque.py → _smooth_sign()` + `torque_friction()`
> **Calibrated values** (all joints):

```python
COULOMB_NM   = [0.2472, 0.2873, 0.2403, 0.1806, 0.2151, 0.0]  # c_j [N·m]
VISCOUS_NM   = [0.0,    0.3,    0.0,    0.051,  0.0042, 0.0]   # v_j [N·m·s/rad]
FRICTION_EPS = 0.0628                                            # ε [rad/s]
```

---

## Friction Calibration — Four Methods Explained

### The Core Challenge

To calibrate friction, we need to measure it. But we cannot measure friction directly —
we can only measure the **total torque** the motor exerts. After subtracting the
RNEA-predicted torque, what remains is:

$$\tau_f^{\text{signal}}(t) = \underbrace{\tau_{\text{load}}(t)}_{\text{measured}} - \underbrace{\tau_{\text{RNEA}}(t)}_{\text{modeled}} = \underbrace{\tau_{\text{friction, true}}(t)}_{\text{what we want}} + \underbrace{\Delta g_j(\mathbf{q})}_{\text{gravity model error}} + \underbrace{\eta(t)}_{\text{noise}}$$

**The contamination problem:** Because α was calibrated on joints 1–2 only, the RNEA
gravity torque has small errors on other joints. These gravity errors leak into the friction
signal. Every calibration method must handle this contamination.

The four methods attack this problem from different angles, each with different trade-offs.
Their results are then combined into a weighted consensus.

---

### Method A: Regime Analysis — "Catch It When It's Barely Moving"

**The idea in plain language:**

When a joint is moving extremely slowly (near zero velocity), the viscous term $v \cdot \dot{q}$
is negligible (nearly zero × something small). At that moment, the friction signal is
approximately just the Coulomb constant:

$$|\tau_f^{\text{signal}}| \approx c_j \quad \text{when } |\dot{q}_j| \text{ is tiny}$$

Similarly, at high speeds, the Coulomb term is a known ±constant, so we can subtract it
and see only the viscous part remaining:

$$\tau_f^{\text{signal}} - c_j \cdot \text{sign}(\dot{q}_j) \approx v_j \cdot \dot{q}_j \quad \text{at high speed}$$

**Step by step:**

1. **Find slow-moving samples:** select all timesteps where $0.01 < |\dot{q}_j| < 0.05$ rad/s
2. **Estimate Coulomb c_j:** take the median absolute value of the friction signal in that regime
   $$\hat{c}_j = \text{median}\!\left(|\tau_f^{\text{signal}}| \;\Big|\; |\dot{q}_j| \in (0.01,\; 0.05)\right)$$
3. **Find fast-moving samples:** select all timesteps where $|\dot{q}_j| > 0.10$ rad/s
4. **Estimate viscous v_j:** subtract the Coulomb part and do linear regression on the remainder
   $$\hat{v}_j = \frac{\sum_k \dot{q}_{j,k} \cdot (\tau_{f,j,k}^{\text{signal}} - \hat{c}_j \cdot \text{sign}(\dot{q}_{j,k}))}{\sum_k \dot{q}_{j,k}^2}$$

**Why it works:** Different velocity regimes isolate different friction components.

**Why it is unreliable:** Gravity model error $\Delta g_j(\mathbf{q})$ is always present.
At low speeds, the arm is often in a specific pose range, so the gravity error is
not zero-mean and directly biases $\hat{c}_j$.

> Reliability: **Low** — useful as a sanity check but noisy.

---

### Method B: Asymmetry Analysis — "The Symmetry Trick"

**The key insight (explained simply):**

Friction always **opposes** motion — it acts in the opposite direction to whichever way the
joint is moving. So if you move a joint forward and measure friction, you get $+c_j$; if you
move it backward, you get $-c_j$. **Friction is antisymmetric in velocity.**

Gravity error, however, depends only on the arm's pose — not on which direction the joint
is moving. A joint bending forward with the arm extended horizontal has the same gravity
torque whether it is moving slowly clockwise or counterclockwise.
**Gravity error is symmetric (sign-independent).**

This difference in symmetry is the key to separating them:

```text
                 τ_f^signal
                      ↑
                      |
             a⁺ ─────•─────────────  ← positive velocity half (line fit)
                    ╱ intercept = +c + b
                   ╱
              b ──╳─────────────────  ← zero: the "true" gravity bias level
                   ╲
             a⁻ ─────•───────────── ← negative velocity half (line fit)
                      intercept = −c + b
                      |
  ───────────────────────────────────→ q̇
                negative  | positive
```

**Step by step:**

1. Split all data into two groups: samples with $\dot{q}_j > 0$ and samples with $\dot{q}_j < 0$
2. **Fit a straight line** to each group: $\tau_f^{\text{signal}} = a^{\pm} + v^{\pm} \cdot \dot{q}_j$
3. The two intercepts $a^+$ and $a^-$ contain friction and gravity bias mixed together:
   - Moving forward: $a^+ = +c_j + b_j$ (friction adds, gravity bias adds)
   - Moving backward: $a^- = -c_j + b_j$ (friction flips sign, gravity bias stays same)
4. **Separate them by algebra:**

$$\hat{c}_j = \frac{a^+ - a^-}{2} \qquad \text{(half the difference → the antisymmetric part = pure friction)}$$

$$\hat{b}_j = \frac{a^+ + a^-}{2} \qquad \text{(half the sum → the symmetric part = gravity error)}$$

$$\hat{v}_j = \frac{v^+ + v^-}{2} \qquad \text{(average slope from both halves = viscous friction)}$$

**Why this works over 124 trajectories:** Each single trajectory may be biased toward
certain poses, giving a specific $b_j$. But across 124 trajectories covering diverse poses
and directions, the gravity error $b_j$ averages toward zero. This makes the Coulomb
estimate very robust.

**Diagnostic metric:** The bias-to-Coulomb ratio $R_j = |b_j|/c_j$ tells you how much
gravity error is contaminating the signal:

| $R_j$ | What it means |
| ----- | ------------- |
| $< 0.3$ | Clean signal — the friction signal dominates |
| $0.3 - 1.0$ | Noticeable gravity bias — result is usable but less precise |
| $> 1.0$ | Gravity-dominated — the friction estimate is unreliable |

> Reliability: **High** — most robust to gravity contamination. Given weight 3.0.
> **Code**: `calibrate_friction.py → asymmetry_analysis()`

---

### Method C: ε Sweep with Analytical LS — "Try All Speeds and Find the Best Fit"

**The idea in plain language:**

The transition width $\varepsilon$ controls how "sharp" the friction switch is at zero
velocity. We don't know the right value of $\varepsilon$ — it depends on the servo gearbox
characteristics. But for any **fixed** value of $\varepsilon$, finding the best $(c_j, v_j)$
is just a linear regression problem that has an exact closed-form solution.

So the strategy is: try 100 different values of $\varepsilon$, solve the linear regression
instantly for each one, and pick the $\varepsilon$ that gives the smallest prediction error
across all joints.

**The linear regression setup (for a fixed ε):**

At each timestep $t$, the friction signal is:
$$\tau_f^{\text{signal}}(t) = \underbrace{b_j}_{\text{gravity bias}} + \underbrace{c_j \cdot \tanh(\dot{q}_j(t)/\varepsilon)}_{\text{Coulomb}} + \underbrace{v_j \cdot \dot{q}_j(t)}_{\text{viscous}}$$

Stack all $N$ timesteps into matrix form:

$$\underbrace{\begin{bmatrix} 1 & \tanh(\dot{q}_{j,1}/\varepsilon) & \dot{q}_{j,1} \\ 1 & \tanh(\dot{q}_{j,2}/\varepsilon) & \dot{q}_{j,2} \\ \vdots & \vdots & \vdots \\ 1 & \tanh(\dot{q}_{j,N}/\varepsilon) & \dot{q}_{j,N} \end{bmatrix}}_{\boldsymbol{\Phi}_j(\varepsilon) \;\in\; \mathbb{R}^{N \times 3}} \cdot \underbrace{\begin{bmatrix} b_j \\ c_j \\ v_j \end{bmatrix}}_{\text{unknowns}} = \underbrace{\begin{bmatrix} \tau_{f,j,1}^{\text{signal}} \\ \vdots \\ \tau_{f,j,N}^{\text{signal}} \end{bmatrix}}_{\text{measured}}$$

The matrix $\boldsymbol{\Phi}_j$ has the tanh column already computed for the given $\varepsilon$.
The ordinary least-squares solution is:

$$\begin{bmatrix} b_j \\ c_j \\ v_j \end{bmatrix}^* = \underbrace{\left(\boldsymbol{\Phi}_j^T \boldsymbol{\Phi}_j\right)^{-1} \boldsymbol{\Phi}_j^T}_{\text{pseudo-inverse}} \cdot \boldsymbol{\tau}_f^{\text{signal}}$$

Although $N \approx 470{,}000$ samples, $\boldsymbol{\Phi}_j^T \boldsymbol{\Phi}_j$ is only **3×3**,
so this inversion is essentially instant. For all 100 values of $\varepsilon$ and 5 joints,
this is 500 small matrix inversions — negligible compute time.

**Finding the best ε:**

$$\varepsilon^* = \arg\min_{\varepsilon \in [0.02,\; 0.50]} \underbrace{\sqrt{\frac{1}{5} \sum_{j=1}^{5} \text{RMS}_j^2(\varepsilon)}}_{\text{joint-averaged prediction error}}$$

After finding $\varepsilon^*$, the final $c_j$ and $v_j$ are the ones from the LS solution at $\varepsilon^*$,
clamped to physical bounds ($c_j \in [0, 0.50]$ N·m, $v_j \in [0, 0.30]$ N·m·s/rad).

**Why include the bias column?** Without it, the gravity error $\Delta g$ leaks into $c_j$:

$$\hat{c}_j^{\text{no bias}} = c_j^{\text{true}} + \frac{\text{cov}(\tanh(\dot{q}/\varepsilon),\;\Delta g)}{\text{var}(\tanh(\dot{q}/\varepsilon))} \neq c_j^{\text{true}}$$

The explicit bias column $\mathbf{1}$ absorbs the mean gravity error, keeping it out of
the friction estimates.

> Reliability: **High** (Coulomb), **Highest** (viscous — full dataset, analytical).
> Given weight 2.0 for Coulomb, 3.0 for viscous.
> **Code**: `calibrate_friction.py → sweep_eps_bias()` + `fit_bias_aware()`

---

### Method D: Per-Joint Nonlinear Optimisation — "Let Each Joint Have Its Own ε"

**The idea in plain language:**

Methods A, B, and C all use a single shared $\varepsilon$ for all joints. But different
joints have different gearboxes (the shoulder joint has a large gear reduction; the wrist
has a smaller, lighter gearbox). Their friction curves may "sharpen" at different velocities.

Method D relaxes this: let every joint independently choose its own best $\varepsilon_j$
by using a numerical optimiser.

**The optimisation:**

For each joint $j$ independently:

$$\min_{c_j,\; v_j,\; \varepsilon_j} \frac{1}{N}\sum_{t=1}^{N} \left(\tau_{f,j}^{\text{signal}}(t) - c_j \tanh\!\left(\frac{\dot{q}_j(t)}{\varepsilon_j}\right) - v_j \dot{q}_j(t)\right)^2$$

subject to physical bounds: $c_j \in [0, 0.50]$, $v_j \in [0, 0.30]$, $\varepsilon_j \in [0.02, 0.50]$

Unlike Method C, $\varepsilon_j$ appears **nonlinearly** inside the tanh, so there is no
closed-form solution. The L-BFGS-B algorithm (a gradient-based optimiser with box constraints)
is used, **warm-started** from Method C's results to avoid bad local minima.

**Converting per-joint ε to a shared value:**

`config.py` uses a single global $\varepsilon$. We take the **median** of the per-joint
estimates, excluding any that hit the bounds (which signals the optimiser got stuck):

$$\varepsilon_{\text{final}} = \text{median}\!\left(\{\varepsilon_j : \varepsilon_j \notin \{0.02, 0.50\}\}\right)$$

> Reliability: **Medium** — may get trapped in local minima without good initialisation.
> Given weight 1.5 for Coulomb, 2.0 for viscous.
> **Code**: `calibrate_friction.py → fit_nonlinear_joint()`

---

### Consistency Check — "Does It Agree Across 124 Trajectories?"

**The idea:** A well-calibrated physical parameter should give the same estimate whether
you use a circle trajectory or a sine-wave trajectory, a fast motion or a slow one.

For each of the $K = 124$ trajectories, Method C is run independently to get a per-trajectory
estimate $c_j^{(k)}$ for each joint. The **Coefficient of Variation (CV%)** measures stability:

$$\text{CV}_j = \frac{\text{std}(c_j^{(1)}, \ldots, c_j^{(K)})}{\text{mean}(c_j^{(1)}, \ldots, c_j^{(K)})} \times 100\%$$

Think of CV% as: "If I re-run the calibration on a different trajectory, how much will my
estimate change?" A CV of 10% means ±10% typical variation — very stable. A CV of 60%
means the parameter is highly trajectory-dependent and unreliable.

| CV% | Confidence level | Action |
| --- | ---------------- | ------ |
| < 20% | HIGH | Trust this value |
| 20–40% | MEDIUM | Accept with caution |
| > 40% | LOW | Check servo hardware, sign convention |

---

### Synthesis — Combining All Four Methods

Each method gives its own $(c_j, v_j)$ estimates. Rather than picking one winner, the final
parameters are a **trust-weighted average**. Methods known to be more robust to gravity
contamination get higher weight:

$$c_j^{\text{final}} = \frac{3.0 \cdot c_j^B + 2.0 \cdot c_j^C + 1.5 \cdot c_j^D + 1.0 \cdot c_j^A}{3.0 + 2.0 + 1.5 + 1.0}$$

$$v_j^{\text{final}} = \frac{3.0 \cdot v_j^C + 2.0 \cdot v_j^D + 2.0 \cdot v_j^B + 1.0 \cdot v_j^A}{3.0 + 2.0 + 2.0 + 1.0}$$

| Method | Coulomb weight | Viscous weight | Why this weight |
| ------ | -------------- | -------------- | --------------- |
| B (Asymmetry) | 3.0 | 2.0 | Most robust to gravity error |
| C (ε sweep LS) | 2.0 | 3.0 | Full dataset, exact — best for viscous |
| D (Nonlinear) | 1.5 | 2.0 | Per-joint flexibility, but may overfit |
| A (Regime) | 1.0 | 1.0 | Simple heuristic, most noisy |

Final clamping ensures physical validity: $c_j \ge 0$, $v_j \ge 0$.

> **Code**: `calibrate_friction.py → synthesize_recommendation()`

---

## The Complete Model & What the PINN Must Learn

### The Full Physics Model

$$\boxed{\hat{\boldsymbol{\tau}}_{\text{model}}(t) = \underbrace{\alpha\left[\mathbf{M}^{(1)}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}^{(1)}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}^{(1)}(\mathbf{q})\right]}_{\text{RNEA with density-corrected URDF}} + \underbrace{\mathbf{c} \odot \tanh\!\left(\frac{\dot{\mathbf{q}}}{\varepsilon}\right) + \mathbf{v} \odot \dot{\mathbf{q}}}_{\text{calibrated friction}}}$$

where $\alpha = 0.1119$ and superscript $(1)$ means evaluated at $\alpha = 1$ (raw URDF).

### The Residual — Learning Target

$$\boxed{\boldsymbol{\tau}_{\text{residual}}(t) = \boldsymbol{\tau}_{\text{load}}(t) - \hat{\boldsymbol{\tau}}_{\text{model}}(t)}$$

Expanding what the residual contains (ordered by expected magnitude):

| Residual component | Physical origin | Depends on | Notes |
| ------------------ | --------------- | ---------- | ----- |
| Gravity model error | Non-uniform α across joints | $\mathbf{q}$ | Smooth, learnable |
| Mass distribution error | Metal servo bodies not modelled | $\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}$ | Largest at wrist |
| Stiction | Break-away friction threshold | Velocity sign changes | Transient |
| Gear backlash | Mechanical play in gearbox | Position | Dead-zone |
| Cable forces | Routing creates bias | $\mathbf{q}$ | Pose-dependent |
| Sensor noise | Quantisation of load register | Random | ~3 mN·m/bit |

### Per-Joint Residual Character

| Joint | What dominates the residual | Learnable? |
| ----- | --------------------------- | ---------- |
| J1 (yaw) | Bearing preload + cable drag; RNEA ≈ 0 | Yes — but small signal |
| J2 (shoulder) | Well-calibrated; residual mostly noise | Partially |
| J3 (elbow) | Slight α overestimate; small residual | Yes |
| J4 (wrist) | Metal servo mass undermodelled | Yes — smooth, position-dependent |
| J5 (wrist) | Most uncertain; check stall torque | Depends on hardware fix |
| J6 (tool) | Passive; residual = full load signal | No — just noise |

---

## Model Validation Metrics

### RNEA/Load Ratio

$$\text{ratio}_j = \frac{\text{RMS}(\tau_{\text{RNEA},j})}{\text{RMS}(\tau_{\text{load},j})}$$

This is a **magnitude comparison**: does the physics model produce torques of the right size?

| Ratio | Interpretation | Action |
| ----- | -------------- | ------ |
| ≈ 1.0 | Excellent magnitude match | — |
| < 1.0 | RNEA under-predicts (friction missing or stall torque overestimated) | Add friction or reduce `STALL_TORQUE_PER_JOINT[j]` |
| > 1.0 | RNEA over-predicts (mass model too heavy) | Reduce `MASS_SCALE` or check joint |

### Normalised Residual RMS (NRMSE)

$$\text{NRMSE}_j = \frac{\|\tau_{\text{load},j} - \tau_{\text{model},j}\|_2}{\|\tau_{\text{load},j}\|_2}$$

This is the **fraction of the load signal that the model fails to explain**. Unlike the
ratio, it accounts for both magnitude and shape errors. NRMSE = 0.15 means the model
explains 85% of the load variance.

| NRMSE | Quality | PINN implication |
| ----- | ------- | ---------------- |
| < 0.20 | Excellent | PINN learning signal is small — good |
| 0.20–0.50 | Acceptable | PINN must learn moderate corrections |
| > 0.50 | Poor | Check hardware parameters before training |

---

## Per-Joint Ratio Interpretation & Troubleshooting

| Joint | Observed ratio | Expected? | Root cause |
| ----- | -------------- | --------- | ---------- |
| J1 (yaw) | ~0.26 | Yes | Vertical yaw axis — near-zero gravity torque. Load captures bearing preload and cable drag which RNEA cannot model. |
| J2 (shoulder) | ~1.05 | Yes | Calibration reference joint — should be ~1.0. |
| J3 (elbow) | ~1.14 | Approximately | Slight mass overestimate from single global α. |
| J4 | ~0.39 | No | Most likely wrong stall torque (using 30 kgf·cm for a 14.8 kgf·cm servo). |
| J5 (wrist) | ~0.055 | No | Check: (1) stall torque, (2) sign convention, (3) servo model. |
| J6 (tool) | ~0.002 | Yes | Passive — RNEA returns ~0, load is quantisation noise. |

### Fixing J4 / J5

If J4 and J5 use smaller servos than J1–J3:

```python
# In config.py — update for your actual servo models
STALL_TORQUE_PER_JOINT = np.array([
    30.0,   # J1 — STS3215
    30.0,   # J2 — STS3215
    30.0,   # J3 — STS3215
    14.8,   # J4 — STS3032  ← update this
    14.8,   # J5 — STS3032  ← update this
    30.0,   # J6 — passive (value unused)
])
```

---

## Bulk Analysis & Statistical Summaries

### What `bulk_analyze.py` Does

Processes all 124 JSON log files in `raw_samples/`. For each file:

1. Loads and repairs data, converts to radians
2. Runs RNEA, friction model, and load torque computation
3. Computes per-joint metrics (NRMSE, RMS ratios, residual RMS)
4. Pre-bins error histograms (100 bins, ±2.5 N·m) for three model variants
5. Saves `infer_torque/<run_id>/torque.png` and `run_summary.json`

After all files, aggregates into `global_summary.json` with:

- `by_shape` — performance broken down by trajectory geometry (circle, ellipse, figure8, …)
- `by_traj_type` — performance by motion profile (ruckig, quintic_poly, …)
- `by_radius_mm` — NRMSE vs trajectory radius
- `model_quality` — fraction of runs where NRMSE < 20% and < 50% per joint
- `error_histograms` — pre-binned counts ready for plotting (no raw samples stored)

### Trajectory Metadata Parsing

Filenames encode the experiment parameters:

```text
circle_r65mm_xz_cx66cyn265cz275_quintic_poly_ctrlmax_fbmax_001.json
│       │      │                 │
shape   radius plane            trajectory type
```

This is extracted by `parse_filename()` and stored in `traj_meta` inside each run summary.

### Global Plots

Ten cross-run plots are generated in `infer_torque/global_plots/`:

| Plot | What it shows |
| ---- | ------------- |
| `rnea_ratio_violin.png` | Distribution of RNEA/Load ratio per joint |
| `nrmse_violin.png` | Distribution of NRMSE per joint |
| `residual_rms_boxplot.png` | Box plots of absolute residual RMS |
| `load_vs_rnea_scatter.png` | Scatter: load RMS vs RNEA RMS per run |
| `accuracy_by_shape.png` | Heatmap: shape × joint, median NRMSE % |
| `accuracy_by_traj_type.png` | Grouped bars per trajectory type |
| `accuracy_vs_radius.png` | NRMSE vs trajectory radius |
| `model_coverage_cdf.png` | CDF of NRMSE — what fraction of runs are "good"? |
| `error_hist_global.png` | Global error histograms (see below) |
| `error_hist_by_shape_J{1-5}.png` | Per-shape error per joint |

---

## Error Histogram Diagnostics

### Three Model Variants

For each joint, the pipeline computes three error signals and plots their distributions:

| Variant | Error definition | What the distribution reveals |
| ------- | ---------------- | ----------------------------- |
| **RNEA only** | $\tau_{\text{load}} - \tau_{\text{RNEA}}$ | Wide, offset histogram → friction is missing |
| **RNEA + Friction** | $\tau_{\text{load}} - (\tau_{\text{RNEA}} + \tau_{\text{fric}})$ | Narrow, centred → full model is good |
| **Friction only** | $\tau_{\text{load}} - \tau_{\text{fric}}$ | Wide offset → friction alone is insufficient |

### Reading the Histograms

A good model produces a **narrow, symmetric, zero-centred** error distribution:

```text
   density
      ↑
      │     ╭───╮           ← RNEA+Friction (narrow peak near 0)
      │    ╭╯   ╰╮
      │  ╭─╯     ╰─╮        ← RNEA only (wider, offset)
      │╭─╯         ╰─╮      ← Friction only (wide, wrong centre)
  ────┼──────────────────── error [N·m]
      0
```

### Memory-Efficient Accumulation

Instead of storing 470K raw error values per joint per variant (expensive), the pipeline
pre-bins counts during each run's processing:

```python
error_hists[j] = {
    "rnea":  np.histogram(err_rnea,  bins=HIST_BINS)[0].tolist(),  # 100 integers
    "model": np.histogram(err_model, bins=HIST_BINS)[0].tolist(),  # 100 integers
    "fric":  np.histogram(err_fric,  bins=HIST_BINS)[0].tolist(),  # 100 integers
}
```

Total storage: 100 bins × 3 variants × 6 joints × 124 runs ≈ 220K integers — trivial.
Integer bin counts accumulate across runs with simple addition (no precision loss).

> **Code**: `bulk_analyze.py` + `plots_global.py → plot_error_histograms_global()` / `plot_error_histograms_by_shape()`

---

## Output Layout

```text
infer_torque/
├── circle_r65mm_xz_..._001/           ← one folder per trajectory
│   ├── torque.png                     ← 6-joint torque comparison
│   └── run_summary.json               ← metrics + pre-binned histograms
│
├── global_plots/
│   ├── rnea_ratio_violin.png
│   ├── nrmse_violin.png
│   ├── residual_rms_boxplot.png
│   ├── load_vs_rnea_scatter.png
│   ├── accuracy_by_shape.png
│   ├── accuracy_by_traj_type.png
│   ├── accuracy_vs_radius.png
│   ├── model_coverage_cdf.png
│   ├── error_hist_global.png
│   ├── error_hist_by_shape_J1.png
│   ├── error_hist_by_shape_J2.png
│   ├── error_hist_by_shape_J3.png
│   ├── error_hist_by_shape_J4.png
│   └── error_hist_by_shape_J5.png
│
└── global_summary.json                ← schema v2.0, full rich metadata
```

---

## Running the Pipeline

```bash
cd ~/Desktop/MTP_PINN

# Step 1 — Mass calibration (once per robot build)
python3 -m Torque_Analysis.calibrate_mass
# → prints fitted α, update MASS_SCALE in config.py

# Step 2 — Friction calibration (bulk mode)
python3 -m Torque_Analysis.calibrate_friction --bulk
# → prints c_j, v_j, ε, update COULOMB_NM / VISCOUS_NM / FRICTION_EPS in config.py

# Step 3 — Single-file sanity check
python3 Torque_Analysis/main.py
# → check RNEA/Load ratios in the printed table; should be ~1.0 for J2, J3

# Step 4 — Full batch run
python3 Torque_Analysis/bulk_analyze.py
# → writes infer_torque/ with per-run folders + global_summary.json + global_plots/
```

---

## Configuration Reference

Everything calibratable lives in `config.py`. After running calibration scripts,
update the values printed to stdout:

```python
# ── Mass calibration ────────────────────────────────────────────
MASS_SCALE   = 0.111893        # global density scale α (from calibrate_mass.py)

# ── Servo hardware ──────────────────────────────────────────────
STALL_TORQUE_PER_JOINT = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
# ↑ Update indices 3,4 if J4/J5 use smaller servo models (e.g. 14.8 for STS3032)
NOM_VOLTAGE  = 12.0            # rated voltage (V)
KGCM_TO_NM   = 0.09807        # kgf·cm → N·m (physical constant, do not change)

# ── Signal processing ───────────────────────────────────────────
DIFF_METHOD      = "savgol"    # "savgol" (default) or "gradient"
SAVGOL_POLYORDER = 3           # polynomial order (3 = cubic, good default)
SMOOTH_WINDOW    = 11          # SG window length (odd integer, samples)

# ── Friction (from calibrate_friction.py --bulk) ────────────────
COULOMB_NM   = np.array([0.2472, 0.2873, 0.2403, 0.1806, 0.2151, 0.0])
VISCOUS_NM   = np.array([0.0,    0.3,    0.0,    0.051,  0.0042, 0.0])
FRICTION_EPS = 0.0628          # tanh transition width (rad/s)
```

---

## Known Limitations & Future Work

### Current Limitations

1. **Single global mass scale**: α was calibrated on J1–J2 only. Distal joints with
   heavy metal servo bodies have a different effective density — per-joint mass scaling
   would improve J3–J5 accuracy.

2. **Load register quantisation**: 0.1% of stall torque ≈ 3 mN·m per LSB. At low loads
   (J5 wrist, J6 tool), the quantisation noise is comparable to the signal itself.

3. **Sign convention validated for J1–J2 only**: The $-d_j$ convention was confirmed by
   calibrate_mass.py. If J4/J5 Pearson correlation with RNEA is negative, those joints
   may need a flipped sign in `torque_from_load`.

4. **Savitzky–Golay assumes near-uniform sampling**: The mean Δt is used as the step size.
   If control-loop jitter exceeds ±20% of mean Δt, consider resampling to a uniform grid.

5. **Global ε for friction**: All joints share one transition width. Method D shows per-joint
   ε differs by up to 40%; a per-joint `FRICTION_EPS` array would improve wrist joints.

### Suggested Improvements

- Measure J4/J5 servo models and update `STALL_TORQUE_PER_JOINT[3:5]`
- Validate and possibly flip sign convention for J4/J5 independently
- Extend mass calibration to joints 3–5 using separate α per link group
- Add payload identification (known end-effector mass at calibration poses)
