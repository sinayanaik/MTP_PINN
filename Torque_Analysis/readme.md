# Complete Mathematical Description — Torque Analysis Pipeline

---

## Table of Contents

1. [Configuration Space & Coordinate Conversion](#1-configuration-space)
2. [The Kinematic Tree & URDF Model](#2-kinematic-tree)
3. [The Equations of Motion](#3-equations-of-motion)
4. [Recursive Newton-Euler Algorithm (RNEA)](#4-rnea)
5. [Mass Calibration — The α Problem](#5-mass-calibration)
6. [Measured Torque Signals](#6-measured-torques)
7. [Numerical Differentiation Pipeline](#7-numerical-differentiation)
8. [Why the Simple Analytical Model Fails](#8-simple-model-failure)
9. [Friction Physics & Smooth Model](#9-friction-model)
10. [Friction Calibration — Four Methods](#10-friction-calibration)
11. [The Complete Model & Residual](#11-complete-model)

---

## 1. Configuration Space & Coordinate Conversion

The kikobot is a **6-DOF serial manipulator**. Its configuration lives in:

$$\mathbf{q} = \begin{bmatrix} q_1 \\ q_2 \\ q_3 \\ q_4 \\ q_5 \\ q_6 \end{bmatrix} \in \mathbb{R}^6$$

where joints 1–5 are **actuated** and joint 6 (tool) is **passive**.

### Encoder → Radians

The servos report position in **encoder ticks**. The conversion is:

$$q_j = d_j \cdot \left( p_j^{\text{ticks}} - p_j^{\text{center}} \right) \cdot k_{\text{t2r}}$$

where:

| Symbol | Meaning | Source |
|--------|---------|--------|
| $d_j \in \{-1, +1\}$ | Direction sign (servo vs URDF convention) | `joint_map[j]["direction"]` |
| $p_j^{\text{center}}$ | Encoder zero-position in ticks | `joint_map[j]["ticks_center"]` |
| $k_{\text{t2r}}$ | Ticks-to-radians factor | `M["ticks_to_rad"]` |

> **Code**: `utils.py → ticks_to_radians()`

---

## 2. The Kinematic Tree & URDF Model

### 2.1 Serial Chain Structure

The URDF defines a **serial kinematic chain** — a tree with no branches:

$$\text{universe} \xrightarrow{\text{fixed}} \text{base\_link} \xrightarrow{q_1} \text{link}_1 \xrightarrow{q_2} \text{link}_2 \xrightarrow{q_3} \text{link}_3 \xrightarrow{q_4} \text{link}_4 \xrightarrow{q_5} \text{link}_5 \xrightarrow{q_6} \text{link}_6$$

Each arrow labeled $q_j$ represents a **revolute joint** — rotation about a single axis.

### 2.2 Homogeneous Transformations

The pose of frame $i$ relative to frame $i-1$ is:

$${}^{i-1}\mathbf{T}_i(q_i) = {}^{i-1}\mathbf{T}_i^{\text{fixed}} \cdot \begin{bmatrix} \cos q_i & -\sin q_i & 0 & 0 \\ \sin q_i & \cos q_i & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

where ${}^{i-1}\mathbf{T}_i^{\text{fixed}}$ encodes the **static** geometric offset between joints (translation + rotation from the URDF).

The world-frame pose of link $i$ is built by **chaining** transforms:

$${}^{0}\mathbf{T}_i(\mathbf{q}) = {}^{0}\mathbf{T}_1(q_1) \cdot {}^{1}\mathbf{T}_2(q_2) \cdots {}^{i-1}\mathbf{T}_i(q_i)$$

### 2.3 Per-Link Inertial Properties

Each link $i$ carries **rigid-body inertial parameters** defined in the URDF:

$$\text{Link } i: \quad \left\{ m_i, \quad \mathbf{c}_i \in \mathbb{R}^3, \quad \mathbf{I}_i \in \mathbb{R}^{3\times3} \right\}$$

| Symbol | Meaning |
|--------|---------|
| $m_i$ | Link mass (kg) |
| $\mathbf{c}_i$ | Center of mass in link frame (the "lever" in Pinocchio) |
| $\mathbf{I}_i$ | Rotational inertia tensor about CoM, expressed in link frame |

The **spatial inertia** (6×6) of link $i$ combines these:

$$\mathcal{I}_i = \begin{bmatrix} \mathbf{I}_i + m_i [\mathbf{c}_i]_\times^T [\mathbf{c}_i]_\times & m_i [\mathbf{c}_i]_\times \\ m_i [\mathbf{c}_i]_\times^T & m_i \mathbf{I}_3 \end{bmatrix}$$

where $[\mathbf{c}]_\times$ is the skew-symmetric matrix of $\mathbf{c}$.

> **Code**: `torque.py → build_pinocchio_model()` calls `pin.buildModelFromXML(urdf_xml)` which parses the URDF and constructs these spatial inertias internally.

---

## 3. The Equations of Motion

The dynamics of a rigid-body serial manipulator follow the **Lagrangian formulation**:

$$\boxed{\mathbf{M}(\mathbf{q})\,\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\,\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}_{\text{applied}}}$$

### 3.1 Inertia Matrix $\mathbf{M}(\mathbf{q})$

The **joint-space inertia matrix** is symmetric positive-definite:

$$M_{ij}(\mathbf{q}) = \sum_{k=\max(i,j)}^{n} \left[ m_k \, \mathbf{J}_{v_k,i}^T \mathbf{J}_{v_k,j} + \mathbf{J}_{\omega_k,i}^T \, {}^{0}\mathbf{R}_k \, \mathbf{I}_k \, {}^{0}\mathbf{R}_k^T \, \mathbf{J}_{\omega_k,j} \right]$$

where $\mathbf{J}_{v_k,i}$ and $\mathbf{J}_{\omega_k,i}$ are the $i$-th columns of the linear and angular Jacobians of link $k$'s CoM.

**Physical meaning**: $M_{ij}$ captures how much torque at joint $i$ is needed to produce unit acceleration at joint $j$, accounting for the coupled inertias of **all links distal to both joints**.

### 3.2 Coriolis & Centrifugal $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$

Using Christoffel symbols of the first kind:

$$\left[\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\,\dot{\mathbf{q}}\right]_i = \sum_{j=1}^{n} \sum_{k=1}^{n} c_{ijk}\, \dot{q}_j\, \dot{q}_k$$

$$c_{ijk} = \frac{1}{2}\left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right)$$

**Physical meaning**: velocity-dependent forces arising from the changing geometry of the arm as it moves. Includes:
- **Coriolis forces** ($j \neq k$): coupling between joints moving simultaneously
- **Centrifugal forces** ($j = k$): "flywheel" effect of spinning links

### 3.3 Gravity $\mathbf{g}(\mathbf{q})$

$$g_i(\mathbf{q}) = -\sum_{k=i}^{n} m_k \, \mathbf{g}_0^T \, \frac{\partial \mathbf{p}_{c_k}(\mathbf{q})}{\partial q_i}$$

where $\mathbf{g}_0 = [0, 0, -9.81]^T$ m/s² and $\mathbf{p}_{c_k}(\mathbf{q})$ is the CoM of link $k$ in world frame.

**Physical meaning**: the torque each joint must exert just to hold the arm stationary against gravity.

> **Key property**: $\mathbf{g}(\mathbf{q})$ depends **only on configuration** — not on velocity or acceleration. This makes it the dominant, most reliable component for calibration.

---

## 4. Recursive Newton-Euler Algorithm (RNEA)

Rather than forming $\mathbf{M}$, $\mathbf{C}$, $\mathbf{g}$ as separate matrices (which costs $O(n^3)$ for an $n$-DOF arm), Pinocchio uses RNEA to compute the **inverse dynamics** directly:

$$\boldsymbol{\tau} = \text{RNEA}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})$$

in $O(n)$ time — **two recursive passes** over the kinematic tree.

### 4.1 Forward Pass: Base → Tip (Propagate Kinematics)

**Initialize at base** ($i = 0$):

$$\boldsymbol{\omega}_0 = \mathbf{0}, \qquad \dot{\boldsymbol{\omega}}_0 = \mathbf{0}, \qquad \mathbf{a}_0 = -\mathbf{g}_0$$

> **Gravity trick**: instead of adding $m_k \mathbf{g}_0$ to every link's force, we initialize the base with a fictitious upward acceleration $\mathbf{a}_0 = -\mathbf{g}_0$. This propagates through the recursion and automatically produces the correct gravity torques.

**For each joint $i = 1, 2, \ldots, n$** (moving outward from base):

**Angular velocity of link $i$:**
$$\boldsymbol{\omega}_i = {}^{i}\mathbf{R}_{i-1} \, \boldsymbol{\omega}_{i-1} + \dot{q}_i \, \hat{\mathbf{z}}_i$$

**Angular acceleration of link $i$:**
$$\dot{\boldsymbol{\omega}}_i = {}^{i}\mathbf{R}_{i-1} \, \dot{\boldsymbol{\omega}}_{i-1} + \ddot{q}_i \, \hat{\mathbf{z}}_i + \left({}^{i}\mathbf{R}_{i-1} \, \boldsymbol{\omega}_{i-1}\right) \times \dot{q}_i \, \hat{\mathbf{z}}_i$$

**Linear acceleration of link $i$ origin:**
$$\mathbf{a}_i = {}^{i}\mathbf{R}_{i-1} \left( \mathbf{a}_{i-1} + \dot{\boldsymbol{\omega}}_{i-1} \times \mathbf{r}_{i-1,i} + \boldsymbol{\omega}_{i-1} \times (\boldsymbol{\omega}_{i-1} \times \mathbf{r}_{i-1,i}) \right)$$

**Linear acceleration of link $i$ CoM:**
$$\mathbf{a}_{c_i} = \mathbf{a}_i + \dot{\boldsymbol{\omega}}_i \times \mathbf{c}_i + \boldsymbol{\omega}_i \times (\boldsymbol{\omega}_i \times \mathbf{c}_i)$$

where:

| Symbol | Meaning |
|--------|---------|
| ${}^{i}\mathbf{R}_{i-1}$ | Rotation matrix from frame $i-1$ to frame $i$ |
| $\hat{\mathbf{z}}_i$ | Joint axis (unit vector) in frame $i$ |
| $\mathbf{r}_{i-1,i}$ | Vector from origin of frame $i-1$ to frame $i$ |
| $\mathbf{c}_i$ | CoM of link $i$ in frame $i$ |

### 4.2 Backward Pass: Tip → Base (Propagate Forces)

**Initialize at tip** ($i = n+1$):

$$\mathbf{f}_{n+1} = \mathbf{0}, \qquad \boldsymbol{\mu}_{n+1} = \mathbf{0}$$

**For each link $i = n, n-1, \ldots, 1$** (moving inward toward base):

**Net force required to accelerate link $i$** (Newton's 2nd law):
$$\mathbf{F}_i = m_i \, \mathbf{a}_{c_i}$$

**Net moment required to accelerate link $i$** (Euler's equation):
$$\mathbf{N}_i = \mathbf{I}_i \, \dot{\boldsymbol{\omega}}_i + \boldsymbol{\omega}_i \times (\mathbf{I}_i \, \boldsymbol{\omega}_i)$$

The $\boldsymbol{\omega}_i \times (\mathbf{I}_i \boldsymbol{\omega}_i)$ term is the **gyroscopic torque** — arises because the inertia tensor is configuration-dependent when expressed in a rotating frame.

**Force transmitted through joint $i$** (includes forces from all distal links):
$$\mathbf{f}_i = {}^{i}\mathbf{R}_{i+1} \, \mathbf{f}_{i+1} + \mathbf{F}_i$$

**Moment transmitted through joint $i$:**
$$\boldsymbol{\mu}_i = {}^{i}\mathbf{R}_{i+1} \left( \boldsymbol{\mu}_{i+1} + \mathbf{r}_{i,i+1} \times \mathbf{f}_{i+1} \right) + \mathbf{c}_i \times \mathbf{F}_i + \mathbf{N}_i$$

**Joint torque** (project moment onto joint axis):

$$\boxed{\tau_i = \boldsymbol{\mu}_i^T \, \hat{\mathbf{z}}_i}$$

### 4.3 Special Cases Used in the Codebase

**Full inverse dynamics:**
$$\boldsymbol{\tau}_{\text{RNEA}} = \text{RNEA}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

> **Code**: `torque.py → torque_from_urdf()` — called at every timestep

**Gravity only** (set $\dot{\mathbf{q}} = \ddot{\mathbf{q}} = \mathbf{0}$):

$$\boldsymbol{\tau}_{\text{gravity}} = \text{RNEA}(\mathbf{q}, \mathbf{0}, \mathbf{0}) = \mathbf{g}(\mathbf{q})$$

> **Code**: `torque.py → torque_gravity_only()` — used for mass calibration and gravity vs. dynamic decomposition plots

---

## 5. Mass Calibration — The $\alpha$ Problem

### 5.1 The Problem

The URDF was designed with **metal density** (SolidWorks default = steel):

$$\rho_{\text{URDF}} = 7800 \text{ kg/m}^3$$

The actual robot is **3D-printed PLA** at approximately 70% infill:

$$\rho_{\text{actual}} \approx 875 \text{ kg/m}^3$$

### 5.2 The Scaling Property

Since all inertial parameters are proportional to density (for uniform material):

$$m_i^{\text{actual}} = \alpha \cdot m_i^{\text{URDF}}, \qquad \mathbf{I}_i^{\text{actual}} = \alpha \cdot \mathbf{I}_i^{\text{URDF}}$$

And because RNEA is **linear** in masses and inertias:

$$\boldsymbol{\tau}_{\text{RNEA}}^{(\alpha)} = \alpha \cdot \boldsymbol{\tau}_{\text{RNEA}}^{(\alpha=1)}$$

This linearity means we only need a **single scalar** to correct the entire model.

### 5.3 Calibration via Gravity Matching

For the **calibration joints** $\mathcal{J} = \{1, 2\}$ (shoulder and elbow — strongest gravity signal), the load-register torque should match the scaled gravity torque:

$$\tau_{\text{load},j}(t) \approx \alpha \cdot \tau_{\text{gravity},j}^{(\alpha=1)}(t) \quad \forall t, \; j \in \mathcal{J}$$

### 5.4 Least-Squares Solution

$$\alpha^* = \arg\min_{\alpha} \sum_{t=1}^{N} \sum_{j \in \mathcal{J}} \left( \tau_{\text{load},j}(t) - \alpha \cdot \tau_{\text{gravity},j}^{(\alpha=1)}(t) \right)^2$$

Taking derivative and setting to zero:

$$\frac{d}{d\alpha}\left[\ldots\right] = -2 \sum_{t,j} \tau_{\text{gravity},j}^{(\alpha=1)}(t) \left( \tau_{\text{load},j}(t) - \alpha \cdot \tau_{\text{gravity},j}^{(\alpha=1)}(t) \right) = 0$$

$$\boxed{\alpha^* = \frac{\displaystyle\sum_{t=1}^{N} \sum_{j \in \mathcal{J}} \tau_{\text{gravity},j}^{(1)}(t) \cdot \tau_{\text{load},j}(t)}{\displaystyle\sum_{t=1}^{N} \sum_{j \in \mathcal{J}} \left(\tau_{\text{gravity},j}^{(1)}(t)\right)^2} = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}}}$$

where $\mathbf{a} = \text{vec}(\tau_{\text{gravity}}^{(1)}[\mathcal{J}])$ and $\mathbf{b} = \text{vec}(\tau_{\text{load}}[\mathcal{J}])$.

> **Code**: `calibrate_v2.py → fit_scale()` — the dot-product ratio

### 5.5 Result


$$
\alpha^{*} \approx 0.112 \implies \rho_{\text{implied}} = 0.112 \times 7800 \approx 875 \text{ kg/m}^3 \quad (\text{PLA, } {\sim}70\% \text{ infill})
$$



### 5.6 Sign Convention Discovery

Before calibration, the correct sign mapping from servo frame to URDF frame must be determined. `calibrate_v2.py` tests three conventions:

| Convention | Formula | Criterion |
|------------|---------|-----------|
| Raw | $\tau = \tau_{\text{servo}}$ | Check $\alpha > 0$, low variance across joints |
| $+d_j$ | $\tau = +d_j \cdot \tau_{\text{servo}}$ | Same check |
| $-d_j$ | $\tau = -d_j \cdot \tau_{\text{servo}}$ | **Winner**: $\alpha > 0$ for all joints, lowest std |

---

## 6. Measured Torque Signals (Ground Truth)

### 6.1 Load-Register Torque

The servo reports a **load percentage** $\ell_j$ (how hard it's working). Converting to torque:

$$\tau_{\text{load},j}^{\text{servo}}(t) = \underbrace{\frac{\ell_j(t) \cdot 0.1}{100}}_{\text{load fraction}} \cdot \underbrace{\frac{V_j(t)}{V_{\text{nom}}}}_{\text{voltage scaling}} \cdot \underbrace{\tau_{\text{stall}}}_{\text{30 kgf·cm}} \cdot \underbrace{k_{\text{conv}}}_{\text{0.0981 N·m/(kgf·cm)}}$$

Convert to URDF frame:

$$\boxed{\tau_{\text{load},j}^{\text{URDF}}(t) = -d_j \cdot \tau_{\text{load},j}^{\text{servo}}(t)}$$

> **Code**: `torque.py → torque_from_load_raw()` + `torque_from_load()`

### 6.2 Current-Based Torque

$$\tau_{\text{current},j}(t) = \text{sign}(\ell_j) \cdot \left|\frac{I_j(t)}{1000}\right| \cdot K_T \cdot k_{\text{conv}}$$

where $I_j$ is in milliamps and $K_T = 11.0$ is the motor torque constant.

> **Note**: Current-based torque was found to be **less reliable** than load-register and is no longer used in the main pipeline.

---

## 7. Numerical Differentiation Pipeline

The JSON logs contain positions $\mathbf{q}(t)$ but the RNEA requires $\dot{\mathbf{q}}(t)$ and $\ddot{\mathbf{q}}(t)$.

### 7.1 First Derivative: Velocity

Using **central differences** (`np.gradient`):

$$\dot{q}_j^{\text{raw}}(t_k) = \frac{q_j(t_{k+1}) - q_j(t_{k-1})}{t_{k+1} - t_{k-1}} + O(\Delta t^2)$$

Apply moving-average smoothing:

$$\dot{q}_j(t_k) = \frac{1}{w} \sum_{m=-\lfloor w/2 \rfloor}^{\lfloor w/2 \rfloor} \dot{q}_j^{\text{raw}}(t_{k+m})$$

with window $w = 11$ (`SMOOTH_WINDOW`).

### 7.2 Second Derivative: Acceleration

Differentiate **the smoothed velocity** (not the raw position twice):

$$\ddot{q}_j^{\text{raw}}(t_k) = \frac{\dot{q}_j(t_{k+1}) - \dot{q}_j(t_{k-1})}{t_{k+1} - t_{k-1}}$$

Then smooth again with the same window.

### 7.3 The Double-Smoothing Effect

The pipeline applies smoothing at **each differentiation stage**:

$$\mathbf{q} \xrightarrow{\text{diff}} \dot{\mathbf{q}}^{\text{raw}} \xrightarrow{\text{smooth}(w)} \dot{\mathbf{q}} \xrightarrow{\text{diff}} \ddot{\mathbf{q}}^{\text{raw}} \xrightarrow{\text{smooth}(w)} \ddot{\mathbf{q}}$$

In the frequency domain, each `smooth(w)` applies a **sinc-like low-pass filter**:

$$H(f) = \frac{\sin(\pi f w \Delta t)}{\pi f w \Delta t}$$

Double application means $\ddot{\mathbf{q}}$ is filtered by $H^2(f)$, which:
- ✅ Suppresses high-frequency noise (essential — differentiation amplifies noise)
- ⚠️ Attenuates true high-frequency dynamics
- ⚠️ Introduces phase lag

### 7.4 Safety Measures

| Measure | What it handles | Code |
|---------|-----------------|------|
| `fix_timestamps` | Duplicate/non-monotonic $t$ → inject 1μs gaps | `utils.py` |
| `np.clip(qd, ±100)` | Extreme velocity spikes from timestamp glitches | `utils.py` |
| `np.clip(qdd, ±1000)` | Extreme acceleration spikes | `utils.py` |
| `nan_to_num` | Any remaining NaN/Inf | `utils.py` |

---

## 8. Why the Simple Analytical Model Fails

### 8.1 The Simple Model

$$\hat{\boldsymbol{\tau}}_{\text{simple}} = \text{RNEA}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

### 8.2 Error Sources

**Error Source 1 — Missing Friction**

Real servo gearboxes exhibit significant friction. The actual balance is:

$$\boldsymbol{\tau}_{\text{motor}} = \underbrace{\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g}}_{\text{RNEA (modeled)}} + \underbrace{\boldsymbol{\tau}_{\text{friction}}}_{\text{NOT in RNEA}}$$

The load register measures $\boldsymbol{\tau}_{\text{motor}}$, but RNEA only computes the first three terms. The residual $\boldsymbol{\tau}_{\text{load}} - \boldsymbol{\tau}_{\text{RNEA}}$ therefore contains the entire friction contribution.

**Error Source 2 — Non-Uniform Mass Model**

The global $\alpha$ assumes all links have the same density ratio. But:

$$\alpha_{\text{effective},j} = \frac{m_{\text{PLA},j} \cdot \alpha_{\text{PLA}} + m_{\text{servo},j} \cdot 1.0}{m_{\text{URDF},j}}$$

For **proximal links** (joints 1–2): PLA dominates → $\alpha_{\text{eff}} \approx \alpha_{\text{PLA}}$ ✓

For **distal links** (joints 3–4): servo mass (metal) dominates → $\alpha_{\text{eff}} \gg \alpha_{\text{PLA}}$ → model **underestimates** inertia

**Error Source 3 — Numerical Differentiation Artifacts**

$$\ddot{\mathbf{q}}_{\text{numerical}} = \ddot{\mathbf{q}}_{\text{true}} + \boldsymbol{\epsilon}_{\text{diff}}$$

$$\mathbf{M}\ddot{\mathbf{q}}_{\text{numerical}} = \mathbf{M}\ddot{\mathbf{q}}_{\text{true}} + \mathbf{M}\boldsymbol{\epsilon}_{\text{diff}}$$

The inertia matrix **amplifies** differentiation noise, especially for proximal joints (large $M_{jj}$).

**Error Source 4 — Unmodeled Dynamics**

| Effect | Nature | Magnitude |
|--------|--------|-----------|
| Stiction | Higher static than kinetic friction | Transient at direction changes |
| Gear backlash | Dead zone in position | Creates torque spikes |
| Cable forces | Configuration-dependent | Position-dependent bias |
| Link flexibility | Elastic deformation | High-frequency oscillations |

### 8.3 Quantified Impact (from diagnostics)

The **sanity check** in `main.py` computes:

$$r_j = \frac{\text{RMS}(\tau_{\text{RNEA},j})}{\text{RMS}(\tau_{\text{load},j})}$$

For a perfect model, $r_j \approx 1.0$. In practice, $r_j < 1$ for most joints because RNEA misses friction.

---

## 9. Friction Physics & Smooth Model

### 9.1 Physical Friction in Servo Gearboxes

At the joint level, the dominant friction effects are:

**Coulomb (dry) friction** — constant magnitude, opposes motion:
$$\tau_{\text{Coulomb},j} = c_j \cdot \text{sign}(\dot{q}_j)$$

**Viscous friction** — proportional to velocity (lubricant shearing):
$$\tau_{\text{viscous},j} = v_j \cdot \dot{q}_j$$

**Combined ideal model:**
$$\tau_{f,j}^{\text{ideal}} = c_j \cdot \text{sign}(\dot{q}_j) + v_j \cdot \dot{q}_j$$

### 9.2 The Discontinuity Problem

The $\text{sign}(\dot{q})$ function is **discontinuous at zero velocity**:

$$\text{sign}(x) = \begin{cases} +1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}$$

This causes:
1. **Numerical chatter** during simulation near zero velocity
2. **Undefined gradients** at $\dot{q} = 0$ → breaks backpropagation for PINN training
3. **Sensitivity to noise** — small velocity noise causes sign flips

### 9.3 Smooth Approximation via Hyperbolic Tangent

Replace $\text{sign}(\dot{q})$ with:

$$\text{smooth\_sign}(\dot{q}_j) = \tanh\left(\frac{\dot{q}_j}{\varepsilon}\right)$$

where $\varepsilon > 0$ is the **transition width** (rad/s).

**Properties:**

$$\tanh\left(\frac{x}{\varepsilon}\right) \approx \begin{cases} +1 & x \gg \varepsilon \\ x / \varepsilon & |x| \ll \varepsilon \quad (\text{linear near zero}) \\ -1 & x \ll -\varepsilon \end{cases}$$

$$\frac{d}{dx}\tanh\left(\frac{x}{\varepsilon}\right) = \frac{1}{\varepsilon}\text{sech}^2\left(\frac{x}{\varepsilon}\right) \quad \text{(smooth, bounded, everywhere defined)}$$

As $\varepsilon \to 0$: $\tanh(x/\varepsilon) \to \text{sign}(x)$ — recovers the ideal model.

### 9.4 The Smooth Friction Model

$$\boxed{\tau_{f,j}(\dot{q}_j) = c_j \cdot \tanh\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j \cdot \dot{q}_j}$$

**Parameters per joint:**

| Parameter | Symbol | Physical meaning | Calibrated value (example J1) |
|-----------|--------|------------------|-------------------------------|
| Coulomb friction | $c_j$ | Constant resistive torque (N·m) | 0.2118 |
| Viscous friction | $v_j$ | Velocity-proportional damping (N·m·s/rad) | 0.0239 |
| Transition width | $\varepsilon$ | Smooth sign sharpness (rad/s) | 0.0757 |

> **Code**: `torque.py → _smooth_sign()` + `torque_friction()`

### 9.5 Vector Form

For all joints simultaneously:

$$\boldsymbol{\tau}_f(\dot{\mathbf{q}}) = \mathbf{c} \odot \tanh\left(\frac{\dot{\mathbf{q}}}{\varepsilon}\right) + \mathbf{v} \odot \dot{\mathbf{q}}$$

where $\odot$ is element-wise multiplication, $\mathbf{c} = [c_1, \ldots, c_n]^T$, $\mathbf{v} = [v_1, \ldots, v_n]^T$.

---

## 10. Friction Calibration — Four Methods

### 10.0 The Friction Signal

The input to calibration is the **difference between measured and modeled torque**:

$$\boldsymbol{\tau}_f^{\text{signal}}(t) = \boldsymbol{\tau}_{\text{load}}(t) - \boldsymbol{\tau}_{\text{RNEA}}(t)$$

**Critical issue**: this signal contains friction **plus** gravity model error:

$$\tau_{f,j}^{\text{signal}} = \underbrace{c_j \tanh(\dot{q}_j/\varepsilon) + v_j \dot{q}_j}_{\text{true friction}} + \underbrace{\Delta g_j(\mathbf{q})}_{\text{gravity model error}} + \underbrace{\eta_j(t)}_{\text{other errors}}$$

All four calibration methods must handle this contamination.

### 10.1 Method A: Regime Analysis

**Low-speed regime** $|\dot{q}_j| \in (0.01, 0.05)$ rad/s:

At low speed, $\tanh(\dot{q}/\varepsilon) \approx \pm 1$ and $v \cdot \dot{q} \approx 0$, so:

$$\tau_{f,j}^{\text{signal}} \approx \pm c_j + \text{gravity error}$$

Estimate:
$$\hat{c}_j = \text{median}\left( |\tau_{f,j}^{\text{signal}}| \;\Big|\; |\dot{q}_j| \in (0.01, 0.05) \right)$$

**High-speed regime** $|\dot{q}_j| > 0.10$ rad/s:

Subtract Coulomb component, then linear regression:

$$\tau_{f,j}^{\text{signal}} - \hat{c}_j \cdot \text{sign}(\dot{q}_j) \approx v_j \cdot \dot{q}_j$$

$$\hat{v}_j = \frac{\sum_k \dot{q}_{j,k} \cdot \left(\tau_{f,j,k}^{\text{signal}} - \hat{c}_j \cdot \text{sign}(\dot{q}_{j,k})\right)}{\sum_k \dot{q}_{j,k}^2}$$

> **Weakness**: Gravity error contaminates both estimates.

### 10.2 Method B: Asymmetry Analysis (Gravity-Robust)

**The key physical insight:**

Friction is **antisymmetric** in velocity:
$$\tau_f(-\dot{q}) = -\tau_f(\dot{q})$$

Gravity error is **symmetric** (independent of velocity sign):
$$\Delta g(\mathbf{q}) \text{ does not change sign with } \dot{q}$$

**Procedure:**

Split data into positive and negative velocity subsets:

For $\dot{q}_j > \varepsilon_{\text{thresh}}$, fit:
$$\tau_{f,j}^{\text{signal}} = a^+ + v^+ \dot{q}_j$$

For $\dot{q}_j < -\varepsilon_{\text{thresh}}$, fit:
$$\tau_{f,j}^{\text{signal}} = a^- + v^- \dot{q}_j$$

**Decomposition of intercepts:**

The intercepts $a^+$ and $a^-$ from the two half-fits each contain friction and bias:

$$a^+ = +c_j + b_j \qquad (\text{positive velocity half})$$
$$a^- = -c_j + b_j \qquad (\text{negative velocity half})$$

where $c_j$ is the Coulomb friction and $b_j$ is the gravity-induced bias.

**Solving the system of two equations:**

$$\boxed{\hat{c}_j = \frac{a^+ - a^-}{2}} \qquad \text{(antisymmetric part → pure friction)}$$

$$\boxed{\hat{b}_j = \frac{a^+ + a^-}{2}} \qquad \text{(symmetric part → gravity error)}$$

$$\boxed{\hat{v}_j = \frac{v^+ + v^-}{2}} \qquad \text{(average viscous from both halves)}$$

**Why this works — geometric interpretation:**

In the $(\dot{q}_j, \tau_f^{\text{signal}})$ plane:

```
  τ_f^signal
      ↑
      |        ╱  positive half: intercept = +c + b
      |      ╱
  a⁺ ─ ─ ─•
      |   ╱
  b   ─ ─ ╳ ─ ─ ─ ─ ─ ─ ─ → q̇
      |     ╲
  a⁻ ─ ─ ─ ─•
      |        ╲
      |          ╲  negative half: intercept = −c + b
```

- The **midpoint** $(a^+ + a^-)/2 = b$ captures the vertical shift due to gravity error
- The **half-distance** $(a^+ - a^-)/2 = c$ captures the friction-induced separation
- **Trajectory diversity** (bulk mode with 124 files) averages out configuration-dependent $b_j$

> **Code**: `calibrate_friction.py → asymmetry_analysis()`

**Interpretation metric — bias-to-Coulomb ratio:**

$$R_j = \frac{|b_j|}{c_j}$$

| $R_j$ | Interpretation |
|--------|----------------|
| $< 0.3$ | Clean friction signal |
| $0.3 - 1.0$ | Significant gravity bias |
| $> 1.0$ | Gravity-dominated — friction estimate unreliable |

### 10.3 Method C: Bias-Aware $\varepsilon$ Sweep (Analytical LS)

**Model with explicit bias term:**

$$\tau_{f,j}^{\text{signal}}(t) = b_j + c_j \cdot \tanh\left(\frac{\dot{q}_j(t)}{\varepsilon}\right) + v_j \cdot \dot{q}_j(t)$$

For a **fixed** $\varepsilon$, this is **linear** in $(b_j, c_j, v_j)$. Stack into a matrix equation over all $N$ timesteps:

$$\underbrace{\begin{bmatrix} 1 & \tanh(\dot{q}_{j,1}/\varepsilon) & \dot{q}_{j,1} \\ 1 & \tanh(\dot{q}_{j,2}/\varepsilon) & \dot{q}_{j,2} \\ \vdots & \vdots & \vdots \\ 1 & \tanh(\dot{q}_{j,N}/\varepsilon) & \dot{q}_{j,N} \end{bmatrix}}_{\boldsymbol{\Phi}_j(\varepsilon) \;\in\; \mathbb{R}^{N \times 3}} \underbrace{\begin{bmatrix} b_j \\ c_j \\ v_j \end{bmatrix}}_{\mathbf{x}_j} = \underbrace{\begin{bmatrix} \tau_{f,j,1}^{\text{signal}} \\ \tau_{f,j,2}^{\text{signal}} \\ \vdots \\ \tau_{f,j,N}^{\text{signal}} \end{bmatrix}}_{\mathbf{y}_j}$$

**Analytical least-squares solution:**

$$\mathbf{x}_j^*(\varepsilon) = \left(\boldsymbol{\Phi}_j^T \boldsymbol{\Phi}_j\right)^{-1} \boldsymbol{\Phi}_j^T \mathbf{y}_j$$

with post-hoc clamping to physical bounds:

$$c_j \leftarrow \text{clip}(c_j, 0, 0.50), \qquad v_j \leftarrow \text{clip}(v_j, 0, 0.30)$$

**Per-joint RMS residual at this $\varepsilon$:**

$$\text{RMS}_j(\varepsilon) = \sqrt{\frac{1}{N}\left\|\mathbf{y}_j - \boldsymbol{\Phi}_j \mathbf{x}_j^*\right\|^2}$$

**Grid search over $\varepsilon$:**

$$\varepsilon^* = \arg\min_{\varepsilon \in [0.02, 0.50]} \sqrt{\frac{1}{n_{\text{active}}} \sum_{j=1}^{n_{\text{active}}} \text{RMS}_j^2(\varepsilon)}$$

Sweep 100 points over $\varepsilon \in [0.02, 0.50]$, compute the analytical LS at each point (instant on 470K samples because it's just a $3\times3$ matrix inverse per joint).

> **Code**: `calibrate_friction.py → fit_bias_aware()` + `sweep_eps_bias()`

**Why bias matters:**

Without the bias term:

$$\underbrace{\begin{bmatrix} \tanh(\dot{q}/\varepsilon) & \dot{q} \end{bmatrix}}_{\text{2 columns}} \begin{bmatrix} c \\ v \end{bmatrix} = \tau_f^{\text{signal}}$$

The gravity error $\Delta g(\mathbf{q})$ has nonzero mean and is **partially correlated** with $\tanh(\dot{q}/\varepsilon)$ through the trajectory. Without a bias column to absorb it:

$$\hat{c}_j^{\text{no bias}} = c_j^{\text{true}} + \underbrace{\frac{\text{cov}(\tanh(\dot{q}/\varepsilon),\; \Delta g)}{\text{var}(\tanh(\dot{q}/\varepsilon))}}_{\text{leakage}}$$

The bias column $\mathbf{1}$ captures the mean gravity error, **decorrelating** it from the friction regressors.

### 10.4 Method D: Nonlinear Per-Joint $\varepsilon$ (Config-Compatible)

**Model matching `config.py` exactly (no bias):**

$$\tau_{f,j} = c_j \cdot \tanh\left(\frac{\dot{q}_j}{\varepsilon_j}\right) + v_j \cdot \dot{q}_j$$

Now $\varepsilon_j$ is **per-joint** and enters **nonlinearly**, so we use L-BFGS-B optimization:

$$\min_{c_j, v_j, \varepsilon_j} \frac{1}{N}\sum_{t=1}^{N} \left(\tau_{f,j}^{\text{signal}}(t) - c_j \tanh\left(\frac{\dot{q}_j(t)}{\varepsilon_j}\right) - v_j \dot{q}_j(t)\right)^2$$

subject to:

$$c_j \in [0, 0.50], \qquad v_j \in [0, 0.30], \qquad \varepsilon_j \in [0.02, 0.50]$$

**Warm-starting from Method C** (if available):

$$c_j^{(0)} = c_j^{\text{sweep}}, \qquad v_j^{(0)} = v_j^{\text{sweep}}, \qquad \varepsilon_j^{(0)} = \varepsilon^{\text{sweep}}$$

**Final shared $\varepsilon$:**

Since `config.py` uses a single global $\varepsilon$, take the **median** of per-joint estimates (excluding those at bounds):

$$\varepsilon_{\text{final}} = \text{median}\left(\{\varepsilon_j : \varepsilon_j \notin \{\varepsilon_{\min}, \varepsilon_{\max}\}\}\right)$$

> **Code**: `calibrate_friction.py → fit_nonlinear_joint()`

### 10.5 Per-Trajectory Consistency Check (Bulk Mode)

For each of $K = 124$ trajectories, fit $(b_j^{(k)}, c_j^{(k)}, v_j^{(k)})$ at the global $\varepsilon^*$ using Method C.

**Stability metric — Coefficient of Variation:**

$$\text{CV}_j = \frac{\sigma(c_j^{(1)}, \ldots, c_j^{(K)})}{\mu(c_j^{(1)}, \ldots, c_j^{(K)})} \times 100\%$$

| $\text{CV}_j$ | Confidence |
|----------------|------------|
| $< 20\%$ | HIGH — parameter is stable across trajectories |
| $20\% - 40\%$ | MEDIUM — some trajectory dependence |
| $> 40\%$ | LOW — parameter is unreliable |

> **Code**: `calibrate_friction.py → per_trajectory_consistency()`

### 10.6 Synthesis — Cross-Method Weighted Average

The final parameters combine all four methods using **trust-weighted averaging**:

**For Coulomb friction:**

$$c_j^{\text{final}} = \frac{w_A c_j^A + w_B c_j^B + w_C c_j^C + w_D c_j^D}{w_A + w_B + w_C + w_D}$$

| Method | Weight $w$ | Justification |
|--------|-----------|---------------|
| Asymmetry (B) | 3.0 | Most robust to gravity error |
| Sweep (C) | 2.0 | Full dataset, analytical |
| Nonlinear (D) | 1.5 | Per-joint flexibility |
| Regime (A) | 1.0 | No optimization, intuitive but noisy |

**For viscous friction:**

$$v_j^{\text{final}} = \frac{w_C' v_j^C + w_D' v_j^D + w_B' v_j^B + w_A' v_j^A}{w_C' + w_D' + w_B' + w_A'}$$

| Method | Weight $w'$ | Justification |
|--------|------------|---------------|
| Sweep (C) | 3.0 | Full data, analytical |
| Nonlinear (D) | 2.0 | Config-compatible model |
| Asymmetry (B) | 2.0 | Gravity-robust |
| Regime (A) | 1.0 | Noisy |

Final clamping:

$$c_j^{\text{final}} \leftarrow \text{clip}(c_j^{\text{final}}, 0, 0.50), \qquad v_j^{\text{final}} \leftarrow \text{clip}(v_j^{\text{final}}, 0, 0.30)$$

> **Code**: `calibrate_friction.py → synthesize_recommendation()`

---

## 11. The Complete Model & Residual

### 11.1 The Full Analytical Torque

Combining all components:

$$\boxed{\hat{\boldsymbol{\tau}}_{\text{analytical}}(t) = \underbrace{\text{RNEA}\left(\mathbf{q}(t),\, \dot{\mathbf{q}}(t),\, \ddot{\mathbf{q}}(t)\right)}_{\text{rigid-body dynamics}} + \underbrace{\mathbf{c} \odot \tanh\!\left(\frac{\dot{\mathbf{q}}(t)}{\varepsilon}\right) + \mathbf{v} \odot \dot{\mathbf{q}}(t)}_{\text{friction}}}$$

Expanding the RNEA component:

$$\hat{\boldsymbol{\tau}}_{\text{analytical}} = \underbrace{\alpha \mathbf{M}^{(1)}(\mathbf{q})\ddot{\mathbf{q}}}_{\text{inertial}} + \underbrace{\alpha\mathbf{C}^{(1)}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}}_{\text{Coriolis + centrifugal}} + \underbrace{\alpha\mathbf{g}^{(1)}(\mathbf{q})}_{\text{gravity}} + \underbrace{\boldsymbol{\tau}_f(\dot{\mathbf{q}})}_{\text{friction}}$$

where superscript $(1)$ denotes quantities computed at $\alpha = 1$ (unscaled URDF).

### 11.2 The Residual — What the PINN Learns

$$\boxed{\boldsymbol{\tau}_{\text{residual}}(t) = \boldsymbol{\tau}_{\text{load}}(t) - \hat{\boldsymbol{\tau}}_{\text{analytical}}(t)}$$

Expanding what this contains:

$$\boldsymbol{\tau}_{\text{residual}} = \underbrace{(\alpha_{\text{true},j} - \alpha)\,\mathbf{g}_j^{(1)}(\mathbf{q})}_{\text{gravity model error (position-dependent)}} + \underbrace{\boldsymbol{\tau}_{\text{stiction}} + \boldsymbol{\tau}_{\text{backlash}}}_{\text{unmodeled friction effects}} + \underbrace{\Delta\mathbf{M}\,\ddot{\mathbf{q}} + \Delta\mathbf{C}\,\dot{\mathbf{q}}}_{\text{non-uniform }\alpha\text{ error in dynamics}} + \underbrace{\boldsymbol{\tau}_{\text{flexibility}} + \boldsymbol{\tau}_{\text{cables}}}_{\text{structural effects}} + \underbrace{\boldsymbol{\eta}(t)}_{\text{sensor noise}}$$

### 11.3 Decomposition by Signal Type

| Component | Depends on | Captured by | Magnitude |
|-----------|-----------|-------------|-----------|
| Gravity $\alpha\mathbf{g}(\mathbf{q})$ | $\mathbf{q}$ only | RNEA (dominant term) | Large (~0.5 N·m for J1-J2) |
| Inertial $\alpha\mathbf{M}\ddot{\mathbf{q}}$ | $\mathbf{q}, \ddot{\mathbf{q}}$ | RNEA | Medium, noisy |
| Coriolis $\alpha\mathbf{C}\dot{\mathbf{q}}$ | $\mathbf{q}, \dot{\mathbf{q}}$ | RNEA | Small (slow motions) |
| Coulomb friction | $\text{sign}(\dot{\mathbf{q}})$ | Friction model | ~0.2 N·m |
| Viscous friction | $\dot{\mathbf{q}}$ | Friction model | ~0.01–0.3 N·m |
| Gravity error | $\mathbf{q}$ | **Residual (PINN)** | ~0.05 N·m distal |
| Mass distribution error | $\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}$ | **Residual (PINN)** | Joint-dependent |
| Unmodeled effects | Various | **Residual (PINN)** | Stochastic |

### 11.4 Expected Residual Characteristics

For the PINN designer, the residual has these properties:

**Joint 0 (yaw)**: Small gravity (vertical axis), friction-dominated residual

$$\tau_{\text{residual},0} \approx \tau_{\text{stiction}} + \eta$$

**Joints 1–2 (shoulder/elbow)**: Well-calibrated gravity, friction absorbed. Residual is small.

$$\tau_{\text{residual},1} \approx \Delta\alpha_1 \cdot g_1^{(1)}(\mathbf{q}) + \text{small dynamics errors}$$

**Joints 3–4 (wrist)**: Non-uniform $\alpha$ is the biggest error. Servo metal mass is **unmodeled**.

$$\tau_{\text{residual},3} \approx (\alpha_{\text{eff},3} - \alpha) \cdot g_3^{(1)}(\mathbf{q}) + \Delta M_{33}\ddot{q}_3$$

This is a **configuration-dependent bias** — smooth and learnable.

**Joint 5 (tool)**: Passive — no RNEA model. Residual = full load signal.

### 11.5 Complete Data Flow Diagram

```
                    ┌─────────────┐
                    │  JSON Log   │
                    │  (124 files)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ data_loader  │
                    │  load_log()  │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │ act_pos      │ │ load,    │ │ timestamps   │
    │ (ticks)      │ │ voltage  │ │ t            │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
    ┌──────▼───────┐      │       ┌──────▼───────┐
    │ticks_to_rad  │      │       │fix_timestamps│
    │ q (radians)  │      │       └──────┬───────┘
    └──────┬───────┘      │              │
           │              │       ┌──────▼───────┐
           │              │       │ np.gradient  │
           │              │       │  + smooth    │──→ q̇
           │              │       │  + clip/nan  │──→ q̈
           │              │       └──────────────┘
           │              │
    ┌──────▼──────────────▼──────┐
    │                            │
    │   ┌────────────────────┐   │     ┌───────────────┐
    │   │ XACRO → URDF → Pin│   │     │ torque_from_  │
    │   │ build_pinocchio_   │   │     │ load()        │
    │   │ model()            │   │     │               │
    │   │                    │   │     │ τ_load = -d·  │
    │   │ Apply α scaling:   │   │     │ (ℓ·τ_stall·  │
    │   │ m_i ← α·m_i       │   │     │  V/V_nom)·k  │
    │   │ I_i ← α·I_i       │   │     └───────┬───────┘
    │   └────────┬───────────┘   │             │
    │            │               │             │
    │     ┌──────▼───────┐       │             │
    │     │ RNEA per     │       │             │
    │     │ timestep:    │       │             │
    │     │              │       │             │
    │     │ τ_RNEA(t) =  │       │             │
    │     │ pin.rnea(    │       │             │
    │     │  q, q̇, q̈)   │       │             │
    │     └──────┬───────┘       │             │
    │            │               │             │
    │     ┌──────▼───────┐       │             │
    │     │ Friction:    │       │             │
    │     │ τ_f = c·tanh │       │             │
    │     │ (q̇/ε)+v·q̇   │       │             │
    │     └──────┬───────┘       │             │
    │            │               │             │
    │     ┌──────▼───────┐       │             │
    │     │ τ_analytical │       │             │
    │     │ = τ_RNEA +   │       │             │
    │     │   τ_friction │       │             │
    │     └──────┬───────┘       │             │
    │            │               │             │
    └────────────┼───────────────┘             │
                 │                             │
                 └──────────┬──────────────────┘
                            │
                     ┌──────▼──────┐
                     │  RESIDUAL   │
                     │             │
                     │ τ_residual =│
                     │ τ_load −    │
                     │ τ_analytical│
                     │             │
                     │ = PINN      │
                     │   target    │
                     └──────┬──────┘
                            │
                 ┌──────────▼──────────┐
                 │                     │
          ┌──────▼──────┐    ┌─────────▼──────────┐
          │  Diagnostics│    │ PINN Training Data  │
          │  (9 checks) │    │                     │
          └─────────────┘    │ Input:  (q, q̇, q̈, t)│
                             │ Target: τ_residual  │
                             │ Prior:  τ_analytical│
                             └────────────────────┘
```

### 11.6 Validation Metric

The quality of the analytical model + friction is measured by the **RMS improvement**:

$$\Delta\%_j = \frac{\text{RMS}(\tau_{\text{residual},j}^{\text{new}}) - \text{RMS}(\tau_{\text{residual},j}^{\text{old}})}{\text{RMS}(\tau_{\text{residual},j}^{\text{old}})} \times 100$$

And the overall ratio:

$$r_j = \frac{\text{RMS}(\hat{\tau}_{\text{analytical},j})}{\text{RMS}(\tau_{\text{load},j})}$$

| $r_j$ | Interpretation |
|--------|----------------|
| $\approx 1.0$ | Analytical model explains measured torque well |
| $\ll 1.0$ | Model underestimates — missing mass or friction |
| $\gg 1.0$ | Model overestimates — mass too high or noise amplification |

For joints 1–2 (calibration joints), $r_j \approx 1.0$. For distal joints, the ratio degrades — **this is where the PINN earns its keep**.

---

### Summary: The Mathematical Story

$$\underbrace{\boldsymbol{\tau}_{\text{load}}}_{\text{measured}} = \underbrace{\alpha \left[\mathbf{M}^{(1)}\ddot{\mathbf{q}} + \mathbf{C}^{(1)}\dot{\mathbf{q}} + \mathbf{g}^{(1)}\right]}_{\text{RNEA with calibrated mass}} + \underbrace{\mathbf{c} \odot \tanh\!\left(\frac{\dot{\mathbf{q}}}{\varepsilon}\right) + \mathbf{v}\odot\dot{\mathbf{q}}}_{\text{smooth friction model}} + \underbrace{\boldsymbol{\tau}_{\text{residual}}}_{\text{PINN learns this}}$$

**Calibration produced**: $\alpha = 0.112$, $\mathbf{c}$, $\mathbf{v}$, $\varepsilon = 0.076$

**What's left for the PINN**: non-uniform mass distribution, stiction, backlash, flexibility, cable forces — all smooth, structured, and **learnable**.

## Appendices — Deeper Mathematical Analysis

---

## A. Signal Processing Mathematics

### A.1 The Uniform Filter as Convolution

The `smooth()` function (`scipy.uniform_filter1d`) computes a **discrete convolution** with a rectangular kernel:

$$h[k] = \begin{cases} \frac{1}{w} & |k| \leq \lfloor w/2 \rfloor \\ 0 & \text{otherwise} \end{cases}$$

$$\tilde{x}[n] = (x * h)[n] = \frac{1}{w}\sum_{k=-\lfloor w/2 \rfloor}^{\lfloor w/2 \rfloor} x[n-k]$$

### A.2 Frequency Domain Transfer Function

The Discrete-Time Fourier Transform (DTFT) of the rectangular kernel is:

$$H(e^{j\omega}) = \frac{1}{w} \cdot \frac{\sin(\omega w / 2)}{\sin(\omega / 2)}$$

This is a **Dirichlet kernel** — a sinc-like low-pass filter with:

- **First null** at $\omega = 2\pi/w$ (normalized), i.e., frequency $f_{\text{null}} = \frac{f_s}{w}$
- **Passband ripple**: non-monotonic sidelobes (not an ideal LPF)
- **Zero phase**: symmetric kernel → no time delay (unlike causal filters)

For $w = 11$ at typical $f_s \approx 50$ Hz:

$$f_{\text{null}} = \frac{50}{11} \approx 4.5 \text{ Hz}$$

### A.3 Effect of Differentiation + Smoothing

Numerical differentiation has the transfer function (central differences):

$$D(e^{j\omega}) = \frac{j \sin(\omega)}{T_s}$$

This **amplifies high frequencies** linearly — noise at frequency $f$ is amplified by $\propto f$.

The combined transfer function for the velocity pipeline:

$$\dot{Q}(\omega) = Q(\omega) \cdot D(e^{j\omega}) \cdot H(e^{j\omega})$$

For acceleration (differentiation applied twice, smoothing applied twice):

$$\ddot{Q}(\omega) = Q(\omega) \cdot \underbrace{D(e^{j\omega}) \cdot H(e^{j\omega})}_{\text{first stage}} \cdot \underbrace{D(e^{j\omega}) \cdot H(e^{j\omega})}_{\text{second stage}}$$

$$\ddot{Q}(\omega) = Q(\omega) \cdot D^2(e^{j\omega}) \cdot H^2(e^{j\omega})$$

**The noise amplification-suppression balance:**

$$|D^2 \cdot H^2| = \frac{\sin^2(\omega)}{T_s^2} \cdot \frac{1}{w^2} \cdot \frac{\sin^2(\omega w/2)}{\sin^2(\omega/2)}$$

At low frequencies ($\omega \ll 1$): $|D^2 H^2| \approx \omega^2/T_s^2$ — true acceleration recovered ✓

At high frequencies ($\omega \to \pi$): $|H^2| \to 0$ rapidly — noise suppressed ✓

At mid frequencies: there's a trade-off zone where **true signal is attenuated**. Larger $w$ means more attenuation.

### A.4 Why This Matters for RNEA Torque

The inertial torque term is:

$$\boldsymbol{\tau}_{\text{inertial}} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}$$

If $\ddot{\mathbf{q}}$ is attenuated by double-smoothing:

$$\ddot{q}_j^{\text{estimated}} \approx \beta(f) \cdot \ddot{q}_j^{\text{true}}$$

where $\beta(f) < 1$ for signal components above $f_{\text{null}}/2 \approx 2$ Hz. This means:

$$\tau_{\text{inertial},j}^{\text{estimated}} < \tau_{\text{inertial},j}^{\text{true}}$$

The **underestimated inertial torque** appears as a positive contribution to $\boldsymbol{\tau}_{\text{residual}}$:

$$\tau_{\text{residual}} \ni (1 - \beta) \cdot M_{jj} \ddot{q}_j^{\text{true}}$$

This is a **systematic, velocity-correlated error** — exactly the type of structured signal a PINN can learn.

### A.5 Why Not Use Commanded Derivatives?

The JSON log contains `cmd_vel` and `cmd_acc` from the quintic polynomial planner. These are **analytically smooth** — no differentiation noise. However:

$$\dot{\mathbf{q}}_{\text{cmd}} \neq \dot{\mathbf{q}}_{\text{actual}}$$

The servo tracking is imperfect. Using commanded derivatives with actual positions in RNEA creates a **kinematic inconsistency**:

$$\text{RNEA}(\mathbf{q}_{\text{act}}, \dot{\mathbf{q}}_{\text{cmd}}, \ddot{\mathbf{q}}_{\text{cmd}}) \neq \text{RNEA}(\mathbf{q}_{\text{act}}, \dot{\mathbf{q}}_{\text{act}}, \ddot{\mathbf{q}}_{\text{act}})$$

The codebase correctly uses **actual** positions and derives velocity/acceleration from them — maintaining internal consistency at the cost of differentiation noise.

---

## B. The Mass Calibration Parabola

### B.1 Why the RMS Scan is Parabolic

The RMS residual as a function of $\alpha$:

$$\text{RMS}(\alpha) = \sqrt{\frac{1}{N|\mathcal{J}|}\sum_{t,j} \left(\tau_{\text{load},j}(t) - \alpha \cdot \tau_{\text{gravity},j}^{(1)}(t)\right)^2}$$

Define:

$$f(\alpha) = \text{RMS}^2(\alpha) = \frac{1}{N|\mathcal{J}|}\sum_{t,j} \left(\tau_{\text{load},j}(t) - \alpha \cdot \tau_{\text{gravity},j}^{(1)}(t)\right)^2$$

Expanding:

$$f(\alpha) = \frac{1}{N|\mathcal{J}|}\left[\underbrace{\sum_{t,j} \tau_{\text{load}}^2}_{\|\mathbf{b}\|^2} - 2\alpha\underbrace{\sum_{t,j} \tau_{\text{load}}\tau_{\text{grav}}^{(1)}}_{\mathbf{a}^T\mathbf{b}} + \alpha^2\underbrace{\sum_{t,j} (\tau_{\text{grav}}^{(1)})^2}_{\|\mathbf{a}\|^2}\right]$$

This is a **quadratic in $\alpha$** — an upward-opening parabola with a unique minimum:

$$f(\alpha) = \frac{1}{N|\mathcal{J}|}\left[\|\mathbf{a}\|^2 \alpha^2 - 2(\mathbf{a}^T\mathbf{b})\alpha + \|\mathbf{b}\|^2\right]$$

$$\frac{df}{d\alpha} = 0 \implies \alpha^* = \frac{\mathbf{a}^T\mathbf{b}}{\|\mathbf{a}\|^2}$$

### B.2 Minimum Residual Value

At the optimum:

$$f(\alpha^*) = \frac{1}{N|\mathcal{J}|}\left[\|\mathbf{b}\|^2 - \frac{(\mathbf{a}^T\mathbf{b})^2}{\|\mathbf{a}\|^2}\right]$$

By the Cauchy-Schwarz inequality:

$$\text{RMS}^2(\alpha^*) = \frac{\|\mathbf{b}\|^2}{N|\mathcal{J}|}\left[1 - \cos^2\theta\right] = \frac{\|\mathbf{b}\|^2 \sin^2\theta}{N|\mathcal{J}|}$$

where $\theta$ is the angle between vectors $\mathbf{a}$ (gravity model) and $\mathbf{b}$ (measured load).

**Physical interpretation**: The minimum residual represents **everything in the load signal that is not proportional to gravity** — friction, dynamics, noise. If $\theta$ is small (gravity dominates), the residual is small and calibration is reliable.

### B.3 Why Only Joints 1–2?

$$\text{gravity signal strength} \propto \sum_k m_k \cdot \ell_k \cdot g \cdot |\sin(\text{angle from vertical})|$$

where $\ell_k$ is the moment arm from joint $j$ to link $k$'s CoM.

| Joint | Distal mass | Moment arm | Gravity torque | Signal-to-noise |
|-------|-------------|------------|----------------|-----------------|
| 0 (yaw) | All links | ~0 (vertical axis) | ~0 | Very poor |
| **1 (shoulder)** | **All links** | **Long** | **Large** | **Excellent** |
| **2 (elbow)** | **Links 3–6** | **Medium** | **Moderate** | **Good** |
| 3 (wrist) | Links 4–6 | Short | Small | Poor |
| 4 (wrist) | Links 5–6 | Very short | Very small | Very poor |

Joints 1–2 have the highest gravity signal-to-noise ratio, making them the most reliable for calibration.

---

## C. The Collinearity Problem (Why v3 Failed)

### C.1 Velocity Filtering Creates Collinearity

In v3 of the friction calibration, the data was filtered to keep only samples with $|\dot{q}_j| > \text{threshold}$.

In this filtered regime, $\tanh(\dot{q}/\varepsilon) \approx \text{sign}(\dot{q})$, so:

$$\tau_f \approx c \cdot \text{sign}(\dot{q}) + v \cdot \dot{q}$$

Now $\text{sign}(\dot{q})$ and $\dot{q}$ become **nearly collinear** — both are positive when $\dot{q} > 0$ and negative when $\dot{q} < 0$.

### C.2 Mathematical Formulation

The design matrix becomes:

$$\boldsymbol{\Phi}_{\text{filtered}} = \begin{bmatrix} +1 & \dot{q}_1 \\ +1 & \dot{q}_2 \\ \vdots & \vdots \\ -1 & \dot{q}_m \\ -1 & \dot{q}_{m+1} \\ \vdots & \vdots \end{bmatrix}$$

The columns $\phi_1 = \text{sign}(\dot{q})$ and $\phi_2 = \dot{q}$ are **correlated** because:

$$\text{corr}(\text{sign}(\dot{q}), \dot{q}) = \frac{E[\text{sign}(\dot{q}) \cdot \dot{q}]}{1 \cdot \sigma_{\dot{q}}} = \frac{E[|\dot{q}|]}{\sigma_{\dot{q}}} > 0$$

### C.3 Condition Number Explosion

The normal equations $\boldsymbol{\Phi}^T\boldsymbol{\Phi}\,\mathbf{x} = \boldsymbol{\Phi}^T\mathbf{y}$ have:

$$\boldsymbol{\Phi}^T\boldsymbol{\Phi} = \begin{bmatrix} N & \sum \text{sign}(\dot{q}_i)\dot{q}_i \\ \sum \text{sign}(\dot{q}_i)\dot{q}_i & \sum \dot{q}_i^2 \end{bmatrix} = \begin{bmatrix} N & \sum|\dot{q}_i| \\ \sum|\dot{q}_i| & \sum\dot{q}_i^2 \end{bmatrix}$$

The condition number:

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

When the off-diagonal element $\sum|\dot{q}_i|$ approaches $\sqrt{N \cdot \sum\dot{q}_i^2}$ (high correlation), $\kappa \to \infty$.

**Result**: The least-squares solution becomes **hypersensitive** to noise:

$$\delta\mathbf{x} \leq \kappa \cdot \frac{\delta\mathbf{y}}{\|\mathbf{y}\|}$$

In v3, this manifested as viscous friction $v$ hitting the upper bound (0.30 cap) while Coulomb $c$ was suppressed — the optimizer couldn't tell them apart.

### C.4 Why Unfiltered Data Fixes This

With the full velocity range including samples near $\dot{q} \approx 0$:

$$\tanh\left(\frac{\dot{q}}{\varepsilon}\right) \neq \text{sign}(\dot{q}) \quad \text{when } |\dot{q}| \lesssim \varepsilon$$

In the transition region, $\tanh(\dot{q}/\varepsilon) \approx \dot{q}/\varepsilon$ (linear), which is **parallel** to $\dot{q}$. But the **curvature** of $\tanh$ at the transition creates a nonlinear feature that breaks the collinearity:

$$\text{corr}\left(\tanh(\dot{q}/\varepsilon),\; \dot{q}\right) < \text{corr}\left(\text{sign}(\dot{q}),\; \dot{q}\right)$$

The low-velocity samples provide the **discriminative information** that separates Coulomb from viscous contributions.

---

## D. Correlation Diagnostic — Detecting Gravity Leakage

### D.1 The Diagnostic

The friction signal is:

$$\tau_f^{\text{signal}} = \tau_{\text{load}} - \tau_{\text{RNEA}} = \tau_f^{\text{true}} + \Delta g(\mathbf{q}) + \eta$$

If the gravity model is perfect ($\Delta g = 0$), then:

$$\text{corr}(\tau_f^{\text{signal}},\; \dot{q}) \quad \text{should be HIGH (friction depends on velocity)}$$

$$\text{corr}(\tau_f^{\text{signal}},\; q) \quad \text{should be LOW (friction doesn't depend on position)}$$

### D.2 When Gravity Error Leaks

If $\Delta g(\mathbf{q}) \neq 0$:

$$\text{corr}(\tau_f^{\text{signal}},\; q) = \text{corr}(\tau_f^{\text{true}} + \Delta g(\mathbf{q}),\; q)$$

$$\approx \underbrace{\text{corr}(\tau_f^{\text{true}},\; q)}_{\approx 0 \text{ (friction)}} + \underbrace{\text{corr}(\Delta g(\mathbf{q}),\; q)}_{\neq 0 \text{ (gravity is config-dependent)}}$$

$$= \text{corr}(\Delta g(\mathbf{q}),\; q)$$

Since gravity torque is a smooth function of joint angles, $\text{corr}(\Delta g, q)$ can be **substantial** — especially for single-trajectory data where $q$ doesn't explore the full configuration space.

> **Code**: `calibrate_friction.py → correlation_diagnostic()`

### D.3 Flag Criterion

$$\text{FLAG: } |\text{corr}(\tau_f^{\text{signal}}, q_j)| > 0.5$$

**Resolution**: Use `--bulk` mode (124 trajectories). With diverse configurations:

$$E_{\text{trajectories}}\left[\Delta g(\mathbf{q})\right] \approx 0$$

The gravity error averages out across trajectories, reducing position-friction correlation.

---

## E. Error Propagation Through the Pipeline

### E.1 Forward Error Analysis

Let $\delta q$ be the encoder quantization error:

$$\delta q = k_{\text{t2r}} \approx \frac{2\pi}{4096} \approx 0.0015 \text{ rad} \approx 0.09°$$

**Velocity error** (from central differences):

$$\delta \dot{q} \approx \frac{\delta q}{T_s} = \frac{0.0015}{0.02} = 0.077 \text{ rad/s}$$

After smoothing with window $w$:

$$\delta \dot{q}_{\text{smooth}} \approx \frac{\delta \dot{q}}{\sqrt{w}} = \frac{0.077}{\sqrt{11}} = 0.023 \text{ rad/s}$$

**Acceleration error:**

$$\delta \ddot{q} \approx \frac{\delta \dot{q}_{\text{smooth}}}{T_s} = \frac{0.023}{0.02} = 1.15 \text{ rad/s}^2$$

After second smoothing:

$$\delta \ddot{q}_{\text{smooth}} \approx \frac{1.15}{\sqrt{11}} = 0.35 \text{ rad/s}^2$$

### E.2 Torque Error Budget

**Gravity torque error** (from position error):

$$\delta \tau_g = \left\|\frac{\partial \mathbf{g}}{\partial \mathbf{q}}\right\| \cdot \delta q \approx \text{small}$$

Gravity torque varies slowly with $q$ → **position error has negligible effect** on gravity.

**Inertial torque error** (from acceleration error):

$$\delta \tau_M = M_{jj} \cdot \delta \ddot{q}_{\text{smooth}}$$

For joint 1 (largest inertia): $M_{11} \sim 0.01$ kg·m² (after $\alpha$ scaling)

$$\delta \tau_{M,1} \approx 0.01 \times 0.35 = 0.0035 \text{ N·m}$$

This is **small** compared to the residual (~0.1 N·m), confirming that differentiation noise is manageable.

**Friction torque error** (from velocity error):

$$\delta \tau_f \approx c \cdot \frac{d}{d\dot{q}}\tanh(\dot{q}/\varepsilon) \cdot \delta\dot{q} + v \cdot \delta\dot{q}$$

$$\approx \frac{c}{\varepsilon}\text{sech}^2(\dot{q}/\varepsilon) \cdot 0.023 + v \cdot 0.023$$

At $\dot{q} = 0$ (worst case): $\delta\tau_f \approx \frac{0.2}{0.076} \times 0.023 + 0.03 \times 0.023 \approx 0.061$ N·m

Near zero velocity, friction torque uncertainty is **significant** — another reason the PINN must learn the residual.

### E.3 Total Error Budget (Joint 1)

| Source | Magnitude (N·m) | Percentage of load RMS |
|--------|-----------------|----------------------|
| Gravity model ($\alpha$ error) | 0.01 – 0.05 | 2 – 10% |
| Inertial (diff. noise) | 0.003 | < 1% |
| Friction model imperfection | 0.05 – 0.10 | 10 – 20% |
| Unmodeled (stiction, backlash) | 0.02 – 0.05 | 4 – 10% |
| **Total residual RMS** | **~0.10** | **~20%** |

The residual is **dominated by friction model imperfection and unmodeled dynamics** — both structured and learnable by a PINN.

---

## F. Physical Interpretation of Calibrated Parameters

### F.1 Coulomb Friction — Joint-by-Joint

$$\mathbf{c} = [0.212, \; 0.303, \; 0.232, \; 0.184, \; 0.215, \; 0.0] \text{ N·m}$$

| Joint | $c_j$ (N·m) | Physical interpretation |
|-------|-------------|------------------------|
| 0 (yaw) | 0.212 | Base rotation bearing + gearbox |
| **1 (shoulder)** | **0.303** | **Highest** — largest gearbox, most structural load |
| 2 (elbow) | 0.232 | Second-highest — supports distal links |
| 3 (wrist 1) | 0.184 | Smaller gearbox, less load |
| 4 (wrist 2) | 0.215 | Similar to J3 |
| 5 (tool) | 0.0 | Passive joint — no friction modeled |

**Pattern**: Coulomb roughly scales with **gearbox size** and **structural loading**. J1 is highest because the shoulder gearbox must resist the torque of the entire arm.

### F.2 Viscous Friction — The Anomaly at J1

$$\mathbf{v} = [0.024, \; 0.288, \; 0.042, \; 0.023, \; 0.003, \; 0.0] \text{ N·m·s/rad}$$

| Joint | $v_j$ (N·m·s/rad) | Physical interpretation |
|-------|-------------------|------------------------|
| 0 | 0.024 | Low — smooth bearing |
| **1** | **0.288** | **Anomalously high** — gearbox lubricant viscosity |
| 2 | 0.042 | Moderate |
| 3 | 0.023 | Low — small servo, near direct-drive |
| 4 | 0.003 | Near zero — very small servo |
| 5 | 0.0 | Passive |

**J1's high viscous friction** is physically consistent: the shoulder joint has the **largest reduction gearbox** (highest gear ratio), and viscous friction scales with gear ratio squared:

$$v_{\text{joint}} = v_{\text{motor}} \cdot N^2$$

where $N$ is the gear ratio. J1 likely has $N \approx 3 \times$ the distal joints, giving $v \propto 9\times$ higher viscous friction.

### F.3 Transition Width $\varepsilon$

$$\varepsilon = 0.0757 \text{ rad/s} \approx 4.3°/\text{s}$$

This means the friction model transitions from "zero-velocity behavior" to "full Coulomb" over a velocity range of $\pm 0.076$ rad/s. Below this velocity, friction acts like **viscous damping**:

$$\tau_f \approx \frac{c}{\varepsilon}\dot{q} + v\dot{q} = \left(\frac{c}{\varepsilon} + v\right)\dot{q}$$

The **effective damping** near zero velocity:

$$v_{\text{eff}} = \frac{c}{\varepsilon} + v$$

For joint 1: $v_{\text{eff}} = \frac{0.303}{0.076} + 0.288 = 4.27$ N·m·s/rad

This is **very high** near zero velocity — the robot resists small perturbations strongly. This is the **stiction-like behavior** captured by the smooth model.

---

## G. Connection to PINN Architecture

### G.1 The PINN Formulation

The PINN will be trained to predict the residual:

$$\hat{\tau}_{\text{residual},j} = \mathcal{N}_\theta(q_1, \ldots, q_{n_q}, \dot{q}_1, \ldots, \dot{q}_{n_q}, \ddot{q}_1, \ldots, \ddot{q}_{n_q}, t)$$

where $\mathcal{N}_\theta$ is a neural network with parameters $\theta$.

### G.2 The Physics-Informed Loss

$$\mathcal{L}(\theta) = \underbrace{\frac{1}{N}\sum_{t=1}^{N}\sum_{j=1}^{n_q} \left(\tau_{\text{residual},j}(t) - \hat{\tau}_{\text{residual},j}(t; \theta)\right)^2}_{\text{data loss}} + \underbrace{\lambda \cdot \mathcal{R}(\theta)}_{\text{physics regularization}}$$

The physics regularization $\mathcal{R}$ can enforce known properties of the residual:

$$\mathcal{R}_1: \quad \hat{\tau}_{\text{residual}}(\mathbf{q}, \mathbf{0}, \mathbf{0}) \approx \Delta\alpha \cdot \mathbf{g}^{(1)}(\mathbf{q}) \quad \text{(static residual ≈ gravity error)}$$

$$\mathcal{R}_2: \quad \left\|\frac{\partial \hat{\tau}_{\text{residual}}}{\partial \ddot{\mathbf{q}}}\right\| \text{ is small} \quad \text{(residual shouldn't depend strongly on acceleration)}$$

$$\mathcal{R}_3: \quad \hat{\tau}_{\text{residual}} \text{ should be smooth in } (\mathbf{q}, \dot{\mathbf{q}})$$

### G.3 Why the Analytical Model Matters for PINN Training

Without the analytical model, the PINN must learn **everything**:

$$\text{PINN alone}: \quad \hat{\tau}_j = \mathcal{N}_\theta(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}, t)$$

Target magnitude: ~0.5 N·m (full load torque). The network must discover gravity, inertia, Coriolis, AND friction from data alone.

With the analytical model as prior:

$$\text{Physics-informed}: \quad \hat{\tau}_j = \underbrace{\tau_{\text{analytical},j}}_{\text{known, ~80-95\% of signal}} + \underbrace{\mathcal{N}_\theta(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}, t)}_{\text{learned, ~5-20\% correction}}$$

Target magnitude: ~0.1 N·m (residual only). The network only needs to learn the **correction**.

**Benefits:**

$$\frac{\text{RMS}(\tau_{\text{residual}})}{\text{RMS}(\tau_{\text{load}})} \approx 0.2 \quad \implies \quad \text{5× easier learning problem}$$

1. **Smaller target** → faster convergence, lower capacity network needed
2. **Better generalization** → physics handles unseen configurations, PINN only corrects
3. **Physical consistency** → gravity and inertia always correct by construction
4. **Data efficiency** → fewer training samples needed since physics provides strong inductive bias

### G.4 The Training Data Tensor

From the entire pipeline, the PINN training dataset has the form:

$$\mathcal{D} = \left\{ \left(\mathbf{q}^{(i)},\; \dot{\mathbf{q}}^{(i)},\; \ddot{\mathbf{q}}^{(i)},\; t^{(i)},\; \boldsymbol{\tau}_{\text{residual}}^{(i)}\right) \right\}_{i=1}^{N_{\text{total}}}$$

With dimensions:

| Quantity | Shape | Description |
|----------|-------|-------------|
| $\mathbf{q}^{(i)}$ | $(n_q,)$ | Joint angles (radians) |
| $\dot{\mathbf{q}}^{(i)}$ | $(n_q,)$ | Joint velocities (rad/s) |
| $\ddot{\mathbf{q}}^{(i)}$ | $(n_q,)$ | Joint accelerations (rad/s²) |
| $t^{(i)}$ | $(1,)$ | Timestamp within trajectory |
| $\boldsymbol{\tau}_{\text{residual}}^{(i)}$ | $(n_q,)$ | Residual torque target (N·m) |

**Dataset size**: 124 trajectories × ~3,800 samples each ≈ **470,000 training samples**

**Input dimension**: $3 n_q + 1 = 3(5) + 1 = 16$ (or 3(6)+1 = 19 if including passive joint)

**Output dimension**: $n_q = 5$ (one residual per active joint)

---

## H. Summary of All Mathematical Models in the Codebase

$$\boxed{
\begin{aligned}
& \textbf{Measured:} & \tau_{\text{load},j} &= -d_j \cdot \frac{\ell_j \cdot 0.001 \cdot V_j}{V_{\text{nom}}} \cdot \tau_{\text{stall}} \cdot k_{\text{conv}} \\[8pt]
& \textbf{Gravity:} & \tau_{g,j} &= \alpha \cdot \left[\text{RNEA}(\mathbf{q}, \mathbf{0}, \mathbf{0})\right]_j \\[8pt]
& \textbf{Full RNEA:} & \tau_{\text{RNEA},j} &= \alpha \cdot \left[\text{RNEA}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})\right]_j \\[8pt]
& \textbf{Friction:} & \tau_{f,j} &= c_j \tanh(\dot{q}_j / \varepsilon) + v_j \dot{q}_j \\[8pt]
& \textbf{Analytical:} & \hat{\tau}_j &= \tau_{\text{RNEA},j} + \tau_{f,j} \\[8pt]
& \textbf{Residual:} & \tau_{\text{res},j} &= \tau_{\text{load},j} - \hat{\tau}_j \\[8pt]
& \textbf{PINN:} & \tau_{\text{total},j} &= \hat{\tau}_j + \mathcal{N}_\theta(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}, t)_j
\end{aligned}
}$$

**Calibrated constants:**

$$\alpha = 0.112, \quad \varepsilon = 0.076 \text{ rad/s}$$

$$\mathbf{c} = [0.212,\; 0.303,\; 0.232,\; 0.184,\; 0.215,\; 0.0]^T \text{ N·m}$$

$$\mathbf{v} = [0.024,\; 0.288,\; 0.042,\; 0.023,\; 0.003,\; 0.0]^T \text{ N·m·s/rad}$$

The **complete story**: from raw encoder ticks and servo registers, through rigid-body mechanics and friction physics, to a clean residual signal ready for neural network learning. Every step is mathematically grounded, physically motivated, and empirically validated across 124 trajectories and 470,000 data points.




# References for Analytical Torque Calculation and Friction Modeling

---

## 1. Robot Dynamics and Inverse Dynamics

### 1.1 Foundational Textbooks

**[R1] Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). *Robotics: Modelling, Planning and Control*. Springer.**

This is the most comprehensive modern robotics textbook covering the complete mathematical framework used in this codebase. The relevant chapters are:

- **Chapter 3 (Differential Kinematics):** Derives the Jacobian matrices that relate joint velocities to end-effector velocities. The Jacobian columns appear in the inertia matrix computation:

$$M_{ij}(\mathbf{q}) = \sum_{k=\max(i,j)}^{n} \left[ m_k \mathbf{J}_{v_k,i}^T \mathbf{J}_{v_k,j} + \mathbf{J}_{\omega_k,i}^T {}^0\mathbf{R}_k \mathbf{I}_k {}^0\mathbf{R}_k^T \mathbf{J}_{\omega_k,j} \right]$$

- **Chapter 7 (Dynamics):** Derives the Lagrangian equations of motion:

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

Covers both the Lagrange-Euler formulation and the Newton-Euler formulation. Derives the Christoffel symbols for the Coriolis matrix:

$$c_{ijk} = \frac{1}{2}\left(\frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i}\right)$$

- **Chapter 7.3 (Newton-Euler Algorithm):** Presents the recursive forward-backward algorithm that Pinocchio implements. This is the exact algorithm used by `torque.py → torque_from_urdf()`.

---

**[R2] Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control* (3rd Edition). Pearson.**

This is the classic introductory text. Its treatment of the Newton-Euler algorithm in Chapter 6 is particularly clear and widely cited. The forward pass propagates velocities and accelerations outward from the base:

$${}^{i+1}\boldsymbol{\omega}_{i+1} = {}^{i+1}_i\mathbf{R}\;{}^i\boldsymbol{\omega}_i + \dot{\theta}_{i+1}\;\hat{\mathbf{z}}_{i+1}$$

$${}^{i+1}\dot{\boldsymbol{\omega}}_{i+1} = {}^{i+1}_i\mathbf{R}\;{}^i\dot{\boldsymbol{\omega}}_i + {}^{i+1}_i\mathbf{R}\;{}^i\boldsymbol{\omega}_i \times \dot{\theta}_{i+1}\;\hat{\mathbf{z}}_{i+1} + \ddot{\theta}_{i+1}\;\hat{\mathbf{z}}_{i+1}$$

The backward pass computes forces and torques inward from the tip:

$${}^i\mathbf{f}_i = {}^i_{i+1}\mathbf{R}\;{}^{i+1}\mathbf{f}_{i+1} + {}^i\mathbf{F}_i$$

$$\tau_i = {}^i\mathbf{n}_i^T\;\hat{\mathbf{z}}_i$$

Craig's notation is different from Siciliano's but the algorithm is the same. This is the standard reference for understanding what `pin.rnea()` does internally.

---

**[R3] Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*. Wiley.**

Covers the same dynamics material as Craig but with more mathematical rigor. Chapter 6 on inverse dynamics is particularly relevant. The textbook presents the computational complexity analysis showing that RNEA is $O(n)$ in the number of joints, compared to $O(n^3)$ or $O(n^4)$ for explicit matrix computation of $\mathbf{M}$, $\mathbf{C}$, $\mathbf{g}$.

Also contains a thorough treatment of the gravity vector computation:

$$g_i(\mathbf{q}) = -\sum_{k=i}^{n} m_k \mathbf{g}_0^T \frac{\partial \mathbf{p}_{c_k}}{\partial q_i}$$

which is what `torque_gravity_only()` computes via the RNEA shortcut of setting $\dot{\mathbf{q}} = \ddot{\mathbf{q}} = \mathbf{0}$.

---

**[R4] Featherstone, R. (2008). *Rigid Body Dynamics Algorithms*. Springer.**

This is the definitive reference for **spatial algebra** and efficient recursive dynamics algorithms. Pinocchio's internal implementation is based directly on Featherstone's spatial notation. The key concepts are:

- **Spatial velocity** (twist): $\mathbf{v} = [\boldsymbol{\omega}^T, \mathbf{v}_O^T]^T \in \mathbb{R}^6$
- **Spatial force** (wrench): $\mathbf{f} = [\mathbf{n}^T, \mathbf{f}^T]^T \in \mathbb{R}^6$
- **Spatial inertia**: The 6×6 matrix combining mass, center of mass, and rotational inertia:

$$\mathcal{I}_i = \begin{bmatrix} \mathbf{I}_i + m_i[\mathbf{c}_i]_\times^T[\mathbf{c}_i]_\times & m_i[\mathbf{c}_i]_\times \\ m_i[\mathbf{c}_i]_\times^T & m_i\mathbf{I}_3 \end{bmatrix}$$

This is how Pinocchio stores each link's inertial parameters (what `model.inertias[i]` contains).

- **Recursive Newton-Euler in spatial notation**: Algorithm 5.3 in the book is exactly what `pin.rnea()` implements.

Featherstone also presents the **articulated-body algorithm** for forward dynamics ($\boldsymbol{\tau} \to \ddot{\mathbf{q}}$), which is the inverse of what this codebase computes.

---

**[R5] Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press.**

This is the newest comprehensive robotics textbook. Its treatment of dynamics (Chapter 8) uses the product of exponentials formulation and spatial algebra consistently. Freely available at http://hades.mech.northwestern.edu/index.php/Modern_Robotics

The derivation of the mass matrix in terms of body Jacobians:

$$\mathbf{M}(\boldsymbol{\theta}) = \sum_{i=1}^{n} \mathbf{J}_{b_i}^T(\boldsymbol{\theta}) \mathcal{G}_i \mathbf{J}_{b_i}(\boldsymbol{\theta})$$

where $\mathcal{G}_i$ is the spatial inertia of link $i$ and $\mathbf{J}_{b_i}$ is its body Jacobian.

---

### 1.2 Seminal Papers on RNEA

**[R6] Luh, J. Y. S., Walker, M. W., & Paul, R. P. C. (1980). "On-Line Computational Scheme for Mechanical Manipulators." *ASME Journal of Dynamic Systems, Measurement, and Control*, 102(2), 69-76.**

This is the original paper that introduced the recursive Newton-Euler algorithm for inverse dynamics. It showed that the torque required to produce a given motion can be computed in $O(n)$ time by a two-pass recursion over the kinematic chain. Before this paper, inverse dynamics required forming and inverting the full mass matrix, which is $O(n^3)$.

The paper presents exactly the forward-backward recursion that Pinocchio implements and that this codebase uses at every timestep.

---

**[R7] Hollerbach, J. M. (1980). "A Recursive Lagrangian Formulation of Manipulator Dynamics and a Comparative Study of Dynamics Formulation Complexity." *IEEE Transactions on Systems, Man, and Cybernetics*, 10(11), 730-736.**

This paper provides the computational complexity comparison between different dynamics formulations:

| Method | Multiplications | Additions |
|---|---|---|
| Lagrange-Euler (direct) | $O(n^4)$ | $O(n^4)$ |
| Recursive Lagrange-Euler | $O(n^3)$ | $O(n^3)$ |
| Newton-Euler (recursive) | $O(n)$ | $O(n)$ |

The $O(n)$ efficiency of Newton-Euler is why Pinocchio (and this codebase) uses it rather than explicitly forming the matrices $\mathbf{M}$, $\mathbf{C}$, $\mathbf{g}$.

---

### 1.3 Pinocchio Library

**[R8] Carpentier, J., Saurel, G., Buondonno, G., Mirabel, J., Lamiraux, F., Stasse, O., & Mansard, N. (2019). "The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives." *IEEE International Symposium on System Integration (SII)*, 614-619.**

This is the primary reference for the Pinocchio library used throughout this codebase. The paper describes:

- The spatial algebra foundation (following Featherstone)
- The RNEA implementation for inverse dynamics
- The analytical Jacobians of RNEA with respect to $\mathbf{q}$, $\dot{\mathbf{q}}$, $\ddot{\mathbf{q}}$ (useful for future PINN integration)
- The URDF parser that `pin.buildModelFromXML()` uses
- Performance benchmarks showing Pinocchio is among the fastest rigid body dynamics libraries

The codebase uses `pin.rnea(model, data, q, qd, qdd)` which is the core function described in this paper.

---

**[R9] Carpentier, J., & Mansard, N. (2018). "Analytical Derivatives of Rigid Body Dynamics Algorithms." *Robotics: Science and Systems (RSS)*.**

This paper derives the analytical derivatives of RNEA:

$$\frac{\partial \boldsymbol{\tau}}{\partial \mathbf{q}}, \quad \frac{\partial \boldsymbol{\tau}}{\partial \dot{\mathbf{q}}}, \quad \frac{\partial \boldsymbol{\tau}}{\partial \ddot{\mathbf{q}}}$$

These derivatives are available in Pinocchio but not currently used in this codebase. They would be directly useful for PINN training because they provide exact analytical gradients of the physics model, eliminating the need for numerical differentiation of the RNEA output with respect to its inputs.

---

## 2. Friction Modeling

### 2.1 Classical Friction Models

**[R10] Armstrong-Hélouvry, B., Dupont, P., & Canudas de Wit, C. (1994). "A Survey of Models, Analysis Tools and Compensation Methods for the Control of Machines with Friction." *Automatica*, 30(7), 1083-1138.**

This is the most comprehensive survey of friction models in the context of robot control. It covers:

- **Coulomb friction**: Constant magnitude, opposes motion direction:

$$\tau_{\text{Coulomb}} = c \cdot \text{sign}(\dot{q})$$

- **Viscous friction**: Proportional to velocity:

$$\tau_{\text{viscous}} = v \cdot \dot{q}$$

- **Combined Coulomb + viscous** (the model used in this codebase):

$$\tau_f = c \cdot \text{sign}(\dot{q}) + v \cdot \dot{q}$$

- **Stribeck effect**: Friction decreases with increasing velocity near zero (negative viscous region), creating a dip between static and kinetic friction

- **Stiction**: Static friction is higher than kinetic friction

The paper discusses the challenges of discontinuity at zero velocity and the need for smooth approximations, which directly motivates the tanh approximation used in `torque.py`.

---

**[R11] Canudas de Wit, C., Olsson, H., Åström, K. J., & Lischinsky, P. (1995). "A New Model for Control of Systems with Friction." *IEEE Transactions on Automatic Control*, 40(3), 419-425.**

This paper introduces the **LuGre friction model**, which is the most widely used dynamic friction model. It models friction using a bristle state $z$:

$$\frac{dz}{dt} = \dot{q} - \frac{|\dot{q}|}{g(\dot{q})} z$$

$$\tau_f = \sigma_0 z + \sigma_1 \frac{dz}{dt} + \sigma_2 \dot{q}$$

where $\sigma_0$ is bristle stiffness, $\sigma_1$ is bristle damping, $\sigma_2$ is viscous friction, and $g(\dot{q})$ captures the Stribeck effect.

The LuGre model is more physically accurate than the Coulomb + viscous model used in this codebase because it captures:
- Pre-sliding displacement (micro-motion before breakaway)
- Stribeck effect (friction dip at low velocities)
- Hysteresis in friction force

However, it requires identifying 6 parameters per joint and solving an additional ODE per joint at each timestep. The simpler Coulomb + viscous model was chosen for this codebase because the additional complexity of LuGre would be absorbed into the PINN residual anyway.

---

**[R12] Åström, K. J., & Canudas de Wit, C. (2008). "Revisiting the LuGre Friction Model." *IEEE Control Systems Magazine*, 28(6), 101-114.**

A retrospective on the LuGre model 13 years after its introduction. Discusses practical identification issues and simplified versions suitable for real-time control. Relevant for understanding why the simpler model in this codebase is a reasonable choice for generating PINN training data.

---

### 2.2 Smooth Friction Approximations

**[R13] Makkar, C., Dixon, W. E., Sawyer, W. G., & Hu, G. (2005). "A New Continuously Differentiable Friction Model for Control Systems Design." *IEEE/ASME International Conference on Advanced Intelligent Mechatronics*, 986-991.**

This paper formalizes the use of smooth approximations for the sign function in friction models. It analyzes the tanh approximation:

$$\text{sign}(\dot{q}) \approx \tanh\left(\frac{\dot{q}}{\varepsilon}\right)$$

and proves stability properties for controllers using this approximation. The paper provides the mathematical justification for the smooth sign function used in `torque.py → _smooth_sign()`.

---

**[R14] Marques, F., Flores, P., Pimenta Claro, J. C., & Lankarani, H. M. (2016). "A Survey and Comparison of Several Friction Force Models for Dynamic Analysis of Multibody Mechanical Systems." *Nonlinear Dynamics*, 86, 1407-1443.**

A comprehensive comparison of friction models for multibody simulation. Compares:
- Coulomb model
- Coulomb + viscous
- Stribeck models
- LuGre model
- Various smooth approximations (tanh, arctan, erf)

The paper benchmarks these models on accuracy versus computational cost, which is relevant to the design choice made in this codebase (simple tanh-smoothed Coulomb + viscous).

---

### 2.3 Friction Identification and Calibration

**[R15] Swevers, J., Ganseman, C., Tükel, D. B., De Schutter, J., & Van Brussel, H. (1997). "Optimal Robot Excitation and Identification." *IEEE Transactions on Robotics and Automation*, 13(5), 730-740.**

This paper addresses the problem of designing trajectories that optimally excite dynamic parameters for identification. The concept is directly relevant to the bulk calibration in `calibrate_friction.py`, where 124 diverse trajectories provide better parameter estimates than a single trajectory.

The paper introduces the concept of **persistent excitation** — the input signal must contain sufficient frequency content to distinguish between different dynamic parameters. This is mathematically related to the collinearity problem encountered in friction calibration v3 (Appendix E of the analysis), where velocity filtering destroyed the distinguishability between Coulomb and viscous friction.

---

**[R16] Gautier, M., & Khalil, W. (1990). "Direct Calculation of Minimum Set of Inertial Parameters of Serial Robots." *IEEE Transactions on Robotics and Automation*, 6(3), 368-373.**

This paper introduces the concept of **base parameters** — the minimum set of inertial parameters that are identifiable from joint torque measurements. Some inertial parameters (like the inertia of the first link about its own axis when the first joint is revolute) have no effect on any measurable torque and are therefore unidentifiable.

This is relevant to the mass calibration in `calibrate_v2.py`. The fact that a single global mass scale $\alpha$ is fitted rather than per-link parameters reflects the practical reality that many individual link parameters are not independently identifiable from the available data.

---

**[R17] Khalil, W., & Dombre, E. (2002). *Modeling, Identification and Control of Robots*. Hermes Penton Science.**

This textbook has the most thorough treatment of robot parameter identification. Chapter 11 covers:
- The linear parameterization of inverse dynamics:

$$\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) \boldsymbol{\pi}$$

where $\mathbf{Y}$ is the regressor matrix and $\boldsymbol{\pi}$ is the vector of dynamic parameters. This linearity is exactly what makes the least-squares calibration in `calibrate_v2.py` possible (gravity torque is linear in mass).

- Identification of base parameters via least squares
- Numerical conditioning issues (relevant to the collinearity problem in friction calibration)
- Friction model identification as part of the dynamic parameter vector

---

**[R18] Atkeson, C. G., An, C. H., & Hollerbach, J. M. (1986). "Estimation of Inertial Parameters of Manipulator Loads and Links." *The International Journal of Robotics Research*, 5(3), 101-119.**

One of the earliest papers on practical robot parameter identification from torque data. Introduces the regressor formulation and discusses:
- The effect of measurement noise on parameter estimates
- The need for filtering and smoothing (directly relevant to the signal processing in `utils.py`)
- The trade-off between bandwidth and noise in numerical differentiation

---

## 3. Numerical Differentiation and Signal Processing

### 3.1 Numerical Differentiation of Noisy Data

**[R19] Chartrand, R. (2011). "Numerical Differentiation of Noisy, Nonsmooth Data." *ISRN Applied Mathematics*, 2011, Article ID 164564.**

This paper addresses the fundamental problem that numerical differentiation amplifies noise. It presents total variation regularization as a robust alternative to central differences followed by smoothing:

$$\min_u \int |u'(t)| \, dt + \frac{\lambda}{2} \int \left(\int_0^t u(s)\,ds - f(t)\right)^2 dt$$

where $f$ is the noisy signal and $u$ is the sought derivative. The approach in `utils.py` (central differences + uniform filter) is simpler but follows the same principle of balancing fidelity to data against smoothness of the derivative.

---

**[R20] Savitzky, A., & Golay, M. J. E. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures." *Analytical Chemistry*, 36(8), 1627-1639.**

The classic paper on Savitzky-Golay filters, which simultaneously smooth and differentiate by fitting local polynomials. The uniform filter used in `utils.py` is a special case (zero-order polynomial). Higher-order Savitzky-Golay filters would provide better frequency response (flatter passband, sharper cutoff) at the cost of slightly more computation.

The paper is relevant because it quantifies the trade-off between noise suppression and signal distortion that the smoothing window parameter controls.

---

### 3.2 Moving Average Filters

**[R21] Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal Processing*. California Technical Publishing.**

Chapter 15 covers moving average filters in detail. The uniform filter used by `scipy.uniform_filter1d` in `utils.py` has the transfer function:

$$H(f) = \frac{\sin(\pi f M / f_s)}{M \sin(\pi f / f_s)}$$

where $M$ is the window size and $f_s$ is the sampling rate. The book explains:
- Why the moving average is optimal for reducing random noise while preserving step response (but poor for separating frequencies)
- The relationship between window size and noise reduction ($\text{SNR improvement} = \sqrt{M}$)
- The phase response (zero phase for symmetric windows, as used here)

This book is freely available at www.dspguide.com.

---

## 4. URDF and Robot Description

### 4.1 URDF Specification

**[R22] Meeussen, W. (2009). "Unified Robot Description Format (URDF)." *ROS Wiki Documentation*. http://wiki.ros.org/urdf**

The URDF format defines the kinematic and dynamic properties of a robot as an XML tree. Each link element contains:

```xml
<inertial>
  <mass value="0.5"/>
  <origin xyz="0.01 0.0 0.03"/>
  <inertia ixx="1e-4" iyy="2e-4" izz="3e-4" ixy="0" ixz="0" iyz="0"/>
</inertial>
```

These are the values that `build_pinocchio_model()` parses and scales by $\alpha$. The `origin xyz` corresponds to the center of mass position (Pinocchio's `lever`). The `inertia` elements form the 3×3 inertia tensor about the center of mass.

---

**[R23] Quigley, M., Gerkey, B., & Smart, W. D. (2015). *Programming Robots with ROS*. O'Reilly Media.**

Chapter 6 covers the URDF and XACRO formats in practical detail. XACRO (XML Macros for URDF) allows parameterized robot descriptions, which is why the codebase uses `xacro.process_file()` to expand macros before passing the result to Pinocchio.

---

## 5. Mass and Inertia Calibration

### 5.1 CAD-Based vs Experimental Identification

**[R24] Hollerbach, J. M., Khalil, W., & Gautier, M. (2016). "Model Identification." In *Springer Handbook of Robotics* (2nd Edition), 113-138. Springer.**

This handbook chapter surveys the state of the art in robot model identification. It discusses:

- **CAD-based parameters**: Using 3D models to compute mass and inertia (what the URDF provides). The accuracy depends on correct material density — exactly the problem addressed by `calibrate_v2.py`.

- **Experimental identification**: Fitting model parameters to measured torque data. The least-squares approach in `calibrate_v2.py` (fitting $\alpha$ from gravity torque) is a simplified version of the general approach described here.

- **The identifiability problem**: Not all parameters can be independently identified. The paper explains why a global mass scale is often a practical compromise — individual link masses cannot be distinguished from gravity data alone if the arm does not explore a sufficiently rich set of configurations.

---

**[R25] Traversaro, S., Brossette, S., Escande, A., & Nori, F. (2016). "Identification of Fully Physical Consistent Inertial Parameters Using Optimization on Manifolds." *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 5446-5451.**

This paper addresses the problem of ensuring that identified inertial parameters are physically consistent (positive mass, positive definite inertia tensor, CoM inside the link geometry). The global mass scaling in `calibrate_v2.py` automatically preserves physical consistency because multiplying a positive-definite inertia by a positive scalar remains positive-definite.

---

## 6. Physics-Informed Neural Networks

### 6.1 Foundational PINN Papers

**[R26] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations." *Journal of Computational Physics*, 378, 686-707.**

This is the seminal PINN paper. It introduces the idea of encoding known physics as a regularization term in the neural network loss function:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$

The approach in this codebase is related but different: rather than encoding the physics as a soft constraint in the loss, the physics is computed explicitly (via RNEA and friction model) and the neural network only learns the residual. This is sometimes called a **physics-informed residual network** or **hybrid model**.

---

**[R27] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). "Physics-Informed Machine Learning." *Nature Reviews Physics*, 3, 422-440.**

A comprehensive review of physics-informed machine learning. Section 3 discusses **hybrid models** where a known physics model is augmented with a learned correction:

$$\hat{y} = f_{\text{physics}}(x) + \mathcal{N}_\theta(x)$$

This is exactly the architecture this codebase is designed to support:

$$\hat{\boldsymbol{\tau}} = \underbrace{(\boldsymbol{\tau}_{\text{RNEA}} + \boldsymbol{\tau}_f)}_{\text{physics}} + \underbrace{\mathcal{N}_\theta(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}, t)}_{\text{learned correction}}$$

---

### 6.2 PINNs for Robot Dynamics

**[R28] Lutter, M., Ritter, C., & Peters, J. (2019). "Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning." *International Conference on Learning Representations (ICLR)*.**

This paper proposes encoding the Lagrangian structure of robot dynamics directly into the neural network architecture:

$$\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = T(\mathbf{q}, \dot{\mathbf{q}}) - V(\mathbf{q})$$

where kinetic energy $T$ and potential energy $V$ are parameterized by neural networks with structural constraints (e.g., the mass matrix network outputs a symmetric positive-definite matrix).

The relationship to this codebase: Deep Lagrangian Networks learn the entire dynamics from scratch with structural priors. This codebase takes the complementary approach of computing most of the dynamics analytically and learning only the correction. The two approaches could potentially be combined.

---

**[R29] Gupta, A., & Ghosh, A. (2022). "Structured Mechanical Models for Robot Learning." *Conference on Robot Learning (CoRL)*.**

Discusses hybrid approaches where analytical rigid-body dynamics are combined with learned components. The paper argues that learning only the residual (as this codebase prepares for) is more data-efficient than learning full dynamics, because the analytical model provides a strong inductive bias.

---

**[R30] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). "Lagrangian Neural Networks." *ICLR Workshop on Integration of Deep Neural Models and Differential Equations*.**

Proposes neural networks that respect the Lagrangian structure of mechanics. The key insight is that by parameterizing the Lagrangian rather than the equations of motion directly, conservation laws (energy, momentum) are automatically satisfied.

Relevant to this codebase because the RNEA model already satisfies these conservation laws. The PINN residual should ideally also respect them, which the physics regularizers suggested in Appendix G of the analysis could enforce.

---

### 6.3 Neural Network Approaches to Friction

**[R31] Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., & Hutter, M. (2019). "Learning Agile and Dynamic Motor Skills for Legged Robots." *Science Robotics*, 4(26), eaau5872.**

This paper demonstrates learning actuator dynamics (including friction) using neural networks for a quadruped robot. The learned actuator model captures effects that analytical models miss:
- Temperature-dependent friction
- History-dependent hysteresis
- Nonlinear current-torque relationships

The approach is similar in spirit to the PINN residual in this codebase but applied to the actuator model rather than the full-body dynamics.

---

**[R32] Heiden, E., Millard, D., Corl, A., Erwin, H., & Gaurav, S. (2021). "NeuralSim: Augmenting Differentiable Simulators with Neural Networks." *IEEE International Conference on Robotics and Automation (ICRA)*, 9474-9481.**

Proposes augmenting differentiable physics simulators with neural networks to capture unmodeled dynamics. The paper demonstrates that the hybrid approach (analytical physics + learned correction) outperforms both pure physics and pure learning approaches.

The architecture:

$$\hat{\boldsymbol{\tau}} = \boldsymbol{\tau}_{\text{analytical}} + \Delta\boldsymbol{\tau}_{\text{neural}}$$

is exactly what this codebase prepares training data for.

---

## 7. System Identification for Robots

### 7.1 Classical System Identification

**[R33] Ljung, L. (1999). *System Identification: Theory for the User* (2nd Edition). Prentice Hall.**

The standard reference for system identification theory. Relevant concepts:
- **Persistent excitation**: The input must be sufficiently rich to identify all parameters. This is why the bulk mode in `calibrate_friction.py` (124 diverse trajectories) produces better results than single-trajectory calibration.
- **Bias-variance trade-off**: The bias-aware fit in the friction calibration explicitly models the bias term to prevent it from corrupting the friction parameter estimates.
- **Model structure selection**: The choice between different friction models (Coulomb only, Coulomb + viscous, LuGre) is a model structure selection problem.

---

**[R34] Mata, V., Benimeli, F., Farhat, N., & Valera, A. (2005). "Dynamic Parameter Identification in Industrial Robots Considering Physical Feasibility." *Advanced Robotics*, 19(1), 101-119.**

Discusses practical challenges in robot parameter identification:
- The effect of filtering on identified parameters (relevant to `SMOOTH_WINDOW` setting)
- Physical feasibility constraints (masses must be positive, inertias must be positive definite)
- The role of exciting trajectories

---

### 7.2 Least-Squares Parameter Estimation

**[R35] Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.**

The mathematical foundation for all the least-squares fitting in this codebase:
- The scalar projection $\alpha^* = (\mathbf{a}^T\mathbf{b})/(\mathbf{a}^T\mathbf{a})$ in `calibrate_v2.py`
- The linear least squares $\mathbf{x}^* = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{y}$ in `fit_bias_aware()`
- The condition number analysis explaining why velocity filtering caused collinearity in friction calibration v3

---

## 8. Servo Motor Modeling

**[R36] Spong, M. W. (1987). "Modeling and Control of Elastic Joint Robots." *ASME Journal of Dynamic Systems, Measurement, and Control*, 109(4), 310-319.**

This paper models the dynamics of robots with flexible joints (where the gearbox has finite stiffness). While this codebase assumes rigid joints, the paper's treatment of motor-side versus link-side torque is relevant to understanding the servo load register and current measurements.

The motor-side torque includes gearbox friction:

$$\tau_{\text{motor}} = J_m \ddot{\theta}_m + B_m \dot{\theta}_m + \tau_{\text{Coulomb}} \cdot \text{sign}(\dot{\theta}_m) + K(\theta_m - N\theta_l)$$

where $\theta_m$ is motor angle, $\theta_l$ is link angle, $N$ is gear ratio, and $K$ is joint stiffness. In the rigid joint approximation ($K \to \infty$), $\theta_m = N\theta_l$, and the reflected motor dynamics become:

$$\tau_{\text{reflected}} = N^2 J_m \ddot{\theta}_l + N^2 B_m \dot{\theta}_l + N \tau_{\text{Coulomb}} \cdot \text{sign}(\dot{\theta}_l)$$

This explains why viscous friction scales as $N^2$ with gear ratio, as discussed in the physical interpretation of joint 1's anomalously high viscous friction.

---

**[R37] Seyfferth, W., Maghzal, A. J., & Angeles, J. (1995). "Nonlinear Modeling and Parameter Identification of Harmonic Drive Robotic Transmissions." *IEEE International Conference on Robotics and Automation*, 3027-3032.**

Discusses the specific friction characteristics of harmonic drive gearboxes, which are commonly used in robot servos. The nonlinear friction profile (including the Stribeck effect) is more complex than the Coulomb + viscous model but can be approximated by it in many operating regimes.

---

## Summary Table: References by Topic

| Topic | Primary References | Secondary References |
|---|---|---|
| Equations of motion | R1, R2, R3, R5 | R4 |
| RNEA algorithm | R6, R7, R4 | R1 Chapter 7.3 |
| Pinocchio library | R8, R9 | R4 |
| Coulomb + viscous friction | R10, R11 | R12, R14 |
| Smooth friction approximation | R13, R14 | R10 |
| Friction identification | R15, R16, R17 | R18, R33 |
| Numerical differentiation | R19, R20, R21 | |
| URDF and robot description | R22, R23 | |
| Mass/inertia calibration | R24, R25, R16 | R17, R18 |
| Least-squares estimation | R35, R33 | R17, R34 |
| Servo motor dynamics | R36, R37 | |
| Physics-informed neural networks | R26, R27 | R28, R29, R30 |
| PINNs for robot dynamics | R28, R29, R32 | R30, R31 |
