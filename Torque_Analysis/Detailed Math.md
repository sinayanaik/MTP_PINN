# Linear & Angular Jacobians — Full Clarification with Example

## First: The Key Distinction You Need

There are **two different contexts** where Jacobians appear:

| Context | Jacobian is for... | Used in... |
|---|---|---|
| **Task-space control** | End-effector only | $\mathbf{v}_{ee} = \mathbf{J}\,\dot{\mathbf{q}}$ |
| **Inertia matrix $\mathbf{M}(\mathbf{q})$** | **Every link's Center of Mass** | Kinetic energy of each link |

> **Your statement is half-right:** Yes, $\mathbf{v}$ and $\boldsymbol{\omega}$ are task-space velocities. But for the inertia matrix, we don't just care about the end-effector — we need the velocity of **every link's CoM**.

---

## Setup: 2R Planar Robot (Concrete Example)

```
        l_c1        l_c2
    ●━━━━◆━━━━━●━━━━◆━━━━━● → end-effector
  Joint 1  CoM₁  Joint 2  CoM₂
    (q₁)          (q₂)
```

### Variable Definitions (every single one):

| Variable | Meaning | Example Value |
|---|---|---|
| $q_1$ | Angle of joint 1 (from x-axis) | — |
| $q_2$ | Angle of joint 2 (relative to link 1) | — |
| $\dot{q}_1, \dot{q}_2$ | Joint angular velocities | — |
| $l_1$ | Full length of link 1 | 1.0 m |
| $l_2$ | Full length of link 2 | 1.0 m |
| $l_{c1}$ | Distance from joint 1 to CoM of link 1 | 0.5 m |
| $l_{c2}$ | Distance from joint 2 to CoM of link 2 | 0.5 m |
| $m_1, m_2$ | Masses of link 1, link 2 | 2 kg, 1 kg |
| $I_1, I_2$ | Moments of inertia about each link's CoM (about z-axis) | scalar for planar |
| $\hat{\mathbf{z}}_0$ | Unit vector along z-axis $= \begin{bmatrix} 0\\0\\1 \end{bmatrix}$ | (rotation axis for both joints) |

---

## Part 1: End-Effector Jacobian (What You Already Know)

### Position of End-Effector

$$\mathbf{p}_{ee} = \begin{bmatrix} l_1\cos q_1 + l_2\cos(q_1+q_2) \\ l_1\sin q_1 + l_2\sin(q_1+q_2) \\ 0 \end{bmatrix}$$

### End-Effector Velocity

$$\mathbf{v}_{ee} = \frac{d\,\mathbf{p}_{ee}}{dt} = \underbrace{\frac{\partial \mathbf{p}_{ee}}{\partial q_1}}_{}\dot{q}_1 + \underbrace{\frac{\partial \mathbf{p}_{ee}}{\partial q_2}}_{}\dot{q}_2$$

$$\mathbf{v}_{ee} = \underbrace{\begin{bmatrix} -l_1 s_1 - l_2 s_{12} & -l_2 s_{12}\\ l_1 c_1 + l_2 c_{12} & l_2 c_{12}\\ 0 & 0 \end{bmatrix}}_{\mathbf{J}_v^{ee}\;(3\times 2)} \begin{bmatrix}\dot{q}_1\\\dot{q}_2\end{bmatrix}$$

where $s_1 = \sin q_1$, $c_1 = \cos q_1$, $s_{12} = \sin(q_1+q_2)$, $c_{12} = \cos(q_1+q_2)$.

### Angular Velocity of End-Effector

The end-effector is rigidly attached to link 2, so it spins at:

$$\boldsymbol{\omega}_{ee} = (\dot{q}_1 + \dot{q}_2)\,\hat{\mathbf{z}}_0 = \underbrace{\begin{bmatrix} 0 & 0\\ 0 & 0\\ 1 & 1 \end{bmatrix}}_{\mathbf{J}_\omega^{ee}\;(3\times 2)} \begin{bmatrix}\dot{q}_1\\\dot{q}_2\end{bmatrix}$$

### Full End-Effector Jacobian

$$\begin{bmatrix}\mathbf{v}_{ee}\\\boldsymbol{\omega}_{ee}\end{bmatrix}_{6\times 1} = \begin{bmatrix}\mathbf{J}_v^{ee}\\\mathbf{J}_\omega^{ee}\end{bmatrix}_{6\times 2} \dot{\mathbf{q}}_{2\times 1}$$

**This is for control/task-space mapping. Now let's move to what the inertia matrix needs.**

---

## Part 2: Jacobians of Each Link's CoM (For Inertia Matrix)

### Link 1's CoM Position

$$\mathbf{p}_{c1} = \begin{bmatrix} l_{c1}\cos q_1 \\ l_{c1}\sin q_1 \\ 0\end{bmatrix}$$

### Link 1's CoM Velocity (Linear Jacobian)

$$\mathbf{v}_{c1} = \frac{d\,\mathbf{p}_{c1}}{dt} = \underbrace{\begin{bmatrix} -l_{c1}\,s_1 & 0\\ l_{c1}\,c_1 & 0\\ 0 & 0 \end{bmatrix}}_{\mathbf{J}_{v_1}\;(3\times2)} \begin{bmatrix}\dot{q}_1\\\dot{q}_2\end{bmatrix}$$

> ⚠️ **Second column is zero!** Joint 2 is *distal* to link 1, so moving joint 2 **cannot** move link 1's CoM.

### Link 1's Angular Velocity (Angular Jacobian)

Link 1 only rotates due to joint 1:

$$\boldsymbol{\omega}_1 = \dot{q}_1\,\hat{\mathbf{z}}_0 = \underbrace{\begin{bmatrix} 0 & 0\\ 0 & 0\\ 1 & 0 \end{bmatrix}}_{\mathbf{J}_{\omega_1}\;(3\times2)} \begin{bmatrix}\dot{q}_1\\\dot{q}_2\end{bmatrix}$$

> ⚠️ Again, second column is zero — joint 2 **cannot** spin link 1.

---

### Link 2's CoM Position

$$\mathbf{p}_{c2} = \begin{bmatrix} l_1\cos q_1 + l_{c2}\cos(q_1+q_2) \\ l_1\sin q_1 + l_{c2}\sin(q_1+q_2) \\ 0\end{bmatrix}$$

### Link 2's CoM Velocity (Linear Jacobian)

$$\mathbf{v}_{c2} = \underbrace{\begin{bmatrix} -l_1 s_1 - l_{c2}\,s_{12} & -l_{c2}\,s_{12}\\ l_1 c_1 + l_{c2}\,c_{12} & l_{c2}\,c_{12}\\ 0 & 0 \end{bmatrix}}_{\mathbf{J}_{v_2}\;(3\times2)} \begin{bmatrix}\dot{q}_1\\\dot{q}_2\end{bmatrix}$$

> ✅ **Both columns nonzero** — both joints affect link 2's CoM.

### Link 2's Angular Velocity (Angular Jacobian)

$$\boldsymbol{\omega}_2 = (\dot{q}_1+\dot{q}_2)\,\hat{\mathbf{z}}_0 = \underbrace{\begin{bmatrix} 0 & 0\\ 0 & 0\\ 1 & 1 \end{bmatrix}}_{\mathbf{J}_{\omega_2}\;(3\times2)} \begin{bmatrix}\dot{q}_1\\\dot{q}_2\end{bmatrix}$$

---

## Part 3: Building $\mathbf{M}(\mathbf{q})$ from These Jacobians

### Formula

$$\mathbf{M}(\mathbf{q}) = \sum_{k=1}^{2}\Big[m_k\,\mathbf{J}_{v_k}^T\mathbf{J}_{v_k} + \mathbf{J}_{\omega_k}^T\,\bar{\mathbf{I}}_k\,\mathbf{J}_{\omega_k}\Big]$$

where $\bar{\mathbf{I}}_k = {}^0\mathbf{R}_k\,\mathbf{I}_k\,{}^0\mathbf{R}_k^T$ (for planar case, just $I_k$ about z-axis).

### Contribution from Link 1 ($k=1$)

$$m_1\,\mathbf{J}_{v_1}^T\mathbf{J}_{v_1} = m_1\begin{bmatrix} l_{c1}^2 & 0\\0 & 0\end{bmatrix}$$

$$\mathbf{J}_{\omega_1}^T\,I_1\,\mathbf{J}_{\omega_1} = I_1\begin{bmatrix} 1 & 0\\0 & 0\end{bmatrix}$$

### Contribution from Link 2 ($k=2$)

$$m_2\,\mathbf{J}_{v_2}^T\mathbf{J}_{v_2} = m_2\begin{bmatrix} l_1^2+l_{c2}^2+2l_1 l_{c2}c_2 & l_{c2}^2+l_1 l_{c2}c_2\\ l_{c2}^2+l_1 l_{c2}c_2 & l_{c2}^2\end{bmatrix}$$

$$\mathbf{J}_{\omega_2}^T\,I_2\,\mathbf{J}_{\omega_2} = I_2\begin{bmatrix} 1 & 1\\1 & 1\end{bmatrix}$$

### Final Inertia Matrix

$$\boxed{\mathbf{M}(\mathbf{q}) = \begin{bmatrix} m_1 l_{c1}^2 + m_2(l_1^2+l_{c2}^2+2l_1 l_{c2}c_2) + I_1+I_2 & m_2(l_{c2}^2+l_1 l_{c2}c_2)+I_2 \\ m_2(l_{c2}^2+l_1 l_{c2}c_2)+I_2 & m_2 l_{c2}^2+I_2 \end{bmatrix}}$$

---

## Summary of the Key Distinction

```
End-Effector Jacobian          vs.     Link CoM Jacobians
━━━━━━━━━━━━━━━━━━━━                   ━━━━━━━━━━━━━━━━━━
ONE Jacobian                           ONE Jacobian PER LINK
Maps q̇ → tip velocity                 Maps q̇ → each link's CoM velocity
Used for: control, IK                  Used for: kinetic energy → M(q)
                                       Distal joints give zero columns!
```

**The punchline:** $\mathbf{J}_v$ and $\mathbf{J}_\omega$ are not inverses — they are the **linear and angular parts** of the same Jacobian, and for the inertia matrix, you need them **for every link**, not just the end-effector.

# How the Inertia Matrix $\mathbf{M}(\mathbf{q})$ is Derived

The inertia matrix emerges directly from the **total kinetic energy** of the robot. Here's the step-by-step logic:

---

## Step 1: Kinetic Energy of a Single Link

Each link $k$ has translational + rotational kinetic energy:

$$T_k = \underbrace{\frac{1}{2}\, m_k\, \mathbf{v}_{c_k}^T \mathbf{v}_{c_k}}_{\text{translational}} + \underbrace{\frac{1}{2}\, \boldsymbol{\omega}_k^T\, \bar{\mathbf{I}}_k\, \boldsymbol{\omega}_k}_{\text{rotational}}$$

where $\bar{\mathbf{I}}_k = {}^{0}\mathbf{R}_k\, \mathbf{I}_k\, {}^{0}\mathbf{R}_k^T$ is the inertia tensor rotated into the world frame.

---

## Step 2: Express Velocities via Jacobians

In a serial chain, the velocity of link $k$'s center of mass depends on joints $1$ through $k$:

$$\mathbf{v}_{c_k} = \mathbf{J}_{v_k}(\mathbf{q})\,\dot{\mathbf{q}}, \qquad \boldsymbol{\omega}_k = \mathbf{J}_{\omega_k}(\mathbf{q})\,\dot{\mathbf{q}}$$

> **Key point:** Columns $j > k$ of $\mathbf{J}_{v_k}$ and $\mathbf{J}_{\omega_k}$ are **zero**, because joint $j$ (distal to link $k$) cannot move link $k$.

---

## Step 3: Substitute and Factor

Substituting into $T_k$:

$$T_k = \frac{1}{2}\,\dot{\mathbf{q}}^T \Big[\, m_k\, \mathbf{J}_{v_k}^T \mathbf{J}_{v_k} \;+\; \mathbf{J}_{\omega_k}^T\, \bar{\mathbf{I}}_k\, \mathbf{J}_{\omega_k} \,\Big]\, \dot{\mathbf{q}}$$

---

## Step 4: Sum Over All Links

Total kinetic energy:

$$\boxed{T = \frac{1}{2}\,\dot{\mathbf{q}}^T\, \mathbf{M}(\mathbf{q})\, \dot{\mathbf{q}}}$$

where we **identify** the inertia matrix as:

$$\mathbf{M}(\mathbf{q}) = \sum_{k=1}^{n} \Big[\, m_k\, \mathbf{J}_{v_k}^T \mathbf{J}_{v_k} + \mathbf{J}_{\omega_k}^T\, \bar{\mathbf{I}}_k\, \mathbf{J}_{\omega_k} \,\Big]$$

---

## Step 5: Element-wise Form and the $\max(i,j)$ Bound

Extracting the $(i,j)$ element:

$$M_{ij} = \sum_{k=1}^{n} \Big[\, m_k\, \mathbf{J}_{v_k,i}^T \mathbf{J}_{v_k,j} + \mathbf{J}_{\omega_k,i}^T\, \bar{\mathbf{I}}_k\, \mathbf{J}_{\omega_k,j} \,\Big]$$

Since $\mathbf{J}_{v_k,i} = \mathbf{0}$ whenever $i > k$ (joint $i$ can't move link $k$ if $i$ is distal), the product is **nonzero only when** $k \geq i$ **and** $k \geq j$, i.e., $k \geq \max(i,j)$. So the sum reduces to:

$$\boxed{M_{ij} = \sum_{k=\max(i,j)}^{n} \Big[\, m_k\, \mathbf{J}_{v_k,i}^T \mathbf{J}_{v_k,j} + \mathbf{J}_{\omega_k,i}^T\, \bar{\mathbf{I}}_k\, \mathbf{J}_{\omega_k,j} \,\Big]}$$

---

## Summary of the Logic Chain

```
Kinetic energy of each link
        │
        ▼
Express link velocities via Jacobians (v = J·q̇)
        │
        ▼
Substitute → get quadratic form in q̇
        │
        ▼
Sum over all links → T = ½ q̇ᵀ M(q) q̇
        │
        ▼
Read off M(q) as the coefficient matrix
        │
        ▼
Exploit serial-chain sparsity → summation starts at max(i,j)
```

**In one sentence:** The inertia matrix is simply the **coefficient matrix of the quadratic kinetic-energy expression** $T = \frac{1}{2}\dot{\mathbf{q}}^T \mathbf{M}\, \dot{\mathbf{q}}$, obtained by writing every link's velocity in terms of joint velocities via Jacobians and summing contributions from all links.

# How the Gravity Term $\mathbf{g}(\mathbf{q})$ is Derived

Again, it all comes from **Lagrangian mechanics** — specifically from the **potential energy** part.

---

## Starting Point: The Lagrangian

The Lagrangian is defined as:

$$\mathcal{L} = T - V$$

where:
| Symbol | Meaning |
|---|---|
| $T$ | Total kinetic energy (gave us $\mathbf{M}$ and $\mathbf{C}$) |
| $V$ | Total **potential energy** (will give us $\mathbf{g}$) |

The Euler-Lagrange equation for joint $i$:

$$\tau_i = \frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}_i} - \frac{\partial \mathcal{L}}{\partial q_i}$$

Since $V$ does **not** depend on $\dot{\mathbf{q}}$ (potential energy depends only on position):

$$\tau_i = \underbrace{\frac{d}{dt}\frac{\partial T}{\partial \dot{q}_i} - \frac{\partial T}{\partial q_i}}_{\text{gives } \mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} \text{ (already derived)}} + \underbrace{\frac{\partial V}{\partial q_i}}_{\text{this is the gravity term!}}$$

So:

$$\boxed{g_i(\mathbf{q}) = \frac{\partial V}{\partial q_i}}$$

That's it conceptually. Now let's build $V$ step by step.

---

## Step 1: Potential Energy of a Single Link

For link $k$ with mass $m_k$ whose center of mass is at position $\mathbf{p}_{c_k}$ in the world frame:

$$V_k = -m_k\,\mathbf{g}_0^T\,\mathbf{p}_{c_k}(\mathbf{q})$$

### Why the Negative Sign and Dot Product?

```
World frame:
                    ↑ z (up)
                    │
                    │    ◆ Link k's CoM at height h
                    │    │
                    │    │ h = z-component of p_ck
    ────────────────●──────── ground (z = 0)

    Gravity vector: g₀ = [0, 0, -9.81]ᵀ  (points DOWN)

    Potential energy = m_k × g × h
                     = m_k × 9.81 × (z-component of p_ck)
```

Let's verify with the formula:

$$V_k = -m_k\,\mathbf{g}_0^T\,\mathbf{p}_{c_k} = -m_k\begin{bmatrix}0\\0\\-9.81\end{bmatrix}^T\begin{bmatrix}x_{c_k}\\y_{c_k}\\z_{c_k}\end{bmatrix} = m_k \times 9.81 \times z_{c_k}$$

✅ This is exactly $mgh$ — the familiar gravitational potential energy!

---

## Step 2: Total Potential Energy

Sum over all links:

$$V(\mathbf{q}) = \sum_{k=1}^{n} V_k = -\sum_{k=1}^{n} m_k\,\mathbf{g}_0^T\,\mathbf{p}_{c_k}(\mathbf{q})$$

---

## Step 3: Differentiate to Get Gravity Torque

$$g_i(\mathbf{q}) = \frac{\partial V}{\partial q_i} = -\sum_{k=1}^{n} m_k\,\mathbf{g}_0^T\,\frac{\partial \mathbf{p}_{c_k}}{\partial q_i}$$

### Why Does the Sum Start at $k = i$ Instead of $k = 1$?

Just like with the Jacobians:

```
Joint 1 ── Link 1 ── Joint 2 ── Link 2 ── Joint 3 ── Link 3

If i = 2:
    Moving joint 2 can move:  Link 2 ✅, Link 3 ✅
    Moving joint 2 CANNOT move: Link 1 ❌ (it's before joint 2)
    
    So: ∂p_c1/∂q₂ = 0  (joint 2 can't change link 1's position)
```

Therefore for links $k < i$, the partial derivative $\frac{\partial \mathbf{p}_{c_k}}{\partial q_i} = \mathbf{0}$, and those terms vanish:

$$\boxed{g_i(\mathbf{q}) = -\sum_{k=i}^{n} m_k\,\mathbf{g}_0^T\,\frac{\partial \mathbf{p}_{c_k}}{\partial q_i}}$$

---

## Concrete Example: 2R Planar Robot in Vertical Plane

```
         ↑ y (up, against gravity)
         │
         │    l_c1       l_c2
         ●━━━━◆━━━━●━━━━◆━━━━● 
       Joint1 CoM₁ Joint2 CoM₂
        (q₁)        (q₂)
         
    g₀ = [0, -9.81, 0]ᵀ  (gravity points DOWN in y)
    (Using 2D: x-y plane, gravity in -y direction)
```

### Variable Definitions

| Variable | Meaning |
|---|---|
| $q_1$ | Angle of link 1 from horizontal |
| $q_2$ | Angle of link 2 relative to link 1 |
| $l_{c1}$ | Distance from joint 1 to CoM of link 1 |
| $l_{c2}$ | Distance from joint 2 to CoM of link 2 |

### CoM Positions

$$\mathbf{p}_{c_1} = \begin{bmatrix} l_{c1}\cos q_1 \\ l_{c1}\sin q_1 \end{bmatrix}$$

$$\mathbf{p}_{c_2} = \begin{bmatrix} l_1\cos q_1 + l_{c2}\cos(q_1+q_2) \\ l_1\sin q_1 + l_{c2}\sin(q_1+q_2) \end{bmatrix}$$

### Total Potential Energy

Only the **y-components** (heights) matter:

$$V = m_1\,g\,\underbrace{l_{c1}\sin q_1}_{\text{height of CoM}_1} + m_2\,g\,\underbrace{\Big(l_1\sin q_1 + l_{c2}\sin(q_1+q_2)\Big)}_{\text{height of CoM}_2}$$

### Compute $g_1(\mathbf{q})$: Gravity Torque at Joint 1

$$g_1 = \frac{\partial V}{\partial q_1}$$

$$g_1 = m_1\,g\,l_{c1}\cos q_1 + m_2\,g\,\Big(l_1\cos q_1 + l_{c2}\cos(q_1+q_2)\Big)$$

$$\boxed{g_1 = (m_1 l_{c1} + m_2 l_1)\,g\cos q_1 + m_2\,l_{c2}\,g\cos(q_1+q_2)}$$

### Physical Meaning of $g_1$

```
        ↑ gravity
        │
        │     q₁ = 0° (horizontal)        q₁ = 90° (vertical)
        │     
        ●━━━━━━━━━━━●                      ●
        Maximum torque needed              │
        (full weight × full moment arm)    │
        cos(0°) = 1                        ━━━━━━━━━━━●
                                           Zero torque!
                                           (weight passes through joint)
                                           cos(90°) = 0
```

> Joint 1 must support the weight of **both links** — hence both $m_1$ and $m_2$ appear.

### Compute $g_2(\mathbf{q})$: Gravity Torque at Joint 2

$$g_2 = \frac{\partial V}{\partial q_2}$$

The only term involving $q_2$ is $l_{c2}\sin(q_1+q_2)$:

$$\boxed{g_2 = m_2\,l_{c2}\,g\cos(q_1+q_2)}$$

### Physical Meaning of $g_2$

```
Joint 2 only needs to support link 2's weight.
It doesn't care about link 1 at all!

        ●━━━━━━━━━━●━━━━◆━━━━●
        Joint 1     Joint 2  CoM₂
                     ↑
                     Only supports this part →
                     
    Only m₂ appears, NOT m₁ ✅
```

---

## The Full Gravity Vector for 2R Robot

$$\boxed{\mathbf{g}(\mathbf{q}) = \begin{bmatrix} (m_1 l_{c1} + m_2 l_1)\,g\cos q_1 + m_2 l_{c2}\,g\cos(q_1+q_2) \\ m_2 l_{c2}\,g\cos(q_1+q_2)\end{bmatrix}}$$

---

## Why $\mathbf{g}(\mathbf{q})$ Depends Only on Configuration

```
 Depends on q?          YES ✅
 ─────────────
 The POSE determines moment arms.
 Horizontal arm → max gravity torque.
 Vertical arm   → zero gravity torque.
 
 
 Depends on q̇?          NO ❌
 ──────────────
 Gravity doesn't care how fast you're moving.
 A stationary arm in a pose feels the SAME 
 gravity torque as a fast-moving arm in that pose.
 
 
 Depends on q̈?          NO ❌
 ──────────────
 Gravity doesn't care about acceleration either.
```

This is why:

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\,\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\,\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

- $\mathbf{M}$ multiplies $\ddot{\mathbf{q}}$ → inertia needs acceleration
- $\mathbf{C}$ multiplies $\dot{\mathbf{q}}$ → Coriolis/centrifugal needs velocity
- $\mathbf{g}$ stands alone → gravity just needs to know **where you are**

---

## Summary: The Logic Chain

```
Every link has mass at some height
            │
            ▼
Potential energy: V = Σ mₖ g hₖ(q)
            │
            ▼
Euler-Lagrange: gᵢ = ∂V/∂qᵢ
            │
            ▼
Chain rule: ∂V/∂qᵢ = Σ mₖ g₀ᵀ (∂p_ck/∂qᵢ)
            │
            ▼
Serial chain sparsity: joint i can't move links before it
            │
            ▼
Sum starts at k = i, not k = 1
            │
            ▼
gᵢ(q) = -Σ(k=i to n) mₖ g₀ᵀ (∂p_ck/∂qᵢ)
```

**In one sentence:** The gravity term is simply the **gradient of gravitational potential energy with respect to joint angles** — it tells each joint how much torque gravity is demanding at the current configuration, based on the heights and moment arms of all the links that joint supports.