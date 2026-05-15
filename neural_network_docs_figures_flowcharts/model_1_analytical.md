# Model 1: Analytical Inverse Dynamics (RNEA + Friction Baseline)

The analytical baseline computes the expected joint torques from first principles using the Recursive Newton-Euler Algorithm (RNEA) implemented via Pinocchio, augmented with a calibrated differentiable friction model. It requires **no training data** and serves as both the physics prior for the neural models and the performance lower bound that data-driven methods must surpass.

---

## 1. Rigid-Body Equation of Motion

For a serial-chain manipulator with $J$ active joints, the continuous-time Newton-Euler equations in joint space are:
$$
\tau_{rnea}(t) = M\!\bigl(q(t)\bigr)\,\ddot{q}(t) + C\!\bigl(q(t),\,\dot{q}(t)\bigr)\,\dot{q}(t) + g\!\bigl(q(t)\bigr)
$$
where:
- $q(t) \in \mathbb{R}^J$: joint configuration vector (radians)
- $\dot{q}(t) \in \mathbb{R}^J$: joint velocity vector (rad/s)
- $\ddot{q}(t) \in \mathbb{R}^J$: joint acceleration vector (rad/s²)
- $M(q) \in \mathbb{R}^{J \times J}$: symmetric positive-definite joint-space inertia matrix; entries depend on configuration through link mass distributions and inertia tensors
- $C(q,\dot{q}) \in \mathbb{R}^{J \times J}$: Coriolis–centripetal matrix; its entries involve the Christoffel symbols of the first kind derived from the Lagrangian kinetic energy
- $g(q) \in \mathbb{R}^J$: gravity loading vector; the $j$-th entry is $-\frac{\partial V(q)}{\partial q_j}$ where $V(q) = \sum_i m_i \mathbf{g}^\top \mathbf{r}_{c_i}(q)$ is the total gravitational potential energy

Direct evaluation of $M(q)$ requires $O(J^3)$ floating-point operations and $C(q,\dot{q})$ requires $O(J^2)$. For $J = 5$ this is acceptable, but the RNEA avoids explicit matrix assembly entirely.

---

## 2. O(n) RNEA Component Decomposition

The Recursive Newton-Euler Algorithm evaluates the full right-hand side $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q)$ in $O(J)$ complexity via an outward propagation of velocities/accelerations (kinematics) followed by an inward propagation of forces (dynamics). Pinocchio's implementation provides this through `pinocchio.rnea(model, data, q, qd, qdd)`.

By exploiting the linearity of $\tau_{rnea}$ with respect to $\ddot{q}$ and $\dot{q}$, we isolate the three force contributions through structured calls with selective zeroing:

**Gravity torque** (configuration-dependent static load):
$$
\tau_g = \text{RNEA}(q,\, \mathbf{0}_J,\, \mathbf{0}_J) \in \mathbb{R}^J
$$
Setting both velocity and acceleration to zero eliminates inertial and Coriolis contributions, leaving only the gravitational generalized forces.

**Inertial torque** (mass-times-acceleration term):
$$
\tau_M = \text{RNEA}(q,\, \mathbf{0}_J,\, \ddot{q}) - \tau_g \in \mathbb{R}^J
$$
Setting $\dot{q} = \mathbf{0}$ eliminates Coriolis terms; the residual after subtracting gravity is exactly $M(q)\ddot{q}$.

**Coriolis–centripetal torque** (velocity-dependent forces):
$$
\tau_C = \text{RNEA}(q,\, \dot{q},\, \mathbf{0}_J) - \tau_g \in \mathbb{R}^J
$$
Setting $\ddot{q} = \mathbf{0}$ eliminates inertial terms; the residual is $C(q,\dot{q})\dot{q}$.

The decomposition satisfies the rigid-body identity exactly:
$$
\tau_{rnea} = \tau_g + \tau_M + \tau_C
$$
This follows directly from the linearity of RNEA in the forcing terms: $\text{RNEA}(q,\dot{q},\ddot{q}) = \text{RNEA}(q,\mathbf{0},\mathbf{0}) + \text{RNEA}(q,\mathbf{0},\ddot{q}) - \text{RNEA}(q,\mathbf{0},\mathbf{0}) + \text{RNEA}(q,\dot{q},\mathbf{0}) - \text{RNEA}(q,\mathbf{0},\mathbf{0})$.

---

## 3. Inertial Parameter Calibration

The URDF was constructed assuming nominal solid-body densities. The Kikobot structure is 3D-printed in PLA at approximately 70% infill; servo motors (metal, ~60 g each) contribute mass that does not scale with printed density. A global density scale factor $\alpha$ is therefore applied to every body in the Pinocchio model:
$$
m_i \leftarrow \alpha\, m_i, \qquad I_i \leftarrow \alpha\, I_i, \quad \forall\, i \in \{1, \ldots, n_{bodies}\}
$$
where $I_i \in \mathbb{R}^{3\times 3}$ is the body's $3\times3$ inertia tensor about its center of mass.

$\alpha$ is identified by weighted least squares over quasi-static trajectories (where $\dot{q} \approx \mathbf{0}$, $\ddot{q} \approx \mathbf{0}$, so $\tau \approx \tau_g(\alpha)$):
$$
\alpha^* = \arg\min_\alpha \sum_{i=1}^{N_{qs}} \left\| \tau_i^{load} - \alpha\, \tau_i^{rnea,unit} \right\|^2 = \frac{\sum_i \tau_i^{load\,\top} \tau_i^{rnea,unit}}{\sum_i \|\tau_i^{rnea,unit}\|^2}
$$
where $\tau_i^{rnea,unit}$ is the RNEA output computed with $\alpha = 1$. Calibration over 470,000 samples yielded $\alpha \approx 0.093$ (Neural_Networks pipeline). Lumped additional masses $\delta m_i$ (kg) can be added to individual links post-scaling to account for servo motor mass not represented in the URDF.

---

## 4. Differentiable Friction Model

Servo gearboxes exhibit static friction (stiction) and viscous drag that are not captured by the rigid-body model. The classical dry-friction model uses $\text{sgn}(\dot{q})$, which is discontinuous at $\dot{q} = 0$, making gradient-based calibration ill-defined. A smooth surrogate replaces the signum with a scaled hyperbolic tangent transition:
$$
\tau_f\!\bigl(\dot{q}\bigr) = c \odot \tanh\!\left(\frac{\dot{q}}{\epsilon}\right) + v \odot \dot{q} \in \mathbb{R}^J
$$
where:
- $c \in \mathbb{R}^J_{\geq 0}$: per-joint **Coulomb friction** coefficients (N·m); the asymptotic breakaway torque at high velocity
- $v \in \mathbb{R}^J_{\geq 0}$: per-joint **viscous drag** coefficients (N·m·s/rad); the linear velocity-proportional damping term
- $\epsilon \in \mathbb{R}_{>0}$: stiction half-width parameter (rad/s); governs the gradient of the stiction-to-sliding transition ($\epsilon = 0.0405\ \text{rad/s}$ by default)
- $\odot$: element-wise (Hadamard) product

The derivative of $\tau_f$ with respect to $\dot{q}_j$ is:
$$
\frac{\partial \tau_{f,j}}{\partial \dot{q}_j} = \frac{c_j}{\epsilon}\,\text{sech}^2\!\left(\frac{\dot{q}_j}{\epsilon}\right) + v_j
$$
which is always finite and positive, confirming that $\tau_f$ is strictly monotone in each joint velocity. At high velocities $|\dot{q}_j| \gg \epsilon$, $\tanh(\dot{q}_j/\epsilon) \to \pm 1$ and the friction saturates to $\pm c_j + v_j \dot{q}_j$ (Coulomb-dominated). Near zero velocity, $\tanh(x) \approx x$ and the friction approaches $(c_j/\epsilon + v_j)\dot{q}_j$ (viscous regime).

Calibration uses a constrained least-squares sweep over $\epsilon$ on full-speed trajectories, minimizing:
$$
\min_{c, v \geq 0} \sum_{i=1}^{N} \left\|\tau_i^{load} - \tau_i^{rnea} - c \odot \tanh\!\left(\frac{\dot{q}_i}{\epsilon}\right) - v \odot \dot{q}_i\right\|^2
$$

---

## 5. Hardware Torque Measurement

The servo load register $L_{k,j} \in [-1000, +1000]$ reports signed shaft torque as a fraction of the rated stall torque at nominal supply voltage. The physical torque in URDF frame is:
$$
\tau_j^{load} = -d_j \cdot \frac{L_{k,j}}{1000} \cdot \bar{\tau}_j \cdot \frac{V_{k,j}}{V_{nom}} \cdot c_{Nm}
$$
where:
- $d_j \in \{-1,+1\}$: URDF-aligned direction flag for joint $j$
- $\bar{\tau}_j$: rated stall torque at $V_{nom}$ (kgf·cm): $\bar{\tau}_{J1,J2,J3,J5} = 30.0$, $\bar{\tau}_{J4} = 14.8$
- $V_{nom} = 12.0\ \text{V}$: nominal supply voltage
- $c_{Nm} = 0.09807\ \text{N·m/(kgf·cm)}$: conversion factor

The voltage-scaling factor $V_{k,j}/V_{nom}$ compensates for battery droop: torque output of a DC motor is approximately proportional to voltage at constant duty cycle. Without this correction a 10% voltage sag would introduce a systematic 10% underestimate of the measured torque.

---

## 6. Complete Analytical Prediction

The full analytical torque baseline combines rigid-body and friction contributions:
$$
\boxed{\tau_{phys} = \underbrace{\tau_g + \tau_M + \tau_C}_{\tau_{rnea}} + \tau_f}
$$

The four additive components are stored as the decomposed physics tensor:
$$
\mathbf{p} = \left[\tau_g^\top,\; \tau_M^\top,\; \tau_C^\top,\; \tau_f^\top\right]^\top \in \mathbb{R}^{4J}
$$
This 20-dimensional vector is materialized at preprocessing time and stored in `filtered_tau_decomposed.csv`. It serves as both the structured physics prior passed to Models 3 and 4, and the reference for the physics consistency regularization loss in Model 3.

The analytical model has **zero learnable parameters** and achieves a test RMSE of approximately 0.20–0.25 N·m on dynamic trajectories, limited primarily by (i) noise in the numerical acceleration $\ddot{q}$ entering $\tau_M = M(q)\ddot{q}$, (ii) unmodeled joint compliance and backlash, and (iii) asymmetric or velocity-dependent friction effects not captured by the smooth Coulomb model.
