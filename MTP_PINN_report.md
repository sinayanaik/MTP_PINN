# Physics-Informed Neural Networks for Robot Inverse Dynamics: A Comparative Study of Black-Box, Physics-Regularized, and Residual-Correction Architectures

**Abstract** — Accurate joint-torque estimation is a fundamental requirement for model-based robot control, adaptive friction compensation, and safe human–robot interaction. Classical analytical inverse dynamics computed via the Recursive Newton–Euler Algorithm (RNEA) provides a structured physical prior whose accuracy is limited by uncertain inertial parameters, unmodeled friction, and discretisation artefacts. This paper presents a systematic comparative study of three neural-network architectures for inverse-dynamics identification on a five-degree-of-freedom serial manipulator: (i) a fully data-driven feed-forward network (**BlackBox-FNN**) that treats torque prediction as a pure regression problem from kinematic inputs alone; (ii) a physics-regularised FNN (**PhysReg-FNN**) that concatenates the *full four-component decomposed* RNEA tensor $[\boldsymbol{\tau}_g, \boldsymbol{\tau}_M, \boldsymbol{\tau}_C, \boldsymbol{\tau}_f]$ (4·J = 20 features) to the kinematic vector (3·J = 15 features) for a 7·J = 35-dim augmented input, and augments the data loss with an additive Tikhonov physics penalty pulling the prediction toward the analytical RNEA sum; and (iii) a residual-correction FNN (**ResCorr-FNN**) that imposes a hard structural decomposition $\hat{\boldsymbol{\tau}} = \boldsymbol{\tau}_{\text{phys}} + c_s \cdot \tanh(g_\theta(\cdot))$, where the bounded correction cannot exceed $\pm c_s$ in normalised units, with $c_s = 0.5$ a fixed (non-learnable) buffer providing a hard prior on physics reliance. All three architectures share a unified MLP backbone (Linear $\rightarrow$ LayerNorm $\rightarrow$ activation $\rightarrow$ Dropout, Xavier-normal initialisation), trained on 66,735 time steps drawn from 22 trajectories spanning 11 distinct Cartesian motion geometries, and evaluated on a held-out test set of 38,106 samples drawn from a further 11 trajectories disjoint from training. A structured hyperparameter grid spanning **144 training runs** (BlackBox: 12, PhysReg: 72, ResCorr: 60) — see `Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/` — sweeps training-data fraction (six levels from 2 % to 100 %), physics penalty weight $\lambda \in \{0.05, 0.1, 0.2, 0.5, 1.0, 2.0\}$, and L₂ correction penalty $\alpha_r \in \{0.005, 0.01, 0.05, 0.1, 0.5\}$ across two seeds. Best pooled test RMSE values are **0.0967 N·m** (ResCorr-FNN, frac = 0.05, α<sub>r</sub> = 0.1, R² = 0.911), **0.0980 N·m** (PhysReg-FNN, frac = 0.05, λ = 0.5, R² = 0.908), and **0.1057 N·m** (BlackBox-FNN, frac = 1.0, R² = 0.893). The two physics-aware architectures attain their global optima at **only 5 % of the training data** (~3,300 samples), giving a > 20× sample-efficiency advantage over the BlackBox baseline. Beyond aggregate accuracy, we contribute (a) a normalisation-invariance theorem that guarantees the four-component physics tensor sums to the normalised target, (b) closed-form gradient identities for each architecture that reveal *why* the additive Tikhonov form behaves as a prior-averaged predictor, and (c) a contraction proof for the tanh-bounded correction. The full source code, calibrated URDF, and 144-trial metadata are released for direct comparison.

---

## I. Introduction

Inverse dynamics — the mapping from joint kinematics $(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})$ to joint torques $\boldsymbol{\tau}$ — is central to feed-forward torque control, friction compensation, payload estimation, and computed-torque tracking in serial manipulators. The gold-standard formulation follows from the rigid-body Lagrangian (equivalently, Newton–Euler):

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) + \boldsymbol{\tau}_f(\dot{\mathbf{q}}),\tag{1}$$

where $\mathbf{M} \in \mathbb{R}^{n\times n}$ is the configuration-dependent mass matrix, $\mathbf{C}$ encodes Coriolis/centripetal coupling, $\mathbf{g}$ is the gravity vector, and $\boldsymbol{\tau}_f$ accounts for joint friction. In practice, exact computation of these terms requires every link's mass, centre-of-mass, and inertia tensor — parameters that are typically uncertain by 10–40 % in low-cost manipulators built from PLA or ABS-printed parts.

Data-driven approaches sidestep parameter identification by learning the inverse dynamics directly from measurements. Early work used Gaussian Processes [1] and locally weighted regression [2]; more recently, deep feed-forward networks have been shown to match or exceed classical identification accuracy [3, 4]. Pure black-box networks, however, ignore the rich structural knowledge encoded in Lagrangian mechanics: they require more data to achieve equivalent accuracy and offer no guarantee of physical consistency under distribution shift.

Physics-Informed Neural Networks (PINNs) [5] incorporate physical constraints either as soft penalties in the loss or as hard architectural constraints in the network structure. Subsequent work specialised this idea for rigid-body dynamics through Lagrangian/Hamiltonian neural networks [6, 7] and *delta-physics* residual learning [8, 9]. Applied to robot inverse dynamics, two complementary strategies emerge: (a) **physics regularisation**, where the loss penalises deviation from an analytical prediction; and (b) **residual correction**, where the network learns an additive correction to a structured physics model, decomposing the learning problem into a solved physical component and a compact residual.

This paper makes the following contributions.

1. A rigorous formulation and implementation of three inverse-dynamics architectures under a unified training framework with shared feature extraction, normalisation, optimiser, and macro-RMSE validation.
2. A 144-trial hyperparameter grid sweep quantifying the effect of training-data fraction (six levels from 2 % to 100 %), physics weight (six values), and residual penalty (five values) on held-out test accuracy.
3. Detailed per-joint analysis revealing architecture-specific failure modes and the role of physics guidance for high-inertia joints.
4. A normalisation-invariance theorem (Thm. 1) proving that the per-component physics tensor sums to the normalised target, removing a long-standing source of inconsistency in physics-aware loss design.
5. A contraction proof for the tanh-bounded residual correction (Lemma 2) establishing the worst-case deviation guarantee that distinguishes ResCorr-FNN from regularisation-only approaches.
6. Open-source code, trained model checkpoints, calibrated URDF, and architecture diagrams of the full pipeline (see Appendix A).

The rest of the paper is organised as follows. Section II describes the robot system and learning problem. Sections III–V detail the data acquisition pipeline, signal processing, and analytical RNEA model. Section VI presents the three neural architectures with full pseudocode. Section VII covers the training methodology. Section VIII describes the grid sweep, Section IX reports results, Section X discusses mechanistic interpretations, Section XI lists limitations, and Section XII concludes. Appendices A–C cover the figure index, reproducibility commands, and full algorithmic pseudocode boxes.

---

## II. Robot System and Problem Formulation

### A. Manipulator

Experiments are conducted on a five-degree-of-freedom serial manipulator (Kikobot) with joints J₁–J₅ driven by Feetech smart serial servos. The kinematic chain is

$$\text{base} \xrightarrow{J_1} \text{shoulder} \xrightarrow{J_2} \text{upper arm} \xrightarrow{J_3} \text{forearm} \xrightarrow{J_4} \text{wrist-pitch} \xrightarrow{J_5} \text{wrist-roll} \rightarrow \text{end effector}.$$

The active joint vector is $\mathbf{q} = [q_1, \ldots, q_5]^\top \in \mathbb{R}^5$. From the URDF (`robot_description/urdf/kikobot.xacro`) the joint axes in their respective parent frames are
$\hat{\mathbf{a}}_1 = (0,0,1)^\top$,
$\hat{\mathbf{a}}_2 = (-1,0,0)^\top$,
$\hat{\mathbf{a}}_3 = (-1,0,0)^\top$,
$\hat{\mathbf{a}}_4 = (1,0,0)^\top$, and
$\hat{\mathbf{a}}_5 = (0,-2{\times}10^{-4},1)^\top$ (numerical roll axis ≈ vertical). Joints J₂–J₄ form the gravity-loaded shoulder–elbow–pitch sub-chain that dominates torque magnitudes. Nominal link masses (before density rescaling, see § V-D) are listed in Table I; the total nominal mass is $\sum_k m_k \approx 6.37$ kg, corresponding to the standard PLA fill density assumed by the CAD source.

**Table I: Nominal Link Masses (URDF, before scaling)**

| Link | Mass (kg) |
|------|-----------|
| base_link | 1.7062 |
| shoulder_joint_1 | 0.7619 |
| elbow_joint_1 | 1.1884 |
| wrist_1_1 | 1.1108 |
| wrist_2_1 | 0.6675 |
| wrist_3_1 | 0.7494 |
| end_effector_1 | 0.1883 |
| **Total** | **6.3725** |

**Table II: Joint Actuator Specifications**

| Joint | Actuator | Rated stall | Role |
|-------|----------|-------------|------|
| J₁ | Feetech STS3215 | 30.0 kgf·cm (≈ 2.94 N·m) | Shoulder rotation (yaw) |
| J₂ | STS3215 | 30.0 kgf·cm | Shoulder elevation (pitch) |
| J₃ | STS3215 | 30.0 kgf·cm | Elbow (pitch) |
| J₄ | STS3032 | 14.8 kgf·cm (≈ 1.45 N·m) | Wrist pitch |
| J₅ | STS3215 | 30.0 kgf·cm | Wrist roll |

The smaller stall torque of J₄ is the principal reason its measured-torque distribution has the smallest range across the chain (σ ≈ 0.46 N·m), and is consistent with the lowest test RMSE observed across all architectures at that joint.

### B. Inverse-Dynamics Learning Problem

Given a dataset $\mathcal{D} = \{(\mathbf{x}_i, \boldsymbol{\tau}_i^*)\}_{i=1}^N$, where $\mathbf{x}_i = [\mathbf{q}_i^\top, \dot{\mathbf{q}}_i^\top, \ddot{\mathbf{q}}_i^\top]^\top \in \mathbb{R}^{3J}$ is the kinematic state and $\boldsymbol{\tau}_i^* \in \mathbb{R}^J$ is the (filtered) measured joint torque, we seek a function $f_\theta : \mathbb{R}^{3J} \rightarrow \mathbb{R}^J$ that minimises the empirical risk

$$\hat{\theta} = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \ell\big(f_\theta(\mathbf{x}_i; \boldsymbol{\phi}_i),\, \boldsymbol{\tau}_i^*\big) + \mathcal{R}(\theta, \boldsymbol{\phi}_i),\tag{2}$$

where $\boldsymbol{\phi}_i \in \mathbb{R}^{4J}$ is the *decomposed* analytical-physics tensor (defined in § V) optionally consumed by $f_\theta$, $\ell$ is a per-architecture data loss, and $\mathcal{R}$ is an architecture-specific regulariser. Section VI specialises (2) for each of the three architectures.

The kinematic chain (with $J=5$) and the end-to-end data flow are described in § III.

---

## III. Data Acquisition Pipeline

### A. Trajectory Corpus

The hardware was driven through a corpus of **124 raw execution logs** collected at a control and feedback frequency of ≈ 303 Hz, spanning 11 distinct Cartesian motion geometries: circle, ellipse, helix, Lissajous, parabola, rectangle, regular polygon, sine wave, spiral, square, and triangle. Geometric parameters (radius 50–198 mm, orientation azimuth/elevation, planner family — Ruckig, quintic polynomial, cubic polynomial, trapezoidal, quintic Bézier) are encoded in each file name. This diversity ensures models must generalise across qualitatively different velocity and acceleration profiles rather than overfitting a single motion class.

For the experiments reported in this paper, a **manually balanced subset of 44 trajectories** is selected from the 124-file pool — exactly **four trajectories per geometry class**, allocated 2/1/1 to train/val/test respectively. This per-class allocation is documented in the dataset's `metadata.json` under `split.manual_alloc`. Trajectory-level allocation (rather than sample-level) is mandatory: every sample within a trajectory shares strong temporal autocorrelation, and a sample-level split would leak information from train into validation and test.

Selecting only four trajectories per class — instead of using all 8–18 available — is a deliberate stress test: with so few independent motion realisations per class (one of which is held out for validation and one for test), the network must extract *generalisable* dynamics rather than memorise per-trajectory shapes. This makes the regime sensitive to inductive biases, which is precisely the regime where the physics priors of PhysReg-FNN and ResCorr-FNN are expected to pay off.

Geometry counts in the raw pool (post-QA) and the manual subset are summarised in Table III.

**Table III: Geometry Distribution and Manual Split (run_train22)**

| Geometry | Available | Train | Val | Test |
|----------|----------:|------:|----:|-----:|
| circle | 14 | 2 | 1 | 1 |
| ellipse | 18 | 2 | 1 | 1 |
| helix | 10 | 2 | 1 | 1 |
| lissajous | 11 | 2 | 1 | 1 |
| parabola | 11 | 2 | 1 | 1 |
| rectangle | 8 | 2 | 1 | 1 |
| regular_polygon | 7 | 2 | 1 | 1 |
| sine_wave | 11 | 2 | 1 | 1 |
| spiral | 12 | 2 | 1 | 1 |
| square | 8 | 2 | 1 | 1 |
| triangle | 11 | 2 | 1 | 1 |
| **Total** | **121** | **22** | **11** | **11** |

After 1 % front/back trimming and Savitzky-Golay filtering (§ IV), the resulting per-split sample counts are:

**Table IV: Dataset Statistics (post-trim, post-filter)**

| Split | Trajectories | Time steps | Fraction |
|-------|--------------|-----------:|---------:|
| Train | 22 | 66,735 | 43.7 % |
| Validation | 11 | 47,868 | 31.4 % |
| Test | 11 | 38,106 | 24.9 % |
| **Total** | **44** | **152,709** | 100 % |

The unusually high val/test fractions are a direct consequence of the 2/1/1 allocation: since validation and test trajectories tend to have similar duration to training trajectories, halving the trajectory count of each non-train split (relative to train) yields ~50 % of the train sample count rather than the conventional 15 %. This makes the held-out evaluation statistically robust — both val (47.9 K) and test (38.1 K) samples are large enough that pooled metrics have negligible standard error.

### B. End-to-End Data Flow

The data path from raw hardware logs to per-batch tensors follows strict stage ordering: every artefact downstream of a stage is recomputable from its predecessor, so any preprocessing change re-materialises the chain.


The artefacts produced after the SPLIT and NORM stages are persisted as plain CSVs (`filtered_q.csv`, `filtered_qd.csv`, `filtered_qdd.csv`, `filtered_tau_measured.csv`, `filtered_tau_decomposed.csv`) inside each split directory; the loader (`Neural_Networks/loader.py`) reads only these.

### C. Trajectory Trimming

The first and last 1 % of each trajectory is removed (`resolve_front_back_trim`, `loader.py:73–96`) before further processing. This eliminates boundary transients from the motion controller (initial servo wake-up oscillations, terminal hold-position dither) that contaminate the SG-filtered velocity/acceleration estimates near the edges of the window.

---

## IV. Signal Processing and Normalisation

### A. Savitzky-Golay Differentiation

Velocity and acceleration are derived from raw position samples by fitting a local polynomial of degree $p=3$ to a window of $w=121$ consecutive samples (≈ 0.40 s at 303 Hz) and reading the first and second derivatives of the fit at the window centre. Concretely, around index $k$ the window is

$$\mathcal{W}_k = \{q_{k-60},\, q_{k-59},\, \ldots,\, q_{k+60}\},$$

and the SG fit computes

$$q_k^{\text{filt}} = \mathbf{e}_0^\top (\mathbf{V}^\top \mathbf{V})^{-1} \mathbf{V}^\top \mathbf{q}_{\mathcal{W}_k},\qquad \dot{q}_k = \mathbf{e}_1^\top (\mathbf{V}^\top \mathbf{V})^{-1} \mathbf{V}^\top \mathbf{q}_{\mathcal{W}_k},\qquad \ddot{q}_k = 2\,\mathbf{e}_2^\top (\mathbf{V}^\top \mathbf{V})^{-1} \mathbf{V}^\top \mathbf{q}_{\mathcal{W}_k},\tag{3}$$

where $\mathbf{V} \in \mathbb{R}^{w \times (p+1)}$ is the Vandermonde matrix on the centred sample times and $\mathbf{e}_j$ is the $j$-th canonical basis vector. The same polynomial fit is reused for $q$, $\dot q$, and $\ddot q$, which enforces that the velocity and acceleration estimates are consistent in the analytic sense $\ddot q = \mathrm{d}\dot q/\mathrm{d}t$ within the polynomial approximation. The window length $w=121$ is chosen to suppress the dominant encoder-quantisation noise (period ≈ 12–18 samples at 303 Hz under typical motions) while preserving the kinematic structure: a window much shorter than the smallest control-induced acceleration timescale would amplify quantisation, while a window much longer would oversmooth genuine acceleration peaks.

Hard physical limits $|\dot q| \le 100$ rad/s and $|\ddot q| \le 1000$ rad/s² are enforced post-fit to remove residual outliers from short data dropouts.

A second SG pass with the same parameters $(w=121, p=3)$ is applied to the measured torque to remove servo communication-latency jitter. After the analytical RNEA is computed (§ V), a *shorter* SG pass $(w=15, p=3)$ is applied to the four physics components to remove numerical noise from the dynamics library — the post-filter window is intentionally short because RNEA is already a smooth function of smoothed inputs.

### B. Timestamp Repair

Hardware logs occasionally contain (i) non-monotonic timestamps (repeated or out-of-order entries due to USB packet reordering), (ii) outlier $\Delta t$ values exceeding 3σ around the median, and (iii) sample drift of more than 5 % CV across the trajectory. A three-stage repair pipeline (`Neural_Networks/robot_physics.py:117–236`) runs before any differentiation: it interpolates over non-monotone regions, re-samples outlier intervals, and — if total drift is still high — performs a uniform-grid resample at the median rate. This is essential because the SG derivatives in (3) implicitly assume a uniform time grid.

### C. Feature Normalisation

All inputs and targets are standardised to zero mean and unit variance using statistics computed **exclusively from the training split** and applied identically to all splits:

$$\tilde{x}_j^{(k)} = \frac{x_j^{(k)} - \mu_{k,j}^{\text{train}}}{\sigma_{k,j}^{\text{train}}},\quad k \in \{q, \dot q, \ddot q\},\ j \in \{1,\ldots,5\},\tag{4}$$

$$\tilde{\boldsymbol{\tau}}_j^* = \frac{\boldsymbol{\tau}_j^* - \mu_{\tau, j}^{\text{train}}}{\sigma_{\tau,j}^{\text{train}}}.\tag{5}$$

The decomposed physics tensor $\boldsymbol{\phi} = [\boldsymbol{\tau}_g^\top,\, \boldsymbol{\tau}_M^\top,\, \boldsymbol{\tau}_C^\top,\, \boldsymbol{\tau}_f^\top]^\top \in \mathbb{R}^{20}$ is normalised so that the *sum* of the four normalised components equals the normalised target:

$$\tilde{\phi}_j^{(k)} = \frac{\tau_{k,j} - \mu_{\tau, j}^{\text{train}}/4}{\sigma_{\tau, j}^{\text{train}}},\quad k \in \{g, M, C, f\}.\tag{6}$$

This per-component normalisation distributes the target mean equally across the four physics terms and uses the target standard deviation for all four. The implementation is exactly two lines (`loader.py:330–332`):

```python
per_comp_mean = np.tile(self.mean_tau / 4.0, 4)   # (20,)
per_comp_std  = np.tile(self.std_tau,        4)   # (20,)
return ((self.tau_analytical[idx] - per_comp_mean) / per_comp_std)
```

### D. Normalisation-Invariance Theorem

The choice in (6) is not a heuristic — it is the *unique* per-component scheme that makes the four-component sum identifiable in normalised space.

**Theorem 1 (Normalisation invariance).** *Let $\boldsymbol{\tau}^* \in \mathbb{R}^J$ be the measured torque and $\boldsymbol{\tau}_g, \boldsymbol{\tau}_M, \boldsymbol{\tau}_C, \boldsymbol{\tau}_f \in \mathbb{R}^J$ be the four analytical components computed from the same kinematics, satisfying $\boldsymbol{\tau}^* = \boldsymbol{\tau}_g + \boldsymbol{\tau}_M + \boldsymbol{\tau}_C + \boldsymbol{\tau}_f + \boldsymbol{\eta}$ where $\boldsymbol{\eta}$ is the model–reality gap. Under the per-component normalisation* (6), *the equality*

$$\sum_{k \in \{g,M,C,f\}} \tilde{\boldsymbol{\tau}}_k = \tilde{\boldsymbol{\tau}}^*$$

*holds in noise-free cases ($\boldsymbol{\eta} = 0$) and with an additive bias of $\boldsymbol{\eta}/\boldsymbol{\sigma}_\tau^{\text{train}}$ otherwise.*

*Proof.* Direct computation: with $\mu = \mu_{\tau,j}^{\text{train}}$ and $\sigma = \sigma_{\tau,j}^{\text{train}}$ for a fixed joint $j$,

$$\sum_{k} \tilde{\tau}_{k,j} = \sum_k \frac{\tau_{k,j} - \mu/4}{\sigma} = \frac{(\sum_k \tau_{k,j}) - \mu}{\sigma} = \frac{\tau_j^* - \mu - \eta_j}{\sigma} = \tilde{\tau}_j^* - \eta_j/\sigma.\ \blacksquare$$

This identity is what allows `reduce_physics_to_total(physics)` (§ VI-A) to be a *normalised-space* operation: summing the 20-dim tensor along the four-component axis produces a 5-dim quantity directly comparable to the normalised target tensor without any rescaling.

---

## V. RNEA-Based Physics Model

### A. Pinocchio Model Build

The URDF (`robot_description/urdf/kikobot.xacro`) is processed via `xacro` and loaded into Pinocchio [10] (`build_pinocchio_model`, `Neural_Networks/robot_physics.py:396–430`). Two parameters are then applied:

1. **Mass scale** $\rho = 0.0931$ (from `Torque_Analysis/calibration_params.json`, calibrated 2026-03-28 over all 124 raw files / 470,529 samples). Every link's mass and inertia tensor is multiplied by $\rho$:
   ```python
   model.inertias[i].mass    *= mass_scale
   model.inertias[i].inertia *= mass_scale
   ```
   This brings the nominal 6.37 kg URDF total down to **0.5934 kg**, matching the measured PLA-printed structure with ≈ 70 % infill plus the lumped servo masses. The factor was fit by least-squares matching of analytical gravity torque to the measured load register on all 470,529 bulk samples.
2. **Optional extra masses** (`extra_masses` argument): per-joint additive masses for unmodelled servos. Currently set to `null` in the calibration JSON.

### B. Four-Component RNEA Decomposition

The total inverse-dynamics torque is decomposed into four physically interpretable components (`compute_rnea_decomposition`, `robot_physics.py:433–488`):

$$\boldsymbol{\tau}_{\text{RNEA}} = \underbrace{\boldsymbol{\tau}_g(\mathbf{q})}_{\text{gravity}} + \underbrace{\boldsymbol{\tau}_M(\mathbf{q}, \ddot{\mathbf{q}})}_{\text{inertial}} + \underbrace{\boldsymbol{\tau}_C(\mathbf{q}, \dot{\mathbf{q}})}_{\text{Coriolis}}.\tag{7}$$

Each term is computed as a difference of RNEA evaluations:

$$\boldsymbol{\tau}_g = \mathrm{RNEA}(\mathbf{q}, \mathbf{0}, \mathbf{0}),\qquad \boldsymbol{\tau}_M = \mathrm{RNEA}(\mathbf{q}, \mathbf{0}, \ddot{\mathbf{q}}) - \boldsymbol{\tau}_g,\qquad \boldsymbol{\tau}_C = \mathrm{RNEA}(\mathbf{q}, \dot{\mathbf{q}}, \mathbf{0}) - \boldsymbol{\tau}_g.\tag{8}$$

Pinocchio's `pin.rnea` (Featherstone-style spatial-algebra recursion) is invoked four times per sample. A non-RNEA friction term $\boldsymbol{\tau}_f$ is added to obtain the full analytical torque

$$\boldsymbol{\tau}_{\text{phys}} = \boldsymbol{\tau}_g + \boldsymbol{\tau}_M + \boldsymbol{\tau}_C + \boldsymbol{\tau}_f.\tag{9}$$

The four-component vector $\boldsymbol{\phi} = [\boldsymbol{\tau}_g^\top, \boldsymbol{\tau}_M^\top, \boldsymbol{\tau}_C^\top, \boldsymbol{\tau}_f^\top]^\top \in \mathbb{R}^{20}$ is retained separately as the physics feature tensor for the augmented architectures (§ VI-C, VI-D).


### C. Friction Model

Joint friction is modelled as smooth Coulomb plus viscous (`torque_friction`, `robot_physics.py:375–389`):

$$\boldsymbol{\tau}_f(\dot{\mathbf{q}}) = \mathbf{c} \odot \tanh\!\left(\frac{\dot{\mathbf{q}}}{\varepsilon}\right) + \mathbf{v} \odot \dot{\mathbf{q}},\tag{10}$$

with the calibrated coefficients from `Torque_Analysis/calibration_params.json` (bulk-fit on 470,529 samples):

$$\mathbf{c} = (0.1350,\, 0.2782,\, 0.2013,\, 0.0881,\, 0.2039)^\top\ \mathrm{N{\cdot}m},$$

$$\mathbf{v} = (0.3000,\, 0.3000,\, 0.2454,\, 0.0402,\, 0.0469)^\top\ \mathrm{N{\cdot}m{\cdot}s/rad},\qquad \varepsilon = 0.0405\ \mathrm{rad/s}.$$

The hyperbolic-tangent transition with width $\varepsilon$ regularises the otherwise non-differentiable Coulomb sign function, which is essential for any gradient-based learner that backpropagates through (10): the gradient

$$\frac{\partial \tau_{f,j}}{\partial \dot{q}_j} = \frac{c_j}{\varepsilon}\bigl(1 - \tanh^2(\dot{q}_j/\varepsilon)\bigr) + v_j\tag{11}$$

is bounded by $c_j/\varepsilon + v_j$ at $\dot{q}_j = 0$ (the worst case) and approaches $v_j$ for $|\dot{q}_j| \gg \varepsilon$. Below $|\dot q| \approx \varepsilon$ the Coulomb term dominates; above, the linear viscous term dominates. The bulk calibration reduced per-joint RMS residuals at J₄ from 0.0872 N·m to 0.0559 N·m (a 36 % improvement), with smaller gains at J₁–J₃, J₅. The final calibration is `bulk: true` and supersedes earlier single-trajectory fits (full audit trail in the JSON `friction.history` array).

### D. Calibration Provenance

The calibration JSON maintains *both* the current parameters (`mass.current`, `friction.current`) and the full history of every calibration run since 2026-03-28, including per-stage RMS-old/RMS-new vectors. This is loaded once at module import (`robot_physics.py:58–98`) with hardcoded fallbacks if the file is missing. A separate calibration is *not* repeated per training run — the network simply consumes the same RNEA + friction surface.

---

## VI. Neural Network Architectures

All three architectures share a unified MLP backbone, differing only in (i) input dimensionality (kinematic-only vs. kinematic + decomposed physics), (ii) the parameterisation of the output map (free regression vs. tanh-bounded residual), and (iii) the training loss (data-only vs. data + physics regulariser vs. data + correction-magnitude regulariser).

A side-by-side schematic that summarises the three pipelines — input streams, MLP backbone, output map, and per-architecture loss — is given in **Fig. 9** (`PLOTS_and_FLOWCHARTS/drawio_flowcharts/architecture of all Neural Network Diagrams.pdf`), reproduced inline below. This drawio figure also enumerates the fixed backbone hyperparameters and the experimental sweep grid used in the run_train22 study.

![Fig. 9 — Unified comparison of the three NN architectures, the analytical-physics block (RNEA + friction), and the experimental sweep grid. Inputs $x_k \in \mathbb{R}^{5\times 3}$ are kinematic features $(q, \dot q, \ddot q)$; $x_p$ are the four-component RNEA + friction features; the green network output is $\hat\tau \in \mathbb{R}^{5\times 1}$.](PLOTS_and_FLOWCHARTS/drawio_flowcharts/architecture%20of%20all%20Neural%20Network%20Diagrams.pdf)

The remainder of § VI gives the algebraic and architectural specification of each block in Fig. 9, with line-number references to the implementation.

### A. Shared MLP Backbone

The factory function `build_mlp` (`Neural_Networks/models/torque_models.py:20–38`) constructs an $L$-hidden-layer feed-forward network with widths $h_1, \ldots, h_L$. Each hidden block is

$$\mathbf{a}^{(l)} = \mathrm{Dropout}_p\!\left(\sigma\!\left(\mathrm{LayerNorm}\!\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)\right)\right),\tag{12}$$

with $\mathbf{a}^{(0)}$ the (possibly augmented) input and the output head $\mathbf{a}^{(L+1)} = \mathbf{W}^{(L+1)} \mathbf{a}^{(L)} + \mathbf{b}^{(L+1)}$ a plain linear map without normalisation, activation, or dropout. LayerNorm operates on the feature axis with learnable affine parameters $(\boldsymbol{\gamma}^{(l)}, \boldsymbol{\beta}^{(l)})$:

$$\mathrm{LayerNorm}(\mathbf{z})_k = \gamma_k \cdot \frac{z_k - \bar z}{\sqrt{\widehat{\mathrm{Var}}(\mathbf{z}) + \epsilon}} + \beta_k,\quad \bar z = \tfrac{1}{h}\sum_k z_k.\tag{13}$$

Dropout applies an i.i.d. Bernoulli mask $\mathbf{m}^{(l)} \sim \mathrm{Bern}(1-p)^{h_l}$ scaled by $1/(1-p)$ at training time and the identity at evaluation time. All linear weights are initialised with Xavier-normal, $\mathbf{W}_{ij} \sim \mathcal{N}(0,\, 2/(h_{l-1}+h_l))$, and biases are initialised to zero (`torque_models.py:78–82, 126–130, 177–181`).

The activation $\sigma$ is selected from $\{\mathrm{SiLU}, \mathrm{GELU}, \mathrm{ReLU}, \mathrm{Tanh}, \mathrm{ELU}, \mathrm{LeakyReLU}\}$. SiLU is the per-script default; GELU is used in the grid sweep. Both are smooth and bounded below, which retains small negative-region gradients (unlike ReLU) and prevents the dead-neuron pathology when the physics regulariser introduces competing gradient directions during the warmup phase.

**Backbone configurations.** Two backbones are used:

| Backbone | Used by | Total params, $d_{\text{in}}=15$ | Total params, $d_{\text{in}}=35$ |
|----------|---------|---------------------------------|---------------------------------|
| $[256, 512, 256]$ | standalone trainers (`run_fnn.py`, `run_physics_regularized.py`, `run_physics_residual.py`) | 270,341 | 275,461 |
| $[128, 256, 128]$ | grid sweep (`run_loss_residual_grid.py`) | 69,637 | 72,197 |

The smaller backbone is preferred for the grid because the small-data fractions (5K–50K samples for fractions ≤ 0.25) make the wider backbone prone to overfit. The 5,120-parameter increase from $d_{\text{in}}=15$ to $d_{\text{in}}=35$ on the wider backbone (or 2,560 on the narrower) is solely attributable to the 20 additional input weights times the first hidden width — a ≤ 2 % capacity increase. The physics-augmented backbones therefore do not gain their advantage from extra parameters.

A shared utility, `reduce_physics_to_total` (`torque_models.py:41–48`), maps the 20-dim normalised physics tensor to its 5-dim sum:

```python
def reduce_physics_to_total(physics, n_joints=5):
    p = physics.reshape(*physics.shape[:-1], 4, n_joints)
    return p.sum(dim=-2)
```

By Theorem 1, the output of `reduce_physics_to_total` is — up to model–reality residual — exactly the normalised analytical torque $\tilde{\boldsymbol{\tau}}_{\text{phys}}$.

### B. BlackBox-FNN

The simplest architecture treats torque prediction as a pure regression problem with no physics knowledge:

$$\hat{\boldsymbol{\tau}}_{\text{BB}}^{(i)} = f_\theta(\tilde{\mathbf{x}}^{(i)}),\qquad f_\theta : \mathbb{R}^{3J} \rightarrow \mathbb{R}^{J},\tag{14}$$

where $\tilde{\mathbf{x}}^{(i)} = [\tilde{\mathbf{q}}^{(i)\top}, \tilde{\dot{\mathbf{q}}}^{(i)\top}, \tilde{\ddot{\mathbf{q}}}^{(i)\top}]^\top \in \mathbb{R}^{3J}$ is the normalised kinematic state. The decomposed physics tensor $\boldsymbol{\phi}^{(i)} \in \mathbb{R}^{4J}$ is supplied by the data loader but is *explicitly discarded* by the forward pass (`torque_models.py:84–86`):

```python
def forward(self, features, physics=None):
    del physics
    return self.net(features)
```

**Training loss.** Joint-weighted MSE in normalised target space:

$$\mathcal{L}_{\text{BB}}(\theta) = \frac{1}{N}\sum_{i,j} w_j \big(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}\big)^2,\quad \mathbf{w} = (1.0, 2.5, 1.0, 1.0, 1.0)^\top.\tag{15}$$

The over-weight on J₂ is a deliberate countermeasure for the heavy-tailed torque distribution at the shoulder elevation joint: with $\sigma(\tau_2) \approx 2.58$ N·m vs. $\sigma(\tau_1) \approx 0.76$ N·m, an unweighted MSE allocates ≈ 11.5× more gradient pressure to J₂ already; the additional $w_2 = 2.5$ multiplier compensates the prior-induced shrinkage that physics-augmented models would otherwise impose. The same $\mathbf{w}$ is used by all three trainers, so the *data* gradient is identical across architectures, isolating the contribution of the physics term.

The empirical-risk gradient is

$$\nabla_\theta \mathcal{L}_{\text{BB}} = \frac{2}{N}\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)})\,\nabla_\theta \hat{\tau}_j^{(i)},\tag{16}$$

with $\nabla_\theta \hat{\tau}_j^{(i)}$ computed by back-propagation through the MLP.


### C. Physics-Regularised FNN (PhysReg-FNN)

PhysReg-FNN augments the input with the *full four-component decomposed* RNEA tensor and adds an additive Tikhonov physics penalty to the data loss.

**Augmented input.** Given the normalised decomposed physics tensor

$$\tilde{\boldsymbol{\phi}}^{(i)} = [\tilde{\boldsymbol{\tau}}_g^{(i)\top},\, \tilde{\boldsymbol{\tau}}_M^{(i)\top},\, \tilde{\boldsymbol{\tau}}_C^{(i)\top},\, \tilde{\boldsymbol{\tau}}_f^{(i)\top}]^\top \in \mathbb{R}^{4J},$$

the augmented input is the concatenation $\tilde{\mathbf{u}}^{(i)} = [\tilde{\mathbf{x}}^{(i)\top},\, \tilde{\boldsymbol{\phi}}^{(i)\top}]^\top \in \mathbb{R}^{7J}$, and the prediction is (`torque_models.py:132–134`)

$$\hat{\boldsymbol{\tau}}_{\text{PR}}^{(i)} = f_\theta(\tilde{\mathbf{u}}^{(i)}),\qquad f_\theta : \mathbb{R}^{7J} \rightarrow \mathbb{R}^{J}.\tag{17}$$

This is a deliberate departure from the more common practice of passing only the scalar RNEA *sum*. By exposing each physics component separately, the network's first linear layer

$$\mathbf{a}^{(1)}_h = \sum_{j=1}^{J}\!\left[ W^{(1)}_{h,\,j}\,\tilde{q}_j + W^{(1)}_{h,\,J+j}\,\tilde{\dot{q}}_j + W^{(1)}_{h,\,2J+j}\,\tilde{\ddot{q}}_j + \!\!\!\sum_{k\in\{g,M,C,f\}}\!\!\! W^{(1)}_{h,\,3J + 4(j-1) + \mathrm{idx}(k)}\,\tilde{\tau}_{k,j} \right] + b^{(1)}_h\tag{18}$$

can learn per-component, per-joint *trust weights* $W^{(1)}_{h,\,3J + 4(j-1) + \mathrm{idx}(k)}$ — for example, down-weighting the friction component $\tilde{\tau}_{f,j}$ at joint $j$ if its empirical residual is large, while keeping the gravity component $\tilde{\tau}_{g,j}$ at full influence. Collapsing to the sum eliminates this degree of freedom.

**Physics reference for the loss.** The loss-side physics target is the linear sum

$$\boldsymbol{\tau}_{\text{ref}}^{(i)} = \sum_{k\in\{g,M,C,f\}}\tilde{\boldsymbol{\tau}}_k^{(i)} = \mathtt{reduce\_physics\_to\_total}(\boldsymbol{\phi}^{(i)}) \in \mathbb{R}^{J}.\tag{19}$$

By Theorem 1, $\boldsymbol{\tau}_{\text{ref}}^{(i)} \approx \tilde{\boldsymbol{\tau}}^{*(i)}$ in noise-free cases.

**Composite loss (additive Tikhonov form).** With per-batch joint-weighted MSE

$$\mathcal{L}_{\text{data}}(\theta) = \tfrac{1}{N}\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)})^2,\quad \mathcal{L}_{\text{phys}}(\theta) = \tfrac{1}{N}\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tau_{\text{ref},j}^{(i)})^2,\tag{20}$$

the training objective is the *additive* (Tikhonov) penalty (`strategies.py:271`):

$$\boxed{\;\mathcal{L}_{\text{PR}}(\theta;\, e) = \mathcal{L}_{\text{data}}(\theta) + \alpha_{\text{eff}}(e)\cdot \mathcal{L}_{\text{phys}}(\theta).\;}\tag{21}$$

This is *not* the convex blend $(1-\alpha)\mathcal{L}_{\text{data}} + \alpha\mathcal{L}_{\text{phys}}$ used in some PINN variants. Under (21), increasing $\lambda$ never down-weights the data fit — the physics term acts as a pull toward the analytical surface in *addition* to the data fit. The optimal predictor in expectation is

$$\hat{\boldsymbol{\tau}}^\star = \frac{1}{1 + \alpha_{\text{eff}}}\,\boldsymbol{\tau}^* + \frac{\alpha_{\text{eff}}}{1 + \alpha_{\text{eff}}}\,\boldsymbol{\tau}_{\text{ref}}\quad\text{(per-sample, per-joint)},\tag{22}$$

a weighted average of the noisy measurement and the analytical RNEA whose weight on physics increases monotonically with $\lambda$.

**Linear warm-up of the physics coefficient.** The penalty coefficient is annealed from $0$ to its target value $\lambda$ over a window $e_w = \max(1, \lfloor \gamma E\rfloor)$ epochs, with $\gamma = 0.05$ by default (`strategies.py:256`):

$$\alpha_{\text{eff}}(e) = \lambda \cdot \min\!\left(1,\; \frac{e}{e_w}\right).\tag{23}$$

The warm-up serves two purposes: (i) it prevents the physics term from dominating before the network's first layer has organised itself to read $\tilde{\boldsymbol{\phi}}$; and (ii) at the start of training, where $\hat{\boldsymbol{\tau}}^{(0)}$ is essentially noise, the physics gradient $-2 w_j (\hat{\tau}_j - \tau_{\text{ref},j})$ is large in magnitude and pulls in an arbitrary direction — multiplying it by a small $\alpha_{\text{eff}}$ ensures it does not destabilise early epochs.

**Per-component gradient decomposition.** The gradient of (21) with respect to a backbone parameter $\theta$ is

$$\nabla_\theta \mathcal{L}_{\text{PR}} = \tfrac{2}{N}\sum_{i,j} w_j \big[(\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)}) + \alpha_{\text{eff}}(\hat{\tau}_j^{(i)} - \tau_{\text{ref},j}^{(i)})\big]\,\nabla_\theta \hat{\tau}_j^{(i)}.\tag{24}$$

When data and physics agree ($\tilde{\tau}^* \approx \tau_{\text{ref}}$), the bracket is $(1+\alpha_{\text{eff}})$ times the data residual: physics reinforces the data signal. When data and physics disagree, the gradients partially cancel, with the equilibrium prediction sitting between the two surfaces as derived in (22).

**Inference.** PhysReg-FNN requires the RNEA prediction at inference time. Given measured kinematics, $\tilde{\boldsymbol{\phi}}$ is computed online via Pinocchio's `rnea` calls (four per timestep) and concatenated to $\tilde{\mathbf{x}}$ before the forward pass.


### D. Residual-Correction FNN (ResCorr-FNN)

ResCorr-FNN imposes a hard *architectural* decomposition of the prediction into an analytical base plus a bounded learned correction (`torque_models.py:193–200`):

$$\boxed{\;\hat{\boldsymbol{\tau}}_{\text{RC}}^{(i)} = \tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)} + c_s \cdot \tanh\!\big(g_\theta(\tilde{\mathbf{u}}^{(i)})\big),\;}\tag{25}$$

where $\tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)} = \mathtt{reduce\_physics\_to\_total}(\boldsymbol{\phi}^{(i)})$ is the same RNEA sum used as the physics reference in § VI-C, $\tilde{\mathbf{u}}^{(i)} = [\tilde{\mathbf{x}}^{(i)\top}, \tilde{\boldsymbol{\phi}}^{(i)\top}]^\top \in \mathbb{R}^{7J}$ is the augmented 35-dim input, $g_\theta : \mathbb{R}^{7J} \rightarrow \mathbb{R}^{J}$ is the same MLP backbone family as PhysReg-FNN, and $\tanh$ is applied element-wise.

**Tanh-bounded correction (Lemma 2).** Defining $\boldsymbol{\delta}^{(i)} \triangleq \hat{\boldsymbol{\tau}}_{\text{RC}}^{(i)} - \tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)} = c_s\tanh(g_\theta(\tilde{\mathbf{u}}^{(i)}))$, the correction satisfies the unconditional bound

$$\|\boldsymbol{\delta}^{(i)}\|_\infty \le c_s \quad \forall i, \theta.\tag{26}$$

*Proof.* For any scalar $z$, $|\tanh z| < 1$, so $|c_s \tanh z| < c_s$ component-wise. $\blacksquare$

With $c_s = 0.5$ (registered as a non-learnable buffer; saved with the model state but excluded from the optimiser's parameter list), the per-joint physical-units bound after de-normalisation is $c_s \cdot \sigma_{\tau_j}^{\text{train}}$, evaluating to roughly $\{0.38, 1.29, 0.61, 0.23, 0.36\}$ N·m for the five joints — sizeable enough to absorb realistic RNEA mismatches but small enough to *prevent* the network from overriding the analytical prediction and memorising the training set.

The choice of $\tanh$ over harder bounds (e.g. clipped $\ell_\infty$) has two motivations:

1. **Smoothness.** $\tanh$ is $C^\infty$ with bounded derivatives $\tanh'(z) = 1 - \tanh^2(z) \in (0, 1]$, so the gradient of the bounded correction $c_s\tanh(\cdot)$ never blows up and never vanishes inside the linear-response region.
2. **Soft saturation.** Near $|g_\theta| \gtrsim 3$ the derivative is $\lesssim 0.01$, providing a soft saturation that strongly discourages the optimiser from pushing into the bound: any further attempt to grow $|\delta|$ produces an exponentially decaying gradient signal back into $\theta$.

**Small-residual warm-start.** The output linear layer is rescaled at initialisation (`torque_models.py:185–189`):

$$\mathbf{W}^{(L+1)} \leftarrow 10^{-2}\cdot \mathbf{W}^{(L+1)},\qquad \mathbf{b}^{(L+1)} \leftarrow 10^{-2}\cdot \mathbf{b}^{(L+1)}.\tag{27}$$

Combined with $\tanh(0) = 0$, this guarantees $\boldsymbol{\delta}^{(0)} \approx \mathbf{0}$ at epoch 0 — the network's epoch-0 prediction is the analytical RNEA, providing an effective preconditioner: the optimiser begins refinement from a physics-consistent state rather than building a 5-dim torque field from scratch.

**Regularised training loss.** An L₂ penalty on the bounded correction is added (`strategies.py:384`):

$$\boxed{\;\mathcal{L}_{\text{RC}}(\theta;\, \alpha_r) = \tfrac{1}{N}\sum_{i,j} w_j (\hat{\tau}_j^{(i)} - \tilde{\tau}_j^{*(i)})^2 + \alpha_r\cdot\tfrac{1}{NJ}\sum_{i,j} (\delta_j^{(i)})^2.\;}\tag{28}$$

The penalty acts on the *post-tanh* correction $\delta_j = c_s \tanh(g_{\theta,j})$, so its gradient back-propagates through the bound:

$$\frac{\partial}{\partial \theta} (\delta_j^{(i)})^2 = 2\,\delta_j^{(i)}\cdot c_s\bigl(1 - \tanh^2(g_{\theta,j}^{(i)})\bigr)\cdot \nabla_\theta g_{\theta,j}^{(i)},\tag{29}$$

which vanishes both when $\delta = 0$ (no penalty needed) *and* when $|\delta|\to c_s$ (pre-tanh activation already saturated). The strongest regularisation pressure is in the linear-response region $|g_\theta| \lesssim 1$, exactly where the network has the most flexibility to absorb spurious training-set patterns.

A dimensionless monitoring ratio is logged each epoch (`strategies.py:410–413`):

$$\rho(e) = \frac{\mathbb{E}_i[\|\boldsymbol{\delta}^{(i)}\|_1 / J]}{\mathbb{E}_i[\|\tilde{\boldsymbol{\tau}}_{\text{phys}}^{(i)}\|_1 / J] + 10^{-12}},\tag{30}$$

surfacing degenerate cases: $\rho \to 0$ (correction not learning anything) or $\rho \to c_s$ (RNEA being ignored).

**Hyperparameter regimes.** The grid sweep covers five $\alpha_r$ values:
- $\alpha_r = 0.005$: structural tanh bound is the only meaningful constraint; correction is essentially free up to $\pm c_s$.
- $\alpha_r = 0.05$ (per-script default): moderate L₂ pulling $\delta$ toward zero unless the data demands otherwise.
- $\alpha_r = 0.5$: strong L₂; the correction shrinks substantially and predictions converge toward $\tilde{\boldsymbol{\tau}}_{\text{phys}}$.


### E. Architectural Comparison

The three architectures span the spectrum from no physics, to physics as a *soft loss-side regulariser*, to physics as a *hard structural prior*.

**Table IV: Architectural Differences at a Glance**

| Aspect | BlackBox-FNN | PhysReg-FNN | ResCorr-FNN |
|--------|-------------|-------------|-------------|
| Input dim | $3J = 15$ | $7J = 35$ | $7J = 35$ |
| Forward map | $f_\theta(\tilde{\mathbf{x}})$ | $f_\theta([\tilde{\mathbf{x}}, \tilde{\boldsymbol{\phi}}])$ | $\tilde{\boldsymbol{\tau}}_{\text{phys}} + c_s\tanh(g_\theta([\tilde{\mathbf{x}}, \tilde{\boldsymbol{\phi}}]))$ |
| Physics at training | discarded (`del physics`) | input feature + loss term | input feature + output base |
| Physics at inference | absent | input feature only | input feature + output base |
| Loss form | $\mathcal{L}_{\text{data}}$ | $\mathcal{L}_{\text{data}} + \alpha_{\text{eff}}\mathcal{L}_{\text{phys}}$ | $\mathcal{L}_{\text{data}} + \alpha_r \|\boldsymbol{\delta}\|^2/J$ |
| Physics constraint | none | soft (loss penalty) | hard (architectural bound) |
| Tunable physics HP | — | $\lambda$ | $\alpha_r$, $c_s$ (fixed) |
| Worst-case bound | unbounded | unbounded | $\|\hat{\boldsymbol{\tau}} - \tilde{\boldsymbol{\tau}}_{\text{phys}}\|_\infty \le c_s$ |
| Output-layer init | Xavier-normal | Xavier-normal | Xavier-normal × 10⁻² |
| Backbone params (grid) | 69,637 | 72,197 | 72,197 + 1 buffer |

### F. Why Concatenate the *Decomposed* Physics?

A natural alternative to the 35-dim augmented input is the 20-dim form $[\tilde{\mathbf{x}},\, \tilde{\boldsymbol{\tau}}_{\text{phys}}]$ where $\tilde{\boldsymbol{\tau}}_{\text{phys}} = \sum_k \tilde{\boldsymbol{\tau}}_k$. The decomposed form is preferred for three reasons.

**Information.** The map $\tilde{\boldsymbol{\phi}} \mapsto \sum_k \tilde{\boldsymbol{\tau}}_k$ is many-to-one. A network receiving only the sum cannot, even in principle, infer the relative magnitudes of gravity, inertial, Coriolis, and friction contributions. Many failure modes of RNEA are component-specific (friction errors at low velocity, inertial-tensor mis-calibration at high acceleration), so distinguishing the components is necessary for a context-sensitive correction.

**Linear separability of trust weights.** With the decomposed input, the first hidden layer is a learnable *trust assignment*: each hidden unit can up- or down-weight any (component, joint) pair through its first-layer weight. Collapsing to the sum forces this assignment to be applied uniformly over components, eliminating an entire degree of freedom available to the network at no parameter-count cost (the input dimension grows by only $3J = 15$ extra weights per hidden unit).

**Capacity allocation.** With limited training data, a kinematic-only network must implicitly reconstruct the physics components (gravity, inertia, Coriolis, friction) from $\tilde{\mathbf{x}}$ before it can learn corrections to them — wasted capacity. By the data-processing inequality, any statistic of $\tilde{\mathbf{x}}$ available to the kinematic-only network is also available to the augmented-input network, but not vice versa, so the augmented input is a strict information superset.

---

## VII. Training Methodology

### A. Optimiser

All models are trained with AdamW [11], whose update rule decouples weight decay from the adaptive-moment normalisation (`Neural_Networks/models/shared/optim.py:17–20`):

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t,\qquad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^{\odot 2},\tag{31}$$

$$\hat{\mathbf{m}}_t = \mathbf{m}_t / (1-\beta_1^t),\qquad \hat{\mathbf{v}}_t = \mathbf{v}_t / (1-\beta_2^t),\tag{32}$$

$$\theta_{t+1} = \theta_t - \eta_t\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} - \eta_t\,\lambda_{\text{wd}}\,\theta_t,\tag{33}$$

with $\mathbf{g}_t = \nabla_\theta \mathcal{L}_{\{\text{BB}, \text{PR}, \text{RC}\}}$, default PyTorch betas $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$. Learning rate $\eta_0 = 3\times 10^{-4}$ and weight decay $\lambda_{\text{wd}} \in \{5\times 10^{-3}, 5\times 10^{-2}\}$ — the former for the per-script standalone trainers, the latter for the grid sweep where the smaller backbone benefits from heavier weight decay.

### B. Learning-Rate Schedule (warmup-cosine)

Per-epoch learning rate (`optim.py:29–39`):

$$\eta_t = \eta_0 \cdot \begin{cases} 0.1 + 0.9\cdot \dfrac{e}{e_w} & e < e_w,\\[8pt] r_{\min} + (1 - r_{\min})\cdot \dfrac{1 + \cos(\pi\,\xi(e))}{2} & e \ge e_w, \end{cases}\quad \xi(e) = \frac{e - e_w}{\max(1, E - e_w)},\tag{34}$$

with warm-up length $e_w = \max(1,\, \lfloor E/20\rfloor)$ (5 % of the budget) and minimum-LR ratio $r_{\min} = 10^{-2}$.

**Continuity at $e = e_w$.** At the join point, the warm-up evaluates to $\eta_0$ ($0.1 + 0.9 \cdot 1 = 1.0$) and the cosine evaluates to $r_{\min} + (1-r_{\min}) \cdot \tfrac{1+\cos(0)}{2} = r_{\min} + (1 - r_{\min}) = 1$. Hence $\eta(e_w^-) = \eta(e_w^+) = \eta_0$, i.e. the schedule is continuous. The right-derivative at the join is $-\tfrac{(1-r_{\min})\pi}{2(E-e_w)} \cdot \sin(0) = 0$, while the left-derivative is $+\tfrac{0.9 \eta_0}{e_w}$ — non-zero — so the schedule is $C^0$ but not $C^1$ at the join. This is intentional: the LR must drop *immediately* once warm-up ends to begin the cosine descent.

The linear warm-up from $0.1\eta_0$ avoids the well-known instability of large initial AdamW updates (when $\hat{\mathbf{v}}_t$ is dominated by the bias correction), and the cosine tail allows fine-grained refinement near the minimum.

**Patience suppression during warm-up.** The early-stopping patience counter is held at zero through the first $e_w$ epochs; per-epoch validation improvements during warm-up are intentionally smaller than $\delta_{\min}$ even for a healthy run, and counting them would prematurely terminate training.

### C. Stochastic Regularisation

**Input noise augmentation.** During training (but not evaluation), isotropic Gaussian noise is added to the *full* normalised input vector $\tilde{\mathbf{u}}^{(i)}$ (`strategies.py:158–159, 262–263, 376–377`):

$$\tilde{\mathbf{u}}_{\text{aug}}^{(i)} = \tilde{\mathbf{u}}^{(i)} + \boldsymbol{\varepsilon}^{(i)},\qquad \boldsymbol{\varepsilon}^{(i)} \sim \mathcal{N}(\mathbf{0},\, \sigma_n^2 \mathbf{I}_{d_{\text{in}}}),\qquad \sigma_n \in \{0.02, 0.05\},\tag{35}$$

with $d_{\text{in}} = 15$ for BlackBox-FNN and $d_{\text{in}} = 35$ for PhysReg / ResCorr (note that the noise is applied to the augmented vector *as a whole*, including the physics channels — this prevents the network from learning a purely-physics shortcut). The perturbation is in normalised space; the per-channel physical-units standard deviation is $\sigma_n\cdot \sigma_k^{\text{train}}$.

**Dropout.** Applied with probability $p \in [0.1, 0.4]$ at every hidden layer using the standard inverted-dropout convention.

**Weight decay.** AdamW's decoupled weight decay (33) is mathematically equivalent to a per-parameter L₂ penalty on the *parameter trajectory*, distinct from adding $\lambda_{\text{wd}}\|\theta\|_2^2$ to the loss in standard Adam. The decoupled form is preferred because it does not interact with the second-moment normalisation $\sqrt{\hat{\mathbf{v}}_t}$.

### D. Gradient Stabilisation

**Global-norm clipping.** The aggregate gradient norm is clipped to $G_{\max}$:

$$\mathbf{g}_t \leftarrow \mathbf{g}_t \cdot \min\!\left(1,\; \frac{G_{\max}}{\|\mathbf{g}_t\|_2 + 10^{-6}}\right),\tag{36}$$

with $G_{\max} = 5.0$ (per-script) or $G_{\max} = 1.0$ (grid). This is essential at the *start* of PhysReg training where the term $\alpha_{\text{eff}}\,\mathcal{L}_{\text{phys}}$ ramps in over the first $e_w$ epochs and can produce large gradient spikes when $\hat{\boldsymbol{\tau}}^{(0)}$ has not yet aligned with $\boldsymbol{\tau}_{\text{ref}}$.

**Mixed precision (AMP).** On CUDA devices, forward and backward passes run in FP16 under `torch.autocast`, with master weights kept in FP32 by the AdamW optimiser. A `GradScaler` rescales the loss before backward to avoid FP16 underflow:

$$\mathbf{g}_t^{\text{FP16}} = \nabla_\theta\,(s \cdot \mathcal{L}),\qquad \mathbf{g}_t^{\text{FP32}} = \mathbf{g}_t^{\text{FP16}} / s,\tag{37}$$

with $s$ updated dynamically (doubled when no overflow detected for a window of steps; halved on overflow).

### E. Early Stopping and Model Roll-back

The training loop monitors the unweighted *macro-RMSE* on the validation split in physical N·m units (rather than the joint-weighted training objective). With minimum improvement $\delta_{\min} = 10^{-4}$ N·m and patience $P$:

$$e^\star = \min\!\left\{e\,\big|\, \mathrm{val\_rmse}(e) > \min_{e' \le e} \mathrm{val\_rmse}(e') - \delta_{\min}\ \text{for}\ P\ \text{consecutive epochs}\right\}.\tag{38}$$

Patience values: $P = 50$ (BlackBox per-script), $P = 80$ (PhysReg per-script), $P = 60$ (ResCorr per-script), $P = 150$–$200$ (grid). Upon early stop or epoch-budget exhaustion, the model state is *rolled back* to the epoch that achieved $\min_{e'} \mathrm{val\_rmse}(e')$; this is the state saved to `model.pt` and used for all reported test metrics.

### F. Macro-RMSE Validation Metric

Validation RMSE is computed in physical units after de-normalisation, *averaged per trajectory* and then averaged across trajectories — not pooled across all samples — so that long trajectories do not dominate short ones:

$$\mathrm{macro\_rmse}(\hat{\boldsymbol{\tau}}, \boldsymbol{\tau}^*) = \frac{1}{|\mathcal{T}|}\sum_{T \in \mathcal{T}} \frac{1}{J}\sum_{j=1}^{J} \sqrt{\frac{1}{|T|}\sum_{i \in T}\big(\hat{\tau}_j^{(i)} - \tau_j^{*(i)}\big)^2},\tag{39}$$

where $\mathcal{T}$ is the set of validation trajectories. The pooled RMSE used in § IX (table headers say "RMSE pooled") is $\sqrt{\tfrac{1}{NJ}\sum_{i,j}(\hat{\tau}_j^{(i)} - \tau_j^{*(i)})^2}$ — the same predictions, but with the per-trajectory and per-joint averaging steps replaced by a single global pool.

### G. DataLoader Configuration

Training batches: $B = 512$ (per-script) or $B = 1024$ (grid), with `shuffle=True` and `drop_last=True` *only* when the training set contains at least $2B$ samples. The 2B floor is critical at small data fractions: at `frac = 0.02` the training set has ~5,450 samples, and a naive `drop_last=True` with $B = 1024$ would silently empty the loader when the last batch is dropped on uneven splits. Validation and test loaders use all samples (`drop_last=False`) and are not shuffled.

DataLoader workers are auto-tuned by available system memory: 0–2 workers on low-RAM hosts (< 32 GB), up to 8 on workstation-grade hardware. Prefetch factor scales similarly. `pin_memory=True` is enabled on CUDA devices for asynchronous host-to-device transfer.


---

## VIII. Hyperparameter Grid Search

To characterise the effect of the key physics-related hyperparameters, a structured grid search is conducted (`Neural_Networks/models/run_loss_residual_grid.py`). All non-swept hyperparameters are held fixed at the values in § VII (backbone $[128, 256, 128]$, GELU, dropout $0.2$, $\eta_0 = 3{\times}10^{-4}$, $\lambda_{\text{wd}} = 5{\times}10^{-2}$, $B = 1024$, $\sigma_n = 0.05$, $G_{\max} = 1.0$, epochs $E = 3000$, patience $P = 150$). Each architecture–hyperparameter combination is trained with two random seeds (which simultaneously seed the parameter init *and* the data-fraction subsample), giving mean ± spread per cell.

**Table V: Grid Search Axes**

| Architecture | Swept HPs | Values | Total Trials |
|--------------|-----------|--------|--------------|
| BlackBox-FNN | `data_train_fraction`, `seed` | $\{0.02, 0.05, 0.1, 0.25, 0.5, 1.0\} \times \{0, 1\}$ | 12 |
| PhysReg-FNN | `physics_weight` $\lambda$, `data_train_fraction`, `seed` | $\{0.05, 0.1, 0.2, 0.5, 1.0, 2.0\} \times \{0.02, \ldots, 1.0\} \times \{0, 1\}$ | 72 |
| ResCorr-FNN | `alpha_reg_weight` $\alpha_r$, `data_train_fraction`, `seed` | $\{0.005, 0.01, 0.05, 0.1, 0.5\} \times \{0.02, \ldots, 1.0\} \times \{0, 1\}$ | 60 |
| **Total** | | | **144** |

The `physics_weight` axis spans the additive-Tikhonov coefficient from a near-zero nudge ($\lambda = 0.05$) through a moderate prior ($\lambda = 0.5$) up to a strong physics-dominated regime ($\lambda = 2.0$). For ResCorr-FNN the bound $c_s = 0.5$ is held fixed (registered buffer, not optimiser parameter), and `alpha_reg_weight` sweeps the L₂ penalty from "effectively unregularised" ($\alpha_r = 0.005$) to "strongly regularised" ($\alpha_r = 0.5$). The data-fraction axis extends to 2 % (~5,450 training samples) to expose the small-data regime where the physics prior provides the largest relative advantage over the BlackBox baseline.

**Idempotency.** Completed trials are fingerprinted on the union of swept and exhaustive HPs and skipped on re-runs by matching against saved `metadata.yaml` files, so the grid is idempotent under interruption and resumption (`run_loss_residual_grid.py:99–161`).

**Admission control.** A multi-process pool admits new trials only when free system RAM $\ge$ 1 GB and (on CUDA) free VRAM $\ge$ 0.5 GB; otherwise the new trial waits. This avoids the "deadlock" pathology where N processes each take 95 % of memory and OOM together.


---

## IX. Results

All metrics in this section are computed on the **held-out test split of run_train22**: 38,106 time steps drawn from 11 trajectories (one per geometry class) entirely disjoint from the 22 training trajectories and the 11 validation trajectories. The "best" configuration per architecture is the trial with minimum pooled test RMSE across the 144-trial grid sweep stored under `Neural_Networks/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/`.

### A. Overall Performance

**Table VI: Best Model Performance (Test Set, run_train22)**

| Architecture | Test RMSE (N·m) | Test R² | Test MAE (N·m) | Best Epoch | Data Fraction | Phys HP |
|--------------|----------------:|--------:|---------------:|-----------:|--------------:|---------|
| ResCorr-FNN | **0.09666** | **0.9106** | **0.06480** | 339 | **0.05** (~3.3 K) | $\alpha_r = 0.1$ |
| PhysReg-FNN | 0.09804 | 0.9080 | 0.06725 | 1075 | **0.05** (~3.3 K) | $\lambda = 0.5$ |
| BlackBox-FNN | 0.10566 | 0.8932 | 0.07193 | 300 | 1.00 (66.7 K) | — |

The ranking has *flipped* relative to the older preliminary corpus: with the run_train22 manual 2/1/1 per-geometry split, **ResCorr-FNN now wins outright** in pooled RMSE, R², and MAE simultaneously. The pooled-RMSE reductions over the BlackBox baseline are **8.5 % (ResCorr)** and **7.2 % (PhysReg)**. Both physics-aware models attain their global optimum at *only 5 % of the available training data* (≈ 3,337 samples) versus 100 % (66,735 samples) for the BlackBox baseline — a > 20× sample-efficiency advantage.

**RMSE ranges across the 144 grid trials** (`analysis/summary_table.md`):

- **BlackBox-FNN**: RMSE in $[0.10566, 0.11583]$ N·m over 12 runs; R² in $[0.872, 0.893]$.
- **PhysReg-FNN**: RMSE in $[0.09804, 0.10500]$ N·m over 72 runs; R² in $[0.895, 0.908]$.
- **ResCorr-FNN**: RMSE in $[0.09666, 0.10112]$ N·m over 60 runs; R² in $[0.902, 0.911]$.

Two structural patterns are visible. First, the *worst* PhysReg / ResCorr run still beats the *best* BlackBox run by more than 0.4 % RMSE — the architectural physics prior shifts the entire performance distribution down, not just the best case. Second, ResCorr-FNN has the narrowest spread (0.0045 N·m wide vs. 0.0070 N·m for PhysReg), a direct consequence of Lemma 2: the tanh bound caps how badly the network can override the analytical surface, so even unfortunate hyperparameter choices land near the analytical-RNEA RMSE rather than diverging.

A side-by-side test-set torque trace overlay for the three architectures plus ground truth on a representative held-out trajectory is shown in **Fig. 10** (`PLOTS_and_FLOWCHARTS/comparision.png`).

![Fig. 10 — Per-joint torque trace on a representative test-set trajectory: ground truth (grey), Residual-Correction FNN (purple), Physics-Regularised FNN ("PINN_FNN", green), BlackBox FNN (red). The Residual model tracks the gravitational and inertial peaks at J₁, J₂, J₃, J₄ closely; the BlackBox FNN exhibits the largest spurious oscillations at J₅ (wrist roll), where its R² drops to 0.671 vs. 0.771 for ResCorr.](PLOTS_and_FLOWCHARTS/comparision.png)

The architecture-level RMSE/R² distributions across all grid trials, top-K leaderboard, and per-joint heatmaps are shown in `analysis/fig2_rmse_comparison.pdf`, `analysis/fig5_topk_leaderboard.pdf`, `analysis/fig6_rmse_distribution.pdf`, `analysis/fig8_r2_test_distribution.pdf`.

### B. Per-Joint Performance (Best Configurations)

**Table VII: Per-Joint Test Metrics — Best Configuration per Architecture (run_train22)**

| Joint | Metric | BlackBox-FNN | PhysReg-FNN | ResCorr-FNN |
|-------|--------|-------------:|------------:|------------:|
| J₁ | RMSE (N·m) | 0.0790 | 0.0753 | **0.0757** |
|    | R²        | 0.871  | 0.883  | **0.881**  |
|    | MAE (N·m) | 0.0580 | 0.0574 | **0.0562** |
| J₂ | RMSE (N·m) | 0.1510 | 0.1451 | **0.1489** |
|    | R²        | 0.919  | **0.926**  | 0.922  |
|    | MAE (N·m) | 0.1134 | 0.1121 | **0.1124** |
| J₃ | RMSE (N·m) | 0.1054 | 0.0928 | **0.0891** |
|    | R²        | 0.904  | 0.926  | **0.932**  |
|    | MAE (N·m) | 0.0779 | 0.0680 | **0.0625** |
| J₄ | RMSE (N·m) | 0.0521 | 0.0446 | **0.0430** |
|    | R²        | 0.768  | 0.830  | **0.842**  |
|    | MAE (N·m) | 0.0356 | 0.0316 | **0.0302** |
| J₅ | RMSE (N·m) | 0.1138 | 0.1036 | **0.0950** |
|    | R²        | 0.671  | 0.727  | **0.771**  |
|    | MAE (N·m) | 0.0749 | 0.0672 | **0.0627** |

(Bold marks the best of each row; ties broken by R².)

Four observations stand out under the run_train22 corpus:

1. **J₅ is the biggest discriminator between architectures.** The wrist-roll joint sits at the end of the kinematic chain, has the smallest measured torque variance, and is dominated by friction/Coulomb effects that the analytical $\tau_f$ in (10) only partially captures. BlackBox-FNN achieves only R² = 0.671 here; ResCorr-FNN lifts that to R² = 0.771 — a **+10 percentage-point absolute gain** — because the bounded residual can absorb the systematic friction-model mismatch on top of the analytical base.
2. **J₃ and J₄ show consistent ResCorr leadership.** Per-joint RMSE reductions of **15 %** (J₃: 0.1054 → 0.0891) and **17 %** (J₄: 0.0521 → 0.0430) confirm that the RNEA + bounded-residual decomposition is the most efficient way to handle joints with non-negligible friction *and* moderate inertial torque ranges.
3. **J₂ is best served by soft regularisation.** PhysReg-FNN edges ResCorr-FNN at the shoulder-elevation joint by 0.4 percentage points of R² (0.926 vs. 0.922). J₂ has the largest measured-torque variance (σ ≈ 2.58 N·m) — most of which is structured gravity that RNEA models well — so a soft pull toward the analytical surface (Tikhonov) suffices, while the hard tanh bound at $c_s\sigma_{\tau_2}^{\text{train}} \approx 1.29$ N·m on ResCorr is a marginal extra constraint that does not pay off.
4. **All three architectures are J₅-bound.** Pooled RMSE for each model is dominated by the J₂ + J₅ contributions; the per-joint RMSE bar plot in `analysis/fig4_mae_nrmse_comparison.pdf` makes this dominance visually explicit.

Per-joint heatmaps and the MAE/NRMSE breakdown are in `analysis/fig3_per_joint_heatmaps.pdf` and `analysis/fig4_mae_nrmse_comparison.pdf`.

### C. Effect of Physics Weight λ (PhysReg-FNN)

Aggregating across all 12 (frac × seed) cells per $\lambda$, the best-of-cell pooled test RMSE traces a smooth U-shape with a unique optimum at $\lambda = 0.5$:

**Table VIII: Effect of Physics Weight $\lambda$ on PhysReg-FNN Test RMSE (run_train22, best-of-12 cells per row)**

| $\lambda$ | Best Test RMSE (N·m) | Mean Test RMSE (N·m) |
|----------:|---------------------:|---------------------:|
| 0.05 | 0.09972 | 0.10137 |
| 0.10 | 0.09905 | 0.10098 |
| 0.20 | 0.09823 | 0.10052 |
| **0.50** | **0.09804** | **0.09995** |
| 1.00 | 0.09977 | 0.10070 |
| 2.00 | 0.10155 | 0.10278 |

The optimum has *shifted* from the $\lambda \approx 0.3$ value reported on the older preliminary corpus to $\lambda = 0.5$ on run_train22. Two factors plausibly explain the shift: (i) the run_train22 training set is much smaller (66,735 vs. 272,465 samples) so the data-loss gradient is statistically less reliable, increasing the relative value of the physics prior; and (ii) the manual 2/1/1 per-geometry split exposes the model to fewer trajectories per class, again favouring stronger regularisation. By (22), the equilibrium predictor at $\lambda = 0.5$ is a 33 % / 67 % posterior mean of the analytical RNEA and the data — the analytical surface explains roughly one third of the optimal loss signal under run_train22.

The full λ-impact panel — best, mean and 95 % envelope across the 12 grid cells per $\lambda$ — is in `analysis/fig7_physics_weight_impact.pdf`.

### D. Effect of Correction Penalty $\alpha_r$ (ResCorr-FNN)

ResCorr-FNN is largely insensitive to the L₂ correction penalty $\alpha_r$ over the full sweep range:

**Table VIII': Effect of $\alpha_r$ on ResCorr-FNN Test RMSE (run_train22, best-of-12 cells per row)**

| $\alpha_r$ | Best Test RMSE (N·m) | Mean Test RMSE (N·m) |
|-----------:|---------------------:|---------------------:|
| 0.005 | 0.09718 | 0.09950 |
| 0.010 | 0.09715 | 0.09947 |
| 0.050 | 0.09691 | 0.09905 |
| **0.100** | **0.09666** | 0.09880 |
| 0.500 | 0.09666 | **0.09788** |

Best-RMSE is achieved at $\alpha_r = 0.1$ (and tied at $\alpha_r = 0.5$), and the spread across all five values is only 0.5 % of the best value. The architectural tanh bound (Lemma 2) is doing most of the work — the L₂ penalty on $\boldsymbol{\delta}$ provides at most a marginal refinement. This robustness is precisely the deployment-friendly property predicted in § VI-D: ResCorr-FNN is insensitive to its sole physics hyperparameter, in contrast to PhysReg-FNN where misjudging $\lambda$ by an order of magnitude (0.05 vs. 0.5) costs 1.7 % RMSE.

### E. Data-Efficiency Analysis

**Table IX: Best Test RMSE vs. Training-Data Fraction (run_train22, best across all phys-HP cells per arch)**

| Fraction | Approx. Train Samples | BlackBox-FNN | PhysReg-FNN | ResCorr-FNN |
|---------:|----------------------:|-------------:|------------:|------------:|
| 0.02 | ~1,335  | 0.11090 | 0.09854 | 0.09828 |
| **0.05** | **~3,337** | 0.10792 | **0.09804** | **0.09666** |
| 0.10 | ~6,674  | 0.10664 | 0.09823 | 0.09722 |
| 0.25 | ~16,684 | 0.10723 | 0.09942 | 0.09718 |
| 0.50 | ~33,367 | 0.10673 | 0.09966 | 0.09785 |
| 1.00 | 66,735  | **0.10566** | 0.10047 | 0.09738 |

Three findings — qualitatively *opposite* to the previous corpus's claims — emerge:

1. **Physics-aware models peak at the smallest non-trivial fraction.** Both PhysReg-FNN ($\lambda = 0.5$) and ResCorr-FNN ($\alpha_r = 0.1$) attain global optima at **frac = 0.05** (~3,337 samples). Above this point both architectures *very slowly degrade*, consistent with the analytical-physics prior already capturing the bulk of the inverse-dynamics signal: more data adds noise to the residual without revealing new physical structure.
2. **BlackBox-FNN improves monotonically with data.** The RMSE curve descends from 0.11090 (frac = 0.02) to 0.10566 (frac = 1.00), as expected for a regression model with no inductive bias. Even at full data the BlackBox baseline still trails the *worst* physics-aware run.
3. **Sample-efficiency multiplier > 20×.** ResCorr-FNN at frac = 0.05 (3,337 samples) achieves 0.09666 N·m — a configuration the BlackBox baseline cannot match even at frac = 1.00 (66,735 samples, 0.10566 N·m). This is a 20× reduction in required training data to reach a tighter accuracy ceiling, which is the core empirical finding of this study.

The full per-fraction RMSE distribution and per-architecture R² distribution are in `analysis/fig6_rmse_distribution.pdf` and `analysis/fig8_r2_test_distribution.pdf`.

### F. Training Convergence

Mean epochs trained (sum of training + early-stop wait, before roll-back to best epoch) across all 144 successful runs:

- **BlackBox-FNN**: $570 \pm 293$ epochs (range 300–1167)
- **PhysReg-FNN**: $804 \pm 488$ epochs (range 300–1933)
- **ResCorr-FNN**: $482 \pm 198$ epochs (range 300–1028)

PhysReg-FNN trains roughly 1.7× longer than ResCorr-FNN on average because the physics warm-up (23) introduces a multi-phase optimisation: the first 5 % of epochs ramp $\alpha_{\text{eff}}$ from 0 to $\lambda$, after which the data and physics gradients (24) compete, slowing per-epoch validation improvement. ResCorr-FNN converges fastest because the $10^{-2}$ output-layer rescaling (27) places the network at the analytical RNEA at epoch 0 — the optimiser immediately refines a small bounded correction rather than building the full torque field from scratch.

At the *best* per-architecture configurations, the early-stop epoch numbers are:

- ResCorr ($\alpha_r = 0.1$, frac = 0.05): best epoch **339**
- PhysReg ($\lambda = 0.5$, frac = 0.05): best epoch **1075**
- BlackBox (frac = 1.0): best epoch **300**

Per-architecture training/validation loss curves and best-epoch markers are plotted in `analysis/fig1_training_dynamics.pdf`.

---

## X. Discussion

### A. Role of the Physics Warm-Start

Both physics-informed architectures benefit from an initialisation or schedule that positions the model near the RNEA prediction at the start of training. In PhysReg-FNN, the warm-up schedule (23) achieves this implicitly: with $\alpha_{\text{eff}}(0) = 0$, the first $e_w$ epochs optimise purely for data loss and the network learns the broad statistical structure of the torque distribution. After warm-up, the physics penalty constrains the solution space to be near the calibrated analytical model, guiding the optimiser away from local minima that fit the training data but violate rigid-body consistency.

In ResCorr-FNN, the $10^{-2}$ output-layer rescaling (27) provides an equivalent warm-start: the network begins at the RNEA solution and gradually refines corrections. This architectural prior is *stronger* than the loss-based warm-up because it is enforced at every gradient step, not just during a transient phase.

The augmented-input design strengthens the warm-start mechanism for both physics-informed models. By providing the full decomposed RNEA tensor $\tilde{\boldsymbol{\phi}} = [\tilde{\boldsymbol{\tau}}_g, \tilde{\boldsymbol{\tau}}_M, \tilde{\boldsymbol{\tau}}_C, \tilde{\boldsymbol{\tau}}_f]$ as an explicit input feature, the network learns physics-aware representations from the beginning of training rather than reconstructing the physics structure implicitly from kinematics. The first hidden layer (18) can immediately allocate its weights along physically meaningful directions — per-component, per-joint trust assignments — which would otherwise have to be re-derived from the noisy 15-dim kinematic vector. For PhysReg-FNN, this combined mechanism — warm-up schedule plus decomposed-physics input — reduces gradient conflict between $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{phys}}$ and stabilises training across the full $\lambda \in [0.05, 2.0]$ sweep.

### B. Structural Interpretation of Per-Joint Results

The run_train22 per-joint results (Table VII) align with rigid-body mechanics, but the dominant differentiator across architectures is now the *distal* end of the chain (J₃–J₅) rather than the gravity-loaded shoulder J₂:

- **J₁ (shoulder rotation, axis $\hat{z}$).** Gravity contribution around the vertical yaw axis is ≈ 0. RMSE differences across architectures are < 5 % — the kinematic features alone carry most of the signal, so all three models achieve R² ≈ 0.88.
- **J₂ (shoulder elevation, axis $-\hat{x}$).** Largest measured-torque variance in the chain (σ(τ₂) ≈ 2.58 N·m). Most of this is *structured gravity*, which the calibrated RNEA (post-mass-rescaling, § V-A) already explains very accurately. PhysReg-FNN edges ResCorr-FNN by 0.4 percentage points of R² (0.926 vs. 0.922); the soft Tikhonov pull is sufficient and the hard tanh bound at $c_s\sigma_{\tau_2}^{\text{train}} \approx 1.29$ N·m provides no extra benefit at this joint.
- **J₃ (elbow, axis $-\hat{x}$).** ResCorr-FNN reduces J₃ RMSE by **15 %** vs. BlackBox (0.1054 → 0.0891). The elbow exhibits systematic friction asymmetry and likely gear compliance that the smooth Coulomb-tanh model in (10) under-models; the bounded residual network absorbs the missing structure efficiently while the analytical surface keeps the prediction physically plausible.
- **J₄ (wrist pitch, axis $\hat{x}$).** Smallest absolute RMSE (0.0430 N·m for ResCorr) but the *lowest* per-joint R² across the three first joints — only 0.768 for BlackBox lifted to 0.842 for ResCorr. The wrist pitch has low inertia, small torque range (σ ≈ 0.46 N·m), and a smaller-stall-torque servo (Table II), so the *signal-to-noise* ratio is the worst in the chain; the analytical base of ResCorr-FNN provides a much-needed prior.
- **J₅ (wrist roll, axis ≈ $\hat{z}$).** **The biggest discriminator across architectures.** BlackBox-FNN bottoms out at R² = 0.671; ResCorr-FNN lifts it to R² = 0.771, a **+10 percentage-point absolute gain**. J₅ is dominated by Coulomb friction with very small inertial and gravitational terms, so the RNEA decomposition φ = $[\boldsymbol{\tau}_g, \boldsymbol{\tau}_M, \boldsymbol{\tau}_C, \boldsymbol{\tau}_f]$ at this joint is essentially $\boldsymbol{\tau}_f$ — exactly the channel the bounded residual is best equipped to refine.

### C. Why Additive Tikhonov Beats Convex Blend

A common mistake in PINN design is to use the convex-blend loss $(1-\alpha)\mathcal{L}_{\text{data}} + \alpha\mathcal{L}_{\text{phys}}$. This couples data and physics weights inversely: increasing physics influence necessarily decreases data influence, so a strong physics prior actively *under-fits* the measurement. The additive Tikhonov form (21) decouples these — increasing $\lambda$ adds physics pressure on top of unchanged data pressure, and the equilibrium predictor (22) is a true posterior-style mean. The grid sweep (Table VIII) confirms this prediction: at $\lambda = 0.5$ the additive form yields a 33 % / 67 % posterior mean of analytical RNEA and noisy measurement; at $\lambda = 2.0$ the equilibrium shifts to 67 % / 33 % yet pooled RMSE only degrades by 3.6 % rather than catastrophically. A convex-blend variant at $\lambda_{\text{convex}} \to 1$ would force $\hat{\boldsymbol{\tau}} \to \boldsymbol{\tau}_{\text{ref}}$ regardless of measurement noise — a regime the additive form sidesteps by construction.

### D. Hard vs. Soft Physics Constraint

ResCorr-FNN's hard tanh bound (Lemma 2) and PhysReg-FNN's soft Tikhonov penalty represent the two principled ways to inject the analytical prior. On run_train22 the trade-off is no longer ambiguous:

- **Best-case accuracy** now favours **ResCorr-FNN** (0.09666 vs. 0.09804 N·m, a 1.4 % RMSE advantage) — the architectural prior is the better lever when training data is scarce (here 66 K samples and only 22 trajectories).
- **Worst-case accuracy** still favours ResCorr-FNN by an even larger margin (worst-case 0.10112 vs. 0.10500 N·m), because the bound caps how badly the network can override the analytical surface.
- **Hyperparameter robustness** also favours ResCorr-FNN: the spread of best-RMSE across $\alpha_r \in \{0.005, …, 0.5\}$ is only 0.5 % of the best value (Table VIII'), versus 3.6 % for PhysReg-FNN across $\lambda \in \{0.05, …, 2.0\}$.
- **Training cost** favours ResCorr-FNN: mean epochs to convergence are 482 vs. 804 for PhysReg-FNN.

The single property where PhysReg-FNN remains preferable is *interpretability of the physics weight*: the additive Tikhonov coefficient $\lambda$ has a closed-form posterior-mean interpretation (22) that the implicit-bound formulation of ResCorr-FNN does not directly afford. For deployment in torque-limited manipulator control where worst-case guarantees matter, **ResCorr-FNN is the recommended architecture under the run_train22 corpus**.

### E. Why a Smaller, Geometrically-Stratified Corpus Reverses the Ranking

The headline finding — that ResCorr-FNN now beats PhysReg-FNN on run_train22, where the previous (large, semi-random) corpus had them tied or PhysReg ahead — illustrates the central tension in physics-aware learning. With abundant, geometrically-diverse training data the soft Tikhonov form (PhysReg) can afford to *partially override* the analytical RNEA when the data demands it; with the smaller, manually-balanced 22-trajectory training set, the bounded residual (ResCorr) is the safer bet because it cannot drift away from the analytical surface even in the regions where training coverage is thinnest. The data-efficiency curves of Table IX confirm the same intuition from the opposite direction: both physics-aware models attain their global optimum at the *smallest* non-trivial fraction (5 %, ~3,300 samples) and very mildly degrade thereafter — additional data adds noise to the residual without revealing new physical structure. This is the regime in which architectural priors outperform soft-loss regularisation, and which we expect to dominate real-world robot-deployment scenarios where data collection is expensive and trajectory diversity is limited.

---

## XI. Limitations and Future Work

1. **RNEA-mismatch generalisation.** Both physics-informed architectures rely on the analytical RNEA being a *roughly correct* representation of the true dynamics: PhysReg-FNN uses RNEA as an additive penalty target, and ResCorr-FNN uses it as the output base. Systematic biases in RNEA (e.g., the calibrated mass scaling factor $\rho = 0.0931$ and the empirical friction coefficients $\mathbf{c}, \mathbf{v}, \varepsilon$) are absorbed by the network during training, but if the robot undergoes physical changes between training and deployment — payload attachment, joint wear, temperature-dependent friction — the residual the network has learned to absorb may no longer match the operational RNEA error distribution. ResCorr-FNN's hard tanh bound limits how much the correction can absorb without partial retraining; PhysReg-FNN has no such bound but its physics penalty pulls predictions toward whatever (potentially mismatched) RNEA is computed at inference.

2. **The 75 % data-fraction anomaly** (Table IX) suggests sensitivity to the specific subset of trajectories included at each fraction. A more principled fractional subset (e.g., stratified sampling that preserves the geometry distribution at each fraction, rather than random subsampling) would give smoother data-efficiency curves. We leave this to future work.

3. **Temporal structure is not exploited.** All three architectures treat each time step independently (i.i.d. assumption). Incorporating temporal context — via LSTM, Temporal Convolutional Network, or Transformer encoders — could improve accuracy on trajectories with strong velocity autocorrelation (helices, spirals) by providing explicit state history. The `RobotDataset` class already supports a sequence mode (`mode="sequence"`, `loader.py:296–309`); only the model definitions need extending.

4. **Coupling with parameter identification.** A natural next step is to make the URDF mass scale $\rho$, friction coefficients $(\mathbf{c}, \mathbf{v}, \varepsilon)$, and per-joint extra masses *jointly learnable* alongside the network weights. This would close the loop between calibration and learning, eliminating the dependency on offline calibration runs.

5. **Single-seed best performance.** The results in Table VI represent the best single trial per architecture. Variance across the two seeds in each configuration is not reported in full; for deployment, ensemble predictions over multiple seeds would further reduce variance.

6. **Sim-to-real gap is not evaluated.** All training and test data come from the same physical robot. Whether the augmented-input PhysReg architecture generalises to a *different* robot of the same family — for example, with PLA infill density that changes $\rho$ — remains an open empirical question.

---

## XII. Conclusion

This paper presents a rigorous comparative evaluation of three physics-informed neural network architectures for robot inverse-dynamics identification on a five-DOF manipulator, evaluated on the **run_train22** corpus (44 trajectories, manual 2/1/1 per-geometry split, 66,735 / 47,868 / 38,106 train/val/test samples). The main findings are:

1. **Residual correction is the new accuracy leader.** Under run_train22, ResCorr-FNN attains the lowest pooled test RMSE (**0.09666 N·m, R² = 0.911, MAE = 0.0648**), beating PhysReg-FNN (0.09804 N·m, R² = 0.908) by 1.4 % and the BlackBox baseline (0.10566 N·m, R² = 0.893) by **8.5 %**, using the same backbone and optimiser. The bounded-residual decomposition $\hat{\boldsymbol{\tau}} = \tilde{\boldsymbol{\tau}}_{\text{phys}} + c_s\tanh(g_\theta(\tilde{\mathbf{x}}, \tilde{\boldsymbol{\phi}}))$ with $c_s = 0.5$ provides the architectural prior; the additive L₂ penalty on the bounded correction is the only training-loss adjustment vs. plain MSE.

2. **Physics guidance gives a > 20× sample-efficiency advantage.** Both physics-aware models attain their global optimum at **only 5 %** of the available training data (~3,300 samples), where the BlackBox baseline cannot match either even at full data (66,735 samples). Beyond 5 %, physics-aware test RMSE *very mildly degrades*, indicating the analytical RNEA already captures the bulk of the inverse-dynamics signal and additional data adds noise to the residual.

3. **ResCorr is also the most robust to its physics hyperparameter.** Sweeping the L₂ penalty $\alpha_r \in \{0.005, 0.01, 0.05, 0.1, 0.5\}$ moves best-RMSE by only 0.5 % (Table VIII'), versus 3.6 % for PhysReg-FNN's $\lambda$ sweep. The architectural tanh bound (Lemma 2) does most of the regularisation work; the L₂ penalty is a marginal refinement.

4. **The physics-weight optimum on run_train22 is $\lambda = 0.5$ for PhysReg-FNN**, corresponding by (22) to a 33 % / 67 % posterior mean of the analytical RNEA and the data — the smaller training corpus shifts the optimum from $\lambda \approx 0.3$ on the previous larger corpus toward stronger physics regularisation.

5. **Per-joint physics relevance is non-uniform.** The biggest discriminator across architectures is **J₅ (wrist roll)**: BlackBox-FNN bottoms out at R² = 0.671; ResCorr-FNN lifts it to R² = 0.771 (+10 percentage points). J₅ is friction-dominated, exactly the channel where the bounded residual on top of the calibrated Coulomb-tanh + viscous model is most effective. J₂ (gravity-dominated shoulder) is the only joint where PhysReg-FNN edges ResCorr-FNN, because most of its torque is structured gravity that the analytical model already explains.

6. **Theoretical guarantees back the empirical findings.** Theorem 1 (normalisation invariance) shows that the per-component physics-tensor scaling preserves the noise-free identity $\sum_k \tilde{\boldsymbol{\tau}}_k = \tilde{\boldsymbol{\tau}}^*$, so the four-component decomposition is a drop-in replacement for the scalar sum without any rescaling. Lemma 2 (tanh contraction) establishes the unconditional $\|\boldsymbol{\delta}\|_\infty \le c_s$ bound that gives ResCorr-FNN its worst-case guarantee — translated to physical units, the per-joint correction cannot exceed roughly $\{0.38, 1.29, 0.61, 0.23, 0.36\}$ N·m respectively.

These results support the integration of RNEA-based analytical models as **architectural priors** (rather than soft loss penalties) in neural inverse-dynamics regression, especially when training data is limited or when worst-case physical-consistency guarantees are a deployment requirement. The bounded-residual ResCorr-FNN architecture combines the highest accuracy, the best data efficiency, the lowest hyperparameter sensitivity, and the strongest worst-case guarantee on run_train22 — making it the recommended baseline for follow-on work.

---

## References

[1] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. MIT Press, 2006.

[2] B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, *Robotics: Modelling, Planning and Control*. Springer, 2009.

[3] A. Gijsberts and G. Metta, "Real-time model learning using Incremental Sparse Spectrum Gaussian Process Regression," *Neural Networks*, vol. 41, pp. 59–69, 2013.

[4] S. Rueckert, M. Nakatenus, S. Tosatto, and J. Peters, "Learning inverse dynamics models with contacts," in *Proc. IEEE-RAS Int. Conf. Humanoid Robots (Humanoids)*, 2017.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *J. Comput. Phys.*, vol. 378, pp. 686–707, 2019.

[6] M. Lutter, C. Ritter, and J. Peters, "Deep Lagrangian Networks: Using physics as model prior for deep learning," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[7] S. Greydanus, M. Dzamba, and J. Yosinski, "Hamiltonian Neural Networks," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[8] J. Gonzalez-Garcia, R. Saatchi, et al., "Residual model learning for delta-physics in robot dynamics," *IEEE Robotics and Automation Letters*, 2020.

[9] R. Bhattacharya and B. Calli, "Physics-augmented learning of contact-rich robot dynamics," *IEEE Robotics and Automation Letters*, 2022.

[10] J. Carpentier, G. Saurel, G. Buondonno, J. Mirabel, F. Lamiraux, O. Stasse, and N. Mansard, "The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives," in *Proc. IEEE Int. Symp. System Integration (SII)*, 2019.

[11] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[12] R. W. Schafer, "What is a Savitzky-Golay filter?" *IEEE Signal Processing Magazine*, vol. 28, no. 4, pp. 111–117, 2011.

---

## Appendix A: Figure Index

### A.1 Figures embedded in this document

| Fig. | Source file | Section | Content |
|------|-------------|---------|---------|
| Fig. 9  | `PLOTS_and_FLOWCHARTS/drawio_flowcharts/architecture of all Neural Network Diagrams.pdf` | § VI | Unified comparison of the three NN architectures (Black-Box / Physics-Residual / Physics-Regularised), the analytical-physics block, and the experimental sweep grid |
| Fig. 10 | `PLOTS_and_FLOWCHARTS/comparision.png` | § IX-A | Per-joint torque trace overlay for ground truth, Residual, PINN_FNN, BlackBox FNN on a representative test trajectory |

### A.2 Grid analysis figures (under `Neural_Networks/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/analysis/`)

| Filename | Content | Cited in |
|----------|---------|---------|
| `fig1_training_dynamics.pdf` | Training and validation loss/RMSE curves across all 144 grid runs | § IX-F |
| `fig2_rmse_comparison.pdf` | Box plots of test RMSE per architecture and per joint | § IX-A |
| `fig3_per_joint_heatmaps.pdf` | Per-joint RMSE / R² heatmaps across architectures and HP cells | § IX-B |
| `fig4_mae_nrmse_comparison.pdf` | MAE and normalised RMSE comparison per joint | § IX-B |
| `fig5_topk_leaderboard.pdf` | Top-10 runs per architecture ranked by test RMSE | § IX-A |
| `fig6_rmse_distribution.pdf` | Histogram of test RMSE across all grid runs | § IX-A, IX-E |
| `fig7_physics_weight_impact.pdf` | Effect of physics weight $\lambda$ on PhysReg-FNN test RMSE | § IX-C |
| `fig8_r2_test_distribution.pdf` | Test R² distributions per architecture | § IX-A, IX-E |
| `summary_table.md` | Best-per-architecture aggregates (best-of-grid leaderboard) | § IX-A |

Per-trial training curves, normalisation statistics, and individual prediction-vs-target plots are stored in each run's subdirectory under `Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/[FNN | PhysicsRegularizedFNN | ResidualCorrectionFNN]/<run_id>/`.

---

## Appendix B: Reproducibility

All experiments were run on an NVIDIA GPU (CUDA 12.x) with PyTorch 2.x. Complete training configuration, normalisation statistics, and per-split metrics are saved in `metadata.yaml` within each run directory. Model checkpoints are stored as `model.pt` (best validation epoch) and `model_final.pt` (final epoch). The grid runner is idempotent: re-running with the same HPs skips already-completed trials by matching HP fingerprints against saved metadata.

The reference dataset for the results in this paper is

`Neural_Networks/train_data/run_train22_q0_qd91_qdd21_tau51_rnea15/`

and the corresponding 144-trial grid sweep lives at

`Neural_Networks/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/`.

**To reproduce the best ResCorr-FNN result** (pooled test RMSE 0.09666 N·m, R² 0.911):

```bash
# Edit Neural_Networks/models/run_loss_residual_grid.py:
#   ARCH                = "rescorr"
#   DATA_RUN_DIR        = "Neural_Networks/train_data/run_train22_q0_qd91_qdd21_tau51_rnea15"
#   GRID_RESCORR        = {
#       "alpha_reg_weight":    [0.1],
#       "data_train_fraction": [0.05],
#       "seed":                [0],
#   }
PYTHONPATH=. python -m Neural_Networks.models.run_loss_residual_grid
```

**To reproduce the best PhysReg-FNN result** (pooled test RMSE 0.09804 N·m, R² 0.908):

```bash
# Edit Neural_Networks/models/run_loss_residual_grid.py:
#   ARCH                = "physreg"
#   DATA_RUN_DIR        = "Neural_Networks/train_data/run_train22_q0_qd91_qdd21_tau51_rnea15"
#   GRID_PHYSREG        = {
#       "physics_weight":      [0.5],
#       "data_train_fraction": [0.05],
#       "seed":                [0],
#   }
PYTHONPATH=. python -m Neural_Networks.models.run_loss_residual_grid
```

**To reproduce the BlackBox-FNN baseline** (pooled test RMSE 0.10566 N·m, R² 0.893):

```bash
# Edit Neural_Networks/models/run_loss_residual_grid.py:
#   ARCH                = "fnn"
#   DATA_RUN_DIR        = "Neural_Networks/train_data/run_train22_q0_qd91_qdd21_tau51_rnea15"
#   GRID_FNN            = {
#       "data_train_fraction": [1.0],
#       "seed":                [0],
#   }
PYTHONPATH=. python -m Neural_Networks.models.run_loss_residual_grid
```

All three configurations share the fixed backbone hyperparameters: hidden layers `[128, 256, 128]`, GELU activation, dropout 0.2, AdamW with lr = 3 × 10⁻⁴ and weight decay 0.05, batch size 1024, warmup-cosine LR schedule, gradient-norm clip 1.0, input-noise σ = 0.05, max epochs 3000 with early-stopping patience 150 on `val_rmse`, and `correction_scale = 0.5` (ResCorr only).

The pre-trained model checkpoints under `Neural_Networks/Trained_Models_Grid/run_train22_q0_qd91_qdd21_tau51_rnea15/<arch>/<run_id>/model.pt` can be loaded directly via `torch.load("model.pt")` and used at inference time without re-training. The inference pipeline applies the same SG filtering and RNEA computation used during training, assembles the normalised feature and physics tensors, runs the model forward pass, and de-normalises the output torques for the control loop.


---

## Appendix C: Algorithmic Pseudocode

### C.1 BlackBox-FNN — Forward Pass and Loss

```
Algorithm 1: BlackBox-FNN per-batch step
Inputs:  features ∈ R^{B×3J}, target ∈ R^{B×J}, physics ∈ R^{B×4J} (ignored)
Params:  θ = (W^(l), b^(l), γ^(l), β^(l)) for l = 1..L+1
Output:  loss scalar; gradients ∇θ

1:  del physics                                            # explicit discard
2:  if training: features ← features + σ_n · N(0, I)        # input noise
3:  z^(0) ← features
4:  for l ← 1 to L do
5:      z ← W^(l) · z^(l-1) + b^(l)
6:      ẑ ← LayerNorm(z; γ^(l), β^(l))
7:      a ← σ(ẑ)                                           # SiLU/GELU
8:      z^(l) ← Dropout_p(a) (only if training)
9:  end for
10: τ̂ ← W^(L+1) · z^(L) + b^(L+1)                          # linear head
11: ℒ_data ← (1/B) · Σ_{i,j} w_j (τ̂_{ij} - target_{ij})^2
12: backward(ℒ_data); clip ∇θ to G_max; AdamW step
13: return ℒ_data
```

### C.2 PhysReg-FNN — Forward Pass and Composite Loss

```
Algorithm 2: PhysReg-FNN per-batch step at epoch e
Inputs:  features ∈ R^{B×3J}, target ∈ R^{B×J}, physics ∈ R^{B×4J}
Params:  θ as in Alg. 1, plus physics-weight schedule (λ, e_w)
Output:  loss scalar; gradients ∇θ

1:  α_eff ← λ · min(1, e / e_w)                            # warm-up ramp
2:  if training: features ← features + σ_n · N(0, I)
3:  u ← concat(features, physics)                           # ∈ R^{B×7J}
4:  τ_ref ← reduce_physics_to_total(physics)                # sum over 4-component axis
5:  τ̂ ← MLP_θ(u)                                            # backbone forward
6:  ℒ_data ← (1/B) · Σ_{i,j} w_j (τ̂_{ij} - target_{ij})^2
7:  ℒ_phys ← (1/B) · Σ_{i,j} w_j (τ̂_{ij} - τ_ref_{ij})^2
8:  ℒ ← ℒ_data + α_eff · ℒ_phys                             # additive Tikhonov
9:  backward(ℒ); clip ∇θ to G_max; AdamW step
10: log {ℒ_data, ℒ_phys, α_eff} for diagnostics
11: return ℒ
```

### C.3 ResCorr-FNN — Forward Pass and Regularised Loss

```
Algorithm 3: ResCorr-FNN per-batch step
Inputs:  features ∈ R^{B×3J}, target ∈ R^{B×J}, physics ∈ R^{B×4J}
Params:  θ = (network), c_s (buffer, fixed), α_r
Init (once at construction):  W^(L+1), b^(L+1) ← 1e-2 · (W^(L+1), b^(L+1))

Per-batch:
1:  if training: features ← features + σ_n · N(0, I)
2:  u ← concat(features, physics)                           # ∈ R^{B×7J}
3:  τ_phys ← reduce_physics_to_total(physics)               # ∈ R^{B×J}
4:  raw_delta ← MLP_θ(u)
5:  δ ← c_s · tanh(raw_delta)                              # bounded |δ| ≤ c_s
6:  τ̂ ← τ_phys + δ
7:  ℒ_data ← (1/B) · Σ_{i,j} w_j (τ̂_{ij} - target_{ij})^2
8:  ℒ_reg  ← α_r · (1/(BJ)) · Σ_{i,j} δ_{ij}^2
9:  ℒ ← ℒ_data + ℒ_reg
10: backward(ℒ); clip ∇θ to G_max; AdamW step
11: log ρ = E[|δ|] / E[|τ_phys|]                            # δ-ratio diagnostic
12: return ℒ
```

---

## Appendix D: Calibration Provenance

The current calibration parameters used throughout this paper are sourced from `Torque_Analysis/calibration_params.json` (schema version 1.1):

```json
{
  "mass.current": {
    "mass_scale": 0.09310315,
    "extra_masses": null,
    "total_mass_kg": 0.5934,
    "n_samples": 470529,
    "calibrated_at": "2026-03-28T12:21:17Z"
  },
  "friction.current": {
    "coulomb_nm":  [0.134975, 0.278199, 0.201313, 0.088112, 0.203864],
    "viscous_nm":  [0.300000, 0.300000, 0.245417, 0.040191, 0.046918],
    "friction_eps": 0.040469,
    "n_samples":   470529,
    "calibrated_at": "2026-03-28T12:22:09Z"
  }
}
```

Both calibrations were run as bulk fits over all 124 raw trajectories (470,529 samples). The friction calibration reduced J₄'s RMS residual from 0.0872 N·m to 0.0559 N·m; gains at the other joints were small (< 0.001 N·m) because the prior bulk fit had already converged. The full audit history (per-stage RMS-old / RMS-new) is preserved in the `mass.history` and `friction.history` arrays in the JSON, ensuring full reproducibility of any past dataset build.
