# Torque Analysis Codebase — Complete Technical Analysis

---

## Module Index

```text
Torque_Analysis/
├── config.py               ← central parameter store
├── data_loader.py          ← JSON → NumPy arrays (L, M, N)
├── utils.py                ← timestamp repair, SG derivatives
├── torque.py               ← RNEA, friction, load-register conversion
├── diagnostics.py          ← 9 data-quality checks
├── plots.py                ← per-run plots (10 types, headless-capable)
├── plots_global.py         ← cross-run summary + histogram plots
├── plots_bulk.py           ← thin headless wrapper for bulk mode
├── main.py                 ← single-file analysis entry point
├── bulk_analyze.py         ← batch processor → infer_torque/ layout
├── calibrate_mass.py       ← gravity-based α calibration
└── calibrate_friction.py   ← four-method friction calibration
```

---

## 1. `config.py` — Central Parameter Store

### Role in the System

Single source of truth for every constant, path, and calibrated parameter.
Every other module imports from here. Changing a parameter propagates throughout
the entire pipeline without touching any other file.

### Key Parameter Groups

#### Robot Structure

```python
DOF          = 6
ACTIVE_JOINTS = 5          # joints 0-4 are actuated; joint 5 (tool) is passive
```

The robot has 6 revolute joints in a serial chain. Only the first 5 are motor-driven.
Joint 5 is a passive tool holder that produces no torque.

#### Motor / Servo Constants

```python
KT           = 11.0        # torque constant (mA → torque)
STALL_TORQUE = 30.0        # fallback stall torque, all joints (kgf·cm)
NOM_VOLTAGE  = 12.0        # rated voltage (V)
KGCM_TO_NM  = 0.09807      # kgf·cm → N·m
```

#### Per-Joint Stall Torques

```python
STALL_TORQUE_PER_JOINT = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
```

Used by `torque_from_load_raw()` instead of the scalar fallback. Allows distal joints
to have different servo models (e.g., STS3032 at 14.8 kgf·cm). Broadcasts across the
time axis: shape `(njoints,)` × shape `(N, njoints)` works element-wise.

#### Mass Calibration

```python
MASS_SCALE   = 0.111893
```

Global density correction factor. The URDF assumes steel density (7800 kg/m³);
the actual robot is PLA at ~70% infill (~875 kg/m³). Their ratio:

$$\alpha = \frac{\rho_{\text{PLA,70\%}}}{\rho_{\text{steel}}} = \frac{875}{7800} \approx 0.112$$

Every link mass and inertia tensor in the URDF is multiplied by this factor.

#### Signal Processing

```python
DIFF_METHOD      = "savgol"   # default differentiation method
SAVGOL_POLYORDER = 3          # polynomial order for Savitzky-Golay filter
SMOOTH_WINDOW    = 11         # SG window length (samples, must be odd)
TORQUE_SMOOTH    = 21         # window for current-based torque smoothing
```

`DIFF_METHOD` switches the entire `numerical_velocity` / `numerical_acceleration`
pipeline between `"savgol"` (Savitzky–Golay, default) and `"gradient"` (NumPy central
differences + moving average, classical fallback).

#### Friction Parameters

```python
COULOMB_NM   = np.array([0.2472, 0.2873, 0.2403, 0.1806, 0.2151, 0.0])
VISCOUS_NM   = np.array([0.0, 0.3, 0.0, 0.051, 0.0042, 0.0])
FRICTION_EPS = 0.0628
```

Friction torque at joint $j$:

$$\tau_{f,j} = c_j \cdot \tanh\!\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j \cdot \dot{q}_j$$

where $c_j$ = Coulomb, $v_j$ = viscous, $\varepsilon$ = smooth sign width.
Joint 5 has all zeros — passive joint.

---

## 2. `data_loader.py` — JSON Log Parser

### Role in the System

Entry point for every analysis. `load_log()` reads one JSON file and returns
structured NumPy arrays.

### Core Arrays

```python
L = {
    "t":        (N,)     # timestamps (s)
    "cmd_pos":  (N, 6)   # commanded joint positions (ticks)
    "act_pos":  (N, 6)   # actual joint positions (ticks)
    "load":     (N, 6)   # servo load register (0.1% units)
    "voltage":  (N, 6)   # bus voltage per joint (V)
    "current":  (N, 6)   # motor current (mA)
    "cmd_ee":   (N, 3)   # commanded end-effector position (m)
    "act_ee":   (N, 3)   # actual end-effector position (m)
}

M = {
    "joint_map":    list of per-joint dicts (direction, ticks_center, servo_id)
    "ticks_to_rad": scalar — encoder resolution (rad/tick)
    "ctrl_hz":      actual control loop rate
    "fb_hz":        actual feedback rate
}
```

### `joint_map` Structure

Each entry in `M["joint_map"]`:

| Key | Type | Meaning |
| --- | ---- | ------- |
| `direction` | ±1 | Maps servo positive rotation to URDF positive rotation |
| `ticks_center` | int | Encoder tick value at URDF zero angle |
| `servo_id` | int | Physical servo identifier |
| `joint_name` | str | Corresponding URDF joint name |

---

## 3. `utils.py` — Preprocessing and Numerical Differentiation

### Role in the System

Provides three essential transformations to go from raw JSON data to the
state vector `(q, qd, qdd)` required by RNEA.

### `fix_timestamps(t)`

Robot logs occasionally contain duplicate or retrograde timestamps (typically
1–2 per trajectory at the control-loop boundary).

**Repair procedure:**

1. Compute `Δt_i = t[i+1] - t[i]`; find indices where `Δt ≤ 0`
2. Interpolate bad timestamps linearly from surrounding good ones:

```python
t_fixed = np.interp(all_indices, good_indices, t[good_indices])
```

1. Final safety pass — enforce strict monotonicity with 1 ns epsilon:

```python
_EPS_T = 1e-9
for k in range(1, len(t_fixed)):
    if t_fixed[k] <= t_fixed[k - 1]:
        t_fixed[k] = t_fixed[k - 1] + _EPS_T
```

**Why the two-stage approach:** `np.interp` is immune to isolated bad timestamps
but cannot guarantee strict monotonicity if two "good" timestamps happen to coincide
at floating-point resolution. The 1 ns loop is a zero-cost safety net that eliminates
`np.gradient` divide-by-zero errors definitively. It is a no-op on clean data.

Previous approach (single-pass 1 μs insertion) was replaced because it could leave
duplicate timestamps when consecutive good indices were also identical.

### `_savgol_window(signal_len, requested_win, polyorder)`

Validates and adjusts the Savitzky–Golay window size:

```python
min_win = polyorder + 1
if min_win % 2 == 0:
    min_win += 1          # SG requires odd window
win = max(requested_win, min_win)
win = min(win, signal_len if signal_len % 2 == 1 else signal_len - 1)
return win
```

Ensures: window is odd, ≥ polyorder + 1, ≤ signal length.

### `ticks_to_radians(act_pos, joint_map, ticks_to_rad, dof=6)`

$$q_j = d_j \cdot \left(p_j^{\text{ticks}} - p_j^{\text{center}}\right) \cdot k_{\text{t2r}}$$

| Symbol | Source |
| ------ | ------ |
| $d_j \in \{-1, +1\}$ | `joint_map[j]["direction"]` |
| $p_j^{\text{center}}$ | `joint_map[j]["ticks_center"]` |
| $k_{\text{t2r}}$ | `M["ticks_to_rad"]` |

### `numerical_velocity(q, t, smooth_window=None, method=None)`

`method=None` reads `C.DIFF_METHOD` (default `"savgol"`).

**Savitzky–Golay path** (`method="savgol"`):

```python
win = _savgol_window(N, smooth_window, C.SAVGOL_POLYORDER)
qd = savgol_filter(q, win, C.SAVGOL_POLYORDER, deriv=1,
                   delta=mean_dt, axis=0)
```

Uses `mean(Δt)` as the uniform step — makes the filter immune to per-sample
zero-spacing artefacts from timestamp repair. Differentiates and smooths in
one polynomial fit.

**Gradient path** (`method="gradient"`):

```python
qd = np.gradient(q, t_fixed, axis=0)   # central differences
qd = smooth(qd, smooth_window)          # uniform moving average
```

Both paths apply:

```python
qd = np.clip(qd, -C.VEL_CLIP, C.VEL_CLIP)   # ±100 rad/s
qd = np.nan_to_num(qd)
```

### `numerical_acceleration(qd, t, smooth_window=None, method=None)`

Applies the same pipeline to the already-smoothed velocity signal.
Result: two rounds of differentiation + two rounds of smoothing.

**Frequency-domain effect:**

$$\ddot{Q}(f) = Q(f) \cdot D^2(f) \cdot H^2(f)$$

where $D(f)$ amplifies high frequencies (differentiation) and $H(f)$ attenuates
them (smoothing). The product suppresses noise while preserving low-frequency
true acceleration.

### Smoothing — `smooth(x, window, axis=0)`

```python
from scipy.ndimage import uniform_filter1d
return uniform_filter1d(x, size=window, axis=axis)
```

Symmetric (non-causal) moving average — **no phase shift**. Frequency response:

$$H(f) = \frac{\sin(\pi f w \Delta t)}{\pi f w \Delta t}$$

---

## 4. `torque.py` — Core Torque Computation Engine

### Part (a): Current-Based Torque — `torque_from_current`

Legacy estimate (not used in main pipeline). Motor current (mA) → torque:

$$\tau_{\text{current},j} = \text{sign}(\ell_j) \cdot \left|\frac{I_j}{1000}\right| \cdot K_T \cdot k_{\text{conv}}$$

Less reliable than load-register because the current register on these servos
is often unsigned (magnitude only), requiring the sign to be inferred from the
load register.

### Part (b): Load-Register Torque — `torque_from_load_raw` / `torque_from_load`

**`torque_from_load_raw(load, voltage, stall_torque_per_joint=None)`:**

```python
if stall_torque_per_joint is None:
    nj = load.shape[1] if load.ndim == 2 else 1
    stall_torque_per_joint = C.STALL_TORQUE_PER_JOINT[:nj]

load_frac = load * 0.1 / 100.0          # fraction of stall torque
v_scale   = voltage / C.NOM_VOLTAGE     # voltage correction
tau = load_frac * stall_torque_per_joint * v_scale * C.KGCM_TO_NM
```

$$\tau_j^{\text{servo}} = \frac{\ell_j \times 0.1}{100} \times \frac{V_j}{V_{\text{nom}}} \times \tau_{\text{stall}}(j) \times k_{\text{conv}}$$

Using `STALL_TORQUE_PER_JOINT` (shape `(njoints,)`) instead of the scalar
`STALL_TORQUE` allows different servo models on different joints. For example,
if J4 and J5 use STS3032 (14.8 kgf·cm) instead of STS3215 (30 kgf·cm),
set `STALL_TORQUE_PER_JOINT[3:5] = 14.8`.

**`torque_from_load(load, voltage, joint_map=None, stall_torque_per_joint=None)`:**

```python
tau_servo = torque_from_load_raw(load, voltage, stall_torque_per_joint)
direction = np.array([jm["direction"] for jm in joint_map])
tau_urdf  = -direction * tau_servo
```

$$\tau_{\text{load},j}^{\text{URDF}} = -d_j \cdot \tau_j^{\text{servo}}$$

The $-d_j$ convention was validated by `calibrate_mass.py` which tested three
sign mappings (raw, $+d_j$, $-d_j$) and selected the one giving positive,
consistent mass scale factors.

### Part (c): Build Pinocchio Model — `build_pinocchio_model`

```python
urdf_xml = xacro.process_file(xacro_path).toxml()
model = pin.buildModelFromXML(urdf_xml)

if mass_scale != 1.0:
    for i in range(model.njoints):
        model.inertias[i].mass    *= mass_scale
        model.inertias[i].inertia *= mass_scale

data = model.createData()
return model, data, model.nq
```

**Step-by-step:**

1. XACRO processing: `xacro.process_file()` expands all macros → standard URDF XML
2. Pinocchio model: builds kinematic tree, per-joint transforms, per-link 6×6 spatial inertias
3. Mass scaling: $m_i \leftarrow \alpha \cdot m_i$, $\mathbf{I}_i \leftarrow \alpha \cdot \mathbf{I}_i$
   — physically correct for uniform-density parts with unchanged geometry
4. Data allocation: `model.createData()` allocates RNEA workspace arrays

Because RNEA is linear in inertial parameters:

$$\text{RNEA}^{(\alpha)}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) = \alpha \cdot \text{RNEA}^{(1)}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})$$

Errors and xacro/pinocchio failures raise descriptive `RuntimeError` with cause.

### Part (d): Full Inverse Dynamics — `torque_from_urdf`

```python
qd  = numerical_velocity(q, t, smooth_window)
qdd = numerical_acceleration(qd, t, smooth_window)

tau = np.zeros((N, model.nq))
for i in range(N):
    tau[i] = pin.rnea(model, data, q[i], qd[i], qdd[i])
```

At each timestep:

$$\boldsymbol{\tau}_{\text{RNEA}}(t_k) = \text{pin.rnea}(\mathbf{q}(t_k), \dot{\mathbf{q}}(t_k), \ddot{\mathbf{q}}(t_k)) = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$$

RNEA does not form M, C, g explicitly; two recursive passes compute τ in $O(n)$.

### Part (e): Gravity-Only Torque — `torque_gravity_only`

```python
zero = np.zeros(model.nv)   # allocated once, reused
for i in range(N):
    tau_g[i] = pin.rnea(model, data, q[i], zero, zero)
```

$$\boldsymbol{\tau}_g = \text{RNEA}(\mathbf{q}, \mathbf{0}, \mathbf{0}) = \mathbf{g}(\mathbf{q})$$

Used for mass calibration (gravity is the most reliable regression signal for α)
and for decomposition plots in diagnostics.

### Part (f): Smooth Friction Model — `_smooth_sign` / `torque_friction`

```python
def _smooth_sign(x, eps=None):
    if eps is None:
        eps = C.FRICTION_EPS    # reads from config, no longer hardcoded
    return np.tanh(x / eps)

def torque_friction(qd, coulomb=None, viscous=None, eps=None):
    return coulomb * _smooth_sign(qd, eps) + viscous * qd
```

$$\tau_{f,j} = c_j \cdot \tanh\!\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j \cdot \dot{q}_j$$

The `eps` default was changed from the hardcoded `0.05` to reading `C.FRICTION_EPS`
so that calibration updates to `config.py` take effect automatically.

Derivative is smooth everywhere:

$$\frac{\partial \tau_{f,j}}{\partial \dot{q}_j} = \frac{c_j}{\varepsilon}\,\text{sech}^2\!\left(\frac{\dot{q}_j}{\varepsilon}\right) + v_j$$

This is essential for PINN backpropagation through the friction model.

---

## 5. `bulk_analyze.py` — Batch Processor

### Role in the System

Processes all JSON logs in `raw_samples/` and writes per-run outputs to
`infer_torque/<run_id>/`. Accumulates cross-run statistics and generates the
rich `global_summary.json` and global plot set.

### Key Constants

```python
HIST_BINS      = np.linspace(-2.5, 2.5, 101)   # 100 bins, ±2.5 N·m
HIST_BIN_EDGES = HIST_BINS.tolist()             # stored in JSON

KNOWN_SHAPES = [
    "regular_polygon", "sine_wave", "spiral",   # longest-match first
    "figure8", "ellipse", "circle", "line",
    "lissajous", "arc",
]
```

### `parse_filename(fname)` → dict

Extracts metadata from filenames like:

```text
circle_r65mm_xz_cx66cyn265cz275_quintic_poly_ctrlmax_fbmax_001.json
```

Returns:

```python
{
    "shape":     "circle",
    "radius_mm": 65,
    "plane":     "xz",
    "traj_type": "quintic_poly",
    "fb_max":    True,
    "seq":       1,
}
```

Longest-match ordering in `KNOWN_SHAPES` ensures `regular_polygon` is matched
before `polygon` (which does not exist, but prevents partial matches).

### Per-Run Processing — `process_one_file(path, model, data, joint_map)`

Outputs per run:

- `infer_torque/<safe_run_id>/torque.png` — joint torque comparison plot
- `infer_torque/<safe_run_id>/run_summary.json` — metrics + histogram counts

**Histogram computation** (no raw samples stored):

```python
HIST_BINS = np.linspace(-2.5, 2.5, 101)

for j in range(nq):
    err_rnea  = tau_load[:, j] - tau_urdf[:, j]
    err_model = tau_load[:, j] - tau_model[:, j]
    err_fric  = tau_load[:, j] - tau_fric[:, j]
    error_hists[j] = {
        "rnea":  np.histogram(err_rnea,  bins=HIST_BINS)[0].tolist(),
        "model": np.histogram(err_model, bins=HIST_BINS)[0].tolist(),
        "fric":  np.histogram(err_fric,  bins=HIST_BINS)[0].tolist(),
        "n":     int(len(err_rnea)),
    }
```

Pre-binned `int` counts keep JSON size small (~18 K integers for all runs + shapes).

### Global Aggregation — `build_global_summary(run_summaries)`

**Histogram accumulation** across all runs and by shape:

```python
models = ("rnea", "model", "fric")
n_bins = len(HIST_BIN_EDGES) - 1

def _add_hists(acc, s):
    for j, jh in s.get("error_histograms", {}).items():
        j = int(j)
        if j not in acc:
            acc[j] = {m: np.zeros(n_bins, dtype=np.int64) for m in models}
        for m in models:
            if m in jh:
                acc[j][m] += np.array(jh[m], dtype=np.int64)
```

Integer bin counts accumulate without any per-sample memory overhead.
Final `gs["error_histograms"]` contains `bin_edges`, `global`, and `by_shape` sub-dicts.

**Rich metadata sections in `global_summary.json`:**

- `by_shape` — per-shape NRMSE medians keyed by geometry type (circle, ellipse, …)
- `by_traj_type` — per-trajectory-type NRMSE (ruckig, quintic_poly, …)
- `by_radius_mm` — NRMSE vs trajectory radius
- `model_quality` — per-joint: `pct_runs_nrmse_lt_20pct`, `pct_runs_nrmse_lt_50pct`,
  `median_nrmse_pct`
- `error_histograms` — pre-binned error counts for all three model variants

---

## 6. `plots_global.py` — Cross-Run Summary Plots

### Role in the System

Generates all cross-run visualisations from the list of `run_summary` dicts
and the `error_histograms` structure built by `bulk_analyze.py`.

Called via `generate_all_global_plots(summaries, save_dir, error_histograms=None)`.

### Plot Functions

| Function | Output file | Description |
| -------- | ----------- | ----------- |
| `plot_rnea_ratio_violin` | `rnea_ratio_violin.png` | Violin per joint — RNEA/Load ratio distribution |
| `plot_nrmse_violin` | `nrmse_violin.png` | Violin per joint — NRMSE distribution |
| `plot_residual_rms_boxplot` | `residual_rms_boxplot.png` | Box plots — absolute residual RMS per joint |
| `plot_load_vs_rnea_scatter` | `load_vs_rnea_scatter.png` | Scatter: load RMS vs RNEA RMS, Pearson r annotated |
| `plot_accuracy_by_shape` | `accuracy_by_shape.png` | Heatmap: shape × joint, cell = median NRMSE % |
| `plot_accuracy_by_traj_type` | `accuracy_by_traj_type.png` | Grouped bars per trajectory type |
| `plot_accuracy_vs_radius` | `accuracy_vs_radius.png` | NRMSE vs radius per joint with trend line |
| `plot_model_coverage_cdf` | `model_coverage_cdf.png` | CDF: fraction of runs vs NRMSE per joint |
| `plot_error_histograms_global` | `error_hist_global.png` | 2×3 grid — global error density for 3 variants |
| `plot_error_histograms_by_shape` | `error_hist_by_shape_J{1-5}.png` | Per-joint error by shape |

### Histogram Plotting Helpers

**`_HIST_STYLES`** — colour scheme for the three model variants:

```python
_HIST_STYLES = {
    "rnea":  {"color": "#4878CF", "label": "RNEA only"},
    "model": {"color": "#6ACC65", "label": "RNEA + Friction"},
    "fric":  {"color": "#D65F5F", "label": "Friction only"},
}
```

**`_plot_hist_axes(ax, bin_edges, hists_dict, title)`** — normalised density
bar chart. Converts accumulated integer counts to probability density:

```python
density = counts / (counts.sum() * bin_width)
```

**`plot_error_histograms_global(error_histograms, save_dir)`:**

- 2×3 grid (5 active joints + summary stats table)
- Each subplot: overlaid density bars for RNEA, RNEA+Friction, Friction-only
- Summary table: mean, std, kurtosis for each variant × joint

**`plot_error_histograms_by_shape(error_histograms, save_dir)`:**

- One figure per active joint (J1–J5) → `error_hist_by_shape_J{j}.png`
- Subplots: one per detected geometry shape
- Reveals which shapes are hardest for the physics model

---

## 7. `calibrate_mass.py` — Mass Scale Calibration

### Role in the System

Fits the global density scale $\alpha$ using gravity torque matching on joints 1 & 2
(strongest gravity signal). Previously called `calibrate_v2.py`.

### Algorithm — `fit_scale(tau_load, tau_grav_raw)`

Minimise the weighted least-squares objective:

$$\alpha^* = \arg\min_\alpha \sum_{t,j} \left(\tau_{\text{load},j}(t) - \alpha \cdot \tau_{\text{grav},j}^{(1)}(t)\right)^2$$

Closed-form solution:

$$\boxed{\alpha^* = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}}}$$

where $\mathbf{a} = \text{vec}(\tau_{\text{grav}}^{(1)}[\mathcal{J}])$ and
$\mathbf{b} = \text{vec}(\tau_{\text{load}}[\mathcal{J}])$.

### Sign Convention Validation

Tests three conventions (`raw`, `+d_j`, `-d_j`) and selects the one where
all calibration joints yield $\alpha > 0$ with lowest cross-joint std.
Result: `−d_j` is the correct convention for Kikobot.

**Update `config.py`** after calibration:

```python
MASS_SCALE = alpha_star   # e.g., 0.111893
```

---

## 8. `calibrate_friction.py` — Friction Calibration

### Role in the System

Runs four complementary calibration methods on the friction residual signal
$\tau_f^{\text{signal}} = \tau_{\text{load}} - \tau_{\text{RNEA}}$ and synthesises
a weighted recommendation.

### Methods Summary

| Method | Key API | Robustness | Notes |
| ------ | ------- | ---------- | ----- |
| A: Regime analysis | `regime_analysis()` | Low | Contaminated by gravity error |
| B: Asymmetry analysis | `asymmetry_analysis()` | High | Gravity-robust via sign decomposition |
| C: ε-sweep LS | `sweep_eps_bias()` + `fit_bias_aware()` | High | Analytical, fast on 470K samples |
| D: Nonlinear per-joint | `fit_nonlinear_joint()` | Medium | Per-joint ε via L-BFGS-B |

Consistency check: per-trajectory CV% flags unstable parameters.

Synthesis: `synthesize_recommendation()` — trust-weighted average
(B: w=3.0, C: w=2.0, D: w=1.5, A: w=1.0 for Coulomb).

### Method C — ε-Sweep Matrix Equation

For fixed $\varepsilon$, the model is linear in $(b_j, c_j, v_j)$:

$$\boldsymbol{\Phi}_j(\varepsilon) \cdot \mathbf{x}_j = \mathbf{y}_j$$

$$\boldsymbol{\Phi}_j(\varepsilon) = \begin{bmatrix} 1 & \tanh(\dot{q}_{j,1}/\varepsilon) & \dot{q}_{j,1} \\ \vdots & \vdots & \vdots \end{bmatrix}$$

$$\mathbf{x}_j^* = \left(\boldsymbol{\Phi}_j^T \boldsymbol{\Phi}_j\right)^{-1} \boldsymbol{\Phi}_j^T \mathbf{y}_j$$

The grid search over 100 ε values reduces to $3\times3$ matrix inversions — instant
on the full 470K-sample dataset.

**Update `config.py`** after calibration:

```python
COULOMB_NM   = np.array([c0, c1, c2, c3, c4, 0.0])
VISCOUS_NM   = np.array([v0, v1, v2, v3, v4, 0.0])
FRICTION_EPS = eps_star
```

---

## 9. Output Layout

```text
infer_torque/
├── {run_id}/                          ← one folder per trajectory
│   ├── torque.png                     ← per-run 6-joint comparison plot
│   └── run_summary.json               ← per-run metrics + histogram counts
├── global_plots/
│   ├── rnea_ratio_violin.png
│   ├── nrmse_violin.png
│   ├── residual_rms_boxplot.png
│   ├── load_vs_rnea_scatter.png
│   ├── accuracy_by_shape.png
│   ├── accuracy_by_traj_type.png
│   ├── accuracy_vs_radius.png
│   ├── model_coverage_cdf.png
│   ├── error_hist_global.png          ← global error histograms (3 variants × 5 joints)
│   └── error_hist_by_shape_J{1-5}.png ← per-shape error histograms per joint
└── global_summary.json                ← schema v2.0, full rich metadata
```

### `run_summary.json` Key Fields

```json
{
  "run_id":      "circle_r65mm_xz_..._001",
  "run_dir":     "infer_torque/circle_r65mm_xz_..._001",
  "traj_meta":   { "shape": "circle", "radius_mm": 65, "traj_type": "quintic_poly" },
  "n_samples":   3791,
  "duration_s":  12.5,
  "joint_metrics": [
    {
      "joint":          0,
      "load_rms":       0.182,
      "rnea_rms":       0.047,
      "model_rms":      0.094,
      "rnea_over_load": 0.258,
      "nrmse":          0.484
    }
  ],
  "error_hists": {
    "0": { "rnea": [0, 0, ...], "model": [...], "fric": [...], "n": 3791 }
  }
}
```

### `global_summary.json` Schema v2.0

Top-level keys:

| Key | Description |
| --- | ----------- |
| `schema_version` | `"2.0"` |
| `config` | Snapshot of config constants used |
| `processing` | `total_files`, `succeeded`, `failed`, `error_rate_pct` |
| `dataset` | `total_samples`, `total_duration_s` |
| `joint_aggregate` | Per-joint medians/means/stds for RNEA ratio, model ratio, NRMSE |
| `by_shape` | Per-shape aggregates (n_runs, NRMSE per joint) |
| `by_traj_type` | Per-trajectory-type aggregates (ruckig, quintic_poly, …) |
| `by_radius_mm` | NRMSE per joint vs radius |
| `model_quality` | Threshold-based quality: pct runs with NRMSE < 20%/50% |
| `error_histograms` | Pre-binned counts: `bin_edges`, `global`, `by_shape` |
| `runs` | List of all per-run summaries |
| `errors` | List of failed files with exception messages |

---

## 10. Calibration Workflow

```bash
# Step 1 — Mass scale (run once per robot build)
python3 -m Torque_Analysis.calibrate_mass
# → update MASS_SCALE in config.py

# Step 2 — Friction (bulk mode, 124 trajectories)
python3 -m Torque_Analysis.calibrate_friction --bulk
# → update COULOMB_NM, VISCOUS_NM, FRICTION_EPS in config.py

# Step 3 — Single-file sanity check
python3 Torque_Analysis/main.py

# Step 4 — Full batch run
python3 Torque_Analysis/bulk_analyze.py
```

---

## 11. Known Parameter Update Locations

| What to update | Where | Config key |
| -------------- | ----- | ---------- |
| Density scale α | `config.py` | `MASS_SCALE` |
| Coulomb friction per joint | `config.py` | `COULOMB_NM` |
| Viscous friction per joint | `config.py` | `VISCOUS_NM` |
| Sign transition width | `config.py` | `FRICTION_EPS` |
| Per-joint stall torques | `config.py` | `STALL_TORQUE_PER_JOINT` |
| Differentiation method | `config.py` | `DIFF_METHOD` |
| SG polynomial order | `config.py` | `SAVGOL_POLYORDER` |
| SG / MA window length | `config.py` | `SMOOTH_WINDOW` |
