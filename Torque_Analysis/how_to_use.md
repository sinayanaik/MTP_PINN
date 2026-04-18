# Torque Analysis Pipeline — User Guide

A Python pipeline for analyzing, validating, and calibrating torque predictions on a
6-DOF robot arm using inverse dynamics, friction modelling, and deep-learning preparation.

---

## Quick Start

```bash
cd ~/Desktop/MTP_PINN

# 1. Calibrate mass scale (bulk, servo-adjusted — run once after hardware assembly)
python3 -m Torque_Analysis.calibrate_mass

# 2. Calibrate friction (bulk — all raw_samples/)
python3 -m Torque_Analysis.calibrate_friction

# 3. Verify on the reference file
python3 Torque_Analysis/main.py

# 4. Batch analysis of all files → infer_torque/
python3 Torque_Analysis/bulk_analyze.py
```

Both calibration scripts process **all files** in `raw_samples/` and write results to
`Torque_Analysis/calibration_params.json` automatically.  `config.py` reads that file
at import time — no manual edits required.

---

## Installation

```bash
pip install pin numpy scipy matplotlib lxml
```

Dependencies:

- `pinocchio` (`pin`) — rigid-body dynamics (RNEA)
- `numpy`, `scipy` — numerical computing and optimisation
- `matplotlib` — visualisation
- `lxml` — URDF/XACRO parsing

---

## Project Structure

```text
Torque_Analysis/
├── config.py                  # All constants and parameters
├── calibration_params.json    # Auto-written by calibration scripts
├── calibration_io.py          # Load/save calibration_params.json
├── calibrate_mass.py          # Mass scale calibration (bulk)
├── calibrate_friction.py      # Friction parameter calibration (bulk)
├── torque.py                  # Core dynamics: RNEA, friction, build_pinocchio_model
├── data_loader.py             # Load motion JSON files
├── utils.py                   # Timestamp repair, Savitzky-Golay differentiation
├── main.py                    # Single-file analysis entry point
├── bulk_analyze.py            # Batch analysis of all raw_samples/
├── plots.py                   # Single-file visualisation
└── plots_global.py            # Cross-trajectory summary plots
```

---

## Key Concepts

### What This Pipeline Does

For every recorded motion, the pipeline decomposes joint torques:

```text
τ_measured = τ_gravity + τ_inertia + τ_friction + τ_residual
```

- **τ_gravity + τ_inertia** — computed via RNEA from the URDF model
- **τ_friction** — calibrated Coulomb + viscous model: `c·tanh(q̇/ε) + v·q̇`
- **τ_residual** — what remains for a PINN to learn

---

## Calibration Workflow

### Step 1 — Mass Calibration

```bash
python3 -m Torque_Analysis.calibrate_mass
```

**What it does:**

Fits a global density scale α to minimise the gravity model error across all trajectories:

```text
τ_load ≈ α · τ_RNEA_unit(q)
α* = (aᵀb) / (aᵀa)   where a = τ_RNEA_unit, b = τ_load
```

The URDF was built from the full robot CAD assembly (structure + servo housings), so
servo masses are already embedded in the link mass definitions.  α scales the whole URDF
uniformly.  The fact that α ≈ 0.093 < 0.112 (pure PLA 70% infill) is expected: metal
servos pull the effective blended density above what PLA alone would give.

**Output:**

- Global α, calibrated total mass
- Per-joint α (diagnostic — detects wrong stall torques)
- Stall torque diagnostic (infers actual servo model from α ratio)
- Saves `calibration_params.json` → mass section

**Stall torque diagnostic:**

```text
τ_stall_inferred = τ_stall_assumed × (α_global / α_j)
```

If the inferred value ≈ 14.8 kgf·cm while the assumed value is 30.0, the joint
almost certainly uses an STS3032 servo instead of STS3215.  Update
`STALL_TORQUE_PER_JOINT` in `config.py` accordingly and re-run.

---

### Step 2 — Friction Calibration

```bash
python3 -m Torque_Analysis.calibrate_friction
```

**What it does:**

Fits `c` (Coulomb) and `v` (viscous) per joint using four complementary methods:

| Method | Model | Speed |
| ---------------------- | ----------------------------------------- | -------- |
| Regime analysis | Mean τ_f by velocity sign | Instant |
| Asymmetry analysis | Separate c⁺, c⁻ for each direction | Instant |
| Bias-aware ε sweep | `b + c·tanh(q̇/ε) + v·q̇` — analytical LS | Fast |
| Nonlinear per-joint ε | `c·tanh(q̇/ε_j) + v·q̇` — bounded optimiser | Moderate |

Results are synthesised into a single recommendation and saved to
`calibration_params.json` → friction section.

**If a joint hits the viscous cap (0.30 N·m·s/rad):**

This is a symptom, not a physical value.  Two common causes:

1. Wrong stall torque → τ_load is overestimated → τ_f is inflated
2. Missing inertia (servo masses) → M(q)·q̈ underestimated → inertial error
   leaks into the friction signal

Fix: run `calibrate_mass.py` first and check the stall torque diagnostic.

---

## Configuration

`config.py` is the single source of truth.  Calibration scripts write to
`calibration_params.json`; `config.py` reads from it at import time.

### Parameters you may need to set manually

```python
# Per-joint stall torques (kgf·cm)
# Update if stall torque diagnostic identifies STS3032 joints:
STALL_TORQUE_PER_JOINT = np.array([30.0, 30.0, 30.0, 14.8, 14.8, 30.0])

# Signal processing
SMOOTH_WINDOW    = 25   # Savitzky-Golay window (samples); larger → smoother qdd
SAVGOL_POLYORDER = 3    # Polynomial order (must be < SMOOTH_WINDOW)
DIFF_METHOD      = "savgol"  # "savgol" (recommended) or "gradient"
```

Everything else (MASS_SCALE, COULOMB_NM, VISCOUS_NM, FRICTION_EPS) is loaded
automatically from `calibration_params.json`.

---

## Input Data Format

Motion files must be JSON:

```json
{
  "logs": [
    {
      "t": 0.001,
      "act_pos": [10923, 10924, 0, 0, 0, 0],
      "act_vel": [0, 1, 0, 0, 0, 0],
      "load": [0.5, -0.3, 0, 0, 0, 0]
    }
  ],
  "metadata": {
    "joint_map": [
      {"name": "joint_1", "ticks_center": 10923, "direction": 1}
    ],
    "ticks_to_rad": 0.0015708
  }
}
```

---

## Output Interpretation

### Console RMS table

```text
SANITY CHECK — Active joints 0..4
==================================================================
  Jnt    Load RMS  RNEA RMS  RNEA+F RMS  Resid RMS  RNEA/Load
    0       0.210     0.012       0.197      0.163      0.055
    1       0.740     0.583       0.620      0.302      0.787
    2       0.422     0.356       0.386      0.175      0.844
    3       0.213     0.054       0.149      0.142      0.254
    4       0.207     0.007       0.138      0.143      0.035
```

| Column | Interpretation |
| ----------- | ----------------------------------------------- |
| Load RMS | Measured torque magnitude |
| RNEA RMS | Physics model (gravity + inertia) |
| RNEA+F RMS | Physics + friction model |
| Resid RMS | Error remaining for PINN |
| RNEA/Load | 1.0 = perfect; <1.0 = model underestimates |

**Typical causes of RNEA/Load << 1:**

- J1/J5: friction-dominated joints — physics contribution is small
- J4: wrong stall torque (use stall torque diagnostic to confirm)
- All joints: re-run mass calibration after fixing stall torques

---

## Troubleshooting

### ModuleNotFoundError: pinocchio

```bash
pip install pin
# or: conda install -c conda-forge pinocchio
```

### Residual RMS > 50% of Load RMS

1. Re-run `calibrate_mass.py` (check stall torque diagnostic)
2. Update `STALL_TORQUE_PER_JOINT` in `config.py` if STS3032 joints detected
3. Re-run `calibrate_friction.py`
4. Re-run `main.py` to verify improvement

### Joints at viscous cap in friction calibration

Fix the stall torque first (see Step 1), then re-run friction calibration.

---

## Tips

1. **Calibrate in order:** mass first, then friction.
   Friction calibration residuals depend on the mass model.

2. **Re-calibrate after any hardware change:**
   servo swap, gripper addition, re-assembly.

3. **Check residual before PINN training:**
   it should be noise-like (zero mean, no trend).
   Target: residual RMS < 30% of Load RMS.

4. **Weigh the robot** to verify the calibrated total mass.
   If actual = X kg: `MASS_SCALE = X / (unscaled URDF mass)`.
