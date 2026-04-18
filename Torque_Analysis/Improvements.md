# Torque Analysis Codebase — Complete Review & Improvement Recommendations

## Executive Summary

This is a **well-engineered** physics-informed torque analysis pipeline for a 6-DOF 3D-printed robot arm (kikobot). The codebase computes analytical torque via RNEA (Pinocchio), calibrates mass scaling and friction parameters, and prepares residual data for PINN training. The documentation is exceptional — the math, design rationale, and failure history are all thoroughly recorded.

That said, after reviewing all 14 files, I've identified **concrete improvements** across six categories: code quality, numerical robustness, performance, architecture, testing, and the PINN interface. Below is a prioritized breakdown.

---

## 1. Performance Bottlenecks

### 1.1 Python-loop RNEA calls — the single biggest bottleneck

In `torque.py`, both `torque_from_urdf()` and `torque_gravity_only()` call `pin.rnea()` inside a Python for-loop over every timestep:

```python
for i in range(N):
    tau[i] = pin.rnea(model, data, q[i], qd[i], qdd[i])
```

With ~3,800 samples per file and 124 files in bulk mode, this is ~470K Python-to-C++ calls with per-call overhead. **Pinocchio supports vectorized RNEA** via `pinocchio.rnea()` on batched data if you restructure the call, or you can use `numba`/`joblib` to parallelize across timesteps.

**Recommendation:**

- Use `pinocchio.computeAllTerms()` if appropriate, or batch the loop with `concurrent.futures.ThreadPoolExecutor` since Pinocchio releases the GIL during computation.
- At minimum, pre-allocate `data` outside the loop (already done — good) and consider converting `q`, `qd`, `qdd` to Pinocchio-native types to avoid per-call conversion overhead.
- For `torque_gravity_only()`, the zero vectors are re-created each iteration — pre-allocating once is a trivial win.

### 1.2 Redundant RNEA computation in `calibrate_friction.py`

`load_friction_data()` calls `torque_from_urdf()`, which internally computes velocity and acceleration from scratch via `numerical_velocity()` and `numerical_acceleration()`. In bulk mode, this means 124 files × full differentiation + smoothing + RNEA. But in `bulk_analyze.py`, the same computation happens again independently.

**Recommendation:** If both scripts run on the same data, cache the intermediate results (e.g., save `qd`, `qdd`, `tau_urdf` as `.npz` files) so the friction calibrator can skip redundant computation. This would cut friction calibration time roughly in half.

### 1.3 Epsilon sweep is embarrassingly parallel

`sweep_eps_bias()` runs 100 epsilon values sequentially, each doing an independent least-squares fit. These are completely independent and could be parallelized with `joblib.Parallel` or even vectorized into a single batched least-squares solve.

---

## 2. Numerical Robustness Issues

### 2.1 `fix_timestamps()` creates artificial 1 MHz sampling

When duplicate timestamps are detected, the fix adds 1 μs gaps:

```python
t_fixed[i] = t_fixed[i - 1] + 1e-6
```

This creates an artificial `dt = 1e-6`, which when used in `np.gradient()` amplifies any position change at that timestep by a factor of 10⁶. The subsequent clipping at ±100 rad/s catches the worst cases, but values just under the clip threshold can still corrupt the smoothed signal.

**Recommendation:** Instead of inserting microsecond gaps, **interpolate the position** at the duplicate timestamp using neighboring samples, or flag duplicate-timestamp samples and exclude them from gradient computation. A cleaner approach:

```python
def fix_timestamps(t):
    dt = np.diff(t)
    bad = dt <= 0
    if bad.any():
        # Linear interpolation of timestamps at bad points
        good_idx = np.where(~np.insert(bad, 0, False))[0]
        t_fixed = np.interp(np.arange(len(t)), good_idx, t[good_idx])
        return t_fixed
    return t
```

### 2.2 `np.gradient` is suboptimal for noisy position data

Central differences (`np.gradient`) have a frequency response that amplifies noise at the Nyquist frequency. The documentation itself references Savitzky-Golay filters (reference [R20]) as superior alternatives. Since you're already smoothing after differentiation, combining these into a single Savitzky-Golay differentiation step would:

- Reduce phase distortion from two sequential operations
- Provide better noise rejection per unit of signal attenuation
- Eliminate the need for separate `numerical_velocity` + `smooth` calls

**Recommendation:**

```python
from scipy.signal import savgol_filter
qd = savgol_filter(q, window_length=11, polyorder=3, deriv=1, delta=dt_mean, axis=0)
```

### 2.3 Hardcoded clip bounds are fragile

Velocity is clipped at ±100 rad/s and acceleration at ±1000 rad/s². These are reasonable for this specific robot but are magic numbers. If the robot configuration changes (different servos, gearboxes), these bounds could silently clip valid data.

**Recommendation:** Derive clip bounds from physical limits in `config.py`:

```python
MAX_JOINT_VEL = 10.0   # rad/s — physical servo speed limit
VEL_CLIP = MAX_JOINT_VEL * 10  # generous safety margin
ACC_CLIP = VEL_CLIP / (1.0 / CTRL_HZ)  # max physically possible
```

---

## 3. Code Architecture Improvements

### 3.1 `plots.py` and `plots_bulk.py` violate DRY

These two files contain nearly identical plotting logic — the only differences are:

- `plots_bulk.py` uses the `Agg` backend
- `plots_bulk.py` calls `plt.close(fig)` instead of `plt.show()`
- `plots_bulk.py` accepts a `title` parameter

This duplication means every plot change must be made in two places.

**Recommendation:** Merge into a single module with a mode flag:

```python
# plots.py
_HEADLESS = False

def set_headless(headless=True):
    global _HEADLESS
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    _HEADLESS = headless

def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=C.DPI, bbox_inches="tight")
    if _HEADLESS:
        plt.close(fig)
    else:
        plt.show()
```

This halves the plotting code and eliminates sync bugs.

### 3.2 `sys.path.insert` hacks in every script

Every executable script (`main.py`, `calibrate_v2.py`, `calibrate_friction.py`, `bulk_analyze.py`) has:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

This is fragile and unnecessary if the package is properly installable.

**Recommendation:** Add a minimal `pyproject.toml` or `setup.py`:

```toml
[project]
name = "torque-analysis"
version = "1.0.0"
[tool.setuptools.packages.find]
include = ["Torque_Analysis"]
```

Then `pip install -e .` and all path hacks disappear.

### 3.3 `config.py` mixes calibration results with runtime constants

`config.py` holds both immutable constants (servo specs, DOF count) and calibrated values (MASS_SCALE, friction parameters) in the same flat namespace. This makes it unclear what's tunable vs. fixed, and makes it impossible to track calibration history.

**Recommendation:** Split into two files:

- `constants.py` — hardware specs, file paths, DOF, servo constants (never changes)
- `calibration.py` — mass scale, friction params, smoothing windows (changes per calibration run)

Or better: have the calibration scripts write their results to a versioned JSON/YAML file that `config.py` loads:

```python
# calibration_results.json
{
  "version": "2024-12-15",
  "mass_scale": 0.111893,
  "coulomb_nm": [0.2118, 0.303, ...],
  ...
}
```

### 3.4 No proper logging

All diagnostic output uses `print()`. In bulk mode with 124 files, the console output is overwhelming and unsearchable.

**Recommendation:** Replace print statements with Python's `logging` module:

```python
import logging
logger = logging.getLogger("torque_analysis")
```

This gives you log levels (DEBUG for per-file stats, WARNING for flag conditions, ERROR for failures), file output, and structured filtering — all for free.

---

## 4. Missing Error Handling & Edge Cases

### 4.1 `load_log()` assumes perfect JSON structure

If any key is missing from the JSON (e.g., a log file from a different firmware version), the function crashes with a `KeyError` with no context about which file or which key failed.

**Recommendation:** Add defensive key access with informative errors:

```python
def _safe_get(data, key, context=""):
    if key not in data:
        raise ValueError(f"Missing key '{key}' in {context}")
    return data[key]
```

### 4.2 `build_pinocchio_model()` silently ignores xacro failures

If the XACRO file has syntax errors or missing includes, `xacro.process_file()` may raise cryptic errors. The function should catch and re-raise with context.

### 4.3 Division-by-zero in `safe_ratio`

The threshold `b < 1e-4` is compared against `b` which can be negative (since it comes from an RMS ratio, it shouldn't be, but `b` isn't explicitly `abs(b)`). This is fine in practice but `abs(b) < 1e-4` would be more defensive.

### 4.4 `calibrate_friction.py` nonlinear fit uses random subsampling

```python
idx = np.random.choice(len(qd_j), NL_MAX_SAMPLES, replace=False)
```

This is non-deterministic — different runs produce different results. Add `np.random.seed()` or use a `RandomState` for reproducibility.

---

## 5. Testing & Validation Gaps

### 5.1 No automated tests exist

The codebase has no `test_*.py` files. The diagnostics in `diagnostics.py` are runtime checks, not unit tests.

**Recommendation:** Add at minimum:

- **Unit tests for `utils.py`**: Test `ticks_to_radians` with known inputs, test `fix_timestamps` with duplicate/reversed timestamps, test `numerical_velocity` against analytical derivatives of known functions (e.g., `q(t) = sin(2πt)` → `qd(t) = 2π·cos(2πt)`).
- **Unit tests for `torque.py`**: Test `torque_friction` output shape and sign conventions, test that `torque_gravity_only` returns zero for a vertical robot configuration.
- **Regression tests**: Save known-good outputs for one trajectory and assert future runs match within tolerance.

### 5.2 No CI/CD or linting

No `pyproject.toml` with tool configs, no `.flake8`, no type checking.

**Recommendation:** Add `ruff` or `flake8` for linting, `mypy` for type checking (the code already has some type hints), and a simple `pytest` runner.

---

## 6. PINN Interface Improvements

### 6.1 No standardized data export for PINN training

The codebase prepares residual data but doesn't export it in a format ready for neural network training. The PINN developer must reverse-engineer the pipeline to extract `(q, qd, qdd, t) → tau_residual` pairs.

**Recommendation:** Add an `export_training_data.py` script that:

```python
def export_pinn_dataset(output_path, model, json_files):
    """Export standardized training data for PINN."""
    all_inputs = []  # (q, qd, qdd, t)
    all_targets = []  # tau_residual
    
    for jf in json_files:
        # ... process ...
        inputs = np.column_stack([q, qd, qdd, t.reshape(-1, 1)])
        targets = tau_residual
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    np.savez_compressed(output_path,
        inputs=np.vstack(all_inputs),
        targets=np.vstack(all_targets),
        feature_names=[...],
        config={...})
```

### 6.2 No train/validation/test split guidance

The 124 trajectories should be split by trajectory (not by timestep) to prevent data leakage from temporal autocorrelation. The codebase doesn't provide utilities for this.

**Recommendation:** Add trajectory-level splitting:

```python
def split_trajectories(json_files, train=0.7, val=0.15, test=0.15, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(json_files))
    n_train = int(len(json_files) * train)
    n_val = int(len(json_files) * val)
    return {
        "train": [json_files[i] for i in indices[:n_train]],
        "val": [json_files[i] for i in indices[n_train:n_train+n_val]],
        "test": [json_files[i] for i in indices[n_train+n_val:]],
    }
```

### 6.3 No normalization statistics provided

PINN training requires input/output normalization. The codebase computes RMS values in bulk analysis but doesn't export per-feature mean/std for standardization.

---

## 7. Minor Code Quality Issues

|File|Issue|Fix|
|---|---|---|
|`torque.py:139`|`_smooth_sign` default `eps=0.05` doesn't match `config.py` value of `0.0757`|Use `eps=None` and default to `C.FRICTION_EPS`|
|`main.py:103`|`dummy_current = np.zeros_like(tau_load)` passed to diagnostics that no longer use current|Remove the parameter from `run_all()` signature entirely|
|`calibrate_v2.py:53`|Builds model with `mass_scale=1.0` but this is implicit — should be explicit keyword|Already fine, just a style note|
|`bulk_analyze.py:180`|`safe_ratio(rnea_rms, js["load_rms"] or 1e-9)` — the `or` fallback can mask `None` from `safe_float`|Check for `None` explicitly before calling `safe_ratio`|
|`data_loader.py:10`|Type hint `tuple[dict, dict, int]` requires Python 3.9+|Use `from __future__ import annotations` or `Tuple[dict, dict, int]` for wider compatibility|
|`calibrate_friction.py:344`|Random subsampling without seed — non-reproducible|Add `rng = np.random.RandomState(42)`|
|Multiple files|`import copy` in `calibrate_v2.py` is imported but never used|Remove unused import|

---

## 8. Improvement Priority Matrix

|Priority|Improvement|Impact|Effort|
|---|---|---|---|
|**P0**|Add PINN data export script|Unblocks downstream work|Low|
|**P0**|Fix `_smooth_sign` default eps mismatch|Prevents silent bugs|Trivial|
|**P1**|Merge `plots.py`/`plots_bulk.py`|Eliminates maintenance burden|Medium|
|**P1**|Replace `np.gradient` + smooth with Savitzky-Golay|Better signal quality|Low|
|**P1**|Add basic unit tests for `utils.py` and `torque.py`|Prevents regressions|Medium|
|**P1**|Fix `fix_timestamps()` to use interpolation|Prevents velocity spikes|Low|
|**P2**|Vectorize/parallelize RNEA loop|~3-5x speedup for bulk|Medium|
|**P2**|Add `pyproject.toml` and remove `sys.path` hacks|Cleaner project structure|Low|
|**P2**|Split `config.py` into constants + calibration|Better organization|Low|
|**P2**|Add Python `logging`|Searchable bulk output|Medium|
|**P3**|Cache intermediate results as `.npz`|Faster re-runs|Medium|
|**P3**|Parallelize epsilon sweep|Minor speedup|Low|
|**P3**|Add seed to nonlinear subsampling|Reproducibility|Trivial|

---

## 9. What's Already Done Well

It would be unfair to end without acknowledging what this codebase does right, because there's a lot:

- **The four-method friction calibration with cross-validation** is genuinely clever. The insight that asymmetry analysis cancels gravity error while bias-aware fitting absorbs it gives real robustness that single-method approaches lack.
- **The documentation is research-grade.** The `readme.md`, `codebase.md`, `Friction_Calibration.md`, and `Detailed_Math.md` files form a complete mathematical reference. Many academic codebases ship with a fraction of this documentation.
- **Design decisions are recorded with failure history.** The appendices documenting why velocity filtering failed (collinearity), why certain sign conventions were tested, and why per-joint epsilon was reduced to a global value — this is invaluable for anyone maintaining the code.
- **The diagnostic checks in `diagnostics.py`** catch real failure modes (joint ordering mismatches, mass sanity, gravity direction, timestamp quality). These are the kinds of checks that prevent weeks of debugging.
- **The physics is correct.** The RNEA implementation via Pinocchio, the mass scaling linearity argument, the smooth friction model with tanh — all are mathematically sound and well-justified.

The codebase is solid for its current purpose. The improvements above would make it production-ready and significantly smoother for the PINN training phase that follows.