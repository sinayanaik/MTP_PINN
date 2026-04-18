"""
Shared utility functions.
utils.py

Signal-processing pipeline for joint-torque estimation.

Key design decisions
--------------------
1. Timestamp repair (fix_timestamps)
   Servo controllers sometimes emit duplicate or out-of-order timestamps.
   We interpolate the bad timestamps from neighbouring good ones, then add
   a 1 ns nudge to guarantee strict monotonicity.  np.gradient requires
   strictly positive spacing; zero gaps would produce ±inf velocity spikes.

2. Single-pass Savitzky-Golay differentiation (velocity_and_acceleration_from_pos)
   The physically correct way to obtain qd and qdd from noisy position data
   is to fit ONE local polynomial to q and read off its 1st and 2nd
   derivatives:
       qd  = p'(t)   (deriv=1 on q)
       qdd = p''(t)  (deriv=2 on q)
   The naive alternative — differentiating qd a second time — applies the SG
   filter twice.  In the frequency domain, two passes:
       • Square the magnitude attenuation  |H(ω)|² ← bad for keeping signal
       • Double the phase lag              ∠H(ω)·2  ← bad for RNEA waveform
   This explains the high NRMSE (~43-47 %) observed on J2/J3 even when the
   RNEA RMS amplitude matches the load: the waveform is phase-shifted, so the
   point-wise squared error is large even though the power is correct.

3. Clip bounds
   Derived from C.VEL_CLIP and C.ACC_CLIP (config.py).  These are generous
   hardware limits used only to catch numerical outliers from differentiation
   — not physical limits that would constrain the motion.
"""

from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from . import config as C

logger = logging.getLogger(__name__)


def ticks_to_radians(act_pos: np.ndarray, joint_map: list,
                     ticks_to_rad: float, dof: int = 6) -> np.ndarray:
    """
    Convert encoder tick counts to joint angles in radians (URDF frame).

    Each joint has a direction sign and a zero-position offset (ticks_center)
    that were determined during physical calibration.  The conversion is:

        q_i = direction_i × (ticks_i − ticks_center_i) × ticks_to_rad

    ticks_to_rad = 2π / ticks_per_revolution (constant for a given encoder).
    """
    N = act_pos.shape[0]
    act_rad = np.zeros((N, dof))
    for i, jm in enumerate(joint_map):
        act_rad[:, i] = (
            jm["direction"]
            * (act_pos[:, i] - jm["ticks_center"])
            * ticks_to_rad
        )
    return act_rad


def smooth(x: np.ndarray, window: int, axis: int = 0) -> np.ndarray:
    """
    Symmetric (zero-phase) moving average via uniform_filter1d.

    Zero-phase means the filter introduces no time-delay — the output at
    sample k is the mean of the window centred on k.  This is important for
    torque estimation: a phase-shifted velocity would produce an RNEA torque
    that is temporally misaligned with the measured load torque.
    """
    return uniform_filter1d(x, size=window, axis=axis)


def fix_timestamps(t: np.ndarray) -> np.ndarray:
    """
    Repair duplicate or non-monotonic timestamps.

    Why this is needed
    ------------------
    The servo feedback loop runs at an independent rate from the logger.
    When the logger flushes a burst of packets at once, multiple consecutive
    log entries can carry identical timestamps.  np.gradient divides by the
    spacing Δt; a zero spacing produces a ±inf velocity spike that then
    contaminates the entire SG window.

    Strategy
    --------
    1. Identify "bad" indices (those where Δt ≤ 0).
    2. Linearly interpolate their timestamps from the surrounding good ones
       (np.interp on the index axis).  This gives physically plausible values.
    3. Final 1 ns nudge: enforce strict monotonicity to guard against the
       rare case where two good timestamps happened to be exactly equal
       (floating-point coincidence after the interp step).
    """
    t_fixed = t.copy().astype(float)
    dt = np.diff(t_fixed)
    bad_mask = dt <= 0

    n_bad = bad_mask.sum()
    if n_bad == 0:
        return t_fixed

    logger.warning("Fixing %d non-monotonic timestamps (%.1f%%)",
                   n_bad, 100 * n_bad / len(dt))

    # Mark which indices are "good" (not preceded by a bad dt)
    good_mask = np.ones(len(t_fixed), dtype=bool)
    # Index i+1 is bad if dt[i] <= 0
    bad_indices = np.where(bad_mask)[0] + 1
    good_mask[bad_indices] = False

    good_idx = np.where(good_mask)[0]

    if len(good_idx) < 2:
        # Fallback: too many bad timestamps, use uniform spacing
        logger.warning("Too many bad timestamps — using uniform spacing")
        t_fixed = np.linspace(float(t[0]), float(t[-1]), len(t))
        return t_fixed

    # Interpolate bad timestamps from surrounding good ones
    t_fixed = np.interp(np.arange(len(t)), good_idx, t[good_idx])

    # Final safety pass: enforce strict monotonicity.
    # np.gradient divides by spacing differences, so even a single
    # zero gap (from floating-point coincidence) causes RuntimeWarning.
    # Use a 1 ns minimum increment to guarantee dx > 0 everywhere.
    _EPS_T = 1e-9  # 1 nanosecond
    for k in range(1, len(t_fixed)):
        if t_fixed[k] <= t_fixed[k - 1]:
            t_fixed[k] = t_fixed[k - 1] + _EPS_T

    return t_fixed


def _savgol_window(signal_len: int, requested_win: int,
                   polyorder: int) -> int:
    """
    Return a valid Savitzky-Golay window length.

    scipy.signal.savgol_filter requires:
      • window_length is odd  (so the window is symmetric around each sample)
      • window_length > polyorder  (need more data points than polynomial terms)
      • window_length <= signal length

    We adjust the requested window upward/downward as needed.
    """
    # Must be odd — SG uses a symmetric window of (win-1)/2 samples either side
    win = requested_win if requested_win % 2 == 1 else requested_win + 1
    # Must be strictly greater than polyorder
    min_win = polyorder + 1
    if min_win % 2 == 0:
        min_win += 1
    win = max(win, min_win)
    # Must not exceed signal length (also keep odd)
    max_win = signal_len if signal_len % 2 == 1 else signal_len - 1
    win = min(win, max_win)
    return win


def velocity_and_acceleration_from_pos(
        q: np.ndarray, t: np.ndarray,
        smooth_window: int = None) -> tuple:
    """
    Compute joint velocity and acceleration from position — one SG pass each.

    Why single-pass matters
    -----------------------
    The naive pipeline is:
        qd  = SG(q,  deriv=1)   ← 1st derivative of position
        qdd = SG(qd, deriv=1)   ← 1st derivative of velocity  ← WRONG

    Applying SG twice filters the signal twice.  In the frequency domain
    the magnitude is attenuated as |H(ω)|² and the phase shift doubles.
    For the inertial term M(q)·q̈ in RNEA, a phase-shifted q̈ produces a
    torque waveform that is correct in RMS amplitude but temporally offset
    from the true torque.  This explains NRMSE of 43-47 % on J2/J3 even
    when RNEA/Load ≈ 1.0.

    The correct approach (this function):
        qd  = SG(q, deriv=1)   ← 1st derivative of q
        qdd = SG(q, deriv=2)   ← 2nd derivative of q

    Both are taken from the SAME polynomial fit of q.  Inside each sliding
    window of length w, SG fits a degree-n polynomial p(t) to q(t) in the
    least-squares sense.  The derivatives are exact:
        p'(t_centre)  → qd   at that sample
        p''(t_centre) → qdd  at that sample

    Because the same polynomial is used for both, no additional filtering
    is introduced when going from qd to qdd.

    Parameters
    ----------
    q             : (N, nq)  joint positions in radians
    t             : (N,)     timestamps in seconds
    smooth_window : SG window length (default C.SMOOTH_WINDOW)

    Returns
    -------
    qd  : (N, nq)  joint velocities in rad/s
    qdd : (N, nq)  joint accelerations in rad/s²
    """
    if smooth_window is None:
        smooth_window = C.SMOOTH_WINDOW

    t_safe  = fix_timestamps(t)
    dt_mean = float(np.mean(np.diff(t_safe)))
    N       = q.shape[0]

    if dt_mean <= 0:
        # Degenerate case: all timestamps identical after repair.
        # Fall back to gradient + smooth so we still return valid arrays.
        logger.warning("Mean timestep <= 0 after repair; falling back to gradient")
        qd  = numerical_velocity(q, t, smooth_window, method="gradient")
        qdd = numerical_acceleration(qd, t, smooth_window, method="gradient")
        return qd, qdd

    win = _savgol_window(N, smooth_window, C.SAVGOL_POLYORDER)

    # Both derivatives from the same polynomial fit of q — no double-filtering.
    # delta=dt_mean treats the signal as uniformly sampled at the mean rate,
    # which is robust to the occasional 1 ns nudge added by fix_timestamps.
    qd  = savgol_filter(q, window_length=win, polyorder=C.SAVGOL_POLYORDER,
                        deriv=1, delta=dt_mean, axis=0)
    qdd = savgol_filter(q, window_length=win, polyorder=C.SAVGOL_POLYORDER,
                        deriv=2, delta=dt_mean, axis=0)

    # Clip at generous hardware limits to catch numerical outliers near window
    # boundaries.  These limits are set in config.py and should never be hit
    # for normal robot motion.
    qd  = np.clip(qd,  -C.VEL_CLIP, C.VEL_CLIP)
    qdd = np.clip(qdd, -C.ACC_CLIP, C.ACC_CLIP)
    qd  = np.nan_to_num(qd,  nan=0.0, posinf=0.0, neginf=0.0)
    qdd = np.nan_to_num(qdd, nan=0.0, posinf=0.0, neginf=0.0)

    return qd, qdd


def numerical_velocity(q: np.ndarray, t: np.ndarray,
                       smooth_window: int = None,
                       method: str = None) -> np.ndarray:
    """
    Compute joint velocities from position data.

    Prefer velocity_and_acceleration_from_pos() when you also need qdd —
    it computes both derivatives from a single SG polynomial fit, avoiding
    the double-filtering phase error that comes from calling this function
    and then numerical_acceleration() separately.

    Parameters
    ----------
    q             : (N, nq)  joint positions in radians
    t             : (N,)     timestamps in seconds
    smooth_window : smoothing window (default C.SMOOTH_WINDOW)
    method        : "savgol" (default) or "gradient"
    """
    if smooth_window is None:
        smooth_window = C.SMOOTH_WINDOW
    if method is None:
        method = C.DIFF_METHOD

    t_safe = fix_timestamps(t)
    N = q.shape[0]

    if method == "savgol":
        dt_mean = np.mean(np.diff(t_safe))
        if dt_mean <= 0:
            logger.warning("Mean timestep <= 0 after repair; falling back to gradient")
            method = "gradient"
        else:
            win = _savgol_window(N, smooth_window, C.SAVGOL_POLYORDER)
            qd = savgol_filter(q, window_length=win,
                               polyorder=C.SAVGOL_POLYORDER,
                               deriv=1, delta=dt_mean, axis=0)

    if method == "gradient":
        qd = np.gradient(q, t_safe, axis=0)
        qd = smooth(qd, smooth_window)

    qd = np.clip(qd, -C.VEL_CLIP, C.VEL_CLIP)
    qd = np.nan_to_num(qd, nan=0.0, posinf=0.0, neginf=0.0)
    return qd


def numerical_acceleration(qd: np.ndarray, t: np.ndarray,
                            smooth_window: int = None,
                            method: str = None) -> np.ndarray:
    """
    Compute joint accelerations from velocity data (second-pass differentiation).

    WARNING: calling this after numerical_velocity() applies the SG filter
    twice, which doubles phase lag and squares magnitude attenuation.
    For RNEA use velocity_and_acceleration_from_pos() instead.

    This function is retained for backward compatibility and for the
    gradient fallback path inside velocity_and_acceleration_from_pos().

    Parameters
    ----------
    qd            : (N, nq)  joint velocities in rad/s
    t             : (N,)     timestamps in seconds
    smooth_window : smoothing window (default C.SMOOTH_WINDOW)
    method        : "savgol" (default) or "gradient"
    """
    if smooth_window is None:
        smooth_window = C.SMOOTH_WINDOW
    if method is None:
        method = C.DIFF_METHOD

    t_safe = fix_timestamps(t)
    N = qd.shape[0]

    if method == "savgol":
        dt_mean = np.mean(np.diff(t_safe))
        if dt_mean <= 0:
            logger.warning("Mean timestep <= 0 after repair; falling back to gradient")
            method = "gradient"
        else:
            win = _savgol_window(N, smooth_window, C.SAVGOL_POLYORDER)
            qdd = savgol_filter(qd, window_length=win,
                                polyorder=C.SAVGOL_POLYORDER,
                                deriv=1, delta=dt_mean, axis=0)

    if method == "gradient":
        qdd = np.gradient(qd, t_safe, axis=0)
        qdd = smooth(qdd, smooth_window)

    qdd = np.clip(qdd, -C.ACC_CLIP, C.ACC_CLIP)
    qdd = np.nan_to_num(qdd, nan=0.0, posinf=0.0, neginf=0.0)
    return qdd
