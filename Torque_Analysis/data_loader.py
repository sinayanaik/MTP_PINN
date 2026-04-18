"""
Load a single JSON log file → L (arrays), M (metadata).
data_loader.py
"""

from __future__ import annotations

import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _safe_get(data: dict, key: str, context: str = ""):
    """Get key from dict with informative error on missing keys."""
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {context or 'data'}")
    return data[key]


def load_log(json_path: str) -> tuple[dict, dict, int]:
    """
    Parameters
    ----------
    json_path : path to the JSON log file

    Returns
    -------
    L : dict of NumPy arrays  (time-series)
    M : dict of scalars/dicts (metadata)
    N : number of log entries
    """
    logger.info("Loading %s", json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    log = _safe_get(data, "log", json_path)
    N = len(log)

    if N == 0:
        raise ValueError(f"Empty log array in {json_path}")

    # --- Time-series arrays ---
    L = {
        # Scalars — (N,)
        "t":        np.array([e["t"]       for e in log]),
        "wp":       np.array([e["wp"]      for e in log]),
        "ee_err":   np.array([e["ee_err"]  for e in log]),

        # Per-joint — (N, 6)
        "cmd_pos":  np.array([e["cmd_pos"] for e in log]),
        "cmd_vel":  np.array([e["cmd_vel"] for e in log]),
        "cmd_acc":  np.array([e["cmd_acc"] for e in log]),
        "act_pos":  np.array([e["act_pos"] for e in log]),
        "act_vel":  np.array([e["act_vel"] for e in log]),
        "load":     np.array([e["load"]    for e in log]),
        "voltage":  np.array([e["voltage"] for e in log]),
        "current":  np.array([e["current"] for e in log]),

        # End-effector — (N, 3)
        "cmd_ee":   np.array([e["cmd_ee"]  for e in log]),
        "act_ee":   np.array([e["act_ee"]  for e in log]),
    }

    # --- Metadata ---
    M = {
        "run_id":        data["run_id"],
        "label":         data["label"],
        "success":       data["success"],
        "dof":           data["dof"],
        "servo_ids":     data["servo_ids"],
        "ticks_to_rad":  data["ticks_to_rad"],
        "ticks_per_rev": data["ticks_per_revolution"],
        "num_entries":   data["num_log_entries"],
        "ctrl_hz":       data["actual_control_hz"],
        "fb_hz":         data["actual_feedback_hz"],
        "duration":      data["actual_duration_sec"],
        "tracking":      data["tracking_quality"],
        "geometry":      data["geometry"],
        "joint_map":     data["joint_mapping"],
    }

    logger.debug("  Loaded %d entries, duration %.2fs", N, M["duration"])
    return L, M, N


def print_summary(L: dict, M: dict, N: int):
    """Quick verification print."""
    print(f"Run:      {M['run_id']}")
    print(f"Label:    {M['label']}")
    print(f"Success:  {M['success']}")
    print(f"Entries:  {N}")
    print(f"Duration: {L['t'][-1] - L['t'][0]:.4f} sec")
    print(f"Ctrl Hz:  {M['ctrl_hz']:.1f}")
    print(f"FB Hz:    {M['fb_hz']:.1f}")
    print(f"\nArray shapes:")
    for key, arr in L.items():
        print(f"  L['{key}']".ljust(22) + f"→ {arr.shape}")
