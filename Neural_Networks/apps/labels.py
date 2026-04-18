"""
Neural_Networks.data.labels
============================
Compact, filesystem-safe parameter tags and short summary strings
derived from preprocessing metadata dicts.

These functions are purely computational (no I/O, no UI).  They are
consumed by:
  - apps/preprocess.py — to build the output folder name
  - Neural_Networks.apps.scanner — inside scan_existing_datasets()
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Single-character mode tokens (used inside the filesystem-safe param tag)
# ---------------------------------------------------------------------------

def sg_mode_token(mode: str) -> str:
    """Convert a Savitzky-Golay edge-mode string to a single filesystem-safe char.

    Maps: interp→i, nearest→n, mirror→m, wrap→w, constant→c.
    Unknown modes fall back to the first two characters.
    """
    m = (mode or "interp").lower()
    return {"interp": "i", "nearest": "n", "mirror": "m",
            "wrap": "w", "constant": "c"}.get(m, m[:2] if len(m) >= 2 else m)


def _wp(w: Any, p: Any) -> str:
    """Format ``window``/``polyorder`` as ``<w>p<p>``."""
    return f"{w}p{p}"


# ---------------------------------------------------------------------------
# Param tag — encodes all preprocessing choices in one compact string
# ---------------------------------------------------------------------------

def build_param_tag(
    *,
    q_smooth: bool,
    q_win: int,
    q_poly: int,
    deriv_win: int,
    deriv_poly: int,
    deriv_mode: str,
    qdd_locked: bool,
    qdd_win: int,
    qdd_poly: int,
    qdd_mode: str,
    tau_smooth: bool,
    tau_win: int,
    tau_poly: int,
    tau_ana_pf: bool,
    tau_ana_win: int,
    tau_ana_poly: int,
    use_rnea: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    trim_front_pct: float,
    trim_back_pct: float,
) -> str:
    """Build a single filesystem-friendly token (no spaces) summarising
    all preprocessing pipeline settings.

    The tag encodes: q smoothing, qd/qdd differentiation params, tau_measured
    smoothing, RNEA post-filter, RNEA on/off, train/val/test split ratios,
    and trim percentages.  It is embedded in the output folder name so the
    full pipeline is visible from the directory listing alone.
    """
    q_part  = f"q{_wp(q_win, q_poly)}" if q_smooth else "qraw"
    d_part  = f"d{_wp(deriv_win, deriv_poly)}{sg_mode_token(deriv_mode)}"
    if qdd_locked:
        dd_part = "ddL"
    else:
        dd_part = f"dd{_wp(qdd_win, qdd_poly)}{sg_mode_token(qdd_mode)}"
    m_part  = f"m{_wp(tau_win, tau_poly)}" if tau_smooth else "mraw"
    ap_part = f"ap{_wp(tau_ana_win, tau_ana_poly)}" if tau_ana_pf else "ap0"
    r_part  = "R" if use_rnea else "r"
    tr  = int(round(train_ratio * 100))
    va  = int(round(val_ratio   * 100))
    te  = int(round(test_ratio  * 100))
    spl = f"{tr}v{va}t{te}"
    tf  = str(trim_front_pct).replace(".", "p")
    tb  = str(trim_back_pct).replace(".", "p")
    trim = f"f{tf}t{tb}"
    tag  = "_".join([q_part, d_part, dd_part, m_part, ap_part, r_part, spl, trim])
    # Sanitise: keep only alphanumerics, dots, underscores, hyphens
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tag)


def param_tag_from_metadata(meta: dict) -> str:
    """Reconstruct the param tag from a loaded metadata.json dict.

    Extracts the same parameters that build_param_tag() accepts and
    delegates to it, so a folder-tag can be regenerated from any saved run.
    """
    pp      = meta.get("preprocessing", {}) or {}
    trim    = pp.get("trim", {}) or {}
    q       = pp.get("q_smooth", {}) or {}
    deriv   = pp.get("differentiation", {}) or {}
    tau_sm  = pp.get("tau_measured_smooth", {}) or {}
    tau_a   = pp.get("tau_analytical", {}) or {}
    tau_pf  = pp.get("tau_analytical_postfilter", {}) or {}
    split   = meta.get("split", {}) or {}
    ratios  = split.get("ratios", {}) or {}

    q_smooth = bool(q.get("enabled", False))
    q_win    = int(q.get("window_length", 15))
    q_poly   = int(q.get("polyorder", 3))

    deriv_qd = deriv.get("qd") if isinstance(deriv.get("qd"), dict) else None
    if deriv_qd is not None:
        deriv_win  = int(deriv_qd.get("window_length", 15))
        deriv_poly = int(deriv_qd.get("polyorder", 3))
        deriv_mode = str(deriv_qd.get("mode", "interp"))
        deriv_qdd  = deriv.get("qdd") or {}
        qdd_locked = bool(deriv_qdd.get("locked_to_qd", True))
        qdd_win    = int(deriv_qdd.get("window_length", deriv_win))
        qdd_poly   = int(deriv_qdd.get("polyorder", deriv_poly))
        qdd_mode   = str(deriv_qdd.get("mode", deriv_mode))
    else:
        deriv_win  = int(deriv.get("window_length", 15))
        deriv_poly = int(deriv.get("polyorder", 3))
        deriv_mode = str(deriv.get("mode", "interp"))
        qdd_locked = True
        qdd_win, qdd_poly, qdd_mode = deriv_win, deriv_poly, deriv_mode

    tau_smooth  = bool(tau_sm.get("enabled", False))
    tau_win     = int(tau_sm.get("window_length", 15))
    tau_poly    = int(tau_sm.get("polyorder", 3))
    tau_ana_pf  = bool(tau_pf.get("enabled", False))
    tau_ana_win = int(tau_pf.get("window_length", 15))
    tau_ana_poly= int(tau_pf.get("polyorder", 3))
    use_rnea    = bool(tau_a.get("rnea_enabled", False))

    return build_param_tag(
        q_smooth=q_smooth, q_win=q_win, q_poly=q_poly,
        deriv_win=deriv_win, deriv_poly=deriv_poly, deriv_mode=deriv_mode,
        qdd_locked=qdd_locked, qdd_win=qdd_win, qdd_poly=qdd_poly, qdd_mode=qdd_mode,
        tau_smooth=tau_smooth, tau_win=tau_win, tau_poly=tau_poly,
        tau_ana_pf=tau_ana_pf, tau_ana_win=tau_ana_win, tau_ana_poly=tau_ana_poly,
        use_rnea=use_rnea,
        train_ratio=float(ratios.get("train", 0.7)),
        val_ratio=float(ratios.get("val", 0.15)),
        test_ratio=float(ratios.get("test", 0.15)),
        trim_front_pct=float(trim.get("front_percent", 0.0)),
        trim_back_pct=float(trim.get("back_percent", 0.0)),
    )


# ---------------------------------------------------------------------------
# Pipeline compact codes — short strings for table cells
# ---------------------------------------------------------------------------

def pipeline_compact_codes(meta: dict) -> dict[str, str]:
    """Return a dict of short codes for the dataset picker table.

    Each value is a compact string: ``w<win>/p<poly>/<mode>`` for SG,
    ``raw`` for no filtering, ``lock→qd`` when qdd is derived from qd.

    Keys: ``q``, ``qd``, ``qdd``, ``tau_m``, ``rnea``, ``tau_apf``.
    """
    pp      = meta.get("preprocessing", {}) or {}
    q       = pp.get("q_smooth", {}) or {}
    deriv   = pp.get("differentiation", {}) or {}
    tau_sm  = pp.get("tau_measured_smooth", {}) or {}
    tau_a   = pp.get("tau_analytical", {}) or {}
    tau_pf  = pp.get("tau_analytical_postfilter", {}) or {}

    q_cell  = (f"w{q.get('window_length')}/p{q.get('polyorder')}"
               if q.get("enabled") else "raw")

    deriv_qd = deriv.get("qd") if isinstance(deriv.get("qd"), dict) else None
    if deriv_qd is not None:
        qd_cell = (f"w{deriv_qd.get('window_length')}/p{deriv_qd.get('polyorder')}/"
                   f"{sg_mode_token(str(deriv_qd.get('mode', '')))}")
        deriv_qdd = deriv.get("qdd") or {}
        if deriv_qdd.get("locked_to_qd", True):
            qdd_cell = "lock→qd"
        else:
            qdd_cell = (f"w{deriv_qdd.get('window_length')}/p{deriv_qdd.get('polyorder')}/"
                        f"{sg_mode_token(str(deriv_qdd.get('mode', '')))}")
    else:
        qd_cell  = f"L{deriv.get('window_length')}/{deriv.get('polyorder')}"
        qdd_cell = "—"

    tm_cell  = (f"w{tau_sm.get('window_length')}/p{tau_sm.get('polyorder')}"
                if tau_sm.get("enabled") else "raw")
    rnea_cell = "Y" if tau_a.get("rnea_enabled") else "N"
    apf_cell  = (f"w{tau_pf.get('window_length')}/p{tau_pf.get('polyorder')}"
                 if tau_pf.get("enabled") else "off")

    return {"q": q_cell, "qd": qd_cell, "qdd": qdd_cell,
            "tau_m": tm_cell, "rnea": rnea_cell, "tau_apf": apf_cell}


# ---------------------------------------------------------------------------
# Sample count formatters and one-liner summaries
# ---------------------------------------------------------------------------

def _compact_count(n: Any) -> str:
    """Format large sample counts as compact strings (e.g. 12.3k, 1.2M)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n >= 1_000_000:
        x = n / 1_000_000
        s = f"{x:.1f}".rstrip("0").rstrip(".")
        return f"{s}M"
    if n >= 1000:
        x = n / 1000
        s = f"{x:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return str(n)


def dataset_stats_one_liner(meta: dict) -> str:
    """Return one compact line: sample counts, split ratios, trim percentages.

    Example: ``n 42k/9k/9k  70/15/15%  trim 0%/0%``
    """
    split_meta = meta.get("split", {}) or {}
    ss   = split_meta.get("stats", {}) or {}
    sr   = split_meta.get("ratios", {}) or {}
    pp   = meta.get("preprocessing", {}) or {}
    trim = pp.get("trim", {}) or {}

    n_tr  = ss.get("train", {}).get("n_samples", 0)
    n_val = ss.get("val", {}).get("n_samples", 0)
    n_te  = ss.get("test", {}).get("n_samples", 0)
    tr    = int(round(float(sr.get("train", 0)) * 100))
    va    = int(round(float(sr.get("val",   0)) * 100))
    te    = int(round(float(sr.get("test",  0)) * 100))
    tf    = trim.get("front_percent", 0)
    tb    = trim.get("back_percent",  0)
    return (f"n {_compact_count(n_tr)}/{_compact_count(n_val)}/{_compact_count(n_te)}"
            f"  {tr}/{va}/{te}%  trim {tf}%/{tb}%")


def compact_summary_one_line(meta: dict, max_len: int = 110) -> str:
    """Single line combining key pipeline facts; truncated with «…» if needed."""
    try:
        ptag = param_tag_from_metadata(meta)
    except Exception:
        ptag = "?"
    codes = pipeline_compact_codes(meta)
    code_s = " ".join(f"{k}={v}" for k, v in codes.items()) if codes else ""
    s = " | ".join(x for x in (ptag, dataset_stats_one_liner(meta), code_s) if x)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Output directory naming
# ---------------------------------------------------------------------------

def default_run_dir(
    train_dir: "Path | str",
    *,
    q_smooth: bool,
    q_win: int,
    q_poly: int,
    deriv_win: int,
    deriv_poly: int,
    deriv_mode: str,
    qdd_locked: bool,
    qdd_win: int,
    qdd_poly: int,
    qdd_mode: str,
    tau_smooth: bool,
    tau_win: int,
    tau_poly: int,
    tau_ana_pf: bool,
    tau_ana_win: int,
    tau_ana_poly: int,
    use_rnea: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    trim_front_pct: float,
    trim_back_pct: float,
    stamp: "datetime | None" = None,
    unique_suffix: str = "",
) -> str:
    """Return full output path: ``<train_dir>/run_MMDD_HHMM_<param_tag>_<uid>``.

    The param-tag component encodes all preprocessing choices so the run
    can be identified without opening metadata.json.
    """
    stamp = stamp or datetime.now()
    tag = build_param_tag(
        q_smooth=q_smooth, q_win=q_win, q_poly=q_poly,
        deriv_win=deriv_win, deriv_poly=deriv_poly, deriv_mode=deriv_mode,
        qdd_locked=qdd_locked, qdd_win=qdd_win, qdd_poly=qdd_poly, qdd_mode=qdd_mode,
        tau_smooth=tau_smooth, tau_win=tau_win, tau_poly=tau_poly,
        tau_ana_pf=tau_ana_pf, tau_ana_win=tau_ana_win, tau_ana_poly=tau_ana_poly,
        use_rnea=use_rnea,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        trim_front_pct=trim_front_pct, trim_back_pct=trim_back_pct,
    )
    u = (unique_suffix or "").strip() or uuid.uuid4().hex[:6]
    folder = f"run_{stamp.strftime('%m%d_%H%M')}_{tag}_{u}"
    return str(Path(train_dir) / folder)


# ---------------------------------------------------------------------------
# Display-oriented table helpers (Rich tables in TUI and training session)
# ---------------------------------------------------------------------------

def preprocessing_pipeline_table_rows(meta: dict) -> list[dict[str, str]]:
    """
    Structured rows for a Rich table: one row per signal / pipeline stage.
    All splits (train/val/test) share the same preprocessing — this describes the pipeline only.
    Keys: quantity, treatment, window, poly, mode_notes.
    """
    pp = meta.get("preprocessing", {}) or {}
    q = pp.get("q_smooth", {}) or {}
    deriv = pp.get("differentiation", {}) or {}
    tau_sm = pp.get("tau_measured_smooth", {}) or {}
    tau_a = pp.get("tau_analytical", {}) or {}
    tau_pf = pp.get("tau_analytical_postfilter", {}) or {}

    rows: list[dict[str, str]] = []

    if q.get("enabled"):
        rows.append({
            "quantity": "q — joint positions",
            "treatment": "Savitzky–Golay smooth",
            "window": str(q.get("window_length", "—")),
            "poly": str(q.get("polyorder", "—")),
            "mode_notes": "—",
        })
    else:
        rows.append({
            "quantity": "q — joint positions",
            "treatment": "Raw (no SG)",
            "window": "—",
            "poly": "—",
            "mode_notes": "—",
        })

    deriv_qd = deriv.get("qd") if isinstance(deriv.get("qd"), dict) else None
    if deriv_qd is not None:
        rows.append({
            "quantity": "qd — velocities",
            "treatment": "SG 1st derivative",
            "window": str(deriv_qd.get("window_length", "—")),
            "poly": str(deriv_qd.get("polyorder", "—")),
            "mode_notes": f"edge: {deriv_qd.get('mode', '—')}",
        })
        deriv_qdd = deriv.get("qdd") or {}
        if deriv_qdd.get("locked_to_qd", True):
            rows.append({
                "quantity": "qdd — accelerations",
                "treatment": "Analytical 2nd deriv (same SG fit as qd)",
                "window": "—",
                "poly": "—",
                "mode_notes": "locked to qd chain",
            })
        else:
            rows.append({
                "quantity": "qdd — accelerations",
                "treatment": "Separate SG 2nd derivative",
                "window": str(deriv_qdd.get("window_length", "—")),
                "poly": str(deriv_qdd.get("polyorder", "—")),
                "mode_notes": f"edge: {deriv_qdd.get('mode', '—')}",
            })
    else:
        rows.append({
            "quantity": "qd / qdd — kinematics",
            "treatment": "Legacy single SG differentiation",
            "window": str(deriv.get("window_length", "—")),
            "poly": str(deriv.get("polyorder", "—")),
            "mode_notes": f"mode: {deriv.get('mode', '—')}",
        })

    if tau_sm.get("enabled"):
        rows.append({
            "quantity": "τ_measured — labels",
            "treatment": "Savitzky–Golay smooth",
            "window": str(tau_sm.get("window_length", "—")),
            "poly": str(tau_sm.get("polyorder", "—")),
            "mode_notes": "—",
        })
    else:
        rows.append({
            "quantity": "τ_measured — labels",
            "treatment": "Raw (no SG)",
            "window": "—",
            "poly": "—",
            "mode_notes": "—",
        })

    if tau_a.get("rnea_enabled"):
        rows.append({
            "quantity": "τ_analytical — RNEA",
            "treatment": "Pinocchio RNEA + friction (preprocessor)",
            "window": "—",
            "poly": "—",
            "mode_notes": "enabled",
        })
    else:
        rows.append({
            "quantity": "τ_analytical — RNEA",
            "treatment": "Disabled",
            "window": "—",
            "poly": "—",
            "mode_notes": "no RNEA column from preprocessor",
        })

    if tau_pf.get("enabled"):
        rows.append({
            "quantity": "τ_analytical — post-filter",
            "treatment": "Savitzky–Golay on τ_a",
            "window": str(tau_pf.get("window_length", "—")),
            "poly": str(tau_pf.get("polyorder", "—")),
            "mode_notes": "—",
        })
    else:
        rows.append({
            "quantity": "τ_analytical — post-filter",
            "treatment": "Off",
            "window": "—",
            "poly": "—",
            "mode_notes": "—",
        })

    return rows


def train_val_test_split_table_rows(meta: dict) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Rows for a per-split sample table + trim footer dict with keys front_pct, back_pct, total.
    """
    split_meta = meta.get("split", {}) or {}
    ss = split_meta.get("stats", {}) or {}
    sr = split_meta.get("ratios", {}) or {}
    pp = meta.get("preprocessing", {}) or {}
    trim = pp.get("trim", {}) or {}

    def _n(split_key: str) -> int:
        try:
            return int(ss.get(split_key, {}).get("n_samples", 0) or 0)
        except (TypeError, ValueError):
            return 0

    n_tr = _n("train")
    n_val = _n("val")
    n_te = _n("test")
    total = max(1, n_tr + n_val + n_te)

    def _pct(split_key: str, n: int) -> str:
        r = float(sr.get(split_key, 0) or 0)
        if r > 0:
            return f"{int(round(r * 100))}%"
        return f"{100.0 * n / total:.1f}%"

    rows = [
        {
            "split": "train",
            "n_samples": f"{n_tr:,}",
            "config_ratio": _pct("train", n_tr),
            "fraction_of_all": f"{100.0 * n_tr / total:.2f}%",
        },
        {
            "split": "val",
            "n_samples": f"{n_val:,}",
            "config_ratio": _pct("val", n_val),
            "fraction_of_all": f"{100.0 * n_val / total:.2f}%",
        },
        {
            "split": "test",
            "n_samples": f"{n_te:,}",
            "config_ratio": _pct("test", n_te),
            "fraction_of_all": f"{100.0 * n_te / total:.2f}%",
        },
    ]
    footer = {
        "front_pct": str(trim.get("front_percent", 0)),
        "back_pct": str(trim.get("back_percent", 0)),
        "total": f"{total:,}",
    }
    return rows, footer
