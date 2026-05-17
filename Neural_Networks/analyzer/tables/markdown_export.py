"""Best-per-architecture markdown summary table."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..io.records import arch_short_label, best_per_type, split_scalar

logger = logging.getLogger(__name__)


def export_summary_markdown(
    groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    best_list = best_per_type(groups)
    lines: list[str] = [
        "# Grid search — best run per architecture",
        "",
        "Best = minimum **test** `rmse_traj_macro` (trajectory-macro RMSE, "
        "the headline estimator; falls back to `rmse_pooled` for older runs) "
        "within each `model_type`.",
        "",
        "| Architecture | test RMSE (N·m) | test R2 overall | test MAE | run id |",
        "|-------------|-----------------|----------------|----------|--------|",
    ]
    for r in best_list:
        mtype = r.get("model_type", "?")
        run_id = str(r.get("run_id", ""))[:80]
        trmse = split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled")
        tr2   = split_scalar(r, "test", "r2_overall")
        mae   = split_scalar(r, "test", "mae_mean")
        lines.append(
            f"| {arch_short_label(mtype)} | {trmse:.5f} | {tr2:.5f} | {mae:.5f} | `{run_id}` |"
        )
    lines.extend(["", "## Test RMSE range across *all* runs in grid", ""])

    for mtype in sorted(groups.keys()):
        recs = groups[mtype]
        rmses: list[float] = []
        r2s: list[float] = []
        for r in recs:
            a = split_scalar(r, "test", "rmse_traj_macro", "rmse_pooled")
            b = split_scalar(r, "test", "r2_overall")
            if a == a and np.isfinite(a):
                rmses.append(a)
            if b == b and np.isfinite(b):
                r2s.append(b)
        if rmses:
            lines.append(
                f"- **{arch_short_label(mtype)}** ({mtype}): RMSE in [{min(rmses):.5f}, {max(rmses):.5f}] "
                f"N·m over {len(rmses)} run(s); R2 in [{min(r2s):.5f}, {max(r2s):.5f}]."
            )

    out = output_dir / "summary_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote: %s", out)
