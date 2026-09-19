from __future__ import annotations

from pathlib import Path
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


SNAPSHOT_RE = re.compile(r"^gw(\\d{2})\\.csv$")


def save_projection_snapshot(
    xpts_horizon: pd.DataFrame,
    prediction_dir: Path,
    start_gw: int,
) -> Path:
    """Save the latest pre-deadline projection for one gameweek.

    Repeated runs for the same planning GW overwrite the prior snapshot, so
    the latest model state becomes the version evaluated after the event.
    """
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    current = xpts_horizon.loc[
        xpts_horizon["GW"] == int(start_gw)
    ].copy()

    if current.empty:
        raise ValueError(f"No projection rows found for GW{int(start_gw)}")

    preferred = [
        "Player ID", "Player", "Team", "FPL Pos", "GW",
        "Effective Mins", "Appearance Prob", "Start Prob", "P60",
        "xG", "xA", "Team CS Prob", "Base xPts", "xPts DefCon",
        "xPts Saves", "xPts Bonus", "xPts Model",
        "Attack Prior Source", "xG Source",
    ]

    cols = [c for c in preferred if c in current.columns]
    out = current[cols].copy()

    out.insert(
        0,
        "Projection Timestamp UTC",
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    path = prediction_dir / f"gw{int(start_gw):02d}.csv"
    out.to_csv(path, index=False)
    return path


def evaluate_projection_snapshot(
    snapshot: pd.DataFrame,
    event_live: pd.DataFrame,
) -> dict:
    """Score one saved projection against realised FPL event data."""
    required_projection = {"Player ID", "xPts Model", "Effective Mins"}
    missing_projection = required_projection - set(snapshot.columns)
    if missing_projection:
        raise ValueError(
            "Projection snapshot missing required columns: "
            f"{sorted(missing_projection)}"
        )

    required_actual = {"Player ID", "total_points", "minutes", "starts"}
    missing_actual = required_actual - set(event_live.columns)
    if missing_actual:
        raise ValueError(
            "event_live missing required columns: "
            f"{sorted(missing_actual)}"
        )

    actual = event_live[["Player ID", "total_points", "minutes", "starts"]].copy()
    actual["Player ID"] = pd.to_numeric(actual["Player ID"], errors="raise").astype(int)
    for col in ["total_points", "minutes", "starts"]:
        actual[col] = pd.to_numeric(actual[col], errors="coerce")

    proj = snapshot.copy()
    proj["Player ID"] = pd.to_numeric(proj["Player ID"], errors="raise").astype(int)

    merged = proj.merge(
        actual,
        on="Player ID",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("Projection snapshot matched no realised players")

    pred_pts = pd.to_numeric(merged["xPts Model"], errors="coerce")
    actual_pts = pd.to_numeric(merged["total_points"], errors="coerce")
    pred_mins = pd.to_numeric(merged["Effective Mins"], errors="coerce")
    actual_mins = pd.to_numeric(merged["minutes"], errors="coerce")

    valid_pts = pred_pts.notna() & actual_pts.notna()
    valid_mins = pred_mins.notna() & actual_mins.notna()
    err = pred_pts[valid_pts] - actual_pts[valid_pts]

    metrics = {
        "Players": int(valid_pts.sum()),
        "xPts MAE": float(err.abs().mean()),
        "xPts RMSE": float(np.sqrt(np.mean(err ** 2))),
        "xPts Bias": float(err.mean()),
        "xPts Spearman": float(
            pred_pts[valid_pts].corr(actual_pts[valid_pts], method="spearman")
        ),
        "Minutes MAE": float(
            (pred_mins[valid_mins] - actual_mins[valid_mins]).abs().mean()
        ),
    }

    if "Start Prob" in merged.columns:
        p_start = pd.to_numeric(merged["Start Prob"], errors="coerce")
        y_start = pd.to_numeric(
            merged["starts"], errors="coerce"
        ).fillna(0).gt(0).astype(float)
        mask = p_start.notna()
        if mask.any():
            metrics["Start Brier"] = float(
                np.mean((p_start[mask].clip(0, 1) - y_start[mask]) ** 2)
            )

    if "P60" in merged.columns:
        p60 = pd.to_numeric(merged["P60"], errors="coerce")
        y60 = pd.to_numeric(
            merged["minutes"], errors="coerce"
        ).fillna(0).ge(60).astype(float)
        mask = p60.notna()
        if mask.any():
            metrics["P60 Brier"] = float(
                np.mean((p60[mask].clip(0, 1) - y60[mask]) ** 2)
            )

    return metrics


def evaluate_saved_predictions(
    prediction_dir: Path,
    season_events,
) -> pd.DataFrame:
    """Evaluate saved GW snapshots for which realised data exists."""
    prediction_dir = Path(prediction_dir)
    if not prediction_dir.exists():
        return pd.DataFrame()

    event_map = {
        int(gw): event
        for gw, event in season_events
        if event is not None and not event.empty
    }

    rows = []
    for path in sorted(prediction_dir.glob("gw*.csv")):
        match = SNAPSHOT_RE.match(path.name)
        if not match:
            continue

        gw = int(match.group(1))
        event = event_map.get(gw)
        if event is None:
            continue

        snapshot = pd.read_csv(path)
        metrics = evaluate_projection_snapshot(snapshot, event)
        rows.append({"GW": gw, **metrics})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("GW").reset_index(drop=True)
    metric_cols = [c for c in out.columns if c not in {"GW", "Players"}]

    weighted = {"GW": "ALL", "Players": int(out["Players"].sum())}
    weights = out["Players"].to_numpy(dtype=float)

    for col in metric_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        mask = values.notna() & np.isfinite(values)

        if not mask.any():
            weighted[col] = np.nan
            continue

        if col == "xPts RMSE":
            # Per-GW RMSE squared is MSE. Weight MSE by player count, then
            # take the square root to recover the exact pooled RMSE.
            weighted[col] = float(
                np.sqrt(
                    np.average(
                        values[mask] ** 2,
                        weights=weights[mask],
                    )
                )
            )
        elif col == "xPts Spearman":
            # Rank correlation cannot be pooled from per-GW correlations
            # without the underlying paired observations.
            weighted[col] = np.nan
        else:
            weighted[col] = float(
                np.average(values[mask], weights=weights[mask])
            )

    return pd.concat(
        [out, pd.DataFrame([weighted])],
        ignore_index=True,
    )
