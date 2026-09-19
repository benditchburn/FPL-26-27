from __future__ import annotations

import numpy as np
import pandas as pd


def build_current_action_report(
    transfer_result: dict,
    xpts_horizon: pd.DataFrame,
    current_players: pd.DataFrame,
    start_gw: int,
    max_gw: int,
    gw_decay: float = 0.90,
) -> pd.DataFrame:
    """Explain the players involved in the optimiser's first-GW action.

    The report is deliberately descriptive rather than pretending the transfer
    pair itself explains the full optimiser gain: formation, captaincy, bench
    value and future transfer-state effects can also matter.
    """

    first = transfer_result["optimal"]["plan"].iloc[0]

    out_ids = [int(x) for x in first.get("Out IDs", [])]
    in_ids = [int(x) for x in first.get("In IDs", [])]

    moved = [("OUT", pid) for pid in out_ids] + [
        ("IN", pid) for pid in in_ids
    ]

    if not moved:
        return pd.DataFrame(
            columns=[
                "Action",
                "Player",
                "Team",
                "FPL Pos",
                "Current £m",
                "Current GW xPts",
                "Current GW Mins",
                "Current GW xG",
                "Current GW xA",
                "Weighted Horizon xPts",
            ]
        )

    meta = current_players[
        ["Player ID", "Player", "Team", "FPL Pos", "Current £m"]
    ].drop_duplicates("Player ID").set_index("Player ID")

    horizon = xpts_horizon[
        xpts_horizon["GW"].between(start_gw, max_gw)
    ].copy()

    rows = []

    for action, pid in moved:
        if pid not in meta.index:
            continue

        p = horizon[horizon["Player ID"] == pid].copy()
        if p.empty:
            continue

        p["GW"] = pd.to_numeric(p["GW"], errors="coerce").astype(int)
        p["Weight"] = p["GW"].map(
            {
                gw: gw_decay ** (gw - start_gw)
                for gw in range(start_gw, max_gw + 1)
            }
        )

        first_gw = p[p["GW"] == start_gw]
        fg = first_gw.iloc[0] if not first_gw.empty else None

        row = {
            "Action": action,
            "Player ID": pid,
            "Player": meta.loc[pid, "Player"],
            "Team": meta.loc[pid, "Team"],
            "FPL Pos": meta.loc[pid, "FPL Pos"],
            "Current £m": float(meta.loc[pid, "Current £m"]),
            "Current GW xPts": (
                float(fg["xPts Model"]) if fg is not None else np.nan
            ),
            "Current GW Mins": (
                float(fg["Effective Mins"]) if fg is not None else np.nan
            ),
            "Current GW xG": (
                float(fg["xG"]) if fg is not None else np.nan
            ),
            "Current GW xA": (
                float(fg["xA"]) if fg is not None else np.nan
            ),
            "Weighted Horizon xPts": float(
                (p["xPts Model"] * p["Weight"]).sum()
            ),
        }

        for gw in range(start_gw, max_gw + 1):
            hit = p.loc[p["GW"] == gw, "xPts Model"]
            row[f"GW{gw} xPts"] = (
                float(hit.iloc[0]) if len(hit) else np.nan
            )

        # Include the current-GW component breakdown when available.
        if fg is not None:
            for col in [
                "Base xPts",
                "xPts DefCon",
                "xPts Saves",
                "xPts Bonus",
                "Team CS Prob",
            ]:
                if col in fg.index:
                    row[col] = float(fg[col])

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        ["Action", "FPL Pos", "Weighted Horizon xPts"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def build_decision_note(transfer_result: dict) -> str:
    """Return a compact human-readable summary of how decisive the model is."""

    margin = transfer_result.get("decision_margin_vs_runner_up", np.nan)
    label = transfer_result.get("decision_margin_label", "unknown")

    best = transfer_result["comparison"].iloc[0]
    recommendation = transfer_result.get("recommendation", "")

    if pd.isna(margin):
        margin_text = "n/a"
    else:
        margin_text = f"{float(margin):.2f}"

    return (
        f"{recommendation} | {label}; "
        f"edge vs next-best current-GW action = {margin_text} utility; "
        f"raw xPts vs roll = {float(best['Raw xPts vs Roll']):+.2f}"
    )
