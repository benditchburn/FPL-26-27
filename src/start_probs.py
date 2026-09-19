import numpy as np
import pandas as pd

from scipy.optimize import brentq


SOURCE_WEIGHTS = {
    "FFScout": 1.0,
    "RotoWire": 1.0,
    "NMA": 1.0,
}

LINEUP_SHARPNESS = 6.0


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _availability_prob(row):

    status = str(
        row.get("Status", "")
    ).lower()

    chance = pd.to_numeric(
        row.get("Chance Play Next"),
        errors="coerce",
    )

    # Explicit FPL probability takes precedence
    if pd.notna(chance):
        return float(
            np.clip(chance / 100, 0, 1)
        )

    if status in ["a", "available"]:
        return 1.0

    if status in [
        "i",
        "injured",
        "s",
        "suspended",
        "u",
        "unavailable",
    ]:
        return 0.0

    if status in ["d", "doubtful"]:
        return 0.75

    return 1.0


def _team_start_probs(group):

    g = group.copy()

    active_sources = [,        source,        for source in SOURCE_WEIGHTS,        if source in g.columns and g[source].notna().any(),    ],,    if active_sources:,        total_weight = sum(SOURCE_WEIGHTS[s] for s in active_sources),        weighted_votes = np.zeros(len(g), dtype=float),,        for source in active_sources:,            weight = SOURCE_WEIGHTS[source],            weighted_votes += (,                weight * g[source].fillna(0).to_numpy(dtype=float),            ),,        evidence = weighted_votes / total_weight,    else:,        # No current predicted-lineup feed: use a neutral lineup signal and,        # let availability + the 11-starter reconciliation determine the team.,        evidence = np.full(len(g), 0.5, dtype=float)

    g["Lineup Evidence"] = evidence

    x = LINEUP_SHARPNESS * (
        evidence - 0.5
    )

    availability = (
        g["Availability Prob"]
        .to_numpy(dtype=float)
    )

    # Sanity check: team must have at least
    # 11 players capable of starting
    if availability.sum() < 11:
        raise ValueError(
            f"{g['Team'].iloc[0]} has total "
            f"availability {availability.sum():.2f} < 11"
        )

    # IMPORTANT:
    # solve so UNCONDITIONAL start probabilities
    # sum to exactly 11
    def constraint(alpha):

        conditional = _sigmoid(
            alpha + x
        )

        unconditional = (
            availability * conditional
        )

        return (
            unconditional.sum() - 11
        )

    alpha = brentq(
        constraint,
        -30,
        30,
    )

    g["Conditional Start Prob"] = (
        _sigmoid(alpha + x)
    )

    # Keep this name for compatibility
    # with our existing diagnostics
    g["Source Start Prob"] = (
        g["Conditional Start Prob"]
    )

    g["Start Prob"] = (
        g["Availability Prob"]
        * g["Conditional Start Prob"]
    )

    return g


def build_start_probs(
    consensus: pd.DataFrame,
) -> pd.DataFrame:

    df = consensus.copy()

    # -------------------------
    # Availability first
    # -------------------------

    df["Availability Prob"] = (
        df.apply(
            _availability_prob,
            axis=1,
        )
    )

    # -------------------------
    # Team lineup probabilities
    # -------------------------

    parts = []

    for team, group in df.groupby(
        "Team",
        sort=False,
    ):

        g = _team_start_probs(
            group.copy()
        )

        g["Team"] = team

        parts.append(g)

    return pd.concat(
        parts,
        ignore_index=True,
    )