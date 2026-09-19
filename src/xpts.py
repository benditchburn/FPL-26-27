import numpy as np
import pandas as pd


GOAL_POINTS = {
    "GK": 10,
    "DEF": 6,
    "MID": 5,
    "FWD": 4,
}

CS_POINTS = {
    "GK": 4,
    "DEF": 4,
    "MID": 1,
    "FWD": 0,
}


def expected_conceded_deduction(lam, max_goals=12):
    """
    E[floor(goals_conceded / 2)] for a Poisson process.
    Used for GK/DEF FPL deductions.
    """
    if pd.isna(lam) or lam <= 0:
        return 0.0

    probs = []
    p = np.exp(-lam)
    probs.append(p)

    for k in range(1, max_goals + 1):
        p = p * lam / k
        probs.append(p)

    return sum(
        (k // 2) * probs[k]
        for k in range(len(probs))
    )


def build_xpts(
    attack: pd.DataFrame,
    appearance: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generic FPL xPts engine.

    Works for:
    - single-GW projections
    - GW1-6 horizon projections

    Expected input on attack:
        Player ID
        FPL Pos
        Effective Mins
        xG
        xA
        Opp xG

    Optional:
        CS Prob
        GW
        Appearance Prob
        P60
        Expected Appearance Pts

    If appearance fields are not already in attack,
    they are merged from the appearance dataframe.
    """

    df = attack.copy()

    # =========================================================
    # 1. STANDARDISE ATTACK COLUMN NAMES
    # =========================================================

    # Allows old GW1 pipeline to keep working.
    if "xG" not in df.columns:
        if "GW1 xG" in df.columns:
            df["xG"] = df["GW1 xG"]
        else:
            raise ValueError(
                "No xG column found."
            )

    if "xA" not in df.columns:
        if "GW1 xA" in df.columns:
            df["xA"] = df["GW1 xA"]
        else:
            raise ValueError(
                "No xA column found."
            )

    if "Opp xG" not in df.columns:
        if "Opponent xG" in df.columns:
            df["Opp xG"] = (
                df["Opponent xG"]
            )
        else:
            raise ValueError(
                "No opponent xG column found."
            )

    # =========================================================
    # 2. APPEARANCE INFORMATION
    # =========================================================

    appearance_cols = [
        "Appearance Prob",
        "P60",
        "Expected Appearance Pts",
    ]

    # Horizon attack already contains these because
    # it was constructed using minutes_horizon.
    has_appearance = all(
        col in df.columns
        for col in appearance_cols
    )

    if not has_appearance:

        if appearance is None:
            raise ValueError(
                "Appearance information is missing "
                "and no appearance dataframe was supplied."
            )

        merge_keys = [
            "Player ID",
        ]

        # Horizon appearance has one observation
        # per Player ID x GW.
        if (
            "GW" in df.columns
            and "GW" in appearance.columns
        ):
            merge_keys.append("GW")

        app = appearance[
            merge_keys
            + appearance_cols
        ].copy()

        df = df.merge(
            app,
            on=merge_keys,
            how="left",
            validate="many_to_one",
        )

    # Check the appearance merge actually worked.
    missing_app = df[
        appearance_cols
    ].isna().any(axis=1)

    if missing_app.any():

        cols = [
            c
            for c in [
                "Player ID",
                "Player",
                "GW",
            ]
            if c in df.columns
        ]

        raise ValueError(
            "Missing appearance information for:\n"
            + df.loc[
                missing_app,
                cols,
            ]
            .head(20)
            .to_string(index=False)
        )

    # =========================================================
    # 3. APPEARANCE POINTS
    # =========================================================

    df["xPts Appearance"] = (
        df["Expected Appearance Pts"]
    )

    # =========================================================
    # 4. GOALS
    # =========================================================

    goal_values = (
        df["FPL Pos"]
        .map(GOAL_POINTS)
    )

    if goal_values.isna().any():

        bad_positions = (
            df.loc[
                goal_values.isna(),
                "FPL Pos",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unknown FPL positions in GOAL_POINTS: "
            f"{bad_positions}"
        )

    df["xPts Goals"] = (
        df["xG"]
        * goal_values
    )

    # =========================================================
    # 5. ASSISTS
    # =========================================================

    df["xPts Assists"] = (
        df["xA"] * 3
    )

    # =========================================================
    # 6. CLEAN SHEETS
    # =========================================================

    # Use bookmaker / fixture-engine CS probability
    # whenever available.
    #
    # Otherwise:
    # P(opponent scores zero) = exp(-opponent xG)
    poisson_cs = np.exp(
        -df["Opp xG"]
    )

    if "CS Prob" in df.columns:

        supplied_cs = pd.to_numeric(
            df["CS Prob"],
            errors="coerce",
        )

        df["Team CS Prob"] = (
            supplied_cs.fillna(
                poisson_cs
            )
        )

    else:

        df["Team CS Prob"] = (
            poisson_cs
        )

    cs_values = (
        df["FPL Pos"]
        .map(CS_POINTS)
        .fillna(0.0)
    )

    # Player needs to reach 60 minutes to earn
    # clean-sheet points.
    df["xPts Clean Sheet"] = (
        df["P60"]
        * df["Team CS Prob"]
        * cs_values
    )

    # =========================================================
    # 7. GOALS-CONCEDED DEDUCTION
    # =========================================================

    # Scale opponent scoring exposure by the amount
    # of time we expect the player to be on the pitch.
    df["Player Opp xG Exposure"] = (
        df["Opp xG"]
        * df["Effective Mins"]
        / 90
    )

    df["xPts Goals Conceded"] = 0.0

    defensive_mask = (
        df["FPL Pos"]
        .isin(
            [
                "GK",
                "DEF",
            ]
        )
    )

    df.loc[
        defensive_mask,
        "xPts Goals Conceded",
    ] = -(
        df.loc[
            defensive_mask,
            "Player Opp xG Exposure",
        ]
        .map(
            expected_conceded_deduction
        )
    )

    # =========================================================
    # 8. BASE EXPECTED POINTS
    # =========================================================

    df["Base xPts"] = (
        df["xPts Appearance"]
        + df["xPts Goals"]
        + df["xPts Assists"]
        + df["xPts Clean Sheet"]
        + df["xPts Goals Conceded"]
    )

    return df


def build_gw1_xpts(
    attack: pd.DataFrame,
    appearance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper for the old GW1 pipeline.
    """

    out = build_xpts(
        attack=attack,
        appearance=appearance,
    )

    # Preserve the name expected elsewhere
    # in the existing GW1 notebook.
    out["GW1 Base xPts"] = (
        out["Base xPts"]
    )

    return out
