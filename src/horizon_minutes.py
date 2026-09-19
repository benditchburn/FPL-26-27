import numpy as np
import pandas as pd
from scipy.optimize import brentq


# How much today's live lineup information matters
# as we move further into the future.
LIVE_WEIGHTS = {
    1: 1.00,
    2: 0.70,
    3: 0.50,
    4: 0.35,
    5: 0.25,
    6: 0.20,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _reconcile_team_starts(group):
    """
    Shift conditional start probabilities so
    unconditional start probabilities sum to 11.
    """

    g = group.copy()

    availability = (
        g["Availability Prob"]
        .to_numpy(dtype=float)
    )

    target = (
        g["Target Conditional Start Prob"]
        .clip(1e-6, 1 - 1e-6)
        .to_numpy(dtype=float)
    )

    logits = _logit(target)

    if availability.sum() < 11:
        raise ValueError(
            f"{g['Team'].iloc[0]} only has "
            f"{availability.sum():.2f} total availability"
        )

    def constraint(alpha):
        conditional = _sigmoid(
            logits + alpha
        )

        return (
            np.sum(
                availability * conditional
            )
            - 11
        )

    alpha = brentq(
        constraint,
        -30,
        30,
    )

    g["Conditional Start Prob"] = (
        _sigmoid(logits + alpha)
    )

    g["Start Prob"] = (
        g["Availability Prob"]
        * g["Conditional Start Prob"]
    )

    return g


def build_minutes_horizon(
    minutes: pd.DataFrame,
    appearance_priors: pd.DataFrame,
    start_gw: int = 1,
    max_gw: int = 6,
):
    """
    Build expected minutes from ``start_gw`` through ``max_gw``.

    Horizon step 1 uses all of the current live-lineup information.
    Later steps progressively regress toward the player's historical
    underlying role and toward full availability.

    ``start_gw`` is deliberately separate from the horizon step so the
    same function works in GW2, GW3, etc. without treating the current
    week as if it were a distant forecast.
    """

    if start_gw < 1:
        raise ValueError("start_gw must be >= 1")
    if max_gw < start_gw:
        raise ValueError("max_gw must be >= start_gw")

    base_cols = [
        "Player ID",
        "Code",
        "Player",
        "Team",
        "FPL Pos",
        "Status",
        "Chance Play Next",
        "News",
        "Availability Prob",
        "Conditional Start Prob",
        "Start Prob",
        "Mins If Start",
        "Expected Mins If Not Start",
    ]

    base = minutes[
        base_cols
    ].copy()

    priors = appearance_priors[
        [
            "Player ID",
            "Starts",
            "NonStarts",
            "P60 If Start",
            "P(App | Not Start)",
            "P60 If Not Start",
        ]
    ].copy()

    base = base.merge(
        priors,
        on="Player ID",
        how="left",
        validate="one_to_one",
    )

    denom = (
        base["Starts"]
        + base["NonStarts"]
    )

    base["Historical Start Prob"] = np.where(
        denom > 0,
        base["Starts"] / denom,
        np.nan,
    )

    # No historical role data:
    # retain today's conditional role estimate.
    base["Historical Start Prob"] = (
        base["Historical Start Prob"]
        .fillna(
            base["Conditional Start Prob"]
        )
    )

    # Missing conditional appearance priors
    base["P60 If Start"] = (
        base["P60 If Start"]
        .fillna(0.90)
    )

    base["P(App | Not Start)"] = (
        base["P(App | Not Start)"]
        .fillna(0.30)
    )

    base["P60 If Not Start"] = (
        base["P60 If Not Start"]
        .fillna(0.01)
    )

    pieces = []

    gameweeks = list(range(start_gw, max_gw + 1))

    for horizon_step, gw in enumerate(gameweeks, start=1):

        w = LIVE_WEIGHTS.get(
            horizon_step,
            LIVE_WEIGHTS[max(LIVE_WEIGHTS)],
        )

        g = base.copy()

        g["GW"] = gw
        g["Horizon Step"] = horizon_step
        g["Live Weight"] = w

        # -------------------------
        # Availability horizon
        # -------------------------

        g["Availability Prob"] = (
            w
            * base["Availability Prob"]
            + (1 - w)
            * 1.0
        )

        # -------------------------
        # Underlying role horizon
        # -------------------------

        g[
            "Target Conditional Start Prob"
        ] = (
            w
            * base["Conditional Start Prob"]
            + (1 - w)
            * base["Historical Start Prob"]
        )

        # Enforce exactly 11 expected starters
        # for every club.
        reconciled = []

        for _, team_group in g.groupby(
            "Team",
            sort=False,
        ):
            reconciled.append(
                _reconcile_team_starts(
                    team_group
                )
            )

        g = pd.concat(
            reconciled,
            ignore_index=True,
        )

        # -------------------------
        # Expected minutes
        # -------------------------

        g["Effective Mins"] = (
            g["Availability Prob"]
            * (
                g["Conditional Start Prob"]
                * g["Mins If Start"]
                +
                (
                    1
                    - g["Conditional Start Prob"]
                )
                * g[
                    "Expected Mins If Not Start"
                ]
            )
        )

        g["Model Expected Mins"] = (
            g["Effective Mins"]
        )

        # -------------------------
        # Appearance probabilities
        # -------------------------

        g["Appearance Prob"] = (
            g["Availability Prob"]
            * (
                g["Conditional Start Prob"]
                +
                (
                    1
                    - g["Conditional Start Prob"]
                )
                * g[
                    "P(App | Not Start)"
                ]
            )
        )

        g["P60"] = (
            g["Availability Prob"]
            * (
                g["Conditional Start Prob"]
                * g["P60 If Start"]
                +
                (
                    1
                    - g["Conditional Start Prob"]
                )
                * g["P60 If Not Start"]
            )
        )

        # FPL appearance scoring:
        # 1 point for appearing
        # + another point for >=60 mins
        g["Expected Appearance Pts"] = (
            g["Appearance Prob"]
            + g["P60"]
        )

        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )