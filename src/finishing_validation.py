import numpy as np
import pandas as pd


def _poisson_deviance(y, mu):
    """
    Poisson deviance per observation.
    Lower is better.
    """
    y = np.asarray(y, dtype=float)
    mu = np.clip(
        np.asarray(mu, dtype=float),
        1e-9,
        None,
    )

    term = np.zeros_like(y)

    positive = y > 0

    term[positive] = (
        y[positive]
        * np.log(
            y[positive] / mu[positive]
        )
    )

    return 2 * (
        term - (y - mu)
    )


def build_clean_finishing_history(
    hist_minutes: pd.DataFrame,
    attack_priors: pd.DataFrame,
):
    h = hist_minutes.copy()

    for col in [
        "GW",
        "minutes",
        "goals_scored",
        "expected_goals",
    ]:
        h[col] = pd.to_numeric(
            h[col],
            errors="coerce",
        )

    penalty_flags = (
        attack_priors[
            [
                "Player ID",
                "Penalty Contamination?",
            ]
        ]
        .drop_duplicates("Player ID")
        .copy()
    )

    penalty_flags[
        "Penalty Contamination?"
    ] = pd.to_numeric(
        penalty_flags[
            "Penalty Contamination?"
        ],
        errors="coerce",
    )

    h = h.merge(
        penalty_flags,
        left_on="player_id",
        right_on="Player ID",
        how="left",
        validate="many_to_one",
    )

    # Only use players explicitly identified
    # as having low penalty contamination.
    h = h[
        (h["Penalty Contamination?"] == 0)
        & h["FPL Pos"].isin(
            ["DEF", "MID", "FWD"]
        )
    ].copy()

    h["goals_scored"] = (
        h["goals_scored"].fillna(0)
    )

    h["expected_goals"] = (
        h["expected_goals"].fillna(0)
    )

    return h


def validate_finishing_fold(
    hist: pd.DataFrame,
    train_end_gw: int,
    test_end_gw: int,
    shrink_k: float,
):
    train = hist[
        hist["GW"] <= train_end_gw
    ].copy()

    test = hist[
        (hist["GW"] > train_end_gw)
        & (hist["GW"] <= test_end_gw)
    ].copy()

    train_player = (
        train.groupby("player_id")
        .agg(
            Train_Goals=(
                "goals_scored",
                "sum",
            ),
            Train_xG=(
                "expected_goals",
                "sum",
            ),
        )
        .reset_index()
    )

    # Gamma-Poisson style shrinkage toward
    # league-average finishing = 1.0
    train_player["Finishing Skill"] = (
        (
            train_player["Train_Goals"]
            + shrink_k
        )
        /
        (
            train_player["Train_xG"]
            + shrink_k
        )
    )

    test_player = (
        test.groupby(
            [
                "player_id",
                "FPL Pos",
            ]
        )
        .agg(
            Test_Goals=(
                "goals_scored",
                "sum",
            ),
            Test_xG=(
                "expected_goals",
                "sum",
            ),
        )
        .reset_index()
    )

    out = test_player.merge(
        train_player,
        on="player_id",
        how="left",
    )

    # No prior sample = average finisher.
    out["Finishing Skill"] = (
        out["Finishing Skill"]
        .fillna(1.0)
    )

    out["Baseline Lambda"] = (
        out["Test_xG"]
    )

    out["Finishing Lambda"] = (
        out["Test_xG"]
        * out["Finishing Skill"]
    )

    out["Baseline Deviance"] = (
        _poisson_deviance(
            out["Test_Goals"],
            out["Baseline Lambda"],
        )
    )

    out["Finishing Deviance"] = (
        _poisson_deviance(
            out["Test_Goals"],
            out["Finishing Lambda"],
        )
    )

    return out


def validate_finishing(
    hist_minutes: pd.DataFrame,
    attack_priors: pd.DataFrame,
    shrink_k: float,
):
    hist = build_clean_finishing_history(
        hist_minutes,
        attack_priors,
    )

    folds = [
        (12, 18),
        (18, 25),
        (25, 38),
    ]

    pieces = []

    for train_end, test_end in folds:

        fold = validate_finishing_fold(
            hist,
            train_end_gw=train_end,
            test_end_gw=test_end,
            shrink_k=shrink_k,
        )

        fold["Train Through"] = train_end
        fold["Test GWs"] = (
            f"{train_end + 1}-{test_end}"
        )

        pieces.append(fold)

    oof = pd.concat(
        pieces,
        ignore_index=True,
    )

    return {
        "oof": oof,
        "baseline_deviance":
            oof["Baseline Deviance"].mean(),
        "finishing_deviance":
            oof["Finishing Deviance"].mean(),
    }