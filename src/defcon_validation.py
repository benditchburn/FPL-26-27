import numpy as np
import pandas as pd


def _threshold(pos):
    if pos == "DEF":
        return 10
    if pos in ["MID", "FWD"]:
        return 12
    return np.nan


def validate_defcon(
    hist: pd.DataFrame,
    train_end_gw=25,
    shrink_k=8,
):
    h = hist.copy()

    for col in [
        "GW",
        "minutes",
        "starts",
        "defensive_contribution",
    ]:
        h[col] = pd.to_numeric(
            h[col],
            errors="coerce",
        )

    chance = pd.to_numeric(
        h["chance_of_playing_this_round"],
        errors="coerce",
    )

    h["Available"] = (
        (h["status"] == "a")
        | (chance > 0)
    )

    h = h[
        h["Available"]
        & h["FPL Pos"].isin(["DEF", "MID", "FWD"])
    ].copy()

    h["Started"] = h["starts"].fillna(0) > 0

    h["Threshold"] = h["FPL Pos"].map(_threshold)

    h["DC Hit"] = (
        h["defensive_contribution"].fillna(0)
        >= h["Threshold"]
    ).astype(float)

    train = h[
        (h["GW"] <= train_end_gw)
        & h["Started"]
    ].copy()

    test = h[
        (h["GW"] > train_end_gw)
        & h["Started"]
    ].copy()

    # -------------------------
    # Position priors
    # -------------------------

    pos_prior = (
        train.groupby("FPL Pos")["DC Hit"]
        .mean()
        .to_dict()
    )

    # -------------------------
    # Player history
    # -------------------------

    player_train = (
        train.groupby(
            ["player_code", "FPL Pos"]
        )
        .agg(
            Train_N=("DC Hit", "size"),
            Train_Hits=("DC Hit", "sum"),
        )
        .reset_index()
    )

    player_train["Position Prior"] = (
        player_train["FPL Pos"]
        .map(pos_prior)
    )

    player_train["Predicted P"] = (
        player_train["Train_Hits"]
        + shrink_k * player_train["Position Prior"]
    ) / (
        player_train["Train_N"]
        + shrink_k
    )

    # -------------------------
    # Attach prediction to test
    # -------------------------

    test = test.merge(
        player_train[
            [
                "player_code",
                "Predicted P",
                "Train_N",
            ]
        ],
        on="player_code",
        how="left",
    )

    test["Position Prior"] = (
        test["FPL Pos"].map(pos_prior)
    )

    test["Predicted P"] = (
        test["Predicted P"]
        .fillna(test["Position Prior"])
    )

    # -------------------------
    # Error metrics
    # -------------------------

    test["Squared Error"] = (
        test["Predicted P"]
        - test["DC Hit"]
    ) ** 2

    brier = test["Squared Error"].mean()

    baseline_brier = (
        (
            test["Position Prior"]
            - test["DC Hit"]
        ) ** 2
    ).mean()

    # -------------------------
    # Calibration buckets
    # -------------------------

    test["Probability Bin"] = pd.cut(
        test["Predicted P"],
        bins=[
            0,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.000001,
        ],
        include_lowest=True,
    )

    calibration = (
        test.groupby(
            "Probability Bin",
            observed=True,
        )
        .agg(
            N=("DC Hit", "size"),
            Predicted=("Predicted P", "mean"),
            Actual=("DC Hit", "mean"),
        )
        .reset_index()
    )

    # -------------------------
    # Position performance
    # -------------------------

    by_position = (
        test.groupby("FPL Pos")
        .agg(
            N=("DC Hit", "size"),
            Predicted=("Predicted P", "mean"),
            Actual=("DC Hit", "mean"),
            Brier=("Squared Error", "mean"),
        )
        .reset_index()
    )

    # -------------------------
    # Player-level audit
    # -------------------------

    player_audit = (
        test.groupby(
            ["player_code", "FPL Pos"]
        )
        .agg(
            Test_N=("DC Hit", "size"),
            Predicted=("Predicted P", "mean"),
            Actual=("DC Hit", "mean"),
            Mean_DC=(
                "defensive_contribution",
                "mean",
            ),
        )
        .reset_index()
    )

    return {
        "test": test,
        "calibration": calibration,
        "by_position": by_position,
        "player_audit": player_audit,
        "brier": brier,
        "baseline_brier": baseline_brier,
    }