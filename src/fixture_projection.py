import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "Intercept",
    "Home",
    "Log Attack",
    "Log Opp Def Weakness",
    "Promoted Team",
    "Promoted Opponent",
]


def _prepare_strengths(
    team_hist: pd.DataFrame,
):
    s = team_hist.copy()

    s["Promoted"] = (
        s["League"] != "Premier League"
    ).astype(float)

    # Centre historical strengths around the returning PL clubs.
    # The intercept fitted to GW1 market data then handles the
    # overall 26/27 scoring environment.
    pl = s[
        s["League"] == "Premier League"
    ].copy()

    pl_mu = 0.5 * (
        pl["xG / Match"].mean()
        + pl["xGA / Match"].mean()
    )

    s["Log Attack"] = np.log(
        s["xG / Match"] / pl_mu
    )

    s["Log Def Weakness"] = np.log(
        s["xGA / Match"] / pl_mu
    )

    return s, pl_mu


def _market_team_rows(
    market_odds: pd.DataFrame,
):
    m = market_odds.copy()

    m["GW"] = pd.to_numeric(
        m["GW"],
        errors="coerce",
    )

    m["Fixture Seq"] = pd.to_numeric(
        m["Fixture Seq"],
        errors="coerce",
    )

    home = pd.DataFrame({
    "GW": m["GW"],
    "Fixture Seq": m["Fixture Seq"],
    "Team": m["Home"],
    "Opponent": m["Away"],
    "H/A": "H",
    "Market Team xG": pd.to_numeric(
        m["Market Home xG"],
        errors="coerce",
    ),
    "Market CS Prob": pd.to_numeric(
        m["Home CS Prob"],
        errors="coerce",
    ),
})

    away = pd.DataFrame({
        "GW": m["GW"],
        "Fixture Seq": m["Fixture Seq"],
        "Team": m["Away"],
        "Opponent": m["Home"],
        "H/A": "A",
        "Market Team xG": pd.to_numeric(
            m["Market Away xG"],
            errors="coerce",
        ),
        "Market CS Prob": pd.to_numeric(
            m["Away CS Prob"],
            errors="coerce",
        ),
    })

    return pd.concat(
        [home, away],
        ignore_index=True,
    )


def _add_strength_features(
    df: pd.DataFrame,
    strengths: pd.DataFrame,
):
    out = df.copy()

    team_strength = strengths[
        [
            "Team",
            "Log Attack",
            "Promoted",
        ]
    ].rename(
        columns={
            "Log Attack": "Log Attack",
            "Promoted": "Promoted Team",
        }
    )

    opp_strength = strengths[
        [
            "Team",
            "Log Def Weakness",
            "Promoted",
        ]
    ].rename(
        columns={
            "Team": "Opponent",
            "Log Def Weakness":
                "Log Opp Def Weakness",
            "Promoted":
                "Promoted Opponent",
        }
    )

    out = out.merge(
        team_strength,
        on="Team",
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        opp_strength,
        on="Opponent",
        how="left",
        validate="many_to_one",
    )

    out["Home"] = (
        out["H/A"] == "H"
    ).astype(float)

    missing = out[
        out[
            [
                "Log Attack",
                "Log Opp Def Weakness",
            ]
        ].isna().any(axis=1)
    ]

    if len(missing):
        raise ValueError(
            "Missing strength data for fixtures:\n"
            + missing[
                ["Team", "Opponent"]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    return out


def _design_matrix(
    df: pd.DataFrame,
):
    return np.column_stack([
        np.ones(len(df)),
        df["Home"].to_numpy(dtype=float),
        df["Log Attack"].to_numpy(dtype=float),
        df["Log Opp Def Weakness"]
            .to_numpy(dtype=float),
        df["Promoted Team"]
            .to_numpy(dtype=float),
        df["Promoted Opponent"]
            .to_numpy(dtype=float),
    ])


def fit_fixture_model(
    team_hist: pd.DataFrame,
    market_odds: pd.DataFrame,
    ridge_lambda: float = 2.0,
):
    """
    Calibrate historical team attack/defence strength
    against current bookmaker-implied xG.

    Attack and defence coefficients are shrunk toward 1.
    Home/promotion adjustments are shrunk toward 0.
    Intercept is unpenalised.
    """

    strengths, pl_mu = _prepare_strengths(
        team_hist
    )

    market = _market_team_rows(
        market_odds
    )

    market = market[
        market["Market Team xG"].notna()
    ].copy()

    train = _add_strength_features(
        market,
        strengths,
    )

    X = _design_matrix(train)

    y = np.log(
        train["Market Team xG"]
        .to_numpy(dtype=float)
        / pl_mu
    )

    # Prior beliefs:
    # historical attack coefficient ≈ 1
    # historical defensive weakness ≈ 1
    # everything else centred at zero.
    prior = np.array([
        0.0,  # intercept
        0.0,  # home
        1.0,  # attack
        1.0,  # opponent defence
        0.0,  # promoted attack adjustment
        0.0,  # promoted opponent adjustment
    ])

    penalty_weights = np.array([
        0.0,  # never penalise intercept
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    P = np.diag(
        ridge_lambda * penalty_weights
    )

    beta = np.linalg.solve(
        X.T @ X + P,
        X.T @ y + P @ prior,
    )

    fitted = (
        pl_mu
        * np.exp(X @ beta)
    )

    fit_table = train[
        [
            "GW",
            "Fixture Seq",
            "Team",
            "Opponent",
            "H/A",
            "Market Team xG",
        ]
    ].copy()

    fit_table["Model xG"] = fitted

    fit_table["Error"] = (
        fit_table["Model xG"]
        - fit_table["Market Team xG"]
    )

    fit_table["Abs Error"] = (
        fit_table["Error"].abs()
    )

    rmse = np.sqrt(
        np.mean(
            fit_table["Error"] ** 2
        )
    )

    mae = (
        fit_table["Abs Error"].mean()
    )

    coefficients = dict(
        zip(
            FEATURE_NAMES,
            beta,
        )
    )

    return {
        "coefficients": coefficients,
        "beta": beta,
        "pl_mu": pl_mu,
        "ridge_lambda": ridge_lambda,
        "rmse": rmse,
        "mae": mae,
        "fit_table": fit_table,
        "strengths": strengths,
    }


def project_team_fixtures(
    fixtures: pd.DataFrame,
    team_hist: pd.DataFrame,
    market_odds: pd.DataFrame,
    start_gw: int = 1,
    max_gw: int = 6,
    ridge_lambda: float = 2.0,
):
    if start_gw < 1:
        raise ValueError("start_gw must be >= 1")
    if max_gw < start_gw:
        raise ValueError("max_gw must be >= start_gw")

    fit = fit_fixture_model(
        team_hist,
        market_odds,
        ridge_lambda=ridge_lambda,
    )

    strengths = fit["strengths"]
    beta = fit["beta"]
    pl_mu = fit["pl_mu"]

    f = fixtures[
        [
            "GW",
            "Fixture ID",
            "Kickoff UTC",
            "Team",
            "Opponent",
            "H/A",
            "Official FDR",
        ]
    ].copy()

    f["GW"] = pd.to_numeric(
        f["GW"],
        errors="coerce",
    )

    f["Fixture ID"] = pd.to_numeric(
        f["Fixture ID"],
        errors="coerce",
    )

    f = f[
        f["GW"].between(
            start_gw,
            max_gw,
        )
    ].copy()

    f = _add_strength_features(
        f,
        strengths,
    )

    X = _design_matrix(f)

    f["Strength Model xG"] = (
        pl_mu
        * np.exp(X @ beta)
    )

    # -------------------------
    # Market overrides
    # -------------------------

    market = _market_team_rows(
        market_odds
    )

    f = f.merge(
        market[
            [
                "GW",
                "Team",
                "Opponent",
                "Market Team xG",
                "Market CS Prob",
            ]
        ],
        on=[
            "GW",
            "Team",
            "Opponent",
        ],
        how="left",
        validate="one_to_one",
    )

    f["Team xG"] = (
        f["Market Team xG"]
        .combine_first(
            f["Strength Model xG"]
        )
    )

    f["xG Source"] = np.where(
        f["Market Team xG"].notna(),
        "Market",
        "Strength model",
    )

    # -------------------------
    # Opponent xG
    # -------------------------

    opp_lookup = f[
        [
            "GW",
            "Fixture ID",
            "Team",
            "Team xG",
        ]
    ].rename(
        columns={
            "Team": "Opponent",
            "Team xG": "Opp xG",
        }
    )

    f = f.merge(
        opp_lookup,
        on=[
            "GW",
            "Fixture ID",
            "Opponent",
        ],
        how="left",
        validate="one_to_one",
    )

    # Exact bookmaker CS probability where available.
    # Otherwise Poisson from projected opponent xG.
    f["CS Prob"] = (
        f["Market CS Prob"]
        .combine_first(
            np.exp(
                -f["Opp xG"]
            )
        )
    )

    return (
        f.sort_values(
            ["GW", "Fixture ID", "H/A"]
        ).reset_index(drop=True),
        fit,
    )