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


def _prepare_strengths(team_hist: pd.DataFrame):
    s = team_hist.copy()

    s["Promoted"] = (s["League"] != "Premier League").astype(float)

    # Centre strengths around returning PL clubs. Because team_hist itself is
    # updated with current-season xG/xGA, the league scoring level can move
    # gradually during the season rather than being frozen at preseason.
    pl = s[s["League"] == "Premier League"].copy()
    pl_mu = 0.5 * (pl["xG / Match"].mean() + pl["xGA / Match"].mean())

    s["Log Attack"] = np.log(s["xG / Match"] / pl_mu)
    s["Log Def Weakness"] = np.log(s["xGA / Match"] / pl_mu)

    return s, pl_mu


def _market_team_rows(market_odds: pd.DataFrame):
    m = market_odds.copy()

    m["GW"] = pd.to_numeric(m["GW"], errors="coerce")
    if "Fixture Seq" not in m.columns:
        m["Fixture Seq"] = np.nan
    m["Fixture Seq"] = pd.to_numeric(m["Fixture Seq"], errors="coerce")

    if "Market Source" not in m.columns:
        m["Market Source"] = "Workbook market"

    if "Market Fit RMSE" not in m.columns:
        m["Market Fit RMSE"] = np.nan

    home = pd.DataFrame({
        "GW": m["GW"],
        "Fixture Seq": m["Fixture Seq"],
        "Team": m["Home"],
        "Opponent": m["Away"],
        "H/A": "H",
        "Market Team xG": pd.to_numeric(m["Market Home xG"], errors="coerce"),
        "Market CS Prob": pd.to_numeric(m["Home CS Prob"], errors="coerce"),
        "Market Source": m["Market Source"],
        "Market Fit RMSE": pd.to_numeric(m["Market Fit RMSE"], errors="coerce"),
    })

    away = pd.DataFrame({
        "GW": m["GW"],
        "Fixture Seq": m["Fixture Seq"],
        "Team": m["Away"],
        "Opponent": m["Home"],
        "H/A": "A",
        "Market Team xG": pd.to_numeric(m["Market Away xG"], errors="coerce"),
        "Market CS Prob": pd.to_numeric(m["Away CS Prob"], errors="coerce"),
        "Market Source": m["Market Source"],
        "Market Fit RMSE": pd.to_numeric(m["Market Fit RMSE"], errors="coerce"),
    })

    return pd.concat([home, away], ignore_index=True)


def _add_strength_features(df: pd.DataFrame, strengths: pd.DataFrame):
    out = df.copy()

    team_strength = strengths[["Team", "Log Attack", "Promoted"]].rename(
        columns={"Promoted": "Promoted Team"}
    )
    opp_strength = strengths[["Team", "Log Def Weakness", "Promoted"]].rename(
        columns={
            "Team": "Opponent",
            "Log Def Weakness": "Log Opp Def Weakness",
            "Promoted": "Promoted Opponent",
        }
    )

    out = out.merge(team_strength, on="Team", how="left", validate="many_to_one")
    out = out.merge(opp_strength, on="Opponent", how="left", validate="many_to_one")
    out["Home"] = (out["H/A"] == "H").astype(float)

    missing = out[out[["Log Attack", "Log Opp Def Weakness"]].isna().any(axis=1)]
    if len(missing):
        raise ValueError(
            "Missing strength data for fixtures:\n"
            + missing[["Team", "Opponent"]].drop_duplicates().to_string(index=False)
        )

    return out


def _design_matrix(df: pd.DataFrame):
    return np.column_stack([
        np.ones(len(df)),
        df["Home"].to_numpy(dtype=float),
        df["Log Attack"].to_numpy(dtype=float),
        df["Log Opp Def Weakness"].to_numpy(dtype=float),
        df["Promoted Team"].to_numpy(dtype=float),
        df["Promoted Opponent"].to_numpy(dtype=float),
    ])


def _ridge_beta(X, y, prior, penalty):
    return np.linalg.solve(
        X.T @ X + penalty,
        X.T @ y + penalty @ prior,
    )


def _fixture_key(row):
    # Hold both sides of a fixture out together in CV. Fixture Seq is preferred,
    # but current-market override files need not know the workbook sequence.
    if pd.notna(row.get("Fixture Seq")):
        return f"{int(row['GW'])}|seq:{int(row['Fixture Seq'])}"
    teams = sorted([str(row["Team"]), str(row["Opponent"])])
    return f"{int(row['GW'])}|{teams[0]}|{teams[1]}"


def _leave_fixture_out_market_cv(train, pl_mu, prior, penalty):
    """Measure how well the strength features reproduce held-out market xG."""
    if train.empty:
        return pd.Series(dtype=float), np.nan, np.nan

    work = train.copy()
    work["_Fixture Key"] = work.apply(_fixture_key, axis=1)
    keys = work["_Fixture Key"].drop_duplicates().tolist()

    # With only a couple of market fixtures, CV is more noise than signal.
    if len(keys) < 4:
        return pd.Series(np.nan, index=work.index), np.nan, np.nan

    preds = pd.Series(np.nan, index=work.index, dtype=float)

    for key in keys:
        test_mask = work["_Fixture Key"].eq(key)
        train_mask = ~test_mask

        X_train = _design_matrix(work.loc[train_mask])
        y_train = np.log(
            work.loc[train_mask, "Market Team xG"].to_numpy(dtype=float) / pl_mu
        )
        beta = _ridge_beta(X_train, y_train, prior, penalty)

        X_test = _design_matrix(work.loc[test_mask])
        preds.loc[test_mask] = pl_mu * np.exp(X_test @ beta)

    actual = work["Market Team xG"].astype(float)
    valid = preds.notna() & actual.notna()
    if not valid.any():
        return preds, np.nan, np.nan

    err = preds[valid] - actual[valid]
    return (
        preds,
        float(np.sqrt(np.mean(err ** 2))),
        float(np.mean(np.abs(err))),
    )


def fit_fixture_model(
    team_hist: pd.DataFrame,
    market_odds: pd.DataFrame,
    ridge_lambda: float = 2.0,
):
    """Calibrate attack/defence strengths against market-implied team xG.

    The model is regularised toward the intuitive prior of one-for-one attack
    and opponent-defence effects. In addition to the training fit, a
    leave-one-fixture-out score is reported so diagnostics do not confuse
    in-sample fit with genuine generalisation to unseen fixtures.
    """
    strengths, pl_mu = _prepare_strengths(team_hist)
    market = _market_team_rows(market_odds)
    market = market[market["Market Team xG"].notna()].copy()

    if market.empty:
        raise ValueError("No market xG rows available to calibrate fixture model")

    train = _add_strength_features(market, strengths)
    X = _design_matrix(train)
    y = np.log(train["Market Team xG"].to_numpy(dtype=float) / pl_mu)

    prior = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    penalty_weights = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    penalty = np.diag(ridge_lambda * penalty_weights)

    beta = _ridge_beta(X, y, prior, penalty)
    fitted = pl_mu * np.exp(X @ beta)

    cv_pred, cv_rmse, cv_mae = _leave_fixture_out_market_cv(
        train, pl_mu, prior, penalty
    )

    fit_table = train[[
        "GW", "Fixture Seq", "Team", "Opponent", "H/A",
        "Market Team xG", "Market Source",
    ]].copy()
    fit_table["Model xG"] = fitted
    fit_table["Error"] = fit_table["Model xG"] - fit_table["Market Team xG"]
    fit_table["Abs Error"] = fit_table["Error"].abs()
    fit_table["CV Model xG"] = cv_pred.reindex(train.index).to_numpy()
    fit_table["CV Error"] = fit_table["CV Model xG"] - fit_table["Market Team xG"]

    rmse = float(np.sqrt(np.mean(fit_table["Error"] ** 2)))
    mae = float(fit_table["Abs Error"].mean())

    return {
        "coefficients": dict(zip(FEATURE_NAMES, beta)),
        "beta": beta,
        "pl_mu": pl_mu,
        "ridge_lambda": ridge_lambda,
        "rmse": rmse,
        "mae": mae,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "market_rows": int(len(train)),
        "market_fixtures": int(train.apply(_fixture_key, axis=1).nunique()),
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

    fit = fit_fixture_model(team_hist, market_odds, ridge_lambda=ridge_lambda)
    strengths = fit["strengths"]
    beta = fit["beta"]
    pl_mu = fit["pl_mu"]

    f = fixtures[[
        "GW", "Fixture ID", "Kickoff UTC", "Team", "Opponent", "H/A",
        "Official FDR",
    ]].copy()

    f["GW"] = pd.to_numeric(f["GW"], errors="coerce")
    f["Fixture ID"] = pd.to_numeric(f["Fixture ID"], errors="coerce")
    f = f[f["GW"].between(start_gw, max_gw)].copy()

    f = _add_strength_features(f, strengths)
    X = _design_matrix(f)
    f["Strength Model xG"] = pl_mu * np.exp(X @ beta)

    market = _market_team_rows(market_odds)
    f = f.merge(
        market[[
            "GW", "Team", "Opponent", "Market Team xG", "Market CS Prob",
            "Market Source", "Market Fit RMSE",
        ]],
        on=["GW", "Team", "Opponent"],
        how="left",
        validate="one_to_one",
    )

    f["Team xG"] = f["Market Team xG"].combine_first(f["Strength Model xG"])
    f["Market vs Strength xG"] = f["Market Team xG"] - f["Strength Model xG"]
    f["xG Source"] = np.where(
        f["Market Team xG"].notna(),
        f["Market Source"].fillna("Market"),
        "Strength model",
    )

    opp_lookup = f[["GW", "Fixture ID", "Team", "Team xG"]].rename(
        columns={"Team": "Opponent", "Team xG": "Opp xG"}
    )
    f = f.merge(
        opp_lookup,
        on=["GW", "Fixture ID", "Opponent"],
        how="left",
        validate="one_to_one",
    )

    f["CS Prob"] = f["Market CS Prob"].combine_first(np.exp(-f["Opp xG"]))
    f["CS Prob Source"] = np.where(
        f["Market CS Prob"].notna(),
        f["Market Source"].fillna("Market"),
        "Poisson from projected opponent xG",
    )

    return (
        f.sort_values(["GW", "Fixture ID", "H/A"]).reset_index(drop=True),
        fit,
    )
