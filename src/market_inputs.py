from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


MARKET_COLUMNS = [
    "GW",
    "Fixture Seq",
    "Home",
    "Away",
    "Market Home xG",
    "Market Away xG",
    "Home CS Prob",
    "Away CS Prob",
    "Market Source",
]


def _poisson_pmf(lam: float, max_goals: int = 12) -> np.ndarray:
    lam = float(lam)
    probs = np.empty(max_goals + 1, dtype=float)
    probs[0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        probs[k] = probs[k - 1] * lam / k
    probs[-1] += max(0.0, 1.0 - probs.sum())
    return probs


def poisson_match_probs(
    home_xg: float,
    away_xg: float,
    max_goals: int = 12,
) -> dict[str, float]:
    """Independent-Poisson 1X2 and O/U 2.5 probabilities."""
    hp = _poisson_pmf(home_xg, max_goals=max_goals)
    ap = _poisson_pmf(away_xg, max_goals=max_goals)
    score = np.outer(hp, ap)

    home = float(np.tril(score, k=-1).sum())
    draw = float(np.trace(score))
    away = float(np.triu(score, k=1).sum())

    over25 = 0.0
    for h in range(score.shape[0]):
        for a in range(score.shape[1]):
            if h + a >= 3:
                over25 += float(score[h, a])

    return {
        "home": home,
        "draw": draw,
        "away": away,
        "over25": over25,
        "under25": 1.0 - over25,
    }


def _normalise_implied_probs(*odds: float) -> np.ndarray:
    arr = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 1.0):
        raise ValueError("Decimal odds must all be finite and > 1.0")
    raw = 1.0 / arr
    return raw / raw.sum()


def derive_team_xg_from_odds(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    over25_odds: float | None = None,
    under25_odds: float | None = None,
) -> dict[str, float]:
    """Infer team scoring rates from bookmaker 1X2 (+ optional O/U 2.5).

    Bookmaker overround is removed within each market. The two Poisson scoring
    rates are then chosen to match the no-vig probabilities as closely as
    possible. This is intentionally a simple transparent market layer rather
    than a claim that football scores are exactly independent Poisson.
    """
    p1x2 = _normalise_implied_probs(home_odds, draw_odds, away_odds)

    have_totals = (over25_odds is not None) and (under25_odds is not None)
    if have_totals:
        p_ou = _normalise_implied_probs(over25_odds, under25_odds)
        target_over = float(p_ou[0])
    else:
        target_over = None

    def residual(log_lams):
        home_xg, away_xg = np.exp(log_lams)
        probs = poisson_match_probs(home_xg, away_xg)
        res = [
            probs["home"] - p1x2[0],
            probs["draw"] - p1x2[1],
            probs["away"] - p1x2[2],
        ]
        if target_over is not None:
            # Totals is especially informative about the overall scoring level.
            res.append(1.25 * (probs["over25"] - target_over))
        return np.asarray(res, dtype=float)

    # 1.45/1.15 is a sensible neutral PL-ish starting point, but optimisation
    # is bounded broadly enough for very uneven fixtures.
    fit = least_squares(
        residual,
        x0=np.log([1.45, 1.15]),
        bounds=(np.log([0.05, 0.05]), np.log([5.0, 5.0])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=500,
    )

    home_xg, away_xg = np.exp(fit.x)
    probs = poisson_match_probs(home_xg, away_xg)
    residuals = residual(fit.x)

    return {
        "Market Home xG": float(home_xg),
        "Market Away xG": float(away_xg),
        "Home CS Prob": float(np.exp(-away_xg)),
        "Away CS Prob": float(np.exp(-home_xg)),
        "Market Fit RMSE": float(np.sqrt(np.mean(residuals ** 2))),
        "Model Home Win Prob": probs["home"],
        "Model Draw Prob": probs["draw"],
        "Model Away Win Prob": probs["away"],
        "Model Over 2.5 Prob": probs["over25"],
    }


def prepare_market_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a user/current-market table to the workbook schema.

    Supported inputs:
    1. direct xG columns: Market Home xG / Market Away xG
    2. decimal odds: Home Odds / Draw Odds / Away Odds, optionally with
       Over 2.5 Odds / Under 2.5 Odds.

    Clean-sheet probabilities are inferred from xG when not supplied.
    """
    out = df.copy()

    required_keys = {"GW", "Home", "Away"}
    missing_keys = required_keys - set(out.columns)
    if missing_keys:
        raise ValueError(
            "market override missing required columns: "
            f"{sorted(missing_keys)}"
        )

    out["GW"] = pd.to_numeric(out["GW"], errors="raise").astype(int)
    if "Fixture Seq" not in out.columns:
        out["Fixture Seq"] = np.nan

    direct = {"Market Home xG", "Market Away xG"}.issubset(out.columns)
    odds = {"Home Odds", "Draw Odds", "Away Odds"}.issubset(out.columns)

    if not direct and not odds:
        raise ValueError(
            "market override needs direct Market Home/Away xG or "
            "Home/Draw/Away Odds"
        )

    if not direct:
        derived = []
        for _, row in out.iterrows():
            over = row.get("Over 2.5 Odds")
            under = row.get("Under 2.5 Odds")
            if pd.isna(over) or pd.isna(under):
                over = None
                under = None
            derived.append(
                derive_team_xg_from_odds(
                    row["Home Odds"],
                    row["Draw Odds"],
                    row["Away Odds"],
                    over25_odds=over,
                    under25_odds=under,
                )
            )
        derived_df = pd.DataFrame(derived, index=out.index)
        for col in derived_df.columns:
            out[col] = derived_df[col]

    for col in ["Market Home xG", "Market Away xG"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any() or (~out[col].between(0.05, 5.0)).any():
            raise ValueError(f"{col} must be between 0.05 and 5.0")

    if "Home CS Prob" not in out.columns:
        out["Home CS Prob"] = np.exp(-out["Market Away xG"])
    if "Away CS Prob" not in out.columns:
        out["Away CS Prob"] = np.exp(-out["Market Home xG"])

    for col in ["Home CS Prob", "Away CS Prob"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any() or (~out[col].between(0.0, 1.0)).any():
            raise ValueError(f"{col} must be between 0 and 1")

    if "Market Source" not in out.columns:
        out["Market Source"] = np.where(
            odds,
            "Odds-derived override",
            "Direct xG override",
        )

    keep = [c for c in MARKET_COLUMNS if c in out.columns]
    extra = [
        c for c in ["Market Fit RMSE"]
        if c in out.columns
    ]
    return out[keep + extra].copy()


def load_market_overrides(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=MARKET_COLUMNS)
    return prepare_market_overrides(pd.read_csv(path))


def combine_market_odds(
    base: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay current market inputs on the workbook market table."""
    b = base.copy()
    if overrides is None or overrides.empty:
        if "Market Source" not in b.columns:
            b["Market Source"] = "Workbook market"
        return b

    o = prepare_market_overrides(overrides)
    keys = ["GW", "Home", "Away"]

    b["GW"] = pd.to_numeric(b["GW"], errors="coerce")
    if "Market Source" not in b.columns:
        b["Market Source"] = "Workbook market"

    # Keep workbook-only metadata such as Fixture Seq while replacing current
    # market estimates on matching fixtures.
    merged = b.merge(
        o,
        on=keys,
        how="outer",
        suffixes=("", "__override"),
        validate="one_to_one",
    )

    override_cols = [
        "Fixture Seq",
        "Market Home xG",
        "Market Away xG",
        "Home CS Prob",
        "Away CS Prob",
        "Market Source",
        "Market Fit RMSE",
    ]

    for col in override_cols:
        ocol = f"{col}__override"
        if ocol not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = merged[ocol].combine_first(merged[col])
        else:
            merged[col] = merged[ocol]
        merged = merged.drop(columns=ocol)

    return merged
