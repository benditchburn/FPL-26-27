import numpy as np
import pandas as pd


def _shrink(success_sum, n, prior, k):
    if pd.isna(n) or n == 0:
        return prior

    return (
        success_sum + k * prior
    ) / (n + k)


def _expected_save_points_poisson(lam, max_saves=30):
    """E[floor(saves / 3)] for Poisson-distributed saves."""
    if pd.isna(lam) or lam <= 0:
        return 0.0

    lam = float(lam)
    p = np.exp(-lam)
    total = 0.0
    prob_sum = p

    # k = 0 contributes zero save points.
    for k in range(1, max_saves + 1):
        p = p * lam / k
        prob_sum += p
        total += (k // 3) * p

    # Tail probability is tiny at realistic GK save rates. Assign it the
    # conservative max bucket rather than silently dropping it.
    tail = max(0.0, 1.0 - prob_sum)
    total += (max_saves // 3) * tail
    return float(total)


def _fit_gk_save_model(h: pd.DataFrame, player_shrink_k: float = 8.0):
    """Fit a simple fixture-sensitive GK save model.

    Historical saves are modelled as a function of xG conceded while the GK is
    on the pitch. This prevents a low-history keeper from receiving both strong
    clean-sheet odds and an unrelated league-average save-points prior.

    Model:
        saves/90 = intercept + slope * xGC/90 + shrunk_player_residual
    """

    required = {"FPL Pos", "starts", "minutes", "saves", "expected_goals_conceded"}
    if not required.issubset(h.columns):
        return {
            "intercept": np.nan,
            "slope": np.nan,
            "player_residuals": pd.DataFrame(
                columns=["player_code", "GK Save Player Residual", "GK Save Sample"]
            ),
            "n": 0,
        }

    gk = h[
        (h["FPL Pos"] == "GK")
        & (h["starts"] > 0)
        & (h["minutes"] >= 45)
    ].copy()

    gk["expected_goals_conceded"] = pd.to_numeric(
        gk["expected_goals_conceded"], errors="coerce"
    )
    gk["saves"] = pd.to_numeric(gk["saves"], errors="coerce")
    gk["minutes"] = pd.to_numeric(gk["minutes"], errors="coerce")

    gk = gk[
        gk["expected_goals_conceded"].notna()
        & gk["saves"].notna()
        & gk["minutes"].gt(0)
    ].copy()

    if len(gk) < 30:
        return {
            "intercept": np.nan,
            "slope": np.nan,
            "player_residuals": pd.DataFrame(
                columns=["player_code", "GK Save Player Residual", "GK Save Sample"]
            ),
            "n": len(gk),
        }

    scale = 90.0 / gk["minutes"].clip(lower=1.0)
    gk["xGC90"] = (gk["expected_goals_conceded"] * scale).clip(0.0, 5.0)
    gk["Saves90"] = (gk["saves"] * scale).clip(0.0, 12.0)

    X = np.column_stack([
        np.ones(len(gk)),
        gk["xGC90"].to_numpy(dtype=float),
    ])
    y = gk["Saves90"].to_numpy(dtype=float)
    weights = (gk["minutes"] / 90.0).clip(0.5, 1.0).to_numpy(dtype=float)

    sw = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)

    # Saves should not fall when opponent scoring threat rises. Clip only to
    # plausible broad bounds; the historical fit still determines the values.
    intercept = float(np.clip(beta[0], 0.0, 4.0))
    slope = float(np.clip(beta[1], 0.0, 4.0))

    gk["Global Expected Saves90"] = intercept + slope * gk["xGC90"]
    gk["Save Residual"] = gk["Saves90"] - gk["Global Expected Saves90"]

    player = (
        gk.groupby("player_code", as_index=False)
        .agg(
            GK_Save_Residual_Sum=("Save Residual", "sum"),
            GK_Save_Sample=("Save Residual", "size"),
        )
    )
    player["GK Save Player Residual"] = (
        player["GK_Save_Residual_Sum"]
        / (player["GK_Save_Sample"] + float(player_shrink_k))
    )
    player = player.rename(columns={"GK_Save_Sample": "GK Save Sample"})[
        ["player_code", "GK Save Player Residual", "GK Save Sample"]
    ]

    return {
        "intercept": intercept,
        "slope": slope,
        "player_residuals": player,
        "n": len(gk),
    }


def build_defensive_priors(
    current_players: pd.DataFrame,
    hist: pd.DataFrame,
) -> pd.DataFrame:

    h = hist.copy()

    for col in [
        "minutes",
        "starts",
        "saves",
        "defensive_contribution",
        "expected_goals_conceded",
    ]:
        if col not in h.columns:
            h[col] = np.nan if col == "expected_goals_conceded" else 0.0
        h[col] = pd.to_numeric(
            h[col],
            errors="coerce",
        )
        if col != "expected_goals_conceded":
            h[col] = h[col].fillna(0)

    # Only use games where player was considered available
    chance = pd.to_numeric(
        h["chance_of_playing_this_round"],
        errors="coerce",
    )

    h["Available"] = (
        (h["status"] == "a")
        | (chance > 0)
    )

    h = h[h["Available"]].copy()

    h["Started"] = h["starts"] > 0

    # --------------------------------
    # Defensive contribution points
    # --------------------------------

    h["DC Threshold"] = np.select(
        [
            h["FPL Pos"] == "DEF",
            h["FPL Pos"].isin(["MID", "FWD"]),
        ],
        [10, 12],
        default=np.nan,
    )

    h["DC Hit"] = (
        h["defensive_contribution"]
        >= h["DC Threshold"]
    ).astype(float)

    # --------------------------------
    # Legacy GK save-point priors
    # --------------------------------
    # Retained as a fallback if fixture Opp xG is unavailable. Normal horizon
    # runs should use the dynamic fixture-sensitive save model below.

    h["Save Points"] = np.where(
        h["FPL Pos"] == "GK",
        np.floor(h["saves"] / 3),
        0.0,
    )

    start = h[h["Started"]].copy()
    bench = h[~h["Started"]].copy()

    pos_dc_start = (
        start.groupby("FPL Pos")["DC Hit"]
        .mean()
        .to_dict()
    )

    pos_dc_bench = (
        bench.groupby("FPL Pos")["DC Hit"]
        .mean()
        .to_dict()
    )

    pos_save_start = (
        start.groupby("FPL Pos")["Save Points"]
        .mean()
        .to_dict()
    )

    pos_save_bench = (
        bench.groupby("FPL Pos")["Save Points"]
        .mean()
        .to_dict()
    )

    p_start = (
        start.groupby("player_code")
        .agg(
            Start_N=("Started", "size"),
            DC_Start_Hits=("DC Hit", "sum"),
            SavePts_Start_Total=("Save Points", "sum"),
        )
        .reset_index()
    )

    p_bench = (
        bench.groupby("player_code")
        .agg(
            Bench_N=("Started", "size"),
            DC_Bench_Hits=("DC Hit", "sum"),
            SavePts_Bench_Total=("Save Points", "sum"),
        )
        .reset_index()
    )

    out = current_players[
        [
            "Player ID",
            "Code",
            "Player",
            "Team",
            "FPL Pos",
        ]
    ].copy()

    out = out.merge(
        p_start,
        left_on="Code",
        right_on="player_code",
        how="left",
    )

    out = out.merge(
        p_bench,
        left_on="Code",
        right_on="player_code",
        how="left",
        suffixes=("", "_bench"),
    )

    K = 4

    def calc(row):
        pos = row["FPL Pos"]

        dc_start_prior = pos_dc_start.get(pos, 0.0)
        dc_bench_prior = pos_dc_bench.get(pos, 0.0)

        save_start_prior = pos_save_start.get(pos, 0.0)
        save_bench_prior = pos_save_bench.get(pos, 0.0)

        n_start = row["Start_N"]
        n_bench = row["Bench_N"]

        p_dc_start = _shrink(
            row["DC_Start_Hits"],
            n_start,
            dc_start_prior,
            K,
        )

        p_dc_bench = _shrink(
            row["DC_Bench_Hits"],
            n_bench,
            dc_bench_prior,
            K,
        )

        if pd.isna(n_start) or n_start == 0:
            save_start = save_start_prior
        else:
            save_start = (
                row["SavePts_Start_Total"]
                + K * save_start_prior
            ) / (n_start + K)

        if pd.isna(n_bench) or n_bench == 0:
            save_bench = save_bench_prior
        else:
            save_bench = (
                row["SavePts_Bench_Total"]
                + K * save_bench_prior
            ) / (n_bench + K)

        return pd.Series({
            "P DC | Start": p_dc_start,
            "P DC | Not Start": p_dc_bench,
            "Save Pts | Start": save_start,
            "Save Pts | Not Start": save_bench,
        })

    probs = out.apply(calc, axis=1)
    out = pd.concat([out, probs], axis=1)

    # --------------------------------
    # Fixture-sensitive GK save model
    # --------------------------------

    save_fit = _fit_gk_save_model(h, player_shrink_k=8.0)
    player_resid = save_fit["player_residuals"]

    if not player_resid.empty:
        out = out.merge(
            player_resid,
            left_on="Code",
            right_on="player_code",
            how="left",
            suffixes=("", "_gk_save"),
        )
    else:
        out["GK Save Player Residual"] = np.nan
        out["GK Save Sample"] = np.nan

    out["GK Save Intercept"] = save_fit["intercept"]
    out["GK Save xG Slope"] = save_fit["slope"]
    out["GK Save Player Residual"] = pd.to_numeric(
        out["GK Save Player Residual"], errors="coerce"
    ).fillna(0.0)
    out["GK Save Sample"] = pd.to_numeric(
        out["GK Save Sample"], errors="coerce"
    ).fillna(0.0)
    out["GK Save Fit N"] = save_fit["n"]

    return out


def build_defensive_xpts(
    start_probs,
    defensive_priors,
    defcon_calibrator=None,
):

    wanted = [
        "Player ID",
        "FPL Pos",
        "Conditional Start Prob",
        "Availability Prob",
    ]
    for optional in [
        "Opp xG",
        "Mins If Start",
        "Expected Mins If Not Start",
    ]:
        if optional in start_probs.columns:
            wanted.append(optional)

    df = start_probs[wanted].merge(
        defensive_priors[
            [
                "Player ID",
                "P DC | Start",
                "P DC | Not Start",
                "Save Pts | Start",
                "Save Pts | Not Start",
                "GK Save Intercept",
                "GK Save xG Slope",
                "GK Save Player Residual",
                "GK Save Sample",
            ]
        ],
        on="Player ID",
        how="left",
    )

    s = df["Conditional Start Prob"]
    a = df["Availability Prob"]

    df["P DC Start Used"] = df["P DC | Start"]

    if defcon_calibrator is not None:

        from src.defcon_calibration import (
            calibrate_defender_prob,
        )

        mask = df["FPL Pos"] == "DEF"

        df.loc[
            mask,
            "P DC Start Used"
        ] = calibrate_defender_prob(
            df.loc[
                mask,
                "P DC | Start"
            ],
            defcon_calibrator,
        )

    df["P Defensive Return"] = a * (
        s * df["P DC Start Used"]
        + (1 - s) * df["P DC | Not Start"]
    )

    df["xPts DefCon"] = (
        2 * df["P Defensive Return"]
    )

    # Legacy static save estimate, retained as a safe fallback.
    df["xPts Saves Static"] = a * (
        s * df["Save Pts | Start"]
        + (1 - s) * df["Save Pts | Not Start"]
    )
    df["xPts Saves"] = df["xPts Saves Static"]
    df["Projected Saves/90"] = np.nan
    df["Projected Saves | Start"] = np.nan
    df["Save Model"] = "Static historical fallback"

    dynamic_mask = (
        df["FPL Pos"].eq("GK")
        & ("Opp xG" in df.columns)
        & df["Opp xG"].notna()
        & df["GK Save Intercept"].notna()
        & df["GK Save xG Slope"].notna()
    )

    if dynamic_mask.any():
        opp_xg = pd.to_numeric(df.loc[dynamic_mask, "Opp xG"], errors="coerce")
        saves90 = (
            df.loc[dynamic_mask, "GK Save Intercept"].astype(float)
            + df.loc[dynamic_mask, "GK Save xG Slope"].astype(float) * opp_xg
            + df.loc[dynamic_mask, "GK Save Player Residual"].astype(float)
        ).clip(lower=0.05, upper=10.0)

        if "Mins If Start" in df.columns:
            mins_start = pd.to_numeric(
                df.loc[dynamic_mask, "Mins If Start"], errors="coerce"
            ).fillna(90.0).clip(0.0, 90.0)
        else:
            mins_start = pd.Series(90.0, index=df.index[dynamic_mask])

        if "Expected Mins If Not Start" in df.columns:
            mins_bench = pd.to_numeric(
                df.loc[dynamic_mask, "Expected Mins If Not Start"], errors="coerce"
            ).fillna(0.0).clip(0.0, 90.0)
        else:
            mins_bench = pd.Series(0.0, index=df.index[dynamic_mask])

        lam_start = saves90 * mins_start / 90.0
        lam_bench = saves90 * mins_bench / 90.0

        pts_start = lam_start.map(_expected_save_points_poisson)
        pts_bench = lam_bench.map(_expected_save_points_poisson)

        a_dyn = a.loc[dynamic_mask]
        s_dyn = s.loc[dynamic_mask]

        df.loc[dynamic_mask, "xPts Saves"] = (
            a_dyn * (
                s_dyn * pts_start
                + (1.0 - s_dyn) * pts_bench
            )
        )
        df.loc[dynamic_mask, "Projected Saves/90"] = saves90
        df.loc[dynamic_mask, "Projected Saves | Start"] = lam_start
        df.loc[dynamic_mask, "Save Model"] = "Fixture xG + shrunk GK residual"

    # Non-goalkeepers cannot earn save points.
    df.loc[~df["FPL Pos"].eq("GK"), "xPts Saves"] = 0.0

    return df
