import numpy as np
import pandas as pd


def build_gw1_team_xg(market_odds: pd.DataFrame) -> pd.DataFrame:
    home = market_odds[
        ["Home", "Away", "Market Home xG", "Market Away xG"]
    ].copy()

    home.columns = [
        "Team",
        "Opponent",
        "Team xG",
        "Opponent xG",
    ]
    home["H/A"] = "H"

    away = market_odds[
        ["Away", "Home", "Market Away xG", "Market Home xG"]
    ].copy()

    away.columns = [
        "Team",
        "Opponent",
        "Team xG",
        "Opponent xG",
    ]
    away["H/A"] = "A"

    return pd.concat(
        [home, away],
        ignore_index=True,
    )


def build_attack_projection(
    minutes: pd.DataFrame,
    attack_priors: pd.DataFrame,
    workbook_players: pd.DataFrame,
    market_odds: pd.DataFrame = None,
    external_priors=None,
    team_xg: pd.DataFrame = None,
    output_prefix: str = "GW1",
) -> pd.DataFrame:

    # --------------------------------
    # 1. Existing workbook priors
    # --------------------------------

    priors = attack_priors.merge(
        workbook_players[
            ["Player ID", "Code"]
        ],
        on="Player ID",
        how="left",
    )

    priors = priors[
        [
            "Code",
            "FPL Pos",
            "Current xG Share Prior",
            "Current xA Share Prior",
            "Uncertainty",
            "History Quality",
            "Penalty Contamination?",
        ]
    ].copy()

    # --------------------------------
    # 2. Position fallback priors
    # --------------------------------

    fallback = (
        priors.groupby("FPL Pos")
        .agg(
            Fallback_xG=(
                "Current xG Share Prior",
                "median",
            ),
            Fallback_xA=(
                "Current xA Share Prior",
                "median",
            ),
        )
        .reset_index()
    )

    # --------------------------------
    # 3. Merge minutes + historical priors
    # --------------------------------

    df = minutes.merge(
        priors,
        on=["Code", "FPL Pos"],
        how="left",
    )

    # --------------------------------
    # 4. External/FotMob priors
    # --------------------------------

    if external_priors is not None and not external_priors.empty:

        external = external_priors[
            [
                "Code",
                "External xG Share",
                "External xA Share",
                "External League",
                "External Team",
            ]
        ].drop_duplicates("Code")

        df = df.merge(
            external,
            on="Code",
            how="left",
        )

    else:
        df["External xG Share"] = np.nan
        df["External xA Share"] = np.nan
        df["External League"] = pd.NA
        df["External Team"] = pd.NA

    # --------------------------------
    # 5. Position fallback
    # --------------------------------

    df = df.merge(
        fallback,
        on="FPL Pos",
        how="left",
    )

    # --------------------------------
    # 6. Choose attacking prior source
    # --------------------------------

    df["xG Share Used"] = (
        df["Current xG Share Prior"]
        .fillna(df["External xG Share"])
        .fillna(df["Fallback_xG"])
    )

    df["xA Share Used"] = (
        df["Current xA Share Prior"]
        .fillna(df["External xA Share"])
        .fillna(df["Fallback_xA"])
    )

    df["Attack Prior Source"] = np.select(
        [
            df["Current xG Share Prior"].notna(),
            df["External xG Share"].notna(),
        ],
        [
            "Player prior",
            "External prior",
        ],
        default="Position fallback",
    )

    # ------------------------------
    # 7. Fixture team xG
    # ------------------------------

    # Backward-compatible GW1 behaviour:
    # if no explicit team_xg table is supplied,
    # build it from the bookmaker market.
    if team_xg is None:

        if market_odds is None:
            raise ValueError(
                "Provide either market_odds "
                "or a team_xg DataFrame."
            )

        fixture_team_xg = build_gw1_team_xg(
            market_odds
        )

    else:

        required = {
            "Team",
            "Team xG",
        }

        missing = (
            required
            - set(team_xg.columns)
        )

        if missing:
            raise ValueError(
                "team_xg missing required columns: "
                f"{sorted(missing)}"
            )

        fixture_team_xg = (
            team_xg[
                [
                    "Team",
                    "Team xG",
                ]
            ]
            .drop_duplicates("Team")
            .copy()
        )


    df = df.merge(
        fixture_team_xg,
        on="Team",
        how="left",
        validate="many_to_one",
    )

    if df["Team xG"].isna().any():

        missing_teams = (
            df.loc[
                df["Team xG"].isna(),
                "Team",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Missing team xG for: "
            f"{missing_teams}"
        )

    # --------------------------------
    # 8. Minutes-adjusted attacking weights
    # --------------------------------

    df["Raw xG Weight"] = (
        df["xG Share Used"]
        * df["Effective Mins"]
        / 90
    )

    df["Raw xA Weight"] = (
        df["xA Share Used"]
        * df["Effective Mins"]
        / 90
    )

    # --------------------------------
    # 9. Reconcile player xG to team xG
    # --------------------------------

    team_weight = (
        df.groupby("Team")["Raw xG Weight"]
        .transform("sum")
    )

    df["Attack Reconciliation Factor"] = np.where(
        team_weight > 0,
        1 / team_weight,
        np.nan,
    )

    xg_output_col = (
        f"{output_prefix} xG"
    )

    xa_output_col = (
        f"{output_prefix} xA"
    )

    df[xg_output_col] = (
        df["Team xG"]
        * df["Raw xG Weight"]
        * df["Attack Reconciliation Factor"]
    )

    df[xa_output_col] = (
        df["Team xG"]
        * df["Raw xA Weight"]
        * df["Attack Reconciliation Factor"]
    )

    return df

def build_attack_horizon(
    minutes: pd.DataFrame,
    attack_priors: pd.DataFrame,
    workbook_players: pd.DataFrame,
    fixture_horizon: pd.DataFrame,
    external_priors=None,
    start_gw: int = 1,
    max_gw: int = 6,
) -> pd.DataFrame:

    required = {
        "GW",
        "Team",
        "Opponent",
        "H/A",
        "Team xG",
        "Opp xG",
        "CS Prob",
        "xG Source",
    }

    missing = (
        required
        - set(fixture_horizon.columns)
    )

    if missing:
        raise ValueError(
            "fixture_horizon missing: "
            f"{sorted(missing)}"
        )

    if start_gw < 1:
        raise ValueError("start_gw must be >= 1")
    if max_gw < start_gw:
        raise ValueError("max_gw must be >= start_gw")

    pieces = []

    gameweeks = sorted(
        fixture_horizon.loc[
            fixture_horizon["GW"].between(start_gw, max_gw),
            "GW",
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    for gw in gameweeks:

        fx = fixture_horizon[
            fixture_horizon["GW"] == gw
        ].copy()

        prefix = f"GW{gw}"

        if "GW" in minutes.columns:
            gw_minutes = minutes[
                minutes["GW"] == gw
            ].copy()
        else:
            gw_minutes = minutes

        projection = build_attack_projection(
            minutes=gw_minutes,
            attack_priors=attack_priors,
            workbook_players=workbook_players,
            market_odds=None,
            external_priors=external_priors,
            team_xg=fx[
                [
                    "Team",
                    "Team xG",
                ]
            ],
            output_prefix=prefix,
        )

        # Standardise output names so downstream
        # code doesn't care which GW it is.
        projection["xG"] = (
            projection[
                f"{prefix} xG"
            ]
        )

        projection["xA"] = (
            projection[
                f"{prefix} xA"
            ]
        )

        projection["GW"] = gw

        # Add fixture context.
        projection = projection.merge(
            fx[
                [
                    "Team",
                    "Opponent",
                    "H/A",
                    "Opp xG",
                    "CS Prob",
                    "xG Source",
                ]
            ],
            on="Team",
            how="left",
            validate="many_to_one",
        )

        pieces.append(
            projection
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )