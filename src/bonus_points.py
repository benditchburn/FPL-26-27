import pandas as pd


def _shrink(total, n, prior, k=8):
    if pd.isna(n) or n == 0:
        return prior

    return (total + k * prior) / (n + k)


def build_bonus_priors(
    current_players: pd.DataFrame,
    hist: pd.DataFrame,
) -> pd.DataFrame:

    h = hist.copy()

    h["minutes"] = pd.to_numeric(
        h["minutes"], errors="coerce"
    ).fillna(0)

    h["starts"] = pd.to_numeric(
        h["starts"], errors="coerce"
    ).fillna(0)

    h["bonus"] = pd.to_numeric(
        h["bonus"], errors="coerce"
    ).fillna(0)

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

    start = h[h["Started"]]
    bench = h[~h["Started"]]

    # Position priors
    pos_start = (
        start.groupby("FPL Pos")["bonus"]
        .mean()
        .to_dict()
    )

    pos_bench = (
        bench.groupby("FPL Pos")["bonus"]
        .mean()
        .to_dict()
    )

    p_start = (
        start.groupby("player_code")
        .agg(
            Bonus_Start_Total=("bonus", "sum"),
            Bonus_Start_N=("bonus", "size"),
        )
        .reset_index()
    )

    p_bench = (
        bench.groupby("player_code")
        .agg(
            Bonus_Bench_Total=("bonus", "sum"),
            Bonus_Bench_N=("bonus", "size"),
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

    def calc(row):
        pos = row["FPL Pos"]

        return pd.Series({
            "Bonus | Start": _shrink(
                row["Bonus_Start_Total"],
                row["Bonus_Start_N"],
                pos_start.get(pos, 0),
            ),
            "Bonus | Not Start": _shrink(
                row["Bonus_Bench_Total"],
                row["Bonus_Bench_N"],
                pos_bench.get(pos, 0),
            ),
        })

    probs = out.apply(calc, axis=1)

    return pd.concat([out, probs], axis=1)


def build_bonus_xpts(
    start_probs: pd.DataFrame,
    bonus_priors: pd.DataFrame,
) -> pd.DataFrame:

    df = start_probs[
        [
            "Player ID",
            "Conditional Start Prob",
            "Availability Prob",
        ]
    ].merge(
        bonus_priors[
            [
                "Player ID",
                "Bonus | Start",
                "Bonus | Not Start",
            ]
        ],
        on="Player ID",
        how="left",
    )

    s = df["Conditional Start Prob"]
    a = df["Availability Prob"]

    df["xPts Bonus"] = a * (
        s * df["Bonus | Start"]
        + (1 - s) * df["Bonus | Not Start"]
    )

    return df