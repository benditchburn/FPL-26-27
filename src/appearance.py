import pandas as pd


def _shrunk_rate(successes, n, prior, k):
    if pd.isna(n) or n == 0:
        return prior

    return (successes + k * prior) / (n + k)


def build_appearance_priors(
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

    chance = pd.to_numeric(
        h["chance_of_playing_this_round"],
        errors="coerce",
    )

    h["Historically Available"] = (
        (h["status"] == "a")
        | (chance > 0)
    )

    h = h[h["Historically Available"]].copy()

    h["Started"] = h["starts"] > 0
    h["Appeared"] = h["minutes"] > 0
    h["Played 60+"] = h["minutes"] >= 60

    # -----------------------
    # Position-level priors
    # -----------------------

    start_rows = h[h["Started"]]
    bench_rows = h[~h["Started"]]

    pos_start_60 = (
        start_rows.groupby("FPL Pos")["Played 60+"]
        .mean()
        .to_dict()
    )

    pos_bench_app = (
        bench_rows.groupby("FPL Pos")["Appeared"]
        .mean()
        .to_dict()
    )

    pos_bench_60 = (
        bench_rows.groupby("FPL Pos")["Played 60+"]
        .mean()
        .to_dict()
    )

    # -----------------------
    # Player histories
    # -----------------------

    start = (
        start_rows.groupby("player_code")
        .agg(
            Starts=("Started", "size"),
            Start60=("Played 60+", "sum"),
        )
        .reset_index()
    )

    bench = (
        bench_rows.groupby("player_code")
        .agg(
            NonStarts=("Started", "size"),
            BenchApps=("Appeared", "sum"),
            Bench60=("Played 60+", "sum"),
        )
        .reset_index()
    )

    out = current_players[
        ["Player ID", "Code", "Player", "Team", "FPL Pos"]
    ].copy()

    out = out.merge(
        start,
        left_on="Code",
        right_on="player_code",
        how="left",
    )

    out = out.merge(
        bench,
        left_on="Code",
        right_on="player_code",
        how="left",
        suffixes=("", "_bench"),
    )

    # Moderate shrinkage toward positional behaviour
    K_START = 8
    K_BENCH = 10

    def calc(row):
        pos = row["FPL Pos"]

        start60_prior = pos_start_60.get(pos, 0.80)
        bench_app_prior = pos_bench_app.get(pos, 0.25)
        bench60_prior = pos_bench_60.get(pos, 0.01)

        p60_start = _shrunk_rate(
            row["Start60"],
            row["Starts"],
            start60_prior,
            K_START,
        )

        papp_bench = _shrunk_rate(
            row["BenchApps"],
            row["NonStarts"],
            bench_app_prior,
            K_BENCH,
        )

        p60_bench = _shrunk_rate(
            row["Bench60"],
            row["NonStarts"],
            bench60_prior,
            K_BENCH,
        )

        return pd.Series({
            "P60 If Start": p60_start,
            "P(App | Not Start)": papp_bench,
            "P60 If Not Start": p60_bench,
        })

    probs = out.apply(calc, axis=1)

    return pd.concat([out, probs], axis=1)


def build_appearance_probs(
    start_probs: pd.DataFrame,
    appearance_priors: pd.DataFrame,
) -> pd.DataFrame:

    df = start_probs.merge(
        appearance_priors[
            [
                "Player ID",
                "P60 If Start",
                "P(App | Not Start)",
                "P60 If Not Start",
            ]
        ],
        on="Player ID",
        how="left",
    )

    s = df["Conditional Start Prob"]
    avail = df["Availability Prob"]

    df["Appearance Prob"] = avail * (
        s
        + (1 - s) * df["P(App | Not Start)"]
    )

    df["P60"] = avail * (
        s * df["P60 If Start"]
        + (1 - s) * df["P60 If Not Start"]
    )

    # Useful directly for FPL appearance points:
    # 1 point for appearing + extra 1 if 60+
    df["Expected Appearance Pts"] = (
        df["Appearance Prob"] + df["P60"]
    )

    return df