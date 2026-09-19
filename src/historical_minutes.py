from pathlib import Path
import pandas as pd
import requests
from io import StringIO


BASE = (
    "https://raw.githubusercontent.com/"
    "olbauday/FPL-Core-Insights/main/"
    "data/2025-2026"
)


POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}


def _read_url(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def fetch_historical_minute_data(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "minute_history_2025_26.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    players = _read_url(
        f"{BASE}/players.csv"
    )

    frames = []

    for gw in range(1, 39):
        url = (
            f"{BASE}/By%20Gameweek/GW{gw}/"
            "player_gameweek_stats.csv"
        )

        try:
            df = _read_url(url)
        except Exception:
            continue

        df["GW"] = gw
        frames.append(df)

    hist = pd.concat(frames, ignore_index=True)

    hist = hist.merge(
        players[
            ["player_id", "player_code", "position"]
        ],
        left_on="id",
        right_on="player_id",
        how="left",
    )

    hist["FPL Pos"] = hist["position"].map(POSITION_MAP)

    hist.to_csv(cache_file, index=False)

    return hist


def build_minute_priors(
    current_players: pd.DataFrame,
    hist: pd.DataFrame,
) -> pd.DataFrame:

    h = hist.copy()

    # Treat available or explicitly >0% chance as eligible.
    chance = pd.to_numeric(
        h["chance_of_playing_this_round"],
        errors="coerce",
    )

    h["Historically Available"] = (
        (h["status"] == "a")
        | (chance > 0)
    )

    h["starts"] = pd.to_numeric(
        h["starts"],
        errors="coerce",
    ).fillna(0)

    h["minutes"] = pd.to_numeric(
        h["minutes"],
        errors="coerce",
    ).fillna(0)

    # Approx mins conditional on starting.
    starter_rows = h[
        (h["starts"] > 0)
        & (h["Historically Available"])
    ].copy()

    starter_rows["Mins Per Start"] = (
        starter_rows["minutes"]
        / starter_rows["starts"]
    ).clip(upper=90)

    player_start = (
        starter_rows.groupby("player_code")
        .agg(
            Player_Start_Mins=("Mins Per Start", "mean"),
            Start_Sample=("Mins Per Start", "size"),
        )
        .reset_index()
    )

    # This is exactly E[minutes | available, did not start].
    # Zero-minute unused-sub games remain in the average.
    bench_rows = h[
        (h["starts"] == 0)
        & (h["Historically Available"])
    ]

    player_bench = (
        bench_rows.groupby("player_code")
        .agg(
            Player_Bench_Exp_Mins=("minutes", "mean"),
            Bench_Sample=("minutes", "size"),
        )
        .reset_index()
    )

    # Position fallback priors.
    pos_start = (
        starter_rows.groupby("FPL Pos")["Mins Per Start"]
        .mean()
        .to_dict()
    )

    pos_bench = (
        bench_rows.groupby("FPL Pos")["minutes"]
        .mean()
        .to_dict()
    )

    out = current_players[
        ["Player ID", "Code", "Player", "Team", "FPL Pos"]
    ].copy()

    out = out.merge(
        player_start,
        left_on="Code",
        right_on="player_code",
        how="left",
    )

    out = out.merge(
        player_bench,
        left_on="Code",
        right_on="player_code",
        how="left",
        suffixes=("", "_bench"),
    )

    # Shrink small samples toward position priors.
    K_START = 5
    K_BENCH = 8

    def shrink_start(row):
        prior = pos_start[row["FPL Pos"]]

        n = row["Start_Sample"]
        value = row["Player_Start_Mins"]

        if pd.isna(n):
            return prior

        return (
            n * value + K_START * prior
        ) / (n + K_START)

    def shrink_bench(row):
        prior = pos_bench[row["FPL Pos"]]

        n = row["Bench_Sample"]
        value = row["Player_Bench_Exp_Mins"]

        if pd.isna(n):
            return prior

        return (
            n * value + K_BENCH * prior
        ) / (n + K_BENCH)

    out["Mins If Start"] = out.apply(
        shrink_start,
        axis=1,
    )

    out["Expected Mins If Not Start"] = out.apply(
        shrink_bench,
        axis=1,
    )

    return out