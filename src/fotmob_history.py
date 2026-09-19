import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import setuptools
import LanusStats as ls


def normalise(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z ]", " ", s).lower()
    return " ".join(s.split())


def name_score(a, b):
    a = normalise(a)
    b = normalise(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Safe case: FPL short name appears as a complete component
    # e.g. "Gonzalo" in "Gonzalo Garcia"
    if len(a) >= 4 and a in b.split():
        return 0.98

    if len(b) >= 4 and b in a.split():
        return 0.98

    # Multi-word containment
    if len(a) >= 5 and (f" {a} " in f" {b} " or f" {b} " in f" {a} "):
        return 0.96

    # Fuzzy matching only for genuinely very close names
    ratio = SequenceMatcher(None, a, b).ratio()

    return ratio if ratio >= 0.92 else 0.0

def fetch_league_history(fotmob, league, season="2025/2026"):

    xg = fotmob.get_players_stats_season(
        league, season, "expected_goals"
    )

    xa = fotmob.get_players_stats_season(
        league, season, "expected_assists"
    )

    team_xg = fotmob.get_teams_stats_season(
        league, season, "expected_goals_team"
    )

    # Original player/team name is column 3;
    # LanusStats creates a second duplicate "name" column
    xg_clean = pd.DataFrame({
        "FotMob Player ID": xg["id"],
        "FotMob Team ID": xg["teamId"],
        "Player External": xg.iloc[:, 2],
        "Hist xG External": pd.to_numeric(xg["value"]),
    })

    xa_clean = pd.DataFrame({
        "FotMob Player ID": xa["id"],
        "Hist xA External": pd.to_numeric(xa["value"]),
    })

    team_clean = pd.DataFrame({
        "FotMob Team ID": team_xg["teamId"],
        "Hist Team External": team_xg.iloc[:, 2],
        "Hist Team xG External": pd.to_numeric(team_xg["value"]),
    })

    out = (
        xg_clean
        .merge(xa_clean, on="FotMob Player ID", how="left")
        .merge(team_clean, on="FotMob Team ID", how="left")
    )

    out["External League"] = league

    out["External xG Share"] = (
        out["Hist xG External"]
        / out["Hist Team xG External"]
    )

    # Temporary existing architecture:
    # xA expressed relative to team scoring opportunity.
    out["External xA Share"] = (
        out["Hist xA External"]
        / out["Hist Team xG External"]
    )

    return out


def fetch_external_history(
    cache_dir: Path,
    leagues,
    season="2025/2026",
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "fotmob_attack_history_2025_26.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    fotmob = ls.FotMob()
    frames = []

    for league in leagues:
        try:
            print(f"Fetching {league}...")
            df = fetch_league_history(
                fotmob,
                league,
                season,
            )
            frames.append(df)

        except Exception as e:
            print(f"SKIP {league}: {e}")

    out = pd.concat(frames, ignore_index=True)

    out.to_csv(cache_file, index=False)

    return out


def match_external_priors(
    fallback_players,
    external_history,
):

    rows = []

    for _, player in fallback_players.iterrows():

        best = None
        best_score = 0

        for _, hist in external_history.iterrows():

            score = max(
                name_score(
                    player["Player"],
                    hist["Player External"],
                ),
                name_score(
                    player["Full Name"],
                    hist["Player External"],
                ),
            )

            if score > best_score:
                best_score = score
                best = hist

        if best_score >= 0.92:

            rows.append({
                "Player ID": player["Player ID"],
                "Code": player["Code"],
                "Player": player["Player"],
                "External Player": best["Player External"],
                "External League": best["External League"],
                "External Team": best["Hist Team External"],
                "External xG Share": best["External xG Share"],
                "External xA Share": best["External xA Share"],
                "External Match Score": best_score,
            })

    return pd.DataFrame(rows)