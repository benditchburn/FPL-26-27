import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


def normalise(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-zA-Z ]", " ", name).lower()
    return " ".join(name.split())

TEAM_ALIASES = {
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Spurs",
    "Leeds United": "Leeds",
    "AFC Bournemouth": "Bournemouth",
}

PLAYER_ALIASES = {
    "Kadioglu": "Kadıoğlu",
    "G. Rutter": "Rutter",
    "Ramsey (Jacob)": "Ramsey",
}

# A predicted-XI source is supposed to provide eleven starters per club. If
# fewer than this many players can be matched, treat that club/source as
# unavailable rather than silently interpreting parser failures as bench votes.
MIN_MATCHED_STARTERS_PER_TEAM = 8

def name_score(source_name, candidate):
    a = normalise(source_name)
    b = normalise(candidate)

    if a == b:
        return 1.0

    # Handles things like "Raya Martin" -> "Raya"
    # and "R. Calafiori" -> "Calafiori"
    if len(b) >= 4 and (b in a or a in b):
        return 0.96

    return SequenceMatcher(None, a, b).ratio()


def match_source_to_players(source_df, players):
    rows = []

    for _, row in source_df.iterrows():

        team = TEAM_ALIASES.get(row["Team"], row["Team"])

        candidates = players[players["Team"] == team].copy()

        best = None
        best_score = 0

        for _, player in candidates.iterrows():

            source_name = PLAYER_ALIASES.get(
            row["Player Raw"],
            row["Player Raw"]
            )

            scores = [
                name_score(source_name, player["Player"]),
                name_score(source_name, player["Full Name"]),
            ]

            score = max(scores)

            if score > best_score:
                best_score = score
                best = player

        rows.append({
            **row.to_dict(),
            "Matched Team": team,
            "Player ID": best["Player ID"] if best_score >= 0.80 else None,
            "Matched Player": best["Player"] if best_score >= 0.80 else None,
            "Match Score": best_score,
        })

    return pd.DataFrame(rows)


def build_consensus(
    current_players,
    ffs_match,
    rw_match,
    nma_match,
):
    """Combine predicted-lineup sources without treating missing feeds as votes.

    A source that failed entirely is left as NaN and excluded from the
    denominator. If a source only contains some clubs, uncovered clubs are
    also NaN rather than being interpreted as eleven negative votes.
    """
    base = current_players[
        [
            "Player ID",
            "Code",
            "Player",
            "Team",
            "FPL Pos",
            "Status",
            "Chance Play Next",
            "News",
        ]
    ].copy()

    sources = {
        "FFScout": ffs_match,
        "RotoWire": rw_match,
        "NMA": nma_match,
    }

    for source_name, source_df in sources.items():
        base[source_name] = np.nan

        if (
            source_df is None
            or source_df.empty
            or "Player ID" not in source_df.columns
        ):
            continue

        starters = set(
            source_df.loc[
                source_df["Predicted Starter"].fillna(False)
                & source_df["Player ID"].notna(),
                "Player ID",
            ].astype(int)
        )

        if "Matched Team" in source_df.columns:
            source_team = source_df["Matched Team"].astype("string")
        elif "Team" in source_df.columns:
            source_team = source_df["Team"].map(
                lambda x: TEAM_ALIASES.get(x, x)
            ).astype("string")
        else:
            source_team = pd.Series(pd.NA, index=source_df.index, dtype="string")

        quality = (
            source_df.assign(_Team=source_team)
            .loc[source_df["Player ID"].notna()]
            .groupby("_Team")["Player ID"]
            .nunique()
        )
        covered_teams = set(
            quality[quality >= MIN_MATCHED_STARTERS_PER_TEAM].index.astype(str)
        )

        covered = base["Team"].astype(str).isin(covered_teams)
        base.loc[covered, source_name] = (
            base.loc[covered, "Player ID"].isin(starters).astype(float)
        )

    source_cols = ["FFScout", "RotoWire", "NMA"]
    base["Lineup Votes"] = base[source_cols].sum(axis=1, skipna=True)
    base["Lineup Sources Available"] = base[source_cols].notna().sum(axis=1)
    base["Lineup Consensus"] = (
        base["Lineup Votes"]
        / base["Lineup Sources Available"].replace(0, np.nan)
    )

    return base
