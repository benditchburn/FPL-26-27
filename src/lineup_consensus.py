import re
import unicodedata
from difflib import SequenceMatcher

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

        starters = set(
            source_df.loc[
                source_df["Predicted Starter"]
                & source_df["Player ID"].notna(),
                "Player ID",
            ].astype(int)
        )

        base[source_name] = (
            base["Player ID"]
            .isin(starters)
            .astype(float)
        )

    base["Lineup Votes"] = (
        base[
            [
                "FFScout",
                "RotoWire",
                "NMA",
            ]
        ].sum(axis=1)
    )

    base["Lineup Consensus"] = (
        base["Lineup Votes"] / 3
    )

    return base