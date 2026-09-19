import requests
import pandas as pd


FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_EVENT_LIVE_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"


def fetch_current_players():
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    teams = {
        team["id"]: team["name"]
        for team in data["teams"]
    }

    positions = {
        pos["id"]: (
            "GK"
            if pos["singular_name_short"] == "GKP"
            else pos["singular_name_short"]
        )
        for pos in data["element_types"]
    }

    rows = []

    for p in data["elements"]:

        chance = p.get("chance_of_playing_next_round")

        if chance is not None:
            availability_prob = float(chance) / 100
        else:
            availability_prob = 1.0 if p["status"] == "a" else 0.0

        rows.append({
            "Player ID": p["id"],
            "Code": p["code"],
            "Player": p["web_name"],
            "Full Name": f'{p["first_name"]} {p["second_name"]}'.strip(),
            "Team": teams[p["team"]],
            "FPL Pos": positions[p["element_type"]],
            "Current £m": p["now_cost"] / 10,
            "Status": p["status"],
            "Status Raw": p["status"],
            "Availability Prob": availability_prob,
            "Chance Play Next": p.get("chance_of_playing_next_round"),
            "News": p.get("news", ""),
            "Ownership %": float(p["selected_by_percent"]),
        })

    return pd.DataFrame(rows)


def fetch_event_live(
    gw: int,
    current_players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fetch official FPL player-level realised stats for one gameweek.

    The output deliberately uses the historical-data column names where
    possible (minutes, starts, expected_goals, expected_assists, etc.) so
    it can be appended to the old-season history used by the prior builders.
    """

    if gw < 1:
        raise ValueError("gw must be >= 1")

    url = FPL_EVENT_LIVE_URL.format(gw=gw)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    elements = response.json().get("elements", [])

    numeric_fields = [
        "minutes",
        "starts",
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "defensive_contribution",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ]

    rows = []

    for item in elements:
        stats = item.get("stats", {}) or {}
        row = {"Player ID": item.get("id"), "GW": gw}
        for field in numeric_fields:
            row[field] = stats.get(field, 0)
        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    for field in numeric_fields:
        out[field] = pd.to_numeric(
            out[field],
            errors="coerce",
        ).fillna(0.0)

    out["Player ID"] = pd.to_numeric(
        out["Player ID"],
        errors="raise",
    ).astype(int)

    if current_players is not None:
        meta = current_players[
            [
                "Player ID",
                "Code",
                "Player",
                "Team",
                "FPL Pos",
                "Status",
                "Chance Play Next",
            ]
        ].drop_duplicates("Player ID")

        out = out.merge(
            meta,
            on="Player ID",
            how="left",
            validate="one_to_one",
        )

    return out
