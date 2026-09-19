import pandas as pd


def build_lineup_template(players: pd.DataFrame) -> pd.DataFrame:
    out = players[
        ["Player ID", "Player", "Team", "FPL Pos", "Status", "Chance Play GW1"]
    ].copy()

    out["Predicted Starter"] = False
    out["Start Prob"] = pd.NA
    out["Mins If Start"] = pd.NA
    out["Mins If Bench"] = pd.NA
    out["User Mins Override"] = pd.NA

    return out