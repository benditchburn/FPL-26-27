from pathlib import Path
import pandas as pd


def load_model(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)

    sheets = [
        "Players",
        "Attack_Priors",
        "Def_Priors",
        "Team_Attack",
        "Team_Defence",
        "Team_26_27",
        "Fixtures",
        "Fixture_Engine",
        "Market_Odds",
    ]

    return {
        sheet: pd.read_excel(path, sheet_name=sheet)
        for sheet in sheets
    }