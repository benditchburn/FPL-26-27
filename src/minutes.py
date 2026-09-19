import pandas as pd


def build_expected_minutes(
    start_probs: pd.DataFrame,
    minute_priors: pd.DataFrame,
) -> pd.DataFrame:

    df = start_probs.merge(
        minute_priors[
            [
                "Player ID",
                "Mins If Start",
                "Expected Mins If Not Start",
                "Start_Sample",
                "Bench_Sample",
            ]
        ],
        on="Player ID",
        how="left",
    )

    df["Model Expected Mins"] = (
        df["Availability Prob"]
        * (
            df["Conditional Start Prob"]
            * df["Mins If Start"]
            +
            (1 - df["Conditional Start Prob"])
            * df["Expected Mins If Not Start"]
        )
    )

    df["User Mins Override"] = pd.NA

    df["Effective Mins"] = (
        pd.to_numeric(
            df["User Mins Override"],
            errors="coerce",
        )
        .fillna(df["Model Expected Mins"])
    )

    return df