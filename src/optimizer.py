import numpy as np
import pandas as pd

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SQUAD_COUNTS = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}


def optimise_gw1_team(
    players: pd.DataFrame,
    budget=100.0,
):

    df = players.copy()

    df = df[
        df["GW1 xPts Model"].notna()
        & df["Current £m"].notna()
        & (df["Availability Prob"] > 0)
    ].reset_index(drop=True)

    n = len(df)

    # Variables:
    # 0:n       = squad
    # n:2n      = starting XI
    # 2n:3n     = captain

    NVAR = 3 * n

    squad = np.arange(0, n)
    start = np.arange(n, 2*n)
    captain = np.arange(2*n, 3*n)

    # ---------------------------------
    # Objective
    # ---------------------------------

    c = np.zeros(NVAR)

    # scipy minimises, hence negative
    c[start] = -df["GW1 xPts Model"].to_numpy()

    # Captain contributes one EXTRA copy
    c[captain] = -df["GW1 xPts Model"].to_numpy()

    # Tiny tie-breaker favouring useful benches
    c[squad] += -0.001 * df["GW1 xPts Model"].to_numpy()

    # ---------------------------------
    # Constraints
    # ---------------------------------

    rows = []
    lower = []
    upper = []

    def add_constraint(coeffs, lb, ub):
        rows.append(coeffs)
        lower.append(lb)
        upper.append(ub)

    # 15-man squad
    r = np.zeros(NVAR)
    r[squad] = 1
    add_constraint(r, 15, 15)

    # 11 starters
    r = np.zeros(NVAR)
    r[start] = 1
    add_constraint(r, 11, 11)

    # exactly one captain
    r = np.zeros(NVAR)
    r[captain] = 1
    add_constraint(r, 1, 1)

    # Budget
    r = np.zeros(NVAR)
    r[squad] = df["Current £m"].to_numpy()
    add_constraint(r, 0, budget)

    # Squad positional counts
    for pos, count in SQUAD_COUNTS.items():

        mask = (
            df["FPL Pos"].eq(pos)
            .astype(float)
            .to_numpy()
        )

        r = np.zeros(NVAR)
        r[squad] = mask

        add_constraint(r, count, count)

    # Starting GK = 1
    gk = df["FPL Pos"].eq("GK").astype(float).to_numpy()

    r = np.zeros(NVAR)
    r[start] = gk

    add_constraint(r, 1, 1)

    # Starting formation bounds
    formation_bounds = {
        "DEF": (3, 5),
        "MID": (2, 5),
        "FWD": (1, 3),
    }

    for pos, (lb, ub) in formation_bounds.items():

        mask = (
            df["FPL Pos"].eq(pos)
            .astype(float)
            .to_numpy()
        )

        r = np.zeros(NVAR)
        r[start] = mask

        add_constraint(r, lb, ub)

    # Max 3 players per club
    for team in df["Team"].unique():

        mask = (
            df["Team"].eq(team)
            .astype(float)
            .to_numpy()
        )

        r = np.zeros(NVAR)
        r[squad] = mask

        add_constraint(r, 0, 3)

    # starter <= squad
    for i in range(n):

        r = np.zeros(NVAR)

        r[start[i]] = 1
        r[squad[i]] = -1

        add_constraint(r, -np.inf, 0)

    # captain <= starter
    for i in range(n):

        r = np.zeros(NVAR)

        r[captain[i]] = 1
        r[start[i]] = -1

        add_constraint(r, -np.inf, 0)

    A = np.vstack(rows)

    constraints = LinearConstraint(
        A,
        np.array(lower),
        np.array(upper),
    )

    result = milp(
        c=c,
        integrality=np.ones(NVAR),
        bounds=Bounds(
            np.zeros(NVAR),
            np.ones(NVAR),
        ),
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(
            f"Optimisation failed: {result.message}"
        )

    solution = result.x

    df["In Squad"] = solution[squad] > 0.5
    df["Starter"] = solution[start] > 0.5
    df["Captain"] = solution[captain] > 0.5

    selected = df[df["In Squad"]].copy()

    selected["Projected GW1 Contribution"] = np.where(
        selected["Starter"],
        selected["GW1 xPts Model"],
        0,
    )

    selected.loc[
        selected["Captain"],
        "Projected GW1 Contribution"
    ] += selected.loc[
        selected["Captain"],
        "GW1 xPts Model"
    ]

    return selected