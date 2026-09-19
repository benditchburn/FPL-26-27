import numpy as np
import pandas as pd

from scipy.optimize import (
    milp,
    LinearConstraint,
    Bounds,
)
from scipy.sparse import lil_matrix


def optimize_horizon_squad(
    xpts_horizon: pd.DataFrame,
    current_players: pd.DataFrame,
    budget: float = 100.0,
    start_gw: int = 1,
    max_gw: int = 6,
    gw_decay: float = 0.90,
    bench_weight: float = 0.12,
    force_players: list = None,
):
    """
    Select one 15-player FPL squad.

    For each GW:
      - chooses optimal legal XI
      - chooses optimal captain

    Objective:
      discounted expected points
      + small bench option value.

    No transfers are modelled yet.
    """

    if start_gw < 1:
        raise ValueError("start_gw must be >= 1")
    if max_gw < start_gw:
        raise ValueError("max_gw must be >= start_gw")

    gameweeks = list(range(start_gw, max_gw + 1))

    # =========================================================
    # 1. PLAYER TABLE
    # =========================================================

    pts = (
        xpts_horizon[
            xpts_horizon["GW"].between(start_gw, max_gw)
        ]
        .pivot_table(
            index="Player ID",
            columns="GW",
            values="xPts Model",
            aggfunc="first",
        )
        .reset_index()
    )

    pts.columns = [
        "Player ID"
        if c == "Player ID"
        else f"GW{int(c)}"
        for c in pts.columns
    ]

    meta_cols = [
        "Player ID",
        "Player",
        "Team",
        "FPL Pos",
        "Current £m",
    ]

    players = (
        current_players[
            meta_cols
        ]
        .drop_duplicates("Player ID")
        .merge(
            pts,
            on="Player ID",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    gw_cols = [
        f"GW{g}"
        for g in gameweeks
    ]

    players[gw_cols] = (
        players[gw_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0.0)
    )

    players["Current £m"] = pd.to_numeric(
        players["Current £m"],
        errors="coerce",
    )

    players = (
        players[
            players["Current £m"].notna()
        ]
        .reset_index(drop=True)
    )

    n = len(players)
    G = len(gameweeks)

    # =========================================================
    # 2. VARIABLE INDEXING
    # =========================================================
    #
    # squad_i
    # starter_i,g
    # captain_i,g
    #
    # all binary
    # =========================================================

    squad_start = 0
    starter_start = n
    captain_start = n + n * G

    n_vars = n + 2 * n * G

    def squad_idx(i):
        return squad_start + i

    def starter_idx(i, g):
        return (
            starter_start
            + g * n
            + i
        )

    def captain_idx(i, g):
        return (
            captain_start
            + g * n
            + i
        )

    # =========================================================
    # 3. OBJECTIVE
    # =========================================================

    # scipy minimises, hence negative utility
    c = np.zeros(n_vars)

    weights = np.array([
        gw_decay ** g
        for g in range(G)
    ])

    for i in range(n):

        # Small value for owning a useful bench player.
        bench_utility = 0.0

        for g_idx, gw in enumerate(gameweeks):
            p = players.loc[
                i,
                f"GW{gw}",
            ]

            bench_utility += (
                weights[g_idx]
                * bench_weight
                * p
            )

            # If starting, upgrade from bench-weight
            # valuation to full valuation.
            c[
                starter_idx(i, g_idx)
            ] = -(
                weights[g_idx]
                * (1 - bench_weight)
                * p
            )

            # Captain gets one additional copy
            # of expected points.
            c[
                captain_idx(i, g_idx)
            ] = -(
                weights[g_idx]
                * p
            )

        c[squad_idx(i)] = (
            -bench_utility
        )

    # =========================================================
    # 4. CONSTRAINT BUILDER
    # =========================================================

    rows = []
    lower = []
    upper = []

    def add_constraint(
        entries,
        lb,
        ub,
    ):
        rows.append(entries)
        lower.append(lb)
        upper.append(ub)

    # -------------------------
    # Squad size = 15
    # -------------------------

    add_constraint(
        [
            (squad_idx(i), 1)
            for i in range(n)
        ],
        15,
        15,
    )

    # -------------------------
    # Budget <= £100m
    # -------------------------

    add_constraint(
        [
            (
                squad_idx(i),
                players.loc[
                    i,
                    "Current £m",
                ],
            )
            for i in range(n)
        ],
        -np.inf,
        budget,
    )

    # -------------------------
    # Exact squad positions
    # -------------------------

    squad_position_counts = {
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    for pos, required in (
        squad_position_counts.items()
    ):

        inds = players.index[
            players["FPL Pos"] == pos
        ]

        add_constraint(
            [
                (squad_idx(i), 1)
                for i in inds
            ],
            required,
            required,
        )

    # -------------------------
    # Max 3 per club
    # -------------------------

    for team in players[
        "Team"
    ].unique():

        inds = players.index[
            players["Team"] == team
        ]

        add_constraint(
            [
                (squad_idx(i), 1)
                for i in inds
            ],
            -np.inf,
            3,
        )

    # -------------------------
    # Optional forced players
    # -------------------------

    if force_players is not None:

        for player_name in force_players:

            inds = players.index[
                players["Player"] == player_name
            ]

            if len(inds) != 1:
                raise ValueError(
                    f"Could not uniquely identify "
                    f"forced player: {player_name}"
                )

            i = inds[0]

            add_constraint(
                [
                    (squad_idx(i), 1)
                ],
                1,
                1,
            )

    # =========================================================
    # 5. GW-SPECIFIC XI CONSTRAINTS
    # =========================================================

    for g in range(G):

        # XI exactly 11
        add_constraint(
            [
                (starter_idx(i, g), 1)
                for i in range(n)
            ],
            11,
            11,
        )

        # Exactly one starting GK
        gk_inds = players.index[
            players["FPL Pos"] == "GK"
        ]

        add_constraint(
            [
                (starter_idx(i, g), 1)
                for i in gk_inds
            ],
            1,
            1,
        )

        # DEF: 3 to 5
        def_inds = players.index[
            players["FPL Pos"] == "DEF"
        ]

        add_constraint(
            [
                (starter_idx(i, g), 1)
                for i in def_inds
            ],
            3,
            5,
        )

        # MID: 2 to 5
        mid_inds = players.index[
            players["FPL Pos"] == "MID"
        ]

        add_constraint(
            [
                (starter_idx(i, g), 1)
                for i in mid_inds
            ],
            2,
            5,
        )

        # FWD: 1 to 3
        fwd_inds = players.index[
            players["FPL Pos"] == "FWD"
        ]

        add_constraint(
            [
                (starter_idx(i, g), 1)
                for i in fwd_inds
            ],
            1,
            3,
        )

        # Exactly one captain
        add_constraint(
            [
                (captain_idx(i, g), 1)
                for i in range(n)
            ],
            1,
            1,
        )

        for i in range(n):

            # starter <= squad
            add_constraint(
                [
                    (starter_idx(i, g), 1),
                    (squad_idx(i), -1),
                ],
                -np.inf,
                0,
            )

            # captain <= starter
            add_constraint(
                [
                    (captain_idx(i, g), 1),
                    (starter_idx(i, g), -1),
                ],
                -np.inf,
                0,
            )

    # =========================================================
    # 6. BUILD SPARSE CONSTRAINT MATRIX
    # =========================================================

    A = lil_matrix(
        (
            len(rows),
            n_vars,
        ),
        dtype=float,
    )

    for r, entries in enumerate(rows):
        for idx, value in entries:
            A[r, idx] = value

    constraints = LinearConstraint(
        A.tocsr(),
        np.array(lower),
        np.array(upper),
    )

    # Every variable binary
    integrality = np.ones(
        n_vars,
        dtype=int,
    )

    bounds = Bounds(
        np.zeros(n_vars),
        np.ones(n_vars),
    )

    # =========================================================
    # 7. SOLVE
    # =========================================================

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": 60,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Optimiser failed: "
            + result.message
        )

    solution = result.x

    # =========================================================
    # 8. OUTPUT SQUAD
    # =========================================================

    selected = []

    for i in range(n):

        if solution[
            squad_idx(i)
        ] > 0.5:

            row = players.loc[i].copy()

            for g_idx, gw in enumerate(gameweeks):

                row[
                    f"Start GW{gw}"
                ] = (
                    solution[
                        starter_idx(i, g_idx)
                    ] > 0.5
                )

                row[
                    f"Captain GW{gw}"
                ] = (
                    solution[
                        captain_idx(i, g_idx)
                    ] > 0.5
                )

            selected.append(row)

    squad = pd.DataFrame(
        selected
    )

    squad["Horizon xPts"] = (
        squad[gw_cols]
        .sum(axis=1)
    )

    # Backwards-compatible display column used by the GW1 notebook.
    squad["6GW xPts"] = squad["Horizon xPts"]

    squad = squad.sort_values(
        [
            "FPL Pos",
            "Horizon xPts",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return {
        "squad": squad,
        "objective": -result.fun,
        "total_cost":
            squad["Current £m"].sum(),
        "solver": result,
        "gw_weights": dict(zip(gameweeks, weights)),
        "gameweeks": gameweeks,
    }