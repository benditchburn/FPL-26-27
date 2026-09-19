from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


SQUAD_POSITION_COUNTS = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}

XI_POSITION_BOUNDS = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}


def _prepare_players(
    xpts_horizon: pd.DataFrame,
    current_players: pd.DataFrame,
    start_gw: int,
    max_gw: int,
):
    if start_gw < 1:
        raise ValueError("start_gw must be >= 1")
    if max_gw < start_gw:
        raise ValueError("max_gw must be >= start_gw")

    gameweeks = list(range(start_gw, max_gw + 1))

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
        "Player ID" if c == "Player ID" else f"GW{int(c)}"
        for c in pts.columns
    ]

    meta = current_players[
        ["Player ID", "Player", "Team", "FPL Pos", "Current £m"]
    ].drop_duplicates("Player ID")

    players = meta.merge(
        pts,
        on="Player ID",
        how="inner",
        validate="one_to_one",
    )

    gw_cols = [f"GW{gw}" for gw in gameweeks]
    for col in gw_cols:
        if col not in players.columns:
            players[col] = 0.0

    players[gw_cols] = (
        players[gw_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    players["Current £m"] = pd.to_numeric(
        players["Current £m"],
        errors="coerce",
    )

    players = (
        players[players["Current £m"].notna()]
        .copy()
        .reset_index(drop=True)
    )
    players["Player ID"] = players["Player ID"].astype(int)

    return players, gameweeks


def _resolve_squad(
    current_squad: Iterable[int | str],
    players: pd.DataFrame,
) -> list[int]:
    ids = []
    valid_ids = set(players["Player ID"].astype(int))

    for item in current_squad:
        if isinstance(item, (int, np.integer)):
            pid = int(item)
            if pid not in valid_ids:
                raise ValueError(f"Unknown Player ID in current_squad: {pid}")
            ids.append(pid)
            continue

        matches = players.loc[
            players["Player"].astype(str).str.casefold()
            == str(item).casefold(),
            "Player ID",
        ].tolist()

        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely resolve current-squad player: {item}"
            )
        ids.append(int(matches[0]))

    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError("current_squad must contain exactly 15 unique players")

    return ids


def _normalise_sell_prices(
    sell_prices: dict | None,
    players: pd.DataFrame,
    squad_ids: list[int],
) -> dict[int, float]:
    by_id = players.set_index("Player ID")
    out = {
        pid: float(by_id.loc[pid, "Current £m"])
        for pid in squad_ids
    }

    if not sell_prices:
        return out

    name_map = {
        str(row["Player"]).casefold(): int(row["Player ID"])
        for _, row in players.iterrows()
    }

    for key, price in sell_prices.items():
        if isinstance(key, (int, np.integer)):
            pid = int(key)
        else:
            pid = name_map.get(str(key).casefold())
            if pid is None:
                raise ValueError(f"Unknown player in sell_prices: {key}")

        if pid not in squad_ids:
            raise ValueError(
                f"sell_prices supplied for non-owned player: {key}"
            )
        out[pid] = float(price)

    return out


def _validate_initial_squad(
    squad_ids: list[int],
    players: pd.DataFrame,
):
    s = players[players["Player ID"].isin(squad_ids)]

    counts = s["FPL Pos"].value_counts().to_dict()
    for pos, required in SQUAD_POSITION_COUNTS.items():
        if counts.get(pos, 0) != required:
            raise ValueError(
                f"Current squad has {counts.get(pos, 0)} {pos}; "
                f"expected {required}."
            )

    if s["Team"].value_counts().max() > 3:
        raise ValueError("Current squad violates max-three-per-club rule")


def _solve_transfer_plan(
    xpts_horizon: pd.DataFrame,
    current_players: pd.DataFrame,
    current_squad: Iterable[int | str],
    bank: float = 0.0,
    free_transfers: int = 1,
    sell_prices: dict | None = None,
    start_gw: int = 2,
    max_gw: int = 6,
    gw_decay: float = 0.90,
    bench_weight: float = 0.12,
    max_bank_fts: int = 5,
    max_transfers_per_gw: int = 2,
    allow_hits: bool = False,
    max_hits_per_gw: int = 1,
    ft_option_value: float = 0.75,
    transfer_friction: float = 0.25,
    terminal_ft_option_value: float | None = None,
    first_gw_transfer_count: int | None = None,
    time_limit: float = 60.0,
):
    """Solve the multi-GW transfer problem as one mixed-integer program."""

    if not 1 <= free_transfers <= max_bank_fts:
        raise ValueError(
            f"free_transfers must be between 1 and {max_bank_fts}"
        )
    if bank < -1e-9:
        raise ValueError("bank must be >= 0")
    if not 0 <= bench_weight <= 1:
        raise ValueError("bench_weight must be between 0 and 1")
    if ft_option_value < 0:
        raise ValueError("ft_option_value must be >= 0")
    if transfer_friction < 0:
        raise ValueError("transfer_friction must be >= 0")
    if terminal_ft_option_value is None:
        terminal_ft_option_value = ft_option_value
    if terminal_ft_option_value < 0:
        raise ValueError("terminal_ft_option_value must be >= 0")

    players, gameweeks = _prepare_players(
        xpts_horizon,
        current_players,
        start_gw,
        max_gw,
    )
    squad_ids = _resolve_squad(current_squad, players)
    _validate_initial_squad(squad_ids, players)
    owned = set(squad_ids)

    sell_map = _normalise_sell_prices(
        sell_prices,
        players,
        squad_ids,
    )

    n = len(players)
    G = len(gameweeks)
    K = max_bank_fts

    id_to_i = {
        int(pid): i
        for i, pid in enumerate(players["Player ID"])
    }
    initial = np.zeros(n)
    for pid in squad_ids:
        initial[id_to_i[pid]] = 1.0

    buy_price = players["Current £m"].to_numpy(dtype=float)
    sell_price = buy_price.copy()
    for pid, price in sell_map.items():
        sell_price[id_to_i[pid]] = float(price)

    # ---------------------------------------------------------
    # FT-state action network
    # ---------------------------------------------------------
    # action tuple = (gw_index, entering_ft, transfer_count, hit_count, next_ft)
    action_meta = []
    for g in range(G):
        for k in range(1, K + 1):
            max_t = min(max_transfers_per_gw, k)
            if allow_hits:
                max_t = min(
                    max_transfers_per_gw,
                    k + max_hits_per_gw,
                )

            for t in range(max_t + 1):
                hits = max(0, t - k)
                next_ft = min(K, max(0, k - t) + 1)
                action_meta.append((g, k, t, hits, next_ft))

    action_lookup = {
        meta: idx
        for idx, meta in enumerate(action_meta)
    }
    ACOUNT = len(action_meta)

    # ---------------------------------------------------------
    # Variable blocks
    # ---------------------------------------------------------
    # All player/GW variables are binary:
    # squad, starter, captain, transfer_in, transfer_out
    block = n * G
    squad_start = 0
    starter_start = squad_start + block
    captain_start = starter_start + block
    tin_start = captain_start + block
    tout_start = tin_start + block
    bank_start = tout_start + block       # G continuous vars
    ft_start = bank_start + G             # G*K binary vars
    action_start = ft_start + G * K       # ACOUNT binary vars
    n_vars = action_start + ACOUNT

    def pg_idx(start, i, g):
        return start + g * n + i

    def squad_idx(i, g):
        return pg_idx(squad_start, i, g)

    def starter_idx(i, g):
        return pg_idx(starter_start, i, g)

    def captain_idx(i, g):
        return pg_idx(captain_start, i, g)

    def tin_idx(i, g):
        return pg_idx(tin_start, i, g)

    def tout_idx(i, g):
        return pg_idx(tout_start, i, g)

    def bank_idx(g):
        return bank_start + g

    def ft_idx(g, k):
        return ft_start + g * K + (k - 1)

    def action_idx(g, k, t, hits, next_ft):
        return action_start + action_lookup[(g, k, t, hits, next_ft)]

    # ---------------------------------------------------------
    # Objective
    # ---------------------------------------------------------
    c = np.zeros(n_vars)
    weights = np.array([gw_decay ** g for g in range(G)])

    for g, gw in enumerate(gameweeks):
        pts = players[f"GW{gw}"].to_numpy(dtype=float)

        # Small option value for owning a useful bench player.
        c[
            [squad_idx(i, g) for i in range(n)]
        ] = -weights[g] * bench_weight * pts

        # Starter upgrades bench-weight valuation to full valuation.
        c[
            [starter_idx(i, g) for i in range(n)]
        ] = -weights[g] * (1.0 - bench_weight) * pts

        # Captain contributes one extra copy.
        c[
            [captain_idx(i, g) for i in range(n)]
        ] = -weights[g] * pts

    # Regularise transfer churn. A deterministic projection cannot capture
    # the informational option value of waiting for team news, injuries,
    # role changes, price moves, etc. Reward carrying extra FTs into future
    # decision points and apply a small turnover penalty to marginal moves.
    # These are decision-robustness terms, not literal FPL points.
    for g in range(1, G):
        for k in range(1, K + 1):
            c[ft_idx(g, k)] += (
                -weights[g] * ft_option_value * max(0, k - 1)
            )

    terminal_weight = gw_decay ** G

    for meta in action_meta:
        g, k, t, hits, next_ft = meta
        idx = action_idx(*meta)
        c[idx] += weights[g] * 4.0 * hits
        c[idx] += weights[g] * transfer_friction * t

        # Do not force the optimiser to burn transfers just because the
        # modelling horizon ends. Preserve a continuation value for FTs
        # carried beyond the final projected GW.
        if g == G - 1:
            c[idx] += (
                -terminal_weight
                * terminal_ft_option_value
                * max(0, next_ft - 1)
            )

    # ---------------------------------------------------------
    # Constraints
    # ---------------------------------------------------------
    rows = []
    lower = []
    upper = []

    def add(entries, lb, ub):
        rows.append(entries)
        lower.append(lb)
        upper.append(ub)

    # Squad flow and no same-player in/out in one GW.
    for g in range(G):
        for i in range(n):
            entries = [
                (squad_idx(i, g), 1.0),
                (tin_idx(i, g), -1.0),
                (tout_idx(i, g), 1.0),
            ]

            if g == 0:
                rhs = initial[i]
            else:
                entries.append((squad_idx(i, g - 1), -1.0))
                rhs = 0.0

            add(entries, rhs, rhs)
            add(
                [
                    (tin_idx(i, g), 1.0),
                    (tout_idx(i, g), 1.0),
                ],
                -np.inf,
                1.0,
            )

    # Legal 15-player squad every GW.
    for g in range(G):
        add(
            [(squad_idx(i, g), 1.0) for i in range(n)],
            15,
            15,
        )

        for pos, required in SQUAD_POSITION_COUNTS.items():
            inds = players.index[players["FPL Pos"] == pos]
            add(
                [(squad_idx(i, g), 1.0) for i in inds],
                required,
                required,
            )

        for team in players["Team"].unique():
            inds = players.index[players["Team"] == team]
            add(
                [(squad_idx(i, g), 1.0) for i in inds],
                -np.inf,
                3,
            )

    # Bank recurrence. Static sell prices are current prices except for owned
    # players where explicit FPL selling prices can be supplied.
    for g in range(G):
        entries = [(bank_idx(g), 1.0)]
        entries += [
            (tout_idx(i, g), -sell_price[i])
            for i in range(n)
        ]
        entries += [
            (tin_idx(i, g), buy_price[i])
            for i in range(n)
        ]

        if g == 0:
            add(entries, bank, bank)
        else:
            entries.append((bank_idx(g - 1), -1.0))
            add(entries, 0.0, 0.0)

    # XI + captain every GW.
    for g in range(G):
        add(
            [(starter_idx(i, g), 1.0) for i in range(n)],
            11,
            11,
        )
        add(
            [(captain_idx(i, g), 1.0) for i in range(n)],
            1,
            1,
        )

        for pos, (lb, ub) in XI_POSITION_BOUNDS.items():
            inds = players.index[players["FPL Pos"] == pos]
            add(
                [(starter_idx(i, g), 1.0) for i in inds],
                lb,
                ub,
            )

        for i in range(n):
            add(
                [
                    (starter_idx(i, g), 1.0),
                    (squad_idx(i, g), -1.0),
                ],
                -np.inf,
                0,
            )
            add(
                [
                    (captain_idx(i, g), 1.0),
                    (starter_idx(i, g), -1.0),
                ],
                -np.inf,
                0,
            )

    # Exactly one FT state entering each GW; initial state is known.
    for g in range(G):
        add(
            [(ft_idx(g, k), 1.0) for k in range(1, K + 1)],
            1,
            1,
        )

    add(
        [(ft_idx(0, free_transfers), 1.0)],
        1,
        1,
    )

    # An action must be chosen conditional on the entering FT state.
    for g in range(G):
        for k in range(1, K + 1):
            matching = [
                meta
                for meta in action_meta
                if meta[0] == g and meta[1] == k
            ]
            entries = [
                (action_idx(*meta), 1.0)
                for meta in matching
            ]
            entries.append((ft_idx(g, k), -1.0))
            add(entries, 0.0, 0.0)

    # Transfer-in/out count equals action transfer count.
    for g in range(G):
        action_entries = []
        for meta in action_meta:
            if meta[0] == g:
                action_entries.append(
                    (action_idx(*meta), -float(meta[2]))
                )

        add(
            [(tin_idx(i, g), 1.0) for i in range(n)] + action_entries,
            0.0,
            0.0,
        )
        add(
            [(tout_idx(i, g), 1.0) for i in range(n)] + action_entries,
            0.0,
            0.0,
        )

    # FT state transition to the next GW.
    for g in range(G - 1):
        for next_k in range(1, K + 1):
            incoming = [
                meta
                for meta in action_meta
                if meta[0] == g and meta[4] == next_k
            ]
            entries = [
                (action_idx(*meta), -1.0)
                for meta in incoming
            ]
            entries.append((ft_idx(g + 1, next_k), 1.0))
            add(entries, 0.0, 0.0)

    # Optional counterfactual constraint for valuing the current FT decision.
    if first_gw_transfer_count is not None:
        if first_gw_transfer_count < 0:
            raise ValueError("first_gw_transfer_count must be >= 0")

        entries = []
        for meta in action_meta:
            if meta[0] == 0:
                entries.append(
                    (action_idx(*meta), float(meta[2]))
                )
        add(
            entries,
            float(first_gw_transfer_count),
            float(first_gw_transfer_count),
        )

    # ---------------------------------------------------------
    # Matrix + domains
    # ---------------------------------------------------------
    A = lil_matrix((len(rows), n_vars), dtype=float)
    for r, entries in enumerate(rows):
        for idx, value in entries:
            A[r, idx] = value

    constraints = LinearConstraint(
        A.tocsr(),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )

    integrality = np.ones(n_vars, dtype=int)
    # Bank variables are continuous.
    integrality[bank_start: bank_start + G] = 0

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[bank_start: bank_start + G] = 100.0

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraints,
        options={"time_limit": time_limit},
    )

    if not result.success:
        raise RuntimeError(
            "Transfer optimiser failed: " + result.message
        )

    x = result.x

    # ---------------------------------------------------------
    # Decode plan
    # ---------------------------------------------------------
    plan_rows = []

    for g, gw in enumerate(gameweeks):
        ft_entering = next(
            k
            for k in range(1, K + 1)
            if x[ft_idx(g, k)] > 0.5
        )

        chosen_meta = next(
            meta
            for meta in action_meta
            if meta[0] == g and x[action_idx(*meta)] > 0.5
        )
        _, _, transfer_count, hit_count, next_ft = chosen_meta

        outs = players.loc[
            [x[tout_idx(i, g)] > 0.5 for i in range(n)],
            ["Player ID", "Player", "FPL Pos", "Team"],
        ].copy()
        ins = players.loc[
            [x[tin_idx(i, g)] > 0.5 for i in range(n)],
            ["Player ID", "Player", "FPL Pos", "Team"],
        ].copy()
        starters = players.loc[
            [x[starter_idx(i, g)] > 0.5 for i in range(n)],
            ["Player ID", "Player", "FPL Pos", "Team", f"GW{gw}"],
        ].copy()
        captain = players.loc[
            [x[captain_idx(i, g)] > 0.5 for i in range(n)],
            ["Player ID", "Player"],
        ].iloc[0]

        squad = players.loc[
            [x[squad_idx(i, g)] > 0.5 for i in range(n)],
            ["Player ID", "Player", "FPL Pos", "Team", "Current £m"],
        ].copy()

        gw_pts = float(starters[f"GW{gw}"].sum())
        captain_pts = float(
            players.loc[
                players["Player ID"] == int(captain["Player ID"]),
                f"GW{gw}",
            ].iloc[0]
        )
        bench_ids = set(squad["Player ID"]) - set(starters["Player ID"])
        bench_pts = float(
            players.loc[
                players["Player ID"].isin(bench_ids),
                f"GW{gw}",
            ].sum()
        )

        plan_rows.append({
            "GW": gw,
            "FT Entering": ft_entering,
            "Transfers": transfer_count,
            "Hits": hit_count,
            "Out": outs["Player"].tolist(),
            "In": ins["Player"].tolist(),
            "Out IDs": outs["Player ID"].astype(int).tolist(),
            "In IDs": ins["Player ID"].astype(int).tolist(),
            "Bank After": float(x[bank_idx(g)]),
            "FT Next": next_ft,
            "XI xPts": gw_pts,
            "Captain": captain["Player"],
            "Captain xPts": captain_pts,
            "Bench Heuristic": bench_weight * bench_pts,
            "GW Utility Before Hit": (
                gw_pts + captain_pts + bench_weight * bench_pts
            ),
            "Hit Cost": 4.0 * hit_count,
            "Squad": squad["Player"].tolist(),
        })

    plan_df = pd.DataFrame(plan_rows)
    projected_points_utility = float(sum(
        weights[g] * (
            plan_df.iloc[g]["GW Utility Before Hit"]
            - plan_df.iloc[g]["Hit Cost"]
        )
        for g in range(G)
    ))

    return {
        "objective": -float(result.fun),
        "projected_points_utility": projected_points_utility,
        "plan": plan_df,
        "solver": result,
        "gameweeks": gameweeks,
        "gw_weights": dict(zip(gameweeks, weights)),
    }


def recommend_transfer(
    xpts_horizon: pd.DataFrame,
    current_players: pd.DataFrame,
    current_squad: Iterable[int | str],
    bank: float = 0.0,
    free_transfers: int = 1,
    sell_prices: dict | None = None,
    start_gw: int = 2,
    max_gw: int = 6,
    gw_decay: float = 0.90,
    bench_weight: float = 0.12,
    max_bank_fts: int = 5,
    max_transfers_per_gw: int = 2,
    allow_hits: bool = False,
    max_hits_per_gw: int = 1,
    ft_option_value: float = 0.75,
    transfer_friction: float = 0.25,
    terminal_ft_option_value: float | None = None,
    time_limit: float = 60.0,
):
    """Compare the optimal plan with each feasible current-GW transfer count.

    The value of a free transfer is endogenous: rolling changes the FT state in
    later GWs, and the optimiser can spend that extra flexibility when it
    becomes useful. No fixed "a transfer is worth X points" assumption is
    imposed.
    """

    common = dict(
        xpts_horizon=xpts_horizon,
        current_players=current_players,
        current_squad=current_squad,
        bank=bank,
        free_transfers=free_transfers,
        sell_prices=sell_prices,
        start_gw=start_gw,
        max_gw=max_gw,
        gw_decay=gw_decay,
        bench_weight=bench_weight,
        max_bank_fts=max_bank_fts,
        max_transfers_per_gw=max_transfers_per_gw,
        allow_hits=allow_hits,
        max_hits_per_gw=max_hits_per_gw,
        ft_option_value=ft_option_value,
        transfer_friction=transfer_friction,
        terminal_ft_option_value=terminal_ft_option_value,
        time_limit=time_limit,
    )

    optimal = _solve_transfer_plan(**common)

    max_first_gw = min(
        max_transfers_per_gw,
        free_transfers + (max_hits_per_gw if allow_hits else 0),
    )

    counterfactuals = {
        transfer_count: _solve_transfer_plan(
            **common,
            first_gw_transfer_count=transfer_count,
        )
        for transfer_count in range(max_first_gw + 1)
    }

    roll = counterfactuals[0]
    one_transfer = counterfactuals.get(1)

    first = optimal["plan"].iloc[0]
    if int(first["Transfers"]) == 0:
        recommendation = "ROLL"
    else:
        outs = ", ".join(first["Out"])
        ins = ", ".join(first["In"])
        recommendation = f"{outs} -> {ins}"

    rows = []

    for transfer_count, result in counterfactuals.items():
        p = result["plan"].iloc[0]

        if transfer_count == 0:
            scenario = "ROLL"
        elif transfer_count == 1:
            scenario = "BEST 1 TRANSFER"
        else:
            scenario = f"BEST {transfer_count} TRANSFERS"

        rows.append({
            "Scenario": scenario,
            "Decision Utility": result["objective"],
            "Projected xPts Utility": result["projected_points_utility"],
            "First GW Transfers": transfer_count,
            "First GW Hits": int(p["Hits"]),
            "First GW Out": p["Out"],
            "First GW In": p["In"],
        })

    comparison = pd.DataFrame(rows)
    roll_value = float(
        comparison.loc[
            comparison["First GW Transfers"] == 0,
            "Decision Utility",
        ].iloc[0]
    )
    comparison["Net vs Roll"] = comparison["Decision Utility"] - roll_value

    raw_roll = float(
        comparison.loc[
            comparison["First GW Transfers"] == 0,
            "Projected xPts Utility",
        ].iloc[0]
    )
    comparison["Raw xPts vs Roll"] = (
        comparison["Projected xPts Utility"] - raw_roll
    )

    # Make the value of an *additional* transfer explicit. This is often more
    # useful than merely knowing which scenario is mathematically best.
    by_count = comparison.sort_values("First GW Transfers").copy()
    by_count["Marginal Decision vs Fewer Transfers"] = (
        by_count["Decision Utility"].diff().fillna(0.0)
    )
    by_count["Marginal Raw xPts vs Fewer Transfers"] = (
        by_count["Projected xPts Utility"].diff().fillna(0.0)
    )
    comparison = comparison.merge(
        by_count[
            [
                "First GW Transfers",
                "Marginal Decision vs Fewer Transfers",
                "Marginal Raw xPts vs Fewer Transfers",
            ]
        ],
        on="First GW Transfers",
        how="left",
        validate="one_to_one",
    )

    ranked = comparison.sort_values(
        "Decision Utility",
        ascending=False,
    ).reset_index(drop=True)

    if len(ranked) >= 2:
        decision_margin = float(
            ranked.loc[0, "Decision Utility"]
            - ranked.loc[1, "Decision Utility"]
        )
    else:
        decision_margin = np.nan

    if pd.isna(decision_margin):
        decision_margin_label = "single scenario"
    elif decision_margin < 0.25:
        decision_margin_label = "near tie"
    elif decision_margin < 0.75:
        decision_margin_label = "small model edge"
    elif decision_margin < 1.50:
        decision_margin_label = "moderate model edge"
    else:
        decision_margin_label = "large model edge"

    return {
        "recommendation": recommendation,
        "optimal": optimal,
        "roll": roll,
        "one_transfer": one_transfer,
        "counterfactuals": counterfactuals,
        "comparison": ranked,
        "decision_margin_vs_runner_up": decision_margin,
        "decision_margin_label": decision_margin_label,
        "assumptions": {
            "prices_static_over_horizon": True,
            "owned_sell_prices_default_to_current_price": sell_prices is None,
            "gw_decay": gw_decay,
            "bench_weight": bench_weight,
            "allow_hits": allow_hits,
            "ft_option_value": ft_option_value,
            "transfer_friction": transfer_friction,
            "terminal_ft_option_value": (
                ft_option_value
                if terminal_ft_option_value is None
                else terminal_ft_option_value
            ),
        },
    }
