# %%
"""
Reusable weekly FPL model run.

Open this file in VS Code and run it cell-by-cell with the Python/Jupyter
extension. The planning GW is discovered from the official FPL API, completed
GWs are folded into the priors cumulatively, and missing lineup sources no
longer kill the whole run.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display


# %%
# -----------------------------
# USER SETTINGS — VERIFY THESE
# -----------------------------

CURRENT_SQUAD = [
    "Raya", "Verbruggen",
    "Gabriel", "Diomande", "Tarkowski", "Virgil", "Thiaw",
    "B.Fernandes", "Rice", "Mbeumo", "Anderson", "Ndiaye",
    "Thiago", "Emersonn", "Simms",
]

BANK = 0.1
FREE_TRANSFERS = 2

# Use the real FPL selling prices here if they differ from current market price.
# Example: {"Mbeumo": 8.1, "Gabriel": 6.0}
SELL_PRICES = None

HORIZON_WEEKS = 4
ALLOW_HITS = False


# %%
# -----------------------------
# PATHS + LIVE GAMEWEEK STATE
# -----------------------------

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

DATA = ROOT / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_model
from src.fpl_api import (
    fetch_current_players,
    fetch_event_live,
    get_gameweek_state,
)

state = get_gameweek_state()
START_GW = int(state["planning_gw"])
LAST_FINISHED_GW = int(state["last_finished_gw"])
MAX_GW = min(38, START_GW + HORIZON_WEEKS - 1)

model_file = next(
    p for p in DATA.glob("*v0.5*FIXED*.xlsx")
    if not p.name.startswith("~$")
)

model = load_model(model_file)
current_players = fetch_current_players()
team_hist_base = pd.read_csv(DATA / "team_strength_25_26.csv")

print("Gameweek state:", state)
print("Projection horizon:", START_GW, "to", MAX_GW)
print("Model file:", model_file.name)
print("Live players:", current_players.shape)


# %%
# -----------------------------
# CURRENT PREDICTED LINEUPS
# -----------------------------

from src.lineup_sources import (
    fetch_lineup_sources,
    parse_ffscout,
    parse_rotowire,
    parse_nma,
)
from src.lineup_consensus import match_source_to_players, build_consensus
from src.start_probs import build_start_probs


def _empty_lineup_frame():
    return pd.DataFrame(
        columns=[
            "Source",
            "Team",
            "Player Raw",
            "Predicted Starter",
        ]
    )


raw_lineups = fetch_lineup_sources(
    ROOT,
    gameweek=START_GW,
    strict=False,
)

ffs = (
    parse_ffscout(raw_lineups["ffscout"])
    if "ffscout" in raw_lineups
    else _empty_lineup_frame()
)
rw = (
    parse_rotowire(raw_lineups["rotowire"])
    if "rotowire" in raw_lineups
    else _empty_lineup_frame()
)
nma = (
    parse_nma(raw_lineups["nma"])
    if "nma" in raw_lineups
    else _empty_lineup_frame()
)

ffs_match = match_source_to_players(ffs, current_players)
rw_match = match_source_to_players(rw, current_players)
nma_match = match_source_to_players(nma, current_players)

consensus = build_consensus(
    current_players,
    ffs_match,
    rw_match,
    nma_match,
)
start_probs = build_start_probs(consensus)

print("Lineup sources fetched:", sorted(raw_lineups))
print(
    "Max team start-prob error:",
    start_probs.groupby("Team")["Start Prob"].sum().sub(11).abs().max(),
)


# %%
# -----------------------------
# COMPLETED 2026/27 EVENTS
# -----------------------------

from src.historical_minutes import (
    fetch_historical_minute_data,
    build_minute_priors,
)
from src.current_season import (
    append_event_to_history,
    build_team_event_actuals,
)
from src.season_update import (
    update_attack_priors_from_events,
    update_external_attack_priors_from_events,
    update_team_strengths_from_events,
)

hist_minutes_base = fetch_historical_minute_data(DATA / "cache")

SEASON_EVENTS = []
for gw in range(1, LAST_FINISHED_GW + 1):
    event = fetch_event_live(gw, current_players=current_players)
    SEASON_EVENTS.append((gw, event))

hist_minutes = hist_minutes_base.copy()
for gw, event in SEASON_EVENTS:
    hist_minutes = append_event_to_history(
        hist_minutes,
        event,
        current_players,
        gw=gw,
    )

attack_priors_updated = update_attack_priors_from_events(
    model["Attack_Priors"],
    SEASON_EVENTS,
    current_players,
    prior_exposure=6.0,
    reference_team_xg=1.5,
    min_evidence_scale=0.35,
    max_evidence_scale=2.0,
)

team_event_parts = []
for gw, event in SEASON_EVENTS:
    actual = build_team_event_actuals(
        event,
        model["Fixtures"],
        gw=gw,
    )
    if not actual.empty:
        team_event_parts.append(actual.assign(Season_GW=gw))

team_events = (
    pd.concat(team_event_parts, ignore_index=True)
    if team_event_parts
    else pd.DataFrame()
)

team_hist = update_team_strengths_from_events(
    team_hist_base,
    team_events,
    prior_matches=10.0,
)

print("Completed GWs folded into priors:", LAST_FINISHED_GW)
print("Historical + current rows:", len(hist_minutes))


# %%
# -----------------------------
# MINUTES / APPEARANCE / DEFENCE
# -----------------------------

from src.minutes import build_expected_minutes
from src.appearance import build_appearance_priors
from src.horizon_minutes import build_minutes_horizon
from src.defensive_points import (
    build_defensive_priors,
    build_defensive_xpts,
)
from src.defcon_calibration import (
    build_defcon_oof,
    fit_defender_calibrator,
)
from src.bonus_points import (
    build_bonus_priors,
    build_bonus_xpts,
)

minute_priors = build_minute_priors(
    current_players,
    hist_minutes,
)
minutes = build_expected_minutes(
    start_probs,
    minute_priors,
)

appearance_priors = build_appearance_priors(
    current_players,
    hist_minutes,
)

minutes_horizon = build_minutes_horizon(
    minutes,
    appearance_priors,
    start_gw=START_GW,
    max_gw=MAX_GW,
)

appearance_horizon = minutes_horizon[
    [
        "Player ID",
        "GW",
        "Appearance Prob",
        "P60",
        "Expected Appearance Pts",
    ]
].copy()

defensive_priors = build_defensive_priors(
    current_players,
    hist_minutes,
)

bonus_priors = build_bonus_priors(
    current_players,
    hist_minutes,
)

# Keep calibration strictly out-of-fold on the historical base sample.
dc_oof = build_defcon_oof(
    hist_minutes_base,
    shrink_k=4,
)
dc_calibrator = fit_defender_calibrator(dc_oof)

print("Minutes horizon:", minutes_horizon.shape)
print(
    "Max expected-starters error:",
    minutes_horizon
    .groupby(["GW", "Team"])["Start Prob"]
    .sum()
    .sub(11)
    .abs()
    .max(),
)


# %%
# -----------------------------
# EXTERNAL ATTACK PRIORS
# -----------------------------

from src.fotmob_history import (
    fetch_external_history,
    match_external_priors,
)

leagues = [
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
]

external_history = fetch_external_history(
    DATA / "cache",
    leagues,
)

existing_priors = attack_priors_updated.merge(
    model["Players"][["Player ID", "Code"]],
    on="Player ID",
    how="left",
)

prior_by_code = existing_priors[
    [
        "Code",
        "Current xG Share Prior",
        "Current xA Share Prior",
    ]
].drop_duplicates("Code")

fallback_players = current_players.merge(
    prior_by_code,
    on="Code",
    how="left",
)

fallback_players = fallback_players[
    fallback_players["Current xG Share Prior"].isna()
].copy()

external_priors = match_external_priors(
    fallback_players,
    external_history,
)

external_priors = update_external_attack_priors_from_events(
    external_priors,
    SEASON_EVENTS,
    current_players,
    prior_exposure=6.0,
    reference_team_xg=1.5,
    min_evidence_scale=0.35,
    max_evidence_scale=2.0,
)

print("External-prior matches:", len(external_priors))


# %%
# -----------------------------
# FIXTURE + PLAYER xPTS HORIZON
# -----------------------------

from src.fixture_projection import project_team_fixtures
from src.attack_projection import build_attack_horizon
from src.xpts import build_xpts

fixture_horizon, fixture_fit = project_team_fixtures(
    model["Fixtures"],
    team_hist,
    model["Market_Odds"],
    start_gw=START_GW,
    max_gw=MAX_GW,
    ridge_lambda=2.0,
)

attack_horizon = build_attack_horizon(
    minutes=minutes_horizon,
    attack_priors=attack_priors_updated,
    workbook_players=model["Players"],
    fixture_horizon=fixture_horizon,
    external_priors=external_priors,
    start_gw=START_GW,
    max_gw=MAX_GW,
)

base_horizon = build_xpts(
    attack_horizon,
    appearance_horizon,
)

def_parts = []
bonus_parts = []

for gw in range(START_GW, MAX_GW + 1):
    gw_starts = minutes_horizon[
        minutes_horizon["GW"] == gw
    ].copy()

    gw_opp = attack_horizon.loc[
        attack_horizon["GW"] == gw,
        ["Player ID", "Opp xG"],
    ].drop_duplicates("Player ID")

    gw_starts = gw_starts.merge(
        gw_opp,
        on="Player ID",
        how="left",
        validate="one_to_one",
    )

    d = build_defensive_xpts(
        gw_starts,
        defensive_priors,
        defcon_calibrator=dc_calibrator,
    )
    d["GW"] = gw
    def_parts.append(d)

    b = build_bonus_xpts(
        gw_starts,
        bonus_priors,
    )
    b["GW"] = gw
    bonus_parts.append(b)

defensive_horizon = pd.concat(
    def_parts,
    ignore_index=True,
)

bonus_horizon = pd.concat(
    bonus_parts,
    ignore_index=True,
)

xpts_horizon = base_horizon.merge(
    defensive_horizon[
        [
            "Player ID",
            "GW",
            "xPts DefCon",
            "xPts Saves",
            "Projected Saves/90",
            "Projected Saves | Start",
            "Save Model",
        ]
    ],
    on=["Player ID", "GW"],
    how="left",
    validate="one_to_one",
)

xpts_horizon = xpts_horizon.merge(
    bonus_horizon[
        [
            "Player ID",
            "GW",
            "xPts Bonus",
        ]
    ],
    on=["Player ID", "GW"],
    how="left",
    validate="one_to_one",
)

extra_cols = [
    "xPts DefCon",
    "xPts Saves",
    "xPts Bonus",
]

xpts_horizon[extra_cols] = (
    xpts_horizon[extra_cols].fillna(0.0)
)

xpts_horizon["xPts Model"] = (
    xpts_horizon["Base xPts"]
    + xpts_horizon["xPts DefCon"]
    + xpts_horizon["xPts Saves"]
    + xpts_horizon["xPts Bonus"]
)

print("xPts horizon:", xpts_horizon.shape)
print("Gameweeks:", sorted(xpts_horizon["GW"].unique().tolist()))

display(
    xpts_horizon.loc[
        xpts_horizon["GW"] == START_GW,
        [
            "Player",
            "Team",
            "FPL Pos",
            "Effective Mins",
            "xG",
            "xA",
            "Base xPts",
            "xPts DefCon",
            "xPts Saves",
            "xPts Bonus",
            "xPts Model",
        ],
    ]
    .sort_values("xPts Model", ascending=False)
    .head(40)
)


# %%
# -----------------------------
# SQUAD + TRANSFER OPTIMISER
# -----------------------------

from src.transfer_optimizer import recommend_transfer

missing_names = [
    name
    for name in CURRENT_SQUAD
    if not current_players["Player"]
    .astype(str)
    .str.casefold()
    .eq(name.casefold())
    .any()
]

if missing_names:
    raise ValueError(
        "Current squad names not found in live FPL API: "
        + str(missing_names)
    )

print("BANK:", BANK)
print("FREE_TRANSFERS:", FREE_TRANSFERS)
print("SELL_PRICES supplied:", SELL_PRICES is not None)

transfer_result = recommend_transfer(
    xpts_horizon=xpts_horizon,
    current_players=current_players,
    current_squad=CURRENT_SQUAD,
    bank=BANK,
    free_transfers=FREE_TRANSFERS,
    sell_prices=SELL_PRICES,
    start_gw=START_GW,
    max_gw=MAX_GW,
    gw_decay=0.90,
    bench_weight=0.12,
    max_transfers_per_gw=2,
    allow_hits=ALLOW_HITS,
    time_limit=60.0,
)

print("\nRECOMMENDATION:", transfer_result["recommendation"])
print("\nCurrent-GW decision comparison")
display(transfer_result["comparison"])

print("\nOptimal path")
display(
    transfer_result["optimal"]["plan"][
        [
            "GW",
            "FT Entering",
            "Transfers",
            "Out",
            "In",
            "Bank After",
            "FT Next",
            "XI xPts",
            "Captain",
            "Captain xPts",
            "Hit Cost",
        ]
    ]
)
