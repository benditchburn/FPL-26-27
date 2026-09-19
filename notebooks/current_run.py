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
    build_player_prior_exposure,
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

attack_prior_exposure = build_player_prior_exposure(
    hist_minutes_base,
    current_players,
    min_exposure=2.5,
    max_exposure=8.0,
    saturation_90s=12.0,
)

attack_priors_updated = update_attack_priors_from_events(
    model["Attack_Priors"],
    SEASON_EVENTS,
    current_players,
    prior_exposure=attack_prior_exposure,
    reference_team_xg=1.5,
    min_evidence_scale=0.35,
    max_evidence_scale=2.0,
    recency_decay=0.90,
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
    recency_decay=0.95,
)

print("Completed GWs folded into priors:", LAST_FINISHED_GW)
print("Historical + current rows:", len(hist_minutes))

role_diag = attack_priors_updated.merge(
    current_players[["Player ID", "Player", "Team", "FPL Pos"]],
    on="Player ID",
    how="left",
    validate="one_to_one",
)

if "Current_Season_Evidence_Weight" in role_diag.columns:
    role_diag["Role Shift Magnitude"] = (
        role_diag["xG Share Posterior Shift"].abs().fillna(0.0)
        + role_diag["xA Share Posterior Shift"].abs().fillna(0.0)
    )
    role_diag = role_diag[
        role_diag["Current_Season_Evidence_Weight"].fillna(0.0) > 0
    ].sort_values("Role Shift Magnitude", ascending=False)

    if not role_diag.empty:
        print("\nLargest current-season attacking-role updates")
        display(
            role_diag[
                [
                    "Player",
                    "Team",
                    "FPL Pos",
                    "Prior Exposure Used",
                    "Current_Season_Evidence_Weight",
                    "Current_Season_Minutes",
                    "xG Share Prior Before",
                    "Current xG Share Prior",
                    "xG Share Posterior Shift",
                    "xA Share Prior Before",
                    "Current xA Share Prior",
                    "xA Share Posterior Shift",
                ]
            ].head(15)
        )


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
    recency_decay=0.90,
)

print("External-prior matches:", len(external_priors))

if not external_priors.empty:
    weakest_external = external_priors.sort_values(
        ["External Match Score", "External Match Gap"],
        ascending=[True, True],
    ).head(10)

    print("\nWeakest accepted external identity matches")
    display(
        weakest_external[
            [
                "Player",
                "Full Name",
                "External Player",
                "External Team",
                "External League",
                "External Match Score",
                "External Full Name Score",
                "External Match Gap",
            ]
        ]
    )


# %%
# -----------------------------
# FIXTURE + PLAYER xPTS HORIZON
# -----------------------------

from src.fixture_projection import project_team_fixtures
from src.attack_projection import build_attack_horizon
from src.xpts import build_xpts
from src.market_inputs import load_market_overrides, combine_market_odds

market_override_path = DATA / "market_overrides.csv"
market_overrides = load_market_overrides(market_override_path)
market_odds = combine_market_odds(
    model["Market_Odds"],
    market_overrides,
)

if not market_overrides.empty:
    print(
        "Current market overrides loaded:",
        len(market_overrides),
        "fixtures from",
        market_override_path.name,
    )
else:
    print(
        "No current market override file; future fixtures will use the "
        "strength model unless the workbook already contains market xG."
    )

fixture_horizon, fixture_fit = project_team_fixtures(
    model["Fixtures"],
    team_hist,
    market_odds,
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

gk_attack_warning = xpts_horizon[
    (xpts_horizon["FPL Pos"] == "GK")
    & (
        (pd.to_numeric(xpts_horizon["xG"], errors="coerce") > 0.01)
        | (pd.to_numeric(xpts_horizon["xA"], errors="coerce") > 0.02)
    )
]

if not gk_attack_warning.empty:
    print("\nWARNING: unusually high goalkeeper attacking projection")
    display(
        gk_attack_warning[
            ["GW", "Player", "Team", "xG", "xA", "Attack Prior Source"]
        ]
        .drop_duplicates()
        .sort_values(["GW", "xG"], ascending=[True, False])
    )

# Save the current planning-GW projection before the deadline so future runs can
# evaluate what the model actually believed at the time, rather than backfilling
# history with hindsight.
from src.evaluation import (
    save_projection_snapshot,
    evaluate_saved_predictions,
)

prediction_dir = DATA / "predictions"
snapshot_path = save_projection_snapshot(
    xpts_horizon,
    prediction_dir=prediction_dir,
    start_gw=START_GW,
)
print("Saved projection snapshot:", snapshot_path.relative_to(ROOT))

backtest = evaluate_saved_predictions(
    prediction_dir,
    SEASON_EVENTS,
)

if not backtest.empty:
    print("\nHistorical projection evaluation")
    display(backtest)

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
from src.decision_report import build_current_action_report, build_decision_note

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
if SELL_PRICES is None:
    print("WARNING: optimiser is using current market price as selling price for owned players.")

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
print("MODEL EDGE:", build_decision_note(transfer_result))

print("\nCurrent-GW decision comparison")
display(
    transfer_result["comparison"][
        [
            "Scenario",
            "Decision Utility",
            "Projected xPts Utility",
            "First GW Transfers",
            "First GW Hits",
            "First GW Out",
            "First GW In",
            "Net vs Roll",
            "Raw xPts vs Roll",
            "Marginal Decision vs Fewer Transfers",
            "Marginal Raw xPts vs Fewer Transfers",
        ]
    ]
)

action_report = build_current_action_report(
    transfer_result,
    xpts_horizon,
    current_players,
    start_gw=START_GW,
    max_gw=MAX_GW,
    gw_decay=0.90,
)

if not action_report.empty:
    print("\nWhy the optimiser likes the current-GW move")
    display(action_report)

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

print("\nFixture-model diagnostics")
print(
    "Market training-fit RMSE:",
    round(float(fixture_fit["rmse"]), 3),
    "| MAE:",
    round(float(fixture_fit["mae"]), 3),
)
print(
    "Leave-one-fixture-out market RMSE:",
    (
        round(float(fixture_fit["cv_rmse"]), 3)
        if pd.notna(fixture_fit["cv_rmse"])
        else "n/a"
    ),
    "| MAE:",
    (
        round(float(fixture_fit["cv_mae"]), 3)
        if pd.notna(fixture_fit["cv_mae"])
        else "n/a"
    ),
)
print(
    "Market calibration sample:",
    fixture_fit["market_fixtures"],
    "fixtures /",
    fixture_fit["market_rows"],
    "team rows",
)
print(
    "Projected fixture xG sources:",
    fixture_horizon["xG Source"].value_counts().to_dict(),
)

market_projected = fixture_horizon[
    fixture_horizon["Market Team xG"].notna()
].copy()

if not market_projected.empty:
    print(
        "Mean |market - strength| xG:",
        round(
            float(
                market_projected["Market vs Strength xG"].abs().mean()
            ),
            3,
        ),
    )
