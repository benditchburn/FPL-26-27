import numpy as np
import pandas as pd

from src.fpl_api import get_gameweek_state
from src.lineup_consensus import build_consensus
from src.start_probs import build_start_probs
from src.decision_report import build_decision_note
from src.evaluation import evaluate_projection_snapshot


def _players(n=14):
    return pd.DataFrame(
        {
            "Player ID": range(1, n + 1),
            "Code": range(101, 101 + n),
            "Player": [f"P{i}" for i in range(1, n + 1)],
            "Full Name": [f"Player {i}" for i in range(1, n + 1)],
            "Team": ["Test FC"] * n,
            "FPL Pos": ["MID"] * n,
            "Status": ["a"] * n,
            "Chance Play Next": [np.nan] * n,
            "News": [""] * n,
        }
    )


def _source(players, count):
    rows = []
    for _, p in players.head(count).iterrows():
        rows.append(
            {
                "Source": "FFScout",
                "Team": "Test FC",
                "Matched Team": "Test FC",
                "Player Raw": p["Player"],
                "Predicted Starter": True,
                "Player ID": p["Player ID"],
                "Matched Player": p["Player"],
                "Match Score": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_gameweek_state_prefers_next_event():
    bootstrap = {
        "events": [
            {"id": 1, "finished": True, "is_current": False, "is_next": False},
            {"id": 2, "finished": False, "is_current": True, "is_next": False},
            {"id": 3, "finished": False, "is_current": False, "is_next": True},
        ]
    }

    state = get_gameweek_state(bootstrap)

    assert state["last_finished_gw"] == 1
    assert state["current_gw"] == 2
    assert state["next_gw"] == 3
    assert state["planning_gw"] == 3


def test_low_coverage_lineup_source_is_ignored():
    players = _players()
    weak = _source(players, 7)
    empty = pd.DataFrame()

    consensus = build_consensus(players, weak, empty, empty)

    assert consensus["FFScout"].isna().all()
    assert consensus["Lineup Sources Available"].eq(0).all()


def test_valid_lineup_source_drives_start_probabilities():
    players = _players()
    source = _source(players, 11)
    empty = pd.DataFrame()

    consensus = build_consensus(players, source, empty, empty)
    starts = build_start_probs(consensus)

    assert abs(starts["Start Prob"].sum() - 11.0) < 1e-8

    predicted = starts[starts["Player ID"].isin(range(1, 12))]["Start Prob"].mean()
    bench = starts[~starts["Player ID"].isin(range(1, 12))]["Start Prob"].mean()
    assert predicted > bench


def test_decision_note_exposes_small_margin():
    result = {
        "recommendation": "A -> B",
        "decision_margin_vs_runner_up": 0.10,
        "decision_margin_label": "near tie",
        "comparison": pd.DataFrame(
            [{"Raw xPts vs Roll": 1.5}]
        ),
    }

    note = build_decision_note(result)

    assert "near tie" in note
    assert "0.10" in note
    assert "+1.50" in note


def test_projection_evaluation_metrics():
    snapshot = pd.DataFrame(
        {
            "Player ID": [1, 2],
            "xPts Model": [5.0, 2.0],
            "Effective Mins": [90.0, 45.0],
            "Start Prob": [0.9, 0.4],
            "P60": [0.8, 0.2],
        }
    )
    actual = pd.DataFrame(
        {
            "Player ID": [1, 2],
            "total_points": [6.0, 1.0],
            "minutes": [90.0, 30.0],
            "starts": [1.0, 0.0],
        }
    )

    metrics = evaluate_projection_snapshot(snapshot, actual)

    assert metrics["Players"] == 2
    assert metrics["xPts MAE"] == 1.0
    assert metrics["Minutes MAE"] == 7.5
    assert 0.0 <= metrics["Start Brier"] <= 1.0
    assert 0.0 <= metrics["P60 Brier"] <= 1.0
