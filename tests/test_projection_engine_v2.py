import numpy as np
import pandas as pd

from src.market_inputs import (
    combine_market_odds,
    derive_team_xg_from_odds,
    poisson_match_probs,
)
from src.season_update import (
    build_player_prior_exposure,
    build_role_evidence,
    update_attack_priors_from_events,
)


def test_market_odds_inversion_recovers_poisson_rates():
    home_xg = 1.70
    away_xg = 1.05
    p = poisson_match_probs(home_xg, away_xg)

    # Feed fair decimal odds back into the inverter.
    result = derive_team_xg_from_odds(
        1.0 / p["home"],
        1.0 / p["draw"],
        1.0 / p["away"],
        1.0 / p["over25"],
        1.0 / p["under25"],
    )

    assert abs(result["Market Home xG"] - home_xg) < 0.02
    assert abs(result["Market Away xG"] - away_xg) < 0.02
    assert result["Market Fit RMSE"] < 1e-3


def test_market_override_replaces_workbook_values():
    base = pd.DataFrame([
        {
            "GW": 6,
            "Fixture Seq": 1,
            "Home": "A",
            "Away": "B",
            "Market Home xG": 1.2,
            "Market Away xG": 0.9,
            "Home CS Prob": 0.40,
            "Away CS Prob": 0.30,
        }
    ])
    override = pd.DataFrame([
        {
            "GW": 6,
            "Home": "A",
            "Away": "B",
            "Market Home xG": 1.8,
            "Market Away xG": 1.1,
            "Home CS Prob": np.exp(-1.1),
            "Away CS Prob": np.exp(-1.8),
            "Market Source": "Current odds",
        }
    ])

    out = combine_market_odds(base, override)
    row = out.iloc[0]
    assert row["Market Home xG"] == 1.8
    assert row["Market Away xG"] == 1.1
    assert row["Fixture Seq"] == 1
    assert row["Market Source"] == "Current odds"


def _current_players():
    return pd.DataFrame([
        {"Player ID": 1, "Code": 101, "Team": "A"},
        {"Player ID": 2, "Code": 102, "Team": "A"},
    ])


def test_prior_exposure_increases_with_historical_minutes():
    hist = pd.DataFrame({
        "player_code": [101, 102],
        "minutes": [180.0, 2700.0],
    })
    out = build_player_prior_exposure(hist, _current_players())
    by_id = out.set_index("Player ID")["Prior Exposure"]

    assert by_id[2] > by_id[1]
    assert 2.5 <= by_id[1] <= 8.0
    assert 2.5 <= by_id[2] <= 8.0


def test_zero_team_xg_provides_no_role_evidence():
    event = pd.DataFrame([
        {
            "Player ID": 1,
            "Team": "A",
            "minutes": 90,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
        },
        {
            "Player ID": 2,
            "Team": "A",
            "minutes": 90,
            "expected_goals": 0.0,
            "expected_assists": 0.0,
        },
    ])

    evidence = build_role_evidence([(1, event)], _current_players())
    assert evidence["Event Evidence Weight"].eq(0.0).all()


def test_recent_role_evidence_gets_more_weight():
    event = pd.DataFrame([
        {
            "Player ID": 1,
            "Team": "A",
            "minutes": 90,
            "expected_goals": 0.5,
            "expected_assists": 0.1,
        },
        {
            "Player ID": 2,
            "Team": "A",
            "minutes": 90,
            "expected_goals": 1.0,
            "expected_assists": 0.2,
        },
    ])
    evidence = build_role_evidence(
        [(1, event), (2, event)],
        _current_players(),
        recency_decay=0.8,
    )
    old = evidence.loc[evidence["Season GW"] == 1, "Event Evidence Weight"].iloc[0]
    new = evidence.loc[evidence["Season GW"] == 2, "Event Evidence Weight"].iloc[0]
    assert np.isclose(old / new, 0.8)


def test_thinner_prior_moves_more_for_same_new_evidence():
    priors = pd.DataFrame({
        "Player ID": [1, 2],
        "Current xG Share Prior": [0.10, 0.10],
        "Current xA Share Prior": [0.10, 0.10],
    })
    players = _current_players()
    # Both players receive the same observed xG/xA, but different prior
    # confidence. Put them on separate teams so their role evidence is equal.
    players.loc[players["Player ID"] == 2, "Team"] = "B"
    event = pd.DataFrame([
        {"Player ID": 1, "Team": "A", "minutes": 90, "expected_goals": 0.6, "expected_assists": 0.3},
        {"Player ID": 2, "Team": "B", "minutes": 90, "expected_goals": 0.6, "expected_assists": 0.3},
    ])
    exposure = pd.DataFrame({
        "Player ID": [1, 2],
        "Prior Exposure": [2.5, 8.0],
    })

    out = update_attack_priors_from_events(
        priors,
        [(1, event)],
        players,
        prior_exposure=exposure,
        recency_decay=1.0,
    ).set_index("Player ID")

    shift_low = abs(out.loc[1, "xG Share Posterior Shift"])
    shift_high = abs(out.loc[2, "xG Share Posterior Shift"])
    assert shift_low > shift_high
