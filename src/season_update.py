from __future__ import annotations

from numbers import Number

import numpy as np
import pandas as pd


def build_player_prior_exposure(
    historical_player_data: pd.DataFrame,
    current_players: pd.DataFrame,
    min_exposure: float = 2.5,
    max_exposure: float = 8.0,
    saturation_90s: float = 12.0,
) -> pd.DataFrame:
    """Convert prior-season playing time into prior confidence.

    A fixed six-match prior treats a 3000-minute established player and a
    180-minute fringe/new player as equally well known. This saturating curve
    keeps established priors sticky while allowing thin-history roles to learn
    from the new season more quickly.
    """
    if min_exposure <= 0 or max_exposure < min_exposure:
        raise ValueError("invalid prior-exposure bounds")
    if saturation_90s <= 0:
        raise ValueError("saturation_90s must be > 0")

    h = historical_player_data.copy()
    if "player_code" not in h.columns or "minutes" not in h.columns:
        raise ValueError("historical player data needs player_code and minutes")

    h["minutes"] = pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0)
    sample = (
        h.groupby("player_code", as_index=False)["minutes"]
        .sum()
        .rename(columns={"minutes": "Historical Minutes"})
    )
    sample["Historical 90s"] = sample["Historical Minutes"] / 90.0

    out = current_players[["Player ID", "Code"]].drop_duplicates("Player ID").copy()
    out = out.merge(
        sample,
        left_on="Code",
        right_on="player_code",
        how="left",
        validate="many_to_one",
    )
    n90 = out["Historical 90s"].fillna(0.0).clip(lower=0.0)
    out["Prior Exposure"] = (
        min_exposure
        + (max_exposure - min_exposure)
        * (1.0 - np.exp(-n90 / float(saturation_90s)))
    )

    return out[["Player ID", "Historical Minutes", "Historical 90s", "Prior Exposure"]]


def _role_evidence_one_event(
    event_live: pd.DataFrame,
    current_players: pd.DataFrame,
    gw: int,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Build one GW of minutes-normalised attacking-role evidence."""
    if event_live is None or event_live.empty:
        return pd.DataFrame()

    meta = current_players[["Player ID", "Team"]].drop_duplicates("Player ID")
    e = event_live.copy()
    if "Team" not in e.columns:
        e = e.merge(meta, on="Player ID", how="left", validate="one_to_one")

    for col in ["minutes", "expected_goals", "expected_assists"]:
        e[col] = pd.to_numeric(e.get(col, 0.0), errors="coerce").fillna(0.0)

    team_xg = (
        e.groupby("Team", as_index=False)["expected_goals"]
        .sum()
        .rename(columns={"expected_goals": "Event Team xG"})
    )
    e = e.merge(team_xg, on="Team", how="left", validate="many_to_one")

    exposure = (e["minutes"] / 90.0).clip(0.0, 1.0)
    valid_team_attack = e["Event Team xG"] > 1e-6
    safe_team_xg = e["Event Team xG"].where(valid_team_attack)
    safe_exposure = exposure.where(exposure > 0)

    e["Observed xG Role Share"] = (
        e["expected_goals"] / safe_team_xg / safe_exposure
    ).clip(lower=0.0, upper=max_role_share)
    e["Observed xA Role Share"] = (
        e["expected_assists"] / safe_team_xg / safe_exposure
    ).clip(lower=0.0, upper=max_role_share)

    e["Role Evidence Scale"] = (
        e["Event Team xG"] / float(reference_team_xg)
    ).clip(lower=float(min_evidence_scale), upper=float(max_evidence_scale))

    e["Raw Event Evidence Weight"] = exposure * e["Role Evidence Scale"]
    # A zero-xG team performance contains no information about attacking share.
    # Previously the denominator still gained weight while the numerator was
    # NaN/zero, which silently dragged every player's posterior toward zero.
    e.loc[(exposure <= 0) | (~valid_team_attack), "Raw Event Evidence Weight"] = 0.0
    e["Event Evidence Weight"] = e["Raw Event Evidence Weight"]
    e["Season GW"] = int(gw)

    return e[[
        "Player ID", "Team", "Season GW", "minutes", "Event Team xG",
        "Observed xG Role Share", "Observed xA Role Share",
        "Role Evidence Scale", "Raw Event Evidence Weight",
        "Event Evidence Weight",
    ]].copy()


def build_role_evidence(
    events,
    current_players: pd.DataFrame,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
    recency_decay: float = 0.90,
) -> pd.DataFrame:
    """Combine current-season role evidence with modest recency weighting."""
    if not (0 < recency_decay <= 1.0):
        raise ValueError("recency_decay must be in (0, 1]")

    pieces = []
    for gw, event_live in events:
        piece = _role_evidence_one_event(
            event_live,
            current_players,
            gw=gw,
            reference_team_xg=reference_team_xg,
            min_evidence_scale=min_evidence_scale,
            max_evidence_scale=max_evidence_scale,
            max_role_share=max_role_share,
        )
        if not piece.empty:
            pieces.append(piece)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True)
    latest_gw = int(out["Season GW"].max())
    out["Recency Weight"] = recency_decay ** (latest_gw - out["Season GW"])
    out["Event Evidence Weight"] = (
        out["Raw Event Evidence Weight"] * out["Recency Weight"]
    )
    return out


def _resolve_prior_exposure(
    players: pd.DataFrame,
    prior_exposure,
    fallback: float = 6.0,
) -> pd.Series:
    if isinstance(prior_exposure, Number):
        if float(prior_exposure) <= 0:
            raise ValueError("prior_exposure must be > 0")
        return pd.Series(float(prior_exposure), index=players.index)

    if isinstance(prior_exposure, pd.DataFrame):
        required = {"Player ID", "Prior Exposure"}
        missing = required - set(prior_exposure.columns)
        if missing:
            raise ValueError(f"prior exposure table missing: {sorted(missing)}")
        lookup = prior_exposure.drop_duplicates("Player ID").set_index("Player ID")["Prior Exposure"]
        values = players["Player ID"].map(lookup)
    elif isinstance(prior_exposure, pd.Series):
        values = players["Player ID"].map(prior_exposure)
    elif isinstance(prior_exposure, dict):
        values = players["Player ID"].map(prior_exposure)
    else:
        raise TypeError("prior_exposure must be a number, mapping, Series or DataFrame")

    values = pd.to_numeric(values, errors="coerce").fillna(float(fallback))
    if (values <= 0).any():
        raise ValueError("resolved prior exposure must be > 0")
    return values


def update_attack_priors_from_events(
    attack_priors: pd.DataFrame,
    events,
    current_players: pd.DataFrame,
    prior_exposure=6.0,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
    recency_decay: float = 0.90,
) -> pd.DataFrame:
    """Empirical-Bayes update of player xG/xA role shares.

    The prior can now have player-specific effective exposure, and current
    season observations receive a modest recency decay. Evidence remains
    opportunity-weighted, so a low-xG game cannot dominate a player's role.
    """
    out = attack_priors.copy()
    out["xG Share Prior Before"] = pd.to_numeric(
        out["Current xG Share Prior"], errors="coerce"
    )
    out["xA Share Prior Before"] = pd.to_numeric(
        out["Current xA Share Prior"], errors="coerce"
    )
    out["Prior Exposure Used"] = _resolve_prior_exposure(out, prior_exposure).to_numpy()

    evidence = build_role_evidence(
        events,
        current_players,
        reference_team_xg=reference_team_xg,
        min_evidence_scale=min_evidence_scale,
        max_evidence_scale=max_evidence_scale,
        max_role_share=max_role_share,
        recency_decay=recency_decay,
    )

    if evidence.empty:
        return out

    e = evidence.copy()
    e["xG Weighted"] = e["Observed xG Role Share"] * e["Event Evidence Weight"]
    e["xA Weighted"] = e["Observed xA Role Share"] * e["Event Evidence Weight"]

    agg = (
        e.groupby("Player ID", as_index=False)
        .agg(
            Current_Season_Evidence_Weight=("Event Evidence Weight", "sum"),
            Current_Season_xG_Weighted=("xG Weighted", "sum"),
            Current_Season_xA_Weighted=("xA Weighted", "sum"),
            Current_Season_Matches=("Season GW", "nunique"),
            Current_Season_Minutes=("minutes", "sum"),
        )
    )
    denom = agg["Current_Season_Evidence_Weight"].replace(0, np.nan)
    agg["Observed xG Role Share"] = agg["Current_Season_xG_Weighted"] / denom
    agg["Observed xA Role Share"] = agg["Current_Season_xA_Weighted"] / denom

    out = out.merge(agg, on="Player ID", how="left", validate="one_to_one")
    w = out["Current_Season_Evidence_Weight"].fillna(0.0)
    prior_w = out["Prior Exposure Used"].astype(float)

    for prior_col, weighted_col in [
        ("Current xG Share Prior", "Current_Season_xG_Weighted"),
        ("Current xA Share Prior", "Current_Season_xA_Weighted"),
    ]:
        prior = pd.to_numeric(out[prior_col], errors="coerce")
        weighted = pd.to_numeric(out[weighted_col], errors="coerce").fillna(0.0)
        valid = prior.notna() & (w > 0)
        updated = prior.copy()
        updated.loc[valid] = (
            prior_w.loc[valid] * prior.loc[valid] + weighted.loc[valid]
        ) / (prior_w.loc[valid] + w.loc[valid])
        out[prior_col] = updated

    out["xG Share Posterior Shift"] = (
        out["Current xG Share Prior"] - out["xG Share Prior Before"]
    )
    out["xA Share Posterior Shift"] = (
        out["Current xA Share Prior"] - out["xA Share Prior Before"]
    )
    return out


def update_external_attack_priors_from_events(
    external_priors: pd.DataFrame,
    events,
    current_players: pd.DataFrame,
    prior_exposure=6.0,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
    recency_decay: float = 0.90,
) -> pd.DataFrame:
    """Apply the same current-season role update to external priors."""
    if external_priors is None or external_priors.empty:
        return external_priors.copy() if external_priors is not None else pd.DataFrame()

    work = external_priors.copy()
    if "Player ID" not in work.columns:
        code_to_id = current_players[["Code", "Player ID"]].drop_duplicates("Code")
        work = work.merge(code_to_id, on="Code", how="left", validate="many_to_one")

    pseudo = work[["Player ID"]].copy()
    pseudo["Current xG Share Prior"] = pd.to_numeric(work["External xG Share"], errors="coerce")
    pseudo["Current xA Share Prior"] = pd.to_numeric(work["External xA Share"], errors="coerce")

    updated = update_attack_priors_from_events(
        pseudo,
        events,
        current_players,
        prior_exposure=prior_exposure,
        reference_team_xg=reference_team_xg,
        min_evidence_scale=min_evidence_scale,
        max_evidence_scale=max_evidence_scale,
        max_role_share=max_role_share,
        recency_decay=recency_decay,
    )

    keep_cols = [
        "Player ID", "Current xG Share Prior", "Current xA Share Prior",
        "Current_Season_Evidence_Weight", "Current_Season_Matches",
        "Current_Season_Minutes", "Prior Exposure Used",
    ]
    keep_cols = [c for c in keep_cols if c in updated.columns]
    keep = updated[keep_cols].copy().rename(columns={
        "Current xG Share Prior": "External xG Share Updated",
        "Current xA Share Prior": "External xA Share Updated",
        "Current_Season_Evidence_Weight": "External Current Season Evidence Weight",
        "Prior Exposure Used": "External Prior Exposure Used",
    })

    work = work.merge(keep, on="Player ID", how="left", validate="many_to_one")
    work["External xG Share"] = work["External xG Share Updated"].combine_first(work["External xG Share"])
    work["External xA Share"] = work["External xA Share Updated"].combine_first(work["External xA Share"])
    return work.drop(columns=["External xG Share Updated", "External xA Share Updated"])


def update_team_strengths_from_events(
    team_hist: pd.DataFrame,
    team_event_actuals: pd.DataFrame,
    prior_matches: float = 10.0,
    recency_decay: float = 0.95,
) -> pd.DataFrame:
    """Blend current-season xG/xGA into historical team strengths.

    Current-season matches are mildly recency weighted. With the default 0.95
    the effect is intentionally small; it allows genuine team-strength changes
    to emerge without letting a couple of volatile matches erase the prior.
    """
    if prior_matches <= 0:
        raise ValueError("prior_matches must be > 0")
    if not (0 < recency_decay <= 1.0):
        raise ValueError("recency_decay must be in (0, 1]")

    out = team_hist.copy()
    if team_event_actuals is None or team_event_actuals.empty:
        return out

    e = team_event_actuals.copy()
    for col in ["Event xG", "Event xGA"]:
        e[col] = pd.to_numeric(e[col], errors="coerce")

    if "Season_GW" in e.columns and e["Season_GW"].notna().any():
        latest = int(pd.to_numeric(e["Season_GW"], errors="coerce").max())
        gw_num = pd.to_numeric(e["Season_GW"], errors="coerce").fillna(latest)
        e["Team Recency Weight"] = recency_decay ** (latest - gw_num)
    else:
        e["Team Recency Weight"] = 1.0

    e["Weighted xG"] = e["Event xG"] * e["Team Recency Weight"]
    e["Weighted xGA"] = e["Event xGA"] * e["Team Recency Weight"]

    agg = e.groupby("Team", as_index=False).agg(
        Current_xG_Sum=("Weighted xG", "sum"),
        Current_xGA_Sum=("Weighted xGA", "sum"),
        Current_Matches=("Team Recency Weight", "sum"),
    )

    out = out.merge(agg, on="Team", how="left", validate="one_to_one")
    n = out["Current_Matches"].fillna(0.0)

    for prior_col, sum_col in [
        ("xG / Match", "Current_xG_Sum"),
        ("xGA / Match", "Current_xGA_Sum"),
    ]:
        prior = pd.to_numeric(out[prior_col], errors="coerce")
        current_sum = pd.to_numeric(out[sum_col], errors="coerce").fillna(0.0)
        mask = prior.notna() & (n > 0)
        out.loc[mask, prior_col] = (
            prior_matches * prior.loc[mask] + current_sum.loc[mask]
        ) / (prior_matches + n.loc[mask])

    return out.drop(columns=["Current_xG_Sum", "Current_xGA_Sum", "Current_Matches"])
