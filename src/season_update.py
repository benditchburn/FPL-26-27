import numpy as np
import pandas as pd


def _role_evidence_one_event(
    event_live: pd.DataFrame,
    current_players: pd.DataFrame,
    gw: int,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Build player attacking-role evidence for one realised GW."""
    if event_live is None or event_live.empty:
        return pd.DataFrame()

    meta = current_players[
        ["Player ID", "Team"]
    ].drop_duplicates("Player ID")

    e = event_live.copy()
    if "Team" not in e.columns:
        e = e.merge(
            meta,
            on="Player ID",
            how="left",
            validate="one_to_one",
        )

    for col in ["minutes", "expected_goals", "expected_assists"]:
        e[col] = pd.to_numeric(
            e.get(col, 0.0),
            errors="coerce",
        ).fillna(0.0)

    team_xg = (
        e.groupby("Team", as_index=False)["expected_goals"]
        .sum()
        .rename(columns={"expected_goals": "Event Team xG"})
    )

    e = e.merge(
        team_xg,
        on="Team",
        how="left",
        validate="many_to_one",
    )

    exposure = (e["minutes"] / 90.0).clip(0.0, 1.0)
    safe_team_xg = e["Event Team xG"].where(e["Event Team xG"] > 0)
    safe_exposure = exposure.where(exposure > 0)

    e["Observed xG Role Share"] = (
        e["expected_goals"] / safe_team_xg / safe_exposure
    ).clip(lower=0.0, upper=max_role_share)

    # Keep xA on the same "share of team scoring opportunity" basis used
    # by the existing attack engine.
    e["Observed xA Role Share"] = (
        e["expected_assists"] / safe_team_xg / safe_exposure
    ).clip(lower=0.0, upper=max_role_share)

    e["Role Evidence Scale"] = (
        e["Event Team xG"] / float(reference_team_xg)
    ).clip(
        lower=float(min_evidence_scale),
        upper=float(max_evidence_scale),
    )

    e["Event Evidence Weight"] = (
        exposure * e["Role Evidence Scale"]
    )

    e["Season GW"] = int(gw)

    return e[
        [
            "Player ID",
            "Team",
            "Season GW",
            "minutes",
            "Event Team xG",
            "Observed xG Role Share",
            "Observed xA Role Share",
            "Role Evidence Scale",
            "Event Evidence Weight",
        ]
    ].copy()


def build_role_evidence(
    events,
    current_players: pd.DataFrame,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Combine role evidence across realised GWs.

    events: iterable of (gw, event_live_df)
    """
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

    return pd.concat(pieces, ignore_index=True)


def update_attack_priors_from_events(
    attack_priors: pd.DataFrame,
    events,
    current_players: pd.DataFrame,
    prior_exposure: float = 6.0,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Cumulative Bayesian-style update using all current-season GWs at once.

    The preseason/last-season prior contributes ``prior_exposure`` effective
    matches. Each realised match contributes an opportunity-adjusted weight:
        minutes/90 * clip(team_xG/reference_team_xg, min_scale, max_scale)

    This avoids accidentally overweighting the most recent GW via repeated
    posterior-on-posterior updates.
    """
    if prior_exposure <= 0:
        raise ValueError("prior_exposure must be > 0")

    out = attack_priors.copy()

    evidence = build_role_evidence(
        events,
        current_players,
        reference_team_xg=reference_team_xg,
        min_evidence_scale=min_evidence_scale,
        max_evidence_scale=max_evidence_scale,
        max_role_share=max_role_share,
    )

    if evidence.empty:
        return out

    e = evidence.copy()
    e["xG Weighted"] = (
        e["Observed xG Role Share"] * e["Event Evidence Weight"]
    )
    e["xA Weighted"] = (
        e["Observed xA Role Share"] * e["Event Evidence Weight"]
    )

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

    agg["Observed xG Role Share"] = (
        agg["Current_Season_xG_Weighted"]
        / agg["Current_Season_Evidence_Weight"].replace(0, np.nan)
    )
    agg["Observed xA Role Share"] = (
        agg["Current_Season_xA_Weighted"]
        / agg["Current_Season_Evidence_Weight"].replace(0, np.nan)
    )
    agg["Event Evidence Weight"] = agg["Current_Season_Evidence_Weight"]

    out = out.merge(
        agg,
        on="Player ID",
        how="left",
        validate="one_to_one",
    )

    w = out["Current_Season_Evidence_Weight"].fillna(0.0)

    for prior_col, weighted_col in [
        ("Current xG Share Prior", "Current_Season_xG_Weighted"),
        ("Current xA Share Prior", "Current_Season_xA_Weighted"),
    ]:
        prior = pd.to_numeric(out[prior_col], errors="coerce")
        weighted = pd.to_numeric(out[weighted_col], errors="coerce").fillna(0.0)

        valid = prior.notna() & (w > 0)
        updated = prior.copy()

        updated.loc[valid] = (
            prior_exposure * prior.loc[valid]
            + weighted.loc[valid]
        ) / (
            prior_exposure + w.loc[valid]
        )

        out[prior_col] = updated

    return out


def update_external_attack_priors_from_events(
    external_priors: pd.DataFrame,
    events,
    current_players: pd.DataFrame,
    prior_exposure: float = 6.0,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Apply the same cumulative update to FotMob/external priors."""
    if external_priors is None or external_priors.empty:
        return external_priors.copy()

    work = external_priors.copy()

    if "Player ID" not in work.columns:
        code_to_id = current_players[
            ["Code", "Player ID"]
        ].drop_duplicates("Code")
        work = work.merge(
            code_to_id,
            on="Code",
            how="left",
            validate="many_to_one",
        )

    pseudo = work[["Player ID"]].copy()
    pseudo["Current xG Share Prior"] = pd.to_numeric(
        work["External xG Share"], errors="coerce"
    )
    pseudo["Current xA Share Prior"] = pd.to_numeric(
        work["External xA Share"], errors="coerce"
    )

    updated = update_attack_priors_from_events(
        pseudo,
        events,
        current_players,
        prior_exposure=prior_exposure,
        reference_team_xg=reference_team_xg,
        min_evidence_scale=min_evidence_scale,
        max_evidence_scale=max_evidence_scale,
        max_role_share=max_role_share,
    )

    keep = updated[
        [
            "Player ID",
            "Current xG Share Prior",
            "Current xA Share Prior",
        ]
    ].rename(
        columns={
            "Current xG Share Prior": "External xG Share Updated",
            "Current xA Share Prior": "External xA Share Updated",
        }
    )

    work = work.merge(
        keep,
        on="Player ID",
        how="left",
        validate="many_to_one",
    )

    work["External xG Share"] = (
        work["External xG Share Updated"]
        .combine_first(work["External xG Share"])
    )
    work["External xA Share"] = (
        work["External xA Share Updated"]
        .combine_first(work["External xA Share"])
    )

    return work.drop(
        columns=[
            "External xG Share Updated",
            "External xA Share Updated",
        ]
    )


def update_team_strengths_from_events(
    team_hist: pd.DataFrame,
    team_event_actuals: pd.DataFrame,
    prior_matches: float = 10.0,
) -> pd.DataFrame:
    """Blend all realised current-season xG/xGA into historical team strength.

    Formula:
        posterior = (prior_matches * prior + sum(current-season actuals))
                    / (prior_matches + n_current_matches)
    """
    if prior_matches <= 0:
        raise ValueError("prior_matches must be > 0")

    out = team_hist.copy()

    if team_event_actuals is None or team_event_actuals.empty:
        return out

    e = team_event_actuals.copy()
    for col in ["Event xG", "Event xGA"]:
        e[col] = pd.to_numeric(e[col], errors="coerce")

    agg = (
        e.groupby("Team", as_index=False)
        .agg(
            Current_xG_Sum=("Event xG", "sum"),
            Current_xGA_Sum=("Event xGA", "sum"),
            Current_Matches=("Event xG", "count"),
        )
    )

    out = out.merge(
        agg,
        on="Team",
        how="left",
        validate="one_to_one",
    )

    n = out["Current_Matches"].fillna(0.0)

    for prior_col, sum_col in [
        ("xG / Match", "Current_xG_Sum"),
        ("xGA / Match", "Current_xGA_Sum"),
    ]:
        prior = pd.to_numeric(out[prior_col], errors="coerce")
        current_sum = pd.to_numeric(
            out[sum_col], errors="coerce"
        ).fillna(0.0)

        mask = prior.notna() & (n > 0)
        out.loc[mask, prior_col] = (
            prior_matches * prior.loc[mask]
            + current_sum.loc[mask]
        ) / (
            prior_matches + n.loc[mask]
        )

    return out.drop(
        columns=[
            "Current_xG_Sum",
            "Current_xGA_Sum",
            "Current_Matches",
        ]
    )
