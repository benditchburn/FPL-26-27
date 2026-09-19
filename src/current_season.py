import numpy as np
import pandas as pd


def append_event_to_history(
    hist: pd.DataFrame,
    event_live: pd.DataFrame,
    current_players: pd.DataFrame,
    gw: int,
) -> pd.DataFrame:
    """Append one realised 2026/27 event to the old player-history table.

    This is intended for the *prior builders* (minutes, appearance, bonus,
    defensive contribution). Keep the original historical table for any
    out-of-fold validation/calibration so current-season observations do not
    leak into the validation folds.
    """

    if event_live.empty:
        return hist.copy()

    meta = current_players[
        [
            "Player ID",
            "Code",
            "FPL Pos",
            "Status",
            "Chance Play Next",
        ]
    ].drop_duplicates("Player ID")

    e = event_live.copy()

    # Avoid duplicate metadata columns if fetch_event_live already merged them.
    for col in ["Code", "FPL Pos", "Status", "Chance Play Next"]:
        if col in e.columns:
            e = e.drop(columns=col)

    e = e.merge(
        meta,
        on="Player ID",
        how="left",
        validate="one_to_one",
    )

    e["player_id"] = e["Player ID"]
    e["player_code"] = e["Code"]

    # Give current-season rows a chronological index beyond the 38 historical
    # GWs. Prior builders only group by player, but this avoids accidental GW1
    # collisions in diagnostics.
    e["GW"] = 38 + int(gw)

    minutes = pd.to_numeric(
        e.get("minutes", 0),
        errors="coerce",
    ).fillna(0)

    chance = pd.to_numeric(
        e.get("Chance Play Next"),
        errors="coerce",
    )

    current_status = e.get(
        "Status",
        pd.Series("a", index=e.index),
    ).astype(str)

    # A player who appeared was necessarily available for the realised event.
    # For zero-minute players, current availability is our best conservative
    # proxy for whether the zero was a selection decision rather than injury.
    was_available = (
        (minutes > 0)
        | current_status.eq("a")
        | chance.fillna(0).gt(0)
    )

    e["status"] = np.where(was_available, "a", current_status)
    e["chance_of_playing_this_round"] = np.where(
        was_available,
        100.0,
        chance.fillna(0.0),
    )

    # Historical files use these exact lower-case names. fetch_event_live
    # already returns them, but create safe defaults for optional/new fields.
    for col in [
        "minutes",
        "starts",
        "saves",
        "bonus",
        "defensive_contribution",
        "goals_scored",
        "assists",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "total_points",
    ]:
        if col not in e.columns:
            e[col] = 0.0

    # Reindex to the historical schema. Extra current-season diagnostics are
    # intentionally not pushed into hist; event_live itself remains available.
    current_rows = pd.DataFrame(index=e.index)
    for col in hist.columns:
        if col in e.columns:
            current_rows[col] = e[col]
        else:
            current_rows[col] = np.nan

    return pd.concat(
        [hist.copy(), current_rows],
        ignore_index=True,
    )


def update_attack_priors_from_event(
    attack_priors: pd.DataFrame,
    event_live: pd.DataFrame,
    current_players: pd.DataFrame,
    prior_exposure: float = 6.0,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Update attacking-role priors using realised *share* of team attack.

    Raw player xG is deliberately not used as the direct new prior. The
    observation is converted into a minutes-normalised share of the team's
    realised attacking opportunity:

        observed xG role share
            = (player xG / team xG) / (minutes / 90)

    The evidence weight is then scaled by how much team attacking opportunity
    we actually observed:

        evidence weight
            = minutes exposure
              * clip(team xG / reference_team_xg,
                     min_evidence_scale,
                     max_evidence_scale)

    Therefore a 40% share in a 3.0-xG team performance is more informative
    about role than a 40% share in a 0.3-xG performance, while extreme games
    are capped so one match can never erase the prior.

    ``prior_exposure`` is the effective number of prior full-match observations.
    With the default 6.0, a normal 90-minute match at 1.5 team xG has weight 1
    and therefore contributes 1 / 7 of the posterior role estimate.
    """

    if prior_exposure <= 0:
        raise ValueError("prior_exposure must be > 0")
    if reference_team_xg <= 0:
        raise ValueError("reference_team_xg must be > 0")
    if min_evidence_scale <= 0:
        raise ValueError("min_evidence_scale must be > 0")
    if max_evidence_scale < min_evidence_scale:
        raise ValueError("max_evidence_scale must be >= min_evidence_scale")

    out = attack_priors.copy()

    if event_live.empty:
        return out

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
            e.get(col, 0),
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

    # Existing architecture expresses xA relative to team scoring opportunity,
    # so retain the same denominator here for consistency.
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

    # No minutes = no role evidence even if the team produced xG.
    e.loc[exposure <= 0, "Event Evidence Weight"] = 0.0

    evidence = e[
        [
            "Player ID",
            "Event Team xG",
            "Observed xG Role Share",
            "Observed xA Role Share",
            "Role Evidence Scale",
            "Event Evidence Weight",
        ]
    ].drop_duplicates("Player ID")

    out = out.merge(
        evidence,
        on="Player ID",
        how="left",
        validate="one_to_one",
    )

    w = out["Event Evidence Weight"].fillna(0.0)

    for prior_col, obs_col in [
        ("Current xG Share Prior", "Observed xG Role Share"),
        ("Current xA Share Prior", "Observed xA Role Share"),
    ]:
        prior = pd.to_numeric(out[prior_col], errors="coerce")
        obs = pd.to_numeric(out[obs_col], errors="coerce")

        valid = prior.notna() & obs.notna() & (w > 0)
        updated = prior.copy()
        updated.loc[valid] = (
            prior_exposure * prior.loc[valid]
            + w.loc[valid] * obs.loc[valid]
        ) / (
            prior_exposure + w.loc[valid]
        )

        out[prior_col] = updated

    return out


def update_external_attack_priors_from_event(
    external_priors: pd.DataFrame,
    event_live: pd.DataFrame,
    current_players: pd.DataFrame,
    prior_exposure: float = 6.0,
    reference_team_xg: float = 1.5,
    min_evidence_scale: float = 0.35,
    max_evidence_scale: float = 2.0,
    max_role_share: float = 2.0,
) -> pd.DataFrame:
    """Apply the same GW evidence update to external/FotMob priors.

    This closes the early-season asymmetry where established PL players learned
    from GW1 but new signings/promoted players stayed frozen at their external
    2025/26 prior. The returned frame keeps the external-prior schema expected
    by ``build_attack_horizon``.
    """

    if external_priors is None or external_priors.empty:
        return external_priors.copy() if external_priors is not None else pd.DataFrame()

    required = {"Code", "External xG Share", "External xA Share"}
    missing = required - set(external_priors.columns)
    if missing:
        raise ValueError(
            "external_priors missing required columns: "
            f"{sorted(missing)}"
        )

    # Convert the external-prior frame to the same Player-ID keyed shape used by
    # the workbook updater, run the identical evidence calculation, then map the
    # posterior back to the external column names.
    
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

    updated = update_attack_priors_from_event(
        pseudo,
        event_live,
        current_players,
        prior_exposure=prior_exposure,
        reference_team_xg=reference_team_xg,
        min_evidence_scale=min_evidence_scale,
        max_evidence_scale=max_evidence_scale,
        max_role_share=max_role_share,
    )

    posterior = updated[[
        "Player ID",
        "Current xG Share Prior",
        "Current xA Share Prior",
    ]].rename(columns={
        "Current xG Share Prior": "External xG Share Updated",
        "Current xA Share Prior": "External xA Share Updated",
    })

    work = work.merge(
        posterior,
        on="Player ID",
        how="left",
        validate="many_to_one",
    )

    work["External xG Share"] = work["External xG Share Updated"].combine_first(
        pd.to_numeric(work["External xG Share"], errors="coerce")
    )
    work["External xA Share"] = work["External xA Share Updated"].combine_first(
        pd.to_numeric(work["External xA Share"], errors="coerce")
    )

    return work.drop(columns=[
        "Player ID",
        "External xG Share Updated",
        "External xA Share Updated",
    ])


def build_team_event_actuals(
    event_live: pd.DataFrame,
    fixtures: pd.DataFrame,
    gw: int,
) -> pd.DataFrame:
    """Aggregate player xG into team xG/xGA for a realised gameweek."""

    if event_live.empty:
        return pd.DataFrame(
            columns=["Team", "Opponent", "Event xG", "Event xGA"]
        )

    if "Team" not in event_live.columns:
        raise ValueError(
            "event_live must contain Team; pass current_players to "
            "fetch_event_live()."
        )

    e = event_live.copy()
    e["expected_goals"] = pd.to_numeric(
        e.get("expected_goals", 0),
        errors="coerce",
    ).fillna(0.0)

    actual = (
        e.groupby("Team", as_index=False)["expected_goals"]
        .sum()
        .rename(columns={"expected_goals": "Event xG"})
    )

    fx = fixtures.copy()
    fx["GW"] = pd.to_numeric(fx["GW"], errors="coerce")
    fx = fx[fx["GW"] == int(gw)][["Team", "Opponent"]].drop_duplicates()

    out = fx.merge(
        actual,
        on="Team",
        how="left",
        validate="one_to_one",
    )

    opp = actual.rename(
        columns={
            "Team": "Opponent",
            "Event xG": "Event xGA",
        }
    )

    out = out.merge(
        opp,
        on="Opponent",
        how="left",
        validate="one_to_one",
    )

    return out


def update_team_strengths_from_event(
    team_hist: pd.DataFrame,
    team_event_actuals: pd.DataFrame,
    prior_matches: float = 10.0,
) -> pd.DataFrame:
    """Blend one current-season match into historical team xG/xGA strength."""

    if prior_matches <= 0:
        raise ValueError("prior_matches must be > 0")

    out = team_hist.copy()

    if team_event_actuals.empty:
        return out

    upd = team_event_actuals[
        ["Team", "Event xG", "Event xGA"]
    ].drop_duplicates("Team")

    out = out.merge(
        upd,
        on="Team",
        how="left",
        validate="one_to_one",
    )

    for prior_col, actual_col in [
        ("xG / Match", "Event xG"),
        ("xGA / Match", "Event xGA"),
    ]:
        prior = pd.to_numeric(out[prior_col], errors="coerce")
        actual = pd.to_numeric(out[actual_col], errors="coerce")
        mask = prior.notna() & actual.notna()

        out.loc[mask, prior_col] = (
            prior_matches * prior.loc[mask]
            + actual.loc[mask]
        ) / (prior_matches + 1.0)

    return out.drop(columns=["Event xG", "Event xGA"])
