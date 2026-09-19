import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


def normalise(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z ]", " ", s).lower()
    return " ".join(s.split())


def name_score(a, b):
    a = normalise(a)
    b = normalise(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Multi-word names contained inside a longer legal/full name are strong
    # evidence. This handles middle names without reducing identity to surname.
    a_tokens = a.split()
    b_tokens = b.split()
    if min(len(a_tokens), len(b_tokens)) >= 2:
        a_set = set(a_tokens)
        b_set = set(b_tokens)
        if a_set.issubset(b_set) or b_set.issubset(a_set):
            return 0.98
        if a_tokens[0] == b_tokens[0] and a_tokens[-1] == b_tokens[-1]:
            return 0.97

    # Single-token containment is useful for display names but is deliberately
    # weaker: a surname alone must not override a contradictory full name.
    if len(a_tokens) == 1 and len(a) >= 4 and a in b_tokens:
        return 0.90
    if len(b_tokens) == 1 and len(b) >= 4 and b in a_tokens:
        return 0.90

    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio if ratio >= 0.92 else 0.0


def external_identity_score(web_name, full_name, external_name):
    """Score an external player match while protecting against surname collisions.

    If both the FPL full name and external name contain multiple tokens, the
    full identity must agree. A short web name such as 'Suzuki' is not enough
    to match 'Zion Suzuki' to an unrelated 'Yuito Suzuki'.
    """
    web = normalise(web_name)
    full = normalise(full_name)
    external = normalise(external_name)

    short_score = name_score(web, external)
    full_score = name_score(full, external)

    full_tokens = full.split()
    external_tokens = external.split()

    if len(full_tokens) >= 2 and len(external_tokens) >= 2:
        overlap = len(set(full_tokens) & set(external_tokens))

        if full_score >= 0.92:
            score = full_score
        elif overlap >= 2:
            # Same two-or-more identity tokens in different ordering / with
            # extra middle names is still strong evidence.
            score = 0.94
        else:
            score = 0.0
    else:
        score = max(short_score, full_score)

    return {
        "score": float(score),
        "full_score": float(full_score),
        "short_score": float(short_score),
    }


def fetch_league_history(fotmob, league, season="2025/2026"):

    xg = fotmob.get_players_stats_season(
        league, season, "expected_goals"
    )

    xa = fotmob.get_players_stats_season(
        league, season, "expected_assists"
    )

    team_xg = fotmob.get_teams_stats_season(
        league, season, "expected_goals_team"
    )

    # Original player/team name is column 3;
    # LanusStats creates a second duplicate "name" column
    xg_clean = pd.DataFrame({
        "FotMob Player ID": xg["id"],
        "FotMob Team ID": xg["teamId"],
        "Player External": xg.iloc[:, 2],
        "Hist xG External": pd.to_numeric(xg["value"]),
    })

    xa_clean = pd.DataFrame({
        "FotMob Player ID": xa["id"],
        "Hist xA External": pd.to_numeric(xa["value"]),
    })

    team_clean = pd.DataFrame({
        "FotMob Team ID": team_xg["teamId"],
        "Hist Team External": team_xg.iloc[:, 2],
        "Hist Team xG External": pd.to_numeric(team_xg["value"]),
    })

    out = (
        xg_clean
        .merge(xa_clean, on="FotMob Player ID", how="left")
        .merge(team_clean, on="FotMob Team ID", how="left")
    )

    out["External League"] = league

    out["External xG Share"] = (
        out["Hist xG External"]
        / out["Hist Team xG External"]
    )

    # xA expressed relative to team scoring opportunity.
    out["External xA Share"] = (
        out["Hist xA External"]
        / out["Hist Team xG External"]
    )

    return out


def fetch_external_history(
    cache_dir: Path,
    leagues,
    season="2025/2026",
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "fotmob_attack_history_2025_26.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file)

    # LanusStats is only needed when refreshing the cache. Keep it lazy so
    # diagnostics/tests that only use matching logic do not require it.
    import setuptools  # noqa: F401  # LanusStats compatibility on some envs
    import LanusStats as ls

    fotmob = ls.FotMob()
    frames = []

    for league in leagues:
        try:
            print(f"Fetching {league}...")
            df = fetch_league_history(
                fotmob,
                league,
                season,
            )
            frames.append(df)

        except Exception as e:
            print(f"SKIP {league}: {e}")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cache_file, index=False)
    return out


def match_external_priors(
    fallback_players,
    external_history,
    min_score: float = 0.92,
    ambiguity_margin: float = 0.03,
):
    """Match new/no-history FPL players to prior-season external identities.

    A candidate must pass the full-identity guard and must not be effectively
    tied with a second external player. Ambiguous matches are safer to leave
    to the position fallback than to inject another player's attacking prior.
    """
    rows = []

    for _, player in fallback_players.iterrows():
        candidates = []

        for _, hist in external_history.iterrows():
            scores = external_identity_score(
                player["Player"],
                player["Full Name"],
                hist["Player External"],
            )

            if scores["score"] <= 0:
                continue

            candidates.append((
                scores["score"],
                scores["full_score"],
                scores["short_score"],
                hist,
            ))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_full, best_short, best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        gap = best_score - second_score

        if best_score < min_score:
            continue

        if second_score >= min_score and gap < ambiguity_margin:
            continue

        rows.append({
            "Player ID": player["Player ID"],
            "Code": player["Code"],
            "Player": player["Player"],
            "Full Name": player["Full Name"],
            "External Player": best["Player External"],
            "External League": best["External League"],
            "External Team": best["Hist Team External"],
            "External xG Share": best["External xG Share"],
            "External xA Share": best["External xA Share"],
            "External Match Score": best_score,
            "External Full Name Score": best_full,
            "External Short Name Score": best_short,
            "External Match Gap": gap,
        })

    return pd.DataFrame(rows)
