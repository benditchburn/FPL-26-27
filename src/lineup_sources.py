from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
import re

import requests
import pandas as pd
from bs4 import BeautifulSoup

SOURCES = {
    "ffscout": "https://www.fantasyfootballscout.co.uk/team-news/",
    "rotowire": "https://www.rotowire.com/soccer/lineups.php",
}

NMA_INDEX = "https://www.nevermanagealone.com/fpl"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def discover_nma_lineup_url(gameweek: int | None = None) -> str:
    """Find the latest Never Manage Alone predicted-lineups article.

    The old code hard-coded the GW1 article, which silently becomes stale.
    When ``gameweek`` is supplied we prefer links whose text mentions that GW.
    """

    response = requests.get(
        NMA_INDEX,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    candidates = []

    for a in soup.find_all("a", href=True):
        text = " ".join(a.stripped_strings).strip()
        low = text.lower()

        if "predicted lineup" not in low and "predicted line-up" not in low:
            continue

        score = 1
        if gameweek is not None:
            patterns = [
                rf"gameweek\s*{int(gameweek)}\b",
                rf"\bgw\s*{int(gameweek)}\b",
            ]
            if any(re.search(p, low) for p in patterns):
                score += 10

        candidates.append((score, urljoin(NMA_INDEX, a["href"])))

    if not candidates:
        raise RuntimeError(
            "Could not discover a Never Manage Alone predicted-lineups article. "
            "Pass nma_url explicitly to fetch_lineup_sources()."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def fetch_lineup_sources(
    root: Path,
    gameweek: int | None = None,
    nma_url: str | None = None,
) -> dict[str, Path]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir = root / "data" / "raw" / "lineups" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = dict(SOURCES)
    urls["nma"] = nma_url or discover_nma_lineup_url(gameweek)

    saved = {}

    for name, url in urls.items():
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()

        path = out_dir / f"{name}.html"
        path.write_text(response.text, encoding="utf-8")

        saved[name] = path

    return saved


def parse_ffscout(path):
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    rows = []

    for card in soup.select("li.team-news-item"):
        team_tag = card.select_one("h2")
        if not team_tag:
            continue

        team = team_tag.get_text(" ", strip=True)

        players = card.select("span.player-name")

        for player in players:
            rows.append({
                "Source": "FFScout",
                "Team": team,
                "Player Raw": player.get_text(" ", strip=True),
                "Predicted Starter": True,
            })

    return pd.DataFrame(rows)

def parse_rotowire(path):
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    rows = []

    for fixture in soup.select("div.lineup.is-soccer"):

        home_tag = fixture.select_one(".lineup__mteam.is-home")
        away_tag = fixture.select_one(".lineup__mteam.is-visit")

        if not home_tag or not away_tag:
            continue

        home = home_tag.get_text(" ", strip=True)
        away = away_tag.get_text(" ", strip=True)

        for side, team in [("home", home), ("visit", away)]:

            lineup = fixture.select_one(f"ul.lineup__list.is-{side}")
            if not lineup:
                continue

            status_tag = lineup.select_one(".lineup__status")
            status = (
                status_tag.get_text(" ", strip=True)
                if status_tag
                else "Unknown"
            )

            players = lineup.select("li.lineup__player")

            # First 11 are the actual predicted XI
            for player in players[:11]:

                pos_tag = player.select_one(".lineup__pos")
                pos = (
                    pos_tag.get_text(" ", strip=True)
                    if pos_tag
                    else ""
                )

                text = player.get_text(" ", strip=True)

                if pos and text.startswith(pos):
                    name = text[len(pos):].strip()
                else:
                    name = text

                # Strip availability suffixes from player name
                for flag in [" OUT", " QUES", " SUS"]:
                    if name.upper().endswith(flag):
                        name = name[:-len(flag)].strip()

                rows.append({
                    "Source": "RotoWire",
                    "Team": team,
                    "Player Raw": name,
                    "Position Raw": pos,
                    "Lineup Status": status,
                    "Predicted Starter": True,
                })

    return pd.DataFrame(rows)


def parse_nma(path):
    html = path.read_text(encoding="utf-8")

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    lines = list(
        soup.stripped_strings
    )

    heading_re = re.compile(
        r"^(.+?) predicted line-up(?:\s+\([^)]+\))?:$",
        re.I,
    )

    rows = []

    for i, line in enumerate(lines):

        m = heading_re.match(line)

        if not m:
            continue

        team = m.group(1).strip()

        if i + 1 >= len(lines):
            continue

        lineup_text = lines[i + 1].strip()

        players = [
            p.strip().rstrip(".")
            for p in re.split(
                r"[;,]",
                lineup_text,
            )
            if p.strip()
        ]

        for player in players:

            rows.append({
                "Source": "NMA",
                "Team": team,
                "Player Raw": player,
                "Predicted Starter": True,
            })

    return pd.DataFrame(rows)