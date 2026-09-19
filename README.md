# FPL 2026/27 projection + transfer model

A Python model for Fantasy Premier League that combines live FPL data, predicted
lineups, historical minutes/role priors, team-strength projections, defensive
contribution modelling and mixed-integer optimisation.

## Current workflow

The preferred weekly entry point is:

```
notebooks/current_run.py
```

It is a VS Code/Jupyter-style Python script using `# %%` cells. It now:

- discovers the planning gameweek from the official FPL API instead of using a
  hard-coded GW number;
- folds every completed 2026/27 gameweek into the preseason priors cumulatively;
- tolerates a missing predicted-lineup source rather than treating it as a full
  set of negative votes;
- projects a configurable multi-GW horizon;
- includes appearance, goals, assists, clean sheets, goals conceded, defensive
  contributions, goalkeeper saves **and bonus points** in `xPts Model`;
- saves the current planning-GW projection to `data/predictions/` so it can be
  evaluated honestly after the event rather than reconstructed with hindsight;
- reports historical xPts/minutes errors plus start/P60 calibration once saved
  gameweeks have completed;
- compares rolling a transfer with the best one-transfer path and then solves
  the optimal multi-GW transfer plan.

Before each run, verify the settings at the top of `current_run.py`:

```python
CURRENT_SQUAD = [...]
BANK = 0.1
FREE_TRANSFERS = 2
SELL_PRICES = None
```

Supplying actual FPL selling prices is important when an owned player's sale
value differs from the live market price.


### Optional current bookmaker inputs

The workbook market sheet is still the calibration anchor, but the weekly run
will also look for:

```
data/market_overrides.csv
```

If that file exists, its current market information overrides the workbook on
matching fixtures. There are two supported formats.

Direct team xG:

```csv
GW,Home,Away,Market Home xG,Market Away xG,Home CS Prob,Away CS Prob
6,Arsenal,Everton,2.05,0.72,0.487,0.129
```

Or ordinary decimal match odds, optionally with the 2.5-goal market:

```csv
GW,Home,Away,Home Odds,Draw Odds,Away Odds,Over 2.5 Odds,Under 2.5 Odds
6,Arsenal,Everton,1.42,4.80,7.50,1.78,2.08
```

For the odds format, the model removes the bookmaker margin and fits
independent-Poisson home/away scoring rates to the 1X2 and O/U probabilities.
Those rates become market-implied team xG and clean-sheet probabilities.

This means near-term fixtures can use fresh market expectations when you have
them, while later fixtures continue to fall back to the strength model. The
weekly diagnostics report exactly which source was used.

## Setup

Create/activate a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `notebooks/current_run.py` in VS Code and run cells with the Python /
Jupyter extension.

## Model structure

- `src/fpl_api.py` — live FPL player and event data
- `src/lineup_sources.py` / `lineup_consensus.py` — predicted XI collection
  and matching
- `src/start_probs.py`, `minutes.py`, `horizon_minutes.py` — start and
  minutes forecasts
- `src/fixture_projection.py` — team xG / clean-sheet fixture engine
- `src/market_inputs.py` — current-market overrides and odds-to-xG inversion
- `src/attack_projection.py` — player xG/xA allocation
- `src/defensive_points.py` — defensive-contribution and goalkeeper-save xPts
- `src/bonus_points.py` — historical/shrunk bonus expectation
- `src/xpts.py` — base FPL scoring
- `src/season_update.py` — adaptive, recency-weighted in-season prior updates
- `src/transfer_optimizer.py` — multi-GW squad/transfer MILP
- `src/evaluation.py` — pre-deadline projection snapshots and post-GW scoring

The older numbered notebooks are retained as historical development runs.

## Important modelling caveats

This is a decision model, not a claim of exact future points. Current limitations
worth addressing next are:

1. **Uncertainty is mostly collapsed to point estimates.** The mean projection
   is now more responsive to current role and can use fresh bookmaker inputs,
   but a Monte Carlo layer for starts, minutes and team goals is still needed
   to distinguish safe picks from high-variance picks with the same mean xPts.
2. **Transfer prices are static over the horizon.** The optimiser does not
   forecast price changes and only knows correct selling values if they are
   supplied.
3. **Cards, own goals, penalty misses and goalkeeper penalty saves are not yet
   explicitly modelled.** Their expectation is small for most players but not
   literally zero.
4. **Team-strength updating is deliberately conservative.** Early-season xG is
   blended with a sizeable prior so one or two matches do not dominate.
5. **Predicted-lineup sources are scraped HTML.** Site markup can change, so
   source row counts and matching diagnostics should be checked each week.

## Repository hygiene

Generated virtual environments, Python caches, notebook checkpoints, macOS
metadata, Excel lock files, raw lineup scrapes and local cache refreshes are
ignored by Git. Historical snapshots already committed can still be recovered
from Git history.
