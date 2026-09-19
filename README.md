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
- `src/attack_projection.py` — player xG/xA allocation
- `src/defensive_points.py` — defensive-contribution and goalkeeper-save xPts
- `src/bonus_points.py` — historical/shrunk bonus expectation
- `src/xpts.py` — base FPL scoring
- `src/season_update.py` — cumulative in-season prior updates
- `src/transfer_optimizer.py` — multi-GW squad/transfer MILP

The older numbered notebooks are retained as historical development runs.

## Important modelling caveats

This is a decision model, not a claim of exact future points. Current limitations
worth addressing next are:

1. **Uncertainty is mostly collapsed to point estimates.** A Monte Carlo layer
   for starts, minutes and team goals would let the optimiser distinguish safe
   picks from high-variance picks with the same mean xPts.
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
