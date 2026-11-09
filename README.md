# FPL engine

Points prediction, transfer planner, chip logic and season simulator for Fantasy
Premier League. The engine does no network access when deciding: you pass the
data in. Fitted files ship as they are.

## Run it on your own team

```bash
node plan.js examples/sample-inputs.json examples/team.csv examples/state.json
```

`examples/` holds a synthetic slate with made-up players, so it runs as is.
For a real gameweek, build `inputs.json` from your own data in the shape set
out in [SCHEMA.md](SCHEMA.md), then list your 15 players in `team.csv` and your
bank, free transfers and gameweek in `state.json`.

Output is every plan the engine considered (no transfer, best one, two and three
transfers), each with expected points this gameweek and over an 8-gameweek
horizon, plus the engine's pick and a captain. The pick comes from a beam search,
so it is not always the row with the highest horizon total.

## Past seasons

`vaastav-fetch.js` downloads past-season data from Vaastav's public archive, and
`squad-sim.js` replays a season with the planner and chips. Data files are read
from `FPL_DATA_DIR` (default: this folder).

## Tests

```bash
node test/determinism.js
```

Same inputs give the same predictions and plan, checked against `test/golden.json`.

## Files

| File | What it does |
|---|---|
| `predict.js`, `models.js`, `calibration.js` | Expected points per player per gameweek |
| `planner.js` | Beam-search transfer planner, plus ranked alternatives |
| `chips.js`, `two-gw-squad.js` | Chip timing and squad rebuilds |
| `squad-sim.js` | Season simulator |
| `fit.js`, `ensemble-fit.js` | Refit the priors and ensemble weights |
| `ml/` | Residual model on top of the formula |
| `*.json` | Fitted priors, weights and rate tables |
