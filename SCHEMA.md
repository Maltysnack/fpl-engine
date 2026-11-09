# inputs.json

```json
{
  "gameweek": 11,
  "players": [ { ...player } ],
  "fixtures": [ { ...fixture } ],
  "context": { ...optional }
}
```

## player

| Field | Type | Notes |
|---|---|---|
| `id` | int | Unique id, referenced by fixtures and plans |
| `code` | int | Stable id across seasons (used by the DefCon tables) |
| `web_name` | string | Display name; `team.csv` matches on it |
| `team` | int | Team id |
| `element_type` | int | 1 GK, 2 DEF, 3 MID, 4 FWD |
| `now_cost` | int | Price in tenths of £1m (55 = £5.5m) |
| `status` | string | `u` marks a player as unavailable to buy |
| `minutes`, `total_points` | int | Season to date |
| `goals_scored`, `assists`, `clean_sheets`, `saves`, `bps`, `defensive_contribution` | int | Season to date |
| `form`, `ep_next` | string or number | Recent points per game, and an external expected points figure if you have one |
| `expected_goal_involvements_per_90`, `expected_goals_conceded_per_90` | number | Season to date |
| `chance_of_playing_next_round` | int or null | 0-100; null means no doubt |
| `penalties_order`, `direct_freekicks_order`, `corners_and_indirect_freekicks_order` | int or null | Set-piece order, 1 = first choice |
| `dream_team_count` | int | Optional |

Missing optional fields fall back to neutral values.

## fixture

| Field | Type | Notes |
|---|---|---|
| `event` | int | Gameweek |
| `team_h`, `team_a` | int | Home and away team ids |
| `team_h_difficulty`, `team_a_difficulty` | int | 1 (easy) to 5 (hard) |

Two fixtures for a team in one gameweek are summed (double gameweek).

## context (all optional)

| Field | Notes |
|---|---|
| `oddsMap` | `{ teamId: { winProb, pOver25, pBTTSno } }` from bookmaker odds |
| `minutesHistory` | `{ playerId: [{ gw, mins }] }`, enables the rotation penalty |
| `newsOverride` | `{ playerId: { chance, source } }`, overrides `chance_of_playing_next_round` |
| `defconLive` | `{ season, by_code: { code: [{ round, minutes, defensive_contribution }] } }`, season-to-date DefCon |
| `calibration`, `priors`, `teamStrength`, `h2h`, `congestedGws`, `managerChanges` | Further corrections; see the header of `predict.js` |
