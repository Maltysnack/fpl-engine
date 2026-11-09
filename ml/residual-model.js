// residual-model.js - ML Phase 4: Node serving for the residual model.
//
// The residual model (gh#5) is xgboost, trained in Python on
// target = actual - formula_pred. This module evaluates it at runtime with
// zero Python dependency: it loads the dumped trees (residual-trees.json,
// produced by dump-model.py) and walks them directly.
//
// predict.js loads this module (Node-only) and, when ctx.residual === true,
// adds predictResidual(extractFeatures(...)) to the formula output.
//
// LEAKAGE NOTE: extractFeatures must produce the SAME 19 features
// build-dataset.js trains on. The feature list and order are pinned by
// residual-trees.json's `features` array - verify-walker.js checks parity.

const fs   = require('fs');
const path = require('path');

let MODEL = null;
function load() {
  if (MODEL) return MODEL;
  const raw = fs.readFileSync(path.join(__dirname, 'residual-trees.json'), 'utf8');
  MODEL = JSON.parse(raw);
  // xgboost compares in float32. Pre-round thresholds so branch decisions
  // match exactly (a float64 comparison flips branches near split points).
  for (const tree of MODEL.trees) {
    for (const n of tree) if (n.t !== undefined) n.t = Math.fround(n.t);
  }
  return MODEL;
}

// Walk one tree. xgboost rule: feature value < threshold -> left child;
// a missing/NaN value follows the precomputed missing-direction index.
// Comparison is done in float32 (Math.fround) to match xgboost exactly.
function walkTree(nodes, feats) {
  let i = 0;
  for (;;) {
    const n = nodes[i];
    if (n.v !== undefined) return n.v;
    const v = feats[n.f];
    i = (v == null || Number.isNaN(v)) ? n.m : (v < n.t ? n.l : n.r);
  }
}

// predictResidual: featObj is keyed by feature name (extractFeatures output).
// Result = base_score + sum of every tree's reached leaf.
function predictResidual(featObj) {
  const m = load();
  const feats = m.features.map(name => {
    const x = featObj[name];
    return (x == null || Number.isNaN(+x)) ? NaN : Math.fround(+x);
  });
  let sum = m.base_score;
  for (const tree of m.trees) sum += walkTree(tree, feats);
  return sum;
}

// Safe per-90 rate. Mirrors build-dataset.js per90() exactly.
function per90(total, minutes) {
  return minutes > 0 ? +((total || 0) / (minutes / 90)).toFixed(4) : 0;
}

// Build the 19-feature vector from a player object + fixture context.
// Works for both call paths:
//   - live FPL element: has points_per_game, starts, now_cost
//   - synthesised player: caller must attach games_played + value
// games_played is derived from FPL's points_per_game when present
// (points_per_game = total_points / appearances, so appearances back out).
function extractFeatures(player, fixList, isHome) {
  const mins = player.minutes || 0;
  let gp = player.games_played;
  if (gp == null) {
    if (player.points_per_game > 0) gp = player.total_points / player.points_per_game;
    else if (player.starts != null) gp = player.starts;
    else gp = mins / 90;
  }
  const games = Math.max(1, Math.round(gp));
  return {
    element_type:        player.element_type,
    form:                parseFloat(player.form)    || 0,
    ep_next:             parseFloat(player.ep_next)  || 0,
    ppg:                 +((player.total_points || 0) / games).toFixed(4),
    total_points_so_far: player.total_points || 0,
    xgi90:               parseFloat(player.expected_goal_involvements_per_90) || 0,
    xgc90:               parseFloat(player.expected_goals_conceded_per_90)    || 0,
    bps_per90:           per90(player.bps,           mins),
    bonus_per90:         per90(player.bonus,         mins),
    goals_per90:         per90(player.goals_scored,  mins),
    assists_per90:       per90(player.assists,       mins),
    saves_per90:         per90(player.saves,         mins),
    cs_rate:             +((player.clean_sheets || 0) / games).toFixed(4),
    minutes_total:       mins,
    games_played:        games,
    avg_minutes:         +(mins / games).toFixed(2),
    was_home:            isHome ? 1 : 0,
    num_fixtures:        fixList.length,
    value:               player.value != null ? player.value
                         : (player.now_cost != null ? player.now_cost : 0)
  };
}

module.exports = { predictResidual, extractFeatures, load };
