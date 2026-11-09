#!/usr/bin/env node
/**
 * projects/fpl/impute-vaastav.js
 *
 * Fixes issue #7: Vaastav's 2022-23 CSVs don't have expected_goal_involvements
 * or expected_goals_conceded fields for GW1-15 (they start at GW16). Adversarial
 * validation flagged this as a 2.36-SD distribution shift on xgc90 between
 * seasons.
 *
 * Strategy: for each player with available xGI/xGC data later in the season
 * (GW16+), compute their per-90 rate from THAT window. Backfill the missing
 * GW1-15 history entries with rate * (minutes/90) per entry. Players who
 * didn't play any GWs after the data starts get position-fallback rates.
 *
 * Players' xGI/xGC rates are mostly stable within a season (team and role
 * don't usually change mid-year), so each player's own later-season rate is
 * a credible estimate of their early-season rate. Not perfect, but eliminates
 * the systematic 3x undershoot that 0-imputation creates.
 *
 * USAGE
 *   node impute-vaastav.js                    # write changes back to vaastav-cache.json
 *   node impute-vaastav.js --dry-run          # show what would change
 */

const fs   = require('fs');
const path = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const VCACHE = path.join(DATA_DIR, 'vaastav-cache.json');
const DRY_RUN = process.argv.includes('--dry-run');

const vcache = JSON.parse(fs.readFileSync(VCACHE, 'utf8'));

// Position-fallback rates from 2023-24/2024-25 means (rough, used when a player
// has zero data we can rate from). These keep the model from defaulting to 0.
const POS_FALLBACK_XGI90 = { 1: 0.01, 2: 0.10, 3: 0.20, 4: 0.40 };
const POS_FALLBACK_XGC90 = { 1: 1.40, 2: 1.40, 3: 1.40, 4: 1.40 };

// For each player in a target season + target GW range, build their xGI/xGC
// rate from a SOURCE window (GW range where data is present).
function imputePlayer(entry, missingRange, sourceRange) {
  const sourceEntries = entry.history.filter(h =>
    h.round >= sourceRange[0] && h.round <= sourceRange[1] &&
    h.minutes > 0 &&
    (h.expected_goal_involvements > 0 || h.expected_goals_conceded > 0)
  );

  let xgi90, xgc90;
  if (sourceEntries.length >= 3) {
    const totalMins = sourceEntries.reduce((s, h) => s + h.minutes, 0);
    const totalXgi  = sourceEntries.reduce((s, h) => s + (h.expected_goal_involvements || 0), 0);
    const totalXgc  = sourceEntries.reduce((s, h) => s + (h.expected_goals_conceded    || 0), 0);
    xgi90 = totalMins > 0 ? totalXgi / (totalMins / 90) : POS_FALLBACK_XGI90[entry.element_type];
    xgc90 = totalMins > 0 ? totalXgc / (totalMins / 90) : POS_FALLBACK_XGC90[entry.element_type];
  } else {
    // Not enough sample data; use position fallback. Conservative: accepts
    // a bias toward the position mean rather than guessing this specific player.
    xgi90 = POS_FALLBACK_XGI90[entry.element_type] || 0.1;
    xgc90 = POS_FALLBACK_XGC90[entry.element_type] || 1.4;
  }

  let imputed = 0;
  for (const h of entry.history) {
    if (h.round < missingRange[0] || h.round > missingRange[1]) continue;
    // Only impute when the value is currently 0 AND the player actually played.
    if (h.minutes === 0) continue;
    if (h.expected_goal_involvements === 0) {
      h.expected_goal_involvements = +(xgi90 * h.minutes / 90).toFixed(3);
      imputed++;
    }
    if (h.expected_goals_conceded === 0) {
      h.expected_goals_conceded = +(xgc90 * h.minutes / 90).toFixed(3);
    }
  }
  return imputed;
}

// Mark already-imputed seasons so re-runs are idempotent.
vcache._imputed = vcache._imputed || {};
const ALREADY = vcache._imputed['2022-23'];
if (ALREADY) {
  console.log('2022-23 already imputed. To force re-impute, delete vaastav-cache.json._imputed["2022-23"] first.');
  process.exit(0);
}

// 2022-23: missing GW1-15, has data GW16-38. Also GW29 had a partial outage.
const MISSING = [1, 15];
const SOURCE  = [16, 38];

let totalPlayers = 0, totalImputed = 0, skipped = 0;
for (const key in vcache.history) {
  const entry = vcache.history[key];
  if (entry.season !== '2022-23') continue;
  totalPlayers++;
  const n = imputePlayer(entry, MISSING, SOURCE);
  totalImputed += n;
  if (n === 0) skipped++;
}

console.log(`Imputed ${totalImputed} GW entries across ${totalPlayers - skipped} players (${skipped} skipped, no minutes in GW1-15).`);

if (!DRY_RUN) {
  vcache._imputed['2022-23'] = { imputedAt: new Date().toISOString(), entries: totalImputed };
  fs.writeFileSync(VCACHE, JSON.stringify(vcache));
  console.log(`Wrote ${VCACHE}.`);
} else {
  console.log('Dry-run; no changes written.');
}
