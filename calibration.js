#!/usr/bin/env node
/**
 * projects/fpl/calibration.js
 *
 * Calibration analysis. Bucket predicted points into ranges, compute the
 * average ACTUAL outcome in each bucket, and report where the model is
 * mis-calibrated. A perfectly calibrated model has actual ≈ predicted in
 * every bucket. Real models typically show one of:
 *
 *   - Regression to the mean: predicted 8 → actual 6 (over-confident high).
 *   - Pessimism: predicted 4 → actual 5 (under-confident).
 *   - Bias: a constant offset across the range.
 *
 * Run with and without priors to see if the OLS stage-2 priors actually
 * straighten the calibration line.
 *
 * USAGE
 *   node projects/fpl/calibration.js                # ALL seasons combined
 *   node projects/fpl/calibration.js --season=2024-25
 *   node projects/fpl/calibration.js --no-priors
 *   node projects/fpl/calibration.js --by-position
 *   node projects/fpl/calibration.js --brier         # haul-probability Brier score
 */

const fs   = require('fs');
const path = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const { predictPlayer } = require('./predict.js');

const CACHE  = path.join(DATA_DIR, 'api-cache.json');
const VCACHE = path.join(DATA_DIR, 'vaastav-cache.json');
const PRIORS = path.join(DATA_DIR, 'priors.json');

function synthesisePlayer(meta, history, gw) {
  const past = history.filter(h => h.round < gw);
  if (past.length === 0) return null;
  const minutes = past.reduce((s, h) => s + h.minutes, 0);
  if (minutes < 90) return null;
  const total_points = past.reduce((s, h) => s + h.total_points, 0);
  const goals        = past.reduce((s, h) => s + h.goals_scored, 0);
  const assists      = past.reduce((s, h) => s + h.assists, 0);
  const clean_sheets = past.reduce((s, h) => s + h.clean_sheets, 0);
  const saves        = past.reduce((s, h) => s + h.saves, 0);
  const bps          = past.reduce((s, h) => s + h.bps, 0);
  const bonus        = past.reduce((s, h) => s + h.bonus, 0);
  const xgi          = past.reduce((s, h) => s + (h.expected_goal_involvements || 0), 0);
  const xgc          = past.reduce((s, h) => s + (h.expected_goals_conceded    || 0), 0);
  const last5 = past.slice(-5);
  const form  = last5.length ? last5.reduce((s, h) => s + h.total_points, 0) / last5.length : 0;
  const last4 = past.slice(-4);
  const ep    = last4.length ? last4.reduce((s, h) => s + h.total_points, 0) / last4.length : 0;
  return {
    id: meta.id, web_name: meta.web_name, team: meta.team, element_type: meta.element_type,
    chance_of_playing_next_round: 100,
    minutes, total_points,
    goals_scored: goals, assists, clean_sheets, saves, bps, bonus,
    expected_goal_involvements_per_90: minutes > 0 ? +(xgi / (minutes / 90)).toFixed(3) : 0,
    expected_goals_conceded_per_90:    minutes > 0 ? +(xgc / (minutes / 90)).toFixed(3) : 0,
    ep_next: ep.toFixed(2), form: form.toFixed(2)
  };
}

function collectPairs(vcache, seasons, ctx) {
  const pairs = [];
  for (const k in vcache.history) {
    const entry = vcache.history[k];
    if (!seasons.includes(entry.season)) continue;
    for (const h of entry.history) {
      if (h.minutes < 45) continue;
      const synth = synthesisePlayer(
        { id: k, web_name: entry.web_name, team: entry.team, element_type: entry.element_type },
        entry.history, h.round
      );
      if (!synth) continue;
      const fix = {
        team_h: h.was_home ? entry.team : h.opponent_team,
        team_a: h.was_home ? h.opponent_team : entry.team,
        team_h_difficulty: 3, team_a_difficulty: 3
      };
      const pred = predictPlayer(synth, [fix], ctx);
      if (pred < 0.5) continue;
      pairs.push({ pred, act: h.total_points, pos: synth.element_type, season: entry.season });
    }
  }
  return pairs;
}

const CAL_HISTORY = path.join(DATA_DIR, 'calibration-history.json');

// Snapshot the per-bucket stats so we can trend over time. Used by the
// monitor function below to flag buckets whose bias drifts dangerously.
function snapshotBuckets(pairs, label) {
  const BIN = 0.5;
  const buckets = {};
  for (const p of pairs) {
    const b = Math.floor(p.pred / BIN);
    if (!buckets[b]) buckets[b] = { n: 0, sumPred: 0, sumAct: 0 };
    buckets[b].n++;
    buckets[b].sumPred += p.pred;
    buckets[b].sumAct  += p.act;
  }
  const out = {};
  for (const k of Object.keys(buckets)) {
    const b = buckets[k];
    out[k] = {
      bucketStart: +(parseInt(k) * BIN).toFixed(1),
      n: b.n,
      meanPred: +(b.sumPred / b.n).toFixed(3),
      meanAct:  +(b.sumAct  / b.n).toFixed(3),
      bias:     +((b.sumAct - b.sumPred) / b.n).toFixed(3)
    };
  }
  return { label, total: pairs.length, buckets: out };
}

// Append the current snapshot to calibration-history.json. Lets us trend
// bucket biases across multiple runs (issue #3 monitor).
function saveSnapshot(snapshot, tag) {
  let hist = { snapshots: [] };
  try { hist = JSON.parse(fs.readFileSync(CAL_HISTORY, 'utf8')); } catch (e) {}
  hist.snapshots.push({
    timestamp: new Date().toISOString(),
    tag: tag || 'untagged',
    ...snapshot
  });
  // Keep last 20 snapshots; trim older to keep file small.
  hist.snapshots = hist.snapshots.slice(-20);
  fs.writeFileSync(CAL_HISTORY, JSON.stringify(hist, null, 2));
}

// Compare the latest snapshot to the previous one. Flag buckets whose:
//   - sample count crossed the 50 threshold (now actionable)
//   - bias drifted by > 0.5 in either direction
//   - bias remains > +1.0 or < -1.0 (persistent issue)
function compareToLast() {
  let hist;
  try { hist = JSON.parse(fs.readFileSync(CAL_HISTORY, 'utf8')); }
  catch (e) { console.log('No previous snapshot for comparison.'); return; }
  if (hist.snapshots.length < 2) { console.log('Only one snapshot in history; need 2+ to compare.'); return; }
  const a = hist.snapshots[hist.snapshots.length - 2];
  const b = hist.snapshots[hist.snapshots.length - 1];
  console.log(`\n── Drift vs previous snapshot (${a.tag} → ${b.tag}) ──`);
  for (const k of Object.keys(b.buckets)) {
    const cur  = b.buckets[k];
    const prev = a.buckets && a.buckets[k];
    if (!prev) continue;
    const sampleGrew50 = prev.n < 50 && cur.n >= 50;
    const biasShift    = Math.abs(cur.bias - prev.bias);
    const biasBad      = Math.abs(cur.bias) > 1.0 && cur.n >= 30;
    if (sampleGrew50 || biasShift > 0.5 || biasBad) {
      const flags = [];
      if (sampleGrew50) flags.push('crossed-50');
      if (biasShift > 0.5) flags.push('shifted-' + biasShift.toFixed(2));
      if (biasBad) flags.push('persistent-bias-' + cur.bias.toFixed(2));
      console.log(`  bucket ${cur.bucketStart}: n=${cur.n}, bias=${cur.bias} (was ${prev.bias})  [${flags.join(', ')}]`);
    }
  }
  console.log('');
}

// Bucket predictions in 0.5-point wide ranges. For each bucket, report:
//   count, mean predicted, mean actual, mean residual (actual - predicted), bias.
// Output as an ASCII table that doubles as a calibration plot.
function calibrationTable(pairs, label) {
  const BIN = 0.5;
  const buckets = {};
  for (const p of pairs) {
    const b = Math.floor(p.pred / BIN);
    if (!buckets[b]) buckets[b] = { n: 0, sumPred: 0, sumAct: 0 };
    buckets[b].n++;
    buckets[b].sumPred += p.pred;
    buckets[b].sumAct  += p.act;
  }

  const keys = Object.keys(buckets).map(Number).sort((a, b) => a - b);
  if (!keys.length) { console.log(`  ${label}: no data`); return; }

  console.log(`\n── ${label} (n=${pairs.length}) ──`);
  console.log('  bucket       n     mean_pred  mean_act   bias    bar');
  for (const k of keys) {
    const b = buckets[k];
    if (b.n < 10) continue;   // skip thin buckets to avoid noise
    const meanPred = b.sumPred / b.n;
    const meanAct  = b.sumAct  / b.n;
    const bias     = meanAct - meanPred;
    // Bar: dots represent mean_pred, # represents mean_act. Aligned to a 25-char scale.
    const W = 25;
    const SCALE = 10;   // max value we show on the bar
    const pPos = Math.min(W, Math.round(meanPred / SCALE * W));
    const aPos = Math.min(W, Math.round(meanAct  / SCALE * W));
    let bar = ' '.repeat(W + 1).split('');
    for (let i = 0; i <= pPos; i++) bar[i] = '.';
    bar[aPos] = '#';
    if (pPos > aPos) bar[pPos] = '|';
    const sign = bias >= 0 ? '+' : '';
    console.log(
      '  ' + ((k * BIN).toFixed(1) + '-' + ((k + 1) * BIN).toFixed(1)).padEnd(9) +
      String(b.n).padStart(6) + '   ' +
      meanPred.toFixed(2).padStart(7) + '    ' +
      meanAct.toFixed(2).padStart(6) + '   ' +
      (sign + bias.toFixed(2)).padStart(6) + '  ' +
      bar.join('')
    );
  }

  // Best-fit slope through buckets (heavier buckets weighted more).
  let W = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const k of keys) {
    const b = buckets[k];
    if (b.n < 10) continue;
    const x = b.sumPred / b.n, y = b.sumAct / b.n, w = b.n;
    W += w; sx += w * x; sy += w * y; sxx += w * x * x; sxy += w * x * y;
  }
  const denom = W * sxx - sx * sx;
  if (denom !== 0) {
    const slope = (W * sxy - sx * sy) / denom;
    const intercept = (sy - slope * sx) / W;
    console.log(`  Best-fit through buckets: actual ≈ ${slope.toFixed(3)} * predicted + ${intercept.toFixed(3)}`);
    if (slope < 0.85)      console.log('    → regression to mean (model over-confident: predicts high, actuals tame)');
    else if (slope > 1.15) console.log('    → pessimism (model under-predicts upside)');
    else                   console.log('    → slope close to 1.0 (calibrated dispersion)');
    if (intercept > 0.5)      console.log('    → systematic under-prediction (actuals ~' + intercept.toFixed(1) + ' pts above predictions)');
    else if (intercept < -0.5) console.log('    → systematic over-prediction');
    else                       console.log('    → intercept close to 0 (no constant bias)');
  }
}

// Brier score for haul prediction (21-step #16). Treats each prediction as the
// rate parameter (lambda) of a Poisson distribution; computes P(actual >= 8 pts)
// = 1 - CDF(7, lambda). Brier = mean((P - I[actual>=8])^2). Lower is better;
// a uniform 0.5 baseline scores ~0.25.
//
// Why Poisson: FPL points-per-match is fat-tailed (lots of 0s, a few 12+s) but
// the Poisson assumption gives a workable probability for the haul threshold
// without learning a separate distribution per player.
function poissonPGreaterEq(k, lambda) {
  // 1 - P(X < k) = 1 - sum_{i=0..k-1} (lambda^i e^-lambda) / i!
  if (lambda <= 0) return 0;
  let p = 0, term = Math.exp(-lambda), fact = 1;
  for (let i = 0; i < k; i++) {
    p += term;
    fact *= (i + 1);
    term *= lambda / (i + 1);
  }
  return Math.max(0, Math.min(1, 1 - p));
}

function brierScore(pairs, threshold) {
  if (!pairs.length) return null;
  let sum = 0, hits = 0, sumP = 0;
  for (const p of pairs) {
    const prob   = poissonPGreaterEq(threshold, p.pred);
    const target = p.act >= threshold ? 1 : 0;
    sum  += (prob - target) ** 2;
    hits += target;
    sumP += prob;
  }
  const mean   = sum / pairs.length;
  const rate   = hits / pairs.length;
  const meanP  = sumP / pairs.length;
  // Baseline Brier from predicting the empirical haul rate for everyone.
  const baseline = rate * (1 - rate);
  return { n: pairs.length, brier: +mean.toFixed(4), hauls: hits, hauleRate: +rate.toFixed(4), meanPredProb: +meanP.toFixed(4), baseline: +baseline.toFixed(4), skill: +(1 - mean / baseline).toFixed(3) };
}

async function main() {
  const args = process.argv.slice(2);
  const sArg = args.find(a => a.startsWith('--season='));
  const seasonFilter = sArg ? [sArg.split('=')[1]] : null;
  const skipPriors   = args.includes('--no-priors');
  const byPosition   = args.includes('--by-position');
  const doBrier      = args.includes('--brier');
  // --save tags this run and appends a snapshot to calibration-history.json so
  // we can monitor drift over time (issue #3 ongoing monitor).
  const saveArg      = args.find(a => a.startsWith('--save='));
  const saveTag      = saveArg ? saveArg.split('=')[1] : (args.includes('--save') ? 'untagged' : null);
  // --compare prints drift vs the previous saved snapshot.
  const doCompare    = args.includes('--compare');

  const cache  = JSON.parse(fs.readFileSync(CACHE,  'utf8'));
  const vcache = JSON.parse(fs.readFileSync(VCACHE, 'utf8'));
  const allSeasons = [...new Set(Object.values(vcache.history).map(v => v.season))].sort();
  const seasons    = seasonFilter || allSeasons;

  const ctx = {
    calibration:  (cache.calibration && cache.calibration.factors) || null,
    teamStrength: cache.teamStrength || null
  };
  if (!skipPriors && fs.existsSync(PRIORS)) {
    try { ctx.priors = JSON.parse(fs.readFileSync(PRIORS, 'utf8')); } catch (e) {}
  }

  const flag = skipPriors ? '[no-priors]' : (ctx.priors ? '[+priors]' : '[priors missing]');
  console.log(`Calibration analysis. Seasons: ${seasons.join(', ')}. ${flag}`);

  const pairs = collectPairs(vcache, seasons, ctx);
  console.log(`Collected ${pairs.length} (pred, actual) pairs.`);

  calibrationTable(pairs, 'ALL');

  if (byPosition) {
    const POS = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' };
    for (const pos of [1, 2, 3, 4]) {
      calibrationTable(pairs.filter(p => p.pos === pos), POS[pos]);
    }
  }

  if (doBrier) {
    const HAUL = 8;
    const b = brierScore(pairs, HAUL);
    console.log(`\n── Brier score: P(actual >= ${HAUL}) via Poisson(lambda=pred) ──`);
    console.log(`  n=${b.n}, hauls=${b.hauls} (${(b.hauleRate * 100).toFixed(1)}%), mean predicted P=${(b.meanPredProb * 100).toFixed(1)}%`);
    console.log(`  Brier=${b.brier} (lower is better; baseline=${b.baseline}, skill=${b.skill})`);
    if (byPosition) {
      const POS = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' };
      for (const pos of [1, 2, 3, 4]) {
        const sub = brierScore(pairs.filter(p => p.pos === pos), HAUL);
        if (sub) console.log(`  ${POS[pos]}: n=${sub.n}, hauls=${sub.hauls}, Brier=${sub.brier}, skill=${sub.skill}`);
      }
    }
  }

  if (saveTag) {
    saveSnapshot(snapshotBuckets(pairs, 'ALL'), saveTag);
    console.log(`\nSaved snapshot to calibration-history.json (tag: ${saveTag}).`);
  }

  if (doCompare) compareToLast();
}

main().catch(e => { console.error(e); process.exit(1); });
