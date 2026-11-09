#!/usr/bin/env node
/**
 * projects/fpl/ensemble-fit.js
 *
 * Learns per-position blend weights for the 5+1 variants defined in models.js.
 *
 * For each walk-forward (player, GW) sample, we compute predictions under
 * every variant. Then per position, we search for non-negative weights that
 * sum to 1 such that the weighted blend minimises mean absolute error against
 * actual points. Output: ensemble-weights.json, consumed by models.predictEnsemble.
 *
 * Search method: coordinate descent. Start at equal weights. Repeatedly nudge
 * pairs of weights (give some from one to another, normalise to sum=1) and
 * keep moves that improve MAE. Converges quickly with only 6 variants per
 * position. Robust against local optima compared to grid search on 6 dims.
 *
 * USAGE
 *   node projects/fpl/ensemble-fit.js                    # all sources
 *   node projects/fpl/ensemble-fit.js --sources=vaastav  # subset
 *   node projects/fpl/ensemble-fit.js --gws=20           # current-season window
 */

const fs    = require('fs');
const path  = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const { predictPlayer } = require('./predict.js');
const { VARIANTS }      = require('./models.js');

const CACHE   = path.join(DATA_DIR, 'api-cache.json');
const HCACHE  = path.join(DATA_DIR, 'history-cache.json');
const VCACHE  = path.join(DATA_DIR, 'vaastav-cache.json');
const SNAPDIR = path.join(DATA_DIR, 'snapshots');
const OUT     = path.join(DATA_DIR, 'ensemble-weights.json');

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

const mae = arr => arr.reduce((s, x) => s + x, 0) / arr.length;

// NDCG@K: how well does the predicted top-K match the actual top-K, weighted
// to penalise high-actual players being ranked low. We bucket samples per GW
// (so the ranking is "best players in THIS GW"), compute NDCG@K each, average.
// Returns 1 - NDCG so lower-is-better and same orientation as MAE.
function negativeNdcgAtK(samples, blendPredFn, K) {
  const byGw = {};
  for (const s of samples) {
    const key = s.gw != null ? s.gw : 'all';
    (byGw[key] = byGw[key] || []).push(s);
  }
  let ndcgSum = 0, n = 0;
  for (const k in byGw) {
    const grp = byGw[k];
    if (grp.length < K) continue;
    const withBlend = grp.map(s => ({ pred: blendPredFn(s.preds), act: s.actual }));
    withBlend.sort((a, b) => b.pred - a.pred);
    const topByPred = withBlend.slice(0, K);
    // DCG: sum of actual_i / log2(rank + 2), iterating from rank 0..K-1.
    const dcg = topByPred.reduce((sum, x, i) => sum + (x.act > 0 ? x.act : 0) / Math.log2(i + 2), 0);
    const idealSorted = [...withBlend].sort((a, b) => b.act - a.act).slice(0, K);
    const idcg = idealSorted.reduce((sum, x, i) => sum + (x.act > 0 ? x.act : 0) / Math.log2(i + 2), 0);
    if (idcg > 0) { ndcgSum += dcg / idcg; n++; }
  }
  if (n === 0) return 1.0;
  return 1.0 - (ndcgSum / n);
}

// Coordinate-descent search over the simplex for per-position blend weights.
// `samples` is an array of { actual, preds: { variantName: pred }, gw? } for one position.
function searchWeights(samples, variantNames, opts) {
  opts = opts || {};
  const loss = opts.loss || 'mae';   // 'mae' or 'ndcg'
  const K    = opts.k || 11;

  let w = {};
  for (const n of variantNames) w[n] = 1 / variantNames.length;

  function blendPred(predsObj, weights) {
    let total = 0, wsum = 0;
    for (const n of variantNames) {
      const wi = weights[n];
      if (wi > 0) { total += wi * predsObj[n]; wsum += wi; }
    }
    return wsum > 0 ? total / wsum : 0;
  }

  function score(weights) {
    if (loss === 'ndcg') {
      return negativeNdcgAtK(samples, p => blendPred(p, weights), K);
    }
    const errs = samples.map(s => Math.abs(blendPred(s.preds, weights) - s.actual));
    return mae(errs);
  }

  let best = score(w);
  const STEPS = [0.25, 0.10, 0.04, 0.015];
  for (const step of STEPS) {
    let improved = true;
    let iter = 0;
    while (improved && iter < 200) {
      improved = false;
      iter++;
      for (const from of variantNames) {
        if (w[from] < step) continue;
        for (const to of variantNames) {
          if (to === from) continue;
          const trial = Object.assign({}, w);
          trial[from] -= step;
          trial[to]   += step;
          const s = score(trial);
          if (s < best - 1e-6) {
            w = trial;
            best = s;
            improved = true;
          }
        }
      }
    }
  }
  return { weights: w, score: best };
}

async function collectSamples(ctx, opts) {
  const { sources, startGw, lastGwId, maxSeason } = opts;
  const variantNames = Object.keys(VARIANTS);
  const samples = [];   // { actual, pos, preds: { variantName: pred } }

  // Helper: produce predictions under each variant for a player+fix.
  function predictAll(player, fixList) {
    const out = {};
    for (const v of variantNames) {
      const variantCtx = Object.assign({}, ctx, { variant: VARIANTS[v] });
      out[v] = predictPlayer(player, fixList, variantCtx);
    }
    return out;
  }

  function processSynthesisCache(cache, sourceTag) {
    const wrange = sourceTag === 'vaastav' ? [2, 38] : [startGw, lastGwId];
    for (const pid in cache.history) {
      const entry = cache.history[pid];
      // gh#39 clean-holdout: skip seasons after maxSeason.
      if (maxSeason && (entry.season || '') > maxSeason) continue;
      for (let gw = wrange[0]; gw <= wrange[1]; gw++) {
        const gwEntries = entry.history.filter(h => h.round === gw);
        if (!gwEntries.length) continue;
        const totalMins = gwEntries.reduce((s, h) => s + h.minutes, 0);
        if (totalMins < 45) continue;
        const totalPts  = gwEntries.reduce((s, h) => s + h.total_points, 0);

        const synth = synthesisePlayer(
          { id: pid, web_name: entry.web_name, team: entry.team, element_type: entry.element_type },
          entry.history, gw
        );
        if (!synth) continue;
        const fixList = gwEntries.map(h => ({
          team_h: h.was_home ? entry.team : h.opponent_team,
          team_a: h.was_home ? h.opponent_team : entry.team,
          team_h_difficulty: 3, team_a_difficulty: 3
        }));
        const preds = predictAll(synth, fixList);
        if (Object.values(preds).every(p => p < 0.5)) continue;
        // gw key is (sourceTag, season, round) so NDCG groups per-GW per-season.
        const gwKey = `${sourceTag}_${entry.season || 'current'}_gw${gw}`;
        samples.push({ actual: totalPts, pos: synth.element_type, preds, gw: gwKey });
      }
    }
  }

  if (sources.includes('history') && fs.existsSync(HCACHE)) {
    processSynthesisCache(JSON.parse(fs.readFileSync(HCACHE, 'utf8')), 'history');
  }
  if (sources.includes('vaastav') && fs.existsSync(VCACHE)) {
    processSynthesisCache(JSON.parse(fs.readFileSync(VCACHE, 'utf8')), 'vaastav');
  }
  if (sources.includes('snapshots') && fs.existsSync(SNAPDIR)) {
    const files = fs.readdirSync(SNAPDIR).filter(f => f.endsWith('.json'));
    for (const f of files) {
      const snap = JSON.parse(fs.readFileSync(path.join(SNAPDIR, f), 'utf8'));
      let live;
      // Actual scores for this GW, supplied by the caller. Missing file: skip the GW.
      try { live = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'snapshots-live', `gw${snap.gwId}.json`), 'utf8')); }
      catch (e) { continue; }
      if (!live || !live.elements) continue;
      const actByEl = {};
      live.elements.forEach(el => { actByEl[el.id] = { pts: el.stats.total_points, mins: el.stats.minutes }; });
      const fixByTeam = {};
      (snap.fixtures || []).forEach(fx => {
        (fixByTeam[fx.team_h] = fixByTeam[fx.team_h] || []).push(fx);
        (fixByTeam[fx.team_a] = fixByTeam[fx.team_a] || []).push(fx);
      });
      for (const p of snap.elements) {
        const a = actByEl[p.id];
        if (!a || a.mins < 45) continue;
        const fix = fixByTeam[p.team];
        if (!fix || !fix.length) continue;
        const preds = predictAll(p, fix);
        if (Object.values(preds).every(v => v < 0.5)) continue;
        samples.push({ actual: a.pts, pos: p.element_type, preds, gw: `snapshot_gw${snap.gwId}` });
      }
    }
  }
  return samples;
}

async function main() {
  const args  = process.argv.slice(2);
  const gwArg = args.find(a => a.startsWith('--gws='));
  const K     = gwArg ? parseInt(gwArg.split('=')[1], 10) : 34;
  const sArg  = args.find(a => a.startsWith('--sources='));
  const sources = sArg ? sArg.split('=')[1].split(',') : ['history', 'snapshots', 'vaastav'];
  const msArg = args.find(a => a.startsWith('--max-season='));
  const maxSeason = msArg ? msArg.split('=')[1] : null;   // gh#39 clean-holdout
  const lossArg = args.find(a => a.startsWith('--loss='));
  const loss    = lossArg ? lossArg.split('=')[1] : 'mae';
  if (loss !== 'mae' && loss !== 'ndcg') { console.error(`Unknown loss: ${loss}`); process.exit(1); }

  if (!fs.existsSync(CACHE)) { console.error(`Missing ${CACHE}.`); process.exit(1); }
  const cache = JSON.parse(fs.readFileSync(CACHE, 'utf8'));
  const ctx = {
    calibration:  (cache.calibration && cache.calibration.factors) || null,
    teamStrength: cache.teamStrength || null,
    priors:       (function () { try { return JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'priors.json'), 'utf8')); } catch (e) { return null; } })()
  };
  const lastGwId = cache.lastGwId;
  const startGw  = Math.max(2, lastGwId - K + 1);

  console.log(`Ensemble fit. Loss: ${loss === 'ndcg' ? 'NDCG@11 (rank-aware)' : 'MAE'}. Sources: ${sources.join(', ')}. Variants: ${Object.keys(VARIANTS).join(', ')}.`);
  console.log(`Collecting samples...`);
  const samples = await collectSamples(ctx, { sources, startGw, lastGwId, maxSeason });
  console.log(`  ${samples.length} samples collected.\n`);

  const variantNames = Object.keys(VARIANTS);
  const posNames = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' };
  const weights = {};

  for (const pos of [1, 2, 3, 4]) {
    const posSamples = samples.filter(s => s.pos === pos);
    if (posSamples.length < 50) {
      console.log(`  ${posNames[pos]}: only ${posSamples.length} samples; using equal blend.`);
      weights[pos] = {};
      variantNames.forEach(n => { weights[pos][n] = 1 / variantNames.length; });
      continue;
    }

    // Baseline: per-variant score on its own.
    const baseline = {};
    for (const v of variantNames) {
      if (loss === 'ndcg') {
        baseline[v] = negativeNdcgAtK(posSamples, p => p[v], 11);
      } else {
        const errs = posSamples.map(s => Math.abs(s.preds[v] - s.actual));
        baseline[v] = mae(errs);
      }
    }

    const result = searchWeights(posSamples, variantNames, { loss });
    weights[pos] = {};
    for (const v of variantNames) weights[pos][v] = +result.weights[v].toFixed(3);

    const bestSingle = Math.min(...Object.values(baseline));
    const scoreLabel = loss === 'ndcg' ? '1-NDCG@11' : 'MAE';

    console.log(`  ${posNames[pos]} (n=${posSamples.length})`);
    console.log(`    Per-variant ${scoreLabel}:`);
    for (const v of variantNames) {
      const isBest = baseline[v] === bestSingle;
      console.log(`      ${v.padEnd(18)} ${baseline[v].toFixed(3)}${isBest ? '  (best single)' : ''}`);
    }
    console.log(`    Learned blend ${scoreLabel}: ${result.score.toFixed(3)}`);
    console.log(`    Learned weights:`);
    const sorted = variantNames.map(n => [n, weights[pos][n]]).sort((a, b) => b[1] - a[1]);
    for (const [n, w] of sorted) {
      if (w >= 0.01) console.log(`      ${n.padEnd(18)} ${(w * 100).toFixed(0)}%`);
    }
    console.log('');
  }

  const out = {
    weights,
    variants: Object.keys(VARIANTS),
    train: {
      loss, sources, gws: { start: startGw, end: lastGwId },
      n: samples.length,
      fittedAt: new Date().toISOString()
    }
  };
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2));
  console.log(`Wrote ${OUT}.`);
  console.log(`Run sync.js to bake into api-cache so the browser uses the ensemble.`);
}

main().catch(e => { console.error(e); process.exit(1); });
