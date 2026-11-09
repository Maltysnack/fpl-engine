#!/usr/bin/env node
/**
 * projects/fpl/fit.js
 *
 * Stage-2 OLS calibration trained on every data source available:
 *
 *   1. history-cache.json     (synthesised, current season, single weight)
 *   2. snapshots/gw{N}.json   (canonical, highest fidelity, 2x weight)
 *   3. vaastav-cache.json     (synthesised, historical seasons, 0.5x if lowFidelity)
 *
 * Fits  actual = slope * predicted + intercept + posOffset[pos]  via weighted
 * least squares. Snapshots get 2x sample weight because they carry real ep_next
 * and chance_of_playing values our synthesised data doesn't have. Pre-2021/22
 * vaastav records (no xGI) get 0.5x weight.
 *
 * USAGE
 *   node projects/fpl/fit.js                       # use all available sources
 *   node projects/fpl/fit.js --sources=history     # exclude snapshots/vaastav
 *   node projects/fpl/fit.js --gws=12              # window for current-season sources
 *
 * Writes priors.json. predict.js reads it via ctx.priors; sync.js bakes it into
 * the api-cache for the browser.
 */

const fs    = require('fs');
const path  = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const { predictPlayer } = require('./predict.js');

const CACHE   = path.join(DATA_DIR, 'api-cache.json');
const HCACHE  = path.join(DATA_DIR, 'history-cache.json');
const VCACHE  = path.join(DATA_DIR, 'vaastav-cache.json');
const SNAPDIR = path.join(DATA_DIR, 'snapshots');
const OUT     = path.join(DATA_DIR, 'priors.json');

// Source-specific sample weights. Snapshots dominate because their features are
// real, not reconstructed. Low-fidelity vaastav gets half weight.
const WEIGHT = { snapshot: 2.0, history: 1.0, vaastav: 1.0, 'vaastav-lowfid': 0.5 };

// Per-season recency decay. weight *= DECAY ^ seasons_back. Tuned so 4 seasons
// back contributes ~25% as much as the current season. Addresses the stationarity
// assumption: scoring rule drift, xGI methodology updates, points inflation.
const SEASON_DECAY = 0.7;

// Map a season string like '2023-24' to "years before current". Current season is 0.
// Bumps each summer; uses a fixed reference so multiple runs in one season agree.
function seasonsBack(seasonStr) {
  if (!seasonStr || seasonStr === 'current') return 0;
  const m = /^(\d{4})-/.exec(seasonStr);
  if (!m) return 0;
  const startYear = parseInt(m[1], 10);
  // Reference: 2025-26 season starts in 2025. (Adjust when the calendar rolls.)
  const CURRENT_START_YEAR = 2025;
  return Math.max(0, CURRENT_START_YEAR - startYear);
}

function synthesisePlayer(meta, history, gw, currentMeta) {
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
    ep_next: ep.toFixed(2), form: form.toFixed(2),
    penalties_order: currentMeta && currentMeta.penalties_order,
    direct_freekicks_order: currentMeta && currentMeta.direct_freekicks_order,
    corners_and_indirect_freekicks_order: currentMeta && currentMeta.corners_and_indirect_freekicks_order
  };
}

// Weighted OLS:  actual = slope * pred + intercept.
function fitWeightedOLS(pairs) {
  const n = pairs.length;
  if (n < 30) return { slope: 1.0, intercept: 0.0, n };
  let W = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const p of pairs) {
    const w = p.weight || 1;
    W   += w;
    sx  += w * p.pred;
    sy  += w * p.act;
    sxx += w * p.pred * p.pred;
    sxy += w * p.pred * p.act;
  }
  const denom = W * sxx - sx * sx;
  if (denom === 0) return { slope: 1.0, intercept: 0.0, n };
  const slope = (W * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / W;
  // 30% shrinkage toward identity (slope=1, intercept=0). Calibrated on the
  // walk-forward result: heavier shrinkage hurt MAE without meaningfully
  // helping rank correlation; lighter shrinkage let the slope crash on noisy
  // training windows.
  const SHRINK = 0.30;
  return {
    slope:     +(slope     * (1 - SHRINK) + 1.0 * SHRINK).toFixed(3),
    intercept: +(intercept * (1 - SHRINK) + 0.0 * SHRINK).toFixed(3),
    n
  };
}

const mae  = ps => ps.reduce((s, p) => s + Math.abs(p.pred - p.act), 0) / ps.length;
const bias = ps => ps.reduce((s, p) => s + (p.act - p.pred), 0) / ps.length;

function collectSynthesis(cache, sourceTag, elemById, startGw, endGw, ctx, maxSeason) {
  const out = [];
  for (const pid in cache.history) {
    const entry = cache.history[pid];
    // gh#39 clean-holdout: skip seasons after maxSeason (string compare works
    // on the YYYY-YY format). null maxSeason = no filtering.
    if (maxSeason && (entry.season || '') > maxSeason) continue;
    for (let gw = startGw; gw <= endGw; gw++) {
      const gwEntries = entry.history.filter(h => h.round === gw);
      if (!gwEntries.length) continue;
      const totalMins = gwEntries.reduce((s, h) => s + h.minutes, 0);
      if (totalMins < 45) continue;
      const totalPts  = gwEntries.reduce((s, h) => s + h.total_points, 0);

      const synth = synthesisePlayer(
        { id: pid, web_name: entry.web_name, team: entry.team, element_type: entry.element_type },
        entry.history, gw, elemById ? elemById[pid] : null
      );
      if (!synth) continue;
      const fixList = gwEntries.map(h => ({
        team_h: h.was_home ? entry.team : h.opponent_team,
        team_a: h.was_home ? h.opponent_team : entry.team,
        team_h_difficulty: 3, team_a_difficulty: 3
      }));
      const pred = predictPlayer(synth, fixList, ctx);
      if (pred < 0.5) continue;
      const baseW   = entry.lowFidelity ? WEIGHT['vaastav-lowfid'] : WEIGHT[sourceTag];
      const decayW  = Math.pow(SEASON_DECAY, seasonsBack(entry.season || 'current'));
      const weight  = baseW * decayW;
      out.push({ pred, act: totalPts, pos: synth.element_type, source: sourceTag, weight, season: entry.season || 'current' });
    }
  }
  return out;
}

async function collectSnapshots(ctx) {
  if (!fs.existsSync(SNAPDIR)) return [];
  const files = fs.readdirSync(SNAPDIR).filter(f => f.endsWith('.json'));
  const out = [];
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
      const pred = predictPlayer(p, fix, ctx);
      if (pred < 0.5) continue;
      out.push({ pred, act: a.pts, pos: p.element_type, source: 'snapshot', weight: WEIGHT.snapshot });
    }
  }
  return out;
}

async function main() {
  const args  = process.argv.slice(2);
  const gwArg = args.find(a => a.startsWith('--gws='));
  const K     = gwArg ? parseInt(gwArg.split('=')[1], 10) : 10;
  const sArg  = args.find(a => a.startsWith('--sources='));
  const sources = sArg ? sArg.split('=')[1].split(',') : ['history', 'snapshots', 'vaastav'];
  const msArg = args.find(a => a.startsWith('--max-season='));
  const maxSeason = msArg ? msArg.split('=')[1] : null;   // gh#39 clean-holdout

  if (!fs.existsSync(CACHE)) { console.error(`Missing ${CACHE}. Run sync.js first.`); process.exit(1); }
  const cache    = JSON.parse(fs.readFileSync(CACHE, 'utf8'));
  const elemById = {};
  cache.elements.forEach(e => { elemById[e.id] = e; });

  const ctx = {
    calibration:  (cache.calibration && cache.calibration.factors) || null,
    teamStrength: cache.teamStrength || null
  };

  const lastGwId = cache.lastGwId;
  const startGw  = Math.max(2, lastGwId - K + 1);

  console.log(`Fitting stage-2 priors. Sources: ${sources.join(', ')}.`);
  console.log(`Window for current-season sources: GW${startGw}..GW${lastGwId}.\n`);

  const all = [];

  if (sources.includes('history') && fs.existsSync(HCACHE)) {
    const hcache = JSON.parse(fs.readFileSync(HCACHE, 'utf8'));
    const samp = collectSynthesis(hcache, 'history', elemById, startGw, lastGwId, ctx, maxSeason);
    console.log(`history-cache: ${samp.length} samples (weight ${WEIGHT.history}x)`);
    all.push(...samp);
  }
  if (sources.includes('snapshots')) {
    const samp = await collectSnapshots(ctx);
    console.log(`snapshots:     ${samp.length} samples (weight ${WEIGHT.snapshot}x)`);
    all.push(...samp);
  }
  if (sources.includes('vaastav') && fs.existsSync(VCACHE)) {
    const vcache = JSON.parse(fs.readFileSync(VCACHE, 'utf8'));
    const samp = collectSynthesis(vcache, 'vaastav', null, 2, 38, ctx, maxSeason);
    const lowfid = samp.filter(s => s.weight === WEIGHT['vaastav-lowfid']).length;
    console.log(`vaastav:       ${samp.length} samples (${samp.length - lowfid}@${WEIGHT.vaastav}x, ${lowfid}@${WEIGHT['vaastav-lowfid']}x)`);
    all.push(...samp);
  }

  if (all.length < 100) {
    console.error(`\nOnly ${all.length} training pairs; need >=100. Run history-fetch / vaastav-fetch to widen.`);
    process.exit(1);
  }

  console.log(`\nTotal training pairs: ${all.length}.`);

  // Show effective per-season weight after both source weight and recency decay.
  const bySeason = {};
  for (const p of all) {
    const k = p.season || 'current';
    if (!bySeason[k]) bySeason[k] = { n: 0, w: 0 };
    bySeason[k].n += 1;
    bySeason[k].w += p.weight || 1;
  }
  console.log('Per-season effective weight share:');
  const totalW = Object.values(bySeason).reduce((s, v) => s + v.w, 0);
  for (const k of Object.keys(bySeason).sort()) {
    const v = bySeason[k];
    console.log(`  ${k.padEnd(10)} n=${String(v.n).padStart(5)}  ` +
      `mean_w=${(v.w / v.n).toFixed(2)}  share=${(100 * v.w / totalW).toFixed(0)}%`);
  }

  const fit = fitWeightedOLS(all);
  console.log(`\nGlobal OLS (weighted, n=${fit.n}): slope=${fit.slope}, intercept=${fit.intercept}`);

  const posOffset = {};
  const posNames = { 1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD' };
  for (const pos of [1, 2, 3, 4]) {
    const pairs = all.filter(p => p.pos === pos);
    if (pairs.length < 20) { posOffset[pos] = 0; continue; }
    let wSum = 0, rSum = 0;
    for (const p of pairs) {
      const res = p.act - (fit.slope * p.pred + fit.intercept);
      rSum += res * (p.weight || 1);
      wSum += (p.weight || 1);
    }
    const meanRes = wSum > 0 ? rSum / wSum : 0;
    const SHRINK = 0.4;
    posOffset[pos] = +(meanRes * (1 - SHRINK)).toFixed(3);
    console.log(`  ${posNames[pos]} (n=${pairs.length}): residual offset = ${posOffset[pos]}`);
  }

  const beforeMAE  = mae(all);
  const corrected  = all.map(p => ({ ...p, pred: fit.slope * p.pred + fit.intercept + (posOffset[p.pos] || 0) }));
  const afterMAE   = mae(corrected);
  const beforeBias = bias(all);
  const afterBias  = bias(corrected);

  console.log(`\nTraining MAE:  ${beforeMAE.toFixed(2)} -> ${afterMAE.toFixed(2)} (delta ${(afterMAE - beforeMAE).toFixed(2)})`);
  console.log(`Training bias: ${beforeBias >= 0 ? '+' : ''}${beforeBias.toFixed(2)} -> ${afterBias >= 0 ? '+' : ''}${afterBias.toFixed(2)}`);

  // Per-source MAE pre/post for honesty.
  console.log('\nPer-source MAE before -> after:');
  const sources2 = [...new Set(all.map(p => p.source))];
  for (const src of sources2) {
    const a = all.filter(p => p.source === src);
    const b = corrected.filter(p => p.source === src);
    if (!a.length) continue;
    console.log(`  ${src.padEnd(10)}: ${mae(a).toFixed(2)} -> ${mae(b).toFixed(2)}  (n=${a.length})`);
  }

  const priors = {
    slope: fit.slope,
    intercept: fit.intercept,
    posOffset,
    train: {
      sources: sources2,
      gws: { start: startGw, end: lastGwId },
      n: all.length,
      maeBefore: +beforeMAE.toFixed(3),
      maeAfter: +afterMAE.toFixed(3),
      biasBefore: +beforeBias.toFixed(3),
      biasAfter: +afterBias.toFixed(3),
      fittedAt: new Date().toISOString()
    }
  };
  fs.writeFileSync(OUT, JSON.stringify(priors, null, 2));
  console.log(`\nWrote ${OUT}. Run sync.js to bake priors into api-cache for the browser.`);
}

main().catch(e => { console.error(e); process.exit(1); });
