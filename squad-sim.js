#!/usr/bin/env node
/**
 * projects/fpl/squad-sim.js
 *
 * Realistic squad-based season simulator. Models real FPL constraints:
 *   - Start with a 15-man squad (2 GKP + 5 DEF + 5 MID + 3 FWD) within a £100m budget.
 *   - At most 3 players from the same club.
 *   - 1 free transfer per gameweek, banking up to 5 (2024-25 rule, issue #34).
 *   - Every extra transfer costs -4 pts ("hit").
 *   - Chips: Triple Captain and Bench Boost (each used once per season).
 *   - 4-4-2 starting formation, captain doubled.
 *
 * Decision logic at each GW:
 *   1. Look at all (out, in) transfer pairs that respect budget + club + position constraints.
 *   2. Score each pair by 3-GW lookahead: sum of predicted points over the next 3 GWs
 *      for the incoming player minus the outgoing player.
 *   3. If the best pair's delta exceeds the hit cost (or is positive when transfers are free),
 *      execute it. Otherwise hold and bank the free transfer.
 *
 * Limitation: this is a SINGLE-TRANSFER greedy. Real FPL planners also do 2-transfer
 * swaps and chip combos. Single is realistic for ~80% of weeks; multi-transfer logic
 * would add another order of complexity.
 *
 * USAGE
 *   node projects/fpl/squad-sim.js                          # 2024-25, ensemble variant
 *   node projects/fpl/squad-sim.js --season=2023-24
 *   node projects/fpl/squad-sim.js --variant=xg_underlying
 *   node projects/fpl/squad-sim.js --variants=default,xg_pure,ensemble
 *   node projects/fpl/squad-sim.js --lookahead=5
 *   node projects/fpl/squad-sim.js --hit-threshold=8       # less aggressive transfers
 */

const fs   = require('fs');
const path = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const predictApi              = require('./predict.js');
const { predictPlayer }       = predictApi;
const { VARIANTS, DEFAULT_BLEND, predictEnsembleRouted } = require('./models.js');
const { plan: planMove }      = require('./planner.js');
const { decideChipsThisGw, initChipState, halfOf, wcShouldFire, valueChipFire, DEFAULTS: CHIPLIB_DEFAULTS } = require('./chips.js');

const CACHE   = path.join(DATA_DIR, 'api-cache.json');
const VCACHE  = path.join(DATA_DIR, 'vaastav-cache.json');
const EWEIGHT = path.join(DATA_DIR, 'ensemble-weights.json');
const NCACHE  = path.join(DATA_DIR, 'news-cache.json');

// Load FPL + BBC news flags from news-cache.json if present. Returns a map
// { [player_id]: { chance, news, source } } compatible with predict.js's
// ctx.newsOverride. Returns null if the file doesn't exist or is stale.
// Caveat: news flags reflect the CURRENT live season. They are valid signal
// only for live play (or for simulating "what if today's flags had applied").
// For pure historical backtesting (Vaastav past seasons), news shouldn't be
// applied because the injury reality was different at the time.
function loadNewsCache() {
  if (!fs.existsSync(NCACHE)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(NCACHE, 'utf8'));
    return data.flags || null;
  } catch (e) { return null; }
}

const BUDGET_TOTAL = 1000;   // £100.0m
const MAX_PER_CLUB = 3;
// Set by simulate() per-run to allow `one_per_club` (and any future variant)
// to bound the club cap without threading the value through every callsite.
let _activeMaxPerClub = MAX_PER_CLUB;
const HIT_COST     = 4;
const SQUAD_QUOTA  = { 1: 2, 2: 5, 3: 5, 4: 3 };
const FORMATION    = { 1: 1, 2: 4, 3: 4, 4: 2 };   // 4-4-2

const BENCHMARKS = [
  { pct: 1, score: 2700 }, { pct: 5, score: 2400 },
  { pct: 10, score: 2350 }, { pct: 20, score: 2250 },
  { pct: 35, score: 2180 }, { pct: 50, score: 2100 }
];
function estimatePercentile(t) {
  if (t >= BENCHMARKS[0].score) return Math.max(0.1, 1 - (t - BENCHMARKS[0].score) / 300);
  for (let i = 0; i < BENCHMARKS.length - 1; i++) {
    const a = BENCHMARKS[i], b = BENCHMARKS[i + 1];
    if (t <= a.score && t >= b.score) return +(a.pct + (b.pct - a.pct) * ((a.score - t) / (a.score - b.score))).toFixed(1);
  }
  const last = BENCHMARKS[BENCHMARKS.length - 1];
  return Math.min(99, +(last.pct + (last.score - t) * 0.15).toFixed(1));
}

function synthesisePlayer(meta, history, gw) {
  const past = history.filter(h => h.round < gw);
  if (past.length === 0) return null;
  const minutes = past.reduce((s, h) => s + h.minutes, 0);
  if (minutes < 1) return null;
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

function predictWithVariant(player, fixList, ctx, variantName, ensembleWeights) {
  // 'ensemble' uses per-player ARCHETYPE ROUTING (step 8): a different variant
  // blend per player archetype. Falls back to position-weighted ensemble when
  // archetype routing returns 0 (shouldn't happen, but defensive).
  if (variantName === 'ensemble') {
    return predictEnsembleRouted(player, fixList, ctx, ensembleWeights);
  }
  return predictPlayer(player, fixList, Object.assign({}, ctx, { variant: VARIANTS[variantName] }));
}

// Sum predicted points across the next `horizon` GWs (1-indexed from gw).
// Uses the current synthesised state (we don't re-synthesise per future GW;
// approximation that lookahead uses today's signals to score future fixtures).
function lookaheadPredict(player, entry, gw, horizon, ctx, variantName, ew) {
  const synth = synthesisePlayer(
    { id: player.id, web_name: player.web_name, team: player.team, element_type: player.element_type },
    entry.history, gw
  );
  if (!synth) return 0;
  let total = 0;
  for (let h = 0; h < horizon; h++) {
    const targetGw = gw + h;
    const fixturesThisGw = entry.history.filter(e => e.round === targetGw);
    if (!fixturesThisGw.length) continue;   // blank GW for this team
    const fixList = fixturesThisGw.map(f => ({
      team_h: f.was_home ? player.team : f.opponent_team,
      team_a: f.was_home ? f.opponent_team : player.team,
      team_h_difficulty: 3, team_a_difficulty: 3
    }));
    total += predictWithVariant(synth, fixList, ctx, variantName, ew);
  }
  return total;
}

// Find the same player's record in the previous season's vaastav data so we can
// seed GW1 predictions with real prior signal instead of lookahead. Matching is
// by normalised web_name + team_short_name (FPL element IDs reset across seasons).
// Returns the previous season's entry or null.
function findPreviousSeasonEntry(player, vcache, currentSeason) {
  // Compute previous season string. e.g. '2024-25' -> '2023-24'.
  const m = /^(\d{4})-(\d{2})$/.exec(currentSeason);
  if (!m) return null;
  const yA = parseInt(m[1], 10) - 1;
  const yB = parseInt(m[2], 10) - 1;
  const prevSeason = `${yA}-${String(yB).padStart(2, '0')}`;

  const target = (player.web_name || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  for (const key in vcache.history) {
    const entry = vcache.history[key];
    if (entry.season !== prevSeason) continue;
    const name = (entry.web_name || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
    if (name === target) return entry;
  }
  return null;
}

// Build a synthetic pre-GW1 state for a player using LAST SEASON's data.
// If no previous-season match exists (new signings, promoted club players),
// return null and the caller falls back to a position-based default.
function buildPriorState(player, vcache, currentSeason) {
  const prev = findPreviousSeasonEntry(player, vcache, currentSeason);
  if (!prev) return null;
  const totalMins  = prev.history.reduce((s, h) => s + h.minutes, 0);
  if (totalMins < 90) return null;
  const totalPts   = prev.history.reduce((s, h) => s + h.total_points, 0);
  const goals      = prev.history.reduce((s, h) => s + h.goals_scored, 0);
  const assists    = prev.history.reduce((s, h) => s + h.assists, 0);
  const csheets    = prev.history.reduce((s, h) => s + h.clean_sheets, 0);
  const saves      = prev.history.reduce((s, h) => s + h.saves, 0);
  const bps        = prev.history.reduce((s, h) => s + h.bps, 0);
  const bonus      = prev.history.reduce((s, h) => s + h.bonus, 0);
  const xgi        = prev.history.reduce((s, h) => s + (h.expected_goal_involvements || 0), 0);
  const xgc        = prev.history.reduce((s, h) => s + (h.expected_goals_conceded    || 0), 0);
  const last5      = prev.history.slice(-5);
  const last4      = prev.history.slice(-4);
  const form       = last5.length ? last5.reduce((s, h) => s + h.total_points, 0) / last5.length : 0;
  const ep         = last4.length ? last4.reduce((s, h) => s + h.total_points, 0) / last4.length : 0;
  return {
    id: player.id, web_name: player.web_name, team: player.team, element_type: player.element_type,
    chance_of_playing_next_round: 100,
    minutes: totalMins,
    total_points: totalPts,
    goals_scored: goals, assists, clean_sheets: csheets, saves, bps, bonus,
    expected_goal_involvements_per_90: +(xgi / (totalMins / 90)).toFixed(3),
    expected_goals_conceded_per_90:    +(xgc / (totalMins / 90)).toFixed(3),
    ep_next: ep.toFixed(2), form: form.toFixed(2)
  };
}

// Score every player with a unified "initial-pick rank" using previous-season
// signals (model prior + leaderboard rank + template ownership + dream team).
// Returns array of { ...player, predicted, price }. Caller decides how to
// assemble those into a 15-man squad (the SQUAD STRATEGIES below).
function scoreInitialPicks(players, ctx, variantName, ew, vcache, season) {
  const scored = [];
  const FALLBACK = { 1: 3.0, 2: 2.5, 3: 3.0, 4: 3.5 };

  // Previous-season lookup.
  const m = /^(\d{4})-(\d{2})$/.exec(season);
  const prevSeason = m ? `${parseInt(m[1], 10) - 1}-${String(parseInt(m[2], 10) - 1).padStart(2, '0')}` : null;
  const prevByName = {};
  if (prevSeason) {
    for (const k in vcache.history) {
      const e = vcache.history[k];
      if (e.season !== prevSeason) continue;
      const norm = (e.web_name || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
      prevByName[norm] = e;
    }
  }
  // Leaderboard rank within position by previous-season total points.
  const prevByPos = { 1: [], 2: [], 3: [], 4: [] };
  for (const norm in prevByName) {
    const e = prevByName[norm];
    if (prevByPos[e.element_type]) prevByPos[e.element_type].push({ norm, pts: e.total_points_season || 0 });
  }
  for (const pos of [1, 2, 3, 4]) prevByPos[pos].sort((a, b) => b.pts - a.pts);
  const rankByNorm = {};
  for (const pos of [1, 2, 3, 4]) {
    prevByPos[pos].forEach((x, i) => { rankByNorm[`${pos}:${x.norm}`] = i + 1; });
  }

  for (const p of players) {
    const first = p.history[0];
    if (!first || first.value <= 0) continue;
    const norm = (p.web_name || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
    const prev = prevByName[norm];

    // Team-continuity check. Previous-season stats only carry the player's old
    // context (teammates, manager, role). If they moved clubs, the prior is
    // overstating their fit at the new team. Down-weight by 50% in that case.
    const teamContinuity = prev ? (prev.team === p.team ? 1.0 : 0.5) : 1.0;

    let modelPred;
    if (prev) {
      const synth = buildPriorState(p, vcache, season);
      if (synth) {
        const fixList = [{
          team_h: first.was_home ? p.team : first.opponent_team,
          team_a: first.was_home ? first.opponent_team : p.team,
          team_h_difficulty: 3, team_a_difficulty: 3
        }];
        modelPred = predictWithVariant(synth, fixList, ctx, variantName, ew) * teamContinuity;
      }
    }
    if (modelPred == null) modelPred = FALLBACK[p.element_type] || 2.5;

    // Leaderboard signal scaled by team continuity.
    const rank = rankByNorm[`${p.element_type}:${norm}`];
    let leaderboardBonus = 0;
    if (rank) {
      if      (rank <= 5)  leaderboardBonus = 1.5;
      else if (rank <= 15) leaderboardBonus = 0.7;
      else if (rank <= 30) leaderboardBonus = 0.3;
    }
    leaderboardBonus *= teamContinuity;

    // Ownership signal: high-ownership players are "template."
    const ownership = prev ? (prev.selected_by_percent || 0) : 0;
    const templateBonus = ownership > 30 ? 0.5 : ownership > 15 ? 0.25 : 0;

    // Dream-team signal: proven elite based on previous-season Team-of-the-Week
    // appearances. 10+ TOTW = "Salah-tier" anchor pick.
    const dtc = prev ? (prev.dreamteam_count || 0) : 0;
    const dreamTeamBonus = Math.min(1.5, dtc * 0.10) * teamContinuity;

    const predicted = modelPred + leaderboardBonus + templateBonus + dreamTeamBonus;
    const ownership_pct = prev ? (prev.selected_by_percent || 0) : 0;
    scored.push({ ...p, predicted, price: first.value, history: p.history, ownership_pct });
  }
  return scored;
}

// ─── SQUAD STRATEGIES ────────────────────────────────────────────────────────
//
// Each strategy is a function that takes scored players + constraints and
// returns { squad, budget, clubCount } or null. The tournament cross-products
// these against the in-season variants to find the best total approach.

function tryAddTo(squad, p, clubCount, budget, filledByPos, maxPerClub = _activeMaxPerClub) {
  if ((clubCount[p.team] || 0) >= maxPerClub) return null;
  if (filledByPos[p.element_type] >= SQUAD_QUOTA[p.element_type]) return null;
  const remaining = estimateRemainingCost(squad, p.element_type, filledByPos[p.element_type] + 1);
  if (p.price > budget - remaining) return null;
  squad.push(p);
  clubCount[p.team] = (clubCount[p.team] || 0) + 1;
  filledByPos[p.element_type]++;
  return budget - p.price;
}

function backfillCheapest(squad, scored, clubCount, budget, filledByPos) {
  // Issue #30 (closed negative): tried sub-aware composite ranking (predicted
  // * 0.6 - price * 0.001) for bench picks. Zero effect across 4 variants ×
  // 3 seasons. The cheapest available players in their price tier were
  // already the highest-predicted (rotation risk and price both correlate
  // with low predicted points), so the new sort produced the same picks.
  // Go-live Gate 3: among equal-price bench candidates prefer the one who
  // projects to play (predicted desc as the tie-break) and skip predicted <= 0
  // while any positive candidate exists. In the backtest every scored player
  // carries a positive fallback prediction so this is inert there (issue #30);
  // it matters for live rebuilds where dead assets predict exactly 0.
  for (const pos of [1, 2, 3, 4]) {
    while (filledByPos[pos] < SQUAD_QUOTA[pos]) {
      let candidates = scored.filter(s => s.element_type === pos && !squad.find(sq => sq.id === s.id));
      const alive = candidates.filter(s => (s.predicted || 0) > 0);
      if (alive.length) candidates = alive;
      candidates.sort((a, b) => a.price - b.price || (b.predicted || 0) - (a.predicted || 0));
      let added = false;
      for (const p of candidates) {
        const newBudget = tryAddTo(squad, p, clubCount, budget, filledByPos);
        if (newBudget !== null) { budget = newBudget; added = true; break; }
      }
      if (!added) return { squad, budget, clubCount, filledByPos, complete: false };
    }
  }
  return { squad, budget, clubCount, filledByPos, complete: true };
}

// ─ Strategy: balanced ─ greedy fill by predicted, multi-pass premium-then-value.
function squadBalanced(scored) {
  const squad = []; const clubCount = {}; let budget = BUDGET_TOTAL;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  // Pass 1: premium anchors (top-2 MID, top-2 FWD, top-1 DEF by predicted).
  const ANCHOR = { 3: 2, 4: 2, 2: 1, 1: 0 };
  for (const pos of [4, 3, 2]) {
    const sorted = scored.filter(s => s.element_type === pos).sort((a, b) => b.predicted - a.predicted);
    let picked = 0;
    for (const p of sorted) {
      if (picked >= ANCHOR[pos]) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  // Pass 2: value fill.
  for (const pos of [1, 2, 3, 4]) {
    const remaining = SQUAD_QUOTA[pos] - filledByPos[pos];
    if (remaining <= 0) continue;
    const candidates = scored.filter(s => s.element_type === pos && !squad.find(sq => sq.id === s.id));
    candidates.sort((a, b) => (b.predicted / Math.max(1, b.price)) - (a.predicted / Math.max(1, a.price)));
    let added = 0;
    for (const p of candidates) {
      if (added >= remaining) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; added++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

// ─ Strategy: premium_loaded ─ maximize premiums (2 MID + 2 FWD + 1 DEF >= price thresholds).
function squadPremiumLoaded(scored) {
  const squad = []; const clubCount = {}; let budget = BUDGET_TOTAL;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  // PREMIUM = top of leaderboard, ignore price ceiling within reason.
  const TARGETS = [
    { pos: 3, count: 2, minPrice: 90 },   // 2 premium MIDs >= £9.0m
    { pos: 4, count: 2, minPrice: 80 },   // 2 premium FWDs >= £8.0m
    { pos: 2, count: 1, minPrice: 60 }    // 1 premium DEF >= £6.0m
  ];
  for (const t of TARGETS) {
    const cands = scored.filter(s => s.element_type === t.pos && s.price >= t.minPrice).sort((a, b) => b.predicted - a.predicted);
    let picked = 0;
    for (const p of cands) {
      if (picked >= t.count) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

// ─ Strategy: value ─ avoid premiums, distribute budget evenly.
function squadValue(scored) {
  const squad = []; const clubCount = {}; let budget = BUDGET_TOTAL;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  // No premiums: cap price at "mid-tier" thresholds. Prefer predicted/price ratio.
  const CAPS = { 1: 55, 2: 65, 3: 80, 4: 75 };
  for (const pos of [1, 2, 3, 4]) {
    const cands = scored.filter(s => s.element_type === pos && s.price <= CAPS[pos]);
    cands.sort((a, b) => (b.predicted / Math.max(1, b.price)) - (a.predicted / Math.max(1, a.price)));
    let picked = 0;
    for (const p of cands) {
      if (picked >= SQUAD_QUOTA[pos]) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

// ─ Strategy: differential ─ avoid template, prefer low ownership + good predicted.
function squadDifferential(scored) {
  const squad = []; const clubCount = {}; let budget = BUDGET_TOTAL;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  // Lower ownership is rewarded; high ownership is penalised.
  const ranked = scored.map(s => ({
    ...s,
    differentialScore: s.predicted - (s.ownership_pct || 0) * 0.05
  })).sort((a, b) => b.differentialScore - a.differentialScore);
  for (const pos of [4, 3, 2, 1]) {
    const cands = ranked.filter(s => s.element_type === pos);
    let picked = 0;
    for (const p of cands) {
      if (picked >= SQUAD_QUOTA[pos]) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

// ─ Strategy: template ─ pure template, sort by previous-season ownership only.
function squadTemplate(scored) {
  const squad = []; const clubCount = {}; let budget = BUDGET_TOTAL;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  const ranked = [...scored].sort((a, b) => (b.ownership_pct || 0) - (a.ownership_pct || 0));
  for (const pos of [4, 3, 2, 1]) {
    const cands = ranked.filter(s => s.element_type === pos);
    let picked = 0;
    for (const p of cands) {
      if (picked >= SQUAD_QUOTA[pos]) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

const SQUAD_STRATEGIES = {
  balanced:       squadBalanced,
  premium_loaded: squadPremiumLoaded,
  value:          squadValue,
  differential:   squadDifferential,
  template:       squadTemplate
};

// Master entry. `strategy` selects from SQUAD_STRATEGIES; defaults to 'balanced'.
// gh#35 Wildcard rebuild. Build a fresh 15-man squad from scratch within the
// available budget (current squad sale value + bank). Greedy fill, same shape
// as the 'balanced' initial-squad strategy but with a caller-supplied budget.
// `scored` items need { id, element_type, team, price, predicted }.
function wildcardRebuild(scored, budgetTotal, priceLambda) {
  // gh#43: λ set → tilted construction. WC rescue passes opts.priceLambda;
  // bestFreeHitXI passes nothing, so Free Hit is intentionally left untilted.
  if (priceLambda != null) return buildTilted(scored, priceLambda, budgetTotal);
  const squad = []; const clubCount = {}; let budget = budgetTotal;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  // Pass 1: premium anchors (top-2 MID, top-2 FWD, top-1 DEF by predicted).
  const ANCHOR = { 3: 2, 4: 2, 2: 1, 1: 0 };
  for (const pos of [4, 3, 2]) {
    const sorted = scored.filter(s => s.element_type === pos).sort((a, b) => b.predicted - a.predicted);
    let picked = 0;
    for (const p of sorted) {
      if (picked >= ANCHOR[pos]) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  // Pass 2: fill remaining slots by points-per-million (gh#43). Raw-points
  // greedy is the textbook-bad knapsack heuristic, it strands the budget on
  // one expensive pick. Ratio greedy spends efficiently; cost-floor reservation
  // in tryAddTo still guarantees enough budget for cheap backfill. Mirrors the
  // anchors-then-ratio-fill pattern squadBalanced uses.
  for (const pos of [1, 2, 3, 4]) {
    const remaining = SQUAD_QUOTA[pos] - filledByPos[pos];
    if (remaining <= 0) continue;
    const candidates = scored.filter(s => s.element_type === pos && !squad.find(sq => sq.id === s.id));
    candidates.sort((a, b) => (b.predicted / Math.max(1, b.price)) - (a.predicted / Math.max(1, a.price)));
    let added = 0;
    for (const p of candidates) {
      if (added >= remaining) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; added++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

// Best legal-XI predicted total from a 15-man squad. scoreOf(id) -> number.
// Mandatory minimums (1 GK, 3 DEF, 2 MID, 1 FWD) plus the 4 best remaining
// outfield players. Used by the WC trigger to gauge squad health.
function bestXIPred(squad15, scoreOf) {
  const byPos = { 1: [], 2: [], 3: [], 4: [] };
  for (const p of squad15) byPos[p.element_type].push(scoreOf(p.id));
  for (const k of [1, 2, 3, 4]) byPos[k].sort((a, b) => b - a);
  let total = (byPos[1][0] || 0)
            + byPos[2].slice(0, 3).reduce((s, x) => s + x, 0)
            + byPos[3].slice(0, 2).reduce((s, x) => s + x, 0)
            + (byPos[4][0] || 0);
  const rest = [...byPos[2].slice(3), ...byPos[3].slice(2), ...byPos[4].slice(1)]
    .sort((a, b) => b - a).slice(0, 4);
  return total + rest.reduce((s, x) => s + x, 0);
}

// gh#43: convex price-tilted player value. `price` is in 0.1m units; convert
// to £m so λ is human-readable. λ=0 → raw predicted points; λ>0 penalizes
// expensive players super-linearly, which at fixed total budget favours a
// spread squad over a stacked one (Σprice² is larger when budget is stacked).
function tiltedValue(predicted, price, lambda) {
  const pm = price / 10;
  return predicted - lambda * pm * pm;
}

// gh#43: λ-tilted squad construction. Greedy fill of the 15-man quota, every
// position ranked by tiltedValue. Mirrors squadValue's structure but with the
// tilt as the sole ranking key and no per-position price caps, λ is the
// continuous concentration control that supersedes those ad-hoc caps.
// `scored` items need { id, element_type, team, price, predicted }.
function buildTilted(scored, lambda, budgetTotal) {
  const squad = []; const clubCount = {}; let budget = budgetTotal;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  for (const pos of [1, 2, 3, 4]) {
    const cands = scored.filter(s => s.element_type === pos).sort((a, b) =>
      tiltedValue(b.predicted, b.price, lambda) - tiltedValue(a.predicted, a.price, lambda));
    let picked = 0;
    for (const p of cands) {
      if (picked >= SQUAD_QUOTA[pos]) break;
      const nb = tryAddTo(squad, p, clubCount, budget, filledByPos);
      if (nb !== null) { budget = nb; picked++; }
    }
  }
  const out = backfillCheapest(squad, scored, clubCount, budget, filledByPos);
  return out.complete ? { squad: out.squad, budget: out.budget, clubCount: out.clubCount } : null;
}

function buildInitialSquad(players, gw, ctx, variantName, ew, vcache, season, strategy, priceLambda) {
  const scored = scoreInitialPicks(players, ctx, variantName, ew, vcache, season);
  // gh#43: λ set → tilted construction; absent → existing strategy, unchanged.
  if (priceLambda != null) return buildTilted(scored, priceLambda, BUDGET_TOTAL);
  const builder = SQUAD_STRATEGIES[strategy] || SQUAD_STRATEGIES.balanced;
  return builder(scored);
}

// Rough cost-floor estimate so the greedy picker doesn't run out of budget.
// Reserves the minimum sensible budget for remaining positions/slots.
function estimateRemainingCost(squadSoFar, currentPos, filledInPos) {
  // Cheapest sensible prices per position (in 0.1m units).
  const MIN_PRICE = { 1: 40, 2: 40, 3: 45, 4: 45 };
  let need = 0;
  const filledByPos = { 1: 0, 2: 0, 3: 0, 4: 0 };
  squadSoFar.forEach(p => { filledByPos[p.element_type]++; });
  if (currentPos !== null) filledByPos[currentPos] = filledInPos;
  for (const pos of [1, 2, 3, 4]) {
    need += Math.max(0, SQUAD_QUOTA[pos] - filledByPos[pos]) * MIN_PRICE[pos];
  }
  return need;
}

// gh#44: the 8 legal FPL formations (GK always 1, outfield 10, DEF 3-5,
// MID 2-5, FWD 1-3). Keyed by name; value is the per-position XI count.
const LEGAL_FORMATIONS = {
  '3-4-3': { 1: 1, 2: 3, 3: 4, 4: 3 },
  '3-5-2': { 1: 1, 2: 3, 3: 5, 4: 2 },
  '4-3-3': { 1: 1, 2: 4, 3: 3, 4: 3 },
  '4-4-2': { 1: 1, 2: 4, 3: 4, 4: 2 },
  '4-5-1': { 1: 1, 2: 4, 3: 5, 4: 1 },
  '5-2-3': { 1: 1, 2: 5, 3: 2, 4: 3 },
  '5-3-2': { 1: 1, 2: 5, 3: 3, 4: 2 },
  '5-4-1': { 1: 1, 2: 5, 3: 4, 4: 1 },
};
const FORMATION_NAMES = Object.keys(LEGAL_FORMATIONS);
const POS_MIN = { 2: 3, 3: 2, 4: 1 };   // outfield XI minimums
const POS_MAX = { 2: 5, 3: 5, 4: 3 };   // outfield XI maximums

// gh#44: legality of an auto-sub, bring a bench player on line posY in for a
// blanked XI player on line posX, given the XI's current per-line counts. A GK
// can only be replaced by a GK. A same-position sub never changes line counts.
// A cross-position sub is legal iff the outgoing line stays >= its minimum and
// the incoming line stays <= its maximum.
function isLegalSub(posX, posY, counts) {
  if (posX === 1 || posY === 1) return posX === 1 && posY === 1;
  if (posX === posY) return true;
  return counts[posX] > POS_MIN[posX] && counts[posY] < POS_MAX[posY];
}

function permute(arr) {
  if (arr.length <= 1) return [arr];
  const res = [];
  for (let i = 0; i < arr.length; i++) {
    const rest = arr.slice(0, i).concat(arr.slice(i + 1));
    for (const p of permute(rest)) res.push([arr[i], ...p]);
  }
  return res;
}

// Nominate the XI for a fixed formation: top-N predicted per position line.
// byPos must be pre-sorted by predicted descending. Bench is position-grouped
// [GK, DEF, MID, FWD], the legacy ordering.
function nominateXI(byPos, formation) {
  const xi = [], bench = [];
  for (const pos of [1, 2, 3, 4]) {
    xi.push(...byPos[pos].slice(0, formation[pos]));
    bench.push(...byPos[pos].slice(formation[pos]));
  }
  return { xi, bench };
}

// Run the FPL auto-sub cascade on a nominated XI using ACTUAL minutes. For each
// XI player who got 0 minutes, bring in the first eligible bench player (one who
// played) whose entry keeps the XI legal. `samePositionFirst` mirrors the legacy
// heuristic, prefer a same-position sub over a legal cross-position one; when
// false the bench is consumed in strict order (already optimised upstream).
function runAutoSubs(xi, bench, samePositionFirst) {
  const counts = { 2: 0, 3: 0, 4: 0 };
  for (const p of xi) if (p.element_type > 1) counts[p.element_type]++;
  const xiFinal = [...xi];
  const benchRem = [...bench];
  let subsFired = 0, blanksUncovered = 0;
  for (let i = 0; i < xiFinal.length; i++) {
    if ((xiFinal[i].minutes || 0) > 0) continue;
    const posX = xiFinal[i].element_type;
    let subIdx = -1;
    if (samePositionFirst) {
      subIdx = benchRem.findIndex(b => b && (b.minutes || 0) > 0 && b.element_type === posX);
    }
    if (subIdx === -1) {
      subIdx = benchRem.findIndex(b => b && (b.minutes || 0) > 0 && isLegalSub(posX, b.element_type, counts));
    }
    if (subIdx !== -1) {
      const b = benchRem[subIdx];
      if (posX > 1 && b.element_type !== posX) { counts[posX]--; counts[b.element_type]++; }
      xiFinal[i] = b;
      benchRem[subIdx] = null;
      subsFired++;
    } else {
      blanksUncovered++;   // gh#43: starter blanked, bench could not cover
    }
  }
  return { xi: xiFinal, bench: benchRem.filter(Boolean), subsFired, blanksUncovered };
}

// gh#44: analytic expected post-auto-sub value of a nominated XI. For each XI
// player, with blank probability q = 1 - avail, the expected contribution is
// avail*pred (they play) plus (1-avail)*replacement (they blank), where the
// replacement is the first bench player in the given order who is a legal sub,
// itself weighted by its own avail. Single-blank independence, multi-blank
// bench-exhaustion is negligible at starter availability levels.
function xiExpectedValue(xi, benchOrder) {
  const counts = { 2: 0, 3: 0, 4: 0 };
  for (const p of xi) if (p.element_type > 1) counts[p.element_type]++;
  let ev = 0;
  for (const p of xi) {
    const a = p.avail == null ? 1 : p.avail;
    let replEV = 0;
    for (const b of benchOrder) {
      if (!b) continue;
      if (isLegalSub(p.element_type, b.element_type, counts)) {
        replEV = (b.avail == null ? 1 : b.avail) * (b.predicted || 0);
        break;
      }
    }
    ev += a * (p.predicted || 0) + (1 - a) * replEV;
  }
  return ev;
}

// gh#44: the GK bench slot is fixed; the 3 outfield bench slots have 6
// permutations. Return the ordering that maximises xiExpectedValue.
function bestBenchOrder(xi, bench) {
  const gk  = bench.filter(b => b.element_type === 1);
  const out = bench.filter(b => b.element_type !== 1);
  if (out.length <= 1) return { order: bench, ev: xiExpectedValue(xi, bench) };
  let best = null;
  for (const perm of permute(out)) {
    const order = [...gk, ...perm];
    const ev = xiExpectedValue(xi, order);
    if (!best || ev > best.ev) best = { order, ev };
  }
  return best;
}

// gh#44: pick the XI from a 15-man squad under a formation policy.
//   '4-4-2' (or any of the 8 names) - locked: nominate that single formation.
//   'max'    - fluid naive: enumerate all 8, nominate the highest Sum-predicted.
//   'robust' - fluid robustness-aware: enumerate all 8 x 6 bench orders,
//              nominate the highest analytic post-auto-sub EV.
// Locked + max use the legacy same-position-first auto-sub heuristic with the
// legacy bench order (so locked '4-4-2' is byte-identical to the old fixed
// behaviour). 'robust' co-optimises the bench order and consumes it in strict
// order. Returns { xi, bench, formation }.
function pickXI(squad, scoredById, formationPolicy) {
  // gh#47 Stage 2 confirmed: 'max' (fluid per-GW formation) beats fixed '4-4-2'
  // by +54 NET at the slim-winner planner cell. Flipped 2026-05-23.
  formationPolicy = formationPolicy || 'max';
  const byPos = { 1: [], 2: [], 3: [], 4: [] };
  for (const p of squad) {
    const s = scoredById[p.id] || { predicted: 0, actual: 0, minutes: 0, isDGW: false, avail: 1 };
    byPos[p.element_type].push({ ...p, predicted: s.predicted, actual: s.actual,
      minutes: s.minutes, isDGW: s.isDGW, avail: s.avail == null ? 1 : s.avail });
  }
  for (const pos of [1, 2, 3, 4]) byPos[pos].sort((a, b) => b.predicted - a.predicted);

  const isRobust = formationPolicy === 'robust';
  const isFluid  = isRobust || formationPolicy === 'max';
  const candidates = isFluid ? FORMATION_NAMES : [formationPolicy];

  let best = null;
  for (const name of candidates) {
    const formation = LEGAL_FORMATIONS[name];
    if (!formation) throw new Error(`Unknown formation policy: ${formationPolicy}`);
    const nom = nominateXI(byPos, formation);
    if (nom.xi.length < 11) continue;
    let score, benchOrder;
    if (isRobust) {
      const bb = bestBenchOrder(nom.xi, nom.bench);
      score = bb.ev; benchOrder = bb.order;
    } else {
      score = nom.xi.reduce((s, p) => s + (p.predicted || 0), 0);
      benchOrder = nom.bench;
    }
    if (!best || score > best.score) best = { name, xi: nom.xi, bench: benchOrder, score };
  }
  if (!best) return { xi: [], bench: [], formation: null, subsFired: 0, blanksUncovered: 0 };

  const subbed = runAutoSubs(best.xi, best.bench, !isRobust);
  return { xi: subbed.xi, bench: subbed.bench, formation: best.name,
           subsFired: subbed.subsFired, blanksUncovered: subbed.blanksUncovered };
}

// Best legal XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD) from a complete 15-man
// squad, ranked by scoreFn. Mandatory minimums first, then the 4 best
// remaining outfield players. Returns { total, xi }.
function bestLegalXI(squad15, scoreFn) {
  const byPos = { 1: [], 2: [], 3: [], 4: [] };
  for (const p of squad15) byPos[p.element_type].push(p);
  for (const k of [1, 2, 3, 4]) byPos[k].sort((a, b) => scoreFn(b) - scoreFn(a));
  const xi = [byPos[1][0], ...byPos[2].slice(0, 3), ...byPos[3].slice(0, 2), byPos[4][0]]
    .filter(Boolean);
  const rest = [...byPos[2].slice(3), ...byPos[3].slice(2), ...byPos[4].slice(1)]
    .sort((a, b) => scoreFn(b) - scoreFn(a)).slice(0, 4);
  xi.push(...rest);
  return { total: xi.reduce((s, p) => s + scoreFn(p), 0), xi };
}

// Free Hit XI (gh#41). A real Free Hit fields a LEGAL 15-man squad, 2/5/5/3,
// within the club cap, within budget, not an unconstrained all-star XI. Build
// the best legal 15 by `key` (wildcardRebuild handles 2/5/5/3 + cap + budget),
// then return its best XI. `key` = 'predicted' (decision signal) or 'actual'
// (scoring). Returns { total, xi }, where xi carries the full scored objects.
function bestFreeHitXI(scoredById, key, budgetTotal) {
  const pool = [];
  for (const pid in scoredById) {
    const s = scoredById[pid];
    if (!s || !s.element_type || s.price == null) continue;
    // 'actual': drop no-shows (minutes 0). 'predicted': drop no-fixture/blank.
    if (key === 'actual' && (s.minutes || 0) === 0) continue;
    if (key === 'predicted' && (s.predicted || 0) <= 0) continue;
    pool.push({ id: pid, element_type: s.element_type, team: s.team,
                price: s.price, predicted: s[key] || 0 });
  }
  const built = wildcardRebuild(pool, budgetTotal);
  if (!built || built.squad.length !== 15) return { total: 0, xi: [] };
  const score = p => (scoredById[p.id] && scoredById[p.id][key]) || 0;
  const best = bestLegalXI(built.squad, score);
  return { total: best.total, xi: best.xi.map(p => scoredById[p.id]) };
}

// Find the best single (out, in) swap given current squad, free transfers,
// budget, and lookahead horizon. Returns null if no swap improves enough.
function findBestTransfer(squad, players, gw, ctx, variantName, ew, lookahead, freeTransfers, hitThreshold, clubCount, budget) {
  let bestDelta = -Infinity;
  let bestSwap = null;

  // Score each squad player by lookahead (we'll reuse for comparing replacements).
  const squadLookahead = {};
  for (const p of squad) {
    squadLookahead[p.id] = lookaheadPredict(p, p, gw, lookahead, ctx, variantName, ew);
  }

  // Candidate pool: same position as the outgoing, not in current squad, has a GW≥gw record.
  const inSquad = new Set(squad.map(p => p.id));
  for (const out of squad) {
    const outScore = squadLookahead[out.id];
    const candidates = players.filter(p =>
      p.element_type === out.element_type &&
      !inSquad.has(p.id) &&
      p.history.some(h => h.round >= gw)
    );
    for (const cand of candidates) {
      const candPriceEntry = cand.history.find(h => h.round >= gw);
      if (!candPriceEntry || !candPriceEntry.value) continue;
      const candPrice = candPriceEntry.value;
      const outPriceEntry = out.history.find(h => h.round >= gw) || out.history.slice(-1)[0];
      const sellPrice = outPriceEntry ? outPriceEntry.value : out.price || 0;
      // Budget check: can we afford the swap given current bank?
      if (budget + sellPrice < candPrice) continue;
      // Club check: would the swap break the club cap?
      const candClubCount = (clubCount[cand.team] || 0) - (out.team === cand.team ? 1 : 0) + 1;
      if (candClubCount > _activeMaxPerClub) continue;

      const inScore = lookaheadPredict(cand, cand, gw, lookahead, ctx, variantName, ew);
      const rawDelta = inScore - outScore;
      // Hit cost: only paid if we'd need more transfers than we have.
      const hitCost = freeTransfers >= 1 ? 0 : HIT_COST;
      const netDelta = rawDelta - hitCost;
      // Threshold: we only do free transfers if delta > 0, hits if delta > hitThreshold.
      const required = freeTransfers >= 1 ? 0.1 : hitThreshold;
      if (netDelta > bestDelta && netDelta > required) {
        bestDelta = netDelta;
        bestSwap = { out, in: cand, sellPrice, buyPrice: candPrice, rawDelta, netDelta, hitCost };
      }
    }
  }
  return bestSwap;
}

function gwActualsAndPreds(players, gw, ctx, variantName, ew) {
  // Compute actuals + predictions for every player who appeared in this GW.
  const out = {};
  for (const p of players) {
    const entries = p.history.filter(h => h.round === gw);
    if (!entries.length) continue;
    const totalMins = entries.reduce((s, h) => s + h.minutes, 0);
    const totalPts  = entries.reduce((s, h) => s + h.total_points, 0);
    const synth = synthesisePlayer(p, p.history, gw);
    const isDGW = entries.length > 1;
    if (!synth) {
      out[p.id] = { predicted: 0, actual: totalPts, minutes: totalMins, isDGW, avail: 1, element_type: p.element_type, team: p.team, price: entries[0].value };
      continue;
    }
    const fixList = entries.map(h => ({
      team_h: h.was_home ? p.team : h.opponent_team,
      team_a: h.was_home ? h.opponent_team : p.team,
      team_h_difficulty: 3, team_a_difficulty: 3
    }));
    const predicted = predictWithVariant(synth, fixList, ctx, variantName, ew);
    // gh#44: per-player play-availability for the robust formation policy.
    // Computed with a history-derived rolling-minutes signal in a throwaway ctx
    // copy, so `predicted` above stays byte-identical (predictWithVariant never
    // sees minutesHistory).
    const mh = p.history.filter(h => h.round < gw).map(h => ({ gw: h.round, mins: h.minutes }));
    const availCtx = Object.assign({}, ctx, { minutesHistory: { [synth.id]: mh }, currentGw: gw });
    const avail = predictApi.availability(synth, availCtx);
    out[p.id] = { predicted, actual: totalPts, minutes: totalMins, isDGW, avail, element_type: p.element_type, team: p.team, price: entries[0].value };
  }
  return out;
}

// Calibrate the hit threshold to this season's predictor noise. Higher MAE
// means we should require bigger predicted gains before taking a hit. We
// measure MAE on the first 5 GWs of the season as an honest "what the
// predictor knows when chip/transfer decisions are first made" signal.
function autoTuneHitThreshold(players, season, ctx, variantName, ew, baseThreshold) {
  const errs = [];
  for (let gw = 2; gw <= 6; gw++) {
    for (const p of players) {
      const entries = p.history.filter(h => h.round === gw);
      if (!entries.length) continue;
      const mins = entries.reduce((s, h) => s + h.minutes, 0);
      if (mins < 45) continue;
      const synth = synthesisePlayer(p, p.history, gw);
      if (!synth) continue;
      const fixList = entries.map(h => ({
        team_h: h.was_home ? p.team : h.opponent_team,
        team_a: h.was_home ? h.opponent_team : p.team,
        team_h_difficulty: 3, team_a_difficulty: 3
      }));
      const pred   = predictWithVariant(synth, fixList, ctx, variantName, ew);
      const actual = entries.reduce((s, h) => s + h.total_points, 0);
      if (pred < 0.5) continue;
      errs.push(Math.abs(pred - actual));
    }
  }
  if (errs.length < 50) return baseThreshold;
  const mae = errs.reduce((s, e) => s + e, 0) / errs.length;
  // Calibrate: baseline MAE of 2.4 maps to threshold 8. Each 0.25 MAE delta
  // shifts threshold by 2. Clamp to [6, 14] so we don't go to extremes.
  const tuned = Math.round(baseThreshold + (mae - 2.4) * 8);
  return Math.max(6, Math.min(14, tuned));
}

function simulate(players, season, variantName, ctx, ensembleWeights, opts) {
  const startGw = 1, endGw = 38;
  const lookahead    = opts.lookahead || 3;
  let hitThreshold   = opts.hitThreshold || 16;
  opts.planHorizon   = opts.planHorizon || 8;
  opts.beamWidth     = opts.beamWidth   || 8;
  opts.topKMoves     = opts.topKMoves   || 5;
  // gh#44: XI formation policy. 'max' (default since gh#47 Stage 2, +54 NET vs
  // fixed 4-4-2 at the slim-winner planner cell) picks the best legal formation
  // per GW. Any of the 8 named formations locks that formation;
  // 'robust' co-optimises bench order using analytic post-auto-sub EV.
  opts.formationPolicy = opts.formationPolicy || 'max';
  const vcache       = opts.vcache;
  // Issue #24: opts.maxPerClub overrides the default 3-per-club cap so we can
  // test the 'one_per_club' variant + any future club-cap experiments.
  _activeMaxPerClub  = opts.maxPerClub || MAX_PER_CLUB;

  // News injection: when opts.applyNews is true and news-cache.json exists,
  // attach the flags to ctx so predict.js consults them as availability signals.
  // We mutate a local copy so callers' ctx isn't affected.
  if (opts.applyNews) {
    const flags = loadNewsCache();
    if (flags) ctx = Object.assign({}, ctx, { newsOverride: flags });
  }

  if (opts.autoTuneHitThreshold !== false) {
    const tuned = autoTuneHitThreshold(players, season, ctx, variantName, ensembleWeights, hitThreshold);
    if (tuned !== hitThreshold) {
      opts.tunedThreshold = tuned;
      hitThreshold = tuned;
    }
  }

  // gh#43 crude cost-constraint cross-checks. Exclude "big-shot" players from
  // the entire candidate universe for the season. opts.priceCapM caps a single
  // player's price (£m); opts.blockTopN drops the N priciest players per
  // position. The ban set is computed once here, so the initial build, the WC
  // rebuild, and the planner all see the same restricted pool. Both absent →
  // no filter, byte-identical to today.
  if (opts.priceCapM != null || opts.blockTopN != null) {
    const priceOf = p => (p.history && p.history[0] ? p.history[0].value : 0);
    const banned = new Set();
    if (opts.priceCapM != null) {
      const capUnits = opts.priceCapM * 10;
      for (const p of players) if (priceOf(p) > capUnits) banned.add(p.id);
    }
    if (opts.blockTopN != null) {
      for (const pos of [1, 2, 3, 4]) {
        const inPos = players.filter(p => p.element_type === pos)
          .sort((a, b) => priceOf(b) - priceOf(a));
        for (let i = 0; i < opts.blockTopN && i < inPos.length; i++) banned.add(inPos[i].id);
      }
    }
    players = players.filter(p => !banned.has(p.id));
  }

  // Build initial squad at GW1 using previous-season prior (or position fallback).
  // opts.initialStrategy picks one of SQUAD_STRATEGIES; defaults to 'balanced'.
  const initial = buildInitialSquad(players, 1, ctx, variantName, ensembleWeights, vcache, season, opts.initialStrategy, opts.priceLambda);
  if (!initial) return null;
  let squad = initial.squad;
  let budget = initial.budget;
  let clubCount = initial.clubCount;

  let freeTransfers = 1;     // start with 1 free
  // Transfer memory: list of { outId, inId, outAt, inAt } so the planner can
  // refuse to flip-flop a player back within a few GWs.
  const recentTransfers = [];
  let totalPoints = 0;
  let totalHits = 0;
  let transfersMade = 0;
  let transfersByGw = [];
  let autoSubsFired = 0;            // gh#43 mechanism counter
  let deadAssetSwaps = 0;           // gh#71 dead-asset clearances
  let benchBlanksUncovered = 0;     // gh#43 mechanism counter
  let gwData = [];
  // gh#35 Wildcard: decided inline in the per-GW loop (a WC rebuilds the squad,
  // which the post-hoc chip pass cannot do). State persists across GWs.
  let wc1Gw = null, wc2Gw = null, wc1Used = false, wc2Used = false;
  // Per-GW trace (issue #25). Captures every transfer decision with predicted
  // gain + lookahead-window predicted scores. Realized actuals filled in at
  // end of season by reading from player.history. Only collected when opts.trace.
  const traceLog = [];
  // gh#44: per-GW formation trace, { gw, formation, dgw }. Always collected
  // (38 small rows); consumed by formation-analyze.js.
  const formationLog = [];
  const TRACE_HORIZON = opts.planHorizon || 4;

  for (let gw = startGw; gw <= endGw; gw++) {
    // Set per-GW context so the manager-change dampener in predict.js activates
    // for the right window during backtests (#6 / #8 step 1).
    ctx.currentGw = gw;
    // Check if any player in squad has any history at all (some early GWs of the
    // season are sparse for new transfers; skip transfer search if data is thin).
    const haveDataForGw = players.some(p => p.history.some(h => h.round === gw));
    if (!haveDataForGw) {
      gwData.push({ gw, xiPoints: 0, captain: null, dgwCount: 0, squad15Pts: 0, transferOut: null });
      continue;
    }

    // Transfer decision via beam-search planner (GW2+).
    let swap = null;
    let wcFired = false;
    if (gw > 1) {
      // Build lookahead cache: predict every relevant player for GW..GW+horizon-1.
      // Players also need their current-time price for budget checks during planning.
      const cache = {};
      const playersForPlanner = [];
      for (const p of players) {
        // Skip players without any history in the planning window.
        const hasFuture = p.history.some(h => h.round >= gw);
        if (!hasFuture) continue;
        const arr = [];
        for (let h = 0; h < opts.planHorizon; h++) {
          const targetGw = gw + h;
          const fixtures = p.history.filter(e => e.round === targetGw);
          if (!fixtures.length) { arr.push(0); continue; }
          const synth = synthesisePlayer(p, p.history, gw);   // use TODAY's pre-state
          if (!synth) { arr.push(0); continue; }
          const fixList = fixtures.map(f => ({
            team_h: f.was_home ? p.team : f.opponent_team,
            team_a: f.was_home ? f.opponent_team : p.team,
            team_h_difficulty: 3, team_a_difficulty: 3
          }));
          arr.push(predictWithVariant(synth, fixList, ctx, variantName, ensembleWeights));
        }
        cache[p.id] = arr;
        // Add price (current GW price) for budget checks.
        const priceEntry = p.history.find(h => h.round >= gw) || p.history.slice(-1)[0];
        playersForPlanner.push({ id: p.id, element_type: p.element_type, team: p.team, price: priceEntry ? priceEntry.value : 0 });
      }
      // Decorate squad with current prices too.
      const squadWithPrice = squad.map(p => {
        const priceEntry = p.history.find(h => h.round >= gw) || p.history.slice(-1)[0];
        return { id: p.id, element_type: p.element_type, team: p.team, price: priceEntry ? priceEntry.value : 0 };
      });

      // gh#35 Wildcard: rescue chip. Score every player by the mean of their
      // lookahead-cache predictions; if the current squad's best XI projects
      // below the trigger, rebuild from scratch (unlimited free transfers,
      // zero hit cost). Decided here, before transfer planning, because the
      // rebuild replaces the squad and must carry forward.
      if (!(halfOf(gw) === 1 ? wc1Used : wc2Used)) {
        const meanPred = id => {
          const a = cache[id] || [];
          return a.length ? a.reduce((s, x) => s + (x || 0), 0) / a.length : 0;
        };
        const xiPredMean = bestXIPred(squad, meanPred);
        const CO = opts.chipOpts || {};
        // gh#80: EV-uplift WC mode (A/B against the locked rescue rule). At
        // each GW in the half's window, price the rebuild's horizon uplift
        // over the current squad, and the same uplift as it would look from
        // each of the next few GWs (preds offset j, no transfers assumed);
        // fire when now clears the floor AND beats the option value of
        // waiting (chips.valueChipFire, same k as TC/BB/FH).
        let wcFireNow = false;
        if (CO.wcMode === 'ev') {
          const half = halfOf(gw);
          const inWindow = gw >= (CO[`minWCEarlyH${half}`] ?? CHIPLIB_DEFAULTS[`minWCEarlyH${half}`])
                        && gw <= (CO[`wcDeadlineH${half}`] ?? CHIPLIB_DEFAULTS[`wcDeadlineH${half}`]);
          if (inWindow) {
            const H = opts.planHorizon || 9;
            const totalBudget = budget + squadWithPrice.reduce((s, p) => s + p.price, 0);
            const evAt = j => {
              const mp = id => { const a = (cache[id] || []).slice(j); return a.length ? a.reduce((s2, x) => s2 + (x || 0), 0) / a.length : 0; };
              const pool = playersForPlanner.map(p => ({ ...p, predicted: mp(p.id) }));
              const rb = wildcardRebuild(pool, totalBudget, opts.priceLambda);
              if (!rb || rb.squad.length !== 15) return 0;
              return (bestXIPred(rb.squad, mp) - bestXIPred(squadWithPrice, mp)) * Math.max(1, H - j);
            };
            const ev0 = evAt(0);
            const futures = [1, 2, 3, 4].map(evAt);
            wcFireNow = ev0 >= (CO.wcMinUplift ?? 12) && valueChipFire(ev0, futures, { k: CO.optionK ?? CHIPLIB_DEFAULTS.optionK });
          }
        } else {
          wcFireNow = wcShouldFire(gw, xiPredMean, { wc1Used, wc2Used }, opts.chipOpts);
        }
        if (wcFireNow) {
          const totalBudget = budget + squadWithPrice.reduce((s, p) => s + p.price, 0);
          const pool = playersForPlanner.map(p => ({ ...p, predicted: meanPred(p.id) }));
          const rebuilt = wildcardRebuild(pool, totalBudget, opts.priceLambda);
          if (rebuilt && rebuilt.squad.length === 15) {
            const newSquad = rebuilt.squad
              .map(rp => players.find(p => p.id === rp.id))
              .filter(Boolean);
            if (newSquad.length === 15) {
              transfersMade += newSquad.filter(p => !squad.some(s => s.id === p.id)).length;
              squad = newSquad;
              budget = rebuilt.budget;
              clubCount = rebuilt.clubCount;
              freeTransfers = 1;   // WC GW transfers are free; 1 FT next GW.
              if (halfOf(gw) === 1) { wc1Used = true; wc1Gw = gw; }
              else                  { wc2Used = true; wc2Gw = gw; }
              wcFired = true;
            }
          }
        }
      }

      if (!wcFired) {
      const planOpts = { horizon: opts.planHorizon, beamWidth: opts.beamWidth, topKMoves: opts.topKMoves,
          hitThreshold, lookaheadCache: cache, hitCost: HIT_COST,
          formationPolicy: opts.formationPolicy, priceLambda: opts.priceLambda,
          benchWeight: opts.benchWeight, ftAccrual: opts.ftAccrual };
      const planFrom = (sq, ft, bud, cc, rt, returnTotal) => planMove(
        sq, gw, ctx, variantName, ensembleWeights, playersForPlanner,
        { freeTransfers: ft, budget: bud, clubCount: cc, minGain: opts.minGain,
          recentTransfers: rt, currentGw: gw, _maxPerClub: _activeMaxPerClub },
        returnTotal ? Object.assign({ returnTotal: true }, planOpts) : planOpts);
      // gh#71 dead-asset modes. 'off' (default) is the historical backtest.
      // 'plan' mirrors the live decision rule: on a planner hold with an FT in hand,
      // clear a dead asset only if the plan from the new state is not worse.
      // 'hold' clears on a hold unconditionally. 'debt' clears it first
      // whenever an FT is in hand, then the planner plans from the new state.
      // Dead = <= 0.5 predicted points over the planning horizon (live rule).
      let deadSwap = null;
      const deadMode = opts.deadAsset || 'off';
      const look = id => (cache[id] || []).reduce((s, x) => s + (x || 0), 0);
      const clearDead = (ftNow, holdTotal) => {
        const inSq = new Set(squadWithPrice.map(p => p.id));
        const recentlyOut = new Set(recentTransfers.filter(t => t.outAt >= gw - 3).map(t => t.outId));
        let best = null;
        for (const out of squadWithPrice.filter(p => look(p.id) <= 0.5)) {
          const cands = playersForPlanner
            .filter(e => e.element_type === out.element_type && !inSq.has(e.id) && !recentlyOut.has(e.id))
            .filter(e => e.price <= out.price + budget && look(e.id) > 0)
            .filter(e => (clubCount[e.team] || 0) - (e.team === out.team ? 1 : 0) + 1 <= _activeMaxPerClub)
            .sort((a, b) => look(b.id) - look(a.id)).slice(0, 4);
          for (const inE of cands) {
            const sq = squadWithPrice.filter(p => p.id !== out.id).concat([inE]);
            const cc = Object.assign({}, clubCount);
            cc[out.team] = (cc[out.team] || 1) - 1; cc[inE.team] = (cc[inE.team] || 0) + 1;
            const r = planFrom(sq, ftNow - 1, budget + out.price - inE.price, cc, recentTransfers, true);
            const t = r && r._total != null ? r._total : -Infinity;
            if (holdTotal != null && t + 1e-9 < holdTotal) continue;
            if (!best || t > best.t) best = { t, out, inE, sq, cc };
          }
        }
        return best;
      };
      let decision;
      if (deadMode === 'debt' && freeTransfers >= 1) deadSwap = clearDead(freeTransfers, null);
      if (deadSwap) {
        const rt = recentTransfers.concat([{ outId: deadSwap.out.id, inId: deadSwap.inE.id, outAt: gw, inAt: gw }]);
        decision = planFrom(deadSwap.sq, freeTransfers - 1, budget + deadSwap.out.price - deadSwap.inE.price, deadSwap.cc, rt, false);
      } else {
        decision = planFrom(squadWithPrice, freeTransfers, budget, clubCount, recentTransfers, false);
        if ((deadMode === 'plan' || deadMode === 'hold') && decision.type === 'hold' && freeTransfers >= 1) {
          let holdTotal = null;
          if (deadMode === 'plan') {
            const r = planFrom(squadWithPrice, freeTransfers, budget, clubCount, recentTransfers, true);
            holdTotal = r && r._total != null ? r._total : -Infinity;
          }
          deadSwap = clearDead(freeTransfers, holdTotal);
        }
      }
      if (deadSwap) deadAssetSwaps++;
      const deadFull = deadSwap && players.find(p => p.id === deadSwap.inE.id);
      const deadPair = deadFull ? [{ out: deadSwap.out, in: deadFull, sellPrice: deadSwap.out.price, buyPrice: deadSwap.inE.price }] : [];
      // Trace capture (issue #25): record decision context BEFORE applying.
      // predicted_gain is the sum-of-pred for IN over the next H GWs minus
      // the same for OUT, using the lookahead cache the planner just used.
      if (opts.trace && (decision.type === 'transfer' || decision.type === 'transferN')) {
        const sumPred = pid => (cache[pid] || []).slice(0, TRACE_HORIZON).reduce((s, x) => s + (x || 0), 0);
        const pairs = decision.type === 'transfer'
          ? [{ out: decision.out, in: decision.in }]
          : decision.pairs.map(pp => ({ out: pp.out, in: pp.in }));
        const transfersOverFree = Math.max(0, pairs.length - freeTransfers);
        traceLog.push({
          gw,
          type: decision.type,
          pairs: pairs.map(pp => ({
            out_id: pp.out.id, out_pos: pp.out.element_type, out_team: pp.out.team,
            in_id:  pp.in.id,  in_pos:  pp.in.element_type,  in_team:  pp.in.team,
            predicted_horizon_out: +sumPred(pp.out.id).toFixed(2),
            predicted_horizon_in:  +sumPred(pp.in.id).toFixed(2)
          })),
          predicted_gain: +pairs.reduce((s, pp) => s + sumPred(pp.in.id) - sumPred(pp.out.id), 0).toFixed(2),
          hit_cost: transfersOverFree * HIT_COST,
          free_transfers_before: freeTransfers,
          budget_before: budget
        });
      }
      if (decision.type === 'transfer') {
        const inFull = players.find(p => p.id === decision.in.id);
        if (inFull) swap = [{ out: decision.out, in: inFull, sellPrice: decision.sellPrice, buyPrice: decision.buyPrice }];
      } else if (decision.type === 'transferN') {
        const built = [];
        let allFound = true;
        for (const pp of decision.pairs) {
          const inFull = players.find(p => p.id === pp.in.id);
          if (!inFull) { allFound = false; break; }
          built.push({ out: pp.out, in: inFull, sellPrice: pp.sellPrice, buyPrice: pp.buyPrice });
        }
        if (allFound) swap = built;
      }
      if (deadPair.length) swap = deadPair.concat(swap || []);
      }  // end if (!wcFired)
    }
    if (swap) {
      for (const t of swap) {
        squad = squad.filter(p => p.id !== t.out.id).concat([t.in]);
        budget = budget + t.sellPrice - t.buyPrice;
        clubCount[t.out.team] = (clubCount[t.out.team] || 1) - 1;
        clubCount[t.in.team]  = (clubCount[t.in.team]  || 0) + 1;
        if (freeTransfers >= 1) freeTransfers -= 1;
        else                    totalHits += HIT_COST;
        transfersMade++;
        recentTransfers.push({ outId: t.out.id, inId: t.in.id, outAt: gw, inAt: gw });
      }
      // gh#79: the weekly +1 applies after a transfer GW too (real FPL rule).
      // Legacy behaviour (no accrual after a transfer) kept behind the flag
      // until the A/B sweep adopts it; planner.js mirrors the same flag.
      if (opts.ftAccrual && gw > 1) freeTransfers = Math.min(5, freeTransfers + 1);
    } else if (gw > 1 && !wcFired) {
      // 2024-25 rule: FT cap raised from 2 to 5 (issue #34). Must stay in
      // lockstep with FT_CAP in planner.js. Skipped on a WC GW (FT already
      // set to 1 by the rebuild).
      freeTransfers = Math.min(5, freeTransfers + 1);
    }
    transfersByGw.push({ gw, swapped: !!swap, freeAfter: freeTransfers });

    // Score this GW. Compute actuals + preds for every player in the league.
    const scored = gwActualsAndPreds(players, gw, ctx, variantName, ensembleWeights);

    // Pick XI from squad (gh#44: formation policy, locked / max / robust).
    const { xi, bench, formation, subsFired, blanksUncovered } = pickXI(squad, scored, opts.formationPolicy);
    autoSubsFired += subsFired;
    benchBlanksUncovered += blanksUncovered;
    formationLog.push({
      gw, formation,
      dgw: squad.filter(p => scored[p.id] && scored[p.id].isDGW).length
    });
    if (xi.length < 11) {
      gwData.push({ gw, xiPoints: 0, captain: null, dgwCount: 0, squad15Pts: 0 });
      continue;
    }
    const xiPoints = xi.reduce((s, x) => s + (x.actual || 0), 0);
    // Captain selection: top 5 by predicted, soft tie-break that mildly
    // prefers attackers because their actual-points UPSIDE is higher (a FWD
    // can score 13+; a DEF caps around 10). Reduced from the previous +0.5
    // / +0.3 bonus to +0.2 / +0.1 because issue #2 calibration showed FWDs
    // in the 6-7 range are over-predicted (slope 0.54). With FWD predictions
    // compressed in predict.js, this bonus now lands closer to where it
    // should: small nudge for genuine premium FWDs, doesn't override clearly-
    // better MIDs.
    const top5 = [...xi].sort((a, b) => (b.predicted || 0) - (a.predicted || 0)).slice(0, 5);
    // Captain mode dispatch (issues #11, #10):
    //   safe     - highest predicted mean (default; tournament-winning baseline)
    //   balanced - small EO penalty (0.02/%) for slight differential lean
    //   chase    - bigger EO penalty (0.04/%) for top-10K hunting
    //   ceiling  - issue #10: maximise P(haul >= 10) instead of E[points].
    //              Empirical lookup via haul-probs.json. Captures fat-tail
    //              upside that mean-prediction smooths over.
    const mode = ctx._captainMode || 'safe';
    let captainScore;
    if (mode === 'ceiling' && ctx.haulProbs) {
      // P(>=10) is the haul threshold; >=15 is too noisy for selection.
      // Position-bonus matches the original tie-break logic so ceiling mode
      // doesn't accidentally regress on edge cases where two players tie.
      captainScore = p => {
        const pos = p.element_type;
        const posBonus = (pos === 4 ? 0.01 : pos === 3 ? 0.005 : 0);
        return predictApi.haulProb(p.predicted || 0, pos, 10, ctx.haulProbs) + posBonus;
      };
    } else {
      const eoCoef = mode === 'chase' ? 0.04 : mode === 'balanced' ? 0.02 : 0.0;
      // Issue #27 (closed negative): tried weighting captainScore by haul
      // probability (0.4 * p10 + 0.2 * p15, then 0.8 * p10 + 0.4 * p15).
      // Both produced ~0 NET delta in backtest because haul probabilities
      // are derived from pred_next via empirical lookup, they encode no
      // independent information. Captain pick is already near-optimal.
      captainScore = p => {
        const pos = p.element_type;
        const posBonus = (pos === 4 ? 0.2 : pos === 3 ? 0.1 : 0);
        const eo = p.ownership_pct || 0;
        return (p.predicted || 0) + posBonus - eoCoef * eo;
      };
    }
    top5.sort((a, b) => captainScore(b) - captainScore(a));
    const captain = top5[0];
    const vice    = top5[1] || captain;
    const effectiveCaptain = (captain && (captain.minutes || 0) > 0) ? captain : vice;
    const dgwCount = xi.filter(x => x.isDGW).length + bench.filter(x => x.isDGW).length;
    const squad15Pts = xiPoints + bench.reduce((s, x) => s + (x.actual || 0), 0);
    const xiPred  = xi.reduce((s, x) => s + (x.predicted || 0), 0);
    const benchPred = bench.reduce((s, x) => s + (x.predicted || 0), 0);
    const squad15Pred = xiPred + benchPred;
    // Free Hit signals:
    //   xiBlanks: how many XI players got 0 minutes this GW (couldn't play,
    //   no fixture, injured-without-news). Triggers FH if >= threshold.
    //   freeHitXIPts: the best XI from a LEGAL Free Hit 15 (gh#41, within the
    //   budget + club cap, not an unconstrained all-star team).
    const xiBlanks = xi.filter(x => (x.minutes || 0) === 0).length;
    // FH budget: the standard £100m. Squad value + bank drifts only a few %
    // from £100m across a season, and a flat value avoids mis-counting on blank
    // GWs (where blanked squad players are absent from `scored`).
    const fhBudget = BUDGET_TOTAL;
    const fhActual = bestFreeHitXI(scored, 'actual', fhBudget);
    const freeHitXIPts  = fhActual.total;
    const freeHitXIPred = bestFreeHitXI(scored, 'predicted', fhBudget).total;
    const fhXI = fhActual.xi;

    // Captain ceiling probability (issue #10): P(captain hauls >= 15). Used
    // by chips.js TC decision when ceiling mode is active. Empirical lookup
    // from haul-probs.json; falls back to Poisson if no table loaded.
    let captainCeiling = 0;
    if (captain) {
      captainCeiling = predictApi.haulProb(
        captain.predicted || 0, captain.element_type, 15, ctx.haulProbs
      );
    }

    gwData.push({
      gw, xiPoints, captain: effectiveCaptain, captainPred: captain, captainCeiling,
      dgwCount, squad15Pts, squad15Pred, benchPred, xiBlanks, freeHitXIPts, freeHitXIPred, fhXI
    });
  }

  // Live chip decisions: walk gwData chronologically; at each GW, the chip
  // module sees only the current GW + next 5 GWs (matching a real manager's
  // visibility horizon), never future actuals. This is functionally identical
  // to deciding in real time.
  //
  // To revert to post-hoc selection (perfect timing with full-season hindsight),
  // pass opts.liveChips = false. Keep this knob for A/B testing the lift.
  // gh#38: 2-per-season chips. Each chip type fires up to twice (H1 + H2).
  let tc1Gw = null, tc2Gw = null, bb1Gw = null, bb2Gw = null;
  let fh1Gw = null, fh2Gw = null;
  const LOOKAHEAD = 5;
  if (opts.liveChips !== false) {
    const chipState = initChipState();
    // WC was already decided inline in the per-GW loop; seed its state so
    // decideChipsThisGw won't re-fire it and TC/BB/FH skip the WC GWs.
    chipState.wc1Used = wc1Used;
    chipState.wc2Used = wc2Used;
    for (let i = 0; i < gwData.length; i++) {
      const allUsed = chipState.tc1Used && chipState.tc2Used
                   && chipState.bb1Used && chipState.bb2Used
                   && chipState.fh1Used && chipState.fh2Used;
      if (allUsed) break;
      const d = gwData[i];
      // One chip per GW: never assign TC/BB/FH to a GW the WC already used.
      if (d.gw === wc1Gw || d.gw === wc2Gw) continue;
      const window = gwData.slice(i, i + LOOKAHEAD + 1).map(g => ({
        gw: g.gw,
        captainPred:    (g.captainPred && g.captainPred.predicted) || 0,
        captainCeiling: g.captainCeiling || 0,
        squad15Pred: g.squad15Pred || 0,
        benchPred:   g.benchPred   || 0,
        dgwCount:    g.dgwCount    || 0,
        xiBlanks:    g.xiBlanks    || 0,
        xiPred:        (g.squad15Pred || 0) - (g.benchPred || 0),
        freeHitXIPred: g.freeHitXIPred || 0
      }));
      const decision = decideChipsThisGw(d.gw, window, chipState, opts.chipOpts);
      const h = decision.halfInstance;
      if (decision.useTC) {
        if (h === 1) { tc1Gw = d.gw; chipState.tc1Used = true; }
        else         { tc2Gw = d.gw; chipState.tc2Used = true; }
      }
      if (decision.useBB) {
        if (h === 1) { bb1Gw = d.gw; chipState.bb1Used = true; }
        else         { bb2Gw = d.gw; chipState.bb2Used = true; }
      }
      if (decision.useFH) {
        if (h === 1) { fh1Gw = d.gw; chipState.fh1Used = true; }
        else         { fh2Gw = d.gw; chipState.fh2Used = true; }
      }
    }
  } else {
    // Post-hoc fallback (full-season hindsight). gh#38: pick best in each half.
    const pickBestInHalf = (cands, scoreFn, halfMin, halfMax) => {
      const inHalf = cands.filter(d => d.gw >= halfMin && d.gw <= halfMax);
      if (!inHalf.length) return null;
      return inHalf.reduce((b, d) => scoreFn(d) > scoreFn(b) ? d : b).gw;
    };
    // Exclude WC GWs (decided inline), one chip per GW.
    const notWc = d => d.gw !== wc1Gw && d.gw !== wc2Gw;
    const tcCands = gwData.filter(d => d.captainPred && notWc(d));
    const bbCands = gwData.filter(notWc);
    const fhCands = gwData.filter(d => (d.xiBlanks || 0) >= 3 && notWc(d));
    tc1Gw = pickBestInHalf(tcCands, d => d.captainPred.predicted, 1, 19);
    tc2Gw = pickBestInHalf(tcCands, d => d.captainPred.predicted, 20, 38);
    bb1Gw = pickBestInHalf(bbCands.filter(d => d.gw !== tc1Gw), d => d.squad15Pred || 0, 1, 19);
    bb2Gw = pickBestInHalf(bbCands.filter(d => d.gw !== tc2Gw), d => d.squad15Pred || 0, 20, 38);
    fh1Gw = pickBestInHalf(fhCands.filter(d => d.gw !== tc1Gw && d.gw !== bb1Gw), d => d.xiBlanks || 0, 1, 19);
    fh2Gw = pickBestInHalf(fhCands.filter(d => d.gw !== tc2Gw && d.gw !== bb2Gw), d => d.xiBlanks || 0, 20, 38);
  }

  // Apply chips and tally. gh#38: per-half chip GWs.
  const fhSet = new Set([fh1Gw, fh2Gw].filter(Boolean));
  const tcSet = new Set([tc1Gw, tc2Gw].filter(Boolean));
  const bbSet = new Set([bb1Gw, bb2Gw].filter(Boolean));
  let withChips = 0;
  for (const d of gwData) {
    let pts;
    if (fhSet.has(d.gw)) {
      pts = (d.freeHitXIPts || 0);
      if (d.fhXI && d.fhXI.length) {
        const fhCap = d.fhXI.reduce((best, p) =>
          (p.predicted || 0) > (best.predicted || 0) ? p : best);
        pts += fhCap.actual || 0;
      }
    } else {
      pts = d.xiPoints;
      if (d.captain) pts += d.captain.actual || 0;
      if (tcSet.has(d.gw) && d.captain) pts += d.captain.actual || 0;       // TC: +1 cap actual
      if (bbSet.has(d.gw)) pts += (d.squad15Pts - d.xiPoints) || 0;          // BB: +bench
    }
    withChips += pts;
  }

  const totalNet = withChips - totalHits;

  // Trace enrichment (issue #25): fill in realized actuals for every transfer.
  // For each (out, in) pair, sum actual points over the H GWs starting at the
  // decision GW. Out is what we LOST (theoretical, since they're gone after gw);
  // In is what we GAINED.
  if (opts.trace && traceLog.length) {
    const playerById = {};
    players.forEach(p => { playerById[p.id] = p; });
    const sumActual = (pid, fromGw, h) => {
      const p = playerById[pid];
      if (!p) return 0;
      let s = 0;
      for (let g = fromGw; g < fromGw + h; g++) {
        const rows = p.history.filter(x => x.round === g);
        s += rows.reduce((acc, r) => acc + (r.total_points || 0), 0);
      }
      return +s.toFixed(2);
    };
    for (const t of traceLog) {
      for (const pp of t.pairs) {
        pp.actual_horizon_out = sumActual(pp.out_id, t.gw, TRACE_HORIZON);
        pp.actual_horizon_in  = sumActual(pp.in_id,  t.gw, TRACE_HORIZON);
      }
      t.realized_gain = +t.pairs.reduce((s, pp) => s + pp.actual_horizon_in - pp.actual_horizon_out, 0).toFixed(2);
      t.surprise      = +(t.realized_gain - t.predicted_gain).toFixed(2);
      t.net_after_hit = +(t.realized_gain - t.hit_cost).toFixed(2);
    }
  }

  return {
    season, variant: variantName,
    totalGross: withChips,
    totalHits,
    totalNet,
    transfersMade,
    // gh#38: 2-per-season chips. Legacy single-GW fields preserved for
    // backward-compat (tcGw = first fire, etc); new explicit pairs added.
    tcGw: tc1Gw || tc2Gw, bbGw: bb1Gw || bb2Gw, fhGw: fh1Gw || fh2Gw,
    tc1Gw, tc2Gw, bb1Gw, bb2Gw, fh1Gw, fh2Gw, wc1Gw, wc2Gw,
    hitThreshold,
    avgPerGw: totalNet / 38,
    initialSquadValue: BUDGET_TOTAL - budget,
    finalBudget: budget,
    pct: estimatePercentile(totalNet),
    trace: opts.trace ? traceLog : undefined,
    formationPolicy: opts.formationPolicy,
    formationTrace: formationLog,
    autoSubsFired,
    deadAssetSwaps,
    benchBlanksUncovered,
    ftBanked: transfersByGw.filter(t => !t.swapped).length
  };
}

async function main() {
  const args = process.argv.slice(2);
  const seasonArg = args.find(a => a.startsWith('--season='));
  const season = seasonArg ? seasonArg.split('=')[1] : '2024-25';
  const variantArg = args.find(a => a.startsWith('--variants='));
  const singleVariantArg = args.find(a => a.startsWith('--variant='));
  const variantList = variantArg
    ? variantArg.split('=')[1].split(',')
    : (singleVariantArg ? [singleVariantArg.split('=')[1]] : ['bps_magnet', 'ensemble', 'default', 'xg_underlying']);
  const lookaheadArg = args.find(a => a.startsWith('--lookahead='));
  const lookahead    = lookaheadArg ? parseInt(lookaheadArg.split('=')[1], 10) : 3;
  const htArg        = args.find(a => a.startsWith('--hit-threshold='));
  // Hit threshold = predicted-gain floor BEYOND the -4 cost of the hit. So 6 means
  // a hit needs gain > 4 (cost) + 6 (margin) = 10 over the planning horizon.
  // Real FPL community consensus: only take hits when predicted >= 8-10 over the
  // next 3-4 GWs. 6 default keeps us slightly aggressive; 10 is fully conservative.
  // Tournament-tuned defaults after issue #10 (haul-probs + ceiling TC) and
  // issue #11 (bookmaker odds): bps_magnet-conservative-value-safe is the
  // top-1 strategy at 2270 avg NET. Conservative = hitThreshold 10, minGain 5.
  // Default raised from 10 to 12 per issue #26 smoketest (#25 trace data showed
  // 2024-25 was wasting hits at ht=10; bumping cut waste while preserving the
  // good 2022-23 hits). Across 4 variants × 3 seasons: +409 NET total.
  // Raised 12 → 16 per the gh#34 full tournament (2026-05): banking 5 FTs
  // shifts the optimal hit-threshold up; bps_magnet-ht16-h8-b8-mpc3 won at
  // 2393 avg NET over 3 seasons.
  // Raised 16 → 20 per gh#52 ultra tournament (2026-05-24, 2880 sims with
  // formation as a control axis): max|bps_magnet-ht20-h9-b6-mpc3 won at
  // 2489 avg NET. Locked as the 2025-26 holdout baseline.
  // 2026-09-20: ht 20 -> 10. ht-full-grid (360 sims, 4 seasons incl. 2025-26,
  // variant x horizon x minGain x ht) put ht20 last of every ht mean (2364 vs
  // ht10 2396); the edge check (ht 6/8/10) is a flat plateau, so 10 is the
  // floor where the threshold stops binding, not a falling edge.
  const hitThreshold = htArg ? parseInt(htArg.split('=')[1], 10) : 10;
  const minGainArg   = args.find(a => a.startsWith('--min-gain='));
  const minGain      = minGainArg ? parseFloat(minGainArg.split('=')[1]) : 5.0;
  const hArg         = args.find(a => a.startsWith('--horizon='));
  // Issue #29: sweep over h ∈ {2,3,4,5,6} × b ∈ {2,4,6,8} on bps_magnet × 3
  // seasons. h=5 b=6 wins on stability: avg 2304 (vs h=4 b=4 = 2265), min 2248
  // (vs 2104), spread 86 (vs 249). For a whole season the floor matters
  // more than the peak; h=2 ties on avg but with huge variance (spread 357).
  // Bumped from 5 → 7 after issue #34 (FT cap 2 → 5 + N-transfer enumeration).
  // Banking 5 FTs only pays off if the planner can see past the bank window;
  // 5 bank GWs + 1 dump GW + 1 follow-up = 7. Cost is linear in horizon.
  // Bumped 7 → 8 per the gh#34 full tournament: h8 beat h6/h7 on mean NET.
  // Bumped 8 → 9 per gh#52 ultra: h9 was the modal best across top-20 cells.
  const planHorizon  = hArg ? parseInt(hArg.split('=')[1], 10) : 9;
  const bArg         = args.find(a => a.startsWith('--beam='));
  // Bumped 6 → 8 per the gh#34 full tournament (b8 took the peak strategy).
  // Dropped 8 → 6 per gh#52 ultra: b6 dominated top-20 (10/20 cells) once
  // formation was added as a control axis. Narrower beams concentrate search
  // on the actually-better moves under fluid formations.
  const beamWidth    = bArg ? parseInt(bArg.split('=')[1], 10) : 6;
  const applyNews    = args.includes('--news');
  const isArg        = args.find(a => a.startsWith('--initial='));
  const initialStrategy = isArg ? isArg.split('=')[1] : 'value';
  // --post-hoc-chips switches off live decisions and uses full-season hindsight
  // for TC/BB timing. Useful for measuring the lift of live decisions.
  const liveChips    = !args.includes('--post-hoc-chips');
  // Captain mode (21-step #11). Tournament finding: with current coefficients
  // (balanced=0.02/%, chase=0.04/%) the EO penalty is too gentle to matter:
  // safe avg=2111, balanced avg=2109, chase avg=2106 across 75 strategies each.
  // Default is now 'safe' (highest predicted, no EO penalty) until coefficients
  // are tuned. Set --mode=chase only if you raise the coefficient in the
  // captainScore eoCoef branch below; current values won't change anything.
  const modeArg      = args.find(a => a.startsWith('--mode='));
  const captainMode  = modeArg ? modeArg.split('=')[1] : 'safe';
  const trace        = args.includes('--trace');
  // gh#44: XI formation policy. One of the 8 formation names (locked), 'max'
  // (fluid naive), or 'robust' (fluid robustness-aware). Default 'max' (gh#47 Stage 2).
  const formationArg    = args.find(a => a.startsWith('--formation='));
  const formationPolicy = formationArg ? formationArg.split('=')[1] : 'max';

  if (!fs.existsSync(VCACHE)) { console.error(`Missing ${VCACHE}.`); process.exit(1); }
  const cache  = JSON.parse(fs.readFileSync(CACHE,  'utf8'));
  const vcache = JSON.parse(fs.readFileSync(VCACHE, 'utf8'));
  // H2H matchup cache (21-step #7) and congested-GW dampener (21-step #6).
  // For backtest, congestedGws is loaded for the SPECIFIC season being simulated.
  const loadOpt = (f) => { try { return JSON.parse(fs.readFileSync(path.join(DATA_DIR, f), 'utf8')); } catch (e) { return null; } };
  const congestedData = loadOpt('congested-gws.json');
  const congestedForSeason = congestedData ? (congestedData[season] || null) : null;
  const h2hCache = loadOpt('h2h-cache.json');
  // Manager-change dampener wiring (#6 / #8 step 1). predict.js's dampener
  // gates on ctx.managerChanges + ctx.teamShortById; without these it was
  // silently off in every backtest. teamShortById is derived from cache.teams
  // (current-season FPL short names); team_short_name is stable enough across
  // recent seasons that the manager-changes.json entries (AVL/WOL/EVE/TOT/MUN)
  // line up with current cache.teams.
  const managerChangesData = loadOpt('manager-changes.json');
  const managerChanges = (managerChangesData && managerChangesData.events) || null;
  const teamShortById = {};
  if (cache.teams) cache.teams.forEach(t => { teamShortById[t.id] = t.short_name; });
  const ctx = {
    calibration:  (cache.calibration && cache.calibration.factors) || null,
    teamStrength: cache.teamStrength || null,
    priors:       loadOpt('priors.json'),
    h2h:          h2hCache,
    congestedGws: congestedForSeason,
    haulProbs:    loadOpt('haul-probs.json'),
    managerChanges,
    teamShortById,
    currentSeason: season,
    _captainMode: captainMode
  };
  let ensembleWeights = null;
  if (fs.existsSync(EWEIGHT)) {
    try { ensembleWeights = JSON.parse(fs.readFileSync(EWEIGHT, 'utf8')).weights; } catch (e) {}
  }
  if (!ensembleWeights) ensembleWeights = DEFAULT_BLEND;

  const players = [];
  for (const key in vcache.history) {
    const entry = vcache.history[key];
    if (entry.season !== season) continue;
    players.push({ id: key, ...entry });
  }
  if (!players.length) { console.error(`No players for season ${season}.`); process.exit(1); }

  console.log(`Squad-sim ${season}. Plan horizon=${planHorizon} GWs, beam=${beamWidth}, hit-threshold=${hitThreshold}. Variants: ${variantList.join(', ')}.`);
  console.log(`${players.length} player-seasons in pool. Budget £100m, max 3/club, 1 free transfer/GW (5 banked max, 2024-25 rule).\n`);

  const results = [];
  for (const v of variantList) {
    if (v !== 'ensemble' && !VARIANTS[v]) { console.warn(`Unknown variant: ${v}`); continue; }
    process.stdout.write(`  ${v}... `);
    const t0 = Date.now();
    const r = simulate(players, season, v, ctx, ensembleWeights, { lookahead, hitThreshold, planHorizon, beamWidth, vcache, minGain, applyNews, initialStrategy, liveChips, trace, formationPolicy });
    if (!r) { console.log('skipped (could not build initial squad)'); continue; }
    console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    if (trace && r.trace) {
      const outPath = path.join(DATA_DIR, `planner-trace-${season}-${v}.json`);
      fs.writeFileSync(outPath, JSON.stringify({ season, variant: v, trace: r.trace }, null, 2));
      console.log(`    wrote ${r.trace.length} trace records to ${path.basename(outPath)}`);
    }
    results.push(r);
  }

  results.sort((a, b) => b.totalNet - a.totalNet);
  console.log(`\nResults (sorted by net, with hits):\n`);
  console.log(`  ${'variant'.padEnd(18)} ${'gross'.padStart(6)} ${'-hits'.padStart(6)} ${'NET'.padStart(7)} ${'(pctile)'.padStart(9)}  ${'tx'.padStart(3)}  ${'h_th'.padStart(4)}   TC1/TC2     BB1/BB2     FH1/FH2     WC1/WC2`);
  const fmtGw = g => g ? `GW${String(g).padStart(2, '0')}` : '----';
  for (const r of results) {
    const tcStr = `${fmtGw(r.tc1Gw)}/${fmtGw(r.tc2Gw)}`;
    const bbStr = `${fmtGw(r.bb1Gw)}/${fmtGw(r.bb2Gw)}`;
    const fhStr = `${fmtGw(r.fh1Gw)}/${fmtGw(r.fh2Gw)}`;
    const wcStr = `${fmtGw(r.wc1Gw)}/${fmtGw(r.wc2Gw)}`;
    console.log(
      `  ${r.variant.padEnd(18)} ${String(r.totalGross).padStart(6)} ${String(-r.totalHits).padStart(6)} ${String(r.totalNet).padStart(7)} ${('(top ' + r.pct + '%)').padStart(9)}  ${String(r.transfersMade).padStart(3)}  ${String(r.hitThreshold).padStart(4)}   ${tcStr}   ${bbStr}   ${fhStr}   ${wcStr}`
    );
  }

  console.log(`\nFor reference, the unlimited-transfer baseline (simulate-season.js) doesn't pay hits.`);
  console.log(`The "NET" figure here is what a real FPL manager actually finishes with.`);
}

if (require.main === module) {
  main().catch(e => { console.error(e); process.exit(1); });
}

// Module exports for tournament.js and other callers.
module.exports = { simulate, buildInitialSquad, SQUAD_STRATEGIES, BUDGET_TOTAL, MAX_PER_CLUB, SQUAD_QUOTA, FORMATION,
  // gh#44: formation policy names + table, exposed for formation-study.js.
  FORMATION_NAMES, LEGAL_FORMATIONS, pickXI,
  // Exposed for ml/build-dataset.js (gh#5). Additive only, no behaviour change.
  synthesisePlayer, predictWithVariant,
  // Exposed so live Free Hit pricing can be compared against the
  // backtest's on identical input (GW6 2026-27 post mortem).
  bestFreeHitXI, wildcardRebuild, bestLegalXI };
