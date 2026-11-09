// Build a single 15-man FPL squad optimized across two upcoming GWs.
//
// The model's "best XI for GW+1" and "best XI for GW+2" are separate top-N
// picks that ignore squad continuity. This module solves the harder problem:
// pick 15 players (2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per club, <= £100m)
// such that sum over GWs of (best legal XI + captain doubled) is maximized.
//
// Algorithm:
//   1. Greedy seed: sort candidates by pred_next + pred_gw2, fill quotas
//      respecting budget + club cap.
//   2. Pairwise swap: for each (in, out) pair where positions match, swap if
//      total objective improves. Iterate until no swap helps.
//
// The pool is small (~200 candidates after filtering) and the swap loop
// converges in well under a second.

const POS_QUOTA = { 1: 2, 2: 5, 3: 5, 4: 3 };
const BUDGET    = 1000;  // tenths of £m
const MAX_CLUB  = 3;

// gh#33/gh#35 (go-live): the builder is generic over which prediction keys it
// sums and what budget it has, so the same local search builds the public
// two-GW squad, a Free Hit 15 (one key, squad value + bank) and a Wildcard 15
// (eight keys). `benchWeight` is a small tie-break so equal-cost bench slots
// prefer players who actually play (the Neave lesson, GOLIVE-PLAN Gate 3);
// it is far too small to buy an expensive bench.
let _keys = ['pred_next', 'pred_gw2'];
let _budget = BUDGET;
let _benchWeight = 0;

// Valid XI formations: 1 GKP plus (def, mid, fwd) with constraints.
const FORMATIONS = (() => {
  const out = [];
  for (let d = 3; d <= 5; d++) {
    for (let f = 1; f <= 3; f++) {
      const m = 10 - d - f;
      if (m >= 2 && m <= 5) out.push({ d, m, f });
    }
  }
  return out;
})();

function bestXIForGw(squad, predKey) {
  const byPos = { 1: [], 2: [], 3: [], 4: [] };
  squad.forEach(p => byPos[p.element_type].push(p));
  for (const k of [1, 2, 3, 4]) byPos[k].sort((a, b) => (b[predKey] || 0) - (a[predKey] || 0));

  let best = { xi: null, total: -Infinity, formation: null, captain: null };
  const gkp = byPos[1][0];

  for (const f of FORMATIONS) {
    const defs = byPos[2].slice(0, f.d);
    const mids = byPos[3].slice(0, f.m);
    const fwds = byPos[4].slice(0, f.f);
    if (defs.length < f.d || mids.length < f.m || fwds.length < f.f) continue;

    const xi  = [gkp, ...defs, ...mids, ...fwds];
    const cap = xi.reduce((best, p) => ((p[predKey] || 0) > (best[predKey] || 0) ? p : best));
    const sum = xi.reduce((s, p) => s + (p[predKey] || 0), 0) + (cap[predKey] || 0); // captain doubled

    if (sum > best.total) {
      best = { xi, total: sum, formation: `${f.d}-${f.m}-${f.f}`, captain: cap };
    }
  }
  return best;
}

function objective(squad) {
  const per = _keys.map(k => bestXIForGw(squad, k));
  let total = per.reduce((s, r) => s + r.total, 0);
  if (_benchWeight) {
    for (let i = 0; i < per.length; i++) {
      const xiIds = new Set((per[i].xi || []).map(p => p.id));
      total += _benchWeight * squad.filter(p => !xiIds.has(p.id)).reduce((s, p) => s + (p[_keys[i]] || 0), 0);
    }
  }
  return { total, gw1: per[0], gw2: per[1] || per[0], per };
}

function squadIsLegal(squad) {
  const cost = squad.reduce((s, p) => s + p.now_cost, 0);
  if (cost > _budget) return false;
  const posCount = { 1: 0, 2: 0, 3: 0, 4: 0 };
  const clubCount = {};
  for (const p of squad) {
    posCount[p.element_type]++;
    clubCount[p.team] = (clubCount[p.team] || 0) + 1;
    if (clubCount[p.team] > MAX_CLUB) return false;
  }
  for (const [pos, want] of Object.entries(POS_QUOTA)) {
    if (posCount[pos] !== want) return false;
  }
  return true;
}

function greedySeed(pool, rank = 'total') {
  const byPos = { 1: [], 2: [], 3: [], 4: [] };
  pool.forEach(p => byPos[p.element_type].push(p));
  const tot = p => _keys.reduce((s, k) => s + (p[k] || 0), 0);
  // 'ratio' seeds by points per £m so budget spreads across positions; the
  // pure 'total' seed can over-spend on DEF/MID and strand the FWD line on
  // fodder, a local optimum the pairwise swap cannot climb out of.
  const score = rank === 'ratio' ? (p => tot(p) / Math.max(1, p.now_cost)) : tot;
  for (const k of [1, 2, 3, 4]) byPos[k].sort((a, b) => score(b) - score(a));

  const picked = [];
  const clubCount = {};
  let budget = _budget;
  const slotsLeft = { 1: POS_QUOTA[1], 2: POS_QUOTA[2], 3: POS_QUOTA[3], 4: POS_QUOTA[4] };

  for (const pos of [1, 2, 3, 4]) {
    for (const p of byPos[pos]) {
      if (!slotsLeft[pos]) break;
      if (p.now_cost > budget - cheapestRemaining(slotsLeft, pos, p, byPos)) continue;
      if ((clubCount[p.team] || 0) >= MAX_CLUB) continue;
      picked.push(p);
      clubCount[p.team] = (clubCount[p.team] || 0) + 1;
      budget -= p.now_cost;
      slotsLeft[pos]--;
    }
  }
  return picked;
}

// Reserve enough budget for the cheapest remaining slot at each position.
function cheapestRemaining(slotsLeft, currentPos, currentPlayer, byPos) {
  let reserve = 0;
  for (const pos of [1, 2, 3, 4]) {
    let need = slotsLeft[pos];
    if (pos === currentPos) need -= 1;
    if (need <= 0) continue;
    const cheap = byPos[pos]
      .filter(p => p.id !== currentPlayer.id)
      .sort((a, b) => a.now_cost - b.now_cost)
      .slice(0, need);
    reserve += cheap.reduce((s, p) => s + p.now_cost, 0);
  }
  return reserve;
}

function swapImprove(squad, pool) {
  const inSquad = new Set(squad.map(p => p.id));
  let improved = true;
  let iter = 0;
  while (improved && iter < 20) {
    improved = false;
    iter++;
    let best = objective(squad).total;

    for (let i = 0; i < squad.length; i++) {
      const out = squad[i];
      for (const inP of pool) {
        if (inSquad.has(inP.id)) continue;
        if (inP.element_type !== out.element_type) continue;
        const trial = squad.slice();
        trial[i] = inP;
        if (!squadIsLegal(trial)) continue;
        const score = objective(trial).total;
        if (score > best + 1e-6) {
          squad = trial;
          inSquad.delete(out.id);
          inSquad.add(inP.id);
          best = score;
          improved = true;
          break;
        }
      }
      if (improved) break;
    }
  }
  return squad;
}

// Generic entry: build the best legal 15 from `pool` under `budget` (0.1m
// units) maximising the summed best-XI (+captain) over `keys`. Returns
// { squad, cost, total, per } or null if no legal 15 exists.
// `seeds`: extra starting squads (e.g. the current 15) so the result is never
// worse than what we already hold. Best of all seeds after local search wins.
function buildSquad(pool, { keys, budget, benchWeight = 0.05, seeds = [] } = {}) {
  _keys = keys && keys.length ? keys : ['pred_next', 'pred_gw2'];
  _budget = budget || BUDGET;
  _benchWeight = benchWeight;
  try {
    if (pool.length < 15) return null;
    // A seed may contain players outside the pool (the current squad's dead
    // or flagged assets). Swap each for the cheapest same-position pool
    // player that keeps the club cap, so local search starts from a legal,
    // fully playable 15 and can never carry a dead asset through a rebuild.
    const inPool = new Set(pool.map(p => p.id));
    const sanitise = seed => {
      const out = seed.slice();
      for (let i = 0; i < out.length; i++) {
        if (inPool.has(out[i].id)) continue;
        const ids = new Set(out.map(p => p.id));
        const clubs = {}; out.forEach((p, j) => { if (j !== i) clubs[p.team] = (clubs[p.team] || 0) + 1; });
        const cand = pool.filter(p => p.element_type === out[i].element_type && !ids.has(p.id) && (clubs[p.team] || 0) < MAX_CLUB)
          .sort((a, b) => a.now_cost - b.now_cost || (b[_keys[0]] || 0) - (a[_keys[0]] || 0))[0];
        if (!cand) return null;
        out[i] = cand;
      }
      return out;
    };
    const starts = [greedySeed(pool, 'total'), greedySeed(pool, 'ratio')].concat(seeds.map(sanitise));
    let best = null;
    for (const seed of starts) {
      if (!seed || seed.length !== 15 || !squadIsLegal(seed)) continue;
      const squad = swapImprove(seed, pool);
      const obj = objective(squad);
      if (!best || obj.total > best.total) best = { squad, cost: squad.reduce((s, p) => s + p.now_cost, 0), total: obj.total, per: obj.per };
    }
    return best;
  } finally {
    _keys = ['pred_next', 'pred_gw2']; _budget = BUDGET; _benchWeight = 0;
  }
}

function buildTwoGwSquad(elements) {
  // Final GW of the season: nothing has a GW+2 prediction. Fall back to
  // single-GW mode so the public page still has a squad to render.
  const hasGw2 = elements.some(p => (p.pred_gw2 || 0) > 0);
  const pool = elements.filter(p =>
    p.minutes > 0 &&
    p.status !== 'u' &&
    p.status !== 'i' &&
    (p.pred_next || 0) > 0 &&
    (hasGw2 ? (p.pred_gw2 || 0) > 0 : true)
  );
  if (pool.length < 15) return null;
  if (!hasGw2) pool.forEach(p => { p.pred_gw2 = p.pred_next; });

  const seed = greedySeed(pool);
  if (seed.length !== 15 || !squadIsLegal(seed)) return null;

  const squad = swapImprove(seed, pool);
  const obj   = objective(squad);
  const cost  = squad.reduce((s, p) => s + p.now_cost, 0);

  const compact = p => ({ id: p.id, web_name: p.web_name, team: p.team, element_type: p.element_type, now_cost: p.now_cost });

  return {
    squad: squad.map(compact),
    cost,
    predicted_total: +obj.total.toFixed(2),
    gw1: {
      formation:      obj.gw1.formation,
      xi:             obj.gw1.xi.map(p => p.id),
      captain:        obj.gw1.captain.id,
      predicted:      +obj.gw1.total.toFixed(2)
    },
    gw2: {
      formation:      obj.gw2.formation,
      xi:             obj.gw2.xi.map(p => p.id),
      captain:        obj.gw2.captain.id,
      predicted:      +obj.gw2.total.toFixed(2)
    }
  };
}

module.exports = { buildTwoGwSquad, buildSquad, bestXIForGw };
