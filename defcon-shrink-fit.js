#!/usr/bin/env node
/**
 * gh#58, empirical-Bayes shrinkage fitter for DefCon per-player hit-rates.
 *
 * Beta-binomial method-of-moments:
 *   Each player has (hits_i, n_i) with p_i = hits_i / n_i.
 *   Model: p_i ~ Beta(alpha, beta); X_i ~ Bin(n_i, p_i).
 *   Pooled mean:        mu = sum(hits) / sum(n)
 *   Observed var of p:  Var(p_i)
 *   Within-player var:  mu*(1-mu) * mean(1/n_i)        (binomial sampling)
 *   Between-player var: max(0, Var(p_i) - within)
 *   Prior strength:     n_eff = mu*(1-mu) / between - 1
 *
 * Shrunk estimate at predict time:
 *   rate_shrunk = (hits_i + mu * n_eff) / (n_i + n_eff)
 *
 * Inputs:  defcon-hit-rates.json (carryover; has by_code[code] = {rate, n, pos}).
 * Outputs: defcon-shrinkage.json = { [pos]: { mu, n_eff, sample_size, sources... } }.
 *
 * Filter: only fit using players with n >= MIN_N to avoid bootstrap noise.
 * Skip pos 1 (GK), no DefCon term applies.
 */
const fs = require('fs');
const path = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const MIN_N = 5;
const POS_LABEL = { 2: 'DEF', 3: 'MID', 4: 'FWD' };

const ROOT = path.join(DATA_DIR);
const carryover = JSON.parse(fs.readFileSync(path.join(ROOT, 'defcon-hit-rates.json'), 'utf8'));
const liveFile = fs.readdirSync(ROOT).filter(f => /^defcon-\d{4}-\d{2}\.json$/.test(f)).sort().pop();
const live = liveFile ? JSON.parse(fs.readFileSync(path.join(ROOT, liveFile), 'utf8')) : null;

// Gather rows per position from the carryover (always present).
// Live season can also feed once it exists, but 2025-26 carryover is the
// primary fit source and the only one available right now.
function rowsFromCarryover(c) {
  const out = { 2: [], 3: [], 4: [] };
  for (const code of Object.keys(c.by_code || {})) {
    const r = c.by_code[code];
    const pos = r.pos;
    if (!out[pos]) continue;
    if (r.n == null || r.rate == null) continue;
    out[pos].push({ rate: r.rate, n: r.n });
  }
  return out;
}

// Compute EB MoM (mu, n_eff) for a single position's rows.
function fitPosition(rows, label) {
  const eligible = rows.filter(r => r.n >= MIN_N);
  if (eligible.length < 5) {
    console.warn(`  ${label}: only ${eligible.length} eligible (n>=${MIN_N}). Skipping fit, using pooled mean only.`);
  }
  const totalHits = eligible.reduce((s, r) => s + r.rate * r.n, 0);
  const totalN    = eligible.reduce((s, r) => s + r.n, 0);
  const mu = totalN > 0 ? totalHits / totalN : 0;

  // Variance of p_i across eligible players.
  const meanP = eligible.reduce((s, r) => s + r.rate, 0) / Math.max(1, eligible.length);
  const varP  = eligible.length > 1
    ? eligible.reduce((s, r) => s + (r.rate - meanP) ** 2, 0) / (eligible.length - 1)
    : 0;
  // Within-player binomial variance averaged.
  const meanInvN = eligible.reduce((s, r) => s + 1 / r.n, 0) / Math.max(1, eligible.length);
  const within = mu * (1 - mu) * meanInvN;
  const between = Math.max(0, varP - within);

  let n_eff;
  if (between <= 0) {
    // All variance explained by sampling: prior is extremely strong.
    // Cap at a large but finite value so the shrunk estimate is well-defined.
    n_eff = 1000;
  } else {
    n_eff = mu * (1 - mu) / between - 1;
    if (!Number.isFinite(n_eff) || n_eff < 0) n_eff = 0;
  }
  return {
    mu: +mu.toFixed(6),
    n_eff: +n_eff.toFixed(2),
    sample_size: eligible.length,
    pooled_n: totalN,
    var_observed: +varP.toFixed(6),
    var_within: +within.toFixed(6),
    var_between: +between.toFixed(6)
  };
}

const carryRows = rowsFromCarryover(carryover);
const out = {
  built_at: new Date().toISOString(),
  source: 'defcon-hit-rates.json (2025-26 carryover)',
  min_n_filter: MIN_N,
  positions: {}
};

console.log('--- gh#58 DefCon EB shrinkage fit ---');
for (const pos of [2, 3, 4]) {
  const label = POS_LABEL[pos];
  const rows = carryRows[pos] || [];
  console.log(`\n${label} (pos=${pos}): ${rows.length} players total`);
  const fit = fitPosition(rows, label);
  out.positions[pos] = fit;
  console.log(`  mu = ${fit.mu}`);
  console.log(`  n_eff = ${fit.n_eff}`);
  console.log(`  eligible (n>=${MIN_N}) = ${fit.sample_size}`);
  console.log(`  variance: observed=${fit.var_observed} within=${fit.var_within} between=${fit.var_between}`);
}

fs.writeFileSync(path.join(ROOT, 'defcon-shrinkage.json'), JSON.stringify(out, null, 2));
console.log(`\nwrote defcon-shrinkage.json`);
