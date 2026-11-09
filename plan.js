#!/usr/bin/env node
// Plan a gameweek for your own team. No network access: you supply the data.
//
//   node plan.js <inputs.json> <team.csv> <state.json>
//
// inputs.json: players, fixtures and the gameweek (see SCHEMA.md).
// team.csv:    web_name,team,sell_price  (15 rows; sell_price in £m)
// state.json:  { "bank": 0.5, "free_transfers": 1, "gameweek": 11 }
//
// Prints the ranked plans (expected points this GW and over the horizon), the
// engine's pick and a captain. Predictions here use the formula and the ML
// residual; the private live pipeline also blends odds, news and a second model.
const fs = require('fs');
const { predictPlayer } = require('./predict.js');
const { VARIANTS } = require('./models.js');
const P = require('./planner.js');

const OPTS = { horizon: 8, beamWidth: 6, topKMoves: 5, hitThreshold: 10, hitCost: 4, formationPolicy: 'max' };
const MIN_GAIN = 5, MAX_PER_CLUB = 3;

function predictAll(inputs) {
  const gw0 = inputs.gameweek, byTeamGw = {};
  for (const f of inputs.fixtures) for (const t of [f.team_h, f.team_a]) (byTeamGw[`${t}:${f.event}`] = byTeamGw[`${t}:${f.event}`] || []).push(f);
  const ctx = Object.assign({}, inputs.context || {}, { variant: VARIANTS.bps_magnet, residual: true });
  const out = {};
  for (const p of inputs.players) {
    out[p.id] = [];
    for (let k = 0; k < OPTS.horizon; k++) {
      const fx = byTeamGw[`${p.team}:${gw0 + k}`] || [];
      out[p.id].push(fx.length ? predictPlayer(p, fx, Object.assign({}, ctx, { currentGw: gw0 + k })) : 0);
    }
  }
  return out;
}

function readTeam(csv, players) {
  const rows = csv.trim().split(/\r?\n/).slice(1).map(l => l.split(',').map(s => s.trim()));
  return rows.map(([name, team, sell]) => {
    const p = players.find(x => x.web_name === name && String(x.team) === team);
    if (!p) throw new Error(`team.csv: no player "${name}" for team ${team}`);
    return { id: p.id, element_type: p.element_type, team: p.team, price: Math.round(parseFloat(sell) * 10) };
  });
}

function main([inputsPath, teamPath, statePath]) {
  if (!statePath) { console.error('usage: node plan.js <inputs.json> <team.csv> <state.json>'); process.exit(2); }
  const inputs = JSON.parse(fs.readFileSync(inputsPath, 'utf8'));
  const st = JSON.parse(fs.readFileSync(statePath, 'utf8'));
  const squad = readTeam(fs.readFileSync(teamPath, 'utf8'), inputs.players);
  if (squad.length !== 15) throw new Error(`team.csv has ${squad.length} players, expected 15`);
  const preds = predictAll(inputs);
  const pool = inputs.players.filter(p => p.status !== 'u').map(p => ({ id: p.id, element_type: p.element_type, team: p.team, price: p.now_cost }));
  const clubCount = {}; for (const p of squad) clubCount[p.team] = (clubCount[p.team] || 0) + 1;
  const state = { freeTransfers: st.free_transfers, budget: Math.round(st.bank * 10), clubCount, minGain: MIN_GAIN, recentTransfers: [], currentGw: st.gameweek, _maxPerClub: MAX_PER_CLUB };
  const r = P.planAlternatives(squad, st.gameweek, null, null, null, pool, state, Object.assign({ lookaheadCache: preds }, OPTS));
  const name = id => (inputs.players.find(p => p.id === id) || {}).web_name || id;
  const moves = m => m.type === 'hold' ? 'No transfer' : (m.type === 'transfer' ? [m] : m.pairs).map(x => `${name(x.out.id)} > ${name(x.in.id)}`).join(', ');
  console.log('Plan | This GW | Horizon | Chosen');
  for (const a of r.alternatives) console.log(`${moves(a.move)} | ${a.first.toFixed(1)} | ${a.total.toFixed(1)} | ${a.chosen ? 'yes' : ''}`);
  const after = new Set(squad.map(p => p.id));
  if (r.chosen.type !== 'hold') for (const x of r.chosen.type === 'transfer' ? [r.chosen] : r.chosen.pairs) { after.delete(x.out.id); after.add(x.in.id); }
  const cap = [...after].sort((a, b) => preds[b][0] - preds[a][0])[0];
  console.log(`Captain: ${name(cap)} (${preds[cap][0].toFixed(2)} this GW)`);
}

if (require.main === module) { try { main(process.argv.slice(2)); } catch (e) { console.error(e.message); process.exit(1); } }
module.exports = { predictAll };
