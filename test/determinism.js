#!/usr/bin/env node
// Same inputs, same outputs: predictions and the planner's move on a fixed
// synthetic slate must match test/golden.json exactly. `--update` rewrites it.
const fs = require('fs'), path = require('path');
const { predictPlayer } = require('../predict.js');
const P = require('../planner.js');

function slate() {
  const players = [];
  for (let id = 1; id <= 60; id++) {
    const pos = id <= 6 ? 1 : id <= 24 ? 2 : id <= 48 ? 3 : 4;
    players.push({
      id, code: 1000 + id, web_name: `P${id}`, team: (id % 20) + 1, element_type: pos,
      now_cost: 40 + (id * 7) % 90, minutes: 400 + (id * 53) % 500, total_points: 10 + (id * 13) % 60,
      form: String(((id * 3) % 70) / 10), ep_next: String(((id * 5) % 80) / 10), chance_of_playing_next_round: id % 17 === 0 ? 50 : null,
      goals_scored: id % 5, assists: id % 4, clean_sheets: id % 3, bps: 50 + id, saves: pos === 1 ? id * 3 : 0,
      expected_goal_involvements_per_90: (id % 9) / 20, expected_goals_conceded_per_90: 1 + (id % 5) / 10,
      defensive_contribution: id % 11, dream_team_count: id % 2, status: 'a'
    });
  }
  const fixFor = (team, gw) => [{ event: gw, team_h: team, team_a: (team % 20) + 1, team_h_difficulty: 2 + (team + gw) % 4, team_a_difficulty: 2 + (team * gw) % 4 }];
  return { players, fixFor };
}

function run() {
  const { players, fixFor } = slate();
  const preds = {};
  for (const p of players) preds[p.id] = [10, 11, 12, 13, 14, 15, 16, 17].map(gw => predictPlayer(p, fixFor(p.team, gw), { currentGw: gw }));
  const pick = (pos, n, from) => players.filter(p => p.element_type === pos).slice(from, from + n);
  const squad = [...pick(1, 2, 0), ...pick(2, 5, 0), ...pick(3, 5, 0), ...pick(4, 3, 0)]
    .map(p => ({ id: p.id, element_type: p.element_type, team: p.team, price: p.now_cost }));
  const pool = players.map(p => ({ id: p.id, element_type: p.element_type, team: p.team, price: p.now_cost }));
  const clubCount = {}; for (const p of squad) clubCount[p.team] = (clubCount[p.team] || 0) + 1;
  const state = { freeTransfers: 1, budget: 20, clubCount, minGain: 5, recentTransfers: [], currentGw: 10, _maxPerClub: 3 };
  const opts = { lookaheadCache: preds, horizon: 8, beamWidth: 6, topKMoves: 5, hitThreshold: 10, hitCost: 4, formationPolicy: 'max' };
  const move = P.plan(squad, 10, null, null, null, pool, state, opts);
  const alts = P.planAlternatives(squad, 10, null, null, null, pool, state, opts).alternatives
    .map(a => ({ size: a.size, first: +a.first.toFixed(4), total: +a.total.toFixed(4), chosen: a.chosen }));
  return { preds, move, alts };
}

module.exports = { slate };
if (require.main === module) {
  const golden = path.join(__dirname, 'golden.json');
  const a = JSON.stringify(run()), b = JSON.stringify(run());
  if (a !== b) { console.error('FAIL: two runs in one process differ'); process.exit(1); }
  if (process.argv.includes('--update')) { fs.writeFileSync(golden, JSON.stringify(JSON.parse(a), null, 1) + '\n'); console.log('golden updated'); process.exit(0); }
  if (JSON.stringify(JSON.parse(fs.readFileSync(golden, 'utf8'))) !== a) { console.error('FAIL: output differs from test/golden.json'); process.exit(1); }
  console.log('determinism OK');
}
