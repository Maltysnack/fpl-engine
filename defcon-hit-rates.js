#!/usr/bin/env node
// gh#42: build the empirical DefCon hit-rate table from the 2025-26 archive.
//
// Reads Vaastav's 2025-26 archive (per-match defensive_contribution counts) and
// produces, per player, the fraction of 60+-minute appearances in which the
// player hit the FPL DefCon points threshold (DEF >= 10 actions, MID/FWD >= 12).
// That rate is what predict.js multiplies by 2pts to get the live-only DefCon EV
// term — replacing the old Poisson bridge, which could not be calibrated.
//
// Keyed by FPL's permanent `code` so the table survives the 2026-27 id reshuffle.
// Per-position pooled averages give a cold-start fallback for unseen players.
//
// USAGE: node defcon-hit-rates.js   ->   writes defcon-hit-rates.json

const fs   = require('fs');
const path = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const { parseCSV } = require('./vaastav-fetch');

const BASE = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26';
const OUT = path.join(DATA_DIR, 'defcon-hit-rates.json');

// element_type: 1 GK, 2 DEF, 3 MID, 4 FWD. GK is skipped (no DefCon scoring).
const thresholdFor = pos => (pos === 2 ? 10 : 12);
const MIN_MINUTES  = 60;

async function fetchCSV(file) {
  const res = await fetch(`${BASE}/${file}`);
  if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
  return parseCSV(await res.text());
}

// Joins per-match rows to the player's position and permanent code.
async function loadPlayers() {
  const players = {};
  for (const r of await fetchCSV('players_raw.csv')) {
    players[r.id] = { element_type: +r.element_type, code: r.code ? +r.code : null, history: [] };
  }
  for (const r of await fetchCSV('gws/merged_gw.csv')) {
    if (!players[r.element]) continue;
    players[r.element].history.push({
      minutes: +r.minutes,
      defensive_contribution: r.defensive_contribution === '' ? null : +r.defensive_contribution
    });
  }
  return players;
}

async function main() {
  const players = await loadPlayers();

  const by_code = {};
  // Pooled per-position accumulators: total hits / total eligible appearances.
  const posAcc = { 2: { hits: 0, elig: 0 }, 3: { hits: 0, elig: 0 }, 4: { hits: 0, elig: 0 } };

  let skippedGK = 0, skippedNoCode = 0, skippedNoElig = 0;

  for (const p of Object.values(players)) {
    const pos = p.element_type;
    if (pos === 1) { skippedGK++; continue; }          // GK: no DefCon points
    if (p.code == null) { skippedNoCode++; continue; }  // cannot key the table

    const threshold = thresholdFor(pos);
    let hits = 0, elig = 0;
    for (const h of p.history || []) {
      // A match can only be classified hit/miss if the player actually played a
      // meaningful share AND the count is recorded.
      if (h.minutes < MIN_MINUTES) continue;
      if (h.defensive_contribution == null) continue;
      elig++;
      if (h.defensive_contribution >= threshold) hits++;
    }
    if (elig === 0) { skippedNoElig++; continue; }      // no signal — omit

    by_code[p.code] = { rate: +(hits / elig).toFixed(4), n: elig, pos };
    if (posAcc[pos]) { posAcc[pos].hits += hits; posAcc[pos].elig += elig; }
  }

  const position_avg = {};
  for (const pos of [2, 3, 4]) {
    const a = posAcc[pos];
    position_avg[pos] = a.elig ? +(a.hits / a.elig).toFixed(4) : 0;
  }

  fs.writeFileSync(OUT, JSON.stringify({
    built_at:      new Date().toISOString(),
    source_season: '2025-26',
    by_code,
    position_avg
  }, null, 2));

  console.log(`Wrote ${OUT}`);
  console.log(`  ${Object.keys(by_code).length} players with a hit-rate.`);
  console.log(`  position_avg (pooled hits/elig): DEF ${position_avg[2]}, MID ${position_avg[3]}, FWD ${position_avg[4]}`);
  console.log(`  skipped: ${skippedGK} GK, ${skippedNoCode} no-code, ${skippedNoElig} no eligible (60+min) match.`);
}

main().catch(e => { console.error(e); process.exit(1); });
