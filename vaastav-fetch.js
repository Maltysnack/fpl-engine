#!/usr/bin/env node
/**
 * projects/fpl/vaastav-fetch.js
 *
 * Imports historical FPL data from Vaastav's archive
 * (github.com/vaastav/Fantasy-Premier-League).
 *
 * Vaastav publishes per-season CSVs of FPL data, committed
 * to git. We pull per-GW gameweek files plus the player metadata file for each
 * season we want, normalise into our `history-cache.json` shape, and write to
 * `vaastav-cache.json`. Downstream walk-forward.js and fit.js merge it in.
 *
 * IMPORTANT: schema drifts across seasons. xGI / xGC only exist from ~2021-22.
 * For seasons older than that, those fields are 0 and the record is flagged
 * lowFidelity=true so fit.js can down-weight it. Set-piece order isn't in
 * the per-GW files anywhere; we leave it null and accept that as a known gap
 * (predict.js gracefully no-ops the set-piece bonus when fields are absent).
 *
 * USAGE
 *   node projects/fpl/vaastav-fetch.js
 *   node projects/fpl/vaastav-fetch.js --seasons=2023-24,2024-25
 *   node projects/fpl/vaastav-fetch.js --seasons=2023-24 --refresh
 *
 * Default seasons are the two most recent that have full xGI coverage.
 */

const https = require('https');
const fs    = require('fs');
const path  = require('path');
const DATA_DIR = process.env.FPL_DATA_DIR || __dirname;

const BASE = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data';
const OUT  = path.join(DATA_DIR, 'vaastav-cache.json');

const DEFAULT_SEASONS = ['2023-24', '2024-25'];

function fetchText(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      timeout: 30000,
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; FPL-history/1.0)' }
    }, res => {
      if (res.statusCode === 404) { resolve(null); return; }   // gw doesn't exist yet
      if (res.statusCode !== 200) { reject(new Error(`HTTP ${res.statusCode} for ${url}`)); return; }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
    });
    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('Timeout')));
  });
}

async function fetchWithRetry(url, attempts = 3) {
  for (let i = 1; i <= attempts; i++) {
    try { return await fetchText(url); }
    catch (e) {
      if (i === attempts) throw e;
      await new Promise(r => setTimeout(r, 1500 * i));
    }
  }
}

// Minimal CSV parser. Handles quoted fields with commas and escaped quotes.
function parseLine(line) {
  const cells = [];
  let cur = '', inQuote = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (inQuote && line[i + 1] === '"') { cur += '"'; i++; }
      else inQuote = !inQuote;
    } else if (c === ',' && !inQuote) {
      cells.push(cur); cur = '';
    } else {
      cur += c;
    }
  }
  cells.push(cur);
  return cells;
}

function parseCSV(text) {
  if (!text) return [];
  const lines = text.split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = parseLine(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const cells = parseLine(lines[i]);
    const row = {};
    for (let j = 0; j < headers.length; j++) row[headers[j]] = cells[j];
    rows.push(row);
  }
  return rows;
}

const num = v => { const n = parseFloat(v); return isFinite(n) ? n : 0; };
const int = v => { const n = parseInt(v, 10); return isFinite(n) ? n : 0; };

// element_type mapping in Vaastav players_raw.csv is the same FPL convention:
// 1 = GKP, 2 = DEF, 3 = MID, 4 = FWD.
function buildPlayerMeta(rows) {
  const meta = {};
  for (const r of rows) {
    const id = int(r.id);
    if (!id) continue;
    meta[id] = {
      web_name: r.web_name || `${r.first_name || ''} ${r.second_name || ''}`.trim(),
      team: int(r.team),
      element_type: int(r.element_type),
      // End-of-season ownership %. Used by template-aware ranking. Per-GW values
      // exist in gws/gw{N}.csv but we'd need to thread them through; this snapshot
      // is sufficient for the leaderboard-based prior.
      selected_by_percent: parseFloat(r.selected_by_percent) || 0,
      total_points: int(r.total_points),
      now_cost: int(r.now_cost),
      // End-of-season Team-of-the-Week appearances. Aggregate "consistently elite" signal.
      dreamteam_count: int(r.dreamteam_count)
    };
  }
  return meta;
}

// Map a Vaastav gw{N}.csv row to our history-entry shape. Returns null if the
// row doesn't have the bare minimum (round + minutes).
function rowToEntry(r, hasXGI) {
  const round = int(r.round || r.GW);
  if (!round) return null;
  return {
    round,
    fixture: int(r.fixture),
    opponent_team: int(r.opponent_team),
    was_home: r.was_home === 'True' || r.was_home === 'true' || r.was_home === true,
    total_points: int(r.total_points),
    minutes: int(r.minutes),
    goals_scored: int(r.goals_scored),
    assists: int(r.assists),
    clean_sheets: int(r.clean_sheets),
    saves: int(r.saves),
    bps: int(r.bps),
    bonus: int(r.bonus),
    expected_goal_involvements: hasXGI ? num(r.expected_goal_involvements) : 0,
    expected_goals_conceded:    hasXGI ? num(r.expected_goals_conceded)    : 0,
    // value: in-game player price at this GW, in 0.1m units (80 = £8.0m). Critical
    // for squad-sim.js budget tracking. Falls back to 0 if absent.
    value: int(r.value),
    // FPL Team of the Week flag for this player in this gameweek.
    in_dreamteam: r.in_dreamteam === 'True' || r.in_dreamteam === 'true' || r.in_dreamteam === true
  };
}

async function importSeason(season) {
  console.log(`\n[${season}] Fetching players_raw.csv...`);
  const playersTxt = await fetchWithRetry(`${BASE}/${season}/players_raw.csv`);
  if (!playersTxt) { console.warn(`  players_raw.csv missing, skipping season.`); return null; }
  const playerMeta = buildPlayerMeta(parseCSV(playersTxt));
  console.log(`  ${Object.keys(playerMeta).length} players in metadata.`);

  // Detect xGI availability via a peek at gw1.csv headers.
  const gw1 = await fetchWithRetry(`${BASE}/${season}/gws/gw1.csv`);
  if (!gw1) { console.warn(`  No gw1.csv, skipping season.`); return null; }
  const gw1Rows = parseCSV(gw1);
  const hasXGI = gw1Rows.length > 0 && 'expected_goal_involvements' in gw1Rows[0];
  console.log(`  xGI coverage: ${hasXGI ? 'yes' : 'NO (lowFidelity)'}`);

  const byPlayer = {};   // namespaced player_id -> { meta, history: [] }

  function consumeRows(rows) {
    for (const r of rows) {
      const elId = int(r.element);
      if (!elId) continue;
      const meta = playerMeta[elId];
      if (!meta) continue;
      const entry = rowToEntry(r, hasXGI);
      if (!entry || entry.minutes === 0) continue;   // skip non-appearances
      const key = `${season}_e${elId}`;
      if (!byPlayer[key]) {
        byPlayer[key] = {
          web_name: meta.web_name,
          team: meta.team,
          element_type: meta.element_type,
          selected_by_percent: meta.selected_by_percent,
          total_points_season: meta.total_points,
          now_cost: meta.now_cost,
          dreamteam_count: meta.dreamteam_count || 0,
          season,
          lowFidelity: !hasXGI,
          history: []
        };
      }
      byPlayer[key].history.push(entry);
    }
  }

  // gw1 we already fetched, reuse.
  consumeRows(gw1Rows);

  // Walk forward until 404 (lets us handle in-progress seasons gracefully).
  for (let gw = 2; gw <= 38; gw++) {
    const txt = await fetchWithRetry(`${BASE}/${season}/gws/gw${gw}.csv`);
    if (!txt) { console.log(`  GW${gw} missing (404), stopping.`); break; }
    const rows = parseCSV(txt);
    consumeRows(rows);
    if (gw % 5 === 0) console.log(`  through GW${gw} (${Object.keys(byPlayer).length} players seen)`);
    await new Promise(r => setTimeout(r, 100));   // politeness delay
  }

  // Sort each player's history by round so cumulative synthesis works downstream.
  for (const key in byPlayer) byPlayer[key].history.sort((a, b) => a.round - b.round);

  console.log(`  [${season}] done: ${Object.keys(byPlayer).length} players with appearances.`);
  return byPlayer;
}

async function main() {
  const args = process.argv.slice(2);
  const sArg = args.find(a => a.startsWith('--seasons='));
  const seasons = sArg ? sArg.split('=')[1].split(',').map(s => s.trim()) : DEFAULT_SEASONS;
  const refresh = args.includes('--refresh');

  let existing = {};
  if (!refresh && fs.existsSync(OUT)) {
    try { existing = JSON.parse(fs.readFileSync(OUT, 'utf8')).history || {}; }
    catch (e) { /* corrupt, start over */ }
  }

  const combined = { ...existing };
  let totalNew = 0;
  for (const season of seasons) {
    const data = await importSeason(season);
    if (!data) continue;
    let added = 0;
    for (const key in data) {
      if (!combined[key]) added++;
      combined[key] = data[key];
    }
    totalNew += added;
  }

  const out = {
    updatedAt: new Date().toISOString(),
    seasons,
    sampleSize: Object.keys(combined).length,
    history: combined
  };
  fs.writeFileSync(OUT, JSON.stringify(out));
  const kb = Math.round(fs.statSync(OUT).size / 1024);
  console.log(`\nWrote ${OUT} (${kb} KB). ${Object.keys(combined).length} player-seasons total, ${totalNew} new.`);
}

if (require.main === module) main().catch(e => { console.error(e); process.exit(1); });

module.exports = { parseCSV };
