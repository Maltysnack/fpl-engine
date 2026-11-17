// Shared FPL prediction engine. Loaded by both sync.js (Node) and index.html (browser).
// Single source of truth for the formula, so calibration learnt in Node lines up with
// what the browser renders.
//
// Inputs:
//   player  : player record (see the fields read below)
//   fixList : array of fixtures for this player's team (length>1 = double GW)
//   ctx     : { oddsMap?, calibration?, minutesHistory?, teamStrength?, lineupOverride?, priors? }
//             oddsMap[teamId]    = { winProb, pOver25, pBTTSno }  (built by sync.js
//                                  from odds-fetch.js's odds-cache.json; issue #11)
//             calibration[posId] = number | { low, mid, high }
//             minutesHistory[playerId] = [{ gw, mins }, ...]      (rolling window)
//             teamStrength = { _leagueAvg, [teamId]: { homeAttack, awayAttack, homeDefence, awayDefence } }
//             lineupOverride[playerId] = 0 | 100 (manual confirmed/excluded after lineups drop)
//             newsOverride[playerId] = { chance: 0..100, source } (from news-scrape.js;
//                 trumps player.chance_of_playing_next_round when present)
//             priors = { slope, intercept, posOffset: { 1, 2, 3, 4 } } (stage-2 OLS, see fit.js)
//             h2h = { <normalized_web_name>: { overall_ppg, vs_opponent: { <opp_id>: { ppg, n } } } }
//                   (built by h2h-build.js; applies player-vs-opponent matchup edge)
//             defconLive = { season, by_code } season-to-date DefCon archive (optional)
//             congestedGws = { <gw_id>: avail_multiplier } for the current season
//                   (built by fixture-density.js; PL fixture-density rotation dampener)
//
// Returns: predicted FPL points for the next GW (or sum across the fixList window).

(function () {
  // ML residual model (gh#5 Phase 4). Node-only: the browser bundle never
  // loads it, so in-browser predictions stay the bare formula. Gated per
  // call by ctx.residual === true; every other caller is unaffected.
  let residualModel = null;
  if (typeof require !== 'undefined') {
    try { residualModel = require('./ml/residual-model.js'); }
    catch (e) { residualModel = null; }
  }

  // gh#42: empirical DefCon hit-rate tables (Node-only, the browser bundle
  // keeps the Poisson cold-start fallback). `carryover` is the 2025-26 archive
  // (defcon-hit-rates.js output); the season-to-date archive arrives per call
  // as ctx.defconLive (optional). A missing file never crashes predict.js.
  let defconCarryover = null, defconShrinkage = null;
  if (typeof require !== 'undefined') {
    try { defconCarryover = require('./defcon-hit-rates.json'); }
    catch (e) { defconCarryover = null; }
    // gh#58: per-position empirical-Bayes priors for DefCon hit-rate shrinkage.
    // Built offline by `defcon-shrink-fit.js`. Missing file -> no shrinkage,
    // raw rates flow through (degrades gracefully).
    try { defconShrinkage = require('./defcon-shrinkage.json'); }
    catch (e) { defconShrinkage = null; }
  }

  // gh#55: per-position means for empirical-Bayes shrinkage of per-90 /
  // per-game features (csRate, savesPer90, goalsPer90, assistsPer90, xgi90,
  // xgc90). Built offline by `per90-shrink-fit.js`. Missing file -> no
  // shrinkage, raw values flow through unchanged.
  let per90Shrinkage = null;
  if (typeof require !== 'undefined') {
    try { per90Shrinkage = require('./per90-shrinkage.json'); }
    catch (e) { per90Shrinkage = null; }
  }

  // EB shrinkage helper for a per-90 / per-game feature.
  //   shrunk = (raw * n + k * mean_pos) / (n + k)
  // Returns raw unchanged if priors aren't loaded or the feature isn't
  // listed for this position.
  function shrinkPer90(raw, n, pos, feature) {
    if (!per90Shrinkage || !per90Shrinkage.means) return raw;
    const m = per90Shrinkage.means[pos];
    if (!m || m[feature] == null) return raw;
    const k = per90Shrinkage.k != null ? per90Shrinkage.k : 5;
    return (raw * n + k * m[feature]) / (n + k);
  }

  // Default weight set. A `variant` passed via ctx can override any subset.
  // The same predictPlayer code path runs for every variant; only these numbers
  // change. Keeps the ensemble cheap (no code branching per variant).
  const DEFAULTS = {
    recency_ep:     0.55,
    recency_form:   0.30,
    recency_season: 0.15,
    gkp_recency:    0.70, gkp_cs:  0.20, gkp_saves: 0.10,
    def_recency:    0.70, def_cs:  0.20, def_att:   0.10,
    mid_recency:    0.65, mid_att: 0.35,
    fwd_recency:    0.65, fwd_att: 0.35,
    pen1: 1.0,  pen2: 0.3,  fk1: 0.20, ck1: 0.25,
    bps_scale: 0.012,
    defcon_confidence: 0.7,  // haircut on the DefCon EV term, it can't be backtested
    rollmins_very_low_thresh: 10, rollmins_very_low_mult: 0.10,  // gh#57: 0 mins recent -> strong discount
    rollmins_low_thresh: 30, rollmins_low_mult: 0.5,
    rollmins_mid_thresh: 60, rollmins_mid_mult: 0.85,
    // Blend between xGI-derived attRet and actual goals/assists per-90.
    // 1.0 = pure xGI (regression-to-mean believer).
    // 0.0 = pure actuals (lucky finishers get credit).
    // 0.5 = balanced (default; matches what predict.js used to do hardcoded).
    att_xgi_blend: 0.5,
    // Team of the Week scaling. Each dream_team_count appearance is a "consistently
    // elite GW" badge. A player with 10 TOTW appearances over 30 GWs (~33% TOTW
    // rate) adds dream_team_scale * 10 = 0.8 pts to base prediction, capped at
    // dream_team_cap. Captures the "yes Salah keeps showing up" pattern that
    // the per-GW signals (xGI, form) sometimes underweight.
    dream_team_scale: 0.08, dream_team_cap: 1.5
  };

  function applyVariant(variant) {
    if (!variant) return DEFAULTS;
    const out = {};
    for (const k in DEFAULTS) out[k] = variant[k] != null ? variant[k] : DEFAULTS[k];
    return out;
  }

  function tierFor(rawPred) {
    if (rawPred < 3) return 'low';
    if (rawPred < 6) return 'mid';
    return 'high';
  }

  function lookupCalibration(calibration, posId, tier) {
    if (!calibration) return 1.0;
    const slot = calibration[posId];
    if (slot == null) return 1.0;
    if (typeof slot === 'number') return slot;
    return slot[tier] != null ? slot[tier] : (slot.mid != null ? slot.mid : 1.0);
  }

  // Normalise a web_name for h2h lookup. Mirrors h2h-build.js's norm().
  function normName(s) {
    return s ? String(s).toLowerCase().replace(/[^a-z]/g, '') : '';
  }

  // gh#67: `currentGw` is the bound that makes this safe against a cache which
  // has mixed seasons together. GW ids restart at 1 each season, so a row for a
  // GW at or beyond the one being predicted cannot belong to this season, it is
  // prior-season carryover. Sorting alone does not save us: sorted ascending,
  // last season's GW31-37 land at the END and win the slice over this season's
  // GW1. sync.js keeps the cache clean; this is the second line of defence for
  // any other caller that builds minutesHistory itself.
  function rollingMinutes(history, currentGw) {
    if (!history || history.length === 0) return null;
    let rows = history.slice().sort((a, b) => a.gw - b.gw);
    if (currentGw != null) rows = rows.filter(h => h.gw < currentGw);
    if (rows.length === 0) return null;
    // gh#63: a player who has RETURNED from an absence should not be priced as
    // if still absent. The empirical curves (docs/injury-easing-report.md, 370
    // events) put the median minutes ratio at ~1.0 from the return GW onward
    // and the mean points ratio at 1.0-1.4, so the right treatment is to drop
    // the absence rows, not to add an easing multiplier. Returner = the latest
    // row is a real appearance (>= 45 mins) that ends a run of >= 3 zero-minute
    // rows. Those zeros are a finished absence, not rotation evidence; the
    // window is then the pre-absence appearances plus everything since.
    const lastIdx = rows.length - 1;
    if ((rows[lastIdx].mins || 0) >= 45) {
      let z = lastIdx - 1;
      while (z >= 0 && (rows[z].mins || 0) === 0) z--;
      const zeros = lastIdx - 1 - z;
      if (zeros >= 3) rows = rows.slice(0, z + 1).concat(rows.slice(lastIdx));
    }
    const last = rows.slice(-5);
    let sum = 0;
    for (const h of last) sum += (h.mins || 0);
    return { avg: sum / last.length, n: last.length };
  }

  // gh#69: how much of the rolling-minutes penalty to actually apply, given how
  // many GWs we have observed. The gh#57 tiers were fit against a full 5-GW
  // window, where an average below 10 minutes really does mean "not playing".
  // Early in a season there may be a single observation, and one rotated or
  // rested appearance is not evidence of that, applying the full 90% write-down
  // off n=1 crushed 269 available players at GW2 2026-27.
  //
  // Ramp the penalty in linearly over that same 5-GW window, so it reaches FULL
  // designed strength at n >= 5 and mid-season behaviour is bit-identical to
  // pre-gh#69. Deliberately not the n/(n+k) shrinkage used for the per-90
  // features (gh#55/gh#58): that curve never quite reaches 1.0, which would
  // permanently weaken a tier that is load-bearing once the window is full.
  const ROLLMINS_RAMP_GWS = 5;
  function shrinkRollminsMult(mult, n) {
    if (!(n > 0)) return 1;
    const w = Math.min(1, n / ROLLMINS_RAMP_GWS);
    return 1 - (1 - mult) * w;
  }

  // Convert rolling team-strength into a single fixtureMod centred on 1.0.
  // Method: expected goals for our team in this fixture, normalised against league avg,
  // then dampened so a "good fixture" gives ~1.25, a "tough fixture" gives ~0.75.
  function teamStrengthMod(ts, teamId, oppId, isHome) {
    if (!ts) return null;
    const us = ts[teamId], opp = ts[oppId];
    if (!us || !opp) return null;
    const leagueAvg = ts._leagueAvg || 1.4;
    const ourAtk = isHome ? us.homeAttack  : us.awayAttack;
    const oppDef = isHome ? opp.awayDefence : opp.homeDefence;
    if (ourAtk == null || oppDef == null) return null;
    // Expected goals for us: attack * defence / leagueAvg (Dixon-Coles style).
    const xg  = (ourAtk * oppDef) / leagueAvg;
    // Map 0 → 0.5, leagueAvg → 1.0, 2*leagueAvg → 1.5, with home advantage built in.
    const raw = 0.5 + xg / (2 * leagueAvg);
    const venueBoost = isHome ? 1.04 : 0.97;
    return Math.max(0.65, Math.min(1.40, raw * venueBoost));
  }

  // Poisson upper-tail probability: P(X >= k) for a count with mean lambda.
  // Used for the DefCon EV term, a player averaging `lambda` defensive actions
  // per 90 clears the +2 threshold `k` with this probability. Summed
  // iteratively (term_i = term_{i-1} * lambda/i) so it stays stable for the
  // small k we use.
  function poissonAtLeast(k, lambda) {
    if (!(lambda > 0) || k <= 0) return 0;
    let term = Math.exp(-lambda);   // P(X = 0)
    let cum  = term;
    for (let i = 1; i < k; i++) {
      term *= lambda / i;
      cum  += term;
    }
    return Math.max(0, 1 - cum);
  }

  // gh#42: per-match DefCon points threshold (DEF >= 10 actions, MID/FWD >= 12).
  // Must stay aligned with defcon-hit-rates.js.
  function defconThreshold(pos) { return pos === 2 ? 10 : 12; }

  // Season-to-date hit-rate for a player code, computed from the live
  // archive's per-match rows. Returns { rate, n } so callers can apply
  // sample-size shrinkage. null when there is no eligible (60+min) sample.
  function liveDefconRate(code, pos, defconLive) {
    if (!defconLive || !defconLive.by_code) return null;
    const rows = defconLive.by_code[code];
    if (!rows || !rows.length) return null;
    const threshold = defconThreshold(pos);
    let hits = 0, elig = 0;
    for (const r of rows) {
      if ((r.minutes || 0) < 60) continue;
      if (r.defensive_contribution == null) continue;
      elig++;
      if (r.defensive_contribution >= threshold) hits++;
    }
    return elig ? { rate: hits / elig, n: elig } : null;
  }

  // gh#58: empirical-Bayes shrinkage of a raw per-player hit-rate toward the
  // position-level prior. Pulls low-N rates toward the league mean; high-N
  // rates stay near raw. Falls back to the raw rate if priors aren't loaded.
  function shrinkDefconRate(rate, n, pos) {
    if (rate == null || n == null) return rate;
    if (!defconShrinkage || !defconShrinkage.positions) return rate;
    const prior = defconShrinkage.positions[pos];
    if (!prior || prior.n_eff == null || prior.mu == null) return rate;
    const successes = rate * n;
    return (successes + prior.mu * prior.n_eff) / (n + prior.n_eff);
  }

  // gh#42: empirical DefCon hit-rate for a player. Lookup order
  // player-code -> position average -> null (caller then uses the Poisson
  // cold-start fallback). Applies the GW8-14 carryover->live cutover blend:
  // GW<=7 pure carryover, GW8-14 linear blend, GW>=15 pure live when available.
  function defconHitRate(player, ctx) {
    if (!defconCarryover || !defconCarryover.by_code) return null;
    const pos  = player.element_type;
    const code = player.code;
    const carryRow = code != null ? defconCarryover.by_code[code] : null;
    // Per-player carryover gets EB shrinkage (gh#58). Position-average fallback
    // is already a population prior so leave it unshrunk.
    let carry;
    if (carryRow && carryRow.rate != null && carryRow.n != null) {
      carry = shrinkDefconRate(carryRow.rate, carryRow.n, pos);
    } else {
      carry = defconCarryover.position_avg ? defconCarryover.position_avg[pos] : null;
    }
    if (carry == null) return null;

    // liveDefconRate now returns { rate, n }; apply the same EB shrinkage
    // before blending so a 1-match player's live rate doesn't dominate.
    const liveRow = code != null ? liveDefconRate(code, pos, ctx.defconLive) : null;
    const live = liveRow ? shrinkDefconRate(liveRow.rate, liveRow.n, pos) : null;
    const gw   = ctx.currentGw;
    if (gw == null || gw <= 7 || live == null) return carry;
    if (gw >= 15) return live;
    const w = (gw - 7) / 7;          // GW8 -> 1/7, ..., GW14 -> 1.0
    return (1 - w) * carry + w * live;
  }

  // Play-availability for a player: a 0..1 multiplier blending the FPL/news
  // chance-of-playing (worse-case wins when both have signal), a rolling-minutes
  // rotation multiplier, and an optional congested-GW dampener. A hard lineup
  // override forces 0 or 1. Extracted (gh#44) so the squad simulator can read it
  // as a blank-probability signal (q = 1 - avail) for auto-sub robustness.
  function availability(player, ctx) {
    ctx = ctx || {};
    const V = applyVariant(ctx.variant);
    // Lineup override (post-confirmation): hard 0 or 1 from the user's paste-in.
    // Skips both chance_of_playing and rolling-minutes, since lineup IS truth.
    const override = ctx.lineupOverride && ctx.lineupOverride[player.id];
    if (override === 0)   return 0;
    if (override === 100) return 1.0;
    // News override (from news-scrape.js): if BBC Sport or FPL flagged this player
    // as doubtful, use that chance %. Worse-case wins when both have signal.
    const news    = ctx.newsOverride && ctx.newsOverride[player.id];
    const newsCh  = news && news.chance != null ? news.chance : null;
    const fplCh   = player.chance_of_playing_next_round;
    const effective = (newsCh != null && fplCh != null) ? Math.min(newsCh, fplCh)
                     : (newsCh != null ? newsCh : fplCh);
    let avail = effective == null ? 1.0 : effective / 100;
    if (avail === 0) return 0;
    // Rotation signal: rolling avg of last 5 GW minutes. Multiplies into avail.
    const mh = ctx.minutesHistory && ctx.minutesHistory[player.id];
    const roll = rollingMinutes(mh, ctx.currentGw);
    if (roll != null) {
      const rollMins = roll.avg;
      let mult = 1;
      if (rollMins < V.rollmins_very_low_thresh) mult = V.rollmins_very_low_mult;
      else if (rollMins < V.rollmins_low_thresh) mult = V.rollmins_low_mult;
      else if (rollMins < V.rollmins_mid_thresh) mult = V.rollmins_mid_mult;
      // gh#69: temper the tier by how many GWs actually back it.
      avail *= shrinkRollminsMult(mult, roll.n);
    }
    // Fixture-density rotation dampener (21-step #6). If this GW is flagged as
    // congested (deadline gap <= 4 days), shave availability to reflect PL
    // rotation reality during festive periods. Skipped for backtest unless
    // ctx.congestedGws is provided.
    if (ctx.congestedGws && ctx.currentGw != null && ctx.congestedGws[ctx.currentGw]) {
      avail *= ctx.congestedGws[ctx.currentGw];
    }
    return avail;
  }

  function predictPlayer(player, fixList, ctx) {
    ctx = ctx || {};
    const V = applyVariant(ctx.variant);
    if (!fixList || !fixList.length) return 0;

    const avail = availability(player, ctx);
    if (avail === 0) return 0;

    // Recency: ep_next, form, seasonAvg are three correlated views of "recent points".
    // Compress into one term so the rest of the weight budget can do differentiated work.
    //
    // New-signing / low-data carve-out: when a player has very little FPL minutes
    // (e.g., a high-profile summer signing 2-3 GWs into the season, or a January
    // signing in their first appearances), the form/seasonAvg signals are noise.
    // What we DO have is ep_next (FPL editorial's expert call) and the player's
    // price (FPL prices expensive players whom managers value highly). Lean on
    // ep_next when minutes < 270 (~3 full games of data); add a small price-tier
    // bump for expensive new arrivals.
    const ep        = parseFloat(player.ep_next) || 0;
    const form      = parseFloat(player.form)    || 0;
    const seasonAvg = player.total_points / Math.max(1, player.minutes / 90);
    const lowData   = (player.minutes != null && player.minutes < 270);
    const recency   = lowData
      ? ep * 0.85 + form * 0.10 + seasonAvg * 0.05
      : ep * V.recency_ep + form * V.recency_form + seasonAvg * V.recency_season;

    const mins90       = Math.max(1, player.minutes / 90);
    const apps         = Math.max(1, player.minutes / 65);
    const pos          = player.element_type;
    // gh#55: shrink per-90 / per-game features toward per-position mean using
    // EB with k=5 phantom games. n=apps for csRate (events-per-game); n=mins90
    // for the per-90 stats. xgi90 / xgc90 come from FPL pre-divided; their
    // implied event count uses mins90 as the sample-size proxy.
    const xgi90Raw = parseFloat(player.expected_goal_involvements_per_90) || 0;
    const xgc90Raw = parseFloat(player.expected_goals_conceded_per_90)    || 0;
    const xgi90    = shrinkPer90(xgi90Raw, mins90, pos, 'xgi90');
    const xgc90    = shrinkPer90(xgc90Raw, mins90, pos, 'xgc90');
    const csRateRaw       = (player.clean_sheets  || 0) / apps;
    const goalsPer90Raw   = (player.goals_scored  || 0) / mins90;
    const assistsPer90Raw = (player.assists       || 0) / mins90;
    const savesPer90Raw   = (player.saves         || 0) / mins90;
    const csRate       = shrinkPer90(csRateRaw,       apps,   pos, 'csRate');
    const goalsPer90   = shrinkPer90(goalsPer90Raw,   mins90, pos, 'goalsPer90');
    const assistsPer90 = shrinkPer90(assistsPer90Raw, mins90, pos, 'assistsPer90');
    const savesPer90   = shrinkPer90(savesPer90Raw,   mins90, pos, 'savesPer90');

    // Set-piece bonus. Replaces the old flat +0.4 for penalties_order===1.
    // Pen EV (rank 1): ~0.4 pens/match * 0.75 conv * 4-5pts ~= 1.0-1.5pts.
    // GKPs skipped: they don't take open-play set pieces.
    let setPieceBonus = 0;
    if (player.element_type !== 1) {
      if (player.penalties_order === 1)      setPieceBonus += V.pen1;
      else if (player.penalties_order === 2) setPieceBonus += V.pen2;
      if (player.direct_freekicks_order === 1)               setPieceBonus += V.fk1;
      if (player.corners_and_indirect_freekicks_order === 1) setPieceBonus += V.ck1;
    }

    // Fixture modifier cascade: bookmaker odds → team-strength rolling goals → FDR fallback.
    const fix    = fixList[0];
    const isHome = fix.team_h === player.team;
    const oppId  = isHome ? fix.team_a : fix.team_h;
    const fdr    = isHome ? fix.team_h_difficulty : fix.team_a_difficulty;
    const odds   = ctx.oddsMap && ctx.oddsMap[player.team];
    let fixtureMod;
    if (odds && odds.winProb != null) {
      fixtureMod = 0.75 + odds.winProb * 0.5;
    } else {
      const tsMod = teamStrengthMod(ctx.teamStrength, player.team, oppId, isHome);
      fixtureMod = tsMod != null
        ? tsMod
        : (fdr === 2 ? 1.12 : fdr === 3 ? 1.00 : fdr === 4 ? 0.87 : 0.72)
          * (isHome ? 1.04 : 0.97);
    }

    // CS bonus scaling for GKP/DEF. O/U 2.5 + bookmaker BTTS (both teams to score)
    // are blended: low O/U + low BTTS means low-scoring game with one likely CS.
    // pBTTSno is "no BTTS" probability (i.e. at least one team CS).
    const pOver25 = (odds && odds.pOver25 != null) ? odds.pOver25 : 0.5;
    const pBTTSno = (odds && odds.pBTTSno != null) ? odds.pBTTSno : null;
    let csFactor;
    if (pBTTSno != null) {
      // Blend: half from O/U (game total), half from BTTS-no (CS likelihood directly).
      csFactor = Math.max(0.3, (1.5 - pOver25) * 0.5 + (0.5 + pBTTSno) * 0.5);
    } else {
      csFactor = Math.max(0.3, 1.5 - pOver25);
    }

    let base;
    if (player.element_type === 1) {
      // GKP: blend two approaches.
      //
      // (A) FPL-rules EV: 2 (app) + 4*csProb + saves/3 - goalsConcededPenalty.
      //     Mirrors actual FPL scoring magnitudes; gives well-calibrated absolute
      //     predictions (slope 0.42 in cross-season calibration).
      //
      // (B) Recency-weighted: dominated by form/ep_next/seasonAvg. Bad calibration
      //     (slope 0.10) but high variance, which preserves rank discrimination
      //     that the initial-squad picker depends on.
      //
      // 50/50 blend captures both: FPL-EV magnitude + recency-based ranking. The
      // blend lands at slope ~0.25 and keeps squad-sim performance.
      const csProb         = csRate * 0.5 + Math.max(0, 1 - xgc90 / 2.0) * 0.5;
      const csPts          = 4 * csProb * csFactor;
      const savesPts       = savesPer90 / 3;
      const gConcededPen   = Math.min(1.0, xgc90 / 2.0);
      const fplEv          = 2 + csPts + savesPts - gConcededPen;

      const csBonus        = csProb * 2.0 * csFactor;
      const savesBonus     = savesPer90 / 3;
      const recencyBased   = recency * V.gkp_recency + csBonus * V.gkp_cs + savesBonus * V.gkp_saves;

      base = Math.min(5.5, fplEv * 0.5 + recencyBased * 0.5);
    } else if (player.element_type === 2) {
      // DEF: original weighted-blend formula. Audit (issue #4) found that
      // CS and attacking contributions are 10x smaller than FPL scoring
      // magnitudes, but multiple FPL-EV-blend attempts (50/50, 30/70 with
      // varying caps) didn't reliably improve squad-sim across seasons.
      // The recency-based ranking is good even if the magnitudes are off,
      // and squad-sim cares about rank more than magnitude. Keeping the
      // original; issue #4 documents the audit results.
      const csProb  = csRate * 0.5 + Math.max(0, 1.0 - xgc90 / 2.0) * 0.5;
      const csBonus = csProb * 2.0 * csFactor;
      const attRet  = (xgi90 * 5) * V.att_xgi_blend + (goalsPer90 * 6 + assistsPer90 * 3) * (1 - V.att_xgi_blend);
      base = recency * V.def_recency + csBonus * V.def_cs + attRet * V.def_att;
    } else if (player.element_type === 3) {
      // MID. Audit (issue #4) found attacking contribution undervalued ~2-3x
      // and CS bonus (1pt) not modelled. Attempted: add CS bonus, regressed
      // squad-sim by 200+ pts (tilted picks toward defensive mids who
      // under-deliver). MID calibration slope (0.85) is already best of all
      // positions; the recency-based ranking is good. No fix shipped.
      const attRet = (xgi90 * 5) * V.att_xgi_blend + (goalsPer90 * 5 + assistsPer90 * 3) * (1 - V.att_xgi_blend);
      base = recency * V.mid_recency + attRet * V.mid_att;
    } else {
      // FWD. Audit (issue #4) found low-recency goalscorers under-predicted.
      // Goal-lift attempt regressed squad-sim 2024-25 by 197 and HURT calibration
      // (slope dropped from 0.564 to 0.517). The goal-lift undid issue #2's
      // compression. No fix shipped; issue #2's compression already handles
      // the over-prediction at the top end where it mattered most.
      const attRet = (xgi90 * 4) * V.att_xgi_blend + (goalsPer90 * 4 + assistsPer90 * 3) * (1 - V.att_xgi_blend);
      base = recency * V.fwd_recency + attRet * V.fwd_att;
    }

    base += setPieceBonus;

    // New-signing premium nudge. Expensive low-data players get a small boost
    // because FPL pricing already reflects expert expectation. Mbappé-grade
    // signings start at ~£11-13m; mid-tier signings at £7-9m. Without this,
    // the model would treat any new arrival as "no data = average."
    if (lowData && player.now_cost != null) {
      if      (player.now_cost >= 110) base += 0.8;   // £11.0m+
      else if (player.now_cost >= 90)  base += 0.4;   // £9.0m+
      else if (player.now_cost >= 70)  base += 0.15;  // £7.0m+
    }

    // Adaptation discount. Players with low season minutes are usually in a
    // transitional state: new to the league, new club within the league, or
    // returning from a long injury / sub role. The "Wirtz / Sesko / Eze / Cunha"
    // pattern: ep_next looks bullish from previous-club form, but actual output
    // is muted while they settle in. A 15% haircut on lowData players costs us
    // some Haaland-style instant-impact upside but stops costly early-season
    // burns. Real FPL community consensus: discount new arrivals for ~6 GWs.
    if (lowData) base *= 0.85;

    // Manager change dampener (issue #8). When a player's team has had a recent
    // mid-season manager change, FPL community consensus is players underperform
    // for ~6 GWs while the new manager beds in. Caller passes ctx.managerChanges
    // (list of {season, team_short, first_gw, dampen_gws}) plus ctx.currentSeason
    // and ctx.currentGw. We look up team_short via ctx.teamShortById[player.team].
    if (ctx.managerChanges && ctx.currentGw != null && ctx.teamShortById) {
      const teamShort = ctx.teamShortById[player.team];
      if (teamShort) {
        const events = ctx.managerChanges.filter(e =>
          e.team_short === teamShort &&
          (!ctx.currentSeason || e.season === ctx.currentSeason) &&
          ctx.currentGw >= e.first_gw &&
          ctx.currentGw < e.first_gw + (e.dampen_gws || 6)
        );
        if (events.length) base *= 0.90;
      }
    }

    // BPS-magnet bonus, position-aware (21-step #1). Captures defenders/midfielders
    // who rack up tackles/clearances/recoveries: bonus pts the goal/assist-driven
    // attRet doesn't see. Per-position multipliers:
    //   DEF (1.0): full credit. BPS-magnet defenders are the classic underrated bucket.
    //   MID (0.7): partial credit. Mids share BPS sources with attRet.
    //   FWD (0.4): minimal. Forwards' BPS is overwhelmingly from goals, already in attRet.
    // csFactor lifts the bonus marginally when the opponent is weak: BPS scrapes
    // are easier when a team controls possession and the ball lives in the opp half.
    if (player.element_type !== 1) {
      const bpsPer90 = (player.bps || 0) / mins90;
      const posMult  = player.element_type === 2 ? 1.0
                     : player.element_type === 3 ? 0.7
                     : 0.4;
      const matchMult = 0.85 + Math.min(1.15, csFactor) * 0.15;
      base += Math.min(0.6, bpsPer90 * V.bps_scale * posMult * matchMult);
    }

    // DefCon EV (2024-25+ scoring): +2 pts when a player's defensive-action
    // count clears the match threshold (DEF: 10 CBIT, MID/FWD: 12 CBIRT).
    // Live-only, gated on the season-total `defensive_contribution`, which FPL
    // bootstrap carries but historical Vaastav backtest data does not, so this
    // term is silently zero in every backtest.
    // Primary path (gh#42): an empirical per-match hit-rate measured from real
    // DefCon counts, with a 0.9 confidence haircut. Cold-start fallback (when
    // defcon-hit-rates.json is absent): the un-backtestable Poisson bridge,
    // which models the season count as a Poisson mean and takes P(>=threshold).
    if (player.element_type >= 2 && player.defensive_contribution != null) {
      const empRate = defconHitRate(player, ctx);
      if (empRate != null) {
        base += empRate * 2 * 0.9;
      } else if ((player.minutes || 0) >= 270) {
        const per90     = player.defensive_contribution / (player.minutes / 90);
        const threshold = player.element_type === 2 ? 10 : 12;
        const hitProb   = poissonAtLeast(threshold, per90);
        base += hitProb * 2 * (V.defcon_confidence ?? 0.7);
      }
    }

    // Dream Team bonus: small additive lift for players consistently in the GW
    // Team of the Week. Sourced from player.dream_team_count (current season
    // feed) or player.dreamteam_count (Vaastav per-season). Either field key
    // is accepted because they come from different pipelines.
    const dtc = player.dream_team_count != null ? player.dream_team_count
              : (player.dreamteam_count != null ? player.dreamteam_count : null);
    if (dtc != null && dtc > 0) {
      base += Math.min(V.dream_team_cap, dtc * V.dream_team_scale);
    }

    // DGW: ep_next already reflects two fixtures. Boost everything else by 1.6x.
    if (fixList.length > 1) {
      const recencyShare = (player.element_type <= 2 ? V.gkp_recency : V.mid_recency);
      const recencyPart  = recency * recencyShare;
      base = recencyPart + (base - recencyPart) * 1.6;
    }

    // gh#57: compute pre-avail prediction, lift via priors, then scale by avail
    // so the intercept floor scales with availability.
    const rawPredNoAvail = base * fixtureMod;
    const tier      = tierFor(rawPredNoAvail);
    const calFactor = lookupCalibration(ctx.calibration, player.element_type, tier);
    let final = rawPredNoAvail * calFactor;

    // Stage-2 OLS adjustment from fit.js (optional). Trained on walk-forward history,
    // it's a global slope/intercept correction with per-position offsets sitting on top
    // of tier calibration. Identity transform when priors aren't supplied.
    if (ctx.priors) {
      const slope     = ctx.priors.slope     != null ? ctx.priors.slope     : 1.0;
      const intercept = ctx.priors.intercept != null ? ctx.priors.intercept : 0.0;
      const posOff    = ctx.priors.posOffset && ctx.priors.posOffset[player.element_type];
      final = slope * final + intercept + (posOff || 0);
    }
    // Prediction shrinkage toward the pool mean (2026-10-14). Live GW1-6
    // regression of actual on predicted gives a slope near 0.7, i.e. the
    // spread is too wide: high predictions overshoot, low ones undershoot.
    // Leave-one-gameweek-out on live data improved MAE 2.085 -> 2.044 at
    // k ~= 0.7. It matters most for anything that maximises over predictions
    // (captaincy, and Free Hit / Wildcard EV, which compare a best-possible
    // XI against our own).
    //
    // SWEPT AND REJECTED 2026-10-14 (pred-shrink-sweep-results.json, 3
    // seasons): every shrinkage level LOSES points, monotonically
    // (0.9 -50, 0.8 -82, 0.7 -68, 0.6 -142 NET vs 1.0), far outside the ~30
    // noise floor. Lower MAE but worse teams: squad selection is a ranking
    // problem, and compressing the spread blurs exactly the gaps the planner
    // and the captaincy pick rely on. MAE rewards hedging; points do not.
    // Kept flag-gated and INERT. Do not enable without new evidence.
    if (ctx.predShrink != null && ctx.predShrink !== 1) {
      const m = ctx.predShrinkMean != null ? ctx.predShrinkMean : 2.6;
      final = m + ctx.predShrink * (final - m);
    }

    final *= avail;

    // FWD over-prediction compression (issue #2). Calibration showed FWDs in
    // the 6-7 predicted range deliver only 4.5-5.0 actual on average. Soft
    // compression starting at 6.0 with a gentle 0.7 factor: aggressive enough
    // to fix the worst bias buckets (6.0-7.0) without flattening elite FWDs.
    //
    // Predicted 6.0 stays 6.0. Predicted 7.0 → 6.7. Predicted 8.0 → 7.4.
    // Threshold 6.0 because below that the calibration is reasonable; above
    // that the model over-predicts systematically.
    if (player.element_type === 4 && final > 6.0) {
      final = 6.0 + (final - 6.0) * 0.7;
    }

    // H2H matchup adjustment (21-step #7). Some players have measurable edges
    // (or deficits) vs specific opponents. We multiply by 1 + clamp(delta, ±0.1)
    // where delta = vs_opponent_ppg / overall_ppg - 1. Capped to keep one fluke
    // game from dominating. Requires h2h cache (h2h-build.js) and current opp id.
    if (ctx.h2h && oppId != null) {
      const rec = ctx.h2h[normName(player.web_name)];
      if (rec && rec.overall_ppg > 0.5) {
        const vs = rec.vs_opponent && rec.vs_opponent[oppId];
        if (vs && vs.n >= 3) {
          const rawDelta = vs.ppg / rec.overall_ppg - 1;
          const delta    = Math.max(-0.10, Math.min(0.10, rawDelta));
          final *= (1 + delta);
        }
      }
    }

    let out = Math.max(0, parseFloat(final.toFixed(2)));

    // ML residual correction (gh#5 Phase 4). Additive, gated, bounded: adds
    // the model's learned estimate of the formula's error. The model trains
    // on (actual - formula_pred), so this is applied to the final formula
    // output. On any failure it falls back to the bare formula.
    if (ctx.residual === true && residualModel) {
      try {
        const delta = residualModel.predictResidual(
          residualModel.extractFeatures(player, fixList, isHome));
        // Half strength (2026-11-02): leave-one-season-out tournament over 4
        // seasons at the locked cell, NET vs no residual: x0.25 +77, x0.5 +111,
        // x0.75 +51, x1.0 -78. The OOF slope of actual-formula on delta is
        // 0.76 overall and 0.41 for large positives, so the full delta overshoots.
        const scale = process.env.RESIDUAL_SCALE != null ? parseFloat(process.env.RESIDUAL_SCALE) : 0.5;
        out = Math.max(0, parseFloat((out + delta * scale).toFixed(2)));
      } catch (e) { /* model unavailable - keep the formula output */ }
    }
    return out;
  }

  // Issue #10: empirical haul-probability lookup. Given a mean-predicted score
  // and position, return P(actual >= threshold) using the precomputed table
  // from haul-probs.js (haul-probs.json). Falls back to a Poisson approximation
  // when no bucket data exists (high predictions for fringe players, etc).
  //
  // Why empirical: the Brier baseline showed Poisson is ~2x too pessimistic
  // for FPL's fat-tailed scoring. A predicted 5.4 MID hauls 22% of the time,
  // not the 11% Poisson predicts. The lookup captures real fat-tail behavior.
  function poissonP(threshold, lambda) {
    if (lambda <= 0) return 0;
    let p = 0, term = Math.exp(-lambda);
    for (let i = 0; i < threshold; i++) { p += term; term *= lambda / (i + 1); }
    return Math.max(0, Math.min(1, 1 - p));
  }

  function haulProb(predMean, posId, threshold, haulTable) {
    if (predMean <= 0) return 0;
    // Try the empirical lookup first.
    if (haulTable && haulTable.table && haulTable.table[posId]) {
      const binSize = haulTable.binSize || 1.0;
      const bucket  = Math.floor(predMean / binSize);
      const rec     = haulTable.table[posId].byBucket[bucket];
      if (rec) {
        const key = 'pHaul' + threshold;
        if (rec[key] != null) return rec[key];
      }
      // Bucket missing (sparse area): linear-interp between adjacent buckets if both exist.
      const lower = haulTable.table[posId].byBucket[bucket - 1];
      const upper = haulTable.table[posId].byBucket[bucket + 1];
      if (lower && upper) {
        const key = 'pHaul' + threshold;
        if (lower[key] != null && upper[key] != null) {
          return (lower[key] + upper[key]) / 2;
        }
      }
    }
    // Final fallback: Poisson approximation. Known to be conservative but
    // monotonic in mean, which preserves rank ordering even if magnitudes lie.
    return poissonP(threshold, predMean);
  }

  const api = { predictPlayer, availability, tierFor, lookupCalibration, haulProb, poissonP, rollingMinutes };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else if (typeof window !== 'undefined') {
    window.fplPredict = api;
  }
})();
