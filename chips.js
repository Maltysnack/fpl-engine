// Live chip decision module. At each gameweek, decide whether to play Triple
// Captain or Bench Boost RIGHT NOW based on signals available at decision
// time, never with hindsight.
//
// Rules we enforce (real FPL):
//   1. Each chip used at most once per season.
//   2. Cannot play both chips in the same GW.
//   3. Must use chip on or before the final eligible GW.
//
// Decision philosophy:
//   - Predicted points already encode double gameweeks: a DGW player's
//     prediction sums both fixtures, so a DGW captain / bench / XI is already
//     worth ~2x. There is NO separate "is this a DGW?" gate — option-pricing
//     over the raw predicted score handles timing and naturally favours DGWs.
//   - TC: fire on a captain-score outlier. BB: fire on a bench-score outlier.
//   - FH: dual trigger — a bad week (own XI projects very low) fires now; a
//     big swing to the best-possible XI fires via option-pricing.
//   - WC: rescue chip — see wcShouldFire.
//   - Forecast a limited lookahead window, the way a real manager sees only
//     the next few GWs. Hard deadline: force-fire TC/BB at the half deadline.

(function () {
  // gh#38: 2-per-season chips (2024-25 rule). Each chip type fires once in
  // H1 (GW1-19) and once in H2 (GW20-38). Windows defined per chip; defaults
  // apply symmetrically to both halves unless an override targets one half.
  const HALF1_END = 19;   // GW1-19 = H1, GW20-38 = H2

  function halfOf(gw) { return gw <= HALF1_END ? 1 : 2; }

  // Default tuning knobs. Pass overrides via opts. Knobs ending in `H1`/`H2`
  // target a specific half; bare knobs apply to both halves symmetrically.
  const DEFAULTS = {
    lookaheadGws:    5,     // GWs of visibility into upcoming fixtures
    // TC: secretary-style observation phase then option-pricing fire.
    minTCEarlyH1:    6,     // earliest TC1 fire (observation phase ends here)
    minTCEarlyH2:   25,     // earliest TC2 fire (4 GWs of H2 observation = GW20-24)
    tcDeadlineH1:   18,     // force-fire TC1 at this GW if unused (1 GW buffer before half ends)
    tcDeadlineH2:   37,     // force-fire TC2 at this GW if unused
    // BB: same shape as TC.
    minBBEarlyH1:    6,
    minBBEarlyH2:   25,
    bbDeadlineH1:   18,
    bbDeadlineH2:   37,
    // FH: earlier eligibility because blank GWs hit any time.
    minFHEarlyH1:    3,
    minFHEarlyH2:   22,
    fhDeadlineH1:   19,
    fhDeadlineH2:   37,
    // Numeric floors (apply to both halves). gh#36: FH dual trigger —
    // fhRescueXIPred = bad-outlier floor (fire now), fhMinUplift = good-outlier
    // floor (fire via option-pricing). Fitted by fh-sweep.js post-gh#41 (against
    // the corrected cap-aware FH metric): fhMinUplift 4 and 8 tie for best and
    // it degrades sharply above 12 (16→20 costs ~185 NET) — 8 chosen as the
    // firmer of the tied-optimal pair. fhRescueXIPred + optionK near-inert.
    fhMinUplift:       8,
    fhRescueXIPred:    40,
    bbBenchThreshold:  16,
    optionK:         1.0,
    // WC (gh#35): rescue chip. Fires purely on a low projected XI — no
    // option-pricing, no deadline force-fire. An unused WC is acceptable;
    // the chip exists to escape a bad squad, not to chase EV.
    minWCEarlyH1:    4,
    minWCEarlyH2:   23,
    wcDeadlineH1:   19,
    wcDeadlineH2:   37,
    wcTriggerXIPred: 40    // fire WC if projected XI over the lookahead falls below this
  };

  // Initial chipState (all 8 unused). Use this rather than ad-hoc object
  // literals to avoid the "forgot to seed one of the 8" bug class.
  function initChipState() {
    return {
      tc1Used: false, tc2Used: false,
      bb1Used: false, bb2Used: false,
      fh1Used: false, fh2Used: false,
      wc1Used: false, wc2Used: false
    };
  }

  // Option-pricing chip-fire rule (issue #14). Treats each unused chip as an
  // option with future opportunities. Fires now if THIS GW's value is at least
  // k standard deviations above the mean of future visible values.
  //
  //   k = 0   : fire whenever this GW is at-or-above future mean (aggressive)
  //   k = 0.5 : fire when this GW is meaningfully above average (default)
  //   k = 1.0 : fire only when this GW is in the top ~16% of future outcomes
  //
  // Edge cases:
  //   - No future visible: fire (last chance in the window).
  //   - Exactly one future visible: fire if current >= future (only sensible call).
  //   - SD effectively zero (all futures equal): fire if current >= mean.
  function valueChipFire(currentVal, futureVals, opts) {
    opts = opts || {};
    const k = opts.k != null ? opts.k : DEFAULTS.optionK;
    if (!futureVals || futureVals.length === 0) return true;
    if (futureVals.length === 1) return currentVal >= futureVals[0];
    const n = futureVals.length;
    const mean = futureVals.reduce((s, v) => s + v, 0) / n;
    const variance = futureVals.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
    const sd = Math.sqrt(variance);
    if (sd < 0.01) return currentVal >= mean;
    return currentVal > mean + k * sd;
  }

  // WC fire rule (gh#35). Wildcard is a rescue chip: fire it the first time
  // the projected XI over the lookahead window drops below the trigger, within
  // this half's window. No option-pricing and no deadline force-fire — leaving
  // a WC unused is fine, it only exists to escape a structurally bad squad.
  // xiPred should be the lookahead-window mean XI prediction, not a single GW
  // (a one-week dip is usually a blank GW, which is Free Hit's job).
  function wcShouldFire(gw, xiPred, chipState, opts) {
    const O = Object.assign({}, DEFAULTS, opts || {});
    const half = halfOf(gw);
    if (chipState && chipState[`wc${half}Used`]) return false;
    if (gw < O[`minWCEarlyH${half}`] || gw > O[`wcDeadlineH${half}`]) return false;
    if (xiPred == null) return false;   // no projection -> never fire on a 0
    return xiPred < O.wcTriggerXIPred;
  }

  // Given the current GW state + per-GW projections, decide whether to fire
  // TC, BB, FH, or WC this GW. Returns { useTC, useBB, useFH, useWC, halfInstance }.
  // gh#38: respects 2-per-season rule (TC1/TC2, BB1/BB2, FH1/FH2, WC1/WC2).
  //
  // Inputs:
  //   currentGw       - the GW about to be played
  //   gwProjections   - array of { gw, captainPred, squad15Pred, dgwCount, xiBlanks }
  //                     covering currentGw .. currentGw + lookaheadGws.
  //   chipState       - { tc1Used, tc2Used, bb1Used, bb2Used, fh1Used, fh2Used, wc1Used, wc2Used }
  //   opts            - merge with DEFAULTS
  function decideChipsThisGw(currentGw, gwProjections, chipState, opts) {
    const O = Object.assign({}, DEFAULTS, opts || {});
    const half = halfOf(currentGw);
    let useTC = false, useBB = false, useFH = false, useWC = false;
    const halfInstance = half;
    if (!gwProjections || !gwProjections.length) return { useTC, useBB, useFH, useWC, halfInstance };
    const thisGw = gwProjections.find(g => g.gw === currentGw);
    if (!thisGw) return { useTC, useBB, useFH, useWC, halfInstance };

    // Per-half configuration. Each chip instance has its own observation
    // floor + deadline pulled from the H1/H2 DEFAULTS.
    const cfg = {
      tc:  { used: chipState[`tc${half}Used`], early: O[`minTCEarlyH${half}`], deadline: O[`tcDeadlineH${half}`] },
      bb:  { used: chipState[`bb${half}Used`], early: O[`minBBEarlyH${half}`], deadline: O[`bbDeadlineH${half}`] },
      fh:  { used: chipState[`fh${half}Used`], early: O[`minFHEarlyH${half}`], deadline: O[`fhDeadlineH${half}`] }
      // WC has no early/deadline cfg row — wcShouldFire reads its window directly.
    };

    // TC decision (gh#36). The DGW is already baked into captainPred /
    // captainCeiling (a DGW captain's prediction sums both fixtures), so there
    // is no separate DGW gate — option-pricing over the raw score handles
    // timing and naturally favours DGWs (~2x score). Force-fire at the half
    // deadline if still unused.
    const scoreTC = g => (g.captainCeiling != null && g.captainCeiling > 0)
                       ? g.captainCeiling : (g.captainPred || 0);
    if (!cfg.tc.used && currentGw >= cfg.tc.early && currentGw <= cfg.tc.deadline) {
      if (currentGw === cfg.tc.deadline) {
        useTC = true;
      } else {
        const futureVals = gwProjections
          .filter(g => g.gw > currentGw && g.gw <= cfg.tc.deadline)
          .map(scoreTC);
        if (valueChipFire(scoreTC(thisGw), futureVals, { k: O.optionK })) useTC = true;
      }
    }

    // BB decision (gh#36). benchPred already sums DGW fixtures, so no DGW gate.
    // Option-pricing over benchPred handles timing; bbBenchThreshold keeps BB
    // from firing on a thin bench. Force-fire at the half deadline.
    if (!cfg.bb.used && currentGw >= cfg.bb.early && currentGw <= cfg.bb.deadline) {
      if (currentGw === cfg.bb.deadline) {
        useBB = true;
      } else if ((thisGw.benchPred || 0) >= O.bbBenchThreshold) {
        const futureVals = gwProjections
          .filter(g => g.gw > currentGw && g.gw <= cfg.bb.deadline)
          .map(g => g.benchPred || 0);
        if (valueChipFire(thisGw.benchPred || 0, futureVals, { k: O.optionK })) useBB = true;
      }
    }

    // FH decision (gh#36). Dual trigger:
    //   bad-outlier  — this GW's own XI projects below fhRescueXIPred (a blank
    //                  or wipeout week). Fire immediately; a blank cannot be
    //                  waited out. Distinct from WC, which triggers on the
    //                  lookahead-window MEAN (a structurally bad squad).
    //   good-outlier — uplift to the best-possible XI >= fhMinUplift (a big
    //                  one-week swing, e.g. a full-DGW XI). Option-pricing
    //                  holds for a bigger swing later in the half.
    // No xiBlanks gate — a low single-GW xiPred is the better "bad week" signal.
    if (!cfg.fh.used && currentGw >= cfg.fh.early && currentGw <= cfg.fh.deadline) {
      const upliftOf = g => Math.max(0, (g.freeHitXIPred || 0) - (g.xiPred || 0));
      if (thisGw.xiPred != null && thisGw.xiPred < O.fhRescueXIPred) {
        useFH = true;
      } else if (upliftOf(thisGw) >= O.fhMinUplift) {
        const futureUplifts = gwProjections
          .filter(g => g.gw > currentGw && g.gw <= cfg.fh.deadline)
          .map(upliftOf);
        if (valueChipFire(upliftOf(thisGw), futureUplifts, { k: O.optionK })) useFH = true;
      }
    }

    // WC decision (gh#35). Rescue chip — pure threshold on the projected XI.
    // thisGw.xiPred should already be the lookahead-window mean.
    if (wcShouldFire(currentGw, thisGw.xiPred, chipState, O)) useWC = true;

    // Conflict resolution: at most ONE chip per GW (real FPL rule).
    // FH > WC > TC/BB precedence.
    if (useFH && (useTC || useBB || useWC)) {
      useTC = false; useBB = false; useWC = false;
    } else if (useWC && (useTC || useBB)) {
      useTC = false; useBB = false;
    } else if (useTC && useBB) {
      const tcGain = thisGw.captainPred;
      const bbGain = thisGw.squad15Pred * (4 / 15);
      if (tcGain >= bbGain) useBB = false; else useTC = false;
    }

    return { useTC, useBB, useFH, useWC, halfInstance };
  }

  // planChips: given the full season's gwData (or just the future GWs from
  // current onward) plus the current chipState, project when each unused chip
  // will most likely fire. Useful for display and for the planner to know which
  // GWs are "chip-locked" so it doesn't plan transfers that fight the chip
  // strategy (e.g., don't burn a transfer to swap out a player you'll triple-
  // captain next GW).
  //
  // Returns { tcGw, bbGw, fhGw, sequence } where sequence describes the per-GW
  // ranking that drove each decision. Pure projection: doesn't mutate state.
  function planChips(gwProjections, chipState, opts) {
    const O = Object.assign({}, DEFAULTS, opts || {});
    const state = Object.assign(initChipState(), chipState || {});
    const plan  = {
      tc1Gw: null, tc2Gw: null, bb1Gw: null, bb2Gw: null,
      fh1Gw: null, fh2Gw: null, wc1Gw: null, wc2Gw: null,
      sequence: []
    };
    if (!gwProjections || !gwProjections.length) return plan;
    const sortedByGw = [...gwProjections].sort((a, b) => a.gw - b.gw);
    for (const g of sortedByGw) {
      const allUsed = state.tc1Used && state.tc2Used
                   && state.bb1Used && state.bb2Used
                   && state.fh1Used && state.fh2Used
                   && state.wc1Used && state.wc2Used;
      if (allUsed) break;
      const window = sortedByGw.filter(x => x.gw >= g.gw && x.gw <= g.gw + O.lookaheadGws);
      const d = decideChipsThisGw(g.gw, window, state, opts);
      const h = d.halfInstance;
      if (d.useTC && !state[`tc${h}Used`]) { plan[`tc${h}Gw`] = g.gw; state[`tc${h}Used`] = true; }
      if (d.useBB && !state[`bb${h}Used`]) { plan[`bb${h}Gw`] = g.gw; state[`bb${h}Used`] = true; }
      if (d.useFH && !state[`fh${h}Used`]) { plan[`fh${h}Gw`] = g.gw; state[`fh${h}Used`] = true; }
      if (d.useWC && !state[`wc${h}Used`]) { plan[`wc${h}Gw`] = g.gw; state[`wc${h}Used`] = true; }
      plan.sequence.push({
        gw: g.gw,
        half: h,
        captainCeiling: g.captainCeiling || 0,
        captainPred:    g.captainPred    || 0,
        benchPred:      g.benchPred      || 0,
        xiBlanks:       g.xiBlanks       || 0,
        dgwCount:       g.dgwCount       || 0,
        firedTC: d.useTC || false, firedBB: d.useBB || false,
        firedFH: d.useFH || false, firedWC: d.useWC || false
      });
    }
    return plan;
  }

  const api = { decideChipsThisGw, planChips, valueChipFire, wcShouldFire, initChipState, halfOf, DEFAULTS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof window !== 'undefined') window.fplChips = api;
})();
