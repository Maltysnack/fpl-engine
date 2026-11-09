// Rolling-horizon transfer planner for the FPL squad simulator.
//
// At each gameweek decision point we don't pick one swap in isolation. We
// search over a 4-GW horizon: for each candidate first move, project the
// squad forward, choose follow-up moves, score the resulting multi-GW XI
// total minus hit costs, and pick the first move that maximises the total.
//
// Algorithm: beam search.
//   - State: { squad, freeTransfers, budget, clubCount, totalScore, firstMove }
//   - At each depth, expand each state with moves: hold, or top-K single swaps.
//   - Prune to beam width B (keep top B states by totalScore).
//   - At horizon, the best state's firstMove is what we execute this GW.
//
// The first move propagates from depth 0 even as deeper moves are explored,
// so we always know "what should I do RIGHT NOW given that I'll keep planning
// optimally next week."
//
// Inputs to plan():
//   squad         - current 15-man squad (array of player objects)
//   gw            - current gameweek (planning starts here)
//   ctx           - prediction context (calibration, priors, etc.)
//   variantName   - which variant to use (default | xg_pure | ensemble | ...)
//   ensembleWeights - per-position blend if variant === 'ensemble'
//   players       - full league pool (filtered to a season)
//   state         - { freeTransfers, budget, clubCount }
//   opts          - { horizon: 4, beamWidth: 4, hitCost: 4, topKMoves: 5,
//                     hitThreshold: 8, predictWithVariant, lookaheadCache }
//
// Returns: { type: 'hold' | 'transfer', out, in, sellPrice, buyPrice } for GW.

(function () {
  // gh#44 Layer B: the 8 legal formations as [DEF, MID, FWD] outfield counts.
  const FORMATION_LIST = [
    [3, 4, 3], [3, 5, 2], [4, 3, 3], [4, 4, 2], [4, 5, 1], [5, 2, 3], [5, 3, 2], [5, 4, 1]
  ];
  const LOCKED_FORMATION = {
    '3-4-3': [3, 4, 3], '3-5-2': [3, 5, 2], '4-3-3': [4, 3, 3], '4-4-2': [4, 4, 2],
    '4-5-1': [4, 5, 1], '5-2-3': [5, 2, 3], '5-3-2': [5, 3, 2], '5-4-1': [5, 4, 1]
  };
  // gh#44 Layer B: formation policy in force for the current plan() call. Set
  // once at plan() entry, so the planner scores transfers/holds against the
  // SAME formation policy the simulator executes with. 'max'/'robust' enumerate
  // all 8 legal shapes per GW; the planner heuristic has no auto-sub or
  // availability signal so 'robust' collapses to 'max' here. A locked name
  // pins that single formation. Default '4-4-2' is byte-identical to the legacy
  // fixed-formation planner.
  let planFormations = [LOCKED_FORMATION['4-4-2']];
  function setFormationPolicy(policy) {
    if (policy === 'max' || policy === 'robust') planFormations = FORMATION_LIST;
    else planFormations = [LOCKED_FORMATION[policy] || LOCKED_FORMATION['4-4-2']];
  }

  // gh#43: convex price tilt. Set once per plan() entry. λ=0 → no tilt.
  let planPriceLambda = 0;
  function setPriceLambda(lam) { planPriceLambda = lam || 0; }

  // 2024-25 rule: free transfers accumulate up to 5 (was 2). The cap leaks
  // into the lookahead beam (hold branch) and squad-sim's per-GW accumulator.
  // Keep both in lockstep. Bumping this alone has limited effect unless the
  // planner can also enumerate 3- and 4-transfer combinations to actually
  // spend banked FTs in one GW (see MAX_SWAP_N below). See issue #34.
  const FT_CAP = 5;
  // Maximum simultaneous swaps the planner will enumerate in a single GW.
  // Matches FT_CAP so a full 5-bank-then-burn is expressible. N=5 is only
  // enumerated when freeTransfers === FT_CAP (zero hits paid), since a
  // hit-paying quint is almost certainly negative-EV and the combinatorics
  // are expensive (C(pool, 5) ≈ 5× C(pool, 4)).
  const MAX_SWAP_N = 5;

  // Pick the best legal XI by predicted, returning its score plus a captain
  // bonus (best XI player doubled). Enumerates planFormations — one shape when
  // locked, all 8 when fluid — and takes the highest-scoring shape. (Used for
  // in-beam scoring; auto-subs and vice-captain happen at execution.)
  // gh#71: weight on bench points. 0 reproduces the historical XI-only scoring
  // exactly (the locked default), so this is inert until deliberately raised.
  // Bench value is contingent, not zero: it buys auto-sub cover when a starter
  // blanks, feeds Bench Boost, and keeps budget in playable assets. Scoring the
  // XI alone makes a dead bench player literally free, which is how a squad ends
  // up holding an unplayable asset nobody is motivated to move.
  let benchWeight = 0;
  function setBenchWeight(w) { benchWeight = w || 0; }

  // gh#79: FPL grants one free transfer EVERY gameweek (capped at FT_CAP),
  // whether or not you transferred. The legacy hold branch modelled the +1
  // but the transfer branches did not, so any transfer left the beam with 0
  // FTs next week and priced the follow-up as a hit. That bias against
  // spending was baked into the ultra lock. `ftAccrual` applies the +1 after
  // a transfer too; off by default until the A/B sweep adopts it.
  let ftAccrual = false;
  function setFtAccrual(v) { ftAccrual = !!v; }
  function nextFree(remaining) { return ftAccrual ? Math.min(FT_CAP, remaining + 1) : remaining; }

  function predictedXIScore(squad, predForGw) {
    const byPos = { 1: [], 2: [], 3: [], 4: [] };
    for (const p of squad) byPos[p.element_type].push(predForGw[p.id] || 0);
    for (const pos of [1, 2, 3, 4]) byPos[pos].sort((a, b) => b - a);
    const gk = byPos[1][0] || 0;
    let bestXI = -Infinity;
    for (const f of planFormations) {
      let t = gk;
      for (let i = 0; i < f[0]; i++) t += byPos[2][i] || 0;
      for (let i = 0; i < f[1]; i++) t += byPos[3][i] || 0;
      for (let i = 0; i < f[2]; i++) t += byPos[4][i] || 0;
      if (t > bestXI) bestXI = t;
    }
    // Captain: highest single predicted across the squad. Every position has
    // >= 1 XI slot in every legal formation, so the global max is always in
    // the chosen XI regardless of shape.
    const bestPred = Math.max(byPos[1][0] || 0, byPos[2][0] || 0, byPos[3][0] || 0, byPos[4][0] || 0);
    let total = bestXI;
    if (bestPred > 0) total += bestPred;
    if (benchWeight) {
      // Bench = whole squad minus the chosen XI. Cheaper than re-deriving which
      // players were selected, and exact.
      let sumAll = 0;
      for (const pos of [1, 2, 3, 4]) for (const v of byPos[pos]) sumAll += v;
      total += benchWeight * (sumAll - bestXI);
    }
    return total;
  }

  // Lookahead-summed XI score: sum the predicted XI total across the next `horizon`
  // gameweeks using the lookahead cache. This is the multi-GW heuristic that
  // surfaces transfers paying off over 3-4 GWs, not just this week.
  function multiGWXIScore(squad, lookaheadCache, horizon) {
    let total = 0;
    for (let h = 0; h < horizon; h++) {
      const predForGw = {};
      for (const id in lookaheadCache) predForGw[id] = (lookaheadCache[id] || [])[h] || 0;
      total += predictedXIScore(squad, predForGw);
    }
    return total;
  }

  // Enumerate candidate single-transfer moves. Returns top-K ranked by the
  // multi-GW XI delta (lookahead sum). Quick enough because we cap candidates
  // per slot and reuse the lookahead cache.
  // forCombinator=true loosens club-cap check (still applies for budget +
  // memory) when state._maxPerClub === 1, so the N-combinator can construct
  // chained-legal moves. Only used by the N-combinator path; singles still
  // get the strict-checked enumeration. See gh#37.
  function enumerateMoves(squad, players, lookaheadCache, horizon, state, K, forCombinator) {
    const inSquad = new Set(squad.map(p => p.id));
    const baseline = multiGWXIScore(squad, lookaheadCache, horizon);

    // Sort candidates per position by total lookahead value once (reused per swap).
    const byPos = { 1: [], 2: [], 3: [], 4: [] };
    for (const p of players) {
      if (inSquad.has(p.id)) continue;
      if (!byPos[p.element_type]) continue;
      const total = (lookaheadCache[p.id] || []).reduce((s, v) => s + v, 0);
      byPos[p.element_type].push({ player: p, total });
    }
    for (const pos of [1, 2, 3, 4]) byPos[pos].sort((a, b) => b.total - a.total);

    const rankedMoves = [];
    const candidatesPerSlot = 6;
    // Transfer memory: don't bring back a player we just transferred OUT in the
    // last `MEMORY_WINDOW` GWs, and don't transfer OUT a player we just brought IN.
    // Prevents flip-flop wasted moves that cost free transfers + hits.
    const MEMORY_WINDOW = 3;
    const recentlyOut   = new Set(((state.recentTransfers || []).filter(t => t.outAt >= (state.currentGw || 0) - MEMORY_WINDOW)).map(t => t.outId));
    const recentlyIn    = new Set(((state.recentTransfers || []).filter(t => t.inAt  >= (state.currentGw || 0) - MEMORY_WINDOW)).map(t => t.inId));

    // gh#37: thread _maxPerClub (was hardcoded > 3). Singles that violate
    // the cap as standalone moves are still pruned here — illegal singles
    // can't be applied directly. The N-combinator gets a SEPARATELY-built
    // wider atom pool (see enumerateChainAtoms below) that includes "would-
    // be-illegal-alone" moves so chained-legal N-tuples can form.
    const maxClub = state._maxPerClub || 3;
    // gh#37: under mpc1, the per-atom club check rejects swaps that are only
    // legal as part of an N-chain. When called for the combinator's atom
    // pool, skip the per-atom check in that case and rely on the combinator's
    // final-state legality verification.
    const skipPerAtomClub = forCombinator && maxClub === 1;
    for (const out of squad) {
      // Don't transfer OUT a player we recently brought IN.
      if (recentlyIn.has(out.id)) continue;
      const candidates = (byPos[out.element_type] || []).slice(0, candidatesPerSlot);
      for (const { player: cand } of candidates) {
        // Don't bring back a player we recently transferred OUT.
        if (recentlyOut.has(cand.id)) continue;
        const sellPrice = out.price;
        const buyPrice  = cand.price;
        if (state.budget + sellPrice < buyPrice) continue;
        if (!skipPerAtomClub) {
          const newClubCount = (state.clubCount[cand.team] || 0) - (out.team === cand.team ? 1 : 0) + 1;
          if (newClubCount > maxClub) continue;
        }

        const tempSquad = squad.filter(p => p.id !== out.id).concat([cand]);
        const newScore  = multiGWXIScore(tempSquad, lookaheadCache, horizon);
        // gh#43: convex price penalty, points-units, so the hit-cost arithmetic
        // downstream stays consistent. Prices are 0.1m units → /10 for £m.
        const buyM = buyPrice / 10, sellM = sellPrice / 10;
        const tilt = planPriceLambda * (buyM * buyM - sellM * sellM);
        const gain = newScore - baseline - tilt;
        if (gain <= 0.01) continue;

        rankedMoves.push({
          type: 'transfer',
          out, in: cand,
          sellPrice, buyPrice,
          quickGain: gain
        });
      }
    }

    // Drop moves with marginal multi-GW gain. Floor scales with free transfers:
    // when freeTransfers >= 1 we use a softer floor; when freeTransfers === 0 we
    // require a much bigger gain because the next "free" transfer is a week away
    // and using one now costs the opportunity to bank a 2nd free.
    const baseMin   = state.minGain != null ? state.minGain : 2.0;
    const noFreeMin = baseMin + 2.0;   // 2pt extra penalty when no free available
    const MIN_GAIN  = state.freeTransfers === 0 ? noFreeMin : baseMin;
    const filtered = rankedMoves.filter(m => m.quickGain >= MIN_GAIN);
    filtered.sort((a, b) => b.quickGain - a.quickGain);
    return filtered.slice(0, K);
  }

  // Generate N-transfer combo moves (N simultaneous swaps in the same GW).
  // Uniform shape: { type: 'transferN', pairs: [{out, in, sellPrice, buyPrice}], quickGain }.
  // Cost = max(0, N - freeTransfers) * hitCost; payoff must justify both the
  // hit AND the opportunity cost of burning multiple free transfers vs banking.
  // Useful when:
  //  - Multiple players need replacing and waiting loses more than the hit.
  //  - Banked FTs (now up to 5) want spending in a single high-payoff GW.
  //  - mpc1 forces dependent eviction chains (drop A from over-rep club,
  //    forced replacement may itself force another swap).
  //
  // Construction: seed from singles, extend by one at each N. We bound the
  // search by capping the per-order beam: triples extend top-4 pairs, quads
  // extend top-3 triples. Combinatorics stay manageable.
  function enumerateNTransferMoves(squad, lookaheadCache, horizon, state, K, singleMoves, N) {
    if (N < 2 || singleMoves.length < N) return [];
    const baseline = multiGWXIScore(squad, lookaheadCache, horizon);

    // Convert singles to pairs-shape seed: each is a 1-pair move.
    let order = singleMoves.map(m => ({
      pairs: [{ out: m.out, in: m.in, sellPrice: m.sellPrice, buyPrice: m.buyPrice }]
    }));

    // Pool of candidate singles to extend with at each step (top-6 keeps
    // breadth without combinatorial blowup).
    const extendPool = singleMoves.slice(0, 6);
    // Per-order beam cap so triples/quads don't explode.
    const PER_ORDER_BEAM = { 2: 8, 3: 6, 4: 4 };

    for (let n = 2; n <= N; n++) {
      const next = [];
      for (const base of order) {
        const usedOut = new Set(base.pairs.map(p => p.out.id));
        const usedIn  = new Set(base.pairs.map(p => p.in.id));
        for (const m of extendPool) {
          if (usedOut.has(m.out.id) || usedIn.has(m.in.id)) continue;
          // Maintain a canonical ordering (sort by out.id) to dedupe permutations.
          const newPairs = base.pairs.concat([{ out: m.out, in: m.in, sellPrice: m.sellPrice, buyPrice: m.buyPrice }]);
          newPairs.sort((a, b) => a.out.id - b.out.id);
          next.push({ pairs: newPairs });
        }
      }
      // Dedupe by canonical key.
      const seen = new Set();
      order = next.filter(o => {
        const key = o.pairs.map(p => p.out.id + ':' + p.in.id).join('|');
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      // Score + prune to per-order beam (only at the target N do we apply K).
      const scored = [];
      for (const o of order) {
        const outIds = new Set(o.pairs.map(p => p.out.id));
        const inPlayers = o.pairs.map(p => p.in);
        const tempSquad = squad.filter(p => !outIds.has(p.id)).concat(inPlayers);
        // Budget check.
        const netCost = o.pairs.reduce((s, p) => s + p.buyPrice - p.sellPrice, 0);
        if (state.budget < netCost) continue;
        // Club cap.
        const clubCount = Object.assign({}, state.clubCount);
        for (const p of o.pairs) {
          clubCount[p.out.team] = (clubCount[p.out.team] || 0) - 1;
          clubCount[p.in.team]  = (clubCount[p.in.team]  || 0) + 1;
        }
        let clubOK = true;
        const maxClub = state._maxPerClub || 3;
        for (const t in clubCount) if (clubCount[t] > maxClub) { clubOK = false; break; }
        if (!clubOK) continue;
        // Score.
        const newScore = multiGWXIScore(tempSquad, lookaheadCache, horizon);
        // gh#43: convex price penalty summed across all pairs in the N-move.
        const tilt = planPriceLambda * o.pairs.reduce((s, p) => {
          const buyM = p.buyPrice / 10, sellM = p.sellPrice / 10;
          return s + buyM * buyM - sellM * sellM;
        }, 0);
        const gain = newScore - baseline - tilt;
        if (gain <= 0.01) continue;
        scored.push({ pairs: o.pairs, quickGain: gain });
      }
      scored.sort((a, b) => b.quickGain - a.quickGain);
      const cap = (n < N) ? (PER_ORDER_BEAM[n] || 4) : K;
      order = scored.slice(0, cap);
      if (!order.length) return [];
    }

    return order.map(o => ({ type: 'transferN', pairs: o.pairs, quickGain: o.quickGain }));
  }

  // Apply a move to a state, returning a new state.
  function applyMove(state, squad, move, hitCost) {
    if (move.type === 'hold') {
      return {
        squad,
        freeTransfers: Math.min(FT_CAP, state.freeTransfers + 1),
        budget: state.budget,
        clubCount: state.clubCount,
        hitsPaid: 0
      };
    }
    if (move.type === 'transferN') {
      const outIds = new Set(move.pairs.map(p => p.out.id));
      const inPlayers = move.pairs.map(p => p.in);
      const newSquad = squad.filter(p => !outIds.has(p.id)).concat(inPlayers);
      const newClubCount = Object.assign({}, state.clubCount);
      for (const pp of move.pairs) {
        newClubCount[pp.out.team] = (newClubCount[pp.out.team] || 1) - 1;
        newClubCount[pp.in.team]  = (newClubCount[pp.in.team]  || 0) + 1;
      }
      const N = move.pairs.length;
      const transfersOverFree = Math.max(0, N - state.freeTransfers);
      const hits = transfersOverFree * hitCost;
      const netBudget = move.pairs.reduce((s, pp) => s + pp.sellPrice - pp.buyPrice, 0);
      return {
        squad: newSquad,
        freeTransfers: nextFree(Math.max(0, state.freeTransfers - N)),
        budget: state.budget + netBudget,
        clubCount: newClubCount,
        hitsPaid: hits
      };
    }
    // Single transfer
    const newSquad = squad.filter(p => p.id !== move.out.id).concat([move.in]);
    const newClubCount = Object.assign({}, state.clubCount);
    newClubCount[move.out.team] = (newClubCount[move.out.team] || 1) - 1;
    newClubCount[move.in.team]  = (newClubCount[move.in.team]  || 0) + 1;
    const hits = state.freeTransfers >= 1 ? 0 : hitCost;
    const newFree = nextFree(Math.max(0, state.freeTransfers - 1));
    return {
      squad: newSquad,
      freeTransfers: newFree,
      budget: state.budget + move.sellPrice - move.buyPrice,
      clubCount: newClubCount,
      hitsPaid: hits
    };
  }

  function plan(squad, gw, ctx, variantName, ensembleWeights, players, state, opts) {
    const horizon   = opts.horizon || 4;
    const beamWidth = opts.beamWidth || 4;
    const hitCost   = opts.hitCost || 4;
    const topK      = opts.topKMoves || 5;
    const hitThreshold = opts.hitThreshold || 8;
    const lookaheadCache = opts.lookaheadCache;   // { [playerId]: [gw0pred, gw1pred, gw2pred, gw3pred] }
    // gh#44 Layer B: align the planner's XI scoring with the execution policy.
    // gh#47 Stage 2: 'max' default (fluid per-GW formation).
    setFormationPolicy(opts.formationPolicy || 'max');
    setPriceLambda(opts.priceLambda);
    setBenchWeight(opts.benchWeight);
    setFtAccrual(opts.ftAccrual);

    // Initial beam: one state (the current state, no moves yet).
    let beam = [{
      squad,
      freeTransfers: state.freeTransfers,
      budget: state.budget,
      clubCount: state.clubCount,
      totalScore: 0,
      firstMove: { type: 'hold' },
      // gh#37: thread through so enumerateMoves can read it in nested calls.
      _maxPerClub: state._maxPerClub,
      // Pass through transfer-memory + currentGw too so deeper-beam atom enum
      // sees the same flip-flop guard as the root call.
      recentTransfers: state.recentTransfers,
      currentGw: state.currentGw,
      minGain: state.minGain
    }];

    for (let depth = 0; depth < horizon; depth++) {
      // Build the predicted-by-id map for THIS planning gameweek (gw + depth).
      const predForGw = {};
      for (const p of players) {
        const arr = lookaheadCache[p.id];
        predForGw[p.id] = arr ? (arr[depth] || 0) : 0;
      }

      const nextBeam = [];
      // gh#37: pass-through fields that don't change between beam steps but
      // must persist so deeper-beam atom enumeration sees them.
      const inherit = (s) => ({
        _maxPerClub: s._maxPerClub,
        recentTransfers: s.recentTransfers,
        currentGw: s.currentGw,
        minGain: s.minGain
      });
      for (const s of beam) {
        // Move: hold (no transfer this GW).
        {
          const newState = applyMove(s, s.squad, { type: 'hold' }, hitCost);
          const xiScore  = predictedXIScore(s.squad, predForGw);
          nextBeam.push({
            squad: newState.squad,
            freeTransfers: newState.freeTransfers,
            budget: newState.budget,
            clubCount: newState.clubCount,
            totalScore: s.totalScore + xiScore,
            firstScore: depth === 0 ? xiScore : s.firstScore,
            firstMove: depth === 0 ? { type: 'hold' } : s.firstMove,
            ...inherit(s)
          });
        }

        // Move: single transfers. Multi-GW heuristic surfaces moves that pay off
        // in 2-4 GWs (not just this one).
        const remainingHorizon = horizon - depth;
        const singleMoves = enumerateMoves(s.squad, players, lookaheadCache, remainingHorizon, s, topK);
        for (const m of singleMoves) {
          if (s.freeTransfers === 0 && m.quickGain < hitThreshold) continue;
          const newState = applyMove(s, s.squad, m, hitCost);
          const xiScore  = predictedXIScore(newState.squad, predForGw);
          nextBeam.push({
            squad: newState.squad,
            freeTransfers: newState.freeTransfers,
            budget: newState.budget,
            clubCount: newState.clubCount,
            totalScore: s.totalScore + xiScore - newState.hitsPaid,
            firstScore: depth === 0 ? xiScore - newState.hitsPaid : s.firstScore,
            firstMove: depth === 0 ? { type: 'transfer', out: m.out, in: m.in, sellPrice: m.sellPrice, buyPrice: m.buyPrice } : s.firstMove,
            ...inherit(s)
          });
        }

        // gh#37: build a wider atom pool for the N-combinator. Under mpc>=2
        // this returns the same set as `singleMoves`; under mpc1 it includes
        // standalone-illegal atoms that may compose into legal N-tuples.
        // _maxPerClub is invariant per plan() call so we read from the
        // outer state, not the beam state.
        const combinatorAtoms = (state._maxPerClub === 1)
          ? enumerateMoves(s.squad, players, lookaheadCache, remainingHorizon, s, topK * 3, true)
          : singleMoves;

        // Move: N-transfer combos (N=2..MAX_SWAP_N). For each N, require the
        // multi-GW gain to clear projected hit by hitThreshold per non-free
        // transfer (banking-aware: with 5 FTs available, triples/quads pay
        // zero hit and should be evaluated against the opportunity cost of
        // not banking, which the beam captures implicitly via the hold path).
        for (let N = 2; N <= MAX_SWAP_N; N++) {
          // Skip enumerating order N if we can't reasonably afford it:
          // require fewer than 3 hits worth of paid transfers (heuristic cap).
          const transfersOverFree = Math.max(0, N - s.freeTransfers);
          if (transfersOverFree > 3) continue;
          // N=5 is expensive (combinatoric) and almost never optimal under
          // a hit. Only enumerate when zero hits are paid (5 banked FTs).
          if (N === 5 && transfersOverFree > 0) continue;
          const nMoves = enumerateNTransferMoves(s.squad, lookaheadCache, remainingHorizon, s, 3, combinatorAtoms, N);
          for (const m of nMoves) {
            const projectedHit = transfersOverFree * hitCost;
            // Threshold scales with hits: per-paid-transfer threshold so a
            // quad that pays 2 hits needs to clear (2*hitCost + 2*hitThreshold).
            // No paid transfers → require positive marginal gain over hold
            // (already enforced by quickGain > 0.01 in the enumerator).
            if (transfersOverFree > 0 && m.quickGain < projectedHit + transfersOverFree * hitThreshold) continue;
            const newState = applyMove(s, s.squad, m, hitCost);
            const xiScore  = predictedXIScore(newState.squad, predForGw);
            nextBeam.push({
              squad: newState.squad,
              freeTransfers: newState.freeTransfers,
              budget: newState.budget,
              clubCount: newState.clubCount,
              totalScore: s.totalScore + xiScore - newState.hitsPaid,
              firstScore: depth === 0 ? xiScore - newState.hitsPaid : s.firstScore,
              firstMove: depth === 0 ? { type: 'transferN', pairs: m.pairs } : s.firstMove,
              ...inherit(s)
            });
          }
        }
      }

      // Prune to beam width. opts.firstFilter (off by default) restricts the
      // first move, so callers can price alternatives such as "hold" or "two
      // transfers" with the same search.
      nextBeam.sort((a, b) => b.totalScore - a.totalScore);
      const kept = depth === 0 && opts.firstFilter ? nextBeam.filter(x => opts.firstFilter(x.firstMove)) : nextBeam;
      beam = kept.slice(0, beamWidth);
      if (!beam.length) break;
    }

    if (!beam.length) return { type: 'hold' };
    // gh#78: callers comparing two starting states (live dead-asset rule) need
    // the plan's horizon total, not just the first move. Off by default so the
    // backtest return shape is byte-identical.
    if (opts.returnTotal) return Object.assign({}, beam[0].firstMove, { _total: beam[0].totalScore, _first: beam[0].firstScore });
    return beam[0].firstMove;
  }

  // Ranked alternatives for the first move: hold, best single, best double,
  // best triple, each with its first-GW and horizon totals. Read-only; the
  // chosen move is still plan()'s.
  const MOVE_SIZE = m => !m || m.type === 'hold' ? 0 : m.type === 'transfer' ? 1 : m.pairs.length;
  function planAlternatives(squad, gw, a, b, c, players, state, opts) {
    const chosen = plan(squad, gw, a, b, c, players, state, Object.assign({}, opts, { returnTotal: true }));
    const out = [];
    for (const n of [0, 1, 2, 3]) {
      const r = plan(squad, gw, a, b, c, players, state, Object.assign({}, opts, { returnTotal: true, firstFilter: m => MOVE_SIZE(m) === n }));
      if (!r || r._total == null || MOVE_SIZE(r) !== n) continue;
      out.push({ size: n, move: r, first: r._first, total: r._total });
    }
    out.sort((x, y) => y.total - x.total);
    const chosenSize = MOVE_SIZE(chosen);
    for (const o of out) o.chosen = o.size === chosenSize;
    return { chosen, alternatives: out };
  }

  const api = { plan, planAlternatives, predictedXIScore, enumerateMoves, enumerateNTransferMoves, FT_CAP, MAX_SWAP_N };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else if (typeof window !== 'undefined') window.fplPlanner = api;
})();
