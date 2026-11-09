// Variant definitions for the FPL prediction ensemble.
//
// Each variant is a partial weight override applied on top of predict.js's
// DEFAULTS. Missing keys inherit from defaults. The same predictPlayer code
// path runs for every variant; only these numbers change.
//
// Design philosophy: each variant is a different SPECIALIST. They aren't
// supposed to be individually better than the baseline. They're supposed to be
// better on a specific player slice (e.g., set-piece takers, rotation risks)
// so an ensemble that learns who to trust where outperforms any one of them.

(function () {
  const VARIANTS = {
    // Reference. Identical to predict.js DEFAULTS. Always include this in the
    // ensemble as a "boring is OK" anchor.
    default: {},

    // Trusts xGI heavily, downweights recency. Picks premium attackers in good
    // fixtures even when they're cold. Tends to over-rate ice-cold stars and
    // under-rate hot bargains.
    xg_pure: {
      mid_recency: 0.40, mid_att: 0.60,
      fwd_recency: 0.40, fwd_att: 0.60,
      def_recency: 0.55, def_cs: 0.20, def_att: 0.25
    },

    // The opposite: maxes ep_next + form, ignores underlying numbers. Catches
    // hot streaks the xG-pure model misses. Vulnerable to mean-reversion.
    recency_pure: {
      recency_ep: 0.65, recency_form: 0.30, recency_season: 0.05,
      mid_recency: 0.85, mid_att: 0.15,
      fwd_recency: 0.85, fwd_att: 0.15,
      def_recency: 0.85, def_cs: 0.10, def_att: 0.05
    },

    // BPS-magnet defenders are the classic case predict.js misses. Triples the
    // bps/90 scaling. Useful for picking unexpected DEFs (TAA-types).
    bps_magnet: {
      bps_scale: 0.030
    },

    // Defcon-magnet: leans into the DefCon EV term — drops the confidence
    // haircut to 1.0 (base default 0.7) so defensive-action specialists are
    // valued at their full expected DefCon points. Only differs in live play;
    // the term is zero in backtests, which lack defensive_contribution data.
    defcon_magnet: {
      defcon_confidence: 1.0
    },

    // Pure-xGI underlying. att_xgi_blend = 1.0 means attRet uses only the
    // expected-goal-involvements signal; actual goals/assists are ignored.
    // The "lucky finisher discount" model: a player overperforming xG gets
    // pulled back to underlying. Best for premium attackers who just had a
    // hot run that won't sustain.
    xg_underlying: {
      att_xgi_blend: 1.0
    },

    // Independent cross-check (issue #13). Deliberately disjoint from every
    // other variant: zero recency block (no ep_next, no form, no last-5),
    // 100% underlying stats. Position blocks pinned to attRet so xGI / xGC
    // are the only signals driving the score. Not blended into the public
    // prediction; only used to flag disagreement with the main model.
    xpts_sim: {
      recency_ep: 0, recency_form: 0, recency_season: 1.0,
      mid_recency: 0,    mid_att: 1.0,
      fwd_recency: 0,    fwd_att: 1.0,
      def_recency: 0,    def_cs: 0.40, def_att: 0.60,
      att_xgi_blend: 1.0
    },

    // Form-heavy. More extreme than recency_pure: form gets 65%, ep_next gets
    // 30%, season-average is near zero. Position blocks lean even more toward
    // recency. Picks players surfing the current wave; vulnerable to mean
    // reversion. Different from recency_pure by inverting the ep/form mix:
    // recency_pure trusts FPL's own model, form_heavy trusts the last 5 GWs.
    form_heavy: {
      recency_ep: 0.30, recency_form: 0.65, recency_season: 0.05,
      mid_recency: 0.85, mid_att: 0.15,
      fwd_recency: 0.85, fwd_att: 0.15,
      def_recency: 0.85, def_cs: 0.10, def_att: 0.05
    }
  };

  // Default equal-weight blend per position. ensemble-fit.js learns better
  // weights and writes them to ensemble-weights.json; this is the fallback.
  // Default blend: equal across the 6 variants. ensemble-fit.js learns better
  // per-position weights and writes ensemble-weights.json; this fallback is used
  // only when those weights aren't available yet.
  const VARIANT_NAMES = ['default', 'xg_pure', 'recency_pure', 'bps_magnet', 'xg_underlying', 'form_heavy'];
  const equalBlend    = () => { const w = {}; VARIANT_NAMES.forEach(n => { w[n] = 1 / VARIANT_NAMES.length; }); return w; };
  const DEFAULT_BLEND = { 1: equalBlend(), 2: equalBlend(), 3: equalBlend(), 4: equalBlend() };

  // Blends sub-model predictions per position using weights[posId][variantName].
  // Falls back to DEFAULT_BLEND if weights are missing. Requires a working
  // fplPredict / predict module to be loaded.
  function predictEnsemble(player, fixList, ctx, weights) {
    const predict = (typeof window !== 'undefined' && window.fplPredict)
      ? window.fplPredict.predictPlayer
      : (typeof require === 'function' ? require('./predict.js').predictPlayer : null);
    if (!predict) throw new Error('predict.js not available');

    const pos = player.element_type;
    const w   = (weights && weights[pos]) || DEFAULT_BLEND[pos];
    let sum = 0, totalW = 0;
    for (const name in w) {
      const wi = w[name];
      if (wi <= 0) continue;
      const variantCtx = Object.assign({}, ctx, { variant: VARIANTS[name] });
      sum    += wi * predict(player, fixList, variantCtx);
      totalW += wi;
    }
    return totalW > 0 ? +(sum / totalW).toFixed(2) : 0;
  }

  // ─── Per-player ensemble routing (step 8) ───────────────────────────────────
  //
  // Different player archetypes respond best to different variants. Hard-coded
  // routing rules below; they're informed by the tournament where recency_pure
  // won overall but the per-variant MAEs showed xg_pure was best for premium
  // attackers and bps_magnet was best for defensive utility players.
  //
  // Archetypes:
  //   - premium_attacker  : high xGI per-90, high minutes share, high price.
  //                         Stable underlying numbers. Trust xg_pure / xg_underlying.
  //   - young_breakout    : low total minutes but high recent form. Trust recency_pure
  //                         and form_heavy; their season totals are misleading.
  //   - defensive_utility : DEF/MID with high BPS but low xGI. Trust bps_magnet
  //                         and default; they score via passing/clearances/CBI.
  //   - template_steady   : high ownership, moderate xGI. The "boring but reliable"
  //                         template picks. Trust the ensemble baseline.
  //   - fringe            : low minutes, no recent form spike. Bench/rotation risk.
  //                         Trust default; nothing else has reliable signal.
  function profileArchetype(player) {
    const minutes = player.minutes || 0;
    const xgi90   = parseFloat(player.expected_goal_involvements_per_90) || 0;
    const bpsPer90 = minutes > 0 ? (player.bps || 0) / (minutes / 90) : 0;
    const form    = parseFloat(player.form) || 0;
    const ownership = parseFloat(player.selected_by_percent) || 0;

    // Order matters: more specific archetypes first.
    if (minutes < 900 && form > 4)                          return 'young_breakout';
    if (minutes < 500)                                       return 'fringe';
    if (xgi90 >= 0.55 && minutes >= 1500)                    return 'premium_attacker';
    if (bpsPer90 >= 22 && xgi90 < 0.30 && player.element_type !== 1) return 'defensive_utility';
    if (ownership >= 20)                                     return 'template_steady';
    return 'fringe';
  }

  // Per-archetype variant blends. Hand-picked from tournament insights.
  // Each blend should sum to 1.0 across listed variants. Missing variants get 0.
  const BLEND_BY_ARCHETYPE = {
    premium_attacker:  { xg_pure: 0.55, xg_underlying: 0.30, default: 0.10, bps_magnet: 0.05 },
    young_breakout:    { recency_pure: 0.55, form_heavy: 0.30, default: 0.10, xg_pure: 0.05 },
    defensive_utility: { bps_magnet: 0.55, default: 0.30, recency_pure: 0.10, xg_underlying: 0.05 },
    template_steady:   { default: 0.40, xg_pure: 0.25, recency_pure: 0.20, bps_magnet: 0.10, xg_underlying: 0.05 },
    fringe:            { default: 0.70, recency_pure: 0.30 }
  };

  // Per-player-routed ensemble. Classifies the player, applies their archetype's
  // blend. Falls back to position-weighted ensemble if archetype unknown.
  function predictEnsembleRouted(player, fixList, ctx, positionWeights) {
    const predict = (typeof window !== 'undefined' && window.fplPredict)
      ? window.fplPredict.predictPlayer
      : (typeof require === 'function' ? require('./predict.js').predictPlayer : null);
    if (!predict) throw new Error('predict.js not available');

    const archetype = profileArchetype(player);
    const blend     = BLEND_BY_ARCHETYPE[archetype] || (positionWeights && positionWeights[player.element_type]) || DEFAULT_BLEND[player.element_type];
    let sum = 0, total = 0;
    for (const name in blend) {
      const w = blend[name];
      if (w <= 0) continue;
      if (!VARIANTS[name]) continue;
      const variantCtx = Object.assign({}, ctx, { variant: VARIANTS[name] });
      sum   += w * predict(player, fixList, variantCtx);
      total += w;
    }
    return total > 0 ? +(sum / total).toFixed(2) : 0;
  }

  const api = { VARIANTS, DEFAULT_BLEND, predictEnsemble, predictEnsembleRouted, profileArchetype, BLEND_BY_ARCHETYPE };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  } else if (typeof window !== 'undefined') {
    window.fplModels = api;
  }
})();
