"""Brax-style rewrites of the per-slot scan bodies used by ``monsters_step_all``.

These mirror ``_mcalc_one_bp_impl`` and ``_mattackm_one_impl`` from
``monster_ai.py`` but follow the Brax / Craftax convention of flattening
``jax.lax.cond`` into ``jnp.where`` masking wherever it is byte-parity-safe
to do so.

Motivation
----------
Both bodies run inside ``jax.lax.scan`` 400 times per env-step.  Under
``vmap``-over-seeds, every ``lax.cond`` is lowered as ``select`` and BOTH
branches are inlined into HLO, regardless of mask.  Brax flattens this
pattern by always computing both sides and selecting via ``jnp.where``,
which produces a single straight-line trace that fuses more cleanly.

Byte-parity caveat (load-bearing)
---------------------------------
Two of the original conds gate ISAAC64-mutating side-effects:

  * ``_mcalc_one_bp_impl``     →  ``rn2(NORMAL_SPEED)`` per valid+alive slot
  * ``_mattackm_one_impl``     →  ``mattackm(...)`` per striker

Vendor draws the RNG ONLY in those branches (see ``mon.c::mcalcmove`` and
``mhitm.c::mattackm``).  Always-advancing under a mask would reorder the
ISAAC64 stream across slots and break the byte-parity validator.

These two conds are therefore preserved verbatim — they are the ONLY
remaining ``lax.cond`` calls in either function.  Every other branch in
the originals was already ``jnp.where``-masked, so the Brax rewrites are
mostly a structural re-statement that documents the "compute eagerly,
mask outputs" pattern explicitly.

Conds flattened
---------------
* ``_mcalc_one_bp_brax_impl``  — 0 flattened, 1 retained for RNG parity.
* ``_mattackm_one_brax_impl``  — 0 flattened, 1 retained for RNG parity.

Total: 0 new flattenings.  Both bodies were already maximally flat apart
from the RNG-gating conds, which CANNOT be flattened without breaking
``RNG draw order preserved exactly`` (project constraint #1).

If/when the byte-parity validator is retired, drop the two ``lax.cond``
calls and replace with the commented-out always-advance template marked
``# BRAX-IDEAL`` in each function.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax import vendor_rng as _vendor_rng_mod
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    _MM_HAS_ZOMBIE_FORM,
    _MM_IS_PURPLE_WORM,
    _MM_IS_SHRIEKER,
    _MM_IS_ZOMBIE_MAKER,
    _MOVEMENT_THRESHOLD,
    mattackm,
)


# ---------------------------------------------------------------------------
# Brax-style _mcalc_one_bp
# ---------------------------------------------------------------------------

def _mcalc_one_bp_brax_impl(state, k_idx, pre_round):
    """Brax-style per-slot mcalcmove body (byte-parity preserved).

    Structural changes vs. ``_mcalc_one_bp_impl``:
      * All scalar predicates computed up front in a single straight-line
        block (no branches gating intermediate values).
      * ``movement_points`` update written as a single ``jnp.where`` over
        the whole array — no ``.at[].set`` inside a mask branch.
      * The one remaining ``lax.cond`` gates the ISAAC64 draw and is
        load-bearing for byte parity (see module docstring).

    Cite: vendor/nethack/src/mon.c::mcalcmove lines 1126-1167.
    """
    NS = jnp.int32(_MOVEMENT_THRESHOLD)
    mi = state.monster_ai

    # --- straight-line predicate block ---------------------------------
    slot = mi.fmon_order[k_idx]
    valid = slot >= jnp.int32(0)
    safe_slot = jnp.where(valid, slot, jnp.int32(0))
    alive_s = jnp.where(valid, mi.alive[safe_slot], jnp.bool_(False))
    need_draw = valid & alive_s

    pre = pre_round[safe_slot]
    adj = pre % NS
    floored = pre - adj

    # --- RNG draw (conditional; parity-load-bearing) -------------------
    # BRAX-IDEAL (byte-parity off):
    #   vrng2, v = _vendor_rng_mod.rn2_jax(state.vendor_rng, NS)
    # Retained cond: vendor only draws when (valid & alive).  Always
    # advancing would desync ISAAC64 across slots where the predicate is
    # False, breaking the multiseed byte-parity validator.
    def _draw(vr):
        return _vendor_rng_mod.rn2_jax(vr, NS)

    def _nodraw(vr):
        return vr, jnp.int32(0)

    vrng2, v = jax.lax.cond(need_draw, _draw, _nodraw, state.vendor_rng)

    # --- masked add_pts + clipped new_mp -------------------------------
    add_pts_full = jnp.where(v < adj, floored + NS, floored)
    add_pts = jnp.where(need_draw, add_pts_full, jnp.int32(0))

    new_mp = jnp.clip(
        mi.movement_points[safe_slot].astype(jnp.int32) + add_pts,
        0, 32000,
    ).astype(jnp.int16)

    # --- masked write into movement_points -----------------------------
    updated_mp = mi.movement_points.at[safe_slot].set(new_mp)
    movement_points_out = jnp.where(valid, updated_mp, mi.movement_points)

    mi2 = mi.replace(movement_points=movement_points_out)
    return state.replace(monster_ai=mi2, vendor_rng=vrng2)


_mcalc_one_bp_brax_jit = jax.jit(_mcalc_one_bp_brax_impl)


# ---------------------------------------------------------------------------
# Brax-style _mattackm_one
# ---------------------------------------------------------------------------

def _mattackm_one_brax_impl(state, atk_slot, key_i, conflict_active):
    """Brax-style per-attacker mattackm strike body (byte-parity preserved).

    Structural changes vs. ``_mattackm_one_impl``:
      * Faction / pair / species predicates computed eagerly in a single
        flat block — no nested ``jnp.where`` chains.
      * Candidate selection collapses to a single ``argmax`` over the
        masked boolean array.
      * The one remaining ``lax.cond`` gates ``mattackm`` itself, which
        consumes ISAAC64 draws and may mutate monster_ai state.  Always
        calling it under a mask would desync RNG across non-striking
        slots; retained for byte parity.

    Cite: vendor/nethack/src/mhitm.c lines 1024-1100 (mattackm);
          vendor/nethack/src/mon.c::mm_aggression lines 2422-2447;
          vendor/nethack/src/uhitm.c — Conflict intrinsic gate (44).
    """
    mi = state.monster_ai

    # --- attacker slot validity / position -----------------------------
    i_raw = atk_slot.astype(jnp.int32)
    valid_i = i_raw >= jnp.int32(0)
    i32 = jnp.where(valid_i, i_raw, jnp.int32(0))

    atk_alive = mi.alive[i32] & valid_i
    pi = mi.pos[i32].astype(jnp.int32)

    # --- adjacency mask (Chebyshev distance == 1) ----------------------
    all_pos = mi.pos.astype(jnp.int32)
    d_row = jnp.abs(all_pos[:, 0] - pi[0])
    d_col = jnp.abs(all_pos[:, 1] - pi[1])
    adj = jnp.maximum(d_row, d_col) == 1

    # --- faction lattice (flat where-chains, no conds) -----------------
    is_tame_all = mi.tame
    is_peace_all = mi.peaceful & ~is_tame_all
    all_faction = jnp.where(
        is_tame_all, jnp.int32(2),
        jnp.where(is_peace_all, jnp.int32(1), jnp.int32(0)),
    )
    a_faction = all_faction[i32]
    is_hostile_atk = a_faction == jnp.int32(0)
    is_nonhostile_tgt = all_faction != jnp.int32(0)

    # --- pair index gate (idx > i32 to avoid double-counting) ----------
    idx_arr = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32)
    pair_ok = idx_arr > i32

    # --- species-specific aggression flags -----------------------------
    a_entry = jnp.clip(
        mi.entry_idx[i32].astype(jnp.int32),
        0, _MM_IS_PURPLE_WORM.shape[0] - 1,
    )
    all_entry = jnp.clip(
        mi.entry_idx.astype(jnp.int32),
        0, _MM_IS_PURPLE_WORM.shape[0] - 1,
    )
    a_is_pw = _MM_IS_PURPLE_WORM[a_entry]
    t_is_shr = _MM_IS_SHRIEKER[all_entry]
    a_is_zm = _MM_IS_ZOMBIE_MAKER[a_entry] & ~mi.cancelled[i32]
    t_has_zform = _MM_HAS_ZOMBIE_FORM[all_entry]

    pets_brawl = mi.tame[i32] & mi.tame

    species_purple = a_is_pw & t_is_shr & ~pets_brawl
    species_zombie = (
        a_is_zm & t_has_zform & ~pets_brawl & ~mi.tame[i32] & ~mi.tame
    )

    conflict_allow = conflict_active
    baseline_allow = is_hostile_atk & is_nonhostile_tgt
    per_target_allow = (
        baseline_allow | species_purple | species_zombie | conflict_allow
    )

    # --- candidate selection -------------------------------------------
    candidates = mi.alive & adj & pair_ok & per_target_allow
    has_target = jnp.any(candidates)
    j_idx = jnp.argmax(candidates).astype(jnp.int32)

    do_strike = atk_alive & has_target

    # --- strike (conditional; parity-load-bearing) ---------------------
    # BRAX-IDEAL (byte-parity off):
    #   struck = mattackm(state, i32, j_idx, key_i)
    #   return jax.tree_util.tree_map(
    #       lambda new, old: jnp.where(do_strike, new, old), struck, state)
    # Retained cond: mattackm advances ISAAC64 and mutates monster_ai.
    # Always calling it under a mask would desync RNG across non-striking
    # attackers; this would break multiseed byte parity.
    def _strike(ss):
        return mattackm(ss, i32, j_idx, key_i)

    def _no_strike(ss):
        return ss

    return jax.lax.cond(do_strike, _strike, _no_strike, state)


_mattackm_one_brax_jit = jax.jit(_mattackm_one_brax_impl)
