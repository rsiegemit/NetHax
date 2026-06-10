"""Brax-style flattened rewrites of the pre-monster / region / stinking-cloud
phase bodies.

Why this file exists
--------------------
``_pre_monster_jit`` and ``_monster_jit`` in ``Nethax/nethax/env.py`` wrap
the per-step pipeline phase bodies in ``jax.lax.cond(state.done, ...)``
short-circuits.  Under ``jax.vmap`` over seeds these lower to
``lax.select`` with **both** branches present in the HLO graph; the body
branch in turn calls into ``_maybe_seed_astral_mplayers`` / ``_dig_tick``
/ ``run_regions`` / ``_tick_stinking_cloud`` whose HLO is then duplicated
into the conditional.  On H100 this is one of the contributors to the
per-step compile blow-up that the Brax / Craftax pattern was designed to
eliminate (compute both branches once, then select with ``jnp.where``).

Following the same pattern as ``combat_helpers_brax`` /
``monster_attack_player_brax`` / ``use_cast_brax``: **always compute the
work**, then mask the result with ``jnp.where`` (via ``jax.tree.map`` for
the full pytree analogue of ``lax.cond``).  Inside each body, every
``lax.cond`` / ``lax.switch`` is replaced by ``jnp.where`` over a
precomputed predicate.

Byte-parity contract
--------------------
1. RNG draw order preserved exactly.  The pre-monster body draws from
   ``rng_astral`` (inside ``_maybe_seed_astral_mplayers``) and ``rng_act``
   (inside ``_dig_tick``) regardless of ``state.done``; ``run_regions``
   draws from its slot-split RNGs unconditionally; ``_tick_stinking_cloud``
   makes no RNG draws.  The originals' ``state.done`` short-circuit only
   discards the *result*, so the draw stream is the same on done/not-done.
2. Mutations byte-identical via ``jnp.where`` over pytrees with the same
   scalar mask (``state.done``) as the original ``lax.cond`` predicate.
3. State pytree shape preserved — both ``on_true`` (original ``state``)
   and ``on_false`` (computed ``ns``) are the same Flax dataclass type.

Conditionals flattened
----------------------
- ``pre_monster_body_brax``     : 1 ``lax.cond`` (the ``state.done`` gate
                                  inside ``_pre_monster_jit_impl``) → 1
                                  pytree-where.  The body itself contains
                                  0 ``lax.cond`` / ``lax.switch``; it
                                  composes two pure leaf calls
                                  (``_maybe_seed_astral_mplayers``,
                                  ``_dig_tick``) whose own internals are
                                  out of scope for this carve-out.
- ``run_regions_brax``          : 0 ``lax.cond`` / ``lax.switch`` in the
                                  vendor body — it is already fully
                                  vectorised with ``jnp.where`` and
                                  ``jax.vmap``.  Re-exposed here under
                                  the ``_brax`` name with no semantic
                                  change so the per-step pipeline can
                                  swap the import sites uniformly.
- ``tick_stinking_cloud_brax``  : 0 ``lax.cond`` / ``lax.switch`` in the
                                  original — already a straight-line
                                  ``jnp.where`` chain.  Re-exposed under
                                  the ``_brax`` name for the same
                                  uniform-swap reason.

The signatures match the originals so these are drop-in replacements.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.digging import dig_tick as _dig_tick
from Nethax.nethax.subsystems.regions import run_regions as _run_regions
from Nethax.nethax.subsystems.mplayer import (
    maybe_seed_astral_mplayers as _maybe_seed_astral_mplayers,
)


def _select_tree(cond, on_true, on_false):
    """Pytree analogue of ``jnp.where(cond, on_true, on_false)``.

    Both ``on_true`` and ``on_false`` must be pytrees with identical
    structure (same Flax dataclass type, same field dtypes).  Mirrors the
    helper in ``combat_helpers_brax`` so the masking style is consistent
    across the Brax-port set.
    """
    return jax.tree.map(lambda a, b: jnp.where(cond, a, b), on_true, on_false)


# ---------------------------------------------------------------------------
# Phase 1a/1b — astral mplayer seed + dig tick.
#
# Original: ``_pre_monster_body`` (env.py:1460) wrapped by
# ``_pre_monster_jit_impl`` (env.py:1478) which does
#
#     jax.lax.cond(state.done, lambda _: state, lambda _: body(...), None)
#
# Brax rewrite: run the body unconditionally and ``jnp.where`` the pytree
# against the pre-step ``state`` on ``state.done``.
# ---------------------------------------------------------------------------

def pre_monster_body_brax(ns, state, rng_act, rng_astral, prev_branch, prev_level):
    """Brax-flat version of ``_pre_monster_jit_impl``.

    Computes both phase-1a (astral seed) and phase-1b (dig tick) on
    ``ns`` unconditionally, then selects between the original pre-step
    ``state`` (done case) and the updated ``ns`` (live case) via a
    single pytree ``jnp.where`` on ``state.done``.

    Parameters
    ----------
    ns : EnvState
        Post-dispatch state (output of ``_dispatch_jit``).
    state : EnvState
        Pre-step state snapshot, returned unchanged when ``state.done``.
    rng_act, rng_astral : jax.Array
        Per-phase RNG splits, consumed by ``_dig_tick`` and
        ``_maybe_seed_astral_mplayers`` respectively.
    prev_branch, prev_level : jax.Array
        Pre-dispatch (dungeon_branch, dungeon_level) snapshot used by the
        astral edge-trigger.
    """
    # 1a. Astral-Plane mplayer trigger — vendor mplayer.c::create_mplayers
    #     (lines 327-355) called from astral.lua MAP section on level
    #     entry.  Edge-triggered on (prev != Astral) → (curr == Astral).
    ns = _maybe_seed_astral_mplayers(ns, rng_astral, prev_branch, prev_level)

    # 1b. Digging tick — advance multi-turn pickaxe dig (dig.c::dodig).
    ns = _dig_tick(ns, rng_act)

    # Flatten the original ``lax.cond(state.done, ...)`` gate via
    # pytree ``jnp.where``.  Both branches have identical structure and
    # dtypes (same EnvState dataclass), so the select is byte-equal to
    # the cond's result on either branch.
    return _select_tree(state.done, state, ns)


# ---------------------------------------------------------------------------
# Phase 2b — per-turn region tick.
#
# Original: ``run_regions`` in ``subsystems/regions.py`` already contains
# 0 ``lax.cond`` / ``lax.switch`` — the slot-table tick is straight-line
# ``jnp.where`` + ``jax.vmap`` over a fixed ``MAX_REGIONS`` slot count.
# No flattening to do; re-export under the ``_brax`` name so the per-step
# pipeline can swap import sites uniformly with the other Brax ports.
# ---------------------------------------------------------------------------

def run_regions_brax(state, rng):
    """Brax-style alias for ``run_regions``.

    The vendor body is already a fixed-shape ``jnp.where`` tick over
    ``MAX_REGIONS`` slots (expire → effect → age) with no host-side
    branches and no ``lax.cond`` / ``lax.switch``.  Re-exposed here under
    the ``_brax`` naming so callers in the per-step pipeline can swap
    all three pre-monster phases to ``*_brax`` with a single import
    pattern.

    Conds flattened: 0 (none present in the original).
    """
    return _run_regions(state, rng)


# ---------------------------------------------------------------------------
# Phase 2c — stinking-cloud tick on the hero.
#
# Original: ``_tick_stinking_cloud`` in ``env.py`` already uses
# ``jnp.where`` everywhere; 0 ``lax.cond`` / ``lax.switch`` in the body.
# Reproduced here verbatim (modulo the ``TimedStatus`` import inlined to
# the module level instead of inside the function) so the Brax port set
# is self-contained.
# ---------------------------------------------------------------------------

# Module-level import mirrors the original function-local import in
# ``env.py:_tick_stinking_cloud``.  Placed at module load time here
# because re-importing on every call adds Python overhead under
# ``vmap``-replicated tracing.
from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS


def tick_stinking_cloud_brax(state):
    """Brax-style rewrite of ``_tick_stinking_cloud``.

    Per-turn stinking-cloud effect on the hero — cite vendor
    ``region.c::inside_gas_cloud`` (called from ``run_regions`` when the
    player tile is inside the gas-cloud rectangle).

    On each turn while ``cloud_turns > 0``:
      * if the hero is within Chebyshev radius of ``cloud_pos``, apply
        1 HP and bump VOMITING by 2 (matches vendor's
        ``losehp(1, ...)`` + ``set_property(VOMITING, ...)``);
      * decrement ``cloud_turns``.

    Conds flattened: 0 (original already uses ``jnp.where`` for all
    branches; this function is a verbatim re-host under the Brax naming).
    """
    turns = state.cloud_turns.astype(jnp.int32)
    active = turns > jnp.int32(0)
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    cr = state.cloud_pos[0].astype(jnp.int32)
    cc = state.cloud_pos[1].astype(jnp.int32)
    dr = jnp.abs(pr - cr)
    dc = jnp.abs(pc - cc)
    cheby = jnp.maximum(dr, dc)
    inside = active & (cheby <= state.cloud_radius.astype(jnp.int32))

    new_hp = jnp.where(
        inside,
        jnp.maximum(state.player_hp - jnp.int32(1), jnp.int32(0)),
        state.player_hp,
    ).astype(state.player_hp.dtype)
    new_done = state.done | (new_hp <= jnp.int32(0))

    ts = state.status.timed_statuses
    cur_vom = ts[int(_TS.VOMITING)].astype(jnp.int32)
    new_vom = jnp.where(inside, cur_vom + jnp.int32(2), cur_vom)
    new_ts = ts.at[int(_TS.VOMITING)].set(new_vom.astype(ts.dtype))
    new_status = state.status.replace(timed_statuses=new_ts)

    new_turns = jnp.where(
        active,
        (turns - jnp.int32(1)).astype(state.cloud_turns.dtype),
        state.cloud_turns,
    )

    return state.replace(
        player_hp=new_hp,
        done=new_done,
        status=new_status,
        cloud_turns=new_turns,
    )
