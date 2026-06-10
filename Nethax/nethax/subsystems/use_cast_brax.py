"""Brax-style rewrites of ``monster_use_item`` and ``monster_cast_spell``.

Why this file exists
--------------------
Under ``jax.vmap`` (the validator's per-monster fast path),
``jax.lax.cond`` lowers to ``lax.select``: BOTH branches are always
compiled into the HLO graph and the result is masked.  Chains of nested
``cond`` calls — like the three-stage muse priority in
``monster_use_item`` or the gated cast in ``monster_cast_spell`` —
expand into deep, branchy HLO that the XLA pattern matcher struggles to
fuse.  On the H100 redesign branch that compile cost explodes.

Brax (and Craftax, the JAX NetHack-style env) avoid this by **always**
computing both sides of every conditional and using ``jnp.where`` to
pick the kept result.  The HLO is then flat and fusion-friendly.

Byte-parity contract
--------------------
The original helpers (``_try_heal`` / ``_try_scroll_teleport`` /
``_try_zap_wand`` / the ``_cast`` inner) are already pure functional —
they consume a split RNG key, build a new state, and return it.
``jax.random.split`` is performed unconditionally in both the original
and the Brax form, so the keys handed to each helper are identical in
the two versions.  Calling a helper and then discarding the resulting
state (because the predicate is False) is **byte-equivalent** to
not calling it at all: no global RNG state, no side effects.

Pytree-mask selection
---------------------
Because every helper returns an ``EnvState`` pytree of the same shape
as the input state, ``jax.tree.map(lambda a, b: jnp.where(p, a, b),
state_true, state_false)`` selects per-leaf with the scalar predicate
``p`` broadcasting across each leaf.  This is the same pattern used in
``items_scrolls._explode_dispatch`` and in Brax's physics step.

Conditionals flattened
----------------------
- ``monster_use_item``     : 3 × ``lax.cond`` → 3 × pytree-where.
- ``monster_cast_spell``   : 1 × ``lax.cond`` → 1 × pytree-where.

The signatures match the originals byte-for-byte so the Brax versions
are drop-in callable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.monster_ai import (
    MCAST_PSI_BOLT,
    _can_use_items,
    _chebyshev_dist,
    _is_mage_entry,
    _mcastu_cooldown,
    _monster_level,
    _try_heal,
    _try_scroll_teleport,
    _try_zap_wand,
)

# Round 2 brax integration: use Brax versions when NETHAX_BRAX_ALL=1.
import os as _os_uc
if _os_uc.environ.get("NETHAX_BRAX_ALL", "0") == "1":
    from Nethax.nethax.subsystems.preloop_brax import (
        monster_can_see_player_brax as monster_can_see_player,
    )
    from Nethax.nethax.subsystems.mattackm_brax import (
        monster_cast_damage_brax as monster_cast_damage,
    )
else:
    from Nethax.nethax.subsystems.monster_ai import (
        monster_can_see_player,
        monster_cast_damage,
    )


def _select_state(pred: jnp.ndarray, state_true, state_false):
    """Broadcast a scalar bool predicate across two same-shape state pytrees.

    Equivalent to ``lax.cond(pred, lambda: state_true, lambda: state_false)``
    when ``state_true`` is already computed.  Used to replace ``lax.cond``
    in the Brax rewrite — both branches are evaluated up front, this
    just picks the leaves.
    """
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b),
                        state_true, state_false)


def monster_use_item_brax(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Brax-style rewrite of ``monster_use_item``.

    Identical behaviour to the original; every ``lax.cond`` is replaced
    by always-compute + pytree ``jnp.where`` selection.

    Vendor citations (unchanged from original):
        - muse.c:1428 ``_can_use_items`` gate.
        - muse.c defensive / misc / offensive priority order.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    # Vendor entry gate (muse.c:1428).
    eligible = (
        _can_use_items(mai.entry_idx[idx])
        & mai.alive[idx]
        & ~mai.asleep[idx]
        & ~mai.peaceful[idx]
    )

    # HP threshold: vendor find_defensive — using 1/4 max as in the original.
    hp_low_quarter = (
        mai.hp[idx].astype(jnp.int32) * jnp.int32(4)
        < mai.hp_max[idx].astype(jnp.int32)
    )
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)
    in_los = monster_can_see_player(state, idx)

    quaff_heal = eligible & hp_low_quarter
    read_tport = eligible & ~hp_low_quarter & (dist == 1)
    zap_wand   = eligible & ~hp_low_quarter & (dist > 1) & in_los & (dist <= 8)

    # Same split order / keys as the original so each helper sees the
    # byte-identical sub-key.
    rng_heal, rng_tport, rng_zap = jax.random.split(rng, 3)

    # Brax pattern: always-compute the "branch taken" state, then mask.
    s_heal   = _try_heal(state, rng_heal, idx)
    s1 = _select_state(quaff_heal, s_heal, state)

    s_tport  = _try_scroll_teleport(s1, rng_tport, idx)
    s2 = _select_state(read_tport, s_tport, s1)

    s_zap    = _try_zap_wand(s2, rng_zap, idx)
    s3 = _select_state(zap_wand, s_zap, s2)

    return s3


def monster_cast_spell_brax(state, rng: jax.Array, monster_idx: jnp.ndarray,
                            spellnum: int = MCAST_PSI_BOLT):
    """Brax-style rewrite of ``monster_cast_spell``.

    The single ``lax.cond(can_cast, _cast, identity, state)`` is replaced
    by always-computing the post-cast state and pytree-masking against
    the original ``state`` with the ``can_cast`` predicate.

    Vendor citations (unchanged from original):
        - mcastu.c:129-305 ``castmu`` outer flow.
        - mcastu.c:175-179 cooldown gate.
        - mcastu.c:184-186 ``mspec_used`` cooldown set.
        - mcastu.c:240-243 damage formula.

    ``spellnum`` remains a Python literal so ``monster_cast_damage``
    dispatches at trace time, matching the original test contract.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    is_mage = _is_mage_entry(mai.entry_idx[idx])
    alive_active = mai.alive[idx] & ~mai.asleep[idx] & ~mai.peaceful[idx]
    in_los = monster_can_see_player(state, idx)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)
    in_range = dist <= 12
    # Vendor cooldown gate (mcastu.c:175-179): cannot cast if mspec_used>0.
    not_on_cd = mai.mspec_used[idx].astype(jnp.int32) <= jnp.int32(0)

    can_cast = is_mage & alive_active & in_los & in_range & not_on_cd

    # Always-compute the post-cast state.  ``rng`` is consumed
    # unconditionally; that's RNG-safe because ``jax.random.split`` /
    # ``jax.random.randint`` are pure functions of the key, so calling
    # them with the same key and discarding the output (via the mask
    # below) is byte-equivalent to not calling them at all.
    ml = _monster_level(mai.entry_idx[idx])
    dmg = monster_cast_damage(rng, spellnum, ml)
    new_hp_cast = jnp.maximum(state.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
    new_done_cast = state.done | (new_hp_cast <= 0)
    cd = _mcastu_cooldown(ml).astype(jnp.int16)
    new_mspec_cast = state.monster_ai.mspec_used.at[idx].set(cd)
    new_mai_cast = state.monster_ai.replace(mspec_used=new_mspec_cast)
    cast_state = state.replace(
        player_hp=new_hp_cast,
        done=new_done_cast,
        monster_ai=new_mai_cast,
    )

    return _select_state(can_cast, cast_state, state)
