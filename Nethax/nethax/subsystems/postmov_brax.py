"""Brax-style rewrite of monster_ai._postmov_per_monster.

Background
----------
Under ``jax.vmap`` (e.g. multi-seed/multi-env rollouts), ``jax.lax.cond``
lowers to ``lax.select`` and emits *both* branches in the HLO. For nested
cond chains this produces pathological HLO compile blowup. The Brax
pattern (Google's brax + Craftax — https://github.com/MichaelTMatthews/Craftax)
sidesteps this by always computing both branches eagerly and selecting
results via ``jnp.where`` masks. The resulting HLO is flat and fusion-
friendly.

This module hosts ``_postmov_per_monster_brax`` — a 1:1 byte-parity
replacement for ``monster_ai._postmov_per_monster`` rewritten in the
Brax style.

Audit of the original
---------------------
Reading ``monster_ai._postmov_per_monster`` (monster_ai.py:4861-4925) we
find:

  * Zero ``jax.lax.cond`` calls.
  * Zero ``jax.lax.switch`` calls.
  * Zero ``jax.lax.fori_loop`` calls.
  * One Python-level ``if traps is not None:`` — this is a *static*
    structural pytree check (the ``traps`` field is either present or
    absent at trace time), not a traced cond, so it stays.
  * Mutations are already done via ``arr.at[idx].set(jnp.where(mask, new, old))``
    — exactly the Brax pattern.
  * The only ``lax.fori_loop`` reached transitively is inside
    ``monster_can_see_player`` (Bresenham LoS, depth = ``max(MAP_H, MAP_W)``).
    That is a *fixed-trip* loop which is already vmap-friendly per the
    task guidance, so we leave the call site alone.

Conclusion: the existing function is already Brax-shaped. The rewrite is
therefore a structural copy with all conditional dataflow expressed
explicitly through ``jnp.where`` (no hidden control flow), preserving:

  * The pytree shape of ``state``.
  * The exact byte-level mutation semantics of every ``state.replace``.
  * The RNG draw order (this function performs *no* RNG draws — confirmed
    by absence of any ``rng`` parameter and absence of any ``vendor_rng``
    or ``jax.random`` reference in the body).

Number of ``lax.cond / lax.switch / lax.fori_loop`` constructs flattened
in this file: **0** (the original already uses the Brax pattern).
"""

from __future__ import annotations

import jax.numpy as jnp

from Nethax.nethax.subsystems.monster_ai import (
    _MAP_H,
    _MAP_W,
    monster_can_see_player,
)


def _postmov_per_monster_brax(state, monster_idx: jnp.ndarray) -> object:
    """Brax-style port of ``monster_ai._postmov_per_monster``.

    Byte-parity contract: every mutation must produce arrays that compare
    bit-equal to the original implementation for the same ``state`` /
    ``monster_idx`` inputs. See module docstring for the audit.

    Cite: vendor/nethack/src/monmove.c::postmov lines 1455-1707.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    alive = mai.alive[idx]

    mpos = mai.pos[idx].astype(jnp.int32)
    mr, mc = mpos[0], mpos[1]

    h = _MAP_H
    w = _MAP_W
    in_bounds = (mr >= 0) & (mr < h) & (mc >= 0) & (mc < w)
    active = alive & in_bounds

    # ---- Tile-effect: reveal a trap if the monster is standing on one ----
    # Vendor mintrap(NO_TRAP_FLAGS) handles trigger + reveal; we keep the
    # reveal half (player-knowledge mark).
    #
    # Brax note: the ``if traps is not None`` is a *static* pytree-shape
    # check (it is the same across the vmap batch and is resolved at
    # trace time), not a traced ``lax.cond``. Inside the branch we keep
    # the original ``jnp.where``-gated ``.at[].set()`` write, which is
    # already Brax-style: the write executes unconditionally, the value
    # written is ``old`` when ``do_reveal`` is False.
    traps = getattr(state, "traps", None)
    if traps is not None:
        max_lv_per_branch = jnp.int32(state.terrain.shape[1])
        b = state.dungeon.current_branch.astype(jnp.int32)
        lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        flat_lv = b * max_lv_per_branch + lv
        safe_lv = jnp.clip(flat_lv, 0, traps.trap_type.shape[0] - 1)
        safe_r = jnp.clip(mr, 0, traps.trap_type.shape[1] - 1)
        safe_c = jnp.clip(mc, 0, traps.trap_type.shape[2] - 1)
        trap_here = traps.trap_type[safe_lv, safe_r, safe_c] != jnp.int8(0)
        do_reveal = active & trap_here
        old_rev_cell = traps.revealed[safe_lv, safe_r, safe_c]
        new_rev_cell = jnp.where(do_reveal, jnp.bool_(True), old_rev_cell)
        new_rev = traps.revealed.at[safe_lv, safe_r, safe_c].set(new_rev_cell)
        new_traps = traps.replace(revealed=new_rev)
        state = state.replace(traps=new_traps)
        mai = state.monster_ai  # refresh local handle after replace

    # ---- last_seen_player_pos refresh (vendor newsym/cansee chain) ----
    # Brax pattern: compute the candidate ``ppos_i16`` unconditionally,
    # then select between it and ``old_seen`` via ``jnp.where``. The
    # ``.at[idx].set`` write is then unconditional with the masked value,
    # which is byte-identical to the original on inactive / out-of-LoS
    # slots (writes the existing value back).
    can_see_now = monster_can_see_player(state, idx) & active
    ppos_i16 = state.player_pos.astype(jnp.int16)
    old_seen = mai.last_seen_player_pos[idx]
    new_seen = jnp.where(can_see_now, ppos_i16, old_seen)
    new_mai = mai.replace(
        last_seen_player_pos=mai.last_seen_player_pos.at[idx].set(new_seen),
    )
    return state.replace(monster_ai=new_mai)
