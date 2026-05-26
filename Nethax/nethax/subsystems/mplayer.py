"""Monster-player (mplayer) NPC spawning.

Vendor: vendor/nethack/src/mplayer.c
  - mk_mplayer            (lines 118-326): spawn one NPC of a chosen role.
  - create_mplayers       (lines 327-355): spawn ``num`` random NPCs.

Vendor's mplayer machinery is the engine that drops Archeologist, Barbarian,
Caveman, Healer, Knight, Monk, Priest, Ranger, Rogue, Samurai, Tourist,
Valkyrie, and Wizard NPCs onto the Astral Plane (and Plane of Earth) for the
ascension run-in.  Without it, the 13 player-class MONSTERS entries in
``Nethax/nethax/constants/monsters.py`` (indices 327..341) are unreachable.

This module supplies a minimal port:

  * ``MPLAYER_ROLES``  — the 13 (entry_idx, role_name) pairs.
  * ``mk_mplayer``     — spawn one NPC in the first dead slot at ``pos``.
  * ``create_mplayers``— spawn ``n`` random NPCs around the player.

The hp formula uses vendor's ``rn1(16, 15)`` (i.e. 15..30) since this
minimal port skips the ``d(m_lev, 10) + (special ? rn1(30, 30) : 30)``
weapon/armor/luckstone drops; hp falls in the lower half of the vendor
range to keep the encounter near vanilla difficulty.

Note: the 13-role pool intentionally uses ``caveman`` (329) and ``priest``
(334) — the Nethax MONSTERS table splits NLE's ``NAMS(caveman, cavewoman)``
and ``NAMS(priest, priestess)`` into two entries each.  Vendor's
``PM_CAVE_DWELLER`` and ``PM_CLERIC`` map to the male variants here.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import rn1, rn2
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL


# ---------------------------------------------------------------------------
# 13 player-class NPC pool.
# Entry indices verified against Nethax/nethax/constants/monsters.py MONSTERS
# (chunk6 starts at entry 322 — see constants/monster_entries/chunk5.py
# trailing "# 321 — Famine (Rider)" sentinel).
# ---------------------------------------------------------------------------
MPLAYER_ROLES: tuple[tuple[int, str], ...] = (
    (327, "archeologist"),
    (328, "barbarian"),
    (329, "caveman"),
    (331, "healer"),
    (332, "knight"),
    (333, "monk"),
    (334, "priest"),
    (336, "ranger"),
    (337, "rogue"),
    (338, "samurai"),
    (339, "tourist"),
    (340, "valkyrie"),
    (341, "wizard"),
)


# ---------------------------------------------------------------------------
# Per-role alignment table (mirrors MONSTERS[idx].alignment for each role,
# chunk6 values).  Static tuple, parallel to MPLAYER_ROLES order.
#   archeologist:  3   barbarian: 0  caveman:  1
#   healer:        0   knight:    3  monk:     0
#   priest:        0   ranger:   -3  rogue:   -3
#   samurai:       3   tourist:   0  valkyrie:-1  wizard: 0
# ---------------------------------------------------------------------------
_ROLE_ALIGNMENTS: tuple[int, ...] = (
    3,  0,  1,
    0,  3,  0,
    0, -3, -3,
    3,  0, -1,  0,
)


# Pre-built JAX arrays indexed by [0..12].
_MPLAYER_ENTRIES = jnp.array(
    [pair[0] for pair in MPLAYER_ROLES], dtype=jnp.int16
)
_MPLAYER_ALIGNS = jnp.array(_ROLE_ALIGNMENTS, dtype=jnp.int8)


# ---------------------------------------------------------------------------
# mk_mplayer — spawn one NPC into the first dead monster_ai slot.
#
# Vendor mk_mplayer (mplayer.c lines 118-326) chooses ``m_lev = rn1(16, 15)``
# (i.e. 15..30) and ``mhp = mhpmax = d(m_lev, 10) + (30 + rnd(30))`` for the
# Astral-Plane special case, plus a class-specific weapon/armor/jewel drop.
# This minimal port skips the loot path and uses ``HP = rn1(16, 15)`` (15..30)
# directly as both hp and hp_max, with m_lev matched.
# ---------------------------------------------------------------------------
def mk_mplayer(state, rng: jax.Array, role_idx: int, pos: jnp.ndarray):
    """Spawn one player-class NPC at ``pos``.

    Parameters
    ----------
    state    : EnvState.
    rng      : JAX PRNG key (consumed for the hp roll).
    role_idx : int in [0, 13) — index into MPLAYER_ROLES.
    pos      : int-array (row, col) where the NPC should be placed.

    Returns
    -------
    state with ``monster_ai`` updated at the first dead slot.  No-op if no
    dead slot is available (the level is already full).

    JIT-compatible: uses ``jnp.where`` masking; no Python branches on traced
    values.  ``role_idx`` and ``pos`` are JAX-tracable.
    """
    mai = state.monster_ai

    # Allocate first dead slot (same pattern as monster_ai _try_wand_create).
    dead_mask = ~mai.alive
    has_dead = jnp.any(dead_mask)
    dead_idx = jnp.argmax(dead_mask.astype(jnp.int32)).astype(jnp.int32)

    # Resolve role -> entry_idx.  Alignment is sourced from MONSTERS table
    # (chunk6 alignment field, see _ROLE_ALIGNMENTS for the static map);
    # MonsterAIState has no per-slot alignment field, so vendor's
    # ``set_malign`` is implicitly satisfied by ``entry_idx`` lookup at
    # combat resolution time.
    r = jnp.asarray(role_idx, dtype=jnp.int32)
    entry = _MPLAYER_ENTRIES[r].astype(jnp.int16)

    # HP = rn1(16, 15)  → range [15, 30].  m_lev tracks hp (vendor parity).
    hp = rn1(rng, 16, 15)
    m_lev = hp.astype(mai.m_lev.dtype)

    # Position (row, col) — int16.
    pos_i16 = pos.astype(jnp.int16)

    should = has_dead

    new_alive   = mai.alive.at[dead_idx].set(
        jnp.where(should, jnp.bool_(True), mai.alive[dead_idx]))
    new_entry   = mai.entry_idx.at[dead_idx].set(
        jnp.where(should, entry, mai.entry_idx[dead_idx]))
    new_hp      = mai.hp.at[dead_idx].set(
        jnp.where(should, hp.astype(mai.hp.dtype), mai.hp[dead_idx]))
    new_hp_max  = mai.hp_max.at[dead_idx].set(
        jnp.where(should, hp.astype(mai.hp_max.dtype), mai.hp_max[dead_idx]))
    new_pos     = mai.pos.at[dead_idx].set(
        jnp.where(should, pos_i16, mai.pos[dead_idx]))
    new_m_lev   = mai.m_lev.at[dead_idx].set(
        jnp.where(should, m_lev, mai.m_lev[dead_idx]))
    # Hostile by default — vendor sets ``mtmp->mpeaceful = 0`` then
    # ``set_malign`` (mplayer.c lines 146-147).
    new_peace   = mai.peaceful.at[dead_idx].set(
        jnp.where(should, jnp.bool_(False), mai.peaceful[dead_idx]))
    new_tame    = mai.tame.at[dead_idx].set(
        jnp.where(should, jnp.bool_(False), mai.tame[dead_idx]))

    new_mai = mai.replace(
        alive=new_alive,
        entry_idx=new_entry,
        hp=new_hp,
        hp_max=new_hp_max,
        pos=new_pos,
        m_lev=new_m_lev,
        peaceful=new_peace,
        tame=new_tame,
    )
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# create_mplayers — spawn ``n`` random NPCs near the player.
#
# Vendor create_mplayers (mplayer.c lines 327-355) loops ``num`` times,
# picking ``pm = rn1(PM_WIZARD - PM_ARCHEOLOGIST + 1, PM_ARCHEOLOGIST)`` and
# a random goodpos() tile.  This port walks a Chebyshev ring around the
# player, looking for empty positions in spawn-priority order.  ``n`` must
# be a Python int (static) since lax.fori_loop needs a static trip count.
# ---------------------------------------------------------------------------
# Up to 24 ring offsets covering Chebyshev radius 1..2.  Pre-built static
# table: ring-1 (8 cells), then ring-2 (16 cells).
_RING_OFFSETS = jnp.array(
    [
        # Chebyshev radius 1
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
        # Chebyshev radius 2
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
        (-1, -2),                              (-1, 2),
        ( 0, -2),                              ( 0, 2),
        ( 1, -2),                              ( 1, 2),
        ( 2, -2), ( 2, -1), ( 2, 0), ( 2, 1), ( 2, 2),
    ],
    dtype=jnp.int32,
)


def _find_open_pos(mai, player_pos, used_mask):
    """Return (found, pos_i16, slot_idx) of the first free ring tile.

    ``used_mask`` is a bool[24] vector flagging ring slots already consumed
    by earlier mplayers within the same ``create_mplayers`` call.
    """
    n_ring = _RING_OFFSETS.shape[0]
    pr = player_pos[0].astype(jnp.int32)
    pc = player_pos[1].astype(jnp.int32)

    def _check(i, carry):
        found, best_pos, best_slot = carry
        dr = _RING_OFFSETS[i, 0]
        dc = _RING_OFFSETS[i, 1]
        ty = pr + dr
        tx = pc + dc
        occupied = jnp.any(
            mai.alive
            & (mai.pos[:, 0].astype(jnp.int32) == ty)
            & (mai.pos[:, 1].astype(jnp.int32) == tx)
        )
        slot_used = used_mask[i]
        take = (~found) & (~occupied) & (~slot_used)
        new_pos = jnp.where(
            take,
            jnp.stack([ty.astype(jnp.int16), tx.astype(jnp.int16)]),
            best_pos,
        )
        new_slot = jnp.where(take, jnp.int32(i), best_slot)
        new_found = found | take
        return new_found, new_pos, new_slot

    init = (jnp.bool_(False),
            jnp.stack([pr.astype(jnp.int16), pc.astype(jnp.int16)]),
            jnp.int32(-1))
    return jax.lax.fori_loop(0, n_ring, _check, init)


def create_mplayers(state, rng: jax.Array, n: int):
    """Spawn ``n`` random player-class NPCs around the player.

    Parameters
    ----------
    state : EnvState.
    rng   : JAX PRNG key (consumed for role + hp rolls).
    n     : Python int, static — number of NPCs to spawn (typical: 3 on
            Astral entry, mirroring vendor astral.lua's mplayer count).

    Returns
    -------
    state with up to ``n`` new entries in ``state.monster_ai``.  Silently
    skips any slot for which no dead monster_ai slot or ring tile is
    available (vendor's ``goodpos`` failure path).

    JIT-compatible: ``n`` must be a Python int so the unroll is static.
    """
    if not isinstance(n, int):
        # JIT path requires static n.  Caller-side guard.
        n = int(n)

    n_roles = len(MPLAYER_ROLES)
    n_ring = _RING_OFFSETS.shape[0]

    # Pre-split rng into 2*n sub-keys: (role-key, hp-key) per spawn.
    keys = jax.random.split(rng, max(2 * n, 1))

    def _spawn_one(i, carry):
        st, used = carry
        role = rn2(keys[2 * i], n_roles)
        # Find an open ring tile; mark it used so the next spawn picks
        # a different one.
        found, pos, slot = _find_open_pos(st.monster_ai, st.player_pos, used)
        # Clamp slot into a valid index range; when ``found`` is False the
        # write is masked out via the ``where`` below, so a clamped value
        # never affects state.
        safe_slot = jnp.clip(slot, 0, n_ring - 1).astype(jnp.int32)
        new_used = used.at[safe_slot].set(
            jnp.where(found, jnp.bool_(True), used[safe_slot])
        )
        # mk_mplayer no-ops on full level; an unfound pos falls back to
        # the player's tile (still safe — mk_mplayer will overlap, vendor
        # ``rloc`` would resolve in a fuller port).
        st2 = jax.lax.cond(
            found,
            lambda s: mk_mplayer(s, keys[2 * i + 1], role, pos),
            lambda s: s,
            st,
        )
        return st2, new_used

    init_used = jnp.zeros(n_ring, dtype=jnp.bool_)
    final_state, _ = jax.lax.fori_loop(0, n, _spawn_one, (state, init_used))
    return final_state


# ---------------------------------------------------------------------------
# Astral-Plane trigger.  Called from env._step_impl on level entry.
#
# Cite: vendor/nethack/src/mklev.c::makelev (Astral case) + astral.lua MAP
# section spawns 3 mplayers as part of the ascension gauntlet.  Branch.ENDGAME
# level 5 is the Astral Plane in our endgame_levels.py mapping.
# ---------------------------------------------------------------------------
ASTRAL_BRANCH: int = 6   # Branch.ENDGAME
ASTRAL_LEVEL:  int = 5   # 1=Earth, 2=Air, 3=Fire, 4=Water, 5=Astral
ASTRAL_MPLAYER_COUNT: int = 3


def maybe_seed_astral_mplayers(state, rng: jax.Array, prev_branch, prev_level):
    """If the player just entered (ENDGAME, 5), spawn 3 mplayers.

    Idempotent across step-loops since the trigger fires on transition
    (prev != curr).  Caller is responsible for passing the *previous*
    branch / level so the predicate stays edge-triggered.

    JIT-pure: uses jax.lax.cond.
    """
    curr_branch = state.dungeon.current_branch.astype(jnp.int32)
    curr_level  = state.dungeon.current_level.astype(jnp.int32)
    prev_b      = jnp.asarray(prev_branch, dtype=jnp.int32)
    prev_l      = jnp.asarray(prev_level,  dtype=jnp.int32)

    on_astral_now  = (curr_branch == jnp.int32(ASTRAL_BRANCH)) \
                   & (curr_level  == jnp.int32(ASTRAL_LEVEL))
    was_not_astral = (prev_b != jnp.int32(ASTRAL_BRANCH)) \
                   | (prev_l != jnp.int32(ASTRAL_LEVEL))
    should_spawn   = on_astral_now & was_not_astral

    return jax.lax.cond(
        should_spawn,
        lambda s: create_mplayers(s, rng, ASTRAL_MPLAYER_COUNT),
        lambda s: s,
        state,
    )
