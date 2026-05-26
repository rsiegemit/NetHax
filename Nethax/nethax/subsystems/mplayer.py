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
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTER_INV,
    MAX_MONSTERS_PER_LEVEL,
)


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
# Per-role equipment table.  Vendor mk_mplayer (mplayer.c lines 159-313) gives
# each role a class-typical weapon + armor + occasional tool/potion.  This
# port uses a fixed deterministic kit per role rather than the vendor's
# stochastic ``rnd_class`` / ``rn2(N)`` branches: vendor builds a sample then
# optionally overrides via ``mk_mplayer_armor`` (BUC + enchantment + erosion),
# but the *categories* of items are deterministic for each role.
#
# Categories are ItemCategory values (subsystems/inventory.ItemCategory):
#   WEAPON = 2, ARMOR = 3, POTION = 8, TOOL = 6.
# type_ids are OBJECTS-table indices (constants/objects.py) except for POTION
# slots which use the PotionEffect enum (items_potions.PotionEffect —
# HEALING=10, WATER=25) so they merge with existing _find_inv_slot lookups
# in monster_ai.
#
# Cite: vendor/nethack/src/mplayer.c lines 159-249 (case PM_*).
# ---------------------------------------------------------------------------
_CAT_WEAPON = 2
_CAT_ARMOR  = 3
_CAT_TOOL   = 6
_CAT_POTION = 8

# OBJECTS-table indices (constants/objects.py).
_OBJ_ARROW              = 1
_OBJ_YA                 = 5
_OBJ_DAGGER             = 17
_OBJ_SCALPEL            = 22
_OBJ_BATTLE_AXE         = 28
_OBJ_SHORT_SWORD        = 29   # samurai's wakizashi
_OBJ_LONG_SWORD         = 37
_OBJ_TWO_HANDED_SWORD   = 38
_OBJ_KATANA             = 39
_OBJ_LANCE              = 46
_OBJ_MACE               = 56
_OBJ_CLUB               = 59
_OBJ_QUARTERSTAFF       = 61
_OBJ_BULLWHIP           = 64
_OBJ_BOW                = 65
_OBJ_YUMI               = 68
_OBJ_SLING              = 69
_OBJ_SPLINT_MAIL        = 103
_OBJ_DWARVISH_MITHRIL   = 105
_OBJ_STUDDED_LEATHER    = 110
_OBJ_RING_MAIL          = 111
_OBJ_LEATHER_ARMOR      = 113
_OBJ_LEATHER_JACKET     = 114
_OBJ_HAWAIIAN_SHIRT     = 115
_OBJ_ROBE               = 122
_OBJ_CLOAK_OF_MAGIC_RES = 127
_OBJ_SMALL_SHIELD       = 129
_OBJ_LOCK_PICK          = 197
_OBJ_EXPENSIVE_CAMERA   = 204
_OBJ_TINNING_KIT        = 213
_OBJ_MAGIC_MARKER       = 217
_OBJ_PICK_AXE           = 234
_OBJ_UNICORN_HORN       = 236

# PotionEffect enum values (items_potions.PotionEffect).
_POT_HEALING = 10
_POT_WATER   = 25  # holy water variant per vendor priest gear

# Each row is up to MAX_MONSTER_INV (cat, type_id) pairs.  Empty trailing
# slots have cat == 0 (ItemCategory.NONE).  Index parallel to MPLAYER_ROLES.
#   archeologist : bullwhip + pick-axe + leather jacket + tinning kit  (4)
#   barbarian    : two-handed sword + battle-axe + studded leather     (3)
#   caveman      : club + sling + small shield                          (3)
#   healer       : unicorn horn + scalpel + potion of healing           (3)
#   knight       : long sword + lance + ring mail + small shield        (4)
#   monk         : robe                                                 (1)
#   priest       : mace + small shield + holy water                     (3)
#   ranger       : bow + arrow stack + leather armor                    (3)
#   rogue        : dagger + leather armor + lock pick                   (3)
#   samurai      : katana + wakizashi + yumi + ya + splint mail         (5)
#   tourist      : magic marker + expensive camera + Hawaiian shirt     (3)
#   valkyrie     : long sword + small shield + dwarvish mithril-coat    (3)
#   wizard       : quarterstaff + cloak of magic resistance + heal pot  (3)
_ROLE_KITS: tuple[tuple[tuple[int, int], ...], ...] = (
    # 0 archeologist
    ((_CAT_WEAPON, _OBJ_BULLWHIP),
     (_CAT_WEAPON, _OBJ_PICK_AXE),
     (_CAT_ARMOR,  _OBJ_LEATHER_JACKET),
     (_CAT_TOOL,   _OBJ_TINNING_KIT)),
    # 1 barbarian
    ((_CAT_WEAPON, _OBJ_TWO_HANDED_SWORD),
     (_CAT_WEAPON, _OBJ_BATTLE_AXE),
     (_CAT_ARMOR,  _OBJ_STUDDED_LEATHER)),
    # 2 caveman
    ((_CAT_WEAPON, _OBJ_CLUB),
     (_CAT_WEAPON, _OBJ_SLING),
     (_CAT_ARMOR,  _OBJ_SMALL_SHIELD)),
    # 3 healer
    ((_CAT_WEAPON, _OBJ_UNICORN_HORN),
     (_CAT_WEAPON, _OBJ_SCALPEL),
     (_CAT_POTION, _POT_HEALING)),
    # 4 knight
    ((_CAT_WEAPON, _OBJ_LONG_SWORD),
     (_CAT_WEAPON, _OBJ_LANCE),
     (_CAT_ARMOR,  _OBJ_RING_MAIL),
     (_CAT_ARMOR,  _OBJ_SMALL_SHIELD)),
    # 5 monk (vendor: robe only; no weapon — martial-arts class)
    ((_CAT_ARMOR,  _OBJ_ROBE),),
    # 6 priest
    ((_CAT_WEAPON, _OBJ_MACE),
     (_CAT_ARMOR,  _OBJ_SMALL_SHIELD),
     (_CAT_POTION, _POT_WATER)),
    # 7 ranger
    ((_CAT_WEAPON, _OBJ_BOW),
     (_CAT_WEAPON, _OBJ_ARROW),
     (_CAT_ARMOR,  _OBJ_LEATHER_ARMOR)),
    # 8 rogue
    ((_CAT_WEAPON, _OBJ_DAGGER),
     (_CAT_ARMOR,  _OBJ_LEATHER_ARMOR),
     (_CAT_TOOL,   _OBJ_LOCK_PICK)),
    # 9 samurai
    ((_CAT_WEAPON, _OBJ_KATANA),
     (_CAT_WEAPON, _OBJ_SHORT_SWORD),
     (_CAT_WEAPON, _OBJ_YUMI),
     (_CAT_WEAPON, _OBJ_YA),
     (_CAT_ARMOR,  _OBJ_SPLINT_MAIL)),
    # 10 tourist
    ((_CAT_TOOL,   _OBJ_MAGIC_MARKER),
     (_CAT_TOOL,   _OBJ_EXPENSIVE_CAMERA),
     (_CAT_ARMOR,  _OBJ_HAWAIIAN_SHIRT)),
    # 11 valkyrie
    ((_CAT_WEAPON, _OBJ_LONG_SWORD),
     (_CAT_ARMOR,  _OBJ_SMALL_SHIELD),
     (_CAT_ARMOR,  _OBJ_DWARVISH_MITHRIL)),
    # 12 wizard
    ((_CAT_WEAPON, _OBJ_QUARTERSTAFF),
     (_CAT_ARMOR,  _OBJ_CLOAK_OF_MAGIC_RES),
     (_CAT_POTION, _POT_HEALING)),
)


def _build_kit_arrays() -> tuple:
    """Pack _ROLE_KITS into fixed [N_ROLES, MAX_MONSTER_INV] arrays.

    Returns (cats, type_ids, quantities) where:
        cats      : int8  [13, MAX_MONSTER_INV]  ItemCategory per slot
        type_ids  : int16 [13, MAX_MONSTER_INV]  OBJECTS index OR PotionEffect
        qty       : int16 [13, MAX_MONSTER_INV]  per-slot stack size
                    (5 for arrows / ya; 1 otherwise — matches vendor's
                    ``otmp->quan += rn2(8)`` for stackables, capped at the
                    deterministic mean.)
    Trailing unused slots are zero-filled (cat=0 ItemCategory.NONE).
    """
    n_roles = len(_ROLE_KITS)
    cats = [[0] * MAX_MONSTER_INV for _ in range(n_roles)]
    tids = [[0] * MAX_MONSTER_INV for _ in range(n_roles)]
    qtys = [[0] * MAX_MONSTER_INV for _ in range(n_roles)]
    for r, kit in enumerate(_ROLE_KITS):
        for s, (cat, tid) in enumerate(kit):
            cats[r][s] = cat
            tids[r][s] = tid
            # Stack size: arrow/ya = 5 (vendor mean of rn2(8) for projectiles).
            if cat == _CAT_WEAPON and tid in (_OBJ_ARROW, _OBJ_YA):
                qtys[r][s] = 5
            else:
                qtys[r][s] = 1
    return (
        jnp.array(cats, dtype=jnp.int8),
        jnp.array(tids, dtype=jnp.int16),
        jnp.array(qtys, dtype=jnp.int16),
    )


_KIT_CATS, _KIT_TIDS, _KIT_QTYS = _build_kit_arrays()


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

    # ---- Role-specific equipment (vendor mplayer.c lines 159-313).
    # Look up the deterministic kit (cat, type_id, quantity) for this role
    # and write it into the dead slot's inventory row.  When ``should`` is
    # False (level full), the new row equals the old row.
    kit_cats = _KIT_CATS[r]   # [MAX_MONSTER_INV] int8
    kit_tids = _KIT_TIDS[r]   # [MAX_MONSTER_INV] int16
    kit_qtys = _KIT_QTYS[r]   # [MAX_MONSTER_INV] int16

    old_cats = mai.inv_category[dead_idx]
    old_tids = mai.inv_type_id[dead_idx]
    old_qtys = mai.inv_quantity[dead_idx]
    write_cats = jnp.where(should, kit_cats, old_cats).astype(mai.inv_category.dtype)
    write_tids = jnp.where(should, kit_tids, old_tids).astype(mai.inv_type_id.dtype)
    write_qtys = jnp.where(should, kit_qtys, old_qtys).astype(mai.inv_quantity.dtype)

    new_invc = mai.inv_category.at[dead_idx].set(write_cats)
    new_invt = mai.inv_type_id.at[dead_idx].set(write_tids)
    new_invq = mai.inv_quantity.at[dead_idx].set(write_qtys)

    new_mai = mai.replace(
        alive=new_alive,
        entry_idx=new_entry,
        hp=new_hp,
        hp_max=new_hp_max,
        pos=new_pos,
        m_lev=new_m_lev,
        peaceful=new_peace,
        tame=new_tame,
        inv_category=new_invc,
        inv_type_id=new_invt,
        inv_quantity=new_invq,
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
