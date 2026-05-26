"""Explosion subsystem — 3x3 AoE damage with per-tile resistance routing.

Canonical source:
    vendor/nethack/src/explode.c::explode (lines 199-696) — 3x3 AoE damage
    around a center tile; per-tile resistance routing via explosionmask
    (lines 26-115).  Vendor's full implementation also handles visual glyphs,
    floor zap_over_floor side-effects, golemeffects, item destruction, shop
    damage, and various corner cases (engulfer, grabbing, etc.).
    vendor/nethack/src/explode.c::scatter (lines 721-947) — dispersal of
    ground items in the 3x3 ring when an explosion / boulder-impact
    fractures the floor stack.  We model the simplified version specified
    by Nethax: each ground-stack slot inside the 3x3 ring has a 50%
    chance to be displaced into a random adjacent walkable tile.  Items
    that break on impact (potions — vendor breaks() classifies POTION_CLASS
    as fragile) are zeroed out rather than relocated.

Wave coverage (this module):
    - Iterates the 3x3 neighbourhood (8 neighbours + center) around
      ``center_pos``.
    - Rolls ``n_dice``d``n_sides`` once per call (vendor: damage is rolled
      once and applied to every target in the AoE — see explode.c line 207
      ``damu = dam`` and the per-tile loop at line 457).
    - Applies that damage to every alive monster at any of the 9 tiles.
    - Applies the same damage to the player if ``player_pos`` is inside the
      3x3 ring.
    - Per-target resistance routing:
        AD_FIRE  -> RESIST_FIRE   / MR_FIRE
        AD_COLD  -> RESIST_COLD   / MR_COLD
        AD_ELEC  -> RESIST_SHOCK  / MR_ELEC
        AD_ACID  -> RESIST_ACID   / MR_ACID
        AD_MAGIC -> MAGIC_RESIST  / (no monster MR bit; no resist for monsters)
      Resisting targets take ``(dam + 1) // 2`` damage (vendor explode.c
      line 538: ``mdam = (dam + 1) / 2``).
    - Disperses ground items in the 3x3 ring (see scatter helper below).

Out of scope (not in vendor lines 199-696 either, or deferred):
    - Visual / shield-effect glyph rendering.
    - Item destruction on the AoE itself (vendor zap_over_floor side-effects
      separate from scatter).
    - Killer/credit attribution, shop damage, gas-spore exposing.
    - Cross-resistance double-damage (resists_cold + AD_FIRE -> 2x dam).
    - inside_engulfer / grabber double-damage corner cases.
    - Weight-bounded multi-tile fling range (vendor scatter throws items
      ``rnd(blastforce - owt/40)`` tiles); Nethax uses a single-step move.

JIT-purity:
    - All control flow uses ``jnp.where`` and array masking — no Python
      conditionals on traced values.
    - The 3x3 monster/player loop is unrolled (9 fixed tiles) so the trace
      shape is constant.
    - Scatter walks 9*MAX_GROUND_STACK source slots through a single
      ``jax.lax.scan`` — fixed trip count, no dynamic branching.
"""
from __future__ import annotations
from enum import IntEnum

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import dice_roll
from Nethax.nethax.constants.monsters import (
    MR_FIRE, MR_COLD, MR_ELEC, MR_ACID,
)
from Nethax.nethax.subsystems.status_effects import Intrinsic
from Nethax.nethax.subsystems.inventory import (
    ItemCategory as _ItemCategory,
    MAX_GROUND_STACK as _MAX_GROUND_STACK,
)
from Nethax.nethax.constants.tiles import TileType as _TileType


class DamageType(IntEnum):
    """Subset of vendor AD_* (vendor/nethack/include/monattk.h).

    Only the explosion-relevant damage types are listed.  Values match
    vendor's enum so that callers can pass either the vendor constant or
    this enum interchangeably.
    """
    AD_PHYS  = 0
    AD_MAGIC = 1   # vendor AD_MAGM
    AD_FIRE  = 2
    AD_COLD  = 3
    AD_ELEC  = 4   # vendor AD_ELEC
    AD_ACID  = 5   # vendor AD_ACID


# Module-level aliases for ergonomic external use.
AD_PHYS  = int(DamageType.AD_PHYS)
AD_MAGIC = int(DamageType.AD_MAGIC)
AD_FIRE  = int(DamageType.AD_FIRE)
AD_COLD  = int(DamageType.AD_COLD)
AD_ELEC  = int(DamageType.AD_ELEC)
AD_ACID  = int(DamageType.AD_ACID)


# ---------------------------------------------------------------------------
# Resistance lookup tables — keyed by DamageType int.
# A value of 0 in the MR column means "no monster resist bit applies"
# (e.g. AD_MAGIC: vendor uses `resists_magm(m)` which is a flag predicate
# not represented in the resists_mask bitfield).  In that case the
# explode helper falls back to: monsters have no resistance, hero uses
# MAGIC_RESIST.  Vendor explode.c lines 78-99 handle this branching.
# ---------------------------------------------------------------------------

# Player-side intrinsic id per damage type (0 means "no intrinsic resist").
_PLAYER_RESIST_INTRINSIC = {
    AD_FIRE:  int(Intrinsic.RESIST_FIRE),
    AD_COLD:  int(Intrinsic.RESIST_COLD),
    AD_ELEC:  int(Intrinsic.RESIST_SHOCK),
    AD_ACID:  int(Intrinsic.RESIST_ACID),
    AD_MAGIC: int(Intrinsic.MAGIC_RESIST),
}

# Monster-side MR_* bit per damage type (0 means "no MR bit applies").
_MONSTER_RESIST_BIT = {
    AD_FIRE:  int(MR_FIRE),
    AD_COLD:  int(MR_COLD),
    AD_ELEC:  int(MR_ELEC),
    AD_ACID:  int(MR_ACID),
    AD_MAGIC: 0,
}


def _player_has_resist(state, dmg_type: int) -> jax.Array:
    """Return scalar bool: does the hero resist this damage type?

    Mirrors vendor explode.c explosionmask hero branch (lines 33-71):
    intrinsic + timed-intrinsic OR-merge per damage type.  Damage types
    not in the table (e.g. AD_PHYS) yield False — physical AoE has no
    resist source in vendor either.
    """
    intr_id = _PLAYER_RESIST_INTRINSIC.get(int(dmg_type), 0)
    if intr_id == 0:
        return jnp.bool_(False)
    perm  = state.status.intrinsics[intr_id]
    timed = state.status.timed_intrinsics[intr_id] > jnp.int32(0)
    return (perm | timed).astype(jnp.bool_)


def _monster_resist_mask(mai_resists: jax.Array, dmg_type: int) -> jax.Array:
    """Per-monster bool array: which monsters resist this damage type?

    Reads the per-slot ``resists`` bitmask (populated at spawn from
    MONSTERS[entry_idx].resists_mask — vendor/nethack/src/monst.c MON()).
    Damage types with no MR bit (AD_MAGIC, AD_PHYS) return all-False
    because no monsters bear a generic-magic-resist bit in that field;
    full vendor parity would consult resists_magm(m), which we defer.
    """
    bit = _MONSTER_RESIST_BIT.get(int(dmg_type), 0)
    if bit == 0:
        return jnp.zeros_like(mai_resists, dtype=jnp.bool_)
    return (mai_resists & jnp.int32(bit)) != jnp.int32(0)


# ---------------------------------------------------------------------------
# Scatter helper — vendor explode.c::scatter (lines 721-947).
#
# When an explosion strikes a tile, the ground stack on each of the 9 tiles
# in the 3x3 ring around the epicentre is rolled for displacement.  Vendor
# treats every object on the affected tile with a per-object random
# direction and a weight-bounded range; objects classified as fragile by
# breaks() (POTION_CLASS, plus eggs / glass) shatter on impact.
#
# Nethax simplification (per task spec):
#   * 50% chance per occupied ground-stack slot to scatter.
#   * Destination = random 8-direction adjacent tile (one step).
#   * If destination is out-of-bounds or non-walkable, item stays put.
#   * If item is a POTION, zero the slot out (it shatters in place rather
#     than landing).
# ---------------------------------------------------------------------------

# 8-direction offsets (vendor xdir/ydir, hack.h N_DIRS=8).
_SCATTER_OFFS = jnp.array(
    [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
    dtype=jnp.int32,
)

# 3x3 ring offsets (including center — vendor scatter is called once
# per explosion epicentre and operates on the ring of affected tiles).
_RING_OFFS = jnp.array(
    [[-1,-1],[-1,0],[-1,1],
     [ 0,-1],[ 0, 0],[ 0,1],
     [ 1,-1],[ 1, 0],[ 1,1]],
    dtype=jnp.int32,
)

# Walkable tile predicate — mirrors the conservative set used by
# features._tile_walkable.  Bad tiles bounce the scatter back to source.
_BAD_TILES_SCATTER = jnp.array(
    [
        int(_TileType.VOID),
        int(_TileType.WALL),
        int(_TileType.CLOSED_DOOR),
        int(_TileType.DRAWBRIDGE_UP),
        int(_TileType.WATER),
        int(_TileType.LAVA),
        int(_TileType.POOL),
    ],
    dtype=jnp.int32,
)


def _scatter_ground_items(state, rng: jax.Array, center_pos: jax.Array):
    """JIT-pure scatter pass.

    For each of the 9 ring tiles × MAX_GROUND_STACK source slots, with
    50% probability either (a) zero the slot if its item is a POTION
    (breaks on impact) or (b) move the item one step in a random
    8-direction to a walkable, in-bounds destination tile.

    Iteration order: ring tile 0 (top-left) through ring tile 8
    (bottom-right), inner loop over stack slots 0..MAX_GROUND_STACK-1.
    The scan carry is the updated ground_items pytree so each step sees
    the previous step's writes — preventing double-processing of the
    same item across consecutive iterations.

    Returns the (possibly updated) ``state``.
    """
    gi = state.ground_items
    # Per-source-slot loop body needs current_branch / current_level and
    # terrain to gate walkability.
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    terrain_level = state.terrain[b, lv]  # [H, W]
    map_h = jnp.int32(terrain_level.shape[0])
    map_w = jnp.int32(terrain_level.shape[1])

    center_r = center_pos[0].astype(jnp.int32)
    center_c = center_pos[1].astype(jnp.int32)

    # Source indices: 9 tiles × MAX_GROUND_STACK stack slots.
    n_ring   = int(_RING_OFFS.shape[0])
    stack_n  = int(_MAX_GROUND_STACK)
    n_total  = n_ring * stack_n

    # Pre-compute per-source random draws.  Each source slot needs its own
    # roll (50% scatter) and direction (1 of 8) — derived from independent
    # subkey splits so the two are uncorrelated.
    base_keys = jax.random.split(rng, n_total)
    roll_keys = jax.vmap(lambda k: jax.random.split(k, 2)[0])(base_keys)
    dir_keys  = jax.vmap(lambda k: jax.random.split(k, 2)[1])(base_keys)
    rolls = jax.vmap(lambda k: jax.random.randint(k, (), 0, 2, dtype=jnp.int32))(roll_keys)
    dirs  = jax.vmap(lambda k: jax.random.randint(k, (), 0, 8, dtype=jnp.int32))(dir_keys)

    # Source-flag mask: only the ORIGINAL 9*MAX_GROUND_STACK slots are
    # eligible.  We compute it from the INPUT gi (closed over) so that
    # downstream iterations of the scan can't promote a freshly-placed
    # item into another scatter source.  Vendor scatter() operates on
    # the level.objects[sx][sy] list it read at entry — same semantics.
    pre_cats = gi.category  # snapshot

    potion_cat = jnp.int8(int(_ItemCategory.POTION))
    none_cat   = jnp.int8(int(_ItemCategory.NONE))

    def _per_source(carry, i):
        gi = carry
        # Decode source ring tile + stack slot from flat index.
        ring_i  = (i // jnp.int32(stack_n)).astype(jnp.int32)
        slot_i  = (i %  jnp.int32(stack_n)).astype(jnp.int32)
        off     = _RING_OFFS[ring_i]
        sr      = center_r + off[0]
        sc      = center_c + off[1]

        # Bounds-check source tile.  Out-of-bounds → no-op (skip).
        in_bounds_src = (
            (sr >= jnp.int32(0)) & (sr < map_h)
            & (sc >= jnp.int32(0)) & (sc < map_w)
        )
        ssr = jnp.clip(sr, 0, map_h - 1)
        ssc = jnp.clip(sc, 0, map_w - 1)

        # Pre-scatter category at source (closed over → not affected by
        # earlier iterations that might have written into this slot from
        # a different ring tile).  Vendor reads level.objects[][] once.
        src_cat_pre = pre_cats[b, lv, ssr, ssc, slot_i]
        # Current category at source — needed to honour that an earlier
        # iteration may have already moved this exact item away.
        src_cat_now = gi.category[b, lv, ssr, ssc, slot_i]

        # Eligible: slot held an item BEFORE the scatter started AND that
        # item is still here (i.e. not consumed by an earlier scatter step).
        occupied = (
            in_bounds_src
            & (src_cat_pre != none_cat)
            & (src_cat_now == src_cat_pre)
        )
        is_potion = src_cat_pre == potion_cat

        # 50% chance to scatter.
        do_action = occupied & (rolls[i] == jnp.int32(0))

        # ---- Pick destination tile via 8-direction offset ----
        dir_off = _SCATTER_OFFS[dirs[i]]
        dr      = sr + dir_off[0]
        dc      = sc + dir_off[1]
        in_bounds_dst = (
            (dr >= jnp.int32(0)) & (dr < map_h)
            & (dc >= jnp.int32(0)) & (dc < map_w)
        )
        sdr = jnp.clip(dr, 0, map_h - 1)
        sdc = jnp.clip(dc, 0, map_w - 1)
        dst_typ = terrain_level[sdr, sdc].astype(jnp.int32)
        dst_walkable = in_bounds_dst & (~jnp.any(_BAD_TILES_SCATTER == dst_typ))

        # Find first empty stack-slot at destination (use the CURRENT gi
        # because earlier iterations may have already added items there).
        dst_stack_cats = gi.category[b, lv, sdr, sdc, :]
        dst_empty_mask = dst_stack_cats == none_cat
        dst_has_room = jnp.any(dst_empty_mask)
        dst_slot = jnp.argmax(dst_empty_mask.astype(jnp.int32)).astype(jnp.int32)

        # ---- Decide outcome ----
        # break_in_place : do_action & is_potion → just zero source.
        # move           : do_action & ~is_potion & dst_walkable & dst_has_room.
        # else           : no change.
        breaks_here = do_action & is_potion
        moves       = do_action & (~is_potion) & dst_walkable & dst_has_room

        # ---- Build per-field updates ----
        # Helper to update a single (array, fill) pair.  Source slot is
        # zeroed out on both break and move; destination is set only on move.
        def _update(arr, fill, src_val):
            cleared = jnp.where(
                breaks_here | moves,
                arr.at[b, lv, ssr, ssc, slot_i].set(fill),
                arr,
            )
            placed = jnp.where(
                moves,
                cleared.at[b, lv, sdr, sdc, dst_slot].set(src_val),
                cleared,
            )
            return placed

        # Read source values (using current gi, since item identity is
        # what matters here — even if the original cat snapshot drove the
        # eligibility gate, the field values come from the live array).
        src_typ_id      = gi.type_id[b, lv, ssr, ssc, slot_i]
        src_buc         = gi.buc_status[b, lv, ssr, ssc, slot_i]
        src_ench        = gi.enchantment[b, lv, ssr, ssc, slot_i]
        src_charges     = gi.charges[b, lv, ssr, ssc, slot_i]
        src_ident       = gi.identified[b, lv, ssr, ssc, slot_i]
        src_qty         = gi.quantity[b, lv, ssr, ssc, slot_i]
        src_weight      = gi.weight[b, lv, ssr, ssc, slot_i]
        src_ac          = gi.ac_bonus[b, lv, ssr, ssc, slot_i]
        src_is_2h       = gi.is_two_handed[b, lv, ssr, ssc, slot_i]
        src_greased     = gi.greased[b, lv, ssr, ssc, slot_i]
        src_oerod       = gi.oeroded[b, lv, ssr, ssc, slot_i]
        src_oerod2      = gi.oeroded2[b, lv, ssr, ssc, slot_i]
        src_oerproof    = gi.oerodeproof[b, lv, ssr, ssc, slot_i]
        src_bknown      = gi.bknown[b, lv, ssr, ssc, slot_i]
        src_lamplit     = gi.lamplit[b, lv, ssr, ssc, slot_i]
        src_olocked     = gi.olocked[b, lv, ssr, ssc, slot_i]
        src_corpse_idx  = gi.corpse_entry_idx[b, lv, ssr, ssc, slot_i]
        src_recharged   = gi.recharged[b, lv, ssr, ssc, slot_i]
        src_corpse_ct   = gi.corpse_creation_turn[b, lv, ssr, ssc, slot_i]
        src_tin_pois    = gi.tin_poisoned[b, lv, ssr, ssc, slot_i]
        src_dknown      = gi.dknown[b, lv, ssr, ssc, slot_i]
        src_rknown      = gi.rknown[b, lv, ssr, ssc, slot_i]
        src_age         = gi.age[b, lv, ssr, ssc, slot_i]
        src_artifact    = gi.artifact_idx[b, lv, ssr, ssc, slot_i]

        new_gi = gi.replace(
            category             = _update(gi.category,             none_cat,                src_cat_now),
            type_id              = _update(gi.type_id,              jnp.int16(0),            src_typ_id),
            buc_status           = _update(gi.buc_status,           jnp.int8(0),             src_buc),
            enchantment          = _update(gi.enchantment,          jnp.int8(0),             src_ench),
            charges              = _update(gi.charges,              jnp.int8(0),             src_charges),
            identified           = _update(gi.identified,           jnp.bool_(False),        src_ident),
            quantity             = _update(gi.quantity,             jnp.int16(0),            src_qty),
            weight               = _update(gi.weight,               jnp.int32(0),            src_weight),
            ac_bonus             = _update(gi.ac_bonus,             jnp.int8(0),             src_ac),
            is_two_handed        = _update(gi.is_two_handed,        jnp.bool_(False),        src_is_2h),
            greased              = _update(gi.greased,              jnp.bool_(False),        src_greased),
            oeroded              = _update(gi.oeroded,              jnp.int8(0),             src_oerod),
            oeroded2             = _update(gi.oeroded2,             jnp.int8(0),             src_oerod2),
            oerodeproof          = _update(gi.oerodeproof,          jnp.bool_(False),        src_oerproof),
            bknown               = _update(gi.bknown,               jnp.bool_(False),        src_bknown),
            lamplit              = _update(gi.lamplit,              jnp.bool_(False),        src_lamplit),
            olocked              = _update(gi.olocked,              jnp.bool_(False),        src_olocked),
            corpse_entry_idx     = _update(gi.corpse_entry_idx,     jnp.int16(-1),           src_corpse_idx),
            recharged            = _update(gi.recharged,            jnp.int8(0),             src_recharged),
            corpse_creation_turn = _update(gi.corpse_creation_turn, jnp.int32(-1),           src_corpse_ct),
            tin_poisoned         = _update(gi.tin_poisoned,         jnp.bool_(False),        src_tin_pois),
            dknown               = _update(gi.dknown,               jnp.bool_(False),        src_dknown),
            rknown               = _update(gi.rknown,               jnp.bool_(False),        src_rknown),
            age                  = _update(gi.age,                  jnp.int32(0),            src_age),
            artifact_idx         = _update(gi.artifact_idx,         jnp.int8(-1),            src_artifact),
        )
        return new_gi, None

    final_gi, _ = jax.lax.scan(_per_source, gi, jnp.arange(n_total, dtype=jnp.int32))
    return state.replace(ground_items=final_gi)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explode(
    state,
    rng: jax.Array,
    center_pos: jax.Array,
    dmg_type: int,
    n_dice: int,
    n_sides: int,
):
    """Apply a 3x3 AoE explosion centered on ``center_pos``.

    Parameters
    ----------
    state       : EnvState — must expose ``monster_ai`` (with ``pos``,
                  ``alive``, ``hp``, ``resists``) plus ``player_pos``,
                  ``player_hp``, and ``status``.
    rng         : JAX PRNG key (consumed).
    center_pos  : int16[2] (row, col) — explosion epicentre.
    dmg_type    : DamageType / AD_* int value.  Drives resistance routing.
    n_dice      : Python int — number of dice to roll for damage (static).
    n_sides     : Python int — sides per die (static).

    Returns
    -------
    new EnvState with updated ``monster_ai.hp`` / ``monster_ai.alive``,
    ``player_hp``, and ``ground_items`` (scattered).

    Notes
    -----
    Vendor explode.c rolls damage once (``dam`` argument is supplied by
    the caller and reused at every tile — see line 207 ``damu = dam`` and
    line 511+ ``int mdam = dam``).  We reproduce that single-roll
    semantics: one ``dice_roll`` for the whole AoE.

    Resisting targets receive ``(dam + 1) // 2`` damage (vendor line 538).

    After damage is applied, ``_scatter_ground_items`` runs on the 3x3
    ring to disperse / shatter ground items (vendor explode.c::scatter
    lines 721-947).
    """
    # Split the inbound key so the damage roll and scatter dispersal are
    # independent — vendor calls rnd() per-object inside scatter() and
    # per-blast for damage; we mirror that decoupling.
    rng_dmg, rng_scatter = jax.random.split(rng, 2)

    # ---- Roll damage once ------------------------------------------------
    dam_full = dice_roll(rng_dmg, int(n_dice), int(n_sides)).astype(jnp.int32)
    dam_half = ((dam_full + jnp.int32(1)) // jnp.int32(2)).astype(jnp.int32)

    center_r = center_pos[0].astype(jnp.int32)
    center_c = center_pos[1].astype(jnp.int32)

    # ---- Monsters --------------------------------------------------------
    mai = state.monster_ai
    mon_r = mai.pos[:, 0].astype(jnp.int32)
    mon_c = mai.pos[:, 1].astype(jnp.int32)

    # In the 3x3 ring (vendor: 8 neighbours + center) iff Chebyshev <= 1.
    dr = mon_r - center_r
    dc = mon_c - center_c
    cheby = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
    in_aoe = mai.alive & (cheby <= jnp.int32(1))

    mon_resists_mask = _monster_resist_mask(mai.resists, int(dmg_type))
    # Each monster gets dam_full or dam_half depending on resist.
    per_mon_dmg = jnp.where(mon_resists_mask, dam_half, dam_full)
    # Apply only to monsters in AoE.
    per_mon_dmg = jnp.where(in_aoe, per_mon_dmg, jnp.int32(0))

    new_hp    = jnp.maximum(mai.hp - per_mon_dmg, jnp.int32(0))
    new_alive = jnp.where(
        in_aoe & (new_hp <= jnp.int32(0)),
        jnp.bool_(False),
        mai.alive,
    )
    new_mai = mai.replace(hp=new_hp, alive=new_alive)

    # ---- Player ----------------------------------------------------------
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    pdr = pr - center_r
    pdc = pc - center_c
    pcheby = jnp.maximum(jnp.abs(pdr), jnp.abs(pdc))
    player_in_aoe = pcheby <= jnp.int32(1)
    player_resists = _player_has_resist(state, int(dmg_type))
    player_dmg_raw = jnp.where(player_resists, dam_half, dam_full)
    player_dmg = jnp.where(player_in_aoe, player_dmg_raw, jnp.int32(0))
    # Vendor: u.uhp -= damu (no floor at 1; death possible).
    new_player_hp = (state.player_hp - player_dmg).astype(state.player_hp.dtype)

    # ---- Scatter ground items in the 3x3 ring -----------------------------
    # Vendor explode.c invokes scatter() implicitly through the blast on
    # affected tiles; potions / fragile glass shatter on impact and the
    # remaining objects are flung 1..N tiles by weight.  We model the
    # simplified per-slot 50% displacement specified in the task brief.
    scattered = state.replace(monster_ai=new_mai, player_hp=new_player_hp)
    return _scatter_ground_items(scattered, rng_scatter, center_pos)
