"""Monster special-action subsystem — Wave 7 active special attacks.

Canonical vendor sources:
    vendor/nethack/src/mhitu.c  — mhitu_AD_SITM, mhitu_AD_SGLD, mhitu_AD_SEDU,
                                   doseduce(), mattacku() AT_BREA dispatch,
                                   AD_WRAP grab / AD_DRST drown path.
    vendor/nethack/src/monmove.c — monster movement, rloc teleport pattern.
    vendor/nethack/src/makemon.c:1317 — PM_STALKER perminvis = TRUE.

Design:
    monster_special_action(state, slot, rng) -> EnvState
        Dispatched by _SPECIAL_ACTION_TYPE[entry_idx]; each action is a
        pure function that may:
          • modify player_hp / player_pw / player_gold / player_* stats
          • teleport the monster (update mai.pos[slot])
          • set player_in_water
        All branching through jax.lax.cond; no Python-level conditionals
        inside JIT-traced paths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Action-type enum (int8 values stored in _SPECIAL_ACTION_TYPE)
# ---------------------------------------------------------------------------
_ACT_NONE:    int = 0
_ACT_NYMPH:   int = 1   # AD_SITM steal + teleport (mhitu.c::could_seduce ~1972)
_ACT_LEPREC:  int = 2   # AD_SGLD gold-steal + teleport (mhitu.c::doseduce ~2269)
_ACT_SEDU:    int = 3   # AD_SEDU ability drain + Pw drain (mhitu.c::doseduce ~2182)
_ACT_LICH:    int = 4   # spell cast: force_bolt/paralyze/fireball (mhitu.c ~412)
_ACT_BREATH:  int = 5   # AT_BREA dragon breath (mhitu.c::mattacku ~873)
_ACT_KRAKEN:  int = 6   # AT_HUGS+AD_WRAP grab → drown (mhitu.c ~1053)
_ACT_SPIT:    int = 7   # AT_SPIT ranged venom/acid (mthrowu.c::spitmu :1268,
                        #   spitmm :1016 — acid/blinding venom projectile)
_ACT_GAZE:    int = 8   # AT_GAZE ranged stare effect (mhitu.c::gazemu :1668 —
                        #   AD_FIRE/AD_CONF/AD_STON/AD_STUN/AD_BLND)

# ---------------------------------------------------------------------------
# Monster index constants (from vendor/nethack/include/monsters.h; mirrors
# the comment-indexed entries across monster_entries/chunk*.py).
# ---------------------------------------------------------------------------
# Chunk1 (indices 0-64): leprechaun = 63
_IDX_LEPRECHAUN:    int = 63
# Chunk2 (indices 65-130): wood/water/mountain nymph = 71,72,73
_IDX_NYMPH_FIRST:   int = 71
_IDX_NYMPH_LAST:    int = 73
# Chunk3 (indices 131-190):
#   gray=147, silver=149, red=150, white=151, orange=152,
#   black=153, blue=154, green=155, yellow=156
#   lich=187, demilich=188, master lich=189, arch-lich=190
_IDX_GRAY_DRAGON:   int = 147
_IDX_SILVER_DRAGON: int = 149
_IDX_RED_DRAGON:    int = 150
_IDX_WHITE_DRAGON:  int = 151
_IDX_ORANGE_DRAGON: int = 152
_IDX_BLACK_DRAGON:  int = 153
_IDX_BLUE_DRAGON:   int = 154
_IDX_GREEN_DRAGON:  int = 155
_IDX_YELLOW_DRAGON: int = 156
_IDX_LICH_FIRST:    int = 187
_IDX_LICH_LAST:     int = 190
# Chunk5: succubus/incubus — resolved by name lookup against MONSTERS so the
# values stay correct if the local table reorders (mirrors
# polymorph._resolve_pm_indices pattern).
def _resolve_sedu_indices() -> tuple[int, int]:
    from Nethax.nethax.constants.monsters import MONSTERS
    succ = inc = -1
    for i, m in enumerate(MONSTERS):
        if m.name == "succubus" and succ == -1:
            succ = i
        elif m.name == "incubus" and inc == -1:
            inc = i
    return succ, inc


_IDX_SUCCUBUS, _IDX_INCUBUS = _resolve_sedu_indices()
# Chunk6 (indices 322+): kraken is entry #6 in chunk6, base=322 → 328
_IDX_KRAKEN:        int = 328

_NUMMONS: int = 381

# ---------------------------------------------------------------------------
# Breath element indices (for _DRAGON_BREATH_ELEMENT)
# ---------------------------------------------------------------------------
_ELEM_NONE:   int = 0
_ELEM_FIRE:   int = 1   # red dragon      AD_FIRE  (mhitu.c AT_BREA, mhitu.c::breamu)
_ELEM_ELEC:   int = 2   # blue dragon     AD_ELEC
_ELEM_COLD:   int = 3   # white dragon    AD_COLD
_ELEM_DISINT: int = 4   # black dragon    AD_DISN  (1d255, mhitu.c)
_ELEM_POISON: int = 5   # green dragon    AD_DRST  (poison gas)
_ELEM_ACID:   int = 6   # yellow dragon   AD_ACID
_ELEM_SLEEP:  int = 7   # orange dragon   AD_SLEE
_ELEM_MAGIC:  int = 8   # gray dragon     AD_MAGM (magic missile)
_ELEM_COLD2:  int = 3   # silver dragon   AD_COLD (same as white)

# Player resistance indices (status_effects.Intrinsic enum values).
# These gate breath damage as in vendor mhitu.c::breamu.
# Cite: status_effects.py lines 72-78.
_RES_FIRE:    int = 1   # Intrinsic.RESIST_FIRE
_RES_COLD:    int = 2   # Intrinsic.RESIST_COLD
_RES_SLEEP:   int = 3   # Intrinsic.RESIST_SLEEP
_RES_DISINT:  int = 4   # Intrinsic.RESIST_DISINT
_RES_ELEC:    int = 5   # Intrinsic.RESIST_SHOCK
_RES_POISON:  int = 6   # Intrinsic.RESIST_POISON
_RES_ACID:    int = 7   # Intrinsic.RESIST_ACID
_N_INTRINSICS: int = 69  # matches status_effects.N_INTRINSICS (prop.h LAST_PROP=68)

# ---------------------------------------------------------------------------
# Precomputed lookup tables (built once at module import, not inside JIT).
# ---------------------------------------------------------------------------

def _build_special_action_table() -> jnp.ndarray:
    """Return int8[_NUMMONS] mapping each entry_idx → _ACT_* value.

    Cite: vendor/nethack/include/monsters.h for monster indices;
          vendor/nethack/src/mhitu.c for action semantics.
    """
    tbl = [_ACT_NONE] * _NUMMONS
    # Nymphs (wood/water/mountain): AD_SITM steal+tele, mhitu.c ~1972
    for i in range(_IDX_NYMPH_FIRST, _IDX_NYMPH_LAST + 1):
        tbl[i] = _ACT_NYMPH
    # Wave 47h: also route animal-class thieves (monkey/ape/lemure) to the
    # same dispatch.  They share the AD_SITM attack but vendor adds a
    # cursed-worn-item gate (steal.c:457-489) — see _nymph_steal for the
    # curse check.  MONSTERS entry indices verified against chunk1/4.
    _IDX_LEMURE_HERE = 53
    _IDX_MONKEY_HERE = 240
    _IDX_APE_HERE    = 241
    tbl[_IDX_LEMURE_HERE] = _ACT_NYMPH
    tbl[_IDX_MONKEY_HERE] = _ACT_NYMPH
    tbl[_IDX_APE_HERE]    = _ACT_NYMPH
    # Leprechaun: AD_SGLD gold-steal+tele, mhitu.c ~2269
    tbl[_IDX_LEPRECHAUN] = _ACT_LEPREC
    # Succubus / incubus: AD_SEDU drain, mhitu.c::doseduce ~2182
    tbl[_IDX_SUCCUBUS] = _ACT_SEDU
    tbl[_IDX_INCUBUS]  = _ACT_SEDU
    # Liches (lich/demilich/master/arch): spell cast, mhitu.c ~412
    for i in range(_IDX_LICH_FIRST, _IDX_LICH_LAST + 1):
        tbl[i] = _ACT_LICH
    # Adult dragons (not babies): AT_BREA breath, mhitu.c::mattacku ~873
    for i in range(_IDX_GRAY_DRAGON, _IDX_YELLOW_DRAGON + 1):
        tbl[i] = _ACT_BREATH
    tbl[_IDX_SILVER_DRAGON] = _ACT_BREATH
    # Kraken: AT_HUGS AD_WRAP grab+drown, mhitu.c ~1053
    tbl[_IDX_KRAKEN] = _ACT_KRAKEN

    # AT_SPIT / AT_GAZE — populate from MONSTERS attack lists.
    # Lower-priority than the bespoke actions above, so the established
    # dragons/lich/etc. routes win when both apply (no current overlap).
    # Vendor cite: mhitu.c::mattacku case AT_SPIT line 878-882 and
    #              case AT_GAZE line 832-837.
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    at_spit = int(AttackType.AT_SPIT)
    at_gaze = int(AttackType.AT_GAZE)
    for i, m in enumerate(MONSTERS):
        if tbl[i] != _ACT_NONE:
            continue
        for atk in m.attacks:
            aatyp = int(atk[0])
            if aatyp == at_spit:
                tbl[i] = _ACT_SPIT
                break
            if aatyp == at_gaze:
                tbl[i] = _ACT_GAZE
                break
    return jnp.array(tbl, dtype=jnp.int8)


def _build_dragon_breath_element_table() -> jnp.ndarray:
    """Return int8[_NUMMONS] mapping each dragon entry_idx → _ELEM_* value.

    Cite: vendor/nethack/include/monsters.h AT_BREA attack entries;
          vendor/nethack/src/mhitu.c::breamu element dispatch.
    Colors → elements:
        Red=fire, Blue=lightning, White=cold, Black=disintegration,
        Green=poison, Yellow=acid, Orange=sleep,
        Gray=magic missile, Silver=cold.
    """
    tbl = [_ELEM_NONE] * _NUMMONS
    tbl[_IDX_RED_DRAGON]    = _ELEM_FIRE
    tbl[_IDX_BLUE_DRAGON]   = _ELEM_ELEC
    tbl[_IDX_WHITE_DRAGON]  = _ELEM_COLD
    tbl[_IDX_BLACK_DRAGON]  = _ELEM_DISINT
    tbl[_IDX_GREEN_DRAGON]  = _ELEM_POISON
    tbl[_IDX_YELLOW_DRAGON] = _ELEM_ACID
    tbl[_IDX_ORANGE_DRAGON] = _ELEM_SLEEP
    tbl[_IDX_GRAY_DRAGON]   = _ELEM_MAGIC
    tbl[_IDX_SILVER_DRAGON] = _ELEM_COLD2
    return jnp.array(tbl, dtype=jnp.int8)


def _build_dragon_breath_damn_table() -> jnp.ndarray:
    """Return int8[_NUMMONS] mapping each dragon entry_idx → mattk->damn.

    Per-dragon dice-count for breath damage roll.  Vendor passes
    ``mattk->damn`` to dobuzz (mthrowu.c::breamm line 1123); zhitu then
    uses ``d(nd, 6)`` for fire/cold/elec/poison/acid/magm.

    Cite: vendor/nethack/include/monsters.h AT_BREA damn values:
        gray=4 (magm), silver=4 (cold), red=6 (fire), white=4 (cold),
        orange=4 (sleep), black=1 (disint), blue=4 (elec),
        green=4 (poison), yellow=4 (acid).
    """
    tbl = [0] * _NUMMONS
    tbl[_IDX_RED_DRAGON]    = 6
    tbl[_IDX_BLUE_DRAGON]   = 4
    tbl[_IDX_WHITE_DRAGON]  = 4
    tbl[_IDX_BLACK_DRAGON]  = 1
    tbl[_IDX_GREEN_DRAGON]  = 4
    tbl[_IDX_YELLOW_DRAGON] = 4
    tbl[_IDX_ORANGE_DRAGON] = 4
    tbl[_IDX_GRAY_DRAGON]   = 4
    tbl[_IDX_SILVER_DRAGON] = 4
    return jnp.array(tbl, dtype=jnp.int8)


def _build_perminvis_table() -> jnp.ndarray:
    """Return bool[_NUMMONS]; True for monsters with permanent natural invisibility.

    Cite: vendor/nethack/src/makemon.c:1317 — 'if (mndx == PM_STALKER)
          mtmp->perminvis = TRUE;'
    Stalker index = 157.
    """
    tbl = [False] * _NUMMONS
    tbl[157] = True   # stalker — makemon.c:1317
    return jnp.array(tbl, dtype=jnp.bool_)


_SPECIAL_ACTION_TYPE:      jnp.ndarray = _build_special_action_table()
_DRAGON_BREATH_ELEMENT:    jnp.ndarray = _build_dragon_breath_element_table()
_DRAGON_BREATH_DAMN:       jnp.ndarray = _build_dragon_breath_damn_table()
_PERMINVIS_TABLE:           jnp.ndarray = _build_perminvis_table()


def _build_spit_attack_tables() -> tuple:
    """Return (damn[N], sides[N], adtyp[N]) for each monster's first AT_SPIT
    attack.  All zeros for non-spitters.

    Cite: vendor/nethack/src/mthrowu.c::spitmm:1016 — AD_ACID/AD_BLND/AD_DRST
    venoms; damage is rolled from the venom object on hit (m_throw → hitmu).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    at_spit = int(AttackType.AT_SPIT)
    damn = [0] * _NUMMONS
    sides = [0] * _NUMMONS
    adtyp = [0] * _NUMMONS
    for i, m in enumerate(MONSTERS):
        for atk in m.attacks:
            if int(atk[0]) == at_spit:
                adtyp[i] = int(atk[1])
                damn[i]  = int(atk[2])
                sides[i] = int(atk[3])
                break
    return (
        jnp.array(damn,  dtype=jnp.int8),
        jnp.array(sides, dtype=jnp.int8),
        jnp.array(adtyp, dtype=jnp.int8),
    )


_SPIT_DAMN, _SPIT_SIDES, _SPIT_ADTYP = _build_spit_attack_tables()


def _build_gaze_attack_tables() -> tuple:
    """Return (damn[N], sides[N], adtyp[N]) for each monster's first AT_GAZE
    attack.  All zeros for non-gazers.

    Cite: vendor/nethack/src/mhitu.c::gazemu:1668 — AD_STON (Medusa, killed),
        AD_FIRE (fire ant), AD_CONF (forest centaur), AD_STUN (umber hulk),
        AD_BLND (floating eye via passive — but kept here for completeness).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    at_gaze = int(AttackType.AT_GAZE)
    damn = [0] * _NUMMONS
    sides = [0] * _NUMMONS
    adtyp = [0] * _NUMMONS
    for i, m in enumerate(MONSTERS):
        for atk in m.attacks:
            if int(atk[0]) == at_gaze:
                adtyp[i] = int(atk[1])
                damn[i]  = int(atk[2])
                sides[i] = int(atk[3])
                break
    return (
        jnp.array(damn,  dtype=jnp.int8),
        jnp.array(sides, dtype=jnp.int8),
        jnp.array(adtyp, dtype=jnp.int8),
    )


_GAZE_DAMN, _GAZE_SIDES, _GAZE_ADTYP = _build_gaze_attack_tables()

# AD_* values used in spit/gaze dispatch (mirror DamageType in
# Nethax/nethax/constants/monsters.py:117-165).
_AD_PHYS: int =  0
_AD_FIRE: int =  2
_AD_DRST: int =  7   # poison (drains strength)
_AD_ACID: int =  8
_AD_BLND: int = 11
_AD_STUN: int = 12
_AD_CONF: int = 25
_AD_STON: int = 18

# Range cap for spit/gaze: BOLT_LIM = 13 per vendor monst.c, but mthrowu uses
# m_lined_up (uses BOLT_LIM-1 effective).  Use the same range as breath.
_SPIT_RANGE: int = 8
_GAZE_RANGE: int = 8

# Resistance gate per breath element: index = _ELEM_* (0..8), value = Intrinsic
# index (or -1 = no resistance for that element).
# Order: NONE, FIRE, ELEC, COLD, DISINT, POISON, ACID, SLEEP, MAGIC.
# Mirrors mhitu.c::breamu resistance checks.
_BREATH_RES_IDX: jnp.ndarray = jnp.array(
    [-1, _RES_FIRE, _RES_ELEC, _RES_COLD, _RES_DISINT,
     _RES_POISON, _RES_ACID, _RES_SLEEP, -1],
    dtype=jnp.int32,
)

# Map-geometry constants (must match monster_ai._MAP_H/_MAP_W).
_MAP_H: int = 21
_MAP_W: int = 80

# Number of possible random teleport destinations (used as modulus).
_N_TELE_TILES: int = _MAP_H * _MAP_W


# ---------------------------------------------------------------------------
# Helper: Chebyshev distance (scalar)
# ---------------------------------------------------------------------------

def _cheby(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    d = jnp.abs(a.astype(jnp.int32) - b.astype(jnp.int32))
    return jnp.maximum(d[0], d[1])


# ---------------------------------------------------------------------------
# Helper: vendor-equivalent m_throw() ray-trace.
#
# Vendor cite: vendor/nethack/src/mthrowu.c::m_throw lines 572-849.  The
# vendor loop (lines 673-826) walks one tile per step along (dx, dy),
# advancing ``bhitpos`` BEFORE checking the destination tile.  At each
# step it checks whether the hero (u_at) is at that tile and invokes the
# hit-effect path; the beam also halts on terrain blockers (walls,
# closed doors, iron bars) and at end of range via MT_FLIGHTCHECK.
#
# This helper models the same step-by-step projectile behaviour:
#   - Advance by (dx, dy) each step.
#   - If the player is on the new tile, call ``hit_fn(state)`` and stop.
#   - If the tile is WALL or CLOSED_DOOR, stop without hitting.
#   - Otherwise continue until ``range`` is exhausted.
#
# JIT-pure: implemented via ``jax.lax.scan`` over a static ``range``.
# ---------------------------------------------------------------------------

def _m_throw_ray(
    state,
    src_pos: jnp.ndarray,
    dx: jnp.ndarray,
    dy: jnp.ndarray,
    range_: int,
    hit_fn,
):
    """Step-by-step projectile ray from ``src_pos`` along (dx, dy).

    Parameters
    ----------
    state    : EnvState
    src_pos  : int32[2] launcher tile (row, col); ray starts ONE step away.
    dx       : col-delta in {-1, 0, 1}  (vendor x-axis / sgn(gt.tbx))
    dy       : row-delta in {-1, 0, 1}  (vendor y-axis / sgn(gt.tby))
    range_   : maximum number of steps (Python int, static for jit).
    hit_fn   : callable(state) -> state, invoked AT MOST ONCE when the
               beam reaches the player tile.

    Returns
    -------
    (final_state, final_pos, hit_player) where ``final_pos`` is the last
    tile the projectile reached (int32[2]) and ``hit_player`` is a bool
    scalar.

    Vendor cite: vendor/nethack/src/mthrowu.c::m_throw lines 572-849
        — see while(range-- > 0) loop at line 673 and MT_FLIGHTCHECK
        blocker at line 799.  Wall / closed-door blocking matches
        IS_DOOR (closed) and IS_ROCK tile types in vendor rm.h.
    """
    from Nethax.nethax.constants.tiles import TileType

    src_pos_i32 = src_pos.astype(jnp.int32)
    dx_i32 = dx.astype(jnp.int32)
    dy_i32 = dy.astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)

    # Current branch/level for terrain lookup.
    br = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    terrain_lv = state.terrain[br, lv]  # int8[MAP_H, MAP_W]

    wall_t = jnp.int8(int(TileType.WALL))
    cdoor_t = jnp.int8(int(TileType.CLOSED_DOOR))

    def _step(carry, i):
        s, pos, stopped, hit = carry
        # Advance one tile (vendor m_throw: bhitpos += dx,dy BEFORE check).
        # i is 0-indexed step number.
        next_row = (src_pos_i32[0] + dy_i32 * (i + jnp.int32(1)))
        next_col = (src_pos_i32[1] + dx_i32 * (i + jnp.int32(1)))

        # Clip to map bounds; OOB also halts.
        h = jnp.int32(_MAP_H)
        w = jnp.int32(_MAP_W)
        oob = (next_row < jnp.int32(0)) | (next_row >= h) | \
              (next_col < jnp.int32(0)) | (next_col >= w)
        safe_row = jnp.clip(next_row, 0, h - 1)
        safe_col = jnp.clip(next_col, 0, w - 1)

        tile = terrain_lv[safe_row, safe_col]
        is_blocker = (tile == wall_t) | (tile == cdoor_t)

        # Does this tile contain the player?
        is_player_tile = (safe_row == ppos[0]) & (safe_col == ppos[1])

        # Apply hit_fn only when: not yet stopped, not OOB/blocker, on player.
        do_hit = (~stopped) & (~oob) & (~is_blocker) & is_player_tile

        s = jax.lax.cond(do_hit, hit_fn, lambda x: x, s)

        new_hit = hit | do_hit
        # Beam halts on: blocker, OOB, or after hitting the player.
        new_stopped = stopped | oob | is_blocker | do_hit
        new_pos = jnp.where(
            stopped,
            pos,
            jnp.stack([safe_row, safe_col]),
        )
        return (s, new_pos, new_stopped, new_hit), None

    init_pos = src_pos_i32
    init_carry = (state, init_pos, jnp.bool_(False), jnp.bool_(False))
    (final_state, final_pos, _stopped, hit_player), _ = jax.lax.scan(
        _step, init_carry, jnp.arange(range_, dtype=jnp.int32)
    )
    return final_state, final_pos, hit_player


def _clear_shop_bill_for_slot(state, steal_slot: jnp.ndarray, do_clear: jnp.ndarray):
    """Clear the shop bill row for ``steal_slot`` when the slot was unpaid.

    Vendor cite: ``steal.c::mpickobj`` lines 639-643 — ``if (otmp->unpaid ...)
    subfrombill(otmp, ...)``.  Mirrors ``subfrombill`` (shk.c) which removes a
    single bill_p entry and decrements the running total.

    JIT-pure.  ``do_clear`` gates the update so callers can pass a guarded
    branch (e.g. only clear when a steal actually happened).
    """
    shop = state.shop
    n = shop.items_owned_by_shop.shape[0]
    s_idx = jnp.clip(steal_slot.astype(jnp.int32), 0, n - 1)
    was_owned = shop.items_owned_by_shop[s_idx]
    effective = do_clear & was_owned
    slot_price = shop.bill_prices[s_idx]
    new_bill = jnp.where(
        effective,
        jnp.maximum(jnp.int32(0), shop.bill - slot_price),
        shop.bill,
    )
    new_owned = shop.items_owned_by_shop.at[s_idx].set(
        jnp.where(effective, jnp.bool_(False), was_owned)
    )
    new_prices = shop.bill_prices.at[s_idx].set(
        jnp.where(effective, jnp.int32(0), slot_price)
    )
    new_shop = shop.replace(
        bill=new_bill,
        items_owned_by_shop=new_owned,
        bill_prices=new_prices,
    )
    return state.replace(shop=new_shop)


# ---------------------------------------------------------------------------
# 1. Nymph steal-and-teleport  (vendor/nethack/src/mhitu.c::could_seduce ~1972,
#    mhitu_AD_SITM handling; rloc teleport via monmove.c pattern)
# ---------------------------------------------------------------------------

def _nymph_steal(state, slot: jnp.ndarray, rng: jax.Array):
    """Adjacent nymph steals a random inventory item, then teleports.

    Vendor cite: mhitu.c lines ~1972-1977 — nymph AD_SITM steals one
    item from hero's inventory; rloc() (monmove.c) teleports the nymph.
    Precondition gate: must be adjacent (Chebyshev dist == 1).
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    adjacent = _cheby(mpos, ppos) <= jnp.int32(1)

    def _apply(s):
        rng_item, rng_tele, rng_punish = jax.random.split(rng, 3)
        items = s.inventory.items
        # Pick a random occupied slot: sample index 0..N_SLOTS-1.
        n_slots = items.category.shape[0]
        raw = jax.random.randint(rng_item, (), 0, n_slots)
        # Rotate: find a slot with category != 0, starting from raw.
        slot_seq = jnp.mod(jnp.arange(n_slots) + raw, n_slots)
        has_item = items.category[slot_seq] != jnp.int8(0)
        pick = jnp.argmax(has_item)   # first occupied slot in rotation
        picked_slot = slot_seq[pick]
        any_item = jnp.any(has_item)

        # Wave 47h: cursed-worn gate (vendor steal.c:457-489).  Animals
        # (monkey/ape/lemure) fail to steal cursed worn items.  Nymphs
        # break the curse and proceed.
        is_animal_thief = (
            (mai.entry_idx[idx].astype(jnp.int32) == jnp.int32(53))   # lemure
            | (mai.entry_idx[idx].astype(jnp.int32) == jnp.int32(240))  # monkey
            | (mai.entry_idx[idx].astype(jnp.int32) == jnp.int32(241))  # ape
        )

        # ---- Wave 47i Item-substitution theft (vendor steal.c:433-446) ----
        # "can't steal ring(s) while wearing gloves -> steal gloves instead"
        # "can't steal shirt while wearing cloak or suit -> steal cloak/suit"
        # Animals skip these substitutions (vendor monkey_business branch
        # at steal.c:403-404 short-circuits the ring/shirt substitution).
        from Nethax.nethax.subsystems.inventory import (
            ItemCategory as _IC, ArmorSlot as _AS,
        )
        wa_i32 = s.inventory.worn_armor.astype(jnp.int32)
        gloves_slot = wa_i32[int(_AS.GLOVES)]
        body_slot   = wa_i32[int(_AS.BODY)]
        has_gloves  = gloves_slot >= jnp.int32(0)
        has_body    = body_slot >= jnp.int32(0)
        picked_cat  = items.category[picked_slot].astype(jnp.int32)
        is_ring     = picked_cat == jnp.int32(int(_IC.RING))
        # SHIRT subtype: armor item in the SHIRT armor slot
        shirt_slot_idx = wa_i32[int(_AS.SHIRT)]
        is_shirt    = (picked_slot.astype(jnp.int32) == shirt_slot_idx) & (shirt_slot_idx >= 0)

        sub_to_gloves = (~is_animal_thief) & is_ring & has_gloves
        sub_to_body   = (~is_animal_thief) & is_shirt & has_body & (~sub_to_gloves)
        steal_slot = jnp.where(
            sub_to_gloves, gloves_slot.astype(picked_slot.dtype),
            jnp.where(
                sub_to_body, body_slot.astype(picked_slot.dtype), picked_slot
            ),
        )

        # "Worn" detection: scan worn_armor / wielded.  Conservative —
        # treat the picked slot as "worn" if it matches any worn pointer.
        wa = s.inventory.worn_armor.astype(jnp.int32)
        worn_match = jnp.any(wa == steal_slot.astype(jnp.int32))
        wielded_match = s.inventory.wielded.astype(jnp.int32) == steal_slot.astype(jnp.int32)
        is_worn = worn_match | wielded_match
        is_cursed_buc = items.buc_status[steal_slot].astype(jnp.int32) == jnp.int32(1)
        animal_blocked = is_animal_thief & is_worn & is_cursed_buc
        steal_allowed = any_item & ~animal_blocked

        def _do_steal(s2):
            old_items = s2.inventory.items
            # Nymph branch (or animal stealing non-worn/non-cursed): zero
            # out the stolen slot AND uncurse it (vendor 469-470).
            uncurse = is_worn & is_cursed_buc & ~is_animal_thief
            new_buc = jnp.where(uncurse, jnp.int8(2), old_items.buc_status[steal_slot])
            new_items = old_items.replace(
                category=old_items.category.at[steal_slot].set(jnp.int8(0)),
                quantity=old_items.quantity.at[steal_slot].set(jnp.int16(0)),
                buc_status=old_items.buc_status.at[steal_slot].set(new_buc),
            )
            new_inv = s2.inventory.replace(items=new_items)
            # If the stolen slot also pointed at a worn-armor pointer, clear
            # that pointer so subsequent AC / wear-state checks stay coherent.
            # Vendor steal.c:498-505 calls worn_item_removal which clears the
            # owornmask before extracting the obj.
            new_wa = new_inv.worn_armor
            stolen_i32 = steal_slot.astype(jnp.int32)
            clear_mask = new_wa.astype(jnp.int32) == stolen_i32
            new_wa = jnp.where(clear_mask, jnp.int8(-1), new_wa)
            new_inv = new_inv.replace(worn_armor=new_wa)
            s3 = s2.replace(inventory=new_inv)
            # Wave 47i Item #1 — clear shop bill row for unpaid stolen item
            # (vendor steal.c::mpickobj lines 639-643 → subfrombill).
            return _clear_shop_bill_for_slot(s3, steal_slot, jnp.bool_(True))

        # Wave 47i Item #4 — Punished ball-and-chain target (vendor steal.c:379-382)
        # When the hero has no inventory items to steal AND is Punished, a
        # non-animal thief has a 75% chance to remove the chain (vendor uses
        # rn2(4) which is non-zero 3/4 of the time).
        inv_empty   = ~any_item
        is_punished = s.is_punished
        punish_target = inv_empty & is_punished & (~is_animal_thief)
        # rn2(4) != 0  →  3/4 chance
        punish_roll = jax.random.randint(rng_punish, (), 0, 4)
        do_unpunish = punish_target & (punish_roll != jnp.int32(0))

        def _do_unpunish(s2):
            return s2.replace(is_punished=jnp.bool_(False))

        s = jax.lax.cond(steal_allowed, _do_steal, lambda s2: s2, s)
        s = jax.lax.cond(do_unpunish, _do_unpunish, lambda s2: s2, s)

        # Teleport nymph to random valid tile.
        raw_tele = jax.random.randint(rng_tele, (), 0, _N_TELE_TILES)
        tele_r = jnp.int16(raw_tele // _MAP_W)
        tele_c = jnp.int16(raw_tele % _MAP_W)
        new_pos = jnp.stack([tele_r, tele_c])
        new_mai = s.monster_ai.replace(
            pos=s.monster_ai.pos.at[idx].set(new_pos)
        )
        return s.replace(monster_ai=new_mai)

    return jax.lax.cond(adjacent, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 2. Leprechaun gold-steal + teleport  (vendor/nethack/src/mhitu.c::doseduce
#    ~2269-2302 — "tries to take your gold", money2mon, rloc)
# ---------------------------------------------------------------------------

_COIN_CATEGORY:   int = 12  # ItemCategory.COIN
_GOLD_PIECE_TID:  int = 410 # objects.py: gold piece


def _floor_gold_at(state, row: jnp.ndarray, col: jnp.ndarray):
    """Return (slot_idx, quantity) for the first COIN/gold ground stack at
    (br, lv, row, col).  slot_idx = -1 + quantity = 0 if no gold present.

    Vendor cite: steal.c:60 — ``fgold = g_at(u.ux, u.uy)`` then advances past
    lesser coins (steal.c:67-68).  Nethax only models gold pieces as COIN, so
    we just scan the 8-deep ground stack for category==COIN.
    """
    g = state.ground_items
    br = state.dungeon.current_branch.astype(jnp.int32)
    lv = (state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1))
    r = row.astype(jnp.int32)
    c = col.astype(jnp.int32)

    cats = g.category[br, lv, r, c, :].astype(jnp.int32)        # (MAX_GROUND_STACK,)
    tids = g.type_id[br, lv, r, c, :].astype(jnp.int32)
    qtys = g.quantity[br, lv, r, c, :].astype(jnp.int32)
    is_gold = (cats == jnp.int32(_COIN_CATEGORY)) & (tids == jnp.int32(_GOLD_PIECE_TID))
    any_gold = jnp.any(is_gold)
    first_idx = jnp.argmax(is_gold.astype(jnp.int32))
    quantity = jnp.where(any_gold, qtys[first_idx], jnp.int32(0))
    safe_idx = jnp.where(any_gold, first_idx, jnp.int32(-1))
    return safe_idx, quantity


def _leprechaun_steal_gold(state, slot: jnp.ndarray, rng: jax.Array):
    """Adjacent leprechaun grabs gold (floor priority, then inventory), then
    teleports.

    Vendor cite: steal.c::stealgold lines 57-115.
      1. ``fgold = g_at(u.ux, u.uy)`` after skipping lesser coins (60-68).
      2. ``ygold = findgold(invent)`` (71).
      3. If ``fgold && (!ygold || fgold->quan > ygold->quan || !rn2(5))``,
         take the whole floor gold pile (74-93); teleport when
         ``!ygold || !rn2(5)`` (94-98).
      4. Else if ``ygold``, take ``somegold(money_cnt(invent))`` from
         inventory (99-115) and always teleport.

    Amount formula (steal.c::somegold lines 13-34) is byte-equal:
        igold < 50    : steal all
        igold < 100   : rn1(igold-25+1, 25)   == uniform[25, igold]
        igold < 500   : rn1(igold-50+1, 50)   == uniform[50, igold]
        igold < 1000  : rn1(igold-100+1, 100) == uniform[100, igold]
        igold < 5000  : rn1(igold-500+1, 500) == uniform[500, igold]
        igold < 10000 : rn1(igold-1000+1, 1000) == uniform[1000, igold]
        else          : rn1(igold-5000+1, 5000) == uniform[5000, igold]
    where rn1(n, x) == x + rn2(n).
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    adjacent = _cheby(mpos, ppos) <= jnp.int32(1)

    def _apply(s):
        rng_amt, rng_tele, rng_pref, rng_tele2 = jax.random.split(rng, 4)
        gold = s.player_gold.astype(jnp.int32)

        # Vendor steal.c:60-68 — floor gold at player tile.
        pr = s.player_pos[0]
        pc = s.player_pos[1]
        fgold_slot, fgold_qty = _floor_gold_at(s, pr, pc)
        has_fgold = fgold_qty > jnp.int32(0)
        has_ygold = gold > jnp.int32(0)

        # steal.c:73 — `fgold && (!ygold || fgold->quan > ygold->quan || !rn2(5))`
        # !rn2(5) == 1/5 chance preferring floor over inventory.
        prefer_floor_roll = jax.random.randint(
            rng_pref, (), 0, 5, dtype=jnp.int32
        ) == jnp.int32(0)
        take_floor = has_fgold & (
            (~has_ygold)
            | (fgold_qty > gold)
            | prefer_floor_roll
        )
        # Vendor steal.c:14-34 somegold(): bracketed rn1 of gold.
        # rn1(n, x) = x + rn2(n) = uniform[x, x+n-1].
        bracket_n = jnp.where(
            gold < jnp.int32(50), jnp.int32(1),
            jnp.where(gold < jnp.int32(100), gold - jnp.int32(25) + jnp.int32(1),
            jnp.where(gold < jnp.int32(500), gold - jnp.int32(50) + jnp.int32(1),
            jnp.where(gold < jnp.int32(1000), gold - jnp.int32(100) + jnp.int32(1),
            jnp.where(gold < jnp.int32(5000), gold - jnp.int32(500) + jnp.int32(1),
            jnp.where(gold < jnp.int32(10000), gold - jnp.int32(1000) + jnp.int32(1),
                                                gold - jnp.int32(5000) + jnp.int32(1)))))),
        )
        bracket_x = jnp.where(
            gold < jnp.int32(50), jnp.int32(0),
            jnp.where(gold < jnp.int32(100), jnp.int32(25),
            jnp.where(gold < jnp.int32(500), jnp.int32(50),
            jnp.where(gold < jnp.int32(1000), jnp.int32(100),
            jnp.where(gold < jnp.int32(5000), jnp.int32(500),
            jnp.where(gold < jnp.int32(10000), jnp.int32(1000),
                                                jnp.int32(5000)))))),
        )
        safe_n = jnp.maximum(bracket_n, jnp.int32(1))
        rn2_roll = jax.random.randint(rng_amt, (), 0, safe_n, dtype=jnp.int32)
        rn1_result = (bracket_x + rn2_roll).astype(jnp.int32)
        # Below 50 gold, vendor returns igold unchanged (steal all).
        stolen_inv = jnp.where(gold < jnp.int32(50), gold, rn1_result)
        # Vendor steal.c:103 caps at ygold->quan (player's gold quantity).
        stolen_inv = jnp.minimum(stolen_inv, gold)
        new_gold_inv = jnp.maximum(gold - stolen_inv, jnp.int32(0)).astype(jnp.int32)

        # Floor-gold branch: take the whole pile (vendor steal.c:74
        # ``obj_extract_self(fgold); add_to_minv(mtmp, fgold);``).
        new_player_gold = jnp.where(take_floor, gold, new_gold_inv)
        # Clear floor gold slot when taken.  safe_idx is valid only when
        # has_fgold is True.
        clear_slot = jnp.where(take_floor, fgold_slot, jnp.int32(0))
        br = s.dungeon.current_branch.astype(jnp.int32)
        lv = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        g = s.ground_items
        zero_cat = jnp.where(take_floor, jnp.int8(0), g.category[br, lv, pr, pc, clear_slot])
        zero_qty = jnp.where(take_floor, jnp.int16(0), g.quantity[br, lv, pr, pc, clear_slot])
        new_ground = g.replace(
            category=g.category.at[br, lv, pr, pc, clear_slot].set(zero_cat),
            quantity=g.quantity.at[br, lv, pr, pc, clear_slot].set(zero_qty),
        )

        # Teleport gating (vendor steal.c:94-97 floor path; :111-113 inv path):
        #   floor: `if (!ygold || !rn2(5)) rloc(); monflee();` — tele only on
        #          1/5 roll OR when inv is empty.
        #   inv:   always rloc + monflee.
        tele_roll = jax.random.randint(rng_tele2, (), 0, 5, dtype=jnp.int32) == jnp.int32(0)
        floor_tele = (~has_ygold) | tele_roll
        do_teleport = jnp.where(take_floor, floor_tele, jnp.bool_(True))

        raw_tele = jax.random.randint(rng_tele, (), 0, _N_TELE_TILES)
        tele_r = jnp.int16(raw_tele // _MAP_W)
        tele_c = jnp.int16(raw_tele % _MAP_W)
        new_mon_pos = jnp.where(
            do_teleport,
            jnp.stack([tele_r, tele_c]),
            s.monster_ai.pos[idx],
        )
        new_mai = s.monster_ai.replace(
            pos=s.monster_ai.pos.at[idx].set(new_mon_pos)
        )
        return s.replace(
            player_gold=new_player_gold,
            monster_ai=new_mai,
            ground_items=new_ground,
        )

    return jax.lax.cond(adjacent, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 3. Succubus/incubus seduce drain  (vendor/nethack/src/mhitu.c::doseduce
#    ~2182-2223 — rn2(5) selects drain outcome; u.uen=0 or adjattrib -1)
# ---------------------------------------------------------------------------

def _succubus_drain(state, slot: jnp.ndarray, rng: jax.Array):
    """Adjacent succubus/incubus seduces player: -1 to a random ability, drains Pw.

    Vendor cite: mhitu.c doseduce() lines ~2182-2223 — on bad seduction
    outcome, switch(rn2(5)):
        case 0: u.uen=0 (Pw drained)
        case 1: adjattrib(A_CON, -1)
        case 2: adjattrib(A_WIS, -1)
        case 3: losexp()
        case 4: losehp()
    Vendor-parity dispatch on rn2(5) with the exact 5 branches above.
    Adjacent check required.
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    adjacent = _cheby(mpos, ppos) <= jnp.int32(1)

    def _apply(s):
        rng_which, _ = jax.random.split(rng)
        which = jax.random.randint(rng_which, (), 0, 5)  # rn2(5) ∈ {0..4}

        is_pw   = which == jnp.int32(0)  # u.uen=0
        is_con  = which == jnp.int32(1)  # adjattrib(A_CON, -1)
        is_wis  = which == jnp.int32(2)  # adjattrib(A_WIS, -1)
        is_xp   = which == jnp.int32(3)  # losexp()  → -1 XL
        is_hp   = which == jnp.int32(4)  # losehp() → -d6 HP

        new_pw  = jnp.where(is_pw,  jnp.int32(0), s.player_pw)
        new_con = jnp.where(is_con, jnp.maximum(jnp.int8(3), s.player_con - jnp.int8(1)), s.player_con)
        new_wis = jnp.where(is_wis, jnp.maximum(jnp.int8(3), s.player_wis - jnp.int8(1)), s.player_wis)
        new_xl  = jnp.where(is_xp,  jnp.maximum(jnp.int32(1), s.player_xl - jnp.int32(1)), s.player_xl)
        hp_dmg  = jax.random.randint(rng_which, (), 1, 7, dtype=jnp.int32)  # d6
        new_hp  = jnp.where(is_hp, jnp.maximum(jnp.int32(0), s.player_hp - hp_dmg), s.player_hp)

        return s.replace(
            player_pw=new_pw,
            player_con=new_con,
            player_wis=new_wis,
            player_xl=new_xl,
            player_hp=new_hp,
        )

    return jax.lax.cond(adjacent, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 4. Lich spell cast  (vendor/nethack/src/mhitu.c ~412 — liches have a
#    touch attack for cold and a spell attack; castmu / mhitu AT_MAGC path)
# ---------------------------------------------------------------------------

# Spell damage base (n dice × sides):
_LICH_SPELL_DMG: jnp.ndarray = jnp.array(
    # force_bolt: 2d6, paralyze: 1 (timed), fireball: 6d6
    [[2, 6], [0, 0], [6, 6]],
    dtype=jnp.int32,
)
_LICH_SPELL_PARALYZE_TURNS: int = 4

# LoS range for lich spell (vendor: spells have infinite range if LoS exists;
# we cap at 10 tiles to keep the check cheap).
_LICH_SPELL_RANGE: int = 10


def _lich_cast(state, slot: jnp.ndarray, rng: jax.Array):
    """Lich casts a random spell at the player if in LoS range.

    Spells: 0=force_bolt (2d6 dmg), 1=paralyze (4 turns), 2=fireball (6d6 dmg).
    Vendor cite: mhitu.c ~412 — 'liches have a touch attack for cold damage
    and also a spell attack'; castmu / AT_MAGC dispatch at ~926-930.
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _cheby(mpos, ppos)
    in_range = dist <= jnp.int32(_LICH_SPELL_RANGE)

    def _apply(s):
        rng_spell, rng_roll = jax.random.split(rng)
        spell = jax.random.randint(rng_spell, (), 0, 3, dtype=jnp.int32)  # 0/1/2

        # Damage roll for force_bolt / fireball.
        n     = _LICH_SPELL_DMG[spell, 0]
        sides = _LICH_SPELL_DMG[spell, 1]
        rolls = jax.random.randint(rng_roll, (6,), 1, 7, dtype=jnp.int32)  # up to 6 dice
        mask  = jnp.arange(6) < n
        dmg   = jnp.sum(jnp.where(mask, jnp.minimum(rolls, jnp.int32(sides)), jnp.int32(0))).astype(jnp.int32)

        new_hp = jnp.maximum(
            jnp.int32(0),
            s.player_hp - jnp.where(spell != jnp.int32(1), dmg, jnp.int32(0))
        )
        new_done = s.done | (new_hp <= jnp.int32(0))

        # Paralyze: apply FROZEN hold timer (status_effects.TimedStatus.FROZEN=21).
        # Cite: mhitu.c ~412 lich spell path; vendor paralyze maps to frozen/held.
        _FROZEN_IDX = 21
        status = s.status
        timed = status.timed_statuses
        is_paralyze = spell == jnp.int32(1)
        frozen_cur = timed[_FROZEN_IDX]
        frozen_new = jnp.where(
            is_paralyze,
            jnp.maximum(frozen_cur, jnp.int32(_LICH_SPELL_PARALYZE_TURNS)),
            frozen_cur,
        )
        new_timed = timed.at[_FROZEN_IDX].set(frozen_new)
        new_status = status.replace(timed_statuses=new_timed)

        return s.replace(
            player_hp=new_hp,
            done=new_done,
            status=new_status,
        )

    return jax.lax.cond(in_range, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 5. Dragon breath  (vendor/nethack/src/mhitu.c::mattacku AT_BREA ~873,
#    breamu; damage d(6,6) per element; resistance gates via MR_* checks)
# ---------------------------------------------------------------------------

_BREATH_RANGE: int = 8  # vendor: breath has range up to 8 tiles (breamu ~line 570)


def _dragon_breath(state, slot: jnp.ndarray, rng: jax.Array):
    """Adult dragon breathes at player if within range.

    Damage: d(nd, 6) where nd = dragon's mattk->damn (vendor uses dobuzz
    ➜ zhitu, which deals d(nd, 6) for fire/cold/elec/poison/acid/magm).
    Reflection (player REFLECTING intrinsic): vendor's dobuzz bounces the
    beam off a reflecting target — equivalent here to zero damage.
    Resistance (player matching intrinsic) blocks damage.

    Range gate: vendor uses Euclidean² distance (``mdistu(mtmp)``, defined
    in hack.h:1531-1532 as ``dist2(x,y,u.ux,u.uy)`` = ``dx*dx+dy*dy``;
    see hacklib.c:672-678).  Breath fires when ``range2`` (mhitu.c:453,
    ``range2 = !monnear(...)`` = ``dist2 >= 3`` per mon.c:2476-2483) is
    true and the target is within ``BOLT_LIM * BOLT_LIM`` (hack.h:49
    ``BOLT_LIM 8``).

    Vendor cite: mhitu.c::mattacku case AT_BREA ~873 — 'if (range2) breamu';
                 mthrowu.c::breamm:1123 dobuzz call with mattk->damn;
                 zap.c::zhitu:4416-4438 d(nd, 6) for fire/cold/elec/acid/magm;
                 zap.c::dobuzz:4873 mon_reflects → beam reflected (no damage).
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    # Euclidean² distance per vendor mdistu/dist2 (hack.h:1531, hacklib.c:672).
    dx = mpos[0].astype(jnp.int32) - ppos[0].astype(jnp.int32)
    dy = mpos[1].astype(jnp.int32) - ppos[1].astype(jnp.int32)
    dist2 = dx * dx + dy * dy
    # range2: !monnear ⇔ dist2 >= 3 (mon.c:2476-2483).
    # Upper bound: BOLT_LIM*BOLT_LIM = 64 (hack.h:49 BOLT_LIM=8).
    _BOLT_LIM_SQ = jnp.int32(_BREATH_RANGE * _BREATH_RANGE)
    in_range = (dist2 >= jnp.int32(3)) & (dist2 <= _BOLT_LIM_SQ)

    safe_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _NUMMONS - 1)
    elem = _DRAGON_BREATH_ELEMENT[safe_entry].astype(jnp.int32)
    nd   = _DRAGON_BREATH_DAMN[safe_entry].astype(jnp.int32)

    def _apply(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        # d(nd, 6): roll up to 6 dice of d6, masked by nd.  Matches
        # zhitu's d(nd, 6) formula; nd is statically <=6 for all dragons.
        rolls = jax.random.randint(rng, (6,), 1, 7)
        mask  = jnp.arange(6) < nd
        raw_dmg = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)

        # Resistance gate: if player has the matching intrinsic, damage = 0.
        # Clamp res_idx to valid range before indexing; -1 signals "no resist".
        res_idx = _BREATH_RES_IDX[elem]
        safe_res = jnp.clip(res_idx, 0, s.status.intrinsics.shape[0] - 1)
        raw_res  = s.status.intrinsics[safe_res]
        has_res  = (res_idx >= jnp.int32(0)) & raw_res

        # Reflection: vendor's dobuzz reverses the beam off a reflecting
        # target so it never hits the original target.  Model as zero damage.
        # Cite: vendor/nethack/src/zap.c::dobuzz lines 4872-4882.
        reflecting = s.status.intrinsics[int(Intrinsic.REFLECTING)]

        dmg = jnp.where(has_res | reflecting, jnp.int32(0), raw_dmg)

        new_hp = jnp.maximum(jnp.int32(0), s.player_hp - dmg)
        new_done = s.done | (new_hp <= jnp.int32(0))
        s = s.replace(player_hp=new_hp, done=new_done)

        # Armor erosion: fire/cold/acid breath erodes worn body armor through
        # the central erode_obj path.  Vendor cite:
        #   vendor/nethack/src/zap.c::destroy_item / ::erode_armor  — invoked
        #   from mhitu.c::breamu when breath element is fire/cold/acid.  We
        #   route through items.erode_obj_slot here.
        from Nethax.nethax.subsystems.items import (
            erode_obj_slot, ERODE_BURN, ERODE_CORRODE,
        )
        from Nethax.nethax.subsystems.inventory import ArmorSlot

        body_slot = s.inventory.worn_armor[int(ArmorSlot.BODY)].astype(jnp.int32)
        has_body  = body_slot >= jnp.int32(0)

        # Element -> erode kind: FIRE/COLD -> ERODE_BURN (cite trap.c case BURN);
        # ACID -> ERODE_CORRODE.  Other elements: leave armor untouched.
        # Reflection causes the beam to never reach armor either.
        is_fire   = elem == jnp.int32(_ELEM_FIRE)
        is_cold   = elem == jnp.int32(_ELEM_COLD)
        is_acid   = elem == jnp.int32(_ELEM_ACID)
        do_burn   = (is_fire | is_cold) & has_body & (~has_res) & (~reflecting)
        do_corrode = is_acid & has_body & (~has_res) & (~reflecting)

        def _erode_burn(items_in):
            safe_b = jnp.clip(body_slot, 0, items_in.oeroded.shape[0] - 1)
            new_items, _ = erode_obj_slot(items_in, safe_b, ERODE_BURN, True)
            return new_items

        def _erode_corrode(items_in):
            safe_b = jnp.clip(body_slot, 0, items_in.oeroded.shape[0] - 1)
            new_items, _ = erode_obj_slot(items_in, safe_b, ERODE_CORRODE, True)
            return new_items

        items_after = jax.lax.cond(do_burn, _erode_burn, lambda x: x, s.inventory.items)
        items_after = jax.lax.cond(do_corrode, _erode_corrode, lambda x: x, items_after)
        new_inv = s.inventory.replace(items=items_after)
        return s.replace(inventory=new_inv)

    return jax.lax.cond(in_range, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 6. Kraken grab + drown  (vendor/nethack/src/mhitu.c ~1053 AT_HUGS+AD_WRAP,
#    and the general aquatic grab → drown path)
# ---------------------------------------------------------------------------

_KRAKEN_HOLD_TURNS: int = 5   # vendor: drown takes several turns (mhitu.c ~1053)

# TimedStatus indices from status_effects.py.
_TIMED_FROZEN: int = 21   # FROZEN = paralyzed/held solid (status_effects.py:164)


def _kraken_grab(state, slot: jnp.ndarray, rng: jax.Array):
    """Kraken grabs adjacent player and drags them into water.

    Sets player_in_water=True and applies FROZEN hold for N turns (the
    paralysis-hold models being gripped; vendor uses engulf/grab flag).
    Vendor cite: mhitu.c ~1053 — AT_HUGS+AD_WRAP 'grabs you, but cannot
    hold onto' / grab path; aquatic monsters drag hero into water (mhitu.c
    ~1068 AD_WRAP drown sequence).
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    adjacent = _cheby(mpos, ppos) <= jnp.int32(1)

    def _apply(s):
        # Set player_in_water and apply hold via FROZEN timer.
        timed = s.status.timed_statuses
        new_timed = timed.at[_TIMED_FROZEN].set(
            jnp.maximum(timed[_TIMED_FROZEN], jnp.int32(_KRAKEN_HOLD_TURNS))
        )
        new_status = s.status.replace(timed_statuses=new_timed)
        return s.replace(player_in_water=jnp.bool_(True), status=new_status)

    return jax.lax.cond(adjacent, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 7. Monster ranged spit  (vendor/nethack/src/mthrowu.c::spitmu :1268,
#    spitmm :1016 — AD_ACID/AD_BLND/AD_DRST venom projectile)
# ---------------------------------------------------------------------------

def _spit_attack(state, slot: jnp.ndarray, rng: jax.Array):
    """Monster spits venom at the player if lined up within range.

    Damage roll: d(damn, sides) per the monster's AT_SPIT attack row.
    Resistance gates per AD type:
        AD_ACID → RESIST_ACID    (zhitu acid path, zap.c)
        AD_DRST → RESIST_POISON  (poison strength drain — d(damn,sides) HP
                                  reduced; full RESIST_POISON blocks)
        AD_BLND → BLIND timer set to a fixed window when not RESIST_BLND
                  (vendor mksobj BLINDING_VENOM → does_blind path,
                  potion.c::peffects PR_BLINDED branch ~ 25-49 turns)

    Projectile model: vendor spitmm (mthrowu.c:1016-1077) calls m_throw()
    along sgn(tbx),sgn(tby) for ``distmin(...)`` steps; m_throw then walks
    one tile per loop iteration (mthrowu.c:673-826) and is blocked by
    walls / closed doors via MT_FLIGHTCHECK.  We model the same step walk
    via _m_throw_ray so that a WALL or CLOSED_DOOR between monster and
    player absorbs the venom.

    Vendor cite: mthrowu.c::spitmm:1016-1077 — venom mksobj + m_throw;
                 mthrowu.c::m_throw:572-849 — per-tile walk + flight check.
    Line-up gate matches vendor m_lined_up / linedup (mthrowu.c:1376-1392,
    hacklib.c::linedup): same row, same col, or 45-degree diagonal.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)

    # Vendor sgn(tbx)/sgn(tby): unit direction toward target.
    # ppos/mpos use (row, col); col → vendor x-axis, row → vendor y-axis.
    drow = ppos[0] - mpos[0]
    dcol = ppos[1] - mpos[1]
    dx = jnp.sign(dcol).astype(jnp.int32)   # vendor x  (col delta)
    dy = jnp.sign(drow).astype(jnp.int32)   # vendor y  (row delta)

    # Line-up: along same row, same col, or 45° diagonal.  (Matches
    # hacklib.c::linedup — tx==x || ty==y || abs(dx)==abs(dy).)
    abs_row = jnp.abs(drow)
    abs_col = jnp.abs(dcol)
    lined_up = (drow == jnp.int32(0)) | (dcol == jnp.int32(0)) | (abs_row == abs_col)

    # Not adjacent (range2 ≡ !monnear), still within spit max range.
    dist2 = drow * drow + dcol * dcol
    _BOLT_LIM_SQ = jnp.int32(_SPIT_RANGE * _SPIT_RANGE)
    in_range = (dist2 >= jnp.int32(3)) & (dist2 <= _BOLT_LIM_SQ) & lined_up

    safe_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _NUMMONS - 1)
    damn  = _SPIT_DAMN[safe_entry].astype(jnp.int32)
    sides = _SPIT_SIDES[safe_entry].astype(jnp.int32)
    adtyp = _SPIT_ADTYP[safe_entry].astype(jnp.int32)

    def _apply(s):
        # d(damn, sides): roll up to 6 dice (damn <= 6 in vendor for spitters).
        rng_roll, rng_blind = jax.random.split(rng)
        rolls = jax.random.randint(rng_roll, (6,), 1, 7)
        mask  = jnp.arange(6) < damn
        capped = jnp.minimum(rolls, jnp.maximum(sides, jnp.int32(1)))
        raw_dmg = jnp.sum(jnp.where(mask, capped, jnp.int32(0))).astype(jnp.int32)

        is_acid = adtyp == jnp.int32(_AD_ACID)
        is_drst = adtyp == jnp.int32(_AD_DRST)
        is_blnd = adtyp == jnp.int32(_AD_BLND)

        # Resistance gates per vendor zhitu / venom branches.
        res_acid = s.status.intrinsics[int(Intrinsic.RESIST_ACID)]
        res_pois = s.status.intrinsics[int(Intrinsic.RESIST_POISON)]
        # AD_BLND venom — vendor's BLINDING_VENOM clears via BLND_RES (an
        # equipment/intrinsic property tested in potion.c).  Use no-intrinsic
        # gating here; eyewear gating is handled in armor_effects.
        gated_dmg = jnp.where(
            (is_acid & res_acid) | (is_drst & res_pois),
            jnp.int32(0),
            raw_dmg,
        )

        # AD_BLND: blind duration (rnd(25)+25 per vendor potion.c PR_BLINDED
        # on blinding venom hit ~lines 1280-1320 in spitmm path).
        blind_dur = jax.random.randint(rng_blind, (), 25, 50)

        def _on_hit(s2):
            new_hp = jnp.maximum(jnp.int32(0), s2.player_hp - gated_dmg)
            s2 = s2.replace(
                player_hp=new_hp,
                done=s2.done | (new_hp <= jnp.int32(0)),
            )
            timed = s2.status.timed_statuses
            cur_blind = timed[int(TimedStatus.BLIND)]
            new_blind = jnp.where(
                is_blnd,
                jnp.maximum(cur_blind, blind_dur.astype(cur_blind.dtype)),
                cur_blind,
            )
            new_timed = timed.at[int(TimedStatus.BLIND)].set(new_blind)
            return s2.replace(status=s2.status.replace(timed_statuses=new_timed))

        # Step the venom along (dx, dy) up to _SPIT_RANGE tiles.  The walk
        # halts on WALL/CLOSED_DOOR so an intervening wall absorbs the spit.
        new_s, _final_pos, _hit = _m_throw_ray(
            s, mpos, dx, dy, _SPIT_RANGE, _on_hit,
        )
        return new_s

    return jax.lax.cond(in_range, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 8. Monster ranged gaze  (vendor/nethack/src/mhitu.c::gazemu :1666-1900 —
#    AD_FIRE/AD_CONF/AD_STON/AD_STUN/AD_BLND gaze effects)
# ---------------------------------------------------------------------------

def _gaze_attack(state, slot: jnp.ndarray, rng: jax.Array):
    """Monster gazes at the player within line-of-sight range.

    Per AD type (vendor cite: mhitu.c::gazemu 1666-1900):
        AD_FIRE → d(damn, sides) fire damage; gated by RESIST_FIRE
                  (gazemu:1830 — fire ant; ureflects RES if Reflecting).
        AD_CONF → CONFUSION timer extended (gazemu:1769 forest centaur).
        AD_STUN → STUNNED timer extended (gazemu umber-hulk path).
        AD_BLND → BLIND timer extended (gazemu floating-eye gaze if any).
        AD_STON → begin petrification (gazemu Medusa:1751); gated by
                  RESIST_STONE.
    Range check uses _GAZE_RANGE; the player must "see" the monster, modeled
    as Chebyshev distance gate (no explicit LoS field).
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _cheby(mpos, ppos)
    in_range = dist <= jnp.int32(_GAZE_RANGE)

    safe_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _NUMMONS - 1)
    damn  = _GAZE_DAMN[safe_entry].astype(jnp.int32)
    sides = _GAZE_SIDES[safe_entry].astype(jnp.int32)
    adtyp = _GAZE_ADTYP[safe_entry].astype(jnp.int32)

    def _apply(s):
        rng_dmg, rng_conf, rng_stun, rng_blnd = jax.random.split(rng, 4)
        # d(damn, sides) damage roll for AD_FIRE.
        rolls = jax.random.randint(rng_dmg, (6,), 1, 7)
        mask = jnp.arange(6) < damn
        capped = jnp.minimum(rolls, jnp.maximum(sides, jnp.int32(1)))
        raw_dmg = jnp.sum(jnp.where(mask, capped, jnp.int32(0))).astype(jnp.int32)

        is_fire = adtyp == jnp.int32(_AD_FIRE)
        is_conf = adtyp == jnp.int32(_AD_CONF)
        is_stun = adtyp == jnp.int32(_AD_STUN)
        is_blnd = adtyp == jnp.int32(_AD_BLND)
        is_ston = adtyp == jnp.int32(_AD_STON)

        res_fire = s.status.intrinsics[int(Intrinsic.RESIST_FIRE)]
        res_ston = s.status.intrinsics[int(Intrinsic.RESIST_STONE)]
        reflecting = s.status.intrinsics[int(Intrinsic.REFLECTING)]

        # Fire damage applied unless RESIST_FIRE / Reflecting.
        fire_dmg = jnp.where(is_fire & (~res_fire) & (~reflecting),
                             raw_dmg, jnp.int32(0))
        new_hp = jnp.maximum(jnp.int32(0), s.player_hp - fire_dmg)
        s = s.replace(player_hp=new_hp, done=s.done | (new_hp <= jnp.int32(0)))

        # CONFUSION (vendor: forest centaur gaze, gazemu:1769) —
        # extend timer by rn1(7, 16) per status_effects.cause_confusion.
        timed = s.status.timed_statuses
        conf_dur = (jax.random.randint(rng_conf, (), 0, 7) + jnp.int32(16))
        cur_conf = timed[int(TimedStatus.CONFUSION)]
        new_conf = jnp.where(is_conf,
                             cur_conf + conf_dur.astype(cur_conf.dtype),
                             cur_conf)

        # STUNNED — extend by rn1(5, 3).
        stun_dur = (jax.random.randint(rng_stun, (), 0, 5) + jnp.int32(3))
        cur_stun = timed[int(TimedStatus.STUNNED)]
        new_stun = jnp.where(is_stun,
                             cur_stun + stun_dur.astype(cur_stun.dtype),
                             cur_stun)

        # BLIND — extend timer by ``d(damn, damd)`` from the monster's
        # AD_BLND attack table.  Cite: vendor/nethack/src/mhitu.c:1804
        # ``int blnd = d((int) mattk->damn, (int) mattk->damd);``
        # The per-attack dice are already rolled into ``raw_dmg`` above
        # (single rng_dmg consumer covering whichever adtyp branch fires).
        # Previously this used ``rn1(25, 25)`` = 25..49, which produced
        # blind durations 10x+ longer than vendor for the dominant
        # blinding monsters (yellow light: vendor d(1,2) = 1..2; raven:
        # d(1,2); flash: d(1,2); etc.).  rng_blnd is left unconsumed
        # here for byte-stream stability but is no longer used.
        del rng_blnd  # no longer consumed
        blnd_dur = raw_dmg
        cur_blnd = timed[int(TimedStatus.BLIND)]
        new_blnd = jnp.where(is_blnd,
                             cur_blnd + blnd_dur.astype(cur_blnd.dtype),
                             cur_blnd)

        # STONED — Medusa gaze begins petrification.  Vendor sets STONED=5
        # (timeout.c::nh_timeout STONED fixed window).  Gated by RESIST_STONE.
        cur_ston = timed[int(TimedStatus.STONED)]
        new_ston = jnp.where(is_ston & (~res_ston) & (~reflecting),
                             jnp.maximum(cur_ston, jnp.int32(5)),
                             cur_ston)

        new_timed = (timed
                     .at[int(TimedStatus.CONFUSION)].set(new_conf)
                     .at[int(TimedStatus.STUNNED)].set(new_stun)
                     .at[int(TimedStatus.BLIND)].set(new_blnd)
                     .at[int(TimedStatus.STONED)].set(new_ston))
        new_status = s.status.replace(timed_statuses=new_timed)
        return s.replace(status=new_status)

    return jax.lax.cond(in_range, _apply, lambda s: s, state)


# ---------------------------------------------------------------------------
# 9. Stalker invisibility rendering helper
#    (vendor/nethack/src/makemon.c:1317, display.h::mon_visible macro ~88)
# ---------------------------------------------------------------------------

def monster_is_perminvis(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff the monster species has permanent natural invisibility.

    Cite: vendor/nethack/src/makemon.c:1317 — 'if (mndx == PM_STALKER)
          mtmp->perminvis = TRUE;'
    Caller should gate rendering: if perminvis and player lacks SEE_INVIS
    intrinsic (status_effects.Intrinsic.SEE_INVIS=29), show INVIS glyph.
    """
    safe = jnp.clip(entry_idx.astype(jnp.int32), 0, _NUMMONS - 1)
    return _PERMINVIS_TABLE[safe]


# ---------------------------------------------------------------------------
# Main dispatch  (called per monster slot each turn)
# ---------------------------------------------------------------------------

def monster_special_action(state, slot: jnp.ndarray, rng: jax.Array):
    """Dispatch the special action for monster at ``slot``.

    Returns an updated EnvState.  Called after movement in monster_ai.monster_turn.
    Only fires when the monster is alive, hostile, and not asleep.
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    safe_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _NUMMONS - 1)
    act = _SPECIAL_ACTION_TYPE[safe_entry].astype(jnp.int32)

    rng_a, rng_b, rng_c, rng_d, rng_e, rng_f, rng_g, rng_h = jax.random.split(rng, 8)

    # Each branch is a closure over its dedicated RNG key; only the matching
    # branch executes (lax.switch is JIT-pure with static num_branches).
    branches = [
        lambda s: s,                                             # 0: NONE
        lambda s: _nymph_steal(s, slot, rng_a),                 # 1: NYMPH
        lambda s: _leprechaun_steal_gold(s, slot, rng_b),       # 2: LEPREC
        lambda s: _succubus_drain(s, slot, rng_c),              # 3: SEDU
        lambda s: _lich_cast(s, slot, rng_d),                   # 4: LICH
        lambda s: _dragon_breath(s, slot, rng_e),               # 5: BREATH
        lambda s: _kraken_grab(s, slot, rng_f),                 # 6: KRAKEN
        lambda s: _spit_attack(s, slot, rng_g),                 # 7: SPIT
        lambda s: _gaze_attack(s, slot, rng_h),                 # 8: GAZE
    ]

    return jax.lax.switch(act, branches, state)
