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
# Chunk5 (indices 261-321): succubus=298, incubus=300
_IDX_SUCCUBUS:      int = 298
_IDX_INCUBUS:       int = 300
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
        rng_item, rng_tele = jax.random.split(rng)
        items = s.inventory.items
        # Pick a random occupied slot: sample index 0..N_SLOTS-1.
        n_slots = items.category.shape[0]
        raw = jax.random.randint(rng_item, (), 0, n_slots)
        # Rotate: find a slot with category != 0, starting from raw.
        slot_seq = jnp.mod(jnp.arange(n_slots) + raw, n_slots)
        has_item = items.category[slot_seq] != jnp.int8(0)
        pick = jnp.argmax(has_item)   # first occupied slot in rotation
        steal_slot = slot_seq[pick]
        any_item = jnp.any(has_item)

        def _do_steal(s2):
            # Zero out the stolen slot (category=0 marks it empty).
            old_items = s2.inventory.items
            new_items = old_items.replace(
                category=old_items.category.at[steal_slot].set(jnp.int8(0)),
                quantity=old_items.quantity.at[steal_slot].set(jnp.int16(0)),
            )
            new_inv = s2.inventory.replace(items=new_items)
            return s2.replace(inventory=new_inv)

        s = jax.lax.cond(any_item, _do_steal, lambda s2: s2, s)

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

def _leprechaun_steal_gold(state, slot: jnp.ndarray, rng: jax.Array):
    """Adjacent leprechaun grabs player gold, then teleports.

    Vendor cite: mhitu.c doseduce() lines ~2269-2302 — leprechaun steals
    gold proportional to player wealth; monster rlocs after theft.
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    adjacent = _cheby(mpos, ppos) <= jnp.int32(1)

    def _apply(s):
        rng_amt, rng_tele = jax.random.split(rng)
        gold = s.player_gold.astype(jnp.int32)
        # Vendor: steal rnd(umoney+10)+500, capped at umoney.
        # JIT-pure approximation: steal 25-75% of gold (min 1 if gold > 0).
        roll = jax.random.uniform(rng_amt, ())
        steal_frac = jnp.float32(0.25) + roll * jnp.float32(0.5)
        stolen = jnp.maximum(
            jnp.int32(1),
            (gold.astype(jnp.float32) * steal_frac).astype(jnp.int32)
        )
        stolen = jnp.minimum(stolen, gold)
        new_gold = jnp.maximum(gold - stolen, jnp.int32(0)).astype(jnp.int32)

        raw_tele = jax.random.randint(rng_tele, (), 0, _N_TELE_TILES)
        tele_r = jnp.int16(raw_tele // _MAP_W)
        tele_c = jnp.int16(raw_tele % _MAP_W)
        new_pos = jnp.stack([tele_r, tele_c])
        new_mai = s.monster_ai.replace(
            pos=s.monster_ai.pos.at[idx].set(new_pos)
        )
        return s.replace(player_gold=new_gold, monster_ai=new_mai)

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
    Our simplification: -1 to one of {str,dex,con,int,wis,cha} chosen randomly,
    and Pw set to 0.  Adjacent check required.
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    adjacent = _cheby(mpos, ppos) <= jnp.int32(1)

    def _apply(s):
        rng_which, _ = jax.random.split(rng)
        which = jax.random.randint(rng_which, (), 0, 6)  # 0..5 → str/dex/con/int/wis/cha

        new_str = jnp.where(which == 0, jnp.maximum(jnp.int16(3), s.player_str - jnp.int16(1)), s.player_str)
        new_dex = jnp.where(which == 1, jnp.maximum(jnp.int8(3), s.player_dex - jnp.int8(1)), s.player_dex)
        new_con = jnp.where(which == 2, jnp.maximum(jnp.int8(3), s.player_con - jnp.int8(1)), s.player_con)
        new_int = jnp.where(which == 3, jnp.maximum(jnp.int8(3), s.player_int - jnp.int8(1)), s.player_int)
        new_wis = jnp.where(which == 4, jnp.maximum(jnp.int8(3), s.player_wis - jnp.int8(1)), s.player_wis)
        new_cha = jnp.where(which == 5, jnp.maximum(jnp.int8(3), s.player_cha - jnp.int8(1)), s.player_cha)

        return s.replace(
            player_pw=jnp.int32(0),
            player_str=new_str,
            player_dex=new_dex,
            player_con=new_con,
            player_int=new_int,
            player_wis=new_wis,
            player_cha=new_cha,
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
        dmg   = jnp.sum(jnp.where(mask, jnp.minimum(rolls, jnp.int32(sides)), jnp.int32(0)))

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

    Vendor cite: mhitu.c::mattacku case AT_BREA ~873 — 'if (range2) breamu';
                 mthrowu.c::breamm:1123 dobuzz call with mattk->damn;
                 zap.c::zhitu:4416-4438 d(nd, 6) for fire/cold/elec/acid/magm;
                 zap.c::dobuzz:4873 mon_reflects → beam reflected (no damage).
    """
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _cheby(mpos, ppos)
    in_range = (dist >= jnp.int32(2)) & (dist <= jnp.int32(_BREATH_RANGE))

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

    Vendor cite: mthrowu.c::spitmm:1016-1077 — venom mksobj + m_throw.
    Range check uses _SPIT_RANGE; line-of-sight gate via Chebyshev distance.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
    mai = state.monster_ai
    idx = slot.astype(jnp.int32)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _cheby(mpos, ppos)
    in_range = (dist >= jnp.int32(2)) & (dist <= jnp.int32(_SPIT_RANGE))

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

        new_hp = jnp.maximum(jnp.int32(0), s.player_hp - gated_dmg)
        s = s.replace(player_hp=new_hp, done=s.done | (new_hp <= jnp.int32(0)))

        # AD_BLND: apply BLIND timer (rnd(25)+25 per vendor potion.c
        # PR_BLINDED on blinding venom hit ~lines 1280-1320 in spitmm path).
        blind_dur = jax.random.randint(rng_blind, (), 25, 50)
        timed = s.status.timed_statuses
        cur_blind = timed[int(TimedStatus.BLIND)]
        new_blind = jnp.where(
            is_blnd,
            jnp.maximum(cur_blind, blind_dur.astype(cur_blind.dtype)),
            cur_blind,
        )
        new_timed = timed.at[int(TimedStatus.BLIND)].set(new_blind)
        new_status = s.status.replace(timed_statuses=new_timed)
        return s.replace(status=new_status)

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

        # BLIND — extend timer (rn1(25, 25) approximation).
        blnd_dur = (jax.random.randint(rng_blnd, (), 0, 25) + jnp.int32(25))
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
