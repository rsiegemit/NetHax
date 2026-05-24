"""Throwing subsystem helpers — vendor/nethack/src/dothrow.c.

Provides precomputed lookup tables and pure helper functions used by
thrown_attack() in combat.py:

  - _HATES_SILVER[N]  — bool mask, monsters hurt by silver
                         (vendor/nethack/src/dothrow.c:1343)
  - _OBJECT_MATERIAL[N_OBJECTS] — int8 material per object
  - compute_throw_range() — STR/weight-dependent range formula
                            (vendor/nethack/src/dothrow.c:1616-1625)
  - BOOMERANG_TYPE_IDS — frozenset of returning-weapon type_ids
                         (vendor/nethack/src/dothrow.c:1601-1611)
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Monster silver-hate table
# vendor/nethack/src/dothrow.c:1343  —  silver damage:
#   "if (obj->material == SILVER && hates_silver(mtmp->data))"
#   hates_silver → M2_UNDEAD | M2_WERE | M2_DEMON
# ---------------------------------------------------------------------------

def _build_hates_silver() -> jnp.ndarray:
    from Nethax.nethax.constants.monsters import (
        MONSTERS, M2_UNDEAD, M2_WERE, M2_DEMON,
    )
    _HATE_MASK = M2_UNDEAD | M2_WERE | M2_DEMON
    return jnp.array(
        [bool(m.flags2 & _HATE_MASK) for m in MONSTERS],
        dtype=jnp.bool_,
    )


_HATES_SILVER: jnp.ndarray = _build_hates_silver()


# ---------------------------------------------------------------------------
# Object material lookup table
# Indexed by type_id (= OBJECTS index).  Used to check SILVER / GLASS /
# POTTERY material for silver-bonus and breaks() logic.
# Vendor ref: vendor/nethack/include/objects.h otyp / oclass / material fields.
# ---------------------------------------------------------------------------

def _build_object_material() -> jnp.ndarray:
    from Nethax.nethax.constants.objects import OBJECTS
    return jnp.array(
        [int(o.material) if o is not None else 0 for o in OBJECTS],
        dtype=jnp.int8,
    )


_OBJECT_MATERIAL: jnp.ndarray = _build_object_material()


# ---------------------------------------------------------------------------
# Boomerang-class type_ids (return to thrower on miss)
# vendor/nethack/src/dothrow.c:1601-1611 — boomerang and aklys return.
# Indices sourced from OBJECTS table comments in objects.py.
# ---------------------------------------------------------------------------

#  9 — boomerang  (WEAPON, WOOD)
# 62 — aklys      (WEAPON, "thonged club", returns)
_BOOMERANG_TYPE_IDS: frozenset = frozenset([9, 62])

# JIT-visible boolean vector (index → is_returning_weapon)
def _build_boomerang_mask() -> jnp.ndarray:
    from Nethax.nethax.constants.objects import NUM_OBJECTS
    arr = [False] * NUM_OBJECTS
    for tid in _BOOMERANG_TYPE_IDS:
        if tid < NUM_OBJECTS:
            arr[tid] = True
    return jnp.array(arr, dtype=jnp.bool_)


_IS_RETURNING_WEAPON: jnp.ndarray = _build_boomerang_mask()


# ---------------------------------------------------------------------------
# Range formula — vendor/nethack/src/dothrow.c:1616-1625
#   urange = max(1, str // 2)
#   range  = urange - weight // 40
#   clamped to [1, 8]
# ---------------------------------------------------------------------------

_THROW_RANGE_MIN: int = 1
_THROW_RANGE_MAX: int = 8


def compute_throw_range(player_str: jnp.ndarray, weight: jnp.ndarray) -> jnp.ndarray:
    """Compute flight range.  JIT-pure.

    vendor/nethack/src/dothrow.c:1616-1625.

    Parameters
    ----------
    player_str : int-like — raw STR (0..125 scale, 118 = 18/100)
    weight     : int-like — item weight in aum

    Returns
    -------
    jnp.int32 in [1, 8]
    """
    str_i = player_str.astype(jnp.int32)
    w_i = weight.astype(jnp.int32)
    urange = jnp.maximum(jnp.int32(1), str_i // jnp.int32(2))
    rng = urange - w_i // jnp.int32(40)
    return jnp.clip(rng, _THROW_RANGE_MIN, _THROW_RANGE_MAX).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Breaktest — vendor/nethack/src/dothrow.c::breaktest (lines 2582-2609)
# vendor logic:
#   nonbreakchance = 1                              (line 2585)
#   if (oclass == ARMOR_CLASS && material == GLASS)  nonbreakchance = 90
#   if (obj_resists(obj, nonbreakchance, 99))        return FALSE
#   if (material == GLASS && !oartifact && oclass != GEM_CLASS) return TRUE
#   switch (oclass == POTION_CLASS ? POT_WATER : otyp):
#     EXPENSIVE_CAMERA / POT_WATER / EGG / CREAM_PIE / MELON
#     ACID_VENOM / BLINDING_VENOM: return TRUE
#   default: return FALSE
#
# obj_resists(obj, ochance, achance):
#   chance = rn2(100); return chance < (oartifact ? achance : ochance)
#
# This helper returns a (does_break, key_out) pair, JIT-pure, using two
# splits of ``rng`` (one for obj_resists, one reserved for breakmsg downstream).
# ---------------------------------------------------------------------------

# Vendor object type_ids — sourced from constants/objects.py byte-equal table.
_OTYP_EXPENSIVE_CAMERA: int = 204
_OTYP_MIRROR:           int = 205
_OTYP_CRYSTAL_BALL:     int = 206
_OTYP_EGG:              int = 241
_OTYP_MELON:            int = 255
_OTYP_CREAM_PIE:        int = 262
_OTYP_BLINDING_VENOM:   int = 451
_OTYP_ACID_VENOM:       int = 452

# ObjectClass ids — mirror Nethax.nethax.constants.objects.ObjectClass.
_OCLASS_WEAPON:  int = 2
_OCLASS_ARMOR:   int = 3
_OCLASS_POTION:  int = 8
_OCLASS_GEM:     int = 13
_OCLASS_VENOM:   int = 17

# Pyrolisk monster table index — vendor/nethack/src/monst.c PM_PYROLISK.
# Nethax constants/monster_entries/chunk1.py:234 — entry index 11.
_PM_PYROLISK: int = 11


def vendor_breaktest(
    rng: jax.Array,
    oclass: jnp.ndarray,
    otyp: jnp.ndarray,
    material: jnp.ndarray,
    is_artifact: jnp.ndarray,
) -> jnp.ndarray:
    """Return scalar bool: does this object break when thrown / dropped?

    Byte-equal to vendor/nethack/src/dothrow.c::breaktest lines 2582-2609.
    JIT-pure; takes a Threefry key and performs exactly one ``rn2(100)`` for
    the obj_resists roll.

    Parameters
    ----------
    rng         : JAX PRNG key (consumed for obj_resists).
    oclass      : int32 — ObjectClass value (= ItemCategory).
    otyp        : int32 — vendor object type_id (Nethax type_id).
    material    : int32 — Material enum value (GLASS = 4 in vendor; see
                  Nethax Material enum for the parallel int).
    is_artifact : bool   — artifact_idx >= 0.
    """
    from Nethax.nethax.constants.objects import Material as _Material
    from Nethax.nethax.rng import rn2 as _rn2_local

    GLASS = jnp.int32(int(_Material.GLASS))
    oclass_i  = oclass.astype(jnp.int32)
    otyp_i    = otyp.astype(jnp.int32)
    mat_i     = material.astype(jnp.int32)
    arti_b    = is_artifact

    # nonbreakchance: ARMOR + GLASS → 90, else 1.
    nonbreak = jnp.where(
        (oclass_i == jnp.int32(_OCLASS_ARMOR)) & (mat_i == GLASS),
        jnp.int32(90),
        jnp.int32(1),
    )
    # obj_resists: artifacts use achance=99 instead of nonbreak.
    rchance = jnp.where(arti_b, jnp.int32(99), nonbreak)
    chance_roll = _rn2_local(rng, 100)
    resisted = chance_roll < rchance

    # GLASS && !oartifact && oclass != GEM_CLASS → TRUE.
    glass_break = (
        (mat_i == GLASS)
        & (~arti_b)
        & (oclass_i != jnp.int32(_OCLASS_GEM))
    )

    # switch (oclass == POTION_CLASS ? POT_WATER : otyp):
    #   EXPENSIVE_CAMERA / POT_WATER / EGG / CREAM_PIE / MELON
    #   ACID_VENOM / BLINDING_VENOM → TRUE.
    is_potion = oclass_i == jnp.int32(_OCLASS_POTION)
    switch_break = (
        is_potion
        | (otyp_i == jnp.int32(_OTYP_EXPENSIVE_CAMERA))
        | (otyp_i == jnp.int32(_OTYP_EGG))
        | (otyp_i == jnp.int32(_OTYP_CREAM_PIE))
        | (otyp_i == jnp.int32(_OTYP_MELON))
        | (otyp_i == jnp.int32(_OTYP_ACID_VENOM))
        | (otyp_i == jnp.int32(_OTYP_BLINDING_VENOM))
    )

    raw_break = glass_break | switch_break
    return raw_break & (~resisted)
