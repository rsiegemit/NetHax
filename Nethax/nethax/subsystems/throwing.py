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
