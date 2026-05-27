"""Vendor-parity object-description shuffle (NLE_BYTEPARITY only).

This module reproduces the ISAAC64 stream consumption of NLE's
``init_objects()`` so that the descr_idx[NUM_OBJECTS] permutation that
drives ``shuffled_glyph()`` in NLE matches byte-for-byte.

Why this matters
----------------
NLE's ``inv_glyphs`` and map ``glyphs`` observations are emitted via
``vendor/nle/win/rl/winrl.cc::shuffled_glyph`` (lines 80-87), which maps
a canonical object glyph (``GLYPH_OBJ_OFF + otyp``) to its *shuffled
appearance* glyph (``GLYPH_OBJ_OFF + objects[otyp].oc_descr_idx``).
The shuffle runs once at game start inside ``init_objects()`` (vendor
``vendor/nle/src/o_init.c`` lines 111-183) and consumes ~200 ISAAC64
draws BEFORE dungeon-gen.  If we skip it the entire ISAAC64 stream is
offset, cascading divergence into every later draw (room layout,
monster placement, hit-point rolls, …).

What this module does
---------------------
* ``compute_descr_shuffle(rng) -> (descr_idx[453], rng')`` — replays
  vendor ``init_objects()`` byte-exactly:

    1. 3 ``rn2`` draws for the GEM_CLASS color jitter
       (o_init.c lines 144-162: turquoise→sapphire, aquamarine→sapphire,
       fluorite→{kept|sapphire|diamond|emerald}).  We ignore the
       *result* (it only affects gem appearance, which our obs doesn't
       expose) but we must consume the draws to keep the stream
       aligned.
    2. ``shuffle_all()`` (o_init.c lines 240-266): for each of the
       7 entire-classes (AMULET, POTION, RING, SCROLL, SPBOOK, WAND,
       VENOM) and 4 ARMOR sub-ranges (HELMET, LEATHER_GLOVES,
       CLOAK_OF_PROTECTION, SPEED_BOOTS), walk ``j`` from low to high
       and swap ``descr_idx[j]`` with ``descr_idx[j + rn2(hi-j+1)]``.
    3. 1 ``rn2(2)`` draw for ``WAN_NOTHING.oc_dir``
       (o_init.c line 182).

* ``shuffled_glyph(glyph, descr_idx)`` — mirror of
  ``winrl.cc::shuffled_glyph`` (lines 80-87): when ``glyph`` is in the
  object-glyph window ``[GLYPH_OBJ_OFF, GLYPH_OBJ_OFF + NUM_OBJECTS)``,
  return ``GLYPH_OBJ_OFF + descr_idx[glyph - GLYPH_OBJ_OFF]``.  All
  other glyph ranges (monsters, terrain, cmap, etc.) pass through
  unchanged.

Assumption: ``oc_name_known`` check skipped
-------------------------------------------
Vendor ``shuffle()`` (o_init.c lines 70-109) skips slots whose
``oc_name_known`` flag is already set, and re-draws ``i`` until the
target slot is also un-known.  At ``init_objects()`` time NO items are
pre-identified — ``oc_name_known`` is a runtime mutable flag set by
``makeknown()`` calls during play — so this skip is trivially false
for every slot.  The inner do/while loop degenerates to a single
``rn2`` draw per ``j``.  Documented here per the byte-parity contract.

JIT compatibility
-----------------
``compute_descr_shuffle`` uses ``lax.fori_loop`` with carry so it is
safe inside ``@jax.jit``.  It is also safe to call eagerly from
``env.reset`` (which is not jit-compiled).

Cites:
- vendor/nle/src/o_init.c::shuffle (lines 70-109)
- vendor/nle/src/o_init.c::obj_shuffle_range (lines 185-238)
- vendor/nle/src/o_init.c::shuffle_all (lines 240-266)
- vendor/nle/src/o_init.c::init_objects (lines 111-183, GEM_CLASS jitter
  + WAN_NOTHING coin-flip)
- vendor/nle/win/rl/winrl.cc::shuffled_glyph (lines 80-87)
- vendor/nle/include/display.h (GLYPH_OBJ_OFF = 1906, MAX_GLYPH window)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax


# ---------------------------------------------------------------------------
# Vendor constants — must match live NLE binary.
# Cite: vendor/nle/include/display.h GLYPH_OBJ_OFF; vendor/nle/src/objects.c
# NUM_OBJECTS (430 named + 23 None-named appearance slots = 453).
# ---------------------------------------------------------------------------

NUM_OBJECTS: int = 453
GLYPH_OBJ_OFF: int = 1906


# ---------------------------------------------------------------------------
# Shuffle ranges — derived from vendor objects.c canonical indices.
#
# Mirrors ``obj_shuffle_range()`` (o_init.c lines 186-237) evaluated at game
# init when no items are pre-identified.  Each (lo, hi) is inclusive.
#
# Order matches ``shuffle_all()`` (o_init.c lines 244-264):
#   1. Entire classes (TRUE = also shuffle material):
#        AMULET, POTION, RING, SCROLL, SPBOOK, WAND, VENOM
#   2. ARMOR sub-ranges (FALSE = description+color only):
#        HELMET, LEATHER_GLOVES, CLOAK_OF_PROTECTION, SPEED_BOOTS
#
# Verified against Nethax/nethax/constants/objects.py OBJECTS table by
# scanning oc_class transitions and unique-or-non-magic break points
# (see vendor objects.c BITS macro: oc_unique=bit3, oc_magic=bit5).
# ---------------------------------------------------------------------------

_SHUFFLE_RANGES_ENTIRE: tuple[tuple[int, int], ...] = (
    (178, 186),  # AMULET_CLASS: bases[AMULET] .. amulet of magical breathing
                 #   (next slot 187 = "cheap plastic imitation" has oc_unique=1)
    (272, 296),  # POTION_CLASS: bases[POTION] .. POT_WATER - 1
                 #   (POT_WATER=297 has fixed description)
    (150, 177),  # RING_CLASS: full class
    (298, 338),  # SCROLL_CLASS: bases[SCROLL] .. last magic slot
                 #   (blank paper=339 has oc_magic=0)
    (340, 379),  # SPBOOK_CLASS: bases[SPBOOK] .. last magic slot
                 #   (blank paper=380, novel=381, BoD=382 are non-magic / unique)
    (383, 409),  # WAND_CLASS: full class (includes 3 None-named extra desc slots)
    (451, 452),  # VENOM_CLASS: blinding venom, acid venom
)

_SHUFFLE_RANGES_ARMOR: tuple[tuple[int, int], ...] = (
    (78, 81),    # HELMET .. HELM_OF_TELEPATHY
    (136, 139),  # LEATHER_GLOVES .. GAUNTLETS_OF_DEXTERITY
    (125, 128),  # CLOAK_OF_PROTECTION .. CLOAK_OF_DISPLACEMENT
    (143, 149),  # SPEED_BOOTS .. LEVITATION_BOOTS
)


# ---------------------------------------------------------------------------
# Shuffle replay
# ---------------------------------------------------------------------------

def _shuffle_range(
    rng: Isaac64State,
    descr_idx: jnp.ndarray,
    lo: int,
    hi: int,
) -> tuple[Isaac64State, jnp.ndarray]:
    """Replay ``shuffle(lo, hi, ...)`` from o_init.c lines 86-108.

    Vendor (with ``oc_name_known`` always false at init time):
        for (j = lo; j <= hi; j++) {
            i = j + rn2(hi - j + 1);
            swap(descr_idx[j], descr_idx[i]);
        }

    Number of ``rn2`` draws == ``hi - lo + 1`` (one per ``j``), each with
    upper bound ``hi - j + 1`` ranging from ``hi - lo + 1`` down to 1.
    """
    n = hi - lo + 1
    if n < 2:
        # Vendor early-exits ``shuffle()`` when ``num_to_shuffle < 2``
        # (line 83).  No RNG draws consumed.
        return rng, descr_idx

    def body(step, carry):
        rng_c, arr_c = carry
        # j = lo + step; upper = hi - j + 1 = n - step
        j = jnp.int32(lo) + step
        upper = jnp.int32(n) - step
        rng_c, off = rn2_jax(rng_c, upper)
        i = j + off
        # Swap arr[j] and arr[i] (functional update).
        vj = arr_c[j]
        vi = arr_c[i]
        arr_c = arr_c.at[j].set(vi)
        arr_c = arr_c.at[i].set(vj)
        return (rng_c, arr_c)

    rng, descr_idx = jax.lax.fori_loop(
        jnp.int32(0), jnp.int32(n), body, (rng, descr_idx)
    )
    return rng, descr_idx


def compute_descr_shuffle(
    rng: Isaac64State,
) -> tuple[Isaac64State, jnp.ndarray]:
    """Replay vendor ``init_objects()`` byte-exactly using ISAAC64.

    Consumes the same ISAAC64 stream prefix as NLE's ``init_objects``:

      1. 3 ``rn2`` draws for GEM_CLASS color jitter
         (o_init.c lines 144-162).
      2. ``shuffle_all()`` over 7 entire classes + 4 ARMOR sub-ranges
         (o_init.c lines 240-266).
      3. 1 ``rn2(2)`` for ``WAN_NOTHING.oc_dir`` (o_init.c line 182).

    The GEM jitter and WAN_NOTHING draws are *consumed but discarded* —
    they don't affect the descr_idx[] permutation but the ISAAC64
    stream must advance to keep subsequent dungeon-gen / monster-spawn
    draws byte-aligned with NLE.

    Returns
    -------
    (rng_after, descr_idx[453]) where descr_idx[otyp] is the shuffled
    appearance index for canonical type ``otyp``.  Non-shuffled slots
    (most weapons, food, gems, etc.) keep ``descr_idx[i] = i``.
    """
    # GEM_CLASS color jitter — 2x rn2(2) + 1x rn2(4).
    # Cite: vendor/nle/src/o_init.c lines 144-162.
    rng, _ = rn2_jax(rng, 2)
    rng, _ = rn2_jax(rng, 2)
    rng, _ = rn2_jax(rng, 4)

    # Initial identity permutation: descr_idx[i] = i for all i.
    # Cite: vendor/nle/src/o_init.c line 130.
    descr_idx = jnp.arange(NUM_OBJECTS, dtype=jnp.int16)

    # shuffle_all() — entire classes first (with domaterial=TRUE), then
    # ARMOR sub-ranges (domaterial=FALSE).  For the descr_idx output
    # only the index swap matters; material/color swaps are vendor
    # state we don't track.  RNG draw count is identical either way.
    # Cite: vendor/nle/src/o_init.c lines 244-264.
    for lo, hi in _SHUFFLE_RANGES_ENTIRE:
        rng, descr_idx = _shuffle_range(rng, descr_idx, lo, hi)
    for lo, hi in _SHUFFLE_RANGES_ARMOR:
        rng, descr_idx = _shuffle_range(rng, descr_idx, lo, hi)

    # WAN_NOTHING.oc_dir coin flip — vendor o_init.c line 182.
    rng, _ = rn2_jax(rng, 2)

    return rng, descr_idx


# ---------------------------------------------------------------------------
# Glyph translation
# ---------------------------------------------------------------------------

def shuffled_glyph(glyph: jnp.ndarray, descr_idx: jnp.ndarray) -> jnp.ndarray:
    """Map a canonical glyph to its shuffled-appearance equivalent.

    Mirrors vendor ``winrl.cc::shuffled_glyph`` (lines 80-87):

        if glyph_is_normal_object(glyph):
            return GLYPH_OBJ_OFF + objects[glyph_to_obj(glyph)].oc_descr_idx
        return glyph

    Parameters
    ----------
    glyph     : int16 scalar or array of glyph IDs.
    descr_idx : int16[NUM_OBJECTS] permutation from
                ``compute_descr_shuffle``.

    Returns
    -------
    int16 (same shape as ``glyph``) with object-window entries
    remapped; all other glyphs unchanged.
    """
    g = glyph.astype(jnp.int32)
    in_obj_window = (g >= jnp.int32(GLYPH_OBJ_OFF)) & \
                    (g < jnp.int32(GLYPH_OBJ_OFF + NUM_OBJECTS))
    # Compute the remapped glyph; clip the otyp index so out-of-window
    # samples don't fault descr_idx[] (the where below discards them).
    otyp = jnp.clip(g - jnp.int32(GLYPH_OBJ_OFF),
                    jnp.int32(0), jnp.int32(NUM_OBJECTS - 1))
    shuffled_otyp = descr_idx[otyp].astype(jnp.int32)
    remapped = jnp.int32(GLYPH_OBJ_OFF) + shuffled_otyp
    return jnp.where(in_obj_window, remapped, g).astype(glyph.dtype)


# ---------------------------------------------------------------------------
# Default / identity descr_idx for non-byte-parity modes.
# ---------------------------------------------------------------------------

def identity_descr_idx() -> jnp.ndarray:
    """Identity permutation — every otyp maps to itself.

    Used by the default ``ParityMode.NLE`` path where the inv_glyphs
    observation still emits *canonical* glyphs (since no ISAAC64 stream
    is threaded).  ``shuffled_glyph()`` becomes the identity function.
    """
    return jnp.arange(NUM_OBJECTS, dtype=jnp.int16)
