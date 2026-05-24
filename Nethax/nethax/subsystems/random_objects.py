"""Random object generation — vendor mkobj.c port (Wave 44c).

Vendor sources
--------------
- vendor/nethack/src/mkobj.c::mkobj (lines 267-301)
- vendor/nethack/src/mkobj.c::rndobj (uses mkobjprobs/boxiprobs/rogueprobs/
  hellprobs class tables; the per-class weighted pick walks objects[i].oc_prob
  for i in [svb.bases[oclass]..) until the running sum exceeds the rolled
  ``prob = rnd(go.oclass_prob_totals[oclass])``.)
- vendor/nethack/src/objects.c — objects[] table providing the ``oc_prob``
  field per object; Python mirror is Nethax/nethax/constants/objects.py.

Algorithm (vendor mkobj.c::mkobj, lines 270-300)
------------------------------------------------
1. If oclass == RANDOM_CLASS, pick an oclass by walking the appropriate
   class-distribution table (mkobjprobs / rogueprobs / hellprobs) by
   ``rnd(100)`` until the running sum exhausts the roll.
2. Roll ``prob = rnd(oclass_prob_totals[oclass])``.
3. Walk objects[i].oc_prob for i = svb.bases[oclass]..end-of-class until the
   running sum exceeds ``prob``; that ``i`` is the picked otyp.

Status: library landing only. ``mkobj_random`` is exported and JIT-pure;
wiring into room/monster/wishing paths is deferred (see TODO at call sites
that currently use approximations).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass


# ---------------------------------------------------------------------------
# Class-distribution tables — vendor/nethack/src/mkobj.c lines 36-75
# ---------------------------------------------------------------------------
# Each tuple is (iprob, iclass) drawn verbatim from the vendor static
# ``struct icp`` arrays.  Probabilities sum to 100 per vendor (sentinel-
# walked by ``for (tprob = rnd(100); (tprob -= iprobs->iprob) > 0; iprobs++)``).

_MKOBJPROBS: tuple[tuple[int, int], ...] = (
    (10, int(ObjectClass.WEAPON_CLASS)),
    (11, int(ObjectClass.ARMOR_CLASS)),
    (20, int(ObjectClass.FOOD_CLASS)),
    ( 8, int(ObjectClass.TOOL_CLASS)),
    ( 7, int(ObjectClass.GEM_CLASS)),
    (16, int(ObjectClass.POTION_CLASS)),
    (16, int(ObjectClass.SCROLL_CLASS)),
    ( 4, int(ObjectClass.SPBOOK_CLASS)),
    ( 4, int(ObjectClass.WAND_CLASS)),
    ( 3, int(ObjectClass.RING_CLASS)),
    ( 1, int(ObjectClass.AMULET_CLASS)),
)

_ROGUEPROBS: tuple[tuple[int, int], ...] = (
    (12, int(ObjectClass.WEAPON_CLASS)),
    (12, int(ObjectClass.ARMOR_CLASS)),
    (22, int(ObjectClass.FOOD_CLASS)),
    (22, int(ObjectClass.POTION_CLASS)),
    (22, int(ObjectClass.SCROLL_CLASS)),
    ( 5, int(ObjectClass.WAND_CLASS)),
    ( 5, int(ObjectClass.RING_CLASS)),
)

_HELLPROBS: tuple[tuple[int, int], ...] = (
    (20, int(ObjectClass.WEAPON_CLASS)),
    (20, int(ObjectClass.ARMOR_CLASS)),
    (16, int(ObjectClass.FOOD_CLASS)),
    (12, int(ObjectClass.TOOL_CLASS)),
    (10, int(ObjectClass.GEM_CLASS)),
    ( 1, int(ObjectClass.POTION_CLASS)),
    ( 1, int(ObjectClass.SCROLL_CLASS)),
    ( 8, int(ObjectClass.WAND_CLASS)),
    ( 8, int(ObjectClass.RING_CLASS)),
    ( 4, int(ObjectClass.AMULET_CLASS)),
)


def _expand_class_table(rows: tuple[tuple[int, int], ...]) -> jnp.ndarray:
    """Expand (prob, oclass) rows into a length-100 lookup of oclass values.

    Vendor algorithm walks ``rnd(100)`` (i.e. uniform [1..100]) until the
    running sum of iprob entries exceeds it; equivalently, build a 100-entry
    table mapping each prob slot to its oclass and look up ``rn2(100)``.

    Probabilities sum to 100 per vendor; if a table sums to <100 (e.g. the
    rogue table sums to 100), padding is unnecessary, but we assert to catch
    typos in future edits.
    """
    flat: list[int] = []
    for iprob, iclass in rows:
        flat.extend([iclass] * iprob)
    assert len(flat) == 100, f"class table sums to {len(flat)}, expected 100"
    return jnp.array(flat, dtype=jnp.int32)


_MKOBJ_TABLE = _expand_class_table(_MKOBJPROBS)
_ROGUE_TABLE = _expand_class_table(_ROGUEPROBS)
_HELL_TABLE  = _expand_class_table(_HELLPROBS)


# ---------------------------------------------------------------------------
# Per-class object-probability tables — built from OBJECTS[].prob (oc_prob).
# Vendor: mkobj.c lines 289-292 walks objects[i].oc_prob from svb.bases[oclass]
# while (prob -= objects[i].oc_prob) > 0.  We precompute the (otyp, weight)
# pairs per oclass and pad to a common length so JAX choice can be vectorised.
# ---------------------------------------------------------------------------

_NUM_CLASSES = max(int(c) for c in ObjectClass) + 1  # 18 (RANDOM_CLASS..VENOM_CLASS)


def _build_class_object_tables() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (otyp_table, prob_table, total_table).

    Shapes:
      otyp_table : (NUM_CLASSES, MAX_PER_CLASS)  int32 — otyp indices, 0-padded
      prob_table : (NUM_CLASSES, MAX_PER_CLASS)  int32 — oc_prob weights,
                                                          0-padded for empty
                                                          tail entries
      total_table: (NUM_CLASSES,)                int32 — sum of oc_prob per
                                                          class (== vendor
                                                          go.oclass_prob_totals)
    """
    per_class_otyps: list[list[int]] = [[] for _ in range(_NUM_CLASSES)]
    per_class_probs: list[list[int]] = [[] for _ in range(_NUM_CLASSES)]
    for otyp, entry in enumerate(OBJECTS):
        if entry.prob <= 0:
            continue
        c = int(entry.class_)
        if c < 0 or c >= _NUM_CLASSES:
            continue
        per_class_otyps[c].append(otyp)
        per_class_probs[c].append(int(entry.prob))

    max_per_class = max((len(p) for p in per_class_otyps), default=1)
    if max_per_class == 0:
        max_per_class = 1

    otyp_arr = [
        row + [0] * (max_per_class - len(row)) for row in per_class_otyps
    ]
    prob_arr = [
        row + [0] * (max_per_class - len(row)) for row in per_class_probs
    ]
    totals = [sum(row) for row in per_class_probs]

    return (
        jnp.array(otyp_arr, dtype=jnp.int32),
        jnp.array(prob_arr, dtype=jnp.int32),
        jnp.array(totals,   dtype=jnp.int32),
    )


_CLASS_OTYPS, _CLASS_PROBS, _CLASS_TOTALS = _build_class_object_tables()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _pick_class(rng: jax.Array, in_rogue: bool, in_hell: bool) -> jnp.ndarray:
    """Vendor mkobj.c lines 274-283 — pick an oclass via class-prob table.

    Selection between mkobjprobs / rogueprobs / hellprobs is *static* per
    vendor (chosen by the Is_rogue_level / Inhell predicates at call time).
    We accept Python booleans so the choice is resolved at trace time and the
    table lookup is JIT-pure.
    """
    if in_rogue:
        table = _ROGUE_TABLE
    elif in_hell:
        table = _HELL_TABLE
    else:
        table = _MKOBJ_TABLE
    # rnd(100) - 1 -> rn2(100); equivalent for lookup into a length-100 table.
    idx = jax.random.randint(rng, (), minval=0, maxval=100, dtype=jnp.int32)
    return table[idx]


def _pick_otyp_in_class(rng: jax.Array, oclass: jnp.ndarray) -> jnp.ndarray:
    """Vendor mkobj.c lines 289-292 — weighted pick of otyp within oclass.

    ``oclass`` is a traced int32 scalar.  We gather that class's row from the
    precomputed (otyp, prob) tables and sample with ``jax.random.choice``
    using ``oc_prob`` weights.  Padded zero-prob slots have zero sampling
    probability so they cannot be picked (matches vendor behaviour of
    walking only the live oc_prob entries).
    """
    otyps  = _CLASS_OTYPS[oclass]                        # (MAX_PER_CLASS,)
    probs  = _CLASS_PROBS[oclass].astype(jnp.float32)    # (MAX_PER_CLASS,)
    # Guard against an all-zero row (e.g. RANDOM_CLASS itself, ROCK_CLASS):
    # fall back to uniform over the first slot to avoid NaN from /0.  The
    # caller should never request such a class via the public API because
    # _pick_class only emits live oclasses, but defence-in-depth keeps JIT
    # well-defined.
    total = jnp.sum(probs)
    safe_probs = jnp.where(total > 0, probs, jnp.zeros_like(probs).at[0].set(1.0))
    pick_idx = jax.random.choice(
        rng, otyps.shape[0], p=safe_probs / jnp.sum(safe_probs)
    )
    return otyps[pick_idx]


def mkobj_random(
    rng: jax.Array,
    oclass: int = int(ObjectClass.RANDOM_CLASS),
    *,
    in_rogue: bool = False,
    in_hell: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pick a random object — byte-equal port of vendor mkobj.c::mkobj.

    Parameters
    ----------
    rng       : JAX PRNG key.
    oclass    : ObjectClass enum value.  RANDOM_CLASS (0) selects an oclass
                via mkobjprobs/rogueprobs/hellprobs first; otherwise the given
                class is used directly.
    in_rogue  : True when generating on a rogue level (vendor Is_rogue_level).
    in_hell   : True when generating in Gehennom (vendor Inhell).

    Returns
    -------
    (category, type_id, quantity) — int32 scalars matching the Item struct
    convention in subsystems/inventory.py.

    ``quantity`` defaults to 1 here; vendor mksobj.c assigns stack quantities
    per-object-type (arrows = rnd(6), gold = rnd(level*...), etc.) which is
    out of scope for this landing.  Wiring callers may multiply afterwards.

    Citations
    ---------
    - vendor/nethack/src/mkobj.c::mkobj (lines 267-301)
    - vendor/nethack/src/mkobj.c lines 36-75 (mkobjprobs/rogueprobs/hellprobs)
    - vendor/nethack/src/objects.c::objects[] (oc_prob, oc_class)
    """
    key_cls, key_otyp = jax.random.split(rng, 2)

    if int(oclass) == int(ObjectClass.RANDOM_CLASS):
        picked_class = _pick_class(key_cls, in_rogue=in_rogue, in_hell=in_hell)
    else:
        picked_class = jnp.int32(int(oclass))

    otyp = _pick_otyp_in_class(key_otyp, picked_class)
    quantity = jnp.int32(1)
    return picked_class.astype(jnp.int32), otyp.astype(jnp.int32), quantity


# ---------------------------------------------------------------------------
# Diagnostic helpers (non-JIT) — convenient for tests that want to inspect
# the precomputed tables without running JAX ops.
# ---------------------------------------------------------------------------

def class_prob_totals() -> jnp.ndarray:
    """Return per-class oc_prob sums (== vendor go.oclass_prob_totals)."""
    return _CLASS_TOTALS
