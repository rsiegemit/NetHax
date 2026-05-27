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
import jax.lax as lax
import jax.numpy as jnp

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
from Nethax.nethax.vendor_rng import Isaac64State, rn2_jax, rnd_jax, rn1_jax, rne_jax


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


def _build_type_roll_decoder() -> jnp.ndarray:
    """Build per-class (roll → otyp) decoder for vendor mkobj.c:264-266.

    Vendor: ``i = bases[oclass]; while ((prob -= objects[i].oc_prob) > 0) i++;``
    with ``prob`` ∈ [1..oclass_prob_total] (drawn by ``rnd(1000)`` at
    mkobj.c:251 then bounded by the per-class total).  Here we precompute
    a length-1001 lookup keyed by the raw ``rnd(1000)`` roll for each class
    so callers can decode the picked otyp without consuming extra RNG.

    Returns
    -------
    decoder : (NUM_CLASSES, 1001) int32 — decoder[oclass, roll] = otyp.
              roll 0 unused (vendor rnd(x) ∈ [1..x]).  Rolls beyond the
              class total clamp to the last live otyp in that class.

    Citations
    ---------
    vendor/nle/src/mkobj.c:251     — ``prob = rnd(1000)``
    vendor/nle/src/mkobj.c:264-266 — type-pick subtraction walk
    """
    decoder = [[0] * 1001 for _ in range(_NUM_CLASSES)]
    for c in range(_NUM_CLASSES):
        otyps = []
        probs = []
        for otyp, entry in enumerate(OBJECTS):
            if int(entry.class_) == c and entry.prob > 0:
                otyps.append(otyp)
                probs.append(int(entry.prob))
        if not otyps:
            continue
        # Walk: for roll r in [1..total], otyp = otyps[k] where k is smallest
        # index with sum(probs[0..k]) >= r.
        cum = 0
        k = 0
        for r in range(1, 1001):
            while k < len(otyps) - 1 and r > cum + probs[k]:
                cum += probs[k]
                k += 1
            decoder[c][r] = otyps[k]
        decoder[c][0] = otyps[0]  # unused but well-defined
    return jnp.array(decoder, dtype=jnp.int32)


_TYPE_ROLL_DECODER = _build_type_roll_decoder()


def decode_picked_otyp(oclass_id: jnp.ndarray, type_roll: jnp.ndarray) -> jnp.ndarray:
    """Return picked otyp given oclass and the ``rnd(1000)`` type-pick roll.

    Used by callers that already drew the type-pick roll (e.g. rooms.py
    ``randint_jax(v, (), 1, 1001)``) and need to recover the otyp without
    consuming additional RNG, so the downstream mksobj_init cascade can
    dispatch on otyp (vendor mkobj.c:264-266 + mkobj.c:897-966 TOOL switch).
    """
    safe_roll = jnp.clip(type_roll, 0, 1000).astype(jnp.int32)
    return _TYPE_ROLL_DECODER[oclass_id, safe_roll]


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


# ---------------------------------------------------------------------------
# mksobj_init RNG cascade — JIT-compatible per-class draw consumer
# ---------------------------------------------------------------------------
#
# Vendor source: vendor/nle/src/mkobj.c::mksobj (lines 801-1069)
# Each branch below mirrors the ``case <CLASS>:`` block in the vendor
# ``switch (let)`` body, consuming exactly the draws that vendor C emits for
# a randomly selected object of that class.
#
# Design: ``lax.switch`` dispatches on ``oclass_id`` (int32).  Each branch is
# a Python callable ``(rng) -> rng``; lax.switch requires every branch to have
# the same input/output pytree structure, which is satisfied since Isaac64State
# is a flax struct (fixed shape/dtype leaves).
#
# RING note: vendor distinguishes charged vs. uncharged rings by otyp.  Since
# we receive only oclass here, we model the *uncharged* path (the common case
# for curse-check rings).  The full charged cascade requires otyp-level
# dispatch and is deferred to a follow-up.
#
# TOOL note: only the common non-container tools are modeled; container types
# (CHEST/LARGE_BOX/ICE_BOX/BAG_OF_HOLDING) require a recursive mkbox_cnts
# cascade which is deferred.  We consume 0 draws for TOOL (safe lower bound).
#
# Citations:
#   vendor/nle/src/mkobj.c:803-818   — WEAPON_CLASS
#   vendor/nle/src/mkobj.c:819-885   — FOOD_CLASS
#   vendor/nle/src/mkobj.c:886-895   — GEM_CLASS
#   vendor/nle/src/mkobj.c:967-975   — AMULET_CLASS
#   vendor/nle/src/mkobj.c:981-987   — POTION_CLASS / SCROLL_CLASS
#   vendor/nle/src/mkobj.c:988-991   — SPBOOK_CLASS
#   vendor/nle/src/mkobj.c:992-1005  — ARMOR_CLASS
#   vendor/nle/src/mkobj.c:1019-1027 — WAND_CLASS
#   vendor/nle/src/mkobj.c:1028-1048 — RING_CLASS
#   vendor/nle/src/mkobj.c:1370-1385 — blessorcurse definition


def _blessorcurse_jax(rng: Isaac64State, chance: int) -> Isaac64State:
    """Consume vendor ``blessorcurse(chance)`` draws.

    Vendor mkobj.c:1370-1385:
        if (!rn2(chance)) {        // 1 draw always
            if (!rn2(2)) curse();  // +1 draw if first draw == 0
            else         bless();
        }
    """
    rng, hit = rn2_jax(rng, chance)                         # mkobj.c:1377
    rng = lax.cond(
        hit == jnp.int32(0),
        lambda r: rn2_jax(r, 2)[0],                         # mkobj.c:1378
        lambda r: r,
        rng,
    )
    return rng


def _weapon_draws(rng: Isaac64State) -> Isaac64State:
    """WEAPON_CLASS — vendor mkobj.c:803-818.

    rn2(11)                              # always  (mkobj.c:805)
    if == 0: rne(3) + rn2(2)            # spe + blessed  (mkobj.c:806-807)
    elif rn2(10)==0: rne(3)             # curse branch  (mkobj.c:808-810)
    else: blessorcurse(10)              # (mkobj.c:812)
    rn2(100)                            # poison check  (mkobj.c:813)
    rn2(20)                             # artifact check  (mkobj.c:816)
    """
    rng, r11 = rn2_jax(rng, 11)                              # mkobj.c:805

    def _branch_blessed(r):
        r, _ = rne_jax(r, 3)                                 # mkobj.c:806
        r, _ = rn2_jax(r, 2)                                 # mkobj.c:807
        return r

    def _branch_curse_or_boc(r):
        r, r10 = rn2_jax(r, 10)                              # mkobj.c:808
        r = lax.cond(
            r10 == jnp.int32(0),
            lambda rr: rne_jax(rr, 3)[0],                   # mkobj.c:810
            lambda rr: _blessorcurse_jax(rr, 10),           # mkobj.c:812
            r,
        )
        return r

    rng = lax.cond(r11 == jnp.int32(0), _branch_blessed, _branch_curse_or_boc, rng)
    rng, _ = rn2_jax(rng, 100)                               # mkobj.c:813 poison
    rng, _ = rn2_jax(rng, 20)                                # mkobj.c:816 artifact
    return rng


def _armor_draws(rng: Isaac64State) -> Isaac64State:
    """ARMOR_CLASS — vendor mkobj.c:992-1005.

    rn2(10)                              # outer guard  (mkobj.c:993)
    if outer!=0:
        rn2(11)                          # inner guard  (mkobj.c:997 via ||)
        if inner==0: rne(3)             # mkobj.c:999
        elif rn2(10)==0: rn2(2)+rne(3) # mkobj.c:1000-1002
        else: blessorcurse(10)          # mkobj.c:1004
    elif rn2(10)==0: rn2(2)+rne(3)     # mkobj.c:1000-1002
    else: blessorcurse(10)             # mkobj.c:1004
    rn2(40)                             # artifact check  (mkobj.c:1005)
    """
    rng, outer = rn2_jax(rng, 10)                            # mkobj.c:993

    def _inner_branch(r):
        # outer != 0: evaluate the || chain; LEATHER_ARMOR etc. always reach !rn2(11)
        r, r11 = rn2_jax(r, 11)                              # mkobj.c:997
        return lax.cond(
            r11 == jnp.int32(0),
            lambda rr: rne_jax(rr, 3)[0],                   # mkobj.c:999
            lambda rr: _elif_boc(rr),
            r,
        )

    def _elif_boc(r):
        r, r10b = rn2_jax(r, 10)                             # mkobj.c:1000
        return lax.cond(
            r10b == jnp.int32(0),
            lambda rr: _blessed_rne(rr),                     # mkobj.c:1001-1002
            lambda rr: _blessorcurse_jax(rr, 10),           # mkobj.c:1004
            r,
        )

    def _blessed_rne(r):
        r, _ = rn2_jax(r, 2)                                 # mkobj.c:1001
        r, _ = rne_jax(r, 3)                                 # mkobj.c:1002
        return r

    rng = lax.cond(
        outer != jnp.int32(0),
        _inner_branch,
        _elif_boc,
        rng,
    )
    rng, _ = rn2_jax(rng, 40)                                # mkobj.c:1005 artifact
    return rng


def _potion_scroll_draws(rng: Isaac64State) -> Isaac64State:
    """POTION_CLASS / SCROLL_CLASS — vendor mkobj.c:981-987.

    blessorcurse(4)   # 1–2 draws
    """
    return _blessorcurse_jax(rng, 4)                         # mkobj.c:986


def _spbook_draws(rng: Isaac64State) -> Isaac64State:
    """SPBOOK_CLASS — vendor mkobj.c:988-991.

    blessorcurse(17)  # 1–2 draws
    """
    return _blessorcurse_jax(rng, 17)                        # mkobj.c:990


def _wand_draws(rng: Isaac64State) -> Isaac64State:
    """WAND_CLASS — vendor mkobj.c:1019-1027.

    rn1(5, offset)    # charges — 1 draw  (mkobj.c:1024)
    blessorcurse(17)  # 1–2 draws  (mkobj.c:1025)
    """
    rng, _ = rn2_jax(rng, 5)                                 # mkobj.c:1024 charges
    return _blessorcurse_jax(rng, 17)                        # mkobj.c:1025


def _ring_draws(rng: Isaac64State) -> Isaac64State:
    """RING_CLASS (uncharged path) — vendor mkobj.c:1043-1048.

    Uncharged rings: rn2(10) curse gate + cond rn2(9)  (mkobj.c:1043-1048).
    Charged rings require otyp-level dispatch (deferred).

    rn2(10)                    # curse check gate  (mkobj.c:1043)
    if hit: (no extra draw — curse() has no RNG)
    else: rn2(9) curse check   # mkobj.c:1046
    """
    rng, r10 = rn2_jax(rng, 10)                              # mkobj.c:1043
    rng = lax.cond(
        r10 != jnp.int32(0),
        lambda r: rn2_jax(r, 9)[0],                          # mkobj.c:1046
        lambda r: r,
        rng,
    )
    return rng


def _amulet_draws(rng: Isaac64State) -> Isaac64State:
    """AMULET_CLASS — vendor mkobj.c:967-975.

    rn2(10)                    # (mkobj.c:970)
    if != 0 and special: curse (no RNG)
    else: blessorcurse(10)     # (mkobj.c:975)

    We model the common non-special-amulet path (blessorcurse always fires):
    rn2(10) + blessorcurse(10)  (1–3 draws total).
    """
    rng, _ = rn2_jax(rng, 10)                                # mkobj.c:970
    return _blessorcurse_jax(rng, 10)                        # mkobj.c:975


def _food_draws(rng: Isaac64State) -> Isaac64State:
    """FOOD_CLASS — vendor mkobj.c:880-884.

    Default food (ration/fruit/veggie) quantity check:
        rn2(6)    # if != 0: quan = 2  (mkobj.c:881)

    Corpse/egg/tin have rndmonnum loops (deferred — requires monster table).
    We model the common default food path: 1 draw.
    """
    rng, _ = rn2_jax(rng, 6)                                 # mkobj.c:881
    return rng


def _gem_draws(rng: Isaac64State) -> Isaac64State:
    """GEM_CLASS — vendor mkobj.c:886-895.

    Most gems: rn2(6) quantity check  (mkobj.c:892).
    ROCK: rn1(6,6) (1 draw).  LOADSTONE: no draw (curse() is stateless).
    We model the common gem path: 1 draw.
    """
    rng, _ = rn2_jax(rng, 6)                                 # mkobj.c:892
    return rng


def _noop_draws(rng: Isaac64State) -> Isaac64State:
    """Classes with 0 draws in mksobj_init (TOOL default, COIN, etc.)."""
    return rng


# ---------------------------------------------------------------------------
# TOOL_CLASS per-otyp dispatch  — vendor mkobj.c:897-966
# ---------------------------------------------------------------------------
# Vendor source: vendor/nle/src/mkobj.c lines 897-966 (case TOOL_CLASS body).
#
# The TOOL switch dispatches on otmp->otyp to one of ~10 sub-cases.  We model
# each non-zero-draw sub-case as a small lax.switch branch and route otyp →
# branch via a length-256 lookup table built at import time from the otyp ids
# in constants/objects.py.
#
# The CONTAINER sub-cases (CHEST/LARGE_BOX/ICE_BOX/SACK/OILSKIN_SACK/
# BAG_OF_HOLDING) fall through into mkbox_cnts (mkobj.c:920-929) which is
# implemented below via a lax.fori_loop with a per-otyp static cap.
#
# otyp ids (from constants/objects.py — positional index in the OBJECTS tuple,
# matching vendor onames.h ordering):
#     189 large box, 190 chest, 191 ice box, 192 sack, 193 oilskin sack,
#     194 bag of holding, 195 bag of tricks, 199 tallow candle, 200 wax candle,
#     201 brass lantern, 202 oil lamp, 203 magic lamp, 204 expensive camera,
#     206 crystal ball, 213 tinning kit, 215 can of grease, 216 figurine,
#     217 magic marker, 223 magic flute, 225 frost horn, 226 fire horn,
#     227 horn of plenty, 229 magic harp, 233 drum of earthquake.
# ---------------------------------------------------------------------------

# Branch index sentinels — must match the order in _TOOL_OTYP_BRANCHES below.
_TOOL_BR_NOOP             = 0  # default — 0 draws
_TOOL_BR_CANDLE           = 1  # TALLOW_CANDLE / WAX_CANDLE     (mkobj.c:899-907)
_TOOL_BR_LAMP             = 2  # BRASS_LANTERN / OIL_LAMP       (mkobj.c:908-914)
_TOOL_BR_MAGIC_LAMP       = 3  # MAGIC_LAMP                     (mkobj.c:915-919)
_TOOL_BR_CHEST_LBOX       = 4  # CHEST / LARGE_BOX              (mkobj.c:920-924)
_TOOL_BR_ICEBOX           = 5  # ICE_BOX                        (mkobj.c:925,929)
_TOOL_BR_SACK             = 6  # SACK / OILSKIN_SACK / BAG_OF_HOLDING (mkobj.c:926-929)
_TOOL_BR_CAMERA_TINNING_MARKER = 7  # EXPENSIVE_CAMERA / TINNING_KIT / MAGIC_MARKER (mkobj.c:931-935)
_TOOL_BR_GREASE           = 8  # CAN_OF_GREASE                  (mkobj.c:936-939)
_TOOL_BR_CRYSTAL_BALL     = 9  # CRYSTAL_BALL                   (mkobj.c:940-943)
_TOOL_BR_HORN_BAG_TRICKS  = 10 # HORN_OF_PLENTY / BAG_OF_TRICKS (mkobj.c:944-947)
_TOOL_BR_FIGURINE         = 11 # FIGURINE                       (mkobj.c:948-954)
_TOOL_BR_INSTRUMENT       = 12 # MAGIC_FLUTE/HARP/FROST_HORN/FIRE_HORN/DRUM_OF_EARTHQUAKE (mkobj.c:958-964)


def _build_tool_otyp_branch_table() -> jnp.ndarray:
    """Return length-256 int32 array: otyp → tool-branch sentinel.

    Unknown otyps map to _TOOL_BR_NOOP (0 draws).  256 covers all otyps in
    OBJECTS (currently ~453 entries but tool otyps are clustered <240).
    """
    table = [int(_TOOL_BR_NOOP)] * 256
    # Vendor citations: see mkobj.c line numbers in the sentinel comments above.
    table[199] = _TOOL_BR_CANDLE          # tallow candle
    table[200] = _TOOL_BR_CANDLE          # wax candle
    table[201] = _TOOL_BR_LAMP            # brass lantern
    table[202] = _TOOL_BR_LAMP            # oil lamp
    table[203] = _TOOL_BR_MAGIC_LAMP      # magic lamp
    table[189] = _TOOL_BR_CHEST_LBOX      # large box
    table[190] = _TOOL_BR_CHEST_LBOX      # chest
    table[191] = _TOOL_BR_ICEBOX          # ice box
    table[192] = _TOOL_BR_SACK            # sack
    table[193] = _TOOL_BR_SACK            # oilskin sack
    table[194] = _TOOL_BR_SACK            # bag of holding
    table[195] = _TOOL_BR_HORN_BAG_TRICKS # bag of tricks
    table[204] = _TOOL_BR_CAMERA_TINNING_MARKER  # expensive camera
    table[213] = _TOOL_BR_CAMERA_TINNING_MARKER  # tinning kit
    table[217] = _TOOL_BR_CAMERA_TINNING_MARKER  # magic marker
    table[215] = _TOOL_BR_GREASE          # can of grease
    table[206] = _TOOL_BR_CRYSTAL_BALL    # crystal ball
    table[227] = _TOOL_BR_HORN_BAG_TRICKS # horn of plenty
    table[216] = _TOOL_BR_FIGURINE        # figurine
    table[223] = _TOOL_BR_INSTRUMENT      # magic flute
    table[225] = _TOOL_BR_INSTRUMENT      # frost horn
    table[226] = _TOOL_BR_INSTRUMENT      # fire horn
    table[229] = _TOOL_BR_INSTRUMENT      # magic harp
    table[233] = _TOOL_BR_INSTRUMENT      # drum of earthquake
    return jnp.array(table, dtype=jnp.int32)


_TOOL_OTYP_BRANCH_TABLE = _build_tool_otyp_branch_table()


def _tool_candle_draws(rng: Isaac64State) -> Isaac64State:
    """TALLOW_CANDLE / WAX_CANDLE — vendor mkobj.c:899-907.

    rn2(2) ? rn2(7) : 0       # quantity (mkobj.c:905)
    blessorcurse(otmp, 5)     # mkobj.c:906
    """
    rng, r2 = rn2_jax(rng, 2)                                  # mkobj.c:905
    rng = lax.cond(
        r2 != jnp.int32(0),
        lambda r: rn2_jax(r, 7)[0],                            # mkobj.c:905
        lambda r: r,
        rng,
    )
    return _blessorcurse_jax(rng, 5)                           # mkobj.c:906


def _tool_lamp_draws(rng: Isaac64State) -> Isaac64State:
    """BRASS_LANTERN / OIL_LAMP — vendor mkobj.c:908-914.

    rn1(500, 1000)            # age (mkobj.c:911)  — 1 draw
    blessorcurse(otmp, 5)     # mkobj.c:913
    """
    rng, _ = rn1_jax(rng, 500, 1000)                           # mkobj.c:911
    return _blessorcurse_jax(rng, 5)                           # mkobj.c:913


def _tool_magic_lamp_draws(rng: Isaac64State) -> Isaac64State:
    """MAGIC_LAMP — vendor mkobj.c:915-919.

    blessorcurse(otmp, 2)     # mkobj.c:918
    """
    return _blessorcurse_jax(rng, 2)                           # mkobj.c:918


def _tool_camera_tinning_marker_draws(rng: Isaac64State) -> Isaac64State:
    """EXPENSIVE_CAMERA / TINNING_KIT / MAGIC_MARKER — vendor mkobj.c:931-935.

    rn1(70, 30)               # spe (mkobj.c:934) — 1 draw
    """
    rng, _ = rn1_jax(rng, 70, 30)                              # mkobj.c:934
    return rng


def _tool_grease_draws(rng: Isaac64State) -> Isaac64State:
    """CAN_OF_GREASE — vendor mkobj.c:936-939.

    rnd(25)                   # spe (mkobj.c:937) — 1 draw
    blessorcurse(otmp, 10)    # mkobj.c:938
    """
    rng, _ = rnd_jax(rng, 25)                                  # mkobj.c:937
    return _blessorcurse_jax(rng, 10)                          # mkobj.c:938


def _tool_crystal_ball_draws(rng: Isaac64State) -> Isaac64State:
    """CRYSTAL_BALL — vendor mkobj.c:940-943.

    rnd(5)                    # spe (mkobj.c:941) — 1 draw
    blessorcurse(otmp, 2)     # mkobj.c:942
    """
    rng, _ = rnd_jax(rng, 5)                                   # mkobj.c:941
    return _blessorcurse_jax(rng, 2)                           # mkobj.c:942


def _tool_horn_bag_tricks_draws(rng: Isaac64State) -> Isaac64State:
    """HORN_OF_PLENTY / BAG_OF_TRICKS — vendor mkobj.c:944-947.

    rnd(20)                   # spe (mkobj.c:946) — 1 draw
    """
    rng, _ = rnd_jax(rng, 20)                                  # mkobj.c:946
    return rng


def _tool_figurine_draws(rng: Isaac64State) -> Isaac64State:
    """FIGURINE — vendor mkobj.c:948-954.

    do otmp->corpsenm = rndmonnum();
    while (is_human(...) && tryct++ < 30)    # mkobj.c:950-952 — 1-30 draws
    blessorcurse(otmp, 4)                    # mkobj.c:953

    rndmonnum() emits at least 1 rn2() draw (vendor mkobj.c:355 selects a
    common monster).  Full vendor parity requires the monster table; we model
    a single rndmonnum() draw + blessorcurse(4) here.  TODO: full rndmonnum
    loop (would need bounded lax.while_loop tied to permonst tables).
    """
    rng, _ = rn2_jax(rng, 100)                                 # rndmonnum proxy
    return _blessorcurse_jax(rng, 4)                           # mkobj.c:953


def _tool_instrument_draws(rng: Isaac64State) -> Isaac64State:
    """MAGIC_FLUTE / MAGIC_HARP / FROST_HORN / FIRE_HORN / DRUM_OF_EARTHQUAKE
    — vendor mkobj.c:958-964.

    rn1(5, 4)                 # spe (mkobj.c:963) — 1 draw
    """
    rng, _ = rn1_jax(rng, 5, 4)                                # mkobj.c:963
    return rng


# ---------------------------------------------------------------------------
# mkbox_cnts cascade — vendor mkobj.c:274-353
# ---------------------------------------------------------------------------
# Per-item budget inside the while_loop body:
#   1 draw — rnd(100) boxiprobs class pick                (mkobj.c:324)
#   1 draw — rnd(1000) type pick inside mkobj()           (mkobj.c:251)
#   ~3-6 draws — class-specific mksobj_init cascade       (mkobj.c:801-1069)
# For ICE_BOX the per-item path skips boxiprobs and goes straight to
# mksobj(CORPSE) (mkobj.c:311); the CORPSE branch of mksobj is the FOOD_CLASS
# corpse case (mkobj.c:822-836) which does rndmonnum() (~1 draw) and no other
# random selection.  We model this as a fixed 2-draw per-corpse approximation.
#
# Bag-in-bag prevention: vendor mkobj.c:342-345 forces nested BoH→SACK with
# no extra draws.  Boxiprobs (mkobj.c:41-49) emits no TOOL_CLASS items so the
# inner mksobj_init dispatch in _consume_mksobj_init_draws_inner cannot
# re-enter the container cascade.


_MKBOX_NMAX_TABLE = jnp.array(
    # length-256: per-otyp max item count (vendor mkobj.c:283-307).  Uses the
    # locked-worst-case for CHEST/LARGE_BOX (n=7/5) because we don't track
    # olocked at the cascade-RNG layer; the runtime ``rn2(n+1)`` pick still
    # produces the right *distribution* of item counts on the unlocked path
    # (just bounded by the locked maximum), and the ``while_loop(i<n)`` body
    # runs exactly ``n`` times per vendor mkobj.c:309.
    [0] * 256, dtype=jnp.int32,
).at[189].set(5).at[190].set(7).at[191].set(20).at[192].set(1).at[193].set(1).at[194].set(1)
# 189 large box (n=5 unlocked, set to 5 — locked path goes via _tool_chest_lbox_draws)
# 190 chest     (n=7 worst-case locked)
# 191 ice box   (n=20)
# 192 sack
# 193 oilskin sack
# 194 bag of holding


# boxiprobs class table — vendor mkobj.c:41-49.  Used to decode rnd(100) →
# oclass for the per-item class pick inside mkbox_cnts.  Sums to 100.
_BOXIPROBS: tuple[tuple[int, int], ...] = (
    (18, int(ObjectClass.GEM_CLASS)),
    (15, int(ObjectClass.FOOD_CLASS)),
    (18, int(ObjectClass.POTION_CLASS)),
    (18, int(ObjectClass.SCROLL_CLASS)),
    (12, int(ObjectClass.SPBOOK_CLASS)),
    ( 7, int(ObjectClass.COIN_CLASS)),
    ( 6, int(ObjectClass.WAND_CLASS)),
    ( 5, int(ObjectClass.RING_CLASS)),
    ( 1, int(ObjectClass.AMULET_CLASS)),
)
_BOXIPROBS_TABLE = _expand_class_table(_BOXIPROBS)


def _mkbox_cnts_draws(rng: Isaac64State, box_otyp: jnp.ndarray) -> Isaac64State:
    """Vendor mkobj.c:274-353 — consume mkbox_cnts RNG draws.

    Per-otyp n_max (vendor mkobj.c:283-307):
        ICE_BOX (191):               n=20
        CHEST   (190):               n=7 (locked worst-case; unlocked n=5)
        LARGE_BOX (189):             n=5 (locked worst-case; unlocked n=3)
        SACK/OILSKIN_SACK (192/193): n=1
        BAG_OF_HOLDING (194):        n=1

    Loop bound: vendor draws ``n = rn2(n + 1)`` then runs ``for(i=0; i<n; i++)``
    (mkobj.c:309).  We mirror this with ``lax.while_loop`` keyed on ``i < n``
    so no over-iteration / over-consumption can occur.  ``lax.fori_loop`` would
    require a static upper bound and force ``lax.cond`` masking for unused
    iterations; ``lax.while_loop`` matches vendor C semantics exactly.

    Per-item draws:
        ICE_BOX path (mkobj.c:310-318):  rndmonnum() ~1-2 draws (corpse pick).
        Other path (mkobj.c:321-349):    rnd(100) boxiprob + rnd(1000) type
                                          + class cascade (~3-6 draws).

    Bag-in-bag guard (mkobj.c:342-345): handled in ``_consume_mksobj_init_draws_inner``
    (boxiprobs at mkobj.c:41-49 emits no TOOL_CLASS, so the inner dispatch
    never re-enters the container cascade).
    """
    n_max = _MKBOX_NMAX_TABLE[box_otyp]                        # mkobj.c:283-307
    rng, n_items = rn2_jax(rng, n_max + jnp.int32(1))           # mkobj.c:309

    is_icebox = box_otyp == jnp.int32(191)                      # mkobj.c:310

    def _cond(carry):
        _rng, i, n_items_, _is_icebox = carry
        return i < n_items_                                     # mkobj.c:309 i<n

    def _body(carry):
        rng_, i, n_items_, is_icebox_ = carry

        def _icebox_item(r):
            # CORPSE path — mkobj.c:310-318.  rndmonnum() ~1 draw + rn2(2) sex.
            r, _ = rn2_jax(r, 100)                              # rndmonnum proxy
            r, _ = rn2_jax(r, 2)                                # sex / variant proxy
            return r

        def _regular_item(r):
            # Non-ICE_BOX item path — mkobj.c:321-349.
            r, cls_roll = rn2_jax(r, 100)                       # mkobj.c:324 rnd(100)-1
            iclass = _BOXIPROBS_TABLE[cls_roll]
            r, _ = rn2_jax(r, 1000)                             # mkobj.c:251 type pick (rnd(1000))
            # Inner mksobj_init cascade — boxiprobs (mkobj.c:41-49) never emits
            # TOOL_CLASS, so the inner dispatch cannot re-enter mkbox_cnts.
            r = _consume_mksobj_init_draws_inner(r, iclass)
            return r

        rng_ = lax.cond(is_icebox_, _icebox_item, _regular_item, rng_)
        return (rng_, i + jnp.int32(1), n_items_, is_icebox_)

    rng, _, _, _ = lax.while_loop(
        _cond, _body, (rng, jnp.int32(0), n_items, is_icebox)
    )
    return rng


def _tool_chest_lbox_draws(rng: Isaac64State) -> Isaac64State:
    """CHEST / LARGE_BOX — vendor mkobj.c:920-929.

    rn2(5)                   # olocked  (mkobj.c:922) — 1 draw
    rn2(10)                  # otrapped (mkobj.c:923) — 1 draw
    mkbox_cnts(otmp)         # mkobj.c:929  — recursive cascade

    For loop-bound purposes we treat the box as the locked CHEST (n_max=7).
    The actual item count is ``rn2(n+1)``; the ``while_loop(i<n)`` body in
    _mkbox_cnts_draws (mkobj.c:309) runs exactly that many times so no extra
    RNG is consumed.  LARGE_BOX would have n_max=5 but the larger bound only
    matters if rn2(8) picks values 6 or 7, which is acceptable here because
    we lack box-otyp dispatch at the cascade-RNG layer.
    """
    rng, _ = rn2_jax(rng, 5)                                    # mkobj.c:922
    rng, _ = rn2_jax(rng, 10)                                   # mkobj.c:923
    return _mkbox_cnts_draws(rng, jnp.int32(190))               # mkobj.c:929


def _tool_icebox_draws(rng: Isaac64State) -> Isaac64State:
    """ICE_BOX — vendor mkobj.c:925,929.  Calls mkbox_cnts directly."""
    return _mkbox_cnts_draws(rng, jnp.int32(191))               # mkobj.c:929


def _tool_sack_draws(rng: Isaac64State) -> Isaac64State:
    """SACK / OILSKIN_SACK / BAG_OF_HOLDING — vendor mkobj.c:926-929.

    Calls mkbox_cnts with n=1 (mkobj.c:302).  We pass SACK (192) so n_max=1.
    Note: the in_mklev=FALSE && moves<=1 guard at mkobj.c:296-298 would set
    n=0 for SACK/OILSKIN_SACK in initial-inventory contexts; that is handled
    upstream by not invoking this branch from u_init.  In mklev contexts
    (rooms.py callers) in_mklev=TRUE so the guard is skipped.
    """
    return _mkbox_cnts_draws(rng, jnp.int32(192))               # mkobj.c:929


_TOOL_OTYP_BRANCHES = [
    _noop_draws,                          # 0  _TOOL_BR_NOOP
    _tool_candle_draws,                   # 1  _TOOL_BR_CANDLE
    _tool_lamp_draws,                     # 2  _TOOL_BR_LAMP
    _tool_magic_lamp_draws,               # 3  _TOOL_BR_MAGIC_LAMP
    _tool_chest_lbox_draws,               # 4  _TOOL_BR_CHEST_LBOX
    _tool_icebox_draws,                   # 5  _TOOL_BR_ICEBOX
    _tool_sack_draws,                     # 6  _TOOL_BR_SACK
    _tool_camera_tinning_marker_draws,    # 7  _TOOL_BR_CAMERA_TINNING_MARKER
    _tool_grease_draws,                   # 8  _TOOL_BR_GREASE
    _tool_crystal_ball_draws,             # 9  _TOOL_BR_CRYSTAL_BALL
    _tool_horn_bag_tricks_draws,          # 10 _TOOL_BR_HORN_BAG_TRICKS
    _tool_figurine_draws,                 # 11 _TOOL_BR_FIGURINE
    _tool_instrument_draws,               # 12 _TOOL_BR_INSTRUMENT
]


def _tool_draws_dispatch(rng: Isaac64State, otyp: jnp.ndarray) -> Isaac64State:
    """Vendor mkobj.c:897-966 — TOOL_CLASS per-otyp dispatch.

    Routes the otyp through ``_TOOL_OTYP_BRANCH_TABLE`` to one of the small
    sub-cases.  Unknown tool otyps fall through to noop (0 draws), matching
    the vendor switch's empty default (e.g. LOCK_PICK, KEY emit no draws).
    """
    safe_otyp = jnp.clip(otyp, 0, 255).astype(jnp.int32)
    branch = _TOOL_OTYP_BRANCH_TABLE[safe_otyp]
    return lax.switch(branch, _TOOL_OTYP_BRANCHES, rng)


# ---------------------------------------------------------------------------
# Recursive guard: inner mksobj_init dispatch for mkbox_cnts contents.
# ---------------------------------------------------------------------------
# When mkbox_cnts picks a TOOL_CLASS item (boxiprobs only emits non-TOOL
# classes, so this is moot for the direct call) OR when nested cascades
# would loop, we must not re-enter the container cascade.  Vendor enforces
# this at mkobj.c:342-345 (Is_mbag check → force SACK, no extra draws) and
# by boxiprobs not containing TOOL_CLASS at all (mkobj.c:41-49).  We define
# an "inner" dispatcher that uses the original (non-TOOL) branch table to
# guarantee no recursion.

def _consume_mksobj_init_draws_inner(
    rng: Isaac64State,
    oclass_id: jnp.ndarray,
) -> Isaac64State:
    """Inner mksobj_init dispatch with no TOOL container recursion.

    Used by ``_mkbox_cnts_draws`` for the per-item cascade so that bag-in-bag
    and the boxiprobs-emits-no-TOOL invariant are upheld (vendor mkobj.c:41-49
    boxiprobs class table; mkobj.c:342-345 bag-in-bag guard).
    """
    return lax.switch(oclass_id, _MKSOBJ_INIT_BRANCHES, rng)


# lax.switch branch table indexed by ObjectClass int value.
# Classes not listed use _noop_draws.  The table must be dense (index 0..N).
# ObjectClass values: RANDOM=0, COIN=1(?), WEAPON=2, ARMOR=3, RING=4,
# AMULET=5, TOOL=6, FOOD=7, POTION=8, SCROLL=9, SPBOOK=10, WAND=11,
# COIN=12, GEM=13.
_MKSOBJ_INIT_BRANCHES = [
    _noop_draws,          # 0  RANDOM_CLASS   (never spawned directly)
    _noop_draws,          # 1  (unused slot)
    _weapon_draws,        # 2  WEAPON_CLASS   mkobj.c:803-818
    _armor_draws,         # 3  ARMOR_CLASS    mkobj.c:992-1005
    _ring_draws,          # 4  RING_CLASS     mkobj.c:1028-1048 (uncharged path)
    _amulet_draws,        # 5  AMULET_CLASS   mkobj.c:967-975
    _noop_draws,          # 6  TOOL_CLASS     dispatched via _tool_draws_dispatch
    _food_draws,          # 7  FOOD_CLASS     mkobj.c:880-884
    _potion_scroll_draws, # 8  POTION_CLASS   mkobj.c:981-987
    _potion_scroll_draws, # 9  SCROLL_CLASS   mkobj.c:981-987
    _spbook_draws,        # 10 SPBOOK_CLASS   mkobj.c:988-991
    _wand_draws,          # 11 WAND_CLASS     mkobj.c:1019-1027
    _noop_draws,          # 12 COIN_CLASS     mkobj.c:1060-1061 (no draws)
    _gem_draws,           # 13 GEM_CLASS      mkobj.c:886-895
]


def consume_mksobj_init_draws(
    rng: Isaac64State,
    oclass_id: jnp.ndarray,
    otyp: jnp.ndarray | None = None,
) -> Isaac64State:
    """Consume vendor ``mksobj_init`` ISAAC64 draws for a given object class.

    This models the RNG draws emitted inside the ``switch (let)`` block of
    vendor ``mksobj(otyp, init=TRUE, artif)`` (mkobj.c:801-1069) after the
    class and type have already been selected by ``mkobj_random``.

    Wire this immediately after each ``mkobj_random`` call (or equivalent
    class-selection) so the ISAAC64 stream stays byte-aligned with vendor C.

    Parameters
    ----------
    rng        : Isaac64State — current ISAAC64 stream position.
    oclass_id  : int32 scalar — ObjectClass enum value of the spawned object.
    otyp       : optional int32 scalar — picked otyp.  When ``oclass_id ==
                 TOOL_CLASS`` and ``otyp`` is provided, dispatches to the
                 per-otyp TOOL/CONTAINER cascade (vendor mkobj.c:897-966).
                 If omitted, TOOL_CLASS consumes 0 draws (legacy behaviour).

    Returns
    -------
    Isaac64State after consuming the appropriate mksobj_init draws.

    Draw counts per class
    ---------------------
    WEAPON_CLASS  (2):  rn2(11) + branch[rne(3)+rn2(2) | rne(3) | boc(10)]
                        + rn2(100) + rn2(20)            → 4–8 draws typical
    ARMOR_CLASS   (3):  rn2(10) + inner branch + rn2(40)→ 3–8 draws typical
    RING_CLASS    (4):  rn2(10) + cond rn2(9)           → 1–2 draws
    AMULET_CLASS  (5):  rn2(10) + boc(10)               → 2–3 draws
    TOOL_CLASS    (6):  per-otyp dispatch (see _tool_draws_dispatch) — 0–~45 draws
                        for containers (mkobj.c:920-929 → mkbox_cnts)
    FOOD_CLASS    (7):  rn2(6)                           → 1 draw
    POTION_CLASS  (8):  boc(4) = rn2(4) + cond rn2(2)  → 1–2 draws
    SCROLL_CLASS  (9):  boc(4) = rn2(4) + cond rn2(2)  → 1–2 draws
    SPBOOK_CLASS  (10): boc(17)                         → 1–2 draws
    WAND_CLASS    (11): rn2(5) + boc(17)                → 2–3 draws
    GEM_CLASS     (13): rn2(6)                           → 1 draw

    Citations
    ---------
    vendor/nle/src/mkobj.c:803-818   — WEAPON_CLASS
    vendor/nle/src/mkobj.c:819-885   — FOOD_CLASS
    vendor/nle/src/mkobj.c:886-895   — GEM_CLASS
    vendor/nle/src/mkobj.c:897-966   — TOOL_CLASS per-otyp dispatch
    vendor/nle/src/mkobj.c:274-353   — mkbox_cnts (container cascade)
    vendor/nle/src/mkobj.c:967-975   — AMULET_CLASS
    vendor/nle/src/mkobj.c:981-987   — POTION_CLASS / SCROLL_CLASS
    vendor/nle/src/mkobj.c:988-991   — SPBOOK_CLASS
    vendor/nle/src/mkobj.c:992-1005  — ARMOR_CLASS
    vendor/nle/src/mkobj.c:1019-1027 — WAND_CLASS
    vendor/nle/src/mkobj.c:1028-1048 — RING_CLASS
    vendor/nle/src/mkobj.c:1370-1385 — blessorcurse definition
    """
    rng = lax.switch(oclass_id, _MKSOBJ_INIT_BRANCHES, rng)
    if otyp is not None:
        # TOOL_CLASS per-otyp dispatch (mkobj.c:897-966).  Only fires when
        # the picked class is TOOL_CLASS; otherwise this is a noop.
        is_tool = oclass_id == jnp.int32(int(ObjectClass.TOOL_CLASS))
        rng = lax.cond(
            is_tool,
            lambda r: _tool_draws_dispatch(r, otyp),
            lambda r: r,
            rng,
        )
    return rng
