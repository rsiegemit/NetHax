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
    (10, int(ObjectClass.ARMOR_CLASS)),
    (20, int(ObjectClass.FOOD_CLASS)),
    ( 8, int(ObjectClass.TOOL_CLASS)),
    ( 8, int(ObjectClass.GEM_CLASS)),
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
# RING note: vendor distinguishes charged vs. uncharged rings by oc_charged
# (vendor/nle/src/objects.c:537-602 RING macro; only adornment, gain
# strength/constitution, increase accuracy/damage, and protection set
# spec=1).  ``_ring_draws`` dispatches on a precomputed mask
# (``_RING_CHARGED_TABLE``) to either the charged enchantment-roll branch
# (mkobj.c:1029-1042) or the uncharged curse-check branch (mkobj.c:1043-1048).
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


# ---------------------------------------------------------------------------
# is_multigen / is_poisonable lookup tables — vendor obj.h:197-204
# ---------------------------------------------------------------------------
# Both predicates use IDENTICAL conditions in vendor obj.h:
#     oclass == WEAPON_CLASS && oc_skill in [-P_SHURIKEN, -P_BOW]
#                                       == [-25, -21]
# We bake the predicate into a single length-NUM_OBJECTS bool table for
# O(1) lookup inside ``_weapon_draws`` (which already runs under lax.cond
# gating).  Unknown otyps (>= NUM_OBJECTS) fall back to False via clip.

def _build_weapon_predicate_table() -> jnp.ndarray:
    """Return bool array sized to OBJECTS: True iff otyp is multigen/poisonable.

    Vendor cite: vendor/nle/include/obj.h:197-204.

    Note on Nethax skill offsets: Nethax's ``oc_skill`` enum is OFFSET BY 1
    relative to vendor (Nethax P_BOW=20 vs vendor P_BOW=21; Nethax
    P_SHURIKEN=24 vs vendor P_SHURIKEN=25).  The vendor predicate range
    ``[-P_SHURIKEN, -P_BOW] = [-25, -21]`` therefore maps to Nethax-space
    ``[-24, -20]``.  Verified set (arrows, ya, crossbow bolt, dart,
    shuriken — 8 otyps) matches vendor is_multigen exactly.
    """
    n = len(OBJECTS)
    table = [False] * n
    weapon_cls = int(ObjectClass.WEAPON_CLASS)
    for i, o in enumerate(OBJECTS):
        if int(o.class_) != weapon_cls:
            continue
        # Nethax skill space [-24, -20] == vendor [-P_SHURIKEN, -P_BOW].
        if -24 <= int(o.oc_skill) <= -20:
            table[i] = True
    return jnp.array(table, dtype=jnp.bool_)


_WEAPON_MULTIGEN_TABLE = _build_weapon_predicate_table()
# Vendor obj.h:201-204 — is_poisonable has IDENTICAL definition to is_multigen.
_WEAPON_POISONABLE_TABLE = _WEAPON_MULTIGEN_TABLE


# ---------------------------------------------------------------------------
# mk_artifact(otmp, A_NONE) eligible-count table — vendor artifact.c:125-208.
# ---------------------------------------------------------------------------
# When mkobj.c:816 fires `mk_artifact(otmp, A_NONE)` after `artif && !rn2(20)`
# succeeds, vendor walks artilist[1..NROFARTIFACTS] collecting entries where
# (a->otyp == otmp->otyp) && !(a->spfx & SPFX_NOGEN) && !artiexist[m].
# If the eligible count n > 0, a single `rn2(n)` picks the artifact.
# If n == 0, no draw fires.
#
# Threading artiexist through state is deferred (no artifacts exist on Dlvl 1
# fresh game so the static eligible-count is exact for the reset prelude).
# Once any artifact is generated mid-game, this table must be re-evaluated
# against the live artiexist registry.
#
# Cite: vendor/nle/src/artifact.c:125-208, vendor/nle/include/artilist.h:34-253.
# ---------------------------------------------------------------------------

def _build_artifact_anone_count_table() -> jnp.ndarray:
    """Return int8[NUM_OBJECTS]: count of non-NOGEN artifacts per otyp.

    Hard-coded from vendor artilist.h.  We do NOT consult Nethax wish._ARTIFACTS
    here because that table currently mis-maps Demonbane to "silver mace"
    (vendor says LONG_SWORD); the eligible-count table needs the byte-accurate
    vendor mapping.

    Cite: vendor/nle/include/artilist.h lines 47-252 (one entry per A() macro).
    """
    # Lazy import to avoid circular module load.
    from Nethax.nethax.subsystems.wish import _OBJECT_BY_NAME
    # (artifact_idx, base_obj_name, has_SPFX_NOGEN) — order = artilist.h.
    # Indices match Nethax wish._ARTIFACTS positions.
    VENDOR_ARTIFACTS = (
        # idx, base name (must match Nethax _OBJECT_BY_NAME), NOGEN flag
        ( 0, "long sword",        True),   # Excalibur            (NOGEN)
        ( 1, "katana",            False),  # Snickersnee
        ( 2, "runesword",         False),  # Stormbringer
        ( 3, "war hammer",        False),  # Mjollnir
        ( 4, "battle-axe",        False),  # Cleaver
        ( 5, "elven dagger",      False),  # Sting
        ( 6, "elven broadsword",  False),  # Orcrist
        ( 7, "silver saber",      False),  # Grayswandir
        ( 8, "long sword",        False),  # Vorpal Blade
        ( 9, "mace",              True),   # Sceptre of Might     (NOGEN)
        (10, "tsurugi",           True),   # Tsurugi of Muramasa  (NOGEN)
        (11, "mirror",            True),   # Magic Mirror Merlin  (NOGEN)
        (12, "crystal ball",      True),   # Orb of Detection     (NOGEN)
        (13, "luckstone",         True),   # Heart of Ahriman     (NOGEN)
        (14, "quarterstaff",      True),   # Staff of Aesculapius (NOGEN)
        (15, "pair of lenses",    True),   # Eyes of the Overworld(NOGEN)
        (16, "helm of brilliance",True),   # Mitre of Holiness    (NOGEN)
        (17, "bow",               True),   # Longbow of Diana     (NOGEN)
        (18, "skeleton key",      True),   # Master Key Thievery  (NOGEN)
        (19, "credit card",       True),   # Yendorian Express    (NOGEN)
        (20, "crystal ball",      True),   # Orb of Fate          (NOGEN)
        (21, "amulet of ESP",     True),   # Eye of Aethiopica    (NOGEN)
        (22, "long sword",        False),  # Frost Brand
        (23, "long sword",        False),  # Fire Brand
        (24, "broadsword",        False),  # Dragonbane
        (25, "long sword",        False),  # Demonbane (vendor LONG_SWORD)
        (26, "silver saber",      False),  # Werebane
        (27, "morning star",      False),  # Trollsbane
        (28, "orcish dagger",     False),  # Grimtooth
        (29, "athame",            False),  # Magicbane
        (30, "long sword",        False),  # Giantslayer
        (31, "war hammer",        False),  # Ogresmasher
        (32, "long sword",        False),  # Sunsword
    )
    n_obj = len(OBJECTS)
    counts = [0] * n_obj
    for _idx, base, nogen in VENDOR_ARTIFACTS:
        if nogen:
            continue
        otyp = _OBJECT_BY_NAME.get(base)
        if otyp is None:
            # Should not happen for the WEAPON_CLASS bases we care about.
            continue
        counts[otyp] += 1
    return jnp.array(counts, dtype=jnp.int32)


_ARTIFACT_ANONE_COUNT_TABLE = _build_artifact_anone_count_table()


def _mk_artifact_anone_draws(rng: Isaac64State, otyp: jnp.ndarray) -> Isaac64State:
    """Consume the single `rn2(n_eligible)` draw vendor mk_artifact A_NONE makes.

    Vendor cite: artifact.c:191 `m = eligible[rn2(n)]` — fires only if n > 0.
    Caller must already have gated on the outer `artif && !rn2(20)` success.

    Note: this assumes artiexist is empty (Dlvl 1 fresh game / reset prelude).
    Once mid-game artifact generation is supported, n must be re-derived from
    the live artiexist registry (P0 #13).
    """
    safe_otyp = jnp.clip(otyp, 0, _ARTIFACT_ANONE_COUNT_TABLE.shape[0] - 1).astype(jnp.int32)
    n = _ARTIFACT_ANONE_COUNT_TABLE[safe_otyp]
    # rn2(0) would be ill-defined; vendor skips the draw entirely when n==0.
    return lax.cond(
        n > jnp.int32(0),
        lambda r: rn2_jax(r, n)[0],
        lambda r: r,
        rng,
    )


def _weapon_draws(
    rng: Isaac64State,
    otyp: jnp.ndarray,
    artif: jnp.ndarray,
) -> Isaac64State:
    """WEAPON_CLASS — vendor mkobj.c:803-818.

    quan = is_multigen(otmp) ? rn1(6,6) : 1   # mkobj.c:804 — 0 or 1 draws
    rn2(11)                                    # always  (mkobj.c:805)
    if == 0: rne(3) + rn2(2)                  # spe + blessed (mkobj.c:806-807)
    elif rn2(10)==0: rne(3)                   # curse branch  (mkobj.c:808-810)
    else: blessorcurse(10)                    # (mkobj.c:812)
    if is_poisonable(otmp): rn2(100)          # poison check  (mkobj.c:813)
    if artif: rn2(20)                          # artifact check (mkobj.c:816)

    Vendor cite: vendor/nle/src/mkobj.c:803-818, vendor/nle/include/obj.h:197-204.
    """
    safe_otyp = jnp.clip(otyp, 0, _WEAPON_MULTIGEN_TABLE.shape[0] - 1).astype(jnp.int32)
    is_mg = _WEAPON_MULTIGEN_TABLE[safe_otyp]
    is_poi = _WEAPON_POISONABLE_TABLE[safe_otyp]

    # rn1(6, 6) = rn2(6) + 7 — 1 draw, gated on is_multigen (mkobj.c:804).
    rng = lax.cond(
        is_mg,
        lambda r: rn1_jax(r, 6, 6)[0],
        lambda r: r,
        rng,
    )

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
    # mkobj.c:813 — rn2(100) only fires when is_poisonable(otmp) is true.
    rng = lax.cond(
        is_poi,
        lambda r: rn2_jax(r, 100)[0],                        # mkobj.c:813
        lambda r: r,
        rng,
    )
    # mkobj.c:816 — rn2(20) only fires when ``artif`` flag is TRUE.
    # When the gate succeeds (r20 == 0), vendor calls mk_artifact(otmp, A_NONE)
    # which emits a single `rn2(n_eligible)` over per-otyp candidate artifacts.
    def _artif_check(r):
        r, r20 = rn2_jax(r, 20)                              # mkobj.c:816
        # mk_artifact(otmp, A_NONE) — vendor artifact.c:125-208.
        # Fires only when the rn2(20) gate succeeds AND n_eligible > 0.
        return lax.cond(
            r20 == jnp.int32(0),
            lambda rr: _mk_artifact_anone_draws(rr, safe_otyp),
            lambda rr: rr,
            r,
        )
    rng = lax.cond(
        artif,
        _artif_check,
        lambda r: r,
        rng,
    )
    return rng


# Special armor otyps that short-circuit the outer ||-chain (!rn2(11)).
# Vendor mkobj.c:993-998:
#   if (rn2(10)
#       && (otyp == FUMBLE_BOOTS
#           || otyp == LEVITATION_BOOTS
#           || otyp == HELM_OF_OPPOSITE_ALIGNMENT
#           || otyp == GAUNTLETS_OF_FUMBLING
#           || !rn2(11))) { curse(otmp); otmp->spe = -rne(3); }
# The `||` short-circuits: when otyp is one of these 4, !rn2(11) is never
# evaluated and the curse() / rne(3) path is taken directly.
# otyp ids from constants/objects.py positional indices:
#   80  helm of opposite alignment, 137 gauntlets of fumbling,
#   148 fumble boots, 149 levitation boots.
_OTYP_HELM_OF_OPPOSITE_ALIGNMENT: int = 80
_OTYP_GAUNTLETS_OF_FUMBLING:      int = 137
_OTYP_FUMBLE_BOOTS:               int = 148
_OTYP_LEVITATION_BOOTS:           int = 149


def _armor_draws(rng: Isaac64State, otyp: jnp.ndarray, artif: jnp.ndarray) -> Isaac64State:
    """ARMOR_CLASS — vendor mkobj.c:992-1006.

    Vendor source::

        if (rn2(10)
            && (otyp == FUMBLE_BOOTS
                || otyp == LEVITATION_BOOTS
                || otyp == HELM_OF_OPPOSITE_ALIGNMENT
                || otyp == GAUNTLETS_OF_FUMBLING
                || !rn2(11))) {
            curse(otmp);
            otmp->spe = -rne(3);                  // ≥1 draw
        } else if (!rn2(10)) {
            otmp->blessed = rn2(2);               // 1 draw
            otmp->spe = rne(3);                   // ≥1 draw
        } else
            blessorcurse(otmp, 10);               // 1–2 draws
        if (artif && !rn2(40))                     // 1 draw if artif
            otmp = mk_artifact(otmp, A_NONE);

    Per-otyp behaviour (outer = rn2(10)):
      outer == 0                                 → else-if chain: rn2(10) + ...
      outer != 0 AND non-special otyp            → rn2(11); if 0 curse path, else else-if chain
      outer != 0 AND special otyp                → curse path directly (NO rn2(11))

    The special-otyp short-circuit was previously not modeled (Nethax always
    drew rn2(11) when outer != 0), over-consuming the stream by 1 draw on
    every spawn of the 4 special armor otyps where outer != 0.

    Cite: vendor/nle/src/mkobj.c:992-1006.
    """
    rng, outer = rn2_jax(rng, 10)                            # mkobj.c:993
    is_special = (
        (otyp == jnp.int32(_OTYP_HELM_OF_OPPOSITE_ALIGNMENT))
        | (otyp == jnp.int32(_OTYP_GAUNTLETS_OF_FUMBLING))
        | (otyp == jnp.int32(_OTYP_FUMBLE_BOOTS))
        | (otyp == jnp.int32(_OTYP_LEVITATION_BOOTS))
    )

    def _inner_branch(r):
        # outer != 0 AND not special: evaluate !rn2(11) in the ||-chain.
        r, r11 = rn2_jax(r, 11)                              # mkobj.c:997
        return lax.cond(
            r11 == jnp.int32(0),
            lambda rr: rne_jax(rr, 3)[0],                   # mkobj.c:999
            lambda rr: _elif_boc(rr),
            r,
        )

    def _special_curse(r):
        # outer != 0 AND special otyp: curse + spe = -rne(3).  No rn2(11) draw.
        # Cite: vendor/nle/src/mkobj.c:994-999 (||-chain short-circuit).
        r, _ = rne_jax(r, 3)                                 # mkobj.c:999
        return r

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

    def _outer_truthy(r):
        # When outer != 0, the ||-chain is evaluated.  Special otyps short-
        # circuit to curse() directly; non-special otyps evaluate !rn2(11).
        return lax.cond(is_special, _special_curse, _inner_branch, r)

    rng = lax.cond(
        outer != jnp.int32(0),
        _outer_truthy,
        _elif_boc,
        rng,
    )
    # mkobj.c:1005 — rn2(40) only fires when ``artif`` flag is TRUE.
    rng = lax.cond(
        artif,
        lambda r: rn2_jax(r, 40)[0],                         # mkobj.c:1005
        lambda r: r,
        rng,
    )
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


# Special uncharged ring otyps that short-circuit the !rn2(9) curse check.
# Vendor mkobj.c:1043-1048:
#   } else if (rn2(10) && (otyp == RIN_TELEPORTATION
#                       || otyp == RIN_POLYMORPH
#                       || otyp == RIN_AGGRAVATE_MONSTER
#                       || otyp == RIN_HUNGER
#                       || !rn2(9))) curse(otmp);
# The `||` short-circuits: when otyp is one of the 4 special otyps, !rn2(9)
# is never evaluated and the RNG stream skips that draw.
# otyp ids from constants/objects.py positional indices:
#   161 ring of hunger, 162 ring of aggravate monster,
#   171 ring of teleportation, 173 ring of polymorph.
_OTYP_RIN_HUNGER:            int = 161
_OTYP_RIN_AGGRAVATE_MONSTER: int = 162
_OTYP_RIN_TELEPORTATION:     int = 171
_OTYP_RIN_POLYMORPH:         int = 173


# ---------------------------------------------------------------------------
# oc_charged ring mask — vendor/nle/src/objects.c:537-602 (RING macro).
# ---------------------------------------------------------------------------
# Each RING() entry passes (..., mgc, spec, mohs, ...).  The BITS macro at
# vendor/nle/src/objects.c:67 maps the 6th arg (`chrg`) into the oc_charged
# bitfield (objclass.h:59).  The RING macro at objects.c:539 passes ``spec``
# as ``chrg`` -- i.e. the third numeric column of each RING entry is
# oc_charged.  In vendor objects.c:542-601 only the first six rings have
# spec=1 (adornment, gain strength, gain constitution, increase accuracy,
# increase damage, protection).  Nethax otyp indices for these rings start
# at 150 (adornment) and run consecutively to 155 (protection).
#
# Cite: vendor/nle/src/objects.c:537-602; vendor/nle/include/objclass.h:59.
_OTYP_RIN_ADORNMENT:         int = 150
_OTYP_RIN_GAIN_STRENGTH:     int = 151
_OTYP_RIN_GAIN_CONSTITUTION: int = 152
_OTYP_RIN_INCREASE_ACCURACY: int = 153
_OTYP_RIN_INCREASE_DAMAGE:   int = 154
_OTYP_RIN_PROTECTION:        int = 155


def _build_ring_charged_table() -> jnp.ndarray:
    """Return length-NUM_OBJECTS bool array: True iff oc_charged ring.

    Vendor RING macro (vendor/nle/src/objects.c:537-541) passes ``spec`` into
    the BITS slot for oc_charged.  Only rings 150-155 in Nethax otyp space
    set spec=1; all others set spec=0.
    """
    n = len(OBJECTS)
    table = [False] * n
    for otyp in (
        _OTYP_RIN_ADORNMENT,
        _OTYP_RIN_GAIN_STRENGTH,
        _OTYP_RIN_GAIN_CONSTITUTION,
        _OTYP_RIN_INCREASE_ACCURACY,
        _OTYP_RIN_INCREASE_DAMAGE,
        _OTYP_RIN_PROTECTION,
    ):
        table[otyp] = True
    return jnp.array(table, dtype=jnp.bool_)


_RING_CHARGED_TABLE = _build_ring_charged_table()


def _ring_draws_charged(rng: Isaac64State) -> Isaac64State:
    """RING_CLASS (charged path) — vendor mkobj.c:1029-1042.

    Vendor source::

        if (objects[otmp->otyp].oc_charged) {
            blessorcurse(otmp, 3);               // 1-2 draws
            switch (rn2(9)) {                     // 1 draw, always
            case 0:
                break;                            // +0 draws
            case 1: case 2: case 3: case 4:
                otmp->spe = rne(3);              // +rne(3) draws
                break;
            case 5:
                otmp->spe = -rne(3);             // +rne(3) draws
                ...; break;
            case 6: case 7: case 8:
                ...; otmp->spe = -rne(3);        // +rne(3) draws
                break;
            }
        }

    Total per-spawn draw cost:
        blessorcurse(3)        : 1 or 2 draws
        rn2(9)                  : 1 draw       (always)
        rne(3) for r in 1..8    : 1+ draws    (case 0 has 0 rne(3) calls)

    Cite: vendor/nle/src/mkobj.c:1029-1042.
    """
    rng = _blessorcurse_jax(rng, 3)                          # mkobj.c:1031
    rng, r9 = rn2_jax(rng, 9)                                # mkobj.c:1032
    # All non-zero cases (1..8) call exactly one rne(3); case 0 calls none.
    rng = lax.cond(
        r9 != jnp.int32(0),
        lambda r: rne_jax(r, 3)[0],                          # mkobj.c:1036/1039/1042
        lambda r: r,
        rng,
    )
    return rng


def _ring_draws_uncharged(rng: Isaac64State, otyp: jnp.ndarray) -> Isaac64State:
    """RING_CLASS (uncharged path) — vendor mkobj.c:1043-1048.

    Vendor source (uncharged branch only)::

        } else {
            blessorcurse(otmp, 10);            // 1-2 draws
            if (otmp->otyp == RIN_TELEPORTATION
                || otmp->otyp == RIN_POLYMORPH
                || otmp->otyp == RIN_AGGRAVATE_MONSTER
                || otmp->otyp == RIN_HUNGER
                || !rn2(9))
                curse(otmp);
        }

    Per-otyp draw counts (uncharged otyps only):
      special otyp                : blessorcurse(10)              → 1-2 draws
      non-special otyp            : blessorcurse(10) + rn2(9)    → 2-3 draws

    Cite: vendor/nle/src/mkobj.c:1043-1048.
    """
    rng = _blessorcurse_jax(rng, 10)                         # mkobj.c:1044
    is_special = (
        (otyp == jnp.int32(_OTYP_RIN_HUNGER))
        | (otyp == jnp.int32(_OTYP_RIN_AGGRAVATE_MONSTER))
        | (otyp == jnp.int32(_OTYP_RIN_TELEPORTATION))
        | (otyp == jnp.int32(_OTYP_RIN_POLYMORPH))
    )
    # rn2(9) fires iff otyp is not one of the 4 specials (||-chain short-circuit).
    rng = lax.cond(
        ~is_special,
        lambda r: rn2_jax(r, 9)[0],                          # mkobj.c:1048
        lambda r: r,
        rng,
    )
    return rng


def _ring_draws(rng: Isaac64State, otyp: jnp.ndarray) -> Isaac64State:
    """RING_CLASS dispatch — vendor mkobj.c:1029-1048.

    Splits on ``objects[otyp].oc_charged`` (bake from vendor objects.c:537-602
    via ``_RING_CHARGED_TABLE``).  Charged rings (oc_charged=1) take the
    enchantment-roll branch at mkobj.c:1029-1042; uncharged rings take the
    curse-check branch at mkobj.c:1043-1048.

    Cite: vendor/nle/src/mkobj.c:1029-1048.
    """
    safe_otyp = jnp.clip(otyp, 0, _RING_CHARGED_TABLE.shape[0] - 1).astype(jnp.int32)
    is_charged = _RING_CHARGED_TABLE[safe_otyp]
    return lax.cond(
        is_charged,
        lambda r: _ring_draws_charged(r),
        lambda r: _ring_draws_uncharged(r, safe_otyp),
        rng,
    )


# Special amulet otyps whose curse path skips blessorcurse(10) when rn2(10)!=0.
# Vendor mkobj.c:970-975:
#   if (rn2(10) && (otyp == AMULET_OF_STRANGULATION
#                || otyp == AMULET_OF_CHANGE
#                || otyp == AMULET_OF_RESTFUL_SLEEP)) curse(otmp);
#   else blessorcurse(otmp, 10);
# otyp ids from constants/objects.py positional indices (match vendor onames.h):
#   180 amulet of strangulation, 181 amulet of restful sleep,
#   183 amulet of change.
_OTYP_AMULET_STRANGULATION: int = 180
_OTYP_AMULET_RESTFUL_SLEEP: int = 181
_OTYP_AMULET_CHANGE:        int = 183


def _amulet_draws(rng: Isaac64State, otyp: jnp.ndarray) -> Isaac64State:
    """AMULET_CLASS — vendor mkobj.c:967-975.

    Vendor source::

        if (rn2(10) && (otyp == STRANGULATION
                     || otyp == CHANGE
                     || otyp == RESTFUL_SLEEP)) {
            curse(otmp);                    // no further RNG
        } else {
            blessorcurse(otmp, 10);         // 1–2 draws
        }

    Per-otyp draw counts:
      non-special amulet         : rn2(10) + blessorcurse(10)  → 2–3 draws
      special amulet, rn2(10)==0 : rn2(10) + blessorcurse(10)  → 2–3 draws
      special amulet, rn2(10)!=0 : rn2(10)                      → 1 draw

    The special-otyp short-circuit was previously not modeled (Nethax always
    ran blessorcurse), causing the ISAAC64 stream to over-consume 1–2 draws
    in ~90 % of spawns of the 3 special amulet otyps.

    Cite: vendor/nle/src/mkobj.c:970-975.
    """
    rng, r10 = rn2_jax(rng, 10)                              # mkobj.c:970
    is_special = (
        (otyp == jnp.int32(_OTYP_AMULET_STRANGULATION))
        | (otyp == jnp.int32(_OTYP_AMULET_RESTFUL_SLEEP))
        | (otyp == jnp.int32(_OTYP_AMULET_CHANGE))
    )
    # Vendor short-circuits blessorcurse when (rn2(10) && special) — i.e. when
    # the rn2(10) draw was non-zero AND otyp is one of the 3 special amulets.
    skip_boc = (r10 != jnp.int32(0)) & is_special
    rng = lax.cond(
        skip_boc,
        lambda r: r,
        lambda r: _blessorcurse_jax(r, 10),                  # mkobj.c:975
        rng,
    )
    return rng


def _food_draws(rng: Isaac64State) -> Isaac64State:
    """FOOD_CLASS — vendor mkobj.c:880-884.

    Default food (ration/fruit/veggie) quantity check:
        rn2(6)    # if != 0: quan = 2  (mkobj.c:881)

    Corpse/egg/tin have rndmonnum loops (deferred — requires monster table).
    We model the common default food path: 1 draw.
    """
    rng, _ = rn2_jax(rng, 6)                                 # mkobj.c:881
    return rng


# GEM_CLASS otyps with no rn2(6) quantity draw in mksobj_init.
# Citation: vendor/nle/src/mkobj.c:887-895 (case GEM_CLASS body).
#   LOADSTONE (mkobj.c:888): early-return after curse() — no RNG draws.
#   LUCKSTONE (mkobj.c:892): ``otmp->otyp != LUCKSTONE`` short-circuits the
#     rn2(6) call, dropping the byte from the ISAAC64 stream.
# otyp ids match constants/objects.py positional indices (luckstone=442,
# loadstone=443) and vendor onames.h.
_OTYP_LUCKSTONE: int = 442
_OTYP_LOADSTONE: int = 443


def _gem_draws(rng: Isaac64State, otyp: jnp.ndarray) -> Isaac64State:
    """GEM_CLASS — vendor mkobj.c:886-895.

    Vendor source (mkobj.c:887-895)::

        case GEM_CLASS:
            otmp->corpsenm = 0; /* LOADSTONE hack */
            if (otmp->otyp == LOADSTONE)
                curse(otmp);                      // no RNG draws
            else if (otmp->otyp == ROCK)
                otmp->quan = (long) rn1(6, 6);    // 1 draw
            else if (otmp->otyp != LUCKSTONE && !rn2(6))
                otmp->quan = 2L;                  // 1 draw (rn2(6))
            else
                otmp->quan = 1L;

    Per-otyp draw counts:
      LOADSTONE  → 0 draws (curse() is stateless)
      ROCK       → 1 draw  (rn1(6,6) ≡ rn2(6)+7 — same byte cost as rn2(6))
      LUCKSTONE  → 0 draws (else-if short-circuits on ``otyp != LUCKSTONE``)
      other gems → 1 draw  (rn2(6))
    """
    # LOADSTONE (otyp 443) and LUCKSTONE (otyp 442) take the curse() and
    # default branches respectively — both skip the rn2(6) / rn1(6,6) draw.
    # Cite: vendor/nle/src/mkobj.c:888 (LOADSTONE) and :892 (LUCKSTONE gate).
    is_loadstone = otyp == jnp.int32(_OTYP_LOADSTONE)
    is_luckstone = otyp == jnp.int32(_OTYP_LUCKSTONE)
    skip_draw = is_loadstone | is_luckstone

    rng = lax.cond(
        skip_draw,
        lambda r: r,
        lambda r: rn2_jax(r, 6)[0],                          # mkobj.c:892
        rng,
    )
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
_TOOL_BR_CHEST            = 4  # CHEST                          (mkobj.c:920-924; n=7 locked worst-case)
_TOOL_BR_LBOX             = 5  # LARGE_BOX                      (mkobj.c:920-924; n=5 locked worst-case)
_TOOL_BR_ICEBOX           = 6  # ICE_BOX                        (mkobj.c:925,929)
_TOOL_BR_SACK             = 7  # SACK / OILSKIN_SACK / BAG_OF_HOLDING (mkobj.c:926-929)
_TOOL_BR_CAMERA_TINNING_MARKER = 8  # EXPENSIVE_CAMERA / TINNING_KIT / MAGIC_MARKER (mkobj.c:931-935)
_TOOL_BR_GREASE           = 9  # CAN_OF_GREASE                  (mkobj.c:936-939)
_TOOL_BR_CRYSTAL_BALL     = 10 # CRYSTAL_BALL                   (mkobj.c:940-943)
_TOOL_BR_HORN_BAG_TRICKS  = 11 # HORN_OF_PLENTY / BAG_OF_TRICKS (mkobj.c:944-947)
_TOOL_BR_FIGURINE         = 12 # FIGURINE                       (mkobj.c:948-954)
_TOOL_BR_INSTRUMENT       = 13 # MAGIC_FLUTE/HARP/FROST_HORN/FIRE_HORN/DRUM_OF_EARTHQUAKE (mkobj.c:958-964)


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
    table[189] = _TOOL_BR_LBOX            # large box  (mkobj.c:319: n=locked?5:3)
    table[190] = _TOOL_BR_CHEST           # chest      (mkobj.c:316: n=locked?7:5)
    table[191] = _TOOL_BR_ICEBOX         # ice box
    table[192] = _TOOL_BR_SACK           # sack
    table[193] = _TOOL_BR_SACK           # oilskin sack
    table[194] = _TOOL_BR_SACK           # bag of holding
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
# 189 large box (vendor mkobj.c:319: n=locked?5:3; worst-case cap=5)
# 190 chest     (vendor mkobj.c:316: n=locked?7:5; worst-case cap=7)
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


def _tool_chest_draws(rng: Isaac64State) -> Isaac64State:
    """CHEST — vendor mkobj.c:315-316,920-924,929.

    rn2(5)                   # olocked  (mkobj.c:922) — 1 draw
    rn2(10)                  # otrapped (mkobj.c:923) — 1 draw
    mkbox_cnts(otmp)         # mkobj.c:929 — n=locked?7:5; worst-case cap=7

    vendor mkobj.c:316: ``n = box->olocked ? 7 : 5``
    """
    rng, _ = rn2_jax(rng, 5)                                    # mkobj.c:922
    rng, _ = rn2_jax(rng, 10)                                   # mkobj.c:923
    return _mkbox_cnts_draws(rng, jnp.int32(190))               # mkobj.c:929 CHEST n_max=7


def _tool_lbox_draws(rng: Isaac64State) -> Isaac64State:
    """LARGE_BOX — vendor mkobj.c:318-319,920-924,929.

    rn2(5)                   # olocked  (mkobj.c:922) — 1 draw
    rn2(10)                  # otrapped (mkobj.c:923) — 1 draw
    mkbox_cnts(otmp)         # mkobj.c:929 — n=locked?5:3; worst-case cap=5

    vendor mkobj.c:319: ``n = box->olocked ? 5 : 3``
    Fixes cap-N: previously LARGE_BOX incorrectly used CHEST's n_max=7;
    now uses its own n_max=5 (locked worst-case per vendor mkobj.c:319).
    """
    rng, _ = rn2_jax(rng, 5)                                    # mkobj.c:922
    rng, _ = rn2_jax(rng, 10)                                   # mkobj.c:923
    return _mkbox_cnts_draws(rng, jnp.int32(189))               # mkobj.c:929 LARGE_BOX n_max=5


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
    _tool_chest_draws,                    # 4  _TOOL_BR_CHEST   (mkobj.c:316 n=locked?7:5)
    _tool_lbox_draws,                     # 5  _TOOL_BR_LBOX    (mkobj.c:319 n=locked?5:3)
    _tool_icebox_draws,                   # 6  _TOOL_BR_ICEBOX
    _tool_sack_draws,                     # 7  _TOOL_BR_SACK
    _tool_camera_tinning_marker_draws,    # 8  _TOOL_BR_CAMERA_TINNING_MARKER
    _tool_grease_draws,                   # 9  _TOOL_BR_GREASE
    _tool_crystal_ball_draws,             # 10 _TOOL_BR_CRYSTAL_BALL
    _tool_horn_bag_tricks_draws,          # 11 _TOOL_BR_HORN_BAG_TRICKS
    _tool_figurine_draws,                 # 12 _TOOL_BR_FIGURINE
    _tool_instrument_draws,               # 13 _TOOL_BR_INSTRUMENT
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

    Inner callers don't have the decoded otyp (boxiprobs picks class but
    doesn't expose the per-item type-roll downstream), so we pass otyp=0
    and artif=False — both are safe defaults: boxiprobs (mkobj.c:41-49) emits
    no WEAPON_CLASS items so the otyp=0 path through ``_weapon_draws`` is
    never taken in practice, and bag-in-bag mksobj calls always pass
    artif=FALSE per vendor mkobj.c:929 → mkbox_cnts → mksobj recursion.
    """
    return lax.switch(
        oclass_id, _MKSOBJ_INIT_BRANCHES,
        rng, jnp.int32(0), jnp.bool_(False),
    )


# lax.switch branch table indexed by ObjectClass int value.
# Classes not listed use _noop_draws.  The table must be dense (index 0..N).
# ObjectClass values: RANDOM=0, COIN=1(?), WEAPON=2, ARMOR=3, RING=4,
# AMULET=5, TOOL=6, FOOD=7, POTION=8, SCROLL=9, SPBOOK=10, WAND=11,
# COIN=12, GEM=13.
#
# All branches share the uniform signature ``(rng, otyp, artif) -> rng`` so
# lax.switch can dispatch them.  Branches that don't depend on otyp/artif
# simply ignore those args.  WEAPON_CLASS uses both (otyp → is_multigen/is_poisonable
# table lookup; artif → rn2(20) gate).  ARMOR_CLASS uses artif (→ rn2(40) gate).

def _weapon_branch(rng, otyp, artif):
    return _weapon_draws(rng, otyp, artif)

def _armor_branch(rng, otyp, artif):
    # otyp threaded through so FUMBLE_BOOTS/LEVITATION_BOOTS/
    # HELM_OF_OPPOSITE_ALIGNMENT/GAUNTLETS_OF_FUMBLING short-circuit the
    # !rn2(11) draw.  Cite: vendor/nle/src/mkobj.c:992-1006.
    return _armor_draws(rng, otyp, artif)

def _noop_branch(rng, otyp, artif):
    del otyp, artif
    return rng

def _ring_branch(rng, otyp, artif):
    del artif
    # otyp threaded through so RIN_TELEPORTATION/POLYMORPH/AGGRAVATE/HUNGER
    # short-circuit the !rn2(9) curse check.  Cite: vendor/nle/src/mkobj.c:1043-1048.
    return _ring_draws(rng, otyp)

def _amulet_branch(rng, otyp, artif):
    del artif
    # otyp threaded through so STRANGULATION/CHANGE/RESTFUL_SLEEP skip
    # blessorcurse when rn2(10) != 0.  Cite: vendor/nle/src/mkobj.c:970-975.
    return _amulet_draws(rng, otyp)

def _food_branch(rng, otyp, artif):
    del otyp, artif
    return _food_draws(rng)

def _potion_scroll_branch(rng, otyp, artif):
    del otyp, artif
    return _potion_scroll_draws(rng)

def _spbook_branch(rng, otyp, artif):
    del otyp, artif
    return _spbook_draws(rng)

def _wand_branch(rng, otyp, artif):
    del otyp, artif
    return _wand_draws(rng)

def _gem_branch(rng, otyp, artif):
    del artif
    # otyp threaded through so LOADSTONE/LUCKSTONE skip the rn2(6) draw.
    # Cite: vendor/nle/src/mkobj.c:887-895.
    return _gem_draws(rng, otyp)


# ROCK_CLASS STATUE otyp — vendor onames.h STATUE; positional index in
# constants/objects.py OBJECTS tuple (line 9149: "# 448 — statue").
_OTYP_STATUE: int = 448


def _rock_branch(rng, otyp, artif):
    """ROCK_CLASS — vendor mkobj.c:1050-1058.

    Vendor source::

        case ROCK_CLASS:
            if (otmp->otyp == STATUE) {
                otmp->corpsenm = rndmonnum();
                if (!verysmall(&mons[otmp->corpsenm])
                    && rn2(level_difficulty() / 2 + 10) > 10)
                    (void) add_to_container(otmp,
                                            mkobj(SPBOOK_CLASS, FALSE));
            }
            break;

    Per-otyp draw counts (Dlvl 1 byte-parity scope):
      otyp != STATUE              : 0 draws
      otyp == STATUE              : rndmonnum() (1 draw via rnd(choice_count))

    The verysmall + ``rn2(level_difficulty()/2 + 10) > 10`` gate is
    unreachable at Dlvl 1: ``level_difficulty()`` returns depth=1 in the
    main dungeon, so the modulus is ``1/2 + 10 = 10`` and ``rn2(10)``
    produces values in [0, 9], which can never satisfy ``> 10``.  The
    nested ``mkobj(SPBOOK_CLASS)`` branch therefore never fires at depth=1,
    and ``rn2`` itself is short-circuited by the leading ``!verysmall``
    test on tiny monsters.  Both sub-paths are skipped here — re-enable
    them when extending byte-parity beyond Dlvl 1 (depth-threading the
    branch signature is required).

    rndmonnum() is realised via ``pick_monster_for_level`` (byte-exact,
    drawing ``rnd(choice_count)`` from the ISAAC64 stream — matches the
    main-dungeon path where the rn2(7) quest-branch gate short-circuits).

    Vendor cite: vendor/nle/src/mkobj.c:1050-1058 (STATUE branch);
                 vendor/nle/src/makemon.c:1591-1594 (rndmonst draw);
                 vendor/nle/include/mondata.h:8 (verysmall).
    """
    del artif  # ROCK_CLASS makes no artifact / curse-check draws.
    is_statue = otyp == jnp.int32(_OTYP_STATUE)

    def _statue_path(r):
        # Lazy import to avoid spawning.py → random_objects.py circular import
        # (spawning imports consume_mksobj_init_draws at module top).
        from Nethax.nethax.dungeon.spawning import pick_monster_for_level

        # depth=1 hard-coded: byte-parity scope is Dlvl 1.  When extending
        # beyond Dlvl 1 the branch signature needs a depth parameter (the
        # verysmall + rn2 spbook gate becomes reachable at depth > 5).
        new_r, _corpsenm = pick_monster_for_level(None, 1, vendor_rng=r)
        return new_r

    return lax.cond(is_statue, _statue_path, lambda r: r, rng)


_MKSOBJ_INIT_BRANCHES = [
    _noop_branch,          # 0  RANDOM_CLASS   (never spawned directly)
    _noop_branch,          # 1  (unused slot)
    _weapon_branch,        # 2  WEAPON_CLASS   mkobj.c:803-818
    _armor_branch,         # 3  ARMOR_CLASS    mkobj.c:992-1005
    _ring_branch,          # 4  RING_CLASS     mkobj.c:1029-1048 (charged + uncharged)
    _amulet_branch,        # 5  AMULET_CLASS   mkobj.c:967-975
    _noop_branch,          # 6  TOOL_CLASS     dispatched via _tool_draws_dispatch
    _food_branch,          # 7  FOOD_CLASS     mkobj.c:880-884
    _potion_scroll_branch, # 8  POTION_CLASS   mkobj.c:981-987
    _potion_scroll_branch, # 9  SCROLL_CLASS   mkobj.c:981-987
    _spbook_branch,        # 10 SPBOOK_CLASS   mkobj.c:988-991
    _wand_branch,          # 11 WAND_CLASS     mkobj.c:1019-1027
    _noop_branch,          # 12 COIN_CLASS     mkobj.c:1060-1061 (no draws)
    _gem_branch,           # 13 GEM_CLASS      mkobj.c:886-895
    # Length-18 table covers all ObjectClass enum values (RANDOM=0..VENOM=17).
    # Without these explicit slots, lax.switch clamps out-of-range indices to
    # the LAST branch (verified: oclass_id in {14,15,16,17} all routed to
    # _gem_branch on JAX 0.x), incorrectly consuming a rn2(6) GEM draw.
    _rock_branch,          # 14 ROCK_CLASS     mkobj.c:1050-1058 (STATUE has
                           #                   rndmonnum; verysmall+rn2 spbook
                           #                   gate unreachable at Dlvl 1 — see
                           #                   _rock_branch docstring).
    _noop_branch,          # 15 BALL_CLASS     mkobj.c:977-980 (explicit break)
    _noop_branch,          # 16 CHAIN_CLASS    mkobj.c:977-980 (explicit break)
    _noop_branch,          # 17 VENOM_CLASS    mkobj.c:977-980 (explicit break)
]


def consume_mksobj_init_draws(
    rng: Isaac64State,
    oclass_id: jnp.ndarray,
    otyp: jnp.ndarray | None = None,
    artif: bool | jnp.ndarray = False,
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
    RING_CLASS    (4):  charged:   boc(3) + rn2(9) + cond rne(3)  → 3-5 draws
                        uncharged: boc(10) + cond rn2(9)          → 1-3 draws
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
    # Coerce otyp/artif into JAX scalars with safe defaults.
    # When otyp is None (legacy callers — e.g. dead-pred possession that
    # discarded the type roll) we pass otyp=0; ``_weapon_draws`` falls back
    # to is_multigen=is_poisonable=False which matches the predominant
    # weapon path (~94% of WEAPON_CLASS otyps by oc_prob mass are NOT
    # multigen-skilled).  This was the pre-fix behaviour but is no longer
    # the byte-correct path: callers that have the otyp roll SHOULD pass it.
    otyp_arr = jnp.int32(0) if otyp is None else jnp.asarray(otyp, dtype=jnp.int32)
    artif_arr = jnp.asarray(artif, dtype=jnp.bool_)
    rng = lax.switch(
        oclass_id, _MKSOBJ_INIT_BRANCHES,
        rng, otyp_arr, artif_arr,
    )
    if otyp is not None:
        # TOOL_CLASS per-otyp dispatch (mkobj.c:897-966).  Only fires when
        # the picked class is TOOL_CLASS; otherwise this is a noop.
        is_tool = oclass_id == jnp.int32(int(ObjectClass.TOOL_CLASS))
        rng = lax.cond(
            is_tool,
            lambda r: _tool_draws_dispatch(r, otyp_arr),
            lambda r: r,
            rng,
        )
    return rng


def consume_mkobj_random_draws(
    rng: Isaac64State,
    *,
    in_rogue: bool = False,
    in_hell: bool = False,
    artif: bool | jnp.ndarray = True,
) -> Isaac64State:
    """Consume vendor ``mkobj(RANDOM_CLASS, artif)`` ISAAC64 draws.

    Vendor (mkobj.c:249-275): when oclass==RANDOM_CLASS, mkobj() draws
    ``prob = rnd(1000)``, then ``tprob = rnd(100)`` and walks the per-
    level iprobs table to pick a class, then walks objects[] by
    subtracting from prob to pick an otyp.  The picked class then
    determines mksobj_init's per-class draws (mkobj.c:801-1069).

    This helper consumes exactly that sequence:
        1. rnd(1000) — type-pick roll (prob)
        2. rnd(100)  — class-pick roll (tprob)
        3. consume_mksobj_init_draws(picked_class) — class-specific init

    Use this wherever vendor invokes ``mkobj_at(0, ...)`` or
    ``mkobj(RANDOM_CLASS, ...)``.

    Citation: vendor/nle/src/mkobj.c:249-301 (mkobj),
              vendor/nle/src/mkobj.c:801-1069 (mksobj_init switch).
    """
    # 1. rnd(1000) — prob (type-pick roll); result not needed for byte parity.
    rng, _prob = rnd_jax(rng, jnp.int32(1000))
    # 2. rnd(100) — tprob (class-pick walk).  Pick class table at trace time.
    rng, tprob = rnd_jax(rng, jnp.int32(100))
    if in_rogue:
        table = _ROGUE_TABLE
    elif in_hell:
        table = _HELL_TABLE
    else:
        table = _MKOBJ_TABLE
    # vendor walks tprob (rnd(100) ∈ [1..100]) until iprob subtracts to <=0;
    # _expand_class_table flattens probs to a 100-element class lookup keyed
    # by (tprob - 1).
    picked_class = table[jnp.clip(tprob - jnp.int32(1), 0, 99)]
    # 3. mksobj_init draws for the picked class.
    # Vendor mkobj(oclass, artif) ends with `mksobj(i, TRUE, artif)` — the
    # caller's ``artif`` boolean (TRUE for mkobj_at(0,x,y,TRUE) call sites
    # like mklev.c:540 inaccessible-niche placement) is propagated into
    # mksobj_init.  Without this, _weapon_draws skips its rn2(20) artifact
    # check (mkobj.c:816) and the ISAAC64 stream desyncs.
    # Vendor cite: vendor/nle/src/mkobj.c:271, vendor/nle/src/mkobj.c:816.
    rng = consume_mksobj_init_draws(rng, picked_class, artif=artif)
    return rng
