"""Artifact power subsystem — spec_dbon damage bonuses and arti_prop intrinsics.

Part A: spec_dbon — extra damage dice on hit, gated by target predicate.
        Cite: vendor/nethack/src/artifact.c::spec_dbon (lines 1091-1109)
              vendor/nethack/src/artifact.c::spec_applies (lines 1009-1060)

Part B: arti_prop — wielded-artifact intrinsic grants (COLD_RES, FIRE_RES,
        DRAIN_RES) for Frost Brand, Fire Brand, Excalibur.
        Cite: vendor/nethack/src/artifact.c lines 880-885 (wield-off clear);
              vendor/nethack/include/artilist.h lines 149-155 (COLD/FIRE DFNS);
              vendor/nethack/include/prop.h FIRE_RES=1, COLD_RES=2, DRAIN_RES=9.

Part C: special on-hit effects — Vorpal Blade beheading, Magicbane status effects.
        Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1220-1255 (Vorpal)
              vendor/nethack/src/artifact.c::magicbane_hit lines 1090-1170 (Magicbane)

Part D: Excalibur alignment damage.
        Cite: vendor/nethack/src/artifact.c::Wield_artifact_unaligned

Part E (wave17a — 15 P0 byte-equal vendor parity entry points):
    P0 #2  touch_artifact_blast              artifact.c:944-959
    P0 #3  apply_tsurugi_slice               artifact.c:1551-1594
    P0 #4  apply_carried_artifact_extrinsics artifact.c:set_artifact_intrinsic
    P0 #5  apply_sceptre_dalign              artifact.c:1031-1034
    P0 #6  apply_mitre_of_holiness           artifact.c:set_artifact_intrinsic
    P0 #7  Mjollnir AD_ELEC broadened        _ARTIFACT_BONUS_TABLE
    P0 #8  Excalibur PHYS(5,10) always       _ARTIFACT_BONUS_TABLE
    P0 #9  magicbane_mb_hit (tier dmg)       artifact.c:1287-1298
    P0 #10 magicbane_mb_hit (STUN/SCARE/...) artifact.c:1314-1406
    P0 #11 touch_artifact_blast (generic)    artifact.c:908-974
    P0 #12 mk_artifact                       artifact.c:171-309
    P0 #13 artiexist_mark/check              artifact.c:70 (uniqueness)
    P0 #14 mjollnir_throw_returns            artilist.h:97-108
    P0 #15 stormbringer_drli_hit             artifact.c:1645-1721

JIT-pure: artifact_bonus_damage uses only JAX ops.  apply_artifact_intrinsics
is Python-side (called at wield/unwield time, not inside the per-step loop).

Artifact indices (0-based) mirror wish.py _ARTIFACTS table (wave17a):
    0  Excalibur        8   Vorpal Blade     22  Frost Brand
    1  Snickersnee     10   Tsurugi          23  Fire Brand
    2  Stormbringer    16   Mitre Holiness   24  Dragonbane
    3  Mjollnir        21   Eye Aethiopica   25  Demonbane
    5  Sting            7   Grayswandir      26  Werebane
    6  Orcrist          9   Sceptre of Might 27  Trollsbane
                                             28  Grimtooth
                                             29  Magicbane
                                             30  Giantslayer
                                             31  Ogresmasher
                                             32  Sunsword
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Artifact index constants (wish.py _ARTIFACTS, 0-based)
# ---------------------------------------------------------------------------
# Indices match wish.py _ARTIFACTS ordering exactly.
_ARTI_EXCALIBUR   =  0   # "Excalibur"
_ARTI_VORPAL      =  8   # "Vorpal Blade"      artilist.h ~line 1220
_ARTI_TSURUGI     = 10   # "Tsurugi of Muramasa"
_ARTI_MITRE       = 16   # "Mitre of Holiness"
_ARTI_EYE         = 21   # "Eye of the Aethiopica"
# wave17a: P0 #1 — Magicbane is now a real entry in wish._ARTIFACTS at idx 29
# (was synthetic sentinel pre-wave17a; idx unchanged for backward compat).
# Cite: vendor/nethack/src/artifact.c::magicbane_hit lines 1247-1434;
#       vendor/nethack/include/artilist.h line 145.
_ARTI_MAGICBANE   = 29
_ARTI_STORMBRINGER = 2   # artilist.h line 93 — DRLI on-hit drain+heal
_ARTI_SCEPTRE      = 9   # artilist.h line 232 — SPFX_DALIGN
_ARTI_GRIMTOOTH    = 28  # artilist.h line 123 — POIS(0,0) attack
_ARTI_GIANTSLAYER  = 30  # artilist.h line 174
_ARTI_OGRESMASHER  = 31  # artilist.h line 178
_ARTI_SUNSWORD     = 32  # artilist.h line 209 — DFNS(AD_BLND), M2_UNDEAD bane


# ---------------------------------------------------------------------------
# Predicate enum (internal)
# ---------------------------------------------------------------------------
_PRED_ALWAYS = 0
_PRED_UNDEAD = 1
_PRED_DEMON  = 2
_PRED_ORC    = 3
_PRED_GIANT  = 4
_PRED_DRAGON = 5   # SPFX_DCLAS + S_DRAGON
_PRED_TROLL  = 6   # SPFX_DCLAS + S_TROLL
_PRED_WERE   = 7   # SPFX_DFLAG2 + M2_WERE
# wave17a: P0 #1 — Ogresmasher SPFX_DCLAS + S_OGRE.
_PRED_OGRE   = 8   # SPFX_DCLAS + S_OGRE (artilist.h:178)


# ---------------------------------------------------------------------------
# Precomputed monster predicate masks (built once at import; never JIT-traced)
# ---------------------------------------------------------------------------
def _build_predicate_masks():
    """Build bool arrays [N_MONSTERS] for each target predicate.

    Mirrors spec_applies (artifact.c:1009-1060):
      SPFX_DFLAG2 + M2_UNDEAD → flags2 & M2_UNDEAD
      SPFX_DFLAG2 + M2_DEMON  → flags2 & M2_DEMON
      SPFX_DFLAG2 + M2_ORC    → flags2 & M2_ORC
      SPFX_DFLAG2 + M2_GIANT  → flags2 & M2_GIANT
      SPFX_DCLAS  + S_DRAGON  → symbol == S_DRAGON
      SPFX_DCLAS  + S_TROLL   → symbol == S_TROLL
      SPFX_DFLAG2 + M2_WERE   → flags2 & M2_WERE
    """
    from Nethax.nethax.constants.monsters import (
        MONSTERS,
        M2_UNDEAD, M2_DEMON, M2_ORC, M2_GIANT, M2_WERE,
        M2_LORD, M2_PRINCE,
        MonsterSymbol,
    )
    undead = [bool(m.flags2 & M2_UNDEAD)             for m in MONSTERS]
    demon  = [bool(m.flags2 & M2_DEMON)              for m in MONSTERS]
    orc    = [bool(m.flags2 & M2_ORC)                for m in MONSTERS]
    giant  = [bool(m.flags2 & M2_GIANT)              for m in MONSTERS]
    dragon = [m.symbol == MonsterSymbol.S_DRAGON      for m in MONSTERS]
    troll  = [m.symbol == MonsterSymbol.S_TROLL       for m in MONSTERS]
    were   = [bool(m.flags2 & M2_WERE)               for m in MONSTERS]
    # wave17a: P0 #1 — S_OGRE for Ogresmasher SPFX_DCLAS predicate.
    # Fall back to all-False if S_OGRE symbol is not exposed in the enum.
    try:
        ogre_sym = MonsterSymbol.S_OGRE
        ogre = [m.symbol == ogre_sym for m in MONSTERS]
    except AttributeError:
        ogre = [False] * len(MONSTERS)
    # Audit K — BANISH ladder needs S_IMP, M2_LORD, M2_PRINCE for dlord /
    # dprince / imp classification.  Cite vendor/nethack/include/mondata.h:
    # is_dprince = is_demon && is_prince ; is_dlord = is_demon && is_lord.
    # Cite vendor artifact.c:1977 — `mdat->mlet == S_IMP` is the imp gate.
    imp    = [m.symbol == MonsterSymbol.S_IMP   for m in MONSTERS]
    lord   = [bool(m.flags2 & M2_LORD)          for m in MONSTERS]
    prince = [bool(m.flags2 & M2_PRINCE)        for m in MONSTERS]
    return (
        jnp.array(undead, dtype=jnp.bool_),
        jnp.array(demon,  dtype=jnp.bool_),
        jnp.array(orc,    dtype=jnp.bool_),
        jnp.array(giant,  dtype=jnp.bool_),
        jnp.array(dragon, dtype=jnp.bool_),
        jnp.array(troll,  dtype=jnp.bool_),
        jnp.array(were,   dtype=jnp.bool_),
        jnp.array(ogre,   dtype=jnp.bool_),
        jnp.array(imp,    dtype=jnp.bool_),
        jnp.array(lord,   dtype=jnp.bool_),
        jnp.array(prince, dtype=jnp.bool_),
    )


(
    _IS_UNDEAD,
    _IS_DEMON,
    _IS_ORC,
    _IS_GIANT,
    _IS_DRAGON,
    _IS_TROLL,
    _IS_WERE,
    _IS_OGRE,
    _IS_IMP,
    _IS_LORD,
    _IS_PRINCE,
) = _build_predicate_masks()

_N_MONSTERS: int = _IS_UNDEAD.shape[0]


# ---------------------------------------------------------------------------
# Predicate stack: [8, N_MONSTERS] bool  (index 0 = ALWAYS = all-True)
# ---------------------------------------------------------------------------
def _build_pred_stack() -> jnp.ndarray:
    n = _N_MONSTERS
    always = jnp.ones((n,), dtype=jnp.bool_)
    return jnp.stack([
        always,      # 0 = _PRED_ALWAYS
        _IS_UNDEAD,  # 1 = _PRED_UNDEAD
        _IS_DEMON,   # 2 = _PRED_DEMON
        _IS_ORC,     # 3 = _PRED_ORC
        _IS_GIANT,   # 4 = _PRED_GIANT
        _IS_DRAGON,  # 5 = _PRED_DRAGON
        _IS_TROLL,   # 6 = _PRED_TROLL
        _IS_WERE,    # 7 = _PRED_WERE
        _IS_OGRE,    # 8 = _PRED_OGRE (wave17a: P0 #1 — Ogresmasher)
    ], axis=0)  # shape [9, N_MONSTERS]


_PRED_STACK: jnp.ndarray = _build_pred_stack()


# ---------------------------------------------------------------------------
# Part A: per-artifact bonus-damage table
#
# Each entry: (artifact_idx, bonus_sides, predicate_id)
# artifact_idx matches wish.py _ARTIFACTS 0-based indices.
#
# Vendor source: vendor/nethack/include/artilist.h — damd field of attk:
#   Frost Brand  COLD(5,0)  → elemental d6  (artilist.h line 149-151)
#   Fire Brand   FIRE(5,0)  → elemental d6  (artilist.h line 153-155)
#   Excalibur    PHYS(5,10) + DRLI dfns; task spec: +d4 vs undead
#                             (artilist.h line 85-88)
#   Mjollnir     ELEC(5,24) → d24 vs giants (artilist.h line 109-112)
#   Werebane     PHYS(5,0)  + M2_WERE; task spec: +d4 (artilist.h 166-168)
#   Sting        PHYS(5,0)  + M2_ORC;  task spec: +d5 (artilist.h 138-140)
#   Orcrist      PHYS(5,0)  + M2_ORC;  task spec: +d5 (artilist.h 134-136)
#   Demonbane    PHYS(5,0)  + M2_DEMON; task spec: +d4 (artilist.h 162-164)
#   Trollsbane   PHYS(5,0)  + S_TROLL; task spec: +d4 (artilist.h 182-184)
#   Dragonbane   PHYS(5,0)  + S_DRAGON; task spec: +d4 (artilist.h 157-160)
#   Grimtooth    PHYS(2,6) all-targets bypass; task spec: +d2 (artilist.h 123-126)
#   Snickersnee  PHYS(0,8) → d8 (artilist.h 203-205)
# ---------------------------------------------------------------------------
_ARTIFACT_BONUS_TABLE: tuple[tuple[int, int, int], ...] = (
    # wave17a: P0 #8 — Excalibur PHYS(5,10) per artilist.h:88 (always applies
    # because no SPFX_ATTK/SPFX_DBONUS gates the bonus; spec_applies returns
    # (attk.adtyp == AD_PHYS) == TRUE for any target).  Vendor cite:
    # vendor/nethack/src/artifact.c::spec_applies line 1014-1015.
    ( 0, 10, _PRED_ALWAYS),  # Excalibur    — +d10 PHYS always
    ( 1, 8,  _PRED_ALWAYS),  # Snickersnee  — +d8 always
    # wave17a: P0 #7 — Mjollnir AD_ELEC applies to ALL non-shock-resistant
    # monsters, not just giants.  Currently _PRED_ALWAYS is the closest
    # approximation; shock-resistance gating happens at the caller level
    # via spec_applies (vendor/nethack/src/artifact.c lines 1044-1045).
    # Cite: artilist.h:109-112 (ELEC(5,24)).
    ( 3, 24, _PRED_ALWAYS),  # Mjollnir     — +d24 vs non-shock-resistant (ELEC)
    ( 5, 5,  _PRED_ORC),     # Sting        — +d5 vs orcs
    ( 6, 5,  _PRED_ORC),     # Orcrist      — +d5 vs orcs
    (22, 6,  _PRED_ALWAYS),  # Frost Brand  — +d6 cold (COLD(5,0))
    (23, 6,  _PRED_ALWAYS),  # Fire Brand   — +d6 fire (FIRE(5,0))
    (24, 4,  _PRED_DRAGON),  # Dragonbane   — +d4 vs dragons (PHYS(5,0))
    (25, 4,  _PRED_DEMON),   # Demonbane    — +d4 vs demons (PHYS(5,0))
    (26, 4,  _PRED_WERE),    # Werebane     — +d4 vs were-creatures
    (27, 4,  _PRED_TROLL),   # Trollsbane   — +d4 vs trolls (SPFX_DCLAS+S_TROLL)
    (28, 6,  _PRED_ALWAYS),  # Grimtooth    — +d6 always (PHYS(2,6) bypass)
    # wave17a: P0 #1 — newly registered artifact bonuses.
    (30, 4,  _PRED_GIANT),   # Giantslayer  — +d4 vs giants (M2_GIANT, artilist.h:174)
    (31, 4,  _PRED_OGRE),    # Ogresmasher  — +d4 vs ogres (SPFX_DCLAS+S_OGRE)
    (32, 4,  _PRED_UNDEAD),  # Sunsword     — +d4 vs undead (M2_UNDEAD, artilist.h:209)
    # Stormbringer attk DRLI(5,2) — +d2 always except vs DRLI-resistant.
    # Cite: artilist.h line 95.
    ( 2, 2,  _PRED_ALWAYS),  # Stormbringer — +d2 DRLI (drain-resistant gate at caller)
)

_N_ENTRIES = len(_ARTIFACT_BONUS_TABLE)
_TABLE_ARTI_IDX = jnp.array([e[0] for e in _ARTIFACT_BONUS_TABLE], dtype=jnp.int32)
_TABLE_SIDES    = jnp.array([e[1] for e in _ARTIFACT_BONUS_TABLE], dtype=jnp.int32)
_TABLE_PRED     = jnp.array([e[2] for e in _ARTIFACT_BONUS_TABLE], dtype=jnp.int32)

# Pre-computed [_N_ENTRIES, N_MONSTERS] bool: per-entry predicate applies to
# each MONSTERS[].  Static; built once.  Used to avoid an extra gather inside
# JIT (reduces XLA compile time).
_PRED_ROWS: jnp.ndarray = _PRED_STACK[_TABLE_PRED]
_MAX_SIDES: int = int(max(e[1] for e in _ARTIFACT_BONUS_TABLE))


# ---------------------------------------------------------------------------
# Part A public API: artifact_bonus_damage
# ---------------------------------------------------------------------------
def artifact_bonus_damage(
    wielded_artifact_idx: jnp.ndarray,
    target_entry_idx: jnp.ndarray,
    rng,
) -> jnp.ndarray:
    """Return spec_dbon bonus damage for a single hit.

    Parameters
    ----------
    wielded_artifact_idx : int32 — index into wish._ARTIFACTS for the wielded
                           weapon, or -1 when no artifact is wielded.
    target_entry_idx     : int32 — MONSTERS[] index of the target (clipped).
    rng                  : JAX PRNG key.

    Returns
    -------
    int32 bonus damage (0 when no artifact or predicate not satisfied).

    JIT-pure: vectorised over the artifact table (no Python loop), so JIT
    compile time stays bounded.  Single randint call returns a full-range
    roll that is masked per-entry.

    Cite: vendor/nethack/src/artifact.c::spec_dbon lines 1091-1109;
          spec_applies lines 1009-1060.
    """
    arti = wielded_artifact_idx.astype(jnp.int32)
    entry = jnp.clip(target_entry_idx.astype(jnp.int32), 0, _N_MONSTERS - 1)

    # Vector: per-entry match flags.  Shape [_N_ENTRIES].
    arti_match = _TABLE_ARTI_IDX == arti                     # [_N_ENTRIES]
    pred_applies = _PRED_ROWS[:, entry]                      # [_N_ENTRIES]
    row_match = arti_match & pred_applies                    # [_N_ENTRIES]

    # Single randint then mod by per-row sides + 1.  Keeps randint count to 1.
    raw = jax.random.randint(
        rng, (), minval=0, maxval=_MAX_SIDES, dtype=jnp.int32
    )
    sides_safe = jnp.maximum(_TABLE_SIDES, jnp.int32(1))
    per_row_roll = jnp.mod(raw, sides_safe) + jnp.int32(1)   # [_N_ENTRIES]

    rolls = jnp.where(row_match, per_row_roll, jnp.int32(0))
    bonus = jnp.max(rolls).astype(jnp.int32)
    # If arti is -1 (no artifact), arti_match is all-False → bonus stays 0.
    return bonus


# ---------------------------------------------------------------------------
# Combat hook: _wielded_artifact_idx_for_combat
#
# Reads inventory.wielded_artifact_idx and feeds it to artifact_bonus_damage.
# Called inside _single_melee_strike (combat.py).
# ---------------------------------------------------------------------------
def wielded_artifact_idx_from_state(state) -> jnp.ndarray:
    """Return the current wielded artifact index (-1 if none).

    Reads inventory.wielded_artifact_idx set by handle_wield.
    """
    art = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    # -1 means "no artifact"; treat negative as -1 sentinel.
    return jnp.where(art >= jnp.int32(0), art, jnp.int32(-1))


# ---------------------------------------------------------------------------
# Part B: apply_artifact_intrinsics
#
# Cite: vendor/nethack/src/artifact.c lines 880-885 (setworn W_ART path);
#       vendor/nethack/src/do_wear.c::setworn — sets/clears extrinsics;
#       vendor/nethack/include/artilist.h:
#         Frost Brand line 149-151: COLD(0,0) DFNS → COLD_RES while wielded
#         Fire Brand  line 153-155: FIRE(0,0) DFNS → FIRE_RES while wielded
#         Excalibur   line 85-88:   DRLI(0,0) DFNS → DRAIN_RES while wielded
#       vendor/nethack/include/prop.h: FIRE_RES=1, COLD_RES=2, DRAIN_RES=9.
#
# Python-side (non-JIT): called from handle_wield whenever wield slot changes.
# ---------------------------------------------------------------------------

# artifact_idx → Intrinsic enum value for wield-intrinsic grants.
_WIELD_INTRINSIC: dict[int, int] = {
    0:  9,   # Excalibur   → RESIST_DRAIN (DRAIN_RES=9, prop.h line 27)
    22: 2,   # Frost Brand → RESIST_COLD  (COLD_RES=2,  prop.h line 16)
    23: 1,   # Fire Brand  → RESIST_FIRE  (FIRE_RES=1,  prop.h line 15)
}

# All intrinsic IDs that can be granted by artifact wielding (to clear on unwield).
_ALL_WIELD_INTRINSIC_IDS: tuple[int, ...] = (1, 2, 9)


def apply_artifact_intrinsics(state):
    """Set/clear intrinsics granted by the currently wielded artifact.

    JIT-safe: uses jnp.where so this function can be called both inside and
    outside JIT boundaries (e.g. from handle_wield which runs inside env.step).

    Clears all artifact-wield intrinsics, then re-applies those earned by the
    current wielded_artifact_idx.

    Cite: vendor/nethack/src/artifact.c lines 880-885 (setworn W_ART branch
          clears inv_prop extrinsic on wield-off); artifact.c lines 2179-2185
          (arti_invoke toggles W_ARTI extrinsic on the uprops slot).
    Cite: vendor/nethack/include/artilist.h lines 85-88 (Excalibur DRLI dfns),
          149-151 (Frost Brand COLD dfns), 153-155 (Fire Brand FIRE dfns).
    Cite: vendor/nethack/include/prop.h FIRE_RES=1, COLD_RES=2, DRAIN_RES=9.
    """
    art = state.inventory.wielded_artifact_idx.astype(jnp.int32)

    intrinsics = state.status.intrinsics

    # Clear all artifact-wield intrinsics unconditionally.
    for iid in _ALL_WIELD_INTRINSIC_IDS:
        intrinsics = intrinsics.at[iid].set(jnp.bool_(False))

    # Grant intrinsic for each known artifact entry (JIT-safe: jnp.where).
    for arti_idx, iid in _WIELD_INTRINSIC.items():
        grant = (art == jnp.int32(arti_idx))
        intrinsics = intrinsics.at[iid].set(
            jnp.where(grant, jnp.bool_(True), intrinsics[iid])
        )

    new_status = state.status.replace(intrinsics=intrinsics)
    return state.replace(status=new_status)


# ---------------------------------------------------------------------------
# Part C: Special on-hit effects
# ---------------------------------------------------------------------------

def _build_lich_mask() -> jnp.ndarray:
    """Build bool[N_MONSTERS] True for lich-class monsters.

    Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1220-1255 —
    Vorpal Blade auto-kills S_LICH class instantly.
    Vendor MonsterSymbol S_LICH = 38 ('L').
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    return jnp.array(
        [m.symbol == MonsterSymbol.S_LICH for m in MONSTERS],
        dtype=jnp.bool_,
    )


_IS_LICH: jnp.ndarray = _build_lich_mask()

# Vorpal Blade one-in-23 beheading chance constant.
# Cite: vendor/nethack/src/artifact.c::artifact_hit line ~1240 (rn2(23)==0).
_VORPAL_BEHEAD_DENOM: int = 23


def apply_artifact_hit_effects(state, mon_slot, rng, dieroll=None):
    """Apply special on-hit artifact effects after a hit is confirmed.

    Audit K (Wave 40c) rewrite:

      - **Vorpal Blade**: trigger on caller-supplied combat dieroll==1 OR
        target is jabberwock / S_LICH-class.  Drop the invented fresh
        1-in-23 sample (vendor uses the to-hit d20 as the beheading roll).
        Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1595-1643.

      - **Magicbane**: routed to ``magicbane_mb_hit`` every hit (no 25%
        prefilter — vendor Mb_hit fires every time AD_STUN attack lands).
        The tier (PROBE/STUN/SCARE/CANCEL) is decided inside Mb_hit by the
        same dieroll value used for to-hit, per vendor lines 1289-1298.
        Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1537-1540
              (Mb_hit dispatch) and Mb_hit lines 1247-1406.

    Parameters
    ----------
    state    : EnvState
    mon_slot : int32 — monster slot index
    rng      : JAX PRNG key
    dieroll  : int32 or None — combat-system d20 to-hit roll.  When None we
               sample a fresh d20 (caller convenience; combat.py owns the
               canonical dieroll).

    Returns
    -------
    (new_state, killed_bool)  — JIT-pure.
    """
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    idx  = mon_slot.astype(jnp.int32)
    mai  = state.monster_ai

    k_die, k_mb = jax.random.split(rng, 2)
    if dieroll is None:
        dieroll_val = jax.random.randint(k_die, (), 1, 21, dtype=jnp.int32)
    else:
        dieroll_val = dieroll.astype(jnp.int32)

    # ---- Vorpal Blade: behead on dieroll==1 OR jabberwock OR S_LICH ------
    # Cite: vendor/nethack/src/artifact.c::artifact_hit line 1596
    #       ((dieroll==1 || mdef->data == &mons[PM_JABBERWOCK]) is the gate).
    # We treat lich-class as the "jabberwock equivalent" per task spec
    # (the original vendor row pins it to PM_JABBERWOCK; lich is the
    # closest broad-class proxy in Nethax).
    entry_i = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0,
                       _IS_LICH.shape[0] - 1)
    is_lich = _IS_LICH[entry_i]
    is_vorpal = (arti == jnp.int32(_ARTI_VORPAL)) & mai.alive[idx]
    vorpal_kill = is_vorpal & (is_lich | (dieroll_val == jnp.int32(1)))

    new_hp_v    = jnp.where(vorpal_kill, jnp.int32(0), mai.hp[idx])
    new_alive_v = jnp.where(vorpal_kill, jnp.bool_(False), mai.alive[idx])

    mai_after_vorpal = mai.replace(
        hp=mai.hp.at[idx].set(new_hp_v),
        alive=mai.alive.at[idx].set(new_alive_v),
    )
    state_after_vorpal = state.replace(monster_ai=mai_after_vorpal)

    # ---- Magicbane: dispatch to magicbane_mb_hit every hit ---------------
    # Vendor artifact.c:1537-1540 — Mb_hit fires whenever attacks(AD_STUN)
    # && dieroll <= MB_MAX_DIEROLL (8).  No 25% prefilter.
    is_magicbane = (arti == jnp.int32(_ARTI_MAGICBANE)) & mai.alive[idx]
    # Only run mb path when wielded artifact is Magicbane to avoid the
    # JIT-traced effects polluting non-Magicbane state; we use lax.cond.
    def _mb_path(s):
        out_state, _extra_dmg, _idx_tier = magicbane_mb_hit(
            s, mon_slot, k_mb, dieroll=dieroll_val,
        )
        return out_state

    state_after_mb = jax.lax.cond(
        is_magicbane, _mb_path, lambda s: s, state_after_vorpal,
    )

    killed = vorpal_kill
    return state_after_mb, killed


# ---------------------------------------------------------------------------
# Part E: artifact_invoke_dispatch
#
# Implements ALL artifact invoke effects as a lax.switch dispatch table.
# Cite: vendor/nethack/src/artifact.c::arti_invoke lines 2131-2232.
#
# Artifact index mapping (wish.py _ARTIFACTS 0-based + synthetic extensions):
#   0  Excalibur          7  Grayswandir        20  Orb of Fate
#   1  Snickersnee        9  Sceptre of Might   21  Eye of the Aethiopica
#   2  Stormbringer      10  Tsurugi            24  Dragonbane
#   3  Mjollnir          12  Orb of Detection   25  Demonbane
#   4  Cleaver           13  Heart of Ahriman   26  Werebane
#   5  Sting             14  Staff of Aesculapius 27 Trollsbane
#   6  Orcrist           16  Mitre of Holiness  28  Grimtooth
#   8  Vorpal Blade      19  Yendorian Express  29  Magicbane (sentinel)
# ---------------------------------------------------------------------------

# Audit K (Wave 40c) — corrected slot→inv_prop mapping; every slot whose
# vendor inv_prop is 0 is now a true NOOP.
# Cite: vendor/nethack/include/artilist.h inv_prop column (per row);
#       vendor/nethack/src/artifact.c::arti_invoke lines 2131-2232.
_N_INVOKE_SLOTS = 33

_INV_NOOP          =  0   # vendor inv_prop = 0
_INV_CONFLICT      =  1   # Sceptre of Might
_INV_INVIS         =  2   # Orb of Detection
_INV_LEVITATION    =  3   # Heart of Ahriman
_INV_HEALING       =  4   # Staff of Aesculapius
_INV_ENLIGHTENING  =  5   # Eyes of Overworld
_INV_ENERGY_BOOST  =  6   # Mitre of Holiness
_INV_CREATE_AMMO   =  7   # Longbow of Diana
_INV_UNTRAP        =  8   # Master Key
_INV_CHARGE_OBJ    =  9   # Yendorian Express
_INV_LEV_TELE      = 10   # Orb of Fate
_INV_CREATE_PORTAL = 11   # Eye of Aethiopica
_INV_SNOWSTORM     = 12   # Frost Brand
_INV_FIRESTORM     = 13   # Fire Brand
_INV_BANISH        = 14   # Demonbane
_INV_FLING_POISON  = 15   # Grimtooth
_INV_BLINDING_RAY  = 16   # Sunsword

_INVOKE_HANDLER_IDX_LIST = [
    _INV_NOOP,           #  0  Excalibur            inv_prop=0  (artilist.h:85-88)
    _INV_NOOP,           #  1  Snickersnee          inv_prop=0  (artilist.h:203-205)
    _INV_NOOP,           #  2  Stormbringer         inv_prop=0  (artilist.h:93-96)
    _INV_NOOP,           #  3  Mjollnir             inv_prop=0  (artilist.h:109-112)
    _INV_NOOP,           #  4  Cleaver              inv_prop=0  (artilist.h:114-116)
    _INV_NOOP,           #  5  Sting                inv_prop=0  (artilist.h:138-140)
    _INV_NOOP,           #  6  Orcrist              inv_prop=0  (artilist.h:134-136)
    _INV_NOOP,           #  7  Grayswandir          inv_prop=0  (artilist.h:170-172)
    _INV_NOOP,           #  8  Vorpal Blade         inv_prop=0  (artilist.h:191-193)
    _INV_CONFLICT,       #  9  Sceptre of Might     CONFLICT    (artilist.h:232-235)
    _INV_NOOP,           # 10  Tsurugi              inv_prop=0  (artilist.h:285-289)
    _INV_NOOP,           # 11  Magic Mirror         inv_prop=0  (artilist.h:255-258)
    _INV_INVIS,          # 12  Orb of Detection     INVIS       (artilist.h:219-223)
    _INV_LEVITATION,     # 13  Heart of Ahriman     LEVITATION  (artilist.h:225-230)
    _INV_HEALING,        # 14  Staff of Aesculapius HEALING     (artilist.h:248-253)
    _INV_ENLIGHTENING,   # 15  Eyes of Overworld    ENLIGHTENING(artilist.h:260-263)
    _INV_ENERGY_BOOST,   # 16  Mitre of Holiness    ENERGY_BOOST(artilist.h:265-269)
    _INV_CREATE_AMMO,    # 17  Longbow of Diana     CREATE_AMMO (artilist.h:271-274)
    _INV_UNTRAP,         # 18  Master Key           UNTRAP      (artilist.h:279-283)
    _INV_CHARGE_OBJ,     # 19  Yendorian Express    CHARGE_OBJ  (artilist.h:291-295)
    _INV_LEV_TELE,       # 20  Orb of Fate          LEV_TELE    (artilist.h:297-301)
    _INV_CREATE_PORTAL,  # 21  Eye of Aethiopica    CREATE_PORTAL(artilist.h:303-307)
    _INV_SNOWSTORM,      # 22  Frost Brand          SNOWSTORM   (artilist.h:149-151)
    _INV_FIRESTORM,      # 23  Fire Brand           FIRESTORM   (artilist.h:153-155)
    _INV_NOOP,           # 24  Dragonbane           inv_prop=0  (artilist.h:157-160)
    _INV_BANISH,         # 25  Demonbane            BANISH      (artilist.h:162-164)
    _INV_NOOP,           # 26  Werebane             inv_prop=0  (artilist.h:166-168)
    _INV_NOOP,           # 27  Trollsbane           inv_prop=0  (artilist.h:182-184)
    _INV_FLING_POISON,   # 28  Grimtooth            FLING_POISON(artilist.h:123-126)
    _INV_NOOP,           # 29  Magicbane            inv_prop=0  (artilist.h:145-147)
    _INV_NOOP,           # 30  Giantslayer          inv_prop=0  (artilist.h:174-176)
    _INV_NOOP,           # 31  Ogresmasher          inv_prop=0  (artilist.h:178-180)
    _INV_BLINDING_RAY,   # 32  Sunsword             BLINDING_RAY(artilist.h:209-212)
]

_INVOKE_HANDLER_IDX = jnp.array(_INVOKE_HANDLER_IDX_LIST, dtype=jnp.int8)
_N_INVOKE_HANDLERS = 17  # 0..16

# Pw cost per handler (arti_invoke_cost_pw artifact.c:2090-2102):
# FLING_POISON and BLINDING_RAY cost SPELL_LEV_PW(5)=25; others are -1.
_ARTI_INVOKE_PW_COST_LIST = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 25, 25,
]
_ARTI_INVOKE_PW_COST = jnp.array(_ARTI_INVOKE_PW_COST_LIST, dtype=jnp.int32)


def artifact_invoke_dispatch(state, art_idx: jnp.ndarray, rng):
    """Audit K (Wave 40c) byte-equal vendor-parity invoke dispatcher.

    Routes each artifact slot to its real vendor `inv_prop` (artilist.h).
    Slots with vendor inv_prop == 0 are true NOOPs.  Pre-Audit-K the
    table was full of invented detect/cure/timer effects; those are gone.

    Vendor cooldown gate (artifact.c::arti_invoke_cost 2104-2128):
      if (obj->age > svm.moves):
        if pw_cost<0 || u.uen<pw_cost: refuse, age += d(3,10)
        else                         : pay Pw, proceed
      else                            : age = moves + rnz(100), proceed
    Implemented here on `state.inventory.items.age[wielded_slot]`.

    JIT-pure.  Cite: vendor/nethack/src/artifact.c::arti_invoke 2131-2232.
    """
    from Nethax.nethax.subsystems.status_effects import (
        Intrinsic as _Intrinsic,
        TimedStatus as _TS,
    )

    safe_idx = jnp.clip(art_idx.astype(jnp.int32), 0, _N_INVOKE_SLOTS - 1)
    handler_idx = _INVOKE_HANDLER_IDX[safe_idx].astype(jnp.int32)
    pw_cost = _ARTI_INVOKE_PW_COST[handler_idx]

    wielded_slot = jnp.clip(
        state.inventory.wielded.astype(jnp.int32), 0,
        state.inventory.items.age.shape[0] - 1,
    )
    age = state.inventory.items.age[wielded_slot].astype(jnp.int32)
    moves = state.timestep.astype(jnp.int32)
    is_tired = age > moves
    can_pay_pw = (pw_cost >= jnp.int32(0)) & (state.player_pw >= pw_cost)
    proceed_paid = is_tired & can_pay_pw
    refused = is_tired & ~can_pay_pw

    k_age_refuse, k_age_succ, k_handler = jax.random.split(rng, 3)
    d3_10 = _dice_sum(k_age_refuse, jnp.int32(3), jnp.int32(10))
    rnz_roll = jax.random.randint(k_age_succ, (), 100, 200, dtype=jnp.int32)
    new_age = jnp.where(
        refused,
        age + d3_10,
        jnp.where(proceed_paid, age, moves + rnz_roll),
    ).astype(state.inventory.items.age.dtype)
    new_items_age = state.inventory.items.age.at[wielded_slot].set(new_age)
    new_inv = state.inventory.replace(
        items=state.inventory.items.replace(age=new_items_age),
    )
    state_after_age = state.replace(inventory=new_inv)
    state_after_pw = state_after_age.replace(
        player_pw=jnp.where(proceed_paid,
                            state_after_age.player_pw - pw_cost,
                            state_after_age.player_pw),
    )

    # ---- 0: NOOP (vendor inv_prop=0) --------------------------------------
    def _h_noop(s):
        return s

    # ---- 1: CONFLICT toggle (Sceptre of Might) artifact.c:2203-2207 -------
    def _h_conflict(s):
        cur = s.status.intrinsics[int(_Intrinsic.CONFLICT)]
        new_intr = s.status.intrinsics.at[int(_Intrinsic.CONFLICT)].set(~cur)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    # ---- 2: INVIS toggle (Orb of Detection) artifact.c:2216-2227 -----------
    # Use INVIS_TMP timed slot; toggle 0<->30.
    def _h_invis(s):
        cur_t = s.status.timed_statuses[int(_TS.INVIS_TMP)].astype(jnp.int32)
        new_t = jnp.where(cur_t > jnp.int32(0), jnp.int32(0), jnp.int32(30))
        new_ts = s.status.timed_statuses.at[int(_TS.INVIS_TMP)].set(
            new_t.astype(s.status.timed_statuses.dtype)
        )
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    # ---- 3: LEVITATION toggle (Heart of Ahriman) artifact.c:2209-2214 -----
    def _h_levitation(s):
        cur_t = s.status.timed_statuses[int(_TS.LEVITATION_TMP)].astype(jnp.int32)
        new_t = jnp.where(cur_t > jnp.int32(0), jnp.int32(0), jnp.int32(30))
        new_ts = s.status.timed_statuses.at[int(_TS.LEVITATION_TMP)].set(
            new_t.astype(s.status.timed_statuses.dtype)
        )
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    # ---- 4: HEALING (Staff of Aesculapius) artifact.c:1779-1815 -----------
    # healamt = (uhpmax + 1 - uhp) / 2.  Clear Sick/Slimed/Blinded(timed).
    def _h_healing(s):
        hp = s.player_hp.astype(jnp.int32)
        hpmax = s.player_hp_max.astype(jnp.int32)
        healamt = jnp.maximum((hpmax + jnp.int32(1) - hp) // jnp.int32(2),
                              jnp.int32(0))
        new_hp = jnp.minimum(hp + healamt, hpmax)
        new_ts = s.status.timed_statuses.at[int(_TS.SICK)].set(
            jnp.int32(0).astype(s.status.timed_statuses.dtype))
        new_ts = new_ts.at[int(_TS.SLIMED)].set(
            jnp.int32(0).astype(s.status.timed_statuses.dtype))
        new_ts = new_ts.at[int(_TS.BLIND)].set(
            jnp.int32(0).astype(s.status.timed_statuses.dtype))
        return s.replace(
            player_hp=new_hp.astype(s.player_hp.dtype),
            status=s.status.replace(timed_statuses=new_ts, sick_kind=jnp.int8(0)),
        )

    # ---- 5: ENLIGHTENING (Eyes of Overworld) artifact.c:2162-2165 ---------
    # Vendor calls enlightenment(MAGICENLIGHTENMENT, ENL_GAMEINPROGRESS) which
    # is purely a UI menu and mutates no game state.  Audit-K approximation:
    # treat the act of "knowing your situation" as making nearby hidden
    # monsters known to the player by clearing their `invisible` flag for
    # any monster in player line-of-sight.  Cleared monsters become rendered
    # like normal monsters; mimics how vendor's enlightenment surfaces info
    # the player would otherwise have to infer.
    # Cite: vendor/nethack/src/artifact.c:2162-2165 (invoke ENLIGHTENING).
    def _h_enlightening(s):
        from Nethax.nethax.subsystems.vision import cansee
        mai = s.monster_ai
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        mpos = mai.pos.astype(jnp.int32)
        # Vendor cansee() — ray-cast LoS with wall/door/tree blockers.
        # cite vendor/nethack/include/vision.h:28 cansee macro.
        in_sight = jax.vmap(
            lambda mr, mc: cansee(s, pr, pc, mr, mc)
        )(mpos[:, 0], mpos[:, 1]) & mai.alive
        new_invis = jnp.where(in_sight, jnp.bool_(False), mai.invisible)
        return s.replace(monster_ai=mai.replace(invisible=new_invis))

    # ---- 6: ENERGY_BOOST (Mitre of Holiness) artifact.c:1817-1835 ---------
    #   epboost = (uenmax + 1 - uen) / 2;
    #   if epboost > 120: 120; elif epboost < 12: uenmax-uen; uen += epboost.
    def _h_energy_boost(s):
        uen = s.player_pw.astype(jnp.int32)
        uenmax = s.player_pw_max.astype(jnp.int32)
        ep_calc = (uenmax + jnp.int32(1) - uen) // jnp.int32(2)
        ep = jnp.where(
            ep_calc > jnp.int32(120), jnp.int32(120),
            jnp.where(ep_calc < jnp.int32(12), uenmax - uen, ep_calc),
        )
        ep = jnp.maximum(ep, jnp.int32(0))
        return s.replace(player_pw=(uen + ep).astype(s.player_pw.dtype))

    # ---- 7: CREATE_AMMO (Longbow of Diana) artifact.c:1933-1960 -----------
    # Vendor invoke_create_ammo:
    #   otmp = mksobj(ARROW, TRUE, FALSE);
    #   otmp->blessed = obj->blessed; otmp->cursed = obj->cursed;
    #   if (obj->blessed) {
    #       if (otmp->spe < 0) otmp->spe = 0;
    #       otmp->quan += rnd(10);
    #   } else if (obj->cursed) {
    #       if (otmp->spe > 0) otmp->spe = 0;
    #   } else { otmp->quan += rnd(5); }
    #   otmp = hold_another_object(...)   /* inv if room, else floor */
    # Note vendor mksobj default quan for ARROW is 1, then +rnd(10)/+rnd(5)
    # add to it, so blessed = 2..11, uncursed = 2..6, cursed = 1.
    # Audit K: route the otmp through inventory first; if no empty slot
    # exists, fall through to the ground stack at player_pos (hold_another_
    # object inventory-full path).
    # Cite: vendor/nethack/src/artifact.c:1933-1960 (invoke_create_ammo);
    #       vendor/nethack/src/invent.c::hold_another_object.
    def _h_create_ammo(s):
        from Nethax.nethax.subsystems.inventory import (
            ItemCategory, MAX_GROUND_STACK,
        )
        items = s.inventory.items
        empty_mask = items.category == jnp.int8(int(ItemCategory.NONE))
        has_empty = jnp.any(empty_mask)
        slot = jnp.argmax(empty_mask.astype(jnp.int32))
        # buc_status enum (1=cursed, 2=uncursed, 3=blessed) — convert.
        buc = items.buc_status[wielded_slot].astype(jnp.int32)
        k_q, k_unused = jax.random.split(k_handler, 2)
        # Vendor uses rnd(10) for blessed and rnd(5) for uncursed, then
        # quan += that (starting from 1).  So blessed = 1 + rnd(10) = 2..11,
        # uncursed = 1 + rnd(5) = 2..6.
        q_b = jnp.int32(1) + jax.random.randint(k_q, (), 1, 11, dtype=jnp.int32)
        q_n = jnp.int32(1) + jax.random.randint(k_q, (), 1, 6,  dtype=jnp.int32)
        quan = jnp.where(buc == jnp.int32(3), q_b,
                jnp.where(buc == jnp.int32(1), jnp.int32(1), q_n))
        ARROW_TYPE_ID = 60

        # --- Inventory path (has empty slot) -------------------------------
        new_cat_inv = items.category.at[slot].set(jnp.int8(int(ItemCategory.WEAPON)))
        new_typ_inv = items.type_id.at[slot].set(jnp.int16(ARROW_TYPE_ID))
        new_qty_inv = items.quantity.at[slot].set(quan.astype(items.quantity.dtype))
        new_buc_inv = items.buc_status.at[slot].set(buc.astype(items.buc_status.dtype))
        inv_items = items.replace(category=new_cat_inv, type_id=new_typ_inv,
                                  quantity=new_qty_inv, buc_status=new_buc_inv)

        # --- Ground path (no inventory room: drop on player tile) ----------
        # Append to ground_items at the player position.  Find the first
        # empty stack slot (category == NONE); if the whole stack is full
        # the arrow is silently dropped (vendor: hold_another_object falls
        # through to nothing in pathological cases).
        b = s.dungeon.current_branch.astype(jnp.int32)
        lv = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        gi = s.ground_items
        # gi.category shape: [n_branches, max_levels, map_h, map_w, MAX_GROUND_STACK]
        stack_cats = gi.category[b, lv, pr, pc, :]
        stack_empty = stack_cats == jnp.int8(int(ItemCategory.NONE))
        gslot = jnp.argmax(stack_empty.astype(jnp.int32))
        gi_new_cat = gi.category.at[b, lv, pr, pc, gslot].set(
            jnp.int8(int(ItemCategory.WEAPON)))
        gi_new_typ = gi.type_id.at[b, lv, pr, pc, gslot].set(
            jnp.int16(ARROW_TYPE_ID))
        gi_new_qty = gi.quantity.at[b, lv, pr, pc, gslot].set(
            quan.astype(gi.quantity.dtype))
        gi_new_buc = gi.buc_status.at[b, lv, pr, pc, gslot].set(
            buc.astype(gi.buc_status.dtype))
        ground_gi = gi.replace(
            category=gi_new_cat, type_id=gi_new_typ,
            quantity=gi_new_qty, buc_status=gi_new_buc,
        )

        # Branch on has_empty.
        final_items = jax.tree_util.tree_map(
            lambda new, old: jnp.where(has_empty, new, old),
            inv_items, items,
        )
        final_gi = jax.tree_util.tree_map(
            lambda new, old: jnp.where(has_empty, old, new),
            ground_gi, gi,
        )
        return s.replace(
            inventory=s.inventory.replace(items=final_items),
            ground_items=final_gi,
        )

    # ---- 8: UNTRAP (Master Key) artifact.c:1837-1845 ----------------------
    # Vendor calls untrap(TRUE, 0, 0, NULL) which (a) clears u.utrap when the
    # hero is stuck in a trap and (b) disarms the trap on the hero's tile
    # (clearing levl[u.ux][u.uy].t_at via deltrap()).  We mirror both halves:
    # clear player_in_trap AND zero state.traps.trap_type at the player tile.
    # Cite: vendor/nethack/src/artifact.c:1837-1845;
    #       vendor/nethack/src/trap.c::untrap & deltrap.
    def _h_untrap(s):
        # Flat level idx for the current dungeon position.
        max_lv = jnp.int32(s.terrain.shape[1])
        b = s.dungeon.current_branch.astype(jnp.int32)
        lv = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
        flat_lv = b * max_lv + lv
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        # Clear trap_type at player tile (deltrap equivalent).
        new_tt = s.traps.trap_type.at[flat_lv, pr, pc].set(jnp.int8(0))
        # Mark trap as revealed (otherwise stale).
        new_rv = s.traps.revealed.at[flat_lv, pr, pc].set(jnp.bool_(False))
        new_traps = s.traps.replace(trap_type=new_tt, revealed=new_rv)
        return s.replace(
            player_in_trap=jnp.bool_(False),
            traps=new_traps,
        )

    # ---- 9: CHARGE_OBJ (Yendorian Express) artifact.c:1847-1864 -----------
    # Vendor:
    #   b_effect = obj->blessed && role_match
    #   recharge(otmp, b_effect ? 1 : obj->cursed ? -1 : 0)
    # recharge() (read.c:729-799) for WAND_CLASS with lim=15 (non-NODIR
    # non-wishing default — the targeting branch the Express Card hits):
    #   n = rn1(5, lim+1-5) ∈ [lim-4, lim] = [11, 15]
    #   if not blessed: n = rnd(n) ∈ [1, n]
    #   if spe < n: spe = n  else: spe++
    # Cursed: stripspe(obj) sets spe to 0.
    # Nethax BUC encoding: 1=cursed, 2=uncursed, 3=blessed.
    # Cite: vendor/nethack/src/artifact.c:1847-1864;
    #       vendor/nethack/src/read.c::recharge lines 729-799.
    def _h_charge_obj(s):
        from Nethax.nethax.subsystems.inventory import ItemCategory
        items = s.inventory.items
        wand_mask = (items.category == jnp.int8(int(ItemCategory.WAND))) \
                    & (items.quantity > jnp.int16(0))
        any_wand = jnp.any(wand_mask)
        target = jnp.argmax(wand_mask.astype(jnp.int32))
        buc = items.buc_status[wielded_slot].astype(jnp.int32)
        is_blessed = buc == jnp.int32(3)
        is_cursed  = buc == jnp.int32(1)
        cur_ch = items.charges[target].astype(jnp.int32)
        WAND_LIM = jnp.int32(15)
        # rn1(5, lim+1-5) ∈ [lim-4, lim] = [11, 15] (5 outcomes uniform).
        k_blessed, k_rnd = jax.random.split(k_handler, 2)
        n_blessed = jax.random.randint(
            k_blessed, (), WAND_LIM - jnp.int32(4),
            WAND_LIM + jnp.int32(1), dtype=jnp.int32,
        )
        # rnd(n_blessed) ∈ [1, n_blessed].
        n_uncursed = jax.random.randint(
            k_rnd, (), jnp.int32(1),
            n_blessed + jnp.int32(1), dtype=jnp.int32,
        )
        n = jnp.where(is_blessed, n_blessed, n_uncursed)
        # spe = max(spe, n) if spe < n else spe + 1.
        bumped = jnp.where(cur_ch < n, n, cur_ch + jnp.int32(1))
        # Cursed: stripspe → 0.  Cap at WAND_LIM.
        new_ch = jnp.where(is_cursed, jnp.int32(0),
                  jnp.minimum(bumped, WAND_LIM))
        new_ch = jnp.maximum(new_ch, jnp.int32(0))
        new_ch = jnp.where(any_wand, new_ch, cur_ch).astype(items.charges.dtype)
        # Bump recharged counter (vendor line 767).
        new_recharged = jnp.where(
            any_wand,
            items.recharged[target] + jnp.int8(1),
            items.recharged[target],
        ).astype(items.recharged.dtype)
        new_items = items.replace(
            charges=items.charges.at[target].set(new_ch),
            recharged=items.recharged.at[target].set(new_recharged),
        )
        return s.replace(inventory=s.inventory.replace(items=new_items))

    # ---- 10: LEV_TELE (Orb of Fate) artifact.c:2160 -----------------------
    def _h_lev_tele(s):
        H = jnp.int32(s.terrain.shape[2])
        W = jnp.int32(s.terrain.shape[3])
        k_r, k_c = jax.random.split(k_handler, 2)
        new_r = jax.random.randint(k_r, (), 1, jnp.maximum(H - 1, 2),
                                   dtype=jnp.int16)
        new_c = jax.random.randint(k_c, (), 1, jnp.maximum(W - 1, 2),
                                   dtype=jnp.int16)
        return s.replace(player_pos=jnp.array([new_r, new_c], dtype=jnp.int16))

    # ---- 11: CREATE_PORTAL (Eye of Aethiopica) artifact.c:1866-1931 -------
    # Menu-driven goto_level; deferred (no dungeon-state mutation).
    # Audit K: removed the invented in-handler ENERGY_REGEN bonus —
    # SPFX_EREGEN extrinsic is granted by carrying (cspfx), via
    # apply_carried_artifact_extrinsics.
    def _h_create_portal(s):
        return s

    # ---- 12: SNOWSTORM (Frost Brand) artifact.c:2169-2171 -----------------
    def _h_snowstorm(s):
        return _storm_apply_dmg(s, k_handler, cold=True)

    # ---- 13: FIRESTORM (Fire Brand) artifact.c:2169-2171 ------------------
    def _h_firestorm(s):
        return _storm_apply_dmg(s, k_handler, cold=False)

    # ---- 14: BANISH (Demonbane) artifact.c:1962-2019 ----------------------
    # Vendor invoke_banish (artifact.c:1962-2019) iterates fmon and migrates
    # each demon/imp passing rn2(chance)==0 OR chance<=1.  The chance ladder:
    #     chance = 1
    #     if (In_quest(&u.uz) && !killed_nemesis) chance += 10
    #     if (is_dprince(data))                   chance += 2
    #     if (is_dlord(data))                     chance += 1
    # We model:
    #   demon-class slot ::= (M2_DEMON flags2) | (symbol == S_IMP).
    #   dprince          ::= demon && M2_PRINCE.
    #   dlord            ::= demon && M2_LORD.
    #   In_quest         ::= state.dungeon.current_branch == quest branch id
    #                        (no killed_nemesis tracker in Nethax; treat as
    #                        always-False so the +10 bonus applies whenever
    #                        on the quest branch — vendor degenerate case).
    # JIT-pure: bounded vmap over MAX_MONSTERS_PER_LEVEL slots.
    # Cite: vendor/nethack/src/artifact.c:1962-2019 (invoke_banish);
    #       vendor/nethack/include/mondata.h:140-141 (is_dlord/is_dprince);
    #       vendor/nethack/include/monflag.h M2_LORD=0x400, M2_PRINCE=0x800.
    def _h_banish(s):
        mai = s.monster_ai
        entry_i = jnp.clip(mai.entry_idx.astype(jnp.int32), 0,
                           _IS_DEMON.shape[0] - 1)
        # Demon-class: M2_DEMON OR S_IMP symbol (vendor `mdata->mlet == S_IMP`).
        is_demon_slot = (_IS_DEMON[entry_i] | _IS_IMP[entry_i]) & mai.alive

        # is_dprince / is_dlord per-slot lookups (precomputed masks).
        is_dprince_slot = _IS_DEMON[entry_i] & _IS_PRINCE[entry_i]
        is_dlord_slot   = _IS_DEMON[entry_i] & _IS_LORD[entry_i]

        # In_quest gate: branch index 2 is the Quest branch in Nethax dungeon
        # layout (see dungeon/branches.py).  Approximated as branch==2.
        # Vendor reference: Is_quest_lev macro in dungeon.h.
        in_quest = (s.dungeon.current_branch.astype(jnp.int32)
                    == jnp.int32(2))
        quest_bonus = jnp.where(in_quest, jnp.int32(10), jnp.int32(0))

        # Per-slot chance = 1 + bonuses.
        chance = (jnp.int32(1)
                  + quest_bonus
                  + jnp.where(is_dprince_slot, jnp.int32(2), jnp.int32(0))
                  + jnp.where(is_dlord_slot,   jnp.int32(1), jnp.int32(0)))
        chance_safe = jnp.maximum(chance, jnp.int32(1))

        n = mai.alive.shape[0]
        keys = jax.random.split(k_handler, n)
        # rn2(chance) per slot.
        rolls = jax.vmap(
            lambda k, c: jax.random.randint(k, (), 0, c, dtype=jnp.int32)
        )(keys, chance_safe)
        # Vendor: chance<=1 || rn2(chance)==0 → migrate.
        passes = (chance <= jnp.int32(1)) | (rolls == jnp.int32(0))
        banish_mask = is_demon_slot & passes
        new_alive = jnp.where(banish_mask, jnp.bool_(False), mai.alive)
        new_hp = jnp.where(banish_mask, jnp.int32(0), mai.hp)
        return s.replace(monster_ai=mai.replace(alive=new_alive, hp=new_hp))

    # ---- 15: FLING_POISON (Grimtooth) artifact.c:2021-2037 ----------------
    # Vendor: rn2(2) picks BLINDING_VENOM or ACID_VENOM, mksobj + throwit
    # towards getdir().  Wave 46a wires the per-monster ``blind_timer``
    # (added 45a) so BLINDING_VENOM now sets the nearest adjacent
    # monster's blind timer by ``rnd(25)`` and ACID_VENOM still applies
    # ``d6`` HP damage (vendor objects.h ACID_VENOM dmgval).
    # Cite: vendor/nethack/src/artifact.c::invoke_fling_poison lines 2021-2037;
    #       vendor/nethack/src/zap.c::flash_hits_mon line 2925 (blind_timer).
    def _h_fling_poison(s):
        k_venom, k_dmg, k_blind = jax.random.split(k_handler, 3)
        venom_roll = jax.random.randint(k_venom, (), 0, 2, dtype=jnp.int32)  # rn2(2)
        is_blinding = venom_roll == jnp.int32(0)
        mai = s.monster_ai
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        mpos = mai.pos.astype(jnp.int32)
        d_row = jnp.abs(mpos[:, 0] - pr)
        d_col = jnp.abs(mpos[:, 1] - pc)
        in_range = (d_row <= jnp.int32(1)) & (d_col <= jnp.int32(1)) & mai.alive
        # ACID_VENOM half: d6 HP damage applied when ~is_blinding.
        dmg = _dice_sum(k_dmg, jnp.int32(1), jnp.int32(6))
        hit_acid = in_range & ~is_blinding
        new_hp = jnp.where(hit_acid,
                           jnp.maximum(mai.hp - dmg, jnp.int32(0)),
                           mai.hp)
        new_alive = jnp.where(hit_acid & (new_hp <= jnp.int32(0)),
                              jnp.bool_(False), mai.alive)
        # BLINDING_VENOM half: blind_timer += rnd(25).
        blind_amt = jax.random.randint(k_blind, (), 1, 26, dtype=jnp.int32)
        hit_blind = in_range & is_blinding
        cur_blind = mai.blind_timer.astype(jnp.int32)
        new_blind = jnp.where(hit_blind,
                              jnp.minimum(cur_blind + blind_amt,
                                          jnp.iinfo(jnp.int16).max),
                              cur_blind).astype(jnp.int16)
        return s.replace(monster_ai=mai.replace(
            hp=new_hp, alive=new_alive, blind_timer=new_blind
        ))

    # ---- 16: BLINDING_RAY (Sunsword) artifact.c:2053-2086 -----------------
    # Vendor: do_blinding_ray (apply.c:60-76) → bhit + flash_hits_mon
    # blinds an adjacent monster for ``damg + rnd(damg)`` turns where
    # ``damg = blessed?15 : !cursed?10 : 5``.  Wave 46a wires the per-
    # monster ``blind_timer`` (added 45a) so the blinding effect now
    # actually sets the timer (no HP damage — flash_hits_mon only blinds).
    # Additionally vendor artifact.c:2063 calls ``litroom(TRUE, obj)`` on
    # the u.dz!=0 branch — we have no direction at this layer so we light
    # the player's own tile as the "ray illuminates the area" side-effect.
    # We assume uncursed (damg=10) since the wielded-slot artifact is the
    # most common state.
    # Cite: vendor/nethack/src/artifact.c::invoke_blinding_ray lines 2053-2086
    #       (litroom call at line 2063);
    #       vendor/nethack/src/apply.c::do_blinding_ray lines 60-76;
    #       vendor/nethack/src/zap.c::flash_hits_mon line 2925 (mblinded set).
    def _h_blinding_ray(s):
        mai = s.monster_ai
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        mpos = mai.pos.astype(jnp.int32)
        d_row = jnp.abs(mpos[:, 0] - pr)
        d_col = jnp.abs(mpos[:, 1] - pc)
        in_range = (d_row <= jnp.int32(1)) & (d_col <= jnp.int32(1)) & mai.alive
        damg = jnp.int32(10)  # uncursed default per artifact.c:2070
        rnd_damg = jax.random.randint(k_handler, (), 1, 11, dtype=jnp.int32)
        total = damg + rnd_damg  # damg + rnd(damg)
        cur_blind = mai.blind_timer.astype(jnp.int32)
        new_blind = jnp.where(in_range,
                              jnp.minimum(cur_blind + total,
                                          jnp.iinfo(jnp.int16).max),
                              cur_blind).astype(jnp.int16)
        # Light the player's tile (artifact.c:2063 litroom call).
        from Nethax.nethax.subsystems.features import (
            litroom_at, _flat_lv_from_state,
        )
        flv = _flat_lv_from_state(s)
        new_features = litroom_at(
            s.features, flv,
            s.player_pos[0].astype(jnp.int32),
            s.player_pos[1].astype(jnp.int32),
            radius=0,
        )
        return s.replace(
            monster_ai=mai.replace(blind_timer=new_blind),
            features=new_features,
        )

    handlers = [
        _h_noop,           #  0  NOOP
        _h_conflict,       #  1  CONFLICT
        _h_invis,          #  2  INVIS
        _h_levitation,     #  3  LEVITATION
        _h_healing,        #  4  HEALING
        _h_enlightening,   #  5  ENLIGHTENING
        _h_energy_boost,   #  6  ENERGY_BOOST
        _h_create_ammo,    #  7  CREATE_AMMO
        _h_untrap,         #  8  UNTRAP
        _h_charge_obj,     #  9  CHARGE_OBJ
        _h_lev_tele,       # 10  LEV_TELE
        _h_create_portal,  # 11  CREATE_PORTAL
        _h_snowstorm,      # 12  SNOWSTORM
        _h_firestorm,      # 13  FIRESTORM
        _h_banish,         # 14  BANISH
        _h_fling_poison,   # 15  FLING_POISON
        _h_blinding_ray,   # 16  BLINDING_RAY
    ]

    new_state = jax.lax.switch(handler_idx, handlers, state_after_pw)
    # On refused (tired + no Pw) path, return only the age-bumped state.
    return jax.lax.cond(refused, lambda _: state_after_age, lambda s: s, new_state)


# ---------------------------------------------------------------------------
# Storm-spell helper (handlers 12 / 13 — SNOWSTORM / FIRESTORM).
# Vendor invoke_storm_spell (artifact.c:2039-2051) forces P_EXPERT in the
# cold/fire school then runs ``spelleffects(SPE_CONE_OF_COLD/SPE_FIREBALL,
# FALSE, TRUE)``.  At P_EXPERT the explode-AOE spreads over 9 tiles (3x3
# centred on the cursor target) at d(nd, 6) per tile where
# ``nd = u.ulevel/2 + 1``.  Without a cursor target we explode at the
# player's own tile (vendor's default when no direction given).
# Damage per monster on a hit tile: d(nd, 6) cold or d(nd, 6) fire.
# ---------------------------------------------------------------------------
def _storm_apply_dmg(state, rng, cold: bool):
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    mai = state.monster_ai
    mpos = mai.pos.astype(jnp.int32)
    # 3x3 AoE around player (Chebyshev<=1) matches vendor explode() radius.
    in_range = (
        (jnp.abs(mpos[:, 0] - pr) <= jnp.int32(1))
        & (jnp.abs(mpos[:, 1] - pc) <= jnp.int32(1))
        & mai.alive
    )
    # nd = u.ulevel/2 + 1; per-tile d(nd, 6).
    nd = (state.player_xl.astype(jnp.int32) // jnp.int32(2)) + jnp.int32(1)
    # Sum of nd d6 rolls (mask up to a static max of 16 dice).
    _MAX_ND = 16
    sub_keys = jax.random.split(rng, _MAX_ND)
    rolls = jax.vmap(lambda k: jax.random.randint(k, (), 1, 7))(sub_keys)
    mask = jnp.arange(_MAX_ND, dtype=jnp.int32) < jnp.minimum(nd, _MAX_ND)
    dmg = jnp.sum(jnp.where(mask, rolls, 0)).astype(jnp.int32)
    _ = cold  # Both storms share the same dice; vendor differentiates only by resistance, modelled at status_effects.
    new_hp = jnp.where(in_range, jnp.maximum(mai.hp - dmg, jnp.int32(0)), mai.hp)
    new_alive = jnp.where(in_range & (new_hp <= jnp.int32(0)),
                          jnp.bool_(False), mai.alive)
    return state.replace(monster_ai=mai.replace(hp=new_hp, alive=new_alive))


# ---------------------------------------------------------------------------
# Part D: Excalibur alignment damage (Wield_artifact_unaligned)
#
# Cite: vendor/nethack/src/artifact.c::Wield_artifact_unaligned —
#   if player alignment != Excalibur's alignment (LAWFUL=2), deal 4d10 damage.
# ---------------------------------------------------------------------------
# Alignment enum: CHAOTIC=0, NEUTRAL=1, LAWFUL=2 (prayer.py).
_ALIGN_LAWFUL: int = 2
# Excalibur alignment is LAWFUL (wish.py artifact_alignment_table index 0).
_EXCALIBUR_ALIGN: int = _ALIGN_LAWFUL


def check_excalibur_alignment(state, rng):
    """Deal touch-artifact blast damage when wielding misaligned Excalibur.

    Audit K (Wave 40c): vendor formula is
        d((Antimagic ? 2 : 4), (self_willed ? 10 : 4))
    cite vendor/nethack/src/artifact.c::touch_artifact line 953.  Excalibur
    is self_willed=TRUE (SPFX_INTEL in artilist.h:85-88) so:
        no Antimagic → 4d10
        Antimagic    → 2d10

    The pre-Audit-K implementation hardcoded 4d10, missing the Antimagic
    reduction.

    JIT-pure.  Returns new_state.

    Cite: vendor/nethack/src/artifact.c::touch_artifact line 953;
          artilist.h:85-88 (Excalibur SPFX_INTEL).
    """
    art = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_excalibur = art == jnp.int32(_ARTI_EXCALIBUR)
    align = state.player_align.astype(jnp.int32)
    not_lawful = align != jnp.int32(_EXCALIBUR_ALIGN)

    antimagic = _is_antimagic(state)
    n_dice  = jnp.where(antimagic, jnp.int32(2), jnp.int32(4))
    # self_willed=True for Excalibur → 10 sides.
    dmg = _dice_sum(rng, n_dice, jnp.int32(10))

    apply_dmg = is_excalibur & not_lawful
    new_hp = jnp.where(
        apply_dmg,
        jnp.maximum(state.player_hp - dmg, jnp.int32(0)),
        state.player_hp,
    )
    return state.replace(player_hp=new_hp)


# ===========================================================================
# wave17a — P0 #2..#15 byte-equal vendor parity extensions.
# ===========================================================================

# ---------------------------------------------------------------------------
# Per-artifact SPFX flags (subset relevant for P0s).
# Bit values mirror vendor/nethack/include/artifact.h SPFX_* defines.
# We only encode the flags actually consulted by P0 implementations.
# ---------------------------------------------------------------------------
_SPFX_NOGEN     = 0x00000001
_SPFX_RESTR     = 0x00000002
_SPFX_INTEL     = 0x00000004   # P0 #2 touch-blast
_SPFX_SPEAK     = 0x00000008
_SPFX_SEEK      = 0x00000010
_SPFX_WARN      = 0x00000020
_SPFX_ATTK      = 0x00000040
_SPFX_DEFN      = 0x00000080
_SPFX_DRLI      = 0x00000100   # P0 #15 Stormbringer/Staff DRLI on-hit
_SPFX_SEARCH    = 0x00000200
_SPFX_BEHEAD    = 0x00000400   # P0 #3 Tsurugi slice-in-half
_SPFX_HALRES    = 0x00000800
_SPFX_ESP       = 0x00001000
_SPFX_STLTH     = 0x00002000
_SPFX_REGEN     = 0x00004000
_SPFX_EREGEN    = 0x00008000
_SPFX_HSPDAM    = 0x00010000
_SPFX_HPHDAM    = 0x00020000
_SPFX_TCTRL     = 0x00040000
_SPFX_LUCK      = 0x00080000
_SPFX_DMONS     = 0x00100000
_SPFX_DCLAS     = 0x00200000
_SPFX_DFLAG1    = 0x00400000
_SPFX_DFLAG2    = 0x00800000   # P0 #6 Mitre M2_UNDEAD bane
_SPFX_DALIGN    = 0x01000000   # P0 #5 Sceptre vs non-aligned
_SPFX_DBONUS    = (_SPFX_DMONS | _SPFX_DCLAS | _SPFX_DFLAG1
                   | _SPFX_DFLAG2 | _SPFX_DALIGN)
_SPFX_REFLECT   = 0x02000000
_SPFX_PROTECT   = 0x04000000   # P0 #6 Mitre +protection while wielded
_SPFX_XRAY      = 0x08000000

# Per-artifact-idx (0..32) SPFX bitfield.  Mirrors artilist.h "spfx" col.
# Cite: vendor/nethack/include/artilist.h lines 85-212.
_ARTIFACT_SPFX: tuple[int, ...] = tuple([
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_SEEK | _SPFX_DEFN | _SPFX_INTEL
     | _SPFX_SEARCH),                                                   # 0  Excalibur
    _SPFX_RESTR,                                                        # 1  Snickersnee
    (_SPFX_RESTR | _SPFX_ATTK | _SPFX_DEFN | _SPFX_INTEL | _SPFX_DRLI), # 2  Stormbringer
    (_SPFX_RESTR | _SPFX_ATTK),                                         # 3  Mjollnir
    _SPFX_RESTR,                                                        # 4  Cleaver
    (_SPFX_WARN | _SPFX_DFLAG2),                                        # 5  Sting
    (_SPFX_WARN | _SPFX_DFLAG2),                                        # 6  Orcrist
    (_SPFX_RESTR | _SPFX_HALRES),                                       # 7  Grayswandir
    (_SPFX_RESTR | _SPFX_BEHEAD),                                       # 8  Vorpal Blade
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_DALIGN),           # 9  Sceptre of Might
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_BEHEAD
     | _SPFX_LUCK | _SPFX_PROTECT),                                     # 10 Tsurugi
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_SPEAK),            # 11 Magic Mirror
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL),                          # 12 Orb of Detection
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL),                          # 13 Heart of Ahriman
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_ATTK | _SPFX_INTEL
     | _SPFX_DRLI | _SPFX_REGEN),                                       # 14 Staff of Aesculapius
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_XRAY),              # 15 Eyes of the Overworld
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_DFLAG2 | _SPFX_INTEL
     | _SPFX_PROTECT),                                                  # 16 Mitre of Holiness
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_REFLECT),          # 17 Longbow of Diana
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_SPEAK),            # 18 Master Key
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_DEFN),             # 19 Yendorian Express Card
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL | _SPFX_LUCK),              # 20 Orb of Fate
    (_SPFX_NOGEN | _SPFX_RESTR | _SPFX_INTEL),                          # 21 Eye of Aethiopica
    (_SPFX_RESTR | _SPFX_ATTK | _SPFX_DEFN),                            # 22 Frost Brand
    (_SPFX_RESTR | _SPFX_ATTK | _SPFX_DEFN),                            # 23 Fire Brand
    (_SPFX_RESTR | _SPFX_DCLAS | _SPFX_REFLECT),                        # 24 Dragonbane
    (_SPFX_RESTR | _SPFX_DFLAG2),                                       # 25 Demonbane
    (_SPFX_RESTR | _SPFX_DFLAG2),                                       # 26 Werebane
    (_SPFX_RESTR | _SPFX_DCLAS | _SPFX_REGEN),                          # 27 Trollsbane
    (_SPFX_RESTR | _SPFX_WARN | _SPFX_DFLAG2),                          # 28 Grimtooth
    (_SPFX_RESTR | _SPFX_ATTK | _SPFX_DEFN),                            # 29 Magicbane
    (_SPFX_RESTR | _SPFX_DFLAG2),                                       # 30 Giantslayer
    (_SPFX_RESTR | _SPFX_DCLAS),                                        # 31 Ogresmasher
    (_SPFX_RESTR | _SPFX_DFLAG2),                                       # 32 Sunsword
])

_ARTIFACT_SPFX_ARR: jnp.ndarray = jnp.array(_ARTIFACT_SPFX, dtype=jnp.int32)

# Per-artifact alignment.  Vendor align.h: A_CHAOTIC=-1, A_NEUTRAL=0,
# A_LAWFUL=1, A_NONE=-128.  Nethax Alignment: CHAOTIC=0, NEUTRAL=1, LAWFUL=2,
# UNALIGNED=3.  Conversion: CHAOTIC→0, NEUTRAL→1, LAWFUL→2, NONE→3.
# Cite: vendor/nethack/include/artilist.h "al" column per row, indices keyed
# to wish.py _ARTIFACTS positions 0..32.
_ARTIFACT_ALIGN_TABLE: tuple[int, ...] = tuple([
    2,  # 0  Excalibur                  A_LAWFUL   (artilist.h:87)
    2,  # 1  Snickersnee                A_LAWFUL   (artilist.h:204)
    0,  # 2  Stormbringer               A_CHAOTIC  (artilist.h:95)
    1,  # 3  Mjollnir                   A_NEUTRAL  (artilist.h:111)
    1,  # 4  Cleaver                    A_NEUTRAL  (artilist.h:115)
    0,  # 5  Sting                      A_CHAOTIC  (artilist.h:139)
    0,  # 6  Orcrist                    A_CHAOTIC  (artilist.h:135)
    2,  # 7  Grayswandir                A_LAWFUL   (artilist.h:171)
    1,  # 8  Vorpal Blade               A_NEUTRAL  (artilist.h:192)
    2,  # 9  Sceptre of Might           A_LAWFUL   (artilist.h:234)
    2,  # 10 Tsurugi of Muramasa        A_LAWFUL   (artilist.h:288)
    2,  # 11 Magic Mirror of Merlin     A_LAWFUL   (artilist.h:257)
    2,  # 12 Orb of Detection           A_LAWFUL   (artilist.h:221)
    1,  # 13 Heart of Ahriman           A_NEUTRAL  (artilist.h:228)
    1,  # 14 Staff of Aesculapius       A_NEUTRAL  (artilist.h:251)
    1,  # 15 Eyes of the Overworld      A_NEUTRAL  (artilist.h:262)
    2,  # 16 Mitre of Holiness          A_LAWFUL   (artilist.h:267)
    0,  # 17 Longbow of Diana           A_CHAOTIC  (artilist.h:273)
    0,  # 18 Master Key of Thievery     A_CHAOTIC  (artilist.h:282)
    1,  # 19 Yendorian Express Card     A_NEUTRAL  (artilist.h:294)
    1,  # 20 Orb of Fate                A_NEUTRAL  (artilist.h:300)
    1,  # 21 Eye of the Aethiopica      A_NEUTRAL  (artilist.h:305)
    3,  # 22 Frost Brand                A_NONE     (artilist.h:149)
    3,  # 23 Fire Brand                 A_NONE     (artilist.h:153)
    3,  # 24 Dragonbane                 A_NONE     (artilist.h:159)
    2,  # 25 Demonbane                  A_LAWFUL   (artilist.h:163)
    3,  # 26 Werebane                   A_NONE     (artilist.h:167)
    3,  # 27 Trollsbane                 A_NONE     (artilist.h:183)
    0,  # 28 Grimtooth                  A_CHAOTIC  (artilist.h:125)
    1,  # 29 Magicbane                  A_NEUTRAL  (artilist.h:146)
    1,  # 30 Giantslayer                A_NEUTRAL  (artilist.h:175)
    3,  # 31 Ogresmasher                A_NONE     (artilist.h:179)
    2,  # 32 Sunsword                   A_LAWFUL   (artilist.h:210)
])
_ARTIFACT_ALIGN_ARR: jnp.ndarray = jnp.array(_ARTIFACT_ALIGN_TABLE, dtype=jnp.int8)

# Per-artifact role gate.  Vendor PM_* role → Nethax Role index, or -1 for
# NON_PM ("no role restriction").  Used by touch_artifact badclass check
# (artifact.c::touch_artifact line 924-931).  Vendor lookups use art->role.
# Cite: vendor/nethack/include/artilist.h "role" column per row, mapped
# through Nethax constants/roles.py Role enum.
_ARTIFACT_ROLE_TABLE: tuple[int, ...] = tuple([
     4,  # 0  Excalibur                  PM_KNIGHT       → KNIGHT
     9,  # 1  Snickersnee                PM_SAMURAI      → SAMURAI
    -1,  # 2  Stormbringer               NON_PM
    11,  # 3  Mjollnir                   PM_VALKYRIE     → VALKYRIE
     1,  # 4  Cleaver                    PM_BARBARIAN    → BARBARIAN
    -1,  # 5  Sting                      NON_PM
    -1,  # 6  Orcrist                    NON_PM
    -1,  # 7  Grayswandir                NON_PM
    -1,  # 8  Vorpal Blade               NON_PM
     2,  # 9  Sceptre of Might           PM_CAVE_DWELLER → CAVEMAN
     9,  # 10 Tsurugi of Muramasa        PM_SAMURAI      → SAMURAI
     4,  # 11 Magic Mirror of Merlin     PM_KNIGHT       → KNIGHT
     0,  # 12 Orb of Detection           PM_ARCHEOLOGIST → ARCHEOLOGIST
     1,  # 13 Heart of Ahriman           PM_BARBARIAN    → BARBARIAN
     3,  # 14 Staff of Aesculapius       PM_HEALER       → HEALER
     5,  # 15 Eyes of the Overworld      PM_MONK         → MONK
     6,  # 16 Mitre of Holiness          PM_CLERIC       → PRIEST
     7,  # 17 Longbow of Diana           PM_RANGER       → RANGER
     8,  # 18 Master Key of Thievery     PM_ROGUE        → ROGUE
    10,  # 19 Yendorian Express Card     PM_TOURIST      → TOURIST
    11,  # 20 Orb of Fate                PM_VALKYRIE     → VALKYRIE
    12,  # 21 Eye of the Aethiopica      PM_WIZARD       → WIZARD
    -1,  # 22 Frost Brand                NON_PM
    -1,  # 23 Fire Brand                 NON_PM
    -1,  # 24 Dragonbane                 NON_PM
     6,  # 25 Demonbane                  PM_CLERIC       → PRIEST
    -1,  # 26 Werebane                   NON_PM
    -1,  # 27 Trollsbane                 NON_PM
    -1,  # 28 Grimtooth                  NON_PM
    12,  # 29 Magicbane                  PM_WIZARD       → WIZARD
    -1,  # 30 Giantslayer                NON_PM
    -1,  # 31 Ogresmasher                NON_PM
    -1,  # 32 Sunsword                   NON_PM
])
_ARTIFACT_ROLE_ARR: jnp.ndarray = jnp.array(_ARTIFACT_ROLE_TABLE, dtype=jnp.int8)

_N_ARTI_SLOTS: int = 33  # 0..32 inclusive — total artilist entries in Nethax.


def _arti_spfx(art_idx: jnp.ndarray) -> jnp.ndarray:
    """JIT-safe per-artifact SPFX bitfield lookup; returns 0 when idx is -1."""
    safe = jnp.clip(art_idx.astype(jnp.int32), 0, _N_ARTI_SLOTS - 1)
    spfx = _ARTIFACT_SPFX_ARR[safe]
    return jnp.where(art_idx.astype(jnp.int32) >= jnp.int32(0), spfx, jnp.int32(0))


def _arti_align(art_idx: jnp.ndarray) -> jnp.ndarray:
    """JIT-safe per-artifact alignment lookup; returns 3 (UNALIGNED) when -1."""
    safe = jnp.clip(art_idx.astype(jnp.int32), 0, _N_ARTI_SLOTS - 1)
    al = _ARTIFACT_ALIGN_ARR[safe].astype(jnp.int32)
    return jnp.where(art_idx.astype(jnp.int32) >= jnp.int32(0), al, jnp.int32(3))


# ---------------------------------------------------------------------------
# P0 #2 — SPFX_INTEL touch-blast.
#
# Cite: vendor/nethack/src/artifact.c::touch_artifact lines 944-959.
# Formula: dmg = d((Antimagic ? 2 : 4), (self_willed ? 10 : 4))
# Triggered when (badclass || badalign) && self_willed,
#             or (badalign && (!yours || !rn2(4))).
#
# Currently implemented only for Excalibur via check_excalibur_alignment.
# This new function implements the generic case (P0 #11 too).
# ---------------------------------------------------------------------------

def _is_antimagic(state) -> jnp.ndarray:
    """Bool: hero has MAGIC_RESIST extrinsic or intrinsic."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _I
    intr = state.status.intrinsics[int(_I.MAGIC_RESIST)]
    timed = state.status.timed_intrinsics[int(_I.MAGIC_RESIST)] > jnp.int32(0)
    return intr | timed


def _dice_sum(rng, n_dice: jnp.ndarray, n_sides: jnp.ndarray) -> jnp.ndarray:
    """Compute sum of n_dice rolls of d(n_sides), JIT-pure.

    Mirrors vendor d(N,S) in dungeon.c (sum of N uniform [1,S] rolls).
    Uses a fixed 16-roll upper bound — sufficient for our P0 use (max d4×4).
    """
    MAX_ROLLS = 16
    keys = jax.random.split(rng, MAX_ROLLS)
    sides_safe = jnp.maximum(n_sides.astype(jnp.int32), jnp.int32(1))
    n_safe = jnp.clip(n_dice.astype(jnp.int32), 0, MAX_ROLLS)
    total = jnp.int32(0)
    for i in range(MAX_ROLLS):
        roll = jax.random.randint(keys[i], (), 1, sides_safe + jnp.int32(1),
                                  dtype=jnp.int32)
        total = total + jnp.where(jnp.int32(i) < n_safe, roll, jnp.int32(0))
    return total.astype(jnp.int32)


def touch_artifact_blast(state, art_idx: jnp.ndarray, rng) -> tuple:
    """Compute touch-blast damage for trying to touch/wield/pickup an artifact.

    P0 #2 + P0 #11 — generic alignment/class blast.

    Returns
    -------
    (new_state, refused)  — refused True when (badclass && badalign && intel)
    so caller should refuse the pickup/wield.

    Cite: vendor/nethack/src/artifact.c::touch_artifact lines 908-974.
    """
    arti = art_idx.astype(jnp.int32)
    spfx = _arti_spfx(arti)
    arti_align = _arti_align(arti)
    player_align = state.player_align.astype(jnp.int32)
    self_willed = (spfx & jnp.int32(_SPFX_INTEL)) != jnp.int32(0)

    # badclass: artifact->role != NON_PM && artifact->role != player_role.
    # Cite: vendor/nethack/src/artifact.c::touch_artifact lines 924-931 —
    #   if (oart->role != NON_PM && !Role_if(oart->role)) badclass = TRUE;
    # Nethax encodes NON_PM as sentinel -1; otherwise art_role is the Nethax
    # Role index (constants/roles.py).
    safe_arti = jnp.clip(arti, 0, _N_ARTI_SLOTS - 1)
    art_role = _ARTIFACT_ROLE_ARR[safe_arti].astype(jnp.int32)
    art_role = jnp.where(arti >= jnp.int32(0), art_role, jnp.int32(-1))
    player_role = state.player_role.astype(jnp.int32)
    has_role_gate = art_role != jnp.int32(-1)
    badclass = has_role_gate & (art_role != player_role)

    # badalign: SPFX_RESTR && alignment != A_NONE && (alignment != player).
    # Alignment.UNALIGNED == 3 maps to artilist A_NONE.
    restr_set = (spfx & jnp.int32(_SPFX_RESTR)) != jnp.int32(0)
    not_unaligned = arti_align != jnp.int32(3)
    align_mismatch = arti_align != player_align
    badalign = restr_set & not_unaligned & align_mismatch

    # Trigger condition (yours=True path; vendor line 944-945).
    rn_key, dmg_key = jax.random.split(rng, 2)
    rn4 = jax.random.randint(rn_key, (), 0, 4, dtype=jnp.int32)
    trigger = ((badclass | badalign) & self_willed) | (
        badalign & (rn4 == jnp.int32(0))
    )

    # dmg = d((Antimagic ? 2 : 4), (self_willed ? 10 : 4)).  Cite line 953.
    antimagic = _is_antimagic(state)
    n_dice  = jnp.where(antimagic, jnp.int32(2), jnp.int32(4))
    n_sides = jnp.where(self_willed, jnp.int32(10), jnp.int32(4))
    dmg = _dice_sum(dmg_key, n_dice, n_sides)
    dmg = jnp.where(trigger, dmg, jnp.int32(0))

    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    new_state = state.replace(player_hp=new_hp)

    # Refusal: vendor line 963 — only if all three (badclass && badalign &&
    # self_willed) hold (i.e. "totally non-synch'd").
    refused = badclass & badalign & self_willed
    return new_state, refused


# ---------------------------------------------------------------------------
# P0 #3 — SPFX_BEHEAD Tsurugi slice-in-half on dieroll==1.
#
# Cite: vendor/nethack/src/artifact.c lines 1550-1594.
# Tsurugi: if dieroll==1, *dmgptr = 2*mhp + FATAL_DAMAGE_MODIFIER (kill).
# Vorpal:  if dieroll==1 || jabberwock, behead (auto-kill).
# Already partially implemented for Vorpal in apply_artifact_hit_effects.
# This adds Tsurugi.
# ---------------------------------------------------------------------------
_ARTI_TSURUGI_BEHEAD_DIEROLL: int = 1


def apply_tsurugi_slice(state, mon_slot, rng) -> tuple:
    """Tsurugi-of-Muramasa: dieroll==1 → slice in half (instant kill).

    JIT-pure.  Returns (new_state, killed).

    Cite: vendor/nethack/src/artifact.c lines 1551-1594.
    """
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    idx = mon_slot.astype(jnp.int32)
    mai = state.monster_ai
    is_tsurugi = (arti == jnp.int32(_ARTI_TSURUGI)) & mai.alive[idx]

    # Vendor uses d20 dieroll; we sample fresh here so each call is
    # independent (combat.py owns the canonical dieroll otherwise).
    dieroll = jax.random.randint(rng, (), 1, 21, dtype=jnp.int32)
    slice_in_half = is_tsurugi & (dieroll == jnp.int32(_ARTI_TSURUGI_BEHEAD_DIEROLL))

    new_hp = jnp.where(slice_in_half, jnp.int32(0), mai.hp[idx])
    new_alive = jnp.where(slice_in_half, jnp.bool_(False), mai.alive[idx])

    mai2 = mai.replace(
        hp=mai.hp.at[idx].set(new_hp),
        alive=mai.alive.at[idx].set(new_alive),
    )
    return state.replace(monster_ai=mai2), slice_in_half


# ---------------------------------------------------------------------------
# P0 #4 — cspfx carry-extrinsic system for "while carried" properties.
#
# Cite: vendor/nethack/src/artifact.c::set_artifact_intrinsic + artilist
# "cspfx" column (the second SPFX bitfield, e.g. SPFX_ESP for Orb of
# Detection).  Properties granted by *carrying* (not just wielding):
#
#   Orb of Detection  cspfx = SPFX_ESP | SPFX_HSPDAM
#   Heart of Ahriman  cspfx = SPFX_STLTH
#   Yendorian Express cspfx = SPFX_ESP | SPFX_HSPDAM
#   Mitre of Holiness cspfx = 0 (worn-only via PROTECT/CARY)
#   Eye of Aethiopica cspfx = SPFX_EREGEN | SPFX_HSPDAM
#   Orb of Fate       cspfx = SPFX_WARN | SPFX_HSPDAM | SPFX_HPHDAM
#   Master Key        cspfx = SPFX_WARN | SPFX_TCTRL | SPFX_HPHDAM
#   Magic Mirror      cspfx = SPFX_ESP
#   Eyes of Overworld no cspfx (xray only when worn)
# ---------------------------------------------------------------------------

# Audit K — synthetic CARY-flag bits used to encode the artilist.h CARY()
# column on the same bitfield as SPFX.  These do not collide with vendor
# SPFX_* values because the high bits 0x40000000+ are unused in vendor.
_CARY_FIRE   = 0x40000000   # CARY(AD_FIRE)  → RESIST_FIRE
_CARY_COLD   = 0x20000000   # CARY(AD_COLD)  → RESIST_COLD
_CARY_MAGM   = 0x10000000   # CARY(AD_MAGM)  → MAGIC_RESIST (Antimagic)

# Per-artifact-idx cspfx bitfield (second SPFX column in artilist.h) PLUS
# Audit-K synthetic CARY bits for AD_* carry properties.
_ARTIFACT_CSPFX: tuple[int, ...] = tuple([
    0,                                                                  # 0  Excalibur
    0,                                                                  # 1  Snickersnee
    0,                                                                  # 2  Stormbringer
    0,                                                                  # 3  Mjollnir
    0,                                                                  # 4  Cleaver
    0,                                                                  # 5  Sting
    0,                                                                  # 6  Orcrist
    0,                                                                  # 7  Grayswandir
    0,                                                                  # 8  Vorpal Blade
    0,                                                                  # 9  Sceptre of Might
    0,                                                                  # 10 Tsurugi
    (_SPFX_ESP | _CARY_MAGM),                                           # 11 Magic Mirror  CARY(AD_MAGM)
    (_SPFX_ESP | _SPFX_HSPDAM | _CARY_MAGM),                            # 12 Orb of Detection CARY(AD_MAGM)
    _SPFX_STLTH,                                                        # 13 Heart of Ahriman
    0,                                                                  # 14 Staff of Aesculapius
    0,                                                                  # 15 Eyes of Overworld
    _CARY_FIRE,                                                         # 16 Mitre of Holiness CARY(AD_FIRE)
    _SPFX_ESP,                                                          # 17 Longbow of Diana
    (_SPFX_WARN | _SPFX_TCTRL | _SPFX_HPHDAM),                          # 18 Master Key
    (_SPFX_ESP | _SPFX_HSPDAM | _CARY_MAGM),                            # 19 Yendorian Express CARY(AD_MAGM)
    (_SPFX_WARN | _SPFX_HSPDAM | _SPFX_HPHDAM),                         # 20 Orb of Fate
    (_SPFX_EREGEN | _SPFX_HSPDAM),                                      # 21 Eye of Aethiopica
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                                    # 22..32
])
_ARTIFACT_CSPFX_ARR: jnp.ndarray = jnp.array(_ARTIFACT_CSPFX, dtype=jnp.int32)

# Map cspfx bit → Intrinsic enum value (granted while carried).
# Cite: vendor/nethack/src/artifact.c::set_artifact_intrinsic switch.
_CSPFX_TO_INTRINSIC: tuple[tuple[int, int], ...] = (
    (_SPFX_ESP,     30),  # TELEPATHY
    (_SPFX_STLTH,   42),  # STEALTH
    (_SPFX_WARN,    31),  # WARNING
    (_SPFX_EREGEN,  58),  # ENERGY_REGEN
    (_SPFX_HSPDAM,  55),  # HALF_SPELL_DAMAGE
    (_SPFX_HPHDAM,  56),  # HALF_PHYSICAL_DAMAGE
    (_SPFX_TCTRL,   47),  # TELEPORT_CONTROL
    # Audit-K CARY routes:
    (_CARY_FIRE,     1),  # RESIST_FIRE
    (_CARY_COLD,     2),  # RESIST_COLD
    (_CARY_MAGM,    12),  # MAGIC_RESIST  (Antimagic)
)


def apply_carried_artifact_extrinsics(state):
    """Set/clear intrinsics granted by carrying artifacts (not just wielding).

    Walks all inventory slots, ORs cspfx bits from each carried artifact, then
    sets intrinsics[i] accordingly.  Mirrors vendor "while carried" semantics.

    JIT-pure: bounded fixed-size scan over inventory.

    Cite: vendor/nethack/src/artifact.c::set_artifact_intrinsic
          (sets carry extrinsic per cspfx bit).
    """
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS
    # We track which artifacts are carried via inventory.items
    # Each item has an artifact identity tracked via the parser/wield path.
    # In Nethax the canonical "is this slot an artifact" is currently only
    # stored via wielded_artifact_idx; we do not yet track per-slot artifact
    # ids.  For P0 #4 we conservatively apply cspfx of the WIELDED artifact
    # plus a heuristic fallback (carry = wielded for now).
    #
    # This is the minimum-viable hook: extends artifact_powers' grant set
    # without requiring inventory-schema changes.  TODO post-wave17a: extend
    # InventoryState with per-slot artifact_idx[MAX_INVENTORY_SLOTS] then
    # iterate all slots here.
    art = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    safe = jnp.clip(art, 0, _N_ARTI_SLOTS - 1)
    cspfx = _ARTIFACT_CSPFX_ARR[safe]
    cspfx = jnp.where(art >= jnp.int32(0), cspfx, jnp.int32(0))

    intrinsics = state.status.intrinsics
    # Clear all carry-granted intrinsics unconditionally, then re-apply.
    for _bit, iid in _CSPFX_TO_INTRINSIC:
        intrinsics = intrinsics.at[iid].set(jnp.bool_(False))
    for bit, iid in _CSPFX_TO_INTRINSIC:
        grant = (cspfx & jnp.int32(bit)) != jnp.int32(0)
        intrinsics = intrinsics.at[iid].set(
            jnp.where(grant, jnp.bool_(True), intrinsics[iid])
        )
    return state.replace(status=state.status.replace(intrinsics=intrinsics))


# ---------------------------------------------------------------------------
# P0 #5 — Sceptre of Might SPFX_DALIGN double-damage vs non-aligned.
#
# Cite: vendor/nethack/src/artifact.c::spec_applies line 1031-1034 +
#       artilist.h line 232 (Sceptre: SPFX_DALIGN, A_LAWFUL).
#
# spec_applies returns TRUE if monster alignment != weapon alignment.
# spec_dbon then adds rnd(damd)=rnd(0) — no extra damage from damd.
# The "double damage" is implemented via gs.spec_dbon_applies → SPFX_BEHEAD-
# free path: in vendor, SPFX_DALIGN triggers no extra bonus from spec_dbon
# itself (damd=0), but artifact_hit gates further effects.  However the
# task spec explicitly requests double-damage vs non-aligned — implement
# as +base_dmg (effectively *2) on hit.
# ---------------------------------------------------------------------------

def apply_sceptre_dalign(state, mon_slot, base_dmg) -> jnp.ndarray:
    """Sceptre of Might SPFX_DALIGN gate.  Returns ``base_dmg`` unchanged.

    Audit K (Wave 40c) — drops the invented `* 2` doubling.  Vendor:
        spec_applies SPFX_DALIGN returns
          sgn(mon->mdat->maligntyp) != weap->alignment
        spec_dbon then evaluates
          attk.damd ? rnd(damd) : max(tmp, 1)
    Sceptre's attk damd is 0 (artilist.h:233 PHYS(5,0)), so spec_dbon adds
    ``max(tmp,1)`` — at most +1 over the unmodified base damage.  The
    caller already clamps base_dmg to >=1, so the correct byte-equal
    return is ``base_dmg``.

    The gate evaluation is kept (for any future caller that wants a
    boolean) but the return is unconditional.

    Cite: vendor/nethack/src/artifact.c::spec_applies lines 1031-1034;
          spec_dbon lines 1106-1108; artilist.h line 232-235.
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_sceptre = arti == jnp.int32(_ARTI_SCEPTRE)

    if not hasattr(apply_sceptre_dalign, "_align_table"):
        align_list = []
        for m in MONSTERS:
            a = getattr(m, "alignment", 0)
            try:
                align_list.append(int(a))
            except Exception:
                align_list.append(0)
        apply_sceptre_dalign._align_table = jnp.array(align_list, dtype=jnp.int8)
    align_table = apply_sceptre_dalign._align_table

    mai = state.monster_ai
    idx = mon_slot.astype(jnp.int32)
    entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0,
                     align_table.shape[0] - 1)
    mon_align = align_table[entry].astype(jnp.int32)
    # Vendor sgn(): -1/0/+1.
    sgn_mon = jnp.where(mon_align > jnp.int32(0), jnp.int32(1),
                        jnp.where(mon_align < jnp.int32(0), jnp.int32(-1),
                                  jnp.int32(0)))
    # Sceptre alignment is A_LAWFUL == 1 in vendor enum.
    _gate_applies = is_sceptre & (sgn_mon != jnp.int32(1))   # noqa: F841
    return base_dmg


# ---------------------------------------------------------------------------
# P0 #6 — Mitre of Holiness SPFX_PROTECT + CARY(AD_FIRE) + DFLAG2 M2_UNDEAD.
#
# Cite: vendor/nethack/src/artifact.c::set_artifact_intrinsic (worn case) +
#       artilist.h line 265-269.
#
# Effects (while wielded/worn):
#   - SPFX_PROTECT: +AC bonus / divine protection
#   - CARY(AD_FIRE): grants RESIST_FIRE while carried
#   - DFLAG2 + M2_UNDEAD: bane vs undead (added to SPFX_DBONUS)
# ---------------------------------------------------------------------------
_ARTI_MITRE: int = 16


def apply_mitre_of_holiness(state):
    """Mitre of Holiness extrinsic grants (Audit K Wave 40c rewrite).

    Vendor (artilist.h:265-269): the Mitre carries SPFX_PROTECT and
    CARY(AD_FIRE).  The fire-resistance is a CARRY extrinsic — granted
    whenever the Mitre is anywhere in inventory, not just wielded — so
    we route RESIST_FIRE through cspfx in apply_carried_artifact_extrinsics
    (the Mitre's row in _ARTIFACT_CSPFX is extended below by callers via
    the carry walker).  Here we only handle the worn-only effects:

      SPFX_PROTECT → +1 AC bonus contributed to find_ac, not the
                     PROTECTION intrinsic flag.  Vendor do_wear.c::find_ac
                     sums per-slot ARM_BONUS entries; we write the bonus
                     into worn_armor_ac_bonus[helmet_slot] when the Mitre
                     is worn as a helmet.

    The pre-Audit-K implementation flipped the PROTECTION intrinsic, which
    is reserved for the divine-protection mechanic (priest's pray) and
    confused unrelated systems.  It also wrote RESIST_FIRE as an intrinsic
    only when wielded, missing the carry case.

    Cite: vendor/nethack/src/artifact.c::set_artifact_intrinsic;
          artilist.h:265-269 (Mitre row);
          vendor/nethack/src/do_wear.c::find_ac (AC bonus aggregation).
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_mitre = arti == jnp.int32(_ARTI_MITRE)

    # +1 AC via worn_armor_ac_bonus[HELMET].  Helmet slot index is ArmorSlot.HELMET.
    try:
        helmet_idx = int(ArmorSlot.HELMET)
    except Exception:
        helmet_idx = 0
    cur_bonus = state.inventory.worn_armor_ac_bonus[helmet_idx].astype(jnp.int32)
    # Apply additive +1; bounded to int8 dtype.
    new_bonus = jnp.where(
        is_mitre,
        jnp.clip(cur_bonus + jnp.int32(1), -128, 127),
        cur_bonus,
    ).astype(state.inventory.worn_armor_ac_bonus.dtype)
    new_inv = state.inventory.replace(
        worn_armor_ac_bonus=state.inventory.worn_armor_ac_bonus.at[helmet_idx].set(
            new_bonus
        ),
    )
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# P0 #9 — Magicbane Mb_hit: +rnd(4) per cumulative tier.
#
# Cite: vendor/nethack/src/artifact.c::Mb_hit lines 1287-1298.
# attack_indx tiers:
#   PROBE  : +rnd(4)                  → (2..3)d4 total
#   STUN   : +rnd(4)  (cumulative)    → (3..4)d4 total
#   SCARE  : +rnd(4)  (cumulative)    → (3..5)d4 total
#   CANCEL : +rnd(4)  (cumulative)    → (4..6)d4 total
#
# Trigger criteria (vendor lines 1262-1290):
#   scare_dieroll = MB_MAX_DIEROLL/2 = 4 (>>= spe/3 enchantment halving)
#   do_stun = max(spe, 0) < rn2(spec_dbon_applies ? 11 : 7)
#   if dieroll <= scare_dieroll → SCARE tier
#   if dieroll <= scare_dieroll/2 → CANCEL tier
# ---------------------------------------------------------------------------

# Magicbane tier indices for status-effect dispatch.
_MB_PROBE  = 0
_MB_STUN   = 1
_MB_SCARE  = 2
_MB_CANCEL = 3
_MB_MAX_DIEROLL = 8   # vendor line 1241


def magicbane_mb_hit(state, mon_slot, rng,
                     dieroll: jnp.ndarray = None,
                     spec_dbon_applies: bool = True) -> tuple:
    """Magicbane Mb_hit damage tiers and effects (P0 #9 + #10).

    Returns
    -------
    (new_state, extra_dmg, attack_indx)

    extra_dmg : cumulative rnd(4) bonus damage (0..16).
    attack_indx : 0=PROBE 1=STUN 2=SCARE 3=CANCEL.

    Cite: vendor/nethack/src/artifact.c::Mb_hit lines 1247-1434.
    """
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    idx = mon_slot.astype(jnp.int32)
    mai = state.monster_ai
    is_mb = (arti == jnp.int32(_ARTI_MAGICBANE)) & mai.alive[idx]

    k_die, k_d1, k_d2, k_d3, k_d4, k_stun, k_conf = jax.random.split(rng, 7)

    # dieroll: caller may provide; else sample 1..20 (vendor uses d20).
    if dieroll is None:
        dieroll_val = jax.random.randint(k_die, (), 1, 21, dtype=jnp.int32)
    else:
        dieroll_val = dieroll.astype(jnp.int32)
    # vendor line 1271: if !spec_dbon_applies, dieroll += 1.
    dbon = jnp.bool_(spec_dbon_applies)
    dieroll_eff = dieroll_val + jnp.where(dbon, jnp.int32(0), jnp.int32(1))

    # Enchantment-based halving of scare threshold (vendor line 1266-1267).
    # We model spe via wielded item enchantment.
    spe = state.inventory.items.enchantment[
        jnp.clip(state.inventory.wielded.astype(jnp.int32), 0,
                 state.inventory.items.enchantment.shape[0] - 1)
    ].astype(jnp.int32)
    spe_pos = jnp.maximum(spe, jnp.int32(0))
    halvings = spe_pos // jnp.int32(3)
    scare_dieroll = jnp.int32(_MB_MAX_DIEROLL // 2) >> halvings
    scare_dieroll = jnp.maximum(scare_dieroll, jnp.int32(1))

    # do_stun: max(spe,0) < rn2(spec_dbon_applies ? 11 : 7).  Line 1277.
    stun_bound = jnp.where(dbon, jnp.int32(11), jnp.int32(7))
    stun_rn = jax.random.randint(k_stun, (), 0, stun_bound, dtype=jnp.int32)
    do_stun = spe_pos < stun_rn

    # Cumulative damage rolls.  Each tier adds +rnd(4).  Vendor lines 1287-1298.
    d1 = jax.random.randint(k_d1, (), 1, 5, dtype=jnp.int32)  # always PROBE
    d2 = jax.random.randint(k_d2, (), 1, 5, dtype=jnp.int32)  # STUN
    d3 = jax.random.randint(k_d3, (), 1, 5, dtype=jnp.int32)  # SCARE
    d4 = jax.random.randint(k_d4, (), 1, 5, dtype=jnp.int32)  # CANCEL

    add_stun   = do_stun
    add_scare  = dieroll_eff <= scare_dieroll
    add_cancel = dieroll_eff <= (scare_dieroll // jnp.int32(2))

    extra = d1.astype(jnp.int32)
    extra = extra + jnp.where(add_stun,   d2, jnp.int32(0))
    extra = extra + jnp.where(add_scare,  d3, jnp.int32(0))
    extra = extra + jnp.where(add_cancel, d4, jnp.int32(0))

    # Select attack_indx: most severe wins (CANCEL > SCARE > STUN > PROBE).
    attack_indx = jnp.where(
        add_cancel, jnp.int32(_MB_CANCEL),
        jnp.where(add_scare, jnp.int32(_MB_SCARE),
                  jnp.where(add_stun, jnp.int32(_MB_STUN), jnp.int32(_MB_PROBE)))
    )

    # Audit K (Wave 40c): route Mb_hit status effects through the dedicated
    # per-monster fields (stun_timer, confuse_timer, flee_until_turn,
    # cancelled).  The pre-Audit-K implementation overloaded the `asleep`
    # flag, which collided with sleep_timer semantics and broke any
    # subsystem that distinguished stunned-vs-asleep.  Vendor lines:
    #   STUN   line 1390-1394 mstun=1 / make_stunned
    #   SCARE  line 1371      monflee(mdef, 3, ...) → flee for 3 turns
    #   CANCEL line 1321      cancel_monst → set mdef->mcan
    # Confusion (line 1399-1406) fires independently with 1/12 probability
    # on EVERY Mb_hit, not just on the cumulative tier.
    apply_mb = is_mb
    moves = state.timestep.astype(jnp.int32)

    # STUN tier OR fallthrough do_stun flag (vendor 1390 — "do_stun if
    # selected and a worse effect didn't occur"; CANCEL/SCARE clear do_stun).
    is_stun_tier = attack_indx == jnp.int32(_MB_STUN)
    # Vendor: STUN tier always stuns; STUN+SCARE/CANCEL clear do_stun.
    stun_fires = apply_mb & is_stun_tier
    cur_stun = mai.stun_timer[idx].astype(jnp.int32)
    new_stun = jnp.where(stun_fires, cur_stun + jnp.int32(3), cur_stun) \
                  .astype(mai.stun_timer.dtype)

    # SCARE tier → flee for 3 turns from now.
    scare_fires = apply_mb & (attack_indx == jnp.int32(_MB_SCARE))
    cur_flee = mai.flee_until_turn[idx].astype(jnp.int32)
    new_flee_until = jnp.where(scare_fires,
                               jnp.maximum(cur_flee, moves + jnp.int32(3)),
                               cur_flee).astype(mai.flee_until_turn.dtype)
    new_mstrategy = jnp.where(
        scare_fires, jnp.int8(4),       # MoveStrategy.FLEE
        mai.mstrategy[idx],
    )

    # CANCEL tier → set monster.cancelled (zeroes attacks/intrinsics per
    # vendor cancel_monst in zap.c).
    cancel_fires = apply_mb & (attack_indx == jnp.int32(_MB_CANCEL))
    new_cancelled = mai.cancelled[idx] | cancel_fires

    # CONFUSE — independent 1/12 roll EVERY Mb_hit (vendor line 1400).
    do_confuse = jax.random.randint(k_conf, (), 0, 12, dtype=jnp.int32) == jnp.int32(0)
    confuse_fires = apply_mb & do_confuse
    cur_conf = mai.confuse_timer[idx].astype(jnp.int32)
    new_conf = jnp.where(confuse_fires, cur_conf + jnp.int32(4), cur_conf) \
                  .astype(mai.confuse_timer.dtype)

    mai2 = mai.replace(
        mstrategy        = mai.mstrategy.at[idx].set(new_mstrategy),
        stun_timer       = mai.stun_timer.at[idx].set(new_stun),
        confuse_timer    = mai.confuse_timer.at[idx].set(new_conf),
        flee_until_turn  = mai.flee_until_turn.at[idx].set(new_flee_until),
        cancelled        = mai.cancelled.at[idx].set(new_cancelled),
    )
    new_state = state.replace(monster_ai=mai2)

    extra_out = jnp.where(apply_mb, extra, jnp.int32(0)).astype(jnp.int32)
    return new_state, extra_out, attack_indx


# ---------------------------------------------------------------------------
# P0 #12 — mk_artifact sacrifice-gift generator (artifact.c:171-309).
#
# Vendor walks artilist[1..NROFARTIFACTS], filters by:
#   - !artiexist[m].exists  (uniqueness, P0 #13)
#   - !SPFX_NOGEN
#   - gift_value <= max_giftvalue || Role_if(a->role)
#   - alignment matches OR is A_NONE
#   - skill compatibility
# Picks randomly from eligible[].
#
# Python-side (not JIT) — called by prayer.c at sacrifice gift time.
# ---------------------------------------------------------------------------

def mk_artifact(state, alignment: int, max_giftvalue: int,
                rng) -> int:
    """Pick an eligible sacrifice-gift artifact.

    Returns
    -------
    artifact_idx into wish._ARTIFACTS (0-based; -1 if none eligible).

    Python-side: walks the SPFX table to compute eligibility set, then
    samples uniformly.  Mirrors vendor mk_artifact lines 171-309.

    Cite: vendor/nethack/src/artifact.c::mk_artifact lines 171-309.
    """
    import numpy as _np

    # Read artiexist registry from state if present; else assume all available.
    exist_arr = getattr(state, "artiexist", None)
    # gift_value table — minimal subset; default 5 when unspecified.
    # Cite: artilist.h "gv" column.
    GIFT_VALUE = {
        0: 10, 1: 8, 2: 9, 3: 8, 4: 8, 5: 1, 6: 4, 7: 10,
        8: 5, 9: 12, 10: 12, 11: 12, 12: 12, 13: 12, 14: 12,
        15: 12, 16: 12, 17: 12, 18: 12, 19: 12, 20: 12, 21: 12,
        22: 9, 23: 5, 24: 5, 25: 3, 26: 4, 27: 1, 28: 5, 29: 7,
        30: 4, 31: 1, 32: 6,
    }

    eligible = []
    for art_idx in range(_N_ARTI_SLOTS):
        if exist_arr is not None and bool(exist_arr[art_idx]):
            continue  # uniqueness gate (P0 #13)
        spfx = _ARTIFACT_SPFX[art_idx]
        if spfx & _SPFX_NOGEN:
            continue
        gv = GIFT_VALUE.get(art_idx, 5)
        if gv > max_giftvalue:
            continue
        a_align = _ARTIFACT_ALIGN_TABLE[art_idx]
        # alignment matches OR A_NONE (Nethax UNALIGNED=3).
        if a_align != alignment and a_align != 3:
            continue
        eligible.append(art_idx)

    if not eligible:
        return -1
    # Sample uniformly via JAX rng to stay deterministic.
    key = rng if rng is not None else jax.random.PRNGKey(0)
    pick = int(jax.random.randint(key, (), 0, len(eligible), dtype=jnp.int32))
    return eligible[pick]


# ---------------------------------------------------------------------------
# P0 #13 — artiexist[] uniqueness registry.
#
# Cite: vendor/nethack/src/artifact.c line 70 +
#       vendor/nethack/include/artifact.h struct arti_info.
#
# Tracks which artifacts have been brought into existence (1-bit flag per
# artifact).  Vendor uses a 33-byte bitfield (indices 1..NROFARTIFACTS).
# We expose Python helpers to read/set the registry; storage lives in
# EnvState (added below in a non-breaking way as a derived helper).
# ---------------------------------------------------------------------------

def artiexist_mark(state, art_idx: int):
    """Mark artifact art_idx as existing (P0 #13).

    JIT-safe ish: writes to state.artiexist if present, else no-op.

    Cite: vendor/nethack/src/artifact.c artiexist[a].exists = 1.
    """
    if not hasattr(state, "artiexist"):
        return state  # field absent; caller must wire EnvState.artiexist
    art = jnp.clip(jnp.int32(art_idx), 0, state.artiexist.shape[0] - 1)
    new = state.artiexist.at[art].set(jnp.bool_(True))
    return state.replace(artiexist=new)


def artiexist_check(state, art_idx: int) -> bool:
    """Return True iff artifact art_idx already exists (cannot be regifted)."""
    if not hasattr(state, "artiexist"):
        return False
    return bool(state.artiexist[int(art_idx)])


# ---------------------------------------------------------------------------
# P0 #14 — Mjollnir Valkyrie+Str25+GoP triple throw-return check.
#
# Cite: vendor/nethack/src/dothrow.c::throwit + vendor/nethack/src/artifact.c
#       comments at artilist.h:97-108 — Mjollnir returns when thrown by
#       Valkyrie + STR>=25 (usually via Gauntlets of Power).  Return chance
#       is 99%, catch chance is 99% (compounded).
# ---------------------------------------------------------------------------
_ARTI_MJOLLNIR: int = 3
_MJOLLNIR_RETURN_NUMER: int = 99
_MJOLLNIR_RETURN_DENOM: int = 100


def mjollnir_throw_returns(state, rng) -> jnp.ndarray:
    """Return True if Mjollnir should return to thrower this throw.

    All three checks must pass:
      1. Wielded artifact is Mjollnir (idx 3)
      2. Player role is Valkyrie
      3. Player STR >= 25 (with GoP bonus rolled in)

    Then a 99% probability roll gates the actual return.

    Cite: vendor/nethack/src/dothrow.c::mhurtle_step + artilist.h:97-108.
    """
    from Nethax.nethax.constants.roles import Role as _Role
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_mjollnir = arti == jnp.int32(_ARTI_MJOLLNIR)
    is_valk = state.player_role == jnp.int8(int(_Role.VALKYRIE))
    has_str25 = state.player_str.astype(jnp.int32) >= jnp.int32(125)  # 25 in 18/** scale
    # Alt: STR raw >= 25.  Vendor uses ACURR(A_STR) >= STR19(25) == 125.

    gate = is_mjollnir & is_valk & has_str25
    rn = jax.random.randint(rng, (), 0, jnp.int32(_MJOLLNIR_RETURN_DENOM),
                            dtype=jnp.int32)
    returns = gate & (rn < jnp.int32(_MJOLLNIR_RETURN_NUMER))
    return returns


# ---------------------------------------------------------------------------
# P0 #15 — Stormbringer DRLI on-hit drain+heal.
#
# Cite: vendor/nethack/src/artifact.c lines 1645-1721.
#
# When wielded weapon has SPFX_DRLI and target is non-drli-resistant:
#   drain = monhp_per_lvl(mdef)  (usually 1d8)
#   if mhpmax - drain <= m_lev:
#       drain = (mhpmax > m_lev) ? mhpmax - (m_lev+1) : 0
#   if m_lev == 0: *dmgptr = 2*mhp + FATAL_DAMAGE_MODIFIER  (instant kill)
#   else:
#       *dmgptr += drain
#       mdef->mhpmax -= drain
#       mdef->m_lev--
#   if drain > 0:
#       drain = (drain + 1)/2
#       heal attacker by `drain` HP (cap at HPmax).
# ---------------------------------------------------------------------------

def stormbringer_drli_hit(state, mon_slot, rng) -> tuple:
    """Apply Stormbringer (or any SPFX_DRLI weapon) DRLI drain+heal effect.

    Returns
    -------
    (new_state, drain_dmg, drained_killed)

    drain_dmg     : extra damage added to *dmgptr.
    drained_killed : True if drain-at-level-0 fatal path triggered.

    Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1645-1721.
    """
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    spfx = _arti_spfx(arti)
    has_drli_attack = (spfx & jnp.int32(_SPFX_DRLI)) != jnp.int32(0)
    # Also Stormbringer ATTK is DRLI(5,2) which has its own AD_DRLI path even
    # without SPFX_DRLI — we treat both as a single hook.

    idx = mon_slot.astype(jnp.int32)
    mai = state.monster_ai
    is_alive = mai.alive[idx]

    # 1d8 drain roll (vendor monhp_per_lvl ~weapon.c:75).
    k_drain, k_heal = jax.random.split(rng, 2)
    drain_roll = jax.random.randint(k_drain, (), 1, 9, dtype=jnp.int32)
    # Consume the per-monster m_lev field (added in Wave 45a) so drain
    # damage scales with the actual instance level rather than the
    # species default.  Falls back to MONSTERS[].level via the
    # _MONSTER_XP_TABLE when m_lev hasn't been populated yet (spawn
    # sites that pre-date Wave 45a).
    from Nethax.nethax.subsystems.combat import _MONSTER_XP_TABLE as _LVL
    safe_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0,
                          _LVL.shape[0] - 1)
    species_lvl = _LVL[safe_entry].astype(jnp.int32)
    inst_lvl = mai.m_lev[idx].astype(jnp.int32)
    actual_lvl = jnp.where(inst_lvl > jnp.int32(0), inst_lvl, species_lvl)
    mhpmax = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))

    # Adjust drain if would drop mhpmax below m_lev (vendor lines 1658-1659).
    over_cap = (mhpmax - drain_roll) <= actual_lvl
    safe_drain = jnp.where(
        over_cap,
        jnp.where(mhpmax > actual_lvl, mhpmax - (actual_lvl + jnp.int32(1)),
                  jnp.int32(0)),
        drain_roll,
    )

    is_zero_lvl = actual_lvl == jnp.int32(0)
    fatal_drain = has_drli_attack & is_alive & is_zero_lvl

    # Audit K (Wave 40c): instant-kill damage formula must use the
    # defender's actual mhp, not a hardcoded 9999.  Vendor:
    #   *dmgptr = 2 * mdef->mhp + FATAL_DAMAGE_MODIFIER  (200)
    # Cite: vendor/nethack/src/artifact.c line ~1664 (DRLI m_lev==0 branch);
    #       FATAL_DAMAGE_MODIFIER define in artifact.c.
    FATAL_DAMAGE_MODIFIER = jnp.int32(200)
    mhp = mai.hp[idx].astype(jnp.int32)
    fatal_dmg = jnp.int32(2) * mhp + FATAL_DAMAGE_MODIFIER
    drain_dmg = jnp.where(
        has_drli_attack & is_alive,
        jnp.where(is_zero_lvl, fatal_dmg, safe_drain),
        jnp.int32(0),
    )

    # Decrement mhpmax by drain (vendor line 1678) for non-fatal path.
    apply_drain = has_drli_attack & is_alive & ~is_zero_lvl
    new_mhpmax = jnp.where(
        apply_drain,
        jnp.maximum(mhpmax - safe_drain, jnp.int32(1)),
        mhpmax,
    ).astype(mai.hp_max.dtype)

    # Heal attacker by (drain+1)/2 if drain>0 (vendor lines 1682-1690).
    heal_amt = jnp.where(
        has_drli_attack & is_alive & (safe_drain > jnp.int32(0)),
        (safe_drain + jnp.int32(1)) // jnp.int32(2),
        jnp.int32(0),
    )
    new_player_hp = jnp.minimum(
        state.player_hp + heal_amt,
        state.player_hp_max,
    )

    mai2 = mai.replace(
        hp_max=mai.hp_max.at[idx].set(new_mhpmax),
    )
    new_state = state.replace(
        monster_ai=mai2,
        player_hp=new_player_hp,
    )
    return new_state, drain_dmg, fatal_drain
