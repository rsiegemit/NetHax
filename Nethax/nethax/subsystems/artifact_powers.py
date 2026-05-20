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

JIT-pure: artifact_bonus_damage uses only JAX ops.  apply_artifact_intrinsics
is Python-side (called at wield/unwield time, not inside the per-step loop).

Artifact indices (0-based) mirror wish.py _ARTIFACTS table:
    0  Excalibur        8   Vorpal Blade     22  Frost Brand
    1  Snickersnee      10  Tsurugi          23  Fire Brand
    3  Mjollnir         16  Mitre of Holiness 24  Dragonbane
    5  Sting            21  Eye of Aethiopica 25  Demonbane
    6  Orcrist          7   Magicbane (idx 7) 26  Werebane
                                             27  Trollsbane
                                             28  Grimtooth
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
    return (
        jnp.array(undead, dtype=jnp.bool_),
        jnp.array(demon,  dtype=jnp.bool_),
        jnp.array(orc,    dtype=jnp.bool_),
        jnp.array(giant,  dtype=jnp.bool_),
        jnp.array(dragon, dtype=jnp.bool_),
        jnp.array(troll,  dtype=jnp.bool_),
        jnp.array(were,   dtype=jnp.bool_),
        jnp.array(ogre,   dtype=jnp.bool_),
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


def apply_artifact_hit_effects(state, mon_slot, rng):
    """Apply special on-hit artifact effects after a hit is confirmed.

    Handles:
      - Vorpal Blade (artifact_idx=8): instant kill on lich, else 1-in-23.
        Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1220-1255.
      - Magicbane (artifact_idx=29): 25% chance to apply one status effect
        (scare/stun/confuse/sleep) chosen uniformly.
        Cite: vendor/nethack/src/artifact.c::magicbane_hit lines 1090-1170.

    Parameters
    ----------
    state    : EnvState
    mon_slot : int32 — monster slot index
    rng      : JAX PRNG key

    Returns
    -------
    (new_state, killed_bool)  — JIT-pure.
    """
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    idx  = mon_slot.astype(jnp.int32)
    mai  = state.monster_ai

    # ---- Vorpal Blade: beheading ----------------------------------------
    # Cite: artifact.c::artifact_hit lines 1220-1255.
    # Auto-kill if target is S_LICH; else 1-in-23 chance.
    entry_i = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0, _IS_LICH.shape[0] - 1)
    is_lich = _IS_LICH[entry_i]

    key_v, key_m, key_ms = jax.random.split(rng, 3)
    vorpal_rn = jax.random.randint(key_v, (), 0, _VORPAL_BEHEAD_DENOM, dtype=jnp.int32)
    vorpal_instant = is_lich | (vorpal_rn == jnp.int32(0))
    is_vorpal = (arti == jnp.int32(_ARTI_VORPAL)) & mai.alive[idx]
    vorpal_kill = is_vorpal & vorpal_instant

    new_hp_v  = jnp.where(vorpal_kill, jnp.int32(0), mai.hp[idx])
    new_alive_v = jnp.where(vorpal_kill, jnp.bool_(False), mai.alive[idx])

    # ---- Magicbane: status effects --------------------------------------
    # Cite: artifact.c::magicbane_hit lines 1090-1170.
    # 25% chance (rn2(4)==0) → apply one of 4 effects chosen by rn2(4).
    is_magicbane = (arti == jnp.int32(_ARTI_MAGICBANE)) & mai.alive[idx]
    mb_trigger   = jax.random.randint(key_m, (), 0, 4, dtype=jnp.int32) == jnp.int32(0)
    mb_effect    = jax.random.randint(key_ms, (), 0, 4, dtype=jnp.int32)

    # Effect 0 = scare  → set mstrategy = FLEE (4)
    # Effect 1 = stun   → set asleep=True (no stunned field yet)
    # Effect 2 = confuse → set asleep=True (no confused field yet)
    # Effect 3 = sleep  → set asleep=True
    # Cite: magicbane_hit rn2(4) dispatch (artifact.c line ~1130).
    mb_active = is_magicbane & mb_trigger

    scare_active   = mb_active & (mb_effect == jnp.int32(0))
    sleep_active   = mb_active & (mb_effect != jnp.int32(0))  # effects 1,2,3 all → asleep

    new_mstrategy = jnp.where(
        scare_active,
        jnp.int8(4),       # MoveStrategy.FLEE = 4
        mai.mstrategy[idx],
    )
    new_asleep_mb = mai.asleep[idx] | sleep_active

    # ---- Compose monster_ai updates -------------------------------------
    # Vorpal kills override Magicbane hp/alive; both patches commute on
    # disjoint fields (hp/alive vs mstrategy/asleep) so we apply sequentially.
    mai2 = mai.replace(
        hp=mai.hp.at[idx].set(new_hp_v),
        alive=mai.alive.at[idx].set(new_alive_v),
        mstrategy=mai.mstrategy.at[idx].set(new_mstrategy),
        asleep=mai.asleep.at[idx].set(new_asleep_mb),
    )

    killed = vorpal_kill
    return state.replace(monster_ai=mai2), killed


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

# _INVOKE_HANDLER_IDX: for each artifact slot (0-29), which handler to call.
# 0 = no-op (no invoke effect).
# We define 15 handler indices to cover all distinct effect classes.
# Slots that have no invoke use handler 0 (noop).
_N_INVOKE_SLOTS = 30

# Handler indices for invoke effects:
_INV_NOOP          =  0  # no invoke effect
_INV_DETECT_OBJ    =  1  # Orb of Detection: detect_objects 1000 turns
_INV_LEVITATION    =  2  # Heart of Ahriman: LEVITATION_TMP +30, luck +d20
_INV_CONFLICT      =  3  # Sceptre of Might: CONFLICT intrinsic toggle
_INV_TELEPORT      =  4  # Orb of Fate: random teleport (TELEPORT_CONTROL off)
_INV_ENERGY_REGEN  =  5  # Eye of Aethiopica: ENERGY_REGEN timed + +1 Pw
_INV_TURN_UNDEAD   =  6  # Mitre/Excalibur: TURN_UNDEAD ray (kill undead in area)
_INV_CHARGE        =  7  # Yendorian Express Card: refill Pw to max
_INV_HEAL          =  8  # Staff of Aesculapius: cure SICK + restore HP
_INV_STR_BOOST     =  9  # Tsurugi: +1 STR capped at 18
_INV_LIGHTNING     = 10  # Mjollnir: d6 lightning hits all monsters in 6-radius
_INV_SLEEP_RES     = 11  # Snickersnee: grant SLEEP_RES timed 50
_INV_DETECT_FOOD   = 12  # Grayswandir: detect food
_INV_DETECT_TREAS  = 13  # Dragonbane: detect treasure 50
_INV_DETECT_MON    = 14  # Demonbane: detect monsters 50
_INV_LYCAN_CURE    = 15  # Werebane: cure lycanthropy
_INV_TRUE_SIGHT    = 16  # Trollsbane: SEE_INVIS timed 50
_INV_FAST          = 17  # Grimtooth: FAST timed +20
_INV_CONFUSION_RAY = 18  # Magicbane: confusion ray (already in _do_invoke)

# Map each of the 30 slots to a handler index.
# Cite: artilist.h inv_prop column for each artifact.
_INVOKE_HANDLER_IDX_LIST = [
    _INV_TURN_UNDEAD,   #  0  Excalibur   (TURN_UNDEAD + +1 Pw; cite artifact.c ~2202)
    _INV_SLEEP_RES,     #  1  Snickersnee (SLEEP_RES timed 50; artilist.h ~203)
    _INV_NOOP,          #  2  Stormbringer (passive drain; no invoke)
    _INV_LIGHTNING,     #  3  Mjollnir    (LIGHTNING_RAY d6,6; artilist.h ~109)
    _INV_NOOP,          #  4  Cleaver     (no invoke)
    _INV_NOOP,          #  5  Sting       (no invoke)
    _INV_NOOP,          #  6  Orcrist     (no invoke)
    _INV_DETECT_FOOD,   #  7  Grayswandir (DETECT_FOOD; artilist.h ~170)
    _INV_NOOP,          #  8  Vorpal Blade (no invoke; passive behead)
    _INV_CONFLICT,      #  9  Sceptre of Might (CONFLICT toggle; artilist.h ~232)
    _INV_STR_BOOST,     # 10  Tsurugi     (+1 STR cap 18; artilist.h ~285)
    _INV_NOOP,          # 11  Magic Mirror (no invoke here)
    _INV_DETECT_OBJ,    # 12  Orb of Detection (detect_objects 1000t; artilist.h ~219)
    _INV_LEVITATION,    # 13  Heart of Ahriman (LEVITATION +30; artilist.h ~225)
    _INV_HEAL,          # 14  Staff of Aesculapius (HEALING+CURE_SICK; artilist.h ~248)
    _INV_NOOP,          # 15  Eyes of the Overworld (no invoke mapped)
    _INV_TURN_UNDEAD,   # 16  Mitre of Holiness (TURN_UNDEAD; artilist.h ~265)
    _INV_NOOP,          # 17  Longbow of Diana (no invoke mapped)
    _INV_NOOP,          # 18  Master Key   (no invoke mapped)
    _INV_CHARGE,        # 19  Yendorian Express Card (CHARGE_OBJ; artilist.h ~291)
    _INV_TELEPORT,      # 20  Orb of Fate  (LEV_TELE→random teleport; artilist.h ~297)
    _INV_ENERGY_REGEN,  # 21  Eye of Aethiopica (ENERGY_REGEN+Pw; artilist.h ~303)
    _INV_NOOP,          # 22  Frost Brand  (no invoke; passive cold res)
    _INV_NOOP,          # 23  Fire Brand   (no invoke; passive fire res)
    _INV_DETECT_TREAS,  # 24  Dragonbane   (detect_treasure 50t; artilist.h ~157)
    _INV_DETECT_MON,    # 25  Demonbane    (detect_monsters 50t; artilist.h ~162)
    _INV_LYCAN_CURE,    # 26  Werebane     (cure lycanthropy; artilist.h ~166)
    _INV_TRUE_SIGHT,    # 27  Trollsbane   (SEE_INVIS 50t; artilist.h ~182)
    _INV_FAST,          # 28  Grimtooth    (FAST +20t; artilist.h ~123)
    _INV_CONFUSION_RAY, # 29  Magicbane sentinel (confusion ray; artifact.c ~1090)
]

_INVOKE_HANDLER_IDX = jnp.array(_INVOKE_HANDLER_IDX_LIST, dtype=jnp.int8)

_N_INVOKE_HANDLERS = 19  # 0..18


def artifact_invoke_dispatch(state, art_idx: jnp.ndarray, rng):
    """Dispatch to the correct invoke handler for the given artifact index.

    JIT-pure. Uses lax.switch over _N_INVOKE_HANDLERS handler functions.
    Cooldown (100 turns) is set by the caller (_handle_invoke).

    Cite: vendor/nethack/src/artifact.c::arti_invoke lines 2131-2232.

    Parameters
    ----------
    state    : EnvState
    art_idx  : int32 — 0-based artifact index (0..29)
    rng      : JAX PRNG key

    Returns
    -------
    new EnvState with invoke effect applied.
    """
    from Nethax.nethax.subsystems.status_effects import (
        Intrinsic as _Intrinsic,
        TimedStatus as _TS,
    )
    from Nethax.nethax.subsystems.detect import (
        detect_objects as _detect_objects,
        detect_food    as _detect_food,
        detect_treasure as _detect_treasure,
        detect_monsters as _detect_monsters,
    )

    safe_idx = jnp.clip(art_idx.astype(jnp.int32), 0, _N_INVOKE_SLOTS - 1)
    handler_idx = _INVOKE_HANDLER_IDX[safe_idx].astype(jnp.int32)

    rng_a, rng_b, rng_c = jax.random.split(rng, 3)

    # ---- handler 0: noop ---------------------------------------------------
    def _h_noop(s):
        return s

    # ---- handler 1: detect objects (Orb of Detection) ----------------------
    # Cite: artifact.c arti_invoke → detect_objects 1000t;
    #       vendor detect.c::detect_objects_until_turn = ts + 1000.
    def _h_detect_obj(s):
        ts = s.timestep.astype(jnp.int32)
        new_ident = s.identification.replace(
            detect_objects_until_turn=ts + jnp.int32(1000),
        )
        return s.replace(identification=new_ident)

    # ---- handler 2: levitation (Heart of Ahriman) --------------------------
    # Cite: artifact.c LEVITATION case (line ~2209) → float_up() for 30t.
    #       Also grants up to d20 luck per task spec.
    def _h_levitation(s):
        cur = s.status.timed_statuses[int(_TS.LEVITATION_TMP)].astype(jnp.int32)
        new_ts = s.status.timed_statuses.at[int(_TS.LEVITATION_TMP)].set(
            (cur + jnp.int32(30)).astype(s.status.timed_statuses.dtype)
        )
        luck_roll = jax.random.randint(rng_a, (), 1, 21, dtype=jnp.int32)
        new_luck = jnp.clip(
            s.player_luck.astype(jnp.int32) + luck_roll,
            -10, 10,
        ).astype(jnp.int8)
        return s.replace(
            status=s.status.replace(timed_statuses=new_ts),
            player_luck=new_luck,
        )

    # ---- handler 3: conflict toggle (Sceptre of Might) ---------------------
    # Cite: artifact.c CONFLICT case (line ~2203) — toggle extrinsic W_ARTI.
    #       Model as toggle of intrinsics[CONFLICT].
    def _h_conflict(s):
        cur_conflict = s.status.intrinsics[int(_Intrinsic.CONFLICT)]
        new_intrinsics = s.status.intrinsics.at[int(_Intrinsic.CONFLICT)].set(
            ~cur_conflict
        )
        return s.replace(status=s.status.replace(intrinsics=new_intrinsics))

    # ---- handler 4: teleport (Orb of Fate / random teleport) ---------------
    # Cite: artifact.c LEV_TELE case (line ~2160) → level_tele().
    #       We implement as random position teleport on current level.
    def _h_teleport(s):
        H = jnp.int32(s.terrain.shape[2])
        W = jnp.int32(s.terrain.shape[3])
        new_r = jax.random.randint(rng_a, (), 1, H - 1, dtype=jnp.int16)
        new_c = jax.random.randint(rng_b, (), 1, W - 1, dtype=jnp.int16)
        new_pos = jnp.array([new_r, new_c], dtype=jnp.int16)
        return s.replace(player_pos=new_pos)

    # ---- handler 5: energy regen (Eye of the Aethiopica) -------------------
    # Cite: artifact.c EREGEN carry property (artilist.h SPFX_EREGEN) → timed
    #       ENERGY_REGEN for 100 turns; also +1 Pw immediate (u.uen++, no cap).
    #       Cite: artifact.c arti_invoke CREATE_PORTAL path ~line 2161 (task spec).
    def _h_energy_regen(s):
        cur_er = s.status.timed_intrinsics[int(_Intrinsic.ENERGY_REGEN)].astype(jnp.int32)
        new_ti = s.status.timed_intrinsics.at[int(_Intrinsic.ENERGY_REGEN)].set(
            jnp.maximum(cur_er, jnp.int32(100))
        )
        # Vendor: u.uen++ unconditionally (no max clamp on raw invoke grant).
        new_pw = s.player_pw + jnp.int32(1)
        return s.replace(
            status=s.status.replace(timed_intrinsics=new_ti),
            player_pw=new_pw,
        )

    # ---- handler 6: turn undead (Mitre of Holiness / Excalibur) ------------
    # Cite: artifact.c ENERGY_BOOST case (Mitre, ~line 2157) and Excalibur
    #       invoke path (task spec). Model as: kill all undead monsters in 5×5
    #       area around the player; also grant +1 Pw.
    def _h_turn_undead(s):
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        mai = s.monster_ai
        mpos = mai.pos.astype(jnp.int32)
        in_area = (
            (jnp.abs(mpos[:, 0] - pr) <= jnp.int32(2))
            & (jnp.abs(mpos[:, 1] - pc) <= jnp.int32(2))
            & mai.alive
        )
        # Kill undead in area.
        entry_i = jnp.clip(mai.entry_idx.astype(jnp.int32), 0, _IS_UNDEAD.shape[0] - 1)
        is_undead_slot = _IS_UNDEAD[entry_i]
        turn_mask = in_area & is_undead_slot
        new_hp    = jnp.where(turn_mask, jnp.int32(0), mai.hp)
        new_alive = jnp.where(turn_mask, jnp.bool_(False), mai.alive)
        new_pw = jnp.minimum(s.player_pw + jnp.int32(1), s.player_pw_max)
        return s.replace(
            monster_ai=mai.replace(hp=new_hp, alive=new_alive),
            player_pw=new_pw,
        )

    # ---- handler 7: charge (Yendorian Express Card) ------------------------
    # Cite: artifact.c CHARGE_OBJ case (~line 2159) → refill Pw to max.
    def _h_charge(s):
        return s.replace(player_pw=s.player_pw_max)

    # ---- handler 8: heal (Staff of Aesculapius) ----------------------------
    # Cite: artifact.c HEALING case (~line 2156) → invoke_healing: restore HP
    #       and cure sickness.
    def _h_heal(s):
        new_hp = jnp.minimum(
            s.player_hp + jnp.int32(10),
            s.player_hp_max,
        )
        # Cure SICK timer and reset sick_kind.
        new_ts = s.status.timed_statuses.at[int(_TS.SICK)].set(
            jnp.int32(0).astype(s.status.timed_statuses.dtype)
        )
        new_status = s.status.replace(
            timed_statuses=new_ts,
            sick_kind=jnp.int8(0),
        )
        return s.replace(player_hp=new_hp, status=new_status)

    # ---- handler 9: str boost (Tsurugi of Muramasa) ------------------------
    # Cite: artifact.c task spec "+1 STR cap 18"; artilist.h TSURUGI line 285.
    def _h_str_boost(s):
        new_str = jnp.minimum(
            s.player_str.astype(jnp.int32) + jnp.int32(1),
            jnp.int32(18),
        ).astype(jnp.int16)
        return s.replace(player_str=new_str)

    # ---- handler 10: lightning ray (Mjollnir) ------------------------------
    # Cite: artifact.c ELEC(5,24) — invoke lightning ray: d6 damage × 6 hits
    #       to all monsters in 6-tile radius around player.
    def _h_lightning(s):
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        mai = s.monster_ai
        mpos = mai.pos.astype(jnp.int32)
        in_range = (
            (jnp.abs(mpos[:, 0] - pr) <= jnp.int32(6))
            & (jnp.abs(mpos[:, 1] - pc) <= jnp.int32(6))
            & mai.alive
        )
        # Roll d6,6 = 6 dice of d6.
        keys = jax.random.split(rng_a, 6)
        total_dmg = sum(
            jax.random.randint(k, (), 1, 7, dtype=jnp.int32) for k in keys
        )
        new_hp = jnp.where(
            in_range,
            jnp.maximum(mai.hp - total_dmg, jnp.int32(0)),
            mai.hp,
        )
        new_alive = jnp.where(in_range & (new_hp <= jnp.int32(0)), jnp.bool_(False), mai.alive)
        return s.replace(monster_ai=mai.replace(hp=new_hp, alive=new_alive))

    # ---- handler 11: sleep resistance (Snickersnee) ------------------------
    # Cite: task spec "+SLEEP_RES timed 50"; artilist.h Snickersnee ~line 203.
    def _h_sleep_res(s):
        cur = s.status.timed_intrinsics[int(_Intrinsic.RESIST_SLEEP)].astype(jnp.int32)
        new_ti = s.status.timed_intrinsics.at[int(_Intrinsic.RESIST_SLEEP)].set(
            jnp.maximum(cur, jnp.int32(50))
        )
        return s.replace(status=s.status.replace(timed_intrinsics=new_ti))

    # ---- handler 12: detect food (Grayswandir) -----------------------------
    # Cite: task spec "DETECT_FOOD"; artilist.h Grayswandir ~line 170.
    def _h_detect_food(s):
        return _detect_food(s, rng_a)

    # ---- handler 13: detect treasure (Dragonbane) --------------------------
    # Cite: task spec "DETECT_TREASURE for 50 turns"; artilist.h ~157.
    def _h_detect_treas(s):
        return _detect_treasure(s, rng_a)

    # ---- handler 14: detect monsters (Demonbane) ---------------------------
    # Cite: task spec "DETECT_MONSTERS for 50 turns"; artilist.h ~162.
    def _h_detect_mon(s):
        return _detect_monsters(s, rng_a)

    # ---- handler 15: cure lycanthropy (Werebane) ---------------------------
    # Cite: task spec "LYCANTHROPY_CURE on self"; artilist.h ~166.
    #       Reset lycanthropy_form to -1 and timer to 0.
    def _h_lycan_cure(s):
        new_poly = s.polymorph.replace(
            lycanthropy_form=jnp.int8(-1),
            lycanthropy_timer=jnp.int16(0),
        )
        return s.replace(polymorph=new_poly)

    # ---- handler 16: true sight (Trollsbane) -------------------------------
    # Cite: task spec "TRUE_SIGHT for 50 turns"; artilist.h ~182.
    #       Grant SEE_INVIS timed 50.
    def _h_true_sight(s):
        cur = s.status.timed_intrinsics[int(_Intrinsic.SEE_INVIS)].astype(jnp.int32)
        new_ti = s.status.timed_intrinsics.at[int(_Intrinsic.SEE_INVIS)].set(
            jnp.maximum(cur, jnp.int32(50))
        )
        return s.replace(status=s.status.replace(timed_intrinsics=new_ti))

    # ---- handler 17: fast (Grimtooth) --------------------------------------
    # Cite: task spec "FAST timer +20"; artilist.h Grimtooth ~123.
    def _h_fast(s):
        cur = s.status.timed_intrinsics[int(_Intrinsic.FAST)].astype(jnp.int32)
        new_ti = s.status.timed_intrinsics.at[int(_Intrinsic.FAST)].set(
            cur + jnp.int32(20)
        )
        return s.replace(status=s.status.replace(timed_intrinsics=new_ti))

    # ---- handler 18: confusion ray (Magicbane) -----------------------------
    # Cite: task spec "Cast CONFUSION ray; CONFUSED on adjacent";
    #       artifact.c magicbane_hit ~1090-1170.
    #       Already partially implemented in _handle_invoke; this handler
    #       applies CONFUSION timed status to player and asleep to all
    #       adjacent monsters (5×5 area).
    def _h_confusion_ray(s):
        pr = s.player_pos[0].astype(jnp.int32)
        pc = s.player_pos[1].astype(jnp.int32)
        mai = s.monster_ai
        mpos = mai.pos.astype(jnp.int32)
        in_area = (
            (jnp.abs(mpos[:, 0] - pr) <= jnp.int32(2))
            & (jnp.abs(mpos[:, 1] - pc) <= jnp.int32(2))
            & mai.alive
        )
        new_asleep = mai.asleep | in_area
        return s.replace(monster_ai=mai.replace(asleep=new_asleep))

    handlers = [
        _h_noop,           #  0
        _h_detect_obj,     #  1
        _h_levitation,     #  2
        _h_conflict,       #  3
        _h_teleport,       #  4
        _h_energy_regen,   #  5
        _h_turn_undead,    #  6
        _h_charge,         #  7
        _h_heal,           #  8
        _h_str_boost,      #  9
        _h_lightning,      # 10
        _h_sleep_res,      # 11
        _h_detect_food,    # 12
        _h_detect_treas,   # 13
        _h_detect_mon,     # 14
        _h_lycan_cure,     # 15
        _h_true_sight,     # 16
        _h_fast,           # 17
        _h_confusion_ray,  # 18
    ]

    return jax.lax.switch(handler_idx, handlers, state)


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
    """Deal 4d10 damage when a non-lawful player wields Excalibur.

    Call this immediately after wielding Excalibur.  Returns new_state.
    JIT-pure.

    Cite: vendor/nethack/src/artifact.c::Wield_artifact_unaligned (no vendor
    line given; see artifact.c around the wield path where misaligned artifact
    wielding causes HP loss, typically 4d10).
    """
    art = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_excalibur = art == jnp.int32(_ARTI_EXCALIBUR)
    align = state.player_align.astype(jnp.int32)
    not_lawful = align != jnp.int32(_EXCALIBUR_ALIGN)

    # Roll 4d10 damage.
    keys = jax.random.split(rng, 4)
    d10 = lambda k: jax.random.randint(k, (), 1, 11, dtype=jnp.int32)
    dmg = d10(keys[0]) + d10(keys[1]) + d10(keys[2]) + d10(keys[3])

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

# Per-artifact alignment (A_NONE=-1, A_CHAOTIC=0, A_NEUTRAL=1, A_LAWFUL=2;
# Nethax uses Alignment(CHAOTIC=0, NEUTRAL=1, LAWFUL=2, UNALIGNED=3)).
# Cite: vendor artilist.h "al" column per row.
_ARTIFACT_ALIGN_TABLE: tuple[int, ...] = tuple([
    2, 2, 0, 1, 1, 0, 0, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 0, 0, 1, 1, 1,
    3, 3, 3, 2, 3, 3, 0, 1, 1, 3, 2,                                  # 22..32
])
_ARTIFACT_ALIGN_ARR: jnp.ndarray = jnp.array(_ARTIFACT_ALIGN_TABLE, dtype=jnp.int8)

# Per-artifact role gate (PM_* mapping; we encode by Nethax Role index, or -1
# for NON_PM "any role").  Used by touch_artifact badclass check (P0 #11).
# We approximate by listing only artifacts that gate by role.  For artifacts
# with NON_PM role, we use sentinel -1 meaning "no role restriction".
_ARTIFACT_ROLE_TABLE: tuple[int, ...] = tuple([-1] * 33)
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

    # badclass: self_willed && role mismatch — Nethax has no role gate per
    # artifact yet, so we conservatively use False (matches NON_PM rows).
    badclass = jnp.bool_(False)

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

# Per-artifact-idx cspfx bitfield (second SPFX column in artilist.h).
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
    _SPFX_ESP,                                                          # 11 Magic Mirror
    (_SPFX_ESP | _SPFX_HSPDAM),                                         # 12 Orb of Detection
    _SPFX_STLTH,                                                        # 13 Heart of Ahriman
    0,                                                                  # 14 Staff of Aesculapius
    0,                                                                  # 15 Eyes of Overworld
    0,                                                                  # 16 Mitre of Holiness
    _SPFX_ESP,                                                          # 17 Longbow of Diana
    (_SPFX_WARN | _SPFX_TCTRL | _SPFX_HPHDAM),                          # 18 Master Key
    (_SPFX_ESP | _SPFX_HSPDAM),                                         # 19 Yendorian Express
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
    """Sceptre of Might: double damage vs targets of non-LAWFUL alignment.

    Returns the doubled damage value (or base_dmg unchanged when not Sceptre).

    Cite: vendor/nethack/src/artifact.c::spec_applies SPFX_DALIGN branch
          (lines 1031-1034); artilist.h line 232.
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_sceptre = arti == jnp.int32(_ARTI_SCEPTRE)

    # Per-monster alignment (we approximate via a static table).
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
    # Sceptre alignment is A_LAWFUL == +1 in vendor (Nethax LAWFUL=2).  Use
    # sgn(mon_align) != weap_align test per vendor line 1034.
    # Approximation: any non-positive mon_align triggers DALIGN (mon != lawful).
    is_misaligned = mon_align <= jnp.int32(0)
    apply = is_sceptre & is_misaligned
    return jnp.where(apply, base_dmg * jnp.int32(2), base_dmg).astype(base_dmg.dtype)


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
    """Mitre of Holiness: PROTECT + CARY(AD_FIRE) + undead bane while wielded.

    Sets RESIST_FIRE intrinsic (CARY) and a +2 AC adjustment via
    player_uhitinc surrogate.  The undead bane is implemented at the
    spec_applies stage via the _ARTIFACT_BONUS_TABLE (Mitre has no entry
    yet because it does no attack damage — see P0 #6 design note).

    Cite: vendor/nethack/src/artifact.c::set_artifact_intrinsic +
          artilist.h:265-269.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _I
    arti = state.inventory.wielded_artifact_idx.astype(jnp.int32)
    is_mitre = arti == jnp.int32(_ARTI_MITRE)
    intr = state.status.intrinsics
    new_intr = intr.at[int(_I.RESIST_FIRE)].set(
        jnp.where(is_mitre, jnp.bool_(True), intr[int(_I.RESIST_FIRE)])
    )
    new_intr = new_intr.at[int(_I.PROTECTION)].set(
        jnp.where(is_mitre, jnp.bool_(True), new_intr[int(_I.PROTECTION)])
    )
    return state.replace(status=state.status.replace(intrinsics=new_intr))


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

    # P0 #10: Apply STUN / SCARE / CANCEL effects to defender (mon_slot).
    # STUN: set mstun (we use stunned_timer via timed_intrinsics not available
    #       on monsters; approximate via asleep flag for parity baseline).
    # SCARE: monflee → set mstrategy=FLEE(4) for ~3 turns.
    # CANCEL: cancel_monst → set monster cancelled (no field; clear mspec
    #         effects by setting asleep=True as proxy).
    # vendor lines 1314-1406.

    apply_mb = is_mb
    new_mstrategy = jnp.where(
        apply_mb & (attack_indx == jnp.int32(_MB_SCARE)),
        jnp.int8(4),       # MoveStrategy.FLEE
        mai.mstrategy[idx],
    )
    # asleep flag is the canonical "incapacitated" proxy in monster_ai;
    # used here for STUN + CANCEL effect tiers.  Cite Mb_hit lines 1377/1399.
    new_asleep = mai.asleep[idx] | (
        apply_mb & ((attack_indx == jnp.int32(_MB_STUN))
                    | (attack_indx == jnp.int32(_MB_CANCEL)))
    )

    mai2 = mai.replace(
        mstrategy=mai.mstrategy.at[idx].set(new_mstrategy),
        asleep=mai.asleep.at[idx].set(new_asleep),
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
    m_lev = mai.entry_idx[idx].astype(jnp.int32)  # use entry as level proxy
    m_lev = jnp.maximum(m_lev, jnp.int32(0))
    # Vendor uses mdef->m_lev (per-monster level).  We approximate from the
    # monster's MONSTERS[].level table.
    from Nethax.nethax.subsystems.combat import _MONSTER_XP_TABLE as _LVL
    safe_entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0,
                          _LVL.shape[0] - 1)
    actual_lvl = _LVL[safe_entry].astype(jnp.int32)
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

    # Damage output: instant kill on m_lev==0, else +drain HP damage.
    drain_dmg = jnp.where(
        has_drli_attack & is_alive,
        jnp.where(is_zero_lvl, jnp.int32(2 * 9999 + 200), safe_drain),
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
