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
# Magicbane is absent from the 22-entry wish.py table; we use index 29 as a
# synthetic sentinel (first slot beyond the table).  Callers set
# wielded_artifact_idx=29 to represent Magicbane.
# Cite: vendor/nethack/src/artifact.c::magicbane_hit lines 1090-1170.
_ARTI_MAGICBANE   = 29


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
    return (
        jnp.array(undead, dtype=jnp.bool_),
        jnp.array(demon,  dtype=jnp.bool_),
        jnp.array(orc,    dtype=jnp.bool_),
        jnp.array(giant,  dtype=jnp.bool_),
        jnp.array(dragon, dtype=jnp.bool_),
        jnp.array(troll,  dtype=jnp.bool_),
        jnp.array(were,   dtype=jnp.bool_),
    )


(
    _IS_UNDEAD,
    _IS_DEMON,
    _IS_ORC,
    _IS_GIANT,
    _IS_DRAGON,
    _IS_TROLL,
    _IS_WERE,
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
    ], axis=0)  # shape [8, N_MONSTERS]


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
    ( 0, 4,  _PRED_UNDEAD),  # Excalibur    — +d4 vs undead
    ( 1, 8,  _PRED_ALWAYS),  # Snickersnee  — +d8 always
    ( 3, 24, _PRED_GIANT),   # Mjollnir     — +d24 vs giants (ELEC)
    ( 5, 5,  _PRED_ORC),     # Sting        — +d5 vs orcs
    ( 6, 5,  _PRED_ORC),     # Orcrist      — +d5 vs orcs
    (22, 6,  _PRED_ALWAYS),  # Frost Brand  — +d6 cold, always (COLD(5,0))
    (23, 6,  _PRED_ALWAYS),  # Fire Brand   — +d6 fire, always (FIRE(5,0))
    (24, 4,  _PRED_DRAGON),  # Dragonbane   — +d4 vs dragons
    (25, 4,  _PRED_DEMON),   # Demonbane    — +d4 vs demons
    (26, 4,  _PRED_WERE),    # Werebane     — +d4 vs were-creatures
    (27, 4,  _PRED_TROLL),   # Trollsbane   — +d4 vs trolls
    (28, 2,  _PRED_ALWAYS),  # Grimtooth    — +d2 always (bypasses spec_applies)
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
