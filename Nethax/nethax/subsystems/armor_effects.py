"""Worn-armor side effects — intrinsic grants and stat bonuses.

Canonical source: vendor/nethack/src/do_wear.c
  Boots_on/off     (~line 187)
  Cloak_on/off     (~line 325)
  Helmet_on/off    (~line 433)
  Gloves_on/off    (~line 575)

Design
------
`apply_armor_effects(state)` is the single entry point.  It:
  1. Clears all armor-sourced intrinsics and stat bonuses.
  2. Iterates the seven worn-armor slots (JAX-pure: jnp.where, no Python
     control flow over dynamic values).
  3. Grants intrinsics / stat bonuses for each currently-worn item type.

Call it after every wear *and* take-off so the state is always consistent.
The caller (`inventory.py`) just records the slot change, then calls here.

Stat bonus storage
------------------
`InventoryState.armor_stat_bonus` — int8[6]: additive bonuses for
[str, dex, con, int, wis, cha] in that order, sourced only from armor.
The base stat (e.g. `state.player_int`) is the original rolled value;
effective stat = base + armor_stat_bonus[idx].  apply_armor_effects
recomputes this array from scratch each call (idempotent, JIT-pure).

Alignment swap (HELM_OF_OPPOSITE_ALIGNMENT)
-------------------------------------------
do_wear.c Helmet_on (~line 462): uchangealign(-u.ualign.type, A_CG_HELM_ON).
Mapping: CHAOTIC=0, NEUTRAL=1, LAWFUL=2 (prayer.py Alignment enum).
We swap LAWFUL(2) ↔ CHAOTIC(0); NEUTRAL(1) stays NEUTRAL(1).

MUMMY_WRAPPING
--------------
do_wear.c Cloak_on (~line 347): blocks INVIS while worn.
We clear the INVIS intrinsic bit when mummy wrapping is in the cloak slot.
"""
from __future__ import annotations

import jax.numpy as jnp

from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus
from Nethax.nethax.subsystems.character import ObjType


# ---------------------------------------------------------------------------
# Armor type IDs — canonical aliases into ObjType (character.py)
# ---------------------------------------------------------------------------

# Helms
_ORCISH_HELM            = ObjType.ORCISH_HELM
_HELM_OF_BRILLIANCE     = ObjType.HELM_OF_BRILLIANCE
_HELM_OF_OPPOSITE_ALIGN = ObjType.HELM_OF_OPPOSITE_ALIGNMENT
_HELM_OF_TELEPATHY      = ObjType.HELM_OF_TELEPATHY
_DUNCE_CAP              = ObjType.DUNCE_CAP
_CORNUTHAUM             = ObjType.CORNUTHAUM
_TINFOIL_HAT            = ObjType.TINFOIL_HAT

# Cloaks
_CLOAK_OF_PROTECTION    = ObjType.CLOAK_OF_PROTECTION
_CLOAK_OF_INVISIBILITY  = ObjType.CLOAK_OF_INVISIBILITY
_CLOAK_OF_MAGIC_RES     = ObjType.CLOAK_OF_MAGIC_RESISTANCE
_CLOAK_OF_DISPLACEMENT  = ObjType.CLOAK_OF_DISPLACEMENT
_ELVEN_CLOAK            = ObjType.ELVEN_CLOAK
_MUMMY_WRAPPING         = ObjType.MUMMY_WRAPPING
# LEATHER_CLOAK and OILSKIN_CLOAK provide basic AC only — no special effects.

# Gloves
_LEATHER_GLOVES         = ObjType.LEATHER_GLOVES
_GAUNTLETS_OF_FUMBLING  = ObjType.GAUNTLETS_OF_FUMBLING
_GAUNTLETS_OF_POWER     = ObjType.GAUNTLETS_OF_POWER
_GAUNTLETS_OF_DEXTERITY = ObjType.GAUNTLETS_OF_DEXTERITY

# Boots
_LOW_BOOTS              = ObjType.LOW_BOOTS
_HIGH_BOOTS             = ObjType.HIGH_BOOTS
_LEVITATION_BOOTS       = ObjType.LEVITATION_BOOTS
_FUMBLE_BOOTS           = ObjType.FUMBLE_BOOTS
_KICKING_BOOTS          = ObjType.KICKING_BOOTS
_JUMPING_BOOTS          = ObjType.JUMPING_BOOTS
_WATER_WALKING_BOOTS    = ObjType.WATER_WALKING_BOOTS
_ELVEN_BOOTS            = ObjType.ELVEN_BOOTS
_SPEED_BOOTS            = ObjType.SPEED_BOOTS

# Alignment sentinels (prayer.py Alignment enum)
_ALIGN_CHAOTIC = 0
_ALIGN_NEUTRAL = 1
_ALIGN_LAWFUL  = 2

# Stat-bonus array indices
_STAT_STR = 0
_STAT_DEX = 1
_STAT_CON = 2
_STAT_INT = 3
_STAT_WIS = 4
_STAT_CHA = 5
N_STAT_BONUS = 6

# Wizard role value (character.py / constants/roles.py Role.WIZARD = 12)
_ROLE_WIZARD = 12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _type_id_at_slot(state, armor_slot_idx: int) -> jnp.ndarray:
    """Return the type_id of the item worn in armor_slot_idx, or 0 if empty."""
    inv_idx = state.inventory.worn_armor[armor_slot_idx].astype(jnp.int32)
    safe_idx = jnp.clip(inv_idx, 0, state.inventory.items.type_id.shape[0] - 1)
    return jnp.where(
        inv_idx >= 0,
        state.inventory.items.type_id[safe_idx].astype(jnp.int32),
        jnp.int32(0),
    )


def _enchant_at_slot(state, armor_slot_idx: int) -> jnp.ndarray:
    """Return the enchantment of the item worn in armor_slot_idx, or 0."""
    inv_idx = state.inventory.worn_armor[armor_slot_idx].astype(jnp.int32)
    safe_idx = jnp.clip(inv_idx, 0, state.inventory.items.enchantment.shape[0] - 1)
    return jnp.where(
        inv_idx >= 0,
        state.inventory.items.enchantment[safe_idx].astype(jnp.int32),
        jnp.int32(0),
    )


def _is_wearing(type_id: jnp.ndarray, target: int) -> jnp.ndarray:
    return type_id == jnp.int32(target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_armor_effects(state) -> object:
    """Recompute all armor-sourced intrinsics and stat bonuses from scratch.

    Call after any wear or take-off.  Idempotent and JIT-pure.

    cite: vendor/nethack/src/do_wear.c  Boots_on/off, Cloak_on/off,
          Helmet_on/off, Gloves_on/off.
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    # ------------------------------------------------------------------
    # 1. Read current worn type_ids
    # ------------------------------------------------------------------
    helm_tid   = _type_id_at_slot(state, int(ArmorSlot.HELM))
    helm_ench  = _enchant_at_slot(state, int(ArmorSlot.HELM))
    cloak_tid  = _type_id_at_slot(state, int(ArmorSlot.CLOAK))
    glove_tid  = _type_id_at_slot(state, int(ArmorSlot.GLOVES))
    glove_ench = _enchant_at_slot(state, int(ArmorSlot.GLOVES))
    boot_tid   = _type_id_at_slot(state, int(ArmorSlot.BOOTS))

    # ------------------------------------------------------------------
    # 2. Start from the current intrinsics array (permanent + timed).
    #    We only clear the specific bits that armor can grant, then
    #    re-grant them if the relevant gear is still worn.
    # ------------------------------------------------------------------
    intr = state.status.intrinsics

    # Bits that are exclusively armor-sourced in this subsystem:
    armor_bits = [
        int(Intrinsic.TELEPATHY),
        int(Intrinsic.MAGIC_RESIST),
        int(Intrinsic.INVIS),
        int(Intrinsic.DISPLACED),
        int(Intrinsic.STEALTH),
        int(Intrinsic.JUMPING),
        int(Intrinsic.LEVITATION),
        int(Intrinsic.WWALKING),
        int(Intrinsic.FAST),
        int(Intrinsic.CLAIRVOYANT),
    ]
    for bit in armor_bits:
        intr = intr.at[bit].set(jnp.bool_(False))

    # ------------------------------------------------------------------
    # 3. HELM effects
    # cite: do_wear.c Helmet_on (~line 433) / Helmet_off (~line 517)
    # ------------------------------------------------------------------

    # HELM_OF_TELEPATHY / TINFOIL_HAT → TELEPATHY
    # cite: do_wear.c line 446 (HELM_OF_TELEPATHY handled by setworn prop),
    #       NetHack 3.7 TINFOIL_HAT also grants TELEPAT.
    has_telep = _is_wearing(helm_tid, _HELM_OF_TELEPATHY) | _is_wearing(helm_tid, _TINFOIL_HAT)
    intr = intr.at[int(Intrinsic.TELEPATHY)].set(has_telep)

    # ------------------------------------------------------------------
    # 4. CLOAK effects
    # cite: do_wear.c Cloak_on (~line 325)
    # ------------------------------------------------------------------

    # CLOAK_OF_MAGIC_RESISTANCE → MAGIC_RESIST
    # cite: do_wear.c line 334 (oc_oprop = ANTIMAGIC for this cloak)
    has_mr = _is_wearing(cloak_tid, _CLOAK_OF_MAGIC_RES)
    intr = intr.at[int(Intrinsic.MAGIC_RESIST)].set(has_mr)

    # CLOAK_OF_INVISIBILITY → INVIS (unless mummy wrapping blocks it)
    # cite: do_wear.c line 355
    # MUMMY_WRAPPING blocks any INVIS while worn:
    # cite: do_wear.c line 347 (Cloak_on: "if HInvis... you can no longer see yourself")
    has_invis = _is_wearing(cloak_tid, _CLOAK_OF_INVISIBILITY)
    has_mummy = _is_wearing(cloak_tid, _MUMMY_WRAPPING)
    intr = intr.at[int(Intrinsic.INVIS)].set(has_invis & ~has_mummy)

    # CLOAK_OF_DISPLACEMENT → DISPLACED
    # cite: do_wear.c line 344 (toggle_displacement)
    has_disp = _is_wearing(cloak_tid, _CLOAK_OF_DISPLACEMENT)
    intr = intr.at[int(Intrinsic.DISPLACED)].set(has_disp)

    # ELVEN_CLOAK → STEALTH (+1 stealth; binary flag here)
    # cite: do_wear.c line 341 (toggle_stealth)
    has_stealth_cloak = _is_wearing(cloak_tid, _ELVEN_CLOAK)

    # ------------------------------------------------------------------
    # 5. BOOT effects
    # cite: do_wear.c Boots_on (~line 187)
    # ------------------------------------------------------------------

    # LEVITATION_BOOTS → LEVITATION
    # cite: do_wear.c line 235
    has_lev = _is_wearing(boot_tid, _LEVITATION_BOOTS)
    intr = intr.at[int(Intrinsic.LEVITATION)].set(has_lev)

    # WATER_WALKING_BOOTS → WWALKING
    # cite: do_wear.c line 199 (oc_oprop = WWALKING)
    has_ww = _is_wearing(boot_tid, _WATER_WALKING_BOOTS)
    intr = intr.at[int(Intrinsic.WWALKING)].set(has_ww)

    # JUMPING_BOOTS → JUMPING
    # cite: do_wear.c line 196 (JUMPING_BOOTS, oc_oprop = JUMPING)
    has_jump = _is_wearing(boot_tid, _JUMPING_BOOTS)
    intr = intr.at[int(Intrinsic.JUMPING)].set(has_jump)

    # ELVEN_BOOTS → STEALTH
    # cite: do_wear.c line 228 (toggle_stealth)
    has_stealth_boots = _is_wearing(boot_tid, _ELVEN_BOOTS)

    # Combine stealth sources
    intr = intr.at[int(Intrinsic.STEALTH)].set(has_stealth_cloak | has_stealth_boots)

    # SPEED_BOOTS → FAST
    # cite: do_wear.c line 219-226 (SPEED_BOOTS) — oc_oprop = FAST is
    #       conferred via setworn: u.uprops[FAST].extrinsic |= W_ARMF.
    #       See objects.h line 707-708 (BOOTS("speed boots", ... FAST ...)).
    has_fast = _is_wearing(boot_tid, _SPEED_BOOTS)
    intr = intr.at[int(Intrinsic.FAST)].set(has_fast)

    # ------------------------------------------------------------------
    # 6. FUMBLING (timed status) from gauntlets or boots
    # cite: do_wear.c line 584 (GAUNTLETS_OF_FUMBLING), line 231 (FUMBLE_BOOTS)
    # Fumbling is a timed status; wearing the item sets a nonzero timer.
    # Taking off clears it.  We represent "armor is worn = fumbling active"
    # by setting timer to 1 when worn (idempotent recompute).
    # ------------------------------------------------------------------
    timed = state.status.timed_statuses
    has_fum_gloves = _is_wearing(glove_tid, _GAUNTLETS_OF_FUMBLING)
    has_fum_boots  = _is_wearing(boot_tid,  _FUMBLE_BOOTS)
    fum_active = has_fum_gloves | has_fum_boots
    # Only set to 1 if currently 0 (so real timers aren't clobbered).
    # When neither is worn we clear the armor contribution to 0.
    cur_fum = timed[int(TimedStatus.FUMBLING)]
    new_fum = jnp.where(fum_active, jnp.maximum(cur_fum, jnp.int32(1)), jnp.int32(0))
    timed = timed.at[int(TimedStatus.FUMBLING)].set(new_fum)

    # ------------------------------------------------------------------
    # 7. Stat bonuses — recomputed from zero each call.
    # ------------------------------------------------------------------
    bonus = jnp.zeros((N_STAT_BONUS,), dtype=jnp.int8)

    # HELM_OF_BRILLIANCE → +spe Int (and +spe Wis, per adj_abon)
    # cite: do_wear.c line 451-452: adj_abon(uarmh, uarmh->spe)
    #       attrib.c adj_abon adds spe to both A_INT and A_WIS for brilliance.
    brilliance_bonus = jnp.where(
        _is_wearing(helm_tid, _HELM_OF_BRILLIANCE),
        helm_ench.astype(jnp.int8),
        jnp.int8(0),
    )
    bonus = bonus.at[_STAT_INT].add(brilliance_bonus)
    bonus = bonus.at[_STAT_WIS].add(brilliance_bonus)

    # DUNCE_CAP → -2 Int, -2 Wis
    # cite: do_wear.c line 475 fallthrough; vendor adj_abon for dunce cap
    #       nets -2 to both INT and WIS (DUNCE_CAP spe is treated as -2).
    dunce_penalty = jnp.where(
        _is_wearing(helm_tid, _DUNCE_CAP),
        jnp.int8(-2),
        jnp.int8(0),
    )
    bonus = bonus.at[_STAT_INT].add(dunce_penalty)
    bonus = bonus.at[_STAT_WIS].add(dunce_penalty)

    # ORCISH_HELM → no stat penalty
    # cite: objects.h line 448-450 — ORCISH_HELM oc_oprop = 0.
    # cite: do_wear.c Helmet_on/off — ORCISH_HELM has no special case (no
    #       call to adj_abon or ABON manipulation).  Vendor applies no
    #       intrinsic or stat effect for wearing ORCISH_HELM.

    # CORNUTHAUM → +1 Cha for wizard, -1 Cha otherwise
    # cite: do_wear.c line 454-460: ABON(A_CHA) += (Role_if(PM_WIZARD) ? 1 : -1)
    is_wizard = state.player_role == jnp.int8(_ROLE_WIZARD)
    has_cornu = _is_wearing(helm_tid, _CORNUTHAUM)
    cornu_bonus = jnp.where(
        has_cornu,
        jnp.where(is_wizard, jnp.int8(1), jnp.int8(-1)),
        jnp.int8(0),
    )
    bonus = bonus.at[_STAT_CHA].add(cornu_bonus)

    # CORNUTHAUM → CLAIRVOYANT for wizards (blocked for non-wizards).
    # cite: objects.h line 457-461 — CORNUTHAUM oc_oprop = CLAIRVOYANT.
    # cite: worn.c line 38-44 w_blocks: CORNUTHAUM blocks CLAIRVOYANT when
    #       worn by a non-wizard.  Net effect: wizards have CLAIRVOYANT,
    #       non-wizards do not (the extrinsic is set but also blocked).
    intr = intr.at[int(Intrinsic.CLAIRVOYANT)].set(has_cornu & is_wizard)

    # GAUNTLETS_OF_DEXTERITY → +spe Dex
    # cite: do_wear.c line 592-593: adj_abon(uarmg, uarmg->spe)
    dex_bonus = jnp.where(
        _is_wearing(glove_tid, _GAUNTLETS_OF_DEXTERITY),
        glove_ench.astype(jnp.int8),
        jnp.int8(0),
    )
    bonus = bonus.at[_STAT_DEX].add(dex_bonus)

    # GAUNTLETS_OF_POWER → effective STR = 125 (STR19(25))
    # cite: do_wear.c line 588-591 (Gloves_on: no adj_abon for GoP).
    # cite: attrib.c::acurr lines 1213-1215 — when uarmg == GAUNTLETS_OF_POWER,
    #       result is forced to STR19(25) = 125 regardless of base/abon.
    # The vendor does NOT add `spe` to STR ABON for GoP (see adj_abon at
    # do_wear.c:3319-3336 which only handles DEX/INT/WIS).  The STR=125 cap
    # is enforced in acurr (not here) — armor_stat_bonus stays 0 for STR
    # when GoP is worn; downstream acurr-equivalent applies the cap.

    # ------------------------------------------------------------------
    # 8. Alignment swap — HELM_OF_OPPOSITE_ALIGNMENT
    # cite: do_wear.c line 462-471: uchangealign(-u.ualign.type, A_CG_HELM_ON)
    #   NetHack A_LAWFUL=1, A_CHAOTIC=-1 → our LAWFUL=2 ↔ CHAOTIC=0; NEUTRAL stays.
    # ------------------------------------------------------------------
    wearing_hoa = _is_wearing(helm_tid, _HELM_OF_OPPOSITE_ALIGN)
    orig_align  = state.player_align.astype(jnp.int32)
    # swap: 2→0, 0→2, 1→1
    swapped_align = jnp.where(
        orig_align == jnp.int32(_ALIGN_LAWFUL), jnp.int32(_ALIGN_CHAOTIC),
        jnp.where(orig_align == jnp.int32(_ALIGN_CHAOTIC), jnp.int32(_ALIGN_LAWFUL),
                  orig_align)
    )
    new_align = jnp.where(wearing_hoa, swapped_align, orig_align).astype(jnp.int8)

    # ------------------------------------------------------------------
    # 9. Write back
    # ------------------------------------------------------------------
    new_status = state.status.replace(
        intrinsics=intr,
        timed_statuses=timed,
    )
    new_inv = state.inventory.replace(
        armor_stat_bonus=bonus,
    )
    return state.replace(
        status=new_status,
        inventory=new_inv,
        player_align=new_align,
    )
