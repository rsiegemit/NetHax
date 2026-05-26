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

from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus, N_INTRINSICS
from Nethax.nethax.subsystems.character import ObjType


# ---------------------------------------------------------------------------
# Worn-slot W_xxx bit-masks (vendor/nethack/include/prop.h lines 101-122).
# Used to populate ``status.extrinsic`` / ``status.blocked`` per-prop
# bitmasks in vendor-parity form (one bit per worn slot granting / blocking
# the prop).
# ---------------------------------------------------------------------------
W_ARM   = 0x00000001  # body armor
W_ARMC  = 0x00000002  # cloak
W_ARMH  = 0x00000004  # helmet
W_ARMS  = 0x00000008  # shield
W_ARMG  = 0x00000010  # gloves
W_ARMF  = 0x00000020  # boots
W_ARMU  = 0x00000040  # undershirt
W_AMUL  = 0x00010000  # amulet
W_RINGL = 0x00020000  # left ring
W_RINGR = 0x00040000  # right ring

# Vendor BOLT_LIM (vendor/nethack/include/hack.h:49).  Telepathy range is
# (BOLT_LIM * BOLT_LIM) * nobjs per worn.c::recalc_telepat_range:66.
_BOLT_LIM = 8


# ---------------------------------------------------------------------------
# Armor type IDs — canonical aliases into ObjType (character.py)
# ---------------------------------------------------------------------------

# Helms
_ORCISH_HELM            = ObjType.ORCISH_HELM
_HELM_OF_BRILLIANCE     = ObjType.HELM_OF_BRILLIANCE
_HELM_OF_OPPOSITE_ALIGN = ObjType.HELM_OF_OPPOSITE_ALIGNMENT
_HELM_OF_TELEPATHY      = ObjType.HELM_OF_TELEPATHY
_HELM_OF_CAUTION        = ObjType.HELM_OF_CAUTION
_DUNCE_CAP              = ObjType.DUNCE_CAP
_CORNUTHAUM             = ObjType.CORNUTHAUM
_TINFOIL_HAT            = ObjType.TINFOIL_HAT

# Dragon scale mails and scales (BODY slot).
# cite: vendor/nethack/include/objects.h:497-553 (DRGN_ARMR).
_GRAY_DSM    = ObjType.GRAY_DRAGON_SCALE_MAIL
_GOLD_DSM    = ObjType.GOLD_DRAGON_SCALE_MAIL
_SILVER_DSM  = ObjType.SILVER_DRAGON_SCALE_MAIL
_RED_DSM     = ObjType.RED_DRAGON_SCALE_MAIL
_WHITE_DSM   = ObjType.WHITE_DRAGON_SCALE_MAIL
_ORANGE_DSM  = ObjType.ORANGE_DRAGON_SCALE_MAIL
_BLACK_DSM   = ObjType.BLACK_DRAGON_SCALE_MAIL
_BLUE_DSM    = ObjType.BLUE_DRAGON_SCALE_MAIL
_GREEN_DSM   = ObjType.GREEN_DRAGON_SCALE_MAIL
_YELLOW_DSM  = ObjType.YELLOW_DRAGON_SCALE_MAIL
_GRAY_DS     = ObjType.GRAY_DRAGON_SCALES
_GOLD_DS     = ObjType.GOLD_DRAGON_SCALES
_SILVER_DS   = ObjType.SILVER_DRAGON_SCALES
_RED_DS      = ObjType.RED_DRAGON_SCALES
_WHITE_DS    = ObjType.WHITE_DRAGON_SCALES
_ORANGE_DS   = ObjType.ORANGE_DRAGON_SCALES
_BLACK_DS    = ObjType.BLACK_DRAGON_SCALES
_BLUE_DS     = ObjType.BLUE_DRAGON_SCALES
_GREEN_DS    = ObjType.GREEN_DRAGON_SCALES
_YELLOW_DS   = ObjType.YELLOW_DRAGON_SCALES

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
# Effective-stat accessors (vendor attrib.c::acurr)
# ---------------------------------------------------------------------------

# STR19(25) — vendor attrib.h::STR19 macro yields 100+y for 19<=y<=25, so
# STR19(25) = 125.  Returned by acurr(A_STR) whenever GoP is worn (vendor
# attrib.c:1213-1215).
_STR19_25 = 125


def compute_effective_str(state) -> jnp.ndarray:
    """Effective Strength accounting for armor and Gauntlets of Power.

    Mirrors the STR branch of vendor ``attrib.c::acurr`` lines 1207-1220::

        tmp = u.abon.a[A_STR] + u.atemp.a[A_STR] + u.acurr.a[A_STR];
        if (tmp >= STR19(25) || (uarmg && uarmg->otyp == GAUNTLETS_OF_POWER))
            result = STR19(25);   /* 125 */
        else
            result = max(tmp, 3);

    Nethax stores base STR directly in ``state.player_str`` (already on the
    0..125 scale, equivalent to vendor's combined ``abon + atemp + acurr``),
    and any additive armor-sourced delta in
    ``inventory.armor_stat_bonus[_STAT_STR]``.  When GoP is in the GLOVES
    slot the effective value is forced to 125 regardless of the additive
    bonus, matching vendor.

    JIT-pure scalar; safe to call from any traced context.
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    glove_tid = _type_id_at_slot(state, int(ArmorSlot.GLOVES))
    wearing_gop = _is_wearing(glove_tid, _GAUNTLETS_OF_POWER)

    base = state.player_str.astype(jnp.int32)
    bonus = state.inventory.armor_stat_bonus[_STAT_STR].astype(jnp.int32)
    additive = jnp.maximum(base + bonus, jnp.int32(3))
    # vendor acurr also clamps when tmp >= STR19(25); since additive >=125
    # is functionally identical to GoP we fold both paths.
    capped = jnp.where(
        wearing_gop | (additive >= jnp.int32(_STR19_25)),
        jnp.int32(_STR19_25),
        additive,
    )
    return capped


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
    body_tid   = _type_id_at_slot(state, int(ArmorSlot.BODY))

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
        # HELM_OF_CAUTION (helm) + dragon scale mail / scales (body).
        # cite: vendor/nethack/include/objects.h:479-481 (WARNING),
        #       :497-553 DRGN_ARMR (resist + REFLECTING + ANTIMAGIC powers).
        int(Intrinsic.WARNING),
        int(Intrinsic.REFLECTING),
        int(Intrinsic.RESIST_FIRE),
        int(Intrinsic.RESIST_COLD),
        int(Intrinsic.RESIST_SLEEP),
        int(Intrinsic.RESIST_DISINT),
        int(Intrinsic.RESIST_SHOCK),
        int(Intrinsic.RESIST_POISON),
        int(Intrinsic.RESIST_ACID),
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

    # HELM_OF_CAUTION -> WARNING
    # cite: vendor/nethack/include/objects.h:479-481 — oc_oprop = WARNING.
    has_warning = _is_wearing(helm_tid, _HELM_OF_CAUTION)
    intr = intr.at[int(Intrinsic.WARNING)].set(has_warning)

    # ------------------------------------------------------------------
    # 3b. BODY armor effects — dragon scale mail / dragon scales.
    # cite: vendor/nethack/include/objects.h:497-553 (DRGN_ARMR).
    # Pairs (DSM, scales) share intrinsic; GOLD has no intrinsic (light only).
    # ------------------------------------------------------------------
    has_gray   = _is_wearing(body_tid, _GRAY_DSM)   | _is_wearing(body_tid, _GRAY_DS)
    has_silver = _is_wearing(body_tid, _SILVER_DSM) | _is_wearing(body_tid, _SILVER_DS)
    has_red    = _is_wearing(body_tid, _RED_DSM)    | _is_wearing(body_tid, _RED_DS)
    has_white  = _is_wearing(body_tid, _WHITE_DSM)  | _is_wearing(body_tid, _WHITE_DS)
    has_orange = _is_wearing(body_tid, _ORANGE_DSM) | _is_wearing(body_tid, _ORANGE_DS)
    has_black  = _is_wearing(body_tid, _BLACK_DSM)  | _is_wearing(body_tid, _BLACK_DS)
    has_blue   = _is_wearing(body_tid, _BLUE_DSM)   | _is_wearing(body_tid, _BLUE_DS)
    has_green  = _is_wearing(body_tid, _GREEN_DSM)  | _is_wearing(body_tid, _GREEN_DS)
    has_yellow = _is_wearing(body_tid, _YELLOW_DSM) | _is_wearing(body_tid, _YELLOW_DS)
    # GRAY -> ANTIMAGIC.  OR with cloak MR below.
    intr = intr.at[int(Intrinsic.MAGIC_RESIST)].set(
        intr[int(Intrinsic.MAGIC_RESIST)] | has_gray
    )
    intr = intr.at[int(Intrinsic.REFLECTING)].set(has_silver)
    intr = intr.at[int(Intrinsic.RESIST_FIRE)].set(has_red)
    intr = intr.at[int(Intrinsic.RESIST_COLD)].set(has_white)
    intr = intr.at[int(Intrinsic.RESIST_SLEEP)].set(has_orange)
    intr = intr.at[int(Intrinsic.RESIST_DISINT)].set(has_black)
    intr = intr.at[int(Intrinsic.RESIST_SHOCK)].set(has_blue)
    intr = intr.at[int(Intrinsic.RESIST_POISON)].set(has_green)
    intr = intr.at[int(Intrinsic.RESIST_ACID)].set(has_yellow)

    # ------------------------------------------------------------------
    # 4. CLOAK effects
    # cite: do_wear.c Cloak_on (~line 325)
    # ------------------------------------------------------------------

    # CLOAK_OF_MAGIC_RESISTANCE → MAGIC_RESIST
    # cite: do_wear.c line 334 (oc_oprop = ANTIMAGIC for this cloak)
    # OR with any MAGIC_RESIST already granted (e.g. GRAY dragon scale mail).
    has_mr = _is_wearing(cloak_tid, _CLOAK_OF_MAGIC_RES)
    intr = intr.at[int(Intrinsic.MAGIC_RESIST)].set(
        intr[int(Intrinsic.MAGIC_RESIST)] | has_mr
    )

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
    new_state = state.replace(
        status=new_status,
        inventory=new_inv,
        player_align=new_align,
    )
    # Wave 50w: vendor worn.c::setworn extrinsic + blocked bitfield bookkeeping
    # and ::recalc_telepat_range.  Re-derived from all worn slots after every
    # armor / ring / amulet wear or take-off.
    # cite: vendor/nethack/src/worn.c lines 73-145 (setworn), 38-44 (w_blocks),
    #       50-69 (recalc_telepat_range).
    return recalc_worn_props(new_state)


# ---------------------------------------------------------------------------
# Worn-slot property bookkeeping — vendor worn.c::setworn extrinsic +
# blocked bitfield and recalc_telepat_range.
#
# Mirrors three vendor mechanisms in one idempotent JIT-pure pass:
#
#   1. ``u.uprops[p].extrinsic`` per-prop OR of W_xxx masks for worn items
#      whose ``objects[otyp].oc_oprop == p`` (worn.c:96-98, 122-125).
#   2. ``u.uprops[p].blocked``   per-prop OR of W_xxx masks from
#      ``w_blocks(o, m)`` (worn.c:38-44): MUMMY_WRAPPING blocks INVIS in cloak
#      slot, CORNUTHAUM blocks CLAIRVOYANT in helm slot for non-wizard role.
#      (BLINDED via ART_EYES_OF_THE_OVERWORLD eyewear slot is punted —
#      Nethax has no worn-eyewear slot yet.)
#   3. ``u.unblind_telepat_range`` set to ``BOLT_LIM^2 * nobjs`` where
#      ``nobjs`` is the count of worn items granting TELEPAT (worn.c:50-69);
#      sentinel ``-1`` when no source.
#
# Per-armor-type → (W_mask, oc_oprop) mapping is encoded inline below so
# the table stays vendor-cited and matches the explicit per-type branches
# already used in ``apply_armor_effects`` above.
# ---------------------------------------------------------------------------

def _slot_type_id(items_type_id: jnp.ndarray, slot: jnp.ndarray) -> jnp.ndarray:
    """Read items.type_id[slot] safely, returning 0 when slot < 0."""
    safe = jnp.clip(slot.astype(jnp.int32), 0, items_type_id.shape[0] - 1)
    return jnp.where(
        slot.astype(jnp.int32) >= 0,
        items_type_id[safe].astype(jnp.int32),
        jnp.int32(0),
    )


# (type_id, oc_oprop) — armor types currently modelled by Nethax that confer
# a single Intrinsic via vendor's setworn extrinsic mechanism.  Each entry's
# W_mask is supplied by the slot it occupies (see callers below).
#
# cite: vendor/nethack/include/objects.h:447-708 — HELM/CLOAK/GLOVES/BOOTS/
# DRGN_ARMR entries with oc_oprop != 0.
_BODY_PROP_TABLE = [
    (int(_GRAY_DSM),   int(Intrinsic.MAGIC_RESIST)),
    (int(_GRAY_DS),    int(Intrinsic.MAGIC_RESIST)),
    (int(_SILVER_DSM), int(Intrinsic.REFLECTING)),
    (int(_SILVER_DS),  int(Intrinsic.REFLECTING)),
    (int(_RED_DSM),    int(Intrinsic.RESIST_FIRE)),
    (int(_RED_DS),     int(Intrinsic.RESIST_FIRE)),
    (int(_WHITE_DSM),  int(Intrinsic.RESIST_COLD)),
    (int(_WHITE_DS),   int(Intrinsic.RESIST_COLD)),
    (int(_ORANGE_DSM), int(Intrinsic.RESIST_SLEEP)),
    (int(_ORANGE_DS),  int(Intrinsic.RESIST_SLEEP)),
    (int(_BLACK_DSM),  int(Intrinsic.RESIST_DISINT)),
    (int(_BLACK_DS),   int(Intrinsic.RESIST_DISINT)),
    (int(_BLUE_DSM),   int(Intrinsic.RESIST_SHOCK)),
    (int(_BLUE_DS),    int(Intrinsic.RESIST_SHOCK)),
    (int(_GREEN_DSM),  int(Intrinsic.RESIST_POISON)),
    (int(_GREEN_DS),   int(Intrinsic.RESIST_POISON)),
    (int(_YELLOW_DSM), int(Intrinsic.RESIST_ACID)),
    (int(_YELLOW_DS),  int(Intrinsic.RESIST_ACID)),
]
_CLOAK_PROP_TABLE = [
    (int(_CLOAK_OF_MAGIC_RES),     int(Intrinsic.MAGIC_RESIST)),
    (int(_CLOAK_OF_INVISIBILITY),  int(Intrinsic.INVIS)),
    (int(_CLOAK_OF_DISPLACEMENT),  int(Intrinsic.DISPLACED)),
    (int(_ELVEN_CLOAK),            int(Intrinsic.STEALTH)),
]
_HELM_PROP_TABLE = [
    (int(_HELM_OF_TELEPATHY), int(Intrinsic.TELEPATHY)),
    (int(_TINFOIL_HAT),       int(Intrinsic.TELEPATHY)),
    (int(_HELM_OF_CAUTION),   int(Intrinsic.WARNING)),
    (int(_CORNUTHAUM),        int(Intrinsic.CLAIRVOYANT)),
]
_GLOVE_PROP_TABLE: list = []   # no glove type currently sets a flat extrinsic
                               # via the setworn path (GoP / fumbling / dex
                               # bonus are handled inline above).
_BOOT_PROP_TABLE = [
    (int(_LEVITATION_BOOTS),    int(Intrinsic.LEVITATION)),
    (int(_WATER_WALKING_BOOTS), int(Intrinsic.WWALKING)),
    (int(_JUMPING_BOOTS),       int(Intrinsic.JUMPING)),
    (int(_ELVEN_BOOTS),         int(Intrinsic.STEALTH)),
    (int(_SPEED_BOOTS),         int(Intrinsic.FAST)),
]


def _grant_mask(type_id: jnp.ndarray, table, w_mask: int) -> jnp.ndarray:
    """Build the per-prop OR-of-W_masks contributed by a single worn slot.

    Returns int32[N_INTRINSICS]: for every (otyp, prop) in ``table`` whose
    type_id matches the worn item, OR in ``w_mask`` at index ``prop``.
    Empty slot (type_id == 0) yields the zero vector.
    """
    contrib = jnp.zeros((N_INTRINSICS,), dtype=jnp.int32)
    if not table:
        return contrib
    for otyp, prop in table:
        match = type_id == jnp.int32(int(otyp))
        contrib = contrib.at[int(prop)].set(
            contrib[int(prop)] | jnp.where(match, jnp.int32(w_mask), jnp.int32(0))
        )
    return contrib


# Ring effect index → Intrinsic — duplicated locally from
# items_jewelry._RING_TO_INTRINSIC to avoid an import cycle.  Kept in sync
# by inspection: any change to the jewelry table needs a mirror here.
# cite: vendor/nethack/include/objects.h:741-827 (RING() oc_oprop).
_RING_OPROP_TABLE_VALS = [
    (6,  int(Intrinsic.REGEN)),                  # REGENERATION
    (7,  int(Intrinsic.SEARCHING)),
    (8,  int(Intrinsic.STEALTH)),
    (9,  int(Intrinsic.FIXED_ABIL)),             # SUSTAIN_ABILITY
    (10, int(Intrinsic.LEVITATION)),
    (12, int(Intrinsic.AGGRAVATE)),              # AGGRAVATE_MONSTER
    (13, int(Intrinsic.CONFLICT)),
    (14, int(Intrinsic.WARNING)),
    (15, int(Intrinsic.RESIST_POISON)),
    (16, int(Intrinsic.RESIST_FIRE)),
    (17, int(Intrinsic.RESIST_COLD)),
    (18, int(Intrinsic.RESIST_SHOCK)),
    (19, int(Intrinsic.FREE_ACTION)),
    (20, int(Intrinsic.SLOW_DIGESTION)),
    (21, int(Intrinsic.TELEPORT)),
    (22, int(Intrinsic.TELEPORT_CONTROL)),
    (23, int(Intrinsic.POLYMORPH)),
    (24, int(Intrinsic.POLYMORPH_CONTROL)),
    (25, int(Intrinsic.INVIS)),
    (26, int(Intrinsic.SEE_INVIS)),
    (27, int(Intrinsic.PROT_FROM_SHAPE_CHANGERS)),
]

# Amulet effect index → Intrinsic — local mirror of
# items_jewelry._AMULET_TO_INTRINSIC.  cite: vendor/nethack/include/objects.h
# :835-875 (AMULET() oc_oprop) — ESP→TELEPAT, etc.
_AMULET_OPROP_TABLE_VALS = [
    (0,  int(Intrinsic.TELEPATHY)),       # ESP
    (1,  int(Intrinsic.LIFESAVED)),       # LIFE_SAVING
    (4,  int(Intrinsic.RESIST_POISON)),   # VERSUS_POISON
    (6,  int(Intrinsic.UNCHANGING)),
    (7,  int(Intrinsic.REFLECTING)),
    (8,  int(Intrinsic.BREATHLESS)),      # MAGICAL_BREATHING
    (9,  int(Intrinsic.PROTECTION)),      # GUARDING
    (10, int(Intrinsic.FLYING)),
]


def recalc_worn_props(state) -> object:
    """Re-derive ``status.extrinsic`` / ``status.blocked`` / ``unblind_telepat_range``
    from all currently-worn slots.

    Mirrors vendor ``setworn`` (worn.c:73-145) extrinsic + blocked bookkeeping
    and ``recalc_telepat_range`` (worn.c:50-69) in a single idempotent pass.
    Idempotent and JIT-pure; safe to call after any wear / take-off.

    Cite: vendor/nethack/src/worn.c lines 73 (setworn), 38 (w_blocks),
          50 (recalc_telepat_range).
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot

    items_tid = state.inventory.items.type_id

    # ---- Worn type_ids per slot -------------------------------------------
    body_tid  = _slot_type_id(items_tid, state.inventory.worn_armor[int(ArmorSlot.BODY)])
    cloak_tid = _slot_type_id(items_tid, state.inventory.worn_armor[int(ArmorSlot.CLOAK)])
    helm_tid  = _slot_type_id(items_tid, state.inventory.worn_armor[int(ArmorSlot.HELM)])
    glove_tid = _slot_type_id(items_tid, state.inventory.worn_armor[int(ArmorSlot.GLOVES)])
    boot_tid  = _slot_type_id(items_tid, state.inventory.worn_armor[int(ArmorSlot.BOOTS)])

    ring_l_tid = _slot_type_id(items_tid, state.inventory.worn_rings[0])
    ring_r_tid = _slot_type_id(items_tid, state.inventory.worn_rings[1])
    amul_tid   = _slot_type_id(items_tid, state.inventory.worn_amulet)

    # ---- Per-slot extrinsic contribution ----------------------------------
    extr = jnp.zeros((N_INTRINSICS,), dtype=jnp.int32)
    extr = extr | _grant_mask(body_tid,  _BODY_PROP_TABLE,  W_ARM)
    extr = extr | _grant_mask(cloak_tid, _CLOAK_PROP_TABLE, W_ARMC)
    extr = extr | _grant_mask(helm_tid,  _HELM_PROP_TABLE,  W_ARMH)
    extr = extr | _grant_mask(glove_tid, _GLOVE_PROP_TABLE, W_ARMG)
    extr = extr | _grant_mask(boot_tid,  _BOOT_PROP_TABLE,  W_ARMF)
    extr = extr | _grant_mask(ring_l_tid, _RING_OPROP_TABLE_VALS, W_RINGL)
    extr = extr | _grant_mask(ring_r_tid, _RING_OPROP_TABLE_VALS, W_RINGR)
    extr = extr | _grant_mask(amul_tid,   _AMULET_OPROP_TABLE_VALS, W_AMUL)

    # ---- w_blocks (worn.c:38-44) ------------------------------------------
    # MUMMY_WRAPPING in cloak slot blocks INVIS.
    is_mummy = cloak_tid == jnp.int32(int(_MUMMY_WRAPPING))
    blocked = jnp.zeros((N_INTRINSICS,), dtype=jnp.int32)
    blocked = blocked.at[int(Intrinsic.INVIS)].set(
        jnp.where(is_mummy, jnp.int32(W_ARMC), jnp.int32(0))
    )
    # CORNUTHAUM in helm slot blocks CLAIRVOYANT for non-wizards.
    is_cornu     = helm_tid == jnp.int32(int(_CORNUTHAUM))
    is_nonwizard = state.player_role != jnp.int8(_ROLE_WIZARD)
    cornu_blocks = is_cornu & is_nonwizard
    blocked = blocked.at[int(Intrinsic.CLAIRVOYANT)].set(
        jnp.where(cornu_blocks, jnp.int32(W_ARMH), jnp.int32(0))
    )
    # NOTE: ART_EYES_OF_THE_OVERWORLD blocks BLINDED via W_TOOL slot — Nethax
    # has no worn-eyewear (W_TOOL) slot yet, so that branch is punted.
    # cite: vendor/nethack/src/worn.c lines 42-43.

    # ---- recalc_telepat_range (worn.c:50-69) ------------------------------
    # Count worn slots (armor + rings + amulet) whose oc_oprop == TELEPAT.
    def _grants_telep(tid: jnp.ndarray, table) -> jnp.ndarray:
        if not table:
            return jnp.bool_(False)
        m = jnp.bool_(False)
        for otyp, prop in table:
            if int(prop) == int(Intrinsic.TELEPATHY):
                m = m | (tid == jnp.int32(int(otyp)))
        return m

    n_telep = (
        _grants_telep(helm_tid,  _HELM_PROP_TABLE).astype(jnp.int32)
        + _grants_telep(amul_tid, _AMULET_OPROP_TABLE_VALS).astype(jnp.int32)
        + _grants_telep(ring_l_tid, _RING_OPROP_TABLE_VALS).astype(jnp.int32)
        + _grants_telep(ring_r_tid, _RING_OPROP_TABLE_VALS).astype(jnp.int32)
    )
    # Vendor sentinel: -1 when nobjs == 0, else BOLT_LIM² * nobjs.
    # cite: vendor/nethack/src/worn.c lines 65-68.
    telep_range = jnp.where(
        n_telep > jnp.int32(0),
        jnp.int32(_BOLT_LIM * _BOLT_LIM) * n_telep,
        jnp.int32(-1),
    )

    new_status = state.status.replace(extrinsic=extr, blocked=blocked)
    return state.replace(status=new_status, unblind_telepat_range=telep_range)
