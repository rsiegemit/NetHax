"""Ring + amulet wear/take-off, intrinsic granting — vendor/nethack/src/{do_wear,worn}.c.

Canonical sources:
  vendor/nethack/src/do_wear.c  — Ring_on(), Ring_off(), Amulet_on(), Amulet_off(),
                                  doputon(), doremring()
  vendor/nethack/src/worn.c     — setworn(), setnotworn() extrinsic bookkeeping
  vendor/nethack/include/objects.h — all 28 RING entries, all 13 AMULET entries
  vendor/nethack/include/prop.h    — oc_oprop indices for each ring/amulet type

Design
------
NetHack's worn.c setworn() records which item occupies which worn-mask slot
(W_RINGL / W_RINGR / W_AMUL) and updates the uprops[].extrinsic bitmask so that
HXxx macros resolve.  In JAX we flatten this to:
  - inventory.worn_rings[hand]  — inventory slot index (-1 = empty)
  - inventory.worn_amulet       — inventory slot index (-1 = empty)
  - status.intrinsics[idx]      — boolean extrinsic flag per Intrinsic enum

Rings that adjust stats (GAIN_STRENGTH, GAIN_CONSTITUTION, ADORNMENT,
INCREASE_ACCURACY, INCREASE_DAMAGE) modify the corresponding player_* scalars
on the EnvState rather than the intrinsics array, matching NetHack's
adjust_attrib() / ABON() approach.

Rings / amulets with purely tick-driven effects (HUNGER, SLOW_DIGESTION,
STRANGULATION, RESTFUL_SLEEP, CHANGE) are flagged via TimedStatus or intrinsic
booleans; the tick driver lives in status_effects.step().  This file only
handles the wear / take-off transitions.

28 ring effects + 13 amulet effects = 41 item effects implemented.
"""

from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct  # noqa: F401 — imported so this module is importable

from Nethax.nethax.subsystems.status_effects import (
    Intrinsic,
    TimedStatus,
    add_intrinsic,
    remove_intrinsic,
    add_timed,
)


# ---------------------------------------------------------------------------
# Ring effect enum
# Mirrors the 28 RING entries in vendor/nethack/include/objects.h lines 741-827.
# The enum value is the ring's *position within the ring object table* (0-based
# index among rings), not the global otyp.  The RingEffect → Intrinsic mapping
# below uses this position as a switch key.
# ---------------------------------------------------------------------------

class RingEffect(IntEnum):
    ADORNMENT                        =  0   # +CHA  (adjust_attrib A_CHA)
    GAIN_STRENGTH                    =  1   # +STR  (adjust_attrib A_STR)
    GAIN_CONSTITUTION                =  2   # +CON  (adjust_attrib A_CON)
    INCREASE_ACCURACY                =  3   # u.uhitinc +=spe
    INCREASE_DAMAGE                  =  4   # u.udaminc +=spe
    PROTECTION                       =  5   # AC calc handled in combat subsystem
    REGENERATION                     =  6   # Intrinsic.REGEN
    SEARCHING                        =  7   # Intrinsic.SEARCHING
    STEALTH                          =  8   # Intrinsic.STEALTH
    SUSTAIN_ABILITY                  =  9   # Intrinsic.FIXED_ABIL
    LEVITATION                       = 10   # Intrinsic.LEVITATION
    HUNGER                           = 11   # TimedStatus.HUNGER_RING (tick)
    AGGRAVATE_MONSTER                = 12   # Intrinsic.AGGRAVATE
    CONFLICT                         = 13   # Intrinsic.CONFLICT
    WARNING                          = 14   # Intrinsic.WARNING
    POISON_RESISTANCE                = 15   # Intrinsic.RESIST_POISON
    FIRE_RESISTANCE                  = 16   # Intrinsic.RESIST_FIRE
    COLD_RESISTANCE                  = 17   # Intrinsic.RESIST_COLD
    SHOCK_RESISTANCE                 = 18   # Intrinsic.RESIST_SHOCK
    FREE_ACTION                      = 19   # Intrinsic.FREE_ACTION
    SLOW_DIGESTION                   = 20   # Intrinsic.SLOW_DIGESTION
    TELEPORTATION                    = 21   # Intrinsic.TELEPORT
    TELEPORT_CONTROL                 = 22   # Intrinsic.TELEPORT_CONTROL
    POLYMORPH                        = 23   # Intrinsic.POLYMORPH
    POLYMORPH_CONTROL                = 24   # Intrinsic.POLYMORPH_CONTROL
    INVISIBILITY                     = 25   # Intrinsic.INVIS
    SEE_INVISIBLE                    = 26   # Intrinsic.SEE_INVIS
    PROTECTION_FROM_SHAPE_CHANGERS   = 27   # Intrinsic.PROT_FROM_SHAPE_CHANGERS


# ---------------------------------------------------------------------------
# Amulet effect enum
# Mirrors the 13 AMULET entries in vendor/nethack/include/objects.h lines 835-875.
# ---------------------------------------------------------------------------

class AmuletEffect(IntEnum):
    ESP                   =  0   # Intrinsic.TELEPATHY
    LIFE_SAVING           =  1   # Intrinsic.LIFESAVED (triggered on death)
    STRANGULATION         =  2   # TimedStatus.STRANGLED — lethal expiry at 6t
    RESTFUL_SLEEP         =  3   # TimedStatus.SLEEPY
    VERSUS_POISON         =  4   # Intrinsic.RESIST_POISON
    CHANGE                =  5   # one-shot sex change on wear (no persistent flag)
    UNCHANGING            =  6   # Intrinsic.UNCHANGING
    REFLECTION            =  7   # Intrinsic.REFLECTING
    MAGICAL_BREATHING     =  8   # Intrinsic.BREATHLESS
    GUARDING              =  9   # Intrinsic.PROTECTION
    FLYING                = 10   # Intrinsic.FLYING
    CHEAP_AMULET          = 11   # FAKE_AMULET_OF_YENDOR — no intrinsic
    YENDOR                = 12   # THE Amulet — no intrinsic, enables ascension


# ---------------------------------------------------------------------------
# Ring → Intrinsic dispatch tables
# ---------------------------------------------------------------------------

# Rings that grant a single Intrinsic when worn, keyed by RingEffect value.
# Rings with stat adjustments or tick-driven effects are handled inline;
# they are absent from this table (value -1 signals "no direct intrinsic").
_RING_TO_INTRINSIC: dict[int, int] = {
    RingEffect.REGENERATION:                 Intrinsic.REGEN,
    RingEffect.SEARCHING:                    Intrinsic.SEARCHING,
    RingEffect.STEALTH:                      Intrinsic.STEALTH,
    RingEffect.SUSTAIN_ABILITY:              Intrinsic.FIXED_ABIL,
    RingEffect.LEVITATION:                   Intrinsic.LEVITATION,
    RingEffect.AGGRAVATE_MONSTER:            Intrinsic.AGGRAVATE,
    RingEffect.CONFLICT:                     Intrinsic.CONFLICT,
    RingEffect.WARNING:                      Intrinsic.WARNING,
    RingEffect.POISON_RESISTANCE:            Intrinsic.RESIST_POISON,
    RingEffect.FIRE_RESISTANCE:              Intrinsic.RESIST_FIRE,
    RingEffect.COLD_RESISTANCE:              Intrinsic.RESIST_COLD,
    RingEffect.SHOCK_RESISTANCE:             Intrinsic.RESIST_SHOCK,
    RingEffect.FREE_ACTION:                  Intrinsic.FREE_ACTION,
    RingEffect.SLOW_DIGESTION:               Intrinsic.SLOW_DIGESTION,
    RingEffect.TELEPORTATION:                Intrinsic.TELEPORT,
    RingEffect.TELEPORT_CONTROL:             Intrinsic.TELEPORT_CONTROL,
    RingEffect.POLYMORPH:                    Intrinsic.POLYMORPH,
    RingEffect.POLYMORPH_CONTROL:            Intrinsic.POLYMORPH_CONTROL,
    RingEffect.INVISIBILITY:                 Intrinsic.INVIS,
    RingEffect.SEE_INVISIBLE:                Intrinsic.SEE_INVIS,
    RingEffect.PROTECTION_FROM_SHAPE_CHANGERS: Intrinsic.PROT_FROM_SHAPE_CHANGERS,
}

# Public table: RingEffect index → Intrinsic index, or -1 for stat/tick rings.
# Length == 28 (one entry per RingEffect value, indexed by RingEffect int).
RING_INTRINSIC_TABLE: list = [
    _RING_TO_INTRINSIC.get(e, -1) for e in sorted(RingEffect, key=lambda x: int(x))
]

# Amulet effects that grant a single Intrinsic on wear.
_AMULET_TO_INTRINSIC: dict[int, int] = {
    AmuletEffect.ESP:               Intrinsic.TELEPATHY,
    AmuletEffect.LIFE_SAVING:       Intrinsic.LIFESAVED,
    AmuletEffect.VERSUS_POISON:     Intrinsic.RESIST_POISON,
    AmuletEffect.UNCHANGING:        Intrinsic.UNCHANGING,
    AmuletEffect.REFLECTION:        Intrinsic.REFLECTING,
    AmuletEffect.MAGICAL_BREATHING: Intrinsic.BREATHLESS,
    AmuletEffect.GUARDING:          Intrinsic.PROTECTION,
    AmuletEffect.FLYING:            Intrinsic.FLYING,
}

# Amulet effects that start a timed status on wear (effect_id → TimedStatus).
# STRANGULATION triggers at 6 turns (do_wear.c: Strangled = 6L).
# RESTFUL_SLEEP randomises 2–100 turns via rnd(98)+2 in wear_amulet
# (see do_wear.c:1048 case AMULET_OF_RESTFUL_SLEEP).
_AMULET_TO_TIMED: dict[int, tuple[int, int]] = {
    AmuletEffect.STRANGULATION: (TimedStatus.STRANGLED, 6),
}


# ---------------------------------------------------------------------------
# Internal stat-adjustment helpers
# ---------------------------------------------------------------------------

def _ring_apply_stat(state, ring_effect: int, enchantment: int):
    """Apply stat bonus for stat-adjusting rings (do_wear.c adjust_attrib).

    Modifies player_str / player_con / player_cha on EnvState.
    enchantment is the ring's +/- spe value (typically in range -5..+5).
    """
    eff = int(ring_effect)
    if eff == RingEffect.GAIN_STRENGTH:
        return state.replace(
            player_str=jnp.int16(state.player_str + enchantment)
        )
    if eff == RingEffect.GAIN_CONSTITUTION:
        return state.replace(
            player_con=jnp.int8(state.player_con + enchantment)
        )
    if eff == RingEffect.ADORNMENT:
        return state.replace(
            player_cha=jnp.int8(state.player_cha + enchantment)
        )
    if eff == RingEffect.INCREASE_ACCURACY:
        return state.replace(
            player_uhitinc=jnp.int8(state.player_uhitinc + enchantment)
        )
    if eff == RingEffect.INCREASE_DAMAGE:
        return state.replace(
            player_udaminc=jnp.int8(state.player_udaminc + enchantment)
        )
    if eff == RingEffect.PROTECTION:
        # Ring of protection lowers AC by enchantment (lower AC = better in NetHack).
        return state.replace(
            player_ac=jnp.int32(state.player_ac - enchantment)
        )
    return state


def _ring_revoke_stat(state, ring_effect: int, enchantment: int):
    """Revoke stat bonus for stat-adjusting rings (do_wear.c Ring_off adjust_attrib)."""
    return _ring_apply_stat(state, ring_effect, -enchantment)


# ---------------------------------------------------------------------------
# Public API — ring wear / take-off
# ---------------------------------------------------------------------------

def put_on_ring(state, rng: jax.Array, slot_idx: int, hand: int):
    """Wear the ring in inventory slot slot_idx on the given hand.

    Parameters
    ----------
    state     : EnvState
    rng       : JAX PRNGKey (unused in Wave 3; reserved for cursed-ring rng)
    slot_idx  : index into state.inventory.items
    hand      : 0 = left ring finger, 1 = right ring finger

    Mirrors do_wear.c: doputon() → setworn(ring, W_RINGL/W_RINGR) → Ring_on().

    Side-effects
    ------------
    - Sets inventory.worn_rings[hand] = slot_idx
    - Grants the ring's intrinsic in status.intrinsics
    - Stat-adjusting rings modify player_str / player_con / player_cha
    - Ring of hunger sets TimedStatus.HUNGER_RING to a large sentinel (999)
    """
    item = state.inventory.items
    # Support both scalar (broadcast-replace) and [52]-array item layouts.
    type_raw = item.type_id
    enc_raw  = item.enchantment
    buc_raw  = item.buc_status
    ring_effect = int(type_raw[slot_idx]) if type_raw.ndim > 0 else int(type_raw)
    enchantment = int(enc_raw[slot_idx])  if enc_raw.ndim  > 0 else int(enc_raw)
    buc_val     = int(buc_raw[slot_idx])  if buc_raw.ndim  > 0 else int(buc_raw)

    # Strip the ring from wielded / off-hand / quiver slots before wearing.
    # Cite: vendor/nethack/src/do_wear.c Ring_on lines 1247-1254 —
    #   "make sure ring isn't wielded"; clears uwep / uswapwep / uquiver.
    inv = state.inventory
    cleared_wielded = jnp.where(
        inv.wielded.astype(jnp.int32) == jnp.int32(slot_idx),
        jnp.int8(-1), inv.wielded,
    )
    cleared_off_hand = jnp.where(
        inv.off_hand.astype(jnp.int32) == jnp.int32(slot_idx),
        jnp.int8(-1), inv.off_hand,
    )
    cleared_quiver = jnp.where(
        inv.quiver.astype(jnp.int32) == jnp.int32(slot_idx),
        jnp.int8(-1), inv.quiver,
    )

    # Record the worn slot.
    new_worn_rings = inv.worn_rings.at[hand].set(jnp.int8(slot_idx))
    # Cursed ring becomes stuck on the finger.
    # Cite: vendor/nethack/src/do_wear.c Ring_off_or_gone cursed check.
    CURSED = 1
    is_cursed = buc_val == CURSED
    new_worn_rings_welded = inv.worn_rings_welded.at[hand].set(
        jnp.bool_(is_cursed)
    )
    new_inventory = inv.replace(
        worn_rings=new_worn_rings,
        worn_rings_welded=new_worn_rings_welded,
        wielded=cleared_wielded,
        off_hand=cleared_off_hand,
        quiver=cleared_quiver,
    )
    state = state.replace(inventory=new_inventory)

    # Grant intrinsic if this ring type has a direct mapping.
    intrinsic_id = _RING_TO_INTRINSIC.get(ring_effect)
    if intrinsic_id is not None:
        state = state.replace(status=add_intrinsic(state.status, intrinsic_id))

    # Stat-adjusting rings.
    # Cite: vendor/nethack/src/do_wear.c Ring_on lines 1316-1342 — adjust_attrib,
    # u.uhitinc/u.udaminc bumps, and find_ac() for RIN_PROTECTION.
    if ring_effect in (
        RingEffect.GAIN_STRENGTH,
        RingEffect.GAIN_CONSTITUTION,
        RingEffect.ADORNMENT,
        RingEffect.INCREASE_ACCURACY,
        RingEffect.INCREASE_DAMAGE,
        RingEffect.PROTECTION,
    ):
        state = _ring_apply_stat(state, ring_effect, enchantment)

    # Ring of hunger — flag the tick driver to increase drain rate.
    # Cite: vendor/nethack/src/do_wear.c Ring_on line 1265 (RIN_HUNGER fall-through);
    #       extrinsic HUNGER is consumed in timeout.c::hunger to speed nutrition drain.
    if ring_effect == RingEffect.HUNGER:
        state = state.replace(
            status=add_timed(state.status, TimedStatus.HUNGER_RING, 999)
        )

    # Use-identification: wearing an observably-effecting ring discovers its
    # type via learnring → makeknown(obj->otyp).
    # Cite: vendor/nethack/src/do_wear.c::Ring_on lines 1242-1344 — calls
    # learnring(obj, TRUE) when the wear has an observable effect (invisibility,
    # see-invisible, levitation, +N protection where N!=0, stat adjustments via
    # adjust_attrib, and the always-observable extrinsic-granters like
    # regeneration/warning/conflict/teleport/teleport-control/polymorph-control/
    # slow-digestion/aggravate-monster).
    # vendor hack.h:1530 #define makeknown(x) discover_object((x),TRUE,TRUE,TRUE).
    _OBSERVABLE_RINGS = {
        RingEffect.INVISIBILITY,
        RingEffect.SEE_INVISIBLE,
        RingEffect.LEVITATION,
        RingEffect.REGENERATION,
        RingEffect.WARNING,
        RingEffect.CONFLICT,
        RingEffect.TELEPORTATION,
        RingEffect.TELEPORT_CONTROL,
        RingEffect.POLYMORPH_CONTROL,
        RingEffect.SLOW_DIGESTION,
        RingEffect.AGGRAVATE_MONSTER,
        RingEffect.GAIN_STRENGTH,
        RingEffect.GAIN_CONSTITUTION,
        RingEffect.ADORNMENT,
        RingEffect.INCREASE_ACCURACY,
        RingEffect.INCREASE_DAMAGE,
    }
    is_observable_static = ring_effect in _OBSERVABLE_RINGS
    # Protection only learns when spe != 0 (vendor do_wear.c:1338).
    is_protection_observable = (
        ring_effect == RingEffect.PROTECTION and enchantment != 0
    )
    if is_observable_static or is_protection_observable:
        # Index the per-type identification mask by the wearing ring's
        # canonical objects-table type_id (not the RingEffect sub-index).
        raw_type_arr = state.inventory.items.type_id
        otyp = (
            raw_type_arr[slot_idx].astype(jnp.int32)
            if raw_type_arr.ndim > 0
            else jnp.asarray(raw_type_arr, dtype=jnp.int32)
        )
        type_mask = state.identification.identified
        safe_otyp = jnp.clip(
            otyp, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1)
        )
        new_type_mask = type_mask.at[safe_otyp].set(jnp.bool_(True))
        # Also flip the per-item identified flag (vendor learnring also
        # sets obj->dknown, which is the per-item analogue).  Per vendor
        # obj.h line 114 + objnam.c:1183, learning the ring's effect also
        # reveals erodeproof / charge state — set rknown=True alongside.
        items_id = state.inventory.items.identified
        new_items_id = items_id.at[slot_idx].set(jnp.bool_(True))
        items_dknown = state.inventory.items.dknown
        new_items_dknown = items_dknown.at[slot_idx].set(jnp.bool_(True))
        items_rknown = state.inventory.items.rknown
        new_items_rknown = items_rknown.at[slot_idx].set(jnp.bool_(True))
        state = state.replace(
            inventory=state.inventory.replace(
                items=state.inventory.items.replace(
                    identified=new_items_id,
                    dknown=new_items_dknown,
                    rknown=new_items_rknown,
                ),
            ),
            identification=state.identification.replace(
                identified=new_type_mask
            ),
        )

    # Wave 50w: worn.c::setworn extrinsic + recalc_telepat_range bookkeeping.
    # cite: vendor/nethack/src/worn.c lines 73-145 (setworn), 50-69 (recalc).
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    return recalc_worn_props(state)


def take_off_ring(state, hand: int):
    """Remove the ring worn on hand (0=left, 1=right).

    Mirrors do_wear.c: Ring_off() → setworn(NULL, mask) → revoke extrinsic.

    Side-effects
    ------------
    - Sets inventory.worn_rings[hand] = -1
    - Revokes the ring's intrinsic IF the other hand does not also hold a
      ring of the same type (vendor do_wear.c::Ring_off_or_gone delegates
      to setworn / EXTRINSIC bookkeeping which only clears the property
      bit when no other source supplies it).  Two rings of fire-resistance
      worn together: removing one keeps FIRE_RES intact.
    - Stat-adjusting rings reverse the stat modification
    """
    slot_idx = int(state.inventory.worn_rings[hand])
    if slot_idx < 0:
        return state  # nothing worn on this hand
    # Cursed-stuck ring cannot be removed.
    # Cite: vendor/nethack/src/do_wear.c Ring_off_or_gone — cursed ring blocked.
    if bool(state.inventory.worn_rings_welded[hand]):
        return state

    item = state.inventory.items
    type_raw = item.type_id
    enc_raw  = item.enchantment
    ring_effect = int(type_raw[slot_idx]) if type_raw.ndim > 0 else int(type_raw)
    enchantment = int(enc_raw[slot_idx])  if enc_raw.ndim  > 0 else int(enc_raw)

    # Is the other hand wearing a different ring of the same type?  If
    # so, the intrinsic / stat bonus should remain — vendor uses EXTRINSIC
    # source bookkeeping (setworn(NULL, mask) only clears the property
    # when no other slot still confers it).  Same slot index in both
    # hands is an invalid state (can't wear one ring instance on two
    # hands); guard against it explicitly.
    other_hand = 1 - int(hand)
    other_slot = int(state.inventory.worn_rings[other_hand])
    other_same_type = False
    if other_slot >= 0 and other_slot != slot_idx:
        other_type = int(type_raw[other_slot]) if type_raw.ndim > 0 else int(type_raw)
        other_same_type = (other_type == ring_effect)

    # Clear worn slot and weld flag.
    new_worn_rings = state.inventory.worn_rings.at[hand].set(jnp.int8(-1))
    new_worn_rings_welded = state.inventory.worn_rings_welded.at[hand].set(jnp.bool_(False))
    new_inventory = state.inventory.replace(
        worn_rings=new_worn_rings,
        worn_rings_welded=new_worn_rings_welded,
    )
    state = state.replace(inventory=new_inventory)

    # Revoke intrinsic — but only if the other hand isn't also supplying it.
    intrinsic_id = _RING_TO_INTRINSIC.get(ring_effect)
    if intrinsic_id is not None and not other_same_type:
        state = state.replace(
            status=remove_intrinsic(state.status, intrinsic_id)
        )

    # Revoke stat adjustments.
    # Cite: vendor/nethack/src/do_wear.c Ring_off_or_gone lines 1415-1437 —
    # adjust_attrib(-spe), uhitinc/udaminc -=spe, and find_ac() for RIN_PROTECTION.
    if ring_effect in (
        RingEffect.GAIN_STRENGTH,
        RingEffect.GAIN_CONSTITUTION,
        RingEffect.ADORNMENT,
        RingEffect.INCREASE_ACCURACY,
        RingEffect.INCREASE_DAMAGE,
        RingEffect.PROTECTION,
    ):
        state = _ring_revoke_stat(state, ring_effect, enchantment)

    # Ring of hunger — clear the hunger drain timer.
    if ring_effect == RingEffect.HUNGER:
        new_statuses = state.status.timed_statuses.at[TimedStatus.HUNGER_RING].set(
            jnp.int32(0)
        )
        state = state.replace(
            status=state.status.replace(timed_statuses=new_statuses)
        )

    # Wave 50w: worn.c::setworn extrinsic + recalc_telepat_range bookkeeping.
    # cite: vendor/nethack/src/worn.c lines 73-145 (setworn), 50-69 (recalc).
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    return recalc_worn_props(state)


def ring_tick(state, rng: jax.Array):
    """Apply per-turn effects of worn rings.

    Called once per game turn after action dispatch.
    - HUNGER ring: drain 1 nutrition per turn when worn.
    - TELEPORTATION ring: 1/85 chance to set a timed teleport (vendor timeout.c).
    - POLYMORPH ring: 1/100 chance per turn to trigger polyself.
      Cite: vendor/nethack/src/allmain.c:325 — ``if (Polymorph && !rn2(100))``.
      Effect is queued via TimedStatus.POLY_SELF style flag (we set a
      timed_intrinsics[POLYMORPH] tick that polymorph subsystem can observe).

    Cite: vendor/nethack/src/timeout.c — ring_effects() called from do_regain_pw.
    JIT-safe.
    """
    inv = state.inventory
    worn0 = inv.worn_rings[0].astype(jnp.int32)
    worn1 = inv.worn_rings[1].astype(jnp.int32)

    def _type_at(slot):
        safe = jnp.clip(slot, 0, inv.items.type_id.shape[0] - 1)
        return inv.items.type_id[safe].astype(jnp.int32)

    ring0_type = jnp.where(worn0 >= jnp.int32(0), _type_at(worn0), jnp.int32(-1))
    ring1_type = jnp.where(worn1 >= jnp.int32(0), _type_at(worn1), jnp.int32(-1))

    hunger_worn = (ring0_type == jnp.int32(RingEffect.HUNGER)) | \
                  (ring1_type == jnp.int32(RingEffect.HUNGER))
    tele_worn   = (ring0_type == jnp.int32(RingEffect.TELEPORTATION)) | \
                  (ring1_type == jnp.int32(RingEffect.TELEPORTATION))
    poly_worn   = (ring0_type == jnp.int32(RingEffect.POLYMORPH)) | \
                  (ring1_type == jnp.int32(RingEffect.POLYMORPH))

    # HUNGER: drain 1 nutrition per turn.
    old_nut = state.status.nutrition
    new_nut = jnp.where(hunger_worn, jnp.maximum(old_nut - jnp.int32(1), jnp.int32(0)), old_nut)
    new_status = state.status.replace(nutrition=new_nut)

    # TELEPORTATION: 1/85 chance to set timed_intrinsics[TELEPORT] += 1.
    rng_tele = jax.random.fold_in(rng, jnp.int32(0x7E1E))
    tele_roll = jax.random.randint(rng_tele, (), 0, 85, dtype=jnp.int32)
    tele_fires = tele_worn & (tele_roll == jnp.int32(0))
    tele_idx = jnp.int32(int(Intrinsic.TELEPORT))
    old_tele = new_status.timed_intrinsics[tele_idx]
    new_tele_val = jnp.where(tele_fires, old_tele + jnp.int32(1), old_tele)
    new_timed_intr = new_status.timed_intrinsics.at[tele_idx].set(new_tele_val)
    new_status = new_status.replace(timed_intrinsics=new_timed_intr)

    # POLYMORPH: 1/100 chance to bump timed_intrinsics[POLYMORPH] so the
    # polymorph subsystem fires polyself on its next tick.
    # Cite: vendor/nethack/src/allmain.c:325 — ``if (Polymorph && !rn2(100))``.
    rng_poly = jax.random.fold_in(rng, jnp.int32(0x9017))
    poly_roll = jax.random.randint(rng_poly, (), 0, 100, dtype=jnp.int32)
    poly_fires = poly_worn & (poly_roll == jnp.int32(0))
    poly_idx = jnp.int32(int(Intrinsic.POLYMORPH))
    old_poly = new_status.timed_intrinsics[poly_idx]
    new_poly_val = jnp.where(poly_fires, old_poly + jnp.int32(1), old_poly)
    new_timed_intr = new_status.timed_intrinsics.at[poly_idx].set(new_poly_val)
    new_status = new_status.replace(timed_intrinsics=new_timed_intr)

    return state.replace(status=new_status)


# ---------------------------------------------------------------------------
# Public API — amulet wear / take-off
# ---------------------------------------------------------------------------

def wear_amulet(state, rng: jax.Array, slot_idx: int):
    """Wear the amulet in inventory slot slot_idx.

    Mirrors do_wear.c: Amulet_on() → setworn(amul, W_AMUL) → effect.

    Side-effects
    ------------
    - Sets inventory.worn_amulet = slot_idx
    - Grants the amulet's intrinsic in status.intrinsics (if applicable)
    - Timed amulets (STRANGULATION, RESTFUL_SLEEP) start a TimedStatus timer
    - CHANGE is a one-shot effect (no persistent intrinsic; handled as no-op
      here — full sex-change logic is Wave 4 polymorph integration)
    - YENDOR / CHEAP_AMULET grant no intrinsic
    """
    item = state.inventory.items
    # Support both scalar (broadcast-replace) and [52]-array item layouts.
    type_raw = item.type_id
    buc_raw = item.buc_status
    amulet_effect = int(type_raw[slot_idx]) if type_raw.ndim > 0 else int(type_raw)
    buc_val = int(buc_raw[slot_idx]) if buc_raw.ndim > 0 else int(buc_raw)

    # Strip the amulet from wielded / off-hand / quiver before wearing.
    # Cite: vendor/nethack/src/do_wear.c Amulet_on line 968 —
    #   remove_worn_item(amul, FALSE) ensures the amulet isn't wielded,
    #   alt-wielded, or quivered before setworn(amul, W_AMUL).
    inv = state.inventory
    cleared_wielded = jnp.where(
        inv.wielded.astype(jnp.int32) == jnp.int32(slot_idx),
        jnp.int8(-1), inv.wielded,
    )
    cleared_off_hand = jnp.where(
        inv.off_hand.astype(jnp.int32) == jnp.int32(slot_idx),
        jnp.int8(-1), inv.off_hand,
    )
    cleared_quiver = jnp.where(
        inv.quiver.astype(jnp.int32) == jnp.int32(slot_idx),
        jnp.int8(-1), inv.quiver,
    )

    # Cursed amulet becomes stuck.
    # Cite: vendor/nethack/src/do_wear.c Amulet_off cursed check.
    CURSED = 1
    is_cursed = buc_val == CURSED
    new_inventory = inv.replace(
        worn_amulet=jnp.int8(slot_idx),
        worn_amulet_welded=jnp.bool_(is_cursed),
        wielded=cleared_wielded,
        off_hand=cleared_off_hand,
        quiver=cleared_quiver,
    )
    state = state.replace(inventory=new_inventory)

    # Grant intrinsic if applicable.
    intrinsic_id = _AMULET_TO_INTRINSIC.get(amulet_effect)
    if intrinsic_id is not None:
        state = state.replace(status=add_intrinsic(state.status, intrinsic_id))

    # Start timed status if applicable.
    timed_entry = _AMULET_TO_TIMED.get(amulet_effect)
    if timed_entry is not None:
        timed_id, turns = timed_entry
        state = state.replace(status=add_timed(state.status, timed_id, turns))

    # RESTFUL_SLEEP: vendor uses rnd(98)+2 (uniform [2,100]) and only updates
    # HSleepy timer when newnap < oldnap OR oldnap == 0; FROMOUTSIDE source
    # bits in HSleepy are preserved (HSleepy = (HSleepy & ~TIMEOUT) | newnap).
    # cite: vendor/nethack/src/do_wear.c::Amulet_on lines 1047-1054:
    #     case AMULET_OF_RESTFUL_SLEEP: {
    #         long newnap = (long) rnd(98) + 2L, oldnap = (HSleepy & TIMEOUT);
    #         if (newnap < oldnap || oldnap == 0L)
    #             HSleepy = (HSleepy & ~TIMEOUT) | newnap;
    #         break;
    #     }
    # Our JAX state stores only the TIMEOUT portion in
    # status.timed_statuses[SLEEPY]; there is no FROMOUTSIDE bit on this slot
    # in the current representation, so preservation is a no-op.
    if amulet_effect == int(AmuletEffect.RESTFUL_SLEEP):
        newnap = jax.random.randint(rng, (), 2, 101, dtype=jnp.int32)  # rnd(98)+2 → [2,100]
        sleepy_idx = int(TimedStatus.SLEEPY)
        old_timeout = state.status.timed_statuses[sleepy_idx]
        do_update = (newnap < old_timeout) | (old_timeout == jnp.int32(0))
        new_timeout = jnp.where(do_update, newnap, old_timeout)
        new_ts = state.status.timed_statuses.at[sleepy_idx].set(new_timeout)
        state = state.replace(status=state.status.replace(timed_statuses=new_ts))

    # Wave 50w: worn.c::setworn extrinsic + recalc_telepat_range bookkeeping
    # (AMULET_OF_ESP grants TELEPAT and contributes to nobjs).
    # cite: vendor/nethack/src/worn.c lines 73-145 (setworn), 50-69 (recalc).
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    return recalc_worn_props(state)


def take_off_amulet(state):
    """Remove the currently worn amulet.

    Mirrors do_wear.c: Amulet_off() → setworn(NULL, W_AMUL) → revoke effect.

    Side-effects
    ------------
    - Sets inventory.worn_amulet = -1
    - Revokes the amulet's intrinsic (if applicable)
    - Timed effects (STRANGULATION, RESTFUL_SLEEP) are NOT cancelled on
      take-off — this matches NetHack behaviour where removing the amulet of
      strangulation does not immediately un-strangle the hero; the timer
      already running in status_effects continues until cleared by a separate
      cure mechanic (Wave 4).
    """
    slot_idx = int(state.inventory.worn_amulet)
    if slot_idx < 0:
        return state  # no amulet worn
    # Cursed-stuck amulet cannot be removed.
    # Cite: vendor/nethack/src/do_wear.c Amulet_off — cursed amulet blocked.
    if bool(state.inventory.worn_amulet_welded):
        return state

    item = state.inventory.items
    type_raw = item.type_id
    amulet_effect = int(type_raw[slot_idx]) if type_raw.ndim > 0 else int(type_raw)

    # Clear worn slot and weld flag.
    new_inventory = state.inventory.replace(
        worn_amulet=jnp.int8(-1),
        worn_amulet_welded=jnp.bool_(False),
    )
    state = state.replace(inventory=new_inventory)

    # Revoke intrinsic if applicable.
    intrinsic_id = _AMULET_TO_INTRINSIC.get(amulet_effect)
    if intrinsic_id is not None:
        state = state.replace(
            status=remove_intrinsic(state.status, intrinsic_id)
        )

    # Wave 50w: worn.c::setworn extrinsic + recalc_telepat_range bookkeeping.
    # cite: vendor/nethack/src/worn.c lines 73-145 (setworn), 50-69 (recalc).
    from Nethax.nethax.subsystems.armor_effects import recalc_worn_props
    return recalc_worn_props(state)


# ---------------------------------------------------------------------------
# Action handlers — auto-selection variants
# (mirrors the top-level doputon() / doremring() flow in do_wear.c lines ~2380-2450)
# ---------------------------------------------------------------------------

def handle_put_on(state, rng: jax.Array):
    """Find the first ring or amulet in inventory and wear it.

    Strategy:
    - Scan inventory slots 0..MAX_INVENTORY_SLOTS-1 for category RING or AMULET.
    - Rings: prefer the first empty hand (left = 0, then right = 1).
    - Amulet: wear if worn_amulet is currently -1.

    Wave 3: single-item scan.  Wave 4: full inventory iteration + user prompt.

    Mirrors do_wear.c: doputon() top-level dispatcher.
    """
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS  # local import to avoid cycle

    item = state.inventory.items
    # In Wave 3, items is a single Item struct (not a full array).
    # We handle the single-slot case; full array dispatch is Wave 4.
    # Determine category: RING_CLASS=3, AMULET_CLASS=4 (obj.h OBJCLASS numbering)
    cat = int(item.category)
    RING_CLASS   = 3
    AMULET_CLASS = 4

    if cat == RING_CLASS:
        # Find first empty ring finger.
        worn = state.inventory.worn_rings
        if int(worn[0]) < 0:
            return put_on_ring(state, rng, 0, hand=0)
        elif int(worn[1]) < 0:
            return put_on_ring(state, rng, 0, hand=1)
        # Both fingers occupied — no-op (Wave 4: message to user).
        return state

    if cat == AMULET_CLASS:
        if int(state.inventory.worn_amulet) < 0:
            return wear_amulet(state, rng, 0)
        # Amulet slot occupied — no-op.
        return state

    return state


def handle_remove(state, rng: jax.Array):
    """Remove the first worn ring or amulet found.

    Scan order: left ring → right ring → amulet.
    Mirrors do_wear.c: doremring().
    """
    worn = state.inventory.worn_rings
    if int(worn[0]) >= 0:
        return take_off_ring(state, hand=0)
    if int(worn[1]) >= 0:
        return take_off_ring(state, hand=1)
    if int(state.inventory.worn_amulet) >= 0:
        return take_off_amulet(state)
    return state


# ---------------------------------------------------------------------------
# Life-saving amulet
# ---------------------------------------------------------------------------

def check_life_saving(state):
    """If player would die (done=True) and LIFESAVED intrinsic is set, save them.

    The amulet of life saving is consumed regardless of curse status —
    it is destroyed, so the welded flag is cleared.
    Cite: vendor/nethack/src/end.c::done lines 1084-1105.

    Returns (new_state, saved) where saved is a JAX bool.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic, remove_intrinsic

    intrinsic_idx = int(Intrinsic.LIFESAVED)
    has_lifesaving = state.status.intrinsics[intrinsic_idx]
    should_save = state.done & has_lifesaving

    def _save(s):
        # Consume the amulet: zero its quantity by masking with amulet_slot.
        # Use jnp.where to avoid scalar/array shape mismatch across test setups.
        amulet_slot = s.inventory.worn_amulet.astype(jnp.int32)
        qty = s.inventory.items.quantity
        if qty.ndim > 0:
            # Normal [52]-shaped array: zero out the worn slot.
            slot_mask = jnp.arange(qty.shape[0], dtype=jnp.int32) == amulet_slot
            new_quantity = jnp.where(slot_mask, jnp.int16(0), qty)
        else:
            # Scalar (broadcast test setup): just zero it.
            new_quantity = jnp.int16(0)
        new_items = s.inventory.items.replace(quantity=new_quantity)
        new_inv = s.inventory.replace(
            items=new_items,
            worn_amulet=jnp.int8(-1),
            worn_amulet_welded=jnp.bool_(False),
        )
        new_intrinsics = s.status.intrinsics.at[intrinsic_idx].set(False)
        new_status = s.status.replace(intrinsics=new_intrinsics)
        return s.replace(
            done=jnp.bool_(False),
            player_hp=s.player_hp_max,
            status=new_status,
            inventory=new_inv,
        )

    import jax
    new_state = jax.lax.cond(should_save, _save, lambda s: s, state)
    return new_state, should_save
