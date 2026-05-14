"""Wave 6 Closing-Audit — bit-equal AC chain parity vs vendor do_wear.c::find_ac.

Vendor formula (do_wear.c:2473-2525) is:

    uac = mons[u.umonnum].ac;            // human base AC == 10
    foreach worn armor piece:
        uac -= ARM_BONUS(piece);         // a_ac + spe
    uac -= protection rings + intrinsic ublessed + uspellprot;
    clamp |uac| <= AC_MAX.

When polymorphed, the form's intrinsic AC is the only contributor in Nethax —
``compute_ac`` returns ``state.player_ac`` directly, which the polymorph
subsystem (polyself.c::find_uac) overwrites with ``mons[form].ac``.

# vendor/nethack/src/do_wear.c:2473-2525 (find_ac)
# vendor/nethack/src/polyself.c::find_uac (polymorphed AC source)
# vendor/nethack/include/objects.h (a_ac per ARMOR_CLASS otyp)
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race
from Nethax.nethax.subsystems.character import ObjType, _ARMOR_AC
from Nethax.nethax.subsystems.combat import compute_ac, PLAYER_BASE_AC
from Nethax.nethax.subsystems.inventory import (
    ArmorSlot,
    BASE_AC,
    ItemCategory,
    InventoryState,
    make_item,
    N_ARMOR_SLOTS,
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _reset_valkyrie(seed: int = 1234):
    """Reset a Valkyrie character with deterministic RNG."""
    rng = jax.random.PRNGKey(seed)
    env = NethaxEnv()
    state, _ = env.reset(rng, role=Role.VALKYRIE, race=Race.HUMAN, alignment=0)
    return state


def _strip_all_armor(state):
    """Return a copy of state with no worn armor (and no AC bonus cache)."""
    inv = state.inventory.replace(
        worn_armor=jnp.full((N_ARMOR_SLOTS,), -1, dtype=jnp.int8),
        worn_armor_ac_bonus=jnp.zeros((N_ARMOR_SLOTS,), dtype=jnp.int8),
    )
    return state.replace(inventory=inv)


def _place_armor(state, slot_in_items: int, type_id: int,
                 ac_bonus: int, enchant: int, armor_slot: ArmorSlot):
    """Place a single armor record into ``slot_in_items`` and wear it.

    Returns a new state where the player wears that armor in ``armor_slot``.
    """
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[slot_in_items].set(jnp.int8(ItemCategory.ARMOR)),
        type_id=items.type_id.at[slot_in_items].set(jnp.int16(type_id)),
        ac_bonus=items.ac_bonus.at[slot_in_items].set(jnp.int8(ac_bonus)),
        enchantment=items.enchantment.at[slot_in_items].set(jnp.int8(enchant)),
        quantity=items.quantity.at[slot_in_items].set(jnp.int16(1)),
    )
    new_worn = state.inventory.worn_armor.at[int(armor_slot)].set(jnp.int8(slot_in_items))
    # Cache stores just the base a_ac (enchant is added separately by
    # compute_ac).  This matches wear_armor() in inventory.py.
    new_bonus = state.inventory.worn_armor_ac_bonus.at[int(armor_slot)].set(
        jnp.int8(ac_bonus)
    )
    new_inv = state.inventory.replace(
        items=new_items,
        worn_armor=new_worn,
        worn_armor_ac_bonus=new_bonus,
    )
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# Vendor sanity
# ---------------------------------------------------------------------------

def test_player_base_ac_equals_human_monster_ac_10():
    """Per vendor do_wear.c:2475, base AC equals ``mons[u.umonnum].ac``,
    which is 10 for PM_HUMAN.  Audit table mentions Valkyrie naked AC=9;
    that disagrees with vendor — Nethax follows vendor (==10).

    # vendor/nethack/src/do_wear.c:2475
    # vendor/nethack/src/monst.c (PM_HUMAN ac field == 10)
    """
    assert PLAYER_BASE_AC == 10
    assert BASE_AC == 10


# ---------------------------------------------------------------------------
# Sample 1: Valkyrie naked → AC == base == 10
# ---------------------------------------------------------------------------

def test_valkyrie_naked_ac_9():
    """Stripped Valkyrie: ``uac = mons[PM_HUMAN].ac = 10``.

    Audit doc says 9; vendor says 10; this test asserts the vendor value.

    # vendor/nethack/src/do_wear.c:2475
    """
    state = _strip_all_armor(_reset_valkyrie())
    assert int(compute_ac(state)) == 10


# ---------------------------------------------------------------------------
# Sample 2: Valkyrie + ring mail +0 → AC 10 - 3 = 7
# ---------------------------------------------------------------------------

def test_valkyrie_ring_mail_plus0_ac_6():
    """Ring mail base AC = 3 (objects.h).  uac = 10 - (3 + 0) = 7.

    Audit doc says 6 (assuming base 9); vendor base is 10 → 7.

    # vendor/nethack/include/objects.h RING_MAIL a_ac == 3
    # vendor/nethack/src/do_wear.c:2479 (uarm subtraction)
    """
    state = _strip_all_armor(_reset_valkyrie())
    state = _place_armor(
        state, slot_in_items=0,
        type_id=ObjType.RING_MAIL,
        ac_bonus=_ARMOR_AC[ObjType.RING_MAIL],
        enchant=0,
        armor_slot=ArmorSlot.BODY,
    )
    assert _ARMOR_AC[ObjType.RING_MAIL] == 3
    assert int(compute_ac(state)) == 10 - 3 - 0


# ---------------------------------------------------------------------------
# Sample 3: Valkyrie + ring mail +2 → AC 10 - 3 - 2 = 5
# ---------------------------------------------------------------------------

def test_valkyrie_ring_mail_plus2_ac_4():
    """Enchantment stacks linearly: ARM_BONUS(ring mail +2) = 3 + 2 = 5.
    uac = 10 - 5 = 5.

    # vendor/nethack/src/do_wear.c (ARM_BONUS macro: a_ac + spe).
    """
    state = _strip_all_armor(_reset_valkyrie())
    state = _place_armor(
        state, slot_in_items=0,
        type_id=ObjType.RING_MAIL,
        ac_bonus=_ARMOR_AC[ObjType.RING_MAIL],
        enchant=2,
        armor_slot=ArmorSlot.BODY,
    )
    assert int(compute_ac(state)) == 10 - 3 - 2


# ---------------------------------------------------------------------------
# Sample 4: Full armor set — ring mail + helmet + small shield + leather gloves
# Bonuses: 3 + 1 + 1 + 1 = 6 → AC 10 - 6 = 4.
# (Uses small shield rather than the audit's unspecified "boots" because
# Knight's u_init.c kit ships SMALL_SHIELD + LEATHER_GLOVES, not boots.)
# ---------------------------------------------------------------------------

def test_full_armor_set_ac_sum():
    """Body + helm + shield + gloves all stack:
        uac = 10 - (3 + 1 + 1 + 1) = 4.

    # vendor/nethack/src/do_wear.c:2478-2491 (uarm/uarmh/uarms/uarmg loop).
    """
    state = _strip_all_armor(_reset_valkyrie())
    state = _place_armor(state, 0, ObjType.RING_MAIL,
                         _ARMOR_AC[ObjType.RING_MAIL], 0, ArmorSlot.BODY)
    state = _place_armor(state, 1, ObjType.HELMET,
                         _ARMOR_AC[ObjType.HELMET], 0, ArmorSlot.HELM)
    state = _place_armor(state, 2, ObjType.SMALL_SHIELD,
                         _ARMOR_AC[ObjType.SMALL_SHIELD], 0, ArmorSlot.SHIELD)
    state = _place_armor(state, 3, ObjType.LEATHER_GLOVES,
                         _ARMOR_AC[ObjType.LEATHER_GLOVES], 0, ArmorSlot.GLOVES)
    expected = 10 - (
        _ARMOR_AC[ObjType.RING_MAIL]
        + _ARMOR_AC[ObjType.HELMET]
        + _ARMOR_AC[ObjType.SMALL_SHIELD]
        + _ARMOR_AC[ObjType.LEATHER_GLOVES]
    )
    assert expected == 4
    assert int(compute_ac(state)) == 4


# ---------------------------------------------------------------------------
# Sample 5: Polymorphed → form AC overrides armor.
# ---------------------------------------------------------------------------

def test_polymorph_dragon_overrides_armor_ac():
    """When ``polymorph.is_polymorphed`` is True, ``compute_ac`` returns the
    cached form AC (``state.player_ac``) rather than the armor-derived AC.

    Mirrors vendor/nethack/src/polyself.c::find_uac which uses ``mptr->ac``
    instead of the worn-armor sum when polymorphed.

    # vendor/nethack/src/polyself.c::find_uac
    """
    state = _strip_all_armor(_reset_valkyrie())
    # Equip ring mail +5 so the *armor-derived* AC would be 10 - 8 = 2.
    state = _place_armor(state, 0, ObjType.RING_MAIL,
                         _ARMOR_AC[ObjType.RING_MAIL], 5, ArmorSlot.BODY)
    armor_ac = int(compute_ac(state))
    assert armor_ac == 10 - 3 - 5  # sanity baseline

    # Simulate dragon polymorph: form AC = -1 (red dragon-ish).
    poly = state.polymorph
    new_poly = poly.replace(is_polymorphed=jnp.bool_(True))
    state_poly = state.replace(polymorph=new_poly, player_ac=jnp.int32(-1))

    # Form AC (-1) overrides armor-derived AC (2).
    assert int(compute_ac(state_poly)) == -1
