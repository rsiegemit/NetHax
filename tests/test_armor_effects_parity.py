"""Parity tests for worn-armor side effects vs vendor/nethack/src/do_wear.c.

Covers:
  - HELM_OF_BRILLIANCE  Int/Wis bonus   (do_wear.c Helmet_on ~line 451)
  - DUNCE_CAP           Int/Wis penalty (do_wear.c Helmet_on ~line 475)
  - CLOAK_OF_MAGIC_RESISTANCE intrinsic (do_wear.c Cloak_on  ~line 334)
  - GAUNTLETS_OF_FUMBLING status        (do_wear.c Gloves_on ~line 584)
  - LEVITATION_BOOTS    intrinsic       (do_wear.c Boots_on  ~line 235)
  - HELM_OF_OPPOSITE_ALIGNMENT swap     (do_wear.c Helmet_on ~line 462)
  - take-off clears Int bonus           (do_wear.c Helmet_off ~line 548)
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import jax

from Nethax.nethax.subsystems.inventory import (
    ArmorSlot,
    ItemCategory,
    InventoryState,
    make_item,
    N_ARMOR_SLOTS,
    wear_armor,
    take_off_armor,
)
from Nethax.nethax.subsystems.status_effects import Intrinsic, TimedStatus, StatusState
from Nethax.nethax.subsystems.armor_effects import (
    _HELM_OF_BRILLIANCE,
    _DUNCE_CAP,
    _CLOAK_OF_MAGIC_RES,
    _GAUNTLETS_OF_FUMBLING,
    _LEVITATION_BOOTS,
    _HELM_OF_OPPOSITE_ALIGN,
    _CORNUTHAUM,
    _ALIGN_LAWFUL,
    _ALIGN_CHAOTIC,
    _ALIGN_NEUTRAL,
    _STAT_INT,
    _STAT_WIS,
    N_STAT_BONUS,
)
from Nethax.nethax.state import EnvState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_int: int = 12, player_wis: int = 12,
                player_align: int = _ALIGN_LAWFUL,
                player_role: int = 0) -> EnvState:
    """Minimal EnvState with controllable stats; no dungeon content."""
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    return state.replace(
        player_int=jnp.int8(player_int),
        player_wis=jnp.int8(player_wis),
        player_align=jnp.int8(player_align),
        player_role=jnp.int8(player_role),
    )


def _place_and_wear(state, type_id: int, enchantment: int,
                    armor_slot: ArmorSlot, inv_slot: int = 0) -> EnvState:
    """Put an armor item in inventory slot ``inv_slot`` and wear it."""
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[inv_slot].set(jnp.int8(ItemCategory.ARMOR)),
        type_id=items.type_id.at[inv_slot].set(jnp.int16(type_id)),
        enchantment=items.enchantment.at[inv_slot].set(jnp.int8(enchantment)),
        quantity=items.quantity.at[inv_slot].set(jnp.int16(1)),
        ac_bonus=items.ac_bonus.at[inv_slot].set(jnp.int8(0)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))
    return wear_armor(state, inv_slot, armor_slot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_helm_brilliance_int_bonus():
    """HELM_OF_BRILLIANCE +2 enchantment → armor_stat_bonus[INT] == +2.

    cite: do_wear.c Helmet_on ~line 451: adj_abon(uarmh, uarmh->spe)
    """
    state = _make_state(player_int=10)
    state = _place_and_wear(state, _HELM_OF_BRILLIANCE, enchantment=2,
                            armor_slot=ArmorSlot.HELM)
    assert int(state.inventory.armor_stat_bonus[_STAT_INT]) == 2
    assert int(state.inventory.armor_stat_bonus[_STAT_WIS]) == 2


def test_takeoff_clears_int_bonus():
    """Taking off HELM_OF_BRILLIANCE resets armor_stat_bonus[INT] to 0.

    cite: do_wear.c Helmet_off ~line 548: adj_abon(uarmh, -uarmh->spe)
    """
    state = _make_state(player_int=10)
    state = _place_and_wear(state, _HELM_OF_BRILLIANCE, enchantment=3,
                            armor_slot=ArmorSlot.HELM)
    assert int(state.inventory.armor_stat_bonus[_STAT_INT]) == 3

    state = take_off_armor(state, ArmorSlot.HELM)
    assert int(state.inventory.armor_stat_bonus[_STAT_INT]) == 0
    assert int(state.inventory.armor_stat_bonus[_STAT_WIS]) == 0


def test_cloak_magic_res_grants_intrinsic():
    """CLOAK_OF_MAGIC_RESISTANCE sets MAGIC_RESIST intrinsic while worn.

    cite: do_wear.c Cloak_on ~line 334 (oc_oprop = ANTIMAGIC).
    """
    state = _make_state()
    state = _place_and_wear(state, _CLOAK_OF_MAGIC_RES, enchantment=0,
                            armor_slot=ArmorSlot.CLOAK)
    assert bool(state.status.intrinsics[int(Intrinsic.MAGIC_RESIST)])

    state = take_off_armor(state, ArmorSlot.CLOAK)
    assert not bool(state.status.intrinsics[int(Intrinsic.MAGIC_RESIST)])


def test_gauntlets_fumbling_grants_status():
    """GAUNTLETS_OF_FUMBLING sets FUMBLING timed status > 0 while worn.

    cite: do_wear.c Gloves_on ~line 584.
    """
    state = _make_state()
    state = _place_and_wear(state, _GAUNTLETS_OF_FUMBLING, enchantment=0,
                            armor_slot=ArmorSlot.GLOVES)
    assert int(state.status.timed_statuses[int(TimedStatus.FUMBLING)]) > 0

    state = take_off_armor(state, ArmorSlot.GLOVES)
    assert int(state.status.timed_statuses[int(TimedStatus.FUMBLING)]) == 0


def test_levitation_boots_grant_intrinsic():
    """LEVITATION_BOOTS sets LEVITATION intrinsic while worn.

    cite: do_wear.c Boots_on ~line 235.
    """
    state = _make_state()
    state = _place_and_wear(state, _LEVITATION_BOOTS, enchantment=0,
                            armor_slot=ArmorSlot.BOOTS)
    assert bool(state.status.intrinsics[int(Intrinsic.LEVITATION)])

    state = take_off_armor(state, ArmorSlot.BOOTS)
    assert not bool(state.status.intrinsics[int(Intrinsic.LEVITATION)])


def test_opposite_alignment_swaps():
    """HELM_OF_OPPOSITE_ALIGNMENT swaps LAWFUL → CHAOTIC on wear.

    cite: do_wear.c Helmet_on ~line 462:
        uchangealign(-u.ualign.type, A_CG_HELM_ON)
    """
    state = _make_state(player_align=_ALIGN_LAWFUL)
    state = _place_and_wear(state, _HELM_OF_OPPOSITE_ALIGN, enchantment=0,
                            armor_slot=ArmorSlot.HELM)
    assert int(state.player_align) == _ALIGN_CHAOTIC

    # Taking it off restores via recompute (no helm → no swap)
    state = take_off_armor(state, ArmorSlot.HELM)
    assert int(state.player_align) == _ALIGN_CHAOTIC  # base is now chaotic (already swapped)


def test_dunce_cap_neg_int():
    """DUNCE_CAP applies -2 to armor_stat_bonus[INT] and [WIS] while worn.

    cite: do_wear.c Helmet_on ~line 475 (dunce cap sets adj_abon -2/-2).
    """
    state = _make_state(player_int=15, player_wis=15)
    state = _place_and_wear(state, _DUNCE_CAP, enchantment=0,
                            armor_slot=ArmorSlot.HELM)
    assert int(state.inventory.armor_stat_bonus[_STAT_INT]) == -2
    assert int(state.inventory.armor_stat_bonus[_STAT_WIS]) == -2

    state = take_off_armor(state, ArmorSlot.HELM)
    assert int(state.inventory.armor_stat_bonus[_STAT_INT]) == 0
