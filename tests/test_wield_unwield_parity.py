"""Parity tests for cursed-stuck wield/wear mechanics.

Canonical sources:
  vendor/nethack/src/wield.c::welded()      lines 1051-1058
  vendor/nethack/src/do_wear.c              line 1900 cursed check
  vendor/nethack/src/end.c::done            lines 1084-1105 (lifesaving)
  vendor/nethack/src/pickup.c               loadstone cursed drop block

Covers:
  test_cursed_weapon_welds           — wield cursed → welded, unwield blocked
  test_uncursed_weapon_not_welded    — wield uncursed → unwield works
  test_uncurse_clears_welded         — remove_curse via prayer clears weld
  test_scroll_remove_curse_clears_welded — remove_curse via scroll clears weld
  test_cursed_armor_cannot_remove    — wear cursed armor → stuck
  test_uncursed_armor_can_remove     — wear uncursed armor → removable
  test_cursed_amulet_cannot_remove   — wear cursed amulet → stuck
  test_cursed_ring_cannot_remove     — wear cursed ring → stuck
  test_loadstone_cursed_cannot_drop  — drop cursed loadstone → no-op
  test_loadstone_uncursed_can_drop   — drop uncursed loadstone → succeeds
  test_lifesaving_consumes_cursed_amulet — cursed amulet of lifesaving fires
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import (
    wield, unwield, wear_armor, take_off_armor,
    ArmorSlot, ItemCategory, make_item, N_ARMOR_SLOTS,
    _empty_ground_items_array,
)
from Nethax.nethax.subsystems.items_jewelry import (
    AmuletEffect, RingEffect,
    wear_amulet, take_off_amulet,
    put_on_ring, take_off_ring,
    check_life_saving,
)
from Nethax.nethax.subsystems.status_effects import Intrinsic

_RNG = jax.random.PRNGKey(0)

BUC_CURSED   = 1
BUC_UNCURSED = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state():
    return EnvState.default(_RNG)


def _set_slot(items, slot: int, **fields):
    """Set individual item fields at slot via .at[slot].set() to preserve [52] shape."""
    for fname, val in fields.items():
        arr = getattr(items, fname)
        items = items.replace(**{fname: arr.at[slot].set(val)})
    return items


def _state_with_weapon(buc_status: int, is_two_handed: bool = False):
    """Return state with a weapon in slot 0 (preserves [52]-shape arrays)."""
    state = _base_state()
    new_items = _set_slot(
        state.inventory.items, 0,
        category=jnp.int8(int(ItemCategory.WEAPON)),
        type_id=jnp.int16(1),
        buc_status=jnp.int8(buc_status),
        is_two_handed=jnp.bool_(is_two_handed),
        quantity=jnp.int16(1),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _state_with_armor(buc_status: int):
    """Return state with armor in slot 0."""
    state = _base_state()
    new_items = _set_slot(
        state.inventory.items, 0,
        category=jnp.int8(int(ItemCategory.ARMOR)),
        type_id=jnp.int16(1),
        buc_status=jnp.int8(buc_status),
        ac_bonus=jnp.int8(3),
        quantity=jnp.int16(1),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _state_with_amulet(effect: AmuletEffect, buc_status: int):
    """Return state with an amulet in slot 0 (broadcast for jewelry scalar API).

    items_jewelry.py reads item.type_id/buc_status/enchantment as scalars via
    broadcast-replace — broadcast ALL fields that put_on_ring/wear_amulet read.
    """
    state = _base_state()
    new_items = state.inventory.items.replace(
        category=jnp.int8(int(ItemCategory.AMULET)),
        type_id=jnp.int16(int(effect)),
        buc_status=jnp.int8(buc_status),
        enchantment=jnp.int8(0),
        quantity=jnp.int16(1),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _state_with_ring(effect: RingEffect, buc_status: int):
    """Return state with a ring in slot 0 (broadcast for jewelry scalar API)."""
    state = _base_state()
    new_items = state.inventory.items.replace(
        category=jnp.int8(int(ItemCategory.RING)),
        type_id=jnp.int16(int(effect)),
        buc_status=jnp.int8(buc_status),
        enchantment=jnp.int8(0),
        quantity=jnp.int16(1),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _state_with_loadstone(buc_status: int):
    """Return state with a loadstone (type_id=443, GEM category) in slot 0."""
    LOADSTONE_TYPE_ID = 443
    state = _base_state()
    new_items = _set_slot(
        state.inventory.items, 0,
        category=jnp.int8(int(ItemCategory.GEM)),
        type_id=jnp.int16(LOADSTONE_TYPE_ID),
        buc_status=jnp.int8(buc_status),
        quantity=jnp.int16(1),
        weight=jnp.int32(500),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


# ---------------------------------------------------------------------------
# Weapon weld tests
# ---------------------------------------------------------------------------

def test_cursed_weapon_welds():
    """Wield cursed weapon → welded=True, unwield is a no-op.

    Cite: vendor/nethack/src/wield.c::welded() lines 1051-1058 — welded()
    returns 1 when obj==uwep and obj->cursed is true.
    """
    state = _state_with_weapon(BUC_CURSED)
    state = wield(state, 0)

    assert bool(state.inventory.welded), "welded should be True after wielding cursed weapon"
    assert int(state.inventory.wielded) == 0, "weapon should be wielded"

    state = unwield(state)
    assert int(state.inventory.wielded) == 0, "cursed weapon should remain wielded (welded)"


def test_uncursed_weapon_not_welded():
    """Wield uncursed weapon → welded=False, unwield succeeds."""
    state = _state_with_weapon(BUC_UNCURSED)
    state = wield(state, 0)

    assert not bool(state.inventory.welded), "welded should be False for uncursed weapon"

    state = unwield(state)
    assert int(state.inventory.wielded) == -1, "uncursed weapon should be unwielded"


def test_uncurse_clears_welded_prayer():
    """Wield cursed weapon, apply remove_curse via prayer → welded cleared, unwield works.

    Cite: vendor/nethack/src/wield.c::welded() — welded re-checks obj->cursed;
    once uncursed the weapon is no longer stuck.
    """
    from Nethax.nethax.subsystems.prayer import _apply_remove_curse

    state = _state_with_weapon(BUC_CURSED)
    state = wield(state, 0)
    assert bool(state.inventory.welded)

    state = _apply_remove_curse(state)
    assert not bool(state.inventory.welded), "welded should be False after remove_curse"

    state = unwield(state)
    assert int(state.inventory.wielded) == -1, "weapon should be unwieldable after uncursing"


def test_scroll_remove_curse_clears_welded():
    """Wield cursed weapon, apply scroll remove_curse → welded cleared.

    Cite: vendor/nethack/src/read.c::SCR_REMOVE_CURSE — uncurses wielded item.
    """
    from Nethax.nethax.subsystems.magic import _effect_remove_curse

    state = _state_with_weapon(BUC_CURSED)
    state = wield(state, 0)
    assert bool(state.inventory.welded)

    # magic.py uses dict-based state
    state_dict = {
        "inventory": state.inventory,
        "status": state.status,
    }
    result_dict = _effect_remove_curse(state_dict, _RNG)
    new_inv = result_dict["inventory"]
    assert not bool(new_inv.welded), "welded should be False after scroll remove_curse"


# ---------------------------------------------------------------------------
# Cursed armor tests
# ---------------------------------------------------------------------------

def test_cursed_armor_cannot_remove():
    """Wear cursed armor → worn_armor_welded set, take_off_armor is no-op.

    Cite: vendor/nethack/src/do_wear.c line 1900 — cursed armor cannot be
    removed.
    """
    state = _state_with_armor(BUC_CURSED)
    state = wear_armor(state, 0, ArmorSlot.BODY)

    assert bool(state.inventory.worn_armor_welded[int(ArmorSlot.BODY)]), \
        "worn_armor_welded[BODY] should be True after wearing cursed armor"
    assert int(state.inventory.worn_armor[int(ArmorSlot.BODY)]) == 0

    state = take_off_armor(state, ArmorSlot.BODY)
    assert int(state.inventory.worn_armor[int(ArmorSlot.BODY)]) == 0, \
        "cursed armor should remain worn (welded)"


def test_uncursed_armor_can_remove():
    """Wear uncursed armor → not stuck, take_off_armor succeeds."""
    state = _state_with_armor(BUC_UNCURSED)
    state = wear_armor(state, 0, ArmorSlot.BODY)

    assert not bool(state.inventory.worn_armor_welded[int(ArmorSlot.BODY)])

    state = take_off_armor(state, ArmorSlot.BODY)
    assert int(state.inventory.worn_armor[int(ArmorSlot.BODY)]) == -1, \
        "uncursed armor should be removable"


# ---------------------------------------------------------------------------
# Cursed amulet tests
# ---------------------------------------------------------------------------

def test_cursed_amulet_cannot_remove():
    """Wear cursed amulet → worn_amulet_welded set, take_off_amulet is no-op.

    Cite: vendor/nethack/src/do_wear.c Amulet_off cursed check.
    """
    state = _state_with_amulet(AmuletEffect.ESP, BUC_CURSED)
    state = wear_amulet(state, _RNG, 0)

    assert bool(state.inventory.worn_amulet_welded), \
        "worn_amulet_welded should be True after wearing cursed amulet"
    assert int(state.inventory.worn_amulet) == 0

    state = take_off_amulet(state)
    assert int(state.inventory.worn_amulet) == 0, \
        "cursed amulet should remain worn (welded)"


def test_uncursed_amulet_can_remove():
    """Wear uncursed amulet → not stuck, take_off_amulet succeeds."""
    state = _state_with_amulet(AmuletEffect.ESP, BUC_UNCURSED)
    state = wear_amulet(state, _RNG, 0)

    assert not bool(state.inventory.worn_amulet_welded)

    state = take_off_amulet(state)
    assert int(state.inventory.worn_amulet) == -1, \
        "uncursed amulet should be removable"


# ---------------------------------------------------------------------------
# Cursed ring tests
# ---------------------------------------------------------------------------

def test_cursed_ring_cannot_remove():
    """Wear cursed ring → worn_rings_welded set, take_off_ring is no-op.

    Cite: vendor/nethack/src/do_wear.c Ring_off_or_gone cursed check.
    """
    state = _state_with_ring(RingEffect.REGENERATION, BUC_CURSED)
    state = put_on_ring(state, _RNG, 0, hand=0)

    assert bool(state.inventory.worn_rings_welded[0]), \
        "worn_rings_welded[0] should be True after wearing cursed ring"
    assert int(state.inventory.worn_rings[0]) == 0

    state = take_off_ring(state, hand=0)
    assert int(state.inventory.worn_rings[0]) == 0, \
        "cursed ring should remain worn (welded)"


def test_uncursed_ring_can_remove():
    """Wear uncursed ring → not stuck, take_off_ring succeeds."""
    state = _state_with_ring(RingEffect.REGENERATION, BUC_UNCURSED)
    state = put_on_ring(state, _RNG, 0, hand=0)

    assert not bool(state.inventory.worn_rings_welded[0])

    state = take_off_ring(state, hand=0)
    assert int(state.inventory.worn_rings[0]) == -1, \
        "uncursed ring should be removable"


# ---------------------------------------------------------------------------
# Cursed loadstone tests
# ---------------------------------------------------------------------------

def test_loadstone_cursed_cannot_drop():
    """Drop cursed loadstone → inventory unchanged (no-op).

    Cite: vendor/nethack/src/pickup.c (items.c::doloadstone) — cursed
    loadstone cannot be dropped.
    """
    from Nethax.nethax.subsystems.inventory import drop
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _state_with_loadstone(BUC_CURSED)
    ground = _empty_ground_items_array(N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W)

    new_state, new_ground = drop(state, _RNG, ground, 0, 0, 0)

    # Item should still be in slot 0
    assert int(new_state.inventory.items.category[0]) != 0, \
        "cursed loadstone should not be dropped"
    # Ground should remain empty
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    assert int(new_ground.category[0, 0, row, col, 0]) == 0, \
        "ground should be empty (cursed loadstone not dropped)"


def test_loadstone_uncursed_can_drop():
    """Drop uncursed loadstone → inventory slot cleared."""
    from Nethax.nethax.subsystems.inventory import drop
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _state_with_loadstone(BUC_UNCURSED)
    ground = _empty_ground_items_array(N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W)

    new_state, new_ground = drop(state, _RNG, ground, 0, 0, 0)

    assert int(new_state.inventory.items.category[0]) == 0, \
        "uncursed loadstone slot should be cleared after drop"


# ---------------------------------------------------------------------------
# Lifesaving amulet — cursed but consumed anyway
# ---------------------------------------------------------------------------

def test_lifesaving_consumes_cursed_amulet():
    """Cursed amulet of lifesaving: fires on death and is consumed despite being stuck.

    Cite: vendor/nethack/src/end.c::done lines 1084-1105 — life-saving amulet
    is consumed (destroyed) regardless of curse status; welded flag is cleared.
    """
    from Nethax.nethax.subsystems.status_effects import add_intrinsic

    state = _state_with_amulet(AmuletEffect.LIFE_SAVING, BUC_CURSED)
    # Wear cursed amulet — becomes welded
    state = wear_amulet(state, _RNG, 0)
    assert bool(state.inventory.worn_amulet_welded), "amulet should be welded (cursed)"

    # Simulate lethal damage
    state = state.replace(
        done=jnp.bool_(True),
        player_hp=jnp.int32(0),
        status=add_intrinsic(state.status, Intrinsic.LIFESAVED),
    )

    new_state, saved = check_life_saving(state)

    assert bool(saved), "life-saving should have fired"
    assert not bool(new_state.done), "player should not be dead after life-saving"
    assert int(new_state.inventory.worn_amulet) == -1, \
        "amulet should be consumed (worn_amulet cleared)"
    assert not bool(new_state.inventory.worn_amulet_welded), \
        "welded flag should be cleared (amulet destroyed)"
    assert int(new_state.player_hp) > 0, "HP should be restored"
