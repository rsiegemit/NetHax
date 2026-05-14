"""Wave 3 tests for inventory operations.

Tests cover:
  - pickup: ground cleared, inventory slot filled
  - drop: inventory slot cleared, ground populated
  - wield: state.inventory.wielded updated
  - wear_armor: player_ac drops by armor's AC contribution
  - take_off_armor: player_ac restored
"""
import pytest


def _make_state():
    """Helper: return a default EnvState with an EMPTY inventory.

    Bypasses character creation so the starting Valkyrie inventory (which
    occupies slots 0-4) doesn't interfere with single-item pickup tests.
    """
    import jax
    from Nethax.nethax.state import EnvState
    rng = jax.random.PRNGKey(42)
    return EnvState.default(rng=rng)


def _make_ground_items(n_branches, max_levels, map_h, map_w):
    from Nethax.nethax.subsystems.inventory import _empty_ground_items_array
    return _empty_ground_items_array(n_branches, max_levels, map_h, map_w)


def _place_item_on_ground(ground_items, item, branch, level, row, col, stack_idx=0):
    """Place a single item into a specific ground tile stack position."""
    import jax.numpy as jnp
    return ground_items.replace(
        category=ground_items.category.at[branch, level, row, col, stack_idx].set(
            item.category
        ),
        type_id=ground_items.type_id.at[branch, level, row, col, stack_idx].set(
            item.type_id
        ),
        weight=ground_items.weight.at[branch, level, row, col, stack_idx].set(
            item.weight
        ),
        quantity=ground_items.quantity.at[branch, level, row, col, stack_idx].set(
            item.quantity
        ),
        ac_bonus=ground_items.ac_bonus.at[branch, level, row, col, stack_idx].set(
            item.ac_bonus
        ),
        is_two_handed=ground_items.is_two_handed.at[branch, level, row, col, stack_idx].set(
            item.is_two_handed
        ),
        buc_status=ground_items.buc_status.at[branch, level, row, col, stack_idx].set(
            item.buc_status
        ),
        enchantment=ground_items.enchantment.at[branch, level, row, col, stack_idx].set(
            item.enchantment
        ),
        identified=ground_items.identified.at[branch, level, row, col, stack_idx].set(
            item.identified
        ),
    )


# ---------------------------------------------------------------------------
# Pickup tests
# ---------------------------------------------------------------------------

def test_pickup_fills_inventory_slot():
    """Picking up an item from the ground populates the first inventory slot."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        pickup, make_item, ItemCategory,
        _empty_ground_items_array,
    )
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _make_state()
    rng   = jax.random.PRNGKey(0)
    b, l, h, w = N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    sword = make_item(category=ItemCategory.WEAPON, type_id=34, quantity=1, weight=40)

    # Place sword at player's current position (row=0, col=0 after default reset)
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])

    ground = _make_ground_items(b, l, h, w)
    ground = _place_item_on_ground(ground, sword, 0, 0, row, col)

    new_state, new_ground = pickup(state, rng, ground, 0, 0)

    # Inventory slot 0 should now be a WEAPON
    assert int(new_state.inventory.items.category[0]) == int(ItemCategory.WEAPON), (
        "Expected WEAPON in slot 0 after pickup"
    )
    assert int(new_state.inventory.items.type_id[0]) == 34


def test_pickup_clears_ground_tile():
    """The ground tile must be cleared (category==0) after pickup."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        pickup, make_item, ItemCategory,
        _empty_ground_items_array,
    )
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _make_state()
    rng   = jax.random.PRNGKey(1)
    b, l, h, w = N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    item  = make_item(category=ItemCategory.POTION, type_id=205, quantity=1, weight=20)
    row   = int(state.player_pos[0])
    col   = int(state.player_pos[1])

    ground = _make_ground_items(b, l, h, w)
    ground = _place_item_on_ground(ground, item, 0, 0, row, col)

    _, new_ground = pickup(state, rng, ground, 0, 0)

    ground_cat = int(new_ground.category[0, 0, row, col, 0])
    assert ground_cat == 0, f"Ground tile should be cleared; got category={ground_cat}"


def test_pickup_updates_total_weight():
    """total_weight should increase after picking up a weighted item."""
    import jax
    from Nethax.nethax.subsystems.inventory import (
        pickup, make_item, ItemCategory,
        _empty_ground_items_array,
    )
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _make_state()
    rng   = jax.random.PRNGKey(2)
    b, l, h, w = N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    item  = make_item(category=ItemCategory.WEAPON, type_id=34, quantity=1, weight=40)
    row   = int(state.player_pos[0])
    col   = int(state.player_pos[1])

    ground = _make_ground_items(b, l, h, w)
    ground = _place_item_on_ground(ground, item, 0, 0, row, col)

    initial_weight = int(state.inventory.total_weight)
    new_state, _ = pickup(state, rng, ground, 0, 0)

    assert int(new_state.inventory.total_weight) == initial_weight + 40, (
        f"Expected weight {initial_weight + 40}; got {int(new_state.inventory.total_weight)}"
    )


# ---------------------------------------------------------------------------
# Drop tests
# ---------------------------------------------------------------------------

def test_drop_clears_inventory_slot():
    """Dropping an item should zero the inventory slot."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        drop, make_item, ItemCategory, InventoryState,
        _empty_ground_items_array, _items_from_list,
    )
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _make_state()
    rng   = jax.random.PRNGKey(3)
    b, l, h, w = N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    sword = make_item(category=ItemCategory.WEAPON, type_id=34, quantity=1, weight=40)
    new_inv = state.inventory.replace(items=_items_from_list([sword]))
    state   = state.replace(inventory=new_inv)

    ground = _make_ground_items(b, l, h, w)
    new_state, _ = drop(state, rng, ground, 0, 0, 0)

    assert int(new_state.inventory.items.category[0]) == 0, (
        "Inventory slot 0 should be cleared after drop"
    )


def test_drop_populates_ground():
    """Dropping an item should place it in the ground stack at player_pos."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        drop, make_item, ItemCategory, InventoryState,
        _empty_ground_items_array, _items_from_list,
    )
    from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    state = _make_state()
    rng   = jax.random.PRNGKey(4)
    b, l, h, w = N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W

    sword = make_item(category=ItemCategory.WEAPON, type_id=34, quantity=1, weight=40)
    new_inv = state.inventory.replace(items=_items_from_list([sword]))
    state   = state.replace(inventory=new_inv)

    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    ground = _make_ground_items(b, l, h, w)

    _, new_ground = drop(state, rng, ground, 0, 0, 0)

    ground_cat = int(new_ground.category[0, 0, row, col, 0])
    assert ground_cat == int(ItemCategory.WEAPON), (
        f"Expected WEAPON on ground; got category={ground_cat}"
    )


# ---------------------------------------------------------------------------
# Wield tests
# ---------------------------------------------------------------------------

def test_wield_sets_wielded_slot():
    """wield() should set inventory.wielded to the given slot index."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        wield, make_item, ItemCategory, _items_from_list,
    )

    state = _make_state()
    sword = make_item(category=ItemCategory.WEAPON, type_id=34, quantity=1, weight=40)
    new_inv = state.inventory.replace(items=_items_from_list([sword]))
    state   = state.replace(inventory=new_inv)

    new_state = wield(state, 0)

    assert int(new_state.inventory.wielded) == 0, (
        f"Expected wielded=0; got {int(new_state.inventory.wielded)}"
    )


def test_wield_two_handed_unequips_shield():
    """Wielding a two-handed weapon must unequip any worn shield."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        wield, make_item, ItemCategory, ArmorSlot, _items_from_list,
    )

    state  = _make_state()
    sword  = make_item(category=ItemCategory.WEAPON, type_id=35, quantity=1,
                       weight=70, is_two_handed=True)
    shield = make_item(category=ItemCategory.ARMOR,  type_id=75, quantity=1,
                       weight=30, ac_bonus=1)

    items  = _items_from_list([sword, shield])
    worn   = jnp.full((7,), -1, dtype=jnp.int8)
    worn   = worn.at[int(ArmorSlot.SHIELD)].set(jnp.int8(1))
    new_inv = state.inventory.replace(items=items, worn_armor=worn)
    state  = state.replace(inventory=new_inv)

    new_state = wield(state, 0)

    shield_slot_val = int(new_state.inventory.worn_armor[int(ArmorSlot.SHIELD)])
    assert shield_slot_val == -1, (
        f"Shield should be unequipped when wielding two-hander; got {shield_slot_val}"
    )


# ---------------------------------------------------------------------------
# Wear / take_off tests
# ---------------------------------------------------------------------------

def test_wear_armor_reduces_player_ac():
    """Wearing leather armor (AC 2) from base 10 should yield AC 8."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        wear_armor, make_item, ItemCategory, ArmorSlot, _items_from_list,
        BASE_AC,
    )

    state  = _make_state()
    armor  = make_item(category=ItemCategory.ARMOR, type_id=62, quantity=1,
                       weight=150, ac_bonus=2)
    items  = _items_from_list([armor])
    new_inv = state.inventory.replace(items=items)
    state  = state.replace(inventory=new_inv, player_ac=jnp.int32(BASE_AC))

    new_state = wear_armor(state, 0, ArmorSlot.BODY)

    expected_ac = BASE_AC - 2  # 8
    assert int(new_state.player_ac) == expected_ac, (
        f"Expected AC {expected_ac} after wearing armor; got {int(new_state.player_ac)}"
    )
    assert int(new_state.inventory.worn_armor[int(ArmorSlot.BODY)]) == 0


def test_take_off_armor_restores_player_ac():
    """Taking off armor should restore player_ac to BASE_AC."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        wear_armor, take_off_armor, make_item, ItemCategory, ArmorSlot,
        _items_from_list, BASE_AC,
    )

    state  = _make_state()
    armor  = make_item(category=ItemCategory.ARMOR, type_id=62, quantity=1,
                       weight=150, ac_bonus=2)
    items  = _items_from_list([armor])
    new_inv = state.inventory.replace(items=items)
    state  = state.replace(inventory=new_inv, player_ac=jnp.int32(BASE_AC))

    # Wear then take off
    state     = wear_armor(state, 0, ArmorSlot.BODY)
    new_state = take_off_armor(state, ArmorSlot.BODY)

    assert int(new_state.player_ac) == BASE_AC, (
        f"Expected AC restored to {BASE_AC}; got {int(new_state.player_ac)}"
    )
    assert int(new_state.inventory.worn_armor[int(ArmorSlot.BODY)]) == -1


def test_wear_multiple_armor_pieces():
    """Wearing body armor + shield should sum both AC bonuses."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import (
        wear_armor, make_item, ItemCategory, ArmorSlot, _items_from_list, BASE_AC,
    )

    state  = _make_state()
    body   = make_item(category=ItemCategory.ARMOR, type_id=62, quantity=1,
                       weight=150, ac_bonus=2)
    shield = make_item(category=ItemCategory.ARMOR, type_id=75, quantity=1,
                       weight=30, ac_bonus=1)
    items  = _items_from_list([body, shield])
    new_inv = state.inventory.replace(items=items)
    state  = state.replace(inventory=new_inv, player_ac=jnp.int32(BASE_AC))

    state     = wear_armor(state, 0, ArmorSlot.BODY)
    new_state = wear_armor(state, 1, ArmorSlot.SHIELD)

    expected_ac = BASE_AC - 2 - 1  # 7
    assert int(new_state.player_ac) == expected_ac, (
        f"Expected AC {expected_ac}; got {int(new_state.player_ac)}"
    )
