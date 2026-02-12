"""Item primitives for Tier 2 and 3."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import ItemType, LEVITATION_ITEMS, COLD_ITEMS


def pickup_item(ground_items, player_pos, inventory):
    """Pick up first ground item at player position into first empty inventory slot.

    Args:
        ground_items: GroundItems struct
        player_pos: jnp.ndarray [2] — (row, col)
        inventory: Inventory struct

    Returns:
        new_ground_items: GroundItems with item removed
        new_inventory: Inventory with item added
        picked_up: bool — whether an item was picked up
    """
    # Find first ground item at player pos
    pos_match_r = ground_items.position[:, 0] == player_pos[0]
    pos_match_c = ground_items.position[:, 1] == player_pos[1]
    at_player = pos_match_r & pos_match_c & ground_items.mask
    any_item = jnp.any(at_player)
    item_idx = jnp.argmax(at_player)
    safe_idx = jnp.where(any_item, item_idx, 0)
    item_type = ground_items.type_id[safe_idx]

    # Find first empty inventory slot
    empty_slots = jnp.logical_not(inventory.item_mask)
    any_empty = jnp.any(empty_slots)
    slot_idx = jnp.argmax(empty_slots)
    safe_slot = jnp.where(any_empty, slot_idx, 0)

    can_pickup = any_item & any_empty

    # Update ground items: remove picked up item
    new_gi_mask = ground_items.mask.at[safe_idx].set(
        jnp.where(can_pickup, False, ground_items.mask[safe_idx])
    )
    new_ground = ground_items.replace(mask=new_gi_mask)

    # Update inventory: add item
    new_inv_ids = inventory.item_ids.at[safe_slot].set(
        jnp.where(can_pickup, item_type, inventory.item_ids[safe_slot])
    )
    new_inv_mask = inventory.item_mask.at[safe_slot].set(
        jnp.where(can_pickup, True, inventory.item_mask[safe_slot])
    )
    new_inv = inventory.replace(item_ids=new_inv_ids, item_mask=new_inv_mask)

    return new_ground, new_inv, can_pickup


def use_first_item(inventory, game_map, player_pos, player_hp, map_h, map_w):
    """Use the first item in inventory.

    Effects:
    - POTION_LEVITATION/RING_LEVITATION/BOOTS_LEVITATION: levitating=True
    - WAND_COLD/FROST_HORN: freeze lava in 5x5
    - SKELETON_KEY: has_key=True
    - APPLE: heal 5 HP

    Args:
        inventory: Inventory struct
        game_map: jnp.ndarray [map_h, map_w]
        player_pos: jnp.ndarray [2]
        player_hp: int
        map_h: int
        map_w: int

    Returns:
        new_inventory: Inventory with item consumed
        new_map: jnp.ndarray [map_h, map_w] (potentially frozen lava)
        new_hp: int (potentially healed)
        levitating: bool — whether levitation was granted
        has_key_flag: bool — whether a skeleton key was used
    """
    from Nethax.minihax.primitives.terrain import freeze_lava_around

    # Find first occupied slot
    has_item = jnp.any(inventory.item_mask)
    slot = jnp.argmax(inventory.item_mask)
    safe_slot = jnp.where(has_item, slot, 0)
    item = inventory.item_ids[safe_slot]

    # Check item type
    is_lev = jnp.any(item == LEVITATION_ITEMS)
    is_cold = jnp.any(item == COLD_ITEMS)
    is_key = item == ItemType.SKELETON_KEY
    is_apple = item == ItemType.APPLE

    # Apply cold effect
    frozen_map = freeze_lava_around(game_map, player_pos, map_h, map_w)
    new_map = jnp.where(has_item & is_cold, frozen_map, game_map)

    # Apply apple effect
    new_hp = jnp.where(has_item & is_apple, player_hp + 5, player_hp)

    # Remove used item from inventory
    new_ids = inventory.item_ids.at[safe_slot].set(
        jnp.where(has_item, ItemType.NONE, inventory.item_ids[safe_slot])
    )
    new_mask = inventory.item_mask.at[safe_slot].set(
        jnp.where(has_item, False, inventory.item_mask[safe_slot])
    )
    new_inv = inventory.replace(item_ids=new_ids, item_mask=new_mask)

    levitating = has_item & is_lev
    has_key_flag = has_item & is_key

    return new_inv, new_map, new_hp, levitating, has_key_flag
