"""Item primitives for Tier 2 and 3."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    ItemType, LEVITATION_ITEMS, COLD_ITEMS, DIRECTIONAL_ITEMS,
    APPLY_ITEMS, FOOD_ITEMS,
    TileType, SOLID_TILES, DIRECTION_VECTORS,
)


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

    # Find first occupied slot that is usable by APPLY (Tier 2 single-step)
    # (skip WAND_DEATH which requires ZAP + direction)
    usable_mask = inventory.item_mask & (inventory.item_ids != ItemType.WAND_DEATH)
    has_item = jnp.any(usable_mask)
    slot = jnp.argmax(usable_mask)
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


# ============================================================================
# Directional zap (Tier 3 only — multi-step: ZAP action then direction)
# ============================================================================

def has_any_zappable(inventory):
    """Check if inventory contains any directional item (wand/horn).

    Returns:
        has_any: bool — True if a zappable item exists
    """
    is_directional = jnp.isin(inventory.item_ids, DIRECTIONAL_ITEMS) & inventory.item_mask
    return jnp.any(is_directional)


def check_zap_slot(inventory, slot_idx):
    """Check if the given inventory slot contains a valid directional item.

    Args:
        inventory: Inventory struct
        slot_idx: int — inventory slot to check

    Returns:
        valid: bool — True if slot has a zappable item
        item_type: int — ItemType at slot (only meaningful if valid)
    """
    has_item = inventory.item_mask[slot_idx]
    item_type = inventory.item_ids[slot_idx]
    is_directional = jnp.isin(item_type, DIRECTIONAL_ITEMS)
    return has_item & is_directional, item_type


def has_any_applicable(inventory):
    """Check if inventory contains any self-targeted item (potion/ring/boots/key)."""
    is_applicable = jnp.isin(inventory.item_ids, APPLY_ITEMS) & inventory.item_mask
    return jnp.any(is_applicable)


def has_any_food(inventory):
    """Check if inventory contains any food item."""
    is_food = jnp.isin(inventory.item_ids, FOOD_ITEMS) & inventory.item_mask
    return jnp.any(is_food)


def check_apply_slot(inventory, slot_idx):
    """Check if the given inventory slot contains a valid self-targeted item.

    Returns:
        valid: bool — True if slot has an APPLY-compatible item
        item_type: int — ItemType at slot (only meaningful if valid)
    """
    has_item = inventory.item_mask[slot_idx]
    item_type = inventory.item_ids[slot_idx]
    is_applicable = jnp.isin(item_type, APPLY_ITEMS)
    return has_item & is_applicable, item_type


def check_food_slot(inventory, slot_idx):
    """Check if the given inventory slot contains a food item.

    Returns:
        valid: bool — True if slot has a food item
        item_type: int — ItemType at slot (only meaningful if valid)
    """
    has_item = inventory.item_mask[slot_idx]
    item_type = inventory.item_ids[slot_idx]
    is_food = jnp.isin(item_type, FOOD_ITEMS)
    return has_item & is_food, item_type


def apply_item_at_slot(inventory, slot_idx, valid):
    """Apply a self-targeted item at slot_idx. Consume it and return effects.

    Handles: levitation items → levitating=True, skeleton key → has_key=True.

    Args:
        inventory: Inventory struct
        slot_idx: int — slot to use
        valid: bool — whether to actually apply (False = no-op)

    Returns:
        new_inventory: Inventory with item consumed
        got_levitation: bool
        got_key: bool
    """
    item_type = inventory.item_ids[slot_idx]
    is_lev = jnp.any(item_type == LEVITATION_ITEMS)
    is_key = item_type == ItemType.SKELETON_KEY

    got_levitation = valid & is_lev
    got_key = valid & is_key

    # Consume item
    new_ids = inventory.item_ids.at[slot_idx].set(
        jnp.where(valid, ItemType.NONE, inventory.item_ids[slot_idx])
    )
    new_mask = inventory.item_mask.at[slot_idx].set(
        jnp.where(valid, False, inventory.item_mask[slot_idx])
    )
    new_inv = inventory.replace(item_ids=new_ids, item_mask=new_mask)

    return new_inv, got_levitation, got_key


def eat_item_at_slot(inventory, slot_idx, player_hp, valid):
    """Eat a food item at slot_idx. Consume it and heal.

    Args:
        inventory: Inventory struct
        slot_idx: int — slot to eat from
        player_hp: int — current HP
        valid: bool — whether to actually eat (False = no-op)

    Returns:
        new_inventory: Inventory with item consumed
        new_hp: int — healed HP (apple = +5)
    """
    # Heal 5 HP for apple
    new_hp = jnp.where(valid, player_hp + 5, player_hp)

    # Consume item
    new_ids = inventory.item_ids.at[slot_idx].set(
        jnp.where(valid, ItemType.NONE, inventory.item_ids[slot_idx])
    )
    new_mask = inventory.item_mask.at[slot_idx].set(
        jnp.where(valid, False, inventory.item_mask[slot_idx])
    )
    new_inv = inventory.replace(item_ids=new_ids, item_mask=new_mask)

    return new_inv, new_hp


def consume_zap_item(inventory, slot_idx, valid):
    """Consume the item at slot_idx from inventory.

    Args:
        inventory: Inventory struct
        slot_idx: int — slot to consume
        valid: bool — whether to actually consume (False = no-op)

    Returns:
        new_inventory: Inventory with item removed
    """
    new_ids = inventory.item_ids.at[slot_idx].set(
        jnp.where(valid, ItemType.NONE, inventory.item_ids[slot_idx])
    )
    new_mask = inventory.item_mask.at[slot_idx].set(
        jnp.where(valid, False, inventory.item_mask[slot_idx])
    )
    return inventory.replace(item_ids=new_ids, item_mask=new_mask)


def apply_death_ray(player_pos, direction, monsters, game_map, map_h, map_w):
    """Fire a death ray from player_pos in direction. Kill first monster hit.

    Traces a straight line until hitting a wall or going out of bounds.

    Args:
        player_pos: jnp.ndarray [2]
        direction: jnp.ndarray [2] — (dr, dc) unit direction vector
        monsters: Monsters struct (with .position, .health, .mask, .type_id)
        game_map: jnp.ndarray [map_h, map_w]
        map_h: int
        map_w: int

    Returns:
        new_monsters: Monsters with killed monster removed
        killed: bool — whether a monster was hit
        killed_idx: int — index of killed monster (only valid if killed)
        killed_type: int — MonsterType of killed monster (only valid if killed)
    """
    max_dist = 80
    offsets = jnp.arange(1, max_dist + 1)
    ray_r = player_pos[0] + offsets * direction[0]
    ray_c = player_pos[1] + offsets * direction[1]

    # Check bounds
    in_bounds = (ray_r >= 0) & (ray_r < map_h) & (ray_c >= 0) & (ray_c < map_w)
    safe_r = jnp.clip(ray_r, 0, map_h - 1)
    safe_c = jnp.clip(ray_c, 0, map_w - 1)

    # Check solid tiles (ray stops at walls)
    tiles = game_map[safe_r, safe_c]
    is_wall = jnp.isin(tiles, SOLID_TILES)

    # Ray valid: in bounds, not wall, and no previous position was blocked
    blocked = ~in_bounds | is_wall
    first_block = jnp.argmax(blocked)
    has_block = jnp.any(blocked)
    ray_length = jnp.where(has_block, first_block, max_dist)
    ray_valid = jnp.arange(max_dist) < ray_length

    # Check each ray position against each monster
    # ray: [max_dist], monsters: [max_m]
    mon_r = monsters.position[:, 0]  # [max_m]
    mon_c = monsters.position[:, 1]  # [max_m]

    match_r = ray_r[:, None] == mon_r[None, :]  # [max_dist, max_m]
    match_c = ray_c[:, None] == mon_c[None, :]  # [max_dist, max_m]
    match = match_r & match_c & monsters.mask[None, :] & ray_valid[:, None]

    # Find first monster along the ray
    has_mon_at_step = jnp.any(match, axis=1)  # [max_dist]
    any_hit = jnp.any(has_mon_at_step)
    first_hit = jnp.argmax(has_mon_at_step)
    safe_hit = jnp.where(any_hit, first_hit, 0)

    # Which monster at that step?
    mon_idx = jnp.argmax(match[safe_hit])
    safe_mon_idx = jnp.where(any_hit, mon_idx, 0)
    killed_type = monsters.type_id[safe_mon_idx]

    # Kill it (set hp=0, mask=False)
    new_health = monsters.health.at[safe_mon_idx].set(
        jnp.where(any_hit, 0, monsters.health[safe_mon_idx])
    )
    new_mask = monsters.mask.at[safe_mon_idx].set(
        jnp.where(any_hit, False, monsters.mask[safe_mon_idx])
    )
    new_monsters = monsters.replace(health=new_health, mask=new_mask)

    return new_monsters, any_hit, safe_mon_idx, killed_type


def apply_cold_ray(player_pos, direction, game_map, map_h, map_w):
    """Fire a cold ray from player_pos in direction. Freeze all lava in path.

    Traces a straight line until hitting a wall or going out of bounds.
    All LAVA tiles along the ray become FLOOR.

    Args:
        player_pos: jnp.ndarray [2]
        direction: jnp.ndarray [2] — (dr, dc) unit direction vector
        game_map: jnp.ndarray [map_h, map_w]
        map_h: int
        map_w: int

    Returns:
        new_map: jnp.ndarray [map_h, map_w] with lava frozen to floor
    """
    max_dist = 80
    offsets = jnp.arange(1, max_dist + 1)
    ray_r = player_pos[0] + offsets * direction[0]
    ray_c = player_pos[1] + offsets * direction[1]

    in_bounds = (ray_r >= 0) & (ray_r < map_h) & (ray_c >= 0) & (ray_c < map_w)
    safe_r = jnp.clip(ray_r, 0, map_h - 1)
    safe_c = jnp.clip(ray_c, 0, map_w - 1)

    tiles = game_map[safe_r, safe_c]
    is_wall = jnp.isin(tiles, SOLID_TILES)

    # Ray stops at first wall/OOB
    blocked = ~in_bounds | is_wall
    first_block = jnp.argmax(blocked)
    has_block = jnp.any(blocked)
    ray_length = jnp.where(has_block, first_block, max_dist)
    ray_valid = jnp.arange(max_dist) < ray_length

    # Mark lava tiles along ray for freezing
    is_lava = (tiles == TileType.LAVA) & ray_valid & in_bounds

    # Build freeze mask on full map via flat indexing (safe for duplicate indices)
    flat_idx = safe_r * map_w + safe_c
    flat_freeze = jnp.zeros(map_h * map_w, dtype=jnp.bool_)
    flat_freeze = flat_freeze.at[flat_idx].max(is_lava)
    freeze_mask = flat_freeze.reshape(map_h, map_w)

    new_map = jnp.where(freeze_mask, TileType.FLOOR, game_map)
    return new_map
