import jax
import jax.numpy as jnp
from jax import lax

from nethax.nethax.util.game_logic_utils import *


def is_game_over(state, params, static_env_params):
    """Check terminal conditions: death, starvation, or max timesteps."""
    done_steps = state.timestep >= params.max_timesteps
    is_dead = state.player_hp <= 0
    is_starved = get_hunger_state(state.player_nutrition) == HungerState.STARVED

    return done_steps | is_dead | is_starved


# ============================================================================
# Combat
# ============================================================================
def do_melee_attack(rng, state, target_pos, params, static_params):
    """Attack a monster at target_pos with wielded weapon.

    Scans monster slots on the current level to find one at target_pos.
    Computes damage from wielded weapon (or 1d2 bare-handed).
    On kill: grants XP, checks level-up (gain 1d8 HP on level up),
    increments monsters_killed.
    """
    level = state.player_level
    max_m = static_params.max_monsters

    # Find the monster at target_pos by scanning all slots
    mon_positions = state.monsters.position[level]  # [max_m, 2]
    mon_mask = state.monsters.mask[level]            # [max_m]

    # Check which monster matches target_pos and is alive
    pos_match_r = mon_positions[:, 0] == target_pos[0]
    pos_match_c = mon_positions[:, 1] == target_pos[1]
    matches = pos_match_r & pos_match_c & mon_mask  # [max_m]

    # Pick the first matching slot (or max_m if none)
    match_indices = jnp.where(matches, jnp.arange(max_m), max_m)
    target_idx = jnp.min(match_indices)
    found = target_idx < max_m
    # Clamp to valid range for indexing (use 0 as safe default)
    safe_idx = jnp.where(found, target_idx, 0)

    # Compute damage
    rng, rng_dmg, rng_lvlup = jax.random.split(rng, 3)
    has_weapon = state.wielded_weapon >= 0
    weapon_inv_idx = jnp.where(has_weapon, state.wielded_weapon, 0)
    weapon_type = jnp.where(
        has_weapon,
        state.inventory.type_id[weapon_inv_idx],
        jnp.int32(WeaponType.NONE),
    )
    enchantment = jnp.where(
        has_weapon,
        state.inventory.enchantment[weapon_inv_idx],
        jnp.int32(0),
    )

    # Weapon damage (or bare-handed: 1d2)
    weapon_damage = compute_damage(rng_dmg, weapon_type, enchantment, state.player_strength)
    rng, rng_bare = jax.random.split(rng)
    bare_damage = jax.random.randint(rng_bare, (), 1, 3)  # 1d2
    damage = jnp.where(has_weapon, weapon_damage, bare_damage)
    damage = jnp.where(found, damage, 0)

    # Apply damage
    old_hp = state.monsters.health[level, safe_idx]
    new_hp = old_hp - damage
    killed = jnp.logical_and(found, new_hp <= 0)

    # Update monster health and mask
    new_health = state.monsters.health.at[level, safe_idx].set(
        jnp.where(found, new_hp, old_hp)
    )
    new_mask = state.monsters.mask.at[level, safe_idx].set(
        jnp.where(killed, False, state.monsters.mask[level, safe_idx])
    )
    monsters = state.monsters.replace(health=new_health, mask=new_mask)

    # XP on kill
    mon_type = state.monsters.type_id[level, safe_idx]
    xp_gain = jnp.where(killed, get_monster_xp(mon_type), 0)
    new_xp = state.player_xp + xp_gain

    # Level-up check
    old_level = state.player_xp_level
    next_level_xp = get_xp_for_level(old_level + 1)
    leveled_up = jnp.logical_and(killed, new_xp >= next_level_xp)
    new_xp_level = jnp.where(leveled_up, old_level + 1, old_level)
    new_xp_level = jnp.minimum(new_xp_level, MAX_PLAYER_LEVEL)

    # HP gain on level up: 1d8
    hp_gain = jnp.where(leveled_up, jax.random.randint(rng_lvlup, (), 1, 9), 0)
    new_max_hp = state.player_max_hp + hp_gain
    new_hp_player = state.player_hp + hp_gain

    # Monsters killed counter
    new_kills = state.monsters_killed + jnp.where(killed, 1, 0)

    state = state.replace(
        monsters=monsters,
        player_xp=new_xp,
        player_xp_level=new_xp_level,
        player_hp=new_hp_player,
        player_max_hp=new_max_hp,
        monsters_killed=new_kills,
    )

    return state


# ============================================================================
# Movement
# ============================================================================
def do_move(rng, state, direction, params, static_params):
    """Handle player movement in a direction.

    Handles: collision with walls, attacking monsters in the target tile,
    and auto-picking up gold.
    """
    delta = DIRECTION_VECTORS[direction]
    new_pos = state.player_position + delta

    # Bounds check
    valid = in_bounds(new_pos, static_params.map_size)

    # Check tile at destination
    target_tile = lax.dynamic_slice(
        state.map[state.player_level],
        (new_pos[0], new_pos[1]),
        (1, 1),
    ).squeeze()

    # Can we walk there? Treat closed doors as walkable
    is_closed_door = target_tile == TileType.CLOSED_DOOR
    walkable = jnp.logical_not(is_solid(target_tile)) | is_closed_door

    # Check for monster at destination (scan all slots on current level)
    level = state.player_level
    max_m = static_params.max_monsters
    mon_positions = state.monsters.position[level]  # [max_m, 2]
    mon_mask = state.monsters.mask[level]            # [max_m]

    pos_match_r = mon_positions[:, 0] == new_pos[0]
    pos_match_c = mon_positions[:, 1] == new_pos[1]
    monster_there = jnp.any(pos_match_r & pos_match_c & mon_mask)
    monster_at_dest = jnp.logical_and(valid, monster_there)

    # Can move if valid, walkable, and no monster (monsters trigger attack instead)
    can_move = jnp.logical_and(valid, walkable) & jnp.logical_not(monster_at_dest)

    # Compute both branches: attack state and move state
    rng, rng_attack = jax.random.split(rng)

    # Always compute attack state (both branches must be computed for jnp.where)
    attack_state = do_melee_attack(rng_attack, state, new_pos, params, static_params)

    # Move: update position and open door if moving through one
    new_position = jnp.where(can_move, new_pos, state.player_position)
    moved_through_door = can_move & is_closed_door
    new_map = state.map.at[state.player_level, new_pos[0], new_pos[1]].set(
        jnp.where(moved_through_door, jnp.int32(TileType.OPEN_DOOR), state.map[state.player_level, new_pos[0], new_pos[1]])
    )
    move_state = state.replace(
        player_position=new_position,
        player_direction=direction,
        map=new_map,
    )

    # Auto-pickup gold on the new tile
    # Ground items at the destination: check slot 0
    ground_cat = move_state.ground_items.category[level, new_position[0], new_position[1], 0]
    ground_qty = move_state.ground_items.quantity[level, new_position[0], new_position[1], 0]
    is_gold = jnp.logical_and(
        ground_cat == ItemCategory.GOLD,
        can_move,
    )
    gold_gained = jnp.where(is_gold, ground_qty, 0)
    new_gold = move_state.gold + gold_gained

    # Clear ground gold slot if picked up
    new_ground_cat = move_state.ground_items.category.at[
        level, new_position[0], new_position[1], 0
    ].set(jnp.where(is_gold, jnp.int32(ItemCategory.NONE), ground_cat))
    new_ground_qty = move_state.ground_items.quantity.at[
        level, new_position[0], new_position[1], 0
    ].set(jnp.where(is_gold, jnp.int32(0), ground_qty))

    new_ground_items = move_state.ground_items.replace(
        category=new_ground_cat,
        quantity=new_ground_qty,
    )
    move_state = move_state.replace(
        gold=new_gold,
        ground_items=new_ground_items,
    )

    # Select result: attack if monster at destination, else move
    # Use tree_map with jnp.where to pick between the two full states
    def _pick(attack_val, move_val):
        return jnp.where(monster_at_dest, attack_val, move_val)

    result_state = jax.tree.map(_pick, attack_state, move_state)

    return result_state


# ============================================================================
# Monster Combat
# ============================================================================
def do_monster_attacks(rng, state, params, static_params):
    """Process all monster attacks against the player.

    Scans all monster slots on the current level. For each alive, adjacent,
    hostile, awake monster: roll to-hit (d20 + monster_level >= player_ac),
    on hit compute damage and subtract from player_hp.
    """
    level = state.player_level
    max_m = static_params.max_monsters
    player_ac = compute_player_ac(state)

    rngs = jax.random.split(rng, max_m + 1)

    def attack_step(carry, i):
        hp = carry
        mon_rng = rngs[i + 1]
        mon_rng_hit, mon_rng_dmg = jax.random.split(mon_rng)

        alive = state.monsters.mask[level, i]
        mon_pos = state.monsters.position[level, i]
        dist = chebyshev_distance(mon_pos, state.player_position)
        adjacent = dist == 1

        mon_type = state.monsters.type_id[level, i]
        flags = MONSTER_FLAGS[mon_type]
        is_hostile = (flags & MF_HOSTILE) != 0
        is_awake = jnp.logical_not(state.monsters.asleep[level, i])

        can_attack = alive & adjacent & is_hostile & is_awake

        # To-hit roll: d20 + monster_level >= player_ac
        mon_level = MONSTER_STATS[mon_type, 0]
        hit_roll = jax.random.randint(mon_rng_hit, (), 1, 21)
        hits = (hit_roll + mon_level) >= player_ac

        # Compute damage
        damage = compute_monster_damage(mon_rng_dmg, mon_type)
        actual_damage = jnp.where(can_attack & hits, damage, 0)

        new_hp = hp - actual_damage
        return new_hp, None

    new_hp, _ = lax.scan(attack_step, state.player_hp, jnp.arange(max_m))
    state = state.replace(player_hp=new_hp)
    return state


# ============================================================================
# Items
# ============================================================================
def do_pickup(rng, state, params, static_params):
    """Pick up item(s) from the ground.

    Checks ground_items at player position (slot 0).
    Gold: add to state.gold and clear ground slot.
    Non-gold: find first empty inventory slot, copy all item fields, clear ground.
    """
    level = state.player_level
    pr, pc = state.player_position[0], state.player_position[1]

    # Check ground item at slot 0
    g_cat = state.ground_items.category[level, pr, pc, 0]
    g_type = state.ground_items.type_id[level, pr, pc, 0]
    g_qty = state.ground_items.quantity[level, pr, pc, 0]
    g_buc = state.ground_items.buc_status[level, pr, pc, 0]
    g_ench = state.ground_items.enchantment[level, pr, pc, 0]
    g_ident = state.ground_items.identified[level, pr, pc, 0]

    has_item = g_cat != ItemCategory.NONE
    is_gold = g_cat == ItemCategory.GOLD

    # Gold pickup
    gold_gain = jnp.where(is_gold, g_qty, 0)

    # Non-gold: find first empty inventory slot
    max_inv = static_params.max_inventory_size
    inv_empty = state.inventory.category == ItemCategory.NONE  # [max_inv]
    empty_indices = jnp.where(inv_empty, jnp.arange(max_inv), max_inv)
    first_empty = jnp.min(empty_indices)
    has_space = first_empty < max_inv
    safe_slot = jnp.where(has_space, first_empty, 0)

    do_inv_pickup = has_item & jnp.logical_not(is_gold) & has_space

    # Update inventory
    new_inv_cat = state.inventory.category.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_cat, state.inventory.category[safe_slot])
    )
    new_inv_type = state.inventory.type_id.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_type, state.inventory.type_id[safe_slot])
    )
    new_inv_qty = state.inventory.quantity.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_qty, state.inventory.quantity[safe_slot])
    )
    new_inv_buc = state.inventory.buc_status.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_buc, state.inventory.buc_status[safe_slot])
    )
    new_inv_ench = state.inventory.enchantment.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_ench, state.inventory.enchantment[safe_slot])
    )
    new_inv_ident = state.inventory.identified.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_ident, state.inventory.identified[safe_slot])
    )
    # charges: copy from ground (ground items use charges field too)
    g_charges = state.ground_items.charges[level, pr, pc, 0]
    new_inv_charges = state.inventory.charges.at[safe_slot].set(
        jnp.where(do_inv_pickup, g_charges, state.inventory.charges[safe_slot])
    )

    new_inventory = state.inventory.replace(
        category=new_inv_cat,
        type_id=new_inv_type,
        quantity=new_inv_qty,
        buc_status=new_inv_buc,
        enchantment=new_inv_ench,
        identified=new_inv_ident,
        charges=new_inv_charges,
    )

    # Clear ground slot if we picked something up
    did_pickup = has_item & (is_gold | do_inv_pickup)
    clear_cat = jnp.where(did_pickup, jnp.int32(ItemCategory.NONE), g_cat)
    clear_qty = jnp.where(did_pickup, jnp.int32(0), g_qty)
    clear_type = jnp.where(did_pickup, jnp.int32(0), g_type)
    clear_buc = jnp.where(did_pickup, jnp.int32(0), g_buc)
    clear_ench = jnp.where(did_pickup, jnp.int32(0), g_ench)
    clear_ident = jnp.where(did_pickup, jnp.int32(0), g_ident)
    clear_charges = jnp.where(did_pickup, jnp.int32(0), g_charges)

    new_g_cat = state.ground_items.category.at[level, pr, pc, 0].set(clear_cat)
    new_g_type = state.ground_items.type_id.at[level, pr, pc, 0].set(clear_type)
    new_g_qty = state.ground_items.quantity.at[level, pr, pc, 0].set(clear_qty)
    new_g_buc = state.ground_items.buc_status.at[level, pr, pc, 0].set(clear_buc)
    new_g_ench = state.ground_items.enchantment.at[level, pr, pc, 0].set(clear_ench)
    new_g_ident = state.ground_items.identified.at[level, pr, pc, 0].set(clear_ident)
    new_g_charges = state.ground_items.charges.at[level, pr, pc, 0].set(clear_charges)

    new_ground = state.ground_items.replace(
        category=new_g_cat,
        type_id=new_g_type,
        quantity=new_g_qty,
        buc_status=new_g_buc,
        enchantment=new_g_ench,
        identified=new_g_ident,
        charges=new_g_charges,
    )

    state = state.replace(
        gold=state.gold + gold_gain,
        inventory=new_inventory,
        ground_items=new_ground,
    )

    return state


def do_drop(rng, state, item_index, params, static_params):
    """Drop an item from inventory to the ground."""
    # TODO: Remove from inventory
    # TODO: Unequip if equipped
    # TODO: Place on ground at player position
    return state


def do_eat(rng, state, item_index, params, static_params):
    """Eat a food item.

    Finds the first food item in inventory.
    Adds FOOD_NUTRITION[food_type] to player_nutrition.
    Decrements quantity; clears slot if quantity reaches 0.
    """
    max_inv = static_params.max_inventory_size
    inv_cat = state.inventory.category  # [max_inv]

    # Find first food item in inventory
    is_food = inv_cat == ItemCategory.FOOD
    food_indices = jnp.where(is_food, jnp.arange(max_inv), max_inv)
    first_food = jnp.min(food_indices)
    has_food = first_food < max_inv
    safe_idx = jnp.where(has_food, first_food, 0)

    food_type = state.inventory.type_id[safe_idx]
    nutrition = FOOD_NUTRITION[food_type]
    nutrition_gain = jnp.where(has_food, nutrition, 0)

    # Decrement quantity
    old_qty = state.inventory.quantity[safe_idx]
    new_qty = old_qty - 1
    depleted = new_qty <= 0

    # If depleted, clear the slot entirely
    new_inv_cat = state.inventory.category.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(ItemCategory.NONE),
                   jnp.where(has_food, state.inventory.category[safe_idx],
                             state.inventory.category[safe_idx]))
    )
    new_inv_type = state.inventory.type_id.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(0), state.inventory.type_id[safe_idx])
    )
    new_inv_qty = state.inventory.quantity.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(0),
                   jnp.where(has_food, new_qty, old_qty))
    )
    new_inv_buc = state.inventory.buc_status.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(0), state.inventory.buc_status[safe_idx])
    )
    new_inv_ench = state.inventory.enchantment.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(0), state.inventory.enchantment[safe_idx])
    )
    new_inv_ident = state.inventory.identified.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(0), state.inventory.identified[safe_idx])
    )
    new_inv_charges = state.inventory.charges.at[safe_idx].set(
        jnp.where(has_food & depleted, jnp.int32(0), state.inventory.charges[safe_idx])
    )

    new_inventory = state.inventory.replace(
        category=new_inv_cat,
        type_id=new_inv_type,
        quantity=new_inv_qty,
        buc_status=new_inv_buc,
        enchantment=new_inv_ench,
        identified=new_inv_ident,
        charges=new_inv_charges,
    )

    state = state.replace(
        player_nutrition=state.player_nutrition + nutrition_gain,
        inventory=new_inventory,
    )

    return state


def do_quaff(rng, state, item_index, params, static_params):
    """Drink a potion."""
    # TODO: Look up potion type from appearance (identification system)
    # TODO: Apply potion effect
    # TODO: Mark potion type as identified if effect is obvious
    # TODO: Remove from inventory
    return state


def do_read(rng, state, item_index, params, static_params):
    """Read a scroll."""
    # TODO: Check not blind
    # TODO: Look up scroll type from appearance
    # TODO: Apply scroll effect (identify, teleport, enchant, etc.)
    # TODO: Consume scroll
    return state


def do_zap(rng, state, item_index, direction, params, static_params):
    """Zap a wand in a direction."""
    # TODO: Check wand has charges
    # TODO: Trace beam in direction
    # TODO: Apply effect to first monster/wall hit
    # TODO: Decrement charges
    return state


def do_wield(rng, state, item_index, params, static_params):
    """Wield a weapon.

    Finds the first weapon in inventory that isn't currently wielded.
    Sets wielded_weapon to that index (or leaves unchanged if none found).
    """
    max_inv = static_params.max_inventory_size
    inv_cat = state.inventory.category  # [max_inv]

    # Find first weapon that is NOT the currently wielded one
    is_weapon = inv_cat == ItemCategory.WEAPON
    not_wielded = jnp.arange(max_inv) != state.wielded_weapon
    candidates = is_weapon & not_wielded
    cand_indices = jnp.where(candidates, jnp.arange(max_inv), max_inv)
    first_weapon = jnp.min(cand_indices)
    has_weapon = first_weapon < max_inv

    new_wielded = jnp.where(has_weapon, first_weapon, state.wielded_weapon)

    state = state.replace(wielded_weapon=new_wielded)
    return state


def do_wear(rng, state, item_index, params, static_params):
    """Put on armor.

    Finds the first armor in inventory not already worn.
    Gets the armor slot from ARMOR_STATS[type, 4], maps to worn_armor index.
    If the slot is empty, equips and recomputes AC.
    """
    max_inv = static_params.max_inventory_size
    inv_cat = state.inventory.category  # [max_inv]

    # Check which inventory indices are already worn
    # worn_armor is [num_armor_slots], each is an inv index or -1
    worn_set = state.worn_armor  # [6]

    # For each inv slot, check if it's in worn_armor
    def _is_worn(inv_idx):
        return jnp.any(worn_set == inv_idx)

    is_worn_mask = jax.vmap(_is_worn)(jnp.arange(max_inv))  # [max_inv]

    is_armor = inv_cat == ItemCategory.ARMOR
    not_worn = jnp.logical_not(is_worn_mask)
    candidates = is_armor & not_worn
    cand_indices = jnp.where(candidates, jnp.arange(max_inv), max_inv)
    first_armor = jnp.min(cand_indices)
    has_armor = first_armor < max_inv
    safe_idx = jnp.where(has_armor, first_armor, 0)

    # Get armor slot from ARMOR_STATS column 4
    armor_type = state.inventory.type_id[safe_idx]
    armor_slot = ARMOR_STATS[armor_type, 4]  # ArmorSlot index

    # Check if that slot is currently empty
    current_in_slot = state.worn_armor[armor_slot]
    slot_empty = current_in_slot == -1
    do_equip = has_armor & slot_empty

    # Update worn_armor
    new_worn = state.worn_armor.at[armor_slot].set(
        jnp.where(do_equip, safe_idx, current_in_slot)
    )

    state = state.replace(worn_armor=new_worn)

    # Recompute AC
    new_ac = compute_player_ac(state)
    state = state.replace(player_ac=new_ac)

    return state


def do_remove_armor(rng, state, slot_index, params, static_params):
    """Remove worn armor."""
    # TODO: Check armor is not cursed
    # TODO: Unequip from slot
    # TODO: Recompute AC
    return state


def do_throw(rng, state, item_index, direction, params, static_params):
    """Throw an item in a direction."""
    # TODO: Remove from inventory (or decrement quantity for stacks)
    # TODO: Trace projectile path
    # TODO: Check for monster hit -> apply damage
    # TODO: Item lands on ground at final position
    return state


# ============================================================================
# Interactions
# ============================================================================
def do_open_door(rng, state, direction, params, static_params):
    """Open a closed door in a direction."""
    target = state.player_position + DIRECTION_VECTORS[direction]

    target_tile = lax.dynamic_slice(
        state.map[state.player_level],
        (target[0], target[1]),
        (1, 1),
    ).squeeze()

    is_closed_door = target_tile == TileType.CLOSED_DOOR

    # Open the door
    new_map = state.map.at[state.player_level, target[0], target[1]].set(
        jnp.where(is_closed_door, TileType.OPEN_DOOR, target_tile)
    )

    state = state.replace(map=new_map)
    return state


def do_close_door(rng, state, direction, params, static_params):
    """Close an open door in a direction."""
    target = state.player_position + DIRECTION_VECTORS[direction]

    target_tile = lax.dynamic_slice(
        state.map[state.player_level],
        (target[0], target[1]),
        (1, 1),
    ).squeeze()

    is_open_door = target_tile == TileType.OPEN_DOOR
    # TODO: Check no monster is standing in the doorway

    new_map = state.map.at[state.player_level, target[0], target[1]].set(
        jnp.where(is_open_door, TileType.CLOSED_DOOR, target_tile)
    )

    state = state.replace(map=new_map)
    return state


def do_search(rng, state, params, static_params):
    """Search adjacent tiles for hidden doors and traps."""
    # TODO: For each of 8 adjacent tiles:
    #   - Random chance to reveal hidden trap -> set traps_revealed
    #   - Random chance to find hidden door (future: hidden doors)
    return state


def do_kick(rng, state, direction, params, static_params):
    """Kick in a direction (break doors, push monsters)."""
    # TODO: If door -> chance to break it open
    # TODO: If monster -> small damage
    return state


def do_pray(rng, state, params, static_params):
    """Pray to your god for help."""
    # TODO: If critically low HP or starving, god may help
    # TODO: Prayer timeout system
    return state


def do_go_up(rng, state, params, static_params):
    """Climb upstairs."""
    pos = state.player_position
    tile = state.map[state.player_level, pos[0], pos[1]]
    on_upstair = tile == TileType.STAIRCASE_UP

    new_level = jnp.where(on_upstair, state.player_level - 1, state.player_level)
    new_level = jnp.maximum(new_level, 0)

    # Place player at the down stair of the level above
    new_pos = jnp.where(
        on_upstair,
        state.down_stairs[new_level],
        state.player_position,
    )

    state = state.replace(
        player_level=new_level,
        player_position=new_pos,
    )
    return state


def do_go_down(rng, state, params, static_params):
    """Descend downstairs."""
    pos = state.player_position
    tile = state.map[state.player_level, pos[0], pos[1]]
    on_downstair = tile == TileType.STAIRCASE_DOWN

    new_level = jnp.where(on_downstair, state.player_level + 1, state.player_level)
    new_level = jnp.minimum(new_level, static_params.num_levels - 1)

    # Place player at the up stair of the level below
    new_pos = jnp.where(
        on_downstair,
        state.up_stairs[new_level],
        state.player_position,
    )

    state = state.replace(
        player_level=new_level,
        player_position=new_pos,
    )
    return state


# ============================================================================
# Monster AI
# ============================================================================
def update_monsters(rng, state, params, static_params):
    """Update all monsters on the current level.

    Each monster either:
    - Sleeps (if asleep and not disturbed) with wake chance based on distance
    - Moves toward the player (if awake, hostile, and aware within dist<10)
    - Wanders randomly (if awake but not hostile or unaware)
    - Does not move if adjacent (attacks are handled separately)
    """
    level = state.player_level
    max_m = static_params.max_monsters
    player_pos = state.player_position

    rngs = jax.random.split(rng, max_m + 1)

    def monster_step(carry, i):
        mon_positions, mon_asleep, map_data = carry
        mon_rng = rngs[i + 1]
        rng_wake, rng_move = jax.random.split(mon_rng)

        alive = state.monsters.mask[level, i]
        pos = mon_positions[level, i]
        mon_type = state.monsters.type_id[level, i]
        was_asleep = mon_asleep[level, i]

        dist = chebyshev_distance(pos, player_pos)

        # Wake check: distance 2 -> 80%, distance 5 -> 30%, else 5%
        wake_prob = jnp.where(dist <= 2, 0.80,
                    jnp.where(dist <= 5, 0.30, 0.05))
        wake_roll = jax.random.uniform(rng_wake)
        wakes_up = jnp.logical_and(was_asleep, wake_roll < wake_prob)
        is_awake_now = jnp.logical_or(jnp.logical_not(was_asleep), wakes_up)
        new_asleep_val = jnp.where(wakes_up, False, was_asleep)

        flags = MONSTER_FLAGS[mon_type]
        is_hostile = (flags & MF_HOSTILE) != 0
        aware = dist < 10
        adjacent = dist <= 1

        # Movement logic
        # If awake, hostile, aware, and not adjacent: move toward player
        diff = player_pos - pos
        abs_diff = jnp.abs(diff)
        # Prefer larger axis for movement direction
        move_r = jnp.sign(diff[0])
        move_c = jnp.sign(diff[1])
        # Move toward player
        toward_delta = jnp.array([move_r, move_c])

        # Random cardinal movement for non-hostile/unaware
        rand_dir = jax.random.randint(rng_move, (), 0, 4)
        # Cardinal directions: N, S, E, W
        cardinal_deltas = jnp.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        random_delta = cardinal_deltas[rand_dir]

        should_pursue = is_awake_now & is_hostile & aware & jnp.logical_not(adjacent)
        should_wander = is_awake_now & jnp.logical_not(should_pursue) & jnp.logical_not(adjacent)

        delta = jnp.where(should_pursue, toward_delta,
                jnp.where(should_wander, random_delta,
                          jnp.array([0, 0])))

        new_pos = pos + delta

        # Validate: in bounds, not solid (treat closed doors as passable), not player tile
        valid_bounds = in_bounds(new_pos, static_params.map_size)

        # Read tile type at new_pos (safe indexing)
        safe_r = jnp.clip(new_pos[0], 0, static_params.map_size[0] - 1)
        safe_c = jnp.clip(new_pos[1], 0, static_params.map_size[1] - 1)
        tile_at_new = map_data[level, safe_r, safe_c]
        is_closed_door_tile = tile_at_new == TileType.CLOSED_DOOR
        not_solid = jnp.logical_not(is_solid(tile_at_new)) | is_closed_door_tile
        not_player = jnp.logical_not(
            jnp.logical_and(new_pos[0] == player_pos[0], new_pos[1] == player_pos[1])
        )

        valid_move = alive & valid_bounds & not_solid & not_player

        final_pos = jnp.where(valid_move, new_pos, pos)

        # Open door if monster moves through it
        opened_door = valid_move & is_closed_door_tile
        new_map = map_data.at[level, safe_r, safe_c].set(
            jnp.where(opened_door, jnp.int32(TileType.OPEN_DOOR), map_data[level, safe_r, safe_c])
        )

        new_positions = mon_positions.at[level, i].set(final_pos)
        new_asleep_arr = mon_asleep.at[level, i].set(
            jnp.where(alive, new_asleep_val, was_asleep)
        )

        return (new_positions, new_asleep_arr, new_map), None

    (new_positions, new_asleep, new_map), _ = lax.scan(
        monster_step,
        (state.monsters.position, state.monsters.asleep, state.map),
        jnp.arange(max_m),
    )

    monsters = state.monsters.replace(
        position=new_positions,
        asleep=new_asleep,
    )
    state = state.replace(monsters=monsters, map=new_map)
    return state


def spawn_monsters(rng, state, params, static_params):
    """Periodically spawn new monsters on the current level.

    ~2% chance per turn (1 in 50).
    Finds first empty monster slot on current level.
    Picks a valid monster type for the current depth.
    Places on a walkable tile NOT in the visible area.
    Monster starts asleep.
    """
    level = state.player_level
    max_m = static_params.max_monsters

    rng, rng_chance, rng_type, rng_pos_r, rng_pos_c = jax.random.split(rng, 5)

    # 2% chance to spawn
    do_spawn_roll = jax.random.uniform(rng_chance) < 0.02

    # Find first empty monster slot on current level
    mon_mask = state.monsters.mask[level]  # [max_m]
    empty_indices = jnp.where(jnp.logical_not(mon_mask), jnp.arange(max_m), max_m)
    first_empty = jnp.min(empty_indices)
    has_slot = first_empty < max_m
    safe_slot = jnp.where(has_slot, first_empty, 0)

    # Current depth (1-indexed for spawn table)
    depth = state.player_level + 1

    # Find valid monster types for this depth
    min_floors = MONSTER_SPAWN_FLOORS[:, 0]  # [NUM_MONSTER_TYPES]
    max_floors = MONSTER_SPAWN_FLOORS[:, 1]  # [NUM_MONSTER_TYPES]
    valid_type = (depth >= min_floors) & (depth <= max_floors)
    # Exclude NONE (type 0)
    valid_type = valid_type.at[0].set(False)

    # Pick a random valid type using weighted sampling (uniform among valid)
    # Create probabilities: valid types get 1, invalid get 0
    probs = jnp.where(valid_type, 1.0, 0.0)
    total = jnp.sum(probs)
    # If no valid types, set uniform to prevent NaN
    safe_probs = jnp.where(total > 0, probs / jnp.maximum(total, 1.0),
                           jnp.ones(NUM_MONSTER_TYPES) / NUM_MONSTER_TYPES)
    chosen_type = jax.random.choice(rng_type, NUM_MONSTER_TYPES, p=safe_probs)
    # Ensure we don't pick NONE
    chosen_type = jnp.where(chosen_type == 0, jnp.int32(1), chosen_type)

    # Pick a random walkable tile NOT in visible area
    map_h, map_w = static_params.map_size
    rand_r = jax.random.randint(rng_pos_r, (), 0, map_h)
    rand_c = jax.random.randint(rng_pos_c, (), 0, map_w)

    tile_at = state.map[level, rand_r, rand_c]
    tile_walkable = jnp.logical_not(is_solid(tile_at))
    not_visible = jnp.logical_not(state.visible[level, rand_r, rand_c])
    valid_pos = tile_walkable & not_visible

    do_spawn = do_spawn_roll & has_slot & valid_pos & (total > 0)

    # Monster HP
    new_mon_hp = get_monster_max_hp(chosen_type)

    # Update monster arrays
    new_positions = state.monsters.position.at[level, safe_slot].set(
        jnp.where(do_spawn, jnp.array([rand_r, rand_c]), state.monsters.position[level, safe_slot])
    )
    new_types = state.monsters.type_id.at[level, safe_slot].set(
        jnp.where(do_spawn, chosen_type, state.monsters.type_id[level, safe_slot])
    )
    new_health = state.monsters.health.at[level, safe_slot].set(
        jnp.where(do_spawn, new_mon_hp, state.monsters.health[level, safe_slot])
    )
    new_mask = state.monsters.mask.at[level, safe_slot].set(
        jnp.where(do_spawn, True, state.monsters.mask[level, safe_slot])
    )
    new_asleep = state.monsters.asleep.at[level, safe_slot].set(
        jnp.where(do_spawn, True, state.monsters.asleep[level, safe_slot])
    )

    monsters = state.monsters.replace(
        position=new_positions,
        type_id=new_types,
        health=new_health,
        mask=new_mask,
        asleep=new_asleep,
    )
    state = state.replace(monsters=monsters)
    return state


# ============================================================================
# Intrinsics / Status effects
# ============================================================================
def update_hunger(state, params):
    """Decrement nutrition each turn. Check for starvation."""
    new_nutrition = state.player_nutrition - params.hunger_rate
    state = state.replace(player_nutrition=new_nutrition)
    return state


def update_status_effects(state):
    """Decrement duration of temporary status effects."""
    state = state.replace(
        player_confused=jnp.maximum(state.player_confused - 1, 0),
        player_blind=jnp.maximum(state.player_blind - 1, 0),
        player_stunned=jnp.maximum(state.player_stunned - 1, 0),
        player_paralyzed=jnp.maximum(state.player_paralyzed - 1, 0),
        player_fast=jnp.maximum(state.player_fast - 1, 0),
        player_invisible=jnp.maximum(state.player_invisible - 1, 0),
    )
    return state


def update_hp_regen(state):
    """Regenerate HP slowly over time (if not starving)."""
    hunger = get_hunger_state(state.player_nutrition)
    can_regen = hunger < HungerState.WEAK

    # Regen 1 HP every ~20 turns (simplified)
    should_regen = jnp.logical_and(can_regen, state.timestep % 20 == 0)
    new_hp = jnp.where(
        should_regen,
        jnp.minimum(state.player_hp + 1, state.player_max_hp),
        state.player_hp,
    )
    state = state.replace(player_hp=new_hp)
    return state


# ============================================================================
# Field of View
# ============================================================================
def compute_fov(state, static_params):
    """Compute which tiles are visible from the player's position.

    Uses a simplified raycasting approach compatible with JAX.
    """
    # TODO: Implement shadowcasting or simple raycasting FOV
    # For now, use a simple radius-based visibility
    player_pos = state.player_position
    map_h, map_w = static_params.map_size

    row_indices = jnp.arange(map_h)
    col_indices = jnp.arange(map_w)
    rows, cols = jnp.meshgrid(row_indices, col_indices, indexing='ij')

    dist = jnp.maximum(
        jnp.abs(rows - player_pos[0]),
        jnp.abs(cols - player_pos[1]),
    )

    # Simple visibility: anything within radius 5 (rooms are usually visible)
    visible = dist <= 5

    # Update explored map
    new_explored = jnp.logical_or(
        state.explored[state.player_level],
        visible,
    )
    new_explored_full = state.explored.at[state.player_level].set(new_explored)

    new_visible = state.visible.at[state.player_level].set(visible)

    state = state.replace(explored=new_explored_full, visible=new_visible)
    return state


# ============================================================================
# Achievement tracking
# ============================================================================
def update_achievements(state, params, static_params):
    """Check and update achievement flags."""
    achievements = state.achievements

    # Exploration achievements
    achievements = achievements.at[Achievement.REACH_FLOOR_2].set(
        jnp.logical_or(achievements[Achievement.REACH_FLOOR_2], state.player_level >= 1)
    )
    achievements = achievements.at[Achievement.REACH_FLOOR_5].set(
        jnp.logical_or(achievements[Achievement.REACH_FLOOR_5], state.player_level >= 4)
    )
    achievements = achievements.at[Achievement.REACH_FLOOR_10].set(
        jnp.logical_or(achievements[Achievement.REACH_FLOOR_10], state.player_level >= 9)
    )

    # Combat
    achievements = achievements.at[Achievement.KILL_FIRST_MONSTER].set(
        jnp.logical_or(achievements[Achievement.KILL_FIRST_MONSTER], state.monsters_killed >= 1)
    )
    achievements = achievements.at[Achievement.KILL_10_MONSTERS].set(
        jnp.logical_or(achievements[Achievement.KILL_10_MONSTERS], state.monsters_killed >= 10)
    )

    # Level
    achievements = achievements.at[Achievement.REACH_XP_LEVEL_5].set(
        jnp.logical_or(achievements[Achievement.REACH_XP_LEVEL_5], state.player_xp_level >= 5)
    )
    achievements = achievements.at[Achievement.REACH_XP_LEVEL_10].set(
        jnp.logical_or(achievements[Achievement.REACH_XP_LEVEL_10], state.player_xp_level >= 10)
    )

    state = state.replace(achievements=achievements)
    return state


# ============================================================================
# Main step function
# ============================================================================
def nethax_step(rng, state, action, params, static_params):
    """Main game step function. Processes one player action and advances game state.

    Composes all sub-systems in sequence:
    1. Process player action (movement, combat, items, interactions)
    2. Update monsters (AI, attacks)
    3. Spawn new monsters
    4. Update hunger and status effects
    5. Regenerate HP
    6. Update field of view
    7. Track achievements and compute reward
    """
    rng, _rng_action, _rng_monsters, _rng_spawn = jax.random.split(rng, 4)

    # Increment timestep
    state = state.replace(timestep=state.timestep + 1)

    # ---- 1. Process player action ----
    # Movement (actions 1-8)
    is_move = jnp.logical_and(action >= Action.MOVE_N, action <= Action.MOVE_SW)
    state = lax.cond(
        is_move,
        lambda s: do_move(_rng_action, s, action, params, static_params),
        lambda s: s,
        state,
    )

    # Stairs
    state = lax.cond(
        action == Action.GO_UP,
        lambda s: do_go_up(_rng_action, s, params, static_params),
        lambda s: s,
        state,
    )
    state = lax.cond(
        action == Action.GO_DOWN,
        lambda s: do_go_down(_rng_action, s, params, static_params),
        lambda s: s,
        state,
    )

    # Doors
    state = lax.cond(
        action == Action.OPEN_DOOR,
        lambda s: do_open_door(_rng_action, s, s.player_direction, params, static_params),
        lambda s: s,
        state,
    )
    state = lax.cond(
        action == Action.CLOSE_DOOR,
        lambda s: do_close_door(_rng_action, s, s.player_direction, params, static_params),
        lambda s: s,
        state,
    )

    # Search
    state = lax.cond(
        action == Action.SEARCH,
        lambda s: do_search(_rng_action, s, params, static_params),
        lambda s: s,
        state,
    )

    # Pickup
    state = lax.cond(
        action == Action.PICKUP,
        lambda s: do_pickup(_rng_action, s, params, static_params),
        lambda s: s,
        state,
    )

    # Eat
    state = lax.cond(
        action == Action.EAT,
        lambda s: do_eat(_rng_action, s, 0, params, static_params),
        lambda s: s,
        state,
    )

    # Wield
    state = lax.cond(
        action == Action.WIELD,
        lambda s: do_wield(_rng_action, s, 0, params, static_params),
        lambda s: s,
        state,
    )

    # Wear
    state = lax.cond(
        action == Action.WEAR,
        lambda s: do_wear(_rng_action, s, 0, params, static_params),
        lambda s: s,
        state,
    )

    # ---- 2. Monster updates ----
    state = update_monsters(_rng_monsters, state, params, static_params)
    state = do_monster_attacks(_rng_monsters, state, params, static_params)
    state = spawn_monsters(_rng_spawn, state, params, static_params)

    # ---- 3. Intrinsics / status ----
    state = update_hunger(state, params)
    state = update_status_effects(state)
    state = update_hp_regen(state)

    # ---- 4. Field of view ----
    state = compute_fov(state, static_params)

    # ---- 5. Achievements and reward ----
    old_achievements = state.achievements
    state = update_achievements(state, params, static_params)
    new_achievements = state.achievements

    # Reward = sum of newly earned achievement weights
    newly_earned = jnp.logical_and(new_achievements, jnp.logical_not(old_achievements))
    reward = jnp.sum(newly_earned * ACHIEVEMENT_REWARD_WEIGHTS)

    return state, reward
