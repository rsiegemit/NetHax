import jax
import jax.numpy as jnp
from jax import lax

from nethax.nethax.constants import *
from nethax.nethax.nethax_state import EnvState, EnvParams, StaticEnvParams, Item, Monsters


def generate_room(rng, map_level, map_size, room_id):
    """Generate a single room on a dungeon level.

    Returns the updated map and the room bounds (top, left, height, width).
    """
    rng, r1, r2, r3, r4 = jax.random.split(rng, 5)

    # Room dimensions: 4-10 tiles
    room_h = jax.random.randint(r1, (), 4, 10)
    room_w = jax.random.randint(r2, (), 4, 10)

    # Room position (with margin from edges)
    room_top = jax.random.randint(r3, (), 1, map_size[0] - room_h - 1)
    room_left = jax.random.randint(r4, (), 1, map_size[1] - room_w - 1)

    # Fill room interior with floor
    rows = jnp.arange(map_size[0])
    cols = jnp.arange(map_size[1])
    row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')

    in_room = jnp.logical_and(
        jnp.logical_and(row_grid >= room_top, row_grid < room_top + room_h),
        jnp.logical_and(col_grid >= room_left, col_grid < room_left + room_w),
    )

    map_level = jnp.where(in_room, TileType.FLOOR, map_level)

    return map_level, jnp.array([room_top, room_left, room_h, room_w])


def connect_rooms(map_level, room_a, room_b, map_size):
    """Connect two rooms with an L-shaped corridor."""
    # Center of each room
    center_a = jnp.array([room_a[0] + room_a[2] // 2, room_a[1] + room_a[3] // 2])
    center_b = jnp.array([room_b[0] + room_b[2] // 2, room_b[1] + room_b[3] // 2])

    rows = jnp.arange(map_size[0])
    cols = jnp.arange(map_size[1])
    row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')

    # Horizontal segment from a to b at a's row
    min_col = jnp.minimum(center_a[1], center_b[1])
    max_col = jnp.maximum(center_a[1], center_b[1])
    h_corridor = jnp.logical_and(
        row_grid == center_a[0],
        jnp.logical_and(col_grid >= min_col, col_grid <= max_col),
    )

    # Vertical segment from a's row to b's row at b's col
    min_row = jnp.minimum(center_a[0], center_b[0])
    max_row = jnp.maximum(center_a[0], center_b[0])
    v_corridor = jnp.logical_and(
        col_grid == center_b[1],
        jnp.logical_and(row_grid >= min_row, row_grid <= max_row),
    )

    corridor = jnp.logical_or(h_corridor, v_corridor)

    # Only carve corridor where there's currently wall/void
    is_wall = jnp.logical_or(
        map_level == TileType.WALL,
        map_level == TileType.VOID,
    )
    should_carve = jnp.logical_and(corridor, is_wall)

    map_level = jnp.where(should_carve, TileType.CORRIDOR, map_level)

    return map_level


def add_walls(map_level, map_size):
    """Add walls around all floor/corridor tiles that border void."""
    rows = jnp.arange(map_size[0])
    cols = jnp.arange(map_size[1])
    row_grid, col_grid = jnp.meshgrid(rows, cols, indexing='ij')

    is_void = map_level == TileType.VOID

    # Check all 8 neighbors for floor/corridor
    has_floor_neighbor = jnp.zeros_like(is_void)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = jnp.roll(jnp.roll(map_level, -dr, axis=0), -dc, axis=1)
            is_floor_or_corridor = jnp.logical_or(
                shifted == TileType.FLOOR,
                shifted == TileType.CORRIDOR,
            )
            has_floor_neighbor = jnp.logical_or(has_floor_neighbor, is_floor_or_corridor)

    should_wall = jnp.logical_and(is_void, has_floor_neighbor)
    map_level = jnp.where(should_wall, TileType.WALL, map_level)

    return map_level


def place_doors(rng, map_level, map_size):
    """Place closed doors where corridors transition to rooms.

    A door candidate is a CORRIDOR tile with walls on two opposite sides
    and passable tiles (FLOOR or CORRIDOR) on the other two sides.
    Doors are placed at ~50% of candidates.
    """
    is_corridor = map_level == TileType.CORRIDOR

    north = jnp.roll(map_level, 1, axis=0)
    south = jnp.roll(map_level, -1, axis=0)
    east = jnp.roll(map_level, -1, axis=1)
    west = jnp.roll(map_level, 1, axis=1)

    def is_passable(tile):
        return jnp.logical_or(tile == TileType.FLOOR, tile == TileType.CORRIDOR)

    # Pattern 1: walls N+S, passable E+W
    pattern1 = jnp.logical_and(
        jnp.logical_and(north == TileType.WALL, south == TileType.WALL),
        jnp.logical_and(is_passable(east), is_passable(west))
    )
    # Pattern 2: walls E+W, passable N+S
    pattern2 = jnp.logical_and(
        jnp.logical_and(east == TileType.WALL, west == TileType.WALL),
        jnp.logical_and(is_passable(north), is_passable(south))
    )

    is_door = jnp.logical_and(is_corridor, jnp.logical_or(pattern1, pattern2))

    rng, _rng = jax.random.split(rng)
    door_mask = jnp.logical_and(is_door, jax.random.bernoulli(_rng, 0.5, map_level.shape))

    return jnp.where(door_mask, TileType.CLOSED_DOOR, map_level)


def place_initial_monsters(rng, map_level, level_idx, max_monsters, map_size):
    """Place initial monsters on a dungeon level.

    Spawns 3-6 monsters on random floor tiles.
    Returns: (positions, type_ids, healths, masks, asleep) arrays.
    """
    rng, _rng_count = jax.random.split(rng)
    num_to_spawn = jax.random.randint(_rng_count, (), 3, 7)  # 3-6 monsters

    # Valid monster types for this depth
    depth = level_idx + 1
    valid_monster = jnp.logical_and(
        MONSTER_SPAWN_FLOORS[:, 0] <= depth,
        MONSTER_SPAWN_FLOORS[:, 1] >= depth
    )
    valid_monster = valid_monster.at[0].set(False)  # Exclude NONE

    # Floor tiles for spawning
    floor_mask = jnp.logical_or(map_level == TileType.FLOOR, map_level == TileType.CORRIDOR)
    flat_floor = floor_mask.reshape(-1).astype(jnp.float32)
    flat_floor = flat_floor / jnp.maximum(flat_floor.sum(), 1.0)

    mon_probs = valid_monster.astype(jnp.float32)
    mon_probs = mon_probs / jnp.maximum(mon_probs.sum(), 1.0)

    positions = jnp.zeros((max_monsters, 2), dtype=jnp.int32)
    type_ids = jnp.zeros(max_monsters, dtype=jnp.int32)
    healths = jnp.zeros(max_monsters, dtype=jnp.int32)
    masks = jnp.zeros(max_monsters, dtype=jnp.bool_)
    asleep = jnp.zeros(max_monsters, dtype=jnp.bool_)

    def spawn_one(carry, i):
        rng, positions, type_ids, healths, masks, asleep = carry
        should_spawn = i < num_to_spawn

        rng, _rng_type, _rng_pos = jax.random.split(rng, 3)
        mon_type = jax.random.choice(_rng_type, jnp.arange(NUM_MONSTER_TYPES), p=mon_probs)
        flat_idx = jax.random.choice(_rng_pos, jnp.arange(map_size[0] * map_size[1]), p=flat_floor)
        pos = jnp.array([flat_idx // map_size[1], flat_idx % map_size[1]])

        hp = MONSTER_MAX_HP[mon_type]

        positions = positions.at[i].set(jnp.where(should_spawn, pos, positions[i]))
        type_ids = type_ids.at[i].set(jnp.where(should_spawn, mon_type, type_ids[i]))
        healths = healths.at[i].set(jnp.where(should_spawn, hp, healths[i]))
        masks = masks.at[i].set(jnp.where(should_spawn, True, masks[i]))
        asleep = asleep.at[i].set(jnp.where(should_spawn, True, asleep[i]))

        return (rng, positions, type_ids, healths, masks, asleep), None

    (_, positions, type_ids, healths, masks, asleep), _ = jax.lax.scan(
        spawn_one, (rng, positions, type_ids, healths, masks, asleep), jnp.arange(max_monsters)
    )

    return positions, type_ids, healths, masks, asleep


def place_ground_items(rng, map_level, level_idx, map_size, max_items_per_tile):
    """Place random ground items on a dungeon level.

    Places 2-4 items on random floor tiles.
    Returns: (categories, type_ids, quantities) arrays for ground_items.
    """
    map_h, map_w = map_size
    rng, _rng_count = jax.random.split(rng)
    num_items = jax.random.randint(_rng_count, (), 2, 5)  # 2-4 items

    floor_mask = jnp.logical_or(map_level == TileType.FLOOR, map_level == TileType.CORRIDOR)
    flat_floor = floor_mask.reshape(-1).astype(jnp.float32)
    flat_floor = flat_floor / jnp.maximum(flat_floor.sum(), 1.0)

    categories = jnp.zeros((map_h, map_w, max_items_per_tile), dtype=jnp.int32)
    type_ids = jnp.zeros((map_h, map_w, max_items_per_tile), dtype=jnp.int32)
    quantities = jnp.zeros((map_h, map_w, max_items_per_tile), dtype=jnp.int32)

    max_placeable = 8  # Fixed scan size

    def place_one(carry, i):
        rng, categories, type_ids, quantities = carry
        should_place = i < num_items

        rng, _rng_cat, _rng_type, _rng_pos, _rng_gold = jax.random.split(rng, 5)

        # Random category
        cat_roll = jax.random.uniform(_rng_cat)
        cat = jnp.where(cat_roll < 0.40, ItemCategory.FOOD,
              jnp.where(cat_roll < 0.60, ItemCategory.WEAPON,
              jnp.where(cat_roll < 0.75, ItemCategory.ARMOR,
                                         ItemCategory.GOLD)))

        # Type within category
        food_types = jnp.array([0, 1, 2, 3])  # FOOD_RATION, CRAM, LEMBAS, APPLE
        weapon_types = jnp.array([1, 5, 7, 25])  # DAGGER, KNIFE, SHORT_SWORD, CLUB
        armor_types = jnp.array([19, 20, 24, 39])  # LEATHER, JACKET, HELMET, LOW_BOOTS

        type_roll = jax.random.randint(_rng_type, (), 0, 4)
        type_id = jnp.where(cat == ItemCategory.FOOD, food_types[type_roll],
                  jnp.where(cat == ItemCategory.WEAPON, weapon_types[type_roll],
                  jnp.where(cat == ItemCategory.ARMOR, armor_types[type_roll], 0)))

        quantity = jnp.where(cat == ItemCategory.GOLD,
                            jax.random.randint(_rng_gold, (), 5, 51), 1)

        # Pick floor tile
        flat_idx = jax.random.choice(_rng_pos, jnp.arange(map_h * map_w), p=flat_floor)
        row = flat_idx // map_w
        col = flat_idx % map_w

        # Place in slot 0
        categories = jnp.where(should_place,
            categories.at[row, col, 0].set(cat), categories)
        type_ids = jnp.where(should_place,
            type_ids.at[row, col, 0].set(type_id), type_ids)
        quantities = jnp.where(should_place,
            quantities.at[row, col, 0].set(quantity), quantities)

        return (rng, categories, type_ids, quantities), None

    (_, categories, type_ids, quantities), _ = jax.lax.scan(
        place_one, (rng, categories, type_ids, quantities), jnp.arange(max_placeable)
    )

    return categories, type_ids, quantities


def generate_dungeon_level(rng, level_idx, map_size, static_params):
    """Generate a single dungeon level with rooms and corridors."""
    rng, _rng = jax.random.split(rng)

    # Start with all void
    map_level = jnp.full(map_size, TileType.VOID, dtype=jnp.int32)

    # Generate 5-8 rooms
    num_rooms = 6  # Fixed for JAX compatibility
    max_rooms = 8

    room_bounds = jnp.zeros((max_rooms, 4), dtype=jnp.int32)

    # Generate rooms sequentially (lax.scan)
    def gen_room_step(carry, i):
        rng, map_level, room_bounds = carry
        rng, _rng = jax.random.split(rng)

        map_level_new, bounds = generate_room(_rng, map_level, map_size, i)

        # Only place room if index < num_rooms
        use_room = i < num_rooms
        map_level = jnp.where(use_room, map_level_new, map_level)
        room_bounds = room_bounds.at[i].set(
            jnp.where(use_room, bounds, room_bounds[i])
        )

        return (rng, map_level, room_bounds), None

    (rng, map_level, room_bounds), _ = lax.scan(
        gen_room_step,
        (rng, map_level, room_bounds),
        jnp.arange(max_rooms),
    )

    # Connect rooms sequentially
    def connect_step(carry, i):
        map_level = carry
        should_connect = i < num_rooms - 1
        map_new = connect_rooms(map_level, room_bounds[i], room_bounds[i + 1], map_size)
        map_level = jnp.where(should_connect, map_new, map_level)
        return map_level, None

    map_level, _ = lax.scan(
        connect_step,
        map_level,
        jnp.arange(max_rooms - 1),
    )

    # Add walls
    map_level = add_walls(map_level, map_size)

    # Place doors at room entrances
    rng, _rng_doors = jax.random.split(rng)
    map_level = place_doors(_rng_doors, map_level, map_size)

    # Place stairs
    rng, r1, r2 = jax.random.split(rng, 3)
    # Up stairs in first room, down stairs in last room
    up_pos = jnp.array([
        room_bounds[0, 0] + room_bounds[0, 2] // 2,
        room_bounds[0, 1] + room_bounds[0, 3] // 2,
    ])
    down_pos = jnp.array([
        room_bounds[num_rooms - 1, 0] + room_bounds[num_rooms - 1, 2] // 2,
        room_bounds[num_rooms - 1, 1] + room_bounds[num_rooms - 1, 3] // 2,
    ])

    # Don't place up stairs on level 0 (top of dungeon)
    has_up = level_idx > 0
    map_level = jnp.where(
        has_up,
        map_level.at[up_pos[0], up_pos[1]].set(TileType.STAIRCASE_UP),
        map_level,
    )
    map_level = map_level.at[down_pos[0], down_pos[1]].set(TileType.STAIRCASE_DOWN)

    # Monsters and items are populated in generate_world (post-step)

    return map_level, up_pos, down_pos


def generate_world(rng, params, static_params):
    """Generate the full dungeon and initialize game state."""
    map_size = static_params.map_size
    num_levels = static_params.num_levels

    # Generate all levels
    rngs = jax.random.split(rng, num_levels + 1)
    rng = rngs[0]
    level_rngs = rngs[1:]

    # Generate each level (vmap over levels)
    level_indices = jnp.arange(num_levels)

    def gen_level(rng_and_idx):
        level_rng, idx = rng_and_idx
        return generate_dungeon_level(level_rng, idx, map_size, static_params)

    results = jax.vmap(gen_level)((level_rngs, level_indices))
    all_maps, all_up_stairs, all_down_stairs = results

    # Initialize empty state arrays
    max_inv = static_params.max_inventory_size
    max_monsters = static_params.max_monsters
    max_items = static_params.max_items_per_tile

    # Empty inventory
    inventory = Item(
        category=jnp.zeros(max_inv, dtype=jnp.int32),
        type_id=jnp.zeros(max_inv, dtype=jnp.int32),
        buc_status=jnp.ones(max_inv, dtype=jnp.int32),  # Default uncursed
        enchantment=jnp.zeros(max_inv, dtype=jnp.int32),
        charges=jnp.zeros(max_inv, dtype=jnp.int32),
        identified=jnp.zeros(max_inv, dtype=jnp.bool_),
        quantity=jnp.zeros(max_inv, dtype=jnp.int32),
    )

    # Give starting equipment: a +0 short sword and +0 ring mail
    inventory = inventory.replace(
        category=inventory.category.at[0].set(ItemCategory.WEAPON),
        type_id=inventory.type_id.at[0].set(WeaponType.SHORT_SWORD),
        identified=inventory.identified.at[0].set(True),
        quantity=inventory.quantity.at[0].set(1),
    )
    inventory = inventory.replace(
        category=inventory.category.at[1].set(ItemCategory.ARMOR),
        type_id=inventory.type_id.at[1].set(ArmorType.RING_MAIL),
        identified=inventory.identified.at[1].set(True),
        quantity=inventory.quantity.at[1].set(1),
    )
    # Starting food rations
    inventory = inventory.replace(
        category=inventory.category.at[2].set(ItemCategory.FOOD),
        type_id=inventory.type_id.at[2].set(FoodType.FOOD_RATION),
        identified=inventory.identified.at[2].set(True),
        quantity=inventory.quantity.at[2].set(2),
    )

    # Empty monsters
    monsters = Monsters(
        position=jnp.zeros((num_levels, max_monsters, 2), dtype=jnp.int32),
        type_id=jnp.zeros((num_levels, max_monsters), dtype=jnp.int32),
        health=jnp.zeros((num_levels, max_monsters), dtype=jnp.int32),
        mask=jnp.zeros((num_levels, max_monsters), dtype=jnp.bool_),
        asleep=jnp.zeros((num_levels, max_monsters), dtype=jnp.bool_),
        confused=jnp.zeros((num_levels, max_monsters), dtype=jnp.int32),
        speed_counter=jnp.zeros((num_levels, max_monsters), dtype=jnp.int32),
    )

    # Empty ground items
    ground_items = Item(
        category=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
        type_id=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
        buc_status=jnp.ones((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
        enchantment=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
        charges=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
        identified=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.bool_),
        quantity=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
    )

    # Populate monsters for each level
    def populate_monsters_for_level(carry, level_idx):
        rng, monsters = carry
        rng, _rng = jax.random.split(rng)
        positions, type_ids, healths, masks, asleep_arr = place_initial_monsters(
            _rng, all_maps[level_idx], level_idx, max_monsters, map_size
        )
        monsters = monsters.replace(
            position=monsters.position.at[level_idx].set(positions),
            type_id=monsters.type_id.at[level_idx].set(type_ids),
            health=monsters.health.at[level_idx].set(healths),
            mask=monsters.mask.at[level_idx].set(masks),
            asleep=monsters.asleep.at[level_idx].set(asleep_arr),
        )
        return (rng, monsters), None

    rng, _rng_monsters = jax.random.split(rng)
    (_, monsters), _ = jax.lax.scan(
        populate_monsters_for_level, (_rng_monsters, monsters), jnp.arange(num_levels)
    )

    # Populate ground items for each level
    def populate_items_for_level(carry, level_idx):
        rng, ground_items = carry
        rng, _rng = jax.random.split(rng)
        cats, types, qtys = place_ground_items(
            _rng, all_maps[level_idx], level_idx, map_size, max_items
        )
        ground_items = ground_items.replace(
            category=ground_items.category.at[level_idx].set(cats),
            type_id=ground_items.type_id.at[level_idx].set(types),
            quantity=ground_items.quantity.at[level_idx].set(qtys),
        )
        return (rng, ground_items), None

    rng, _rng_items = jax.random.split(rng)
    (_, ground_items), _ = jax.lax.scan(
        populate_items_for_level, (_rng_items, ground_items), jnp.arange(num_levels)
    )

    # Randomize potion/scroll appearances
    rng, r1, r2 = jax.random.split(rng, 3)
    potion_appearance_map = jax.random.permutation(r1, NUM_POTION_TYPES)
    scroll_appearance_map = jax.random.permutation(r2, NUM_SCROLL_TYPES)

    # Player starts on level 0 at the up-stair position (or center of first room)
    player_start = all_up_stairs[0]
    # Level 0 has no up stair, so use down stair area
    player_start = jnp.where(
        all_up_stairs[0].sum() == 0,
        all_down_stairs[0] - jnp.array([1, 0]),  # Near down stair on level 0
        all_up_stairs[0],
    )

    state = EnvState(
        map=all_maps,
        item_map=jnp.zeros((num_levels, map_size[0], map_size[1], max_items), dtype=jnp.int32),
        explored=jnp.zeros((num_levels, map_size[0], map_size[1]), dtype=jnp.bool_),
        visible=jnp.zeros((num_levels, map_size[0], map_size[1]), dtype=jnp.bool_),
        down_stairs=all_down_stairs,
        up_stairs=all_up_stairs,
        player_position=player_start,
        player_level=0,
        player_direction=Action.MOVE_S,
        player_hp=16,
        player_max_hp=16,
        player_xp=0,
        player_xp_level=1,
        player_ac=3,  # Ring mail AC
        player_strength=16,
        player_nutrition=900,
        player_confused=0,
        player_blind=0,
        player_stunned=0,
        player_paralyzed=0,
        player_fast=0,
        player_invisible=0,
        player_intrinsics=jnp.zeros(NUM_INTRINSICS, dtype=jnp.bool_),
        inventory=inventory,
        wielded_weapon=0,  # Short sword at index 0
        worn_armor=jnp.array([1, -1, -1, -1, -1, -1], dtype=jnp.int32),  # Ring mail at index 1
        gold=0,
        ground_items=ground_items,
        monsters=monsters,
        potion_appearance_map=potion_appearance_map,
        scroll_appearance_map=scroll_appearance_map,
        potion_identified=jnp.zeros(NUM_POTION_TYPES, dtype=jnp.bool_),
        scroll_identified=jnp.zeros(NUM_SCROLL_TYPES, dtype=jnp.bool_),
        fountains_used=jnp.zeros((num_levels, map_size[0], map_size[1]), dtype=jnp.bool_),
        doors_state=jnp.zeros((num_levels, map_size[0], map_size[1]), dtype=jnp.int32),
        traps_revealed=jnp.zeros((num_levels, map_size[0], map_size[1]), dtype=jnp.bool_),
        monsters_killed=0,
        achievements=jnp.zeros(NUM_ACHIEVEMENTS, dtype=jnp.bool_),
        score=0,
        state_rng=rng,
        timestep=0,
    )

    return state
