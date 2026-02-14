"""World generator for Minihax-Quest-v0.

From quest.des: 3 rooms connected by corridors. ~66 wide x 7 tall.
- Room 1: rows 0-6, cols 0-12. Lava column at col 6.
  Player starts in left_bank (rows 1-5, cols 1-5).
  Skeleton key on right_bank (rows 1-5, cols 7-11).
- Room 2: rows 1-5, cols 16-22. Chest with wand of death.
- Room 3: rows 2-4, cols 40-65. Minotaur (asleep) + downstair.
- Corridors connecting rooms.
- 50% levitation item or 50% cold item placed in left_bank.

.des coordinates are (col, row) -- converted to (row, col) internally.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, MonsterType, MONSTER_MAX_HP,
)
from Nethax.minihax.primitives.visibility import compute_visible
from Nethax.minihax.world_gen.combat_common import (
    pad_map, empty_combat_state,
)


# Active map dimensions
_MAP_H = 7
_MAP_W = 66


def _make_quest_map():
    """Build the quest map with 3 rooms and corridors."""
    game_map = jnp.full((_MAP_H, _MAP_W), TileType.VOID, dtype=jnp.int32)

    rows = jnp.arange(_MAP_H)[:, None]
    cols = jnp.arange(_MAP_W)[None, :]

    # Room 1: rows 0-6, cols 0-12 (walls on border, floor inside)
    r1_wall_top = (rows == 0) & (cols >= 0) & (cols <= 12)
    r1_wall_bot = (rows == 6) & (cols >= 0) & (cols <= 12)
    r1_wall_l = (cols == 0) & (rows >= 1) & (rows <= 5)
    r1_wall_r = (cols == 12) & (rows >= 1) & (rows <= 5)
    r1_floor = (rows >= 1) & (rows <= 5) & (cols >= 1) & (cols <= 11)
    r1_lava = (cols == 6) & (rows >= 1) & (rows <= 5)

    game_map = jnp.where(r1_wall_top | r1_wall_bot, TileType.HWALL, game_map)
    game_map = jnp.where(r1_wall_l | r1_wall_r, TileType.VWALL, game_map)
    game_map = jnp.where(r1_floor, TileType.FLOOR, game_map)
    game_map = jnp.where(r1_lava, TileType.LAVA, game_map)

    # Room 2: rows 1-5, cols 16-22
    r2_wall_top = (rows == 1) & (cols >= 16) & (cols <= 22)
    r2_wall_bot = (rows == 5) & (cols >= 16) & (cols <= 22)
    r2_wall_l = (cols == 16) & (rows >= 2) & (rows <= 4)
    r2_wall_r = (cols == 22) & (rows >= 2) & (rows <= 4)
    r2_floor = (rows >= 2) & (rows <= 4) & (cols >= 17) & (cols <= 21)

    game_map = jnp.where(r2_wall_top | r2_wall_bot, TileType.HWALL, game_map)
    game_map = jnp.where(r2_wall_l | r2_wall_r, TileType.VWALL, game_map)
    game_map = jnp.where(r2_floor, TileType.FLOOR, game_map)

    # Room 3: rows 2-4, cols 40-65
    r3_wall_top = (rows == 2) & (cols >= 40) & (cols <= 65)
    r3_wall_bot = (rows == 4) & (cols >= 40) & (cols <= 65)
    r3_wall_l = (cols == 40) & (rows == 3)
    r3_wall_r = (cols == 65) & (rows == 3)
    r3_floor = (rows == 3) & (cols >= 41) & (cols <= 64)

    game_map = jnp.where(r3_wall_top | r3_wall_bot, TileType.HWALL, game_map)
    game_map = jnp.where(r3_wall_l | r3_wall_r, TileType.VWALL, game_map)
    game_map = jnp.where(r3_floor, TileType.FLOOR, game_map)

    # Corridors: Room 1 -> Room 2 (row=3, cols 13-15)
    c1 = (rows == 3) & (cols >= 13) & (cols <= 15)
    game_map = jnp.where(c1, TileType.CORRIDOR, game_map)

    # Corridors: Room 2 -> Room 3 (row=3, cols 23-39)
    c2 = (rows == 3) & (cols >= 23) & (cols <= 39)
    game_map = jnp.where(c2, TileType.CORRIDOR, game_map)

    return game_map


def generate_quest(rng, params, static_params):
    """Generate a Quest environment state."""
    rng, rng_player_r, rng_player_c, rng_item_branch, rng_item_sub, rng_item_sub2 = \
        jax.random.split(rng, 6)
    rng, rng_key_r, rng_key_c, rng_state = jax.random.split(rng, 4)

    game_map = _make_quest_map()

    # Player starts in left_bank: rows 1-5, cols 1-5
    player_r = jax.random.randint(rng_player_r, (), 1, 6)
    player_c = jax.random.randint(rng_player_c, (), 1, 6)
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    # Downstair at (row=3, col=64)
    stair_pos = jnp.array([3, 64], dtype=jnp.int32)
    game_map = game_map.at[3, 64].set(TileType.DOWNSTAIR)

    # Pad map
    padded_map = pad_map(game_map, static_params)

    # Skeleton key on right_bank (rows 1-5, cols 7-11)
    key_r = jax.random.randint(rng_key_r, (), 1, 6)
    key_c = jax.random.randint(rng_key_c, (), 7, 12)

    # Random item: 50% levitation, 50% cold
    branch = jax.random.uniform(rng_item_branch, ()) < 0.5
    sub_roll = jax.random.uniform(rng_item_sub, ())
    sub_roll2 = jax.random.uniform(rng_item_sub2, ())

    lev_item = jnp.where(sub_roll < 0.33, ItemType.POTION_LEVITATION,
               jnp.where(sub_roll < 0.66, ItemType.RING_LEVITATION,
                         ItemType.BOOTS_LEVITATION))
    cold_item = jnp.where(sub_roll2 < 0.5, ItemType.WAND_COLD, ItemType.FROST_HORN)
    rand_item = jnp.where(branch, lev_item, cold_item)

    # Place random item in left_bank
    rng, rng_ri_r, rng_ri_c = jax.random.split(rng, 3)
    ri_r = jax.random.randint(rng_ri_r, (), 1, 6)
    ri_c = jax.random.randint(rng_ri_c, (), 1, 6)

    # Wand of death in entry_room (rows 3-6, cols 3-6 per quest_hard.des)
    rng, rng_wd_r, rng_wd_c = jax.random.split(rng, 3)
    wd_r = jax.random.randint(rng_wd_r, (), 3, 7)
    wd_c = jax.random.randint(rng_wd_c, (), 3, 7)

    # Ground items: [0] = skeleton key, [1] = random item, [2] = wand_death
    max_gi = static_params.max_ground_items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([key_r, key_c]))
    gi_positions = gi_positions.at[1].set(jnp.array([ri_r, ri_c]))
    gi_positions = gi_positions.at[2].set(jnp.array([wd_r, wd_c]))

    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(ItemType.SKELETON_KEY)
    gi_types = gi_types.at[1].set(rand_item)
    gi_types = gi_types.at[2].set(ItemType.WAND_DEATH)

    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)
    gi_mask = gi_mask.at[1].set(True)
    gi_mask = gi_mask.at[2].set(True)

    # Minotaur at (row=3, col=63), asleep
    max_m = static_params.max_monsters
    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(jnp.array([3, 63]))

    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(MonsterType.MINOTAUR)

    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[MonsterType.MINOTAUR])

    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)

    mon_sleeping = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_sleeping = mon_sleeping.at[0].set(True)  # Minotaur starts asleep

    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        monsters=state.monsters.replace(
            position=mon_positions,
            type_id=mon_types,
            health=mon_health,
            mask=mon_mask,
            is_sleeping=mon_sleeping,
        ),
        ground_items=state.ground_items.replace(
            position=gi_positions,
            type_id=gi_types,
            mask=gi_mask,
        ),
    )
    return state
