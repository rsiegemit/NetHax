"""World generator for Minihax-QuestHard-v0.

Baalzebub-inspired quest_hard.des: MAZE level with corrmaze flag.
- Fixed MAP (13 rows x 49 cols) placed on the right side, centered vertically.
- Procedural maze generated via MAZEWALK fills all void areas (left, above,
  below, and right-side voids around the MAP).
- Entry room at MAP rows 4-8, cols 3-6. Items placed here.
- Closed doors at MAP (row=6, col=0) and MAP (row=6, col=10).
- Minotaur at MAP (row=6, col=34) asleep.
- Downstair at MAP (row=6, col=44).
- 50% lev/cold item + wand of death in entry room.

.des coordinates are (col, row) -- converted to (row, col) internally.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, MonsterType, MONSTER_MAX_HP,
)
from Nethax.minihax.primitives.visibility import compute_visible
from Nethax.minihax.world_gen.combat_common import (
    pad_map, empty_combat_state, parse_map,
)
from Nethax.minihax.world_gen.procedural import mazewalk, wall_cleanup


# The quest_hard MAP from the .des file (13 rows x 49 cols)
# Exact transcription of the MAZE MAP section from quest_hard.des
_MAP_LINES = [
    "-------------------------------------------------",
    "|                   ----               ----",
    "|          ----     |     -----------  |",
    "| ------      |  ---------|.........|--|",
    "| |....|  -------|.LLLLL..-----------------",
    "---....|--|........LLLLL..................|----",
    "+.........+........LLLLL......................|",
    "---....|--|........LLLLL..................|----",
    "| |....|  -------|.LLLLL..-----------------",
    "| ------      |  ---------|.........|--|",
    "|          ----     |     -----------  |",
    "|                   ----               ----",
    "-------------------------------------------------",
]

# MAP placement offsets in the full level
# GEOMETRY:right,center with 80-col, 21-row level and 13x49 MAP
_MAP_COL_OFFSET = 31  # 80 - 49 = 31
_MAP_ROW_OFFSET = 4   # (21 - 13) // 2 = 4


def generate_quest_hard(rng, params, static_params):
    """Generate a QuestHard environment state."""
    rng, rng_maze, rng_player, rng_item_branch, rng_item_sub, rng_item_sub2, rng_state = \
        jax.random.split(rng, 7)

    # Parse the fixed MAP (13 rows x 49 cols)
    fixed_map = parse_map(_MAP_LINES)
    fixed_h, fixed_w = fixed_map.shape  # 13, 49

    # Full level dimensions from static params (21 rows x 80 cols)
    level_h = static_params.map_height  # 21
    level_w = static_params.map_width   # 80

    # Start with VOID
    game_map = jnp.full((level_h, level_w), TileType.VOID, dtype=jnp.int32)

    # Place fixed MAP centered: rows 4-16, cols 31-79
    game_map = game_map.at[
        _MAP_ROW_OFFSET:_MAP_ROW_OFFSET + fixed_h,
        _MAP_COL_OFFSET:_MAP_COL_OFFSET + fixed_w
    ].set(fixed_map)

    # Full-level border walls
    # Top and bottom borders (HWALL)
    game_map = game_map.at[0, :].set(TileType.HWALL)
    game_map = game_map.at[level_h - 1, :].set(TileType.HWALL)
    # Left border (VWALL) -- no right border since MAP extends to col 79
    game_map = game_map.at[1:level_h - 1, 0].set(TileType.VWALL)

    # Generate maze via MAZEWALK filling all void areas
    # .des: MAZEWALK:(00,06),west -- start at MAP (col=0, row=6) going west
    # MAP (row=6, col=0) = level (row=10, col=31) is a DOOR_CLOSED
    # Start maze one tile to the left at level (10, 30)
    maze_start = jnp.array([_MAP_ROW_OFFSET + 6, _MAP_COL_OFFSET - 1], dtype=jnp.int32)
    game_map = mazewalk(rng_maze, game_map, maze_start, level_h, level_w)

    # Force west corridor from maze start to match original MAZEWALK:(00,06),west
    # The DFS picks random directions; original NetHack forces first step west.
    # Carve the midpoint to guarantee connection from door to maze body.
    game_map = game_map.at[_MAP_ROW_OFFSET + 6, _MAP_COL_OFFSET - 2].set(TileType.CORRIDOR)

    # Remove orphan walls (walls not adjacent to any walkable tile)
    game_map = wall_cleanup(game_map, level_h, level_w)

    # Re-stamp the fixed MAP to restore structural walls that wall_cleanup may have removed
    game_map = game_map.at[
        _MAP_ROW_OFFSET:_MAP_ROW_OFFSET + fixed_h,
        _MAP_COL_OFFSET:_MAP_COL_OFFSET + fixed_w
    ].set(fixed_map)

    # Explicitly place closed doors
    # .des: DOOR:closed,(00,06) and DOOR:closed,(10,06)
    door_row = _MAP_ROW_OFFSET + 6  # level row 10
    game_map = game_map.at[door_row, _MAP_COL_OFFSET + 0].set(TileType.DOOR_CLOSED)
    game_map = game_map.at[door_row, _MAP_COL_OFFSET + 10].set(TileType.DOOR_CLOSED)

    # Player starts in the maze area (left region)
    # .des: STAIR:levregion(01,00,15,20),...,up -> upstair in cols 1-15 of full level
    # Find a random CORRIDOR tile in the maze region for player start
    flat = game_map.reshape(-1)
    indices = jnp.arange(flat.shape[0])
    rows = indices // level_w
    cols = indices % level_w
    is_corridor = (flat == TileType.CORRIDOR)
    in_maze_region = (cols >= 1) & (cols <= 15) & (rows >= 1) & (rows <= level_h - 2)
    valid = is_corridor & in_maze_region
    n_valid = valid.sum()
    cumsum = jnp.cumsum(valid)
    chosen_idx = jax.random.randint(rng_player, (), 0, jnp.maximum(n_valid, 1))
    match = (cumsum == chosen_idx + 1) & valid
    flat_idx = jnp.argmax(match)
    player_pos = jnp.array([flat_idx // level_w, flat_idx % level_w], dtype=jnp.int32)

    # Downstair at MAP (row=6, col=44) -> level (10, 75)
    # .des: STAIR:(44,06),down
    stair_pos = jnp.array([_MAP_ROW_OFFSET + 6, _MAP_COL_OFFSET + 44], dtype=jnp.int32)
    game_map = game_map.at[_MAP_ROW_OFFSET + 6, _MAP_COL_OFFSET + 44].set(TileType.DOWNSTAIR)

    # Pad map to static params dimensions (should be identity since we use full level_h x level_w)
    padded_map = pad_map(game_map, static_params)

    # Random item: 50% levitation, 50% cold
    # .des: $entry_room = selection:fillrect (3,4,6,8) -> MAP cols 3-6, rows 4-8
    branch = jax.random.uniform(rng_item_branch, ()) < 0.5
    sub_roll = jax.random.uniform(rng_item_sub, ())
    sub_roll2 = jax.random.uniform(rng_item_sub2, ())

    lev_item = jnp.where(sub_roll < 0.33, ItemType.POTION_LEVITATION,
               jnp.where(sub_roll < 0.66, ItemType.RING_LEVITATION,
                         ItemType.BOOTS_LEVITATION))
    cold_item = jnp.where(sub_roll2 < 0.5, ItemType.WAND_COLD, ItemType.FROST_HORN)
    rand_item = jnp.where(branch, lev_item, cold_item)

    # Ground items in entry room
    # MAP (5,5) and (6,5) -> level (9, 36) and (10, 36)
    max_gi = static_params.max_ground_items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([_MAP_ROW_OFFSET + 5, _MAP_COL_OFFSET + 5]))
    gi_positions = gi_positions.at[1].set(jnp.array([_MAP_ROW_OFFSET + 6, _MAP_COL_OFFSET + 5]))

    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(rand_item)
    gi_types = gi_types.at[1].set(ItemType.WAND_DEATH)

    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)
    gi_mask = gi_mask.at[1].set(True)

    # Minotaur at MAP (row=6, col=34) -> level (10, 65), asleep
    # .des: MONSTER:('H',"Minotaur"),(34,06),asleep
    max_m = static_params.max_monsters
    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(jnp.array([_MAP_ROW_OFFSET + 6, _MAP_COL_OFFSET + 34]))

    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(MonsterType.MINOTAUR)

    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[MonsterType.MINOTAUR])

    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)

    mon_sleeping = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_sleeping = mon_sleeping.at[0].set(True)

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
