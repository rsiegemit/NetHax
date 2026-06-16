"""World generator for Minihax-QuestMedium-v0.

From quest_medium.des: 37x5 map with left room, corridor, large room with lava.
Cold items at player start, 6 giant rats near the lava, stair in goal room.

.des coordinate convention: (col, row). Our code: (row, col).

Map layout (37 cols x 5 rows):
-----       -------------------------
|....##     |...................L...|
|...| #     |...................L...|
|...| ######....................L...|
-----       -------------------------
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, MonsterType, MONSTER_MAX_HP, RoleType, RaceType,
)
from Nethax.minihax.states import (
    HazardState, HazardStaticParams, Inventory, SimpleMonsters, GroundItems,
)
from Nethax.minihax.primitives.visibility import compute_visible, compute_lit_map
from Nethax.minihax.primitives.leveling import compute_initial_stats

_ACTIVE_H = 5
_ACTIVE_W = 37


def _build_quest_medium_map():
    """Build the quest_medium map from the .des layout."""
    game_map = jnp.full((_ACTIVE_H, _ACTIVE_W), TileType.VOID, dtype=jnp.int32)

    rows = jnp.arange(_ACTIVE_H)[:, None]
    cols = jnp.arange(_ACTIVE_W)[None, :]

    # Left room: 5x5 box (rows 0-4, cols 0-4)
    left_top = (rows == 0) & (cols >= 0) & (cols <= 4)
    left_bot = (rows == 4) & (cols >= 0) & (cols <= 4)
    left_lwall = (cols == 0) & (rows >= 1) & (rows <= 3)
    left_rwall = (cols == 4) & (rows >= 1) & (rows <= 2)  # opening at row 3
    left_floor = (rows >= 1) & (rows <= 3) & (cols >= 1) & (cols <= 3)

    # Corridor from left room to right section
    # Row 1: cols 5-6 are ## (corridor)
    # Row 2: nothing (wall at col 4 continues)
    # Row 3: cols 5-6 are ## then cols 7-12 is # path
    corr_r1 = (rows == 1) & (cols >= 5) & (cols <= 6)
    corr_r2 = (rows == 2) & (cols == 6)  # # at col 6
    corr_r3 = (rows == 3) & (cols >= 6) & (cols <= 12)

    # Right section: rows 0-4, cols 12-36
    right_top = (rows == 0) & (cols >= 12) & (cols <= 36)
    right_bot = (rows == 4) & (cols >= 12) & (cols <= 36)
    right_lwall = (cols == 12) & (rows >= 1) & (rows <= 3)
    right_rwall = (cols == 36) & (rows >= 1) & (rows <= 3)
    right_floor = (rows >= 1) & (rows <= 3) & (cols >= 13) & (cols <= 35)

    # Opening from corridor into right section at row 3, col 12
    corr_entry = (rows == 3) & (cols == 12)

    # Lava column at col 33, rows 1-3
    lava = (cols == 33) & (rows >= 1) & (rows <= 3)

    # Build
    game_map = jnp.where(left_top | left_bot | right_top | right_bot, TileType.HWALL, game_map)
    game_map = jnp.where(left_lwall | left_rwall | right_lwall | right_rwall, TileType.VWALL, game_map)
    game_map = jnp.where(left_floor | right_floor, TileType.FLOOR, game_map)
    game_map = jnp.where(corr_r1 | corr_r2 | corr_r3 | corr_entry, TileType.CORRIDOR, game_map)
    game_map = jnp.where(lava, TileType.LAVA, game_map)

    # Closed door at the corridor-to-goal-room entry (col 12, row 3) — gates
    # progression from corridor into the lava room.
    game_map = game_map.at[3, 12].set(TileType.DOOR_CLOSED)

    return game_map


def generate_quest_medium(rng, params, static_params):
    """Generate a QuestMedium environment state."""
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng, rng_stair, rng_stats = jax.random.split(rng, 3)

    player_stats = compute_initial_stats(rng_stats, RoleType.MONK, RaceType.HUMAN)

    game_map = _build_quest_medium_map()

    # Pad to static dimensions
    padded_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)
    padded_map = padded_map.at[:_ACTIVE_H, :_ACTIVE_W].set(game_map)

    # Player at (row 2, col 2) — .des BRANCH:(2,2,2,2)
    player_pos = jnp.array([2, 2], dtype=jnp.int32)

    # Stair in goal room: rows 1-3, cols 34-35
    # .des: $goal_room = selection:fillrect (33,1,35,3) -> (row 1-3, col 33-35)
    stair_r = jax.random.randint(rng_stair, (), 1, 4)
    rng, rng_sc = jax.random.split(rng)
    stair_c = jax.random.randint(rng_sc, (), 34, 36)
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)
    padded_map = padded_map.at[stair_pos[0], stair_pos[1]].set(TileType.DOWNSTAIR)

    # Items: wand of cold and frost horn at (row 2, col 2)
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([2, 2]))
    gi_positions = gi_positions.at[1].set(jnp.array([2, 2]))
    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(ItemType.WAND_COLD)
    gi_types = gi_types.at[1].set(ItemType.FROST_HORN)
    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)
    gi_mask = gi_mask.at[1].set(True)

    # 6 giant rats at specific positions
    # .des: (30,1),(30,2),(30,3),(31,4),(31,2),(31,3)
    # -> (row 1,col 30),(row 2,col 30),(row 3,col 30),(row 3,col 31),(row 2,col 31),(row 3,col 31)
    # Note: (31,4) would be row 4 which is a wall. Use row 3 instead.
    rat_positions = jnp.array([
        [1, 30], [2, 30], [3, 30],
        [3, 31], [2, 31], [1, 31],
    ], dtype=jnp.int32)

    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_types = jnp.full(max_m, MonsterType.NONE, dtype=jnp.int32)
    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)

    rat_hp = MONSTER_MAX_HP[MonsterType.GIANT_RAT]

    # Place up to 6 rats (or max_m, whichever is smaller)
    def place_rat(carry, i):
        positions, types, health, mask = carry
        active = i < 6
        positions = positions.at[i].set(jnp.where(active, rat_positions[jnp.minimum(i, 5)], positions[i]))
        types = types.at[i].set(jnp.where(active, MonsterType.GIANT_RAT, types[i]))
        health = health.at[i].set(jnp.where(active, rat_hp, health[i]))
        mask = mask.at[i].set(jnp.where(active, True, mask[i]))
        return (positions, types, health, mask), None

    (mon_positions, mon_types, mon_health, mon_mask), _ = jax.lax.scan(
        place_rat, (mon_positions, mon_types, mon_health, mon_mask), jnp.arange(max_m)
    )

    monsters = SimpleMonsters(
        position=mon_positions,
        type_id=mon_types,
        health=mon_health,
        mask=mon_mask,
    )

    inv = Inventory(
        item_ids=jnp.zeros(max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
    )

    ground_items = GroundItems(
        position=gi_positions,
        type_id=gi_types,
        mask=gi_mask,
    )

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, map_h, map_w, lit_map)
    return HazardState(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        player_stats=player_stats,
        player_levitating=False,
        levitation_turns=0,
        inventory=inv,
        monsters=monsters,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        timestep=0,
        prev_action=0,
        terminal=False,
        state_rng=rng,
    )
