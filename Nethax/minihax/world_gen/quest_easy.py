"""World generator for Minihax-QuestEasy-v0.

From quest_easy.des: 29x7 map with lava section, passage to goal room.
Cold items at (col 2, row 2), 2 random monsters, stair in goal room.

.des coordinate convention: (col, row). Our code: (row, col).

Map layout (29 cols x 7 rows):
-------------|
|.....L......|--      |-----|
|.....L........|------|.....|
|.....L.....................|
|.....L........|------|.....|
|.....L......|--      |-----|
-------------|
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

_ACTIVE_H = 7
_ACTIVE_W = 29

# Random monster pool for "random" monsters in .des
_RANDOM_MONSTERS = jnp.array([
    MonsterType.GIANT_RAT,
    MonsterType.COCKATRICE,
    MonsterType.NAGA_HATCHLING,
    MonsterType.OGRE,
], dtype=jnp.int32)


def _build_quest_easy_map():
    """Build the quest_easy map from the .des layout.

    .des MAP (29 cols x 7 rows):
    Row 0: -------------|               (cols 0-12 HWALL, col 13 VWALL, cols 14-28 VOID)
    Row 1: |.....L......|--      |-----| (VOID at cols 16-21)
    Row 2: |.....L........|------|.....|
    Row 3: |.....L.....................|  (open corridor cols 1-27)
    Row 4: |.....L........|------|.....|
    Row 5: |.....L......|--      |-----| (VOID at cols 16-21)
    Row 6: -------------|               (cols 0-12 HWALL, col 13 VWALL, cols 14-28 VOID)
    """
    game_map = jnp.full((_ACTIVE_H, _ACTIVE_W), TileType.VOID, dtype=jnp.int32)

    rows = jnp.arange(_ACTIVE_H)[:, None]
    cols = jnp.arange(_ACTIVE_W)[None, :]

    # --- Top/bottom borders (row 0 and 6): cols 0-12 HWALL, col 13 VWALL ---
    top_bot_hwall = ((rows == 0) | (rows == 6)) & (cols >= 0) & (cols <= 12)
    top_bot_vwall = ((rows == 0) | (rows == 6)) & (cols == 13)

    # --- Left wall: col 0, rows 1-5 ---
    left_wall = (cols == 0) & (rows >= 1) & (rows <= 5)

    # --- Left room right wall: col 13, rows 1 and 5 only (opening at rows 2-4) ---
    left_rwall = (cols == 13) & ((rows == 1) | (rows == 5))

    # --- Floor in left room: rows 1-5, cols 1-12 ---
    left_floor = (rows >= 1) & (rows <= 5) & (cols >= 1) & (cols <= 12)

    # --- Col 13, rows 2-4: FLOOR (opening in left room wall) ---
    opening_floor = (cols == 13) & (rows >= 2) & (rows <= 4)

    # --- Lava column: col 6, rows 1-5 ---
    lava = (cols == 6) & (rows >= 1) & (rows <= 5)

    # --- Passage walls rows 1,5: cols 14-15 HWALL ---
    passage_hwall = ((rows == 1) | (rows == 5)) & (cols >= 14) & (cols <= 15)

    # --- Passage inner walls: row 2,4 col 15 VWALL, cols 16-21 HWALL ---
    passage_inner_vwall = ((rows == 2) | (rows == 4)) & (cols == 15)
    passage_inner_hwall = ((rows == 2) | (rows == 4)) & (cols >= 16) & (cols <= 21)

    # --- Goal room left wall: col 22, rows 1,2,4,5 VWALL (opening at row 3) ---
    goal_lwall = (cols == 22) & ((rows == 1) | (rows == 2) | (rows == 4) | (rows == 5))

    # --- Goal room top/bottom walls: cols 23-27, rows 1 and 5 HWALL ---
    goal_hwall = ((rows == 1) | (rows == 5)) & (cols >= 23) & (cols <= 27)

    # --- Goal room right wall: col 28, rows 1-5 VWALL ---
    goal_rwall = (cols == 28) & (rows >= 1) & (rows <= 5)

    # --- Floor: passage rows 2-4 (cols 14 to 27 where not walls) ---
    # Row 2: cols 14 floor, col 15 VWALL, cols 16-21 HWALL, col 22 VWALL, cols 23-27 floor
    passage_floor_r2r4 = ((rows == 2) | (rows == 4)) & (cols == 14)
    goal_inner_floor = ((rows == 2) | (rows == 4)) & (cols >= 23) & (cols <= 27)
    # Row 3: cols 14-27 all floor (open corridor)
    passage_floor_r3 = (rows == 3) & (cols >= 14) & (cols <= 27)

    # --- Build map: walls first, then floor, then lava (overrides) ---
    all_hwall = top_bot_hwall | passage_hwall | passage_inner_hwall | goal_hwall
    all_vwall = (top_bot_vwall | left_wall | left_rwall | passage_inner_vwall
                 | goal_lwall | goal_rwall)
    all_floor = (left_floor | opening_floor | passage_floor_r2r4
                 | goal_inner_floor | passage_floor_r3)

    game_map = jnp.where(all_hwall, TileType.HWALL, game_map)
    game_map = jnp.where(all_vwall, TileType.VWALL, game_map)
    game_map = jnp.where(all_floor, TileType.FLOOR, game_map)
    game_map = jnp.where(lava, TileType.LAVA, game_map)

    return game_map


def generate_quest_easy(rng, params, static_params):
    """Generate a QuestEasy environment state."""
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng, rng_stair, rng_mon1, rng_mon2, rng_mon1t, rng_mon2t, rng_mon1p, rng_mon2p, rng_stats = \
        jax.random.split(rng, 9)

    player_stats = compute_initial_stats(rng_stats, RoleType.MONK, RaceType.HUMAN)

    game_map = _build_quest_easy_map()

    # Pad to static dimensions
    padded_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)
    padded_map = padded_map.at[:_ACTIVE_H, :_ACTIVE_W].set(game_map)

    # Player at (row 2, col 2) — .des BRANCH:(2,2,2,2)
    player_pos = jnp.array([2, 2], dtype=jnp.int32)

    # Stair in goal room: rows 2-4, cols 25-27
    stair_r = jax.random.randint(rng_stair, (), 2, 5)
    rng, rng_sc = jax.random.split(rng)
    stair_c = jax.random.randint(rng_sc, (), 25, 28)
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

    # 2 random monsters on random walkable tiles (MONSTER:random,random)
    mon1_type_idx = jax.random.randint(rng_mon1t, (), 0, _RANDOM_MONSTERS.shape[0])
    mon2_type_idx = jax.random.randint(rng_mon2t, (), 0, _RANDOM_MONSTERS.shape[0])
    mon1_type = _RANDOM_MONSTERS[mon1_type_idx]
    mon2_type = _RANDOM_MONSTERS[mon2_type_idx]

    # Place on random FLOOR tiles (after stair placement, so stair excluded)
    flat = padded_map.reshape(-1)
    is_floor = (flat == TileType.FLOOR)
    n_floor = is_floor.sum()
    cumsum = jnp.cumsum(is_floor)

    chosen1 = jax.random.randint(rng_mon1p, (), 0, jnp.maximum(n_floor, 1))
    match1 = (cumsum == chosen1 + 1) & is_floor
    flat_idx1 = jnp.argmax(match1)
    mon1_r = flat_idx1 // map_w
    mon1_c = flat_idx1 % map_w

    chosen2 = jax.random.randint(rng_mon2p, (), 0, jnp.maximum(n_floor, 1))
    match2 = (cumsum == chosen2 + 1) & is_floor
    flat_idx2 = jnp.argmax(match2)
    mon2_r = flat_idx2 // map_w
    mon2_c = flat_idx2 % map_w

    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(jnp.array([mon1_r, mon1_c]))
    mon_positions = mon_positions.at[1].set(jnp.array([mon2_r, mon2_c]))
    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(mon1_type)
    mon_types = mon_types.at[1].set(mon2_type)
    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[mon1_type])
    mon_health = mon_health.at[1].set(MONSTER_MAX_HP[mon2_type])
    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)
    mon_mask = mon_mask.at[1].set(True)

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
