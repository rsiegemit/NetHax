"""World generator for Minihax-LavaCrossing-v0.

From lava_crossing.des: 13x7 map with walls, floor interior, lava column
at col 6. Random item (levitation or cold) on left bank. Player on left
bank, stair on right bank.

.des coordinate convention: (col, row). Our code: (row, col).
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, MonsterType, PLAYER_START_HP, MONSTER_MAX_HP,
)
from Nethax.minihax.states import (
    HazardState, HazardStaticParams, Inventory, SimpleMonsters, GroundItems,
)
from Nethax.minihax.primitives.visibility import compute_visible


# Active map: 13 wide x 7 tall
_ACTIVE_W = 13
_ACTIVE_H = 7

# Left bank: cols 1-5, rows 1-5  (in row,col)
# Right bank: cols 7-11, rows 1-5


def _make_lava_crossing_map():
    """Build the static lava crossing map."""
    game_map = jnp.full((_ACTIVE_H, _ACTIVE_W), TileType.VOID, dtype=jnp.int32)

    # Walls: top and bottom rows
    game_map = game_map.at[0, :].set(TileType.HWALL)
    game_map = game_map.at[_ACTIVE_H - 1, :].set(TileType.HWALL)

    # Walls: left and right columns (interior rows)
    game_map = game_map.at[1:_ACTIVE_H - 1, 0].set(TileType.VWALL)
    game_map = game_map.at[1:_ACTIVE_H - 1, _ACTIVE_W - 1].set(TileType.VWALL)

    # Floor interior
    rows = jnp.arange(_ACTIVE_H)[:, None]
    cols = jnp.arange(_ACTIVE_W)[None, :]
    interior = (rows >= 1) & (rows <= 5) & (cols >= 1) & (cols <= 11)
    game_map = jnp.where(interior, TileType.FLOOR, game_map)

    # Lava column at col 6, rows 1-5
    lava_col = (cols == 6) & (rows >= 1) & (rows <= 5)
    game_map = jnp.where(lava_col, TileType.LAVA, game_map)

    return game_map


def _pad_to_static(game_map, static_params):
    """Pad map to static dimensions."""
    sh = static_params.map_height
    sw = static_params.map_width
    padded = jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32)
    padded = padded.at[:_ACTIVE_H, :_ACTIVE_W].set(game_map)
    return padded


def generate_lava_crossing(rng, params, static_params):
    """Generate a LavaCrossing environment state.

    Args:
        rng: JAX PRNG key
        params: EnvParams
        static_params: HazardStaticParams

    Returns:
        HazardState
    """
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items

    rng, rng_player, rng_stair, rng_item_branch, rng_item_sub, rng_item_sub2 = \
        jax.random.split(rng, 6)

    game_map = _make_lava_crossing_map()

    # Player position: random in left bank (rows 1-5, cols 1-5)
    player_r = jax.random.randint(rng_player, (), 1, 6)
    rng, rng_pc = jax.random.split(rng)
    player_c = jax.random.randint(rng_pc, (), 1, 6)
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    # Stair position: random in right bank (rows 1-5, cols 7-11)
    stair_r = jax.random.randint(rng_stair, (), 1, 6)
    rng, rng_sc = jax.random.split(rng)
    stair_c = jax.random.randint(rng_sc, (), 7, 12)
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)

    # Place downstair on map
    game_map = game_map.at[stair_pos[0], stair_pos[1]].set(TileType.DOWNSTAIR)

    # Random item on left bank
    # 50% levitation branch, 50% cold branch
    # Levitation: 33% potion, 33% ring, 34% boots
    # Cold: 50% wand, 50% frost horn
    branch = jax.random.uniform(rng_item_branch, ()) < 0.5
    sub_roll = jax.random.uniform(rng_item_sub, ())
    sub_roll2 = jax.random.uniform(rng_item_sub2, ())

    lev_item = jnp.where(sub_roll < 0.33, ItemType.POTION_LEVITATION,
               jnp.where(sub_roll < 0.66, ItemType.RING_LEVITATION,
                         ItemType.BOOTS_LEVITATION))
    cold_item = jnp.where(sub_roll2 < 0.5, ItemType.WAND_COLD, ItemType.FROST_HORN)
    item_type = jnp.where(branch, lev_item, cold_item)

    # Random item position on left bank
    rng, rng_ir, rng_ic = jax.random.split(rng, 3)
    item_r = jax.random.randint(rng_ir, (), 1, 6)
    item_c = jax.random.randint(rng_ic, (), 1, 6)

    # Ground items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([item_r, item_c]))
    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(item_type)
    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)

    # Pad map
    padded_map = _pad_to_static(game_map, static_params)

    # Empty inventory
    inv = Inventory(
        item_ids=jnp.zeros(max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
    )

    # No monsters in lava crossing
    monsters = SimpleMonsters(
        position=jnp.zeros((max_m, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_m, dtype=jnp.int32),
        health=jnp.zeros(max_m, dtype=jnp.int32),
        mask=jnp.zeros(max_m, dtype=jnp.bool_),
    )

    ground_items = GroundItems(
        position=gi_positions,
        type_id=gi_types,
        mask=gi_mask,
    )

    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width)
    return HazardState(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        player_hp=PLAYER_START_HP,
        player_max_hp=PLAYER_START_HP,
        player_levitating=False,
        levitation_turns=0,
        inventory=inv,
        monsters=monsters,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        timestep=0,
        prev_action=0,
        terminal=False,
        state_rng=rng,
    )
