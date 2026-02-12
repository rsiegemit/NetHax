"""World generator for Minihax-TreasureDash-v0.

From treasure_dash.des: 73x3 corridor with walls, 24 gold items at even
columns 6-52, downstair at (1,1), player start at (1,5).

.des coordinate convention: (col, row). Our code: (row, col).
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, PLAYER_START_HP, MONSTER_MAX_HP,
)
from Nethax.minihax.states import (
    HazardState, HazardStaticParams, Inventory, SimpleMonsters, GroundItems,
)
from Nethax.minihax.primitives.visibility import compute_visible


# Active map: 73 wide x 3 tall
_ACTIVE_W = 73
_ACTIVE_H = 3

# Gold positions: .des GOLD:1 at (col,1) for col in 6,8,10,...,52 -> 24 items
# In our (row, col) coords: all at row=1, cols=6,8,...,52
_GOLD_COLS = jnp.array(
    [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
     34, 36, 38, 40, 42, 44, 46, 48, 50, 52], dtype=jnp.int32
)
_NUM_GOLD = 24


def _make_treasure_dash_map():
    """Build the static treasure dash map."""
    game_map = jnp.full((_ACTIVE_H, _ACTIVE_W), TileType.VOID, dtype=jnp.int32)

    # Walls: top and bottom rows
    game_map = game_map.at[0, :].set(TileType.HWALL)
    game_map = game_map.at[_ACTIVE_H - 1, :].set(TileType.HWALL)

    # Walls: left and right columns (interior rows)
    game_map = game_map.at[1, 0].set(TileType.VWALL)
    game_map = game_map.at[1, _ACTIVE_W - 1].set(TileType.VWALL)

    # Floor interior: row 1, cols 1-71
    game_map = game_map.at[1, 1:_ACTIVE_W - 1].set(TileType.FLOOR)

    # Downstair at (row=1, col=1) — .des STAIR:(1,1),down
    game_map = game_map.at[1, 1].set(TileType.DOWNSTAIR)

    return game_map


def _pad_to_static(game_map, static_params):
    """Pad map to static dimensions."""
    sh = static_params.map_height
    sw = static_params.map_width
    padded = jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32)
    padded = padded.at[:_ACTIVE_H, :_ACTIVE_W].set(game_map)
    return padded


def generate_treasure_dash(rng, params, static_params):
    """Generate a TreasureDash environment state.

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

    game_map = _make_treasure_dash_map()

    # Player start: (row=1, col=5) — .des BRANCH:(5,1,5,1)
    player_pos = jnp.array([1, 5], dtype=jnp.int32)

    # Stair position: (row=1, col=1)
    stair_pos = jnp.array([1, 1], dtype=jnp.int32)

    # Gold items: 24 gold pieces at (row=1, col=6,8,...,52)
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[:_NUM_GOLD, 0].set(1)  # all row=1
    gi_positions = gi_positions.at[:_NUM_GOLD, 1].set(_GOLD_COLS)

    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[:_NUM_GOLD].set(ItemType.GOLD)

    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[:_NUM_GOLD].set(True)

    # Pad map
    padded_map = _pad_to_static(game_map, static_params)

    # Empty inventory
    inv = Inventory(
        item_ids=jnp.zeros(max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
    )

    # No monsters
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

    visible_map = compute_visible(
        player_pos, padded_map, static_params.map_height, static_params.map_width
    )
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
        terminal=False,
        state_rng=rng,
    )
