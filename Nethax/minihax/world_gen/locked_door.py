"""World generators for LockedDoor environments (2 variants).

From .des files:
- locked_door.des: 13x7, locked door at (col 6, row 3), player random in left room,
  stair random in right room.
- locked_door_fixed.des: Same map but player at (col 3, row 3), stair at (col 8, row 3).

.des coordinate convention: (col, row). Our code: (row, col).

Map layout (13 cols x 7 rows):
-------------
|.....|.....|
|.....|.....|
|.....+.....|
|.....|.....|
|.....|.....|
-------------
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, PLAYER_START_HP,
)
from Nethax.minihax.states import (
    HazardState, HazardStaticParams, Inventory, SimpleMonsters, GroundItems,
)
from Nethax.minihax.primitives.visibility import compute_visible

_ACTIVE_H = 7
_ACTIVE_W = 13


def _build_locked_door_map():
    """Build the locked door map from the .des layout."""
    game_map = jnp.full((_ACTIVE_H, _ACTIVE_W), TileType.VOID, dtype=jnp.int32)

    rows = jnp.arange(_ACTIVE_H)[:, None]
    cols = jnp.arange(_ACTIVE_W)[None, :]

    # Top and bottom walls
    top = (rows == 0) & (cols >= 0) & (cols <= 12)
    bot = (rows == 6) & (cols >= 0) & (cols <= 12)

    # Left and right walls
    lwall = (cols == 0) & (rows >= 1) & (rows <= 5)
    rwall = (cols == 12) & (rows >= 1) & (rows <= 5)

    # Center dividing wall at col 6, rows 1-5 (except row 3 = door)
    center_wall = (cols == 6) & (rows >= 1) & (rows <= 5) & (rows != 3)

    # Locked door at (col 6, row 3) = our (row 3, col 6)
    door = (rows == 3) & (cols == 6)

    # Floor: left room (rows 1-5, cols 1-5) and right room (rows 1-5, cols 7-11)
    left_floor = (rows >= 1) & (rows <= 5) & (cols >= 1) & (cols <= 5)
    right_floor = (rows >= 1) & (rows <= 5) & (cols >= 7) & (cols <= 11)

    # Build
    game_map = jnp.where(top | bot, TileType.HWALL, game_map)
    game_map = jnp.where(lwall | rwall | center_wall, TileType.VWALL, game_map)
    game_map = jnp.where(left_floor | right_floor, TileType.FLOOR, game_map)
    game_map = jnp.where(door, TileType.DOOR_LOCKED, game_map)

    return game_map


def _make_locked_door_state(rng, params, static_params, player_pos, stair_pos):
    """Common state construction for both locked door variants."""
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items
    map_h = static_params.map_height
    map_w = static_params.map_width

    game_map = _build_locked_door_map()

    # Pad to static dimensions
    padded_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)
    padded_map = padded_map.at[:_ACTIVE_H, :_ACTIVE_W].set(game_map)

    # Place downstair
    padded_map = padded_map.at[stair_pos[0], stair_pos[1]].set(TileType.DOWNSTAIR)

    # No monsters
    monsters = SimpleMonsters(
        position=jnp.zeros((max_m, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_m, dtype=jnp.int32),
        health=jnp.zeros(max_m, dtype=jnp.int32),
        mask=jnp.zeros(max_m, dtype=jnp.bool_),
    )

    # No items
    inv = Inventory(
        item_ids=jnp.zeros(max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
    )
    ground_items = GroundItems(
        position=jnp.zeros((max_gi, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_gi, dtype=jnp.int32),
        mask=jnp.zeros(max_gi, dtype=jnp.bool_),
    )

    visible_map = compute_visible(player_pos, padded_map, map_h, map_w)
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


def generate_locked_door(rng, params, static_params):
    """Generate LockedDoor: random player in left room, random stair in right room.

    .des: BRANCH:(1,1,5,5) -> player random in (rows 1-5, cols 1-5)
    .des: STAIR:rndcoord($right_room) -> stair random in (rows 1-5, cols 7-11)
    """
    rng, rng_pr, rng_pc, rng_sr, rng_sc = jax.random.split(rng, 5)

    player_r = jax.random.randint(rng_pr, (), 1, 6)
    player_c = jax.random.randint(rng_pc, (), 1, 6)
    player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

    stair_r = jax.random.randint(rng_sr, (), 1, 6)
    stair_c = jax.random.randint(rng_sc, (), 7, 12)
    stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)

    return _make_locked_door_state(rng, params, static_params, player_pos, stair_pos)


def generate_locked_door_fixed(rng, params, static_params):
    """Generate LockedDoorFixed: player at (3,3), stair at (3,8).

    .des: BRANCH:(3,3,3,3) -> (col 3, row 3) = our (row 3, col 3)
    .des: STAIR:(8,3) -> (col 8, row 3) = our (row 3, col 8)
    """
    player_pos = jnp.array([3, 3], dtype=jnp.int32)
    stair_pos = jnp.array([3, 8], dtype=jnp.int32)

    return _make_locked_door_state(rng, params, static_params, player_pos, stair_pos)
