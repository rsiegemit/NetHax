"""Parity tests for the multi-turn pickaxe digging subsystem.

Vendor reference: vendor/nethack/src/dig.c
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.inventory import (
    InventoryState, ItemCategory, make_item, _items_from_list, MAX_INVENTORY_SLOTS,
    N_ARMOR_SLOTS, USER_NAME_LEN,
)
from Nethax.nethax.subsystems.digging import (
    DigState, start_dig, dig_tick, _terrain_hardness, PICKAXE_TYPE_ID, MATTOCK_TYPE_ID,
    DIG_DOWN,
)
from Nethax.nethax.constants.tiles import TileType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(wielded_type_id=None, wielded_category=None):
    """Return a default EnvState, optionally with a wielded tool.

    Player is placed at row=5, col=5 so adjacent tiles are always valid.
    """
    rng = jax.random.PRNGKey(42)
    state = EnvState.default(rng=rng)
    # Place player away from edges so north/south/east/west tiles are in-bounds.
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))

    if wielded_type_id is not None:
        cat = wielded_category if wielded_category is not None else int(ItemCategory.TOOL)
        from Nethax.nethax.subsystems.inventory import _empty_items_array
        items = _empty_items_array()
        items = items.replace(
            category=items.category.at[0].set(jnp.int8(cat)),
            type_id=items.type_id.at[0].set(jnp.int16(wielded_type_id)),
            quantity=items.quantity.at[0].set(jnp.int16(1)),
        )
        inv = state.inventory.replace(items=items, wielded=jnp.int8(0))
        state = state.replace(inventory=inv)

    return state


def _place_wall_north(state):
    """Set the tile north of player (row-1, col) to WALL and return updated state."""
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv, row - 1, col].set(jnp.int8(TileType.WALL))
    return state.replace(terrain=new_terrain)


def _run_dig_ticks(state, n):
    """Run dig_tick n times (non-jit, for test simplicity)."""
    rng = jax.random.PRNGKey(0)
    for _ in range(n):
        state = dig_tick(state, rng)
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_start_dig_requires_pickaxe():
    """start_dig is a no-op when no pickaxe/mattock is wielded.

    Cite: vendor/nethack/src/dig.c line 445 (wield check).
    """
    state = _make_state()  # bare hands
    # Ensure tile to north is wall so only the wield check can block us
    state = _place_wall_north(state)

    result = start_dig(state, direction=0)
    assert not bool(result.dig.active), "dig should not start without a pickaxe"


def test_dig_progresses_over_turns():
    """Effort accumulates over successive dig_tick calls."""
    state = _make_state(wielded_type_id=PICKAXE_TYPE_ID)
    state = _place_wall_north(state)
    state = start_dig(state, direction=0)

    assert bool(state.dig.active), "dig should have started"
    initial_effort = int(state.dig.effort)

    rng = jax.random.PRNGKey(0)
    state = dig_tick(state, rng)
    assert int(state.dig.effort) > initial_effort or not bool(state.dig.active), \
        "effort should increase each tick (or dig completed)"


def test_dig_completes_creates_doorway():
    """After enough ticks on a normal (non-maze, non-cavernous) level, the
    wall tile becomes an OPEN_DOOR (vendor D_NODOOR).

    Cite: vendor/nethack/src/dig.c lines 488-501.  Normal level WALL is
    converted to ``DOOR with D_NODOOR`` (an open doorway).  Maze levels
    convert to ROOM; cavernous levels (Gnomish Mines) convert to CORR.
    The earlier port always set CORRIDOR which was the bug fixed by D5.
    """
    state = _make_state(wielded_type_id=PICKAXE_TYPE_ID)
    state = _place_wall_north(state)

    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1

    # Verify wall is in place
    assert int(state.terrain[b, lv, row - 1, col]) == int(TileType.WALL)

    state = start_dig(state, direction=0)
    assert bool(state.dig.active)

    # Effort gain ≈ 14/tick; WALL_THRESHOLD = 100, so 30 ticks is plenty.
    state = _run_dig_ticks(state, 60)

    tile = int(state.terrain[b, lv, row - 1, col])
    assert tile == int(TileType.OPEN_DOOR), \
        f"expected OPEN_DOOR ({int(TileType.OPEN_DOOR)}) after dig, got {tile}"
    assert not bool(state.dig.active), "dig should be inactive after completion"


def test_dig_down_creates_hole_and_descends():
    """Downward dig completes → player descends one level.

    Cite: vendor/nethack/src/dig.c::digactualhole.
    """
    state = _make_state(wielded_type_id=PICKAXE_TYPE_ID)
    initial_level = int(state.dungeon.current_level)

    state = start_dig(state, direction=DIG_DOWN)
    assert bool(state.dig.active), "down dig should start"

    state = _run_dig_ticks(state, 200)

    new_level = int(state.dungeon.current_level)
    assert new_level == initial_level + 1, \
        f"expected descent to level {initial_level + 1}, got {new_level}"
    assert not bool(state.dig.active)


def test_dig_cancelled_when_moving():
    """If player moves away from adjacent target tile, dig cancels.

    Cite: vendor/nethack/src/dig.c::dodig (position check per turn).
    """
    state = _make_state(wielded_type_id=PICKAXE_TYPE_ID)
    state = _place_wall_north(state)
    state = start_dig(state, direction=0)
    assert bool(state.dig.active)

    # Move player far away by directly setting player_pos to a different row
    row = int(state.player_pos[0])
    col = int(state.player_pos[1])
    # Move 3 rows south (away from north wall target)
    new_pos = jnp.array([row + 3, col], dtype=jnp.int16)
    state = state.replace(player_pos=new_pos)

    rng = jax.random.PRNGKey(0)
    state = dig_tick(state, rng)

    assert not bool(state.dig.active), \
        "dig should be cancelled when player moves away from target"


def test_dig_hardness_lookup():
    """_terrain_hardness returns 200 for WALL and 0 for FLOOR."""
    wall_h = int(_terrain_hardness(jnp.int8(TileType.WALL)))
    floor_h = int(_terrain_hardness(jnp.int8(TileType.FLOOR)))

    assert wall_h == 200, f"WALL hardness should be 200, got {wall_h}"
    assert floor_h == 0, f"FLOOR hardness should be 0, got {floor_h}"
