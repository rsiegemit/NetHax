"""Tests for the MiniHack-compatible ``LevelGenerator`` builder API.

Verifies that builder directives translate correctly into the nethax
``EnvState`` schema without modifying the schema itself.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from Nethax.minihax.level_generator import (
    LevelGenerator,
    TERRAIN_CHAR_TO_TILE,
    TRAP_NAME_TO_TYPE,
    _MONSTER_NAME_TO_IDX,
    _OBJECT_NAME_TO_IDX,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.objects import OBJECTS
from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.traps import TrapType


def test_empty_level_creates_valid_envstate():
    """``LevelGenerator().get_factory()(rng)`` returns a valid EnvState."""
    lg = LevelGenerator(w=20, h=10)
    factory = lg.get_factory()
    rng = jax.random.PRNGKey(0)

    state = factory(rng)
    assert isinstance(state, EnvState)
    # Terrain array preserves the static shape (n_branches, max_levels, h, w).
    assert state.terrain.shape == (7, 32, 21, 80)
    # Within the requested (h=10, w=20) sub-region every cell is FLOOR (the
    # default fill character "." → FLOOR).
    sub = state.terrain[0, 0, :10, :20]
    assert bool(jnp.all(sub == int(TileType.FLOOR)))


def test_add_room_writes_floor_tiles():
    """``add_room`` carves a wall-bordered floor rectangle."""
    lg = LevelGenerator(w=20, h=10, fill=" ")  # blank fill so room stands out
    room_id = lg.add_room(x=2, y=2, w=5, h=4)
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(7))

    # Interior of the room must be FLOOR.
    interior = state.terrain[0, 0, 2:6, 2:7]   # rows y..y+h-1, cols x..x+w-1
    assert bool(jnp.all(interior == int(TileType.FLOOR))), (
        f"interior not all floor: {interior}"
    )
    # At least one wall tile around the border.
    wall_above = state.terrain[0, 0, 1, 2:7]
    assert bool(jnp.all(wall_above == int(TileType.WALL))), (
        f"wall row missing: {wall_above}"
    )
    # The returned room_id resolves in the metadata dict.
    assert room_id in lg._room_directives


def test_add_monster_named():
    """``add_monster("gnome")`` resolves to the gnome MONSTERS entry index."""
    expected_gnome_idx = next(
        i for i, m in enumerate(MONSTERS) if m.name == "gnome"
    )
    assert expected_gnome_idx == _MONSTER_NAME_TO_IDX["gnome"]

    lg = LevelGenerator(w=15, h=8)
    lg.add_monster("gnome", place=(4, 3))
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(11))

    # Build trace captured the resolved entry id.
    assert lg.last_monster_entry_ids == [expected_gnome_idx]

    # First monster slot is alive and at (row=3, col=4).
    assert bool(state.monster_ai.alive[0])
    assert int(state.monster_ai.pos[0, 0]) == 3
    assert int(state.monster_ai.pos[0, 1]) == 4
    # Monster AC matches the gnome row in the spawning lookup table.
    from Nethax.nethax.dungeon.spawning import _BASE_AC
    assert int(state.monster_ai.ac[0]) == int(_BASE_AC[expected_gnome_idx])


def test_add_object_named():
    """``add_object("apple")`` places an apple in ground_items."""
    expected_apple_idx = _OBJECT_NAME_TO_IDX["apple"]
    apple_entry = OBJECTS[expected_apple_idx]

    lg = LevelGenerator(w=12, h=8)
    lg.add_object("apple", place=(3, 4))
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(3))

    assert lg.last_object_entry_ids == [expected_apple_idx]
    # ground item at (row=4, col=3) stack index 0
    cat = int(state.ground_items.category[0, 0, 4, 3, 0])
    type_id = int(state.ground_items.type_id[0, 0, 4, 3, 0])
    assert cat == int(apple_entry.class_), f"expected food class, got {cat}"
    assert type_id == expected_apple_idx


def test_add_trap_named():
    """``add_trap("teleport", place=(5, 5))`` writes a TELEP_TRAP cell."""
    lg = LevelGenerator(w=12, h=8)
    lg.add_trap("teleport", place=(5, 5))
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(2))

    # traps state is indexed [num_levels, map_h, map_w] with branch 0 level 0
    # at flat index 0.
    trap_val = int(state.traps.trap_type[0, 5, 5])
    assert trap_val == int(TrapType.TELEP_TRAP), (
        f"expected TELEP_TRAP={int(TrapType.TELEP_TRAP)}, got {trap_val}"
    )
    assert lg.last_trap_types == [int(TrapType.TELEP_TRAP)]


def test_fill_terrain_lava():
    """``fill_terrain("L", x1, y1, x2, y2)`` writes LAVA tiles."""
    lg = LevelGenerator(w=15, h=10)
    lg.fill_terrain("L", x1=2, y1=2, x2=4, y2=3)
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(1))

    sub = state.terrain[0, 0, 2:4, 2:5]
    assert bool(jnp.all(sub == int(TileType.LAVA))), (
        f"lava rect not filled: {sub}"
    )
    # Neighboring cell outside the rect should still be the default FLOOR.
    assert int(state.terrain[0, 0, 4, 4]) == int(TileType.FLOOR)


def test_start_pos_sets_player_pos():
    """``set_start_pos(x, y)`` writes ``player_pos = (y, x)``."""
    lg = LevelGenerator(w=12, h=8)
    lg.set_start_pos(x=7, y=4)
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(99))

    assert int(state.player_pos[0]) == 4   # row
    assert int(state.player_pos[1]) == 7   # col
    assert lg.last_player_pos == (7, 4)


def test_reproducible_with_seed():
    """Same rng key produces identical EnvState terrain + monster pos."""
    def build():
        lg = LevelGenerator(w=20, h=10)
        lg.add_room(w=4, h=3)        # random room
        lg.add_monster("gnome")       # random floor placement
        return lg.get_factory()(jax.random.PRNGKey(123))

    a = build()
    b = build()

    assert bool(jnp.array_equal(a.terrain, b.terrain))
    assert bool(jnp.array_equal(a.monster_ai.pos, b.monster_ai.pos))
    assert bool(jnp.array_equal(a.monster_ai.alive, b.monster_ai.alive))


def test_goal_pos_marks_staircase_down():
    """``set_goal_pos`` stamps a STAIRCASE_DOWN tile."""
    lg = LevelGenerator(w=10, h=6)
    lg.set_goal_pos(x=4, y=3)
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(0))
    assert int(state.terrain[0, 0, 3, 4]) == int(TileType.STAIRCASE_DOWN)
    assert lg.last_goal_pos == (4, 3)


def test_unknown_monster_name_raises():
    """A name not in MONSTERS surfaces a clear KeyError at factory time."""
    lg = LevelGenerator(w=10, h=6)
    lg.add_monster("not_a_real_monster", place=(2, 2))
    with pytest.raises(KeyError):
        lg.get_factory()(jax.random.PRNGKey(0))


def test_corridor_carves_between_endpoints():
    """``add_corridor`` carves CORRIDOR tiles along an L-shape."""
    lg = LevelGenerator(w=15, h=10, fill=" ")
    lg.add_corridor(src=(2, 3), dst=(7, 6))
    factory = lg.get_factory()
    state = factory(jax.random.PRNGKey(0))
    # Horizontal leg along row 3 from x=2 to x=7
    horizontal = state.terrain[0, 0, 3, 2:8]
    assert bool(jnp.all(horizontal == int(TileType.CORRIDOR))), (
        f"horizontal leg missing: {horizontal}"
    )
    # Vertical leg along col 7 from y=3 to y=6
    vertical = state.terrain[0, 0, 3:7, 7]
    assert bool(jnp.all(vertical == int(TileType.CORRIDOR))), (
        f"vertical leg missing: {vertical}"
    )


def test_trap_name_table_complete():
    """All MiniHack trap names map to a valid TrapType."""
    for name, trap_type in TRAP_NAME_TO_TYPE.items():
        assert isinstance(trap_type, TrapType), f"{name} → {trap_type!r}"
