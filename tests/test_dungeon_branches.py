"""Wave 4 — dungeon-branches tests.

Tests for init_branch_graph, generate_mines_level, generate_sokoban_level,
generate_quest_level, and traverse_stair_cross_branch.

All imports are lazy (inside test functions) to keep collection robust if
sibling Wave 4 agents are still updating other modules.
"""

import pytest


# ---------------------------------------------------------------------------
# init_branch_graph
# ---------------------------------------------------------------------------

def test_init_branch_graph_creates_main_to_mines_link():
    """Main Dlvl 3 must have a down-link pointing into Mines Dlvl 1."""
    import jax
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    rng = jax.random.PRNGKey(0)
    graph = init_branch_graph(rng, None)

    # stair_links: int8[N_BRANCHES, MAX_LEVELS, 2]  (dst_branch, dst_level)
    dst_branch = int(graph.stair_links[Branch.MAIN, 3 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 3 - 1, 1])
    assert dst_branch == int(Branch.GNOMISH_MINES), (
        f"expected Main Dlvl 3 → Mines, got branch={dst_branch}"
    )
    assert dst_level == 1, f"expected Mines Dlvl 1, got {dst_level}"


def test_init_branch_graph_creates_main_to_sokoban_link():
    """Sokoban entrance must be at the canonical Main Dlvl 8 (Oracle +1 up).

    Vendor dungeon.def: ``CHAINBRANCH: "Sokoban" "oracle" + (1, 0) up`` —
    Oracle sits in Main Dlvl 5..10 and Sokoban entry is one level above it,
    yielding the canonical 6..10 range.  We pick mid-point 8.
    """
    import jax
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    graph = init_branch_graph(jax.random.PRNGKey(1), None)
    dst_branch = int(graph.stair_links[Branch.MAIN, 8 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 8 - 1, 1])
    assert dst_branch == int(Branch.SOKOBAN)
    assert dst_level == 1


def test_init_branch_graph_creates_main_to_quest_link():
    """Quest portal entry at Main Dlvl 14 (XL14 gate, Oracle + 6 ±2).

    Vendor dungeon.def: ``CHAINBRANCH: "The Quest" "oracle" + (6, 2) portal``.
    """
    import jax
    from Nethax.nethax.dungeon.branches import init_branch_graph, Branch

    graph = init_branch_graph(jax.random.PRNGKey(2), None)
    dst_branch = int(graph.stair_links[Branch.MAIN, 14 - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, 14 - 1, 1])
    assert dst_branch == int(Branch.QUEST)
    assert dst_level == 1


# ---------------------------------------------------------------------------
# generate_mines_level
# ---------------------------------------------------------------------------

def test_generate_mines_level_has_cave_terrain():
    """Mines layout must be irregular (not a single rectangular room).

    Heuristic: the boundary between floor and wall must zig-zag enough that
    floor tiles do not form a single perfectly rectangular block.  We check
    that the bounding box of floor tiles strictly contains some non-floor
    interior cell — which would be impossible for a single room.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import generate_mines_level
    from Nethax.nethax.constants.tiles import TileType

    rng = jax.random.PRNGKey(3)
    terrain, monsters, _items = generate_mines_level(rng, depth=2)

    terrain_np = jnp.asarray(terrain).__array__()

    floor_mask = (terrain_np == int(TileType.FLOOR)) | \
                 (terrain_np == int(TileType.STAIRCASE_UP)) | \
                 (terrain_np == int(TileType.STAIRCASE_DOWN))

    assert floor_mask.any(), "Mines level has no floor tiles"

    # Compute bounding box of floor tiles.
    rows = floor_mask.any(axis=1)
    cols = floor_mask.any(axis=0)
    r_lo, r_hi = int(rows.argmax()), len(rows) - 1 - int(rows[::-1].argmax())
    c_lo, c_hi = int(cols.argmax()), len(cols) - 1 - int(cols[::-1].argmax())

    bbox = floor_mask[r_lo:r_hi + 1, c_lo:c_hi + 1]
    fill_ratio = bbox.mean()
    # A pure rectangular room would have ratio == 1.0.  A cave should have
    # < 0.95 — i.e. interior wall pockets break the rectangle.
    assert fill_ratio < 0.95, (
        f"Mines layout looks like one rectangle (fill {fill_ratio:.2f}); "
        f"expected cave-style irregular boundary."
    )

    # Mines spawns should include gnomes / dwarves / kobolds / hobbits.
    assert len(monsters) > 0, "Mines generator returned no monster suggestions"


# ---------------------------------------------------------------------------
# generate_sokoban_level
# ---------------------------------------------------------------------------

def test_generate_sokoban_level_has_boulders():
    """A Sokoban floor must contain at least one BOULDER tile."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import (
        generate_sokoban_level,
        BOULDER_TILE,
    )

    rng = jax.random.PRNGKey(4)
    terrain, boulders, pits = generate_sokoban_level(rng, floor_number=2)
    terrain_np = jnp.asarray(terrain).__array__()
    assert (terrain_np == BOULDER_TILE).sum() >= 1, (
        "Sokoban layout has no boulders"
    )
    assert len(boulders) >= 1, "boulder_positions empty"


# ---------------------------------------------------------------------------
# generate_quest_level
# ---------------------------------------------------------------------------

def test_generate_quest_level_has_role_appropriate_monsters():
    """Quest generator returns at least one monster type for each role."""
    import jax
    from Nethax.nethax.dungeon.branches import generate_quest_level
    from Nethax.nethax.constants.monsters import MONSTERS

    rng = jax.random.PRNGKey(5)
    # Test a knight (role=4) — guardian should be "wraith" per our table.
    terrain, monsters, _items = generate_quest_level(rng, depth=2, role=4)
    assert len(monsters) >= 1, "Quest generator returned no monsters"

    # The first suggested monster should be a wraith for knight.
    first_name = MONSTERS[monsters[0]].name
    assert first_name == "wraith", (
        f"expected knight quest guardian = wraith, got {first_name}"
    )


# ---------------------------------------------------------------------------
# traverse_stair_cross_branch — happy path
# ---------------------------------------------------------------------------

def _state_with_branch_graph(rng):
    """Helper: build a default EnvState and apply init_branch_graph linking."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.state import EnvState, StaticParams
    from Nethax.nethax.dungeon.branches import (
        init_branch_graph,
        apply_branch_graph_to_dungeon,
    )

    state = EnvState.default(rng=rng, static=StaticParams())
    graph = init_branch_graph(rng, None)
    new_dungeon = apply_branch_graph_to_dungeon(state.dungeon, graph)
    state = state.replace(dungeon=new_dungeon)
    return state, graph


def test_traverse_stair_cross_branch_to_mines():
    """Start on Main Dlvl 3 → go down → land on Mines Dlvl 1."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch

    rng = jax.random.PRNGKey(7)
    state, _graph = _state_with_branch_graph(rng)

    # Place player on Main Dlvl 3.
    state = state.replace(
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.MAIN),
            current_level=jnp.int8(3),
        )
    )

    out = traverse_stair_cross_branch(
        state, rng, target_branch=-1, direction=+1
    )

    assert int(out.dungeon.current_branch) == int(Branch.GNOMISH_MINES), (
        f"expected to be in Mines after descend, got branch="
        f"{int(out.dungeon.current_branch)}"
    )
    assert int(out.dungeon.current_level) == 1


def test_traverse_stair_cross_branch_returns_main():
    """Descend Main→Mines, then ascend → back on Main Dlvl 3."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch

    rng = jax.random.PRNGKey(8)
    state, _graph = _state_with_branch_graph(rng)

    state = state.replace(
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.MAIN),
            current_level=jnp.int8(3),
        )
    )

    # Step 1: descend into Mines.
    mid = traverse_stair_cross_branch(state, rng, target_branch=-1, direction=+1)
    assert int(mid.dungeon.current_branch) == int(Branch.GNOMISH_MINES)
    assert int(mid.dungeon.current_level) == 1

    # Step 2: ascend back to Main.
    out = traverse_stair_cross_branch(mid, rng, target_branch=-1, direction=-1)
    assert int(out.dungeon.current_branch) == int(Branch.MAIN)
    assert int(out.dungeon.current_level) == 3

    # And the Mines level should still be marked as generated in level_memory
    # (state preserved across the round-trip).
    assert bool(out.level_memory.generated[int(Branch.GNOMISH_MINES), 0])


def test_cross_branch_player_pos_lands_on_stair():
    """After descending Main→Mines, player_pos must equal a STAIRCASE_UP tile."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch
    from Nethax.nethax.constants.tiles import TileType

    rng = jax.random.PRNGKey(11)
    state, _graph = _state_with_branch_graph(rng)
    state = state.replace(
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.MAIN),
            current_level=jnp.int8(3),
        )
    )

    out = traverse_stair_cross_branch(state, rng, target_branch=-1, direction=+1)

    r = int(out.player_pos[0])
    c = int(out.player_pos[1])
    dst_terrain = out.terrain[int(out.dungeon.current_branch),
                              int(out.dungeon.current_level) - 1]
    tile = int(dst_terrain[r, c])
    assert tile == int(TileType.STAIRCASE_UP), (
        f"expected player on STAIRCASE_UP after descend, got tile={tile} "
        f"at ({r}, {c})"
    )
