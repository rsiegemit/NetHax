"""Wave 5 Phase 2 — Gehennom branch, Valley of the Dead, vibrating-square portal.

Tests for the Gehennom branch extension:
  * init_branch_graph wires Gehennom from Main bottom.
  * Valley of the Dead generator (Gehennom L1) contains a vibrating square.
  * Gehennom L2..L16 are walkable and demon-populated.
  * Stepping on the vibrating square reveals a magic-portal tile.
  * Triggering the magic portal advances current_branch / current_level.

All imports are lazy to keep test collection robust if sibling Wave-5
agents are still updating other modules.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest


# ---------------------------------------------------------------------------
# Branch graph
# ---------------------------------------------------------------------------

def test_gehennom_branch_in_graph():
    """BRANCH_TABLE must contain a Gehennom branch with 16 levels."""
    from Nethax.nethax.dungeon.branches import BRANCH_TABLE, Branch

    geh_entry = None
    for info in BRANCH_TABLE:
        if int(info.branch_id) == int(Branch.GEHENNOM):
            geh_entry = info
            break
    assert geh_entry is not None, "Gehennom not present in BRANCH_TABLE"
    assert int(geh_entry.num_levels) == 16, (
        f"Expected 16 Gehennom levels, got {int(geh_entry.num_levels)}"
    )


def test_init_branch_graph_creates_main_to_gehennom_link():
    """Main bottom (Dlvl 26) must link down into Gehennom L1."""
    import jax
    from Nethax.nethax.dungeon.branches import (
        init_branch_graph, Branch, _MAIN_DLVL_GEHENNOM_ENTRY,
    )

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    dst_branch = int(graph.stair_links[Branch.MAIN, _MAIN_DLVL_GEHENNOM_ENTRY - 1, 0])
    dst_level  = int(graph.stair_links[Branch.MAIN, _MAIN_DLVL_GEHENNOM_ENTRY - 1, 1])
    assert dst_branch == int(Branch.GEHENNOM), (
        f"Main Dlvl {_MAIN_DLVL_GEHENNOM_ENTRY} should link to Gehennom, "
        f"got branch={dst_branch}"
    )
    assert dst_level == 1


def test_valley_of_dead_level_above_gehennom():
    """Gehennom L1 (Valley) entry exit must be back to Main, not deeper."""
    import jax
    from Nethax.nethax.dungeon.branches import (
        init_branch_graph, Branch, _MAIN_DLVL_GEHENNOM_ENTRY,
    )

    graph = init_branch_graph(jax.random.PRNGKey(0), None)
    # The up-link slot at Gehennom L1 (index 0) points back to Main.
    back_branch = int(graph.stair_links[Branch.GEHENNOM, 0, 0])
    back_level  = int(graph.stair_links[Branch.GEHENNOM, 0, 1])
    assert back_branch == int(Branch.MAIN)
    assert back_level == _MAIN_DLVL_GEHENNOM_ENTRY


# ---------------------------------------------------------------------------
# Per-level Gehennom traversal (L1 → L16)
# ---------------------------------------------------------------------------

def _state_with_gehennom(rng):
    """Build EnvState with the branch graph applied and player parked at
    the Castle (Main Dlvl 26)."""
    import jax.numpy as jnp
    from Nethax.nethax.state import EnvState, StaticParams
    from Nethax.nethax.dungeon.branches import (
        init_branch_graph, apply_branch_graph_to_dungeon, Branch,
        _MAIN_DLVL_GEHENNOM_ENTRY,
    )

    state = EnvState.default(rng=rng, static=StaticParams())
    graph = init_branch_graph(rng, None)
    new_dungeon = apply_branch_graph_to_dungeon(state.dungeon, graph)
    state = state.replace(
        dungeon=new_dungeon.replace(
            current_branch=jnp.int8(Branch.MAIN),
            current_level=jnp.int8(_MAIN_DLVL_GEHENNOM_ENTRY),
        )
    )
    return state


def test_gehennom_l1_to_l16_descend():
    """Player can descend Main→Gehennom L1, then L1→L2→...→L16."""
    import jax
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch

    rng = jax.random.PRNGKey(42)
    state = _state_with_gehennom(rng)

    # Step 1: descend Main → Gehennom L1.
    state = traverse_stair_cross_branch(state, rng, target_branch=-1, direction=+1)
    assert int(state.dungeon.current_branch) == int(Branch.GEHENNOM)
    assert int(state.dungeon.current_level) == 1

    # Step 2..16: descend through Gehennom.
    for expected_level in range(2, 17):
        state = traverse_stair_cross_branch(
            state, rng, target_branch=-1, direction=+1,
        )
        assert int(state.dungeon.current_branch) == int(Branch.GEHENNOM), (
            f"Expected to stay in Gehennom at level {expected_level}, "
            f"got branch {int(state.dungeon.current_branch)}"
        )
        assert int(state.dungeon.current_level) == expected_level, (
            f"Descent {expected_level - 1}→{expected_level} failed: now at "
            f"L{int(state.dungeon.current_level)}"
        )


def test_gehennom_levels_have_demons():
    """generate_gehennom_level must propose at least one demon-class
    monster for each non-Valley depth."""
    import jax
    from Nethax.nethax.dungeon.branches import generate_gehennom_level
    from Nethax.nethax.constants.monsters import MONSTERS

    demon_names = {
        "water demon", "incubus", "horned devil", "erinys", "barbed devil",
        "marilith", "vrock", "hezrou", "bone devil", "ice devil",
        "nalfeshnee", "pit fiend", "sandestin", "balrog",
        "Juiblex", "Yeenoghu", "Orcus", "Geryon", "Dispater",
        "Baalzebub", "Asmodeus", "Demogorgon",
    }

    for depth in (2, 4, 7, 9, 11, 15, 16):
        rng = jax.random.PRNGKey(depth)
        _terrain, monsters, _items = generate_gehennom_level(rng, depth)
        assert len(monsters) >= 1, (
            f"Gehennom L{depth} returned no monster suggestions"
        )
        names = {MONSTERS[i].name for i in monsters}
        overlap = names & demon_names
        assert overlap, (
            f"Gehennom L{depth} has no demon-class monsters; got {names}"
        )


# ---------------------------------------------------------------------------
# Vibrating square + magic portal wiring
# ---------------------------------------------------------------------------

def test_vibrating_square_trap_present_in_valley():
    """generate_valley_of_dead must place a TRAP tile and return the
    vibrating-square position."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.dungeon.branches import generate_valley_of_dead
    from Nethax.nethax.constants.tiles import TileType

    rng = jax.random.PRNGKey(0)
    terrain, monsters, vs_pos = generate_valley_of_dead(rng)
    terrain_np = jnp.asarray(terrain).__array__()
    assert (terrain_np == int(TileType.TRAP)).sum() >= 1, (
        "Valley of the Dead must contain at least one trap tile"
    )
    r, c = vs_pos
    assert int(terrain_np[r, c]) == int(TileType.TRAP), (
        f"Vibrating-square position ({r},{c}) should host a TRAP tile, "
        f"got tile={int(terrain_np[r, c])}"
    )
    # Ghostly opponents proposed.
    assert len(monsters) >= 1, "Valley should propose ghostly opponents"


def test_vibrating_square_step_reveals_portal():
    """trigger_vibrating_square must materialise a MAGIC_PORTAL trap in
    one of the orthogonal neighbour tiles."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.state import EnvState, StaticParams
    from Nethax.nethax.dungeon.branches import (
        Branch, generate_valley_of_dead,
    )
    from Nethax.nethax.subsystems.traps import (
        TrapType, place_trap, trigger_vibrating_square,
    )

    rng = jax.random.PRNGKey(7)
    state = EnvState.default(rng=rng, static=StaticParams())

    # Park player in Gehennom L1 with the Valley terrain in place.
    state = state.replace(
        dungeon=state.dungeon.replace(
            current_branch=jnp.int8(Branch.GEHENNOM),
            current_level=jnp.int8(1),
        )
    )
    valley_terrain, _mons, vs_pos = generate_valley_of_dead(rng)
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain_all = state.terrain.at[b, lv].set(valley_terrain)
    state = state.replace(terrain=new_terrain_all)

    # Place a VIBRATING_SQUARE trap at the canonical vibrating-square tile.
    max_lv = int(state.terrain.shape[1])
    flat_lv = b * max_lv + lv
    pos = jnp.array([flat_lv, vs_pos[0], vs_pos[1]], dtype=jnp.int32)
    new_traps = place_trap(state.traps, pos, TrapType.VIBRATING_SQUARE, rng)
    state = state.replace(
        traps=new_traps,
        player_pos=jnp.array([vs_pos[0], vs_pos[1]], dtype=jnp.int16),
    )

    # Step it: portal should appear.
    state2 = trigger_vibrating_square(state, state.player_pos)

    # Count magic-portal cells in the trap layer for this level.
    portal_count = int(
        (state2.traps.trap_type[flat_lv] == int(TrapType.MAGIC_PORTAL)).sum()
    )
    assert portal_count >= 1, (
        f"Expected at least one MAGIC_PORTAL trap after stepping on "
        f"vibrating square, got {portal_count}"
    )
    # Reveal flag should also be set on the DungeonState.
    assert bool(state2.dungeon.vibrating_square_revealed), (
        "vibrating_square_revealed flag must be True after trigger"
    )


def test_magic_portal_traverses_branches():
    """Stepping on a MAGIC_PORTAL while on Gehennom L1 must advance the
    current_branch / current_level to a different (branch, level)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.state import EnvState, StaticParams
    from Nethax.nethax.dungeon.branches import (
        Branch, init_branch_graph, apply_branch_graph_to_dungeon,
    )
    from Nethax.nethax.subsystems.traps import trigger_magic_portal

    rng = jax.random.PRNGKey(11)
    state = EnvState.default(rng=rng, static=StaticParams())
    graph = init_branch_graph(rng, None)
    state = state.replace(
        dungeon=apply_branch_graph_to_dungeon(state.dungeon, graph).replace(
            current_branch=jnp.int8(Branch.GEHENNOM),
            current_level=jnp.int8(1),
        )
    )

    before_branch = int(state.dungeon.current_branch)
    before_level  = int(state.dungeon.current_level)

    new_state = trigger_magic_portal(state, rng)

    after_branch = int(new_state.dungeon.current_branch)
    after_level  = int(new_state.dungeon.current_level)
    # Either branch or level should change (default mapping: L1 → L2 in
    # the same branch).  Accept any change.
    assert (after_branch, after_level) != (before_branch, before_level), (
        f"Magic portal did not move the player: still at "
        f"({after_branch}, {after_level})"
    )
