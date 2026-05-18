"""Stair traversal parity tests.

Covers:
  - Within-branch stair-down descends one level (current_level += 1).
  - Within-branch stair-up ascends one level (current_level -= 1).
  - Round-trip: stair-down then stair-up returns player to matching up-stair pos.
  - Trapdoor falls land on a random FLOOR tile, NOT on the up-staircase.
  - Branch transition to Gnomish Mines via stair-down at the Mines-entry level.

Citation:
  vendor/nethack/src/do.c::dolook (down) + dohol (up).
  vendor/nethack/src/trap.c::dotrap TRAPDOOR.
  vendor/nethack/src/dungeon.c::init_dungeons (branch links).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants import TileType
from Nethax.nethax.dungeon.branches import (
    Branch,
    _MAIN_DLVL_MINES_ENTRY,
    init_branch_graph,
    apply_branch_graph_to_dungeon,
)
from Nethax.nethax.subsystems.action_dispatch import _stair_down, _stair_up
from Nethax.nethax.subsystems.traps import _trap_hole, _trap_trapdoor
from Nethax.nethax.dungeon.level_memory import traverse_stair_cross_branch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    """Reset NethaxEnv and return (state, rng)."""
    rng = jax.random.PRNGKey(42)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return state, rng


def _place_stair_under_player(state, tile: TileType):
    """Stamp *tile* at the player's current position on the current level."""
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    r, c = int(state.player_pos[0]), int(state.player_pos[1])
    new_terrain = state.terrain.at[b, lv, r, c].set(jnp.int8(int(tile)))
    return state.replace(terrain=new_terrain)


def _stamp_tile_at(state, branch, level_1based, row, col, tile: TileType):
    """Write *tile* at an explicit (branch, level, row, col) coordinate."""
    lv = level_1based - 1
    new_terrain = state.terrain.at[branch, lv, row, col].set(jnp.int8(int(tile)))
    return state.replace(terrain=new_terrain)


def _wire_stair_link(state, src_branch, src_level_1based, direction, dst_branch, dst_level_1based):
    """Set stair_links[src_branch, src_level-1, direction] = (dst_branch, dst_level)."""
    new_sl = state.dungeon.stair_links.at[
        src_branch, src_level_1based - 1, direction
    ].set(jnp.array([dst_branch, dst_level_1based], dtype=jnp.int8))
    new_dungeon = state.dungeon.replace(stair_links=new_sl)
    return state.replace(dungeon=new_dungeon)


# ---------------------------------------------------------------------------
# test_stair_down_descends_one
# ---------------------------------------------------------------------------

def test_stair_down_descends_one():
    """Standing on STAIRCASE_DOWN at level 1 → current_level becomes 2.

    Citation: do.c::dodown() → goto_level(level+1).
    """
    state, rng = _fresh_state()
    assert int(state.dungeon.current_level) == 1

    # Place player on STAIRCASE_DOWN; wire a same-branch link (dir=1 = down).
    state = _place_stair_under_player(state, TileType.STAIRCASE_DOWN)
    # Wire link: Main L1 down → Main L2.
    state = _wire_stair_link(state, int(Branch.MAIN), 1, 1, int(Branch.MAIN), 2)
    # Ensure STAIRCASE_UP exists on L2 so traverse_stair_cross_branch can land there.
    state = _stamp_tile_at(state, int(Branch.MAIN), 2, 10, 10, TileType.STAIRCASE_UP)

    new_state = _stair_down(state, rng)
    assert int(new_state.dungeon.current_level) == 2


# ---------------------------------------------------------------------------
# test_stair_up_ascends_one
# ---------------------------------------------------------------------------

def test_stair_up_ascends_one():
    """Standing on STAIRCASE_UP at level 2 → current_level becomes 1.

    Citation: do.c::dohol() → goto_level(level-1).
    """
    state, rng = _fresh_state()

    # Move to level 2 manually.
    new_dungeon = state.dungeon.replace(current_level=jnp.int8(2))
    state = state.replace(dungeon=new_dungeon)

    # Place STAIRCASE_UP under player on L2; wire link L2 up → L1.
    state = _place_stair_under_player(state, TileType.STAIRCASE_UP)
    state = _wire_stair_link(state, int(Branch.MAIN), 2, 0, int(Branch.MAIN), 1)
    # Ensure STAIRCASE_DOWN on L1 for landing.
    r, c = int(state.player_pos[0]), int(state.player_pos[1])
    state = _stamp_tile_at(state, int(Branch.MAIN), 1, r, c, TileType.STAIRCASE_DOWN)

    new_state = _stair_up(state, rng)
    assert int(new_state.dungeon.current_level) == 1


# ---------------------------------------------------------------------------
# test_player_lands_on_matching_stair
# ---------------------------------------------------------------------------

def test_player_lands_on_matching_stair():
    """Stair-down then stair-up returns player to the STAIRCASE_UP pos on L1.

    Citation: do.c — symmetric stair pair placement.
    """
    state, rng = _fresh_state()

    # Put an up-stair at a known position on L1, down-stair at player pos.
    up_row, up_col = 5, 5
    state = _place_stair_under_player(state, TileType.STAIRCASE_DOWN)
    state = _stamp_tile_at(state, int(Branch.MAIN), 1, up_row, up_col, TileType.STAIRCASE_UP)

    # Wire L1 down → L2; L2 up → L1.
    state = _wire_stair_link(state, int(Branch.MAIN), 1, 1, int(Branch.MAIN), 2)
    state = _wire_stair_link(state, int(Branch.MAIN), 2, 0, int(Branch.MAIN), 1)
    # Put STAIRCASE_UP on L2 at (10,10) for the first descent landing.
    state = _stamp_tile_at(state, int(Branch.MAIN), 2, 10, 10, TileType.STAIRCASE_UP)
    # Put STAIRCASE_DOWN on L2 at (10,10) for re-ascent.
    state = _stamp_tile_at(state, int(Branch.MAIN), 2, 10, 11, TileType.STAIRCASE_DOWN)

    # Go down to L2.
    state2 = _stair_down(state, rng)
    assert int(state2.dungeon.current_level) == 2

    # Place STAIRCASE_UP under player on L2 and stamp it.
    state2 = _place_stair_under_player(state2, TileType.STAIRCASE_UP)

    # Go back up to L1.
    state1 = _stair_up(state2, rng)
    assert int(state1.dungeon.current_level) == 1

    # Player should land on the STAIRCASE_DOWN tile of L1 — which is the
    # known position we put STAIRCASE_DOWN at (the original player pos on L1).
    landing_tile = int(
        state1.terrain[int(Branch.MAIN), 0,
                       int(state1.player_pos[0]), int(state1.player_pos[1])]
    )
    assert landing_tile == int(TileType.STAIRCASE_DOWN)


# ---------------------------------------------------------------------------
# test_trapdoor_random_landing
# ---------------------------------------------------------------------------

def test_trapdoor_random_landing():
    """Trapdoor fall lands on a FLOOR tile, not on the STAIRCASE_UP.

    Citation: trap.c::dotrap TRAPDOOR — goto_level with TELEPATH_RANDOM,
    player does NOT land on the stair.
    """
    state, rng = _fresh_state()
    b  = int(state.dungeon.current_branch)

    # Paint destination level (L2) with FLOOR everywhere except one STAIRCASE_UP.
    stair_row, stair_col = 5, 5
    new_terrain = state.terrain
    for r in range(state.terrain.shape[2]):
        for c in range(state.terrain.shape[3]):
            new_terrain = new_terrain.at[b, 1, r, c].set(jnp.int8(int(TileType.FLOOR)))
    new_terrain = new_terrain.at[b, 1, stair_row, stair_col].set(
        jnp.int8(int(TileType.STAIRCASE_UP))
    )
    state = state.replace(terrain=new_terrain)

    new_state = _trap_trapdoor(state, rng)

    assert int(new_state.dungeon.current_level) == 2, "Should descend to level 2"

    land_r = int(new_state.player_pos[0])
    land_c = int(new_state.player_pos[1])
    # Must NOT land on the staircase.
    assert not (land_r == stair_row and land_c == stair_col), (
        f"Trapdoor landed on STAIRCASE_UP at ({stair_row},{stair_col})"
    )
    # Must land on a FLOOR tile.
    assert int(new_state.terrain[b, 1, land_r, land_c]) == int(TileType.FLOOR), (
        "Trapdoor landing tile should be FLOOR"
    )


# ---------------------------------------------------------------------------
# test_branch_transition_to_mines
# ---------------------------------------------------------------------------

def test_branch_transition_to_mines():
    """Cross-branch traversal to GNOMISH_MINES via traverse_stair_cross_branch.

    _stair_down is JIT-safe (within-branch level+1 only); cross-branch
    transitions are handled by traverse_stair_cross_branch which is called by
    env.step when it detects a cross-branch stair link.  This test exercises
    traverse_stair_cross_branch directly with the canonical Mines link from
    init_branch_graph.

    Citation: dungeon.c::init_dungeons — Main Dlvl 3 (down) wired to
              Gnomish Mines Dlvl 1.  do.c::dodown branch-entry path.
    """
    state, rng = _fresh_state()

    # Set player to Main level _MAIN_DLVL_MINES_ENTRY.
    new_dungeon = state.dungeon.replace(
        current_branch=jnp.int8(int(Branch.MAIN)),
        current_level=jnp.int8(_MAIN_DLVL_MINES_ENTRY),
    )
    state = state.replace(dungeon=new_dungeon)

    # Place STAIRCASE_DOWN under player and STAIRCASE_UP on Mines L1 for landing.
    state = _place_stair_under_player(state, TileType.STAIRCASE_DOWN)
    state = _stamp_tile_at(state, int(Branch.GNOMISH_MINES), 1, 10, 10, TileType.STAIRCASE_UP)

    # Wire the branch graph (NethaxEnv.reset leaves stair_links all -1).
    graph = init_branch_graph(rng, None)
    state = state.replace(dungeon=apply_branch_graph_to_dungeon(state.dungeon, graph))

    # Confirm the Mines link is wired at [MAIN, L3-1, down-dir=1].
    dst_b = int(state.dungeon.stair_links[int(Branch.MAIN), _MAIN_DLVL_MINES_ENTRY - 1, 1, 0])
    dst_l = int(state.dungeon.stair_links[int(Branch.MAIN), _MAIN_DLVL_MINES_ENTRY - 1, 1, 1])
    assert dst_b == int(Branch.GNOMISH_MINES), (
        f"Expected Mines link at Main L{_MAIN_DLVL_MINES_ENTRY}, got branch {dst_b}"
    )
    assert dst_l == 1

    # traverse_stair_cross_branch handles cross-branch transitions; direction=+1=down.
    new_state = traverse_stair_cross_branch(state, rng, target_branch=-1, direction=+1)
    assert int(new_state.dungeon.current_branch) == int(Branch.GNOMISH_MINES), (
        "Should have transitioned to GNOMISH_MINES branch"
    )
    assert int(new_state.dungeon.current_level) == 1
