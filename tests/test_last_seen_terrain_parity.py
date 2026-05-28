"""Parity tests for last_seen_terrain stale glyph layer.

Vendor reference: vendor/nethack/src/display.c::lastseentyp[x][y] (~line 850)
— stores the terrain type last visible at each cell so off-FOV explored tiles
render their last-known state rather than live truth.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.glyphs import NO_GLYPH, GLYPH_CMAP_OFF
from Nethax.nethax.obs.nle_obs import build_glyphs

_RNG = jax.random.PRNGKey(99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_state() -> EnvState:
    return EnvState.default(_RNG)


def _branch_lv(state):
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    return b, lv


def _set_terrain(state, row, col, tile):
    b, lv = _branch_lv(state)
    new_terrain = state.terrain.at[b, lv, row, col].set(jnp.int8(tile))
    return state.replace(terrain=new_terrain)


def _mark_explored(state, row, col):
    b, lv = _branch_lv(state)
    new_explored = state.explored.at[b, lv, row, col].set(True)
    return state.replace(explored=new_explored)


def _set_visible(state, row, col, val=True):
    new_visible = state.visible.at[row, col].set(jnp.bool_(val))
    return state.replace(visible=new_visible)


def _set_last_seen(state, row, col, tile):
    b, lv = _branch_lv(state)
    new_lst = state.last_seen_terrain.at[b, lv, row, col].set(jnp.int8(tile))
    return state.replace(last_seen_terrain=new_lst)


def _no_glyph_val():
    return int(jnp.int16(NO_GLYPH & 0xFFFF))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLastSeenTerrain:
    def test_explored_door_stale(self):
        """Explored-but-not-visible tile shows last_seen value, not live terrain.

        Scenario:
          1. Player previously saw a CLOSED_DOOR at (8, 10) — last_seen set.
          2. Terrain is mutated to OPEN_DOOR (simulating monster opening it).
          3. Tile is explored but NOT currently visible.
          4. build_glyphs should render the CLOSED_DOOR glyph (stale memory).

        Vendor: display.c lastseentyp — off-FOV tiles use the cached type.
        """
        state = _default_state()

        # Step 1: set last_seen to CLOSED_DOOR (what the player last saw)
        state = _set_last_seen(state, 8, 10, TileType.CLOSED_DOOR)
        # Step 2: live terrain is now OPEN_DOOR (monster opened it)
        state = _set_terrain(state, 8, 10, TileType.OPEN_DOOR)
        # Step 3: explored = True, visible = False (tile is off-FOV)
        state = _mark_explored(state, 8, 10)
        # visible array defaults to all-False; ensure (8,10) is not visible
        state = state.replace(visible=jnp.zeros_like(state.visible))

        glyphs = build_glyphs(state)
        glyph_val = int(glyphs[8, 10])

        # Should NOT be NO_GLYPH (tile is explored)
        assert glyph_val != _no_glyph_val(), "Explored tile should not show NO_GLYPH"

        # Should render CLOSED_DOOR (the stale memory), not OPEN_DOOR
        # We verify by building glyphs for the live-truth scenario and checking
        # they differ — the stale glyph should match what CLOSED_DOOR maps to.
        state_live = _default_state()
        state_live = _set_terrain(state_live, 8, 10, TileType.CLOSED_DOOR)
        state_live = _mark_explored(state_live, 8, 10)
        state_live = _set_last_seen(state_live, 8, 10, TileType.CLOSED_DOOR)
        state_live = state_live.replace(visible=jnp.zeros_like(state_live.visible))
        expected_glyph = int(build_glyphs(state_live)[8, 10])

        assert glyph_val == expected_glyph, (
            f"Stale glyph {glyph_val} should match CLOSED_DOOR glyph {expected_glyph}"
        )

    def test_visible_door_current(self):
        """Currently-visible tile renders live terrain, not last_seen.

        Same scenario as above but tile IS currently visible → shows OPEN_DOOR.
        """
        state = _default_state()

        # last_seen says CLOSED_DOOR
        state = _set_last_seen(state, 8, 10, TileType.CLOSED_DOOR)
        # live terrain is OPEN_DOOR
        state = _set_terrain(state, 8, 10, TileType.OPEN_DOOR)
        # explored AND visible
        state = _mark_explored(state, 8, 10)
        state = _set_visible(state, 8, 10, True)

        glyphs = build_glyphs(state)
        glyph_val = int(glyphs[8, 10])

        # Build a reference for what OPEN_DOOR looks like when visible
        state_ref = _default_state()
        state_ref = _set_terrain(state_ref, 8, 10, TileType.OPEN_DOOR)
        state_ref = _mark_explored(state_ref, 8, 10)
        state_ref = _set_last_seen(state_ref, 8, 10, TileType.OPEN_DOOR)
        state_ref = _set_visible(state_ref, 8, 10, True)
        expected_open_glyph = int(build_glyphs(state_ref)[8, 10])

        assert glyph_val == expected_open_glyph, (
            f"Visible tile glyph {glyph_val} should match live OPEN_DOOR "
            f"glyph {expected_open_glyph}"
        )

    def test_unexplored_stone_glyph(self):
        """Unexplored tiles render as cmap_to_glyph(S_stone) (= GLYPH_CMAP_OFF
        + 0 = 2359), NOT NO_GLYPH.

        NLE's RL window seeds the entire ``glyphs`` obs array with
        ``nul_glyph = cmap_to_glyph(S_stone)`` and re-fills any never-seen /
        background cell with the same value.  NO_GLYPH (== MAX_GLYPH == 5976)
        is reserved for inventory slots / internal sentinels and is never
        written into the map observation.

        Vendor cite: vendor/nle/win/rl/winrl.cc:61
        (``const int nul_glyph = cmap_to_glyph(S_stone);``) + winrl.cc:250,304,646
        (``glyphs_.fill(nul_glyph)`` / ``std::fill_n(obs->glyphs, ..., nul_glyph)``)
        and vendor/nethack/src/display.c:436
        (``levl[x][y].glyph = cmap_to_glyph(S_stone); /* default val */``).
        """
        state = _default_state()

        # Set some non-trivial values but do NOT mark explored
        state = _set_terrain(state, 5, 5, TileType.FLOOR)
        state = _set_last_seen(state, 5, 5, TileType.FLOOR)
        # Not explored, not visible

        glyphs = build_glyphs(state)
        glyph_val = int(glyphs[5, 5])

        expected = GLYPH_CMAP_OFF  # cmap_to_glyph(S_stone), S_stone == 0
        assert glyph_val == expected, (
            f"Unexplored tile should be cmap_to_glyph(S_stone) ({expected}), "
            f"got {glyph_val}"
        )
        assert glyph_val != _no_glyph_val(), (
            "Unexplored map cell must NOT be NO_GLYPH; NLE writes nul_glyph "
            "(S_stone) into the glyphs obs array, never NO_GLYPH."
        )

    def test_last_seen_sentinel_explored_not_visible(self):
        """Explored tile with last_seen=-1 (never stamped) renders as stone/void.

        When last_seen_terrain=-1 (sentinel), we fall back to tile 0 (VOID/stone),
        which maps to a valid cmap glyph (not NO_GLYPH, since tile is explored).
        """
        state = _default_state()

        # Mark explored but leave last_seen at -1 (default sentinel)
        state = _mark_explored(state, 7, 7)
        state = state.replace(visible=jnp.zeros_like(state.visible))

        b, lv = _branch_lv(state)
        last_seen_val = int(state.last_seen_terrain[b, lv, 7, 7])
        assert last_seen_val == -1, "Sentinel should be -1 at initialization"

        glyphs = build_glyphs(state)
        glyph_val = int(glyphs[7, 7])

        # Explored tile must not be NO_GLYPH — sentinel maps to tile 0 (stone)
        assert glyph_val != _no_glyph_val(), (
            "Explored tile with sentinel last_seen should not render as NO_GLYPH"
        )

    def test_apply_fov_stamps_last_seen(self):
        """_apply_fov stamps currently-visible terrain into last_seen_terrain."""
        from Nethax.nethax.subsystems.action_dispatch import _apply_fov

        state = _default_state()
        b, lv = _branch_lv(state)

        # Set up terrain and put player in FOV of a FLOOR tile
        state = state.replace(
            player_pos=jnp.array([10, 10], dtype=jnp.int16),
        )
        state = _set_terrain(state, 10, 10, TileType.FLOOR)

        # Confirm last_seen starts at sentinel
        assert int(state.last_seen_terrain[b, lv, 10, 10]) == -1

        new_state = _apply_fov(state)

        # After FOV, player's own tile should be stamped
        stamped = int(new_state.last_seen_terrain[b, lv, 10, 10])
        assert stamped == int(TileType.FLOOR), (
            f"_apply_fov should stamp visible tile terrain={int(TileType.FLOOR)}, "
            f"got {stamped}"
        )
