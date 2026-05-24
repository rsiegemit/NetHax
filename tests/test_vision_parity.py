"""Parity tests for the LoS helpers in subsystems/vision.py.

Canonical sources:
  vendor/nethack/include/vision.h
      ``#define cansee(x, y)``   (line 28)
      ``#define couldsee(x, y)`` (line 29)
      ``#define m_cansee(...)``  (line 42)
  vendor/nethack/src/vision.c
      ``does_block()``           (lines 152-202)
      ``clear_path()``           (lines 1612-1636)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems import vision as _vision
from Nethax.nethax.subsystems.status_effects import TimedStatus


RNG = jax.random.PRNGKey(0)


def _state():
    return EnvState.default(RNG, StaticParams())


def _open_level(state):
    """Return a state whose current level is entirely FLOOR."""
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    H, W = state.terrain.shape[2], state.terrain.shape[3]
    new_plane = jnp.full((H, W), jnp.int8(int(TileType.FLOOR)), dtype=state.terrain.dtype)
    new_terrain = state.terrain.at[b, lv].set(new_plane)
    return state.replace(terrain=new_terrain)


def _set_tile(state, r, c, tile):
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    plane = state.terrain[b, lv]
    plane = plane.at[r, c].set(jnp.int8(int(tile)))
    new_terrain = state.terrain.at[b, lv].set(plane)
    return state.replace(terrain=new_terrain)


def _move_player(state, r, c):
    return state.replace(player_pos=jnp.array([r, c], dtype=state.player_pos.dtype))


# ---------------------------------------------------------------------------
# clear_path
# ---------------------------------------------------------------------------

class TestClearPath:
    def test_same_tile_is_clear(self):
        """row1==row2 && col1==col2 -> result = 1 (vision.c line 1626-1627)."""
        s = _open_level(_state())
        ok = _vision.clear_path(s, jnp.int32(5), jnp.int32(5), jnp.int32(5), jnp.int32(5))
        assert bool(ok)

    def test_adjacent_tile_is_clear(self):
        """Adjacent endpoints have no intermediate tile; trivially clear."""
        s = _open_level(_state())
        ok = _vision.clear_path(s, jnp.int32(5), jnp.int32(5), jnp.int32(5), jnp.int32(6))
        assert bool(ok)

    def test_open_floor_line_is_clear(self):
        """A straight line over FLOOR tiles is unblocked."""
        s = _open_level(_state())
        ok = _vision.clear_path(s, jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(10))
        assert bool(ok)

    def test_wall_between_blocks(self):
        """A WALL between the two endpoints blocks the line."""
        s = _open_level(_state())
        s = _set_tile(s, 3, 6, TileType.WALL)
        ok = _vision.clear_path(s, jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(10))
        assert not bool(ok)

    def test_closed_door_between_blocks(self):
        """A CLOSED_DOOR between the two endpoints blocks the line.

        Cite: vision.c::does_block line 167-168 (IS_DOOR with D_CLOSED).
        """
        s = _open_level(_state())
        s = _set_tile(s, 3, 6, TileType.CLOSED_DOOR)
        ok = _vision.clear_path(s, jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(10))
        assert not bool(ok)

    def test_tree_between_blocks(self):
        """A TREE between the two endpoints blocks the line.

        Cite: vision.c::does_block line 166 (lev->typ == TREE).
        """
        s = _open_level(_state())
        s = _set_tile(s, 3, 6, TileType.TREE)
        ok = _vision.clear_path(s, jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(10))
        assert not bool(ok)

    def test_endpoints_not_checked(self):
        """Endpoints are NOT tested (vendor: vision.c line 1191).

        A WALL at the destination cell should not by itself block the path.
        """
        s = _open_level(_state())
        s = _set_tile(s, 3, 10, TileType.WALL)
        ok = _vision.clear_path(s, jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(10))
        assert bool(ok)

    def test_diagonal_clear(self):
        """A pure-diagonal line over FLOOR is unblocked."""
        s = _open_level(_state())
        ok = _vision.clear_path(s, jnp.int32(2), jnp.int32(2), jnp.int32(7), jnp.int32(7))
        assert bool(ok)

    def test_diagonal_wall_blocks(self):
        """A diagonal line through a wall is blocked."""
        s = _open_level(_state())
        s = _set_tile(s, 5, 5, TileType.WALL)
        ok = _vision.clear_path(s, jnp.int32(2), jnp.int32(2), jnp.int32(7), jnp.int32(7))
        assert not bool(ok)


# ---------------------------------------------------------------------------
# couldsee / cansee
# ---------------------------------------------------------------------------

class TestCouldsee:
    def test_couldsee_open_floor(self):
        """couldsee(state, r, c) is True when LoS is unobstructed."""
        s = _open_level(_state())
        s = _move_player(s, 5, 5)
        assert bool(_vision.couldsee(s, jnp.int32(5), jnp.int32(12)))

    def test_couldsee_wall_blocks(self):
        """A WALL on the line breaks couldsee."""
        s = _open_level(_state())
        s = _move_player(s, 5, 5)
        s = _set_tile(s, 5, 8, TileType.WALL)
        assert not bool(_vision.couldsee(s, jnp.int32(5), jnp.int32(12)))

    def test_couldsee_player_self(self):
        """Player's own tile is trivially seen (same-tile vendor branch)."""
        s = _open_level(_state())
        s = _move_player(s, 5, 5)
        assert bool(_vision.couldsee(s, jnp.int32(5), jnp.int32(5)))


class TestCansee:
    def test_cansee_open(self):
        """cansee(from, to) over an open path is True."""
        s = _open_level(_state())
        ok = _vision.cansee(s, jnp.int32(4), jnp.int32(4),
                               jnp.int32(4), jnp.int32(12))
        assert bool(ok)

    def test_cansee_blocked(self):
        """cansee(from, to) is False when a wall sits on the line."""
        s = _open_level(_state())
        s = _set_tile(s, 4, 8, TileType.WALL)
        ok = _vision.cansee(s, jnp.int32(4), jnp.int32(4),
                               jnp.int32(4), jnp.int32(12))
        assert not bool(ok)


# ---------------------------------------------------------------------------
# cansee_with_blind -- blind player can never see remote tiles
# ---------------------------------------------------------------------------

class TestCanseeWithBlind:
    def test_sighted_sees_open(self):
        s = _open_level(_state())
        s = _move_player(s, 5, 5)
        assert bool(_vision.cansee_with_blind(s, jnp.int32(5), jnp.int32(10)))

    def test_blind_cannot_see(self):
        """When BLIND timer > 0, cansee_with_blind is always False even with
        a clear geometric path.

        Cite: vendor display.h ``canseemon`` folds ``!Blind`` into ``cansee``.
        """
        s = _open_level(_state())
        s = _move_player(s, 5, 5)
        new_status = s.status.replace(
            timed_statuses=s.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(50)),
        )
        s = s.replace(status=new_status)
        assert not bool(_vision.cansee_with_blind(s, jnp.int32(5), jnp.int32(10)))

    def test_blind_does_not_break_couldsee(self):
        """couldsee ignores blindness (vendor: COULD_SEE bit, vision.h:29)."""
        s = _open_level(_state())
        s = _move_player(s, 5, 5)
        new_status = s.status.replace(
            timed_statuses=s.status.timed_statuses.at[int(TimedStatus.BLIND)].set(jnp.int32(50)),
        )
        s = s.replace(status=new_status)
        assert bool(_vision.couldsee(s, jnp.int32(5), jnp.int32(10)))


# ---------------------------------------------------------------------------
# JIT compatibility -- helpers must be tracer-safe.
# ---------------------------------------------------------------------------

class TestJitSafety:
    def test_clear_path_jits(self):
        s = _open_level(_state())

        @jax.jit
        def f(state, r0, c0, r1, c1):
            return _vision.clear_path(state, r0, c0, r1, c1)

        ok = f(s, jnp.int32(3), jnp.int32(3), jnp.int32(3), jnp.int32(7))
        assert bool(ok)

    def test_cansee_with_blind_jits(self):
        s = _open_level(_state())
        s = _move_player(s, 5, 5)

        @jax.jit
        def f(state, r, c):
            return _vision.cansee_with_blind(state, r, c)

        ok = f(s, jnp.int32(5), jnp.int32(10))
        assert bool(ok)


# ---------------------------------------------------------------------------
# Re-export sanity -- detect.py exposes the same helpers for callers that
# already import from detect (advisory; the canonical home is vision.py).
# ---------------------------------------------------------------------------

class TestDetectReExports:
    def test_detect_module_reexports_vision_helpers(self):
        from Nethax.nethax.subsystems import detect as _detect
        assert _detect.clear_path is _vision.clear_path
        assert _detect.cansee is _vision.cansee
        assert _detect.couldsee is _vision.couldsee
        assert _detect.cansee_with_blind is _vision.cansee_with_blind
