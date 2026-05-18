"""Light-source and dark-room FOV parity tests.

Vendor references:
  vendor/nethack/src/vision.c:328  -- rooms[rnum].rlit gates IN_SIGHT vs COULD_SEE
  vendor/nethack/src/light.c:169   -- do_light_sources() propagates TEMP_LIT
  vendor/nethack/src/hack.c:1016   -- Underwater restricts perception
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.action_dispatch import _apply_fov
from Nethax.nethax.constants import TileType

_RNG = jax.random.PRNGKey(0)

# Tile integer values
_FLOOR    = int(TileType.FLOOR)
_CORRIDOR = int(TileType.CORRIDOR)


def _make_open_state(tile: int = _FLOOR, player_pos=(10, 10)):
    """State on an all-`tile` map, player at given position."""
    state = EnvState.default(_RNG)
    terrain = jnp.full(
        state.terrain[0, 0].shape, tile, dtype=state.terrain.dtype
    )
    return state.replace(
        terrain=state.terrain.at[0, 0].set(terrain),
        player_pos=jnp.array(list(player_pos), dtype=jnp.int16),
    )


def _set_spell_lit(state, active: bool):
    """Set lit_radius_until_turn so spell light is active/inactive."""
    ts = state.timestep.astype(jnp.int32)
    deadline = jnp.where(active, ts + jnp.int32(100), jnp.int32(-1))
    new_dungeon = state.dungeon.replace(lit_radius_until_turn=deadline)
    return state.replace(dungeon=new_dungeon)


# ---------------------------------------------------------------------------
# test 1: lit room (FLOOR tile) -> DEFAULT_SIGHT_RADIUS (~7)
# ---------------------------------------------------------------------------

def test_lit_room_default_radius():
    """Player on FLOOR tile should see with full radius ~7.

    Vendor: vision.c:328 -- rlit -> IN_SIGHT for full room.
    """
    state = _make_open_state(tile=_FLOOR)
    new_state = _apply_fov(state)
    visible_count = int(jnp.sum(new_state.visible))
    # Radius-7 open map: (2*7+1)^2 = 225 tiles visible.
    assert visible_count > 9, (
        f"Lit room expected >9 visible tiles, got {visible_count}."
    )


# ---------------------------------------------------------------------------
# test 2: dark room (CORRIDOR tile) -> DARK_ROOM_SIGHT_RADIUS (~2)
# ---------------------------------------------------------------------------

def test_dark_room_reduced_radius():
    """Player on CORRIDOR tile without light should see radius ~2.

    Vendor: vision.c:328 -- dark room sets COULD_SEE only (adjacent).
    """
    state = _make_open_state(tile=_CORRIDOR)
    new_state = _apply_fov(state)
    visible_count = int(jnp.sum(new_state.visible))
    # Radius-2: (2*2+1)^2 = 25 tiles maximum.
    assert visible_count <= 25, (
        f"Dark room expected <=25 visible tiles (radius 2), got {visible_count}."
    )
    # Must be clearly less than full-radius view.
    assert visible_count < 100, (
        f"Dark room radius should not be close to full sight, got {visible_count}."
    )


# ---------------------------------------------------------------------------
# test 3: dark room with active light spell -> DEFAULT_SIGHT_RADIUS
# ---------------------------------------------------------------------------

def test_dark_room_with_light_spell():
    """Player in dark corridor with active spell light should see radius ~7.

    Vendor: light.c::do_light_sources line 169 -- hero's light source
    extends TEMP_LIT to COULD_SEE tiles.
    """
    state = _make_open_state(tile=_CORRIDOR)
    state = _set_spell_lit(state, active=True)
    new_state = _apply_fov(state)
    visible_count = int(jnp.sum(new_state.visible))
    # Spell restores full radius -> (2*7+1)^2 = 225.
    assert visible_count > 9, (
        f"Dark room + spell light expected >9 visible tiles, got {visible_count}."
    )


# ---------------------------------------------------------------------------
# test 4: underwater overrides lit-room radius -> BLIND_SIGHT_RADIUS
# ---------------------------------------------------------------------------

def test_underwater_overrides_lit_radius():
    """In water + lit room -> radius 1 (at most 3x3 = 9 tiles).

    Vendor: hack.c:1016 -- Underwater restricts perception.
    """
    state = _make_open_state(tile=_FLOOR)
    state = state.replace(player_in_water=jnp.bool_(True))
    new_state = _apply_fov(state)
    visible_count = int(jnp.sum(new_state.visible))
    assert visible_count <= 9, (
        f"Underwater FOV expected <=9 tiles (radius 1), got {visible_count}."
    )
