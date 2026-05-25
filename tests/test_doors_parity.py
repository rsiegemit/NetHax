"""Wave 6 closing-audit parity tests for door / secret-door / drawbridge ops.

Each test cites the vendor source it asserts against:

  open_door, close_door            — vendor/nethack/src/lock.c::doopen, doclose
  kick_door                        — vendor/nethack/src/dokick.c::kick_door
  picklock_door, forcelock_door    — vendor/nethack/src/lock.c::picklock, forcelock
  handle_search / SECRET discovery — vendor/nethack/src/detect.c::dosearch0
  destroy_drawbridge               — vendor/nethack/src/dbridge.c::destroy_drawbridge
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.features import (
    DoorState,
    open_door,
    close_door,
    kick_door,
    unlock_door,
    picklock_door,
    forcelock_door,
    discover_secret_door,
    destroy_drawbridge,
    handle_search,
)


_RNG = jax.random.PRNGKey(2026)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 5, player_col: int = 5) -> EnvState:
    s = EnvState.default(_RNG)
    return s.replace(player_pos=jnp.array([player_row, player_col], dtype=jnp.int16))


def _flat_lv(state: EnvState) -> int:
    b      = int(state.dungeon.current_branch)
    lv     = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _door_pos(state: EnvState, row: int, col: int) -> jnp.ndarray:
    return jnp.array([_flat_lv(state), row, col], dtype=jnp.int32)


def _set_door(state: EnvState, row: int, col: int, ds: DoorState) -> EnvState:
    new_door = state.features.door_state.at[_flat_lv(state), row, col].set(jnp.int8(ds))
    return state.replace(features=state.features.replace(door_state=new_door))


def _get_door(state: EnvState, row: int, col: int) -> int:
    return int(state.features.door_state[_flat_lv(state), row, col])


# ---------------------------------------------------------------------------
# open_door / close_door — vendor lock.c::doopen, doclose
# ---------------------------------------------------------------------------

def test_open_closed_becomes_isopen():
    """vendor lock.c::doopen lines 903-913: CLOSED + open success → D_ISOPEN."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.CLOSED)
    new_features, _dmg = open_door(s.features, _door_pos(s, 4, 5))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.OPEN)


def test_open_locked_fails():
    """vendor lock.c::doopen lines 855-895: LOCKED → 'door is locked' message,
    doormask unchanged.  Our open_door must leave LOCKED intact."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.LOCKED)
    new_features, _dmg = open_door(s.features, _door_pos(s, 4, 5))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.LOCKED)


def test_close_isopen_becomes_closed():
    """vendor lock.c::doclose lines 1033-1043: D_ISOPEN + close success → D_CLOSED."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.OPEN)
    new_features = close_door(s.features, _door_pos(s, 4, 5), jnp.bool_(False))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.CLOSED)


def test_close_blocked_by_monster_fails():
    """vendor lock.c::doclose lines 1023-1024: obstructed(x, y) bails out before
    setting doormask.  Closing while blocked must leave the door OPEN."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.OPEN)
    new_features = close_door(s.features, _door_pos(s, 4, 5), jnp.bool_(True))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.OPEN)


# ---------------------------------------------------------------------------
# kick_door — vendor dokick.c::kick_door
# ---------------------------------------------------------------------------

def test_kick_closed_25_percent_breaks():
    """vendor dokick.c::kick_door — break rate per spec is 1/4 per kick.
    Run 200 seeded kicks; expect break count near 50 with generous slack."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.CLOSED)
    pos = _door_pos(s, 4, 5)

    breaks = 0
    keys = jax.random.split(_RNG, 200)
    for i in range(200):
        new_features, _ = kick_door(s.features, keys[i], pos)
        if int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.BROKEN):
            breaks += 1
    # Expectation 50, ±25 for 200 trials at p=0.25 (well within Chernoff).
    assert 25 <= breaks <= 75, f"Expected ~50 breaks, got {breaks}/200"


def test_kick_locked_can_break_lock():
    """vendor dokick.c::kick_door — LOCKED takes the same 1/4 break path; we
    additionally model the 1/8 chance the lock alone snaps (LOCKED → CLOSED).
    Across 400 trials we should see BOTH BROKEN and CLOSED outcomes appear."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.LOCKED)
    pos = _door_pos(s, 4, 5)

    seen_broken = False
    seen_closed = False
    seen_locked = False
    keys = jax.random.split(_RNG, 400)
    for i in range(400):
        new_features, _ = kick_door(s.features, keys[i], pos)
        v = int(new_features.door_state[_flat_lv(s), 4, 5])
        if v == int(DoorState.BROKEN):
            seen_broken = True
        elif v == int(DoorState.CLOSED):
            seen_closed = True
        elif v == int(DoorState.LOCKED):
            seen_locked = True
    assert seen_broken, "kicking LOCKED never produced BROKEN over 400 trials"
    assert seen_closed, "kicking LOCKED never snapped the lock (LOCKED→CLOSED) over 400 trials"
    assert seen_locked, "kicking LOCKED never failed (LOCKED unchanged) over 400 trials"


def test_kick_isopen_noop():
    """vendor dokick.c::kick_door lines 914-918 — D_ISOPEN routes to kick_dumb
    and returns without touching doormask.  OPEN must remain OPEN."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.OPEN)
    pos = _door_pos(s, 4, 5)
    # Run many seeds so any stray write path would surface as a failure.
    keys = jax.random.split(_RNG, 50)
    for i in range(50):
        new_features, _ = kick_door(s.features, keys[i], pos)
        assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.OPEN), (
            f"OPEN must stay OPEN; got {int(new_features.door_state[_flat_lv(s), 4, 5])} at seed {i}"
        )


# ---------------------------------------------------------------------------
# picklock_door / forcelock_door — vendor lock.c::picklock, forcelock
# ---------------------------------------------------------------------------

def test_picklock_locked_becomes_closed():
    """vendor lock.c::picklock lines 138-150: success branch sets
    door->doormask = D_CLOSED (when previously D_LOCKED)."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.LOCKED)
    new_features, success = picklock_door(s.features, _door_pos(s, 4, 5))
    assert bool(success)
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.CLOSED)


def test_forcelock_locked_becomes_broken():
    """vendor lock.c::forcelock + breakchestlock: forced lock leaves the door
    BROKEN (lock destroyed, door swings free)."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.LOCKED)
    new_features, success = forcelock_door(s.features, _door_pos(s, 4, 5))
    assert bool(success)
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.BROKEN)


# ---------------------------------------------------------------------------
# Secret door SEARCH discovery — vendor detect.c::dosearch0
# ---------------------------------------------------------------------------

def test_secret_door_search_discovers():
    """vendor detect.c::dosearch0 lines 2042-2051: rnl(7-fund); on 0 reveal.
    With a SECRET door at (4,5) adjacent to player(5,5), 100 seeded SEARCH
    calls should reveal it many times (binomial p=1/7 → ≈ 14 expected)."""
    s = _make_state(player_row=5, player_col=5)
    s = _set_door(s, 4, 5, DoorState.SECRET)

    discovered = 0
    keys = jax.random.split(_RNG, 100)
    for i in range(100):
        out = handle_search(s, keys[i])
        if int(out.features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.CLOSED):
            discovered += 1
    # E[X] = 100/7 ≈ 14; demand at least 3 to catch a stuck no-op.
    assert discovered >= 3, f"Expected many discoveries, got {discovered}/100"
    assert discovered <= 60, f"Too many — sanity check failed: {discovered}/100"


def test_secret_door_discovered_becomes_closed():
    """vendor detect.c::dosearch0 + cvt_sdoor_to_door: revealed SDOOR becomes
    a normal door retaining its lock/closed mask.  Our discover_secret_door
    maps SECRET → CLOSED unconditionally."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.SECRET)
    new_features = discover_secret_door(s.features, _door_pos(s, 4, 5))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.CLOSED)


def test_secret_door_search_non_adjacent_unchanged():
    """vendor detect.c::dosearch0: the 3x3 sweep only touches u.ux ± 1.
    A SECRET door 4 tiles away must NEVER be revealed by SEARCH."""
    s = _make_state(player_row=5, player_col=5)
    s = _set_door(s, 10, 10, DoorState.SECRET)
    keys = jax.random.split(_RNG, 50)
    for i in range(50):
        out = handle_search(s, keys[i])
        assert int(out.features.door_state[_flat_lv(s), 10, 10]) == int(DoorState.SECRET)


# ---------------------------------------------------------------------------
# Drawbridge — vendor dbridge.c::destroy_drawbridge
# ---------------------------------------------------------------------------

def test_drawbridge_destroy_clears_tiles():
    """vendor dbridge.c::destroy_drawbridge lines 888-960: the two halves of
    the drawbridge (the bridge tile and its paired wall) become non-bridge
    terrain (FLOOR / ROOM in our abstraction).  Asserts the (row, col) tile
    no longer reports as DRAWBRIDGE_UP after destroy_drawbridge."""
    s = _make_state()
    b  = int(s.dungeon.current_branch)
    lv = int(s.dungeon.current_level) - 1

    # Plant a drawbridge spanning (3,3) and its eastern neighbour (3,4).
    terrain = s.terrain.at[b, lv, 3, 3].set(jnp.int8(int(TileType.DRAWBRIDGE_UP)))
    terrain = terrain.at[b, lv, 3, 4].set(jnp.int8(int(TileType.DRAWBRIDGE_UP)))
    s = s.replace(terrain=terrain)

    pos = jnp.array([b, lv, 3, 3], dtype=jnp.int32)
    _, new_terrain = destroy_drawbridge(s.features, s.terrain, pos)

    assert int(new_terrain[b, lv, 3, 3]) != int(TileType.DRAWBRIDGE_UP), (
        f"Bridge tile (3,3) should be cleared, still got {int(new_terrain[b, lv, 3, 3])}"
    )
    assert int(new_terrain[b, lv, 3, 4]) != int(TileType.DRAWBRIDGE_UP), (
        f"Paired wall tile (3,4) should be cleared, still got {int(new_terrain[b, lv, 3, 4])}"
    )


def test_drawbridge_destroy_preserves_other_tiles():
    """vendor dbridge.c::destroy_drawbridge only touches the bridge halves —
    surrounding terrain (walls, floor) remains intact.  We seed a WALL at
    (1,1) and a FLOOR at (10,10), destroy a far-away drawbridge, and assert
    those tiles are unchanged."""
    s = _make_state()
    b  = int(s.dungeon.current_branch)
    lv = int(s.dungeon.current_level) - 1

    terrain = s.terrain.at[b, lv, 5, 5].set(jnp.int8(int(TileType.DRAWBRIDGE_UP)))
    terrain = terrain.at[b, lv, 1, 1].set(jnp.int8(int(TileType.WALL)))
    terrain = terrain.at[b, lv, 10, 10].set(jnp.int8(int(TileType.FLOOR)))
    s = s.replace(terrain=terrain)

    pos = jnp.array([b, lv, 5, 5], dtype=jnp.int32)
    _, new_terrain = destroy_drawbridge(s.features, s.terrain, pos)

    assert int(new_terrain[b, lv, 1, 1]) == int(TileType.WALL)
    assert int(new_terrain[b, lv, 10, 10]) == int(TileType.FLOOR)


# ---------------------------------------------------------------------------
# Misc parity — open/close on edge states
# ---------------------------------------------------------------------------

def test_open_broken_unchanged():
    """vendor lock.c::doopen lines 859-861: D_BROKEN → 'is broken' message,
    no state change.  Our open_door must not touch BROKEN doors."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.BROKEN)
    new_features, _dmg = open_door(s.features, _door_pos(s, 4, 5))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.BROKEN)


def test_close_broken_unchanged():
    """vendor lock.c::doclose lines 1025-1027: D_BROKEN → 'door is broken',
    return without state change."""
    s = _make_state()
    s = _set_door(s, 4, 5, DoorState.BROKEN)
    new_features = close_door(s.features, _door_pos(s, 4, 5), jnp.bool_(False))
    assert int(new_features.door_state[_flat_lv(s), 4, 5]) == int(DoorState.BROKEN)
