"""Lighting subsystem parity tests.

Vendor references:
  vendor/nethack/src/light.c::new_light_source (line 62)
  vendor/nethack/src/light.c::del_light_source  (line 99)
  vendor/nethack/src/light.c::do_light_sources  (line 169)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.lighting import (
    LightingState,
    add_light_source,
    remove_light_source,
    compute_lit_at,
    tick_light_sources,
    MAX_LIGHT_SOURCES,
    LTYPE_TORCH,
    LTYPE_CANDLE,
    LTYPE_NONE,
)
import jax

_RNG = jax.random.PRNGKey(42)


def _base_state():
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# test 1: add_light_source writes the first free slot correctly
# ---------------------------------------------------------------------------

def test_add_light_source_writes_slot():
    """After adding one source, slot 0 holds the correct type and position.

    Cite: light.c::new_light_source line 62.
    """
    state = _base_state()
    pos = jnp.array([5, 7], dtype=jnp.int16)
    state2 = add_light_source(
        state,
        pos=pos,
        radius=jnp.int8(3),
        ltype=jnp.int8(LTYPE_TORCH),
        owner=jnp.int16(-1),
        duration=jnp.int32(-1),
    )
    assert int(state2.lighting.source_type[0]) == LTYPE_TORCH
    assert int(state2.lighting.source_pos[0, 0]) == 5
    assert int(state2.lighting.source_pos[0, 1]) == 7


# ---------------------------------------------------------------------------
# test 2: tile within source radius is lit
# ---------------------------------------------------------------------------

def test_compute_lit_at_within_radius():
    """Source at (5,5) radius 3 should light tile (6,6).

    Cite: light.c::do_light_sources line 169.
    """
    state = _base_state()
    state = add_light_source(
        state,
        pos=jnp.array([5, 5], dtype=jnp.int16),
        radius=jnp.int8(3),
        ltype=jnp.int8(LTYPE_TORCH),
        owner=jnp.int16(-1),
        duration=jnp.int32(-1),
    )
    lit = compute_lit_at(state, jnp.int32(6), jnp.int32(6))
    assert bool(lit), "Tile (6,6) should be lit by source at (5,5) radius 3"


# ---------------------------------------------------------------------------
# test 3: tile beyond radius is not lit
# ---------------------------------------------------------------------------

def test_compute_lit_at_beyond_radius():
    """Source at (5,5) radius 3 should NOT light tile (10,10).

    Cite: light.c::do_light_sources line 169.
    """
    state = _base_state()
    state = add_light_source(
        state,
        pos=jnp.array([5, 5], dtype=jnp.int16),
        radius=jnp.int8(3),
        ltype=jnp.int8(LTYPE_TORCH),
        owner=jnp.int16(-1),
        duration=jnp.int32(-1),
    )
    lit = compute_lit_at(state, jnp.int32(10), jnp.int32(10))
    assert not bool(lit), "Tile (10,10) should NOT be lit by source at (5,5) radius 3"


# ---------------------------------------------------------------------------
# test 4: expired source is cleared by tick_light_sources
# ---------------------------------------------------------------------------

def test_lit_source_expires():
    """Source with until_turn=5 at timestep=6 should be cleared.

    Cite: light.c::do_light_sources line 169 — vendor removes dead sources.
    """
    state = _base_state()
    # Force timestep to 4 so add_light_source sets until_turn = 4+1 = 5.
    state = state.replace(timestep=jnp.int32(4))
    state = add_light_source(
        state,
        pos=jnp.array([5, 5], dtype=jnp.int16),
        radius=jnp.int8(3),
        ltype=jnp.int8(LTYPE_TORCH),
        owner=jnp.int16(-1),
        duration=jnp.int32(1),   # expires at turn 4+1 = 5
    )
    assert int(state.lighting.source_until_turn[0]) == 5

    # Advance timestep to 6, then tick.
    state = state.replace(timestep=jnp.int32(6))
    state = tick_light_sources(state)
    assert int(state.lighting.source_type[0]) == LTYPE_NONE, (
        "Expired source should have type LTYPE_NONE after tick"
    )


# ---------------------------------------------------------------------------
# test 5: capacity — 32 sources fill all slots; 33rd is ignored
# ---------------------------------------------------------------------------

def test_max_sources_capacity():
    """Adding MAX_LIGHT_SOURCES sources fills all slots; the 33rd is a no-op.

    Cite: design cap of 32 (MAX_LIGHT_SOURCES).
    """
    state = _base_state()
    for i in range(MAX_LIGHT_SOURCES):
        state = add_light_source(
            state,
            pos=jnp.array([i % 21, i % 80], dtype=jnp.int16),
            radius=jnp.int8(2),
            ltype=jnp.int8(LTYPE_CANDLE),
            owner=jnp.int16(i),
            duration=jnp.int32(-1),
        )

    active = jnp.sum(state.lighting.source_type != jnp.int8(LTYPE_NONE))
    assert int(active) == MAX_LIGHT_SOURCES, (
        f"Expected {MAX_LIGHT_SOURCES} active sources, got {int(active)}"
    )

    # 33rd add should be a no-op (no free slot).
    state_before = state
    state = add_light_source(
        state,
        pos=jnp.array([0, 0], dtype=jnp.int16),
        radius=jnp.int8(1),
        ltype=jnp.int8(LTYPE_TORCH),
        owner=jnp.int16(-1),
        duration=jnp.int32(-1),
    )
    active_after = jnp.sum(state.lighting.source_type != jnp.int8(LTYPE_NONE))
    assert int(active_after) == MAX_LIGHT_SOURCES, (
        "33rd add should not increase active count beyond MAX_LIGHT_SOURCES"
    )
