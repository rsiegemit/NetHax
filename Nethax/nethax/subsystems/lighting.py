"""Lighting subsystem — per-source light source registry.

Replaces the single scalar ``lit_radius_until_turn`` with a fixed-capacity
array of light sources, mirroring the ``light_base`` linked list in
vendor/nethack/src/light.c.

Vendor references:
  light.c::new_light_source  (line 62)  — create a light source
  light.c::del_light_source  (line 99)  — delete a light source
  light.c::do_light_sources  (line 169) — propagate lit tiles
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

# Maximum concurrent light sources.  Vendor uses a linked list; we cap at 32.
MAX_LIGHT_SOURCES: int = 32

# source_type sentinels (mirrors vendor light.h LS_* constants).
LTYPE_NONE:    int = 0
LTYPE_TORCH:   int = 1
LTYPE_CANDLE:  int = 2
LTYPE_SPELL:   int = 3
LTYPE_MONSTER: int = 4  # yellow-light monster


@struct.dataclass
class LightingState:
    """Fixed-capacity registry of active light sources.

    All arrays share index [i] for source i.

    Fields:
        source_pos:        int16[MAX, 2]  (row, col) of each source
        source_radius:     int8[MAX]      radius in tiles
        source_type:       int8[MAX]      LTYPE_* constant; 0 = empty slot
        source_owner:      int16[MAX]     monster slot, or -1 for player / fixed
        source_until_turn: int32[MAX]     expiry timestep; -1 = permanent
    """
    source_pos:        jax.Array  # int16[MAX_LIGHT_SOURCES, 2]
    source_radius:     jax.Array  # int8[MAX_LIGHT_SOURCES]
    source_type:       jax.Array  # int8[MAX_LIGHT_SOURCES]
    source_owner:      jax.Array  # int16[MAX_LIGHT_SOURCES]
    source_until_turn: jax.Array  # int32[MAX_LIGHT_SOURCES]

    @classmethod
    def default(cls) -> "LightingState":
        """Return an empty lighting state (no active sources)."""
        return cls(
            source_pos=jnp.zeros((MAX_LIGHT_SOURCES, 2), dtype=jnp.int16),
            source_radius=jnp.zeros((MAX_LIGHT_SOURCES,), dtype=jnp.int8),
            source_type=jnp.zeros((MAX_LIGHT_SOURCES,), dtype=jnp.int8),
            source_owner=jnp.full((MAX_LIGHT_SOURCES,), -1, dtype=jnp.int16),
            source_until_turn=jnp.full((MAX_LIGHT_SOURCES,), -1, dtype=jnp.int32),
        )


# ---------------------------------------------------------------------------
# Core functions — all JIT-pure
# ---------------------------------------------------------------------------

def add_light_source(
    state,
    pos: jax.Array,       # int[2]
    radius: jax.Array,    # int
    ltype: jax.Array,     # int  LTYPE_*
    owner: jax.Array,     # int  monster slot or -1
    duration: jax.Array,  # int  duration in turns; -1 = permanent
) -> object:
    """Register a new light source in the first free slot.

    A slot is free when source_type[i] == LTYPE_NONE (0).
    If all slots are occupied the call is a no-op (capped registry).

    Cite: vendor/nethack/src/light.c::new_light_source (line 62).

    Args:
        state:    EnvState
        pos:      (row, col) of the source
        radius:   light radius in tiles
        ltype:    LTYPE_* constant
        owner:    monster slot index, or -1 for player-held / fixed
        duration: turns until expiry; -1 means permanent

    Returns:
        Updated EnvState.
    """
    lighting = state.lighting
    # Find first free slot (type == 0).
    free_mask = lighting.source_type == jnp.int8(LTYPE_NONE)  # [MAX]
    # Use argmax to get the lowest free index; if none free, idx = 0 but
    # we gate the write behind `any_free`.
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask).astype(jnp.int32)

    until_turn = jnp.where(
        duration == jnp.int32(-1),
        jnp.int32(-1),
        state.timestep.astype(jnp.int32) + duration.astype(jnp.int32),
    )

    new_lighting = jax.lax.cond(
        any_free,
        lambda ls: ls.replace(
            source_pos=ls.source_pos.at[slot].set(pos.astype(jnp.int16)),
            source_radius=ls.source_radius.at[slot].set(jnp.int8(radius)),
            source_type=ls.source_type.at[slot].set(jnp.int8(ltype)),
            source_owner=ls.source_owner.at[slot].set(jnp.int16(owner)),
            source_until_turn=ls.source_until_turn.at[slot].set(until_turn),
        ),
        lambda ls: ls,
        lighting,
    )
    return state.replace(lighting=new_lighting)


def remove_light_source(state, owner: jax.Array) -> object:
    """Clear the slot(s) matching ``owner``.

    Sets source_type to LTYPE_NONE for every slot whose source_owner equals
    ``owner``, mirroring how del_light_source walks the list.

    Cite: vendor/nethack/src/light.c::del_light_source (line 99).

    Args:
        state: EnvState
        owner: monster slot index (or -1 for player-held sources)

    Returns:
        Updated EnvState.
    """
    lighting = state.lighting
    match_mask = lighting.source_owner == owner.astype(jnp.int16)  # [MAX]
    new_type = jnp.where(match_mask, jnp.int8(LTYPE_NONE), lighting.source_type)
    return state.replace(lighting=lighting.replace(source_type=new_type))


def compute_lit_at(state, r: jax.Array, c: jax.Array) -> jax.Array:
    """Return True if tile (r, c) is covered by any active light source.

    A source is active when:
      - source_type != LTYPE_NONE, AND
      - timestep < source_until_turn  OR  source_until_turn == -1.

    Distance check uses Chebyshev (L∞) metric, matching the circle-table
    approach in do_light_sources which marks a square bounding box.

    Cite: vendor/nethack/src/light.c::do_light_sources (line 169).

    JIT-pure: implemented via vectorised operations over the sources axis.
    """
    lighting = state.lighting
    ts = state.timestep.astype(jnp.int32)

    # Active: type != 0 AND (until_turn == -1 OR timestep < until_turn)
    not_empty = lighting.source_type != jnp.int8(LTYPE_NONE)  # [MAX]
    permanent = lighting.source_until_turn == jnp.int32(-1)    # [MAX]
    not_expired = ts < lighting.source_until_turn              # [MAX]
    active = not_empty & (permanent | not_expired)             # [MAX]

    # Chebyshev distance from each source to (r, c).
    dr = jnp.abs(lighting.source_pos[:, 0].astype(jnp.int32) - r.astype(jnp.int32))
    dc = jnp.abs(lighting.source_pos[:, 1].astype(jnp.int32) - c.astype(jnp.int32))
    dist = jnp.maximum(dr, dc)  # [MAX]
    within = dist <= lighting.source_radius.astype(jnp.int32)  # [MAX]

    return jnp.any(active & within)


def tick_light_sources(state) -> object:
    """Expire sources whose until_turn has passed.

    For each slot: if source_until_turn != -1 AND timestep >= until_turn,
    set source_type = LTYPE_NONE.

    Cite: vendor/nethack/src/light.c::do_light_sources (line 169) —
    vendor clears sources after their object is gone; we expire by time.

    Returns:
        Updated EnvState.
    """
    lighting = state.lighting
    ts = state.timestep.astype(jnp.int32)
    timed = lighting.source_until_turn != jnp.int32(-1)   # [MAX]
    expired = timed & (ts >= lighting.source_until_turn)   # [MAX]
    new_type = jnp.where(expired, jnp.int8(LTYPE_NONE), lighting.source_type)
    return state.replace(lighting=lighting.replace(source_type=new_type))


def _player_in_lit_area(state) -> jax.Array:
    """Return True if the player is within range of any active light source.

    Delegates to ``compute_lit_at`` with the player's current position.
    """
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    return compute_lit_at(state, pr, pc)
