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


def _has_los(terrain_2d: jax.Array,
             sr: jax.Array, sc: jax.Array,
             tr: jax.Array, tc: jax.Array) -> jax.Array:
    """Bresenham-line clear_path between (sr,sc) and (tr,tc).

    Returns True if every tile strictly between the two endpoints is
    non-opaque (i.e. light propagation is not blocked).  The endpoints
    themselves are NOT tested — vendor light.c:218-260 walks the circle and
    only excludes a tile when clear_path() from the source to it fails on
    intervening squares, equivalent to couldsee() (vision.c::couldsee /
    clear_path).

    Cite: vendor/nethack/src/light.c::do_light_sources lines 205-260.
    JIT-pure: fixed-length scan, gated mask updates.
    """
    h, w = terrain_2d.shape
    dr = (tr - sr).astype(jnp.int32)
    dc = (tc - sc).astype(jnp.int32)
    abs_dr = jnp.abs(dr)
    abs_dc = jnp.abs(dc)
    dominant_row = abs_dr >= abs_dc
    steps = jnp.maximum(abs_dr, abs_dc)  # tiles to walk including endpoint

    step_dom_r = jnp.where(dominant_row, jnp.sign(dr), jnp.int32(0))
    step_dom_c = jnp.where(dominant_row, jnp.int32(0), jnp.sign(dc))
    step_min_r = jnp.where(dominant_row, jnp.int32(0), jnp.sign(dr))
    step_min_c = jnp.where(dominant_row, jnp.sign(dc), jnp.int32(0))
    dom_len    = jnp.where(dominant_row, abs_dr, abs_dc)

    # Static loop bound; rays beyond the actual distance contribute no blocks.
    # 32 covers DEFAULT_SIGHT_RADIUS (7) plus the largest source radius cap.
    MAX_STEPS = 32

    from Nethax.nethax.fov import _OPAQUE_TABLE, _OPAQUE_TABLE_SIZE

    def body(carry, i):
        cur_r, cur_c, err, blocked = carry
        # Walk one Bresenham step from current cell.
        new_err = err + jnp.where(dominant_row, abs_dc, abs_dr)
        do_minor = new_err * 2 >= dom_len
        new_err = jnp.where(do_minor, new_err - dom_len, new_err)
        nxt_r = cur_r + step_dom_r + jnp.where(do_minor, step_min_r, jnp.int32(0))
        nxt_c = cur_c + step_dom_c + jnp.where(do_minor, step_min_c, jnp.int32(0))

        # Only count this cell if i < steps - 1 (strictly between endpoints)
        # AND i < steps (don't walk past target).
        strictly_between = (jnp.int32(i) < (steps - jnp.int32(1)))
        in_bounds = (nxt_r >= 0) & (nxt_r < h) & (nxt_c >= 0) & (nxt_c < w)
        tile_idx = jnp.where(in_bounds, terrain_2d[nxt_r, nxt_c], jnp.int8(0))
        tile_idx_clipped = jnp.clip(tile_idx.astype(jnp.int32), 0, _OPAQUE_TABLE_SIZE - 1)
        is_opaque = _OPAQUE_TABLE[tile_idx_clipped]
        new_blocked = blocked | (strictly_between & in_bounds & is_opaque)
        return (nxt_r, nxt_c, new_err, new_blocked), None

    (_, _, _, blocked), _ = jax.lax.scan(
        body,
        (sr.astype(jnp.int32), sc.astype(jnp.int32), jnp.int32(0), jnp.bool_(False)),
        jnp.arange(MAX_STEPS),
    )
    # If source == target (steps == 0), trivially in LOS.
    return (~blocked) | (steps == jnp.int32(0))


def compute_lit_at(state, r: jax.Array, c: jax.Array) -> jax.Array:
    """Return True if tile (r, c) is covered by any active light source.

    A source is active when:
      - source_type != LTYPE_NONE, AND
      - timestep < source_until_turn  OR  source_until_turn == -1.

    Distance check uses Chebyshev (L∞) metric, matching the circle-table
    bounding box in do_light_sources.  In addition, a source only lights a
    tile when the path between source and tile is unobstructed by opaque
    terrain — vendor light.c walks the circle and excludes tiles that fail
    clear_path()/couldsee() (lines 205-260).

    Cite: vendor/nethack/src/light.c::do_light_sources (lines 169-260).

    JIT-pure: vectorised over the sources axis with a Bresenham LOS check.
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

    # LOS gating — vendor light.c:205-260 requires clear_path from each source
    # to the lit cell.  Compute per-source LOS to (r, c) on the current level.
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    terrain_2d = state.terrain[b, lv]
    tr = r.astype(jnp.int32)
    tc = c.astype(jnp.int32)

    def _los_one(src_r, src_c):
        return _has_los(terrain_2d, src_r.astype(jnp.int32),
                        src_c.astype(jnp.int32), tr, tc)

    los_mask = jax.vmap(_los_one)(
        lighting.source_pos[:, 0],
        lighting.source_pos[:, 1],
    )  # [MAX]

    return jnp.any(active & within & los_mask)


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
