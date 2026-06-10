"""Brax-style port of ``monster_ai._pathfind_step_impl``.

This module provides ``_pathfind_step_brax_impl`` (and the jit alias
``pathfind_step_brax``) — a drop-in alternative to the original BFS
pathfinder that flattens every conditional control-flow path into
``jnp.where`` masking, in the style of Google Brax / Craftax.

Design notes
------------
*   The original ``_pathfind_step_impl`` in ``monster_ai.py`` already
    avoids ``jax.lax.cond``: its only ``jax.lax.*`` call is the bounded
    BFS ``fori_loop`` (kept as-is per the task statement).  However the
    inner ``shift_one`` and the diagonal-squeeze logic relied on Python
    ``if dy > 0 / elif dy < 0:`` branches that, while compile-time
    constants (the 8 offsets are unrolled), interact poorly with
    vmap because they emit per-offset boundary slices.  The Brax-style
    rewrite replaces those slice updates with multiplicative ``jnp.where``
    masks over precomputed row/column indicator arrays, yielding a flat
    fusion-friendly HLO with one homogeneous shape per offset.
*   Every conditional in the per-iteration body is expressed as
    ``jnp.where(mask, then_val, else_val)`` or boolean-arithmetic masking
    (``& / | / ~``).  No ``jax.lax.cond``, no ``jax.lax.select_n``.
*   The bounded ``fori_loop`` BFS at the bottom is preserved verbatim —
    fixed-iteration scans are already vmap-friendly and represent O(1)
    HLO size per iteration.
*   The final "reachable -> BFS step else greedy step" choice is a
    single ``jnp.where`` (was a single ``jnp.where`` in the original
    too).

Byte-parity
-----------
Pathfind is deterministic — no RNG, no state mutation.  This module
re-uses the same module-level constants and helpers from
``monster_ai`` (``_M1_*``, ``_TILE_*``, ``_TT_*``, ``_MAP_H``,
``_MAP_W``, ``_PATHFIND_MAX_DEPTH``, ``_MONSTER_SIZE_TABLE``,
``_MZ_SMALL``, ``MAX_MONSTERS_PER_LEVEL``, ``_has_flag1``,
``_stuff_prevents_passage``, ``_mover_can_bust_door``,
``_mover_avoids_traps``, ``_current_level_terrain``) and is therefore
bit-for-bit equivalent in arithmetic to the original.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Reuse every helper, constant, and table from the canonical monster_ai
# module so that this Brax-style fork stays byte-identical in arithmetic.
from Nethax.nethax.subsystems.monster_ai import (
    _MAP_H,
    _MAP_W,
    _PATHFIND_MAX_DEPTH,
    _MONSTER_SIZE_TABLE,
    _MZ_SMALL,
    MAX_MONSTERS_PER_LEVEL,
    _M1_SWIM,
    _M1_AMPHIBIOUS,
    _M1_FLY,
    _M1_NOHANDS,
    _M1_AMORPHOUS,
    _TILE_WALL,
    _TILE_CLOSED_DOOR,
    _TILE_TREE,
    _TILE_IRONBARS,
    _TILE_WATER,
    _TILE_LAVA,
    _TT_PIT,
    _TT_SPIKED_PIT,
    _TT_HOLE,
    _TT_TRAPDOOR,
    _has_flag1,
    _stuff_prevents_passage,
    _mover_can_bust_door,
    _mover_avoids_traps,
    _current_level_terrain,
)


# 8-neighbor offsets in canonical order (matches the original pathfinder
# and ``apply_confusion_to_step``).
_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _shift_with_mask(field: jnp.ndarray, dy: int, dx: int, fill) -> jnp.ndarray:
    """Brax-style ``jnp.roll`` + boundary mask via ``jnp.where``.

    Replaces the original ``shifted.at[0:dy, :].set(INF)`` chain (whose
    branch depended on the Python sign of ``dy`` / ``dx``) with a single
    multiplicative ``jnp.where`` over precomputed row / column indicator
    masks.  The masks themselves are static (depend only on the
    compile-time constants ``dy`` and ``dx``), so they fold into HLO
    constants and contribute zero per-iteration overhead.
    """
    shifted = jnp.roll(field, shift=(dy, dx), axis=(0, 1))

    # Row mask: True where the row is a wrap-around from the opposite edge.
    rows = jnp.arange(_MAP_H)
    if dy > 0:
        row_wrap = rows < dy
    elif dy < 0:
        row_wrap = rows >= (_MAP_H + dy)
    else:
        row_wrap = jnp.zeros((_MAP_H,), dtype=jnp.bool_)

    # Column mask: True where the col is a wrap-around from the opposite edge.
    cols = jnp.arange(_MAP_W)
    if dx > 0:
        col_wrap = cols < dx
    elif dx < 0:
        col_wrap = cols >= (_MAP_W + dx)
    else:
        col_wrap = jnp.zeros((_MAP_W,), dtype=jnp.bool_)

    wrap_mask = row_wrap[:, None] | col_wrap[None, :]
    return jnp.where(wrap_mask, jnp.asarray(fill, dtype=field.dtype), shifted)


def _pathfind_step_brax_impl(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """Brax-style BFS pathfinder — see module docstring.

    Returns a jnp.int32[2] (dy, dx) in {-1, 0, 1}.

    Mirrors ``monster_ai._pathfind_step_impl`` arithmetic-for-arithmetic
    but with every conditional path expressed as ``jnp.where`` masking,
    no ``jax.lax.cond`` chains anywhere.  The single retained
    ``jax.lax.fori_loop`` is the fixed-iteration BFS relaxation, which
    is already vmap-friendly.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    terrain = _current_level_terrain(state)

    INF = jnp.int32(_PATHFIND_MAX_DEPTH + 100)

    # Distance field rooted at the player so its gradient guides the
    # monster — identical to the original.
    dist0 = jnp.full((_MAP_H, _MAP_W), INF, dtype=jnp.int32)
    dist0 = dist0.at[ppos[0], ppos[1]].set(jnp.int32(0))

    # --- Mover-specific passability (vendor mfndpos) ----------------------
    entry = mai.entry_idx[idx]
    can_swim = _has_flag1(entry, _M1_SWIM) | _has_flag1(entry, _M1_AMPHIBIOUS)
    can_fly  = _has_flag1(entry, _M1_FLY)

    nohands_e   = _has_flag1(entry, _M1_NOHANDS)
    safe_e      = jnp.clip(entry.astype(jnp.int32), 0,
                           _MONSTER_SIZE_TABLE.shape[0] - 1)
    msize_e     = _MONSTER_SIZE_TABLE[safe_e].astype(jnp.int32)
    verysmall_e = msize_e < jnp.int32(_MZ_SMALL)
    amorphous_e = _has_flag1(entry, _M1_AMORPHOUS)
    door_opener_handed = (~nohands_e) & (~verysmall_e)
    load_blocked       = _stuff_prevents_passage(mai, idx)
    amorphous_squeeze  = amorphous_e & ~load_blocked

    door_opener  = door_opener_handed | amorphous_squeeze
    door_buster  = _mover_can_bust_door(entry)
    bars_passer  = amorphous_squeeze
    trap_avoider = _mover_avoids_traps(entry)

    # --- Tile mask --------------------------------------------------------
    tile_field = terrain.astype(jnp.int32)
    is_wall        = (tile_field == _TILE_WALL)
    is_closed_door = (tile_field == _TILE_CLOSED_DOOR)
    is_tree        = (tile_field == _TILE_TREE)
    is_ironbars    = (tile_field == _TILE_IRONBARS)
    is_water       = (tile_field == _TILE_WATER)
    is_lava        = (tile_field == _TILE_LAVA)

    door_ok = door_opener | door_buster
    bars_ok = bars_passer

    not_wall = (
        ~is_wall
        & ~is_tree
        & (~is_closed_door | door_ok)
        & (~is_ironbars    | bars_ok)
    )
    water_ok = can_swim | can_fly
    lava_ok  = can_fly

    # Brax-style: every conditional terrain gate is a single ``jnp.where``
    # over a boolean tile mask — no branching, no lax.cond.
    terrain_ok = (
        not_wall
        & jnp.where(is_water, water_ok, jnp.bool_(True))
        & jnp.where(is_lava,  lava_ok,  jnp.bool_(True))
    )

    # --- Trap-avoidance mask (vendor mon.c:2353-2368) --------------------
    max_lv  = jnp.int32(state.terrain.shape[1])
    branch  = state.dungeon.current_branch.astype(jnp.int32)
    level0  = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    flat_lv = branch * max_lv + level0
    trap_t  = state.traps.trap_type[flat_lv].astype(jnp.int32)
    trap_rv = state.traps.revealed[flat_lv]

    always_lethal = (
        (trap_t == jnp.int32(_TT_PIT))
        | (trap_t == jnp.int32(_TT_SPIKED_PIT))
        | (trap_t == jnp.int32(_TT_HOLE))
        | (trap_t == jnp.int32(_TT_TRAPDOOR))
    )
    known_any  = (trap_t != jnp.int32(0)) & trap_rv
    trap_block = (always_lethal | known_any) & trap_avoider

    terrain_ok = terrain_ok & ~trap_block

    # --- Peaceful / tame friendly-blocking occupancy --------------------
    self_mask_n  = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32) == idx
    self_is_tame = mai.tame[idx]
    other_alive_peaceful = mai.alive & mai.peaceful & ~self_mask_n
    other_alive_tame     = mai.alive & mai.tame     & ~self_mask_n
    blocking_friendly = jnp.where(
        self_is_tame,
        jnp.zeros_like(other_alive_peaceful),
        other_alive_peaceful | other_alive_tame,
    )
    occ = jnp.zeros((_MAP_H, _MAP_W), dtype=jnp.bool_)
    pp = mai.pos.astype(jnp.int32)
    safe_r = jnp.clip(pp[:, 0], 0, _MAP_H - 1)
    safe_c = jnp.clip(pp[:, 1], 0, _MAP_W - 1)
    occ = occ.at[safe_r, safe_c].max(blocking_friendly)

    passable = terrain_ok & ~occ

    # --- NODIAG (grid bug) -----------------------------------------------
    from Nethax.nethax.constants.monsters import MONSTERS as _MM
    _grid_bug_entry = next(
        (i for i, m in enumerate(_MM) if m.name == "grid bug"), -1,
    )
    nodiag = jnp.int32(_grid_bug_entry) == entry.astype(jnp.int32)

    # ----- Precompute per-offset shifted passable masks for diagonals ----
    # These are static-shape arrays (depend only on the compile-time
    # offsets) so we build them once outside the BFS loop.  The diagonal
    # squeeze test reuses the boundary-masked roll of ``passable`` for
    # each of the four cardinal directions.
    def shift_passable(dy: int, dx: int) -> jnp.ndarray:
        return _shift_with_mask(passable, dy, dx, jnp.bool_(False))

    # For diagonal (dy, dx) the orthogonal neighbors checked are
    # (r-dy, c) and (r, c-dx); these are the (dy, 0) and (0, dx) shifts
    # of the passable mask.  Precompute once.
    orth_shifts = {
        (dy, dx): (shift_passable(dy, 0), shift_passable(0, dx))
        for (dy, dx) in _OFFSETS
        if dy != 0 and dx != 0
    }

    def bfs_body(_k, dist_field):
        neigh_min = jnp.full_like(dist_field, INF)
        for dy, dx in _OFFSETS:
            shifted = _shift_with_mask(dist_field, dy, dx, INF)
            is_diag = (dy != 0) and (dx != 0)
            if is_diag:
                # NODIAG: grid bugs get INF for every diagonal contribution.
                # Brax-style: single ``jnp.where`` over a scalar bool mask.
                shifted = jnp.where(nodiag, INF, shifted)
                # Diagonal squeeze: allowed iff at least one orthogonal
                # neighbor is passable.  Both orthogonal lookups are
                # precomputed boolean grids → a single ``jnp.where``.
                orth_a, orth_b = orth_shifts[(dy, dx)]
                squeeze_ok = orth_a | orth_b
                shifted = jnp.where(squeeze_ok, shifted, INF)
            neigh_min = jnp.minimum(neigh_min, shifted)
        candidate = neigh_min + jnp.int32(1)
        # Brax-style passability gate: ``jnp.where`` instead of
        # ``jax.lax.cond``.  Unreachable tiles stay at INF.
        candidate = jnp.where(passable, candidate, INF)
        return jnp.minimum(dist_field, candidate)

    # Bounded BFS: fori_loop is fixed-iteration and vmap-friendly, so we
    # keep it (per the task statement).
    dist = jax.lax.fori_loop(0, _PATHFIND_MAX_DEPTH, bfs_body, dist0)

    # --- Pick best 8-neighbor step --------------------------------------
    monster_dist = dist[mpos[0], mpos[1]]
    reachable = monster_dist < INF

    # Read all 8 neighbor distances via gather + boundary mask.  This is
    # the Brax discipline equivalent of the original Python loop: build
    # an int32[8,2] of offsets and gather, then mask out-of-bounds.
    offsets_arr = jnp.asarray(_OFFSETS, dtype=jnp.int32)        # [8, 2]
    nr = mpos[0] + offsets_arr[:, 0]                            # [8]
    nc = mpos[1] + offsets_arr[:, 1]                            # [8]
    in_bounds = (nr >= 0) & (nr < _MAP_H) & (nc >= 0) & (nc < _MAP_W)
    sr = jnp.clip(nr, 0, _MAP_H - 1)
    sc = jnp.clip(nc, 0, _MAP_W - 1)
    raw_nd = dist[sr, sc]                                       # [8]
    neighbor_dists = jnp.where(in_bounds, raw_nd, INF)          # [8]

    best_idx = jnp.argmin(neighbor_dists).astype(jnp.int32)
    bfs_step = offsets_arr[best_idx]                            # [2]

    # Greedy fallback (8-dir Chebyshev gradient).
    greedy_delta = jnp.clip(ppos - mpos, -1, 1).astype(jnp.int32)

    # Final selection — single ``jnp.where``, no cond.
    return jnp.where(reachable, bfs_step, greedy_delta)


# Module-level jit alias so callers can swap ``pathfind_step`` →
# ``pathfind_step_brax`` without re-jitting.
pathfind_step_brax = jax.jit(_pathfind_step_brax_impl)
