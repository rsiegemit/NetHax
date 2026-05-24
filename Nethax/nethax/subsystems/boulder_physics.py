"""Boulder rolling physics — vendor ``launch_obj(BOULDER, ...)`` port.

Canonical sources:
  vendor/nethack/src/trap.c::launch_obj       lines 3260-3470  (rolling loop)
  vendor/nethack/src/trap.c                   lines 2672-2700  (ROLLING_BOULDER_TRAP fire)
  vendor/nethack/src/hack.c                   line 612         (moverock -> launch_obj)
  vendor/nethack/include/objects.h            BOULDER dmgval ≈ d(2,6)

The vendor ``launch_obj`` loop walks the boulder one tile at a time along
``(dx, dy)`` until one of the following terminates the roll:
  * Out-of-bounds / wall (``isok`` / OBSTRUCTED) → drop in current tile.
  * A monster occupies the tile → ``ohitmon`` (we model as squish kill).
  * The player occupies the tile → ``thitu`` with ``dmgval`` damage.

We model a JIT-pure variant using a fixed-shape ``jax.lax.fori_loop`` of
``STEPS_MAX`` iterations.  After any termination condition the ``stopped``
flag latches True and further iterations are no-ops (all mutations gated
on ``~stopped``).  If the loop exits without stopping the boulder is
dropped at the final tile as a ROCK_CLASS ground item.

Only ``roll_boulder`` is exported.  No off-limits files (action_dispatch,
monster_ai, magic, artifact_powers, traps, env) are modified — this
module is a pure helper available for a follow-up wiring wave.
"""
from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp

from Nethax.nethax.subsystems.boulders import (
    BOULDER_CATEGORY,
    BOULDER_TYPE_ID,
    _TILE_WALL,
    _TILE_VOID,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of tiles a boulder may traverse in a single launch.
# Vendor ``launch_obj`` uses ``dist = distmin(x1, y1, x2, y2)`` (trap.c:3313)
# — the trap's launch.x2/y2 is at most ~12 tiles away on standard maps.
# A static cap of 12 covers all vendor levels while keeping the unrolled
# loop body small and JIT-friendly.
STEPS_MAX: int = 12


# ---------------------------------------------------------------------------
# Helpers (boulder dmgval, ground-item write)
# ---------------------------------------------------------------------------

def _boulder_dmgval(rng) -> jnp.ndarray:
    """Vendor ``dmgval(BOULDER, &youmonst)`` ≈ ``d(2,6)`` = 2..12.

    Cite: vendor/nethack/include/objects.h BOULDER entry — dmgval rolls
    2 d6 (``d(2,6)``).
    """
    k1, k2 = jax.random.split(rng, 2)
    r1 = jax.random.randint(k1, (), minval=1, maxval=7, dtype=jnp.int32)
    r2 = jax.random.randint(k2, (), minval=1, maxval=7, dtype=jnp.int32)
    return r1 + r2


def _place_boulder_ground(ground_items, b, lv, r, c):
    """Write a BOULDER into ground_items[b, lv, r, c, 0].

    Mirrors ``boulders._place_boulder`` but inlined here so we don't depend
    on a private symbol of that module.
    """
    gi = ground_items
    return gi.replace(
        category=gi.category.at[b, lv, r, c, 0].set(jnp.int8(BOULDER_CATEGORY)),
        type_id=gi.type_id.at[b, lv, r, c, 0].set(jnp.int16(BOULDER_TYPE_ID)),
        quantity=gi.quantity.at[b, lv, r, c, 0].set(jnp.int16(1)),
        weight=gi.weight.at[b, lv, r, c, 0].set(jnp.int32(1000)),
        identified=gi.identified.at[b, lv, r, c, 0].set(jnp.bool_(True)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def roll_boulder(state, rng, start_pos, dir):
    """Roll a boulder from ``start_pos`` along ``dir`` until it stops.

    Vendor cite: vendor/nethack/src/trap.c::launch_obj  (lines 3260-3470).

    Parameters
    ----------
    state     : EnvState
    rng       : jax.random.PRNGKey  (consumed for dmgval rolls)
    start_pos : int32[2]   starting tile (row, col).  This is the tile the
                            boulder is launched FROM — the first traversed
                            tile is ``start_pos + dir``.
    dir       : int32[2]   step delta (dy, dx), each in {-1, 0, +1}.

    Returns
    -------
    (new_state, final_pos)
        ``new_state``  — EnvState with monster squish / player damage /
                          ground-item drop applied.
        ``final_pos``  — int32[2] tile where the boulder came to rest
                          (or where it was destroyed/absorbed).
    """
    start_pos = jnp.asarray(start_pos, dtype=jnp.int32)
    dir       = jnp.asarray(dir,       dtype=jnp.int32)

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)

    terrain_2d = state.terrain[b, lv]
    map_h, map_w = terrain_2d.shape

    # Carry: (state, stopped, cur_pos, rng).
    init = (
        state,
        jnp.bool_(False),
        start_pos,
        rng,
    )

    def body(_i, carry):
        st, stopped, cur, key = carry

        # Tentative next tile = cur + dir.
        nxt = cur + dir

        in_bounds = (
            (nxt[0] >= jnp.int32(0)) & (nxt[0] < jnp.int32(map_h))
            & (nxt[1] >= jnp.int32(0)) & (nxt[1] < jnp.int32(map_w))
        )

        # Clamp for safe reads when out-of-bounds (the writes are gated).
        sr = jnp.clip(nxt[0], 0, map_h - 1)
        sc = jnp.clip(nxt[1], 0, map_w - 1)

        # Wall / void = solid → stop without entering (vendor: !isok or
        # OBSTRUCTED stops the roll, boulder is dropped at previous tile).
        tile = terrain_2d[sr, sc].astype(jnp.int32)
        is_solid = (tile == jnp.int32(_TILE_WALL)) | (tile == jnp.int32(_TILE_VOID))
        hit_wall = (~in_bounds) | is_solid

        # Monster at next tile? (vendor ohitmon → we model as squish kill.)
        mai = st.monster_ai
        mon_r = mai.pos[:, 0].astype(jnp.int32)
        mon_c = mai.pos[:, 1].astype(jnp.int32)
        mon_here = mai.alive & (mon_r == nxt[0]) & (mon_c == nxt[1])
        hit_monster = jnp.any(mon_here)

        # Player at next tile? (vendor u_at(x,y) → thitu w/ dmgval).
        player_here = (
            (st.player_pos[0].astype(jnp.int32) == nxt[0])
            & (st.player_pos[1].astype(jnp.int32) == nxt[1])
        )

        # Effective gates: only fire when not already stopped.
        active = ~stopped
        do_wall    = active & hit_wall
        do_monster = active & ~hit_wall & hit_monster
        do_player  = active & ~hit_wall & ~hit_monster & player_here
        do_step    = active & ~hit_wall & ~hit_monster & ~player_here

        # -- Monster squish: alive=False, hp=0 for every monster on tile. --
        mai_new = mai.replace(
            alive=jnp.where(mon_here & do_monster, jnp.bool_(False), mai.alive),
            hp=jnp.where(mon_here & do_monster, jnp.int32(0), mai.hp),
        )
        st1 = st.replace(monster_ai=mai_new)

        # -- Player hit: apply dmgval damage (clamped at 0). --
        key, key_dmg = jax.random.split(key, 2)
        dmg = _boulder_dmgval(key_dmg)
        new_hp = jnp.where(
            do_player,
            jnp.maximum(jnp.int32(0), st1.player_hp - dmg),
            st1.player_hp,
        )
        st2 = st1.replace(player_hp=new_hp)

        # -- Advance cur_pos for live steps; freeze otherwise. --
        new_cur = jnp.where(active & ~hit_wall, nxt, cur)

        # -- Latch stopped on any termination condition. --
        new_stopped = stopped | do_wall | do_monster | do_player

        return (st2, new_stopped, new_cur, key)

    final_state, final_stopped, final_pos, _ = lax.fori_loop(
        0, STEPS_MAX, body, init
    )

    # If the loop exited without stopping, drop the boulder at final_pos
    # (vendor: when ``dist`` exhausts naturally the boulder rests in place).
    fr = final_pos[0].astype(jnp.int32)
    fc = final_pos[1].astype(jnp.int32)

    # Only place if final_pos is in bounds (it always should be since cur
    # only advances when in_bounds was True, but guard regardless).
    in_bounds_final = (
        (fr >= jnp.int32(0)) & (fr < jnp.int32(map_h))
        & (fc >= jnp.int32(0)) & (fc < jnp.int32(map_w))
    )
    place_now = (~final_stopped) & in_bounds_final

    new_ground = _place_boulder_ground(
        final_state.ground_items, b, lv, fr, fc
    )
    out_state = lax.cond(
        place_now,
        lambda s: s.replace(ground_items=new_ground),
        lambda s: s,
        final_state,
    )

    return out_state, final_pos
