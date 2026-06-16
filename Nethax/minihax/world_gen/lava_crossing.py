"""World generator for Minihax-LavaCrossing-v0 and the MiniGrid-ported
LavaCrossingS{W}N{n_strips}-v0 family.

Vendor reference: MiniGrid's ``LavaCrossingEnv`` carves ``n_strips`` vertical
lava barriers at (approximately) equally-spaced x positions across a WxH
interior.  Each strip has exactly one random opening so the agent can pass
through.  The default ``n_strips=1`` matches the shipped
``vendor/minihack/minihack/dat/lava_crossing.des`` (13x7, single column at
col 6).

.des coordinate convention: (col, row). Our code: (row, col).
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, ItemType, MonsterType, MONSTER_MAX_HP, RoleType, RaceType,
)
from Nethax.minihax.states import (
    HazardState, HazardStaticParams, Inventory, SimpleMonsters, GroundItems,
)
from Nethax.minihax.primitives.visibility import compute_visible, compute_lit_map
from Nethax.minihax.primitives.leveling import compute_initial_stats


# Default active map (single-strip variant): 13 wide x 7 tall.  Left bank
# cols 1-5, right bank cols 7-11, lava column at col 6.
_DEFAULT_W = 13
_DEFAULT_H = 7


def _strip_cols_list(active_w: int, n_strips: int) -> list:
    """Return the ``n_strips`` x-coordinates (cols) at which to carve lava
    as a plain Python list of ints (static under JIT).

    Mirrors MiniGrid LavaCrossingEnv: strips are evenly spaced across the
    interior so positions are ``round(i * active_w / (n_strips + 1))`` for
    ``i in 1..n_strips``.  Clamped to ``[1, active_w - 2]`` so a strip never
    overlaps the bounding wall.
    """
    cols = []
    for i in range(1, n_strips + 1):
        c = int(round(i * active_w / (n_strips + 1)))
        c = max(1, min(active_w - 2, c))
        cols.append(c)
    return cols


def _make_lava_crossing_map(active_w: int, active_h: int,
                            strip_cols: list) -> jnp.ndarray:
    """Build the static lava crossing map (walls + floor + lava strips).

    ``strip_cols`` is a plain Python list (static under JIT).  Openings in
    the lava strips are NOT carved here — they are stamped in
    ``generate_lava_crossing`` once the per-reset openings are sampled.
    """
    game_map = jnp.full((active_h, active_w), TileType.VOID, dtype=jnp.int32)

    # Walls: top and bottom rows
    game_map = game_map.at[0, :].set(TileType.HWALL)
    game_map = game_map.at[active_h - 1, :].set(TileType.HWALL)

    # Walls: left and right columns (interior rows)
    game_map = game_map.at[1:active_h - 1, 0].set(TileType.VWALL)
    game_map = game_map.at[1:active_h - 1, active_w - 1].set(TileType.VWALL)

    # Floor interior
    rows = jnp.arange(active_h)[:, None]
    cols = jnp.arange(active_w)[None, :]
    interior = (
        (rows >= 1) & (rows <= active_h - 2)
        & (cols >= 1) & (cols <= active_w - 2)
    )
    game_map = jnp.where(interior, TileType.FLOOR, game_map)

    # Lava strips: each strip col is a full-height interior column.
    # Python loop over a static list keeps the trace flat.
    for c in strip_cols:
        game_map = game_map.at[1:active_h - 1, c].set(TileType.LAVA)

    return game_map


def _pad_to_static(game_map, static_params, active_h, active_w):
    """Pad map to static dimensions."""
    sh = static_params.map_height
    sw = static_params.map_width
    padded = jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32)
    padded = padded.at[:active_h, :active_w].set(game_map)
    return padded


def generate_lava_crossing(rng, params, static_params, *,
                           n_strips: int = 1,
                           active_w: int = _DEFAULT_W,
                           active_h: int = _DEFAULT_H):
    """Generate a LavaCrossing environment state.

    Args:
        rng: JAX PRNG key.
        params: EnvParams.
        static_params: HazardStaticParams.
        n_strips: Number of vertical lava strips to carve.  ``n_strips=1``
            (default) preserves the shipped lava_crossing.des layout.
        active_w / active_h: Active map dimensions.  Defaults to 13x7 for
            the single-strip variant; MiniGrid-ported variants pass the
            grid size encoded in the env_id (e.g. S9N1 → 9x9, S11N5 → 11x11).

    Returns:
        HazardState
    """
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items

    rng, rng_player, rng_stair, rng_item_branch, rng_item_sub, rng_item_sub2, rng_stats = \
        jax.random.split(rng, 7)

    player_stats = compute_initial_stats(rng_stats, RoleType.MONK, RaceType.HUMAN)

    strip_cols = _strip_cols_list(active_w, n_strips)
    game_map = _make_lava_crossing_map(active_w, active_h, strip_cols)

    # One random opening per strip (jax-random determined per reset).
    # Each opening is a row in [1, active_h - 2].  Width-1 buffer prevents
    # a zero-size sample when n_strips == 0.
    rng, rng_openings = jax.random.split(rng)
    opening_rows = jax.random.randint(
        rng_openings, (max(n_strips, 1),), 1, active_h - 1,
    )
    # n_strips and strip_cols are static — a Python loop keeps the trace flat
    # and avoids a needless lax.scan with a tiny trip count.
    for i, c in enumerate(strip_cols):
        game_map = game_map.at[opening_rows[i], c].set(TileType.FLOOR)

    if n_strips <= 1 and active_w == _DEFAULT_W and active_h == _DEFAULT_H:
        # Backward-compatible single-strip path matches the shipped
        # lava_crossing.des: player and stair randomly placed in the
        # left/right banks around the single col-6 lava strip.
        player_r = jax.random.randint(rng_player, (), 1, 6)
        rng, rng_pc = jax.random.split(rng)
        player_c = jax.random.randint(rng_pc, (), 1, 6)
        player_pos = jnp.array([player_r, player_c], dtype=jnp.int32)

        stair_r = jax.random.randint(rng_stair, (), 1, 6)
        rng, rng_sc = jax.random.split(rng)
        stair_c = jax.random.randint(rng_sc, (), 7, 12)
        stair_pos = jnp.array([stair_r, stair_c], dtype=jnp.int32)
    else:
        # MiniGrid-ported variants: deterministic start/goal corners
        # (vendor: lg.set_start_pos(0, 0); lg.add_stair_down(x=_w-1, y=_h-1)).
        player_pos = jnp.array([1, 1], dtype=jnp.int32)
        stair_pos = jnp.array([active_h - 2, active_w - 2], dtype=jnp.int32)

    # Place downstair on map
    game_map = game_map.at[stair_pos[0], stair_pos[1]].set(TileType.DOWNSTAIR)

    # Random item on the player's side (cols 1..first_strip-1).  Falls back
    # to col 1 when the player sits flush against the first strip.
    first_strip_col = strip_cols[0] if n_strips > 0 else active_w - 1
    # 50% levitation branch, 50% cold branch
    # Levitation: 33% potion, 33% ring, 34% boots
    # Cold: 50% wand, 50% frost horn
    branch = jax.random.uniform(rng_item_branch, ()) < 0.5
    sub_roll = jax.random.uniform(rng_item_sub, ())
    sub_roll2 = jax.random.uniform(rng_item_sub2, ())

    lev_item = jnp.where(sub_roll < 0.33, ItemType.POTION_LEVITATION,
               jnp.where(sub_roll < 0.66, ItemType.RING_LEVITATION,
                         ItemType.BOOTS_LEVITATION))
    cold_item = jnp.where(sub_roll2 < 0.5, ItemType.WAND_COLD, ItemType.FROST_HORN)
    item_type = jnp.where(branch, lev_item, cold_item)

    # Random item position left of the first strip.  randint upper is
    # exclusive; clamp to ≥2 so the range is non-empty.
    rng, rng_ir, rng_ic = jax.random.split(rng, 3)
    item_r = jax.random.randint(rng_ir, (), 1, active_h - 1)
    item_c_hi = max(2, first_strip_col)
    item_c = jax.random.randint(rng_ic, (), 1, item_c_hi)

    # Ground items
    gi_positions = jnp.zeros((max_gi, 2), dtype=jnp.int32)
    gi_positions = gi_positions.at[0].set(jnp.array([item_r, item_c]))
    gi_types = jnp.zeros(max_gi, dtype=jnp.int32)
    gi_types = gi_types.at[0].set(item_type)
    gi_mask = jnp.zeros(max_gi, dtype=jnp.bool_)
    gi_mask = gi_mask.at[0].set(True)

    # Pad map
    padded_map = _pad_to_static(game_map, static_params, active_h, active_w)

    # Empty inventory
    inv = Inventory(
        item_ids=jnp.zeros(max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
    )

    # No monsters in lava crossing
    monsters = SimpleMonsters(
        position=jnp.zeros((max_m, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_m, dtype=jnp.int32),
        health=jnp.zeros(max_m, dtype=jnp.int32),
        mask=jnp.zeros(max_m, dtype=jnp.bool_),
    )

    ground_items = GroundItems(
        position=gi_positions,
        type_id=gi_types,
        mask=gi_mask,
    )

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width, lit_map)
    return HazardState(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        player_stats=player_stats,
        player_levitating=False,
        levitation_turns=0,
        inventory=inv,
        monsters=monsters,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        timestep=0,
        prev_action=0,
        terminal=False,
        state_rng=rng,
    )
