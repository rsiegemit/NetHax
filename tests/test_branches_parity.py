"""Branch parity tests: Mine Town generation and Sokoban prize placement.

Citations:
    vendor/nethack/src/mklev.c::mineend_level   — Mine Town special dispatch
    vendor/nethack/src/sokoban.c::sokoban_prize  — prize after 4 pits filled
    vendor/nethack/dat/dungeon.lua               — name="minetn" base=3 range=2
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon.branches import (
    Branch,
    _is_minetown_level,
    _MINES_MINETOWN_DEPTH,
    generate_mines_level,
)
from Nethax.nethax.dungeon.special_levels import (
    _T_ALTAR,
    _T_THRONE,
    _MON_SHOPKEEPER,
    _MON_WATCHMAN,
    _MON_PRIEST,
)
from Nethax.nethax.subsystems.boulders import (
    BOULDER_CATEGORY,
    BOULDER_TYPE_ID,
    SOKOBAN_BRANCH_IDX,
    SOKOBAN_PITS_TO_FILL,
    try_push_boulder,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.state import EnvState

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sokoban_state():
    """EnvState placed in the Sokoban branch at level 1."""
    state = EnvState.default(rng=_RNG)
    new_dungeon = state.dungeon.replace(
        current_branch=jnp.int8(SOKOBAN_BRANCH_IDX),
        current_level=jnp.int8(1),
    )
    return state.replace(dungeon=new_dungeon)


def _set_terrain(state, row, col, tile):
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    return state.replace(terrain=state.terrain.at[b, lv, row, col].set(jnp.int8(int(tile))))


def _place_boulder(state, row, col):
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    gi = state.ground_items
    return state.replace(ground_items=gi.replace(
        category  =gi.category.at[b, lv, row, col, 0].set(jnp.int8(BOULDER_CATEGORY)),
        type_id   =gi.type_id.at[b, lv, row, col, 0].set(jnp.int16(BOULDER_TYPE_ID)),
        quantity  =gi.quantity.at[b, lv, row, col, 0].set(jnp.int16(1)),
        weight    =gi.weight.at[b, lv, row, col, 0].set(jnp.int32(1000)),
        identified=gi.identified.at[b, lv, row, col, 0].set(jnp.bool_(True)),
    ))


def _set_pit_trap(state, row, col):
    from Nethax.nethax.subsystems.traps import TrapType
    b   = int(state.dungeon.current_branch)
    lv  = int(state.dungeon.current_level) - 1
    max_lv = state.terrain.shape[1]
    flat   = b * max_lv + lv
    return state.replace(traps=state.traps.replace(
        trap_type=state.traps.trap_type.at[flat, row, col].set(jnp.int8(int(TrapType.PIT)))
    ))


def _fill_n_pits(state, n: int):
    """Push n boulders into pits in sequence to accumulate sokoban_boulders_pitted."""
    # Use distinct row pairs so each boulder/pit pair occupies fresh tiles.
    for i in range(n):
        player_row = 2 + i * 3
        boulder_row = player_row + 1
        pit_row     = player_row + 2
        col = 5

        state = _set_terrain(state, boulder_row, col, TileType.FLOOR)
        state = _set_terrain(state, pit_row,     col, TileType.TRAP)
        state = _place_boulder(state, boulder_row, col)
        state = _set_pit_trap(state, pit_row, col)
        state = state.replace(player_pos=jnp.array([player_row, col], dtype=jnp.int16))

        from_pos = jnp.array([player_row, col],   dtype=jnp.int32)
        to_pos   = jnp.array([boulder_row, col], dtype=jnp.int32)
        state, _pushed = try_push_boulder(state, from_pos, to_pos, 1, 0)

    return state


# ---------------------------------------------------------------------------
# 1. Mine Town has temple (ALTAR tile)
# ---------------------------------------------------------------------------

def test_mine_town_has_temple():
    """Mines level 4 (Mine Town) must contain an ALTAR tile (the temple).

    Citation: vendor/nethack/src/mklev.c::mineend_level — Mine Town special
    level includes a temple with a priest and altar.
    """
    terrain, _monsters, _items = generate_mines_level(_RNG, depth=_MINES_MINETOWN_DEPTH)
    n_altar = int(jnp.sum(terrain == jnp.int8(_T_ALTAR)))
    assert n_altar >= 1, (
        f"Mine Town (Mines depth {_MINES_MINETOWN_DEPTH}) must contain an ALTAR tile; "
        f"found {n_altar}"
    )


# ---------------------------------------------------------------------------
# 2. Mine Town has shop (shopkeeper monster slot + items)
# ---------------------------------------------------------------------------

def test_mine_town_has_shop():
    """Mines level 4 must include shopkeeper monster placements.

    Citation: vendor/nethack/src/mklev.c::mineend_level — Mine Town contains
    1-2 shops with shopkeepers placed at room doors.
    """
    from Nethax.nethax.dungeon.special_levels import generate_mine_town

    _terrain, monsters, _items = generate_mine_town(_RNG)
    n_shopkeepers = int(jnp.sum(monsters[:, 2] == jnp.int16(_MON_SHOPKEEPER)))
    assert n_shopkeepers >= 1, (
        f"Mine Town must have at least 1 shopkeeper; found {n_shopkeepers}"
    )

    # Verify the dispatch from generate_mines_level also hits Mine Town.
    terrain2, monster_ids, _items2 = generate_mines_level(_RNG, depth=_MINES_MINETOWN_DEPTH)
    # _MON_SHOPKEEPER (2) should be in the returned monster type ids list.
    assert _MON_SHOPKEEPER in monster_ids, (
        "generate_mines_level at Mine Town depth must include shopkeeper type id"
    )


# ---------------------------------------------------------------------------
# 3. Mine Town has throne room (THRONE tile)
# ---------------------------------------------------------------------------

def test_mine_town_has_throne():
    """Mines level 4 must contain a THRONE tile (throne room with watchmen).

    Citation: vendor/nethack/src/mklev.c::mineend_level — Mine Town contains a
    throne room populated with peaceful watchmen guards.
    """
    terrain, _monsters, _items = generate_mines_level(_RNG, depth=_MINES_MINETOWN_DEPTH)
    n_throne = int(jnp.sum(terrain == jnp.int8(_T_THRONE)))
    assert n_throne >= 1, (
        f"Mine Town (Mines depth {_MINES_MINETOWN_DEPTH}) must contain a THRONE tile; "
        f"found {n_throne}"
    )


# ---------------------------------------------------------------------------
# 4. Sokoban prize appears after 4 pits filled
# ---------------------------------------------------------------------------

def test_sokoban_prize_at_top():
    """After filling SOKOBAN_PITS_TO_FILL pits the prize spawns on the level.

    Prize is a bag of holding OR amulet of reflection (random per
    sokoban.c::sokoban_prize).  We verify: category != 0 at the prize tile
    (1, 1), which is the level-exit stand-in used by boulders.py.

    Citation: vendor/nethack/src/sokoban.c::sokoban_prize — prize spawns at
    level end (top of Sokoban branch) once all pits are filled.
    """
    state = _make_sokoban_state()
    state = _fill_n_pits(state, SOKOBAN_PITS_TO_FILL)

    assert int(state.sokoban_boulders_pitted) >= SOKOBAN_PITS_TO_FILL, (
        f"Expected sokoban_boulders_pitted >= {SOKOBAN_PITS_TO_FILL}, "
        f"got {int(state.sokoban_boulders_pitted)}"
    )

    # Prize must appear at (1, 1) on the current Sokoban level.
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    prize_cat = int(state.ground_items.category[b, lv, 1, 1, 0])
    assert prize_cat != 0, (
        "After filling all Sokoban pits, a prize item must appear at tile (1,1). "
        f"category at (1,1) = {prize_cat}"
    )
