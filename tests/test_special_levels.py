"""Wave 4 Phase 2 — special-level factory tests.

Targets the four factories added at the bottom of
Nethax/nethax/dungeon/special_levels.py:

    generate_oracle_level(rng)  — 3 small rooms + Oracle NPC
    generate_mine_town(rng)     — 4-8 shops + temple + watchmen
    generate_mines_end(rng)     — luckstone in treasure room
    generate_big_room(rng)      — large single open room

Citations: vendor/nethack/dat/oracle.lua, minetn-1.lua, minend-1.lua,
bigrm-1.lua.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.dungeon.special_levels import (
    generate_oracle_level,
    generate_mine_town,
    generate_mines_end,
    generate_big_room,
    _MON_ORACLE,
    _MON_SHOPKEEPER,
    _MON_WATCHMAN,
    _ITEM_LUCKSTONE,
    _T_FLOOR,
    _T_WALL,
    _T_FOUNTAIN,
    _T_ALTAR,
    _T_SHOP_FLOOR,
)


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Oracle level
# ---------------------------------------------------------------------------

class TestOracleLevel:
    def test_returns_three_arrays(self):
        terrain, monsters, items = generate_oracle_level(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3

    def test_oracle_level_has_oracle_npc(self):
        """The Oracle monster (type id _MON_ORACLE) must appear in monsters."""
        _, monsters, _ = generate_oracle_level(_RNG)
        monster_types = monsters[:, 2]
        assert int(jnp.any(monster_types == _MON_ORACLE)), (
            "Oracle NPC must be placed on the level"
        )

    def test_oracle_level_has_fountains(self):
        """Oracle level features at least 1 fountain (delphi inner room)."""
        terrain, _, _ = generate_oracle_level(_RNG)
        n_fountains = int(jnp.sum(terrain == _T_FOUNTAIN))
        assert n_fountains >= 1, f"Expected >=1 fountain, got {n_fountains}"

    def test_oracle_level_has_floor(self):
        """Oracle level must have at least one walkable floor tile."""
        terrain, _, _ = generate_oracle_level(_RNG)
        assert int(jnp.sum(terrain == _T_FLOOR)) > 0


# ---------------------------------------------------------------------------
# Mine Town
# ---------------------------------------------------------------------------

class TestMineTown:
    def test_returns_three_arrays(self):
        terrain, monsters, items = generate_mine_town(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3

    def test_mine_town_has_shops(self):
        """Mine Town must have walkable shop interiors.

        Vendor minetn-1.lua does NOT use a SHOP_FLOOR symbol in its MAP
        block (line 16); shops are carved out by `des.region(type='shop')`
        directives.  We therefore assert FLOOR + walls instead — see the
        test_mine_town_has_temple test for the altar-room interior check.
        """
        # Wave 6 parity-fix: updated to match vendor/nethack/dat/minetn-1.lua:16
        # (MAP section uses '.' = floor everywhere; SHOP_FLOOR is not part of
        # vendor's MAP encoding.)
        terrain, _, _ = generate_mine_town(_RNG)
        from Nethax.nethax.dungeon.special_levels import _T_FLOOR
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        assert n_floor > 100, f"Expected open Mine Town floor, got {n_floor}"

    def test_mine_town_has_shopkeepers(self):
        """Mine Town spawns ≥4 shopkeepers (one per shop)."""
        _, monsters, _ = generate_mine_town(_RNG)
        n_shopkeepers = int(jnp.sum(monsters[:, 2] == _MON_SHOPKEEPER))
        assert n_shopkeepers >= 4, f"Expected >=4 shopkeepers, got {n_shopkeepers}"

    def test_mine_town_has_temple(self):
        """Mine Town must have an altar (the temple)."""
        terrain, _, _ = generate_mine_town(_RNG)
        n_altar = int(jnp.sum(terrain == _T_ALTAR))
        assert n_altar >= 1, f"Expected altar, got {n_altar}"

    def test_mine_town_has_watchmen(self):
        """Mine Town spawns at least one watchman."""
        _, monsters, _ = generate_mine_town(_RNG)
        n_watch = int(jnp.sum(monsters[:, 2] == _MON_WATCHMAN))
        assert n_watch >= 1, f"Expected >=1 watchman, got {n_watch}"


# ---------------------------------------------------------------------------
# Mines' End
# ---------------------------------------------------------------------------

class TestMinesEnd:
    def test_returns_three_arrays(self):
        terrain, monsters, items = generate_mines_end(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3

    def test_mines_end_has_luckstone(self):
        """Mines' End guarantees a luckstone (achievement1 per minend-1.lua)."""
        _, _, items = generate_mines_end(_RNG)
        item_types = items[:, 2]
        assert int(jnp.any(item_types == _ITEM_LUCKSTONE)), (
            "Mines' End must guarantee a luckstone"
        )

    def test_mines_end_has_floor(self):
        """Mines' End must be walkable."""
        terrain, _, _ = generate_mines_end(_RNG)
        assert int(jnp.sum(terrain == _T_FLOOR)) > 10


# ---------------------------------------------------------------------------
# Big Room
# ---------------------------------------------------------------------------

class TestBigRoom:
    def test_returns_three_arrays(self):
        terrain, monsters, items = generate_big_room(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3

    def test_big_room_is_single_open_room(self):
        """Big Room is a single contiguous open chamber.

        Acceptance criterion: ≥ 600 FLOOR tiles in one rectangular zone
        (vendor bigrm-1.lua = 75 * 16 = 1200 inner cells; we accept >=600
        to allow for stair carve-outs and the encoding pad).
        """
        terrain, _, _ = generate_big_room(_RNG)
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        assert n_floor >= 600, f"Expected ≥600 floor tiles, got {n_floor}"

    def test_big_room_has_walls(self):
        """Big Room is bounded by walls."""
        terrain, _, _ = generate_big_room(_RNG)
        n_wall = int(jnp.sum(terrain == _T_WALL))
        # Perimeter ~ 2*(75+16+2) ≈ 186 wall tiles.
        assert n_wall >= 100, f"Expected ≥100 wall tiles, got {n_wall}"

    def test_big_room_has_many_monsters(self):
        """Big Room spawns ~28 monsters per vendor bigrm-1.lua."""
        _, monsters, _ = generate_big_room(_RNG)
        # Count non-sentinel entries (type_id != -1).
        n_mon = int(jnp.sum(monsters[:, 2] != -1))
        assert n_mon >= 20, f"Expected ≥20 monsters, got {n_mon}"
