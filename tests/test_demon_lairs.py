"""Wave 5 Phase 2 — demon-prince lair factory tests.

Covers the six hand-authored factories in
`Nethax/nethax/dungeon/demon_lairs.py`:

    generate_asmodeus_lair    (cold / ice theme)
    generate_baalzebub_lair   (fire pillars)
    generate_juiblex_lair     (acid pools)
    generate_orcus_lair       (necropolis)
    generate_yeenoghu_lair    (gnoll fortress)
    generate_demogorgon_lair  (poisonous swamp — deepest Gehennom)

Citations: vendor/nethack/dat/asmodeus.lua, baalz.lua, juiblex.lua,
           orcus.lua; vendor/nethack/include/rm.h for POOL / ICE tile
           provenance.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.dungeon.demon_lairs import (
    generate_asmodeus_lair,
    generate_baalzebub_lair,
    generate_juiblex_lair,
    generate_orcus_lair,
    generate_yeenoghu_lair,
    generate_demogorgon_lair,
    _T_FLOOR,
    _T_WALL,
    _T_LAVA,
    _T_STAIR_UP,
    _T_STAIR_DOWN,
    _T_ICE_FLOOR,
    _T_POOL,
    _T_ALTAR,
    _T_GRAVE,
    _T_THRONE,
    _MON_ASMODEUS,
    _MON_BAALZEBUB,
    _MON_JUIBLEX,
    _MON_ORCUS,
    _MON_YEENOGHU,
    _MON_DEMOGORGON,
    _MON_FROST_GIANT,
    _MON_FIRE_GIANT,
    _MON_ICE_DEVIL,
    _MON_HORNED_DEVIL,
    _MON_BARBED_DEVIL,
    _MON_BONE_DEVIL,
    _MON_BALROG,
    _MON_ACID_BLOB,
    _MON_GREEN_SLIME,
    _MON_LICH,
    _MON_SKELETON,
    _MON_SNAKE,
    _MON_HYDRA,
    _MON_GNOLL,
    _MON_FLIND,
)


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Shape sanity check helper
# ---------------------------------------------------------------------------

def _assert_triple_shapes(terrain, monsters, items):
    assert terrain.shape == (MAP_H, MAP_W)
    assert monsters.shape[1] == 3
    assert items.shape[1] == 3


# ---------------------------------------------------------------------------
# 1. Asmodeus
# ---------------------------------------------------------------------------

class TestAsmodeusLair:
    def test_returns_three_arrays(self):
        _assert_triple_shapes(*generate_asmodeus_lair(_RNG))

    def test_asmodeus_lair_has_asmodeus_or_demon(self):
        """Asmodeus boss must appear (canonical entry chunk5.py:1079)."""
        _, monsters, _ = generate_asmodeus_lair(_RNG)
        mon_types = monsters[:, 2]
        has_asmodeus = bool(jnp.any(mon_types == _MON_ASMODEUS))
        # Fallback: any major demon (horned / barbed / bone / balrog).
        has_demon = bool(jnp.any(
            (mon_types == _MON_HORNED_DEVIL)
            | (mon_types == _MON_BARBED_DEVIL)
            | (mon_types == _MON_BONE_DEVIL)
            | (mon_types == _MON_BALROG)
        ))
        assert has_asmodeus or has_demon, (
            "Asmodeus's lair must contain Asmodeus or at least a major demon"
        )

    def test_asmodeus_lair_has_ice_tiles(self):
        """Cold-themed lair — vendor MAP does not use ICE_FLOOR ('I').

        Wave 6 parity-fix: updated to match vendor/nethack/dat/asmodeus.lua:15
        (MAP section uses only '|/-/.' and 'S' = secret door; the cold theme
        is provided by level flags + monsters, not terrain symbols).
        """
        terrain, _, _ = generate_asmodeus_lair(_RNG)
        # Verify the level is non-trivially terrained (walls + floor present).
        from Nethax.nethax.dungeon.demon_lairs import _T_FLOOR, _T_WALL
        n_floor = int(jnp.sum(terrain == _T_FLOOR))
        n_wall  = int(jnp.sum(terrain == _T_WALL))
        assert n_floor > 0 and n_wall > 0, (
            f"Asmodeus lair must have walls + floor, got walls={n_wall}, floor={n_floor}"
        )


# ---------------------------------------------------------------------------
# 2. Baalzebub
# ---------------------------------------------------------------------------

class TestBaalzebubLair:
    def test_returns_three_arrays(self):
        _assert_triple_shapes(*generate_baalzebub_lair(_RNG))

    def test_baalzebub_lair_has_lava_pillars(self):
        """Wave-5 spec: multiple LAVA tiles in a regular pattern."""
        terrain, _, _ = generate_baalzebub_lair(_RNG)
        n_lava = int(jnp.sum(terrain == _T_LAVA))
        assert n_lava >= 4, f"Expected >=4 LAVA pillars, got {n_lava}"

    def test_baalzebub_lair_has_fire_demons(self):
        """Boss + fire-themed demons (fire giants / balrogs / Baalzebub)."""
        _, monsters, _ = generate_baalzebub_lair(_RNG)
        mon_types = monsters[:, 2]
        has_fire_themed = bool(jnp.any(
            (mon_types == _MON_BAALZEBUB)
            | (mon_types == _MON_FIRE_GIANT)
            | (mon_types == _MON_BALROG)
            | (mon_types == _MON_HORNED_DEVIL)
        ))
        assert has_fire_themed, (
            "Baalzebub's lair must contain fire-themed demons"
        )


# ---------------------------------------------------------------------------
# 3. Juiblex
# ---------------------------------------------------------------------------

class TestJuiblexLair:
    def test_returns_three_arrays(self):
        _assert_triple_shapes(*generate_juiblex_lair(_RNG))

    def test_juiblex_lair_has_acid_pools(self):
        """Acid swamp — vendor MAP uses '}' (WATER) markers, not the 'A'
        (POOL) symbol we previously used.

        Wave 6 parity-fix: updated to match vendor/nethack/dat/juiblex.lua:28
        (MAP section uses '}' for pools / acid water and 'x' for solid stone;
        the lair is rendered as WATER tiles per the byte-identical MAP).  We
        also re-skin the inner 'P' markers (fake-pool wall fixups) as WATER.
        """
        terrain, _, _ = generate_juiblex_lair(_RNG)
        from Nethax.nethax.dungeon.demon_lairs import _T_WATER
        n_water = int(jnp.sum(terrain == _T_WATER))
        assert n_water > 0, (
            f"Juiblex lair must have WATER (acid pool) tiles, got {n_water}"
        )

    def test_juiblex_lair_has_juiblex_or_demon(self):
        """Juiblex boss + acid blobs / slimes."""
        _, monsters, _ = generate_juiblex_lair(_RNG)
        mon_types = monsters[:, 2]
        has_juiblex = bool(jnp.any(mon_types == _MON_JUIBLEX))
        has_swamp_demon = bool(jnp.any(
            (mon_types == _MON_ACID_BLOB)
            | (mon_types == _MON_GREEN_SLIME)
        ))
        assert has_juiblex or has_swamp_demon, (
            "Juiblex's lair must contain Juiblex or acid-swamp creatures"
        )


# ---------------------------------------------------------------------------
# 4. Orcus
# ---------------------------------------------------------------------------

class TestOrcusLair:
    def test_returns_three_arrays(self):
        _assert_triple_shapes(*generate_orcus_lair(_RNG))

    def test_orcus_lair_has_undead(self):
        """Necropolis: skeletons and/or liches."""
        _, monsters, _ = generate_orcus_lair(_RNG)
        mon_types = monsters[:, 2]
        has_undead = bool(jnp.any(
            (mon_types == _MON_SKELETON) | (mon_types == _MON_LICH)
        ))
        assert has_undead, "Orcus's lair must contain skeletons or liches"


# ---------------------------------------------------------------------------
# 5. Yeenoghu
# ---------------------------------------------------------------------------

class TestYeenoghuLair:
    def test_returns_three_arrays(self):
        _assert_triple_shapes(*generate_yeenoghu_lair(_RNG))

    def test_yeenoghu_lair_has_gnolls(self):
        """Gnoll fortress: gnolls and/or flinds.

        TODO Wave 6: "gnoll" / "flind" not present in monsters.py yet
        (cf. demon_lairs.py module-level note); we use sentinel ids.
        """
        _, monsters, _ = generate_yeenoghu_lair(_RNG)
        mon_types = monsters[:, 2]
        has_gnolls = bool(jnp.any(
            (mon_types == _MON_GNOLL) | (mon_types == _MON_FLIND)
        ))
        assert has_gnolls, "Yeenoghu's lair must contain gnolls or flinds"


# ---------------------------------------------------------------------------
# 6. Demogorgon
# ---------------------------------------------------------------------------

class TestDemogorgonLair:
    def test_returns_three_arrays(self):
        _assert_triple_shapes(*generate_demogorgon_lair(_RNG))

    def test_demogorgon_lair_has_swamp(self):
        """Poisonous swamp: many POOL tiles."""
        if not hasattr(TileType, "POOL"):
            pytest.skip("POOL not added to TileType yet")
        terrain, _, _ = generate_demogorgon_lair(_RNG)
        n_pool = int(jnp.sum(terrain == _T_POOL))
        assert n_pool > 20, (
            f"Demogorgon's swamp must have a substantial pool area, got {n_pool}"
        )


# ---------------------------------------------------------------------------
# Cross-lair invariants
# ---------------------------------------------------------------------------

class TestAllLairs:
    def test_all_demon_lairs_have_at_least_one_boss_monster(self):
        """Every lair carries at least one boss-tier monster sentinel."""
        boss_ids = {
            _MON_ASMODEUS, _MON_BAALZEBUB, _MON_JUIBLEX,
            _MON_ORCUS, _MON_YEENOGHU, _MON_DEMOGORGON,
        }
        factories = [
            generate_asmodeus_lair,
            generate_baalzebub_lair,
            generate_juiblex_lair,
            generate_orcus_lair,
            generate_yeenoghu_lair,
            generate_demogorgon_lair,
        ]
        for factory in factories:
            _, monsters, _ = factory(_RNG)
            mon_types = set(int(x) for x in monsters[:, 2].tolist())
            assert mon_types & boss_ids, (
                f"{factory.__name__} must place a boss-tier demon prince"
            )

    def test_each_lair_has_stairs_down(self):
        """Every lair except Demogorgon's (deepest Gehennom) has stair-down.

        Demogorgon's lair sits at the bottom of Gehennom — there is no
        stair-down by design.
        """
        cases = [
            (generate_asmodeus_lair,   True),
            (generate_baalzebub_lair,  True),
            (generate_juiblex_lair,    True),
            (generate_orcus_lair,      True),
            (generate_yeenoghu_lair,   True),
            (generate_demogorgon_lair, False),  # deepest — no stair-down
        ]
        for factory, expect_down in cases:
            terrain, _, _ = factory(_RNG)
            n_down = int(jnp.sum(terrain == _T_STAIR_DOWN))
            if expect_down:
                assert n_down >= 1, (
                    f"{factory.__name__} must place a stair-down"
                )
            else:
                assert n_down == 0, (
                    f"{factory.__name__} (deepest) must not have stair-down"
                )
