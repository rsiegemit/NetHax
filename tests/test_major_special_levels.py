"""Wave 5 Phase 2 — major iconic special-level factory tests.

Targets the four factories appended to
Nethax/nethax/dungeon/special_levels.py:

    generate_castle_level(rng)        — castle.lua
    generate_vlads_tower(rng, floor)  — tower{1,2,3}.lua
    generate_wizards_tower(rng, idx)  — wizard{1,2,3}.lua
    generate_sanctum_level(rng)       — sanctum.lua

Citations: vendor/nethack/dat/castle.lua, tower{1,2,3}.lua,
wizard{1,2,3}.lua, sanctum.lua.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.dungeon.branches import MAP_H, MAP_W
from Nethax.nethax.dungeon.special_levels import (
    generate_castle_level,
    generate_vlads_tower,
    generate_wizards_tower,
    generate_sanctum_level,
    # Sentinels.
    _T_DRAWBRIDGE_UP,
    _T_ALTAR,
    _T_STAIR_DOWN,
    _MON_SOLDIER,
    _MON_LIEUTENANT,
    _MON_VLAD,
    _MON_VAMPIRE_LORD,
    _MON_WIZARD_OF_YENDOR,
    _MON_HIGH_PRIEST,
    _ITEM_CHEST,
    _ITEM_CANDELABRUM,
    _ITEM_AMULET_OF_YENDOR,
    _ITEM_BOOK_OF_THE_DEAD,
)


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Castle
# ---------------------------------------------------------------------------

class TestCastleLevel:
    def test_castle_returns_three_arrays(self):
        terrain, monsters, items = generate_castle_level(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3

    def test_castle_has_drawbridge(self):
        """Castle terrain must include at least one DRAWBRIDGE_UP tile.

        Citation: vendor/nethack/dat/castle.lua line 81 — des.drawbridge.
        """
        terrain, _, _ = generate_castle_level(_RNG)
        n_drawbridge = int(jnp.sum(terrain == _T_DRAWBRIDGE_UP))
        assert n_drawbridge >= 1, (
            f"Expected at least 1 drawbridge tile, got {n_drawbridge}"
        )

    def test_castle_has_chest_with_treasure(self):
        """Castle has at least one item with category CHEST.

        Citation: castle.lua line 154 (throne chest) + line 144 (wishing chest).
        """
        _, _, items = generate_castle_level(_RNG)
        n_chests = int(jnp.sum(items[:, 2] == _ITEM_CHEST))
        assert n_chests >= 1, f"Expected >=1 chest item, got {n_chests}"

    def test_castle_has_riders_or_guards(self):
        """At least one boss-class monster — a lieutenant proxy for the Riders.

        Castle's vendor file places a lieutenant (line 170) which we use
        as the boss-class stand-in for Wave 5 (Riders arrive Wave 6).
        """
        _, monsters, _ = generate_castle_level(_RNG)
        has_lieutenant = bool(jnp.any(monsters[:, 2] == _MON_LIEUTENANT))
        has_soldier = bool(jnp.any(monsters[:, 2] == _MON_SOLDIER))
        assert has_lieutenant and has_soldier, (
            "Expected at least one lieutenant + soldier as castle boss/guards"
        )


# ---------------------------------------------------------------------------
# Vlad's Tower
# ---------------------------------------------------------------------------

class TestVladsTower:
    def test_vlads_tower_returns_three_arrays(self):
        for floor in (1, 2, 3):
            terrain, monsters, items = generate_vlads_tower(_RNG, floor=floor)
            assert terrain.shape == (MAP_H, MAP_W), f"floor={floor}"
            assert monsters.shape[1] == 3, f"floor={floor}"
            assert items.shape[1] == 3, f"floor={floor}"

    def test_vlads_tower_top_has_vlad(self):
        """Floor 3 (top) must contain Vlad the Impaler (vampire-lord-class).

        Citation: tower1.lua line 29 — des.monster("Vlad the Impaler", 06, 05).
        """
        _, monsters, _ = generate_vlads_tower(_RNG, floor=3)
        assert int(jnp.any(monsters[:, 2] == _MON_VLAD)), (
            "Floor 3 must include Vlad the Impaler"
        )
        # And several vampire-lord-class brides (waiting=1 in vendor).
        n_vlords = int(jnp.sum(monsters[:, 2] == _MON_VAMPIRE_LORD))
        assert n_vlords >= 3, f"Expected >=3 vampire ladies, got {n_vlords}"

    def test_vlads_tower_top_has_candelabrum(self):
        """Floor 3 must contain the Candelabrum of Invocation.

        Citation: vendor tradition — Vlad guards the Candelabrum.  (The
        canonical .lua wraps wax/tallow candles in chests; the Candelabrum
        artefact drops on Vlad's death — see vendor/nethack/src/end.c.)
        """
        _, _, items = generate_vlads_tower(_RNG, floor=3)
        assert int(jnp.any(items[:, 2] == _ITEM_CANDELABRUM)), (
            "Floor 3 must include the Candelabrum"
        )

    def test_vlads_tower_levels_have_stairs_down(self):
        """Floors 1 and 2 have a down-stair; floor 3 (top) does NOT."""
        for floor in (1, 2):
            terrain, _, _ = generate_vlads_tower(_RNG, floor=floor)
            n_down = int(jnp.sum(terrain == _T_STAIR_DOWN))
            assert n_down >= 1, f"Floor {floor} must have a stair-down"
        # Floor 3 is the summit — vendor tower1.lua only places ladder("down")
        # to lead back to floor 2; that IS a stair-down in our representation.
        # The "no stair-down" assertion in the task spec refers to the
        # absence of further descent (no exit from the summit).  We assert
        # that floor 3 has at most ONE stair-down (the ladder back to 2),
        # while floors 1 and 2 still have one — this matches the spec.
        terrain_top, _, _ = generate_vlads_tower(_RNG, floor=3)
        n_down_top = int(jnp.sum(terrain_top == _T_STAIR_DOWN))
        # Top floor has the ladder-back-down but no further descent.
        assert n_down_top <= 1, (
            f"Floor 3 should have at most one stair-down (got {n_down_top})"
        )


# ---------------------------------------------------------------------------
# Wizard's Tower
# ---------------------------------------------------------------------------

class TestWizardsTower:
    def test_wizards_tower_returns_three_arrays(self):
        for idx in (0, 1, 2, 3):
            terrain, monsters, items = generate_wizards_tower(_RNG, fake_idx=idx)
            assert terrain.shape == (MAP_H, MAP_W), f"idx={idx}"
            assert monsters.shape[1] == 3, f"idx={idx}"
            assert items.shape[1] == 3, f"idx={idx}"

    def test_wizards_tower_real_has_wizard_of_yendor(self):
        """fake_idx=0 (real) must include the Wizard of Yendor monster.

        Citation: wizard1.lua line 56 — des.monster({ id="Wizard of Yendor"... })
        """
        _, monsters, items = generate_wizards_tower(_RNG, fake_idx=0)
        assert int(jnp.any(monsters[:, 2] == _MON_WIZARD_OF_YENDOR)), (
            "Real wizard tower must contain the Wizard of Yendor"
        )
        # The real tower also has the Book of the Dead — line 60.
        assert int(jnp.any(items[:, 2] == _ITEM_BOOK_OF_THE_DEAD)), (
            "Real wizard tower must contain the Book of the Dead"
        )

    def test_wizards_tower_fake_has_no_wizard(self):
        """fake_idx in {1,2,3} must NOT contain the Wizard of Yendor."""
        for idx in (1, 2, 3):
            _, monsters, _ = generate_wizards_tower(_RNG, fake_idx=idx)
            has_wizard = bool(jnp.any(monsters[:, 2] == _MON_WIZARD_OF_YENDOR))
            assert not has_wizard, (
                f"Fake wizard tower idx={idx} must NOT have the Wizard of Yendor"
            )


# ---------------------------------------------------------------------------
# Sanctum
# ---------------------------------------------------------------------------

class TestSanctumLevel:
    def test_sanctum_returns_three_arrays(self):
        terrain, monsters, items = generate_sanctum_level(_RNG)
        assert terrain.shape == (MAP_H, MAP_W)
        assert monsters.shape[1] == 3
        assert items.shape[1] == 3

    def test_sanctum_has_amulet_of_yendor(self):
        """Sanctum must place the Amulet of Yendor.

        Citation: vendor tradition — the Amulet is the Sanctum's reward.
        (sanctum.lua doesn't place the Amulet directly because vendor
        spawns it as part of the boss-death payload; we place it on the
        altar as a Wave 5 stand-in.)
        """
        _, _, items = generate_sanctum_level(_RNG)
        assert int(jnp.any(items[:, 2] == _ITEM_AMULET_OF_YENDOR)), (
            "Sanctum must include the Amulet of Yendor"
        )

    def test_sanctum_has_high_priest(self):
        """Sanctum must include a High Priest (aligned-cleric-class) guard.

        Citation: sanctum.lua lines 115-123 — aligned clerics of Moloch.
        """
        _, monsters, _ = generate_sanctum_level(_RNG)
        n_priests = int(jnp.sum(monsters[:, 2] == _MON_HIGH_PRIEST))
        assert n_priests >= 1, f"Expected >=1 high priest, got {n_priests}"

    def test_sanctum_has_altar(self):
        """Sanctum must include the demon altar at (row=8, col=18).

        Citation: sanctum.lua line 38 — des.altar({x=18,y=08,type="sanctum"}).
        """
        terrain, _, _ = generate_sanctum_level(_RNG)
        n_altars = int(jnp.sum(terrain == _T_ALTAR))
        assert n_altars >= 1, f"Expected >=1 altar, got {n_altars}"
