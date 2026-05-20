"""Parity tests for the detection subsystem (detect.py).

Canonical sources:
  vendor/nethack/src/detect.c::food_detect    (~line 479)
  vendor/nethack/src/detect.c::object_detect  (~line 603)
  vendor/nethack/src/detect.c::monster_detect (~line 798)
  vendor/nethack/src/detect.c::do_clairvoyance (~line 1446)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems import detect as _detect
from Nethax.nethax.constants.tiles import TileType, VendorTileType


RNG = jax.random.PRNGKey(0)


def _state():
    return EnvState.default(RNG, StaticParams())


# ---------------------------------------------------------------------------
# 1. detect_food sets timer
# ---------------------------------------------------------------------------

class TestDetectFood:
    def test_detect_food_sets_timer(self):
        """detect_food sets detect_food_until_turn = timestep + 50.

        Cite: vendor/nethack/src/detect.c::food_detect (~line 479).
        """
        state = _state()
        ts = int(state.timestep)
        result = _detect.detect_food(state, RNG)
        expected = ts + 50
        assert int(result.identification.detect_food_until_turn) == expected

    def test_detect_food_does_not_touch_other_timers(self):
        state = _state()
        result = _detect.detect_food(state, RNG)
        assert int(result.identification.detect_monsters_until_turn) == -1
        assert int(result.identification.detect_treasure_until_turn) == -1


# ---------------------------------------------------------------------------
# 2. detect_treasure sets timer
# ---------------------------------------------------------------------------

class TestDetectTreasure:
    def test_detect_treasure_sets_timer(self):
        """detect_treasure sets detect_treasure_until_turn = timestep + 50.

        Cite: vendor/nethack/src/detect.c::object_detect (~line 603).
        """
        state = _state()
        ts = int(state.timestep)
        result = _detect.detect_treasure(state, RNG)
        assert int(result.identification.detect_treasure_until_turn) == ts + 50

    def test_detect_treasure_does_not_touch_food_timer(self):
        state = _state()
        result = _detect.detect_treasure(state, RNG)
        assert int(result.identification.detect_food_until_turn) == -1


# ---------------------------------------------------------------------------
# 3. detect_monsters sets timer
# ---------------------------------------------------------------------------

class TestDetectMonsters:
    def test_detect_monsters_sets_timer(self):
        """detect_monsters sets detect_monsters_until_turn = timestep + 100.

        Cite: vendor/nethack/src/detect.c::monster_detect (~line 798).
        """
        state = _state()
        ts = int(state.timestep)
        result = _detect.detect_monsters(state, RNG)
        assert int(result.identification.detect_monsters_until_turn) == ts + 100

    def test_detect_monsters_timer_advances_with_timestep(self):
        state = _state()
        # Simulate 10 turns having passed.
        state = state.replace(timestep=jnp.int32(10))
        result = _detect.detect_monsters(state, RNG)
        assert int(result.identification.detect_monsters_until_turn) == 110


# ---------------------------------------------------------------------------
# 4. detect_unseen reveals SDOOR -> CLOSED_DOOR
# ---------------------------------------------------------------------------

class TestDetectUnseen:
    def test_detect_unseen_reveals_sdoor(self):
        """SDOOR tiles become CLOSED_DOOR after detect_unseen.

        Cite: vendor/nethack/src/detect.c (SPE_DETECT_UNSEEN branch).
        Place a VendorTileType.SDOOR at (5, 5); after detect_unseen it
        becomes TileType.CLOSED_DOOR (our local enum value).
        """
        state = _state()
        b = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        sdoor_val = jnp.int8(int(VendorTileType.SDOOR))
        new_terrain = state.terrain.at[b, lv, 5, 5].set(sdoor_val)
        state = state.replace(terrain=new_terrain)

        result = _detect.detect_unseen(state, RNG)

        tile = int(result.terrain[b, lv, 5, 5])
        assert tile == int(TileType.CLOSED_DOOR), (
            f"Expected CLOSED_DOOR ({int(TileType.CLOSED_DOOR)}), got {tile}"
        )

    def test_detect_unseen_reveals_scorr(self):
        """SCORR tiles become CORRIDOR after detect_unseen.

        Cite: vendor/nethack/src/detect.c (SPE_DETECT_UNSEEN branch).
        """
        state = _state()
        b = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        scorr_val = jnp.int8(int(VendorTileType.SCORR))
        new_terrain = state.terrain.at[b, lv, 3, 7].set(scorr_val)
        state = state.replace(terrain=new_terrain)

        result = _detect.detect_unseen(state, RNG)

        tile = int(result.terrain[b, lv, 3, 7])
        assert tile == int(TileType.CORRIDOR), (
            f"Expected CORRIDOR ({int(TileType.CORRIDOR)}), got {tile}"
        )

    def test_detect_unseen_leaves_other_tiles_unchanged(self):
        """Non-secret tiles are not altered by detect_unseen."""
        state = _state()
        b = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1
        floor_val = jnp.int8(int(TileType.FLOOR))
        new_terrain = state.terrain.at[b, lv, 2, 2].set(floor_val)
        state = state.replace(terrain=new_terrain)

        result = _detect.detect_unseen(state, RNG)
        assert int(result.terrain[b, lv, 2, 2]) == int(TileType.FLOOR)


# ---------------------------------------------------------------------------
# 5. clairvoyance reveals 5x5 around player
# ---------------------------------------------------------------------------

class TestClairvoyance:
    def test_clairvoyance_reveals_5x5_around_player(self):
        """clairvoyance marks the 5x5 Chebyshev-2 region as explored.

        Cite: vendor/nethack/src/detect.c::do_clairvoyance (~line 1446).
        do_clairvoyance -> do_vicinity_map with radius 2.
        """
        state = _state()
        # Place player at (10, 10) so the full 5x5 is in bounds.
        state = state.replace(player_pos=jnp.array([10, 10], dtype=jnp.int16))
        b = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1

        result = _detect.clairvoyance(state, RNG)

        # All cells within Chebyshev radius 2 should be explored.
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = 10 + dr, 10 + dc
                assert bool(result.explored[b, lv, r, c]), (
                    f"Cell ({r},{c}) should be explored after clairvoyance"
                )

    def test_clairvoyance_does_not_reveal_beyond_radius(self):
        """Cells outside the vendor asymmetric rectangle stay unexplored.

        Cite: vendor/nethack/src/detect.c::do_vicinity_map lines 1464-1467:
            lo_y = max(0, u.uy - 5); hi_y = min(ROWNO-1, u.uy + 6)
            lo_x = max(1, u.ux - 9); hi_x = min(COLNO-1, u.ux + 10)
        For player at (pr=10, pc=10) the inclusive box covers
        rows ∈ [5, 16] and cols ∈ [1, 20]. Cells strictly outside that
        rectangle must remain unexplored.
        """
        state = _state()
        state = state.replace(player_pos=jnp.array([10, 10], dtype=jnp.int16))
        b = int(state.dungeon.current_branch)
        lv = int(state.dungeon.current_level) - 1

        result = _detect.clairvoyance(state, RNG)

        # (10, 21) is one column east of hi_x=20 — outside rectangle.
        assert not bool(result.explored[b, lv, 10, 21])
        # (17, 10) is one row south of hi_y=16 — outside rectangle.
        assert not bool(result.explored[b, lv, 17, 10])
        # (4, 10) is one row north of lo_y=5 — outside rectangle.
        assert not bool(result.explored[b, lv, 4, 10])
