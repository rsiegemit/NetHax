"""Wave 3 trap tests.

Tests cover trigger_trap for key trap types and the integration with
action_dispatch._try_step (player steps onto a TRAP tile).
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.actions import CompassCardinalDirection
from Nethax.nethax.subsystems.action_dispatch import dispatch_action
from Nethax.nethax.subsystems.traps import TrapType, trigger_trap, place_trap
from Nethax.nethax.subsystems.status_effects import TimedStatus

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 5, player_col: int = 5) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
    )


def _set_tile(state: EnvState, row: int, col: int, tile: TileType) -> EnvState:
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[b, lv, row, col].set(jnp.int8(tile))
    return state.replace(terrain=new_terrain)


def _flat_lv(state: EnvState) -> int:
    b   = int(state.dungeon.current_branch)
    lv  = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _place_trap(state: EnvState, row: int, col: int, kind: TrapType) -> EnvState:
    flat = _flat_lv(state)
    pos = jnp.array([flat, row, col], dtype=jnp.int32)
    new_traps = place_trap(state.traps, pos, kind, _RNG)
    return state.replace(traps=new_traps)


def _trigger_at(state: EnvState, row: int, col: int):
    """Call trigger_trap directly on the given position."""
    flat = _flat_lv(state)
    pos = jnp.array([flat, row, col], dtype=jnp.int32)
    return trigger_trap(state.traps, _RNG, pos)


# ---------------------------------------------------------------------------
# Unit tests for trigger_trap
# ---------------------------------------------------------------------------

class TestArrowTrap:
    def test_damage_positive(self):
        """ARROW_TRAP must deal d6 damage (≥ 1)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.ARROW_TRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert int(dmg) >= 1, f"Expected damage ≥ 1, got {dmg}"

    def test_damage_at_most_6(self):
        """ARROW_TRAP must deal ≤ d6 = 6 damage."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.ARROW_TRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert int(dmg) <= 6, f"Expected damage ≤ 6, got {dmg}"

    def test_reveals_trap(self):
        """Triggering must mark the trap as revealed."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.ARROW_TRAP)
        flat = _flat_lv(state)
        new_traps, _, _ = _trigger_at(state, 5, 5)
        assert bool(new_traps.revealed[flat, 5, 5]), "Trap must be revealed after trigger"


class TestDartTrap:
    def test_damage_range(self):
        """DART_TRAP must deal 1–3 damage."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.DART_TRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert 1 <= int(dmg) <= 3, f"Expected damage in [1,3], got {dmg}"


class TestRocktrap:
    def test_damage_range(self):
        """ROCKTRAP (d2 + d20) must deal 2–22 damage."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.ROCKTRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert 2 <= int(dmg) <= 22, f"Expected damage in [2,22], got {dmg}"


class TestSqkyBoard:
    def test_no_damage(self):
        """SQKY_BOARD must deal 0 damage."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.SQKY_BOARD)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert int(dmg) == 0, f"Expected 0 damage, got {dmg}"

    def test_wake_flag(self):
        """SQKY_BOARD must set side_effects[3] (wake monsters) to 1."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.SQKY_BOARD)
        _, _, se = _trigger_at(state, 5, 5)
        assert int(se[3]) == 1, f"Expected wake flag=1, got se={se}"


class TestBearTrap:
    def test_damage_range(self):
        """BEAR_TRAP must deal d(2,4) = 2..8 HP damage (vendor trap.c:1490)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.BEAR_TRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert 2 <= int(dmg) <= 8, f"Expected damage in [2,8], got {dmg}"

    def test_freeze_turns(self):
        """BEAR_TRAP must hold rn1(4,4) = 4..7 turns (vendor trap.c:1506)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.BEAR_TRAP)
        _, _, se = _trigger_at(state, 5, 5)
        assert 4 <= int(se[0]) <= 7, f"Expected freeze in [4,7], got se={se}"


class TestPit:
    def test_damage_range(self):
        """PIT must deal rnd(6) = 1..6 HP fall damage (vendor trap.c:1950)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.PIT)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert 1 <= int(dmg) <= 6, f"Expected damage in [1,6], got {dmg}"

    def test_freeze_turns(self):
        """PIT must hold rn1(6,2) = 2..7 turns (vendor trap.c:1920)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.PIT)
        _, _, se = _trigger_at(state, 5, 5)
        assert 2 <= int(se[0]) <= 7, f"Expected freeze in [2,7], got se={se}"


class TestSpikedPit:
    def test_damage_range(self):
        """SPIKED_PIT must deal rnd(10) = 1..10 damage (vendor trap.c:1925)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.SPIKED_PIT)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert 1 <= int(dmg) <= 10, f"Expected damage in [1,10], got {dmg}"

    def test_freeze_turns(self):
        """SPIKED_PIT must hold rn1(6,2) = 2..7 turns (vendor trap.c:1920)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.SPIKED_PIT)
        _, _, se = _trigger_at(state, 5, 5)
        assert 2 <= int(se[0]) <= 7, f"Expected freeze in [2,7], got se={se}"


class TestTelepTrap:
    def test_teleport_flag(self):
        """TELEP_TRAP must set side_effects[2] (teleport flag) to 1."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.TELEP_TRAP)
        _, dmg, se = _trigger_at(state, 5, 5)
        assert int(dmg) == 0
        assert int(se[2]) == 1, f"Expected teleport flag=1, got se={se}"


class TestSlpGasTrap:
    def test_no_damage(self):
        """SLP_GAS_TRAP must deal 0 damage."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.SLP_GAS_TRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert int(dmg) == 0

    def test_sleep_turns(self):
        """SLP_GAS_TRAP must set side_effects[1] (sleep) in [1, 20]."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.SLP_GAS_TRAP)
        _, _, se = _trigger_at(state, 5, 5)
        assert 1 <= int(se[1]) <= 20, f"Expected sleep in [1,20], got se={se}"


class TestWebTrap:
    def test_no_damage(self):
        """WEB must deal 0 damage."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.WEB)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert int(dmg) == 0

    def test_freeze_turns(self):
        """WEB must set side_effects[0] (freeze) in [1, 6]."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.WEB)
        _, _, se = _trigger_at(state, 5, 5)
        assert 1 <= int(se[0]) <= 6, f"Expected freeze in [1,6], got se={se}"


class TestFireTrap:
    def test_damage_range(self):
        """FIRE_TRAP must deal d(2,4) = 2..8 fire damage (vendor trap.c:4238)."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.FIRE_TRAP)
        _, dmg, _ = _trigger_at(state, 5, 5)
        assert 2 <= int(dmg) <= 8, f"Expected damage in [2,8], got {dmg}"


class TestVibratingSqr:
    def test_no_damage_no_se(self):
        """VIBRATING_SQUARE must deal 0 damage and no side effects."""
        state = _make_state()
        state = _place_trap(state, 5, 5, TrapType.VIBRATING_SQUARE)
        _, dmg, se = _trigger_at(state, 5, 5)
        assert int(dmg) == 0
        assert int(se[0]) == 0 and int(se[1]) == 0


# ---------------------------------------------------------------------------
# Integration tests: step onto TRAP tile via dispatch_action
# ---------------------------------------------------------------------------

class TestStepOntoTrap:
    def test_arrow_trap_hp_decreases(self):
        """Stepping onto ARROW_TRAP tile: player HP must decrease."""
        state = _make_state(player_row=5, player_col=5)
        # Place floor at (4,5) so player can enter, then mark it as TRAP.
        state = _set_tile(state, 4, 5, TileType.TRAP)
        state = _place_trap(state, 4, 5, TrapType.ARROW_TRAP)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        assert int(result.player_hp) < 50, (
            f"Expected HP < 50, got {result.player_hp}"
        )
        assert jnp.array_equal(result.player_pos, jnp.array([4, 5], dtype=jnp.int16)), (
            f"Player should have moved onto trap tile, got {result.player_pos}"
        )

    def test_pit_frozen_status(self):
        """Stepping onto PIT trap: timed_statuses[FROZEN] must be set."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.TRAP)
        state = _place_trap(state, 4, 5, TrapType.PIT)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        frozen = int(result.status.timed_statuses[int(TimedStatus.FROZEN)])
        assert frozen >= 1, f"Expected FROZEN >= 1 turn, got {frozen}"

    def test_telep_trap_pos_changes(self):
        """Stepping onto TELEP_TRAP: teleport flag is set; player moved to tile."""
        # The dispatch integration sets the teleport side-effect but does not
        # move the player a second time (actual teleport applied by env loop).
        # We verify the player DID move to the trap tile.
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.TRAP)
        state = _place_trap(state, 4, 5, TrapType.TELEP_TRAP)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        assert jnp.array_equal(result.player_pos, jnp.array([4, 5], dtype=jnp.int16)), (
            f"Player should be on trap tile, got {result.player_pos}"
        )

    def test_trap_revealed_after_step(self):
        """Stepping onto a trap must mark it as revealed."""
        state = _make_state(player_row=5, player_col=5)
        state = _set_tile(state, 4, 5, TileType.TRAP)
        state = _place_trap(state, 4, 5, TrapType.ARROW_TRAP)

        result = dispatch_action(state, jnp.int32(CompassCardinalDirection.N), _RNG)

        flat = _flat_lv(result)
        assert bool(result.traps.revealed[flat, 4, 5]), "Trap must be revealed after stepping on it"
