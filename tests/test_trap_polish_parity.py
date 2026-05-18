"""Parity tests for trap/lock/kick mechanic polish.

Covers:
  - HOLE fall damage (vendor/nethack/src/trap.c dotrap HOLE branch)
  - TRAPDOOR fall damage (same)
  - picklock_door Dex-dependent success rate (vendor/nethack/src/lock.c:636-644)
  - kick-monster damage (vendor/nethack/src/dokick.c:146-291)
  - kick blocked when WOUNDED_LEGS (vendor/nethack/src/dokick.c:1265-1310)
  - TRAPPED_CHEST effect: 1d10 HP + 25% poison (vendor/nethack/src/lock.c:104-114)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.traps import TrapType, place_trap, trigger_trap_envstate
from Nethax.nethax.subsystems.features import (
    DoorState, picklock_door, handle_kick,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic

_RNG = jax.random.PRNGKey(99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(hp: int = 50) -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp),
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )


def _flat_lv(state: EnvState) -> int:
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    return b * max_lv + lv


def _place_trap_at(state: EnvState, row: int, col: int, kind: TrapType) -> EnvState:
    flat = _flat_lv(state)
    pos = jnp.array([flat, row, col], dtype=jnp.int32)
    new_traps = place_trap(state.traps, pos, kind, _RNG)
    return state.replace(traps=new_traps)


def _set_door(state: EnvState, row: int, col: int, door_val: int) -> EnvState:
    flat = _flat_lv(state)
    new_ds = state.features.door_state.at[flat, row, col].set(jnp.int8(door_val))
    new_feat = state.features.replace(door_state=new_ds)
    return state.replace(features=new_feat)


def _set_wounded_legs(state: EnvState, turns: int) -> EnvState:
    new_ts = state.status.timed_statuses.at[int(TimedStatus.WOUNDED_LEGS)].set(
        jnp.int32(turns)
    )
    return state.replace(status=state.status.replace(timed_statuses=new_ts))


def _spawn_monster(state: EnvState, row: int, col: int, hp: int = 10) -> EnvState:
    """Place a live, hostile monster at (row, col)."""
    mai = state.monster_ai
    # Use slot 1 (slot 0 is sentinel).
    slot = 1
    new_mai = mai.replace(
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        pos=mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(False)),
    )
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHoleFallDamage:
    """HOLE trap applies rnd(6) fall damage — vendor/nethack/src/trap.c dotrap HOLE."""

    def test_hole_applies_fall_damage(self):
        state = _make_state(hp=50)
        state = _place_trap_at(state, 5, 5, TrapType.HOLE)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.player_hp) < 50, "HOLE should reduce HP"
        assert int(new_state.player_hp) >= 44, "HOLE fall damage is at most 6 (rnd(6))"

    def test_hole_damage_range(self):
        """Verify damage is 1–6 across multiple seeds."""
        damages = set()
        for i in range(30):
            rng = jax.random.PRNGKey(i)
            state = _make_state(hp=50)
            state = _place_trap_at(state, 5, 5, TrapType.HOLE)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            dmg = 50 - int(new_state.player_hp)
            assert 1 <= dmg <= 6, f"HOLE damage {dmg} out of [1,6] range"
            damages.add(dmg)
        # Should see some spread over 30 seeds.
        assert len(damages) > 1, "Expected varied HOLE damage values"


class TestTrapdoorFallDamage:
    """TRAPDOOR trap applies rnd(6) fall damage — vendor/nethack/src/trap.c."""

    def test_trapdoor_applies_fall_damage(self):
        state = _make_state(hp=50)
        state = _place_trap_at(state, 5, 5, TrapType.TRAPDOOR)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        assert int(new_state.player_hp) < 50, "TRAPDOOR should reduce HP"
        assert int(new_state.player_hp) >= 44, "TRAPDOOR fall damage is at most 6"

    def test_trapdoor_damage_range(self):
        for i in range(20):
            rng = jax.random.PRNGKey(i + 100)
            state = _make_state(hp=50)
            state = _place_trap_at(state, 5, 5, TrapType.TRAPDOOR)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            dmg = 50 - int(new_state.player_hp)
            assert 1 <= dmg <= 6, f"TRAPDOOR damage {dmg} out of [1,6]"


class TestPicklockDexDependent:
    """picklock_door success rate scales with Dex — vendor/nethack/src/lock.c:636-644."""

    def _success_rate(self, dex: int, trials: int = 100) -> float:
        state = _make_state()
        state = _set_door(state, 3, 3, DoorState.LOCKED)
        flat = _flat_lv(state)
        pos = jnp.array([flat, 3, 3], dtype=jnp.int32)
        successes = 0
        for i in range(trials):
            rng = jax.random.PRNGKey(i + 200)
            _, success = picklock_door(
                state.features, pos, rng=rng, player_dex=dex
            )
            if bool(success):
                successes += 1
        return successes / trials

    def test_picklock_dex_dependent(self):
        """Dex=18 should succeed more often than Dex=6 over 100 trials."""
        rate_high = self._success_rate(dex=18, trials=100)
        rate_low = self._success_rate(dex=6, trials=100)
        assert rate_high > rate_low, (
            f"Expected Dex=18 ({rate_high:.2f}) > Dex=6 ({rate_low:.2f})"
        )

    def test_picklock_high_dex_succeeds_often(self):
        """Dex=18 LOCK_PICK: ch=54 → ~54% chance. Should be clearly > 0."""
        rate = self._success_rate(dex=18, trials=100)
        assert rate > 0.2, f"Dex=18 success rate {rate:.2f} unexpectedly low"

    def test_picklock_low_dex_succeeds_rarely(self):
        """Dex=6 LOCK_PICK: ch=18 → ~18% chance. Should be < high-dex rate."""
        rate = self._success_rate(dex=6, trials=100)
        assert rate < 0.5, f"Dex=6 success rate {rate:.2f} unexpectedly high"


class TestKickMonsterDamages:
    """Kicking toward a monster damages it — vendor/nethack/src/dokick.c:146-291."""

    def test_kick_monster_damages(self):
        """Monster HP should decrease after being kicked."""
        state = _make_state(hp=50)
        # Place monster at player's tile for Wave 3 simplified kick.
        state = _spawn_monster(state, 5, 5, hp=30)
        state = state.replace(
            player_str=jnp.int16(18),
            player_dex=jnp.int8(14),
            player_con=jnp.int8(12),
        )
        new_state = handle_kick(state, _RNG)
        new_hp = int(new_state.monster_ai.hp[1])
        assert new_hp < 30, f"Monster HP should decrease; got {new_hp}"

    def test_kick_monster_minimum_damage_1(self):
        """kick damage = max(1, (Str+Dex+Con)/15) — always at least 1."""
        state = _make_state(hp=50)
        # Very weak player: Str=3, Dex=3, Con=3 → (9/15)=0 → clamped to 1.
        state = state.replace(
            player_str=jnp.int16(3),
            player_dex=jnp.int8(3),
            player_con=jnp.int8(3),
        )
        state = _spawn_monster(state, 5, 5, hp=10)
        new_state = handle_kick(state, _RNG)
        new_hp = int(new_state.monster_ai.hp[1])
        assert new_hp <= 9, "Even weakest kick deals at least 1 damage"

    def test_kick_monster_sets_not_peaceful(self):
        """Kicking a monster makes it hostile."""
        state = _make_state(hp=50)
        # Start peaceful.
        mai = state.monster_ai
        slot = 1
        new_mai = mai.replace(
            alive=mai.alive.at[slot].set(jnp.bool_(True)),
            pos=mai.pos.at[slot].set(jnp.array([5, 5], dtype=jnp.int16)),
            hp=mai.hp.at[slot].set(jnp.int32(20)),
            hp_max=mai.hp_max.at[slot].set(jnp.int32(20)),
            peaceful=mai.peaceful.at[slot].set(jnp.bool_(True)),
        )
        state = state.replace(monster_ai=new_mai)
        new_state = handle_kick(state, _RNG)
        assert not bool(new_state.monster_ai.peaceful[slot]), "Kicked monster should be hostile"


class TestKickBlockedWhenWoundedLegs:
    """Kick is a no-op when WOUNDED_LEGS > 0 — vendor/nethack/src/dokick.c:1265-1310."""

    def test_kick_blocked_when_wounded_legs(self):
        """With WOUNDED_LEGS=10, handle_kick returns state unchanged."""
        state = _make_state(hp=50)
        state = _set_wounded_legs(state, 10)
        # Put a door at the player tile so we'd normally get a change.
        state = _set_door(state, 5, 5, DoorState.LOCKED)
        new_state = handle_kick(state, _RNG)
        # Door state must be unchanged.
        flat = _flat_lv(state)
        orig_door = int(state.features.door_state[flat, 5, 5])
        new_door = int(new_state.features.door_state[flat, 5, 5])
        assert orig_door == new_door, "Kick should be blocked by WOUNDED_LEGS"

    def test_kick_allowed_when_legs_healed(self):
        """With WOUNDED_LEGS=0, kick proceeds normally."""
        state = _make_state(hp=50)
        state = _set_wounded_legs(state, 0)
        state = _set_door(state, 5, 5, DoorState.CLOSED)
        # Run enough seeds that at least one breaks the door.
        changed = False
        flat = _flat_lv(state)
        orig_door = int(state.features.door_state[flat, 5, 5])
        for i in range(20):
            rng = jax.random.PRNGKey(i + 300)
            new_state = handle_kick(state, rng)
            if int(new_state.features.door_state[flat, 5, 5]) != orig_door:
                changed = True
                break
        assert changed, "Kick should sometimes succeed when legs are healthy"


class TestTrappedChestEffect:
    """TRAPPED_CHEST fires 1d10 HP + 25% poison — vendor/nethack/src/lock.c:104-114."""

    def test_trapped_chest_effect(self):
        """Triggering TRAPPED_CHEST reduces HP by 1–10."""
        state = _make_state(hp=50)
        state = _place_trap_at(state, 5, 5, TrapType.TRAPPED_CHEST)
        new_state = trigger_trap_envstate(state, _RNG, 5, 5)
        dmg = 50 - int(new_state.player_hp)
        assert 1 <= dmg <= 10, f"TRAPPED_CHEST damage {dmg} out of [1,10]"

    def test_trapped_chest_damage_range(self):
        """Damage spans 1–10 across seeds."""
        damages = set()
        for i in range(40):
            rng = jax.random.PRNGKey(i + 400)
            state = _make_state(hp=50)
            state = _place_trap_at(state, 5, 5, TrapType.TRAPPED_CHEST)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            dmg = 50 - int(new_state.player_hp)
            assert 1 <= dmg <= 10
            damages.add(dmg)
        assert len(damages) > 2, "Expected spread of TRAPPED_CHEST damage values"

    def test_trapped_chest_poison_possible(self):
        """25% poison chance: at least one of 40 trials should set SICK."""
        got_poison = False
        for i in range(40):
            rng = jax.random.PRNGKey(i + 500)
            state = _make_state(hp=50)
            state = _place_trap_at(state, 5, 5, TrapType.TRAPPED_CHEST)
            new_state = trigger_trap_envstate(state, rng, 5, 5)
            sick = int(new_state.status.timed_statuses[int(TimedStatus.SICK)])
            if sick > 0:
                got_poison = True
                break
        assert got_poison, "TRAPPED_CHEST should sometimes poison (25% chance)"
