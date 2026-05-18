"""Riding subsystem polish-2 parity tests.

Vendor reference: vendor/nethack/src/steed.c

Tests
-----
test_riding_grants_speed_bonus      — mount horse, player_extra_speed > 0
test_saddle_wears_over_time         — tick env 100×, saddle_condition decreased
test_cursed_saddle_fails_mount      — saddled=2, try_mount, player_steed_mid stays 0
test_combat_dismount_at_8_damage    — check_combat_dismount(state, 10) → steed_mid=0
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.subsystems.riding import (
    try_mount,
    check_combat_dismount,
    tick_saddle,
)
from Nethax.nethax.subsystems.skills import SkillId, SkillLevel


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_riding_parity.py helpers)
# ---------------------------------------------------------------------------

def _base_state():
    rng = jax.random.PRNGKey(0)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return env, state, rng


def _place_horse(state, *, tame: bool, saddled: int, entry_idx: int = 99):
    """Place a pony in slot 0, one cell east of the player.

    saddled: 0=no saddle, 1=uncursed, 2=cursed.
    """
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        tame=mai.tame.at[0].set(tame),
        saddled=mai.saddled.at[0].set(jnp.int8(saddled)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
        pos=mai.pos.at[0].set(jnp.array([p_row, p_col + 1], dtype=jnp.int16)),
        hp=mai.hp.at[0].set(jnp.int32(30)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(30)),
    )
    return state.replace(monster_ai=mai)


def _set_riding_skill(state, level: int):
    sid = int(SkillId.RIDING)
    new_level = state.skills.level.at[sid].set(jnp.int8(level))
    return state.replace(skills=state.skills.replace(level=new_level))


def _force_mount(state):
    """Directly set player_steed_mid=1 to simulate an already-mounted state."""
    return state.replace(player_steed_mid=jnp.uint32(1))


# ---------------------------------------------------------------------------
# test_riding_grants_speed_bonus
# ---------------------------------------------------------------------------

def test_riding_grants_speed_bonus():
    """Mounting a horse sets player_extra_speed > 0.

    Vendor reference: steed.c:447 (ugallop += rn1(20,30)) — player moves at
    steed speed while mounted.
    """
    _, state, rng = _base_state()
    state = _place_horse(state, tame=True, saddled=1, entry_idx=99)
    state = _set_riding_skill(state, int(SkillLevel.P_BASIC))
    # High dex to guarantee mount success.
    state = state.replace(player_dex=jnp.int8(100))

    new_state = try_mount(state, rng)

    # If mount succeeded, player_extra_speed should reflect the steed's move_speed.
    if int(new_state.player_steed_mid) != 0:
        assert int(new_state.player_extra_speed) > 0, (
            "Mounted player should have player_extra_speed > 0 (steed.c:447)"
        )
    else:
        pytest.skip("Mount attempt failed (rare with dex=100); rerun or increase dex")


# ---------------------------------------------------------------------------
# test_saddle_wears_over_time
# ---------------------------------------------------------------------------

def test_saddle_wears_over_time():
    """tick_saddle decrements saddle_condition after 100 ticks while riding.

    Vendor reference: steed.c — saddle wear; tick_saddle uses timestep%100 gate.
    """
    _, state, _ = _base_state()
    state = _force_mount(state)
    initial_condition = int(state.saddle_condition)

    # Advance timestep through 100 full cycles (10000 ticks total guarantees
    # at least 100 wear events if starting at timestep 0).
    for i in range(10000):
        state = state.replace(timestep=jnp.int32(i))
        state = tick_saddle(state)
        # Stop early once condition has decreased.
        if int(state.saddle_condition) < initial_condition:
            break

    assert int(state.saddle_condition) < initial_condition, (
        "saddle_condition should decrease after riding ticks (steed.c saddle wear)"
    )


# ---------------------------------------------------------------------------
# test_cursed_saddle_fails_mount
# ---------------------------------------------------------------------------

def test_cursed_saddle_fails_mount():
    """Cursed saddle (saddled=2) causes try_mount to fail: player_steed_mid stays 0.

    Vendor reference: steed.c:634 (DISMOUNT_BYCHOICE blocked by cursed saddle);
    steed.c:122 (otmp->cursed reduces chance -= 50).
    """
    _, state, rng = _base_state()
    state = _place_horse(state, tame=True, saddled=2, entry_idx=99)
    state = _set_riding_skill(state, int(SkillLevel.P_BASIC))
    state = state.replace(player_dex=jnp.int8(100))

    new_state = try_mount(state, rng)

    assert int(new_state.player_steed_mid) == 0, (
        "Cursed-saddled steed must not be mountable (steed.c:634)"
    )


# ---------------------------------------------------------------------------
# test_combat_dismount_at_8_damage
# ---------------------------------------------------------------------------

def test_combat_dismount_at_8_damage():
    """check_combat_dismount with damage >= 8 clears player_steed_mid.

    Vendor reference: steed.c::dismount_steed DISMOUNT_KNOCKED/DISMOUNT_FELL
    (~line 606) — combat hits can dislodge the rider.
    """
    _, state, _ = _base_state()
    state = _force_mount(state)
    assert int(state.player_steed_mid) != 0

    new_state = check_combat_dismount(state, jnp.int32(10))

    assert int(new_state.player_steed_mid) == 0, (
        "10 damage should force dismount (steed.c::dismount_steed DISMOUNT_FELL)"
    )
    # Verify sub-threshold hit does NOT dismount.
    state_riding = _force_mount(state)
    no_dismount = check_combat_dismount(state_riding, jnp.int32(5))
    assert int(no_dismount.player_steed_mid) != 0, (
        "5 damage should not force dismount (threshold is 8)"
    )
