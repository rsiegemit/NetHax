"""Riding subsystem parity tests.

Vendor reference: vendor/nethack/src/steed.c

Tests
-----
test_initial_state_not_riding     — player_steed_mid == 0 on reset.
test_mount_tame_horse_succeeds    — tame+saddled horse adjacent, high dex → mounts.
test_mount_requires_saddle        — tame horse WITHOUT saddle → stays unmounted.
test_dismount_clears_steed        — riding=True, try_dismount → player_steed_mid==0.
test_fall_off_damages             — fall_off_steed → hp decreases by 1-6.
test_ride_action_dispatched       — RIDE action via env.step changes state when
                                    conditions are met.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants.actions import Command
from Nethax.nethax.subsystems.riding import try_mount, try_dismount, fall_off_steed
from Nethax.nethax.subsystems.skills import SkillId, SkillLevel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_state():
    """Return a freshly-reset state and env."""
    rng = jax.random.PRNGKey(42)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return env, state, rng


def _place_horse(state, *, tame: bool, saddled: bool, entry_idx: int = 99):
    """Place a pony (entry_idx=99 by default) in slot 0, adjacent to player.

    The monster is placed one cell east of the player (Chebyshev dist 1).
    """
    p_row = int(state.player_pos[0])
    p_col = int(state.player_pos[1])
    m_row, m_col = p_row, p_col + 1

    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        tame=mai.tame.at[0].set(tame),
        saddled=mai.saddled.at[0].set(jnp.int8(1 if saddled else 0)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
        pos=mai.pos.at[0].set(jnp.array([m_row, m_col], dtype=jnp.int16)),
        hp=mai.hp.at[0].set(jnp.int32(30)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(30)),
    )
    return state.replace(monster_ai=mai)


def _set_riding_skill(state, level: int):
    """Set the RIDING skill to the given level."""
    sid = int(SkillId.RIDING)
    new_level = state.skills.level.at[sid].set(jnp.int8(level))
    return state.replace(skills=state.skills.replace(level=new_level))


def _set_dex(state, dex: int):
    return state.replace(player_dex=jnp.int8(dex))


# ---------------------------------------------------------------------------
# test_initial_state_not_riding
# ---------------------------------------------------------------------------

def test_initial_state_not_riding():
    """player_steed_mid is 0 on fresh reset (steed.c: u.usteed = NULL at game start)."""
    _, state, _ = _base_state()
    assert int(state.player_steed_mid) == 0


# ---------------------------------------------------------------------------
# test_mount_tame_horse_succeeds
# ---------------------------------------------------------------------------

def test_mount_tame_horse_succeeds():
    """With tame saddled horse adjacent + high dex, eventually mounts.

    Uses a deterministic loop over 20 different RNG keys to handle the
    probabilistic success roll.  At dex=18 + skill=2 the threshold is
    5+10+18=33 out of 100, so ~33% per attempt; 20 trials << p(all fail)=0.67^20≈0.003.
    """
    _, state, _ = _base_state()
    state = _place_horse(state, tame=True, saddled=True, entry_idx=99)
    state = _set_riding_skill(state, int(SkillLevel.P_SKILLED))
    state = _set_dex(state, 18)

    mounted = False
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        result = try_mount(state, rng)
        if int(result.player_steed_mid) != 0:
            mounted = True
            break

    assert mounted, "Expected mount to succeed at least once in 20 attempts"


# ---------------------------------------------------------------------------
# test_mount_requires_saddle
# ---------------------------------------------------------------------------

def test_mount_requires_saddle():
    """Tame horse without saddle must not be mountable (steed.c:281-284)."""
    _, state, _ = _base_state()
    state = _place_horse(state, tame=True, saddled=False, entry_idx=99)
    state = _set_riding_skill(state, int(SkillLevel.P_SKILLED))
    state = _set_dex(state, 25)  # max dex — still no saddle

    # Try many seeds; none should succeed.
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        result = try_mount(state, rng)
        assert int(result.player_steed_mid) == 0, (
            f"Mounted without saddle on seed {seed}"
        )


# ---------------------------------------------------------------------------
# test_dismount_clears_steed
# ---------------------------------------------------------------------------

def test_dismount_clears_steed():
    """try_dismount clears player_steed_mid (steed.c:380 u.usteed = NULL)."""
    _, state, _ = _base_state()
    # Manually put player in a riding state.
    state = state.replace(player_steed_mid=jnp.uint32(5))
    assert int(state.player_steed_mid) == 5

    rng = jax.random.PRNGKey(0)
    result = try_dismount(state, rng)
    assert int(result.player_steed_mid) == 0


# ---------------------------------------------------------------------------
# test_fall_off_damages
# ---------------------------------------------------------------------------

def test_fall_off_damages():
    """fall_off_steed reduces HP by 1-6 and clears steed (steed.c::dismount_steed FELL)."""
    _, state, _ = _base_state()
    state = state.replace(
        player_hp=jnp.int32(100),
        player_steed_mid=jnp.uint32(3),
    )
    hp_before = int(state.player_hp)

    # Try several seeds to verify damage is always in [1, 6].
    for seed in range(10):
        rng = jax.random.PRNGKey(seed)
        result = fall_off_steed(state, rng)
        hp_after = int(result.player_hp)
        damage = hp_before - hp_after
        assert 1 <= damage <= 6, f"Unexpected damage={damage} on seed {seed}"
        assert int(result.player_steed_mid) == 0, "Steed not cleared after fall"


# ---------------------------------------------------------------------------
# test_ride_action_dispatched
# ---------------------------------------------------------------------------

def test_ride_action_dispatched():
    """RIDE action via env.step calls _handle_ride.

    Verifies the RIDE action slot is wired: after enough env.step calls with
    conditions met, player_steed_mid should become non-zero at least once.
    We give the player max dex and skilled riding for a high success rate
    (~73% per attempt), and run 30 trials.

    Also verifies via dispatch_action directly as a fast secondary check.
    """
    from Nethax.nethax.subsystems.action_dispatch import dispatch_action

    env, state, _ = _base_state()
    # Bump HP well above max so fall damage can't mask the effect.
    state = state.replace(
        player_hp=jnp.int32(100),
        player_hp_max=jnp.int32(100),
    )
    state = _place_horse(state, tame=True, saddled=True, entry_idx=99)
    state = _set_riding_skill(state, int(SkillLevel.P_SKILLED))
    state = _set_dex(state, 18)

    ride_action = jnp.int32(int(Command.RIDE))

    # --- Fast check: dispatch_action directly (no monster AI / status pipeline) ---
    direct_effect = False
    for seed in range(30):
        rng = jax.random.PRNGKey(seed)
        result = dispatch_action(state, ride_action, rng)
        mid = int(result.player_steed_mid)
        hp_after = int(result.player_hp)
        if mid != 0 or hp_after < 100:
            direct_effect = True
            break

    assert direct_effect, (
        "dispatch_action(RIDE) produced no observable effect in 30 attempts — "
        "handler not wired"
    )

    # --- Full pipeline check via env.step ---
    effect_seen = False
    for seed in range(30):
        rng = jax.random.PRNGKey(seed + 200)
        result, _obs, _rew, _done, _info = env.step(state, ride_action, rng)
        mid = int(result.player_steed_mid)
        hp_after = int(result.player_hp)
        if mid != 0 or hp_after < 100:
            effect_seen = True
            break

    assert effect_seen, (
        "env.step(RIDE) produced no observable effect in 30 steps — "
        "handler may not survive full pipeline"
    )
