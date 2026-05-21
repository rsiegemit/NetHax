"""Combat polish 2 parity tests.

Vendor references:
  vendor/nethack/src/uhitm.c:1467-1468  — two-handed STR damage scaling (1.5×)
  vendor/nethack/src/uhitm.c:387-394    — target-state to-hit bonuses
  vendor/nethack/src/weapon.c:2199-2200 — quarterstaff splcaster -3
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

import Nethax.nethax.subsystems.artifact_powers  # noqa: F401
import Nethax.nethax.subsystems.weapon_dice  # noqa: F401

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.subsystems.combat import melee_attack, SKILL_BASIC
from Nethax.nethax.subsystems.monster_ai import monsters_step_all

_RNG = jax.random.PRNGKey(42)
_N_TRIALS = 600

# Weapon type IDs (vendor/nethack/include/objects.h)
_LONG_SWORD_TYPE_ID      = 37
_TWO_HANDED_SWORD_TYPE_ID = 38


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(role: Role = Role.VALKYRIE):
    state = EnvState.default(_RNG)
    state = state.replace(
        player_role=jnp.int8(int(role)),
        player_str=jnp.int16(18 + 100),  # STR18(100) → max dbon = 6
        player_dex=jnp.int8(14),
        player_xl=jnp.int32(5),
        player_luck=jnp.int8(0),
        player_uhitinc=jnp.int8(0),
        player_udaminc=jnp.int8(0),
    )
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(9999)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(9999)),
        ac=mai.ac.at[0].set(jnp.int8(10)),   # AC=10 → easy to hit
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
        mstrategy=mai.mstrategy.at[0].set(jnp.int8(0)),
        pos=mai.pos.at[0].set(jnp.array([3, 3], dtype=jnp.int16)),
    )
    return state.replace(monster_ai=mai)


def _state_with_weapon(type_id: int):
    """Return a state where slot 0 is wielded with the given weapon type_id."""
    state = _base_state()
    items = state.inventory.items
    items = items.replace(
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        enchantment=items.enchantment.at[0].set(jnp.int8(0)),
    )
    inv = state.inventory.replace(items=items, wielded=jnp.int8(0))
    return state.replace(inventory=inv)


def _melee_damage_samples(state, n=_N_TRIALS, seed=7):
    """Run melee_attack n times; collect damage values.

    JIT-compiled + vmapped: a single compile of `melee_attack` is reused
    across all n trials.  Eager calls retrace the giant combat graph each
    invocation and exceed the 120s pytest timeout for n=600.  Mirrors the
    wave 33d pattern in tests/test_combat_polish_parity.py.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    idx = jnp.int32(0)
    vsample = jax.jit(jax.vmap(lambda k: melee_attack(state, k, idx)[1]))
    damages = vsample(keys)
    return [int(d) for d in damages]


def _melee_hit_rate(state, n=_N_TRIALS, seed=17):
    """Hit rate over n trials — vmapped to avoid per-trial JIT retrace."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    idx = jnp.int32(0)
    vsample = jax.jit(jax.vmap(lambda k: melee_attack(state, k, idx)[2]))
    hits = vsample(keys)
    return float(jnp.sum(hits.astype(jnp.int32))) / n


# ---------------------------------------------------------------------------
# 1. Two-handed STR scaling — vendor/nethack/src/uhitm.c:1467-1468
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_two_handed_str_scaling():
    """TWO_HANDED_SWORD with max STR should average more damage than LONG_SWORD.

    vendor/nethack/src/uhitm.c:1467-1468: two-handed weapon scales dbon by 1.5×.
    At STR18(100) dbon=6; two-handed dbon = 6*3//2 = 9 (+3 over long sword).
    """
    state_ls  = _state_with_weapon(_LONG_SWORD_TYPE_ID)
    state_2hs = _state_with_weapon(_TWO_HANDED_SWORD_TYPE_ID)

    dmg_ls  = _melee_damage_samples(state_ls)
    dmg_2hs = _melee_damage_samples(state_2hs)

    hits_ls  = [d for d in dmg_ls  if d > 0]
    hits_2hs = [d for d in dmg_2hs if d > 0]

    assert hits_ls and hits_2hs, "No hits recorded"
    mean_ls  = sum(hits_ls)  / len(hits_ls)
    mean_2hs = sum(hits_2hs) / len(hits_2hs)

    assert mean_2hs > mean_ls, (
        f"two-handed sword mean dmg ({mean_2hs:.2f}) should exceed "
        f"long sword mean dmg ({mean_ls:.2f})"
    )


# ---------------------------------------------------------------------------
# 2. Sleeping target uses sleep_timer — vendor/nethack/src/uhitm.c:387
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_sleeping_target_uses_sleep_timer():
    """sleep_timer=10 should give +2 to-hit vs sleep_timer=0.

    vendor/nethack/src/uhitm.c:387: sleeping target → +2 tmp.
    """
    state = _base_state()
    mai = state.monster_ai

    state_awake = state.replace(
        monster_ai=mai.replace(
            sleep_timer=mai.sleep_timer.at[0].set(jnp.int16(0)),
            asleep=mai.asleep.at[0].set(jnp.bool_(False)),
        )
    )
    state_asleep = state.replace(
        monster_ai=mai.replace(
            sleep_timer=mai.sleep_timer.at[0].set(jnp.int16(10)),
            asleep=mai.asleep.at[0].set(jnp.bool_(True)),
        )
    )

    # Use low-AC target (tough to hit) to make the +2 measurable
    hard_mai_awake  = state_awake.monster_ai.replace(
        ac=state_awake.monster_ai.ac.at[0].set(jnp.int8(-5))
    )
    hard_mai_asleep = state_asleep.monster_ai.replace(
        ac=state_asleep.monster_ai.ac.at[0].set(jnp.int8(-5))
    )
    state_awake  = state_awake.replace(monster_ai=hard_mai_awake)
    state_asleep = state_asleep.replace(monster_ai=hard_mai_asleep)

    rate_awake  = _melee_hit_rate(state_awake)
    rate_asleep = _melee_hit_rate(state_asleep)

    assert rate_asleep > rate_awake, (
        f"Sleeping target hit rate ({rate_asleep:.3f}) should exceed "
        f"awake target hit rate ({rate_awake:.3f})"
    )


# ---------------------------------------------------------------------------
# 3. Paralyzed target — vendor/nethack/src/uhitm.c:393-394
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_paralyzed_target_immobile_bonus():
    """paralyzed_timer=5 should give +4 to-hit (immobile category).

    vendor/nethack/src/uhitm.c:393-394: !mcanmove → tmp += 4.
    """
    state = _base_state()
    mai = state.monster_ai

    # Use low-AC to make the +4 bonus clearly visible
    hard_ac = jnp.int8(-8)
    state_normal = state.replace(
        monster_ai=mai.replace(
            paralyzed_timer=mai.paralyzed_timer.at[0].set(jnp.int16(0)),
            ac=mai.ac.at[0].set(hard_ac),
        )
    )
    state_paralyzed = state.replace(
        monster_ai=mai.replace(
            paralyzed_timer=mai.paralyzed_timer.at[0].set(jnp.int16(5)),
            ac=mai.ac.at[0].set(hard_ac),
        )
    )

    rate_normal    = _melee_hit_rate(state_normal)
    rate_paralyzed = _melee_hit_rate(state_paralyzed)

    assert rate_paralyzed > rate_normal, (
        f"Paralyzed target hit rate ({rate_paralyzed:.3f}) should exceed "
        f"normal target hit rate ({rate_normal:.3f})"
    )


# ---------------------------------------------------------------------------
# 4. Status timers tick down — vendor timeout.c::run_timers pattern
# ---------------------------------------------------------------------------

@pytest.mark.timeout(300)
def test_status_timers_tick_down():
    """After one monsters_step_all tick, sleep_timer should decrease by 1.

    Timers are decremented by 1 (clamped at 0) each game turn.
    """
    state = _base_state()
    mai = state.monster_ai
    # Set sleep_timer=10 on slot 0
    mai = mai.replace(
        sleep_timer=mai.sleep_timer.at[0].set(jnp.int16(10)),
        asleep=mai.asleep.at[0].set(jnp.bool_(True)),
        # Make monster peaceful so monster_turn won't reset asleep via wake
        peaceful=mai.peaceful.at[0].set(jnp.bool_(True)),
    )
    state = state.replace(monster_ai=mai)

    rng = jax.random.PRNGKey(99)
    new_state = monsters_step_all(state, rng)

    timer_after = int(new_state.monster_ai.sleep_timer[0])
    assert timer_after == 9, (
        f"sleep_timer expected 9 after one tick, got {timer_after}"
    )
