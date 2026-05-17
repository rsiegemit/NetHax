"""Parity tests for corpse-eating intrinsic-award system (cpostfx port).

Canonical source: vendor/nethack/src/eat.c::cpostfx lines 1129-1328
                  vendor/nethack/src/eat.c::corpse_intrinsic lines 1338-1373

Tests
-----
- test_floating_eye_grants_telepathy
- test_tengu_grants_teleport
- test_newt_pw_bonus
- test_poisonous_corpse_damages
- test_poison_resist_blocks_damage
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.subsystems.status_effects import (
    StatusState,
    Intrinsic,
    N_INTRINSICS,
)
from Nethax.nethax.subsystems.items_corpses import (
    apply_corpse_postfx,
    _NEWT_IDX_NP,
    _WRAITH_IDX_NP,
    _NURSE_IDX_NP,
    _QUANTUM_MECHANIC_IDX_NP,
)
from Nethax.nethax.state import EnvState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_monster_idx(name: str) -> int:
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise ValueError(f"Monster not found: {name!r}")


def _default_state() -> EnvState:
    rng = jax.random.PRNGKey(0)
    return EnvState.default(rng)


def _eat_corpse(state: EnvState, monster_name: str, rng_seed: int = 42) -> EnvState:
    """Call apply_corpse_postfx for named monster and return new state."""
    idx = _find_monster_idx(monster_name)
    rng = jax.random.PRNGKey(rng_seed)
    return apply_corpse_postfx(state, rng, jnp.int32(idx))


# ---------------------------------------------------------------------------
# test_floating_eye_grants_telepathy
# cite: eat.c::cpostfx default branch → check_intrinsics = TRUE
#       intrinsic_possible TELEPAT case → telepathic(ptr) == TRUE for floating eye
#       (mondata.h line 84: PM_FLOATING_EYE is in the telepathic() macro)
# ---------------------------------------------------------------------------

def test_floating_eye_grants_telepathy():
    """Eating a floating eye corpse must set TELEPATHY intrinsic."""
    state = _default_state()
    assert not bool(state.status.intrinsics[int(Intrinsic.TELEPATHY)]), \
        "TELEPATHY should not be set before eating"

    # Run enough seeds to beat the probabilistic selection (100% for single candidate)
    # Floating eye only grants TELEPATHY, so it must be chosen.
    for seed in range(10):
        result = _eat_corpse(state, "floating eye", rng_seed=seed)
        if bool(result.status.intrinsics[int(Intrinsic.TELEPATHY)]):
            return  # passed

    pytest.fail("TELEPATHY was never granted after 10 floating eye eats")


# ---------------------------------------------------------------------------
# test_tengu_grants_teleport
# cite: eat.c::cpostfx default branch → check_intrinsics = TRUE
#       intrinsic_possible TELEPORT case → can_teleport(ptr) → M1_TPORT flag
#       tengu has M1_TPORT | M1_TPORT_CNTRL and conveys_mask=MR_POISON
# ---------------------------------------------------------------------------

def test_tengu_grants_teleport():
    """Eating a tengu corpse must eventually set TELEPORT intrinsic."""
    state = _default_state()
    granted_teleport = False
    for seed in range(50):
        result = _eat_corpse(state, "tengu", rng_seed=seed)
        if bool(result.status.intrinsics[int(Intrinsic.TELEPORT)]):
            granted_teleport = True
            break

    assert granted_teleport, "TELEPORT never granted after 50 tengu corpse eats"


# ---------------------------------------------------------------------------
# test_newt_pw_bonus
# cite: eat.c::eye_of_newt_buzz lines 1102-1123
#       called from cpostfx line 1312 when pm == PM_NEWT
#       Roughly 1/3 chance of pw_max += 1 per eat.
# ---------------------------------------------------------------------------

def test_newt_pw_bonus():
    """Eating newt corpse over 100 trials should bump pw_max at least once."""
    state = _default_state()
    baseline_pw_max = int(state.player_pw_max)
    bumped = False
    for seed in range(100):
        result = _eat_corpse(state, "newt", rng_seed=seed)
        if int(result.player_pw_max) > baseline_pw_max:
            bumped = True
            break

    assert bumped, (
        f"player_pw_max never increased after 100 newt eats "
        f"(baseline={baseline_pw_max})"
    )


# ---------------------------------------------------------------------------
# test_poisonous_corpse_damages
# cite: eat.c::cprefx / cpostfx — poisonous corpse (M1_POIS) deals rnd(15)
#       hp damage when player lacks RESIST_POISON.
# ---------------------------------------------------------------------------

def test_poisonous_corpse_damages():
    """Eating a poisonous corpse without RESIST_POISON must reduce HP."""
    state = _default_state()
    # Ensure player has no poison resistance.
    assert not bool(state.status.intrinsics[int(Intrinsic.RESIST_POISON)])

    baseline_hp = int(state.player_hp)
    damaged = False
    for seed in range(20):
        result = _eat_corpse(state, "kobold", rng_seed=seed)
        if int(result.player_hp) < baseline_hp:
            damaged = True
            break

    assert damaged, (
        f"HP never decreased after 20 kobold (poisonous) eats "
        f"(baseline_hp={baseline_hp})"
    )


# ---------------------------------------------------------------------------
# test_poison_resist_blocks_damage
# cite: eat.c::cprefx — poisonous corpse damage is skipped when
#       player has POISON_RES intrinsic (Stone_resistance gate equivalent).
# ---------------------------------------------------------------------------

def test_poison_resist_blocks_damage():
    """Eating a poisonous corpse with RESIST_POISON must leave HP unchanged."""
    state = _default_state()
    # Grant poison resistance.
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.RESIST_POISON)].set(True)
    state = state.replace(status=state.status.replace(intrinsics=new_intrinsics))
    assert bool(state.status.intrinsics[int(Intrinsic.RESIST_POISON)])

    baseline_hp = int(state.player_hp)
    for seed in range(20):
        result = _eat_corpse(state, "kobold", rng_seed=seed)
        assert int(result.player_hp) == baseline_hp, (
            f"HP changed despite RESIST_POISON (seed={seed}, "
            f"hp={int(result.player_hp)} vs {baseline_hp})"
        )
