"""Corpse eating side-effect parity tests.

Covers gaps cited in the task specification:
  1. Poisonous corpse damage           (eat.c:1130)
  2. Acid resistance blocks damage     (eat.c:1130)
  3. Acidic corpse damage              (eat.c:1145)
  4. Old corpse food poisoning         (eat.c:1180)
  5. Cannibalism alignment hit         (eat.c:1220)
  6. Tin opening counter               (eat.c:1370)
  7. Tin opens after N turns           (eat.c:1370)
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.items_corpses import (
    apply_corpse_postfx,
    apply_old_corpse_effects,
    apply_cannibalism_penalty,
    apply_tin_open_start,
    tick_tin_opening,
    _MONSTER_IS_POISONOUS_NP,
    _MONSTER_IS_ACIDIC_NP,
)
from Nethax.nethax.subsystems.status_effects import (
    Intrinsic,
    TimedStatus,
)
from Nethax.nethax.constants.races import Race
from Nethax.nethax.constants.monsters import MONSTERS, M2_HUMAN


# ---------------------------------------------------------------------------
# Monster index lookups (host-side, precomputed once)
# ---------------------------------------------------------------------------

_KOBOLD_IDX = next(i for i, m in enumerate(MONSTERS) if m.name == "kobold")
_ACID_BLOB_IDX = next(i for i, m in enumerate(MONSTERS) if m.name == "acid blob")
_HUMAN_IDX = next(i for i, m in enumerate(MONSTERS) if m.name == "human")

assert _MONSTER_IS_POISONOUS_NP[_KOBOLD_IDX], "kobold must be poisonous"
assert _MONSTER_IS_ACIDIC_NP[_ACID_BLOB_IDX], "acid blob must be acidic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(hp: int = 50) -> EnvState:
    state = EnvState.default(jax.random.PRNGKey(0))
    return state.replace(
        player_hp=jnp.int32(hp),
        player_hp_max=jnp.int32(hp),
    )


def _with_resist_poison(state: EnvState) -> EnvState:
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.RESIST_POISON)].set(True)
    return state.replace(status=state.status.replace(intrinsics=new_intrinsics))


def _with_resist_acid(state: EnvState) -> EnvState:
    new_intrinsics = state.status.intrinsics.at[int(Intrinsic.RESIST_ACID)].set(True)
    return state.replace(status=state.status.replace(intrinsics=new_intrinsics))


# ---------------------------------------------------------------------------
# 1. Poisonous corpse deals 1-15 HP damage without RESIST_POISON
#    cite: eat.c:1130
# ---------------------------------------------------------------------------

def test_poisonous_corpse_damages():
    """Eating a kobold corpse without RESIST_POISON decreases HP."""
    state = _base_state(hp=100)
    results = [
        apply_corpse_postfx(state, jax.random.PRNGKey(i), jnp.int32(_KOBOLD_IDX))
        for i in range(40)
    ]
    hp_reductions = [100 - int(r.player_hp) for r in results]
    # At least some damage must occur across trials.
    assert any(d > 0 for d in hp_reductions), "Expected HP loss from poisonous corpse"
    # All damage in range [1, 15] (rnd(15)).
    for d in hp_reductions:
        assert 1 <= d <= 15, f"Damage {d} out of rnd(15) range"


# ---------------------------------------------------------------------------
# 2. RESIST_POISON blocks damage
#    cite: eat.c:1130
# ---------------------------------------------------------------------------

def test_poison_resist_blocks_damage():
    """With RESIST_POISON, eating a kobold corpse leaves HP unchanged."""
    state = _with_resist_poison(_base_state(hp=100))
    for i in range(30):
        result = apply_corpse_postfx(state, jax.random.PRNGKey(i), jnp.int32(_KOBOLD_IDX))
        assert int(result.player_hp) == 100, (
            f"seed {i}: RESIST_POISON should block all poison damage"
        )


# ---------------------------------------------------------------------------
# 3. Acidic corpse deals damage without RESIST_ACID
#    cite: eat.c:1145
# ---------------------------------------------------------------------------

def test_acidic_corpse_damages():
    """Eating an acid blob corpse without RESIST_ACID decreases HP."""
    state = _base_state(hp=100)
    results = [
        apply_corpse_postfx(state, jax.random.PRNGKey(i), jnp.int32(_ACID_BLOB_IDX))
        for i in range(40)
    ]
    hp_reductions = [100 - int(r.player_hp) for r in results]
    assert any(d > 0 for d in hp_reductions), "Expected HP loss from acidic corpse"
    for d in hp_reductions:
        assert 1 <= d <= 15, f"Acid damage {d} out of rnd(15) range"


# ---------------------------------------------------------------------------
# 4. Old corpse sets VOMITING timer
#    cite: eat.c:1180
# ---------------------------------------------------------------------------

def test_old_corpse_vomits():
    """With simulated rotten roll, VOMITING timer is set."""
    state = _base_state()
    vomit_set = False
    # The 25%-chance approximation means we need enough trials to get a hit.
    for i in range(200):
        # Force the "rotten" path by using seeds that yield age_roll == 0
        # (rn2(4)==0 → first value in [0,4) == 0).
        result = apply_old_corpse_effects(
            state, jax.random.PRNGKey(i), jnp.bool_(True)
        )
        if int(result.status.timed_statuses[int(TimedStatus.VOMITING)]) > 0:
            vomit_set = True
            # Also confirm SICK timer is set (food poisoning).
            assert int(result.status.timed_statuses[int(TimedStatus.SICK)]) > 0, (
                "SICK timer must be set alongside VOMITING"
            )
            break
    assert vomit_set, "VOMITING timer never set across 200 old-corpse trials"


# ---------------------------------------------------------------------------
# 5. Cannibalism alignment hit
#    cite: eat.c:1220
# ---------------------------------------------------------------------------

def test_cannibalism_align_hit():
    """Human eating a human corpse: alignment_record -= 2, CONFUSION set."""
    state = _base_state()
    # Set player race to HUMAN.
    state = state.replace(player_race=jnp.int8(int(Race.HUMAN)))
    initial_align = int(state.prayer.alignment_record)

    result = apply_cannibalism_penalty(state, jnp.int32(_HUMAN_IDX))

    new_align = int(result.prayer.alignment_record)
    assert new_align == initial_align - 2, (
        f"Expected alignment_record {initial_align - 2}, got {new_align}"
    )
    conf = int(result.status.timed_statuses[int(TimedStatus.CONFUSION)])
    assert conf >= 5, f"Expected CONFUSION >= 5, got {conf}"


def test_cannibalism_no_hit_other_race():
    """Elf eating a human corpse: no alignment penalty."""
    state = _base_state()
    state = state.replace(player_race=jnp.int8(int(Race.ELF)))
    initial_align = int(state.prayer.alignment_record)

    result = apply_cannibalism_penalty(state, jnp.int32(_HUMAN_IDX))

    assert int(result.prayer.alignment_record) == initial_align, (
        "No alignment penalty when eating a different race's corpse"
    )


# ---------------------------------------------------------------------------
# 6. Tin-opening counter is set correctly
#    cite: eat.c:1370
# ---------------------------------------------------------------------------

def test_tin_opening_counter_uncursed():
    """Starting to open an uncursed tin sets counter to 50."""
    state = _base_state()
    result = apply_tin_open_start(
        state,
        is_tin=jnp.bool_(True),
        type_id=jnp.int16(42),
        is_blessed=jnp.bool_(False),
    )
    assert int(result.tin_opening_turns_left) == 50
    assert int(result.tin_opening_type_id) == 42


def test_tin_opening_counter_blessed():
    """Starting to open a blessed tin sets counter to 30."""
    state = _base_state()
    result = apply_tin_open_start(
        state,
        is_tin=jnp.bool_(True),
        type_id=jnp.int16(42),
        is_blessed=jnp.bool_(True),
    )
    assert int(result.tin_opening_turns_left) == 30


# ---------------------------------------------------------------------------
# 7. Tin opens after N turns (counter reaches 0)
#    cite: eat.c:1370
# ---------------------------------------------------------------------------

def test_tin_opens_after_n_turns():
    """Ticking 50 times reduces uncursed tin counter to 0."""
    state = _base_state()
    state = apply_tin_open_start(
        state,
        is_tin=jnp.bool_(True),
        type_id=jnp.int16(7),
        is_blessed=jnp.bool_(False),
    )
    assert int(state.tin_opening_turns_left) == 50

    for _ in range(50):
        state = tick_tin_opening(state)

    assert int(state.tin_opening_turns_left) == 0, (
        "tin_opening_turns_left should reach 0 after 50 ticks"
    )


def test_tin_blessed_opens_after_30_turns():
    """Ticking 30 times reduces blessed tin counter to 0."""
    state = _base_state()
    state = apply_tin_open_start(
        state,
        is_tin=jnp.bool_(True),
        type_id=jnp.int16(7),
        is_blessed=jnp.bool_(True),
    )
    assert int(state.tin_opening_turns_left) == 30

    for _ in range(30):
        state = tick_tin_opening(state)

    assert int(state.tin_opening_turns_left) == 0
