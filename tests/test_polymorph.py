"""Wave 4 polymorph subsystem tests.

Covers:
  - Player polymorph: save originals, swap form, recompute AC, attack-set
    swap, drop incompatible armor, timer range, revert on expiry.
  - Monster polymorph: entry_idx change, HP scaling, orig_entry_idx save.
  - Lycanthropy bookkeeping fields and timer decrement.
  - Conduct: POLYSELFLESS violated.
  - Controlled poly: counter increments + poly_controlled flag set.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.polymorph import (
    polymorph_player,
    polymorph_monster,
    revert_polymorph,
    step as poly_step,
    _monster_tables,
    _form_ac,
    _can_wear_armor,
    NATTK,
)
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
from Nethax.nethax.constants.monsters import MONSTERS


_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_form_with_attack() -> int:
    """Return an index of a monster form that has at least one attack."""
    for i, m in enumerate(MONSTERS):
        if m.attacks and m.attacks[0][0] != 0:
            return i
    return 0


def _find_form_with_hands() -> int:
    """Return an index whose form is humanoid + has hands (can wear armor)."""
    M1_HUMANOID = 0x00020000
    M1_NOHANDS = 0x00002000
    for i, m in enumerate(MONSTERS):
        if (m.flags1 & M1_HUMANOID) and not (m.flags1 & M1_NOHANDS):
            return i
    return 0


def _find_form_without_hands() -> int:
    """Return an index whose form has no hands (forces armor drop)."""
    M1_NOHANDS = 0x00002000
    for i, m in enumerate(MONSTERS):
        if m.flags1 & M1_NOHANDS:
            return i
    return 0


def _base_state(armor_worn: bool = False) -> EnvState:
    state = EnvState.default(_RNG)
    state = state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(12),
        player_con=jnp.int8(14),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
        player_role=jnp.int8(0),
        player_ac=jnp.int32(10),
    )
    if armor_worn:
        new_worn = state.inventory.worn_armor.at[0].set(jnp.int8(3))  # arbitrary slot
        state = state.replace(inventory=state.inventory.replace(worn_armor=new_worn))
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPolymorphPlayerSaveOriginals:
    def test_orig_str_dex_con_saved(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        assert int(new.polymorph.orig_str) == 18
        assert int(new.polymorph.orig_dex) == 12
        assert int(new.polymorph.orig_con) == 14

    def test_orig_hp_max_and_ac_saved(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        assert int(new.polymorph.orig_hp_max) == 20
        assert int(new.polymorph.orig_ac) == 10


class TestPolymorphChangeForm:
    def test_current_form_idx_set(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        assert int(new.polymorph.current_form_idx) == target
        assert bool(new.polymorph.is_polymorphed) is True

    def test_attack_set_swapped(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        # First attack of the chosen form should be reflected in attack_types[0]
        expected_type = int(MONSTERS[target].attacks[0][0])
        assert int(new.polymorph.attack_types[0]) == expected_type


class TestPolymorphRecomputeAC:
    def test_ac_matches_form_base(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        expected_ac = int(MONSTERS[target].ac)
        assert int(new.player_ac) == expected_ac


class TestPolymorphArmorDrop:
    def test_drops_when_no_hands(self):
        target = _find_form_without_hands()
        state = _base_state(armor_worn=True)
        new = polymorph_player(state, _RNG, target, controlled=False)
        # All worn-armor slots should be -1 (empty)
        for i in range(N_ARMOR_SLOTS):
            assert int(new.inventory.worn_armor[i]) == -1

    def test_keeps_armor_when_humanoid_with_hands(self):
        target = _find_form_with_hands()
        state = _base_state(armor_worn=True)
        # Sanity: the chosen form should pass _can_wear_armor
        assert bool(_can_wear_armor(jnp.int16(target)))
        new = polymorph_player(state, _RNG, target, controlled=False)
        # Slot 0 was set to 3; it should still hold 3.
        assert int(new.inventory.worn_armor[0]) == 3


class TestPolyTimerRange:
    def test_timer_in_expected_range(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        t = int(new.polymorph.poly_timer)
        assert 500 <= t < 1000, f"poly_timer out of range: {t}"


class TestRevertOnExpiry:
    def test_revert_restores_originals(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        # Force the timer to 1, then step → should expire and revert
        new = new.replace(polymorph=new.polymorph.replace(poly_timer=jnp.int16(1)))
        new = poly_step(new, _RNG)
        # After one step from timer=1 → new_timer=0 → revert.
        assert bool(new.polymorph.is_polymorphed) is False
        assert int(new.player_str) == 18
        assert int(new.player_dex) == 12
        assert int(new.player_con) == 14
        assert int(new.player_hp_max) == 20
        assert int(new.player_ac) == 10

    def test_direct_revert_clears_form(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        reverted = revert_polymorph(new, _RNG)
        assert bool(reverted.polymorph.is_polymorphed) is False
        assert int(reverted.polymorph.current_form_idx) == -1


class TestMonsterPolymorph:
    def _state_with_monster(self):
        state = _base_state()
        mai = state.monster_ai
        mai = mai.replace(
            alive=mai.alive.at[0].set(True),
            hp=mai.hp.at[0].set(jnp.int32(10)),
            hp_max=mai.hp_max.at[0].set(jnp.int32(20)),
            entry_idx=mai.entry_idx.at[0].set(jnp.int16(5)),
        )
        return state.replace(monster_ai=mai)

    def test_entry_idx_changes(self):
        state = self._state_with_monster()
        target = 42
        new = polymorph_monster(state, _RNG, 0, target)
        assert int(new.monster_ai.entry_idx[0]) == target

    def test_orig_entry_idx_saved(self):
        state = self._state_with_monster()
        new = polymorph_monster(state, _RNG, 0, 42)
        assert int(new.monster_ai.orig_entry_idx[0]) == 5

    def test_hp_scaled_proportionally(self):
        state = self._state_with_monster()
        new = polymorph_monster(state, _RNG, 0, 0)
        # Original hp ratio was 10/20 = 0.5; new hp should remain near 50% of new max.
        new_hp = int(new.monster_ai.hp[0])
        new_max = int(new.monster_ai.hp_max[0])
        assert new_hp >= 1
        # Ratio should be in [0.3, 0.7] given proportional scaling (allow rounding)
        ratio = new_hp / max(new_max, 1)
        assert 0.3 <= ratio <= 0.7, f"unexpected scaled hp ratio: {ratio}"


class TestConductPolyselfless:
    def test_violated_on_player_poly(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        assert bool(new.conduct.violations[int(Conduct.POLYSELFLESS)]) is True


class TestControlledPoly:
    def test_controlled_flag_and_count(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=True)
        assert bool(new.polymorph.poly_controlled) is True
        assert int(new.polymorph.controlled_poly_count) == 1

    def test_controlled_picks_specified_target(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=True)
        assert int(new.polymorph.current_form_idx) == target


class TestStepTimerDecrement:
    def test_timer_decrements(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        before = int(new.polymorph.poly_timer)
        new2 = poly_step(new, _RNG)
        after = int(new2.polymorph.poly_timer)
        assert after == before - 1

    def test_lycanthropy_timer_decrements(self):
        state = _base_state()
        # Set a fake lycanthropy timer.
        new = state.replace(polymorph=state.polymorph.replace(
            lycanthropy_timer=jnp.int16(5)))
        new = poly_step(new, _RNG)
        assert int(new.polymorph.lycanthropy_timer) == 4


class TestPolymorphJIT:
    def test_polymorph_player_jits(self):
        """polymorph_player should be JIT-compilable."""
        state = _base_state()
        target = _find_form_with_attack()
        # Use a static target_form_idx (int) so the function compiles cleanly.
        f = jax.jit(lambda s, r: polymorph_player(s, r, target, False))
        new = f(state, _RNG)
        assert bool(new.polymorph.is_polymorphed) is True

    def test_step_jits(self):
        state = _base_state()
        target = _find_form_with_attack()
        new = polymorph_player(state, _RNG, target, controlled=False)
        f = jax.jit(poly_step)
        new2 = f(new, _RNG)
        assert int(new2.polymorph.poly_timer) == int(new.polymorph.poly_timer) - 1
