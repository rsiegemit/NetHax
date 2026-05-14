"""Spell-parity tests against vendor/nethack (Wave 6 Phase B+ audit #64).

Verifies bit-equal vendor behaviour for:
  - spell.c::percent_success           — spell success percentage
  - spell.c::spelleffects (Pw cost)    — SPELL_LEV_PW(lvl) = lvl * 5
  - spell.h::decrnknow                 — spell memory decays 1/turn (sp_know--)

The "decays 2 per turn" wording in the audit brief refers to the per-turn
update; vendor's ``decrnknow`` decrements by exactly 1 each turn
(spell.h line 31: ``#define decrnknow(spell) svs.spl_book[spell].sp_know--``).
We follow the vendor source: 1 per turn.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import (
    MAX_SPELL_MEMORY,
    SPELL_DECAY_PER_TURN,
    SPELL_KEEN,
    SpellId,
    _SPELL_LEVELS,
    cast_spell,
    spell_fail_chance,
    spell_success_chance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(
    player_pw: int = 50,
    player_pw_max: int = 50,
    player_hp: int = 20,
    player_hp_max: int = 30,
    player_int: int = 16,
    player_wis: int = 10,
    player_xl: int = 5,
    player_role: int = 12,
) -> EnvState:
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    return state.replace(
        player_pw=jnp.int32(player_pw),
        player_pw_max=jnp.int32(player_pw_max),
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(player_hp_max),
        player_int=jnp.int8(player_int),
        player_wis=jnp.int8(player_wis),
        player_xl=jnp.int32(player_xl),
        player_role=jnp.int8(player_role),
    )


def _with_known_spell(state: EnvState, spell_id: int, memory: int = MAX_SPELL_MEMORY):
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(memory))
    return state.replace(magic=magic.replace(spell_known=new_known, spell_memory=new_mem))


# ---------------------------------------------------------------------------
# A. percent_success — exact-value parity (spell.c::percent_success)
# ---------------------------------------------------------------------------

class TestPercentSuccess:
    def test_percent_success_wizard_int16_xl5_healing(self):
        """Wizard (role=12), INT=16, XL=5, casting HEALING (lv1) → success% = 16.

        Hand-trace per spell.c::percent_success (lines 2173-2292):
          splcaster = spelbase(0) + spelheal(1)   = 1   # HEALING is healing
          splcaster = min(1, 20)                  = 1
          chance    = 11 * 16 / 2                 = 88
          skill     = max(0, 0) - 1               = -1   # P_UNSKILLED
          difficulty= (1-1)*4 - ((-1)*6 + 5/3 + 1)= 4
          chance   -= isqrt(900*4 + 2000)
                    = isqrt(5600)                 = 74
          chance   -= 74                          = 14
          chance    = 14 * (20-1) / 15 - 1
                    = 266 / 15 - 1                = 17 - 1 = 16
        """
        s = spell_success_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        )
        assert int(s) == 16, f"Expected 16% success, got {int(s)}%"

        # spell_fail_chance is the back-compat wrapper returning 100 - success.
        f = spell_fail_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        )
        assert int(f) == 84, f"Expected 84% fail, got {int(f)}%"

    def test_percent_success_low_int_high_level_spell(self):
        """Caveman INT=8 XL=1 casting FINGER_OF_DEATH (lv 7) → success% = 0.

        Hand-trace per spell.c::percent_success:
          splcaster = spelbase(0)                 = 0   # not healing spell
          chance    = 11 * 8 / 2                  = 44
          difficulty= (7-1)*4 - ((-1)*6 + 1/3 + 1)= 24 - (-5) = 29
          chance   -= isqrt(900*29 + 2000)
                    = isqrt(28100)                = 167
          chance    = 44 - 167                    = -123 → clamp 0
          chance    = 0 * 20/15 - 0               = 0
          → success% = 0   (cast nearly always fails)
        """
        s = spell_success_chance(
            jnp.int32(2),   # Caveman
            jnp.int32(SpellId.FINGER_OF_DEATH),
            jnp.int32(1),
            jnp.int8(8),
            jnp.int8(8),
        )
        assert int(s) == 0, f"Expected 0% success for under-leveled caveman, got {int(s)}%"

    def test_percent_success_wizard_int18_xl30_healing(self):
        """Wizard INT=18 XL=30 HEALING — should be 100% (capped)."""
        s = spell_success_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(30),
            jnp.int8(18),
            jnp.int8(10),
        )
        assert int(s) == 100, f"Expected 100% success, got {int(s)}%"


# ---------------------------------------------------------------------------
# B. Pw cost — SPELL_LEV_PW(lvl) = lvl * 5 (spell.h line 36)
# ---------------------------------------------------------------------------

class TestSpellPwCost:
    @pytest.mark.parametrize(
        "spell_id,expected_level",
        [
            (SpellId.HEALING,         1),  # level-1 → 5 Pw
            (SpellId.MAGIC_MISSILE,   2),  # level-2 → 10 Pw
            (SpellId.CAUSE_FEAR,      3),  # level-3 → 15 Pw
            (SpellId.FIREBALL,        4),  # level-4 → 20 Pw
            (SpellId.DIG,             5),  # level-5 → 25 Pw
            (SpellId.POLYMORPH,       6),  # level-6 → 30 Pw
            (SpellId.FINGER_OF_DEATH, 7),  # level-7 → 35 Pw
        ],
    )
    def test_spell_pw_cost_level_n_equals_5n(self, spell_id, expected_level):
        """Pw cost equals spell_level * 5 for every level (spell.h SPELL_LEV_PW)."""
        lv = int(_SPELL_LEVELS[spell_id])
        assert lv == expected_level, (
            f"Spell {SpellId(spell_id).name} should be level {expected_level}, got {lv}"
        )
        expected_cost = expected_level * 5

        # Deduct Pw via cast_spell with a Wizard at high stats; check Pw delta.
        state = _base_state(player_pw=50, player_pw_max=50,
                            player_int=18, player_xl=30, player_role=12)
        state = _with_known_spell(state, spell_id)

        rng = jax.random.PRNGKey(7)
        new_state, _ = cast_spell(state, rng, spell_id)
        delta = int(state.player_pw) - int(new_state.player_pw)
        assert delta == expected_cost, (
            f"{SpellId(spell_id).name} should cost {expected_cost} Pw, deducted {delta}"
        )


# ---------------------------------------------------------------------------
# C. Spell memory decay — decrnknow() = sp_know-- (one per turn)
# ---------------------------------------------------------------------------

class TestSpellMemoryDecay:
    def test_spell_memory_decay_2_per_turn(self):
        """Vendor decrnknow decrements sp_know by 1 per turn (spell.h line 31).

        The audit brief originally specified "2 per turn"; vendor source
        contradicts this and decrements by 1.  We follow the vendor
        canonical value (1/turn) and assert SPELL_DECAY_PER_TURN reflects it.
        """
        # Module-level constant matches vendor's decrnknow semantics.
        assert SPELL_DECAY_PER_TURN == 1, (
            "Vendor spell.h decrnknow decrements sp_know by 1 per turn."
        )

        # cast_spell decrements by 1 (per-cast simplification); confirm.
        state = _base_state(player_pw=50, player_pw_max=50,
                            player_int=18, player_xl=30, player_role=12)
        state = _with_known_spell(state, SpellId.HEALING, memory=10)

        rng = jax.random.PRNGKey(42)
        new_state, _ = cast_spell(state, rng, SpellId.HEALING)
        new_mem = int(new_state.magic.spell_memory[SpellId.HEALING])
        assert new_mem == 9, f"Spell memory should decrement by 1, got {new_mem}"

    def test_spell_keen_constant_matches_vendor(self):
        """Vendor spell.c line 17: #define KEEN 20000."""
        assert SPELL_KEEN == 20000

    def test_spell_memory_floor_zero(self):
        """Spell memory never goes negative (matches u.uen >= 0 invariant)."""
        state = _base_state(player_pw=50, player_pw_max=50,
                            player_int=18, player_xl=30, player_role=12)
        state = _with_known_spell(state, SpellId.HEALING, memory=0)

        rng = jax.random.PRNGKey(99)
        new_state, _ = cast_spell(state, rng, SpellId.HEALING)
        new_mem = int(new_state.magic.spell_memory[SpellId.HEALING])
        assert new_mem == 0, "Spell memory must floor at 0"


# ---------------------------------------------------------------------------
# D. cast_spell — empirical success rate matches vendor percent_success
# (Wave 6 Phase B+ FIX-IT #73)
# ---------------------------------------------------------------------------

class TestCastSpellSuccessRate:
    def test_wizard_int16_xl5_force_bolt_succeeds_about_18_percent(self):
        """Wizard INT=16 XL=5 casting FORCE_BOLT (lv 1): vendor success% = 18.

        Run 1000 trials and verify empirical success rate is within a wide
        tolerance (±5 percentage points) of the vendor-computed success%.

        Wave 6 #73: locks the cast-success semantics — vendor's
        percent_success() returns chance-of-cast; ``rnd(100) > chance`` →
        fail.  Cite: vendor/nethack/src/spell.c::percent_success.
        """
        success_pct = int(spell_success_chance(
            jnp.int32(12),               # Wizard
            jnp.int32(SpellId.FORCE_BOLT),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        ))
        assert 10 <= success_pct <= 30, (
            f"Sanity: expected FORCE_BOLT success in [10..30]%, got {success_pct}%"
        )

        n_trials = 1000
        successes = 0
        for seed in range(n_trials):
            state = _base_state(player_pw=50, player_pw_max=50,
                                player_int=16, player_xl=5, player_role=12)
            state = _with_known_spell(state, SpellId.FORCE_BOLT)
            rng = jax.random.PRNGKey(seed)
            _, success = cast_spell(state, rng, SpellId.FORCE_BOLT)
            if success:
                successes += 1

        empirical_pct = 100 * successes / n_trials
        # ±5 pp tolerance over 1000 trials (std ~1.2 pp at p=0.18).
        assert abs(empirical_pct - success_pct) <= 5.0, (
            f"Empirical success rate {empirical_pct:.1f}% should be within "
            f"±5 pp of vendor success% = {success_pct}% ({successes}/{n_trials})"
        )

    def test_cast_spell_succeeds_at_vendor_success_rate(self):
        """1000-trial empirical match for Wizard INT=16 XL=5 HEALING.

        Vendor percent_success for HEALING here = 16% (see
        test_percent_success_wizard_int16_xl5_healing).  Cast must succeed
        ~16% of the time.  Wave 6 #73 fix verification.
        """
        success_pct = int(spell_success_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        ))
        assert success_pct == 16

        n_trials = 1000
        successes = 0
        for seed in range(n_trials):
            state = _base_state(player_pw=50, player_pw_max=50,
                                player_int=16, player_xl=5, player_role=12)
            state = _with_known_spell(state, SpellId.HEALING)
            rng = jax.random.PRNGKey(seed + 10_000)
            _, success = cast_spell(state, rng, SpellId.HEALING)
            if success:
                successes += 1

        empirical_pct = 100 * successes / n_trials
        assert abs(empirical_pct - success_pct) <= 5.0, (
            f"Empirical HEALING success {empirical_pct:.1f}% should be within "
            f"±5 pp of {success_pct}% ({successes}/{n_trials})"
        )

    def test_zero_percent_success_never_casts(self):
        """A spell with vendor success% = 0 must never succeed.

        Wave 6 #73: locks the semantic that ``failed = roll >= success_pct``;
        when success_pct == 0, no roll can satisfy ``roll < 0``.
        """
        # Caveman INT=8 XL=1 FINGER_OF_DEATH → 0% (see parity test).
        success_pct = int(spell_success_chance(
            jnp.int32(2),
            jnp.int32(SpellId.FINGER_OF_DEATH),
            jnp.int32(1),
            jnp.int8(8),
            jnp.int8(8),
        ))
        assert success_pct == 0

        for seed in range(50):
            state = _base_state(player_pw=50, player_pw_max=50,
                                player_int=8, player_wis=8, player_xl=1,
                                player_role=2)
            state = _with_known_spell(state, SpellId.FINGER_OF_DEATH)
            rng = jax.random.PRNGKey(seed)
            _, success = cast_spell(state, rng, SpellId.FINGER_OF_DEATH)
            assert not success, f"Cast should never succeed at 0%; seed={seed}"
