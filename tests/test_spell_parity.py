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
    KEEN,
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


def _with_known_spell(state: EnvState, spell_id: int, memory: int = KEEN):
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(memory))
    return state.replace(magic=magic.replace(spell_known=new_known, spell_memory=new_mem))


# ---------------------------------------------------------------------------
# A. percent_success — exact-value parity (spell.c::percent_success)
# ---------------------------------------------------------------------------

class TestPercentSuccess:
    def test_percent_success_wizard_int16_xl5_healing(self):
        """Wizard (role=12), INT=16, XL=5, casting HEALING (lv1) → success% = 100.

        Audit-K corrected hand-trace per spell.c::percent_success
        (lines 2173-2292).  Vendor: ``skill = max(P_SKILL, P_UNSKILLED) - 1``
        — for an *un*skilled Wizard P_SKILL == P_UNSKILLED == 1, so
        ``skill - 1 == 0`` (not ``-1`` as the previous Nethax impl
        produced).  In Nethax's 0-based encoding, ``skill_adj =
        max(level, 0)`` likewise gives 0.

          splcaster = spelbase(0) + spelheal(1)   = 1
          chance    = 11 * 16 / 2                 = 88
          skill_adj = max(0, 0)                   = 0
          difficulty= (1-1)*4 - (0*6 + 5/3 + 1)   = -2
          learning  = min(15*2/1, 20)             = 20
          chance   += 20                          = 108
          clamp [0,120]                           = 108
          chance    = 108 * (20-1) / 15 - 1
                    = 2052 / 15 - 1               = 136 - 1 = 135
          clamp [0,100]                           = 100
        Cite: vendor/nethack/src/spell.c line 2238.
        """
        s = spell_success_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        )
        assert int(s) == 100, f"Expected 100% success, got {int(s)}%"

        # spell_fail_chance is the back-compat wrapper returning 100 - success.
        f = spell_fail_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        )
        assert int(f) == 0, f"Expected 0% fail, got {int(f)}%"

    def test_percent_success_low_int_high_level_spell(self):
        """Caveman INT=8 XL=1 casting FINGER_OF_DEATH (lv 7) → success% = 0.

        Audit-K corrected hand-trace per spell.c::percent_success:
          splcaster = spelbase(0)                 = 0   # not healing spell
          chance    = 11 * 8 / 2                  = 44
          skill_adj = max(0, 0)                   = 0
          difficulty= (7-1)*4 - (0*6 + 1/3 + 1)   = 24 - 1 = 23
          chance   -= isqrt(900*23 + 2000)
                    = isqrt(22700)                = 150
          chance    = 44 - 150                    = -106 → clamp 0
          chance    = 0 * 20/15 - 0               = 0
          → success% = 0   (cast nearly always fails)
        Cite: vendor/nethack/src/spell.c line 2238.
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
        """Pw cost equals spell_level * 5 on success (spell.h SPELL_LEV_PW).

        Audit-K note: a failed cast drains only ``energy / 2`` Pw
        (vendor spell.c:1374 ``u.uen -= energy / 2``), so we must
        retry seeds until we observe a success to compare against the
        full cost.
        """
        lv = int(_SPELL_LEVELS[spell_id])
        assert lv == expected_level, (
            f"Spell {SpellId(spell_id).name} should be level {expected_level}, got {lv}"
        )
        expected_cost = expected_level * 5
        half_cost     = expected_cost // 2  # vendor C int-div on failure

        state = _base_state(player_pw=50, player_pw_max=50,
                            player_int=18, player_xl=30, player_role=12)
        state = _with_known_spell(state, spell_id)
        # Bump every spell-school skill to P_EXPERT (= 3 in Nethax 0-based)
        # so high-level spells can succeed, and so the full-cost branch is
        # observable for the level-6/7 cases.
        from Nethax.nethax.subsystems.skills import SkillId, SkillLevel
        skills = state.skills
        new_lv = skills.level
        new_cap = skills.max_level
        for sk in (
            SkillId.ATTACK_SPELL, SkillId.HEALING_SPELL,
            SkillId.DIVINATION_SPELL, SkillId.ENCHANTMENT_SPELL,
            SkillId.CLERIC_SPELL, SkillId.ESCAPE_SPELL, SkillId.MATTER_SPELL,
        ):
            new_lv  = new_lv.at[int(sk)].set(jnp.int8(int(SkillLevel.P_EXPERT)))
            new_cap = new_cap.at[int(sk)].set(jnp.int8(int(SkillLevel.P_EXPERT)))
        state = state.replace(skills=skills.replace(level=new_lv, max_level=new_cap))

        # Loop seeds: assert delta is either expected_cost (success) or
        # half_cost (failure).  Require at least one observed success
        # within the seed window to lock the full-cost branch.
        observed_success = False
        for seed in range(40):
            rng = jax.random.PRNGKey(seed + 7)
            new_state, success = cast_spell(state, rng, spell_id)
            delta = int(state.player_pw) - int(new_state.player_pw)
            if success:
                observed_success = True
                assert delta == expected_cost, (
                    f"{SpellId(spell_id).name} success: expected {expected_cost} "
                    f"Pw drain, got {delta}"
                )
                break
            else:
                assert delta == half_cost, (
                    f"{SpellId(spell_id).name} failure: expected {half_cost} "
                    f"Pw drain (energy/2), got {delta}"
                )
        assert observed_success, (
            f"{SpellId(spell_id).name} never succeeded in 40 seeds — "
            f"cannot verify full-cost branch"
        )


# ---------------------------------------------------------------------------
# C. Spell memory decay — decrnknow() = sp_know-- (one per turn)
# ---------------------------------------------------------------------------

class TestSpellMemoryDecay:
    def test_spell_memory_decay_1_per_turn(self):
        """Vendor decrnknow decrements sp_know by 1 per turn (spell.h line 31).

        Vendor parity: ``spelleffects`` does NOT touch ``sp_know`` on cast
        (vendor/nethack/src/spell.c::spelleffects).  Memory only decays via
        the per-turn ``age_spells`` loop (vendor spell.c lines 669-682,
        called from allmain.c::moveloop).  We therefore assert that cast
        leaves ``spell_memory`` unchanged.
        """
        # Module-level constant matches vendor's decrnknow semantics.
        assert SPELL_DECAY_PER_TURN == 1, (
            "Vendor spell.h decrnknow decrements sp_know by 1 per turn."
        )

        # cast_spell must NOT decrement spell_memory (vendor spelleffects
        # does not touch sp_know).
        state = _base_state(player_pw=50, player_pw_max=50,
                            player_int=18, player_xl=30, player_role=12)
        state = _with_known_spell(state, SpellId.HEALING, memory=10)

        rng = jax.random.PRNGKey(42)
        new_state, _ = cast_spell(state, rng, SpellId.HEALING)
        new_mem = int(new_state.magic.spell_memory[SpellId.HEALING])
        assert new_mem == 10, (
            f"Spell memory must not change on cast (vendor parity); got {new_mem}"
        )

    def test_spell_keen_constant_matches_vendor(self):
        """Vendor spell.c line 17: #define KEEN 20000."""
        assert SPELL_KEEN == 20000
        assert KEEN == 20000

    def test_spell_memory_zero_unchanged_by_cast(self):
        """A cast against a spell at memory=0 leaves memory at 0.

        Vendor: ``spelleffects`` does not touch ``sp_know``; an already-
        forgotten spell (``spellknow(i) <= 0``) cannot be cast in vendor,
        but even if dispatched, sp_know is not written.
        """
        state = _base_state(player_pw=50, player_pw_max=50,
                            player_int=18, player_xl=30, player_role=12)
        state = _with_known_spell(state, SpellId.HEALING, memory=0)

        rng = jax.random.PRNGKey(99)
        new_state, _ = cast_spell(state, rng, SpellId.HEALING)
        new_mem = int(new_state.magic.spell_memory[SpellId.HEALING])
        assert new_mem == 0, "Spell memory floor must remain at 0"


# ---------------------------------------------------------------------------
# D. cast_spell — empirical success rate matches vendor percent_success
# (Wave 6 Phase B+ FIX-IT #73)
# ---------------------------------------------------------------------------

class TestCastSpellSuccessRate:
    def test_wizard_int16_xl5_force_bolt_empirical_matches_vendor(self):
        """Wizard INT=16 XL=5 casting FORCE_BOLT (lv 1): empirical rate matches.

        Audit-K corrected vendor success%: with proper skill_adj=0 (not -1),
        an un-armored Wizard at INT=16 XL=5 lands FORCE_BOLT every time
        (success% = 100).  Verify empirical agreement over 200 trials.

        Cite: vendor/nethack/src/spell.c::percent_success line 2238.
        """
        success_pct = int(spell_success_chance(
            jnp.int32(12),               # Wizard
            jnp.int32(SpellId.FORCE_BOLT),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        ))
        # Sanity bound — naked Wizard at xl=5 INT=16 should cast lv1 cleanly.
        assert 80 <= success_pct <= 100, (
            f"Sanity: expected FORCE_BOLT success in [80..100]%, got {success_pct}%"
        )

        n_trials = 200
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
        assert abs(empirical_pct - success_pct) <= 5.0, (
            f"Empirical success rate {empirical_pct:.1f}% should be within "
            f"±5 pp of vendor success% = {success_pct}% ({successes}/{n_trials})"
        )

    def test_cast_spell_succeeds_at_vendor_success_rate(self):
        """Empirical match for Wizard INT=16 XL=5 HEALING.

        Audit-K corrected: vendor percent_success for HEALING here = 100%
        once the skill_adj underflow bug is fixed.  Verify the cast
        succeeds ~100% of the time over 200 trials.
        Cite: vendor/nethack/src/spell.c::percent_success line 2238.
        """
        success_pct = int(spell_success_chance(
            jnp.int32(12),
            jnp.int32(SpellId.HEALING),
            jnp.int32(5),
            jnp.int8(16),
            jnp.int8(10),
        ))
        assert success_pct == 100

        n_trials = 200
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
