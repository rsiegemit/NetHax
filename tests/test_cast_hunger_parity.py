"""Vendor-parity tests: hunger drain on spell cast.

Cite: vendor/nethack/src/spell.c::spelleffects_check lines 1322-1367.
Vendor formula: morehungry(energy * 2) where energy = spelllev * 5.
Wizard reduction: hunger cost -= ACURR(A_INT) for wizards (~line 1340).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import SpellId, cast_spell, _SPELL_LEVELS, _ROLE_WIZARD


def _state_with_known_spell(spell_id: int, nutrition: int = 900,
                             player_int: int = 10, player_role: int = 0,
                             player_pw: int = 100) -> EnvState:
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem = magic.spell_memory.at[spell_id].set(jnp.int32(20000))
    state = state.replace(
        magic=magic.replace(spell_known=new_known, spell_memory=new_mem),
        player_pw=jnp.int32(player_pw),
        player_pw_max=jnp.int32(player_pw),
        player_int=jnp.int8(player_int),
        player_wis=jnp.int8(10),
        player_xl=jnp.int32(10),
        player_role=jnp.int8(player_role),
        status=state.status.replace(nutrition=jnp.int32(nutrition)),
    )
    return state


def test_cast_drains_nutrition():
    """Successful spell cast drains nutrition by spelllev * 5 * 2.

    Cite: vendor/nethack/src/spell.c::spelleffects_check lines 1322-1367.
    MAGIC_MISSILE is level 2 → cost = 2*5*2 = 20 nutrition.
    We use a Wizard with INT=18 at xl=10 to ensure near-certain success,
    then verify nutrition dropped by exactly the expected amount.
    """
    spell_id = int(SpellId.MAGIC_MISSILE)
    lv = int(_SPELL_LEVELS[spell_id])
    expected_cost = lv * 5 * 2

    initial_nutrition = 900
    # Use Wizard with high INT for reliable success
    state = _state_with_known_spell(
        spell_id, nutrition=initial_nutrition,
        player_int=18, player_role=_ROLE_WIZARD, player_pw=100
    )

    # Run multiple seeds to get a success
    for seed in range(50):
        rng = jax.random.PRNGKey(seed + 1)
        new_state, success = cast_spell(state, rng, spell_id)
        if success:
            drain = initial_nutrition - int(new_state.status.nutrition)
            # Wizard reduction: nutrition_cost -= min(nutrition_cost, INT)
            # With INT=18, expected_cost=20: reduction=min(20,18)=18, net=2
            # So wizard drain = max(0, expected_cost - INT) = 2
            wizard_drain = max(0, expected_cost - 18)
            assert drain == wizard_drain, (
                f"Wizard nutrition drain {drain} != expected {wizard_drain} "
                f"(spelllev={lv}, INT=18)"
            )
            return

    raise AssertionError("No successful cast in 50 tries — test inconclusive")


def test_non_wizard_cast_drains_full_nutrition():
    """Non-wizard spell cast drains full spelllev * 5 * 2 nutrition.

    Cite: vendor/nethack/src/spell.c::spelleffects_check lines 1322-1367.
    HEALING is level 1 -> full cost = 1*5*2 = 10 nutrition (no wizard discount).
    Uses Caveman (role=2, spelbase=0, INT-based) at xl=20 for reliable success.
    """
    spell_id = int(SpellId.HEALING)
    lv = int(_SPELL_LEVELS[spell_id])
    expected_cost = lv * 5 * 2  # = 10

    initial_nutrition = 900
    # Caveman (role=2) has spelbase=0 (no penalty) and is INT-based.
    # xl=20 + INT=18 gives near-certain success.
    state = _state_with_known_spell(
        spell_id, nutrition=initial_nutrition,
        player_int=18, player_role=2, player_pw=100
    )
    state = state.replace(player_xl=jnp.int32(20))

    for seed in range(100):
        rng = jax.random.PRNGKey(seed + 100)
        new_state, success = cast_spell(state, rng, spell_id)
        if success:
            drain = initial_nutrition - int(new_state.status.nutrition)
            assert drain == expected_cost, (
                f"Non-wizard nutrition drain {drain} != {expected_cost} "
                f"(spelllev={lv})"
            )
            return

    raise AssertionError("No successful cast in 100 tries — test inconclusive")


def test_wizard_int_reduces_hunger_cost():
    """Wizard with high INT has lower nutrition drain than a non-wizard.

    Cite: vendor/nethack/src/spell.c::spelleffects_check ~line 1340.
    Wizard drain = max(0, spelllev*5*2 - INT).
    Non-wizard drain = spelllev*5*2.
    With INT=18 and HEALING (lv=1, cost=10): wizard drain = max(0,10-18) = 0.
    """
    spell_id = int(SpellId.HEALING)
    lv = int(_SPELL_LEVELS[spell_id])
    full_cost = lv * 5 * 2  # 10

    initial_nutrition = 900

    wizard_state = _state_with_known_spell(
        spell_id, nutrition=initial_nutrition,
        player_int=18, player_role=_ROLE_WIZARD, player_pw=100
    )
    # Caveman (role=2, spelbase=0, INT-based) at xl=20 for reliable non-wizard success
    nonwiz_state = _state_with_known_spell(
        spell_id, nutrition=initial_nutrition,
        player_int=18, player_role=2, player_pw=100
    ).replace(player_xl=jnp.int32(20))

    wizard_drain = None
    nonwiz_drain = None

    for seed in range(50):
        rng = jax.random.PRNGKey(seed + 200)
        if wizard_drain is None:
            ns, ok = cast_spell(wizard_state, rng, spell_id)
            if ok:
                wizard_drain = initial_nutrition - int(ns.status.nutrition)
        if nonwiz_drain is None:
            ns2, ok2 = cast_spell(nonwiz_state, rng, spell_id)
            if ok2:
                nonwiz_drain = initial_nutrition - int(ns2.status.nutrition)
        if wizard_drain is not None and nonwiz_drain is not None:
            break

    assert wizard_drain is not None, "Wizard never succeeded in 50 tries"
    assert nonwiz_drain is not None, "Non-wizard never succeeded in 50 tries"
    assert wizard_drain < nonwiz_drain, (
        f"Wizard drain ({wizard_drain}) should be < non-wizard drain ({nonwiz_drain})"
    )
    # Non-wizard should pay full cost; wizard pays max(0, cost - INT)
    assert nonwiz_drain == full_cost
    expected_wiz = max(0, full_cost - 18)
    assert wizard_drain == expected_wiz, (
        f"Wizard drain {wizard_drain} != expected {expected_wiz}"
    )


def test_failed_cast_no_nutrition_drain():
    """A failed cast (insufficient Pw) must not drain nutrition.

    Cite: vendor/nethack/src/spell.c::spelleffects_check lines 1322-1367
    — hunger only applied on successful cast.
    """
    spell_id = int(SpellId.MAGIC_MISSILE)
    lv = int(_SPELL_LEVELS[spell_id])
    pw_cost = lv * 5

    initial_nutrition = 900
    # Give insufficient Pw to force an early failure (no cast attempt)
    state = _state_with_known_spell(
        spell_id, nutrition=initial_nutrition,
        player_int=10, player_role=0, player_pw=pw_cost - 1
    )

    rng = jax.random.PRNGKey(42)
    new_state, success = cast_spell(state, rng, spell_id)
    assert not success
    assert int(new_state.status.nutrition) == initial_nutrition, (
        "Nutrition should not change on Pw-check failure"
    )
