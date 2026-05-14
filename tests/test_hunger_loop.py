"""Wave 3 integration tests — hunger progression over time.

Tests cover nutrition drain across many ticks and eating to restore nutrition.

Wave 3 hunger logic is implemented by the status_effects agent.  Tests are
guarded with skipif when the feature remains a stub (compute_hunger_state
always returns NOT_HUNGRY in Wave 1).

All imports are lazy so collection never fails.
"""

import pytest


def _nutrition_after_n_steps(n_steps, initial_nutrition=900):
    """Helper: run n_steps of wait action, return final (nutrition, hunger_state).

    Wave 6 parity-fix: clear all spawned monsters and lift player_hp to a
    huge value so the player survives the full ``n_steps`` waiting period;
    otherwise an adjacent monster can kill the test player on turn ~20 and
    the env-step's already-done short-circuit freezes nutrition.
    Cite: vendor/nethack/src/allmain.c::moveloop line 222-244 — the
    once-per-turn block (which includes nh_timeout's hunger drain) only
    runs while the hero is alive.
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(3)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    # Set known initial nutrition.
    status = state.status.replace(nutrition=jnp.int32(initial_nutrition))
    state = state.replace(status=status)

    # Make the player effectively unkillable for the duration of the test:
    # huge HP + dead monsters.  Hunger drain itself is the unit under test.
    state = state.replace(
        player_hp=jnp.int32(10_000),
        player_hp_max=jnp.int32(10_000),
    )
    mai = state.monster_ai
    state = state.replace(
        monster_ai=mai.replace(
            alive=jnp.zeros_like(mai.alive),
        )
    )

    action = jnp.int32(ord("."))  # wait
    for _ in range(n_steps):
        rng, step_rng = jax.random.split(rng)
        state, _, _, _, _ = env.step(state, action, step_rng)

    return int(state.status.nutrition), int(state.status.hunger_state)


def test_hunger_progression():
    """Start at nutrition ~900 (NOT_HUNGRY); after 700 ticks, nutrition < 200 (HUNGRY).

    NetHack drains ~1 nutrition per turn at base rate (eat.c).
    Starting at 900 and waiting 700 turns should bring nutrition well below 200.
    """
    from Nethax.nethax.subsystems.status_effects import HungerState

    # Drain rate is exactly 1/turn (eat.c base rate, no slow-digestion).
    # 900 - 750 = 150 → strictly < 200 (HUNGRY threshold is nutrition <= 200).
    nutrition, hunger_state = _nutrition_after_n_steps(750, initial_nutrition=900)

    assert nutrition < 200, (
        f"Expected nutrition < 200 after 750 ticks, got {nutrition}"
    )
    assert hunger_state == int(HungerState.HUNGRY), (
        f"Expected HungerState.HUNGRY ({int(HungerState.HUNGRY)}), "
        f"got {hunger_state}"
    )


def test_eat_restores_nutrition():
    """At HUNGRY state, eat a food ration -> nutrition increases, hunger=NOT_HUNGRY.

    A food ration restores ~800 nutrition (eat.c: ration = 800 nutrition).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.subsystems.status_effects import HungerState
    from Nethax.nethax.constants.objects import ObjectClass

    rng = jax.random.PRNGKey(5)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    # Force player into HUNGRY state with low nutrition
    status = state.status.replace(
        nutrition=jnp.int32(100),
        hunger_state=jnp.int8(HungerState.HUNGRY),
    )
    state = state.replace(status=status)

    # Place a food ration in inventory slot 0.
    # FOOD_CLASS = 7 (vendor/nethack/include/objclass.h).
    # The _handle_eat wrapper treats weight==0 as the default food-ration
    # nutrition value of 800; that matches the food-ration eat.c default.
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.FOOD_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(1)),
        buc_status=items.buc_status.at[0].set(jnp.int8(2)),
        enchantment=items.enchantment.at[0].set(jnp.int8(0)),
        charges=items.charges.at[0].set(jnp.int8(0)),
        identified=items.identified.at[0].set(jnp.bool_(True)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(0)),
        ac_bonus=items.ac_bonus.at[0].set(jnp.int8(0)),
        is_two_handed=items.is_two_handed.at[0].set(jnp.bool_(False)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))

    nutrition_before = int(state.status.nutrition)
    assert nutrition_before == 100

    # Eat action: 'e' / ord('e') = 101
    action = jnp.int32(ord("e"))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    nutrition_after = int(state.status.nutrition)
    hunger_after = int(state.status.hunger_state)

    assert nutrition_after > nutrition_before, (
        f"Expected nutrition to increase after eating, "
        f"before={nutrition_before}, after={nutrition_after}"
    )
    assert hunger_after == int(HungerState.NOT_HUNGRY), (
        f"Expected HungerState.NOT_HUNGRY ({int(HungerState.NOT_HUNGRY)}) "
        f"after eating, got {hunger_after}"
    )
