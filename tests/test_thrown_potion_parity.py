"""Thrown-potion shatter tests — vendor/nethack/src/dothrow.c:2262-2400 (potionhit).

Verifies that throwing a potion at a monster causes it to shatter and apply
the appropriate liquid effect (sleep, heal, acid, etc.) rather than just
weight-based damage.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(2026)

_BUC_UNCURSED = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_thrown_potion(effect, monster_hp=20, monster_hp_max=None, monster_dist=3):
    """Return a state with:
      - slot 0: one potion of ``effect`` (uncursed, qty=1)
      - monster[0]: alive, ``monster_hp`` HP, ``monster_dist`` tiles east
      - player_str/dex/xl set high so the potion always 'hits' the to-hit roll
    """
    from Nethax.nethax.state import EnvState
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.items_potions import _POTION_BASE_ID

    if monster_hp_max is None:
        monster_hp_max = monster_hp

    state = EnvState.default(_RNG).replace(
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
        player_str=jnp.int16(18 + 100),   # max str → reliable hit
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(10),
    )

    # Slot 0: potion (weight=10 so weight-based dmg = max(10//30,1)=1, overridden)
    type_id = _POTION_BASE_ID + int(effect)
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ItemCategory.POTION))),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(10)),
        buc_status=items.buc_status.at[0].set(jnp.int8(_BUC_UNCURSED)),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))

    # Monster 0: alive, fixed position east of player
    mai = state.monster_ai
    n_slots = mai.alive.shape[0]
    # Clear all monsters first
    mai = mai.replace(alive=jnp.zeros((n_slots,), dtype=bool))
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(monster_hp)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(monster_hp_max)),
        pos=mai.pos.at[0].set(jnp.array([5, 5 + monster_dist], dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
    )
    return state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_thrown_sleep_potion_sleeps_monster():
    """Throwing a sleeping potion at a monster sets its asleep flag True.

    Vendor: dothrow.c::potionhit:2262 — POT_SLEEPING shatters and calls
    sleep_monst (sets asleep).
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.items_potions import PotionEffect

    state = _state_with_thrown_potion(PotionEffect.SLEEPING, monster_hp=20)
    assert not bool(state.monster_ai.asleep[0]), "monster should start awake"

    rng = jax.random.PRNGKey(42)
    # Throw east: direction (0, 1); monster is 3 tiles east
    result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))

    assert bool(result.monster_ai.asleep[0]), (
        "thrown sleeping potion should set monster asleep"
    )


def test_thrown_healing_potion_heals_monster():
    """Throwing a healing potion at a damaged monster restores its HP.

    Vendor: dothrow.c::potionhit:2262 — POT_HEALING shatters and heals target.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.items_potions import PotionEffect

    state = _state_with_thrown_potion(PotionEffect.HEALING, monster_hp=5, monster_hp_max=50)
    hp_before = int(state.monster_ai.hp[0])
    assert hp_before == 5

    rng = jax.random.PRNGKey(42)
    result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))

    hp_after = int(result.monster_ai.hp[0])
    assert hp_after > hp_before, (
        f"thrown healing potion should increase monster HP; before={hp_before}, after={hp_after}"
    )


def test_thrown_acid_potion_damages_monster():
    """Throwing an acid potion at a monster reduces its HP.

    Vendor: dothrow.c::potionhit:2262 — POT_ACID shatters and deals damage.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.items_potions import PotionEffect

    state = _state_with_thrown_potion(PotionEffect.ACID, monster_hp=50)
    hp_before = int(state.monster_ai.hp[0])

    rng = jax.random.PRNGKey(42)
    result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))

    hp_after = int(result.monster_ai.hp[0])
    assert hp_after < hp_before, (
        f"thrown acid potion should damage monster; before={hp_before}, after={hp_after}"
    )


def test_thrown_potion_consumed():
    """Throwing any potion decrements its inventory quantity to 0 (qty started at 1).

    Vendor: dothrow.c::breaks (line 1825) — item is destroyed on throw.
    """
    from Nethax.nethax.subsystems.combat import thrown_attack
    from Nethax.nethax.subsystems.items_potions import PotionEffect

    state = _state_with_thrown_potion(PotionEffect.SLEEPING, monster_hp=20)
    qty_before = int(state.inventory.items.quantity[0])
    assert qty_before == 1

    rng = jax.random.PRNGKey(42)
    result = thrown_attack(state, rng, jnp.int32(0), jnp.array([0, 1], dtype=jnp.int32))

    qty_after = int(result.inventory.items.quantity[0])
    assert qty_after == 0, (
        f"thrown potion should be consumed (qty→0); got qty={qty_after}"
    )
    # Category should be cleared when qty reaches 0
    cat_after = int(result.inventory.items.category[0])
    assert cat_after == 0, (
        f"slot category should be cleared after consuming last potion; got {cat_after}"
    )
