"""Wave 3 integration tests — item use flow across subsystems.

Covers: potion quaff, scroll read, wand zap charge decrement, zero-charge wand.

All Wave 3 item-use logic is implemented by parallel agents.  Each test is
guarded with skipif when the feature remains a stub.

All imports are lazy so collection never fails.
"""

import pytest


def _make_env_with_item(category_val, type_id_val, charges_val=0, quantity_val=1):
    """Helper: reset env and place an item in inventory slot 0.

    Wave 4: InventoryState.items is a batched ``Item`` of length
    ``MAX_INVENTORY_SLOTS``; we overwrite slot 0 in-place rather than
    replacing the field with a scalar Item (which would break handler
    indexing).

    Returns (env, state, rng).
    """
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(7)
    env = NethaxEnv()
    state, _ = env.reset(rng)

    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(category_val)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id_val)),
        buc_status=items.buc_status.at[0].set(jnp.int8(2)),  # uncursed
        enchantment=items.enchantment.at[0].set(jnp.int8(0)),
        charges=items.charges.at[0].set(jnp.int8(charges_val)),
        identified=items.identified.at[0].set(jnp.bool_(True)),
        quantity=items.quantity.at[0].set(jnp.int16(quantity_val)),
        weight=items.weight.at[0].set(jnp.int32(0)),
        ac_bonus=items.ac_bonus.at[0].set(jnp.int8(0)),
        is_two_handed=items.is_two_handed.at[0].set(jnp.bool_(False)),
    )
    inv = state.inventory.replace(items=new_items)
    state = state.replace(inventory=inv)
    return env, state, rng


# ---------------------------------------------------------------------------
# Potion quaff
# ---------------------------------------------------------------------------

def test_potion_quaff_consumes():
    """Quaff a potion of healing -> slot quantity becomes 0 (consumed)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants.objects import ObjectClass

    # POTION_CLASS = 8 in vendor/nethack/include/objclass.h.
    # type_id 1 = a sentinel healing potion (Wave 5 will define full id map).
    env, state, rng = _make_env_with_item(
        category_val=int(ObjectClass.POTION_CLASS), type_id_val=1, quantity_val=1,
    )

    qty_before = int(state.inventory.items.quantity[0])
    assert qty_before == 1, f"Expected quantity=1 before quaff, got {qty_before}"

    # Quaff action: 'q' in NLE / ord('q') = 113
    action = jnp.int32(ord("q"))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    qty_after = int(state.inventory.items.quantity[0])
    assert qty_after == 0, (
        f"Expected quantity=0 after quaff, got {qty_after}"
    )


# ---------------------------------------------------------------------------
# Scroll read
# ---------------------------------------------------------------------------

def test_scroll_read_consumes():
    """Read a scroll -> slot quantity becomes 0 (consumed)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants.objects import ObjectClass

    # SCROLL_CLASS = 9 (vendor/nethack/include/objclass.h).
    env, state, rng = _make_env_with_item(
        category_val=int(ObjectClass.SCROLL_CLASS), type_id_val=1, quantity_val=1,
    )

    qty_before = int(state.inventory.items.quantity[0])
    assert qty_before == 1, f"Expected quantity=1 before read, got {qty_before}"

    # Read action: 'r' / ord('r') = 114
    action = jnp.int32(ord("r"))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    qty_after = int(state.inventory.items.quantity[0])
    assert qty_after == 0, (
        f"Expected quantity=0 after read, got {qty_after}"
    )


# ---------------------------------------------------------------------------
# Wand zap — decrements charges
# ---------------------------------------------------------------------------

def test_wand_zap_decrements_charges():
    """Zap a wand of striking with 5 charges -> charges decreases by 1."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.items_wands import ITEM_CATEGORY_WAND

    # ITEM_CATEGORY_WAND = 11.  type_id 1 = wand of striking (WandEffect.STRIKING).
    env, state, rng = _make_env_with_item(
        category_val=int(ITEM_CATEGORY_WAND), type_id_val=1, charges_val=5
    )

    charges_before = int(state.inventory.items.charges[0])
    assert charges_before == 5, f"Expected 5 charges before zap, got {charges_before}"

    # Zap action: 'z' / ord('z') = 122
    action = jnp.int32(ord("z"))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    charges_after = int(state.inventory.items.charges[0])
    assert charges_after == 4, (
        f"Expected 4 charges after zap, got {charges_after}"
    )


def test_wand_zero_charges():
    """Zap wand at 0 charges -> no effect, charges stay 0."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.items_wands import ITEM_CATEGORY_WAND

    # Wand with 0 charges
    env, state, rng = _make_env_with_item(
        category_val=int(ITEM_CATEGORY_WAND), type_id_val=1, charges_val=0
    )

    charges_before = int(state.inventory.items.charges[0])
    assert charges_before == 0, f"Expected 0 charges, got {charges_before}"

    hp_before = int(state.player_hp)

    # Zap action: 'z'
    action = jnp.int32(ord("z"))
    rng, step_rng = jax.random.split(rng)
    state, _, _, _, _ = env.step(state, action, step_rng)

    charges_after = int(state.inventory.items.charges[0])
    assert charges_after == 0, (
        f"Charges should remain 0 after zapping empty wand, got {charges_after}"
    )
    # No self-damage from zapping empty wand
    assert int(state.player_hp) == hp_before, (
        "HP should not change when zapping an empty wand"
    )
