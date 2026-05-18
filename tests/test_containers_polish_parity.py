"""Container polish parity tests.

Covers:
  - Bag-of-holding weight reduction on put-in (uncursed → ×½, blessed → ×¼)
  - Cursed bag-of-holding item consumption (cursed_bag_consume, 1-in-10)
  - Bag of tricks monster spawning (use_bag_of_tricks, charges decrement)
  - Container trap on open (buc_status sentinel 4 → 1d10 damage)

Canonical sources: vendor/nethack/src/pickup.c::use_container,
                   pickup.c::in_container, pickup.c::bagotricks,
                   pickup.c::container_trap.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_state():
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(42))
    return state


def _put_item_in_slot(state, slot, weight=100):
    """Place a generic food item of given weight into inventory slot."""
    from Nethax.nethax.subsystems.inventory import ItemCategory
    inv = state.inventory.items
    inv = inv.replace(
        category=inv.category.at[slot].set(jnp.int8(int(ItemCategory.FOOD))),
        type_id=inv.type_id.at[slot].set(jnp.int16(1)),
        quantity=inv.quantity.at[slot].set(jnp.int16(1)),
        weight=inv.weight.at[slot].set(jnp.int32(weight)),
    )
    # Also bump total_weight so the starting value is consistent.
    new_tw = state.inventory.total_weight + jnp.int32(weight)
    return state.replace(inventory=state.inventory.replace(items=inv, total_weight=new_tw))


# ---------------------------------------------------------------------------
# 1. Uncursed BoH reduces weight by half (pickup.c::in_container)
# ---------------------------------------------------------------------------

def test_uncursed_boh_reduces_weight():
    """Putting a 100-weight item into an uncursed BoH increases total_weight by 50.

    Canonical: pickup.c::in_container — uncursed multiplier is 1/2.
    """
    from Nethax.nethax.subsystems.containers import (
        install_container, put_in_container, ContainerType, BUCStatus,
    )

    state = _base_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                               buc=int(BUCStatus.UNCURSED))
    state = _put_item_in_slot(state, slot=0, weight=100)

    weight_before = int(state.inventory.total_weight)
    state = put_in_container(state, container_idx=0, src_slot=0)
    weight_after = int(state.inventory.total_weight)

    delta = weight_after - weight_before
    # Item was already counted at full weight; moving into uncursed BoH should
    # apply a -50 correction (effective 50, was 100).
    assert delta == -50, f"Expected weight delta -50 for uncursed BoH, got {delta}"


# ---------------------------------------------------------------------------
# 2. Blessed BoH reduces weight to 1/4 (pickup.c::in_container)
# ---------------------------------------------------------------------------

def test_blessed_boh_reduces_more():
    """Putting a 100-weight item into a blessed BoH increases total_weight by 25.

    Canonical: pickup.c::in_container — blessed multiplier is 1/4.
    """
    from Nethax.nethax.subsystems.containers import (
        install_container, put_in_container, ContainerType, BUCStatus,
    )

    state = _base_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                               buc=int(BUCStatus.BLESSED))
    state = _put_item_in_slot(state, slot=0, weight=100)

    weight_before = int(state.inventory.total_weight)
    state = put_in_container(state, container_idx=0, src_slot=0)
    weight_after = int(state.inventory.total_weight)

    delta = weight_after - weight_before
    # blessed BoH: effective weight = 25, was 100 → delta = -75.
    assert delta == -75, f"Expected weight delta -75 for blessed BoH, got {delta}"


# ---------------------------------------------------------------------------
# 3. Cursed BoH eats items (pickup.c::use_container)
# ---------------------------------------------------------------------------

def test_cursed_bag_eats():
    """cursed_bag_consume run 100× eventually destroys at least one item.

    Canonical: pickup.c::use_container — cursed BoH has 1/10 chance per op
    of eating a contained item.
    """
    from Nethax.nethax.subsystems.containers import (
        install_container, put_in_container, cursed_bag_consume,
        ContainerType, BUCStatus,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _base_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                               buc=int(BUCStatus.CURSED))

    # Load 5 items into the bag directly via put_in_container.
    for slot in range(5):
        state = _put_item_in_slot(state, slot=slot, weight=10)
        state = put_in_container(state, container_idx=0, src_slot=slot)

    items_before = int(jnp.sum(
        state.containers.items_category[0] != jnp.int8(0)
    ))
    assert items_before == 5, "Setup: 5 items should be in the bag"

    # Run 100 consume operations; with p=0.1 the chance of zero eats is 0.9^100 < 3e-5.
    rng = jax.random.PRNGKey(7)
    for i in range(100):
        rng, sub = jax.random.split(rng)
        state = cursed_bag_consume(state, sub, container_idx=0)

    items_after = int(jnp.sum(
        state.containers.items_category[0] != jnp.int8(0)
    ))
    assert items_after < items_before, (
        f"Expected cursed BoH to eat at least one item over 100 ops; "
        f"before={items_before} after={items_after}"
    )


# ---------------------------------------------------------------------------
# 4. Bag of tricks spawns monsters (pickup.c::bagotricks)
# ---------------------------------------------------------------------------

def test_bag_of_tricks_spawns():
    """use_bag_of_tricks called 10× should increase alive monster count.

    Canonical: pickup.c::bagotricks — each use spawns a random low-tier
    monster and decrements charges.
    """
    from Nethax.nethax.subsystems.containers import (
        install_container, use_bag_of_tricks, ContainerType, BUCStatus,
    )

    state = _base_state()
    state = install_container(state, 0, ContainerType.BAG_OF_TRICKS,
                               buc=int(BUCStatus.UNCURSED))

    # Give the bag 10 charges (stored in items_charges[0, 0]).
    cs = state.containers
    cs = cs.replace(
        items_charges=cs.items_charges.at[0, 0].set(jnp.int8(10))
    )
    state = state.replace(containers=cs)

    alive_before = int(jnp.sum(state.monster_ai.alive))

    rng = jax.random.PRNGKey(13)
    for i in range(10):
        rng, sub = jax.random.split(rng)
        state = use_bag_of_tricks(state, sub)

    alive_after = int(jnp.sum(state.monster_ai.alive))
    assert alive_after > alive_before, (
        f"Expected monsters to spawn from bag of tricks; "
        f"before={alive_before} after={alive_after}"
    )
    # Charges should have been fully spent.
    charges_left = int(state.containers.items_charges[0, 0])
    assert charges_left == 0, f"Expected 0 charges remaining, got {charges_left}"


# ---------------------------------------------------------------------------
# 5. Container trap fires on open (pickup.c::container_trap)
# ---------------------------------------------------------------------------

def test_container_trap_on_open():
    """Opening a trapped container (buc sentinel=4) deals 1d10 HP damage.

    Canonical: pickup.c::container_trap — otrapped flag triggers explosion on
    open; modelled here as buc_status sentinel value 4.
    """
    from Nethax.nethax.subsystems.containers import (
        install_container, open_container, ContainerType,
    )

    _TRAPPED_SENTINEL = 4

    state = _base_state()
    # Install a large box with the trapped sentinel as its buc value.
    state = install_container(state, 0, ContainerType.LARGE_BOX,
                               buc=_TRAPPED_SENTINEL)

    hp_before = int(state.player_hp)
    state = open_container(state, slot_idx=0)
    hp_after = int(state.player_hp)

    assert hp_after < hp_before, (
        f"Expected HP to decrease after opening trapped container; "
        f"before={hp_before} after={hp_after}"
    )
    # Trap sentinel should be cleared after firing.
    from Nethax.nethax.subsystems.containers import BUCStatus
    buc_after = int(state.containers.container_buc[0])
    assert buc_after != _TRAPPED_SENTINEL, (
        f"Trapped sentinel should be cleared after trap fires, got buc={buc_after}"
    )
