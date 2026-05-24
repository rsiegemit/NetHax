"""Wave 5 Phase 3 tests for the containers subsystem.

Canonical: vendor/nethack/src/pickup.c::in_container / out_container /
use_container, vendor/nethack/include/objects.h container defs.
"""
import pytest


def _make_state():
    """Default EnvState with empty inventory and empty containers."""
    import jax
    from Nethax.nethax.state import EnvState
    rng = jax.random.PRNGKey(42)
    return EnvState.default(rng=rng)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_container_state_initial_empty():
    """Fresh state has N_CONTAINERS slots, all empty (NONE type)."""
    from Nethax.nethax.subsystems.containers import (
        N_CONTAINERS, MAX_ITEMS_PER_CONTAINER, ContainerType,
    )

    state = _make_state()
    cs = state.containers
    assert cs.container_type.shape == (N_CONTAINERS,)
    assert cs.items_category.shape == (N_CONTAINERS, MAX_ITEMS_PER_CONTAINER)
    # All slots NONE.
    assert int(cs.container_type.sum()) == 0
    # No items inside any container.
    assert int(cs.items_category.sum()) == 0
    # No container open.
    assert not bool(cs.is_open.any())


# ---------------------------------------------------------------------------
# Open
# ---------------------------------------------------------------------------

def test_open_bag_marks_open():
    """Installing an uncursed bag-of-holding and opening it flips is_open."""
    from Nethax.nethax.subsystems.containers import (
        open_container, install_container, ContainerType, BUCStatus,
    )

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                              buc=int(BUCStatus.UNCURSED))
    new_state = open_container(state, 0)
    assert bool(new_state.containers.is_open[0])


# ---------------------------------------------------------------------------
# Put / Take
# ---------------------------------------------------------------------------

def _place_inventory_apple(state, slot_idx=0, weight=5):
    """Helper: write a FOOD (apple) into inventory slot ``slot_idx``."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import ItemCategory

    inv = state.inventory.items
    new_inv = inv.replace(
        category   = inv.category.at[slot_idx].set(jnp.int8(ItemCategory.FOOD)),
        type_id    = inv.type_id.at[slot_idx].set(jnp.int16(123)),  # apple type_id stub
        quantity   = inv.quantity.at[slot_idx].set(jnp.int16(1)),
        weight     = inv.weight.at[slot_idx].set(jnp.int32(weight)),
        identified = inv.identified.at[slot_idx].set(jnp.bool_(True)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_inv))


def test_put_item_in_bag():
    """slot 0 with apple → bag[0][0] is apple, inventory[0] empty."""
    from Nethax.nethax.subsystems.containers import (
        put_in_container, install_container, ContainerType,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING)
    state = _place_inventory_apple(state, slot_idx=0, weight=5)

    new_state = put_in_container(state, 0, 0)
    assert int(new_state.containers.items_category[0, 0]) == int(ItemCategory.FOOD)
    assert int(new_state.containers.items_type_id[0, 0]) == 123
    # Inventory slot 0 emptied.
    assert int(new_state.inventory.items.category[0]) == 0
    assert int(new_state.inventory.items.quantity[0]) == 0


def test_take_item_from_bag():
    """bag[0][0] apple → inventory[0] apple, bag[0][0] empty."""
    from Nethax.nethax.subsystems.containers import (
        put_in_container, take_from_container, install_container, ContainerType,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING)
    state = _place_inventory_apple(state, slot_idx=0, weight=5)
    state = put_in_container(state, 0, 0)
    # Now bag[0][0] holds the apple, inventory[0] is empty.

    out_state = take_from_container(state, 0, 0)
    assert int(out_state.inventory.items.category[0]) == int(ItemCategory.FOOD)
    assert int(out_state.inventory.items.type_id[0]) == 123
    assert int(out_state.containers.items_category[0, 0]) == 0


# ---------------------------------------------------------------------------
# Bag-of-holding weight multipliers
# ---------------------------------------------------------------------------

def _stuff_bag_with_weights(state, container_idx, weights):
    """Helper: write the given list of weights directly into container slots."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.inventory import ItemCategory
    cs = state.containers
    new_cat = cs.items_category
    new_w   = cs.items_weight
    for i, w in enumerate(weights):
        new_cat = new_cat.at[container_idx, i].set(jnp.int8(ItemCategory.FOOD))
        new_w   = new_w.at[container_idx, i].set(jnp.int16(w))
    return state.replace(containers=cs.replace(
        items_category=new_cat,
        items_weight=new_w,
    ))


def test_bag_of_holding_reduces_weight():
    """5kg uncursed BoH → ceiling((5+1)/2) = 3.

    Audit L #1/#2: vendor mkobj.c::weight 1944-1953 uses ceiling rounds
    for BoH weight reduction, not floors.  Uncursed formula is
    ``(cwt + 1) / 2`` (ceiling of cwt/2): 5 → (5+1)/2 = 3.  The old
    assertion ``w == 2`` was pinning the off-by-one floor bug.
    """
    from Nethax.nethax.subsystems.containers import (
        container_total_weight, install_container, ContainerType, BUCStatus,
    )

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                              buc=int(BUCStatus.UNCURSED))
    # One 5-weight item.
    state = _stuff_bag_with_weights(state, 0, [5])

    w = int(container_total_weight(state.containers, 0))
    # vendor uncursed: (5+1)/2 = 3.
    assert w == 3, f"expected (5+1)//2 = 3, got {w}"


def test_blessed_bag_of_holding_quarter_weight():
    """Blessed BoH gives 1/4 multiplier: 8 → 2."""
    from Nethax.nethax.subsystems.containers import (
        container_total_weight, install_container, ContainerType, BUCStatus,
    )

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                              buc=int(BUCStatus.BLESSED))
    state = _stuff_bag_with_weights(state, 0, [8])

    w = int(container_total_weight(state.containers, 0))
    assert w == 2, f"expected 8//4 = 2, got {w}"


def test_cursed_bag_of_holding_double_weight():
    """Cursed BoH doubles weight: 5 → 10."""
    from Nethax.nethax.subsystems.containers import (
        container_total_weight, install_container, ContainerType, BUCStatus,
    )

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                              buc=int(BUCStatus.CURSED))
    state = _stuff_bag_with_weights(state, 0, [5])

    w = int(container_total_weight(state.containers, 0))
    assert w == 10, f"expected 5*2 = 10, got {w}"


# ---------------------------------------------------------------------------
# Cursed bag-of-tricks explosion
# ---------------------------------------------------------------------------

def test_cursed_bag_of_tricks_explodes_on_open():
    """Seeded RNG with a roll < 0.5 → a hostile monster is spawned (alive)."""
    import jax
    from Nethax.nethax.subsystems.containers import (
        open_container, install_container, ContainerType, BUCStatus,
    )

    # Choose a seed that yields uniform() < 0.5 to trigger explosion.
    found = False
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        roll = float(jax.random.uniform(jax.random.split(rng)[0], shape=()))
        if roll < 0.5:
            found = True
            break
    assert found, "could not find a seed with explode roll"

    state = _make_state()
    state = state.replace(rng=rng)
    state = install_container(state, 0, ContainerType.BAG_OF_TRICKS,
                              buc=int(BUCStatus.CURSED))

    pre_alive_count = int(state.monster_ai.alive.sum())
    new_state = open_container(state, 0)
    post_alive_count = int(new_state.monster_ai.alive.sum())
    assert post_alive_count >= pre_alive_count + 1, (
        f"expected at least one new alive monster; "
        f"pre={pre_alive_count}, post={post_alive_count}"
    )
    # New monster is hostile (peaceful=False).
    # Find which slot got woken: it's the first dead slot in the pre-state.
    import jax.numpy as jnp
    first_dead = int(jnp.argmax(~state.monster_ai.alive))
    assert bool(new_state.monster_ai.alive[first_dead])
    assert not bool(new_state.monster_ai.peaceful[first_dead])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_put_too_many_items_caps_at_max():
    """Filling beyond MAX_ITEMS_PER_CONTAINER is a no-op for extra items."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.containers import (
        put_in_container, install_container, ContainerType,
        MAX_ITEMS_PER_CONTAINER,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING)
    # Pre-fill all MAX_ITEMS_PER_CONTAINER slots in the bag.
    cs = state.containers
    full_cat = cs.items_category.at[0, :].set(jnp.int8(ItemCategory.FOOD))
    full_w   = cs.items_weight.at[0, :].set(jnp.int16(1))
    state = state.replace(containers=cs.replace(
        items_category=full_cat,
        items_weight=full_w,
    ))
    # Now place an apple in inventory and try to put it in.
    state = _place_inventory_apple(state, slot_idx=0, weight=5)

    new_state = put_in_container(state, 0, 0)
    # Inventory was NOT cleared (no room in bag).
    assert int(new_state.inventory.items.category[0]) == int(ItemCategory.FOOD)
    # Container still full of the original 1-weight stubs.
    assert int(new_state.containers.items_category[0, 0]) == int(ItemCategory.FOOD)
    assert int(new_state.containers.items_weight[0, 0]) == 1
    # Total occupied count unchanged.
    assert int((new_state.containers.items_category[0] != 0).sum()) == MAX_ITEMS_PER_CONTAINER


def test_take_from_empty_container_noop():
    """Taking from an empty position in a container leaves state unchanged."""
    from Nethax.nethax.subsystems.containers import (
        take_from_container, install_container, ContainerType,
    )

    state = _make_state()
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING)
    pre_cat_sum = int(state.inventory.items.category.sum())

    new_state = take_from_container(state, 0, 0)
    post_cat_sum = int(new_state.inventory.items.category.sum())
    assert pre_cat_sum == post_cat_sum
