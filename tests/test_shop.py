"""Wave 6 Phase A — simplified shop tests.

Covers:
    - accrue_bill: pickup inside shop accrues bill; outside does not.
    - pay_at_exit: deducts gold when affordable; angers shopkeeper otherwise.
    - shopkeeper_attack: angry shopkeeper pursues and bites player.
    - drop_in_shop: dropping in shop clears the owned flag and refunds bill.
    - kill_shopkeeper: clears bill but flips angry=True.
    - ShopState defaults.
    - Mine Town shop registration: make_mine_town_shop_state sets shop_active.

Citations:
    vendor/nethack/src/shk.c::pay_for_obj  — bill accrual
    vendor/nethack/src/shk.c::dopayobj     — pay at exit
    vendor/nethack/src/shk.c::hot_pursuit  — angry shopkeeper mode
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax import NethaxEnv
from Nethax.nethax.subsystems.shop import (
    ShopState,
    accrue_bill,
    drop_in_shop,
    pay_at_exit,
    shopkeeper_attack,
    shop_step,
    kill_shopkeeper,
    DEFAULT_ITEM_PRICE,
    SHOPKEEPER_ANGRY_DAMAGE,
)
from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS
from Nethax.nethax.dungeon.special_levels import make_mine_town_shop_state


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_shop_state(
    row_min=5, col_min=5, row_max=7, col_max=10,
    door_row=8, door_col=5, shopkeeper_idx=0,
):
    """Helper: build an active ShopState with the given bounds and door."""
    return ShopState(
        shop_active=jnp.bool_(True),
        shopkeeper_idx=jnp.int8(shopkeeper_idx),
        shop_room_min=jnp.array([row_min, col_min], dtype=jnp.int8),
        shop_room_max=jnp.array([row_max, col_max], dtype=jnp.int8),
        door_pos=jnp.array([door_row, door_col], dtype=jnp.int8),
        bill=jnp.int32(0),
        items_owned_by_shop=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        angry=jnp.bool_(False),
    )


def _env_with_shop(shop=None, player_pos=(6, 6), player_gold=100,
                   shopkeeper_pos=(5, 5), shopkeeper_alive=True):
    """Build a fresh NethaxEnv state with a configured shop + placed shopkeeper."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(42))

    if shop is None:
        shop = _make_shop_state()

    # Place player inside the shop room by default.
    new_pp = jnp.array(player_pos, dtype=jnp.int16)

    # Place the shopkeeper monster at slot shopkeeper_idx.
    idx = int(shop.shopkeeper_idx)
    mai = state.monster_ai
    new_pos = mai.pos.at[idx].set(jnp.array(shopkeeper_pos, dtype=jnp.int16))
    new_alive = mai.alive.at[idx].set(jnp.bool_(shopkeeper_alive))
    new_hp = mai.hp.at[idx].set(jnp.int32(50))
    new_hp_max = mai.hp_max.at[idx].set(jnp.int32(50))
    new_mai = mai.replace(pos=new_pos, alive=new_alive, hp=new_hp, hp_max=new_hp_max)

    return state.replace(
        shop=shop,
        player_pos=new_pp,
        player_gold=jnp.int32(player_gold),
        monster_ai=new_mai,
    )


# ---------------------------------------------------------------------------
# 1. ShopState defaults
# ---------------------------------------------------------------------------
def test_shop_state_initial_inactive():
    """Default ShopState reports no shop on the level."""
    shop = ShopState.default()
    assert bool(shop.shop_active) is False
    assert int(shop.shopkeeper_idx) == -1
    assert int(shop.bill) == 0
    assert bool(shop.angry) is False
    assert shop.items_owned_by_shop.shape == (MAX_INVENTORY_SLOTS,)
    assert not bool(jnp.any(shop.items_owned_by_shop))


# ---------------------------------------------------------------------------
# 2. Bill accrual
# ---------------------------------------------------------------------------
def test_pickup_in_shop_accrues_bill():
    """Pickup inside the shop room flags the slot and bumps the bill."""
    state = _env_with_shop(player_pos=(6, 7))  # inside (rows 5-7, cols 5-10)
    new_state = accrue_bill(state, slot_idx=3)
    assert int(new_state.shop.bill) == DEFAULT_ITEM_PRICE
    assert bool(new_state.shop.items_owned_by_shop[3]) is True
    # Other slots untouched.
    assert bool(new_state.shop.items_owned_by_shop[0]) is False


def test_pickup_outside_shop_no_bill():
    """Pickup outside the shop room is a no-op."""
    state = _env_with_shop(player_pos=(15, 15))  # outside any room
    new_state = accrue_bill(state, slot_idx=3)
    assert int(new_state.shop.bill) == 0
    assert not bool(jnp.any(new_state.shop.items_owned_by_shop))


def test_pickup_when_shop_inactive_is_noop():
    """Pickup when no shop is active does nothing."""
    shop = ShopState.default()
    state = _env_with_shop(shop=shop, player_pos=(6, 6))
    new_state = accrue_bill(state, slot_idx=3)
    assert int(new_state.shop.bill) == 0
    assert not bool(jnp.any(new_state.shop.items_owned_by_shop))


# ---------------------------------------------------------------------------
# 3. Pay at exit
# ---------------------------------------------------------------------------
def test_pay_at_exit_deducts_gold():
    """Crossing the door tile with enough gold settles the bill.

    Wave 15 (shop vendor-parity): accrual against EMPTY slots resolves to
    ``DEFAULT_ITEM_PRICE`` via the fallback in ``_compute_item_price_from_slot``;
    slots 0-2 carry starting inventory whose real ``get_cost`` differs by
    role and CHA tier.  This test exercises the pay path, not pricing.
    """
    state = _env_with_shop(player_pos=(6, 6), player_gold=100)
    # Accrue a bill of 30gp against three empty slots (post-starting inv).
    state = accrue_bill(state, slot_idx=5)
    state = accrue_bill(state, slot_idx=6)
    state = accrue_bill(state, slot_idx=7)
    assert int(state.shop.bill) == 3 * DEFAULT_ITEM_PRICE
    # Move the player to the door.
    state = state.replace(player_pos=jnp.array([8, 5], dtype=jnp.int16))
    new_state = pay_at_exit(state)
    assert int(new_state.shop.bill) == 0
    assert int(new_state.player_gold) == 100 - 3 * DEFAULT_ITEM_PRICE
    assert not bool(jnp.any(new_state.shop.items_owned_by_shop))
    assert bool(new_state.shop.angry) is False


def test_pay_at_exit_insufficient_gold_angers_shopkeeper():
    """Crossing the door tile broke flips angry=True; bill remains.

    Wave 15: see note in test_pay_at_exit_deducts_gold — empty slots used
    so the DEFAULT_ITEM_PRICE fallback applies.
    """
    state = _env_with_shop(player_pos=(6, 6), player_gold=5)
    state = accrue_bill(state, slot_idx=5)
    state = accrue_bill(state, slot_idx=6)
    # bill = 20, gold = 5 → broke.
    state = state.replace(player_pos=jnp.array([8, 5], dtype=jnp.int16))
    new_state = pay_at_exit(state)
    assert bool(new_state.shop.angry) is True
    assert int(new_state.shop.bill) == 2 * DEFAULT_ITEM_PRICE
    assert int(new_state.player_gold) == 5
    # Ownership flags persist when the player walked out unpaid.
    assert bool(new_state.shop.items_owned_by_shop[5])
    assert bool(new_state.shop.items_owned_by_shop[6])


def test_pay_at_exit_with_zero_bill_noop():
    """Door cross with no bill leaves state untouched."""
    state = _env_with_shop(player_pos=(8, 5), player_gold=50)
    new_state = pay_at_exit(state)
    assert int(new_state.shop.bill) == 0
    assert int(new_state.player_gold) == 50
    assert bool(new_state.shop.angry) is False


def test_pay_at_exit_only_fires_on_door_tile():
    """Standing somewhere else inside the shop does not trigger payment.

    Wave 15: accrue against an empty slot so DEFAULT_ITEM_PRICE applies.
    """
    state = _env_with_shop(player_pos=(6, 6), player_gold=100)
    state = accrue_bill(state, slot_idx=5)
    # Player still in shop, NOT at door.
    new_state = pay_at_exit(state)
    assert int(new_state.shop.bill) == DEFAULT_ITEM_PRICE  # unchanged
    assert int(new_state.player_gold) == 100              # unchanged


# ---------------------------------------------------------------------------
# 4. Drop in shop
# ---------------------------------------------------------------------------
def test_drop_item_in_shop_clears_from_owned():
    """Dropping a shop-owned item back in the shop clears its flag + refunds."""
    state = _env_with_shop(player_pos=(6, 6))
    state = accrue_bill(state, slot_idx=4)
    assert int(state.shop.bill) == DEFAULT_ITEM_PRICE
    assert bool(state.shop.items_owned_by_shop[4]) is True

    new_state = drop_in_shop(state, slot_idx=4)
    assert int(new_state.shop.bill) == 0
    assert bool(new_state.shop.items_owned_by_shop[4]) is False


def test_drop_item_outside_shop_is_noop():
    """Dropping a shop-owned item OUTSIDE the shop is a no-op (still owned)."""
    state = _env_with_shop(player_pos=(6, 6))
    state = accrue_bill(state, slot_idx=4)
    # Walk out (no payment yet); now the player is outside the room.
    state = state.replace(player_pos=jnp.array([15, 15], dtype=jnp.int16))
    new_state = drop_in_shop(state, slot_idx=4)
    assert int(new_state.shop.bill) == DEFAULT_ITEM_PRICE
    assert bool(new_state.shop.items_owned_by_shop[4]) is True


# ---------------------------------------------------------------------------
# 5. Angry shopkeeper attack
# ---------------------------------------------------------------------------
def test_angry_shopkeeper_attacks_player():
    """When angry, the shopkeeper closes in and damages the player."""
    # Shopkeeper starts at (5, 5); player at (5, 6) — already adjacent.
    state = _env_with_shop(
        player_pos=(5, 6),
        shopkeeper_pos=(5, 5),
    )
    state = state.replace(shop=state.shop.replace(angry=jnp.bool_(True)))
    hp_before = int(state.player_hp)
    new_state = shopkeeper_attack(state, _RNG)
    assert int(new_state.player_hp) == max(0, hp_before - SHOPKEEPER_ANGRY_DAMAGE)


def test_non_angry_shopkeeper_does_not_attack():
    """A shopkeeper that hasn't been angered is inert."""
    state = _env_with_shop(player_pos=(5, 6), shopkeeper_pos=(5, 5))
    hp_before = int(state.player_hp)
    new_state = shopkeeper_attack(state, _RNG)
    assert int(new_state.player_hp) == hp_before


def test_angry_shopkeeper_pursues_when_distant():
    """An angry shopkeeper moves toward the player even from far away."""
    state = _env_with_shop(
        player_pos=(10, 10),
        shopkeeper_pos=(5, 5),
    )
    state = state.replace(shop=state.shop.replace(angry=jnp.bool_(True)))
    idx = int(state.shop.shopkeeper_idx)
    new_state = shopkeeper_attack(state, _RNG)
    new_pos = new_state.monster_ai.pos[idx]
    # Greedy 8-dir step toward (10, 10) from (5, 5) → (6, 6).
    assert int(new_pos[0]) == 6 and int(new_pos[1]) == 6
    # Still not adjacent → no damage yet.
    assert int(new_state.player_hp) == int(state.player_hp)


# ---------------------------------------------------------------------------
# 6. Killing the shopkeeper
# ---------------------------------------------------------------------------
def test_kill_shopkeeper_clears_bill_but_angers_neighbors():
    """Killing the shopkeeper wipes the bill but sets angry=True."""
    state = _env_with_shop(player_pos=(6, 6))
    state = accrue_bill(state, slot_idx=0)
    state = accrue_bill(state, slot_idx=1)
    assert int(state.shop.bill) > 0

    new_state = kill_shopkeeper(state)
    # Bill cleared, ownership cleared.
    assert int(new_state.shop.bill) == 0
    assert not bool(jnp.any(new_state.shop.items_owned_by_shop))
    # But "neighbours" (the global angry flag) are now hostile.
    assert bool(new_state.shop.angry) is True
    # Shopkeeper itself is dead.
    idx = int(new_state.shop.shopkeeper_idx)
    assert not bool(new_state.monster_ai.alive[idx])
    assert int(new_state.monster_ai.hp[idx]) == 0


# ---------------------------------------------------------------------------
# 7. Mine Town shop registration
# ---------------------------------------------------------------------------
def test_mine_town_has_shop_active():
    """make_mine_town_shop_state must produce a shop with shop_active=True."""
    shop = make_mine_town_shop_state()
    assert bool(shop.shop_active) is True
    # Bounding box must be a valid rectangle.
    assert int(shop.shop_room_min[0]) <= int(shop.shop_room_max[0])
    assert int(shop.shop_room_min[1]) <= int(shop.shop_room_max[1])
    # Shopkeeper is a real monster slot.
    assert int(shop.shopkeeper_idx) >= 0
    # Door pos is set.
    assert int(shop.door_pos[0]) >= 0
    assert int(shop.door_pos[1]) >= 0


# ---------------------------------------------------------------------------
# 8. Integration — shop_step ticks pay-at-exit + pursuit together
# ---------------------------------------------------------------------------
def test_shop_step_combines_pay_and_pursuit():
    """shop_step calls pay_at_exit then shopkeeper_attack atomically.

    Wave 15: accrue against empty slot 5 so DEFAULT_ITEM_PRICE applies
    (slot 0 carries the starting-inventory weapon whose real ``get_cost``
    differs by role + CHA tier).
    """
    # Player picks up an item then walks to the door with enough gold.
    state = _env_with_shop(player_pos=(6, 6), player_gold=50)
    state = accrue_bill(state, slot_idx=5)
    state = state.replace(player_pos=jnp.array([8, 5], dtype=jnp.int16))
    new_state = shop_step(state, _RNG)
    # Bill paid, no anger.
    assert int(new_state.shop.bill) == 0
    assert int(new_state.player_gold) == 50 - DEFAULT_ITEM_PRICE
    assert bool(new_state.shop.angry) is False
