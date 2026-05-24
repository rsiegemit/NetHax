"""Wave-45d — shopkeeper class variant smoke tests.

Citations:
    vendor/nethack/src/shk.c::sk_close, oid_price_adjustment
    vendor/nethack/src/vault.c::vault_gd_watching_player
    vendor/nethack/src/monst.c PM_CROESUS definition
    vendor/nethack/src/priest.c (aligned priest shopkeeper)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax import NethaxEnv
from Nethax.nethax.subsystems.shop import (
    ShopState,
    ShopkeeperKind,
    croesus_clamp,
    vault_keeper_collect,
    aligned_priest_price,
    VAULT_KEEPER_DEMAND_DEFAULT,
    ALIGNED_PRIEST_MISALIGN_MUL,
)
from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS


def _make_variant_shop(kind: int, door=(8, 5)):
    """Build an active ShopState configured for the given variant."""
    return ShopState(
        shop_active=jnp.bool_(True),
        shopkeeper_idx=jnp.int8(0),
        shop_room_min=jnp.array([5, 5], dtype=jnp.int8),
        shop_room_max=jnp.array([7, 10], dtype=jnp.int8),
        door_pos=jnp.array(list(door), dtype=jnp.int8),
        bill=jnp.int32(0),
        items_owned_by_shop=jnp.zeros((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
        angry=jnp.bool_(False),
        kind=jnp.int8(kind),
    )


def _env_with(shop, player_pos=(6, 6), player_gold=200):
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    return state.replace(
        shop=shop,
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_gold=jnp.int32(player_gold),
    )


# ---------------------------------------------------------------------------
# 1. Default kind
# ---------------------------------------------------------------------------
def test_default_shop_kind_is_generic():
    """ShopState.default() picks the generic shopkeeper variant."""
    shop = ShopState.default()
    assert int(shop.kind) == ShopkeeperKind.GENERIC


# ---------------------------------------------------------------------------
# 2. Croesus — bill clamps to 0, angry stays sticky-False
# ---------------------------------------------------------------------------
def test_croesus_clamp_zeros_bill_and_keeps_angry_false():
    """Croesus variant: any prior bill is wiped, angry forced False."""
    shop = _make_variant_shop(ShopkeeperKind.CROESUS).replace(
        bill=jnp.int32(999),
        angry=jnp.bool_(True),
        items_owned_by_shop=jnp.ones((MAX_INVENTORY_SLOTS,), dtype=jnp.bool_),
    )
    state = _env_with(shop)
    new_state = croesus_clamp(state)
    assert int(new_state.shop.bill) == 0
    assert bool(new_state.shop.angry) is False
    assert not bool(jnp.any(new_state.shop.items_owned_by_shop))


def test_croesus_clamp_leaves_generic_untouched():
    """Generic shop bill must NOT be zeroed by croesus_clamp."""
    shop = _make_variant_shop(ShopkeeperKind.GENERIC).replace(bill=jnp.int32(42))
    state = _env_with(shop)
    new_state = croesus_clamp(state)
    assert int(new_state.shop.bill) == 42


# ---------------------------------------------------------------------------
# 3. Vault keeper — gold demand handshake
# ---------------------------------------------------------------------------
def test_vault_keeper_deducts_gold_when_player_can_pay():
    """Vault keeper deducts the demand from player_gold when affordable."""
    shop = _make_variant_shop(ShopkeeperKind.VAULT_KEEPER, door=(9, 9))
    state = _env_with(shop, player_gold=2000)
    new_state = vault_keeper_collect(state, demand=500)
    assert int(new_state.player_gold) == 1500
    # Did not get teleported to the door — payment path.
    assert int(new_state.player_pos[0]) == 6
    assert int(new_state.player_pos[1]) == 6


def test_vault_keeper_leads_to_door_when_insufficient_gold():
    """Insufficient gold → player teleported to ``door_pos``."""
    shop = _make_variant_shop(ShopkeeperKind.VAULT_KEEPER, door=(9, 9))
    state = _env_with(shop, player_gold=10)
    new_state = vault_keeper_collect(state, demand=500)
    # Gold untouched on refusal path.
    assert int(new_state.player_gold) == 10
    # Player escorted to the door tile.
    assert int(new_state.player_pos[0]) == 9
    assert int(new_state.player_pos[1]) == 9


def test_vault_keeper_noop_for_non_vault_variant():
    """Non-vault shops are untouched by vault_keeper_collect."""
    shop = _make_variant_shop(ShopkeeperKind.GENERIC)
    state = _env_with(shop, player_gold=10)
    new_state = vault_keeper_collect(state, demand=500)
    assert int(new_state.player_gold) == 10
    assert int(new_state.player_pos[0]) == 6
    assert int(new_state.player_pos[1]) == 6


# ---------------------------------------------------------------------------
# 4. Aligned priest — 2x misalignment surcharge
# ---------------------------------------------------------------------------
def test_aligned_priest_doubles_misaligned_price():
    """Misaligned customer pays 2x base price."""
    out = aligned_priest_price(
        base_price=jnp.int32(100),
        shop_kind=jnp.int8(ShopkeeperKind.ALIGNED_PRIEST),
        player_alignment=jnp.int32(1),   # lawful
        shop_alignment=jnp.int32(-1),    # chaotic temple
    )
    assert int(out) == 100 * ALIGNED_PRIEST_MISALIGN_MUL


def test_aligned_priest_aligned_customer_pays_base():
    """Aligned customer pays exactly base price."""
    out = aligned_priest_price(
        base_price=jnp.int32(100),
        shop_kind=jnp.int8(ShopkeeperKind.ALIGNED_PRIEST),
        player_alignment=jnp.int32(1),
        shop_alignment=jnp.int32(1),
    )
    assert int(out) == 100


def test_aligned_priest_surcharge_skipped_for_generic_shop():
    """Generic shopkeepers never trigger the alignment surcharge."""
    out = aligned_priest_price(
        base_price=jnp.int32(100),
        shop_kind=jnp.int8(ShopkeeperKind.GENERIC),
        player_alignment=jnp.int32(1),
        shop_alignment=jnp.int32(-1),
    )
    assert int(out) == 100


# ---------------------------------------------------------------------------
# 5. Vault keeper demand-default constant sanity
# ---------------------------------------------------------------------------
def test_vault_keeper_demand_default_in_vendor_range():
    """vendor rn1(1000, 50) → demand ∈ [50, 1049]; midpoint must fit."""
    assert 50 <= VAULT_KEEPER_DEMAND_DEFAULT <= 1049
