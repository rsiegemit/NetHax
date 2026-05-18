"""Parity tests for AMAX restore, cursed-book wielded-drop, and lock-pick chest.

Covers:
  1. test_amax_restored         — restore ability raises drained stat to player_amax
  2. test_amax_doesnt_restore_above — stat already at/above amax is unchanged
  3. test_cursed_book_drops_wielded — explode branch (b=1..4) sets wielded=-1
  4. test_lock_pick_opens_chest     — success roll unlocks is_locked container slot
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.inventory import ItemCategory, make_item, MAX_INVENTORY_SLOTS
from Nethax.nethax.subsystems.items_potions import _effect_restore_ability, _BUC_UNCURSED
from Nethax.nethax.subsystems.items_spellbooks import read_spellbook, _BUC_CURSED
from Nethax.nethax.subsystems.apply_tools import dispatch_apply, _LOCK_PICK_TYPE_ID

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state() -> EnvState:
    static = StaticParams()
    return EnvState.default(rng=_RNG, static=static)


def _state_with_amax(amax_list):
    """Return a state with player_amax set to amax_list (6 ints: str,int,wis,dex,con,cha)."""
    s = _base_state()
    return s.replace(player_amax=jnp.array(amax_list, dtype=jnp.int8))


# ---------------------------------------------------------------------------
# 1. test_amax_restored
#    player_str drained to 8; player_amax[0]=18; after restore → str=18
# ---------------------------------------------------------------------------

def test_amax_restored():
    s = _state_with_amax([18, 18, 18, 18, 18, 18])
    s = s.replace(player_str=jnp.int16(8))
    s2 = _effect_restore_ability(s, _RNG, _BUC_UNCURSED)
    assert int(s2.player_str) == 18, f"Expected str=18 after restore, got {int(s2.player_str)}"


# ---------------------------------------------------------------------------
# 2. test_amax_doesnt_restore_above
#    player_str=18 but amax[0]=15; after restore str should stay 18 (not drop)
#    Vendor full_restore: only restores UP to amax, never reduces current value.
# ---------------------------------------------------------------------------

def test_amax_doesnt_restore_above():
    # amax[0]=15 (str), current str=18 (above amax) — should not change
    s = _state_with_amax([15, 18, 18, 18, 18, 18])
    s = s.replace(player_str=jnp.int16(18))
    s2 = _effect_restore_ability(s, _RNG, _BUC_UNCURSED)
    assert int(s2.player_str) == 18, (
        f"Expected str=18 unchanged (already above amax=15), got {int(s2.player_str)}"
    )


# ---------------------------------------------------------------------------
# 3. test_cursed_book_drops_wielded
#    Force explode branch (b in 1..4) via a seeded RNG that yields b=1.
#    After reading: inventory.wielded should be -1.
# ---------------------------------------------------------------------------

def test_cursed_book_drops_wielded():
    from Nethax.nethax.subsystems.magic import SpellId, N_SPELLS
    from Nethax.nethax.subsystems.inventory import make_item

    s = _base_state()

    # Place a cursed spellbook (SpellId.FORCE_BOLT = 10) in slot 0.
    new_items = s.inventory.items
    book = make_item(
        category=ItemCategory.SPBOOK,
        type_id=int(SpellId.FORCE_BOLT),
        quantity=1,
        weight=50,
        buc_status=_BUC_CURSED,
    )
    # Patch each field of the items struct at slot 0.
    new_cat = new_items.category.at[0].set(jnp.int8(int(ItemCategory.SPBOOK)))
    new_tid = new_items.type_id.at[0].set(jnp.int16(int(SpellId.FORCE_BOLT)))
    new_buc = new_items.buc_status.at[0].set(jnp.int8(_BUC_CURSED))
    new_qty = new_items.quantity.at[0].set(jnp.int16(1))
    patched_items = new_items.replace(
        category=new_cat, type_id=new_tid, buc_status=new_buc, quantity=new_qty
    )
    # Wield slot 1 (a placeholder weapon so wielded >= 0).
    wep = make_item(category=ItemCategory.WEAPON, type_id=17, quantity=1, weight=10)
    new_cat2 = patched_items.category.at[1].set(jnp.int8(int(ItemCategory.WEAPON)))
    new_tid2 = patched_items.type_id.at[1].set(jnp.int16(17))
    new_qty2 = patched_items.quantity.at[1].set(jnp.int16(1))
    patched_items2 = patched_items.replace(category=new_cat2, type_id=new_tid2, quantity=new_qty2)
    inv = s.inventory.replace(items=patched_items2, wielded=jnp.int8(1))
    s = s.replace(inventory=inv)

    # Seed-scan: find an RNG that triggers the explode branch (b in [1,4]).
    # The first rnd(20) call in cursed_book uses sub_b from jax.random.split(rng, 7)[1].
    # We iterate seeds until we land in branch 0 (b in 1..4).
    explode_state = None
    for seed in range(200):
        rng = jax.random.PRNGKey(seed)
        s2 = read_spellbook(s, rng, slot_idx=0)
        if int(s2.inventory.wielded) == -1:
            explode_state = s2
            break

    assert explode_state is not None, (
        "Could not find a seed that triggers the explode branch in 200 tries"
    )
    assert int(explode_state.inventory.wielded) == -1, (
        f"Expected wielded=-1 after cursed book explode, got {int(explode_state.inventory.wielded)}"
    )


# ---------------------------------------------------------------------------
# 4. test_lock_pick_opens_chest
#    Place a locked container in slot 0; wield a lock-pick; high dex → success.
#    After dispatch_apply: containers.is_locked[0] == False.
# ---------------------------------------------------------------------------

def test_lock_pick_opens_chest():
    s = _base_state()

    # Set player dex to 20 → ch = 3*20 = 60, very high success rate.
    s = s.replace(player_dex=jnp.int8(20))

    # Place a lock-pick in inventory slot 0 and wield it.
    new_cat = s.inventory.items.category.at[0].set(jnp.int8(int(ItemCategory.TOOL)))
    new_tid = s.inventory.items.type_id.at[0].set(jnp.int16(_LOCK_PICK_TYPE_ID))
    new_qty = s.inventory.items.quantity.at[0].set(jnp.int16(1))
    patched_items = s.inventory.items.replace(category=new_cat, type_id=new_tid, quantity=new_qty)
    inv = s.inventory.replace(items=patched_items, wielded=jnp.int8(0))
    s = s.replace(inventory=inv)

    # Mark container slot 0 as locked.
    new_is_locked = s.containers.is_locked.at[0].set(jnp.bool_(True))
    s = s.replace(containers=s.containers.replace(is_locked=new_is_locked))

    # Find an RNG seed that produces success (roll < 60).
    opened_state = None
    for seed in range(50):
        rng = jax.random.PRNGKey(seed)
        s2 = dispatch_apply(s, rng)
        if not bool(s2.containers.is_locked[0]):
            opened_state = s2
            break

    assert opened_state is not None, (
        "Lock pick failed to open chest in 50 tries (dex=20, ch=60)"
    )
    assert not bool(opened_state.containers.is_locked[0]), (
        "Expected containers.is_locked[0]=False after successful lock pick"
    )
