"""Parity tests for AMAX restore, cursed-book explode, and lock-pick chest.

Covers:
  1. test_amax_restored         — restore ability raises drained stat to player_amax
  2. test_amax_doesnt_restore_above — stat already at/above amax is unchanged
  3. test_cursed_book_explode_destroys_book_and_damages_hp
        — vendor branch 6 (rn2(lev)==6) destroys the book + 2*rnd(10)+5 hp loss
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
# 3. test_cursed_book_explode_destroys_book_and_damages_hp
#    Vendor branch 6 (rn2(lev)==6) destroys the book and deals 2*rnd(10)+5
#    damage; Antimagic (unmodelled) would gate the damage to 0.
#
# REBALANCE: this slot previously tested ``wielded == -1`` after an "explode
# branch" hit.  That behaviour was invented — vendor cursed_book does not
# drop the wielded weapon (vendor/nethack/src/spell.c::cursed_book line 169
# only calls ``losehp()`` and returns TRUE so the caller useup()s the book;
# there is no setuwep(NULL) call).  Test rewritten to assert vendor truth:
# the book is destroyed and hp drops by 7..25.
#
# Cite: vendor/nethack/src/spell.c::cursed_book lines 169-179.
# ---------------------------------------------------------------------------

def test_cursed_book_explode_destroys_book_and_damages_hp():
    from Nethax.nethax.subsystems.magic import SpellId, _SPELL_LEVELS
    from Nethax.nethax.subsystems.inventory import make_item
    from Nethax.nethax.rng import rn2 as nethax_rn2

    # CANCELLATION = level 7 so rn2(lev)==6 (vendor explode branch) is hit.
    spell_id = int(SpellId.CANCELLATION)
    assert int(_SPELL_LEVELS[spell_id]) == 7

    initial_hp = 60
    s = _base_state()
    s = s.replace(player_hp=jnp.int32(initial_hp),
                  player_hp_max=jnp.int32(initial_hp))

    # Patch slot 0 to be the cursed level-7 spellbook.
    new_items = s.inventory.items
    new_cat = new_items.category.at[0].set(jnp.int8(int(ItemCategory.SPBOOK)))
    new_tid = new_items.type_id.at[0].set(jnp.int16(spell_id))
    new_buc = new_items.buc_status.at[0].set(jnp.int8(_BUC_CURSED))
    new_qty = new_items.quantity.at[0].set(jnp.int16(1))
    patched_items = new_items.replace(
        category=new_cat, type_id=new_tid, buc_status=new_buc, quantity=new_qty
    )
    s = s.replace(inventory=s.inventory.replace(items=patched_items))

    # Seed-scan for the explode branch (rn2(7) == 6).  Mirrors the RNG
    # split inside :func:`_cursed_book_backfire` (9-way split → branch
    # selector is the first sub-key).
    explode_state = None
    for seed in range(1000):
        rng = jax.random.PRNGKey(seed)
        subs = jax.random.split(rng, 9)
        if int(nethax_rn2(subs[0], 7)) == 6:
            explode_state = read_spellbook(s, rng, slot_idx=0)
            break

    assert explode_state is not None, (
        "No seed found where rn2(7)==6 (vendor explode branch) in 1000 tries"
    )
    # Vendor: book is useup'd (quantity → 0).
    qty_after = int(explode_state.inventory.items.quantity[0])
    assert qty_after == 0, (
        f"Vendor explode destroys the book; expected quantity 0, got {qty_after}"
    )
    # Vendor damage range: dmg = 2*rnd(10)+5 → [7, 25].
    hp_loss = initial_hp - int(explode_state.player_hp)
    assert 7 <= hp_loss <= 25, (
        f"Vendor explode hp loss {hp_loss} should be in [7, 25] (2*rnd(10)+5)"
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
