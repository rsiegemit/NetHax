"""Parity tests for the four previously-stubbed scroll effects + mail.

Covers:
  1. SCR_SCARE_MONSTER  — flee_until_turn set on normal monsters.
  2. SCR_SCARE_MONSTER  — immune-class monsters unchanged.
  3. SCR_CONFUSE_MONSTER — confuse_attack_pending set (sane branch).
  4. SCR_CREATE_MONSTER  — alive monster count +1 (uncursed).
  5. SCR_CREATE_MONSTER  — confused → multiple spawns.
  6. SCR_TAMING          — only the in-radius monster becomes tame.
  7. SCR_MAIL            — a SCR_MAIL scroll appears in inventory.

Vendor citations:
  read.c::seffect_scare_monster ~1454, seffect_confuse_monster ~1399,
  seffect_create_monster ~1608, seffect_taming ~1679,
  seffect_mail ~2157, mail.c::ckmailstatus.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.inventory import (
    InventoryState,
    ItemCategory,
    make_item,
)
from Nethax.nethax.subsystems.items_scrolls import (
    ScrollEffect,
    _SCROLL_BASE_ID,
    _IS_SCARE_IMMUNE,
    read_scroll,
)
from Nethax.nethax.subsystems.monster_ai import MonsterAIState
from Nethax.nethax.subsystems.status_effects import TimedStatus

_RNG = jax.random.PRNGKey(0)
_BUC_UNCURSED = 2
_BUC_CURSED   = 1
_BUC_BLESSED  = 3


def _scroll(effect: ScrollEffect, buc: int = _BUC_UNCURSED):
    return make_item(
        category=int(ItemCategory.SCROLL),
        type_id=_SCROLL_BASE_ID + int(effect),
        quantity=1,
        buc_status=buc,
    )


def _base_state(items=None):
    state = EnvState.default(_RNG)
    if items:
        state = state.replace(inventory=InventoryState.from_items(items))
    return state


def _place_monster(state, slot, row, col, entry_idx=1, alive=True):
    """Place a monster in the given slot."""
    mai = state.monster_ai
    new_alive = mai.alive.at[slot].set(jnp.bool_(alive))
    new_pos   = mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_entry = mai.entry_idx.at[slot].set(jnp.int16(entry_idx))
    new_hp    = mai.hp.at[slot].set(jnp.int32(10))
    new_hp_max = mai.hp_max.at[slot].set(jnp.int32(10))
    new_mai   = mai.replace(alive=new_alive, pos=new_pos, entry_idx=new_entry,
                            hp=new_hp, hp_max=new_hp_max)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# 1. scare_monster — sets flee_until_turn on normal monsters
# vendor read.c::seffect_scare_monster ~1454
# ---------------------------------------------------------------------------

def test_scare_monster_sets_flee_timer():
    """Normal (non-immune) alive monster gets flee_until_turn > timestep."""
    # entry_idx=1 (giant ant) — not scare-immune.
    state = _base_state([_scroll(ScrollEffect.SCARE_MONSTER)])
    state = _place_monster(state, slot=1, row=5, col=5, entry_idx=1)

    result = read_scroll(state, _RNG, 0)

    flee = int(result.monster_ai.flee_until_turn[1])
    ts   = int(result.timestep)
    assert flee > ts, (
        f"flee_until_turn ({flee}) should exceed timestep ({ts}) after scare"
    )


# ---------------------------------------------------------------------------
# 2. scare_monster — immune-class monsters unchanged
# ---------------------------------------------------------------------------

def test_scare_monster_immune_class_unchanged():
    """Scare-immune monster (demon/angel class) keeps flee_until_turn == 0."""
    from Nethax.nethax.constants.monsters import MONSTERS, M2_DEMON, MonsterSymbol
    # Find a demon entry_idx.
    demon_idx = next(
        i for i, m in enumerate(MONSTERS)
        if bool(m.flags2 & M2_DEMON)
    )
    state = _base_state([_scroll(ScrollEffect.SCARE_MONSTER)])
    state = _place_monster(state, slot=1, row=5, col=5, entry_idx=demon_idx)

    result = read_scroll(state, _RNG, 0)

    flee = int(result.monster_ai.flee_until_turn[1])
    assert flee == 0, (
        f"Immune monster flee_until_turn should stay 0, got {flee}"
    )


# ---------------------------------------------------------------------------
# 3. confuse_monster — sets confuse_attack_pending (sane branch)
# vendor read.c::seffect_confuse_monster ~1399, u.umconf += incr line 1449
# ---------------------------------------------------------------------------

def test_confuse_monster_sets_pending():
    """Sane (non-confused) confuse-monster sets confuse_attack_pending=True."""
    state = _base_state([_scroll(ScrollEffect.CONFUSE_MONSTER)])

    result = read_scroll(state, _RNG, 0)

    assert bool(result.status.confuse_attack_pending), (
        "confuse_attack_pending must be True after sane SCR_CONFUSE_MONSTER"
    )


# ---------------------------------------------------------------------------
# 4. create_monster — spawns exactly 1 new alive monster (uncursed)
# vendor read.c::seffect_create_monster ~1615
# ---------------------------------------------------------------------------

def test_create_monster_spawns_1():
    """Uncursed create_monster increases alive monster count by exactly 1."""
    state = _base_state([_scroll(ScrollEffect.CREATE_MONSTER)])
    before = int(jnp.sum(state.monster_ai.alive))

    result = read_scroll(state, _RNG, 0)
    after  = int(jnp.sum(result.monster_ai.alive))

    assert after == before + 1, (
        f"Expected {before + 1} alive monsters, got {after}"
    )


# ---------------------------------------------------------------------------
# 5. create_monster — confused spawns more (13)
# vendor read.c line 1615: 1 + ((confused || scursed) ? 12 : 0)
# ---------------------------------------------------------------------------

def test_create_monster_confused_spawns_more():
    """Confused create_monster spawns 13 monsters (1 + 12)."""
    state = _base_state([_scroll(ScrollEffect.CREATE_MONSTER)])
    # Set player confused.
    ts = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(jnp.int32(10))
    state = state.replace(status=state.status.replace(timed_statuses=ts))
    before = int(jnp.sum(state.monster_ai.alive))

    result = read_scroll(state, _RNG, 0)
    after  = int(jnp.sum(result.monster_ai.alive))

    assert after == before + 13, (
        f"Confused: expected {before + 13} alive, got {after}"
    )


# ---------------------------------------------------------------------------
# 6. taming — only the in-radius monster becomes tame
# vendor read.c::seffect_taming ~1689: bd=1 for uncursed
# ---------------------------------------------------------------------------

def test_taming_within_radius():
    """Only the monster within Chebyshev-1 of player becomes tame."""
    state = _base_state([_scroll(ScrollEffect.TAMING)])
    state = state.replace(player_pos=jnp.array([10, 10], dtype=jnp.int16))

    # Slot 1: adjacent (row=10, col=11) — within bd=1.
    # Slot 2: far away (row=10, col=20) — outside bd=1.
    # entry_idx=1 (giant ant) — not tame-immune.
    state = _place_monster(state, slot=1, row=10, col=11, entry_idx=1)
    state = _place_monster(state, slot=2, row=10, col=20, entry_idx=1)

    result = read_scroll(state, _RNG, 0)

    assert bool(result.monster_ai.tame[1]), "Monster within radius should be tamed"
    assert not bool(result.monster_ai.tame[2]), "Monster outside radius must NOT be tamed"


# ---------------------------------------------------------------------------
# 7. mail — creates a SCR_MAIL item in inventory
# vendor read.c::seffect_mail ~2157, mail.c::ckmailstatus
# ---------------------------------------------------------------------------

def test_mail_creates_scroll_item():
    """Reading SCR_MAIL places a SCR_MAIL item in the first empty inv slot."""
    state = _base_state([_scroll(ScrollEffect.MAIL)])

    result = read_scroll(state, _RNG, 0)

    expected_tid = _SCROLL_BASE_ID + int(ScrollEffect.MAIL)
    inv_cat = result.inventory.items.category
    inv_tid = result.inventory.items.type_id

    mail_slots = (inv_cat == int(ItemCategory.SCROLL)) & (inv_tid == expected_tid)
    assert bool(jnp.any(mail_slots)), (
        f"Expected a SCR_MAIL item (tid={expected_tid}) in inventory after reading scroll of mail"
    )
