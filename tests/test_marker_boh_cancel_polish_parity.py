"""Parity tests: magic-marker user_name scroll selection, BoH cancellation,
and martial-arts skill threshold confirmation.

Canonical sources:
  vendor/nethack/src/apply.c::write_with_marker (~line 4320)
  vendor/nethack/src/zap.c::cancel_item (line 720)
  vendor/nethack/include/skills.h — practice_needed_to_advance macro
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state():
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, _ = env.reset(_RNG)
    return state


def _floor_state():
    from Nethax.nethax.state import EnvState, StaticParams
    from Nethax.nethax.constants.tiles import TileType
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), int(TileType.FLOOR), dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
    )


def _set_user_name(state, slot: int, name: str):
    """Set inventory.user_names[slot] to the given string."""
    from Nethax.nethax.subsystems.inventory import USER_NAME_LEN
    b = name.encode("ascii")[:USER_NAME_LEN]
    b = b + b"\x00" * (USER_NAME_LEN - len(b))
    name_row = jnp.array(list(b), dtype=jnp.int8)
    new_user_names = state.inventory.user_names.at[slot].set(name_row)
    return state.replace(inventory=state.inventory.replace(user_names=new_user_names))


def _wield_marker(state, slot: int = 0):
    """Place a magic marker in slot and wield it."""
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.apply_tools import _MAGIC_MARKER_TYPE_ID
    inv = state.inventory
    new_cat = inv.items.category.at[slot].set(jnp.int8(int(ItemCategory.TOOL)))
    new_tid = inv.items.type_id.at[slot].set(jnp.int16(_MAGIC_MARKER_TYPE_ID))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid)
    return state.replace(inventory=inv.replace(items=new_items, wielded=jnp.int8(slot)))


def _add_blank_scroll(state, slot: int = 1):
    """Place a blank scroll in the given inventory slot."""
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.apply_tools import _SCR_BLANK_PAPER_ID
    inv = state.inventory
    new_cat = inv.items.category.at[slot].set(jnp.int8(int(ItemCategory.SCROLL)))
    new_tid = inv.items.type_id.at[slot].set(jnp.int16(_SCR_BLANK_PAPER_ID))
    new_items = inv.items.replace(category=new_cat, type_id=new_tid)
    return state.replace(inventory=inv.replace(items=new_items))


# ---------------------------------------------------------------------------
# 1. Magic marker: user_name "scroll of light" → SCR_LIGHT
# ---------------------------------------------------------------------------

def test_marker_writes_named_scroll():
    """Magic marker with user_name='scroll of light' writes SCR_LIGHT.

    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320) —
    vendor presents a menu; headless mode parses user_name field.
    """
    from Nethax.nethax.subsystems.apply_tools import dispatch_apply, _SCROLL_BASE_ID
    from Nethax.nethax.subsystems.items_scrolls import ScrollEffect

    state = _floor_state()
    state = _wield_marker(state, slot=0)
    state = _set_user_name(state, slot=0, name="scroll of light")
    state = _add_blank_scroll(state, slot=1)

    result = dispatch_apply(state, _RNG)

    expected_tid = _SCROLL_BASE_ID + int(ScrollEffect.LIGHT)
    actual_tid = int(result.inventory.items.type_id[1])
    assert actual_tid == expected_tid, (
        f"Expected SCR_LIGHT (type_id={expected_tid}), got {actual_tid}"
    )


# ---------------------------------------------------------------------------
# 2. Magic marker: empty user_name → default SCR_MAGIC_MAPPING
# ---------------------------------------------------------------------------

def test_marker_default_when_empty():
    """Magic marker with empty user_name defaults to SCR_MAGIC_MAPPING.

    Cite: vendor/nethack/src/apply.c::write_with_marker (~line 4320) —
    headless default when no name is provided.
    """
    from Nethax.nethax.subsystems.apply_tools import (
        dispatch_apply, _SCROLL_BASE_ID, _SCR_MAGIC_MAPPING_OFFSET,
    )

    state = _floor_state()
    state = _wield_marker(state, slot=0)
    # user_name stays all-zero (empty) — default
    state = _add_blank_scroll(state, slot=1)

    result = dispatch_apply(state, _RNG)

    expected_tid = _SCROLL_BASE_ID + _SCR_MAGIC_MAPPING_OFFSET
    actual_tid = int(result.inventory.items.type_id[1])
    assert actual_tid == expected_tid, (
        f"Expected SCR_MAGIC_MAPPING (type_id={expected_tid}), got {actual_tid}"
    )


# ---------------------------------------------------------------------------
# 3. cancel_bag_of_holding: empties contents and demotes type to SACK
# ---------------------------------------------------------------------------

def test_cancel_bag_of_holding_empties_and_demotes():
    """cancel_bag_of_holding zeros all items_quantity and sets type to SACK.

    Cite: vendor/nethack/src/zap.c::cancel_item (line 720) — BoH implodes,
    destroying everything inside; bag becomes a plain sack.
    """
    from Nethax.nethax.subsystems.containers import (
        cancel_bag_of_holding, install_container, ContainerType, BUCStatus,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _base_state()

    # Install a BoH in container slot 0.
    state = install_container(state, 0, ContainerType.BAG_OF_HOLDING,
                               parent_slot=0, buc=int(BUCStatus.UNCURSED))

    # Populate 5 items into the container.
    cs = state.containers
    for i in range(5):
        cs = cs.replace(
            items_category=cs.items_category.at[0, i].set(jnp.int8(int(ItemCategory.FOOD))),
            items_type_id=cs.items_type_id.at[0, i].set(jnp.int16(1)),
            items_quantity=cs.items_quantity.at[0, i].set(jnp.int16(3)),
        )
    state = state.replace(containers=cs)

    # Verify items are present before cancel.
    assert int(jnp.sum(state.containers.items_quantity[0])) == 15

    result = cancel_bag_of_holding(state, 0)

    # All quantities zeroed.
    total_qty = int(jnp.sum(result.containers.items_quantity[0]))
    assert total_qty == 0, f"Expected 0 total quantity after cancel, got {total_qty}"

    # Container demoted from BAG_OF_HOLDING to SACK.
    new_ctype = int(result.containers.container_type[0])
    assert new_ctype == int(ContainerType.SACK), (
        f"Expected SACK ({int(ContainerType.SACK)}), got {new_ctype}"
    )


# ---------------------------------------------------------------------------
# 4. Martial arts skill thresholds match default formula
# ---------------------------------------------------------------------------

def test_martial_arts_thresholds_match_default():
    """Vendor Skill_M uses the same practice_needed_to_advance formula as all skills.

    Cite: vendor/nethack/include/skills.h — practice_needed_to_advance(level)
    = level * level * 20, uniform across all weapon skills including martial arts.
    Confirmed via vendor/nethack/src/u_init.c::Skill_Mon (P_MARTIAL_ARTS entry)
    which sets only the cap (P_GRAND_MASTER for Monk), not custom thresholds.
    """
    from Nethax.nethax.subsystems.skills import practice_needed_to_advance, SkillLevel

    expected = {
        SkillLevel.P_UNSKILLED:    0,    # 0*0*20
        SkillLevel.P_BASIC:       20,    # 1*1*20
        SkillLevel.P_SKILLED:     80,    # 2*2*20
        SkillLevel.P_EXPERT:     180,    # 3*3*20
        SkillLevel.P_MASTER:     320,    # 4*4*20
    }

    for level, exp_val in expected.items():
        got = int(practice_needed_to_advance(jnp.int32(int(level))))
        assert got == exp_val, (
            f"practice_needed_to_advance({level.name}={int(level)}) = {got}, "
            f"expected {exp_val}"
        )
