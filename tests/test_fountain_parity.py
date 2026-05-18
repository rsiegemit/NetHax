"""Fountain subsystem parity tests.

Canonical source: vendor/nethack/src/fountain.c
  dipfountain()   fountain.c:394-554
  drinkfountain() fountain.c:243-390
  dryup()         fountain.c:200-238

Tests:
  test_dip_random_effect        — after N=100 dips, multiple outcomes observed
  test_drink_heals_sometimes    — after N=100 drinks, HP gain observed at least once
  test_dry_after_many_uses      — 50 dips → fountain becomes FLOOR
  test_excalibur_for_lawful_knight — Knight XL=5 + long sword → Excalibur in inventory
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.fountain import (
    dip_fountain,
    drink_fountain,
    dry_fountain,
    _EXCALIBUR_TYPE_ID,
    _LONG_SWORD_TYPE_ID,
    _DRY_THRESHOLD,
)
from Nethax.nethax.subsystems.inventory import ItemCategory, make_item
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.subsystems.prayer import Alignment

_RNG = jax.random.PRNGKey(42)
_STATIC = StaticParams()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> EnvState:
    return EnvState.default(rng=_RNG, static=_STATIC)


def _with_long_sword(state: EnvState, slot: int = 0) -> EnvState:
    """Place a long sword (type_id=37) in inventory slot."""
    inv = state.inventory.items
    new_inv = inv.replace(
        category=inv.category.at[slot].set(jnp.int8(int(ItemCategory.WEAPON))),
        type_id=inv.type_id.at[slot].set(jnp.int16(_LONG_SWORD_TYPE_ID)),
        quantity=inv.quantity.at[slot].set(jnp.int16(1)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_inv))


def _set_fountain_tile(state: EnvState) -> EnvState:
    """Place a fountain at player position on branch 0, level 0."""
    r = int(state.player_pos[0])
    c = int(state.player_pos[1])
    new_terrain = state.terrain.at[0, 0, r, c].set(jnp.int8(int(TileType.FOUNTAIN)))
    return state.replace(terrain=new_terrain)


# ---------------------------------------------------------------------------
# test_dip_random_effect
# ---------------------------------------------------------------------------

def test_dip_random_effect():
    """After N=100 dips with different RNG seeds, multiple outcomes observed.

    Vendor fountain.c:458 switch(rnd(30)) has 11 distinct outcome groups.
    We verify at least 3 distinct result states arise from 100 independent dips.

    Cite: vendor/nethack/src/fountain.c:458-553 dipfountain() switch block.
    """
    base = _fresh()
    base = _with_long_sword(base, slot=0)
    base = base.replace(
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
    )

    outcomes = set()
    for i in range(100):
        rng = jax.random.PRNGKey(i * 7 + 1)
        s2 = dip_fountain(base, rng, slot_idx=0)
        # Record observable: (hp, buc_status[0], type_id[0])
        key = (
            int(s2.player_hp),
            int(s2.inventory.items.buc_status[0]),
            int(s2.inventory.items.type_id[0]),
        )
        outcomes.add(key)

    assert len(outcomes) >= 3, (
        f"Expected >= 3 distinct dip outcomes across 100 trials, got {len(outcomes)}"
    )


# ---------------------------------------------------------------------------
# test_drink_heals_sometimes
# ---------------------------------------------------------------------------

def test_drink_heals_sometimes():
    """After N=100 drinks, HP gain observed at least once.

    Vendor fountain.c:279-283: fate < 10 → cool draught refreshes you.
    Our REFRESH outcome adds 5 HP.

    Cite: vendor/nethack/src/fountain.c:279 drinkfountain() fate<10 branch.
    """
    base = _fresh()
    base = base.replace(
        player_hp=jnp.int32(5),
        player_hp_max=jnp.int32(50),
    )

    healed = False
    for i in range(100):
        rng = jax.random.PRNGKey(i * 13 + 3)
        s2 = drink_fountain(base, rng)
        if int(s2.player_hp) > 5:
            healed = True
            break

    assert healed, "Expected at least one HP-healing drink outcome in 100 trials"


# ---------------------------------------------------------------------------
# test_dry_after_many_uses
# ---------------------------------------------------------------------------

def test_dry_after_many_uses():
    """50 dips → fountain_uses reaches threshold → tile becomes FLOOR.

    Vendor fountain.c:200-238 dryup(): called after every dipfountain/drinkfountain.
    Our simplified model dries the fountain after _DRY_THRESHOLD uses.

    Cite: vendor/nethack/src/fountain.c:200 dryup().
    """
    state = _fresh()
    state = _with_long_sword(state, slot=0)
    state = _set_fountain_tile(state)
    state = state.replace(
        player_hp=jnp.int32(50),
        player_hp_max=jnp.int32(50),
    )

    for i in range(50):
        # Use a fixed RNG seed that produces the NOTHING/noop outcome to avoid
        # side effects from demon/nymph, and keep sword in slot.
        rng = jax.random.PRNGKey(i * 100 + 99)
        state = dip_fountain(state, rng, slot_idx=0)
        # Re-insert sword if nymph stole it (keep test focused on dry mechanic).
        state = _with_long_sword(state, slot=0)

    r = int(state.player_pos[0])
    c = int(state.player_pos[1])
    tile = int(state.terrain[0, 0, r, c])
    assert tile == int(TileType.FLOOR), (
        f"Expected FLOOR ({int(TileType.FLOOR)}) after {_DRY_THRESHOLD}+ uses, "
        f"got tile={tile}"
    )


# ---------------------------------------------------------------------------
# test_excalibur_for_lawful_knight
# ---------------------------------------------------------------------------

def test_excalibur_for_lawful_knight():
    """Knight XL=5, Lawful, wielding long sword → Excalibur granted.

    Vendor fountain.c:404-447 dipfountain(): if long sword + Knight + Lawful
    + XL>=5, the Lady of the Lake grants Excalibur and the fountain dries.

    Cite: vendor/nethack/src/fountain.c:404-447 dipfountain() Excalibur path.
    """
    state = _fresh()
    state = _with_long_sword(state, slot=0)
    state = _set_fountain_tile(state)
    state = state.replace(
        player_xl=jnp.int32(5),
        player_role=jnp.int8(int(Role.KNIGHT)),
        player_align=jnp.int8(int(Alignment.LAWFUL)),
    )

    rng = jax.random.PRNGKey(0)
    result = dip_fountain(state, rng, slot_idx=0)

    # Sword in slot 0 should now be Excalibur.
    new_type_id = int(result.inventory.items.type_id[0])
    assert new_type_id == _EXCALIBUR_TYPE_ID, (
        f"Expected Excalibur type_id={_EXCALIBUR_TYPE_ID}, got {new_type_id}"
    )

    # Fountain should have dried (tile = FLOOR).
    r = int(state.player_pos[0])
    c = int(state.player_pos[1])
    tile = int(result.terrain[0, 0, r, c])
    assert tile == int(TileType.FLOOR), (
        f"Expected fountain to dry (FLOOR={int(TileType.FLOOR)}), got {tile}"
    )
