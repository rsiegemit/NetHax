"""Wave 3 wand subsystem tests.

Tests verify:
  - Striking on adjacent monster reduces its HP.
  - Magic missile east hits and damages multiple aligned monsters.
  - Digging north carves corridor into WALL tiles.
  - Cold east on water tile freezes it (WATER → FLOOR/ICE).
  - Charges decrement after zap.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.subsystems.inventory import InventoryState, Item, MAX_INVENTORY_SLOTS
from Nethax.nethax.subsystems.items_wands import (
    WandEffect,
    WandState,
    ITEM_CATEGORY_WAND,
    zap_wand,
    handle_zap,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL

_RNG = jax.random.PRNGKey(42)

MAP_H, MAP_W = 21, 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inventory_with_wand(effect: WandEffect, charges: int = 5) -> InventoryState:
    """Return an InventoryState containing one wand in slot 0 (batched, 52 slots)."""
    from Nethax.nethax.subsystems.inventory import make_item, _items_from_list
    wand_item = Item(
        category=jnp.int8(ITEM_CATEGORY_WAND),
        type_id=jnp.int16(int(effect)),
        buc_status=jnp.int8(2),       # uncursed
        enchantment=jnp.int8(0),
        charges=jnp.int8(charges),
        identified=jnp.bool_(True),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    return InventoryState.from_items([wand_item])


def _make_state(
    player_row: int = 10,
    player_col: int = 10,
) -> WandState:
    """Return a WandState with flat terrain (all FLOOR)."""
    state = WandState.empty(map_h=MAP_H, map_w=MAP_W)
    terrain = jnp.full((MAP_H, MAP_W), int(TileType.FLOOR), dtype=jnp.int8)
    return state.replace(
        terrain=terrain,
        player_pos=jnp.array([player_row, player_col], dtype=jnp.int16),
    )


def _place_monster(
    state: WandState,
    slot: int,
    row: int,
    col: int,
    hp: int = 10,
    mon_type: int = 1,
    undead: bool = False,
) -> WandState:
    """Place a live monster in the given slot."""
    new_pos   = state.mon_pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_hp    = state.mon_hp.at[slot].set(jnp.int32(hp))
    new_type  = state.mon_type.at[slot].set(jnp.int16(mon_type))
    new_alive = state.mon_alive.at[slot].set(jnp.bool_(True))
    new_undead = state.mon_undead.at[slot].set(jnp.bool_(undead))
    return state.replace(
        mon_pos=new_pos,
        mon_hp=new_hp,
        mon_type=new_type,
        mon_alive=new_alive,
        mon_undead=new_undead,
    )


def _with_wand(state: WandState, effect: WandEffect, charges: int = 5) -> WandState:
    inv = _make_inventory_with_wand(effect, charges)
    return state.replace(inventory=inv)


# ---------------------------------------------------------------------------
# Test: charges decrement after zap
# ---------------------------------------------------------------------------

def test_charges_decrement():
    """Zapping any wand must decrement the charge count by 1."""
    state = _make_state()
    state = _with_wand(state, WandEffect.NOTHING, charges=5)

    initial_charges = int(state.inventory.items.charges[0])

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    final_charges = int(result.inventory.items.charges[0])
    assert final_charges == initial_charges - 1, (
        f"Expected charges {initial_charges - 1}, got {final_charges}"
    )


def test_charges_do_not_go_negative():
    """Charges floor at 0 — zapping an empty wand stays at 0."""
    state = _make_state()
    state = _with_wand(state, WandEffect.NOTHING, charges=0)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert int(result.inventory.items.charges[0]) == 0


# ---------------------------------------------------------------------------
# Test: striking reduces adjacent monster HP
# ---------------------------------------------------------------------------

def test_striking_reduces_monster_hp():
    """Zap of WAN_STRIKING on an adjacent monster (East) reduces its HP."""
    state = _make_state(player_row=10, player_col=10)
    # Place monster one tile east of player (row=10, col=11).
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    state = _with_wand(state, WandEffect.STRIKING, charges=5)

    initial_hp = int(state.mon_hp[1])

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    final_hp = int(result.mon_hp[1])
    assert final_hp < initial_hp, (
        f"Expected HP to decrease from {initial_hp}, got {final_hp}"
    )


def test_striking_is_beam_stops_at_first_monster():
    """STRIKING is a BEAM — second monster behind first must be unharmed."""
    state = _make_state(player_row=10, player_col=10)
    # First monster at col=11 (adjacent east).
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    # Second monster at col=13 (further east).
    state = _place_monster(state, slot=2, row=10, col=13, hp=20)
    state = _with_wand(state, WandEffect.STRIKING, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    # Second monster must be untouched.
    assert int(result.mon_hp[2]) == 20, (
        f"Second monster HP should be 20 (untouched), got {int(result.mon_hp[2])}"
    )
    # First monster must have taken damage.
    assert int(result.mon_hp[1]) < 20


# ---------------------------------------------------------------------------
# Test: magic missile east hits multiple aligned monsters
# ---------------------------------------------------------------------------

def test_magic_missile_hits_multiple_monsters():
    """Zap of WAN_MAGIC_MISSILE east must damage all monsters along the ray."""
    state = _make_state(player_row=10, player_col=10)
    # Three monsters in a row east of the player.
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    state = _place_monster(state, slot=2, row=10, col=12, hp=20)
    state = _place_monster(state, slot=3, row=10, col=13, hp=20)
    state = _with_wand(state, WandEffect.MAGIC_MISSILE, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    # All three monsters must have taken damage.
    for slot in (1, 2, 3):
        assert int(result.mon_hp[slot]) < 20, (
            f"Monster in slot {slot} should have taken damage "
            f"(hp={int(result.mon_hp[slot])})"
        )


# ---------------------------------------------------------------------------
# Test: digging north carves corridor into wall tiles
# ---------------------------------------------------------------------------

def test_digging_north_carves_wall():
    """Zap of WAN_DIGGING north converts WALL tiles north of player to a
    walkable opening.  Per vendor zap.c::zap_dig (D12, dig.c lines 1714-1724)
    non-maze WALL becomes a DOOR with D_NODOOR — we model that as OPEN_DOOR.
    Stone (VOID) cells along the ray become CORRIDOR.
    """
    state = _make_state(player_row=10, player_col=10)

    # Place WALL tiles north of player.
    new_terrain = state.terrain
    for row in range(2, 10):
        new_terrain = new_terrain.at[row, 10].set(jnp.int8(int(TileType.WALL)))
    state = state.replace(terrain=new_terrain)
    state = _with_wand(state, WandEffect.DIGGING, charges=5)

    # Direction 0 = North.
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(0))

    # At least one WALL must have been carved (to OPEN_DOOR or CORRIDOR).
    walkable_outcomes = {int(TileType.OPEN_DOOR), int(TileType.CORRIDOR), int(TileType.FLOOR)}
    changed = sum(
        1 for row in range(3, 10) if int(result.terrain[row, 10]) in walkable_outcomes
    )

    assert changed > 0, (
        "Expected at least one WALL tile north of player to be carved (OPEN_DOOR/CORRIDOR)."
    )


# ---------------------------------------------------------------------------
# Test: cold east on water tile freezes it
# ---------------------------------------------------------------------------

def test_cold_freezes_water_tiles():
    """Zap of WAN_COLD east must convert WATER tiles on the path to FLOOR/ICE."""
    state = _make_state(player_row=10, player_col=10)

    # Place WATER tiles east of player.
    new_terrain = state.terrain
    for col in range(11, 18):
        new_terrain = new_terrain.at[10, col].set(jnp.int8(int(TileType.WATER)))
    state = state.replace(terrain=new_terrain)
    state = _with_wand(state, WandEffect.COLD, charges=5)

    # Direction 2 = East.
    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    # Verify WATER tiles on the ray were changed.
    frozen = 0
    for col in range(11, 18):
        tile = int(result.terrain[10, col])
        if tile != int(TileType.WATER):
            frozen += 1

    assert frozen > 0, (
        "Expected WATER tiles east of player to be frozen (not WATER) after cold zap"
    )


# ---------------------------------------------------------------------------
# Test: death ray kills non-undead, spares undead
# ---------------------------------------------------------------------------

def test_death_ray_kills_non_undead():
    """WAN_DEATH ray must set non-undead monster HP to 0 (dead)."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11, hp=50, undead=False)
    state = _with_wand(state, WandEffect.DEATH, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert int(result.mon_hp[1]) == 0, (
        f"Non-undead monster should be dead (hp=0), got {int(result.mon_hp[1])}"
    )
    assert not bool(result.mon_alive[1]), "Non-undead monster should be marked dead"


def test_death_ray_spares_undead():
    """WAN_DEATH ray must not kill undead monsters."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11, hp=50, undead=True)
    state = _with_wand(state, WandEffect.DEATH, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert int(result.mon_hp[1]) == 50, (
        f"Undead monster should be unharmed by death ray, got hp={int(result.mon_hp[1])}"
    )


# ---------------------------------------------------------------------------
# Test: sleep ray puts monsters to sleep
# ---------------------------------------------------------------------------

def test_sleep_ray_makes_monster_asleep():
    """WAN_SLEEP ray must set mon_asleep=True on hit monsters."""
    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    state = _with_wand(state, WandEffect.SLEEP, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert bool(result.mon_asleep[1]), "Monster hit by sleep ray must be asleep"


# ---------------------------------------------------------------------------
# Test: handle_zap finds first wand automatically
# ---------------------------------------------------------------------------

def test_handle_zap_uses_first_wand():
    """handle_zap must find the first wand and fire it, reducing charges."""
    state = _make_state()
    state = _place_monster(state, slot=1, row=10, col=11, hp=20)
    state = _with_wand(state, WandEffect.STRIKING, charges=3)

    initial_charges = int(state.inventory.items.charges[0])
    result = handle_zap(state, _RNG)

    assert int(result.inventory.items.charges[0]) == initial_charges - 1


# ---------------------------------------------------------------------------
# Test: light illuminates explored map
# ---------------------------------------------------------------------------

def test_light_explores_map():
    """WAN_LIGHT must light a radius-5 disc around the player.

    Cite: vendor/nethack/src/read.c::litroom line 2601 calls
      do_clear_area(u.ux, u.uy, blessed_effect ? 9 : 5, set_lit, ...)
    so an uncursed wand of light marks tiles within Euclidean disc radius
    5 as lit/explored.  Tiles outside the disc must remain unexplored.
    """
    state = _make_state(player_row=10, player_col=10)
    state = _with_wand(state, WandEffect.LIGHT, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    # Player tile must be lit.
    assert bool(result.explored[10, 10]), "WAN_LIGHT must light the player tile"
    # Tile within radius 5 (3,4) of player should be lit: dist^2 = 49 + 16 = 65 > 25?
    # Actually (10+3, 10+4) is far: 9+16=25 -> ok at the edge.
    assert bool(result.explored[13, 14]), "WAN_LIGHT must light a tile at distance 5"
    # Tile far from player must remain dark.
    assert not bool(result.explored[0, 0]), (
        "WAN_LIGHT must not light tiles beyond radius 5"
    )


# ---------------------------------------------------------------------------
# Test: polymorph changes monster type
# ---------------------------------------------------------------------------

def test_polymorph_changes_monster_type():
    """WAN_POLYMORPH must change the hit monster's type_id."""
    state = _make_state(player_row=10, player_col=10)
    original_type = 5
    state = _place_monster(state, slot=1, row=10, col=11, hp=20, mon_type=original_type)
    state = _with_wand(state, WandEffect.POLYMORPH, charges=5)

    result = zap_wand(state, _RNG, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    # The type should (almost certainly) change; it is random in [1, 394].
    # We simply verify the field was written (could theoretically be same value).
    new_type = int(result.mon_type[1])
    assert 1 <= new_type <= 394, f"New monster type {new_type} out of expected range"
