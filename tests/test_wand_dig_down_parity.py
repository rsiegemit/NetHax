"""Parity tests for WAN_DIGGING down-dig and M-T (#tip) down-dig wiring.

Vendor reference:
  - vendor/nethack/src/zap.c::zap_dig line 1548 — downward zap branch.
  - vendor/nethack/src/dig.c::digactualhole line 640 — sets levl[].typ = HOLE.

Verifies:
  - ``_effect_digging`` with direction==8 sets terrain[player_row, player_col]
    to ``TileType.HOLE``.
  - Direction 0..7 still uses the horizontal carve path (regression).
  - The M-T action byte (``_M_byte("T")``) routes through dispatch_action and
    invokes the down-dig path, replacing the player-tile with HOLE.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.items_wands import _effect_digging, WandState
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL
from Nethax.nethax.subsystems.inventory import InventoryState
from Nethax.nethax.subsystems.traps import TrapState
from Nethax.nethax.subsystems.action_dispatch import (
    dispatch_action, _SLOT_TIP_DOWN, _M_byte, _ACTION_TO_HANDLER_IDX,
)


_RNG = jax.random.PRNGKey(0)


def _make_wand_state(player_pos=(5, 5), map_h=21, map_w=80):
    n = MAX_MONSTERS_PER_LEVEL
    return WandState(
        mon_pos       = jnp.zeros((n, 2), dtype=jnp.int16),
        mon_hp        = jnp.zeros(n, dtype=jnp.int32),
        mon_hp_max    = jnp.zeros(n, dtype=jnp.int32),
        mon_type      = jnp.zeros(n, dtype=jnp.int16),
        mon_alive     = jnp.zeros(n, dtype=bool),
        mon_asleep    = jnp.zeros(n, dtype=bool),
        mon_undead    = jnp.zeros(n, dtype=bool),
        mon_invisible = jnp.zeros(n, dtype=bool),
        mon_resists   = jnp.zeros(n, dtype=jnp.int32),
        mon_speed_mod = jnp.zeros(n, dtype=jnp.int8),
        mon_cancelled = jnp.zeros(n, dtype=bool),
        terrain       = jnp.full((map_h, map_w), int(TileType.FLOOR), dtype=jnp.int8),
        explored      = jnp.zeros((map_h, map_w), dtype=bool),
        inventory     = InventoryState.empty(),
        player_pos    = jnp.array(player_pos, dtype=jnp.int16),
        dungeon_level = jnp.int8(1),
        probed_hp     = jnp.int32(0),
        probed_idx    = jnp.int32(-1),
        player_reflecting = jnp.bool_(False),
        branch        = jnp.int8(0),
        traps         = TrapState.default(num_levels=1, map_h=map_h, map_w=map_w),
        wall_info     = jnp.zeros((1, map_h, map_w), dtype=jnp.bool_),
    )


def test_tile_type_has_hole():
    """TileType.HOLE was added with value 21."""
    assert int(TileType.HOLE) == 21


def test_dig_down_sets_hole_at_player_pos():
    """direction==8 → terrain[player_row, player_col] = TileType.HOLE."""
    ws = _make_wand_state(player_pos=(7, 12))
    new_ws, _ = _effect_digging(ws, _RNG, direction=jnp.int32(8))
    assert int(new_ws.terrain[7, 12]) == int(TileType.HOLE)
    # Surrounding floor tiles remain FLOOR.
    assert int(new_ws.terrain[7, 11]) == int(TileType.FLOOR)
    assert int(new_ws.terrain[6, 12]) == int(TileType.FLOOR)


def test_dig_horizontal_unaffected_by_down_branch():
    """direction in [0..7] carves a wall away — wand-dig walls become
    OPEN_DOOR (D_NODOOR per vendor zap.c::zap_dig line 1723).

    Cite: vendor/nethack/src/dig.c lines 1714-1724: non-maze WALL → DOOR
    with D_NODOOR (an open doorway).  The earlier port produced CORRIDOR
    unconditionally; D12 fixes the per-tile vendor outcomes.
    """
    ws = _make_wand_state(player_pos=(5, 5))
    # Place a wall east of player.
    new_terrain = ws.terrain.at[5, 6].set(jnp.int8(TileType.WALL))
    ws = ws.replace(terrain=new_terrain)
    # direction=2 == East.
    new_ws, _ = _effect_digging(ws, _RNG, direction=jnp.int32(2))
    # The wall has been carved to OPEN_DOOR (D_NODOOR).
    assert int(new_ws.terrain[5, 6]) == int(TileType.OPEN_DOOR)
    # Player tile is NOT a hole — horizontal dig does not touch player_pos.
    assert int(new_ws.terrain[5, 5]) != int(TileType.HOLE)


def test_mt_routes_to_tip_down_slot():
    """M-T action byte maps to _SLOT_TIP_DOWN."""
    mt = _M_byte("T")
    assert int(_ACTION_TO_HANDLER_IDX[mt]) == _SLOT_TIP_DOWN


def test_mt_dispatch_empties_carried_container_to_floor():
    """dispatch_action(M-T) empties a carried container onto the floor.

    Wave 43c rebalance: prior expectation was ``terrain[5,5] == HOLE`` after
    M-T, which pinned a Nethax-only divergence wiring #tip to the WAN_DIGGING
    down-dig path.  Vendor ``#tip`` (vendor/nethack/src/pickup.c::dotip 3562)
    calls ``tipcontainer`` which empties the chosen container's contents onto
    the floor at the hero's spot.

    Old expectation → new expectation:
      terrain[player] == HOLE  →  ground[player] holds the apple AND
                                   container slot is empty.
    """
    from Nethax.nethax.subsystems.containers import (
        install_container, put_in_container, ContainerType, BUCStatus,
    )
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = EnvState.default(rng=_RNG)
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1

    # Install a carried BoH at container slot 0 (parent_slot=2 → inv slot 2).
    state = install_container(
        state, 0, ContainerType.BAG_OF_HOLDING,
        parent_slot=2, buc=int(BUCStatus.UNCURSED),
    )

    # Place an apple in inventory slot 0 and stash it into the BoH.
    inv = state.inventory.items
    new_inv = inv.replace(
        category   = inv.category.at[0].set(jnp.int8(ItemCategory.FOOD)),
        type_id    = inv.type_id.at[0].set(jnp.int16(123)),
        quantity   = inv.quantity.at[0].set(jnp.int16(1)),
        weight     = inv.weight.at[0].set(jnp.int32(5)),
        identified = inv.identified.at[0].set(jnp.bool_(True)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_inv))
    state = put_in_container(state, 0, 0)

    # Sanity: container has the apple before tip.
    assert int(state.containers.items_category[0, 0]) == int(ItemCategory.FOOD)
    assert int(state.ground_items.category[b, lv, 5, 5, 0]) == 0

    out = dispatch_action(state, jnp.int32(_M_byte("T")), _RNG)

    # Container slot 0 is now empty …
    assert int(out.containers.items_category[0, 0]) == 0
    # … and the apple is on the ground at the player's tile.
    assert int(out.ground_items.category[b, lv, 5, 5, 0]) == int(ItemCategory.FOOD)
    assert int(out.ground_items.type_id[b, lv, 5, 5, 0]) == 123
