"""Wand specials polish parity — vendor/nethack/src/zap.c.

Tests cover four parity fixes introduced in this wave:
  1. WAN_DEATH full immunity (zhitm ~4308): demons, golems, magic-resistant.
  2. WAN_CREATE_MONSTER level-appropriate generation.
  3. WAN_TELEPORTATION lands only on walkable tiles (FLOOR/CORRIDOR).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import InventoryState, Item
from Nethax.nethax.subsystems.items_wands import (
    ITEM_CATEGORY_WAND,
    N_MONSTERS,
    _MONSTER_GEN_LEVEL,
    WandEffect,
    WandState,
    zap_wand,
)
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL

MAP_H, MAP_W = 21, 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(player_row: int = 10, player_col: int = 10) -> WandState:
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
    hp: int = 50,
    mon_type: int = 1,
    entry_idx: int | None = None,
) -> WandState:
    """Place a live monster.  entry_idx sets mon_type to a specific MONSTERS index."""
    effective_type = entry_idx if entry_idx is not None else mon_type
    new_pos   = state.mon_pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16))
    new_hp    = state.mon_hp.at[slot].set(jnp.int32(hp))
    new_type  = state.mon_type.at[slot].set(jnp.int16(effective_type))
    new_alive = state.mon_alive.at[slot].set(jnp.bool_(True))
    return state.replace(mon_pos=new_pos, mon_hp=new_hp,
                         mon_type=new_type, mon_alive=new_alive)


def _with_wand(state: WandState, effect: WandEffect, charges: int = 10) -> WandState:
    wand_item = Item(
        category=jnp.int8(ITEM_CATEGORY_WAND),
        type_id=jnp.int16(int(effect)),
        buc_status=jnp.int8(2),
        enchantment=jnp.int8(0),
        charges=jnp.int8(charges),
        identified=jnp.bool_(True),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    return state.replace(inventory=InventoryState.from_items([wand_item]))


# ---------------------------------------------------------------------------
# 1. WAN_DEATH immunity — demons
# Cite: vendor/nethack/src/zap.c::zhitm ~4308:
#   if (nonliving(mon->data) || is_demon(mon->data) || ...) { break; }
# ---------------------------------------------------------------------------

def test_death_blocks_demon():
    """WAN_DEATH must leave a demon alive (M2_DEMON immunity).

    Uses MONSTERS tuple index 285 (water demon, vendor#297, chunk5 offset 36).
    Chunk5 starts at tuple index 249 (64+62+60+63=249); water demon is the
    37th entry (offset 36), so tuple index = 249+36 = 285.
    Cite: vendor/nethack/src/zap.c::zhitm ~4308 is_demon check.
    """
    # Verify our index really is a demon before testing.
    from Nethax.nethax.constants.monsters import MONSTERS, M2_DEMON
    assert bool(MONSTERS[285].flags2 & M2_DEMON), (
        f"Precondition: MONSTERS[285] must have M2_DEMON set; got {MONSTERS[285].name}"
    )

    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11, hp=50, entry_idx=285)
    state = _with_wand(state, WandEffect.DEATH)

    rng = jax.random.PRNGKey(42)
    result = zap_wand(state, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert bool(result.mon_alive[1]), (
        "WAN_DEATH must not kill a demon (M2_DEMON immune; zap.c::zhitm ~4308)"
    )
    assert int(result.mon_hp[1]) > 0, (
        "Demon HP must remain positive after WAN_DEATH"
    )


# ---------------------------------------------------------------------------
# 2. WAN_DEATH immunity — golems (nonliving)
# Cite: vendor/nethack/src/zap.c::zhitm ~4308 nonliving() check
#       (nonliving = is_undead | is_golem | is_vortex)
# ---------------------------------------------------------------------------

def test_death_blocks_nonliving():
    """WAN_DEATH must leave a golem alive (nonliving immunity).

    Uses MONSTERS tuple index 249 (leather golem, vendor#261, chunk5 offset 0).
    Chunk5 starts at tuple index 249 (64+62+60+63=249); leather golem is
    the first entry (offset 0), so tuple index = 249.
    Cite: vendor/nethack/src/zap.c::zhitm ~4308 nonliving() check.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    assert MONSTERS[249].symbol == MonsterSymbol.S_GOLEM, (
        f"Precondition: MONSTERS[249] must be a golem (S_GOLEM); got {MONSTERS[249].name}"
    )

    state = _make_state(player_row=10, player_col=10)
    state = _place_monster(state, slot=1, row=10, col=11, hp=50, entry_idx=249)
    state = _with_wand(state, WandEffect.DEATH)

    rng = jax.random.PRNGKey(99)
    result = zap_wand(state, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))

    assert bool(result.mon_alive[1]), (
        "WAN_DEATH must not kill a golem (nonliving immune; zap.c::zhitm ~4308)"
    )
    assert int(result.mon_hp[1]) > 0, (
        "Golem HP must remain positive after WAN_DEATH"
    )


# ---------------------------------------------------------------------------
# 3. WAN_CREATE_MONSTER level-appropriate
# Cite: vendor/nethack/src/zap.c wand_create_monster level logic
# ---------------------------------------------------------------------------

def test_create_monster_level_appropriate():
    """Over 30 trials, WAN_CREATE_MONSTER must only spawn monsters with
    gen_level <= dungeon_level + 3.

    Cite: vendor/nethack/src/zap.c wand_create_monster — level-appropriate
    selection via makemon.  _MONSTER_GEN_LEVEL uses MonsterEntry.level proxy.
    """
    dungeon_lv = 5
    max_allowed = dungeon_lv + 3

    state = _make_state()
    state = state.replace(dungeon_level=jnp.int8(dungeon_lv))
    state = _with_wand(state, WandEffect.CREATE_MONSTER, charges=35)

    for trial in range(30):
        rng = jax.random.PRNGKey(trial + 1000)
        result = zap_wand(state, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))
        # Find the newly spawned monster (first alive non-sentinel slot).
        for slot in range(1, MAX_MONSTERS_PER_LEVEL):
            if bool(result.mon_alive[slot]):
                mtype = int(result.mon_type[slot])
                mtype = max(0, min(mtype, N_MONSTERS - 1))
                gen_lv = int(_MONSTER_GEN_LEVEL[mtype])
                assert gen_lv <= max_allowed, (
                    f"Trial {trial}: spawned monster type {mtype} "
                    f"has gen_level {gen_lv} > dungeon_level+3={max_allowed} "
                    f"(zap.c wand_create_monster level check)"
                )
                break  # only check the first newly alive slot


# ---------------------------------------------------------------------------
# 4. WAN_TELEPORTATION dest validity
# Cite: vendor/nethack/src/zap.c::u_teleport_mon — picks typ==ROOM or CORR
# ---------------------------------------------------------------------------

def test_teleportation_lands_on_walkable():
    """Over 200 trials, WAN_TELEPORTATION dest must always be FLOOR or CORRIDOR.

    Cite: vendor/nethack/src/zap.c::u_teleport_mon ~line 345 —
    destination restricted to typ==ROOM (FLOOR) or CORR (CORRIDOR).
    Uses rejection sampling under lax.while_loop; JIT-pure.
    """
    # Build a mixed terrain map: mostly WALL, with a band of FLOOR in the middle.
    terrain = jnp.full((MAP_H, MAP_W), int(TileType.WALL), dtype=jnp.int8)
    # Rows 8-12, cols 5-75 are FLOOR.
    for r in range(8, 13):
        terrain = terrain.at[r, 5:76].set(jnp.int8(int(TileType.FLOOR)))
    # Add a CORRIDOR row.
    terrain = terrain.at[7, 5:76].set(jnp.int8(int(TileType.CORRIDOR)))

    state = WandState.empty(map_h=MAP_H, map_w=MAP_W)
    state = state.replace(
        terrain=terrain,
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
    )
    # Place a monster one step east of the player.
    state = _place_monster(state, slot=1, row=10, col=11, hp=50)
    # Use charges=5 and re-apply _with_wand each trial so int8 doesn't overflow.
    base_state = _with_wand(state, WandEffect.TELEPORTATION, charges=5)

    floor_v    = int(TileType.FLOOR)
    corridor_v = int(TileType.CORRIDOR)

    for trial in range(200):
        # Rebuild monster HP each trial (teleport doesn't change HP, just pos).
        trial_state = _place_monster(base_state, slot=1, row=10, col=11, hp=50)
        rng = jax.random.PRNGKey(trial + 5000)
        result = zap_wand(trial_state, rng, slot_idx=jnp.int32(0), direction=jnp.int32(2))
        dest_row = int(result.mon_pos[1, 0])
        dest_col = int(result.mon_pos[1, 1])
        tile = int(terrain[dest_row, dest_col])
        assert tile in (floor_v, corridor_v), (
            f"Trial {trial}: teleported to ({dest_row},{dest_col}) "
            f"tile={tile}, expected FLOOR={floor_v} or CORRIDOR={corridor_v} "
            f"(zap.c::u_teleport_mon dest validity)"
        )
