"""Stair polish parity tests — wave-14 fixups.

Covers:
  1. M-e (0xE5) → _SLOT_ENHANCE, not _SLOT_EAT (cmd.c:1716 vendor-correct).
     Plain 'e' (0x65) still routes to _SLOT_EAT.
  2. pet_follow_on_stair wired from _stair_down: adjacent tame pet becomes
     not-alive on the source level after descending.
  3. Level memory round-trip: terrain stamped on L1 before descent is
     restored from cache after returning via stair-up.

Citation:
  vendor/nethack/src/cmd.c:1716   — M('e') → enhance_weapon_skill
  vendor/nethack/src/dog.c        — stair_pet (pet follows on stair)
  vendor/nethack/src/dungeon.c    — save_dungeon / restore_dungeon
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.constants import TileType
from Nethax.nethax.dungeon.branches import Branch
from Nethax.nethax.subsystems.action_dispatch import (
    _ACTION_TO_HANDLER_IDX,
    _stair_down,
    _stair_up,
    _SLOT_EAT,
    _SLOT_ENHANCE,
)
from Nethax.nethax.subsystems.monster_ai import make_monster_ai_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    rng = jax.random.PRNGKey(7)
    env = NethaxEnv()
    state, _ = env.reset(rng)
    return state, rng


def _slot(c) -> int:
    if isinstance(c, str):
        c = ord(c)
    return int(_ACTION_TO_HANDLER_IDX[c])


def _wire_stair_link(state, src_branch, src_level_1based, direction, dst_branch, dst_level_1based):
    new_sl = state.dungeon.stair_links.at[
        src_branch, src_level_1based - 1, direction
    ].set(jnp.array([dst_branch, dst_level_1based], dtype=jnp.int8))
    return state.replace(dungeon=state.dungeon.replace(stair_links=new_sl))


def _stamp_tile(state, branch, level_1based, row, col, tile: TileType):
    lv = level_1based - 1
    new_terrain = state.terrain.at[branch, lv, row, col].set(jnp.int8(int(tile)))
    return state.replace(terrain=new_terrain)


def _place_stair_under_player(state, tile: TileType):
    b  = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    r, c = int(state.player_pos[0]), int(state.player_pos[1])
    return _stamp_tile(state, b, lv + 1, r, c, tile)


# ---------------------------------------------------------------------------
# 1. test_meta_e_aliases_to_enhance
# ---------------------------------------------------------------------------

def test_meta_e_aliases_to_enhance():
    """M-e (0xE5) → _SLOT_ENHANCE per vendor cmd.c:1716.

    Plain 'e' (0x65) must still route to _SLOT_EAT so normal eat is intact.
    Citation: cmd.c:1716 M('e') → enhance_weapon_skill.
    """
    assert _slot(0xE5) == _SLOT_ENHANCE, (
        "M-e should map to _SLOT_ENHANCE (cmd.c:1716); "
        "wave-14 remap may not have applied"
    )
    assert _slot(ord("e")) == _SLOT_EAT, (
        "plain 'e' must still map to _SLOT_EAT"
    )


# ---------------------------------------------------------------------------
# 2. test_pet_follows_on_stair_down
# ---------------------------------------------------------------------------

def test_pet_follows_on_stair_down():
    """Adjacent tame pet is marked not-alive on the source level after _stair_down.

    Mirrors vendor/nethack/src/dog.c::stair_pet: pets within Chebyshev 1 of
    the player follow when the player descends stairs.
    """
    state, rng = _fresh_state()
    b = int(state.dungeon.current_branch)

    # Place STAIRCASE_DOWN under player; wire L1→L2; ensure L2 has STAIRCASE_UP.
    state = _place_stair_under_player(state, TileType.STAIRCASE_DOWN)
    state = _wire_stair_link(state, b, 1, 1, b, 2)
    state = _stamp_tile(state, b, 2, 10, 10, TileType.STAIRCASE_UP)

    # Spawn a tame pet adjacent (Chebyshev 1) to the player.
    pr, pc = int(state.player_pos[0]), int(state.player_pos[1])
    pet_idx = 0
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[pet_idx].set(True),
        tame=mai.tame.at[pet_idx].set(True),
        hp=mai.hp.at[pet_idx].set(10),
        hp_max=mai.hp_max.at[pet_idx].set(10),
        pos=mai.pos.at[pet_idx].set(jnp.array([pr + 1, pc], dtype=jnp.int16)),
    )
    state = state.replace(monster_ai=mai)

    assert bool(state.monster_ai.alive[pet_idx]), "pet should start alive"

    new_state = _stair_down(state, rng)

    assert int(new_state.dungeon.current_level) == 2, "player should be on L2"
    assert not bool(new_state.monster_ai.alive[pet_idx]), (
        "pet adjacent to player should be marked not-alive on source level "
        "after stair descent (dog.c::stair_pet)"
    )


# ---------------------------------------------------------------------------
# 3. test_level_memory_persists
# ---------------------------------------------------------------------------

def test_level_memory_persists():
    """Terrain stamped on L1 before descent is restored after returning.

    Sequence:
      - Stamp a WALL tile at a known position on L1.
      - Descend to L2 (triggers leave_level snapshot).
      - Ascend back to L1 (triggers enter_level restore from cache).
      - Confirm the WALL tile is still present at the original position.

    Citation: vendor/nethack/src/dungeon.c save_dungeon / restore_dungeon.
    """
    state, rng = _fresh_state()
    b = int(state.dungeon.current_branch)

    # Pick a position away from the player to stamp a sentinel tile.
    mark_row, mark_col = 3, 3
    sentinel = TileType.WALL

    # Stamp the sentinel on L1.
    state = _stamp_tile(state, b, 1, mark_row, mark_col, sentinel)
    assert int(state.terrain[b, 0, mark_row, mark_col]) == int(sentinel)

    # Wire stairs: L1 down → L2, L2 up → L1.
    state = _place_stair_under_player(state, TileType.STAIRCASE_DOWN)
    state = _wire_stair_link(state, b, 1, 1, b, 2)
    state = _wire_stair_link(state, b, 2, 0, b, 1)
    state = _stamp_tile(state, b, 2, 10, 10, TileType.STAIRCASE_UP)
    # Need STAIRCASE_DOWN on L2 for stair-up landing.
    state = _stamp_tile(state, b, 2, 10, 11, TileType.STAIRCASE_DOWN)

    # Descend to L2 — leave_level snapshots L1 into level_memory.
    state2 = _stair_down(state, rng)
    assert int(state2.dungeon.current_level) == 2

    # Place stair-up under player on L2.
    state2 = _place_stair_under_player(state2, TileType.STAIRCASE_UP)

    # Ascend back to L1 — enter_level restores L1 from cache.
    state1 = _stair_up(state2, rng)
    assert int(state1.dungeon.current_level) == 1

    restored_tile = int(state1.terrain[b, 0, mark_row, mark_col])
    assert restored_tile == int(sentinel), (
        f"Expected WALL ({int(sentinel)}) at ({mark_row},{mark_col}) on L1 "
        f"after round-trip, got {restored_tile}. "
        "level_memory snapshot/restore may not be wired in _stair_down/_stair_up."
    )
