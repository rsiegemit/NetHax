"""Wave 5 Phase 1 deepened-AI tests.

Covers (one per task spec):
    - monster_can_see_player: LoS blocked by wall vs clear
    - pathfind_step: navigates around wall; falls back when unreachable
    - maybe_retreat: low HP → step away; high HP → no retreat step
    - monster_cast_spell: mage-class wizard hurts the player in LoS
    - pet_move: follows player, attacks adjacent hostile
    - maybe_wake_monster: asleep monster wakes when player enters LoS
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.monster_ai import (
    MAX_MONSTERS_PER_LEVEL,
    MoveStrategy,
    monster_can_see_player,
    monster_cast_spell,
    monster_turn,
    monster_use_item,
    maybe_retreat,
    maybe_wake_monster,
    pathfind_step,
    pet_move,
    _MAGE_ENTRY_LO,
)

# Wave 6 parity-fix: updated to match vendor/nethack/src/mcastu.c::castmu
# which gates on MonsterEntry.sound == MS_SPELL / MS_PRIEST.  We use the
# vendor "titan" entry (idx=173, MS_SPELL) as a real spellcaster instead of
# the synthetic _MAGE_ENTRY_LO range.
_TITAN_ENTRY_IDX = 173    # MONSTERS index for "titan" (sound = MS_SPELL = 42)
_KITTEN_ENTRY_IDX = 32    # MONSTERS index for "kitten" (sound = MS_MEW)

_RNG = jax.random.PRNGKey(7)


def _floor_state() -> EnvState:
    """EnvState with all-floor terrain on the current branch+level."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)
    floor_map = jnp.full((static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8)
    return state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_hp=jnp.int32(30),
        player_hp_max=jnp.int32(30),
    )


def _set_monster(
    state: EnvState,
    slot: int,
    pos,
    hp: int = 20,
    hp_max: int = 20,
    tame: bool = False,
    peaceful: bool = False,
    asleep: bool = False,
    entry_idx: int = 0,
) -> EnvState:
    """Stamp one monster slot."""
    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array(pos, dtype=jnp.int16)),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp_max)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        asleep=mai.asleep.at[slot].set(jnp.bool_(asleep)),
        tame=mai.tame.at[slot].set(jnp.bool_(tame)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(peaceful)),
        ac=mai.ac.at[slot].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[slot].set(jnp.int8(1)),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(jnp.int8(4)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# 1. LoS
# ---------------------------------------------------------------------------

def test_monster_los_clear():
    """Open floor between monster and player → LoS clear."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    assert bool(monster_can_see_player(state, jnp.int32(0))), (
        "Expected clear LoS on open floor"
    )


def test_monster_los_blocked_by_wall():
    """A wall between monster and player blocks LoS."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    # Stamp a wall segment between them at col=12.
    new_terrain = state.terrain.at[0, 0, 10, 12].set(jnp.int8(TileType.WALL))
    state = state.replace(terrain=new_terrain)
    assert not bool(monster_can_see_player(state, jnp.int32(0))), (
        "Wall between monster and player should block LoS"
    )


# ---------------------------------------------------------------------------
# 2. Pathfinding
# ---------------------------------------------------------------------------

def test_pathfind_navigates_around_wall():
    """Wall in the direct path → pathfinder picks a non-greedy step.

    Monster at (10, 10), player at (10, 14).  We block (10, 11), (10, 12),
    (10, 13).  Direct greedy goes col+1; pathfinder must move via row±1.
    """
    state = _floor_state().replace(player_pos=jnp.array([10, 14], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    new_terrain = state.terrain
    for c in (11, 12, 13):
        new_terrain = new_terrain.at[0, 0, 10, c].set(jnp.int8(TileType.WALL))
    state = state.replace(terrain=new_terrain)

    step = pathfind_step(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Pathfinder must NOT move into the wall directly to the east.
    # It should go diagonally (dy != 0) so it can route around.
    assert dy != 0, f"Expected detour via row-change, got ({dy}, {dx})"


def test_pathfind_fallback_when_unreachable():
    """If walls completely seal off the player, pathfinder falls back to
    a greedy 8-dir step (Chebyshev gradient)."""
    state = _floor_state().replace(player_pos=jnp.array([10, 14], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    # Build a closed wall box around the player.
    new_terrain = state.terrain
    for r in (9, 10, 11):
        for c in (13, 14, 15):
            if (r, c) != (10, 14):
                new_terrain = new_terrain.at[0, 0, r, c].set(jnp.int8(TileType.WALL))
    state = state.replace(terrain=new_terrain)

    step = pathfind_step(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Greedy fallback should head east — dx == +1.
    assert dx == 1, f"Greedy fallback should head east, got ({dy}, {dx})"


# ---------------------------------------------------------------------------
# 3. Retreat
# ---------------------------------------------------------------------------

def test_retreat_when_low_hp():
    """HP at 1/20 (5%) → maybe_retreat steps AWAY from player."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=20)
    step = maybe_retreat(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    # Player is east; retreat is west → dx == -1.
    assert dx == -1, f"Expected retreat west, got ({dy}, {dx})"


def test_no_retreat_when_high_hp():
    """HP at 16/20 (80%) → maybe_retreat returns (0, 0)."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=16, hp_max=20)
    step = maybe_retreat(state, jnp.int32(0))
    dy, dx = int(step[0]), int(step[1])
    assert (dy, dx) == (0, 0), f"Expected no retreat, got ({dy}, {dx})"


# ---------------------------------------------------------------------------
# 4. Spell casting
# ---------------------------------------------------------------------------

def test_mage_monster_casts_spell_at_player():
    """A mage-class monster in LoS within range deals spell damage."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(
        state, 0, pos=(10, 10),
        hp=40, hp_max=40,
        # Wave 6 parity-fix: use real MS_SPELL monster (titan) per vendor mcastu.c
        entry_idx=_TITAN_ENTRY_IDX,
    )
    orig_hp = int(state.player_hp)
    any_damage = False
    for seed in range(10):
        rng = jax.random.PRNGKey(seed + 500)
        new_state = monster_cast_spell(state, rng, jnp.int32(0))
        if int(new_state.player_hp) < orig_hp:
            any_damage = True
            break
    assert any_damage, "Mage monster never dealt spell damage across 10 seeds"


def test_non_mage_does_not_cast_spell():
    """A non-mage monster (entry_idx out of mage range) does not cast."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=40, hp_max=40, entry_idx=10)
    orig_hp = int(state.player_hp)
    for seed in range(5):
        rng = jax.random.PRNGKey(seed + 600)
        new_state = monster_cast_spell(state, rng, jnp.int32(0))
        assert int(new_state.player_hp) == orig_hp


# ---------------------------------------------------------------------------
# 5. Pet behavior
# ---------------------------------------------------------------------------

def test_pet_follows_player():
    """A pet (tame monster) not adjacent to anyone hostile steps toward the player."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), tame=True)
    rng = jax.random.PRNGKey(10)
    new_state = pet_move(state, rng, jnp.int32(0))
    orig_col = int(state.monster_ai.pos[0, 1])
    new_col = int(new_state.monster_ai.pos[0, 1])
    assert new_col > orig_col, (
        f"Pet should step east toward player; col went {orig_col} -> {new_col}"
    )


def test_pet_attacks_hostile_monster():
    """A pet next to a hostile monster deals damage to it (instead of moving)."""
    state = _floor_state().replace(player_pos=jnp.array([0, 0], dtype=jnp.int16))
    # Pet at (10, 10), hostile at (10, 11).
    state = _set_monster(state, 0, pos=(10, 10), tame=True, hp=20, hp_max=20)
    state = _set_monster(state, 1, pos=(10, 11), tame=False, hp=10, hp_max=10)

    orig_hostile_hp = int(state.monster_ai.hp[1])
    rng = jax.random.PRNGKey(11)
    new_state = pet_move(state, rng, jnp.int32(0))
    new_hostile_hp = int(new_state.monster_ai.hp[1])
    assert new_hostile_hp < orig_hostile_hp, (
        f"Pet should damage adjacent hostile; hp went {orig_hostile_hp} -> {new_hostile_hp}"
    )


# ---------------------------------------------------------------------------
# 6. Wake on player visible
# ---------------------------------------------------------------------------

def test_sleeping_monster_wakes_on_player_seen():
    """Asleep monster + player in LoS → asleep flips False."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), asleep=True)
    assert bool(state.monster_ai.asleep[0])
    new_state = maybe_wake_monster(state, jnp.int32(0))
    assert not bool(new_state.monster_ai.asleep[0]), (
        "Sleeping monster should wake when player enters LoS"
    )


def test_sleeping_monster_stays_asleep_behind_wall():
    """Asleep monster + wall blocking LoS → stays asleep."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), asleep=True)
    new_terrain = state.terrain.at[0, 0, 10, 12].set(jnp.int8(TileType.WALL))
    state = state.replace(terrain=new_terrain)
    new_state = maybe_wake_monster(state, jnp.int32(0))
    assert bool(new_state.monster_ai.asleep[0]), (
        "Wall-blocked monster should stay asleep"
    )


# ---------------------------------------------------------------------------
# Smoke: monster_use_item runs cleanly (Wave 5 stub branches)
# ---------------------------------------------------------------------------

def test_monster_use_item_runs():
    """Stub branches should be JIT-callable and return state unchanged."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    # Wave 6 parity-fix: use real MS_SPELL monster (titan) per vendor mcastu.c
    state = _set_monster(state, 0, pos=(10, 10), hp=2, hp_max=20,
                         entry_idx=_TITAN_ENTRY_IDX)
    rng = jax.random.PRNGKey(42)
    new_state = monster_use_item(state, rng, jnp.int32(0))
    # No fields should have changed (stubs are no-ops in Phase 1).
    assert int(new_state.monster_ai.hp[0]) == int(state.monster_ai.hp[0])


# ---------------------------------------------------------------------------
# Integration: monster_turn dispatches through new code paths
# ---------------------------------------------------------------------------

def test_monster_turn_with_pathfind_moves_toward_player():
    """Through the full monster_turn refactor: non-pet monster moves toward player."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10))
    rng = jax.random.PRNGKey(99)
    new_state = monster_turn(state, rng, jnp.int32(0))
    orig_col = int(state.monster_ai.pos[0, 1])
    new_col = int(new_state.monster_ai.pos[0, 1])
    assert new_col > orig_col


def test_monster_turn_retreats_at_low_hp():
    """Low-HP monster steps AWAY from player via monster_turn."""
    state = _floor_state().replace(player_pos=jnp.array([10, 15], dtype=jnp.int16))
    state = _set_monster(state, 0, pos=(10, 10), hp=1, hp_max=20)
    rng = jax.random.PRNGKey(123)
    new_state = monster_turn(state, rng, jnp.int32(0))
    orig_col = int(state.monster_ai.pos[0, 1])
    new_col = int(new_state.monster_ai.pos[0, 1])
    # Retreating, so col should DECREASE (move west, away from east player).
    assert new_col < orig_col, (
        f"Low-HP monster should retreat west; col went {orig_col} -> {new_col}"
    )
