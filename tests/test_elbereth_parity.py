"""Elbereth-fear parity tests.

Covers:
  - test_elbereth_freezes_monster     : non-humanoid monster stays put when
                                        player's tile has Elbereth engraving.
  - test_elbereth_ignored_by_human    : S_HUMAN monster (watchman) still
                                        approaches despite Elbereth.
  - test_elbereth_erodes_on_step      : walking over dust Elbereth 30 times
                                        erodes it at least once.

Cite: vendor/nethack/src/monmove.c::onscary lines 241-303
      vendor/nethack/src/engrave.c::wipe_engr_at lines 270-290
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.actions import Command, CompassCardinalDirection
from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
from Nethax.nethax.subsystems.monster_ai import monster_turn, MAX_MONSTERS_PER_LEVEL
from Nethax.nethax.subsystems.engrave import handle_engrave, ENGR_DUST
from Nethax.nethax.subsystems.action_dispatch import dispatch_action, _try_step

_RNG = jax.random.PRNGKey(42)

# Resolve entry indices by name so we don't depend on chunk ordering.
# giant ant (S_ANT) — non-humanoid, non-exempt
_ENTRY_GIANT_ANT: int = next(
    i for i, m in enumerate(MONSTERS) if m.name == "giant ant"
)
# watchman (S_HUMAN) — exempt from Elbereth fear
_ENTRY_WATCHMAN: int = next(
    i for i, m in enumerate(MONSTERS) if m.name == "watchman"
)
assert MONSTERS[_ENTRY_GIANT_ANT].symbol != MonsterSymbol.S_HUMAN, \
    "giant ant must not be S_HUMAN"
assert MONSTERS[_ENTRY_WATCHMAN].symbol == MonsterSymbol.S_HUMAN, \
    "watchman must be S_HUMAN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _floor_state(player_pos=(10, 15), monster_pos=(10, 10), entry_idx: int = 0) -> EnvState:
    """Return a state with all-floor terrain and one live monster at slot 0."""
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)

    floor_map = jnp.full(
        (static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8
    )
    state = state.replace(
        terrain=state.terrain.at[0, 0].set(floor_map),
        player_pos=jnp.array(player_pos, dtype=jnp.int16),
        player_hp=jnp.int32(30),
        player_hp_max=jnp.int32(30),
    )

    mai = state.monster_ai
    mai = mai.replace(
        pos=mai.pos.at[0].set(jnp.array(monster_pos, dtype=jnp.int16)),
        hp=mai.hp.at[0].set(jnp.int32(20)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(20)),
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        asleep=mai.asleep.at[0].set(jnp.bool_(False)),
        peaceful=mai.peaceful.at[0].set(jnp.bool_(False)),
        ac=mai.ac.at[0].set(jnp.int8(10)),
        attack_dice_n=mai.attack_dice_n.at[0].set(jnp.int8(1)),
        attack_dice_sides=mai.attack_dice_sides.at[0].set(jnp.int8(4)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


def _write_elbereth(state, row: int, col: int) -> EnvState:
    """Write Elbereth at (row, col) by placing the player there and engraving."""
    old_pos = state.player_pos
    state = state.replace(player_pos=jnp.array([row, col], dtype=jnp.int16))
    state = handle_engrave(state, _RNG)
    return state.replace(player_pos=old_pos)


# ---------------------------------------------------------------------------
# test_elbereth_freezes_monster
# ---------------------------------------------------------------------------

def test_elbereth_freezes_monster():
    """Non-humanoid monster adjacent to Elbereth'd player tile must not move.

    Cite: vendor/nethack/src/monmove.c::onscary lines 241-303.
    We choose giant ant (entry_idx=0, S_ANT) which is not exempt.
    Monster at (10, 12), player at (10, 15) — monster would normally step
    toward player.  With Elbereth at player's tile it stays at (10, 12).
    """
    player_r, player_c = 10, 15
    monster_r, monster_c = 10, 12
    state = _floor_state(
        player_pos=(player_r, player_c),
        monster_pos=(monster_r, monster_c),
        entry_idx=_ENTRY_GIANT_ANT,
    )
    # Write Elbereth at the player's tile.
    state = _write_elbereth(state, player_r, player_c)

    assert bool(state.engrave.has_engraving[player_r, player_c]), \
        "Elbereth should be present at player tile before monster turn"

    new_state = monster_turn(state, _RNG, jnp.int32(0))

    final_pos = new_state.monster_ai.pos[0]
    assert int(final_pos[0]) == monster_r and int(final_pos[1]) == monster_c, (
        f"Scared monster should not move; expected ({monster_r},{monster_c}) "
        f"but got ({int(final_pos[0])},{int(final_pos[1])})"
    )


# ---------------------------------------------------------------------------
# test_elbereth_ignored_by_human
# ---------------------------------------------------------------------------

def test_elbereth_ignored_by_human():
    """S_HUMAN monster (watchman) still approaches despite Elbereth.

    Cite: vendor/nethack/src/monmove.c::onscary lines 241-303 — humanoids
    are exempt from Elbereth fear (``ishumanoid`` branch).
    Monster at (10, 12), player at (10, 15); monster should step to col 13.
    """
    player_r, player_c = 10, 15
    monster_r, monster_c = 10, 12
    state = _floor_state(
        player_pos=(player_r, player_c),
        monster_pos=(monster_r, monster_c),
        entry_idx=_ENTRY_WATCHMAN,
    )
    # Write Elbereth at player's tile — watchman ignores it.
    state = _write_elbereth(state, player_r, player_c)

    new_state = monster_turn(state, _RNG, jnp.int32(0))

    final_pos = new_state.monster_ai.pos[0]
    # Monster should have moved closer (col 13 or row changed), not stayed.
    stayed = (int(final_pos[0]) == monster_r and int(final_pos[1]) == monster_c)
    assert not stayed, (
        f"S_HUMAN watchman should ignore Elbereth and move; "
        f"still at ({monster_r},{monster_c})"
    )


# ---------------------------------------------------------------------------
# test_elbereth_erodes_on_step
# ---------------------------------------------------------------------------

def test_elbereth_erodes_on_step():
    """Walking over a dust Elbereth 30 times should erase it at least once.

    Cite: vendor/nethack/src/engrave.c::wipe_engr_at lines 270-290.
    Probability 1/4 per step → P(never erased in 30 steps) = (3/4)^30 ≈ 0.18%.
    We use 30 steps which makes this failure probability negligibly small.
    """
    static = StaticParams()
    state = EnvState.default(rng=_RNG, static=static)

    # All-floor map so player can freely step back and forth.
    floor_map = jnp.full(
        (static.map_h, static.map_w), TileType.FLOOR, dtype=jnp.int8
    )
    state = state.replace(terrain=state.terrain.at[0, 0].set(floor_map))

    # Start player at (5, 10); engrave Elbereth there.
    engrave_r, engrave_c = 5, 10
    state = state.replace(player_pos=jnp.array([engrave_r, engrave_c], dtype=jnp.int16))
    state = handle_engrave(state, _RNG)
    assert bool(state.engrave.has_engraving[engrave_r, engrave_c]), \
        "Elbereth should exist before stepping"

    # Step player east off the engraving, then west back onto it, 30 times.
    # Each west step (dy=0, dx=-1) lands on (engrave_r, engrave_c) and has
    # 1/4 chance to erase the dust Elbereth.
    # Use _try_step directly (dispatch_action has a pre-existing _apply_fov
    # JIT issue unrelated to the wipe logic under test).
    erased = False
    for i in range(30):
        if not bool(state.engrave.has_engraving[engrave_r, engrave_c]):
            erased = True
            break
        rng_i = jax.random.PRNGKey(i + 100)
        # Step east (off the engraving tile)
        state = _try_step(state, 0, 1, rng_i)
        # Step west (back onto the engraving tile)
        rng_j = jax.random.PRNGKey(i + 1000)
        state = _try_step(state, 0, -1, rng_j)

    if not erased:
        erased = not bool(state.engrave.has_engraving[engrave_r, engrave_c])

    assert erased, (
        "Elbereth dust engraving should have been erased within 30 back-and-forth "
        "steps (p(not erased) ≈ 0.18%)"
    )
