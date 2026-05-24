"""Tests for priest extensions: pri_move, findpriest, mk_roamer.

Vendor sources:
  vendor/nethack/src/priest.c::pri_move    (lines 177-216)
  vendor/nethack/src/priest.c::findpriest  (lines 392-404)
  vendor/nethack/src/priest.c::mk_roamer   (lines 723-752)
"""

import pytest


def _fresh_state():
    import jax
    from Nethax.nethax.state import EnvState
    return EnvState.default(jax.random.PRNGKey(7))


# ---------------------------------------------------------------------------
# findpriest
# ---------------------------------------------------------------------------

def test_findpriest_returns_minus_one_when_no_priest():
    """Empty level → no priest, return -1.

    Vendor priest.c lines 392-404: walks fmon and returns 0 if no
    ispriest=true monster on shroom; we model that as int -1 sentinel.
    """
    from Nethax.nethax.subsystems.priest import findpriest
    state = _fresh_state()
    idx = findpriest(state, 0)
    assert int(idx) == -1


def test_findpriest_finds_peaceful_monster_on_shrine():
    """A peaceful monster standing on a shrine tile is reported as priest."""
    import jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import findpriest

    state = _fresh_state()
    # Place a shrine at (5, 5) on level 0.
    new_shrine = state.features.altar_shrine.at[0, 5, 5].set(jnp.bool_(True))
    new_has_temple = state.features.has_temple.at[0].set(jnp.bool_(True))
    new_features = state.features.replace(
        altar_shrine=new_shrine, has_temple=new_has_temple
    )
    # Make monster slot 3 alive + peaceful + on shrine tile (5,5).
    mai = state.monster_ai
    new_alive = mai.alive.at[3].set(jnp.bool_(True))
    new_peaceful = mai.peaceful.at[3].set(jnp.bool_(True))
    new_pos = mai.pos.at[3].set(jnp.array([5, 5], dtype=jnp.int16))
    new_mai = mai.replace(alive=new_alive, peaceful=new_peaceful, pos=new_pos)
    state = state.replace(features=new_features, monster_ai=new_mai)

    idx = findpriest(state, 0)
    assert int(idx) == 3


# ---------------------------------------------------------------------------
# mk_roamer
# ---------------------------------------------------------------------------

def test_mk_roamer_spawns_hostile_priest_in_first_empty_slot():
    """mk_roamer fills the lowest empty monster slot as hostile.

    Vendor priest.c lines 723-752: ``mpeaceful = peaceful`` (here False
    per task spec), ``isminion=1``, ``ispriest=0``.
    """
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import mk_roamer

    state = _fresh_state()
    s2 = mk_roamer(state, jax.random.PRNGKey(42))
    assert bool(s2.monster_ai.alive[0])
    assert not bool(s2.monster_ai.peaceful[0])
    assert int(s2.monster_ai.hp[0]) > 0


def test_mk_roamer_into_first_empty_after_existing():
    """If slot 0 is occupied, mk_roamer should use slot 1."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import mk_roamer

    state = _fresh_state()
    mai = state.monster_ai
    state = state.replace(
        monster_ai=mai.replace(alive=mai.alive.at[0].set(jnp.bool_(True)))
    )
    s2 = mk_roamer(state, jax.random.PRNGKey(43))
    assert bool(s2.monster_ai.alive[0])  # untouched
    assert bool(s2.monster_ai.alive[1])  # new spawn


# ---------------------------------------------------------------------------
# pri_move
# ---------------------------------------------------------------------------

def test_pri_move_walks_peaceful_priest_toward_shrine():
    """A peaceful priest near its shrine should step one tile toward it.

    Vendor priest.c lines 191-216: ggx/ggy = EPRI->shrpos; greedy 1-step
    via move_special.  We collapse to a single greedy step (clip(-1,+1)).
    """
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import pri_move

    state = _fresh_state()
    mai = state.monster_ai
    new_alive    = mai.alive.at[2].set(jnp.bool_(True))
    new_peaceful = mai.peaceful.at[2].set(jnp.bool_(True))
    new_pos      = mai.pos.at[2].set(jnp.array([3, 3], dtype=jnp.int16))
    state = state.replace(monster_ai=mai.replace(
        alive=new_alive, peaceful=new_peaceful, pos=new_pos
    ))
    # Shrine at (5, 5).
    new_shrine = state.features.altar_shrine.at[0, 5, 5].set(jnp.bool_(True))
    state = state.replace(features=state.features.replace(altar_shrine=new_shrine))

    s2 = pri_move(state, jnp.int32(2), jax.random.PRNGKey(44))
    # Greedy step toward (5,5) from (3,3) → (4,4).
    new_y = int(s2.monster_ai.pos[2, 0])
    new_x = int(s2.monster_ai.pos[2, 1])
    assert (new_y, new_x) == (4, 4), f"got ({new_y},{new_x}), expected (4,4)"


def test_pri_move_chases_player_when_hostile():
    """A hostile priest chases the player instead of the shrine.

    Vendor priest.c lines 197-211: !mpeaceful → ggx=u.ux, ggy=u.uy.
    """
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import pri_move

    state = _fresh_state()
    mai = state.monster_ai
    state = state.replace(
        player_pos=jnp.array([10, 10], dtype=jnp.int16),
        monster_ai=mai.replace(
            alive=mai.alive.at[1].set(jnp.bool_(True)),
            peaceful=mai.peaceful.at[1].set(jnp.bool_(False)),  # hostile
            pos=mai.pos.at[1].set(jnp.array([8, 8], dtype=jnp.int16)),
        ),
    )
    s2 = pri_move(state, jnp.int32(1), jax.random.PRNGKey(45))
    new_y = int(s2.monster_ai.pos[1, 0])
    new_x = int(s2.monster_ai.pos[1, 1])
    assert (new_y, new_x) == (9, 9), f"got ({new_y},{new_x}), expected (9,9)"


def test_pri_move_is_noop_for_dead_slot():
    """Dead slots stay put."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import pri_move

    state = _fresh_state()
    # Slot 5 stays alive=False; record its pos.
    before = tuple(int(v) for v in state.monster_ai.pos[5])
    s2 = pri_move(state, jnp.int32(5), jax.random.PRNGKey(46))
    after = tuple(int(v) for v in s2.monster_ai.pos[5])
    assert before == after


# ---------------------------------------------------------------------------
# JIT safety
# ---------------------------------------------------------------------------

def test_priest_helpers_are_jit_safe():
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import pri_move, findpriest, mk_roamer

    state = _fresh_state()
    # JIT each one
    jax.jit(mk_roamer)(state, jax.random.PRNGKey(0))
    jax.jit(pri_move)(state, jnp.int32(0), jax.random.PRNGKey(1))
    # findpriest takes a static level idx; use static_argnums.
    jax.jit(findpriest, static_argnums=(1,))(state, 0)
