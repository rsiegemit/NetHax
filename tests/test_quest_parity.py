"""Quest progression parity tests.

Covers the four spec-required stage transitions:
  - test_meet_leader_advances_stage      (on_enter_quest_level → met_leader=True, stage=1)
  - test_pickup_artifact_advances_stage  (on_artifact_picked_up → stage=2)
  - test_kill_nemesis_sets_flag          (on_nemesis_killed → nemesis_killed=True)
  - test_return_to_leader_completes      (on_return_to_leader → completed=True)

Citations:
  vendor/nethack/src/quest.c::chat_with_leader (~321-324) — met_leader
  vendor/nethack/src/quest.c::artitouch (~127-134)         — touched_artifact
  vendor/nethack/src/quest.c::nemdead (~109-113)           — killed_nemesis
  vendor/nethack/src/quest.c::finish_quest (~263-279)      — qcompleted
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.dungeon.branches import Branch
from Nethax.nethax.subsystems.quest import (
    on_enter_quest_level,
    on_nemesis_killed,
    on_artifact_picked_up,
    on_return_to_leader,
    _QSTAGE_BEGUN,
    _QSTAGE_GOT_OBJ,
    _QSTAGE_COMPLETE,
)


_RNG = jax.random.PRNGKey(42)


def _make_state(**quest_overrides):
    """Return a fresh EnvState with optional quest field overrides."""
    state = EnvState.default(_RNG)
    if quest_overrides:
        new_quest = state.quest.replace(**quest_overrides)
        state = state.replace(quest=new_quest)
    return state


def _on_quest_branch(state):
    """Return state with current_branch=QUEST, current_level=1."""
    new_dungeon = state.dungeon.replace(
        current_branch=jnp.int8(Branch.QUEST),
        current_level=jnp.int8(1),
    )
    return state.replace(dungeon=new_dungeon)


# ---------------------------------------------------------------------------
# test_meet_leader_advances_stage
# ---------------------------------------------------------------------------

def test_meet_leader_advances_stage():
    """Entering Quest branch level 1 sets met_leader=True and stage=1.

    Mirrors quest.c::chat_with_leader ~321-324:
      if (!Qstat(met_leader)) { Qstat(met_leader) = TRUE; ... }
    and qstplay.c on_leader_level / expulsion gate.
    """
    state = _on_quest_branch(_make_state())
    assert not bool(state.quest.met_leader), "precondition: met_leader=False"

    new_state = on_enter_quest_level(state)

    assert bool(new_state.quest.met_leader), "met_leader must be True after entering leader level"
    assert int(new_state.quest.stage) == int(_QSTAGE_BEGUN), (
        f"stage must be {int(_QSTAGE_BEGUN)} (BEGUN_QUEST), got {int(new_state.quest.stage)}"
    )


def test_meet_leader_idempotent():
    """Calling on_enter_quest_level twice keeps stage at BEGUN_QUEST (1)."""
    state = _on_quest_branch(_make_state())
    s1 = on_enter_quest_level(state)
    s2 = on_enter_quest_level(s1)
    assert bool(s2.quest.met_leader)
    assert int(s2.quest.stage) == int(_QSTAGE_BEGUN)


# ---------------------------------------------------------------------------
# test_pickup_artifact_advances_stage
# ---------------------------------------------------------------------------

def test_pickup_artifact_advances_stage():
    """Picking up the quest artifact sets touched_artifact=True and stage=2.

    Mirrors quest.c::artitouch ~127-134:
      if (!Qstat(touched_artifact)) { Qstat(touched_artifact) = TRUE; ... }
    """
    state = _make_state()
    assert not bool(state.quest.touched_artifact), "precondition: not yet touched"

    new_state = on_artifact_picked_up(state)

    assert bool(new_state.quest.touched_artifact), "touched_artifact must be True"
    assert bool(new_state.quest.artifact_carried), "artifact_carried must be True"
    assert int(new_state.quest.stage) == int(_QSTAGE_GOT_OBJ), (
        f"stage must be {int(_QSTAGE_GOT_OBJ)} (GOT_QUEST_OBJECT), got {int(new_state.quest.stage)}"
    )


def test_pickup_artifact_does_not_regress_stage():
    """on_artifact_picked_up does not lower a stage already beyond GOT_OBJ."""
    state = _make_state()
    # Manually advance stage past GOT_OBJ.
    new_quest = state.quest.replace(stage=jnp.int8(3))
    state = state.replace(quest=new_quest)

    new_state = on_artifact_picked_up(state)
    assert int(new_state.quest.stage) == 3, "stage must not regress"


# ---------------------------------------------------------------------------
# test_kill_nemesis_sets_flag
# ---------------------------------------------------------------------------

def test_kill_nemesis_sets_flag():
    """Killing the nemesis monster sets nemesis_killed=True.

    Mirrors quest.c::nemdead ~109-113:
      if (!Qstat(killed_nemesis)) { Qstat(killed_nemesis) = TRUE; ... }
    """
    state = _make_state()
    assert not bool(state.quest.nemesis_killed), "precondition: nemesis alive"

    # Pass a dummy monster_entry_idx (the value is accepted; the caller in
    # combat.py already confirmed it matches the role's nemesis).
    nemesis_entry = jnp.int32(355)  # PM_MINION_OF_HUHETOTL (Arc nemesis)
    new_state = on_nemesis_killed(state, nemesis_entry)

    assert bool(new_state.quest.nemesis_killed), "nemesis_killed must be True"
    assert not bool(new_state.quest.nemesis_alive), "nemesis_alive must be False"


def test_kill_nemesis_idempotent():
    """Calling on_nemesis_killed twice is a no-op after the first call."""
    state = _make_state()
    s1 = on_nemesis_killed(state, jnp.int32(355))
    s2 = on_nemesis_killed(s1, jnp.int32(355))
    assert bool(s2.quest.nemesis_killed)
    assert not bool(s2.quest.nemesis_alive)


# ---------------------------------------------------------------------------
# test_return_to_leader_completes
# ---------------------------------------------------------------------------

def test_return_to_leader_completes():
    """Returning to the leader with the artifact sets completed=True and stage=4.

    Mirrors quest.c::finish_quest ~263-279:
      u.uevent.qcompleted = 1;
    """
    # Set up: nemesis killed + artifact carried (prerequisite conditions).
    state = _make_state()
    state = on_enter_quest_level(state)       # stage=1, met_leader=True
    state = on_nemesis_killed(state, jnp.int32(355))  # nemesis_killed=True
    state = on_artifact_picked_up(state)      # stage=2, touched_artifact=True
    assert not bool(state.quest.completed), "precondition: not yet completed"

    new_state = on_return_to_leader(state)

    assert bool(new_state.quest.completed), "completed must be True"
    assert int(new_state.quest.stage) == int(_QSTAGE_COMPLETE), (
        f"stage must be {int(_QSTAGE_COMPLETE)} (COMPLETED), got {int(new_state.quest.stage)}"
    )


def test_return_to_leader_completes_idempotent():
    """on_return_to_leader called twice keeps completed=True."""
    state = _make_state()
    state = on_nemesis_killed(state, jnp.int32(355))
    state = on_artifact_picked_up(state)
    s1 = on_return_to_leader(state)
    s2 = on_return_to_leader(s1)
    assert bool(s2.quest.completed)
    assert int(s2.quest.stage) == int(_QSTAGE_COMPLETE)
