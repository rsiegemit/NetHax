"""Smoke tests for ``Nethax.nethax.subsystems.mplayer``.

Vendor: vendor/nethack/src/mplayer.c::mk_mplayer (lines 118-326),
        vendor/nethack/src/mplayer.c::create_mplayers (lines 327-355).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.subsystems.mplayer import (
    ASTRAL_BRANCH,
    ASTRAL_LEVEL,
    ASTRAL_MPLAYER_COUNT,
    MPLAYER_ROLES,
    create_mplayers,
    maybe_seed_astral_mplayers,
    mk_mplayer,
)


def test_mplayer_roles_resolve_to_canonical_names():
    """The 13 entry_idx values point at the right MONSTERS entries."""
    assert len(MPLAYER_ROLES) == 13
    for idx, name in MPLAYER_ROLES:
        assert MONSTERS[idx].name == name


def test_mk_mplayer_spawns_in_first_dead_slot():
    """``mk_mplayer`` claims the first dead slot at the requested pos."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    mai0 = state.monster_ai

    # Force a known dead slot.  Reset spawns 5 wild monsters in slots 0..4
    # and the starting pet in slot 5; slot 6 is the first dead one.
    dead_mask = ~mai0.alive
    assert bool(jnp.any(dead_mask))
    expected_slot = int(jnp.argmax(dead_mask.astype(jnp.int32)))

    pos = jnp.array([4, 7], dtype=jnp.int16)
    state2 = mk_mplayer(state, jax.random.PRNGKey(1), 0, pos)  # archeologist
    mai = state2.monster_ai

    assert bool(mai.alive[expected_slot])
    assert int(mai.entry_idx[expected_slot]) == MPLAYER_ROLES[0][0]
    assert int(mai.pos[expected_slot, 0]) == 4
    assert int(mai.pos[expected_slot, 1]) == 7
    # vendor: mhp = mhpmax — invariant must hold.
    assert int(mai.hp[expected_slot]) == int(mai.hp_max[expected_slot])
    # rn1(16, 15) → [15, 30] (vendor: lower half of the special-case range).
    assert 15 <= int(mai.hp[expected_slot]) <= 30
    # Hostile by default — vendor mplayer.c line 146.
    assert bool(mai.peaceful[expected_slot]) is False
    assert bool(mai.tame[expected_slot]) is False


def test_create_mplayers_spawns_three_npcs_with_player_class_entries():
    """``create_mplayers(state, rng, 3)`` produces 3 alive mplayers with
    entry_idx values drawn from MPLAYER_ROLES."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(42))

    # Snapshot pre-spawn alive count for sanity.
    alive_before = int(jnp.sum(state.monster_ai.alive.astype(jnp.int32)))

    state2 = create_mplayers(state, jax.random.PRNGKey(7), 3)
    mai = state2.monster_ai

    alive_after = int(jnp.sum(mai.alive.astype(jnp.int32)))
    assert alive_after - alive_before == 3, (
        f"expected 3 new alive monsters, got {alive_after - alive_before}"
    )

    # Identify the new slots (the ones that became alive).
    pre_alive  = state.monster_ai.alive
    post_alive = mai.alive
    new_slots  = jnp.where(post_alive & ~pre_alive)[0]
    assert int(new_slots.shape[0]) == 3

    valid_entries = {idx for idx, _ in MPLAYER_ROLES}
    for s in new_slots.tolist():
        ei = int(mai.entry_idx[s])
        assert ei in valid_entries, f"slot {s} entry_idx={ei} not in MPLAYER_ROLES"
        assert int(mai.hp[s]) == int(mai.hp_max[s])
        assert 15 <= int(mai.hp[s]) <= 30


def test_mk_mplayer_populates_role_equipment():
    """Each of the 13 roles spawns with at least one item in its kit.

    Vendor mk_mplayer (mplayer.c lines 159-249) gives each role a class-
    typical weapon/armor/tool bundle.  This smoke verifies the deterministic
    core of the kit lands in monster_ai.inv_* arrays.
    """
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(99))

    for role_idx, (entry_idx, name) in enumerate(MPLAYER_ROLES):
        pos = jnp.array([3, 5], dtype=jnp.int16)
        s = mk_mplayer(state, jax.random.PRNGKey(100 + role_idx), role_idx, pos)
        mai = s.monster_ai
        # Find the slot we just wrote (first slot newly-alive vs original).
        new_alive = mai.alive & ~state.monster_ai.alive
        new_slots = jnp.where(new_alive)[0]
        assert int(new_slots.shape[0]) == 1
        slot = int(new_slots[0])
        # At least one inventory slot has a non-NONE category.
        cats = mai.inv_category[slot]
        nonempty = int(jnp.sum((cats != 0).astype(jnp.int32)))
        assert nonempty >= 1, (
            f"role {name} (idx {role_idx}) spawned with empty inventory"
        )
        # Item count bounded by MAX_MONSTER_INV (= 8 in MonsterAIState).
        assert nonempty <= 8


def test_create_mplayers_smoke_three_have_nonempty_inventory():
    """Spawn 3 random NPCs; each must have a non-empty inventory row.

    Per the task spec smoke test:
      spawn 3 random NPCs; check they have non-zero inventory.
    """
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(2026))
    state2 = create_mplayers(state, jax.random.PRNGKey(7), 3)
    mai = state2.monster_ai

    new_alive = mai.alive & ~state.monster_ai.alive
    new_slots = jnp.where(new_alive)[0]
    assert int(new_slots.shape[0]) == 3
    for s in new_slots.tolist():
        cats = mai.inv_category[s]
        nonempty = int(jnp.sum((cats != 0).astype(jnp.int32)))
        assert nonempty >= 1, f"new mplayer slot {s} has empty inventory"


def test_create_mplayers_jit():
    """create_mplayers traces under jit — JIT-purity smoke."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(3))

    @jax.jit
    def _go(s, k):
        return create_mplayers(s, k, 3)

    out = _go(state, jax.random.PRNGKey(9))
    new_count = int(jnp.sum(out.monster_ai.alive.astype(jnp.int32))) - int(
        jnp.sum(state.monster_ai.alive.astype(jnp.int32))
    )
    assert new_count == 3


def test_maybe_seed_astral_mplayers_edge_triggered():
    """The Astral seeder fires exactly once on (prev != ENDGAME,5) →
    (curr == ENDGAME,5) transition."""
    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(11))

    # Force the player into the Astral Plane.
    new_dungeon = state.dungeon.replace(
        current_branch=jnp.int8(ASTRAL_BRANCH),
        current_level=jnp.int8(ASTRAL_LEVEL),
    )
    state_astral = state.replace(dungeon=new_dungeon)

    alive_before = int(jnp.sum(state_astral.monster_ai.alive.astype(jnp.int32)))

    # Edge: prev was Main/Level1 (branch 0, level 1).
    state_after = maybe_seed_astral_mplayers(
        state_astral, jax.random.PRNGKey(12), prev_branch=0, prev_level=1
    )
    alive_after = int(jnp.sum(state_after.monster_ai.alive.astype(jnp.int32)))
    assert alive_after - alive_before == ASTRAL_MPLAYER_COUNT

    # No-edge: prev already on Astral → no extra spawn.
    state_after2 = maybe_seed_astral_mplayers(
        state_after, jax.random.PRNGKey(13),
        prev_branch=ASTRAL_BRANCH, prev_level=ASTRAL_LEVEL,
    )
    alive_after2 = int(jnp.sum(state_after2.monster_ai.alive.astype(jnp.int32)))
    assert alive_after2 == alive_after
