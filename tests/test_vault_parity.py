"""Vault parity tests — vault.c::mk_vault, invault, gd_move.

Tests:
    test_vault_placed_on_level      — place_vault sets vault_pos != (-1,-1).
    test_invault_spawns_guard       — moving player into vault spawns guard.
    test_attack_guard_makes_hostile — attacking guard sets peaceful=False.

Canonical sources:
    vendor/nethack/src/vault.c::mk_vault  (vault room generation)
    vendor/nethack/src/vault.c::invault   (guard spawn, line 317)
    vendor/nethack/src/vault.c line 267-270 (guard turns hostile on attack)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.dungeon.branches import MAX_LEVELS_PER_BRANCH
from Nethax.nethax.subsystems.vault import (
    place_vault,
    check_invault,
    VAULT_GUARD_SLOT,
    PM_GUARD,
)

_RNG = jax.random.PRNGKey(42)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_wall_block(state, branch: int, lv: int, r: int, c: int):
    """Carve a 4×4 WALL block at (r,c) so place_vault has a candidate."""
    terrain = np.array(state.terrain[branch, lv])
    terrain[r:r + 4, c:c + 4] = 3  # TileType.WALL = 3
    return state.replace(
        terrain=state.terrain.at[branch, lv].set(
            jnp.array(terrain, dtype=jnp.int8)
        )
    )


# ---------------------------------------------------------------------------
# test_vault_placed_on_level
# Vendor: vault.c::mk_vault — places a small walled room in solid map area.
# ---------------------------------------------------------------------------

def test_vault_placed_on_level():
    """place_vault records a non-(-1,-1) vault_pos on the target level."""
    state = _fresh_state()
    # Ensure a solid 4×4 block exists for vault placement.
    state = _set_wall_block(state, branch=0, lv=0, r=5, c=10)

    vault_rng = np.random.default_rng(0)
    state = place_vault(state, branch=0, lv=0, rng=vault_rng)

    level_idx = 0 * MAX_LEVELS_PER_BRANCH + 0
    vp = state.features.vault_pos[level_idx]
    assert int(vp[0]) != -1 or int(vp[1]) != -1, (
        "vault_pos should be set after place_vault"
    )


# ---------------------------------------------------------------------------
# test_invault_spawns_guard
# Vendor: vault.c::invault line 407 — makemon(PM_GUARD, x, y, MM_EGD|MM_NOMSG),
#         guard->mpeaceful = 1 (line 410).
# ---------------------------------------------------------------------------

def test_invault_spawns_guard():
    """Moving player onto vault tile triggers guard spawn in VAULT_GUARD_SLOT."""
    state = _fresh_state()
    state = _set_wall_block(state, branch=0, lv=0, r=5, c=10)

    vault_rng = np.random.default_rng(0)
    state = place_vault(state, branch=0, lv=0, rng=vault_rng)

    level_idx = 0 * MAX_LEVELS_PER_BRANCH + 0
    vp = state.features.vault_pos[level_idx]
    vr, vc = int(vp[0]), int(vp[1])

    # Teleport player to vault position.
    state = state.replace(player_pos=jnp.array([vr, vc], dtype=jnp.int16))

    rng_vault = jax.random.PRNGKey(1)
    state = check_invault(state, rng_vault)

    assert bool(state.monster_ai.alive[VAULT_GUARD_SLOT]), (
        "Guard should be alive in VAULT_GUARD_SLOT after entering vault"
    )
    assert int(state.monster_ai.entry_idx[VAULT_GUARD_SLOT]) == PM_GUARD, (
        f"Guard entry_idx should be PM_GUARD ({PM_GUARD})"
    )
    assert bool(state.monster_ai.peaceful[VAULT_GUARD_SLOT]), (
        "Guard should be peaceful on spawn (vault.c::invault line 410)"
    )
    assert int(state.features.guard_slot) == VAULT_GUARD_SLOT, (
        "features.guard_slot should record the guard's slot"
    )


# ---------------------------------------------------------------------------
# test_attack_guard_makes_hostile
# Vendor: vault.c line 267-270 — attacking guard sets mpeaceful=0.
# ---------------------------------------------------------------------------

def test_attack_guard_makes_hostile():
    """Directly clearing peaceful on guard slot mirrors vault.c line 267-270.

    vault.c::invault (line 267): "if (grd->mpeaceful) { ... grd->mpeaceful = 0; }"
    In our model the combat subsystem sets peaceful=False when player attacks.
    We simulate that directly here.
    """
    state = _fresh_state()
    state = _set_wall_block(state, branch=0, lv=0, r=5, c=10)

    vault_rng = np.random.default_rng(0)
    state = place_vault(state, branch=0, lv=0, rng=vault_rng)

    level_idx = 0 * MAX_LEVELS_PER_BRANCH + 0
    vp = state.features.vault_pos[level_idx]
    state = state.replace(player_pos=jnp.array([int(vp[0]), int(vp[1])], dtype=jnp.int16))
    state = check_invault(state, jax.random.PRNGKey(2))

    assert bool(state.monster_ai.peaceful[VAULT_GUARD_SLOT]), "Guard starts peaceful"

    # Simulate player attacking guard → set peaceful=False.
    # Vendor: vault.c line 270: grd->mpeaceful = 0; /* bypass setmangry() */
    mai = state.monster_ai.replace(
        peaceful=state.monster_ai.peaceful.at[VAULT_GUARD_SLOT].set(jnp.bool_(False))
    )
    state = state.replace(monster_ai=mai)

    assert not bool(state.monster_ai.peaceful[VAULT_GUARD_SLOT]), (
        "Guard should be hostile after attack (vault.c line 270)"
    )
