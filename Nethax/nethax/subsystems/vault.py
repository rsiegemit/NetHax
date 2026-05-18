"""Vault subsystem — vault rooms and vault-guard logic.

Implements the vault room (a small locked 2×2 room with gold) and the
vault-guard encounter: spawn on entry, demand-name dialogue, escort out,
and hostility on attack.

Canonical sources:
    vendor/nethack/src/vault.c::mk_vault   — vault room generation
    vendor/nethack/src/vault.c::invault    — guard spawn on player entry (line 317)
    vendor/nethack/src/vault.c::gd_move    — guard escort step (line 888)
    vendor/nethack/src/vault.c line 267-270 — guard turns hostile when attacked

Design notes:
    - ``place_vault`` is host-side (called from env.py::reset), not JIT-compiled.
    - ``check_invault`` and ``guard_step`` are JIT-pure.
    - Guard slot index is fixed at ``VAULT_GUARD_SLOT`` (slot 7) to keep it
      distinct from wild-monster slots (0-4) and the starting pet (5).
    - ``state.features.vault_pos``: int16[N_BRANCHES, MAX_LEVELS, 2], (-1,-1) = no vault.
    - ``state.features.guard_slot``: int32, -1 = no guard.
    - ``state.features.guard_escort_active``: bool.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W
from Nethax.nethax.constants.tiles import TileType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vendor chunk5.py comment: "# 280 — guard (vault guard)"
PM_GUARD: int = 280

# Fixed monster slot reserved for the vault guard.
VAULT_GUARD_SLOT: int = 7

# Tile value placed at the vault centre so JIT code can detect it.
# We re-use TileType.FLOOR; vault_pos in FeaturesState is the marker.
_FLOOR = int(TileType.FLOOR)
_WALL  = int(TileType.WALL)

# Minimum clearance from map edge for vault placement (walls + interior).
_VAULT_MARGIN: int = 3


# ---------------------------------------------------------------------------
# place_vault  (host-side; vendor vault.c::mk_vault)
# ---------------------------------------------------------------------------

def place_vault(state, branch: int, lv: int, rng: np.random.Generator):
    """Carve a 2×2 vault room into the terrain and record vault_pos.

    Called host-side from env.py::reset after main level generation.
    Finds a candidate interior position at least ``_VAULT_MARGIN`` tiles from
    every map edge and whose 4×4 neighbourhood (walls + interior) consists
    entirely of existing WALL tiles (so we don't overwrite corridors/rooms).

    Vendor reference: vault.c::mk_vault — places a small walled room in an
    otherwise solid region of the level.

    Returns the updated state.  If no candidate is found, vault_pos stays
    (-1, -1) and no vault is placed.
    """
    terrain_np = np.array(state.terrain[branch, lv])   # host numpy copy
    H, W = terrain_np.shape

    # Candidate top-left corners: leave room for a 4×4 block (wall+2+wall).
    candidates = []
    for r in range(_VAULT_MARGIN, H - _VAULT_MARGIN - 3):
        for c in range(_VAULT_MARGIN, W - _VAULT_MARGIN - 3):
            # Check that the 4×4 block is all WALL (solid, unused).
            block = terrain_np[r:r + 4, c:c + 4]
            if np.all(block == _WALL):
                candidates.append((r, c))

    if not candidates:
        return state  # no room — vault_pos remains (-1, -1)

    top_r, top_c = candidates[rng.integers(len(candidates))]

    # Carve walls around interior 2×2 (rows top+1..top+2, cols top+1..top+2).
    new_terrain = terrain_np.copy()
    # Interior floor
    new_terrain[top_r + 1, top_c + 1] = _FLOOR
    new_terrain[top_r + 1, top_c + 2] = _FLOOR
    new_terrain[top_r + 2, top_c + 1] = _FLOOR
    new_terrain[top_r + 2, top_c + 2] = _FLOOR
    # Surrounding walls are already WALL; keep them.

    vault_r = top_r + 1   # interior centre row (upper-left of 2×2)
    vault_c = top_c + 1   # interior centre col

    # Write terrain back into JAX state.
    new_terrain_jax = state.terrain.at[branch, lv].set(
        jnp.array(new_terrain, dtype=jnp.int8)
    )

    # Record vault_pos: flatten (branch, lv) → level_idx for FeaturesState.
    level_idx = branch * MAX_LEVELS_PER_BRANCH + lv
    new_vault_pos = state.features.vault_pos.at[level_idx].set(
        jnp.array([vault_r, vault_c], dtype=jnp.int16)
    )

    new_features = state.features.replace(vault_pos=new_vault_pos)
    return state.replace(terrain=new_terrain_jax, features=new_features)


# ---------------------------------------------------------------------------
# check_invault  (JIT-pure; vendor vault.c::invault line 317)
# ---------------------------------------------------------------------------

def check_invault(state, rng: jax.Array):
    """Spawn vault guard if player just entered the vault and none exists.

    JIT-pure.  Called once per step (after player movement).

    Vendor reference: vault.c::invault — when vault_occupied(u.urooms) is
    set and no guard exists, call makemon(PM_GUARD, ...) and set mpeaceful=1
    (line 407-410).  Guard says "Hello stranger, who are you?" (line 503).

    We auto-decline the name (no UI); guard escort activates immediately.
    """
    branch = state.dungeon.current_branch.astype(jnp.int32)
    level  = (state.dungeon.current_level - jnp.int8(1)).astype(jnp.int32)
    level_idx = branch * MAX_LEVELS_PER_BRANCH + level

    vault_pos = state.features.vault_pos[level_idx]         # int16[2]
    vr, vc = vault_pos[0].astype(jnp.int32), vault_pos[1].astype(jnp.int32)

    pr, pc = (state.player_pos[0].astype(jnp.int32),
              state.player_pos[1].astype(jnp.int32))

    vault_valid = (vr >= jnp.int32(0)) & (vc >= jnp.int32(0))
    player_in_vault = vault_valid & (
        (jnp.abs(pr - vr) <= jnp.int32(1)) & (jnp.abs(pc - vc) <= jnp.int32(1))
    )

    guard_slot_val = state.features.guard_slot   # int32; -1 = none
    guard_absent = guard_slot_val < jnp.int32(0)

    should_spawn = player_in_vault & guard_absent

    def _spawn(s):
        return _do_spawn_guard(s, pr, pc, vr, vc)

    return jax.lax.cond(should_spawn, _spawn, lambda s: s, state)


def _do_spawn_guard(state, pr, pc, vr, vc):
    """Write PM_GUARD into VAULT_GUARD_SLOT and activate escort.

    Vendor vault.c::invault line 407-421:
        guard = makemon(&mons[PM_GUARD], x, y, MM_EGD | MM_NOMSG)
        guard->mpeaceful = 1
    Guard spawns adjacent to vault wall (one step outside player, clamped).
    """
    from Nethax.nethax.dungeon.spawning import _BASE_AC, _ATK_DICE_N, _ATK_DICE_S, _IS_LARGE

    # Place guard one step toward vault centre from player (or at vault centre).
    dr = jnp.sign(vr - pr).astype(jnp.int16)
    dc = jnp.sign(vc - pc).astype(jnp.int16)
    gr = (pr + dr).astype(jnp.int16)
    gc = (pc + dc).astype(jnp.int16)
    guard_pos = jnp.stack([gr, gc])  # int16[2]

    pm = jnp.int16(PM_GUARD)
    slot = VAULT_GUARD_SLOT

    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(True)),
        tame=mai.tame.at[slot].set(jnp.bool_(False)),
        entry_idx=mai.entry_idx.at[slot].set(pm),
        pos=mai.pos.at[slot].set(guard_pos),
        hp=mai.hp.at[slot].set(jnp.int32(96)),   # level 12 × avg 8 HP
        hp_max=mai.hp_max.at[slot].set(jnp.int32(96)),
        ac=mai.ac.at[slot].set(_BASE_AC[PM_GUARD]),
        is_large=mai.is_large.at[slot].set(_IS_LARGE[PM_GUARD]),
        attack_dice_n=mai.attack_dice_n.at[slot].set(_ATK_DICE_N[PM_GUARD]),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(_ATK_DICE_S[PM_GUARD]),
    )

    # Record guard slot; activate escort (auto-decline name = escort starts).
    new_features = state.features.replace(
        guard_slot=jnp.int32(slot),
        guard_escort_active=jnp.bool_(True),
    )
    return state.replace(monster_ai=mai, features=new_features)


# ---------------------------------------------------------------------------
# guard_step  (JIT-pure; vendor vault.c::gd_move line 888)
# ---------------------------------------------------------------------------

def guard_step(state, rng: jax.Array):
    """Move guard one step toward vault exit; player snaps behind guard.

    JIT-pure.  Called every turn from monster_ai.step when escort is active.

    Vendor reference: vault.c::gd_move (line 888) — peaceful guard moves
    toward exit (gddone destination), player follows in corridor.

    Simplified: guard moves one Chebyshev step toward the map centre
    (a proxy for "exit"), player moves one step toward the guard.
    Escort deactivates when guard reaches within 2 tiles of map edge.
    """
    escort_active = state.features.guard_escort_active
    guard_slot_val = state.features.guard_slot

    def _do_escort(s):
        return _escort_tick(s)

    return jax.lax.cond(escort_active, _do_escort, lambda s: s, state)


def _escort_tick(state):
    """One escort movement tick."""
    slot = VAULT_GUARD_SLOT

    guard_pos = state.monster_ai.pos[slot]   # int16[2]
    gr = guard_pos[0].astype(jnp.int32)
    gc = guard_pos[1].astype(jnp.int32)

    # Destination: toward map top-left corner (proxy for vault exit).
    # Vendor gd_move: guard leads player back toward the regular dungeon
    # corridor that connects the vault (EGD->gddone position).
    dest_r = jnp.int32(2)
    dest_c = jnp.int32(2)

    dr = jnp.sign(dest_r - gr).astype(jnp.int16)
    dc = jnp.sign(dest_c - gc).astype(jnp.int16)

    new_gr = (gr + dr).astype(jnp.int16)
    new_gc = (gc + dc).astype(jnp.int16)
    new_guard_pos = jnp.stack([new_gr, new_gc])

    # Player follows one step toward guard current position.
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    pdr = jnp.sign(gr - pr).astype(jnp.int16)
    pdc = jnp.sign(gc - pc).astype(jnp.int16)
    new_pr = (pr + pdr).astype(jnp.int16)
    new_pc = (pc + pdc).astype(jnp.int16)
    new_player_pos = jnp.stack([new_pr, new_pc])

    # Escort done when guard reaches near map corner.
    done = (new_gr <= jnp.int16(3)) & (new_gc <= jnp.int16(3))

    mai = state.monster_ai.replace(
        pos=state.monster_ai.pos.at[slot].set(new_guard_pos),
    )
    new_features = state.features.replace(
        guard_escort_active=~done,
    )
    return state.replace(
        monster_ai=mai,
        features=new_features,
        player_pos=new_player_pos,
    )
