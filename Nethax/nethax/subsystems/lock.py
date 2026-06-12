"""Lock-picking subsystem — pick_lock / picklock / forcelock.

Canonical source: ``vendor/nethack/src/lock.c``.

Chance table (vendor lock.c:520-534, 632-644).  ``ch`` is the per-turn
unlock chance used by ``picklock``'s ``rn2(100) >= chance`` poll:

    Box / chest:
      CREDIT_CARD   :  ACURR(A_DEX) + 20 * Role_if(PM_ROGUE)   (lock.c:522)
      LOCK_PICK     : 4*ACURR(A_DEX) + 25 * Role_if(PM_ROGUE)  (lock.c:525)
      SKELETON_KEY  : 75 + ACURR(A_DEX)                        (lock.c:528)

    Door:
      CREDIT_CARD   : 2*ACURR(A_DEX) + 20 * Role_if(PM_ROGUE)  (lock.c:634)
      LOCK_PICK     : 3*ACURR(A_DEX) + 30 * Role_if(PM_ROGUE)  (lock.c:637)
      SKELETON_KEY  : 70 + ACURR(A_DEX)                        (lock.c:640)

    Modifiers:
      otmp->cursed  : ch /= 2                                  (lock.c:533-534)
      magic_key + trapped target : ch += 20 (less effort next time,
                                              lock.c:107)

``picklock`` occupation tick (lock.c:68-159):
  - per turn, ``rn2(100) >= chance`` → still busy
  - ``usedtime >= 50`` → give up
  - on success: door D_LOCKED → D_CLOSED; container.olocked toggles;
    trapped chest fires ``chest_trap`` (lock.c:155).

``forcelock`` blade-erosion (lock.c:228-240):
  - ``rn2(1000 - spe) > 992 - greatest_erosion*10`` → weapon breaks
  - uses uwep enchant level (``spe``) and the larger of oeroded /
    oeroded2 (``greatest_erosion``).

Magic key (Master Key of Thievery, vendor artifact.c:2775-2786):
  - rogue: non-cursed suffices
  - non-rogue: must be blessed
  - grants +20 chance bump on trapped doors / boxes and the option to
    disarm (we model the +20 only; disarm UI is omitted).

Public API
----------
``door_chance(tool_tid, dex, role_is_rogue, cursed)``
``box_chance(tool_tid, dex, role_is_rogue, cursed)``
``start_pick_lock_door(state, pos, tool_tid, tool_cursed, magic_key)``
``start_pick_lock_box(state, container_idx, tool_tid, tool_cursed, magic_key)``
``tick_pick_lock(state, rng)`` — per-turn poll fired by occupation.tick.
``force_lock(state, rng, spe, greatest_erosion, cursed)``
    — blade-erosion roll; returns ``(new_state, weapon_broke)``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Tool otyps (vendor/nethack/include/objects.h via apply_tools.py).
SKELETON_KEY_TYPE_ID = 196
LOCK_PICK_TYPE_ID    = 197
CREDIT_CARD_TYPE_ID  = 198

# Cap chance at 99 so rn2(100) still has a non-zero failure tail (mirrors
# vendor's implicit cap from the rn2(100) >= chance comparison).
_CH_CAP = 99

# Vendor lock.c:92 — give up after 50 ticks.
PICK_LOCK_GIVE_UP_TURNS = 50


# ---------------------------------------------------------------------------
# Pure chance helpers
# ---------------------------------------------------------------------------

def box_chance(
    tool_tid: jnp.ndarray,
    dex: jnp.ndarray,
    role_is_rogue: jnp.ndarray,
    cursed: jnp.ndarray,
) -> jnp.ndarray:
    """Vendor lock.c:520-534 — chest/box ``ch`` setting.

    All inputs are JAX arrays / scalars (JIT-pure).
    """
    tool = tool_tid.astype(jnp.int32)
    dex  = dex.astype(jnp.int32)
    rogue = role_is_rogue.astype(jnp.int32)

    ch_card = dex + jnp.int32(20) * rogue                 # lock.c:522
    ch_pick = jnp.int32(4) * dex + jnp.int32(25) * rogue  # lock.c:525
    ch_key  = jnp.int32(75) + dex                         # lock.c:528

    ch = jnp.where(tool == jnp.int32(SKELETON_KEY_TYPE_ID), ch_key,
         jnp.where(tool == jnp.int32(LOCK_PICK_TYPE_ID),    ch_pick,
         jnp.where(tool == jnp.int32(CREDIT_CARD_TYPE_ID),  ch_card,
                   jnp.int32(0))))

    # Cursed tool halves chance (lock.c:533-534).
    ch = jnp.where(cursed.astype(jnp.bool_), ch // jnp.int32(2), ch)
    return jnp.minimum(ch, jnp.int32(_CH_CAP))


def door_chance(
    tool_tid: jnp.ndarray,
    dex: jnp.ndarray,
    role_is_rogue: jnp.ndarray,
    cursed: jnp.ndarray,
) -> jnp.ndarray:
    """Vendor lock.c:632-644 — door ``ch`` setting."""
    tool = tool_tid.astype(jnp.int32)
    dex  = dex.astype(jnp.int32)
    rogue = role_is_rogue.astype(jnp.int32)

    ch_card = jnp.int32(2) * dex + jnp.int32(20) * rogue  # lock.c:634
    ch_pick = jnp.int32(3) * dex + jnp.int32(30) * rogue  # lock.c:637
    ch_key  = jnp.int32(70) + dex                         # lock.c:640

    ch = jnp.where(tool == jnp.int32(SKELETON_KEY_TYPE_ID), ch_key,
         jnp.where(tool == jnp.int32(LOCK_PICK_TYPE_ID),    ch_pick,
         jnp.where(tool == jnp.int32(CREDIT_CARD_TYPE_ID),  ch_card,
                   jnp.int32(0))))

    ch = jnp.where(cursed.astype(jnp.bool_), ch // jnp.int32(2), ch)
    return jnp.minimum(ch, jnp.int32(_CH_CAP))


def is_magic_key(
    tool_artifact_idx: jnp.ndarray,
    role_is_rogue: jnp.ndarray,
    blessed: jnp.ndarray,
    cursed: jnp.ndarray,
) -> jnp.ndarray:
    """Vendor artifact.c:2775-2786 — is this object the Master Key of
    Thievery in a state that makes it "magic" for the wielder?

    - tool_artifact_idx : roles.ART_MASTER_KEY_OF_THIEVERY (== 8) when
                          the item is that artifact, else any other value.
    - rogue + non-cursed  → magic
    - non-rogue + blessed → magic
    """
    from Nethax.nethax.constants.roles import ART_MASTER_KEY_OF_THIEVERY
    is_master_key = tool_artifact_idx.astype(jnp.int32) == jnp.int32(
        int(ART_MASTER_KEY_OF_THIEVERY))
    rogue = role_is_rogue.astype(jnp.bool_)
    # Rogue: not cursed.  Non-rogue: blessed.
    rogue_ok    = rogue & ~cursed.astype(jnp.bool_)
    nonrogue_ok = (~rogue) & blessed.astype(jnp.bool_)
    return is_master_key & (rogue_ok | nonrogue_ok)


# ---------------------------------------------------------------------------
# Occupation-target encoding.
#
# The shared occupation_target (int32 scalar) stores either:
#   - container_idx (small positive int) for PICK_LOCK_BOX, or
#   - packed (level, row, col) flat index for PICK_LOCK_DOOR.
#
# The kind discriminates them (see occupation.OccupationKind).
# Chance + magic_key are stored in two new int8 fields on EnvState
# (``pick_lock_chance``, ``pick_lock_magic_key``) added separately.
# ---------------------------------------------------------------------------

def encode_door_target(
    flat_lv: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray, map_w: int,
) -> jnp.ndarray:
    """Pack (flat_lv, row, col) into a single int32 for occupation_target.

    Layout (LSB → MSB): col [0..map_w), row * map_w, lv * MAP_H * map_w.
    """
    from Nethax.nethax.dungeon.branches import MAP_H
    flat_lv = flat_lv.astype(jnp.int32)
    row = row.astype(jnp.int32)
    col = col.astype(jnp.int32)
    stride_per_lv = jnp.int32(MAP_H * map_w)
    return flat_lv * stride_per_lv + row * jnp.int32(map_w) + col


def decode_door_target(
    target: jnp.ndarray, map_w: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Inverse of ``encode_door_target``."""
    from Nethax.nethax.dungeon.branches import MAP_H
    target = target.astype(jnp.int32)
    stride_per_lv = jnp.int32(MAP_H * map_w)
    flat_lv = target // stride_per_lv
    rem = target - flat_lv * stride_per_lv
    row = rem // jnp.int32(map_w)
    col = rem - row * jnp.int32(map_w)
    return flat_lv, row, col


# ---------------------------------------------------------------------------
# Occupation start helpers.
# ---------------------------------------------------------------------------

def start_pick_lock_door(
    state,
    flat_lv: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    tool_tid: jnp.ndarray,
    tool_cursed: jnp.ndarray,
    magic_key: jnp.ndarray,
):
    """Begin a PICK_LOCK_DOOR occupation (vendor lock.c:650-654).

    Computes per-turn chance now and stores it in
    ``state.pick_lock_chance``.  ``occupation_remaining`` is set to
    ``PICK_LOCK_GIVE_UP_TURNS``; each tick decrements it (vendor uses
    ``usedtime++ >= 50``; equivalent here as a countdown from 50).
    """
    from Nethax.nethax.subsystems.occupation import OccupationKind
    from Nethax.nethax.constants.roles import Role as _Role
    map_w = state.terrain.shape[3]

    role_is_rogue = state.player_role.astype(jnp.int32) == jnp.int32(int(_Role.ROGUE))
    ch = door_chance(
        tool_tid,
        state.player_dex.astype(jnp.int32),
        role_is_rogue,
        tool_cursed,
    )
    target = encode_door_target(flat_lv, row, col, map_w)
    return state.replace(
        occupation_kind=jnp.int8(int(OccupationKind.PICK_LOCK_DOOR)),
        occupation_target=target.astype(jnp.int32),
        occupation_remaining=jnp.int8(PICK_LOCK_GIVE_UP_TURNS),
        pick_lock_chance=jnp.clip(ch, 0, 127).astype(jnp.int8),
        pick_lock_magic_key=magic_key.astype(jnp.bool_),
    )


def start_pick_lock_box(
    state,
    container_idx: jnp.ndarray,
    tool_tid: jnp.ndarray,
    tool_cursed: jnp.ndarray,
    magic_key: jnp.ndarray,
):
    """Begin a PICK_LOCK_BOX occupation."""
    from Nethax.nethax.subsystems.occupation import OccupationKind
    from Nethax.nethax.constants.roles import Role as _Role
    role_is_rogue = state.player_role.astype(jnp.int32) == jnp.int32(int(_Role.ROGUE))
    ch = box_chance(
        tool_tid,
        state.player_dex.astype(jnp.int32),
        role_is_rogue,
        tool_cursed,
    )
    return state.replace(
        occupation_kind=jnp.int8(int(OccupationKind.PICK_LOCK_BOX)),
        occupation_target=container_idx.astype(jnp.int32),
        occupation_remaining=jnp.int8(PICK_LOCK_GIVE_UP_TURNS),
        pick_lock_chance=jnp.clip(ch, 0, 127).astype(jnp.int8),
        pick_lock_magic_key=magic_key.astype(jnp.bool_),
    )


# ---------------------------------------------------------------------------
# Per-turn pick-lock tick (vendor lock.c:68-159).
# ---------------------------------------------------------------------------

def _tick_door(state, rng):
    """Per-turn picklock callback for a door target."""
    from Nethax.nethax.subsystems.features import DoorState

    map_w = state.terrain.shape[3]
    flat_lv, row, col = decode_door_target(state.occupation_target, map_w)

    # vendor lock.c:98 — if (rn2(100) >= chance) return 1 (still busy)
    rng_roll, _ = jax.random.split(rng)
    roll = jax.random.randint(rng_roll, shape=(), minval=0, maxval=100)
    chance = state.pick_lock_chance.astype(jnp.int32)
    succeeded = roll < chance

    # Magic-key trap-detect bump (lock.c:104-107): on a trapped door, the
    # chance increases by 20 for next turn.  We can't easily inspect door
    # trap state here (no per-door trapped flag in features yet) — skip
    # the +20 for doors and leave a TODO; we DO apply it for boxes (which
    # have is_trapped) in _tick_box.

    n_levels = state.features.door_state.shape[0]
    h = state.features.door_state.shape[1]
    w = state.features.door_state.shape[2]
    in_bounds = ((flat_lv >= 0) & (flat_lv < jnp.int32(n_levels))
                 & (row >= 0) & (row < jnp.int32(h))
                 & (col >= 0) & (col < jnp.int32(w)))
    safe_lv  = jnp.clip(flat_lv, 0, n_levels - 1)
    safe_row = jnp.clip(row,     0, h - 1)
    safe_col = jnp.clip(col,     0, w - 1)
    current = state.features.door_state[safe_lv, safe_row, safe_col].astype(jnp.int32)
    is_locked = (current == jnp.int32(DoorState.LOCKED)) & in_bounds
    do_unlock = succeeded & is_locked
    new_val = jnp.where(do_unlock, jnp.int32(DoorState.CLOSED), current).astype(jnp.int8)
    new_door_state = state.features.door_state.at[safe_lv, safe_row, safe_col].set(new_val)
    return state.replace(features=state.features.replace(door_state=new_door_state)), succeeded


def _tick_box(state, rng):
    """Per-turn picklock callback for a container target."""
    cs = state.containers
    c_idx = jnp.clip(state.occupation_target.astype(jnp.int32),
                     0, cs.is_locked.shape[0] - 1)

    rng_roll, rng_after = jax.random.split(rng)
    roll = jax.random.randint(rng_roll, shape=(), minval=0, maxval=100)
    chance = state.pick_lock_chance.astype(jnp.int32)
    succeeded = roll < chance

    # Magic-key + trapped chest: bump chance by 20 for next turn
    # (vendor lock.c:107).
    is_trapped = cs.is_trapped[c_idx]
    bump = state.pick_lock_magic_key & is_trapped
    new_chance = jnp.where(
        bump,
        jnp.minimum(chance + jnp.int32(20), jnp.int32(_CH_CAP)),
        chance,
    ).astype(jnp.int8)

    # On success: clear is_locked.  Fire chest_trap if otrapped
    # (vendor lock.c:154-155).
    was_locked = cs.is_locked[c_idx]
    do_unlock = succeeded & was_locked
    new_is_locked = cs.is_locked.at[c_idx].set(
        jnp.where(do_unlock, jnp.bool_(False), cs.is_locked[c_idx])
    )
    new_cs = cs.replace(is_locked=new_is_locked)
    state = state.replace(containers=new_cs, pick_lock_chance=new_chance)

    # Trap firing: lock.c:155 — chest_trap fires on otrapped boxes when
    # the player just unlocked them.  Reuse containers.fire_container_trap.
    from Nethax.nethax.subsystems.containers import fire_container_trap

    def _fire_trap(s):
        s2, _ = fire_container_trap(s, c_idx)
        return s2

    state = jax.tree_util.tree_map(
        lambda f, o: jnp.where(do_unlock & is_trapped, f, o), _fire_trap(state), state,
    )
    return state, succeeded


def tick_pick_lock(state, rng):
    """Per-turn lock-picking poll (vendor lock.c::picklock lines 92-159).

    Dispatches to _tick_door or _tick_box based on occupation_kind.
    Returns ``(new_state, finished)`` — ``finished`` is True iff the
    occupation should clear this turn (either succeeded or used up all
    50 turns).
    """
    from Nethax.nethax.subsystems.occupation import OccupationKind

    kind = state.occupation_kind.astype(jnp.int32)
    is_door = kind == jnp.int32(int(OccupationKind.PICK_LOCK_DOOR))
    is_box  = kind == jnp.int32(int(OccupationKind.PICK_LOCK_BOX))
    active  = is_door | is_box

    # Both branches must run under jax.lax.cond — only one fires.
    state_door, succ_door = _tick_door(state, rng)
    state_box,  succ_box  = _tick_box(state, rng)
    new_state = jax.tree_util.tree_map(
        lambda d, b, x: jnp.where(is_door, d, jnp.where(is_box, b, x)),
        state_door, state_box, state,
    )
    succeeded = jnp.where(is_door, succ_door,
                jnp.where(is_box,  succ_box,
                          jnp.bool_(False)))

    # Vendor: usedtime++ >= 50 → give up.  We store remaining countdown,
    # so finished_by_timeout = (remaining <= 1) at entry (i.e. this is
    # the last tick).
    finished_by_timeout = new_state.occupation_remaining.astype(jnp.int32) <= jnp.int32(1)
    finished = active & (succeeded | finished_by_timeout)
    return new_state, finished


# ---------------------------------------------------------------------------
# Force-lock blade erosion (vendor lock.c:216-256).
# ---------------------------------------------------------------------------

def force_lock_blade_breaks(
    rng,
    spe: jnp.ndarray,
    greatest_erosion: jnp.ndarray,
    cursed: jnp.ndarray,
) -> jnp.ndarray:
    """Vendor lock.c:228-240 blade-erosion roll.

    Weapon breaks iff
       rn2(1000 - spe) > 992 - erosion*10  AND  NOT cursed
    (cursed weapons survive the attempt — they're already protected by
    a different code path; vendor `!uwep->cursed` guard).

    For a +0 weapon, P(survive an unsuccessful attempt) = (.992)^1 ≈ .992,
    so 50 attempts ≈ (.992)^50 ≈ .67 (vendor comment lines 231-233).

    Returns a bool scalar.
    """
    spe = spe.astype(jnp.int32)
    erosion = greatest_erosion.astype(jnp.int32)
    cursed_b = cursed.astype(jnp.bool_)

    upper = jnp.maximum(jnp.int32(1000) - spe, jnp.int32(1))
    roll = jax.random.randint(rng, shape=(), minval=0, maxval=upper)
    threshold = jnp.int32(992) - erosion * jnp.int32(10)
    return (roll > threshold) & (~cursed_b)
