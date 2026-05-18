"""Riding subsystem — mount and dismount a steed.

Vendor reference: vendor/nethack/src/steed.c

Exported functions
------------------
try_mount(state, rng) -> EnvState
try_dismount(state, rng) -> EnvState
fall_off_steed(state, rng, force) -> EnvState

JIT-pure: no Python control-flow on traced JAX values.
"""

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.subsystems.skills import SkillId, SkillLevel

# ---------------------------------------------------------------------------
# _IS_RIDEABLE — bool[381] precomputed at module load.
#
# Horses / ponies / warhorses are the common case (indices 99, 103, 104).
# Dragons and unicorns require special handling — deferred with TODO.
#
# Vendor reference: steed.c can_ride() / can_saddle() + steeds[] symbol set
# (steed.c:8): S_QUADRUPED, S_UNICORN, S_ANGEL, S_CENTAUR, S_DRAGON, S_JABBERWOCK.
# We implement the minimum viable set: ponies + horses.
# ---------------------------------------------------------------------------
_N_MONSTERS = 381

# Indices that are unconditionally rideable (pony=99, horse=103, warhorse=104).
# Determined by: .venv/bin/python -c "from Nethax.nethax.constants.monsters import MONSTERS;
#   print([(i,m.name) for i,m in enumerate(MONSTERS) if any(x in m.name.lower()
#   for x in ['pony','horse','warhorse'])])"
# Result: [(99, 'pony'), (103, 'horse'), (104, 'warhorse')]
_RIDEABLE_INDICES = frozenset([99, 103, 104])
# TODO: dragons (e.g. baby gray dragon), unicorns — require special handling per
# steed.c:can_ride() (steeds[] symbol check, MZ_MEDIUM size gate).

def _build_is_rideable() -> jnp.ndarray:
    table = [False] * _N_MONSTERS
    for idx in _RIDEABLE_INDICES:
        table[idx] = True
    return jnp.array(table, dtype=bool)

_IS_RIDEABLE: jnp.ndarray = _build_is_rideable()

# Minimum skill level required to attempt mounting.
# Vendor steed.c:200 — knight checks P_RIDING; we require P_BASIC (1).
_RIDING_SKILL_MIN: int = int(SkillLevel.P_BASIC)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _riding_skill_level(state) -> jnp.ndarray:
    """Return player's current RIDING skill level (int32)."""
    sid = int(SkillId.RIDING)
    return state.skills.level[sid].astype(jnp.int32)


def _find_adjacent_tame_saddled_rideable(state) -> jnp.ndarray:
    """Return slot index (int32) of first eligible mount, or -1 if none.

    Eligible: alive, tame, saddled (mai.saddled[idx]==1), rideable species,
    Chebyshev distance 1 from player.

    Vendor reference: steed.c:197 — mount_steed() validates the target monster.
    """
    mai = state.monster_ai
    p_pos = state.player_pos.astype(jnp.int32)  # [2]

    # Chebyshev distance for all slots simultaneously.
    m_pos = mai.pos.astype(jnp.int32)            # [N, 2]
    dr = jnp.abs(m_pos[:, 0] - p_pos[0])
    dc = jnp.abs(m_pos[:, 1] - p_pos[1])
    cheb = jnp.maximum(dr, dc)                   # [N]

    # Species check via entry_idx into _IS_RIDEABLE table.
    e = jnp.clip(mai.entry_idx.astype(jnp.int32), 0, _N_MONSTERS - 1)
    rideable = _IS_RIDEABLE[e]                    # [N] bool

    eligible = (
        mai.alive
        & mai.tame
        & (mai.saddled == jnp.int8(1))
        & rideable
        & (cheb == jnp.int32(1))
    )                                             # [N] bool

    n = mai.alive.shape[0]
    indices = jnp.arange(n, dtype=jnp.int32)
    # Replace non-eligible slots with n (sentinel > any valid index), then take
    # the minimum.  If none are eligible, result == n → map to -1.
    masked = jnp.where(eligible, indices, jnp.int32(n))
    best = masked.min()
    result = jnp.where(best < jnp.int32(n), best, jnp.int32(-1))
    return result


# ---------------------------------------------------------------------------
# fall_off_steed
# ---------------------------------------------------------------------------

def fall_off_steed(state, rng: jax.Array, force: bool = False):
    """Apply riding accident: 1d6 HP damage, WOUNDED_LEGS +10, clear steed.

    Vendor reference: steed.c::dismount_steed (DISMOUNT_FELL branch, ~line 607)
    — losehp(rn1(10,10), ...) + set_wounded_legs(...).  We use 1d6 as the
    minimal damage model (simplified from vendor's rn1(10,10)).
    """
    dmg = jax.random.randint(rng, (), minval=1, maxval=7, dtype=jnp.int32)
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))

    # WOUNDED_LEGS += 10 (capped at int32 max).
    wl_idx = int(TimedStatus.WOUNDED_LEGS)
    old_wl = state.status.timed_statuses[wl_idx]
    new_wl = old_wl + jnp.int32(10)
    new_ts = state.status.timed_statuses.at[wl_idx].set(new_wl)

    return state.replace(
        player_hp=new_hp,
        player_steed_mid=jnp.uint32(0),
        status=state.status.replace(timed_statuses=new_ts),
    )


# ---------------------------------------------------------------------------
# try_mount
# ---------------------------------------------------------------------------

def try_mount(state, rng: jax.Array):
    """Attempt to mount an adjacent tame saddled horse.

    Checks (in order):
      1. Not already riding.
      2. An adjacent tame saddled rideable monster exists.
      3. Player has at least P_BASIC riding skill (else skip check).
      4. Success roll: rn2(100) < 5 + 5*skill_level + Dex.

    On success: player_steed_mid = slot index (entry_idx used as proxy id).
    On failure: fall_off_steed (slip damage).

    Vendor reference: steed.c:197 mount_steed() — skill/saddle checks, then
    success gate at steed.c:339 (Confusion || Fumbling || ... || level+mtame < rnd).
    We simplify to: roll < 5 + 5*skill + Dex.
    """
    already_riding = state.player_steed_mid != jnp.uint32(0)

    slot = _find_adjacent_tame_saddled_rideable(state)
    found = slot >= jnp.int32(0)

    skill_lv = _riding_skill_level(state)
    skill_ok = skill_lv >= jnp.int32(_RIDING_SKILL_MIN)

    dex = state.player_dex.astype(jnp.int32)
    threshold = jnp.int32(5) + jnp.int32(5) * skill_lv + dex

    k_roll, k_fall = jax.random.split(rng)
    roll = jax.random.randint(k_roll, (), minval=0, maxval=100, dtype=jnp.int32)
    success_roll = roll < threshold

    # Overall success: not riding, found a valid mount, skill ok, roll passes.
    success = (~already_riding) & found & skill_ok & success_roll

    # On success: set player_steed_mid to (slot+1) so 0 remains "not riding".
    # Using slot+1 as a simple unique id proxy (entry_idx-based id deferred).
    mid_on_success = (slot + jnp.int32(1)).astype(jnp.uint32)

    state_mounted = state.replace(
        player_steed_mid=jnp.where(success, mid_on_success, state.player_steed_mid)
    )

    # On failure (found mount but roll failed): fall_off_steed.
    should_fall = (~already_riding) & found & skill_ok & (~success_roll)
    state_after_fall = fall_off_steed(state_mounted, k_fall)

    # Merge: if should_fall apply fall result, else keep mounted state.
    # We do this field-by-field to stay JIT-pure.
    final_hp = jnp.where(should_fall, state_after_fall.player_hp, state_mounted.player_hp)
    final_mid = jnp.where(should_fall, state_after_fall.player_steed_mid,
                          state_mounted.player_steed_mid)
    old_wl = state_mounted.status.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]
    fall_wl = state_after_fall.status.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]
    final_wl = jnp.where(should_fall, fall_wl, old_wl)
    new_ts = state_mounted.status.timed_statuses.at[int(TimedStatus.WOUNDED_LEGS)].set(final_wl)

    return state_mounted.replace(
        player_hp=final_hp,
        player_steed_mid=final_mid,
        status=state_mounted.status.replace(timed_statuses=new_ts),
    )


# ---------------------------------------------------------------------------
# try_dismount
# ---------------------------------------------------------------------------

def try_dismount(state, rng: jax.Array):
    """Dismount the current steed.

    No-op if not riding.  On dismount, clears player_steed_mid.

    Vendor reference: steed.c:380 (u.usteed = NULL after releasing steed) and
    steed.c::dismount_steed DISMOUNT_BYCHOICE branch (~line 632).
    """
    new_mid = jnp.where(
        state.player_steed_mid != jnp.uint32(0),
        jnp.uint32(0),
        state.player_steed_mid,
    )
    return state.replace(player_steed_mid=new_mid)
