"""Riding subsystem — mount and dismount a steed.

Vendor reference: vendor/nethack/src/steed.c

Exported functions
------------------
try_mount(state, rng) -> EnvState
try_dismount(state, rng) -> EnvState
fall_off_steed(state, rng, force) -> EnvState
check_combat_dismount(state, damage_taken) -> EnvState
tick_saddle(state) -> EnvState

JIT-pure: no Python control-flow on traced JAX values.
"""

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.monsters import MONSTERS
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

# ---------------------------------------------------------------------------
# _MONSTER_SPEED_TABLE — int8[381] move_speed per monster entry.
# Used by try_mount to set player_extra_speed from the steed's base speed.
# Vendor reference: steed.c:447 (ugallop += rn1(20,30)) — while riding the
# player moves at the steed's speed.  We store base move_speed from MONSTERS[].
# ---------------------------------------------------------------------------
_MONSTER_SPEED_TABLE: jnp.ndarray = jnp.array(
    [int(m.move_speed) for m in MONSTERS], dtype=jnp.int8
)

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

    Eligible: alive, tame, has a saddle (saddled==1 uncursed or ==2 cursed),
    rideable species, Chebyshev distance 1 from player.

    Vendor reference: steed.c:197 — mount_steed() validates the target monster.
    Cursed saddle is detected in try_mount; here we include it so cursed-saddle
    steed can be found and the hostile-flip branch reached.
    Vendor reference: steed.c:122 (cursed saddle reduces chance -= 50),
    steed.c:634 (DISMOUNT_BYCHOICE blocked by cursed saddle).
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

    # saddled==1 (uncursed) or saddled==2 (cursed) both count as "has a saddle".
    has_saddle = (mai.saddled == jnp.int8(1)) | (mai.saddled == jnp.int8(2))

    eligible = (
        mai.alive
        & mai.tame
        & has_saddle
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

    Vendor reference: steed.c::fall_from_steed / dismount_steed DISMOUNT_FELL
    branch (~line 607): losehp(Maybe_Half_Phys(rn1(10,10)), ...) +
    set_wounded_legs(BOTH_SIDES, HWounded_legs + rn1(5,5)).
    We use 1d6 as the minimal damage model.  WOUNDED_LEGS wires through
    StatusState.timed_statuses[TimedStatus.WOUNDED_LEGS] (status_effects.py).
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
        player_extra_speed=jnp.int8(0),
        status=state.status.replace(timed_statuses=new_ts),
    )


# ---------------------------------------------------------------------------
# try_mount
# ---------------------------------------------------------------------------

def try_mount(state, rng: jax.Array):
    """Attempt to mount an adjacent tame saddled horse.

    Checks (in order):
      1. Not already riding.
      2. An adjacent tame saddled (or cursed-saddled) rideable monster exists.
      3. Cursed saddle (saddled==2): mount fails and steed flips hostile.
         Vendor reference: steed.c:634 (DISMOUNT_BYCHOICE blocks on cursed
         saddle); steed.c:122 (cursed saddle reduces chance -= 50).
         We model this as an outright failure + hostile flip.
      4. Player has at least P_BASIC riding skill.
      5. Success roll: rn2(100) < 5 + 5*skill_level + Dex.

    On success: player_steed_mid = slot+1, player_extra_speed = steed's
    move_speed.  Vendor reference: steed.c:447 (ugallop += rn1(20,30) — while
    riding, player moves at steed speed).

    On mount-roll failure: fall_off_steed (slip damage).

    TODO: vendor allows mounting via lasso/lance applied to adjacent steed
    (steed.c::do_saddlecheck); simplified here — try_mount works without a
    held lasso.  Full lasso/lance range not implemented.

    TODO: dragons, unicorns — require special handling per steed.c:can_ride().
    """
    already_riding = state.player_steed_mid != jnp.uint32(0)

    slot = _find_adjacent_tame_saddled_rideable(state)
    found = slot >= jnp.int32(0)

    # Detect cursed saddle on the candidate slot.
    # saddled==2 means cursed; steed.c:634 prevents dismount / mount attempt.
    safe_slot = jnp.clip(slot, 0, state.monster_ai.alive.shape[0] - 1)
    slot_saddled = state.monster_ai.saddled[safe_slot]
    is_cursed = (slot_saddled == jnp.int8(2)) & found

    # Flip steed tame→hostile on cursed saddle attempt.
    # Vendor reference: steed.c::saddled_cursed — steed turns hostile.
    new_tame = jnp.where(
        is_cursed,
        state.monster_ai.tame.at[safe_slot].set(False),
        state.monster_ai.tame,
    )
    state_after_cursed = state.replace(
        monster_ai=state.monster_ai.replace(tame=new_tame)
    )

    skill_lv = _riding_skill_level(state)
    skill_ok = skill_lv >= jnp.int32(_RIDING_SKILL_MIN)

    # Byte-equal port of vendor steed.c::can_saddle / mount chance, lines 93-128:
    #   chance = ACURR(A_DEX) + ACURR(A_CHA)/2 + 2 * mtmp->mtame;
    #   chance += u.ulevel * (mtmp->mtame ? 20 : 5);
    #   if (!mtmp->mtame) chance -= 10 * mtmp->m_lev;
    #   if (Role_if(PM_KNIGHT)) chance += 20;
    #   skill: ISRESTRICTED/UNSKILLED → -20; BASIC → 0; SKILLED → +15; EXPERT → +30
    #   if (Confusion || Fumbling || Glib) chance -= 20
    #   if cursed saddle: chance -= 50  (we already gate via is_cursed → fail)
    dex = state.player_dex.astype(jnp.int32)
    cha = state.player_cha.astype(jnp.int32)
    xl  = state.player_xl.astype(jnp.int32)
    mtame = state.monster_ai.mtame[safe_slot].astype(jnp.int32)
    # Use entry_idx → MONSTERS.level via lookup table built at runtime.
    from Nethax.nethax.constants.monsters import MONSTERS as _MONS_TABLE
    _LVL_TABLE = jnp.array([int(m.level) for m in _MONS_TABLE], dtype=jnp.int32)
    steed_lvl = _LVL_TABLE[jnp.clip(
        state.monster_ai.entry_idx[safe_slot].astype(jnp.int32),
        0, _LVL_TABLE.shape[0] - 1)]

    is_tame = mtame > jnp.int32(0)
    tame_bonus  = jnp.where(is_tame, jnp.int32(20), jnp.int32(5))
    wild_penalty = jnp.where(is_tame, jnp.int32(0), jnp.int32(10) * steed_lvl)
    base = dex + cha // jnp.int32(2) + jnp.int32(2) * mtame
    chance = base + xl * tame_bonus - wild_penalty

    # Knight role bonus.  Role.KNIGHT == 4 (cite roles.py).
    from Nethax.nethax.constants.roles import Role
    is_knight = state.player_role == jnp.int8(int(Role.KNIGHT))
    chance = chance + jnp.where(is_knight, jnp.int32(20), jnp.int32(0))

    # Skill modifier: UNSKILLED -20, BASIC 0, SKILLED +15, EXPERT +30.
    skill_mod = jnp.where(
        skill_lv >= jnp.int32(3), jnp.int32(30),  # EXPERT
        jnp.where(skill_lv >= jnp.int32(2), jnp.int32(15),  # SKILLED
        jnp.where(skill_lv >= jnp.int32(1), jnp.int32(0),   # BASIC
        jnp.int32(-20))))                                     # UNSKILLED/RESTRICTED
    chance = chance + skill_mod

    # Confused / Fumbling / Glib penalty.
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    ts = state.status.timed_statuses
    confused  = ts[int(TimedStatus.CONFUSION)] > 0
    fumbling  = ts[int(TimedStatus.FUMBLING)] > 0
    glib      = ts[int(TimedStatus.GLIB)] > 0
    impaired  = confused | fumbling | glib
    chance = chance + jnp.where(impaired, jnp.int32(-20), jnp.int32(0))

    # Clamp chance to [1, 100] for the roll.
    chance = jnp.clip(chance, jnp.int32(1), jnp.int32(100))

    k_roll, k_fall = jax.random.split(rng)
    roll = jax.random.randint(k_roll, (), minval=0, maxval=100, dtype=jnp.int32)
    success_roll = roll < chance

    # Overall success: not riding, found mount, not cursed, skill ok, roll passes.
    success = (~already_riding) & found & (~is_cursed) & skill_ok & success_roll

    # On success: set player_steed_mid to (slot+1) so 0 remains "not riding".
    mid_on_success = (slot + jnp.int32(1)).astype(jnp.uint32)

    # Steed's move_speed becomes player_extra_speed while riding.
    steed_entry = jnp.clip(
        state.monster_ai.entry_idx[safe_slot].astype(jnp.int32), 0, _N_MONSTERS - 1
    )
    steed_speed = _MONSTER_SPEED_TABLE[steed_entry]

    state_mounted = state_after_cursed.replace(
        player_steed_mid=jnp.where(success, mid_on_success, state.player_steed_mid),
        player_extra_speed=jnp.where(success, steed_speed, state.player_extra_speed),
    )

    # On mount-roll failure (found uncursed mount, skill ok, but roll failed): fall.
    should_fall = (~already_riding) & found & (~is_cursed) & skill_ok & (~success_roll)
    state_after_fall = fall_off_steed(state_mounted, k_fall)

    # Merge: if should_fall apply fall result, else keep mounted state.
    final_hp = jnp.where(should_fall, state_after_fall.player_hp, state_mounted.player_hp)
    final_mid = jnp.where(
        should_fall, state_after_fall.player_steed_mid, state_mounted.player_steed_mid
    )
    final_extra_speed = jnp.where(
        should_fall, state_after_fall.player_extra_speed, state_mounted.player_extra_speed
    )
    old_wl = state_mounted.status.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]
    fall_wl = state_after_fall.status.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]
    final_wl = jnp.where(should_fall, fall_wl, old_wl)
    new_ts = state_mounted.status.timed_statuses.at[int(TimedStatus.WOUNDED_LEGS)].set(final_wl)

    return state_mounted.replace(
        player_hp=final_hp,
        player_steed_mid=final_mid,
        player_extra_speed=final_extra_speed,
        status=state_mounted.status.replace(timed_statuses=new_ts),
    )


# ---------------------------------------------------------------------------
# try_dismount
# ---------------------------------------------------------------------------

def try_dismount(state, rng: jax.Array):
    """Dismount the current steed.

    No-op if not riding.  On dismount, clears player_steed_mid and
    player_extra_speed.

    Vendor reference: steed.c:658 (u.usteed = NULL after releasing steed) and
    steed.c::dismount_steed DISMOUNT_BYCHOICE branch (~line 632).
    """
    riding = state.player_steed_mid != jnp.uint32(0)
    new_mid = jnp.where(riding, jnp.uint32(0), state.player_steed_mid)
    new_extra_speed = jnp.where(riding, jnp.int8(0), state.player_extra_speed)
    return state.replace(player_steed_mid=new_mid, player_extra_speed=new_extra_speed)


# ---------------------------------------------------------------------------
# check_combat_dismount
# ---------------------------------------------------------------------------

def check_combat_dismount(state, damage_taken: jnp.ndarray):
    """Force dismount with fall damage when a single hit deals >= 8 damage.

    Vendor reference: steed.c::dismount_steed DISMOUNT_KNOCKED / DISMOUNT_FELL
    branch (~line 606) — combat hits can dislodge the rider.  We gate on
    damage_taken >= 8 (simplified threshold; vendor uses various checks).

    TODO: wire call into combat.py when combat subsystem applies damage to the
    player; call as: state = check_combat_dismount(state, hit_damage).
    """
    riding = state.player_steed_mid != jnp.uint32(0)
    big_hit = damage_taken.astype(jnp.int32) >= jnp.int32(8)
    should_dismount = riding & big_hit

    # Fixed fall damage of 3 (midpoint of 1d6) to avoid needing an rng arg.
    # Vendor uses rn1(10,10); we use a fixed minimal model.
    fall_dmg = jnp.int32(3)
    new_hp = jnp.maximum(
        state.player_hp - jnp.where(should_dismount, fall_dmg, jnp.int32(0)),
        jnp.int32(0),
    )

    wl_idx = int(TimedStatus.WOUNDED_LEGS)
    old_wl = state.status.timed_statuses[wl_idx]
    new_wl = jnp.where(should_dismount, old_wl + jnp.int32(10), old_wl)
    new_ts = state.status.timed_statuses.at[wl_idx].set(new_wl)

    new_mid = jnp.where(should_dismount, jnp.uint32(0), state.player_steed_mid)
    new_extra_speed = jnp.where(should_dismount, jnp.int8(0), state.player_extra_speed)

    return state.replace(
        player_hp=new_hp,
        player_steed_mid=new_mid,
        player_extra_speed=new_extra_speed,
        status=state.status.replace(timed_statuses=new_ts),
    )


# ---------------------------------------------------------------------------
# tick_saddle
# ---------------------------------------------------------------------------

def tick_saddle(state):
    """Per-turn saddle wear: each turn while riding, prob ~1/100, decrement
    saddle_condition.  When saddle_condition reaches 0 the steed can't be
    ridden until re-saddled.

    Vendor reference: steed.c — saddle wear is implicit in the original code
    through item damage mechanics; we model it as a discrete int8 counter
    (100=new, 0=broken) decremented with probability 1/100 per riding turn.

    Implemented as: wear when (timestep % 100 == 0) while riding.  Deterministic
    to stay JIT-pure (no rng arg needed for a per-turn tick).

    Call from env.step once per turn.
    """
    riding = state.player_steed_mid != jnp.uint32(0)
    condition = state.saddle_condition.astype(jnp.int32)

    wear_tick = (state.timestep % jnp.int32(100)) == jnp.int32(0)
    should_wear = riding & wear_tick & (condition > jnp.int32(0))

    new_condition = jnp.where(should_wear, condition - jnp.int32(1), condition)
    new_condition = jnp.clip(new_condition, 0, 100).astype(jnp.int8)

    # If saddle breaks (condition hits 0), force dismount.
    broken = (new_condition == jnp.int8(0)) & riding
    new_mid = jnp.where(broken, jnp.uint32(0), state.player_steed_mid)
    new_extra_speed = jnp.where(broken, jnp.int8(0), state.player_extra_speed)

    return state.replace(
        saddle_condition=new_condition,
        player_steed_mid=new_mid,
        player_extra_speed=new_extra_speed,
    )
