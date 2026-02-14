"""Player leveling primitives matching NetHack 3.7."""
import jax
import jax.numpy as jnp

from Nethax.minihax.states import PlayerStats
from Nethax.minihax.constants import (
    ROLE_BASE_STATS, ROLE_HP_ADV, ROLE_PW_ADV, ROLE_XLEV, ROLE_ENERMOD,
    RACE_HP_ADV, RACE_PW_ADV,
    ROLE_INTRINSICS, RACE_INTRINSICS,
    CON_HP_BONUS, MAX_PLAYER_LEVEL, RoleType,
)
from Nethax.minihax.util.game_logic_utils import get_xp_for_level


# ============================================================================
# Random helpers (NetHack convention)
# ============================================================================

def _rnd(rng, n):
    """NetHack rnd(n): random integer in [1, n]. Returns 0 if n <= 0."""
    safe_n = jnp.maximum(n, 1)
    val = jax.random.randint(rng, (), 1, safe_n + 1)
    return jnp.where(n > 0, val, 0)


def _rn1(rng, x, y):
    """NetHack rn1(x, y): random int in [y, y+x-1]. Returns y if x <= 0."""
    safe_x = jnp.maximum(x, 1)
    val = jax.random.randint(rng, (), 0, safe_x) + y
    return jnp.where(x > 0, val, y)


# ============================================================================
# Per-level HP / Power gain
# ============================================================================

def newhp(rng, stats):
    """Calculate HP gain for leveling up. Matches NetHack attrib.c newhp().

    HP advancement depends on role, race, level tier, and constitution.
    Below ROLE_XLEV uses lo-tier dice; at/above uses hi-tier dice.
    """
    role_id = stats.role_id
    race_id = stats.race_id
    role_hp = ROLE_HP_ADV[role_id]
    race_hp = RACE_HP_ADV[race_id]
    xlev = ROLE_XLEV[role_id]

    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    # Low tier (below xlev): lofix + rnd(lornd)
    lo_hp = role_hp[2] + race_hp[2] + _rnd(rng1, role_hp[3]) + _rnd(rng2, race_hp[3])
    # High tier (at or above xlev): hifix + rnd(hirnd)
    hi_hp = role_hp[4] + race_hp[4] + _rnd(rng3, role_hp[5]) + _rnd(rng4, race_hp[5])

    hp = jnp.where(stats.xp_level < xlev, lo_hp, hi_hp)

    # CON bonus
    con = jnp.clip(stats.constitution, 0, 25)
    hp = hp + CON_HP_BONUS[con]

    # Minimum 1
    hp = jnp.maximum(hp, 1)
    return hp


def newpw(rng, stats):
    """Calculate power/energy gain for leveling up. Matches NetHack exper.c newpw().

    Energy advancement uses WIS/2 plus role/race dice, then scaled by enermod.
    """
    role_id = stats.role_id
    race_id = stats.race_id
    role_pw = ROLE_PW_ADV[role_id]
    race_pw = RACE_PW_ADV[race_id]
    xlev = ROLE_XLEV[role_id]
    enermod_pct = ROLE_ENERMOD[role_id]

    below_xlev = stats.xp_level < xlev

    # enrnd and enfix depend on tier
    lo_rnd = stats.wisdom // 2 + role_pw[3] + race_pw[3]
    lo_fix = role_pw[2] + race_pw[2]
    hi_rnd = stats.wisdom // 2 + role_pw[5] + race_pw[5]
    hi_fix = role_pw[4] + race_pw[4]

    enrnd = jnp.where(below_xlev, lo_rnd, hi_rnd)
    enfix = jnp.where(below_xlev, lo_fix, hi_fix)

    # rn1(enrnd, enfix) = rn2(enrnd) + enfix
    en = _rn1(rng, enrnd, enfix)

    # Apply enermod: en = en * enermod_pct / 100
    en = (en * enermod_pct) // 100

    # Minimum 1
    en = jnp.maximum(en, 1)
    return en


# ============================================================================
# Initial character creation
# ============================================================================

def compute_initial_stats(rng, role_id, race_id):
    """Create initial PlayerStats for a new character.

    Computes base attributes from role, initial HP from role+race infix/inrnd,
    initial energy with enermod scaling, and starting intrinsics.
    """
    rng, rng_hp1, rng_hp2, rng_pw1, rng_pw2 = jax.random.split(rng, 5)

    # Base stats from role
    base_stats = ROLE_BASE_STATS[role_id]
    str_val = base_stats[0]
    int_val = base_stats[1]
    wis_val = base_stats[2]
    dex_val = base_stats[3]
    con_val = base_stats[4]
    cha_val = base_stats[5]

    # Initial HP = role.infix + race.infix + rnd(role.inrnd) + rnd(race.inrnd)
    role_hp = ROLE_HP_ADV[role_id]
    race_hp = RACE_HP_ADV[race_id]
    init_hp = role_hp[0] + race_hp[0] + _rnd(rng_hp1, role_hp[1]) + _rnd(rng_hp2, race_hp[1])

    # Initial Energy = role.infix + race.infix + rnd(role.inrnd) + rnd(race.inrnd), then enermod
    role_pw = ROLE_PW_ADV[role_id]
    race_pw = RACE_PW_ADV[race_id]
    init_en = role_pw[0] + race_pw[0] + _rnd(rng_pw1, role_pw[1]) + _rnd(rng_pw2, race_pw[1])
    enermod_pct = ROLE_ENERMOD[role_id]
    init_en = jnp.maximum((init_en * enermod_pct) // 100, 1)

    # Initial AC: 10 for all (Monk AC bonus handled in gain_level/combat)
    init_ac = 10

    # Initial intrinsics
    init_intrinsics = ROLE_INTRINSICS[role_id, 1] | RACE_INTRINSICS[race_id, 1]

    # HP/energy history arrays
    hp_inc = jnp.zeros(30, dtype=jnp.int32).at[0].set(init_hp)
    en_inc = jnp.zeros(30, dtype=jnp.int32).at[0].set(init_en)

    return PlayerStats(
        role_id=role_id,
        race_id=race_id,
        strength=str_val,
        intelligence=int_val,
        wisdom=wis_val,
        dexterity=dex_val,
        constitution=con_val,
        charisma=cha_val,
        xp=0,
        xp_level=1,
        hp=init_hp,
        max_hp=init_hp,
        energy=init_en,
        max_energy=init_en,
        ac=init_ac,
        hp_inc=hp_inc,
        en_inc=en_inc,
        intrinsics=init_intrinsics,
        score=0,
        monsters_killed=0,
    )


# ============================================================================
# Level gain / loss
# ============================================================================

def gain_level(rng, stats):
    """Gain one experience level. Call when xp >= threshold for next level.

    Increments level, adds HP (newhp) and energy (newpw), stores gains in
    history arrays for reversible level drain, updates intrinsics and Monk AC.
    """
    rng_hp, rng_pw = jax.random.split(rng)

    new_level = jnp.minimum(stats.xp_level + 1, MAX_PLAYER_LEVEL)

    # HP gain
    hp_gain = newhp(rng_hp, stats)
    new_max_hp = stats.max_hp + hp_gain
    new_hp = stats.hp + hp_gain

    # Energy gain
    en_gain = newpw(rng_pw, stats)
    new_max_en = stats.max_energy + en_gain
    new_en = stats.energy + en_gain

    # Store in history arrays (new_level - 1 is 0-indexed)
    level_idx = jnp.clip(new_level - 1, 0, 29)
    new_hp_inc = stats.hp_inc.at[level_idx].set(hp_gain)
    new_en_inc = stats.en_inc.at[level_idx].set(en_gain)

    # Update intrinsics
    new_intrinsics = ROLE_INTRINSICS[stats.role_id, new_level] | RACE_INTRINSICS[stats.race_id, new_level]

    # Monk AC bonus: AC = 10 - (level - 1)
    is_monk = (stats.role_id == RoleType.MONK)
    monk_ac = 10 - (new_level - 1)
    new_ac = jnp.where(is_monk, monk_ac, stats.ac)

    return stats.replace(
        xp_level=new_level,
        hp=new_hp,
        max_hp=new_max_hp,
        energy=new_en,
        max_energy=new_max_en,
        ac=new_ac,
        hp_inc=new_hp_inc,
        en_inc=new_en_inc,
        intrinsics=new_intrinsics,
    )


def lose_level(stats):
    """Lose one experience level (level drain). Reverses exact gains.

    Subtracts the recorded HP/energy gains for the current level, clamps
    minimums, sets XP just below the new level threshold, and recalculates
    intrinsics and Monk AC.
    """
    old_level = stats.xp_level
    new_level = jnp.maximum(old_level - 1, 1)

    level_idx = jnp.clip(old_level - 1, 0, 29)
    hp_loss = stats.hp_inc[level_idx]
    en_loss = stats.en_inc[level_idx]

    new_max_hp = jnp.maximum(stats.max_hp - hp_loss, 1)
    new_hp = jnp.minimum(stats.hp, new_max_hp)
    new_max_en = jnp.maximum(stats.max_energy - en_loss, 0)
    new_en = jnp.minimum(stats.energy, new_max_en)

    # Set XP to just below new level threshold
    new_xp = get_xp_for_level(new_level) - 1
    new_xp = jnp.maximum(new_xp, 0)

    # Recalculate intrinsics for new level
    new_intrinsics = ROLE_INTRINSICS[stats.role_id, new_level] | RACE_INTRINSICS[stats.race_id, new_level]

    # Monk AC
    is_monk = (stats.role_id == RoleType.MONK)
    monk_ac = 10 - (new_level - 1)
    new_ac = jnp.where(is_monk, monk_ac, stats.ac)

    # Clear the history entry for the lost level
    new_hp_inc = stats.hp_inc.at[level_idx].set(0)
    new_en_inc = stats.en_inc.at[level_idx].set(0)

    return stats.replace(
        xp=new_xp,
        xp_level=new_level,
        hp=new_hp,
        max_hp=new_max_hp,
        energy=new_en,
        max_energy=new_max_en,
        ac=new_ac,
        hp_inc=new_hp_inc,
        en_inc=new_en_inc,
        intrinsics=new_intrinsics,
    )


# ============================================================================
# Multi-level XP check (while_loop for JIT)
# ============================================================================

def check_multi_levelup(rng, stats):
    """Check and apply all pending level-ups. Handles multi-level XP gains.

    Uses jax.lax.while_loop to repeatedly call gain_level as long as the
    player's XP exceeds the threshold for the next level, up to MAX_PLAYER_LEVEL.
    """
    def cond_fn(carry):
        s, r = carry
        next_xp = get_xp_for_level(s.xp_level + 1)
        return (s.xp >= next_xp) & (s.xp_level < MAX_PLAYER_LEVEL)

    def body_fn(carry):
        s, r = carry
        r, r_lvl = jax.random.split(r)
        s = gain_level(r_lvl, s)
        return (s, r)

    stats, _ = jax.lax.while_loop(cond_fn, body_fn, (stats, rng))
    return stats
