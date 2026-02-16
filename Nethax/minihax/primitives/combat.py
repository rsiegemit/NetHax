"""Combat primitives shared across Tier 2 and 3."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    MONSTER_STATS, MONSTER_XP_SCORE,
    ABON_STR, ABON_DEX, DBON_STR,
    MONK_MARTIAL_SIDES, MONK_MARTIAL_BONUS,
    RoleType,
)
from Nethax.minihax.primitives.leveling import check_multi_levelup


def do_melee_attack(rng, state, target_pos, static_params):
    """Attack a monster at target_pos with full NetHack to-hit and damage.

    To-hit: 1 + abon(STR) + abon(DEX) + low_level_bonus + monk_hit_bonus + ac + xp_level > d20.
    Monk martial arts: to-hit bonus (level/3 + 2), damage scales by bracket (d4/d6/d8/d10/d12).
    Non-monk bare-handed: 1d2 (no skill bonus).
    STR damage bonus from dbon() table.
    On kill: grant XP with multi-level-up, per-monster scoring.
    """
    max_m = static_params.max_monsters

    # Find monster at target_pos
    mon_positions = state.monsters.position  # [max_m, 2]
    mon_mask = state.monsters.mask           # [max_m]

    pos_match_r = mon_positions[:, 0] == target_pos[0]
    pos_match_c = mon_positions[:, 1] == target_pos[1]
    matches = pos_match_r & pos_match_c & mon_mask

    match_indices = jnp.where(matches, jnp.arange(max_m), max_m)
    target_idx = jnp.min(match_indices)
    found = target_idx < max_m
    safe_idx = jnp.where(found, target_idx, 0)

    rng, rng_dmg, rng_dmg2, rng_hit, rng_lvl = jax.random.split(rng, 5)

    stats = state.player_stats

    # --- To-hit roll (weapon.c abon + hitval) ---
    mon_type = state.monsters.type_id[safe_idx]
    mon_ac = MONSTER_STATS[mon_type, 2]

    str_bonus = ABON_STR[jnp.clip(stats.strength, 0, 25)]
    dex_bonus = ABON_DEX[jnp.clip(stats.dexterity, 0, 25)]
    low_level_bonus = jnp.where(stats.xp_level < 3, 1, 0)

    # Monk martial arts to-hit bonus (uhitm.c:400)
    is_monk = (stats.role_id == RoleType.MONK)
    monk_hit_bonus = jnp.where(is_monk, stats.xp_level // 3 + 2, 0)

    # To-hit: 1 + abon + low_level_bonus + monk_hit_bonus + ac + level > d20
    to_hit = 1 + str_bonus + dex_bonus + low_level_bonus + monk_hit_bonus + mon_ac + stats.xp_level
    hit_roll = jax.random.randint(rng_hit, (), 1, 21)  # d20
    hits = to_hit > hit_roll

    # --- Damage calculation ---
    # Monk martial arts: damage die scales by level bracket
    bracket = jnp.minimum((stats.xp_level - 1) // 4, 4)
    monk_sides = MONK_MARTIAL_SIDES[bracket]
    monk_damage = jax.random.randint(rng_dmg, (), 1, monk_sides + 1) + MONK_MARTIAL_BONUS

    # Non-monk bare-handed: 1d2
    non_monk_damage = jax.random.randint(rng_dmg2, (), 1, 3)  # 1d2

    base_damage = jnp.where(is_monk, monk_damage, non_monk_damage)

    # STR damage bonus
    str_dmg = DBON_STR[jnp.clip(stats.strength, 0, 25)]
    damage = base_damage + str_dmg
    damage = jnp.maximum(damage, 1)  # Minimum 1 on hit

    damage = jnp.where(found & hits, damage, 0)

    # --- Apply damage to monster ---
    old_hp = state.monsters.health[safe_idx]
    new_hp = old_hp - damage
    killed = jnp.logical_and(found, new_hp <= 0)

    new_health = state.monsters.health.at[safe_idx].set(
        jnp.where(found, new_hp, old_hp)
    )
    new_mask = state.monsters.mask.at[safe_idx].set(
        jnp.where(killed, False, state.monsters.mask[safe_idx])
    )
    monsters = state.monsters.replace(health=new_health, mask=new_mask)

    # --- XP gain and multi-level-up on kill ---
    xp_gain = jnp.where(killed, MONSTER_STATS[mon_type, 13], 0)
    new_stats = stats.replace(xp=stats.xp + xp_gain)
    new_stats = check_multi_levelup(rng_lvl, new_stats)

    # --- Per-monster scoring ---
    score_gain = jnp.where(killed, MONSTER_XP_SCORE[mon_type], 0)
    new_stats = new_stats.replace(
        score=new_stats.score + score_gain,
        monsters_killed=new_stats.monsters_killed + jnp.where(killed, 1, 0),
    )

    state = state.replace(
        monsters=monsters,
        player_stats=new_stats,
    )

    return state
