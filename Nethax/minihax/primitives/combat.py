"""Combat primitives shared across Tier 2 and 3."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    MONSTER_STATS, MAX_PLAYER_LEVEL, ZOMBIE_KILL_SCORE,
)
from Nethax.minihax.util.game_logic_utils import get_xp_for_level


def do_melee_attack(rng, state, target_pos, static_params):
    """Attack a monster at target_pos with bare hands (1d2 damage).

    No weapons in ZombieHorde — player is unarmed.
    On kill: grant XP, check level-up, increment score and kills.
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

    # Monk martial arts: 1d4 base damage (uhitm.c: rnd(4) when martial_bonus())
    rng, rng_dmg, rng_lvlup = jax.random.split(rng, 3)
    damage = jax.random.randint(rng_dmg, (), 1, 5)  # 1-4
    damage = jnp.where(found, damage, 0)

    # Apply damage
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

    # XP on kill
    mon_type = state.monsters.type_id[safe_idx]
    xp_gain = jnp.where(killed, MONSTER_STATS[mon_type, 7], 0)
    new_xp = state.player_xp + xp_gain

    # Level-up check
    old_level = state.player_xp_level
    next_level_xp = get_xp_for_level(old_level + 1)
    leveled_up = jnp.logical_and(killed, new_xp >= next_level_xp)
    new_xp_level = jnp.where(leveled_up, old_level + 1, old_level)
    new_xp_level = jnp.minimum(new_xp_level, MAX_PLAYER_LEVEL)

    # HP gain on level up: 1d8
    hp_gain = jnp.where(leveled_up, jax.random.randint(rng_lvlup, (), 1, 9), 0)
    new_max_hp = state.player_max_hp + hp_gain
    new_player_hp = state.player_hp + hp_gain

    # Score and kill count
    score_gain = jnp.where(killed, ZOMBIE_KILL_SCORE, 0)
    new_score = state.score + score_gain
    new_kills = state.monsters_killed + jnp.where(killed, 1, 0)

    state = state.replace(
        monsters=monsters,
        player_xp=new_xp,
        player_xp_level=new_xp_level,
        player_hp=new_player_hp,
        player_max_hp=new_max_hp,
        score=new_score,
        monsters_killed=new_kills,
    )

    return state
