import jax
import jax.numpy as jnp

from nethax.nethax.constants import *
from nethax.nethax.nethax_state import *


def is_solid(tile_type):
    """Check if a tile blocks movement."""
    return jnp.isin(tile_type, SOLID_TILES)


def is_opaque(tile_type):
    """Check if a tile blocks line of sight."""
    return jnp.isin(tile_type, OPAQUE_TILES)


def get_hunger_state(nutrition):
    """Convert raw nutrition value to HungerState enum."""
    state = jnp.where(nutrition > HUNGER_THRESHOLDS[0], HungerState.SATIATED,
            jnp.where(nutrition > HUNGER_THRESHOLDS[1], HungerState.NOT_HUNGRY,
            jnp.where(nutrition > HUNGER_THRESHOLDS[2], HungerState.HUNGRY,
            jnp.where(nutrition > HUNGER_THRESHOLDS[3], HungerState.WEAK,
            jnp.where(nutrition > HUNGER_THRESHOLDS[4], HungerState.FAINTING,
                       HungerState.STARVED)))))
    return state


def roll_dice(rng, num_dice, die_sides):
    """Roll num_dice d(die_sides) and return the sum.

    JAX-compatible: always rolls MAX_DICE dice and masks out extras.
    """
    MAX_DICE = 6  # Covers all NetHack monsters (max 6 attacks)
    rolls = jax.random.randint(rng, (MAX_DICE,), 1, jnp.maximum(die_sides, 1) + 1)
    mask = jnp.arange(MAX_DICE) < num_dice
    return (rolls * mask).sum()


def compute_player_ac(state):
    """Compute total armor class from equipped armor.

    Base AC is 10 (unarmored). Each armor piece contributes its AC value.
    NetHack formula: total = sum of individual armor AC values (lower = better).
    """
    base_ac = 10
    num_slots = 6

    def process_slot(carry, slot_idx):
        total_ac = carry
        inv_idx = state.worn_armor[slot_idx]
        has_armor = inv_idx >= 0

        armor_type = jnp.where(has_armor, state.inventory.type_id[inv_idx], 0)
        armor_ac = ARMOR_STATS[armor_type, 0]  # AC column
        enchant = jnp.where(has_armor, state.inventory.enchantment[inv_idx], 0)

        # Each armor piece: its AC value replaces base 10 for that slot
        # Effective contribution = armor_ac - enchantment (lower = better)
        slot_ac = jnp.where(has_armor, armor_ac - enchant, 10)

        # Combine: take the minimum of current total and this slot
        # Actually NetHack sums: base 10 - sum of (10 - armor_ac) for each slot
        # = 10 - n*10 + sum(ac_i) = 10 - n*10 + sum(ac_i) ...
        # Simpler: subtract (10 - armor_ac + enchant) for each worn piece
        reduction = jnp.where(has_armor, 10 - armor_ac + enchant, 0)
        new_total = total_ac - reduction
        return new_total, None

    ac, _ = jax.lax.scan(process_slot, base_ac, jnp.arange(num_slots))
    return ac


def compute_damage(rng, weapon_type, enchantment, strength):
    """Compute melee damage given weapon, enchantment, and strength.

    WEAPON_STATS columns: [sdam, ldam, hit_bonus, weight, cost, two_handed]
    sdam = die size for damage vs small monsters (roll 1d(sdam))
    For simplicity, we use sdam (column 0) for all targets.
    """
    sdam = WEAPON_STATS[weapon_type, 0]

    rng, _rng = jax.random.split(rng)
    # Roll 1d(sdam). If sdam is 0, damage is 0.
    base_damage = jnp.where(
        sdam > 0,
        jax.random.randint(_rng, (), 1, jnp.maximum(sdam, 1) + 1),
        0
    )

    # Add enchantment bonus
    damage = base_damage + enchantment

    # Strength bonus (simplified: +1 per point above 16)
    str_bonus = jnp.where(strength > 16, strength - 16, 0)
    damage = damage + str_bonus

    return jnp.maximum(damage, 0)


def compute_monster_damage(rng, monster_type):
    """Compute damage dealt by a monster.

    MONSTER_STATS columns: [level, speed, ac, mr, atk_dice, atk_sides, ...]
    Rolls atk_dice d(atk_sides).
    """
    atk_dice = MONSTER_STATS[monster_type, 4]
    atk_sides = MONSTER_STATS[monster_type, 5]
    return roll_dice(rng, atk_dice, atk_sides)


def get_monster_max_hp(monster_type):
    """Get max HP for a monster type."""
    return MONSTER_MAX_HP[monster_type]


def get_monster_speed(monster_type):
    """Get speed for a monster type."""
    return MONSTER_STATS[monster_type, 1]


def get_monster_ac(monster_type):
    """Get AC for a monster type."""
    return MONSTER_STATS[monster_type, 2]


def get_monster_mr(monster_type):
    """Get magic resistance for a monster type."""
    return MONSTER_STATS[monster_type, 3]


def get_monster_xp(monster_type):
    """Get XP reward for killing a monster (uses difficulty rating)."""
    return MONSTER_STATS[monster_type, 8]


def get_xp_for_level(level):
    """Get XP required to reach a given experience level (1-indexed).

    Uses the XP_TABLE from constants (real NetHack formula).
    Level is clamped to valid range [1, 30].
    """
    idx = jnp.clip(level - 1, 0, 29)
    return XP_TABLE[idx]


def in_bounds(position, map_size):
    """Check if a position is within map bounds."""
    return jnp.logical_and(
        jnp.logical_and(position[0] >= 0, position[0] < map_size[0]),
        jnp.logical_and(position[1] >= 0, position[1] < map_size[1]),
    )


def manhattan_distance(pos1, pos2):
    """Compute Manhattan distance between two positions."""
    return jnp.abs(pos1[0] - pos2[0]) + jnp.abs(pos1[1] - pos2[1])


def chebyshev_distance(pos1, pos2):
    """Compute Chebyshev distance (king move distance) between two positions."""
    return jnp.maximum(jnp.abs(pos1[0] - pos2[0]), jnp.abs(pos1[1] - pos2[1]))
