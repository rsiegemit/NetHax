"""Helper utilities for minihax game logic."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import SOLID_TILES


def is_solid(tile_type):
    """Check if a tile blocks movement."""
    return jnp.isin(tile_type, SOLID_TILES)


def in_bounds(position, map_height, map_width):
    """Check if a position is within map bounds."""
    return jnp.logical_and(
        jnp.logical_and(position[0] >= 0, position[0] < map_height),
        jnp.logical_and(position[1] >= 0, position[1] < map_width),
    )


def chebyshev_distance(pos1, pos2):
    """Compute Chebyshev distance (king move distance)."""
    return jnp.maximum(jnp.abs(pos1[0] - pos2[0]), jnp.abs(pos1[1] - pos2[1]))


def dist2(pos1, pos2):
    """Squared Euclidean distance (NetHack hacklib.c dist2)."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    return dx * dx + dy * dy


def roll_dice(rng, num_dice, die_sides):
    """Roll num_dice d(die_sides), JAX-compatible."""
    MAX_DICE = 6
    rolls = jax.random.randint(rng, (MAX_DICE,), 1, jnp.maximum(die_sides, 1) + 1)
    mask = jnp.arange(MAX_DICE) < num_dice
    return (rolls * mask).sum()


def compute_monster_damage(rng, monster_type):
    """Compute total damage from all attack slots of a monster.

    Each monster has up to 4 attack slots (columns 4-11 of MONSTER_STATS).
    Slots with dice=0 are skipped. Total damage = sum of all slot rolls.
    """
    from Nethax.minihax.constants import MONSTER_STATS

    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    # Attack slot 1
    a1_dice = MONSTER_STATS[monster_type, 4]
    a1_sides = MONSTER_STATS[monster_type, 5]
    d1 = jnp.where(a1_dice > 0, roll_dice(rng1, a1_dice, a1_sides), 0)

    # Attack slot 2
    a2_dice = MONSTER_STATS[monster_type, 6]
    a2_sides = MONSTER_STATS[monster_type, 7]
    d2 = jnp.where(a2_dice > 0, roll_dice(rng2, a2_dice, a2_sides), 0)

    # Attack slot 3
    a3_dice = MONSTER_STATS[monster_type, 8]
    a3_sides = MONSTER_STATS[monster_type, 9]
    d3 = jnp.where(a3_dice > 0, roll_dice(rng3, a3_dice, a3_sides), 0)

    # Attack slot 4
    a4_dice = MONSTER_STATS[monster_type, 10]
    a4_sides = MONSTER_STATS[monster_type, 11]
    d4 = jnp.where(a4_dice > 0, roll_dice(rng4, a4_dice, a4_sides), 0)

    return d1 + d2 + d3 + d4


def get_xp_for_level(level):
    """Get XP required for a given experience level."""
    from Nethax.minihax.constants import XP_TABLE
    idx = jnp.clip(level - 1, 0, 29)
    return XP_TABLE[idx]
