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
    """Compute damage dealt by a monster."""
    from Nethax.minihax.constants import MONSTER_STATS
    atk_dice = MONSTER_STATS[monster_type, 4]
    atk_sides = MONSTER_STATS[monster_type, 5]
    return roll_dice(rng, atk_dice, atk_sides)


def get_xp_for_level(level):
    """Get XP required for a given experience level."""
    from Nethax.minihax.constants import XP_TABLE
    idx = jnp.clip(level - 1, 0, 29)
    return XP_TABLE[idx]
