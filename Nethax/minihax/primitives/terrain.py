"""Terrain damage primitives for Tier 2 and 3."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import TileType, INTRINSIC_FIRE_RES


def apply_terrain_damage(player_pos, player_hp, game_map, player_levitating, intrinsics, rng):
    """Apply terrain damage at player position.

    LAVA:
    - Without fire resistance AND without levitation: instant death (-9999 damage)
    - With fire resistance but without levitation: 6d6 fire damage
    - With levitation: no damage

    Args:
        player_pos: jnp.ndarray [2] — (row, col)
        player_hp: int — current HP
        game_map: jnp.ndarray [map_h, map_w] — tile type IDs
        player_levitating: bool — whether player is levitating
        intrinsics: int — player intrinsic flags bitfield
        rng: JAX PRNG key

    Returns:
        new_hp: int — HP after terrain damage
        new_rng: JAX PRNG key
    """
    r, c = player_pos[0], player_pos[1]
    tile = game_map[r, c]
    on_lava = tile == TileType.LAVA
    has_fire_res = (intrinsics & INTRINSIC_FIRE_RES) != 0
    rng, rng_dmg = jax.random.split(rng)

    # Lethal without fire resistance or levitation
    takes_lethal = on_lava & jnp.logical_not(player_levitating) & jnp.logical_not(has_fire_res)
    # Burns with fire resistance but no levitation
    takes_burn = on_lava & jnp.logical_not(player_levitating) & has_fire_res

    lethal_dmg = jnp.where(takes_lethal, 9999, 0)
    burn_dmg = jnp.where(takes_burn, jax.random.randint(rng_dmg, (), 6, 37), 0)  # 6d6
    damage = lethal_dmg + burn_dmg
    return player_hp - damage, rng


def freeze_lava_around(game_map, player_pos, map_h, map_w):
    """Freeze LAVA tiles in 5x5 area around player to FLOOR.

    Used by wand of cold / frost horn.

    Args:
        game_map: jnp.ndarray [map_h, map_w]
        player_pos: jnp.ndarray [2] — (row, col)
        map_h: int — map height
        map_w: int — map width

    Returns:
        new_map: jnp.ndarray [map_h, map_w] with lava frozen to floor
    """
    rows = jnp.arange(map_h)[:, None]
    cols = jnp.arange(map_w)[None, :]
    r, c = player_pos[0], player_pos[1]
    in_range = (jnp.abs(rows - r) <= 2) & (jnp.abs(cols - c) <= 2)
    is_lava = game_map == TileType.LAVA
    should_freeze = in_range & is_lava
    return jnp.where(should_freeze, TileType.FLOOR, game_map)
