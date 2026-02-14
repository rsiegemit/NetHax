"""World generators for HideNSeek environments (4 variants).

From .des files:
- hidenseek.des: 11x9 floor, 33% cloud, 25% tree, 2 diagonal paths, 1 monster
- hidenseek_big.des: 15x15 version
- hidenseek_lava.des: 11x9 + 5% lava
- hidenseek_mapped.des: 11x9 premapped (same generation, flag only)

.des coordinate convention: (col, row). Our code: (row, col).

Monster pool: {COCKATRICE, NAGA_HATCHLING, MINOTAUR, OGRE, BABY_RED_DRAGON, TROLL}
3 shuffled positions: one for monster, one for player (BRANCH), one for stair.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, MonsterType, MONSTER_MAX_HP, ItemType, RoleType, RaceType,
)
from Nethax.minihax.states import (
    HazardState, HazardStaticParams, Inventory, SimpleMonsters, GroundItems,
)
from Nethax.minihax.primitives.visibility import compute_visible
from Nethax.minihax.primitives.leveling import compute_initial_stats
from Nethax.minihax.world_gen.procedural import replace_terrain_random, randline


_MONSTER_POOL = jnp.array([
    MonsterType.COCKATRICE,
    MonsterType.NAGA_HATCHLING,
    MonsterType.MINOTAUR,
    MonsterType.OGRE,
    MonsterType.BABY_RED_DRAGON,
    MonsterType.TROLL,
], dtype=jnp.int32)


def _generate_hidenseek_common(rng, params, static_params, active_h, active_w, has_lava):
    """Common generation logic for all HideNSeek variants.

    Args:
        rng: JAX PRNG key
        params: EnvParams
        static_params: HazardStaticParams
        active_h: int — active map height
        active_w: int — active map width
        has_lava: bool — whether to add 5% lava replacement

    Returns:
        HazardState
    """
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items
    map_h = static_params.map_height
    map_w = static_params.map_width

    rng, rng_cloud, rng_tree, rng_lava, rng_line1, rng_line2, rng_monster, rng_shuffle, rng_pos, rng_stats = \
        jax.random.split(rng, 10)

    player_stats = compute_initial_stats(rng_stats, RoleType.MONK, RaceType.HUMAN)

    # Start with all-floor map (active area)
    game_map = jnp.full((map_h, map_w), TileType.VOID, dtype=jnp.int32)
    rows = jnp.arange(map_h)[:, None]
    cols = jnp.arange(map_w)[None, :]
    active = (rows < active_h) & (cols < active_w)
    game_map = jnp.where(active, TileType.FLOOR, game_map)

    # REPLACE_TERRAIN: 33% floor -> cloud
    region = (0, 0, active_h - 1, active_w - 1)
    game_map = replace_terrain_random(rng_cloud, game_map, region,
                                      TileType.FLOOR, TileType.CLOUD, 33.0, map_h, map_w)

    # REPLACE_TERRAIN: 25% floor -> tree
    game_map = replace_terrain_random(rng_tree, game_map, region,
                                      TileType.FLOOR, TileType.TREE, 25.0, map_h, map_w)

    # Optional: 5% floor -> lava (hidenseek_lava)
    game_map = jnp.where(
        has_lava,
        replace_terrain_random(rng_lava, game_map, region,
                               TileType.FLOOR, TileType.LAVA, 5.0, map_h, map_w),
        game_map,
    )

    # Two diagonal randline paths for connectivity
    # .des: TERRAIN:randline (0,active_h-1),(active_w-1,0), 5, '.'
    # .des coords: (col, row) -> our (row, col)
    # Line 1: from (row=active_h-1, col=0) to (row=0, col=active_w-1)
    start1 = jnp.array([active_h - 1, 0], dtype=jnp.int32)
    end1 = jnp.array([0, active_w - 1], dtype=jnp.int32)
    game_map = randline(rng_line1, game_map, start1, end1, 5, TileType.FLOOR, map_h, map_w)

    # Line 2: from (row=0, col=0) to (row=active_h-1, col=active_w-1)
    start2 = jnp.array([0, 0], dtype=jnp.int32)
    end2 = jnp.array([active_h - 1, active_w - 1], dtype=jnp.int32)
    game_map = randline(rng_line2, game_map, start2, end2, 5, TileType.FLOOR, map_h, map_w)

    # 3 positions for monster/player/stair (shuffled)
    # .des for 11x9: $place = { (10,8),(0,8),(10,0) }
    # (col,row) -> (row,col): (8,10), (8,0), (0,10)
    # .des for 15x15: $place = { (14,14),(0,14),(14,0) }
    # (col,row) -> (row,col): (14,14), (14,0), (0,14)
    places_11x9 = jnp.array([
        [active_h - 1, active_w - 1],    # (10,8) -> (8,10) but h=9,w=11 so (8,10)
        [active_h - 1, 0],               # (0,8) -> (8,0)
        [0, active_w - 1],               # (10,0) -> (0,10)
    ], dtype=jnp.int32)

    # Shuffle the 3 positions
    perm = jax.random.permutation(rng_shuffle, 3)
    places = places_11x9[perm]

    monster_pos = places[0]
    player_pos = places[1]  # BRANCH is the second in shuffled list (player start)
    stair_pos = places[2]

    # Place downstair on map
    game_map = game_map.at[stair_pos[0], stair_pos[1]].set(TileType.DOWNSTAIR)

    # Ensure player and monster positions are floor
    game_map = game_map.at[player_pos[0], player_pos[1]].set(TileType.FLOOR)
    game_map = game_map.at[monster_pos[0], monster_pos[1]].set(TileType.FLOOR)

    # Pick random monster from pool
    mon_idx = jax.random.randint(rng_monster, (), 0, 6)
    mon_type = _MONSTER_POOL[mon_idx]

    # Create SimpleMonsters (1 monster)
    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(monster_pos)
    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(mon_type)
    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[mon_type])
    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)

    monsters = SimpleMonsters(
        position=mon_positions,
        type_id=mon_types,
        health=mon_health,
        mask=mon_mask,
    )

    # Empty inventory and ground items
    inv = Inventory(
        item_ids=jnp.zeros(max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
    )
    ground_items = GroundItems(
        position=jnp.zeros((max_gi, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_gi, dtype=jnp.int32),
        mask=jnp.zeros(max_gi, dtype=jnp.bool_),
    )

    visible_map = compute_visible(player_pos, game_map, map_h, map_w)
    return HazardState(
        map=game_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        player_stats=player_stats,
        player_levitating=False,
        levitation_turns=0,
        inventory=inv,
        monsters=monsters,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        timestep=0,
        prev_action=0,
        terminal=False,
        state_rng=rng,
    )


def generate_hidenseek(rng, params, static_params):
    """Generate HideNSeek (11x9, no lava)."""
    return _generate_hidenseek_common(rng, params, static_params, 9, 11, False)


def generate_hidenseek_big(rng, params, static_params):
    """Generate HideNSeekBig (15x15, no lava)."""
    return _generate_hidenseek_common(rng, params, static_params, 15, 15, False)


def generate_hidenseek_lava(rng, params, static_params):
    """Generate HideNSeekLava (11x9, with 5% lava)."""
    return _generate_hidenseek_common(rng, params, static_params, 9, 11, True)


def generate_hidenseek_mapped(rng, params, static_params):
    """Generate HideNSeekMapped (11x9, no lava, premapped flag).

    Same map generation as hidenseek. The premapped flag only affects
    NetHack's fog-of-war (not relevant for our symbolic obs).
    """
    return _generate_hidenseek_common(rng, params, static_params, 9, 11, False)
