"""World generators for Tier 1 CORRIDOR navigation environments.

Implements 5 corridor environments from MiniHack .des files:
- corridor2.des  -> generate_corridor2   (2 rooms)
- corridor3.des  -> generate_corridor3   (3 rooms)
- corridor5.des  -> generate_corridor5   (5 rooms)
- corridor8.des  -> generate_corridor8   (8 rooms)
- corridor10.des -> generate_corridor10  (10 rooms)

Each .des places N lit rooms connected by RANDOM_CORRIDORS.
Room 0 has upstair (player start), room 1 has downstair (goal).
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.states import NavigationState, NavigationStaticParams, GroundItems
from Nethax.minihax.primitives.visibility import compute_visible, compute_lit_map
from Nethax.minihax.world_gen.procedural import random_corridors


def _empty_ground_items(max_gi):
    """Create empty GroundItems (no items on ground)."""
    return GroundItems(
        position=jnp.zeros((max_gi, 2), dtype=jnp.int32),
        type_id=jnp.zeros(max_gi, dtype=jnp.int32),
        mask=jnp.zeros(max_gi, dtype=jnp.bool_),
    )


def _generate_corridor(rng, params, static_params, num_rooms):
    """Shared generator for all corridor environments."""
    rng, gen_rng = jax.random.split(rng)
    game_map, player_pos, stair_pos = random_corridors(
        gen_rng, num_rooms,
        static_params.map_height, static_params.map_width,
    )
    lit_map = compute_lit_map(game_map)
    visible_map = compute_visible(player_pos, game_map, static_params.map_height, static_params.map_width, lit_map)
    return NavigationState(
        map=game_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        ground_items=_empty_ground_items(static_params.max_ground_items),
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        timestep=0,
        prev_action=0,
        terminal=False,
        state_rng=rng,
    )


def generate_corridor2(rng, params, static_params):
    """corridor2.des: 2 rooms connected by corridors."""
    return _generate_corridor(rng, params, static_params, 2)


def generate_corridor3(rng, params, static_params):
    """corridor3.des: 3 rooms connected by corridors."""
    return _generate_corridor(rng, params, static_params, 3)


def generate_corridor5(rng, params, static_params):
    """corridor5.des: 5 rooms connected by corridors."""
    return _generate_corridor(rng, params, static_params, 5)


def generate_corridor8(rng, params, static_params):
    """corridor8.des: 8 rooms connected by corridors."""
    return _generate_corridor(rng, params, static_params, 8)


def generate_corridor10(rng, params, static_params):
    """corridor10.des: 10 rooms connected by corridors."""
    return _generate_corridor(rng, params, static_params, 10)
