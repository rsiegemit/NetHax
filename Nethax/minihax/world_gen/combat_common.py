"""Shared helpers for Tier 3 combat world generators."""
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, PLAYER_START_HP, PLAYER_START_MAX_HP,
    PLAYER_START_AC, PLAYER_START_STRENGTH, PLAYER_START_XP_LEVEL,
)
from Nethax.minihax.states import (
    CombatState, Inventory, Monsters, Traps, GroundItems,
)


def pad_map(game_map, static_params):
    """Pad a map to static_params dimensions with VOID."""
    sh = static_params.map_height
    sw = static_params.map_width
    h, w = game_map.shape
    padded = jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32)
    padded = padded.at[:h, :w].set(game_map)
    return padded


def empty_combat_state(static_params, rng):
    """Create an empty CombatState with all fields zeroed/default.

    Caller should replace fields as needed.
    """
    max_m = static_params.max_monsters
    max_items = static_params.max_items
    max_gi = static_params.max_ground_items
    max_traps = static_params.max_traps
    sh = static_params.map_height
    sw = static_params.map_width

    return CombatState(
        map=jnp.full((sh, sw), TileType.VOID, dtype=jnp.int32),
        player_position=jnp.zeros(2, dtype=jnp.int32),
        downstair_position=jnp.zeros(2, dtype=jnp.int32),
        player_hp=PLAYER_START_HP,
        player_max_hp=PLAYER_START_MAX_HP,
        player_xp=jnp.int32(0),
        player_xp_level=PLAYER_START_XP_LEVEL,
        player_ac=PLAYER_START_AC,
        player_strength=PLAYER_START_STRENGTH,
        player_levitating=False,
        levitation_turns=jnp.int32(0),
        player_has_key=False,
        inventory=Inventory(
            item_ids=jnp.zeros(max_items, dtype=jnp.int32),
            item_mask=jnp.zeros(max_items, dtype=jnp.bool_),
        ),
        monsters=Monsters(
            position=jnp.zeros((max_m, 2), dtype=jnp.int32),
            type_id=jnp.zeros(max_m, dtype=jnp.int32),
            health=jnp.zeros(max_m, dtype=jnp.int32),
            mask=jnp.zeros(max_m, dtype=jnp.bool_),
            movement_points=jnp.zeros(max_m, dtype=jnp.int32),
            is_sleeping=jnp.zeros(max_m, dtype=jnp.bool_),
        ),
        traps=Traps(
            position=jnp.zeros((max_traps, 2), dtype=jnp.int32),
            type_id=jnp.zeros(max_traps, dtype=jnp.int32),
            triggered=jnp.zeros(max_traps, dtype=jnp.bool_),
            mask=jnp.zeros(max_traps, dtype=jnp.bool_),
        ),
        ground_items=GroundItems(
            position=jnp.zeros((max_gi, 2), dtype=jnp.int32),
            type_id=jnp.zeros(max_gi, dtype=jnp.int32),
            mask=jnp.zeros(max_gi, dtype=jnp.bool_),
        ),
        seen_map=jnp.zeros((sh, sw), dtype=jnp.bool_),
        visible_map=jnp.zeros((sh, sw), dtype=jnp.bool_),
        score=jnp.int32(0),
        monsters_killed=jnp.int32(0),
        timestep=jnp.int32(0),
        prev_action=0,
        terminal=False,
        state_rng=rng,
    )


# Map character to TileType mapping for parsing .des ASCII maps
CHAR_TO_TILE = {
    ' ': TileType.VOID,
    '.': TileType.FLOOR,
    '-': TileType.HWALL,
    '|': TileType.VWALL,
    'L': TileType.LAVA,
    '#': TileType.CORRIDOR,
    'F': TileType.IRON_BARS,
    '+': TileType.DOOR_CLOSED,
    '<': TileType.UPSTAIR,
    '>': TileType.DOWNSTAIR,
    '_': TileType.ALTAR,
    'S': TileType.DOOR_CLOSED,  # Secret door = closed door for us
    '{': TileType.FLOOR,        # Fountain = floor for simplicity
    '}': TileType.LAVA,         # Water/moat = lava for simplicity
}


def parse_map(lines):
    """Parse ASCII map lines into a jnp array of TileType values.

    Args:
        lines: list of strings, each representing a map row

    Returns:
        jnp.ndarray [H, W] of TileType int32 values
    """
    h = len(lines)
    w = max(len(line) for line in lines)
    result = []
    for line in lines:
        row = []
        for c in line:
            row.append(int(CHAR_TO_TILE.get(c, TileType.VOID)))
        # Pad row to width
        while len(row) < w:
            row.append(int(TileType.VOID))
        result.append(row)
    return jnp.array(result, dtype=jnp.int32)
