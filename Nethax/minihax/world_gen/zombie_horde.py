"""World generation for ZombieHorde: fixed 18x15 room with 16 zombies + priest."""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, MonsterType, MONSTER_MAX_HP,
)
from Nethax.minihax.states import (
    CombatState, Monsters, Inventory, GroundItems, Traps, EnvParams,
)
from Nethax.minihax.primitives.leveling import compute_initial_stats
from Nethax.minihax.primitives.visibility import compute_visible


def generate_zombie_horde(rng, params, static_params):
    """Generate the ZombieHorde level.

    Map layout from zombie_horde.des:
    - 18 wide x 15 tall (including wall border)
    - Walls on all borders, floor inside
    - Altar at (2, 2) - row 2, col 2
    - Upstaircase at (2, 1) - from BRANCH:(1,2,1,2)
    - Temple region at (1,1)-(3,3) with priest at (2,3)
    - 16 human zombies randomly placed in rect (6,6)-(12,12)
    - Player starts at position (2, 1) -- BRANCH:(1,2,1,2) means col=1, row=2

    Note: .des uses (col, row) coordinates. We use (row, col) internally.
    """
    max_m = static_params.max_monsters  # 17

    # Build original 18-wide map in local variables
    local_h, local_w = 15, 18
    local_map = jnp.full((local_h, local_w), TileType.FLOOR, dtype=jnp.int32)

    # Top and bottom walls (horizontal)
    local_map = local_map.at[0, :].set(TileType.HWALL)
    local_map = local_map.at[local_h - 1, :].set(TileType.HWALL)
    # Left and right walls (vertical)
    local_map = local_map.at[:, 0].set(TileType.VWALL)
    local_map = local_map.at[:, local_w - 1].set(TileType.VWALL)
    # Corners
    local_map = local_map.at[0, 0].set(TileType.TLCORN)
    local_map = local_map.at[0, local_w - 1].set(TileType.TRCORN)
    local_map = local_map.at[local_h - 1, 0].set(TileType.BLCORN)
    local_map = local_map.at[local_h - 1, local_w - 1].set(TileType.BRCORN)

    # Altar at (row=2, col=2) -- from .des ALTAR:(2,2) which is col=2, row=2
    local_map = local_map.at[2, 2].set(TileType.ALTAR)

    # Upstaircase at (row=2, col=1) -- from .des BRANCH:(1,2,1,2)
    local_map = local_map.at[2, 1].set(TileType.UPSTAIR)

    # Embed into full-size map
    game_map = jnp.full((static_params.map_height, static_params.map_width), TileType.VOID, dtype=jnp.int32)
    game_map = jax.lax.dynamic_update_slice(game_map, local_map, (0, 0))

    # Place 16 zombies randomly in rect (6,6)-(12,12) in .des coords
    # .des uses (col, row), so fillrect(6,6,12,12) = cols 6-12, rows 6-12
    # In our (row, col) format: rows 6-12, cols 6-12
    num_zombies = 16
    rng, rng_rows, rng_cols = jax.random.split(rng, 3)

    # Random positions in [6, 12] for both row and col (inclusive)
    zombie_rows = jax.random.randint(rng_rows, (num_zombies,), 6, 13)
    zombie_cols = jax.random.randint(rng_cols, (num_zombies,), 6, 13)

    # Build monster arrays: 16 zombies + 1 priest = 17 total
    # Zombies at indices 0-15, priest at index 16
    mon_rows = jnp.concatenate([zombie_rows, jnp.array([2])])
    mon_cols = jnp.concatenate([zombie_cols, jnp.array([3])])
    mon_positions = jnp.stack([mon_rows, mon_cols], axis=-1)  # [17, 2]

    mon_types = jnp.concatenate([
        jnp.full((num_zombies,), MonsterType.HUMAN_ZOMBIE, dtype=jnp.int32),
        jnp.array([MonsterType.PRIEST], dtype=jnp.int32),
    ])
    mon_health = jnp.concatenate([
        jnp.full((num_zombies,), MONSTER_MAX_HP[MonsterType.HUMAN_ZOMBIE], dtype=jnp.int32),
        jnp.array([MONSTER_MAX_HP[MonsterType.PRIEST]], dtype=jnp.int32),
    ])
    mon_mask = jnp.ones((max_m,), dtype=jnp.bool_)

    # Movement points: start at 0 (monsters accumulate via mcalcmove each turn)
    mon_movement = jnp.zeros((max_m,), dtype=jnp.int32)

    # Sleep: all monsters start awake (NetHack default -- no asleep attribute in .des)
    mon_sleeping = jnp.concatenate([
        jnp.zeros((num_zombies,), dtype=jnp.bool_),   # zombies start awake (NetHack default)
        jnp.zeros((1,), dtype=jnp.bool_),              # priest awake
    ])

    monsters = Monsters(
        position=mon_positions,
        type_id=mon_types,
        health=mon_health,
        mask=mon_mask,
        movement_points=mon_movement,
        is_sleeping=mon_sleeping,
    )

    # Player starts at BRANCH:(1,2,1,2) -> col=1, row=2 -> (row=2, col=1)
    player_position = jnp.array([2, 1], dtype=jnp.int32)

    # Player stats from role/race
    rng, rng_stats, state_rng = jax.random.split(rng, 3)
    player_stats = compute_initial_stats(rng_stats, params.role_id, params.race_id)

    # Empty sub-structures
    inventory = Inventory(
        item_ids=jnp.zeros(static_params.max_items, dtype=jnp.int32),
        item_mask=jnp.zeros(static_params.max_items, dtype=jnp.bool_),
    )
    ground_items = GroundItems(
        position=jnp.zeros((static_params.max_ground_items, 2), dtype=jnp.int32),
        type_id=jnp.zeros(static_params.max_ground_items, dtype=jnp.int32),
        mask=jnp.zeros(static_params.max_ground_items, dtype=jnp.bool_),
    )
    traps = Traps(
        position=jnp.zeros((static_params.max_traps, 2), dtype=jnp.int32),
        type_id=jnp.zeros(static_params.max_traps, dtype=jnp.int32),
        triggered=jnp.zeros(static_params.max_traps, dtype=jnp.bool_),
        mask=jnp.zeros(static_params.max_traps, dtype=jnp.bool_),
    )

    visible_map = compute_visible(player_position, game_map, static_params.map_height, static_params.map_width)

    state = CombatState(
        map=game_map,
        player_position=player_position,
        downstair_position=jnp.array([0, 0], dtype=jnp.int32),  # No downstair in ZombieHorde
        player_stats=player_stats,
        player_levitating=False,
        levitation_turns=0,
        player_has_key=False,
        inventory=inventory,
        monsters=monsters,
        traps=traps,
        ground_items=ground_items,
        seen_map=visible_map,
        visible_map=visible_map,
        timestep=jnp.int32(0),
        prev_action=0,
        terminal=False,
        state_rng=state_rng,
    )

    return state
