"""World generators for Memento environments (easy, short, hard).

Memory-based environments: player sees a hint monster at the start, then must
remember which corridor to take at the end. Wrong corridor has a board trap
(instant death). Correct corridor has a grid bug (harmless).

memento_easy.des (80x9): Long corridor with upper/lower branch at end.
memento_short.des (15x9): Same concept but shorter.
memento_hard.des (80x9): 4-way branching with 2 hint monsters.

.des coordinates are (col, row) -- converted to (row, col) internally.
"""
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, MonsterType, MONSTER_MAX_HP,
)
from Nethax.minihax.primitives.visibility import compute_visible, compute_lit_map
from Nethax.minihax.world_gen.combat_common import (
    pad_map, empty_combat_state, parse_map,
)


# ============================================================================
# Memento Easy (80x9)
# ============================================================================

_MEMENTO_EASY_MAP = [
    "                                                           --------------",
    " ---                                                       |............|",
    " |.|                                                       |.|-----------",
    "|-F|--------------------------------------------------------.|           ",
    "|............................................................|           ",
    "------------------------------------------------------------.|           ",
    "                                                           |.|-----------",
    "                                                           |............|",
    "                                                           --------------",
]


def generate_memento_easy(rng, params, static_params):
    """Generate a MementoEasy environment state.

    Long corridor. Left side: small room with iron bars, hint monster inside.
    Right side: upper and lower rooms connected by vertical corridor.
    50% randomization: variant A or B.
    - Variant A: blue_jelly hint at (row=2,col=2), grid_bug at (row=1,col=71),
      board_trap at (row=7,col=67), downstair at (row=1,col=73)
    - Variant B: lichen hint at (row=2,col=2), grid_bug at (row=7,col=71),
      board_trap at (row=1,col=67), downstair at (row=7,col=73)
    """
    rng, rng_variant, rng_state = jax.random.split(rng, 3)

    game_map = parse_map(_MEMENTO_EASY_MAP)

    # Player starts at (row=4, col=1)
    player_pos = jnp.array([4, 1], dtype=jnp.int32)

    # Variant: 50% A vs B
    is_variant_a = jax.random.uniform(rng_variant, ()) < 0.5

    # Hint monster position: always at (row=2, col=2) behind iron bars
    hint_pos = jnp.array([2, 2], dtype=jnp.int32)
    # Hint monster type: blue_jelly (A) or lichen (B)
    hint_type = jnp.where(is_variant_a, MonsterType.BLUE_JELLY, MonsterType.LICHEN)

    # Grid bug position: upper corridor end (A) or lower corridor end (B)
    grid_pos = jnp.where(is_variant_a,
                         jnp.array([1, 71], dtype=jnp.int32),
                         jnp.array([7, 71], dtype=jnp.int32))

    # Board trap position: lower corridor (A) or upper corridor (B)
    trap_pos = jnp.where(is_variant_a,
                         jnp.array([7, 67], dtype=jnp.int32),
                         jnp.array([1, 67], dtype=jnp.int32))

    # No downstair — goal is to kill the grid bug (goal_type=1)
    stair_pos = jnp.array([0, 0], dtype=jnp.int32)  # dummy, unreachable

    padded_map = pad_map(game_map, static_params)

    # Monsters: [0] = hint monster, [1] = grid bug
    max_m = static_params.max_monsters
    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(hint_pos)
    mon_positions = mon_positions.at[1].set(grid_pos)

    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(hint_type)
    mon_types = mon_types.at[1].set(MonsterType.GRID_BUG)

    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[hint_type])
    mon_health = mon_health.at[1].set(MONSTER_MAX_HP[MonsterType.GRID_BUG])

    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)
    mon_mask = mon_mask.at[1].set(True)

    mon_sleeping = jnp.zeros(max_m, dtype=jnp.bool_)

    # Traps: [0] = board trap
    max_traps = static_params.max_traps
    trap_positions = jnp.zeros((max_traps, 2), dtype=jnp.int32)
    trap_positions = trap_positions.at[0].set(trap_pos)

    trap_types = jnp.zeros(max_traps, dtype=jnp.int32)
    trap_types = trap_types.at[0].set(1)  # Board trap

    trap_triggered = jnp.zeros(max_traps, dtype=jnp.bool_)
    trap_mask = jnp.zeros(max_traps, dtype=jnp.bool_)
    trap_mask = trap_mask.at[0].set(True)

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width, lit_map)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        monsters=state.monsters.replace(
            position=mon_positions,
            type_id=mon_types,
            health=mon_health,
            mask=mon_mask,
            is_sleeping=mon_sleeping,
        ),
        traps=state.traps.replace(
            position=trap_positions,
            type_id=trap_types,
            triggered=trap_triggered,
            mask=trap_mask,
        ),
    )
    return state


# ============================================================================
# Memento Short (15x9)
# ============================================================================

_MEMENTO_SHORT_MAP = [
    "       -------",
    " ---   |.....|",
    " |.|   |.|----",
    "|-F|----.|    ",
    "|........|    ",
    "--------.|    ",
    "       |.|----",
    "       |.....|",
    "       -------",
]


def generate_memento_short(rng, params, static_params):
    """Generate a MementoShort environment state.

    Same concept as memento_easy but shorter corridor (15 wide).
    """
    rng, rng_variant, rng_state = jax.random.split(rng, 3)

    game_map = parse_map(_MEMENTO_SHORT_MAP)

    # Player starts at (row=4, col=1)
    player_pos = jnp.array([4, 1], dtype=jnp.int32)

    # Variant: 50% A vs B
    is_variant_a = jax.random.uniform(rng_variant, ()) < 0.5

    # Hint monster position: (row=2, col=2) behind iron bars
    hint_pos = jnp.array([2, 2], dtype=jnp.int32)
    hint_type = jnp.where(is_variant_a, MonsterType.BLUE_JELLY, MonsterType.LICHEN)

    # Grid bug: upper end (A) or lower end (B)
    grid_pos = jnp.where(is_variant_a,
                         jnp.array([1, 11], dtype=jnp.int32),
                         jnp.array([7, 11], dtype=jnp.int32))

    # Board trap: lower (A) or upper (B)
    trap_pos = jnp.where(is_variant_a,
                         jnp.array([7, 9], dtype=jnp.int32),
                         jnp.array([1, 9], dtype=jnp.int32))

    # No downstair — goal is to kill the grid bug (goal_type=1)
    stair_pos = jnp.array([0, 0], dtype=jnp.int32)  # dummy, unreachable

    padded_map = pad_map(game_map, static_params)

    # Monsters
    max_m = static_params.max_monsters
    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(hint_pos)
    mon_positions = mon_positions.at[1].set(grid_pos)

    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(hint_type)
    mon_types = mon_types.at[1].set(MonsterType.GRID_BUG)

    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[hint_type])
    mon_health = mon_health.at[1].set(MONSTER_MAX_HP[MonsterType.GRID_BUG])

    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)
    mon_mask = mon_mask.at[1].set(True)

    # Traps
    max_traps = static_params.max_traps
    trap_positions = jnp.zeros((max_traps, 2), dtype=jnp.int32)
    trap_positions = trap_positions.at[0].set(trap_pos)

    trap_types = jnp.zeros(max_traps, dtype=jnp.int32)
    trap_types = trap_types.at[0].set(1)

    trap_mask = jnp.zeros(max_traps, dtype=jnp.bool_)
    trap_mask = trap_mask.at[0].set(True)

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width, lit_map)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        monsters=state.monsters.replace(
            position=mon_positions,
            type_id=mon_types,
            health=mon_health,
            mask=mon_mask,
        ),
        traps=state.traps.replace(
            position=trap_positions,
            type_id=trap_types,
            mask=trap_mask,
        ),
    )
    return state


# ============================================================================
# Memento Hard (80x9)
# ============================================================================

_MEMENTO_HARD_MAP = [
    "                                                           --------------",
    " ---                                                       |............|",
    " |.|                                                       |.-----------|",
    "--F|-------------------------------------------------------|............|",
    "|............................................................-----------|",
    "--F--------------------------------------------------------|............|",
    " |.|                                                       |.-----------|",
    " ---                                                       |............|",
    "                                                           --------------",
]


def generate_memento_hard(rng, params, static_params):
    """Generate a MementoHard environment state.

    4-way branching with 2 hint monsters, 3 board traps, 1 safe path.
    Two iron bars entries at rows 3 and 5 for two hint monsters.
    4 variants based on 2 nested 50% rolls.

    Each variant has:
    - 2 hint monsters (jelly type at row 3, mold type at row 5)
    - 3 board traps (on the 3 wrong corridors)
    - 1 grid bug (on the safe corridor)
    - downstair near the grid bug

    4-way branching: corridors at rows 1, 3, 5, 7. Two hint monsters behind
    iron bars at rows 2 and 6. 2 coin flips select the safe corridor.
    """
    rng, rng_v1, rng_v2, rng_state = jax.random.split(rng, 4)

    game_map = parse_map(_MEMENTO_HARD_MAP)

    # Player starts at (row=4, col=1)
    player_pos = jnp.array([4, 1], dtype=jnp.int32)

    # 2 coin flips determine which of 4 corridors is safe
    # Corridors are at rows 1, 3, 5, 7 in the right-side room
    flip1 = jax.random.uniform(rng_v1, ()) < 0.5
    flip2 = jax.random.uniform(rng_v2, ()) < 0.5

    # Safe corridor row (matches .des variant logic)
    # flip1=T, flip2=T -> row 1 | flip1=T, flip2=F -> row 3
    # flip1=F, flip2=T -> row 5 | flip1=F, flip2=F -> row 7
    safe_row = jnp.where(flip1,
                         jnp.where(flip2, jnp.int32(1), jnp.int32(3)),
                         jnp.where(flip2, jnp.int32(5), jnp.int32(7)))

    # Grid bug at safe corridor end
    grid_pos = jnp.array([safe_row, jnp.int32(71)], dtype=jnp.int32)

    # Trap rows: all 4 corridor rows except the safe one
    all_rows = jnp.array([1, 3, 5, 7], dtype=jnp.int32)
    trap_rows = jnp.where(all_rows == safe_row,
                          jnp.array([-1, -1, -1, -1], dtype=jnp.int32),
                          all_rows)
    # Compact: pick the 3 non-(-1) rows
    # Since exactly one matches, shift the remaining 3 to the front
    is_trap = trap_rows >= 0
    # Use cumsum trick to get first 3 valid rows
    trap_row0 = all_rows[jnp.where(safe_row == 1, 1, 0)]  # first non-safe row
    trap_row1 = all_rows[jnp.where(safe_row <= 3, 2, 1)]  # second
    trap_row2 = all_rows[jnp.where(safe_row == 7, 2, 3)]  # third

    trap0_pos = jnp.array([trap_row0, jnp.int32(67)], dtype=jnp.int32)
    trap1_pos = jnp.array([trap_row1, jnp.int32(67)], dtype=jnp.int32)
    trap2_pos = jnp.array([trap_row2, jnp.int32(67)], dtype=jnp.int32)

    # Hint monsters behind iron bars
    # Row 2 hint: blue_jelly (flip2=T) or spotted_jelly (flip2=F)
    hint1_type = jnp.where(flip2, MonsterType.BLUE_JELLY, MonsterType.SPOTTED_JELLY)
    hint1_pos = jnp.array([2, 2], dtype=jnp.int32)

    # Row 6 hint: red_mold (flip1=T) or green_mold (flip1=F)
    hint2_type = jnp.where(flip1, MonsterType.RED_MOLD, MonsterType.GREEN_MOLD)
    hint2_pos = jnp.array([6, 2], dtype=jnp.int32)

    # No downstair — goal is to kill the grid bug (goal_type=1)
    stair_pos = jnp.array([0, 0], dtype=jnp.int32)  # dummy, unreachable

    padded_map = pad_map(game_map, static_params)

    # Monsters: [0] = hint1, [1] = hint2, [2] = grid_bug
    max_m = static_params.max_monsters
    mon_positions = jnp.zeros((max_m, 2), dtype=jnp.int32)
    mon_positions = mon_positions.at[0].set(hint1_pos)
    mon_positions = mon_positions.at[1].set(hint2_pos)
    mon_positions = mon_positions.at[2].set(grid_pos)

    mon_types = jnp.zeros(max_m, dtype=jnp.int32)
    mon_types = mon_types.at[0].set(hint1_type)
    mon_types = mon_types.at[1].set(hint2_type)
    mon_types = mon_types.at[2].set(MonsterType.GRID_BUG)

    mon_health = jnp.zeros(max_m, dtype=jnp.int32)
    mon_health = mon_health.at[0].set(MONSTER_MAX_HP[hint1_type])
    mon_health = mon_health.at[1].set(MONSTER_MAX_HP[hint2_type])
    mon_health = mon_health.at[2].set(MONSTER_MAX_HP[MonsterType.GRID_BUG])

    mon_mask = jnp.zeros(max_m, dtype=jnp.bool_)
    mon_mask = mon_mask.at[0].set(True)
    mon_mask = mon_mask.at[1].set(True)
    mon_mask = mon_mask.at[2].set(True)

    # Traps: 3 board traps
    max_traps = static_params.max_traps
    trap_positions = jnp.zeros((max_traps, 2), dtype=jnp.int32)
    trap_positions = trap_positions.at[0].set(trap0_pos)
    trap_positions = trap_positions.at[1].set(trap1_pos)
    trap_positions = trap_positions.at[2].set(trap2_pos)

    trap_types = jnp.zeros(max_traps, dtype=jnp.int32)
    trap_types = trap_types.at[0].set(1)  # Board trap
    trap_types = trap_types.at[1].set(1)
    trap_types = trap_types.at[2].set(1)

    trap_mask = jnp.zeros(max_traps, dtype=jnp.bool_)
    trap_mask = trap_mask.at[0].set(True)
    trap_mask = trap_mask.at[1].set(True)
    trap_mask = trap_mask.at[2].set(True)

    lit_map = compute_lit_map(padded_map)
    visible_map = compute_visible(player_pos, padded_map, static_params.map_height, static_params.map_width, lit_map)
    state = empty_combat_state(static_params, rng_state)
    state = state.replace(
        map=padded_map,
        player_position=player_pos,
        downstair_position=stair_pos,
        seen_map=visible_map,
        visible_map=visible_map,
        lit_map=lit_map,
        monsters=state.monsters.replace(
            position=mon_positions,
            type_id=mon_types,
            health=mon_health,
            mask=mon_mask,
        ),
        traps=state.traps.replace(
            position=trap_positions,
            type_id=trap_types,
            mask=trap_mask,
        ),
    )
    return state
