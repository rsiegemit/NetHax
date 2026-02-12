import jax
import jax.numpy as jnp

from nethax.nethax.constants import *
from nethax.nethax.nethax_state import EnvState, StaticEnvParams


def render_nethax_symbolic(state: EnvState, static_params: StaticEnvParams = None):
    """Render the game state as a flat symbolic observation vector.

    NetHack uses a full top-down view of the entire level -- there is no
    scrolling viewport. The agent sees every explored tile (fog-of-war masks
    unexplored tiles as VOID). The observation encodes:

    1. Full level map one-hot encoded for tile types (fog-of-war applied)
    2. Monster positions on the current level
    3. Player stats (HP, AC, XP level, nutrition, etc.)
    4. Inventory summary
    5. Status effects and intrinsics
    6. Identification knowledge
    """
    if static_params is None:
        static_params = StaticEnvParams()

    map_h, map_w = static_params.map_size
    level = state.player_level

    # ---- Full level map with fog-of-war ----
    # Show explored tiles, mask unexplored as VOID
    level_map = state.map[level]
    explored = state.explored[level]

    # Apply fog of war: unexplored tiles appear as VOID
    visible_map = jnp.where(explored, level_map, TileType.VOID)

    # One-hot encode tile types
    tile_one_hot = jax.nn.one_hot(visible_map, NUM_TILE_TYPES)
    tile_flat = tile_one_hot.reshape(-1)

    # ---- Player position (one-hot on the map grid) ----
    player_pos = state.player_position
    player_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)
    player_map = player_map.at[player_pos[0], player_pos[1]].set(1.0)
    player_flat = player_map.reshape(-1)

    # ---- Monster presence on current level ----
    monster_map = jnp.zeros((map_h, map_w), dtype=jnp.float32)
    visible_current = state.visible[level]
    max_monsters = static_params.max_monsters

    def place_monster_on_map(m_map, i):
        pos = state.monsters.position[level, i]
        is_alive = state.monsters.mask[level, i]
        is_visible = visible_current[pos[0], pos[1]]
        should_show = jnp.logical_and(is_alive, is_visible)
        m_map = m_map.at[pos[0], pos[1]].set(
            jnp.where(should_show, 1.0, m_map[pos[0], pos[1]])
        )
        return m_map, None

    monster_map, _ = jax.lax.scan(place_monster_on_map, monster_map,
                                   jnp.arange(max_monsters))
    monster_flat = monster_map.reshape(-1)

    # ---- Player stats (normalized) ----
    stats = jnp.array([
        state.player_hp / jnp.maximum(state.player_max_hp, 1),
        state.player_max_hp / 50.0,
        state.player_xp_level / 30.0,
        state.player_ac / 20.0,
        state.player_strength / 25.0,
        state.player_nutrition / 2000.0,
        state.player_level / 25.0,
        state.timestep / 100000.0,
    ], dtype=jnp.float32)

    # ---- Status effects (binary) ----
    status = jnp.array([
        state.player_confused > 0,
        state.player_blind > 0,
        state.player_stunned > 0,
        state.player_paralyzed > 0,
        state.player_fast > 0,
        state.player_invisible > 0,
    ], dtype=jnp.float32)

    # ---- Intrinsics ----
    intrinsics = state.player_intrinsics.astype(jnp.float32)

    # ---- Hunger state ----
    hunger = get_hunger_state_vec(state.player_nutrition)

    # ---- Inventory summary ----
    inv_summary = jnp.array([
        (state.inventory.category == cat).sum()
        for cat in range(len(ItemCategory))
    ], dtype=jnp.float32) / static_params.max_inventory_size

    # ---- Identification knowledge ----
    id_knowledge = jnp.concatenate([
        state.potion_identified.astype(jnp.float32),
        state.scroll_identified.astype(jnp.float32),
    ])

    # Concatenate all
    obs = jnp.concatenate([
        tile_flat,
        player_flat,
        monster_flat,
        stats,
        status,
        intrinsics,
        hunger,
        inv_summary,
        id_knowledge,
    ])

    return obs


def get_hunger_state_vec(nutrition):
    """One-hot encode hunger state."""
    state = jnp.where(nutrition > 2000, 0,
            jnp.where(nutrition > 1000, 1,
            jnp.where(nutrition > 300, 2,
            jnp.where(nutrition > 150, 3,
            jnp.where(nutrition > 0, 4, 5)))))
    return jax.nn.one_hot(state, len(HungerState))


def render_nethax_text(state: EnvState, static_params: StaticEnvParams = None):
    """Render the game state as a human-readable text string.

    Full top-down view using traditional NetHack ASCII characters.
    Unexplored tiles are shown as spaces. The player is always '@'.
    """
    if static_params is None:
        static_params = StaticEnvParams()

    TILE_CHARS = {
        TileType.VOID: ' ',
        TileType.FLOOR: '.',
        TileType.CORRIDOR: '#',
        TileType.WALL: '-',
        TileType.CLOSED_DOOR: '+',
        TileType.OPEN_DOOR: '|',
        TileType.STAIRCASE_UP: '<',
        TileType.STAIRCASE_DOWN: '>',
        TileType.WATER: '~',
        TileType.LAVA: '~',
        TileType.ALTAR: '_',
        TileType.FOUNTAIN: '{',
        TileType.TRAP: '^',
        TileType.HIDDEN_TRAP: '.',
        TileType.THRONE: '\\',
        TileType.GRAVE: '|',
        TileType.SHOP_FLOOR: '.',
    }

    # This is for human play only, not JIT-compiled
    level = int(state.player_level)
    pos = (int(state.player_position[0]), int(state.player_position[1]))
    map_data = state.map[level]
    explored = state.explored[level]
    visible = state.visible[level]

    # Build monster position lookup (visible only)
    monster_positions = {}
    max_mons = static_params.max_monsters if static_params else 20
    for i in range(max_mons):
        if bool(state.monsters.mask[level, i]):
            r = int(state.monsters.position[level, i, 0])
            c = int(state.monsters.position[level, i, 1])
            if bool(visible[r, c]):
                mon_type = int(state.monsters.type_id[level, i])
                monster_positions[(r, c)] = MONSTER_SYMBOLS[mon_type]

    # Build ground item position lookup (explored tiles)
    ITEM_CHARS = {
        int(ItemCategory.WEAPON): ')',
        int(ItemCategory.ARMOR): '[',
        int(ItemCategory.FOOD): '%',
        int(ItemCategory.GOLD): '$',
        int(ItemCategory.POTION): '!',
        int(ItemCategory.SCROLL): '?',
        int(ItemCategory.WAND): '/',
        int(ItemCategory.RING): '=',
        int(ItemCategory.AMULET): '"',
    }
    item_positions = {}
    for r in range(static_params.map_size[0]):
        for c in range(static_params.map_size[1]):
            if bool(explored[r, c]):
                cat = int(state.ground_items.category[level, r, c, 0])
                if cat != 0:
                    item_positions[(r, c)] = ITEM_CHARS.get(cat, '*')

    lines = []
    for r in range(static_params.map_size[0]):
        row = ""
        for c in range(static_params.map_size[1]):
            if (r, c) == pos:
                row += "@"
            elif (r, c) in monster_positions:
                row += monster_positions[(r, c)]
            elif (r, c) in item_positions:
                row += item_positions[(r, c)]
            elif not bool(explored[r, c]):
                row += " "
            else:
                tile = int(map_data[r, c])
                row += TILE_CHARS.get(tile, '?')
        lines.append(row)

    # Status line (NetHack-style)
    hp_str = f"HP:{int(state.player_hp)}({int(state.player_max_hp)})"
    ac_str = f"AC:{int(state.player_ac)}"
    xp_str = f"Xp:{int(state.player_xp_level)}/{int(state.player_xp)}"
    dlvl_str = f"Dlvl:{level + 1}"
    gold_str = f"${int(state.gold)}"
    turns_str = f"T:{int(state.timestep)}"

    hunger = get_hunger_state_int(state.player_nutrition)
    hunger_names = ["Satiated", "", "Hungry", "Weak", "Fainting", "Starved"]
    hunger_str = hunger_names[int(hunger)]

    status_line = f"{dlvl_str}  {gold_str}  {hp_str}  AC:{int(state.player_ac)}  {xp_str}  {turns_str}"
    if hunger_str:
        status_line += f"  {hunger_str}"

    lines.append("")
    lines.append(status_line)

    return "\n".join(lines)


def get_hunger_state_int(nutrition):
    """Get hunger state as integer."""
    return jnp.where(nutrition > 2000, 0,
           jnp.where(nutrition > 1000, 1,
           jnp.where(nutrition > 300, 2,
           jnp.where(nutrition > 150, 3,
           jnp.where(nutrition > 0, 4, 5)))))
