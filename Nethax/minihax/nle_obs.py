"""NLE-style dict observation builder for minihax environments.

Produces observations matching the MiniHack default keys:
  glyphs, chars, colors, specials, blstats, message
  + agent-centered crop variants (glyphs_crop, chars_crop, colors_crop, specials_crop)

Glyph space is compact (50 IDs, uint8) rather than full NLE (5991):
  0-22:  TileType enum values (VOID=0 is also the unseen/stone glyph)
  23-38: MonsterType + 23 offset
  39-48: ItemType + 39 offset
  49:    Player (@)
"""
import numpy as np
import jax
import jax.numpy as jnp

from Nethax.minihax.constants import (
    TileType, MonsterType, ItemType,
    NUM_TILE_TYPES, NUM_MONSTER_TYPES, NUM_ITEM_TYPES,
    TILE_CHARS, MONSTER_SYMBOLS,
)


# ============================================================================
# Glyph constants
# ============================================================================
GLYPH_TILE_OFFSET = 0
GLYPH_MONSTER_OFFSET = NUM_TILE_TYPES       # 23
GLYPH_ITEM_OFFSET = NUM_TILE_TYPES + NUM_MONSTER_TYPES  # 39
GLYPH_PLAYER = NUM_TILE_TYPES + NUM_MONSTER_TYPES + NUM_ITEM_TYPES  # 49
GLYPH_UNSEEN = int(TileType.VOID)           # 0
NUM_GLYPHS = GLYPH_PLAYER + 1               # 50

# Lookup tables use numpy (not jnp) to avoid tracer leaks when module is
# imported inside JIT scope. JAX implicitly converts them during tracing.

# Tile type -> glyph ID (identity since offset is 0)
TILE_GLYPH_TABLE = np.arange(NUM_TILE_TYPES, dtype=np.uint8)

# Monster type -> glyph ID
MONSTER_GLYPH_TABLE = np.arange(NUM_MONSTER_TYPES, dtype=np.uint8) + GLYPH_MONSTER_OFFSET

# Item type -> glyph ID
ITEM_GLYPH_TABLE = np.arange(NUM_ITEM_TYPES, dtype=np.uint8) + GLYPH_ITEM_OFFSET


# ============================================================================
# Glyph -> ASCII char lookup table
# ============================================================================
_glyph_to_char = [ord(' ')] * NUM_GLYPHS

# Tiles (0-22)
for tile_val, char in TILE_CHARS.items():
    _glyph_to_char[tile_val + GLYPH_TILE_OFFSET] = ord(char)

# Monsters (23-38)
for mon_val, char in MONSTER_SYMBOLS.items():
    _glyph_to_char[mon_val + GLYPH_MONSTER_OFFSET] = ord(char)

# Items (39-48)
_ITEM_CHARS = {
    int(ItemType.NONE): ' ',
    int(ItemType.POTION_LEVITATION): '!',
    int(ItemType.RING_LEVITATION): '=',
    int(ItemType.BOOTS_LEVITATION): '[',
    int(ItemType.WAND_COLD): '/',
    int(ItemType.FROST_HORN): '(',
    int(ItemType.WAND_DEATH): '/',
    int(ItemType.SKELETON_KEY): '(',
    int(ItemType.APPLE): '%',
    int(ItemType.GOLD): '$',
}
for item_val, char in _ITEM_CHARS.items():
    _glyph_to_char[item_val + GLYPH_ITEM_OFFSET] = ord(char)

# Player
_glyph_to_char[GLYPH_PLAYER] = ord('@')

GLYPH_TO_CHAR_TABLE = np.array(_glyph_to_char, dtype=np.uint8)


# ============================================================================
# Glyph -> NLE color code lookup table (0-15, terminal ANSI)
# ============================================================================
# 0=black, 1=red, 2=green, 3=brown, 4=blue, 5=magenta, 6=cyan, 7=gray
# 8=dark_gray, 9=orange, 10=bright_green, 11=yellow, 12=bright_blue,
# 13=bright_magenta, 14=bright_cyan, 15=white

_glyph_to_color = [7] * NUM_GLYPHS  # default gray

# Tiles
_TILE_COLORS = {
    int(TileType.VOID): 0,         # black (stone)
    int(TileType.FLOOR): 7,        # gray
    int(TileType.VWALL): 7,        # gray
    int(TileType.HWALL): 7,        # gray
    int(TileType.TLCORN): 7,
    int(TileType.TRCORN): 7,
    int(TileType.BLCORN): 7,
    int(TileType.BRCORN): 7,
    int(TileType.ALTAR): 7,
    int(TileType.UPSTAIR): 7,
    int(TileType.DOWNSTAIR): 7,
    int(TileType.LAVA): 9,         # orange
    int(TileType.CLOUD): 7,
    int(TileType.TREE): 2,         # green
    int(TileType.IRON_BARS): 6,    # cyan
    int(TileType.CORRIDOR): 7,
    int(TileType.DOOR_CLOSED): 3,  # brown
    int(TileType.DOOR_OPEN): 3,
    int(TileType.DOOR_LOCKED): 3,
    int(TileType.BOULDER): 7,
    int(TileType.PIT): 7,
    int(TileType.PIT_FILLED): 7,
    int(TileType.TRAP_BOARD): 7,
}
for tile_val, color in _TILE_COLORS.items():
    _glyph_to_color[tile_val + GLYPH_TILE_OFFSET] = color

# Monsters
_MONSTER_COLORS = {
    int(MonsterType.NONE): 0,
    int(MonsterType.HUMAN_ZOMBIE): 7,       # gray Z
    int(MonsterType.PRIEST): 15,            # white @
    int(MonsterType.GIANT_RAT): 3,          # brown r
    int(MonsterType.COCKATRICE): 11,        # yellow L (cockatrice)
    int(MonsterType.NAGA_HATCHLING): 1,     # red N
    int(MonsterType.MINOTAUR): 3,           # brown H
    int(MonsterType.OGRE): 3,               # brown O
    int(MonsterType.BABY_RED_DRAGON): 1,    # red D
    int(MonsterType.TROLL): 7,              # gray T
    int(MonsterType.BLUE_JELLY): 4,         # blue j
    int(MonsterType.SPOTTED_JELLY): 2,      # green j
    int(MonsterType.LICHEN): 10,            # bright green F
    int(MonsterType.RED_MOLD): 1,           # red F
    int(MonsterType.GREEN_MOLD): 2,         # green F
    int(MonsterType.GRID_BUG): 5,           # magenta x
}
for mon_val, color in _MONSTER_COLORS.items():
    _glyph_to_color[mon_val + GLYPH_MONSTER_OFFSET] = color

# Items
_ITEM_COLORS = {
    int(ItemType.NONE): 0,
    int(ItemType.POTION_LEVITATION): 4,     # blue !
    int(ItemType.RING_LEVITATION): 9,       # orange =
    int(ItemType.BOOTS_LEVITATION): 3,      # brown [
    int(ItemType.WAND_COLD): 4,             # blue /
    int(ItemType.FROST_HORN): 15,           # white (
    int(ItemType.WAND_DEATH): 0,            # black / (death)
    int(ItemType.SKELETON_KEY): 7,          # gray (
    int(ItemType.APPLE): 2,                 # green %
    int(ItemType.GOLD): 11,                 # yellow $
}
for item_val, color in _ITEM_COLORS.items():
    _glyph_to_color[item_val + GLYPH_ITEM_OFFSET] = color

# Player
_glyph_to_color[GLYPH_PLAYER] = 15  # white

GLYPH_TO_COLOR_TABLE = np.array(_glyph_to_color, dtype=np.uint8)


# ============================================================================
# blstats constants
# ============================================================================
BLSTATS_SIZE = 27

# Index names for documentation
BL_X = 0
BL_Y = 1
BL_STR25 = 2
BL_STR125 = 3
BL_DEX = 4
BL_CON = 5
BL_INT = 6
BL_WIS = 7
BL_CHA = 8
BL_SCORE = 9
BL_HP = 10
BL_HPMAX = 11
BL_DEPTH = 12
BL_GOLD = 13
BL_ENE = 14
BL_ENEMAX = 15
BL_AC = 16
BL_HD = 17
BL_XP = 18
BL_EXP = 19
BL_TIME = 20
BL_HUNGER = 21
BL_CAP = 22
BL_DNUM = 23
BL_DLEVEL = 24
# Minihax extensions
BL_PITS_REMAINING = 25
BL_MONSTERS_KILLED = 26

# Default crop size (MiniHack default)
DEFAULT_CROP_SIZE = 9


# ============================================================================
# Core observation building functions
# ============================================================================

def build_glyph_map(game_map, seen_map, visible_map, player_position,
                    monsters_position=None, monsters_type_id=None,
                    monsters_mask=None, max_monsters=0,
                    ground_items_position=None, ground_items_type_id=None,
                    ground_items_mask=None, max_ground_items=0,
                    traps_position=None, traps_mask=None,
                    traps_triggered=None, max_traps=0):
    """Build (H, W) glyph grid from game state.

    Layering order (later overwrites earlier):
    1. Base terrain glyphs (fog-of-war: unseen=VOID, seen=terrain)
    2. Ground items (visible only)
    3. Traps (seen only, active + not triggered)
    4. Monsters (visible only)
    5. Player
    """
    # Convert numpy lookup tables to JAX arrays inside JIT scope (safe here,
    # problematic only at module level where they'd leak tracers on import).
    _tile_tbl = jnp.array(TILE_GLYPH_TABLE)
    _item_tbl = jnp.array(ITEM_GLYPH_TABLE)
    _mon_tbl = jnp.array(MONSTER_GLYPH_TABLE)

    # Fog-of-war masked terrain
    fog_map = jnp.where(seen_map, game_map, TileType.VOID)
    glyphs = _tile_tbl[fog_map]  # (H, W) uint8

    # Overlay ground items (visible only)
    if max_ground_items > 0:
        def _place_item(g, i):
            pos = ground_items_position[i]
            exists = ground_items_mask[i]
            vis = visible_map[pos[0], pos[1]]
            item_glyph = _item_tbl[ground_items_type_id[i]]
            g = g.at[pos[0], pos[1]].set(
                jnp.where(exists & vis, item_glyph, g[pos[0], pos[1]])
            )
            return g, None

        glyphs, _ = jax.lax.scan(_place_item, glyphs, jnp.arange(max_ground_items))

    # Overlay traps (seen tiles, active + not triggered)
    if max_traps > 0:
        def _place_trap(g, i):
            pos = traps_position[i]
            active = traps_mask[i] & jnp.logical_not(traps_triggered[i])
            seen = seen_map[pos[0], pos[1]]
            # Use TRAP_BOARD glyph for all traps
            trap_glyph = jnp.uint8(int(TileType.TRAP_BOARD))
            g = g.at[pos[0], pos[1]].set(
                jnp.where(active & seen, trap_glyph, g[pos[0], pos[1]])
            )
            return g, None

        glyphs, _ = jax.lax.scan(_place_trap, glyphs, jnp.arange(max_traps))

    # Overlay monsters (visible only)
    if max_monsters > 0:
        def _place_monster(g, i):
            pos = monsters_position[i]
            alive = monsters_mask[i]
            vis = visible_map[pos[0], pos[1]]
            mon_glyph = _mon_tbl[monsters_type_id[i]]
            g = g.at[pos[0], pos[1]].set(
                jnp.where(alive & vis, mon_glyph, g[pos[0], pos[1]])
            )
            return g, None

        glyphs, _ = jax.lax.scan(_place_monster, glyphs, jnp.arange(max_monsters))

    # Player on top
    glyphs = glyphs.at[player_position[0], player_position[1]].set(
        jnp.uint8(GLYPH_PLAYER)
    )

    return glyphs


def build_char_map(glyphs):
    """Convert glyph grid to ASCII character grid."""
    return jnp.array(GLYPH_TO_CHAR_TABLE)[glyphs]


def build_color_map(glyphs):
    """Convert glyph grid to NLE color code grid."""
    return jnp.array(GLYPH_TO_COLOR_TABLE)[glyphs]


def build_crop(array_2d, player_position, crop_size, pad_value=0):
    """Extract crop_size x crop_size window centered on player.

    Uses jnp.pad + jax.lax.dynamic_slice for JIT compatibility.
    """
    half = crop_size // 2
    padded = jnp.pad(array_2d, half, mode='constant', constant_values=pad_value)
    # Player position in padded coordinates
    r = player_position[0]  # padding shifts everything by half
    c = player_position[1]
    return jax.lax.dynamic_slice(padded, (r, c), (crop_size, crop_size))


def _build_obs_dict(glyphs, player_position, crop_size):
    """Build the full observation dict from a glyph map."""
    chars = build_char_map(glyphs)
    colors = build_color_map(glyphs)
    specials = jnp.zeros_like(glyphs)
    message = jnp.zeros(256, dtype=jnp.uint8)

    return {
        "glyphs": glyphs,
        "chars": chars,
        "colors": colors,
        "specials": specials,
        "message": message,
        "glyphs_crop": build_crop(glyphs, player_position, crop_size, GLYPH_UNSEEN),
        "chars_crop": build_crop(chars, player_position, crop_size, ord(' ')),
        "colors_crop": build_crop(colors, player_position, crop_size, 0),
        "specials_crop": build_crop(specials, player_position, crop_size, 0),
    }


# ============================================================================
# Per-tier blstats builders
# ============================================================================

def build_blstats_navigation(state):
    """Tier 1 navigation: position + timestep."""
    bl = jnp.zeros(BLSTATS_SIZE, dtype=jnp.int32)
    bl = bl.at[BL_X].set(state.player_position[1])     # x = col (NLE convention)
    bl = bl.at[BL_Y].set(state.player_position[0])     # y = row
    bl = bl.at[BL_HP].set(16)                           # placeholder full health
    bl = bl.at[BL_HPMAX].set(16)
    bl = bl.at[BL_DEPTH].set(1)
    bl = bl.at[BL_TIME].set(state.timestep)
    return bl


def build_blstats_hazard(state):
    """Tier 2 hazard: + hp, levitation status."""
    bl = jnp.zeros(BLSTATS_SIZE, dtype=jnp.int32)
    bl = bl.at[BL_X].set(state.player_position[1])
    bl = bl.at[BL_Y].set(state.player_position[0])
    bl = bl.at[BL_HP].set(state.player_hp)
    bl = bl.at[BL_HPMAX].set(state.player_max_hp)
    bl = bl.at[BL_DEPTH].set(1)
    bl = bl.at[BL_TIME].set(state.timestep)
    return bl


def build_blstats_combat(state):
    """Tier 3 combat: full stats."""
    bl = jnp.zeros(BLSTATS_SIZE, dtype=jnp.int32)
    bl = bl.at[BL_X].set(state.player_position[1])
    bl = bl.at[BL_Y].set(state.player_position[0])
    bl = bl.at[BL_STR25].set(state.player_strength)
    bl = bl.at[BL_STR125].set(state.player_strength)
    bl = bl.at[BL_SCORE].set(state.score)
    bl = bl.at[BL_HP].set(state.player_hp)
    bl = bl.at[BL_HPMAX].set(state.player_max_hp)
    bl = bl.at[BL_DEPTH].set(1)
    bl = bl.at[BL_AC].set(state.player_ac)
    bl = bl.at[BL_XP].set(state.player_xp_level)
    bl = bl.at[BL_EXP].set(state.player_xp)
    bl = bl.at[BL_TIME].set(state.timestep)
    bl = bl.at[BL_MONSTERS_KILLED].set(state.monsters_killed)
    return bl


def build_blstats_zombie_horde(state):
    """ZombieHorde (legacy EnvState): full combat stats."""
    bl = jnp.zeros(BLSTATS_SIZE, dtype=jnp.int32)
    bl = bl.at[BL_X].set(state.player_position[1])
    bl = bl.at[BL_Y].set(state.player_position[0])
    bl = bl.at[BL_STR25].set(state.player_strength)
    bl = bl.at[BL_STR125].set(state.player_strength)
    bl = bl.at[BL_SCORE].set(state.score)
    bl = bl.at[BL_HP].set(state.player_hp)
    bl = bl.at[BL_HPMAX].set(state.player_max_hp)
    bl = bl.at[BL_DEPTH].set(1)
    bl = bl.at[BL_AC].set(state.player_ac)
    bl = bl.at[BL_XP].set(state.player_xp_level)
    bl = bl.at[BL_EXP].set(state.player_xp)
    bl = bl.at[BL_TIME].set(state.timestep)
    bl = bl.at[BL_MONSTERS_KILLED].set(state.monsters_killed)
    return bl


def build_blstats_sokoban(state):
    """Tier 4 sokoban: position, timestep, pits remaining."""
    bl = jnp.zeros(BLSTATS_SIZE, dtype=jnp.int32)
    bl = bl.at[BL_X].set(state.player_position[1])
    bl = bl.at[BL_Y].set(state.player_position[0])
    bl = bl.at[BL_HP].set(16)
    bl = bl.at[BL_HPMAX].set(16)
    bl = bl.at[BL_DEPTH].set(1)
    bl = bl.at[BL_TIME].set(state.timestep)
    bl = bl.at[BL_PITS_REMAINING].set(state.pits_remaining)
    return bl


# ============================================================================
# Per-tier top-level render functions
# ============================================================================

def render_nle_navigation(state, static_params, crop_size=DEFAULT_CROP_SIZE, prev_action=0):
    """Render Tier 1 navigation state as NLE-style dict."""
    glyphs = build_glyph_map(
        state.map, state.seen_map, state.visible_map, state.player_position,
        ground_items_position=state.ground_items.position,
        ground_items_type_id=state.ground_items.type_id,
        ground_items_mask=state.ground_items.mask,
        max_ground_items=static_params.max_ground_items,
    )
    obs = _build_obs_dict(glyphs, state.player_position, crop_size)
    obs["blstats"] = build_blstats_navigation(state)
    obs["prev_actions"] = jnp.int32(prev_action)
    return obs


def render_nle_hazard(state, static_params, crop_size=DEFAULT_CROP_SIZE, prev_action=0):
    """Render Tier 2 hazard state as NLE-style dict."""
    glyphs = build_glyph_map(
        state.map, state.seen_map, state.visible_map, state.player_position,
        monsters_position=state.monsters.position,
        monsters_type_id=state.monsters.type_id,
        monsters_mask=state.monsters.mask,
        max_monsters=static_params.max_monsters,
        ground_items_position=state.ground_items.position,
        ground_items_type_id=state.ground_items.type_id,
        ground_items_mask=state.ground_items.mask,
        max_ground_items=static_params.max_ground_items,
    )
    obs = _build_obs_dict(glyphs, state.player_position, crop_size)
    obs["blstats"] = build_blstats_hazard(state)
    obs["prev_actions"] = jnp.int32(prev_action)
    return obs


def render_nle_combat(state, static_params, crop_size=DEFAULT_CROP_SIZE, prev_action=0):
    """Render Tier 3 combat state as NLE-style dict."""
    glyphs = build_glyph_map(
        state.map, state.seen_map, state.visible_map, state.player_position,
        monsters_position=state.monsters.position,
        monsters_type_id=state.monsters.type_id,
        monsters_mask=state.monsters.mask,
        max_monsters=static_params.max_monsters,
        ground_items_position=state.ground_items.position,
        ground_items_type_id=state.ground_items.type_id,
        ground_items_mask=state.ground_items.mask,
        max_ground_items=static_params.max_ground_items,
        traps_position=state.traps.position,
        traps_mask=state.traps.mask,
        traps_triggered=state.traps.triggered,
        max_traps=static_params.max_traps,
    )
    obs = _build_obs_dict(glyphs, state.player_position, crop_size)
    obs["blstats"] = build_blstats_combat(state)
    obs["prev_actions"] = jnp.int32(prev_action)
    return obs


def render_nle_zombie_horde(state, static_params, crop_size=DEFAULT_CROP_SIZE, prev_action=0):
    """Render ZombieHorde (legacy EnvState) as NLE-style dict."""
    glyphs = build_glyph_map(
        state.map, state.seen_map, state.visible_map, state.player_position,
        monsters_position=state.monsters.position,
        monsters_type_id=state.monsters.type_id,
        monsters_mask=state.monsters.mask,
        max_monsters=static_params.max_monsters,
    )
    obs = _build_obs_dict(glyphs, state.player_position, crop_size)
    obs["blstats"] = build_blstats_zombie_horde(state)
    obs["prev_actions"] = jnp.int32(prev_action)
    return obs


def render_nle_sokoban(state, static_params, crop_size=DEFAULT_CROP_SIZE, prev_action=0):
    """Render Tier 4 Sokoban state as NLE-style dict."""
    glyphs = build_glyph_map(
        state.map, state.seen_map, state.visible_map, state.player_position,
    )
    obs = _build_obs_dict(glyphs, state.player_position, crop_size)
    obs["blstats"] = build_blstats_sokoban(state)
    obs["prev_actions"] = jnp.int32(prev_action)
    return obs


# ============================================================================
# Observation space descriptor (for observation_space() method)
# ============================================================================

def nle_observation_space(map_height, map_width, crop_size=DEFAULT_CROP_SIZE):
    """Return a dict describing the NLE observation space shapes and dtypes."""
    return {
        "glyphs": {"shape": (map_height, map_width), "dtype": jnp.uint8},
        "chars": {"shape": (map_height, map_width), "dtype": jnp.uint8},
        "colors": {"shape": (map_height, map_width), "dtype": jnp.uint8},
        "specials": {"shape": (map_height, map_width), "dtype": jnp.uint8},
        "blstats": {"shape": (BLSTATS_SIZE,), "dtype": jnp.int32},
        "message": {"shape": (256,), "dtype": jnp.uint8},
        "glyphs_crop": {"shape": (crop_size, crop_size), "dtype": jnp.uint8},
        "chars_crop": {"shape": (crop_size, crop_size), "dtype": jnp.uint8},
        "colors_crop": {"shape": (crop_size, crop_size), "dtype": jnp.uint8},
        "specials_crop": {"shape": (crop_size, crop_size), "dtype": jnp.uint8},
        "prev_actions": {"shape": (), "dtype": jnp.int32},
    }
