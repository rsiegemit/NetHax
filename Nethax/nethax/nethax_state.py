from typing import Tuple, Any

import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Item:
    """Represents a single item.

    Items are stored in fixed-size arrays. Empty slots have category=0 (NONE).
    """
    category: jnp.ndarray       # ItemCategory enum value
    type_id: jnp.ndarray        # Weapon/Armor/Potion/Scroll/etc type
    buc_status: jnp.ndarray     # BUCStatus (-1=unknown, 0=cursed, 1=uncursed, 2=blessed)
    enchantment: jnp.ndarray    # +/- enchantment level
    charges: jnp.ndarray        # Remaining charges (wands)
    identified: jnp.ndarray     # Whether the item is identified (bool)
    quantity: jnp.ndarray       # Stack count (arrows, gold, etc.)


@struct.dataclass
class Monsters:
    """Monster arrays for a single dungeon level.

    All arrays indexed by [level, monster_index].
    """
    position: jnp.ndarray       # [num_levels, max_monsters, 2] - row, col
    type_id: jnp.ndarray        # [num_levels, max_monsters] - MonsterType
    health: jnp.ndarray         # [num_levels, max_monsters] - current HP
    mask: jnp.ndarray           # [num_levels, max_monsters] - alive flag
    asleep: jnp.ndarray         # [num_levels, max_monsters] - sleeping flag
    confused: jnp.ndarray       # [num_levels, max_monsters] - confusion turns
    speed_counter: jnp.ndarray  # [num_levels, max_monsters] - for speed system


@struct.dataclass
class EnvState:
    # ---- Dungeon map ----
    map: jnp.ndarray             # [num_levels, map_h, map_w] tile type IDs
    item_map: jnp.ndarray        # [num_levels, map_h, map_w, max_items_per_tile] item presence
    explored: jnp.ndarray        # [num_levels, map_h, map_w] bool - has player seen this tile
    visible: jnp.ndarray         # [num_levels, map_h, map_w] bool - currently in FOV

    # ---- Stairs connections ----
    down_stairs: jnp.ndarray     # [num_levels, 2] - position of down staircase per level
    up_stairs: jnp.ndarray       # [num_levels, 2] - position of up staircase per level

    # ---- Player state ----
    player_position: jnp.ndarray # [2] - row, col
    player_level: int            # Current dungeon level (0-indexed)
    player_direction: int        # Last movement direction

    # Player vitals
    player_hp: int               # Current hit points
    player_max_hp: int           # Maximum hit points
    player_xp: int               # Experience points
    player_xp_level: int         # Experience level (1-30)
    player_ac: int               # Armor class (computed from equipment)
    player_strength: int         # Strength stat
    player_nutrition: int        # Hunger counter (decreases over time)

    # Status effects
    player_confused: int         # Turns of confusion remaining
    player_blind: int            # Turns of blindness remaining
    player_stunned: int          # Turns of stun remaining
    player_paralyzed: int        # Turns of paralysis remaining
    player_fast: int             # Turns of speed remaining
    player_invisible: int        # Turns of invisibility remaining

    # Intrinsics (permanent resistances)
    player_intrinsics: jnp.ndarray  # [NUM_INTRINSICS] bool

    # ---- Inventory ----
    inventory: Item              # Fixed-size item array [max_inventory_size]
    wielded_weapon: int          # Index into inventory (-1 = none)
    worn_armor: jnp.ndarray      # [num_armor_slots] indices into inventory (-1 = none)
    gold: int                    # Gold pieces

    # ---- Equipment slots ----
    # worn_armor covers: body, shield, helm, gloves, boots, cloak

    # ---- Ground items ----
    ground_items: Item           # [num_levels, map_h, map_w, max_items_per_tile]

    # ---- Monsters ----
    monsters: Monsters

    # ---- Identification knowledge (per-run randomization) ----
    potion_appearance_map: jnp.ndarray   # [NUM_POTION_TYPES] shuffled appearance IDs
    scroll_appearance_map: jnp.ndarray   # [NUM_SCROLL_TYPES] shuffled appearance IDs
    potion_identified: jnp.ndarray       # [NUM_POTION_TYPES] bool - type-level identification
    scroll_identified: jnp.ndarray       # [NUM_SCROLL_TYPES] bool - type-level identification

    # ---- Dungeon features ----
    fountains_used: jnp.ndarray  # [num_levels, map_h, map_w] bool - dried up fountains
    doors_state: jnp.ndarray     # [num_levels, map_h, map_w] - door open/closed/locked state
    traps_revealed: jnp.ndarray  # [num_levels, map_h, map_w] bool - revealed traps

    # ---- Tracking ----
    monsters_killed: int         # Total monsters killed
    achievements: jnp.ndarray    # [NUM_ACHIEVEMENTS] bool
    score: int                   # Running score

    state_rng: Any               # JAX PRNG key
    timestep: int                # Current turn number


@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    god_mode: bool = False
    hunger_rate: int = 1         # Nutrition loss per turn


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (40, 80)   # NetHack-style: taller than wide
    num_levels: int = 25                    # Dungeon depth

    # Monsters
    max_monsters: int = 20                  # Per level

    # Inventory
    max_inventory_size: int = 52            # a-zA-Z like NetHack
    num_armor_slots: int = 6                # body, shield, helm, gloves, boots, cloak
    max_items_per_tile: int = 5             # Max items stacked on one ground tile
