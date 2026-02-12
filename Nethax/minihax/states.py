"""Per-tier state dataclasses for minihax environments."""
from typing import Any
from flax import struct
import jax.numpy as jnp


# ============================================================================
# Shared sub-structures
# ============================================================================

@struct.dataclass
class Inventory:
    """Fixed-size inventory for items (Tier 2, 3)."""
    item_ids: jnp.ndarray       # [max_items] — ItemType enum values
    item_mask: jnp.ndarray      # [max_items] — slot occupied flag (bool)


@struct.dataclass
class GroundItems:
    """Items on the ground (Tier 2, 3)."""
    position: jnp.ndarray       # [max_ground_items, 2] — row, col
    type_id: jnp.ndarray        # [max_ground_items] — ItemType enum values
    mask: jnp.ndarray           # [max_ground_items] — exists flag (bool)


@struct.dataclass
class SimpleMonsters:
    """Simple monster arrays for Tier 2 (no movement points, no sleeping)."""
    position: jnp.ndarray       # [max_monsters, 2] — row, col
    type_id: jnp.ndarray        # [max_monsters] — MonsterType enum values
    health: jnp.ndarray         # [max_monsters] — current HP
    mask: jnp.ndarray           # [max_monsters] — alive flag (bool)


@struct.dataclass
class Monsters:
    """Full monster arrays with movement points (Tier 3, same as existing)."""
    position: jnp.ndarray       # [max_monsters, 2] — row, col
    type_id: jnp.ndarray        # [max_monsters] — MonsterType
    health: jnp.ndarray         # [max_monsters] — current HP
    mask: jnp.ndarray           # [max_monsters] — alive flag
    movement_points: jnp.ndarray  # [max_monsters] — accumulated movement (need 12 to act)
    is_sleeping: jnp.ndarray    # [max_monsters] — asleep until disturbed


@struct.dataclass
class Traps:
    """Fixed-size trap arrays (Tier 3)."""
    position: jnp.ndarray       # [max_traps, 2] — row, col
    type_id: jnp.ndarray        # [max_traps] — trap type (0=none, 1=board, 2=pit)
    triggered: jnp.ndarray      # [max_traps] — already triggered flag (bool)
    mask: jnp.ndarray           # [max_traps] — exists flag (bool)


# ============================================================================
# Tier 1: Navigation State (minimal)
# ============================================================================

@struct.dataclass
class NavigationState:
    """State for Tier 1 navigation environments (corridor, maze)."""
    map: jnp.ndarray             # [map_h, map_w] tile type IDs
    player_position: jnp.ndarray # [2] — row, col
    downstair_position: jnp.ndarray  # [2] — row, col (goal)
    ground_items: GroundItems    # [max_ground_items] — apples in ExploreMaze, empty elsewhere
    seen_map: jnp.ndarray        # [map_h, map_w] bool — tiles ever seen
    visible_map: jnp.ndarray     # [map_h, map_w] bool — tiles currently visible
    timestep: int
    prev_action: int
    terminal: bool
    state_rng: Any


# ============================================================================
# Tier 4: Sokoban State (boulders + pits as tiles)
# ============================================================================

@struct.dataclass
class SokobanState:
    """State for Tier 4 Sokoban environments."""
    map: jnp.ndarray             # [map_h, map_w] — boulders and pits ARE tile types
    player_position: jnp.ndarray # [2] — row, col
    downstair_position: jnp.ndarray  # [2] — row, col (goal)
    pits_remaining: int          # count of unfilled PIT tiles
    seen_map: jnp.ndarray        # [map_h, map_w] bool — tiles ever seen
    visible_map: jnp.ndarray     # [map_h, map_w] bool — tiles currently visible
    timestep: int
    prev_action: int
    terminal: bool
    state_rng: Any


# ============================================================================
# Tier 2: Hazard State (HP, items, simple monsters)
# ============================================================================

@struct.dataclass
class HazardState:
    """State for Tier 2 hazard environments (lava, items, simple monsters)."""
    map: jnp.ndarray
    player_position: jnp.ndarray
    downstair_position: jnp.ndarray
    player_hp: int
    player_max_hp: int
    player_levitating: bool
    levitation_turns: int
    inventory: Inventory
    monsters: SimpleMonsters
    ground_items: GroundItems
    seen_map: jnp.ndarray        # [map_h, map_w] bool — tiles ever seen
    visible_map: jnp.ndarray     # [map_h, map_w] bool — tiles currently visible
    timestep: int
    prev_action: int
    terminal: bool
    state_rng: Any


# ============================================================================
# Tier 3: Combat State (full combat, generalizes existing EnvState)
# ============================================================================

@struct.dataclass
class CombatState:
    """State for Tier 3 combat environments (quest, memento, zombie horde)."""
    map: jnp.ndarray
    player_position: jnp.ndarray
    downstair_position: jnp.ndarray
    player_hp: int
    player_max_hp: int
    player_xp: int
    player_xp_level: int
    player_ac: int
    player_strength: int
    player_levitating: bool
    levitation_turns: int
    player_has_key: bool
    inventory: Inventory
    monsters: Monsters
    traps: Traps
    ground_items: GroundItems
    seen_map: jnp.ndarray        # [map_h, map_w] bool — tiles ever seen
    visible_map: jnp.ndarray     # [map_h, map_w] bool — tiles currently visible
    score: int
    monsters_killed: int
    timestep: int
    prev_action: int
    terminal: bool
    state_rng: Any


# ============================================================================
# Shared EnvParams (runtime parameters)
# ============================================================================

@struct.dataclass
class EnvParams:
    """Runtime parameters shared across all tiers."""
    max_timesteps: int = 1500


# ============================================================================
# Per-tier StaticParams (compile-time, determines array shapes)
# ============================================================================

@struct.dataclass
class NavigationStaticParams:
    """Static params for Tier 1 navigation environments."""
    map_height: int = 21
    map_width: int = 79
    max_ground_items: int = 4   # ExploreMaze uses 4 apples; 0 would work but 4 keeps shapes uniform


@struct.dataclass
class SokobanStaticParams:
    """Static params for Tier 4 Sokoban environments."""
    map_height: int = 18
    map_width: int = 30


@struct.dataclass
class HazardStaticParams:
    """Static params for Tier 2 hazard environments."""
    map_height: int = 10
    map_width: int = 38
    max_monsters: int = 6
    max_items: int = 3
    max_ground_items: int = 5


@struct.dataclass
class CombatStaticParams:
    """Static params for Tier 3 combat environments."""
    map_height: int = 15
    map_width: int = 80
    max_monsters: int = 17
    max_items: int = 3
    max_ground_items: int = 5
    max_traps: int = 4
    # Temple-specific (ZombieHorde only, zeroed for others)
    has_temple: bool = False
    # Goal type: 0 = reach downstair (default), 1 = kill target monster
    goal_type: int = 0
    goal_monster_idx: int = 0   # Monster index in monsters array to kill (for goal_type=1)
