from typing import Tuple, Any
import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Monsters:
    """Monster arrays for ZombieHorde (single level, no level dimension)."""
    position: jnp.ndarray       # [max_monsters, 2] - row, col
    type_id: jnp.ndarray        # [max_monsters] - MonsterType
    health: jnp.ndarray         # [max_monsters] - current HP
    mask: jnp.ndarray           # [max_monsters] - alive flag
    movement_points: jnp.ndarray  # [max_monsters] - accumulated movement (need 12 to act)
    is_sleeping: jnp.ndarray    # [max_monsters] - asleep until disturbed


@struct.dataclass
class EnvState:
    # Map (single level)
    map: jnp.ndarray             # [map_h, map_w] tile type IDs

    # Player
    player_position: jnp.ndarray # [2] - row, col
    player_hp: int
    player_max_hp: int
    player_xp: int
    player_xp_level: int
    player_ac: int
    player_strength: int

    # Monsters
    monsters: Monsters

    # Visibility (fog of war)
    seen_map: jnp.ndarray        # [map_h, map_w] bool — tiles ever seen
    visible_map: jnp.ndarray     # [map_h, map_w] bool — tiles currently visible

    # Tracking
    score: int
    monsters_killed: int
    timestep: int
    terminal: bool

    state_rng: Any


@struct.dataclass
class EnvParams:
    max_timesteps: int = 1500


@struct.dataclass
class StaticEnvParams:
    map_height: int = 15      # From .des: 18 columns wide, 15 rows tall (including walls)
    map_width: int = 18
    max_monsters: int = 17    # 16 zombies + 1 temple priest


# ============================================================================
# Backward compatibility aliases
# ============================================================================
from Nethax.minihax.states import (
    NavigationState, SokobanState, HazardState, CombatState,
    NavigationStaticParams, SokobanStaticParams, HazardStaticParams, CombatStaticParams,
)

# Re-export for backward compatibility
# Monsters and EnvParams are kept as-is above for ZombieHorde
# CombatState will eventually replace EnvState
