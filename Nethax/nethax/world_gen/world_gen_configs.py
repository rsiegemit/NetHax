from flax import struct
import jax.numpy as jnp


@struct.dataclass
class DungeonLevelConfig:
    """Configuration for generating a dungeon level."""
    min_rooms: int = 5
    max_rooms: int = 8
    min_room_size: int = 4
    max_room_size: int = 10
    has_shop: bool = False
    shop_chance: float = 0.1
    fountain_chance: float = 0.15
    altar_chance: float = 0.05
    trap_chance: float = 0.1
    monster_density: float = 0.02   # Monsters per floor tile
    item_density: float = 0.01     # Items per floor tile


# Default configs per dungeon depth range
EARLY_DUNGEON = DungeonLevelConfig(
    min_rooms=4,
    max_rooms=7,
    min_room_size=4,
    max_room_size=8,
    shop_chance=0.15,
    fountain_chance=0.2,
    trap_chance=0.05,
    monster_density=0.015,
    item_density=0.01,
)

MID_DUNGEON = DungeonLevelConfig(
    min_rooms=5,
    max_rooms=8,
    min_room_size=4,
    max_room_size=10,
    shop_chance=0.1,
    fountain_chance=0.1,
    altar_chance=0.08,
    trap_chance=0.15,
    monster_density=0.025,
    item_density=0.012,
)

DEEP_DUNGEON = DungeonLevelConfig(
    min_rooms=5,
    max_rooms=9,
    min_room_size=3,
    max_room_size=12,
    shop_chance=0.05,
    fountain_chance=0.05,
    altar_chance=0.1,
    trap_chance=0.25,
    monster_density=0.04,
    item_density=0.015,
)
