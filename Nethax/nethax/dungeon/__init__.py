"""Nethax dungeon-generation subsystem.

Wave 2 — re-exports key public types for the dungeon package.

Citation: vendor/nethack/src/dungeon.c, vendor/nethack/include/dungeon.h,
          vendor/nethack/include/mkroom.h
"""

from Nethax.nethax.dungeon.rooms import (
    Room,
    RoomType,
    generate_rooms,
    carve_rooms_into_terrain,
    connect_rooms,
    MAX_ROOMS_PER_LEVEL,
)
from Nethax.nethax.dungeon.mazes import generate_maze_kruskal, generate_maze_perfect, generate_maze_dla
from Nethax.nethax.dungeon.corridors import connect_segments, place_doors
from Nethax.nethax.dungeon.branches import (
    Branch,
    BranchInfo,
    BranchConnectionType,
    BRANCH_TABLE,
    DungeonState,
    current_dungeon_level,
    traverse_stair,
    enter_branch,
    generate_main_branch_l1,
    generate_main_branch_l1_with_features,
    MAP_H,
    MAP_W,
    N_BRANCHES,
    MAX_LEVELS_PER_BRANCH,
)
from Nethax.nethax.dungeon.special_levels import SpecialLevel, generate_special_level
from Nethax.nethax.dungeon.level_memory import LevelMemoryState, make_empty_level_memory, enter_level, leave_level

__all__ = [
    # rooms
    "Room",
    "RoomType",
    "generate_rooms",
    "carve_rooms_into_terrain",
    "connect_rooms",
    "MAX_ROOMS_PER_LEVEL",
    # mazes
    "generate_maze_kruskal",
    "generate_maze_perfect",
    "generate_maze_dla",
    # corridors
    "connect_segments",
    "place_doors",
    # branches
    "Branch",
    "BranchInfo",
    "BranchConnectionType",
    "BRANCH_TABLE",
    "DungeonState",
    "current_dungeon_level",
    "traverse_stair",
    "enter_branch",
    "generate_main_branch_l1",
    "generate_main_branch_l1_with_features",
    "MAP_H",
    "MAP_W",
    "N_BRANCHES",
    "MAX_LEVELS_PER_BRANCH",
    # special levels
    "SpecialLevel",
    "generate_special_level",
    # level memory
    "LevelMemoryState",
    "make_empty_level_memory",
    "enter_level",
    "leave_level",
]
