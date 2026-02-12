"""World generation for Sokoban environments (Tier 4).

All maps are fixed (from .des files), not procedural.
Maps stored as string arrays, converted to JAX arrays at init.
"""
import jax.numpy as jnp
from Nethax.minihax.constants import TileType


# Character to TileType mapping for .des MAP parsing
CHAR_TO_TILE = {
    ' ': TileType.VOID,
    '.': TileType.FLOOR,
    '-': TileType.HWALL,
    '|': TileType.VWALL,
    '+': TileType.DOOR_CLOSED,
}


def _finalize_map(tile_map, branch_colrow, stair_colrow, static_params):
    """Convert .des (col,row) coords to (row,col), place downstair, and pad map.

    Args:
        tile_map: jnp.ndarray [H, W] tile map
        branch_colrow: (col, row) player start from .des BRANCH directive
        stair_colrow: (col, row) downstair position from .des STAIR directive
        static_params: StaticParams with map_height, map_width

    Returns:
        tile_map, player_pos [row, col], stair_pos [row, col]
    """
    # Place downstair on map (before padding, coords are in original map space)
    stair_row, stair_col = stair_colrow[1], stair_colrow[0]
    tile_map = tile_map.at[stair_row, stair_col].set(TileType.DOWNSTAIR)

    # Pad to static_params dimensions
    h, w = tile_map.shape
    if h < static_params.map_height or w < static_params.map_width:
        padded = jnp.full((static_params.map_height, static_params.map_width),
                         TileType.VOID, dtype=jnp.int32)
        padded = padded.at[:h, :w].set(tile_map)
        tile_map = padded

    # Convert (col, row) .des coords to (row, col) internal coords
    player_pos = jnp.array([branch_colrow[1], branch_colrow[0]], dtype=jnp.int32)
    stair_pos = jnp.array([stair_row, stair_col], dtype=jnp.int32)

    return tile_map, player_pos, stair_pos


def parse_map(map_lines):
    """Convert .des MAP strings to tile array.

    Args:
        map_lines: list of strings (map rows)

    Returns:
        jnp.ndarray [H, W] of TileType IDs
    """
    height = len(map_lines)
    width = max(len(line) for line in map_lines)

    # Build tile array
    tiles = []
    for line in map_lines:
        # Pad line to width
        padded = line.ljust(width)
        row = [CHAR_TO_TILE.get(ch, TileType.VOID) for ch in padded]
        tiles.append(row)

    return jnp.array(tiles, dtype=jnp.int32)


def place_objects(tile_map, boulders, pits):
    """Place boulders and pits on the map.

    Args:
        tile_map: jnp.ndarray [H, W]
        boulders: list of (col, row) positions (NetHack coords)
        pits: list of (col, row) positions (NetHack coords)

    Returns:
        new_map: jnp.ndarray [H, W] with boulders and pits
        pits_remaining: int
    """
    new_map = tile_map

    # Place boulders
    for col, row in boulders:
        new_map = new_map.at[row, col].set(TileType.BOULDER)

    # Place pits
    for col, row in pits:
        new_map = new_map.at[row, col].set(TileType.PIT)

    pits_remaining = len(pits)

    return new_map, pits_remaining


# ============================================================================
# Soko1a (14x13)
# ============================================================================
SOKO1A_MAP = [
    "------  ----- ",
    "|....|  |...| ",
    "|....----...| ",
    "|...........| ",
    "|..|-|.|-|..| ",
    "---------|.---",
    "|......|.....|",
    "|..----|.....|",
    "--.|   |.....|",
    " |.|---|.....|",
    " |...........|",
    " |..|---------",
    " ----         ",
]

SOKO1A_BOULDERS = [
    (2, 2), (2, 3), (10, 2), (9, 3), (10, 4),
    (8, 7), (9, 8), (9, 9), (8, 10), (10, 10),
]

SOKO1A_PITS = [
    (3, 6), (4, 6), (5, 6), (2, 8), (2, 9),
    (4, 10), (5, 10), (6, 10), (7, 10),
]

SOKO1A_STAIR = (6, 6)
SOKO1A_BRANCH = (6, 4)  # player start


def make_soko1a(rng, static_params):
    """Generate soko1a initial state."""
    tile_map = parse_map(SOKO1A_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO1A_BOULDERS, SOKO1A_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO1A_BRANCH, SOKO1A_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko1b (15x11)
# ============================================================================
SOKO1B_MAP = [
    "-------- ------",
    "|.|....|-|....|",
    "|.|-..........|",
    "|.||....|.....|",
    "|.||....|.....|",
    "|.|-----|.-----",
    "|.|    |......|",
    "|.-----|......|",
    "|.............|",
    "|..|---|......|",
    "----   --------",
]

SOKO1B_BOULDERS = [
    (5, 2), (6, 2), (6, 3), (7, 3), (9, 5),
    (10, 3), (11, 2), (12, 3), (7, 8), (8, 8),
    (9, 8), (10, 8),
]

SOKO1B_PITS = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (1, 7), (3, 8), (4, 8), (5, 8), (6, 8),
]

SOKO1B_STAIR = (1, 1)
SOKO1B_BRANCH = (3, 1)


def make_soko1b(rng, static_params):
    """Generate soko1b initial state."""
    tile_map = parse_map(SOKO1B_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO1B_BOULDERS, SOKO1B_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO1B_BRANCH, SOKO1B_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko2a (26x14)
# ============================================================================
SOKO2A_MAP = [
    " ----          -----------",
    "-|..|-------   |.........|",
    "|..........|   |.........|",
    "|..-----.-.|   |.........|",
    "|..|...|...|   |.........|",
    "|.........-|   |.........|",
    "|.......|..|   |.........|",
    "|.----..--.|   |.........|",
    "|........|.--  |.........|",
    "|.---.-.....------------+|",
    "|...|...-................|",
    "|.........----------------",
    "----|..|..|               ",
    "    -------               ",
]

SOKO2A_BOULDERS = [
    (2, 3), (8, 3), (9, 4), (2, 5), (4, 5),
    (9, 5), (2, 6), (5, 6), (6, 7), (3, 8),
    (7, 8), (5, 9), (10, 9), (7, 10), (10, 10), (3, 11),
]

SOKO2A_PITS = [
    (12, 10), (13, 10), (14, 10), (15, 10), (16, 10), (17, 10),
    (18, 10), (19, 10), (20, 10), (21, 10), (22, 10), (23, 10),
]

SOKO2A_STAIR = (20, 4)
SOKO2A_BRANCH = (3, 1)


def make_soko2a(rng, static_params):
    """Generate soko2a initial state."""
    tile_map = parse_map(SOKO2A_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO2A_BOULDERS, SOKO2A_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO2A_BRANCH, SOKO2A_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko2b (29x12)
# ============================================================================
SOKO2B_MAP = [
    "-----------       -----------",
    "|....|....|--     |.........|",
    "|....|......|     |.........|",
    "|.........|--     |.........|",
    "|....|....|       |.........|",
    "|-.---------      |.........|",
    "|....|.....|      |.........|",
    "|....|.....|      |.........|",
    "|..........|      |.........|",
    "|....|.....|---------------+|",
    "|....|......................|",
    "-----------------------------",
]

SOKO2B_BOULDERS = [
    (3, 2), (4, 2), (6, 2), (6, 3), (7, 2),
    (3, 6), (2, 7), (3, 7), (3, 8), (2, 9),
    (3, 9), (4, 9), (6, 7), (6, 9), (8, 7),
    (8, 10), (9, 8), (9, 9), (10, 7), (10, 10),
]

SOKO2B_PITS = [
    (12, 10), (13, 10), (14, 10), (15, 10), (16, 10), (17, 10),
    (18, 10), (19, 10), (20, 10), (21, 10), (22, 10), (23, 10),
    (24, 10), (25, 10), (26, 10),
]

SOKO2B_STAIR = (23, 4)
SOKO2B_BRANCH = (11, 2)


def make_soko2b(rng, static_params):
    """Generate soko2b initial state."""
    tile_map = parse_map(SOKO2B_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO2B_BOULDERS, SOKO2B_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO2B_BRANCH, SOKO2B_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko3a (20x12)
# ============================================================================
SOKO3A_MAP = [
    "--------------------",
    "|........|...|.....|",
    "|.....-..|.-.|.....|",
    "|..|.....|...|.....|",
    "|-.|..-..|.-.|.....|",
    "|...--.......|.....|",
    "|...|...-...-|.....|",
    "|...|..|...--|.....|",
    "|-..|..|----------+|",
    "|..................|",
    "|...|..|------------",
    "--------            ",
]

SOKO3A_BOULDERS = [
    (2, 2), (3, 2), (5, 3), (7, 3), (7, 2),
    (8, 2), (10, 3), (11, 3), (2, 7), (2, 8),
    (3, 9), (5, 7), (6, 6),
]

SOKO3A_PITS = [
    (8, 9), (9, 9), (10, 9), (11, 9), (12, 9),
    (13, 9), (14, 9), (15, 9), (16, 9), (17, 9),
]

SOKO3A_STAIR = (16, 4)
SOKO3A_BRANCH = (6, 10)


def make_soko3a(rng, static_params):
    """Generate soko3a initial state."""
    tile_map = parse_map(SOKO3A_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO3A_BOULDERS, SOKO3A_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO3A_BRANCH, SOKO3A_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko3b (20x13)
# ============================================================================
SOKO3B_MAP = [
    "  --------          ",
    "--|.|....|          ",
    "|........|----------",
    "|.-...-..|.|.......|",
    "|...-......|.......|",
    "|.-....|...|.......|",
    "|....-.--.-|.......|",
    "|..........|.......|",
    "|.--...|...|.......|",
    "|....-.|---|.......|",
    "--|....|----------+|",
    "  |................|",
    "  ------------------",
]

SOKO3B_BOULDERS = [
    (4, 2), (4, 3), (5, 3), (7, 3), (8, 3),
    (2, 4), (3, 4), (5, 5), (6, 6), (9, 6),
    (3, 7), (4, 7), (7, 7), (6, 9), (5, 10), (5, 11),
]

SOKO3B_PITS = [
    (7, 11), (8, 11), (9, 11), (10, 11), (11, 11),
    (12, 11), (13, 11), (14, 11), (15, 11), (16, 11), (17, 11),
]

SOKO3B_STAIR = (15, 6)
SOKO3B_BRANCH = (6, 11)


def make_soko3b(rng, static_params):
    """Generate soko3b initial state."""
    tile_map = parse_map(SOKO3B_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO3B_BOULDERS, SOKO3B_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO3B_BRANCH, SOKO3B_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko4a (26x17) - note: has 4 doors, stair is SHUFFLED
# ============================================================================
SOKO4A_MAP = [
    "--------------------------",
    "|........................|",
    "|.......|---------------.|",
    "-------.------         |.|",
    " |...........|         |.|",
    " |...........|         |.|",
    "--------.-----         |.|",
    "|............|         |.|",
    "|............|         |.|",
    "-----.--------   ------|.|",
    " |..........|  --|.....|.|",
    " |..........|  |.+.....|.|",
    " |.........|-  |-|.....|.|",
    "-------.----   |.+.....+.|",
    "|........|     |-|.....|--",
    "|........|     |.+.....|  ",
    "|...|-----     --|.....|  ",
    "-----            -------  ",
]

SOKO4A_BOULDERS = [
    (3, 5), (5, 5), (7, 5), (9, 5), (11, 5),
    (4, 7), (4, 8), (6, 7), (9, 7), (11, 7),
    (3, 12), (4, 10), (5, 12), (6, 10), (7, 11),
    (8, 10), (9, 12), (3, 14),
]

SOKO4A_PITS = [
    (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1),
    (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1),
]

SOKO4A_STAIR = (16, 11)  # Using first SHUFFLE position as default
SOKO4A_BRANCH = (1, 1)


def make_soko4a(rng, static_params):
    """Generate soko4a initial state."""
    tile_map = parse_map(SOKO4A_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO4A_BOULDERS, SOKO4A_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO4A_BRANCH, SOKO4A_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining


# ============================================================================
# Soko4b (26x17) - note: has 4 doors, stair is SHUFFLED
# ============================================================================
SOKO4B_MAP = [
    "  ------------------------",
    "  |......................|",
    "  |..-------------------.|",
    "----.|    -----        |.|",
    "|..|.--  --...|        |.|",
    "|.....|--|....|        |.|",
    "|.....|..|....|        |.|",
    "--....|......--        |.|",
    " |.......|...|   ------|.|",
    " |....|..|...| --|.....|.|",
    " |....|--|...| |.+.....|.|",
    " |.......|..-- |-|.....|.|",
    " ----....|.--  |.+.....+.|",
    "    ---.--.|   |-|.....|--",
    "     |.....|   |.+.....|  ",
    "     |..|..|   --|.....|  ",
    "     -------     -------  ",
]

SOKO4B_BOULDERS = [
    (4, 4), (2, 6), (3, 6), (4, 7), (5, 7),
    (2, 8), (5, 8), (3, 9), (4, 9), (3, 10),
    (5, 10), (6, 12), (7, 14), (11, 5), (12, 6),
    (10, 7), (11, 7), (10, 8), (12, 9), (11, 10),
]

SOKO4B_PITS = [
    (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1),
    (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1),
    (21, 1), (22, 1),
]

SOKO4B_STAIR = (16, 10)  # Using first SHUFFLE position as default
SOKO4B_BRANCH = (6, 15)


def make_soko4b(rng, static_params):
    """Generate soko4b initial state."""
    tile_map = parse_map(SOKO4B_MAP)
    tile_map, pits_remaining = place_objects(tile_map, SOKO4B_BOULDERS, SOKO4B_PITS)
    tile_map, player_pos, stair_pos = _finalize_map(
        tile_map, SOKO4B_BRANCH, SOKO4B_STAIR, static_params)
    return tile_map, player_pos, stair_pos, pits_remaining
