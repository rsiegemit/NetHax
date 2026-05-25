"""Tile types and tile-property tables.

Canonical source: vendor/nethack/include/rm.h (terrain symbols),
                  vendor/nethack/src/drawing.c (default_showsyms).

Status: Wave 2 — migrated from legacy Nethax/nethax/constants.py.

This module holds the JAX-side terrain enum used by the dungeon generator,
FOV, action dispatch, and observation builders.  NLE's `cmap` glyph offset
scheme is layered on top of these — see `Nethax/nethax/obs/nle_obs.py`
for the TileType → cmap_index lookup.
"""
from enum import IntEnum

import jax.numpy as jnp


class TileType(IntEnum):
    VOID = 0           # Out of bounds / unexplored
    FLOOR = 1          # Room floor (.)
    CORRIDOR = 2       # Corridor (#)
    WALL = 3           # Wall (|-)
    CLOSED_DOOR = 4    # Closed door (+)
    OPEN_DOOR = 5      # Open door (|)
    STAIRCASE_UP = 6   # Upstairs (<)
    STAIRCASE_DOWN = 7 # Downstairs (>)
    WATER = 8          # Pool / moat (~)
    LAVA = 9           # Lava (~)
    ALTAR = 10         # Altar (_)
    FOUNTAIN = 11      # Fountain ({)
    TRAP = 12          # Known trap (^)
    HIDDEN_TRAP = 13   # Hidden trap (looks like floor)
    THRONE = 14        # Throne (\\)
    GRAVE = 15         # Grave (|)
    SHOP_FLOOR = 16    # Shop tile
    DRAWBRIDGE_UP = 17 # Drawbridge raised — castle bottleneck.
                       # Citation: vendor/nethack/include/rm.h line 75
                       # (DRAWBRIDGE_UP = 19 in vendor; we use 17 to stay
                       # contiguous in our local enum).
    ICE_FLOOR = 18     # Ice floor — Asmodeus's lair cold theme.
                       # Citation: vendor/nethack/include/rm.h line 89 (ICE = 33).
    POOL = 19          # Acid / swamp pool — Juiblex / Demogorgon lairs.
                       # Citation: vendor/nethack/include/rm.h line 72 (POOL = 16).
    TREE = 20          # Tree (T) — used in big-room random terrain overlays
                       # and elsewhere.  Walkable.
                       # Citation: vendor/nethack/include/rm.h TREE.
    HOLE = 21          # Hole in the floor (^), descend one level if walked on.
                       # Created by WAN_DIGGING down-zap and pickaxe down-dig.
                       # Citation: vendor/nethack/src/dig.c::digactualhole line 640;
                       #            vendor/nethack/src/dig.c::zap_dig line 1548.
    SINK = 22          # Sink ({) — kicking and applying rings have effects.
                       # Citation: vendor/nethack/include/rm.h line 81 (SINK = 30);
                       #            vendor/nethack/src/dokick.c::kick_sink;
                       #            vendor/nethack/src/do_wear.c::dosinkring.
                       # We use local index 22 to stay contiguous with the
                       # other TileType entries; VendorTileType.SINK (line
                       # 102 below) preserves the vendor 30.


NUM_TILE_TYPES: int = len(TileType)


# ---------------------------------------------------------------------------
# Vendor-faithful tile-type enum (Wave 6 closing-audit parity).
#
# Mirrors ``enum levl_typ_types`` in vendor/nethack/include/rm.h lines 55-94
# byte-for-byte.  Our internal ``TileType`` above is a smaller, locally
# numbered enum kept for backward compatibility with ~600 existing JAX call
# sites; ``VendorTileType`` is the ground-truth mapping used for parity tests
# and any code that needs to round-trip with the vendor ``levl.typ`` field.
#
# Cite: vendor/nethack/include/rm.h::levl_typ_types (lines 55-94).
# ---------------------------------------------------------------------------
class VendorTileType(IntEnum):
    """Byte-exact mirror of vendor levl_typ_types.

    Source: vendor/nethack/include/rm.h lines 55-94.
    """
    STONE           = 0
    VWALL           = 1
    HWALL           = 2
    TLCORNER        = 3
    TRCORNER        = 4
    BLCORNER        = 5
    BRCORNER        = 6
    CROSSWALL       = 7
    TUWALL          = 8
    TDWALL          = 9
    TLWALL          = 10
    TRWALL          = 11
    DBWALL          = 12
    TREE            = 13
    SDOOR           = 14
    SCORR           = 15
    POOL            = 16
    MOAT            = 17
    WATER           = 18
    DRAWBRIDGE_UP   = 19
    LAVAPOOL        = 20
    LAVAWALL        = 21
    IRONBARS        = 22
    DOOR            = 23
    CORR            = 24
    ROOM            = 25
    STAIRS          = 26
    LADDER          = 27
    FOUNTAIN        = 28
    THRONE          = 29
    SINK            = 30
    GRAVE           = 31
    ALTAR           = 32
    ICE             = 33
    DRAWBRIDGE_DOWN = 34
    AIR             = 35
    CLOUD           = 36


# Vendor MAX_TYPE = 37 (one past CLOUD).  Used in parity tests.
VENDOR_MAX_TYPE: int = 37


# Tiles that block movement.
SOLID_TILES = jnp.array(
    [TileType.VOID, TileType.WALL, TileType.CLOSED_DOOR],
    dtype=jnp.int32,
)

# Tiles that block line of sight (default opacity table).
# vendor/nethack/src/vision.c:167-168, 760 — IS_DOOR with closed/locked/trapped
# AND secret-door (SDOOR) and secret corridor (SCORR) are all opaque.
# The internal TileType enum here does NOT carry SDOOR/SCORR; those live in
# VendorTileType (rm.h-exact) below. Secret doors that aren't yet discovered
# are represented at the internal-tile layer as WALL (vendor's display
# behavior), so the existing WALL entry already blocks LOS for them.
OPAQUE_TILES = jnp.array(
    [TileType.VOID, TileType.WALL, TileType.CLOSED_DOOR],
    dtype=jnp.int32,
)
