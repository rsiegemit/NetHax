"""Nethax-native procedural 16x16 tile atlas.

Heavy alternative to the vendor `tiles.npy` (which is the original NLE/NetHack
tile sprites, ported through nle).  This module emits a *parallel* atlas of the
same shape — ``(N_TILES, 16, 16, 3)`` uint8 — with clean, minimalist art
generated procedurally from the glyph metadata tables.

The vendor atlas is the default and remains unchanged.  Switch to this atlas
at render time with::

    from Nethax.tiles.native_atlas import load_tiles_nethax
    tiles = load_tiles_nethax()
    pixels = render_pixels(glyphs, GLYPH2TILE, tiles)

The art per tile is determined by what the tile represents:

- **Monster** tiles  → dark slate bg, a soft round vignette, the monster-class
                       character (a-z, A-Z, plus the specials &@';:~ etc.) in
                       its NetHack-canonical colour, bold.
- **Object** tiles   → dark navy bg, a small accent stripe across the bottom
                       in the object-class colour, the class character
                       (``)`` weapon, ``[`` armour, ``!`` potion, …) in light.
- **Cmap** tiles     → terrain-specific glyph (``|``, ``-``, ``.``, ``+``,
                       ``<``, ``>``, ``{``, ``}``, ``^``, …) rendered against
                       a dim background — walls in cyan, floor in slate dots,
                       doors in brown, stairs in white.
- **Explode/Zap**    → bright accent flash on black.
- **Warning**        → bold yellow ``!`` on black.
- **Invis/Body**     → dim grey placeholder.
- **Unused**         → pure black.

Generation is one-shot.  The resulting npy is cached at module import time and
written next to this file as ``tiles_nethax.npy``.

Vendor source citations (for the categorical mappings):
  vendor/nethack/include/defsym.h       — cmap symbol indices + chars
  vendor/nethack/include/monsym.h       — monster class symbols (S_*)
  vendor/nethack/src/objects.c          — object class chars + colours
  vendor/nle/include/glyph.h            — GLYPH_*_OFF + MAX_GLYPH
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Canonical NetHack 16-colour palette (vendor/nethack/win/tty/term_termcap.c).
# Index = NetHack colour ID (CLR_*).
# ---------------------------------------------------------------------------

NETHACK_PALETTE = [
    (0,   0,   0),     # 0  CLR_BLACK
    (170, 0,   0),     # 1  CLR_RED
    (0,   170, 0),     # 2  CLR_GREEN
    (170, 85,  0),     # 3  CLR_BROWN (dark yellow)
    (0,   0,   170),   # 4  CLR_BLUE
    (170, 0,   170),   # 5  CLR_MAGENTA
    (0,   170, 170),   # 6  CLR_CYAN
    (170, 170, 170),   # 7  CLR_GRAY
    (85,  85,  85),    # 8  NO_COLOR / CLR_DARK_GRAY
    (255, 85,  85),    # 9  CLR_ORANGE / bright red
    (85,  255, 85),    # 10 CLR_BRIGHT_GREEN
    (255, 255, 85),    # 11 CLR_YELLOW
    (85,  85,  255),   # 12 CLR_BRIGHT_BLUE
    (255, 85,  255),   # 13 CLR_BRIGHT_MAGENTA
    (85,  255, 255),   # 14 CLR_BRIGHT_CYAN
    (255, 255, 255),   # 15 CLR_WHITE
]


# ---------------------------------------------------------------------------
# Object class -> character.   (vendor/nethack/src/objects.c::def_oc_syms)
# ---------------------------------------------------------------------------

_OBJCLASS_CHAR = {
    0:  '*',   # RANDOM_CLASS
    1:  ' ',   # ILLOBJ_CLASS
    2:  ')',   # WEAPON_CLASS
    3:  '[',   # ARMOR_CLASS
    4:  '=',   # RING_CLASS
    5:  '"',   # AMULET_CLASS
    6:  '(',   # TOOL_CLASS
    7:  '%',   # FOOD_CLASS
    8:  '!',   # POTION_CLASS
    9:  '?',   # SCROLL_CLASS
    10: '+',   # SPBOOK_CLASS
    11: '/',   # WAND_CLASS
    12: '$',   # COIN_CLASS
    13: '*',   # GEM_CLASS
    14: '`',   # ROCK_CLASS
    15: '0',   # BALL_CLASS
    16: '_',   # CHAIN_CLASS
    17: '.',   # VENOM_CLASS
}


# ---------------------------------------------------------------------------
# Monster symbol -> character.   (vendor/nethack/include/monsym.h::S_*)
# Mirrors MonsterSymbol in Nethax/nethax/constants/monsters.py.
# ---------------------------------------------------------------------------

_MONSYM_CHAR = {
    1:  'a', 2:  'b', 3:  'c', 4:  'd', 5:  'e', 6:  'f', 7:  'g',
    8:  'h', 9:  'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n',
    15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u',
    22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z',
    27: 'A', 28: 'B', 29: 'C', 30: 'D', 31: 'E', 32: 'F', 33: 'G',
    34: 'H', 35: 'I', 36: 'J', 37: 'K', 38: 'L', 39: 'M', 40: 'N',
    41: 'O', 42: 'P', 43: 'Q', 44: 'R', 45: 'S', 46: 'T', 47: 'U',
    48: 'V', 49: 'W', 50: 'X', 51: 'Y', 52: 'Z',
    53: '@', 54: ' ', 55: "'", 56: '&', 57: ';', 58: ':', 59: '~',
    60: ']',
}


# ---------------------------------------------------------------------------
# Cmap symbol -> (char, color).  (vendor/nethack/include/defsym.h)
# Only the indices we typically render are listed; everything else falls back
# to a generic '.' floor tile.
# ---------------------------------------------------------------------------

_CMAP_GLYPHS = {
    0:  ('.',  7),   # S_stone (dark)
    1:  ('|',  6),   # S_vwall
    2:  ('-',  6),   # S_hwall
    3:  ('-',  6),   # S_tlcorn
    4:  ('-',  6),   # S_trcorn
    5:  ('-',  6),   # S_blcorn
    6:  ('-',  6),   # S_brcorn
    7:  ('+',  6),   # S_crwall
    8:  ('-',  6),   # S_tuwall
    9:  ('-',  6),   # S_tdwall
    10: ('|',  6),   # S_tlwall
    11: ('|',  6),   # S_trwall
    12: ('.',  7),   # S_ndoor (no door)
    13: ('|',  3),   # S_vodoor (open vertical door)
    14: ('-',  3),   # S_hodoor (open horizontal door)
    15: ('+',  3),   # S_vcdoor (closed door)
    16: ('+',  3),   # S_hcdoor
    17: ('#',  7),   # S_bars
    18: ('}', 12),   # S_tree (blue-ish)
    19: ('.',  8),   # S_room (lit floor)
    20: ('.',  8),   # S_darkroom
    21: ('#',  3),   # S_corr
    22: ('#',  6),   # S_litcorr
    23: ('<',  7),   # S_upstair
    24: ('>',  7),   # S_dnstair
    25: ('<', 11),   # S_upladder
    26: ('>', 11),   # S_dnladder
    27: ('_',  7),   # S_altar
    28: ('|',  3),   # S_grave
    29: '\\',         # placeholder (set below)
    30: ('{', 14),   # S_fountain
    31: ('}',  4),   # S_pool
    32: ('.', 11),   # S_ice
    33: ('}',  9),   # S_lava
    34: ('.',  6),   # S_vodbridge
    35: ('.',  6),   # S_hodbridge
    36: ('#',  3),   # S_vcdbridge
    37: ('#',  3),   # S_hcdbridge
    38: ('.',  8),   # S_air
    39: ('}',  4),   # S_cloud
    40: ('}',  4),   # S_water
    # Trap symbols (cmap idx ~41+) — render as caret in red.
}
_CMAP_GLYPHS[29] = ('\\', 7)  # S_throne


# ---------------------------------------------------------------------------
# Font + render helpers
# ---------------------------------------------------------------------------

_TILE_SIZE = 16

# Cache the font lookup.
_FONT_CACHE: dict[int, object] = {}


def _get_font(size: int):
    """Locate a usable monospace TTF on the host.  Falls back to PIL default."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/Library/Fonts/SF-Mono-Bold.otf",
        "/Library/Fonts/SF-Mono-Regular.otf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(path, size)
                return _FONT_CACHE[size]
            except OSError:
                continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _draw_char(img: "Image.Image", char: str, color: tuple[int, int, int],
               font_size: int = 13, dy: int = -1) -> None:
    """Centre-draw a single character onto the 16x16 tile image."""
    if char == '' or char == ' ':
        return
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (_TILE_SIZE - w) // 2 - bbox[0]
        y = (_TILE_SIZE - h) // 2 - bbox[1] + dy
    except Exception:
        x, y = 4, 1
    draw.text((x, y), char, fill=color, font=font)


def _tile_mon(symbol_idx: int, color_idx: int) -> np.ndarray:
    """Monster tile: dark slate bg + soft vignette + bold class letter."""
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (22, 28, 42))
    draw = ImageDraw.Draw(img)
    # Subtle circular vignette
    draw.ellipse((1, 1, 14, 14), outline=(36, 46, 66), width=1)
    char = _MONSYM_CHAR.get(symbol_idx, '?')
    color = NETHACK_PALETTE[color_idx % 16]
    _draw_char(img, char, color, font_size=12)
    return np.asarray(img, dtype=np.uint8)


def _tile_obj(class_idx: int, color_idx: int) -> np.ndarray:
    """Object tile: dark navy bg + bottom accent + class char."""
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (14, 19, 32))
    draw = ImageDraw.Draw(img)
    color = NETHACK_PALETTE[color_idx % 16]
    # Bottom accent stripe in the object colour
    draw.rectangle((1, 13, 14, 14), fill=color)
    char = _OBJCLASS_CHAR.get(class_idx, '*')
    _draw_char(img, char, (235, 235, 235), font_size=12, dy=-2)
    return np.asarray(img, dtype=np.uint8)


def _tile_cmap(cmap_idx: int) -> np.ndarray:
    """Terrain tile: dim bg + symbol-specific glyph + symbol-specific colour."""
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (5, 8, 16))
    info = _CMAP_GLYPHS.get(cmap_idx, ('.', 7))
    if isinstance(info, str):
        # corrupted entry; fall back
        info = ('.', 7)
    char, color_idx = info
    color = NETHACK_PALETTE[color_idx % 16]
    # Walls get a slightly lighter background to evoke stone
    if char in "|-+":
        img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (10, 16, 26))
    _draw_char(img, char, color, font_size=13)
    return np.asarray(img, dtype=np.uint8)


def _tile_explode(rel_idx: int) -> np.ndarray:
    """Explosion tile: black bg + bright orange burst character."""
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (4, 2, 0))
    _draw_char(img, '*', NETHACK_PALETTE[9], font_size=14)
    return np.asarray(img, dtype=np.uint8)


def _tile_zap(rel_idx: int) -> np.ndarray:
    """Zap-beam tile: black bg + bright cyan slash."""
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (0, 4, 8))
    # zap directions 0-3 → /\| _ etc — too granular; use '*' generic.
    char = "/-\\|"[rel_idx % 4]
    _draw_char(img, char, NETHACK_PALETTE[14], font_size=14)
    return np.asarray(img, dtype=np.uint8)


def _tile_warning(rel_idx: int) -> np.ndarray:
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (0, 0, 0))
    _draw_char(img, '!', NETHACK_PALETTE[11], font_size=14)
    return np.asarray(img, dtype=np.uint8)


def _tile_invisible() -> np.ndarray:
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (16, 16, 24))
    _draw_char(img, 'I', (140, 140, 160), font_size=12)
    return np.asarray(img, dtype=np.uint8)


def _tile_body() -> np.ndarray:
    img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (14, 19, 32))
    _draw_char(img, '%', (200, 80, 80), font_size=12)
    return np.asarray(img, dtype=np.uint8)


def _tile_unused() -> np.ndarray:
    return np.zeros((_TILE_SIZE, _TILE_SIZE, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Main atlas generator
# ---------------------------------------------------------------------------

def _build_atlas() -> np.ndarray:
    """Generate the full (N, 16, 16, 3) nethax-native atlas."""
    if not _HAS_PIL:
        raise RuntimeError(
            "Pillow (PIL) is required to generate the native tile atlas. "
            "Install with: pip install pillow"
        )

    # Local imports kept here so this module's import cost stays low when the
    # atlas npy is already cached on disk.
    from Nethax.tiles import GLYPH2TILE
    from Nethax.nethax.constants.glyphs import (
        GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_INVIS_OFF, GLYPH_DETECT_OFF,
        GLYPH_BODY_OFF, GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF, GLYPH_CMAP_OFF,
        GLYPH_EXPLODE_OFF, GLYPH_ZAP_OFF, GLYPH_SWALLOW_OFF,
        GLYPH_WARNING_OFF, GLYPH_STATUE_OFF,
    )
    from Nethax.nethax.constants.monsters import MONSTERS
    from Nethax.nethax.constants.objects import OBJECTS

    g = np.asarray(GLYPH2TILE)

    # Match the vendor atlas size (1082) for shape compatibility.
    N_TILES = 1082

    # For each tile index, pick the *first* glyph that maps to it as the
    # representative; that tells us the kind + char + colour to use.
    first_glyph = np.full(N_TILES, -1, dtype=np.int32)
    for gid in range(len(g)):
        tid = int(g[gid])
        if 0 <= tid < N_TILES and first_glyph[tid] == -1:
            first_glyph[tid] = gid

    atlas = np.zeros((N_TILES, _TILE_SIZE, _TILE_SIZE, 3), dtype=np.uint8)

    for tid in range(N_TILES):
        rep = int(first_glyph[tid])
        if rep < 0:
            atlas[tid] = _tile_unused()
            continue

        # Categorize by glyph offset.
        if rep < GLYPH_PET_OFF:
            mon_idx = rep - GLYPH_MON_OFF
            sym = int(MONSTERS[mon_idx].symbol)
            col = int(MONSTERS[mon_idx].color)
            atlas[tid] = _tile_mon(sym, col)

        elif rep < GLYPH_INVIS_OFF:
            mon_idx = rep - GLYPH_PET_OFF
            sym = int(MONSTERS[mon_idx].symbol)
            atlas[tid] = _tile_mon(sym, 2)  # pets coloured green

        elif rep < GLYPH_DETECT_OFF:
            atlas[tid] = _tile_invisible()

        elif rep < GLYPH_BODY_OFF:
            mon_idx = rep - GLYPH_DETECT_OFF
            sym = int(MONSTERS[mon_idx].symbol)
            col = int(MONSTERS[mon_idx].color)
            # detected monsters: dim version
            atlas[tid] = _tile_mon(sym, col)

        elif rep < GLYPH_RIDDEN_OFF:
            atlas[tid] = _tile_body()

        elif rep < GLYPH_OBJ_OFF:
            mon_idx = rep - GLYPH_RIDDEN_OFF
            sym = int(MONSTERS[mon_idx].symbol)
            col = int(MONSTERS[mon_idx].color)
            atlas[tid] = _tile_mon(sym, col)

        elif rep < GLYPH_CMAP_OFF:
            obj_idx = rep - GLYPH_OBJ_OFF
            obj = OBJECTS[obj_idx]
            class_idx = int(obj.class_) if obj is not None else 0
            col = int(getattr(obj, "color", 7)) if obj is not None else 7
            atlas[tid] = _tile_obj(class_idx, col)

        elif rep < GLYPH_EXPLODE_OFF:
            cmap_idx = rep - GLYPH_CMAP_OFF
            atlas[tid] = _tile_cmap(cmap_idx)

        elif rep < GLYPH_ZAP_OFF:
            atlas[tid] = _tile_explode(rep - GLYPH_EXPLODE_OFF)

        elif rep < GLYPH_SWALLOW_OFF:
            atlas[tid] = _tile_zap(rep - GLYPH_ZAP_OFF)

        elif rep < GLYPH_WARNING_OFF:
            atlas[tid] = _tile_zap(rep - GLYPH_SWALLOW_OFF)

        elif rep < GLYPH_STATUE_OFF:
            atlas[tid] = _tile_warning(rep - GLYPH_WARNING_OFF)

        else:
            # Statues — render as `\` against stone background.
            img = Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (28, 28, 32))
            _draw_char(img, '\\', (200, 200, 210), font_size=12)
            atlas[tid] = np.asarray(img, dtype=np.uint8)

    return atlas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ATLAS_PATH = Path(__file__).parent / "tiles_nethax.npy"


def generate_and_save(force: bool = False) -> Path:
    """Generate the atlas and save it next to this module.  Returns the path."""
    if _ATLAS_PATH.exists() and not force:
        return _ATLAS_PATH
    atlas = _build_atlas()
    np.save(_ATLAS_PATH, atlas)
    return _ATLAS_PATH


def load_tiles_nethax() -> np.ndarray:
    """Return the nethax-native atlas as a numpy uint8 array of shape
    ``(N_TILES, 16, 16, 3)``.  Generates + caches it on first call.
    """
    if not _ATLAS_PATH.exists():
        generate_and_save()
    return np.load(_ATLAS_PATH)


if __name__ == "__main__":
    path = generate_and_save(force=True)
    print(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
