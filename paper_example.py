"""Generate a long-corridor maze graphic with NetHack tile sprites.

Designed to fit as the leftmost panel of a 1x3 LaTeX subfigure.
Renders a single long corridor split into visible segments with "..." gaps.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILES_PATH = os.path.join(SCRIPT_DIR, "Nethax", "tiles", "tiles.npy")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "maze_graphic.png")

# ============================================================================
# Constants
# ============================================================================
TILE_SIZE = 16

VOID = 0
FLOOR = 1
VWALL = 2
HWALL = 3
TLCORN = 4
TRCORN = 5
BLCORN = 6
BRCORN = 7
DOWNSTAIR = 10
LAVA = 11

TILE_TYPE_SPRITES = np.array([
    850, 869, 851, 852, 853, 854, 855, 856,
    877, 873, 874, 884, 890, 868, 867, 872,
    865, 863, 865, 869, 902, 869, 895,
], dtype=np.int32)

SPRITE_PLAYER = 345

WALL_AUTOTILE = np.array([
    851, 852, 852, 852, 851, 853, 854, 859,
    851, 855, 856, 858, 851, 861, 860, 857,
], dtype=np.int32)

WALL_TYPE_LIST = [VWALL, HWALL, TLCORN, TRCORN, BLCORN, BRCORN]


# ============================================================================
# Helpers
# ============================================================================

def add_room(m, r, c, h, w):
    m[r + 1:r + h - 1, c + 1:c + w - 1] = FLOOR
    m[r, c + 1:c + w - 1] = HWALL
    m[r + h - 1, c + 1:c + w - 1] = HWALL
    m[r + 1:r + h - 1, c] = VWALL
    m[r + 1:r + h - 1, c + w - 1] = VWALL
    m[r, c] = TLCORN
    m[r, c + w - 1] = TRCORN
    m[r + h - 1, c] = BLCORN
    m[r + h - 1, c + w - 1] = BRCORN


def autotile_walls(game_map, tile_indices):
    wc = np.isin(game_map, WALL_TYPE_LIST + [16, 17])
    is_wall = np.isin(game_map, WALL_TYPE_LIST)
    up = np.pad(wc[:-1, :], ((1, 0), (0, 0)))
    down = np.pad(wc[1:, :], ((0, 1), (0, 0)))
    left = np.pad(wc[:, :-1], ((0, 0), (1, 0)))
    right = np.pad(wc[:, 1:], ((0, 0), (0, 1)))
    conn = (up.astype(np.int32) * 8
            + down.astype(np.int32) * 4
            + left.astype(np.int32) * 2
            + right.astype(np.int32))
    return np.where(is_wall, WALL_AUTOTILE[conn], tile_indices)


# ============================================================================
# Main
# ============================================================================

def main():
    tiles = np.load(TILES_PATH)

    # --- Build the full corridor ---
    # 17 rows tall (6 void + 5 room + 6 void) for ~4:3 aspect after compositing
    MAP_H = 17
    MAP_W = 25
    m = np.full((MAP_H, MAP_W), VOID, dtype=np.int32)

    # Corridor: 3-tile-high room centered vertically (rows 7-9, 1 floor row)
    add_room(m, 7, 0, 3, MAP_W)

    # Replace north and south walls with lava
    m[7, :] = LAVA
    m[9, :] = LAVA

    # Replace left and right walls with lava (only four corners stay as walls)
    m[8, 0] = LAVA
    m[8, MAP_W - 1] = LAVA

    # Player near left, staircase at bottom of final walkable position
    player_pos = (8, 2)
    m[9, 23] = DOWNSTAIR

    # --- Render full corridor to pixels ---
    ti = TILE_TYPE_SPRITES[m]
    ti = autotile_walls(m, ti)
    sprites = tiles[ti]
    full_px = sprites.transpose(0, 2, 1, 3, 4).reshape(
        MAP_H * TILE_SIZE, MAP_W * TILE_SIZE, 3
    ).astype(np.float32)

    # Overlay player sprite
    pr, pc = player_pos
    floor_s = tiles[TILE_TYPE_SPRITES[FLOOR]].astype(np.float32)
    player_s = tiles[SPRITE_PLAYER].astype(np.float32)
    alpha = (np.sum(player_s, axis=-1, keepdims=True) > 0).astype(np.float32)
    blended = player_s * alpha + floor_s * (1.0 - alpha)
    ry, cx = pr * TILE_SIZE, pc * TILE_SIZE
    full_px[ry:ry + TILE_SIZE, cx:cx + TILE_SIZE] = blended

    full_px = np.clip(full_px, 0, 255).astype(np.uint8)

    # --- Cut into 3 segments with 2 "..." gaps ---
    segments = [
        (0, 7),    # Seg 1: tiles 0-6  (player at tile 2)
        (9, 16),   # Seg 2: tiles 9-15
        (18, 25),  # Seg 3: tiles 18-24 (downstair at tile 22)
    ]
    gap_width_1x = 24   # pixels at 1x (will be 96px at 4x)
    fade_px = TILE_SIZE  # 1 tile of fade at gap-facing edges

    seg_images = []
    for i, (s, e) in enumerate(segments):
        seg = full_px[:, s * TILE_SIZE:e * TILE_SIZE, :].copy().astype(np.float32)
        w = seg.shape[1]

        # Fade right edge toward gap
        if i < len(segments) - 1:
            fade = np.linspace(1.0, 0.0, fade_px).reshape(1, -1, 1)
            seg[:, w - fade_px:w, :] *= fade

        # Fade left edge from gap
        if i > 0:
            fade = np.linspace(0.0, 1.0, fade_px).reshape(1, -1, 1)
            seg[:, :fade_px, :] *= fade

        seg_images.append(np.clip(seg, 0, 255).astype(np.uint8))

    # --- Compose final image ---
    num_gaps = len(segments) - 1
    total_w = sum(s.shape[1] for s in seg_images) + num_gaps * gap_width_1x
    final = np.zeros((MAP_H * TILE_SIZE, total_w, 3), dtype=np.uint8)

    x = 0
    gap_centers_x = []
    for i, seg in enumerate(seg_images):
        sw = seg.shape[1]
        final[:, x:x + sw, :] = seg
        x += sw
        if i < len(seg_images) - 1:
            gap_centers_x.append(x + gap_width_1x // 2)
            x += gap_width_1x

    # --- Scale up 4x (nearest-neighbor for pixel-art crispness) ---
    scale = 4
    img = Image.fromarray(final)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    # --- Draw "..." dots in each gap ---
    draw = ImageDraw.Draw(img)
    dot_radius = 5
    dot_spacing = 18
    dot_color = (190, 190, 190)

    for gcx in gap_centers_x:
        cx = gcx * scale
        cy = img.height // 2
        for offset in [-dot_spacing, 0, dot_spacing]:
            x0 = cx + offset - dot_radius
            y0 = cy - dot_radius
            x1 = cx + offset + dot_radius
            y1 = cy + dot_radius
            draw.ellipse([x0, y0, x1, y1], fill=dot_color)

    # --- Save at 300 DPI (suitable for print) ---
    img.save(OUTPUT_PATH, dpi=(300, 300), optimize=True)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size:  {img.width} x {img.height} px")
    print(f"Aspect ratio: {img.width / img.height:.2f}:1")


if __name__ == "__main__":
    main()