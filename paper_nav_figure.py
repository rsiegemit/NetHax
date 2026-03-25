"""Generate paper-ready figures for Navigation tier environments.

Saves two separate figures (no titles/labels — intended for Figma compositing):
  - corridors_figure.png: Corridor2, Corridor3, Corridor5 side by side
  - mazes_figure.png: ExploreMazeEasy, ExploreMazeHard side by side
"""
import sys
import os
sys.path.insert(0, '/home/renos/nethax')

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from Nethax.nethax_env import make_minihax_env_from_name
from Nethax.tiles.renderer import load_tiles
from Nethax.minihax.pixel_renderer import render_pixels_no_monsters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CORRIDORS = [
    ("Corridor2", "Minihax-Corridor2-Pixels-v0", 42),
    ("Corridor3", "Minihax-Corridor3-Pixels-v0", 42),
    ("Corridor5", "Minihax-Corridor5-Pixels-v0", 42),
]

MAZES = [
    ("MazeEasy", "Minihax-ExploreMazeEasy-Pixels-v0", 42),
    ("MazeHard", "Minihax-ExploreMazeHard-Pixels-v0", 42),
]


def render_env_nofog(env_name, tiles_array, seed=42):
    """Reset an env and render the full map without fog of war."""
    env = make_minihax_env_from_name(env_name)
    params = env.default_params
    static_params = env.static_params if hasattr(env, 'static_params') else env.static_env_params

    rng = jax.random.PRNGKey(seed)
    _, state = env.reset(rng, params)

    map_h = static_params.map_height
    map_w = static_params.map_width
    state = state.replace(
        visible_map=jnp.ones((map_h, map_w), dtype=jnp.bool_),
        seen_map=jnp.ones((map_h, map_w), dtype=jnp.bool_),
    )

    pixels = render_pixels_no_monsters(state, static_params, tiles_array)
    return np.array(pixels)


def crop_void(img_np, tile_size=16):
    """Crop rows/cols that are entirely black (VOID tiles)."""
    row_has_content = img_np.max(axis=(1, 2)) > 0
    col_has_content = img_np.max(axis=(0, 2)) > 0

    rows = np.where(row_has_content)[0]
    cols = np.where(col_has_content)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img_np

    r0 = (rows[0] // tile_size) * tile_size
    r1 = ((rows[-1] // tile_size) + 1) * tile_size
    c0 = (cols[0] // tile_size) * tile_size
    c1 = ((cols[-1] // tile_size) + 1) * tile_size

    return img_np[r0:r1, c0:c1]


def compose_row(images, scale=3, padding=12):
    """Compose images into a single horizontal strip, vertically centered."""
    scaled = []
    for img in images:
        pil = Image.fromarray(img)
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
        scaled.append(np.array(pil))

    max_h = max(im.shape[0] for im in scaled)
    total_w = sum(im.shape[1] for im in scaled) + padding * (len(scaled) - 1)

    # Transparent background (RGBA) so Figma can handle it
    canvas = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))

    x = 0
    for im in scaled:
        h, w = im.shape[:2]
        y_offset = (max_h - h) // 2
        pil_img = Image.fromarray(im).convert("RGBA")
        canvas.paste(pil_img, (x, y_offset))
        x += w + padding

    return canvas


def main():
    print("Loading tiles...")
    tiles_array = load_tiles()

    # --- Corridors ---
    corridor_imgs = []
    for name, factory, seed in CORRIDORS:
        print(f"  Rendering {name}...")
        img = render_env_nofog(factory, tiles_array, seed)
        img = crop_void(img)
        corridor_imgs.append(img)
        print(f"    -> {img.shape[1]}x{img.shape[0]} px")

    corridors_fig = compose_row(corridor_imgs)
    out_path = os.path.join(SCRIPT_DIR, "corridors_figure.png")
    corridors_fig.save(out_path, dpi=(300, 300), optimize=True)
    print(f"\nSaved: {out_path}  ({corridors_fig.width}x{corridors_fig.height})")

    # --- Mazes ---
    maze_imgs = []
    for name, factory, seed in MAZES:
        print(f"  Rendering {name}...")
        img = render_env_nofog(factory, tiles_array, seed)
        img = crop_void(img)
        maze_imgs.append(img)
        print(f"    -> {img.shape[1]}x{img.shape[0]} px")

    mazes_fig = compose_row(maze_imgs)
    out_path = os.path.join(SCRIPT_DIR, "mazes_figure.png")
    mazes_fig.save(out_path, dpi=(300, 300), optimize=True)
    print(f"Saved: {out_path}  ({mazes_fig.width}x{mazes_fig.height})")


if __name__ == "__main__":
    main()
