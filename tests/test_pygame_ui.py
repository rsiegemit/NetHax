"""Headless smoke test for the Nethax Pygame UI.

Runs ~5 env steps using SDL_VIDEODRIVER=dummy so no display is required.
Skipped automatically if pygame is not installed.
"""

import os
import pytest

# Headless SDL before any pygame import.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

pygame = pytest.importorskip("pygame", reason="pygame not installed")

import jax
import jax.numpy as jnp

from Nethax.nethax.env import NethaxEnv
from Nethax.tiles import render_pixels, GLYPH2TILE, TILE_SIZE
from Nethax.tiles.renderer import load_tiles
from Nethax.nethax.constants.actions import ACTIONS, Action


def test_module_imports():
    """The UI module must import without error."""
    from Nethax.ui import pygame_app  # noqa: F401


def test_render_pixels_from_obs():
    """render_pixels produces the expected shape from a real observation."""
    import numpy as np

    tiles_array = load_tiles()
    env = NethaxEnv()
    rng = jax.random.PRNGKey(42)
    state, obs = env.reset(rng)

    glyphs = np.asarray(obs["glyphs"])          # (21, 79)
    glyphs_80 = np.pad(glyphs, ((0, 0), (0, 1)), constant_values=0)  # (21, 80)

    pixels_jax = render_pixels(
        jnp.array(glyphs_80, dtype=jnp.int16),
        GLYPH2TILE,
        tiles_array,
    )
    pixels = np.asarray(pixels_jax)

    assert pixels.shape == (MAP_ROWS * TILE_SIZE, 80 * TILE_SIZE, 3)
    assert pixels.dtype == np.uint8


MAP_ROWS = 21


def test_five_steps_no_crash():
    """Run 5 env steps and assert no exception is raised."""
    env = NethaxEnv()
    rng = jax.random.PRNGKey(7)
    state, obs = env.reset(rng)

    # Use WAIT action (value = ord('.') = 46) for all steps.
    wait_action = jnp.int32(int(Action.WAIT))

    for _ in range(5):
        rng, step_rng = jax.random.split(rng)
        state, obs, reward, done, info = env.step(state, wait_action, step_rng)
        assert obs["blstats"].shape == (27,)
        assert obs["glyphs"].shape == (21, 79)


def test_keymap_covers_cardinal_directions():
    """Key map must include all 4 cardinal direction actions."""
    from Nethax.ui.pygame_app import _build_keymap

    key_map, _ = _build_keymap()
    mapped_actions = set(key_map.values())

    for action in (Action.COMPASS_N, Action.COMPASS_S, Action.COMPASS_E, Action.COMPASS_W):
        assert int(action) in mapped_actions, f"{action.name} missing from key_map"


def test_headless_pygame_window():
    """Open a dummy Pygame window, blit one frame, and close without error."""
    import numpy as np
    from Nethax.ui.pygame_app import (
        WINDOW_W, WINDOW_H, TILE_PANE_H,
        _render_tile_pane, _draw_status_panel,
    )

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    font = pygame.font.SysFont("monospace", 15)
    font_large = pygame.font.SysFont("monospace", 24, bold=True)

    tiles_array = load_tiles()
    env = NethaxEnv()
    rng = jax.random.PRNGKey(99)
    state, obs = env.reset(rng)

    pixels = _render_tile_pane(obs, tiles_array)
    tile_surface = pygame.surfarray.make_surface(pixels.transpose(1, 0, 2))
    screen.blit(tile_surface, (0, 0))
    _draw_status_panel(screen, font, font_large, obs, "WAIT", TILE_PANE_H)
    pygame.display.flip()

    pygame.quit()
