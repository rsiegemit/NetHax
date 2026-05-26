"""nethax observation builders.

Public API for all observation formats. RL agents trained on NLE should use
build_nle_observation (or empty_nle_observation for shape/dtype inspection)
and expect identical keys, shapes, and dtypes to the original NLE environment.
"""

from Nethax.nethax.obs.nle_obs import (
    NLE_OBSERVATION_DTYPES,
    NLE_OBSERVATION_KEYS,
    NLE_OBSERVATION_SHAPES,
    build_blstats,
    build_glyphs,
    build_inventory_strings,
    build_message,
    build_nle_observation,
    build_tty,
    empty_nle_observation,
)
from Nethax.nethax.obs.strength_format import format_strength
from Nethax.nethax.obs.rank_titles import rank_title
from Nethax.nethax.obs.pixel_obs import (
    MAP_H,
    MAP_W,
    PIXEL_OBS_SHAPE,
    TILE_PX,
    build_pixel_observation,
)
from Nethax.nethax.obs.symbolic_obs import (
    SYMBOLIC_OBS_DIM,
    build_symbolic_observation,
)
from Nethax.nethax.obs.text_obs import (
    TERM_COLS,
    TERM_ROWS,
    TEXT_OBS_SHAPE,
    build_text_observation,
)

__all__ = [
    # NLE-parity
    "NLE_OBSERVATION_KEYS",
    "NLE_OBSERVATION_SHAPES",
    "NLE_OBSERVATION_DTYPES",
    "empty_nle_observation",
    "build_nle_observation",
    "build_glyphs",
    "build_blstats",
    "build_message",
    "build_inventory_strings",
    "build_tty",
    "format_strength",
    "rank_title",
    # Symbolic
    "SYMBOLIC_OBS_DIM",
    "build_symbolic_observation",
    # Pixel
    "MAP_H",
    "MAP_W",
    "TILE_PX",
    "PIXEL_OBS_SHAPE",
    "build_pixel_observation",
    # Text / TTY
    "TERM_ROWS",
    "TERM_COLS",
    "TEXT_OBS_SHAPE",
    "build_text_observation",
]
