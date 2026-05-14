"""ASCII text observation builder for nethax.

Renders the game state as a 24x80 ASCII byte grid — the same representation
as NLE's tty_chars. Useful for debugging, human readability, and NLE parity
checks.

Canonical reference:
  - vendor/nle/include/nleobs.h  tty_chars: NLE_TERM_LI x NLE_TERM_CO (24x80)
  - vendor/nle/nle/nethack/nethack.py  TERMINAL_SHAPE = (24, 80)

Wave 1 status: returns a zero 24x80 uint8 grid.
"""

# TODO Wave 2: render the full map + status line into the 24x80 grid.
#              Layout (matching NLE's tty_chars convention):
#                row  0      : message line (latest game message)
#                rows 1-21   : dungeon map (21 rows x 79 cols, right-padded)
#                row 22      : blank separator
#                row 23      : status line (hp, ac, level, gold, …)
#              Convert glyph IDs to ASCII via cmap/monster/object tables,
#              matching NLE's rendering exactly so tty_chars diff is zero.

import jax.numpy as jnp

# Terminal dimensions — match NLE's NLE_TERM_LI x NLE_TERM_CO
TERM_ROWS: int = 24
TERM_COLS: int = 80

TEXT_OBS_SHAPE: tuple[int, int] = (TERM_ROWS, TERM_COLS)


def build_text_observation(env_state) -> jnp.ndarray:
    """Build a 24x80 ASCII byte grid from nethax EnvState.

    Wave 1: returns jnp.zeros((24, 80), uint8).

    Wave 2 will render:
      - message line (row 0)
      - dungeon map rows (rows 1-21)
      - status line (row 23)
    by converting env_state glyphs and player stats to ASCII bytes, producing
    output byte-identical to NLE's tty_chars for equivalent game states.

    Args:
        env_state: nethax EnvState (unused in Wave 1).

    Returns:
        jnp.ndarray of shape (24, 80) uint8.
    """
    return jnp.zeros(TEXT_OBS_SHAPE, dtype=jnp.uint8)
