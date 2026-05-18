"""ASCII text observation builder for nethax.

Renders the game state as a 24x80 ASCII byte grid matching NLE's tty_chars.

Layout:
  Row 0:     message line (state.messages.message_buffer[:80])
  Rows 1-21: map glyphs converted to ASCII chars
  Row 22:    status line 1 ("Player the Title St:X Dx:Y ... HP:X(Y) Pw:X(Y) AC:X")
  Row 23:    status line 2 ("Dlvl:N $:gold ... Turns:T")

JIT-pure. Delegates to the existing nle_obs.build_tty which is the
canonical implementation of this rendering.
"""

import jax.numpy as jnp

TERM_ROWS: int = 24
TERM_COLS: int = 80

TEXT_OBS_SHAPE: tuple[int, int] = (TERM_ROWS, TERM_COLS)


def build_text_observation(env_state) -> jnp.ndarray:
    """Build a 24x80 ASCII byte grid from nethax EnvState.

    Delegates to nle_obs.build_tty (the canonical NLE tty_chars renderer):
      - Row 0:     message line
      - Rows 1-21: map glyphs -> ASCII via _MON_IDX_TO_CHAR / _OBJ_IDX_TO_CHAR /
                   _CMAP_IDX_TO_CHAR tables (all in nle_obs.py)
      - Row 22:    status line 1 (HP, Pw, AC, attributes, alignment)
      - Row 23:    status line 2 (Dlvl, gold, XP, turns)

    Args:
        env_state: nethax EnvState.

    Returns:
        jnp.ndarray of shape (24, 80) uint8.
    """
    from Nethax.nethax.obs.nle_obs import build_tty
    tty = build_tty(env_state)
    return tty["tty_chars"]
