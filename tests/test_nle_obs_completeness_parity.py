"""NLE observation completeness + parity tests.

Verifies that build_nle_observation produces all 17 NLE-spec keys with
correct shapes, dtypes, and semantically correct values.

Vendor sources:
  - vendor/nle/include/nleobs.h    — struct layout, sizes, NLE_* constants
  - vendor/nle/win/rl/winrl.cc     — fill_obs: program_state, internal, misc
  - vendor/nle/src/nle.c:110-111  — tty_cursor[0]=cur->r, tty_cursor[1]=cur->c
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_RNG = jax.random.PRNGKey(0)

# ---------------------------------------------------------------------------
# Fixture: one env.reset() state + obs shared across the module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_state_obs():
    from Nethax.nethax.env import NethaxEnv
    env = NethaxEnv()
    state, obs = env.reset(_RNG)
    return env, state, obs


# ---------------------------------------------------------------------------
# 1. All 17 keys present with correct shapes + dtypes
# ---------------------------------------------------------------------------

_EXPECTED = {
    "glyphs":               ((21, 79),   jnp.int16),
    "chars":                ((21, 79),   jnp.uint8),
    "colors":               ((21, 79),   jnp.uint8),
    "specials":             ((21, 79),   jnp.uint8),
    "blstats":              ((27,),      jnp.int64),
    "message":              ((256,),     jnp.uint8),
    "program_state":        ((6,),       jnp.int32),
    "internal":             ((9,),       jnp.int32),
    "inv_glyphs":           ((55,),      jnp.int16),
    "inv_letters":          ((55,),      jnp.uint8),
    "inv_oclasses":         ((55,),      jnp.uint8),
    "inv_strs":             ((55, 80),   jnp.uint8),
    "screen_descriptions":  ((21, 79, 80), jnp.uint8),
    "tty_chars":            ((24, 80),   jnp.uint8),
    "tty_colors":           ((24, 80),   jnp.int8),
    "tty_cursor":           ((2,),       jnp.uint8),
    "misc":                 ((3,),       jnp.int32),
}


def test_all_17_keys_present(env_state_obs):
    """env.reset() -> obs has exactly the 17 NLE keys (nleobs.h:53-71)."""
    _, _, obs = env_state_obs
    assert set(obs.keys()) == set(_EXPECTED.keys()), (
        f"key mismatch: extra={set(obs) - set(_EXPECTED)}, "
        f"missing={set(_EXPECTED) - set(obs)}"
    )


@pytest.mark.parametrize("key,shape_dtype", list(_EXPECTED.items()))
def test_shape_and_dtype(env_state_obs, key, shape_dtype):
    """Each key has the NLE-canonical shape and dtype."""
    _, _, obs = env_state_obs
    expected_shape, expected_dtype = shape_dtype
    assert obs[key].shape == expected_shape, (
        f"{key}: shape {obs[key].shape} != {expected_shape}"
    )
    assert obs[key].dtype == expected_dtype, (
        f"{key}: dtype {obs[key].dtype} != {expected_dtype}"
    )


# ---------------------------------------------------------------------------
# 2. tty_chars — 24 rows x 80 cols, correct layout
#    Row 0 = message line; rows 1-21 = map (map row r -> tty row r+1);
#    rows 22-23 = status.
#    Citation: nle.c via VT terminal; nleobs.h NLE_TERM_LI=24, NLE_TERM_CO=80.
# ---------------------------------------------------------------------------

def test_tty_chars_24_rows_80_cols(env_state_obs):
    """tty_chars is uint8[24, 80] — NLE_TERM_LI * NLE_TERM_CO (nleobs.h:13-14)."""
    _, _, obs = env_state_obs
    assert obs["tty_chars"].shape == (24, 80)
    assert obs["tty_chars"].dtype == jnp.uint8


def test_tty_chars_row0_is_message_line(env_state_obs):
    """tty_chars row 0 is the message line (first 80 bytes of message_buffer).

    In a freshly-reset game the message buffer may be all-zero (blank) or
    contain a welcome message.  Either way row 0 must be identical to
    obs['message'][:80].
    """
    _, _, obs = env_state_obs
    row0 = np.array(obs["tty_chars"][0])
    msg80 = np.array(obs["message"][:80])
    np.testing.assert_array_equal(row0, msg80)


def test_tty_chars_map_rows_1_to_21(env_state_obs):
    """tty_chars rows 1-21 contain printable ASCII map characters.

    Each cell must be a printable ASCII value (>= 0x20) or NUL (before the
    map is populated).  The map area must not be all-zero after reset.
    """
    _, _, obs = env_state_obs
    map_rows = np.array(obs["tty_chars"][1:22])  # shape (21, 80)
    printable_or_nul = (map_rows >= 0x20) | (map_rows == 0)
    assert np.all(printable_or_nul), (
        "tty_chars rows 1-21 contain non-printable non-NUL bytes"
    )
    # At least some cells should be non-zero (floor/wall chars).
    assert np.any(map_rows > 0), "tty_chars map area is entirely zero after reset"


def test_tty_chars_status_rows_22_23(env_state_obs):
    """tty_chars rows 22-23 are the status lines (non-empty after reset)."""
    _, _, obs = env_state_obs
    row22 = np.array(obs["tty_chars"][22])
    row23 = np.array(obs["tty_chars"][23])
    assert np.any(row22 > 0), "status line 1 (row 22) is blank"
    assert np.any(row23 > 0), "status line 2 (row 23) is blank"


# ---------------------------------------------------------------------------
# 3. tty_cursor — [row, col] of player position in tty coordinates
#    Citation: vendor/nle/src/nle.c:110-111
#      tty_cursor[0] = (unsigned char) cur->r;   // row
#      tty_cursor[1] = (unsigned char) cur->c;   // col
#    The player at map row r appears at tty row r+1 (row 0 is the message line).
# ---------------------------------------------------------------------------

def test_tty_cursor_at_player_pos(env_state_obs):
    """tty_cursor[0] == player_map_row + 1, tty_cursor[1] == player_col - 1.

    Map rows 1-21 in tty correspond to map rows 0-20; row 0 is the message
    line.  NLE drops NetHack's unused internal column 0, so the displayed
    (tty / obs) column is the internal column minus 1 — the cursor follows
    the same convention (vendor tty_curs does ``--x``).  Citation:
    nle.c:110-111 writes cur->r, cur->c from the VT terminal cursor, which
    NLE positions at the player after each render step.
    """
    _, state, obs = env_state_obs
    cur = np.array(obs["tty_cursor"])
    pr = int(state.player_pos[0])
    pc = int(state.player_pos[1])
    # Player at map row pr -> tty row pr+1 (offset by message line).
    assert cur[0] == pr + 1, (
        f"tty_cursor row {cur[0]} != player_map_row+1 ({pr+1})"
    )
    # Internal col pc -> displayed col pc-1 (NLE drops internal column 0).
    assert cur[1] == pc - 1, (
        f"tty_cursor col {cur[1]} != player_col-1 ({pc - 1})"
    )


# ---------------------------------------------------------------------------
# 4. program_state[3] == 1 (in_moveloop) after reset
#    Citation: vendor/nle/win/rl/winrl.cc::fill_obs lines 262-271
#      obs->program_state[0] = program_state.gameover;
#      obs->program_state[1] = program_state.panicking;
#      obs->program_state[2] = program_state.exiting;
#      obs->program_state[3] = program_state.in_moveloop;   <- 1 during play
#      obs->program_state[4] = program_state.in_impossible;
#      obs->program_state[5] = program_state.something_worth_saving; <- 1 once started
# ---------------------------------------------------------------------------

def test_program_state_in_moveloop_set(env_state_obs):
    """program_state[3] == 1 (in_moveloop) after env.reset().

    Citation: winrl.cc::fill_obs line 265 sets program_state[3] = in_moveloop.
    nethax always sets this to 1 once the game has started.
    """
    _, _, obs = env_state_obs
    ps = np.array(obs["program_state"])
    assert ps[3] == 1, f"program_state[3] (in_moveloop) expected 1, got {ps[3]}"


def test_program_state_something_worth_saving_set(env_state_obs):
    """program_state[5] == 1 (something_worth_saving) after reset.

    Citation: winrl.cc::fill_obs line 267.
    """
    _, _, obs = env_state_obs
    ps = np.array(obs["program_state"])
    assert ps[5] == 1, (
        f"program_state[5] (something_worth_saving) expected 1, got {ps[5]}"
    )


def test_program_state_gameover_zero_on_reset(env_state_obs):
    """program_state[0] == 0 (gameover) on a fresh game.

    Citation: winrl.cc::fill_obs line 263.
    """
    _, _, obs = env_state_obs
    ps = np.array(obs["program_state"])
    assert ps[0] == 0, f"program_state[0] (gameover) expected 0, got {ps[0]}"


# ---------------------------------------------------------------------------
# 5. screen_descriptions — uint8[21, 79, 80]: non-zero for visible tiles
#    Citation: winrl.cc::store_screen_description fills per-tile ASCII text;
#    nethax builds this via _GLYPH_TO_DESCRIPTION_BYTES[glyphs].
# ---------------------------------------------------------------------------

def test_screen_descriptions_nonzero_for_visible_tiles(env_state_obs):
    """screen_descriptions has non-zero bytes at visible explored tiles.

    After reset, at least some tiles must have a non-empty description
    (e.g. "floor", "wall", "corridor").
    """
    _, _, obs = env_state_obs
    sd = np.array(obs["screen_descriptions"])  # (21, 79, 80)
    assert sd.shape == (21, 79, 80)
    assert sd.dtype == np.uint8
    # At least one tile must have a non-zero first byte (non-empty description).
    assert np.any(sd[:, :, 0] > 0), (
        "screen_descriptions: no tile has a non-empty description after reset"
    )


def test_screen_descriptions_player_tile_has_description(env_state_obs):
    """The player's tile has a non-empty description (monster name or terrain).

    Citation: winrl.cc::store_screen_description writes the lookat() string
    for the glyph at (x, y).
    """
    _, state, obs = env_state_obs
    pr = int(state.player_pos[0])
    pc = int(np.clip(state.player_pos[1], 0, 78))
    sd = np.array(obs["screen_descriptions"])
    tile_desc = sd[pr, pc]
    assert tile_desc[0] > 0, (
        f"screen_descriptions[{pr},{pc}] is empty (player tile should have desc)"
    )
