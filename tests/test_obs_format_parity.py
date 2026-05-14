"""Wave 6 Closing-Audit: byte-equal observation format parity vs NLE vendor.

Each test asserts a shape/dtype/index property against vendor/nle/include/nleobs.h.
These tests pin the *format contract* so any future drift fails fast.

Vendor citations (vendor/nle/include/nleobs.h):
  - line  5:  NLE_MESSAGE_SIZE 256
  - line  6:  NLE_BLSTATS_SIZE 27
  - line  7:  NLE_PROGRAM_STATE_SIZE 6
  - line  8:  NLE_INTERNAL_SIZE 9
  - line  9:  NLE_MISC_SIZE 3
  - line 10:  NLE_INVENTORY_SIZE 55
  - line 11:  NLE_INVENTORY_STR_LENGTH 80
  - line 12:  NLE_SCREEN_DESCRIPTION_LENGTH 80
  - line 13:  NLE_TERM_CO 80
  - line 14:  NLE_TERM_LI 24
  - lines 17-43: NLE_BL_* indices for the 27 blstats fields
  - lines 48-72: nle_observation struct member types
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.obs.nle_obs import (
    NLE_OBSERVATION_KEYS,
    build_nle_observation,
)


_RNG = jax.random.PRNGKey(0)


def _obs() -> dict[str, jnp.ndarray]:
    state = EnvState.default(rng=_RNG)
    return build_nle_observation(state)


# ---------------------------------------------------------------------------
# Per-key shape + dtype contracts
# ---------------------------------------------------------------------------

def test_glyphs_shape_21_79_int16():
    """nleobs.h:53 — short *glyphs; ROWNO*(COLNO-1) = 21*79."""
    o = _obs()
    assert o["glyphs"].shape == (21, 79)
    assert o["glyphs"].dtype == jnp.int16


def test_chars_shape_21_79_uint8():
    """nleobs.h:54 — unsigned char *chars; 21*79."""
    o = _obs()
    assert o["chars"].shape == (21, 79)
    assert o["chars"].dtype == jnp.uint8


def test_colors_shape_21_79_uint8():
    """nleobs.h:55 — unsigned char *colors; 21*79."""
    o = _obs()
    assert o["colors"].shape == (21, 79)
    assert o["colors"].dtype == jnp.uint8


def test_specials_shape_21_79_uint8():
    """nleobs.h:56 — unsigned char *specials; 21*79."""
    o = _obs()
    assert o["specials"].shape == (21, 79)
    assert o["specials"].dtype == jnp.uint8


def test_blstats_shape_27_int64():
    """nleobs.h:57 — long *blstats; NLE_BLSTATS_SIZE=27 (nleobs.h:6).

    NLE compiles on 64-bit Unix where C `long` == 64 bits, so int64 is the
    matching numpy/jax dtype.
    """
    o = _obs()
    assert o["blstats"].shape == (27,)
    assert o["blstats"].dtype == jnp.int64


def test_blstats_index_hp_at_position_10():
    """nleobs.h:27 — #define NLE_BL_HP 10. HP must land at blstats[10]."""
    from Nethax.nethax.constants.blstats import BL_HP
    assert BL_HP == 10

    state = EnvState.default(rng=_RNG).replace(player_hp=jnp.int32(37))
    obs = build_nle_observation(state)
    assert int(obs["blstats"][10]) == 37


def test_blstats_index_xp_at_position_18():
    """nleobs.h:35 — #define NLE_BL_XP 18. Experience level lives at blstats[18]."""
    from Nethax.nethax.constants.blstats import BL_XP
    assert BL_XP == 18

    state = EnvState.default(rng=_RNG).replace(player_xl=jnp.int32(9))
    obs = build_nle_observation(state)
    assert int(obs["blstats"][18]) == 9


def test_blstats_index_align_at_position_26():
    """nleobs.h:43 — #define NLE_BL_ALIGN 26. Alignment at blstats[26].

    Vendor (botl.c::status_bl_init) writes u.ualign.type: 1=Lawful, 0=Neutral,
    -1=Chaotic.  Our state uses 0=L/1=N/2=C; build_blstats applies 1 - x.
    """
    from Nethax.nethax.constants.blstats import BL_ALIGN
    assert BL_ALIGN == 26

    # Chaotic in state-internal encoding (=2) maps to vendor -1.
    state = EnvState.default(rng=_RNG).replace(player_align=jnp.int8(2))
    obs = build_nle_observation(state)
    assert int(obs["blstats"][26]) == -1

    # Lawful (state 0) -> vendor 1
    state = EnvState.default(rng=_RNG).replace(player_align=jnp.int8(0))
    obs = build_nle_observation(state)
    assert int(obs["blstats"][26]) == 1


def test_message_shape_256_uint8():
    """nleobs.h:58 + :5 — unsigned char *message; NLE_MESSAGE_SIZE=256."""
    o = _obs()
    assert o["message"].shape == (256,)
    assert o["message"].dtype == jnp.uint8


def test_inv_strs_shape_55_80_uint8():
    """nleobs.h:62-63 + :10-11 — unsigned char *inv_strs; 55*80."""
    o = _obs()
    assert o["inv_strs"].shape == (55, 80)
    assert o["inv_strs"].dtype == jnp.uint8


def test_inv_glyphs_shape_55_int16():
    """nleobs.h:61 — short *inv_glyphs; NLE_INVENTORY_SIZE=55."""
    o = _obs()
    assert o["inv_glyphs"].shape == (55,)
    assert o["inv_glyphs"].dtype == jnp.int16


def test_inv_letters_shape_55_uint8():
    """nleobs.h:64 — unsigned char *inv_letters; 55."""
    o = _obs()
    assert o["inv_letters"].shape == (55,)
    assert o["inv_letters"].dtype == jnp.uint8


def test_inv_oclasses_shape_55_uint8():
    """nleobs.h:65 — unsigned char *inv_oclasses; 55."""
    o = _obs()
    assert o["inv_oclasses"].shape == (55,)
    assert o["inv_oclasses"].dtype == jnp.uint8


def test_screen_descriptions_shape_21_79_80_uint8():
    """nleobs.h:66-67 + :12 — unsigned char *screen_descriptions;
    ROWNO*(COLNO-1)*NLE_SCREEN_DESCRIPTION_LENGTH = 21*79*80."""
    o = _obs()
    assert o["screen_descriptions"].shape == (21, 79, 80)
    assert o["screen_descriptions"].dtype == jnp.uint8


def test_tty_chars_shape_24_80():
    """nleobs.h:68 + :13-14 — unsigned char *tty_chars; 24*80."""
    o = _obs()
    assert o["tty_chars"].shape == (24, 80)
    assert o["tty_chars"].dtype == jnp.uint8


def test_tty_colors_shape_24_80_int8():
    """nleobs.h:69 — signed char *tty_colors; 24*80. Must be SIGNED."""
    o = _obs()
    assert o["tty_colors"].shape == (24, 80)
    assert o["tty_colors"].dtype == jnp.int8


def test_tty_cursor_shape_2_uint8():
    """nleobs.h:70 — unsigned char *tty_cursor; size 2 (y, x)."""
    o = _obs()
    assert o["tty_cursor"].shape == (2,)
    assert o["tty_cursor"].dtype == jnp.uint8


def test_misc_shape_3_int32():
    """nleobs.h:71 + :9 — int *misc; NLE_MISC_SIZE=3."""
    o = _obs()
    assert o["misc"].shape == (3,)
    assert o["misc"].dtype == jnp.int32


def test_internal_shape_9_int32():
    """nleobs.h:60 + :8 — int *internal; NLE_INTERNAL_SIZE=9."""
    o = _obs()
    assert o["internal"].shape == (9,)
    assert o["internal"].dtype == jnp.int32


def test_program_state_shape_6_int32():
    """nleobs.h:59 + :7 — int *program_state; NLE_PROGRAM_STATE_SIZE=6."""
    o = _obs()
    assert o["program_state"].shape == (6,)
    assert o["program_state"].dtype == jnp.int32


# ---------------------------------------------------------------------------
# Whole-dict contract
# ---------------------------------------------------------------------------

def test_all_17_keys_present_in_obs_dict():
    """nleobs.h:53-71 — all 17 named pointer fields in nle_observation."""
    o = _obs()
    expected = {
        "glyphs", "chars", "colors", "specials", "blstats", "message",
        "program_state", "internal", "inv_glyphs", "inv_strs", "inv_letters",
        "inv_oclasses", "screen_descriptions", "tty_chars", "tty_colors",
        "tty_cursor", "misc",
    }
    assert set(o.keys()) == expected
    assert len(NLE_OBSERVATION_KEYS) == 17
