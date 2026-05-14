"""Wave 7 visualization parity tests.

Pin per-key observation invariants against vendor NLE behavior.  Each test
cites a vendor source file + line range so future regressions can be traced
back to a concrete authoritative reference.

Vendor citations summary:
  - tty_chars layout      : vendor/nle/src/nle.c::nle_vt_callback (lines 83-117)
  - tty_cursor mapping    : vendor/nle/src/nle.c (lines 108-112)
  - specials MG_* bits    : vendor/nethack/include/display.h (lines 995-1009)
  - internal layout       : vendor/nle/win/rl/winrl.cc::fill_obs (lines 272-288)
  - misc layout           : vendor/nle/win/rl/winrl.cc::fill_obs (lines 289-293)
  - program_state layout  : vendor/nle/win/rl/winrl.cc::fill_obs (lines 262-271)
  - tty_colors signedness : vendor/nle/include/nleobs.h (line 69)
  - screen_descriptions   : vendor/nethack/src/pager.c::do_screen_description
  - message buffer        : vendor/nle/win/rl/winrl.cc (lines 335-355)
  - status lines (bot1/2) : vendor/nethack/src/botl.c (lines 48-249)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.obs.nle_obs import (
    build_nle_observation,
    build_specials,
    build_internal,
    build_program_state,
    build_misc,
    role_rank_title,
    _xlev_to_rank,
)


_RNG = jax.random.PRNGKey(0)


def _state() -> EnvState:
    return EnvState.default(rng=_RNG)


# ---------------------------------------------------------------------------
# 1. tty_chars layout
# ---------------------------------------------------------------------------

def test_tty_chars_row0_matches_message_buffer():
    """tty_chars row 0 is the toplines/message buffer; matches obs['message'][:80].

    vendor/nle/win/rl/winrl.cc::fill_obs:335-354 — obs->message is the same
    string the terminal renders at row 0.
    """
    obs = build_nle_observation(_state())
    tty = np.asarray(obs["tty_chars"])
    msg = np.asarray(obs["message"])
    assert tty.shape == (24, 80)
    # Row 0 must be the first 80 bytes of the message buffer.
    assert np.array_equal(tty[0, :80], msg[:80])


def test_tty_chars_player_at_at_expected_row_col():
    """The player's '@' lands at tty_chars[player_row+1, player_col]."""
    state = _state().replace(player_pos=jnp.array([5, 9], dtype=jnp.int16))
    obs = build_nle_observation(state)
    tty = np.asarray(obs["tty_chars"])
    assert chr(int(tty[5 + 1, 9])) == "@"


def test_tty_chars_rows_22_23_contain_status_text():
    """Rows 22-23 contain non-null bytes matching the vendor status format."""
    obs = build_nle_observation(_state())
    tty = np.asarray(obs["tty_chars"])
    row22 = bytes(tty[22]).decode("ascii", errors="replace")
    row23 = bytes(tty[23]).decode("ascii", errors="replace")
    # Vendor row 22 always contains "St:" and the alignment word.
    assert "St:" in row22, f"row 22 missing St:, got: {row22!r}"
    assert ("Lawful" in row22) or ("Neutral" in row22) or ("Chaotic" in row22), row22
    # Vendor row 23 always contains "Dlvl:" and "$:" and "HP:".
    assert "Dlvl:" in row23, f"row 23 missing Dlvl:, got: {row23!r}"
    assert "$:" in row23, f"row 23 missing $:, got: {row23!r}"
    assert "HP:" in row23, f"row 23 missing HP:, got: {row23!r}"


# ---------------------------------------------------------------------------
# 2. tty_colors layout
# ---------------------------------------------------------------------------

def test_tty_colors_is_signed_int8():
    """vendor/nle/include/nleobs.h:69 — signed char *tty_colors."""
    obs = build_nle_observation(_state())
    assert obs["tty_colors"].dtype == jnp.int8
    assert obs["tty_colors"].shape == (24, 80)


# ---------------------------------------------------------------------------
# 3. tty_cursor mapping — vendor (r, c) == (player_y+1, player_x)
# ---------------------------------------------------------------------------

def test_tty_cursor_y_first_x_second():
    """vendor/nle/src/nle.c:108-112 — cursor[0]=r (y), cursor[1]=c (x)."""
    state = _state().replace(player_pos=jnp.array([7, 11], dtype=jnp.int16))
    obs = build_nle_observation(state)
    cur = np.asarray(obs["tty_cursor"])
    # row = player_y + 1 (map window offset)
    assert int(cur[0]) == 7 + 1
    # col = player_x
    assert int(cur[1]) == 11


# ---------------------------------------------------------------------------
# 4. colors — player tile is bright (15)
# ---------------------------------------------------------------------------

def test_colors_player_position_is_bright():
    """build_colors marks the player tile with color 15."""
    state = _state().replace(player_pos=jnp.array([3, 4], dtype=jnp.int16))
    obs = build_nle_observation(state)
    colors = np.asarray(obs["colors"])
    assert int(colors[3, 4]) == 15


# ---------------------------------------------------------------------------
# 5. specials — vendor MG_* bit layout
# ---------------------------------------------------------------------------

def test_specials_mg_hero_at_player_position():
    """display.h:995 — MG_HERO = 0x01.  Player tile must have bit 0x01 set."""
    state = _state().replace(player_pos=jnp.array([8, 12], dtype=jnp.int16))
    specials = np.asarray(build_specials(state))
    assert (int(specials[8, 12]) & 0x01) == 0x01


def test_specials_mg_corpse_for_food_corpse():
    """display.h:996 — MG_CORPSE = 0x02."""
    state = _state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    gi = state.ground_items
    cat = gi.category.at[branch, level, 6, 6, 0].set(jnp.int8(7))    # FOOD_CLASS
    typ = gi.type_id.at[branch, level, 6, 6, 0].set(jnp.int16(260))  # corpse
    new_gi = gi.replace(category=cat, type_id=typ)
    state = state.replace(ground_items=new_gi)
    specials = np.asarray(build_specials(state))
    assert (int(specials[6, 6]) & 0x02) == 0x02


def test_specials_mg_objpile_for_two_stacks():
    """display.h:1002 — MG_OBJPILE = 0x80 when 2+ object stacks coexist."""
    state = _state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    gi = state.ground_items
    cat = gi.category.at[branch, level, 4, 4, 0].set(jnp.int8(2))
    cat = cat.at[branch, level, 4, 4, 1].set(jnp.int8(3))
    new_gi = gi.replace(category=cat)
    state = state.replace(ground_items=new_gi)
    specials = np.asarray(build_specials(state))
    assert (int(specials[4, 4]) & 0x80) == 0x80


# ---------------------------------------------------------------------------
# 6. screen_descriptions — vendor pager.c
# ---------------------------------------------------------------------------

def test_screen_descriptions_non_null_at_player_position():
    """Player tile produces a non-empty (non-null-byte) description."""
    state = _state().replace(player_pos=jnp.array([10, 10], dtype=jnp.int16))
    obs = build_nle_observation(state)
    desc = np.asarray(obs["screen_descriptions"])
    cell = bytes(desc[10, 10])
    # The first 79 bytes contain ASCII; trailing null is allowed.
    text = cell.split(b"\x00", 1)[0].decode("ascii", errors="replace")
    # Player glyph is a monster glyph; the description maps to the race name.
    assert text != "" or cell[0] != 0, "expected non-empty description at player tile"


# ---------------------------------------------------------------------------
# 7. message — pass-through from state buffer
# ---------------------------------------------------------------------------

def test_message_buffer_passthrough():
    """obs['message'] equals state.messages.message_buffer[:256]."""
    state = _state()
    # Inject a known message.
    payload = b"You hear distant thunder."
    new_buf = state.messages.message_buffer.at[: len(payload)].set(
        jnp.array(list(payload), dtype=jnp.uint8)
    )
    state = state.replace(
        messages=state.messages.replace(message_buffer=new_buf)
    )
    obs = build_nle_observation(state)
    msg = np.asarray(obs["message"])
    assert msg.shape == (256,)
    assert bytes(msg[: len(payload)]) == payload


# ---------------------------------------------------------------------------
# 8. internal — vendor 9-int layout
# ---------------------------------------------------------------------------

def test_internal_shape_and_dtype():
    """vendor nleobs.h:60+8 — int *internal; size 9."""
    obs = build_nle_observation(_state())
    assert obs["internal"].shape == (9,)
    assert obs["internal"].dtype == jnp.int32


def test_internal_indices_1_2_3_are_zero():
    """internal[1..3] (in_yn_function, in_getlin, xwaitingforspace) == 0."""
    internal = np.asarray(build_internal(_state()))
    assert int(internal[1]) == 0
    assert int(internal[2]) == 0
    assert int(internal[3]) == 0


def test_internal_indices_5_6_are_zero():
    """internal[5,6] are legacy seed slots; vendor pins them at 0."""
    internal = np.asarray(build_internal(_state()))
    assert int(internal[5]) == 0
    assert int(internal[6]) == 0


# ---------------------------------------------------------------------------
# 9. misc — vendor 3-int layout (all zeros for non-menu states)
# ---------------------------------------------------------------------------

def test_misc_all_zero_in_normal_play():
    """vendor fill_obs:289-293 — misc == [in_yn_function, in_getlin,
    xwaitingforspace].  All zero when no menu/prompt is active."""
    misc = np.asarray(build_misc(_state()))
    assert misc.shape == (3,)
    assert misc.dtype == np.int32
    assert int(misc[0]) == 0
    assert int(misc[1]) == 0
    assert int(misc[2]) == 0


# ---------------------------------------------------------------------------
# 10. program_state — vendor 6-int layout
# ---------------------------------------------------------------------------

def test_program_state_in_moveloop_at_index_3():
    """vendor fill_obs:266 — program_state[3] = in_moveloop."""
    ps = np.asarray(build_program_state(_state()))
    assert int(ps[3]) == 1


def test_program_state_in_impossible_at_index_4_is_zero():
    """vendor fill_obs:267 — program_state[4] = in_impossible.  Must be 0
    in normal play (nethax never sets it)."""
    ps = np.asarray(build_program_state(_state()))
    assert int(ps[4]) == 0


def test_program_state_something_worth_saving_at_index_5():
    """vendor fill_obs:268 — program_state[5] = something_worth_saving."""
    ps = np.asarray(build_program_state(_state()))
    assert int(ps[5]) == 1


# ---------------------------------------------------------------------------
# 11. Role rank title — vendor botl.c::rank_of
# ---------------------------------------------------------------------------

def test_xlev_to_rank_matches_vendor_table():
    """vendor botl.c:296-311 — xlev_to_rank piecewise table."""
    # Spot-check a few boundary cases.
    assert _xlev_to_rank(1) == 0
    assert _xlev_to_rank(2) == 0
    assert _xlev_to_rank(3) == 1
    assert _xlev_to_rank(5) == 1
    assert _xlev_to_rank(6) == 2
    assert _xlev_to_rank(10) == 3
    assert _xlev_to_rank(30) == 8
    assert _xlev_to_rank(31) == 8  # clamps to 8


def test_role_rank_title_archeologist_level_1():
    """role.c:32 — Archeologist rank 0 title is 'Digger'."""
    assert role_rank_title(0, 1) == "Digger"


def test_role_rank_title_valkyrie_level_30():
    """role.c:502 — Valkyrie rank 8 title is 'Lord' (male form)."""
    # Valkyrie role index = 11
    assert role_rank_title(11, 30) == "Lord"


# ---------------------------------------------------------------------------
# 12. End-to-end shape contract
# ---------------------------------------------------------------------------

def test_all_keys_match_vendor_shapes_after_reset():
    """All 17 obs keys produced by build_nle_observation have correct shapes."""
    obs = build_nle_observation(_state())
    expected = {
        "glyphs": (21, 79),
        "chars": (21, 79),
        "colors": (21, 79),
        "specials": (21, 79),
        "blstats": (27,),
        "message": (256,),
        "program_state": (6,),
        "internal": (9,),
        "inv_glyphs": (55,),
        "inv_letters": (55,),
        "inv_oclasses": (55,),
        "inv_strs": (55, 80),
        "screen_descriptions": (21, 79, 80),
        "tty_chars": (24, 80),
        "tty_colors": (24, 80),
        "tty_cursor": (2,),
        "misc": (3,),
    }
    for k, shape in expected.items():
        assert obs[k].shape == shape, f"{k} shape mismatch"


# ---------------------------------------------------------------------------
# 13. Live NLE byte-equality smoke (skipped if NLE not installed)
# ---------------------------------------------------------------------------

def test_live_nle_obs_keys_match_nethax():
    """Smoke: live NLE produces the same set of observation keys we do.

    Skipped if nle isn't installed.  Verifies the key list contract, not
    per-byte content (which requires identical levels/seeds).
    """
    nle = pytest.importorskip("nle.env.base", reason="nle not installed")
    pytest.importorskip("nle.nethack")
    # Construct a default NLE env and reset it.
    from nle.env import base as nle_base
    env = nle_base.NLE()
    nle_obs, _info = env.reset()
    env.close()

    nethax_obs = build_nle_observation(_state())
    # Keys must be a superset on the nethax side (we never drop a key).
    for k in nle_obs:
        assert k in nethax_obs, f"missing key in nethax obs: {k}"

    # Shapes must also match for the shared keys.
    for k in nle_obs:
        assert tuple(nle_obs[k].shape) == tuple(nethax_obs[k].shape), (
            f"shape mismatch for {k}: nle={nle_obs[k].shape} "
            f"nethax={nethax_obs[k].shape}"
        )
