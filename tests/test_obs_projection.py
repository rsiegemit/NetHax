"""Wave 2 NLE observation projection tests.

Covers:
  - build_blstats field projection
  - build_glyphs player overlay and unexplored masking
  - build_message passthrough
  - build_tty player char rendering
  - JIT compilation of env.step (confirms all projections are jit-compatible)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.obs.nle_obs import (
    build_blstats,
    build_colors,
    build_glyphs,
    build_inv_glyphs,
    build_inv_letters,
    build_inv_oclasses,
    build_message,
    build_tty,
    build_nle_observation,
)
from Nethax.nethax.constants.blstats import (
    BL_HP, BL_HPMAX, BL_DEPTH, BL_DNUM, BL_DLEVEL,
    BL_X, BL_Y, BL_STR25, BL_STR125,
    BL_DEX, BL_CON, BL_INT, BL_WIS, BL_CHA,
    BL_GOLD, BL_ENE, BL_ENEMAX, BL_XP, BL_EXP, BL_TIME,
    BL_HUNGER, BL_CAP, BL_ALIGN, BL_SCORE,
)
from Nethax.nethax.constants.glyphs import GLYPH_MON_OFF, NO_GLYPH


_RNG = jax.random.PRNGKey(42)


def _default_state() -> EnvState:
    return EnvState.default(rng=_RNG)


# ---------------------------------------------------------------------------
# build_blstats
# ---------------------------------------------------------------------------

def test_blstats_hp_projection():
    state = _default_state()
    state = state.replace(player_hp=jnp.int32(42))
    obs = build_nle_observation(state)
    assert int(obs["blstats"][BL_HP]) == 42


def test_blstats_hpmax_projection():
    state = _default_state()
    state = state.replace(player_hp_max=jnp.int32(55))
    obs = build_nle_observation(state)
    assert int(obs["blstats"][BL_HPMAX]) == 55


def test_blstats_depth_projection():
    state = _default_state()
    # current_level is int8 and 1-based
    new_dungeon = state.dungeon.replace(current_level=jnp.int8(5))
    state = state.replace(dungeon=new_dungeon)
    obs = build_nle_observation(state)
    assert int(obs["blstats"][BL_DEPTH]) == 5


def test_blstats_dlevel_equals_depth():
    state = _default_state()
    new_dungeon = state.dungeon.replace(current_level=jnp.int8(3))
    state = state.replace(dungeon=new_dungeon)
    blstats = build_blstats(state)
    assert int(blstats[BL_DEPTH]) == int(blstats[BL_DLEVEL])


def test_blstats_dnum_projection():
    state = _default_state()
    new_dungeon = state.dungeon.replace(current_branch=jnp.int8(1))
    state = state.replace(dungeon=new_dungeon)
    blstats = build_blstats(state)
    assert int(blstats[BL_DNUM]) == 1


def test_blstats_position_projection():
    state = _default_state()
    state = state.replace(player_pos=jnp.array([7, 15], dtype=jnp.int16))
    blstats = build_blstats(state)
    assert int(blstats[BL_Y]) == 7   # row
    assert int(blstats[BL_X]) == 15  # col


def test_blstats_str_clamped():
    state = _default_state()
    # str = 30 (>25) -> BL_STR25 should be clamped to 25, BL_STR125 = 30
    state = state.replace(player_str=jnp.int16(30))
    blstats = build_blstats(state)
    assert int(blstats[BL_STR25]) == 25
    assert int(blstats[BL_STR125]) == 30


def test_blstats_str_below_cap():
    state = _default_state()
    # str = 16 (<= 25) -> BL_STR25 = 16, BL_STR125 = 16
    state = state.replace(player_str=jnp.int16(16))
    blstats = build_blstats(state)
    assert int(blstats[BL_STR25]) == 16
    assert int(blstats[BL_STR125]) == 16


def test_blstats_gold():
    state = _default_state()
    state = state.replace(player_gold=jnp.int32(999))
    blstats = build_blstats(state)
    assert int(blstats[BL_GOLD]) == 999


def test_blstats_xp_xl():
    state = _default_state()
    state = state.replace(player_xl=jnp.int32(7), player_xp=jnp.int32(1234))
    blstats = build_blstats(state)
    assert int(blstats[BL_XP]) == 7
    assert int(blstats[BL_EXP]) == 1234


def test_blstats_time():
    state = _default_state()
    state = state.replace(timestep=jnp.int32(100))
    blstats = build_blstats(state)
    assert int(blstats[BL_TIME]) == 100


def test_blstats_shape_and_dtype():
    state = _default_state()
    blstats = build_blstats(state)
    assert blstats.shape == (27,)
    assert blstats.dtype == jnp.int64


# ---------------------------------------------------------------------------
# build_glyphs
# ---------------------------------------------------------------------------

def test_glyphs_player_visible():
    """Player position should have the player glyph (GLYPH_MON_OFF = 0)."""
    state = _default_state()
    # Place player at (10, 40), which is within the 21x79 NLE grid
    state = state.replace(player_pos=jnp.array([10, 40], dtype=jnp.int16))
    obs = build_nle_observation(state)
    player_glyph = int(obs["glyphs"][10, 40])
    assert player_glyph == GLYPH_MON_OFF


def test_glyphs_player_not_zero():
    """Regression: glyphs at player pos should differ from unexplored (NO_GLYPH)."""
    state = _default_state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))
    obs = build_nle_observation(state)
    assert int(obs["glyphs"][5, 5]) != (NO_GLYPH & 0xFFFF)


def test_glyphs_unexplored_is_no_glyph():
    """Unexplored tiles (explored==False) should have NO_GLYPH value."""
    state = _default_state()
    # Default state: explored array is all False; player at (0,0)
    # Any tile NOT at player_pos should be NO_GLYPH
    obs = build_nle_observation(state)
    # Tile (5, 5) is not the player position (0,0) and is unexplored
    tile_val = int(obs["glyphs"][5, 5])
    expected = int(jnp.int16(NO_GLYPH & 0xFFFF))
    assert tile_val == expected


def test_glyphs_shape_and_dtype():
    state = _default_state()
    glyphs = build_glyphs(state)
    assert glyphs.shape == (21, 79)
    assert glyphs.dtype == jnp.int16


def test_glyphs_explored_tile_has_cmap_glyph():
    """An explored floor tile should return a cmap glyph, not NO_GLYPH."""
    from Nethax.nethax.constants.glyphs import GLYPH_CMAP_OFF
    from Nethax.nethax.constants import TileType

    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1

    # Set tile (8, 8) to FLOOR and mark it explored; keep player elsewhere
    new_terrain = state.terrain.at[branch, level, 8, 8].set(jnp.int8(TileType.FLOOR))
    new_explored = state.explored.at[branch, level, 8, 8].set(True)
    state = state.replace(
        terrain=new_terrain,
        explored=new_explored,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )
    glyphs = build_glyphs(state)
    glyph_val = int(glyphs[8, 8])
    # Should be a cmap glyph (GLYPH_CMAP_OFF + some cmap index), not NO_GLYPH
    assert glyph_val != int(jnp.int16(NO_GLYPH & 0xFFFF))
    assert glyph_val >= GLYPH_CMAP_OFF


# ---------------------------------------------------------------------------
# build_message
# ---------------------------------------------------------------------------

def test_message_projection():
    """Message buffer bytes should appear verbatim in obs['message']."""
    state = _default_state()
    # Write "Hello" into the message buffer
    msg_bytes = [ord('H'), ord('e'), ord('l'), ord('l'), ord('o')]
    new_buf = state.messages.message_buffer.at[:5].set(
        jnp.array(msg_bytes, dtype=jnp.uint8)
    )
    new_messages = state.messages.replace(message_buffer=new_buf)
    state = state.replace(messages=new_messages)

    obs = build_nle_observation(state)
    for i, ch in enumerate(msg_bytes):
        assert int(obs["message"][i]) == ch, f"message[{i}] mismatch"


def test_message_shape_and_dtype():
    state = _default_state()
    msg = build_message(state)
    assert msg.shape == (256,)
    assert msg.dtype == jnp.uint8


def test_message_passthrough_full_buffer():
    """Full 256-byte buffer should pass through unchanged."""
    state = _default_state()
    test_data = jnp.arange(256, dtype=jnp.uint8)
    new_messages = state.messages.replace(message_buffer=test_data)
    state = state.replace(messages=new_messages)
    msg = build_message(state)
    assert jnp.array_equal(msg, test_data)


# ---------------------------------------------------------------------------
# build_tty
# ---------------------------------------------------------------------------

def test_tty_player_char():
    """tty_chars at the player's row+1 (map offset) and col should be ord('@')."""
    state = _default_state()
    state = state.replace(player_pos=jnp.array([10, 40], dtype=jnp.int16))
    tty = build_tty(state)
    # Row 0 is the message line; map starts at row 1
    tty_row = 10 + 1  # map offset
    assert int(tty["tty_chars"][tty_row, 40]) == ord('@')


def test_tty_cursor_at_player():
    """tty_cursor should point to (player_row + 1, player_col)."""
    state = _default_state()
    state = state.replace(player_pos=jnp.array([5, 20], dtype=jnp.int16))
    tty = build_tty(state)
    cursor = tty["tty_cursor"]
    assert int(cursor[0]) == 6   # row 5 + 1 (message line offset)
    assert int(cursor[1]) == 20


def test_tty_message_row():
    """Row 0 of tty_chars should contain the first 80 bytes of the message buffer."""
    state = _default_state()
    msg_bytes = [ord('T'), ord('e'), ord('s'), ord('t')]
    new_buf = state.messages.message_buffer.at[:4].set(
        jnp.array(msg_bytes, dtype=jnp.uint8)
    )
    new_messages = state.messages.replace(message_buffer=new_buf)
    state = state.replace(messages=new_messages)
    tty = build_tty(state)
    for i, ch in enumerate(msg_bytes):
        assert int(tty["tty_chars"][0, i]) == ch


def test_tty_shapes():
    state = _default_state()
    tty = build_tty(state)
    assert tty["tty_chars"].shape == (24, 80)
    assert tty["tty_colors"].shape == (24, 80)
    assert tty["tty_cursor"].shape == (2,)


def test_tty_dtypes():
    state = _default_state()
    tty = build_tty(state)
    assert tty["tty_chars"].dtype == jnp.uint8
    assert tty["tty_colors"].dtype == jnp.int8
    assert tty["tty_cursor"].dtype == jnp.uint8


# ---------------------------------------------------------------------------
# Full observation dict
# ---------------------------------------------------------------------------

def test_build_nle_observation_keys():
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_KEYS
    state = _default_state()
    obs = build_nle_observation(state)
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)


def test_build_nle_observation_shapes():
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_SHAPES
    state = _default_state()
    obs = build_nle_observation(state)
    for key, expected_shape in NLE_OBSERVATION_SHAPES.items():
        assert tuple(obs[key].shape) == expected_shape, (
            f"{key}: expected {expected_shape}, got {obs[key].shape}"
        )


def test_build_nle_observation_dtypes():
    from Nethax.nethax.obs.nle_obs import NLE_OBSERVATION_DTYPES
    state = _default_state()
    obs = build_nle_observation(state)
    for key, expected_dtype in NLE_OBSERVATION_DTYPES.items():
        assert obs[key].dtype == expected_dtype, (
            f"{key}: expected dtype {expected_dtype}, got {obs[key].dtype}"
        )


# ---------------------------------------------------------------------------
# JIT compatibility — confirms all projections survive jax.jit
# ---------------------------------------------------------------------------

def test_jit_build_nle_observation():
    """build_nle_observation must be jit-compilable."""
    state = _default_state()
    jitted = jax.jit(build_nle_observation)
    obs = jitted(state)
    # Spot-check one value to force materialisation
    assert obs["blstats"].shape == (27,)


def test_jit_env_step():
    """env.step must be jit-compilable end-to-end."""
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(_RNG)

    jitted_step = jax.jit(env.step)
    action = jnp.int32(0)
    rng2 = jax.random.PRNGKey(1)
    new_state, obs, reward, done, info = jitted_step(state, action, rng2)

    # Confirm observation has correct shapes and the projection ran
    assert obs["blstats"].shape == (27,)
    assert obs["glyphs"].shape == (21, 79)
    assert obs["message"].shape == (256,)
    assert obs["tty_chars"].shape == (24, 80)


# ---------------------------------------------------------------------------
# Wave 3: colors
# ---------------------------------------------------------------------------

def test_colors_player_is_bright_yellow():
    """Player tile color must be 15 (CLR_WHITE / CLR_BRIGHT_YELLOW)."""
    state = _default_state()
    state = state.replace(player_pos=jnp.array([10, 40], dtype=jnp.int16))
    obs = build_nle_observation(state)
    assert int(obs["colors"][10, 40]) == 15


def test_colors_explored_floor_is_gray():
    """An explored floor tile should have color 7 (CLR_GRAY)."""
    from Nethax.nethax.constants import TileType
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[branch, level, 8, 8].set(jnp.int8(TileType.FLOOR))
    new_explored = state.explored.at[branch, level, 8, 8].set(True)
    state = state.replace(
        terrain=new_terrain,
        explored=new_explored,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )
    obs = build_nle_observation(state)
    assert int(obs["colors"][8, 8]) == 7


def test_colors_unexplored_is_black():
    """Unexplored tiles should have color 0 (black)."""
    state = _default_state()
    # Default state: all tiles unexplored; player at (0,0)
    obs = build_nle_observation(state)
    # Tile (5, 5) is unexplored (and not the player position)
    assert int(obs["colors"][5, 5]) == 0


def test_colors_shape_and_dtype():
    state = _default_state()
    colors = build_colors(state)
    assert colors.shape == (21, 79)
    assert colors.dtype == jnp.uint8


def test_tty_colors_message_row_is_white():
    """tty_colors row 0 (message line) should be all 7 (white)."""
    state = _default_state()
    obs = build_nle_observation(state)
    assert int(obs["tty_colors"][0, 0]) == 7
    assert int(obs["tty_colors"][0, 79]) == 7


def test_tty_colors_status_rows_are_white():
    """tty_colors rows 22-23 (status lines) should be all 7 (white)."""
    state = _default_state()
    obs = build_nle_observation(state)
    assert int(obs["tty_colors"][22, 0]) == 7
    assert int(obs["tty_colors"][23, 0]) == 7


def test_tty_colors_player_tile_is_bright_yellow():
    """tty_colors at the player's map row+1 should be 15."""
    state = _default_state()
    state = state.replace(player_pos=jnp.array([10, 40], dtype=jnp.int16))
    obs = build_nle_observation(state)
    assert int(obs["tty_colors"][11, 40]) == 15  # row 10 + 1 for message offset


# ---------------------------------------------------------------------------
# Wave 3: inventory
# ---------------------------------------------------------------------------

def test_inv_letters_slot0_is_a():
    """inv_letters[0] must be ord('a') = 97."""
    state = _default_state()
    obs = build_nle_observation(state)
    assert int(obs["inv_letters"][0]) == ord('a')


def test_inv_letters_slot25_is_z():
    """inv_letters[25] must be ord('z') = 122."""
    state = _default_state()
    inv_letters = build_inv_letters(state)
    assert int(inv_letters[25]) == ord('z')


def test_inv_letters_slot26_is_A():
    """inv_letters[26] must be ord('A') = 65."""
    state = _default_state()
    inv_letters = build_inv_letters(state)
    assert int(inv_letters[26]) == ord('A')


def test_inv_letters_slot51_is_Z():
    """inv_letters[51] must be ord('Z') = 90."""
    state = _default_state()
    inv_letters = build_inv_letters(state)
    assert int(inv_letters[51]) == ord('Z')


def test_inv_oclasses_empty_slots_zero():
    """Empty inventory (default state) -> all oclasses zero."""
    state = _default_state()
    obs = build_nle_observation(state)
    assert int(obs["inv_oclasses"][0]) == 0
    assert int(jnp.sum(obs["inv_oclasses"])) == 0


def test_inv_oclasses_occupied_slot_matches_category():
    """inv_oclasses[0] must match inventory.items.category when occupied."""
    from Nethax.nethax.subsystems.inventory import Item
    state = _default_state()
    # Place a weapon (category=2) in slot 0
    new_item = Item(
        category=jnp.int8(2),
        type_id=jnp.int16(0),
        buc_status=jnp.int8(0),
        enchantment=jnp.int8(0),
        charges=jnp.int8(0),
        identified=jnp.bool_(False),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    new_inv = state.inventory.replace(items=new_item)
    state = state.replace(inventory=new_inv)
    inv_oclasses = build_inv_oclasses(state)
    assert int(inv_oclasses[0]) == 2


def test_inv_glyphs_empty_is_zero():
    """Empty inventory -> all inv_glyphs zero."""
    state = _default_state()
    obs = build_nle_observation(state)
    assert int(jnp.sum(obs["inv_glyphs"])) == 0


def test_inv_glyphs_occupied_slot_matches_obj_table():
    """inv_glyphs[0] must be GLYPH_OBJ_OFF + type_id when slot is occupied."""
    from Nethax.nethax.subsystems.inventory import Item
    from Nethax.nethax.constants.glyphs import GLYPH_OBJ_OFF
    state = _default_state()
    # Wave 6 parity-fix: updated to match vendor/nle/src/objects.c:117
    # (PROJECTILE("orcish arrow", "crude arrow", ...) — vendor index 3)
    type_id = 3  # orcish arrow
    new_item = Item(
        category=jnp.int8(2),   # WEAPON_CLASS — non-zero means occupied
        type_id=jnp.int16(type_id),
        buc_status=jnp.int8(0),
        enchantment=jnp.int8(0),
        charges=jnp.int8(0),
        identified=jnp.bool_(False),
        quantity=jnp.int16(1),
        weight=jnp.int32(0),
        ac_bonus=jnp.int8(0),
        is_two_handed=jnp.bool_(False),
    )
    new_inv = state.inventory.replace(items=new_item)
    state = state.replace(inventory=new_inv)
    inv_glyphs = build_inv_glyphs(state)
    assert int(inv_glyphs[0]) == GLYPH_OBJ_OFF + type_id
