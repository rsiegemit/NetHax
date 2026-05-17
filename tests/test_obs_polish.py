"""Wave 4 NLE observation polish tests.

Covers the 4 keys wired in Wave 4 Phase 3:
  - colors  (terrain / monster / object color overlays)
  - specials (corpse / pile / trap / object flag bits)
  - internal (NLE 9-int internal state vector)
  - screen_descriptions (per-tile description bytes)

Plus the 17-key contract check confirming full NLE-parity coverage.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants import TileType
from Nethax.nethax.constants.glyphs import GLYPH_MON_OFF
from Nethax.nethax.obs.nle_obs import (
    NLE_OBSERVATION_KEYS,
    build_colors,
    build_specials,
    build_internal,
    build_screen_descriptions,
    build_nle_observation,
)


_RNG = jax.random.PRNGKey(123)


def _default_state() -> EnvState:
    return EnvState.default(rng=_RNG)


def _bytes_to_str(b: jnp.ndarray) -> str:
    """Convert a 1-D uint8 array into a python str (truncated at null)."""
    raw = bytes(int(x) & 0xFF for x in b)
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


# ---------------------------------------------------------------------------
# colors — terrain / monster / object overlays
# ---------------------------------------------------------------------------

def test_colors_terrain_floor_is_gray():
    """Floor tile color must be 7 (CLR_GRAY)."""
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
    colors = build_colors(state)
    assert int(colors[8, 8]) == 7


def test_colors_monster_dog_is_white():
    """Dog monster glyph maps to CLR_WHITE (15, HI_DOMESTIC)."""
    from Nethax.nethax.constants.monsters import MONSTERS
    from Nethax.nethax.obs.nle_obs import _GLYPH_TO_COLOR
    dog_idx = next(i for i, m in enumerate(MONSTERS) if m.name == "dog")
    dog_glyph = GLYPH_MON_OFF + dog_idx
    assert int(_GLYPH_TO_COLOR[dog_glyph]) == 15  # HI_DOMESTIC


def test_colors_lava_is_red():
    """LAVA tile must map to color CLR_RED (1)."""
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[branch, level, 5, 5].set(jnp.int8(TileType.LAVA))
    new_explored = state.explored.at[branch, level, 5, 5].set(True)
    state = state.replace(
        terrain=new_terrain,
        explored=new_explored,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )
    colors = build_colors(state)
    assert int(colors[5, 5]) == 1  # CLR_RED


# ---------------------------------------------------------------------------
# specials — trap / pile / corpse / object flags
# ---------------------------------------------------------------------------

def test_specials_pile_flag_set_when_multiple_items():
    """Two+ stacks at same tile must set MG_OBJPILE (0x80).

    Vendor bits (display.h:1002): MG_OBJPILE = 0x80.
    """
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1

    gi = state.ground_items
    # Stack slot 0: weapon, slot 1: armor (both occupied)
    cat = gi.category.at[branch, level, 9, 10, 0].set(jnp.int8(2))
    cat = cat.at[branch, level, 9, 10, 1].set(jnp.int8(3))
    new_gi = gi.replace(category=cat)
    state = state.replace(ground_items=new_gi)

    specials = build_specials(state)
    val = int(specials[9, 10])
    assert (val & 0x80) != 0, f"MG_OBJPILE bit not set, got {val}"


def test_specials_corpse_flag():
    """A corpse on the floor must set MG_CORPSE (0x02).

    Vendor bits (display.h:996): MG_CORPSE = 0x02.
    """
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1

    gi = state.ground_items
    # corpse = FOOD_CLASS (7), type_id = 260 (corpse object index)
    cat = gi.category.at[branch, level, 4, 4, 0].set(jnp.int8(7))
    typ = gi.type_id.at[branch, level, 4, 4, 0].set(jnp.int16(260))
    new_gi = gi.replace(category=cat, type_id=typ)
    state = state.replace(ground_items=new_gi)

    specials = build_specials(state)
    val = int(specials[4, 4])
    assert (val & 0x02) != 0, f"MG_CORPSE bit not set, got {val}"


def test_specials_hero_bit_at_player_position():
    """The player tile must set MG_HERO (0x01).

    Vendor bits (display.h:995): MG_HERO = 0x01.
    """
    state = _default_state().replace(
        player_pos=jnp.array([10, 12], dtype=jnp.int16),
    )
    specials = build_specials(state)
    val = int(specials[10, 12])
    assert (val & 0x01) != 0, f"MG_HERO bit not set at player pos, got {val}"


def test_specials_shape_and_dtype():
    """Shape (21, 79) uint8 — matches NLE nleobs.h specials field."""
    state = _default_state()
    specials = build_specials(state)
    assert specials.shape == (21, 79)
    assert specials.dtype == jnp.uint8


# ---------------------------------------------------------------------------
# internal — NLE 9-int internal vector
# ---------------------------------------------------------------------------

def test_internal_array_length_9():
    """internal must be shape (9,) int32."""
    state = _default_state()
    internal = build_internal(state)
    assert internal.shape == (9,)
    assert internal.dtype == jnp.int32


def test_internal_current_level_matches_dungeon_state():
    """internal[0] should track current dungeon level (deepest_lev_reached proxy)."""
    state = _default_state()
    new_dungeon = state.dungeon.replace(current_level=jnp.int8(5))
    state = state.replace(dungeon=new_dungeon)
    internal = build_internal(state)
    assert int(internal[0]) == 5


def test_internal_xplevel_matches_player_score():
    """internal[8] should mirror player score (NLE u.urexp)."""
    state = _default_state()
    # Bump scoring.score
    new_scoring = state.scoring.replace(score=jnp.int32(4242))
    state = state.replace(scoring=new_scoring)
    internal = build_internal(state)
    assert int(internal[8]) == 4242


def test_internal_stairs_down_flag():
    """internal[4] should be 1 when player stands on a down-stair."""
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    # Place player at (3, 3) and put a STAIRCASE_DOWN there
    new_terrain = state.terrain.at[branch, level, 3, 3].set(jnp.int8(TileType.STAIRCASE_DOWN))
    state = state.replace(
        terrain=new_terrain,
        player_pos=jnp.array([3, 3], dtype=jnp.int16),
    )
    internal = build_internal(state)
    assert int(internal[4]) == 1


# ---------------------------------------------------------------------------
# screen_descriptions — per-tile description bytes
# ---------------------------------------------------------------------------

def test_screen_descriptions_shape():
    """Shape (21, 79, 80) uint8 — matches NLE nleobs.h screen_descriptions."""
    state = _default_state()
    sd = build_screen_descriptions(state)
    assert sd.shape == (21, 79, 80)
    assert sd.dtype == jnp.uint8


def test_screen_descriptions_wall_string():
    """A wall tile's description should start with 'wall'."""
    # vendor pager.c::lookat: tile must be in visible to avoid last_seen_terrain shadowing
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[branch, level, 2, 2].set(jnp.int8(TileType.WALL))
    new_explored = state.explored.at[branch, level, 2, 2].set(True)
    new_visible = state.visible.at[2, 2].set(True)
    state = state.replace(
        terrain=new_terrain,
        explored=new_explored,
        visible=new_visible,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )
    sd = build_screen_descriptions(state)
    desc = _bytes_to_str(sd[2, 2])
    assert desc.startswith("wall"), f"expected 'wall*', got {desc!r}"


def test_screen_descriptions_floor_string():
    """A floor tile's description should be 'floor'."""
    # vendor pager.c::lookat: tile must be in visible to avoid last_seen_terrain shadowing
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[branch, level, 7, 9].set(jnp.int8(TileType.FLOOR))
    new_explored = state.explored.at[branch, level, 7, 9].set(True)
    new_visible = state.visible.at[7, 9].set(True)
    state = state.replace(
        terrain=new_terrain,
        explored=new_explored,
        visible=new_visible,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )
    sd = build_screen_descriptions(state)
    desc = _bytes_to_str(sd[7, 9])
    assert desc == "floor", f"expected 'floor', got {desc!r}"


def test_screen_descriptions_lava_string():
    """A lava tile description should mention 'lava'."""
    # vendor pager.c::lookat: tile must be in visible to avoid last_seen_terrain shadowing
    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    new_terrain = state.terrain.at[branch, level, 5, 5].set(jnp.int8(TileType.LAVA))
    new_explored = state.explored.at[branch, level, 5, 5].set(True)
    new_visible = state.visible.at[5, 5].set(True)
    state = state.replace(
        terrain=new_terrain,
        explored=new_explored,
        visible=new_visible,
        player_pos=jnp.array([0, 0], dtype=jnp.int16),
    )
    sd = build_screen_descriptions(state)
    desc = _bytes_to_str(sd[5, 5])
    assert "lava" in desc, f"expected 'lava' in desc, got {desc!r}"


# ---------------------------------------------------------------------------
# Full 17-key contract — confirms NLE parity is complete
# ---------------------------------------------------------------------------

def test_all_17_obs_keys_present_after_wave_4():
    """build_obs(state).keys() must equal the 17 NLE observation keys."""
    state = _default_state()
    obs = build_nle_observation(state)
    assert set(obs.keys()) == set(NLE_OBSERVATION_KEYS)
    assert len(NLE_OBSERVATION_KEYS) == 17


def test_internal_and_screen_descriptions_jit_safe():
    """build_internal + build_screen_descriptions must be jit-compilable."""
    state = _default_state()
    jit_internal = jax.jit(build_internal)
    jit_sd = jax.jit(build_screen_descriptions)
    _ = jit_internal(state)
    _ = jit_sd(state)
