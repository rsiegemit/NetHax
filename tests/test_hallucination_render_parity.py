"""Hallucination render parity tests.

Vendor refs:
  display.c:599  — monster glyph randomization when hallucinating
  display.c:340  — object glyph randomization when hallucinating
  do_name.c:1199 — rndmonnam(): random monster name for look output

Covers:
  - Normal state (no hallucination): glyphs unchanged.
  - Hallucinating state: monster glyph differs from true glyph across timesteps.
  - Per-frame determinism: same (timestep, tile) → same scrambled glyph.
  - Scramble changes across frames: different timestep → different glyph (usually).
  - look.py: build_look_text returns random name (not true name) when hallucinating.
  - Terrain glyphs are never scrambled by hallucination.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.obs.nle_obs import build_glyphs
from Nethax.nethax.obs.look import build_look_text
from Nethax.nethax.constants.glyphs import (
    GLYPH_MON_OFF, GLYPH_PET_OFF, GLYPH_INVIS_OFF,
    GLYPH_OBJ_OFF, GLYPH_CMAP_OFF, NO_GLYPH,
)
from Nethax.nethax.constants.monsters import MONSTERS

_RNG = jax.random.PRNGKey(7)

# HALLUCINATION is timed_statuses index 10 (TimedStatus.HALLUCINATION = 10).
_HALLU_IDX = 10


def _default_state() -> EnvState:
    return EnvState.default(rng=_RNG)


def _with_hallucination(state: EnvState, timer: int = 50) -> EnvState:
    """Return state with the HALLUCINATION timer set to `timer`."""
    new_ts = state.status.timed_statuses.at[_HALLU_IDX].set(jnp.int32(timer))
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


def _place_monster(state: EnvState, row: int, col: int, entry_idx: int = 0) -> EnvState:
    """Place a live, visible NPC monster at (row, col) with given entry_idx."""
    mai = state.monster_ai
    # Slot 0 is reserved for pet in env.py; use slot 1 to avoid conflicts.
    slot = 1
    new_mai = mai.replace(
        alive=mai.alive.at[slot].set(True),
        pos=mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
    )
    # Mark tile as visible and explored so the glyph renders.
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1
    new_visible = state.visible.at[row, col].set(True)
    new_explored = state.explored.at[branch, level, row, col].set(True)
    return state.replace(
        monster_ai=new_mai,
        visible=new_visible,
        explored=new_explored,
    )


# ---------------------------------------------------------------------------
# test_normal_state_no_scramble
# ---------------------------------------------------------------------------

def test_normal_state_no_scramble():
    """When HALLUCINATION timer is 0, build_glyphs is the baseline (no scramble).

    Place a monster at a visible tile and verify glyphs match the expected
    GLYPH_MON_OFF + entry_idx value.
    """
    entry_idx = 5   # arbitrary valid monster index
    state = _default_state()
    state = _place_monster(state, row=10, col=10, entry_idx=entry_idx)

    glyphs_baseline = build_glyphs(state)
    glyphs_no_hallu = build_glyphs(state)

    # Glyphs must be stable across calls with same state (deterministic).
    assert jnp.array_equal(glyphs_baseline, glyphs_no_hallu)

    # The monster tile should carry the true glyph.
    tile_glyph = int(glyphs_baseline[10, 10])
    expected = GLYPH_MON_OFF + entry_idx
    assert tile_glyph == expected, (
        f"normal state: expected glyph {expected}, got {tile_glyph}"
    )


# ---------------------------------------------------------------------------
# test_hallucinating_scrambles_monsters
# ---------------------------------------------------------------------------

def test_hallucinating_scrambles_monsters():
    """When hallucinating, monster glyph differs from the true glyph on most frames."""
    entry_idx = 5
    state = _default_state()
    state = _place_monster(state, row=10, col=10, entry_idx=entry_idx)
    true_glyph = GLYPH_MON_OFF + entry_idx

    state_hallu = _with_hallucination(state, timer=50)

    # Collect scrambled glyphs across 50 different timesteps.
    scrambled_glyphs = set()
    for t in range(50):
        s = state_hallu.replace(timestep=jnp.int32(t))
        g = int(build_glyphs(s)[10, 10])
        scrambled_glyphs.add(g)

    # The scrambled glyph must still be in the monster glyph range.
    for g in scrambled_glyphs:
        assert GLYPH_MON_OFF <= g < GLYPH_INVIS_OFF, (
            f"scrambled glyph {g} out of monster range [{GLYPH_MON_OFF}, {GLYPH_INVIS_OFF})"
        )

    # Across many timesteps, the true glyph should not be the only value seen.
    # (With 381 monsters and 50 trials the probability of always hitting the same
    # value is negligibly small, ~(1/381)^49.)
    assert len(scrambled_glyphs) > 1, (
        "hallucination scramble produced only one glyph value across 50 timesteps"
    )


# ---------------------------------------------------------------------------
# test_hallucinating_scrambles_consistent_within_step
# ---------------------------------------------------------------------------

def test_hallucinating_scrambles_consistent_within_step():
    """Same (timestep, tile) always yields the same scrambled glyph."""
    entry_idx = 3
    state = _default_state()
    state = _place_monster(state, row=5, col=15, entry_idx=entry_idx)
    state_hallu = _with_hallucination(state, timer=50)
    # Fix a specific timestep.
    state_t = state_hallu.replace(timestep=jnp.int32(42))

    glyphs_a = int(build_glyphs(state_t)[5, 15])
    glyphs_b = int(build_glyphs(state_t)[5, 15])

    assert glyphs_a == glyphs_b, (
        f"same (timestep, tile) produced different scrambled glyphs: {glyphs_a} vs {glyphs_b}"
    )


# ---------------------------------------------------------------------------
# test_hallucinating_scrambles_changes_per_step
# ---------------------------------------------------------------------------

def test_hallucinating_scrambles_changes_per_step():
    """Different timesteps should (usually) produce different scrambled glyphs.

    We check that across 40 timesteps at least 5 distinct values appear —
    this fails with probability < (5/381)^35 ≈ 10^{-55}.
    """
    entry_idx = 7
    state = _default_state()
    state = _place_monster(state, row=8, col=20, entry_idx=entry_idx)
    state_hallu = _with_hallucination(state, timer=50)

    seen = set()
    for t in range(40):
        s = state_hallu.replace(timestep=jnp.int32(t))
        seen.add(int(build_glyphs(s)[8, 20]))

    assert len(seen) >= 5, (
        f"expected >= 5 distinct scrambled glyphs across 40 timesteps, got {len(seen)}: {seen}"
    )


# ---------------------------------------------------------------------------
# test_look_text_random_monster_name
# ---------------------------------------------------------------------------

def test_look_text_random_monster_name():
    """When hallucinating, build_look_text returns a name other than the true name.

    Vendor do_name.c:1199 rndmonnam() replaces the true monster name.
    We run 30 trials; in each trial the returned name should not equal the true
    name (probability of always matching: (1/381)^30 ≈ 10^{-78}).
    """
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))

    # Place monster at a non-player tile.
    entry_idx = 10
    row, col = 5, 5
    state = _place_monster(state, row=row, col=col, entry_idx=entry_idx)
    state_hallu = _with_hallucination(state, timer=50)

    true_name = None
    if 0 <= entry_idx < len(MONSTERS) and MONSTERS[entry_idx] is not None:
        true_name = MONSTERS[entry_idx].name or "creature"

    if true_name is None:
        pytest.skip("Monster entry_idx has no name; cannot test name scramble.")

    # Run 30 trials with different random seeds (look.py uses random.choice).
    import random
    different_count = 0
    for seed in range(30):
        random.seed(seed)
        name = build_look_text(state_hallu, row, col)
        # build_look_text returns "the <name>"; strip the article for comparison.
        bare = name
        if bare.startswith("the "):
            bare = bare[4:]
        if bare != true_name:
            different_count += 1

    assert different_count >= 25, (
        f"hallucinated name equalled true name '{true_name}' too often; "
        f"only {different_count}/30 trials differed"
    )


# ---------------------------------------------------------------------------
# test_terrain_glyphs_not_scrambled
# ---------------------------------------------------------------------------

def test_terrain_glyphs_not_scrambled():
    """Terrain (cmap) glyphs must not be altered by hallucination.

    Vendor: only monster and object glyphs are scrambled; terrain is unaffected.
    We check a visible floor tile (no monster, no object on it) across several
    hallucinating timesteps.
    """
    from Nethax.nethax.constants import TileType

    state = _default_state()
    branch = int(state.dungeon.current_branch)
    level = int(state.dungeon.current_level) - 1

    # Place a floor tile at (15, 50) and mark it visible + explored.
    row, col = 15, 50
    new_terrain = state.terrain.at[branch, level, row, col].set(
        jnp.int8(int(TileType.FLOOR))
    )
    new_visible = state.visible.at[row, col].set(True)
    new_explored = state.explored.at[branch, level, row, col].set(True)
    state = state.replace(
        terrain=new_terrain,
        visible=new_visible,
        explored=new_explored,
    )

    # Baseline (no hallucination).
    baseline_glyph = int(build_glyphs(state)[row, col])
    assert GLYPH_CMAP_OFF <= baseline_glyph < NO_GLYPH, (
        f"expected a cmap glyph at ({row},{col}), got {baseline_glyph}"
    )

    # Hallucinating — terrain glyph must be stable.
    state_hallu = _with_hallucination(state, timer=50)
    for t in range(20):
        s = state_hallu.replace(timestep=jnp.int32(t))
        g = int(build_glyphs(s)[row, col])
        assert g == baseline_glyph, (
            f"terrain glyph at ({row},{col}) was scrambled at timestep {t}: "
            f"expected {baseline_glyph}, got {g}"
        )
