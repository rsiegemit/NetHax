"""Wave 8 vendor-parity tests — divergences surfaced and fixed in the
comprehensive Wave 8 audit.

Each test pins one divergence:
  - internal[0] now reflects scoring.deepest_level (was current_level)
  - internal[7] now reflects raw nutrition (was hunger_state enum)
  - blstats[BL_CONDITION] reflects timed_statuses bitmask (was always 0)
  - HUD role rank title uses state.player_role (was hard-coded role 0)
  - Default player name autogen returns a stable seed-derived string

Citations are inline above each test.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.nle_obs import build_nle_observation, role_rank_title
from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.constants.blstats import (
    BL_CONDITION,
    BL_MASK_BLIND,
    BL_MASK_CONF,
    BL_MASK_STUN,
    BL_MASK_HALLU,
    BL_MASK_STONE,
    BL_MASK_SLIME,
    BL_MASK_STRNGL,
    BL_MASK_FOODPOIS,
    BL_MASK_TERMILL,
    BL_MASK_PARLYZ,
    BL_MASK_LEV,
    BL_MASK_FLY,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env_and_state():
    """Build a NethaxEnv and reset to get a real state + obs.

    Done once per module because env.reset triggers a long JIT compile.
    """
    env = NethaxEnv()
    state, obs = env.reset(jax.random.PRNGKey(0))
    return env, state, obs


# ---------------------------------------------------------------------------
# internal[0] — deepest_lev_reached (vendor: vendor/nle/win/rl/winrl.cc:278)
# ---------------------------------------------------------------------------
def test_internal_0_is_deepest_level(env_and_state):
    """internal[0] must equal scoring.deepest_level (vendor parity).

    Cite: vendor/nle/win/rl/winrl.cc::update_observation lines 278-287 maps
    `internal[0] = deepest_lev_reached(FALSE)`.
    """
    env, state, obs = env_and_state
    internal = np.asarray(obs["internal"])
    deepest = int(state.scoring.deepest_level)
    current = int(state.dungeon.current_level)
    # internal[0] should equal max(deepest_level, current_level) — by parity
    # with vendor it tracks the maximum dungeon depth ever visited.
    assert int(internal[0]) == max(deepest, current)


def test_internal_0_tracks_deepest_after_stair_down(env_and_state):
    """After advancing down a stair, deepest_level must be non-decreasing.

    Cite: vendor/nethack/src/dungeon.c::deepest_lev_reached.
    """
    env, state, _ = env_and_state
    # Directly bump current_level on a copy and re-derive obs; verify the
    # observation reflects the new deepest.
    new_state = state.replace(
        dungeon=state.dungeon.replace(current_level=jnp.int8(5)),
        scoring=state.scoring.replace(deepest_level=jnp.int8(5)),
    )
    obs = build_nle_observation(new_state)
    assert int(np.asarray(obs["internal"])[0]) == 5


# ---------------------------------------------------------------------------
# internal[7] — uhunger raw nutrition (vendor: vendor/nle/win/rl/winrl.cc:286)
# ---------------------------------------------------------------------------
def test_internal_7_is_raw_nutrition(env_and_state):
    """internal[7] must equal status.nutrition (vendor's u.uhunger).

    Cite: vendor/nle/win/rl/winrl.cc::update_observation line 286 maps
    `internal[7] = u.uhunger` (raw counter, 0..2000 typical).
    """
    env, state, obs = env_and_state
    internal = np.asarray(obs["internal"])
    nutrition = int(state.status.nutrition)
    assert int(internal[7]) == nutrition
    # Default game start is 900 (eat.c line 129).
    assert int(internal[7]) == 900


def test_internal_7_changes_with_nutrition(env_and_state):
    """If we drain nutrition, internal[7] follows."""
    env, state, _ = env_and_state
    new_state = state.replace(
        status=state.status.replace(nutrition=jnp.int32(150)),
    )
    obs = build_nle_observation(new_state)
    assert int(np.asarray(obs["internal"])[7]) == 150


# ---------------------------------------------------------------------------
# BL_CONDITION bitmask (vendor: vendor/nethack/include/botl.h:107-134)
# ---------------------------------------------------------------------------
def test_bl_condition_zero_at_start(env_and_state):
    """At game start, no timed statuses are active -> BL_CONDITION is 0."""
    env, state, obs = env_and_state
    assert int(np.asarray(obs["blstats"])[BL_CONDITION]) == 0


def _set_timed_status(state, slot: int, turns: int):
    """Helper: set a single TimedStatus counter on a state."""
    ts = state.status.timed_statuses.at[slot].set(jnp.int32(turns))
    return state.replace(status=state.status.replace(timed_statuses=ts))


@pytest.mark.parametrize("slot, mask", [
    (TimedStatus.BLIND,         BL_MASK_BLIND),
    (TimedStatus.CONFUSION,     BL_MASK_CONF),
    (TimedStatus.STUNNED,       BL_MASK_STUN),
    (TimedStatus.HALLUCINATION, BL_MASK_HALLU),
    (TimedStatus.STONED,        BL_MASK_STONE),
    (TimedStatus.SLIMED,        BL_MASK_SLIME),
    (TimedStatus.STRANGLED,     BL_MASK_STRNGL),
    (TimedStatus.FROZEN,        BL_MASK_PARLYZ),
    (TimedStatus.LEVITATION_TMP, BL_MASK_LEV),
    (TimedStatus.FLYING_TMP,    BL_MASK_FLY),
])
def test_bl_condition_individual_bits(env_and_state, slot, mask):
    """Each TimedStatus countdown maps to the correct BL_MASK_* bit.

    Cite: vendor/nethack/include/botl.h lines 107-134 (BL_MASK_* values).
    """
    env, state, _ = env_and_state
    new_state = _set_timed_status(state, int(slot), 5)
    obs = build_nle_observation(new_state)
    cond = int(np.asarray(obs["blstats"])[BL_CONDITION])
    assert (cond & mask) == mask


def test_bl_condition_multiple_bits(env_and_state):
    """Multiple simultaneous statuses OR their bits together."""
    env, state, _ = env_and_state
    ts = state.status.timed_statuses
    ts = ts.at[int(TimedStatus.BLIND)].set(jnp.int32(10))
    ts = ts.at[int(TimedStatus.STUNNED)].set(jnp.int32(3))
    new_state = state.replace(
        status=state.status.replace(timed_statuses=ts),
    )
    obs = build_nle_observation(new_state)
    cond = int(np.asarray(obs["blstats"])[BL_CONDITION])
    assert (cond & BL_MASK_BLIND) == BL_MASK_BLIND
    assert (cond & BL_MASK_STUN) == BL_MASK_STUN


def test_bl_condition_foodpois_vs_termill(env_and_state):
    """SICK with sick_kind=1 (food poisoning) sets FOODPOIS; otherwise TERMILL.

    Cite: vendor/nethack/src/timeout.c::sick_dialogue, eat.c::poisoned.
    """
    env, state, _ = env_and_state
    # food poisoning path
    ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(jnp.int32(5))
    fp_state = state.replace(
        status=state.status.replace(timed_statuses=ts, sick_kind=jnp.int8(1)),
    )
    obs_fp = build_nle_observation(fp_state)
    cond_fp = int(np.asarray(obs_fp["blstats"])[BL_CONDITION])
    assert (cond_fp & BL_MASK_FOODPOIS) == BL_MASK_FOODPOIS
    assert (cond_fp & BL_MASK_TERMILL) == 0
    # chronic illness path
    ti_state = state.replace(
        status=state.status.replace(timed_statuses=ts, sick_kind=jnp.int8(2)),
    )
    obs_ti = build_nle_observation(ti_state)
    cond_ti = int(np.asarray(obs_ti["blstats"])[BL_CONDITION])
    assert (cond_ti & BL_MASK_TERMILL) == BL_MASK_TERMILL
    assert (cond_ti & BL_MASK_FOODPOIS) == 0


# ---------------------------------------------------------------------------
# HUD role rank title (vendor: vendor/nethack/src/botl.c::rank_of)
# ---------------------------------------------------------------------------
def test_role_rank_title_valkyrie_level_1():
    """A level-1 Valkyrie's rank title is 'Stripling', not 'Digger'.

    Cite: vendor/nethack/src/botl.c::rank_of (lines 331-358) — Valkyrie is
    role index 11 in nethax's Role enum (alphabetical: Archeologist=0,
    Barbarian=1, …, Tourist=10, Valkyrie=11, Wizard=12), and the
    _ROLE_RANK_TITLES table uses that same indexing.
    """
    assert role_rank_title(11, 1) == "Stripling"


def test_role_rank_title_archeologist_level_1():
    """A level-1 Archeologist's rank title is 'Digger'."""
    assert role_rank_title(0, 1) == "Digger"


def test_pygame_status_uses_role_idx():
    """_format_status_lines accepts a role_idx parameter (Wave 8 fix)."""
    from Nethax.ui.pygame_app import _format_status_lines
    # Build a minimal obs dict
    blstats = np.zeros(27, dtype=np.int64)
    blstats[18] = 1   # BL_XP level 1
    obs = {
        "blstats": blstats,
        "message": np.zeros(256, dtype=np.uint8),
    }
    # Role.VALKYRIE = 11 in the alphabetical Role enum (see constants/roles.py).
    name_line, _, _ = _format_status_lines(obs, role_idx=11)
    assert "Stripling" in name_line
    # role_idx 0 (Archeologist) returns "Digger" at level 1
    name_line0, _, _ = _format_status_lines(obs, role_idx=0)
    assert "Digger" in name_line0


# ---------------------------------------------------------------------------
# Player-name autogen
# ---------------------------------------------------------------------------
def test_autogen_player_name_is_stable():
    """Same seed -> same name (deterministic)."""
    from Nethax.ui.pygame_app import autogen_player_name
    assert autogen_player_name(0) == autogen_player_name(0)
    assert autogen_player_name(42) == autogen_player_name(42)


def test_autogen_player_name_format():
    """Name looks like 'Prefix<digits>' (not the legacy 'Player')."""
    from Nethax.ui.pygame_app import autogen_player_name
    name = autogen_player_name(742)
    assert name != "Player"
    assert any(c.isdigit() for c in name)


# ---------------------------------------------------------------------------
# Smoke test: full obs shape unchanged
# ---------------------------------------------------------------------------
def test_obs_shape_unchanged(env_and_state):
    """Wave 8 changes must not alter the NLE-canonical obs schema."""
    env, state, obs = env_and_state
    assert np.asarray(obs["blstats"]).shape == (27,)
    assert np.asarray(obs["internal"]).shape == (9,)
    assert np.asarray(obs["glyphs"]).shape == (21, 79)
