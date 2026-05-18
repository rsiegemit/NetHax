"""Endgame subsystem — per-plane entry requirements, per-turn damage, ascension.

Implements the four endgame-plane hazard mechanics described in
vendor/nethack/src/endgame.c (the canonical ``do_endgame`` / ``level_tele``
logic) and vendor/nethack/dat/{earth,air,fire,water,astral}.lua.

Per-plane entry requirements (endgame.c::check_enter_plane):
    Plane of Earth (index 0) — no requirement.
    Plane of Air   (index 1) — requires LEVITATION intrinsic.
    Plane of Fire  (index 2) — requires RESIST_FIRE intrinsic.
    Plane of Water (index 3) — requires MAGIC_BREATHING intrinsic.
    Plane of Astral (index 4) — no intrinsic requirement (offering triggers).

Per-turn damage (endgame.c::endgame_env_damage, called from allmain.c loop):
    Plane of Fire  — 1 HP/turn without RESIST_FIRE (lava-edge burn).
    Plane of Water — 1 HP/turn without MAGIC_BREATHING (drowning).

Ascension via try_ascend (endgame.c::done_ascend / pray.c::dosacrifice):
    Player must be on Astral Plane (Endgame L5), standing on a coaligned
    altar, carrying the Amulet of Yendor.

Citations:
    vendor/nethack/src/endgame.c — check_enter_plane, endgame_env_damage
    vendor/nethack/src/end.c     — done(ASCENDED) path
    vendor/nethack/src/pray.c    — dosacrifice / real_amulet offering
    vendor/nethack/include/prop.h — LEVITATION=48, FIRE_RES=1, MAGICAL_BREATHING=52
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from Nethax.nethax.dungeon.branches import Branch
from Nethax.nethax.subsystems.status_effects import Intrinsic
from Nethax.nethax.subsystems.ascension import (
    check_ascension,
    ascend,
)

# ---------------------------------------------------------------------------
# N_PLANES and per-plane required intrinsic
# Index = Endgame level - 1  (0=Earth, 1=Air, 2=Fire, 3=Water, 4=Astral)
# -1 means no intrinsic requirement.
# Citation: vendor/nethack/src/endgame.c check_enter_plane
# ---------------------------------------------------------------------------

N_PLANES: int = 5

# _PLANE_ENTRY_REQUIREMENTS[plane_idx] -> Intrinsic index (-1 = none)
_PLANE_ENTRY_REQUIREMENTS: tuple[int, ...] = (
    -1,                              # Earth  — no requirement
    int(Intrinsic.LEVITATION),       # Air    — must levitate
    int(Intrinsic.RESIST_FIRE),      # Fire   — must resist fire
    int(Intrinsic.MAGIC_BREATHING),  # Water  — must breathe magically
    -1,                              # Astral — no intrinsic requirement
)

# Planes that deal per-turn passive damage.
# Citation: vendor/nethack/src/endgame.c endgame_env_damage
_FIRE_PLANE_IDX:  int = 2   # Endgame level 3 (1-based)
_WATER_PLANE_IDX: int = 3   # Endgame level 4 (1-based)

# Public aliases matching dungeon/endgame_levels.py naming convention.
BRANCH_PLANE_EARTH:  int = 1
BRANCH_PLANE_AIR:    int = 2
BRANCH_PLANE_FIRE:   int = 3
BRANCH_PLANE_WATER:  int = 4
BRANCH_PLANE_ASTRAL: int = 5


def _plane_idx(state) -> jnp.ndarray:
    """Return 0-based plane index for the current endgame level."""
    return (state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1))


def _has_intrinsic(state, intrinsic_id: int) -> jnp.ndarray:
    """Return bool — player has the given intrinsic (permanent or timed)."""
    perm  = state.status.intrinsics[intrinsic_id]
    timed = state.status.timed_intrinsics[intrinsic_id] > jnp.int32(0)
    return perm | timed


def _in_endgame(state) -> jnp.ndarray:
    """Return bool — player is currently in the Endgame branch."""
    return state.dungeon.current_branch == jnp.int8(int(Branch.ENDGAME))


# ---------------------------------------------------------------------------
# on_enter_plane
# ---------------------------------------------------------------------------

def on_enter_plane(state, plane_idx: int):
    """Apply per-plane setup when the player first enters a plane.

    Currently a no-op setup (damage is applied per-turn by tick_plane_damage;
    level generation is handled by dungeon/endgame.py generators).  Kept as
    an explicit hook so future waves can add portal-spawning, speech events,
    etc., mirroring vendor/nethack/src/endgame.c::do_endgame().

    Citation: vendor/nethack/src/endgame.c do_endgame()

    Args:
        state:     EnvState.
        plane_idx: 0-based plane index (0=Earth … 4=Astral).

    Returns:
        (Possibly modified) EnvState.
    """
    # Placeholder: no immediate side-effects in current wave.
    return state


# ---------------------------------------------------------------------------
# tick_plane_damage
# ---------------------------------------------------------------------------

def tick_plane_damage(state, rng: jax.Array):
    """Apply per-turn environmental damage on Fire and Water planes.

    Fire  (plane 2): 1 HP damage per turn unless player has RESIST_FIRE.
    Water (plane 3): 1 HP damage per turn unless player has MAGIC_BREATHING.

    Citation:
        vendor/nethack/src/endgame.c::endgame_env_damage
        vendor/nethack/src/hack.c::lava_effects (fire damage on lava tiles)
        vendor/nethack/src/mon.c::water_damage  (drowning without amphibiousness)

    Args:
        state: EnvState.
        rng:   JAX PRNG key (unused; kept for API symmetry with other tickers).

    Returns:
        EnvState with player_hp decremented if applicable.
    """
    del rng

    in_eg   = _in_endgame(state)
    pidx    = _plane_idx(state)

    # Fire plane damage: 1 HP unless resistant.
    on_fire   = pidx == jnp.int32(_FIRE_PLANE_IDX)
    has_fres  = _has_intrinsic(state, int(Intrinsic.RESIST_FIRE))
    fire_dmg  = in_eg & on_fire & ~has_fres

    # Water plane damage: 1 HP unless magic breathing.
    on_water  = pidx == jnp.int32(_WATER_PLANE_IDX)
    has_mbrth = _has_intrinsic(state, int(Intrinsic.MAGIC_BREATHING))
    water_dmg = in_eg & on_water & ~has_mbrth

    take_dmg  = fire_dmg | water_dmg
    new_hp    = jnp.where(take_dmg, state.player_hp - jnp.int32(1), state.player_hp)
    # Clamp to 0; death detection handled by the normal done-check in env.py.
    new_hp    = jnp.maximum(new_hp, jnp.int32(0))
    return state.replace(player_hp=new_hp)


# ---------------------------------------------------------------------------
# try_ascend
# ---------------------------------------------------------------------------

def try_ascend(state):
    """If the player is on a coaligned Astral altar with the Amulet, ascend.

    Delegates to subsystems/ascension.py::check_ascension + ascend, which
    implement the full vendor condition (Astral Plane + matching altar +
    Amulet of Yendor).

    Citation:
        vendor/nethack/src/end.c::done(ASCENDED)
        vendor/nethack/src/pray.c::dosacrifice / real_amulet

    Args:
        state: EnvState; player has presumably just dropped/offered the Amulet.

    Returns:
        EnvState — state.done=True and Achievement.ASCENDED recorded if the
        ascension condition is met; otherwise state is returned unchanged.
    """
    return jax.lax.cond(
        check_ascension(state),
        ascend,
        lambda s: s,
        state,
    )
