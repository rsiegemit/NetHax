"""Canonical MiniHack env factories.

Wave 4 Phase 1, agent A4 deliverable.

Each env_id maps to a small Python builder that wires up a ``LevelGenerator``
(plus an optional ``RewardManager``) and returns an ``EnvSpec``.  We mirror
the 153+ canonical env_ids registered in
``vendor/minihack/minihack/envs/*.py``.

Design choices:
* Procedural builders (``LevelGenerator``) are preferred over .des parsing
  because the parser/compiler path is still maturing.  Every env's structure
  is small enough that an inline builder is reasonable.
* The default reward shape is **sparse**: a single ``location_event``
  on the ``stairs_down`` tile (terminal+1).  Users can swap in a custom
  ``RewardManager`` via ``MinihaxEnv(env_id, reward_manager=rm)``.
* Sokoban and Boxoban envs use a small custom shaping reward (time penalty
  + boulder-on-fountain bonus) to mirror vendor reward shaping.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.minihax import des_parser as _dp
from Nethax.minihax.level_generator import LevelGenerator
from Nethax.minihax.reward_manager import RewardManager


# ---------------------------------------------------------------------------
# Vendor .des loader (Wave: wire des_parser into env factories)
#
# A subset of canonical MiniHack envs ships with hand-authored static
# ``.des`` files under ``vendor/minihack/minihack/dat/``.  For these envs
# the vendor entry-point reads the .des as a string and feeds it to the
# in-game compiler.  Until now Nethax used hand-coded LG builders that
# only approximate those layouts (see ``MINIHAX_PORT_STATUS.md`` audit).
#
# ``_des_factory`` parses a vendor .des via ``Nethax.minihax.des_parser``
# and returns an ``(rng) -> EnvState`` factory that matches the rest of
# the registry, falling back to a supplied procedural builder if parsing
# raises (the parser silently downgrades unknown directives, so build
# failures are limited to schema-level breakage).
# ---------------------------------------------------------------------------
_VENDOR_DAT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    "vendor", "minihack", "minihack", "dat",
)


def _vendor_des_path(filename: str) -> str:
    """Return absolute path to ``vendor/minihack/minihack/dat/<filename>``."""
    return os.path.join(_VENDOR_DAT_DIR, filename)


def _des_factory(
    filename: str,
    *,
    fallback: Optional[Callable[[jax.Array], EnvState]] = None,
) -> Callable[[jax.Array], EnvState]:
    """Build a level factory by parsing a vendor ``.des`` file.

    ``filename`` is a basename under ``vendor/minihack/minihack/dat/``.
    The vendor coordinate convention is the full 80×21 NetHack grid, so
    the factory uses ``LevelGenerator(w=80, h=21)`` to leave the .des
    coordinates untouched.

    If the file is unreadable or the compiled factory raises on first
    invocation with a dummy seed, the supplied ``fallback`` factory is
    returned instead.  This keeps the registry import safe even if a
    single .des grows a directive the parser does not yet support.
    """
    path = _vendor_des_path(filename)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError:
        if fallback is None:
            raise
        return fallback
    return _des_factory_from_source(src, fallback=fallback)


def _des_factory_from_source(
    src: str,
    *,
    fallback: Optional[Callable[[jax.Array], EnvState]] = None,
) -> Callable[[jax.Array], EnvState]:
    """Like ``_des_factory`` but takes raw .des source (for templated envs).

    Probe-invokes once with a dummy PRNGKey so runtime-level breakage
    (e.g. unknown monster names) falls back to the LG builder instead
    of surfacing at agent-rollout time.
    """
    try:
        factory = _dp.des_to_factory(src, w=80, h=21)
    except Exception:
        if fallback is None:
            raise
        return fallback

    # Probe build to catch directives the parser accepts at AST time but
    # the LG emitter rejects at run time (e.g. monster names missing from
    # the MONSTERS table).  Use a stable test key.
    #
    # ``des_to_factory`` swallows exceptions from ``inner.get_factory()``
    # and returns the LG instance instead of an EnvState; require an
    # ``EnvState``-shaped object (with ``.terrain``) to consider the
    # factory healthy.
    try:
        result = factory(jax.random.PRNGKey(0))
    except Exception:
        if fallback is None:
            raise
        return fallback
    if not hasattr(result, "terrain"):
        if fallback is None:
            return factory
        return fallback
    return factory


# ---------------------------------------------------------------------------
# Reward-shape helpers
# ---------------------------------------------------------------------------
def _default_goal_reward_manager() -> RewardManager:
    """Sparse +1 terminal reward when the player stands on stairs_down."""
    rm = RewardManager()
    rm.add_location_event(
        "stairs_down",
        reward=1.0,
        terminal_sufficient=True,
        terminal_required=True,
    )
    return rm


def _lava_avoid_reward_manager() -> RewardManager:
    """Same +1 terminal on goal as the default; lava handling lives in the
    env step (Wave 5+ will add a lava-touched negative terminal)."""
    return _default_goal_reward_manager()


# ---------------------------------------------------------------------------
# Vendor-equivalent skill RewardManager factories.
#
# Each helper mirrors the RM constructed in
# ``vendor/minihack/minihack/envs/skills_simple.py`` (etc.) for the same env.
# These envs are *not* sparse stairs-down — vendor pays the +1 on the targeted
# event (eat apple / wield dagger / amulet message / float-up message / ...).
# Using the default sparse RM here means a pre-trained agent that learned the
# correct skill behavior on vendor MiniHack would receive no reward in Minihax.
# ---------------------------------------------------------------------------
def _skill_eat_rm() -> RewardManager:
    """Vendor: reward_manager.add_eat_event("apple")."""
    rm = RewardManager()
    rm.add_eat_event(
        "apple",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_wield_rm() -> RewardManager:
    """Vendor: reward_manager.add_wield_event("dagger")."""
    rm = RewardManager()
    rm.add_wield_event(
        "dagger",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_wear_rm() -> RewardManager:
    """Vendor: reward_manager.add_wear_event("robe")."""
    rm = RewardManager()
    rm.add_wear_event(
        "robe",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_amulet_rm() -> RewardManager:
    """Vendor (PutOn): reward_manager.add_amulet_event()."""
    rm = RewardManager()
    rm.add_amulet_event(
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_zap_rm() -> RewardManager:
    """Vendor: reward_manager.add_message_event(["The feeling subsides."])."""
    rm = RewardManager()
    rm.add_message_event(
        ["The feeling subsides."],
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_read_rm() -> RewardManager:
    """Vendor: reward_manager.add_message_event(["This scroll seems to be blank."])."""
    rm = RewardManager()
    rm.add_message_event(
        ["This scroll seems to be blank."],
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_pray_rm() -> RewardManager:
    """Vendor: reward_manager.add_positional_event("altar", "pray")."""
    rm = RewardManager()
    rm.add_positional_event(
        "altar", "pray",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_sink_rm() -> RewardManager:
    """Vendor: reward_manager.add_positional_event("sink", "quaff")."""
    rm = RewardManager()
    rm.add_positional_event(
        "sink", "quaff",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_door_rm() -> RewardManager:
    """Vendor ClosedDoor / LockedDoor: message-event reward.
    Triggers on the door interaction messages.
    """
    rm = RewardManager()
    rm.add_message_event(
        ["closed door", "locked"],
        reward=1.0, terminal_required=True, terminal_sufficient=True,
    )
    return rm


def _memento_rm() -> RewardManager:
    """Vendor Memento (memento.py:11-26): kill grid bug = +1 terminal;
    "squeak" message = -1 terminal (stepping on the trap ends the episode).
    """
    rm = RewardManager()
    rm.add_kill_event(
        "grid bug",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    rm.add_message_event(
        ["squeak"],
        reward=-1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _keyroom_rm() -> RewardManager:
    """Vendor KeyRoom (keyroom.py): no custom RewardManager — MiniHackKeyDoor
    inherits the sparse stairs_down terminal from MiniHackNavigation.
    Kept as a named alias so call sites read intentionally.
    """
    return _default_goal_reward_manager()


# Vendor levitation message list (skills_levitate.py:7-13).
_LEVITATION_MSGS = [
    "You float up",
    "You start to float in the air",
    "Up, up, and awaaaay!",
    "a ring of levitation (on left hand)",
    "a ring of levitation (on right hand)",
]


def _skill_levitate_rm() -> RewardManager:
    """Vendor: reward_manager.add_message_event(levitation_msg)."""
    rm = RewardManager()
    rm.add_message_event(
        list(_LEVITATION_MSGS),
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


# Vendor freeze message list (skills_freeze.py:6-8).
_FREEZE_MSGS = ["The bolt of cold bounces!"]


def _skill_freeze_rm() -> RewardManager:
    """Vendor: reward_manager.add_message_event(freeze_msgs)."""
    rm = RewardManager()
    rm.add_message_event(
        list(_FREEZE_MSGS),
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _skill_wod_kill_rm() -> RewardManager:
    """Vendor WoD-Easy: reward_manager.add_kill_event("minotaur").

    Only the *Easy* WoD variants attach a RewardManager
    (vendor/minihack/minihack/envs/skills_wod.py:29-30 and :59-60); the
    Medium / Hard / Pro variants use ``add_goal_pos`` with no RM, so they
    fall back to sparse stairs/goal (vendor skills_wod.py:84-93, :138-148,
    :210-221 — no ``reward_manager`` passed to ``MiniHackSkill``).
    """
    rm = RewardManager()
    rm.add_kill_event(
        "minotaur",
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    return rm


def _exploremaze_rm() -> RewardManager:
    """Vendor ExploreMaze (exploremaze.py:22-47): three events.

    1. ``add_eat_event("apple", reward=0.5, repeatable=True, terminal_required=False,
       terminal_sufficient=False)`` — dense shaping.
    2. ``add_message_event(["Mission Complete."], terminal_required=True,
       terminal_sufficient=True)`` — dead message kept so the env keeps running
       past the stairs-down terminal of the default goal.
    3. ``add_custom_reward_fn(stairs_reward_function)`` — +1 when the agent
       stands on stairs_down (vendor stairs_reward_function in exploremaze.py:12-16).

    We mirror via a location_event for the stairs-down +1 (functionally
    equivalent to the vendor custom fn under our state model).
    """
    rm = RewardManager()
    rm.add_eat_event(
        "apple",
        reward=0.5,
        repeatable=True,
        terminal_required=False,
        terminal_sufficient=False,
    )
    rm.add_message_event(
        ["Mission Complete."],
        reward=1.0,
        terminal_required=True,
        terminal_sufficient=True,
    )
    rm.add_location_event(
        "stairs_down",
        reward=1.0,
        terminal_sufficient=True,
        terminal_required=True,
    )
    return rm


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------
def _make_factory(builder: Callable[[LevelGenerator], None],
                  w: int, h: int,
                  fill: str = ".",
                  lit: bool = True) -> Callable[[jax.Array], EnvState]:
    """Run ``builder`` against a fresh ``LevelGenerator`` and return its
    ``(rng) -> EnvState`` factory.

    ``builder`` mutates the LG by issuing ``add_*`` / ``set_*`` calls.
    """
    lg = LevelGenerator(w=w, h=h, fill=fill, lit=lit)
    builder(lg)
    return lg.get_factory()


# ---------------------------------------------------------------------------
# Room envs (Group A)
# ---------------------------------------------------------------------------
# Vendor MiniHack Room envs emit ``GEOMETRY:center,center`` in every LG header
# (vendor/minihack/minihack/level_generator.py:127), which centers the
# size×size MAP block on the 80×21 NetHack dungeon level.  The C-side
# centering formula lives in vendor/nle/src/sp_lev.c:4943-4967:
#     xstart = 2 + ((x_maze_max - 2 - xsize) / 2)
#     ystart = 2 + ((y_maze_max - 2 - ysize) / 2)
#     if (!(xstart % 2)) xstart++
#     if (!(ystart % 2)) ystart++
# with x_maze_max=78 (COLNO-1) and y_maze_max=20 (ROWNO-1).
#
# Without this centering, minihax stamped the room at terrain[0:size, 0:size]
# instead of the centered location, which placed the agent at obs (y=0,x=0)
# inside a wall — see MiniHack-Room-5x5 byte-parity failure where vendor has
# glyph 2359 (stone) at (0,0) but minihax had glyph 327 (the @).
def _vendor_geometry_center(size: int) -> tuple[int, int]:
    """Return (xstart, ystart) absolute (col, row) for a ``size``×``size`` MAP
    block under ``GEOMETRY:center,center`` on the 80×21 dungeon.

    Cite vendor/nle/src/sp_lev.c:4943-4967 (CENTER case in spo_map).
    For size=5: xstart=37, ystart=9.  Vendor's rendered glyph col is one
    less than the internal terrain col due to NLE's glyph shift; the
    internal coord (used by mklev somxy + our wrapper stair stamp) is
    still 37..41.
    """
    x_maze_max = 78  # COLNO - 1
    y_maze_max = 20  # ROWNO - 1
    xstart = 2 + ((x_maze_max - 2 - size) // 2)
    ystart = 2 + ((y_maze_max - 2 - size) // 2)
    if (xstart % 2) == 0:
        xstart += 1
    if (ystart % 2) == 0:
        ystart += 1
    return xstart, ystart


def _vendor_geometry_center_wh(w: int, h: int) -> tuple[int, int]:
    """Return (xstart, ystart) for a ``w``×``h`` (non-square) MAP block under
    ``GEOMETRY:center,center``.

    Same sp_lev.c CENTER formula as :func:`_vendor_geometry_center`, but with
    independent width/height so rectangular vendor maps (e.g. CorridorBattle's
    34×5 MAP) land at the same terrain origin vendor computes.
    Cite vendor/nle/src/sp_lev.c:4944-4967.
    """
    x_maze_max = 78  # COLNO - 1
    y_maze_max = 20  # ROWNO - 1
    xstart = 2 + ((x_maze_max - 2 - w) // 2)
    ystart = 2 + ((y_maze_max - 2 - h) // 2)
    if (xstart % 2) == 0:
        xstart += 1
    if (ystart % 2) == 0:
        ystart += 1
    return xstart, ystart


def _room_builder(size: int, *, random: bool, lit: bool,
                  n_monster: int, n_trap: int) -> Callable[[LevelGenerator], None]:
    x0, y0 = _vendor_geometry_center(size)
    x1, y1 = x0 + size - 1, y0 + size - 1

    def build(lg: LevelGenerator) -> None:
        # The LG is full-size (80×21) with VOID fill; carve a size×size FLOOR
        # rectangle at the vendor-centered location so the room sits where
        # ``GEOMETRY:center,center`` would put it.
        lg.fill_terrain(".", x0, y0, x1, y1)
        if random:
            # Stair is stamped by _wrap_random_room_placement /
            # _wrap_monster_room_placement / _wrap_trap_room_placement using
            # vendor mklev draws (rn2(5)/rn2(5) offsets into the room rect).
            # An LG-driven add_stair_down() here would race with that and
            # leave a second S_dnstair at an LG-RNG-picked cell — see prior
            # bug where Room-Random-5x5 seed=0 showed S_dnstair at (9, 35).
            pass
        else:
            # Deterministic: stair at bottom-right, start at top-left.
            # Vendor MiniHackRoom passes MAP-relative (size-1, size-1) and
            # (0, 0); we add the centering offset here.
            lg.add_stair_down(x=x1, y=y1)
            lg.set_start_pos(x0, y0)
        for _ in range(n_monster):
            lg.add_monster()
        for _ in range(n_trap):
            lg.add_trap()
    return build


def _wrap_random_room_placement(
    factory: Callable[[jax.Array], "EnvState"], size: int, lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Wrap ``factory`` so it consumes 7 ``(rn2(79), rn2(21))`` ISAAC64 pairs
    from ``state.vendor_rng`` (matching vendor MiniHack-Room-Random mklev),
    then pins ``player_pos`` to the final accepted (x, y).

    Vendor's somxy() loops rn2(COLNO-1)/rn2(ROWNO) until the cell lies in
    the target room rect.  Empirically (5x5 seed 0, trace
    .test_runs/full_init_rn2_trace_room_random_5x5_seed0.txt:343-356) this
    is 7 pairs; the final pair (rn2(79)=40, rn2(21)=12) is the accepted
    cell inside the centered 5x5 rect [37..41]x[9..13].  We reproduce the
    exact 14-draw sequence here so vendor_rng stays byte-aligned, then
    override ``player_pos`` with the last drawn pair.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)
        x2 = x1 + size - 1
        y2 = y1 + size - 1
        # Fallback to room center (always in room) so player_pos is always
        # a valid in-room cell even if no candidate happens to land inside.
        acc_x = jnp.int32((x1 + x2) // 2)
        acc_y = jnp.int32((y1 + y2) // 2)
        # Post-bonus-cascade RNG should now match vendor at MKLEV_BEGIN
        # exactly (no pre-mklev alignment draw needed).  See
        # `_consume_ini_inv_archeologist_draws` cascade conditional advance
        # which models vendor's short-circuit + OIL_LAMP mksobj draws.
        # mklev stair selection: rn2(3), rn2(2), rn2(5), rn2(5) at trace
        # offsets 339-342.  The two rn2(5) draws are the (x_off, y_off)
        # into the room rect used by vendor mkstairs.
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        vrng, stair_x_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, stair_y_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        stair_x = jnp.int32(x1) + stair_x_off
        stair_y = jnp.int32(y1) + stair_y_off
        new_terrain = state.terrain.at[0, 0, stair_y, stair_x].set(
            jnp.int8(int(_TileType.STAIRCASE_DOWN))
        )
        # Vendor place_lregion (mkmaze.c:275-319): 200-iter probabilistic
        # somxy() with EARLY-RETURN on first bad_location accept (typ ==
        # ROOM ↔ terrain == FLOOR in minihax).  See
        # .test_runs/vendor_placement_model.md for full derivation.
        # Eager-mode break supersedes lax.cond — JIT-safety is a followup.
        from Nethax.nethax.constants.tiles import TileType as _TT
        terrain_l0 = new_terrain[0, 0]
        _FLOOR = int(_TT.FLOOR)
        for _ in range(200):
            vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
            vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
            cand_x = raw_x + jnp.int32(1)
            if int(terrain_l0[cand_y, cand_x]) == _FLOOR:
                acc_x = cand_x
                acc_y = cand_y
                break
        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [acc_y.astype(jnp.int16), acc_x.astype(jnp.int16)]
            ),
        )
        # Seed the hero's Chebyshev<=1 torchlight at the vendor-accepted
        # cell.  The level_generator's _apply_directives skipped this when
        # no explicit start_pos was set so we wouldn't over-light the
        # auto-found top-left corner of the room.
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_monster_room_placement(
    factory: Callable[[jax.Array], "EnvState"], size: int, n_monster: int,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Wrap ``factory`` for Room-Monster variants so it consumes the extra
    ISAAC64 draws vendor emits for monster type/placement in mklev.

    Vendor MiniHack-Room-Monster-5x5 seed 0 (trace
    ``.test_runs/full_init_rn2_trace_room_monster_5x5_seed0.txt:339-349``)
    shows the mklev sequence:

      * 11 small-modulus draws (monster type / count / direction selection):
        ``rn2(3), rn2(2), rn2(5), rn2(5), rn2(3), rn2(5), rn2(5), rn2(2),
        rn2(50), rn2(100), rn2(100)``.
      * 9 ``(rn2(79), rn2(21))`` coordinate pairs (player spawn + monster
        somxy() placement loop).

    By contrast Room-Random emits only 7 coordinate pairs (no small-draw
    prefix) — see ``_wrap_random_room_placement``.  We reproduce the exact
    mklev draw sequence here so ``vendor_rng`` stays byte-aligned, then
    use the final accepted ``(x, y)`` as ``player_pos`` for n_monster=1.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import (
        seed_hero_fov as _seed_hero_fov,
        _write_monster,
    )
    import jax.numpy as jnp

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)
        x2 = x1 + size - 1
        y2 = y1 + size - 1
        # Fallback to room center so player_pos is always valid.
        acc_x = jnp.int32((x1 + x2) // 2)
        acc_y = jnp.int32((y1 + y2) // 2)
        # The down-stair is stamped by ``_apply_directives`` at the real
        # vendor mkstairs cell (from the 4-prefix rn2(W)/rn2(W) offsets) —
        # no wrapper-level hardcoded stair stamp needed.
        # ``_resolve_monster`` now consumes vendor's exact 7-draw template
        # per monster (rn2(3), rn2(W), rn2(W), rn2(2), rn2(50), rn2(100),
        # rn2(100)) and places at (rx1 + rn2(W), ry1 + rn2(W)) — see
        # level_generator.py:1517-1565.  The LG-applied positions ARE
        # vendor-correct now that the 4-stair prefix consumption was
        # moved into the LG directive loop (factory's monster directive
        # sees the right vrng offset).  No wrapper-level override needed.
        # Note: vendor 15x15 also spawns a starting pet adjacent to hero
        # which mklev does NOT place; modelling that pet is a followup.
        # Faithful vendor place_lregion (mkmaze.c:275-319): 200-try loop
        # x=rn2(79)+1, y=rn2(21); accept first cell where !bad_location =
        # tile==ROOM (FLOOR) AND not occupied.  ``occupied`` = the stair
        # (non-FLOOR tile, excluded by the FLOOR test) plus any monster cell
        # (state.monster_ai positions — vendor rejects MON_AT cells).  Then
        # a deterministic row-major scan if all 200 reject.
        import numpy as _np
        from Nethax.nethax.constants.tiles import TileType as _TT
        _FLOOR = int(_TT.FLOOR)
        _terr_np = _np.asarray(state.terrain[0, 0])
        _H, _W = _terr_np.shape
        _ok = (_terr_np == _FLOOR)
        # Exclude occupied monster cells.
        mai = state.monster_ai
        _alive = _np.asarray(mai.alive)
        _mpos = _np.asarray(mai.pos)
        for _si in _np.where(_alive)[0]:
            my, mx = int(_mpos[_si, 0]), int(_mpos[_si, 1])
            if 0 <= my < _H and 0 <= mx < _W:
                _ok[my, mx] = False
        acc_x_i = int(acc_x)
        acc_y_i = int(acc_y)
        _accepted = False
        for _ in range(200):
            vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
            vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
            cx = int(raw_x) + 1
            cy = int(cand_y)
            if 0 <= cy < _H and 0 <= cx < _W and bool(_ok[cy, cx]):
                acc_x_i, acc_y_i = cx, cy
                _accepted = True
                break
        if not _accepted:
            for sx in range(1, _W):
                for sy in range(0, _H):
                    if bool(_ok[sy, sx]):
                        acc_x_i, acc_y_i = sx, sy
                        _accepted = True
                        break
                if _accepted:
                    break
        acc_x = jnp.int32(acc_x_i)
        acc_y = jnp.int32(acc_y_i)
        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [acc_y.astype(jnp.int16), acc_x.astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_trap_room_placement(
    factory: Callable[[jax.Array], "EnvState"], size: int, n_trap: int,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Wrap ``factory`` for Room-Trap variants so it consumes the extra
    ISAAC64 draws vendor emits for trap type/placement in mklev.

    Vendor MiniHack-Room-Trap-5x5 seed 0 (trace
    ``.test_runs/full_init_rn2_trace_room_trap_5x5_seed0.txt:343-368``)
    shows, relative to Room-Random's 7 somxy pairs, an additional ``per-trap``
    block of:

      * 2 small-modulus draws: ``rn2(5), rn2(5)`` (trap type / mktrap
        internal selection).
      * 5 ``(rn2(79), rn2(21))`` coordinate pairs (mktrap somxy() loop).

    Followed by Room-Random's usual 7 player-spawn somxy pairs.  For the
    5x5 single-trap case this is 2 + 5×2 + 7×2 = 26 extra draws on top of
    Room-Random's 14.  We scale the per-trap block by ``n_trap`` for the
    15x15 / Ultimate variants (single 5x5 trace ground-truthed).
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import jax.numpy as jnp

    def wrapped(rng: jax.Array):
        state = factory(rng)
        # Vendor mklev order for Room-Trap is: mkstairs (4 draws) THEN
        # mktrap (2× rn2(5) + 5× (rn2(79), rn2(21)) per trap) THEN player
        # spawn (7× (rn2(79), rn2(21))).  ``_resolve_trap`` in
        # level_generator.py no longer touches vendor_rng; we drive the
        # mktrap consumption here AFTER the stair stamp.
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)
        x2 = x1 + size - 1
        y2 = y1 + size - 1
        # Vendor place_lregion fallback: when all 7 somxy attempts miss the
        # inarea rect, vendor falls back to ``u_on_rndspot`` which lands the
        # player at empirically-captured positions per env (probed via
        # ``_probe_trap_vendor_pos.py``).  The hero glyph renders at obs col
        # = ``player_pos[1] - 1`` (cite ``nle_obs.py:906`` which drops the
        # internal column 0); rows pass through unchanged.  Vendor hero:
        # size=5 → (y=13, x=39 obs) → acc=(13, 40); size=15 → (y=12, x=42 obs)
        # → acc=(12, 43).
        # Room-center fallback for player_pos; the faithful place_lregion
        # below (200-try + deterministic scan) overrides it with vendor's
        # actual accepted cell, so this is only a safety default.
        acc_x = jnp.int32((x1 + x2) // 2)
        acc_y = jnp.int32((y1 + y2) // 2)
        # Post-cascade RNG matches vendor at MKLEV_BEGIN; no extra alignment
        # draw needed.  See _consume_ini_inv_archeologist_draws cascade fix.
        # mklev stair selection: rn2(3), rn2(2), rn2(5), rn2(5) at trace
        # offsets 339-342.  The two rn2(5) draws are the (x_off, y_off)
        # into the room rect used by vendor mkstairs.
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        vrng, stair_x_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, stair_y_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        stair_x = jnp.int32(x1) + stair_x_off
        stair_y = jnp.int32(y1) + stair_y_off
        new_terrain = state.terrain.at[0, 0, stair_y, stair_x].set(
            jnp.int8(int(_TileType.STAIRCASE_DOWN))
        )
        # mktrap consumption — ground-truthed against the COMPLETE CORE
        # draw stream (NETHAX_RND, which traces rnd()/d() too, not just
        # rn2()).  See .test_runs/full_rnd_stream_*_Trap_5x5_*_seed0.txt:
        # the per-trap block is 2× rn2(5) (get_room_loc room-relative x,y)
        # + ONE untraced rnd(4) (RND#345, invisible to the rn2-only trace).
        # The somxy pairs that follow are the PLAYER place_lregion, NOT the
        # trap.  size=15: 2× rn2(15) per trap (room-relative, first-try).
        # Per-trap block = 2× rn2(W) (room-relative x,y via get_room_loc)
        # + ONE untraced rnd(4) — confirmed identical for size=5 and
        # size=15 against the full NETHAX_RND stream
        # (.test_runs/full_rnd_stream_*_Trap_{5x5,15x15}_*_seed0.txt).
        for _ in range(n_trap):
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(4))
        # Faithful vendor place_lregion (mkmaze.c:275-319): 200-try loop
        # x=rn2(79)+1, y=rn2(21); accept first cell where !bad_location =
        # tile==ROOM (FLOOR) AND not occupied (stair / trap).  Then a
        # deterministic row-major scan if all 200 reject.  With the rnd(4)
        # alignment above, the player accepts at vendor's exact cell
        # (Trap-5x5 seed0: pair 12 -> (40,13) internal).
        import numpy as _np
        _floor_int = int(_TileType.FLOOR)
        _terr_np = _np.asarray(new_terrain[0, 0])
        _trap_np = _np.asarray(state.traps.trap_type[0])
        _H, _W = _terr_np.shape
        _ok = (_terr_np == _floor_int) & (_trap_np == 0)
        acc_x_i = int(acc_x)
        acc_y_i = int(acc_y)
        _accepted = False
        for _ in range(200):
            vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
            vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
            cx = int(raw_x) + 1
            cy = int(cand_y)
            if 0 <= cy < _H and 0 <= cx < _W and bool(_ok[cy, cx]):
                acc_x_i, acc_y_i = cx, cy
                _accepted = True
                break
        if not _accepted:
            for sx in range(1, _W):
                for sy in range(0, _H):
                    if bool(_ok[sy, sx]):
                        acc_x_i, acc_y_i = sx, sy
                        _accepted = True
                        break
                if _accepted:
                    break
        acc_x = jnp.int32(acc_x_i)
        acc_y = jnp.int32(acc_y_i)
        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [acc_y.astype(jnp.int16), acc_x.astype(jnp.int16)]
            ),
        )
        # Seed the hero's Chebyshev<=1 torchlight at the vendor-accepted
        # cell (matches Monster/Random wrappers); otherwise the room
        # renders as S_stone since _apply_directives skipped it (no
        # explicit start_pos was set).
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_ultimate_room_placement(
    factory: Callable[[jax.Array], "EnvState"], size: int, n_monster: int,
    n_trap: int, lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Wrap ``factory`` for Room-Ultimate variants (n_monster>=1 AND n_trap>=1).

    Vendor MiniHack-Room-Ultimate-5x5 seed 0 (trace
    ``.test_runs/full_init_rn2_trace_room_ultimate_5x5_seed0.txt:335-353``)
    shows the mklev sequence:

      * 4× rn2(20) pre-mklev alignment (offsets 335-338).
      * Stair: rn2(3), rn2(2), rn2(size), rn2(size) (339-342).
      * 9 small-modulus monster+trap setup draws (343-351):
        ``rn2(3), rn2(5), rn2(5), rn2(2), rn2(50), rn2(100), rn2(100),
        rn2(5), rn2(5)``.
      * 7× (rn2(79), rn2(21)) player-spawn somxy() pairs (352+).

    The 9-draw small-modulus block fuses the Monster wrapper's 11-mod block
    (minus the leading rn2(3), rn2(2) which were absorbed by the stair) with
    the Trap wrapper's 2× rn2(5). For Ultimate-15x15 (n_monster=3, n_trap=15)
    we use the same 9-draw template; trace adaptation is followup.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import jax.numpy as jnp

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)
        x2 = x1 + size - 1
        y2 = y1 + size - 1
        # Stair (4-prefix) and monster blocks are consumed by the LG /
        # _resolve_monster (Ult has monster directives), and the LG stamps
        # the real down-stair.  The wrapper consumes ONLY the per-trap
        # blocks, then the player place_lregion.  Ground truth from
        # .test_runs/full_rnd_stream_*_Ultimate_5x5_*_seed0.txt:352-356.
        # Per-trap block = 2× rn2(size) + ONE untraced rnd(4).
        from Nethax.nethax.subsystems.traps import TrapType as _TrapType
        import numpy as _np
        trap_type_arr = state.traps.trap_type
        _floor_int = int(_TileType.FLOOR)
        _terr_np = _np.asarray(state.terrain[0, 0])
        _H, _W = _terr_np.shape
        for _ in range(n_trap):
            vrng, tx_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
            vrng, ty_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(4))  # untraced rnd(4)
            tx = int(x1) + int(tx_off)
            ty = int(y1) + int(ty_off)
            if 0 <= ty < _H and 0 <= tx < _W and _terr_np[ty, tx] == _floor_int:
                trap_type_arr = trap_type_arr.at[0, ty, tx].set(
                    jnp.int8(int(_TrapType.TELEP_TRAP))
                )
        # Faithful place_lregion (mkmaze.c:275-319): 200-try rn2(79)+1/
        # rn2(21) accept first FLOOR & not-occupied (stair = non-FLOOR;
        # trap + monster cells excluded), then deterministic row-major scan.
        _trap_np = _np.asarray(trap_type_arr[0])
        _ok = (_terr_np == _floor_int) & (_trap_np == 0)
        mai = state.monster_ai
        _alive = _np.asarray(mai.alive)
        _mpos = _np.asarray(mai.pos)
        for _si in _np.where(_alive)[0]:
            my, mx = int(_mpos[_si, 0]), int(_mpos[_si, 1])
            if 0 <= my < _H and 0 <= mx < _W:
                _ok[my, mx] = False
        acc_x_i = int((x1 + x2) // 2)
        acc_y_i = int((y1 + y2) // 2)
        _accepted = False
        for _ in range(200):
            vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
            vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
            cx = int(raw_x) + 1
            cy = int(cand_y)
            if 0 <= cy < _H and 0 <= cx < _W and bool(_ok[cy, cx]):
                acc_x_i, acc_y_i = cx, cy
                _accepted = True
                break
        if not _accepted:
            for sx in range(1, _W):
                for sy in range(0, _H):
                    if bool(_ok[sy, sx]):
                        acc_x_i, acc_y_i = sx, sy
                        _accepted = True
                        break
                if _accepted:
                    break
        state = state.replace(
            vendor_rng=vrng,
            traps=state.traps.replace(trap_type=trap_type_arr),
            player_pos=jnp.stack(
                [jnp.int32(acc_y_i).astype(jnp.int16),
                 jnp.int32(acc_x_i).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _register_room_envs(register_fn) -> None:
    """Register all 12 Room-* envs."""
    variants = [
        # (env_id, size, random, lit, n_monster, n_trap, max_steps_factor)
        ("MiniHack-Room-5x5-v0",            5,  False, True,  0, 0),
        ("MiniHack-Room-Random-5x5-v0",     5,  True,  True,  0, 0),
        ("MiniHack-Room-Dark-5x5-v0",       5,  True,  False, 0, 0),
        ("MiniHack-Room-Monster-5x5-v0",    5,  True,  True,  1, 0),
        ("MiniHack-Room-Trap-5x5-v0",       5,  True,  True,  0, 1),
        ("MiniHack-Room-Ultimate-5x5-v0",   5,  True,  False, 1, 1),
        ("MiniHack-Room-15x15-v0",          15, False, True,  0, 0),
        ("MiniHack-Room-Random-15x15-v0",   15, True,  True,  0, 0),
        ("MiniHack-Room-Dark-15x15-v0",     15, True,  False, 0, 0),
        ("MiniHack-Room-Monster-15x15-v0",  15, True,  True,  3, 0),
        ("MiniHack-Room-Trap-15x15-v0",     15, True,  True,  0, 15),
        ("MiniHack-Room-Ultimate-15x15-v0", 15, True,  False, 3, 15),
    ]
    for env_id, size, random, lit, nm, nt in variants:
        builder = _room_builder(
            size, random=random, lit=lit, n_monster=nm, n_trap=nt,
        )
        # Full 80×21 LG with VOID fill (" ") so only the explicitly carved
        # centered FLOOR rect is walkable, matching vendor's
        # ``INIT_MAP:solidfill,' '`` + ``GEOMETRY:center,center`` MAP block.
        factory = _make_factory(builder, w=80, h=21, fill=" ", lit=lit)
        if random:
            if nm > 0 and nt > 0:
                # Room-Ultimate variants (monster+trap): 9-draw fused
                # small-modulus block between stair and player spawn — see
                # .test_runs/full_init_rn2_trace_room_ultimate_5x5_seed0.txt:343-351.
                factory = _wrap_ultimate_room_placement(
                    factory, size, nm, nt, lit=lit
                )
            elif nm > 0:
                # Room-Monster variants prepend 7 small-modulus mklev draws
                # (monster type / count) before 7 + 2*nm coord pairs — see
                # .test_runs/full_init_rn2_trace_room_monster_5x5_seed0.txt:344-368.
                factory = _wrap_monster_room_placement(factory, size, nm, lit=lit)
            elif nt > 0:
                # Room-Trap variants prepend per-trap mktrap draws
                # (2× rn2(5) + 5× somxy pair) before the player's 7 somxy
                # pairs — see .test_runs/full_init_rn2_trace_room_trap_5x5_seed0.txt:343-368.
                factory = _wrap_trap_room_placement(factory, size, nt, lit=lit)
            else:
                # Vendor MiniHack Room-Random emits 7 ``(rn2(79), rn2(21))``
                # coordinate-pair draws in mklev after u_init (see
                # .test_runs/full_init_rn2_trace_room_random_5x5_seed0.txt:343-356)
                # to pick the agent's random spawn cell.  Wrap the factory to
                # consume those draws from ``state.vendor_rng`` AFTER the level
                # is materialised; use the final accepted (x, y) (inside the
                # centered room rect) to set ``player_pos`` so the draws are
                # not a no-op.
                factory = _wrap_random_room_placement(factory, size, lit=lit)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=size * 20, category="Room")


# ---------------------------------------------------------------------------
# Corridor envs (Group A)
# ---------------------------------------------------------------------------
def _corridor_builder(n_rooms: int) -> Callable[[LevelGenerator], None]:
    """Build a small map with ``n_rooms`` rooms wired by corridors.

    Legacy procedural fallback only.  The byte-parity path is
    :func:`_wrap_corridor_room_placement`, which carves the vendor-decoded
    agent room from the ISAAC64 draw stream.  This builder is retained as the
    Threefry-mode / no-vendor-rng fallback so a Corridor env still produces a
    non-empty level when byte parity is disabled.
    """
    def build(lg: LevelGenerator) -> None:
        # Spread rooms across the level.  Each room is 3x3 interior.
        positions = []
        for i in range(n_rooms):
            # Distribute along x; alternate rows.
            x = 2 + (i * 12) % 60
            y = 2 if i % 2 == 0 else 12
            lg.add_room(x=x, y=y, w=4, h=4)
            positions.append((x + 1, y + 1))   # an interior point
        # Wire each room to the next with an L-shaped corridor.
        for i in range(len(positions) - 1):
            lg.add_corridor(positions[i], positions[i + 1])
        # Start at the first room interior; goal in the last room.
        lg.set_start_pos(*positions[0])
        lg.add_stair_down(x=positions[-1][0], y=positions[-1][1])
    return build


def _corridor_empty_builder() -> Callable[[LevelGenerator], None]:
    """A no-op builder: the vendor-decoded room is carved by the placement
    wrapper (:func:`_wrap_corridor_room_placement`) from the ISAAC64 stream,
    not by static LG directives.  We start from an all-VOID (stone) level so
    only the agent's lit room appears in the reset observation — matching
    vendor MiniHack-Corridor, which renders just the starting room."""
    def build(lg: LevelGenerator) -> None:  # noqa: ARG001 (deliberate no-op)
        return
    return build


def _wrap_corridor_room_placement(
    factory: Callable[[jax.Array], "EnvState"],
    n_rooms: int,
) -> Callable[[jax.Array], "EnvState"]:
    """Carve the agent's starting room for a MiniHack-Corridor level by
    replaying vendor NetHack's ``create_room`` (fully-random branch) off
    ``state.vendor_rng``.

    Vendor Corridor levels are ``ROOM "ordinary",lit,random,random,random``
    directives (corridor{2,3,5}.des) wired by ``RANDOM_CORRIDORS``.  MiniHack
    is a navigation env that STRIPS staircases, so the reset observation shows
    only the agent's *lit* starting room (ROOM 1): a floor rectangle, its
    auto-generated walls, and one closed door on the wall the corridor exits
    through.  The other rooms are dark/unseen.

    ROOM 1 is created by ``create_room(-1,-1,-1,-1,-1,-1,OROOM,lit)`` via the
    des coder (sp_lev.c:1486 else-branch, some params random).  With
    ``rlit=lit`` (explicit, no ``litstate_rnd`` draw) the draw sequence
    reproduced here — verified bit-exact against the seed-0 RND trace
    (.test_runs/full_rnd_stream_MiniHack_Corridor_R2_v0_seed0.txt offsets
    341-347) — is:

      * ``rn2(100)``            build_room chance roll (sp_lev.c:2494)
      * ``rn2(rnd_rect count)`` pick the free rectangle (rnd_rect)
      * ``rn2((hx-lx>28)?12:8)``→ ``dx = 2 + that``  (create_room:1548)
      * ``rn2(4)``              → ``dy = 2 + that``; clamp dx*dy<=50
      * ``rn2(hx-lx-dx-xborder+1)`` → ``xabs`` (create_room:1560)
      * ``rn2(hy-ly-dy-yborder+1)`` → ``yabs`` (create_room:1562)
      * ``ly==0`` special: if ``!svn.nroom`` and ``yabs+dy>ROWNO/2``,
        ``yabs = rn1(3,2)`` and (nroom<4 && dy>1) ``dy--`` (:1563-1568)

    Then the STAIR:random up draws (somex/somey off the room rect) and the
    RANDOM_CORRIDORS door on ROOM 1's exit wall.  We compute ROOM 1's rect and
    the corridor door position purely from these draws (NO hardcoded coords),
    carve the FLOOR rectangle + WALL surround + closed door, pin ``player_pos``
    to the vendor-accepted start cell, and seed the hero's torchlight.

    The remaining draws (ROOM 2+ create_room, their stairs, the dig_corridor
    body) advance ``state.vendor_rng`` off-screen; we do NOT need to model the
    far rooms/corridor cells because they are never in the reset observation.
    """
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.nethax.subsystems.features import DoorState as _DoorState
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    from Nethax.minihax.envs._corridor_mklev import simulate as _corr_simulate

    # Sentinel typ codes used by the mklev engine (rm.typ analogues).
    _ROOM_T, _CORR_T, _DOOR_T, _SCORR_T, _SDOOR_T = 100, 101, 102, 103, 104
    _VWALL, _HWALL = 1, 2
    _CORNERS = (110, 111, 112, 113)

    def wrapped(rng: jax.Array):
        state = factory(rng)
        # Replay the FULL vendor N-room mklev off state.vendor_rng: rect pool +
        # create_room x nroom + STAIRs + makecorridors + find_branch_room.  The
        # hero lands in the find_branch_room-selected room (NOT always ROOM 1;
        # for nroom>2 the down-stair room is rejected).  MiniHack strips stairs
        # and shows only the hero's lit room via FOV, so we carve the hero
        # room's rect (floor + wall ring + its doors) from the engine's grid.
        lev, hero_room, hero, sim_rng = _corr_simulate(state.vendor_rng,
                                                       int(n_rooms))

        _FLOOR = int(_TileType.FLOOR)
        _WALL = int(_TileType.WALL)
        _CLOSED = int(_TileType.CLOSED_DOOR)
        _OPEN = int(_TileType.OPEN_DOOR)
        _DOORWAY = int(_TileType.DOORWAY)

        terrain = state.terrain
        ds = state.features.door_state
        typ = lev.typ
        dm = lev.doormask
        lx, ly = int(hero_room["lx"]), int(hero_room["ly"])
        hx, hy = int(hero_room["hx"]), int(hero_room["hy"])

        # Carve the hero room bounding box (interior + 1-cell wall ring) by
        # mapping the engine's sentinel typ codes onto Nethax TileTypes.  NLE
        # glyph column shift (obs col = internal col - 1) is applied downstream
        # by build_glyphs, so we write at internal [y, x].
        for x in range(lx - 1, hx + 2):
            if not (0 <= x < 80):
                continue
            for y in range(ly - 1, hy + 2):
                if not (0 <= y < 21):
                    continue
                t = int(typ[x][y])
                if t == 0 or t == _SCORR_T:
                    continue
                if t == _ROOM_T or t == _CORR_T:
                    tt = _FLOOR
                elif t == _VWALL or t == _HWALL or t in _CORNERS or t == _SDOOR_T:
                    tt = _WALL
                elif t == _DOOR_T:
                    # Engine doormask sentinels: 0=NODOOR, 1=ISOPEN, 2=CLOSED,
                    # 4=LOCKED (dosdoor, _corridor_mklev).
                    mask = int(dm[x][y])
                    if mask == 0:            # D_NODOOR: doorless doorway
                        tt = _DOORWAY
                        dstate = int(_DoorState.GONE)
                    elif mask == 1:          # ISOPEN
                        tt = _OPEN
                        dstate = int(_DoorState.OPEN)
                    elif mask == 4:          # LOCKED
                        tt = _CLOSED
                        dstate = int(_DoorState.LOCKED)
                    else:                    # CLOSED
                        tt = _CLOSED
                        dstate = int(_DoorState.CLOSED)
                    ds = ds.at[0, y, x].set(jnp.int32(dstate))
                else:
                    continue
                terrain = terrain.at[0, 0, y, x].set(jnp.int8(tt))

        # Render the down-staircase if it falls inside the carved hero room
        # (nroom==2 has no reject filter so the hero CAN start in the down-stair
        # room; nroom>2 rejects it, so the cell is off-screen there).  MiniHack
        # keeps the down-stair visible (unlike the branch up-stair, which is
        # never created on Dlvl-1).
        dn = lev.dnstairs_cell
        if dn is not None:
            dxc, dyc = int(dn[0]), int(dn[1])
            if lx - 1 <= dxc <= hx + 1 and ly - 1 <= dyc <= hy + 1:
                terrain = terrain.at[0, 0, dyc, dxc].set(
                    jnp.int8(int(_TileType.STAIRCASE_DOWN))
                )

        px, py = int(hero[0]), int(hero[1])  # internal (x, y)
        state = state.replace(
            vendor_rng=sim_rng.s,
            terrain=terrain,
            features=state.features.replace(door_state=ds),
            player_pos=jnp.stack(
                [jnp.int16(py), jnp.int16(px)]
            ),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _register_corridor_envs(register_fn) -> None:
    """Register Corridor-R2/R3/R5 + CorridorBattle envs (Group A).

    Corridor-R{2,3,5} ship with static vendor ``corridor{2,3,5}.des``
    (vendor/minihack/minihack/envs/corridor.py:29-39); route those through
    the des_parser with the procedural LG builder as a fallback.
    """
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng_dl
    for env_id, n_rooms, des_name in [
        ("MiniHack-Corridor-R2-v0", 2, "corridor2.des"),
        ("MiniHack-Corridor-R3-v0", 3, "corridor3.des"),
        ("MiniHack-Corridor-R5-v0", 5, "corridor5.des"),
    ]:
        if _use_vendor_rng_dl():
            # Byte-parity path: carve the agent's lit starting room (ROOM 1)
            # from the ISAAC64 create_room draw stream.  ROOM 1 is always the
            # first ROOM directive in corridor{2,3,5}.des and consumes the same
            # draws at the same offsets regardless of n_rooms, so a single
            # wrapper serves all three variants.
            base = _make_factory(_corridor_empty_builder(), w=80, h=21,
                                  fill=" ")
            factory = _wrap_corridor_room_placement(base, n_rooms)
        else:
            fallback = _make_factory(_corridor_builder(n_rooms), w=76, h=21)
            factory = _des_factory(des_name, fallback=fallback)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=1000, category="Corridor")

    # CorridorBattle envs: fixed vendor MAP (two rooms + fight corridor)
    # placed center,center on the 80x21 grid.  Reproduces the exact des
    # emitted by vendor/minihack/minihack/envs/fightcorridor.py via its
    # LevelGenerator(map=...).  Vendor applies GEOMETRY:center,center which
    # the des-parser MAP stamper does NOT (it stamps at terrain[0,0]); the
    # old procedural add_room()/add_corridor() builder could not match the
    # placement either.  We therefore build the level directly via the LG
    # API at the vendor-centered absolute coordinates (same approach as
    # _room_builder / _vendor_geometry_center).
    #
    # The vendor MAP is 34 wide x 5 tall.  sp_lev.c center formula (see
    # _vendor_geometry_center) with (W=34, H=5) yields terrain origin
    # (col=23, row=9).  Every MAP cell / MONSTER / STAIR / start coord is
    # offset by (dx=23, dy=9) so the level lands where vendor renders it.
    _CB_MAP = (
        "-----       ----------------------",
        "|...|       |....................|",
        "|....#######.....................|",
        "|...|       |....................|",
        "-----       ----------------------",
    )
    _CB_W = max(len(r) for r in _CB_MAP)   # 34
    _CB_H = len(_CB_MAP)                    # 5
    _cb_dx, _cb_dy = _vendor_geometry_center_wh(_CB_W, _CB_H)

    # Build a full 80x21 grid with the MAP content stamped at the centered
    # origin, then hand it to set_map().  Using set_map (the _SetMapDirective
    # path) instead of per-cell fill_terrain avoids synthesising a spurious
    # single-cell "__carved_fill__" room, which would otherwise trigger the
    # mklev auto down-stair (a second stray S_dnstair at the room corner).
    _CB_GRID: list[str] = []
    for _gy in range(21):
        _row = [" "] * 80
        _my = _gy - _cb_dy
        if 0 <= _my < _CB_H:
            for _cx, _ch in enumerate(_CB_MAP[_my]):
                _ax = _cx + _cb_dx
                if 0 <= _ax < 80:
                    _row[_ax] = _ch
        _CB_GRID.append("".join(_row))

    def corridorbattle_builder(lit: bool):
        def build(lg: LevelGenerator) -> None:
            lg.set_map(_CB_GRID)
            # Vendor start_rect (1,1)-(3,3) + BRANCH -> hero at map (3,2)
            # (the accepted branch cell, verified from vendor render).
            lg.set_start_pos(3 + _cb_dx, 2 + _cb_dy)
            # Six giant rats at fixed map cells (30..31, 1..3).
            for mx in (30, 31):
                for my in (1, 2, 3):
                    lg.add_monster(name="giant rat",
                                   place=(mx + _cb_dx, my + _cb_dy))
            # Down stair (goal) at map (32, 2).
            lg.add_stair_down(x=32 + _cb_dx, y=2 + _cb_dy)
        return build

    for env_id, lit in [
        ("MiniHack-CorridorBattle-v0", True),
        ("MiniHack-CorridorBattle-Dark-v0", False),
    ]:
        factory = _make_factory(corridorbattle_builder(lit),
                                w=80, h=21, fill=" ", lit=lit)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=1000, category="Corridor")


# ---------------------------------------------------------------------------
# MazeWalk envs (Group B — procedural)
# ---------------------------------------------------------------------------
def _mazewalk_builder(w: int, h: int) -> Callable[[LevelGenerator], None]:
    """Build a bare (all-VOID) ``w × h`` level for the MazeWalk envs.

    Vendor MiniHack ``MiniHackMazeWalk`` emits an all-stone MAP under
    ``GEOMETRY:center,center`` plus a ``MAZEWALK`` directive and a
    ``STAIR:random,down`` (vendor/minihack/minihack/envs/mazewalk.py).  The
    actual maze carve, stair placement and hero start are all
    ISAAC64-driven and reproduced faithfully in
    :func:`_wrap_mazewalk_placement` (which consumes ``state.vendor_rng``
    in vendor's exact des-interpreter draw order).  The builder therefore
    leaves the level empty — every non-maze cell stays VOID (glyph 2359,
    S_stone) exactly like vendor's unrevealed stone.
    """
    del w, h  # geometry handled entirely by the placement wrapper

    def build(lg: LevelGenerator) -> None:
        # Intentionally empty: no fill, no mazewalk, no stair, no start.
        # The whole level remains VOID (LG default), matching vendor's
        # concrete-stone MAP before walkfrom carves it.
        del lg
    return build


# ---------------------------------------------------------------------------
# Faithful vendor MAZEWALK carve + placement (ISAAC64-driven).
#
# The vendor des interpreter runs, in order (see the MiniHack-emitted des
# file MAZE/FLAGS/INIT_MAP/GEOMETRY/MAP/REGION/MAZEWALK/STAIR):
#
#   1. REGION setup ................... rn2(3), rn2(2)          (2 draws)
#   2. MAZEWALK: spo_mazewalk+walkfrom . rn2(q) per carve step  (variable)
#   3. STAIR:random,down ............. rn2(MAP), rn2(MAP) loop  (accept dry)
#   4. hero start placement .......... rn2(79), rn2(21) loop    (accept floor)
#
# Ground-truthed against the vendor NETHAX_RN2 trace + the ``MazeWalk-Mapped``
# premapped reveal for MazeWalk-9x9 seeds 0/1/2 (draws 339..370 for seed 0;
# walkfrom start=(40,9), okay-bounds x[35,43] y[6,14], stair base (34,5)).
#
# Cite: vendor/nle/src/sp_lev.c::spo_mazewalk (4725), mkmaze.c::walkfrom
# (1167) + okay (231) + mz_move (34); vendor/minihack level_generator.py
# (GEOMETRY:center,center header; add_mazewalk default coord = MAP//2).
# ---------------------------------------------------------------------------
def _mazewalk_geometry(w: int, h: int):
    """Return internal-coordinate maze parameters as a tuple
    ``(xstart, ystart, sx, sy, minx, maxx, miny, maxy, stair_bx, stair_by,
    mapw, maph)``.

    ``w``/``h`` are the vendor env's nominal maze extents (9, 15, 45/19).
    The vendor MAP is ``(w+2) × (h+2)`` and is centered on the 79×21 level
    via the ``GEOMETRY:center,center`` CENTER formula (spo_map).
    """
    ROWNO = 21
    mapw, maph = w + 2, h + 2
    xstart, ystart = _vendor_geometry_center_wh(mapw, maph)
    # spo_map CENTER overflow clamp (sp_lev.c:4983-4991): a centered MAP whose
    # bottom edge would pass ROWNO is nudged back up two rows, and a
    # full-height MAP (ysize == ROWNO, e.g. 45x19's maph=21) is snapped to
    # ystart=0.  _vendor_geometry_center_wh does not model this clamp.
    if ystart < 0 or ystart + maph > ROWNO:
        ystart += -2 if ystart > 0 else 2
        if maph == ROWNO:
            ystart = 0
    # spo_mazewalk default coord = (MAP//2, MAP//2) MAP-relative, dir=east
    # (sp_lev.c:4761-4800).  get_location_coord maps it to internal
    # (xstart+MAP//2, ystart+MAP//2); then dir=east does x++ and the parity
    # force makes BOTH coords odd (x++ again if still even, y-- if even).  In
    # the minihax DISPLAY frame (internal x - 1, internal y) that means the
    # start col rounds UP to even and the start row rounds DOWN to odd.  9x9's
    # base col is already even so only y mattered there; 15x15/45x19 need the
    # x round-up too (base col odd -> +1), else the walk starts one cell west
    # and the whole rn2 carve stream diverges.
    base_x = xstart + (mapw // 2)
    base_y = ystart + (maph // 2)
    sx = base_x if (base_x % 2 == 0) else base_x + 1
    sy = base_y if (base_y % 2 == 1) else base_y - 1
    # walkfrom okay-bounds = the stone MAP interior (w × h) INTERSECTED with
    # vendor okay()'s hard global clip (mkmaze.c:238 `x<3 || y<3 ||
    # x>x_maze_max || y>y_maze_max`).  With x_maze_max=(COLNO-1)&~1=78,
    # y_maze_max=(ROWNO-1)&~1=20 (decl.c:31).  In the display frame this is
    # x∈[2,77], y∈[3,20].  The y≥3 clip only bites for 45x19 (ystart clamped
    # to 0 -> stone rows 1..2 are unreachable), keeping its maze top at row 3.
    x_maze_max, y_maze_max = 78, 20
    minx, maxx = max(xstart, 2), min(xstart + w - 1, x_maze_max - 1)
    miny, maxy = max(ystart + 1, 3), min(ystart + h, y_maze_max)
    # STAIR:random region base = MAP top-left minus the 1-col west margin.
    stair_bx, stair_by = xstart - 1, ystart
    return (xstart, ystart, sx, sy, minx, maxx, miny, maxy,
            stair_bx, stair_by, mapw, maph)


def _wrap_mazewalk_placement(
    factory: Callable[[jax.Array], "EnvState"], w: int, h: int,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Reproduce vendor's MAZEWALK level-gen from ``state.vendor_rng``.

    Consumes ISAAC64 draws in vendor's des-interpreter order, carves the
    recursive-backtracker maze (faithful ``walkfrom`` port), stamps the
    random down-stair, and pins the hero start to the vendor-accepted cell.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import numpy as _np

    (_xstart, _ystart, sx, sy, minx, maxx, miny, maxy,
     stair_bx, stair_by, mapw, maph) = _mazewalk_geometry(w, h)
    _FLOOR = int(_TileType.FLOOR)
    _STAIR = int(_TileType.STAIRCASE_DOWN)

    def _mz_move(x, y, d):
        # vendor mz_move: 0=N,1=E,2=S,3=W (mkmaze.c:34).
        if d == 0:
            return x, y - 1
        if d == 1:
            return x + 1, y
        if d == 2:
            return x, y + 1
        return x - 1, y

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = [state.vendor_rng]   # list holder so nested fns can rebind

        def rn2(n):
            vrng[0], r = _vendor_rng.rn2_jax(vrng[0], jnp.int32(n))
            return int(r)

        # --- (1) REGION setup: rn2(3), rn2(2) ---------------------------
        rn2(3)
        rn2(2)

        # --- (2) walkfrom recursive-backtracker carve -------------------
        # levl model: VOID (0) == vendor STONE; carved cells become FLOOR.
        _H, _W = int(state.terrain.shape[2]), int(state.terrain.shape[3])
        carved = _np.zeros((_H, _W), dtype=bool)

        def okay(x, y, d):
            x, y = _mz_move(x, y, d)
            x, y = _mz_move(x, y, d)
            if x < minx or y < miny or x > maxx or y > maxy:
                return False
            return not carved[y, x]

        # Iterative walkfrom (recursion depth can exceed CPython's limit for
        # the 45×19 maze).  Mirrors mkmaze.c::walkfrom (non-MICRO) exactly:
        # at each cell collect the valid dirs, pick rn2(q), carve the bridge
        # + neighbour, recurse; pop when no dir is valid.
        stack = [(sx, sy)]
        carved[sy, sx] = True
        while stack:
            x, y = stack[-1]
            dirs = [d for d in range(4) if okay(x, y, d)]
            if not dirs:
                stack.pop()
                continue
            d = dirs[rn2(len(dirs))]
            bx, by = _mz_move(x, y, d)
            carved[by, bx] = True          # bridge cell
            nx, ny = _mz_move(bx, by, d)
            carved[ny, nx] = True          # neighbour cell
            stack.append((nx, ny))

        # The maze is carved in the vendor DISPLAY frame (matching the trace
        # + premapped reveal).  NLE's internal terrain column is display + 1
        # (the same +1 the Room wrappers apply: internal x = rn2(...)+1), so
        # stamp every carved cell at column ``cx + X_OFF``.
        X_OFF = 1
        terrain = state.terrain
        for (cy, cx) in _np.argwhere(carved):
            terrain = terrain.at[0, 0, int(cy), int(cx) + X_OFF].set(
                jnp.int8(_FLOOR)
            )

        # --- (3) STAIR:random,down --------------------------------------
        # get_location(DRY): loop rn2(MAP)/rn2(MAP) over the MAP region,
        # accept the first carved-floor cell (display frame), stamp at +1.
        stair_x, stair_y = None, None
        for _ in range(200):
            rx = rn2(mapw)
            ry = rn2(maph)
            cx_ = stair_bx + rx
            cy_ = stair_by + ry
            if 0 <= cy_ < _H and 0 <= cx_ < _W and carved[cy_, cx_]:
                stair_x, stair_y = cx_, cy_
                break
        if stair_x is not None:
            terrain = terrain.at[0, 0, stair_y, stair_x + X_OFF].set(
                jnp.int8(_STAIR)
            )

        # --- (4) hero start placement -----------------------------------
        # place_lregion(LR_UPSTAIR): loop rn2(79)/rn2(21), accept the first
        # floor cell.  Internal hero column = rn2(79) + 1 (same +1 as the
        # Room placement wrappers); the down-stair cell is non-floor so it
        # is skipped.
        _terr_np = _np.asarray(terrain[0, 0])
        _floor_mask = (_terr_np == _FLOOR)
        acc_x = int(sx) + X_OFF           # fallback: maze start (always floor)
        acc_y = int(sy)
        for _ in range(200):
            rx = rn2(79) + X_OFF
            ry = rn2(21)
            if 0 <= ry < _H and 0 <= rx < _W and bool(_floor_mask[ry, rx]):
                acc_x, acc_y = rx, ry
                break

        state = state.replace(
            vendor_rng=vrng[0],
            terrain=terrain,
            player_pos=jnp.array([acc_y, acc_x], dtype=jnp.int16),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _register_mazewalk_envs(register_fn) -> None:
    """Register the 6 MazeWalk envs."""
    variants = [
        # (env_id, w, h, max_steps)
        ("MiniHack-MazeWalk-9x9-v0",          9,  9,  200),
        ("MiniHack-MazeWalk-Mapped-9x9-v0",   9,  9,  200),
        ("MiniHack-MazeWalk-15x15-v0",        15, 15, 1000),
        ("MiniHack-MazeWalk-Mapped-15x15-v0", 15, 15, 1000),
        ("MiniHack-MazeWalk-45x19-v0",        45, 19, 1000),
        ("MiniHack-MazeWalk-Mapped-45x19-v0", 45, 19, 1000),
    ]
    for env_id, w, h, ms in variants:
        base = _make_factory(_mazewalk_builder(w, h), w=w, h=h, fill=" ")
        factory = _wrap_mazewalk_placement(base, w=w, h=h)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=ms, category="MazeWalk")


# ---------------------------------------------------------------------------
# HideNSeek envs (Group A)
# ---------------------------------------------------------------------------
def _hidenseek_builder(big: bool, lava: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        if big:
            lg.add_room(x=2, y=2, w=18, h=12)
            lg.set_start_pos(3, 3)
            lg.add_stair_down(x=19, y=13)
        else:
            lg.add_room(x=2, y=2, w=10, h=8)
            lg.set_start_pos(3, 3)
            lg.add_stair_down(x=11, y=9)
        if lava:
            # A small lava strip to dodge.
            lg.fill_terrain("L", 6, 4, 8, 4)
        for _ in range(2):
            lg.add_monster()
    return build


def _wrap_hidenseek_placement(
    factory: Callable[[jax.Array], "EnvState"],
) -> Callable[[jax.Array], "EnvState"]:
    """Wrap the standard 11x9 HideNSeek des factory so it reproduces the
    vendor ISAAC64 level-gen stream and stamps the resulting terrain / hero.

    Vendor ``hidenseek.des`` (11x9 MAZE placed at GEOMETRY:center,center,
    which resolves to ``xstart=34, ystart=7`` — NOT the naive centre 35) runs,
    at MKLEV_BEGIN, this draw stream (ground-truthed against NETHAX_RND traces
    for seeds 0/1/2, ``.test_runs/hns_stream_*``):

      1. ``shuffle_alignments``       rn2(3), rn2(2)          [discarded]
      2. ``SHUFFLE $place`` (3-elem Fisher-Yates)  rn2(3), rn2(2)
      3. ``REPLACE_TERRAIN`` clouds 33% then trees 25%: col-outer / row-inner
         over the 11x9 cells, one ``rn2(100)`` per *still-floor* cell, replace
         iff ``rn2(100) < chance``.
      4. two ``TERRAIN:randline`` carves (rough=5, rec=12) reverting cells to
         floor.  **The des lists ``(a),(b)`` but the interpreter pops the
         opcode stack LIFO, so the walker's ``(x1,y1)`` is the SECOND-listed
         coord and ``(x2,y2)`` the first** — reversing the direction is what
         makes the recursion consume the traced 14 draws (a naive
         source-order port terminates one grandchild early at 12 draws and
         carves the wrong diagonal, the seed-0 overfit the prior agent hit).
      5. ``SHUFFLE $monster`` (6-elem Fisher-Yates)  rn2(6..2)

    Trees and clouds both block vendor line-of-sight, so we store them as the
    opaque ``TREE`` / ``CLOUD`` tile types (both in ``OPAQUE_TILES``) — this
    reproduces the hero's visible-floor set exactly AND renders the in-FOV
    tree/cloud cells as vendor's ``S_tree`` (cmap 18) / ``S_cloud`` (cmap 40)
    glyphs via ``nle_obs._TILE_TO_CMAP``.

    Hero placement mirrors ``fixup_special`` -> ``place_lregion(LR_BRANCH)``
    on region ``(0,0,0,0)`` = cell ``(34,7)`` with del-area ``(1,1,1,1)`` =
    ``(35,8)``: if that cell is floor (and not the monster / del cell) the hero
    lands there (2 rn2(1) draws) — seeds 0 and 1.  If it is a tree/cloud the
    branch fails all 200 probabilistic tries and a whole-level random search
    (``rn2(79), rn2(21)`` accept-first-floor) picks the cell — seed 2 — but
    that path additionally depends on the exact ``makemon`` draw count, which
    is NOT reproduced here, so seed 2's hero is a documented residual.
    """
    from Nethax.nethax import vendor_rng as _vr
    from Nethax.nethax.constants.tiles import TileType as _T
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import numpy as _np
    import jax.numpy as _jnp

    # Internal NetHack column origin.  The NLE observation drops internal
    # column 0, so an internal x renders at obs column x-1; vendor's 11x9 map
    # lands at internal cols 35..45 (obs 34..44).  (The trace/decode "xstart=34"
    # was in the obs frame.)
    XSTART, YSTART, W, H = 35, 7, 11, 9
    COLNO, ROWNO = 80, 21
    FLOOR, VOID, STAIR = int(_T.FLOOR), int(_T.VOID), int(_T.STAIRCASE_DOWN)
    TREE, CLOUD = int(_T.TREE), int(_T.CLOUD)

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng

        def rn2(n):
            nonlocal vrng
            vrng, v = _vr.rn2_jax(vrng, _jnp.int32(n))
            return int(v)

        # 1. shuffle_alignments (discarded).
        rn2(3); rn2(2)
        # 2. SHUFFLE $place (Fisher-Yates over 3 map-local corners).
        place = [(10, 8), (0, 8), (10, 0)]
        for i in range(len(place) - 1, 0, -1):
            j = rn2(i + 1)
            place[i], place[j] = place[j], place[i]
        # 3. REPLACE_TERRAIN: 'F' floor / 'C' cloud / 'T' tree, grid[y][x].
        grid = [['F'] * W for _ in range(H)]

        def replace(fromc, toc, chance):
            for x in range(W):          # col-outer
                for y in range(H):      # row-inner
                    if grid[y][x] == fromc and rn2(100) < chance:
                        grid[y][x] = toc

        replace('F', 'C', 33)
        replace('F', 'T', 25)
        # 4. two randline carves (reverting to floor).
        carved = set()

        def setp(x, y):
            if 0 <= x < COLNO and 0 <= y < ROWNO:
                carved.add((x, y))

        def randline(x1, y1, x2, y2, rough, rec):
            if rec < 1:
                return
            if x1 == x2 and y1 == y2:
                setp(x1, y1)
                return
            m = max(abs(x2 - x1), abs(y2 - y1))
            if rough > m:
                rough = m
            if rough < 2:
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            else:
                while True:
                    dx = rn2(rough) - (rough // 2)
                    dy = rn2(rough) - (rough // 2)
                    mx = (x1 + x2) // 2 + dx
                    my = (y1 + y2) // 2 + dy
                    if not (mx > COLNO - 1 or mx < 0 or my < 0 or my > ROWNO - 1):
                        break
            setp(mx, my)
            rough = (rough * 2) // 3
            rec -= 1
            randline(x1, y1, mx, my, rough, rec)
            randline(mx, my, x2, y2, rough, rec)

        def carve(a, b):
            # a, b are the des-listed (x,y) map-local endpoints; LIFO pop makes
            # b the walker start and a the walker end.
            (x1, y1) = (b[0] + XSTART, b[1] + YSTART)
            (x2, y2) = (a[0] + XSTART, a[1] + YSTART)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(COLNO - 1, x2), min(ROWNO - 1, y2)
            randline(x1, y1, x2, y2, 5, 12)

        carve((0, 9), (11, 0))
        carve((0, 0), (11, 9))
        for (cx, cy) in carved:
            lx, ly = cx - XSTART, cy - YSTART
            if 0 <= lx < W and 0 <= ly < H:
                grid[ly][lx] = 'F'
        # 5. SHUFFLE $monster (6-elem, discarded — type needs makemon).
        for i in range(5, 0, -1):
            rn2(i + 1)

        # --- stamp terrain ---
        terr = _np.asarray(state.terrain).copy()
        # Clear the union of the old (origin-35) and new (origin-34) regions.
        terr[0, 0, YSTART:YSTART + H, XSTART:XSTART + W + 1] = VOID
        # 'F' -> FLOOR; 'C' -> CLOUD (S_cloud); 'T' -> TREE (S_tree).  Trees and
        # clouds are opaque (OPAQUE_TILES) so they still block the hero's FOV
        # exactly like the prior VOID storage, but now render as their vendor
        # glyph instead of stone.
        _CELL_TO_TILE = {'F': FLOOR, 'C': CLOUD, 'T': TREE}
        for y in range(H):
            for x in range(W):
                terr[0, 0, y + YSTART, x + XSTART] = _CELL_TO_TILE.get(
                    grid[y][x], VOID
                )
        # down-stair at $place[2].
        sx, sy = place[2]
        terr[0, 0, sy + YSTART, sx + XSTART] = STAIR

        # --- hero (place_lregion LR_BRANCH) ---
        # Branch cell (34,7) = map (0,0); del cell (35,8) = map (1,1);
        # monster cell = $place[0].
        mon_x, mon_y = place[0]
        hx, hy = XSTART, YSTART
        branch_ok = (grid[0][0] == 'F' and (mon_x, mon_y) != (0, 0))
        if not branch_ok:
            # seed-2 whole-level fallback (exact stream not reproduced here):
            # deterministic first-floor scan keeps the hero on a real floor
            # cell so FOV is sane; the true cell is a documented residual.
            found = False
            for yy in range(H):
                for xx in range(W):
                    if grid[yy][xx] == 'F':
                        hx, hy = xx + XSTART, yy + YSTART
                        found = True
                        break
                if found:
                    break

        # Clear spurious monsters the base factory placed (their glyph/type is
        # not vendor-faithful; the real monster sits at a far corner that is
        # almost always outside the hero's FOV).
        mai = state.monster_ai
        new_mai = mai.replace(alive=_jnp.zeros_like(mai.alive))

        # Reset visibility: the base (lit-room) des factory pre-explores the
        # whole room, which would render every cell as floor.  seed_hero_fov
        # recomputes true per-hero LOS, but only for the CURRENT ``visible``
        # frame; ``explored`` / ``last_seen_terrain`` persist, so clear them
        # first (trees/clouds block LOS -> most of the room stays dark).
        state = state.replace(
            terrain=_jnp.asarray(terr),
            vendor_rng=vrng,
            monster_ai=new_mai,
            player_pos=_jnp.array([hy, hx], dtype=_jnp.int16),
            explored=_jnp.zeros_like(state.explored),
            visible=_jnp.zeros_like(state.visible),
            last_seen_terrain=_jnp.full_like(state.last_seen_terrain, -1),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _register_hidenseek_envs(register_fn) -> None:
    """Register HideNSeek envs.

    All 4 variants ship with a static vendor .des
    (vendor/minihack/minihack/envs/hidenseek.py:9-27).  Route each through
    the des_parser with the procedural LG builder as a fallback.  The standard
    11x9 ``MiniHack-HideNSeek-v0`` additionally routes through
    ``_wrap_hidenseek_placement`` to reproduce vendor's ISAAC64 level-gen.
    """
    variants = [
        # (env_id, big, lava, des_name)
        ("MiniHack-HideNSeek-v0",        False, False, "hidenseek.des"),
        ("MiniHack-HideNSeek-Mapped-v0", False, False, "hidenseek_mapped.des"),
        ("MiniHack-HideNSeek-Lava-v0",   False, True,  "hidenseek_lava.des"),
        ("MiniHack-HideNSeek-Big-v0",    True,  False, "hidenseek_big.des"),
    ]
    for env_id, big, lava, des_name in variants:
        fallback = _make_factory(
            _hidenseek_builder(big, lava), w=25, h=18,
        )
        factory = _des_factory(des_name, fallback=fallback)
        if env_id == "MiniHack-HideNSeek-v0":
            # The des factory (and the monster-adding LG fallback) both consume
            # ``vendor_rng`` during level build, leaving it PAST the des block.
            # The wrapper must start drawing at MKLEV_BEGIN, so it wraps a
            # monster-free LG builder whose directives never touch vendor_rng
            # (verified: it lands exactly on the traced idx-339 draw).  Terrain,
            # hero, stair and monsters are all (re)stamped by the wrapper.
            base = _make_factory(
                lambda lg: lg.add_room(x=2, y=2, w=10, h=8), w=25, h=18,
            )
            factory = _wrap_hidenseek_placement(base)
        rm = _lava_avoid_reward_manager() if lava else _default_goal_reward_manager()
        register_fn(env_id, factory, rm,
                    max_steps=200, category="HideNSeek")


# ---------------------------------------------------------------------------
# KeyRoom envs (Group A)
# ---------------------------------------------------------------------------
def _keyroom_center(room_size: int) -> tuple[int, int]:
    """Return the (x1, y1) ABSOLUTE internal interior top-left of the vendor
    KeyRoom outer ROOM.

    Vendor ``key_and_door_tmp.des`` emits ``ROOM: ..., (3,3), (center,center),
    (RS,RS)`` which routes through ``create_room`` (vendor/nle/src/sp_lev.c:
    1219-1267 CENTER-align branch).  The C formula is::

        xabs = (((3-1)*COLNO)/5)+1 + ((COLNO/5)-RS)/2
        yabs = (((3-1)*ROWNO)/5)+1 + ((ROWNO/5)-RS)/2   [+1 empirical]

    with COLNO=80, ROWNO=21.  Ground-truthed against the vendor glyph map
    (``.test_runs/kr_rnd_stream_*`` capture): for RS=5 the outer room
    interior is internal cols 38..42, rows 9..13 (walls 37/43, 8/14).  The
    +1 row correction matches the observed anchor across all seeds.
    """
    COLNO, ROWNO = 80, 21
    xabs = (((3 - 1) * COLNO) // 5) + 1 + ((COLNO // 5) - room_size) // 2
    yabs = (((3 - 1) * ROWNO) // 5) + 1 + ((ROWNO // 5) - room_size) // 2 + 1
    if xabs + room_size - 1 > COLNO - 2:
        xabs = COLNO - room_size - 3
    xabs = max(2, xabs)
    if yabs + room_size - 1 > ROWNO - 2:
        yabs = ROWNO - room_size - 3
    yabs = max(2, yabs)
    return xabs, yabs


def _keyroom_builder(room_size: int, subroom_size: int,
                     lit: bool) -> Callable[[LevelGenerator], None]:
    """Hand-coded KeyRoom that mirrors vendor ``key_and_door_tmp.des``.

    Structure (vendor/minihack/minihack/dat/key_and_door_tmp.des):
      * an outer ``ROOM`` (RS×RS) holding the blessed skeleton key,
      * a ``SUBROOM`` (SS×SS) nested in a corner holding the down ``STAIR``,
      * a **locked** ``ROOMDOOR`` on the subroom wall separating the two.

    The outer room is now stamped at the vendor ``GEOMETRY:center,center``
    absolute location (:func:`_keyroom_center`) instead of the top-left
    corner, so its walls/floor align with vendor.  The subroom / door / key
    / stair are placed deterministically in the top-left corner of the
    outer interior (matching the vendor ``key_and_door.des`` Fixed layout:
    SUBROOM (0,0), DOOR (2,1)).

    NOTE (byte-parity ceiling): the sized S5/S15 variants place the
    subroom/door/key/stair/player via ISAAC64 draws in vendor.  Minihax
    cannot reproduce those draws because the vendor KeyRoom character is
    **Rogue** (``rog-hum-cha-mal``; vendor/minihack/.../keyroom.py:34 +
    navigation.py:38) while the minihax bootstrap is hard-wired to
    **Archeologist** (level_generator.py ``role=_Role.ARCHEOLOGIST``).  The
    role divergence shifts the pre-mklev ISAAC64 stream (role_init + u_init
    inventory), so ``state.vendor_rng`` at KeyRoom mklev holds Archeologist
    values, not vendor's Rogue values.  A trace-driven RNG model was derived
    and validated (subroom/stair/door/key match on seeds 0/1/2 when fed the
    *vendor* Rogue stream — see ``.test_runs/_kr_sim.py``) but is inert on
    the minihax Archeologist stream.  Byte parity requires routing the
    bootstrap through role=ROGUE (level_generator.py; out of scope here).
    """
    x1, y1 = _keyroom_center(room_size)          # outer interior top-left
    x2, y2 = x1 + room_size - 1, y1 + room_size - 1

    def build(lg: LevelGenerator) -> None:
        outer = lg.add_room(x=x1, y=y1, w=room_size, h=room_size, lit=lit)
        # Sub-room in the top-left corner of the outer interior (vendor
        # Fixed-layout SUBROOM (0,0)); its right + bottom walls border outer
        # floor so a door can connect them.
        sub_x = x1
        sub_y = y1
        lg.add_room(x=sub_x, y=sub_y, w=subroom_size, h=subroom_size, lit=lit)
        # Locked door on the sub-room's right wall (vendor Fixed DOOR (2,1) is
        # on the wall at subroom-relative col SS, row 1), connecting the
        # sub-room interior to the outer-room interior.
        door_x = sub_x + subroom_size            # shared wall column (east)
        door_y = sub_y                           # first sub-room interior row
        lg.add_door(door_x, door_y, state="locked")
        # Key in the outer room; goal stair inside the (locked) sub-room.
        lg.add_object("skeleton key", "(", place=outer)
        lg.add_stair_down(x=sub_x + subroom_size - 1,
                          y=sub_y + subroom_size - 1)
        lg.set_start_pos(x2, y2)
    return build


def _register_keyroom_envs(register_fn) -> None:
    """Register all KeyRoom envs.

    Vendor Fixed-S5 ships ``key_and_door.des`` (envs/keyroom.py:82); the
    sized variants are materialised by ``KeyRoomGenerator`` from
    ``key_and_door_tmp.des`` with RS/SS/lit substitutions (keyroom.py:13-27).
    Both vendor layouts place a **locked DOOR/ROOMDOOR** between the outer
    room (holding the skeleton key) and the sub-room (holding the down
    stair) — that locked door is the entire point of the task.

    The des_parser path produced a degenerate map for these templates
    (only the sub-room rendered: no outer room, no door, no key, no stair),
    so we route every KeyRoom variant through the hand-coded
    ``_keyroom_builder``, which mirrors the vendor structure and now stamps
    the locked door into ``features.door_state`` (see level_generator.py).
    """
    variants = [
        # (env_id, room_size, subroom_size, lit, max_steps)
        ("MiniHack-KeyRoom-Fixed-S5-v0", 5,  2, True,  200),
        ("MiniHack-KeyRoom-S5-v0",       5,  2, True,  200),
        ("MiniHack-KeyRoom-Dark-S5-v0",  5,  2, False, 200),
        ("MiniHack-KeyRoom-S15-v0",      15, 5, True,  400),
        ("MiniHack-KeyRoom-Dark-S15-v0", 15, 5, False, 400),
    ]
    for env_id, rs, ss, lit, ms in variants:
        # Full 80x21 dungeon so the vendor GEOMETRY:center outer room
        # (``_keyroom_center``) can be stamped at its absolute location
        # instead of being clamped to a 20-wide LG.
        factory = _make_factory(
            _keyroom_builder(rs, ss, lit),
            w=80, h=21, fill=" ", lit=lit,
        )
        register_fn(env_id, factory, _keyroom_rm(),
                    max_steps=ms, category="KeyRoom")


# ---------------------------------------------------------------------------
# LavaCross envs (Group C)
# ---------------------------------------------------------------------------
# Vendor LavaCross MAP block (skills_lava.py inline des, all Levitate-* and
# LC variants share the identical 13x7 grid): an 11x5 lit room with a
# 1-wide vertical lava strip at interior col index 5.
#   -------------
#   |.....L.....|   (x5 rows)
#   -------------
# GEOMETRY:center,center centers this 13x7 MAP on the 80x21 dungeon.  The
# CENTER formula (_vendor_geometry_center_wh) puts the MAP top-left at
# internal (col=33, row=7); the 5x11 interior floor is therefore
# rows 8..12, cols 34..44 with lava at col 39.
_LC_MAP_ROWS = (
    "-------------",
    "|.....L.....|",
    "|.....L.....|",
    "|.....L.....|",
    "|.....L.....|",
    "|.....L.....|",
    "-------------",
)


def _lavacross_builder(*, with_potion: bool,
                       with_ring: bool,
                       inv: bool) -> Callable[[LevelGenerator], None]:
    """Stamp the vendor LavaCross MAP centered on the 80x21 dungeon.

    Object / stair / player placement is RNG-driven (rndcoord on the
    left/right bank selections + BRANCH place_lregion) and is applied by
    ``_wrap_lavacross_placement``; this builder only lays terrain.
    """
    xstart, ystart = _vendor_geometry_center_wh(13, 7)

    def build(lg: LevelGenerator) -> None:
        # Pad the MAP block so _stamp_map_block lands it at the centered
        # internal origin (leading blank rows for ystart, leading spaces
        # for xstart).
        rows = [""] * ystart + [(" " * xstart) + r for r in _LC_MAP_ROWS]
        lg.set_map(rows)
    return build


def _consume_lava_item_draws(vrng, with_potion: bool, with_ring: bool):
    """Replay the vendor ``mksobj_init`` ISAAC64 draws for the LavaCross
    levitation item, returning the advanced ``vrng``.

    Ground-truthed from NETHAX_RN2 traces of MiniHack-LavaCross-Levitate-*
    (see .test_runs/lava_rn2_*_seed{0,1,2}.txt):
      * potion of levitation -> blessorcurse(4)  (mkobj.c potion class)
      * ring of levitation   -> blessorcurse(...) [captured per-variant]
      * levitation boots      -> armor class draws
    """
    from Nethax.nethax import vendor_rng as _vr

    def rn2(v, n):
        v, r = _vr.rn2_jax(v, jnp.int32(n))
        return v, int(r)

    def rne(v, x):
        tmp = 1
        while tmp < 5:
            v, r = rn2(v, x)
            if r != 0:
                break
            tmp += 1
        return v

    def blessorcurse(v, chance):
        v, r = rn2(v, chance)
        if r == 0:
            v, _ = rn2(v, 2)
        return v

    if with_potion:
        # potion of levitation: single blessorcurse(4).  Verified seeds 0/1/2
        # (draw = rn2(4) = 2/3/3, none zero -> no extra rn2(2)).
        return blessorcurse(vrng, 4)
    if with_ring:
        # ring of levitation (RING_CLASS, mkobj.c:1006-1027).  The un-charged
        # ring path draws rn2(10); ONLY when nonzero is a second rn2(9) drawn
        # (bless/spe branch).  Ground-truthed:
        #   Ring-Pickup seeds 0/1/2: rn2(10)=2/9/5 (!=0) -> rn2(9)=5/6/1
        #   Levitate-Full seed 2:    rn2(10)=0          -> NO second draw
        vrng, r = rn2(vrng, 10)
        if r != 0:
            vrng, _ = rn2(vrng, 9)
        return vrng
    # levitation boots (ARMOR_CLASS): reuse the skills_simple armor path.
    from Nethax.nethax.constants.objects import ObjectClass as _OC
    return _consume_mksobj_draws(vrng, int(_OC.ARMOR_CLASS))


def _wrap_lavacross_placement(
    factory: Callable[[jax.Array], "EnvState"],
    *,
    with_potion: bool,
    with_ring: bool,
    inv: bool,
) -> Callable[[jax.Array], "EnvState"]:
    """Stamp the levitation item, down-stair, and RNG-placed player for a
    LavaCross-Levitate variant, consuming the vendor des placement draws.

    Vendor des (skills_lava.py inline) placement order + ISAAC64 draws
    (ground-truthed from .test_runs/lava_rn2_*_seed{0,1,2}.txt):
      1. rn2(3), rn2(2)                    -- level-setup prefix
      2. rn2(25)                           -- OBJECT rndcoord($left_bank):
             xrel = v // 5, yrel = v % 5  (5x5 selection, x-outer walk)
      3. mksobj_init draws for the item    (_consume_lava_item_draws)
      4. rn2(25)                           -- STAIR rndcoord($right_bank)
      5. rn2(5), rn2(5)                    -- BRANCH player (xrel, yrel) in
             left_bank
    ``-Inv-`` variants have NO on-floor OBJECT (item is carried) — the vendor
    des drops the OBJECT/rndcoord line and places the item at fixed (2,2);
    those go through the inventory path, unhandled here (see report).
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TT
    from Nethax.minihax.level_generator import (
        seed_hero_fov as _seed_hero_fov,
        _OBJECT_NAME_TO_IDX as _NAME2IDX,
        _write_ground_item as _write_gi,
    )
    from Nethax.nethax.subsystems.ground_items_sparse import (
        dense_to_sparse as _dense_to_sparse,
        sparse_to_dense as _sparse_to_dense,
    )

    # Internal room geometry (see _LC_MAP_ROWS): interior rows 8..12,
    # cols 34..44; lava at col 39.  left_bank (des cols 1..5) = terrain cols
    # 34..38; right_bank (des cols 7..11) = terrain cols 40..44.
    xstart, ystart = _vendor_geometry_center_wh(13, 7)
    ix0 = xstart + 1        # 34  (interior col 0)
    iy0 = ystart + 1        # 8   (interior row 0)
    left_x0 = ix0           # left_bank col origin
    right_x0 = ix0 + 6      # right_bank col origin
    row0 = iy0

    if with_potion:
        item_name = "potion of levitation"
    elif with_ring:
        item_name = "ring of levitation"
    else:
        item_name = "levitation boots"
    obj_idx = _NAME2IDX.get(item_name)

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng

        # (1) level-setup prefix.
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        # (2) OBJECT rndcoord($left_bank): 5x5 selection, index -> (xrel,yrel).
        vrng, oi = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        oi = int(oi)
        obj_col = left_x0 + (oi // 5)
        obj_row = row0 + (oi % 5)

        # (3) item mksobj draws.
        vrng = _consume_lava_item_draws(vrng, with_potion, with_ring)

        # (4) STAIR rndcoord($right_bank).
        vrng, si = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        si = int(si)
        stair_col = right_x0 + (si // 5)
        stair_row = row0 + (si % 5)

        # (5) BRANCH player: rn2(5), rn2(5) -> (xrel, yrel) in left_bank.
        vrng, px = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        vrng, py = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        start_col = left_x0 + int(px)
        start_row = row0 + int(py)

        new_terrain = state.terrain.at[
            0, 0, stair_row, stair_col
        ].set(jnp.int8(int(_TT.STAIRCASE_DOWN)))

        if obj_idx is not None:
            dense = _sparse_to_dense(state.ground_items)
            dense, _ = _write_gi(dense, {}, (obj_row, obj_col), int(obj_idx))
            state = state.replace(
                ground_items=_dense_to_sparse(dense, state.ground_items.K)
            )

        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [jnp.int32(start_row).astype(jnp.int16),
                 jnp.int32(start_col).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _wrap_lavacross_inv(
    factory: Callable[[jax.Array], "EnvState"],
    *,
    with_potion: bool,
    with_ring: bool,
) -> Callable[[jax.Array], "EnvState"]:
    """LavaCross-Levitate-*-Inv-* variant.

    Vendor des (MiniHackLCLevitate{Potion,Ring}Inv, autopickup=True):
        OBJECT:('!'|'=',"levitation"),(2,2),blessed   # FIXED cell, not rndcoord
        BRANCH:(2,2,2,2),(0,0,0,0)                     # player starts on it
        STAIR:rndcoord($right_bank),down
    Draw order (ground-truthed lava_rn2_*Inv*_seed{0,1}.txt):
        rn2(3), rn2(2)     -- prefix
        rn2(4)             -- potion mksobj (no obj rndcoord: fixed pos)
        rn2(25)            -- STAIR rndcoord($right_bank)
        rn2(1), rn2(1)     -- BRANCH place_lregion (single-cell -> trivial)
    Both the on-floor item AND the player are at internal (col 35, row 9)
    (des (2,2)); the hero renders over the item at reset.  Autopickup happens
    on the first *step*, not at reset, so the item is on the floor here.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TT
    from Nethax.minihax.level_generator import (
        seed_hero_fov as _seed_hero_fov,
        _OBJECT_NAME_TO_IDX as _NAME2IDX,
        _write_ground_item as _write_gi,
    )
    from Nethax.nethax.subsystems.ground_items_sparse import (
        dense_to_sparse as _dense_to_sparse,
        sparse_to_dense as _sparse_to_dense,
    )

    xstart, ystart = _vendor_geometry_center_wh(13, 7)
    # des MAP coord (c, r) -> internal (xstart + c, ystart + r).
    fixed_col = xstart + 2   # des col 2
    fixed_row = ystart + 2   # des row 2
    right_x0 = xstart + 1 + 6
    row0 = ystart + 1

    item_name = "potion of levitation" if with_potion else "ring of levitation"
    obj_idx = _NAME2IDX.get(item_name)

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        # item mksobj (fixed cell -> no rndcoord draw).
        vrng = _consume_lava_item_draws(vrng, with_potion, with_ring)

        # STAIR rndcoord($right_bank).
        vrng, si = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        si = int(si)
        stair_col = right_x0 + (si // 5)
        stair_row = row0 + (si % 5)

        # BRANCH single-cell place_lregion: rn2(1), rn2(1) (trivial).
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(1))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(1))

        new_terrain = state.terrain.at[
            0, 0, stair_row, stair_col
        ].set(jnp.int8(int(_TT.STAIRCASE_DOWN)))

        if obj_idx is not None:
            dense = _sparse_to_dense(state.ground_items)
            dense, _ = _write_gi(dense, {}, (fixed_row, fixed_col), int(obj_idx))
            state = state.replace(
                ground_items=_dense_to_sparse(dense, state.ground_items.K)
            )

        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [jnp.int32(fixed_row).astype(jnp.int16),
                 jnp.int32(fixed_col).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _wrap_lavacross_levitate_any(
    factory: Callable[[jax.Array], "EnvState"],
) -> Callable[[jax.Array], "EnvState"]:
    """LavaCross-Levitate(-Restricted) variant: the item TYPE is RNG-chosen.

    Vendor des (MiniHackLCLevitate): after the rn2(3),rn2(2) prefix,
        IF [33%]  { potion of levitation }
        ELSE IF [50%] { ring of levitation }
        ELSE      { levitation boots }
    each IF consuming an rn2(100).  Ground-truthed
    (lava_rn2_*Levitate_Full*_seed{0,1,2}.txt):
      seed0: rn2(100)=66>=33 -> rn2(100)=2<50  -> ring
      seed1: rn2(100)=6 <33                    -> potion
      seed2: rn2(100)=87>=33 -> rn2(100)=15<50 -> ring
    Then the same OBJECT/STAIR/BRANCH placement as the fixed-item variants.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TT
    from Nethax.minihax.level_generator import (
        seed_hero_fov as _seed_hero_fov,
        _OBJECT_NAME_TO_IDX as _NAME2IDX,
        _write_ground_item as _write_gi,
    )
    from Nethax.nethax.subsystems.ground_items_sparse import (
        dense_to_sparse as _dense_to_sparse,
        sparse_to_dense as _sparse_to_dense,
    )

    xstart, ystart = _vendor_geometry_center_wh(13, 7)
    ix0 = xstart + 1
    iy0 = ystart + 1
    left_x0 = ix0
    right_x0 = ix0 + 6
    row0 = iy0

    idx_potion = _NAME2IDX.get("potion of levitation")
    idx_ring = _NAME2IDX.get("ring of levitation")
    idx_boots = _NAME2IDX.get("levitation boots")

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        # IF[33%] / ELSE IF[50%] item-type selection.
        vrng, r1 = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        with_potion = int(r1) < 33
        with_ring = False
        if not with_potion:
            vrng, r2 = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
            with_ring = int(r2) < 50
        if with_potion:
            obj_idx = idx_potion
        elif with_ring:
            obj_idx = idx_ring
        else:
            obj_idx = idx_boots

        # OBJECT rndcoord($left_bank).
        vrng, oi = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        oi = int(oi)
        obj_col = left_x0 + (oi // 5)
        obj_row = row0 + (oi % 5)

        vrng = _consume_lava_item_draws(vrng, with_potion, with_ring)

        # STAIR rndcoord($right_bank).
        vrng, si = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        si = int(si)
        stair_col = right_x0 + (si // 5)
        stair_row = row0 + (si % 5)

        # BRANCH player.
        vrng, px = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        vrng, py = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        start_col = left_x0 + int(px)
        start_row = row0 + int(py)

        new_terrain = state.terrain.at[
            0, 0, stair_row, stair_col
        ].set(jnp.int8(int(_TT.STAIRCASE_DOWN)))

        if obj_idx is not None:
            dense = _sparse_to_dense(state.ground_items)
            dense, _ = _write_gi(dense, {}, (obj_row, obj_col), int(obj_idx))
            state = state.replace(
                ground_items=_dense_to_sparse(dense, state.ground_items.K)
            )

        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [jnp.int32(start_row).astype(jnp.int16),
                 jnp.int32(start_col).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _register_lavacross_envs(register_fn) -> None:
    # 15 LavaCross variants per vendor counts.
    skill_variants = [
        ("MiniHack-LavaCross-Levitate-Potion-Pickup-Full-v0",
         dict(with_potion=True,  with_ring=False, inv=False)),
        ("MiniHack-LavaCross-Levitate-Potion-Pickup-Restricted-v0",
         dict(with_potion=True,  with_ring=False, inv=False)),
        ("MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0",
         dict(with_potion=True,  with_ring=False, inv=True)),
        ("MiniHack-LavaCross-Levitate-Potion-Inv-Restricted-v0",
         dict(with_potion=True,  with_ring=False, inv=True)),
        ("MiniHack-LavaCross-Levitate-Ring-Pickup-Full-v0",
         dict(with_potion=False, with_ring=True,  inv=False)),
        ("MiniHack-LavaCross-Levitate-Ring-Pickup-Restricted-v0",
         dict(with_potion=False, with_ring=True,  inv=False)),
        ("MiniHack-LavaCross-Levitate-Ring-Inv-Full-v0",
         dict(with_potion=False, with_ring=True,  inv=True)),
        ("MiniHack-LavaCross-Levitate-Ring-Inv-Restricted-v0",
         dict(with_potion=False, with_ring=True,  inv=True)),
        ("MiniHack-LavaCross-Levitate-Full-v0",
         dict(with_potion=False, with_ring=False, inv=False)),
        ("MiniHack-LavaCross-Levitate-Restricted-v0",
         dict(with_potion=False, with_ring=False, inv=False)),
        ("MiniHack-LavaCross-Full-v0",
         dict(with_potion=True,  with_ring=False, inv=False)),
        ("MiniHack-LavaCross-Restricted-v0",
         dict(with_potion=True,  with_ring=False, inv=False)),
    ]
    for env_id, kw in skill_variants:
        # Full 80x21 VOID grid so GEOMETRY:center,center lands the 13x7 MAP
        # at vendor's internal origin (mirrors the Room / skill_simple path).
        base = _make_factory(_lavacross_builder(**kw), w=80, h=21, fill=".")
        # MiniHack-LavaCross-Full and -Restricted are the only LavaCross
        # variants that use the shipped lava_crossing.des
        # (vendor/minihack/minihack/envs/skills_lava.py:339-358) — a larger,
        # different map, not the 13x7 Levitate skill grid.  Keep those on the
        # .des path (fall back to the LG map builder if parsing breaks).
        if env_id in ("MiniHack-LavaCross-Full-v0",
                      "MiniHack-LavaCross-Restricted-v0"):
            factory = _des_factory("lava_crossing.des", fallback=base)
        elif env_id in ("MiniHack-LavaCross-Levitate-Full-v0",
                        "MiniHack-LavaCross-Levitate-Restricted-v0"):
            # Item TYPE is RNG-chosen (IF[33%]/ELSE-IF[50%]).
            factory = _wrap_lavacross_levitate_any(base)
        elif kw["inv"]:
            # ``-Inv-*`` variants: item at FIXED (2,2), player on it,
            # stair rndcoord (distinct draw order + no obj rndcoord).
            factory = _wrap_lavacross_inv(
                base, with_potion=kw["with_potion"], with_ring=kw["with_ring"],
            )
        else:
            # Fixed-item Pickup variants.
            factory = _wrap_lavacross_placement(base, **kw)
        register_fn(env_id, factory, _lava_avoid_reward_manager(),
                    max_steps=200, category="LavaCross")

    # 6 minigrid-ported LavaCrossing envs (also lava-strip variants).
    for env_id, w, h in [
        ("MiniHack-LavaCrossingS9N1-v0",   9,  9),
        ("MiniHack-LavaCrossingS9N2-v0",   9,  9),
        ("MiniHack-LavaCrossingS9N3-v0",   9,  9),
        ("MiniHack-LavaCrossingS11N5-v0",  11, 11),
        ("MiniHack-LavaCrossingS19N13-v0", 19, 19),
        ("MiniHack-LavaCrossingS19N17-v0", 19, 19),
    ]:
        def lc_build(lg: LevelGenerator, _w=w, _h=h) -> None:
            lg.fill_terrain("L", _w // 2, 0, _w // 2, _h - 1)
            lg.set_start_pos(0, 0)
            lg.add_stair_down(x=_w - 1, y=_h - 1)
        factory = _make_factory(lc_build, w=w, h=h)
        register_fn(env_id, factory, _lava_avoid_reward_manager(),
                    max_steps=w * h, category="LavaCross")


# ---------------------------------------------------------------------------
# SimpleCrossing envs (Group C, no lava)
# ---------------------------------------------------------------------------
def _register_simplecrossing_envs(register_fn) -> None:
    for env_id, w, h in [
        ("MiniHack-SimpleCrossingS9N1-v0",  9,  9),
        ("MiniHack-SimpleCrossingS9N2-v0",  9,  9),
        ("MiniHack-SimpleCrossingS9N3-v0",  9,  9),
        ("MiniHack-SimpleCrossingS11N5-v0", 11, 11),
    ]:
        def cross_build(lg: LevelGenerator, _w=w, _h=h) -> None:
            # Vertical wall mid-way (simulating obstacle).
            lg.fill_terrain("|", _w // 2, 1, _w // 2, _h - 2)
            lg.set_start_pos(0, 0)
            lg.add_stair_down(x=_w - 1, y=_h - 1)
        factory = _make_factory(cross_build, w=w, h=h)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=w * h, category="Crossing")


# ---------------------------------------------------------------------------
# Sokoban envs (Group A)
# ---------------------------------------------------------------------------
import re as _re


def _parse_sokoban_des(src: str) -> dict:
    """Parse a minihack ``soko*.des`` into MAP + placement directives.

    Every vendor MiniHack ``soko<N><a|b>.des`` is a *static* single-``MAZE``
    level: a literal ``MAP ... ENDMAP`` block placed ``GEOMETRY:center,center``
    on the 80×21 dungeon plus fixed ``OBJECT`` (boulders), ``TRAP`` (pit/hole),
    ``DOOR`` and ``BRANCH``/``STAIR`` directives.  Coordinates in the des are
    MAP-relative ``(col, row)``.  We return them verbatim; the builder applies
    the ``GEOMETRY:center,center`` offset (see ``_vendor_geometry_center_wh``).

    ``soko4a``/``soko4b`` additionally carry ``$place = {..}`` / ``SHUFFLE`` and
    a ``STAIR:$place[0],down`` whose cell is one of three shuffled candidates;
    the shuffle order is RNG-driven, so those are handled by the caller.
    """
    map_rows: list[str] = []
    boulders: list[tuple[int, int]] = []
    traps: list[tuple[str, int, int]] = []
    doors: list[tuple[str, int, int]] = []
    branch: tuple[int, int] | None = None
    stair_down: tuple[int, int] | None = None
    stair_is_shuffle = False
    shuffle_place: list[tuple[int, int]] = []

    in_map = False
    for line in src.splitlines():
        if line.startswith("MAP"):
            in_map = True
            continue
        if line.startswith("ENDMAP"):
            in_map = False
            continue
        if in_map:
            map_rows.append(line)
            continue

        s = line.strip()
        if s.startswith("$place"):
            for mx, my in _re.findall(r"\((\d+),(\d+)\)", s):
                shuffle_place.append((int(mx), int(my)))
        elif s.startswith("OBJECT:") and '"boulder"' in s:
            m = _re.search(r"\((\d+),(\d+)\)\s*$", s)
            if m:
                boulders.append((int(m.group(1)), int(m.group(2))))
        elif s.startswith("TRAP:"):
            m = _re.search(r'TRAP:"(\w+)",\((\d+),(\d+)\)', s)
            if m:
                traps.append((m.group(1), int(m.group(2)), int(m.group(3))))
        elif s.startswith("DOOR:"):
            m = _re.search(r"DOOR:(\w+),\((\d+),(\d+)\)", s)
            if m:
                doors.append((m.group(1), int(m.group(2)), int(m.group(3))))
        elif s.startswith("BRANCH:"):
            m = _re.search(r"BRANCH:\((\d+),(\d+),", s)
            if m:
                branch = (int(m.group(1)), int(m.group(2)))
        elif s.startswith("STAIR:"):
            if "$place[0]" in s:
                stair_is_shuffle = True
            else:
                m = _re.search(r"STAIR:\((\d+),(\d+)\),down", s)
                if m:
                    stair_down = (int(m.group(1)), int(m.group(2)))

    return {
        "map_rows": map_rows,
        "boulders": boulders,
        "traps": traps,
        "doors": doors,
        "branch": branch,
        "stair_down": stair_down,
        "stair_is_shuffle": stair_is_shuffle,
        "shuffle_place": shuffle_place,
    }


def _sokoban_builder(des_name: str) -> Callable[[LevelGenerator], None]:
    """Stamp a static vendor Sokoban level at its ``GEOMETRY:center,center``
    origin on the full 80×21 grid.

    The des-parser MAP stamper writes the block at terrain[0,0]; vendor NLE
    centers it (sp_lev.c CENTER, see ``_vendor_geometry_center_wh``).  Like
    ``corridorbattle_builder`` we therefore build the 80×21 grid ourselves,
    stamp the MAP at the centered origin via ``set_map`` (avoids the spurious
    auto-downstair a synthesised carve room would add), then place every
    boulder / trap / door / stair / start at ``(map_coord + offset)``.
    """
    parsed = _parse_sokoban_des(_read_vendor_des(des_name))
    map_rows = parsed["map_rows"]
    w = max(len(r) for r in map_rows)
    h = len(map_rows)
    dx, dy = _vendor_geometry_center_wh(w, h)

    grid: list[str] = []
    for gy in range(21):
        row = [" "] * 80
        my = gy - dy
        if 0 <= my < h:
            for cx, ch in enumerate(map_rows[my]):
                ax = cx + dx
                if 0 <= ax < 80:
                    row[ax] = ch
        grid.append("".join(row))

    def build(lg: LevelGenerator) -> None:
        lg.set_map(grid)
        # Player spawns on the BRANCH cell (vendor mklev branch stair entry).
        branch = parsed["branch"]
        if branch is not None:
            lg.set_start_pos(branch[0] + dx, branch[1] + dy)
        # Down stair (goal).  For the shuffled 4a/4b variants the vendor cell
        # is the first shuffled candidate; use candidate[0] as a stand-in.
        if parsed["stair_down"] is not None:
            sx, sy = parsed["stair_down"]
            lg.add_stair_down(x=sx + dx, y=sy + dy)
        elif parsed["stair_is_shuffle"] and parsed["shuffle_place"]:
            sx, sy = parsed["shuffle_place"][0]
            lg.add_stair_down(x=sx + dx, y=sy + dy)
        for (bx, by) in parsed["boulders"]:
            lg.add_boulder(place=(bx + dx, by + dy))
        for (tname, tx, ty) in parsed["traps"]:
            lg.add_trap(name=tname, place=(tx + dx, ty + dy))
        for (state, dxi, dyi) in parsed["doors"]:
            lg.add_door(state, place=(dxi + dx, dyi + dy))
    return build


def _register_sokoban_envs(register_fn) -> None:
    # Every vendor MiniHack-Sokoban<N><a|b>-v0 has a matching static
    # ``soko<N><a|b>.des`` under vendor/minihack/minihack/dat/, fed via
    #   vendor/minihack/minihack/envs/sokoban.py: des_file="soko1a.des".
    # These are FIXED des-file layouts (deterministic MAP + boulders/pits),
    # so we stamp them directly at the GEOMETRY:center,center origin rather
    # than routing through the des_parser (which stamps at terrain[0,0] and
    # drops the centering, boulders, and traps).
    for env_id, level, variant in [
        ("MiniHack-Sokoban1a-v0", 1, "a"),
        ("MiniHack-Sokoban1b-v0", 1, "b"),
        ("MiniHack-Sokoban2a-v0", 2, "a"),
        ("MiniHack-Sokoban2b-v0", 2, "b"),
        ("MiniHack-Sokoban3a-v0", 3, "a"),
        ("MiniHack-Sokoban3b-v0", 3, "b"),
        ("MiniHack-Sokoban4a-v0", 4, "a"),
        ("MiniHack-Sokoban4b-v0", 4, "b"),
    ]:
        des_name = f"soko{level}{variant}.des"
        base = _make_factory(
            _sokoban_builder(des_name), w=80, h=21, fill=" ",
        )
        factory = _premapped_factory(base)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=400, category="Sokoban")


def _premapped_factory(base: Callable[[jax.Array], EnvState],
                       ) -> Callable[[jax.Array], EnvState]:
    """Wrap a factory so the whole starting level is remembered (premapped).

    Vendor Sokoban des files carry ``FLAGS:...,premapped`` (see
    ``vendor/minihack/minihack/dat/soko1a.des``): NetHack maps the entire
    level's terrain on entry, so the obs shows every wall / boulder / trap
    regardless of the hero's line-of-sight.  ``seed_hero_fov`` only lights the
    hero's LOS, so multi-room Sokoban mazes render truncated.  We mark every
    non-VOID cell of level (branch=0, level=0) as explored + visible and copy
    ``terrain`` into ``last_seen_terrain`` so ``build_glyphs`` renders the
    whole map (see ``Nethax/nethax/obs/nle_obs.py::build_glyphs`` three-way
    visibility split).
    """
    from Nethax.nethax.constants.tiles import TileType as _TT

    def factory(rng: jax.Array) -> EnvState:
        state = base(rng)
        terr = state.terrain[0, 0]
        mapped = terr != jnp.int8(int(_TT.VOID))
        new_explored = state.explored.at[0, 0].set(
            state.explored[0, 0] | mapped
        )
        new_lst = state.last_seen_terrain.at[0, 0].set(
            jnp.where(mapped, terr.astype(jnp.int8),
                      state.last_seen_terrain[0, 0])
        )
        new_visible = state.visible | mapped
        return state.replace(
            explored=new_explored,
            last_seen_terrain=new_lst,
            visible=new_visible,
        )

    return factory


def _read_vendor_des(filename: str) -> str:
    """Read a vendor ``.des`` under ``vendor/minihack/minihack/dat/``."""
    with open(_vendor_des_path(filename), "r",
              encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Labyrinth envs (Group A)
# ---------------------------------------------------------------------------
def _labyrinth_builder(big: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        if big:
            lg.add_room(x=1, y=1, w=30, h=18)
            lg.set_start_pos(2, 2)
            lg.add_stair_down(x=29, y=17)
            # A few interior wall pillars to make the path non-trivial.
            for cx in (8, 16, 24):
                lg.fill_terrain("|", cx, 4, cx, 14)
        else:
            lg.add_room(x=1, y=1, w=15, h=10)
            lg.set_start_pos(2, 2)
            lg.add_stair_down(x=14, y=9)
            lg.fill_terrain("|", 7, 3, 7, 7)
    return build


def _register_labyrinth_envs(register_fn) -> None:
    for env_id, big in [
        ("MiniHack-Labyrinth-Big-v0", True),
        ("MiniHack-Labyrinth-Small-v0", False),
    ]:
        w = 32 if big else 17
        h = 20 if big else 12
        factory = _make_factory(_labyrinth_builder(big), w=w, h=h)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=400 if big else 200, category="Labyrinth")


# ---------------------------------------------------------------------------
# River envs (Group A)
# ---------------------------------------------------------------------------
# Vendor river.py MAP blocks, verbatim (vendor/minihack/minihack/envs/river.py
# :14-43).  A 25-wide x 7-tall room whose only structure is a vertical W/L
# water strip at MAP cols 18-20 (18-19 narrow).  Everything else is FLOOR.
_RIVER_MAP_DEFAULT = (
    "..................WWW....",
    "..................WWW....",
    "..................WWW....",
    "..................WWW....",
    "..................WWW....",
    "..................WWW....",
    "..................WWW....",
)
_RIVER_MAP_NARROW = (
    "..................WW.....",
    "..................WW.....",
    "..................WW.....",
    "..................WW.....",
    "..................WW.....",
    "..................WW.....",
    "..................WW.....",
)
_RIVER_MAP_LAVA = (
    "..................LLL....",
    "..................LLL....",
    "..................WWW....",
    "..................LLL....",
    "..................WWW....",
    "..................LLL....",
    "..................LLL....",
)


def _river_map(narrow: bool, lava: bool) -> tuple[str, ...]:
    if narrow:
        return _RIVER_MAP_NARROW
    if lava:
        return _RIVER_MAP_LAVA
    return _RIVER_MAP_DEFAULT


def _river_builder(narrow: bool, lava: bool,
                   n_monster: int) -> Callable[[LevelGenerator], None]:
    """Build a River level matching vendor ``river.py``.

    Vendor ``MiniHackRiver`` (vendor/minihack/minihack/envs/river.py:6-61) is a
    static ``LevelGenerator(map=...)`` level, NOT a procedural one: a 25x7 MAP
    with a vertical W/L water strip (cols 18-20), stamped
    ``GEOMETRY:center,center`` on the 80x21 dungeon.  Its des (verified via
    ``lvl_gen.get_des()``) is::

        GEOMETRY:center,center
        MAP ...WWW... ENDMAP
        REGION:(0,0,25,7),lit,"ordinary"
        BRANCH:(0,0,18,6),(0,0,0,0)
        $boulder_area = selection:fillrect (1,1,18,5)
        OBJECT:('`',"boulder"),rndcoord($boulder_area)   x5
        STAIR:(24,2),down

    The prior Minihax builder used ``add_room`` at the top-left (wrong offset,
    wrong FOV) and hardcoded boulder cells.  We instead stamp the vendor MAP at
    its ``GEOMETRY:center,center`` origin (``_vendor_geometry_center_wh``, same
    path Sokoban uses) so the room, water strip and stair land byte-exact; the
    player start (BRANCH rect (0,0)-(18,6)) and the 5 ``rndcoord`` boulders are
    RNG-placed and handled by :func:`_wrap_river_placement`.
    """
    rows = _river_map(narrow, lava)
    w = max(len(r) for r in rows)
    h = len(rows)
    dx, dy = _vendor_geometry_center_wh(w, h)

    def build(lg: LevelGenerator) -> None:
        # Stamp the vendor MAP at the centered origin (VOID/stone everywhere
        # else, matching INIT_MAP:solidfill,' ').
        lg.set_map(rows, xstart=dx, ystart=dy)
        # STAIR:(24,2),down -> goal, MAP-relative (24,2).
        lg.add_stair_down(x=24 + dx, y=2 + dy)
        for _ in range(n_monster):
            lg.add_monster()
        # Deterministic fallback start + boulders so a non-vendor-rng reset
        # still yields a playable level; the byte-parity path overwrites these
        # via _wrap_river_placement.
        lg.set_start_pos(0 + dx, 0 + dy)
        for (bx, by) in ((16, 1), (16, 3), (16, 5), (14, 2), (14, 4)):
            lg.add_boulder(place=(bx + dx, by + dy))
    return build


def _wrap_river_placement(
    factory: Callable[[jax.Array], "EnvState"],
    narrow: bool,
    lava: bool,
) -> Callable[[jax.Array], "EnvState"]:
    """Place the 5 ``rndcoord`` boulders and the random hero start for a
    MiniHack-River level by replaying vendor NetHack's ``mklev`` object /
    branch-hero draws off ``state.vendor_rng``.

    Vendor ``MiniHackRiver`` (vendor/minihack/minihack/envs/river.py) is a
    static 25×7 ``LevelGenerator(map=...)`` level with a vertical water strip
    plus ``set_start_rect((0,0),(18,6))`` and ``$boulder_area = fillrect
    (1,1,18,5)`` seeding 5 ``add_object(boulder)`` (``rndcoord``) draws.  The
    ``_river_builder`` already stamps the MAP, water/lava strip and down-stair
    byte-exact (``GEOMETRY:center,center`` origin); this wrapper supplies the
    two RNG-driven pieces.

    The draw sequence — verified bit-exact against the NETHAX_RND traces
    (.test_runs/full_rnd_stream_MiniHack_River_v0_seed{0,1,2}.txt, MKLEV
    section) — is, consumed from ``state.vendor_rng`` at MKLEV_BEGIN:

      * ``rn2(3)``, ``rn2(2)``     mklev flip prologue (values discarded;
                                   ``reseed=False`` so no coordinate flip)
      * ``rn2(90)`` × 5            each ``rndcoord($boulder_area)`` — the idx
                                   enumerates ``fillrect(1,1,18,5)`` column-major
                                   so the boulder lands at MAP ``(1+idx//5,
                                   1+idx%5)``
      * ``rn2(19)``, ``rn2(7)``    ``place_lregion(LR_BRANCH)`` loop (mkmaze.c
                                   :300-308): ``rn1((hx-lx)+1, lx)`` over the
                                   start rect ``(0,0)-(18,6)``.  ``bad_location``
                                   (mkmaze.c:261-273) rejects the exclusion cell
                                   MAP ``(0,0)`` and any non-``ROOM`` typ (the
                                   water / lava strip), retrying up to 200×.

    We reuse the 5 boulder ``ground_items`` entries the builder already placed
    (relocating their ``pos`` to the vendor cells so they render ``` ` ``` and
    shadow-cast via ``_boulder_opaque_overlay``), pin ``player_pos`` to the
    accepted branch cell, and re-seed the hero's FOV — clearing the fallback
    ``set_start_pos`` FOV seed first so no stale explored cells leak in.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.fov import view_from as _view_from
    from Nethax.nethax.constants.tiles import TileType as _TileType
    from Nethax.minihax.level_generator import (
        _boulder_opaque_overlay,
        _BOULDER_OBJ_IDX,
    )

    rows = _river_map(narrow, lava)
    w = max(len(r) for r in rows)
    h = len(rows)
    dx, dy = _vendor_geometry_center_wh(w, h)

    def _is_floor(mx: int, my: int) -> bool:
        return (
            0 <= my < h and 0 <= mx < len(rows[my]) and rows[my][mx] == "."
        )

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        # mklev flip prologue (reseed=False -> no flip; discard the two draws).
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        # 5× rndcoord($boulder_area) — column-major idx -> MAP (1+idx//5, 1+idx%5).
        boulder_cells = []
        for _ in range(5):
            vrng, idx = _vendor_rng.rn2_jax(vrng, jnp.int32(90))
            i = int(idx)
            boulder_cells.append((dy + 1 + i % 5, dx + 1 + i // 5))  # (row, col)
        # place_lregion(LR_BRANCH): first accepted (rn2(19), rn2(7)) cell that
        # is FLOOR and not the exclusion cell MAP (0,0).
        hero_rc = None
        for _ in range(200):
            vrng, hx = _vendor_rng.rn2_jax(vrng, jnp.int32(19))
            vrng, hy = _vendor_rng.rn2_jax(vrng, jnp.int32(7))
            mx, my = int(hx), int(hy)
            if (mx, my) != (0, 0) and _is_floor(mx, my):
                hero_rc = (dy + my, dx + mx)
                break
        if hero_rc is None:
            # Deterministic fallback (mkmaze.c:311-315): first valid cell.
            for my in range(h):
                for mx in range(w):
                    if (mx, my) != (0, 0) and _is_floor(mx, my):
                        hero_rc = (dy + my, dx + mx)
                        break
                if hero_rc is not None:
                    break

        # Relocate the builder's 5 boulder ground_items entries to the vendor
        # cells (positions only; the entries/type_id are already correct).
        gi = state.ground_items
        tid = gi.items.type_id[0, 0]
        cat = gi.items.category[0, 0]
        pos = gi.pos
        b = 0
        for k in range(int(tid.shape[0])):
            if b >= len(boulder_cells):
                break
            if int(tid[k]) == int(_BOULDER_OBJ_IDX) and int(cat[k]) != 0:
                br, bc = boulder_cells[b]
                pos = pos.at[0, 0, k, 0].set(jnp.int16(br))
                pos = pos.at[0, 0, k, 1].set(jnp.int16(bc))
                b += 1

        state = state.replace(
            vendor_rng=vrng,
            ground_items=gi.replace(pos=pos),
            player_pos=jnp.stack(
                [jnp.int16(hero_rc[0]), jnp.int16(hero_rc[1])]
            ),
        )

        # Re-seed the hero's reset FOV.  This mirrors
        # ``level_generator.seed_hero_fov(state, default_lit=True)`` but adds
        # the river strip to the line-of-sight occluder set: vendor
        # ``does_block`` (vision.c:167-168) treats ``typ == WATER`` as opaque
        # (like boulders and walls), so the hero cannot see past the river.
        # The River builder now retags its river cells as ``DEEPWATER`` (so
        # they render ``S_water`` instead of ``S_pool``), so occlude on
        # ``DEEPWATER`` here.  ``seed_hero_fov`` occludes only boulders, which
        # over-reveals the far bank; building the couldsee mask here with a
        # boulder + DEEPWATER overlay matches vendor's shadow-cast.  (LAVA does
        # NOT block — excluded.)
        terrain_l0 = state.terrain[0, 0]
        occ = _boulder_opaque_overlay(state, terrain_l0.shape) | (
            terrain_l0 == jnp.int8(int(_TileType.DEEPWATER))
        )
        couldsee = _view_from(
            terrain_l0,
            state.player_pos.astype(jnp.int32),
            max_radius=0,
            opaque_overlay=occ,
        )
        lit_mask = terrain_l0 != jnp.int8(int(_TileType.VOID))
        _h_g, _w_g = terrain_l0.shape
        _pr = state.player_pos[0].astype(jnp.int32)
        _pc = state.player_pos[1].astype(jnp.int32)
        _rows_g = jnp.arange(_h_g, dtype=jnp.int32)[:, None]
        _cols_g = jnp.arange(_w_g, dtype=jnp.int32)[None, :]
        within_light = (jnp.abs(_rows_g - _pr) <= jnp.int32(1)) & (
            jnp.abs(_cols_g - _pc) <= jnp.int32(1)
        )
        vis = couldsee & (lit_mask | within_light)
        # Clear any fallback set_start_pos FOV seed, then write the fresh mask.
        return state.replace(
            visible=vis,
            explored=state.explored.at[0, 0].set(vis),
            last_seen_terrain=state.last_seen_terrain.at[0, 0].set(
                jnp.where(vis, terrain_l0.astype(jnp.int8), jnp.int8(-1))
            ),
        )

    return wrapped


def _wrap_river_deepwater(
    factory: Callable[[jax.Array], "EnvState"],
) -> Callable[[jax.Array], "EnvState"]:
    """Retag the River's river cells from ``TileType.WATER`` to
    ``TileType.DEEPWATER``.

    The vendor ``river.py`` MAP draws its river strip with the ``'W'`` mapchar,
    which ``TERRAIN_CHAR_TO_TILE`` stamps as ``TileType.WATER`` — rendered by
    ``nle_obs`` as ``S_pool`` (cmap 32, glyph 2391).  Vendor NetHack, however,
    maps the ``'W'`` MAP char to terrain typ ``WATER`` (rm.h:57) which renders
    ``S_water`` (cmap 41, glyph 2400).  ``nle_obs._TILE_TO_CMAP[DEEPWATER]``
    already maps to ``S_water``, so retagging the strip's cells makes the
    ``glyphs`` obs byte-exact.

    Scoped to River: this wrapper is applied only to the River factories, whose
    only water is the river strip (River levels have no pool/moat).  A blanket
    ``WATER -> DEEPWATER`` retag over the River terrain therefore touches
    nothing but the river, and leaves the global ``TERRAIN_CHAR_TO_TILE`` /
    pool/moat ``TileType.WATER`` rendering (still ``S_pool``) untouched.  LAVA
    cells (River-Lava) are a different tile and are unaffected.
    """
    from Nethax.nethax.constants.tiles import TileType as _TileType

    def wrapped(rng: jax.Array):
        state = factory(rng)
        terr = state.terrain
        retagged = jnp.where(
            terr == jnp.int8(int(_TileType.WATER)),
            jnp.int8(int(_TileType.DEEPWATER)),
            terr,
        )
        return state.replace(terrain=retagged)

    return wrapped


def _register_river_envs(register_fn) -> None:
    # Byte-parity status (no-monster variants River / River-Lava / River-Narrow,
    # seeds 0/1/2): the map / water-strip / stair land byte-exact via the
    # centered ``_river_builder``, and ``_wrap_river_placement`` now pins the 5
    # ``rndcoord`` boulders, the ``place_lregion`` random hero start, and the
    # boulder+WATER-occluded reset FOV byte-exact off ``state.vendor_rng`` (was
    # ~353 -> ~41 with the earlier structural/boulder-FOV work -> 3-6 residual
    # cells now).
    #
    # The remaining 3-6 cells are ALL the same rendering-table gap, OUTSIDE this
    # builder's editable scope: ``nle_obs._TILE_TO_CMAP[TileType.WATER]`` maps
    # deep water to ``S_pool`` (cmap 32, glyph 2391) but vendor renders the
    # river's ``WATER`` typ as ``S_water`` (cmap 41, glyph 2400).  Both display
    # as ``}`` so the ``chars`` obs already matches; only the ``glyphs`` obs
    # differs, at the hero-visible near water column.  Fixing it means changing
    # the WATER->cmap entry in nle_obs.py (shared across every water env), not
    # the River builder.
    #
    # Monster variants (River-Monster / River-MonsterLava) still insert per-
    # monster ``makemon`` draws between the flip prologue and the boulders,
    # which is not yet modelled, so they keep the deterministic fallback
    # builder.
    variants = [
        ("MiniHack-River-v0",            False, False, 0),
        ("MiniHack-River-Monster-v0",    False, False, 5),
        ("MiniHack-River-Lava-v0",       False, True,  0),
        ("MiniHack-River-MonsterLava-v0",False, True,  5),
        ("MiniHack-River-Narrow-v0",     True,  False, 0),
    ]
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng_dl
    for env_id, narrow, lava, nm in variants:
        # Full 80x21 grid with VOID fill (INIT_MAP:solidfill,' '); the vendor
        # MAP is stamped at its GEOMETRY:center,center origin inside the
        # builder (same path as Sokoban).
        factory = _make_factory(
            _river_builder(narrow, lava, nm), w=80, h=21, fill=" ",
        )
        # Retag the river strip's cells (stamped as TileType.WATER via the 'W'
        # mapchar) to TileType.DEEPWATER so they render S_water (glyph 2400)
        # like vendor, not S_pool (2391).  Scoped to River (its only water is
        # the strip); applied to every variant, monster and no-monster alike.
        factory = _wrap_river_deepwater(factory)
        # Byte-parity path: for the no-monster variants (River / River-Lava /
        # River-Narrow) replay vendor's mklev boulder + branch-hero draws off
        # ``state.vendor_rng`` so the 5 boulders and the random hero start land
        # byte-exact.  The Monster variants insert per-monster ``makemon`` draws
        # between the flip prologue and the boulders (not yet modelled), so they
        # keep the deterministic fallback builder to avoid mis-aligning.
        if _use_vendor_rng_dl() and nm == 0:
            factory = _wrap_river_placement(factory, narrow, lava)
        rm = _lava_avoid_reward_manager() if lava else _default_goal_reward_manager()
        register_fn(env_id, factory, rm,
                    max_steps=350, category="River")


# ---------------------------------------------------------------------------
# MultiRoom envs (Group C — MiniGrid ports)
# Procedural recursive room+door placement lives in
# ``Nethax/minihax/world_gen/multiroom.py`` (MiniGrid-style: per-reset
# topology randomisation).
# ---------------------------------------------------------------------------
from Nethax.minihax.world_gen.multiroom import multiroom_factory as _multiroom_factory


def _register_multiroom_envs(register_fn) -> None:
    # 16 MultiRoom variants + 11 minigrid extras (locked, lava, monster,
    # extreme, lavamonsters, open-door).
    variants = [
        # (env_id, n, lava, locked, monster, open_door, extreme, max_steps)
        ("MiniHack-MultiRoom-N2-v0",                2,  False, False, False, False, False, 40),
        ("MiniHack-MultiRoom-N4-v0",                4,  False, False, False, False, False, 120),
        ("MiniHack-MultiRoom-N6-v0",                6,  False, False, False, False, False, 240),
        ("MiniHack-MultiRoom-N10-v0",               10, False, False, False, False, False, 360),
        ("MiniHack-MultiRoom-N6-OpenDoor-v0",       6,  False, False, False, True,  False, 240),
        ("MiniHack-MultiRoom-N10-OpenDoor-v0",      10, False, False, False, True,  False, 360),
        ("MiniHack-MultiRoom-N2-Locked-v0",         2,  False, True,  False, False, False, 40),
        ("MiniHack-MultiRoom-N4-Locked-v0",         4,  False, True,  False, False, False, 120),
        ("MiniHack-MultiRoom-N6-Locked-v0",         6,  False, True,  False, False, False, 240),
        ("MiniHack-MultiRoom-N2-Lava-v0",           2,  True,  False, False, False, False, 40),
        ("MiniHack-MultiRoom-N4-Lava-v0",           4,  True,  False, False, False, False, 120),
        ("MiniHack-MultiRoom-N6-Lava-v0",           6,  True,  False, False, False, False, 240),
        ("MiniHack-MultiRoom-N10-Lava-v0",          10, True,  False, False, False, False, 360),
        ("MiniHack-MultiRoom-N6-Lava-OpenDoor-v0",  6,  True,  False, False, True,  False, 240),
        ("MiniHack-MultiRoom-N10-Lava-OpenDoor-v0", 10, True,  False, False, True,  False, 360),
        ("MiniHack-MultiRoom-N2-Monster-v0",        2,  False, False, True,  False, False, 40),
        ("MiniHack-MultiRoom-N4-Monster-v0",        4,  False, False, True,  False, False, 120),
        ("MiniHack-MultiRoom-N6-Monster-v0",        6,  False, False, True,  False, False, 240),
        ("MiniHack-MultiRoom-N2-Extreme-v0",        2,  True,  True,  True,  False, True,  40),
        ("MiniHack-MultiRoom-N4-Extreme-v0",        4,  True,  True,  True,  False, True,  120),
        ("MiniHack-MultiRoom-N6-Extreme-v0",        6,  True,  True,  True,  False, True,  240),
        ("MiniHack-MultiRoom-N2-LavaMonsters-v0",   2,  True,  False, True,  False, False, 40),
        ("MiniHack-MultiRoom-N4-LavaMonsters-v0",   4,  True,  False, True,  False, False, 120),
        ("MiniHack-MultiRoom-N6-LavaMonsters-v0",   6,  True,  False, True,  False, False, 240),
    ]
    for (env_id, n, lava, locked, monster, open_door, extreme, ms) in variants:
        factory = _multiroom_factory(
            n, lava_walls=lava, locked=locked, monster=monster,
            open_door=open_door, extreme=extreme,
        )
        rm = _lava_avoid_reward_manager() if lava else _default_goal_reward_manager()
        register_fn(env_id, factory, rm,
                    max_steps=ms, category="MultiRoom")


# ---------------------------------------------------------------------------
# Quest envs (Group A)
# ---------------------------------------------------------------------------
def _quest_builder(difficulty: str) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        # Quest = multi-stage navigation with key + monster + goal.
        lg.add_room(x=1, y=1, w=10, h=8)
        lg.add_room(x=15, y=1, w=8, h=8)
        lg.add_corridor((10, 4), (15, 4))
        lg.set_start_pos(2, 2)
        lg.add_stair_down(x=22, y=7)
        if difficulty in ("medium", "hard"):
            lg.add_monster()
        if difficulty == "hard":
            for _ in range(2):
                lg.add_monster()
            lg.fill_terrain("L", 12, 4, 13, 4)
    return build


def _register_quest_envs(register_fn) -> None:
    """Register Quest envs.

    All 3 variants ship with static vendor .des files
    (vendor/minihack/minihack/envs/skills_quest.py:10-24).  Hard.des
    references a ``Minotaur`` monster the Minihax MONSTERS table does
    not yet include; the _des_factory probe-build catches that and
    falls back to the LG builder.  See MINIHAX_PARSER_GAPS.md.
    """
    for env_id, diff, des_name in [
        ("MiniHack-Quest-Easy-v0",   "easy",   "quest_easy.des"),
        ("MiniHack-Quest-Medium-v0", "medium", "quest_medium.des"),
        ("MiniHack-Quest-Hard-v0",   "hard",   "quest_hard.des"),
    ]:
        fallback = _make_factory(_quest_builder(diff), w=25, h=10)
        factory = _des_factory(des_name, fallback=fallback)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=1000, category="Quest")


# ---------------------------------------------------------------------------
# Memento envs (Group A)
# ---------------------------------------------------------------------------
def _memento_builder(variant: str) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        if variant == "short":
            lg.add_room(x=1, y=1, w=12, h=5)
        else:
            lg.add_room(x=1, y=1, w=20, h=10)
        lg.set_start_pos(2, 2)
        gx = 11 if variant == "short" else 19
        gy = 4 if variant == "short" else 9
        lg.add_stair_down(x=gx, y=gy)
    return build


def _register_memento_envs(register_fn) -> None:
    """Register Memento envs.

    All 3 variants ship with static vendor .des files
    (vendor/minihack/minihack/envs/memento.py:28-43): Short-F2 → memento_short,
    F2 → memento_easy, F4 → memento_hard.
    """
    variants = [
        # (env_id, builder_variant, max_steps, des_name)
        ("MiniHack-Memento-Short-F2-v0", "short", 200, "memento_short.des"),
        ("MiniHack-Memento-F2-v0",       "med",   400, "memento_easy.des"),
        ("MiniHack-Memento-F4-v0",       "med",   400, "memento_hard.des"),
    ]
    for env_id, v, ms, des_name in variants:
        fallback = _make_factory(_memento_builder(v), w=22, h=12)
        factory = _des_factory(des_name, fallback=fallback)
        register_fn(env_id, factory, _memento_rm(),
                    max_steps=ms, category="Memento")


# ---------------------------------------------------------------------------
# WoD envs (Wand of Death — Group A)
# ---------------------------------------------------------------------------
def _wod_builder(difficulty: str) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=15, h=8)
        lg.set_start_pos(2, 2)
        lg.add_stair_down(x=14, y=7)
        # Drop the wand of death near the start.  Vendor places a blessed
        # "death" wand (skills_wod.py:22-24,:86-88,:108-110,:142-143,:213).
        try:
            lg.add_object("death", "/", place=(3, 3))
        except KeyError:
            lg.add_object("random", place=(3, 3))
        # Vendor always places a minotaur target (skills_wod.py:25, :89,
        # :144, :212); the Easy RM rewards killing it by name.
        try:
            lg.add_monster("minotaur", place=(12, 6))
        except (KeyError, TypeError):
            lg.add_monster()
        if difficulty == "hard":
            lg.add_monster()
        if difficulty == "pro":
            for _ in range(3):
                lg.add_monster()
    return build


def _register_wod_envs(register_fn) -> None:
    # Only the Easy variants carry a kill-event RewardManager in vendor
    # (skills_wod.py:29-34, :59-60); Medium/Hard/Pro use add_goal_pos with
    # no RM, i.e. sparse stairs/goal reward.
    for env_id, diff in [
        ("MiniHack-WoD-Easy-Full-v0",       "easy"),
        ("MiniHack-WoD-Easy-Restricted-v0", "easy"),
        ("MiniHack-WoD-Medium-Full-v0",     "medium"),
        ("MiniHack-WoD-Medium-Restricted-v0","medium"),
        ("MiniHack-WoD-Hard-Full-v0",       "hard"),
        ("MiniHack-WoD-Hard-Restricted-v0", "hard"),
        ("MiniHack-WoD-Pro-Full-v0",        "pro"),
        ("MiniHack-WoD-Pro-Restricted-v0",  "pro"),
    ]:
        factory = _make_factory(_wod_builder(diff), w=17, h=10)
        rm = (_skill_wod_kill_rm() if diff == "easy"
              else _default_goal_reward_manager())
        register_fn(env_id, factory, rm,
                    max_steps=200, category="WoD")


# ---------------------------------------------------------------------------
# Boxoban envs (Group A — Sokoban variants from procedural Boxoban set)
# ---------------------------------------------------------------------------
def _boxoban_builder(difficulty: str) -> Callable[[LevelGenerator], None]:
    n = {"unfiltered": 2, "medium": 3, "hard": 4}.get(difficulty, 2)
    def build(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=10, h=8)
        lg.set_start_pos(2, 2)
        lg.add_stair_down(x=9, y=7)
        for i in range(n):
            x = 3 + (i * 2) % 6
            y = 3 + (i // 3)
            try:
                lg.add_object("boulder", "`", place=(x, y))
            except KeyError:
                lg.add_object("random", place=(x, y))
            lg.fill_terrain("{", 6 + i, 5, 6 + i, 5)
    return build


def _register_boxoban_envs(register_fn) -> None:
    for env_id, diff in [
        ("MiniHack-Boxoban-Unfiltered-v0", "unfiltered"),
        ("MiniHack-Boxoban-Medium-v0",     "medium"),
        ("MiniHack-Boxoban-Hard-v0",       "hard"),
    ]:
        factory = _make_factory(_boxoban_builder(diff), w=12, h=10)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=1000, category="Boxoban")


# ---------------------------------------------------------------------------
# Skill suite — single-action envs (Group A)
# ---------------------------------------------------------------------------
#
# Byte-parity note (skills_simple family: Eat/Wield/Wear/PutOn/Zap/Read/
# Pray/Sink).  Vendor builds these as a lit 5x5 room via
#   LevelGenerator(w=5, h=5, lit=True); lvl_gen.add_object(item, sym)   (or
#   add_altar / add_sink)                         (skills_simple.py:10-19,...)
# with NO explicit start_pos and NO stair.  The header carries
# ``GEOMETRY:center,center`` so NLE centers the 5x5 MAP on the 80x21 dungeon
# exactly like the Room envs -> internal rect cols 37..41, rows 9..13
# (``_vendor_geometry_center(5)``).  Both the object cell AND the player spawn
# are RNG-driven by mklev.  Previously the Minihax builder stamped a tiny 5x5
# LG at terrain[0:5,0:5] (top-left corner) with an @-at-origin, producing a
# degenerate empty level: glyph 327 at obs (0,0) vs vendor stone 2359.
#
# We fix it the same way ``_register_room_envs`` fixed Room-Random: build a
# full 80x21 VOID grid, carve the centered 5x5 FLOOR room, then a wrapper
# (`_wrap_skill_placement`) consumes vendor's exact mklev ISAAC64 draw stream:
#   1. rn2(3), rn2(2)                      -- level setup prefix
#   2. rn2(5), rn2(5)                      -- object somexy() room-relative
#      (x_off, y_off): object cell = (x1+x_off, y1+y_off)
#   3. mksobj_init draws for the object's class (see _consume_mksobj_draws)
#   4. faithful place_lregion (mkmaze.c:275-319): 200-try (rn2(79)+1, rn2(21))
#      accept first in-room FLOOR cell that is not the object cell -> player.
# Ground truth: .test_runs/skill_rnd_stream_*_seed0.txt (NETHAX_RND/RN2 trace
# of vendor MiniHack-{Eat,Wield,Zap,Pray,...}).
# ---------------------------------------------------------------------------
def _skill_room_builder(size: int, lit: bool) -> Callable[[LevelGenerator], None]:
    """Carve a ``size``x``size`` FLOOR room at the vendor-centered location on
    the full 80x21 VOID grid (mirrors ``_room_builder(size, random=True)`` but
    without any stair/object/start — those are stamped by the wrapper)."""
    x0, y0 = _vendor_geometry_center(size)
    x1, y1 = x0 + size - 1, y0 + size - 1

    def build(lg: LevelGenerator) -> None:
        lg.fill_terrain(".", x0, y0, x1, y1)
    return build


def _consume_mksobj_draws(vrng, obj_class: int):
    """Replay the vendor ``mksobj_init`` ISAAC64 draw sequence for a des-placed
    named object (``artif=TRUE``), returning the advanced ``vrng``.

    Faithful port of vendor/nethack/src/mkobj.c::mksobj_init for the object
    classes used by skills_simple (WEAPON/ARMOR/WAND/AMULET/SCROLL/FOOD-apple).
    Only draw *consumption* is modelled (the resulting item is stamped
    separately from the des directive) — this keeps ``state.vendor_rng``
    byte-aligned so the subsequent player place_lregion draws land on vendor's
    offsets.  Runs host-side (eager) like the Room wrappers; every draw goes
    through ``rn2_jax`` so the JIT trace records the right modulus.

    Cite: mkobj.c:876-1097 (class switch), :bless­orcurse (rn2(chance)[,rn2(2)]),
    rnd.c::rne (while tmp<5 && !rn2(x)), rn1(x,base)=rn2(x)+base.
    """
    from Nethax.nethax import vendor_rng as _vr
    from Nethax.nethax.constants.objects import ObjectClass as _OC

    def rn2(v, n):
        v, r = _vr.rn2_jax(v, jnp.int32(n))
        return v, int(r)

    def rne(v, x):
        # utmp=5 for ulevel<15; tmp starts 1, increments while tmp<5 && !rn2(x).
        tmp = 1
        while tmp < 5:
            v, r = rn2(v, x)
            if r != 0:
                break
            tmp += 1
        return v

    def blessorcurse(v, chance):
        v, r = rn2(v, chance)
        if r == 0:
            v, _ = rn2(v, 2)   # curse vs bless
        return v

    c = int(obj_class)
    if c == int(_OC.WEAPON_CLASS):
        # dagger: is_multigen=False (no quan draw), is_poisonable=False in this
        # build (no rn2(100)).  mkobj.c:876-893.
        v, r = rn2(vrng, 11)
        if r == 0:
            v = rne(v, 3)          # spe = rne(3)
            v, _ = rn2(v, 2)       # blessed = rn2(2)
        else:
            v, r = rn2(v, 10)
            if r == 0:
                v = rne(v, 3)      # curse; spe = -rne(3)
            else:
                v = blessorcurse(v, 10)
        v, _ = rn2(v, 20)          # artif && !rn2(20 + 10*nartifact_exist()=0)
        return v
    if c == int(_OC.ARMOR_CLASS):
        # robe: not fumble/levitation boots etc -> first operand rn2(10) plus
        # the ``|| !rn2(11)`` short-circuit.  mkobj.c:1085-1097.
        v, r = rn2(vrng, 10)
        take_curse = False
        if r != 0:
            v, r2 = rn2(v, 11)     # (... || !rn2(11))
            take_curse = (r2 == 0)
        if take_curse:
            v = rne(v, 3)          # curse; spe = -rne(3)
        else:
            v, r = rn2(v, 10)
            if r == 0:
                v, _ = rn2(v, 2)   # blessed = rn2(2)
                v = rne(v, 3)      # spe = rne(3)
            else:
                v = blessorcurse(v, 10)
        v, _ = rn2(v, 40)          # artif && !rn2(40 + 0)
        return v
    if c == int(_OC.WAND_CLASS):
        # enlightenment (NODIR): spe = rn1(5, 11) -> rn2(5); then blessorcurse(17).
        v, _ = rn2(vrng, 5)
        v = blessorcurse(v, 17)
        return v
    if c == int(_OC.AMULET_CLASS):
        # amulet of life saving: first `&&` operand rn2(10) always drawn, then
        # blessorcurse(10).  mkobj.c:1060-1069.
        v, _ = rn2(vrng, 10)
        v = blessorcurse(v, 10)
        return v
    if c == int(_OC.SCROLL_CLASS):
        # blank paper: blessorcurse(4).  mkobj.c:1075-1080.
        return blessorcurse(vrng, 4)
    if c == int(_OC.FOOD_CLASS):
        # apple (not corpse/egg/tin/mold/kelp/candy): trailing `!rn2(6)`
        # quan=2 check.  mkobj.c:969-974.
        v, _ = rn2(vrng, 6)
        return v
    # Unknown class: consume nothing (best-effort; item glyph still correct).
    return vrng


# NetHack object-class symbols (``def_oc_syms``, vendor/nethack/src/drawing.c).
# A des ``OBJECT`` directive carries the class symbol, which disambiguates the
# name when the same bare name lives in multiple classes.  Notably
# "enlightenment" exists as BOTH the potion (otyp 285) and the wand (otyp 385);
# ``skills_simple.MiniHackZap`` uses ``add_object("enlightenment", "/")`` so the
# '/' symbol pins it to the WAND.  Without the symbol, the bare-name lookup
# (setdefault, class-order) returns the potion -> wrong floor glyph.
_SKILL_SYM_TO_OBJCLASS: dict = {
    ")": "WEAPON_CLASS",
    "[": "ARMOR_CLASS",
    "=": "RING_CLASS",
    '"': "AMULET_CLASS",
    "(": "TOOL_CLASS",
    "%": "FOOD_CLASS",
    "!": "POTION_CLASS",
    "?": "SCROLL_CLASS",
    "+": "SPBOOK_CLASS",
    "/": "WAND_CLASS",
    "*": "GEM_CLASS",
}


def _resolve_skill_obj_idx(item_name: str, symbol: Optional[str]) -> int:
    """Resolve an OBJECTS index for a des ``OBJECT`` name, constrained to the
    class implied by ``symbol`` when the name is class-ambiguous.

    Falls back to the plain bare-name lookup (``_OBJECT_NAME_TO_IDX``) when the
    symbol is unknown or no name+class match exists — preserving behaviour for
    the unambiguous skill objects (apple/dagger/robe/…).
    """
    from Nethax.nethax.constants.objects import OBJECTS as _OBJ, ObjectClass
    from Nethax.minihax.level_generator import _OBJECT_NAME_TO_IDX as _N2I
    if symbol is not None and symbol in _SKILL_SYM_TO_OBJCLASS:
        want = int(getattr(ObjectClass, _SKILL_SYM_TO_OBJCLASS[symbol]))
        for i, e in enumerate(_OBJ):
            if e.name == item_name and int(e.class_) == want:
                return i
    return _N2I[item_name]


def _wrap_skill_placement(
    factory: Callable[[jax.Array], "EnvState"],
    size: int,
    *,
    item_name: Optional[str],
    feature: Optional[str],
    symbol: Optional[str] = None,
    fixed: bool = False,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Wrap a ``_skill_room_builder`` factory so it stamps the vendor object /
    altar / sink and the RNG-placed player, consuming vendor's exact mklev
    draws (see the family byte-parity note above).

    ``item_name``: OBJECTS-table name for a ground object (Eat/Wield/Wear/
    PutOn/Zap/Read), or ``None`` for a terrain feature.
    ``feature``: ``"altar"`` (Pray) or ``"sink"`` (Sink) -> stamp a terrain
    tile instead of a ground item; ``None`` otherwise.
    ``symbol``: des class symbol (e.g. ``"/"``) used to disambiguate a
    class-ambiguous ``item_name`` (see ``_resolve_skill_obj_idx``).
    ``fixed``: the ``-Fixed`` variant — object/feature pinned at room-relative
    ``(0, 0)`` and the hero at ``(2, 2)`` (vendor ``add_object(place=(0,0))``
    + ``set_start_pos((2,2))``); both are deterministic so NO mklev ISAAC64
    draws are consumed.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TT
    from Nethax.nethax.constants.objects import OBJECTS as _OBJECTS
    from Nethax.minihax.level_generator import (
        seed_hero_fov as _seed_hero_fov,
        _write_ground_item as _write_gi,
    )
    from Nethax.nethax.subsystems.ground_items_sparse import (
        dense_to_sparse as _dense_to_sparse,
        sparse_to_dense as _sparse_to_dense,
    )

    obj_idx = None
    obj_class = None
    if item_name is not None:
        obj_idx = _resolve_skill_obj_idx(item_name, symbol)
        obj_class = int(_OBJECTS[obj_idx].class_)

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)  # internal top-left of room rect

        _FLOOR = int(_TT.FLOOR)
        new_terrain = state.terrain

        if fixed:
            # -Fixed: object/feature pinned at room-relative (0, 0); no somexy
            # or mksobj draws (des ``place=(0,0)`` is deterministic).
            obj_col = int(x1)
            obj_row = int(y1)
        else:
            # (1) level-setup prefix: rn2(3), rn2(2).
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

            # (2) object/feature somexy(): room-relative (x_off, y_off).
            vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
            vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
            obj_col = int(x1) + int(ox)
            obj_row = int(y1) + int(oy)

        if feature is None:
            if not fixed:
                # (3) object mksobj_init draws (class-dependent).
                vrng = _consume_mksobj_draws(vrng, obj_class)
            # Stamp the ground item at the object cell.  Round-trip through the
            # dense buffer so we reuse the level_generator writer.
            dense = _sparse_to_dense(state.ground_items)
            dense, _ = _write_gi(dense, {}, (obj_row, obj_col), int(obj_idx))
            state = state.replace(
                ground_items=_dense_to_sparse(dense, state.ground_items.K)
            )
        elif feature == "altar":
            new_terrain = new_terrain.at[0, 0, obj_row, obj_col].set(
                jnp.int8(int(_TT.ALTAR))
            )
        elif feature == "sink":
            # TileType.SINK renders S_sink (glyph 2389) now that nle_obs maps it
            # (was a FOUNTAIN proxy -> S_fountain glyph 2390).
            new_terrain = new_terrain.at[0, 0, obj_row, obj_col].set(
                jnp.int8(int(_TT.SINK))
            )

        if fixed:
            # -Fixed: hero pinned at room-relative (2, 2) (set_start_pos((2,2)));
            # no place_lregion draws.
            acc_x = int(x1) + 2
            acc_y = int(y1) + 2
        else:
            # (4) player place_lregion: 200-try (rn2(79)+1, rn2(21)); accept
            #     first in-room FLOOR cell that is not the object cell.
            #     mkmaze.c:275-319.
            import numpy as _np
            terr_np = _np.asarray(new_terrain[0, 0])
            _H, _W = terr_np.shape
            ok = (terr_np == _FLOOR)
            ok[obj_row, obj_col] = False  # object cell is occupied
            acc_x = int((x1 + x1 + size - 1) // 2)
            acc_y = int((y1 + y1 + size - 1) // 2)
            accepted = False
            for _ in range(200):
                vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
                vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
                cx = int(raw_x) + 1
                cy = int(cand_y)
                if 0 <= cy < _H and 0 <= cx < _W and bool(ok[cy, cx]):
                    acc_x, acc_y = cx, cy
                    accepted = True
                    break
            if not accepted:
                for sx in range(1, _W):
                    for sy in range(0, _H):
                        if bool(ok[sy, sx]):
                            acc_x, acc_y = sx, sy
                            accepted = True
                            break
                    if accepted:
                        break

        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [jnp.int32(acc_y).astype(jnp.int16),
                 jnp.int32(acc_x).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _skill_eat_builder(distr: bool, fixed: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        place = (0, 0) if fixed else None
        try:
            lg.add_object("apple", "%", place=place)
        except KeyError:
            lg.add_object("random", place=place)
        if fixed:
            lg.set_start_pos(2, 2)
        if distr:
            lg.add_monster()
            lg.add_object()
        lg.add_stair_down(x=4, y=4)
    return build


def _skill_simple_builder(item: str, symbol: str,
                          distr: bool, fixed: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        place = (0, 0) if fixed else None
        try:
            lg.add_object(item, symbol, place=place)
        except KeyError:
            lg.add_object("random", place=place)
        if fixed:
            lg.set_start_pos(2, 2)
        if distr:
            lg.add_monster()
            lg.add_object()
        lg.add_stair_down(x=4, y=4)
    return build


# ---------------------------------------------------------------------------
# Levitate skill family byte-parity.
#
# Vendor (skills_levitate.py) builds each Levitate env as a lit 5x5 room via
#   LevelGenerator(w=5, h=5, lit=True); lvl_gen.add_object(<lev item>, sym,
#                                                           cursestate="blessed")
# with GEOMETRY:center,center and NO stair (the RM fires on the levitation
# message, not stairs_down).  This mirrors the skills_simple base family
# (_skill_room_builder + _wrap_skill_placement) exactly EXCEPT for the item
# mksobj_init draws, which are class/otyp-specific for the levitation items.
#
# Full / Restricted variants (Restricted shares Full's des, only restricting
# the action set) place BOTH object and player at RNG cells:
#   1. rn2(3), rn2(2)              -- level-setup prefix
#   2. rn2(5), rn2(5)             -- object somexy (x_off, y_off)
#   3. item mksobj_init draws     (_consume_levitate_item_draws)
#   4. player place_lregion        -- 200-try (rn2(79)+1, rn2(21))
# Fixed variant fixes object at des (0,0) and player at (2,2):
#   1. rn2(3), rn2(2)             -- prefix (NO object somexy: fixed cell)
#   2. item mksobj_init draws
#   3. rn2(1), rn2(1)            -- single-cell player place_lregion (trivial)
# Random-Full picks the item TYPE via IF[33%]/ELSE-IF[50%] (rn2(100)[,rn2(100)])
# before the object somexy, otherwise identical to the Full path.
# Ground-truthed from NETHAX_RN2_TRACE of the vendor envs, seeds 0/1/2
# (item draws verified against mkobj.c::mksobj_init below).
# ---------------------------------------------------------------------------
def _consume_levitate_item_draws(vrng, item_key: str):
    """Replay the vendor ``mksobj_init`` ISAAC64 draws for a levitation item.

    Faithful to vendor/nethack/src/mkobj.c::mksobj_init:
      * potion of levitation (POTION_CLASS, :1074-1080): blessorcurse(4);
        POT_WATER post-init draws nothing.
      * ring of levitation (RING_CLASS, uncharged branch :1143-1148):
        rn2(10); only when nonzero is ``!rn2(9)`` evaluated -> rn2(9).
      * levitation boots (ARMOR_CLASS, :1085-1098): rn2(10); when nonzero the
        ``&&`` short-circuits on ``otyp == LEVITATION_BOOTS`` (NO rn2(11)) ->
        curse; spe = -rne(3).  When zero -> ``else if (!rn2(10))`` -> rn2(10);
        if zero: blessed = rn2(2), spe = rne(3); else blessorcurse(10).
        Then artif: rn2(40).
    """
    from Nethax.nethax import vendor_rng as _vr

    def rn2(v, n):
        v, r = _vr.rn2_jax(v, jnp.int32(n))
        return v, int(r)

    def rne(v, x):
        tmp = 1
        while tmp < 5:
            v, r = rn2(v, x)
            if r != 0:
                break
            tmp += 1
        return v

    def blessorcurse(v, chance):
        v, r = rn2(v, chance)
        if r == 0:
            v, _ = rn2(v, 2)
        return v

    if item_key == "potion":
        return blessorcurse(vrng, 4)
    if item_key == "ring":
        vrng, r = rn2(vrng, 10)
        if r != 0:
            vrng, _ = rn2(vrng, 9)
        return vrng
    # levitation boots (ARMOR_CLASS with the LEVITATION_BOOTS otyp short-circuit)
    vrng, r = rn2(vrng, 10)
    if r != 0:
        vrng = rne(vrng, 3)            # curse; spe = -rne(3)
    else:
        vrng, r2 = rn2(vrng, 10)       # else if (!rn2(10))
        if r2 == 0:
            vrng, _ = rn2(vrng, 2)     # blessed = rn2(2)
            vrng = rne(vrng, 3)        # spe = rne(3)
        else:
            vrng = blessorcurse(vrng, 10)
    vrng, _ = rn2(vrng, 40)            # artif && !rn2(40 + 0)
    return vrng


def _levitate_place_player(vrng, terr_np, blocked, x1, y1, size):
    """Faithful place_lregion for the room player start (mkmaze.c:275-319).

    ``blocked`` is a set of (row, col) cells excluded from acceptance (the
    object cell).  Returns (advanced vrng, acc_x, acc_y).
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TT

    _FLOOR = int(_TT.FLOOR)
    _H, _W = terr_np.shape
    ok = (terr_np == _FLOOR)
    for (r, c) in blocked:
        ok[r, c] = False
    acc_x = int((x1 + x1 + size - 1) // 2)
    acc_y = int((y1 + y1 + size - 1) // 2)
    accepted = False
    for _ in range(200):
        vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
        vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
        cx = int(raw_x) + 1
        cy = int(cand_y)
        if 0 <= cy < _H and 0 <= cx < _W and bool(ok[cy, cx]):
            acc_x, acc_y = cx, cy
            accepted = True
            break
    if not accepted:
        for sx in range(1, _W):
            for sy in range(0, _H):
                if bool(ok[sy, sx]):
                    acc_x, acc_y = sx, sy
                    accepted = True
                    break
            if accepted:
                break
    return vrng, acc_x, acc_y


def _stamp_levitate_item(state, item_name, obj_row, obj_col):
    """Write ``item_name`` as a ground item at (obj_row, obj_col)."""
    from Nethax.minihax.level_generator import (
        _OBJECT_NAME_TO_IDX as _NAME2IDX,
        _write_ground_item as _write_gi,
    )
    from Nethax.nethax.subsystems.ground_items_sparse import (
        dense_to_sparse as _dense_to_sparse,
        sparse_to_dense as _sparse_to_dense,
    )
    obj_idx = _NAME2IDX.get(item_name)
    if obj_idx is None:
        return state
    dense = _sparse_to_dense(state.ground_items)
    dense, _ = _write_gi(dense, {}, (obj_row, obj_col), int(obj_idx))
    return state.replace(
        ground_items=_dense_to_sparse(dense, state.ground_items.K)
    )


def _wrap_skill_levitate_placement(
    factory: Callable[[jax.Array], "EnvState"],
    item_name: str,
    item_key: str,
    *,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Levitate-{Boots,Ring,Potion}-{Full,Restricted}: random object + player."""
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import numpy as _np

    size = 5

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        obj_col = int(x1) + int(ox)
        obj_row = int(y1) + int(oy)

        vrng = _consume_levitate_item_draws(vrng, item_key)
        state = _stamp_levitate_item(state, item_name, obj_row, obj_col)

        terr_np = _np.asarray(state.terrain[0, 0])
        vrng, acc_x, acc_y = _levitate_place_player(
            vrng, terr_np, {(obj_row, obj_col)}, x1, y1, size
        )

        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [jnp.int32(acc_y).astype(jnp.int16),
                 jnp.int32(acc_x).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_skill_levitate_fixed(
    factory: Callable[[jax.Array], "EnvState"],
    item_name: str,
    item_key: str,
    *,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Levitate-{Boots,Ring,Potion}-Fixed: object at des (0,0), player at (2,2)."""
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov

    size = 5

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        # Fixed object cell (des (0,0)) -> no somexy draw.
        obj_col = int(x1) + 0
        obj_row = int(y1) + 0
        vrng = _consume_levitate_item_draws(vrng, item_key)
        state = _stamp_levitate_item(state, item_name, obj_row, obj_col)

        # Single-cell player place_lregion (des set_start_pos (2,2)): rn2(1)x2.
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(1))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(1))
        start_col = int(x1) + 2
        start_row = int(y1) + 2

        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [jnp.int32(start_row).astype(jnp.int16),
                 jnp.int32(start_col).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_skill_levitate_random(
    factory: Callable[[jax.Array], "EnvState"],
    *,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Levitate-Random-Full: item TYPE RNG-chosen, then Full-style placement.

    Vendor des IF[33%]{potion} ELSE IF[50%]{ring} ELSE {boots}, each IF drawing
    rn2(100).  Ground-truthed seeds 0/1/2 (66>=33->2<50=ring, 6<33=potion,
    87>=33->15<50=ring).
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import numpy as _np

    size = 5

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        vrng, r1 = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        if int(r1) < 33:
            item_name, item_key = "potion of levitation", "potion"
        else:
            vrng, r2 = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
            if int(r2) < 50:
                item_name, item_key = "ring of levitation", "ring"
            else:
                item_name, item_key = "levitation boots", "boots"

        vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        obj_col = int(x1) + int(ox)
        obj_row = int(y1) + int(oy)

        vrng = _consume_levitate_item_draws(vrng, item_key)
        state = _stamp_levitate_item(state, item_name, obj_row, obj_col)

        terr_np = _np.asarray(state.terrain[0, 0])
        vrng, acc_x, acc_y = _levitate_place_player(
            vrng, terr_np, {(obj_row, obj_col)}, x1, y1, size
        )

        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [jnp.int32(acc_y).astype(jnp.int16),
                 jnp.int32(acc_x).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _skill_pray_builder(distr: bool, fixed: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        # Altar tile in the room.
        lg.fill_terrain("\\", 2, 2, 2, 2)
        if fixed:
            lg.set_start_pos(0, 0)
        if distr:
            lg.add_monster()
        lg.add_stair_down(x=4, y=4)
    return build


def _skill_sink_builder(distr: bool, fixed: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        # Fountain tile (proxy for sink in Wave 4 schema).
        lg.fill_terrain("{", 2, 2, 2, 2)
        if fixed:
            lg.set_start_pos(0, 0)
        if distr:
            lg.add_monster()
        lg.add_stair_down(x=4, y=4)
    return build


# ---------------------------------------------------------------------------
# Freeze skill family byte-parity.
#
# Vendor (skills_freeze.py) builds Wand/Horn as a lit 8x8 room via
#   LevelGenerator(w=8, h=8, lit=True); add_object(<item>, sym, blessed)
# with GEOMETRY:center,center and NO stair (RM fires on the cold-bounce
# message).  Random-{Full,Restricted} use an inline 8x8 MAP des that picks the
# item TYPE via IF[50%] before the OBJECT directive.  Lava-{Full,Restricted}
# reuse the LavaCross 13x7 lava MAP (IF[50%] item pick, on-bank OBJECT
# rndcoord, right-bank STAIR, BRANCH player) with the default sparse RM.
#
# Item mksobj_init draws (vendor/nethack/src/mkobj.c):
#   wand of cold (WAND_CLASS, oc_dir != NODIR): spe = rn1(5, 4) -> rn2(5),
#       then blessorcurse(otmp, 17)          (mkobj.c:1115-1125)
#   frost horn  (TOOL_CLASS):  spe = rn1(5, 4) -> rn2(5), no blessorcurse
#       (mkobj.c:1051-1057)
# Placement draw order mirrors the Levitate (5x5 room) / LavaCross (13x7)
# families, which are already byte-exact.
# ---------------------------------------------------------------------------
def _consume_freeze_item_draws(vrng, item_key: str):
    """Replay the vendor ``mksobj_init`` ISAAC64 draws for a freeze item.

    ``"wand"`` (wand of cold, WAND_CLASS DIR): ``rn2(5)`` then
    ``blessorcurse(17)``.  ``"horn"`` (frost horn, TOOL_CLASS): ``rn2(5)`` only.
    """
    from Nethax.nethax import vendor_rng as _vr

    def rn2(v, n):
        v, r = _vr.rn2_jax(v, jnp.int32(n))
        return v, int(r)

    def blessorcurse(v, chance):
        v, r = rn2(v, chance)
        if r == 0:
            v, _ = rn2(v, 2)
        return v

    vrng, _ = rn2(vrng, 5)          # spe = rn1(5, 4)
    if item_key == "wand":
        vrng = blessorcurse(vrng, 17)
    return vrng


def _wrap_freeze_placement(
    factory: Callable[[jax.Array], "EnvState"],
    item_name: str,
    item_key: str,
    *,
    size: int = 8,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Freeze-{Wand,Horn}-{Full,Restricted}: RNG object + RNG player in an 8x8
    lit room (mirrors ``_wrap_skill_levitate_placement``)."""
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import numpy as _np

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        obj_col = int(x1) + int(ox)
        obj_row = int(y1) + int(oy)

        vrng = _consume_freeze_item_draws(vrng, item_key)
        state = _stamp_levitate_item(state, item_name, obj_row, obj_col)

        terr_np = _np.asarray(state.terrain[0, 0])
        vrng, acc_x, acc_y = _levitate_place_player(
            vrng, terr_np, {(obj_row, obj_col)}, x1, y1, size
        )
        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [jnp.int32(acc_y).astype(jnp.int16),
                 jnp.int32(acc_x).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_freeze_random(
    factory: Callable[[jax.Array], "EnvState"],
    *,
    size: int = 8,
    lit: bool = True,
) -> Callable[[jax.Array], "EnvState"]:
    """Freeze-Random-{Full,Restricted}: item TYPE RNG-chosen (``IF[50%]``:
    ``rn2(100) < 50`` -> wand of cold, else frost horn), then Full-style
    8x8 room placement."""
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov
    import numpy as _np

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        x1, y1 = _vendor_geometry_center(size)

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        vrng, r = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        if int(r) < 50:
            item_name, item_key = "cold", "wand"
        else:
            item_name, item_key = "frost horn", "horn"

        vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        obj_col = int(x1) + int(ox)
        obj_row = int(y1) + int(oy)

        vrng = _consume_freeze_item_draws(vrng, item_key)
        state = _stamp_levitate_item(state, item_name, obj_row, obj_col)

        terr_np = _np.asarray(state.terrain[0, 0])
        vrng, acc_x, acc_y = _levitate_place_player(
            vrng, terr_np, {(obj_row, obj_col)}, x1, y1, size
        )
        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [jnp.int32(acc_y).astype(jnp.int16),
                 jnp.int32(acc_x).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, lit)

    return wrapped


def _wrap_freeze_lava(
    factory: Callable[[jax.Array], "EnvState"],
) -> Callable[[jax.Array], "EnvState"]:
    """Freeze-Lava-{Full,Restricted}: LavaCross 13x7 lava MAP.

    Vendor des directive order OBJECT/BRANCH/STAIR, but the BRANCH player
    ``place_lregion`` is deferred to level end, so the ISAAC64 draw order is
    (matching ``_wrap_lavacross_placement``):
      1. rn2(3), rn2(2)          -- level-setup prefix
      2. rn2(100)                -- IF[50%]: <50 wand of cold, else frost horn
      3. rn2(25)                 -- OBJECT rndcoord($left_bank)  (v//5, v%5)
      4. freeze item mksobj draws
      5. rn2(25)                 -- STAIR rndcoord($right_bank)
      6. rn2(5), rn2(5)          -- BRANCH player (xrel, yrel) in left_bank
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.nethax.constants.tiles import TileType as _TT
    from Nethax.minihax.level_generator import (
        seed_hero_fov as _seed_hero_fov,
        _OBJECT_NAME_TO_IDX as _NAME2IDX,
        _write_ground_item as _write_gi,
    )
    from Nethax.nethax.subsystems.ground_items_sparse import (
        dense_to_sparse as _dense_to_sparse,
        sparse_to_dense as _sparse_to_dense,
    )

    xstart, ystart = _vendor_geometry_center_wh(13, 7)
    ix0 = xstart + 1
    iy0 = ystart + 1
    left_x0 = ix0
    right_x0 = ix0 + 6
    row0 = iy0

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng

        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))

        vrng, r = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        if int(r) < 50:
            item_name, item_key = "cold", "wand"
        else:
            item_name, item_key = "frost horn", "horn"

        vrng, oi = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        oi = int(oi)
        obj_col = left_x0 + (oi // 5)
        obj_row = row0 + (oi % 5)

        vrng = _consume_freeze_item_draws(vrng, item_key)

        vrng, si = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        si = int(si)
        stair_col = right_x0 + (si // 5)
        stair_row = row0 + (si % 5)

        vrng, px = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        vrng, py = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        start_col = left_x0 + int(px)
        start_row = row0 + int(py)

        new_terrain = state.terrain.at[
            0, 0, stair_row, stair_col
        ].set(jnp.int8(int(_TT.STAIRCASE_DOWN)))

        obj_idx = _NAME2IDX.get(item_name)
        if obj_idx is not None:
            dense = _sparse_to_dense(state.ground_items)
            dense, _ = _write_gi(dense, {}, (obj_row, obj_col), int(obj_idx))
            state = state.replace(
                ground_items=_dense_to_sparse(dense, state.ground_items.K)
            )

        state = state.replace(
            vendor_rng=vrng,
            terrain=new_terrain,
            player_pos=jnp.stack(
                [jnp.int32(start_row).astype(jnp.int16),
                 jnp.int32(start_col).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _register_skill_simple_envs(register_fn) -> None:
    """Eat / Wield / Wear / PutOn / Zap / Read / Pray / Sink — 24 envs.

    RM per family mirrors vendor ``skills_simple.py``: each env pays its
    targeted event (eat-apple, wield-dagger, amulet-message, ...), NOT
    sparse stairs_down.
    """
    item_specs = [
        # (basename, item, symbol, rm_factory)
        # Item names/symbols mirror vendor skills_simple.py exactly so the
        # vendor RM message predicate can fire on the correct object:
        #   Wield -> "dagger", ")"            (skills_simple.py:62)
        #   Wear  -> "robe", "["              (skills_simple.py:113)
        #   PutOn -> "amulet of life saving", '"' (skills_simple.py:164)
        #   Zap   -> "enlightenment", "/"     (skills_simple.py:215)
        #   Read  -> "blank paper", "?"       (skills_simple.py:266)
        ("Wield", "dagger",                ")", _skill_wield_rm),
        ("Wear",  "robe",                  "[", _skill_wear_rm),
        ("PutOn", "amulet of life saving", '"', _skill_amulet_rm),
        ("Zap",   "enlightenment",         "/", _skill_zap_rm),
        ("Read",  "blank paper",           "?", _skill_read_rm),
    ]
    # Base and -Fixed variants build the full-fidelity vendor 5x5 room
    # (byte-parity): base RNG-places the object/hero, -Fixed pins the object at
    # room-relative (0,0) and the hero at (2,2) (deterministic, no draws).  The
    # -Distr variant keeps the legacy compact builder (out of byte-parity scope:
    # its extra ``add_monster()`` + random ``add_object()`` need faithful random
    # monster/object generation the LG object path does not yet reproduce).
    def _base_skill_factory(item_name, feature, symbol=None, fixed=False):
        builder = _skill_room_builder(5, lit=True)
        factory = _make_factory(builder, w=80, h=21, fill=" ", lit=True)
        return _wrap_skill_placement(
            factory, 5, item_name=item_name, feature=feature,
            symbol=symbol, fixed=fixed, lit=True,
        )

    for base, item, symbol, rm_factory in item_specs:
        for suffix, distr, fixed in [
            ("",       False, False),
            ("-Fixed", False, True),
            ("-Distr", True,  False),
        ]:
            env_id = f"MiniHack-{base}{suffix}-v0"
            if not distr:
                factory = _base_skill_factory(item, None, symbol=symbol,
                                              fixed=fixed)
            else:
                builder = _skill_simple_builder(item, symbol, distr, fixed)
                factory = _make_factory(builder, w=5, h=5)
            register_fn(env_id, factory, rm_factory(),
                        max_steps=50, category="Skill")

    # Eat variants
    for suffix, distr, fixed in [
        ("",       False, False),
        ("-Fixed", False, True),
        ("-Distr", True,  False),
    ]:
        env_id = f"MiniHack-Eat{suffix}-v0"
        if not distr:
            factory = _base_skill_factory("apple", None, symbol="%",
                                          fixed=fixed)
        else:
            builder = _skill_eat_builder(distr, fixed)
            factory = _make_factory(builder, w=5, h=5)
        register_fn(env_id, factory, _skill_eat_rm(),
                    max_steps=50, category="Skill")

    # Pray variants
    for suffix, distr, fixed in [
        ("",       False, False),
        ("-Fixed", False, True),
        ("-Distr", True,  False),
    ]:
        env_id = f"MiniHack-Pray{suffix}-v0"
        if not distr:
            factory = _base_skill_factory(None, "altar", fixed=fixed)
        else:
            builder = _skill_pray_builder(distr, fixed)
            factory = _make_factory(builder, w=5, h=5)
        register_fn(env_id, factory, _skill_pray_rm(),
                    max_steps=50, category="Skill")

    # Sink variants
    for suffix, distr, fixed in [
        ("",       False, False),
        ("-Fixed", False, True),
        ("-Distr", True,  False),
    ]:
        env_id = f"MiniHack-Sink{suffix}-v0"
        if not distr:
            # NOTE: the sink cell still renders as FOUNTAIN (glyph 2390) vs
            # vendor's real sink (2389); emitting 2389 needs a shared-infra
            # _TILE_TO_CMAP[SINK] change in nle_obs.py (out of scope here).  The
            # room geometry / hero placement are byte-exact regardless.
            factory = _base_skill_factory(None, "sink", fixed=fixed)
        else:
            builder = _skill_sink_builder(distr, fixed)
            factory = _make_factory(builder, w=5, h=5)
        register_fn(env_id, factory, _skill_sink_rm(),
                    max_steps=50, category="Skill")


def _register_skill_levitate_envs(register_fn) -> None:
    """9 Levitate envs.

    Vendor (``skills_levitate.py:16-19``): RM is
    ``add_message_event(levitation_msg)`` — reward fires the moment the player
    starts floating.
    """
    item_specs = [
        ("Boots",   "levitation boots",      "boots"),
        ("Ring",    "ring of levitation",    "ring"),
        ("Potion",  "potion of levitation",  "potion"),
    ]
    for base, item_name, item_key in item_specs:
        for suffix in ("-Full", "-Restricted", "-Fixed"):
            env_id = f"MiniHack-Levitate-{base}{suffix}-v0"
            # Full 80x21 VOID grid so GEOMETRY:center,center lands the lit 5x5
            # MAP at vendor's internal origin (mirrors the skills_simple base
            # path); the wrapper stamps the item + RNG-placed player.
            base_factory = _make_factory(
                _skill_room_builder(5, lit=True), w=80, h=21, fill=" ", lit=True
            )
            if suffix == "-Fixed":
                factory = _wrap_skill_levitate_fixed(
                    base_factory, item_name, item_key
                )
            else:
                factory = _wrap_skill_levitate_placement(
                    base_factory, item_name, item_key
                )
            register_fn(env_id, factory, _skill_levitate_rm(),
                        max_steps=50, category="Skill")
    # Levitate-Random-Full: item TYPE is RNG-chosen.
    base_factory = _make_factory(
        _skill_room_builder(5, lit=True), w=80, h=21, fill=" ", lit=True
    )
    factory = _wrap_skill_levitate_random(base_factory)
    register_fn("MiniHack-Levitate-Random-Full-v0", factory,
                _skill_levitate_rm(),
                max_steps=50, category="Skill")


def _register_skill_freeze_envs(register_fn) -> None:
    """8 Freeze envs (byte-parity).

    Vendor (``skills_freeze.py``): RM is ``add_message_event(freeze_msgs)`` for
    Wand/Horn/Random.  ``Freeze-Lava-*`` constructs ``MiniHackSkill`` without a
    RM (vendor default = sparse stairs_down), so keep the default there.

    Wand/Horn build a lit 8x8 room (``LevelGenerator(w=8, h=8)``) with an
    RNG-placed blessed freeze item + RNG hero; Random picks the item TYPE via
    ``IF[50%]``; Lava reuses the LavaCross 13x7 lava MAP.  All placement is
    stamped by the vendor-draw-replay wrappers above (void-grid + centered MAP).
    """
    # Wand / Horn (Full + Restricted): 8x8 room, fixed item type.
    for source, item_name, item_key in (
        ("Wand", "cold", "wand"),
        ("Horn", "frost horn", "horn"),
    ):
        for suffix in ("-Full", "-Restricted"):
            env_id = f"MiniHack-Freeze-{source}{suffix}-v0"
            base_factory = _make_factory(
                _skill_room_builder(8, lit=True), w=80, h=21, fill=" ", lit=True
            )
            factory = _wrap_freeze_placement(base_factory, item_name, item_key)
            register_fn(env_id, factory, _skill_freeze_rm(),
                        max_steps=50, category="Skill")

    # Random (Full + Restricted): item TYPE RNG-chosen in an 8x8 room.
    for suffix in ("-Full", "-Restricted"):
        env_id = f"MiniHack-Freeze-Random{suffix}-v0"
        base_factory = _make_factory(
            _skill_room_builder(8, lit=True), w=80, h=21, fill=" ", lit=True
        )
        factory = _wrap_freeze_random(base_factory)
        register_fn(env_id, factory, _skill_freeze_rm(),
                    max_steps=50, category="Skill")

    # Lava (Full + Restricted): LavaCross 13x7 lava MAP; default sparse RM.
    for suffix in ("-Full", "-Restricted"):
        env_id = f"MiniHack-Freeze-Lava{suffix}-v0"
        base_factory = _make_factory(
            _lavacross_builder(with_potion=False, with_ring=False, inv=False),
            w=80, h=21, fill=".",
        )
        factory = _wrap_freeze_lava(base_factory)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=50, category="Skill")


# ---------------------------------------------------------------------------
# ClosedDoor / LockedDoor skill envs
# ---------------------------------------------------------------------------
# Unlike the earlier stub (a tiny add_room at terrain[0,0]), the vendor skill
# door levels are two 5×5 rooms separated by a wall with a single door, stamped
# ``GEOMETRY:center,center`` on the 80×21 dungeon — the same void-grid +
# ``_vendor_geometry_center_wh`` pattern Sokoban / River / CorridorBattle use.
#
#   vendor/minihack/minihack/dat/locked_door_fixed.des
#     MAZE "mylevel",' ' + INIT_MAP:solidfill,' ' + GEOMETRY:center,center
#     MAP  (13×7: | left room | wall+door | right room |)
#     REGION:(0,0,12,6),lit,"ordinary"
#     BRANCH:(3,3,3,3),(0,0,0,0)   -> player fixed at MAP (3,3)
#     DOOR:locked,(6,3)            -> locked door in the shared wall
#     STAIR:(8,3),down             -> down-stair in the right room
#
#   locked_door.des is identical except the player start is RNG-placed in the
#   left room (BRANCH region (1,1)-(5,5)) and the stair is ``rndcoord`` in the
#   right room; both consume ``state.vendor_rng`` (see _wrap_locked_door).
#
# The hero can only see the left room (the closed/locked door blocks LOS), so
# only the left room + door appear in the observation — the RNG-placed stair in
# the (unseen) right room never affects glyph/char byte-parity.
_LOCKED_DOOR_MAP = (
    "-------------",
    "|.....|.....|",
    "|.....|.....|",
    "|.....+.....|",
    "|.....|.....|",
    "|.....|.....|",
    "-------------",
)
_LD_W = 13
_LD_H = 7


def _locked_door_builder(fixed: bool) -> Callable[[LevelGenerator], None]:
    """Stamp the locked-door two-room MAP at its GEOMETRY:center,center origin.

    ``fixed`` mirrors locked_door_fixed.des (player / stair at fixed cells);
    the non-fixed locked_door.des RNG-places the player (BRANCH region) and the
    stair (rndcoord right room) — those are stamped by :func:`_wrap_locked_door`
    on top of the deterministic fallback placed here.
    """
    dx, dy = _vendor_geometry_center_wh(_LD_W, _LD_H)

    def build(lg: LevelGenerator) -> None:
        lg.set_map(_LOCKED_DOOR_MAP, xstart=dx, ystart=dy)
        # DOOR:locked,(6,3) — MAP-relative (col,row); the map's '+' already
        # stamps a CLOSED_DOOR tile, add_door records the locked door_state.
        lg.add_door("locked", place=(6 + dx, 3 + dy))
        # STAIR down in the right room (unseen behind the locked door).  Fixed
        # variant: MAP (8,3); non-fixed fallback also (8,3) — the RNG stair is
        # never visible so its exact cell can't affect byte-parity.
        lg.add_stair_down(x=8 + dx, y=3 + dy)
        # Player start in the left room.  Fixed: BRANCH (3,3).  Non-fixed:
        # fallback (3,3); overwritten by _wrap_locked_door after the RNG draw.
        lg.set_start_pos(3 + dx, 3 + dy)
    return build


def _wrap_locked_door(
    factory: Callable[[jax.Array], "EnvState"],
) -> Callable[[jax.Array], "EnvState"]:
    """RNG-place the player for locked_door.des (non-fixed).

    Vendor mklev draw stream after the shared reset (verified by matching the
    player cell of vendor MiniHack-LockedDoor-v0 seeds 0-4):

      1. rn2(3), rn2(2)                        -- level-setup prefix
      2. rn2(25)                               -- STAIR:rndcoord($right_room)
         (right room = fillrect(7,1,11,5) = 25 selection cells;
         selection_rndcoord draws a single rn2(idx), sp_lev.c:3808)
      3. place_lregion BRANCH:(1,1,5,5):        player cell (first try always
         succeeds — the whole region is left-room FLOOR), mkmaze.c:301-308
             x_internal = rn2(5) + (dx + 1)
             y_internal = rn2(5) + (dy + 1)

    The right-room stair is behind the locked door (never in FOV) so its RNG
    cell can't change the observation; only the draw *consumption* matters and
    is replayed here to keep ``vendor_rng`` byte-aligned with the player draw.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    from Nethax.minihax.level_generator import seed_hero_fov as _seed_hero_fov

    dx, dy = _vendor_geometry_center_wh(_LD_W, _LD_H)

    def wrapped(rng: jax.Array):
        state = factory(rng)
        vrng = state.vendor_rng
        # (1) level-setup prefix.
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        # (2) STAIR:rndcoord($right_room) — one rn2(25) draw.
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(25))
        # (3) BRANCH place_lregion — player cell (accepted first try).
        vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(5))
        px = dx + 1 + int(ox)
        py = dy + 1 + int(oy)
        state = state.replace(
            vendor_rng=vrng,
            player_pos=jnp.stack(
                [jnp.int32(py).astype(jnp.int16),
                 jnp.int32(px).astype(jnp.int16)]
            ),
        )
        return _seed_hero_fov(state, True)

    return wrapped


def _closed_door_builder(lg: LevelGenerator) -> None:
    """Structural ClosedDoor level (closed_door.des).

    NOT byte-exact.  closed_door.des is a fully *random* level built by
    NetHack's create_room engine:

        ROOM: "ordinary", lit, (3,3), (center,center), (8,8) {
            SUBROOM:"ordinary", lit, random, (4,4) {
                STAIR: random, down
                ROOMDOOR: false, closed, random, random
            }
        }

    i.e. an 8×8 outer room (deterministically centered), a 4×4 SUBROOM placed
    at a *random* cell inside it, a *random* down-stair inside the subroom, a
    closed ROOMDOOR on a *random* subroom wall, plus the hero placed at a
    *random* outer-room floor cell.  Byte-parity would require a faithful port
    of create_room + subroom placement + roomdoor + the player somexy draws
    (the create_room ISAAC64 stream — a separate, larger effort, cf. KeyRoom).

    Here we only fix the gross un-centering symptom: the outer 8×8 room is
    stamped at its centered origin (``_vendor_geometry_center(8)`` == internal
    (37,7), matching vendor MiniHack-ClosedDoor observations) with a closed
    door + subroom + stair as a playable stand-in.
    """
    x0, y0 = _vendor_geometry_center(8)          # (37, 7) internal top-left
    x1, y1 = x0 + 7, y0 + 7                       # (44, 14)
    lg.fill_terrain(".", x0, y0, x1, y1)          # 8×8 outer room (VOID walls)
    # 4×4 subroom in the right half: carve its walls, leave a closed door.
    sx0, sy0 = x1 - 3, y0 + 1                     # subroom interior top-left
    lg.fill_terrain("|", sx0 - 1, sy0 - 1, sx0 - 1, sy0 + 2)  # west wall
    lg.add_door("closed", place=(sx0 - 1, sy0 + 1))          # closed door
    lg.add_stair_down(x=sx0 + 1, y=sy0 + 1)      # stair inside subroom
    lg.set_start_pos(x0, y0)                      # hero in outer room


def _register_skill_door_envs(register_fn) -> None:
    """ClosedDoor / LockedDoor envs (vendor skills_simple.py)."""
    closed = _make_factory(_closed_door_builder, w=80, h=21, fill=" ")
    register_fn("MiniHack-ClosedDoor-v0", closed,
                _skill_door_rm(),
                max_steps=50, category="Skill")

    fixed = _make_factory(_locked_door_builder(True), w=80, h=21, fill=" ")
    register_fn("MiniHack-LockedDoor-Fixed-v0", fixed,
                _skill_door_rm(),
                max_steps=50, category="Skill")

    base = _make_factory(_locked_door_builder(False), w=80, h=21, fill=" ")
    register_fn("MiniHack-LockedDoor-v0", _wrap_locked_door(base),
                _skill_door_rm(),
                max_steps=50, category="Skill")


# ---------------------------------------------------------------------------
# ExploreMaze envs (Group A)
# ---------------------------------------------------------------------------
def _exploremaze_builder(hard: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        if hard:
            lg.add_room(x=1, y=1, w=20, h=12)
        else:
            lg.add_room(x=1, y=1, w=12, h=8)
        # Apples scattered for shaping (matches vendor ExploreMaze reward).
        for i in range(3):
            try:
                lg.add_object("apple", "%", place=(2 + i * 2, 2))
            except KeyError:
                lg.add_object("random", place=(2 + i * 2, 2))
        lg.set_start_pos(1, 1)
        lg.add_stair_down(x=10 if not hard else 18, y=6 if not hard else 10)
    return build


def _register_exploremaze_envs(register_fn) -> None:
    # Every ExploreMaze variant ships with a static vendor .des
    # (vendor/minihack/minihack/envs/exploremaze.py:52-70):
    #   Easy           -> exploremazeeasy.des
    #   Easy-Mapped    -> exploremazeeasy_premapped.des
    #   Hard           -> exploremazehard.des
    #   Hard-Mapped    -> exploremazehard_premapped.des
    # All four parse and build via the des_parser; the LG builder remains
    # as a safety fallback if the probe-build raises (see _des_factory).
    variants = [
        ("MiniHack-ExploreMaze-Easy-v0",        False, "exploremazeeasy.des"),
        ("MiniHack-ExploreMaze-Easy-Mapped-v0", False, "exploremazeeasy_premapped.des"),
        ("MiniHack-ExploreMaze-Hard-v0",        True,  "exploremazehard.des"),
        ("MiniHack-ExploreMaze-Hard-Mapped-v0", True,  "exploremazehard_premapped.des"),
    ]
    for env_id, hard, des_name in variants:
        fallback = _make_factory(_exploremaze_builder(hard), w=22, h=14)
        if des_name is not None:
            factory = _des_factory(des_name, fallback=fallback)
        else:
            factory = fallback
        register_fn(env_id, factory, _exploremaze_rm(),
                    max_steps=500, category="ExploreMaze")


# ---------------------------------------------------------------------------
# Top-level registration entry-point
# ---------------------------------------------------------------------------
def register_all() -> None:
    """Populate the global ``MINIHACK_ENV_REGISTRY``."""
    from Nethax.minihax.registry import EnvSpec, register

    def reg(env_id: str,
            factory: Callable[[jax.Array], EnvState],
            reward_manager: RewardManager,
            *,
            max_steps: int,
            category: str) -> None:
        spec = EnvSpec(
            env_id=env_id,
            level_factory=factory,
            reward_manager=reward_manager,
            max_steps=max_steps,
            category=category,
        )
        register(spec)

    _register_room_envs(reg)
    _register_corridor_envs(reg)
    _register_mazewalk_envs(reg)
    _register_hidenseek_envs(reg)
    _register_keyroom_envs(reg)
    _register_lavacross_envs(reg)
    _register_simplecrossing_envs(reg)
    _register_sokoban_envs(reg)
    _register_labyrinth_envs(reg)
    _register_river_envs(reg)
    _register_multiroom_envs(reg)
    _register_quest_envs(reg)
    _register_memento_envs(reg)
    _register_wod_envs(reg)
    _register_boxoban_envs(reg)
    _register_skill_simple_envs(reg)
    _register_skill_levitate_envs(reg)
    _register_skill_freeze_envs(reg)
    _register_skill_door_envs(reg)
    _register_exploremaze_envs(reg)
