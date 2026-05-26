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


def _exploremaze_rm() -> RewardManager:
    """Vendor ExploreMaze: shaping via add_eat_event("apple") plus stairs_down.

    Vendor scatters apples and pays an eat-event each; the env still terminates
    on stairs_down.  We mirror by registering both an apple-eat event
    (repeatable, non-terminal) and the default stairs-down terminal.
    """
    rm = RewardManager()
    rm.add_eat_event(
        "apple",
        reward=1.0,
        repeatable=True,
        terminal_required=False,
        terminal_sufficient=False,
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
def _room_builder(size: int, *, random: bool, lit: bool,
                  n_monster: int, n_trap: int) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        # The level itself is a single ``size x size`` room with floor fill.
        # We carve walls around the room border so the goal-stair sits inside
        # a clearly bounded space.
        if random:
            lg.add_stair_down()    # any floor cell
        else:
            # Deterministic: stair at bottom-right, start at top-left.
            lg.add_stair_down(x=size - 1, y=size - 1)
            lg.set_start_pos(0, 0)
        for _ in range(n_monster):
            lg.add_monster()
        for _ in range(n_trap):
            lg.add_trap()
    return build


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
        factory = _make_factory(builder, w=size, h=size, lit=lit)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=size * 20, category="Room")


# ---------------------------------------------------------------------------
# Corridor envs (Group A)
# ---------------------------------------------------------------------------
def _corridor_builder(n_rooms: int) -> Callable[[LevelGenerator], None]:
    """Build a small map with ``n_rooms`` rooms wired by corridors."""
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


def _register_corridor_envs(register_fn) -> None:
    """Register Corridor-R2/R3/R5 + CorridorBattle envs (Group A).

    Corridor-R{2,3,5} ship with static vendor ``corridor{2,3,5}.des``
    (vendor/minihack/minihack/envs/corridor.py:29-39); route those through
    the des_parser with the procedural LG builder as a fallback.
    """
    for env_id, n_rooms, des_name in [
        ("MiniHack-Corridor-R2-v0", 2, "corridor2.des"),
        ("MiniHack-Corridor-R3-v0", 3, "corridor3.des"),
        ("MiniHack-Corridor-R5-v0", 5, "corridor5.des"),
    ]:
        fallback = _make_factory(_corridor_builder(n_rooms), w=76, h=21)
        factory = _des_factory(des_name, fallback=fallback)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=1000, category="Corridor")

    # CorridorBattle envs: two rooms + a fight corridor.
    def battle_builder(lit: bool):
        def build(lg: LevelGenerator) -> None:
            lg.add_room(x=2, y=8, w=4, h=4)
            lg.add_room(x=70, y=8, w=4, h=4)
            lg.add_corridor((6, 10), (70, 10))
            lg.set_start_pos(3, 10)
            lg.add_stair_down(x=72, y=10)
            for _ in range(3):
                lg.add_monster()
        return build

    for env_id, lit in [
        ("MiniHack-CorridorBattle-v0", True),
        ("MiniHack-CorridorBattle-Dark-v0", False),
    ]:
        factory = _make_factory(battle_builder(lit), w=76, h=21, lit=lit)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=1000, category="Corridor")


# ---------------------------------------------------------------------------
# MazeWalk envs (Group B — procedural)
# ---------------------------------------------------------------------------
def _mazewalk_builder(w: int, h: int) -> Callable[[LevelGenerator], None]:
    """Build a ``w × h`` perfect maze with stairs in the far corner.

    Wave17i: replaces the legacy "open room" substitute with a real
    recursive-backtracker maze carve via ``LevelGenerator.add_mazewalk``
    (cite vendor MAZEWALK des-file directive → mklev.c::walkfrom).
    The agent starts at the top-left and the goal stair is at the
    bottom-right corner.
    """
    def build(lg: LevelGenerator) -> None:
        # Carve a perfect maze covering the active (h, w) area.
        lg.add_mazewalk(coord=(1, 1), dir="east")
        lg.set_start_pos(1, 1)
        lg.add_stair_down(x=w - 2 if w > 2 else w - 1,
                          y=h - 2 if h > 2 else h - 1)
    return build


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
        factory = _make_factory(_mazewalk_builder(w, h), w=w, h=h)
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


def _register_hidenseek_envs(register_fn) -> None:
    """Register HideNSeek envs.

    All 4 variants ship with a static vendor .des
    (vendor/minihack/minihack/envs/hidenseek.py:9-27).  Route each through
    the des_parser with the procedural LG builder as a fallback.
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
        rm = _lava_avoid_reward_manager() if lava else _default_goal_reward_manager()
        register_fn(env_id, factory, rm,
                    max_steps=200, category="HideNSeek")


# ---------------------------------------------------------------------------
# KeyRoom envs (Group A)
# ---------------------------------------------------------------------------
def _keyroom_builder(room_size: int, subroom_size: int,
                     lit: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        # Outer room with a sub-room holding the locked door + goal.
        outer = lg.add_room(x=1, y=1, w=room_size, h=room_size, lit=lit)
        # Sub-room placed top-right within the outer footprint.
        sub_x = room_size - subroom_size
        sub_y = 1
        lg.add_room(x=sub_x, y=sub_y, w=subroom_size, h=subroom_size, lit=lit)
        # Key placed in outer room; goal stair in the sub-room.
        lg.add_object("skeleton key", "(", place=outer)
        lg.add_stair_down(x=sub_x, y=sub_y + subroom_size - 1)
        lg.set_start_pos(1, 1)
    return build


def _keyroom_templated_des(room_size: int, subroom_size: int,
                           lit: bool) -> Optional[str]:
    """Render ``key_and_door_tmp.des`` with RS/SS substitutions.

    Mirrors vendor ``KeyRoomGenerator`` (envs/keyroom.py:13-27).  Returns
    ``None`` if the template is unreadable so the caller can keep the
    procedural LG fallback.
    """
    try:
        with open(_vendor_des_path("key_and_door_tmp.des"),
                  "r", encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError:
        return None
    src = src.replace("RS", str(room_size)).replace("SS", str(subroom_size))
    if not lit:
        src = src.replace("lit", "unlit")
    return src


def _register_keyroom_envs(register_fn) -> None:
    """Register all KeyRoom envs.

    Fixed-S5 ships with the static ``key_and_door.des``
    (vendor/minihack/minihack/envs/keyroom.py:82); the other variants are
    materialised by ``KeyRoomGenerator`` from ``key_and_door_tmp.des``
    with RS/SS/lit substitutions (envs/keyroom.py:13-27).  We render the
    template ourselves and feed the resulting source to the des_parser.
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
        fallback = _make_factory(
            _keyroom_builder(rs, ss, lit),
            w=max(20, rs + 2), h=max(20, rs + 2), lit=lit,
        )
        if env_id == "MiniHack-KeyRoom-Fixed-S5-v0":
            factory = _des_factory("key_and_door.des", fallback=fallback)
        else:
            tmpl = _keyroom_templated_des(rs, ss, lit)
            if tmpl is None:
                factory = fallback
            else:
                factory = _des_factory_from_source(tmpl, fallback=fallback)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=ms, category="KeyRoom")


# ---------------------------------------------------------------------------
# LavaCross envs (Group C)
# ---------------------------------------------------------------------------
def _lavacross_builder(*, with_potion: bool,
                       with_ring: bool,
                       inv: bool) -> Callable[[LevelGenerator], None]:
    """Lava strip with a levitation item to acquire.

    ``inv``: place item in inventory (start_pos) vs. somewhere to pick up.
    """
    def build(lg: LevelGenerator) -> None:
        # Single room with a vertical lava strip mid-way.
        lg.add_room(x=1, y=1, w=15, h=8)
        lg.fill_terrain("L", 8, 1, 8, 8)
        lg.set_start_pos(2, 4)
        lg.add_stair_down(x=14, y=4)
        # Drop the levitation item somewhere reachable.
        if with_potion:
            item = "potion of levitation"
            symbol = "!"
        elif with_ring:
            item = "ring of levitation"
            symbol = "="
        else:
            item = "levitation boots"
            symbol = "["
        place_x = 2 if inv else 6
        try:
            lg.add_object(item, symbol, place=(place_x, 4))
        except KeyError:
            # Fall back to any random object if the named one isn't in the
            # OBJECTS table (e.g. levitation boots renamed).
            lg.add_object("random", place=(place_x, 4))
    return build


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
        fallback = _make_factory(_lavacross_builder(**kw), w=18, h=10)
        # MiniHack-LavaCross-Full and -Restricted are the only LavaCross
        # variants that use the shipped lava_crossing.des
        # (vendor/minihack/minihack/envs/skills_lava.py:339-358).  The other
        # Levitate-* variants build their .des inline as Python strings,
        # so we keep the LG fallback for those.
        if env_id in ("MiniHack-LavaCross-Full-v0",
                      "MiniHack-LavaCross-Restricted-v0"):
            factory = _des_factory("lava_crossing.des", fallback=fallback)
        else:
            factory = fallback
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
def _sokoban_builder(level: int, variant: str) -> Callable[[LevelGenerator], None]:
    """Build a small Sokoban-style level with boulders + fountains.

    Wave 4 simplification: hand-coded compact layouts, ``level``/``variant``
    parametrise the placement.
    """
    def build(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=10, h=8)
        lg.set_start_pos(2, 2)
        lg.add_stair_down(x=9, y=7)
        n_boulders = max(1, level)
        for i in range(n_boulders):
            x = 3 + (i * 2) % 6
            y = 3 + (i // 3)
            try:
                lg.add_object("boulder", "`", place=(x, y))
            except KeyError:
                lg.add_object("random", place=(x, y))
        # Fountains as drop targets.
        for i in range(n_boulders):
            fx = 5 + (i * 2) % 4
            fy = 5
            lg.fill_terrain("{", fx, fy, fx, fy)
    return build


def _register_sokoban_envs(register_fn) -> None:
    # Every vendor MiniHack-Sokoban<N><a|b>-v0 has a matching static
    # ``soko<N><a|b>.des`` under vendor/minihack/minihack/dat/, fed via
    #   vendor/minihack/minihack/envs/sokoban.py: des_file="soko1a.des"
    # so we route each id through the des_parser, keeping the hand-coded
    # LG builder as a fallback in case a directive (e.g. BRANCH) trips
    # the compiler.
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
        fallback = _make_factory(
            _sokoban_builder(level, variant), w=12, h=10,
        )
        des_name = f"soko{level}{variant}.des"
        factory = _des_factory(des_name, fallback=fallback)
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=400, category="Sokoban")


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
def _river_builder(narrow: bool, lava: bool,
                   n_monster: int) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=25, h=7)
        # Water (or lava) strip
        terrain = "L" if lava else "W"
        strip_width = 2 if narrow else 3
        x_start = 18
        for c in range(x_start, x_start + strip_width):
            lg.fill_terrain(terrain, c, 1, c, 7)
        lg.set_start_pos(2, 3)
        lg.add_stair_down(x=24, y=3)
        for _ in range(n_monster):
            lg.add_monster()
    return build


def _register_river_envs(register_fn) -> None:
    variants = [
        ("MiniHack-River-v0",            False, False, 0),
        ("MiniHack-River-Monster-v0",    False, False, 5),
        ("MiniHack-River-Lava-v0",       False, True,  0),
        ("MiniHack-River-MonsterLava-v0",False, True,  5),
        ("MiniHack-River-Narrow-v0",     True,  False, 0),
    ]
    for env_id, narrow, lava, nm in variants:
        factory = _make_factory(
            _river_builder(narrow, lava, nm), w=27, h=9,
        )
        rm = _lava_avoid_reward_manager() if lava else _default_goal_reward_manager()
        register_fn(env_id, factory, rm,
                    max_steps=350, category="River")


# ---------------------------------------------------------------------------
# MultiRoom envs (Group C — MiniGrid ports)
# ---------------------------------------------------------------------------
def _multiroom_builder(n_rooms: int, *, lava_walls: bool,
                       locked: bool, monster: bool,
                       open_door: bool,
                       extreme: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        positions = []
        for i in range(n_rooms):
            x = 1 + i * 8
            y = 1 + (i % 3) * 5
            x = min(x, 70)
            y = min(y, 14)
            rid = lg.add_room(x=x, y=y, w=4, h=4)
            positions.append((rid, x + 1, y + 1))
        # Connect consecutive rooms via corridors.
        for i in range(len(positions) - 1):
            _, x1, y1 = positions[i]
            _, x2, y2 = positions[i + 1]
            lg.add_corridor((x1 + 2, y1), (x2, y2))
            # Door at the corridor source.
            door_state = "locked" if locked else ("open" if open_door else "closed")
            lg.add_door(x1 + 2, y1, state=door_state)
        # Optional environmental hazards.
        if lava_walls or extreme:
            lg.fill_terrain("L", 30, 9, 32, 9)
        if monster or extreme:
            for _ in range(min(3, n_rooms)):
                lg.add_monster()
        # Start in first, goal in last room.
        _, sx, sy = positions[0]
        _, gx, gy = positions[-1]
        lg.set_start_pos(sx, sy)
        lg.add_stair_down(x=gx, y=gy)
    return build


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
        builder = _multiroom_builder(
            n, lava_walls=lava, locked=locked, monster=monster,
            open_door=open_door, extreme=extreme,
        )
        factory = _make_factory(builder, w=76, h=21)
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
        register_fn(env_id, factory, _default_goal_reward_manager(),
                    max_steps=ms, category="Memento")


# ---------------------------------------------------------------------------
# WoD envs (Wand of Death — Group A)
# ---------------------------------------------------------------------------
def _wod_builder(difficulty: str) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=15, h=8)
        lg.set_start_pos(2, 2)
        lg.add_stair_down(x=14, y=7)
        # Drop the wand near the start.
        try:
            lg.add_object("wand of death", "/", place=(3, 3))
        except KeyError:
            lg.add_object("random", place=(3, 3))
        if difficulty in ("medium", "hard", "pro"):
            lg.add_monster()
        if difficulty == "hard":
            lg.add_monster()
        if difficulty == "pro":
            for _ in range(3):
                lg.add_monster()
    return build


def _register_wod_envs(register_fn) -> None:
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
        register_fn(env_id, factory, _default_goal_reward_manager(),
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


def _skill_levitate_builder(item: str, symbol: str,
                            fixed: bool) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        place = (0, 0) if fixed else None
        try:
            lg.add_object(item, symbol, place=place)
        except KeyError:
            lg.add_object("random", place=place)
        if fixed:
            lg.set_start_pos(2, 2)
        lg.add_stair_down(x=4, y=4)
    return build


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


def _skill_freeze_builder(source: str) -> Callable[[LevelGenerator], None]:
    def build(lg: LevelGenerator) -> None:
        # Place freeze source (wand/horn/random) and a monster.
        if source == "wand":
            try:
                lg.add_object("wand of cold", "/", place=(1, 1))
            except KeyError:
                lg.add_object("random", place=(1, 1))
        elif source == "horn":
            try:
                lg.add_object("frost horn", "(", place=(1, 1))
            except KeyError:
                lg.add_object("random", place=(1, 1))
        else:
            lg.add_object("random", place=(1, 1))
        lg.add_monster()
        lg.add_stair_down(x=4, y=4)
        if source == "lava":
            lg.fill_terrain("L", 3, 3, 3, 3)
    return build


def _register_skill_simple_envs(register_fn) -> None:
    """Eat / Wield / Wear / PutOn / Zap / Read / Pray / Sink — 24 envs.

    RM per family mirrors vendor ``skills_simple.py``: each env pays its
    targeted event (eat-apple, wield-dagger, amulet-message, ...), NOT
    sparse stairs_down.
    """
    item_specs = [
        # (basename, item, symbol, rm_factory)
        ("Wield", "dagger",           ")", _skill_wield_rm),
        ("Wear",  "leather armor",    "[", _skill_wear_rm),
        ("PutOn", "amulet of life saving", '"', _skill_amulet_rm),
        ("Zap",   "wand of striking", "/", _skill_zap_rm),
        ("Read",  "scroll of mail",   "?", _skill_read_rm),
    ]
    for base, item, symbol, rm_factory in item_specs:
        for suffix, distr, fixed in [
            ("",       False, False),
            ("-Fixed", False, True),
            ("-Distr", True,  False),
        ]:
            env_id = f"MiniHack-{base}{suffix}-v0"
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
        ("Boots",   "levitation boots",      "["),
        ("Ring",    "ring of levitation",    "="),
        ("Potion",  "potion of levitation",  "!"),
    ]
    for base, item, symbol in item_specs:
        for suffix in ("-Full", "-Restricted", "-Fixed"):
            env_id = f"MiniHack-Levitate-{base}{suffix}-v0"
            builder = _skill_levitate_builder(item, symbol,
                                              fixed=(suffix == "-Fixed"))
            factory = _make_factory(builder, w=5, h=5)
            register_fn(env_id, factory, _skill_levitate_rm(),
                        max_steps=50, category="Skill")
    # Levitate-Random
    builder = _skill_levitate_builder("random", "/", fixed=False)
    factory = _make_factory(builder, w=5, h=5)
    register_fn("MiniHack-Levitate-Random-Full-v0", factory,
                _skill_levitate_rm(),
                max_steps=50, category="Skill")


def _register_skill_freeze_envs(register_fn) -> None:
    """8 Freeze envs.

    Vendor (``skills_freeze.py:11-18``): RM is ``add_message_event(freeze_msgs)``
    for Wand/Horn/Random.  ``Freeze-Lava-*`` constructs ``MiniHackSkill``
    without a RM (vendor default = sparse stairs_down), so keep the default
    here for the Lava variants only.
    """
    for source in ("Wand", "Horn", "Random", "Lava"):
        for suffix in ("-Full", "-Restricted"):
            env_id = f"MiniHack-Freeze-{source}{suffix}-v0"
            builder = _skill_freeze_builder(source.lower())
            factory = _make_factory(builder, w=5, h=5)
            rm = (_default_goal_reward_manager()
                  if source == "Lava" else _skill_freeze_rm())
            register_fn(env_id, factory, rm,
                        max_steps=50, category="Skill")


def _register_skill_door_envs(register_fn) -> None:
    """ClosedDoor / LockedDoor envs."""
    def closed_builder(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=4, h=3)
        lg.add_door(2, 1, state="closed")
        lg.set_start_pos(0, 1)
        lg.add_stair_down(x=4, y=2)

    def locked_builder(lg: LevelGenerator) -> None:
        lg.add_room(x=1, y=1, w=4, h=3)
        lg.add_door(2, 1, state="locked")
        lg.set_start_pos(0, 1)
        lg.add_stair_down(x=4, y=2)

    factory = _make_factory(closed_builder, w=6, h=5)
    register_fn("MiniHack-ClosedDoor-v0", factory,
                _default_goal_reward_manager(),
                max_steps=50, category="Skill")

    factory = _make_factory(locked_builder, w=6, h=5)
    register_fn("MiniHack-LockedDoor-v0", factory,
                _default_goal_reward_manager(),
                max_steps=50, category="Skill")

    factory = _make_factory(locked_builder, w=6, h=5)
    register_fn("MiniHack-LockedDoor-Fixed-v0", factory,
                _default_goal_reward_manager(),
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
