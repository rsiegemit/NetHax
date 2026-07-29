"""MiniHack-compatible ``LevelGenerator`` API ported to nethax.

This module provides the Python builder API used by MiniHack at environment
construction time to author levels.  It mirrors the public surface of
``vendor/minihack/minihack/level_generator.py`` so existing MiniHack level
scripts can be ported with minimal edits, while emitting a JAX ``EnvState``
suitable for the nethax engine.

Coordinate conventions
----------------------
MiniHack uses ``(x, y)`` = (column, row); nethax uses ``(row, col)``.  The
public API of this module accepts MiniHack ``(x, y)`` arguments to match the
vendor API; the factory converts to nethax row/col when writing into JAX
arrays.

Status
------
Wave 4 Phase 1, agent A1 deliverable.  Implements the builder + factory
without modifying ``EnvState`` schema.  Goal positions are recorded as
``STAIRCASE_DOWN`` tiles (consistent with MiniHack's
``add_stair_down``/``add_goal_pos`` aliasing).
"""
from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from Nethax.nethax.constants.monsters import MONSTERS
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass, OBJECT_NAME_ALIASES
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.features import DoorState
from Nethax.nethax.dungeon.spawning import (
    _ATK_DICE_N,
    _ATK_DICE_S,
    _BASE_AC,
    _IS_LARGE,
)
from Nethax.nethax.state import EnvState, StaticParams
from Nethax.nethax.subsystems.inventory import MAX_GROUND_STACK
from Nethax.nethax.subsystems.traps import TrapType


# ---------------------------------------------------------------------------
# Per-env player-role bootstrap override
# ---------------------------------------------------------------------------
# The NLE_BYTEPARITY reset bootstrap in ``_apply_directives`` routes through
# ``NethaxEnv.reset`` to advance the ISAAC64 stream (init_objects -> role_init
# -> u_init).  The canonical MiniHack character is Archeologist-Human-Lawful
# ("arc-hum-law-mal") and that is the default.  A handful of MiniHack envs
# hardcode a *different* player role in their vendor class (e.g. Quest-Medium
# and CorridorBattle use "kni-hum-law-fem"; the KeyRoom family uses
# "rog-hum-cha-mal").  For those envs both the vendor build (byte-parity
# harness) and the minihax bootstrap must use the env's true role, otherwise
# the hero @-glyph, inventory, and role_init RNG diverge.
#
# ``(role, race, alignment)`` — role/race are ``Role``/``Race`` enum values,
# alignment is 0=lawful / 1=neutral / 2=chaotic.  ``None`` means "use the
# Archeologist default", so Room / skills / Levitate envs are unaffected.
# The registry (canonical.register_all) wraps the factories of the named
# envs in ``bootstrap_character(...)`` so the override is active only while
# that env's factory runs.
_BOOTSTRAP_CHARACTER: Optional[Tuple[Any, Any, int]] = None


@contextlib.contextmanager
def bootstrap_character(role: Any, race: Any, alignment: int, gender: int = 0):
    """Temporarily override the NLE_BYTEPARITY reset bootstrap character.

    ``role``/``race`` are ``Role``/``Race`` enum values; ``alignment`` is
    0=lawful, 1=neutral, 2=chaotic; ``gender`` is 0=male, 1=female (only
    affects the display @-glyph for Caveman/Priest).  Restores the previous
    value on exit so the default (Archeologist) applies to every other env.
    """
    global _BOOTSTRAP_CHARACTER
    prev = _BOOTSTRAP_CHARACTER
    _BOOTSTRAP_CHARACTER = (role, race, int(alignment), int(gender))
    try:
        yield
    finally:
        _BOOTSTRAP_CHARACTER = prev


# ---------------------------------------------------------------------------
# Public maps mirroring MiniHack's vendor API
# ---------------------------------------------------------------------------

#: MiniHack terrain-character to nethax ``TileType``.
#: Source: ``vendor/minihack/minihack/level_generator.py`` ``MAP_CHARS``.
TERRAIN_CHAR_TO_TILE: dict = {
    " ": TileType.VOID,
    "#": TileType.CORRIDOR,
    ".": TileType.FLOOR,
    # A des MAP authors wall orientation with the '-' / '|' char.  Vendor
    # fix_wall_spines KEEPS this authored typ (HWALL / VWALL) when the wall has
    # no orthogonal wall neighbours (spine bitmask == 0), and OVERRIDES it with
    # a corner / T / cross variant when it does.  We stamp the authored typ here
    # so a free-standing des wall (e.g. Sokoban interior stub, ExploreMaze pillar)
    # renders S_hwall / S_vwall byte-exact; the render-time spine pass
    # (nle_obs::_apply_wall_angle) treats HWALL / VWALL as ordinary WALL
    # continuations and re-derives the variant whenever bits != 0, so walls in a
    # run are unaffected.  Procedurally-generated Room walls are stamped as the
    # generic ``WALL`` directly (branches.py / _carve_room), NOT through this
    # map, so they keep deriving their variant purely from the spine pass.
    "-": TileType.HWALL,
    "|": TileType.VWALL,
    # A bare '+' in a MAP...ENDMAP block is a DOOR cell whose doormask is
    # cleared to 0 == D_NODOOR by the map loader (vendor sp_lev.c:5010 sets
    # levl[][].flags = 0, and `doormask` is a #define alias of `flags`; only
    # SDOOR gets forced to D_CLOSED at :5021).  A D_NODOOR door renders as the
    # doorless doorway S_ndoor (glyph 2371), NOT a closed door.  An explicit
    # DOOR:state,(x,y) directive still overrides this cell to the requested
    # CLOSED/OPEN door tile via `_place_door`.
    "+": TileType.DOORWAY,
    "}": TileType.WATER,
    "P": TileType.WATER,
    "W": TileType.WATER,
    "L": TileType.LAVA,
    "{": TileType.FOUNTAIN,
    # 'F' is the vendor MAP mapchar for IRONBARS ("Fe = iron"; cite
    # vendor/nethack/src/nhlua.c:374 char2typ table), rendered as S_bars ('#',
    # glyph 2376).  TileType.IRONBARS + the nle_obs _TILE_TO_CMAP entry now
    # produce that exact S_bars glyph (previously mapped to CORRIDOR, which
    # shares the '#' char but renders S_corr / glyph 2380).
    # (The MONSTER 'F' lichen glyph is a separate directive, not a MAP char.)
    "F": TileType.IRONBARS,
    "\\": TileType.THRONE,
    "<": TileType.STAIRCASE_UP,
    ">": TileType.STAIRCASE_DOWN,
    # HideNSeek line-of-sight overlays.  Canonical TileType has no walkable
    # CLOUD tile, so we map both vendor glyphs to TREE: TREE is walkable AND
    # opaque (vendor/nethack vision.c:166-169) which matches CLOUD's role as
    # a hide-mechanic occluder.  vendor des: hidenseek*.des REPLACE_TERRAIN.
    "T": TileType.TREE,
    "C": TileType.TREE,
}

#: MiniHack trap-name to nethax ``TrapType``.
#: Source: ``vendor/minihack/minihack/level_generator.py`` ``TRAP_NAMES``.
TRAP_NAME_TO_TYPE: dict = {
    "anti magic":      TrapType.ANTI_MAGIC,
    "arrow":           TrapType.ARROW_TRAP,
    "bear":            TrapType.BEAR_TRAP,
    "board":           TrapType.SQKY_BOARD,
    "dart":            TrapType.DART_TRAP,
    "falling rock":    TrapType.ROCKTRAP,
    "fire":            TrapType.FIRE_TRAP,
    "hole":            TrapType.HOLE,
    "land mine":       TrapType.LANDMINE,
    "level teleport":  TrapType.LEVEL_TELEP,
    "magic portal":    TrapType.MAGIC_PORTAL,
    "magic":           TrapType.MAGIC_TRAP,
    "pit":             TrapType.PIT,
    "polymorph":       TrapType.POLY_TRAP,
    "rolling boulder": TrapType.ROLLING_BOULDER_TRAP,
    "rust":            TrapType.RUST_TRAP,
    "sleep gas":       TrapType.SLP_GAS_TRAP,
    "spiked pit":      TrapType.SPIKED_PIT,
    "statue":          TrapType.STATUE_TRAP,
    "teleport":        TrapType.TELEP_TRAP,
    "trap door":       TrapType.TRAPDOOR,
    "web":             TrapType.WEB,
}


#: MiniHack door-state string to nethax ``DoorState`` (vendor rm.h doormask).
#: ``random`` is treated as ``closed`` here (deterministic) — the LG does not
#: roll door states.  ``nodoor`` leaves the doorway as floor (state GONE).
_DOOR_STATE_VALUE: dict = {
    "open":   int(DoorState.OPEN),
    "closed": int(DoorState.CLOSED),
    "locked": int(DoorState.LOCKED),
    "random": int(DoorState.CLOSED),
    "nodoor": int(DoorState.GONE),
}


# ---------------------------------------------------------------------------
# Name → table-index lookups (one-time at import)
# ---------------------------------------------------------------------------

def _build_monster_name_lookup() -> dict:
    table = {}
    for idx, entry in enumerate(MONSTERS):
        table.setdefault(entry.name, idx)
    return table


def _build_object_name_lookup() -> dict:
    """Map MiniHack-style object names to OBJECTS indices.

    Wave 6 parity-fix (CA #63): OBJECTS regenerated from vendor objects.c
    contains anonymous separator rows (``name is None``).  Skip them and
    merge ``OBJECT_NAME_ALIASES`` so MiniHack scripts can still ask for
    "potion of levitation" (now stored bare as "levitation" + alias).
    Cite: vendor/nethack/src/objects.c — bare canonical names per class.
    """
    table: dict = {}
    for idx, entry in enumerate(OBJECTS):
        if entry.name is None:
            continue
        table.setdefault(entry.name, idx)
    # Merge "<prefix> <name>" aliases (e.g. "potion of levitation" -> 248).
    for alias, idx in OBJECT_NAME_ALIASES.items():
        table.setdefault(alias, idx)
    return table


_MONSTER_NAME_TO_IDX: dict = _build_monster_name_lookup()
_OBJECT_NAME_TO_IDX: dict = _build_object_name_lookup()


def _build_monster_group_flags():
    """Per-monster (G_SGROUP, G_LGROUP) booleans for m_initgrp triggering.

    Vendor makemon.c:1370-1377 — a freshly-made monster with G_SGROUP (or
    G_LGROUP) spawns a same-type group via m_initgrp.  Cite monst.c geno
    flags; mirrored on ``MonsterEntry.generation_mask``.
    """
    import numpy as _np
    from Nethax.nethax.constants.monsters import MONSTERS, G_SGROUP, G_LGROUP
    n = len(MONSTERS)
    sg = _np.zeros(n, dtype=bool)
    lg = _np.zeros(n, dtype=bool)
    for i, m in enumerate(MONSTERS):
        gm = int(m.generation_mask)
        sg[i] = bool(gm & G_SGROUP)
        lg[i] = bool(gm & G_LGROUP)
    return sg, lg


_MON_SGROUP, _MON_LGROUP = _build_monster_group_flags()


def _build_monster_armed():
    """Per-monster ``is_armed`` boolean (vendor mondata: has an AT_WEAP attack).

    Vendor makemon.c:1442 only calls ``m_initweap()`` when ``is_armed(ptr)``.
    ``is_armed`` == the monster has any attack of type AT_WEAP.  Mirrored on
    ``MonsterEntry.attacks``.
    """
    import numpy as _np
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    n = len(MONSTERS)
    arr = _np.zeros(n, dtype=bool)
    for i, m in enumerate(MONSTERS):
        arr[i] = any(a[0] == AttackType.AT_WEAP for a in m.attacks)
    return arr


_MON_ARMED = _build_monster_armed()

# Object / monster type indices used by the m_initweap mlet switch, resolved
# from the OBJECTS / MONSTERS tables by name at import (never hardcoded map
# positions).  ``.get`` so a missing entry yields None rather than KeyError.
_OBJ_BY_NAME = {ob.name: i for i, ob in enumerate(OBJECTS)}
_MON_BY_NAME = {m.name: i for i, m in enumerate(MONSTERS)}
_PM_GOBLIN = _MON_BY_NAME.get("goblin")
_PM_ORC_SHAMAN = _MON_BY_NAME.get("orc shaman")
_PM_ORC_CAPTAIN = _MON_BY_NAME.get("orc-captain")
# Armor types whose mksobj() cursed-branch short-circuits the rn2(11) draw
# (vendor mkobj.c:1087-1090); none are granted by depth-1 monsters but the set
# keeps _mksobj_armor_draws faithful for the general case.
_SPECIAL_CURSE_ARMOR = frozenset(
    i for i in (
        _OBJ_BY_NAME.get("fumble boots"),
        _OBJ_BY_NAME.get("levitation boots"),
        _OBJ_BY_NAME.get("helm of opposite alignment"),
        _OBJ_BY_NAME.get("gauntlets of fumbling"),
    ) if i is not None
)


def _rn2(vrng, x):
    """One vendor ``rn2(x)`` draw off the ISAAC64 stream -> (vrng, int result)."""
    from Nethax.nethax import vendor_rng as _vr
    vrng, v = _vr.rn2_jax(vrng, jnp.int32(x))
    return vrng, int(v)


def _rne_draws(vrng, x=3):
    """Vendor ``rne(x)`` (rnd.c:208) at u.ulevel==1 (utmp==5): draw rn2(x)
    up to 4 times, stopping early on the first non-zero result."""
    utmp = 5  # u.ulevel == 1 at mklev  ->  (ulevel<15) ? 5 : ...
    tmp = 1
    while tmp < utmp:
        vrng, r = _rn2(vrng, x)
        if r != 0:
            break
        tmp += 1
    return vrng


def _blessorcurse_draws(vrng, chance):
    """Vendor ``blessorcurse(otmp, chance)`` (mkobj.c): draw rn2(chance); when
    it is 0, draw an extra rn2(2) to choose curse vs bless."""
    vrng, r = _rn2(vrng, chance)
    if r == 0:
        vrng, _ = _rn2(vrng, 2)
    return vrng


def _is_multigen(otyp):
    """Vendor ``is_multigen`` (obj.h): WEAPON_CLASS ammo (oc_skill in
    [-P_SHURIKEN, -P_BOW] == [-24, -20]) -> stacks via rn1(6,6)."""
    ob = OBJECTS[otyp]
    return int(ob.class_) == int(ObjectClass.WEAPON_CLASS) and -24 <= int(ob.oc_skill) <= -20


def _is_poisonable(otyp):
    """Vendor ``is_poisonable`` (obj.h): same launcher-ammo range as multigen
    (permapoisoned specials are ignored — none are granted at depth 1)."""
    return _is_multigen(otyp)


def _mksobj_draws(vrng, otyp):
    """Vendor ``mksobj(otyp, TRUE, FALSE)`` -> ``mksobj_init`` object-creation
    RNG for the WEAPON/ARMOR classes a monster can be granted (mkobj.c:876,
    1085).  ``artif`` is FALSE (mongets/m_initthrow), so the artifact rn2 is
    never drawn.  Only the draw-count/moduli matter for stream alignment."""
    cls = int(OBJECTS[otyp].class_)
    if cls == int(ObjectClass.WEAPON_CLASS):
        if _is_multigen(otyp):
            vrng, _ = _rn2(vrng, 6)          # quan = rn1(6, 6)
        vrng, r11 = _rn2(vrng, 11)
        if r11 == 0:
            vrng = _rne_draws(vrng, 3)       # spe = rne(3)
            vrng, _ = _rn2(vrng, 2)          # blessed = rn2(2)
        else:
            vrng, r10 = _rn2(vrng, 10)
            if r10 == 0:
                vrng = _rne_draws(vrng, 3)   # spe = -rne(3)
            else:
                vrng = _blessorcurse_draws(vrng, 10)
        if _is_poisonable(otyp):
            vrng, _ = _rn2(vrng, 100)        # opoisoned = !rn2(100)
        return vrng
    if cls == int(ObjectClass.ARMOR_CLASS):
        vrng, a = _rn2(vrng, 10)
        cond_curse = False
        if a != 0:
            if otyp in _SPECIAL_CURSE_ARMOR:
                cond_curse = True            # || short-circuits the rn2(11)
            else:
                vrng, b = _rn2(vrng, 11)
                cond_curse = (b == 0)
        if cond_curse:
            vrng = _rne_draws(vrng, 3)       # spe = -rne(3)
        else:
            vrng, c = _rn2(vrng, 10)
            if c == 0:
                vrng, _ = _rn2(vrng, 2)      # blessed = rn2(2)
                vrng = _rne_draws(vrng, 3)   # spe = rne(3)
            else:
                vrng = _blessorcurse_draws(vrng, 10)
        return vrng
    # Other object classes are not granted by the depth-1 mlet cases handled
    # below; leave the stream untouched.
    return vrng


def _m_initthrow_draws(vrng, otyp, oquan):
    """Vendor ``m_initthrow(otyp, oquan)`` (makemon.c:148): mksobj(otyp) then
    ``quan = rn1(oquan, 3)`` (one rn2(oquan) draw)."""
    if otyp is None:
        return vrng
    vrng = _mksobj_draws(vrng, otyp)
    vrng, _ = _rn2(vrng, oquan)
    return vrng


def _m_initweap_default_draws(vrng, idx):
    """Vendor ``m_initweap`` default case (makemon.c:526-567): a bias-weighted
    ``rnd(14 - 2*bias)`` roll selecting one weapon grant."""
    m = MONSTERS[idx]
    f2 = int(m.flags2)
    from Nethax.nethax.constants.monsters import (
        M2_LORD, M2_PRINCE, M2_NASTY, M2_STRONG,
    )
    bias = (1 if f2 & M2_LORD else 0) + (2 if f2 & M2_PRINCE else 0) \
        + (1 if f2 & M2_NASTY else 0)
    strong = bool(f2 & M2_STRONG)
    vrng, w = _rn2(vrng, 14 - 2 * bias)      # rnd(14 - 2*bias)
    case = w + 1
    O = _OBJ_BY_NAME.get
    if case == 1:
        vrng = _mksobj_draws(vrng, O("battle-axe")) if strong \
            else _m_initthrow_draws(vrng, O("dart"), 12)
    elif case == 2:
        if strong:
            vrng = _mksobj_draws(vrng, O("two-handed sword"))
        else:
            vrng = _mksobj_draws(vrng, O("crossbow"))
            vrng = _m_initthrow_draws(vrng, O("crossbow bolt"), 12)
    elif case == 3:
        vrng = _mksobj_draws(vrng, O("bow"))
        vrng = _m_initthrow_draws(vrng, O("arrow"), 12)
    elif case == 4:
        vrng = _mksobj_draws(vrng, O("long sword")) if strong \
            else _m_initthrow_draws(vrng, O("dagger"), 3)
    elif case == 5:
        vrng = _mksobj_draws(vrng, O("lucern hammer")) if strong \
            else _mksobj_draws(vrng, O("aklys"))
    return vrng


def _m_initweap_draws(vrng, idx):
    """Vendor ``m_initweap(mtmp)`` (makemon.c:161): mlet-keyed weapon/armor
    grants (each consuming ``mksobj`` RNG) + the final ``rnd_offensive_item``
    gate.  Only called for ``is_armed`` monsters.  Fully faithful for the
    mlets reachable at dungeon depth 1 (S_ORC goblin, S_KOBOLD kobold); other
    mlets fall through to the generic default case."""
    m = MONSTERS[idx]
    sym = int(m.symbol)
    S_KOBOLD, S_ORC = 11, 15
    S_GIANT, S_OGRE, S_TROLL = 34, 41, 46
    O = _OBJ_BY_NAME.get
    if sym == S_GIANT:
        # vendor makemon.c:183-186 — if (rn2(2)) mongets(ETTIN ? CLUB : BOULDER)
        vrng, r = _rn2(vrng, 2)
        if r:
            vrng = _mksobj_draws(
                vrng, O("club") if m.name == "ettin" else O("boulder")
            )
    elif sym == S_OGRE:
        # vendor makemon.c:433-437 — !rn2(king?3:lord?6:12) picks BATTLE_AXE else CLUB
        mod = 3 if m.name == "ogre king" else 6 if m.name == "ogre lord" else 12
        vrng, r = _rn2(vrng, mod)
        vrng = _mksobj_draws(vrng, O("battle-axe") if r == 0 else O("club"))
    elif sym == S_TROLL:
        # vendor makemon.c:439-454 — if (!rn2(2)) switch(rn2(4)) polearm grant
        vrng, r = _rn2(vrng, 2)
        if r == 0:
            vrng, w = _rn2(vrng, 4)
            _pole = {0: "ranseur", 1: "partisan", 2: "glaive", 3: "spetum"}[w]
            vrng = _mksobj_draws(vrng, O(_pole))
    elif sym == S_KOBOLD:
        vrng, r = _rn2(vrng, 4)
        if r == 0:
            vrng = _m_initthrow_draws(vrng, O("dart"), 12)
    elif sym == S_ORC:
        vrng, r = _rn2(vrng, 2)                       # ORCISH_HELM gate
        if r != 0:
            vrng = _mksobj_draws(vrng, O("orcish helm"))
        # switch selector draws rn2(2) only for ORC_CAPTAIN (not depth-1);
        # every other orc uses its own mndx -> the default sub-case.
        if idx == _PM_ORC_CAPTAIN:
            vrng, _sel = _rn2(vrng, 2)
        if idx != _PM_ORC_SHAMAN:
            vrng, g = _rn2(vrng, 2)
            if g != 0:
                if idx == _PM_GOBLIN:
                    otyp = O("orcish dagger")        # || short-circuits rn2(2)
                else:
                    vrng, d2 = _rn2(vrng, 2)
                    otyp = O("orcish dagger") if d2 == 0 else O("scimitar")
                vrng = _mksobj_draws(vrng, otyp)
    else:
        vrng = _m_initweap_default_draws(vrng, idx)
    # if ((int) mtmp->m_lev > rn2(75)) mongets(rnd_offensive_item(mtmp));
    vrng, r75 = _rn2(vrng, 75)
    # At depth 1 the picked monsters have m_lev 0..1 so this gate is virtually
    # always false; rnd_offensive_item's internal draws are not ported (would
    # only fire for higher-level armed monsters, absent from these envs).
    return vrng


# ---------------------------------------------------------------------------
# HideNSeek MONSTER directive — full faithful makemon draw replay
# ---------------------------------------------------------------------------
# Vendor sp_lev.c create_monster() -> mkclass() -> mk_roamer() -> makemon()
# for ``MONSTER: <class-letter>, <coord>, hostile``.  The class-letter monster
# is placed at a fixed coordinate (no somexy retry) and ``anymon`` is FALSE, so
# no m_initgrp group spawning fires.  The draw stream is ground-truthed against
# NETHAX_RND captures (.test_runs/hns_stream_*_seed{0,1,2}.txt): the rock-troll
# (HideNSeek seed 2), plain-troll+granted-polearm (Lava seed 2) and dragon
# (seed 0) makemon streams reproduce byte-for-byte.
from Nethax.nethax.constants.monsters import (
    M2_MALE as _M2_MALE, M2_FEMALE as _M2_FEMALE, M2_STRONG as _M2_STRONG,
    M2_GREEDY as _M2_GREEDY,
)

_G_UNIQ = 0x1000
_G_FREQ = 0x0007
_PM_GRAY_DRAGON = _MON_BY_NAME.get("gray dragon")
_PM_LONG_WORM = _MON_BY_NAME.get("long worm")
_PM_GIANT_EEL = _MON_BY_NAME.get("giant eel")
_PM_WUMPUS = _MON_BY_NAME.get("wumpus")
# mkclass placeholders (vendor mondata.h:163 is_placeholder): the class "base"
# pseudo-monsters that mkclass() skips.
_MKCLASS_PLACEHOLDERS = frozenset(
    n for n in ("giant", "orc", "elf", "human") if n in _MON_BY_NAME
)


def _mkclass_pick(vrng, class_sym: int):
    """Vendor ``mkclass_aligned(class, G_NOGEN, A_NONE)`` at dungeon depth 1.

    Returns ``(vrng, idx)`` for the chosen species.  ``maxmlev`` =
    ``level_difficulty() >> 1`` = 0 at depth 1, so every candidate is
    ``toostrong`` and the strictly-increasing-difficulty ``rn2(2)`` break
    check fires per gen-ok member (after the first) whose difficulty exceeds
    the array-previous class member's.  Frequency-weighted ``rnd(num)`` then
    picks the species.  Cite: vendor/nle/src/makemon.c:1643-1712.
    """
    maxmlev = 0
    members = [i for i, m in enumerate(MONSTERS) if int(m.symbol) == class_sym]
    nums: dict = {}
    num = 0
    prev_diff = None
    for idx in members:
        m = MONSTERS[idx]
        diff = int(m.difficulty)
        geno = int(m.generation_mask)
        gen_ok = (not (geno & _G_UNIQ)) and (m.name not in _MKCLASS_PLACEHOLDERS)
        if gen_ok:
            if num and (diff > maxmlev) and (prev_diff is not None
                                             and diff > prev_diff):
                vrng, br = _rn2(vrng, 2)
                if br:
                    break
            kf = geno & _G_FREQ
            if kf > 0:
                alev = _adj_lev_depth1(int(m.level))
                nums[idx] = kf + 1 - (1 if alev > 2 else 0)  # u.ulevel*2 == 2
                num += nums[idx]
        prev_diff = diff
    if num == 0:
        return vrng, members[0]
    vrng, rr = _rn2(vrng, num)               # rnd(num) == rn2(num) + 1
    val = rr + 1
    for idx in members:
        if idx not in nums:
            continue
        val -= nums[idx]
        if val <= 0:
            return vrng, idx
    return vrng, members[0]


def _newmonhp_draws(vrng, idx: int):
    """Vendor ``newmonhp(mon, mndx)`` (makemon.c:983-1015) HP-roll draws at
    depth 1.  Returns ``(vrng, m_lev)`` where ``m_lev = adj_lev(ptr)``."""
    m = MONSTERS[idx]
    sym = int(m.symbol)
    mlev = _adj_lev_depth1(int(m.level))
    # is_golem: fixed golemhp (no draw); is_rider: d(10,8); mlevel>49: fixed.
    from Nethax.nethax.dungeon.spawning import (
        _IS_GOLEM, _IS_RIDER,
    )
    is_golem = bool(_IS_GOLEM[idx]) if 0 <= idx < _IS_GOLEM.shape[0] else False
    is_rider = bool(_IS_RIDER[idx]) if 0 <= idx < _IS_RIDER.shape[0] else False
    if is_golem:
        return vrng, mlev                    # golemhp() is deterministic
    if is_rider:
        for _ in range(10):
            vrng, _ = _rn2(vrng, 8)          # d(10, 8)
        return vrng, mlev
    if int(m.level) > 49:
        return vrng, mlev                    # "special" fixed hp, no draw
    is_adult_dragon = (
        sym == 30 and _PM_GRAY_DRAGON is not None and idx >= _PM_GRAY_DRAGON
    )
    if is_adult_dragon:
        for _ in range(mlev):
            vrng, _ = _rn2(vrng, 4)          # 4*m_lev + d(m_lev, 4)
        return vrng, mlev
    if mlev == 0:
        vrng, _ = _rn2(vrng, 4)              # rnd(4)
        return vrng, mlev
    for _ in range(mlev):
        vrng, _ = _rn2(vrng, 8)              # d(m_lev, 8)
    return vrng, mlev


def _m_initinv_draws(vrng, idx: int, mlev: int):
    """Vendor ``m_initinv(mtmp)`` (makemon.c:590-801): mlet-keyed inventory
    grants plus the unconditional ``rn2(50)/rn2(100)`` defensive/misc tail and
    the ``!rn2(5)`` greedy-gold gate.  Faithful for the mlets the HideNSeek
    classes reach (S_NYMPH); other mlets carry no prefix draws at depth 1."""
    m = MONSTERS[idx]
    sym = int(m.symbol)
    f2 = int(m.flags2)
    O = _OBJ_BY_NAME.get
    S_NYMPH = 14
    if sym == S_NYMPH:
        # makemon.c:762-766 — !rn2(2) MIRROR, !rn2(2) POT_OBJECT_DETECTION
        vrng, a = _rn2(vrng, 2)
        if a == 0:
            vrng = _mksobj_draws(vrng, O("mirror"))
        vrng, b = _rn2(vrng, 2)
        if b == 0:
            vrng = _mksobj_draws(vrng, O("potion of object detection"))
    # makemon.c:794-800 — defensive + misc + (greedy) gold tail.
    vrng, _ = _rn2(vrng, 50)
    vrng, _ = _rn2(vrng, 100)
    if f2 & _M2_GREEDY:
        vrng, _ = _rn2(vrng, 5)
    return vrng


def _makemon_species_draws(vrng, idx: int, player_align: int = 1):
    """Replay the ``makemon()`` draw stream for a *known* species ``idx`` at
    dungeon depth 1, from ``newmonhp`` through the saddle probe (i.e. every
    draw AFTER the ``induced_align`` amask and the optional ``mkclass``
    species pick).  Returns the advanced ``vrng``.

    Draw order (vendor makemon.c):
      3. ``newmonhp()``                         (:func:`_newmonhp_draws`)
      4. female ``rn2(2)`` unless M2_MALE/M2_FEMALE
      5. ``peace_minded()`` co-aligned tail (only when the species' maligntyp
         sign matches the hero's; two rn2 with C short-circuit)
      6. makemon mlet-switch sleep — S_NYMPH/S_JABBERWOCK ``rn2(5)``
      7. ``m_initweap`` (if is_armed)           (:func:`_m_initweap_draws`)
      8. ``m_initinv``                          (:func:`_m_initinv_draws`)
      9. saddle ``rn2(100)`` (short-circuits before is_domestic)
    """
    S_NYMPH, S_JABBERWOCK = 14, 37
    m = MONSTERS[idx]
    sym = int(m.symbol)
    f2 = int(m.flags2)
    mal = int(m.alignment)
    # 3. newmonhp.
    vrng, mlev = _newmonhp_draws(vrng, idx)
    # 4. female.
    if not (f2 & (_M2_MALE | _M2_FEMALE)):
        vrng, _ = _rn2(vrng, 2)
    # 5. peace_minded co-aligned tail (makemon.c:2039-2041).  For the always-
    #    hostile-forced MONSTER directive this draw still fires during makemon
    #    when the species is co-aligned with the hero (the result is later
    #    overridden).  u.ualign.record == 0 at game start.
    def _sgn(v):
        return (v > 0) - (v < 0)
    if _sgn(mal) == _sgn(player_align):
        vrng, r1 = _rn2(vrng, 16)            # 16 + max(record,-15) == 16
        if r1:
            vrng, _ = _rn2(vrng, 2 + abs(mal))
    # 6. makemon mlet-switch sleep.
    if sym in (S_NYMPH, S_JABBERWOCK):
        vrng, _ = _rn2(vrng, 5)
    # (in_mklev sleep / longworm initworm / angel emin never fire for the
    #  species reached by the MiniHack MONSTER directives at depth 1.)
    # 7. m_initweap (armed only).
    is_armed = any(int(a[0]) == 254 for a in m.attacks)   # AT_WEAP == 254
    if is_armed:
        vrng = _m_initweap_draws(vrng, idx)
    # 8. m_initinv.
    vrng = _m_initinv_draws(vrng, idx, mlev)
    # 9. saddle: rn2(100) evaluated for every monster (C short-circuit).
    vrng, _ = _rn2(vrng, 100)
    return vrng


def _makemon_fixed_draws(vrng, idx: int, player_align: int = 1):
    """Replay the full ``MONSTER: <name>, <coord>, hostile`` ISAAC64 draw
    stream for a *fixed-species* directive (placed by name, so no ``mkclass``
    pick) and return the advanced ``vrng``.

    Draw order (vendor create_monster -> makemon):
      1. ``induced_align(80)`` -> ``rn2(3)``  (amask; no dungeon align flag)
      2. species body                          (:func:`_makemon_species_draws`)
    """
    # 1. induced_align(80): no special-level / dungeon align flag in MiniHack,
    #    so the two rn2(100) probes are skipped and al = rn2(3) - 1.
    vrng, _ = _rn2(vrng, 3)
    return _makemon_species_draws(vrng, idx, player_align)


def _hidenseek_monster_draws(vrng, class_sym: int, player_align: int = 1):
    """Replay the full ``MONSTER: <class>, <coord>, hostile`` ISAAC64 draw
    stream and return ``(vrng, idx)`` for the placed species.

    Draw order (vendor create_monster -> mk_roamer -> makemon):
      1. ``induced_align(80)`` -> ``rn2(3)``  (amask; no dungeon align flag)
      2. ``mkclass()``                          (:func:`_mkclass_pick`)
      3. species body                          (:func:`_makemon_species_draws`)
    """
    # 1. induced_align(80): no special-level / dungeon align flag in MiniHack,
    #    so the two rn2(100) probes are skipped and al = rn2(3) - 1.
    vrng, _ = _rn2(vrng, 3)
    # 2. mkclass.
    vrng, idx = _mkclass_pick(vrng, class_sym)
    # 3-9. species body.
    vrng = _makemon_species_draws(vrng, idx, player_align)
    return vrng, idx


# ---------------------------------------------------------------------------
# Directive dataclasses
# ---------------------------------------------------------------------------

# Place specification: either a (col, row) tuple, a string room_id, or None.
Place = Union[None, Tuple[int, int], str]


@dataclasses.dataclass
class _RoomDirective:
    room_id: str
    x: int           # left col; -1 = random
    y: int           # top row;  -1 = random
    w: int           # width;    -1 = random
    h: int           # height;   -1 = random
    lit: bool


@dataclasses.dataclass
class _CorridorDirective:
    src: Tuple[int, int]   # (col, row)
    dst: Tuple[int, int]


@dataclasses.dataclass
class _DoorDirective:
    x: int
    y: int
    state: str   # 'closed' | 'open' | 'locked' | 'nodoor' | 'random'


@dataclasses.dataclass
class _MonsterDirective:
    name: str
    symbol: Optional[str]
    place: Place
    args: tuple


@dataclasses.dataclass
class _TrapDirective:
    name: str
    place: Place


@dataclasses.dataclass
class _ObjectDirective:
    name: str
    symbol: Optional[str]
    place: Place
    cursestate: str   # 'random' | 'blessed' | 'uncursed' | 'cursed'


@dataclasses.dataclass
class _StairDirective:
    direction: str    # 'up' | 'down'
    x: int            # -1 = random / use place
    y: int
    place: Place


@dataclasses.dataclass
class _FillTerrainDirective:
    terrain: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclasses.dataclass
class _ReplaceTerrainDirective:
    """REPLACE_TERRAIN: probabilistic per-cell tile swap.

    Mirrors vendor des ``REPLACE_TERRAIN:(x1,y1,x2,y2), from, to, chance%``.
    Used by HideNSeek to scatter TREE/CLOUD line-of-sight occluders.
    """
    from_terrain: str
    to_terrain: str
    x1: int
    y1: int
    x2: int
    y2: int
    chance: int        # 0..100


@dataclasses.dataclass
class _StartPosDirective:
    x: int
    y: int


@dataclasses.dataclass
class _GoalPosDirective:
    x: int
    y: int


# ---------------------------------------------------------------------------
# Wave17i additions: directive types for add_altar / add_sink / add_gold /
# add_mazewalk.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _AltarOverride:
    x: int = -1
    y: int = -1
    place: Place = None


@dataclasses.dataclass
class _SinkOverride:
    place: Place = None


@dataclasses.dataclass
class _GoldDirective:
    amount: int
    place: Place = None


@dataclasses.dataclass
class _MazeWalkDirective:
    x: int
    y: int
    direction: str


@dataclasses.dataclass
class _StartingInventoryDirective:
    """Pre-populate a starting-inventory slot at reset time.

    Mirrors the vendor des ``INV:`` directive (placed-on-hero starting kit).
    Used by LavaCross-Levitate ``-Inv-`` variants where the levitation item
    must be carried at episode start rather than scattered on the floor.
    """
    category: int       # ItemCategory enum value (e.g. POTION=8, RING=4)
    type_id: int        # vendor object index (e.g. POT_LEVITATION=278)
    quantity: int
    weight: int
    buc_status: int     # 0=unknown / 1=cursed / 2=uncursed / 3=blessed
    identified: bool


@dataclasses.dataclass
class _SetMapDirective:
    """A literal ``MAP`` block from a vendor ``.des`` file.

    Source: vendor des-file ``MAP ... ENDMAP`` grids.  Each ``row`` is one
    terrain line in MiniHack ``(x=col, y=row)`` order; the level is stamped
    starting at the top-left of the active ``(h, w)`` region.  Unlike the
    default ``fill`` block, every cell — *including* spaces (which map to
    ``VOID`` per ``TERRAIN_CHAR_TO_TILE``) — is written, so the MAP block
    is authoritative and the level is correctly bounded by stone/void rather
    than leaking open FLOOR into the rest of the 80x21 grid.

    ``xstart``/``ystart`` are the internal terrain origin computed from the
    vendor ``GEOMETRY`` header (des_parser._compute_map_geometry).  The grid
    is stamped starting there rather than at ``terrain[0, 0]``.
    """
    rows: Tuple[str, ...]
    xstart: int = 0
    ystart: int = 0


@dataclasses.dataclass
class _LitRegionDirective:
    """A ``REGION:...,lit`` sub-rect that lights only its own cells.

    Source: vendor des ``REGION:(x1,y1,x2,y2),lit,"..."`` on a globally-unlit
    level (e.g. memento_short.des).  Coordinates are internal terrain
    (row, col) — the GEOMETRY offset has already been applied by the parser.
    """
    row: int
    col: int
    height: int
    width: int


# ---------------------------------------------------------------------------
# LevelGenerator
# ---------------------------------------------------------------------------

class LevelGenerator:
    """Python-side builder for MiniHack-style levels.

    Calling ``add_*`` appends a directive to an internal list.  ``get_factory``
    returns a closure that walks the directives and produces a fully populated
    ``EnvState`` using the supplied PRNG key.

    The builder is *not* JIT-traceable; it runs once on the Python side at
    environment-reset time.  The resulting ``EnvState`` is a plain Flax pytree
    that downstream ``env.step`` invocations can JIT-compile against.
    """

    def __init__(
        self,
        w: int = 80,
        h: int = 21,
        fill: str = ".",
        lit: bool = True,
    ) -> None:
        if w <= 0 or h <= 0:
            raise ValueError(f"map dimensions must be positive, got w={w} h={h}")
        if fill not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"fill char {fill!r} is not a known terrain symbol")
        # nethax terrain arrays are sized to StaticParams (default 80×21).
        # Generated levels can be smaller; we only write into the top-left
        # (w × h) sub-region and leave the rest as VOID.
        static = StaticParams()
        if w > static.map_w or h > static.map_h:
            raise ValueError(
                f"requested map {w}x{h} exceeds static bounds "
                f"{static.map_w}x{static.map_h}"
            )

        self.w = w
        self.h = h
        self.fill = fill
        self.default_lit = lit
        self._static = static

        self._directives: List[Any] = []
        self._room_directives: dict = {}   # room_id -> _RoomDirective
        self._room_counter = 0

        # Build-trace metadata captured each time the factory runs.
        # Tests inspect these to verify name→index resolution.
        self.last_monster_entry_ids: List[int] = []
        self.last_object_entry_ids: List[int] = []
        self.last_trap_types: List[int] = []
        self.last_player_pos: Optional[Tuple[int, int]] = None
        self.last_goal_pos: Optional[Tuple[int, int]] = None

    # ---- Builder API -----------------------------------------------------

    def add_room(
        self,
        x: int = -1,
        y: int = -1,
        w: int = -1,
        h: int = -1,
        *,
        lit: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> str:
        """Reserve a rectangular room region.

        Coordinates use MiniHack convention: ``x`` is column, ``y`` is row.
        ``-1`` requests random placement / size at factory time.
        Returns a stable ``room_id`` string that can be passed to
        ``place=`` arguments on other directives.
        """
        if name is None:
            name = f"room_{self._room_counter}"
        self._room_counter += 1
        eff_lit = self.default_lit if lit is None else bool(lit)
        directive = _RoomDirective(room_id=name, x=x, y=y, w=w, h=h, lit=eff_lit)
        self._directives.append(directive)
        self._room_directives[name] = directive
        return name

    def add_corridor(self, src: Tuple[int, int], dst: Tuple[int, int]) -> None:
        """Carve an L-shaped corridor between two ``(x, y)`` endpoints."""
        self._directives.append(_CorridorDirective(src=tuple(src), dst=tuple(dst)))

    def add_door(self, *args, state: str = "closed", place=None) -> None:
        """Vendor-parity add_door.

        Two signatures supported (Wave17i):
          * Vendor (level_generator.py): ``add_door(state, place=(x, y))``
            where ``state`` is a string and ``place`` is a ``(col, row)``
            coord tuple.
          * Legacy nethax: ``add_door(x, y, state="closed")``.
        """
        # Decode positional args.
        x: int = -1
        y: int = -1
        if len(args) == 1 and isinstance(args[0], str):
            # Vendor form: add_door("closed", place=(x, y))
            state = args[0]
        elif len(args) == 1 and isinstance(args[0], tuple):
            # add_door((x, y), state=...)
            x, y = int(args[0][0]), int(args[0][1])
        elif len(args) == 2:
            a0, a1 = args
            if isinstance(a0, int) and isinstance(a1, int):
                # Legacy: add_door(x, y, state=...)
                x, y = int(a0), int(a1)
            elif isinstance(a0, str):
                # add_door("closed", (x, y))
                state = a0
                if isinstance(a1, tuple) and len(a1) == 2:
                    x, y = int(a1[0]), int(a1[1])
        elif len(args) == 3:
            # Legacy: add_door(x, y, state)
            x, y, state = int(args[0]), int(args[1]), str(args[2])
        elif len(args) == 0:
            pass  # state/place as kwargs only
        else:
            raise TypeError(f"add_door: too many positional args ({len(args)})")

        if place is not None:
            if isinstance(place, tuple) and len(place) == 2:
                x, y = int(place[0]), int(place[1])

        s = str(state)
        if s not in ("closed", "open", "locked", "nodoor", "random"):
            raise ValueError(f"unknown door state: {s!r}")
        self._directives.append(_DoorDirective(x=x, y=y, state=s))

    def add_monster(
        self,
        name: str = "random",
        symbol: Optional[str] = None,
        place: Place = None,
        args: tuple = (),
    ) -> None:
        """Spawn a monster on the level."""
        self._directives.append(_MonsterDirective(
            name=name, symbol=symbol, place=place, args=tuple(args),
        ))

    def add_trap(self, name: str = "teleport", place: Place = None) -> None:
        """Place a trap of the named kind."""
        if name != "random" and name not in TRAP_NAME_TO_TYPE:
            raise ValueError(
                f"unknown trap name {name!r}; valid: {sorted(TRAP_NAME_TO_TYPE)}"
            )
        self._directives.append(_TrapDirective(name=name, place=place))

    def add_object(
        self,
        name: str = "random",
        symbol: Optional[str] = None,
        place: Place = None,
        cursestate: str = "random",
    ) -> None:
        """Place an object (item) on the ground."""
        if cursestate not in ("random", "blessed", "uncursed", "cursed"):
            raise ValueError(f"unknown cursestate: {cursestate!r}")
        self._directives.append(_ObjectDirective(
            name=name, symbol=symbol, place=place, cursestate=cursestate,
        ))

    def add_stair_up(
        self,
        x: int = -1,
        y: int = -1,
        *,
        place: Place = None,
    ) -> None:
        """Add an up-staircase tile."""
        self._directives.append(_StairDirective(
            direction="up", x=x, y=y, place=place,
        ))

    def add_stair_down(
        self,
        x=-1,
        y: int = -1,
        *,
        place: Place = None,
    ) -> None:
        """Add a down-staircase tile (also the canonical 'goal' tile).

        Vendor-parity (Wave17i): accepts either ``add_stair_down((x, y))`` or
        ``add_stair_down(x, y)`` to match vendor level_generator.py which
        passes a ``coord`` tuple.
        """
        if isinstance(x, tuple) and len(x) == 2:
            cx, cy = int(x[0]), int(x[1])
        else:
            cx, cy = int(x), int(y)
        self._directives.append(_StairDirective(
            direction="down", x=cx, y=cy, place=place,
        ))

    def fill_terrain(
        self,
        terrain: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Fill the inclusive rectangle between ``(x1, y1)`` and ``(x2, y2)``."""
        if terrain not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"unknown terrain char: {terrain!r}")
        self._directives.append(_FillTerrainDirective(
            terrain=terrain, x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    def replace_terrain(
        self,
        from_terrain: str,
        to_terrain: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        chance: int = 100,
    ) -> None:
        """Probabilistically replace ``from_terrain`` with ``to_terrain``.

        Mirrors vendor des ``REPLACE_TERRAIN:(x1,y1,x2,y2), from, to, chance%``.
        Per-cell Bernoulli sampling at factory time uses the directive-walk
        PRNG so generation is deterministic per reset key.
        """
        if from_terrain not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"unknown from_terrain char: {from_terrain!r}")
        if to_terrain not in TERRAIN_CHAR_TO_TILE:
            raise ValueError(f"unknown to_terrain char: {to_terrain!r}")
        c = max(0, min(100, int(chance)))
        self._directives.append(_ReplaceTerrainDirective(
            from_terrain=from_terrain, to_terrain=to_terrain,
            x1=x1, y1=y1, x2=x2, y2=y2, chance=c,
        ))

    def set_start_pos(self, x, y: int = -1) -> None:
        """Place the player at MiniHack ``(x, y)``.

        Vendor-parity (Wave17i): accepts ``set_start_pos((x, y))`` or
        ``set_start_pos(x, y)``.
        """
        if isinstance(x, tuple) and len(x) == 2:
            cx, cy = int(x[0]), int(x[1])
        else:
            cx, cy = int(x), int(y)
        self._directives.append(_StartPosDirective(x=cx, y=cy))

    def set_goal_pos(self, x, y: int = -1) -> None:
        """Mark a goal tile.  Stored as a STAIRCASE_DOWN tile for symmetry
        with MiniHack's ``add_goal_pos == add_stair_down`` alias.
        """
        if isinstance(x, tuple) and len(x) == 2:
            cx, cy = int(x[0]), int(x[1])
        else:
            cx, cy = int(x), int(y)
        self._directives.append(_GoalPosDirective(x=cx, y=cy))

    # ------------------------------------------------------------------
    # Wave17i: missing vendor methods
    # Cite: vendor/minihack/minihack/level_generator.py add_altar/add_sink/
    #       add_gold/add_boulder/add_mazewalk.
    # ------------------------------------------------------------------
    def add_altar(
        self,
        place: Place = None,
        align: str = "noalign",
        type: str = "altar",
    ) -> None:
        """Place an altar tile.  Vendor add_altar(place, align, type)."""
        del align, type  # nethax has a single altar tile
        # Resolve a concrete (x, y) at factory time; here we emit a
        # FillTerrainDirective covering a 1×1 region.
        if isinstance(place, tuple) and len(place) == 2:
            x, y = int(place[0]), int(place[1])
            self._directives.append(_FillTerrainDirective(
                terrain="\\",  # backslash maps to THRONE; altar uses '_' which
                                # we substitute via an inline directive below.
                x1=x, y1=y, x2=x, y2=y,
            ))
            # Replace the throne tile with ALTAR via a direct override
            # directive (handled by writing the proper tile in pass 2).
            self._directives.append(_AltarOverride(x=x, y=y))
        else:
            self._directives.append(_AltarOverride(x=-1, y=-1, place=place))

    def add_sink(self, place: Place = None) -> None:
        """Place a sink (vendor add_sink).  nethax uses FOUNTAIN as a stand-in
        because no dedicated SINK tile exists yet."""
        if isinstance(place, tuple) and len(place) == 2:
            x, y = int(place[0]), int(place[1])
            self._directives.append(_FillTerrainDirective(
                terrain="{", x1=x, y1=y, x2=x, y2=y,
            ))
        else:
            self._directives.append(_SinkOverride(place=place))

    def add_gold(
        self,
        amount: int = 1,
        place: Place = None,
    ) -> None:
        """Spawn a gold pile.  Vendor add_gold(amount, place=(x, y))."""
        # Gold maps to OBJECTS table entry "gold piece".
        # We dispatch through the existing _ObjectDirective with a custom
        # quantity annotation.
        self._directives.append(_GoldDirective(amount=int(amount), place=place))

    def add_boulder(self, place: Place = None) -> None:
        """Add a boulder.  Vendor add_boulder(place=(x, y))."""
        # Reuse add_object: vendor "boulder" exists in OBJECTS.
        self._directives.append(_ObjectDirective(
            name="boulder", symbol=None, place=place, cursestate="uncursed",
        ))

    def add_mazewalk(
        self,
        coord=None,
        dir: str = "east",
    ) -> None:
        """Carve a recursive-backtracker maze starting at ``coord``.

        Vendor MAZEWALK directive (level_generator.py + dat/lib des-file
        ``MAZEWALK: place,dir``) carves a perfect maze across the entire
        map starting from ``coord`` and propagating in ``dir``.

        nethax implementation (Wave17i): records a directive that triggers
        a recursive-backtracker carve in the factory pass (replaces the
        legacy "open room" stand-in in canonical.py:175-186).
        """
        if isinstance(coord, tuple) and len(coord) == 2:
            x, y = int(coord[0]), int(coord[1])
        else:
            x, y = 0, 0
        self._directives.append(_MazeWalkDirective(
            x=x, y=y, direction=str(dir),
        ))

    def set_map(self, rows, xstart: int = 0, ystart: int = 0) -> None:
        """Ingest a literal vendor ``MAP`` block.

        Source: vendor des-file ``MAP ... ENDMAP`` grid (e.g.
        ``vendor/minihack/minihack/dat/lava_crossing.des``).  ``rows`` is an
        iterable of strings, one per terrain line, in MiniHack ``(x, y)`` =
        (col, row) order.  The grid is stamped authoritatively at factory
        time: every glyph — including spaces, which resolve to ``VOID`` —
        is written into the terrain so the level is bounded by stone rather
        than the LG's default open-FLOOR fill.

        ``xstart``/``ystart`` place the grid's top-left at the internal
        terrain origin computed from the vendor ``GEOMETRY`` header
        (des_parser._compute_map_geometry).  Default ``(0, 0)`` preserves the
        legacy terrain[0,0] stamp for callers that pre-offset their coords
        (e.g. the Room builders in canonical.py that use ``fill_terrain``).
        """
        clean = tuple(str(r) for r in rows)
        self._directives.append(
            _SetMapDirective(rows=clean, xstart=int(xstart), ystart=int(ystart))
        )

    def add_lit_region(
        self, row: int, col: int, height: int = 1, width: int = 1,
    ) -> None:
        """Mark a rectangular region as lit (vendor ``REGION:...,lit``).

        On a globally-unlit level only these cells (plus the hero's own
        torchlight) are lit; every other cell renders dark.  Coordinates are
        internal terrain (row, col) — the GEOMETRY offset is pre-applied by
        the des parser.
        """
        self._directives.append(_LitRegionDirective(
            row=int(row), col=int(col), height=int(height), width=int(width),
        ))

    def add_random_corridors(self) -> None:
        """Vendor ``RANDOM_CORRIDORS`` directive (no-op stand-in).

        Source: vendor des-file ``RANDOM_CORRIDORS`` (e.g.
        ``vendor/minihack/minihack/dat/corridor2.des``) carves connecting
        corridors between every declared room using NetHack's
        ``join()``/``makecorridors`` (vendor/nethack/src/sp_lev.c).  nethax
        room placement already lays floor; explicit corridor carving is left
        to ``add_corridor`` directives.  This method exists so the des
        emitter drives the real LevelGenerator instead of falling back.
        """
        # No directive emitted: rooms are already navigable floor regions.
        return None

    def add_starting_inventory_item(
        self,
        category: int,
        type_id: int,
        *,
        quantity: int = 1,
        weight: int = 0,
        buc_status: int = 2,  # _BUC_UNCURSED — matches vendor ini_inv defaults
        identified: bool = True,
    ) -> None:
        """Place an item directly into the hero's starting inventory.

        Mirrors the vendor des ``INV:`` directive.  Used by LavaCross-Levitate
        ``-Inv-`` variants whose vendor counterparts ship with the levitation
        item already carried (vendor/minihack/minihack/envs/skills_lava.py
        ``MiniHackLCLevitatePotionInv`` / ``MiniHackLCLevitateRingInv``).

        Args mirror ``Nethax.nethax.subsystems.inventory.make_item``.
        """
        self._directives.append(_StartingInventoryDirective(
            category=int(category),
            type_id=int(type_id),
            quantity=int(quantity),
            weight=int(weight),
            buc_status=int(buc_status),
            identified=bool(identified),
        ))

    def mazewalk(self, row=None, col=None, direction: str = "east") -> None:
        """Vendor ``MAZEWALK`` directive via row/col emitter kwargs.

        Source: vendor des-file ``MAZEWALK: place,dir`` (e.g.
        ``vendor/minihack/minihack/dat/mazewalk.des``).  Adapter passes
        ``row``/``col`` (nethax convention); forward to ``add_mazewalk``
        which records a recursive-backtracker carve directive.
        """
        x = 0 if col is None else int(col)
        y = 0 if row is None else int(row)
        self.add_mazewalk(coord=(x, y), dir=direction)

    # ---- Factory --------------------------------------------------------

    def get_factory(self) -> Callable[[jax.Array], EnvState]:
        """Return a ``(rng) -> EnvState`` closure that materialises the level.

        Calling the closure multiple times with the same ``rng`` is
        deterministic: directives that involve randomness consume keys split
        from the input.
        """
        directives = list(self._directives)
        rooms_meta = dict(self._room_directives)
        w, h = self.w, self.h
        fill = self.fill
        static = self._static

        def factory(rng: jax.Array) -> EnvState:
            return _apply_directives(
                self, rng, directives, rooms_meta, w, h, fill, static,
            )

        return factory


# ---------------------------------------------------------------------------
# Factory implementation (Python-side, not JIT'd)
# ---------------------------------------------------------------------------

def _apply_directives(
    lg: "LevelGenerator",
    rng: jax.Array,
    directives: List[Any],
    rooms_meta: dict,
    w: int,
    h: int,
    fill: str,
    static: StaticParams,
) -> EnvState:
    """Walk the directive list and produce a populated ``EnvState``."""
    # Reset captured build-trace metadata so tests see a fresh snapshot.
    lg.last_monster_entry_ids = []
    lg.last_object_entry_ids = []
    lg.last_trap_types = []
    lg.last_player_pos = None
    lg.last_goal_pos = None

    # 1. Allocate the base EnvState.
    #
    # Under NLE_BYTEPARITY, route the bootstrap through ``NethaxEnv.reset``
    # so the ISAAC64 CORE stream is advanced through the full vendor
    # init_objects -> role_init -> init_dungeons -> u_init -> mklev sequence
    # (339 pre-cascade draws + the Archeologist u_init optional-item rn2
    # cascade at u_init.c:652-660).  This produces a state whose
    # ``vendor_rng``, character stats, starting inventory (incl. the
    # OIL_LAMP / TIN_OPENER / MAGIC_MARKER bonus item at slot 8) and
    # DISP-stream offsets are byte-aligned with vendor MiniHack.  LG
    # directives below then overwrite the dungeon-shaped fields
    # (terrain, features, traps, ground_items, monster_ai, FOV) on top of
    # the vendor-aligned base — no inventory or RNG cascade work is
    # needed in this factory anymore.
    #
    # Outside NLE_BYTEPARITY we keep the lightweight default path: minihax
    # consumers in Threefry mode don't care about ISAAC64 alignment and
    # spinning up a full NethaxEnv.reset adds non-trivial latency.
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vrng_bootstrap
    if _use_vrng_bootstrap():
        from Nethax.nethax.env import NethaxEnv as _NethaxEnv
        from Nethax.nethax.constants.roles import Role as _Role
        from Nethax.nethax.constants.races import Race as _Race
        from Nethax.nethax.subsystems.features import FeaturesState as _FeaturesState
        from Nethax.nethax.subsystems.traps import TrapState as _TrapState
        from Nethax.nethax.subsystems.monster_ai import (
            make_monster_ai_state as _make_monster_ai_state,
        )
        from Nethax.nethax.state import _empty_ground_items_array as _empty_gi
        # NethaxEnv.reset's vendor-rng branch indexes rng[0]/rng[1] to
        # rebuild the uint64 ISAAC64 seed (env.py:168-170).  Callers from
        # the minihax harness pass a typed PRNGKey (jax.random.key(...))
        # which is 0-D and cannot be subscripted; unwrap to the raw
        # uint32 pair.
        try:
            _raw_key = jax.random.key_data(rng)
        except (TypeError, ValueError):
            _raw_key = rng
        _engine = _NethaxEnv(static=static)
        # Archeologist-Human-Lawful is the canonical MiniHack character
        # ("arc-hum-law-mal" — .test_runs/minihax_byteparity.py:149).  Envs
        # that hardcode a different vendor role override it via
        # ``bootstrap_character`` (see _BOOTSTRAP_CHARACTER above); the default
        # (None) keeps Archeologist so Room / skills / Levitate are unchanged.
        # ``fast_reset=True``: skip mklev dungeon-gen / pet spawn / view_from
        # since LG directives below stamp the terrain authoritatively and the
        # ``default_lit`` block at the tail of this factory seeds FoV.
        # The ISAAC64 stream is still advanced through init_objects ->
        # role_init -> init_dungeons -> u_init so descr_idx + inventory
        # remain byte-aligned with vendor MiniHack.
        if _BOOTSTRAP_CHARACTER is not None:
            _boot_role, _boot_race, _boot_align, _boot_gender = (
                _BOOTSTRAP_CHARACTER
            )
        else:
            _boot_role, _boot_race, _boot_align, _boot_gender = (
                _Role.ARCHEOLOGIST, _Race.HUMAN, 0, 0,
            )
        state, _ = _engine.reset(
            _raw_key,
            role=_boot_role,
            race=_boot_race,
            alignment=_boot_align,
            gender=_boot_gender,
            fast_reset=True,
        )
        # NethaxEnv.reset populated the state with a full vendor dungeon
        # level — rooms, fountains, sleeping monsters, dropped items,
        # traps.  LG owns terrain authorship in minihax, so wipe those
        # entity planes back to EnvState.default empties before applying
        # LG directives.  Vendor-aligned bits we want to KEEP are:
        #   - vendor_rng / vendor_rng_disp (ISAAC64 stream offsets)
        #   - descr_idx (object-description shuffle)
        #   - inventory (Archeologist ini_inv + u_init rn2 cascade bonus)
        #   - player stats (HP / AC / role / race / align / luck)
        #   - messages (role-intro line)
        _b = static.n_branches
        _l = static.max_levels_per_branch
        _hf = static.map_h
        _wf = static.map_w
        state = state.replace(
            terrain=jnp.zeros((_b, _l, _hf, _wf), dtype=jnp.int8),
            explored=jnp.zeros((_b, _l, _hf, _wf), dtype=jnp.bool_),
            visible=jnp.zeros((_hf, _wf), dtype=jnp.bool_),
            last_seen_terrain=jnp.full(
                (_b, _l, _hf, _wf), -1, dtype=jnp.int8,
            ),
            features=_FeaturesState.default(
                num_levels=_b * _l, map_h=_hf, map_w=_wf,
            ),
            traps=_TrapState.default(
                num_levels=_b * _l, map_h=_hf, map_w=_wf,
            ),
            ground_items=_empty_gi(_b, _l, _hf, _wf),
            monster_ai=_make_monster_ai_state(),
        )
    else:
        state = EnvState.default(rng, static)
        # Default (Threefry) mode: EnvState.default leaves ``vendor_rng`` as a
        # constant empty ISAAC64 stream.  The Room-Random / -Monster / -Dark
        # placement wrappers (canonical.py ``_wrap_*_room_placement``) draw the
        # player-spawn and stair cells from ``state.vendor_rng``, so an unseeded
        # stream makes EVERY reset produce an identical layout (player + stair
        # fixed regardless of the episode key) — unlike real MiniHack, which
        # randomizes placement each episode.  Seed the stream from the episode
        # ``rng`` so default-mode layouts vary per-episode.  (Byteparity mode
        # seeds ``vendor_rng`` via ``NethaxEnv.reset`` above and is untouched.)
        from Nethax.nethax import vendor_rng as _vrng_seed_mod
        _iso_seed = jax.random.randint(
            rng, (), 0, jnp.iinfo(jnp.int32).max, dtype=jnp.int32
        ).astype(jnp.uint64)
        state = state.replace(vendor_rng=_vrng_seed_mod.init_jax(_iso_seed))

    # 2. Initialise terrain[0, 0] sub-region with the fill character.
    fill_tile = int(TERRAIN_CHAR_TO_TILE[fill])
    terrain_np = jnp.asarray(state.terrain)
    fill_block = jnp.full((h, w), jnp.int8(fill_tile), dtype=jnp.int8)
    terrain_np = terrain_np.at[0, 0, :h, :w].set(fill_block)

    # Per-room resolved bounding boxes filled in during the room pass.
    # Stored as (y1_row, x1_col, y2_row, x2_col).
    resolved_rooms: dict = {}

    # 3. Walk directives.  We split the input rng repeatedly so each random
    # decision gets independent keys; this preserves reproducibility.
    rng_pool = rng

    def _next_key():
        nonlocal rng_pool
        rng_pool, sub = jax.random.split(rng_pool)
        return sub

    # Track ground-stack depth per (row, col) so multiple add_object calls on
    # the same tile stack into successive slots.
    stack_index: dict = {}

    # Accumulate (row, col, DoorState) so doors get their open/closed/locked
    # status written into ``state.features.door_state`` at commit time.
    door_states: List[Tuple[int, int, int]] = []

    # Accumulate starting-inventory directives (vendor des INV: equivalent).
    # Materialised into ``state.inventory`` once at the end so item letters
    # are assigned positionally (see InventoryState.from_items).
    starting_inv: List[_StartingInventoryDirective] = []

    # Trap state buffer (we modify state.traps once at the end).
    trap_type_arr = jnp.asarray(state.traps.trap_type)
    # Trap state stores [num_levels, map_h, map_w] flattened across branches:
    # num_levels == n_branches * max_levels_per_branch.  For branch=0 level=0
    # the flat index is 0.
    trap_lvl_idx = 0

    # Ground-items DENSE working buffer (Item pytree).  ``EnvState.ground_items``
    # is now the sparse representation; level-gen builds items into a dense
    # [B,L,H,W,S] scratch grid and converts to sparse once at commit (gen is not
    # hot, so dense-build-then-convert is simplest — see Phase 3 migration).
    from Nethax.nethax.subsystems.inventory import _empty_dense_ground_items
    ground = _empty_dense_ground_items(
        static.n_branches, static.max_levels_per_branch,
        static.map_h, static.map_w,
    )

    # Pass 0: stamp literal MAP blocks before anything else so subsequent
    # directives (stairs, objects) write on top of the authoritative grid.
    # Source: vendor des ``MAP ... ENDMAP`` (e.g. lava_crossing.des).  A MAP
    # block is authoritative: clear the default open-FLOOR fill to VOID first
    # (mirrors vendor ``INIT_MAP:solidfill,' '`` stone) so the level is bounded
    # by stone, then stamp the grid on top.
    has_map = any(isinstance(d, _SetMapDirective) for d in directives)
    if has_map:
        void_block = jnp.full((h, w), jnp.int8(int(TileType.VOID)), dtype=jnp.int8)
        terrain_np = terrain_np.at[0, 0, :h, :w].set(void_block)
        for d in directives:
            if isinstance(d, _SetMapDirective):
                terrain_np = _stamp_map_block(
                    terrain_np, d.rows, w, h, d.xstart, d.ystart,
                )

    # Pass 1: resolve rooms (room placements are needed before other directives
    # that reference them by id).
    for d in directives:
        if isinstance(d, _RoomDirective):
            terrain_np, bbox = _resolve_and_carve_room(
                terrain_np, d, w, h, _next_key,
            )
            resolved_rooms[d.room_id] = bbox

    # Vendor mklev opens with a 4-draw stair selection block
    # (rn2(3), rn2(2), rn2(W), rn2(W)) at offsets 339-342 — see
    # .test_runs/full_init_rn2_trace_room_ultimate_15x15_seed0.txt:344-347.
    # When this LG run is processing a single-room env with monsters AND
    # we're in vendor_rng mode, consume the prefix here so subsequent
    # ``_resolve_monster`` calls see the same vrng offset vendor's
    # makemon does.  Single-room + has-monster matches Room-Monster and
    # Room-Ultimate; Trap/Random/Dark wrappers handle the prefix
    # themselves and don't have monster directives.
    has_monster_dir = any(isinstance(d, _MonsterDirective) for d in directives)
    # Room envs carve via _FillTerrainDirective (not _RoomDirective) so
    # ``resolved_rooms`` is empty.  Derive the room bbox from the FLOOR-fill
    # directive's rect (the fill is applied later in pass 2, so we can't
    # read it off ``terrain_np`` yet — read it off the directive instead)
    # so the 4-prefix uses vendor's rn2(W) modulus and ``_resolve_monster``
    # lands monsters in-room (vs the (10, 39) map-center fallback when
    # room_w defaults to map_w=80).
    if not resolved_rooms:
        _floor_glyph = "."
        for _fd in directives:
            if (
                isinstance(_fd, _FillTerrainDirective)
                and _fd.terrain == _floor_glyph
            ):
                resolved_rooms["__carved_fill__"] = (
                    int(_fd.y1), int(_fd.x1), int(_fd.y2), int(_fd.x2),
                )
                break

    _mklev_stair_cell = None  # (row, col) of the vendor mkstairs down-stair
    # NOTE: not gated on ``_use_vendor_rng_dl()``.  The Monster/Ultimate room
    # wrappers (canonical.py ``_wrap_monster_room_placement`` etc.) consume
    # ``state.vendor_rng`` for layout in *both* parity modes — the down-stair
    # cell must be stamped in both too, else default (Threefry) mode produces a
    # goal-less level and the agent can never reach stairs-down (0% transfer).
    if (
        state is not None
        and has_monster_dir
        and len(resolved_rooms) == 1
    ):
        from Nethax.nethax import vendor_rng as _vendor_rng
        ry1, rx1, ry2, rx2 = next(iter(resolved_rooms.values()))
        room_w = max(1, rx2 - rx1 + 1)
        room_h = max(1, ry2 - ry1 + 1)
        vrng = state.vendor_rng
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
        # mkstairs down-stair offset within the room (rn2(W), rn2(W)).
        vrng, _stair_xoff = _vendor_rng.rn2_jax(vrng, jnp.int32(room_w))
        vrng, _stair_yoff = _vendor_rng.rn2_jax(vrng, jnp.int32(room_h))
        state = state.replace(vendor_rng=vrng)
        _mklev_stair_cell = (ry1 + int(_stair_yoff), rx1 + int(_stair_xoff))

    # Pass 2: everything else.
    for d in directives:
        if isinstance(d, _RoomDirective):
            continue   # already handled
        elif isinstance(d, _SetMapDirective):
            continue   # stamped in pass 0
        elif isinstance(d, _CorridorDirective):
            terrain_np = _carve_corridor(terrain_np, d.src, d.dst, w, h)
        elif isinstance(d, _DoorDirective):
            terrain_np = _place_door(terrain_np, d, w, h)
            # Record the door's open/closed/locked status so the engine
            # treats it correctly.  Movement code reads
            # ``state.features.door_state`` (DoorState enum) — NOT just the
            # terrain tile — to decide whether a closed door is locked
            # (vendor: rm.h D_CLOSED/D_LOCKED; engine action_dispatch.py:676).
            # Without this the LG-authored locked doors in KeyRoom /
            # MultiRoom-Locked / LockedDoor would default to D_NODOOR (0) and
            # the agent could walk straight through.
            if 0 <= d.y < h and 0 <= d.x < w:
                door_states.append((d.y, d.x, _DOOR_STATE_VALUE[d.state]))
        elif isinstance(d, _FillTerrainDirective):
            terrain_np = _fill_terrain_rect(terrain_np, d, w, h)
        elif isinstance(d, _ReplaceTerrainDirective):
            terrain_np = _replace_terrain_rect(
                terrain_np, d, w, h, _next_key,
            )
        elif isinstance(d, _StairDirective):
            terrain_np, pos = _place_stair(
                terrain_np, d, w, h, resolved_rooms, _next_key,
            )
            if d.direction == "down" and lg.last_goal_pos is None:
                lg.last_goal_pos = pos
        elif isinstance(d, _GoalPosDirective):
            terrain_np = _set_tile(
                terrain_np, d.y, d.x, int(TileType.STAIRCASE_DOWN), w, h,
            )
            lg.last_goal_pos = (d.x, d.y)
        elif isinstance(d, _StartPosDirective):
            lg.last_player_pos = (d.x, d.y)
        elif isinstance(d, _MonsterDirective):
            # Build the occupancy set (earlier monsters only) so enexto places
            # m_initgrp members on free cells like vendor.  Vendor goodpos()
            # (teleport.c:25-105) rejects cells that hold an existing monster
            # or the player, but a staircase is still `accessible()` — so the
            # down-stair cell IS a valid enexto candidate.  Including it here
            # wrongly dropped one candidate, shifting enexto's rn2(num_good)
            # modulus/index and mis-placing the group member (Room-Monster
            # seed 6: member landed at the leader's up-left diagonal instead
            # of directly above).  Do NOT add the stair to `_occ`.
            _occ = set()
            import numpy as _np_occ
            _al = _np_occ.asarray(state.monster_ai.alive)
            _mp = _np_occ.asarray(state.monster_ai.pos)
            for _si in _np_occ.where(_al)[0]:
                _occ.add((int(_mp[_si, 0]), int(_mp[_si, 1])))
            pos_rc, mon_idx, state, members = _resolve_monster(
                d, terrain_np, w, h, resolved_rooms, _next_key, state,
                occupied=_occ, stair_cell=_mklev_stair_cell,
            )
            state = _write_monster(state, pos_rc, mon_idx)
            lg.last_monster_entry_ids.append(mon_idx)
            # Write any m_initgrp group members as additional monsters.
            for _mpos, _midx in members:
                state = _write_monster(state, _mpos, _midx)
                lg.last_monster_entry_ids.append(_midx)
        elif isinstance(d, _TrapDirective):
            pos_rc, trap_type, state = _resolve_trap(
                d, terrain_np, w, h, resolved_rooms, _next_key, state,
            )
            # The Room-Trap wrapper (canonical.py ``_wrap_trap_room_placement``)
            # reads this placeholder trap out of ``state.traps`` to exclude it
            # from the hero place_lregion scan, so keep stamping it there.  But
            # the Room-Ultimate wrapper stamps the REAL vendor trap cells itself
            # and also reads ``state.traps`` for the same exclusion — so the
            # placeholder (first-floor cell) would be a SECOND, spurious trap
            # that can sit on the true hero cell and shove the hero one cell
            # over (Room-Ultimate-5x5 seed 9: placeholder lands on (9,37), the
            # vendor hero cell).  Skip the placeholder when monsters are present
            # (i.e. the Ultimate wrapper owns trap stamping).
            if not has_monster_dir:
                trap_type_arr = trap_type_arr.at[
                    trap_lvl_idx, pos_rc[0], pos_rc[1]
                ].set(jnp.int8(trap_type))
            lg.last_trap_types.append(trap_type)
        elif isinstance(d, _ObjectDirective):
            pos_rc, obj_idx, state = _resolve_object(
                d, terrain_np, w, h, resolved_rooms, _next_key, state,
            )
            ground, stack_index = _write_ground_item(
                ground, stack_index, pos_rc, obj_idx,
            )
            lg.last_object_entry_ids.append(obj_idx)
        elif isinstance(d, _AltarOverride):
            # Place an ALTAR tile.  Resolve coordinates if needed.
            if d.x >= 0 and d.y >= 0:
                row, col = d.y, d.x
            else:
                rc = _resolve_place(
                    d.place, terrain_np, w, h, resolved_rooms, _next_key,
                )
                if rc is None:
                    continue
                row, col = rc
            terrain_np = _set_tile(
                terrain_np, row, col, int(TileType.ALTAR), w, h,
            )
        elif isinstance(d, _SinkOverride):
            # No dedicated SINK tile in nethax — use FOUNTAIN as analogue.
            rc = _resolve_place(
                d.place, terrain_np, w, h, resolved_rooms, _next_key,
            )
            if rc is None:
                continue
            terrain_np = _set_tile(
                terrain_np, rc[0], rc[1], int(TileType.FOUNTAIN), w, h,
            )
        elif isinstance(d, _GoldDirective):
            # Gold pile — emit as a ground item with type "gold piece".
            gold_idx = _OBJECT_NAME_TO_IDX.get(
                "gold piece", _OBJECT_NAME_TO_IDX.get("gold", 0),
            )
            rc = _resolve_place(
                d.place, terrain_np, w, h, resolved_rooms, _next_key,
            )
            if rc is None:
                continue
            ground, stack_index = _write_ground_item(
                ground, stack_index, rc, gold_idx,
            )
            lg.last_object_entry_ids.append(gold_idx)
        elif isinstance(d, _MazeWalkDirective):
            # Wave17i: recursive-backtracker maze starting at (d.x, d.y).
            # Carves CORRIDOR tiles through a WALL-filled region.
            terrain_np = _carve_maze(
                terrain_np, d.x, d.y, w, h, _next_key,
            )
        elif isinstance(d, _LitRegionDirective):
            # Collected below (after the loop) into ``lit_regions`` for FoV.
            pass
        elif isinstance(d, _StartingInventoryDirective):
            # Buffer; committed after the walk so all items are assigned
            # contiguous letters via InventoryState.from_items.
            starting_inv.append(d)
        else:
            # Defensive: an unknown directive class signals a programming bug.
            raise RuntimeError(f"unhandled directive type: {type(d).__name__}")

    # Stamp the vendor mkstairs down-stair at its real (seed-dependent) cell,
    # computed from the rn2(W)/rn2(W) offsets consumed in the 4-prefix block
    # above (Monster/Ultimate envs).  Done AFTER the pass-2 FLOOR fill so it
    # isn't overwritten; replaces the per-wrapper hardcoded stair stamp.
    if _mklev_stair_cell is not None:
        _scy, _scx = _mklev_stair_cell
        if 0 <= _scy < h and 0 <= _scx < w:
            terrain_np = terrain_np.at[0, 0, _scy, _scx].set(
                jnp.int8(int(TileType.STAIRCASE_DOWN))
            )

    # 3b. Vendor hero/monster collision bump (allmain.c::newgame):
    #     mklev();  u_on_upstairs();  if (MON_AT(u.ux,u.uy)) mnexto(...);
    # ``u_on_upstairs`` places the hero via place_lregion (mkmaze.c) at the
    # first FLOOR cell of its rn2(79)/rn2(21) scan — that scan does NOT skip a
    # monster-occupied cell, so the hero can land on a des monster; the
    # follow-up ``mnexto`` then relocates that squatting monster to an adjacent
    # ``enexto`` cell.  The canonical.py room wrappers instead reject
    # monster-occupied FLOOR cells when picking the hero cell, so whenever a
    # des monster's somexy cell equals the place_lregion cell the wrapper
    # over-draws and the hero lands one cell late (Room-Monster-15x15 seed 7,
    # Room-Ultimate-5x5 seed 9).  Pre-relocate that monster here so the
    # wrapper's FLOOR-minus-monster scan accepts the true hero cell unchanged.
    if state is not None and has_monster_dir and len(resolved_rooms) == 1:
        _n_trap_dir = sum(1 for _d in directives if isinstance(_d, _TrapDirective))
        state = _bump_hero_collision_monster(
            state, terrain_np, resolved_rooms, w, h, _n_trap_dir,
        )

    # 4. Commit accumulated terrain/traps/grounds.  Convert the dense scratch
    # ground buffer to the sparse representation EnvState stores (K from the
    # existing empty sparse struct so the pytree structure stays consistent).
    from Nethax.nethax.subsystems.ground_items_sparse import dense_to_sparse
    new_traps = state.traps.replace(trap_type=trap_type_arr)
    state = state.replace(
        terrain=terrain_np,
        traps=new_traps,
        ground_items=dense_to_sparse(ground, state.ground_items.K),
    )

    # 4b. Commit door open/closed/locked status into the features overlay.
    # Branch=0 level=0 flat index is 0 (same convention as traps above).
    if door_states:
        ds_arr = jnp.asarray(state.features.door_state)
        for row, col, dval in door_states:
            ds_arr = ds_arr.at[0, row, col].set(jnp.int8(dval))
        state = state.replace(
            features=state.features.replace(door_state=ds_arr),
        )

    # 4c. Materialise starting inventory.
    #
    # Vendor MiniHack envs run NetHack's full startup, so every hero spawns
    # with the role's ``ini_inv(...)`` items (vendor/nethack/src/u_init.c)
    # FIRST, and only then do des ``INV:`` directives append additional
    # carried items (vendor/nle/src/sp_lev.c::create_object with INV flag).
    #
    # Under NLE_BYTEPARITY, ``state.inventory`` was already populated by the
    # NethaxEnv.reset bootstrap at the top of this function — that runs the
    # full vendor u_init path (ini_inv + the Archeologist rn2(10)/rn2(4)/
    # rn2(5) optional-item cascade at u_init.c:652-660) reading the same
    # ISAAC64 stream offsets vendor C reads.  No further inventory work
    # needed here.  LG ``add_inventory_item`` directives are not used by
    # any canonical Room/Corridor/MazeWalk/etc env builder today (verified
    # by grep over Nethax/minihax/{envs,world_gen}); if a future env wires
    # INV: directives, extend this branch to read existing item slots out
    # of ``state.inventory.items`` and rebuild via InventoryState.from_items.
    #
    # Outside NLE_BYTEPARITY (legacy Threefry path) the state inventory is
    # the EnvState.default zero-init, so we still need to seed the
    # role-specific ini_inv items here.  No rn2 cascade in that mode — the
    # ISAAC64 stream is not modelled, so the optional bonus item is
    # deterministically omitted (matches Threefry behaviour before Lead
    # E/G's commits).
    from Nethax.nethax.subsystems.inventory import (
        InventoryState as _InventoryState,
        make_item as _make_item,
    )
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng_inv
    if _use_vendor_rng_inv():
        assert not starting_inv, (
            "minihax NLE_BYTEPARITY path: LG add_inventory_item directives "
            "(starting_inv) are not wired through the NethaxEnv.reset "
            "bootstrap.  Extend _apply_directives to read existing inventory "
            "slots out of state.inventory.items before appending."
        )
    else:
        from Nethax.nethax.subsystems.character import (
            STARTING_INVENTORY as _STARTING_INVENTORY,
            Role as _Role,
        )
        items = list(_STARTING_INVENTORY[_Role.ARCHEOLOGIST])
        items.extend(
            _make_item(
                category=d.category,
                type_id=d.type_id,
                quantity=d.quantity,
                weight=d.weight,
                buc_status=d.buc_status,
                identified=d.identified,
                bknown=True, dknown=True, rknown=True,
            )
            for d in starting_inv
        )
        state = state.replace(inventory=_InventoryState.from_items(items))

    # 5. Apply player start position (default: any free floor tile).
    # Track whether the position came from an explicit ``set_start_pos`` so
    # step 6 can skip FoV seeding when the actual hero cell will be picked
    # later by a vendor-RNG wrapper (e.g. ``_wrap_random_room_placement`` in
    # canonical.py).  Without this guard, FoV seeds at the auto-found
    # top-left corner of the first room and over-lights its Chebyshev<=1
    # neighbourhood, which becomes wrong as soon as the wrapper rewrites
    # ``player_pos`` to the vendor-accepted random cell.
    explicit_start_pos = lg.last_player_pos is not None
    if explicit_start_pos:
        px, py = lg.last_player_pos
        state = state.replace(
            player_pos=jnp.array([py, px], dtype=jnp.int16),
        )
    else:
        # Pick the first FLOOR tile we can find within the (h, w) region.
        start_rc = _find_first_floor_tile(terrain_np, w, h)
        if start_rc is not None:
            r, c = start_rc
            state = state.replace(
                player_pos=jnp.array([r, c], dtype=jnp.int16),
            )
            lg.last_player_pos = (int(c), int(r))

    # 6. Seed initial FOV / last_seen_terrain so the starting room renders as
    # lit floor (S_room, cmap=19, glyph=2378) instead of S_stone (glyph=2359).
    # Skipped when the player position is not explicit — the random-room
    # wrappers in ``Nethax/minihax/envs/canonical.py`` call
    # :func:`seed_hero_fov` themselves after pinning ``player_pos`` to the
    # vendor-accepted cell.
    if explicit_start_pos:
        # Collect any per-region lit rects (vendor ``REGION:...,lit`` on a
        # globally-unlit level).  When present, the level is globally dark
        # except for these rects plus the hero's torchlight — pass them to
        # seed_hero_fov and force default_lit off so the rest stays stone.
        lit_regions = [
            (d.row, d.col, d.height, d.width)
            for d in directives if isinstance(d, _LitRegionDirective)
        ]
        if lit_regions:
            state = seed_hero_fov(state, False, lit_regions=lit_regions)
        else:
            state = seed_hero_fov(state, lg.default_lit)

    return state


_BOULDER_OBJ_IDX: int = _OBJECT_NAME_TO_IDX["boulder"]


def _boulder_opaque_overlay(state: EnvState, shape: Tuple[int, int]) -> jnp.ndarray:
    """Build a ``bool[H, W]`` mask of cells holding a boulder on level (0, 0).

    Vendor ``does_block`` (vision.c:156-184) treats a boulder as opaque, so
    the hero's line-of-sight FOV must shadow-cast behind boulders.  We scan
    the sparse ``ground_items`` for entries whose ``type_id`` is the vendor
    "boulder" object index (any stack slot occludes) and scatter their
    (row, col) into a mask that augments ``view_from``'s terrain opacity.
    Levels with no boulders (e.g. Room) yield an all-False mask -> no-op.
    """
    h, w = shape
    gi = state.ground_items
    type_id = gi.items.type_id[0, 0]                      # [K]
    category = gi.items.category[0, 0]                    # [K]
    pos = gi.pos[0, 0].astype(jnp.int32)                  # [K, 3]
    is_boulder = (type_id == jnp.int16(_BOULDER_OBJ_IDX)) & (category != 0)
    row = jnp.clip(pos[:, 0], 0, h - 1)
    col = jnp.clip(pos[:, 1], 0, w - 1)
    flat = jnp.where(is_boulder, row * w + col, jnp.int32(h * w))
    mask = jnp.zeros((h * w + 1,), dtype=jnp.bool_).at[flat].set(True)
    return mask[:h * w].reshape(h, w)


def seed_hero_fov(
    state: EnvState,
    default_lit: bool,
    lit_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> EnvState:
    """Seed ``visible`` / ``explored`` / ``last_seen_terrain`` for the
    hero's current cell on level (branch=0, level=0).

    Mirrors ``Nethax/nethax/env.py:796-836`` (vendor ``vision_recalc`` on
    level entry).  Without this seed the engine-side ``fast_reset=True``
    bootstrap leaves ``last_seen_terrain`` at the -1 sentinel and every
    interior cell renders as stone (S_stone, glyph 2359) instead of lit
    floor (S_room, glyph 2378).

    Hero-radius (Chebyshev<=1) torchlight applies to both lit and dark
    rooms (vendor lights the hero's own 3x3 even in dark rooms).  The
    flood-fill ``lit_mask`` path is gated on ``default_lit`` because only
    ``LevelGenerator(lit=True)`` marks every carved tile as rlit=1.

    Per-cell visibility is gated by ``view_from`` so walls correctly block
    line-of-sight; this prevents over-lighting cells through a wall when
    the hero stands adjacent to the room boundary.
    """
    from Nethax.nethax.fov import view_from as _view_from
    terrain_l0 = state.terrain[0, 0]
    boulder_overlay = _boulder_opaque_overlay(state, terrain_l0.shape)
    couldsee = _view_from(
        terrain_l0,
        state.player_pos.astype(jnp.int32),
        max_radius=0,
        opaque_overlay=boulder_overlay,
    )
    if lit_regions:
        # Per-region lighting (vendor ``REGION:...,lit`` on an unlit level):
        # light only the union of the given internal-coord rects; every other
        # cell stays dark and renders as stone unless within hero torchlight.
        _h_r, _w_r = terrain_l0.shape
        _rows_r = jnp.arange(_h_r, dtype=jnp.int32)[:, None]
        _cols_r = jnp.arange(_w_r, dtype=jnp.int32)[None, :]
        lit_mask = jnp.zeros_like(terrain_l0, dtype=jnp.bool_)
        for (rr, cc, hh, ww) in lit_regions:
            # Vendor light_region (sp_lev.c:2848-2854) grows a lit rect by one
            # cell in every direction so the bounding walls are lit too (x
            # clamped to >=1, y to >=0).
            _lo_r = max(rr - 1, 0)
            _hi_r = rr + hh - 1 + 1
            _lo_c = max(cc - 1, 1)
            _hi_c = cc + ww - 1 + 1
            in_rect = (
                (_rows_r >= jnp.int32(_lo_r))
                & (_rows_r <= jnp.int32(_hi_r))
                & (_cols_r >= jnp.int32(_lo_c))
                & (_cols_r <= jnp.int32(_hi_c))
            )
            lit_mask = lit_mask | in_rect
        # Only actual (non-VOID) tiles can be lit.
        lit_mask = lit_mask & (terrain_l0 != jnp.int8(int(TileType.VOID)))
    elif default_lit:
        lit_mask = terrain_l0 != jnp.int8(int(TileType.VOID))
    else:
        lit_mask = jnp.zeros_like(terrain_l0, dtype=jnp.bool_)
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    _h_g, _w_g = terrain_l0.shape
    rows_g = jnp.arange(_h_g, dtype=jnp.int32)[:, None]
    cols_g = jnp.arange(_w_g, dtype=jnp.int32)[None, :]
    within_light = (
        (jnp.abs(rows_g - pr) <= jnp.int32(1))
        & (jnp.abs(cols_g - pc) <= jnp.int32(1))
    )
    vis = couldsee & (lit_mask | within_light)

    # Vendor ``vision_recalc`` dark-hallway wall/door rule (vision.c:745-770).
    # ``view_from`` sets ``could_see`` for the whole opaque run of a wall (its
    # "jump to the far side of a stone wall" pass), so a lit wall/door whose
    # facing floor is a DARK corridor would over-reveal.  Vendor guards this:
    # for a ``could_see`` + lit cell that is a DOOR/SDOOR/WALL and opaque
    # (``!viz_clear``) and NOT already in the hero's own 3x3 (which is always
    # IN_SIGHT), it is shown ONLY if the cell one step TOWARD the hero
    # (dx=sign(ux-col), dy=sign(uy-row)) is lit; otherwise ``goto
    # not_in_sight`` — the wall is at the end of a dark hallway and stays
    # stone.  Iron bars / open doorways / trees are transparent or non-wall
    # (vendor "else" path) so they are excluded here.
    walldoor = (
        (terrain_l0 == jnp.int8(int(TileType.WALL)))
        | (terrain_l0 == jnp.int8(int(TileType.CLOSED_DOOR)))
        | (terrain_l0 == jnp.int8(int(TileType.HWALL)))
        | (terrain_l0 == jnp.int8(int(TileType.VWALL)))
    )
    toward_dy = jnp.sign(pr - rows_g).astype(jnp.int32)  # step toward hero row
    toward_dx = jnp.sign(pc - cols_g).astype(jnp.int32)  # step toward hero col
    neigh_r = jnp.clip(rows_g + toward_dy, 0, _h_g - 1)
    neigh_c = jnp.clip(cols_g + toward_dx, 0, _w_g - 1)
    neigh_lit = lit_mask[neigh_r, neigh_c]
    hide_dark_wall = (
        walldoor & lit_mask & (~within_light) & (~neigh_lit)
    )
    vis = vis & (~hide_dark_wall)
    old_lst = state.last_seen_terrain[0, 0]
    new_lst = jnp.where(vis, terrain_l0.astype(jnp.int8), old_lst)
    new_explored = state.explored.at[0, 0].set(
        state.explored[0, 0] | vis
    )
    return state.replace(
        explored=new_explored,
        visible=vis,
        last_seen_terrain=state.last_seen_terrain.at[0, 0].set(new_lst),
    )


# ---------------------------------------------------------------------------
# Terrain helpers
# ---------------------------------------------------------------------------

def _set_tile(
    terrain_np: jax.Array, row: int, col: int, tile: int, w: int, h: int,
) -> jax.Array:
    """Set ``terrain[0, 0, row, col]`` if the cell is inside (h, w)."""
    if not (0 <= row < h and 0 <= col < w):
        return terrain_np
    return terrain_np.at[0, 0, row, col].set(jnp.int8(tile))


def _stamp_map_block(
    terrain_np: jax.Array, rows: Tuple[str, ...], w: int, h: int,
    xstart: int = 0, ystart: int = 0,
) -> jax.Array:
    """Write a literal vendor MAP grid into ``terrain[0, 0]``.

    Source: vendor des ``MAP ... ENDMAP`` blocks.  Each character is mapped
    through ``TERRAIN_CHAR_TO_TILE``; unknown glyphs (object/monster overlay
    symbols that the des places via separate directives) fall back to FLOOR
    so the tile is walkable.  Spaces resolve to ``VOID`` (vendor
    ``INIT_MAP:solidfill,' '`` stone), giving the level a hard boundary
    instead of the LG's default open-FLOOR fill.

    ``xstart``/``ystart`` offset the grid to the internal terrain origin
    computed from the GEOMETRY header (matching vendor spo_map), so the
    stamped level lands where NLE renders it.
    """
    floor = int(TileType.FLOOR)
    for y, line in enumerate(rows):
        ty = y + ystart
        if ty >= h or ty < 0:
            continue
        for x, ch in enumerate(line):
            tx = x + xstart
            if tx >= w or tx < 0:
                continue
            tile = TERRAIN_CHAR_TO_TILE.get(ch)
            if tile is None:
                # Glyph is an object/monster placement char (e.g. '!', '/');
                # the underlying terrain is open floor.
                tile = floor
            else:
                tile = int(tile)
            terrain_np = terrain_np.at[0, 0, ty, tx].set(jnp.int8(tile))
    return terrain_np


def _resolve_and_carve_room(
    terrain_np: jax.Array,
    d: _RoomDirective,
    w: int,
    h: int,
    next_key,
) -> Tuple[jax.Array, Tuple[int, int, int, int]]:
    """Pick a concrete bbox for the room directive and carve it.

    Returns the updated terrain plus the (y1, x1, y2, x2) interior bbox in
    nethax row/col convention.
    """
    # MiniHack coords: x = col, y = row.  Random values use next_key.
    rw = d.w if d.w > 0 else int(jax.random.randint(next_key(), (), 3, min(7, max(4, w // 2))))
    rh = d.h if d.h > 0 else int(jax.random.randint(next_key(), (), 3, min(6, max(4, h // 2))))

    # Constrain room interior to (h, w) including a 1-cell wall margin.
    rw = max(1, min(rw, w - 2))
    rh = max(1, min(rh, h - 2))

    if d.x >= 0:
        x1 = d.x
    else:
        max_x = max(1, w - rw - 1)
        x1 = int(jax.random.randint(next_key(), (), 1, max_x + 1))
    if d.y >= 0:
        y1 = d.y
    else:
        max_y = max(1, h - rh - 1)
        y1 = int(jax.random.randint(next_key(), (), 1, max_y + 1))

    x2 = min(x1 + rw - 1, w - 2)
    y2 = min(y1 + rh - 1, h - 2)

    # Carve walls then floor.
    wall = int(TileType.WALL)
    floor = int(TileType.FLOOR)
    # Wall border (one cell outside interior).
    for r in range(max(0, y1 - 1), min(h, y2 + 2)):
        for c in range(max(0, x1 - 1), min(w, x2 + 2)):
            if r < y1 or r > y2 or c < x1 or c > x2:
                terrain_np = terrain_np.at[0, 0, r, c].set(jnp.int8(wall))
    # Floor interior.
    for r in range(y1, y2 + 1):
        for c in range(x1, x2 + 1):
            terrain_np = terrain_np.at[0, 0, r, c].set(jnp.int8(floor))

    return terrain_np, (y1, x1, y2, x2)


def _carve_corridor(
    terrain_np: jax.Array,
    src: Tuple[int, int],
    dst: Tuple[int, int],
    w: int,
    h: int,
) -> jax.Array:
    """L-shaped corridor between (x1, y1) and (x2, y2).

    Cells that are already FLOOR are left alone; everything else becomes
    CORRIDOR.
    """
    x1, y1 = src
    x2, y2 = dst
    corridor = int(TileType.CORRIDOR)
    floor = int(TileType.FLOOR)
    # Horizontal then vertical: row y1 from min(x1,x2) to max(x1,x2),
    # then column x2 from min(y1,y2) to max(y1,y2).
    for c in range(min(x1, x2), max(x1, x2) + 1):
        if 0 <= y1 < h and 0 <= c < w:
            existing = int(terrain_np[0, 0, y1, c])
            if existing != floor:
                terrain_np = terrain_np.at[0, 0, y1, c].set(jnp.int8(corridor))
    for r in range(min(y1, y2), max(y1, y2) + 1):
        if 0 <= r < h and 0 <= x2 < w:
            existing = int(terrain_np[0, 0, r, x2])
            if existing != floor:
                terrain_np = terrain_np.at[0, 0, r, x2].set(jnp.int8(corridor))
    return terrain_np


def _place_door(
    terrain_np: jax.Array, d: _DoorDirective, w: int, h: int,
) -> jax.Array:
    tile = TileType.CLOSED_DOOR if d.state != "open" else TileType.OPEN_DOOR
    if d.state == "nodoor":
        tile = TileType.FLOOR
    return _set_tile(terrain_np, d.y, d.x, int(tile), w, h)


def _fill_terrain_rect(
    terrain_np: jax.Array, d: _FillTerrainDirective, w: int, h: int,
) -> jax.Array:
    """Inclusive rectangle fill at terrain[0, 0]."""
    tile = int(TERRAIN_CHAR_TO_TILE[d.terrain])
    y_lo, y_hi = sorted((d.y1, d.y2))
    x_lo, x_hi = sorted((d.x1, d.x2))
    y_lo = max(0, y_lo); y_hi = min(h - 1, y_hi)
    x_lo = max(0, x_lo); x_hi = min(w - 1, x_hi)
    if y_lo > y_hi or x_lo > x_hi:
        return terrain_np
    block = jnp.full((y_hi - y_lo + 1, x_hi - x_lo + 1), jnp.int8(tile), dtype=jnp.int8)
    return terrain_np.at[0, 0, y_lo:y_hi + 1, x_lo:x_hi + 1].set(block)


def _replace_terrain_rect(
    terrain_np: jax.Array,
    d: _ReplaceTerrainDirective,
    w: int,
    h: int,
    next_key,
) -> jax.Array:
    """Probabilistic per-cell tile swap (vendor REPLACE_TERRAIN).

    Bernoulli draws derived from the directive-walk PRNG; cells holding
    ``from_terrain`` flip to ``to_terrain`` when draw < chance.  Other
    cells (e.g. corridor-carved floor, stairs) are left untouched, matching
    vendor behaviour where REPLACE_TERRAIN runs before TERRAIN:randline.
    """
    from_tile = int(TERRAIN_CHAR_TO_TILE[d.from_terrain])
    to_tile = int(TERRAIN_CHAR_TO_TILE[d.to_terrain])
    y_lo, y_hi = sorted((d.y1, d.y2))
    x_lo, x_hi = sorted((d.x1, d.x2))
    y_lo = max(0, y_lo); y_hi = min(h - 1, y_hi)
    x_lo = max(0, x_lo); x_hi = min(w - 1, x_hi)
    if y_lo > y_hi or x_lo > x_hi or d.chance <= 0:
        return terrain_np
    rh = y_hi - y_lo + 1
    rw = x_hi - x_lo + 1
    key = next_key()
    draws = jax.random.uniform(key, (rh, rw), minval=0.0, maxval=100.0)
    flip = draws < float(d.chance)
    region = terrain_np[0, 0, y_lo:y_hi + 1, x_lo:x_hi + 1]
    eligible = region == jnp.int8(from_tile)
    new_region = jnp.where(eligible & flip, jnp.int8(to_tile), region)
    return terrain_np.at[0, 0, y_lo:y_hi + 1, x_lo:x_hi + 1].set(new_region)


def _place_stair(
    terrain_np: jax.Array,
    d: _StairDirective,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Tuple[jax.Array, Tuple[int, int]]:
    tile = (
        int(TileType.STAIRCASE_UP) if d.direction == "up"
        else int(TileType.STAIRCASE_DOWN)
    )
    # Coordinate priority: explicit (x, y) > place > random.
    if d.x >= 0 and d.y >= 0:
        col, row = d.x, d.y
    else:
        rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
        if rc is None:
            return terrain_np, (0, 0)
        row, col = rc
    # Vendor stairs need a "dry" cell (get_location DRY -> SPACE_POS: floor /
    # corridor / doorway).  A des ``STAIR`` whose coord lands on a WALL/VOID
    # MAP cell (e.g. memento_short.des ``STAIR:(1,5),up`` on the level's wall
    # border, which marks the branch-entry rather than a navigable stair) is
    # not materialised there — vendor renders the underlying map terrain.  Skip
    # the stamp so we don't punch a stair through a wall.
    if 0 <= row < h and 0 <= col < w:
        _existing = int(terrain_np[0, 0, row, col])
        # HWALL / VWALL are des-authored walls that behave exactly like the
        # generic WALL here: a STAIR whose coord lands on any wall (e.g.
        # memento_short.des ``STAIR:(1,5),up`` on the '-' border) is NOT
        # materialised, so vendor renders the underlying wall terrain.
        if _existing in (int(TileType.WALL), int(TileType.HWALL),
                         int(TileType.VWALL), int(TileType.VOID)):
            return terrain_np, (col, row)
    terrain_np = _set_tile(terrain_np, row, col, tile, w, h)
    return terrain_np, (col, row)


def _find_first_floor_tile(
    terrain_np: jax.Array, w: int, h: int,
) -> Optional[Tuple[int, int]]:
    """Linear scan for the first FLOOR cell in terrain[0, 0, :h, :w]."""
    sub = terrain_np[0, 0, :h, :w]
    floor = int(TileType.FLOOR)
    mask = (sub == floor)
    flat = mask.reshape(-1)
    # jnp.argmax on bool returns first True index, or 0 if all False.
    any_true = bool(jnp.any(flat))
    if not any_true:
        return None
    idx = int(jnp.argmax(flat))
    return (idx // w, idx % w)


# ---------------------------------------------------------------------------
# Placement resolution
# ---------------------------------------------------------------------------

def _resolve_place(
    place: Place,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
) -> Optional[Tuple[int, int]]:
    """Convert a ``place`` spec to a concrete ``(row, col)`` cell.

    Returns ``None`` only if the resolution failed entirely (no candidate
    tile available).
    """
    if isinstance(place, tuple):
        col, row = place
        return (int(row), int(col))
    if isinstance(place, str) and place in resolved_rooms:
        y1, x1, y2, x2 = resolved_rooms[place]
        return _random_cell_in_rect(next_key(), y1, x1, y2, x2)
    # place is None or unknown string → random floor cell on the level.
    return _random_floor_cell(terrain_np, w, h, next_key())


def _random_cell_in_rect(
    rng: jax.Array, y1: int, x1: int, y2: int, x2: int,
) -> Tuple[int, int]:
    rh = y2 - y1 + 1
    rw = x2 - x1 + 1
    k1, k2 = jax.random.split(rng)
    dy = int(jax.random.randint(k1, (), 0, max(1, rh)))
    dx = int(jax.random.randint(k2, (), 0, max(1, rw)))
    return (y1 + dy, x1 + dx)


def _random_floor_cell(
    terrain_np: jax.Array, w: int, h: int, rng: jax.Array,
) -> Optional[Tuple[int, int]]:
    """Pick a uniformly-random FLOOR tile in the (h, w) sub-region."""
    sub = terrain_np[0, 0, :h, :w]
    floor = int(TileType.FLOOR)
    mask = (sub == floor).reshape(-1)
    count = int(jnp.sum(mask))
    if count == 0:
        return None
    probs = mask.astype(jnp.float32) / count
    idx = int(jax.random.choice(rng, h * w, p=probs))
    return (idx // w, idx % w)


# ---------------------------------------------------------------------------
# Monster / object / trap directive resolution
# ---------------------------------------------------------------------------

def _enexto(terrain_np, occupied: set, xx: int, yy: int, w: int, h: int,
            vrng):
    """Replicate vendor ``enexto_core`` (teleport.c:126-218).

    Walk the borders of expanding squares centred on ``(xx, yy)`` (xx=col,
    yy=row), collecting ``goodpos`` cells; stop at the first range that has
    any, then pick one via ``rn2(num_good)``.  ``goodpos`` here = in-bounds
    FLOOR tile not in ``occupied`` (set of ``(row, col)``).  Returns
    ``((row, col)|None, vrng)``.  The ``rn2(num_good)`` draw is the same one
    vendor's m_initgrp consumes, so consuming it here keeps vrng aligned.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    floor = int(TileType.FLOOR)
    sub = terrain_np[0, 0]
    MAX_GOOD = 15

    def goodpos(x, y):
        if not (0 <= y < h and 0 <= x < w):
            return False
        if (y, x) in occupied:
            return False
        return int(sub[y, x]) == floor

    xmax0 = max(xx - 1, (w - 1) - xx)
    ymax0 = max(yy - 0, (h - 1) - yy)
    rangemax = max(xmax0, ymax0)
    good = []
    full = False
    rng = 1
    while True:
        xmin = max(1, xx - rng)
        xmax = min(w - 1, xx + rng)
        ymin = max(0, yy - rng)
        ymax = min(h - 1, yy + rng)
        for x in range(xmin, xmax + 1):
            if goodpos(x, ymin):
                good.append((ymin, x))
                if len(good) == MAX_GOOD:
                    full = True
                    break
            if goodpos(x, ymax):
                good.append((ymax, x))
                if len(good) == MAX_GOOD:
                    full = True
                    break
        if not full:
            for y in range(ymin, ymax):
                if goodpos(xmin, y):
                    good.append((y, xmin))
                    if len(good) == MAX_GOOD:
                        full = True
                        break
                if goodpos(xmax, y):
                    good.append((y, xmax))
                    if len(good) == MAX_GOOD:
                        full = True
                        break
        rng += 1
        if full or good or rng > rangemax:
            break
    if not good:
        return None, vrng
    vrng, i = _vendor_rng.rn2_jax(vrng, jnp.int32(len(good)))
    return good[int(i)], vrng


def _bump_hero_collision_monster(
    state: EnvState, terrain_np, resolved_rooms: dict, w: int, h: int,
    n_trap: int,
) -> EnvState:
    """Reproduce vendor ``allmain.c::newgame`` hero/monster collision.

    Vendor order is ``mklev()`` (places the des monsters) then
    ``u_on_upstairs()`` which drops the hero on the first FLOOR cell of a
    ``place_lregion`` ``rn2(79)/rn2(21)`` scan (mkmaze.c).  That scan does NOT
    skip a monster-occupied cell, so the hero can land on a des monster; the
    immediately-following ``if (MON_AT(u.ux,u.uy)) mnexto(...)`` then relocates
    the squatting monster to an adjacent ``enexto`` cell.

    The canonical.py room wrappers instead build the hero cell by rejecting
    monster-occupied FLOOR cells, so when a monster's somexy cell equals the
    place_lregion cell they over-scan and the hero lands one cell late.  Here
    we mirror vendor by relocating that one monster off the true hero cell, so
    the wrapper's later FLOOR-minus-monster scan accepts the same cell vendor's
    hero occupies.

    The mnexto ``rn2(num_good)`` draw vendor makes lands AFTER the hero
    placement (i.e. after the observation-relevant part of the ISAAC64 stream),
    so we run this on a CLONE of ``state.vendor_rng`` and leave the real stream
    — which the wrapper consumes for the hero placement — untouched.
    """
    from Nethax.nethax import vendor_rng as _vendor_rng
    import numpy as _np

    ry1, rx1, ry2, rx2 = next(iter(resolved_rooms.values()))
    size = rx2 - rx1 + 1

    floor = int(TileType.FLOOR)
    terr = _np.asarray(terrain_np[0, 0])

    # Clone of the pre-hero stream.  ``place_lregion`` (and, for Ultimate
    # variants, the wrapper's per-trap draws that precede it) advance this
    # clone; the real ``state.vendor_rng`` stays put for the wrapper.
    vrng = state.vendor_rng

    # Ultimate wrappers draw n_trap per-trap blocks (rn2(size), rn2(size),
    # rn2(4)) before the hero place_lregion — replicate them so the clone is at
    # the place_lregion offset, and mark the resulting trap cells (place_lregion
    # rejects them, matching bad_location).  Monster wrappers have n_trap == 0.
    trap_cells = set()
    for _ in range(n_trap):
        vrng, tx_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, ty_off = _vendor_rng.rn2_jax(vrng, jnp.int32(size))
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(4))
        tx = rx1 + int(tx_off)
        ty = ry1 + int(ty_off)
        if 0 <= ty < h and 0 <= tx < w and int(terr[ty, tx]) == floor:
            trap_cells.add((ty, tx))

    # place_lregion (mkmaze.c): first FLOOR (non-trap) cell of the rn2(79)+1 /
    # rn2(21) scan wins.  Unlike the wrapper we do NOT exclude monster cells —
    # vendor's hero lands on the monster.  200 random tries then a column-major
    # deterministic scan.
    def _ok(cy, cx):
        return (
            0 <= cy < h and 0 <= cx < w
            and int(terr[cy, cx]) == floor
            and (cy, cx) not in trap_cells
        )

    hero = None
    for _ in range(200):
        vrng, raw_x = _vendor_rng.rn2_jax(vrng, jnp.int32(79))
        vrng, cand_y = _vendor_rng.rn2_jax(vrng, jnp.int32(21))
        cx = int(raw_x) + 1
        cy = int(cand_y)
        if _ok(cy, cx):
            hero = (cy, cx)
            break
    if hero is None:
        for sx in range(1, w):
            for sy in range(0, h):
                if _ok(sy, sx):
                    hero = (sy, sx)
                    break
            if hero is not None:
                break
    if hero is None:
        return state

    # Is a monster sitting on the hero cell?  If not, the wrapper already picks
    # the right cell and no bump is needed.
    mai = state.monster_ai
    alive = _np.asarray(mai.alive)
    pos = _np.asarray(mai.pos)
    hit_slot = -1
    occupied = set()
    for si in _np.where(alive)[0]:
        cell = (int(pos[si, 0]), int(pos[si, 1]))
        occupied.add(cell)
        if cell == hero:
            hit_slot = int(si)
    if hit_slot < 0:
        return state

    # mnexto: relocate the squatting monster to an adjacent enexto cell.  The
    # rn2(num_good) draw here is vendor's post-hero mnexto draw (clone offset).
    dest, vrng = _enexto(terrain_np, occupied, hero[1], hero[0], w, h, vrng)
    if dest is None:
        return state
    new_pos = pos.copy()
    new_pos[hit_slot, 0] = dest[0]
    new_pos[hit_slot, 1] = dest[1]
    return state.replace(
        monster_ai=mai.replace(pos=jnp.asarray(new_pos, dtype=mai.pos.dtype)),
    )


def _adj_lev_depth1(mlevel: int) -> int:
    """Vendor ``adj_lev(ptr)`` (makemon.c:1757) evaluated at the first dungeon
    level: ``level_difficulty()==1`` and ``u.ulevel==1``.

    Returns the adjusted monster level that ``newmonhp`` uses to size the HP
    roll (and hence the ISAAC64 draw-count: adj_lev==0 -> one rnd(4);
    adj_lev>=1 -> ``adj_lev`` d(_,8) rolls).  No shape-changer / demon
    special-cases fire at depth 1 for the common ``rndmonst`` picks.
    """
    tmp = mlevel
    if tmp > 49:
        return 50
    tmp2 = 1 - tmp                 # level_difficulty() - mlevel, diff == 1
    if tmp2 < 0:
        tmp -= 1
    else:
        tmp += tmp2 // 5
    tmp2 = 1 - mlevel             # u.ulevel - mlevel, ulevel == 1
    if tmp2 > 0:
        tmp += tmp2 // 4
    cap = (3 * mlevel) // 2       # crude upper limit
    if cap > 49:
        cap = 49
    if tmp > cap:
        return cap
    return tmp if tmp > 0 else 0


def _resolve_monster(
    d: _MonsterDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
    state: Optional[EnvState] = None,
    occupied: Optional[set] = None,
    stair_cell: Optional[Tuple[int, int]] = None,
) -> Tuple[Tuple[int, int], int, Optional[EnvState], list]:
    """Resolve a monster directive to ``((row, col), monster_idx, new_state, members)``.

    ``members`` is a list of ``((row, col), member_idx)`` for any m_initgrp
    group members spawned by a group-flagged monster (empty otherwise).
    ``occupied`` is a set of ``(row, col)`` cells already taken by earlier
    monsters / the stair, used by ``enexto`` for member placement.

    Under ``use_vendor_rng()``, draws come from ``state.vendor_rng`` so that
    monster placement consumes the same ISAAC64 stream offsets vendor
    ``mkmonster`` consumes (vendor/nethack/src/makemon.c + mklev.c somxy
    loop).  Per the seed-0 5x5 trace diff vs the Trap variant
    (.test_runs/full_init_rn2_trace_room_monster_5x5_seed0.txt offsets
    345-369), the monster directive consumes:

    * 5× small-modulus draws for monster-type / makemon internal picks:
      ``rn2(5)``, ``rn2(2)``, ``rn2(50)``, ``rn2(100)``, ``rn2(100)``.
    * 10× ``(rn2(79), rn2(21))`` somxy() coordinate pairs (retry loop;
      we accept the first FLOOR cell, otherwise keep the last drawn pair —
      matches the bounded-retry pattern used by ``_resolve_trap``).

    The new ``state`` (with advanced ``vendor_rng``) is returned; callers
    MUST adopt it.
    """
    if d.name == "random":
        # Wave 5+ TODO: depth-aware random pick.  For Wave 4 we substitute a
        # deterministic fallback so the directive always produces a monster.
        # Vendor MiniHack-Room-Monster-5x5 seed=0 spawns a "newt" (glyph 318)
        # via Python random; use that as the byte-parity placeholder so the
        # glyph table matches vendor at the placement cell.
        idx = _MONSTER_NAME_TO_IDX.get("newt", 0)
    else:
        # vendor .des files capitalize monster names (e.g. "Minotaur") but
        # Nethax MONSTERS uses lowercase ("minotaur").  Try lowercase first
        # before raising KeyError.  Cite: vendor/minihack/minihack/dat/
        # quest_hard.des line 63 "MONSTER:('a',\"Minotaur\")".
        lookup = d.name if d.name in _MONSTER_NAME_TO_IDX else d.name.lower()
        if lookup not in _MONSTER_NAME_TO_IDX:
            raise KeyError(
                f"unknown monster name {d.name!r}; not present in MONSTERS table"
            )
        idx = _MONSTER_NAME_TO_IDX[lookup]

    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng
    # Explicitly-placed named monster (vendor des ``MONSTER:"name",(x,y)``):
    # vendor's create_monster uses the literal coordinate via
    # get_location_coord() and does NOT run the mkroom somxy() retry loop, so
    # there is no per-monster placement RNG to consume.  Honour the fixed cell
    # directly.  This only fires for name!="random" AND an explicit place tuple
    # (e.g. CorridorBattle's six giant rats); random room monsters (place=None)
    # still fall through to the vendor-rng somxy path below.
    if (
        d.name != "random"
        and d.place is not None
        and isinstance(d.place, tuple)
        and len(d.place) == 2
    ):
        px, py = int(d.place[0]), int(d.place[1])
        if 0 <= py < h and 0 <= px < w:
            return (py, px), idx, state, []
    if state is not None and _use_vendor_rng():
        from Nethax.nethax import vendor_rng as _vendor_rng
        vrng = state.vendor_rng
        # Resolve room geometry (vendor: croom->lx/hx/ly/hy).
        if resolved_rooms:
            ry1, rx1, ry2, rx2 = next(iter(resolved_rooms.values()))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w - 1, h - 1
        room_w = max(1, rx2 - rx1 + 1)
        room_h = max(1, ry2 - ry1 + 1)
        # Vendor per-monster 7-draw template (sp_lev.c:create_monster ->
        # get_location_coord -> mkroom.c:somexy + makemon.c:makemon ->
        # m_initweap).  Captured in
        # .test_runs/full_init_rn2_trace_room_ultimate_15x15_seed0.txt:343-349
        # and ..._room_monster_5x5_seed0.txt:343-349:
        #   rn2(3)        — mkclass mlet pick (3-class slice)
        #   rn2(room_w)   — somex(croom) (x offset in room)
        #   rn2(room_h)   — somey(croom) (y offset in room)
        #   rn2(2)        — somexy post-check / mk_roamer align
        #   rn2(50)       — m_initweap defensive item check (m_lev > rn2(50))
        #   rn2(100)      — m_initweap misc item check
        #   rn2(100)      — m_initweap follow-up (rnd_misc_item internal)
        # The monster lands at (rx1 + x_off, ry1 + y_off).  Variable-length
        # extras for grouping monsters (m_initgrp / m_initweap class
        # branches in makemon.c:163-800) are followup.
        # Per-monster template — GROUND-TRUTHED against the COMPLETE CORE
        # draw stream (NETHAX_RND, captures untraced rnd()/d() too).  See
        # .test_runs/full_rnd_stream_*_Monster_{5x5,15x15}_*_seed0.txt.
        # Monster-5x5 M1 (offsets 343-351) and Mon-15x15 M1 (343-351):
        #   rn2(3)   — mkclass mlet pick
        #   rn2(W)   — somex room x offset
        #   rn2(W)   — somey room y offset
        #   rnd(21)  — UNTRACED (RND#346; makemon mon setup)
        #   rnd(4)   — UNTRACED (RND#347)
        #   rn2(2)   — mk_roamer align / peace
        #   rn2(50)  — m_initweap defensive-item check
        #   rn2(100) — m_initweap misc-item check
        #   rn2(100) — m_initweap follow-up
        # Note: rn2_jax consumes exactly one ISAAC64 u64 per call regardless
        # of modulus (vendor RND = isaac64_next_uint64 % x, no rejection),
        # so the untraced fillers' moduli only matter for faithfulness.
        vrng, mkclass_val = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
        # Leader coordinate: vendor create_monster -> get_location_coord ->
        # get_location's random path (sp_lev.c:892) draws somex/somey and
        # rejects via is_ok_location(DRY) until an acceptable cell (up to 100
        # tries).  is_ok_location(DRY) accepts ROOM/CORR floor but REJECTS the
        # down-staircase cell (typ==STAIRS) that mkstairs placed earlier in
        # mklev.  The port stamps the down-stair AFTER this directive pass, so
        # terrain_np still shows FLOOR there — reject ``stair_cell`` explicitly
        # so the retry draw-count matches vendor.  Without this the leader lands
        # on the stair with NO retry, the whole ISAAC64 stream shifts, and the
        # player @ (placed later) + monster cells land wrong.  E.g.
        # Room-Monster-5x5 seed 8: the leader's first somexy == the stair cell
        # -> vendor draws an extra somex/somey retry pair the port used to miss.
        floor = int(TileType.FLOOR)
        sub = terrain_np[0, 0, :h, :w]
        _stair = tuple(stair_cell) if stair_cell is not None else None
        xi = rx1
        yi = ry1
        _cpt = 0
        while True:
            vrng, mx_off = _vendor_rng.rn2_jax(vrng, jnp.int32(room_w))
            vrng, my_off = _vendor_rng.rn2_jax(vrng, jnp.int32(room_h))
            xi = rx1 + int(mx_off)
            yi = ry1 + int(my_off)
            _ok = (
                0 <= yi < h and 0 <= xi < w
                and int(sub[yi, xi]) == floor
                and (_stair is None or (yi, xi) != _stair)
            )
            _cpt += 1
            if _ok or _cpt >= 100:
                break
        # The untraced rnd(21) IS vendor's rndmonst monster pick:
        # MONSTER:random -> create_monster(class=0) -> makemon(NULL) ->
        # rndmonst() draws rnd(choice_count) (choice_count==21 at depth 1,
        # matching the stream's rnd(21)) and walks the freq-weighted table.
        # Compute the real monster identity from it instead of a fixed newt.
        from Nethax.nethax.dungeon.spawning import pick_monster_for_level
        vrng, _picked_idx = pick_monster_for_level(None, 1, vendor_rng=vrng)
        if d.name == "random":
            idx = int(_picked_idx)
        # newmonhp (makemon.c:983): HP-roll draw-count varies BY MONSTER TYPE.
        # adj_lev(mlevel) at level_difficulty()==1, u.ulevel==1: adj_lev==0 ->
        # rnd(4) (1 draw); adj_lev>=1 -> d(adj_lev, 8) == adj_lev draws.  The
        # port previously assumed a fixed 1-draw rnd(4), which under-counts for
        # base-level>=3 monsters and shifts the downstream stream.
        _mlev = int(MONSTERS[idx].level) if 0 <= idx < len(MONSTERS) else 0
        _alev = _adj_lev_depth1(_mlev)
        _n_hp = _alev if _alev >= 1 else 1
        for _ in range(_n_hp):
            vrng, _ = _vendor_rng.rn2_jax(
                vrng, jnp.int32(8 if _alev >= 1 else 4),
            )
        vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))   # gender rn2(2)
        # Leader cell resolved by the retry loop above (room-relative somexy).
        if 0 <= yi < h and 0 <= xi < w and int(sub[yi, xi]) == floor:
            rc = (yi, xi)
        else:
            rc = ((ry1 + ry2) // 2, (rx1 + rx2) // 2)
        # m_initgrp group spawn (vendor makemon.c:1369-1378).  A freshly-made
        # G_SGROUP / G_LGROUP monster spawns a same-type group:
        #     if ((geno & G_SGROUP) && rn2(2))       m_initgrp(n=3)
        #     else if (geno & G_LGROUP)              rn2(3) ? lgrp(10) : sgrp(3)
        # m_initgrp (makemon.c:79-144): cnt = rnd(n); cnt /= (ulevel<3)?4 ...;
        # if (!cnt) cnt++; then for each member: enexto() + makemon(member).
        # u.ulevel==1 at mklev so the divisor is 4.  Each member draws
        # enexto's rn2(num_good) + the member makemon draws; afterwards the
        # leader's own m_initweap runs.  All branch on the picked monster's
        # group flag (computed from MONSTERS.generation_mask) — no hardcodes.
        # Ground truth: full_rnd_stream_*_Monster_15x15_*_seed0.txt M2 (gridbug).
        _ULEVEL_DIV = 4  # u.ulevel == 1 at mklev: (ulevel<3)?4:(ulevel<5)?2:1
        members: list = []
        is_sgroup = bool(_MON_SGROUP[idx]) if 0 <= idx < _MON_SGROUP.shape[0] else False
        is_lgroup = bool(_MON_LGROUP[idx]) if 0 <= idx < _MON_LGROUP.shape[0] else False
        grp_n = 0
        if is_sgroup:
            vrng, _gate = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
            if int(_gate) != 0:
                grp_n = 3
        elif is_lgroup:
            vrng, _lg = _vendor_rng.rn2_jax(vrng, jnp.int32(3))
            grp_n = 10 if int(_lg) != 0 else 3
        if grp_n > 0:
            # cnt = rnd(grp_n) (== rn2(grp_n)+1), divided by the ulevel
            # factor, floored, min 1.
            vrng, _cnt_raw = _vendor_rng.rn2_jax(vrng, jnp.int32(grp_n))
            cnt = (int(_cnt_raw) + 1) // _ULEVEL_DIV
            if cnt < 1:
                cnt = 1
            occ = set(occupied) if occupied else set()
            occ.add(rc)  # leader occupies its own cell
            for _m in range(cnt):
                mpos, vrng = _enexto(terrain_np, occ, xi, yi, w, h, vrng)
                # member makemon draws: newmonhp rn2(4), gender rn2(2),
                # [m_initweap if armed], m_initinv rn2(50)+rn2(100), saddle
                # rn2(100).  Members are the same species as the leader.
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(4))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(2))
                if _MON_ARMED[idx]:
                    vrng = _m_initweap_draws(vrng, idx)
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(50))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
                vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
                if mpos is not None:
                    members.append((mpos, idx))
                    occ.add(mpos)
            # Leader: m_initweap (if armed) precedes m_initinv + saddle.
            if _MON_ARMED[idx]:
                vrng = _m_initweap_draws(vrng, idx)
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(50))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        else:
            # Non-group leader: m_initweap (if armed) then m_initinv + saddle.
            # m_initweap (makemon.c:1442) is is_armed-GUARDED so unarmed gate
            # monsters (newt/gridbug/jackal) consume ZERO extra draws here.
            if _MON_ARMED[idx]:
                vrng = _m_initweap_draws(vrng, idx)
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(50))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
            vrng, _ = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        new_state = state.replace(vendor_rng=vrng)
        return rc, idx, new_state, members

    # Default (Threefry) mode for ``MONSTER:random``: pick a level-appropriate
    # random monster identity (vendor rndmonst) instead of the fixed ``newt``
    # placeholder, so spawns vary by type+strength like real MiniHack.  Without
    # this, every Room-Monster/-Ultimate monster is the same tanky entry, which
    # a real-MiniHack-trained policy can't clear (it expects mostly weak depth-1
    # monsters) — Room-Monster-15x15 transfer collapses to ~30%.  ``vendor_rng``
    # is seeded per-episode in ``_apply_directives`` (default branch), so the
    # pick varies by seed.  Byte-parity mode is handled by the
    # ``use_vendor_rng()`` block above and never reaches here.
    if d.name == "random" and state is not None:
        from Nethax.nethax import vendor_rng as _vendor_rng_dm
        from Nethax.nethax.dungeon.spawning import pick_monster_for_level
        _vrng_dm = state.vendor_rng
        _vrng_dm, _picked_dm = pick_monster_for_level(None, 1, vendor_rng=_vrng_dm)
        idx = int(_picked_dm)
        state = state.replace(vendor_rng=_vrng_dm)

    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)

    # m_initgrp group spawn in default (Threefry) mode too (vendor makemon.c
    # :1369-1378).  G_SGROUP / G_LGROUP species (jackal, sewer rat, gnome, …)
    # spawn a same-type pack around the leader; real MiniHack Room-Monster/
    # -Ultimate therefore has MORE, weaker monsters than a lone spawn.  Without
    # this the default path emitted singletons (e.g. 3 monsters where real has
    # 4), a per-spawn count/placement mismatch vs real.  Members are placed via
    # vendor ``enexto`` around the leader using the per-episode-seeded
    # ``vendor_rng``.  We skip the m_initweap rn2 alignment draws (only needed
    # for byte parity, which is handled by the use_vendor_rng() block above and
    # never reaches here).
    members: list = []
    if d.name == "random":
        is_sgroup = bool(_MON_SGROUP[idx]) if 0 <= idx < _MON_SGROUP.shape[0] else False
        is_lgroup = bool(_MON_LGROUP[idx]) if 0 <= idx < _MON_LGROUP.shape[0] else False
        if is_sgroup or is_lgroup:
            from Nethax.nethax import vendor_rng as _vrng_g
            vrng = state.vendor_rng
            grp_n = 0
            if is_sgroup:
                vrng, _gate = _vrng_g.rn2_jax(vrng, jnp.int32(2))
                if int(_gate) != 0:
                    grp_n = 3
            else:
                vrng, _lg = _vrng_g.rn2_jax(vrng, jnp.int32(3))
                grp_n = 10 if int(_lg) != 0 else 3
            if grp_n > 0:
                vrng, _cnt_raw = _vrng_g.rn2_jax(vrng, jnp.int32(grp_n))
                cnt = max(1, (int(_cnt_raw) + 1) // 4)  # u.ulevel==1 divisor
                occ = set(occupied) if occupied else set()
                occ.add(rc)
                xi, yi = rc[1], rc[0]
                for _m in range(cnt):
                    mpos, vrng = _enexto(terrain_np, occ, xi, yi, w, h, vrng)
                    if mpos is not None:
                        members.append((mpos, idx))
                        occ.add(mpos)
            state = state.replace(vendor_rng=vrng)
    return rc, idx, state, members


# ---------------------------------------------------------------------------
# mkobj — faithful random-object generation (vendor/nethack/src/mkobj.c).
#
# A des ``OBJECT:random,random`` (MiniHack -Distr distractor) resolves to
# ``mkobj_at(RANDOM_CLASS, ..., artif=TRUE)`` (sp_lev.c:1859), i.e.
# ``mkobj(RANDOM_CLASS, TRUE)``:
#   1. prob  = rnd(1000)                       (mkobj.c:251)
#   2. tprob = rnd(100); walk ``mkobjprobs`` -> object class  (mkobj.c:259-261)
#   3. i = bases[class]; walk ``objects[i].oc_prob`` until prob<=0 -> otyp
#   4. mksobj(otyp, TRUE, artif)               (class/otyp mksobj_init draws)
# The picked *true* otyp is stamped as the ground item; the NLE description
# shuffle (obs.glyph_shuffle) maps it to the per-run appearance glyph, so we
# only need the true otyp + an exact ISAAC64 draw count so the downstream
# player place_lregion stays byte-aligned.
# ---------------------------------------------------------------------------

# mkobjprobs[] (mkobj.c:29-39): (iprob, object-class).
_MKOBJPROBS = (
    (10, int(ObjectClass.WEAPON_CLASS)),
    (10, int(ObjectClass.ARMOR_CLASS)),
    (20, int(ObjectClass.FOOD_CLASS)),
    (8,  int(ObjectClass.TOOL_CLASS)),
    (8,  int(ObjectClass.GEM_CLASS)),
    (16, int(ObjectClass.POTION_CLASS)),
    (16, int(ObjectClass.SCROLL_CLASS)),
    (4,  int(ObjectClass.SPBOOK_CLASS)),
    (4,  int(ObjectClass.WAND_CLASS)),
    (3,  int(ObjectClass.RING_CLASS)),
    (1,  int(ObjectClass.AMULET_CLASS)),
)

# Box-content class probabilities (mkobj.c:41-49 ``boxiprobs``).  Used by
# ``mkbox_cnts`` to pick the class of each object generated inside a
# chest / large box.
_MKBOX_IPROBS = (
    (18, int(ObjectClass.GEM_CLASS)),
    (15, int(ObjectClass.FOOD_CLASS)),
    (18, int(ObjectClass.POTION_CLASS)),
    (18, int(ObjectClass.SCROLL_CLASS)),
    (12, int(ObjectClass.SPBOOK_CLASS)),
    (7,  int(ObjectClass.COIN_CLASS)),
    (6,  int(ObjectClass.WAND_CLASS)),
    (5,  int(ObjectClass.RING_CLASS)),
    (1,  int(ObjectClass.AMULET_CLASS)),
)


def _build_mkobj_prob_tables():
    """bases[class] (first otyp of each class) and the effective oc_prob table.

    Replays vendor ``init_objects`` (o_init.c:164-172): a class whose static
    probabilities sum to 0 (only RING_CLASS here) is rescaled to
    ``(1000 + i - first) / (last - first)``.  All other classes keep their
    static ``OBJECTS[i].prob``.
    """
    n = len(OBJECTS)
    bases: dict = {}
    for i, o in enumerate(OBJECTS):
        c = int(o.class_)
        if c not in bases:
            bases[c] = i
    eff = [int(o.prob) for o in OBJECTS]
    # per-class rescale when static sum == 0 (rings)
    first = 0
    while first < n:
        c = int(OBJECTS[first].class_)
        last = first + 1
        while last < n and int(OBJECTS[last].class_) == c:
            last += 1
        s = sum(eff[first:last])
        if s == 0:
            span = last - first
            for i in range(first, last):
                eff[i] = (1000 + i - first) // span
        first = last

    # setgemprobs (o_init.c:45-67) at depth 1 (dlvl 1 — the MiniHack skill
    # level): zero the ``9 - lev/3`` most-precious gems, then rescale the
    # semi-precious gems ``[first9 .. LAST_GEM]`` to
    # ``(171 + j - first9) / (LAST_GEM + 1 - first9)``.  Worthless-glass gems
    # (kept at their high static prob) and the gray stones (luckstone/loadstone/
    # touchstone/flint/rock, which live past LAST_GEM in the GEM_CLASS range)
    # are untouched.  LAST_GEM = the otyp just before the first "worthless
    # piece..." entry.  This is a deterministic (no-RNG) init step; without it a
    # random GEM object picks the wrong otyp (its glyph is the true otyp, gems
    # are not description-shuffled).
    gem_first = bases[int(ObjectClass.GEM_CLASS)]
    last_gem = None
    i = gem_first
    while i < n and int(OBJECTS[i].class_) == int(ObjectClass.GEM_CLASS):
        if (OBJECTS[i].name or "").startswith("worthless"):
            last_gem = i - 1
            break
        i += 1
    if last_gem is not None:
        nzero = 9 - 1 // 3  # lev == 1 at dlvl 1
        for j in range(gem_first, gem_first + nzero):
            eff[j] = 0
        rf = gem_first + nzero
        div = last_gem + 1 - rf
        for j in range(rf, last_gem + 1):
            eff[j] = (171 + j - rf) // div
    return bases, eff


_MKOBJ_BASES, _MKOBJ_EFF_PROB = _build_mkobj_prob_tables()

# Charged rings (objects.c RING(...) ``spec`` field == 1): adornment, gain
# strength, gain constitution, increase accuracy, increase damage, protection.
_CHARGED_RINGS = frozenset(range(150, 156))
# Armour otyps whose mksobj forces the curse branch (mkobj.c:1002-1004).
_ARMOR_CURSE_SPECIALS = frozenset({148, 149, 80, 137})  # fumble/levit boots,
#   helm of opposite alignment, gauntlets of fumbling.
# Amulets that curse when rn2(10) is nonzero (mkobj.c:1062-1066).
_AMULET_CURSE_SPECIALS = frozenset({180, 183, 181})  # strangulation/change/
#   restful sleep.


def _mkbox_cnts_draws(vrng, box_otyp: int, olocked: bool):
    """Consume the ISAAC64 draws of vendor ``mkbox_cnts`` (mkobj.c:275-349).

    A freshly-made chest / large box is filled with ``rn2(n+1)`` random
    contents, where ``n`` depends on the box type and its locked state
    (chest: locked?7:5; large box: locked?5:3).  Each content is a
    ``boxiprobs``-weighted class pick (``rnd(100)``) followed by
    ``mkobj(iclass)`` — which draws its own ``rnd(1000)`` otyp roll then the
    class ``mksobj`` init draws.  Coin contents instead roll
    ``rnd(level_difficulty()+2) * rnd(75)`` (2 draws) for their quantity.

    Only draw *consumption* matters here (box contents are never rendered in
    the reset observation); getting the count exact keeps the downstream
    ISAAC64 stream — the hero ``place_lregion`` — aligned with vendor.
    """
    from Nethax.nethax import vendor_rng as _vr
    OC = ObjectClass
    _CHEST, _LARGE_BOX = 190, 189
    if box_otyp == _CHEST:
        n = 7 if olocked else 5
    elif box_otyp == _LARGE_BOX:
        n = 5 if olocked else 3
    else:
        n = 0
    v = vrng
    v, cnt = _vr.rn2_jax(v, jnp.int32(n + 1))
    cnt = int(cnt)
    coin_cls = int(OC.COIN_CLASS)
    for _ in range(cnt):
        # boxiprobs class pick: tprob = rnd(100).
        v, tprob_v = _vr.rn2_jax(v, jnp.int32(100))
        tprob = int(tprob_v) + 1
        iclass = _MKBOX_IPROBS[-1][1]
        for p, c in _MKBOX_IPROBS:
            tprob -= p
            if tprob <= 0:
                iclass = c
                break
        # mkobj(iclass): prob = rnd(1000) then otyp walk within class.
        v, prob_v = _vr.rn2_jax(v, jnp.int32(1000))
        prob = int(prob_v) + 1
        i = _MKOBJ_BASES[iclass]
        while True:
            prob -= _MKOBJ_EFF_PROB[i]
            if prob <= 0:
                break
            i += 1
        content_otyp = i
        if iclass == coin_cls:
            # mkbox_cnts COIN special: quan = rnd(level_difficulty()+2)*rnd(75).
            # level_difficulty()==1 on the depth-1 MiniHack skill level.
            v, _ = _vr.rn2_jax(v, jnp.int32(3))
            v, _ = _vr.rn2_jax(v, jnp.int32(75))
        else:
            v = _mksobj_init_draws(v, content_otyp)
    return v


def _mksobj_init_draws(vrng, otyp: int):
    """Consume the exact ``mksobj`` init-block ISAAC64 draws for ``otyp``.

    Faithful port of vendor/nethack/src/mkobj.c::mksobj (init=TRUE, artif=TRUE)
    class switch.  Only draw *consumption* matters (the object glyph is the
    true otyp, stamped separately).  Rare monster-typed branches (CORPSE / EGG /
    TIN / FIGURINE / STATUE ``rndmonnum`` loops, ``mk_artifact`` follow-ups) are
    approximated to their first-order draw count; the -Distr distractor never
    resolves to those on the covered seeds.
    """
    from Nethax.nethax import vendor_rng as _vr
    OC = ObjectClass
    o = OBJECTS[otyp]
    cls = int(o.class_)

    def rn2(v, n):
        v, r = _vr.rn2_jax(v, jnp.int32(n))
        return v, int(r)

    def rne(v, x):
        # utmp = 5 for u.ulevel < 15; tmp advances while tmp<5 and !rn2(x).
        tmp = 1
        while tmp < 5:
            v, r = rn2(v, x)
            if r != 0:
                break
            tmp += 1
        return v, tmp

    def blessorcurse(v, chance):
        v, r = rn2(v, chance)
        bcsign = 0
        if r == 0:
            v, r2 = rn2(v, 2)
            bcsign = -1 if r2 == 0 else 1
        return v, bcsign

    def is_missile(oi):
        # is_multigen / is_poisonable: launched-missile weapon skills
        # (mkobj.c obj.h:197-204). Nethax stores these skills as -20..-24.
        return cls == int(OC.WEAPON_CLASS) and -24 <= int(OBJECTS[oi].oc_skill) <= -20

    v = vrng
    if cls == int(OC.WEAPON_CLASS):
        if is_missile(otyp):
            v, _ = rn2(v, 6)                 # quan = rn1(6, 6)
        v, r = rn2(v, 11)
        if r == 0:
            v, _ = rne(v, 3)                 # spe = rne(3)
            v, _ = rn2(v, 2)                 # blessed = rn2(2)
        else:
            v, r2 = rn2(v, 10)
            if r2 == 0:
                v, _ = rne(v, 3)             # curse; spe = -rne(3)
            else:
                v, _ = blessorcurse(v, 10)
        if is_missile(otyp):
            v, _ = rn2(v, 100)               # is_poisonable && !rn2(100)
        v, _ = rn2(v, 20)                    # artif && !rn2(20)
        return v
    if cls == int(OC.ARMOR_CLASS):
        v, r1 = rn2(v, 10)
        take_curse = False
        if r1 != 0:
            if otyp in _ARMOR_CURSE_SPECIALS:
                take_curse = True
            else:
                v, r2 = rn2(v, 11)
                take_curse = (r2 == 0)
        if take_curse:
            v, _ = rne(v, 3)                 # curse; spe = -rne(3)
        else:
            v, r3 = rn2(v, 10)
            if r3 == 0:
                v, _ = rn2(v, 2)             # blessed = rn2(2)
                v, _ = rne(v, 3)             # spe = rne(3)
            else:
                v, _ = blessorcurse(v, 10)
        v, _ = rn2(v, 40)                    # artif && !rn2(40)
        return v
    if cls == int(OC.RING_CLASS):
        if otyp in _CHARGED_RINGS:
            v, bcsign = blessorcurse(v, 3)
            spe = 0
            v, r = rn2(v, 10)
            if r != 0:
                v, r2 = rn2(v, 10)
                if r2 != 0 and bcsign != 0:
                    v, e = rne(v, 3)
                    spe = bcsign * e
                else:
                    v, b = rn2(v, 2)
                    v, e = rne(v, 3)
                    spe = e if b != 0 else -e
            if spe == 0:
                v, a = rn2(v, 4)
                v, c = rn2(v, 3)
                spe = a - c
            if spe < 0:
                v, _ = rn2(v, 5)             # spe<0 && rn2(5) -> curse
        else:
            v, r = rn2(v, 10)
            if r != 0:
                # RIN_TELEPORTATION/POLYMORPH/AGGRAVATE_MONSTER/HUNGER or !rn2(9)
                specials = {161, 162, 171, 173}
                if otyp not in specials:
                    v, _ = rn2(v, 9)
        return v
    if cls == int(OC.AMULET_CLASS):
        v, r = rn2(v, 10)
        if r != 0 and otyp in _AMULET_CURSE_SPECIALS:
            pass                             # curse (no further draw)
        else:
            v, _ = blessorcurse(v, 10)
        return v
    if cls in (int(OC.POTION_CLASS), int(OC.SCROLL_CLASS)):
        v, _ = blessorcurse(v, 4)
        return v
    if cls == int(OC.SPBOOK_CLASS):
        v, _ = blessorcurse(v, 17)
        return v
    if cls == int(OC.WAND_CLASS):
        if otyp == 387:                      # wand of wishing: spe = rnd(3)
            v, _ = rn2(v, 3)
        else:
            v, _ = rn2(v, 5)                 # spe = rn1(5, ...) -> rn2(5)
        v, _ = blessorcurse(v, 17)
        return v
    if cls == int(OC.FOOD_CLASS):
        if otyp == 250:                      # kelp frond: quan = rnd(2)
            v, _ = rn2(v, 2)
        if otyp not in (240, 245, 250):      # not corpse / meat ring / kelp
            v, _ = rn2(v, 6)                 # !rn2(6) -> quan = 2
        return v
    if cls == int(OC.GEM_CLASS):
        if otyp == 446:                      # gem-class rock: quan = rn1(6, 6)
            v, _ = rn2(v, 6)
        elif otyp not in (442, 443):         # not luckstone / loadstone
            v, _ = rn2(v, 6)                 # !rn2(6) -> quan = 2
        return v
    if cls == int(OC.TOOL_CLASS):
        if otyp in (199, 200):               # tallow/wax candle
            v, r = rn2(v, 2)
            if r != 0:
                v, _ = rn2(v, 7)
            v, _ = blessorcurse(v, 5)
        elif otyp in (201, 202):             # brass lantern / oil lamp
            v, _ = rn2(v, 500)               # age = rn1(500, 1000)
            v, _ = blessorcurse(v, 5)
        elif otyp == 203:                    # magic lamp
            v, _ = blessorcurse(v, 2)
        elif otyp in (190, 189):             # chest / large box
            v, r_lock = rn2(v, 5)            # olocked = !!(rn2(5))
            v, _ = rn2(v, 10)                # otrapped = !(rn2(10))
            v = _mkbox_cnts_draws(v, otyp, r_lock != 0)  # fill contents
        elif otyp in (191, 192, 193, 194):   # ice box / sack / oilskin / boh
            v, _ = rn2(v, 1)                 # mkbox_cnts content roll (n small)
        elif otyp in (204, 213, 217):        # camera / tinning kit / marker
            v, _ = rn2(v, 70)                # spe = rn1(70, 30)
        elif otyp == 215:                    # can of grease
            v, _ = rn2(v, 25)                # spe = rnd(25)
            v, _ = blessorcurse(v, 10)
        elif otyp == 206:                    # crystal ball
            v, _ = rn2(v, 5)                 # spe = rnd(5)
            v, _ = blessorcurse(v, 2)
        elif otyp in (227, 195):             # horn of plenty / bag of tricks
            v, _ = rn2(v, 20)                # spe = rnd(20)
        elif otyp in (223, 229, 225, 226, 233):  # magic instruments
            v, _ = rn2(v, 5)                 # spe = rn1(5, 4)
        # bell of opening / default tools: no init draws.
        return v
    # Other classes (illobj/coin/rock/ball/chain/venom) are never reachable
    # via mkobjprobs; consume nothing.
    return v


def _resolve_object(
    d: _ObjectDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
    state: Optional[EnvState] = None,
) -> Tuple[Tuple[int, int], int, Optional[EnvState]]:
    """Resolve an object directive to ``((row, col), object_idx, new_state)``.

    ``OBJECT:random`` under ``use_vendor_rng()`` is a full mkobj replay: the
    coordinate somexy() + ``mkobj(RANDOM_CLASS, TRUE)`` (class/otyp walk +
    mksobj init draws) all consume ``state.vendor_rng``.  The advanced
    ``state`` is returned; callers MUST adopt it.  All other cases keep the
    legacy behaviour (state passed through unchanged).
    """
    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng
    if state is not None and _use_vendor_rng() and d.name == "random":
        from Nethax.nethax import vendor_rng as _vendor_rng
        vrng = state.vendor_rng
        if resolved_rooms:
            ry1, rx1, ry2, rx2 = next(iter(resolved_rooms.values()))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w - 1, h - 1
        room_w = max(1, rx2 - rx1 + 1)
        room_h = max(1, ry2 - ry1 + 1)
        floor = int(TileType.FLOOR)
        sub = terrain_np[0, 0, :h, :w]
        # get_location_coord(DRY, random): somex/somey retry until DRY floor
        # (sp_lev.c:892 / mkroom.c somexy) — accept the first floor cell.
        xi, yi = rx1, ry1
        for _ in range(100):
            vrng, ox = _vendor_rng.rn2_jax(vrng, jnp.int32(room_w))
            vrng, oy = _vendor_rng.rn2_jax(vrng, jnp.int32(room_h))
            xi = rx1 + int(ox)
            yi = ry1 + int(oy)
            if 0 <= yi < h and 0 <= xi < w and int(sub[yi, xi]) == floor:
                break
        rc = (yi, xi)
        # mkobj(RANDOM_CLASS, TRUE): prob = rnd(1000); tprob = rnd(100).
        vrng, prob_v = _vendor_rng.rn2_jax(vrng, jnp.int32(1000))
        prob = int(prob_v) + 1
        vrng, tprob_v = _vendor_rng.rn2_jax(vrng, jnp.int32(100))
        tprob = int(tprob_v) + 1
        oclass = None
        for p, c in _MKOBJPROBS:
            tprob -= p
            if tprob <= 0:
                oclass = c
                break
        i = _MKOBJ_BASES[oclass]
        while True:
            prob -= _MKOBJ_EFF_PROB[i]
            if prob <= 0:
                break
            i += 1
        otyp = i
        vrng = _mksobj_init_draws(vrng, otyp)
        new_state = state.replace(vendor_rng=vrng)
        return rc, otyp, new_state

    if d.name == "random":
        idx = _OBJECT_NAME_TO_IDX.get("apple", 0)
    else:
        if d.name not in _OBJECT_NAME_TO_IDX:
            raise KeyError(
                f"unknown object name {d.name!r}; not present in OBJECTS table"
            )
        idx = _OBJECT_NAME_TO_IDX[d.name]
    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)
    return rc, idx, state


def _resolve_trap(
    d: _TrapDirective,
    terrain_np: jax.Array,
    w: int,
    h: int,
    resolved_rooms: dict,
    next_key,
    state: Optional[EnvState] = None,
) -> Tuple[Tuple[int, int], int, Optional[EnvState]]:
    """Resolve a trap directive to ``((row, col), trap_kind, new_state)``.

    Under ``use_vendor_rng()``, draws come from ``state.vendor_rng`` so that
    trap placement consumes the same ISAAC64 stream offsets vendor
    ``mktrap`` consumes (vendor/nethack/src/mklev.c:1318-1366 trap-kind
    picker + somxy loop): 2× ``rn2(5)`` for type/internal selection, then
    up to 5× ``(rn2(79), rn2(21))`` coordinate pairs (somxy retry loop),
    accepting the first FLOOR cell.  The new ``state`` (with advanced
    ``vendor_rng``) is returned; callers MUST adopt it.
    """
    if d.name == "random":
        trap_kind = int(TrapType.TELEP_TRAP)
    else:
        trap_kind = int(TRAP_NAME_TO_TYPE[d.name])

    from Nethax.nethax.parity_mode import use_vendor_rng as _use_vendor_rng
    if state is not None and _use_vendor_rng():
        # Vendor-rng draws for the trap (2× rn2(5) kind + 5× somxy pairs)
        # are now consumed in ``_wrap_trap_room_placement`` AFTER the stair
        # stamp, matching vendor mklev order (mkstairs precedes mktrap).
        # We pick a deterministic placeholder position here without
        # touching the vendor stream; full trap-glyph parity is a follow-up.
        floor = int(TileType.FLOOR)
        sub = terrain_np[0, 0, :h, :w]
        rc: Optional[Tuple[int, int]] = None
        for yi in range(h):
            for xi in range(w):
                if int(sub[yi, xi]) == floor:
                    rc = (yi, xi)
                    break
            if rc is not None:
                break
        if rc is None:
            rc = (0, 0)
        return rc, trap_kind, state

    rc = _resolve_place(d.place, terrain_np, w, h, resolved_rooms, next_key)
    if rc is None:
        rc = (0, 0)
    return rc, trap_kind, state


# ---------------------------------------------------------------------------
# EnvState writers
# ---------------------------------------------------------------------------

def _write_monster(
    state: EnvState, pos_rc: Tuple[int, int], mon_idx: int,
) -> EnvState:
    """Populate the first empty monster_ai slot with a freshly placed monster.

    Wave 4 simplification: we use a Python-side scan for the first
    ``alive=False`` slot (this whole function runs on the host).
    """
    mai = state.monster_ai
    alive_np = jnp.asarray(mai.alive)
    # Find first inactive slot.
    free_mask = ~alive_np
    if not bool(jnp.any(free_mask)):
        # No room — drop silently.  (Wave 5: surface a warning.)
        return state
    slot = int(jnp.argmax(free_mask.astype(jnp.int8)))

    row, col = pos_rc
    mon_idx_clipped = max(0, min(mon_idx, int(_BASE_AC.shape[0]) - 1))

    # HP from the monster's level (vendor makemon.c::newmonhp), NOT a flat 8.
    # A flat 8 made every monster equally tanky regardless of species, so a
    # real-MiniHack-trained policy that expects mostly weak depth-1 monsters
    # (newt/sewer-rat = 1..4 HP) couldn't clear Room-Monster levels and got
    # drained to death — Room-Monster-15x15 transfer ~30%.  Roll per-spawn with
    # a key folded from the episode rng + slot so it's reproducible and varies
    # by monster.  Monster HP is not part of the NLE obs (glyphs/blstats), so
    # byte parity is unaffected.
    from Nethax.nethax.subsystems.monster_ai import _newmonhp_roll as _nmhp
    _m_lev = int(MONSTERS[mon_idx_clipped].level)
    _hp_key = jax.random.fold_in(state.rng, jnp.int32(slot * 1009 + mon_idx_clipped + 1))
    _hp_val = int(_nmhp(_hp_key, jnp.int32(_m_lev)))

    new_mai = mai.replace(
        pos=mai.pos.at[slot].set(jnp.array([row, col], dtype=jnp.int16)),
        # entry_idx selects the monster glyph (GLYPH_MON_OFF + entry_idx ==
        # nle glyph; constants/glyphs.py: GLYPH_MON_OFF=0).  Previously left
        # default (0 = uninitialized), which rendered as NUL on the map and
        # produced glyph-table divergence for Monster-* room variants.
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(mon_idx_clipped)),
        hp=mai.hp.at[slot].set(jnp.int32(_hp_val)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(_hp_val)),
        m_lev=mai.m_lev.at[slot].set(jnp.int16(_m_lev)),
        alive=mai.alive.at[slot].set(jnp.bool_(True)),
        ac=mai.ac.at[slot].set(_BASE_AC[mon_idx_clipped]),
        is_large=mai.is_large.at[slot].set(_IS_LARGE[mon_idx_clipped]),
        attack_dice_n=mai.attack_dice_n.at[slot].set(
            _ATK_DICE_N[mon_idx_clipped].astype(jnp.int8)
        ),
        attack_dice_sides=mai.attack_dice_sides.at[slot].set(
            _ATK_DICE_S[mon_idx_clipped].astype(jnp.int8)
        ),
        asleep=mai.asleep.at[slot].set(jnp.bool_(False)),
        peaceful=mai.peaceful.at[slot].set(jnp.bool_(False)),
    )
    return state.replace(monster_ai=new_mai)


def _write_ground_item(
    ground: Any,
    stack_index: dict,
    pos_rc: Tuple[int, int],
    obj_idx: int,
) -> Tuple[Any, dict]:
    """Stamp an item into the top of the ground stack at ``pos_rc``.

    Stack overflow (> MAX_GROUND_STACK items on one tile) drops the new item.
    """
    row, col = pos_rc
    key = (row, col)
    depth = stack_index.get(key, 0)
    if depth >= MAX_GROUND_STACK:
        return ground, stack_index

    entry = OBJECTS[obj_idx] if 0 <= obj_idx < len(OBJECTS) else None
    cat_value = int(entry.class_) if entry is not None else int(ObjectClass.FOOD_CLASS)
    weight = entry.weight if entry is not None else 0

    new_ground = ground.replace(
        category=ground.category.at[0, 0, row, col, depth].set(jnp.int8(cat_value)),
        type_id=ground.type_id.at[0, 0, row, col, depth].set(jnp.int16(obj_idx)),
        buc_status=ground.buc_status.at[0, 0, row, col, depth].set(jnp.int8(0)),
        enchantment=ground.enchantment.at[0, 0, row, col, depth].set(jnp.int8(0)),
        charges=ground.charges.at[0, 0, row, col, depth].set(jnp.int8(0)),
        identified=ground.identified.at[0, 0, row, col, depth].set(jnp.bool_(False)),
        quantity=ground.quantity.at[0, 0, row, col, depth].set(jnp.int16(1)),
        weight=ground.weight.at[0, 0, row, col, depth].set(jnp.int32(weight)),
        ac_bonus=ground.ac_bonus.at[0, 0, row, col, depth].set(jnp.int8(0)),
        is_two_handed=ground.is_two_handed.at[0, 0, row, col, depth].set(jnp.bool_(False)),
    )
    new_stack = dict(stack_index)
    new_stack[key] = depth + 1
    return new_ground, new_stack


# ---------------------------------------------------------------------------
# Wave17i: recursive-backtracker maze carver for ``add_mazewalk``.
# Cite: vendor MiniHack uses NetHack's MAZEWALK des-file directive which
# triggers a recursive maze dig in mklev.c::makemaz / sp_lev.c::create_maze.
# We approximate the layout with a standard recursive-backtracker on a
# grid that walks in 2-cell strides (the same algorithm used by NetHack's
# walkfrom in mklev.c).
# ---------------------------------------------------------------------------


def _carve_maze(
    terrain_np: jax.Array,
    start_x: int,
    start_y: int,
    w: int,
    h: int,
    next_key,
) -> jax.Array:
    """Recursive-backtracker maze carve into the (h, w) top-left subregion.

    The maze is carved with WALL tiles separating CORRIDOR cells.  We walk in
    2-cell steps so each "stride" carves both the bridge cell and the target
    cell, matching the vendor's walkfrom() behaviour
    (vendor/nethack/src/mklev.c::walkfrom).

    Args:
        terrain_np: current terrain array.
        start_x:    starting column.
        start_y:    starting row.
        w, h:       active map extent.
        next_key:   PRNG factory for shuffling neighbour order.

    Returns:
        terrain_np with maze carved in.
    """
    # Materialise the sub-region as Python lists for the iterative carve.
    # This entire function runs at Python init time (level-build, not the
    # per-step JIT trace), so we use plain Python containers and pull
    # randomness from the JAX PRNG stream (via ``next_key``) instead of
    # round-tripping through numpy's RandomState.
    wall = int(TileType.WALL)
    corridor = int(TileType.CORRIDOR)
    # Fill with WALL first.
    sub = [[wall for _ in range(w)] for _ in range(h)]

    sx = max(0, min(int(start_x), w - 1))
    sy = max(0, min(int(start_y), h - 1))
    # Align to odd coords so the 2-stride walk stays in-bounds.
    if sx % 2 == 0:
        sx = min(w - 1, sx + 1)
    if sy % 2 == 0:
        sy = min(h - 1, sy + 1)

    visited = [[False for _ in range(w)] for _ in range(h)]
    stack = [(sy, sx)]
    visited[sy][sx] = True
    sub[sy][sx] = corridor

    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                neighbours.append((nr, nc, dr, dc))
        if not neighbours:
            stack.pop()
            continue
        # Pure JAX-stream randomness: pull a fresh subkey from the
        # build-time PRNG factory each step.  Equivalent in distribution
        # to ``random.randrange(len(neighbours))`` but stays entirely on
        # the JAX side — no numpy RandomState round-trip.
        idx = int(jax.random.randint(next_key(), (), 0, len(neighbours)))
        nr, nc, dr, dc = neighbours[idx]
        # Carve bridge cell + neighbour.
        br, bc = r + dr // 2, c + dc // 2
        sub[br][bc] = corridor
        sub[nr][nc] = corridor
        visited[nr][nc] = True
        stack.append((nr, nc))

    # Write the carved sub-region back.
    sub_arr = jnp.asarray(sub, dtype=terrain_np.dtype)
    new_terrain = terrain_np.at[0, 0, :h, :w].set(sub_arr)
    return new_terrain
