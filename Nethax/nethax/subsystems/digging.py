"""Digging subsystem — multi-turn pickaxe/mattock/axe dig mechanic.

Canonical source: vendor/nethack/src/dig.c

Design
------
``DigState`` tracks an in-progress dig.  Each turn ``dig_tick`` is called from
``_step_impl``.  Vendor tracks a single global ``svc.context.digging.effort``
(dig.c:303) and triggers terrain change at fixed thresholds:

    Horizontal dig: effort > 100  →  carve WALL/door/tree   (dig.c:440)
    Down dig:       effort > 50   →  PIT phase (no descent) (dig.c:372,380)
                    effort > 250  →  HOLE + descend         (dig.c:372)

Effort per swing (dig.c:365-368):

    effort += 10 + rn2(5) + abon() + uwep->spe
                  - greatest_erosion(uwep) + u.udaminc
    if (Race_if(PM_DWARF)) effort *= 2

Direction encoding (matches ``_DIR_TABLE`` in action_dispatch.py):
    0=N  1=E  2=S  3=W  4=NE  5=SE  6=SW  7=NW  8=DOWN
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.constants.races import PM_DWARF

# ---------------------------------------------------------------------------
# Pickaxe / mattock / axe type IDs (vendor/nethack/include/objects.h).
# Cite: vendor/nethack/include/obj.h lines 217-222 (is_axe / is_pick macros).
# pick-axe         : type_id 234 (TOOL_CLASS)   — objects.py line 4864
# dwarvish mattock : type_id 50  (WEAPON_CLASS) — objects.py line 1184
# axe              : type_id 27  (WEAPON_CLASS) — objects.py line 724
# battle-axe       : type_id 28  (WEAPON_CLASS) — objects.py line 744
# ---------------------------------------------------------------------------
PICKAXE_TYPE_ID:    int = 234
MATTOCK_TYPE_ID:    int = 50
AXE_TYPE_ID:        int = 27
BATTLE_AXE_TYPE_ID: int = 28

# Direction constant for digging down.
DIG_DOWN: int = 8

# Vendor effort thresholds (dig.c:372, 440).
PIT_THRESHOLD:   int = 50    # down: effort > 50 → PIT phase begins
HOLE_THRESHOLD:  int = 250   # down: effort > 250 → HOLE + descend
WALL_THRESHOLD:  int = 100   # horizontal: effort > 100 → carve terrain

# Retained for backward compatibility with test_dig_hardness_lookup.
# The dig logic no longer uses this table — vendor uses a single global
# effort counter with fixed thresholds (see PIT/HOLE/WALL_THRESHOLD above).
# Cite: vendor/nethack/src/dig.c::dig (lines 365-368, 372, 440).
_HARDNESS_TABLE: jnp.ndarray = jnp.array(
    [
        0,    # VOID        = 0
        0,    # FLOOR       = 1
        0,    # CORRIDOR    = 2
        200,  # WALL        = 3   (legacy: tests still inspect this value)
        50,   # CLOSED_DOOR = 4
        0,    # OPEN_DOOR   = 5
        0,    # STAIRCASE_UP= 6
        0,    # STAIRCASE_DOWN=7
        0,    # WATER       = 8
        0,    # LAVA        = 9
        0,    # ALTAR       = 10
        0,    # FOUNTAIN    = 11
        0,    # TRAP        = 12
        0,    # HIDDEN_TRAP = 13
        0,    # THRONE      = 14
        0,    # GRAVE       = 15
        0,    # SHOP_FLOOR  = 16
        0,    # DRAWBRIDGE  = 17
        0,    # ICE_FLOOR   = 18
        0,    # POOL        = 19
        200,  # TREE        = 20
    ],
    dtype=jnp.int32,
)

# Direction delta table: dir_idx (0-7) -> (dy, dx).  8=DOWN has no delta.
_DIR_DY = jnp.array([-1,  0,  1,  0, -1,  1,  1, -1], dtype=jnp.int32)
_DIR_DX = jnp.array([ 0,  1,  0, -1,  1,  1, -1, -1], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# State struct
# ---------------------------------------------------------------------------

@struct.dataclass
class DigState:
    """Per-turn pickaxe/axe dig state.

    Mirrors ``svc.context.digging`` (vendor/nethack/src/decl.c::svctx,
    referenced by dig.c lines 303, 312, 365, 372, 541).  Vendor uses a single
    global ``effort`` counter with fixed thresholds (PIT > 50, HOLE > 250,
    WALL > 100); see PIT_THRESHOLD / HOLE_THRESHOLD / WALL_THRESHOLD above.

    Fields
    ------
    target_pos  : int16[2] — (row, col) of tile being dug; (-1, -1) if inactive.
                  Mirrors ``svc.context.digging.pos.{x,y}`` (dig.c:303).
    effort      : int32    — accumulated effort points this dig session.
                  Mirrors ``svc.context.digging.effort`` (dig.c:365).
    needed      : int32    — RETAINED for legacy tests; not used by the
                  threshold model.  Set to WALL_THRESHOLD on horizontal start,
                  HOLE_THRESHOLD on down-start so existing test inspections
                  observe a non-zero value.
    direction   : int8     — direction index (0-7 cardinal/intercardinal,
                  8=DOWN).  Mirrors ``svc.context.digging.down`` plus the
                  direction-of-attack carried in ``u.dx/u.dy``.
    active      : bool     — whether a dig is currently in progress.
                  Mirrors occupation == dig (dig.c:599).
    level       : int8     — branch+level the dig was started on (encoded as
                  ``branch * 100 + level``).  Cancels the dig if the hero
                  leaves the level.  Mirrors ``svc.context.digging.level``
                  (dig.c:312, 434-435, 543-544; on_level check).
    lastdigtime : int32    — timestep at which the most recent successful
                  swing landed.  Mirrors ``svc.context.digging.lastdigtime``
                  (dig.c:541).  Reserved for future use by occupation resume.
    """

    target_pos: jnp.ndarray   # int16[2]
    effort: jnp.ndarray       # int32
    needed: jnp.ndarray       # int32
    direction: jnp.ndarray    # int8
    active: jnp.ndarray       # bool
    level: jnp.ndarray        # int16  (branch*100 + level), -1 = inactive
    lastdigtime: jnp.ndarray  # int32

    @classmethod
    def default(cls) -> "DigState":
        return cls(
            target_pos=jnp.full((2,), -1, dtype=jnp.int16),
            effort=jnp.int32(0),
            needed=jnp.int32(0),
            direction=jnp.int8(0),
            active=jnp.bool_(False),
            level=jnp.int16(-1),
            lastdigtime=jnp.int32(0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terrain_hardness(tile_type: jnp.ndarray) -> jnp.ndarray:
    """Return effort needed to dig through ``tile_type``.

    Returns 0 for non-diggable tiles.
    Cite: vendor/nethack/src/dig.c::dig_typ (line ~445).
    """
    idx = jnp.clip(tile_type.astype(jnp.int32), 0, len(_HARDNESS_TABLE) - 1)
    return _HARDNESS_TABLE[idx]


def _current_level_terrain(state):
    """Extract the 2-D terrain slice for the player's current branch/level."""
    b = state.dungeon.current_branch
    lv = state.dungeon.current_level - jnp.int8(1)  # 1-based → 0-based
    return state.terrain[b, lv]


def _wielded_type_id(state) -> jnp.ndarray:
    """Return the type_id of the wielded item (-1 if bare hands)."""
    wslot = state.inventory.wielded.astype(jnp.int32)
    # If wslot == -1 (bare hands) clamp to slot 0 and return 0 (no item).
    safe_slot = jnp.clip(wslot, 0, None)
    raw_type_id = state.inventory.items.type_id[safe_slot]
    return jnp.where(wslot >= 0, raw_type_id.astype(jnp.int32), jnp.int32(-1))


def _is_pick(state) -> jnp.ndarray:
    """Return True if the wielded item is a pickaxe or mattock (P_PICK_AXE skill).

    Cite: vendor/nethack/include/obj.h line 220-222 (is_pick macro).
    """
    tid = _wielded_type_id(state)
    return (tid == jnp.int32(PICKAXE_TYPE_ID)) | (tid == jnp.int32(MATTOCK_TYPE_ID))


def _is_axe(state) -> jnp.ndarray:
    """Return True if the wielded item is an axe / battle-axe (P_AXE skill).

    Cite: vendor/nethack/include/obj.h line 217-219 (is_axe macro).
    """
    tid = _wielded_type_id(state)
    return (tid == jnp.int32(AXE_TYPE_ID)) | (tid == jnp.int32(BATTLE_AXE_TYPE_ID))


def _has_digging_tool(state) -> jnp.ndarray:
    """Return True if the wielded item can dig anything (pick or axe).

    Vendor accepts pickaxes for DIGTYP_ROCK (carve walls / down-dig) and any
    axe additionally for DIGTYP_DOOR / DIGTYP_TREE (dig.c::dig line 311).

    Cite: vendor/nethack/src/dig.c line 311 ``(!ispick && !is_axe(uwep))``.
    """
    return _is_pick(state) | _is_axe(state)


def _abon_from_str(player_str: jnp.ndarray) -> jnp.ndarray:
    """Mirror vendor abon() strength component (weapon.c:950-973).

    Returns the strength piece of abon() — we omit the dex adjustment and
    ulevel<3 game-tuning kludge because they're not load-bearing for dig
    parity and would couple this module to xl tracking.  Range: [-2, +3].

    Cite: vendor/nethack/src/weapon.c::abon lines 962-973.
    """
    s = player_str.astype(jnp.int32)
    # NetHack STR encoding: 6..17 normal, 18 = 18/00, 19..21 = 18/01..18/49,
    # 22 = 18/50, 23..28 = 18/51..18/99, 29 = 18/100, etc.  For the simplified
    # in-game encoding here we treat ``player_str`` directly as STR.  STR18(50)
    # ≈ 22 and STR18(100) ≈ 29 in NetHack's table; for raw-cap 18 STR these
    # branches collapse to sbon = 1.
    sbon = jnp.where(s < 6, jnp.int32(-2),
            jnp.where(s < 8, jnp.int32(-1),
             jnp.where(s < 17, jnp.int32(0),
              jnp.where(s < 22, jnp.int32(1),
               jnp.where(s < 29, jnp.int32(2), jnp.int32(3))))))
    return sbon


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _encoded_level(state) -> jnp.ndarray:
    """Encode (branch, level) as a single int16 = branch*100 + level.

    Used to mirror ``on_level(&svc.context.digging.level, &u.uz)`` checks
    (dig.c:312).  We pack the (dnum, dlevel) pair into a single int16 so
    the comparison stays a scalar equality.
    """
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32)
    return (b * jnp.int32(100) + lv).astype(jnp.int16)


def start_dig(state, direction: int):
    """Initiate a dig in ``direction``.

    Sets DigState.target_pos, direction, effort=0, level, active=True.  Is a
    no-op when the player wields no digging tool or when the target tile is
    not diggable for the wielded tool's dig-typ.

    Cite: vendor/nethack/src/dig.c::dig lines 311 (tool check) and 317-334
          (dig_typ / IS_TREE / IS_OBSTRUCTED gating).

    Parameters
    ----------
    state     : EnvState
    direction : int — 0-7 compass direction or 8 for down.
    """
    dir_i8 = jnp.int8(direction)

    is_pick = _is_pick(state)
    is_axe = _is_axe(state)
    has_tool = is_pick | is_axe

    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    dir_idx = jnp.int32(direction)
    is_down = dir_idx == jnp.int32(DIG_DOWN)

    dy = jnp.where(is_down, jnp.int32(0), _DIR_DY[jnp.clip(dir_idx, 0, 7)])
    dx = jnp.where(is_down, jnp.int32(0), _DIR_DX[jnp.clip(dir_idx, 0, 7)])

    trow = jnp.clip(row + dy, 0, state.terrain.shape[2] - 1)
    tcol = jnp.clip(col + dx, 0, state.terrain.shape[3] - 1)

    terrain_2d = _current_level_terrain(state)
    tile = terrain_2d[trow, tcol].astype(jnp.int32)

    # dig_typ gating (dig.c::dig_typ, called at lines 324, 329, 445, 549):
    #   Pickaxes can dig ROCK (WALL/STONE) and DOORs.  Axes can dig DOORs and
    #   TREEs only.  Down-dig is always permitted for any digging tool.
    is_wall  = tile == jnp.int32(TileType.WALL)
    is_door  = tile == jnp.int32(TileType.CLOSED_DOOR)
    is_tree  = tile == jnp.int32(TileType.TREE)

    horiz_diggable_by_pick = is_wall | is_door | is_tree
    horiz_diggable_by_axe  = is_door | is_tree
    is_diggable_horiz = (is_pick & horiz_diggable_by_pick) | (is_axe & horiz_diggable_by_axe)
    is_diggable = jnp.where(is_down, has_tool, is_diggable_horiz)

    can_start = has_tool & is_diggable

    # ``needed`` retained for legacy test inspection; reflect the active
    # threshold (250 down, 100 horizontal) so tests see a positive number.
    needed_val = jnp.where(is_down, jnp.int32(HOLE_THRESHOLD), jnp.int32(WALL_THRESHOLD))
    new_level = _encoded_level(state)

    new_dig = DigState(
        target_pos=jnp.where(
            can_start,
            jnp.array([trow, tcol], dtype=jnp.int16),
            state.dig.target_pos,
        ),
        effort=jnp.where(can_start, jnp.int32(0), state.dig.effort),
        needed=jnp.where(can_start, needed_val, state.dig.needed),
        direction=jnp.where(can_start, dir_i8, state.dig.direction),
        active=jnp.where(can_start, jnp.bool_(True), state.dig.active),
        level=jnp.where(can_start, new_level, state.dig.level),
        lastdigtime=jnp.where(can_start, state.timestep, state.dig.lastdigtime),
    )
    return state.replace(dig=new_dig)


def _is_maze_level(state) -> jnp.ndarray:
    """Best-effort proxy for ``svl.level.flags.is_maze_lev``.

    Vendor flags individual levels as maze; we lack that bit, so we use a
    branch heuristic: Gehennom (branch 5) is maze.

    Cite: vendor/nethack/src/dig.c lines 493-494 (is_maze_lev check).
    """
    return state.dungeon.current_branch.astype(jnp.int32) == jnp.int32(5)


def _is_cavernous_level(state) -> jnp.ndarray:
    """Read ``svl.level.flags.is_cavernous_lev`` for the current level.

    Wave 48b: replaced the Mines-branch proxy with a direct read of the
    per-level ``features.is_cavernous_lev`` bit, mirroring vendor/
    nethack/include/rm.h line 454 (``Bitfield(is_cavernous_lev, 1)``).
    The flag is set at level generation by mkmap.c line 483 (cave-shaped
    levels) and consumed by dig.c lines 495-497 to carve walls into
    CORR rather than D_NODOOR.

    Cite: vendor/nethack/src/dig.c lines 495-497 (is_cavernous_lev check);
          vendor/nethack/include/rm.h line 454 (levelflags bitfield);
          vendor/nethack/src/mkmap.c line 483 (cavernous setter).
    """
    max_levels = state.terrain.shape[1]  # MAX_LEVELS_PER_BRANCH
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    flat_lv = b * jnp.int32(max_levels) + lv
    return state.features.is_cavernous_lev[flat_lv]


def _complete_dig(state):
    """Finalise a completed dig: modify terrain or descend.

    Horizontal: carve the target tile.  Vendor splits the outcome on the
    *type* of dug terrain and the level kind (dig.c:488-501):

        WALL on maze level       → ROOM
        WALL on cavernous level  → CORR
        WALL otherwise           → DOOR (D_NODOOR; we model as OPEN_DOOR)
        TREE                     → ROOM (FLOOR equivalent here)
        STONE/SCORR              → CORR

    Down: create HOLE at player pos and advance player one level.
    Cite: vendor/nethack/src/dig.c::digactualhole (line 640) sets
          levl[u.ux][u.uy].typ = HOLE.
    """
    b = state.dungeon.current_branch
    lv = (state.dungeon.current_level - jnp.int8(1)).astype(jnp.int32)

    trow = state.dig.target_pos[0].astype(jnp.int32)
    tcol = state.dig.target_pos[1].astype(jnp.int32)

    is_down = state.dig.direction.astype(jnp.int32) == jnp.int32(DIG_DOWN)

    # --- horizontal outcome (dig.c:465-501) ---
    target_tile = state.terrain[b, lv, trow, tcol].astype(jnp.int32)
    is_wall = target_tile == jnp.int32(TileType.WALL)
    is_tree = target_tile == jnp.int32(TileType.TREE)
    is_door_t = target_tile == jnp.int32(TileType.CLOSED_DOOR)

    maze = _is_maze_level(state)
    cavernous = _is_cavernous_level(state)

    # WALL → ROOM (maze) / CORR (cavernous) / DOOR (D_NODOOR ≈ OPEN_DOOR else)
    wall_out = jnp.where(
        maze,
        jnp.int8(TileType.FLOOR),
        jnp.where(cavernous, jnp.int8(TileType.CORRIDOR), jnp.int8(TileType.OPEN_DOOR)),
    )
    # Closed door broken → OPEN_DOOR (D_BROKEN ≈ open) (dig.c:507-516).
    door_out = jnp.int8(TileType.OPEN_DOOR)
    # Tree cut → ROOM (FLOOR) (dig.c:478-479).
    tree_out = jnp.int8(TileType.FLOOR)
    # Stone / other rock → CORR (dig.c:485-486).
    default_out = jnp.int8(TileType.CORRIDOR)

    new_tile = jnp.where(
        is_wall, wall_out,
        jnp.where(is_door_t, door_out,
        jnp.where(is_tree, tree_out, default_out))
    )
    new_terrain_horiz = state.terrain.at[b, lv, trow, tcol].set(new_tile)

    # --- down outcome ---
    # Down dig completes only when effort > HOLE_THRESHOLD (caller's
    # responsibility).  Carve HOLE at player tile and advance one level.
    # Cite: vendor/nethack/src/dig.c::digactualhole line 640.
    prow = state.player_pos[0].astype(jnp.int32)
    pcol = state.player_pos[1].astype(jnp.int32)
    new_terrain_down = state.terrain.at[b, lv, prow, pcol].set(
        jnp.int8(TileType.HOLE)
    )

    new_terrain = jnp.where(is_down, new_terrain_down, new_terrain_horiz)

    max_level = jnp.int8(state.terrain.shape[1])
    new_level = jnp.where(
        is_down,
        jnp.minimum(max_level, state.dungeon.current_level + jnp.int8(1)),
        state.dungeon.current_level,
    )
    new_dungeon = state.dungeon.replace(current_level=new_level)

    return state.replace(terrain=new_terrain, dungeon=new_dungeon)


def _is_dwarf(state) -> jnp.ndarray:
    """Return True if the hero's race is PM_DWARF.

    Cite: vendor/nethack/src/dig.c line 367 ``if (Race_if(PM_DWARF))``.
    Our character.py stores ``player_race`` as a small role-table index, not
    the PM_DWARF monster id (356).  We treat race index 3 as dwarf to match
    races.py table ordering (Human=0, Elf=1, Gnome=2, Dwarf=3, ...).
    """
    return state.player_race.astype(jnp.int32) == jnp.int32(3)


def _wielded_spe(state) -> jnp.ndarray:
    """Return the enchantment bonus (``->spe``) of the wielded weapon.

    Cite: vendor/nethack/src/dig.c line 366 ``+ uwep->spe``.
    """
    wslot = state.inventory.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(wslot, 0, None)
    spe = state.inventory.items.enchantment[safe_slot].astype(jnp.int32)
    return jnp.where(wslot >= 0, spe, jnp.int32(0))


def _wielded_erosion(state) -> jnp.ndarray:
    """Return ``greatest_erosion(uwep)`` for the wielded weapon.

    Vendor's greatest_erosion returns max(rust, corrode, fire, rot) — caps at
    3.  Our inventory has a single ``erosion`` int8 already aggregating these,
    so we just clip to non-negative.

    Cite: vendor/nethack/src/dig.c line 366 ``- greatest_erosion(uwep)``;
          vendor/nethack/src/objnam.c::greatest_erosion.
    """
    wslot = state.inventory.wielded.astype(jnp.int32)
    safe_slot = jnp.clip(wslot, 0, None)
    items = state.inventory.items
    # Some inventory schemas store erosion under ``erosion`` or ``oeroded``;
    # fall back to 0 if neither is present.
    if hasattr(items, "erosion"):
        ero = items.erosion[safe_slot].astype(jnp.int32)
    elif hasattr(items, "oeroded"):
        ero = items.oeroded[safe_slot].astype(jnp.int32)
    else:
        ero = jnp.int32(0)
    ero = jnp.where(wslot >= 0, ero, jnp.int32(0))
    return jnp.clip(ero, 0, 3)


def _effort_gain(state, rng) -> jnp.ndarray:
    """Per-swing effort delta — vendor dig.c lines 365-368.

        effort += 10 + rn2(5) + abon() + uwep->spe
                       - greatest_erosion(uwep) + u.udaminc
        if (Race_if(PM_DWARF)) effort *= 2  # of the running sum

    Returns the per-swing delta (the multiplier on the running sum is applied
    at the call site to match vendor semantics).

    Cite: vendor/nethack/src/dig.c lines 365-368.
    """
    base = jnp.int32(10) + jax.random.randint(rng, (), 0, 5, dtype=jnp.int32)
    abon = _abon_from_str(state.player_str)
    spe = _wielded_spe(state)
    ero = _wielded_erosion(state)
    udam = state.player_udaminc.astype(jnp.int32)
    return base + abon + spe - ero + udam


def dig_tick(state, rng):
    """Advance one turn of an in-progress dig.

    Called every turn from ``_step_impl``.  Mirrors vendor's ``dig()``
    occupation callback (dig.c:299-568):

      1. Cancel if uswallow / no uwep / weapon isn't pick-or-axe (line 311).
      2. Cancel if target tile no longer adjacent (or for down-dig, no longer
         under the hero) and bail (line 313-314).
      3. Cancel if hero changed level (line 312, on_level check).
      4. If Fumbling and rn2(3)==0: 1/3 of those swings drops weapon, wakes
         monsters, or simply misses (lines 336-363) — the swing produces no
         effort.
      5. Otherwise effort += vendor formula (lines 365-366); if dwarf, the
         running sum is doubled (line 367-368).
      6. Down-dig: effort > 250 → HOLE + descend (line 372); effort <= 50 →
         still digging (line 380); 50 < effort <= 250 → PIT phase, no
         descent yet (vendor calls dighole(TRUE, ...) but the hero stays put
         this turn).
      7. Horizontal: effort > 100 → carve and complete (line 440).
    """
    def _tick(s):
        rng_local = rng

        is_down = s.dig.direction.astype(jnp.int32) == jnp.int32(DIG_DOWN)

        # --- cancel: weapon / swallow / adjacency / level (dig.c:311-314) ---
        has_tool = _has_digging_tool(s)
        # u.uswallow proxy: swallowed when SwallowState.active is True.
        if hasattr(s, "swallow") and hasattr(s.swallow, "active"):
            uswallow = s.swallow.active
        else:
            uswallow = jnp.bool_(False)

        prow = s.player_pos[0].astype(jnp.int32)
        pcol = s.player_pos[1].astype(jnp.int32)
        trow = s.dig.target_pos[0].astype(jnp.int32)
        tcol = s.dig.target_pos[1].astype(jnp.int32)
        dist = jnp.maximum(jnp.abs(prow - trow), jnp.abs(pcol - tcol))
        # Horizontal: !next2u(dpx, dpy) cancels (dig.c:314).
        not_adj   = (~is_down) & (dist > jnp.int32(1))
        # Down: must still be standing on target (dig.c:313).
        moved_off = is_down & ((prow != trow) | (pcol != tcol))

        # on_level check (dig.c:312).
        level_changed = _encoded_level(s) != s.dig.level

        cancel = uswallow | (~has_tool) | not_adj | moved_off | level_changed

        # --- fumble miss (dig.c:336-363): 1/3 of swings under Fumbling skip
        # effort.  Our status system may not expose Fumbling separately —
        # check StatusState if available.  When absent, assume not fumbling.
        rng_local, sub_fum, sub_eff = jax.random.split(rng_local, 3)
        if hasattr(s, "status") and hasattr(s.status, "fumbling"):
            fumbling = s.status.fumbling.astype(jnp.bool_)
        else:
            fumbling = jnp.bool_(False)
        # rn2(3) == 0 with probability 1/3.
        fum_roll = jax.random.randint(sub_fum, (), 0, 3, dtype=jnp.int32) == jnp.int32(0)
        miss_this_swing = fumbling & fum_roll

        # --- effort delta (dig.c:365-368) ---
        delta = _effort_gain(s, sub_eff)
        # Dwarves get the running sum doubled each swing (vendor line 367-368
        # multiplies the entire ``svc.context.digging.effort`` after adding
        # the delta).
        sum_after_add = s.dig.effort + delta
        sum_doubled = sum_after_add * jnp.int32(2)
        new_effort = jnp.where(_is_dwarf(s), sum_doubled, sum_after_add)
        # If we missed the swing this turn, no effort accrues.
        new_effort = jnp.where(miss_this_swing, s.dig.effort, new_effort)

        # --- thresholds ---
        # Horizontal completion: effort > 100 (dig.c:440).
        horiz_done = (~is_down) & (new_effort > jnp.int32(WALL_THRESHOLD))
        # Down: effort > 250 → HOLE + descend (dig.c:372).
        down_done  = is_down  & (new_effort > jnp.int32(HOLE_THRESHOLD))
        # Down: 50 < effort <= 250 → PIT phase (no descent); we continue
        # without level change.  effort <= 50 → still digging (line 380).
        # Both subcases keep the dig active with updated effort.
        done = horiz_done | down_done

        # --- Earth-level cavearea branch (dig.c:467-475).  Stub: we have no
        # Earth plane representation yet, so leave terrain unchanged but
        # consume the swing.  Mark via the lastdigtime update.  Cite:
        # vendor/nethack/src/dig.c::mkcavearea.  Hook left in place for
        # future Earth-plane wiring.

        # Build the "still digging" DigState for the in-progress branch.
        still_digging = DigState(
            target_pos=s.dig.target_pos,
            effort=new_effort,
            needed=s.dig.needed,
            direction=s.dig.direction,
            active=jnp.bool_(True),
            level=s.dig.level,
            lastdigtime=jnp.where(miss_this_swing, s.dig.lastdigtime, s.timestep),
        )
        reset_dig = DigState.default()

        completed_state = _complete_dig(s.replace(dig=still_digging)).replace(dig=reset_dig)
        cancelled_state = s.replace(dig=reset_dig)
        in_progress_state = s.replace(dig=still_digging)

        s1 = jax.tree_util.tree_map(
            lambda t, f: jnp.where(done, t, f), completed_state, in_progress_state
        )
        s2 = jax.tree_util.tree_map(
            lambda t, f: jnp.where(cancel, t, f), cancelled_state, s1
        )
        return s2

    ticked_state = _tick(state)
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(state.dig.active, t, f), ticked_state, state
    )


# ---------------------------------------------------------------------------
# Vendor-named public entry points
# ---------------------------------------------------------------------------

def dig(state, rng, direction: int = 0):
    """Vendor-named alias for the dig action — one swing of the pickaxe.

    Mirrors vendor/nethack/src/dig.c::dig (the occupation callback at line 425
    invoked once per turn).  In the JAX port the multi-turn occupation is
    split into ``start_dig`` (initiate) and ``dig_tick`` (per-turn effort).
    ``dig(state, rng, direction)`` exposes the vendor entry name: if no dig is
    active it starts one in the given direction; otherwise it advances the
    in-progress dig by one tick.

    Cite: vendor/nethack/src/dig.c::dig (line 425, occupation callback).
    """
    started = start_dig(state, direction=direction)
    ticked = dig_tick(state, rng)
    return jax.tree_util.tree_map(
        lambda t, f: jnp.where(state.dig.active, t, f), ticked, started
    )


def dig_down(state, rng):
    """Vendor-named wrapper for downward pickaxe-dig — create HOLE, descend.

    Mirrors vendor/nethack/src/dig.c::zap_dig (line 1548), specifically the
    ``u.dz > 0`` branch which routes through ``dighole`` → ``digactualhole``
    to set ``levl[u.ux][u.uy].typ = HOLE`` and drop the player one level.

    Cite: vendor/nethack/src/dig.c::zap_dig line 1548;
          vendor/nethack/src/dig.c::digactualhole line 640 (HOLE creation).
    """
    return dig(state, rng, direction=DIG_DOWN)
