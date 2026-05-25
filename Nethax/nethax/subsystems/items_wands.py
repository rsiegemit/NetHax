"""Wand effects + ray tracing — vendor/nethack/src/zap.c.

Canonical sources:
  vendor/nethack/src/zap.c  — dozap, zapyourself, zappable, buzz, bhitm
  vendor/nethack/include/objclass.h — oc_dir values (NODIR/IMMEDIATE/RAY)

Wave 3 status:
  Full dispatch table with 28 effects.  Ray-tracing via jax.lax.scan.
  Monster tables live in WandState (pos, hp, type, alive, asleep).
  Wave 4 will wire WandState.monsters into main EnvState.

Design notes:
  - All per-effect handlers are pure functions compatible with jax.jit.
  - jax.lax.switch is used for effect dispatch (static-shape requirement).
  - jax.lax.scan is used for ray stepping (no Python loops in jit path).
  - Ray stepping uses unit-step integer direction vectors (dx, dy in
    {-1, 0, 1}).  This matches vendor exactly: vendor/nethack/src/zap.c
    dobuzz (lines 4829-4833) and bhit (lines 3870-3877) advance one
    tile per loop iteration along (ddx, ddy) — there is no Bresenham
    line-drawing in NetHack's ray code.
  - Monster table is a flat array; slot 0 is reserved / always dead so
    "no hit" returns index 0 safely.
"""
from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.constants.monsters import (
    MONSTERS,
    M2_DEMON,
    M2_MAGIC,
    M2_SHAPESHIFTER,
    M2_UNDEAD,
    MonsterSymbol,
)
from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS,
    Item,
    InventoryState,
)
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL
from Nethax.nethax.subsystems.traps import TrapState, TrapType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WANDS: int = 28

# Total canonical monster count — len(MONSTERS).
N_MONSTERS: int = len(MONSTERS)

# Magic resistance flag for mon_resists bitmask.
# Uses bit 8 (0x100) to avoid int32 overflow; M2_MAGIC = 0x80000000 is
# too large for int32.  The _DEATH_IMMUNE table uses M2_MAGIC via Python
# (no overflow there); this constant is for runtime mon_resists checks.
# Cite: vendor/nethack/src/zap.c::zhitm ~4308 (resists_magm check).
MR_MAGIC: int = 0x00000100

# Maximum number of items in a wand-state inventory batch.
# (Reuses the global constant for shape consistency.)
_INV_SLOTS = MAX_INVENTORY_SLOTS

# ItemCategory value for wands (matches ObjectClass.WAND_CLASS = 11).
ITEM_CATEGORY_WAND: int = 11

# Tile that a frozen water tile becomes.
ICE_TILE: int = int(TileType.FLOOR)   # re-use FLOOR; Wave 4 can add TileType.ICE

# Tile that digging carves.
DIG_TILE: int = int(TileType.CORRIDOR)

# Maximum ray range (steps) before the beam dissipates.
DEFAULT_RAY_RANGE: int = 8

# Direction vectors (dy, dx) indexed 0-7:
#   0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
_DIR_DY = jnp.array([-1, -1,  0,  1,  1,  1,  0, -1], dtype=jnp.int16)
_DIR_DX = jnp.array([ 0,  1,  1,  1,  0, -1, -1, -1], dtype=jnp.int16)

# ---------------------------------------------------------------------------
# Precomputed per-monster tables (built at module load, not inside jit).
# ---------------------------------------------------------------------------

def _build_death_immune() -> jax.Array:
    """Bool mask: True if monster is immune to WAN_DEATH.

    Cite: vendor/nethack/src/zap.c::zhitm ~4308:
      if (nonliving(mon->data) || is_demon(mon->data)
          || is_vampshifter(mon) || resists_magm(mon)) { break; }

    nonliving = undead | golem (S_GOLEM) | vortex (S_VORTEX).
    vampshifter = M2_SHAPESHIFTER (vampire/vampire-lord/Vlad in shifted form).
    resists_magm proxy = M2_MAGIC (inherent magic resistance).
    """
    flags = []
    for m in MONSTERS:
        is_undead   = bool(m.flags2 & M2_UNDEAD)
        is_demon    = bool(m.flags2 & M2_DEMON)
        is_golem    = (m.symbol == MonsterSymbol.S_GOLEM)
        is_vortex   = (m.symbol == MonsterSymbol.S_VORTEX)
        is_vampshft = bool(m.flags2 & M2_SHAPESHIFTER)
        is_magicres = bool(m.flags2 & M2_MAGIC)
        flags.append(
            is_undead or is_demon or is_golem or is_vortex or is_vampshft or is_magicres
        )
    return jnp.array(flags, dtype=jnp.bool_)


# bool[N_MONSTERS] — JIT-pure via jnp.take in _effect_death.
_DEATH_IMMUNE: jax.Array = _build_death_immune()

# int8[N_MONSTERS] — generation level proxy (vendor mons.c gen_level).
# Uses MonsterEntry.level as the difficulty proxy.
# Cite: vendor/nethack/src/zap.c wand_create_monster level-appropriate logic.
_MONSTER_GEN_LEVEL: jax.Array = jnp.array(
    [m.level for m in MONSTERS], dtype=jnp.int8
)

# Byte-equal new-form HP roll lives in polymorph._form_hp_max (imported at
# the call site to avoid a circular module load).  It mirrors vendor
# makemon.c:1012-1054 newmonhp:
#   mlvl==0           → rnd(4)
#   mlvl> 0           → d(mlvl, 8)   (+ home_elemental*3, dragon/golem branches)
# Cite: vendor/nethack/src/zap.c:5373 newcham → newmonhp(mtmp, monsndx(mdat));
#       vendor/nethack/src/makemon.c:1012 newmonhp.


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WandEffect(IntEnum):
    """Wand effect indices — must match the dispatch table order in zap_wand.

    Ordering mirrors vendor/nethack/include/objects.h WAN_* defines.
    """
    LIGHT                = 0
    NOTHING              = 1
    SECRET_DOOR_DETECTION = 2
    OPENING              = 3
    LOCKING              = 4
    PROBING              = 5
    MAGIC_MISSILE        = 6
    STRIKING             = 7
    SLOW_MONSTER         = 8
    SPEED_MONSTER        = 9
    CANCELLATION         = 10
    POLYMORPH            = 11
    TELEPORTATION        = 12
    DEATH                = 13
    SLEEP                = 14
    COLD                 = 15
    FIRE                 = 16
    LIGHTNING            = 17
    DIGGING              = 18
    ENLIGHTENMENT        = 19
    CREATE_MONSTER       = 20
    WISHING              = 21
    STASIS               = 22
    MAKE_INVISIBLE       = 23
    UNDEAD_TURNING       = 24
    DRAINING             = 25
    ACID                 = 26
    POISON_GAS           = 27


class WandClass(IntEnum):
    """Ray behaviour class for each wand effect.

    SELF         — no direction; immediate self effect (light, secret-door-detect)
    NODIR        — immediate area effect (opening of all locks visible)
    RAY          — beam that travels full range, can hit multiple targets
    BEAM         — stops at first target hit
    AT_LOCATION  — acts on a map tile (digging)
    """
    SELF        = 0
    NODIR       = 1
    RAY         = 2
    BEAM        = 3
    AT_LOCATION = 4


# Map each WandEffect → WandClass (used by zap_wand to choose dispatch path).
_EFFECT_CLASS: list[int] = [
    WandClass.SELF,          # LIGHT
    WandClass.NODIR,         # NOTHING
    WandClass.SELF,          # SECRET_DOOR_DETECTION
    WandClass.NODIR,         # OPENING
    WandClass.NODIR,         # LOCKING
    WandClass.NODIR,         # PROBING
    WandClass.RAY,           # MAGIC_MISSILE
    WandClass.BEAM,          # STRIKING
    WandClass.RAY,           # SLOW_MONSTER
    WandClass.RAY,           # SPEED_MONSTER
    WandClass.BEAM,          # CANCELLATION
    WandClass.BEAM,          # POLYMORPH
    WandClass.BEAM,          # TELEPORTATION
    WandClass.RAY,           # DEATH
    WandClass.RAY,           # SLEEP
    WandClass.RAY,           # COLD
    WandClass.RAY,           # FIRE
    WandClass.RAY,           # LIGHTNING
    WandClass.AT_LOCATION,   # DIGGING
    WandClass.SELF,          # ENLIGHTENMENT
    WandClass.SELF,          # CREATE_MONSTER
    WandClass.SELF,          # WISHING
    WandClass.BEAM,          # STASIS
    WandClass.BEAM,          # MAKE_INVISIBLE
    WandClass.RAY,           # UNDEAD_TURNING
    WandClass.BEAM,          # DRAINING
    WandClass.BEAM,          # ACID
    WandClass.RAY,           # POISON_GAS
]

WAND_EFFECT_CLASS = jnp.array(_EFFECT_CLASS, dtype=jnp.int8)


# ---------------------------------------------------------------------------
# WandState — self-contained state slice for wand operations
# ---------------------------------------------------------------------------

@struct.dataclass
class WandState:
    """Minimal game-state slice consumed by all wand functions.

    Arrays with shape [MAX_MONSTERS_PER_LEVEL] represent the current level's
    monster table.  Slot 0 is a sentinel (always alive=False) so "no hit"
    returns a safe default index.

    Fields
    ------
    mon_pos       : (row, col) for each monster.  int16[N, 2].
    mon_hp        : current hit points.           int32[N].
    mon_type      : monster species index.        int16[N].
    mon_alive     : whether the slot is occupied. bool[N].
    mon_asleep    : whether the monster is asleep. bool[N].
    mon_undead    : whether the monster is undead. bool[N].
    mon_invisible : whether the monster is invisible. bool[N].
    mon_resists   : per-monster resistance bitmask (MR_FIRE etc). int32[N].
    mon_speed_mod : speed modifier (-1=slowed, +1=hasted). int8[N].
    mon_cancelled : cancellation flag. bool[N].

    terrain       : current-level terrain tiles.  int8[MAP_H, MAP_W].
    explored      : explored mask for lighting.   bool[MAP_H, MAP_W].

    inventory     : player inventory (items + charges).

    player_pos    : (row, col).  int16[2].
    dungeon_level : current dungeon depth (1-based). int8 scalar.

    probed_hp     : HP of last WAN_PROBING hit. int32 scalar.
    probed_idx    : slot index of last WAN_PROBING hit. int32 scalar.
    """
    mon_pos:       jax.Array   # int16[N, 2]
    mon_hp:        jax.Array   # int32[N]
    mon_hp_max:    jax.Array   # int32[N]  — used by drain_life HP-max reduction
    mon_type:      jax.Array   # int16[N]
    mon_alive:     jax.Array   # bool[N]
    mon_asleep:    jax.Array   # bool[N]
    mon_undead:    jax.Array   # bool[N]
    mon_invisible: jax.Array   # bool[N]
    mon_resists:   jax.Array   # int32[N]
    mon_speed_mod: jax.Array   # int8[N]
    mon_cancelled: jax.Array   # bool[N]
    # mon_paralyzed_timer mirrors MonsterAIState.paralyzed_timer; used by
    # WAN_SLEEP / WAN_STRIKING etc. to set vendor mfrozen duration.
    mon_paralyzed_timer: jax.Array  # int16[N]

    terrain:       jax.Array   # int8[MAP_H, MAP_W]
    explored:      jax.Array   # bool[MAP_H, MAP_W]

    inventory:     InventoryState

    player_pos:    jax.Array   # int16[2]
    dungeon_level: jax.Array   # int8 scalar (1-based)

    probed_hp:     jax.Array   # int32 scalar
    probed_idx:    jax.Array   # int32 scalar

    # Reflection intrinsic — when True, rays hitting player are bounced back.
    # Cite: vendor/nethack/src/artifact.c::arti_prop AMULET_OF_REFLECTION +
    # zap.c::buzz reflection path.
    player_reflecting: jax.Array  # bool scalar

    # Wave 48e: dungeon branch + traps + wall_info slices for full
    # vendor dig_check fail-code coverage in _effect_digging.
    # Cite: vendor/nethack/src/dig.c::dig_check lines 207-260
    #   DIG_FAIL_AIRLEVEL / DIG_FAIL_WATERLEVEL : branch-based
    #   DIG_FAIL_UNDESTROYABLETRAP              : MAGIC_PORTAL / VIBRATING_SQUARE
    #   DIG_FAIL_W_NONDIGGABLE                  : per-tile wall_info flag
    branch:        jax.Array   # int8 scalar — dungeon Branch ordinal
    traps:         TrapState   # full per-level TrapState slice (or zeros)
    wall_info:     jax.Array   # bool[num_levels, map_h, map_w] — W_NONDIGGABLE

    @classmethod
    def empty(cls, map_h: int = 21, map_w: int = 80) -> "WandState":
        """Return a zero-initialised WandState (no monsters, empty inventory).

        The trap/wall_info slices are sized for a single-level test fixture
        (num_levels=1).  Real callers (action_dispatch._handle_zap) project
        the full multi-level EnvState.traps / wall_info arrays directly.
        """
        n = MAX_MONSTERS_PER_LEVEL
        return cls(
            mon_pos=jnp.zeros((n, 2), dtype=jnp.int16),
            mon_hp=jnp.zeros(n, dtype=jnp.int32),
            mon_hp_max=jnp.zeros(n, dtype=jnp.int32),
            mon_type=jnp.zeros(n, dtype=jnp.int16),
            mon_alive=jnp.zeros(n, dtype=bool),
            mon_asleep=jnp.zeros(n, dtype=bool),
            mon_undead=jnp.zeros(n, dtype=bool),
            mon_invisible=jnp.zeros(n, dtype=bool),
            mon_resists=jnp.zeros(n, dtype=jnp.int32),
            mon_speed_mod=jnp.zeros(n, dtype=jnp.int8),
            mon_cancelled=jnp.zeros(n, dtype=bool),
            mon_paralyzed_timer=jnp.zeros(n, dtype=jnp.int16),
            terrain=jnp.zeros((map_h, map_w), dtype=jnp.int8),
            explored=jnp.zeros((map_h, map_w), dtype=bool),
            inventory=InventoryState.empty(),
            player_pos=jnp.zeros(2, dtype=jnp.int16),
            dungeon_level=jnp.int8(1),
            probed_hp=jnp.int32(0),
            probed_idx=jnp.int32(0),
            player_reflecting=jnp.bool_(False),
            branch=jnp.int8(0),
            traps=TrapState.default(num_levels=1, map_h=map_h, map_w=map_w),
            wall_info=jnp.zeros((1, map_h, map_w), dtype=jnp.bool_),
        )


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def _rng_d(rng: jax.Array, n_dice: int, sides: int) -> tuple[jax.Array, jax.Array]:
    """Roll n_dice d<sides>.  Returns (new_rng, total)."""
    rng, sub = jax.random.split(rng)
    rolls = jax.random.randint(sub, shape=(n_dice,), minval=1, maxval=sides + 1)
    return rng, jnp.sum(rolls, dtype=jnp.int32)


def _rng_rnd(rng: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Return a random int in [1, n].  Equivalent to NetHack rnd(n)."""
    rng, sub = jax.random.split(rng)
    return rng, jax.random.randint(sub, shape=(), minval=1, maxval=n + 1)


# ---------------------------------------------------------------------------
# Ray-tracing core
# ---------------------------------------------------------------------------

def _find_monster_at(state: WandState, pos: jax.Array) -> jax.Array:
    """Return the index of a live monster at *pos*, else 0.

    Uses a vectorised equality check over all slots; returns the first match
    (lowest index).  Slot 0 is the sentinel / no-hit value.
    """
    match_row = state.mon_pos[:, 0] == pos[0]
    match_col = state.mon_pos[:, 1] == pos[1]
    occupied  = state.mon_alive & match_row & match_col
    # argmax over bool: first True index; 0 if none (sentinel).
    return jnp.argmax(occupied).astype(jnp.int32)


def cast_ray(
    state: WandState,
    rng: jax.Array,
    start_pos: jax.Array,
    direction: int | jax.Array,
    ray_range: int = DEFAULT_RAY_RANGE,
    on_hit_fn=None,
    stop_on_hit: bool = False,
) -> tuple[WandState, jax.Array]:
    """Walk a ray from *start_pos* in *direction* for up to *ray_range* steps.

    At each step:
      1. Advance position by direction delta.
      2. Clip to map bounds.
      3. If the tile is a wall (WALL), the beam is absorbed; stop.
      4. If a live monster occupies the tile, call on_hit_fn(state, rng, pos).
      5. If stop_on_hit=True (BEAM behaviour), stop after the first monster.

    Parameters
    ----------
    state       : current WandState
    rng         : JAX PRNGKey
    start_pos   : int16[2] (row, col) — player or source position
    direction   : int in [0, 7] corresponding to _DIR_DY / _DIR_DX
    ray_range   : maximum number of steps (default 8)
    on_hit_fn   : callable(state, rng, monster_idx) -> (state, rng)
                  Called when the beam hits a live monster.
    stop_on_hit : if True, beam stops at first monster hit (BEAM class)

    Returns
    -------
    (new_state, rng)
    """
    dy0 = _DIR_DY[direction].astype(jnp.int16)
    dx0 = _DIR_DX[direction].astype(jnp.int16)

    map_h, map_w = state.terrain.shape

    if on_hit_fn is None:
        def on_hit_fn(s, r, _idx):
            return s, r

    def _step(carry, _step_i):
        s, r, pos, dy, dx, stopped, reflected = carry

        next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)

        # Clip to valid map range.
        next_row = jnp.clip(next_pos[0], 0, map_h - 1)
        next_col = jnp.clip(next_pos[1], 0, map_w - 1)
        next_pos = jnp.array([next_row, next_col], dtype=jnp.int16)

        # Abort if out-of-bounds move would require wrapping (hit edge).
        oob = (next_pos[0] != pos[0] + dy) | (next_pos[1] != pos[1] + dx)

        tile = s.terrain[next_pos[0], next_pos[1]].astype(jnp.int32)
        is_wall = tile == int(TileType.WALL)

        mon_idx = _find_monster_at(s, next_pos)
        has_monster = (mon_idx > 0) & s.mon_alive[mon_idx]

        # Reflection: when the beam reaches a reflecting player, reverse the
        # travel direction once. vendor/nethack/src/zap.c::buzz reflection.
        hits_player = (
            (next_pos[0] == s.player_pos[0])
            & (next_pos[1] == s.player_pos[1])
        )
        do_reflect = (~stopped) & (~reflected) & hits_player & s.player_reflecting
        new_dy = jnp.where(do_reflect, -dy, dy)
        new_dx = jnp.where(do_reflect, -dx, dx)
        new_reflected = reflected | do_reflect

        # Apply effect when there is a live monster and we are not stopped.
        def _apply(args):
            _s, _r = args
            return on_hit_fn(_s, _r, mon_idx)

        def _noop(args):
            return args

        s, r = lax.cond(
            (~stopped) & has_monster,
            _apply,
            _noop,
            (s, r),
        )

        # Determine whether the beam should halt after this step.
        beam_stopped = stopped | is_wall | oob
        beam_stopped = lax.cond(
            (~stopped) & has_monster & stop_on_hit,
            lambda _: jnp.bool_(True),
            lambda _: beam_stopped,
            None,
        )

        return (s, r, next_pos, new_dy, new_dx, beam_stopped, new_reflected), None

    init_carry = (
        state,
        rng,
        start_pos.astype(jnp.int16),
        dy0,
        dx0,
        jnp.bool_(False),
        jnp.bool_(False),
    )
    (final_state, final_rng, _, _, _, _, _), _ = lax.scan(
        _step, init_carry, jnp.arange(ray_range)
    )
    return final_state, final_rng


# ---------------------------------------------------------------------------
# Per-effect helper: damage a single monster
# ---------------------------------------------------------------------------

def _deal_damage(state: WandState, mon_idx: jax.Array, dmg: jax.Array) -> WandState:
    """Subtract *dmg* HP from monster *mon_idx*; mark dead if HP <= 0."""
    new_hp   = state.mon_hp.at[mon_idx].add(-dmg)
    new_hp   = jnp.maximum(new_hp, 0)
    is_dead  = new_hp[mon_idx] <= 0
    new_alive = state.mon_alive.at[mon_idx].set(
        state.mon_alive[mon_idx] & ~is_dead
    )
    return state.replace(mon_hp=new_hp, mon_alive=new_alive)


# ---------------------------------------------------------------------------
# Per-effect handlers (all pure, jit-compatible)
# ---------------------------------------------------------------------------

def _effect_light(state: WandState, rng: jax.Array) -> tuple[WandState, jax.Array]:
    """WAN_LIGHT — illuminate a radius-5 disk around the player.

    Cite: vendor/nethack/src/read.c::litroom line 2601 calls
      do_clear_area(u.ux, u.uy, blessed_effect ? 9 : 5, set_lit, ...)
    so an uncursed wand of light lights tiles within Euclidean-disc radius
    5 around the hero (vendor/nethack/src/vision.c::do_clear_area uses
    ``circle_ptr(range)`` to compute the row-major disc limits).

    Implementation: mark every tile whose squared distance from the player
    is <= 5*5 as explored.  JIT-pure (broadcast comparison; no Python loop).
    """
    map_h, map_w = state.explored.shape
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    rows = jnp.arange(map_h, dtype=jnp.int32)[:, None]
    cols = jnp.arange(map_w, dtype=jnp.int32)[None, :]
    dy = rows - pr
    dx = cols - pc
    # Euclidean disc with vendor uncursed radius=5.
    in_disc = (dy * dy + dx * dx) <= jnp.int32(5 * 5)
    new_explored = state.explored | in_disc
    return state.replace(explored=new_explored), rng


def _effect_nothing(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 0
) -> tuple[WandState, jax.Array]:
    """WAN_NOTHING — no effect.  Charges were still spent."""
    return state, rng


def _effect_secret_door_detection(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_SECRET_DOOR_DETECTION — reveal SDOOR/SCORR within BOLT_LIM of @.

    Cite: vendor/nethack/src/detect.c::findit line 1815:
      do_clear_area(u.ux, u.uy, BOLT_LIM, findone, ...)
    where BOLT_LIM = 8 (vendor/nethack/include/hack.h line 49).  findone()
    converts SDOOR -> DOOR (closed) and SCORR -> CORR, but only at tiles
    visited by do_clear_area's disc walk (i.e. within radius 8 of @).

    Implementation: mask the SDOOR/SCORR replacement by a radius-8 disc
    centred on player_pos so tiles further away keep their hidden type.
    JIT-pure (broadcast comparison; no Python loop).
    """
    from Nethax.nethax.constants.tiles import VendorTileType

    map_h, map_w = state.terrain.shape
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    rows = jnp.arange(map_h, dtype=jnp.int32)[:, None]
    cols = jnp.arange(map_w, dtype=jnp.int32)[None, :]
    dy = rows - pr
    dx = cols - pc
    # BOLT_LIM = 8 — vendor/nethack/include/hack.h line 49.
    in_disc = (dy * dy + dx * dx) <= jnp.int32(8 * 8)

    is_sdoor = (state.terrain == jnp.int8(int(VendorTileType.SDOOR))) & in_disc
    is_scorr = (state.terrain == jnp.int8(int(VendorTileType.SCORR))) & in_disc
    new_terrain = jnp.where(is_sdoor,
                            jnp.int8(int(TileType.CLOSED_DOOR)),
                            state.terrain)
    new_terrain = jnp.where(is_scorr,
                            jnp.int8(int(TileType.CORRIDOR)),
                            new_terrain)
    return state.replace(terrain=new_terrain), rng


def _effect_opening(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_OPENING — IMMEDIATE beam: opens the first closed door along path.

    Cite: vendor/nethack/src/zap.c::bhit line 4056-4074 — when bhit walks a
    ZAPPED_WAND over a tile with IS_DOOR(typ) or typ == SDOOR, it calls
    doorlock(obj, x, y) for WAN_OPENING/LOCKING/STRIKING.  The handler
    doorlock() (vendor/nethack/src/lock.c::doorlock line 1102) translates
    door state per wand type.

    Container handling: vendor/nethack/src/zap.c::bhitpile dispatches each
    object at the tile to boxlock() (vendor/nethack/src/lock.c line 1056),
    which sets obj->olocked = 0 for WAN_OPENING.  Our WandState does not
    carry the level's floor-object table (containers live in
    ``state.containers`` on the full EnvState).  Floor-container unlocking
    therefore happens on the EnvState dispatch path, not here.  See
    Nethax/nethax/subsystems/containers.py::open_container.

    Door tile mapping in our local enum:
      * Our TileType.CLOSED_DOOR collapses vendor's {D_CLOSED, D_LOCKED}
        because we lack a per-tile doormask bit.  Opening therefore
        transitions CLOSED_DOOR -> OPEN_DOOR (which subsumes the vendor
        D_LOCKED -> D_CLOSED unlock).
    """
    def on_hit_door(s, r, pos):
        # Open the door at this tile.
        tr, tc = pos[0], pos[1]
        cur = s.terrain[tr, tc]
        is_closed = cur == jnp.int8(int(TileType.CLOSED_DOOR))
        new_t = jnp.where(is_closed,
                          jnp.int8(int(TileType.OPEN_DOOR)),
                          cur)
        new_terrain = s.terrain.at[tr, tc].set(new_t)
        return s.replace(terrain=new_terrain), r

    # cast_ray walks tiles; stop at first door (any closed door).
    return _cast_ray_terrain_predicate(
        state, rng, direction,
        target=int(TileType.CLOSED_DOOR),
        on_tile_fn=on_hit_door,
    )


def _effect_locking(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_LOCKING — IMMEDIATE beam: closes the first open door along path.

    Cite: vendor/nethack/src/lock.c::doorlock line 1135 WAN_LOCKING branch —
    transitions D_ISOPEN -> D_LOCKED (with D_CLOSED -> D_LOCKED for already
    closed doors).  Our local enum collapses D_LOCKED and D_CLOSED into
    TileType.CLOSED_DOOR, so we simply transition OPEN_DOOR -> CLOSED_DOOR.

    Floor-container locking: vendor/nethack/src/lock.c::boxlock line 1061
    sets obj->olocked = 1.  As with WAN_OPENING above, floor containers live
    in ``state.containers`` on the full EnvState and are not part of the
    minimal WandState consumed by this handler.
    """
    def on_hit_door(s, r, pos):
        tr, tc = pos[0], pos[1]
        cur = s.terrain[tr, tc]
        is_open = cur == jnp.int8(int(TileType.OPEN_DOOR))
        new_t = jnp.where(is_open,
                          jnp.int8(int(TileType.CLOSED_DOOR)),
                          cur)
        new_terrain = s.terrain.at[tr, tc].set(new_t)
        return s.replace(terrain=new_terrain), r

    return _cast_ray_terrain_predicate(
        state, rng, direction,
        target=int(TileType.OPEN_DOOR),
        on_tile_fn=on_hit_door,
    )


def _cast_ray_terrain_predicate(state, rng, direction, target, on_tile_fn,
                                max_range: int = 13):
    """Walk a ray from player_pos in `direction` up to `max_range` tiles,
    firing `on_tile_fn(state, rng, pos)` on the first tile matching `target`.

    Vendor: zap.c::buzz line 4823 — rn1(7,7) range = 7..13. We use the upper
    bound here (worst case), gated by tile match.

    JIT-pure via lax.fori_loop with a `done` flag.
    """
    dir_table = jnp.array([
        [-1,  0],  # N
        [-1,  1],  # NE
        [ 0,  1],  # E
        [ 1,  1],  # SE
        [ 1,  0],  # S
        [ 1, -1],  # SW
        [ 0, -1],  # W
        [-1, -1],  # NW
    ], dtype=jnp.int32)
    dir_idx = jnp.clip(jnp.asarray(direction, jnp.int32), 0, 7)
    dy = dir_table[dir_idx, 0]
    dx = dir_table[dir_idx, 1]
    map_h, map_w = state.terrain.shape
    start_r = state.player_pos[0].astype(jnp.int32)
    start_c = state.player_pos[1].astype(jnp.int32)
    target_t = jnp.int8(int(target))

    def body(i, carry):
        s, r, done = carry
        step = i + jnp.int32(1)
        tr = start_r + dy * step
        tc = start_c + dx * step
        in_bounds = (tr >= 0) & (tr < map_h) & (tc >= 0) & (tc < map_w)
        # Safe-clamped read.
        rr = jnp.clip(tr, 0, map_h - 1)
        cc = jnp.clip(tc, 0, map_w - 1)
        cur = s.terrain[rr, cc]
        matches = in_bounds & (cur == target_t) & ~done
        def _do_hit(args):
            ss, rr_ = args
            return on_tile_fn(ss, rr_, jnp.array([rr, cc], dtype=jnp.int32))
        s_new, r_new = jax.lax.cond(matches, _do_hit, lambda a: a, (s, r))
        return s_new, r_new, done | matches

    final_state, final_rng, _ = jax.lax.fori_loop(
        0, max_range, body, (state, rng, jnp.bool_(False)))
    return final_state, final_rng


def _effect_probing(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_PROBING — IMMEDIATE beam, reveal first monster hit.

    Cite: vendor/nethack/src/zap.c::probe_monster line 626:
      mstatusline(mtmp);
      ... if (mtmp->minvent) probe_objchain(mtmp->minvent); ...
    The vendor effect prints the monster's HP status line and inventory.
    We persist the essential numeric state — current HP and the slot index
    — into ``state.probed_hp`` / ``state.probed_idx`` for inspection by the
    observation layer.  The vendor message-line output is a UI-only effect
    and is intentionally not part of the JIT-pure state slice.
    """
    def on_hit(s, r, mon_idx):
        hp = s.mon_hp[mon_idx]
        return s.replace(
            probed_hp=hp.astype(jnp.int32),
            probed_idx=mon_idx.astype(jnp.int32),
        ), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_magic_missile(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_MAGIC_MISSILE — RAY, 2d6 damage per monster hit.

    Cite: vendor/nethack/src/zap.c:3464-3465: nd=2 for WAN_MAGIC_MISSILE.
    Immune if mon_resists has MR_MAGIC flag (proxy for resists_magm).
    Cite: vendor/nethack/src/zap.c::zhitm ~4252 (resists_magm gate).
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 2, 6)
        is_immune = (s.mon_resists[mon_idx] & MR_MAGIC).astype(jnp.bool_)
        actual_dmg = jnp.where(is_immune, jnp.int32(0), dmg)
        s = _deal_damage(s, mon_idx, actual_dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_striking(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_STRIKING — BEAM, 2d12 damage to first monster hit.

    vendor/nethack/src/zap.c bhitm() WAN_STRIKING: d(2,12).
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 2, 12)
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_slow_monster(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_SLOW_MONSTER — RAY, set mon_speed_mod = -1 on each hit monster.

    Cite: vendor/nethack/src/zap.c mon_adjust_speed ~line 4400.
    """
    def on_hit(s, r, mon_idx):
        new_spd = s.mon_speed_mod.at[mon_idx].set(jnp.int8(-1))
        return s.replace(mon_speed_mod=new_spd), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_speed_monster(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_SPEED_MONSTER — RAY, set mon_speed_mod = +1 on each hit monster.

    Cite: vendor/nethack/src/zap.c mon_adjust_speed ~line 4400.
    """
    def on_hit(s, r, mon_idx):
        new_spd = s.mon_speed_mod.at[mon_idx].set(jnp.int8(1))
        return s.replace(mon_speed_mod=new_spd), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_cancellation(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_CANCELLATION — BEAM, set mon_cancelled flag on first hit monster.

    Cite: vendor/nethack/src/zap.c cancel_monst ~line 4500.
    """
    def on_hit(s, r, mon_idx):
        new_canc = s.mon_cancelled.at[mon_idx].set(jnp.bool_(True))
        return s.replace(mon_cancelled=new_canc), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_polymorph(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_POLYMORPH — BEAM, change monster type via newcham().

    Cite: vendor/nethack/src/zap.c bhitm() WAN_POLYMORPH (~line 263):
      newcham(mtmp, NULL, ncflags) — picks eligible form via
      select_newcham_form which filters G_UNIQ and M2_NOPOLY.

    Implementation mirrors polymorph.py::polymorph_monster (zap.c::newcham):
      1. Rejection-sample a valid form using _POLY_FORM_VALID.
      2. Roll new_hp_max via polymorph._form_hp_max (byte-equal newmonhp:
         rnd(4) for mlvl==0; d(mlvl, 8) otherwise; +home_elemental*3 /
         dragon / golem branches).  Cite: vendor/nethack/src/zap.c:5373
         newcham → newmonhp; makemon.c:1012-1054.
      3. Scale current HP proportionally: new_hp = cur_hp * new_max / old_max.
      4. Update entry_idx (mon_type) and hp arrays.
    """
    from Nethax.nethax.subsystems.polymorph import _POLY_FORM_VALID, _form_hp_max

    def on_hit(s, r, mon_idx):
        # Rejection-sample a _POLY_FORM_VALID index in [1, N_MONSTERS-1].
        def _cond(ws):
            _, candidate = ws
            return ~_POLY_FORM_VALID[candidate]

        def _body(ws):
            r_, _ = ws
            r_, sub = jax.random.split(r_)
            c = jax.random.randint(sub, shape=(), minval=1,
                                   maxval=N_MONSTERS, dtype=jnp.int32)
            return (r_, c)

        r, sub0 = jax.random.split(r)
        init_c = jax.random.randint(sub0, shape=(), minval=1,
                                    maxval=N_MONSTERS, dtype=jnp.int32)
        r, new_type = lax.while_loop(_cond, _body, (r, init_c))
        new_type = new_type.astype(jnp.int32)

        # Byte-equal newmonhp roll (vendor/nethack/src/makemon.c:1012).
        r, sub_hp = jax.random.split(r)
        new_hp_max = _form_hp_max(new_type.astype(jnp.int16),
                                  sub_hp).astype(jnp.int32)
        old_hp_max = jnp.maximum(s.mon_hp_max[mon_idx].astype(jnp.float32),
                                 jnp.float32(1.0))
        ratio  = s.mon_hp[mon_idx].astype(jnp.float32) / old_hp_max
        new_hp = jnp.maximum(jnp.int32(1),
                             (ratio * new_hp_max.astype(jnp.float32)).astype(jnp.int32))

        s = s.replace(
            mon_type=s.mon_type.at[mon_idx].set(new_type.astype(jnp.int16)),
            mon_hp=s.mon_hp.at[mon_idx].set(new_hp),
            mon_hp_max=s.mon_hp_max.at[mon_idx].set(new_hp_max),
        )
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_teleportation(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_TELEPORTATION — BEAM, monster pos → random FLOOR or CORRIDOR tile.

    Cite: vendor/nethack/src/zap.c::u_teleport_mon — picks typ==ROOM or CORR.
    Uses rejection sampling under lax.while_loop to guarantee walkable dest.
    JIT-pure.
    """
    map_h, map_w = state.terrain.shape
    _floor    = jnp.int8(_TILE_FLOOR)
    _corridor = jnp.int8(_TILE_CORRIDOR)

    def on_hit(s, r, mon_idx):
        # Rejection-sample a walkable tile.
        def _cond(wstate):
            _, _, row, col = wstate
            tile = s.terrain[row, col].astype(jnp.int8)
            return (tile != _floor) & (tile != _corridor)

        def _body(wstate):
            r_, _, _, _ = wstate
            r_, sub_r = jax.random.split(r_)
            row = jax.random.randint(sub_r, shape=(), minval=0, maxval=map_h)
            r_, sub_c = jax.random.split(r_)
            col = jax.random.randint(sub_c, shape=(), minval=0, maxval=map_w)
            return (r_, jnp.int32(0), row, col)

        # Seed initial candidate.
        r, sub0 = jax.random.split(r)
        init_row = jax.random.randint(sub0, shape=(), minval=0, maxval=map_h)
        r, sub1 = jax.random.split(r)
        init_col = jax.random.randint(sub1, shape=(), minval=0, maxval=map_w)

        r, _, dest_row, dest_col = lax.while_loop(
            _cond, _body, (r, jnp.int32(0), init_row, init_col)
        )
        new_pos = jnp.array([dest_row, dest_col], dtype=jnp.int16)
        new_mon_pos = s.mon_pos.at[mon_idx].set(new_pos)
        return s.replace(mon_pos=new_mon_pos), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


# Walkable tile int values (used by teleportation).
_TILE_FLOOR    = int(TileType.FLOOR)
_TILE_CORRIDOR = int(TileType.CORRIDOR)


def _effect_death(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_DEATH — RAY, instant-kill unless monster is immune.

    Cite: vendor/nethack/src/zap.c::zhitm ~4308:
      if (nonliving(mon->data) || is_demon(mon->data)
          || is_vampshifter(mon) || resists_magm(mon)) { break; }

    Immunity lookup uses _DEATH_IMMUNE[mon_type] via JIT-pure jnp.take,
    supplemented by state.mon_undead for runtime-set undead flag.
    """
    def on_hit(s, r, mon_idx):
        mtype = s.mon_type[mon_idx].astype(jnp.int32)
        mtype = jnp.clip(mtype, 0, N_MONSTERS - 1)
        tbl_immune  = jnp.take(_DEATH_IMMUNE, mtype, axis=0)
        flag_undead = s.mon_undead[mon_idx]
        is_immune   = tbl_immune | flag_undead
        dmg = lax.cond(
            is_immune,
            lambda _: jnp.int32(0),
            lambda _: s.mon_hp[mon_idx],
            None,
        )
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_sleep(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_SLEEP — RAY, sleep target for d(1, 12) turns.

    Vendor: zap.c:483-486 calls ``sleep_monst(mtmp, d(1 + otmp->spe, 12),
    WAND_CLASS)`` then ``slept_monst(mtmp)``.  We don't propagate the wand
    spe into the on_hit context, so we use the spe=0 baseline d(1, 12) ∈
    [1, 12].  Duration is written to mon_paralyzed_timer (vendor mfrozen)
    so the monster can't act until the timer expires; mon_asleep is also
    set so an unbumped wake-check uses the same "sleeping" status.

    Cite: vendor/nethack/src/zap.c::buzz line 483, ZT_SLEEP at line 4296.
    """
    def on_hit(s, r, mon_idx):
        # d(1, 12) = randint [1, 12].
        r, key = jax.random.split(r)
        duration = jax.random.randint(
            key, (), jnp.int32(1), jnp.int32(13), dtype=jnp.int32
        ).astype(s.mon_paralyzed_timer.dtype)
        new_asleep = s.mon_asleep.at[mon_idx].set(jnp.bool_(True))
        cur = s.mon_paralyzed_timer[mon_idx]
        new_timer = jnp.maximum(cur, duration)
        new_paralyzed = s.mon_paralyzed_timer.at[mon_idx].set(new_timer)
        return s.replace(
            mon_asleep=new_asleep,
            mon_paralyzed_timer=new_paralyzed,
        ), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_cold(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_COLD — RAY, 6d6 cold damage + freeze water tiles along the path.

    Cite: vendor/nethack/src/zap.c:3464-3465: nd=6 for elemental wands.
    buzz() ZT_COLD: bhitm() d(nd=6, 6).  Immune if mon_resists & MR_COLD.
    Special: pools/moats along beam freeze (become ICE / floor).
    """
    from Nethax.nethax.constants.monsters import MR_COLD
    dy = _DIR_DY[direction].astype(jnp.int16)
    dx = _DIR_DX[direction].astype(jnp.int16)
    map_h, map_w = state.terrain.shape

    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 6, 6)
        is_immune = (s.mon_resists[mon_idx] & int(MR_COLD)).astype(jnp.bool_)
        actual_dmg = jnp.where(is_immune, jnp.int32(0), dmg)
        s = _deal_damage(s, mon_idx, actual_dmg)
        return s, r

    state, rng = cast_ray(state, rng, state.player_pos, direction,
                          on_hit_fn=on_hit, stop_on_hit=False)

    # Freeze water tiles along the ray path.
    def _freeze_step(carry, step_i):
        terrain, pos = carry
        next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)
        nr = jnp.clip(next_pos[0], 0, map_h - 1)
        nc = jnp.clip(next_pos[1], 0, map_w - 1)
        is_water = terrain[nr, nc] == int(TileType.WATER)
        new_terrain = lax.cond(
            is_water,
            lambda t: t.at[nr, nc].set(jnp.int8(ICE_TILE)),
            lambda t: t,
            terrain,
        )
        return (new_terrain, jnp.array([nr, nc], dtype=jnp.int16)), None

    (frozen_terrain, _), _ = lax.scan(
        _freeze_step,
        (state.terrain, state.player_pos.astype(jnp.int16)),
        jnp.arange(DEFAULT_RAY_RANGE, dtype=jnp.int32),
    )
    return state.replace(terrain=frozen_terrain), rng


def _effect_fire(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_FIRE — RAY, 6d6 fire damage per monster.

    Cite: vendor/nethack/src/zap.c:3464-3465: nd=6 for elemental wands.
    buzz() ZT_FIRE: bhitm() d(nd=6, 6).  Immune if mon_resists & MR_FIRE.
    Cite: vendor/nethack/src/zap.c::zhitm ~4261 (resists_fire gate).
    Also burns scrolls/spellbooks in inventory (not modelled yet; Wave 4).
    """
    from Nethax.nethax.constants.monsters import MR_FIRE

    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 6, 6)
        is_immune = (s.mon_resists[mon_idx] & int(MR_FIRE)).astype(jnp.bool_)
        actual_dmg = jnp.where(is_immune, jnp.int32(0), dmg)
        s = _deal_damage(s, mon_idx, actual_dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_lightning(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_LIGHTNING — RAY, 6d6 electric damage per monster.

    Cite: vendor/nethack/src/zap.c:3464-3465: nd=6 for elemental wands.
    buzz() ZT_LIGHTNING: bhitm() d(nd=6, 6).  Immune if mon_resists & MR_ELEC.
    Metal items reflect lightning (not modelled; Wave 4).
    """
    from Nethax.nethax.constants.monsters import MR_ELEC

    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 6, 6)
        is_immune = (s.mon_resists[mon_idx] & int(MR_ELEC)).astype(jnp.bool_)
        actual_dmg = jnp.where(is_immune, jnp.int32(0), dmg)
        s = _deal_damage(s, mon_idx, actual_dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_digging(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 0
) -> tuple[WandState, jax.Array]:
    """WAN_DIGGING — vendor/nethack/src/dig.c::zap_dig (line 1548).

    Three branches, matching vendor's u.dz / u.dx,u.dy switch:

    1. Down (direction == 8, u.dz > 0):  set player tile to HOLE.  Vendor
       routes through digactualhole() (dig.c line 640) after dig_check.
       Wave 48e: full dig_check fail-code coverage —
         FAIL_ONSTAIRS / FAIL_ALTAR / FAIL_THRONE     (dig.c:211-223, tile-based)
         FAIL_AIRLEVEL / FAIL_WATERLEVEL              (dig.c:225-227, branch-based)
         FAIL_UNDESTROYABLETRAP                        (dig.c:231-237, trap_type-based)
         FAIL_W_NONDIGGABLE                            (dig.c:228-230, wall_info-based)

    2. Up   (direction == 9, u.dz < 0):  loosen a rock from the ceiling and
       hit the hero for rnd(6) damage (rnd(2) with a hard helmet).  No map
       change.  Cite: vendor/nethack/src/dig.c lines 1584-1610.

    3. Cardinal (0..7):  ray carves up to ``digdepth = rn1(18, 8)`` tiles
       (8..25 inclusive).  Doors decrement digdepth by 2; closed_door /
       SDOOR / IS_TREE / IS_WALL handled separately for maze vs non-maze
       levels.  Cite: vendor/nethack/src/dig.c lines 1669-1731.  Each
       wall carve also respects W_NONDIGGABLE (dig.c:228-230) by leaving
       the tile untouched when wall_info[flat_lv, pos] is set.
    """
    map_h, map_w = state.terrain.shape
    dir_idx = jnp.int32(direction)

    # Resolve flat-level index for traps / wall_info lookups.
    # WandState.traps / wall_info may have shape [1, H, W] (single-level
    # fixture) OR [num_branches*num_levels, H, W] (full EnvState projection).
    # We clamp the level index to 0 when the array has length 1.
    _n_levels = state.traps.trap_type.shape[0]
    flat_lv = jnp.where(
        jnp.int32(_n_levels) > jnp.int32(1),
        jnp.clip(state.dungeon_level.astype(jnp.int32) - 1, 0, _n_levels - 1),
        jnp.int32(0),
    )

    # AIR / WATER plane test — vendor Is_airlevel / Is_waterlevel are
    # special-level predicates.  We track Endgame as Branch.ENDGAME (id=6),
    # which spans Earth/Air/Fire/Water/Astral planes; treating any Endgame
    # plane as no-dig-down is a faithful superset of vendor's AIR+WATER
    # fail codes (vendor's Earth/Fire/Astral planes also reject HOLE carves
    # via other mechanisms — none are reachable carve targets).  Cite:
    # vendor/nethack/src/dig.c::dig_check lines 225-227.
    is_airwater_level = state.branch == jnp.int8(6)

    # ---------- dig_check filter ----------
    # Cite: vendor/nethack/src/dig.c::dig_check (dig.c:207-260).
    def _set_hole(t):
        pr = state.player_pos[0].astype(jnp.int32)
        pc = state.player_pos[1].astype(jnp.int32)
        here = t[pr, pc].astype(jnp.int32)
        # Tile-based protection: stairs/altar/throne (dig.c:211-223).
        tile_protected = (
            (here == jnp.int32(TileType.STAIRCASE_UP))
            | (here == jnp.int32(TileType.STAIRCASE_DOWN))
            | (here == jnp.int32(TileType.ALTAR))
            | (here == jnp.int32(TileType.THRONE))
        )
        # UNDESTROYABLETRAP: MAGIC_PORTAL (17) and VIBRATING_SQUARE (23)
        # (vendor/nethack/include/trap.h enum trap_types).
        # Cite: vendor/nethack/src/dig.c::dig_check lines 231-237.
        tt = state.traps.trap_type[flat_lv, pr, pc].astype(jnp.int32)
        trap_protected = (
            (tt == jnp.int32(int(TrapType.MAGIC_PORTAL)))
            | (tt == jnp.int32(int(TrapType.VIBRATING_SQUARE)))
        )
        # W_NONDIGGABLE flag (dig.c:228-230) on the player tile.
        wall_protected = state.wall_info[flat_lv, pr, pc]
        protected = (
            tile_protected
            | trap_protected
            | wall_protected
            | is_airwater_level
        )
        return lax.cond(
            protected,
            lambda tt_: tt_,
            lambda tt_: tt_.at[pr, pc].set(jnp.int8(TileType.HOLE)),
            t,
        )

    # ---------- up-dig falling rock (dig.c:1584-1610) ----------
    def _up_dig(t):
        # Map change: none.  Damage applied via player_hp side-effect on the
        # returned state below.
        return t

    # ---------- horizontal ray dig (dig.c:1622-1737) ----------
    def _normal_dig(t):
        """Horizontal ray dig — vendor zap_dig main while loop.

        digdepth = rn1(18, 8)  → uniform in [8, 25].
        Each step:
          * closed_door / SDOOR → carve to OPEN_DOOR, digdepth -= 2.
          * maze level WALL/TREE/STONE → carve to ROOM/CORR and break.
          * IS_WALL (non-maze) → DOOR; IS_TREE → ROOM; else → CORR.
            digdepth -= 2 for walls, -=1 for stone/corridor.
        """
        safe_dir = jnp.clip(dir_idx, 0, 7)
        dy = _DIR_DY[safe_dir].astype(jnp.int16)
        dx = _DIR_DX[safe_dir].astype(jnp.int16)

        # rn1(18, 8) = uniform [8, 25].
        digdepth0 = jax.random.randint(rng, (), 8, 26, dtype=jnp.int32)

        # Maze-level branch (vendor dig.c:1700-1719) gates wall→corridor
        # carving in Gehennom mazes.  JAX-required divergence: WandState
        # carries no branch slice, so we default to the non-maze code path
        # (which subsumes the standard dungeon, Mines, and Quest layouts).
        # EnvState callers route through action_dispatch where the proper
        # branch is available.
        maze_dig = jnp.int32(0)

        def _dig_step(carry, _):
            terrain, pos, remaining, stopped = carry
            next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)
            nr = jnp.clip(next_pos[0], 0, map_h - 1)
            nc = jnp.clip(next_pos[1], 0, map_w - 1)
            tile = terrain[nr, nc].astype(jnp.int32)

            is_closed_door = tile == jnp.int32(TileType.CLOSED_DOOR)
            is_wall  = tile == jnp.int32(TileType.WALL)
            is_tree  = tile == jnp.int32(TileType.TREE)
            is_stone = (tile == jnp.int32(TileType.VOID))

            # W_NONDIGGABLE — vendor/nethack/src/dig.c::dig_check lines
            # 228-230.  An undestroyable wall blocks the carve at that
            # tile (vendor returns DIG_FAIL_W_NONDIGGABLE and zap_dig
            # aborts the loop at line 1726).
            nondig = state.wall_info[flat_lv, nr, nc]

            still_going = (~stopped) & (remaining > jnp.int32(0))

            # Carve outcome per tile kind.
            #   door  → OPEN_DOOR, decrement -2 (line 1681).
            #   wall  → DOOR     , decrement -2 (line 1725) (non-maze).
            #   tree  → FLOOR    , decrement -2 (line 1728).
            #   stone → CORRIDOR , decrement -1 (line 1731).
            carve = (is_closed_door | is_wall | is_tree | is_stone) & (~nondig)

            new_tile = jnp.where(
                is_closed_door, jnp.int8(TileType.OPEN_DOOR),
                jnp.where(is_wall, jnp.int8(TileType.OPEN_DOOR),  # D_NODOOR
                jnp.where(is_tree, jnp.int8(TileType.FLOOR),
                jnp.where(is_stone, jnp.int8(TileType.CORRIDOR), terrain[nr, nc])))
            )
            cost = jnp.where(
                is_closed_door, jnp.int32(2),
                jnp.where(is_wall, jnp.int32(2),
                jnp.where(is_tree, jnp.int32(2),
                jnp.where(is_stone, jnp.int32(1), jnp.int32(1))))
            )

            do_write = still_going & carve
            new_terrain = lax.cond(
                do_write,
                lambda t2: t2.at[nr, nc].set(new_tile),
                lambda t2: t2,
                terrain,
            )

            # A wall/tree/door encounter completes the carve at that tile and
            # halts the ray on the next iteration when maze_dig is set
            # (matches vendor break-after-carve).  In non-maze levels the ray
            # continues until digdepth expires.  We model this by decrementing
            # remaining and never setting stopped=True (non-maze behaviour).
            new_remaining = jnp.where(still_going, remaining - cost, remaining)
            new_pos = jnp.where(still_going[..., None],
                                jnp.array([nr, nc], dtype=jnp.int16),
                                pos)
            return (new_terrain, new_pos, new_remaining, stopped), None

        init = (
            t,
            state.player_pos.astype(jnp.int16),
            digdepth0,
            jnp.bool_(False),
        )
        (out_terrain, _, _, _), _ = lax.scan(
            _dig_step,
            init,
            jnp.arange(26, dtype=jnp.int32),  # upper bound covers rn1(18, 8) max
        )
        return out_terrain

    is_down = dir_idx == jnp.int32(8)
    is_up   = dir_idx == jnp.int32(9)

    new_terrain = lax.cond(
        is_down, _set_hole,
        lambda t: lax.cond(is_up, _up_dig, _normal_dig, t),
        state.terrain,
    )

    # --- up-dig falling rock damage (dig.c:1594-1597) ---
    # rnd(6) damage normally, rnd(2) if wearing a hard helmet.
    # WandState does not directly track helmet; default to rnd(6).
    rng, sub = jax.random.split(rng)
    rock_dmg = jax.random.randint(sub, (), 1, 7, dtype=jnp.int32)
    # WandState may not have player_hp; only apply if present.
    if hasattr(state, "player_hp"):
        new_hp = jnp.where(is_up, state.player_hp - rock_dmg, state.player_hp)
        return state.replace(terrain=new_terrain, player_hp=new_hp), rng

    return state.replace(terrain=new_terrain), rng


def _effect_enlightenment(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_ENLIGHTENMENT — display intrinsics (no map change).

    Cite: vendor/nethack/src/zap.c do_enlightenment: shows character
    stats and intrinsics; does not alter the map.  We return state unchanged
    to match this behaviour.
    """
    return state, rng


def _effect_create_monster(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_CREATE_MONSTER — spawn a level-appropriate monster adjacent to player.

    Cite: vendor/nethack/src/zap.c wand_create_monster — uses makemon with
    level-appropriate selection.  We filter to monsters with
    _MONSTER_GEN_LEVEL[type] <= dungeon_level + 3, then sample uniformly
    via rejection sampling under lax.while_loop.  JIT-pure.
    """
    map_h, map_w = state.terrain.shape
    max_level = state.dungeon_level.astype(jnp.int32) + jnp.int32(3)

    # Rejection-sample a type index whose gen_level fits the current depth.
    def _type_cond(wstate):
        _, _, candidate = wstate
        return _MONSTER_GEN_LEVEL[candidate].astype(jnp.int32) > max_level

    def _type_body(wstate):
        r_, _, _ = wstate
        r_, sub = jax.random.split(r_)
        c = jax.random.randint(sub, shape=(), minval=1, maxval=N_MONSTERS,
                               dtype=jnp.int32)
        return (r_, jnp.int32(0), c)

    rng, sub_init = jax.random.split(rng)
    init_candidate = jax.random.randint(
        sub_init, shape=(), minval=1, maxval=N_MONSTERS, dtype=jnp.int32
    )
    rng, _, new_type = lax.while_loop(
        _type_cond, _type_body, (rng, jnp.int32(0), init_candidate)
    )

    # Find first dead slot (slot 0 is sentinel; start from 1).
    dead_mask = ~state.mon_alive
    dead_mask = dead_mask.at[0].set(False)  # skip sentinel
    slot = jnp.argmax(dead_mask).astype(jnp.int32)
    rng, sub2 = jax.random.split(rng)
    new_row = jnp.clip(
        state.player_pos[0] + jax.random.randint(sub2, shape=(), minval=-1, maxval=2),
        0, map_h - 1
    ).astype(jnp.int16)
    rng, sub3 = jax.random.split(rng)
    new_col = jnp.clip(
        state.player_pos[1] + jax.random.randint(sub3, shape=(), minval=-1, maxval=2),
        0, map_w - 1
    ).astype(jnp.int16)
    new_pos = jnp.array([new_row, new_col], dtype=jnp.int16)
    new_hp = jax.random.randint(rng, shape=(), minval=4, maxval=20, dtype=jnp.int32)
    new_mon_pos   = state.mon_pos.at[slot].set(new_pos)
    new_mon_hp    = state.mon_hp.at[slot].set(new_hp)
    new_mon_type  = state.mon_type.at[slot].set(new_type.astype(jnp.int16))
    new_mon_alive = state.mon_alive.at[slot].set(jnp.bool_(True))
    return state.replace(
        mon_pos=new_mon_pos, mon_hp=new_mon_hp,
        mon_type=new_mon_type, mon_alive=new_mon_alive,
    ), rng


def _effect_wishing(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_WISHING — grant a wish via wish.handle_wand_of_wishing if available.

    Cite: vendor/nethack/src/zap.c::zapyourself WAN_WISHING branch and
    vendor/nethack/src/wizard.c::makewish.

    On a full EnvState, delegates to subsystems.wish.handle_wand_of_wishing
    which parses the wish_string via wishymatch/readobjnam (objnam.c) and
    grants the named object; when no wish_string is supplied it falls back
    to "blessed greased +3 gray dragon scale mail" — the canonical NetHack
    reference wish (vendor wizard.c::makewish offers free-form input).
    On a bare WandState (tests), the wish_string path isn't available, so
    we grant a detectable JIT-pure side effect by recharging inventory
    wands to 15 charges (the vendor zap.c::recharge max).

    WandState does not carry the conduct slice so the wish handler cannot
    update WISHLESS / ARTIWISHLESS here; that is handled by callers routing
    through action_dispatch on a full EnvState.
    """
    # WandState fallback path (no full EnvState available):
    # vendor wishing requires the wish-parser (vendor/nethack/src/wizard.c::
    # makewish) and the full object table, neither of which fits on the
    # minimal WandState shape.  We grant a detectable JIT-pure side-effect
    # by recharging every wand in inventory to 15 charges — the maximum
    # zappable count under vendor charging conventions (vendor/nethack/src/
    # zap.c::recharge ~line 1100).  Callers that have a real EnvState route
    # through Nethax/nethax/subsystems/wish.handle_wand_of_wishing instead.
    inv = state.inventory
    is_wand = inv.items.category == ITEM_CATEGORY_WAND
    new_charges = jnp.where(is_wand, jnp.int8(15), inv.items.charges)
    new_items = inv.items.replace(charges=new_charges)
    new_inv = inv.replace(items=new_items)
    return state.replace(inventory=new_inv), rng


def _effect_stasis(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_STASIS — BEAM, freeze first monster in place (asleep flag).

    NetHack stasis = monster cannot act.  We reuse mon_asleep as the freeze
    flag; Wave 4 can add a distinct mon_stasis array.
    """
    def on_hit(s, r, mon_idx):
        new_asleep = s.mon_asleep.at[mon_idx].set(jnp.bool_(True))
        return s.replace(mon_asleep=new_asleep), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_make_invisible(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_MAKE_INVISIBLE — BEAM, first monster becomes invisible."""
    def on_hit(s, r, mon_idx):
        new_invis = s.mon_invisible.at[mon_idx].set(jnp.bool_(True))
        return s.replace(mon_invisible=new_invis), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_undead_turning(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_UNDEAD_TURNING — RAY, 1d8 damage to undead; peaceful otherwise.

    vendor/nethack/src/zap.c bhitm() WAN_UNDEAD_TURNING: rnd(8) to undead.
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 8)
        is_undead = s.mon_undead[mon_idx]
        actual_dmg = jnp.where(is_undead, dmg, jnp.int32(0))
        s = _deal_damage(s, mon_idx, actual_dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_draining(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_DRAINING — BEAM, drain one experience level from the target.

    Cite: vendor/nethack/src/zap.c::bhitm SPE_DRAIN_LIFE branch (~line 521):
      dmg = monhp_per_lvl(mtmp)  [defaults to rnd(8)]
      mtmp->mhp    -= dmg
      mtmp->mhpmax -= dmg
      if mhpmax <= 0: killed

    monhp_per_lvl() default path: rnd(8) — makemon.c:986.
    resists_drli: undead are immune (mon_undead flag).
    WandState does not carry a per-monster XL array; we model the HP-max
    reduction which is the mechanically significant part.
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_rnd(r, 8)            # rnd(8) — makemon.c:989
        is_immune = s.mon_undead[mon_idx]  # resists_drli proxy
        actual_dmg = jnp.where(is_immune, jnp.int32(0), dmg)
        # Reduce both current HP and HP-max (zap.c:533-534).
        new_hp     = jnp.maximum(s.mon_hp[mon_idx]     - actual_dmg, jnp.int32(0))
        new_hp_max = jnp.maximum(s.mon_hp_max[mon_idx] - actual_dmg, jnp.int32(0))
        is_killed  = (new_hp_max <= jnp.int32(0)) | (new_hp <= jnp.int32(0))
        s = s.replace(
            mon_hp=s.mon_hp.at[mon_idx].set(new_hp),
            mon_hp_max=s.mon_hp_max.at[mon_idx].set(new_hp_max),
            mon_alive=s.mon_alive.at[mon_idx].set(~is_immune & ~is_killed),
        )
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_acid(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_ACID — BEAM, 1d6 acid damage to first target.

    Not a canonical wand in 3.6; present in some variants.  Modelled as a
    single-target d6 acid beam.
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 6)
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_poison_gas(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_POISON_GAS — RAY, 1d6 poison damage per monster hit."""
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 6)
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


# ---------------------------------------------------------------------------
# jax.lax.switch dispatch table (must be indexed by WandEffect int value)
# ---------------------------------------------------------------------------

# Each branch must have the same signature: (state, rng, direction) -> (state, rng).
# SELF/NODIR effects ignore `direction`; wrap them with a lambda.

def _dispatch_light(state, rng, direction):
    return _effect_light(state, rng)

def _dispatch_nothing(state, rng, direction):
    return _effect_nothing(state, rng, direction)

def _dispatch_secret_door(state, rng, direction):
    return _effect_secret_door_detection(state, rng)

def _dispatch_opening(state, rng, direction):
    return _effect_opening(state, rng, direction)

def _dispatch_locking(state, rng, direction):
    return _effect_locking(state, rng, direction)

def _dispatch_probing(state, rng, direction):
    return _effect_probing(state, rng, direction)

def _dispatch_magic_missile(state, rng, direction):
    return _effect_magic_missile(state, rng, direction)

def _dispatch_striking(state, rng, direction):
    return _effect_striking(state, rng, direction)

def _dispatch_slow(state, rng, direction):
    return _effect_slow_monster(state, rng, direction)

def _dispatch_speed(state, rng, direction):
    return _effect_speed_monster(state, rng, direction)

def _dispatch_cancellation(state, rng, direction):
    return _effect_cancellation(state, rng, direction)

def _dispatch_polymorph(state, rng, direction):
    return _effect_polymorph(state, rng, direction)

def _dispatch_teleport(state, rng, direction):
    return _effect_teleportation(state, rng, direction)

def _dispatch_death(state, rng, direction):
    return _effect_death(state, rng, direction)

def _dispatch_sleep(state, rng, direction):
    return _effect_sleep(state, rng, direction)

def _dispatch_cold(state, rng, direction):
    return _effect_cold(state, rng, direction)

def _dispatch_fire(state, rng, direction):
    return _effect_fire(state, rng, direction)

def _dispatch_lightning(state, rng, direction):
    return _effect_lightning(state, rng, direction)

def _dispatch_digging(state, rng, direction):
    return _effect_digging(state, rng, direction)

def _dispatch_enlightenment(state, rng, direction):
    return _effect_enlightenment(state, rng)

def _dispatch_create_monster(state, rng, direction):
    return _effect_create_monster(state, rng)

def _dispatch_wishing(state, rng, direction):
    return _effect_wishing(state, rng)

def _dispatch_stasis(state, rng, direction):
    return _effect_stasis(state, rng, direction)

def _dispatch_make_invisible(state, rng, direction):
    return _effect_make_invisible(state, rng, direction)

def _dispatch_undead_turning(state, rng, direction):
    return _effect_undead_turning(state, rng, direction)

def _dispatch_draining(state, rng, direction):
    return _effect_draining(state, rng, direction)

def _dispatch_acid(state, rng, direction):
    return _effect_acid(state, rng, direction)

def _dispatch_poison_gas(state, rng, direction):
    return _effect_poison_gas(state, rng, direction)


# Ordered by WandEffect int value (0..27).
_EFFECT_BRANCHES = [
    _dispatch_light,           # 0  LIGHT
    _dispatch_nothing,         # 1  NOTHING
    _dispatch_secret_door,     # 2  SECRET_DOOR_DETECTION
    _dispatch_opening,         # 3  OPENING
    _dispatch_locking,         # 4  LOCKING
    _dispatch_probing,         # 5  PROBING
    _dispatch_magic_missile,   # 6  MAGIC_MISSILE
    _dispatch_striking,        # 7  STRIKING
    _dispatch_slow,            # 8  SLOW_MONSTER
    _dispatch_speed,           # 9  SPEED_MONSTER
    _dispatch_cancellation,    # 10 CANCELLATION
    _dispatch_polymorph,       # 11 POLYMORPH
    _dispatch_teleport,        # 12 TELEPORTATION
    _dispatch_death,           # 13 DEATH
    _dispatch_sleep,           # 14 SLEEP
    _dispatch_cold,            # 15 COLD
    _dispatch_fire,            # 16 FIRE
    _dispatch_lightning,       # 17 LIGHTNING
    _dispatch_digging,         # 18 DIGGING
    _dispatch_enlightenment,   # 19 ENLIGHTENMENT
    _dispatch_create_monster,  # 20 CREATE_MONSTER
    _dispatch_wishing,         # 21 WISHING
    _dispatch_stasis,          # 22 STASIS
    _dispatch_make_invisible,  # 23 MAKE_INVISIBLE
    _dispatch_undead_turning,  # 24 UNDEAD_TURNING
    _dispatch_draining,        # 25 DRAINING
    _dispatch_acid,            # 26 ACID
    _dispatch_poison_gas,      # 27 POISON_GAS
]

assert len(_EFFECT_BRANCHES) == N_WANDS, (
    f"_EFFECT_BRANCHES has {len(_EFFECT_BRANCHES)} entries; expected {N_WANDS}"
)


# ---------------------------------------------------------------------------
# Charge decrement
# ---------------------------------------------------------------------------

def _decrement_charges(inv: InventoryState, slot_idx: jax.Array) -> InventoryState:
    """Return inventory with charges[slot_idx] decremented by 1 (min 0)."""
    old_charges = inv.items.charges
    new_charges = old_charges.at[slot_idx].add(-1)
    new_charges = jnp.maximum(new_charges, jnp.int8(0))
    new_items = inv.items.replace(charges=new_charges)
    return inv.replace(items=new_items)


# ---------------------------------------------------------------------------
# Primary dispatch: zap_wand
# ---------------------------------------------------------------------------

def zap_wand(
    state: WandState,
    rng: jax.Array,
    slot_idx: jax.Array,
    direction: jax.Array,
) -> WandState:
    """Zap the wand in inventory slot *slot_idx* in *direction*.

    Steps (mirrors vendor/nethack/src/zap.c dozap()):
      1. Read wand type_id from inventory slot.
      2. Map type_id → WandEffect index (type_id IS the WandEffect index here).
      3. Decrement charges.
      4. Dispatch to the per-effect handler via jax.lax.switch.

    Parameters
    ----------
    state     : WandState
    rng       : JAX PRNGKey
    slot_idx  : scalar int — index into state.inventory.items (0..51)
    direction : scalar int in [0, 7]  (N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7)

    Returns
    -------
    Updated WandState.

    Note: in this codebase the Item.type_id for a wand equals its WandEffect
    ordinal (0..27).  Wave 4 will add an appearance-shuffle layer on top.
    """
    effect_idx = state.inventory.items.type_id[slot_idx].astype(jnp.int32)
    effect_idx = jnp.clip(effect_idx, 0, N_WANDS - 1)

    # Decrement charges before applying effect.
    new_inv = _decrement_charges(state.inventory, slot_idx)
    state = state.replace(inventory=new_inv)

    # Dispatch.
    state, rng = lax.switch(
        effect_idx,
        _EFFECT_BRANCHES,
        state, rng, direction,
    )
    return state


# ---------------------------------------------------------------------------
# Action handler export (Wave 3 entry point)
# ---------------------------------------------------------------------------

def handle_zap(state: WandState, rng: jax.Array) -> WandState:
    """Zap the first wand found in inventory toward the player's last direction.

    Wave 4 will add a directional selection menu; for now we:
      - Find the first inventory slot with category == ITEM_CATEGORY_WAND.
      - Use direction = 2 (East) as a default stand-in for "last move".

    This function is the exported entry point consumed by action_dispatch.py.
    """
    categories = state.inventory.items.category
    is_wand    = categories == jnp.int8(ITEM_CATEGORY_WAND)
    slot_idx   = jnp.argmax(is_wand).astype(jnp.int32)  # 0 if none found

    # Default direction: East (index 2).  Wave 4 will read state.player_direction.
    direction = jnp.int32(2)

    return zap_wand(state, rng, slot_idx, direction)


# ---------------------------------------------------------------------------
# EnvState-level self-zap for wand of polymorph  (zap.c::zapyourself)
# ---------------------------------------------------------------------------

def zap_polymorph_at_self(state, rng: jax.Array, slot_idx: jax.Array):
    """Apply a wand-of-polymorph self-zap to the full EnvState.

    vendor/nethack/src/zap.c::zapyourself WAN_POLYMORPH branch:
      calls polyself() which routes to polymorph_player with a random form.

    This function operates on a full EnvState (not WandState) because
    polymorph_player needs the entire state.  It is called from callers
    that detect direction==self (direction sentinel 0 or explicit self-zap).

    Cite: zap.c::zapyourself (WAN_POLYMORPH branch).
    Cite: polyself.c:280 for valid-form selection.
    """
    from Nethax.nethax.subsystems.polymorph import (
        polymorph_player,
        choose_random_polymorph_form,
    )
    # Decrement charges using inventory slice.
    inv = state.inventory
    old_charges = inv.items.charges
    new_charges = old_charges.at[slot_idx].add(jnp.int8(-1))
    new_charges = jnp.maximum(new_charges, jnp.int8(0))
    new_inv = inv.replace(items=inv.items.replace(charges=new_charges))
    state = state.replace(inventory=new_inv)

    rng, sub = jax.random.split(rng)
    form = choose_random_polymorph_form(state, sub)
    rng, sub2 = jax.random.split(rng)
    return polymorph_player(state, sub2, form, controlled=False)
