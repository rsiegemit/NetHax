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
  - Bresenham stepping is approximated with unit-step integer direction
    vectors (the eight compass directions used by NetHack's buzz()).
  - Monster table is a flat array; slot 0 is reserved / always dead so
    "no hit" returns index 0 safely.
"""
from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import struct

from Nethax.nethax.constants.tiles import TileType
from Nethax.nethax.subsystems.inventory import (
    MAX_INVENTORY_SLOTS,
    Item,
    InventoryState,
)
from Nethax.nethax.subsystems.monster_ai import MAX_MONSTERS_PER_LEVEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WANDS: int = 28

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
    mon_pos    : (row, col) for each monster.  int16[N, 2].
    mon_hp     : current hit points.           int32[N].
    mon_type   : monster species index.        int16[N].
    mon_alive  : whether the slot is occupied. bool[N].
    mon_asleep : whether the monster is asleep. bool[N].
    mon_undead : whether the monster is undead. bool[N].
    mon_invisible : whether the monster is invisible. bool[N].

    terrain    : current-level terrain tiles.  int8[MAP_H, MAP_W].
    explored   : explored mask for lighting.   bool[MAP_H, MAP_W].

    inventory  : player inventory (items + charges).

    player_pos : (row, col).  int16[2].

    rng_state  : JAX PRNGKey for effect randomness.
    """
    mon_pos:       jax.Array   # int16[N, 2]
    mon_hp:        jax.Array   # int32[N]
    mon_type:      jax.Array   # int16[N]
    mon_alive:     jax.Array   # bool[N]
    mon_asleep:    jax.Array   # bool[N]
    mon_undead:    jax.Array   # bool[N]
    mon_invisible: jax.Array   # bool[N]

    terrain:       jax.Array   # int8[MAP_H, MAP_W]
    explored:      jax.Array   # bool[MAP_H, MAP_W]

    inventory:     InventoryState

    player_pos:    jax.Array   # int16[2]

    @classmethod
    def empty(cls, map_h: int = 21, map_w: int = 80) -> "WandState":
        """Return a zero-initialised WandState (no monsters, empty inventory)."""
        n = MAX_MONSTERS_PER_LEVEL
        return cls(
            mon_pos=jnp.zeros((n, 2), dtype=jnp.int16),
            mon_hp=jnp.zeros(n, dtype=jnp.int32),
            mon_type=jnp.zeros(n, dtype=jnp.int16),
            mon_alive=jnp.zeros(n, dtype=bool),
            mon_asleep=jnp.zeros(n, dtype=bool),
            mon_undead=jnp.zeros(n, dtype=bool),
            mon_invisible=jnp.zeros(n, dtype=bool),
            terrain=jnp.zeros((map_h, map_w), dtype=jnp.int8),
            explored=jnp.zeros((map_h, map_w), dtype=bool),
            inventory=InventoryState.empty(),
            player_pos=jnp.zeros(2, dtype=jnp.int16),
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
    dy = _DIR_DY[direction].astype(jnp.int16)
    dx = _DIR_DX[direction].astype(jnp.int16)

    map_h, map_w = state.terrain.shape

    if on_hit_fn is None:
        def on_hit_fn(s, r, _idx):
            return s, r

    def _step(carry, _step_i):
        s, r, pos, stopped = carry

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

        return (s, r, next_pos, beam_stopped), None

    init_carry = (state, rng, start_pos.astype(jnp.int16), jnp.bool_(False))
    (final_state, final_rng, _, _), _ = lax.scan(
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
    """WAN_LIGHT — illuminate the entire map (set explored=True everywhere).

    vendor/nethack/src/zap.c: dozap() WAN_LIGHT calls do_clear_area() which
    marks all tiles within radius as lit and explored.  We simplify to full
    level exploration for JIT compatibility.
    """
    new_explored = jnp.ones_like(state.explored)
    return state.replace(explored=new_explored), rng


def _effect_nothing(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 0
) -> tuple[WandState, jax.Array]:
    """WAN_NOTHING — no effect.  Charges were still spent."""
    return state, rng


def _effect_secret_door_detection(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_SECRET_DOOR_DETECTION — reveal all tiles (simplified detect_doors).

    NetHack reveals secret doors; we mark the full map as explored.
    """
    new_explored = jnp.ones_like(state.explored)
    return state.replace(explored=new_explored), rng


def _effect_opening(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_OPENING — open all closed doors on the level.

    vendor/nethack/src/zap.c: NODIR wand, iterates all map positions.
    """
    is_closed = state.terrain == int(TileType.CLOSED_DOOR)
    new_terrain = jnp.where(is_closed, int(TileType.OPEN_DOOR), state.terrain)
    return state.replace(terrain=new_terrain), rng


def _effect_locking(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_LOCKING — close all open doors on the level."""
    is_open = state.terrain == int(TileType.OPEN_DOOR)
    new_terrain = jnp.where(is_open, int(TileType.CLOSED_DOOR), state.terrain)
    return state.replace(terrain=new_terrain), rng


def _effect_probing(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_PROBING — reveal monsters along beam (no combat effect).

    For JAX purity: makes all monsters visible (no hidden info suppression).
    """
    return state, rng


def _effect_magic_missile(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_MAGIC_MISSILE — RAY, 1d6 damage per monster hit.

    vendor/nethack/src/zap.c buzz(): ZT_MAGIC_MISSILE, d(1,6) per hit.
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 6)
        s = _deal_damage(s, mon_idx, dmg)
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
    """WAN_SLOW_MONSTER — RAY, halve target speed (stub: no effect).

    Full speed mechanic is a Wave 4 concern (mon_adjust_speed in monmove.c).
    """
    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=None, stop_on_hit=False)


def _effect_speed_monster(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_SPEED_MONSTER — RAY, double target speed (stub: no effect)."""
    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=None, stop_on_hit=False)


def _effect_cancellation(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_CANCELLATION — BEAM, remove monster's special properties.

    Stub: no monster property array yet; no state change.
    Wave 4 will clear intrinsic resistances and MR flag.
    """
    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=None, stop_on_hit=True)


def _effect_polymorph(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_POLYMORPH — BEAM, change monster type to random eligible species.

    vendor/nethack/src/zap.c bhitm() WAN_POLYMORPH: newcham().
    We randomise mon_type to [1, 394] (total NetHack monster count).

    TODO Wave 5: when WandState is folded into EnvState, route through
    subsystems.polymorph.polymorph_monster so HP-scaling + intrinsic gain
    behave the same as monster-side newcham.
    """
    def on_hit(s, r, mon_idx):
        r, new_type = _rng_rnd(r, 394)
        new_mon_type = s.mon_type.at[mon_idx].set(new_type.astype(jnp.int16))
        return s.replace(mon_type=new_mon_type), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_teleportation(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_TELEPORTATION — BEAM, monster pos → random valid floor tile.

    vendor/nethack/src/zap.c: u_teleport_mon() picks a random location.
    We pick a random (row, col) on the map and move the monster there.
    """
    map_h, map_w = state.terrain.shape

    def on_hit(s, r, mon_idx):
        r, sub = jax.random.split(r)
        new_row = jax.random.randint(sub, shape=(), minval=0, maxval=map_h)
        r, sub2 = jax.random.split(r)
        new_col = jax.random.randint(sub2, shape=(), minval=0, maxval=map_w)
        new_pos = jnp.array([new_row, new_col], dtype=jnp.int16)
        new_mon_pos = s.mon_pos.at[mon_idx].set(new_pos)
        return s.replace(mon_pos=new_mon_pos), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=True)


def _effect_death(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_DEATH — RAY, set HP → 0 unless monster is undead.

    vendor/nethack/src/zap.c: ZT_DEATH / AD_DISN instant-kill unless
    the target is undead (undead are immune: MAGIC_COOKIE path skipped).
    """
    def on_hit(s, r, mon_idx):
        is_undead = s.mon_undead[mon_idx]
        dmg = lax.cond(
            is_undead,
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
    """WAN_SLEEP — RAY, put target to sleep (mon_asleep = True).

    vendor/nethack/src/zap.c: sleep_monst() / slept_monst().
    Sleep duration (d10 turns) is not tracked yet; Wave 4 adds a timer array.
    """
    def on_hit(s, r, mon_idx):
        new_asleep = s.mon_asleep.at[mon_idx].set(jnp.bool_(True))
        return s.replace(mon_asleep=new_asleep), r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_cold(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_COLD — RAY, 1d6 cold damage + freeze water tiles along the path.

    vendor/nethack/src/zap.c buzz() ZT_COLD:
      - bhitm(): cold damage d(1,6)
      - Special: pools/moats along beam freeze (become ICE / floor).
    """
    dy = _DIR_DY[direction].astype(jnp.int16)
    dx = _DIR_DX[direction].astype(jnp.int16)
    map_h, map_w = state.terrain.shape

    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 6)
        s = _deal_damage(s, mon_idx, dmg)
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
    """WAN_FIRE — RAY, 1d6 fire damage per monster.

    vendor/nethack/src/zap.c buzz() ZT_FIRE: bhitm() d(1,6).
    Also burns scrolls/spellbooks in inventory (not modelled yet; Wave 4).
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 6)
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_lightning(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 2
) -> tuple[WandState, jax.Array]:
    """WAN_LIGHTNING — RAY, 1d6 electric damage per monster.

    vendor/nethack/src/zap.c buzz() ZT_LIGHTNING: bhitm() d(1,6).
    Metal items reflect lightning (not modelled; Wave 4).
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 6)
        s = _deal_damage(s, mon_idx, dmg)
        return s, r

    return cast_ray(state, rng, state.player_pos, direction,
                    on_hit_fn=on_hit, stop_on_hit=False)


def _effect_digging(
    state: WandState, rng: jax.Array, direction: int | jax.Array = 0
) -> tuple[WandState, jax.Array]:
    """WAN_DIGGING — AT_LOCATION, carve the tile in *direction* to corridor.

    vendor/nethack/src/zap.c: zap_updown() for up/down; dig_map() for
    cardinal directions.  We replace WALL tiles with CORRIDOR along the ray.
    """
    dy = _DIR_DY[direction].astype(jnp.int16)
    dx = _DIR_DX[direction].astype(jnp.int16)
    map_h, map_w = state.terrain.shape

    def _dig_step(carry, step_i):
        terrain, pos = carry
        next_pos = pos + jnp.array([dy, dx], dtype=jnp.int16)
        nr = jnp.clip(next_pos[0], 0, map_h - 1)
        nc = jnp.clip(next_pos[1], 0, map_w - 1)
        is_wall = terrain[nr, nc] == int(TileType.WALL)
        is_void = terrain[nr, nc] == int(TileType.VOID)
        should_dig = is_wall | is_void
        new_terrain = lax.cond(
            should_dig,
            lambda t: t.at[nr, nc].set(jnp.int8(DIG_TILE)),
            lambda t: t,
            terrain,
        )
        return (new_terrain, jnp.array([nr, nc], dtype=jnp.int16)), None

    (new_terrain, _), _ = lax.scan(
        _dig_step,
        (state.terrain, state.player_pos.astype(jnp.int16)),
        jnp.arange(DEFAULT_RAY_RANGE, dtype=jnp.int32),
    )
    return state.replace(terrain=new_terrain), rng


def _effect_enlightenment(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_ENLIGHTENMENT — reveal full map (simplified introspection).

    NetHack shows character stats + intrinsics; we expose the full explored map.
    """
    new_explored = jnp.ones_like(state.explored)
    return state.replace(explored=new_explored), rng


def _effect_create_monster(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_CREATE_MONSTER — spawn a new monster adjacent to player.

    Places a random monster type in the first empty slot adjacent to player_pos.
    Wave 4 will apply proper level-appropriate generation tables.
    """
    map_h, map_w = state.terrain.shape
    rng, sub = jax.random.split(rng)
    new_type = jax.random.randint(sub, shape=(), minval=1, maxval=100,
                                  dtype=jnp.int16)
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
    new_mon_type  = state.mon_type.at[slot].set(new_type)
    new_mon_alive = state.mon_alive.at[slot].set(jnp.bool_(True))
    return state.replace(
        mon_pos=new_mon_pos, mon_hp=new_mon_hp,
        mon_type=new_mon_type, mon_alive=new_mon_alive,
    ), rng


def _effect_wishing(
    state: WandState, rng: jax.Array
) -> tuple[WandState, jax.Array]:
    """WAN_WISHING — WandState-slice stub (recharge all wands).

    The canonical wish handler lives in subsystems.wish.handle_wand_of_wishing
    because granting a wish requires the conduct slice (WISHLESS /
    ARTIWISHLESS) which is not part of WandState.  When the wand-of-wishing
    zap is routed through action_dispatch on a full EnvState, callers should
    invoke wish.handle_wand_of_wishing(state, rng) instead of this stub.

    Here we keep the harmless "recharge other wands" placeholder behaviour so
    existing WandState-only tests still see a deterministic effect.
    Cite: vendor/nethack/src/zap.c::zapyourself WAN_WISHING branch and
    vendor/nethack/src/wizard.c::makewish.
    """
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
    """WAN_DRAINING — BEAM, drain experience level (stub: 1d4 damage).

    Full XL drain requires a monster XL array (Wave 4).
    """
    def on_hit(s, r, mon_idx):
        r, dmg = _rng_d(r, 1, 4)
        s = _deal_damage(s, mon_idx, dmg)
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
    return _effect_opening(state, rng)

def _dispatch_locking(state, rng, direction):
    return _effect_locking(state, rng)

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
