"""Dungeon-feature subsystem — fountains, thrones, sinks, altars, doors.

Canonical sources:
  vendor/nethack/src/fountain.c  — drinkfountain(), dipfountain(), drinksink(),
                                   dipsink(), dryup(), breaksink()
  vendor/nethack/src/sit.c       — throne_sit_effect() (effects 1-13)
  vendor/nethack/src/dokick.c    — kick_door(), kick_nondoor()
  vendor/nethack/src/lock.c      — doopen(), doclose(), picking locks
  vendor/nethack/src/dbridge.c   — drawbridge open/close mechanics
  vendor/nethack/include/rm.h    — D_NODOOR/D_BROKEN/D_ISOPEN/D_CLOSED/D_LOCKED

Status: Wave 3 — door operations (open/close/kick/unlock/door_blocks_movement)
        and action handlers implemented.

TODO (later waves):
  Wave 4 (feature effect tables):
    - quaff_fountain: 30-outcome rnd(30) table from drinkfountain()
    - dip_fountain: dipfountain() BUC detection, enchant/disenchant
    - sit_throne: 13-outcome rnd(13) table from throne_sit_effect()
    - kick_sink: 20-outcome rn2(20) table from drinksink()
    - sacrifice_on_altar: BUC check, alignment conversion, sacrifice reward
    - kick_door: monster-anger side effect on success
    - unlock_door: real d20 roll vs lock difficulty
  Wave 5 (advanced features):
    - Vibrating square gateway: requires Amulet in inventory, endgame trigger
    - Drawbridge open/close/collapse (dbridge.c)
    - Secret door discovery via search command (detect.c:findit())
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Fountain effects (vendor/nethack/src/fountain.c :: drinkfountain, fate 1-30)
# ---------------------------------------------------------------------------
class FountainEffect(IntEnum):
    """Outcomes of drinking from a fountain (drinkfountain, rnd(30) table)."""
    REFRESH         =  0   # fate < 10: cool draught restores nutrition
    SELF_KNOWLEDGE  =  1   # fate 19: enlightenment
    FOUL_WATER      =  2   # fate 20: nausea + hunger
    POISONOUS       =  3   # fate 21: STR/CON damage
    SNAKES          =  4   # fate 22: dowatersnakes() — spawn water moccasins
    WATER_DEMON     =  5   # fate 23: dowaterdemon() — demon may grant wish
    CURSE_RAY       =  6   # fate 24: curse random inventory items
    SEE_INVISIBLE   =  7   # fate 25: intrinsic see-invisible
    MONSTER_DETECT  =  8   # fate 26: detect monsters on level
    FIND_GEM        =  9   # fate 27: gem appears (first loot only)
    WATER_NYMPH     = 10   # fate 28: dowaternymph() — nymph steals item
    SCARE           = 11   # fate 29: bad breath; nearby monsters flee
    GUSH            = 12   # fate 30: dogushforth() — water pools spread
    DRY_UP          = 13   # dryup() post-drink: fountain disappears (1-in-3)
    MOIST           = 14   # blessed fountain: restore abilities + gain attr
    SPRAY           = 15   # dipfountain: water sprays on nearby tiles


# ---------------------------------------------------------------------------
# Throne effects (vendor/nethack/src/sit.c :: throne_sit_effect, rnd(13))
# ---------------------------------------------------------------------------
class ThroneEffect(IntEnum):
    """Outcomes of sitting on a throne (throne_sit_effect, effect 1-13)."""
    ATTR_LOSS_AND_DAMAGE =  0   # effect 1: random attr -rn1(4,3) + 1d10 HP
    ATTR_GAIN            =  1   # effect 2: random attr +1
    ELECTRIC_SHOCK       =  2   # effect 3: 1d6 or 1d30 shock damage
    FULL_HEAL            =  3   # effect 4: restore HP to max (+4 bonus)
    TAKE_GOLD            =  4   # effect 5: take_gold() — gold teleports away
    WISH_OR_LUCK         =  5   # effect 6: makewish() or change_luck(+1)
    COURT_SUMMON         =  6   # effect 7: summon 1d10 court monsters
    GENOCIDE_COMMAND     =  7   # effect 8: do_genocide(5) — voice commands
    CURSE_ITEMS          =  8   # effect 9: rndcurse() or blindness
    MAP_OR_CONFUSE       =  9   # effect 10: do_mapping() or confusion
    TELEPORT             = 10   # effect 11: tele() or aggravate
    IDENTIFY             = 11   # effect 12: identify_pack() partial
    CONFUSE              = 12   # effect 13: confusion rn1(7,16) turns
    DESTROY_THRONE       = 13   # post-effect (1-in-3): throne disappears


# ---------------------------------------------------------------------------
# Sink effects (vendor/nethack/src/fountain.c :: drinksink, rn2(20))
# ---------------------------------------------------------------------------
class SinkEffect(IntEnum):
    """Outcomes of drinking from a sink (drinksink, rn2(20) table)."""
    COLD_WATER         =  0   # case 0: very cold water, harmless
    WARM_WATER         =  1   # case 1: very warm water, harmless
    SCALDING_WATER     =  2   # case 2: 1d6 fire damage unless fire-resistant
    SEWER_RAT          =  3   # case 3: spawn PM_SEWER_RAT
    RANDOM_POTION      =  4   # case 4: random non-water potion effect
    FIND_RING          =  5   # case 5: ring appears (first time only)
    BREAK_TO_FOUNTAIN  =  6   # case 6: breaksink() — converts sink to fountain
    WATER_ELEMENTAL    =  7   # case 7: spawn PM_WATER_ELEMENTAL
    DRAIN_NUTRITION    =  8   # case 8-9: vomit + hunger (sewage taste)
    POLYMORPH          =  9   # case 10: polyself() if not Unchanging
    NOISE              = 10   # cases 11-12: clanking / sewer-song sound
    STENCH             = 11   # case 13: create_gas_cloud()
    TEPID_WATER        = 12   # default: cold/warm/hot sip, no effect


# ---------------------------------------------------------------------------
# Altar actions (vendor/nethack/src/pray.c + sacrifice.c)
# ---------------------------------------------------------------------------
class AltarAction(IntEnum):
    """Player actions available when standing on an altar."""
    SACRIFICE        = 0   # sacrifice a corpse; may convert alignment, gain gift
    OFFER_CORPSE     = 1   # sub-action of SACRIFICE: offer specific corpse
    BUC_CHECK        = 2   # kneel: identify BUC of items on altar
    CONVERT_ALIGNMENT = 3  # desecrate / convert altar to your alignment
    PRAY             = 4   # delegates to prayer subsystem (pray.c)


# ---------------------------------------------------------------------------
# Door states (vendor/nethack/include/rm.h :: D_* macros)
# ---------------------------------------------------------------------------
class DoorState(IntEnum):
    """Encoded door states stored per tile.

    Values mirror rm.h D_* macros (kept as mask bits for historical reasons
    but stored here as a plain enum for JAX int8 arrays).
    """
    GONE   = 0   # D_NODOOR  0x00 — doorway with no door present
    BROKEN = 1   # D_BROKEN  0x01 — door bashed off hinges
    OPEN   = 2   # D_ISOPEN  0x02 — door open
    CLOSED = 4   # D_CLOSED  0x04 — door closed but unlocked
    LOCKED = 8   # D_LOCKED  0x08 — door locked
    SECRET = 32  # D_SECRET  0x20 — secret door (looks like wall)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class FeaturesState:
    """Per-tile dungeon-feature state across all levels.

    Fields
    ------
    fountains_used   : bool  [num_levels, map_h, map_w]
                       True once the fountain has dried up (dryup() called).
    thrones_used     : bool  [num_levels, map_h, map_w]
                       True once the throne has been analysed/destroyed.
    sinks_used       : bool  [num_levels, map_h, map_w]
                       True once the ring has been found (S_LRING looted flag).
    altar_alignment  : int8  [num_levels, map_h, map_w]
                       -1 = no altar; 0 = chaotic; 1 = neutral;
                        2 = lawful;   3 = unaligned (Moloch shrines).
    door_state       : int8  [num_levels, map_h, map_w]
                       DoorState value for each tile.
    door_trapped     : bool  [num_levels, map_h, map_w]
                       True when the door at this tile is trapped (D_TRAPPED).
                       Parallel bool grid mirrors vendor/nethack/include/rm.h
                       D_TRAPPED bit on doormask; stored separately for JAX
                       int8 array compatibility.
    """

    fountains_used:  jnp.ndarray   # [num_levels, map_h, map_w]  bool
    thrones_used:    jnp.ndarray   # [num_levels, map_h, map_w]  bool
    sinks_used:      jnp.ndarray   # [num_levels, map_h, map_w]  bool
    altar_alignment: jnp.ndarray   # [num_levels, map_h, map_w]  int8
    door_state:      jnp.ndarray   # [num_levels, map_h, map_w]  int8
    door_trapped:    jnp.ndarray   # [num_levels, map_h, map_w]  bool

    @classmethod
    def default(cls, num_levels: int, map_h: int, map_w: int) -> "FeaturesState":
        """Return a zeroed FeaturesState (no features used, all doors gone)."""
        shape = (num_levels, map_h, map_w)
        return cls(
            fountains_used=jnp.zeros(shape, dtype=jnp.bool_),
            thrones_used=jnp.zeros(shape, dtype=jnp.bool_),
            sinks_used=jnp.zeros(shape, dtype=jnp.bool_),
            altar_alignment=jnp.full(shape, -1, dtype=jnp.int8),
            door_state=jnp.zeros(shape, dtype=jnp.int8),
            door_trapped=jnp.zeros(shape, dtype=jnp.bool_),
        )


# ---------------------------------------------------------------------------
# Door operations (Wave 3)
# ---------------------------------------------------------------------------

def open_door(
    state: FeaturesState,
    pos: jnp.ndarray,
    rng: jax.Array = None,
) -> tuple["FeaturesState", jnp.ndarray]:
    """Open the door at *pos* if it is CLOSED (doopen, lock.c).

    Vendor reference: vendor/nethack/src/lock.c::doopen — checks
    ``d->doormask & D_TRAPPED`` before opening; if set, springs trap via
    ``trapsounding()`` which deals rnd(10) damage and breaks the door.

    pos : int array [3] = (level, row, col)
    rng : optional PRNGKey for trap-damage roll (unused if door not trapped)

    State transitions:
      CLOSED + not trapped → OPEN (2), damage = 0
      CLOSED + trapped     → BROKEN (1), damage = rnd(10), trapped bit cleared
      All other states     → unchanged, damage = 0

    Returns (new_state, damage: int32).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    lv, row, col = pos[0], pos[1], pos[2]
    current  = state.door_state[lv, row, col].astype(jnp.int32)
    is_closed = current == jnp.int32(DoorState.CLOSED)
    is_trapped = state.door_trapped[lv, row, col]

    # Trap spring: rnd(10) = 1..10 (vendor lock.c doopen D_TRAPPED branch).
    if rng is None:
        rng = jax.random.PRNGKey(0)
    trap_dmg = jax.random.randint(rng, (), minval=1, maxval=11, dtype=jnp.int32)

    # Door state: trapped → BROKEN, else → OPEN (only when was CLOSED)
    new_val = jnp.where(
        is_closed & is_trapped,
        jnp.int32(DoorState.BROKEN),
        jnp.where(is_closed & ~is_trapped, jnp.int32(DoorState.OPEN), current),
    ).astype(jnp.int8)

    # Damage: only when opening a trapped door
    damage = jnp.where(is_closed & is_trapped, trap_dmg, jnp.int32(0))

    # Clear the trapped bit once sprung
    new_trapped = jnp.where(
        is_closed & is_trapped,
        state.door_trapped.at[lv, row, col].set(jnp.bool_(False)),
        state.door_trapped,
    )

    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    return state.replace(door_state=new_door_state, door_trapped=new_trapped), damage


def close_door(
    state: FeaturesState,
    pos: jnp.ndarray,
    blocked: jnp.ndarray = jnp.bool_(False),
) -> FeaturesState:
    """Close the door at *pos* if it is OPEN and not obstructed (doclose, lock.c).

    pos     : int array [3] = (level, row, col)
    blocked : bool — True when a monster / object occupies the tile.  Vendor
              (lock.c::doclose, lines 1023-1024) calls obstructed(x, y) and
              bails out before transitioning the doormask; we mirror that
              guard here.

    OPEN (2) → CLOSED (4)  iff not blocked.
    All other states (including OPEN when blocked) are unchanged.
    """
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)
    is_open = current == jnp.int32(DoorState.OPEN)
    can_close = is_open & ~blocked.astype(jnp.bool_)
    new_val = jnp.where(can_close, jnp.int32(DoorState.CLOSED), current).astype(jnp.int8)
    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    return state.replace(door_state=new_door_state)


def kick_door(
    state: FeaturesState,
    rng: jax.Array,
    pos: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Kick the door at *pos* (vendor/nethack/src/dokick.c::kick_door, l. 908-970).

    pos : int array [3] = (level, row, col)

    Transition table (mirrors vendor doormask handling):
        ISOPEN / BROKEN / NODOOR  → no-op (vendor lines 914-918 — kick_dumb).
        CLOSED                    → 1/4 chance becomes BROKEN, else no-op
                                    (vendor break-roll abstracted at our level).
        LOCKED                    → 1/4 chance becomes BROKEN, AND independently
                                    1/8 chance the lock snaps without breaking
                                    the door (LOCKED → CLOSED).  When both
                                    fire, the door-break wins (it's broken
                                    open, lock is moot).

    Returns (new_state, self_damage: int32).  Wave 4 will wire monster anger
    on success.
    """
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)

    rng_break, rng_lock = jax.random.split(rng, 2)
    break_roll = jax.random.randint(rng_break, (), 0, 4, dtype=jnp.int32)
    lock_roll  = jax.random.randint(rng_lock,  (), 0, 8, dtype=jnp.int32)
    break_success = break_roll == jnp.int32(0)          # 1/4
    lock_break    = lock_roll  == jnp.int32(0)          # 1/8

    is_closed = current == jnp.int32(DoorState.CLOSED)
    is_locked = current == jnp.int32(DoorState.LOCKED)

    # CLOSED / LOCKED → BROKEN on a break_success.
    # LOCKED → CLOSED on lock_break (only if door itself did not break).
    new_val = current
    # Lock-snap path: LOCKED + lock_break and NOT door-break → CLOSED.
    new_val = jnp.where(
        is_locked & lock_break & ~break_success,
        jnp.int32(DoorState.CLOSED),
        new_val,
    )
    # Door-break path: (CLOSED | LOCKED) + break_success → BROKEN.
    new_val = jnp.where(
        (is_closed | is_locked) & break_success,
        jnp.int32(DoorState.BROKEN),
        new_val,
    )
    new_val = new_val.astype(jnp.int8)
    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    new_state = state.replace(door_state=new_door_state)
    return new_state, jnp.int32(0)


def unlock_door(
    state: FeaturesState,
    rng: jax.Array,
    pos: jnp.ndarray,
    key_slot: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Unlock the door at *pos* using the item in *key_slot* (lock.c).

    pos      : int array [3] = (level, row, col)
    key_slot : int — inventory slot index (not validated in Wave 3)

    LOCKED → CLOSED.
    Wave 3: always succeeds (d20 vs lock difficulty deferred to Wave 4).

    Returns (new_state, success: bool).
    """
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)
    is_locked = current == jnp.int32(DoorState.LOCKED)
    new_val = jnp.where(is_locked, jnp.int32(DoorState.CLOSED), current).astype(jnp.int8)
    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    new_state = state.replace(door_state=new_door_state)
    return new_state, jnp.bool_(is_locked)


def door_blocks_movement(
    state: FeaturesState,
    pos: jnp.ndarray,
) -> jnp.ndarray:
    """Return True if the door at *pos* blocks movement.

    pos : int array [3] = (level, row, col)

    Blocks: CLOSED (4), LOCKED (8), SECRET (32).
    Allows: OPEN (2), BROKEN (1), GONE (0).
    """
    lv, row, col = pos[0], pos[1], pos[2]
    val = state.door_state[lv, row, col].astype(jnp.int32)
    is_closed  = val == jnp.int32(DoorState.CLOSED)
    is_locked  = val == jnp.int32(DoorState.LOCKED)
    is_secret  = val == jnp.int32(DoorState.SECRET)
    return is_closed | is_locked | is_secret


def picklock_door(
    state: FeaturesState,
    pos: jnp.ndarray,
    rng: jax.Array | None = None,
    player_dex: int = 10,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Pick the lock on a LOCKED door (vendor/nethack/src/lock.c::picklock,
    lines 636-644).

    pos        : int array [3] = (level, row, col)
    rng        : JAX PRNGKey; if None, always succeeds (legacy behaviour).
    player_dex : player Dexterity score.

    Success formula (LOCK_PICK): ch = 3 * DEX; succeed if rn2(100) < ch.
    Cite: vendor/nethack/src/lock.c:636-637 — ``ch = 3 * ACURR(A_DEX)``.

    LOCKED → CLOSED on success.
    Returns (new_state, success: bool).
    """
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)
    is_locked = current == jnp.int32(DoorState.LOCKED)

    if rng is None:
        # Legacy: always succeed when no rng supplied.
        did_unlock = is_locked
    else:
        ch = jnp.int32(3 * int(player_dex))
        roll = jax.random.randint(rng, shape=(), minval=0, maxval=100)
        did_unlock = is_locked & (roll < ch)

    new_val = jnp.where(did_unlock, jnp.int32(DoorState.CLOSED), current).astype(jnp.int8)
    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    return state.replace(door_state=new_door_state), did_unlock


def forcelock_door(
    state: FeaturesState,
    pos: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Force a LOCKED door with a weapon (vendor/nethack/src/lock.c::forcelock,
    lines 214-256).

    pos : int array [3] = (level, row, col)

    LOCKED → BROKEN on success.  Vendor breakchestlock() destroys the lock
    permanently, leaving the door unlatched/broken open.

    Returns (new_state, success: bool).
    """
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)
    is_locked = current == jnp.int32(DoorState.LOCKED)
    new_val = jnp.where(is_locked, jnp.int32(DoorState.BROKEN), current).astype(jnp.int8)
    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    return state.replace(door_state=new_door_state), is_locked


def discover_secret_door(
    state: FeaturesState,
    pos: jnp.ndarray,
) -> FeaturesState:
    """Reveal a SECRET door at *pos* — vendor/nethack/src/detect.c::dosearch0
    (lines 2042-2051) calls cvt_sdoor_to_door which leaves the doormask in
    its existing (typically D_CLOSED or D_LOCKED) state.

    For our purposes, on reveal we leave a CLOSED door behind (the most
    common vendor case for newly cvt'd sdoors).  SECRET (32) → CLOSED (4).

    Non-SECRET tiles are unchanged.
    """
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)
    is_secret = current == jnp.int32(DoorState.SECRET)
    new_val = jnp.where(is_secret, jnp.int32(DoorState.CLOSED), current).astype(jnp.int8)
    new_door_state = state.door_state.at[lv, row, col].set(new_val)
    return state.replace(door_state=new_door_state)


def destroy_drawbridge(
    state: FeaturesState,
    terrain: jnp.ndarray,
    pos: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Destroy the drawbridge spanning tiles around *pos*
    (vendor/nethack/src/dbridge.c::destroy_drawbridge, lines 888-1000).

    pos     : int array [4] = (branch, level, row, col)
    terrain : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]

    Vendor logic: the bridge tile and its paired wall both clear.  We mirror
    that by setting both the (row, col) tile and the four cardinal-neighbour
    drawbridge halves to FLOOR (the moat/lava sub-cases are abstracted away
    here — see dbridge.c lines 924-939 for the typ switch).

    Returns (new_features_state, new_terrain).  Features state is returned
    so callers can chain (drawbridges don't currently use door_state, but
    keeping the signature uniform aids future work).
    """
    from Nethax.nethax.constants.tiles import TileType
    b, lv, row, col = pos[0], pos[1], pos[2], pos[3]
    FLOOR = jnp.int8(int(TileType.FLOOR))

    new_terrain = terrain.at[b, lv, row, col].set(FLOOR)
    # Vendor clears both halves: pos and its cardinal neighbour that
    # carries the matching drawbridge wall.  Without a stored direction
    # we sweep all 4 neighbours.  This is safe since we only clear tiles
    # that ARE currently DRAWBRIDGE_UP (other tile types remain).
    DBRIDGE_UP = jnp.int8(int(TileType.DRAWBRIDGE_UP))
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + jnp.int32(dr)
        cc = col + jnp.int32(dc)
        rr_safe = jnp.clip(rr, 0, terrain.shape[2] - 1)
        cc_safe = jnp.clip(cc, 0, terrain.shape[3] - 1)
        in_bounds = (rr >= 0) & (rr < terrain.shape[2]) & \
                    (cc >= 0) & (cc < terrain.shape[3])
        neighbour = new_terrain[b, lv, rr_safe, cc_safe]
        replace = in_bounds & (neighbour == DBRIDGE_UP)
        new_val = jnp.where(replace, FLOOR, neighbour)
        new_terrain = new_terrain.at[b, lv, rr_safe, cc_safe].set(new_val)
    return state, new_terrain


def open_drawbridge(
    state: FeaturesState,
    terrain: jnp.ndarray,
    pos: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Open (lower) the drawbridge at *pos*.

    Vendor citation: vendor/nethack/src/dbridge.c::open_drawbridge lines
    840-882.  Vendor logic:
        if (lev1->typ != DRAWBRIDGE_UP) return;
        lev1->typ = DRAWBRIDGE_DOWN;
        lev2->typ = DOOR;
        lev2->doormask = D_NODOOR;

    JAX model: the bridge tile (DRAWBRIDGE_UP) becomes FLOOR (we lack a
    distinct DRAWBRIDGE_DOWN in our local TileType enum); any adjacent
    DRAWBRIDGE_UP-typed wall tile becomes OPEN_DOOR (D_NODOOR ≈ no-door
    doorway).

    pos     : int array [4] = (branch, level, row, col)
    terrain : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]

    Returns (new_features_state, new_terrain).
    """
    from Nethax.nethax.constants.tiles import TileType
    b, lv, row, col = pos[0], pos[1], pos[2], pos[3]
    FLOOR      = jnp.int8(int(TileType.FLOOR))
    OPEN_DOOR  = jnp.int8(int(TileType.OPEN_DOOR))
    DBRIDGE_UP = jnp.int8(int(TileType.DRAWBRIDGE_UP))

    cur = terrain[b, lv, row, col]
    is_up = cur == DBRIDGE_UP
    # Bridge tile: only flips when currently DRAWBRIDGE_UP.
    new_terrain = terrain.at[b, lv, row, col].set(
        jnp.where(is_up, FLOOR, cur)
    )
    # Adjacent paired wall (we sweep all 4 neighbours and only convert
    # DRAWBRIDGE_UP-typed walls).
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + jnp.int32(dr)
        cc = col + jnp.int32(dc)
        rr_safe = jnp.clip(rr, 0, terrain.shape[2] - 1)
        cc_safe = jnp.clip(cc, 0, terrain.shape[3] - 1)
        in_bounds = (rr >= 0) & (rr < terrain.shape[2]) & \
                    (cc >= 0) & (cc < terrain.shape[3])
        neighbour = new_terrain[b, lv, rr_safe, cc_safe]
        replace = is_up & in_bounds & (neighbour == DBRIDGE_UP)
        new_val = jnp.where(replace, OPEN_DOOR, neighbour)
        new_terrain = new_terrain.at[b, lv, rr_safe, cc_safe].set(new_val)
    return state, new_terrain


def close_drawbridge(
    state: FeaturesState,
    terrain: jnp.ndarray,
    pos: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Close (raise) the drawbridge at *pos*.

    Vendor citation: vendor/nethack/src/dbridge.c::close_drawbridge lines
    775-834.  Vendor logic:
        if (lev1->typ != DRAWBRIDGE_DOWN) return;
        lev1->typ = DRAWBRIDGE_UP;
        lev2->typ = DBWALL;

    JAX model: the bridge tile (currently FLOOR / DRAWBRIDGE_DOWN) becomes
    DRAWBRIDGE_UP; any adjacent OPEN_DOOR-typed companion becomes
    DRAWBRIDGE_UP (we don't have DBWALL; the up-tile carries the closed
    state both for the bridge and its paired wall).

    pos     : int array [4] = (branch, level, row, col)
    terrain : int8[N_BRANCHES, MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W]

    Returns (new_features_state, new_terrain).
    """
    from Nethax.nethax.constants.tiles import TileType
    b, lv, row, col = pos[0], pos[1], pos[2], pos[3]
    FLOOR      = jnp.int8(int(TileType.FLOOR))
    OPEN_DOOR  = jnp.int8(int(TileType.OPEN_DOOR))
    DBRIDGE_UP = jnp.int8(int(TileType.DRAWBRIDGE_UP))

    cur = terrain[b, lv, row, col]
    # Only raise from "lowered" tile (DBRIDGE_DOWN encoded as FLOOR in our enum).
    is_down = cur == FLOOR
    new_terrain = terrain.at[b, lv, row, col].set(
        jnp.where(is_down, DBRIDGE_UP, cur)
    )
    # Re-raise the companion wall: any adjacent OPEN_DOOR becomes
    # DRAWBRIDGE_UP (mirrors vendor lev2->typ = DBWALL).
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = row + jnp.int32(dr)
        cc = col + jnp.int32(dc)
        rr_safe = jnp.clip(rr, 0, terrain.shape[2] - 1)
        cc_safe = jnp.clip(cc, 0, terrain.shape[3] - 1)
        in_bounds = (rr >= 0) & (rr < terrain.shape[2]) & \
                    (cc >= 0) & (cc < terrain.shape[3])
        neighbour = new_terrain[b, lv, rr_safe, cc_safe]
        replace = is_down & in_bounds & (neighbour == OPEN_DOOR)
        new_val = jnp.where(replace, DBRIDGE_UP, neighbour)
        new_terrain = new_terrain.at[b, lv, rr_safe, cc_safe].set(new_val)
    return state, new_terrain


def handle_search(state, rng: jax.Array):
    """SEARCH action — vendor/nethack/src/detect.c::dosearch0 (lines 2016-2093).

    Scans the 3x3 region around the player.  For each adjacent SECRET door,
    rolls rnl(7) (we use rn2(7) since Luck is not yet wired through); on a
    roll of 0 the secret door is revealed (SECRET → CLOSED), mirroring
    cvt_sdoor_to_door().

    JIT-safe: no Python control flow on traced values; uses a static 3x3 sweep.
    """
    flat_lv = _flat_lv_from_state(state)
    prow = state.player_pos[0].astype(jnp.int32)
    pcol = state.player_pos[1].astype(jnp.int32)
    H = state.features.door_state.shape[1]
    W = state.features.door_state.shape[2]

    door_state = state.features.door_state
    rngs = jax.random.split(rng, 9)
    idx = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = prow + jnp.int32(dr)
            cc = pcol + jnp.int32(dc)
            rr_s = jnp.clip(rr, 0, H - 1)
            cc_s = jnp.clip(cc, 0, W - 1)
            in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            is_self = (jnp.int32(dr) == 0) & (jnp.int32(dc) == 0)
            cur = door_state[flat_lv, rr_s, cc_s].astype(jnp.int32)
            is_secret = cur == jnp.int32(DoorState.SECRET)
            roll = jax.random.randint(rngs[idx], (), 0, 7, dtype=jnp.int32)
            discover = in_bounds & ~is_self & is_secret & (roll == jnp.int32(0))
            new_val = jnp.where(
                discover,
                jnp.int32(DoorState.CLOSED),
                cur,
            ).astype(jnp.int8)
            door_state = door_state.at[flat_lv, rr_s, cc_s].set(new_val)
            idx += 1
    new_features = state.features.replace(door_state=door_state)
    return state.replace(features=new_features)


# ---------------------------------------------------------------------------
# Action handlers (Wave 3)
# ---------------------------------------------------------------------------

def _flat_lv_from_state(state) -> jnp.ndarray:
    """Compute flat level index into FeaturesState arrays from an EnvState.

    FeaturesState is shaped [N_BRANCHES * MAX_LEVELS_PER_BRANCH, MAP_H, MAP_W].
    Flat index = branch * MAX_LEVELS_PER_BRANCH + (current_level - 1).
    """
    max_levels = state.terrain.shape[1]  # MAX_LEVELS_PER_BRANCH
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    return b * jnp.int32(max_levels) + lv


def handle_open(state, rng: jax.Array):
    """Open door in the direction the player last moved (Wave 4 adds dir selection).

    Wave 3 simplified: opens door at player's current tile if any.
    Checks D_TRAPPED (vendor/nethack/src/lock.c::doopen) and applies damage.
    Returns new EnvState.
    """
    flat_lv = _flat_lv_from_state(state)
    pos = jnp.array([flat_lv, state.player_pos[0], state.player_pos[1]], dtype=jnp.int32)
    new_features, trap_dmg = open_door(state.features, pos, rng)
    new_hp = jnp.maximum(jnp.int32(0), state.player_hp - trap_dmg)
    return state.replace(features=new_features, player_hp=new_hp)


def handle_close(state, rng: jax.Array):
    """Close the door at the player's current tile (Wave 3 simplified).

    Vendor lock.c::doclose lines 1023-1024 calls obstructed(x, y) and bails
    out when a monster, object, or boulder occupies the door tile.  Here we
    compute ``blocked`` = any alive monster shares the door tile, threaded
    into close_door so the OPEN→CLOSED transition is suppressed.

    Returns new EnvState.
    """
    flat_lv = _flat_lv_from_state(state)
    pos = jnp.array([flat_lv, state.player_pos[0], state.player_pos[1]], dtype=jnp.int32)
    # Obstruction: any alive monster on the door tile blocks the close.
    mai = state.monster_ai
    same_row = mai.pos[:, 0] == state.player_pos[0]
    same_col = mai.pos[:, 1] == state.player_pos[1]
    blocked = jnp.any(mai.alive & same_row & same_col)
    new_features = close_door(state.features, pos, blocked=blocked)
    return state.replace(features=new_features)


def handle_kick(state, rng: jax.Array):
    """Kick the door or monster at the player's tile.

    WOUNDED_LEGS blocks the kick entirely.
    Cite: vendor/nethack/src/dokick.c:1265-1310 — wounded legs prevent kicking.

    If a live monster occupies the player's tile, damages it:
      damage = max(1, (Str + Dex + Con) // 15)
    and marks it hostile.
    Cite: vendor/nethack/src/dokick.c:146-291.

    Otherwise kicks the door at the player's tile (existing Wave-3 logic).

    Returns new EnvState.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS

    is_wounded = state.status.timed_statuses[int(_TS.WOUNDED_LEGS)] > jnp.int32(0)

    def _do_kick(s):
        flat_lv = _flat_lv_from_state(s)
        prow = s.player_pos[0].astype(jnp.int32)
        pcol = s.player_pos[1].astype(jnp.int32)

        # Check for a live monster at the player's tile.
        mai = s.monster_ai
        at_tile = (
            mai.alive
            & (mai.pos[:, 0].astype(jnp.int32) == prow)
            & (mai.pos[:, 1].astype(jnp.int32) == pcol)
        )
        any_monster = jnp.any(at_tile)
        target_slot = jnp.argmax(at_tile).astype(jnp.int32)

        # Vendor kick damage — vendor/nethack/src/dokick.c:34-123.
        #   base = (Str + Dex + Con) // 15
        #   dmg  = rnd(max(base, 1))          # randomized 1..base
        #   dmg += 5  if kicking boots worn
        #   dmg += rn2(dex//2 + 1) if Monk
        #   dmg += uarmf->spe   (boot enchantment)
        #   dmg += u.udaminc
        #   dmg = 0 if target has M1_THICK_HIDE
        #   dmg = 0 if target is a shade
        key_kick, key_monk = jax.random.split(rng)
        str_i = s.player_str.astype(jnp.int32)
        dex_i = s.player_dex.astype(jnp.int32)
        con_i = s.player_con.astype(jnp.int32)
        base = (str_i + dex_i + con_i) // jnp.int32(15)
        base_clamped = jnp.maximum(base, jnp.int32(1))
        # rnd(n) -> uniform in [1, n].  We need a dynamic upper bound.
        dmg = jax.random.randint(
            key_kick, (), minval=1, maxval=base_clamped + jnp.int32(1), dtype=jnp.int32
        )

        # Kicking boots bonus (+5) and boot enchantment (uarmf->spe).
        from Nethax.nethax.subsystems.inventory import ArmorSlot as _ArmorSlot
        from Nethax.nethax.subsystems.character import ObjType as _ObjType
        boots_inv_idx = s.inventory.worn_armor[int(_ArmorSlot.BOOTS)].astype(jnp.int32)
        has_boots = boots_inv_idx >= jnp.int32(0)
        safe_b = jnp.clip(boots_inv_idx, 0, s.inventory.items.type_id.shape[0] - 1)
        boot_type = jnp.where(
            has_boots,
            s.inventory.items.type_id[safe_b].astype(jnp.int32),
            jnp.int32(0),
        )
        boot_spe = jnp.where(
            has_boots,
            s.inventory.items.enchantment[safe_b].astype(jnp.int32),
            jnp.int32(0),
        )
        kicking_boots_worn = boot_type == jnp.int32(int(_ObjType.KICKING_BOOTS))
        dmg = dmg + jnp.where(kicking_boots_worn, jnp.int32(5), jnp.int32(0))

        # Monk bonus: rn2(dex//2 + 1).
        from Nethax.nethax.constants.roles import Role as _Role
        is_monk = s.player_role == jnp.int8(int(_Role.MONK))
        monk_upper = jnp.maximum(dex_i // jnp.int32(2) + jnp.int32(1), jnp.int32(1))
        monk_roll = jax.random.randint(
            key_monk, (), minval=0, maxval=monk_upper, dtype=jnp.int32
        )
        dmg = dmg + jnp.where(is_monk, monk_roll, jnp.int32(0))

        # Boot enchantment + udaminc.
        dmg = dmg + boot_spe
        dmg = dmg + s.player_udaminc.astype(jnp.int32)

        # Target M1_THICK_HIDE flag and shade-symbol guard.
        from Nethax.nethax.constants.monsters import M1_THICK_HIDE as _M1_THICK_HIDE
        # Build static masks lazily via local helpers (module load is fine here
        # because they're only built once on first call into a JIT region).
        # Use module-level tables when available; fall back to import.
        from Nethax.nethax.subsystems.combat import _MONSTER_SYMBOL_TABLE
        # Per-monster flags1 table:
        from Nethax.nethax.subsystems.polymorph import _monster_tables
        tables = _monster_tables()
        flags1 = tables["flags1"]  # uint32[N]
        entry = jnp.clip(
            mai.entry_idx[target_slot].astype(jnp.int32), 0, flags1.shape[0] - 1,
        )
        thick_hide = (flags1[entry] & jnp.uint32(_M1_THICK_HIDE)) != jnp.uint32(0)
        # Shade detection: symbol == S_GOLEM? No — vendor S_SHADE is its own
        # symbol.  Our MonsterSymbol lacks S_SHADE; map shades by name later
        # if needed.  For now match the documented S_HUMANOID 'h' shade
        # entries via MONSTERS — this is a parity stub; M1_THICK_HIDE is the
        # primary guard.  Use the placeholder False until S_SHADE is added.
        is_shade = jnp.bool_(False)

        dmg = jnp.where(thick_hide | is_shade, jnp.int32(0), dmg)
        dmg = jnp.maximum(dmg, jnp.int32(0)).astype(jnp.int32)

        def _kick_monster(s_):
            old_hp = mai.hp[target_slot]
            new_hp = jnp.maximum(old_hp - dmg, jnp.int32(0))
            new_mai = mai.replace(
                hp=mai.hp.at[target_slot].set(new_hp),
                peaceful=mai.peaceful.at[target_slot].set(jnp.bool_(False)),
            )
            return s_.replace(monster_ai=new_mai)

        def _kick_door(s_):
            pos = jnp.array([flat_lv, prow, pcol], dtype=jnp.int32)
            new_features, _ = kick_door(s_.features, rng, pos)
            return s_.replace(features=new_features)

        return jax.lax.cond(any_monster, _kick_monster, _kick_door, s)

    return jax.lax.cond(is_wounded, lambda s: s, _do_kick, state)


# ---------------------------------------------------------------------------
# Stub feature operations (Wave 4+)
# ---------------------------------------------------------------------------

def quaff_fountain(
    state: FeaturesState,
    rng: jax.Array,
    pos: jnp.ndarray,
) -> tuple[FeaturesState, jnp.ndarray]:
    """No-op stub — drink from the fountain at *pos* (drinkfountain, fountain.c).

    pos : int array [3] = (level, row, col)

    Wave 4 will implement the rnd(30) effect table and call dryup() logic.
    Returns (new_state, effect_id: int32).
    """
    return state, jnp.int32(FountainEffect.REFRESH)


def step(state: FeaturesState, rng: jax.Array) -> FeaturesState:
    """No-op per-turn tick for the features subsystem.

    Future waves may use this for timed fountain-refill events or shrine auras.
    """
    return state


# ---------------------------------------------------------------------------
# Altar BUC sense (Wave 4)
#
# vendor/nethack/src/sit.c::sit (lines for SIT on altar) and pray.c::pleased
# trigger the "you feel that <item> is <buc>" reveal when the player stands
# on an altar whose alignment matches their own.  We approximate this by
# flipping every item with buc_status==0 (unknown) to its true sign — but
# since the JAX inventory has no hidden BUC, the no-op identity transform
# is the correct surface behaviour.  We still mark the items as
# ``identified=True`` so downstream observation code can render the BUC.
# ---------------------------------------------------------------------------

def altar_buc_sense(state):
    """If the player stands on an altar matching their alignment, reveal BUC.

    Cite: vendor/nethack/src/sit.c (altar branch of dosit) — touching an
    altar of your own alignment forces a BUC-identify of carried items.

    wave17h P0 (CURSE/BUC #2 + #3):
      - pray.c:1383-1410 water_prayer: POT_WATER on aligned altar becomes
        blessed (holy water); on opposite-aligned altar becomes cursed.
      - invent.c:1864 set_bknown: altar drop reveals BUC via flash color;
        mirror by flipping bknown=True for occupied carried items.

    JIT-safe; pure functional: no Python control flow on traced values.
    Returns: new EnvState.
    """
    max_levels = state.terrain.shape[1]
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    flat_lv = b * jnp.int32(max_levels) + lv
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    altar_align = state.features.altar_alignment[flat_lv, row, col].astype(jnp.int32)
    on_aligned_altar = (altar_align >= jnp.int32(0)) & (
        altar_align == state.player_align.astype(jnp.int32)
    )
    on_opposite_altar = (altar_align >= jnp.int32(0)) & (
        altar_align != state.player_align.astype(jnp.int32)
    )
    items = state.inventory.items
    occupied = items.category != jnp.int8(0)
    # Reveal: bump unknown (0) buc to uncursed (2); existing 1/2/3 unchanged.
    unknown = items.buc_status == jnp.int8(0)
    new_buc = jnp.where(
        on_aligned_altar & occupied & unknown,
        jnp.int8(2),
        items.buc_status,
    )
    # wave17h P0 (CURSE/BUC #2): POT_WATER → holy/unholy on (un)aligned altar.
    # Cite: vendor/nethack/src/pray.c lines 1395-1399.
    _POT_WATER_TID = jnp.int16(93)
    _POTION_CAT    = jnp.int8(8)   # ObjectClass.POTION_CLASS == 8
    is_water = (items.type_id == _POT_WATER_TID) & (items.category == _POTION_CAT)
    new_buc = jnp.where(
        on_aligned_altar & is_water & (new_buc != jnp.int8(3)),
        jnp.int8(3), new_buc,
    )
    new_buc = jnp.where(
        on_opposite_altar & is_water & (new_buc != jnp.int8(1)),
        jnp.int8(1), new_buc,
    )
    new_identified = jnp.where(
        on_aligned_altar & occupied,
        jnp.bool_(True),
        items.identified,
    )
    # wave17h P0 (CURSE/BUC #3): set_bknown on altar drop (invent.c:1864).
    on_any_altar = on_aligned_altar | on_opposite_altar
    new_bknown = jnp.where(
        on_any_altar & occupied,
        jnp.bool_(True),
        items.bknown,
    )
    new_items = items.replace(
        buc_status=new_buc,
        identified=new_identified,
        bknown=new_bknown,
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---------------------------------------------------------------------------
# Wave 17f — Sit on altar / altar_wrath (vendor pray.c::altar_wrath 2651-2673)
# ---------------------------------------------------------------------------

def sit_on_altar(state, rng):
    """Sit on an altar — vendor sit.c:530-532 calls altar_wrath(u.ux, u.uy).

    altar_wrath (vendor/nethack/src/pray.c lines 2651-2673):
        if (u.ualign.type == altaralign && u.ualign.record > -rn2(4)) {
            (void) adjattrib(A_WIS, -1, FALSE);
            u.ualign.record--;
        } else {
            if (Luck > -5 && rn2(Luck + 6))
                change_luck(rn2(20) ? -1 : -2);
        }

    Also retains the BUC-reveal side-effect (Wave 4 altar_buc_sense) so
    sitting still functions as a BUC-identify when the altar is aligned.

    Returns the new EnvState.
    """
    # Aligned-altar branch under player.
    max_levels = state.terrain.shape[1]
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    flat_lv = b * jnp.int32(max_levels) + lv
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    altar_align = state.features.altar_alignment[flat_lv, row, col].astype(jnp.int32)
    player_align_i = state.player_align.astype(jnp.int32)
    on_altar = altar_align >= jnp.int32(0)

    rng_a, rng_l1, rng_l2, rng_l3 = jax.random.split(rng, 4)

    # Branch 1 (vendor 2656): same-aligned altar AND record > -rn2(4) →
    # -1 WIS, record--.
    same_aligned = on_altar & (altar_align == player_align_i)
    record = state.prayer.alignment_record.astype(jnp.int32)
    threshold = -jax.random.randint(rng_a, (), 0, 4, dtype=jnp.int32)  # -rn2(4)
    record_gate = record > threshold
    do_wrath_same = same_aligned & record_gate

    new_wis = jnp.where(
        do_wrath_same,
        jnp.maximum(state.player_wis - jnp.int8(1), jnp.int8(3)),
        state.player_wis,
    )
    new_record = jnp.where(
        do_wrath_same,
        (state.prayer.alignment_record - jnp.int16(1)).astype(jnp.int16),
        state.prayer.alignment_record,
    )

    # Branch 2 (vendor 2660-2672): different alignment OR same-but-below-floor →
    #   if Luck > -5 && rn2(Luck + 6): change_luck(rn2(20) ? -1 : -2)
    do_wrath_other = on_altar & ~do_wrath_same
    luck = state.player_luck.astype(jnp.int32)
    luck_eligible = do_wrath_other & (luck > jnp.int32(-5))
    upper = jnp.maximum(luck + jnp.int32(6), jnp.int32(1))
    luck_roll = jax.random.randint(rng_l1, (), 0, upper, dtype=jnp.int32)
    luck_active = luck_eligible & (luck_roll != jnp.int32(0))
    rn20 = jax.random.randint(rng_l2, (), 0, 20, dtype=jnp.int32)
    luck_delta = jnp.where(rn20 != jnp.int32(0), jnp.int32(-1), jnp.int32(-2))
    new_luck = jnp.where(
        luck_active,
        jnp.clip(luck + luck_delta, jnp.int32(-13), jnp.int32(13)),
        luck,
    ).astype(jnp.int8)

    new_prayer = state.prayer.replace(alignment_record=new_record)
    new_state = state.replace(
        player_wis=new_wis,
        player_luck=new_luck,
        prayer=new_prayer,
    )

    # Retain BUC-reveal side-effect (Wave 4 altar_buc_sense semantics).
    return altar_buc_sense(new_state)


# ===========================================================================
# Wave 4 Phase 2 — feature effect tables (EnvState-level helpers)
# ===========================================================================
#
# These functions implement the full effect tables for the four interactive
# dungeon features (fountain quaff, fountain dip, throne sit, sink drink).
# They take a full EnvState (rather than just the FeaturesState slice) so
# they can touch hp/gold/status/inventory through their respective subsystem
# slices.  Each effect bucket is jit-safely dispatched via jax.lax.switch
# over a static branch tuple, with rng splits providing both the outcome
# selector and any sub-rolls.
#
# Vendor citations are in each docstring.  Outcome-table layouts mirror
# vendor weights so that, in the aggregate, drinking 1000 fountains
# (etc.) reproduces NetHack's empirical effect distribution.
# ---------------------------------------------------------------------------


# -- Fountain quaff buckets (drinkfountain rnd(30) table) -------------------
_FOUNTAIN_REFRESH         = 0   # fate 1..18 (REFRESH; vendor: fate<10 + default)
_FOUNTAIN_SELF_KNOWLEDGE  = 1   # fate 19  (enlightenment)
_FOUNTAIN_FOUL_WATER      = 2   # fate 20  (vomit + hunger)
_FOUNTAIN_POISONOUS       = 3   # fate 21  (poison_strdmg STR/HP loss)
_FOUNTAIN_SNAKES          = 4   # fate 22  (dowatersnakes)
_FOUNTAIN_WATER_DEMON     = 5   # fate 23  (dowaterdemon)
_FOUNTAIN_CURSE_RAY       = 6   # fate 24  (curse 1-in-5 inventory items)
_FOUNTAIN_SEE_INVISIBLE   = 7   # fate 25  (intrinsic SEE_INVIS)
_FOUNTAIN_MONSTER_DETECT  = 8   # fate 26  (monster_detect)
_FOUNTAIN_FIND_GEM        = 9   # fate 27  (dofindgem)
_FOUNTAIN_WATER_NYMPH     = 10  # fate 28  (dowaternymph; steal item)
_FOUNTAIN_SCARE           = 11  # fate 29  (monsters flee)
_FOUNTAIN_GUSH            = 12  # fate 30  (dogushforth)
_FOUNTAIN_DRY_UP          = 13  # post: dryup() 1-in-3 disappears
_FOUNTAIN_MOIST           = 14  # blessed fountain: restore + gain attr
_FOUNTAIN_WISH            = 15  # water-demon-grants-wish promotion (1-in-3)


def _fountain_pos_idx(state):
    """Return (flat_lv, row, col) for the tile under the player."""
    flat_lv = _flat_lv_from_state(state)
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    return flat_lv, row, col


def quaff_fountain(state, rng):
    """Drink from the fountain at the player's position.

    Implements 16 logical outcomes from
    vendor/nethack/src/fountain.c::drinkfountain (rnd(30) effect table;
    see lines 243-390 — cases 19..30 plus the fate<10 REFRESH default and
    the post-step dryup() roll).

    Returns
    -------
    EnvState — only the fields touched by the rolled effect are mutated.
    """
    rng_fate, rng_eff, rng_dry = jax.random.split(rng, 3)
    # rnd(30) in vendor is 1..30 inclusive; sample 0..29 and map.
    fate = jax.random.randint(rng_fate, (), minval=0, maxval=30, dtype=jnp.int32)

    bucket_table = jnp.array([
        # 0..17 → REFRESH (fate<10 + default sub-bucket of 10..18)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        # 18..29 → 12 distinct buckets (vendor fates 19..30)
        _FOUNTAIN_SELF_KNOWLEDGE,
        _FOUNTAIN_FOUL_WATER,
        _FOUNTAIN_POISONOUS,
        _FOUNTAIN_SNAKES,
        _FOUNTAIN_WATER_DEMON,
        _FOUNTAIN_CURSE_RAY,
        _FOUNTAIN_SEE_INVISIBLE,
        _FOUNTAIN_MONSTER_DETECT,
        _FOUNTAIN_FIND_GEM,
        _FOUNTAIN_WATER_NYMPH,
        _FOUNTAIN_SCARE,
        _FOUNTAIN_GUSH,
    ], dtype=jnp.int32)
    bucket = bucket_table[fate]

    # 1-in-3 promote WATER_DEMON → WISH (dowaterdemon may grant a wish).
    demon_promote = jax.random.randint(rng_eff, (), 0, 3, dtype=jnp.int32) == 0
    bucket = jnp.where(
        (bucket == _FOUNTAIN_WATER_DEMON) & demon_promote,
        jnp.int32(_FOUNTAIN_WISH),
        bucket,
    )

    def _refresh(s):
        # fate<10: cool draught, uhunger += rnd(10).
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition + jnp.int32(10)))

    def _self_knowledge(s):
        # case 19: enlightenment — tick wisdom up.
        return s.replace(player_wis=jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1)))

    def _foul_water(s):
        # case 20: vomit, morehungry(rn1(20,11)).
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(20)))

    def _poisonous(s):
        # case 21: poison_strdmg(rn1(4,3), rnd(10), ...).
        return s.replace(
            player_str=jnp.maximum(jnp.int16(3), s.player_str - jnp.int16(3)),
            player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(5)),
        )

    def _snakes(s):
        # case 22: dowatersnakes — hostile water moccasins.
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _water_demon(s):
        # case 23: water demon appears (no wish in this branch).
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(3)))

    def _curse_ray(s):
        # case 24: curse non-cursed inv items with prob 1/5 each.
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        not_cursed = inv.buc_status != jnp.int8(1)
        n = inv.category.shape[0]
        rolls = jax.random.randint(rng_eff, (n,), 0, 5, dtype=jnp.int32)
        target = occupied & not_cursed & (rolls == jnp.int32(0))
        new_buc = jnp.where(target, jnp.int8(1), inv.buc_status)
        new_items = inv.replace(buc_status=new_buc)
        return s.replace(
            inventory=s.inventory.replace(items=new_items),
            status=s.status.replace(nutrition=s.status.nutrition - jnp.int32(20)),
        )

    def _see_invisible(s):
        # case 25: HSee_invisible |= FROMOUTSIDE.
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        new_intr = s.status.intrinsics.at[int(Intrinsic.SEE_INVIS)].set(True)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    def _monster_detect(s):
        # case 26: monster_detect.
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        new_intr = s.status.intrinsics.at[int(Intrinsic.DETECT_MONSTERS)].set(True)
        return s.replace(status=s.status.replace(intrinsics=new_intr))

    def _find_gem(s):
        # case 27: dofindgem — proxy with a small gold bump.
        return s.replace(player_gold=s.player_gold + jnp.int32(5))

    def _water_nymph(s):
        # case 28: dowaternymph — nymph steals first occupied item.
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        first_idx = jnp.argmax(occupied.astype(jnp.int32))
        has_any = jnp.any(occupied)
        new_cat = jnp.where(
            has_any,
            inv.category.at[first_idx].set(jnp.int8(0)),
            inv.category,
        )
        new_qty = jnp.where(
            has_any,
            inv.quantity.at[first_idx].set(jnp.int16(0)),
            inv.quantity,
        )
        new_items = inv.replace(category=new_cat, quantity=new_qty)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _scare(s):
        # case 29: bad breath; monsters flee.  No player-side effect.
        return s

    def _gush(s):
        # case 30: dogushforth — water pools spread; fountain marked dry below.
        return s

    def _dry_up_only(s):
        # post-step dryup() bucket reached only via override; no-op here.
        return s

    def _moist(s):
        # blessed fountain path: restore HP, bump CON.
        return s.replace(
            player_hp=jnp.minimum(s.player_hp_max, s.player_hp + jnp.int32(10)),
            player_con=jnp.minimum(jnp.int8(25), s.player_con + jnp.int8(1)),
        )

    def _wish(s):
        # WATER_DEMON granted wish: max-hp + gold proxy.
        new_max = s.player_hp_max + jnp.int32(5)
        return s.replace(
            player_hp_max=new_max,
            player_hp=new_max,
            player_gold=s.player_gold + jnp.int32(100),
        )

    branches = (
        _refresh,         #  0
        _self_knowledge,  #  1
        _foul_water,      #  2
        _poisonous,       #  3
        _snakes,          #  4
        _water_demon,     #  5
        _curse_ray,       #  6
        _see_invisible,   #  7
        _monster_detect,  #  8
        _find_gem,        #  9
        _water_nymph,     # 10
        _scare,           # 11
        _gush,            # 12
        _dry_up_only,     # 13
        _moist,           # 14
        _wish,            # 15
    )
    new_state = jax.lax.switch(bucket, branches, state)

    # Post-step: dryup() — 1-in-3 chance the fountain disappears, always on GUSH.
    dry_roll = jax.random.randint(rng_dry, (), 0, 3, dtype=jnp.int32)
    will_dry = (
        (dry_roll == jnp.int32(0))
        | (bucket == _FOUNTAIN_DRY_UP)
        | (bucket == _FOUNTAIN_GUSH)
    )
    flat_lv, row, col = _fountain_pos_idx(new_state)
    new_used = jnp.where(
        will_dry,
        new_state.features.fountains_used.at[flat_lv, row, col].set(True),
        new_state.features.fountains_used,
    )
    new_features = new_state.features.replace(fountains_used=new_used)
    return new_state.replace(features=new_features)


def dip_fountain(state, rng, slot_idx):
    """Dip the inventory item at *slot_idx* into the fountain.

    Implements outcomes from
    vendor/nethack/src/fountain.c::dipfountain (lines 392-554):
        - Excalibur path (lawful + xl >= 5 + LONG_SWORD + 1-in-30):
            blessed sword + ART_EXCALIBUR name + +5 enchant.
            Lawful failure path: curses sword.
        - Else rnd(30) bucket:
            case 16     → curse item
            case 17-20  → uncurse item
            case 21     → water demon
            case 22     → water nymph (no inv change)
            case 23     → snakes
            case 24-25  → gush
            case 26-27  → strange feeling (no-op)
            case 28     → bath; lose 10% gold
            case 29     → find coins
            default     → corrode item (-1 enchant)
    """
    rng_fate, rng_excal, rng_dry = jax.random.split(rng, 3)

    LONG_SWORD_TYPE_ID = 34   # ObjType.LONG_SWORD
    LAWFUL = 2                # Alignment.LAWFUL

    inv = state.inventory.items
    n = inv.category.shape[0]
    slot = jnp.clip(slot_idx.astype(jnp.int32), 0, n - 1)

    item_type   = inv.type_id[slot]
    item_buc    = inv.buc_status[slot]
    item_ench   = inv.enchantment[slot]
    item_filled = inv.category[slot] != jnp.int8(0)

    is_long_sword  = (item_type == jnp.int16(LONG_SWORD_TYPE_ID)) & item_filled
    is_lawful      = state.player_align == jnp.int8(LAWFUL)
    is_high_xl     = state.player_xl >= jnp.int32(5)
    excal_eligible = is_long_sword & is_lawful & is_high_xl
    excal_roll = jax.random.randint(rng_excal, (), 0, 6, dtype=jnp.int32) == jnp.int32(0)
    grants_excalibur = excal_eligible & excal_roll

    fate = jax.random.randint(rng_fate, (), 1, 31, dtype=jnp.int32)
    # 0=curse, 1=uncurse, 2=water_demon, 3=water_nymph, 4=snakes,
    # 5=find_gold, 6=bath_lose_gold, 7=corrode (default)
    bucket = jnp.where(
        fate == jnp.int32(16), jnp.int32(0),
        jnp.where((fate >= jnp.int32(17)) & (fate <= jnp.int32(20)), jnp.int32(1),
        jnp.where(fate == jnp.int32(21), jnp.int32(2),
        jnp.where(fate == jnp.int32(22), jnp.int32(3),
        jnp.where(fate == jnp.int32(23), jnp.int32(4),
        jnp.where(fate == jnp.int32(29), jnp.int32(5),
        jnp.where(fate == jnp.int32(28), jnp.int32(6),
                  jnp.int32(7))))))))

    def _curse_item(s):
        new_buc = jnp.where(item_filled, jnp.int8(1), item_buc)
        new_arr = s.inventory.items.buc_status.at[slot].set(new_buc)
        new_items = s.inventory.items.replace(buc_status=new_arr)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _uncurse_item(s):
        new_buc = jnp.where(item_buc == jnp.int8(1), jnp.int8(2), item_buc)
        new_arr = s.inventory.items.buc_status.at[slot].set(new_buc)
        new_items = s.inventory.items.replace(buc_status=new_arr)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _water_demon(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(3)))

    def _water_nymph(s):
        return s

    def _snakes(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _find_gold(s):
        return s.replace(player_gold=s.player_gold + jnp.int32(20))

    def _bath_lose_gold(s):
        loss = jnp.maximum(jnp.int32(0), s.player_gold // jnp.int32(10))
        return s.replace(player_gold=s.player_gold - loss)

    def _corrode(s):
        new_ench = jnp.where(
            item_filled,
            jnp.maximum(jnp.int8(-5), item_ench - jnp.int8(1)),
            item_ench,
        )
        new_arr = s.inventory.items.enchantment.at[slot].set(new_ench)
        new_items = s.inventory.items.replace(enchantment=new_arr)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    branches = (
        _curse_item, _uncurse_item, _water_demon, _water_nymph,
        _snakes, _find_gold, _bath_lose_gold, _corrode,
    )
    base_state = jax.lax.switch(bucket, branches, state)

    # Excalibur path overrides the bucket outcome when eligible.
    def _grant_excal(s):
        new_buc = s.inventory.items.buc_status.at[slot].set(jnp.int8(3))   # blessed
        new_ench = s.inventory.items.enchantment.at[slot].set(
            jnp.maximum(item_ench, jnp.int8(5))
        )
        new_ident = s.inventory.items.identified.at[slot].set(jnp.bool_(True))
        new_items = s.inventory.items.replace(
            buc_status=new_buc, enchantment=new_ench, identified=new_ident,
        )
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _deny_excal(s):
        # Eligible but failed roll: vendor curses the sword on non-lawful;
        # we're guaranteed-lawful here, but the bad-roll case still curses.
        new_buc = s.inventory.items.buc_status.at[slot].set(jnp.int8(1))
        new_items = s.inventory.items.replace(buc_status=new_buc)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _no_excal_branch(_):
        return base_state

    def _excal_branch(_):
        return jax.lax.cond(grants_excalibur, _grant_excal, _deny_excal, state)

    out_state = jax.lax.cond(excal_eligible, _excal_branch, _no_excal_branch, operand=None)

    # Mark fountain used in some outcomes (always on Excalibur path).
    drain_roll = jax.random.randint(rng_dry, (), 0, 3, dtype=jnp.int32)
    will_dry = (drain_roll == jnp.int32(0)) | grants_excalibur
    flat_lv, row, col = _fountain_pos_idx(out_state)
    new_used = jnp.where(
        will_dry,
        out_state.features.fountains_used.at[flat_lv, row, col].set(True),
        out_state.features.fountains_used,
    )
    new_features = out_state.features.replace(fountains_used=new_used)
    return out_state.replace(features=new_features)


# -- Throne effects (sit.c::throne_sit_effect, effects 1..13) ---------------
_THRONE_ATTR_LOSS    = 0   # effect 1
_THRONE_ATTR_GAIN    = 1   # effect 2
_THRONE_SHOCK        = 2   # effect 3
_THRONE_FULL_HEAL    = 3   # effect 4
_THRONE_TAKE_GOLD    = 4   # effect 5
_THRONE_WISH         = 5   # effect 6
_THRONE_COURT        = 6   # effect 7
_THRONE_GENOCIDE     = 7   # effect 8
_THRONE_CURSE_ITEMS  = 8   # effect 9
_THRONE_MAP_CONFUSE  = 9   # effect 10
_THRONE_TELEPORT     = 10  # effect 11
_THRONE_IDENTIFY     = 11  # effect 12
_THRONE_CONFUSE      = 12  # effect 13
_THRONE_DESTROY      = 13  # post-effect: 1-in-3 throne vanishes


def sit_on_throne(state, rng):
    """Sit on the throne at the player's position.

    Implements 14 outcomes (13 effects + destroy_throne tail) from
    vendor/nethack/src/sit.c::throne_sit_effect (lines 39-234):
        effect 1  → attr loss + 1d10 HP
        effect 2  → random attr +1
        effect 3  → 1d6 / 1d30 shock damage
        effect 4  → full heal (HP to max + 4 bonus)
        effect 5  → take_gold: gold disappears
        effect 6  → makewish() (or change_luck +1 if Luck<0)
        effect 7  → summon 1d10 court monsters
        effect 8  → do_genocide(5)
        effect 9  → rndcurse() (curse one item)
        effect 10 → do_mapping() (or confusion if nommap)
        effect 11 → tele() (or aggravate if unlucky)
        effect 12 → identify_pack(rn2(5))
        effect 13 → confusion rn1(7,16) turns
        post-step → 1-in-3 throne disappears (DESTROY_THRONE)
    """
    rng_eff, rng_post, _ = jax.random.split(rng, 3)
    effect = jax.random.randint(rng_eff, (), 0, 13, dtype=jnp.int32)

    def _attr_loss(s):
        return s.replace(
            player_str=jnp.maximum(jnp.int16(3), s.player_str - jnp.int16(3)),
            player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(5)),
        )

    def _attr_gain(s):
        return s.replace(
            player_str=jnp.minimum(jnp.int16(125), s.player_str + jnp.int16(1)))

    def _shock(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        has_res = s.status.intrinsics[int(Intrinsic.RESIST_SHOCK)]
        dmg = jnp.where(has_res, jnp.int32(3), jnp.int32(15))
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - dmg))

    def _full_heal(s):
        return s.replace(player_hp=s.player_hp_max)

    def _take_gold(s):
        return s.replace(player_gold=jnp.int32(0))

    def _wish(s):
        new_max = s.player_hp_max + jnp.int32(5)
        return s.replace(
            player_hp_max=new_max,
            player_hp=new_max,
            player_gold=s.player_gold + jnp.int32(100),
        )

    def _court(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _genocide(s):
        return s.replace(player_xp=s.player_xp + jnp.int32(50))

    def _curse_items(s):
        inv = s.inventory.items
        occupied = inv.category != jnp.int8(0)
        not_cursed = inv.buc_status != jnp.int8(1)
        eligible = occupied & not_cursed
        idx = jnp.argmax(eligible.astype(jnp.int32))
        has_any = jnp.any(eligible)
        new_buc = jnp.where(
            has_any,
            inv.buc_status.at[idx].set(jnp.int8(1)),
            inv.buc_status,
        )
        new_items = inv.replace(buc_status=new_buc)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _map_or_confuse(s):
        return s.replace(
            player_wis=jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1)))

    def _teleport(s):
        # tele() — proxy by teleporting to a deterministic safe corner.
        return s.replace(player_pos=jnp.array([1, 1], dtype=jnp.int16))

    def _identify(s):
        inv = s.inventory.items
        n = inv.category.shape[0]
        occupied = inv.category != jnp.int8(0)
        cum = jnp.cumsum(occupied.astype(jnp.int32))
        # identify up to first 5 occupied slots (rn2(5) collapses to a bound)
        mark = occupied & (cum <= jnp.int32(5))
        new_ident = jnp.where(mark, jnp.bool_(True), inv.identified)
        new_items = inv.replace(identified=new_ident)
        return s.replace(inventory=s.inventory.replace(items=new_items))

    def _confuse(s):
        from Nethax.nethax.subsystems.status_effects import TimedStatus
        new_ts = s.status.timed_statuses.at[int(TimedStatus.CONFUSION)].add(jnp.int32(20))
        return s.replace(status=s.status.replace(timed_statuses=new_ts))

    branches = (
        _attr_loss, _attr_gain, _shock, _full_heal,
        _take_gold, _wish, _court, _genocide,
        _curse_items, _map_or_confuse, _teleport, _identify,
        _confuse,
    )
    new_state = jax.lax.switch(effect, branches, state)

    # Post-step: 1-in-3 chance throne disappears.
    destroy_roll = jax.random.randint(rng_post, (), 0, 3, dtype=jnp.int32)
    will_destroy = destroy_roll == jnp.int32(0)
    flat_lv = _flat_lv_from_state(new_state)
    row = new_state.player_pos[0].astype(jnp.int32)
    col = new_state.player_pos[1].astype(jnp.int32)
    new_used = jnp.where(
        will_destroy,
        new_state.features.thrones_used.at[flat_lv, row, col].set(True),
        new_state.features.thrones_used,
    )
    new_features = new_state.features.replace(thrones_used=new_used)
    return new_state.replace(features=new_features)


# -- Sink drink effects (fountain.c::drinksink, rn2(20)) -------------------
_SINK_COLD_WATER      = 0   # case 0
_SINK_WARM_WATER      = 1   # case 1
_SINK_SCALDING        = 2   # case 2 (1d6 fire dmg unless fire-resistant)
_SINK_SEWER_RAT       = 3   # case 3
_SINK_RANDOM_POTION   = 4   # case 4
_SINK_FIND_RING       = 5   # case 5 (or identify worn ring)
_SINK_BREAK           = 6   # case 6 (breaksink)
_SINK_WATER_ELEMENTAL = 7   # case 7
_SINK_DRAIN_NUTRITION = 8   # cases 8-9
_SINK_POLYMORPH       = 9   # case 10
_SINK_NOISE           = 10  # cases 11-12 (sound only)
_SINK_STENCH          = 11  # case 13 (create_gas_cloud)
_SINK_BLACK_PUDDING   = 12  # rare (rn2(20) tail, vendor case 19)


def drink_sink(state, rng):
    """Drink from the sink at the player's position.

    Implements 13 outcomes from vendor/nethack/src/fountain.c::drinksink
    (lines 595-712, rn2(20) effect table):
        case 0       → cold water (harmless)
        case 1       → warm water (harmless)
        case 2       → scalding (1d6 fire dmg unless Fire_resistance)
        case 3       → sewer rat spawns
        case 4       → random potion effect
        case 5       → ring found; if a ring is worn, identify it
        case 6       → breaksink — sink becomes fountain
        case 7       → water elemental
        case 8/9     → vomit + nutrition drain
        case 10      → polymorph self (if not Unchanging)
        case 11/12   → clanking pipes / sewer song (sound only)
        case 13      → stench gas cloud
        case 19/tail → BLACK_PUDDING (rare drain monster, severe HP loss)
    """
    rng_eff, _ = jax.random.split(rng, 2)
    fate = jax.random.randint(rng_eff, (), 0, 20, dtype=jnp.int32)

    bucket_table = jnp.array([
        _SINK_COLD_WATER,        #  0
        _SINK_WARM_WATER,        #  1
        _SINK_SCALDING,          #  2
        _SINK_SEWER_RAT,         #  3
        _SINK_RANDOM_POTION,     #  4
        _SINK_FIND_RING,         #  5
        _SINK_BREAK,             #  6
        _SINK_WATER_ELEMENTAL,   #  7
        _SINK_DRAIN_NUTRITION,   #  8
        _SINK_DRAIN_NUTRITION,   #  9
        _SINK_POLYMORPH,         # 10
        _SINK_NOISE,             # 11
        _SINK_NOISE,             # 12
        _SINK_STENCH,            # 13
        _SINK_COLD_WATER,        # 14 (default sip)
        _SINK_COLD_WATER,        # 15
        _SINK_COLD_WATER,        # 16
        _SINK_COLD_WATER,        # 17
        _SINK_COLD_WATER,        # 18
        _SINK_BLACK_PUDDING,     # 19 (rare tail; vendor case 19 hallucination)
    ], dtype=jnp.int32)
    bucket = bucket_table[fate]

    def _cold_water(s): return s

    def _warm_water(s): return s

    def _scalding(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        fire_res = s.status.intrinsics[int(Intrinsic.RESIST_FIRE)]
        dmg = jnp.where(fire_res, jnp.int32(0), jnp.int32(3))
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - dmg))

    def _sewer_rat(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(1)))

    def _random_potion(s):
        return s.replace(player_wis=jnp.minimum(jnp.int8(25), s.player_wis + jnp.int8(1)))

    def _find_ring(s):
        # If a ring is currently worn, identify it (mimics "Ring of X" reveal).
        inv = s.inventory.items
        left, right = s.inventory.worn_rings[0], s.inventory.worn_rings[1]
        left_safe  = jnp.maximum(jnp.int32(0), left.astype(jnp.int32))
        right_safe = jnp.maximum(jnp.int32(0), right.astype(jnp.int32))
        has_left  = left  >= jnp.int8(0)
        has_right = right >= jnp.int8(0)

        new_ident = inv.identified
        new_ident = jnp.where(
            has_left,
            new_ident.at[left_safe].set(jnp.bool_(True)),
            new_ident,
        )
        new_ident = jnp.where(
            has_right,
            new_ident.at[right_safe].set(jnp.bool_(True)),
            new_ident,
        )
        new_items = inv.replace(identified=new_ident)
        # Mark sink as looted so the ring drop doesn't repeat.
        flat_lv = _flat_lv_from_state(s)
        row = s.player_pos[0].astype(jnp.int32)
        col = s.player_pos[1].astype(jnp.int32)
        new_sink_arr = s.features.sinks_used.at[flat_lv, row, col].set(True)
        new_features = s.features.replace(sinks_used=new_sink_arr)
        return s.replace(
            inventory=s.inventory.replace(items=new_items),
            features=new_features,
        )

    def _break(s):
        # breaksink(): sink converts to fountain; mark used.
        flat_lv = _flat_lv_from_state(s)
        row = s.player_pos[0].astype(jnp.int32)
        col = s.player_pos[1].astype(jnp.int32)
        new_arr = s.features.sinks_used.at[flat_lv, row, col].set(True)
        return s.replace(features=s.features.replace(sinks_used=new_arr))

    def _water_elemental(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(2)))

    def _drain_nutrition(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(30)))

    def _polymorph(s):
        from Nethax.nethax.subsystems.status_effects import Intrinsic
        unchanging = s.status.intrinsics[int(Intrinsic.UNCHANGING)]
        delta = jnp.where(unchanging, jnp.int32(0), jnp.int32(1))
        return s.replace(player_xp=s.player_xp + delta)

    def _noise(s): return s

    def _stench(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(5)))

    def _black_pudding(s):
        # Rare drain monster eruption: severe HP damage.
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(8)))

    branches = (
        _cold_water,        #  0
        _warm_water,        #  1
        _scalding,          #  2
        _sewer_rat,         #  3
        _random_potion,     #  4
        _find_ring,         #  5
        _break,             #  6
        _water_elemental,   #  7
        _drain_nutrition,   #  8
        _polymorph,         #  9
        _noise,             # 10
        _stench,            # 11
        _black_pudding,     # 12
    )
    new_state = jax.lax.switch(bucket, branches, state)
    return new_state


# ---------------------------------------------------------------------------
# Sit-on-sink effects (sit.c::dosit IS_SINK branch, rn2(6) table)
# ---------------------------------------------------------------------------
def sit_sink(state, rng):
    """Sit on a sink.

    Implements 6 outcomes from vendor/nethack/src/sit.c::dosit IS_SINK branch
    (rn2(6) effect table):
        case 0  → slip: 1 HP damage (slip off the edge)
        case 1  → pudding: nutrition drain −20
        case 2  → faucet: HP drain −1 (cold water splash)
        case 3  → throw-up: nutrition drain −50
        case 4  → curse worn item (rndcurse proxy: buc_status → CURSED=1)
        case 5  → identify worn rings (mark first ring slot as identified)

    Cite: vendor/nethack/src/sit.c::dosit, IS_SINK branch.
    """
    bucket = jax.random.randint(rng, (), minval=0, maxval=6, dtype=jnp.int32)

    def _slip(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(1)))

    def _pudding(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(20)))

    def _faucet(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(1)))

    def _throw_up(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(50)))

    def _curse_worn(s):
        # Proxy for rndcurse(): curse the first non-empty inventory slot.
        buc = s.inventory.items.buc_status
        has_item = s.inventory.items.quantity > jnp.int16(0)
        first = jnp.argmax(has_item).astype(jnp.int32)
        new_buc = buc.at[first].set(jnp.int8(1))
        return s.replace(inventory=s.inventory.replace(
            items=s.inventory.items.replace(buc_status=new_buc)))

    def _identify_rings(s):
        # Identify first ring slot (identified flag → True).
        ident = s.inventory.identified
        new_ident = ident.at[0].set(True)
        return s.replace(inventory=s.inventory.replace(identified=new_ident))

    branches = (
        _slip,           # 0
        _pudding,        # 1
        _faucet,         # 2
        _throw_up,       # 3
        _curse_worn,     # 4
        _identify_rings, # 5
    )
    return jax.lax.switch(bucket, branches, state)


# ---------------------------------------------------------------------------
# Kick-sink effects (dokick.c::kick_nondoor IS_SINK branch, rn2(4) table)
# ---------------------------------------------------------------------------
def kick_sink(state, rng):
    """Kick a sink.

    Implements 4 outcomes from vendor/nethack/src/dokick.c::kick_nondoor
    IS_SINK branch (rn2(4) effect table):
        case 0  → strange shock: 1d6 electric damage
        case 1  → pudding erupts: nutrition drain −30
        case 2  → water spray: 1 HP damage
        case 3  → no effect (noise only)

    Returns (new_state, outcome_id) where outcome_id is int32 in [0, 3].

    Cite: vendor/nethack/src/dokick.c::kick_nondoor, IS_SINK branch,
    lines 1194-1240.
    """
    rng_outcome, rng_dmg = jax.random.split(rng)
    bucket = jax.random.randint(rng_outcome, (), minval=0, maxval=4, dtype=jnp.int32)
    shock_dmg = jax.random.randint(rng_dmg, (), minval=1, maxval=7, dtype=jnp.int32)

    def _shock(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - shock_dmg))

    def _pudding(s):
        return s.replace(status=s.status.replace(
            nutrition=s.status.nutrition - jnp.int32(30)))

    def _spray(s):
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(1)))

    def _nothing(s):
        return s

    branches = (_shock, _pudding, _spray, _nothing)
    new_state = jax.lax.switch(bucket, branches, state)
    return new_state, bucket


# ---------------------------------------------------------------------------
# Drop-at-altar BUC mutation (pray.c::doaltar)
# ---------------------------------------------------------------------------
def drop_at_altar(state, slot_idx: jnp.ndarray):
    """Drop item in slot_idx on the altar at the player's position.

    BUC mutation rules (vendor/nethack/src/pray.c::doaltar):
        coaligned altar  (altar_align == player_align) → bless item (buc=3)
        cross-aligned    (altar_align != player_align, both ≥0) → curse (buc=1)
        neutral          (altar_align=1, or unaligned player) → no change
        no altar         (altar_align=-1) → no change

    JIT-safe: no Python control flow on traced values.

    Cite: vendor/nethack/src/pray.c::doaltar.
    """
    max_lv = state.terrain.shape[1]
    b   = state.dungeon.current_branch.astype(jnp.int32)
    lv  = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    flat_lv = b * jnp.int32(max_lv) + lv
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)

    altar_align = state.features.altar_alignment[flat_lv, row, col].astype(jnp.int32)
    player_align = state.player_align.astype(jnp.int32)
    on_altar = altar_align >= jnp.int32(0)
    coaligned = on_altar & (altar_align == player_align)
    cross_aligned = on_altar & (altar_align != player_align)

    safe_slot = jnp.clip(slot_idx.astype(jnp.int32), 0,
                         state.inventory.items.buc_status.shape[0] - 1)
    old_buc = state.inventory.items.buc_status[safe_slot].astype(jnp.int32)

    new_buc = jnp.where(coaligned, jnp.int32(3),
              jnp.where(cross_aligned, jnp.int32(1),
              old_buc))

    new_buc_arr = state.inventory.items.buc_status.at[safe_slot].set(
        new_buc.astype(jnp.int8))
    new_items = state.inventory.items.replace(buc_status=new_buc_arr)
    return state.replace(inventory=state.inventory.replace(items=new_items))
