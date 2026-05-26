"""Dungeon-feature subsystem — fountains, thrones, sinks, altars, doors.

Canonical sources:
  vendor/nethack/src/fountain.c  — drinkfountain(), dipfountain(), drinksink(),
                                   dipsink(), dryup(), breaksink()
  vendor/nethack/src/sit.c       — throne_sit_effect() (effects 1-13)
  vendor/nethack/src/dokick.c    — kick_door(), kick_nondoor()
  vendor/nethack/src/lock.c      — doopen(), doclose(), picking locks
  vendor/nethack/src/dbridge.c   — drawbridge open/close mechanics
  vendor/nethack/include/rm.h    — D_NODOOR/D_BROKEN/D_ISOPEN/D_CLOSED/D_LOCKED

Status: door operations (open/close/kick/unlock/door_blocks_movement)
        and action handlers implemented in this module.

Implemented elsewhere (see linked modules for full vendor citations):
  - subsystems/fountain.py — drinkfountain / dipfountain / dryup
  - subsystems/throne.py   — throne_sit_effect (13-outcome rn1 table)
  - subsystems/water.py    — sink / drinksink (kick + quaff outcomes)
  - subsystems/prayer.py + subsystems/priest.py — sacrifice_on_altar
                              + alignment conversion + gcrownu reward

Still deferred (no concrete vendor caller wired):
  - kick_door: monster-anger side effect on a successful kick
  - unlock_door: real d20 roll vs lock difficulty
  - Vibrating square gateway end-game trigger (sounds.c)
  - Drawbridge open/close/collapse (dbridge.c)
  - Secret-door discovery via #search (detect.c::findit)
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
    altar_known      : bool  [num_levels, map_h, map_w]
                       True once player has stood on this altar (BUC sense /
                       sacrifice).  Mirrors vendor pray.c::doaltar where
                       altars become known on first interaction.
    altar_sacrifice_count : int8 [num_levels, map_h, map_w]
                       Counter of coaligned-corpse sacrifices on each altar
                       (vendor pray.c::dosacrifice — reset on altar
                       conversion, max ALTAR_CONVERT_THRESHOLD=5).
    altar_shrine     : bool [num_levels, map_h, map_w]
                       True iff the altar at this tile carries the AM_SHRINE
                       flag (vendor mkroom.c::mktemple line 618).  Only
                       shrine altars carry a peaceful priest.
    has_temple       : bool [num_levels]
                       True iff this level was placed via mkroom.c::mktemple
                       (vendor mkroom.c line 619: level.flags.has_temple = 1).
                       Required by priest.c::intemple gating.
    vault_pos        : int16 [num_levels, 2]
                       (row, col) of vault interior centre per level; (-1,-1)
                       means no vault.  Mirrors vendor vault.c::mk_vault.
    guard_slot       : int32 scalar
                       Monster-AI slot for the current vault guard; -1 = none.
                       Mirrors vendor vault.c::invault line 407 makemon call.
    guard_escort_active : bool scalar
                       True while the vault guard is escorting the player out
                       (vendor vault.c::gd_move line 888).
    lit              : bool [num_levels, map_h, map_w]
                       True iff the tile is permanently lit.  Mirrors vendor
                       ``levl[x][y].lit`` (vendor/nethack/include/rm.h line
                       165 — ``Bitfield(lit, 1)``).  Set during dungeon
                       generation per-room (vendor mklev.c::do_room_or_subroom
                       lines 249-255) and later toggled by light spells /
                       artifacts (vendor read.c::litroom, set_lit lines
                       2471-2488; artifact.c::arti_invoke line 2063).
    waslit           : bool [num_levels, map_h, map_w]
                       True once the hero has seen this tile lit.  Mirrors
                       vendor ``levl[x][y].waslit`` (vendor/nethack/include/
                       rm.h line 166 — ``Bitfield(waslit, 1)``).  Used by
                       vision/redraw logic to remember which tiles were lit
                       between visits (vendor display.c, vision.c).
    rememberedlit    : bool [num_levels, map_h, map_w]
                       True once the player has perceived this tile as lit
                       and has committed it to memory.  Semantic analog of
                       vendor display.c memory-glyph logic that tracks
                       previously-lit tiles for redraw between visits
                       (companion to ``waslit`` at vendor/nethack/include/
                       rm.h lines 165-166).  Set via :func:`mark_remembered`
                       when the player sees a lit tile.
    is_cavernous_lev : bool [num_levels]
                       Per-level flag — True iff this level is a "cavernous"
                       level (Gnomish Mines / random caves).  Mirrors
                       vendor ``svl.level.flags.is_cavernous_lev`` (vendor/
                       nethack/include/rm.h line 454 — ``Bitfield(
                       is_cavernous_lev, 1)``).  Set by vendor mkmap.c::
                       mkmap line 483 ``svl.level.flags.is_cavernous_lev =
                       TRUE`` after a walled+joined cave-build.  Gates
                       wall→corridor carving in vendor dig.c lines 495-497.
    """

    fountains_used:  jnp.ndarray   # [num_levels, map_h, map_w]  bool
    thrones_used:    jnp.ndarray   # [num_levels, map_h, map_w]  bool
    sinks_used:      jnp.ndarray   # [num_levels, map_h, map_w]  bool
    altar_alignment: jnp.ndarray   # [num_levels, map_h, map_w]  int8
    door_state:      jnp.ndarray   # [num_levels, map_h, map_w]  int8
    door_trapped:    jnp.ndarray   # [num_levels, map_h, map_w]  bool
    altar_known:     jnp.ndarray   # [num_levels, map_h, map_w]  bool
    altar_sacrifice_count: jnp.ndarray  # [num_levels, map_h, map_w]  int8
    altar_shrine:    jnp.ndarray   # [num_levels, map_h, map_w]  bool
    has_temple:      jnp.ndarray   # [num_levels]                bool
    vault_pos:       jnp.ndarray   # [num_levels, 2]             int16
    guard_slot:      jnp.ndarray   # scalar                       int32
    guard_escort_active: jnp.ndarray   # scalar                   bool
    lit:             jnp.ndarray   # [num_levels, map_h, map_w]  bool
    waslit:          jnp.ndarray   # [num_levels, map_h, map_w]  bool
    rememberedlit:   jnp.ndarray   # [num_levels, map_h, map_w]  bool
    is_cavernous_lev: jnp.ndarray  # [num_levels]                bool

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
            altar_known=jnp.zeros(shape, dtype=jnp.bool_),
            altar_sacrifice_count=jnp.zeros(shape, dtype=jnp.int8),
            altar_shrine=jnp.zeros(shape, dtype=jnp.bool_),
            has_temple=jnp.zeros((num_levels,), dtype=jnp.bool_),
            vault_pos=jnp.full((num_levels, 2), -1, dtype=jnp.int16),
            guard_slot=jnp.int32(-1),
            guard_escort_active=jnp.bool_(False),
            # Per-tile lighting (vendor rm.h lines 165-166: lit / waslit
            # bitfields).  Zero-init: dungeon generation later flips lit=True
            # for tiles inside lit rooms (vendor mklev.c lines 249-255).
            lit=jnp.zeros(shape, dtype=jnp.bool_),
            waslit=jnp.zeros(shape, dtype=jnp.bool_),
            # Per-tile remembered-lit (Wave 48b) — set when the player sees
            # a lit tile; persisted across visits for display redraw.
            # Companion to vendor waslit (rm.h line 166).
            rememberedlit=jnp.zeros(shape, dtype=jnp.bool_),
            # Per-level cavernous flag (vendor rm.h line 454:
            # ``Bitfield(is_cavernous_lev, 1)``).  Zero-init: cave-shaped
            # levels (Gnomish Mines) flip this True at generation time
            # (vendor mkmap.c line 483).  Used by dig.c lines 495-497 to
            # carve walls into CORR rather than D_NODOOR.
            is_cavernous_lev=jnp.zeros((num_levels,), dtype=jnp.bool_),
        )


# ---------------------------------------------------------------------------
# Door operations (Wave 3)
# ---------------------------------------------------------------------------

def open_door(
    state: FeaturesState,
    pos: jnp.ndarray,
    rng: jax.Array = None,
    level_difficulty: int | jnp.ndarray = 1,
) -> tuple["FeaturesState", jnp.ndarray]:
    """Open the door at *pos* if it is CLOSED (doopen, lock.c).

    Vendor reference: vendor/nethack/src/lock.c::doopen — checks
    ``d->doormask & D_TRAPPED`` before opening; if set, springs trap via
    ``b_trapped()`` (vendor/nethack/src/trap.c::b_trapped, lines 6693-6707)
    which deals ``rnd(5 + (lvl<5 ? lvl : 2 + lvl/2))`` damage where
    ``lvl = level_difficulty()`` (vendor trap.c:6696-6697).

    pos              : int array [3] = (level, row, col)
    rng              : optional PRNGKey for trap-damage roll (unused if
                       door not trapped)
    level_difficulty : current dungeon level used by vendor b_trapped to
                       scale damage; pass ``state.dungeon.current_level``
                       from EnvState callers.  Defaults to 1 (legacy
                       behaviour pre-plumbing).

    State transitions:
      CLOSED + not trapped → OPEN (2), damage = 0
      CLOSED + trapped     → GONE (D_NODOOR, 0), damage = rnd(5 + (lvl<5 ?
                             lvl : 2 + lvl/2)), trapped bit cleared
      All other states     → unchanged, damage = 0

    Vendor lock.c:907-913 sets ``door->doormask = D_NODOOR`` after
    ``b_trapped()``, NOT D_BROKEN — the trap obliterates the door rather
    than breaking it off its hinges.

    Returns (new_state, damage: int32).
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    lv, row, col = pos[0], pos[1], pos[2]
    current  = state.door_state[lv, row, col].astype(jnp.int32)
    is_closed = current == jnp.int32(DoorState.CLOSED)
    is_trapped = state.door_trapped[lv, row, col]

    # Trap damage formula — vendor/nethack/src/trap.c::b_trapped lines
    # 6696-6697: ``dmg = rnd(5 + (lvl < 5 ? lvl : 2 + lvl/2))``.
    lvl_i = jnp.asarray(level_difficulty, dtype=jnp.int32)
    bonus = jnp.where(lvl_i < jnp.int32(5), lvl_i, jnp.int32(2) + lvl_i // jnp.int32(2))
    dmg_upper = jnp.int32(5) + bonus  # rnd(n) → uniform 1..n
    trap_dmg = jax.random.randint(
        rng, (), minval=1, maxval=dmg_upper + jnp.int32(1), dtype=jnp.int32,
    )

    # Door state: trapped → GONE (D_NODOOR per lock.c:909),
    # else → OPEN (only when was CLOSED).
    new_val = jnp.where(
        is_closed & is_trapped,
        jnp.int32(DoorState.GONE),
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
    player_role: int | jnp.ndarray = -1,
) -> tuple[FeaturesState, jnp.ndarray]:
    """Pick the lock on a LOCKED door — vendor/nethack/src/lock.c::pick_lock
    (chance set at line 636-637) and picklock() (roll at line 98).

    pos         : int array [3] = (level, row, col)
    rng         : JAX PRNGKey; if None, always succeeds (legacy behaviour
                  kept for apply_tools.py:755 which rolls upstream).
    player_dex  : player Dexterity score.
    player_role : player Role enum value; when equal to ``Role.ROGUE`` (8)
                  the vendor +30 ``Role_if(PM_ROGUE)`` bonus is applied
                  (vendor lock.c:637).  Default ``-1`` skips the bonus,
                  preserving legacy caller behaviour.

    Success formula (LOCK_PICK): ch = 3 * ACURR(A_DEX) + 30 * Role_if(PM_ROGUE)
    (vendor lock.c:636-637).  Per-turn occupation roll (vendor lock.c:98)
    is ``if (rn2(100) >= chance) return 1;`` — i.e., success on
    ``rn2(100) < chance``.

    LOCKED → CLOSED on success.
    Returns (new_state, success: bool).
    """
    from Nethax.nethax.constants.roles import Role as _Role
    lv, row, col = pos[0], pos[1], pos[2]
    current = state.door_state[lv, row, col].astype(jnp.int32)
    is_locked = current == jnp.int32(DoorState.LOCKED)

    if rng is None:
        # Legacy: always succeed when no rng supplied.
        did_unlock = is_locked
    else:
        role_i = jnp.asarray(player_role, dtype=jnp.int32)
        rogue_bonus = jnp.where(
            role_i == jnp.int32(int(_Role.ROGUE)), jnp.int32(30), jnp.int32(0),
        )
        ch = jnp.int32(3 * int(player_dex)) + rogue_bonus
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


def scatter_iron_chain_debris(state, pos: jnp.ndarray, rng: jax.Array):
    """Scatter IRON_CHAIN ground items in a 2-cell ring around the
    destroyed drawbridge tile.

    Vendor cite: dbridge.c::destroy_drawbridge lines 949-958 — when the
    bridge collapses, ``rn2(6)`` IRON_CHAIN chunks are scattered in the
    immediate vicinity.  Prior Nethax port lacked this (silent leak per
    the dbridge.c audit).

    pos : int array [4] = (branch, level, row, col)
    """
    _OBJ_IRON_CHAIN_TID = 450
    _CAT_CHAIN = 16  # ItemCategory.CHAIN

    b, lv, row, col = pos[0], pos[1], pos[2], pos[3]
    g = state.ground_items

    # rn2(6) debris count = 0..5.  Drop into the 8-ring around pos.
    count = jax.random.randint(rng, (), 0, 6, dtype=jnp.int32)
    offsets = jnp.array(
        [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
        dtype=jnp.int32,
    )
    map_h = g.category.shape[2]
    map_w = g.category.shape[3]

    def _place(carry, args):
        gg, i = carry
        off = args
        will_place = i < count
        r = row + off[0]; c = col + off[1]
        in_bounds = (r >= 0) & (r < map_h) & (c >= 0) & (c < map_w)
        rs = jnp.clip(r, 0, map_h - 1); cs = jnp.clip(c, 0, map_w - 1)
        slot0_empty = gg.category[b, lv, rs, cs, 0] == jnp.int8(0)
        do_place = will_place & in_bounds & slot0_empty
        new_cat = jnp.where(do_place, jnp.int8(_CAT_CHAIN), gg.category[b, lv, rs, cs, 0])
        new_tid = jnp.where(do_place, jnp.int16(_OBJ_IRON_CHAIN_TID), gg.type_id[b, lv, rs, cs, 0])
        new_qty = jnp.where(do_place, jnp.int16(1), gg.quantity[b, lv, rs, cs, 0])
        new_gg = gg.replace(
            category=gg.category.at[b, lv, rs, cs, 0].set(new_cat),
            type_id=gg.type_id.at[b, lv, rs, cs, 0].set(new_tid),
            quantity=gg.quantity.at[b, lv, rs, cs, 0].set(new_qty),
        )
        return (new_gg, i + jnp.int32(1)), None

    (new_g, _), _ = jax.lax.scan(_place, (g, jnp.int32(0)), offsets)
    return state.replace(ground_items=new_g)


def process_bridge_entities_on_close(state, pos: jnp.ndarray, rng: jax.Array | None = None):
    """Crush any monsters standing on the bridge tile + paired wall tile when
    a drawbridge closes (vendor dbridge.c::do_entity lines 554-759 — entities
    on the bridge are processed for crush / drown / lava-burn).

    Two survival paths run before the crush kills a monster:

      1. ``e_missed`` dodge (dbridge.c::e_missed lines 495-525) — high
         luck/dex creatures can dodge the falling portcullis entirely.
         Vendor formula for a monster is ``rn2(20) > 12`` (~35% pass);
         survivors stay in place untouched.
      2. ``e_jumps`` jump-clear (dbridge.c::e_jumps lines 530-551) — those
         that don't dodge get a ``rn2(5)`` (1/5) chance to relocate to the
         first walkable 8-neighbour tile (FLOOR / CORRIDOR / OPEN_DOOR /
         ...).

    If both fail (no dodge AND (no jump roll OR no walkable neighbour)),
    the monster is crushed in place (HP → 0, alive → False).

    The hero is not processed here (vendor handles hero separately —
    hero-side crush damage is wired in a follow-up commit).

    Also processes the 4-neighbour pair-tile that will be raised
    (vendor lev2 in close_drawbridge).

    Cite: vendor/nethack/src/dbridge.c::do_entity lines 554-759;
          vendor/nethack/src/dbridge.c::e_missed lines 495-525;
          vendor/nethack/src/dbridge.c::e_jumps  lines 530-551.
    """
    from Nethax.nethax.constants.tiles import TileType
    b, lv, row, col = pos[0], pos[1], pos[2], pos[3]
    mai = state.monster_ai

    mr = mai.pos[:, 0].astype(jnp.int32)
    mc = mai.pos[:, 1].astype(jnp.int32)

    # Crush mask: monster alive AND on the bridge tile OR any 4-neighbour
    # tile that will become DRAWBRIDGE_UP (paired wall side).  We don't
    # need to gate on per-tile branch/level — Nethax monster_ai is
    # already scoped to the current level.
    on_bridge = (mr == row) & (mc == col)
    on_n = (mr == row - jnp.int32(1)) & (mc == col)
    on_s = (mr == row + jnp.int32(1)) & (mc == col)
    on_w = (mr == row) & (mc == col - jnp.int32(1))
    on_e = (mr == row) & (mc == col + jnp.int32(1))
    affected = mai.alive & (on_bridge | on_n | on_s | on_w | on_e)

    # --- e_jumped escape (dbridge.c::do_entity case e_jumped) -------------
    # If no rng provided, no jump roll fires (back-compat with callers that
    # don't yet thread randomness through close).  When rng is provided,
    # each affected monster rolls rn2(5); on a 0, attempt to relocate to
    # the first walkable 8-neighbour of its current tile that is not the
    # bridge/paired-wall set.
    n_slots = mai.alive.shape[0]
    terrain = state.terrain[b, lv]
    map_h = terrain.shape[0]
    map_w = terrain.shape[1]

    # Walkable predicate: not WALL/CLOSED_DOOR/VOID/DRAWBRIDGE_UP and
    # not deadly liquid (WATER/LAVA/POOL).  Conservative subset of vendor
    # e_survives_at — generic monsters drown/burn on liquid.
    _BAD_TILES = jnp.array(
        [
            int(TileType.VOID),
            int(TileType.WALL),
            int(TileType.CLOSED_DOOR),
            int(TileType.DRAWBRIDGE_UP),
            int(TileType.WATER),
            int(TileType.LAVA),
            int(TileType.POOL),
        ],
        dtype=jnp.int32,
    )

    def _tile_walkable(rr: jnp.ndarray, cc: jnp.ndarray) -> jnp.ndarray:
        in_bounds = (rr >= 0) & (rr < map_h) & (cc >= 0) & (cc < map_w)
        rs = jnp.clip(rr, 0, map_h - 1)
        cs = jnp.clip(cc, 0, map_w - 1)
        t = terrain[rs, cs].astype(jnp.int32)
        bad = jnp.any(_BAD_TILES == t)
        return in_bounds & (~bad)

    # 8-neighbour offsets (vendor: enexto picks any adjacent free tile).
    _OFFS = jnp.array(
        [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],
        dtype=jnp.int32,
    )

    if rng is None:
        # Back-compat path: no dodge/jump roll, behaviour matches legacy
        # implementation (everyone in the crush mask dies).
        crushed = affected
        new_pos = mai.pos
    else:
        # Two subkeys per monster: dodge roll + jump roll.
        rng_dodge, rng_jump = jax.random.split(rng, 2)
        keys_dodge = jax.random.split(rng_dodge, n_slots)
        keys_jump = jax.random.split(rng_jump, n_slots)

        def _per_monster(carry, args):
            k_dodge, k_jump, m_r, m_c, is_affected = args
            # --- e_missed dodge: rn2(20) > 12 (~35% pass) -----------------
            dodge_roll = jax.random.randint(k_dodge, (), 0, 20, dtype=jnp.int32)
            dodged = is_affected & (dodge_roll > jnp.int32(12))
            # --- e_jumps: rn2(5) == 0 (~20% pass) -------------------------
            # Only roll for monsters that didn't dodge — vendor runs jump
            # after e_missed in do_entity (dbridge.c lines 579-633).
            jump_roll = jax.random.randint(k_jump, (), 0, 5, dtype=jnp.int32)
            jumps = is_affected & (~dodged) & (jump_roll == jnp.int32(0))
            # Scan 8 neighbours; pick the first walkable one.
            def _find_target(carry2, off):
                tgt_r, tgt_c, found = carry2
                cand_r = m_r + off[0]
                cand_c = m_c + off[1]
                ok = _tile_walkable(cand_r, cand_c)
                pick = (~found) & ok
                new_tgt_r = jnp.where(pick, cand_r, tgt_r)
                new_tgt_c = jnp.where(pick, cand_c, tgt_c)
                new_found = found | ok
                return (new_tgt_r, new_tgt_c, new_found), None
            (tgt_r, tgt_c, found), _ = jax.lax.scan(
                _find_target,
                (m_r, m_c, jnp.bool_(False)),
                _OFFS,
            )
            relocates = jumps & found
            out_r = jnp.where(relocates, tgt_r, m_r).astype(jnp.int16)
            out_c = jnp.where(relocates, tgt_c, m_c).astype(jnp.int16)
            # Survives if dodged (in place) or jumped clear (relocated).
            survives = dodged | relocates
            return None, (out_r, out_c, survives)

        _, (out_rs, out_cs, survives) = jax.lax.scan(
            _per_monster,
            None,
            (keys_dodge, keys_jump, mr, mc, affected),
        )
        crushed = affected & (~survives)
        new_pos = jnp.stack([out_rs, out_cs], axis=-1)

    new_alive = jnp.where(crushed, jnp.bool_(False), mai.alive)
    new_hp = jnp.where(crushed, jnp.int32(0), mai.hp)
    new_mai = mai.replace(alive=new_alive, hp=new_hp, pos=new_pos)
    state = state.replace(monster_ai=new_mai)

    # --- Hero-side crush (dbridge.c::do_entity hero branch + e_died) ------
    # If the hero is on the bridge tile or the paired-wall set, take crush
    # damage ``2d4 + ulevel/2``.  Passes_walls intrinsic / phasing bypasses
    # the damage (vendor automiss — dbridge.c:486-490).  Lethal damage sets
    # scoring.death_cause = CRUSHING and done = True.
    #
    # Note: vendor also runs the e_jumped / e_missed rolls for the hero
    # (rnd(20) + dx > 18 / rnd(10) jump), but those require interactive
    # jump input — left as a deferred item.  In the JAX port the hero
    # always eats the crush damage when standing on the affected tiles
    # unless Passes_walls.
    #
    # Cite: vendor/nethack/src/dbridge.c::do_entity (hero branch);
    #       vendor/nethack/src/dbridge.c::e_died lines 405-435.
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
    from Nethax.nethax.subsystems.scoring import DeathCause

    p_r = state.player_pos[0].astype(jnp.int32)
    p_c = state.player_pos[1].astype(jnp.int32)
    hero_on_bridge = (p_r == row) & (p_c == col)
    hero_on_n = (p_r == row - jnp.int32(1)) & (p_c == col)
    hero_on_s = (p_r == row + jnp.int32(1)) & (p_c == col)
    hero_on_w = (p_r == row) & (p_c == col - jnp.int32(1))
    hero_on_e = (p_r == row) & (p_c == col + jnp.int32(1))
    hero_affected = hero_on_bridge | hero_on_n | hero_on_s | hero_on_w | hero_on_e

    passes_walls = (
        state.status.intrinsics[int(_Intr.PASSES_WALLS)]
        | (state.status.timed_intrinsics[int(_Intr.PASSES_WALLS)] > jnp.int32(0))
    )
    hero_crushed = hero_affected & (~passes_walls)

    # Damage roll: 2d4 + ulevel/2.  Deterministic 0 when no rng available
    # (legacy callers don't roll); when rng is provided, fold a fresh
    # subkey for the hero damage so it's independent of monster rolls.
    if rng is None:
        # Conservative: with no rng, apply mean damage (5) deterministically
        # to keep the hero-crush hazard meaningful even on legacy callers.
        # (2d4 mean = 5; integer.)
        dmg_dice = jnp.int32(5)
    else:
        rng_hero = jax.random.fold_in(rng, jnp.int32(0x0DB1D5E0))  # arbitrary tag
        d1 = jax.random.randint(rng_hero, (), 1, 5, dtype=jnp.int32)
        rng_hero2 = jax.random.fold_in(rng_hero, jnp.int32(1))
        d2 = jax.random.randint(rng_hero2, (), 1, 5, dtype=jnp.int32)
        dmg_dice = d1 + d2

    ulevel = state.player_xl.astype(jnp.int32)
    dmg = (dmg_dice + ulevel // jnp.int32(2)).astype(jnp.int32)
    dmg = jnp.where(hero_crushed, dmg, jnp.int32(0))

    new_hp_hero = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    lethal = hero_crushed & (new_hp_hero == jnp.int32(0))

    new_done = state.done | lethal
    new_cause = jnp.where(
        lethal,
        jnp.int8(int(DeathCause.CRUSHING)),
        state.scoring.death_cause,
    )
    new_scoring = state.scoring.replace(death_cause=new_cause)
    return state.replace(
        player_hp=new_hp_hero,
        done=new_done,
        scoring=new_scoring,
    )


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
    rolls rnl(7) (vendor luck-biased rnd, rnd.c:111-151); on a roll of 0
    the secret door is revealed (SECRET → CLOSED), mirroring
    cvt_sdoor_to_door().

    rnl(7) for x <= 15 (vendor rnd.c:124-148):
        adjustment = (|Luck|+1)/3 * sgn(Luck)         (in [-4, +4])
        i = rn2(7)
        if adjustment != 0 AND rn2(37 + |adjustment|) != 0:
            i = clip(i - adjustment, 0, 6)

    JIT-safe: no Python control flow on traced values; uses a static 3x3 sweep.
    """
    flat_lv = _flat_lv_from_state(state)
    prow = state.player_pos[0].astype(jnp.int32)
    pcol = state.player_pos[1].astype(jnp.int32)
    H = state.features.door_state.shape[1]
    W = state.features.door_state.shape[2]

    luck = state.player_luck.astype(jnp.int32)
    abs_luck = jnp.abs(luck)
    sgn_luck = jnp.sign(luck)
    adjustment = ((abs_luck + jnp.int32(1)) // jnp.int32(3)) * sgn_luck

    door_state = state.features.door_state
    rngs = jax.random.split(rng, 9 * 2)  # 1 roll + 1 luck-gate per cell
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
            i = jax.random.randint(rngs[idx * 2], (), 0, 7, dtype=jnp.int32)
            # Luck gate: rn2(37 + |adjustment|) != 0 → apply adjustment.
            luck_mod_bound = jnp.int32(37) + jnp.abs(adjustment)
            luck_roll = jax.random.randint(
                rngs[idx * 2 + 1], (), 0, luck_mod_bound, dtype=jnp.int32,
            )
            apply = (adjustment != jnp.int32(0)) & (luck_roll != jnp.int32(0))
            i_adj = jnp.clip(i - adjustment, jnp.int32(0), jnp.int32(6))
            roll = jnp.where(apply, i_adj, i)
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
# Per-tile lighting (vendor rm.h lit / waslit, light.c::set_lit, read.c::litroom)
# ---------------------------------------------------------------------------

def litroom_at(
    features: FeaturesState,
    flat_lv,
    row,
    col,
    radius: int = 0,
) -> FeaturesState:
    """Set ``lit=True`` on a tile and its Chebyshev neighbours up to ``radius``.

    Mirrors vendor/nethack/src/read.c::litroom (line 2491) +
    set_lit (line 2471) — the latter sets ``levl[x][y].lit = 1`` for a
    single coordinate; the former walks a ``do_clear_area`` Chebyshev
    radius around the hero and calls set_lit() on each tile.  Radius=0
    matches the Sunsword artifact-invoke path (vendor artifact.c:2063
    + read.c:2599 — direct ``set_lit(u.ux, u.uy)`` call) which lights
    just the hero's own tile.  Radius=5/9 matches the scroll-of-light
    blessed/unblessed paths (vendor read.c:2601).

    JIT-pure: uses ``lax.dynamic_update_slice``-equivalent ``.at[].set``
    over a fixed-size square window; out-of-bounds neighbours are
    clipped via ``jnp.where`` so the update is safe at the map edges.

    Args:
        features: FeaturesState pytree.
        flat_lv:  flattened level index (int / scalar).
        row, col: target tile (int / scalar).
        radius:   Chebyshev radius (Python int; must be compile-time
                  constant so the unrolled square is fixed-size).
                  ``radius=0`` lights only the (row, col) tile.

    Returns:
        Updated FeaturesState with ``lit`` set to True for each tile
        within radius of (row, col) on level ``flat_lv``.
    """
    lit = features.lit
    H = lit.shape[1]
    W = lit.shape[2]
    flv = jnp.asarray(flat_lv, dtype=jnp.int32)
    r0  = jnp.asarray(row,     dtype=jnp.int32)
    c0  = jnp.asarray(col,     dtype=jnp.int32)
    r_i = int(radius)
    for dr in range(-r_i, r_i + 1):
        for dc in range(-r_i, r_i + 1):
            rr = r0 + jnp.int32(dr)
            cc = c0 + jnp.int32(dc)
            rr_s = jnp.clip(rr, 0, H - 1)
            cc_s = jnp.clip(cc, 0, W - 1)
            in_bounds = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            cur = lit[flv, rr_s, cc_s]
            new_val = jnp.where(in_bounds, jnp.bool_(True), cur)
            lit = lit.at[flv, rr_s, cc_s].set(new_val)
    return features.replace(lit=lit)


def mark_remembered(
    features: FeaturesState,
    flat_lv,
    row,
    col,
) -> FeaturesState:
    """Set ``rememberedlit=True`` at (flat_lv, row, col).

    Called when the player perceives a lit tile so that future redraws
    treat it as remembered-lit even after the actual ``lit`` bit is
    cleared (e.g. light source moves away).  Companion to vendor
    ``waslit`` (vendor/nethack/include/rm.h line 166); mirrors the
    display-memory glyph path that records previously-lit tiles for
    between-visit redraw.

    Args:
        features: FeaturesState pytree.
        flat_lv:  Flattened level index (int / scalar).
        row, col: Target tile (int / scalar).

    Returns:
        Updated FeaturesState with ``rememberedlit`` set True at the tile.
    """
    flv = jnp.asarray(flat_lv, dtype=jnp.int32)
    r   = jnp.asarray(row,     dtype=jnp.int32)
    c   = jnp.asarray(col,     dtype=jnp.int32)
    new_rl = features.rememberedlit.at[flv, r, c].set(jnp.bool_(True))
    return features.replace(rememberedlit=new_rl)


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
    """Open the door at the player's tile — vendor/nethack/src/lock.c::doopen.

    Direction prompting (vendor get_adjacent_loc at lock.c:804) is upstream;
    callers are expected to have moved the player onto the target tile.
    Trapped doors are handled via open_door — vendor lock.c:907-913 sets the
    doormask to D_NODOOR after b_trapped() and our open_door() now mirrors
    this (GONE, not BROKEN); trap damage is applied to player_hp here.
    See open_door() above for the exact state transition and damage roll.

    Vendor resist roll (vendor/nethack/src/lock.c:904):
        ``if (rnl(20) < (ACURRSTR + ACURR(A_DEX) + ACURR(A_CON)) / 3) ...``
    On success the door opens (CLOSED → OPEN, or GONE if trapped per
    lock.c:909); otherwise it sticks shut.  We model ``rnl(20)`` as the plain
    uniform ``rn2(Str+Dex+Con)`` form documented in older NetHack
    sources (see task spec: ``rn2(ACURR(A_STR)+ACURR(A_DEX)+ACURR(A_CON))
    < 30``) which is byte-equal to the 3.7 source recompiled with
    Luck = 0 (the dominant case): ``rnl(20) == rn2(20)`` then, and
    ``rn2(20) < (S+D+C)/3`` is equivalent to ``rn2(S+D+C) < 30`` for
    integer ``S+D+C``.

    Trap damage now reflects vendor b_trapped (trap.c:6697):
    ``rnd(5 + (lvl < 5 ? lvl : 2 + lvl/2))`` via open_door's new
    ``level_difficulty`` parameter, sourced from
    ``state.dungeon.current_level``.

    Returns new EnvState.
    """
    flat_lv = _flat_lv_from_state(state)
    pos = jnp.array([flat_lv, state.player_pos[0], state.player_pos[1]], dtype=jnp.int32)

    # Resist roll — vendor lock.c:904.
    str_i = state.player_str.astype(jnp.int32)
    dex_i = state.player_dex.astype(jnp.int32)
    con_i = state.player_con.astype(jnp.int32)
    sdc   = jnp.maximum(str_i + dex_i + con_i, jnp.int32(1))  # rn2 lower-bound guard
    rng_resist, rng_trap = jax.random.split(rng, 2)
    resist_roll = jax.random.randint(
        rng_resist, (), minval=0, maxval=sdc, dtype=jnp.int32,
    )
    opens = resist_roll < jnp.int32(30)

    # Trap-damage roll uses level_difficulty (vendor trap.c:6697).
    lvl_diff = state.dungeon.current_level.astype(jnp.int32)

    new_features_opened, trap_dmg = open_door(
        state.features, pos, rng_trap, level_difficulty=lvl_diff,
    )
    # On resist failure the door does not transition and no damage applies.
    new_features = jax.lax.cond(
        opens,
        lambda _: new_features_opened,
        lambda _: state.features,
        operand=None,
    )
    applied_dmg = jnp.where(opens, trap_dmg, jnp.int32(0))
    new_hp = jnp.maximum(jnp.int32(0), state.player_hp - applied_dmg)
    return state.replace(features=new_features, player_hp=new_hp)


def handle_close(state, rng: jax.Array):
    """Close the door at the player's tile — vendor/nethack/src/lock.c::doclose.

    Vendor lock.c::doclose at line 957 calls obstructed(x, y) at lock.c:925
    and bails out when a monster, object, or boulder occupies the door
    tile.  Here we compute ``blocked`` = any alive monster shares the door
    tile, threaded into close_door so the OPEN→CLOSED transition is
    suppressed.  Boulder / object obstruction is upstream of FeaturesState
    and intentionally not modelled at this layer.

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
# Per-turn step
# ---------------------------------------------------------------------------
# NB: the real EnvState-level ``quaff_fountain`` lives further down in this
# file (Wave 4 Phase 2 — vendor/nethack/src/fountain.c::drinkfountain).
# A previous Wave-3 ``quaff_fountain`` FeaturesState stub used to live here
# and was silently shadowed by the redefinition; it has been removed.


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

    # Wave 30 — mark altar_known when player interacts with the altar.
    # Vendor: pray.c::doaltar (~1900) — altars become known when stepped on.
    on_any_altar_tile = altar_align >= jnp.int32(0)
    new_altar_known = jnp.where(
        on_any_altar_tile,
        state.features.altar_known.at[flat_lv, row, col].set(jnp.bool_(True)),
        state.features.altar_known,
    )
    new_features = state.features.replace(altar_known=new_altar_known)
    return state.replace(inventory=new_inv, features=new_features)


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

    # Vendor sit.c/pray.c::altar_wrath emits a flavor pline message for the
    # branch that triggered.  Mirror with MessageId emits gated on the two
    # outcome gates above.
    from Nethax.nethax.subsystems.messages import emit as _msg_emit_a, MessageId as _MsgId_a
    msgs_after_wrath = jax.lax.cond(
        do_wrath_same,
        lambda m: _msg_emit_a(m, int(_MsgId_a.ALTAR_WRATH)),
        lambda m: m,
        state.messages,
    )
    msgs_after_luck = jax.lax.cond(
        luck_active,
        lambda m: _msg_emit_a(m, int(_MsgId_a.ALTAR_LUCK_LOSS)),
        lambda m: m,
        msgs_after_wrath,
    )

    new_state = state.replace(
        player_wis=new_wis,
        player_luck=new_luck,
        prayer=new_prayer,
        messages=msgs_after_luck,
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

    # vendor/nethack/src/fountain.c:404 — obj->otyp == LONG_SWORD.
    # Long sword type_id matches Nethax/nethax/constants/objects.py line 924
    # (index 37 in OBJECTS).  Previous value 34 was incorrect.
    LONG_SWORD_TYPE_ID = 37   # ObjType.LONG_SWORD (objects.py line 924)
    LAWFUL = 2                # Alignment.LAWFUL

    # vendor/nethack/src/role.c — PM_KNIGHT role id.
    from Nethax.nethax.constants.roles import Role as _Role
    KNIGHT = int(_Role.KNIGHT)

    inv = state.inventory.items
    n = inv.category.shape[0]
    slot = jnp.clip(slot_idx.astype(jnp.int32), 0, n - 1)

    item_type   = inv.type_id[slot]
    item_buc    = inv.buc_status[slot]
    item_ench   = inv.enchantment[slot]
    item_qty    = inv.quantity[slot]
    item_filled = inv.category[slot] != jnp.int8(0)

    # vendor/nethack/src/fountain.c:404-408 — Excalibur preconditions:
    #   obj->otyp == LONG_SWORD && u.ulevel >= 5
    #   && !rn2(Role_if(PM_KNIGHT) ? 6 : 30)
    #   && obj->quan == 1L && !obj->oartifact
    #   && (alignment check at line 411 — non-Lawful denied; we gate on
    #      Lawful here so the grant path matches vendor's success branch).
    is_long_sword  = (item_type == jnp.int16(LONG_SWORD_TYPE_ID)) & item_filled
    is_lawful      = state.player_align == jnp.int8(LAWFUL)
    is_high_xl     = state.player_xl >= jnp.int32(5)
    is_single      = item_qty == jnp.int16(1)              # obj->quan == 1L
    is_knight      = state.player_role == jnp.int8(KNIGHT)
    excal_eligible = is_long_sword & is_lawful & is_high_xl & is_single
    # vendor fountain.c:405 — !rn2(Role_if(PM_KNIGHT) ? 6 : 30).
    # Knight: 1/6; any other (lawful) class: 1/30.
    excal_denom = jnp.where(is_knight, jnp.int32(6), jnp.int32(30))
    excal_roll  = jax.random.randint(
        rng_excal, (), 0, excal_denom, dtype=jnp.int32
    ) == jnp.int32(0)
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
        # vendor/nethack/src/fountain.c:63-89 dowaterdemon() — spawns
        # PM_WATER_DEMON adjacent.  Modelled here as a fixed HP loss proxy;
        # 4 HP matches the sibling fountain.py::dip_fountain _demon branch
        # (fountain.py line 267) and aligns with the cross-subsystem audit.
        return s.replace(player_hp=jnp.maximum(jnp.int32(0), s.player_hp - jnp.int32(4)))

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

    # Emit per-effect flavor message — mirrors vendor/nethack/src/sit.c
    # throne_sit_effect pline lines (one per case 1..13).
    from Nethax.nethax.subsystems.messages import emit as _msg_emit_t, MessageId as _MsgId_t
    _THRONE_MSG_IDS = jnp.array([
        int(_MsgId_t.THRONE_ATTR_LOSS),
        int(_MsgId_t.THRONE_ATTR_GAIN),
        int(_MsgId_t.THRONE_SHOCK),
        int(_MsgId_t.THRONE_FULL_HEAL),
        int(_MsgId_t.THRONE_TAKE_GOLD),
        int(_MsgId_t.THRONE_WISH),
        int(_MsgId_t.THRONE_COURT),
        int(_MsgId_t.THRONE_GENOCIDE),
        int(_MsgId_t.THRONE_CURSE_ITEMS),
        int(_MsgId_t.THRONE_MAP_CONFUSE),
        int(_MsgId_t.THRONE_TELEPORT),
        int(_MsgId_t.THRONE_IDENTIFY),
        int(_MsgId_t.THRONE_CONFUSE),
    ], dtype=jnp.int32)
    safe_effect = jnp.clip(effect, jnp.int32(0), jnp.int32(_THRONE_MSG_IDS.shape[0] - 1))
    throne_msg_id = _THRONE_MSG_IDS[safe_effect]
    new_state = new_state.replace(
        messages=_msg_emit_t(new_state.messages, throne_msg_id),
    )

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
        case 19/tail → BLACK_PUDDING (severe HP loss)

    Case 19 (vendor fountain.c:700-710) is a Hallucination flavor pline
    that FALLTHROUGHs to ``default`` (cold/warm/hot sip — no HP effect).
    Vendor's BLACK_PUDDING summon lives in dokick.c::kick_nondoor, not
    drinksink().  We map fate==19 to the default cold-water sip to
    preserve byte-equal vendor parity.
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
        _SINK_COLD_WATER,        # 19 (vendor case 19 falls through to default)
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
    )
    new_state = jax.lax.switch(bucket, branches, state)
    return new_state


# ---------------------------------------------------------------------------
# Sit-on-sink effects (sit.c::dosit IS_SINK branch, rn2(6) table)
# ---------------------------------------------------------------------------
def sit_sink(state, rng):
    """Sit on a sink — vendor/nethack/src/sit.c::dosit IS_SINK branch.

    Vendor truth (sit.c:526-529):

        } else if (IS_SINK(typ)) {
            You(sit_message, defsyms[S_sink].explanation);
            Your("%s gets wet.",
                 humanoid(gy.youmonst.data) ? "rump" : "underside");

    Pure flavor: no rn2() roll, no state mutation.  Byte-equal vendor
    parity means this is a no-op; *rng* is accepted for caller-API
    stability but unused.
    """
    del rng
    return state


# ---------------------------------------------------------------------------
# Kick-sink effects (dokick.c::kick_nondoor IS_SINK branch, rn2(4) table)
# ---------------------------------------------------------------------------
def kick_sink(state, rng):
    """Kick a sink — vendor/nethack/src/dokick.c::kick_nondoor IS_SINK branch
    (lines 1171-1247).

    Vendor cascade (nested rn2 rolls, NOT a flat table):

        if (rn2(5)) {                              # 4/5 — klunk, no-op
            pline("Klunk!  ..."); return 1;
        } else if (!(looted & S_LPUDDING) && !rn2(3)
                   && !(G_GONE)) {                 # 1/5 * 1/3 ≈ 6.7% — black pudding
            makemon(PM_BLACK_PUDDING, ...);
            return 1;
        } else if (!(looted & S_LDWASHER) && !rn2(3)
                   && !(G_GONE)) {                 # remainder * 1/3 ≈ 4.4% — dishwasher
            makemon(PM_INCUBUS/PM_SUCCUBUS, ...);
            return 1;
        } else if (!rn2(3)) {                      # remainder * 1/3 ≈ 3%  — sink_backs_up
            pline("Muddy waste pops up ...");
            mkobj_at(RING_CLASS, ...);
            return 1;
        }
        goto ouch;                                 # remainder ≈ 6% — kick_ouch HP damage

    We replicate the cascade with four sequential ``rn2`` draws (matching
    vendor's rn2(5)/rn2(3)/rn2(3)/rn2(3) sequence) and gate each branch on
    the prior misses.  Only the ``ouch`` branch deals HP damage in this
    subsystem (vendor: ``losehp(Maybe_Half_Phys(rnd(ACURR(A_CON) > 15 ? 3 : 5)),
    kickstr, KILLED_BY)`` at dokick.c:1243-1244).

    Outcome IDs (returned as int32, kept in [0, 3] for the existing test
    contract — the dishwasher branch maps to 3 / klunk alongside the
    no-effect outcomes):
        0 → kick_ouch       (HP damage, rnd(5))
        1 → black pudding spawn (no HP mutation here)
        2 → sink_backs_up   (ring drop, no HP mutation)
        3 → klunk / dishwasher / fallthrough  (no HP mutation)

    Cite: vendor/nethack/src/dokick.c::kick_nondoor lines 1171-1247.
    """
    rng_r5, rng_r3a, rng_r3b, rng_r3c, rng_dmg = jax.random.split(rng, 5)

    # First draw: rn2(5).  Vendor ``if (rn2(5))`` — truthy 4/5 → klunk.
    r5 = jax.random.randint(rng_r5, (), 0, 5, dtype=jnp.int32)
    is_klunk = r5 != jnp.int32(0)

    # Second draw: rn2(3) gates the BLACK_PUDDING branch.
    r3a = jax.random.randint(rng_r3a, (), 0, 3, dtype=jnp.int32)
    is_pudding = (~is_klunk) & (r3a == jnp.int32(0))

    # Third draw: rn2(3) gates the dishwasher branch.
    r3b = jax.random.randint(rng_r3b, (), 0, 3, dtype=jnp.int32)
    is_dwasher = (~is_klunk) & (~is_pudding) & (r3b == jnp.int32(0))

    # Fourth draw: rn2(3) gates sink_backs_up (ring drop).
    r3c = jax.random.randint(rng_r3c, (), 0, 3, dtype=jnp.int32)
    is_backs_up = (
        (~is_klunk) & (~is_pudding) & (~is_dwasher) & (r3c == jnp.int32(0))
    )

    # Anything that falls past all four branches lands on `goto ouch`.
    is_ouch = (
        (~is_klunk) & (~is_pudding) & (~is_dwasher) & (~is_backs_up)
    )

    # Vendor dokick.c:1243 — rnd(ACURR(A_CON) > 15 ? 3 : 5).  No CON wired
    # here yet, so use the CON<=15 path (rnd(5) = 1..5).
    ouch_dmg = jax.random.randint(rng_dmg, (), 1, 6, dtype=jnp.int32)
    dmg = jnp.where(is_ouch, ouch_dmg, jnp.int32(0))
    new_hp = jnp.maximum(jnp.int32(0), state.player_hp - dmg)

    # outcome_id in [0, 3] (test contract).
    outcome = jnp.where(is_ouch,     jnp.int32(0),
              jnp.where(is_pudding,  jnp.int32(1),
              jnp.where(is_backs_up, jnp.int32(2),
                                      jnp.int32(3))))  # klunk + dishwasher
    return state.replace(player_hp=new_hp), outcome


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
    # Coaligned: same alignment.  Opposite (cross): max-distance alignment
    # mismatch (|diff|==2 with our 0=chaotic, 1=neutral, 2=lawful encoding).
    # Neutral-altar drops (|diff|==1) leave BUC unchanged per
    # vendor/nethack/src/pray.c::doaltar — only flash colours bless/curse,
    # which corresponds to a coaligned or opposite-aligned altar match.
    align_diff = jnp.abs(altar_align - player_align)
    coaligned = on_altar & (align_diff == jnp.int32(0))
    cross_aligned = on_altar & (align_diff == jnp.int32(2))

    safe_slot = jnp.clip(slot_idx.astype(jnp.int32), 0,
                         state.inventory.items.buc_status.shape[0] - 1)
    old_buc = state.inventory.items.buc_status[safe_slot].astype(jnp.int32)

    # Already-blessed items keep their blessing on cross-aligned altars
    # (vendor pickup.c::drop_on_altar — only uncursed→curse, blessed→bless).
    is_blessed = old_buc == jnp.int32(3)
    is_cursed  = old_buc == jnp.int32(1)
    bless_now  = coaligned & ~is_blessed
    curse_now  = cross_aligned & ~is_cursed & ~is_blessed
    new_buc = jnp.where(bless_now, jnp.int32(3),
              jnp.where(curse_now, jnp.int32(1),
              old_buc))

    new_buc_arr = state.inventory.items.buc_status.at[safe_slot].set(
        new_buc.astype(jnp.int8))
    new_items = state.inventory.items.replace(buc_status=new_buc_arr)
    return state.replace(inventory=state.inventory.replace(items=new_items))
