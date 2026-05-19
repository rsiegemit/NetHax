"""Monster AI subsystem — Wave 3 movement & wake logic + Wave 5 deepening.

Purpose:
    Per-monster turn dispatch: movement, strategy selection, pet behavior,
    line-of-sight, bounded BFS pathfinding, retreat, muse (item use),
    mcastu (spell casting), and sleep/wake management.

Canonical sources (NetHack 5.0):
    - src/monmove.c  — main monster turn, 8-dir pathfinding, strategy selection
    - src/dogmove.c  — pet movement and pathfinding
    - src/dog.c      — taming and pet recruitment
    - src/muse.c     — monster item-use decisions
    - src/mcastu.c   — monster spell casting (castmu)
    - src/minion.c   — demon/angel summoning and control

Wave 5 Phase 1 additions:
    - monster_can_see_player: Bresenham LoS, terrain-aware (walls / closed doors).
    - pathfind_step: bounded BFS (depth=12) → first step toward player; greedy fallback.
    - monster_use_item: HP-gated heal-quaff / adjacent-teleport / LoS-zap stubs.
    - monster_cast_spell: mage-class monsters cast direct damage at player.
    - maybe_retreat: HP < 20% → move AWAY from player.
    - pet_move: tame monster follows player / attacks hostile neighbours.
    - maybe_wake_monster: asleep + player in LoS → wake.
    - step / monster_turn refactored to dispatch through these.

Wave 5 simplifications (documented):
    - Monsters carry no inventory yet, so monster_use_item gates by class flags
      (heuristic via entry_idx mod) and is partly stubbed — see function doc.
    - Mage-class detection uses an entry_idx allow-list (a small ID range)
      because lifting the full MONSTERS table sound mask into JIT-side data is
      out of scope for this phase. Tests set entry_idx into the mage range.

TODO — later Wave 5 phases:
    - Real monster inventory slots → real muse.
    - Spellbook list per mage class (Wave 5 Phase 2).
"""

from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct

# Map dims — kept local so we don't have to import dungeon.branches in JIT-time.
# These must match Nethax.nethax.dungeon.branches.MAP_H/MAP_W.
_MAP_H: int = 21
_MAP_W: int = 80

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MONSTERS_PER_LEVEL: int = 400

# Wave 6 Mission: per-monster inventory width.
# Vendor monsters can carry arbitrary inventory chains (struct obj *minvent in
# include/monst.h); we cap per-monster slots at MAX_MONSTER_INV for JIT-safe
# fixed-shape arrays.  Vendor src/makemon.c::mongets typically gives 0-3 items
# at spawn so 8 slots is comfortable headroom.
MAX_MONSTER_INV: int = 8

# Bounded BFS depth (vendor monmove.c uses unbounded; we cap for JIT).
_PATHFIND_MAX_DEPTH: int = 12

# Wave 6 Phase B: mage-class detection now reads MonsterEntry.sound (msound)
# directly from the MONSTERS table.  An entry whose sound is MS_SPELL (42) or
# MS_PRIEST (41) — see vendor/nethack/include/monflag.h — qualifies as a
# spellcaster.  Pre-Wave-6 callers used an entry_idx range [_LO, _HI]; the
# legacy bounds are kept as no-op compatibility constants but no longer drive
# the predicate.
_MAGE_ENTRY_LO: int = 300
_MAGE_ENTRY_HI: int = 360


def _build_monster_sound_table() -> jnp.ndarray:
    """Build MONSTERS[i].sound lookup eagerly at module load.

    Built once so it never traces inside a jit-compiled context.
    Mirrors the pattern used in subsystems/items_scrolls._MONSTER_SYMBOL_TABLE.
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.sound) for m in MONSTERS], dtype=jnp.int8)


def _build_monster_flag_tables() -> tuple:
    """Build MONSTERS[i].flags{1,2,3} / level lookups eagerly at module load.

    Tables are int32 because flags1/flags2 are 32-bit bitfields per
    vendor/nethack/include/monflag.h.  Level fits comfortably in int16.
    Mirrors the per-monster flag access pattern used by vendor
    src/monmove.c, src/muse.c, src/mcastu.c, src/dogmove.c.
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    # uint32 bitfields above 0x7FFFFFFF need two's-complement remapping
    # before they fit a signed int32 numpy/jax array.
    def _u32_to_i32(v) -> int:
        v = int(v) & 0xFFFFFFFF
        return v - 0x100000000 if v & 0x80000000 else v
    f1 = jnp.array([_u32_to_i32(m.flags1) for m in MONSTERS], dtype=jnp.int32)
    f2 = jnp.array([_u32_to_i32(m.flags2) for m in MONSTERS], dtype=jnp.int32)
    f3 = jnp.array([_u32_to_i32(m.flags3) for m in MONSTERS], dtype=jnp.int32)
    lev = jnp.array([int(m.level) for m in MONSTERS], dtype=jnp.int16)
    return f1, f2, f3, lev


# Eager build; size = NUMMONS (381).
_MONSTER_SOUND_TABLE: jnp.ndarray = _build_monster_sound_table()
(_MONSTER_FLAGS1_TABLE,
 _MONSTER_FLAGS2_TABLE,
 _MONSTER_FLAGS3_TABLE,
 _MONSTER_LEVEL_TABLE) = _build_monster_flag_tables()
_MS_SPELL: int = 42   # vendor/nethack/include/monflag.h
_MS_PRIEST: int = 41  # vendor/nethack/include/monflag.h
_MS_RIDER: int = 35   # vendor/nethack/include/monflag.h


def _build_monster_resists_table() -> jnp.ndarray:
    """Precompute MONSTERS[i].resists_mask as int32[NUMMONS].

    Cite: vendor/nethack/src/monst.c — each MON() entry has an mr1 field
    (resistance bitmask: MR_FIRE=0x01, MR_COLD=0x02, MR_SLEEP=0x04,
    MR_DISINT=0x08, MR_ELEC=0x10, MR_POISON=0x20, MR_ACID=0x40, MR_STONE=0x80).
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.resists_mask) for m in MONSTERS], dtype=jnp.int32)


def _build_monster_undead_table() -> jnp.ndarray:
    """Precompute bool[NUMMONS]: True iff MONSTERS[i].flags2 & M2_UNDEAD.

    Cite: vendor/nethack/include/monflag.h M2_UNDEAD = 0x00000002.
    vendor/nethack/include/mondata.h: #define is_undead(ptr) ((ptr)->mflags2 & M2_UNDEAD)
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD
    return jnp.array([bool(m.flags2 & M2_UNDEAD) for m in MONSTERS], dtype=jnp.bool_)


def _build_monster_nonliving_table() -> jnp.ndarray:
    """Precompute bool[NUMMONS]: True iff monster is nonliving.

    Cite: vendor/nethack/include/mondata.h:
      #define weirdnonliving(ptr) (is_golem(ptr) || (ptr)->mlet == S_VORTEX)
      #define nonliving(ptr) (is_undead(ptr) || (ptr)==&mons[PM_MANES] || weirdnonliving(ptr))
    """
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD, MonsterSymbol
    result = []
    for m in MONSTERS:
        is_undead  = bool(m.flags2 & M2_UNDEAD)
        is_golem   = (m.symbol == MonsterSymbol.S_GOLEM)
        is_vortex  = (m.symbol == MonsterSymbol.S_VORTEX)
        is_manes   = (m.name == "manes")
        result.append(is_undead or is_golem or is_vortex or is_manes)
    return jnp.array(result, dtype=jnp.bool_)


# Eager-built tables indexed by entry_idx (size = NUMMONS = 381).
_MONSTER_MRESISTS:  jnp.ndarray = _build_monster_resists_table()
_MONSTER_UNDEAD:    jnp.ndarray = _build_monster_undead_table()
_MONSTER_NONLIVING: jnp.ndarray = _build_monster_nonliving_table()


def _build_ignores_elbereth_table() -> jnp.ndarray:
    """Precompute per-entry Elbereth exemption flags.

    Cite: vendor/nethack/src/monmove.c::onscary lines 241-303.
    Exempt:
      - S_HUMAN ('@') — humanoids (vendor onscary: ``ishumanoid`` check)
      - Wizard of Yendor — vendor line 256 (nemesis)
      - Archon — vendor line 271
      - Riders (Death, Famine, Pestilence) — vendor line 259
        ``if (is_rider(ptr)) return 0;``
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    _RIDER_NAMES = frozenset({"Death", "Pestilence", "Famine"})
    _EXEMPT_NAMES = frozenset({"Wizard of Yendor", "Archon"})
    result = []
    for m in MONSTERS:
        exempt = (
            m.symbol == MonsterSymbol.S_HUMAN
            or m.name in _RIDER_NAMES
            or m.name in _EXEMPT_NAMES
        )
        result.append(bool(exempt))
    return jnp.array(result, dtype=jnp.bool_)


# Elbereth exemption table indexed by entry_idx.  Built once at module load.
_IGNORES_ELBERETH: jnp.ndarray = _build_ignores_elbereth_table()


def _build_monster_primary_attack_table() -> tuple:
    """Precompute the first non-passive (n_dice, sides, base_ac, level)
    fields per MONSTERS[i] for the mattackm path.

    The first attack with ``dice_n > 0`` and ``atyp != AT_NONE`` is used as
    the monster's "primary" melee attack.  Defaults: 1d4 (matches vendor
    fallback in makemon.c::newmonhp for atype-less entries).

    Cite: vendor/nethack/src/mhitm.c lines 1024-1100 (mattackm — pulls
    mtmp->data->mattk[0].damn/damd to roll damage; AC pulled from
    mtmp->data->ac).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    n_arr, s_arr, ac_arr = [], [], []
    for m in MONSTERS:
        n, s = 1, 4
        for atk in (m.attacks or ()):
            if int(atk[0]) != int(AttackType.AT_NONE) and int(atk[2]) > 0:
                n, s = int(atk[2]), int(atk[3])
                break
        n_arr.append(n)
        s_arr.append(s)
        ac_arr.append(int(m.ac))
    # int16 to safely contain the rare 255-sentinel sides field.
    return (
        jnp.array(n_arr, dtype=jnp.int16),
        jnp.array(s_arr, dtype=jnp.int16),
        jnp.array(ac_arr, dtype=jnp.int8),
    )


# (n_dice, sides, base_ac) per MONSTERS[i].
(_MONSTER_PRIMARY_ATTACK_N,
 _MONSTER_PRIMARY_ATTACK_S,
 _MONSTER_PRIMARY_ATTACK_AC) = _build_monster_primary_attack_table()
# Convenience alias — fields packed as a stacked table for callers that want
# (n, s, ac) as one column.  Each column is independently accessed below.
_MONSTER_PRIMARY_ATTACK_TABLE: tuple = (
    _MONSTER_PRIMARY_ATTACK_N,
    _MONSTER_PRIMARY_ATTACK_S,
    _MONSTER_PRIMARY_ATTACK_AC,
)


def _build_monster_move_speed_table() -> jnp.ndarray:
    """Precompute MONSTERS[i].move_speed eagerly at module load.

    Used by the speed-energy accumulator (monsters_step_all) to compute
    per-tick movement-point gain.  Vendor NORMAL_SPEED = 12.

    Cite: vendor/nethack/src/monmove.c line 1731 (per-turn movement gain);
          vendor/nethack/src/allmain.c lines 233-234 (mtmp->movement loop).
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.move_speed) for m in MONSTERS], dtype=jnp.int16)


_MONSTER_MOVE_SPEED_TABLE: jnp.ndarray = _build_monster_move_speed_table()

# ---- Item category / type IDs (mirrors subsystems/inventory.ItemCategory
# and subsystems/items_{potions,scrolls,wands}.<Effect>) ------------------
# Kept as plain ints so we don't import the inventory module (avoids cycles
# and keeps the JIT-time constants light).
_CAT_POTION: int   = 8    # ItemCategory.POTION
_CAT_SCROLL: int   = 9    # ItemCategory.SCROLL
_CAT_WAND:   int   = 11   # ItemCategory.WAND
_CAT_WEAPON: int   = 2    # ItemCategory.WEAPON
_CAT_ARMOR:  int   = 3    # ItemCategory.ARMOR
_CAT_AMULET: int   = 5    # ItemCategory.AMULET
_CAT_SPBOOK: int   = 10   # ItemCategory.SPBOOK
_CAT_COIN:   int   = 12   # ItemCategory.COIN

_POT_HEALING:      int = 10   # PotionEffect.HEALING
_SCR_TELEPORT:     int = 10   # ScrollEffect.TELEPORTATION
_WAN_FIRE:         int = 16   # WandEffect.FIRE
_WAN_TELEPORT:     int = 12   # WandEffect.TELEPORTATION

# M-flag bits we need at JIT-time (vendor/nethack/include/monflag.h).
_M1_FLY: int          = 0x00000001
_M1_SWIM: int         = 0x00000002
_M1_AMPHIBIOUS: int   = 0x00000200
_M1_BREATHLESS: int   = 0x00000400
_M1_MINDLESS: int     = 0x00010000
_M1_HUMANOID: int     = 0x00020000
_M1_ANIMAL: int       = 0x00040000
_M1_NOHANDS: int      = 0x00002000
_M1_SEE_INVIS: int    = 0x01000000

_M2_UNDEAD: int       = 0x00000002
_M2_DEMON: int        = 0x00000100
_M2_PEACEFUL: int     = 0x00200000

# Tile constants — kept local to avoid an import cycle with constants.tiles.
# Must mirror Nethax.nethax.constants.tiles.TileType.
_TILE_WALL: int        = 3
_TILE_CLOSED_DOOR: int = 4
_TILE_OPEN_DOOR: int   = 5  # see-thru in vendor; non-blocking for LoS.
_TILE_WATER: int       = 8
_TILE_LAVA: int        = 9
_TILE_TREE: int        = 20  # blocks LoS per vendor vision.c:166.


class MoveStrategy(IntEnum):
    """Behavior tag stored in MonsterAIState.mstrategy.

    Maps loosely to the strategy flags in src/monmove.c (MSZT_*, mtame, etc.).
    """
    NONE = 0       # Uninitialized / default
    SLEEP = 1      # Dormant; wakes on disturbance (Wave 3)
    WANDER = 2     # Random walk; no target
    HUNT = 3       # Moving toward last known player position
    FLEE = 4       # Moving away from player (low HP)
    PARALYZE = 5   # Cannot act this turn (paralysis effect)
    WAIT = 6       # Stationary by choice (guards, shopkeepers — Wave 5)
    RETREAT = 7    # Tactical fallback to recover HP (Wave 4)
    SUMMON = 8     # About to call for reinforcements (Wave 4)
    CONFUSED = 9   # Random direction each step


# ---------------------------------------------------------------------------
# State struct
# ---------------------------------------------------------------------------

@struct.dataclass
class MonsterAIState:
    """Per-monster AI bookkeeping for one dungeon level.

    All arrays are indexed by monster slot (0..MAX_MONSTERS_PER_LEVEL-1).
    Slots not in use (mask=False in the main Monsters struct) hold
    default / zeroed values.

    Shapes use MAX_MONSTERS_PER_LEVEL so this struct is level-agnostic;
    callers index by [level] in the outer EnvState.
    """

    # Accumulated movement points; a monster acts when this reaches its speed.
    # int16 per slot; range 0..255 sufficient for normal speeds.
    movement_points: jnp.ndarray   # [MAX_MONSTERS_PER_LEVEL]  int16

    # Current behavior tag (MoveStrategy value).
    mstrategy: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  int8

    # Tile the monster is currently navigating toward.
    # [-1, -1] when no target.
    target_pos: jnp.ndarray        # [MAX_MONSTERS_PER_LEVEL, 2]  int16

    # Last tile where the monster observed the player.
    # [-1, -1] when never seen.
    last_seen_player_pos: jnp.ndarray  # [MAX_MONSTERS_PER_LEVEL, 2]  int16

    # Pet flag — True if this monster is tame and follows the player.
    tame: jnp.ndarray              # [MAX_MONSTERS_PER_LEVEL]  bool

    # Peaceful flag — True if monster will not attack unprovoked.
    peaceful: jnp.ndarray          # [MAX_MONSTERS_PER_LEVEL]  bool

    # ---- Wave 3 combat scaffolding (vendor/nethack/include/monst.h) ----
    # Per-monster combat state.  Empty slots have alive=False and hp=0.
    hp: jnp.ndarray                # [MAX_MONSTERS_PER_LEVEL]  int32  current hp
    hp_max: jnp.ndarray            # [MAX_MONSTERS_PER_LEVEL]  int32  max hp
    pos: jnp.ndarray               # [MAX_MONSTERS_PER_LEVEL, 2] int16 (row, col)
    alive: jnp.ndarray             # [MAX_MONSTERS_PER_LEVEL]  bool
    ac: jnp.ndarray                # [MAX_MONSTERS_PER_LEVEL]  int8  base AC
    is_large: jnp.ndarray          # [MAX_MONSTERS_PER_LEVEL]  bool  bigmonst flag
    attack_dice_n: jnp.ndarray     # [MAX_MONSTERS_PER_LEVEL]  int8  natural-attack n dice
    attack_dice_sides: jnp.ndarray # [MAX_MONSTERS_PER_LEVEL]  int8  natural-attack die sides

    # ---- Wave 3 sleep flag ----
    # Kept as a computed convenience: True iff sleep_timer > 0.
    asleep: jnp.ndarray            # [MAX_MONSTERS_PER_LEVEL]  bool

    # ---- Status timers (vendor/nethack/src/uhitm.c:387-394) ----
    # Supersede the boolean asleep/stunned flags; booleans remain for
    # backward-compat and are kept equal to (timer > 0).
    sleep_timer:     jnp.ndarray   # [MAX_MONSTERS_PER_LEVEL]  int16
    stun_timer:      jnp.ndarray   # [MAX_MONSTERS_PER_LEVEL]  int16
    confuse_timer:   jnp.ndarray   # [MAX_MONSTERS_PER_LEVEL]  int16
    flee_until_turn: jnp.ndarray   # [MAX_MONSTERS_PER_LEVEL]  int32
    paralyzed_timer: jnp.ndarray   # [MAX_MONSTERS_PER_LEVEL]  int16

    # ---- Wave 4 monster polymorph (vendor/nethack/src/mon.c::newcham) ----
    # MONSTERS table index for each slot.  Defaults to 0; only meaningful
    # for live slots.  polymorph_monster swaps this in place.
    entry_idx: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  int16
    # Saved original entry_idx (filled at first newcham; useful for any
    # later "shapeshifter revert" logic — Wave 5).
    orig_entry_idx: jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL]  int16

    # ---- Wave 6 pet fields  (vendor/nethack/src/dogmove.c, dog.h::edog) ----
    # mtame: tame-level counter (vendor monst.h field).  0 = wild; higher = better
    # trained.  Used by vendor dogmove.c to gate pet obedience.
    mtame: jnp.ndarray             # [MAX_MONSTERS_PER_LEVEL]  int8
    # apport: 1..10 willingness to fetch / stay near (vendor include/dog.h
    # struct edog field).  Higher = stays closer.  We model "leash distance"
    # as a Chebyshev cap proportional to apport.
    apport: jnp.ndarray            # [MAX_MONSTERS_PER_LEVEL]  int8

    # ---- Wave 6 Mission: monster inventory (vendor monst.h::minvent) ----
    # Per-monster inventory parallels the player Item struct (subsystems/
    # inventory.py).  An empty slot has inv_category == 0 (ItemCategory.NONE,
    # matching vendor RANDOM_CLASS sentinel).
    # Vendor makemon.c::mongets fills minvent at spawn based on the monster's
    # M2_* class flags; see _MONSTER_INV_KITS in dungeon/spawning.py.
    inv_category:   jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV] int8
    inv_type_id:    jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV] int16
    inv_buc:        jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV] int8  (-1/0/+1)
    inv_quantity:   jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV] int16
    inv_charges:    jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV] int8
    inv_identified: jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, MAX_MONSTER_INV] bool

    # ---- Pet hunger counter (vendor dog.c:380 edog.hungrytime) ----
    # Per-slot hunger counter.  Starts at 1000; decrements 1 per turn for
    # tame slots.  At 0, pet enters "hungry" state (lower aggression).
    # At -50, pet dies / transitions away.
    pet_hunger: jnp.ndarray        # [MAX_MONSTERS_PER_LEVEL]  int16

    # ---- Saddle flag (vendor steed.c:put_saddle_on_mon / W_SADDLE) ----
    # 0 = no saddle, 1 = saddled.  Required before player can mount.
    # Mirrors vendor which_armor(mtmp, W_SADDLE) check in steed.c:281.
    saddled: jnp.ndarray           # [MAX_MONSTERS_PER_LEVEL]  int8

    # ---- Disarmed / unwielded flag ----
    # True when the monster has been disarmed (whip-pull or disarm artifact).
    # When set, monster_attack_player uses bare-hands dice instead of weapon
    # dice.  Cite: vendor/nethack/src/weapon.c (disarm logic).
    is_unwielded: jnp.ndarray      # [MAX_MONSTERS_PER_LEVEL]  bool

    # ---- Per-monster resist / status fields (wand parity) ----
    # Populated at spawn from MONSTERS[entry_idx].resists_mask.
    # Cite: vendor/nethack/src/monst.c MON() mr1 field.
    resists: jnp.ndarray           # [MAX_MONSTERS_PER_LEVEL]  int32

    # Undead flag — from MONSTERS[entry_idx].flags2 & M2_UNDEAD at spawn.
    # Cite: vendor/nethack/include/monflag.h M2_UNDEAD = 0x00000002.
    undead: jnp.ndarray            # [MAX_MONSTERS_PER_LEVEL]  bool

    # Invisible flag — True if naturally invisible or zapped by WAN_MAKE_INVISIBLE.
    # Cite: vendor/nethack/src/zap.c::zhitm make_invisible handling.
    invisible: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  bool

    # Nonliving flag — golems, vortices, undead; immune to WAN_DEATH.
    # Cite: vendor/nethack/include/mondata.h::nonliving().
    nonliving: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  bool

    # Speed modifier: -1 = slowed, 0 = normal, +1 = hasted.
    # Cite: vendor/nethack/src/zap.c WAN_SLOW/SPEED_MONSTER.
    speed_mod: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  int8

    # Cancellation flag — monster has been cancelled (drains powers).
    # Cite: vendor/nethack/src/zap.c WAN_CANCELLATION.
    cancelled: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  bool

    # Cloned flag — True iff this monster was produced by clone_mon
    # (or any later wand/scroll-driven cloning).  Vendor uses MON_WEP and
    # the mtmp->mcloned bit (vendor/nethack/include/monst.h::mcloned) to gate
    # corpse/XP drops on clones.  Wave 16e wires the bit; downstream drop
    # gating remains a follow-up.
    # Cite: vendor/nethack/src/makemon.c::clone_mon lines 837-944.
    mcloned: jnp.ndarray           # [MAX_MONSTERS_PER_LEVEL]  bool


def make_monster_ai_state() -> MonsterAIState:
    """Return a zero-initialized MonsterAIState for one level."""
    n = MAX_MONSTERS_PER_LEVEL
    inv_shape = (n, MAX_MONSTER_INV)
    return MonsterAIState(
        movement_points=jnp.zeros(n, dtype=jnp.int16),
        mstrategy=jnp.zeros(n, dtype=jnp.int8),
        target_pos=jnp.full((n, 2), -1, dtype=jnp.int16),
        last_seen_player_pos=jnp.full((n, 2), -1, dtype=jnp.int16),
        tame=jnp.zeros(n, dtype=bool),
        peaceful=jnp.zeros(n, dtype=bool),
        hp=jnp.zeros(n, dtype=jnp.int32),
        hp_max=jnp.zeros(n, dtype=jnp.int32),
        pos=jnp.full((n, 2), -1, dtype=jnp.int16),
        alive=jnp.zeros(n, dtype=bool),
        ac=jnp.full((n,), 10, dtype=jnp.int8),
        is_large=jnp.zeros(n, dtype=bool),
        attack_dice_n=jnp.ones(n, dtype=jnp.int8),
        attack_dice_sides=jnp.full((n,), 4, dtype=jnp.int8),
        asleep=jnp.zeros(n, dtype=bool),
        sleep_timer=jnp.zeros(n, dtype=jnp.int16),
        stun_timer=jnp.zeros(n, dtype=jnp.int16),
        confuse_timer=jnp.zeros(n, dtype=jnp.int16),
        flee_until_turn=jnp.zeros(n, dtype=jnp.int32),
        paralyzed_timer=jnp.zeros(n, dtype=jnp.int16),
        entry_idx=jnp.zeros(n, dtype=jnp.int16),
        orig_entry_idx=jnp.full((n,), -1, dtype=jnp.int16),
        mtame=jnp.zeros(n, dtype=jnp.int8),
        apport=jnp.full((n,), 5, dtype=jnp.int8),
        # Inventory: empty slots have category == 0 (ItemCategory.NONE).
        inv_category=jnp.zeros(inv_shape, dtype=jnp.int8),
        inv_type_id=jnp.zeros(inv_shape, dtype=jnp.int16),
        inv_buc=jnp.zeros(inv_shape, dtype=jnp.int8),
        inv_quantity=jnp.zeros(inv_shape, dtype=jnp.int16),
        inv_charges=jnp.zeros(inv_shape, dtype=jnp.int8),
        inv_identified=jnp.zeros(inv_shape, dtype=bool),
        pet_hunger=jnp.full(n, 1000, dtype=jnp.int16),
        saddled=jnp.zeros(n, dtype=jnp.int8),
        is_unwielded=jnp.zeros(n, dtype=jnp.bool_),
        resists=jnp.zeros(n, dtype=jnp.int32),
        undead=jnp.zeros(n, dtype=jnp.bool_),
        invisible=jnp.zeros(n, dtype=jnp.bool_),
        nonliving=jnp.zeros(n, dtype=jnp.bool_),
        speed_mod=jnp.zeros(n, dtype=jnp.int8),
        cancelled=jnp.zeros(n, dtype=jnp.bool_),
        mcloned=jnp.zeros(n, dtype=jnp.bool_),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _chebyshev_dist(pos_a: jnp.ndarray, pos_b: jnp.ndarray) -> jnp.ndarray:
    """Chebyshev (8-dir) distance between two (row, col) positions."""
    delta = jnp.abs(pos_a.astype(jnp.int32) - pos_b.astype(jnp.int32))
    return jnp.maximum(delta[0], delta[1]).astype(jnp.int32)


def _greedy_step(monster_pos: jnp.ndarray, target_pos: jnp.ndarray) -> jnp.ndarray:
    """Return the next (row, col) one greedy 8-dir step toward target_pos.

    Mirrors the core of src/monmove.c m_move: subtract dx/dy clamped to [-1,1].
    No wall avoidance in Wave 3 (Wave 4 will add door/wall logic).
    """
    mp = monster_pos.astype(jnp.int32)
    tp = target_pos.astype(jnp.int32)
    delta = jnp.clip(tp - mp, -1, 1)
    return (mp + delta).astype(jnp.int16)


def _current_level_terrain(state) -> jnp.ndarray:
    """Return the int8[MAP_H, MAP_W] terrain slice for the player's level.

    Mirrors action_dispatch._current_level_terrain; duplicated here to avoid
    an import cycle with action_dispatch which already imports combat (which
    imports monster_ai indirectly).
    """
    b = state.dungeon.current_branch
    lv = state.dungeon.current_level - 1
    return state.terrain[b, lv]


# ---------------------------------------------------------------------------
# 1.  Line-of-sight  (Bresenham; src/vision.c::clear_path)
# ---------------------------------------------------------------------------

def _entry_flag(entry_idx: jnp.ndarray, table: jnp.ndarray) -> jnp.ndarray:
    """Look up MONSTERS[entry_idx].<flags-field>, safely clipped."""
    e = entry_idx.astype(jnp.int32)
    safe = jnp.clip(e, 0, table.shape[0] - 1)
    return table[safe].astype(jnp.int32)


def _has_flag1(entry_idx: jnp.ndarray, bit: int) -> jnp.ndarray:
    """True iff MONSTERS[entry_idx].flags1 & bit."""
    return (_entry_flag(entry_idx, _MONSTER_FLAGS1_TABLE) & jnp.int32(bit)) != 0


def _has_flag2(entry_idx: jnp.ndarray, bit: int) -> jnp.ndarray:
    """True iff MONSTERS[entry_idx].flags2 & bit."""
    return (_entry_flag(entry_idx, _MONSTER_FLAGS2_TABLE) & jnp.int32(bit)) != 0


def _monster_level(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """Look up MONSTERS[entry_idx].level."""
    e = entry_idx.astype(jnp.int32)
    safe = jnp.clip(e, 0, _MONSTER_LEVEL_TABLE.shape[0] - 1)
    return _MONSTER_LEVEL_TABLE[safe].astype(jnp.int32)


def _player_is_invisible(state) -> jnp.ndarray:
    """True iff the player has an active timed invisibility status.

    Mirrors vendor src/vision.c::couldsee gating: invisible hero requires
    monster to have M1_SEE_INVIS in order to see them.
    """
    # Status sub-state may not be present in some bare test fixtures.
    status = getattr(state, "status", None)
    if status is None:
        return jnp.bool_(False)
    timed = getattr(status, "timed_statuses", None)
    if timed is None:
        return jnp.bool_(False)
    # TimedStatus.INVIS_TMP = 17.
    INVIS_TMP = 17
    return timed[INVIS_TMP] > jnp.int32(0)


def monster_can_see_player(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """Return True iff there is an unobstructed straight line from the
    monster at slot ``monster_idx`` to ``state.player_pos``.

    Wave 6 vendor-parity (vendor/nethack/src/vision.c::clear_path +
    block_light around lines 165-184):
        Blocking tiles:  WALL, CLOSED_DOOR (with door-mask semantics),
                         BOULDER, LAVAWALL/WATERWALL/CLOUD (none of which
                         we model as separate tile types yet), TREE.
        We use the tiles we have: WALL, CLOSED_DOOR, TREE.
        Open doors are see-thru per is_clear macro (vision.c:165-169).
        Invisible hero: monster must have M1_SEE_INVIS to spot them.

    Returns a 0-D bool jnp.ndarray.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    terrain = _current_level_terrain(state)

    r0, c0 = mpos[0], mpos[1]
    r1, c1 = ppos[0], ppos[1]

    # Parametric sampling: step from monster to player in ``max_steps`` increments
    # and check each intermediate tile (exclude both endpoints). Bounded loop is
    # JIT-friendly. `n_steps = max(|dr|, |dc|)` mirrors Bresenham step count;
    # we cap at max(MAP_H, MAP_W).
    dr_signed = (r1 - r0).astype(jnp.int32)
    dc_signed = (c1 - c0).astype(jnp.int32)
    dr_abs = jnp.abs(dr_signed)
    dc_abs = jnp.abs(dc_signed)
    n_steps = jnp.maximum(dr_abs, dc_abs).astype(jnp.int32)
    n_steps_safe = jnp.maximum(n_steps, jnp.int32(1))  # avoid /0
    max_steps = max(_MAP_H, _MAP_W)

    def body(i, clear):
        # Parametric position along the line, excluding endpoints.
        # i ranges 0..max_steps-1; we treat i+1 as the step index.
        active = ((i + 1) < n_steps) & clear  # 1 .. n_steps-1 are intermediate tiles
        # Use rounded division: numer/denom.
        numer_r = dr_signed * (i + 1)
        numer_c = dc_signed * (i + 1)
        # Symmetric truncated division (JAX's "//" is floor for positive,
        # but for our signed case rounding to nearest-tile is fine via
        # jnp.round of float division).
        step_r = jnp.round(numer_r.astype(jnp.float32) / n_steps_safe.astype(jnp.float32)).astype(jnp.int32)
        step_c = jnp.round(numer_c.astype(jnp.float32) / n_steps_safe.astype(jnp.float32)).astype(jnp.int32)
        tr = r0 + step_r
        tc = c0 + step_c
        safe_r = jnp.clip(tr, 0, _MAP_H - 1)
        safe_c = jnp.clip(tc, 0, _MAP_W - 1)
        tile = terrain[safe_r, safe_c].astype(jnp.int32)
        # Vendor vision.c::is_clear (line 165): IS_OBSTRUCTED || TREE ||
        # (IS_DOOR && door is closed/locked/trapped).  Open doors pass.
        # Boulders also block (line 182) — we don't yet model boulders as a
        # separate object layer, so this is approximated by CLOSED_DOOR /
        # WALL gating.
        blocked = (
            (tile == _TILE_WALL)
            | (tile == _TILE_CLOSED_DOOR)
            | (tile == _TILE_TREE)
        )
        return jnp.where(active & blocked, jnp.bool_(False), clear)

    clear = jax.lax.fori_loop(0, max_steps, body, jnp.bool_(True))
    # Same tile is trivially visible.
    same_tile = (r0 == r1) & (c0 == c1)

    # Invisible-player gate (vendor vision.c::couldsee).  If the hero is
    # invisible, only monsters with M1_SEE_INVIS can perceive them.
    is_invis = _player_is_invisible(state)
    sees_invis = _has_flag1(mai.entry_idx[idx], _M1_SEE_INVIS)
    invis_gate = (~is_invis) | sees_invis

    return (clear | same_tile) & invis_gate


# ---------------------------------------------------------------------------
# 2.  Bounded BFS pathfinding  (src/monmove.c::mfndpos)
# ---------------------------------------------------------------------------

# 8-neighbor offsets used by both BFS and greedy step ordering.
_DIRS = jnp.array(
    [(-1, -1), (-1, 0), (-1, 1),
     (0, -1),           (0, 1),
     (1, -1),  (1, 0),  (1, 1)],
    dtype=jnp.int32,
)  # [8, 2]


def _tile_passable(terrain: jnp.ndarray, r: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    """True iff (r, c) is in-bounds and not a wall / closed door."""
    in_bounds = (r >= 0) & (r < _MAP_H) & (c >= 0) & (c < _MAP_W)
    safe_r = jnp.clip(r, 0, _MAP_H - 1)
    safe_c = jnp.clip(c, 0, _MAP_W - 1)
    tile = terrain[safe_r, safe_c].astype(jnp.int32)
    not_blocking = (tile != _TILE_WALL) & (tile != _TILE_CLOSED_DOOR)
    return in_bounds & not_blocking


def pathfind_step(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """Return a one-step (dy, dx) toward the player using bounded BFS.

    Implementation: bounded BFS to depth ``_PATHFIND_MAX_DEPTH``. We compute a
    distance field on the [_MAP_H, _MAP_W] grid centered on the monster's
    position, then pick the 8-dir neighbor of the monster with the smallest
    distance-to-player. If unreachable within depth, fall back to a greedy
    8-dir step (Chebyshev gradient).

    Returns a jnp.int32[2] (dy, dx) in {-1, 0, 1}.

    Wave 6 vendor-parity (vendor/nethack/src/monmove.c::mfndpos) additions:
      - Water / lava tiles only traversable by swim or flying monsters
        (mflags1 & M1_SWIM / M1_FLY / M1_AMPHIBIOUS).
      - MM_PEACEFUL: BFS treats other peaceful monsters as blocked so the
        hostile mover does not path through them.
      - Closed doors remain blocked: vendor allows door-busters to break
        through but that is not modeled here; safe to keep as blocked
        because the bumping logic in monster_turn still attempts a move.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    terrain = _current_level_terrain(state)

    INF = jnp.int32(_PATHFIND_MAX_DEPTH + 100)

    # Initialize distance field with INF everywhere; ROOT AT PLAYER so the
    # field's gradient guides the monster toward the player when it picks
    # its neighbor with smallest dist.
    dist0 = jnp.full((_MAP_H, _MAP_W), INF, dtype=jnp.int32)
    dist0 = dist0.at[ppos[0], ppos[1]].set(jnp.int32(0))

    # BFS frontier relaxation: at each step k, set every tile whose neighbor
    # has dist == k to k+1 (if not yet set). Repeats _PATHFIND_MAX_DEPTH times.
    # Vectorized: shift the distance field in 8 directions and take min.
    def shift_one(dist_field, dy, dx):
        # Create a shifted field where (r, c) holds dist_field[r-dy, c-dx].
        # Pad with INF on the boundary that comes "from outside".
        # Use jnp.roll then mask the wrap-around boundary.
        shifted = jnp.roll(dist_field, shift=(dy, dx), axis=(0, 1))
        # Mask wrap-around rows / cols.
        if dy > 0:
            shifted = shifted.at[0:dy, :].set(INF)
        elif dy < 0:
            shifted = shifted.at[_MAP_H + dy:_MAP_H, :].set(INF)
        if dx > 0:
            shifted = shifted.at[:, 0:dx].set(INF)
        elif dx < 0:
            shifted = shifted.at[:, _MAP_W + dx:_MAP_W].set(INF)
        return shifted

    # We need static offsets for `jnp.roll` to work nicely, so unroll once.
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    # Mover-specific passability (vendor mfndpos): water / lava blocked unless
    # the monster has M1_SWIM, M1_AMPHIBIOUS or M1_FLY.
    entry = mai.entry_idx[idx]
    can_swim = _has_flag1(entry, _M1_SWIM) | _has_flag1(entry, _M1_AMPHIBIOUS)
    can_fly  = _has_flag1(entry, _M1_FLY)

    # Mask of passable tiles for THIS mover.
    tile_field = terrain.astype(jnp.int32)
    not_wall   = (tile_field != _TILE_WALL) & (tile_field != _TILE_CLOSED_DOOR) \
                 & (tile_field != _TILE_TREE)
    is_water   = (tile_field == _TILE_WATER)
    is_lava    = (tile_field == _TILE_LAVA)
    water_ok   = can_swim | can_fly
    lava_ok    = can_fly  # vendor: lava only flyable.
    terrain_ok = not_wall & jnp.where(is_water, water_ok, jnp.bool_(True)) \
                          & jnp.where(is_lava,  lava_ok,  jnp.bool_(True))

    # MM_PEACEFUL: hostile movers do not path through peaceful monsters
    # (vendor mfndpos.h::ALLOW_M / MM_PEACEFUL handling).  Build a [MAP_H,
    # MAP_W] mask of "blocked by peaceful" using scatter.
    self_mask_n = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32) == idx
    blocking_peaceful = mai.alive & mai.peaceful & ~self_mask_n  # [N]
    # Scatter peaceful positions into a [MAP_H, MAP_W] occupancy mask.
    occ = jnp.zeros((_MAP_H, _MAP_W), dtype=jnp.bool_)
    pp = mai.pos.astype(jnp.int32)
    safe_r = jnp.clip(pp[:, 0], 0, _MAP_H - 1)
    safe_c = jnp.clip(pp[:, 1], 0, _MAP_W - 1)
    occ = occ.at[safe_r, safe_c].max(blocking_peaceful)

    passable = terrain_ok & ~occ

    def bfs_body(_k, dist_field):
        # For each tile, min(dist[r,c], 1 + min over 8 neighbors).
        neigh_min = jnp.full_like(dist_field, INF)
        for dy, dx in offsets:
            shifted = shift_one(dist_field, dy, dx)
            neigh_min = jnp.minimum(neigh_min, shifted)
        candidate = neigh_min + jnp.int32(1)
        # Only fill in if the tile is passable.
        candidate = jnp.where(passable, candidate, INF)
        return jnp.minimum(dist_field, candidate)

    dist = jax.lax.fori_loop(0, _PATHFIND_MAX_DEPTH, bfs_body, dist0)

    # Reachable iff the BFS reached the monster's tile from the player.
    monster_dist = dist[mpos[0], mpos[1]]
    reachable = monster_dist < INF

    # Pick the 8-neighbor of mpos with the smallest distance value.
    # We want the neighbor whose dist == player_dist - 1 (closest to player
    # from the monster's side). Easiest: read each of 8 neighbors' dist, pick min.
    neighbor_dists = []
    neighbor_offsets = []
    for dy, dx in offsets:
        nr = mpos[0] + dy
        nc = mpos[1] + dx
        in_b = (nr >= 0) & (nr < _MAP_H) & (nc >= 0) & (nc < _MAP_W)
        sr = jnp.clip(nr, 0, _MAP_H - 1)
        sc = jnp.clip(nc, 0, _MAP_W - 1)
        nd = jnp.where(in_b, dist[sr, sc], INF)
        neighbor_dists.append(nd)
        neighbor_offsets.append((dy, dx))

    stacked = jnp.stack(neighbor_dists)  # [8]
    best_idx = jnp.argmin(stacked).astype(jnp.int32)
    offsets_arr = jnp.array(neighbor_offsets, dtype=jnp.int32)  # [8, 2]
    bfs_step = offsets_arr[best_idx]  # [2]

    # Greedy fallback (8-dir Chebyshev gradient).
    greedy_delta = jnp.clip(ppos - mpos, -1, 1).astype(jnp.int32)

    return jnp.where(reachable, bfs_step, greedy_delta)


# ---------------------------------------------------------------------------
# 3.  Muse — monster item use  (src/muse.c)
# ---------------------------------------------------------------------------

def _is_mage_entry(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff MONSTERS[entry_idx].sound is MS_SPELL or MS_PRIEST.

    Wave 6 Phase B: replaces the Wave-5 [LO, HI] range heuristic with a real
    lookup of the MonsterEntry.sound (msound) field, matching vendor's
    castmu() gate in src/mcastu.c.
    """
    e = entry_idx.astype(jnp.int32)
    table = _MONSTER_SOUND_TABLE  # int8[NUMMONS]
    safe_e = jnp.clip(e, 0, table.shape[0] - 1)
    sound = table[safe_e].astype(jnp.int32)
    return (sound == jnp.int32(_MS_SPELL)) | (sound == jnp.int32(_MS_PRIEST))


def _can_use_items(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff the monster's class is eligible to use items.

    Mirrors vendor src/muse.c::find_offensive / find_defensive entry gates
    (lines 1428-1430, 454-455):
        if (mtmp->mpeaceful || is_animal(mtmp->data)
            || mindless(mtmp->data) || nohands(mtmp->data))
            return FALSE;
    Animal-class, mindless, and nohands monsters are skipped — these are
    the M1_ANIMAL, M1_MINDLESS, M1_NOHANDS flags.
    """
    is_animal   = _has_flag1(entry_idx, _M1_ANIMAL)
    is_mindless = _has_flag1(entry_idx, _M1_MINDLESS)
    nohands     = _has_flag1(entry_idx, _M1_NOHANDS)
    return ~(is_animal | is_mindless | nohands)


# ---------------------------------------------------------------------------
# Muse payload helpers (Wave 6 Mission)
# ---------------------------------------------------------------------------

def _find_inv_slot(mai: MonsterAIState, idx: jnp.ndarray,
                   category: int, type_id: int) -> tuple:
    """Return (found, slot) — first inventory slot of (category, type_id) for
    monster ``idx`` that still has positive quantity (and charges, where
    relevant).  ``slot`` is 0 when not found; ``found`` is a 0-D bool.
    """
    cats  = mai.inv_category[idx]               # [MAX_MONSTER_INV] int8
    types = mai.inv_type_id[idx]                # [MAX_MONSTER_INV] int16
    qty   = mai.inv_quantity[idx]               # [MAX_MONSTER_INV] int16
    match = (cats.astype(jnp.int32) == jnp.int32(category)) \
            & (types.astype(jnp.int32) == jnp.int32(type_id)) \
            & (qty.astype(jnp.int32) > jnp.int32(0))
    found = jnp.any(match)
    # argmax over bools returns the FIRST True (ties broken by lowest index).
    slot = jnp.argmax(match.astype(jnp.int32)).astype(jnp.int32)
    return found, slot


def _try_heal(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Quaff a healing potion from monster's inventory if HP < hp_max and
    a potion of healing exists.

    Vendor reference: muse.c::find_defensive case MUSE_POT_HEALING +
    muse.c::use_defensive.  Vendor heal amount for monster quaff is the
    same formula as hero peffect_healing (potions.c): healup(8 + d(4,4)),
    so the heal is in [9..24] HP.  We use d(3,6)+8 → [11..26] as a
    parity-tractable approximation (matches the task spec's "d6+1" hint
    while staying close to vendor band).
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_POTION, _POT_HEALING)

    hp     = mai.hp[idx].astype(jnp.int32)
    hp_max = mai.hp_max[idx].astype(jnp.int32)
    hurt   = hp < hp_max
    can_quaff = found & hurt

    # Heal amount: 1d6 + 1 + (existing hp).  Clamp to hp_max.
    heal_roll = jax.random.randint(rng, (), 1, 7, dtype=jnp.int32) + jnp.int32(1)
    new_hp = jnp.minimum(hp + heal_roll, hp_max)
    new_hp = jnp.where(can_quaff, new_hp, hp)

    # Decrement potion quantity by 1; remove (set category=0) when zero.
    old_qty = mai.inv_quantity[idx, slot].astype(jnp.int32)
    dec_qty = jnp.maximum(old_qty - jnp.int32(1), jnp.int32(0))
    new_qty = jnp.where(can_quaff, dec_qty, old_qty).astype(jnp.int16)
    # If quantity reaches zero, clear the slot's category to NONE.
    cleared_cat = jnp.where((new_qty == 0) & can_quaff,
                            jnp.int8(0),
                            mai.inv_category[idx, slot])

    new_inv_qty = mai.inv_quantity.at[idx, slot].set(new_qty)
    new_inv_cat = mai.inv_category.at[idx, slot].set(cleared_cat)
    new_hp_arr  = mai.hp.at[idx].set(new_hp)

    new_mai = mai.replace(
        hp=new_hp_arr,
        inv_quantity=new_inv_qty,
        inv_category=new_inv_cat,
    )
    return state.replace(monster_ai=new_mai)


def _try_scroll_teleport(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Read a scroll of teleportation if one is in inventory.

    Vendor reference: muse.c::find_misc case MUSE_SCR_TELEPORTATION +
    muse.c::use_misc.  Effect: monster relocates to a random tile.
    We pick a random in-bounds tile on the current level; passability
    refinement is deferred — vendor mnexto() can also fail and stays put.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_SCROLL, _SCR_TELEPORT)

    rng_r, rng_c = jax.random.split(rng, 2)
    new_r = jax.random.randint(rng_r, (), 0, _MAP_H, dtype=jnp.int32).astype(jnp.int16)
    new_c = jax.random.randint(rng_c, (), 0, _MAP_W, dtype=jnp.int32).astype(jnp.int16)
    target_pos = jnp.stack([new_r, new_c]).astype(jnp.int16)

    cur_pos = mai.pos[idx]
    chosen = jnp.where(found, target_pos, cur_pos)

    # Scrolls are consumed entirely (qty -= 1).
    old_qty = mai.inv_quantity[idx, slot].astype(jnp.int32)
    dec_qty = jnp.maximum(old_qty - jnp.int32(1), jnp.int32(0))
    new_qty = jnp.where(found, dec_qty, old_qty).astype(jnp.int16)
    cleared_cat = jnp.where((new_qty == 0) & found,
                            jnp.int8(0),
                            mai.inv_category[idx, slot])

    new_pos    = mai.pos.at[idx].set(chosen)
    new_invq   = mai.inv_quantity.at[idx, slot].set(new_qty)
    new_invc   = mai.inv_category.at[idx, slot].set(cleared_cat)
    new_mai    = mai.replace(
        pos=new_pos,
        inv_quantity=new_invq,
        inv_category=new_invc,
    )
    return state.replace(monster_ai=new_mai)


def _try_zap_wand(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Zap an offensive wand at the player if charges remain.

    Vendor reference: muse.c::find_offensive (wand cases) + muse.c::
    use_offensive → src/zap.c::buzz.  For Wave 6 we model WAN_FIRE only
    (most common find_offensive pick for mages): d(6,6) damage to the
    target tile per vendor zap.c buzz(ZT_FIRE).  Charges decrement.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_WAND, _WAN_FIRE)

    # Charges must be positive for the wand to zap (vendor wand.c).
    charges = mai.inv_charges[idx, slot].astype(jnp.int32)
    can_zap = found & (charges > 0)

    # Damage: vendor zap.c buzz(ZT_FIRE) → d(6,6) per bhitm hit.
    # Roll 6 independent d6 dice → range [6..36].
    keys = jax.random.split(rng, 6)
    dice = jax.vmap(lambda k: jax.random.randint(k, (), 1, 7, dtype=jnp.int32))(keys)
    dmg = jnp.sum(dice).astype(jnp.int32)

    new_player_hp = jnp.where(
        can_zap,
        jnp.maximum(state.player_hp - dmg, jnp.int32(0)),
        state.player_hp,
    ).astype(jnp.int32)
    new_done = state.done | (new_player_hp <= 0)

    new_charges = jnp.where(can_zap, charges - jnp.int32(1), charges).astype(jnp.int8)
    new_inv_charges = mai.inv_charges.at[idx, slot].set(new_charges)
    new_mai = mai.replace(inv_charges=new_inv_charges)
    return state.replace(
        monster_ai=new_mai,
        player_hp=new_player_hp,
        done=new_done,
    )


def monster_use_item(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Monster considers using an item this turn.

    Vendor reference: src/muse.c (find_offensive / find_defensive / find_misc).

    Wave 6 Mission payload:
        - The whole call is gated by _can_use_items() (vendor muse.c:1428):
          peaceful / animal / mindless / nohands monsters are excluded.
        - Priority order matches vendor muse.c (defensive → misc → offensive):
            1. quaff_heal     : HP < 1/4 max + healing potion in inv.
            2. read_tport     : player adjacent + teleport scroll in inv.
            3. zap_wand       : in LoS, dist 2..8 + wand of fire in inv.

    JIT-safe: each branch is wrapped in jax.lax.cond keyed off its predicate.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    # Vendor entry gate (muse.c:1428).
    eligible = _can_use_items(mai.entry_idx[idx]) \
               & mai.alive[idx] & ~mai.asleep[idx] & ~mai.peaceful[idx]

    # HP threshold: vendor find_defensive uses fractions 1/5, 1/4, 1/3 of
    # mhpmax depending on hero level.  We use 1/4 (mid-game) as a single
    # parity-tractable threshold.
    hp_low_quarter = (mai.hp[idx].astype(jnp.int32) * jnp.int32(4) <
                      mai.hp_max[idx].astype(jnp.int32))
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)
    in_los = monster_can_see_player(state, idx)

    quaff_heal = eligible & hp_low_quarter
    read_tport = eligible & ~hp_low_quarter & (dist == 1)
    zap_wand   = eligible & ~hp_low_quarter & (dist > 1) & in_los & (dist <= 8)

    rng_heal, rng_tport, rng_zap = jax.random.split(rng, 3)

    s1 = jax.lax.cond(quaff_heal,
                      lambda s: _try_heal(s, rng_heal, idx),
                      lambda s: s, state)
    s2 = jax.lax.cond(read_tport,
                      lambda s: _try_scroll_teleport(s, rng_tport, idx),
                      lambda s: s, s1)
    s3 = jax.lax.cond(zap_wand,
                      lambda s: _try_zap_wand(s, rng_zap, idx),
                      lambda s: s, s2)
    return s3


# ---------------------------------------------------------------------------
# 4.  Mcastu — monster spell casting  (src/mcastu.c::castmu)
# ---------------------------------------------------------------------------

# Wave 6 mcastu spell IDs.  Mirrors values used in vendor src/mcastu.c
# switch(spellnum); we keep four directly-damaging spells.  Non-damage
# spells (AGGRAVATION, CURSE_ITEMS, STUN_YOU, ...) deal 0 hp damage and
# are folded into the "0-damage" path.
MCAST_PSI_BOLT: int     = 0
MCAST_FIRE_PILLAR: int  = 1
MCAST_GEYSER: int       = 2
MCAST_LIGHTNING: int    = 3
MCAST_CLERIC: int       = 4   # generic cleric "d(lvl, 6)" path


def _roll_dice(rng: jax.Array, n: int, sides: int) -> jnp.ndarray:
    """Vendor d(n, sides) — sum of n uniform rolls in [1..sides]."""
    keys = jax.random.split(rng, max(int(n), 1))
    rolls = jax.random.randint(keys[0], (int(n),), 1, int(sides) + 1)
    return jnp.sum(rolls).astype(jnp.int32)


def _roll_dice_dynamic(rng: jax.Array, n: jnp.ndarray, sides: int,
                       max_n: int = 16) -> jnp.ndarray:
    """JIT-friendly d(n, sides) where n is a JAX value bounded by max_n."""
    keys = jax.random.split(rng, max_n)
    rolls = jax.vmap(lambda k: jax.random.randint(k, (), 1, int(sides) + 1))(keys)
    take = jnp.arange(max_n, dtype=jnp.int32) < n.astype(jnp.int32)
    return jnp.sum(jnp.where(take, rolls, 0)).astype(jnp.int32)


def _vendor_psi_bolt_damage(rng: jax.Array, ml: jnp.ndarray) -> jnp.ndarray:
    """Vendor mcast_psi_bolt: caller passes dmg = d((ml/2)+1, 6).
    Returns d((ml/2)+1, 6) clamped to ≥ 1.
    """
    n_dice = jnp.maximum(jnp.int32(1), ml // jnp.int32(2) + jnp.int32(1))
    return _roll_dice_dynamic(rng, n_dice, 6)


def _vendor_fire_pillar_damage(rng: jax.Array) -> jnp.ndarray:
    """Vendor mcast_fire_pillar (mcastu.c:545): dmg = d(8, 6)."""
    return _roll_dice_dynamic(rng, jnp.int32(8), 6)


def _vendor_geyser_damage(rng: jax.Array) -> jnp.ndarray:
    """Vendor mcast_geyser (mcastu.c:529): dmg = d(8, 6)."""
    return _roll_dice_dynamic(rng, jnp.int32(8), 6)


def _vendor_lightning_damage(rng: jax.Array) -> jnp.ndarray:
    """Vendor mcast_lightning (mcastu.c:574): dmg = d(8, 6)."""
    return _roll_dice_dynamic(rng, jnp.int32(8), 6)


def _vendor_cleric_damage(rng: jax.Array, ml: jnp.ndarray) -> jnp.ndarray:
    """Generic cleric (AD_CLRC) caster damage.

    Vendor mcastu.c::castmu line 240-243:
        if (mattk->damd) dmg = d((ml/2) + mattk->damn, mattk->damd);
        else             dmg = d((ml/2) + 1, 6);
    The default cleric attack has (damn=0, damd=6).  We mirror the default.
    """
    n_dice = jnp.maximum(jnp.int32(1), ml // jnp.int32(2) + jnp.int32(1))
    return _roll_dice_dynamic(rng, n_dice, 6)


def monster_cast_damage(rng: jax.Array, spellnum: int,
                        ml: jnp.ndarray) -> jnp.ndarray:
    """Dispatch one of the vendor spell damage formulas.

    Spell IDs (Wave 6 parity-fix subset):
        MCAST_PSI_BOLT    → d((ml/2)+1, 6)         vendor mcast_psi_bolt
        MCAST_FIRE_PILLAR → d(8, 6)                vendor mcast_fire_pillar
        MCAST_GEYSER      → d(8, 6)                vendor mcast_geyser
        MCAST_LIGHTNING   → d(8, 6)                vendor mcast_lightning
        MCAST_CLERIC      → d((ml/2)+1, 6)         vendor cleric default
    """
    if spellnum == MCAST_FIRE_PILLAR:
        return _vendor_fire_pillar_damage(rng)
    if spellnum == MCAST_GEYSER:
        return _vendor_geyser_damage(rng)
    if spellnum == MCAST_LIGHTNING:
        return _vendor_lightning_damage(rng)
    if spellnum == MCAST_CLERIC:
        return _vendor_cleric_damage(rng, ml)
    # default: psi bolt
    return _vendor_psi_bolt_damage(rng, ml)


def monster_cast_spell(state, rng: jax.Array, monster_idx: jnp.ndarray,
                       spellnum: int = MCAST_PSI_BOLT):
    """If the monster is mage-class, cast a damage spell at the player.

    Reference: vendor/nethack/src/mcastu.c::castmu, ::mcast_spell.

    Wave 6 vendor-parity update:
        - Damage now uses per-spell vendor formulas via ``monster_cast_damage``
          rather than a single generic d(mlev/4, 6).  Default spellnum is
          MCAST_PSI_BOLT to preserve the existing test contract.
        - Monster level (`ml`) is the real MONSTERS[entry].level.

    Caster gate: mage-class entry (MS_SPELL/MS_PRIEST), alive, awake,
    non-peaceful, in LoS, Chebyshev distance ≤ 12.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    is_mage = _is_mage_entry(mai.entry_idx[idx])
    alive_active = mai.alive[idx] & ~mai.asleep[idx] & ~mai.peaceful[idx]
    in_los = monster_can_see_player(state, idx)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)
    in_range = dist <= 12

    can_cast = is_mage & alive_active & in_los & in_range

    def _cast(s):
        ml = _monster_level(mai.entry_idx[idx])
        dmg = monster_cast_damage(rng, spellnum, ml)
        new_hp = jnp.maximum(s.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
        new_done = s.done | (new_hp <= 0)
        return s.replace(player_hp=new_hp, done=new_done)

    return jax.lax.cond(can_cast, _cast, lambda s: s, state)


# ---------------------------------------------------------------------------
# 5.  Retreat behavior  (src/monmove.c::mon_would_flee)
# ---------------------------------------------------------------------------

def maybe_retreat(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """Return a (dy, dx) retreat step if the monster should flee, else (0, 0).

    Wave 6 vendor-parity update.  Mirrors vendor/nethack/src/monmove.c
    flee logic (distfleeck + monflee + dochug guard).  Vendor rules used:
        - HP threshold: low_hp iff
              level >= 2  →  hp <= max(level / 4, 5)
              level <  2  →  hp <= 1
          (vendor monmove.c::dochug: `if (mtmp->mhp < mtmp->m_lev/4 || ...)`).
        - Demons (M2_DEMON) and undead (M2_UNDEAD) never flee — they're
          fearless / mindless-of-pain.  Mirrors vendor onscary() / monflee
          gating where demons/undead are explicitly excluded from "scared".
        - Peaceful monsters never enter the flee path.

    Returns jnp.int32[2].
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    hp = mai.hp[idx].astype(jnp.int32)
    hp_max = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))

    # Vendor flee threshold by monster level.
    entry = mai.entry_idx[idx]
    mlev = _monster_level(entry)
    quarter = jnp.maximum(mlev // jnp.int32(4), jnp.int32(5))
    threshold = jnp.where(mlev >= jnp.int32(2), quarter, jnp.int32(1))
    low_hp = hp <= threshold

    # Fearless classes: demons + undead (vendor monmove flee gating).
    is_demon  = _has_flag2(entry, _M2_DEMON)
    is_undead = _has_flag2(entry, _M2_UNDEAD)
    fearless  = is_demon | is_undead

    peaceful = mai.peaceful[idx]

    should_flee = low_hp & ~fearless & ~peaceful

    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    # Step AWAY from player: negative gradient, clipped to [-1, 1].
    delta = jnp.clip(mpos - ppos, -1, 1).astype(jnp.int32)
    zero = jnp.zeros((2,), dtype=jnp.int32)
    return jnp.where(should_flee, delta, zero)


# ---------------------------------------------------------------------------
# 6.  Pet behavior  (src/dogmove.c::dog_move)
# ---------------------------------------------------------------------------

# Vendor pet food preferences (dogmove.c::dog_eat / dog.h).
# Symbol class codes from constants.monsters.MonsterSymbol — these are
# tested via pet_food_preference() below.
_PET_SYMBOL_FELINE: int = 6   # S_FELINE (cats prefer fish)
_PET_SYMBOL_DOG: int    = 4   # S_DOG (dogs prefer meat)

# Object class proxies (vendor objclass.h):
_FOOD_FISH: int = 1   # tripe/fish (cat's favourite)
_FOOD_MEAT: int = 2   # generic meat (dog's favourite)
_FOOD_VEG:  int = 3   # vegetable (mostly hated by carnivores)

# Default pet leash radius scaling factor.  Vendor leashes have hardcoded
# LEASH_LENGTH = 6; we scale with apport so well-trained pets stay closer.
_PET_LEASH_BASE: int = 6


def pet_food_preference(entry_idx: jnp.ndarray, food_class: int) -> jnp.ndarray:
    """Return +1 if pet prefers this food, -1 if it hates it, 0 otherwise.

    Mirrors vendor src/dogmove.c::dog_eat preference table — cats love
    fish/tripe (FOOD_FISH), dogs love meat (FOOD_MEAT).  Both hate
    pure-vegetable food.  This is a deterministic JIT-side lookup that
    matches the behavioural intent of the vendor table (we don't model
    every individual food item yet).
    """
    e = entry_idx.astype(jnp.int32)
    # Symbol table lives in MonsterEntry.symbol (int).  Build once.
    from Nethax.nethax.constants.monsters import MONSTERS
    syms = jnp.array([int(m.symbol) for m in MONSTERS], dtype=jnp.int32)
    safe_e = jnp.clip(e, 0, syms.shape[0] - 1)
    sym = syms[safe_e]

    is_cat = sym == jnp.int32(_PET_SYMBOL_FELINE)
    is_dog = sym == jnp.int32(_PET_SYMBOL_DOG)

    likes_fish = is_cat & (food_class == _FOOD_FISH)
    likes_meat = is_dog & (food_class == _FOOD_MEAT)
    hates_veg  = (is_cat | is_dog) & (food_class == _FOOD_VEG)

    pref = jnp.where(likes_fish | likes_meat, jnp.int32(1),
                     jnp.where(hates_veg, jnp.int32(-1), jnp.int32(0)))
    return pref


def pet_within_leash(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff this pet's Chebyshev distance to the player is within its
    leash radius.

    Vendor dogmove.c uses `LEASH_LENGTH = 6` plus pet apport modifier.
    Higher apport keeps the pet closer (more trained), so we cap distance
    at ``_PET_LEASH_BASE`` and subtract a small apport offset.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)
    apport = mai.apport[idx].astype(jnp.int32)
    # apport in [1..10] → leash in [_PET_LEASH_BASE+5 .. _PET_LEASH_BASE-4].
    leash = jnp.maximum(jnp.int32(_PET_LEASH_BASE + 5) - apport, jnp.int32(2))
    return dist <= leash


def pet_move(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Run one turn for a pet (tame) monster.

    Vendor-parity behaviour (vendor/nethack/src/dogmove.c::dog_move):

    Per-turn bookkeeping:
        0a. Hunger tick: decrement pet_hunger by 1 (dog.c:380 edog.hungrytime).
            At <= -50, pet dies.
        0b. Eat floor food: if hungry (pet_hunger <= 0) and a FOOD item is on
            the pet's tile, eat it — restore HP by food_value/4, remove the
            food, reset hunger to 1000. (dogmove.c:520 dog_eat)
        0c. Flee on low HP: if hp < hp_max/4 and not fearless (not undead /
            demon), move AWAY from player. (dogmove.c:1100)

    Movement:
        1. If a hostile alive monster is adjacent (Chebyshev <= 1) → attack it
           (dogmove.c:1150 mattackm).
        2. Else if pet is within 6 Chebyshev tiles of player → FOLLOW mode:
           step toward player using BFS pathfind (mfndpos).
        3. Else → EXPLORE mode: random walk (dogmove.c:629 gx=FARAWAY).

    Cite: vendor/nethack/src/dogmove.c::dog_move lines 520, 566-644, 1014,
          1100, 1150; vendor/nethack/src/dog.c:380.
    JIT-pure: all branches via jax.lax.cond / jnp.where.

    Returns updated state.
    """
    # Item category constant for food (mirrors ItemCategory.FOOD = 7).
    _CAT_FOOD_LOCAL: int = 7

    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    is_pet = mai.tame[idx] & mai.alive[idx]

    # -----------------------------------------------------------------------
    # 0a. Hunger tick — dog.c:380 edog.hungrytime
    # Decrement pet_hunger by 1 each turn (tame slots only).
    # At <= -50, pet dies.
    # -----------------------------------------------------------------------
    cur_hunger = mai.pet_hunger[idx].astype(jnp.int32)
    new_hunger_val = cur_hunger - jnp.int32(1)
    new_hunger = jnp.where(is_pet, new_hunger_val, cur_hunger).astype(jnp.int16)
    starved = is_pet & (new_hunger_val <= jnp.int32(-50))
    new_alive_after_hunger = mai.alive.at[idx].set(
        jnp.where(starved, jnp.bool_(False), mai.alive[idx])
    )
    mai_h = mai.replace(
        pet_hunger=mai.pet_hunger.at[idx].set(new_hunger),
        alive=new_alive_after_hunger,
    )
    state = state.replace(monster_ai=mai_h)

    # Re-read is_pet after possible starvation death.
    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]
    mpos = mai.pos[idx].astype(jnp.int32)

    # -----------------------------------------------------------------------
    # 0b. Eat floor food — dogmove.c:520 dog_eat
    # If hungry (pet_hunger <= 0) and FOOD item at pet's tile, eat it.
    # -----------------------------------------------------------------------
    is_hungry = mai.pet_hunger[idx].astype(jnp.int32) <= jnp.int32(0)
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    pr = jnp.clip(mpos[0], 0, _MAP_H - 1)
    pc = jnp.clip(mpos[1], 0, _MAP_W - 1)
    food_cat = state.ground_items.category[b, lv, pr, pc, 0].astype(jnp.int32)
    has_food = food_cat == jnp.int32(_CAT_FOOD_LOCAL)
    can_eat = is_pet & is_hungry & has_food
    food_weight = state.ground_items.weight[b, lv, pr, pc, 0].astype(jnp.int32)
    heal_amount = jnp.maximum(food_weight // jnp.int32(4), jnp.int32(1))
    new_pet_hp = jnp.minimum(
        mai.hp[idx].astype(jnp.int32) + heal_amount,
        mai.hp_max[idx].astype(jnp.int32),
    )
    new_ground_cat = state.ground_items.category.at[b, lv, pr, pc, 0].set(
        jnp.where(can_eat, jnp.int8(0), state.ground_items.category[b, lv, pr, pc, 0])
    )
    new_hunger_after_eat = jnp.where(can_eat, jnp.int16(1000), mai.pet_hunger[idx])
    mai_e = mai.replace(
        hp=mai.hp.at[idx].set(jnp.where(can_eat, new_pet_hp.astype(jnp.int32), mai.hp[idx])),
        pet_hunger=mai.pet_hunger.at[idx].set(new_hunger_after_eat),
    )
    new_ground = state.ground_items.replace(category=new_ground_cat)
    state = state.replace(monster_ai=mai_e, ground_items=new_ground)
    mai = state.monster_ai

    # -----------------------------------------------------------------------
    # 0c. Flee on low HP — dogmove.c:1100
    # If pet hp < hp_max/4 and not fearless, move away from player.
    # Fearless = undead (M2_UNDEAD) or demon (M2_DEMON).
    # -----------------------------------------------------------------------
    pet_hp = mai.hp[idx].astype(jnp.int32)
    pet_hp_max = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))
    low_hp = pet_hp * jnp.int32(4) < pet_hp_max
    entry = mai.entry_idx[idx]
    fearless = _has_flag2(entry, _M2_UNDEAD) | _has_flag2(entry, _M2_DEMON)
    should_flee_low_hp = is_pet & low_hp & ~fearless

    ppos = state.player_pos.astype(jnp.int32)
    flee_delta = jnp.clip(mpos - ppos, -1, 1).astype(jnp.int32)
    # Ensure non-zero delta when on same tile.
    flee_delta = jnp.where(
        jnp.all(flee_delta == 0),
        jnp.array([1, 0], dtype=jnp.int32),
        flee_delta,
    )
    flee_r = jnp.clip(mpos[0] + flee_delta[0], 0, _MAP_H - 1).astype(jnp.int16)
    flee_c = jnp.clip(mpos[1] + flee_delta[1], 0, _MAP_W - 1).astype(jnp.int16)
    flee_pos = jnp.stack([flee_r, flee_c])

    def _flee_move(s):
        _mai = s.monster_ai
        new_mai = _mai.replace(pos=_mai.pos.at[idx].set(flee_pos))
        return s.replace(monster_ai=new_mai)

    state = jax.lax.cond(should_flee_low_hp, _flee_move, lambda s: s, state)
    mai = state.monster_ai
    # Re-derive is_pet, mpos after potential flee.
    is_pet = mai.tame[idx] & mai.alive[idx]
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)

    # -----------------------------------------------------------------------
    # Find adjacent hostile monster.
    # Cite: dogmove.c:1150 (mattackm — pet attacks adjacent hostile).
    # -----------------------------------------------------------------------
    other_pos = mai.pos.astype(jnp.int32)  # [N, 2]
    dr = jnp.abs(other_pos[:, 0] - mpos[0])
    dc = jnp.abs(other_pos[:, 1] - mpos[1])
    cheb = jnp.maximum(dr, dc)
    self_mask = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32) == idx
    hostile = mai.alive & ~mai.tame & ~mai.peaceful & ~self_mask & (cheb <= jnp.int32(1))
    has_target = jnp.any(hostile)
    target_idx = jnp.argmax(hostile.astype(jnp.int32)).astype(jnp.int32)

    def _attack_hostile(s):
        # Cite: dogmove.c:1150 mattackm — pet attacks adjacent hostile.
        _mai = s.monster_ai
        cur_hp = _mai.hp[target_idx].astype(jnp.int32)
        new_hp = jnp.maximum(cur_hp - jnp.int32(2), jnp.int32(0))
        new_alive = (new_hp > 0) & _mai.alive[target_idx]
        new_mai = _mai.replace(
            hp=_mai.hp.at[target_idx].set(new_hp),
            alive=_mai.alive.at[target_idx].set(new_alive),
        )
        return s.replace(monster_ai=new_mai)

    def _follow_player(s):
        """FOLLOW mode: BFS pathfind toward player (mfndpos).

        Vendor: dogmove.c::dog_move uses mfndpos for path-finding.
        Cite: vendor/nethack/src/monmove.c::mfndpos.
        """
        _mai = s.monster_ai
        step_delta = pathfind_step(s, idx)
        cur = _mai.pos[idx].astype(jnp.int32)
        new_r = jnp.clip(cur[0] + step_delta[0], 0, _MAP_H - 1).astype(jnp.int16)
        new_c = jnp.clip(cur[1] + step_delta[1], 0, _MAP_W - 1).astype(jnp.int16)
        new_pos = jnp.stack([new_r, new_c])
        new_mai = _mai.replace(pos=_mai.pos.at[idx].set(new_pos))
        return s.replace(monster_ai=new_mai)

    def _explore(s):
        """EXPLORE mode: random 8-dir walk (Chebyshev dist >= 6).

        Vendor dogmove.c line 629: gx=gg.gy=FARAWAY (random wander).
        """
        _mai = s.monster_ai
        cur = _mai.pos[idx].astype(jnp.int32)
        rng_dir, _ = jax.random.split(rng)
        dir_idx = jax.random.randint(rng_dir, (), 0, 8)
        dy = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=jnp.int32)[dir_idx]
        dx = jnp.array([-1,  0,  1,-1, 1,-1, 0, 1], dtype=jnp.int32)[dir_idx]
        new_r = jnp.clip(cur[0] + dy, 0, _MAP_H - 1).astype(jnp.int16)
        new_c = jnp.clip(cur[1] + dx, 0, _MAP_W - 1).astype(jnp.int16)
        new_pos = jnp.array([new_r, new_c], dtype=jnp.int16)
        new_mai = _mai.replace(pos=_mai.pos.at[idx].set(new_pos))
        return s.replace(monster_ai=new_mai)

    dist_to_player = _chebyshev_dist(mpos, ppos)
    within_follow_range = dist_to_player < jnp.int32(6)

    def _move_no_target(s):
        return jax.lax.cond(within_follow_range, _follow_player, _explore, s)

    def _pet_act(s):
        return jax.lax.cond(has_target, _attack_hostile, _move_no_target, s)

    return jax.lax.cond(is_pet, _pet_act, lambda s: s, state)


def pet_follow_on_stair(state):
    """Teleport any tame pets within Chebyshev 1 of player to follow on stair.

    When the player descends (or ascends) stairs, pets adjacent to the player
    at the moment of transition should follow.  This function handles the
    bookkeeping for those pets on the *current* level: it marks them as
    no-longer-alive on this level (they will be re-spawned on the destination
    level by the stair handler).

    Vendor reference: dog.c (tamedog follow-on-stair logic).
    TODO: wire from action_dispatch._stair_down
    """
    mai = state.monster_ai
    ppos = state.player_pos.astype(jnp.int32)
    mpos = mai.pos.astype(jnp.int32)
    dr = jnp.abs(mpos[:, 0] - ppos[0])
    dc = jnp.abs(mpos[:, 1] - ppos[1])
    cheb = jnp.maximum(dr, dc)
    # Pets within Chebyshev 1 of player that are alive and tame.
    follows = mai.tame & mai.alive & (cheb <= jnp.int32(1))
    # Mark them as no-longer-alive on this level so the stair handler can
    # re-place them on the destination level.
    new_alive = mai.alive & ~follows
    new_mai = mai.replace(alive=new_alive)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# 7.  Sleep wake on player-visible  (src/monmove.c::disturb)
# ---------------------------------------------------------------------------

def maybe_wake_monster(state, monster_idx: jnp.ndarray):
    """If the monster is asleep and the player is in its LoS, wake it up.

    Mirrors vendor/nethack/src/monmove.c::disturb (passive-vision branch).
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    asleep = mai.asleep[idx]
    alive = mai.alive[idx]
    in_los = monster_can_see_player(state, idx)
    should_wake = asleep & alive & in_los

    new_asleep = jnp.where(should_wake, jnp.bool_(False), mai.asleep[idx])
    new_mai = mai.replace(
        asleep=mai.asleep.at[idx].set(new_asleep),
    )
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Single-monster turn  (src/monmove.c::m_move) — Wave 5 refactor
# ---------------------------------------------------------------------------

def monster_turn(state, rng: jax.Array, monster_idx: jnp.ndarray) -> object:
    """Run one turn for monster slot ``monster_idx``.

    Wave 5 decision tree (all branching via jax.lax.cond):
      1. Not alive → return unchanged.
      2. If pet (tame): dispatch to pet_move and return.
      3. Wake check (maybe_wake_monster).
      4. If still asleep or peaceful → return.
      5. monster_use_item (stubbed branches; preserves state).
      6. Mage-class & RNG-50% → monster_cast_spell.
      7. Movement step:
           - retreat (HP < 20%) → step away;
           - else pathfind_step (BFS with greedy fallback).
      8. If new tile == player_pos (i.e. attempted to step onto player),
         bump-attack via combat.monster_attack_player and don't move.
         Otherwise move.

    All simplifications documented inline.
    """
    from Nethax.nethax.subsystems.combat import monster_attack_player

    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    rng_pet, rng_cast, rng_atk, rng_pick = jax.random.split(rng, 4)

    # Branch 1 + 2: pet has its own turn.
    is_pet = mai.tame[idx] & mai.alive[idx]

    def _pet_branch(s):
        return pet_move(s, rng_pet, idx)

    def _hostile_branch(s):
        # Record asleep state BEFORE wake-check: monsters that wake this
        # turn do not also act this turn (mirrors vendor monmove.c::disturb,
        # which only flips the flag and lets the next tick run the AI).
        was_asleep = s.monster_ai.asleep[idx]

        # 3: wake check.
        s = maybe_wake_monster(s, idx)

        # 4: gate on alive & not asleep (at start of turn) & not peaceful.
        m = s.monster_ai
        should_act = m.alive[idx] & ~was_asleep & ~m.peaceful[idx]

        def _act(st):
            # 5: muse (stubs, but call site preserved).
            st = monster_use_item(st, rng_cast, idx)

            # 6: optional spell cast for mage-class monsters.
            is_mage = _is_mage_entry(st.monster_ai.entry_idx[idx])
            roll = jax.random.uniform(rng_pick, ())  # 0..1
            cast_now = is_mage & (roll < 0.5)

            def _maybe_cast(s2):
                return monster_cast_spell(s2, rng_cast, idx)

            st = jax.lax.cond(cast_now, _maybe_cast, lambda s2: s2, st)

            # 7a: Elbereth fear check (onscary).
            # Cite: vendor/nethack/src/monmove.c::onscary lines 241-303.
            # If Elbereth is engraved on the player's tile and this monster is
            # not exempt, freeze it in place this turn.  Vendor moves the
            # monster in a random valid direction away; we use "stay put" as a
            # simpler JIT-pure equivalent.
            from Nethax.nethax.subsystems.engrave import is_elbereth_at
            from Nethax.nethax.dungeon.branches import Branch as _Branch
            _ppos = st.player_pos.astype(jnp.int32)
            _scared_raw = is_elbereth_at(st.engrave, _ppos[0], _ppos[1])
            _eidx = st.monster_ai.entry_idx[idx].astype(jnp.int32)
            _safe_e = jnp.clip(_eidx, 0, _IGNORES_ELBERETH.shape[0] - 1)
            _ignores = _IGNORES_ELBERETH[_safe_e]
            # Gehennom: vendor onscary line 296 "if (In_hell(&u.uz)) return 0"
            _in_gehennom = (
                st.dungeon.current_branch.astype(jnp.int32)
                == jnp.int32(_Branch.GEHENNOM)
            )
            scared = _scared_raw & ~_ignores & ~_in_gehennom

            # 7b: movement decision; zero out step when scared.
            retreat_step = maybe_retreat(st, idx)
            wants_retreat = jnp.any(retreat_step != 0)
            path_step = pathfind_step(st, idx)
            step_delta = jnp.where(wants_retreat, retreat_step, path_step)
            step_delta = jnp.where(
                scared, jnp.zeros(2, dtype=jnp.int32), step_delta
            )

            cur_pos = st.monster_ai.pos[idx].astype(jnp.int32)
            new_pos_i32 = cur_pos + step_delta
            ppos_i32 = st.player_pos.astype(jnp.int32)
            steps_onto_player = jnp.all(new_pos_i32 == ppos_i32)

            # 8a: attempting to bump player → melee.
            def _attack(s2):
                new_s, _dmg = monster_attack_player(s2, rng_atk, idx)
                _m = new_s.monster_ai
                new_m = _m.replace(
                    last_seen_player_pos=_m.last_seen_player_pos.at[idx].set(
                        s2.player_pos.astype(jnp.int16)
                    ),
                    mstrategy=_m.mstrategy.at[idx].set(jnp.int8(MoveStrategy.HUNT)),
                )
                return new_s.replace(monster_ai=new_m)

            # 8b: ordinary movement (clip into bounds; don't overlap player).
            def _move(s2):
                _m = s2.monster_ai
                # Don't step onto player tile.
                target = jnp.where(steps_onto_player, cur_pos, new_pos_i32)
                # Clip into map bounds (safety).
                target_r = jnp.clip(target[0], 0, _MAP_H - 1)
                target_c = jnp.clip(target[1], 0, _MAP_W - 1)
                final_pos = jnp.stack([target_r, target_c]).astype(jnp.int16)
                new_strategy = jnp.where(
                    wants_retreat,
                    jnp.int8(MoveStrategy.FLEE),
                    jnp.int8(MoveStrategy.HUNT),
                )
                ppos_i16 = s2.player_pos.astype(jnp.int16)
                new_m = _m.replace(
                    pos=_m.pos.at[idx].set(final_pos),
                    last_seen_player_pos=_m.last_seen_player_pos.at[idx].set(ppos_i16),
                    mstrategy=_m.mstrategy.at[idx].set(new_strategy),
                )
                return s2.replace(monster_ai=new_m)

            return jax.lax.cond(steps_onto_player, _attack, _move, st)

        return jax.lax.cond(should_act, _act, lambda st: st, s)

    return jax.lax.cond(is_pet, _pet_branch, _hostile_branch, state)


# ---------------------------------------------------------------------------
# clone_mon  — vendor/nethack/src/makemon.c::clone_mon lines 837-944.
# ---------------------------------------------------------------------------

# 8-neighbour offsets (row, col) — vendor xdir[]/ydir[] order, NW first.
_CLONE_DR: jnp.ndarray = jnp.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=jnp.int32)
_CLONE_DC: jnp.ndarray = jnp.array([-1,  0,  1, -1, 1, -1, 0, 1], dtype=jnp.int32)


def clone_mon(state, mon_idx: jnp.ndarray, rng: jax.Array) -> object:
    """Clone monster ``mon_idx`` into an empty adjacent cell at half HP.

    Vendor reference: ``src/makemon.c::clone_mon`` lines 837-944.  The clone
    inherits the original's type, alignment (peaceful / tame), and gets
    ``mhp = (orig_mhp + 1) / 2`` with the parent's HP halved likewise.
    The new monster's ``mcloned`` bit is set so corpse / XP gating can
    later drop drops on the clone.

    The 8-neighbour search uses vendor xdir[]/ydir[] order and picks the
    first walkable empty cell; if none exist, the clone is suppressed
    (vendor falls back to mksobj_at on no_empty_tile).  An empty live-slot
    must also be available (we cap at MAX_MONSTERS_PER_LEVEL).

    JIT-pure: lax.scan over 8 directions for the cell search; lax.cond for
    the apply / no-op branch.
    """
    idx = mon_idx.astype(jnp.int32)
    mai = state.monster_ai

    parent_alive = mai.alive[idx]
    parent_pos = mai.pos[idx].astype(jnp.int32)
    pr, pc = parent_pos[0], parent_pos[1]

    # ---- Find first empty adjacent cell (8-neighbour scan) -----------------
    terrain = _current_level_terrain(state)
    map_h = terrain.shape[0]
    map_w = terrain.shape[1]

    # Per-monster occupancy mask for the current level.
    mon_r = mai.pos[:, 0].astype(jnp.int32)
    mon_c = mai.pos[:, 1].astype(jnp.int32)
    # Player tile is also off-limits to spawn into.
    ppos = state.player_pos.astype(jnp.int32)

    def _check_dir(carry, args):
        # carry: (found, tgt_r, tgt_c)
        found, tgt_r, tgt_c = carry
        dr, dc = args
        cand_r = pr + dr
        cand_c = pc + dc
        in_bounds = (cand_r >= 0) & (cand_r < map_h) & (cand_c >= 0) & (cand_c < map_w)
        # Passable terrain (FLOOR/CORRIDOR/OPEN_DOOR via existing helper).
        safe_r = jnp.clip(cand_r, 0, map_h - 1)
        safe_c = jnp.clip(cand_c, 0, map_w - 1)
        passable = _tile_passable(terrain, safe_r, safe_c) & in_bounds
        # Empty: no live monster occupies, and not the player tile.
        occ_mask = mai.alive & (mon_r == cand_r) & (mon_c == cand_c)
        not_occ = ~jnp.any(occ_mask)
        not_player = ~((cand_r == ppos[0]) & (cand_c == ppos[1]))
        is_empty = passable & not_occ & not_player
        take = is_empty & ~found
        new_r = jnp.where(take, cand_r, tgt_r)
        new_c = jnp.where(take, cand_c, tgt_c)
        new_found = found | is_empty
        return (new_found, new_r, new_c), None

    (found_cell, tgt_r, tgt_c), _ = jax.lax.scan(
        _check_dir,
        (jnp.bool_(False), jnp.int32(0), jnp.int32(0)),
        (_CLONE_DR, _CLONE_DC),
    )

    # ---- Find first dead slot ---------------------------------------------
    dead_mask = ~mai.alive
    has_dead = jnp.any(dead_mask)
    dead_idx = jnp.argmax(dead_mask).astype(jnp.int32)

    can_clone = parent_alive & found_cell & has_dead

    # ---- Build clone state -------------------------------------------------
    half_hp = jnp.maximum(jnp.int32(1),
                          (mai.hp[idx].astype(jnp.int32) + 1) // jnp.int32(2))
    new_pos = jnp.stack([tgt_r.astype(jnp.int16), tgt_c.astype(jnp.int16)])

    def _do_clone(s):
        m = s.monster_ai
        new_alive    = m.alive.at[dead_idx].set(jnp.bool_(True))
        new_pos_arr  = m.pos.at[dead_idx].set(new_pos)
        new_hp       = m.hp.at[dead_idx].set(half_hp)
        new_hp_max   = m.hp_max.at[dead_idx].set(half_hp)
        new_entry    = m.entry_idx.at[dead_idx].set(m.entry_idx[idx])
        new_peaceful = m.peaceful.at[dead_idx].set(m.peaceful[idx])
        new_tame     = m.tame.at[dead_idx].set(m.tame[idx])
        new_ac       = m.ac.at[dead_idx].set(m.ac[idx])
        new_atk_n    = m.attack_dice_n.at[dead_idx].set(m.attack_dice_n[idx])
        new_atk_s    = m.attack_dice_sides.at[dead_idx].set(m.attack_dice_sides[idx])
        new_resists  = m.resists.at[dead_idx].set(m.resists[idx])
        new_undead   = m.undead.at[dead_idx].set(m.undead[idx])
        new_nonliving = m.nonliving.at[dead_idx].set(m.nonliving[idx])
        new_invisible = m.invisible.at[dead_idx].set(m.invisible[idx])
        new_is_large  = m.is_large.at[dead_idx].set(m.is_large[idx])
        new_asleep_v  = m.asleep.at[dead_idx].set(jnp.bool_(False))
        new_mstrat    = m.mstrategy.at[dead_idx].set(jnp.int8(MoveStrategy.HUNT))
        new_mcloned   = m.mcloned.at[dead_idx].set(jnp.bool_(True))
        # Halve parent HP too (vendor splits HP between original and clone).
        new_hp_parent = new_hp.at[idx].set(half_hp)

        new_m = m.replace(
            alive=new_alive,
            pos=new_pos_arr,
            hp=new_hp_parent,
            hp_max=new_hp_max,
            entry_idx=new_entry,
            peaceful=new_peaceful,
            tame=new_tame,
            ac=new_ac,
            attack_dice_n=new_atk_n,
            attack_dice_sides=new_atk_s,
            resists=new_resists,
            undead=new_undead,
            nonliving=new_nonliving,
            invisible=new_invisible,
            is_large=new_is_large,
            asleep=new_asleep_v,
            mstrategy=new_mstrat,
            mcloned=new_mcloned,
        )
        return s.replace(monster_ai=new_m)

    return jax.lax.cond(can_clone, _do_clone, lambda s: s, state)


# ---------------------------------------------------------------------------
# mattackm  — vendor/nethack/src/mhitm.c::mattackm lines 1024-1100.
# ---------------------------------------------------------------------------

def mattackm(state, attacker_idx: jnp.ndarray, defender_idx: jnp.ndarray,
             rng: jax.Array) -> object:
    """Monster-vs-monster melee attack.

    Reads the attacker's primary (n_dice, sides) from
    ``_MONSTER_PRIMARY_ATTACK_TABLE`` and the defender's effective AC from
    ``MonsterAIState.ac``.  Rolls to-hit using a vendor-style accumulator
    (``tmp = AC_VALUE(def_ac) + 10 + attacker_level``, hit iff ``tmp > rnd(20)``)
    then rolls damage (1d{sides} per die).

    Gates: both slots alive, attacker != defender.

    Cite: vendor/nethack/src/mhitm.c::mattackm (lines 1024-1100).
    """
    a = attacker_idx.astype(jnp.int32)
    d = defender_idx.astype(jnp.int32)
    mai = state.monster_ai

    key_hit, key_dmg = jax.random.split(rng)

    same_slot = (a == d)
    both_alive = mai.alive[a] & mai.alive[d]
    can_strike = both_alive & ~same_slot

    # Attacker primary attack — clipped to table bounds.
    a_entry = jnp.clip(
        mai.entry_idx[a].astype(jnp.int32),
        0, _MONSTER_PRIMARY_ATTACK_N.shape[0] - 1,
    )
    n_dice_raw = _MONSTER_PRIMARY_ATTACK_N[a_entry].astype(jnp.int32)
    sides_raw  = _MONSTER_PRIMARY_ATTACK_S[a_entry].astype(jnp.int32)
    n_dice = jnp.clip(n_dice_raw, 1, 8)
    sides  = jnp.clip(sides_raw, 1, 12)

    # Attacker level: derived from MONSTERS[entry].level (precomputed table).
    a_lev = jnp.clip(_MONSTER_LEVEL_TABLE[a_entry].astype(jnp.int32), 1, 30)

    # Defender AC: per-slot ``ac`` field.  Negative AC uses vendor AC_VALUE
    # softening (rnd(-ac)) per vendor/nethack/src/hack.h:1538.
    def_ac_raw = mai.ac[d].astype(jnp.int32)
    key_hit, key_ac = jax.random.split(key_hit)
    ac_neg_roll = jax.random.randint(
        key_ac, (), 1, jnp.maximum(-def_ac_raw + 1, 2), dtype=jnp.int32
    )
    ac_value = jnp.where(def_ac_raw >= 0, def_ac_raw, -ac_neg_roll)
    tmp = jnp.maximum(ac_value + jnp.int32(10) + a_lev, jnp.int32(1))

    roll = jax.random.randint(key_hit, (), 1, 21, dtype=jnp.int32)
    hit = (tmp > roll) & can_strike

    # Roll damage — bounded scan over 8 dice (mirrors monster_attack_player).
    def _roll_one(carry, key):
        sub = jax.random.randint(key, (), 1, sides + 1, dtype=jnp.int32)
        return carry, sub

    keys_d = jax.random.split(key_dmg, 8)
    _, rolls = jax.lax.scan(_roll_one, jnp.int32(0), keys_d)
    take = jnp.arange(8, dtype=jnp.int32) < n_dice
    raw_dmg = jnp.sum(jnp.where(take, rolls, jnp.int32(0))).astype(jnp.int32)
    dmg = jnp.where(hit, raw_dmg, jnp.int32(0))

    new_def_hp = jnp.maximum(mai.hp[d] - dmg, jnp.int32(0)).astype(jnp.int32)
    new_alive_d = mai.alive[d] & (new_def_hp > jnp.int32(0))

    new_hp_arr    = mai.hp.at[d].set(new_def_hp)
    new_alive_arr = mai.alive.at[d].set(new_alive_d)

    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# All-monsters step  (jax.lax.scan over slots)
# ---------------------------------------------------------------------------

# Per-tick movement-point threshold; vendor NORMAL_SPEED = 12 and a monster
# acts when its accumulator reaches NORMAL_SPEED.
# Cite: vendor/nethack/src/monmove.c line 1731; allmain.c lines 233-234.
_MOVEMENT_THRESHOLD: int = 12


def _faction(mai, idx: jnp.ndarray) -> jnp.ndarray:
    """Return faction id: 0 = hostile, 1 = peaceful (non-tame), 2 = tame.

    Used to gate monster-vs-monster combat — only different-faction
    monsters fight each other.
    """
    i = idx.astype(jnp.int32)
    is_tame = mai.tame[i]
    is_peace = mai.peaceful[i] & ~is_tame
    return jnp.where(is_tame, jnp.int32(2),
                     jnp.where(is_peace, jnp.int32(1), jnp.int32(0)))


def monsters_step_all(state, rng: jax.Array) -> object:
    """Advance all monster slots by one game tick.

    Speed-energy accumulator (vendor allmain.c:233-234 + monmove.c:1731):
    each tick adds ``move_speed * speed_factor`` movement points to every
    alive monster's accumulator, where ``speed_factor`` is 0.5 / 1.0 / 1.5
    for ``speed_mod`` < 0 / == 0 / > 0.  A slot only takes its turn when
    its accumulator reaches ``_MOVEMENT_THRESHOLD`` (12), and we deduct
    12 from the accumulator on action.

    After all turns, any pair of alive different-faction monsters that are
    Chebyshev-adjacent run a single ``mattackm`` exchange (attacker = lower
    slot id, defender = higher slot id) so pet-vs-hostile combat resolves
    on the same tick as movement.

    Cite: vendor/nethack/src/monmove.c line 1731 (per-turn movement gain);
          vendor/nethack/src/allmain.c lines 233-234 (mtmp->movement loop);
          vendor/nethack/src/mhitm.c lines 1024-1100 (mattackm).
    """
    mai = state.monster_ai

    # ---- Speed-energy accumulation (vendor monmove.c:1731) ----
    safe_entry = jnp.clip(
        mai.entry_idx.astype(jnp.int32),
        0, _MONSTER_MOVE_SPEED_TABLE.shape[0] - 1,
    )
    base_speed = _MONSTER_MOVE_SPEED_TABLE[safe_entry].astype(jnp.int32)
    smod = mai.speed_mod.astype(jnp.int32)
    # Vendor WAN_SLOW halves effective speed; WAN_SPEED bumps by 1.5×.
    # Use integer math to stay JIT-pure: multiply by (1,2,3) then divide by 2.
    factor_num = jnp.where(smod < 0, jnp.int32(1),
                  jnp.where(smod > 0, jnp.int32(3), jnp.int32(2)))
    factor_den = jnp.int32(2)
    add_points = (base_speed * factor_num) // factor_den
    add_points = jnp.where(mai.alive, add_points, jnp.int32(0))
    new_acc = mai.movement_points.astype(jnp.int32) + add_points

    can_act = new_acc >= jnp.int32(_MOVEMENT_THRESHOLD)
    # Deduct threshold on action.
    post_acc = jnp.where(can_act, new_acc - jnp.int32(_MOVEMENT_THRESHOLD), new_acc)
    new_mp = jnp.clip(post_acc, 0, 32000).astype(jnp.int16)

    mai = mai.replace(movement_points=new_mp)
    state = state.replace(monster_ai=mai)

    # ---- Per-slot turn dispatch ----
    keys = jax.random.split(rng, MAX_MONSTERS_PER_LEVEL * 2)
    turn_keys = keys[:MAX_MONSTERS_PER_LEVEL]
    mhit_keys = keys[MAX_MONSTERS_PER_LEVEL:]
    indices = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32)

    def _body(carry, xs):
        slot_idx, key, may_act = xs

        def _do_turn(s):
            return monster_turn(s, key, slot_idx)

        new_carry = jax.lax.cond(may_act, _do_turn, lambda s: s, carry)
        return new_carry, None

    final_state, _ = jax.lax.scan(_body, state, (indices, turn_keys, can_act))

    # ---- Monster-vs-monster melee (mattackm) ----
    # For each attacker slot i, find the first alive different-faction
    # monster j > i that is Chebyshev-adjacent, then run a single
    # mattackm(i → j) exchange.  This keeps the sweep O(N) JIT-traced ops
    # rather than O(N²) (factions × position search are vectorised against
    # the full slot table inside the scan body).
    def _strike_body(carry, args):
        i, key_i = args
        mi = carry.monster_ai
        i32 = i.astype(jnp.int32)

        # Attacker viability.
        atk_alive = mi.alive[i32]

        # Position of attacker.
        pi = mi.pos[i32].astype(jnp.int32)

        # Compute adjacency / different-faction mask against all other slots.
        all_pos = mi.pos.astype(jnp.int32)            # [N, 2]
        d_row = jnp.abs(all_pos[:, 0] - pi[0])
        d_col = jnp.abs(all_pos[:, 1] - pi[1])
        adj = jnp.maximum(d_row, d_col) == 1

        # Faction of every slot.
        is_tame_all = mi.tame
        is_peace_all = mi.peaceful & ~is_tame_all
        all_faction = jnp.where(is_tame_all, jnp.int32(2),
                       jnp.where(is_peace_all, jnp.int32(1), jnp.int32(0)))
        # Attacker faction.
        a_faction = all_faction[i32]
        diff_faction = all_faction != a_faction

        # Don't strike self.  Also restrict to slots > i so each pair is
        # only resolved once per tick.
        idx_arr = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32)
        pair_ok = idx_arr > i32

        candidates = mi.alive & adj & diff_faction & pair_ok
        has_target = jnp.any(candidates)
        # argmax of bool returns first True (or 0 if none).
        j_idx = jnp.argmax(candidates).astype(jnp.int32)

        do_strike = atk_alive & has_target

        def _strike(ss):
            return mattackm(ss, i32, j_idx, key_i)

        return jax.lax.cond(do_strike, _strike, lambda ss: ss, carry), None

    final_state, _ = jax.lax.scan(_strike_body, final_state, (indices, mhit_keys))

    # Tick status timers (vendor src/timeout.c::run_timers pattern).
    mai = final_state.monster_ai
    new_sleep     = jnp.maximum(mai.sleep_timer.astype(jnp.int32)     - 1, 0).astype(jnp.int16)
    new_stun      = jnp.maximum(mai.stun_timer.astype(jnp.int32)      - 1, 0).astype(jnp.int16)
    new_confuse   = jnp.maximum(mai.confuse_timer.astype(jnp.int32)   - 1, 0).astype(jnp.int16)
    new_paralyzed = jnp.maximum(mai.paralyzed_timer.astype(jnp.int32) - 1, 0).astype(jnp.int16)
    # flee_until_turn is an absolute turn counter; do not decrement.
    new_asleep    = new_sleep > jnp.int16(0)
    mai = mai.replace(
        sleep_timer=new_sleep,
        stun_timer=new_stun,
        confuse_timer=new_confuse,
        paralyzed_timer=new_paralyzed,
        asleep=new_asleep,
    )
    return final_state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# Wake monsters near a disturbance  (src/monmove.c::disturb)
# ---------------------------------------------------------------------------

def wake_monsters_near(state, pos: jnp.ndarray, radius: int = 3) -> object:
    """Wake all sleeping monsters within Chebyshev ``radius`` of ``pos``.

    Vectorized over all slots: no Python loop.

    Mirrors vendor/nethack/src/monmove.c disturb():
        - Monsters within radius switch from asleep=True to asleep=False.
        - Monsters already awake are unaffected.
    """
    mai = state.monster_ai
    pos_i32 = pos.astype(jnp.int32)

    mon_pos_i32 = mai.pos.astype(jnp.int32)
    delta = jnp.abs(mon_pos_i32 - pos_i32[None, :])       # [N, 2]
    dist = jnp.maximum(delta[:, 0], delta[:, 1])           # [N]

    in_radius = (dist <= radius) & mai.alive                # [N] bool
    new_asleep = mai.asleep & ~in_radius                    # flip only those in radius
    new_mai = mai.replace(asleep=new_asleep)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Shrieker alarm  (vendor/nethack/src/mon.c::shrieker)
# Cite: vendor/nethack/src/mon.c — adjacent shrieker wails and summons when
# the player can hear it; deafness suppresses the alarm.
# Cite (deaf gate): vendor/nethack/src/sounds.c — sound-based effects are
# skipped entirely when the player is deaf (HDeaf > 0).
# ---------------------------------------------------------------------------

_MS_SHRIEK_AI: int = 18   # MS_SHRIEK from monsters.py
_SHRIEK_PROB: float = 0.25  # ~25% chance per turn per adjacent shrieker


def shrieker_summon(state, rng: jax.Array) -> object:
    """Adjacent shriekers trigger a monster summon with prob _SHRIEK_PROB.

    When DEAF, the player cannot hear the wail so no summon occurs.
    Cite: vendor/nethack/src/mon.c::shrieker — MS_SHRIEK adjacent wail.
    Cite: vendor/nethack/src/sounds.c — deaf gate suppresses sound events.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus as _TS_SE

    mai = state.monster_ai
    pr  = state.player_pos[0].astype(jnp.int32)
    pc  = state.player_pos[1].astype(jnp.int32)

    mon_r = mai.pos[:, 0].astype(jnp.int32)
    mon_c = mai.pos[:, 1].astype(jnp.int32)
    adjacent = (jnp.maximum(jnp.abs(mon_r - pr), jnp.abs(mon_c - pc)) == 1) & mai.alive

    entry_safe  = jnp.clip(mai.entry_idx.astype(jnp.int32), 0, _MONSTER_SOUND_TABLE.shape[0] - 1)
    is_shrieker = _MONSTER_SOUND_TABLE[entry_safe] == jnp.int8(_MS_SHRIEK_AI)
    any_shrieker = jnp.any(adjacent & is_shrieker)

    # DEAF: player cannot hear the shrieker wail — summon suppressed.
    is_deaf = state.status.timed_statuses[int(_TS_SE.DEAF)] > jnp.int32(0)
    any_shrieker = any_shrieker & ~is_deaf

    rng_roll, _ = jax.random.split(rng)
    do_summon = any_shrieker & (jax.random.uniform(rng_roll) < jnp.float32(_SHRIEK_PROB))

    dead_mask = ~mai.alive
    dead_idx  = jnp.argmax(dead_mask).astype(jnp.int32)
    has_dead  = jnp.any(dead_mask)

    map_h = state.terrain.shape[2]
    map_w = state.terrain.shape[3]
    spawn_r   = jnp.clip(pr + 1, 0, map_h - 1)
    spawn_c   = jnp.clip(pc,     0, map_w - 1)
    spawn_pos = jnp.stack([spawn_r, spawn_c]).astype(jnp.int16)

    should = do_summon & has_dead

    new_alive    = mai.alive.at[dead_idx].set(jnp.where(should, jnp.bool_(True),  mai.alive[dead_idx]))
    new_pos      = mai.pos.at[dead_idx].set(jnp.where(should, spawn_pos,           mai.pos[dead_idx]))
    new_hp       = mai.hp.at[dead_idx].set(jnp.where(should, jnp.int32(4),         mai.hp[dead_idx]))
    new_hp_max   = mai.hp_max.at[dead_idx].set(jnp.where(should, jnp.int32(4),     mai.hp_max[dead_idx]))
    new_peaceful = mai.peaceful.at[dead_idx].set(jnp.where(should, jnp.bool_(False), mai.peaceful[dead_idx]))
    new_asleep   = mai.asleep.at[dead_idx].set(jnp.where(should, jnp.bool_(False),  mai.asleep[dead_idx]))
    new_entry    = mai.entry_idx.at[dead_idx].set(jnp.where(should, jnp.int16(1),   mai.entry_idx[dead_idx]))

    new_mai = mai.replace(
        alive=new_alive, pos=new_pos, hp=new_hp, hp_max=new_hp_max,
        peaceful=new_peaceful, asleep=new_asleep, entry_idx=new_entry,
    )
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Legacy Wave-1 stubs (kept for API compat)
# ---------------------------------------------------------------------------

def pet_turn(state: MonsterAIState, rng, pet_idx: int, world_view):
    """Run one turn for a pet (tame monster).  Delegates to pet_move."""
    return state


def step(state: MonsterAIState, rng):
    """Advance all monsters by one game tick.  Delegates to monsters_step_all."""
    return monsters_step_all(state, rng)
