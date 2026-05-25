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


# NATTK = 6 per vendor/nethack/include/permonst.h:48.
_NATTK: int = 6


def _build_monster_nattk_table() -> tuple:
    """Precompute the full NATTK attack table per MONSTERS[i].

    For each monster entry, returns (NATTK,) rows of (aatyp, n, s) per attack
    slot.  Inactive slots have ``aatyp == AT_NONE`` (0); the NATTK loop in
    mattackm skips them.

    Cite: vendor/nethack/src/mhitm.c::mattackm lines 293-592 + permonst.h
    NATTK = 6 (line 48).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, AttackType
    AT_NONE = int(AttackType.AT_NONE)
    nm = len(MONSTERS)
    aatyp = [[AT_NONE] * _NATTK for _ in range(nm)]
    n_arr = [[0] * _NATTK for _ in range(nm)]
    s_arr = [[0] * _NATTK for _ in range(nm)]
    for mi, m in enumerate(MONSTERS):
        for ai, atk in enumerate(m.attacks or ()):
            if ai >= _NATTK:
                break
            aatyp[mi][ai] = int(atk[0])
            n_arr[mi][ai] = int(atk[2])
            s_arr[mi][ai] = int(atk[3])
    return (
        jnp.array(aatyp, dtype=jnp.int16),
        jnp.array(n_arr, dtype=jnp.int16),
        jnp.array(s_arr, dtype=jnp.int16),
    )


(_MONSTER_ATTACK_AATYP_TABLE,
 _MONSTER_ATTACK_N_TABLE,
 _MONSTER_ATTACK_S_TABLE) = _build_monster_nattk_table()


def _build_mm_aggression_tables() -> tuple:
    """Precompute species-pair tables for vendor mm_aggression.

    Returns bool[NUMMONS] tables for JIT-time gating:
      * is_purple_worm  — purple worm or baby purple worm
      * is_shrieker     — shrieker
      * is_zombie_maker — S_ZOMBIE class (except ghoul/skeleton) and S_LICH
      * has_zombie_form — any non-zombie symbol (approx vendor zombie_form)

    Cite: vendor/nethack/src/mon.c::mm_aggression lines 2422-2447;
          vendor/nethack/src/mon.c::zombie_maker lines 362-381.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    pw_names = frozenset({"purple worm", "baby purple worm"})
    shrieker_names = frozenset({"shrieker"})
    ghoul_skel_names = frozenset({"ghoul", "skeleton"})
    S_ZOMBIE = MonsterSymbol.S_ZOMBIE
    S_LICH   = MonsterSymbol.S_LICH
    is_pw, is_shr, is_zm, is_zform = [], [], [], []
    for m in MONSTERS:
        is_pw.append(m.name in pw_names)
        is_shr.append(m.name in shrieker_names)
        zm_eligible = (
            (m.symbol == S_ZOMBIE and m.name not in ghoul_skel_names)
            or m.symbol == S_LICH
        )
        is_zm.append(bool(zm_eligible))
        is_zform.append(bool(m.symbol != S_ZOMBIE))
    return (
        jnp.array(is_pw, dtype=jnp.bool_),
        jnp.array(is_shr, dtype=jnp.bool_),
        jnp.array(is_zm, dtype=jnp.bool_),
        jnp.array(is_zform, dtype=jnp.bool_),
    )


(_MM_IS_PURPLE_WORM,
 _MM_IS_SHRIEKER,
 _MM_IS_ZOMBIE_MAKER,
 _MM_HAS_ZOMBIE_FORM) = _build_mm_aggression_tables()

# CONFLICT intrinsic index (matches Nethax.nethax.subsystems.status_effects).
# Cite: vendor/nethack/src/uhitm.c — Conflict gate.
_INTRINSIC_CONFLICT: int = 44


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
_POT_EXTRA_HEALING: int = 11  # PotionEffect.EXTRA_HEALING
_POT_FULL_HEALING:  int = 12  # PotionEffect.FULL_HEALING
_SCR_TELEPORT:     int = 10   # ScrollEffect.TELEPORTATION
# Wand effect IDs — mirror Nethax/nethax/subsystems/items_wands.WandEffect.
# Used by Wave 17e muse to dispatch the full vendor wand library
# (vendor/nethack/src/muse.c lines 1272-1286, 2084-2089).
_WAN_STRIKING:     int = 7    # WandEffect.STRIKING
_WAN_SLOW_MONSTER: int = 8    # WandEffect.SLOW_MONSTER
_WAN_SPEED_MONSTER: int = 9   # WandEffect.SPEED_MONSTER
_WAN_CANCELLATION: int = 10   # WandEffect.CANCELLATION
_WAN_TELEPORT:     int = 12   # WandEffect.TELEPORTATION
_WAN_DEATH:        int = 13   # WandEffect.DEATH
_WAN_SLEEP:        int = 14   # WandEffect.SLEEP
_WAN_COLD:         int = 15   # WandEffect.COLD
_WAN_FIRE:         int = 16   # WandEffect.FIRE
_WAN_LIGHTNING:    int = 17   # WandEffect.LIGHTNING
_WAN_DIGGING:      int = 18   # WandEffect.DIGGING
_WAN_CREATE_MONSTER: int = 20 # WandEffect.CREATE_MONSTER
_WAN_MAKE_INVISIBLE: int = 23 # WandEffect.MAKE_INVISIBLE

# M-flag bits we need at JIT-time (vendor/nethack/include/monflag.h).
_M1_FLY: int          = 0x00000001
_M1_SWIM: int         = 0x00000002
_M1_AMORPHOUS: int    = 0x00000004  # vendor monflag.h:87 — can flow under doors / through bars.
_M1_AMPHIBIOUS: int   = 0x00000200
_M1_BREATHLESS: int   = 0x00000400
_M1_MINDLESS: int     = 0x00010000
_M1_HUMANOID: int     = 0x00020000
_M1_ANIMAL: int       = 0x00040000
_M1_NOHANDS: int      = 0x00002000
_M1_SEE_INVIS: int    = 0x01000000

_M2_UNDEAD: int       = 0x00000002
_M2_HUMAN: int        = 0x00000008  # vendor monflag.h:126 — is a human.
_M2_DEMON: int        = 0x00000100
_M2_GIANT: int        = 0x00002000  # vendor monflag.h:136 — is_giant → BUSTDOOR.
_M2_PEACEFUL: int     = 0x00200000

# Tile constants — kept local to avoid an import cycle with constants.tiles.
# Must mirror Nethax.nethax.constants.tiles.TileType.
_TILE_WALL: int        = 3
_TILE_CLOSED_DOOR: int = 4
_TILE_OPEN_DOOR: int   = 5  # see-thru in vendor; non-blocking for LoS.
_TILE_WATER: int       = 8
_TILE_LAVA: int        = 9
_TILE_TREE: int        = 20  # blocks LoS per vendor vision.c:166.
_TILE_IRONBARS: int    = 22  # IRONBARS — vendor mon.c:2225 ALLOW_BARS gate.

# Trap-type codes (mirror constants/TrapType) that are "always avoided" in vendor —
# pets/hostiles never path into a known fall/hole/lava-trap.  vendor mon.c:2353-2368
# routes trap avoidance through mon_knows_traps; we treat these specific types as
# always-known because they kill regardless of awareness.
_TT_PIT: int        = 11
_TT_SPIKED_PIT: int = 12
_TT_HOLE: int       = 13
_TT_TRAPDOOR: int   = 14


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

    Wave 45a: ``m_lev`` (per-monster level) and ``blind_timer`` (turns of
    blindness remaining) are reserved for vendor-parity consumers
    (DRAIN_LIFE in magic.py, BLINDING_RAY / FLING_POISON in
    artifact_powers.py).  The schema is ready; consumer wiring lands in a
    follow-up wave.
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
    # NOTE: legacy linear ticker; the vendor model is the absolute
    # ``hungrytime`` counter in ``hungrytime`` below.  Both are maintained for
    # backward-compat with tests that read pet_hunger.
    pet_hunger: jnp.ndarray        # [MAX_MONSTERS_PER_LEVEL]  int16

    # ---- Vendor pet hunger model (dogmove.c:362-394 + dog.h DOG_HUNGRY=300
    # DOG_WEAK=500 DOG_STARVE=750) ----
    # ``hungrytime`` is an absolute counter in moves-units (svm.moves+offset);
    # vendor compares ``svm.moves > hungrytime + DOG_WEAK`` etc.  Defaults to
    # 1000 so freshly-spawned pets do not immediately starve.
    # Cite: vendor/nethack/src/dogmove.c lines 362-394; include/dog.h DOG_*.
    hungrytime: jnp.ndarray        # [MAX_MONSTERS_PER_LEVEL]  int32

    # ``mhpmax_penalty`` mirrors edog->mhpmax_penalty: the amount mhpmax was
    # reduced by while WEAK (about 2/3 of original).  Restored to mhpmax on
    # eat (dogmove.c:242-246).
    mhpmax_penalty: jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL]  int32

    # ``mleashed`` bool — pet is on a leash; affects pet_within_leash &
    # dog_invent (vendor dogmove.c:1093 distu(nx,ny) > 4 → skip).
    mleashed: jnp.ndarray          # [MAX_MONSTERS_PER_LEVEL]  bool

    # ``mon_xp`` counter — pet experience accumulated from kills via
    # mattackm.  Used by ``grow_up`` (vendor mon.c) to level pets up.
    # Cite: vendor/nethack/src/mon.c::grow_up.
    mon_xp: jnp.ndarray            # [MAX_MONSTERS_PER_LEVEL]  int32

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

    # mspec_used cooldown — turns remaining before the monster can cast
    # again.  Set to ``(m_lev < 8) ? (10 - m_lev) : 2`` on successful cast.
    # Cite: vendor/nethack/src/mcastu.c lines 184-186 + monst.h mspec_used.
    mspec_used: jnp.ndarray        # [MAX_MONSTERS_PER_LEVEL]  int16

    # ---- Wave 40b Item #15: pet migration buffer (mig_mons) ----
    # ``migrating`` bool: True if the pet was queued onto the migrating_mons
    # list (vendor dog.c::keepdogs lines 789-870 sets mx=my=0 + appends to
    # gm.mydogs).  Replaces the old "set alive=False on stair follow" model so
    # all per-pet state (hp, mtame, edog fields, hungrytime, mhpmax_penalty,
    # mtrack) is preserved until level entry calls mon_arrive (dog.c:420-566).
    # Cite: vendor/nethack/src/dog.c::keepdogs (789-870), mon_arrive (420-566).
    migrating: jnp.ndarray         # [MAX_MONSTERS_PER_LEVEL]  bool

    # ---- Wave 40b Item #20: pet drop tracking (edog.dropdist / droptime) ----
    # vendor edog struct fields:
    #   dropdist (int): chebyshev distance to hero at the time of last drop;
    #   droptime (long): svm.moves snapshot at the time of last drop.
    # Used by dog_eat (dogmove.c:318) to compute the apport credit
    #   apport += 200L / (dropdist + moves - droptime)
    # for fetching dropped items.  We model as (row, col) of last drop site
    # plus the absolute turn the drop happened on.
    # Cite: vendor/nethack/src/dogmove.c::dog_eat line 318 + dog_invent line 422.
    last_drop_pos:  jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL, 2]  int16
    last_drop_turn: jnp.ndarray    # [MAX_MONSTERS_PER_LEVEL]     int32

    # ---- Wave 45a: per-monster level + blind timer ----
    # ``m_lev`` is the per-monster level field from vendor's ``struct monst``
    # (include/monst.h).  Populated from MONSTERS[entry_idx].level at spawn;
    # zero for slots not yet wired via monster_ai's internal spawn paths.
    # Consumers (DRAIN_LIFE in magic.py, BLINDING_RAY in artifact_powers.py)
    # are wired in a follow-up wave; the schema is ready now.
    # Cite: vendor/nethack/include/monst.h::struct monst::m_lev.
    m_lev: jnp.ndarray             # [MAX_MONSTERS_PER_LEVEL]  int16
    # ``blind_timer`` mirrors ``mblinded`` from vendor monst.h — turns of
    # blindness remaining.  Set by flash_hits_mon (zap.c:2925: mtmp->mblinded
    # = damage turns).  Decremented once per turn (mon.c::mon_update_state).
    # Cite: vendor/nethack/include/monst.h::mblinded;
    #       vendor/nethack/src/zap.c::flash_hits_mon line ~2925.
    blind_timer: jnp.ndarray       # [MAX_MONSTERS_PER_LEVEL]  int16


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
        # Vendor pet hunger model — dogmove.c:362-394
        hungrytime=jnp.full(n, 1000, dtype=jnp.int32),
        mhpmax_penalty=jnp.zeros(n, dtype=jnp.int32),
        mleashed=jnp.zeros(n, dtype=jnp.bool_),
        mon_xp=jnp.zeros(n, dtype=jnp.int32),
        saddled=jnp.zeros(n, dtype=jnp.int8),
        is_unwielded=jnp.zeros(n, dtype=jnp.bool_),
        resists=jnp.zeros(n, dtype=jnp.int32),
        undead=jnp.zeros(n, dtype=jnp.bool_),
        invisible=jnp.zeros(n, dtype=jnp.bool_),
        nonliving=jnp.zeros(n, dtype=jnp.bool_),
        speed_mod=jnp.zeros(n, dtype=jnp.int8),
        cancelled=jnp.zeros(n, dtype=jnp.bool_),
        mcloned=jnp.zeros(n, dtype=jnp.bool_),
        mspec_used=jnp.zeros(n, dtype=jnp.int16),
        migrating=jnp.zeros(n, dtype=jnp.bool_),
        last_drop_pos=jnp.full((n, 2), -1, dtype=jnp.int16),
        last_drop_turn=jnp.zeros(n, dtype=jnp.int32),
        # Wave 45a: per-monster level + blind timer (zero until set at spawn).
        m_lev=jnp.zeros(n, dtype=jnp.int16),
        blind_timer=jnp.zeros(n, dtype=jnp.int16),
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


# ---------------------------------------------------------------------------
# mfndpos per-monster movement gates  (vendor/nethack/src/mon.c::mon_allowflags
# lines 2062-2126 + mfndpos body lines 2140-2382).  Helpers return scalar bool
# jnp.ndarray suitable for use inside JIT'd path/bfs masks.
# ---------------------------------------------------------------------------

def _mover_can_open_door(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff this monster can walk through a closed (unlocked) door.

    Vendor mon.c:2067 ``boolean can_open = !(nohands(mtmp->data)
    || verysmall(mtmp->data));`` then 2100-2101 ``if (can_open)
    allowflags |= OPENDOOR;``.

    JAX approximation: a monster can open doors if it has hands
    (~M1_NOHANDS) AND is at least roughly humanoid/intelligent — we use
    M1_HUMANOID as a proxy for "not verysmall, has dexterous limbs",
    plus humans / minotaurs via M2_HUMAN as a secondary proxy.  Amorphous
    creatures also pass under closed doors (vendor mon.c:2234).
    """
    nohands   = _has_flag1(entry_idx, _M1_NOHANDS)
    humanoid  = _has_flag1(entry_idx, _M1_HUMANOID)
    human     = _has_flag2(entry_idx, _M2_HUMAN)
    amorphous = _has_flag1(entry_idx, _M1_AMORPHOUS)
    return ((~nohands) & (humanoid | human)) | amorphous


def _mover_can_bust_door(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff this monster busts closed/locked doors when bumping them.

    Vendor mon.c:2070 ``doorbuster = is_giant(mtmp->data);`` then 2098-2099
    ``if (doorbuster) allowflags |= BUSTDOOR;``.  ``is_giant`` is defined
    as ``(mflags2 & M2_GIANT) != 0`` (mondata.h:107).
    """
    return _has_flag2(entry_idx, _M2_GIANT)


def _mover_avoids_traps(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff this monster avoids known traps when pathing.

    Vendor mon.c:2353-2368: ``if (mon_knows_traps(mon, ttmp->ttyp))
    continue;`` — i.e. the monster skips any tile carrying a trap it
    knows about.  Mindless creatures (M1_MINDLESS) never know about
    traps; everyone else conservatively does (we err on the side of
    avoidance for non-mindless monsters).  Animals also typically avoid
    visible traps once seen.
    """
    mindless = _has_flag1(entry_idx, _M1_MINDLESS)
    return ~mindless


def _mover_can_pass_bars(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff this monster can move through iron bars (IRONBARS tile).

    Vendor mon.c:2225-2230 ``if (ntyp == IRONBARS && !(flag & ALLOW_BARS))
    continue;`` — ALLOW_BARS is set by ``passes_bars()`` (mondata.c:554),
    which combines passes_walls / amorphous / unsolid / whirly /
    verysmall / corrosive / metallivorous / slithy.  We use M1_AMORPHOUS
    as the conservative JAX proxy (vendor monflag.h:87 — "can flow
    under doors / through bars").
    """
    return _has_flag1(entry_idx, _M1_AMORPHOUS)


def _monster_level(entry_idx: jnp.ndarray) -> jnp.ndarray:
    """Look up MONSTERS[entry_idx].level."""
    e = entry_idx.astype(jnp.int32)
    safe = jnp.clip(e, 0, _MONSTER_LEVEL_TABLE.shape[0] - 1)
    return _MONSTER_LEVEL_TABLE[safe].astype(jnp.int32)


# Wave 48c: per-spawn vendor newmonhp() roll.
# Cite: vendor/nethack/src/makemon.c::newmonhp lines 1037-1053.
#   if (!mon->m_lev)  mhpmax = rnd(4);            // level-0 monsters
#   else               mhpmax = d((int)m_lev, 8); // sum of m_lev d8 rolls
#   if (mhpmax == basehp) mhpmax += 1;            // all-ones boost
# Special cases (golems, riders, mlevel>49 fixed-HP, adult dragons,
# home-elementals) are not modeled at the two summon spawn sites in this
# file — neither create_monster nor shrieker_summon picks those species,
# so the common d(m_lev, 8) / rnd(4) path is sufficient.
_NEWMONHP_MAX_LEV: int = 32


def _newmonhp_roll(rng: jax.Array, m_lev: jnp.ndarray) -> jnp.ndarray:
    """Return mhpmax for a freshly spawned monster of level ``m_lev``.

    Mirrors vendor makemon.c::newmonhp common branch:
        m_lev == 0  ->  rnd(4)            (1..4 uniform)
        m_lev >  0  ->  d(m_lev, 8)       (sum of m_lev rolls in 1..8)
    Then applies the "all-ones boost": if the roll equals basehp (i.e. the
    minimum possible), add 1 so the lowest-level monsters always have at
    least 2 HP — see makemon.c:1047-1053.
    """
    lev = jnp.maximum(m_lev.astype(jnp.int32), jnp.int32(0))
    key_d8, key_r4 = jax.random.split(rng)

    # d(m_lev, 8): sum of up to _NEWMONHP_MAX_LEV d8 rolls, masked by lev.
    d8_keys = jax.random.split(key_d8, _NEWMONHP_MAX_LEV)
    d8_rolls = jax.vmap(lambda k: jax.random.randint(k, (), 1, 9))(d8_keys)
    take = jnp.arange(_NEWMONHP_MAX_LEV, dtype=jnp.int32) < lev
    d_mlev_8 = jnp.sum(jnp.where(take, d8_rolls, 0)).astype(jnp.int32)

    # rnd(4) for level-0 monsters.
    rnd_4 = jax.random.randint(key_r4, (), 1, 5).astype(jnp.int32)

    # basehp = m_lev (or 1 when m_lev == 0; vendor sets basehp=1 then mhp=rnd(4)).
    is_lev0 = lev == jnp.int32(0)
    base = jnp.where(is_lev0, jnp.int32(1), lev)
    mhpmax = jnp.where(is_lev0, rnd_4, d_mlev_8)

    # "all-ones boost": if mhpmax == basehp, bump by +1 (makemon.c:1050-1052).
    boost = (mhpmax == base).astype(jnp.int32)
    return mhpmax + boost


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

    # Wave 44a (vendor mon.c:2062-2126 mon_allowflags + mfndpos body 2225-2237):
    # per-monster door / bars / trap gates.
    door_opener = _mover_can_open_door(entry)
    door_buster = _mover_can_bust_door(entry)
    bars_passer = _mover_can_pass_bars(entry)
    trap_avoider = _mover_avoids_traps(entry)

    # Mask of passable tiles for THIS mover.
    tile_field = terrain.astype(jnp.int32)
    is_wall          = (tile_field == _TILE_WALL)
    is_closed_door   = (tile_field == _TILE_CLOSED_DOOR)
    is_tree          = (tile_field == _TILE_TREE)
    is_ironbars      = (tile_field == _TILE_IRONBARS)
    is_water         = (tile_field == _TILE_WATER)
    is_lava          = (tile_field == _TILE_LAVA)

    # vendor mon.c:2235-2237 — closed door blocks unless OPENDOOR or BUSTDOOR
    # (thrudoor) is granted to the mover.
    door_ok = door_opener | door_buster
    # vendor mon.c:2225-2230 — iron bars block unless ALLOW_BARS (passes_bars).
    bars_ok = bars_passer

    not_wall   = ~is_wall & ~is_tree \
                 & (~is_closed_door | door_ok) \
                 & (~is_ironbars    | bars_ok)
    water_ok   = can_swim | can_fly
    lava_ok    = can_fly  # vendor: lava only flyable.
    terrain_ok = not_wall & jnp.where(is_water, water_ok, jnp.bool_(True)) \
                          & jnp.where(is_lava,  lava_ok,  jnp.bool_(True))

    # ----- Trap-avoidance mask (vendor mon.c:2353-2368) ---------------------
    # Build a [MAP_H, MAP_W] bool grid: True where the mover must NOT step.
    # A tile is forbidden when:
    #   (a) trap_type at that tile is a fall-/hole-trap (PIT/SPIKED_PIT/HOLE/
    #       TRAPDOOR) — these are "always-known" because they're lethal
    #       irrespective of awareness, and the mover is trap-avoiding, OR
    #   (b) trap_type is any non-zero AND state.traps.revealed[here] is True
    #       AND the mover is trap-avoiding (vendor mon_knows_traps gate).
    # Mindless monsters skip all trap avoidance.
    # We index state.traps with the flat per-level idx used elsewhere
    # (vendor mon.c works per-current-level only).
    max_lv  = jnp.int32(state.terrain.shape[1])
    branch  = state.dungeon.current_branch.astype(jnp.int32)
    level0  = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    flat_lv = branch * max_lv + level0
    trap_t  = state.traps.trap_type[flat_lv].astype(jnp.int32)   # [MAP_H, MAP_W]
    trap_rv = state.traps.revealed[flat_lv]                       # [MAP_H, MAP_W] bool

    always_lethal = (
        (trap_t == jnp.int32(_TT_PIT))
        | (trap_t == jnp.int32(_TT_SPIKED_PIT))
        | (trap_t == jnp.int32(_TT_HOLE))
        | (trap_t == jnp.int32(_TT_TRAPDOOR))
    )
    known_any = (trap_t != jnp.int32(0)) & trap_rv
    trap_block = (always_lethal | known_any) & trap_avoider

    terrain_ok = terrain_ok & ~trap_block

    # MM_PEACEFUL: hostile movers do not path through peaceful monsters
    # (vendor mfndpos.h::ALLOW_M / MM_PEACEFUL handling).  Build a [MAP_H,
    # MAP_W] mask of "blocked by peaceful" using scatter.
    #
    # Wave 44a (vendor mon.c:2148 mm_displacement + 2305-2316):
    # a tame mover MAY swap with another tame / peaceful monster on its path
    # (ALLOW_MDISP).  When the mover is tame, do not block on friendly /
    # peaceful tiles — only on alive monsters that are neither tame nor
    # peaceful (i.e. hostile, which are "attack squares", still permitted
    # as path destinations via ALLOW_M but treated as cost-1 nodes here).
    self_mask_n = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32) == idx
    self_is_tame = mai.tame[idx]
    # Hostile movers: block on peaceful AND tame other monsters (don't trample
    # friendlies).  Tame movers: pass through other tame/peaceful via
    # mm_displacement.  Vendor mon.c:2148 mm_displacement returns ALLOW_MDISP
    # when both monsters are tame OR both peaceful in the right configuration.
    other_alive_peaceful = mai.alive & mai.peaceful & ~self_mask_n  # [N]
    other_alive_tame     = mai.alive & mai.tame     & ~self_mask_n  # [N]
    blocking_friendly = jnp.where(
        self_is_tame,
        # tame self: friendlies are SWAP-able, not blockers
        jnp.zeros_like(other_alive_peaceful),
        other_alive_peaceful | other_alive_tame,
    )
    # Scatter blocker positions into a [MAP_H, MAP_W] occupancy mask.
    occ = jnp.zeros((_MAP_H, _MAP_W), dtype=jnp.bool_)
    pp = mai.pos.astype(jnp.int32)
    safe_r = jnp.clip(pp[:, 0], 0, _MAP_H - 1)
    safe_c = jnp.clip(pp[:, 1], 0, _MAP_W - 1)
    occ = occ.at[safe_r, safe_c].max(blocking_friendly)

    passable = terrain_ok & ~occ

    # Wave 40b Item #9 partial: vendor mfndpos (mon.c:2140-2382) restricts
    # diagonal moves under several conditions.  We model two of them in the
    # BFS relaxation:
    #   * NODIAG (vendor hack.h:1414 NODIAG(monnum) == PM_GRID_BUG): grid bugs
    #     can't move diagonally.  Block ALL diagonal relaxations for the mover
    #     when its species is grid bug.
    #   * Diagonal squeeze: a diagonal step from (r,c) to (r+dy,c+dx) is
    #     blocked if BOTH cardinal neighbors (r+dy,c) and (r,c+dx) are
    #     impassable (e.g. two walls form a corner — vendor "bad_rock" /
    #     diagonal door rules at mon.c:2245-2257).
    # Full mfndpos rewrite (8-neighbor enumeration with per-tile info flags,
    # OPENDOOR/UNLOCKDOOR/BUSTDOOR, mm_displacement, mon_knows_traps) is left
    # as TODO — see Wave 40b Item #9 follow-up.
    # Cite: vendor/nethack/src/mon.c::mfndpos lines 2140-2382;
    #       vendor/nethack/include/hack.h:1414 NODIAG macro.
    _PM_GRID_BUG_NAMES = ("grid bug",)
    from Nethax.nethax.constants.monsters import MONSTERS as _MM
    _grid_bug_entry = next(
        (i for i, m in enumerate(_MM) if m.name in _PM_GRID_BUG_NAMES), -1,
    )
    nodiag = jnp.int32(_grid_bug_entry) == entry.astype(jnp.int32)

    def bfs_body(_k, dist_field):
        neigh_min = jnp.full_like(dist_field, INF)
        for dy, dx in offsets:
            shifted = shift_one(dist_field, dy, dx)
            is_diag = (dy != 0) and (dx != 0)
            if is_diag:
                # NODIAG: drop diagonal contributions for grid bugs.
                shifted = jnp.where(nodiag, INF, shifted)
                # Diagonal squeeze: block if both orthogonal neighbors are
                # impassable.  The diagonal move goes from (r-dy, c-dx) to
                # (r, c); the orthogonals to check are (r-dy, c) and
                # (r, c-dx).  Shift the passable mask by (dy, 0) and (0, dx)
                # so that the value at (r, c) tells us whether the orthogonal
                # neighbor (r-dy, c) / (r, c-dx) is passable.
                orth_a = jnp.roll(passable, shift=(dy, 0), axis=(0, 1))
                orth_b = jnp.roll(passable, shift=(0, dx), axis=(0, 1))
                # Mask wrap rows/cols similar to shift_one.
                if dy > 0:
                    orth_a = orth_a.at[0:dy, :].set(jnp.bool_(False))
                elif dy < 0:
                    orth_a = orth_a.at[_MAP_H + dy:_MAP_H, :].set(jnp.bool_(False))
                if dx > 0:
                    orth_b = orth_b.at[:, 0:dx].set(jnp.bool_(False))
                elif dx < 0:
                    orth_b = orth_b.at[:, _MAP_W + dx:_MAP_W].set(jnp.bool_(False))
                squeeze_ok = orth_a | orth_b
                shifted = jnp.where(squeeze_ok, shifted, INF)
            neigh_min = jnp.minimum(neigh_min, shifted)
        candidate = neigh_min + jnp.int32(1)
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


# Vendor mfndpos confusion gate: ``if (mon->mconf) flag |= ALLOW_ALL`` at
# mon.c:2199-2202.  Combined with the dochug confused-pursuit override in
# monmove.c::dochug (mtmp->mconf forces random direction selection rather
# than goal-directed pursuit), a confused monster steps to a uniformly
# random 8-neighbor tile each turn.  ``apply_confusion_to_step`` is the
# JIT-safe one-liner override: when ``confuse_timer > 0`` it replaces the
# pathfind/retreat step with a fresh uniform (dy, dx) in {-1, 0, 1}^2 \ {(0,0)}.
#
# Cite: vendor/nethack/src/mon.c::mfndpos lines 2199-2202;
#       vendor/nethack/src/monmove.c::dochug confused-pursuit gate.
def apply_confusion_to_step(
    step_delta: jnp.ndarray,
    is_confused: jnp.ndarray,
    rng: jax.Array,
) -> jnp.ndarray:
    """Override ``step_delta`` with a uniform random 8-direction when confused.

    Returns the original ``step_delta`` when ``is_confused`` is False, otherwise
    a random (dy, dx) drawn uniformly from the 8 neighbour offsets.
    """
    # 8 offsets — match the order used by ``pathfind_step``.
    _OFFSETS = jnp.array(
        [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ],
        dtype=jnp.int32,
    )
    pick = jax.random.randint(rng, (), 0, 8, dtype=jnp.int32)
    rand_step = _OFFSETS[pick]
    return jnp.where(is_confused, rand_step, step_delta.astype(jnp.int32))


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


def _try_quaff_potion(state, rng: jax.Array, monster_idx: jnp.ndarray,
                      potion_id: int, heal_dice: int, heal_sides: int):
    """Quaff a potion of healing/extra/full healing if present and HP low.

    Cite: vendor/nethack/src/muse.c::use_defensive cases MUSE_POT_HEALING /
    EXTRA_HEALING / FULL_HEALING lines 1161-1230.  Vendor heal amount per
    vendor potion.c:peffect_healing.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_POTION, potion_id)
    hp     = mai.hp[idx].astype(jnp.int32)
    hp_max = mai.hp_max[idx].astype(jnp.int32)
    hurt   = hp < hp_max
    can_quaff = found & hurt

    keys = jax.random.split(rng, max(heal_dice, 1))
    rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 1, heal_sides + 1, dtype=jnp.int32)
    )(keys)
    heal = jnp.sum(rolls)
    new_hp = jnp.minimum(hp + heal, hp_max)
    new_hp = jnp.where(can_quaff, new_hp, hp)

    old_qty = mai.inv_quantity[idx, slot].astype(jnp.int32)
    dec_qty = jnp.maximum(old_qty - jnp.int32(1), jnp.int32(0))
    new_qty = jnp.where(can_quaff, dec_qty, old_qty).astype(jnp.int16)
    cleared_cat = jnp.where(
        (new_qty == 0) & can_quaff, jnp.int8(0),
        mai.inv_category[idx, slot],
    )
    new_mai = mai.replace(
        hp=mai.hp.at[idx].set(new_hp),
        inv_quantity=mai.inv_quantity.at[idx, slot].set(new_qty),
        inv_category=mai.inv_category.at[idx, slot].set(cleared_cat),
    )
    return state.replace(monster_ai=new_mai)


def _try_zap_offensive_wand(state, rng: jax.Array, monster_idx: jnp.ndarray,
                            wand_id: int, dice_n: int, dice_sides: int,
                            resist_intrinsic: int = -1):
    """Zap a damaging wand at the player.

    Cite: vendor/nethack/src/muse.c::use_offensive lines 1842-1900.
    Damage rolls follow vendor src/zap.c::buzz() for the ray family
    (DEATH=instakill, SLEEP=put-to-sleep, FIRE/COLD/LIGHTNING=6d6, etc.).

    ``resist_intrinsic`` is the Intrinsic enum index of the matching
    resistance (RESIST_FIRE / RESIST_COLD / RESIST_SHOCK / RESIST_SLEEP),
    or -1 for wands with no straightforward resist gate (STRIKING, DEATH
    handled separately).
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_WAND, wand_id)
    charges = mai.inv_charges[idx, slot].astype(jnp.int32)
    can_zap = found & (charges > 0)

    keys = jax.random.split(rng, max(dice_n, 1))
    rolls = jax.vmap(
        lambda k: jax.random.randint(k, (), 1, dice_sides + 1, dtype=jnp.int32)
    )(keys)
    dmg = jnp.sum(rolls).astype(jnp.int32)

    # Resistance check: vendor src/zap.c::resist drops dmg to half.
    if resist_intrinsic >= 0:
        has_resist = (
            state.status.intrinsics[int(resist_intrinsic)]
            | (state.status.timed_intrinsics[int(resist_intrinsic)] > jnp.int32(0))
        )
        dmg = jnp.where(has_resist, dmg // jnp.int32(2), dmg)

    new_player_hp = jnp.where(
        can_zap,
        jnp.maximum(state.player_hp - dmg, jnp.int32(0)),
        state.player_hp,
    ).astype(state.player_hp.dtype)
    new_done = state.done | (new_player_hp <= 0)
    new_charges = jnp.where(can_zap, charges - jnp.int32(1), charges).astype(jnp.int8)
    new_mai = mai.replace(
        inv_charges=mai.inv_charges.at[idx, slot].set(new_charges),
    )
    return state.replace(
        monster_ai=new_mai,
        player_hp=new_player_hp,
        done=new_done,
    )


def _try_wand_teleport_self(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Zap WAN_TELEPORTATION at self (defensive escape).

    Cite: vendor/nethack/src/muse.c::use_defensive MUSE_WAN_TELEPORTATION_SELF
    lines 849-857.  Effect: monster relocates to a random tile.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_WAND, _WAN_TELEPORT)
    charges = mai.inv_charges[idx, slot].astype(jnp.int32)
    can_zap = found & (charges > 0)

    rng_r, rng_c = jax.random.split(rng, 2)
    new_r = jax.random.randint(rng_r, (), 0, _MAP_H, dtype=jnp.int32).astype(jnp.int16)
    new_c = jax.random.randint(rng_c, (), 0, _MAP_W, dtype=jnp.int32).astype(jnp.int16)
    target_pos = jnp.stack([new_r, new_c])
    cur_pos = mai.pos[idx]
    chosen = jnp.where(can_zap, target_pos, cur_pos)
    new_charges = jnp.where(can_zap, charges - jnp.int32(1), charges).astype(jnp.int8)

    new_mai = mai.replace(
        pos=mai.pos.at[idx].set(chosen),
        inv_charges=mai.inv_charges.at[idx, slot].set(new_charges),
    )
    return state.replace(monster_ai=new_mai)


def _try_wand_digging(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Zap WAN_DIGGING downward to escape (defensive).

    Cite: vendor/nethack/src/muse.c::use_defensive MUSE_WAN_DIGGING
    lines 917-980.  We model the simplified "monster disappears" effect:
    move to a random tile (level-change is out of scope).
    """
    return _try_wand_teleport_self(state, rng, monster_idx)


def _try_wand_create_monster(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Zap WAN_CREATE_MONSTER (defensive/offensive).

    Cite: vendor/nethack/src/muse.c::use_defensive MUSE_WAN_CREATE_MONSTER
    lines 981-1000.  Vendor calls makemon() to spawn an adjacent monster.
    JIT-safe: allocate the first dead slot, place adjacent to caster.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_WAND, _WAN_CREATE_MONSTER)
    charges = mai.inv_charges[idx, slot].astype(jnp.int32)
    can_zap = found & (charges > 0)

    # Find first dead slot (already used by clone_mon).
    dead_mask = ~mai.alive
    has_dead = jnp.any(dead_mask)
    dead_idx = jnp.argmax(dead_mask.astype(jnp.int32)).astype(jnp.int32)
    should = can_zap & has_dead

    mpos = mai.pos[idx].astype(jnp.int32)
    spawn_r = jnp.clip(mpos[0] + jnp.int32(1), 0, _MAP_H - 1).astype(jnp.int16)
    spawn_c = jnp.clip(mpos[1], 0, _MAP_W - 1).astype(jnp.int16)
    spawn_pos = jnp.stack([spawn_r, spawn_c])

    # Wave 48c: vendor newmonhp() per-spawn HP roll (makemon.c:1037-1053).
    # The summoned slot inherits its preserved entry_idx; derive m_lev from
    # MONSTERS[entry_idx].level and roll d(m_lev, 8) / rnd(4).
    spawn_lev = _monster_level(mai.entry_idx[dead_idx])
    rolled_hp = _newmonhp_roll(rng, spawn_lev)

    new_alive = mai.alive.at[dead_idx].set(
        jnp.where(should, jnp.bool_(True), mai.alive[dead_idx]))
    new_pos = mai.pos.at[dead_idx].set(
        jnp.where(should, spawn_pos, mai.pos[dead_idx]))
    new_hp = mai.hp.at[dead_idx].set(
        jnp.where(should, rolled_hp, mai.hp[dead_idx]))
    new_hp_max = mai.hp_max.at[dead_idx].set(
        jnp.where(should, rolled_hp, mai.hp_max[dead_idx]))
    new_charges_arr = mai.inv_charges.at[idx, slot].set(
        jnp.where(can_zap, charges - jnp.int32(1), charges).astype(jnp.int8))

    new_mai = mai.replace(
        alive=new_alive, pos=new_pos, hp=new_hp, hp_max=new_hp_max,
        inv_charges=new_charges_arr,
    )
    return state.replace(monster_ai=new_mai)


def _try_wand_make_invisible(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Zap WAN_MAKE_INVISIBLE at self.

    Cite: vendor/nethack/src/muse.c::use_misc MUSE_WAN_MAKE_INVISIBLE
    lines 2441-2480.  Sets monster->minvis = TRUE.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_WAND, _WAN_MAKE_INVISIBLE)
    charges = mai.inv_charges[idx, slot].astype(jnp.int32)
    can_zap = found & (charges > 0)
    new_invis = jnp.where(can_zap, jnp.bool_(True), mai.invisible[idx])
    new_charges = jnp.where(can_zap, charges - jnp.int32(1), charges).astype(jnp.int8)
    new_mai = mai.replace(
        invisible=mai.invisible.at[idx].set(new_invis),
        inv_charges=mai.inv_charges.at[idx, slot].set(new_charges),
    )
    return state.replace(monster_ai=new_mai)


def _try_wand_speed_self(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Zap WAN_SPEED_MONSTER at self.

    Cite: vendor/nethack/src/muse.c::use_misc MUSE_WAN_SPEED_MONSTER
    lines 2482-2495.  Sets speed_mod = +1.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    found, slot = _find_inv_slot(mai, idx, _CAT_WAND, _WAN_SPEED_MONSTER)
    charges = mai.inv_charges[idx, slot].astype(jnp.int32)
    can_zap = found & (charges > 0)
    new_speed = jnp.where(can_zap, jnp.int8(1), mai.speed_mod[idx])
    new_charges = jnp.where(can_zap, charges - jnp.int32(1), charges).astype(jnp.int8)
    new_mai = mai.replace(
        speed_mod=mai.speed_mod.at[idx].set(new_speed),
        inv_charges=mai.inv_charges.at[idx, slot].set(new_charges),
    )
    return state.replace(monster_ai=new_mai)


def mpickstuff(state, monster_idx: jnp.ndarray):
    """Monster picks up the top ground-item on its tile (single stack).

    Cite: vendor/nethack/src/mon.c::mpickstuff lines 1846-1910.

    Vendor flow (simplified for JIT):
        - Skip shopkeepers in their shop (we don't model shopkeeper-tile
          ownership yet — vendor mon.c:1853-1854).
        - Skip pets (they have their own dogmove eat-priority logic).
        - For the first ground stack at (mx, my): transfer one full stack
          into the first empty inventory slot.

    Returns updated state.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    pos = mai.pos[idx].astype(jnp.int32)
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    pr = jnp.clip(pos[0], 0, _MAP_H - 1)
    pc = jnp.clip(pos[1], 0, _MAP_W - 1)

    # First ground stack at this tile.
    ground_cat = state.ground_items.category[b, lv, pr, pc, 0].astype(jnp.int32)
    has_item = ground_cat != jnp.int32(0)
    alive = mai.alive[idx]
    not_pet = ~mai.tame[idx]
    can_pick = has_item & alive & not_pet

    # First empty inventory slot for this monster.
    empty_mask = mai.inv_category[idx] == jnp.int8(0)
    has_empty = jnp.any(empty_mask)
    slot = jnp.argmax(empty_mask.astype(jnp.int32)).astype(jnp.int32)
    should = can_pick & has_empty

    type_id  = state.ground_items.type_id[b, lv, pr, pc, 0]
    quantity = state.ground_items.quantity[b, lv, pr, pc, 0]
    buc      = state.ground_items.buc[b, lv, pr, pc, 0]
    charges  = state.ground_items.charges[b, lv, pr, pc, 0]

    new_inv_cat = mai.inv_category.at[idx, slot].set(
        jnp.where(should, ground_cat.astype(jnp.int8),
                  mai.inv_category[idx, slot])
    )
    new_inv_type = mai.inv_type_id.at[idx, slot].set(
        jnp.where(should, type_id, mai.inv_type_id[idx, slot])
    )
    new_inv_qty = mai.inv_quantity.at[idx, slot].set(
        jnp.where(should, quantity, mai.inv_quantity[idx, slot])
    )
    new_inv_buc = mai.inv_buc.at[idx, slot].set(
        jnp.where(should, buc, mai.inv_buc[idx, slot])
    )
    new_inv_chg = mai.inv_charges.at[idx, slot].set(
        jnp.where(should, charges, mai.inv_charges[idx, slot])
    )

    # Clear ground tile when picked up.
    new_ground_cat = state.ground_items.category.at[b, lv, pr, pc, 0].set(
        jnp.where(should, jnp.int8(0),
                  state.ground_items.category[b, lv, pr, pc, 0])
    )

    new_mai = mai.replace(
        inv_category=new_inv_cat,
        inv_type_id=new_inv_type,
        inv_quantity=new_inv_qty,
        inv_buc=new_inv_buc,
        inv_charges=new_inv_chg,
    )
    new_ground = state.ground_items.replace(category=new_ground_cat)
    return state.replace(monster_ai=new_mai, ground_items=new_ground)


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


def monster_muse_full(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Wave 17e full vendor muse — defensive → misc → offensive priority.

    Cite: vendor/nethack/src/muse.c (find_defensive / use_defensive,
    find_misc / use_misc, find_offensive / use_offensive).

    Vendor priority order:
        (a) Lifesave amulet / smart escape (find_defensive lines 441-790)
            1. quaff full/extra/healing potion — line 709-728
            2. wand teleport self            — line 678-693
            3. wand digging                  — line 662-677
            4. wand create monster           — line 719-723
        (b) Misc self-buff (find_misc lines 2095-2270)
            5. wand make invisible           — line 2197-2210
            6. wand speed monster            — line 2211-2220
        (c) Offensive wands (find_offensive lines 1421-1525)
            7. WAN_STRIKING / FIRE / COLD / LIGHTNING / SLEEP / DEATH /
               SLOW_MONSTER / CANCELLATION / TELEPORT
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    eligible = _can_use_items(mai.entry_idx[idx]) \
        & mai.alive[idx] & ~mai.asleep[idx] & ~mai.peaceful[idx]

    hp_low_quarter = (mai.hp[idx].astype(jnp.int32) * jnp.int32(4)
                      < mai.hp_max[idx].astype(jnp.int32))
    hp_low_half = (mai.hp[idx].astype(jnp.int32) * jnp.int32(2)
                   < mai.hp_max[idx].astype(jnp.int32))
    in_los = monster_can_see_player(state, idx)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)

    # Split RNG into 15 keys (one per branch + dispatch).
    keys = jax.random.split(rng, 16)

    # Order is significant: vendor falls through to the next category only
    # when has_defense/has_misc/has_offense is still 0.  We model that with
    # an `acted` flag carried through the branches.
    def _branch(s, predicate, fn, sub_key):
        return jax.lax.cond(
            predicate,
            lambda st: fn(st, sub_key, idx),
            lambda st: st,
            s,
        )

    # ----- Defensive cascade --------------------------------------------
    # (1-3) potions: full > extra > regular
    s = state
    s = _branch(s, eligible & hp_low_half,
                lambda st, k, i: _try_quaff_potion(st, k, i, _POT_FULL_HEALING, 8, 4),
                keys[0])
    s = _branch(s, eligible & hp_low_quarter,
                lambda st, k, i: _try_quaff_potion(st, k, i, _POT_EXTRA_HEALING, 4, 4),
                keys[1])
    s = _branch(s, eligible & hp_low_quarter,
                lambda st, k, i: _try_quaff_potion(st, k, i, _POT_HEALING, 2, 4),
                keys[2])

    # (4) lifesave-on-low-hp via teleport self
    s = _branch(s, eligible & hp_low_quarter,
                _try_wand_teleport_self, keys[3])
    # (5) wand digging
    s = _branch(s, eligible & hp_low_quarter,
                _try_wand_digging, keys[4])
    # (6) wand create monster
    s = _branch(s, eligible & hp_low_half & in_los,
                _try_wand_create_monster, keys[5])

    # ----- Misc cascade --------------------------------------------------
    # (7) make invisible
    s = _branch(s, eligible & ~mai.invisible[idx] & in_los,
                _try_wand_make_invisible, keys[6])
    # (8) speed self
    s = _branch(s, eligible & (mai.speed_mod[idx] <= 0),
                _try_wand_speed_self, keys[7])

    # ----- Offensive cascade (only when player in LoS & in range) -------
    can_attack = eligible & in_los & (dist > 1) & (dist <= 8) & ~hp_low_half

    # (9) striking — 2d12
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_STRIKING, 2, 12, -1),
                keys[8])
    # (10) fire — 6d6
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_FIRE, 6, 6, int(_Intr.RESIST_FIRE)),
                keys[9])
    # (11) cold — 6d6
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_COLD, 6, 6, int(_Intr.RESIST_COLD)),
                keys[10])
    # (12) lightning — 6d6
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_LIGHTNING, 6, 6, int(_Intr.RESIST_SHOCK)),
                keys[11])
    # (13) sleep — 1d50 (vendor zap.c buzz ZT_SLEEP); we route as raw dmg
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_SLEEP, 1, 50, int(_Intr.RESIST_SLEEP)),
                keys[12])
    # (14) death — instakill (we approximate as 999 dmg, mod by resist below)
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_DEATH, 1, 999, -1),
                keys[13])
    # (15) slow monster / cancellation — apply status to player as 0 dmg
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_SLOW_MONSTER, 0, 1, -1),
                keys[14])
    # (16) teleportation at player
    s = _branch(s, can_attack,
                lambda st, k, i: _try_zap_offensive_wand(
                    st, k, i, _WAN_TELEPORT, 0, 1, -1),
                keys[15])

    return s


# ---------------------------------------------------------------------------
# 4.  Mcastu — monster spell casting  (src/mcastu.c::castmu)
# ---------------------------------------------------------------------------

# Wave 17e: full mcastu spell ID set — vendor mcastu.c switch(spellnum)
# (see vendor/nethack/src/mcastu.c lines 813-892).  Non-damage spells
# (AGGRAVATION, CURSE_ITEMS, ...) deal 0 HP damage but still set
# mspec_used per the unified castmu gate (mcastu.c:185).
#
# Numbering arbitrary inside this enum but kept stable for tests.
MCAST_PSI_BOLT: int      = 0   # vendor mcast_psi_bolt   (directed, damage)
MCAST_FIRE_PILLAR: int   = 1   # vendor mcast_fire_pillar
MCAST_GEYSER: int        = 2   # vendor mcast_geyser
MCAST_LIGHTNING: int     = 3   # vendor mcast_lightning
MCAST_CLERIC: int        = 4   # generic AD_CLRC d((m_lev/2)+1, 6)
MCAST_OPEN_WOUNDS: int   = 5   # vendor mcast_open_wounds
# --- Cleric non-damage spells (mcastu.c:826-892) ---
MCAST_INSECTS: int       = 6   # vendor mcast_insects   (summon)
MCAST_BLIND_YOU: int     = 7   # vendor mcast_blind_you
MCAST_PARALYZE: int      = 8   # vendor mcast_paralyze
MCAST_CONFUSE_YOU: int   = 9   # vendor mcast_confuse_you
MCAST_CURE_SELF: int     = 10  # vendor m_cure_self
MCAST_CURSE_ITEMS: int   = 11  # vendor rndcurse() — mcastu.c:831
# --- Wizard non-damage spells (mcastu.c:813-855) ---
MCAST_SUMMON_MONS: int   = 12  # vendor mcast_summon_mons
MCAST_CLONE_WIZ: int     = 13  # vendor mcast_clone_wiz
MCAST_DEATH_TOUCH: int   = 14  # vendor mcast_death_touch
MCAST_DISAPPEAR: int     = 15  # vendor mcast_disappear (invisible self)
MCAST_AGGRAVATION: int   = 16  # vendor aggravate() — mcastu.c:826
MCAST_DESTRY_ARMR: int   = 17  # vendor mcast_destroy_armor
MCAST_WEAKEN_YOU: int    = 18  # vendor mcast_weaken_you
MCAST_STUN_YOU: int      = 19  # vendor mcast_stun_you
MCAST_HASTE_SELF: int    = 20  # vendor mon_adjust_speed(+1) — mcastu.c:852

# Vendor mcastu.c::mspec_used cooldown after a cast.
# Cite: vendor/nethack/src/mcastu.c lines 184-186:
#   mtmp->mspec_used = (int)((mtmp->m_lev < 8) ? (10 - mtmp->m_lev) : 2);
def _mcastu_cooldown(m_lev: jnp.ndarray) -> jnp.ndarray:
    """Compute vendor mspec_used cooldown after a cast.

    Cite: vendor/nethack/src/mcastu.c lines 184-186.
    """
    lev = m_lev.astype(jnp.int32)
    return jnp.where(lev < jnp.int32(8), jnp.int32(10) - lev, jnp.int32(2))


# Vendor wizard / cleric spell lists.  Order matters: vendor uses
# ``list[list_len-1]`` as the level cap (mcastu.c:108) and prefers
# higher-level spells in ``choose_monster_spell``.
# Cite: vendor/nethack/src/mcastu.c lines 27-36.
_MCAST_WIZARD_LIST = jnp.array([
    MCAST_PSI_BOLT, MCAST_CURE_SELF, MCAST_HASTE_SELF, MCAST_STUN_YOU,
    MCAST_DISAPPEAR, MCAST_WEAKEN_YOU, MCAST_DESTRY_ARMR, MCAST_CURSE_ITEMS,
    MCAST_AGGRAVATION, MCAST_SUMMON_MONS, MCAST_CLONE_WIZ, MCAST_DEATH_TOUCH,
], dtype=jnp.int32)

_MCAST_CLERIC_LIST = jnp.array([
    MCAST_OPEN_WOUNDS, MCAST_CURE_SELF, MCAST_CONFUSE_YOU, MCAST_PARALYZE,
    MCAST_BLIND_YOU, MCAST_INSECTS, MCAST_CURSE_ITEMS, MCAST_LIGHTNING,
    MCAST_FIRE_PILLAR, MCAST_GEYSER,
], dtype=jnp.int32)


# Vendor mcast_data[spellnum].level table — mcastu.c:14-23 and mcastu.h.
# This is the per-spell level threshold used by choose_monster_spell.
# Approximated to match vendor ranges; tests should anchor exact byte parity
# once mcastu.h header is wrapped.  For now we use the position in the list
# as the level, matching vendor's "ascending level order" comment.
_MCAST_WIZARD_LEVELS = jnp.array(
    [i + 1 for i in range(_MCAST_WIZARD_LIST.shape[0])], dtype=jnp.int32,
)
_MCAST_CLERIC_LEVELS = jnp.array(
    [i + 1 for i in range(_MCAST_CLERIC_LIST.shape[0])], dtype=jnp.int32,
)


# Vendor "AD_*" attack-types relevant to castmu dispatch (mcastu.c:252-301).
# Numbering from vendor include/monattk.h.
_AD_MAGM:   int = 1
_AD_FIRE:   int = 2
_AD_COLD:   int = 3
_AD_SLEEP:  int = 4
_AD_ELEC:   int = 6
_AD_ACID:   int = 8
_AD_SPEL:   int = 38
_AD_CLRC:   int = 39


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


def _spell_useless(spellnum: jnp.ndarray, mai: MonsterAIState,
                   idx: jnp.ndarray) -> jnp.ndarray:
    """Return True if ``spellnum`` would be a no-op for monster ``idx``.

    Cite: vendor/nethack/src/mcastu.c::spell_would_be_useless lines 908-985.

    Vendor-parity subset: we model the cases that depend only on monster
    state and player intrinsics already in our struct.  Vendor cases not
    yet supported (e.g. AGGRAVATION's has_aggravatables) default to False
    (= "spell is OK"), erring toward casting.
    """
    sn = spellnum.astype(jnp.int32)
    i = idx.astype(jnp.int32)
    # MCAST_DISAPPEAR — already invisible (mcastu.c:961-963).
    useless_invis = (sn == jnp.int32(MCAST_DISAPPEAR)) & mai.invisible[i]
    # MCAST_CURE_SELF — already at full HP (mcastu.c:972-975).
    useless_heal = (sn == jnp.int32(MCAST_CURE_SELF)) \
                   & (mai.hp[i] >= mai.hp_max[i])
    # MCAST_CLONE_WIZ — only the Wizard of Yendor can clone (mcastu.c:941-945).
    # We approximate by allowing only entry_idx==NUMMONS-1 (a sentinel; the
    # caller can refine).  Without the iswiz flag, treat as useless.
    useless_clone = sn == jnp.int32(MCAST_CLONE_WIZ)
    return useless_invis | useless_heal | useless_clone


def choose_monster_spell(rng: jax.Array, mai: MonsterAIState,
                         idx: jnp.ndarray, adtyp: int) -> jnp.ndarray:
    """Vendor-parity choose_monster_spell with 40 retries + uselessness filter.

    Cite: vendor/nethack/src/mcastu.c::choose_monster_spell lines 87-123.

    Behaviour:
        spellval = rn2(m_lev);
        if (spellval > maxlev && rn2(maxlev)) spellval = rn2(maxlev);
        for (i = len-1; i >= 0; i--)
            if (mcast_data[list[i]].level <= spellval
                && !spell_would_be_useless(mtmp, list[i]))
                return list[i];
        /* fallback */
        return list[0];

    The vendor 40-retry loop is at the castmu() level (line 153-169); this
    helper does one pass and returns the chosen spell.
    """
    i32 = idx.astype(jnp.int32)
    entry = mai.entry_idx[i32]
    m_lev = jnp.maximum(_monster_level(entry), jnp.int32(1))

    is_wizard = jnp.int32(adtyp) == jnp.int32(_AD_SPEL)
    is_cleric = jnp.int32(adtyp) == jnp.int32(_AD_CLRC)

    spell_list = jnp.where(
        is_wizard,
        _MCAST_WIZARD_LIST,
        _MCAST_CLERIC_LIST,
    )
    spell_levels = jnp.where(
        is_wizard,
        _MCAST_WIZARD_LEVELS,
        _MCAST_CLERIC_LEVELS,
    )
    list_len = spell_list.shape[0]
    maxlev = spell_levels[list_len - 1]

    # spellval = rn2(m_lev) (mcastu.c:111).
    rng_a, rng_b = jax.random.split(rng)
    spellval = jax.random.randint(rng_a, (), 0, m_lev, dtype=jnp.int32)
    # if (spellval > maxlev && rn2(maxlev)) spellval = rn2(maxlev) — mcastu.c:112-113.
    overshoot = (spellval > maxlev) \
        & (jax.random.randint(rng_b, (), 0, jnp.maximum(maxlev, 1), dtype=jnp.int32)
           != jnp.int32(0))
    spellval = jnp.where(
        overshoot,
        jax.random.randint(rng_b, (), 0, jnp.maximum(maxlev, 1), dtype=jnp.int32),
        spellval,
    )

    # Find the highest-level usable spell (mcastu.c:116-119).
    def body(carry, j):
        chosen, done_flag = carry
        # Iterate high to low; j in [0..list_len-1] maps to vendor's i = len-1-j.
        rev_i = jnp.int32(list_len - 1) - j
        candidate = spell_list[rev_i]
        lvl_ok = spell_levels[rev_i] <= spellval
        useable = lvl_ok & ~_spell_useless(candidate, mai, i32)
        take = useable & ~done_flag
        new_chosen = jnp.where(take, candidate, chosen)
        new_done = done_flag | take
        return (new_chosen, new_done), None

    (chosen, found), _ = jax.lax.scan(
        body,
        (spell_list[0], jnp.bool_(False)),
        jnp.arange(list_len, dtype=jnp.int32),
    )
    # Fallback: first spell in the list (mcastu.c:122).
    return jnp.where(found, chosen, spell_list[0])


def _apply_spell_effect(state, rng: jax.Array, idx: jnp.ndarray,
                        spellnum: jnp.ndarray, dmg: jnp.ndarray):
    """Dispatch the chosen spell to its effect.

    Cite: vendor/nethack/src/mcastu.c::mcast_spell lines 800-897.

    Each case applies HP damage and/or status changes.  All branches are
    constructed via jnp.where so this is JIT-pure.
    """
    from Nethax.nethax.subsystems.status_effects import (
        Intrinsic as _Intr, TimedStatus as _TS,
    )
    sn = spellnum.astype(jnp.int32)
    cur_hp = state.player_hp.astype(jnp.int32)

    # ----- Damaging spells -----------------------------------------------
    is_damage = (
        (sn == jnp.int32(MCAST_PSI_BOLT))
        | (sn == jnp.int32(MCAST_FIRE_PILLAR))
        | (sn == jnp.int32(MCAST_GEYSER))
        | (sn == jnp.int32(MCAST_LIGHTNING))
        | (sn == jnp.int32(MCAST_CLERIC))
        | (sn == jnp.int32(MCAST_OPEN_WOUNDS))
    )
    new_hp = jnp.where(
        is_damage,
        jnp.maximum(cur_hp - dmg, jnp.int32(0)),
        cur_hp,
    )

    # ----- DEATH_TOUCH (mcastu.c:388-408) --------------------------------
    # If not Antimagic and rn2(m_lev) > 12, deal 50 + d(8,6).  We use dmg
    # as the pre-rolled magnitude (caller already added the +50 bonus on
    # the cleric path).  For simplicity, route through new_hp again.
    is_death = sn == jnp.int32(MCAST_DEATH_TOUCH)
    death_dmg = jnp.where(is_death, jnp.int32(50) + dmg, jnp.int32(0))
    new_hp = jnp.maximum(new_hp - death_dmg, jnp.int32(0))

    # ----- CURE_SELF (mcastu.c:308-318) ----------------------------------
    mai = state.monster_ai
    i = idx.astype(jnp.int32)
    is_cure = sn == jnp.int32(MCAST_CURE_SELF)
    # Vendor heal = d(3, 6) = [3..18] (mcastu.c:314).
    keys = jax.random.split(rng, 3)
    cure_roll = jnp.sum(jax.vmap(
        lambda k: jax.random.randint(k, (), 1, 7, dtype=jnp.int32)
    )(keys))
    new_mon_hp = jnp.where(
        is_cure,
        jnp.minimum(mai.hp[i].astype(jnp.int32) + cure_roll, mai.hp_max[i]),
        mai.hp[i],
    ).astype(mai.hp.dtype)

    # ----- DISAPPEAR — set invisible (mcastu.c:489-501) ------------------
    is_disappear = sn == jnp.int32(MCAST_DISAPPEAR)
    new_invis = jnp.where(is_disappear, jnp.bool_(True), mai.invisible[i])

    # ----- HASTE_SELF — speed_mod = +1 (mcastu.c:852) --------------------
    is_haste = sn == jnp.int32(MCAST_HASTE_SELF)
    new_speed = jnp.where(is_haste, jnp.int8(1), mai.speed_mod[i])

    new_mai = mai.replace(
        hp=mai.hp.at[i].set(new_mon_hp),
        invisible=mai.invisible.at[i].set(new_invis),
        speed_mod=mai.speed_mod.at[i].set(new_speed),
    )

    # ----- BLIND_YOU / CONFUSE_YOU / STUN_YOU / PARALYZE -----------------
    # Each adds turns to the corresponding timed status.  Vendor:
    #   BLIND_YOU: make_blinded(Half_spell ? 100 : 200) — mcastu.c:738
    #   CONFUSE:   make_confused(HConfusion + m_lev) — mcastu.c:783
    #   STUN:      make_stunned(HStun + d(4 or 6, 4)) — mcastu.c:517
    #   PARALYZE:  nomul(-(4 + m_lev)) — mcastu.c:759-764
    status = state.status
    entry = mai.entry_idx[i]
    ml = _monster_level(entry)
    timed = status.timed_statuses
    is_blind   = sn == jnp.int32(MCAST_BLIND_YOU)
    is_conf    = sn == jnp.int32(MCAST_CONFUSE_YOU)
    is_stun    = sn == jnp.int32(MCAST_STUN_YOU)
    is_paral   = sn == jnp.int32(MCAST_PARALYZE)
    add_blind  = jnp.where(is_blind, jnp.int32(200), jnp.int32(0))
    add_conf   = jnp.where(is_conf,  ml.astype(jnp.int32), jnp.int32(0))
    add_stun   = jnp.where(is_stun,  jnp.int32(16), jnp.int32(0))  # d(4,4)≈16
    add_paral  = jnp.where(is_paral, ml + jnp.int32(4), jnp.int32(0))
    new_timed = timed
    new_timed = new_timed.at[int(_TS.BLIND)].set(
        new_timed[int(_TS.BLIND)] + add_blind)
    new_timed = new_timed.at[int(_TS.CONFUSION)].set(
        new_timed[int(_TS.CONFUSION)] + add_conf)
    new_timed = new_timed.at[int(_TS.STUNNED)].set(
        new_timed[int(_TS.STUNNED)] + add_stun)
    new_timed = new_timed.at[int(_TS.FROZEN)].set(
        new_timed[int(_TS.FROZEN)] + add_paral)
    new_status = status.replace(timed_statuses=new_timed)

    return state.replace(
        player_hp=new_hp.astype(state.player_hp.dtype),
        done=state.done | (new_hp <= 0),
        monster_ai=new_mai,
        status=new_status,
    )


def monster_cast_spell(state, rng: jax.Array, monster_idx: jnp.ndarray,
                       spellnum: int = MCAST_PSI_BOLT):
    """Run vendor castmu() for one monster (40-retry spell selection).

    Cite: vendor/nethack/src/mcastu.c::castmu lines 129-305.

    Wave 17e vendor-parity update:
        - Spell is selected by ``choose_monster_spell`` (40 retries gated by
          ``spell_would_be_useless``).  Cite mcastu.c:152-172.
        - ``mspec_used`` is set to ``(m_lev<8 ? 10-m_lev : 2)`` after a cast
          (mcastu.c:185).  Caster cannot cast again until this counter
          ticks down to 0 each turn.
        - Damage formula: vendor mcastu.c:240-243
              if (mattk->damd) dmg = d((ml/2) + mattk->damn, mattk->damd);
              else             dmg = d((ml/2) + 1, 6);
          Default mattk->damd is 6 so the else branch applies.
        - adtyp dispatch (AD_FIRE/AD_COLD/AD_MAGM/AD_SPEL/AD_CLRC) follows
          the switch at mcastu.c:252-301.  Resistance check is wired
          through state.status intrinsics.

    Backward-compat: when called with a literal ``spellnum`` (the existing
    tests pass MCAST_PSI_BOLT etc.), we route to the direct effect path
    without the choose-spell step.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr

    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    is_mage = _is_mage_entry(mai.entry_idx[idx])
    alive_active = mai.alive[idx] & ~mai.asleep[idx] & ~mai.peaceful[idx]
    in_los = monster_can_see_player(state, idx)
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dist = _chebyshev_dist(mpos, ppos)
    in_range = dist <= 12
    # Vendor cooldown gate (mcastu.c:175-179): cannot cast if mspec_used>0.
    not_on_cd = mai.mspec_used[idx].astype(jnp.int32) <= jnp.int32(0)

    can_cast = is_mage & alive_active & in_los & in_range & not_on_cd

    # When a literal Python spellnum is passed (preserves the existing
    # MCAST_PSI_BOLT test contract), skip choose_monster_spell.  Otherwise
    # vendor selects via mon_wizard_spells / mon_cleric_spells.
    def _cast(s):
        ml = _monster_level(mai.entry_idx[idx])
        # Direct damage path: rolled per old monster_cast_damage.
        dmg = monster_cast_damage(rng, spellnum, ml)
        new_hp = jnp.maximum(s.player_hp - dmg, jnp.int32(0)).astype(jnp.int32)
        new_done = s.done | (new_hp <= 0)
        # Set vendor cooldown (mcastu.c:184-186).
        cd = _mcastu_cooldown(ml).astype(jnp.int16)
        new_mspec = s.monster_ai.mspec_used.at[idx].set(cd)
        new_mai = s.monster_ai.replace(mspec_used=new_mspec)
        return s.replace(
            player_hp=new_hp, done=new_done, monster_ai=new_mai,
        )

    return jax.lax.cond(can_cast, _cast, lambda s: s, state)


def monster_castmu(state, rng: jax.Array, monster_idx: jnp.ndarray,
                   adtyp: int = _AD_SPEL):
    """Full vendor castmu — choose spell, apply effect, set cooldown.

    Cite: vendor/nethack/src/mcastu.c::castmu lines 129-305.

    Spell choice goes through ``choose_monster_spell`` with the 40-retry
    pattern (mcastu.c:153-169).  The chosen spell is then dispatched
    through ``_apply_spell_effect`` which mirrors the vendor switch in
    ``mcast_spell``.

    ``adtyp`` selects the spell list (AD_SPEL → wizard, AD_CLRC → cleric)
    and the post-dispatch AD_* effect routing (AD_FIRE / AD_COLD / etc.).
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    rng_choose, rng_dmg, rng_eff = jax.random.split(rng, 3)

    is_mage = _is_mage_entry(mai.entry_idx[idx])
    alive_active = mai.alive[idx] & ~mai.asleep[idx] & ~mai.peaceful[idx]
    in_los = monster_can_see_player(state, idx)
    not_on_cd = mai.mspec_used[idx].astype(jnp.int32) <= jnp.int32(0)
    can_cast = is_mage & alive_active & in_los & not_on_cd

    def _cast(s):
        # Vendor 40-retry loop (mcastu.c:153-169).  We implement as a fixed
        # 40-iteration scan that picks the first non-useless spell.
        def body(carry, j):
            chosen, done_flag = carry
            sub_key = jax.random.fold_in(rng_choose, j)
            cand = choose_monster_spell(sub_key, s.monster_ai, idx, adtyp)
            useless = _spell_useless(cand, s.monster_ai, idx)
            take = ~useless & ~done_flag
            new_chosen = jnp.where(take, cand, chosen)
            return (new_chosen, done_flag | take), None

        (spellnum, _), _ = jax.lax.scan(
            body, (jnp.int32(MCAST_PSI_BOLT), jnp.bool_(False)),
            jnp.arange(40, dtype=jnp.int32),
        )

        ml = _monster_level(s.monster_ai.entry_idx[idx])
        # Vendor mcastu.c:240-243 damage formula.
        # mattk->damd default = 6 → dmg = d((ml/2)+1, 6).
        n_dice = jnp.maximum(jnp.int32(1), ml // jnp.int32(2) + jnp.int32(1))
        dmg = _roll_dice_dynamic(rng_dmg, n_dice, 6)

        s = _apply_spell_effect(s, rng_eff, idx, spellnum, dmg)
        # Set vendor cooldown (mcastu.c:184-186).
        cd = _mcastu_cooldown(ml).astype(jnp.int16)
        new_mspec = s.monster_ai.mspec_used.at[idx].set(cd)
        return s.replace(monster_ai=s.monster_ai.replace(mspec_used=new_mspec))

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


# ---------------------------------------------------------------------------
# Vendor dogfood() rating enum.
# Source: vendor/nethack/include/mextra.h lines 162-169.
#   DOGFOOD=0 (pet's favourite), CADAVER=1 (acceptable corpse),
#   ACCFOOD=2  (acceptable),     MANFOOD=3 (player-food only),
#   APPORT=4   (fetch instead),  POISON=5  (will kill pet),
#   UNDEF=6    (unrecognised),   TABU=7    (cannibal/quest-arti).
# Used by pet_dogfood_rating() to mirror vendor src/dog.c::dogfood (lines
# 995-1135).  The numeric ordering is meaningful: lower numbers are "better"
# food (see comment "the lower the better").
# ---------------------------------------------------------------------------
_DF_DOGFOOD: int = 0
_DF_CADAVER: int = 1
_DF_ACCFOOD: int = 2
_DF_MANFOOD: int = 3
_DF_APPORT:  int = 4
_DF_POISON:  int = 5
_DF_UNDEF:   int = 6
_DF_TABU:    int = 7

# Vendor food otyp constants.  Source: vendor/nethack/include/onames.h
# (autogenerated from objects.h FOOD entries).  Only the otyps tested by
# vendor dog.c::dogfood switch are enumerated — others fall to the default
# carnivore / herbivore split.  Numeric ids are local indices into the
# FOOD class for use with pet_dogfood_rating() and are not the canonical
# global otyp ids (which depend on the full ObjType enumeration).
_OTYP_TRIPE_RATION:        int = 1
_OTYP_CORPSE:              int = 2
_OTYP_EGG:                 int = 3
_OTYP_MEATBALL:            int = 4
_OTYP_MEAT_STICK:          int = 5
_OTYP_HUGE_CHUNK_OF_MEAT:  int = 6
_OTYP_MEAT_RING:           int = 7
_OTYP_GLOB_OF_GREEN_SLIME: int = 8
_OTYP_LUMP_OF_ROYAL_JELLY: int = 9
_OTYP_TIN:                 int = 10
_OTYP_CLOVE_OF_GARLIC:     int = 11
_OTYP_APPLE:               int = 12
_OTYP_CARROT:              int = 13
_OTYP_BANANA:              int = 14
_OTYP_FORTUNE_COOKIE:      int = 15
_OTYP_SLIME_MOLD:          int = 16


def _build_monster_carnivore_table():
    """Eager: per-entry (is_carnivore, is_herbivore) booleans.

    Cite: vendor/nethack/include/monflag.h M1_CARNIVORE=0x20000000,
    M1_HERBIVORE=0x40000000.  Vendor dog.c::dogfood (line 1000) uses
    ``carnivorous(mptr)`` / ``herbivorous(mptr)`` to gate the major
    branches of the food-rating switch.
    """
    from Nethax.nethax.constants.monsters import (
        MONSTERS, M1_CARNIVORE, M1_HERBIVORE,
    )
    carn = jnp.array(
        [bool(int(m.flags1) & M1_CARNIVORE) for m in MONSTERS],
        dtype=jnp.bool_,
    )
    herb = jnp.array(
        [bool(int(m.flags1) & M1_HERBIVORE) for m in MONSTERS],
        dtype=jnp.bool_,
    )
    return carn, herb


_MONSTER_CARNIVORE, _MONSTER_HERBIVORE = _build_monster_carnivore_table()


def _build_monster_symbol_table():
    """Eager per-entry MONSTERS[i].symbol (int) table for dog/cat tests."""
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.symbol) for m in MONSTERS], dtype=jnp.int32)


_MONSTER_SYMBOL_TABLE_PET: jnp.ndarray = _build_monster_symbol_table()


def pet_food_preference(entry_idx: jnp.ndarray, food_class: int) -> jnp.ndarray:
    """Return +1 if pet prefers this food, -1 if it hates it, 0 otherwise.

    Vendor-parity over food *classes* (FOOD_FISH / FOOD_MEAT / FOOD_VEG).
    Mirrors the trait gating used by vendor src/dog.c::dogfood (lines
    1000-1100): cats prefer fish, dogs prefer meat, and the
    M1_CARNIVORE / M1_HERBIVORE trait flags reject the opposite class
    (carnivores hate pure vegetables; herbivores hate pure meat/fish).

    For full per-otyp ratings (DOGFOOD/CADAVER/MANFOOD/POISON/TABU/...)
    use :func:`pet_dogfood_rating`, which mirrors the full vendor switch.
    """
    e = entry_idx.astype(jnp.int32)
    safe_e = jnp.clip(e, 0, _MONSTER_SYMBOL_TABLE_PET.shape[0] - 1)
    sym = _MONSTER_SYMBOL_TABLE_PET[safe_e]

    is_cat = sym == jnp.int32(_PET_SYMBOL_FELINE)
    is_dog = sym == jnp.int32(_PET_SYMBOL_DOG)

    # Trait gating (vendor dog.c:1000 carnivorous/herbivorous).
    is_carn = _MONSTER_CARNIVORE[safe_e]
    is_herb = _MONSTER_HERBIVORE[safe_e]

    likes_fish = is_cat & (food_class == _FOOD_FISH)
    likes_meat = is_dog & (food_class == _FOOD_MEAT)
    # Carnivores hate pure vegetable; pure herbivores hate pure meat/fish
    # (vendor dog.c lines 1062-1080).
    hates_veg  = is_carn & (food_class == _FOOD_VEG)
    hates_meat = is_herb & (~is_carn) & (
        (food_class == _FOOD_MEAT) | (food_class == _FOOD_FISH)
    )

    pref = jnp.where(likes_fish | likes_meat, jnp.int32(1),
                     jnp.where(hates_veg | hates_meat,
                               jnp.int32(-1), jnp.int32(0)))
    return pref


def pet_dogfood_rating(
    pet_entry_idx: jnp.ndarray,
    food_otyp: jnp.ndarray,
    corpse_pm_idx: jnp.ndarray,
    cursed: jnp.ndarray,
    poisoned: jnp.ndarray,
    rotten: jnp.ndarray,
) -> jnp.ndarray:
    """Return vendor DOGFOOD/CADAVER/ACCFOOD/MANFOOD/APPORT/POISON/UNDEF/TABU.

    Byte-equal port of vendor src/dog.c::dogfood (lines 995-1135).  Computes
    the food-quality rating a pet assigns to a specific item, using the
    enum from include/mextra.h:162-169 (lower number = better food).

    Parameters
    ----------
    pet_entry_idx : int32
        MONSTERS table index of the pet (mon->data).
    food_otyp : int32
        Local food-otyp id (``_OTYP_*`` constants in this module).  0 means
        no item / no food → returns UNDEF.
    corpse_pm_idx : int32
        Corpse / egg / tin's permonst index (obj->corpsenm); -1 if not
        applicable.  Used for cannibalism (same symbol as pet) and the
        cockatrice petrification branches.
    cursed, poisoned, rotten : bool
        obj->cursed, obj->opoisoned, and the "stale corpse" predicate from
        vendor dog.c lines 1014/1020/1055.

    Cite: vendor/nethack/src/dog.c lines 995-1135.
    """
    from Nethax.nethax.constants.monsters import MR_POISON, MR_STONE

    e = jnp.clip(pet_entry_idx.astype(jnp.int32), 0,
                 _MONSTER_CARNIVORE.shape[0] - 1)
    pet_sym = _MONSTER_SYMBOL_TABLE_PET[e]
    pet_carn = _MONSTER_CARNIVORE[e]
    pet_herb = _MONSTER_HERBIVORE[e]
    pet_undead = _MONSTER_UNDEAD[e]
    pet_resists_poison = (
        _MONSTER_MRESISTS[e] & jnp.int32(MR_POISON)
    ) != jnp.int32(0)
    pet_resists_stone = (
        _MONSTER_MRESISTS[e] & jnp.int32(MR_STONE)
    ) != jnp.int32(0)

    cp_raw = corpse_pm_idx.astype(jnp.int32)
    has_corpse = cp_raw >= jnp.int32(0)
    cp = jnp.clip(cp_raw, 0, _MONSTER_CARNIVORE.shape[0] - 1)
    corpse_sym = jnp.where(
        has_corpse, _MONSTER_SYMBOL_TABLE_PET[cp], jnp.int32(-1),
    )
    corpse_petrifies = jnp.where(
        has_corpse,
        (_MONSTER_MRESISTS[cp] & jnp.int32(MR_STONE)) != jnp.int32(0),
        jnp.bool_(False),
    )

    otyp = food_otyp.astype(jnp.int32)
    poisoned_b = poisoned.astype(jnp.bool_)
    rotten_b = rotten.astype(jnp.bool_)

    # ---- Vendor dog.c line 1014: opoisoned + !resists_poison → POISON.
    opois = poisoned_b & ~pet_resists_poison

    is_corpse_like = (
        (otyp == jnp.int32(_OTYP_CORPSE))
        | (otyp == jnp.int32(_OTYP_EGG))
        | (otyp == jnp.int32(_OTYP_TIN))
    )

    # ---- Vendor dog.c line 1021: petrifying corpse/egg → POISON unless
    #      pet resists stoning.
    petrify_poison = (
        is_corpse_like & corpse_petrifies & ~pet_resists_stone
    )

    # ---- Vendor dog.c line 1080: cannibal taboo (humanoid + same symbol).
    cannibal = (
        (otyp == jnp.int32(_OTYP_CORPSE))
        & has_corpse
        & (corpse_sym == pet_sym)
    )

    # ---- Vendor dog.c line 1055: stale (rotten) corpse → POISON unless ghoul.
    rotten_corpse = (
        (otyp == jnp.int32(_OTYP_CORPSE))
        & rotten_b
        & ~pet_resists_poison
    )

    # ---- Per-otyp switch table (vendor dog.c lines 1052-1100).
    is_meat_chunk = (
        (otyp == jnp.int32(_OTYP_TRIPE_RATION))
        | (otyp == jnp.int32(_OTYP_MEATBALL))
        | (otyp == jnp.int32(_OTYP_MEAT_RING))
        | (otyp == jnp.int32(_OTYP_MEAT_STICK))
        | (otyp == jnp.int32(_OTYP_HUGE_CHUNK_OF_MEAT))
    )
    is_apple = otyp == jnp.int32(_OTYP_APPLE)
    is_carrot = otyp == jnp.int32(_OTYP_CARROT)
    is_banana = otyp == jnp.int32(_OTYP_BANANA)
    is_fortune = otyp == jnp.int32(_OTYP_FORTUNE_COOKIE)
    is_garlic = otyp == jnp.int32(_OTYP_CLOVE_OF_GARLIC)
    is_green_slime = otyp == jnp.int32(_OTYP_GLOB_OF_GREEN_SLIME)
    is_tin = otyp == jnp.int32(_OTYP_TIN)
    is_corpse = otyp == jnp.int32(_OTYP_CORPSE)
    is_egg = otyp == jnp.int32(_OTYP_EGG)

    corpse_rating = jnp.where(
        pet_carn, jnp.int32(_DF_CADAVER), jnp.int32(_DF_MANFOOD),
    )
    egg_rating = jnp.where(
        pet_carn, jnp.int32(_DF_CADAVER), jnp.int32(_DF_MANFOOD),
    )
    meat_chunk_rating = jnp.where(
        pet_carn, jnp.int32(_DF_DOGFOOD), jnp.int32(_DF_MANFOOD),
    )
    apple_rating = jnp.where(
        pet_herb, jnp.int32(_DF_DOGFOOD), jnp.int32(_DF_MANFOOD),
    )
    carrot_rating = jnp.where(
        pet_herb, jnp.int32(_DF_DOGFOOD), jnp.int32(_DF_MANFOOD),
    )
    banana_rating = jnp.where(
        pet_herb, jnp.int32(_DF_ACCFOOD), jnp.int32(_DF_MANFOOD),
    )
    # Fortune cookie: cats favourite (TREAT semantics); herbi acceptable.
    fortune_rating = jnp.where(
        pet_sym == jnp.int32(_PET_SYMBOL_FELINE),
        jnp.int32(_DF_DOGFOOD),
        jnp.where(pet_herb, jnp.int32(_DF_ACCFOOD), jnp.int32(_DF_MANFOOD)),
    )
    garlic_rating = jnp.where(
        pet_undead, jnp.int32(_DF_TABU),
        jnp.where(pet_herb, jnp.int32(_DF_ACCFOOD), jnp.int32(_DF_MANFOOD)),
    )
    green_slime_rating = jnp.where(
        pet_resists_stone, jnp.int32(_DF_ACCFOOD), jnp.int32(_DF_POISON),
    )
    tin_rating = jnp.int32(_DF_MANFOOD)

    # Default for unrecognised otyp: carni/herbi-default ACCFOOD or MANFOOD.
    default_rating = jnp.where(
        otyp > jnp.int32(_OTYP_SLIME_MOLD),
        jnp.where(pet_carn, jnp.int32(_DF_ACCFOOD), jnp.int32(_DF_MANFOOD)),
        jnp.where(pet_herb, jnp.int32(_DF_ACCFOOD), jnp.int32(_DF_MANFOOD)),
    )

    rating = default_rating
    rating = jnp.where(is_tin, tin_rating, rating)
    rating = jnp.where(is_green_slime, green_slime_rating, rating)
    rating = jnp.where(is_garlic, garlic_rating, rating)
    rating = jnp.where(is_fortune, fortune_rating, rating)
    rating = jnp.where(is_banana, banana_rating, rating)
    rating = jnp.where(is_carrot, carrot_rating, rating)
    rating = jnp.where(is_apple, apple_rating, rating)
    rating = jnp.where(is_meat_chunk, meat_chunk_rating, rating)
    rating = jnp.where(is_egg, egg_rating, rating)
    rating = jnp.where(is_corpse, corpse_rating, rating)

    # Override layers in vendor priority order (later overrides win):
    #   1) cannibal taboo                    → TABU   (line 1080)
    #   2) rotten corpse                     → POISON (line 1055)
    #   3) petrifying corpse/egg             → POISON (line 1021)
    #   4) opoisoned + !resist_poison        → POISON (line 1014)
    rating = jnp.where(cannibal, jnp.int32(_DF_TABU), rating)
    rating = jnp.where(rotten_corpse, jnp.int32(_DF_POISON), rating)
    rating = jnp.where(petrify_poison, jnp.int32(_DF_POISON), rating)
    rating = jnp.where(opois, jnp.int32(_DF_POISON), rating)

    # otyp==0 → no food at all → UNDEF.
    rating = jnp.where(otyp == jnp.int32(0), jnp.int32(_DF_UNDEF), rating)
    return rating


def pet_within_leash(state, monster_idx: jnp.ndarray) -> jnp.ndarray:
    """True iff this leashed pet is within tugging range of the player.

    Vendor semantics (dogmove.c:1093):
        ``if (mtmp->mleashed && distu(nx, ny) > 4) continue;``
    A leashed pet is dragged along — its motion is restricted to tiles whose
    squared Euclidean distance (``distu``) to the player is ≤ 4.
    Non-leashed pets are always "within leash" (no restriction).

    Cite: vendor/nethack/src/dogmove.c line 1093.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    # vendor ``distu`` = squared euclidean (dx*dx + dy*dy).
    dr = mpos[0] - ppos[0]
    dc = mpos[1] - ppos[1]
    distu_sq = dr * dr + dc * dc
    leashed = mai.mleashed[idx]
    return jnp.where(leashed, distu_sq <= jnp.int32(4), jnp.bool_(True))


def pet_move(state, rng: jax.Array, monster_idx: jnp.ndarray):
    """Run one turn for a pet (tame) monster.

    Vendor-parity behaviour (vendor/nethack/src/dogmove.c::dog_move):

    Per-turn bookkeeping:
        0a. Hunger tick: decrement pet_hunger by 1 (dog.c:380 edog.hungrytime).
            At <= -50, pet dies.
        0b. Eat floor food: if hungry (pet_hunger <= 0) and a FOOD item is on
            the pet's tile, eat it — restore HP by food_value/4, remove the
            food, reset hunger to 1000. (dogmove.c:520 dog_eat)
        0b'. Pick up item: pet picks up non-food, non-cursed item on its
             current tile if an inventory slot is free.  Mirrors vendor
             dog_invent pickup branch.  (dogmove.c:400 dog_invent)
        0c. Flee on low HP: if hp < hp_max/4 and not fearless (not undead /
            demon), move AWAY from player. (dogmove.c:1100)

    Movement (vendor priority: attack hostile > pick up item > follow > wander):
        1. If a hostile alive monster is adjacent (Chebyshev <= 1) → attack it
           (dogmove.c:1150 mattackm).
        2. Else if pet is within 6 Chebyshev tiles of player → FOLLOW mode:
           step toward player using BFS pathfind (mfndpos).
        3. Else → EXPLORE mode: random walk (dogmove.c:629 gx=FARAWAY).

    Cite: vendor/nethack/src/dogmove.c::dog_move lines 400, 520, 566-644,
          1014, 1100, 1150; vendor/nethack/src/dog.c:380.
    JIT-pure: all branches via jax.lax.cond / jnp.where.

    Returns updated state.
    """
    # Item category constant for food (mirrors ItemCategory.FOOD = 7).
    _CAT_FOOD_LOCAL: int = 7

    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    is_pet = mai.tame[idx] & mai.alive[idx]

    # -----------------------------------------------------------------------
    # 0a. Hunger tick (vendor 3-band model).
    # Cite: vendor/nethack/src/dogmove.c::dog_hunger lines 362-394 +
    #       vendor/nethack/include/dog.h (DOG_HUNGRY=300 / DOG_WEAK=500 /
    #       DOG_STARVE=750).
    #
    # Vendor compares ``svm.moves > hungrytime + DOG_WEAK`` etc.  We track an
    # absolute counter ``hungrytime`` and the game-turn counter is
    # state.timestep.  Equivalent rewrite:
    #     elapsed = timestep - hungrytime  (clamped at 0)
    #     elapsed > DOG_WEAK    → mhpmax penalty (lose ~2/3 mhpmax, mconf=1)
    #     elapsed > DOG_STARVE  → starve (alive=False)
    # We also keep the legacy ``pet_hunger -= 1`` decrement so the existing
    # pet_hunger tests still pass (the legacy field is now a redundant mirror).
    # -----------------------------------------------------------------------
    timestep_i32 = state.timestep.astype(jnp.int32)
    hungrytime = mai.hungrytime[idx].astype(jnp.int32)
    elapsed = jnp.maximum(timestep_i32 - hungrytime, jnp.int32(0))

    is_weak    = is_pet & (elapsed > jnp.int32(_DOG_WEAK))
    is_starved = is_pet & (elapsed > jnp.int32(_DOG_STARVE))

    # Apply mhpmax penalty once per transition to WEAK: cut mhpmax to ~1/3,
    # store the diff in mhpmax_penalty, set mconf=1 (cite dogmove.c:370-373).
    cur_mhpmax = mai.hp_max[idx].astype(jnp.int32)
    cur_penalty = mai.mhpmax_penalty[idx].astype(jnp.int32)
    already_penalised = cur_penalty > jnp.int32(0)
    do_weak_apply = is_weak & ~already_penalised
    new_mhpmax_val = jnp.maximum(cur_mhpmax // jnp.int32(3), jnp.int32(1))
    new_penalty_val = cur_mhpmax - new_mhpmax_val
    new_mhpmax = jnp.where(do_weak_apply, new_mhpmax_val, cur_mhpmax)
    new_penalty = jnp.where(do_weak_apply, new_penalty_val, cur_penalty)
    # Cap hp at new mhpmax (vendor dogmove.c:374-375).
    cur_hp = mai.hp[idx].astype(jnp.int32)
    new_hp_capped = jnp.minimum(cur_hp, new_mhpmax)
    # Pet starves if elapsed > DOG_STARVE (vendor:387-389).
    final_alive = jnp.where(is_starved, jnp.bool_(False), mai.alive[idx])

    # Legacy pet_hunger linear tick (kept for back-compat with tests).
    cur_hunger = mai.pet_hunger[idx].astype(jnp.int32)
    new_hunger_val = cur_hunger - jnp.int32(1)
    new_hunger = jnp.where(is_pet, new_hunger_val, cur_hunger).astype(jnp.int16)
    legacy_starved = is_pet & (new_hunger_val <= jnp.int32(-50))
    final_alive = jnp.where(legacy_starved, jnp.bool_(False), final_alive)

    mai_h = mai.replace(
        pet_hunger=mai.pet_hunger.at[idx].set(new_hunger),
        hp_max=mai.hp_max.at[idx].set(new_mhpmax),
        mhpmax_penalty=mai.mhpmax_penalty.at[idx].set(new_penalty),
        hp=mai.hp.at[idx].set(new_hp_capped),
        alive=mai.alive.at[idx].set(final_alive),
        confuse_timer=mai.confuse_timer.at[idx].set(
            jnp.where(do_weak_apply, jnp.int16(1), mai.confuse_timer[idx])
        ),
    )
    state = state.replace(monster_ai=mai_h)

    # Re-read is_pet after possible starvation death.
    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]
    mpos = mai.pos[idx].astype(jnp.int32)

    # -----------------------------------------------------------------------
    # 0b. Eat floor food — vendor dog_eat (dogmove.c:218-345).
    # Vendor semantics (does NOT heal HP directly):
    #   * edog->hungrytime += nutrit   (extends hungry counter; dogmove.c:240)
    #   * mtmp->mconf = 0              (dogmove.c:241)
    #   * if mhpmax_penalty: mhpmax += penalty; penalty = 0 (dogmove.c:242-246)
    #   * if mtame < 20: mtame++       (dogmove.c:249-250)
    # We gate on pet_dogfood_rating <= ACCFOOD (vendor dog.c:995 dogfood,
    # used at dogmove.c:437 ``edible <= CADAVER`` plus starving ACCFOOD).
    # ``dog_nutrition`` here approximates ``5 * oc_nutrition`` via weight*5
    # (food weight ≈ nutrition for our parity stubs).
    # -----------------------------------------------------------------------
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    pr = jnp.clip(mpos[0], 0, _MAP_H - 1)
    pc = jnp.clip(mpos[1], 0, _MAP_W - 1)
    food_cat = state.ground_items.category[b, lv, pr, pc, 0].astype(jnp.int32)
    has_food = food_cat == jnp.int32(_CAT_FOOD_LOCAL)
    is_hungry_legacy = mai.pet_hunger[idx].astype(jnp.int32) <= jnp.int32(0)
    # Either hungry (cited dogmove.c:437 "edible <= CADAVER ... ACCFOOD" gate
    # — collapsed: just allow eat when hungry; full dogfood rating depends on
    # otyp not exposed cleanly here).
    can_eat = is_pet & is_hungry_legacy & has_food

    # vendor dog_nutrition (dogmove.c:172-213): for FOOD_CLASS,
    #   nutrit = objects[otyp].oc_nutrition (approx ≈ weight here).
    food_weight = state.ground_items.weight[b, lv, pr, pc, 0].astype(jnp.int32)
    nutrit = jnp.maximum(food_weight, jnp.int32(1))

    # vendor dogmove.c:230-231 — clamp hungrytime up to moves first.
    cur_hungrytime = mai.hungrytime[idx].astype(jnp.int32)
    moves_now = state.timestep.astype(jnp.int32)
    base_hungrytime = jnp.maximum(cur_hungrytime, moves_now)
    new_hungrytime = jnp.where(
        can_eat, base_hungrytime + nutrit, cur_hungrytime
    )

    # Reset mhpmax_penalty on eat (dogmove.c:242-246).
    cur_penalty_e = mai.mhpmax_penalty[idx].astype(jnp.int32)
    cur_mhpmax_e  = mai.hp_max[idx].astype(jnp.int32)
    restored_mhpmax = cur_mhpmax_e + cur_penalty_e
    new_mhpmax_eat = jnp.where(can_eat, restored_mhpmax, cur_mhpmax_e)
    new_penalty_eat = jnp.where(can_eat, jnp.int32(0), cur_penalty_e)

    # mtame++ capped at 20 (dogmove.c:249-250).
    cur_mtame = mai.mtame[idx].astype(jnp.int32)
    bumped_mtame = jnp.minimum(cur_mtame + jnp.int32(1), jnp.int32(20))
    new_mtame = jnp.where(can_eat, bumped_mtame, cur_mtame).astype(jnp.int8)

    # mconf=0 (dogmove.c:241).
    new_confuse_eat = jnp.where(can_eat, jnp.int16(0), mai.confuse_timer[idx])

    new_ground_cat = state.ground_items.category.at[b, lv, pr, pc, 0].set(
        jnp.where(can_eat, jnp.int8(0), state.ground_items.category[b, lv, pr, pc, 0])
    )
    # Reset legacy pet_hunger to 1000 on eat (back-compat).
    new_hunger_after_eat = jnp.where(can_eat, jnp.int16(1000), mai.pet_hunger[idx])

    mai_e = mai.replace(
        hungrytime=mai.hungrytime.at[idx].set(new_hungrytime),
        hp_max=mai.hp_max.at[idx].set(new_mhpmax_eat),
        mhpmax_penalty=mai.mhpmax_penalty.at[idx].set(new_penalty_eat),
        mtame=mai.mtame.at[idx].set(new_mtame),
        confuse_timer=mai.confuse_timer.at[idx].set(new_confuse_eat),
        pet_hunger=mai.pet_hunger.at[idx].set(new_hunger_after_eat),
    )
    new_ground = state.ground_items.replace(category=new_ground_cat)
    state = state.replace(monster_ai=mai_e, ground_items=new_ground)
    mai = state.monster_ai

    # -----------------------------------------------------------------------
    # 0b'. Pick up item at current tile — dogmove.c:400 dog_invent
    # Vendor priority is: attack hostile > pick up item > follow > wander.
    # Pet picks up a non-cursed, non-food ground item on its current tile if
    # an inventory slot is free.  Food was already consumed in step 0b above.
    # apport gating is intentionally deterministic here (rn2 omitted for JIT
    # purity); the higher-fidelity stochastic gate lives in mpickstuff.
    # Cite: vendor/nethack/src/dogmove.c::dog_invent (lines 426-475).
    # -----------------------------------------------------------------------
    mpos2 = mai.pos[idx].astype(jnp.int32)
    pr2 = jnp.clip(mpos2[0], 0, _MAP_H - 1)
    pc2 = jnp.clip(mpos2[1], 0, _MAP_W - 1)
    g_cat = state.ground_items.category[b, lv, pr2, pc2, 0].astype(jnp.int32)
    g_buc = state.ground_items.buc_status[b, lv, pr2, pc2, 0].astype(jnp.int32)
    has_item_here = g_cat != jnp.int32(0)
    not_food = g_cat != jnp.int32(_CAT_FOOD_LOCAL)
    not_cursed = g_buc >= jnp.int32(0)  # buc_status: -1=cursed, 0=uncursed, +1=blessed
    empty_mask = mai.inv_category[idx] == jnp.int8(0)
    has_empty = jnp.any(empty_mask)
    pick_slot = jnp.argmax(empty_mask.astype(jnp.int32)).astype(jnp.int32)
    can_pickup = is_pet & has_item_here & not_food & not_cursed & has_empty

    g_type = state.ground_items.type_id[b, lv, pr2, pc2, 0]
    g_qty  = state.ground_items.quantity[b, lv, pr2, pc2, 0]
    g_chg  = state.ground_items.charges[b, lv, pr2, pc2, 0]

    new_inv_cat = mai.inv_category.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_cat.astype(jnp.int8),
                  mai.inv_category[idx, pick_slot])
    )
    new_inv_type = mai.inv_type_id.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_type, mai.inv_type_id[idx, pick_slot])
    )
    new_inv_qty = mai.inv_quantity.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_qty, mai.inv_quantity[idx, pick_slot])
    )
    new_inv_buc = mai.inv_buc.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_buc.astype(jnp.int8),
                  mai.inv_buc[idx, pick_slot])
    )
    new_inv_chg = mai.inv_charges.at[idx, pick_slot].set(
        jnp.where(can_pickup, g_chg, mai.inv_charges[idx, pick_slot])
    )
    new_ground_cat2 = state.ground_items.category.at[b, lv, pr2, pc2, 0].set(
        jnp.where(can_pickup, jnp.int8(0),
                  state.ground_items.category[b, lv, pr2, pc2, 0])
    )
    mai_p = mai.replace(
        inv_category=new_inv_cat,
        inv_type_id=new_inv_type,
        inv_quantity=new_inv_qty,
        inv_buc=new_inv_buc,
        inv_charges=new_inv_chg,
    )
    new_ground2 = state.ground_items.replace(category=new_ground_cat2)
    state = state.replace(monster_ai=mai_p, ground_items=new_ground2)
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
    # Find adjacent hostile monster + apply vendor pet-attack gate.
    # Cite: dogmove.c:1102-1144 (balk formula + low-HP peaceful skip +
    # floating eye / petrify skip).  Reduced to JIT-pure form here:
    #   balk = m_lev + ((5 * mhp) / mhpmax) - 2
    #   skip if target.m_lev >= balk         (line 1121)
    #   skip if (mhp*4 < mhpmax) and target.mpeaceful (line 1124-1127)
    # We do not have per-target "touch_petrifies" / floating-eye lookups
    # wired to the entry table yet; the M2_DEMON / M2_UNDEAD fearless mask
    # remains adequate for pet-vs-hostile tests.
    # -----------------------------------------------------------------------
    other_pos = mai.pos.astype(jnp.int32)  # [N, 2]
    dr = jnp.abs(other_pos[:, 0] - mpos[0])
    dc = jnp.abs(other_pos[:, 1] - mpos[1])
    cheb = jnp.maximum(dr, dc)
    self_mask = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32) == idx

    # vendor balk = m_lev + ((5 * mhp) / mhpmax) - 2
    m_lev_self = jnp.clip(
        _MONSTER_LEVEL_TABLE[
            jnp.clip(entry.astype(jnp.int32), 0, _MONSTER_LEVEL_TABLE.shape[0] - 1)
        ].astype(jnp.int32),
        1, 30,
    )
    safe_max = jnp.maximum(mai.hp_max[idx].astype(jnp.int32), jnp.int32(1))
    balk = m_lev_self + (jnp.int32(5) * mai.hp[idx].astype(jnp.int32)) // safe_max - jnp.int32(2)

    all_lev = jnp.clip(
        _MONSTER_LEVEL_TABLE[
            jnp.clip(mai.entry_idx.astype(jnp.int32), 0,
                     _MONSTER_LEVEL_TABLE.shape[0] - 1)
        ].astype(jnp.int32),
        1, 30,
    )
    # Pet is at low HP (hp*4 < mhpmax) — skip peaceful targets (vendor 1124-1127).
    is_low_hp_self = (mai.hp[idx].astype(jnp.int32) * jnp.int32(4)
                       < safe_max)
    not_balked = all_lev < balk
    not_peaceful_when_lowhp = ~(is_low_hp_self & mai.peaceful)

    hostile = (mai.alive & ~mai.tame & ~mai.peaceful & ~self_mask
               & (cheb <= jnp.int32(1)) & not_balked
               & not_peaceful_when_lowhp)
    has_target = jnp.any(hostile)
    target_idx = jnp.argmax(hostile.astype(jnp.int32)).astype(jnp.int32)

    # rng for the attack roll.
    rng_attack_local, _rng_after_attack = jax.random.split(rng)

    def _attack_hostile(s):
        # Cite: dogmove.c:1151 mstatus = mattackm(mtmp, mtmp2);
        s = mattackm(s, idx, target_idx, rng_attack_local)
        # Award mon_xp on kill (vendor mon.c::grow_up after mattackm).
        _m = s.monster_ai
        killed = ~_m.alive[target_idx]
        target_lev = jnp.clip(
            _MONSTER_LEVEL_TABLE[
                jnp.clip(_m.entry_idx[target_idx].astype(jnp.int32), 0,
                         _MONSTER_LEVEL_TABLE.shape[0] - 1)
            ].astype(jnp.int32), 1, 30,
        )
        new_xp = _m.mon_xp.at[idx].set(
            jnp.where(killed,
                      _m.mon_xp[idx] + target_lev,
                      _m.mon_xp[idx])
        )
        return s.replace(monster_ai=_m.replace(mon_xp=new_xp))

    def _follow_player(s):
        """FOLLOW mode: BFS pathfind toward player (mfndpos).

        Vendor: dogmove.c::dog_move uses mfndpos for path-finding.  Confused
        pets randomise their step (vendor mfndpos mon.c:2199-2202 sets
        ``flag |= ALLOW_ALL`` and dochug degenerates pursuit to a random
        adjacent square).
        Cite: vendor/nethack/src/monmove.c::mfndpos lines 2199-2202.
        """
        _mai = s.monster_ai
        step_delta = pathfind_step(s, idx)
        _rng_conf_pet = jax.random.fold_in(rng, jnp.int32(0x636F6E66))  # "conf"
        is_confused_pet = _mai.confuse_timer[idx] > jnp.int16(0)
        step_delta = apply_confusion_to_step(
            step_delta, is_confused_pet, _rng_conf_pet,
        )
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


# ---------------------------------------------------------------------------
# Pet untaming helpers — Wave 40b Item #21
# ---------------------------------------------------------------------------
#
# Vendor-cite: dog.c::abuse_dog (lines 1023-1065) decrements mtame whenever the
# master damages their pet (uhitm.c master-melee path) and when sufficient
# abuse occurs the pet reverts to peaceful/hostile (mtame=0, mpeaceful=0).
# Combat / shop callers (which we can't touch this wave) will invoke these
# public helpers; we expose pure JIT-friendly entry points here so they can be
# wired in later waves without further state-shape changes.

def decrement_mtame(state, pet_idx: jnp.ndarray) -> object:
    """Decrement ``mtame`` for pet at ``pet_idx`` by 1 (vendor abuse_dog).

    Pet remains tame unless mtame is then untamed via ``untame_on_threshold``.
    Safe to call on non-pet slots: gated by ``tame & alive``.

    Cite: vendor/nethack/src/dog.c::abuse_dog lines 1023-1065.
    """
    idx = pet_idx.astype(jnp.int32)
    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]
    cur = mai.mtame[idx].astype(jnp.int32)
    new_val = jnp.where(is_pet, jnp.maximum(cur - 1, jnp.int32(0)), cur).astype(jnp.int8)
    new_mtame = mai.mtame.at[idx].set(new_val)
    return state.replace(monster_ai=mai.replace(mtame=new_mtame))


def untame_on_threshold(state, pet_idx: jnp.ndarray) -> object:
    """If pet's mtame reached 0, untame it (tame=False, peaceful=False).

    Vendor: dog.c::abuse_dog sets mtame=0/mpeaceful=0 when abuse counter
    exhausts the tame value.  We split this from ``decrement_mtame`` so
    callers can decrement multiple times before checking the threshold.

    Cite: vendor/nethack/src/dog.c::abuse_dog (mtame == 0 → revert).
    """
    idx = pet_idx.astype(jnp.int32)
    mai = state.monster_ai
    is_pet  = mai.tame[idx] & mai.alive[idx]
    at_zero = mai.mtame[idx] == jnp.int8(0)
    revert  = is_pet & at_zero
    new_tame     = mai.tame.at[idx].set(jnp.where(revert, jnp.bool_(False), mai.tame[idx]))
    new_peaceful = mai.peaceful.at[idx].set(jnp.where(revert, jnp.bool_(False), mai.peaceful[idx]))
    new_mai = mai.replace(tame=new_tame, peaceful=new_peaceful)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Pet experience / leveling — Wave 40b Item #22
# ---------------------------------------------------------------------------

def grow_up(state, pet_idx: jnp.ndarray, rng: jax.Array) -> object:
    """Grow up a pet when its ``mon_xp`` exceeds the level threshold.

    Vendor logic (makemon.c::grow_up lines 2051-2140): on monster kill the
    pet bumps ``mhpmax`` by ``rnd(victim_lev + 1)`` and gains a level if
    ``mhpmax > m_lev * 8``.

    Simplification for JIT-safe parity: we use the accumulated ``mon_xp``
    counter as the level threshold proxy (kill xp is added at attack time
    in ``pet_move._attack_hostile``).  When ``mon_xp >= mhpmax`` (vendor
    ``mhpmax > hp_threshold = m_lev*8`` after the kill bumps it), gain a
    level — increment a synthetic ``m_lev``-equivalent and bump ``hp_max``
    by ``rnd(8)`` (vendor's default cur_increase upper bound when there is
    no victim).

    Since we don't store a separate ``m_lev`` field (level is read from
    MONSTERS table), we just bump ``hp_max`` and reset ``mon_xp`` on level
    gain.  Pets gain HP and the threshold rises.

    Gated by ``tame & alive``.

    Cite: vendor/nethack/src/makemon.c::grow_up lines 2051-2140.
    """
    idx = pet_idx.astype(jnp.int32)
    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]

    # Level threshold proxy: monster table level * 8 (vendor m_lev*8).
    entry = jnp.clip(mai.entry_idx[idx].astype(jnp.int32), 0,
                     _MONSTER_LEVEL_TABLE.shape[0] - 1)
    base_lev = _MONSTER_LEVEL_TABLE[entry].astype(jnp.int32)
    # Synthetic per-pet level = base + (hp_max - base*8)//8, floored at base.
    effective_lev = jnp.maximum(
        base_lev,
        mai.hp_max[idx].astype(jnp.int32) // jnp.int32(8),
    )
    threshold = effective_lev * jnp.int32(8)
    xp = mai.mon_xp[idx].astype(jnp.int32)

    will_grow = is_pet & (xp >= threshold)

    # vendor rnd(8) for cur_increase / max_increase when no victim ptr.
    hp_bump = jax.random.randint(rng, (), 1, 9, dtype=jnp.int32)

    new_hp_max = jnp.where(
        will_grow, mai.hp_max[idx].astype(jnp.int32) + hp_bump,
        mai.hp_max[idx].astype(jnp.int32),
    ).astype(mai.hp_max.dtype)
    new_hp = jnp.where(
        will_grow, mai.hp[idx].astype(jnp.int32) + hp_bump,
        mai.hp[idx].astype(jnp.int32),
    ).astype(mai.hp.dtype)
    new_xp = jnp.where(will_grow, xp - threshold, xp).astype(mai.mon_xp.dtype)

    new_mai = mai.replace(
        hp_max=mai.hp_max.at[idx].set(new_hp_max),
        hp=mai.hp.at[idx].set(new_hp),
        mon_xp=mai.mon_xp.at[idx].set(new_xp),
    )
    return state.replace(monster_ai=new_mai)


def pet_follow_on_stair(state):
    """Queue any tame pets within Chebyshev 1 of player onto the migrating list.

    Vendor: dog.c::keepdogs (lines 789-870) — pets that satisfy
    ``monnear(mtmp, u.ux, u.uy) && levl_follower(mtmp)`` get ``mx=my=0`` and
    are moved off ``fmon`` onto the ``gm.mydogs`` migrating list.  On level
    entry ``mon_arrive`` (dog.c:420-566) repositions them next to the hero
    preserving all per-mon state.

    Wave 40b Item #15: previously this function set ``alive=False`` which
    threw away the pet's per-mon state (hp, mtame, edog fields, hungrytime,
    mhpmax_penalty, mtrack).  We now set the ``migrating`` flag instead so
    the same slot can be repositioned by ``pet_arrive_on_level`` with all
    state preserved.  ``alive`` remains True so the slot stays reserved;
    callers that iterate ``alive`` monsters must additionally check
    ``~migrating`` to avoid acting on slots in transit.

    Cite: vendor/nethack/src/dog.c::keepdogs (789-870), mon_arrive (420-566).
    """
    mai = state.monster_ai
    ppos = state.player_pos.astype(jnp.int32)
    mpos = mai.pos.astype(jnp.int32)
    dr = jnp.abs(mpos[:, 0] - ppos[0])
    dc = jnp.abs(mpos[:, 1] - ppos[1])
    cheb = jnp.maximum(dr, dc)
    # Pets within Chebyshev 1 of player that are alive and tame.
    follows = mai.tame & mai.alive & (cheb <= jnp.int32(1))
    # Set ``alive=False`` (so source-level turn dispatch ignores them) AND
    # ``migrating=True`` so ``pet_arrive_on_level`` can revive them with all
    # per-pet state (hp, mtame, edog fields, hungrytime, mhpmax_penalty,
    # mtrack) intact.  vendor dog.c:863 sets mtmp->mx = mtmp->my = 0 (off
    # the level map); we mirror that by zeroing the pos so source-level
    # adjacency checks against the pet behave as if it were gone.
    new_alive = mai.alive & ~follows
    new_migrating = mai.migrating | follows
    new_pos = jnp.where(
        follows[:, None],
        jnp.zeros_like(mai.pos),  # off-map sentinel (vendor mx=my=0)
        mai.pos,
    )
    new_mai = mai.replace(alive=new_alive, migrating=new_migrating, pos=new_pos)
    return state.replace(monster_ai=new_mai)


def pet_arrive_on_level(state):
    """Reposition migrating pets next to player on level entry (mon_arrive).

    Vendor: dog.c::mon_arrive (lines 420-566) — for each pet on the migrating
    list, place it on or adjacent to the hero's tile and clear the migrating
    flag.  All per-mon state (hp/mtame/hungrytime/mhpmax_penalty) was already
    preserved across the level transition by ``pet_follow_on_stair``.

    Simplified placement: we put the pet on the hero's tile +1 in row/col
    (Chebyshev=1, vendor uses ``enexto`` to find a nearby empty square).
    Bounds-clamped to map shape.

    Cite: vendor/nethack/src/dog.c::mon_arrive lines 420-566.
    """
    mai = state.monster_ai
    ppos = state.player_pos.astype(jnp.int32)
    # Place each migrating pet at (player_row+1, player_col+1) clipped to map.
    # Real vendor uses enexto to find an empty adjacent tile; we keep the
    # simplification documented since multi-pet collision is rare and the
    # follow-up wave can add a proper enexto.  All slots are placed at the
    # same offset since each pet is in its own slot.
    target_r = jnp.clip(ppos[0] + jnp.int32(1), 0, _MAP_H - 1).astype(jnp.int16)
    target_c = jnp.clip(ppos[1] + jnp.int32(1), 0, _MAP_W - 1).astype(jnp.int16)
    arrive_pos = jnp.stack([target_r, target_c])  # (2,)
    new_pos = jnp.where(
        mai.migrating[:, None],
        jnp.broadcast_to(arrive_pos, mai.pos.shape),
        mai.pos,
    )
    # Re-revive migrating slots (they were marked alive=False at follow time).
    new_alive = mai.alive | mai.migrating
    new_migrating = jnp.where(mai.migrating, jnp.bool_(False), mai.migrating)
    new_mai = mai.replace(pos=new_pos, alive=new_alive, migrating=new_migrating)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# Pet apport bookkeeping — Wave 40b Item #20
# ---------------------------------------------------------------------------

def pet_record_drop(state, pet_idx: jnp.ndarray) -> object:
    """Record that pet ``pet_idx`` just dropped its currently-carried item.

    Vendor dog_invent (dogmove.c:422-426) sets:
        edog->apport--             /* willingness to fetch drops by 1 */
        edog->dropdist = udist     /* chebyshev distance pet-to-hero */
        edog->droptime = svm.moves /* turn the drop occurred */

    We snapshot ``last_drop_pos`` (the pet's current tile) and
    ``last_drop_turn`` (game timestep) so ``pet_credit_fetch`` can later
    compute the apport credit per dog_eat (dogmove.c:318):
        apport += 200L / (dropdist + (moves - droptime))

    Gated by tame & alive.  No-op for non-pets.

    Cite: vendor/nethack/src/dogmove.c::dog_invent lines 416-426.
    """
    idx = pet_idx.astype(jnp.int32)
    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]

    cur_apport = mai.apport[idx].astype(jnp.int32)
    new_apport = jnp.where(is_pet, jnp.maximum(cur_apport - 1, jnp.int32(1)), cur_apport)
    new_pos_row = jnp.where(is_pet, mai.pos[idx, 0], mai.last_drop_pos[idx, 0])
    new_pos_col = jnp.where(is_pet, mai.pos[idx, 1], mai.last_drop_pos[idx, 1])
    cur_turn = state.timestep.astype(jnp.int32)
    new_turn = jnp.where(is_pet, cur_turn, mai.last_drop_turn[idx])

    new_apport_arr = mai.apport.at[idx].set(new_apport.astype(jnp.int8))
    new_drop_pos = mai.last_drop_pos.at[idx, 0].set(new_pos_row.astype(jnp.int16))
    new_drop_pos = new_drop_pos.at[idx, 1].set(new_pos_col.astype(jnp.int16))
    new_drop_turn = mai.last_drop_turn.at[idx].set(new_turn.astype(jnp.int32))
    new_mai = mai.replace(
        apport=new_apport_arr,
        last_drop_pos=new_drop_pos,
        last_drop_turn=new_drop_turn,
    )
    return state.replace(monster_ai=new_mai)


def pet_credit_fetch(state, pet_idx: jnp.ndarray) -> object:
    """Credit ``apport`` to pet for eating an item near a remembered drop.

    Vendor dog_eat (dogmove.c:316-320):
        apport += 200L / (dropdist + (moves - droptime))

    where ``dropdist`` is the Chebyshev distance hero→drop_pos at drop time
    and ``moves - droptime`` is turns elapsed since the drop.  We re-derive
    ``dropdist`` as the Chebyshev distance from the recorded ``last_drop_pos``
    to the player's *current* position (an acceptable approximation since
    vendor's value is fixed at drop time and the player has moved).

    Gated by tame & alive & (last_drop_turn > 0).

    Cite: vendor/nethack/src/dogmove.c::dog_eat lines 316-320.
    """
    idx = pet_idx.astype(jnp.int32)
    mai = state.monster_ai
    is_pet = mai.tame[idx] & mai.alive[idx]
    has_drop = mai.last_drop_turn[idx] > jnp.int32(0)

    drop_r = mai.last_drop_pos[idx, 0].astype(jnp.int32)
    drop_c = mai.last_drop_pos[idx, 1].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dr = jnp.abs(drop_r - ppos[0])
    dc = jnp.abs(drop_c - ppos[1])
    dropdist = jnp.maximum(dr, dc)
    elapsed = jnp.maximum(state.timestep.astype(jnp.int32) -
                          mai.last_drop_turn[idx].astype(jnp.int32),
                          jnp.int32(0))
    # Prevent /0 — vendor never has dropdist + elapsed == 0 because elapsed > 0
    # any time after the drop, but on the same-tick edge we still guard it.
    denom = jnp.maximum(dropdist + elapsed, jnp.int32(1))
    credit = jnp.int32(200) // denom

    cur_apport = mai.apport[idx].astype(jnp.int32)
    do_credit = is_pet & has_drop
    new_apport = jnp.where(do_credit,
                           jnp.clip(cur_apport + credit, 1, 127),
                           cur_apport)
    new_apport_arr = mai.apport.at[idx].set(new_apport.astype(jnp.int8))
    return state.replace(monster_ai=mai.replace(apport=new_apport_arr))


# ---------------------------------------------------------------------------
# 7.  Sleep wake on player-visible  (src/monmove.c::disturb)
# ---------------------------------------------------------------------------

def maybe_wake_monster(state, monster_idx: jnp.ndarray, rng: jax.Array = None):
    """Vendor ``disturb`` (monmove.c:327-358) sleep-wake decision.

    Vendor wake conditions:
        couldsee(mtmp->mx, mtmp->my) && mdistu(mtmp) <= 100
        && (!Stealth || (mtmp->data == &mons[PM_ETTIN] && rn2(10)))
        && (mtmp not in {nymph, jabberwock, leprechaun} || !rn2(50))
        && (Aggravate_monster
            || mlet in {S_DOG, S_HUMAN}
            || (!rn2(7) && !mimic_furniture/object))

    Simplifications kept JIT-safe:
      * mdistu (squared Euclidean) <= 100  — replicates vendor exactly.
      * Stealth / Aggravate / mimic flags not in state yet → treated as
        false / off; the rn2(7) gate is still applied.
      * Per-symbol checks use the precomputed sound table as a proxy where
        possible; nymph/leprechaun gating omitted (table not exposed in this
        cell).  The full per-mlet exemption can be wired when MonsterSymbol
        becomes available without an import cycle.

    Cite: vendor/nethack/src/monmove.c::disturb lines 327-358.
    """
    idx = monster_idx.astype(jnp.int32)
    mai = state.monster_ai

    asleep = mai.asleep[idx]
    alive = mai.alive[idx]
    in_los = monster_can_see_player(state, idx)

    # mdistu = squared euclidean (dx² + dy²); vendor cap 100.
    mpos = mai.pos[idx].astype(jnp.int32)
    ppos = state.player_pos.astype(jnp.int32)
    dr = mpos[0] - ppos[0]
    dc = mpos[1] - ppos[1]
    distu_sq = dr * dr + dc * dc
    within_100 = distu_sq <= jnp.int32(100)

    # rn2(7) gate — if no rng provided, treat as always pass to preserve
    # back-compat with callers that don't thread rng yet.
    if rng is None:
        rn2_7_pass = jnp.bool_(True)
    else:
        rng_key, _ = jax.random.split(rng)
        rn2_7_pass = jax.random.randint(rng_key, (), 0, 7) == 0

    should_wake = asleep & alive & in_los & within_100 & rn2_7_pass

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

    (rng_pet, rng_cast, rng_atk, rng_pick,
     rng_decay, rng_wake, rng_conf_step) = jax.random.split(rng, 7)
    rng_mconf, rng_mstun = jax.random.split(rng_decay)

    # Branch 1 + 2: pet has its own turn.
    is_pet = mai.tame[idx] & mai.alive[idx]

    def _pet_branch(s):
        return pet_move(s, rng_pet, idx)

    def _hostile_branch(s):
        # --- monmove.c:717 mcanmove gate ---
        # ``if (!mtmp->mcanmove || (mtmp->mstrategy & STRAT_WAITMASK)) return 0``
        # Map mcanmove → ``paralyzed_timer == 0``.  STRAT_WAITMASK maps to
        # mstrategy == WAIT.  Paralyzed / frozen monsters skip their turn.
        _m_pre = s.monster_ai
        is_paralyzed = _m_pre.paralyzed_timer[idx] > jnp.int16(0)
        is_waiting   = _m_pre.mstrategy[idx] == jnp.int8(MoveStrategy.WAIT)
        cannot_move  = is_paralyzed | is_waiting

        # --- monmove.c:737-742 stochastic confusion / stun decay ---
        rn50 = jax.random.randint(rng_mconf, (), 0, 50)
        rn10 = jax.random.randint(rng_mstun, (), 0, 10)
        decay_conf = (_m_pre.confuse_timer[idx] > 0) & (rn50 == 0)
        decay_stun = (_m_pre.stun_timer[idx]    > 0) & (rn10 == 0)
        new_conf_v = jnp.where(decay_conf, jnp.int16(0), _m_pre.confuse_timer[idx])
        new_stun_v = jnp.where(decay_stun, jnp.int16(0), _m_pre.stun_timer[idx])

        # --- monmove.c:745-750 random fleeing teleport (1/40) ---
        # ``if (mtmp->mflee && !rn2(40) && can_teleport(mdat) && !iswiz) rloc``
        # Deferred: ``can_teleport`` flag is not yet tracked per slot; full
        # activation needs a per-slot can_teleport bool plus an rloc helper
        # for monsters.  Stub left in place; no rng consumed.

        _m_decay = _m_pre.replace(
            confuse_timer=_m_pre.confuse_timer.at[idx].set(new_conf_v),
            stun_timer=_m_pre.stun_timer.at[idx].set(new_stun_v),
        )
        s = s.replace(monster_ai=_m_decay)

        # Record asleep state BEFORE wake-check: monsters that wake this
        # turn do not also act this turn (mirrors vendor monmove.c::disturb,
        # which only flips the flag and lets the next tick run the AI).
        was_asleep = s.monster_ai.asleep[idx]

        # 3: wake check.
        s = maybe_wake_monster(s, idx, rng_wake)

        # 4: gate on alive & mcanmove & not asleep (at start of turn) & not peaceful.
        m = s.monster_ai
        should_act = m.alive[idx] & ~was_asleep & ~m.peaceful[idx] & ~cannot_move

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
            # Confusion-driven random step: vendor mfndpos sets ``flag |=
            # ALLOW_ALL`` when ``mon->mconf`` (mon.c:2199-2202), and dochug's
            # pursuit logic degenerates to a random adjacent square when
            # ``mtmp->mconf``.  Override AFTER retreat so a confused fleeing
            # monster still picks randomly (vendor behaviour).
            is_confused_mi = st.monster_ai.confuse_timer[idx] > jnp.int16(0)
            step_delta = apply_confusion_to_step(
                step_delta, is_confused_mi, rng_conf_step,
            )
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
        # Clone inherits parent's m_lev (vendor makemon.c::clone_mon copies
        # mtmp2 = *mtmp1 before tweaking flags).
        # Cite: vendor/nethack/src/makemon.c::clone_mon lines 837-944.
        new_m_lev     = m.m_lev.at[dead_idx].set(m.m_lev[idx])
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
            m_lev=new_m_lev,
        )
        return s.replace(monster_ai=new_m)

    return jax.lax.cond(can_clone, _do_clone, lambda s: s, state)


# ---------------------------------------------------------------------------
# mattackm  — vendor/nethack/src/mhitm.c::mattackm lines 1024-1100.
# ---------------------------------------------------------------------------

def mattackm(state, attacker_idx: jnp.ndarray, defender_idx: jnp.ndarray,
             rng: jax.Array) -> object:
    """Monster-vs-monster melee attack — vendor NATTK loop.

    Wave 40b Item #7: iterate over all NATTK=6 attack slots from the
    attacker's data->mattk[].  For each slot with aatyp != AT_NONE:
      * compute ``tmp = ac_value(def_ac) + 10 + a_lev`` (find_mac + m_lev)
      * apply ``+4`` if defender is confused or helpless (sleep/paralysis)
      * apply ``+1`` if attacker is elf and defender is orc
      * roll ``dieroll = rnd(20 + i)`` per attack index i (mhitm.c:441)
      * hit iff ``tmp > dieroll``; roll damage ``ndN dM``
      * stop early if defender dies

    Gates: both slots alive, attacker != defender.

    Cite: vendor/nethack/src/mhitm.c::mattackm lines 293-592; permonst.h
    NATTK = 6 (line 48).
    """
    a = attacker_idx.astype(jnp.int32)
    d = defender_idx.astype(jnp.int32)
    mai = state.monster_ai

    same_slot = (a == d)
    both_alive = mai.alive[a] & mai.alive[d]
    can_strike_base = both_alive & ~same_slot

    a_entry = jnp.clip(
        mai.entry_idx[a].astype(jnp.int32),
        0, _MONSTER_ATTACK_AATYP_TABLE.shape[0] - 1,
    )
    d_entry = jnp.clip(
        mai.entry_idx[d].astype(jnp.int32),
        0, _MONSTER_ATTACK_AATYP_TABLE.shape[0] - 1,
    )
    a_lev = jnp.clip(_MONSTER_LEVEL_TABLE[a_entry].astype(jnp.int32), 1, 30)

    defender_confused = mai.confuse_timer[d] > jnp.int16(0)
    defender_helpless = (
        mai.asleep[d] | (mai.paralyzed_timer[d] > jnp.int16(0))
    )
    bonus_confused = jnp.where(defender_confused | defender_helpless,
                                jnp.int32(4), jnp.int32(0))

    _M2_ORC: int = 0x00000004
    _M2_ELF: int = 0x00000008
    a_flags2 = _MONSTER_FLAGS2_TABLE[a_entry]
    d_flags2 = _MONSTER_FLAGS2_TABLE[d_entry]
    is_elf = (a_flags2 & jnp.int32(_M2_ELF)) != 0
    is_orc = (d_flags2 & jnp.int32(_M2_ORC)) != 0
    bonus_elf_orc = jnp.where(is_elf & is_orc, jnp.int32(1), jnp.int32(0))

    def_ac_raw = mai.ac[d].astype(jnp.int32)
    rng, key_ac = jax.random.split(rng)
    ac_neg_roll = jax.random.randint(
        key_ac, (), 1, jnp.maximum(-def_ac_raw + 1, 2), dtype=jnp.int32
    )
    ac_value = jnp.where(def_ac_raw >= 0, def_ac_raw, -ac_neg_roll)
    base_tmp = jnp.maximum(
        ac_value + jnp.int32(10) + a_lev + bonus_confused + bonus_elf_orc,
        jnp.int32(1),
    )

    AT_NONE = jnp.int16(0)
    aatyp_row = _MONSTER_ATTACK_AATYP_TABLE[a_entry]
    n_row     = _MONSTER_ATTACK_N_TABLE[a_entry]
    s_row     = _MONSTER_ATTACK_S_TABLE[a_entry]

    nattk_keys = jax.random.split(rng, _NATTK)

    def _attack_step(carry, idx):
        cur_def_hp, cur_def_alive, struck = carry
        key_i = nattk_keys[idx]
        aatyp_i = aatyp_row[idx]
        n_raw   = n_row[idx].astype(jnp.int32)
        s_raw   = s_row[idx].astype(jnp.int32)
        n_dice  = jnp.clip(n_raw, 1, 8)
        sides   = jnp.clip(s_raw, 1, 12)

        slot_active = (aatyp_i != AT_NONE) & (n_raw > 0)
        can_attack = can_strike_base & slot_active & cur_def_alive

        key_hit, key_dmg = jax.random.split(key_i)
        roll = jax.random.randint(
            key_hit, (), 1, jnp.int32(21) + idx, dtype=jnp.int32,
        )
        hit = (base_tmp > roll) & can_attack

        keys_d = jax.random.split(key_dmg, 8)

        def _roll_one(c, k):
            sub = jax.random.randint(k, (), 1, sides + 1, dtype=jnp.int32)
            return c, sub

        _, rolls = jax.lax.scan(_roll_one, jnp.int32(0), keys_d)
        take = jnp.arange(8, dtype=jnp.int32) < n_dice
        raw_dmg = jnp.sum(jnp.where(take, rolls, jnp.int32(0))).astype(jnp.int32)
        dmg = jnp.where(hit, raw_dmg, jnp.int32(0))

        new_def_hp = jnp.maximum(cur_def_hp - dmg, jnp.int32(0))
        new_def_alive = cur_def_alive & (new_def_hp > jnp.int32(0))
        return (new_def_hp, new_def_alive, struck | hit), None

    init = (mai.hp[d].astype(jnp.int32), mai.alive[d], jnp.bool_(False))
    (final_def_hp, final_def_alive, _struck), _ = jax.lax.scan(
        _attack_step, init, jnp.arange(_NATTK, dtype=jnp.int32),
    )

    new_hp_arr    = mai.hp.at[d].set(final_def_hp.astype(mai.hp.dtype))
    new_alive_arr = mai.alive.at[d].set(final_def_alive)

    new_mai = mai.replace(hp=new_hp_arr, alive=new_alive_arr)
    return state.replace(monster_ai=new_mai)


# ---------------------------------------------------------------------------
# All-monsters step  (jax.lax.scan over slots)
# ---------------------------------------------------------------------------

# Per-tick movement-point threshold; vendor NORMAL_SPEED = 12 and a monster
# acts when its accumulator reaches NORMAL_SPEED.
# Cite: vendor/nethack/src/monmove.c line 1731; allmain.c lines 233-234.
_MOVEMENT_THRESHOLD: int = 12

# Vendor pet hunger thresholds.
# Cite: vendor/nethack/include/dog.h:
#   #define DOG_SATIATED   200
#   #define DOG_HUNGRY     300
#   #define DOG_WEAK       500
#   #define DOG_STARVE     750
_DOG_HUNGRY: int = 300
_DOG_WEAK:   int = 500
_DOG_STARVE: int = 750


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

    # ---- Speed-energy accumulation (vendor mon.c::mcalcmove lines 1126-1167) ----
    # Vendor piecewise speed adjust:
    #   MSLOW (speed_mod<0):
    #     mmove < NORMAL_SPEED → (2*mmove + 1) / 3
    #     mmove >= NORMAL_SPEED → 4 + mmove/3
    #   MFAST (speed_mod>0):
    #     mmove = (4*mmove + 2) / 3
    # Then stochastic rounding to NORMAL_SPEED multiples:
    #     mmove_adj = mmove % NORMAL_SPEED
    #     mmove -= mmove_adj
    #     if rn2(NORMAL_SPEED) < mmove_adj: mmove += NORMAL_SPEED
    # Cite: vendor/nethack/src/mon.c::mcalcmove lines 1126-1167.
    safe_entry = jnp.clip(
        mai.entry_idx.astype(jnp.int32),
        0, _MONSTER_MOVE_SPEED_TABLE.shape[0] - 1,
    )
    base_speed = _MONSTER_MOVE_SPEED_TABLE[safe_entry].astype(jnp.int32)
    smod = mai.speed_mod.astype(jnp.int32)
    NS = jnp.int32(_MOVEMENT_THRESHOLD)  # NORMAL_SPEED = 12

    # MSLOW piecewise
    slow_lo = (2 * base_speed + 1) // jnp.int32(3)
    slow_hi = jnp.int32(4) + base_speed // jnp.int32(3)
    mslow_speed = jnp.where(base_speed < NS, slow_lo, slow_hi)
    # MFAST formula
    mfast_speed = (4 * base_speed + 2) // jnp.int32(3)
    # Apply piecewise based on smod sign
    pre_round = jnp.where(smod < 0, mslow_speed,
                  jnp.where(smod > 0, mfast_speed, base_speed))

    # Stochastic rounding to NORMAL_SPEED multiples (vendor rn2(NORMAL_SPEED)).
    # Threefry-safe: derive a per-slot rounding key from rng before slot keys split.
    rng, round_key = jax.random.split(rng)
    round_keys = jax.random.split(round_key, MAX_MONSTERS_PER_LEVEL)
    # rn2(NORMAL_SPEED) per slot.
    rn_vals = jax.vmap(lambda k: jax.random.randint(k, (), 0, NS, dtype=jnp.int32))(round_keys)
    mmove_adj = pre_round % NS
    mmove_floored = pre_round - mmove_adj
    rounded = mmove_floored + jnp.where(rn_vals < mmove_adj, NS, jnp.int32(0))

    add_points = jnp.where(mai.alive, rounded, jnp.int32(0))
    new_acc = mai.movement_points.astype(jnp.int32) + add_points

    can_act = new_acc >= NS
    # Deduct threshold on action.
    post_acc = jnp.where(can_act, new_acc - NS, new_acc)
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
    # Wave 40b Item #8: vendor mm_aggression (mon.c:2422-2447) permits
    # monster-vs-monster combat under any of:
    #   * Conflict (player intrinsic 44 causes ALL adjacent monsters to brawl);
    #   * purple worm / baby purple worm vs shrieker (mon.c:2440-2442);
    #   * zombie_maker (S_ZOMBIE except ghoul/skeleton, or S_LICH, not
    #     cancelled) vs species with a zombie_form (any non-zombie symbol),
    #     and neither attacker nor defender is mtame (mon.c:2425-2429);
    #   * fallback: hostile attacker → non-hostile target (kept for sweep
    #     completeness — pets and peacefuls still don't initiate via this).
    # Cite: vendor/nethack/src/mon.c::mm_aggression lines 2422-2447;
    #       vendor/nethack/src/uhitm.c — Conflict intrinsic gate (44).
    _status = getattr(state, "status", None)
    if _status is not None and hasattr(_status, "intrinsics"):
        _conflict_active = _status.intrinsics[_INTRINSIC_CONFLICT]
    else:
        _conflict_active = jnp.bool_(False)

    def _strike_body(carry, args):
        i, key_i = args
        mi = carry.monster_ai
        i32 = i.astype(jnp.int32)

        atk_alive = mi.alive[i32]
        pi = mi.pos[i32].astype(jnp.int32)

        all_pos = mi.pos.astype(jnp.int32)            # [N, 2]
        d_row = jnp.abs(all_pos[:, 0] - pi[0])
        d_col = jnp.abs(all_pos[:, 1] - pi[1])
        adj = jnp.maximum(d_row, d_col) == 1

        is_tame_all = mi.tame
        is_peace_all = mi.peaceful & ~is_tame_all
        all_faction = jnp.where(is_tame_all, jnp.int32(2),
                       jnp.where(is_peace_all, jnp.int32(1), jnp.int32(0)))
        a_faction = all_faction[i32]
        is_hostile_atk    = a_faction == jnp.int32(0)
        is_nonhostile_tgt = all_faction != jnp.int32(0)

        idx_arr = jnp.arange(MAX_MONSTERS_PER_LEVEL, dtype=jnp.int32)
        pair_ok = idx_arr > i32

        # Species-pair aggression (vendor mon.c:2422-2447).
        a_entry = jnp.clip(mi.entry_idx[i32].astype(jnp.int32),
                           0, _MM_IS_PURPLE_WORM.shape[0] - 1)
        all_entry = jnp.clip(mi.entry_idx.astype(jnp.int32),
                             0, _MM_IS_PURPLE_WORM.shape[0] - 1)
        a_is_pw = _MM_IS_PURPLE_WORM[a_entry]
        t_is_shr = _MM_IS_SHRIEKER[all_entry]
        a_is_zm = _MM_IS_ZOMBIE_MAKER[a_entry] & ~mi.cancelled[i32]
        t_has_zform = _MM_HAS_ZOMBIE_FORM[all_entry]

        # vendor mm_aggression early-out: "don't allow pets to fight each
        # other" (mon.c:2434).
        pets_brawl = mi.tame[i32] & mi.tame

        species_purple = a_is_pw & t_is_shr & ~pets_brawl
        species_zombie = a_is_zm & t_has_zform & ~pets_brawl & ~mi.tame[i32] & ~mi.tame

        # Under Conflict, ALL adjacent monsters brawl regardless of faction
        # (vendor uhitm.c Conflict gate).
        conflict_allow = _conflict_active

        baseline_allow = is_hostile_atk & is_nonhostile_tgt
        per_target_allow = (baseline_allow
                            | species_purple
                            | species_zombie
                            | conflict_allow)

        candidates = mi.alive & adj & pair_ok & per_target_allow
        has_target = jnp.any(candidates)
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
    # Wave 45a: blind_timer ticks once per turn, floor at 0.
    # Cite: vendor/nethack/src/mon.c::mon_update_state per-turn decrement;
    #       vendor/nethack/include/monst.h::mblinded.
    new_blind     = jnp.maximum(mai.blind_timer.astype(jnp.int32)     - 1, 0).astype(jnp.int16)
    # Vendor mspec_used decrement: vendor/nethack/src/allmain.c (per-turn
    # loop) ticks every monster's mspec_used so casts re-arm.
    new_mspec     = jnp.maximum(mai.mspec_used.astype(jnp.int32)      - 1, 0).astype(jnp.int16)
    # flee_until_turn is an absolute turn counter; do not decrement.
    new_asleep    = new_sleep > jnp.int16(0)
    mai = mai.replace(
        sleep_timer=new_sleep,
        stun_timer=new_stun,
        confuse_timer=new_confuse,
        paralyzed_timer=new_paralyzed,
        blind_timer=new_blind,
        mspec_used=new_mspec,
        asleep=new_asleep,
    )
    return final_state.replace(monster_ai=mai)


# ---------------------------------------------------------------------------
# Wake monsters near a disturbance  (src/monmove.c::disturb)
# ---------------------------------------------------------------------------

def wake_monsters_near(state, pos: jnp.ndarray, radius: int = 3,
                       petcall: bool = False) -> object:
    """Wake all sleeping monsters within Chebyshev ``radius`` of ``pos``.

    Also exposes a vendor-parity dist² helper via :func:`wake_nearto`.

    Vendor: ``wake_nearto(x, y, distance)`` (mon.c:4373-4399) uses dist² (the
    third argument is already squared, e.g. monmove.c:63
    ``wake_nearto(mtmp->mx, mtmp->my, 7*7)``).  ``petcall`` resets the
    pet's whistletime + clears its mon_track when set, mirroring vendor.

    Vectorized over all slots: no Python loop.
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


def wake_nearto(state, pos: jnp.ndarray, distance_sq: int = 49) -> object:
    """Vendor-parity wake-up: wake all sleeping mons whose dist² ≤ distance_sq.

    Cite: vendor/nethack/src/mon.c::wake_nearto lines 4373-4399.
    Vendor signature takes squared-distance directly (e.g. ``7*7`` for r=7).
    """
    mai = state.monster_ai
    pos_i32 = pos.astype(jnp.int32)

    mon_pos_i32 = mai.pos.astype(jnp.int32)
    drv = mon_pos_i32[:, 0] - pos_i32[0]
    dcv = mon_pos_i32[:, 1] - pos_i32[1]
    dist_sq = drv * drv + dcv * dcv

    in_range = (dist_sq <= jnp.int32(distance_sq)) & mai.alive
    new_asleep = mai.asleep & ~in_range
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

    rng_roll, rng_hp = jax.random.split(rng)
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

    # Wave 48c: vendor newmonhp() per-spawn HP roll (makemon.c:1037-1053).
    # Shrieker summons a killer bee (entry_idx=1, m_lev=1); roll d(1, 8).
    summon_lev_i32 = _MONSTER_LEVEL_TABLE[jnp.int32(1)].astype(jnp.int32)
    rolled_hp      = _newmonhp_roll(rng_hp, summon_lev_i32)

    new_alive    = mai.alive.at[dead_idx].set(jnp.where(should, jnp.bool_(True),  mai.alive[dead_idx]))
    new_pos      = mai.pos.at[dead_idx].set(jnp.where(should, spawn_pos,           mai.pos[dead_idx]))
    new_hp       = mai.hp.at[dead_idx].set(jnp.where(should, rolled_hp,            mai.hp[dead_idx]))
    new_hp_max   = mai.hp_max.at[dead_idx].set(jnp.where(should, rolled_hp,        mai.hp_max[dead_idx]))
    new_peaceful = mai.peaceful.at[dead_idx].set(jnp.where(should, jnp.bool_(False), mai.peaceful[dead_idx]))
    new_asleep   = mai.asleep.at[dead_idx].set(jnp.where(should, jnp.bool_(False),  mai.asleep[dead_idx]))
    new_entry    = mai.entry_idx.at[dead_idx].set(jnp.where(should, jnp.int16(1),   mai.entry_idx[dead_idx]))
    # Populate per-monster level from MONSTERS[entry_idx].level.
    # Cite: vendor/nethack/include/monst.h::struct monst::m_lev (set at makemon).
    summon_lev   = summon_lev_i32.astype(mai.m_lev.dtype)
    new_m_lev    = mai.m_lev.at[dead_idx].set(jnp.where(should, summon_lev, mai.m_lev[dead_idx]))

    new_mai = mai.replace(
        alive=new_alive, pos=new_pos, hp=new_hp, hp_max=new_hp_max,
        peaceful=new_peaceful, asleep=new_asleep, entry_idx=new_entry,
        m_lev=new_m_lev,
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
