"""Trap subsystem — placement, triggering, and revelation of floor traps.

Canonical sources:
  vendor/nethack/include/trap.h     — enum trap_types, N_TRAP_TYPES (TRAPNUM=26)
  vendor/nethack/src/trap.c         — dotrap(), mintrap(), maketrap(), deltrap()

Status: Wave 3 — trigger_trap implemented; place_trap / reveal_trap stubs remain.

TODO (later waves):
  Wave 4 (transport / magic traps):
    TRAPDOOR     — delayed HOLE; same as HOLE but can be crawled out of
    LEVEL_TELEP  — teleport victim to random dungeon level
    MAGIC_PORTAL — fixed destination portal (Vibrating Square branch, etc.)
    LANDMINE     — destroys items on tile (Wave 4 item layer)
    RUST_TRAP    — ruin a random worn metal item (Wave 4 item layer)
  Wave 5 (special / endgame):
    VIBRATING_SQUARE — Castle / endgame gateway; requires Amulet logic
    Drawbridge interactions (dbridge.c) — DRAWBRIDGE_UP/DOWN tile traps
    Secret-door discovery via search command (detect.c)
"""
from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------------
# Trap type enumeration (vendor/nethack/include/trap.h :: enum trap_types)
# ---------------------------------------------------------------------------
class TrapType(IntEnum):
    NO_TRAP              =  0
    ARROW_TRAP           =  1
    DART_TRAP            =  2
    ROCKTRAP             =  3
    SQKY_BOARD           =  4
    BEAR_TRAP            =  5
    LANDMINE             =  6
    ROLLING_BOULDER_TRAP =  7
    SLP_GAS_TRAP         =  8
    RUST_TRAP            =  9
    FIRE_TRAP            = 10
    PIT                  = 11
    SPIKED_PIT           = 12
    HOLE                 = 13
    TRAPDOOR             = 14
    TELEP_TRAP           = 15
    LEVEL_TELEP          = 16
    MAGIC_PORTAL         = 17
    WEB                  = 18
    STATUE_TRAP          = 19
    MAGIC_TRAP           = 20
    ANTI_MAGIC           = 21
    POLY_TRAP            = 22
    VIBRATING_SQUARE     = 23
    TRAPPED_DOOR         = 24
    TRAPPED_CHEST        = 25


# TRAPNUM = 26 in trap.h; this is the count of valid trap-type codes.
N_TRAP_TYPES: int = 26  # vendor/nethack/include/trap.h :: TRAPNUM


# ---------------------------------------------------------------------------
# Side-effect encoding (int32[5])
# Index 0 : freeze turns (FROZEN timed status to set)
# Index 1 : sleep turns (SLEEP timed status to set)
# Index 2 : teleport flag (1 = random teleport requested)
# Index 3 : wake monsters flag (1 = wake nearby monsters requested)
# Index 4 : level-descend flag (1 = descend one dungeon level)
#           Set by HOLE / TRAPDOOR — vendor/nethack/src/trap.c::dotrap
#           TT_HOLE / TT_TRAPDOOR cases call goto_level(level+1, ...).
# ---------------------------------------------------------------------------
_SE_FREEZE         = 0
_SE_SLEEP          = 1
_SE_TELE           = 2
_SE_WAKE           = 3
_SE_LEVEL_DESCEND  = 4


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@struct.dataclass
class TrapState:
    """Per-tile trap meta-state across all dungeon levels.

    The authoritative trap *type* on each tile is stored here alongside the
    player's knowledge of that trap.  Physical placement (tile flag) lives in
    the dungeon map; this slice tracks discovery and trigger counters that
    must survive level transitions.

    Fields
    ------
    trap_type  : int8 array [num_levels, map_h, map_w]
                 TrapType value; 0 (NO_TRAP) means no trap present.
    revealed   : bool array [num_levels, map_h, map_w]
                 True once the player has seen / been told about this trap.
    """

    trap_type: jnp.ndarray   # [num_levels, map_h, map_w]  int8
    revealed:  jnp.ndarray   # [num_levels, map_h, map_w]  bool

    @classmethod
    def default(cls, num_levels: int, map_h: int, map_w: int) -> "TrapState":
        """Return a zeroed TrapState (no traps, nothing revealed)."""
        shape = (num_levels, map_h, map_w)
        return cls(
            trap_type=jnp.zeros(shape, dtype=jnp.int8),
            revealed=jnp.zeros(shape, dtype=jnp.bool_),
        )


# ---------------------------------------------------------------------------
# Trigger helpers (pure functions operating on rng + scalars)
# ---------------------------------------------------------------------------

def _d(rng: jax.Array, sides: int) -> jnp.ndarray:
    """Roll 1dN (sides faces).  Returns int32 in [1, sides]."""
    return jax.random.randint(rng, (), minval=1, maxval=sides + 1, dtype=jnp.int32)


def _no_se() -> jnp.ndarray:
    return jnp.zeros(5, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def place_trap(
    state: TrapState,
    pos: jnp.ndarray,
    kind: TrapType,
    rng: jax.Array,
) -> TrapState:
    """Place a trap of *kind* at *pos* on the current level.

    pos : int array [3] = (level, row, col)
    """
    new_trap_type = state.trap_type.at[pos[0], pos[1], pos[2]].set(jnp.int8(kind))
    return state.replace(trap_type=new_trap_type)


def trigger_trap(
    state: TrapState,
    rng: jax.Array,
    victim_pos: jnp.ndarray,
) -> tuple[TrapState, jnp.ndarray, jnp.ndarray]:
    """Trigger whatever trap is at *victim_pos*.

    victim_pos : int array [3] = (level, row, col)

    Returns
    -------
    new_state    : TrapState with revealed[victim_pos] = True
    damage       : int32 — HP damage dealt (0 if none)
    side_effects : int32[5]
                   [0] freeze turns, [1] sleep turns,
                   [2] teleport flag, [3] wake-monsters flag,
                   [4] level-descend flag (HOLE / TRAPDOOR)

    Trap dispatch mirrors vendor/nethack/src/trap.c :: dotrap().
    """
    lv, row, col = victim_pos[0], victim_pos[1], victim_pos[2]
    trap_kind = state.trap_type[lv, row, col].astype(jnp.int32)

    # Mark the trap as revealed after triggering.
    new_revealed = state.revealed.at[lv, row, col].set(True)
    new_state = state.replace(revealed=new_revealed)

    # Split rng into per-roll keys.
    # vendor/nethack/src/trap.c::dotrap — each trap consumer must get its
    # own random draw; PIT (line 1950: rnd(6)), SPIKED_PIT (line 1925: rnd(10))
    # and HOLE/TRAPDOOR (rnd(6)) are independent rolls in vendor, so allocate
    # distinct keys (k9, k10, k11) here rather than reusing one key.
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = jax.random.split(rng, 12)

    # Compute result for every trap type (all branches execute; jnp.select picks).
    # damage per type — vendor citations in comments.
    dmg_arrow        = _d(k0, 6)                          # vendor trap.c:1213 thitu(8, dmgval(arrow))=d6
    dmg_dart         = _d(k1, 3)                          # vendor trap.c:1278 thitu(7, dmgval(dart))=d3
    # vendor trap.c:1339 — d(2,6) = 2..12 (was incorrectly d2+d20=2..22).
    dmg_rock         = _d(k2, 6) + _d(k3, 6)              # d(2,6)
    dmg_boulder      = jnp.where(_d(k4, 6) >= 3,          # dodge if roll>=3
                                 jnp.int32(0),
                                 _d(k5, 6) + _d(k6, 6))   # d6+d6 on hit
    # vendor trap.c:4238 — d(2,4) inside dofiretrap (was incorrectly d2).
    dmg_fire         = _d(k7, 4) + _d(k8, 4)              # d(2,4)=2..8
    # vendor trap.c:1950 — rnd(6) fall damage (was incorrectly 0).
    # Use k9 for PIT (independent draw per vendor dotrap PIT branch).
    dmg_pit          = _d(k9, 6)
    # vendor trap.c:1925 — rnd(10) = 1..10 (was incorrectly d4).
    # Use k10 (distinct from k9) so SPIKED_PIT damage is independent of PIT —
    # vendor draws a separate rnd() inside dotrap's SPIKED_PIT branch.
    dmg_spiked_pit   = _d(k10, 10)                        # rnd(10)
    # Use k11 for HOLE/TRAPDOOR (vendor: rnd(6) inside its own branch).
    dmg_hole         = _d(k11, 6)                         # d6 (stub for HOLE/TRAPDOOR)
    dmg_anti_magic   = jnp.int32(0)

    # side_effects per type
    se_zeros = _no_se()

    # SQKY_BOARD: wake nearby monsters
    se_sqky = se_zeros.at[_SE_WAKE].set(1)

    # BEAR_TRAP: held rn1(4,4) = rn2(4)+4 = 4..7 turns.
    # vendor trap.c:1506 — set_utrap((unsigned) rn1(4, 4), TT_BEARTRAP);
    freeze_bear = _d(k0, 4) + jnp.int32(3)  # 4..7
    se_bear = se_zeros.at[_SE_FREEZE].set(freeze_bear)

    # SLP_GAS_TRAP: sleep rnd(25) turns.
    # vendor trap.c:1575 — fall_asleep(-rnd(25), TRUE);  range [1, 25].
    sleep_slp = _d(k1, 25)
    se_slp = se_zeros.at[_SE_SLEEP].set(sleep_slp)

    # PIT / SPIKED_PIT: trapped rn1(6,2) = rn2(6)+2 = 2..7 turns.
    # vendor trap.c:1920 — set_utrap((unsigned) rn1(6, 2), TT_PIT);
    freeze_pit = _d(k2, 6) + jnp.int32(1)  # 2..7
    se_pit = se_zeros.at[_SE_FREEZE].set(freeze_pit)
    se_spiked_pit = se_zeros.at[_SE_FREEZE].set(freeze_pit)

    # TELEP_TRAP / LEVEL_TELEP: request teleport
    se_tele = se_zeros.at[_SE_TELE].set(1)

    # HOLE / TRAPDOOR: request descend one dungeon level.
    # vendor/nethack/src/trap.c::dotrap TT_HOLE / TT_TRAPDOOR cases
    # (lines ~1950-2050) — losehp(rnd(6), ...) then goto_level(level+1, ...).
    se_descend = se_zeros.at[_SE_LEVEL_DESCEND].set(1)

    # WEB: held rn1(4,2) = rn2(4)+2 = 2..5 turns (avg-strength hero).
    # vendor trap.c:2187-2188 — STR 9..11 → tim = rn1(4, 2).
    freeze_web = _d(k3, 4) + jnp.int32(1)  # 2..5
    se_web = se_zeros.at[_SE_FREEZE].set(freeze_web)

    # DART_TRAP poison: 1/3 chance — simplified as rn2(3)==0 → halve max HP (Wave 4)
    # Wave 3: roll stored in side_effect[0] = 0 (no freeze), poison handled in caller
    se_dart = se_zeros

    # MAGIC_TRAP: d20 roll selects effect; implement 6 representative outcomes.
    # Outcomes: 1=gain ability(noop), 2=levitation(timed), 3=polymorph(noop),
    #           4=teleport, 5=confusion(noop), 6=heal(noop), 7-20=no effect.
    magic_roll = _d(k5, 20)
    se_magic = jnp.where(magic_roll == 4, se_tele, se_zeros)  # only teleport active

    # ANTI_MAGIC: drain d6 Pw (encoded as negative damage — caller interprets)
    # Wave 3: represent as -d6 in damage field (caller drains Pw not HP).
    dmg_anti_magic_drain = _d(k6, 6)  # Pw drain amount

    # VIBRATING_SQUARE: no damage, set reveal flag (already done above)
    # No special side effects.

    # Build lookup arrays indexed by TrapType.
    # Damage table: index = TrapType value
    DMG = jnp.array([
        0,                  # NO_TRAP
        dmg_arrow,          # ARROW_TRAP
        dmg_dart,           # DART_TRAP
        dmg_rock,           # ROCKTRAP
        0,                  # SQKY_BOARD
        _d(k4, 4) + _d(k5, 4),  # BEAR_TRAP d(2,4) (vendor trap.c:1490)
        _d(k7, 16),         # LANDMINE rnd(16)=1..16 (vendor trap.c:2533)
        dmg_boulder,        # ROLLING_BOULDER_TRAP
        0,                  # SLP_GAS_TRAP
        0,                  # RUST_TRAP (status message only)
        dmg_fire,           # FIRE_TRAP
        dmg_pit,            # PIT
        dmg_spiked_pit,     # SPIKED_PIT
        dmg_hole,           # HOLE
        dmg_hole,           # TRAPDOOR
        0,                  # TELEP_TRAP
        0,                  # LEVEL_TELEP
        0,                  # MAGIC_PORTAL
        0,                  # WEB
        0,                  # STATUE_TRAP
        0,                  # MAGIC_TRAP
        0,                  # ANTI_MAGIC (Pw drain, not HP damage)
        0,                  # POLY_TRAP
        0,                  # VIBRATING_SQUARE
        0,                  # TRAPPED_DOOR
        0,                  # TRAPPED_CHEST
    ], dtype=jnp.int32)

    # Side-effects table: each row is int32[4].  Stack as [N_TRAP_TYPES, 4].
    SE = jnp.stack([
        se_zeros,       # NO_TRAP
        se_zeros,       # ARROW_TRAP
        se_dart,        # DART_TRAP
        se_zeros,       # ROCKTRAP
        se_sqky,        # SQKY_BOARD
        se_bear,        # BEAR_TRAP
        se_zeros,       # LANDMINE
        se_zeros,       # ROLLING_BOULDER_TRAP
        se_slp,         # SLP_GAS_TRAP
        se_zeros,       # RUST_TRAP
        se_zeros,       # FIRE_TRAP
        se_pit,         # PIT
        se_spiked_pit,  # SPIKED_PIT
        se_descend,     # HOLE — descend one level (vendor trap.c::dotrap TT_HOLE)
        se_descend,     # TRAPDOOR — descend one level (vendor trap.c::dotrap TT_TRAPDOOR)
        se_tele,        # TELEP_TRAP
        se_tele,        # LEVEL_TELEP (stub: same-level tele)
        se_tele,        # MAGIC_PORTAL (stub)
        se_web,         # WEB
        se_zeros,       # STATUE_TRAP (Wave 4: spawn monster)
        se_magic,       # MAGIC_TRAP
        se_zeros,       # ANTI_MAGIC
        se_zeros,       # POLY_TRAP — actual polymorph applied at higher level
                        # via polymorph.poly_trap_effect(state, rng).
                        # TODO Wave 5: thread an EnvState through trigger_trap
                        # so we can call polymorph_player here directly
                        # (cur. trigger_trap only knows TrapState).
        se_zeros,       # VIBRATING_SQUARE
        se_zeros,       # TRAPPED_DOOR
        se_zeros,       # TRAPPED_CHEST
    ])  # shape [26, 4]

    # Clamp trap_kind to valid range.
    safe_kind = jnp.clip(trap_kind, 0, N_TRAP_TYPES - 1)
    damage = DMG[safe_kind]
    side_effects = SE[safe_kind]

    return new_state, damage, side_effects


def reveal_trap(
    state: TrapState,
    pos: jnp.ndarray,
) -> TrapState:
    """Mark the trap at *pos* as known to the player.

    pos : int array [3] = (level, row, col)
    """
    new_revealed = state.revealed.at[pos[0], pos[1], pos[2]].set(True)
    return state.replace(revealed=new_revealed)


def wake_monsters_near(monster_ai_state, player_pos: jnp.ndarray, radius: int):
    """Wake nearby sleeping monsters — delegates to monster_ai.wake_monsters_near."""
    # Import here to avoid circular imports.
    from Nethax.nethax.subsystems.monster_ai import wake_monsters_near as _wake
    return _wake(monster_ai_state, player_pos, radius)


def step(state: TrapState, rng: jax.Array) -> TrapState:
    """No-op per-turn tick for the trap subsystem.

    Future waves may use this to tick timed traps (e.g. LANDMINE arming
    delay) or animate rolling boulders.
    """
    return state


# ---------------------------------------------------------------------------
# Wave 5 Phase 2 — Vibrating Square / Magic Portal state-aware handlers
# ---------------------------------------------------------------------------

def trigger_vibrating_square(state, player_pos):
    """Handle player stepping onto a VIBRATING_SQUARE trap.

    Wave 5 Phase 2.  The vibrating square deals no damage but reveals a
    nearby MAGIC_PORTAL tile, which the player can then step on to be
    teleported into Gehennom (or, deep in Gehennom, into the Endgame).

    Behaviour:
      1. Sets the player-facing flag (dungeon.vibrating_square_revealed
         if that field exists; otherwise the visible trap reveal alone
         indicates discovery).
      2. Materialises a MAGIC_PORTAL trap on an adjacent floor tile of
         the current level (preferably 1 step south of the vibrating
         square; falls back to any orthogonal neighbour).

    Citation: vendor/nethack/src/trap.c (TRAP_VIBRATING_SQUARE case in
              dotrap()), vendor/nethack/include/dungeon.h
              vibrating_square mapseen flag.

    Args:
        state:      EnvState.
        player_pos: int16[2] — (row, col) of the vibrating square.

    Returns:
        Updated EnvState.
    """
    import jax.numpy as jnp
    from Nethax.nethax.constants.tiles import TileType

    # Locate the level slice in the per-tile trap_type/revealed arrays.
    b   = int(state.dungeon.current_branch)
    lv  = int(state.dungeon.current_level) - 1
    max_lv = int(state.terrain.shape[1])
    flat_lv = b * max_lv + lv

    pr = int(player_pos[0])
    pc = int(player_pos[1])
    map_h = int(state.terrain.shape[2])
    map_w = int(state.terrain.shape[3])

    # Mark the vibrating-square trap as revealed in the trap layer
    # (place_trap may have placed it; if not, do nothing).
    new_revealed = state.traps.revealed.at[flat_lv, pr, pc].set(True)

    # Find an orthogonal neighbour that is FLOOR; place a MAGIC_PORTAL
    # there in both the terrain (visible cue) and the trap layer.
    terrain_2d = state.terrain[b, lv]
    # Order of preference: south, east, north, west.
    candidates = [(pr + 1, pc), (pr, pc + 1), (pr - 1, pc), (pr, pc - 1)]
    portal_pos = None
    for (rr, cc) in candidates:
        if 0 <= rr < map_h and 0 <= cc < map_w:
            tile = int(terrain_2d[rr, cc])
            if tile == int(TileType.FLOOR):
                portal_pos = (rr, cc)
                break
    if portal_pos is None:
        # No floor neighbour — fall back to the vibrating square tile.
        portal_pos = (pr, pc)

    prow, pcol = portal_pos
    # Stamp a MAGIC_PORTAL trap on the chosen tile.
    new_trap_type = state.traps.trap_type.at[flat_lv, prow, pcol].set(
        jnp.int8(TrapType.MAGIC_PORTAL)
    )
    # Also mark the terrain as TRAP so movement code can detect it.
    new_terrain = state.terrain.at[b, lv, prow, pcol].set(
        jnp.int8(TileType.TRAP)
    )
    # Reveal the new portal too — player has been told about it.
    new_revealed = new_revealed.at[flat_lv, prow, pcol].set(True)

    new_traps = state.traps.replace(
        trap_type=new_trap_type,
        revealed=new_revealed,
    )

    # If the DungeonState was extended with a vibrating_square_revealed
    # flag (optional field), set it.  Use getattr so older state pytrees
    # still work transparently.
    new_dungeon = state.dungeon
    if hasattr(new_dungeon, "vibrating_square_revealed"):
        new_dungeon = new_dungeon.replace(
            vibrating_square_revealed=jnp.bool_(True)
        )

    return state.replace(
        traps=new_traps,
        terrain=new_terrain,
        dungeon=new_dungeon,
    )


def trigger_magic_portal(
    state,
    rng,
    target_branch: int = -1,
    target_level: int = -1,
):
    """Handle player stepping onto a MAGIC_PORTAL trap.

    Wave 5 Phase 2.  Magic portals link to a fixed (branch, level)
    destination.  If the caller doesn't override, default mapping is:

        Valley of Dead  (Gehennom L1)  -> Gehennom L2
        Gehennom L16    (top)          -> Endgame L1
        anywhere else                   -> Gehennom L1 (Valley)

    Citation: vendor/nethack/src/trap.c (TRAP_MAGIC_PORTAL case),
              vendor/nethack/dat/dungeon.lua (Vibrating Square →
              Elemental Planes wiring).

    Args:
        state:         EnvState.
        rng:           JAX PRNG key (level-gen seed on first visit).
        target_branch: optional override (-1 = use default mapping).
        target_level:  optional override (-1 = use default mapping).

    Returns:
        Updated EnvState (current branch / level / terrain / player_pos
        all advanced via traverse_portal).
    """
    from Nethax.nethax.dungeon.branches import Branch
    from Nethax.nethax.dungeon.level_memory import traverse_portal

    curr_branch = int(state.dungeon.current_branch)
    curr_level  = int(state.dungeon.current_level)

    if int(target_branch) < 0 or int(target_level) < 0:
        # Default routing.
        if curr_branch == int(Branch.GEHENNOM) and curr_level >= 16:
            dst_b, dst_l = int(Branch.ENDGAME), 1
        elif curr_branch == int(Branch.GEHENNOM) and curr_level == 1:
            dst_b, dst_l = int(Branch.GEHENNOM), 2
        else:
            dst_b, dst_l = int(Branch.GEHENNOM), 1
    else:
        dst_b, dst_l = int(target_branch), int(target_level)

    return traverse_portal(state, rng, dst_b, dst_l)


# ---------------------------------------------------------------------------
# POLY_TRAP pile branch  (vendor/nethack/src/trap.c::do_poly_pile lines 200-260)
#
# Wave 5 Phase 4 — when a POLY_TRAP fires on a tile that has items on the
# floor, each item polymorphs (rather than the player).  Mirrors do_poly_pile
# in trap.c which iterates the object stack at the tile and randomises each
# item's type via mkobj/polyobj.
# ---------------------------------------------------------------------------

def poly_pile_effect(state, rng, row, col):
    """When POLY_TRAP fires on a tile with items: each item polymorphs.

    Wave 5 simplification: scan ``ground_items`` at (row, col) on the current
    level; for each non-empty entry, randomise its ``type_id`` to a fresh
    value in [1, 255] using rng-derived per-slot keys.  Sets the POLYPILELESS
    conduct flag whenever at least one item polymorphed.

    Reference: vendor/nethack/src/trap.c::do_poly_pile.
    """
    from Nethax.nethax.subsystems.inventory import MAX_GROUND_STACK
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if

    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    safe_row = jnp.int32(row)
    safe_col = jnp.int32(col)

    ground = state.ground_items
    # Stack of items at this tile across the MAX_GROUND_STACK depth.
    cats = ground.category[b, lv, safe_row, safe_col]   # int8[MAX_GROUND_STACK]
    tids = ground.type_id[b, lv, safe_row, safe_col]    # int16[MAX_GROUND_STACK]

    # Generate a new random type_id per slot (deterministic given rng).
    keys = jax.random.split(rng, MAX_GROUND_STACK)

    def _scan_slot(carry, idx):
        cats_a, tids_a, any_changed = carry
        occupied = cats_a[idx] != jnp.int8(0)
        new_tid = jax.random.randint(
            keys[idx], shape=(), minval=1, maxval=256
        ).astype(jnp.int16)
        out_tid = jnp.where(occupied, new_tid, tids_a[idx])
        return (cats_a, tids_a.at[idx].set(out_tid),
                any_changed | occupied), None

    init = (cats, tids, jnp.bool_(False))
    (_, new_tids, any_changed), _ = jax.lax.scan(
        _scan_slot, init, jnp.arange(MAX_GROUND_STACK, dtype=jnp.int32)
    )

    # Write the updated type_id stack back into ground_items.
    new_type_id_arr = ground.type_id.at[b, lv, safe_row, safe_col].set(new_tids)
    new_ground = ground.replace(type_id=new_type_id_arr)
    new_state = state.replace(ground_items=new_ground)

    # Conduct: POLYPILELESS violated when at least one item polymorphed.
    return mark_violated_if(new_state, int(Conduct.POLYPILELESS), any_changed)


# ---------------------------------------------------------------------------
# Wave 5 Phase 4 — Wide-carrier EnvState dispatch via jax.lax.switch
#
# Mirrors vendor/nethack/src/trap.c::dotrap which is a single switch(ttyp)
# over the full set of trap types.  Each branch is a pure
# ``(state, rng) -> state`` function so they can be packed into a tuple and
# fed to ``jax.lax.switch``.
#
# All branches MUST return an EnvState with the same pytree shape (the
# "wide carrier" pattern).  Any effect that would change pytree structure
# (e.g. monster spawn, polymorph) is delegated to a helper that itself
# preserves shape.
# ---------------------------------------------------------------------------

def _flat_level_idx(state) -> jnp.ndarray:
    """Return the flat per-level index used to address state.traps arrays."""
    max_lv = jnp.int32(state.terrain.shape[1])
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    return b * max_lv + lv


def _apply_hp_damage(state, dmg):
    """Subtract ``dmg`` HP from the player, clamped at 0."""
    new_hp = jnp.maximum(jnp.int32(0), state.player_hp - dmg.astype(jnp.int32))
    return state.replace(player_hp=new_hp)


def _set_timed_status(state, status_id: int, turns):
    """Set ``state.status.timed_statuses[status_id] = max(current, turns)``."""
    cur = state.status.timed_statuses[int(status_id)]
    new_val = jnp.maximum(cur, turns.astype(jnp.int32))
    new_ts = state.status.timed_statuses.at[int(status_id)].set(new_val)
    return state.replace(status=state.status.replace(timed_statuses=new_ts))


# ---- Per-trap-type branches (all signature: (state, rng) -> state) --------

def _trap_noop(state, rng):
    """No-op branch (NO_TRAP, unimplemented Wave 5 traps)."""
    return state


def _trap_arrow(state, rng):
    """ARROW_TRAP — d6 damage.  vendor/nethack/src/trap.c::ARROW_TRAP."""
    return _apply_hp_damage(state, _d(rng, 6))


def _trap_dart(state, rng):
    """DART_TRAP — d3 damage + 1/6 poison → A_CON drain + rnd(10) HP.

    Wave 42b (Audit M #1, #2): vendor uses ``!rn2(6)`` (1/6 chance), not 1/3,
    and on poison calls ``poisoned("dart", A_CON, ..., 10, TRUE)`` which drains
    a CON point and (in the dominant path) applies rnd(10)-ish HP poison damage
    on top of the dart damage.

    Citation: vendor/nethack/src/trap.c:1273-1284 DART_TRAP poison branch
    (``if (!rn2(6)) otmp->opoisoned = 1`` then
    ``poisoned("dart", A_CON, "little dart", 10, TRUE)``).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    k0, k1, k2, k3 = jax.random.split(rng, 4)
    s = _apply_hp_damage(state, _d(k0, 3))
    # Audit M #1 — 1/6 poison chance via !rn2(6).
    poisoned = jax.random.randint(k1, (), 0, 6) == jnp.int32(0)
    # Audit M #2 — vendor drains A_CON + up to rnd(10) HP poison damage.
    poison_hp_dmg = _d(k2, 10)
    poison_sick_turns = _d(k3, 10)

    def _do_poison(s_):
        new_sick_ts = s_.status.timed_statuses.at[int(TimedStatus.SICK)].set(
            jnp.maximum(s_.status.timed_statuses[int(TimedStatus.SICK)],
                        poison_sick_turns)
        )
        new_status = s_.status.replace(
            sick_kind=jnp.int8(1),
            timed_statuses=new_sick_ts,
        )
        # vendor A_CON -1 drain (was incorrectly STR; cite trap.c:1281 "A_CON").
        new_con = jnp.maximum(s_.player_con - jnp.int8(1), jnp.int8(3))
        new_hp = jnp.maximum(s_.player_hp - poison_hp_dmg.astype(jnp.int32),
                             jnp.int32(0))
        return s_.replace(status=new_status, player_con=new_con,
                          player_hp=new_hp)

    return jax.lax.cond(poisoned, _do_poison, lambda s_: s_, s)


def _trap_rock(state, rng):
    """ROCKTRAP — 2d6 damage (wide-carrier branch)."""
    k0, k1 = jax.random.split(rng, 2)
    return _apply_hp_damage(state, _d(k0, 6) + _d(k1, 6))


def _trap_sqky_board(state, rng):
    """SQKY_BOARD — wakes nearby monsters; no direct EnvState change here."""
    return state


def _trap_bear(state, rng):
    """BEAR_TRAP — d(2,4)=2..8 dmg + FROZEN rn1(4,4)=4..7 + WOUNDED_LEGS 10..19.

    Wave 42b (Audit M #5): vendor also fires
    ``set_wounded_legs(rn2(2) ? RIGHT_SIDE : LEFT_SIDE, rn1(10, 10))``
    (10..19 turns) immediately after the iron-shoes check.

    Audit M #6 (iron-shoes protection: ``wearing_iron_shoes`` skips both the
    wounded-legs roll and the losehp call) and Audit M #7 (immunity for
    amorphous / is_whirly / unsolid / msize <= MZ_SMALL polyforms) require
    state plumbing that isn't yet exposed to traps (worn-armor slot type and
    polyform body-tag tests), and are DEFERRED with citations.

    Citations:
      vendor/nethack/src/trap.c:1490         ``int dmg = d(2, 4);``
      vendor/nethack/src/trap.c:1495-1505    (amorphous/whirly/unsolid/MZ_SMALL skips)
      vendor/nethack/src/trap.c:1506         ``set_utrap((unsigned) rn1(4, 4), TT_BEARTRAP);``
      vendor/nethack/src/trap.c:1517-1521    iron-shoes branch / set_wounded_legs
    rn1(4,4) = rn2(4)+4 = 4..7 turns; rn1(10,10) = rn2(10)+10 = 10..19 turns.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    k0, k1, k2, k3 = jax.random.split(rng, 4)
    # d(2,4) HP damage — losehp(Maybe_Half_Phys(dmg), "bear trap", ...).
    s = _apply_hp_damage(state, _d(k0, 4) + _d(k1, 4))
    # rn1(4,4) = rn2(4)+4 = 4..7 turns held.
    turns = _d(k2, 4) + jnp.int32(3)
    s = _set_timed_status(s, int(TimedStatus.FROZEN), turns)
    # Audit M #5 — rn1(10,10) = rn2(10)+10 = 10..19 turns of WOUNDED_LEGS.
    leg_turns = _d(k3, 10) + jnp.int32(9)
    return _set_timed_status(s, int(TimedStatus.WOUNDED_LEGS), leg_turns)


def _trap_landmine(state, rng):
    """LANDMINE — vendor-parity ``trapeffect_landmine`` (lines 2527-2597).

    Mirrors vendor/nethack/src/trap.c::trapeffect_landmine player branch:

    * Audit M #25 — Iron-shoes damage gate (line 2537-2538): when wearing
      ``IRON_SHOES``, damage is reduced to ``(damage + 3) / 4``.
    * Audit M #26 — Levitation/Flying short-circuit (lines 2548-2550):
      when ``Levitation`` or ``Flying`` is set, 2/3 chance (``rn2(3)``
      non-zero) to skip trigger entirely and return without damage or
      pit conversion.
    * Audit M #24 — Per-leg WOUNDED_LEGS counters (lines 2581-2582):
      vendor stores left/right side timers separately via
      ``set_wounded_legs(LEFT_SIDE, rn1(35, 41))`` /
      ``set_wounded_legs(RIGHT_SIDE, rn1(35, 41))``.  We APPROXIMATE
      with a single ``WOUNDED_LEGS`` slot storing the max of the two
      rolls (status_effects.py has only one WOUNDED_LEGS index).
    * Audit M #27 — Recursive PIT conversion (lines 2587-2596): vendor
      sets ``trap->ttyp = PIT`` then calls ``dotrap(trap, RECURSIVETRAP)``
      to deliver pit damage from the freshly-converted tile.  We
      approximate by applying ``rnd(6)`` PIT damage immediately after
      landmine damage (the trap-type conversion happens later via the
      caller's state mutation; ``trigger_trap_envstate`` already marks
      the tile as revealed).  We also update ``state.traps.trap_type``
      on the tile to ``PIT`` so subsequent triggers fire the pit branch.

    Citations:
      vendor/nethack/src/trap.c:2533 — ``int damage = rnd(16);``
      vendor/nethack/src/trap.c:2537-2538 — iron-shoes ``(damage+3)/4``
      vendor/nethack/src/trap.c:2548-2550 — Levitation/Flying skip
      vendor/nethack/src/trap.c:2581-2582 — per-leg ``rn1(35, 41)``
      vendor/nethack/src/trap.c:2587-2597 — PIT conversion + recursive trap
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    from Nethax.nethax.subsystems.status_effects import (
        TimedStatus,
        Intrinsic as _Intr,
    )
    from Nethax.nethax.subsystems.character import ObjType as _ObjType

    k_dmg, k_legL, k_legR, k_lev_skip, k_pit_dmg, k_pit_freeze = jax.random.split(rng, 6)

    # vendor trap.c:2533 — rnd(16) base damage.
    base_damage = _d(k_dmg, 16)

    # Audit M #25 — iron-shoes damage gate.
    inv = state.inventory
    boots_idx = inv.worn_armor[int(ArmorSlot.BOOTS)].astype(jnp.int32)
    has_boots = boots_idx >= jnp.int32(0)
    safe_b = jnp.clip(boots_idx, 0, inv.items.type_id.shape[0] - 1)
    boot_type = jnp.where(
        has_boots,
        inv.items.type_id[safe_b].astype(jnp.int32),
        jnp.int32(0),
    )
    iron_shoes_worn = has_boots & (boot_type == jnp.int32(int(_ObjType.IRON_SHOES)))
    damage = jnp.where(
        iron_shoes_worn,
        (base_damage + jnp.int32(3)) // jnp.int32(4),
        base_damage,
    )

    # Audit M #26 — Levitation/Flying 2/3 skip (vendor: !already_seen branch
    # returns immediately on rn2(3) non-zero, second already_seen branch also
    # skips on rn2(3) non-zero; we collapse to a single 2/3 skip roll).
    intrinsics = state.status.intrinsics
    has_lev   = intrinsics[int(_Intr.LEVITATION)]
    has_fly   = intrinsics[int(_Intr.FLYING)]
    lev_or_fly = has_lev | has_fly
    skip_roll = jax.random.randint(k_lev_skip, (), 0, 3) != jnp.int32(0)
    skip_trigger = lev_or_fly & skip_roll

    def _do_skip(s):
        return s

    def _do_fire(s):
        # Apply landmine damage.
        s1 = _apply_hp_damage(s, damage)
        # Audit M #24 — two per-leg rn1(35, 41) rolls; approximated as max
        # since traps state has only one WOUNDED_LEGS slot.
        legs_left  = _d(k_legL, 35) + jnp.int32(40)
        legs_right = _d(k_legR, 35) + jnp.int32(40)
        s2 = _set_timed_status(
            s1, int(TimedStatus.WOUNDED_LEGS), jnp.maximum(legs_left, legs_right)
        )
        # Audit M #27 — Recursive PIT: convert this tile's trap to PIT and
        # immediately apply PIT damage + FROZEN climb-out turns (mirrors
        # vendor's ``dotrap(trap, RECURSIVETRAP)`` call after trap->ttyp=PIT).
        flat_lv = _flat_level_idx(s2)
        row = s2.player_pos[0].astype(jnp.int32)
        col = s2.player_pos[1].astype(jnp.int32)
        new_trap_type = s2.traps.trap_type.at[flat_lv, row, col].set(
            jnp.int8(int(TrapType.PIT))
        )
        new_traps = s2.traps.replace(trap_type=new_trap_type)
        s3 = s2.replace(traps=new_traps)
        # Recursive PIT effect: rnd(6) fall dmg + rn1(6,2)=2..7 FROZEN turns.
        s4 = _apply_hp_damage(s3, _d(k_pit_dmg, 6))
        return _set_timed_status(
            s4, int(TimedStatus.FROZEN), _d(k_pit_freeze, 6) + jnp.int32(1)
        )

    return jax.lax.cond(skip_trigger, _do_skip, _do_fire, state)


def _trap_rolling_boulder(state, rng):
    """ROLLING_BOULDER_TRAP — spawn a boulder and apply boulder dmgval to player.

    Audit M #37: vendor (trap.c:2672) calls
    ``launch_obj(BOULDER, launch.x, launch.y, launch2.x, launch2.y,
                ROLL|LAUNCH_KNOWN)``
    which spawns an actual boulder object that rolls along the path; the
    boulder strikes the player when its path crosses the player tile and
    applies ``dmgval(BOULDER, mon)`` damage per strike (vendor objects.c
    BOULDER entry: ``int dmgval`` ≈ d(2,6) = 2..12 plus the rolling
    distance multiplier).

    Approximation: drop a BOULDER object (ROCK_CLASS, type_id=447) on the
    trap's tile via ``ground_items`` and apply ``dmgval(BOULDER)`` damage
    immediately to the player.  We use a stable ``2d6 + d4`` total
    (= 4..16) as the canonical boulder dmgval — the +d4 covers the
    rolling-distance bonus that vendor delivers across multiple strikes
    in a single trap fire.

    Citations:
      vendor/nethack/src/trap.c:2672  (launch_obj BOULDER call)
      vendor/nethack/src/dothrow.c::launch_obj  (rolls + dmgval per tile)
      vendor/nethack/include/objects.h          (BOULDER dmgval=d(2,6))
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory

    k_a, k_b, k_extra = jax.random.split(rng, 3)
    # Boulder dmgval ≈ d(2,6) + d4 (rolling-distance bonus approximation).
    dmg = _d(k_a, 6) + _d(k_b, 6) + _d(k_extra, 4)
    s = _apply_hp_damage(state, dmg)

    # Spawn a boulder on the trap's tile via ground_items.  Vendor places
    # the BOULDER at launch.x/launch.y which we approximate by the
    # player's tile (post-roll resting position).
    _BOULDER_TYPE_ID = 447
    b = s.dungeon.current_branch.astype(jnp.int32)
    lv = s.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    pr = s.player_pos[0].astype(jnp.int32)
    pc = s.player_pos[1].astype(jnp.int32)

    ground = s.ground_items
    # Find first empty slot at this tile (category == 0).
    tile_cats = ground.category[b, lv, pr, pc]
    empty_mask = tile_cats == jnp.int8(0)
    any_empty = jnp.any(empty_mask)
    slot = jnp.argmax(empty_mask.astype(jnp.int32)).astype(jnp.int32)

    def _do_spawn(s_):
        g = s_.ground_items
        new_cat = g.category.at[b, lv, pr, pc, slot].set(
            jnp.int8(int(ItemCategory.ROCK))
        )
        new_tid = g.type_id.at[b, lv, pr, pc, slot].set(jnp.int16(_BOULDER_TYPE_ID))
        new_qty = g.quantity.at[b, lv, pr, pc, slot].set(jnp.int16(1))
        new_g = g.replace(category=new_cat, type_id=new_tid, quantity=new_qty)
        return s_.replace(ground_items=new_g)

    return jax.lax.cond(any_empty, _do_spawn, lambda s_: s_, s)


def _trap_sleep_gas(state, rng):
    """SLP_GAS_TRAP — SLEEP for rnd(25)=1..25 turns.

    vendor/nethack/src/trap.c:1575 — ``fall_asleep(-rnd(25), TRUE);``
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    return _set_timed_status(state, int(TimedStatus.SLEEP), _d(rng, 25))


def _trap_rust(state, rng):
    """RUST_TRAP — water-damage random worn slot; instant death as iron golem.

    Wave 42b (Audit M #13, #14): vendor switches on ``rn2(5)`` (trap.c:1610-1644):
      case 0 → uarmh
      case 1 → uarms (shield) → uwep|uswapwep if two-handed → uarmg
      case 2 → uwep + uarmg
      default → splash-lit objects + (uarmc OR uarm OR uarmu)
    Each branch calls ``water_damage`` which rusts/rots the targeted item.
    Then (trap.c:1647-1651) if hero is polymorphed into PM_IRON_GOLEM (form
    idx 267) the trap is instant lethal: ``losehp(u.mhmax)``.

    Audit M #13 5-way switch: this implementation decrements enchantment on
    the body-armor slot for every roll (a simplification — the full per-slot
    targeting matrix requires worn_armor helm/shield/cloak/shirt plumbing
    that traps doesn't yet have).  We keep the simplification but DOCUMENT
    the vendor switch; tests rely only on body-armor enchantment dropping.

    Audit M #14 (PM_IRON_GOLEM instant kill): wired here using the polymorph
    state's current_form_idx field (iron golem = monster index 267 in our
    constants/monster_entries/chunk5.py).
    """
    # Audit M #14 — iron-golem polyform: losehp(uhmax) = instant death.
    is_iron_golem = (
        state.polymorph.is_polymorphed
        & (state.polymorph.current_form_idx == jnp.int16(267))
    )

    def _do_iron_golem_death(s):
        # vendor trap.c:1648 ``int dam = u.mhmax; losehp(Maybe_Half_Phys(dam), "rusting away", ...)``
        return s.replace(player_hp=jnp.int32(0))

    def _do_normal_rust(s):
        inv = s.inventory
        body_slot = inv.worn_armor[0]
        has_armor = body_slot >= jnp.int8(0)
        safe_idx = jnp.clip(body_slot.astype(jnp.int32), 0,
                            inv.items.category.shape[0] - 1)
        cur_ench = inv.items.enchantment[safe_idx]
        new_ench = jnp.maximum(cur_ench - jnp.int8(1), jnp.int8(-3))
        upd_ench = jnp.where(has_armor, new_ench, cur_ench)
        new_items = inv.items.replace(
            enchantment=inv.items.enchantment.at[safe_idx].set(upd_ench)
        )
        return s.replace(inventory=inv.replace(items=new_items))

    return jax.lax.cond(is_iron_golem, _do_iron_golem_death,
                        _do_normal_rust, state)


def _trap_fire(state, rng):
    """FIRE_TRAP — vendor-parity dofiretrap (Audit M #8-#12, Wave 42b).

    Implements the player branch of vendor/nethack/src/trap.c::dofiretrap
    (lines 4232-4314):

    * (#10) Underwater / box-in-pool branch (line 4244): if standing in
      water, take ``rnd(3)`` boiling-water HP (0 if Fire_resistance) and
      return — no UHPMAX drain, no item destruction.
    * (#9) Fire_resistance branch (line 4258): ``num = rn2(2)`` HP damage
      (0..1) instead of the full d(2,4).
    * (#8) UHPMAX drain (line 4289): if not Fire_resistance and not Upolyd,
      ``u.uhpmax -= rn2(min(u.uhpmax, num+1))``.
    * (#11) Item destruction gate (line 4306): items only burn when
      ``burnarmor(youmonst) || rn2(3)`` — i.e. 2/3 chance per trigger
      (we omit the burnarmor short-circuit; corpse-strapping
      burnarmor plumbing not exposed to traps).
    * (#12) Per-item luck save (line 4514, ``fire_damage``): each item
      survives when ``(Luck + 5) > rn2(20)`` (uses ``player_luck``).
      Iceboxes / statues are vendor-immune (we don't model those item
      sub-types here; SCROLL/SPBOOK/POTION are all eligible).

    Citations:
      vendor/nethack/src/trap.c:4238 — ``orig_dmg = num = d(2, 4);``
      vendor/nethack/src/trap.c:4244 — underwater branch ``rnd(3)``
      vendor/nethack/src/trap.c:4255-4258 — fire-resistance ``num = rn2(2)``
      vendor/nethack/src/trap.c:4285-4297 — UHPMAX drain
      vendor/nethack/src/trap.c:4306-4309 — ``burnarmor(...) || rn2(3)`` gate
      vendor/nethack/src/trap.c:4453+ ``fire_damage`` (per-item luck save)
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    k0, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)

    has_fire_res = state.status.intrinsics[int(Intrinsic.RESIST_FIRE)]
    in_water     = state.player_in_water

    # vendor trap.c:4287 — base ``num = d(2, 4) = 2..8`` for the normal player branch.
    base_num = _d(k0, 4) + _d(k1, 4)
    # vendor trap.c:4258 — fire-resistance: ``num = rn2(2) = 0..1``.
    fire_res_num = jax.random.randint(k2, (), 0, 2).astype(jnp.int32)
    # vendor trap.c:4250 — underwater: ``rnd(3) = 1..3`` boiling water, 0 if fire-res.
    water_num = _d(k3, 3)

    # Pick the damage according to which branch fires.
    num = jnp.where(
        in_water,
        jnp.where(has_fire_res, jnp.int32(0), water_num),
        jnp.where(has_fire_res, fire_res_num, base_num),
    )

    new_hp = jnp.maximum(state.player_hp - num, jnp.int32(0))
    # Audit M #8 — UHPMAX drain ``rn2(min(uhpmax, num+1))`` when not fire-res
    # and not polymorphed.  We don't track u.uhpmax separately on poly so the
    # not-polymorphed branch always applies.
    uhpmax_cap = jnp.minimum(state.player_hp_max, num + jnp.int32(1))
    # rn2(cap): cap may be <= 0 so guard against an invalid randint range.
    uhpmax_drain = jnp.where(
        uhpmax_cap > jnp.int32(0),
        jax.random.randint(k4, (), 0, jnp.maximum(uhpmax_cap, jnp.int32(1))).astype(jnp.int32),
        jnp.int32(0),
    )
    apply_uhpmax_drain = (~has_fire_res) & (~in_water)
    new_hp_max = jnp.where(
        apply_uhpmax_drain,
        jnp.maximum(state.player_hp_max - uhpmax_drain, jnp.int32(1)),
        state.player_hp_max,
    )
    new_hp_clamped = jnp.minimum(new_hp, new_hp_max)
    s = state.replace(player_hp=new_hp_clamped, player_hp_max=new_hp_max)

    # Audit M #11 — item destruction gate ``rn2(3) == 0 is the SKIP case`` so
    # 2/3 of triggers actually burn items.  Underwater branch skips item burn
    # entirely (vendor returns early after the boiling-water message).
    burn_items_roll = jax.random.randint(k5, (), 0, 3) != jnp.int32(0)
    burn_items_active = burn_items_roll & (~in_water)

    # Audit M #12 — per-item luck save: object survives when (Luck + 5) > rn2(20).
    inv = s.inventory
    luck = state.player_luck.astype(jnp.int32) + state.player_moreluck.astype(jnp.int32)
    n_slots = inv.items.category.shape[0]
    luck_keys = jax.random.split(k6, n_slots)  # distinct stream per slot.
    luck_rolls = jax.vmap(lambda k: jax.random.randint(k, (), 0, 20))(luck_keys)
    # Item is destroyed when (Luck + 5) <= rn2(20).
    luck_destroys = (jnp.int32(luck + jnp.int32(5))) <= luck_rolls.astype(jnp.int32)
    is_burnable_cat = (
        (inv.items.category == jnp.int8(int(ItemCategory.SCROLL)))
        | (inv.items.category == jnp.int8(int(ItemCategory.SPBOOK)))
        | (inv.items.category == jnp.int8(int(ItemCategory.POTION)))
    )
    destroyed = burn_items_active & is_burnable_cat & luck_destroys
    new_qty = jnp.where(destroyed, jnp.int16(0), inv.items.quantity)
    new_cat = jnp.where(destroyed, jnp.int8(0), inv.items.category)
    new_items = inv.items.replace(quantity=new_qty, category=new_cat)
    return s.replace(inventory=inv.replace(items=new_items))


def _trap_pit(state, rng):
    """PIT — rnd(6)=1..6 damage; FROZEN for rn1(6,2)=2..7 climb-out turns.

    Wave 42b (Audit M #4): vendor uses ``rnd(adj_pit ? 3 : 6)`` and skips fall
    damage entirely when ``conjoined_pit`` is set or the player is ``Flying``
    or ``is_clinger``.  Those flags aren't plumbed into ``EnvState`` yet, so
    we apply the default ``rnd(6)`` branch (matches vendor when adj_pit=FALSE
    and player is grounded), and DEFER the corner-case branches.

    Citations:
      vendor/nethack/src/trap.c:1920 — ``set_utrap((unsigned) rn1(6, 2), TT_PIT);``
      vendor/nethack/src/trap.c:1948-1953 — ``if (!conj_pit && !deliberate
        && !(plunged && (Flying || is_clinger(...)))) losehp(rnd(adj_pit ? 3 : 6), ...);``
    rn1(6,2) = rn2(6)+2 = 2..7 turns trapped (we model as `_d(rng,6)+1` = 2..7).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    k0, k1 = jax.random.split(rng, 2)
    s = _apply_hp_damage(state, _d(k0, 6))
    # rn1(6,2) = rn2(6)+2 = 2..7  →  _d(_,6)+1 = 2..7
    return _set_timed_status(s, int(TimedStatus.FROZEN), _d(k1, 6) + jnp.int32(1))


def _trap_spiked_pit(state, rng):
    """SPIKED_PIT — rnd(10)=1..10 damage + 1/6 poison: A_STR drain + rnd(8) HP.

    Wave 42b (Audit M #3): vendor calls
    ``poisoned("spikes", A_STR, ..., 8, FALSE)`` which drains an A_STR point
    and may add up to rnd(8) HP poison damage on top of the spike damage.

    Citations:
      vendor/nethack/src/trap.c:1920 — ``set_utrap((unsigned) rn1(6, 2), TT_PIT);``
      vendor/nethack/src/trap.c:1925 — ``losehp(... rnd(conj_pit ? 4 : adj_pit ? 6 : 10) ...);``
      vendor/nethack/src/trap.c:1938-1945 — ``if (!rn2(6)) poisoned("spikes", A_STR, ..., 8, FALSE);``
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    k0, k1, k2, k3, k4 = jax.random.split(rng, 5)
    s = _apply_hp_damage(state, _d(k0, 10))
    # rn1(6,2) = 2..7 turns held.
    s = _set_timed_status(s, int(TimedStatus.FROZEN), _d(k1, 6) + jnp.int32(1))
    # 1/6 poison: vendor uses !rn2(6).
    poisoned = jax.random.randint(k2, (), 0, 6) == jnp.int32(0)
    poison_hp_dmg = _d(k3, 8)            # Audit M #3 — +rnd(8) HP poison.
    poison_sick_turns = _d(k4, 10)

    def _do_poison(s_):
        new_sick = s_.status.replace(
            sick_kind=jnp.int8(1),
            timed_statuses=s_.status.timed_statuses.at[int(TimedStatus.SICK)].set(
                jnp.maximum(s_.status.timed_statuses[int(TimedStatus.SICK)],
                            poison_sick_turns)
            ),
        )
        # Audit M #3 — vendor drains A_STR (NOT just SICK timer).
        new_str = jnp.maximum(s_.player_str - jnp.int16(1), jnp.int16(3))
        new_hp = jnp.maximum(s_.player_hp - poison_hp_dmg.astype(jnp.int32),
                             jnp.int32(0))
        return s_.replace(status=new_sick, player_str=new_str, player_hp=new_hp)

    return jax.lax.cond(poisoned, _do_poison, lambda s_: s_, s)


def _trap_hole(state, rng):
    """HOLE — descend one level + apply rnd(6) fall damage + land on random FLOOR.

    vendor/nethack/src/trap.c dotrap HOLE branch:
      losehp(rnd(6), ...) then goto_level(level+1, ...).
    Player lands on a random FLOOR tile on the destination level, NOT on the
    up-staircase (trap.c goto_level uses TELEPATH_RANDOM for hole/trapdoor
    placement).  Clamps at MAX_LEVELS_PER_BRANCH so the carrier shape is stable.

    Citation: vendor/nethack/src/trap.c dotrap HOLE/TRAPDOOR.
    """
    from Nethax.nethax.constants.tiles import TileType
    rng, k_dmg, k_land = jax.random.split(rng, 3)
    # rnd(6) = 1..6 fall damage (vendor trap.c dotrap HOLE).
    dmg = jax.random.randint(k_dmg, shape=(), minval=1, maxval=7).astype(jnp.int32)
    new_hp = jnp.maximum(state.player_hp - dmg, jnp.int32(0))
    max_lv = jnp.int8(state.terrain.shape[1])
    new_level = jnp.minimum(state.dungeon.current_level + jnp.int8(1), max_lv)

    # Land on a random FLOOR tile of the destination level (not on staircase).
    b = state.dungeon.current_branch.astype(jnp.int32)
    dst_lv = new_level.astype(jnp.int32) - jnp.int32(1)
    dst_terrain = state.terrain[b, dst_lv]
    floor_mask = (dst_terrain == jnp.int8(int(TileType.FLOOR))).reshape(-1)
    uni = jax.random.uniform(k_land, shape=floor_mask.shape, dtype=jnp.float32)
    scores = jnp.where(floor_mask, uni, jnp.float32(-1.0))
    flat_idx = jnp.argmax(scores).astype(jnp.int32)
    row = (flat_idx // dst_terrain.shape[1]).astype(jnp.int16)
    col = (flat_idx %  dst_terrain.shape[1]).astype(jnp.int16)
    any_floor = jnp.any(floor_mask)
    new_row = jnp.where(any_floor, row, state.player_pos[0])
    new_col = jnp.where(any_floor, col, state.player_pos[1])

    new_dungeon = state.dungeon.replace(current_level=new_level)
    return state.replace(
        dungeon=new_dungeon,
        player_hp=new_hp,
        player_pos=jnp.stack([new_row, new_col]).astype(jnp.int16),
    )


def _trap_trapdoor(state, rng):
    """TRAPDOOR — same effect as HOLE (rnd(6) fall damage + descend level).

    vendor/nethack/src/trap.c dotrap TRAPDOOR — identical to HOLE branch.
    """
    return _trap_hole(state, rng)


def _trap_telep(state, rng):
    """TELEP_TRAP — teleport to a random FLOOR / CORRIDOR tile on the level.

    Audit-M #32: vendor's tele_trap selects any open square (floor OR
    corridor) — see vendor/nethack/src/teleport.c::tele_trap which calls
    teleok() / placebc() to find an "open" passable tile.

    vendor/nethack/src/trap.c::TELEP_TRAP — tele() picks an open square.
    """
    from Nethax.nethax.constants.tiles import TileType
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    terrain_2d = state.terrain[b, lv]
    # Audit-M #32 — also allow CORRIDOR per vendor teleok() valid-tile mask.
    open_mask = (
        (terrain_2d == jnp.int8(int(TileType.FLOOR)))
        | (terrain_2d == jnp.int8(int(TileType.CORRIDOR)))
    ).reshape(-1)
    k0, _ = jax.random.split(rng, 2)
    uni = jax.random.uniform(k0, shape=open_mask.shape, dtype=jnp.float32)
    scores = jnp.where(open_mask, uni, jnp.float32(-1.0))
    flat_idx = jnp.argmax(scores).astype(jnp.int32)
    row = (flat_idx // terrain_2d.shape[1]).astype(jnp.int16)
    col = (flat_idx %  terrain_2d.shape[1]).astype(jnp.int16)
    any_open = jnp.any(open_mask)
    new_row = jnp.where(any_open, row, state.player_pos[0])
    new_col = jnp.where(any_open, col, state.player_pos[1])
    return state.replace(
        player_pos=jnp.stack([new_row, new_col]).astype(jnp.int16)
    )


def _trap_level_telep(state, rng):
    """LEVEL_TELEP — teleport to a random level in the current branch.

    vendor/nethack/src/trap.c::LEVEL_TELEP — level_tele() picks a depth.
    """
    max_lv = state.terrain.shape[1]
    new_level = jax.random.randint(rng, (), 1, max_lv + 1).astype(jnp.int8)
    new_dungeon = state.dungeon.replace(current_level=new_level)
    return state.replace(dungeon=new_dungeon)


def _trap_magic_portal(state, rng):
    """MAGIC_PORTAL — teleport to fixed (dest_branch, dest_level) stored in
    state.dungeon.portal_destination[branch, level-1].

    Citation: vendor/nethack/src/trap.c::dotrap MAGIC_PORTAL branch —
    the trap stores a d_level destination (d_level::dnum / d_level::dlevel)
    and calls goto_level(&trap->dst, ...) unconditionally.

    Reads portal_destination[branch, level-1] -> (dest_branch, dest_level).
    -1 in either field means no portal configured; state is returned unchanged.
    """
    b   = state.dungeon.current_branch.astype(jnp.int32)
    lv  = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    dst = state.dungeon.portal_destination[b, lv]   # int8[2]
    dst_branch = dst[0].astype(jnp.int8)
    dst_level  = dst[1].astype(jnp.int8)
    configured = (dst[0] != jnp.int8(-1)) & (dst[1] != jnp.int8(-1))
    new_branch = jnp.where(configured, dst_branch, state.dungeon.current_branch)
    new_level  = jnp.where(configured, dst_level,  state.dungeon.current_level)
    new_dungeon = state.dungeon.replace(
        current_branch=new_branch,
        current_level=new_level,
    )
    return state.replace(dungeon=new_dungeon)


def _trap_web(state, rng):
    """WEB — STR-tiered held timer per vendor trap.c:2181-2202.

    Vendor's per-strength escape table (Audit M #28):
        STR ≤ 3   :  rn1(6, 6)  = 6..11
        STR < 6   :  rn1(6, 4)  = 4..9
        STR < 9   :  rn1(4, 4)  = 4..7
        STR < 12  :  rn1(4, 2)  = 2..5
        STR < 15  :  rn1(2, 2)  = 2..3
        STR < 18  :  rnd(2)     = 1..2
        STR < 69  :  1
        STR ≥ 69  :  0 (web destroyed immediately)

    rn1(x, y) = rn2(x) + y; rnd(x) = rn2(x) + 1.

    Previously this used the fixed STR-9..11 case only (rn1(4, 2)).

    Cite: vendor/nethack/src/trap.c:2181-2202 (per-strength web-escape
    branches inside dotrap WEB).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    # Per-tier dice rolls (each as a fresh draw against ``rng``).  We
    # split the rng up-front and pick the one matching the player's STR
    # tier so the byte-stream stays deterministic per seed.
    k0, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
    s = state.player_str.astype(jnp.int32)

    # All tiers (low-STR first so jnp.where chains in narrow-to-broad order).
    t_3   = _d(k0, 6) + jnp.int32(5)     # rn1(6,6) = 6..11
    t_6   = _d(k1, 6) + jnp.int32(3)     # rn1(6,4) = 4..9
    t_9   = _d(k2, 4) + jnp.int32(3)     # rn1(4,4) = 4..7
    t_12  = _d(k3, 4) + jnp.int32(1)     # rn1(4,2) = 2..5
    t_15  = _d(k4, 2) + jnp.int32(1)     # rn1(2,2) = 2..3
    t_18  = jax.random.randint(k5, (), 1, 3, dtype=jnp.int32)  # rnd(2) = 1..2

    timer = jnp.where(
        s <= jnp.int32(3),  t_3,
        jnp.where(s < jnp.int32(6),  t_6,
            jnp.where(s < jnp.int32(9),  t_9,
                jnp.where(s < jnp.int32(12), t_12,
                    jnp.where(s < jnp.int32(15), t_15,
                        jnp.where(s < jnp.int32(18), t_18,
                            jnp.where(s < jnp.int32(69), jnp.int32(1),
                                                          jnp.int32(0))))))),
    )
    return _set_timed_status(state, int(TimedStatus.FROZEN), timer)


def _trap_statue(state, rng):
    """STATUE_TRAP — animate the statue's stored monster (Audit M #38).

    Vendor: trap.c::activate_statue_trap (line 907) calls
    ``animate_statue(otmp, ...)`` which uses the statue object's stored
    ``corpsenm`` (monster index) to spawn that specific species with its
    proper hp/AC/attack table.

    Approximation: scan ``ground_items`` at the player's tile for a STATUE
    object (type_id == 448).  Decode the stored monster index from the
    item's ``charges`` field (Nethax has no per-item ``corpsenm`` slot;
    ``charges`` is unused for statues so we re-purpose it as the encoded
    monster index — Item.charges is int8 so the range is 0..127, which
    covers ~half of the 700-entry MONSTERS table; for indexes ≥128 we
    fall back to a random monster).  If no statue is on the tile, fall
    back to spawning a random low-level monster (former behaviour).

    HP is derived from the species' ``mlevel`` via ``(level + 1) * 4``
    (vendor uses ``rnd((mlevel+1)*8)`` but we want a deterministic mean).

    Citation: vendor/nethack/src/trap.c:907-936 (activate_statue_trap),
              vendor/nethack/src/mkobj.c (statue->corpsenm storage).
    """
    from Nethax.nethax.subsystems.polymorph import _MONSTER_TABLES

    mai = state.monster_ai
    dead_mask = ~mai.alive
    dead_mask = dead_mask.at[0].set(False)  # skip sentinel slot 0
    slot = jnp.argmax(dead_mask).astype(jnp.int32)
    any_dead = jnp.any(dead_mask)

    # ---- Read statue at player tile (Audit M #38) ---------------------
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1)
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)

    ground = state.ground_items
    # Stack of items at this tile: shape [MAX_GROUND_STACK].
    g_tids    = ground.type_id[b, lv, pr, pc]
    g_charges = ground.charges[b, lv, pr, pc]
    # Find first STATUE in the stack.
    _STATUE_TYPE_ID = 448
    is_statue = g_tids == jnp.int16(_STATUE_TYPE_ID)
    has_statue = jnp.any(is_statue)
    statue_slot = jnp.argmax(is_statue.astype(jnp.int32)).astype(jnp.int32)
    encoded_mon_idx = g_charges[statue_slot].astype(jnp.int32)
    # Charges is int8 → 0..127 valid range; treat <=0 or >=N as "use random".
    n_mons = jnp.int32(_MONSTER_TABLES["n"])
    encoded_valid = (
        has_statue
        & (encoded_mon_idx > jnp.int32(0))
        & (encoded_mon_idx < n_mons)
    )

    # Random fallback monster (low-level, idx 1..min(20, n-1)).
    k_rand_mon, k_rand_hp = jax.random.split(rng, 2)
    rand_mon_idx = jax.random.randint(
        k_rand_mon, (), 1, jnp.minimum(jnp.int32(20), n_mons)
    ).astype(jnp.int32)
    mon_idx = jnp.where(encoded_valid, encoded_mon_idx, rand_mon_idx)
    safe_mon = jnp.clip(mon_idx, jnp.int32(1), n_mons - jnp.int32(1))
    # Derive hp_max from species level: (level + 1) * 4 (deterministic mean
    # of vendor ``rnd((mlevel+1)*8)``).
    mon_level = _MONSTER_TABLES["level"][safe_mon].astype(jnp.int32)
    mon_hp = jnp.maximum(
        (mon_level + jnp.int32(1)) * jnp.int32(4),
        jnp.int32(1),
    )

    def _do_spawn(s):
        m = s.monster_ai
        ppos = s.player_pos.astype(jnp.int16)
        new_mai = m.replace(
            alive=m.alive.at[slot].set(jnp.bool_(True)),
            pos=m.pos.at[slot].set(ppos),
            hp=m.hp.at[slot].set(mon_hp),
            hp_max=m.hp_max.at[slot].set(mon_hp),
            peaceful=m.peaceful.at[slot].set(jnp.bool_(False)),
        )
        # Encode species into monster_ai.entry_idx if the field exists
        # (it stores MONSTERS index for combat/AI lookups).
        if hasattr(m, "entry_idx"):
            new_mai = new_mai.replace(
                entry_idx=m.entry_idx.at[slot].set(safe_mon.astype(m.entry_idx.dtype))
            )
        return s.replace(monster_ai=new_mai)

    return jax.lax.cond(any_dead, _do_spawn, lambda s: s, state)


def _trap_magic(state, rng):
    """MAGIC_TRAP — vendor-parity mini-switch over ``rnd(20)`` + 1/30 explosion.

    Mirrors vendor/nethack/src/trap.c::trapeffect_magic_trap (line 2293) which
    first rolls ``!rn2(30)`` for a magical explosion, otherwise calls
    ``domagictrap`` (line 4317) with ``fate = rnd(20)``:

      fate 1     → gain ability: +1 random stat (str/dex/con/int/wis/cha).
                   Citation: vendor/nethack/src/trap.c::domagictrap fate=1
                   (adjattrib random stat +1).
      fate 2..3  → monster summon, blind ``rn1(5,10)``=10..14, deafen
                   ``rn1(20,30)``=30..49.
      fate 3 (idx 2) → polymorph self (vendor domagictrap fate=3 → polyself).
                   Citation: vendor/nethack/src/trap.c::domagictrap fate=3.
      fate 4..9  → monster summon, blind + deaf (same as fate 2).
      fate 5 (idx 4) → confusion rn1(5,15)=5..19 turns.
                   Citation: vendor/nethack/src/trap.c::domagictrap fate=5.
      fate 6 (idx 5) → heal: hp = hp_max.
                   Citation: vendor/nethack/src/trap.c::domagictrap fate=6.
      fate = 10  → nothing.
      fate = 11  → toggle intrinsic invisibility (INVIS_TMP timer).
      fate = 12  → flash of fire (calls ``dofiretrap`` — d(2,4)=2..8).
      fate = 13  → shiver (no mechanical effect).
      fate = 14  → distant howling (no mechanical effect).
      fate = 15  → yearning (no mechanical effect).
      fate = 16  → pack shakes (no mechanical effect).
      fate = 17  → smell food (no mechanical effect).
      fate = 18  → feel tired (no mechanical effect).
      fate = 19  → tame nearby monsters + Cha+1 (we grant gold proxy).
      fate = 20  → uncurse stuff (no easy parity; small heal proxy).

    The pre-explosion branch (1/30) loses ``rnd(10)``=1..10 HP and gains
    ``+2`` to ``player_pw_max`` / ``player_pw``.

    All 21 sub-outcomes (1 explosion + 20 fate values) preserve pytree shape.
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    from Nethax.nethax.subsystems.polymorph import poly_trap_effect
    k_xpl, k_fate, k_use, k_stat = jax.random.split(rng, 4)

    # vendor trap.c:2300 — if (!rn2(30)) magical explosion.
    is_explosion = jax.random.randint(k_xpl, (), 0, 30) == jnp.int32(0)

    # Pre-compute domagictrap outcomes (fate 1..20).  Branch index 0..19.
    fate_idx = jax.random.randint(k_fate, (), 0, 20).astype(jnp.int32)

    # fate 1 (idx 0): gain +1 to a random stat.
    # Citation: vendor/nethack/src/trap.c::domagictrap fate=1 → adjattrib(rn2(A_MAX), 1).
    def b_gain_ability(s, r):
        stat_idx = jax.random.randint(r, (), 0, 6).astype(jnp.int32)
        new_str = jnp.where(stat_idx == 0, s.player_str + jnp.int16(1), s.player_str)
        new_dex = jnp.where(stat_idx == 1, s.player_dex + jnp.int8(1),  s.player_dex)
        new_con = jnp.where(stat_idx == 2, s.player_con + jnp.int8(1),  s.player_con)
        new_int = jnp.where(stat_idx == 3, s.player_int + jnp.int8(1),  s.player_int)
        new_wis = jnp.where(stat_idx == 4, s.player_wis + jnp.int8(1),  s.player_wis)
        new_cha = jnp.where(stat_idx == 5, s.player_cha + jnp.int8(1),  s.player_cha)
        return s.replace(player_str=new_str, player_dex=new_dex, player_con=new_con,
                         player_int=new_int, player_wis=new_wis, player_cha=new_cha)

    def b_monsters(s, r):
        # vendor trap.c:4323-4352 — summons + blind rn1(5,10) + deaf rn1(20,30).
        s1 = _set_timed_status(s, int(TimedStatus.BLIND),
                               _d(r, 5) + jnp.int32(9))   # rn1(5,10)=10..14
        return _set_timed_status(s1, int(TimedStatus.DEAF),
                                 _d(r, 20) + jnp.int32(29))  # rn1(20,30)=30..49

    # fate 3 (idx 2): polymorph self.
    # Citation: vendor/nethack/src/trap.c::domagictrap fate=3 → polyself().
    def b_polymorph(s, r):
        return poly_trap_effect(s, r)

    def b_nothing(s, r):  # fate 10 - sometimes nothing happens
        return s

    def b_invis(s, r):    # fate 11 — toggle intrinsic invisibility
        # Vendor trap.c:4377: ``HInvis = HInvis ? 0L : HInvis | FROMOUTSIDE;``
        # The fate toggles the permanent intrinsic flag, not a timed bump.
        # Audit M #20.  Previously this set TimedStatus.INVIS_TMP=50, which
        # was a temporary timer.  Now flip the persistent
        # ``Intrinsic.INVIS`` boolean (vendor FROMOUTSIDE-equivalent).
        # Cite: vendor/nethack/src/trap.c:4377 (b_invis branch).
        from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
        ints = s.status.intrinsics
        idx = int(_Intr.INVIS)
        cur = ints[idx]
        new_ints = ints.at[idx].set(~cur)
        return s.replace(status=s.status.replace(intrinsics=new_ints))

    def b_fire(s, r):     # fate 12 - flash of fire = dofiretrap d(2,4)
        k0, k1 = jax.random.split(r, 2)
        return _apply_hp_damage(s, _d(k0, 4) + _d(k1, 4))

    def b_shiver(s, r):   return s  # fate 13
    def b_howling(s, r):  return s  # fate 14
    def b_yearning(s, r): return s  # fate 15
    def b_shakes(s, r):   return s  # fate 16
    def b_smell(s, r):    return s  # fate 17
    def b_tired(s, r):    return s  # fate 18

    def b_tame(s, r):     # fate 19 — Cha+1 + tame nearby monsters
        # vendor trap.c:4422-4430:
        #   if (!Confusion) adjattrib(A_CHA, 1, A_BEYOND_BREATH);
        #   for ... 3x3 around hero: if mtmp && rn2(2): tamedog(mtmp)
        # We bump CHA by 1 (capped at 25 per ATTRMAX) and tame each
        # adjacent (Chebyshev=1) alive hostile monster on a 50% roll.
        # Audit M item #21.
        # Cite: vendor/nethack/src/trap.c:4422-4430 (b_tame branch).
        cha_new = jnp.minimum(s.player_cha.astype(jnp.int32) + jnp.int32(1),
                              jnp.int32(25)).astype(s.player_cha.dtype)
        s1 = s.replace(player_cha=cha_new)

        # Tame nearby (3x3 around player_pos).
        mai = s1.monster_ai
        pr = s1.player_pos[0].astype(jnp.int32)
        pc = s1.player_pos[1].astype(jnp.int32)
        all_pos = mai.pos.astype(jnp.int32)
        d_row = jnp.abs(all_pos[:, 0] - pr)
        d_col = jnp.abs(all_pos[:, 1] - pc)
        adj = jnp.maximum(d_row, d_col) <= jnp.int32(1)

        # Eligible: alive, hostile (not already tame), within 3x3.
        eligible = mai.alive & ~mai.tame & adj

        # Per-monster 50/50 roll via vmap on split keys.
        n = mai.alive.shape[0]
        k_tame_keys = jax.random.split(r, n)
        tame_rolls = jax.vmap(lambda k: jax.random.randint(k, (), 0, 2))(k_tame_keys)
        do_tame = eligible & (tame_rolls == jnp.int32(0))

        new_tame      = jnp.where(do_tame, jnp.bool_(True),  mai.tame)
        new_peaceful  = jnp.where(do_tame, jnp.bool_(True),  mai.peaceful)
        new_mai = mai.replace(tame=new_tame, peaceful=new_peaceful)
        return s1.replace(monster_ai=new_mai)

    def b_uncurse(s, r):  # fate 20 — uncurse all worn/wielded items
        # vendor trap.c:4433-4445: ``(void) seffects(SPE_REMOVE_CURSE, ...)``
        # which uncurses every worn/wielded inventory slot.  Audit M #22.
        # Vendor BUC encoding: 1=cursed, 2=uncursed, 3=blessed.  We flip
        # every cursed item to uncursed; blessed/uncursed are left alone.
        # Cite: vendor/nethack/src/trap.c:4433-4445;
        #       vendor/nethack/src/read.c::seffect_remove_curse 1489+.
        items = s.inventory.items
        cur_buc = items.buc_status
        was_cursed = cur_buc == jnp.int8(1)
        new_buc = jnp.where(was_cursed, jnp.int8(2), cur_buc)
        new_items = items.replace(buc_status=new_buc)
        new_inv = s.inventory.replace(items=new_items)
        return s.replace(inventory=new_inv)

    # fate 5 (idx 4): confusion rn1(5,15)=5..19 turns.
    # Citation: vendor/nethack/src/trap.c::domagictrap fate=5 → make_confused().
    def b_confusion(s, r):
        return _set_timed_status(s, int(TimedStatus.CONFUSION),
                                 _d(r, 15) + jnp.int32(4))  # rn1(5,15)=5..19

    # fate 6 (idx 5): heal — hp = hp_max.
    # Citation: vendor/nethack/src/trap.c::domagictrap fate=6 → healup(u.uhpmax, 0).
    def b_heal(s, r):
        return s.replace(player_hp=s.player_hp_max)

    # 20 branches matching vendor fate 1..20 in order (index = fate - 1).
    # Audit M item #15-#23: vendor ``trap.c::domagictrap`` (lines 4322-4370)
    # uses ``if (fate < 10) {`` for ALL of fates 1..9 — they share the
    # single "summon monsters + blind + deaf" b_monsters branch.  The
    # previous mapping invented four distinct effects at fates 1/3/5/6
    # (gain_ability, polymorph, confusion, heal) that don't exist in
    # vendor.  Now reshaped so all fates 1..9 → b_monsters.
    # Cite: vendor/nethack/src/trap.c:4322-4370 (domagictrap fate<10 arm).
    fate_branches = (
        b_monsters,          # fate 1  (idx 0): summon + blind + deaf
        b_monsters,          # fate 2  (idx 1)
        b_monsters,          # fate 3  (idx 2)  was b_polymorph (invented)
        b_monsters,          # fate 4  (idx 3)
        b_monsters,          # fate 5  (idx 4)  was b_confusion (invented)
        b_monsters,          # fate 6  (idx 5)  was b_heal (invented)
        b_monsters,          # fate 7  (idx 6)
        b_monsters,          # fate 8  (idx 7)
        b_monsters,          # fate 9  (idx 8)
        b_nothing,           # fate 10 (idx 9)
        b_invis,             # fate 11 (idx 10)
        b_fire,              # fate 12 (idx 11)
        b_shiver,            # fate 13 (idx 12)
        b_howling,           # fate 14 (idx 13)
        b_yearning,          # fate 15 (idx 14)
        b_shakes,            # fate 16 (idx 15)
        b_smell,             # fate 17 (idx 16)
        b_tired,             # fate 18 (idx 17)
        b_tame,              # fate 19 (idx 18)
        b_uncurse,           # fate 20 (idx 19)
    )

    def _do_explosion(s, r):
        # vendor trap.c:2304-2306 — losehp(rnd(10), ...); u.uenmax += 2.
        k0, k1 = jax.random.split(r, 2)
        s1 = _apply_hp_damage(s, _d(k0, 10))
        new_pw_max = s1.player_pw_max + jnp.int32(2)
        new_pw     = s1.player_pw + jnp.int32(2)
        return s1.replace(player_pw=new_pw, player_pw_max=new_pw_max)

    def _do_fate(s, r):
        return jax.lax.switch(fate_idx, fate_branches, s, r)

    return jax.lax.cond(is_explosion, _do_explosion, _do_fate, state, k_use)


def _trap_anti_magic(state, rng):
    """ANTI_MAGIC — vendor-parity Pw drain / Antimagic dmg / iron-shoes branch.

    Mirrors vendor/nethack/src/trap.c::trapeffect_anti_magic (lines 2322-2398):

    * Audit M #36 — Iron-shoes intercept (lines 2328-2343): when wearing iron
      shoes with ``spe > 0``, the trap drains 1 enchantment from the shoes
      and skips the player Pw drain entirely.
    * Audit M #34 — Antimagic intrinsic branch (lines 2351-2378): when
      ``Antimagic`` (MAGIC_RESIST), take ``dmgval2 = rnd(4)`` HP damage in
      place of Pw drain.  ``dmgval2`` is boosted by ``rnd(4)`` when the
      hero wields Magicbane (vendor cite: ``u_wield_art(ART_MAGICBANE)``)
      or has Half_physical/Half_spell damage active; halved (``(d+3)/4``)
      when ``Passes_walls`` is set.
    * Audit M #35 — uenmax split (lines 2386-2398): the base 2d6 drain
      first siphons ``halfd = rnd(drain/2)`` from ``player_pw_max`` (if
      ``uenmax > drain``) and the remaining ``drain - halfd`` from
      current Pw — previously the full drain came from current Pw only.

    Citations:
      vendor/nethack/src/trap.c:2328-2343 (iron-shoes branch / spe-=1)
      vendor/nethack/src/trap.c:2351-2378 (Antimagic dmgval2 dmgs HP)
      vendor/nethack/src/trap.c:2386-2398 (drain / halfd uenmax split)
    """
    from Nethax.nethax.subsystems.inventory import ArmorSlot
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _Intr
    from Nethax.nethax.subsystems.character import ObjType as _ObjType

    k_dmg, k_drain1, k_drain2, k_halfd, k_mb, k_half = jax.random.split(rng, 6)

    # ------------------------------------------------------------------
    # Iron-shoes intercept (vendor trap.c:2328-2343).
    # ------------------------------------------------------------------
    inv = state.inventory
    boots_idx = inv.worn_armor[int(ArmorSlot.BOOTS)].astype(jnp.int32)
    has_boots = boots_idx >= jnp.int32(0)
    safe_b = jnp.clip(boots_idx, 0, inv.items.type_id.shape[0] - 1)
    boot_type = jnp.where(
        has_boots,
        inv.items.type_id[safe_b].astype(jnp.int32),
        jnp.int32(0),
    )
    boot_spe = jnp.where(
        has_boots,
        inv.items.enchantment[safe_b].astype(jnp.int32),
        jnp.int32(0),
    )
    iron_shoes_worn = has_boots & (boot_type == jnp.int32(int(_ObjType.IRON_SHOES)))
    iron_shoes_protect = iron_shoes_worn & (boot_spe > jnp.int32(0))

    def _do_iron_shoes(s):
        new_spe = jnp.maximum(boot_spe - jnp.int32(1), jnp.int32(-3)).astype(
            s.inventory.items.enchantment.dtype
        )
        new_items = s.inventory.items.replace(
            enchantment=s.inventory.items.enchantment.at[safe_b].set(new_spe)
        )
        return s.replace(inventory=s.inventory.replace(items=new_items))

    # ------------------------------------------------------------------
    # Antimagic intrinsic branch (vendor trap.c:2351-2378).
    # ------------------------------------------------------------------
    intrinsics = state.status.intrinsics
    has_antimagic    = intrinsics[int(_Intr.MAGIC_RESIST)]
    has_passes_walls = intrinsics[int(_Intr.PASSES_WALLS)]
    has_half_phys    = intrinsics[int(_Intr.HALF_PHYSICAL_DAMAGE)]
    has_half_spell   = intrinsics[int(_Intr.HALF_SPELL_DAMAGE)]

    # Vendor: ``dmgval2 = rnd(4)``; +rnd(4) if Half_phys or Half_spell;
    # +rnd(4) if u_wield_art(ART_MAGICBANE); halve if Passes_walls.
    # We omit the carried-artifact AD_MAGM bonus (no carried-artifact
    # plumbing for AD_MAGM defends_when_carried in traps yet — DEFERRED).
    _ARTI_MAGICBANE = 29
    wields_magicbane = (
        state.inventory.wielded_artifact_idx.astype(jnp.int32)
        == jnp.int32(_ARTI_MAGICBANE)
    )
    dmgval2 = _d(k_dmg, 4)
    dmgval2 = dmgval2 + jnp.where(
        has_half_phys | has_half_spell, _d(k_half, 4), jnp.int32(0)
    )
    dmgval2 = dmgval2 + jnp.where(wields_magicbane, _d(k_mb, 4), jnp.int32(0))
    dmgval2 = jnp.where(
        has_passes_walls, (dmgval2 + jnp.int32(3)) // jnp.int32(4), dmgval2
    )

    # ------------------------------------------------------------------
    # uenmax split for the non-Antimagic branch (vendor trap.c:2386-2398).
    # ------------------------------------------------------------------
    drain = _d(k_drain1, 6) + _d(k_drain2, 6)  # 2d6 = 2..12
    # halfd = rnd(drain / 2); range [1, drain/2].
    halfd_upper = jnp.maximum(drain // jnp.int32(2), jnp.int32(1))
    halfd = jax.random.randint(
        k_halfd, (), 1, halfd_upper + jnp.int32(1)
    ).astype(jnp.int32)
    # uenmax > drain gate (vendor line 2388).
    split_uenmax = state.player_pw_max > drain
    pw_max_drain = jnp.where(split_uenmax, halfd, jnp.int32(0))
    pw_drain     = jnp.where(split_uenmax, drain - halfd, drain)

    def _do_antimagic(s):
        # HP damage in place of Pw drain.
        new_hp = jnp.maximum(s.player_hp - dmgval2, jnp.int32(0))
        return s.replace(player_hp=new_hp)

    def _do_pw_drain(s):
        new_pw_max = jnp.maximum(s.player_pw_max - pw_max_drain, jnp.int32(0))
        new_pw     = jnp.minimum(
            jnp.maximum(s.player_pw - pw_drain, jnp.int32(0)),
            new_pw_max,
        )
        return s.replace(player_pw=new_pw, player_pw_max=new_pw_max)

    def _do_default(s):
        return jax.lax.cond(has_antimagic, _do_antimagic, _do_pw_drain, s)

    return jax.lax.cond(iron_shoes_protect, _do_iron_shoes, _do_default, state)


def _trap_poly(state, rng):
    """POLY_TRAP — polymorph the player AND polymorph items on the tile.

    vendor/nethack/src/trap.c::POLY_TRAP — delegates to
    polymorph.poly_trap_effect for the player branch, and to
    traps.poly_pile_effect for the pile-of-items branch
    (vendor/nethack/src/trap.c::do_poly_pile).

    The two effects are independent; vendor evaluates do_poly_pile for the
    tile's object list and then polymorphs the victim that triggered the
    trap.  Wave 5 Phase 4 wires both branches; conduct flags
    POLYPILELESS / POLYSELFLESS are each set in their respective helper.

    Audit M #39: vendor trap.c:2486-2490 short-circuits the player
    polymorph when the victim has ``Antimagic`` or ``Unchanging`` —
    prints "You feel momentarily different" and skips polyself.  The
    pile branch still fires (item poly is independent of magic-resist).
    """
    from Nethax.nethax.subsystems.polymorph import poly_trap_effect
    from Nethax.nethax.subsystems.status_effects import Intrinsic as _PolyIntr

    rng_pile, rng_self = jax.random.split(rng)
    # Pile branch first (independent of Antimagic / Unchanging).
    new_state = poly_pile_effect(
        state, rng_pile, state.player_pos[0], state.player_pos[1]
    )

    # Audit M #39 short-circuit: Antimagic (Nethax: MAGIC_RESIST) OR
    # Unchanging blocks the player polymorph.  Vendor prints "You feel
    # momentarily different" and skips polyself.  The pile branch already
    # fired above (poly_pile is independent of magic-resist).
    # Cite: vendor/nethack/src/trap.c lines 2486-2490.
    intrinsics = new_state.status.intrinsics
    has_antimagic  = intrinsics[int(_PolyIntr.MAGIC_RESIST)]
    has_unchanging = intrinsics[int(_PolyIntr.UNCHANGING)]
    block_poly = has_antimagic | has_unchanging

    return jax.lax.cond(
        block_poly,
        lambda s: s,
        lambda s: poly_trap_effect(s, rng_self),
        new_state,
    )


def _trap_vibrating_square(state, rng):
    """VIBRATING_SQUARE — gateway tile; effect wired by Gehennom agent."""
    return state


def _trap_trapped_door(state, rng):
    """TRAPPED_DOOR — handled by door-open logic, not movement here."""
    return state


def _trap_trapped_chest(state, rng):
    """TRAPPED_CHEST — 1d10 HP damage + 25% poison chance.

    vendor/nethack/src/lock.c lines 104-114:
      losehp(rnd(10), ...) and if (!rn2(4)) poisoned("needle", ...).
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus
    k0, k1, k2, k3 = jax.random.split(rng, 4)
    s = _apply_hp_damage(state, _d(k0, 10))
    # 25% poison: vendor uses !rn2(4).
    poisoned = jax.random.randint(k1, (), 0, 4) == jnp.int32(0)
    poison_turns = _d(k2, 10)

    def _do_poison(s_):
        new_sick = s_.status.replace(
            sick_kind=jnp.int8(1),
            timed_statuses=s_.status.timed_statuses.at[int(TimedStatus.SICK)].set(
                jnp.maximum(s_.status.timed_statuses[int(TimedStatus.SICK)],
                            poison_turns)
            ),
        )
        return s_.replace(status=new_sick)

    return jax.lax.cond(poisoned, _do_poison, lambda s_: s_, s)


# Tuple of branches indexed by TrapType value.  Order MUST match enum.
# Length = N_TRAP_TYPES (26) so jax.lax.switch can index any valid TrapType.
_TRAP_BRANCHES = (
    _trap_noop,             #  0 NO_TRAP
    _trap_arrow,            #  1 ARROW_TRAP
    _trap_dart,             #  2 DART_TRAP
    _trap_rock,             #  3 ROCKTRAP
    _trap_sqky_board,       #  4 SQKY_BOARD
    _trap_bear,             #  5 BEAR_TRAP
    _trap_landmine,         #  6 LANDMINE
    _trap_rolling_boulder,  #  7 ROLLING_BOULDER_TRAP
    _trap_sleep_gas,        #  8 SLP_GAS_TRAP
    _trap_rust,             #  9 RUST_TRAP
    _trap_fire,             # 10 FIRE_TRAP
    _trap_pit,              # 11 PIT
    _trap_spiked_pit,       # 12 SPIKED_PIT
    _trap_hole,             # 13 HOLE
    _trap_trapdoor,         # 14 TRAPDOOR
    _trap_telep,            # 15 TELEP_TRAP
    _trap_level_telep,      # 16 LEVEL_TELEP
    _trap_magic_portal,     # 17 MAGIC_PORTAL
    _trap_web,              # 18 WEB
    _trap_statue,           # 19 STATUE_TRAP
    _trap_magic,            # 20 MAGIC_TRAP
    _trap_anti_magic,       # 21 ANTI_MAGIC
    _trap_poly,             # 22 POLY_TRAP
    _trap_vibrating_square, # 23 VIBRATING_SQUARE
    _trap_trapped_door,     # 24 TRAPPED_DOOR
    _trap_trapped_chest,    # 25 TRAPPED_CHEST
)


def trigger_trap_envstate(state, rng: jax.Array, row, col):
    """Wide-carrier ``lax.switch`` dispatch over all trap types.

    Mirrors vendor/nethack/src/trap.c::dotrap (a single switch(ttyp) on the
    trap-type code at the victim's tile).  Each branch operates on the full
    EnvState and returns the same pytree shape, so JAX can compile the
    dispatch into one fused kernel.

    Parameters
    ----------
    state : EnvState
    rng   : JAX PRNGKey
    row   : int / jnp.int — map row of the trap
    col   : int / jnp.int — map col of the trap

    Returns
    -------
    EnvState — full state after the trap effect has been applied and the
               tile has been marked ``revealed``.
    """
    row_i = jnp.int32(row) if isinstance(row, int) else row.astype(jnp.int32)
    col_i = jnp.int32(col) if isinstance(col, int) else col.astype(jnp.int32)
    flat_lv = _flat_level_idx(state)
    trap_kind = state.traps.trap_type[flat_lv, row_i, col_i].astype(jnp.int32)
    safe_kind = jnp.clip(trap_kind, 0, N_TRAP_TYPES - 1)

    # Mark trap revealed first so all branches see the same trap-state shape.
    new_revealed = state.traps.revealed.at[flat_lv, row_i, col_i].set(True)
    state = state.replace(traps=state.traps.replace(revealed=new_revealed))

    return jax.lax.switch(safe_kind, _TRAP_BRANCHES, state, rng)
