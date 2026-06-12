"""Generic per-turn timer queue (vendor src/timeout.c TIMER infrastructure).

This subsystem mirrors the TIMER queue exposed by vendor's
``start_timer`` / ``stop_timer`` / ``run_timers`` triplet
(timeout.c lines 2222-2400).  Vendor uses a sorted linked list of
``timer_element`` structs; we use a fixed-size SoA pytree of N=64 slots
to stay JIT-friendly.

Slot layout
-----------
For each slot ``i`` in ``[0, MAX_TIMERS)``:
  - ``active[i]``     : bool          — True iff this slot is in use
  - ``fire_turn[i]``  : int32         — absolute game turn at which the
                                        callback fires (svm.moves equivalent)
  - ``func_idx[i]``   : int8          — TimerFunc enum value
  - ``target_id[i]``  : int32         — opaque target identifier; encoding
                                        depends on the callback (see
                                        ``encode_ground_slot`` /
                                        ``encode_tile``)

Target-id encoding schemes
--------------------------
There are two encoding schemes, both flat int32 values:

1. **Ground-slot** (used by ROT_CORPSE / REVIVE_MON / ZOMBIFY_MON /
   HATCH_EGG — TIMER_OBJECT in vendor parlance).  Mirrors vendor's
   ``obj_to_any`` which stores a pointer to a single ground-stack item.
   We encode the 5-tuple
   ``(branch, level, row, col, stack_slot)`` as

       tid = (((branch * MAX_LEVELS + level) * MAP_H + row) * MAP_W + col)
             * MAX_GROUND_STACK + stack_slot

   With N_BRANCHES=7, MAX_LEVELS_PER_BRANCH=32, MAP_H=21, MAP_W=80,
   MAX_GROUND_STACK=8 the maximum value is 7*32*21*80*8 ≈ 3.0M, easily
   within int32 range.

2. **Tile** (used by MELT_ICE_AWAY — TIMER_LEVEL in vendor parlance).
   Vendor packs ``(x << 16) | y`` into a long; we use a flat 4-tuple
   ``(branch, level, row, col)`` linear index because per-level ICE
   timers also need branch/level scoping in the JAX port (the timer
   queue is global, not per-level).

       tid = ((branch * MAX_LEVELS + level) * MAP_H + row) * MAP_W + col

Callbacks
---------
``func_idx`` selects one of N fixed callbacks via ``jax.lax.switch``.
Each callback has signature ``(state, target_id) -> EnvState``.  See
``TIMER_CALLBACKS`` for the table and ``_callback_*`` helpers below.

Cite: vendor/nethack/src/timeout.c lines 2222-2400 (start/stop/run_timers
infrastructure), 1978-1991 (timeout_funcs dispatch table).
"""
from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct

from Nethax.nethax.dungeon.branches import (
    MAP_H as _MAP_H,
    MAP_W as _MAP_W,
    MAX_LEVELS_PER_BRANCH as _MAX_LEVELS,
)


MAX_TIMERS: int = 64

# Constants used by encode/decode.  Mirror the values in
# Nethax.nethax.subsystems.inventory (MAX_GROUND_STACK = 8).  Kept as a
# module-level constant to avoid a circular import on inventory.
_MAX_GROUND_STACK: int = 8


class TimerFunc(IntEnum):
    """Vendor src/timeout.c timeout_funcs dispatch indices.

    Aligns ordering with vendor 1978-1991 where possible; gaps reserved
    for future additions.  ``NONE`` (0) means slot is logically empty
    even when ``active=True`` so ``func_idx=0`` is a no-op.
    """
    NONE          = 0
    HATCH_EGG     = 1
    FIG_TRANSFORM = 2
    BURN_OBJECT   = 3
    ROT_CORPSE    = 4
    REVIVE_MON    = 5
    ZOMBIFY_MON   = 6
    MELT_ICE_AWAY = 7
    SHRINK_GLOB   = 8


@struct.dataclass
class TimerState:
    """Fixed-size SoA timer queue."""
    active: jnp.ndarray      # bool[MAX_TIMERS]
    fire_turn: jnp.ndarray   # int32[MAX_TIMERS]
    func_idx: jnp.ndarray    # int8[MAX_TIMERS]
    target_id: jnp.ndarray   # int32[MAX_TIMERS]

    @classmethod
    def default(cls) -> "TimerState":
        return cls(
            active=jnp.zeros((MAX_TIMERS,), dtype=jnp.bool_),
            fire_turn=jnp.zeros((MAX_TIMERS,), dtype=jnp.int32),
            func_idx=jnp.zeros((MAX_TIMERS,), dtype=jnp.int8),
            target_id=jnp.zeros((MAX_TIMERS,), dtype=jnp.int32),
        )


# ---------------------------------------------------------------------------
# target_id encoding / decoding helpers
# ---------------------------------------------------------------------------

def encode_ground_slot(
    branch: jnp.ndarray,
    level: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    stack_slot: jnp.ndarray,
) -> jnp.ndarray:
    """Pack a (branch, level, row, col, stack_slot) tuple into an int32 id.

    See module docstring "Target-id encoding schemes" #1.
    """
    b = branch.astype(jnp.int32)
    l = level.astype(jnp.int32)
    r = row.astype(jnp.int32)
    c = col.astype(jnp.int32)
    s = stack_slot.astype(jnp.int32)
    return (
        (((b * jnp.int32(_MAX_LEVELS) + l) * jnp.int32(_MAP_H) + r)
            * jnp.int32(_MAP_W) + c)
        * jnp.int32(_MAX_GROUND_STACK) + s
    )


def decode_ground_slot(target_id: jnp.ndarray):
    """Inverse of ``encode_ground_slot``; returns (branch, level, row, col, stack_slot)."""
    t = target_id.astype(jnp.int32)
    s = t % jnp.int32(_MAX_GROUND_STACK)
    t = t // jnp.int32(_MAX_GROUND_STACK)
    c = t % jnp.int32(_MAP_W)
    t = t // jnp.int32(_MAP_W)
    r = t % jnp.int32(_MAP_H)
    t = t // jnp.int32(_MAP_H)
    l = t % jnp.int32(_MAX_LEVELS)
    b = t // jnp.int32(_MAX_LEVELS)
    return b, l, r, c, s


def encode_tile(
    branch: jnp.ndarray,
    level: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
) -> jnp.ndarray:
    """Pack a (branch, level, row, col) tuple into an int32 id.

    See module docstring "Target-id encoding schemes" #2.
    """
    b = branch.astype(jnp.int32)
    l = level.astype(jnp.int32)
    r = row.astype(jnp.int32)
    c = col.astype(jnp.int32)
    return ((b * jnp.int32(_MAX_LEVELS) + l) * jnp.int32(_MAP_H) + r) \
        * jnp.int32(_MAP_W) + c


def decode_tile(target_id: jnp.ndarray):
    """Inverse of ``encode_tile``; returns (branch, level, row, col)."""
    t = target_id.astype(jnp.int32)
    c = t % jnp.int32(_MAP_W)
    t = t // jnp.int32(_MAP_W)
    r = t % jnp.int32(_MAP_H)
    t = t // jnp.int32(_MAP_H)
    l = t % jnp.int32(_MAX_LEVELS)
    b = t // jnp.int32(_MAX_LEVELS)
    return b, l, r, c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_timer(
    state,
    relative_when: jnp.ndarray,
    func: TimerFunc,
    target_id: jnp.ndarray,
):
    """Schedule callback ``func`` to fire ``relative_when`` turns from now.

    Vendor cite: timeout.c::start_timer lines 2247-2289.  Vendor inserts
    into the sorted linked list with O(log N) seek; this version places
    the timer in the first inactive slot (O(N) scan via argmin).
    Capacity exhaustion silently drops the timer (vendor returns FALSE).
    """
    ts = state.timers
    when_abs = state.timestep.astype(jnp.int32) + relative_when.astype(jnp.int32)

    # First inactive slot.
    free_mask = ~ts.active
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)

    new_ts = ts.replace(
        active=ts.active.at[slot].set(any_free | ts.active[slot]),
        fire_turn=ts.fire_turn.at[slot].set(
            jnp.where(any_free, when_abs, ts.fire_turn[slot])
        ),
        func_idx=ts.func_idx.at[slot].set(
            jnp.where(any_free, jnp.int8(int(func)), ts.func_idx[slot])
        ),
        target_id=ts.target_id.at[slot].set(
            jnp.where(any_free, target_id.astype(jnp.int32), ts.target_id[slot])
        ),
    )
    return state.replace(timers=new_ts)


def stop_timer(
    state,
    func: TimerFunc,
    target_id: jnp.ndarray,
):
    """Cancel the first timer matching ``(func, target_id)``.

    Vendor cite: timeout.c::stop_timer lines 2299-2360.  Vendor returns
    remaining time; we just clear the slot.
    """
    ts = state.timers
    tid = target_id.astype(jnp.int32)
    match = ts.active & (ts.func_idx == jnp.int8(int(func))) & (ts.target_id == tid)
    any_match = jnp.any(match)
    slot = jnp.argmax(match.astype(jnp.int32)).astype(jnp.int32)
    new_active = jnp.where(any_match, ts.active.at[slot].set(False), ts.active)
    return state.replace(timers=ts.replace(active=new_active))


# ---------------------------------------------------------------------------
# Lookup tables shared by callbacks.
# Built at module load (static, NOT inside jit) and frozen into jnp arrays.
# ---------------------------------------------------------------------------

def _build_monster_lookup_tables():
    """Build per-MONSTERS-row lookup tables used by corpse / revive callbacks.

    Cite:
      - vendor/nethack/src/mkobj.c::start_corpse_timeout lines 1402-1429
        (lizard/lichen no-rot, troll/rider revive, gz.zombify path).
      - vendor/nethack/src/mon.c::zombie_form line 386 (which species
        have a non-NON_PM zombify_form).
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol

    n = len(MONSTERS)

    # is_lizard_or_lichen: ROT_AGE-exempt monsters (mkobj.c:1403-1404).
    # Vendor compares PM_LIZARD / PM_LICHEN literally.  We match by name to
    # avoid wiring a new PM_* constant tree.
    is_lizard_or_lichen = [
        m.name in ("lizard", "lichen") for m in MONSTERS
    ]

    # is_troll: revive on a 1/TROLL_REVIVE_CHANCE per-age roll
    # (mkobj.c:1418-1424).  S_TROLL = 46.
    is_troll = [m.symbol == int(MonsterSymbol.S_TROLL) for m in MONSTERS]

    # is_zombie: monsters whose mlet is S_ZOMBIE.  Vendor's zombie_form()
    # returns the matching zombie species for a wide range of humanoids;
    # here we conservatively treat anything whose own symbol is already
    # S_ZOMBIE as "stays a zombie" — a non-NON_PM zombify_form.  For
    # non-zombie species the timer falls through to ROT_CORPSE at callback
    # time, matching vendor do.c::zombify_mon line 2312-2314 fallback.
    is_zombie_form = [m.symbol == int(MonsterSymbol.S_ZOMBIE) for m in MONSTERS]

    return (
        jnp.array(is_lizard_or_lichen, dtype=jnp.bool_),
        jnp.array(is_troll, dtype=jnp.bool_),
        jnp.array(is_zombie_form, dtype=jnp.bool_),
    )


_IS_LIZARD_OR_LICHEN, _IS_TROLL, _IS_ZOMBIE_FORM = _build_monster_lookup_tables()


# Object type ids needed by callbacks.  Mirrors throwing.py / combat.py
# definitions to avoid a heavy import.  Cite: vendor onames.h.
_OTYP_EGG:    int = 241   # vendor objects.h FOOD class EGG entry.
_OTYP_CORPSE: int = 260   # vendor objects.h FOOD class CORPSE entry.

# ItemCategory.FOOD value (matches inventory.py::ItemCategory.FOOD).
_FOOD_CATEGORY: int = 7


def _clear_ground_slot(state, branch, level, row, col, stack_slot):
    """Zero out a single ground-item slot (category=0 == empty).

    Used by ROT_CORPSE / ZOMBIFY_MON / HATCH_EGG to remove the now-consumed
    item from the ground stack.  Only ``category`` is touched: leaving the
    other fields stale is fine because every reader gates on
    ``category != 0`` first.
    """
    gi = state.ground_items
    new_cat = gi.category.at[branch, level, row, col, stack_slot].set(jnp.int8(0))
    return state.replace(ground_items=gi.replace(category=new_cat))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _callback_noop(state, target_id):
    return state


def _callback_hatch_egg(state, target_id):
    """Vendor: hatch egg into a creature at the egg's tile.

    Vendor cite: timeout.c::hatch_egg lines 1015-1189.

    JAX implementation: the embedded monster species is read from
    ``ground_items.corpse_entry_idx`` (vendor's ``obj->corpsenm``); the
    egg is removed from the ground stack on hatch.  Sterilised eggs
    (corpsenm == -1, vendor NON_PM at line 1029) and zero-quantity slots
    are skipped, matching vendor's early-return.

    Full vendor behaviour (spawning a tame monster via ``makemon`` at
    ``enexto`` coords, ``hatchcount`` based on stack quantity, "egg
    drops from your pack" messaging) requires a clean monster-slot
    allocator that operates on EnvState; that wiring is deferred.
    The egg removal alone keeps the timer queue logically correct: the
    egg disappears at hatch time rather than persisting forever.
    """
    b, l, r, c, s = decode_ground_slot(target_id)
    gi = state.ground_items

    # Index-safety clips so out-of-range ids degrade to a no-op.
    safe_b = jnp.clip(b, 0, gi.category.shape[0] - 1)
    safe_l = jnp.clip(l, 0, gi.category.shape[1] - 1)
    safe_r = jnp.clip(r, 0, gi.category.shape[2] - 1)
    safe_c = jnp.clip(c, 0, gi.category.shape[3] - 1)
    safe_s = jnp.clip(s, 0, gi.category.shape[4] - 1)

    cat   = gi.category[safe_b, safe_l, safe_r, safe_c, safe_s]
    tid   = gi.type_id[safe_b, safe_l, safe_r, safe_c, safe_s]
    cnm   = gi.corpse_entry_idx[safe_b, safe_l, safe_r, safe_c, safe_s].astype(jnp.int32)

    # vendor 1029: sterilised egg (corpsenm == NON_PM) just returns.
    is_egg = (cat == jnp.int8(_FOOD_CATEGORY)) & (tid == jnp.int16(_OTYP_EGG))
    valid = is_egg & (cnm >= jnp.int32(0))

    new_cat = gi.category.at[safe_b, safe_l, safe_r, safe_c, safe_s].set(
        jnp.where(valid, jnp.int8(0), cat)
    )
    return state.replace(ground_items=gi.replace(category=new_cat))


def _callback_fig_transform(state, target_id):
    """Vendor: figurine becomes a monster at its tile.

    Cite: vendor/nethack/src/timeout.c::fig_transform lines 1204-1220.
    DEFERRED — figurines need new state plumbing; see task spec.
    """
    return state


def _callback_burn_object(state, target_id):
    """Vendor: light source extinguishes (lamp/candle/artifact).

    The existing ``lighting.py::source_until_turn`` already implements
    timed expiry via timestep comparison.  This callback exists for
    parity with vendor's TIMER queue model so future light sources can
    be migrated; it is currently a no-op.

    Cite: vendor/nethack/src/timeout.c::burn_object lines 1383-1712.
    """
    return state


def _callback_rot_corpse(state, target_id):
    """Vendor: corpse passes the no-longer-edible threshold.

    Cite: vendor/nethack/src/timeout.c::rot_corpse line 1980;
          vendor/nethack/src/dig.c::rot_corpse lines 2145-2189
          (obj_extract_self + obfree path on OBJ_FLOOR).

    JAX implementation: removes the corpse from the ground stack.  The
    OBJ_INVENT / OBJ_MINVENT branches in vendor (lines 2156-2173) do not
    apply because our ROT_CORPSE timers are only armed on floor corpses
    (combat.py::_place_corpse).  A monster hiding under the corpse
    becoming exposed (lines 2179-2186) is a display/AI nicety not
    modelled here.
    """
    b, l, r, c, s = decode_ground_slot(target_id)
    gi = state.ground_items
    safe_b = jnp.clip(b, 0, gi.category.shape[0] - 1)
    safe_l = jnp.clip(l, 0, gi.category.shape[1] - 1)
    safe_r = jnp.clip(r, 0, gi.category.shape[2] - 1)
    safe_c = jnp.clip(c, 0, gi.category.shape[3] - 1)
    safe_s = jnp.clip(s, 0, gi.category.shape[4] - 1)

    cat = gi.category[safe_b, safe_l, safe_r, safe_c, safe_s]
    tid = gi.type_id[safe_b, safe_l, safe_r, safe_c, safe_s]
    is_corpse = (cat == jnp.int8(_FOOD_CATEGORY)) & (tid == jnp.int16(_OTYP_CORPSE))

    new_cat = gi.category.at[safe_b, safe_l, safe_r, safe_c, safe_s].set(
        jnp.where(is_corpse, jnp.int8(0), cat)
    )
    return state.replace(ground_items=gi.replace(category=new_cat))


def _callback_revive_mon(state, target_id):
    """Vendor: corpse revives back into a live monster.

    Cite: vendor/nethack/src/timeout.c::revive_mon line 1982;
          vendor/nethack/src/do.c::revive_mon lines 2248-2295.

    JAX implementation: removes the corpse from the ground stack to
    represent the "if we succeed, the corpse is gone" outcome (vendor
    do.c:2277).  Full vendor behaviour also re-spawns a live monster
    via ``revive_corpse`` → ``makemon``; that requires a clean monster
    slot allocator hooked into EnvState.monster_ai and is deferred.

    Until the spawn path lands, the callback is conservative: it just
    consumes the corpse so the timer slot is freed and the stack
    eventually clears.  Without spawn-back, this effectively behaves
    like ROT_CORPSE for revivable monsters — acceptable scaffolding.
    """
    b, l, r, c, s = decode_ground_slot(target_id)
    gi = state.ground_items
    safe_b = jnp.clip(b, 0, gi.category.shape[0] - 1)
    safe_l = jnp.clip(l, 0, gi.category.shape[1] - 1)
    safe_r = jnp.clip(r, 0, gi.category.shape[2] - 1)
    safe_c = jnp.clip(c, 0, gi.category.shape[3] - 1)
    safe_s = jnp.clip(s, 0, gi.category.shape[4] - 1)

    cat = gi.category[safe_b, safe_l, safe_r, safe_c, safe_s]
    tid = gi.type_id[safe_b, safe_l, safe_r, safe_c, safe_s]
    is_corpse = (cat == jnp.int8(_FOOD_CATEGORY)) & (tid == jnp.int16(_OTYP_CORPSE))

    new_cat = gi.category.at[safe_b, safe_l, safe_r, safe_c, safe_s].set(
        jnp.where(is_corpse, jnp.int8(0), cat)
    )
    return state.replace(ground_items=gi.replace(category=new_cat))


def _callback_zombify_mon(state, target_id):
    """Vendor: corpse rises as a zombie.

    Cite: vendor/nethack/src/timeout.c::zombify_mon line 1983;
          vendor/nethack/src/do.c::zombify_mon lines 2297-2315.

    Vendor logic (do.c:2299-2314):
        zmon = zombie_form(&mons[body->corpsenm]);
        if (zmon != NON_PM && !G_GENOD) {
            set_corpsenm(body, zmon);
            revive_mon(arg, timeout);   // -> revive as zombie
        } else {
            rot_corpse(arg, timeout);   // fall through
        }

    JAX implementation: looks up ``_IS_ZOMBIE_FORM`` for the corpse's
    species; if the species has a zombify form we behave like
    ``_callback_revive_mon`` (consume corpse — full spawn deferred);
    otherwise we behave like ``_callback_rot_corpse`` (consume corpse).
    The observable side-effect is the same — corpse removed — but the
    branch structure mirrors vendor for future spawn-path wiring.
    """
    b, l, r, c, s = decode_ground_slot(target_id)
    gi = state.ground_items
    safe_b = jnp.clip(b, 0, gi.category.shape[0] - 1)
    safe_l = jnp.clip(l, 0, gi.category.shape[1] - 1)
    safe_r = jnp.clip(r, 0, gi.category.shape[2] - 1)
    safe_c = jnp.clip(c, 0, gi.category.shape[3] - 1)
    safe_s = jnp.clip(s, 0, gi.category.shape[4] - 1)

    cat = gi.category[safe_b, safe_l, safe_r, safe_c, safe_s]
    tid = gi.type_id[safe_b, safe_l, safe_r, safe_c, safe_s]
    cnm = gi.corpse_entry_idx[safe_b, safe_l, safe_r, safe_c, safe_s].astype(jnp.int32)
    safe_cnm = jnp.clip(cnm, 0, _IS_ZOMBIE_FORM.shape[0] - 1)

    is_corpse = (cat == jnp.int8(_FOOD_CATEGORY)) & (tid == jnp.int16(_OTYP_CORPSE))
    # has_zombify is read for future spawn-path branching; behavioural
    # effect is identical (consume corpse) regardless.
    _has_zombify = _IS_ZOMBIE_FORM[safe_cnm] & (cnm >= jnp.int32(0))

    new_cat = gi.category.at[safe_b, safe_l, safe_r, safe_c, safe_s].set(
        jnp.where(is_corpse, jnp.int8(0), cat)
    )
    return state.replace(ground_items=gi.replace(category=new_cat))


def _callback_melt_ice(state, target_id):
    """Vendor: ICE tile melts back to MOAT/POOL.

    Cite: vendor/nethack/src/timeout.c::melt_ice_away line 1989;
          vendor/nethack/src/zap.c::melt_ice_away lines 5118-5132 →
          ``melt_ice`` at the tile.

    JAX implementation: if the encoded tile is ICE_FLOOR, convert it
    back to WATER.  Other tile types are left untouched (the ice may
    have been destroyed earlier by digging / other effects).
    """
    from Nethax.nethax.constants.tiles import TileType
    b, l, r, c = decode_tile(target_id)
    safe_b = jnp.clip(b, 0, state.terrain.shape[0] - 1)
    safe_l = jnp.clip(l, 0, state.terrain.shape[1] - 1)
    safe_r = jnp.clip(r, 0, state.terrain.shape[2] - 1)
    safe_c = jnp.clip(c, 0, state.terrain.shape[3] - 1)

    cur = state.terrain[safe_b, safe_l, safe_r, safe_c]
    is_ice = cur == jnp.int8(int(TileType.ICE_FLOOR))
    new_tile = jnp.where(is_ice, jnp.int8(int(TileType.WATER)), cur)
    new_terrain = state.terrain.at[safe_b, safe_l, safe_r, safe_c].set(new_tile)
    return state.replace(terrain=new_terrain)


def _callback_shrink_glob(state, target_id):
    """Vendor: shrink effect reverts.

    Cite: vendor/nethack/src/timeout.c::shrink_glob lines 1987.
    DEFERRED — potion shrink not implemented; see task spec.
    """
    return state


# Order must match TimerFunc enum.  All callbacks share the same
# ``(state, target_id) -> state`` signature so lax.switch is well-typed.
TIMER_CALLBACKS = (
    _callback_noop,            # 0  NONE
    _callback_hatch_egg,       # 1  HATCH_EGG
    _callback_fig_transform,   # 2  FIG_TRANSFORM
    _callback_burn_object,     # 3  BURN_OBJECT
    _callback_rot_corpse,      # 4  ROT_CORPSE
    _callback_revive_mon,      # 5  REVIVE_MON
    _callback_zombify_mon,     # 6  ZOMBIFY_MON
    _callback_melt_ice,        # 7  MELT_ICE_AWAY
    _callback_shrink_glob,     # 8  SHRINK_GLOB
)


# ---------------------------------------------------------------------------
# Convenience producers — call these from "creation sites" to arm timers
# with the correct encoding scheme.
# ---------------------------------------------------------------------------

# Vendor mkobj.c::start_corpse_timeout constants (lines 1396-1429).
# ROT_AGE is the corpse-edibility / decay threshold, defined in
# vendor/nethack/include/mkroom.h as ROT_AGE 250.  The added jitter
# (``rnz(rot_adjust)``) is approximated here as a fixed mid-range value
# to avoid an extra rng split at every kill — exact jitter is a parity
# follow-up.
_ROT_AGE: int = 250


def start_corpse_timer(
    state,
    branch: jnp.ndarray,
    level: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    stack_slot: jnp.ndarray,
    corpse_entry_idx: jnp.ndarray,
):
    """Arm the appropriate corpse timer (ROT_CORPSE / REVIVE_MON).

    Mirrors vendor mkobj.c::start_corpse_timeout lines 1388-1432 with
    the simplifications documented in ``_build_monster_lookup_tables``:

      * lizards and lichens get no timer (line 1402-1404).
      * trolls get REVIVE_MON with a short fuse (line 1418-1424).
      * everything else gets ROT_CORPSE with vendor's ROT_AGE fuse.

    ZOMBIFY_MON arming requires the ``gz.zombify`` global flag that
    vendor sets in monst.c::summoner paths; not modelled here yet, so
    the ZOMBIFY_MON callback only fires when something else arms it
    (kept callable for future producers).
    """
    safe_cnm = jnp.clip(
        corpse_entry_idx.astype(jnp.int32), 0, _IS_LIZARD_OR_LICHEN.shape[0] - 1
    )
    is_inert  = _IS_LIZARD_OR_LICHEN[safe_cnm] & (corpse_entry_idx >= jnp.int32(0))
    is_troll  = _IS_TROLL[safe_cnm] & (corpse_entry_idx >= jnp.int32(0))

    tid = encode_ground_slot(branch, level, row, col, stack_slot)

    # Troll: short revive fuse (vendor uses an age-iter loop; we pick the
    # midpoint of vendor's 2..TAINT_AGE=50 range).
    troll_when = jnp.int32(25)
    # Plain rot: ROT_AGE turns from now.
    rot_when   = jnp.int32(_ROT_AGE)

    func_val = jnp.where(
        is_troll, jnp.int8(int(TimerFunc.REVIVE_MON)),
        jnp.int8(int(TimerFunc.ROT_CORPSE)),
    )
    when = jnp.where(is_troll, troll_when, rot_when)

    # Inert corpses (lizard/lichen) skip the timer entirely.
    timed = _start_timer_dyn(state, when, func_val, tid)
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(is_inert, a, b), state, timed,
    )


def _start_timer_dyn(state, when, func_val_int8, target_id):
    """Variant of ``start_timer`` whose func index is a tracer, not a Python
    enum.  Used when the callback selection depends on traced data
    (e.g. troll vs non-troll corpse).
    """
    ts = state.timers
    when_abs = state.timestep.astype(jnp.int32) + when.astype(jnp.int32)
    free_mask = ~ts.active
    any_free = jnp.any(free_mask)
    slot = jnp.argmax(free_mask.astype(jnp.int32)).astype(jnp.int32)
    new_ts = ts.replace(
        active=ts.active.at[slot].set(any_free | ts.active[slot]),
        fire_turn=ts.fire_turn.at[slot].set(
            jnp.where(any_free, when_abs, ts.fire_turn[slot])
        ),
        func_idx=ts.func_idx.at[slot].set(
            jnp.where(any_free, func_val_int8, ts.func_idx[slot])
        ),
        target_id=ts.target_id.at[slot].set(
            jnp.where(any_free, target_id.astype(jnp.int32), ts.target_id[slot])
        ),
    )
    return state.replace(timers=new_ts)


def start_egg_hatch_timer(
    state,
    branch: jnp.ndarray,
    level: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    stack_slot: jnp.ndarray,
    when: jnp.ndarray,
):
    """Arm a HATCH_EGG timer for the egg at the given ground slot.

    Vendor cite: timeout.c::attach_egg_hatch_timeout lines 974-1005.
    Vendor uses a random ``i = rnd(12)`` short timer for re-arming and
    a longer ``rnz(150)`` initial timer; we accept ``when`` from the
    caller for flexibility.
    """
    tid = encode_ground_slot(branch, level, row, col, stack_slot)
    return start_timer(state, when, TimerFunc.HATCH_EGG, tid)


def start_melt_ice_timer(
    state,
    branch: jnp.ndarray,
    level: jnp.ndarray,
    row: jnp.ndarray,
    col: jnp.ndarray,
    when: jnp.ndarray,
):
    """Arm a MELT_ICE_AWAY timer for the tile (branch, level, row, col).

    Vendor cite: zap.c::start_melt_ice_timeout lines 5087-5111.  Vendor
    samples ``when`` from a 50..2000 range with a rejection loop; we
    accept ``when`` directly so callers can pick a fixed or pre-sampled
    value (keeps JIT pure with no embedded while_loop).
    """
    tid = encode_tile(branch, level, row, col)
    return start_timer(state, when, TimerFunc.MELT_ICE_AWAY, tid)


# ---------------------------------------------------------------------------
# Per-turn drain (vendor timeout.c::run_timers lines 2222-2245)
# ---------------------------------------------------------------------------

def tick_timers(state):
    """Fire all timers whose ``fire_turn <= state.timestep`` then clear them.

    Vendor cite: timeout.c::run_timers lines 2222-2245.  Vendor iterates
    the sorted list while ``head->when <= moves`` and pops each.  This
    implementation scans the fixed-size SoA once.

    Each fired timer invokes its callback via ``jax.lax.switch`` then is
    deactivated.  Within the scan, ``active`` reads stale state — that's
    OK because we deactivate at the end (callbacks are pure on state).
    """
    ts = state.timers
    now = state.timestep.astype(jnp.int32)

    # Per-slot fire mask.
    expired = ts.active & (ts.fire_turn <= now)

    def _body(carry_state, slot_idx):
        s = carry_state
        fire = expired[slot_idx]
        fidx = ts.func_idx[slot_idx].astype(jnp.int32)
        tgt  = ts.target_id[slot_idx]

        # Brax-flat: compute all timer callbacks then cascading jnp.where on fidx,
        # then mask the whole result by `fire`.
        cb_results = [cb(s, tgt) for cb in TIMER_CALLBACKS]
        cb_sel = cb_results[0]
        for i in range(1, len(cb_results)):
            cb_sel = jax.tree_util.tree_map(
                lambda a, b, i=i: jnp.where(fidx == jnp.int32(i), a, b),
                cb_results[i], cb_sel,
            )
        s2 = jax.tree_util.tree_map(
            lambda a, b: jnp.where(fire, a, b), cb_sel, s,
        )
        return s2, None

    final_state, _ = jax.lax.scan(
        _body, state, jnp.arange(MAX_TIMERS, dtype=jnp.int32)
    )

    # Clear all fired slots.
    new_active = ts.active & ~expired
    new_ts = ts.replace(active=new_active)
    return final_state.replace(timers=new_ts)
