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
                                        depends on the callback (e.g.
                                        ground-item linear index, monster
                                        slot, level id, ...)

Callbacks
---------
``func_idx`` selects one of N fixed callbacks via ``jax.lax.switch``.
Each callback has signature ``(state, target_id) -> EnvState``.  See
``TIMER_CALLBACKS`` for the table and ``_callback_*`` helpers below.

This file does NOT alter EnvState; the timer field must be added
separately in ``state.py`` (one-time plumbing) and ``env.py`` must call
``tick_timers`` once per turn.

Cite: vendor/nethack/src/timeout.c lines 2222-2400 (start/stop/run_timers
infrastructure), 1978-1991 (timeout_funcs dispatch table).
"""
from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp
from flax import struct


MAX_TIMERS: int = 64


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
# Per-turn drain (vendor timeout.c::run_timers lines 2222-2245)
# ---------------------------------------------------------------------------

def _callback_noop(state, target_id):
    return state


def _callback_hatch_egg(state, target_id):
    """Vendor: hatch egg into a tame creature at the egg's tile.

    Wave 47e MVP: target_id encodes the ground-item linear index
    ``(branch * MAX_LEVELS + level) * MAP_H * MAP_W * MAX_STACK + ...``.
    The encoding/decoding scheme is owned by the producer.  Without a
    full ground-item index decoder, this callback no-ops; the slot is
    cleared so the timer doesn't re-fire.  Future enhancement: decode
    target_id, spawn monster, remove egg.

    Cite: vendor/nethack/src/timeout.c::hatch_egg lines 1017-1192.
    """
    # No-op placeholder; logically the slot is cleared when active=False
    # after run_timers writes the new active vector.
    return state


def _callback_fig_transform(state, target_id):
    """Vendor: figurine becomes a monster at its tile.

    Cite: vendor/nethack/src/timeout.c::fig_transform lines 1204-1220.
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

    Cite: vendor/nethack/src/timeout.c::rot_corpse line 1980.
    """
    return state


def _callback_revive_mon(state, target_id):
    """Vendor: corpse revives back into a live monster.

    Cite: vendor/nethack/src/timeout.c::revive_mon line 1982.
    """
    return state


def _callback_zombify_mon(state, target_id):
    """Vendor: corpse rises as a zombie.

    Cite: vendor/nethack/src/timeout.c::zombify_mon line 1983.
    """
    return state


def _callback_melt_ice(state, target_id):
    """Vendor: ICE tile melts back to MOAT/POOL.

    Cite: vendor/nethack/src/timeout.c::melt_ice_away line 1989.
    """
    return state


def _callback_shrink_glob(state, target_id):
    """Vendor: shrink effect reverts.

    Cite: vendor/nethack/src/timeout.c::shrink_glob line 1987.
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

        def _do_fire(s_):
            return jax.lax.switch(fidx, TIMER_CALLBACKS, s_, tgt)

        s2 = jax.lax.cond(fire, _do_fire, lambda x: x, s)
        return s2, None

    final_state, _ = jax.lax.scan(
        _body, state, jnp.arange(MAX_TIMERS, dtype=jnp.int32)
    )

    # Clear all fired slots.
    new_active = ts.active & ~expired
    new_ts = ts.replace(active=new_active)
    return final_state.replace(timers=new_ts)
