"""Vendor-parity RNG port — ISAAC64 + NetHack helpers.

Goal
----
Provide a side-by-side alternative to Threefry that is BYTE-IDENTICAL with
the RNG NLE actually executes.  The exact PRNG used by NLE is established by
auditing the vendor sources:

    vendor/nle/include/config.h:584   ->  ``#define USE_ISAAC64``
    vendor/nle/src/rnd.c              ->  ``#ifdef USE_ISAAC64`` path active
    vendor/nle/src/isaac64.c          ->  Terriberry's public-domain ISAAC64
    vendor/nle/src/hacklib.c:854-868  ->  ``set_random`` -> ``init_isaac64``
    vendor/nle/src/nle.c:410-411      ->  per-RNG seed via ``nle_seeds_init``

NetHack's vendor tree also ships an older BSD ``random()`` (TYPE_3, deg=31,
sep=3) in ``vendor/nethack/sys/share/random.c``.  That file is **only
compiled** when ``USE_ISAAC64`` is undefined.  Because NLE always defines
``USE_ISAAC64``, the BSD random() path is dead code for any NLE rollout.
Reproducing it would NOT yield byte parity with NLE.

This module therefore implements ISAAC64 (the live path) and exposes the
NetHack helpers (``rn2``, ``rnd``, ``rn1``, ``rnl``, ``rne``, ``d``) on top
of it, exactly matching ``vendor/nethack/src/rnd.c``.

Status: SKELETON.  The pure-Python ISAAC64 reference is byte-exact with the
C implementation; the JAX wrapper carries a stateful pytree but is *not yet*
wired into ``EnvState`` -- audit deliverable.  See companion audit notes in
this commit for migration plan.

Public API
----------
- ``Isaac64State`` -- frozen pytree carrying ``(a, b, c, m, r, n)``
- ``init(seed: int) -> Isaac64State``
- ``next_uint64(state) -> (state, uint64)``
- ``rn2(state, x)``, ``rnd(state, x)``, ``rn1(state, n, x)``,
  ``rnl(state, x, luck=0)``, ``rne(state, x, ulevel=14)``,
  ``d(state, n, x)``

All helpers return ``(new_state, value)`` to remain functionally pure under
JAX.  This is the price of vendor parity: ISAAC64 is stateful, so the caller
must thread the new state through every site that previously consumed a
Threefry key.

The reference Python implementation mirrors ``vendor/nle/src/isaac64.c``
line-for-line so it can be diff-validated against the C source.

JAX wrapper
-----------
``Isaac64State`` is a ``@flax.struct.dataclass``-style frozen pytree (using
``jax.tree_util.register_pytree_node_class``) so it can live inside
``EnvState`` and be threaded through ``jit``/``vmap`` like any other state.
"""
from __future__ import annotations

import os as _os
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# RNG trace (NETHAX_RNG_TRACE=/path/to/file) — host-side draws only.
#
# Mirrors the vendor instrumentation in vendor/nle/src/isaac64.c + rnd.c so
# we can diff sequence-by-sequence against the vendor stream.
# ---------------------------------------------------------------------------
_TRACE_FP = None
_TRACE_OPS_FP = None
_TRACE_ISAAC_COUNTER = 0
_TRACE_OP_COUNTER = 0
_TRACE_INITED = False

# ---------------------------------------------------------------------------
# Draw counter — counts every host-side ISAAC64 uint64 draw.
# JAX-traced draws (rn2_jax, rn1_jax, etc.) are NOT counted here because
# they run inside lax.fori_loop / lax.while_loop and bypass _trace_isaac.
# Use get_draw_count() / reset_draw_count() to instrument env.reset() phases.
# ---------------------------------------------------------------------------
_DRAW_COUNT: int = 0


def get_draw_count() -> int:
    """Return the current host-side ISAAC64 draw count."""
    return _DRAW_COUNT


def reset_draw_count() -> None:
    """Reset the host-side draw counter to zero."""
    global _DRAW_COUNT
    _DRAW_COUNT = 0


def _trace_init():
    global _TRACE_FP, _TRACE_OPS_FP, _TRACE_INITED
    if _TRACE_INITED:
        return
    p = _os.environ.get("NETHAX_RNG_TRACE")
    if p:
        _TRACE_FP = open(p, "w")
    p2 = _os.environ.get("NETHAX_RNG_TRACE_OPS")
    if p2:
        _TRACE_OPS_FP = open(p2, "w")
    _TRACE_INITED = True


def _trace_isaac(val: int):
    global _TRACE_ISAAC_COUNTER, _DRAW_COUNT
    _DRAW_COUNT += 1
    _trace_init()
    if _TRACE_FP is not None:
        _TRACE_FP.write(f"ISAAC {_TRACE_ISAAC_COUNTER} val={val:016x}\n")
        _TRACE_FP.flush()
        _TRACE_ISAAC_COUNTER += 1


def _trace_op(op: str, modulus, result):
    global _TRACE_OP_COUNTER
    _trace_init()
    if _TRACE_OPS_FP is not None:
        _TRACE_OPS_FP.write(f"{_TRACE_OP_COUNTER} {op} mod={modulus} res={result}\n")
        _TRACE_OPS_FP.flush()
        _TRACE_OP_COUNTER += 1


# ---------------------------------------------------------------------------
# JIT-aware trace: jax.debug.callback fires both during JIT tracing AND at
# runtime inside lax.scan / lax.while_loop bodies, so this lets us see draws
# that the host-only _trace_op misses.  Gated by env var so prod stays fast.
# ---------------------------------------------------------------------------
_JIT_TRACE_ENABLED = None  # cached env-var lookup


def _jit_trace_enabled() -> bool:
    global _JIT_TRACE_ENABLED
    if _JIT_TRACE_ENABLED is None:
        _JIT_TRACE_ENABLED = bool(_os.environ.get("NETHAX_RNG_TRACE_OPS_JIT"))
    return _JIT_TRACE_ENABLED


def _emit_op_callback(op_bytes, modulus, result):
    """Concrete callback target — runs on host with concrete np arrays."""
    op = op_bytes.decode() if isinstance(op_bytes, (bytes, bytearray)) else str(op_bytes)
    try:
        m = int(np.asarray(modulus))
        r = int(np.asarray(result))
    except Exception:
        m = modulus
        r = result
    _trace_op(op, m, r)


# ---------------------------------------------------------------------------
# Constants (mirror vendor/nle/include/isaac64.h)
# ---------------------------------------------------------------------------

ISAAC64_SZ_LOG = 8
ISAAC64_SZ = 1 << ISAAC64_SZ_LOG          # 256
ISAAC64_SEED_SZ_MAX = ISAAC64_SZ << 3     # 2048
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C13                # vendor constant


# ---------------------------------------------------------------------------
# Reference Python ISAAC64 -- byte-exact with vendor/nle/src/isaac64.c
# ---------------------------------------------------------------------------

def _u64(x: int) -> int:
    return x & MASK64


def _isaac64_mix(x: list[int]) -> None:
    """Mirror ``isaac64_mix`` (vendor/nle/src/isaac64.c:99-111)."""
    shift = (9, 9, 23, 15, 14, 20, 17, 14)
    i = 0
    while i < 8:
        x[i] = _u64(x[i] - x[(i + 4) & 7])
        x[(i + 5) & 7] = _u64(x[(i + 5) & 7] ^ (x[(i + 7) & 7] >> shift[i]))
        x[(i + 7) & 7] = _u64(x[(i + 7) & 7] + x[i])
        i += 1
        x[i] = _u64(x[i] - x[(i + 4) & 7])
        x[(i + 5) & 7] = _u64(x[(i + 5) & 7] ^ (x[(i + 7) & 7] << shift[i]) & MASK64)
        x[(i + 7) & 7] = _u64(x[(i + 7) & 7] + x[i])
        i += 1


def _isaac64_update(m: list[int], r: list[int], a: int, b: int, c: int) -> tuple[int, int, int]:
    """Mirror ``isaac64_update`` (vendor/nle/src/isaac64.c:46-97)."""
    c = _u64(c + 1)
    b = _u64(b + c)
    half = ISAAC64_SZ // 2

    def lower(x: int) -> int:
        return (x & ((ISAAC64_SZ - 1) << 3)) >> 3

    def upper(y: int) -> int:
        return (y >> (ISAAC64_SZ_LOG + 3)) & (ISAAC64_SZ - 1)

    # First half: paired with m[i + ISAAC64_SZ/2]
    i = 0
    while i < half:
        x = m[i]
        a = _u64(_u64((~(a ^ (a << 21))) & MASK64) + m[i + half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1
        x = m[i]
        a = _u64((a ^ (a >> 5)) + m[i + half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1
        x = m[i]
        a = _u64((a ^ ((a << 12) & MASK64)) + m[i + half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1
        x = m[i]
        a = _u64((a ^ (a >> 33)) + m[i + half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1

    # Second half: paired with m[i - ISAAC64_SZ/2]
    while i < ISAAC64_SZ:
        x = m[i]
        a = _u64(_u64((~(a ^ (a << 21))) & MASK64) + m[i - half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1
        x = m[i]
        a = _u64((a ^ (a >> 5)) + m[i - half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1
        x = m[i]
        a = _u64((a ^ ((a << 12) & MASK64)) + m[i - half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1
        x = m[i]
        a = _u64((a ^ (a >> 33)) + m[i - half])
        y = _u64(m[lower(x)] + a + b)
        m[i] = y
        b = _u64(m[upper(y)] + x)
        r[i] = b
        i += 1

    return a, b, c


def _isaac64_init_py(seed_bytes: bytes) -> tuple[list[int], list[int], int, int, int, int]:
    """Mirror ``isaac64_init`` + ``isaac64_reseed`` (vendor isaac64.c:114-155)."""
    m = [0] * ISAAC64_SZ
    r = [0] * ISAAC64_SZ
    a = b = c = 0

    # Reseed: pack seed bytes into r[] little-endian, 8 bytes per slot.
    nseed = min(len(seed_bytes), ISAAC64_SEED_SZ_MAX)
    i = 0
    while i < nseed >> 3:
        ri = 0
        for k in range(8):
            ri |= seed_bytes[(i << 3) | k] << (k * 8)
        r[i] ^= ri
        i += 1
    rem = nseed - (i << 3)
    if rem > 0:
        ri = seed_bytes[i << 3]
        for j in range(1, rem):
            ri |= seed_bytes[(i << 3) | j] << (j * 8)
        r[i] ^= ri

    x = [GOLDEN] * 8
    for _ in range(4):
        _isaac64_mix(x)
    for i in range(0, ISAAC64_SZ, 8):
        for j in range(8):
            x[j] = _u64(x[j] + r[i + j])
        _isaac64_mix(x)
        for j in range(8):
            m[i + j] = x[j]
    for i in range(0, ISAAC64_SZ, 8):
        for j in range(8):
            x[j] = _u64(x[j] + m[i + j])
        _isaac64_mix(x)
        for j in range(8):
            m[i + j] = x[j]

    a, b, c = _isaac64_update(m, r, a, b, c)
    n = ISAAC64_SZ
    return m, r, a, b, c, n


def init_py(seed: int) -> tuple[list[int], list[int], int, int, int, int]:
    """Initialize ISAAC64 the way NLE does (see ``init_isaac64`` in rnd.c:42-58).

    NLE packs the seed as ``sizeof(unsigned long)`` little-endian bytes.  On
    LP64 platforms (Linux/macOS, where NLE is built) that is 8 bytes.
    """
    seed_bytes = (seed & MASK64).to_bytes(8, "little")
    return _isaac64_init_py(seed_bytes)


def next_uint64_py(
    state: tuple[list[int], list[int], int, int, int, int],
) -> tuple[tuple[list[int], list[int], int, int, int, int], int]:
    """Mirror ``isaac64_next_uint64`` (vendor isaac64.c:157-160).

    ISAAC fills ``r[]`` in batches of 256 and drains from the top
    (``r[--n]``).  When ``n == 0``, refill via ``isaac64_update``.
    """
    m, r, a, b, c, n = state
    if n == 0:
        a, b, c = _isaac64_update(m, r, a, b, c)
        n = ISAAC64_SZ
    n -= 1
    val = r[n]
    _trace_isaac(val)
    return (m, r, a, b, c, n), val


# ---------------------------------------------------------------------------
# NetHack helpers -- byte-exact with vendor/nethack/src/rnd.c
# ---------------------------------------------------------------------------
#
# Recall the vendor macros (rnd.c:60-64 under USE_ISAAC64):
#     static int RND(int x) { return isaac64_next_uint64(rng) % x; }
#
# So rn2/rnd/rn1/d/rnl/rne all bottom out in ``next_uint64 % x``.

def rn2_py(state, x: int) -> tuple:
    """``rn2(x)`` -- ``isaac64_next_uint64() % x``."""
    state, v = next_uint64_py(state)
    result = v % x
    _trace_op("rn2", x, result)
    return state, result


def rnd_py(state, x: int) -> tuple:
    """``rnd(x)`` -- ``rn2(x) + 1``."""
    # Direct draw so the op trace shows "rnd" not "rn2" — mirror vendor rnd.c
    m, r, a, b, c, n = state
    if n == 0:
        a, b, c = _isaac64_update(m, r, a, b, c)
        n = ISAAC64_SZ
    n -= 1
    val = r[n]
    _trace_isaac(val)
    result = (val % x) + 1
    _trace_op("rnd", x, result)
    return (m, r, a, b, c, n), result


def rn1_py(state, n: int, x: int) -> tuple:
    """``rn1(n, x)`` -- ``x + rn2(n)``."""
    state, v = rn2_py(state, n)
    return state, x + v


def d_py(state, n: int, x: int) -> tuple:
    """``d(n, x)`` -- sum of n rolls in [1, x]."""
    tmp = n
    for _ in range(n):
        state, v = next_uint64_py(state)
        tmp += v % x
    return state, tmp


def rne_py(state, x: int, ulevel: int = 14) -> tuple:
    """``rne(x)`` -- vendor rnd.c:191-210.

    ``utmp = (ulevel < 15) ? 5 : ulevel / 3``; tmp starts at 1, increments
    while ``tmp < utmp && rn2(x) == 0``.
    """
    utmp = 5 if ulevel < 15 else ulevel // 3
    tmp = 1
    while tmp < utmp:
        state, v = rn2_py(state, x)
        if v != 0:
            break
        tmp += 1
    return state, tmp


def rnl_py(state, x: int, luck: int = 0) -> tuple:
    """``rnl(x)`` -- vendor rnd.c:111-150.  Luck-biased uniform."""
    adjustment = luck
    if x <= 15:
        a = (abs(luck) + 1) // 3
        adjustment = a if luck > 0 else (-a if luck < 0 else 0)

    state, i = rn2_py(state, x)
    if adjustment != 0:
        state, gate = rn2_py(state, 37 + abs(adjustment))
        if gate != 0:
            i -= adjustment
            if i < 0:
                i = 0
            elif i >= x:
                i = x - 1
    return state, i


# ---------------------------------------------------------------------------
# JAX wrapper -- carry ISAAC64 state through pytree
# ---------------------------------------------------------------------------
#
# The state is large (256 m[] + 256 r[] + a + b + c + n = ~516 int64 words),
# but pure-functional: every call returns a new state.  We expose it as a
# registered pytree so JAX transformations (vmap/jit/scan) can carry it.

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Isaac64State:
    """JAX-threadable ISAAC64 state -- one slot per ``rnglist[]`` entry.

    Fields mirror ``isaac64_ctx`` (vendor isaac64.h:29-36):
        m : uint64[256]   -- internal mixing array
        r : uint64[256]   -- output buffer (drained top-down via ``n``)
        a, b, c : uint64  -- scalar mixing state
        n : int32         -- index into r[]; refill when 0
    """
    m: jnp.ndarray
    r: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    n: jnp.ndarray
    # JIT-aware draw counter — incremented by every ``next_uint64_jax`` call
    # (and host-side ``next_uint64`` / ``rn2`` / ``rnd``) so total uint64
    # consumption can be inspected after JIT-traced code runs.  int64 scalar.
    # See ``get_draw_count()`` / ``reset_draw_count()`` for the parallel
    # host-only counter (kept for legacy phase instrumentation).
    draws: jnp.ndarray = None  # type: ignore[assignment]

    def tree_flatten(self):
        return (self.m, self.r, self.a, self.b, self.c, self.n, self.draws), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)

    @classmethod
    def empty(cls) -> "Isaac64State":
        """Zero-valued sentinel state — used as the EnvState default.

        Not a valid RNG (every output would be 0); call :func:`init` with a
        real seed before draining.  Kept as the default so JIT-cached pytree
        shapes are stable whether or not ``NLE_BYTEPARITY`` is active.
        """
        return cls(
            m=jnp.zeros((ISAAC64_SZ,), dtype=jnp.uint64),
            r=jnp.zeros((ISAAC64_SZ,), dtype=jnp.uint64),
            a=jnp.zeros((), dtype=jnp.uint64),
            b=jnp.zeros((), dtype=jnp.uint64),
            c=jnp.zeros((), dtype=jnp.uint64),
            n=jnp.zeros((), dtype=jnp.int32),
            draws=jnp.zeros((), dtype=jnp.int64),
        )


def init(seed: int) -> Isaac64State:
    """Initialize ISAAC64 state from an integer seed (matches NLE seeding)."""
    m, r, a, b, c, n = init_py(int(seed))
    return Isaac64State(
        m=jnp.asarray(m, dtype=jnp.uint64),
        r=jnp.asarray(r, dtype=jnp.uint64),
        a=jnp.asarray(a, dtype=jnp.uint64),
        b=jnp.asarray(b, dtype=jnp.uint64),
        c=jnp.asarray(c, dtype=jnp.uint64),
        n=jnp.asarray(n, dtype=jnp.int32),
        draws=jnp.zeros((), dtype=jnp.int64),
    )


def reseed_random(state: "Isaac64State", has_strong_rngseed: bool = False) -> "Isaac64State":
    """Vendor-parity ``reseed_random(fn)`` (hacklib.c:906-914).

    Vendor pseudocode::

        void reseed_random(fn) {
            if (has_strong_rngseed)
                init_random(fn);              /* re-seeds from sys_random_seed */
        }

    Under the NLE byte-parity validator configuration the env is created
    with ``env.seed(seeds=(s, s), reseed=False)``
    (vendor/nle/nle/env/base.py:441), which sets ``has_strong_rngseed`` to
    ``False`` in ``init_random`` (vendor/nle/src/nle.c:412).  In that
    case ``reseed_random`` is a NO-OP and the ISAAC64 stream is preserved
    untouched across ``mklev()`` entry/exit and ``goto_level``.

    This helper mirrors that semantics: when ``has_strong_rngseed`` is
    False (the validator default), the input state is returned unchanged
    — preserving byte parity across the four call sites:

      * vendor/nle/src/mklev.c:996  ``reseed_random(rn2);``
      * vendor/nle/src/mklev.c:997  ``reseed_random(rn2_on_display_rng);``
      * vendor/nle/src/mklev.c:1034 ``reseed_random(rn2);``
      * vendor/nle/src/mklev.c:1035 ``reseed_random(rn2_on_display_rng);``
      * vendor/nle/src/do.c:1458    ``reseed_random(rn2_on_display_rng);``

    When ``has_strong_rngseed`` is True, vendor calls ``init_random``
    which fetches a fresh platform entropy word and ``set_random``-s the
    stream.  We currently do not model that path — it is non-deterministic
    in C and breaks byte parity by definition.  Passing ``True`` will
    raise to make accidental use loud.
    """
    if has_strong_rngseed:
        raise NotImplementedError(
            "reseed_random with has_strong_rngseed=True draws from "
            "sys_random_seed() which is non-deterministic; byte-parity "
            "validator runs with reseed=False so this path is unused."
        )
    return state


# ---------------------------------------------------------------------------
# Host-side JAX wrappers — eager scalar draws used in non-JIT setup paths.
#
# These pull host-side ints out of the pytree, call the reference Python
# helpers, and pack the result back into a fresh ``Isaac64State``.  They are
# NOT JIT-traceable yet (ISAAC64 refill is non-trivial inside jit; see the
# note at the bottom of this module).  Reset/dungeon-gen runs eagerly, so
# host-side wrappers are sufficient for the first byte-parity wiring.
# ---------------------------------------------------------------------------

def _state_to_py(state: Isaac64State):
    """Pull host-side Python lists/ints out of an ``Isaac64State`` pytree."""
    m = [int(v) for v in np.asarray(state.m)]
    r = [int(v) for v in np.asarray(state.r)]
    a = int(np.asarray(state.a))
    b = int(np.asarray(state.b))
    c = int(np.asarray(state.c))
    n = int(np.asarray(state.n))
    return m, r, a, b, c, n


def _state_from_py(py_state, draws: jnp.ndarray) -> Isaac64State:
    """Repack a host-side ``(m, r, a, b, c, n)`` tuple as an ``Isaac64State``.

    ``draws`` is carried separately (host-side roundtrip doesn't touch it)
    so the JIT-aware counter survives ``_state_to_py`` -> Python helper ->
    ``_state_from_py``.  Callers pass the incremented value explicitly.
    """
    m, r, a, b, c, n = py_state
    return Isaac64State(
        m=jnp.asarray(m, dtype=jnp.uint64),
        r=jnp.asarray(r, dtype=jnp.uint64),
        a=jnp.asarray(a, dtype=jnp.uint64),
        b=jnp.asarray(b, dtype=jnp.uint64),
        c=jnp.asarray(c, dtype=jnp.uint64),
        n=jnp.asarray(n, dtype=jnp.int32),
        draws=jnp.asarray(draws, dtype=jnp.int64),
    )


def rn2(state: Isaac64State, x: int) -> Tuple[Isaac64State, int]:
    """Host-side ``rn2(x)`` — returns ``(new_state, value_in_[0, x))``.

    Mirrors ``vendor/nethack/src/rnd.c::rn2`` under ``USE_ISAAC64``: a single
    ``isaac64_next_uint64() % x`` draw.  ``x`` must be a positive Python int.
    """
    py_state = _state_to_py(state)
    py_state, v = rn2_py(py_state, int(x))
    new_draws = jnp.asarray(int(np.asarray(state.draws)) + 1, dtype=jnp.int64)
    return _state_from_py(py_state, new_draws), int(v)


def next_uint64(state: Isaac64State) -> Tuple[Isaac64State, int]:
    """Host-side raw 64-bit draw — returns ``(new_state, uint64)``."""
    py_state = _state_to_py(state)
    py_state, v = next_uint64_py(py_state)
    new_draws = jnp.asarray(int(np.asarray(state.draws)) + 1, dtype=jnp.int64)
    return _state_from_py(py_state, new_draws), int(v)


def rne(state: Isaac64State, x: int, ulevel: int = 0) -> Tuple[Isaac64State, int]:
    """Host-side ``rne(x)`` — returns ``(new_state, value_in_[1, utmp])``.

    Mirrors vendor rnd.c:196-215.  ``utmp = 5`` for ulevel < 15.
    Draws rn2(x) up to ``utmp - 1`` times, stopping on first non-zero.

    Citations
    ---------
    vendor/nle/src/rnd.c:196-215   -- rne implementation
    vendor/nle/src/rnd.c:194       -- range comment
    """
    # rne_py drains the trace global counter — read draws delta from there.
    before = _DRAW_COUNT
    py_state = _state_to_py(state)
    py_state, v = rne_py(py_state, int(x), int(ulevel))
    delta = _DRAW_COUNT - before
    new_draws = jnp.asarray(int(np.asarray(state.draws)) + delta, dtype=jnp.int64)
    return _state_from_py(py_state, new_draws), int(v)


# ---------------------------------------------------------------------------
# JAX-traceable ISAAC64 — usable INSIDE @jax.jit.
#
# The host-side helpers above suffice for ``env.reset`` (which runs eagerly).
# For byte-exact NLE replay, the per-step game logic also needs to draw from
# the same ISAAC64 stream — and that runs inside ``_step_impl`` under JIT.
# This section ports ``isaac64_update`` + ``next_uint64`` to JAX primitives.
#
# Cite: vendor/nle/src/isaac64.c::isaac64_update (lines 46-97) and
#       isaac64_next_uint64 (lines 157-160).
# ---------------------------------------------------------------------------

# Constant uint64 mask — JAX's uint64 add/shift naturally wrap mod 2^64, but
# the bitwise NOT (~) produces signed-style results we need to re-mask.
_MASK64_U = jnp.uint64(MASK64)


def _isaac64_one_mix(carry, i):
    """One of the 4 unrolled mix steps inside isaac64_update.

    Vendor unrolls the 256-iter loop into stanzas of 4 mixes; each stanza
    uses a different ``a`` shuffle:
        mix 0: a = ~(a ^ (a << 21)) + m[i + half]
        mix 1: a =  (a ^ (a >>  5)) + m[i + half]
        mix 2: a =  (a ^ (a << 12)) + m[i + half]
        mix 3: a =  (a ^ (a >> 33)) + m[i + half]

    For the second half (i >= half=128), ``+ m[i + half]`` becomes
    ``+ m[i - half]`` because m is treated cyclically with stride ``half``.
    JAX-traceable: we encode the mix-kind as ``i & 3`` and the half-pair
    offset as ``i ^ half`` (since (i + half) % SZ == (i ^ half) when SZ
    is a power of 2 and stride == half).
    """
    m, r, a, b, c = carry
    half = jnp.uint32(ISAAC64_SZ // 2)
    mask = jnp.uint64(ISAAC64_SZ - 1)
    mix_kind = i & jnp.uint32(3)

    x = m[i]
    pair_idx = i ^ half  # (i + half) for i < half; (i - half) for i >= half
    paired = m[pair_idx]

    # Pick the right a-shuffle based on mix_kind.
    a_xor_21 = a ^ jnp.left_shift(a, jnp.uint64(21))
    a_kind0 = (~a_xor_21) & _MASK64_U
    a_kind1 = a ^ jnp.right_shift(a, jnp.uint64(5))
    a_kind2 = a ^ (jnp.left_shift(a, jnp.uint64(12)) & _MASK64_U)
    a_kind3 = a ^ jnp.right_shift(a, jnp.uint64(33))
    a_shuffled = jnp.where(
        mix_kind == jnp.uint32(0), a_kind0,
        jnp.where(
            mix_kind == jnp.uint32(1), a_kind1,
            jnp.where(mix_kind == jnp.uint32(2), a_kind2, a_kind3),
        ),
    )
    a_new = (a_shuffled + paired) & _MASK64_U

    # lower(x) = (x & ((SZ-1) << 3)) >> 3 — picks bits 3..10 of x.
    lower_idx = jnp.right_shift(
        jnp.bitwise_and(x, mask << jnp.uint64(3)), jnp.uint64(3)
    ).astype(jnp.uint32)
    y = (m[lower_idx] + a_new + b) & _MASK64_U
    m_new = m.at[i].set(y)

    # upper(y) = (y >> (SZ_LOG + 3)) & (SZ - 1) — picks bits SZ_LOG+3..SZ_LOG+11 of y.
    upper_idx = jnp.bitwise_and(
        jnp.right_shift(y, jnp.uint64(ISAAC64_SZ_LOG + 3)), mask
    ).astype(jnp.uint32)
    b_new = (m_new[upper_idx] + x) & _MASK64_U
    r_new = r.at[i].set(b_new)

    return (m_new, r_new, a_new, b_new, c), None


def _isaac64_refill_jax(rng: "Isaac64State") -> "Isaac64State":
    """JAX-traceable refill — one full 256-iter pass through m/r.

    Equivalent to a single call to ``isaac64_update``; produces 256 new
    output words into ``r[]`` and resets the slot pointer ``n`` to 256.
    """
    c_new = (rng.c + jnp.uint64(1)) & _MASK64_U
    b_new = (rng.b + c_new) & _MASK64_U
    carry = (rng.m, rng.r, rng.a, b_new, c_new)
    (m_out, r_out, a_out, b_out, c_out), _ = jax.lax.scan(
        _isaac64_one_mix, carry, jnp.arange(ISAAC64_SZ, dtype=jnp.uint32)
    )
    return Isaac64State(
        m=m_out, r=r_out, a=a_out, b=b_out, c=c_out,
        n=jnp.int32(ISAAC64_SZ), draws=rng.draws,
    )


def next_uint64_jax(rng: "Isaac64State") -> Tuple["Isaac64State", jax.Array]:
    """JAX-traceable counterpart to ``next_uint64_py``.

    Returns ``(new_rng, uint64)``.  Refills automatically when ``n == 0``.
    Safe to call inside ``@jax.jit``.
    """
    needs_refill = rng.n <= jnp.int32(0)
    refilled = jax.lax.cond(
        needs_refill,
        _isaac64_refill_jax,
        lambda r: r,
        rng,
    )
    new_n = refilled.n - jnp.int32(1)
    val = refilled.r[new_n.astype(jnp.uint32)]
    return Isaac64State(
        m=refilled.m, r=refilled.r,
        a=refilled.a, b=refilled.b, c=refilled.c,
        n=new_n,
        draws=refilled.draws + jnp.int64(1),
    ), val


def rn2_jax(rng: "Isaac64State", x) -> Tuple["Isaac64State", jax.Array]:
    """JAX-traceable ``rn2(x)`` — returns ``(new_rng, int32 in [0, x))``.

    Mirrors vendor ``rnd.c::rn2`` under ``USE_ISAAC64``: a single
    ``isaac64_next_uint64() % x`` draw.  ``x`` must be a positive scalar.
    """
    new_rng, v = next_uint64_jax(rng)
    result = (v % jnp.uint64(x)).astype(jnp.int32)
    if _jit_trace_enabled():
        jax.debug.callback(
            lambda mod, res: _emit_op_callback(b"rn2", mod, res),
            jnp.asarray(x, dtype=jnp.int64), result,
        )
    return new_rng, result


def rnd_jax(rng: "Isaac64State", x) -> Tuple["Isaac64State", jax.Array]:
    """JAX-traceable ``rnd(x)`` — returns ``(new_rng, int32 in [1, x])``."""
    # Inline the draw so the trace records "rnd" not "rn2" (mirror vendor rnd.c).
    new_rng, v = next_uint64_jax(rng)
    result = (v % jnp.uint64(x)).astype(jnp.int32) + jnp.int32(1)
    if _jit_trace_enabled():
        jax.debug.callback(
            lambda mod, res: _emit_op_callback(b"rnd", mod, res),
            jnp.asarray(x, dtype=jnp.int64), result,
        )
    return new_rng, result


def rn1_jax(rng: "Isaac64State", x, base) -> Tuple["Isaac64State", jax.Array]:
    """JAX-traceable ``rn1(x, base)`` — ``base + rn2(x)`` (int32 result).

    Inlines its own draw + trace so the trace line records the underlying
    ``rn2(x)`` modulus/result (matching vendor RND macro expansion in
    ``vendor/nethack/src/rnd.c::rn1``).
    """
    new_rng, v = next_uint64_jax(rng)
    result = (v % jnp.uint64(x)).astype(jnp.int32)
    if _jit_trace_enabled():
        jax.debug.callback(
            lambda mod, res: _emit_op_callback(b"rn2", mod, res),
            jnp.asarray(x, dtype=jnp.int64), result,
        )
    return new_rng, result + jnp.int32(base)


def rne_jax(rng: "Isaac64State", x, ulevel: int = 0) -> Tuple["Isaac64State", jax.Array]:
    """JAX-traceable ``rne(x)`` — vendor rnd.c:196-215.

    ``utmp = (ulevel < 15) ? 5 : ulevel / 3``; tmp starts at 1 and
    increments while ``tmp < utmp && rn2(x) == 0``.

    Uses ``lax.while_loop`` so it is JIT-compatible.  The loop body consumes
    at most ``utmp - 1`` draws (4 for ulevel 0-14, fewer for higher levels).

    Citations
    ---------
    vendor/nle/src/rnd.c:196-215   -- rne implementation
    vendor/nle/src/rnd.c:194       -- range comment ``1 <= rne(x) <= max(...)``
    """
    utmp = jnp.int32(5 if ulevel < 15 else ulevel // 3)

    def cond(carry):
        _rng, tmp = carry
        return tmp < utmp

    def body(carry):
        _rng, tmp = carry
        new_rng, v = rn2_jax(_rng, x)
        # stop incrementing when rn2(x) != 0 by jumping to utmp (terminates cond)
        new_tmp = jax.lax.cond(v == jnp.int32(0), lambda: tmp + jnp.int32(1), lambda: utmp)
        return new_rng, new_tmp

    rng_out, tmp_out = jax.lax.while_loop(cond, body, (rng, jnp.int32(1)))
    return rng_out, tmp_out


def isaac_weighted_choice(rng: "Isaac64State", weights: jax.Array) -> Tuple["Isaac64State", jax.Array]:
    """JIT-pure weighted choice consuming the ISAAC64 stream.

    weights: int/float array, summed to compute total.
    Returns (new_rng, chosen_index).  Mirrors vendor C
    ``rndmonst_inner``: draw rn2(total), find first cumsum bucket > draw.
    """
    cdf = jnp.cumsum(weights.astype(jnp.uint64))
    total = cdf[-1]
    new_rng, draw = next_uint64_jax(rng)
    sampled = (draw % total).astype(jnp.uint64)
    # argmax(cdf > sampled) — first bucket whose cumulative weight exceeds the draw.
    idx = jnp.argmax(cdf > sampled).astype(jnp.int32)
    if _jit_trace_enabled():
        # Trace as rn2(total) — vendor rndmonst_inner.
        jax.debug.callback(
            lambda mod, res: _emit_op_callback(b"rn2", mod, res),
            total.astype(jnp.int64), sampled.astype(jnp.int32),
        )
    return new_rng, idx


def randint_jax(rng: "Isaac64State", shape, minval, maxval) -> Tuple["Isaac64State", jax.Array]:
    """JAX-traceable drop-in for ``jax.random.randint(key, shape, minval, maxval)``.

    Returns ``(new_rng, value)`` where value is in [minval, maxval).

    The output is byte-exact with vendor C ``minval + isaac64_next_uint64() %
    (maxval - minval)`` for each scalar slot.  Unlike ``jax.random.randint``,
    this consumes from the ISAAC64 stream — so dungeon-gen call sites that
    swap in ``randint_jax(state.vendor_rng, ...)`` will produce trajectories
    that match vendor NLE on the same seed.

    Currently supports scalar output only (shape=()); multi-dim sampling
    requires N sequential draws and should use a scan loop.
    """
    range_size = jnp.uint64(maxval) - jnp.uint64(minval)
    new_rng, v = next_uint64_jax(rng)
    sampled = (v % range_size).astype(jnp.int32) + jnp.int32(minval)
    if _jit_trace_enabled():
        # Trace as rn2 of the (maxval-minval) range — closest vendor analog.
        jax.debug.callback(
            lambda mod, res: _emit_op_callback(b"rn2", mod, res),
            jnp.asarray(range_size, dtype=jnp.int64),
            sampled - jnp.int32(minval),
        )
    return new_rng, sampled
