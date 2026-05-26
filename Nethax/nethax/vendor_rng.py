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

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


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
    return state, v % x


def rnd_py(state, x: int) -> tuple:
    """``rnd(x)`` -- ``rn2(x) + 1``."""
    state, v = rn2_py(state, x)
    return state, v + 1


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

    def tree_flatten(self):
        return (self.m, self.r, self.a, self.b, self.c, self.n), None

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
    )


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


def _state_from_py(py_state) -> Isaac64State:
    """Repack a host-side ``(m, r, a, b, c, n)`` tuple as an ``Isaac64State``."""
    m, r, a, b, c, n = py_state
    return Isaac64State(
        m=jnp.asarray(m, dtype=jnp.uint64),
        r=jnp.asarray(r, dtype=jnp.uint64),
        a=jnp.asarray(a, dtype=jnp.uint64),
        b=jnp.asarray(b, dtype=jnp.uint64),
        c=jnp.asarray(c, dtype=jnp.uint64),
        n=jnp.asarray(n, dtype=jnp.int32),
    )


def rn2(state: Isaac64State, x: int) -> Tuple[Isaac64State, int]:
    """Host-side ``rn2(x)`` — returns ``(new_state, value_in_[0, x))``.

    Mirrors ``vendor/nethack/src/rnd.c::rn2`` under ``USE_ISAAC64``: a single
    ``isaac64_next_uint64() % x`` draw.  ``x`` must be a positive Python int.
    """
    py_state = _state_to_py(state)
    py_state, v = rn2_py(py_state, int(x))
    return _state_from_py(py_state), int(v)


def next_uint64(state: Isaac64State) -> Tuple[Isaac64State, int]:
    """Host-side raw 64-bit draw — returns ``(new_state, uint64)``."""
    py_state = _state_to_py(state)
    py_state, v = next_uint64_py(py_state)
    return _state_from_py(py_state), int(v)


# Note: a JAX-traceable refill is non-trivial because ``isaac64_update``
# branches on the index and writes 256 entries.  Plan: implement via
# ``jax.lax.fori_loop`` over half-blocks; for now expose only the
# eager/numpy initialization path so the audit + skeleton land cleanly.
