"""JAX PRNG utilities — dice rolls, key splitting, weighted sampling.

Canonical sources:
  vendor/nethack/src/rnd.c      — rn2(), rnd(), d(), rnl(), rne(); ISAAC64
                                   seeding via set_random() (rnd.c lines 31-120)
  vendor/nethack/src/isaac64.c  — ISAAC-64 CSPRNG implementation used by
                                   NetHack 3.7 for both core and display RNGs

JAX PRNG conventions used in Nethax
-------------------------------------
JAX uses a functional, explicit-key PRNG model (Threefry/Counter-based) that
is stateless and fully reproducible given the same key.  Key rules:

1. NEVER reuse a key.  After using a key, split it first:
       rng, subkey = jax.random.split(rng)
2. Pass ``rng`` through every function that needs randomness; store it in
   ``EnvState.state_rng``.
3. For multiple independent draws in one call, use ``split_n``:
       keys = split_n(rng, 4)   # shape (4, 2)
4. The JAX key shape is (2,) (uint32 pair); always treat it as opaque.
5. We do NOT use ISAAC-64 at runtime — NetHack's ISAAC is cited for parity
   only.  JAX's Threefry provides equivalent statistical quality with the
   added benefit of hardware-accelerated batch generation on GPU/TPU.

Status: Wave 2 — real JAX random implementations, JIT-compatible.
"""
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Key splitting
# ---------------------------------------------------------------------------

def split_n(rng: jax.Array, n: int) -> jax.Array:
    """Split a JAX PRNG key into n independent subkeys.

    Parameters
    ----------
    rng : JAX key of shape (2,).
    n   : number of subkeys to produce.

    Returns
    -------
    keys : array of shape (n, 2) — each row is an independent subkey.

    Example
    -------
    >>> keys = split_n(rng, 3)
    >>> roll_key, move_key, ai_key = keys[0], keys[1], keys[2]
    """
    return jax.random.split(rng, n)


# ---------------------------------------------------------------------------
# Dice rolls
# ---------------------------------------------------------------------------

def dice_roll(rng: jax.Array, n: int, sides: int) -> jnp.int32:
    """Sum of n rolls of a die with `sides` faces.

    Matches NetHack's ``d(n, sides)`` function in rnd.c.

    Parameters
    ----------
    rng   : JAX PRNG key (consumed; do not reuse).
    n     : number of dice (Python int, static).
    sides : faces per die (Python int, static).

    Returns
    -------
    Total roll as jnp.int32 in [n, n*sides].
    """
    rolls = jax.random.randint(rng, (n,), minval=1, maxval=sides + 1, dtype=jnp.int32)
    return jnp.sum(rolls).astype(jnp.int32)


def rnd(rng: jax.Array, n: int) -> jnp.int32:
    """Single die roll in [1, n].

    Matches NetHack's ``rnd(n)`` in rnd.c.

    Parameters
    ----------
    rng : JAX PRNG key (consumed; do not reuse).
    n   : number of faces (Python int, static).

    Returns
    -------
    jnp.int32 in [1, n].
    """
    return jax.random.randint(rng, (), minval=1, maxval=n + 1, dtype=jnp.int32)


def rn2(rng: jax.Array, n: int) -> jnp.int32:
    """Uniform integer in [0, n).

    Matches NetHack's ``rn2(n)`` in rnd.c.

    Parameters
    ----------
    rng : JAX PRNG key (consumed; do not reuse).
    n   : exclusive upper bound (Python int, static).

    Returns
    -------
    jnp.int32 in [0, n).
    """
    return jax.random.randint(rng, (), minval=0, maxval=n, dtype=jnp.int32)


def rn1(rng: jax.Array, n: int, x: int) -> jnp.int32:
    """Offset uniform integer in [x, x+n-1].

    Matches NetHack's ``rn1(n, x)`` (defined in rnd.h as ``x + rn2(n)``).

    Parameters
    ----------
    rng : JAX PRNG key (consumed; do not reuse).
    n   : range size (Python int, static).
    x   : offset (Python int, static).

    Returns
    -------
    jnp.int32 in [x, x+n-1].
    """
    return (rn2(rng, n) + jnp.int32(x)).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Extended NetHack RNG helpers
# ---------------------------------------------------------------------------

def rne(rng: jax.Array, x: int, cap: int = 9) -> jnp.int32:
    """Enchantment roll — geometric distribution starting at 1.

    Vendor formula (rnd.c::rne, line 191):
        tmp = 1;
        while (tmp < cap && !rn2(x)) tmp++;
        return tmp;

    Vendor invariant: at most ``cap - 1`` rolls happen because the loop
    exits the moment ``tmp == cap``.  So the result is in ``[1, cap]``
    inclusive.

    JAX implementation: simulate the geometric loop via a bounded
    vmap'd batch of ``cap`` rolls, then clamp the running count at
    ``cap``.  Clamping (rather than running only ``cap - 1`` rolls) keeps
    the trace shape constant for varying caps and exactly reproduces the
    vendor distribution.

    Returns
    -------
    jnp.int32 in [1, cap] — typically used for weapon/wand enchant levels.
    """
    keys = jax.random.split(rng, cap)
    rolls = jax.vmap(lambda k: jax.random.randint(k, (), 0, x))(keys) == 0
    consecutive = jnp.cumprod(rolls.astype(jnp.int32))
    raw = jnp.int32(1) + jnp.sum(consecutive)
    return jnp.minimum(raw, jnp.int32(cap)).astype(jnp.int32)


def rnl(rng: jax.Array, x: int, luck: int = 0) -> jnp.int32:
    """Luck-adjusted uniform in [0, x).

    Vendor formula (rnd.c::rnl):
        i = rn2(x)
        if (Luck && rn2(50 - 2*Luck)):
            i += Luck * (x / 50 - 1)
            clamp i to [0, x-1]
        return i

    For Wave 6 simplification, accept luck as a Python int (-13..+13).

    Returns
    -------
    jnp.int32 in [0, x).
    """
    rng_a, rng_b = jax.random.split(rng, 2)
    i = rn2(rng_a, x)
    # Adjustment: if luck != 0 and a side-roll fires, shift i by luck.
    if luck == 0:
        return i
    adj_chance_denom = max(1, 50 - 2 * int(luck))
    side = rn2(rng_b, adj_chance_denom)
    delta = jnp.int32(int(luck) * max(1, (x // 50) - 1))
    i_adjusted = jnp.where(side != 0, i, i + delta)
    return jnp.clip(i_adjusted, 0, x - 1).astype(jnp.int32)


def rnz(rng: jax.Array, i: int) -> jnp.int32:
    """Time-scaling roll — vendor formula for timeout durations.

    Vendor formula (rnd.c::rnz):
        x = i
        tmp = 1000 + rn2(1000)
        tmp *= rne(4)
        if (rn2(2)):
            x = (x * tmp) // 1000
        else:
            x = (x * 1000) // tmp
        return x

    Result is roughly in [i/3, i*3], skewed by rne(4) tail.

    Returns
    -------
    jnp.int32 scaling of i — used for status-effect timeouts.
    """
    keys = jax.random.split(rng, 3)
    tmp = jnp.int32(1000) + rn2(keys[0], 1000)
    tmp = tmp * rne(keys[1], 4)
    side = rn2(keys[2], 2)
    x = jnp.int32(i)
    grow = (x * tmp) // jnp.int32(1000)
    shrink = (x * jnp.int32(1000)) // tmp
    return jnp.where(side != 0, grow, shrink).astype(jnp.int32)


def rnf(rng: jax.Array, num: int, den: int) -> jnp.bool_:
    """Fractional probability check — returns True with probability num/den.

    Vendor formula (rnd.c::rnf):
        return rn2(den) < num;

    Returns
    -------
    jnp.bool_ — True with probability num/den.
    """
    return (rn2(rng, den) < jnp.int32(num))


# ---------------------------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------------------------

def weighted_choice(rng: jax.Array, weights: jnp.ndarray) -> jnp.int32:
    """Sample one index proportional to weights.

    Mirrors usage of rn2() with scaled weight tables throughout NetHack
    (e.g. monster generation in monst.c, item selection in mkobj.c).

    Parameters
    ----------
    rng     : JAX PRNG key (consumed; do not reuse).
    weights : 1-D non-negative array of unnormalised probabilities.
              Shape must be static (known at trace time).

    Returns
    -------
    Sampled index as jnp.int32.
    """
    p = weights / jnp.sum(weights)
    return jax.random.choice(rng, len(weights), p=p).astype(jnp.int32)
