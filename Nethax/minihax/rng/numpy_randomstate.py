"""Bit-exact pure-Python port of numpy's legacy ``RandomState`` (MT19937).

Target: numpy 2.4.4 legacy ``numpy.random.RandomState``. numpy's legacy
RandomState algorithm has been byte-stable across numpy versions (the modern
``Generator`` API changed, but ``RandomState`` is frozen for reproducibility).

This module exists so Minihax can reproduce, byte-for-byte, the exact draw
sequence that MiniGrid / MiniHack's MultiRoom (and other MiniGrid-derived)
level generators consume from ``self.np_random`` -- a numpy legacy RandomState
seeded the gym-0.21 way. It ports:

  * MT19937 core: ``init_genrand`` (scalar seed) and ``init_by_array`` (array
    seed), plus ``genrand_int32`` with the tempering transform and 624-word
    reload.
  * ``RandomState.seed`` scalar-int (-> init_genrand) and int-array
    (-> init_by_array) forms, matching numpy's ``_legacy_seeding`` routing.
  * ``randint(low, high)`` -- numpy legacy bounded-integer masked rejection.
    Default dtype is ``np.int_`` (int64 here) but numpy's
    ``random_bounded_uint64_fill`` uses a 32-bit draw when the span fits in
    32 bits (the common MiniGrid case), else a 64-bit (two-word) draw.
  * ``random_sample`` / ``uniform`` (53-bit double from two words).
  * ``choice`` (replace=True scalar index path) and ``shuffle`` (legacy
    Fisher-Yates using ``random_interval``).

The gym-0.21 seeding chain helpers (``create_seed`` / ``hash_seed`` /
``int_list_from_bigint``) are reproduced here so a MiniGrid layout seeded via
``seed_gym(env_seed)`` matches the canonical MiniHack reference authored
against gym 0.21.

Pure-Python int math -- byte-parity level-gen runs eager, this does not need to
be JAX-jittable.
"""

import hashlib
import struct

__all__ = [
    "RandomState",
    "create_seed",
    "hash_seed",
    "int_list_from_bigint",
    "bigint_from_bytes",
]

# --- MT19937 constants ---
_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF


# ======================================================================
# gym-0.21 seeding chain (replica of gym.utils.seeding, matches the shim)
# ======================================================================
def bigint_from_bytes(bt: bytes) -> int:
    sizeof_int = 4
    padding = sizeof_int - len(bt) % sizeof_int
    bt = bt + b"\0" * padding
    int_count = int(len(bt) / sizeof_int)
    unpacked = struct.unpack(f"{int_count}I", bt)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def int_list_from_bigint(bigint: int):
    if bigint < 0:
        raise ValueError("Seed must be non-negative")
    if bigint == 0:
        return [0]
    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints


def hash_seed(seed, max_bytes: int = 8) -> int:
    h = hashlib.sha512(str(seed).encode("utf8")).digest()
    return bigint_from_bytes(h[:max_bytes])


def create_seed(a, max_bytes: int = 8) -> int:
    if a is None:
        raise ValueError("create_seed(None) is non-deterministic; pass an int")
    elif isinstance(a, str):
        bt = a.encode("utf8") + hashlib.sha512(a.encode("utf8")).digest()
        a = bigint_from_bytes(bt[:max_bytes])
    elif isinstance(a, int):
        a = int(a % 2 ** (8 * max_bytes))
    else:
        raise ValueError(f"Invalid seed type {type(a)}")
    return a


# ======================================================================
# MT19937 + legacy RandomState
# ======================================================================
class RandomState:
    """Bit-exact port of ``numpy.random.RandomState`` (legacy MT19937)."""

    def __init__(self, seed=None):
        self._mt = [0] * _N
        self._mti = _N + 1  # uninitialised sentinel (matches C: mti==N+1)
        if seed is not None:
            self.seed(seed)
        else:
            # numpy default seeds from OS entropy; for determinism require a seed.
            self.seed(0)

    # ---------------- MT19937 core ----------------
    def _init_genrand(self, s: int) -> None:
        mt = self._mt
        mt[0] = s & _U32
        for i in range(1, _N):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & _U32
        self._mti = _N

    def _init_by_array(self, init_key) -> None:
        self._init_genrand(19650218)
        mt = self._mt
        key_length = len(init_key)
        i = 1
        j = 0
        k = _N if _N > key_length else key_length
        while k:
            mt[i] = (
                (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525))
                + init_key[j]
                + j
            ) & _U32
            i += 1
            j += 1
            if i >= _N:
                mt[0] = mt[_N - 1]
                i = 1
            if j >= key_length:
                j = 0
            k -= 1
        k = _N - 1
        while k:
            mt[i] = (
                (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i
            ) & _U32
            i += 1
            if i >= _N:
                mt[0] = mt[_N - 1]
                i = 1
            k -= 1
        mt[0] = 0x80000000
        self._mti = _N

    def genrand_int32(self) -> int:
        """One raw 32-bit MT19937 output (with tempering)."""
        mt = self._mt
        if self._mti >= _N:
            if self._mti == _N + 1:
                # never seeded -> numpy uses seed(5489); we require explicit seed
                self._init_genrand(5489)
            mag01 = (0, _MATRIX_A)
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ mag01[y & 1]
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ mag01[y & 1]
            y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ mag01[y & 1]
            self._mti = 0

        y = mt[self._mti]
        self._mti += 1
        # tempering
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & _U32

    def _next_uint64(self) -> int:
        hi = self.genrand_int32()
        lo = self.genrand_int32()
        return ((hi << 32) | lo) & _U64

    # ---------------- seeding ----------------
    def seed(self, seed) -> None:
        """Match numpy ``RandomState.seed`` legacy routing exactly.

        numpy's ``_legacy_seeding``:
          * a Python int (or numpy integer scalar) -> ``init_genrand``.
          * an ndarray is ``.squeeze()``-ed first; if it collapses to 0-d it
            takes the scalar ``init_genrand`` path (so ``np.array([0])`` seeds
            like the scalar ``0``!), otherwise ``init_by_array``.
          * a Python list/tuple (no ``squeeze``, not index-able) ->
            ``init_by_array``. This is the gym-0.21 / MiniGrid path.
        """
        # ndarray: replicate numpy's squeeze-then-try-scalar behaviour.
        if hasattr(seed, "squeeze") and hasattr(seed, "ndim"):
            sq = seed.squeeze()
            if sq.ndim == 0:
                self._scalar_seed(int(sq))
                return
            self._init_by_array([int(x) & _U32 for x in sq.tolist()])
            return
        if isinstance(seed, (list, tuple)):
            self._init_by_array([int(x) & _U32 for x in seed])
            return
        # scalar int (or anything index-able as an int)
        self._scalar_seed(int(seed))

    def _scalar_seed(self, idx: int) -> None:
        if idx < 0 or idx > _U32:
            raise ValueError("Seed must be between 0 and 2**32 - 1")
        self._init_genrand(idx)

    def seed_gym(self, env_seed) -> None:
        """Seed the gym-0.21 way used by MiniGrid/MiniHack.

        Equivalent to gym 0.21::
            seed = create_seed(env_seed)
            rng  = RandomState()
            rng.seed(int_list_from_bigint(hash_seed(seed)))
        """
        s = create_seed(env_seed)
        self.seed(int_list_from_bigint(hash_seed(s)))

    # ---------------- bounded integers ----------------
    @staticmethod
    def _gen_mask(rng_incl: int) -> int:
        mask = rng_incl
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        mask |= mask >> 32
        return mask & _U64

    def _bounded(self, rng_incl: int) -> int:
        """Return an int uniformly in [0, rng_incl] via masked rejection.

        Mirrors numpy's ``random_bounded_uint64_fill`` / ``random_interval``:
        a 32-bit draw is used when the span fits in 32 bits, else a 64-bit
        (two-word) draw. rng_incl == 0 draws no words.
        """
        if rng_incl == 0:
            return 0
        mask = self._gen_mask(rng_incl)
        if rng_incl <= _U32:
            while True:
                val = self.genrand_int32() & mask
                if val <= rng_incl:
                    return val
        else:
            while True:
                val = self._next_uint64() & mask
                if val <= rng_incl:
                    return val

    def randint(self, low, high=None):
        """numpy legacy ``randint(low, high)`` -> int in [low, high).

        Scalar draw only (size=None). Default dtype int64 semantics, but the
        span-fits-in-32-bits optimization is honoured.
        """
        if high is None:
            low, high = 0, low
        low = int(low)
        high = int(high)
        if low >= high:
            raise ValueError("low >= high")
        rng_incl = high - low - 1  # inclusive max offset
        return low + self._bounded(rng_incl)

    # ---------------- floats ----------------
    def random_sample(self):
        """53-bit double in [0, 1) -- numpy ``rk_double`` (two words)."""
        a = self.genrand_int32() >> 5  # 27 bits
        b = self.genrand_int32() >> 6  # 26 bits
        return (a * 67108864.0 + b) / 9007199254740992.0

    # numpy alias
    def random(self):
        return self.random_sample()

    def uniform(self, low=0.0, high=1.0):
        # numpy legacy: ``low + range * next_double`` compiled with fp
        # contraction -> a single fused multiply-add (correctly-rounded).
        # Python has no math.fma before 3.13, so compute the exact product+sum
        # and round once via Fraction (== IEEE fma nearest-even).
        from fractions import Fraction

        x = self.random_sample()
        rng = high - low  # float subtraction (matches numpy's Cython)
        return float(Fraction(rng) * Fraction(x) + Fraction(low))

    # ---------------- choice / shuffle ----------------
    def choice(self, a):
        """``choice(a)`` scalar, replace=True, no p.

        numpy's replace=True/no-p/size=None path draws a single index via
        ``randint(0, pop_size)`` and returns ``a[idx]``. If ``a`` is an int it
        is treated as ``arange(a)`` (numpy semantics), returning the index.
        """
        if isinstance(a, int):
            pop_size = a
            idx = self.randint(0, pop_size)
            return idx
        seq = list(a)
        pop_size = len(seq)
        if pop_size == 0:
            raise ValueError("a must be non-empty")
        idx = self.randint(0, pop_size)
        return seq[idx]

    def shuffle(self, x) -> None:
        """In-place legacy Fisher-Yates (matches numpy ``RandomState.shuffle``).

        for i in reversed(range(1, n)): j = random_interval(i); swap x[i], x[j]
        Mutates ``x`` (a list or numpy array) in place, like numpy.
        """
        n = len(x)
        for i in reversed(range(1, n)):
            j = self._bounded(i)  # random_interval(i) -> [0, i]
            x[i], x[j] = x[j], x[i]

    # ---------------- introspection (for tests) ----------------
    def get_key_state(self):
        """Return (mt_copy, pos) comparable to numpy get_state()[1], [2]."""
        pos = self._mti if self._mti <= _N else _N
        return list(self._mt), pos
