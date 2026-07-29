"""Shim: make gym_minigrid.MiniGridEnv.seed() use a numpy MT19937 RandomState
seeded exactly the way OLD gym 0.21 gym/utils/seeding.np_random did, so that the
generated MiniGrid layout byte-matches the canonical MiniHack reference (which was
authored against gym 0.21).

Old gym 0.21 np_random(seed):
    seed = create_seed(seed)                       # int passthrough
    rng  = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed

These helpers are reimplemented here (they also still exist, deprecated, in
gym 0.23's gym/utils/seeding.py) so the shim does not depend on whether
sys.modules['gym'] points at real gym or gymnasium.

Usage (before constructing any vendor MiniGrid env):
    import _minigrid_rs_shim; _minigrid_rs_shim.apply()
Do NOT modify anything under vendor/.
"""
import hashlib
import struct
import numpy as np


def _bigint_from_bytes(bt: bytes) -> int:
    sizeof_int = 4
    padding = sizeof_int - len(bt) % sizeof_int
    bt = bt + b"\0" * padding
    int_count = int(len(bt) / sizeof_int)
    unpacked = struct.unpack(f"{int_count}I", bt)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint: int):
    if bigint < 0:
        raise ValueError("Seed must be non-negative")
    if bigint == 0:
        return [0]
    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints


def _hash_seed(seed, max_bytes: int = 8) -> int:
    h = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(h[:max_bytes])


def _create_seed(a, max_bytes: int = 8) -> int:
    if a is None:
        a = _bigint_from_bytes(np.random.bytes(max_bytes))
    elif isinstance(a, str):
        bt = a.encode("utf8") + hashlib.sha512(a.encode("utf8")).digest()
        a = _bigint_from_bytes(bt[:max_bytes])
    elif isinstance(a, int):
        a = int(a % 2 ** (8 * max_bytes))
    else:
        raise ValueError(f"Invalid seed type {type(a)}")
    return a


def old_np_random(seed=None):
    """Byte-for-byte replica of gym 0.21 gym.utils.seeding.np_random."""
    seed = _create_seed(seed)
    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(_hash_seed(seed)))
    return rng, seed


def _patched_seed(self, seed=1337):
    self.np_random, _ = old_np_random(seed)
    return [seed]


def _patched_reset(self, *, seed=None, options=None):
    """gymnasium-compatible reset that preserves old gym_minigrid semantics.

    gym.make in vendor MiniGridHack is aliased to gymnasium.make, whose
    OrderEnforcing/PassiveEnvChecker/TimeLimit wrappers call reset(seed=..,
    options=..) and expect a (obs, info) tuple. Old MiniGridEnv.reset() took no
    args and returned obs only. We honour a supplied seed (via the RandomState
    shim) but, matching original behaviour, do NOT reseed when seed is None so
    the layout stays whatever __init__'s default-seed stream produced.
    """
    if seed is not None:
        self.seed(seed)
    self.agent_pos = None
    self.agent_dir = None
    self._gen_grid(self.width, self.height)
    assert self.agent_pos is not None
    assert self.agent_dir is not None
    start_cell = self.grid.get(*self.agent_pos)
    assert start_cell is None or start_cell.can_overlap()
    self.carrying = None
    self.step_count = 0
    obs = self.gen_obs()
    return obs, {}


class _MakeProxy:
    """Proxy standing in for the ``gym`` module inside minihack.envs.minigrid so
    that gym.make(env_name) disables gymnasium's PassiveEnvChecker (old
    gym_minigrid returns obs dicts with extra keys the checker rejects)."""

    def __init__(self, real):
        self._real = real

    def make(self, *args, **kwargs):
        kwargs.setdefault("disable_env_checker", True)
        env = self._real.make(*args, **kwargs)
        # Return the bare MiniGridEnv: gymnasium's OrderEnforcing/TimeLimit
        # wrappers do not passthrough attributes (.grid/.width/.agent_pos) that
        # MiniHack.get_env_map reads, and old gym_minigrid was used unwrapped.
        return env.unwrapped

    def __getattr__(self, name):
        return getattr(self._real, name)


def apply():
    import gym_minigrid.minigrid as mg
    mg.MiniGridEnv.seed = _patched_seed
    mg.MiniGridEnv.reset = _patched_reset
    # Disable gymnasium env-checker for the vendor MiniGrid make(), if the
    # minihack vendor module is already imported.
    try:
        import minihack.envs.minigrid as mhmg
        if not isinstance(mhmg.gym, _MakeProxy):
            mhmg.gym = _MakeProxy(mhmg.gym)
    except Exception:
        pass
    return mg.MiniGridEnv
