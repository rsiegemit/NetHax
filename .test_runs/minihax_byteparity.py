"""Minihax vs vendor MiniHack byte-parity trace harness (Phase 1).

Mirrors ``.test_runs/multiseed_byteparity.py`` (Nethax<->NLE) but for the
MiniHack family.  Given an ``env_id`` and a seed ``N``, builds the env on
both sides, dumps ``glyphs`` / ``chars`` / agent ``(y,x)`` / inventory and
byte-byte diffs them.

Usage
-----
    .venv/bin/python .test_runs/minihax_byteparity.py \
        --env MiniHack-Room-5x5-v0 --seed 0
    .venv/bin/python .test_runs/minihax_byteparity.py --all-rooms --seed 0

Exit: 0 on PASS, 1 on FAIL.

RNG quirks (audit findings)
---------------------------
* Vendor MiniHack ``base.py`` line 370-376: if ``sample_seed=True`` (default)
  and ``_level_seeds is not None``, ``reset()`` does ``random.choice(_level_seeds)``,
  burning a Python ``random`` draw and IGNORING the seed we pass.  We construct
  envs without passing ``seeds=`` (so ``_level_seeds`` stays ``None`` and that
  branch is skipped).
* Vendor NLE ``env.reset()`` does NOT accept a ``seed=`` kwarg; the NLE convention
  is ``env.seed(core=N, disp=N, reseed=False)`` followed by ``env.reset()``.  We
  use that explicitly to control ISAAC64 seeding, matching the
  ``env.seed(seeds=(s, s))`` pattern referenced in Nethax's NLE_BYTEPARITY path
  (vendor/nle/src/nle.c:530-532).
* The vendor MiniHack base class is built on top of gymnasium-spaces while the
  underlying NLE wrappers ``import gym`` (legacy ``gym``).  When both packages
  are installed the construction trips an ``isinstance`` assert.  We side-step
  the conflict with a sys.modules shim ``gym -> gymnasium`` at import time.
"""
from __future__ import annotations

import argparse
import os
import sys

# JAX setup: CPU + eager mode (matches multiseed_byteparity.py defaults).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
if os.environ.get("NETHAX_EAGER", "0") == "1":
    import jax
    jax.config.update("jax_disable_jit", True)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_REPO_ROOT, ".jax_cache"))

# Vendor minihack uses ``import gymnasium as gym`` while vendor NLE uses
# legacy ``import gym``.  When both are installed, gym.spaces.Box (legacy) is
# not a subclass of gymnasium.spaces.Space, and the MiniHack base class
# construction fails an isinstance assert inside gymnasium.spaces.Dict.
# Aliasing 'gym' -> 'gymnasium' before vendor minihack is imported makes the
# Box instances live under the same gymnasium hierarchy and lets construction
# go through.
import gymnasium as _gymnasium  # noqa: E402
sys.modules["gym"] = _gymnasium
sys.modules["gym.spaces"] = _gymnasium.spaces
sys.modules["gym.envs"] = _gymnasium.envs
sys.modules["gym.envs.registration"] = _gymnasium.envs.registration

import random  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from Nethax.nethax.parity_mode import ParityMode, set_parity_mode  # noqa: E402

set_parity_mode(ParityMode.NLE_BYTEPARITY)

from Nethax.minihax.minihax_env import MinihaxEnv  # noqa: E402
from Nethax.nethax.obs.nle_obs import build_nle_observation  # noqa: E402


# ---------------------------------------------------------------------------
# 12 canonical Room env_ids (Phase 1 today-pass scope).  Cite:
# Nethax/minihax/envs/canonical.py::_register_room_envs (lines 415-428).
# ---------------------------------------------------------------------------
ROOM_ENV_IDS = [
    "MiniHack-Room-5x5-v0",
    "MiniHack-Room-Random-5x5-v0",
    "MiniHack-Room-Dark-5x5-v0",
    "MiniHack-Room-Monster-5x5-v0",
    "MiniHack-Room-Trap-5x5-v0",
    "MiniHack-Room-Ultimate-5x5-v0",
    "MiniHack-Room-15x15-v0",
    "MiniHack-Room-Random-15x15-v0",
    "MiniHack-Room-Dark-15x15-v0",
    "MiniHack-Room-Monster-15x15-v0",
    "MiniHack-Room-Trap-15x15-v0",
    "MiniHack-Room-Ultimate-15x15-v0",
]

# Observation keys both sides will produce (minihax build_nle_observation emits
# all of these; vendor we request explicitly).
_OBS_KEYS = (
    "glyphs", "chars", "blstats", "inv_glyphs", "inv_letters", "inv_strs",
)

# Per-env vendor player character.  Default is the canonical MiniHack
# "arc-hum-law-mal" (Archeologist).  A few envs hardcode a different role in
# their vendor class, so the vendor build MUST use the same character the
# vendor env forces — otherwise the hero @-glyph / inventory / role_init RNG
# diverge from the minihax bootstrap (which spawns the matching role via
# canonical._env_role_overrides).  Cite:
#   * skills_quest.py:16 / fightcorridor.py:8 -> "kni-hum-law-fem"
#   * keyroom.py:34                          -> "rog-hum-cha-mal"
_DEFAULT_CHARACTER = "arc-hum-law-mal"
_ENV_CHARACTER = {
    "MiniHack-Quest-Medium-v0": "kni-hum-law-fem",
    "MiniHack-CorridorBattle-v0": "kni-hum-law-fem",
    "MiniHack-CorridorBattle-Dark-v0": "kni-hum-law-fem",
    "MiniHack-KeyRoom-Fixed-S5-v0": "rog-hum-cha-mal",
    "MiniHack-KeyRoom-S5-v0": "rog-hum-cha-mal",
    "MiniHack-KeyRoom-Dark-S5-v0": "rog-hum-cha-mal",
    "MiniHack-KeyRoom-S15-v0": "rog-hum-cha-mal",
    "MiniHack-KeyRoom-Dark-S15-v0": "rog-hum-cha-mal",
}


def _env_id_to_vendor_cls(env_id: str):
    """Map a canonical MiniHack env_id to its vendor class.

    Avoids hitting the gymnasium registry, which is fragile in our split-gym
    environment.  Only Room-* envs are wired today (Phase 1 scope).
    """
    from minihack.envs import room as _room_mod
    from minihack.envs import skills_simple as _skills_mod
    from minihack.envs import hidenseek as _hns_mod
    from minihack.envs import fightcorridor as _fc_mod
    from minihack.envs import mazewalk as _mw_mod
    from minihack.envs import corridor as _corr_mod
    from minihack.envs import sokoban as _soko_mod
    from minihack.envs import skills_lava as _lava_mod
    from minihack.envs import skills_levitate as _levit_mod
    from minihack.envs import skills_freeze as _freeze_mod
    from minihack.envs import river as _river_mod
    from minihack.envs import keyroom as _kr_mod
    from minihack.envs import skills_quest as _quest_mod
    from minihack.envs import exploremaze as _em_mod
    from minihack.envs import lab as _lab_mod
    from minihack.envs import skills_wod as _wod_mod

    table = {
        "MiniHack-ExploreMaze-Easy-v0":        _em_mod.MiniHackExploreMazeEasy,
        "MiniHack-ExploreMaze-Easy-Mapped-v0": _em_mod.MiniHackExploreMazeEasyMapped,
        "MiniHack-ExploreMaze-Hard-v0":        _em_mod.MiniHackExploreMazeHard,
        "MiniHack-ExploreMaze-Hard-Mapped-v0": _em_mod.MiniHackExploreMazeHardMapped,
        "MiniHack-Labyrinth-Big-v0":           _lab_mod.MiniHackLabyrinth,
        "MiniHack-Labyrinth-Small-v0":         _lab_mod.MiniHackLabyrinthSmall,
        "MiniHack-WoD-Easy-Full-v0":           _wod_mod.MiniHackWoDEasy,
        "MiniHack-WoD-Medium-Full-v0":         _wod_mod.MiniHackWoDMedium,
        "MiniHack-WoD-Hard-Full-v0":           _wod_mod.MiniHackWoDHard,
        "MiniHack-WoD-Pro-Full-v0":            _wod_mod.MiniHackWoDPro,
        "MiniHack-Quest-Medium-v0":        _quest_mod.MiniHackQuestMedium,
        "MiniHack-Levitate-Boots-Full-v0":       _levit_mod.MiniHackLevitateBoots,
        "MiniHack-Levitate-Boots-Restricted-v0":
            _levit_mod.MiniHackLevitateBootsRestrictedActions,
        "MiniHack-Levitate-Boots-Fixed-v0":      _levit_mod.MiniHackLevitateBootsFixed,
        "MiniHack-Levitate-Ring-Full-v0":        _levit_mod.MiniHackLevitateRing,
        "MiniHack-Levitate-Ring-Restricted-v0":
            _levit_mod.MiniHackLevitateRingRestrictedActions,
        "MiniHack-Levitate-Ring-Fixed-v0":       _levit_mod.MiniHackLevitateRingFixed,
        "MiniHack-Levitate-Potion-Full-v0":      _levit_mod.MiniHackLevitatePotion,
        "MiniHack-Levitate-Potion-Restricted-v0":
            _levit_mod.MiniHackLevitatePotionRestrictedActions,
        "MiniHack-Levitate-Potion-Fixed-v0":     _levit_mod.MiniHackLevitatePotionFixed,
        "MiniHack-Levitate-Random-Full-v0":      _levit_mod.MiniHackLevitateRandom,
        "MiniHack-Freeze-Wand-Full-v0":          _freeze_mod.MiniHackFreezeWand,
        "MiniHack-Freeze-Wand-Restricted-v0":
            _freeze_mod.MiniHackFreezeWandRestrictedActions,
        "MiniHack-Freeze-Horn-Full-v0":          _freeze_mod.MiniHackFreezeHorn,
        "MiniHack-Freeze-Horn-Restricted-v0":
            _freeze_mod.MiniHackFreezeHornRestrictedActions,
        "MiniHack-Freeze-Random-Full-v0":        _freeze_mod.MiniHackFreezeRandom,
        "MiniHack-Freeze-Random-Restricted-v0":
            _freeze_mod.MiniHackFreezeRandomRestrictedActions,
        "MiniHack-Freeze-Lava-Full-v0":          _freeze_mod.MiniHackFreezeLava,
        "MiniHack-Freeze-Lava-Restricted-v0":
            _freeze_mod.MiniHackFreezeLavaRestrictedActions,
        "MiniHack-KeyRoom-Fixed-S5-v0":    _kr_mod.MiniHackKeyRoom5x5Fixed,
        "MiniHack-KeyRoom-S5-v0":          _kr_mod.MiniHackKeyRoom5x5,
        "MiniHack-KeyRoom-Dark-S5-v0":     _kr_mod.MiniHackKeyRoom5x5Dark,
        "MiniHack-KeyRoom-S15-v0":         _kr_mod.MiniHackKeyRoom15x15,
        "MiniHack-KeyRoom-Dark-S15-v0":    _kr_mod.MiniHackKeyRoom15x15Dark,
        "MiniHack-LavaCross-Levitate-Potion-Pickup-Full-v0":
            _lava_mod.MiniHackLCLevitatePotionPickup,
        "MiniHack-LavaCross-Levitate-Potion-Pickup-Restricted-v0":
            _lava_mod.MiniHackLCLevitatePotionPickupRestrictedActions,
        "MiniHack-LavaCross-Levitate-Potion-Inv-Full-v0":
            _lava_mod.MiniHackLCLevitatePotionInv,
        "MiniHack-LavaCross-Levitate-Potion-Inv-Restricted-v0":
            _lava_mod.MiniHackLCLevitatePotionInvRestrictedActions,
        "MiniHack-LavaCross-Levitate-Ring-Pickup-Full-v0":
            _lava_mod.MiniHackLCLevitateRingPickup,
        "MiniHack-LavaCross-Levitate-Ring-Pickup-Restricted-v0":
            _lava_mod.MiniHackLCLevitateRingPickupRestrictedActions,
        "MiniHack-LavaCross-Levitate-Ring-Inv-Full-v0":
            _lava_mod.MiniHackLCLevitateRingInv,
        "MiniHack-LavaCross-Levitate-Ring-Inv-Restricted-v0":
            _lava_mod.MiniHackLCLevitateRingInvRestrictedActions,
        "MiniHack-LavaCross-Levitate-Full-v0":
            _lava_mod.MiniHackLCLevitate,
        "MiniHack-LavaCross-Levitate-Restricted-v0":
            _lava_mod.MiniHackLCLevitateRestrictedActions,
        "MiniHack-LavaCross-Full-v0":
            _lava_mod.MiniHackLC,
        "MiniHack-LavaCross-Restricted-v0":
            _lava_mod.MiniHackLCRestrictedActions,
        "MiniHack-River-v0":               _river_mod.MiniHackRiver,
        "MiniHack-River-Monster-v0":       _river_mod.MiniHackRiverMonster,
        "MiniHack-River-Lava-v0":          _river_mod.MiniHackRiverLava,
        "MiniHack-River-MonsterLava-v0":   _river_mod.MiniHackRiverMonsterLava,
        "MiniHack-River-Narrow-v0":        _river_mod.MiniHackRiverNarrow,
        "MiniHack-Corridor-R2-v0":         _corr_mod.MiniHackCorridor2,
        "MiniHack-Corridor-R3-v0":         _corr_mod.MiniHackCorridor3,
        "MiniHack-Corridor-R5-v0":         _corr_mod.MiniHackCorridor5,
        "MiniHack-Sokoban1a-v0":           _soko_mod.MiniHackSokoban1a,
        "MiniHack-Sokoban1b-v0":           _soko_mod.MiniHackSokoban1b,
        "MiniHack-Sokoban2a-v0":           _soko_mod.MiniHackSokoban2a,
        "MiniHack-Sokoban2b-v0":           _soko_mod.MiniHackSokoban2b,
        "MiniHack-Sokoban3a-v0":           _soko_mod.MiniHackSokoban3a,
        "MiniHack-Sokoban3b-v0":           _soko_mod.MiniHackSokoban3b,
        "MiniHack-Sokoban4a-v0":           _soko_mod.MiniHackSokoban4a,
        "MiniHack-Sokoban4b-v0":           _soko_mod.MiniHackSokoban4b,
        "MiniHack-ClosedDoor-v0":          _skills_mod.MiniHackClosedDoor,
        "MiniHack-LockedDoor-v0":          _skills_mod.MiniHackLockedDoor,
        "MiniHack-LockedDoor-Fixed-v0":    _skills_mod.MiniHackLockedDoorFixed,
        "MiniHack-Eat-v0":                 _skills_mod.MiniHackEat,
        "MiniHack-Pray-v0":                _skills_mod.MiniHackPray,
        "MiniHack-Wield-v0":               _skills_mod.MiniHackWield,
        "MiniHack-Zap-v0":                 _skills_mod.MiniHackZap,
        "MiniHack-Read-v0":                _skills_mod.MiniHackRead,
        "MiniHack-Wear-v0":                _skills_mod.MiniHackWear,
        "MiniHack-PutOn-v0":               _skills_mod.MiniHackPutOn,
        "MiniHack-Sink-v0":                _skills_mod.MiniHackSink,
        "MiniHack-Eat-Fixed-v0":           _skills_mod.MiniHackEatFixed,
        "MiniHack-Pray-Fixed-v0":          _skills_mod.MiniHackPrayFixed,
        "MiniHack-Wield-Fixed-v0":         _skills_mod.MiniHackWieldFixed,
        "MiniHack-Zap-Fixed-v0":           _skills_mod.MiniHackZapFixed,
        "MiniHack-Read-Fixed-v0":          _skills_mod.MiniHackReadFixed,
        "MiniHack-Wear-Fixed-v0":          _skills_mod.MiniHackWearFixed,
        "MiniHack-PutOn-Fixed-v0":         _skills_mod.MiniHackPutOnFixed,
        "MiniHack-Sink-Fixed-v0":          _skills_mod.MiniHackSinkFixed,
        "MiniHack-Eat-Distr-v0":           _skills_mod.MiniHackEatDistr,
        "MiniHack-Pray-Distr-v0":          _skills_mod.MiniHackPrayDistr,
        "MiniHack-Wield-Distr-v0":         _skills_mod.MiniHackWieldDistr,
        "MiniHack-Zap-Distr-v0":           _skills_mod.MiniHackZapDistr,
        "MiniHack-Read-Distr-v0":          _skills_mod.MiniHackReadDistr,
        "MiniHack-Wear-Distr-v0":          _skills_mod.MiniHackWearDistr,
        "MiniHack-PutOn-Distr-v0":         _skills_mod.MiniHackPutOnDistr,
        "MiniHack-Sink-Distr-v0":          _skills_mod.MiniHackSinkDistr,
        "MiniHack-CorridorBattle-v0":      _fc_mod.MiniHackFightCorridor,
        "MiniHack-CorridorBattle-Dark-v0": _fc_mod.MiniHackFightCorridorDark,
        "MiniHack-HideNSeek-v0":           _hns_mod.MiniHackHideAndSeek,
        "MiniHack-HideNSeek-Mapped-v0":    _hns_mod.MiniHackHideAndSeekMapped,
        "MiniHack-HideNSeek-Lava-v0":      _hns_mod.MiniHackHideAndSeekLava,
        "MiniHack-HideNSeek-Big-v0":       _hns_mod.MiniHackHideAndSeekBig,
        "MiniHack-MazeWalk-9x9-v0":        _mw_mod.MiniHackMazeWalk9x9,
        "MiniHack-MazeWalk-15x15-v0":      _mw_mod.MiniHackMazeWalk15x15,
        "MiniHack-MazeWalk-45x19-v0":      _mw_mod.MiniHackMazeWalk45x19,
        "MiniHack-Room-5x5-v0":            _room_mod.MiniHackRoom5x5,
        "MiniHack-Room-Random-5x5-v0":     _room_mod.MiniHackRoom5x5Random,
        "MiniHack-Room-Dark-5x5-v0":       _room_mod.MiniHackRoom5x5Dark,
        "MiniHack-Room-Monster-5x5-v0":    _room_mod.MiniHackRoom5x5Monster,
        "MiniHack-Room-Trap-5x5-v0":       _room_mod.MiniHackRoom5x5Trap,
        "MiniHack-Room-Ultimate-5x5-v0":   _room_mod.MiniHackRoom5x5Ultimate,
        "MiniHack-Room-15x15-v0":          _room_mod.MiniHackRoom15x15,
        "MiniHack-Room-Random-15x15-v0":   _room_mod.MiniHackRoom15x15Random,
        "MiniHack-Room-Dark-15x15-v0":     _room_mod.MiniHackRoom15x15Dark,
        "MiniHack-Room-Monster-15x15-v0":  _room_mod.MiniHackRoom15x15Monster,
        "MiniHack-Room-Trap-15x15-v0":     _room_mod.MiniHackRoom15x15Trap,
        "MiniHack-Room-Ultimate-15x15-v0": _room_mod.MiniHackRoom15x15Ultimate,
    }
    if env_id not in table:
        raise KeyError(
            f"env_id {env_id!r} not in Phase-1 vendor class table; "
            "extend _env_id_to_vendor_cls() to wire it."
        )
    return table[env_id]


def _agent_yx_from_blstats(blstats: np.ndarray) -> tuple[int, int]:
    """NLE blstats layout: [x, y, ...] -> return (y, x)."""
    return int(blstats[1]), int(blstats[0])


def vendor_dump(env_id: str, seed: int) -> dict:
    """Build vendor MiniHack env, seed via NLE convention, return obs dump."""
    cls = _env_id_to_vendor_cls(env_id)
    # IMPORTANT: do NOT pass ``seeds=`` -> keeps ``_level_seeds`` None so the
    # vendor base.py reset() branch (line 373) is skipped (no Python random
    # burn).
    random.seed(seed)  # neutralise any other ad-hoc random use during setup
    # Minihax MinihaxEnv stores defaults as player_role=0=ARCHEOLOGIST,
    # player_race=0=HUMAN, player_align=0=LAWFUL on the EnvState (verified
    # at reset via key(0)).  Despite env.py:105 saying Role.VALKYRIE is the
    # reset default, the MinihaxEnv path keeps the EnvState zero-init,
    # producing @ glyph 327 (Archeologist PM).  Vendor character format
    # is "role-race-align-gender" (vendor/minihack/minihack/navigation.py:38).
    character = _ENV_CHARACTER.get(env_id, _DEFAULT_CHARACTER)
    env = cls(observation_keys=_OBS_KEYS, character=character)
    assert env._level_seeds is None, (
        "vendor env should have _level_seeds=None to avoid silent draw"
    )
    # NLE seed convention: (core, disp).  Cite vendor/nle/src/nle.c:530-532.
    env.seed(seed, seed, reseed=False)
    r = env.reset()
    obs = r[0] if isinstance(r, tuple) else r
    glyphs = np.asarray(obs["glyphs"])
    chars = np.asarray(obs["chars"])
    blstats = np.asarray(obs["blstats"])
    inv_glyphs = np.asarray(obs["inv_glyphs"])
    inv_letters = np.asarray(obs["inv_letters"])
    inv_strs = np.asarray(obs["inv_strs"])
    try:
        env.close()
    except Exception:
        pass
    return {
        "glyphs":       glyphs,
        "chars":        chars,
        "agent_yx":     _agent_yx_from_blstats(blstats),
        "inv_glyphs":   inv_glyphs,
        "inv_letters":  inv_letters,
        "inv_strs":     inv_strs,
    }


def minihax_dump(env_id: str, seed: int) -> dict:
    """Build minihax env, reset with jax.random.key(seed), return obs dump."""
    env = MinihaxEnv(env_id)
    state, _info = env.reset(jax.random.key(seed))
    obs = build_nle_observation(state)
    glyphs = np.asarray(obs["glyphs"])
    chars = np.asarray(obs["chars"])
    blstats = np.asarray(obs["blstats"])
    inv_glyphs = np.asarray(obs["inv_glyphs"])
    inv_letters = np.asarray(obs["inv_letters"])
    inv_strs = np.asarray(obs["inv_strs"])
    return {
        "glyphs":       glyphs,
        "chars":        chars,
        "agent_yx":     _agent_yx_from_blstats(blstats),
        "inv_glyphs":   inv_glyphs,
        "inv_letters":  inv_letters,
        "inv_strs":     inv_strs,
    }


def diff_dumps(vendor: dict, minihax: dict) -> str | None:
    """Return None on byte-equal match, else a short diagnostic string.

    Diff order: glyphs (per-cell), chars (per-cell), agent_yx, inv_glyphs,
    inv_letters, inv_strs.  Reports first divergent cell only.
    """
    # 2-D arrays first
    for key in ("glyphs", "chars"):
        v = vendor[key]
        m = minihax[key]
        if v.shape != m.shape:
            return f"{key}: shape mismatch vendor={v.shape} minihax={m.shape}"
        ne = np.argwhere(v != m)
        if ne.size:
            y, x = int(ne[0, 0]), int(ne[0, 1])
            return (
                f"{key} first div at (y={y},x={x}): "
                f"vendor={int(v[y, x])} minihax={int(m[y, x])}"
            )

    # Agent position
    if vendor["agent_yx"] != minihax["agent_yx"]:
        return (
            f"agent_yx mismatch vendor={vendor['agent_yx']} "
            f"minihax={minihax['agent_yx']}"
        )

    # 1-D inventory arrays
    for key in ("inv_glyphs", "inv_letters"):
        v = vendor[key]
        m = minihax[key]
        if v.shape != m.shape:
            return f"{key}: shape mismatch vendor={v.shape} minihax={m.shape}"
        ne = np.argwhere(v != m)
        if ne.size:
            i = int(ne[0, 0])
            return (
                f"{key} first div at i={i}: "
                f"vendor={int(v[i])} minihax={int(m[i])}"
            )

    # inv_strs is 2-D (slot, char) but flatten for byte compare
    v = vendor["inv_strs"]
    m = minihax["inv_strs"]
    if v.shape != m.shape:
        return f"inv_strs: shape mismatch vendor={v.shape} minihax={m.shape}"
    if not np.array_equal(v, m):
        ne = np.argwhere(v != m)
        i, j = int(ne[0, 0]), int(ne[0, 1])
        return (
            f"inv_strs first div at (slot={i},byte={j}): "
            f"vendor={int(v[i, j])} minihax={int(m[i, j])}"
        )

    return None


def run_one(env_id: str, seed: int, *, verbose: bool = True) -> bool:
    """Run a single env_id/seed pair.  Return True on PASS, False on FAIL."""
    try:
        vendor = vendor_dump(env_id, seed)
    except Exception as e:
        print(f"  {env_id}  seed={seed}  ERROR (vendor): {type(e).__name__}: {e}")
        return False
    try:
        minihax = minihax_dump(env_id, seed)
    except Exception as e:
        print(f"  {env_id}  seed={seed}  ERROR (minihax): {type(e).__name__}: {e}")
        return False

    diag = diff_dumps(vendor, minihax)
    if diag is None:
        if verbose:
            print(f"  {env_id}  seed={seed}  PASS")
        return True
    if verbose:
        print(f"  {env_id}  seed={seed}  FAIL  {diag}")
    return False


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", default="MiniHack-Room-5x5-v0",
                   help="Single env_id (ignored if --all-rooms is set).")
    p.add_argument("--seed", type=int, default=0, help="Seed for both sides.")
    p.add_argument("--all-rooms", action="store_true",
                   help="Iterate all 12 Room-* envs and tally PASS/FAIL.")
    args = p.parse_args(argv)

    if args.all_rooms:
        print(f"[minihax_byteparity] --all-rooms  seed={args.seed}")
        n_pass = 0
        for env_id in ROOM_ENV_IDS:
            if run_one(env_id, args.seed, verbose=True):
                n_pass += 1
        print(f"\n[summary] envs passed: {n_pass} / {len(ROOM_ENV_IDS)}")
        return 0 if n_pass == len(ROOM_ENV_IDS) else 1

    print(f"[minihax_byteparity] env={args.env}  seed={args.seed}")
    ok = run_one(args.env, args.seed, verbose=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
