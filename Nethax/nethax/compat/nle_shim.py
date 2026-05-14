"""NLE compatibility shim — wraps ``NethaxEnv`` to look like ``nle.env.NLE``.

This shim exposes a gymnasium-0.26+ style API on top of the underlying
``NethaxEnv`` JAX environment, with the canonical NLE observation/action
contract so RL agents targeting NLE can run on nethax with no rewiring.

API contract (mirrors vendor/nle/nle/env/base.py::NLE):
    nh = NLECompat(seed=0, character="mon-hum-neu-mal")
    obs, info = nh.reset()
    obs, reward, terminated, truncated, info = nh.step(action_index_or_int)

Surface:
    - ``actions``           : tuple of 121 canonical NLE action ints
    - ``action_set``        : alias of ``actions``
    - ``observation_space`` : ``gymnasium.spaces.Dict`` with 17 NLE keys
    - ``action_space``      : ``gymnasium.spaces.Discrete(121)``
    - ``StepStatus``        : IntEnum (ABORTED=-1, RUNNING=0, DEATH=1)
    - ``metadata``          : {"render_modes": ["human", "ansi", "full"]}
    - glyph helpers (static)
        nethack_glyph_to_char(glyph) -> str
        nethack_glyph_is_monster(glyph) -> bool
        nethack_glyph_is_object(glyph)  -> bool
        nethack_glyph_is_cmap(glyph)    -> bool
        nethack_glyph_is_pet(glyph)     -> bool
        nethack_glyph_is_body(glyph)    -> bool
        nethack_glyph_is_invisible(g)   -> bool
        nethack_glyph_is_statue(glyph)  -> bool
        nethack_glyph_is_swallow(glyph) -> bool
        nethack_glyph_is_warning(glyph) -> bool

Citations:
    vendor/nle/nle/nethack/nethack.py — ``Nethack`` class, ``OBSERVATION_DESC``.
    vendor/nle/nle/env/base.py        — ``NLE`` gym env wrapper, step/reset.
    vendor/nle/nle/nethack/actions.py — ``ACTIONS`` tuple of 121 ints.
    vendor/nethack/include/display.h   — glyph_is_* C macro definitions.
"""
from __future__ import annotations

import enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from Nethax.nethax.env import NethaxEnv
from Nethax.nethax.obs.nle_obs import (
    NLE_OBSERVATION_KEYS,
    NLE_OBSERVATION_SHAPES,
    NLE_OBSERVATION_DTYPES,
)
from Nethax.nethax.constants.actions import ACTIONS, N_ACTIONS
from Nethax.nethax.constants.glyphs import (
    GLYPH_MON_OFF,
    GLYPH_PET_OFF,
    GLYPH_INVIS_OFF,
    GLYPH_DETECT_OFF,
    GLYPH_BODY_OFF,
    GLYPH_RIDDEN_OFF,
    GLYPH_OBJ_OFF,
    GLYPH_CMAP_OFF,
    GLYPH_EXPLODE_OFF,
    GLYPH_ZAP_OFF,
    GLYPH_SWALLOW_OFF,
    GLYPH_WARNING_OFF,
    GLYPH_STATUE_OFF,
    MAX_GLYPH,
    NO_GLYPH,
    NUMMONS,
    NUM_OBJECTS,
)
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.constants.races import Race


# ---------------------------------------------------------------------------
# Character string parsing (e.g. "mon-hum-neu-mal" -> Role.MONK, Race.HUMAN, ...)
# Matches vendor/nle/nle/env/base.py default character="mon-hum-neu-mal".
# ---------------------------------------------------------------------------

_ROLE_TOKEN_TO_ENUM: Dict[str, Role] = {
    "arc": Role.ARCHEOLOGIST,
    "bar": Role.BARBARIAN,
    "cav": Role.CAVEMAN,
    "hea": Role.HEALER,
    "kni": Role.KNIGHT,
    "mon": Role.MONK,
    "pri": Role.PRIEST,
    "ran": Role.RANGER,
    "rog": Role.ROGUE,
    "sam": Role.SAMURAI,
    "tou": Role.TOURIST,
    "val": Role.VALKYRIE,
    "wiz": Role.WIZARD,
}

_RACE_TOKEN_TO_ENUM: Dict[str, Race] = {
    "hum": Race.HUMAN,
    "elf": Race.ELF,
    "dwa": Race.DWARF,
    "gno": Race.GNOME,
    "orc": Race.ORC,
}

_ALIGN_TOKEN_TO_INT: Dict[str, int] = {
    "law": 0,
    "neu": 1,
    "cha": 2,
}


def _parse_character_string(
    character: str,
) -> Tuple[Optional[Role], Optional[Race], Optional[int]]:
    """Parse ``"mon-hum-neu-mal"`` into ``(Role, Race, alignment_int)``.

    Tokens follow vendor/nle convention.  Unknown tokens are ignored.
    Returns ``(None, None, None)`` for the wildcard ``"@"`` (random role).
    """
    if not character or character == "@":
        return None, None, None
    parts = [p.strip().lower() for p in character.split("-") if p.strip()]
    role: Optional[Role] = None
    race: Optional[Race] = None
    align: Optional[int] = None
    for tok in parts:
        if role is None and tok in _ROLE_TOKEN_TO_ENUM:
            role = _ROLE_TOKEN_TO_ENUM[tok]
        elif race is None and tok in _RACE_TOKEN_TO_ENUM:
            race = _RACE_TOKEN_TO_ENUM[tok]
        elif align is None and tok in _ALIGN_TOKEN_TO_INT:
            align = _ALIGN_TOKEN_TO_INT[tok]
        # "mal"/"fem" gender tokens are accepted but ignored.
    return role, race, align


# ---------------------------------------------------------------------------
# Observation space builder — mirrors vendor/nle/nle/env/base.py NLE_SPACE_ITEMS
# ---------------------------------------------------------------------------


def _np_dtype_for(jax_dtype) -> np.dtype:
    """Map a JAX dtype to its numpy equivalent for gym Box construction."""
    return np.dtype(jnp.dtype(jax_dtype))


def _build_observation_space():
    """Build a ``gymnasium.spaces.Dict`` matching the 17 NLE observation keys.

    Raises:
        ImportError if gymnasium is unavailable.
    """
    from gymnasium import spaces  # local import keeps gymnasium optional at import time

    # Per-key (low, high) bounds, matching vendor/nle NLE_SPACE_ITEMS.
    low_high: Dict[str, Tuple[int, int]] = {
        "glyphs":              (0, MAX_GLYPH),
        "chars":               (0, 255),
        "colors":              (0, 15),
        "specials":            (0, 255),
        "blstats":             (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        "message":             (0, 255),
        "program_state":       (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        "internal":            (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        "inv_glyphs":          (0, MAX_GLYPH),
        "inv_strs":            (0, 255),
        "inv_letters":         (0, 127),
        "inv_oclasses":        (0, 18),  # MAXOCLASSES in vendor NLE
        "screen_descriptions": (0, 127),
        "tty_chars":           (0, 255),
        "tty_colors":          (0, 31),
        "tty_cursor":          (0, 255),
        "misc":                (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
    }
    items = {}
    for key in NLE_OBSERVATION_KEYS:
        shape = NLE_OBSERVATION_SHAPES[key]
        dtype = _np_dtype_for(NLE_OBSERVATION_DTYPES[key])
        low, high = low_high[key]
        items[key] = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return spaces.Dict(items)


def _build_action_space():
    """Build a ``gymnasium.spaces.Discrete(121)``."""
    from gymnasium import spaces
    return spaces.Discrete(N_ACTIONS)


# ---------------------------------------------------------------------------
# Main shim class
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Optional gymnasium.Env base — fall back to ``object`` if gymnasium is missing
# so the module still imports in headless / no-gym environments.
# Citation: vendor/nle/nle/env/base.py imports ``gym`` (aliased to gymnasium
# in NLE 1.3.0) and defines ``class NLE(gym.Env)``.
# ---------------------------------------------------------------------------

try:
    import gymnasium as _gym  # type: ignore
    _GymEnvBase = _gym.Env
    _HAS_GYM = True
except ImportError:  # pragma: no cover — gymnasium is a hard test dep
    _GymEnvBase = object  # type: ignore[misc,assignment]
    _HAS_GYM = False


class NLECompat(_GymEnvBase):  # type: ignore[misc,valid-type]
    """Drop-in replacement for ``nle.env.NLE`` backed by ``NethaxEnv``.

    Inherits from :class:`gymnasium.Env` (when available) so gymnasium
    wrappers — ``TimeLimit``, ``RecordEpisodeStatistics``, etc. — can
    wrap this env without modification (vendor/nle/nle/env/base.py uses
    ``class NLE(gym.Env)`` with ``gym`` aliased to gymnasium).

    Args:
        seed:        PRNG seed (int).
        character:   NLE-style "role-race-align-gender" string, e.g.
                     ``"mon-hum-neu-mal"``. ``"@"`` or empty -> wildcard.
        role:        explicit Role enum override (takes precedence over
                     ``character``).
        race:        explicit Race enum override.
        alignment:   explicit alignment int override (0=law, 1=neu, 2=cha).
        observation_keys: optional subset of NLE_OBSERVATION_KEYS to return
                     from reset/step.  Defaults to all 17 keys.
        max_episode_steps: NLE-style step-limit truncation, default 5000
                     (matches vendor/nle/nle/env/base.py default).
        savedir:     ignored — NethaxEnv has no ttyrec stream. Surface
                     attribute kept for parity (always ``None``).
        save_ttyrec_every / wizard / allow_all_yn_questions / allow_all_modes /
        spawn_monsters / options / fix_moon_phase:
                     accepted for vendor-signature parity; no-ops.
        render_mode: gymnasium render-mode hint, default ``"human"``.

    Attributes:
        actions, action_set: tuple of 121 canonical NLE action ints.
        observation_space:   gymnasium.spaces.Dict of 17 NLE keys.
        action_space:        gymnasium.spaces.Discrete(121).
        StepStatus:          IntEnum (ABORTED=-1, RUNNING=0, DEATH=1).
        metadata:            {"render_modes": [...]}.
        last_observation:    tuple of np.ndarrays in ``_observation_keys`` order
                             (vendor parity — populated after reset/step).
    """

    # Class-level tuple of canonical NLE action ints — matches vendor exactly.
    actions: Tuple[int, ...] = tuple(int(a) for a in ACTIONS)
    action_set: Tuple[int, ...] = actions  # alias

    # gymnasium metadata — vendor uses {"render.modes": [...]} (old gym),
    # we use the new key ``render_modes`` per gymnasium 0.26+.
    metadata: Dict[str, Any] = {"render_modes": ["human", "ansi", "full"]}

    # Citation: vendor/nle/nle/env/base.py::NLE.StepStatus
    class StepStatus(enum.IntEnum):
        ABORTED = -1
        RUNNING = 0
        DEATH = 1

    def __init__(
        self,
        seed: int = 0,
        character: str = "mon-hum-neu-mal",
        role: Optional[Role] = None,
        race: Optional[Race] = None,
        alignment: Optional[int] = None,
        observation_keys: Optional[Iterable[str]] = None,
        max_episode_steps: int = 5000,
        savedir: Optional[str] = None,
        save_ttyrec_every: int = 0,
        wizard: bool = False,
        allow_all_yn_questions: bool = False,
        allow_all_modes: bool = False,
        spawn_monsters: bool = True,
        options: Optional[Any] = None,
        fix_moon_phase: bool = False,
        render_mode: str = "human",
    ):
        if _HAS_GYM:
            super().__init__()  # type: ignore[misc]
        self._env = NethaxEnv()
        self._seed = int(seed)
        self._rng = jax.random.PRNGKey(self._seed)
        self._state = None  # populated on first reset
        self.character = character

        # Resolve role/race/alignment: explicit args win, else parse character.
        parsed_role, parsed_race, parsed_align = _parse_character_string(character)
        self._role: Optional[Role] = role if role is not None else parsed_role
        self._race: Optional[Race] = race if race is not None else parsed_race
        if alignment is not None:
            self._alignment: int = int(alignment)
        elif parsed_align is not None:
            self._alignment = int(parsed_align)
        else:
            self._alignment = 0

        # Observation key filter.
        if observation_keys is None:
            self._observation_keys: Tuple[str, ...] = tuple(NLE_OBSERVATION_KEYS)
        else:
            keys = tuple(observation_keys)
            for k in keys:
                if k not in NLE_OBSERVATION_KEYS:
                    raise ValueError(f"Unknown observation key: {k!r}")
            self._observation_keys = keys

        # Vendor-parity surface — populated lazily, all no-ops in nethax.
        self._max_episode_steps = int(max_episode_steps)
        self.savedir: Optional[str] = None  # ttyrec is not produced by NethaxEnv
        self._save_ttyrec_every = int(save_ttyrec_every)
        self._wizard = bool(wizard)
        self._allow_all_yn_questions = bool(allow_all_yn_questions)
        self._allow_all_modes = bool(allow_all_modes)
        self._spawn_monsters = bool(spawn_monsters)
        self._options = options
        self._fix_moon_phase = bool(fix_moon_phase)
        self.render_mode = render_mode

        # last_observation: vendor stores a tuple of np.ndarrays in
        # ``_observation_keys`` order (vendor/nle/nle/env/base.py line 238).
        self.last_observation: Tuple[np.ndarray, ...] = ()
        # Per-episode step counter — used for max_episode_steps truncation.
        self._steps = 0
        self._episode = -1

        # Build gymnasium spaces lazily so the import only fires if used.
        self._observation_space = None
        self._action_space = None

    # ------------------------------------------------------------------
    # gymnasium.spaces (lazy)
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        if self._observation_space is None:
            full = _build_observation_space()
            # Filter to the active keys (preserve insertion order).
            if set(self._observation_keys) != set(NLE_OBSERVATION_KEYS):
                from gymnasium import spaces
                self._observation_space = spaces.Dict(
                    {k: full.spaces[k] for k in self._observation_keys}
                )
            else:
                self._observation_space = full
        return self._observation_space

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = _build_action_space()
        return self._action_space

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _filter_obs(self, obs: Dict[str, jax.Array]) -> Dict[str, jax.Array]:
        """Filter observation dict to the configured subset of keys."""
        if set(self._observation_keys) == set(NLE_OBSERVATION_KEYS):
            return dict(obs)
        return {k: obs[k] for k in self._observation_keys}

    def _set_last_observation(self, obs_dict: Dict[str, jax.Array]) -> None:
        """Store ``obs_dict`` as a positional tuple in vendor order.

        Vendor parity: ``last_observation`` is a tuple of np.ndarrays in the
        same order as ``_observation_keys`` (vendor/nle/nle/env/base.py:238).
        """
        self.last_observation = tuple(
            np.asarray(obs_dict[k]) for k in self._observation_keys
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, jax.Array], Dict[str, Any]]:
        """Reset the environment.

        Gymnasium 0.26+ API: returns ``(obs, info)``.  Optional ``seed`` arg
        reseeds the PRNG, matching gymnasium.Env.reset semantics.

        Returns:
            (obs_dict, info_dict)
        """
        if seed is not None:
            self._seed = int(seed)
            self._rng = jax.random.PRNGKey(self._seed)
        self._rng, sub = jax.random.split(self._rng)
        self._state, obs = self._env.reset(
            sub,
            role=self._role,
            race=self._race,
            alignment=self._alignment,
        )
        self._steps = 0
        self._episode += 1
        filtered = self._filter_obs(obs)
        self._set_last_observation(filtered)
        info: Dict[str, Any] = {}
        return filtered, info

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, jax.Array], float, bool, bool, Dict[str, Any]]:
        """Apply ``action`` and return the 5-tuple per gymnasium 0.26+.

        Args:
            action: either a discrete action index in [0, 120] or a raw
                    ASCII action int (one of ``self.actions``).

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        if self._state is None:
            self.reset()
        # Allow both Discrete index and raw ASCII int.
        a = int(action)
        if 0 <= a < N_ACTIONS and a not in self.actions:
            # Interpret as discrete index into the action tuple.
            a = int(self.actions[a])
        self._rng, sub = jax.random.split(self._rng)
        state, obs, reward, done, info = self._env.step(
            self._state, jnp.int32(a), sub,
        )
        self._state = state
        self._steps += 1
        terminated = bool(done)
        # Vendor parity: NLE returns ``truncated=True`` when the episode hits
        # ``max_episode_steps`` (vendor/nle/nle/env/base.py::_check_abort).
        truncated = self._steps >= self._max_episode_steps
        filtered = self._filter_obs(obs)
        self._set_last_observation(filtered)
        # Vendor-style end_status info field.
        if terminated:
            end_status = self.StepStatus.DEATH
        elif truncated:
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self.StepStatus.RUNNING
        info_out = dict(info)
        info_out.setdefault("end_status", end_status)
        return (
            filtered,
            float(reward),
            terminated,
            truncated,
            info_out,
        )

    def close(self) -> None:  # pragma: no cover — nothing to release
        """Release resources. NethaxEnv is pure JAX so this is a no-op."""
        self._state = None

    def seed(
        self,
        core: Optional[int] = None,
        disp: Optional[int] = None,
        reseed: bool = False,
    ) -> Tuple[int, int, bool]:
        """Set the PRNG seed.  Vendor-compatible signature.

        Vendor (vendor/nle/nle/env/base.py::seed) uses ``(core, disp, reseed)``
        and returns the triple.  NethaxEnv has a single JAX PRNG, so we map:
            - ``core`` -> PRNG key seed (when provided)
            - ``disp`` -> stored for parity but unused
            - ``reseed`` -> stored for parity but always False internally

        Returns:
            ``(core, disp, reseed)`` matching the vendor contract.
        """
        if core is None:
            # Vendor uses random.SystemRandom; we use the existing PRNG state
            # to derive a deterministic-but-fresh int so seed() is reproducible
            # when no args are supplied yet we still return *some* int.
            core_key, self._rng = jax.random.split(self._rng)
            core = int(jax.random.randint(
                core_key, (), 0, np.iinfo(np.int32).max
            ))
        if disp is None:
            disp_key, self._rng = jax.random.split(self._rng)
            disp = int(jax.random.randint(
                disp_key, (), 0, np.iinfo(np.int32).max
            ))
        self._seed = int(core)
        self._rng = jax.random.PRNGKey(self._seed)
        self._disp_seed = int(disp)
        self._reseed_flag = bool(reseed)
        return (int(core), int(disp), bool(reseed))

    def get_seeds(self) -> Tuple[int, int, bool]:
        """Return the current ``(core, disp, reseed)`` triple (vendor parity).

        Citation: vendor/nle/nle/env/base.py::get_seeds.
        """
        disp = getattr(self, "_disp_seed", 0)
        reseed = getattr(self, "_reseed_flag", False)
        return (int(self._seed), int(disp), bool(reseed))

    def render(self, mode: Optional[str] = None) -> Optional[str]:
        """Render the environment.

        Modes (vendor/nle/nle/env/base.py::render):
            - ``"human"``: print a tty-style frame to stdout, return None.
            - ``"ansi"``: return a string representation of ``tty_chars``.
            - ``"full"``: print the message + inventory and return None.

        ``mode=None`` defers to ``self.render_mode``.
        """
        if mode is None:
            mode = self.render_mode
        if not self.last_observation:
            return None
        try:
            chars_idx = self._observation_keys.index("tty_chars")
        except ValueError:
            chars_idx = None

        if mode == "ansi" and chars_idx is not None:
            tty_chars = self.last_observation[chars_idx]
            lines: List[str] = []
            for row in np.asarray(tty_chars):
                line = "".join(
                    chr(c) if 32 <= int(c) < 127 else " " for c in row
                )
                lines.append(line.rstrip())
            return "\n".join(lines)

        if mode == "human" and chars_idx is not None:
            tty_chars = self.last_observation[chars_idx]
            for row in np.asarray(tty_chars):
                line = "".join(
                    chr(c) if 32 <= int(c) < 127 else " " for c in row
                )
                print(line.rstrip())
            return None

        if mode == "full":
            try:
                msg_idx = self._observation_keys.index("message")
                msg = bytes(np.asarray(self.last_observation[msg_idx]))
                if b"\0" in msg:
                    msg = msg[: msg.index(b"\0")]
                print(msg.decode("ascii", errors="replace"))
            except ValueError:
                pass
            return None

        return None

    def print_action_meanings(self) -> None:
        """Print each action with its ASCII char.  Vendor parity helper.

        Citation: vendor/nle/nle/env/base.py::print_action_meanings.
        """
        for idx, a in enumerate(self.actions):
            ch = chr(a) if 32 <= int(a) < 127 else f"\\x{int(a):02x}"
            print(f"{idx}: {a} ({ch})")

    @property
    def obs_keys(self) -> Tuple[str, ...]:
        """Return the canonical 17-tuple of NLE observation keys."""
        return tuple(NLE_OBSERVATION_KEYS)

    # ------------------------------------------------------------------
    # Static glyph helpers — mirror vendor/nethack/include/display.h
    #   glyph_is_*  macros and nle.nethack.glyph_is_*  functions.
    # ------------------------------------------------------------------

    @staticmethod
    def nethack_glyph_to_char(glyph: int) -> str:
        """Return the printable ASCII char for a glyph, or '?' if non-printable."""
        g = int(glyph)
        if 32 <= g < 127:
            return chr(g)
        return "?"

    @staticmethod
    def nethack_glyph_is_monster(glyph: int) -> bool:
        """True for normal monster glyphs (incl. pets, detected, ridden)."""
        g = int(glyph)
        # Vendor C macro glyph_is_monster covers: normal mon, pet, detected, ridden.
        return (
            (GLYPH_MON_OFF <= g < GLYPH_MON_OFF + NUMMONS)
            or (GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS)
            or (GLYPH_DETECT_OFF <= g < GLYPH_DETECT_OFF + NUMMONS)
            or (GLYPH_RIDDEN_OFF <= g < GLYPH_RIDDEN_OFF + NUMMONS)
        )

    @staticmethod
    def nethack_glyph_is_normal_monster(glyph: int) -> bool:
        """True for the plain monster band only (no pet/detect/ridden)."""
        g = int(glyph)
        return GLYPH_MON_OFF <= g < GLYPH_MON_OFF + NUMMONS

    @staticmethod
    def nethack_glyph_is_pet(glyph: int) -> bool:
        """True if ``glyph`` is in the pet band."""
        g = int(glyph)
        return GLYPH_PET_OFF <= g < GLYPH_PET_OFF + NUMMONS

    @staticmethod
    def nethack_glyph_is_body(glyph: int) -> bool:
        """True if ``glyph`` is a corpse/body."""
        g = int(glyph)
        return GLYPH_BODY_OFF <= g < GLYPH_BODY_OFF + NUMMONS

    @staticmethod
    def nethack_glyph_is_invisible(glyph: int) -> bool:
        """True if ``glyph`` is the single invisible-monster glyph."""
        return int(glyph) == GLYPH_INVIS_OFF

    @staticmethod
    def nethack_glyph_is_object(glyph: int) -> bool:
        """True if ``glyph`` is in the object band."""
        g = int(glyph)
        return GLYPH_OBJ_OFF <= g < GLYPH_OBJ_OFF + NUM_OBJECTS

    @staticmethod
    def nethack_glyph_is_cmap(glyph: int) -> bool:
        """True if ``glyph`` is in the cmap (terrain) band."""
        g = int(glyph)
        return GLYPH_CMAP_OFF <= g < GLYPH_EXPLODE_OFF

    @staticmethod
    def nethack_glyph_is_swallow(glyph: int) -> bool:
        """True if ``glyph`` is a swallow effect glyph."""
        g = int(glyph)
        return GLYPH_SWALLOW_OFF <= g < GLYPH_WARNING_OFF

    @staticmethod
    def nethack_glyph_is_warning(glyph: int) -> bool:
        """True if ``glyph`` is a warning glyph."""
        g = int(glyph)
        return GLYPH_WARNING_OFF <= g < GLYPH_STATUE_OFF

    @staticmethod
    def nethack_glyph_is_statue(glyph: int) -> bool:
        """True if ``glyph`` is a statue glyph."""
        g = int(glyph)
        return GLYPH_STATUE_OFF <= g < GLYPH_STATUE_OFF + NUMMONS


# Module-level convenience aliases (callable as plain functions, matching
# vendor/nle/nle/nethack/__init__.py exports).
nethack_glyph_to_char        = NLECompat.nethack_glyph_to_char
nethack_glyph_is_monster     = NLECompat.nethack_glyph_is_monster
nethack_glyph_is_normal_monster = NLECompat.nethack_glyph_is_normal_monster
nethack_glyph_is_pet         = NLECompat.nethack_glyph_is_pet
nethack_glyph_is_body        = NLECompat.nethack_glyph_is_body
nethack_glyph_is_invisible   = NLECompat.nethack_glyph_is_invisible
nethack_glyph_is_object      = NLECompat.nethack_glyph_is_object
nethack_glyph_is_cmap        = NLECompat.nethack_glyph_is_cmap
nethack_glyph_is_swallow     = NLECompat.nethack_glyph_is_swallow
nethack_glyph_is_warning     = NLECompat.nethack_glyph_is_warning
nethack_glyph_is_statue      = NLECompat.nethack_glyph_is_statue
