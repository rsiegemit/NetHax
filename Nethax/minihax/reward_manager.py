"""MiniHack ``RewardManager`` port for nethax.

Canonical source: vendor/minihack/minihack/reward_manager.py

Design summary
--------------
MiniHack's ``RewardManager`` is a Python-side registry of *events*.  Each
event has:
  * a *predicate* that inspects the (prev_state, new_state) transition,
  * a scalar ``reward`` paid when the predicate first fires,
  * three flags: ``repeatable``, ``terminal_required``, ``terminal_sufficient``.

Per step, the manager:
  1. Evaluates every not-yet-fired event's predicate.
  2. Sums rewards from newly-firing events.
  3. Reports ``done`` if any ``terminal_sufficient`` event fired, **or** every
     ``terminal_required`` event has now fired.

For nethax this needs to be JIT-compatible.  The approach here:

  * Events are built at construction time on the Python side — their type
    and any static arguments (target coordinate, location-tile id, custom
    callable, ...) live in plain Python.
  * The manager's ``compute_reward`` method takes a ``fired_mask`` JAX
    bool array of shape ``[N_EVENTS]`` (one bit per registered event) and
    returns ``(reward_scalar, done_flag, new_fired_mask)``.
  * The caller (env / wrapper) is responsible for threading ``fired_mask``
    across steps.  ``initial_fired_mask()`` builds a fresh all-False array.
  * Each event exposes a JIT-friendly predicate
    ``predicate(prev_state, new_state) -> jnp.bool_`` — pure JAX, no Python
    control flow on traced values.

Events whose predicate cannot be implemented cleanly without further
subsystem work (e.g. eat/kill which need a message-id system; pickup which
needs an inventory-delta diff) register a *stub* event whose predicate is
constantly False.  The event still occupies a slot in ``fired_mask`` so the
indices stay stable — meaning a Wave 5 patch only has to swap the predicate
in.  Stubs are flagged via ``Event.implemented`` for the test suite to skip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.constants.tiles import TileType


# ---------------------------------------------------------------------------
# Static event-type ids (mirrors vendor EventType enum)
# ---------------------------------------------------------------------------
EVT_MESSAGE = 0
EVT_COORD = 1
EVT_LOC = 2
EVT_EAT = 3
EVT_KILL = 4
EVT_PICKUP = 5
EVT_WIELD = 6
EVT_WEAR = 7
EVT_AMULET = 8
EVT_LEVITATE = 9
EVT_POSITIONAL = 10
EVT_CUSTOM = 11


# ---------------------------------------------------------------------------
# Location name → TileType mapping (subset of MiniHack's location vocabulary)
# ---------------------------------------------------------------------------
_LOCATION_TILE = {
    "stairs_down": int(TileType.STAIRCASE_DOWN),
    "stairs_up": int(TileType.STAIRCASE_UP),
    "altar": int(TileType.ALTAR),
    "fountain": int(TileType.FOUNTAIN),
    "throne": int(TileType.THRONE),
    "grave": int(TileType.GRAVE),
    "door": int(TileType.OPEN_DOOR),
    "closed_door": int(TileType.CLOSED_DOOR),
    "lava": int(TileType.LAVA),
    "water": int(TileType.WATER),
    "sink": int(TileType.FOUNTAIN),  # closest analogue; Wave 5: add SINK tile
}


# ---------------------------------------------------------------------------
# Event dataclass — Python-side registry record
# ---------------------------------------------------------------------------
@dataclass
class Event:
    """A single registered event.

    ``predicate`` is a pure JAX-compatible callable
    ``(prev: EnvState, new: EnvState) -> jnp.bool_``.

    ``implemented`` distinguishes real predicates from Wave-5 stubs.  Stubs
    register so indices stay stable across releases but always return False.
    """
    event_type: int
    name: str
    predicate: Callable[[EnvState, EnvState], jnp.ndarray]
    reward: float = 1.0
    repeatable: bool = False
    terminal_required: bool = True
    terminal_sufficient: bool = False
    implemented: bool = True


# ---------------------------------------------------------------------------
# Tile-under-player helper (JIT-safe)
# ---------------------------------------------------------------------------
def _tile_under_player(state: EnvState) -> jnp.ndarray:
    """Return the TileType integer at the player's current position."""
    branch = state.dungeon.current_branch.astype(jnp.int32)
    level = (state.dungeon.current_level.astype(jnp.int32) - jnp.int32(1))
    row = state.player_pos[0].astype(jnp.int32)
    col = state.player_pos[1].astype(jnp.int32)
    # Clamp to valid shape range so traced indexing is safe.
    b, l, h, w = state.terrain.shape
    branch_s = jnp.clip(branch, 0, b - 1)
    level_s = jnp.clip(level, 0, l - 1)
    row_s = jnp.clip(row, 0, h - 1)
    col_s = jnp.clip(col, 0, w - 1)
    return state.terrain[branch_s, level_s, row_s, col_s].astype(jnp.int32)


# ---------------------------------------------------------------------------
# Predicate factories
# ---------------------------------------------------------------------------
def _make_coord_predicate(x: int, y: int) -> Callable:
    """Player has reached (x, y) where (x, y) follows MiniHack's
    (col, row) convention but nethax stores ``player_pos`` as (row, col).
    """
    target_row = jnp.int32(y)
    target_col = jnp.int32(x)

    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        return (
            (new.player_pos[0].astype(jnp.int32) == target_row)
            & (new.player_pos[1].astype(jnp.int32) == target_col)
        )

    return predicate


def _make_positional_predicate(pos: Tuple[int, int]) -> Callable:
    """Same as coord but takes a ``(row, col)`` pair directly.

    Mirrors MiniHack's ``add_positional_event`` semantics (a "place" the
    player must be standing on).  In nethax we collapse the LocActionEvent
    of vendor MiniHack down to "stand on this tile" because the Y_cmd
    confirm-step doesn't apply outside the descend-stairs cycle.
    """
    target_row = jnp.int32(pos[0])
    target_col = jnp.int32(pos[1])

    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        return (
            (new.player_pos[0].astype(jnp.int32) == target_row)
            & (new.player_pos[1].astype(jnp.int32) == target_col)
        )

    return predicate


def _make_location_predicate(location: str) -> Callable:
    """Player stands on a tile of the named feature type."""
    tile_id = _LOCATION_TILE.get(location.lower(), -1)
    target = jnp.int32(tile_id)

    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        return _tile_under_player(new) == target

    return predicate


def _make_custom_predicate(fn: Callable[[EnvState, EnvState], Any]) -> Callable:
    """Wrap a user-provided callable so its result becomes a bool array.

    Custom callables return a *reward delta*; the predicate fires when the
    delta is non-zero.  The actual scalar value is read separately in
    ``compute_reward`` via the same callable.
    """
    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        value = jnp.asarray(fn(prev, new))
        return value != 0

    return predicate


def _always_false_predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
    """Stub predicate for events whose subsystem support lands in Wave 5."""
    return jnp.bool_(False)


# ---------------------------------------------------------------------------
# RewardManager
# ---------------------------------------------------------------------------
class RewardManager:
    """JIT-friendly port of MiniHack's ``RewardManager``.

    Usage::

        rm = RewardManager()
        rm.add_coordinate_event(5, 5, reward=1.0, terminal_sufficient=True)
        fired = rm.initial_fired_mask()
        ...
        reward, done, fired = rm.compute_reward(prev_state, new_state, fired)
    """

    def __init__(self) -> None:
        self._events: List[Event] = []
        # Custom callables are also kept here so compute_reward can read their
        # raw delta values (they may return rewards that aren't all 1.0).
        self._custom_fns: List[Optional[Callable]] = []

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def initial_fired_mask(self) -> jnp.ndarray:
        """Return a fresh ``[N_EVENTS]`` bool array of all False."""
        n = max(len(self._events), 1)
        return jnp.zeros((n,), dtype=jnp.bool_)

    # ------------------------------------------------------------------
    # Generic registration helper
    # ------------------------------------------------------------------
    def _add(
        self,
        event_type: int,
        name: str,
        predicate: Callable,
        reward: float,
        repeatable: bool,
        terminal_required: bool,
        terminal_sufficient: bool,
        implemented: bool = True,
        custom_fn: Optional[Callable] = None,
    ) -> "RewardManager":
        self._events.append(
            Event(
                event_type=event_type,
                name=name,
                predicate=predicate,
                reward=float(reward),
                repeatable=bool(repeatable),
                terminal_required=bool(terminal_required),
                terminal_sufficient=bool(terminal_sufficient),
                implemented=bool(implemented),
            )
        )
        self._custom_fns.append(custom_fn)
        return self

    # ------------------------------------------------------------------
    # Event factories — match vendor MiniHack public API
    # ------------------------------------------------------------------
    def add_coordinate_event(
        self,
        x: int,
        y: int,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        """Fire when player reaches blstats coordinate ``(x, y)``."""
        return self._add(
            EVT_COORD,
            f"coord({x},{y})",
            _make_coord_predicate(x, y),
            reward, repeatable, terminal_required, terminal_sufficient,
        )

    def add_positional_event(
        self,
        pos: Tuple[int, int],
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        """Fire when player is standing at ``(row, col)``."""
        return self._add(
            EVT_POSITIONAL,
            f"pos({pos[0]},{pos[1]})",
            _make_positional_predicate(pos),
            reward, repeatable, terminal_required, terminal_sufficient,
        )

    def add_location_event(
        self,
        location: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        """Fire when player stands on the named feature tile.

        ``location`` is one of: stairs_down, stairs_up, altar, fountain,
        throne, grave, door, closed_door, lava, water, sink.
        """
        return self._add(
            EVT_LOC,
            f"loc({location})",
            _make_location_predicate(location),
            reward, repeatable, terminal_required, terminal_sufficient,
        )

    def add_custom_reward_fn(
        self,
        fn: Callable[[EnvState, EnvState], float],
        *,
        reward: float = 1.0,
        repeatable: bool = True,
        terminal_required: bool = False,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        """Register a user-defined reward function.

        ``fn(prev_state, new_state)`` returns a JAX-scalar reward delta.
        The event "fires" whenever the delta is non-zero.  Unlike other
        events, the scalar value of the delta is paid as-is (rather than
        the ``reward`` constant), so users can implement dense shaping.
        Default ``repeatable=True`` matches typical shaping use.
        """
        return self._add(
            EVT_CUSTOM,
            f"custom({getattr(fn, '__name__', 'fn')})",
            _make_custom_predicate(fn),
            reward, repeatable, terminal_required, terminal_sufficient,
            custom_fn=fn,
        )

    # ---- Wave-5 stubs: register-but-predicate-always-false ----
    # These need message-id wiring and richer subsystem hooks before they
    # can fire.  They register here so the public API matches vendor and
    # downstream code paths don't break.
    def add_eat_event(
        self,
        name: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): wire to MessageId.EAT_FOOD + food-type tag.
        return self._add(
            EVT_EAT, f"eat({name})",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_kill_event(
        self,
        name: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): wire to combat.last_killed_monster_idx + name match.
        return self._add(
            EVT_KILL, f"kill({name})",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_message_event(
        self,
        messages: List[str],
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): wire to MessageId enum + substring match on rendered text.
        return self._add(
            EVT_MESSAGE, f"msg({messages!r})",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_pickup_event(
        self,
        name: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): diff inventory.items between prev/new.
        return self._add(
            EVT_PICKUP, f"pickup({name})",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_wield_event(
        self,
        name: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): compare new.inventory.wielded type_id to ``name``.
        return self._add(
            EVT_WIELD, f"wield({name})",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_wear_event(
        self,
        name: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): check inventory.worn_armor diff.
        return self._add(
            EVT_WEAR, f"wear({name})",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_amulet_event(
        self,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): check inventory.worn_amulet transition -1 -> >=0.
        return self._add(
            EVT_AMULET, "amulet",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    def add_levitate_event(
        self,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # TODO(Wave 5): add status.levitating field & test transition.
        return self._add(
            EVT_LEVITATE, "levitate",
            _always_false_predicate,
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=False,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def compute_reward(
        self,
        prev_state: EnvState,
        new_state: EnvState,
        fired_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Evaluate all events.

        Parameters
        ----------
        prev_state : EnvState
            State immediately before the most recent ``env.step``.
        new_state : EnvState
            State immediately after the step.
        fired_mask : jnp.ndarray, shape ``[N_EVENTS]``, dtype bool
            Whether each event has already fired this episode.

        Returns
        -------
        reward : jnp.float32 scalar — sum of rewards for newly-firing events.
        done   : jnp.bool_ scalar — whether the episode should terminate.
        new_fired_mask : jnp.ndarray, shape ``[N_EVENTS]``, dtype bool
            Updated mask reflecting events that fired this step (with
            ``repeatable`` events left as False so they can fire again).
        """
        if not self._events:
            return (
                jnp.float32(0.0),
                jnp.bool_(False),
                fired_mask,
            )

        n = len(self._events)

        # Evaluate every predicate.  These are pure jax functions so the
        # whole loop is unrolled at trace time; n is a Python int.
        fired_now = jnp.stack(
            [ev.predicate(prev_state, new_state) for ev in self._events]
        )  # [N] bool

        repeatable = jnp.array(
            [ev.repeatable for ev in self._events], dtype=jnp.bool_,
        )
        rewards = jnp.array(
            [ev.reward for ev in self._events], dtype=jnp.float32,
        )
        term_required = jnp.array(
            [ev.terminal_required for ev in self._events], dtype=jnp.bool_,
        )
        term_sufficient = jnp.array(
            [ev.terminal_sufficient for ev in self._events], dtype=jnp.bool_,
        )

        # An event "newly fires" if its predicate is True AND it has not
        # already fired (or it's repeatable).
        can_fire = fired_now & (~fired_mask | repeatable)

        # Reward = sum over events.  For custom events we want the raw delta
        # value rather than the static reward constant — handle these in a
        # second pass.
        base_pay = jnp.where(can_fire, rewards, jnp.float32(0.0))

        # Custom events: replace base_pay slot with raw delta value.
        for i, ev in enumerate(self._events):
            if ev.event_type == EVT_CUSTOM and self._custom_fns[i] is not None:
                fn = self._custom_fns[i]
                delta = jnp.asarray(fn(prev_state, new_state)).astype(jnp.float32)
                # Only paid when can_fire[i] is True.
                base_pay = base_pay.at[i].set(
                    jnp.where(can_fire[i], delta, jnp.float32(0.0))
                )

        reward = jnp.sum(base_pay).astype(jnp.float32)

        # Update fired_mask: any non-repeatable event that just fired stays
        # latched True.  Repeatable events leave the mask unchanged.
        new_fired_mask = jnp.where(
            can_fire & ~repeatable,
            jnp.bool_(True),
            fired_mask,
        )

        # Termination logic.  Match vendor _check_complete():
        #   * If ANY terminal_sufficient event has fired -> done.
        #   * Else if EVERY terminal_required event has fired -> done.
        any_sufficient = jnp.any(term_sufficient & new_fired_mask)

        # "Every required event has fired" — vacuously true if there are no
        # required events at all (mirrors vendor behaviour where _result
        # starts True).  But we must avoid the degenerate case where a brand
        # new manager with no events trivially reports done=True; the
        # n==0 short-circuit at the top already handles that.
        required_satisfied = jnp.all(~term_required | new_fired_mask)

        # Vendor also requires that at least one required event exists for
        # ``required_satisfied`` to count as a real termination — otherwise
        # an episode would end immediately on construction.  Encode that:
        has_any_required = jnp.array(
            any(ev.terminal_required for ev in self._events),
            dtype=jnp.bool_,
        )
        done = any_sufficient | (has_any_required & required_satisfied)

        return reward, done, new_fired_mask

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> jnp.ndarray:
        """Return a fresh fired_mask (alias for ``initial_fired_mask``)."""
        return self.initial_fired_mask()
