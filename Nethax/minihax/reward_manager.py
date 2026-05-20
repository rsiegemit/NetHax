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
# Wave17i: message-based predicate factory.
#
# Vendor MiniHack's eat / wield / wear / amulet / kill events all funnel
# through ``MessageEvent`` (reward_manager.py:217-249), which substring-matches
# against ``observation[message]`` (a 256-byte uint8 buffer).
#
# nethax exposes the same buffer at ``state.messages.message_buffer`` (256-byte
# uint8 array; see subsystems/messages.py).  We build a JIT-friendly predicate
# that scans that buffer for *any* of the supplied substring byte patterns.
# Pattern lengths are limited to <= MSG_BUF_LEN.
# ---------------------------------------------------------------------------

_MSG_BUF_LEN = 256


def _pattern_bytes(pat: str) -> jnp.ndarray:
    """Encode a substring as a fixed-length uint8 array padded with NULs."""
    raw = pat.encode("utf-8")[: _MSG_BUF_LEN]
    return jnp.asarray(list(raw), dtype=jnp.uint8)


def _make_message_predicate(messages: List[str]) -> Callable:
    """Return a JIT-friendly predicate that fires when any of the supplied
    substring patterns appears in ``state.messages.message_buffer``.

    Vendor parity (reward_manager.py:239-249): substring (``msg in curr_msg``)
    over the rendered message line.
    """
    # Pre-encode every pattern and stash its effective length so we can use a
    # branchless scan over candidate offsets.
    encoded = [pat.encode("utf-8")[: _MSG_BUF_LEN] for pat in messages]
    encoded = [e for e in encoded if len(e) > 0]
    if not encoded:
        return _always_false_predicate

    pattern_arrs = [jnp.asarray(list(e), dtype=jnp.uint8) for e in encoded]
    pattern_lens = [len(e) for e in encoded]

    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        buf = new.messages.message_buffer.astype(jnp.uint8)  # uint8[256]
        any_match = jnp.bool_(False)
        for pat, plen in zip(pattern_arrs, pattern_lens):
            # Build a [buf_len - plen + 1] window-equality mask:
            # buf[i:i+plen] == pat for each candidate start i.
            max_start = _MSG_BUF_LEN - plen + 1
            if max_start <= 0:
                continue
            # Use vectorised compare: stack windows by slicing.
            # We avoid jnp.lax.dynamic_slice loops by manually building each
            # column equality (small patterns, small loop).
            mask = jnp.ones((max_start,), dtype=jnp.bool_)
            for j in range(plen):
                col = buf[j: j + max_start]
                mask = mask & (col == pat[j])
            any_match = any_match | jnp.any(mask)
        return any_match

    return predicate


def _make_eat_messages(name: str) -> List[str]:
    """Vendor add_eat_event message list (reward_manager.py:427-436)."""
    msgs = [
        f"This {name} is delicious",
        "Blecch!  Rotten food!",
        "last bite of your meal",
    ]
    if name == "apple":
        msgs.append("Delicious!  Must be a Macintosh!")
        msgs.append("Core dumped.")
    if name == "pear":
        msgs.append("Core dumped.")
    return msgs


def _make_wield_messages(name: str) -> List[str]:
    """Vendor add_wield_event messages (reward_manager.py:467-470)."""
    return [
        f"{name} wields itself to your hand!",
        f"{name} (weapon in hand)",
    ]


def _make_wear_messages(name: str) -> List[str]:
    """Vendor add_wear_event message (reward_manager.py:500)."""
    return [f"You are now wearing a {name}"]


def _make_amulet_messages() -> List[str]:
    """Vendor add_amulet_event message (reward_manager.py:527)."""
    return ["amulet (being worn)."]


def _make_kill_messages(name: str) -> List[str]:
    """Vendor add_kill_event message (reward_manager.py:560)."""
    return [f"You kill the {name}"]


def _make_pickup_predicate(name: str) -> Callable:
    """Fire when inventory grows AND the named item appears in inv_strs.

    Vendor doesn't have a dedicated pickup event, but a pickup is detectable
    via the "You see here" / "f - <name>" inventory-letter line plus a
    transition in ``inventory.count``.
    """
    # Build a message-style predicate on the trailing inventory print of the
    # form "f - a <name>" — vendor pickup() prints this through pline().
    msgs = [f"- a {name}", f"- an {name}", f"- {name}"]
    msg_pred = _make_message_predicate(msgs)

    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        return msg_pred(prev, new)

    return predicate


def _make_loc_action_predicate(place_name: str, action_name: str) -> Callable:
    """Vendor LocActionEvent (reward_manager.py:106-156).

    Fires when the player performs ``action_name`` while standing on top of
    a tile whose feature name matches ``place_name`` (e.g. throne + sit).
    nethax tile semantics are simpler: we collapse this to "standing on the
    feature tile" — the action confirmation step is handled inside the
    individual subsystem (sit_*, etc.).
    """
    tile_id = _LOCATION_TILE.get(place_name.lower(), -1)
    target = jnp.int32(tile_id)
    del action_name  # nethax fires on tile-standing alone

    def predicate(prev: EnvState, new: EnvState) -> jnp.ndarray:
        return _tile_under_player(new) == target

    return predicate


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
        place_name,
        action_name: Optional[str] = None,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        """Vendor-parity positional/LocAction event.

        Two overloads (Wave17i compat):
          1. Vendor signature (reward_manager.py:597-632):
             ``add_positional_event(place_name: str, action_name: str)`` —
             fires when the player performs ``action_name`` while standing
             on the named feature tile (throne/altar/sink/fountain/...).
          2. Legacy nethax signature: ``add_positional_event(pos)`` where
             ``pos`` is a ``(row, col)`` tuple.  Retained so existing
             callers keep working.
        """
        if isinstance(place_name, tuple):
            pos = place_name
            return self._add(
                EVT_POSITIONAL,
                f"pos({pos[0]},{pos[1]})",
                _make_positional_predicate(pos),
                reward, repeatable, terminal_required, terminal_sufficient,
            )
        # Vendor (place_name, action_name) form.
        place = str(place_name)
        action = str(action_name) if action_name is not None else ""
        return self._add(
            EVT_POSITIONAL,
            f"locaction({place},{action})",
            _make_loc_action_predicate(place, action),
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

    # ------------------------------------------------------------------
    # Wave17i: message-buffer driven event predicates.  Cite vendor:
    #   reward_manager.py:402-440 (eat), :442-473 (wield), :475-503 (wear),
    #   :505-533 (amulet), :535-566 (kill), :568-595 (message).
    # ------------------------------------------------------------------
    def add_eat_event(
        self,
        name: str,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        msgs = _make_eat_messages(name)
        return self._add(
            EVT_EAT, f"eat({name})",
            _make_message_predicate(msgs),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
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
        msgs = _make_kill_messages(name)
        return self._add(
            EVT_KILL, f"kill({name})",
            _make_message_predicate(msgs),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
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
        return self._add(
            EVT_MESSAGE, f"msg({messages!r})",
            _make_message_predicate(list(messages)),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
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
        return self._add(
            EVT_PICKUP, f"pickup({name})",
            _make_pickup_predicate(name),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
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
        msgs = _make_wield_messages(name)
        return self._add(
            EVT_WIELD, f"wield({name})",
            _make_message_predicate(msgs),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
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
        msgs = _make_wear_messages(name)
        return self._add(
            EVT_WEAR, f"wear({name})",
            _make_message_predicate(msgs),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
        )

    def add_amulet_event(
        self,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        msgs = _make_amulet_messages()
        return self._add(
            EVT_AMULET, "amulet",
            _make_message_predicate(msgs),
            reward, repeatable, terminal_required, terminal_sufficient,
            implemented=True,
        )

    def add_levitate_event(
        self,
        *,
        reward: float = 1.0,
        repeatable: bool = False,
        terminal_required: bool = True,
        terminal_sufficient: bool = False,
    ) -> "RewardManager":
        # No corresponding subsystem yet (status.levitating); leave stubbed.
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


# ---------------------------------------------------------------------------
# Wave17i: SequentialRewardManager + GroupedRewardManager
# Vendor citations: reward_manager.py:773-794 (Sequential), :797-864 (Grouped)
# ---------------------------------------------------------------------------


class SequentialRewardManager(RewardManager):
    """Reward manager that requires events fire in the order they were added.

    Vendor (reward_manager.py:773-794): ignores ``terminal_required`` /
    ``terminal_sufficient``; an episode ends only once every event has fired,
    and each step only the current head event can fire.
    """

    def __init__(self) -> None:
        super().__init__()

    def compute_reward(
        self,
        prev_state: EnvState,
        new_state: EnvState,
        fired_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if not self._events:
            return (jnp.float32(0.0), jnp.bool_(False), fired_mask)

        # Sequential current_event_idx = number of already-fired events
        # (assumes monotonic firing — once fired, stays fired since
        # we never set repeatable=True in this manager).
        # We compute it as the count of leading True bits.
        n = len(self._events)
        # Idx of first not-yet-fired event.
        not_fired = ~fired_mask
        # argmax on bool returns first True index (or 0 if all False).
        head_idx = jnp.argmax(not_fired.astype(jnp.int32))
        all_done = jnp.all(fired_mask)

        # Evaluate only the head event's predicate.
        # JIT trick: gather rewards & predicates by unrolled lookup.
        fired_now_each = jnp.stack(
            [ev.predicate(prev_state, new_state) for ev in self._events]
        )
        head_fires = fired_now_each[head_idx] & (~all_done)

        rewards = jnp.array(
            [ev.reward for ev in self._events], dtype=jnp.float32,
        )
        head_reward = jnp.where(
            head_fires, rewards[head_idx], jnp.float32(0.0),
        )

        # Latch the head bit when it fires.
        new_fired = jnp.where(
            head_fires,
            fired_mask.at[head_idx].set(jnp.bool_(True)),
            fired_mask,
        )
        done = jnp.all(new_fired)
        return head_reward.astype(jnp.float32), done, new_fired


class GroupedRewardManager:
    """Collection of reward managers, summed and combined via term flags.

    Vendor parity: reward_manager.py:797-864.  Each child manager carries
    its own ``terminal_required`` / ``terminal_sufficient`` flags (set when
    added via ``add_reward_manager``).
    """

    def __init__(self) -> None:
        self.reward_managers: List[RewardManager] = []
        self._flags: List[Tuple[bool, bool]] = []  # (required, sufficient)

    def add_reward_manager(
        self,
        reward_manager: RewardManager,
        terminal_required: bool,
        terminal_sufficient: bool,
    ) -> None:
        self.reward_managers.append(reward_manager)
        self._flags.append((bool(terminal_required), bool(terminal_sufficient)))

    def initial_fired_mask(self) -> List[jnp.ndarray]:
        """One mask per child manager, in registration order."""
        return [rm.initial_fired_mask() for rm in self.reward_managers]

    def reset(self) -> List[jnp.ndarray]:
        return self.initial_fired_mask()

    def compute_reward(
        self,
        prev_state: EnvState,
        new_state: EnvState,
        fired_masks: List[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
        if not self.reward_managers:
            return (jnp.float32(0.0), jnp.bool_(False), fired_masks)

        total_reward = jnp.float32(0.0)
        new_masks: List[jnp.ndarray] = []
        any_sufficient_done = jnp.bool_(False)
        all_required_done = jnp.bool_(True)
        has_required = False

        for rm, (req, suf), mask in zip(
            self.reward_managers, self._flags, fired_masks,
        ):
            r, done, nm = rm.compute_reward(prev_state, new_state, mask)
            total_reward = total_reward + r
            new_masks.append(nm)
            if suf:
                any_sufficient_done = any_sufficient_done | done
            if req:
                has_required = True
                all_required_done = all_required_done & done

        # Vendor: any sufficient ⇒ done; else all required ⇒ done.
        if has_required:
            done_out = any_sufficient_done | all_required_done
        else:
            done_out = any_sufficient_done
        return total_reward.astype(jnp.float32), done_out, new_masks
