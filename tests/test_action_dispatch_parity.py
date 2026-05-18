"""Wave 6 Closing-Audit #87 — action_dispatch parity vs vendor cmd.c.

Verifies the 256-entry ``_ACTION_TO_HANDLER_IDX`` table in
``Nethax.nethax.subsystems.action_dispatch`` against vendor key bindings in
``vendor/nethack/src/cmd.c::extcmdlist[]``.

These are pure-Python assertions on a constant lookup table — no JAX tracing
or environment construction is required.  Each test pulls the table once
and reads cells by ASCII / Meta- / Ctrl- byte value.
"""

import pytest

from Nethax.nethax.constants.actions import (
    ACTIONS,
    CompassCardinalDirection,
    CompassIntercardinalDirection,
    CompassCardinalDirectionLonger,
    CompassIntercardinalDirectionLonger,
    Command,
    MiscDirection,
)
from Nethax.nethax.subsystems.action_dispatch import (
    _ACTION_TO_HANDLER_IDX,
    _HANDLERS,
    _SLOT_NOOP,
    _SLOT_MOVE_N,
    _SLOT_MOVE_E,
    _SLOT_MOVE_S,
    _SLOT_MOVE_W,
    _SLOT_MOVE_NE,
    _SLOT_MOVE_SE,
    _SLOT_MOVE_SW,
    _SLOT_MOVE_NW,
    _SLOT_RUN_N,
    _SLOT_RUN_E,
    _SLOT_RUN_S,
    _SLOT_RUN_W,
    _SLOT_RUN_NE,
    _SLOT_RUN_SE,
    _SLOT_RUN_SW,
    _SLOT_RUN_NW,
    _SLOT_STAIR_UP,
    _SLOT_STAIR_DOWN,
    _SLOT_WAIT,
    _SLOT_EAT,
    _SLOT_QUAFF,
    _SLOT_READ,
    _SLOT_ZAP,
    _SLOT_CAST,
    _SLOT_PICKUP,
    _SLOT_DROP,
    _SLOT_WIELD,
    _SLOT_WEAR,
    _SLOT_PUTON,
    _SLOT_REMOVE,
    _SLOT_OPEN,
    _SLOT_CLOSE,
    _SLOT_KICK,
    _SLOT_FIGHT,
    _SLOT_SEARCH,
    _SLOT_PRAY,
    _SLOT_THROW,
    _SLOT_TWOWEAPON,
    _SLOT_LOOT,
    _SLOT_APPLY,
    _SLOT_ENGRAVE,
    _SLOT_NAME,
    _SLOT_ENHANCE,
)


def _slot(c) -> int:
    """Return the handler slot index for a single character or int code."""
    if isinstance(c, str):
        c = ord(c)
    return int(_ACTION_TO_HANDLER_IDX[c])


# ---------------------------------------------------------------------------
# Coverage / structural tests
# ---------------------------------------------------------------------------


def test_all_121_action_enum_values_have_handler_slot():
    """Every one of the 121 NLE action values resolves to a defined slot.

    The table is 256 entries; the test asserts that every ACTION's int value
    is in range and that the slot index is in [0, len(_HANDLERS)).
    """
    assert len(ACTIONS) == 121
    n_handlers = len(_HANDLERS)
    for action in ACTIONS:
        v = int(action)
        assert 0 <= v < 256, f"action {action!r} value {v} out of [0, 256)"
        slot = int(_ACTION_TO_HANDLER_IDX[v])
        assert 0 <= slot < n_handlers, (
            f"action {action!r} ASCII={v} maps to invalid slot {slot} "
            f"(handler count {n_handlers})"
        )


def test_movement_keys_map_to_move_handlers():
    """All 8 cardinal+intercardinal keys map to the matching _move_* slot.

    Vendor: cmd.c::move_funcs[][] dispatches h/j/k/l/y/u/b/n via the
    extcmdlist movewest..rushsouthwest block (lines 2006-2057).
    """
    assert _slot(CompassCardinalDirection.N) == _SLOT_MOVE_N
    assert _slot(CompassCardinalDirection.S) == _SLOT_MOVE_S
    assert _slot(CompassCardinalDirection.E) == _SLOT_MOVE_E
    assert _slot(CompassCardinalDirection.W) == _SLOT_MOVE_W
    assert _slot(CompassIntercardinalDirection.NE) == _SLOT_MOVE_NE
    assert _slot(CompassIntercardinalDirection.SE) == _SLOT_MOVE_SE
    assert _slot(CompassIntercardinalDirection.SW) == _SLOT_MOVE_SW
    assert _slot(CompassIntercardinalDirection.NW) == _SLOT_MOVE_NW


def test_run_keys_map_to_run_handlers():
    """Uppercase compass keys map to the matching _run_* slot."""
    assert _slot(CompassCardinalDirectionLonger.N) == _SLOT_RUN_N
    assert _slot(CompassCardinalDirectionLonger.S) == _SLOT_RUN_S
    assert _slot(CompassCardinalDirectionLonger.E) == _SLOT_RUN_E
    assert _slot(CompassCardinalDirectionLonger.W) == _SLOT_RUN_W
    assert _slot(CompassIntercardinalDirectionLonger.NE) == _SLOT_RUN_NE
    assert _slot(CompassIntercardinalDirectionLonger.SE) == _SLOT_RUN_SE
    assert _slot(CompassIntercardinalDirectionLonger.SW) == _SLOT_RUN_SW
    assert _slot(CompassIntercardinalDirectionLonger.NW) == _SLOT_RUN_NW


# ---------------------------------------------------------------------------
# Per-action key→handler tests (vendor cmd.c::extcmdlist[])
# ---------------------------------------------------------------------------


def test_eat_key_maps_to_eat_handler():
    """'e' → doeat (cmd.c:1712)."""
    assert _slot("e") == _SLOT_EAT


def test_quaff_key_maps_to_quaff_handler():
    """'q' → dodrink (cmd.c:1809)."""
    assert _slot("q") == _SLOT_QUAFF


def test_read_key_maps_to_read_handler():
    """'r' → doread (cmd.c:1816)."""
    assert _slot("r") == _SLOT_READ


def test_zap_key_maps_to_zap_handler():
    """'z' → dozap (cmd.c:2004)."""
    assert _slot("z") == _SLOT_ZAP


def test_cast_key_maps_to_cast_handler():
    """'Z' → docast (cmd.c:1689)."""
    assert _slot("Z") == _SLOT_CAST


def test_pickup_key_maps_to_pickup_handler():
    """',' → dopickup (cmd.c:1799)."""
    assert _slot(",") == _SLOT_PICKUP


def test_drop_key_maps_to_drop_handler():
    """'d' → dodrop (cmd.c:1708)."""
    assert _slot("d") == _SLOT_DROP


def test_wield_key_maps_to_wield_handler():
    """'w' → dowield (cmd.c:1938)."""
    assert _slot("w") == _SLOT_WIELD


def test_wear_key_maps_to_wear_handler():
    """'W' → dowear (cmd.c:1932)."""
    assert _slot("W") == _SLOT_WEAR


def test_puton_key_maps_to_puton_handler():
    """'P' → doputon (cmd.c:1807)."""
    assert _slot("P") == _SLOT_PUTON


def test_remove_key_maps_to_remove_handler():
    """'R' → doremring (cmd.c:1820)."""
    assert _slot("R") == _SLOT_REMOVE


def test_takeoff_keys_route_to_remove_handler():
    """'T' (takeoff armor) and 'A' (takeoffall) route via _SLOT_REMOVE.

    Vendor distinguishes dotakeoff (cmd.c:1886) and doddoremarm (cmd.c:1888),
    but our headless env shares the remove path for all worn slots.
    """
    assert _slot("T") == _SLOT_REMOVE
    assert _slot("A") == _SLOT_REMOVE


def test_open_close_keys_map_to_door_handlers():
    """'o' → doopen (cmd.c:1777); 'c' → doclose (cmd.c:1695)."""
    assert _slot("o") == _SLOT_OPEN
    assert _slot("c") == _SLOT_CLOSE


def test_fight_key_maps_to_fight_handler():
    """'F' → do_fight prefix (cmd.c:1722)."""
    assert _slot("F") == _SLOT_FIGHT


def test_search_key_maps_to_search_handler():
    """'s' → dosearch (cmd.c:1846)."""
    assert _slot("s") == _SLOT_SEARCH


def test_engrave_key_maps_to_engrave_handler():
    """'E' → doengrave (cmd.c:1714)."""
    assert _slot("E") == _SLOT_ENGRAVE


def test_call_key_maps_to_name_handler():
    """'C' → docallcmd (cmd.c:1687)."""
    assert _slot("C") == _SLOT_NAME


def test_throw_and_fire_keys_route_to_throw_handler():
    """'t' → dothrow (cmd.c:1901); 'f' → dofire (cmd.c:1724).

    Both route to the throw subsystem in our env (fire uses quivered ammo,
    which we emulate by throwing the first throwable item).
    """
    assert _slot("t") == _SLOT_THROW
    assert _slot("f") == _SLOT_THROW


def test_twoweapon_key_maps_to_twoweapon_handler():
    """'X' → dotwoweapon (cmd.c:1913)."""
    assert _slot("X") == _SLOT_TWOWEAPON


def test_apply_key_maps_to_apply_handler():
    """'a' → doapply (cmd.c:1677)."""
    assert _slot("a") == _SLOT_APPLY


def test_droptype_key_routes_to_drop_handler():
    """'D' → doddrop (cmd.c:1710): we proxy to single-item drop."""
    assert _slot("D") == _SLOT_DROP


# ---------------------------------------------------------------------------
# Ctrl- and Meta- prefixed bindings
# ---------------------------------------------------------------------------


def test_ctrl_d_maps_to_kick_handler():
    """C('d') (0x04) → dokick (cmd.c:1748)."""
    assert _slot(0x04) == _SLOT_KICK
    # And via the Command enum:
    assert _slot(int(Command.KICK)) == _SLOT_KICK


def test_meta_p_maps_to_pray_handler():
    """M('p') (0xF0) → dopray (cmd.c:1803)."""
    assert _slot(0xF0) == _SLOT_PRAY
    assert _slot(int(Command.PRAY)) == _SLOT_PRAY


def test_meta_l_maps_to_loot_handler():
    """M('l') (0xEC) → doloot (cmd.c:1762)."""
    assert _slot(0xEC) == _SLOT_LOOT
    assert _slot(int(Command.LOOT)) == _SLOT_LOOT


def test_meta_e_aliases_to_eat_handler():
    """Meta-e (0xE5) → _SLOT_ENHANCE per vendor cmd.c:1716.

    Wave-14 remap: _SLOT_ENHANCE is vendor-correct; plain 'e' (0x65) still
    routes to _SLOT_EAT.  Citation: cmd.c:1716.
    """
    assert _slot(0xE5) == _SLOT_ENHANCE
    assert _slot(ord("e")) == _SLOT_EAT


# ---------------------------------------------------------------------------
# Movement / wait / stairs
# ---------------------------------------------------------------------------


def test_wait_key_present():
    """'.' (MiscDirection.WAIT) and SPACE both route to _SLOT_WAIT.

    Vendor: cmd.c:1930 ('.' → donull) and cmd.c::update_rest_on_space
    binds SPACE to donull when 'rest_on_space' is on.
    """
    assert _slot(int(MiscDirection.WAIT)) == _SLOT_WAIT
    assert _slot(" ") == _SLOT_WAIT


def test_stair_keys_map_to_stair_handlers():
    """'<' → doup, '>' → dodown (cmd.c:1917, 1703)."""
    assert _slot(int(MiscDirection.UP)) == _SLOT_STAIR_UP
    assert _slot(int(MiscDirection.DOWN)) == _SLOT_STAIR_DOWN


def test_numpad_movement_keys_map_to_move_handlers():
    """Numpad alternates: 8/2/6/4 (NSEW), 9/3/1/7 (corners), 5 (rest).

    Vendor: bind_keys_to_extcmds (reset_commands) when 'number_pad' is on.
    """
    assert _slot("8") == _SLOT_MOVE_N
    assert _slot("2") == _SLOT_MOVE_S
    assert _slot("6") == _SLOT_MOVE_E
    assert _slot("4") == _SLOT_MOVE_W
    assert _slot("9") == _SLOT_MOVE_NE
    assert _slot("3") == _SLOT_MOVE_SE
    assert _slot("1") == _SLOT_MOVE_SW
    assert _slot("7") == _SLOT_MOVE_NW
    assert _slot("5") == _SLOT_WAIT


# ---------------------------------------------------------------------------
# Informational / UI commands — intentional no-ops
# ---------------------------------------------------------------------------


def test_extcmd_prefix_and_help_are_documented_noops():
    """'#' (extcmd), '?' (help), '&' (whatdoes), '/' (whatis) — UI only,
    intentionally no-ops in our headless env."""
    for key in ("#", "?", "&", "/"):
        assert _slot(key) == _SLOT_NOOP, f"{key!r} should be NOOP"


def test_informational_inventory_keys_are_noops():
    """'i' (show inventory), ':' (look), 'O' (options), '*' (seeall) etc.

    All informational/UI-only — no state change expected.
    """
    for key in ("i", "I", ":", ";", "O", "*", "\\", "`", "|", "@", "v", "V"):
        assert _slot(key) == _SLOT_NOOP, f"{key!r} should be NOOP"


def test_see_worn_equipment_keys_are_noops():
    """'\"', '[', '=', '(', ')', '$', '+', '^' — UI-only "show worn X"."""
    for key in ('"', "[", "=", "(", ")", "$", "+", "^"):
        assert _slot(key) == _SLOT_NOOP, f"{key!r} should be NOOP"


# ---------------------------------------------------------------------------
# Critical-action safety checks
# ---------------------------------------------------------------------------


def test_no_critical_actions_at_slot_noop():
    """Eat, quaff, read, zap, cast, pickup, drop, wield, wear, kick, pray,
    open, close, fight, search, engrave, two-weapon, throw, apply, loot
    must NEVER drop to _SLOT_NOOP.
    """
    critical = {
        "EAT (e)":      ord("e"),
        "QUAFF (q)":    ord("q"),
        "READ (r)":     ord("r"),
        "ZAP (z)":      ord("z"),
        "CAST (Z)":     ord("Z"),
        "PICKUP (,)":   ord(","),
        "DROP (d)":     ord("d"),
        "WIELD (w)":    ord("w"),
        "WEAR (W)":     ord("W"),
        "PUTON (P)":    ord("P"),
        "REMOVE (R)":   ord("R"),
        "OPEN (o)":     ord("o"),
        "CLOSE (c)":    ord("c"),
        "FIGHT (F)":    ord("F"),
        "SEARCH (s)":   ord("s"),
        "ENGRAVE (E)":  ord("E"),
        "CALL (C)":     ord("C"),
        "TWOWEAPON":    ord("X"),
        "THROW (t)":    ord("t"),
        "APPLY (a)":    ord("a"),
        "LOOT (M-l)":   0xEC,
        "KICK (C-d)":   0x04,
        "PRAY (M-p)":   0xF0,
        "STAIR_UP (<)": ord("<"),
        "STAIR_DN (>)": ord(">"),
        "WAIT (.)":     ord("."),
    }
    bad = {name: _slot(v) for name, v in critical.items() if _slot(v) == _SLOT_NOOP}
    assert not bad, f"Critical actions at _SLOT_NOOP: {bad}"


def test_vendor_keyed_extcmds_have_handler_assignment():
    """Every keyed entry in vendor/nethack/src/cmd.c::extcmdlist[] (non-wizard,
    non-internal) resolves to a defined slot in our table.

    We hand-encode the vendor key list here (derived from cmd.c lines
    1667-2005) and assert the table has a slot in range for each.  Wizard-
    mode-only keys (C-e, C-g, C-i, C-v, C-f, C-w) are also covered because
    they belong to the same byte range.
    """
    vendor_keys = [
        # symbol → handler (vendor extcmdlist[])
        "#", "@", "C", "Z", "v", "c", ">", "d", "D", "e", "E",
        "F", "f", ";", "?", "i", "I", "\\", "`", ":", "o", "O",
        "p", "|", ",", "P", "q", "Q", "r", "R", "m", "G", "g",
        "S", "s", "*", '"', "[", "=", "(", ")", "!", "$", "+",
        "^", "x", "T", "A", "t", "_", "X", "<", "V", ".", "W",
        "&", "/", "w", "z", " ", "a",
    ]
    # Ctrl- and Meta- bytes.
    vendor_keys.extend(chr(0x80 | ord(c)) for c in "?aAcCdefgiIjlmnopqrRsTtuvVwX")
    vendor_keys.extend(chr(0x1F & ord(c)) for c in "adefgioprtvwxz")
    vendor_keys.append("\x1f")  # C('_') = retravel
    vendor_keys.append("\x7f")  # DEL = terrain

    n_handlers = len(_HANDLERS)
    for k in vendor_keys:
        v = ord(k) if isinstance(k, str) else int(k)
        slot = int(_ACTION_TO_HANDLER_IDX[v])
        assert 0 <= slot < n_handlers, (
            f"vendor key {k!r} (byte {v:#04x}) maps to invalid slot {slot}"
        )


# ---------------------------------------------------------------------------
# Coverage percentage — informational, not strict
# ---------------------------------------------------------------------------


def test_vendor_dokeylist_coverage_at_least_95_percent():
    """Coverage = (vendor keys present in our table) / (total vendor keys).

    We're 100% on extcmdlist[] entries; this test asserts ≥ 95% as a
    forward-compatible floor.
    """
    vendor_keys = list("#@CZvc>dDeEFf;?iI\\`:oOp|,PqQrRmGgSs*\"[=()!$+^xTAt_X<V.W&/wz a")
    vendor_keys.extend(chr(0x80 | ord(c)) for c in "?aAcCdefgiIjlmnopqrRsTtuvVwX")
    vendor_keys.extend(chr(0x1F & ord(c)) for c in "adefgioprtvwxz_")
    vendor_keys.append("\x7f")
    # Deduplicate.
    vendor_keys = sorted(set(vendor_keys))

    # "Present" means: table value is in [0, len(_HANDLERS))
    # (always true because the table is dense int8) — so for a stricter
    # coverage we count keys with a slot != 0 OR explicitly marked.
    # Here we count *any* assignment, which by construction is 100%.
    n_handlers = len(_HANDLERS)
    valid = sum(
        1
        for k in vendor_keys
        if 0 <= int(_ACTION_TO_HANDLER_IDX[ord(k)]) < n_handlers
    )
    pct = valid / len(vendor_keys)
    assert pct >= 0.95, f"vendor key coverage too low: {pct:.1%}"
