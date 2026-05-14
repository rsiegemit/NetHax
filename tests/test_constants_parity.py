"""Wave 6 closing-audit #89 — constants/structs parity tests vs vendor.

Verifies that:
  * messages ring buffer matches pline.c (MSGCOUNT=20, BUFSZ=256).
  * EnvState carries the new u.* fields added in this audit (luck, conduct
    coverage etc.).
  * VendorTileType mirrors vendor/nethack/include/rm.h levl_typ_types exactly
    (37 entries, STONE=0, DRAWBRIDGE_UP=19 present).
  * Command enum keystrokes match vendor cmd.c::extcmdlist key column for
    spot-checked entries and the 121-entry NLE total.

Citations are inlined per test.  All imports are lazy so test collection
never fails even when a downstream subsystem cannot import.
"""


# ---------------------------------------------------------------------------
# A. messages.py ring-buffer vs vendor/nethack/src/pline.c
# ---------------------------------------------------------------------------

def test_message_history_ring_buffer_20_entries():
    """pline.c saves MSGCOUNT=20 recent lines in saved_plines[]."""
    from Nethax.nethax.subsystems.messages import HISTORY_LEN, MessageState

    assert HISTORY_LEN == 20
    state = MessageState.default()
    # ring buffer shape is (HISTORY_LEN, MSG_BUF_LEN).
    assert state.message_history.shape[0] == 20


def test_message_max_bufsz_256():
    """hack.h defines BUFSZ=256; each message line is capped at that width."""
    from Nethax.nethax.subsystems.messages import MSG_BUF_LEN, MessageState

    assert MSG_BUF_LEN == 256
    state = MessageState.default()
    assert state.message_buffer.shape[0] == 256
    assert state.message_history.shape[1] == 256


# ---------------------------------------------------------------------------
# B. state.py field coverage vs vendor/nethack/include/you.h::struct you
# ---------------------------------------------------------------------------

def test_state_has_player_hp_uhp_equivalent():
    """u.uhp / u.uhpmax → state.player_hp / state.player_hp_max.

    Cite: vendor/nethack/include/you.h line 476.
    """
    import jax
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert hasattr(state, "player_hp")
    assert hasattr(state, "player_hp_max")
    # Sanity: initialised positive.
    assert int(state.player_hp) > 0
    assert int(state.player_hp_max) > 0


def test_state_has_uconduct_equivalent():
    """u.uconduct → state.conduct (ConductState slice).

    Cite: vendor/nethack/include/you.h line 447.
    """
    import jax
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert hasattr(state, "conduct")
    # ConductState exposes a .violations array.
    assert hasattr(state.conduct, "violations")


def test_state_has_uluck_and_moreluck():
    """u.uluck / u.moreluck → player_luck / player_moreluck.

    Added in Wave 6 closing-audit #89.
    Cite: vendor/nethack/include/you.h line 460.
    """
    import jax
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert hasattr(state, "player_luck")
    assert hasattr(state, "player_moreluck")
    # Default luck is zero on a non-special day per you.h comment.
    assert int(state.player_luck) == 0
    assert int(state.player_moreluck) == 0


def test_state_has_in_water_and_buried():
    """u.uinwater / u.uburied → player_in_water / player_buried bitfields.

    Cite: vendor/nethack/include/you.h lines 431, 436.
    """
    import jax
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert hasattr(state, "player_in_water")
    assert hasattr(state, "player_buried")
    assert bool(state.player_in_water) is False
    assert bool(state.player_buried) is False


def test_state_has_mortality_counter():
    """u.umortality → player_mortality (running death count).

    Cite: vendor/nethack/include/you.h line 497.
    """
    import jax
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    state, _ = env.reset(jax.random.PRNGKey(0))
    assert hasattr(state, "player_mortality")
    assert int(state.player_mortality) == 0


# ---------------------------------------------------------------------------
# C. tiles.py enum vs vendor/nethack/include/rm.h::levl_typ_types
# ---------------------------------------------------------------------------

def test_tiletype_stone_zero():
    """vendor: STONE = 0 (rm.h line 56)."""
    from Nethax.nethax.constants.tiles import VendorTileType
    assert int(VendorTileType.STONE) == 0


def test_tiletype_drawbridge_up_present():
    """vendor: DRAWBRIDGE_UP = 19 (rm.h line 75)."""
    from Nethax.nethax.constants.tiles import VendorTileType
    assert hasattr(VendorTileType, "DRAWBRIDGE_UP")
    assert int(VendorTileType.DRAWBRIDGE_UP) == 19


def test_tiletype_count_matches_vendor():
    """vendor: MAX_TYPE = 37 (rm.h line 94); 37 entries STONE..CLOUD."""
    from Nethax.nethax.constants.tiles import VendorTileType, VENDOR_MAX_TYPE
    assert VENDOR_MAX_TYPE == 37
    assert len(VendorTileType) == 37
    # CLOUD is the last real entry at value 36.
    assert int(VendorTileType.CLOUD) == 36


def test_tiletype_vendor_specific_entries():
    """Spot-check a handful of mid-table entries for byte-exact value match."""
    from Nethax.nethax.constants.tiles import VendorTileType
    expected = {
        "VWALL":           1,
        "HWALL":           2,
        "TREE":           13,
        "SDOOR":          14,
        "POOL":           16,
        "MOAT":           17,
        "WATER":          18,
        "LAVAPOOL":       20,
        "IRONBARS":       22,
        "DOOR":           23,
        "CORR":           24,
        "ROOM":           25,
        "STAIRS":         26,
        "LADDER":         27,
        "FOUNTAIN":       28,
        "THRONE":         29,
        "SINK":           30,
        "GRAVE":          31,
        "ALTAR":          32,
        "ICE":            33,
        "DRAWBRIDGE_DOWN":34,
        "AIR":            35,
        "CLOUD":          36,
    }
    for name, value in expected.items():
        assert int(getattr(VendorTileType, name)) == value, (
            f"VendorTileType.{name} expected {value}, "
            f"got {int(getattr(VendorTileType, name))}"
        )


# ---------------------------------------------------------------------------
# D. actions.py Command enum byte-equal cmd.c::extcmdlist key column
# ---------------------------------------------------------------------------

def test_command_eat_value_lowercase_e():
    """vendor cmd.c line 1712: { 'e', "eat", ... }."""
    from Nethax.nethax.constants.actions import Command
    assert int(Command.EAT) == ord("e")


def test_command_pray_value_meta_p():
    """vendor cmd.c: { M('p'), "pray", ... }.  Meta sets bit 7 (0x80)."""
    from Nethax.nethax.constants.actions import Command
    assert int(Command.PRAY) == (0x80 | ord("p"))


def test_command_count_121():
    """NLE canonical action count is 121 (verified Wave 2 against vendor/nle)."""
    from Nethax.nethax.constants.actions import N_ACTIONS, ACTIONS
    assert N_ACTIONS == 121
    assert len(ACTIONS) == 121


def test_command_spot_check_keystrokes():
    """Byte-equal spot-check of several vendor cmd.c extcmdlist entries."""
    from Nethax.nethax.constants.actions import Command, _M, _C
    # vendor cmd.c lines 1667-1750 (selected entries):
    assert int(Command.EXTCMD)   == ord("#")       # '#'
    assert int(Command.APPLY)    == ord("a")       # 'a'
    assert int(Command.CAST)     == ord("Z")       # 'Z'
    assert int(Command.CLOSE)    == ord("c")       # 'c'
    assert int(Command.DROP)     == ord("d")       # 'd'
    assert int(Command.ENGRAVE)  == ord("E")       # 'E'
    assert int(Command.FIRE)     == ord("f")       # 'f'
    assert int(Command.ADJUST)   == _M("a")        # M('a')
    assert int(Command.ENHANCE)  == _M("e")        # M('e')
    assert int(Command.ATTRIBUTES) == _C("x")      # C('x')
    assert int(Command.KICK)     == _C("d")        # C('d')
