"""Parity mode selector — choose which upstream byte-layout to mirror.

Two of Nethax's upstream references disagree on byte-level details:

* **NLE** (NetHack Learning Environment, vendor/nle/) — what
  reinforcement-learning agents are trained against.  Cite:
  vendor/nle/include/hack.h, vendor/nle/include/nletypes.h.

* **NetHack 3.7** (vendor/nethack/src/) — the canonical game source.
  Cite: vendor/nethack/include/display.h, hack.h, you.h.

These diverge in a small number of obs-channel byte layouts:

  - MG_* bit layout in the specials channel (NLE drops MG_HERO and
    shifts every bit; NetHack 3.7 reserves bit 0x01 for MG_HERO).
  - blstats column ordering for some fields added post-NLE-fork.
  - tty_chars rendering of certain special tiles.
  - Glyph offset tables when monster/object counts diverge.

By default Nethax mirrors **NLE** — that's what trained agents see, and
this is the configuration most users want.  Set
``set_parity_mode(ParityMode.NETHACK)`` to flip every per-byte choice
to NetHack 3.7 byte layout instead (useful for replaying a real
NetHack save file or comparing against C builds).

Usage:
  from Nethax.nethax.parity_mode import ParityMode, set_parity_mode
  set_parity_mode(ParityMode.NETHACK)  # opt out of NLE mode
  # ... obs builders now use NetHack 3.7 layout ...

This module is intentionally non-stateful at the JAX level — the mode
is a host-side Python global that affects how the observation arrays
are constructed.  JIT-compiled functions still see static int8/uint8
constants; the choice resolves at trace time.
"""

from enum import IntEnum


class ParityMode(IntEnum):
    """Which upstream byte layout to mirror."""

    NLE = 0              # default — matches vendor/nle/ for agent transferability
    NETHACK = 1          # matches vendor/nethack/ for replay parity with C builds
    NLE_BYTEPARITY = 2   # NLE layout + ISAAC64 vendor RNG (byte-exact rollouts)


_current: ParityMode = ParityMode.NLE


def set_parity_mode(mode: ParityMode) -> None:
    """Switch the global parity mode.  Affects subsequent obs builds.

    Note: changing the mode while a JIT-cached function is in flight
    will not re-trace until the cache is invalidated.  For deterministic
    behavior, set the mode once at process startup before calling
    ``env.reset`` / ``env.step``.
    """
    global _current
    _current = ParityMode(mode)


def get_parity_mode() -> ParityMode:
    """Return the current parity mode (default ParityMode.NLE)."""
    return _current


def is_nle_mode() -> bool:
    """True iff parity mode is NLE-style obs layout (default or byte-parity)."""
    return _current in (ParityMode.NLE, ParityMode.NLE_BYTEPARITY)


def is_nethack_mode() -> bool:
    """True iff parity mode is vendor NetHack 3.7."""
    return _current == ParityMode.NETHACK


def use_vendor_rng() -> bool:
    """True iff the active mode requires byte-exact ISAAC64 RNG.

    Threefry is fast and JIT-friendly but its bytes never match NLE.  When
    this returns True, the env threads an ``Isaac64State`` through every
    site that would otherwise consume a Threefry key.
    Cite: vendor/nle/include/config.h:584 ``#define USE_ISAAC64``.
    """
    return _current == ParityMode.NLE_BYTEPARITY


# ---------------------------------------------------------------------------
# MG_* special-bit layouts.
#
# Cite vendor/nle/include/hack.h:77-84 (NLE) vs
#      vendor/nethack/include/display.h:995-1009 (NetHack 3.7).
# ---------------------------------------------------------------------------

class _MGBits_NLE:
    """NLE-bundled MG_* layout (vendor/nle/include/hack.h:77-84)."""
    MG_HERO     = 0x00  # not present in NLE
    MG_CORPSE   = 0x01
    MG_INVIS    = 0x02
    MG_DETECT   = 0x04
    MG_PET      = 0x08
    MG_RIDDEN   = 0x10
    MG_STATUE   = 0x20
    MG_OBJPILE  = 0x40
    MG_BW_LAVA  = 0x80


class _MGBits_NetHack:
    """NetHack 3.7 MG_* layout (vendor/nethack/include/display.h:995-1009).

    NetHack reserves bit 0x01 for the hero tile.  All other bits shift up.
    NetHack uses a 32-bit field internally; the high bits (BW_ICE/BW_SINK/
    BW_ENGR/NOTHING) don't fit in a uint8.
    """
    MG_HERO     = 0x01
    MG_CORPSE   = 0x02
    MG_INVIS    = 0x04
    MG_DETECT   = 0x08
    MG_PET      = 0x10
    MG_RIDDEN   = 0x20
    MG_STATUE   = 0x40
    MG_OBJPILE  = 0x80
    MG_BW_LAVA  = 0x00  # doesn't fit in uint8 under NetHack layout


def mg_bits():
    """Return the active MG_* bit table (NLE by default)."""
    return _MGBits_NLE if is_nle_mode() else _MGBits_NetHack
