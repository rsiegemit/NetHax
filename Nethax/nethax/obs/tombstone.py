"""Tombstone screen — byte-equal port of vendor/nethack/src/rip.c.

Vendor source: vendor/nethack/src/rip.c::genl_outrip (lines 27-163).

The RIP card is a 15-line ASCII tombstone built from a static template
(``rip_txt`` lines 27-43 of rip.c).  Four lines (NAME, GOLD, DEATH, YEAR)
are overwritten with centered text via the ``center()`` helper (rip.c
lines 75-83), which copies bytes starting at column
``STONE_LINE_CENT - ((strlen(text)+1) >> 1)``.

Constants (byte-equal to vendor):
    STONE_LINE_CENT = 28  # column of stone-face center  (rip.c line 44)
    STONE_LINE_LEN  = 16  # max chars per line, includes 1-space border (rip.c L68)
    NAME_LINE  = 6        # rip_txt index for player name  (rip.c line 70)
    GOLD_LINE  = 7        # ... for gold                    (rip.c line 71)
    DEATH_LINE = 8        # ... for death description (4 lines, DEATH..YEAR-1) (rip.c L72)
    YEAR_LINE  = 12       # ... for year of death           (rip.c line 73)

Host-side helper (not jit'd) — used for end-of-game display + parity tests.
"""
from __future__ import annotations

from typing import List, Optional


# vendor/nethack/src/rip.c lines 27-43  (single-tombstone form, NH320_DEDICATION undef).
# Byte-equal copy — DO NOT EDIT spacing or characters.
_RIP_TXT: List[str] = [
    "                       ----------",
    "                      /          \\",
    "                     /    REST    \\",
    "                    /      IN      \\",
    "                   /     PEACE      \\",
    "                  /                  \\",
    "                  |                  |",      # NAME_LINE = 6
    "                  |                  |",      # GOLD_LINE = 7
    "                  |                  |",      # DEATH_LINE = 8
    "                  |                  |",      #
    "                  |                  |",      #
    "                  |                  |",      #
    "                  |       1001       |",      # YEAR_LINE = 12
    "                 *|     *  *  *      | *",
    "        _________)/\\\\_//(\\/(/\\)/\\//\\/|_)_______",
]

STONE_LINE_CENT = 28
STONE_LINE_LEN  = 16
NAME_LINE  = 6
GOLD_LINE  = 7
DEATH_LINE = 8
YEAR_LINE  = 12


def _center(line: str, text: str) -> str:
    """Overwrite bytes of `line` starting at the centered position.

    Direct port of vendor/nethack/src/rip.c::center (lines 75-83):

        op = &gr.rip[line][STONE_LINE_CENT - ((strlen(text) + 1) >> 1)];
        while (*ip) *op++ = *ip++;

    i.e. `text` is COPIED INTO `line` starting at column
    `STONE_LINE_CENT - ((len(text) + 1) // 2)`.  No truncation, no padding —
    the surrounding line characters around the substitution are preserved.
    """
    if not text:
        return line
    col = STONE_LINE_CENT - ((len(text) + 1) >> 1)
    if col < 0:
        col = 0
    # In C, this writes len(text) bytes starting at `op`.  Replicate exactly.
    new = list(line)
    for i, ch in enumerate(text):
        if col + i < len(new):
            new[col + i] = ch
    return "".join(new)


def _split_killer(killer: str) -> List[str]:
    """Split `killer` into ≤ STONE_LINE_LEN-wide chunks at spaces, like
    rip.c::genl_outrip lines 116-135.

    Vendor algorithm: walk from STONE_LINE_LEN backward looking for a space;
    if found, split there; else split at exactly STONE_LINE_LEN.  Up to 4
    lines (DEATH_LINE..YEAR_LINE-1).
    """
    out: List[str] = []
    s = killer
    while s and len(out) < (YEAR_LINE - DEATH_LINE):
        if len(s) <= STONE_LINE_LEN:
            out.append(s)
            break
        i0 = len(s)
        # Walk backward from STONE_LINE_LEN looking for a space.
        i = STONE_LINE_LEN
        while i > 0 and i0 > STONE_LINE_LEN:
            if i < len(s) and s[i] == " ":
                i0 = i
            i -= 1
        if i0 > STONE_LINE_LEN:
            i0 = STONE_LINE_LEN
        out.append(s[:i0])
        # Skip the split-space if it was one.
        if i0 < len(s) and s[i0] == " ":
            s = s[i0 + 1:]
        else:
            s = s[i0:]
    return out


def build_tombstone(
    state,
    *,
    name: str = "Adventurer",
    killer: str = "killed",
    year: Optional[int] = None,
    gold: Optional[int] = None,
) -> List[str]:
    """Return the RIP card as 15 lines.

    Parameters mirror rip.c::genl_outrip:
      name   -> svp.plname            (NAME_LINE)
      gold   -> gd.done_money         (GOLD_LINE; defaults to state's gold)
      killer -> formatkiller(how)     (DEATH_LINE..YEAR_LINE-1)
      year   -> yyyymmdd(when)/10000  (YEAR_LINE; defaults to 2026)

    Vendor citation: vendor/nethack/src/rip.c::genl_outrip (lines 86-163).
    """
    # Defaults sourced from state when not explicitly provided.
    if gold is None:
        try:
            from Nethax.nethax.constants.blstats import BL_GOLD
            from Nethax.nethax.obs.nle_obs import build_blstats
            import numpy as np
            gold = int(np.asarray(build_blstats(state))[BL_GOLD])
        except Exception:
            gold = 0
    if year is None:
        import datetime
        year = datetime.datetime.now().year

    rip = list(_RIP_TXT)

    # NAME_LINE: truncate to STONE_LINE_LEN bytes (vendor uses %.*s with that width).
    name_text = name[:STONE_LINE_LEN]
    rip[NAME_LINE] = _center(rip[NAME_LINE], name_text)

    # GOLD_LINE: "<cash> Au" centered.
    cash = max(int(gold), 0)
    if cash > 999_999_999:
        cash = 999_999_999
    gold_text = f"{cash} Au"
    rip[GOLD_LINE] = _center(rip[GOLD_LINE], gold_text)

    # DEATH_LINE..YEAR_LINE-1: vendor splits killer string at spaces.
    death_lines = _split_killer(killer)
    for i, dline in enumerate(death_lines):
        rip[DEATH_LINE + i] = _center(rip[DEATH_LINE + i], dline)

    # YEAR_LINE: 4-digit year.
    rip[YEAR_LINE] = _center(rip[YEAR_LINE], f"{year:4d}")

    return rip
