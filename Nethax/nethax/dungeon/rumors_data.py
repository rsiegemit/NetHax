"""Pre-computed false rumor tables for vendor-parity wipeout_text simulation.

Vendor cite:
    vendor/nle/src/rumors.c::getrumor (lines 91-179) — seek false_rumor_start
        + rn2(false_rumor_size), discard partial line via dlb_fgets, then read
        next full line via dlb_fgets.  Trailing PAD_RUMORS_TO underscores are
        stripped.
    vendor/nle/util/makedefs.c::do_rumors (lines 1013-1103) — header layout and
        per-line PAD_RUMORS_TO=60 padding scheme.
    vendor/nle/src/engrave.c::wipeout_text (lines 82-142) — character-level
        rubout simulation invoked by random_engraving (line 23).

The byte-parity validator drives reset.fill_ordinary_rooms with the vendor
ISAAC64 stream.  For each ordinary room whose graffiti gate fires, vendor
consumes:

    rn2(4)                              — engrave.c:20 branch chooser
    rn2(2)                              — rumors.c:124 truth-coin
    rn2(false_rumor_size)               — rumors.c:133 tidbit offset
    [wipeout_text loop, cnt = strlen(rumor) / 4 iters]
        rn2(strlen(rumor))              — engrave.c:95 char position
        rn2(4)                          — engrave.c:96 use_rubout flag
        [conditionally] rn2(strlen(wipeto))  — engrave.c:124 substitute pick

This module pre-computes (at import time) the data structures needed to
reproduce that draw stream byte-exactly for the false_rumor path:

    FALSE_LEN_BY_TIDBIT[25515] : int32        — strlen(rumor) for each tidbit
    FALSE_CHARS_BY_TIDBIT[25515, 80] : uint8  — padded char codes per tidbit

(For each tidbit value, we record the rumor that getrumor would select —
collapsing the (rumor_idx, char-table) two-level lookup into a single flat
indexable table for JIT-friendly access.)

The rubout substitution table from engrave.c:28-78 is mirrored as:

    RUBOUTS_LEN[256] : int32   — len(wipeto) for each input char (0 = not in
                                  table, treated as ``i == SIZE(rubouts)``)

Tables are built only when ``NETHAX_RUMORS_TABLES`` is requested (i.e. when
called from rooms.py); the import cost is ~30 ms.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Vendor rumor file paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPILED_RUMORS = (
    _REPO_ROOT / "vendor" / "nle" / "build" /
    "temp.macosx-14.0-arm64-cpython-312" / "dat" / "rumors"
)
_RUMORS_FAL = _REPO_ROOT / "vendor" / "nle" / "dat" / "rumors.fal"
_RUMORS_TRU = _REPO_ROOT / "vendor" / "nle" / "dat" / "rumors.tru"

# Vendor header values (cross-checked at runtime against compiled file).
# false_rumor_size = 25515 — matches false_rumor_size on seed=1 trace draw 1311.
FALSE_RUMOR_SIZE: int = 25515
TRUE_RUMOR_SIZE: int = 23549


# ---------------------------------------------------------------------------
# Build helpers — replicate vendor makedefs do_rumors PAD_RUMORS_TO logic
# ---------------------------------------------------------------------------

def _xcrypt(data: bytes) -> bytes:
    """Mirror vendor xcrypt (util/makedefs.c:886-903).  Symmetric encode/decode.
    """
    out = bytearray()
    bitmask = 1
    for c in data:
        b = c
        if b & (32 | 64):
            b ^= bitmask
        out.append(b)
        bitmask <<= 1
        if bitmask >= 32:
            bitmask = 1
    return bytes(out)


_PAD_RUMORS_TO = 60


def _pad_rumor_line(line: bytes) -> bytes:
    """Mirror makedefs read_rumors_file PAD_RUMORS_TO logic (lines 927-946).

    Vendor C:
        int len = (int) strlen(line);   // INCLUDES the trailing '\\n'
        if (len <= PAD_RUMORS_TO) {
            char *base = index(line, '\\n');
            while (len++ < PAD_RUMORS_TO) { *base++ = '_'; }
            *base++ = '\\n';
            *base = '\\0';
        }

    Net effect: lines whose ``len(line_with_newline) <= 60`` are padded to
    exactly 60 bytes total (59 chars of body + ``\\n``).  Longer lines are
    emitted unchanged.
    """
    if line.endswith(b"\n"):
        body, term = line[:-1], b"\n"
    else:
        body, term = line, b""
    # Vendor compares strlen(line_with_newline) <= 60 ⇒ len(body) <= 59
    # before padding kicks in.  The loop appends '_' until the total length
    # (incl. newline) reaches 60.
    total_len = len(body) + len(term)
    if total_len <= _PAD_RUMORS_TO:
        pad_count = _PAD_RUMORS_TO - total_len
        body = body + b"_" * pad_count
    return body + term


def _build_compiled_false_section() -> Tuple[bytes, bytes]:
    """Replicate the false-rumor section of the compiled `rumors` file.

    Returns
    -------
    section : bytes
        The xcrypt-encoded, PAD_RUMORS_TO-padded false-rumor block.  This is
        exactly the byte range ``[false_rumor_start, false_rumor_end)`` of the
        compiled dlb-wrapped file.
    raw_section : bytes
        Same shape but pre-xcrypt (for human inspection).
    """
    raw_lines = []
    with open(_RUMORS_FAL, "rb") as fp:
        for line in fp:
            if not line.endswith(b"\n"):
                line = line + b"\n"
            raw_lines.append(_pad_rumor_line(line))
    raw_section = b"".join(raw_lines)
    section = b"".join(_xcrypt(line) for line in raw_lines)
    assert len(section) == FALSE_RUMOR_SIZE, (
        f"compiled false section length {len(section)} != "
        f"FALSE_RUMOR_SIZE {FALSE_RUMOR_SIZE}"
    )
    return section, raw_section


def _decode_rumor_at(section: bytes, tidbit: int) -> bytes:
    """Mirror vendor rumors.c::getrumor seek+fgets+fgets for a given tidbit.

    The compiled file's false-rumor block starts at section[0].  Vendor seeks
    to ``false_rumor_start + tidbit``, calls ``dlb_fgets`` to consume up to and
    including the next newline (the partial line — discarded), then calls
    ``dlb_fgets`` again to read the next full line.

    If the second fgets fails or pushes past ``false_rumor_end``, vendor
    rewinds to ``false_rumor_start`` and reads the first full line
    (rumors.c:142-146).

    After reading, vendor strips the newline (line 147-148), then strips
    trailing ``_`` padding (lines 166-176) and replaces the terminator with a
    NUL — net effect: the returned rumor has the trailing newline AND the
    padding both removed.

    Returns
    -------
    decoded : bytes
        Plaintext rumor with newline / underscore-padding stripped.  This is
        what wipeout_text sees as its input buffer.
    """
    sz = len(section)
    pos = tidbit
    # First fgets: from pos up to and including the next '\n'.
    nl1 = section.find(b"\n", pos)
    if nl1 < 0:
        # Reached end of section without finding newline → wrap.
        line_start = 0
    else:
        line_start = nl1 + 1
    # Second fgets: from line_start up to and including the next '\n'.
    nl2 = section.find(b"\n", line_start)
    if nl2 < 0 or line_start >= sz:
        # Either second fgets fails (line_start past EOF) or we'd cross
        # false_rumor_end.  Vendor wraps: rewind to false_rumor_start, fgets.
        line_start = 0
        nl2 = section.find(b"\n", line_start)
    raw_line = section[line_start:nl2]      # excludes '\n'
    decoded = _xcrypt(raw_line)
    # Strip trailing '_' padding.
    end = len(decoded)
    while end > 0 and decoded[end - 1:end] == b"_":
        end -= 1
    return decoded[:end]


# ---------------------------------------------------------------------------
# Rubouts table — engrave.c:28-78
# ---------------------------------------------------------------------------

_RUBOUTS = {
    ord('A'): "^",
    ord('B'): "Pb[",
    ord('C'): "(",
    ord('D'): "|)[",
    ord('E'): "|FL[_",
    ord('F'): "|-",
    ord('G'): "C(",
    ord('H'): "|-",
    ord('I'): "|",
    ord('K'): "|<",
    ord('L'): "|_",
    ord('M'): "|",
    ord('N'): "|\\",
    ord('O'): "C(",
    ord('P'): "F",
    ord('Q'): "C(",
    ord('R'): "PF",
    ord('T'): "|",
    ord('U'): "J",
    ord('V'): "/\\",
    ord('W'): "V/\\",
    ord('Z'): "/",
    ord('b'): "|",
    ord('d'): "c|",
    ord('e'): "c",
    ord('g'): "c",
    ord('h'): "n",
    ord('j'): "i",
    ord('k'): "|",
    ord('l'): "|",
    ord('m'): "nr",
    ord('n'): "r",
    ord('o'): "c",
    ord('q'): "c",
    ord('w'): "v",
    ord('y'): "v",
    ord(':'): ".",
    ord(';'): ",:",
    ord(','): ".",
    ord('='): "-",
    ord('+'): "-|",
    ord('*'): "+",
    ord('@'): "0",
    ord('0'): "C(",
    ord('1'): "|",
    ord('6'): "o",
    ord('7'): "/",
    ord('8'): "3o",
}

# Punctuation that gets blanked (engrave.c:110 ``index("?.,'`-|_", *s)``).
_PUNCT_BLANK = set(ord(c) for c in "?.,'`-|_")


# Per-char outcome class used by the JIT-side wipeout simulator.
# 0 = ' ' (space): continue, no extra draw
# 1 = punctuation blank: continue, no extra draw
# 2 = use_rubout=0 path (assigns '?', no extra draw)  ← determined at runtime
# 3 = use_rubout!=0 AND char in rubouts: 1 extra rn2(wipeto_len) draw
# 4 = use_rubout!=0 AND char not in rubouts: assigns '?', no extra draw
#
# We pre-encode only the *static* per-char information:
#   - is_space          : bool
#   - is_punct_blank    : bool
#   - rubout_len        : int (0 if char not in rubouts)
#
# The runtime branch on use_rubout==0 is handled at the loop call site.


# ---------------------------------------------------------------------------
# Per-tidbit table builder
# ---------------------------------------------------------------------------

# Each rumor is at most this many chars (vendor max false_rumor strlen = 79;
# we use 96 for headroom).
MAX_RUMOR_LEN: int = 96


# Char-class encoding used by the JIT body to dispatch behaviour:
#   0 → other (use_rubout=0 → '?'; use_rubout!=0 + char not in rubouts → '?')
#       — no extra draw in either sub-branch
#   1 → space         → no extra draw (continue)
#   2 → punct blank   → no extra draw (continue)
#   3 → rubout-eligible (with use_rubout!=0): draw one rn2(rubout_len)
#       (with use_rubout==0): no extra draw → '?'
CHAR_CLASS_OTHER = 0
CHAR_CLASS_SPACE = 1
CHAR_CLASS_PUNCT = 2
CHAR_CLASS_RUBOUT = 3


def _build_tidbit_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the flat numpy arrays indexed by tidbit ∈ [0, FALSE_RUMOR_SIZE).

    Returns
    -------
    text_len[FALSE_RUMOR_SIZE] : int32
        strlen(rumor) for the rumor that getrumor returns at this tidbit.
    char_class[FALSE_RUMOR_SIZE, MAX_RUMOR_LEN] : int8
        Per-char dispatch class (see CHAR_CLASS_* constants above).  Cells
        beyond the rumor's text_len are filled with CHAR_CLASS_OTHER (0) and
        never indexed at runtime (the rn2(text_len) draw bounds nxt to the
        valid prefix).
    rubout_len[FALSE_RUMOR_SIZE, MAX_RUMOR_LEN] : int32
        len(rubouts.wipeto) for chars[i,j].  Zero when char_class != RUBOUT.
    """
    section, _ = _build_compiled_false_section()
    text_len = np.zeros(FALSE_RUMOR_SIZE, dtype=np.int32)
    char_class = np.zeros((FALSE_RUMOR_SIZE, MAX_RUMOR_LEN), dtype=np.int8)
    rubout_len = np.zeros((FALSE_RUMOR_SIZE, MAX_RUMOR_LEN), dtype=np.int32)
    for tidbit in range(FALSE_RUMOR_SIZE):
        text = _decode_rumor_at(section, tidbit)
        L = min(len(text), MAX_RUMOR_LEN)
        text_len[tidbit] = L
        for j in range(L):
            c = text[j]
            if c == ord(' '):
                char_class[tidbit, j] = CHAR_CLASS_SPACE
            elif c in _PUNCT_BLANK:
                char_class[tidbit, j] = CHAR_CLASS_PUNCT
            elif c in _RUBOUTS:
                char_class[tidbit, j] = CHAR_CLASS_RUBOUT
                rubout_len[tidbit, j] = len(_RUBOUTS[c])
            # else: stays CHAR_CLASS_OTHER (0)
    return text_len, char_class, rubout_len


# ---------------------------------------------------------------------------
# Disk cache — rebuilding the tables takes ~1.4 s, so we serialise to NPZ.
# ---------------------------------------------------------------------------

_CACHE_PATH = (
    _REPO_ROOT / "Nethax" / "nethax" / "dungeon" / "_rumors_tables.npz"
)


def _load_or_build_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (text_len, char_class, rubout_len), loading from NPZ if present."""
    if _CACHE_PATH.exists():
        try:
            with np.load(_CACHE_PATH) as data:
                if (
                    "text_len" in data
                    and "char_class" in data
                    and "rubout_len" in data
                    and data["text_len"].shape == (FALSE_RUMOR_SIZE,)
                ):
                    return (
                        data["text_len"].copy(),
                        data["char_class"].copy(),
                        data["rubout_len"].copy(),
                    )
        except Exception:
            pass
    text_len, char_class, rubout_len = _build_tidbit_tables()
    try:
        np.savez_compressed(
            _CACHE_PATH,
            text_len=text_len,
            char_class=char_class,
            rubout_len=rubout_len,
        )
    except Exception:
        pass
    return text_len, char_class, rubout_len


# ---------------------------------------------------------------------------
# Lazy JAX-side wrappers
# ---------------------------------------------------------------------------

_TABLES = None


def get_jax_tables():
    """Return JAX-backed (text_len, char_class, rubout_len) tables.

    Built lazily on first call (NPZ cache load: ~10 ms; cold build: ~1.4 s).
    Subsequent calls return the cached DeviceArrays.

    Shapes / dtypes
    ---------------
    text_len    : int32[FALSE_RUMOR_SIZE]
    char_class  : int8[FALSE_RUMOR_SIZE, MAX_RUMOR_LEN]
    rubout_len  : int32[FALSE_RUMOR_SIZE, MAX_RUMOR_LEN]
    """
    global _TABLES
    if _TABLES is None:
        import jax.numpy as jnp
        text_len, char_class, rubout_len = _load_or_build_tables()
        _TABLES = (
            jnp.asarray(text_len, dtype=jnp.int32),
            jnp.asarray(char_class, dtype=jnp.int8),
            jnp.asarray(rubout_len, dtype=jnp.int32),
        )
    return _TABLES
