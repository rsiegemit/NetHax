"""NLE glyph offset scheme — values verified against a live NLE build.

Glyphs are integer IDs that the NLE environment returns in the `glyphs`
observation array.  Each segment of glyph-space is contiguous and starts
at a named offset.

These values were read directly from `nle.nethack.GLYPH_*_OFF` constants
exposed by the compiled NLE library (`vendor/nle/win/rl/pynethack.cc`).
They are the source of truth — DO NOT recompute them from formulas, since
NLE's offsets depend on the exact NetHack build (NUMMONS, NUM_OBJECTS,
MAXPCHARS, et al.) and any drift will silently break agent parity.

Canonical source: vendor/nle/win/rl/pynethack.cc:477-494,
                  vendor/nethack/include/display.h:497-546
Status: Wave 2 — synced to live NLE build via `.venv/bin/python -c
                  "import nle.nethack as n; print(n.GLYPH_MON_OFF, ...)"`.
"""

# ---------------------------------------------------------------------------
# Compiled-time counts (read from `nle.nethack`)
# ---------------------------------------------------------------------------

NUMMONS: int      = 381
NUM_OBJECTS: int  = 453
EXPL_MAX: int     = 7      # vendor/nethack/include/display.h:323
NUM_ZAP: int      = 8      # vendor/nethack/include/display.h:359
WARNCOUNT: int    = 6      # vendor/nethack/include/sym.h:174
MAXEXPCHARS: int  = 9      # vendor/nethack/include/sym.h:94

# ---------------------------------------------------------------------------
# Glyph offset constants — canonical NLE values
# ---------------------------------------------------------------------------

GLYPH_MON_OFF: int      = 0
GLYPH_PET_OFF: int      = 381
GLYPH_INVIS_OFF: int    = 762
GLYPH_DETECT_OFF: int   = 763
GLYPH_BODY_OFF: int     = 1144
GLYPH_RIDDEN_OFF: int   = 1525
GLYPH_OBJ_OFF: int      = 1906
GLYPH_CMAP_OFF: int     = 2359
GLYPH_EXPLODE_OFF: int  = 2446
GLYPH_ZAP_OFF: int      = 2509
GLYPH_SWALLOW_OFF: int  = 2541
GLYPH_WARNING_OFF: int  = 5589
GLYPH_STATUE_OFF: int   = 5595
MAX_GLYPH: int          = 5976
NO_GLYPH: int           = 5976

# Convenience alias used in some NLE call sites
GLYPH_INVISIBLE: int    = GLYPH_INVIS_OFF
