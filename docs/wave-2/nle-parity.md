# Wave 2 — NLE-parity status (re-verified against live NLE)

Wave 1 estimated several NLE constants from C header arithmetic. Wave 2 installed NLE in the project venv and re-verified by reading constants directly from `nle.nethack.*`.

## Live NLE values (re-read this wave)

```
NUMMONS = 381                  (was estimated 394)
NUM_OBJECTS = 453              (matched estimate)

GLYPH_MON_OFF      = 0
GLYPH_PET_OFF      = 381
GLYPH_INVIS_OFF    = 762
GLYPH_DETECT_OFF   = 763
GLYPH_BODY_OFF     = 1144
GLYPH_RIDDEN_OFF   = 1525
GLYPH_OBJ_OFF      = 1906
GLYPH_CMAP_OFF     = 2359      (was estimated 4000)
GLYPH_EXPLODE_OFF  = 2446
GLYPH_ZAP_OFF      = 2509
GLYPH_SWALLOW_OFF  = 2541
GLYPH_WARNING_OFF  = 5589
GLYPH_STATUE_OFF   = 5595
MAX_GLYPH          = 5976      (was estimated 10186)
NO_GLYPH           = 5976
```

The two largest Wave 1 errors:
- `GLYPH_CMAP_OFF` was 4000 (estimate) vs **2359** (live). Off by ~70%.
- `MAX_GLYPH` was 10186 vs **5976**.

Both are because Wave 1 estimated `_MAXPCHARS = 114` for the cmap region and used arithmetic that didn't match NLE's actual layout. Wave 2 just reads `nle.nethack.GLYPH_*_OFF` directly — no more arithmetic.

These now live in `Nethax/nethax/constants/glyphs.py`.

## Vendor-parity test

`tests/test_vendor_parity.py` was added in Wave 2. It runs only when NLE is importable (`@pytest.mark.skipif(not nle_installed)`), and asserts:

- Every action in our `Action` enum has the same int value as `nle.nethack.ACTIONS` (overlap names)
- Every `BL_*` constant matches `nle.nethack.NLE_BL_*`
- Every `GLYPH_*_OFF` constant matches `nle.nethack.GLYPH_*_OFF`
- `MAX_GLYPH` and `NO_GLYPH` match

All three currently pass.

## Action count

Wave 1's audit reported "119 actions / 95 useful". Direct exec of vendor source revealed **121 / 101**. Wave 2 fixed:

- `Nethax/nethax/constants/actions.py` — `N_ACTIONS == 121`, `len(USEFUL_ACTIONS) == 101`
- All related tests updated to assert these canonical counts.

## Observation dict

The 17-key observation builder in `obs/nle_obs.py` is now projecting real values (no longer all zeros). Specifically:

| Key | Wave 1 | Wave 2 |
|---|---|---|
| `glyphs` | zeros | projected from terrain + player overlay, with fog-of-war |
| `chars` | zeros | projected via `_CMAP_TO_CHAR` lookup |
| `blstats` | zeros | all 27 fields populated (HP, Pw, depth, gold, time, hunger, etc.) |
| `message` | zeros | `state.messages.message_buffer` |
| `tty_chars` | zeros | 24-row terminal: message + map + status |
| `tty_cursor` | zeros | `(player_row+1, player_col)` |
| `colors`, `specials`, `tty_colors`, `inv_*`, `screen_descriptions`, `program_state`, `internal`, `misc` | zeros | zeros (Wave 3+) |

11 of 17 keys carry real values; 6 still zero.

## Data table parity

- **MONSTERS**: 390 (NLE canonical 381). Wave 6 polish will trim to exact match by removing entries that NLE excludes via `#if 0` blocks.
- **OBJECTS**: 503 (NLE canonical 453). Wave 3+ canonicalizes the dual-named potions/scrolls/wands to bring count to ~453.

## What still differs from NLE

| NLE feature | Wave 2 status | Target wave |
|---|---|---|
| Auto-handling of `<More>` prompts and Y/N dialogs | JAX env has no prompts — n/a | n/a |
| `info["end_status"]` (`RUNNING`/`DEATH`/`TASK_SUCCESSFUL`/`ABORTED`) | empty info | Wave 6 |
| `info["is_ascended"]` | empty | Wave 6 |
| `wizkit_items` reset kwarg | not yet | Wave 4 (after inventory wired) |
| Explicit `seed(core, disp, reseed)` API | env takes `rng` directly | Wave 3 if needed for parity |
| Color tables for `tty_colors` | zero | Wave 3 |
| Inventory observation keys | zero | Wave 3 |
| Status line color encoding | grayscale | Wave 3 |
