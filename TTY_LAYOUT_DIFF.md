# TTY_LAYOUT_DIFF — `tty_chars[24, 80]` Byte-Level Audit

Comparison of Nethax's `build_tty` (Nethax/nethax/obs/nle_obs.py) against vendor NLE
(`vendor/nle/src/nle.c`, `vendor/nle/win/rl/winrl.cc`, `vendor/nethack/win/tty/wintty.c`,
`vendor/nethack/src/botl.c`).

NLE produces `tty_chars` by capturing the libtmt virtual-terminal output of NetHack's
TTY windowport. Rows are 0-indexed.

---

## Row-Level Layout (high level — MATCHES)

| Row(s) | Content                              | Vendor source                                 | Nethax build_tty                  | Status |
|--------|--------------------------------------|-----------------------------------------------|-----------------------------------|--------|
| 0      | Top-line message (NHW_MESSAGE)       | wintty.c create_nhwindow: offy=0              | `tty[0,:] = message_buffer[:80]`  | ROW OK |
| 1-21   | Map (NHW_MAP, ROWNO=21 rows)         | wintty.c: NHW_MAP offy=1, rows=ROWNO=21       | `tty[1:22,:] = map_chars_80`      | ROW OK |
| 22-23  | Status (NHW_STATUS, 2 lines)         | wintty.c: NHW_STATUS rowoffset = LI - 2 = 22  | `tty[22,:] = row22; tty[23,:] = row23` | ROW OK |

`COLNO=80`, `ROWNO=21`, `NLE_TERM_LI=24`, `NLE_TERM_CO=80` confirmed in
`vendor/nethack/include/global.h:382-383` and NLE pynethack.cc.

---

## Byte-Level Divergences

### D1. Background-fill byte: `0x00` vs `0x20` (space)  [TRIVIAL — FIXED]

- Vendor: libtmt VT initialises every cell to ASCII space (`' '` = 0x20) per
  `vendor/nle/src/nle.c::nle_vt_callback` (TMT_MSG_UPDATE copies `tmt_c->c` which
  defaults to space).
- Nethax (pre-fix): `tty = jnp.zeros((24, 80), dtype=jnp.uint8)` — every unwritten
  cell is `0x00`.
- Fix applied: initialise grid with `0x20` and remap message-row zero bytes to space.

### D2. Message row — zero-padding instead of space-padding  [TRIVIAL — FIXED]

- Vendor: `tty_putstr(WIN_MESSAGE, ...)` writes message then libtmt fills tail with
  space (terminal default).
- Nethax (pre-fix): `tty[0,:] = message_buffer[:80]` where `message_buffer` is
  zero-padded after the message text (see `subsystems/messages.py:148`).
- Fix applied: any `0x00` byte in the row-0 slice is rewritten to `0x20`.

### D3. Row 23 numeric fields — right-aligned vs vendor left-justified  [NON-TRIVIAL]

Vendor `botl.c::do_statusline2` uses (line 130, 143, 463):
- `"Dlvl:%-2d"` → `"Dlvl:1 "` (left-justified, trailing space if 1 digit)
- `"$:%-2ld"`   → `"$:0 "`
- `"HP:%d(%d) Pw:%d(%d) AC:%-2d"` → `"HP:15(15) Pw:5(5) AC:-1"` (no width on HP/Pw)
- `"Xp:%d/%-1ld"` (showexp) or `"Xp:%d"` — no width
- `"T:%ld"` — no width

Nethax `_build_status_row2` uses `_uint_to_bytes(..., W)` which is **right-aligned
with leading-space fill**:
- `Dlvl: 1` (leading space) vs vendor `Dlvl:1 `
- `$:   0` vs vendor `$:0 `
- `HP:  15(  15)` vs vendor `HP:15(15)`
- `Pw:  5(  5)` vs vendor `Pw:5(5)`
- `AC: 6` (right-aligned w=2) vs vendor `AC:6 ` (left-aligned w=2)
- `Xp: 1` vs vendor `Xp:1`
- `T:    1` (w=5) vs vendor `T:1`

Every numeric byte in row 23 is shifted. Fix requires a new `_uint_to_bytes_left`
helper or per-field formatting. Out of scope for "trivial" pass — flagged here.

### D4. Row 22 missing condition tail / score / hunger  [NON-TRIVIAL]

Vendor `do_statusline1` ends with `"  Lawful"`/`"  Neutral"`/`"  Chaotic"` and
optionally `" S:<score>"`. Nethax matches alignment string but skips score.
Row 22 also does not include the optional `#SCORE_ON_BOTL` field.

### D5. Row 23 condition keyword list — incomplete  [NON-TRIVIAL]

Vendor (`botl.c:173-205`) emits conditions in order:
` Stone Slime Strngl FoodPois TermIll <Hunger> <Encumbrance> Blind Deaf Stun Conf
 Hallu Lev Fly Ride`.

Nethax emits a different subset/order in `build_status_conditions`:
` Conf Stun Hallu Blind FoodPois Ill Slime Strngl <Encumbrance>`.
Missing: ` Stone`, ` TermIll` (Nethax uses ` Ill`), ` <Hunger>` (e.g. `Hungry`/`Weak`),
` Deaf`, ` Lev`, ` Fly`, ` Ride`. Order is also reversed for Stun/Conf vs vendor.

### D6. Row 22 header padding width  [NON-TRIVIAL]

Vendor (botl.c:79-83): pads name+rank field to `gm.mrank_sz + 15` columns.
Nethax pre-computes a fixed 27-byte header (`_HEADER_PAD_W = 27`). `mrank_sz` is the
max-length of any rank title for the current role; padding width is dynamic.

### D7. tty_colors background  [NON-TRIVIAL]

`build_tty_colors` sets entire rows 0, 22, 23 to color `7` (CLR_GRAY). Vendor's
libtmt default-color extract (`nle.c::vt_char_color_extract` line 49) emits
**CLR_BLACK (0)** for space characters with default color. Bytes-mismatch on every
trailing-space cell.

### D8. Column 0 of map rows — vendor "col 0 unused" convention  [DESIGN-LEVEL]

- Vendor: `level.locations[0][y]` is unused. Glyphs displayed at vendor x=1..79 are
  rendered at tty columns 0..78 (after `tty_curs --x`). Tty column 79 untouched.
  NLE's `chars[r][0]` maps to vendor x=1 (`store_mapped_glyph: i = (x - 1)`).
- Nethax: `terrain[..., :21, :79]` slices terrain cols 0..78. Cave generator stamps
  `grid[:, 0] = 0` (wall) and `grid[:, w-1] = 0`, so terrain col 0 is always a
  boundary wall; rooms are placed at col >= 2. Whether Nethax's terrain col 0
  corresponds to vendor x=0 (unused) or x=1 (first usable) is a project-level
  convention that affects every map-emit subsystem. Not a `build_tty` issue per se.

If Nethax intends terrain col 0 == vendor x=0 (unused), then `tty[1:22, 0]` will
show a wall char rather than the floor/feature at vendor x=1, and there is a
left-shift-by-1 across the entire map versus NLE. Investigation requires running
an end-to-end NLE-vs-Nethax frame compare and is out of scope here.

---

## Cursor

Nethax: `(player_row+1, player_col)`. Vendor: cursor lives in tty coords; map's
offy=1 means `vt_cursor = (player_y + 1, player_x - 1)` (since tty_curs --x). If
Nethax's `player_pos[1]` is already 0-indexed tty column (vendor x - 1), match
holds; otherwise off-by-one column on cursor. Flagged for D8-related audit.

---

## Summary of Fixes Applied in This Audit Pass

Commit: `audit(tty): byte-level tty_chars layout audit + trivial fixes`

1. `build_tty` initial grid bytes: `0x00` → `0x20` (D1).
2. Row 0 message-line: replace `0x00` bytes with `0x20` after copy (D2).

All other divergences (D3–D8) are flagged but left for follow-up because each
requires a non-trivial reformatter, vendor-rank-size lookup, condition-token
table expansion, color-by-char dispatch, or project-level coord-convention audit.
