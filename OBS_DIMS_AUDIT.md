# NLE Observation Dims + Channel Audit

Date: 2026-05-26
Branch: `worktree-agent-a01782cc9c19e07e6`

Scope: compare every key in `Nethax/nethax/obs/nle_obs.py::NLE_OBSERVATION_SHAPES`
to the real NLE shapes defined in `vendor/nle/include/nleobs.h` and
`vendor/nle/nle/nethack/nethack.py`, then audit the encoding of `inv_oclasses`
and the byte content of `screen_descriptions`.

## 1. Source-of-truth constants (vendor/nle)

| Constant                         | File                                  | Value |
|---------------------------------|---------------------------------------|-------|
| `ROWNO`                         | `include/global.h:328`                | 21    |
| `COLNO`                         | `include/global.h:327`                | 80    |
| `NLE_TERM_LI`                   | `include/nleobs.h:14`                 | 24    |
| `NLE_TERM_CO`                   | `include/nleobs.h:13`                 | 80    |
| `NLE_INVENTORY_SIZE`            | `include/nleobs.h:10`                 | 55    |
| `NLE_INVENTORY_STR_LENGTH`      | `include/nleobs.h:11`                 | 80    |
| `NLE_SCREEN_DESCRIPTION_LENGTH` | `include/nleobs.h:12`                 | 80    |
| `NLE_MESSAGE_SIZE`              | `include/nleobs.h:5`                  | 256   |
| `NLE_BLSTATS_SIZE`              | `include/nleobs.h:6`                  | 27    |
| `NLE_PROGRAM_STATE_SIZE`        | `include/nleobs.h:7`                  | 6     |
| `NLE_INTERNAL_SIZE`             | `include/nleobs.h:8`                  | 9     |
| `NLE_MISC_SIZE`                 | `include/nleobs.h:9`                  | 3     |

The NLE python wrapper derives shapes from these constants:

```
DUNGEON_SHAPE          = (ROWNO, COLNO - 1)   = (21, 79)
TERMINAL_SHAPE         = (NLE_TERM_LI, NLE_TERM_CO) = (24, 80)
INV_SIZE               = (NLE_INVENTORY_SIZE,) = (55,)
INV_STRS_SHAPE         = (55, 80)
SCREEN_DESCRIPTIONS    = (21, 79, 80)
```

Reference: `vendor/nle/nle/nethack/nethack.py:16-30`.

**Note on the task brief**: the brief stated "NLE may use `[21, 80]`" and
"`tty_chars[24, 80]`" for the dungeon. The dungeon shape is `(21, 79)`
because winrl strips column 0 (`COLNO - 1`). The brief is incorrect on this
point; `(21, 79)` is the canonical NLE dungeon shape. `tty_chars` is `(24, 80)`.

## 2. Dim comparison table

`Nethax/nethax/obs/nle_obs.py:139-157`:

| Key                  | NLE (vendor) shape  | Nethax shape       | Match |
|----------------------|---------------------|--------------------|-------|
| `glyphs`             | (21, 79) int16      | (21, 79) int16     | OK    |
| `chars`              | (21, 79) uint8      | (21, 79) uint8     | OK    |
| `colors`             | (21, 79) uint8      | (21, 79) uint8     | OK    |
| `specials`           | (21, 79) uint8      | (21, 79) uint8     | OK    |
| `blstats`            | (27,) int64         | (27,) int64        | OK    |
| `message`            | (256,) uint8        | (256,) uint8       | OK    |
| `program_state`      | (6,) int32          | (6,) int32         | OK    |
| `internal`           | (9,) int32          | (9,) int32         | OK    |
| `inv_glyphs`         | (55,) int16         | (55,) int16        | OK    |
| `inv_letters`        | (55,) uint8         | (55,) uint8        | OK    |
| `inv_oclasses`       | (55,) uint8         | (55,) uint8        | OK    |
| `inv_strs`           | (55, 80) uint8      | (55, 80) uint8     | OK    |
| `screen_descriptions`| (21, 79, 80) uint8  | (21, 79, 80) uint8 | OK    |
| `tty_chars`          | (24, 80) uint8      | (24, 80) uint8     | OK    |
| `tty_colors`         | (24, 80) int8       | (24, 80) int8      | OK    |
| `tty_cursor`         | (2,) uint8          | (2,) uint8         | OK    |
| `misc`               | (3,) int32          | (3,) int32         | OK    |

Verdict: **all 17 observation channels match NLE shape AND dtype exactly.**
No dimension fixes required.

## 3. `inv_oclasses` value encoding

Source of truth: `vendor/nle/include/objclass.h:147-164`. NLE writes
`obj->oclass` (the `ObjectClass` enum) into `inv_oclasses[i]`; see
`vendor/nle/win/rl/winrl.cc:407-413` (`obs->inv_oclasses[i] = item.object_class;`
and trailing slots set to `MAXOCLASSES`).

| Class          | NLE enum value | ASCII sym (drawing.c) | Nethax `ItemCategory` value |
|----------------|----------------|------------------------|------------------------------|
| ILLOBJ_CLASS   | 1              | ']'                    | (no slot — illegal)          |
| WEAPON_CLASS   | 2              | ')'                    | `WEAPON = 2`                 |
| ARMOR_CLASS    | 3              | '['                    | `ARMOR  = 3`                 |
| RING_CLASS     | 4              | '='                    | `RING   = 4`                 |
| AMULET_CLASS   | 5              | '"'                    | `AMULET = 5`                 |
| TOOL_CLASS     | 6              | '('                    | `TOOL   = 6`                 |
| FOOD_CLASS     | 7              | '%'                    | `FOOD   = 7`                 |
| POTION_CLASS   | 8              | '!'                    | `POTION = 8`                 |
| SCROLL_CLASS   | 9              | '?'                    | `SCROLL = 9`                 |
| SPBOOK_CLASS   | 10             | '+'                    | `SPBOOK = 10`                |
| WAND_CLASS     | 11             | '/'                    | `WAND   = 11`                |
| COIN_CLASS     | 12             | '$'                    | `COIN   = 12`                |
| GEM_CLASS      | 13             | '*'                    | `GEM    = 13`                |
| ROCK_CLASS     | 14             | '`'                    | `ROCK   = 14`                |
| BALL_CLASS     | 15             | '0'                    | `BALL   = 15`                |
| CHAIN_CLASS    | 16             | '_'                    | `CHAIN  = 16`                |
| VENOM_CLASS    | 17             | '.'                    | `VENOM  = 17`                |
| MAXOCLASSES    | 18             | — (sentinel)           | `_MAXOCLASSES = 18` (empty)  |

References:
- `Nethax/nethax/subsystems/inventory.py:30-48` (`ItemCategory` enum)
- `Nethax/nethax/obs/nle_obs.py:982-999` (`build_inv_oclasses`)

Verdict: **Nethax emits raw enum integers, which is exactly what NLE
emits.** The task brief's claim that NLE writes ASCII symbol chars
(`)`, `[`, `!`, `?` ...) into `inv_oclasses` is incorrect. Those chars are
the `def_oc_syms` rendered for the *user-facing inventory UI*, but the
`inv_oclasses` observation buffer carries the **integer `oclass` enum**, not
the ASCII symbol. Encoding matches.

Empty-slot sentinel matches too: vendor uses `MAXOCLASSES = 18`, Nethax uses
`_MAXOCLASSES = 18`.

## 4. `screen_descriptions` content

Source of truth: `vendor/nle/win/rl/winrl.cc:491-516`
(`NetHackRL::store_screen_description`). For each visible tile it calls
`do_screen_description(cc, TRUE, sym, tmpbuf, &firstmatch, NULL)` and writes
the `firstmatch` string (NUL-terminated, 80-byte capped) into
`screen_descriptions[r, c, :]`. `firstmatch` is the generic name returned by
vendor `pager.c::do_screen_description` (e.g. `"human warrior"`, `"fountain"`,
`"orange dragon"`, `"floor of a room"`).

It is **not** the personalised lookat string like
`"Croesus, the Lord of the Vault"` — the task brief overstates this.
Personalised names come from `lookat`, not from `do_screen_description`.

Nethax (`Nethax/nethax/obs/nle_obs.py:847-866`, `_build_glyph_lookups` at
`431-540`) builds a static `glyph_id → ASCII bytes[80]` table at module load,
sourced from:

- monsters: `MONSTERS[i].name` (e.g. `"orange dragon"`)
- pets / detected / ridden: same name as the base monster
- bodies (corpses): `f"{m.name} corpse"`
- invisible monster glyph: `"invisible creature"`
- objects: `OBJECTS[i].name`
- cmap (terrain): hard-coded names ("wall", "doorway", "open door",
  "fountain", "altar", "staircase up", "molten lava", trap names, ...) — see
  `cmap_desc` at lines 454-477.
- unknown glyphs (NO_GLYPH and unmapped slots): all-zero 80-byte rows
  (matches vendor's `strncpy(.., "", 80)` for unknown tiles).

Verdict: **Nethax emits real ASCII bytes**, not empty bytes. Coverage is
close to vendor for static glyph → generic-name mapping.

### Known content gaps vs vendor (non-dim, non-encoding — informational only)

These do not affect shape/dtype contract; they are accuracy gaps in the
`firstmatch` text. A trained NLE agent that *reads* screen_descriptions as
strings could perform worse on these:

- **Stateful suffixes**: vendor pager appends `" (pet)"`, `" (tame)"`,
  `" (peaceful)"`, `" (asleep)"`, etc. via lookat fields — Nethax static
  table produces only the bare monster name for pet/detected/ridden glyphs.
- **Corpse age / freshness adjectives**: vendor sometimes prefixes with
  age (`"old human corpse"`, `"very old ..."`); Nethax emits
  `"<name> corpse"` only.
- **Object qualifiers**: vendor emits the *currently-known* shuffled
  appearance for unidentified scrolls/potions/wands (`"ZELGO MER scroll"`,
  `"swirly potion"`). Nethax `OBJECTS[i].name` already encodes the per-
  episode appearance via `_build_glyph_lookups` since OBJECTS is rebuilt
  each reset — verify in the wider obs review.
- **Cmap details**: vendor `do_screen_description` can distinguish
  `"open door"` vs `"broken door"` and gives `"web"` vs `"spider web"`
  depending on cmap index. Nethax cmap_desc table is coarse but covers the
  main 64-slot cmap range.

These are content-fidelity items, not shape/encoding bugs.

## 5. Result

- Shape/dtype contract: **clean, no fix needed.** All 17 channels match
  `(ROWNO, COLNO-1)`, `(NLE_TERM_LI, NLE_TERM_CO)`, and inventory sizes
  exactly.
- `inv_oclasses` encoding: **clean** — raw `ObjectClass` enum 2..17, with
  `MAXOCLASSES = 18` as the trailing-slot sentinel, mirroring vendor.
- `screen_descriptions`: **non-empty real ASCII bytes** sourced from
  MONSTERS/OBJECTS/cmap_desc; content fidelity vs vendor `firstmatch` is a
  separate accuracy concern (see gaps above) but is out of scope for this
  dims audit.

No code change is required. This is documentation-only.
