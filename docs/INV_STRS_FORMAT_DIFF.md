# inv_strs Format Diff: NLE vs Nethax

Audit of byte-level divergences between NLE's `inv_strs[55][80]` observation
and Nethax's `build_inv_strs` renderer (`Nethax/nethax/obs/inv_strs.py`).

Canonical references:
- `vendor/nle/win/rl/winrl.cc::observation_inv` lines 382-397
  (writes `obs->inv_strs[i] = item.str[j]`)
- `vendor/nle/win/rl/winrl.cc::update_inventory_method` lines 444-462
  (`item.str = doname(otmp)`)
- `vendor/nethack/src/objnam.c::doname_base` lines 1221-1754

## Byte-for-byte comparison: Valkyrie starting long sword

Scenario: a freshly-rolled Valkyrie has invent slot 0 = uncursed identified
long sword, `obj->known=1`, `obj->bknown=0` (Val starts with bknown=0 by
default in nethack 3.6+), `obj->spe=0`, wielded.

### NLE vendor output (`item.str` payload only)

```
"a +0 long sword (weapon in hand)\0\0...\0"
```

Construction (vendor):
- `xname(obj)` → `"long sword"`            (objnam.c:836)
- `prefix` starts as `"a "`               (objnam.c:1298)
- bknown branch skipped (`bknown==0`)     (objnam.c:1318)
- `add_erosion_words` → no-op             (objnam.c:1156, oeroded==0)
- `known` ⇒ `Sprintf(eos(prefix),"%+d ",obj->spe)` → `"a +0 "`  (objnam.c:1423)
- WEAPON case appends `" (weapon in hand)"`              (objnam.c:1591)
- final assembly `bp = strprepend(bp, prefix)` ⇒ `"a +0 long sword (weapon in hand)"`

Bytes (33 bytes + NULs to 80):
```
'a',' ','+','0',' ','l','o','n','g',' ','s','w','o','r','d',' ',
'(','w','e','a','p','o','n',' ','i','n',' ','h','a','n','d',')','\0',…
```

### Nethax `build_inv_strs` output (row 0)

```
"a - a +0 long sword (weapon in hand)\0...\0"
```

Construction (Nethax `_render_slot`):
- Step 1: `"<letter> - "` (`a`, `' '`, `'-'`, `' '`)   (inv_strs.py:866-870)
- Step 2: `_write_article_space` → `"a "`              (inv_strs.py:876-881)
- Step 3: `_write_buc` skipped (bknown=False)
- Step 3b: erosion skipped
- Step 4: `_write_enchant` → `"+0 "`                    (inv_strs.py:1097)
- Step 5: `_write_true_name` → `"long sword"`           (inv_strs.py:1149)
- Step 6: `_write_equip` → `" (weapon in hand)"`        (inv_strs.py:1310)

Bytes (37 bytes + NULs):
```
'a',' ','-',' ',  'a',' ','+','0',' ','l','o','n','g',' ','s','w','o','r','d',' ',
'(','w','e','a','p','o','n',' ','i','n',' ','h','a','n','d',')','\0',…
```

**Δ = 4 extra leading bytes (`"a - "`)** in Nethax versus the vendor payload.

## Top divergences

### 1. `"<letter> - "` (4-byte) menu-selector prefix is included in Nethax

- **NLE**: `inv_strs[i]` is the raw `doname(otmp)` output. The invlet lives in
  `inv_letters[i]` (see `winrl.cc:398-407`). Vendor never emits `"x - "` into
  the doname buffer; that prefix is produced by tty/curses windowing code at
  display time (`outsel.c`, `windows.c`).
- **Nethax**: `_render_slot` step 1 (`inv_strs.py:865-870`) writes
  `"<letter> - "` (4 bytes) at cursor 0 for every non-empty slot.
- **Status**: project-wide convention; many tests assert `s.startswith("a - ")`
  (e.g. `tests/test_inv_strs.py:158, 287, 401`; `tests/test_inv_strs_polish.py:142`).
  Fixing breaks the test suite. Documented divergence — **NOT** a byte-for-byte
  candidate to fix without a coordinated migration of `inv_strs` consumers and
  tests.

### 2. Slot ordering: by invlet-index vs invent linked-list order

- **NLE**: `inv_strs` rows are packed in invent linked-list traversal order
  (`for (otmp = invent; otmp; otmp = otmp->nobj)`, `winrl.cc:456`). Rows beyond
  the actual item count are zero-filled. The invlet is delivered via
  `inv_letters[i]` and can be in any A-Z/a-z order.
- **Nethax**: `build_inv_strs` calls `lax.map(render_one, jnp.arange(55))`
  (`inv_strs.py:1399`). Row `i` corresponds to slot `i` (`a`=0, …, `z`=25,
  `A`=26, …, `Z`=51) — not packed and not in linked-list order.
- **Impact**: For an agent observing only `inv_strs[i]` (or zipping with
  `inv_letters[i]`), this produces structurally different observations once an
  item is dropped/picked-up out of letter order.

### 3. Missing prefix segments from vendor `doname`

Several vendor prefix tokens are **never** emitted by Nethax:

| Vendor token (objnam.c)          | Condition                                            | Nethax |
|----------------------------------|------------------------------------------------------|--------|
| `"empty "`           (line 1316) | container with no contents (or empty bag-of-tricks)  | omitted |
| `"trapped "`         (line 1357) | `Is_box(obj) && obj->otrapped && obj->tknown`        | omitted |
| `"broken "/"locked "/"unlocked "` (1359-1367) | `lknown && Is_box(obj)`                  | omitted |
| `"greased "`         (line 1371) | `obj->greased`                                       | omitted |
| `" containing N item(s)"` (1379) | `cknown && Has_contents(obj)`                        | omitted |
| `"poisoned "`        (line 1420) | `WEAPON_CLASS && ispoisoned`                         | omitted |
| `"partly eaten "`    (line 1506) | `FOOD_CLASS && obj->oeaten`                          | omitted |
| `"diluted "`         (line 833)  | POTION_CLASS && dknown && odiluted                   | omitted |
| `<pmname> " "`       (1530-1532) | known/MV_KNOWS_EGG egg with monster type             | omitted |

These are subsystem-feature gaps (containers, traps, glob, eggs, eat hooks,
poisoned weapons), not formatting bugs; each requires the corresponding state
fields to exist before rendering.

### 4. `"uncursed "` is always emitted when `bknown` — no implicit-uncursed gate

- **Vendor** objnam.c:1328-1348: `flags.implicit_uncursed` (default True) plus
  identified weapon/armor/ring or oc_charged items **suppress** the
  `"uncursed "` qualifier (because the +/− or `(n:m)` already implies BUC).
- **Nethax**: `_write_buc` is called whenever `buc_known & buc_status==UNCURSED`.
  Vendor's "would-print-charges-anyway" gate is absent.

### 5. `GemStone` / `MINERAL` rock suffix not emitted for GEM_CLASS

- **Vendor** xname GEM_CLASS path (objnam.c:914-928):
  - identified gemstones: `Strcpy(buf, actualn); if (GemStone(typ)) Strcat(buf, " stone");`
    → `"diamond stone"` for an identified diamond.
  - unidentified rocks (`oc_material == MINERAL`): `Sprintf(buf, "%s %s", dn, "stone")`
    → `"gray stone"` not `"gray gem"`.
- **Nethax**: `_CLASS_NOUN_STRS[GEM_CLASS] = " gem"` (no `" stone"` variant);
  identified path writes only the canonical name (no GemStone suffix).
- Affects 5 MINERAL gem objects (touchstone, flint, luckstone, loadstone,
  rock) and ~10 identified gemstones (diamond, ruby, sapphire, …).

### 6. SPE_BOOK_OF_THE_DEAD: vendor omits the `"spellbook of "` prefix

- Vendor objnam.c:896-897: `if (typ != SPE_BOOK_OF_THE_DEAD) Strcpy(buf, "spellbook of ");`
- Nethax `_CLASS_PREFIX_STRS[SPBOOK_CLASS] = "spellbook of "` is applied to
  every identified spellbook → `"spellbook of Book of the Dead"` instead of
  just `"Book of the Dead"`.

### 7. Charges format: `"(n:m)"` vs `" (n:m)"`

- Both use the same `" (recharged:charges)"` format with a leading space.
- Verified byte-equal: `_write_charges` in inv_strs.py:1341-1354.
  **No divergence here.**

### 8. Erosion: separate `"very "`/`"thoroughly "` words vs pre-combined

- Vendor (objnam.c:1156-1164) emits `"very "` then `"rusty "` as two `Strcat` calls.
- Nethax (`_EROSION_OERODED_STRS`) stores `"very rusty "` as one row.
- Resulting bytes are identical when concatenated.
  **No divergence.**

### 9. `add_erosion_words` ordering relative to enchant

- Vendor: erosion words are appended to `prefix` *before* `"%+d "` enchant
  (objnam.c:1421 then 1423). Result order: `"rusty +0 long sword"`.
- Nethax: step 3b erosion → step 4 enchant. Same order.
  **No divergence.**

### 10. `"a"`/`"an"` article: BUC-prefix overrides item-name first letter

- Vendor: `just_an(prefix, *tmpbuf ? tmpbuf : bp)` (objnam.c:1690) — when a BUC
  word precedes, the article check uses the BUC word's first letter. All BUC
  words start with consonants ⇒ always `"a "`.
- Nethax: `article_use_an = jnp.where(buc_known, False, noun_use_an)`
  (inv_strs.py:860). Equivalent (BUC always forces `"a "`).
  **No divergence.**

### 11. COIN_CLASS BUC suppression (fixed in this audit)

- Vendor objnam.c:1318: `if (bknown && obj->oclass != COIN_CLASS && …)` —
  coins are never prefixed with `"blessed/cursed/uncursed"`.
- Nethax (pre-fix): `show_buc = buc_known & ~is_water_special`. No COIN gate.
- **Fix applied**: `show_buc = buc_known & ~is_water_special & ~is_coin`
  (inv_strs.py step 3 in `render_nonempty`).
