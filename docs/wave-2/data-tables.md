# Wave 2 — Data tables

Full canonical NetHack 3.7 monster and object tables ported from C macro tables into Python dataclass literals.

## Monsters

**`Nethax/nethax/constants/monsters.py`**: schema + aggregation.
**`Nethax/nethax/constants/monster_entries/chunk{1..6}.py`**: 6 chunk files populated by parallel sub-agents.

| Chunk | Source range (monsters.h) | Entries | Coverage |
|---|---|---|---|
| chunk1 | lines 89–648 | 65 | Ants, blobs, cockatrices, canines, dogs, eyes, felines, gremlins, gargoyles, hobbits through master mind flayers, manes through tengu, jellies, kobolds, leprechaun, small mimic |
| chunk2 | lines 649–1275 | 65 | Kobold shaman through orc shaman, piercers, rothe through mastodon, rats, spiders, lurkers, unicorns, vortices, worms, grid bug / xan, lights, zruty, couatl / Aleax / Angel / ki-rin / Archon, bats |
| chunk3 | lines 1276–1916 | 62 | Big bat through eels and remaining mid-game monsters |
| chunk4 | lines 1917–2537 | 64 | Mummies, nagas, ogres, puddings, quantum mechanics, rust monsters, snakes, trolls, umber hulks, vampires (vampire mage `#if 0`), wraiths, xorn, apelike beasts, zombies, golems |
| chunk5 | lines 2538–3212 | 65 | Final golems, humans / elves / weres, ghosts, demons / devils / Riders, djinni, jellyfish — includes Charon (`#ifdef CHARON`) and mail daemon (`#ifdef MAIL_STRUCTURES`) since we don't `#if 0` them |
| chunk6 | lines 3213–3915 | 69 | Sea monsters, lizards, long worm tail, 13 character-class hero monsters, 13 quest leaders, 12 quest nemeses, 14 quest guardians |
| **Total** | — | **390** | — |

**Canonical NLE NUMMONS = 381.** Our 390 includes 2 conditional entries (Charon + mail daemon) and ~7 entries that may correspond to `#if 0` blocks we erred toward including. Wave 6 polish can trim to 381 exactly.

### Schema

```python
@struct.dataclass
class MonsterEntry:
    name: str
    symbol: MonsterSymbol         # IntEnum from monsym.h
    level: int                    # HD (hit dice)
    move_speed: int               # moves per 12 player moves
    ac: int
    mr: int                       # magic resistance %
    alignment: int
    generation_mask: int          # G_* flags (G_GENO, G_SGROUP, G_UNIQ, etc.)
    attacks: Tuple[Attack, ...]   # up to 6 (AT_*, AD_*, dice, sides) tuples
    weight: int
    nutrition: int                # corpse nutrition
    sound: int                    # MS_*
    size: int                     # MZ_*
    resists_mask: int             # MR_*
    conveys_mask: int             # intrinsics from corpse
    flags1: int                   # M1_*
    flags2: int                   # M2_*
    flags3: int                   # M3_*
    color: int                    # CLR_*
```

### Notable inclusions

- All **uniques** (`G_UNIQ`): Wizard of Yendor, Asmodeus, Baalzebub, Orcus, Juiblex, Demogorgon, Yeenoghu, Geryon, Dispater, Pestilence, Famine, Death, Croesus, Medusa, Oracle, Vlad the Impaler, Wizard of Yendor, Cthulhu (placeholder), and the 13 quest leaders / 12 quest nemeses.
- All 7 **dragons** (cardinal + chromatic) with breath-weapon attacks.
- All 9 **demons** and 4 demon lords.
- All 5 **ghost / shade** monsters.

### Notable exclusions

- `vampire mage` — guarded by `#if 0 /* DEFERRED */` in vendor source.
- `beholder` — same.
- 4 entries in `#if 0` blocks in chunk6 (Earendil, Elwing, Goblin King, High-elf).

---

## Objects

**`Nethax/nethax/constants/objects.py`**: schema + 134 inline entries (`OBJECTS_BASE`) + aggregation block.
**`Nethax/nethax/constants/object_entries/*.py`**: 9 chunk files.

| File | Source category | Entries | NLE canonical names? |
|---|---|---|---|
| `OBJECTS_BASE` (inline in `objects.py`) | weapons / armor / potions / scrolls / wands (legacy population from earlier passes) | 134 | Verbose: "potion of healing", "scroll of identify", "wand of striking" |
| `rings_amulets.py` | RING + AMULET macros | 41 | Canonical |
| `spellbooks.py` | SPELL macro | 42 | Canonical (42 active + 2 `#if 0` deferred) |
| `tools.py` | TOOL + CONTAINER + INSTRUMENT | 50 | Canonical |
| `food_gems.py` | FOOD + GEM + ROCK | 68 | Canonical |
| `specials.py` | OBJECT() direct calls — coins, balls, chains, venoms, Amulet of Yendor, Candelabrum, Bell, Book of the Dead, boulder, statue | 12 | Canonical |
| `weapons_extra.py` | 29 missing weapons (boomerang, scimitar, halberd, polearms, etc.) | 29 | Canonical |
| `armor_extra.py` | 60 missing armor (helms, all 22 dragon scale variants, cloaks, shields, gloves, boots) | 60 | Canonical |
| `misc_gaps.py` | 16 GENERIC placeholders + 26 raw-named potions + 23 raw-named/missing scrolls + 25 raw-named/missing wands | 91 | Canonical (raw) |
| **Aggregated `OBJECTS`** | dedup by `(name, class_)` | **503** | Mixed |

**Canonical NLE NUM_OBJECTS = 453.** Our 503 over-counts by ~50 due to **dual naming**: `OBJECTS_BASE` has "potion of healing" while `misc_gaps` has "healing" (canonical). Both survive dedup because the names differ. Wave 3+ should canonicalize to NLE bare names (preferring `misc_gaps` versions) and drop the verbose forms.

### Class breakdown

| Class | Count | NLE canonical |
|---|---|---|
| ILLOBJ_CLASS | 1 | 1 |
| WEAPON_CLASS | 72 | ~70 |
| ARMOR_CLASS | 87 | ~80 |
| RING_CLASS | 29 | 28 |
| AMULET_CLASS | 14 | 13 |
| TOOL_CLASS | 51 | ~70 |
| FOOD_CLASS | 33 | ~33 |
| POTION_CLASS | 43 | 26 (overcount from dual naming) |
| SCROLL_CLASS | 35 | 23 (overcount from dual naming) |
| SPBOOK_CLASS | 44 | 46 |
| WAND_CLASS | 45 | 28 (overcount from dual naming) |
| COIN_CLASS | 2 | 1 |
| GEM_CLASS | 37 | ~33 |
| ROCK_CLASS | 3 | 5 |
| BALL_CLASS | 2 | 1 |
| CHAIN_CLASS | 2 | 1 |
| VENOM_CLASS | 3 | 2 |
| **Total** | **503** | **~453** |

Categories that **underperform**: TOOL (51 / 70 — missing instruments, traps-as-tools, some containers).

### Schema

```python
@struct.dataclass
class ObjectEntry:
    name: str
    symbol: str            # single-char class symbol
    class_: ObjectClass    # IntEnum (WEAPON_CLASS, ARMOR_CLASS, etc.)
    prob: int              # spawn probability
    weight: int
    cost: int              # base gold cost
    sdam: Tuple[int, int]  # small-target damage (n_dice, n_sides)
    ldam: Tuple[int, int]  # large-target damage
    oc1: int               # class-specific flag (e.g., AC for armor, school for spell)
    oc2: int               # secondary flag (e.g., spell level)
    nutr: int              # nutrition (food/corpses)
    color: int             # CLR_*
    material: int          # MAT_*
```

### Notable additions Wave 2

- **All 22 dragon scale variants** (11 colors × scale-mail + scales raw form).
- **All 7 special objects** (Amulet of Yendor, Candelabrum of Invocation, Bell of Opening, Book of the Dead, Heavy Iron Ball, boulder, statue).
- **All 16 GENERIC placeholders** (`generic strange`, `generic weapon`, etc.) — these correspond to `OBJECT(OBJ("generic X", "X"), BITS(...), ...)` entries in objects.h used as sentinels.
- **All 42 spellbooks** at one level (next to one inline `Book of the Dead` special).
- **Both naming conventions for potions/scrolls/wands** — verbose form (`"potion of healing"`) and canonical NLE bare form (`"healing"`). Will canonicalize Wave 3.

### Decisions

- **Dedup key**: `(name, int(class_))`. Bare-name vs prefixed-name entries are NOT considered duplicates (intentional in Wave 2; cleanup deferred).
- **Color aliases** (`HI_COPPER`, `HI_GOLD`, `HI_PAPER`, `HI_MINERAL`, `HI_METAL`, `HI_LEATHER`, `HI_CLOTH`, `HI_ORGANIC`, `HI_SILVER`, `HI_WOOD`) resolved to their `CLR_*` equivalents per `vendor/nethack/include/color.h`.
- **Material constants** (`MAT_*`) — many chunks defined them locally with vendor citations; `objects.py` does not centralize them yet.

---

## How to query

```python
from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass

# All gnomes
gnomes = [m for m in MONSTERS if m.symbol == MonsterSymbol.S_GNOME]

# All cursed-only items (none yet — BUC isn't stored at the class level)
# All scroll-class objects
scrolls = [o for o in OBJECTS if o.class_ == ObjectClass.SCROLL_CLASS]
```

---

## What's still missing

- ~19 tools (instruments, traps-as-tools, magical drum / horn variants)
- ~2 spellbooks (`flame sphere`, `freeze sphere` — both `#if 0` deferred in vendor)
- The dual-naming dedup pass for potions/scrolls/wands.

These are tracked in `docs/wave-2/gaps.md`.
