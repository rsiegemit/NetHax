# Wave 5 — Special Levels: Full Inventory

Wave 5 brings the count from **4** (Wave 4) to **35+** unique level factories.

All factories return the same triple `(terrain, monsters, items)`:
- `terrain` : `int8[MAP_H=21, MAP_W=80]` matching the `TileType` enum.
- `monsters`: `int16[64, 3]` packed `(row, col, type_id)` triples.
- `items`   : `int16[64, 3]` packed `(row, col, object_id)` triples.

Padding rows are `(-1, -1, -1)`.

---

## Wave 4 set (unchanged)

| Factory | Module | Notes |
|---|---|---|
| `generate_oracle_level` | `special_levels.py` | Delphi + 4 fountains.  Citation: `vendor/nethack/dat/oracle.lua`. |
| `generate_minetown_level` | `special_levels.py` | Shops + altar + watchmen + priest.  Citation: `vendor/nethack/dat/minetn-*.lua`. |
| `generate_minesend_level` | `special_levels.py` | Mine's End with Luckstone.  Citation: `vendor/nethack/dat/minend.lua`. |
| `generate_bigroom_level` | `special_levels.py` | Single 60x15 room.  Citation: `vendor/nethack/dat/bigroom.lua`. |

---

## Wave 5 — major special levels

| Factory | Module | Notes / Citation |
|---|---|---|
| `generate_castle_level` | `special_levels.py` | Drawbridge + 4 corner towers + wand of wishing.  `dat/castle.lua`. |
| `generate_vlad_lower` | `special_levels.py` | Lower tower; trapdoor + 8 vampires.  `dat/tower1.lua`. |
| `generate_vlad_middle` | `special_levels.py` | Mid tower; bone pile + 4 wraiths.  `dat/tower2.lua`. |
| `generate_vlad_top` | `special_levels.py` | Vlad's throne + Candelabrum of Invocation.  `dat/tower3.lua`. |
| `generate_wizard_tower` | `special_levels.py` | Real tower; Wizard of Yendor + book of the dead.  `src/wizard.c`. |
| `generate_wizard_fake_1` | `special_levels.py` | Decoy 1; fake wizard, no real items. |
| `generate_wizard_fake_2` | `special_levels.py` | Decoy 2; same shape, different mob density. |
| `generate_wizard_fake_3` | `special_levels.py` | Decoy 3; trap-heavy variant. |
| `generate_sanctum_level` | `special_levels.py` | Amulet of Yendor + high priest + 4 acolytes.  `dat/sanctum.lua`. |

15 dedicated tests in `tests/test_major_special_levels.py`.

---

## Wave 5 — demon lairs

All in `Nethax/nethax/dungeon/demon_lairs.py`.

| Factory | Theme | Citation |
|---|---|---|
| `generate_asmodeus_lair`   | Ice palace, cold-resist required | `dat/asmodeus.lua` |
| `generate_baalzebub_lair`  | Fire pillars + acid pits | `dat/baalzebu.lua` |
| `generate_juiblex_lair`    | Acid pits + slime mound | `dat/juiblex.lua` |
| `generate_orcus_lair`      | Skeleton hall + Wand of Death | `dat/orcus.lua` |
| `generate_yeenoghu_lair`   | War camp + flail tribe | `dat/yeenoghu.lua` |
| `generate_demogorgon_lair` | Twin-tower swamp + tentacle pool | `dat/demogorgon.lua` |

17 tests in `tests/test_demon_lairs.py`.

---

## Wave 5 — Valley of the Dead + Gehennom procedural

| Factory | Module | Notes |
|---|---|---|
| `generate_valley_of_dead` | `branches.py` | Gehennom L1: vibrating square + magic portal to ascend back to Castle. Citation: `dat/valley.lua`, `src/trap.c::TRAP_VIBRATING_SQUARE`. |
| `generate_gehennom_level(rng, depth)` | `branches.py` | 12 procedural levels (L2-L15 minus the 4 unique inserts at L1/L4/L8/L16). Procedural caverns + lava + occasional fire trap.  Citation: `src/mkmaze.c::mkgehennom`. |

8 tests in `tests/test_gehennom.py`.

---

## Wave 5 — Endgame (Astral planes)

All in `Nethax/nethax/dungeon/endgame.py`.

| Factory | Theme | Citation |
|---|---|---|
| `generate_earth_plane` | Caverns in solid rock | `dat/earth.lua` |
| `generate_air_plane`   | Almost no floor; flight needed | `dat/air.lua` |
| `generate_fire_plane`  | Lava lake with floor islands | `dat/fire.lua` |
| `generate_water_plane` | Pool everywhere + floor bubbles | `dat/water.lua` |
| `generate_astral_plane`| Open field + 3 altars (Lawful, Neutral, Chaotic) | `dat/astral.lua` |

`generate_endgame_level(rng, depth)` dispatches by `depth in 1..5`.

19 tests in `tests/test_endgame.py`.

---

## Wave 5 — Quest (13 role-specific)

In `Nethax/nethax/dungeon/quest_levels.py`:

| Role | Function | Theme |
|---|---|---|
| Archeologist | `generate_arc_quest_level` | Mines temple |
| Barbarian    | `generate_bar_quest_level` | Cave with thrones |
| Caveman      | `generate_cav_quest_level` | Tribal cave |
| Healer       | `generate_hea_quest_level` | Cave hospital |
| Knight       | `generate_kni_quest_level` | Tournament field |
| Monk         | `generate_mon_quest_level` | Monastery |
| Priest       | `generate_pri_quest_level` | Cathedral |
| Ranger       | `generate_ran_quest_level` | Forest |
| Rogue        | `generate_rog_quest_level` | Thieves' den |
| Samurai      | `generate_sam_quest_level` | Dojo |
| Tourist      | `generate_tou_quest_level` | Desert oasis |
| Valkyrie     | `generate_val_quest_level` | Mead hall |
| Wizard       | `generate_wiz_quest_level` | Library tower |

`dispatch_quest_level(rng, role)` switches on the role enum.

15 tests in `tests/test_quest.py`.

---

## Total: 35+ unique level factories

|  | Wave 4 | Wave 5 |
|---|---|---|
| Wave-4 set (Oracle, MineTown, MinesEnd, BigRoom) | 4 | 4 |
| Major special levels | 0 | 9 |
| Demon lairs | 0 | 6 |
| Valley + Gehennom procedural | 0 | 2 |
| Endgame planes | 0 | 5 |
| Quest (per role) | 0 | 13 |
| **Total** | **4** | **39** |

Plus the procedural mines + sokoban + main generators inherited from Wave 4.

---

## Citation pattern

Every layout cites the canonical vendor source line in its module docstring.  Examples:

```python
def generate_castle_level(rng):
    """Castle (Dlvl 26).

    Citation: vendor/nethack/dat/castle.lua
              vendor/nethack/src/mkmap.c::mkcastle
    """
```
