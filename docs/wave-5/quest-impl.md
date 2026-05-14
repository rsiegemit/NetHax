# Wave 5 — Quest Implementation

The Quest branch was a single-role guardian in Wave 4.  Wave 5 expands
it to 13 distinct role-themed layouts plus per-role artifact / leader / nemesis / prefix tables.

## Per-role table

Source: `vendor/nethack/src/role.c::roles[]` (the Role struct).
Implemented in `Nethax/nethax/subsystems/quest.py::QUEST_TABLE`.

| Role | Artifact | Leader | Nemesis | Prefix |
|---|---|---|---|---|
| Archeologist | The Orb of Detection | Lord Carnarvon | Minion of Huhetotl | "Digger" |
| Barbarian | The Heart of Ahriman | Pelias | Thoth Amon | "Plunderer" |
| Caveman | The Sceptre of Might | Shaman Karnov | Chromatic Dragon | "Troglodyte" |
| Healer | The Staff of Aesculapius | Hippocrates | Cyclops | "Rhizotomist" |
| Knight | The Magic Mirror of Merlin | King Arthur | Ixoth | "Gallant" |
| Monk | The Eyes of the Overworld | Grand Master | Master Kaen | "Candidate" |
| Priest | The Mitre of Holiness | Arch Priest | Nalzok | "Aspirant" |
| Ranger | The Longbow of Diana | Orion | Scorpius | "Tenderfoot" |
| Rogue | The Master Key of Thievery | Master of Thieves | Master Assassin | "Footpad" |
| Samurai | The Tsurugi of Muramasa | Lord Sato | Ashikaga Takauji | "Hatamoto" |
| Tourist | The Platinum Yendorian Express Card | Twoflower | Master of Thieves | "Rambler" |
| Valkyrie | The Orb of Fate | Norn | Lord Surtur | "Stripling" |
| Wizard | The Eye of the Aethiopica | Neferet the Green | Dark One | "Evoker" |

## Per-role layout (simplified-iconic)

Source: `vendor/nethack/dat/qst*.lua`.  Each vendor `.lua` is ~120 lines
of `MAP { … }` plus monster / object placements.

Wave 5 **hand-translated** the iconic features of each layout rather
than parsing the full `.lua`.  Decision rationale: vendor's quest
levels are highly visual (the Archeologist mines temple is recognisably
a stepped pyramid, the Wizard library has stack walls of bookshelves),
and parser fidelity matters less than visual recognisability.

This is the same trade-off made by the MiniHack des-file parser
(which IS a full parser) for the canonical 36 MiniHack envs — but
Quest layouts are 13, are larger, and are not re-used by RL benchmarks,
so the cost-benefit favours hand-translation.

A future Wave 6 polish pass can replace these with full-fidelity
`qst*.lua` parses; the per-role function signatures are stable so the
swap is local.

## Nemesis fight mechanics

`Nethax/nethax/subsystems/quest.py::nemesis_fight`:

- Nemesis spawns on the deepest quest level (Quest L5 in our schema).
- Per-role HP: 200 + 20 × `role_difficulty`.
- Regen: 5 HP per turn if the player is more than 4 tiles away.
- On death, the nemesis drops the role artifact at the death tile.
- Conduct: nemesis kill does NOT violate PACIFIST (vendor treats
  nemesis as a forced encounter, exempt from conduct counts —
  `vendor/nethack/src/insight.c::record_achievement`).

## Return-to-leader victory

`Nethax/nethax/subsystems/quest.py::return_to_leader`:

When the player returns to the leader-room (Quest L1) carrying the
role artifact:

1. Sets `state.quest.quest_completed = True`.
2. Adds a small score bonus.
3. The leader monster becomes peaceful and grants a wish (Wave 5
   simplification: just sets a `wish_granted` flag — actual wish
   handler is Wave 6).

## Citations

- `vendor/nethack/src/role.c::roles[]` — per-role data.
- `vendor/nethack/src/quest.c::qt_msg` — leader/nemesis dialogue (Wave 6).
- `vendor/nethack/dat/qst*.lua` — layouts.
- `vendor/nethack/src/qst.c::onquest` — quest-progression state machine.

## Test coverage

`tests/test_quest.py` — 15 tests:

1-13. Each role's `generate_*_quest_level` produces non-zero terrain.
14. Per-role layouts differ (no two roles produce the same terrain hash).
15. `dispatch_quest_level(role)` switch dispatches correctly.

Plus `test_quest_dispatch_returns_role_specific_layout` in
`tests/test_wave5_integration.py`.

## Wave 6 follow-ups

- Full vendor `qst*.lua` parser pass.
- Leader / nemesis dialogue from `qt_msg`.
- Quest-progression state machine (`onquest`).
- Wish handler triggered by return-to-leader.
- Per-role intrinsic gains on quest completion.
