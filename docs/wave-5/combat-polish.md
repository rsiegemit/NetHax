# Wave 5 — Combat Polish

Four deliverables in `subsystems/combat.py`, all wired through `env.step` via the bump-attack bridge and the two new action handlers (`Command.TWOWEAPON`, `Command.THROW`).

## Per-slot armor AC bonus

Vendor source: `vendor/nethack/src/do_wear.c::Armor` table.

Vendor stores the per-slot AC bonus on each armor object's `a_can` field
plus its `subtyp` (helmet / shield).  We capture this on
`InventoryState.worn_armor` as a cached int8 per slot.

The cache lives on `InventoryState` rather than being recomputed every
step:

- helmet small  →  −1 AC
- helmet medium →  −1 AC
- helmet large  →  −2 AC
- shield small  →  −1 AC
- shield medium →  −1 AC
- shield large  →  −2 AC

Total AC = `BASE_AC` − Σ(worn armor AC) − Dex bonus.  Citation:
`vendor/nethack/src/worn.c::find_ac`.

## Two-weapon combat

Vendor source: `vendor/nethack/src/wield.c::dotwoweapon`,
`vendor/nethack/src/uhitm.c::known_hitum` (the two-strike path).

`subsystems/combat.py::handle_twoweapon` toggles
`state.combat.two_weapon` on/off.  When set:

1. `bump_attack` rolls a second strike with the off-hand weapon, using
   the same to-hit pipeline but no skill bonus and `−2` for the off-hand.
2. Both strikes share the same RNG sub-key (via `jax.random.split`).
3. `combat.melee_attack` branches on `two_weapon` via `lax.cond` so the
   pytree shape stays identical.

## Thrown attack

Vendor source: `vendor/nethack/src/dothrow.c::throwit`,
`vendor/nethack/src/dothrow.c::ohitmon`.

`subsystems/combat.py::thrown_attack` and the action handler
`handle_throw`:

1. Pulls the first valid item from `state.inventory.quiver` (Wave 5
   default: slot 0; full quiver wiring is Wave 6).
2. Walks the trajectory 8 tiles in the player's facing direction
   (`lax.fori_loop` over a 12-step bounded arc).
3. On first occupied tile: rolls a to-hit, applies damage if hit.
4. Land tile: pushes the item to the ground stack at that tile (the
   "lodge in target" path is Wave 6).

## Polymorph + combat integration

Vendor source: `vendor/nethack/src/polyself.c::mhitm`.

`bump_attack` is enriched: when `state.polymorph.is_polymorphed` is
True, it reads `polymorph.attack_*` (the NATTK=6 attack table that
was already populated by `polymorph.polymorph_player`).  Each attack
slot becomes a separate strike in a `lax.scan` of length 6, with
unused slots (attack[0]==0) becoming no-ops.

## Test coverage

`tests/test_combat_polish.py`:

1. per-slot helmet AC application
2. per-slot shield AC application
3. AC composition (helmet + shield + body) sums correctly
4. `handle_twoweapon` toggle flips `state.combat.two_weapon`
5. two-weapon dispatched once via `env.step` does not crash
6. `thrown_attack` against an adjacent monster does damage
7. thrown-weapon lands on the trajectory's last empty floor tile
8. polymorphed-player bump-attack uses `polymorph.attack_*` damage
9. polymorph + two-weapon together compose without shape drift

All 9 pass.

## Bump-attack bridge

Separate from this file but tightly coupled: `subsystems/action_dispatch.py::_try_step` now consults `state.monster_ai.alive` at the target tile and, if a live hostile is present, routes through `combat.bump_attack` instead of refusing the move.  This is the Wave-5 fix for the previously-skipped `test_walk_through_door_via_dispatch` and `test_player_kills_monster_via_bump_dispatch`.

## Citations

- `vendor/nethack/src/uhitm.c::known_hitum`  — main melee path.
- `vendor/nethack/src/do_wear.c::Armor`      — per-slot AC table.
- `vendor/nethack/src/dothrow.c::throwit`    — throw arc.
- `vendor/nethack/src/wield.c::dotwoweapon`  — toggle handler.
- `vendor/nethack/src/polyself.c::mhitm`     — polymorph attack handoff.
- `vendor/nethack/src/hack.c::domove`        — bump-attack bridge spec.
