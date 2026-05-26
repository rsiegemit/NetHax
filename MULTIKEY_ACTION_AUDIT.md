# Multi-Key Action Support Audit (Nethax)

Date: 2026-05-26
Scope: Audit whether Nethax supports vendor-NetHack's two-step action protocol
(command key + follow-up letter / direction), e.g. `W` then `b` to wear the
armor in slot `b`.

## Findings

### Step 2 — `action_dispatch.py`

All inventory-consuming command handlers in
`Nethax/nethax/subsystems/action_dispatch.py` are **single-step** wrappers
that take only `(state, rng)` and auto-pick the first valid inventory slot
via a `lax.scan` / `jnp.argmax` over `inventory.items.category`. No handler
accepts a follow-up letter key.

Concrete evidence (line refs in `action_dispatch.py`):

- `_handle_eat` (1577): scans `category == FOOD`, picks `jnp.argmax(valid)`.
- `_handle_quaff` (1693): direct delegate to `items_potions.handle_quaff(state, rng)`.
- `_handle_read`  (1698): direct delegate to `items_scrolls.handle_read(state, rng)`.
- `_handle_zap`   (1703): projects `WandState`, calls
  `items_wands.handle_zap(state, rng, direction=0)` — direction is a kwarg
  default, not a follow-up action key from the env.
- `_handle_cast`  (1848): `jnp.argmax` over `known & (memory > 0)`.
- `_handle_wield` (1937): direct delegate → `inventory.handle_wield` which
  scans for the first `WEAPON` slot.
- `_handle_wear`  (1942): direct delegate → `inventory.handle_wear` which
  scans for the first `ARMOR` slot, hardcoded `ArmorSlot.BODY`.
- `_handle_put_on` (1947): picks first RING or AMULET slot.
- `_handle_throw` (2518): docstring says "Throws the first quivered /
  weapon-class inventory item east (Wave 5 Phase 1 default direction)" —
  direction east is hardcoded.
- `_handle_apply` (2542): branches on `_has_digging_tool`; pickaxe dig
  defaults direction=0 (NORTH). Otherwise routes to container apply with
  no slot selection.
- `_handle_pray`  (2065): direct delegate.

Underlying handlers in subsystems confirm the same shape:

- `items_scrolls.handle_read(state, rng)`        — no slot arg.
- `items_potions.handle_quaff(state, rng)`       — no slot arg.
- `inventory.handle_wield(state, rng)`            — auto-picks first WEAPON.
- `inventory.handle_wear(state, rng)`             — auto-picks first ARMOR.
- `combat.handle_throw(state, rng)`               — first quivered/throw-cap,
  fixed east direction.

### Step 3 — `state.py` & subsystems

Grep results for `pending_action`, `pending_letter`, `prompt_buffer`,
`awaiting_`, `pending_slot`, `getobj_`, `two_step`, `multi_key`, `follow_up`
across `Nethax/nethax/state.py` and `Nethax/nethax/subsystems/`: **zero
matches**.

`state.py` also has no `throw_direction`, `zap_direction`, or similar
follow-up scratch field. There is no state machine for multi-key actions
at all — every `env.step(action)` call is treated as a complete, atomic
vendor command. The vendor's `getobj()` / `getdir()` prompt loop is not
modeled.

## Per-Action Status Table

| Action | Vendor prompt(s)            | Nethax behavior                                     | Status              |
|--------|-----------------------------|-----------------------------------------------------|---------------------|
| WEAR   | letter (armor slot)         | first ARMOR slot, hardcoded `ArmorSlot.BODY`        | auto-picks slot     |
| WIELD  | letter (weapon slot)        | first WEAPON slot                                   | auto-picks slot     |
| QUAFF  | letter (potion slot)        | first POTION (delegated to `items_potions`)         | auto-picks slot     |
| EAT    | letter (food slot)          | first FOOD slot with stock                          | auto-picks slot     |
| READ   | letter (scroll/book slot)   | first scroll/book (delegated to `items_scrolls`)    | auto-picks slot     |
| ZAP    | letter + direction          | first wand, direction=0 (north) hardcoded kwarg     | auto-picks slot, fixed direction |
| THROW  | letter + direction          | first quivered/throwable, direction east hardcoded  | auto-picks slot, fixed direction |
| APPLY  | letter (+ direction for dig)| pickaxe → dig NORTH; else first container/tool      | auto-picks slot, fixed direction |
| CAST   | spell letter + direction    | first known+memorized spell (`argmax`)              | auto-picks spell, no direction |
| PRAY   | y/n confirm                 | direct delegate, no confirmation                    | no-op confirm (works)|
| PUTON  | letter (ring/amulet)        | first RING or AMULET; chooses left ring then right  | auto-picks slot     |
| REMOVE | letter (worn)               | priority left-ring → right-ring → amulet            | auto-picks slot     |

Legend:
- **auto-picks slot** = vendor would prompt for a letter; Nethax skips the
  prompt and uses the first matching slot.
- **fixed direction** = vendor would prompt for a direction; Nethax uses a
  hardcoded compass default (THROW=east, ZAP=north, APPLY-dig=north).

No action is fully no-op; none currently supports a follow-up key.

## Design Proposal — `pending_action` State Machine

Goal: let an NLE-trained policy (which emits the two-step sequence
`Command.WEAR` then `TextCharacters` letter `b`) actually pick a specific
inventory slot in Nethax, matching vendor semantics.

### EnvState additions

```python
# state.py (added to EnvState pytree)
pending_action:    jnp.int8   # 0 = none, else a Command code awaiting a follow-up
pending_kind:      jnp.int8   # 0 = none, 1 = letter (slot), 2 = direction, 3 = y/n
pending_rng:       jax.Array  # PRNG split off when the first key arrived
```

All three are `int8 / PRNGKey`, JIT-safe, fixed dtype. `pending_action == 0`
indicates "no pending command".

### Dispatch rule

```
if state.pending_action != 0:
    follow_up_handler[state.pending_action](state, action_byte, rng)
    clear pending fields
else:
    if action is in {WEAR, WIELD, QUAFF, EAT, READ, ZAP, THROW, APPLY,
                     CAST, PUTON, REMOVE, FIRE, ...}:
        set pending_action = action; pending_kind = LETTER (or DIR for ZAP/THROW)
        return state unchanged  (vendor: print prompt, no turn cost)
    else:
        dispatch normally
```

For two-prompt actions like ZAP (`letter, direction`), encode a small
sub-state-machine: first follow-up sets `pending_kind = DIR` and stashes
the chosen slot in a new `pending_slot: int8` field; second follow-up
fires the effect.

### Slot resolution

The "letter" follow-up is an ASCII byte in `a..z, A..Z` mapping to
inventory index `0..51`. `slot = ord(letter) - ord('a')` for lowercase,
`slot = 26 + ord(letter) - ord('A')` for uppercase. Validate
`category[slot] != 0 & quantity[slot] > 0` and that the slot's category
matches the command (e.g. WEAR → ARMOR). On mismatch, return state
unchanged and clear pending (vendor would re-prompt; Nethax can simply
abort).

### Backward compatibility

A boolean env flag (`auto_pick_when_no_followup`) preserves the current
single-step behavior for legacy callers/tests: if set and the next step
arrives without a matching letter (e.g. an ESC or unrelated command),
fall back to the existing `argmax` auto-pick. Default OFF for NLE
training; default ON for existing Nethax tests so they keep passing
without modification.

### Migration plan

1. Add the three (later four) `pending_*` fields with default zeros — pure
   additive change.
2. Wire `dispatch_action` to consult `pending_action` first.
3. Implement WEAR follow-up as the prototype, then WIELD, QUAFF, EAT,
   READ, PUTON in order of frequency.
4. Extend with `pending_slot` to support ZAP+THROW (letter then direction).
5. Add tests: `WEAR + 'b'` selects slot 1; `WEAR + 'q'` (no armor at q) is
   a no-op turn; auto-pick legacy path remains under the flag.
