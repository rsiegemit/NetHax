# Skill-Action Runtime Message/Effect Gaps (MiniHack)

Read-only audit of the Nethax (NetHack -> JAX) runtime. For each MiniHack skill
env category the RewardManager (RM) is correctly wired, but a reward only fires
if the game engine **applies the effect** AND **emits the exact vendor message**
the RM listens for at runtime.

Status legend: YES / NO / partial. "Emits msg?" means the action handler pushes
the listened-for string through the messages subsystem
(`Nethax/nethax/subsystems/messages.py::emit`). The messages subsystem can only
emit strings that exist as a `MessageId` template
(`_MESSAGE_TEMPLATES`, ids 0-42); none of the missing strings below are baked,
so even a `emit()` call could not produce them today.

## Per-action results

| Env | RM listens for | Handler | Handler exists? | Applies effect? | Emits msg? | Exact vendor string + source |
|---|---|---|---|---|---|---|
| Eat-* | `add_eat_event("apple")` | `status_effects.py:1153 handle_eat` | YES | YES (nutrition+hunger) | **NO** | `"This apple tastes delicious!"` — vendor `eat.c:1709-1720` (`pline("%s%s %s %s%c", "This ", pmxnam, "tastes", "delicious", '!')`). Generic food path uses `EAT_FOOD`="You eat the food." (`eat.c::eatcorpse`) but `handle_eat` never emits even that. |
| Wield-* | `add_wield_event("dagger")` | `inventory.py:1545 handle_wield` -> `wield` | YES | YES (sets `wielded`) | **NO** | `"d - a dagger"` — vendor `wield.c:191 prinv((char*)0, wep, 0L)` -> `invent.c:2440 prinv` -> `xprname` produces `"<letter> - <doname>"`. |
| Wear-* | `add_wear_event("robe")` | `inventory.py:1593 handle_wear` -> `inventory.py:1358 wear_armor` | YES | YES (worn_armor+AC) | **NO** | `"You are now wearing a robe."` — vendor `do_wear.c:79 You("are now wearing %s%s.", an(otmp_name), how)`. |
| PutOn-* | `add_amulet_event()` | `items_jewelry.py:529 wear_amulet` | YES | YES (worn_amulet+intrinsic) | **NO** | `"<letter> - <amulet> (being worn)"` — vendor `do_wear.c:2061 prinv((char*)0, obj, 0L)`; `(being worn)` suffix from `doname`/`xprname` (`do_wear.c:1267,1484`). NOTE: no NLE action verb routes to `wear_amulet`/`put_on_ring` — see "Routing gaps". |
| Zap-* | `"The feeling subsides."` | `items_wands.py:1383 _effect_enlightenment` | YES | n/a (no map change) | **NO** | `"The feeling subsides."` — vendor `zap.c:2188 pline_The("feeling subsides.")`. Handler is a literal `return state, rng` no-op with no `enlightenment()` and no message. **This is the flagged example.** |
| Read-* | `"This scroll seems to be blank."` | `items_scrolls.py:1509 _effect_blank_paper` | YES | n/a (nothing happens) | **NO** | `"This scroll seems to be blank."` — vendor `read.c:1266 pline("This scroll seems to be blank.")`. Handler is `return state` no-op. |
| Pray-* | `add_positional_event("altar","pray")` | `prayer.py:1xxx handle_pray` | YES | YES (prayer outcome) | partial | Emits `YOU_PRAY`="You begin praying to your god." (`pray.c::dopray`) via `messages.emit`. RM is **positional**, not message-based — fires if env exposes player-on-altar + pray action. Engine applies prayer, so likely OK; verify altar-position event wiring in the env, not the handler. |
| Sink-* | `add_positional_event("sink","quaff")` | `items_potions.py:1527 handle_quaff` | YES (quaff) | **NO sink path** | **NO** | Vendor `potion.c:506-511`: dodrink at a sink prompts `"Drink from the sink?"` then calls `drinksink()` (`fountain.c:520`). `handle_quaff` only drinks a potion from inventory; there is **no quaff-at-sink branch** and no sink terrain check. RM is positional (sink tile + quaff), but the engine performs no sink interaction. |
| Levitate-* | `"You start to float"` | `items_potions.py:665 _effect_levitation` (also ring/boots) | YES | YES (`Intrinsic.LEVITATION` timer) | **NO** | `"You start to float in the air!"` — vendor `trap.c:2891 You("start to float in the air!")`, reached via `potion.c:1034 float_up()`. Handler sets the timer but never calls a float_up equivalent / emits the message. |
| Freeze-* | cold bolt msg + kill | `apply_tools.py:727 frost horn branch` | YES | YES (6d6 cold dmg + kill) | **NO** | `"The bolt of cold hits the <monster>!"` — vendor `music.c:599 buzz(AD_COLD-1,...)` -> `zap.c:3158 hit(fltxt, mon, "!")` with `flash_types[AD_COLD-1]`=`"bolt of cold"` (`zap.c:63`). Handler deals damage/kill but emits no cold-bolt message. |

## Routing gaps (separate from message gaps)

- **PutOn-***: grep of `action_dispatch.py` shows `handle_wield`/`handle_wear`/
  `handle_quaff`/`handle_pray`/`handle_eat` imported, but no `wear_amulet` /
  `put_on_ring` handler is dispatched from an NLE action. Even with a message,
  the amulet env may never invoke `wear_amulet`. Verify the action table.
- **Sink-***: no sink interaction exists anywhere in `handle_quaff`; needs a new
  branch (terrain==SINK -> drinksink-equivalent) before any reward can fire.

## Top fixes, ranked by skill envs unblocked

Each fix unblocks exactly one env category, but the cheapest/highest-confidence
ones are message-only on handlers that already apply the effect.

1. **Eat-*** — add `EAT_FOOD`-style / per-food "tastes delicious!" message to
   `handle_eat`. Effect already applied. (1 env; trivial.)
2. **Wear-*** — emit `"You are now wearing a <armor>."` from `wear_armor`.
   Effect applied. (1 env; trivial; new MessageId with arg slot.)
3. **Wield-*** — emit `"<letter> - a <weapon>"` from `handle_wield`. Effect
   applied. (1 env; needs letter+doname formatting.)
4. **Zap-*** (the flagged bug) — emit `"The feeling subsides."` from
   `_effect_enlightenment`. No effect needed. (1 env; trivial.)
5. **Read-*** — emit `"This scroll seems to be blank."` from
   `_effect_blank_paper`. No effect needed. (1 env; trivial.)
6. **Levitate-*** — emit `"You start to float in the air!"` when
   `_effect_levitation` (and ring/boots equivalents) start levitation. Effect
   applied. (1 env; check RM substring `"You start to float"` matches.)
7. **Freeze-*** — emit `"The bolt of cold hits the <monster>!"` from the frost
   horn branch in `apply_tools.py`. Effect applied. (1 env; needs monster-name
   arg slot; confirm RM's exact substring.)
8. **PutOn-*** — emit amulet `"<letter> - <amulet> (being worn)"` AND wire
   `wear_amulet` into the action dispatch. (1 env; 2-part fix.)
9. **Sink-*** — add a quaff-at-sink branch (terrain check + drinksink effect +
   positional event). Largest fix: new engine behavior, not just a message.
   (1 env.)

Pray-* is the only category whose handler already emits via the messages
subsystem; it is RM-positional and most likely already functional — verify the
altar-position event in the env wrapper rather than the handler.

### Shared blocker

All seven message-only fixes (1-7) require new `MessageId` enum entries +
`_MESSAGE_TEMPLATES` rows in `Nethax/nethax/subsystems/messages.py`; none of the
target strings exist there today (highest id = 42 `LEVI_WOBBLE`). Adding the
templates is the common prerequisite that unblocks the entire batch.
