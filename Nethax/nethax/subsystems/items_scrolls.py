"""Scroll effects — vendor/nethack/src/read.c::seffects."""
from enum import IntEnum

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.constants.objects import ObjectClass


# ---------------------------------------------------------------------------
# Canonical type_id values — position in the compiled object table.
# Order matches vendor/nethack/include/objects.h SCROLL() macro sequence.
# The first scroll (enchant armor) follows the last potion in the table.
#
# Canonical order from objects.h (sequential, starting at _SCROLL_BASE_ID):
#   0  enchant_armor       (SCR_ENCHANT_ARMOR)
#   1  destroy_armor       (SCR_DESTROY_ARMOR)
#   2  confuse_monster     (SCR_CONFUSE_MONSTER)
#   3  scare_monster       (SCR_SCARE_MONSTER)
#   4  remove_curse        (SCR_REMOVE_CURSE)
#   5  enchant_weapon      (SCR_ENCHANT_WEAPON)
#   6  create_monster      (SCR_CREATE_MONSTER)
#   7  taming              (SCR_TAMING)
#   8  genocide            (SCR_GENOCIDE)
#   9  light               (SCR_LIGHT)
#  10  teleportation       (SCR_TELEPORTATION)
#  11  gold_detection      (SCR_GOLD_DETECTION)
#  12  food_detection      (SCR_FOOD_DETECTION)
#  13  identify            (SCR_IDENTIFY)
#  14  magic_mapping       (SCR_MAGIC_MAPPING)
#  15  amnesia             (SCR_AMNESIA)
#  16  fire                (SCR_FIRE)
#  17  earth               (SCR_EARTH)
#  18  punishment          (SCR_PUNISHMENT)
#  19  charging            (SCR_CHARGING)
#  20  stinking_cloud      (SCR_STINKING_CLOUD)
#  21  mail                (SCR_MAIL) — conditional on MAIL_STRUCTURES
#  22  blank_paper         (SCR_BLANK_PAPER)
# ---------------------------------------------------------------------------

# The first scroll entry follows the last potion (water) in objects.h.
# From objects.py: potions run 68–83 (16 entries in objects.py), but
# objects.h canonical order has 26 potions so scrolls start at 68+26 = 94.
_SCROLL_BASE_ID = 94   # first scroll entry in compiled object table


class ScrollEffect(IntEnum):
    """Canonical scroll effect identifiers.

    Values are sequential indices into the scroll sub-table (type_id minus
    _SCROLL_BASE_ID), matching the SCROLL() macro order in objects.h.
    """
    ENCHANT_ARMOR    =  0   # SCR_ENCHANT_ARMOR
    DESTROY_ARMOR    =  1   # SCR_DESTROY_ARMOR
    CONFUSE_MONSTER  =  2   # SCR_CONFUSE_MONSTER
    SCARE_MONSTER    =  3   # SCR_SCARE_MONSTER
    REMOVE_CURSE     =  4   # SCR_REMOVE_CURSE
    ENCHANT_WEAPON   =  5   # SCR_ENCHANT_WEAPON
    CREATE_MONSTER   =  6   # SCR_CREATE_MONSTER
    TAMING           =  7   # SCR_TAMING
    GENOCIDE         =  8   # SCR_GENOCIDE
    LIGHT            =  9   # SCR_LIGHT
    TELEPORTATION    = 10   # SCR_TELEPORTATION
    GOLD_DETECTION   = 11   # SCR_GOLD_DETECTION
    FOOD_DETECTION   = 12   # SCR_FOOD_DETECTION
    IDENTIFY         = 13   # SCR_IDENTIFY
    MAGIC_MAPPING    = 14   # SCR_MAGIC_MAPPING
    AMNESIA          = 15   # SCR_AMNESIA
    FIRE             = 16   # SCR_FIRE
    EARTH            = 17   # SCR_EARTH
    PUNISHMENT       = 18   # SCR_PUNISHMENT
    CHARGING         = 19   # SCR_CHARGING
    STINKING_CLOUD   = 20   # SCR_STINKING_CLOUD
    MAIL             = 21   # SCR_MAIL
    BLANK_PAPER      = 22   # SCR_BLANK_PAPER


N_SCROLLS = 23


# ---------------------------------------------------------------------------
# BUC sentinel constants (matches items.py BUCStatus)
# ---------------------------------------------------------------------------

_BUC_CURSED   = 1
_BUC_UNCURSED = 2
_BUC_BLESSED  = 3


def _is_blessed(buc):
    return jnp.int32(buc) == jnp.int32(_BUC_BLESSED)


def _is_cursed(buc):
    return jnp.int32(buc) == jnp.int32(_BUC_CURSED)


# ---------------------------------------------------------------------------
# Genocide full class table  (vendor/nethack/src/read.c::do_genocide)
#
# Vendor accepts ANY monster class letter (S_ANT 'a' through '@' for humans,
# plus the long-tail symbols ' ' / '&' / ';' / ':' / '~' / ']' that map to
# MonsterSymbol values >= 53).  The class is converted into a MonsterSymbol
# value at the call site and the kill sweep uses that symbol directly.
# ---------------------------------------------------------------------------

# Letter → MonsterSymbol value table (mirrors vendor/nethack/include/monsym.h
# DEF_MONSYMS array).  We build this eagerly at module load so JIT does not
# trace the table.
def _build_class_letter_to_symbol() -> dict:
    """Return {letter (int char code) : MonsterSymbol int value}."""
    from Nethax.nethax.constants.monsters import MonsterSymbol
    table = {}
    # 'a'..'z' map to S_ANT..S_ZRUTY (values 1..26).
    for offset in range(26):
        table[ord('a') + offset] = 1 + offset
    # 'A'..'Z' map to S_ANGEL..S_ZOMBIE (values 27..52).
    for offset in range(26):
        table[ord('A') + offset] = 27 + offset
    # Long-tail glyphs (vendor monsym.h).
    table[ord('@')] = int(MonsterSymbol.S_HUMAN)
    table[ord(' ')] = int(MonsterSymbol.S_GHOST)
    table[ord("'")] = int(MonsterSymbol.S_GOLEM)
    table[ord('&')] = int(MonsterSymbol.S_DEMON)
    table[ord(';')] = int(MonsterSymbol.S_EEL)
    table[ord(':')] = int(MonsterSymbol.S_LIZARD)
    table[ord('~')] = int(MonsterSymbol.S_WORM_TAIL)
    table[ord(']')] = int(MonsterSymbol.S_MIMIC_DEF)
    return table


_CLASS_LETTER_TO_SYMBOL: dict = _build_class_letter_to_symbol()


def _build_monster_symbol_table() -> jnp.ndarray:
    """Build MONSTERS[i].symbol lookup eagerly at module load.

    Built once so it never traces inside a jit-compiled context.
    """
    from Nethax.nethax.constants.monsters import MONSTERS
    return jnp.array([int(m.symbol) for m in MONSTERS], dtype=jnp.int8)


# Eager build (mirrors polymorph._build_monster_lookup_tables pattern).
_MONSTER_SYMBOL_TABLE: jnp.ndarray = _build_monster_symbol_table()

# Wave 5 random-pool: a small subset retained for the scroll-read code path
# that selects a class at random.  Vendor scrolls always let the *player*
# pick; we keep a uniform pick for the scroll-read flow until a UI layer
# can supply the player's choice.
_GENOCIDE_CLASS_VALUES: tuple = (
    33,   # S_GNOME
    11,   # S_KOBOLD
    15,   # S_ORC
    18,   # S_RODENT
    8,    # S_HUMANOID
)
_GENOCIDE_CLASS_POOL: jnp.ndarray = jnp.array(
    _GENOCIDE_CLASS_VALUES, dtype=jnp.int8
)


def _kill_all_of_symbol(state, chosen_class):
    """Kill every live monster on the current level whose MONSTERS[entry].symbol
    equals ``chosen_class`` (jnp int32 scalar).

    JIT-safe.  Always marks GENOCIDELESS regardless of whether any matched.
    """
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated

    symbol_table = _MONSTER_SYMBOL_TABLE  # int8[n_monsters]
    mai = state.monster_ai
    safe_entry = jnp.clip(mai.entry_idx.astype(jnp.int32),
                          0, symbol_table.shape[0] - 1)
    mon_symbols = symbol_table[safe_entry].astype(jnp.int32)

    is_match = mai.alive & (mon_symbols == jnp.int32(chosen_class))
    new_alive = jnp.where(is_match, jnp.bool_(False), mai.alive)
    new_hp    = jnp.where(is_match, jnp.int32(0), mai.hp)

    new_mai = mai.replace(alive=new_alive, hp=new_hp)
    new_state = state.replace(monster_ai=new_mai)
    return mark_violated(new_state, int(Conduct.GENOCIDELESS))


def _apply_genocide(state, rng):
    """Random-class genocide for the scroll-read flow (Wave 5 simplification).

    The player picks a class letter in vendor; we sample a class uniformly
    from ``_GENOCIDE_CLASS_POOL`` until a higher-layer UI supplies the pick.
    """
    class_pool = _GENOCIDE_CLASS_POOL
    n_classes = class_pool.shape[0]
    pick_idx = jax.random.randint(rng, (), 0, n_classes).astype(jnp.int32)
    chosen_class = class_pool[pick_idx].astype(jnp.int32)
    return _kill_all_of_symbol(state, chosen_class)


def apply_genocide(state, rng, class_letter=None):
    """Apply scroll/spell of genocide.

    Per vendor/nethack/src/read.c::do_genocide:
      - For every alive monster on level whose MONSTERS[entry_idx].symbol
        matches the chosen class, set ``alive = False`` and ``hp = 0``.
      - Always set the GENOCIDELESS conduct.

    Parameters
    ----------
    state         : EnvState
    rng           : jax.random.PRNGKey   used only if class_letter is None
                    (random-pool scroll fallback).
    class_letter  : str | int | None
                    - str of length 1 (e.g. 'd', 'L', '@'): genocide that class.
                    - int: treated as a MonsterSymbol enum value directly.
                    - None: sample uniformly from the legacy ``_GENOCIDE_CLASS_POOL``
                      to preserve the Wave-5 scroll-read code path.
                    Unknown letters become a no-op (still flips GENOCIDELESS).
    """
    if class_letter is None:
        return _apply_genocide(state, rng)
    if isinstance(class_letter, str):
        # Map letter → MonsterSymbol value (or -1 for unknown letters).
        symbol_val = _CLASS_LETTER_TO_SYMBOL.get(ord(class_letter[:1]), -1)
    else:
        symbol_val = int(class_letter)
    return _kill_all_of_symbol(state, jnp.int32(symbol_val))


# ---------------------------------------------------------------------------
# Per-effect implementations
# Each takes (state, rng, buc: jnp scalar int8) → state.
# ---------------------------------------------------------------------------

# ---- identification -------------------------------------------------------

def _effect_identify(state, rng, buc):
    """scroll of identify — identify inventory items.

    Canonical: seffect_identify — identify 1 item (uncursed), all (blessed),
    or ask which (interactive).
    Wave 3: blessed identifies first 4 unidentified items; uncursed identifies
    the first unidentified item; cursed no-op.
    Identification sets item.identified = True on the relevant slots.
    """
    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)

    # Number of items to identify
    n_to_id = jnp.where(blessed, jnp.int32(4),
              jnp.where(cursed,  jnp.int32(0), jnp.int32(1)))

    # Walk through inventory slots and flip identified=True for the first
    # n_to_id unidentified items.
    old_identified = state.inventory.items.identified  # [52] bool

    def _mark_up_to_n(carry, slot_idx):
        identified_arr, remaining = carry
        is_unid   = ~identified_arr[slot_idx]
        should_id = is_unid & (remaining > jnp.int32(0))
        new_arr   = jnp.where(should_id,
                              identified_arr.at[slot_idx].set(jnp.bool_(True)),
                              identified_arr)
        new_rem   = jnp.where(should_id, remaining - jnp.int32(1), remaining)
        return (new_arr, new_rem), None

    n_slots = old_identified.shape[0]
    (new_identified, _), _ = jax.lax.scan(
        _mark_up_to_n,
        (old_identified, n_to_id),
        jnp.arange(n_slots, dtype=jnp.int32),
    )
    new_items = state.inventory.items.replace(identified=new_identified)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---- enchantment ----------------------------------------------------------

def _effect_enchant_weapon(state, rng, buc):
    """scroll of enchant weapon — +1 enchant on wielded weapon.

    Canonical: seffect_enchant_weapon — enchant wielded weapon +1 (blessed +2,
    cursed chance of destroy at high enchant).
    Wave 3: +1 enchantment on wielded slot item; blessed +2; cursed -1.
    """
    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)
    delta   = jnp.where(blessed, jnp.int8(2),
              jnp.where(cursed,  jnp.int8(-1), jnp.int8(1)))

    wielded = state.inventory.wielded.astype(jnp.int32)
    valid   = wielded >= jnp.int32(0)

    old_enc  = state.inventory.items.enchantment
    new_enc_val = jnp.clip(old_enc[wielded] + delta, jnp.int8(-7), jnp.int8(7))
    new_enc  = jnp.where(valid,
                         old_enc.at[wielded].set(new_enc_val),
                         old_enc)
    new_items = state.inventory.items.replace(enchantment=new_enc)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _effect_enchant_armor(state, rng, buc):
    """scroll of enchant armor — +1 enchant on worn body armor.

    Canonical: seffect_enchant_armor — enchant a piece of worn armor.
    Wave 3: +1 enchantment on worn body armor (slot 0); blessed +2; cursed -1.
    """
    cursed   = _is_cursed(buc)
    blessed  = _is_blessed(buc)
    delta    = jnp.where(blessed, jnp.int8(2),
               jnp.where(cursed,  jnp.int8(-1), jnp.int8(1)))

    armor_slot = state.inventory.worn_armor[0].astype(jnp.int32)  # body armor
    valid      = armor_slot >= jnp.int32(0)

    old_enc  = state.inventory.items.enchantment
    new_enc_val = jnp.clip(old_enc[armor_slot] + delta, jnp.int8(-7), jnp.int8(7))
    new_enc  = jnp.where(valid,
                         old_enc.at[armor_slot].set(new_enc_val),
                         old_enc)
    new_items = state.inventory.items.replace(enchantment=new_enc)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _effect_destroy_armor(state, rng, buc):
    """scroll of destroy armor — destroy a piece of worn armor.

    Canonical: seffect_destroy_armor — set enchant to -6 or destroy.
    Wave 3: cursed or uncursed: set worn body armor enchantment to -6;
    blessed: no-op (Canonically the scroll has no blessed effect distinct
    from uncursed when read normally).
    """
    cursed   = _is_cursed(buc)
    blessed  = _is_blessed(buc)

    armor_slot = state.inventory.worn_armor[0].astype(jnp.int32)
    valid      = armor_slot >= jnp.int32(0)
    do_damage  = valid & ~blessed

    old_enc = state.inventory.items.enchantment
    new_enc = jnp.where(do_damage,
                        old_enc.at[armor_slot].set(jnp.int8(-6)),
                        old_enc)
    new_items = state.inventory.items.replace(enchantment=new_enc)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _effect_charging(state, rng, buc):
    """scroll of charging — recharge a wand/tool.

    Canonical: seffect_charging — restore charges on a wand.
    Wave 3: +5 charges on the first wand found (WAND_CLASS) in inventory;
    blessed: all wands get +5.
    """
    blessed    = _is_blessed(buc)
    categories = state.inventory.items.category  # [52]
    charges    = state.inventory.items.charges   # [52]

    is_wand    = categories == jnp.int8(ObjectClass.WAND_CLASS)

    # Blessed: charge all wands; else charge first found wand only.
    first_wand = jnp.argmax(is_wand).astype(jnp.int32)
    slot_mask  = jnp.where(blessed,
                           is_wand,
                           jnp.arange(52, dtype=jnp.int32) == first_wand)
    found_any  = jnp.any(is_wand)
    new_charges = jnp.where(
        found_any,
        jnp.where(slot_mask,
                  jnp.clip(charges.astype(jnp.int32) + 5, 0, 40).astype(jnp.int8),
                  charges),
        charges,
    )
    new_items = state.inventory.items.replace(charges=new_charges)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---- curse/bless ----------------------------------------------------------

def _effect_remove_curse(state, rng, buc):
    """scroll of remove curse — uncurse inventory items.

    Canonical: seffect_remove_curse — uncurse worn/wielded items (blessed:
    all inventory items; uncursed: worn+wielded; cursed: re-curses items).
    Wave 3: uncursed → all items with buc_status==CURSED become UNCURSED;
    blessed → same; cursed → all items become CURSED.
    """
    cursed     = _is_cursed(buc)
    old_buc    = state.inventory.items.buc_status  # [52] int8
    # uncurse or curse all items
    new_buc = jnp.where(
        cursed,
        jnp.where(old_buc != jnp.int8(0),
                  jnp.full_like(old_buc, _BUC_CURSED),
                  old_buc),
        jnp.where(old_buc == jnp.int8(_BUC_CURSED),
                  jnp.full_like(old_buc, _BUC_UNCURSED),
                  old_buc),
    )
    new_items = state.inventory.items.replace(buc_status=new_buc)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


# ---- detection -----------------------------------------------------------

def _effect_gold_detection(state, rng, buc):
    """scroll of gold detection — sense nearby gold.

    Canonical: seffect_gold_detection — sense_gold / show gold locations.
    Wave 3: no-op (gold-on-map not yet modelled).
    """
    return state


def _effect_food_detection(state, rng, buc):
    """scroll of food detection — sense food items.

    Canonical: seffect_food_detection — sense_food / show food locations.
    Wave 3: no-op (food-on-map not yet modelled).
    """
    return state


# ---- mapping / teleport ---------------------------------------------------

def _effect_magic_mapping(state, rng, buc):
    """scroll of magic mapping — reveal the entire current level.

    Canonical: seffect_magic_mapping — level_mapalot() (reveal all tiles).
    Wave 3: set explored[current_branch, current_level-1] to all True.
    Cursed: also confuses the player for 30 turns.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1

    new_explored = state.explored.at[b, lv].set(
        jnp.ones_like(state.explored[b, lv])
    )
    new_state = state.replace(explored=new_explored)

    # Cursed: add 30-turn confusion
    cursed   = _is_cursed(buc)
    cur_conf = new_state.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_conf = jnp.where(cursed, jnp.maximum(cur_conf, jnp.int32(30)), cur_conf)
    new_ts   = new_state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_conf)
    new_status = new_state.status.replace(timed_statuses=new_ts)
    return new_state.replace(status=new_status)


def _effect_teleportation(state, rng, buc):
    """scroll of teleportation — teleport the player.

    Canonical: seffect_teleportation — tele(); blessed: controlled tele.
    Wave 3: randomise player_pos within the map bounds using rng.
    Cursed: teleports to a random dungeon level instead (simplified: random pos).
    """
    rng1, rng2 = jax.random.split(rng)
    h, w = state.terrain.shape[2], state.terrain.shape[3]
    new_row = jax.random.randint(rng1, shape=(), minval=0, maxval=h).astype(jnp.int16)
    new_col = jax.random.randint(rng2, shape=(), minval=0, maxval=w).astype(jnp.int16)
    new_pos = jnp.array([new_row, new_col], dtype=jnp.int16)
    return state.replace(player_pos=new_pos)


def _effect_light(state, rng, buc):
    """scroll of light — illuminate the current level.

    Canonical: seffect_light — set litroom / litcorridor for nearby tiles.
    Wave 3: mark the current level explored (same as mapping but only 1 level).
    Blessed: reveal entire level (same effect as magic mapping for Wave 3).
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    new_explored = state.explored.at[b, lv].set(
        jnp.ones_like(state.explored[b, lv])
    )
    return state.replace(explored=new_explored)


# ---- monster effects ------------------------------------------------------

def _effect_scare_monster(state, rng, buc):
    """scroll of scare monster — scare nearby monsters.

    Canonical: seffect_scare_monster — set mflee on visible monsters.
    Wave 3: no-op (monster flee state not yet in MonsterAIState).
    """
    return state


def _effect_confuse_monster(state, rng, buc):
    """scroll of confuse monster — confuse nearby monsters.

    Canonical: seffect_confuse_monster — set mconf on nearby monsters.
    Wave 3: no-op (monster confusion state stub).
    """
    return state


def _effect_create_monster(state, rng, buc):
    """scroll of create monster — create a monster near the player.

    Canonical: seffect_create_monster — makemon(NULL, ux, uy, ...).
    Wave 3: no-op (monster creation deferred to monster subsystem Wave 3+).
    """
    return state


def _effect_taming(state, rng, buc):
    """scroll of taming — tame nearby monsters.

    Canonical: seffect_taming — tamedog / tamemonst on nearby monsters.
    Wave 3: no-op (taming state stub).
    """
    return state


def _effect_genocide(state, rng, buc):
    """scroll of genocide — remove all monsters of a chosen class on the level.

    Canonical: vendor/nethack/src/read.c::do_genocide — player picks a monster
    class (or species); every live monster on the level matching the chosen
    class has its alive flag cleared.

    Wave 5 simplification: pick a class uniformly at random from the
    candidate list ``_GENOCIDE_CLASSES`` and kill every monster whose
    MONSTERS[entry_idx].symbol matches that class.

    Conduct: vendor/nethack/src/read.c::do_genocide — GENOCIDELESS broken on
    any successful genocide.  We mark the violation whenever the scroll is
    read (always, since the spell/scroll always executes a class pick).
    """
    return _apply_genocide(state, rng)


# ---- harmful effects -------------------------------------------------------

def _effect_amnesia(state, rng, buc):
    """scroll of amnesia — forget the current level map.

    Canonical: seffect_amnesia — forget(FORGET_LEVELS | FORGET_SPELLS).
    Wave 3: set explored[current_branch, current_level-1] to all False.
    """
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    new_explored = state.explored.at[b, lv].set(
        jnp.zeros_like(state.explored[b, lv])
    )
    return state.replace(explored=new_explored)


def _effect_fire(state, rng, buc):
    """scroll of fire — fire explosion centered on player.

    Canonical: seffect_fire — explode(hero, EXPL_FIERY, ...).
    Wave 3: deal 10 HP fire damage to the player (blessed: 5 HP, cursed: 20 HP).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    dmg     = jnp.where(blessed, jnp.int32(5),
              jnp.where(cursed,  jnp.int32(20), jnp.int32(10)))
    new_hp  = jnp.maximum(state.player_hp - dmg, jnp.int32(1))
    return state.replace(player_hp=new_hp)


def _effect_earth(state, rng, buc):
    """scroll of earth — summon rocks / boulders from ceiling.

    Canonical: seffect_earth — drop rocks on monsters/player.
    Wave 3: deal 5 HP blunt damage (blessed: 0, cursed: 15).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    dmg     = jnp.where(blessed, jnp.int32(0),
              jnp.where(cursed,  jnp.int32(15), jnp.int32(5)))
    new_hp  = jnp.maximum(state.player_hp - dmg, jnp.int32(1))
    return state.replace(player_hp=new_hp)


def _effect_punishment(state, rng, buc):
    """scroll of punishment — ball and chain appear on player.

    Canonical: seffect_punishment — attach iron ball and chain.
    Wave 3: deal 5 HP damage and add 30-turn STUMBLING-equivalent (WOUNDED_LEGS).
    Blessed: no-op.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    dmg     = jnp.where(blessed, jnp.int32(0), jnp.int32(5))
    new_hp  = jnp.maximum(state.player_hp - dmg, jnp.int32(1))
    turns   = jnp.where(blessed, jnp.int32(0), jnp.int32(30))
    cur_wl  = state.status.timed_statuses[int(TimedStatus.WOUNDED_LEGS)]
    new_wl  = jnp.maximum(cur_wl, turns)
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.WOUNDED_LEGS)].set(new_wl)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(player_hp=new_hp, status=new_status)


def _effect_stinking_cloud(state, rng, buc):
    """scroll of stinking cloud — create nausea cloud at player position.

    Canonical: seffect_stinking_cloud — do_stinking_cloud(); create cloud object.
    Wave 3: add 15-turn VOMITING status (blessed 0, cursed 30).
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)
    turns   = jnp.where(blessed, jnp.int32(0),
              jnp.where(cursed,  jnp.int32(30), jnp.int32(15)))
    cur_vom = state.status.timed_statuses[int(TimedStatus.VOMITING)]
    new_vom = jnp.maximum(cur_vom, turns)
    new_ts  = state.status.timed_statuses.at[int(TimedStatus.VOMITING)].set(new_vom)
    new_status = state.status.replace(timed_statuses=new_ts)
    return state.replace(status=new_status)


# ---- misc -----------------------------------------------------------------

def _effect_mail(state, rng, buc):
    """scroll of mail — deliver a mail message.

    Canonical: seffect_mail — deliver a mail message from a mail daemon.
    Wave 3: no-op (mail system not applicable to headless JAX sim).
    """
    return state


def _effect_blank_paper(state, rng, buc):
    """scroll of blank paper — nothing happens.

    Canonical: seffect_blank_paper — "This scroll seems to be blank."
    Wave 3: pure no-op.
    """
    return state


# ---------------------------------------------------------------------------
# Effect dispatch table — indexed by ScrollEffect value.
# Must contain exactly N_SCROLLS entries in enum order.
# ---------------------------------------------------------------------------

_EFFECT_TABLE = (
    _effect_enchant_armor,    #  0  ENCHANT_ARMOR
    _effect_destroy_armor,    #  1  DESTROY_ARMOR
    _effect_confuse_monster,  #  2  CONFUSE_MONSTER
    _effect_scare_monster,    #  3  SCARE_MONSTER
    _effect_remove_curse,     #  4  REMOVE_CURSE
    _effect_enchant_weapon,   #  5  ENCHANT_WEAPON
    _effect_create_monster,   #  6  CREATE_MONSTER
    _effect_taming,           #  7  TAMING
    _effect_genocide,         #  8  GENOCIDE
    _effect_light,            #  9  LIGHT
    _effect_teleportation,    # 10  TELEPORTATION
    _effect_gold_detection,   # 11  GOLD_DETECTION
    _effect_food_detection,   # 12  FOOD_DETECTION
    _effect_identify,         # 13  IDENTIFY
    _effect_magic_mapping,    # 14  MAGIC_MAPPING
    _effect_amnesia,          # 15  AMNESIA
    _effect_fire,             # 16  FIRE
    _effect_earth,            # 17  EARTH
    _effect_punishment,       # 18  PUNISHMENT
    _effect_charging,         # 19  CHARGING
    _effect_stinking_cloud,   # 20  STINKING_CLOUD
    _effect_mail,             # 21  MAIL
    _effect_blank_paper,      # 22  BLANK_PAPER
)

assert len(_EFFECT_TABLE) == N_SCROLLS, (
    f"Effect table has {len(_EFFECT_TABLE)} entries; expected {N_SCROLLS}"
)

# Build lax.switch branch list: each branch unpacks (state, rng, buc).
_SWITCH_BRANCHES = [
    (lambda operand, fn=fn: fn(operand[0], operand[1], operand[2]))
    for fn in _EFFECT_TABLE
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_scroll(state, rng, slot_idx):
    """Apply the scroll in inventory slot `slot_idx`.

    Looks up type_id → ScrollEffect, dispatches via jax.lax.switch, then
    decrements quantity (marks slot empty when exhausted).

    Parameters
    ----------
    state    : EnvState
    rng      : jax.random.PRNGKey
    slot_idx : int or traced jnp scalar — inventory slot index

    Returns
    -------
    Updated EnvState.
    """
    slot_idx  = jnp.int32(slot_idx)
    items     = state.inventory.items
    type_id   = items.type_id[slot_idx].astype(jnp.int32)
    buc       = items.buc_status[slot_idx]

    effect_id = jnp.clip(
        type_id - jnp.int32(_SCROLL_BASE_ID),
        0,
        N_SCROLLS - 1,
    )

    # Dispatch: operand is (state, rng, buc); each branch returns new state.
    new_state = jax.lax.switch(effect_id, _SWITCH_BRANCHES, (state, rng, buc))

    # Decrement quantity; clear category when exhausted.
    old_qty  = new_state.inventory.items.quantity[slot_idx]
    new_qty  = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
    new_cat  = jnp.where(new_qty == jnp.int16(0),
                         jnp.int8(0),
                         new_state.inventory.items.category[slot_idx])
    new_quantity = new_state.inventory.items.quantity.at[slot_idx].set(new_qty)
    new_category = new_state.inventory.items.category.at[slot_idx].set(new_cat)
    new_items    = new_state.inventory.items.replace(
        quantity=new_quantity, category=new_category
    )
    new_inv = new_state.inventory.replace(items=new_items)
    return new_state.replace(inventory=new_inv)


def handle_read(state, rng):
    """Find the first valid scroll in inventory and read it.

    Wave 3: uses "first valid item" strategy; Wave 4 will add a menu.
    A valid scroll slot has category == SCROLL_CLASS and quantity > 0.
    Falls back to no-op if no scrolls found.

    Parameters
    ----------
    state : EnvState
    rng   : jax.random.PRNGKey

    Returns
    -------
    Updated EnvState.
    """
    categories = state.inventory.items.category   # [MAX_INVENTORY_SLOTS]
    quantities = state.inventory.items.quantity    # [MAX_INVENTORY_SLOTS]

    is_scroll  = categories == jnp.int8(ObjectClass.SCROLL_CLASS)
    has_stock  = quantities > jnp.int16(0)
    valid_mask = is_scroll & has_stock

    slot_idx = jnp.argmax(valid_mask).astype(jnp.int32)
    found    = jnp.any(valid_mask)

    new_state = jax.lax.cond(
        found,
        lambda s_r: read_scroll(s_r[0], s_r[1], slot_idx),
        lambda s_r: s_r[0],
        (state, rng),
    )
    # Conduct: vendor/nethack/src/read.c::doread — ILLITERATE broken on any
    # successful scroll read (insight.c ~2147, u.uconduct.literate).
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated_if
    return mark_violated_if(new_state, int(Conduct.ILLITERATE), found)
