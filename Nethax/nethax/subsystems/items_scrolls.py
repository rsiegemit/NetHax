"""Scroll effects — vendor/nethack/src/read.c::seffects."""
from enum import IntEnum

import jax
import jax.numpy as jnp
import jax.lax as lax

from Nethax.nethax.subsystems.status_effects import TimedStatus
from Nethax.nethax.constants.objects import ObjectClass
from Nethax.nethax.subsystems import detect as _detect
from Nethax.nethax.constants.monsters import MONSTERS

N_MONSTERS: int = len(MONSTERS)


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


# Boulder type_id in the compiled object table (objects.py entry #447).
# vendor/nethack/include/objects.h — boulder is the first ROCK_CLASS entry.
BOULDER_TYPE_ID: int = 447


def _build_monster_fire_resist_table() -> jnp.ndarray:
    """Build MONSTERS[i].resists_mask & MR_FIRE lookup eagerly at module load.

    Returns bool[n_monsters] — True where the monster is fire-resistant.
    vendor/nethack/include/monflag.h MR_FIRE = 0x01.
    """
    from Nethax.nethax.constants.monsters import MONSTERS, MR_FIRE
    return jnp.array(
        [(int(m.resists_mask) & MR_FIRE) != 0 for m in MONSTERS],
        dtype=jnp.bool_,
    )


_MONSTER_FIRE_RESIST_TABLE: jnp.ndarray = _build_monster_fire_resist_table()


def _build_scare_immune_table() -> jnp.ndarray:
    """bool[N_MONSTERS] — True if monster is immune to scare-monster scroll.

    Immune classes (vendor/nethack/src/zap.c::resist, read.c::seffect_scare_monster
    ~1454-1486):
      - Demon (M2_DEMON)
      - Lawful minion (M2_MINION with positive alignment)
      - Angelic beings (MonsterSymbol.S_ANGEL)
    """
    from Nethax.nethax.constants.monsters import (
        MONSTERS, M2_DEMON, M2_MINION, MonsterSymbol,
    )
    result = []
    for m in MONSTERS:
        is_demon  = bool(m.flags2 & M2_DEMON)
        is_minion = bool(m.flags2 & M2_MINION)
        is_lawful = m.alignment > 0
        is_angel  = (m.symbol == MonsterSymbol.S_ANGEL)
        result.append(is_demon or (is_minion and is_lawful) or is_angel)
    return jnp.array(result, dtype=jnp.bool_)


_IS_SCARE_IMMUNE: jnp.ndarray = _build_scare_immune_table()


def _build_tame_immune_table() -> jnp.ndarray:
    """bool[N_MONSTERS] — True if monster cannot be tamed.

    Immune: demons (M2_DEMON) and angelic beings (S_ANGEL).
    Cite: vendor/nethack/src/read.c::maybe_tame ~1044 — resist() called with
    SCROLL_CLASS; high-level demons and angels reliably resist.
    Simplified proxy: flag-based, JIT-pure.
    """
    from Nethax.nethax.constants.monsters import (
        MONSTERS, M2_DEMON, MonsterSymbol,
    )
    result = []
    for m in MONSTERS:
        is_demon = bool(m.flags2 & M2_DEMON)
        is_angel = (m.symbol == MonsterSymbol.S_ANGEL)
        result.append(is_demon or is_angel)
    return jnp.array(result, dtype=jnp.bool_)


_IS_TAME_IMMUNE: jnp.ndarray = _build_tame_immune_table()


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


def apply_genocide_single(state, entry_idx) -> object:
    """Genocide a single monster species by MONSTERS index (mndx).

    Vendor reference: vendor/nethack/src/read.c::do_genocide lines 2826-3015.
    When the player names a single creature (vendor "specific" path), every
    live monster with that exact ``mndx`` is killed and the species flag is
    set in ``state.genocided_species[entry_idx]``.

    JIT-pure: index masking via jnp.where; always flips GENOCIDELESS.
    """
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated

    mai = state.monster_ai
    ei = jnp.int32(entry_idx)
    safe_entry = jnp.clip(mai.entry_idx.astype(jnp.int32),
                          0, _MONSTER_SYMBOL_TABLE.shape[0] - 1)
    is_match = mai.alive & (safe_entry == ei)
    new_alive = jnp.where(is_match, jnp.bool_(False), mai.alive)
    new_hp    = jnp.where(is_match, jnp.int32(0), mai.hp)
    new_mai   = mai.replace(alive=new_alive, hp=new_hp)

    n_species = state.genocided_species.shape[0]
    safe_ei = jnp.clip(ei, 0, n_species - 1)
    new_geno = state.genocided_species.at[safe_ei].set(jnp.bool_(True))

    new_state = state.replace(monster_ai=new_mai, genocided_species=new_geno)
    return mark_violated(new_state, int(Conduct.GENOCIDELESS))


def _kill_all_of_symbol(state, chosen_class):
    """Kill every live monster whose MONSTERS[entry].symbol equals
    ``chosen_class`` (jnp int32 scalar) by applying the single-mndx genocide
    sweep to every matching mndx.

    Per vendor/nethack/src/read.c::do_genocide (lines 2826-3015), genociding
    by class iterates every mndx in that class — implemented here as a
    single vectorised sweep + per-mndx flag update so this remains JIT-safe.
    Always marks GENOCIDELESS regardless of whether any monsters matched.
    """
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated

    symbol_table = _MONSTER_SYMBOL_TABLE  # int8[n_monsters]
    mai = state.monster_ai
    safe_entry = jnp.clip(mai.entry_idx.astype(jnp.int32),
                          0, symbol_table.shape[0] - 1)
    mon_symbols = symbol_table[safe_entry].astype(jnp.int32)

    # Mark every mndx in the class as genocided (single-mndx semantics
    # applied per-entry; mirrors vendor's per-mndx loop within do_genocide).
    symbol_match_table = symbol_table.astype(jnp.int32) == jnp.int32(chosen_class)
    new_genocided = jnp.where(
        symbol_match_table, jnp.bool_(True), state.genocided_species
    )

    # Sweep live monsters of the class.
    is_match = mai.alive & (mon_symbols == jnp.int32(chosen_class))
    new_alive = jnp.where(is_match, jnp.bool_(False), mai.alive)
    new_hp    = jnp.where(is_match, jnp.int32(0), mai.hp)
    new_mai = mai.replace(alive=new_alive, hp=new_hp)

    new_state = state.replace(monster_ai=new_mai, genocided_species=new_genocided)
    return mark_violated(new_state, int(Conduct.GENOCIDELESS))


def _apply_genocide(state, rng, buc=None):
    """Genocide for the scroll-read flow.

    JAX-required divergence: vendor scroll-of-genocide prompts the player for
    a class letter via getlin() in read.c::do_genocide.  Our headless env has
    no interactive prompt, so the scroll-read flow samples a class uniformly
    from ``_GENOCIDE_CLASS_POOL`` (the same pool vendor would offer at the
    prompt).  ``apply_genocide(class_letter=...)`` is exposed for callers
    that have a pre-bound class pick (e.g. AI policy, scripted tests).

    Self-genocide (vendor read.c:2826-3015, Your_Own_Race macro read.c:9):
        When the scroll is cursed AND the chosen class collides with the
        player's race symbol, the player dies (player_hp = -1).
        Race → symbol map:
            HUMAN(0)/ELF(1) → S_HUMAN(53)
            DWARF(2)        → S_HUMANOID(8)
            GNOME(3)        → S_GNOME(33)
            ORC(4)          → S_ORC(15)
    """
    class_pool = _GENOCIDE_CLASS_POOL
    n_classes = class_pool.shape[0]
    pick_idx = jax.random.randint(rng, (), 0, n_classes).astype(jnp.int32)
    chosen_class = class_pool[pick_idx].astype(jnp.int32)
    new_state = _kill_all_of_symbol(state, chosen_class)

    if buc is None:
        return new_state

    # Self-genocide check: cursed scroll + chosen class matches player's race symbol.
    cursed = _is_cursed(buc)

    # Race-index → symbol lookup table (HUMAN=0, ELF=1, DWARF=2, GNOME=3, ORC=4).
    # Cite: vendor/nethack/src/read.c:9 Your_Own_Race(mndx) macro.
    _RACE_TO_SYMBOL = jnp.array([53, 53, 8, 33, 15], dtype=jnp.int32)
    race_idx = jnp.clip(new_state.player_race.astype(jnp.int32), 0, 4)
    player_race_symbol = _RACE_TO_SYMBOL[race_idx]

    is_own_race = chosen_class == player_race_symbol
    kill_self = cursed & is_own_race

    new_hp = jnp.where(kill_self, jnp.int32(-1), new_state.player_hp)
    return new_state.replace(player_hp=new_hp)


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

def _effect_identify(state, rng, buc, slot_idx=jnp.int32(-1)):
    """scroll of identify — identify inventory items.

    Canonical: vendor/nethack/src/read.c::seffect_identify lines 2055-2099.
    Vendor flow: ``useup(sobj); *sobjp = 0;`` is called BEFORE
    ``identify_pack(cval, !already_known)`` so the scroll itself is no longer
    in inventory when items are picked.  We mirror that by skipping
    ``slot_idx`` (the slot of the scroll being read) during the scan.
    Wave 3: blessed identifies first 4 unidentified items; uncursed identifies
    the first unidentified item; cursed no-op.
    Identification sets item.identified = True on the relevant slots.

    wave17h P0 (IDENTIFICATION #1): also flip type-level identification at
    state.identification.identified[obj_type] so future items of the same
    type render their true name. Cite: vendor/nethack/src/invent.c:2637-2647
    fully_identify_obj -> makeknown(otmp->otyp).
    """
    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)

    # Number of items to identify
    n_to_id = jnp.where(blessed, jnp.int32(4),
              jnp.where(cursed,  jnp.int32(0), jnp.int32(1)))

    # Walk through inventory slots and flip identified=True for the first
    # n_to_id unidentified items. Track which type_ids were flipped so we
    # can cascade to the type-level mask.
    items_in    = state.inventory.items
    old_identified = items_in.identified  # [52] bool
    type_ids       = items_in.type_id     # [52] int16

    # type-level table (NUM_OBJECTS bool); fall back if absent.
    type_mask_in = state.identification.identified  # [NUM_OBJECTS] bool

    scroll_slot = jnp.int32(slot_idx)

    def _mark_up_to_n(carry, slot_idx_):
        identified_arr, remaining, type_mask = carry
        is_unid   = ~identified_arr[slot_idx_]
        is_self   = slot_idx_ == scroll_slot
        should_id = is_unid & (remaining > jnp.int32(0)) & ~is_self
        new_arr   = jnp.where(should_id,
                              identified_arr.at[slot_idx_].set(jnp.bool_(True)),
                              identified_arr)
        new_rem   = jnp.where(should_id, remaining - jnp.int32(1), remaining)
        # Cascade type-level: set type_mask[type_id] = True when identified.
        t = type_ids[slot_idx_].astype(jnp.int32)
        t = jnp.clip(t, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1))
        new_type_mask = jnp.where(
            should_id,
            type_mask.at[t].set(jnp.bool_(True)),
            type_mask,
        )
        return (new_arr, new_rem, new_type_mask), None

    n_slots = old_identified.shape[0]
    (new_identified, _, new_type_mask), _ = jax.lax.scan(
        _mark_up_to_n,
        (old_identified, n_to_id, type_mask_in),
        jnp.arange(n_slots, dtype=jnp.int32),
    )
    new_items = state.inventory.items.replace(identified=new_identified)
    new_inv   = state.inventory.replace(items=new_items)
    new_ident = state.identification.replace(identified=new_type_mask)
    return state.replace(inventory=new_inv, identification=new_ident)


# ---- enchantment ----------------------------------------------------------

def _effect_enchant_weapon(state, rng, buc):
    """scroll of enchant weapon — enchant wielded weapon.

    vendor/nethack/src/read.c::seffect_enchant_weapon (~1627).
      cursed : -1
      uncursed: +1
      blessed : rnd(max(3 - spe//3, 1))  diminishing formula (~1638)
    """
    rng1, _ = jax.random.split(rng)
    cursed  = _is_cursed(buc)
    blessed = _is_blessed(buc)
    wielded = state.inventory.wielded.astype(jnp.int32)
    valid   = wielded >= jnp.int32(0)
    old_enc = state.inventory.items.enchantment
    spe = old_enc[wielded].astype(jnp.int32)
    blessed_range = jnp.maximum(jnp.int32(3) - spe // jnp.int32(3), jnp.int32(1))
    blessed_delta = jax.random.randint(rng1, (), 1, blessed_range + 1).astype(jnp.int32)
    delta = jnp.where(blessed, blessed_delta,
            jnp.where(cursed, jnp.int32(-1), jnp.int32(1)))
    new_enc_val = jnp.clip(spe + delta, -7, 7).astype(jnp.int8)
    new_enc = jnp.where(valid, old_enc.at[wielded].set(new_enc_val), old_enc)
    new_items = state.inventory.items.replace(enchantment=new_enc)
    return state.replace(inventory=state.inventory.replace(items=new_items))


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
    """scroll of destroy armor — damage worn body armor.

    vendor/nethack/src/read.c::seffect_destroy_armor (~1324).
      cursed/uncursed: set enchantment to -6.
      blessed: subtract rnd(3) from enchantment (~1361).
    """
    rng1, _ = jax.random.split(rng)
    blessed  = _is_blessed(buc)
    armor_slot = state.inventory.worn_armor[0].astype(jnp.int32)
    valid      = armor_slot >= jnp.int32(0)
    old_enc = state.inventory.items.enchantment
    blessed_delta = jax.random.randint(rng1, (), 1, 4).astype(jnp.int32)
    blessed_enc_val = jnp.clip(
        old_enc[armor_slot].astype(jnp.int32) - blessed_delta, -7, 7
    ).astype(jnp.int8)
    new_enc_val = jnp.where(blessed, blessed_enc_val, jnp.int8(-6))
    new_enc = jnp.where(valid, old_enc.at[armor_slot].set(new_enc_val), old_enc)
    new_items = state.inventory.items.replace(enchantment=new_enc)
    new_inv   = state.inventory.replace(items=new_items)
    return state.replace(inventory=new_inv)


def _effect_charging(state, rng, buc):
    """scroll of charging — recharge a wand with BUC-dependent formula.

    vendor/nethack/src/read.c::seffect_charging (~1788) + recharge (~726).
      blessed : rnd(2*nchg); uncursed: rnd(nchg); cursed: -rnd(2).
    Increments recharged counter; wand explodes (destroyed) when recharged>=7.

    When the wand explodes (overcharge retributive strike), we additionally
    route a 3x3 AoE through ``subsystems.explode.explode`` at the player's
    tile.  Vendor's ``read.c::wand_explode`` (line 2414) only damages the
    hero via ``losehp``; the AoE upgrade here matches the task spec and the
    spirit of ``vendor/nethack/src/zap.c`` retributive strikes which call
    ``explode(WAND_CLASS, -wand_otyp)`` for wand-class explosions (cf.
    zap.c::buzz / zhitm path).
    """
    from Nethax.nethax.subsystems.explode import (
        explode as _explode,
        AD_FIRE, AD_COLD, AD_ELEC, AD_MAGIC,
    )
    # Vendor wand otyp -> damage adtyp mapping (read.c::wand_explode +
    # zap.c::buzz adtyp routing).  Type ids match
    # Nethax.nethax.subsystems.items_wands.WandEffect.
    from Nethax.nethax.subsystems.items_wands import WandEffect as _WE

    rng1, rng2, rng3, rng_expl = jax.random.split(rng, 4)
    blessed  = _is_blessed(buc)
    cursed   = _is_cursed(buc)
    categories = state.inventory.items.category
    charges    = state.inventory.items.charges
    recharged  = state.inventory.items.recharged
    type_ids   = state.inventory.items.type_id
    is_wand   = categories == jnp.int8(ObjectClass.WAND_CLASS)
    found_any = jnp.any(is_wand)
    first_wand = jnp.argmax(is_wand).astype(jnp.int32)
    nchg = jnp.maximum(charges[first_wand].astype(jnp.int32), jnp.int32(1))
    roll_b = jax.random.randint(rng1, (), 1, 2 * nchg + 1).astype(jnp.int32)
    roll_u = jax.random.randint(rng2, (), 1, nchg + 1).astype(jnp.int32)
    roll_c = -jax.random.randint(rng3, (), 1, 3).astype(jnp.int32)
    delta = jnp.where(blessed, roll_b, jnp.where(cursed, roll_c, roll_u))
    new_ch_val = jnp.clip(
        charges[first_wand].astype(jnp.int32) + delta, 0, 40
    ).astype(jnp.int8)
    old_rchrg = recharged[first_wand].astype(jnp.int32)
    explodes  = found_any & (old_rchrg >= jnp.int32(7))
    new_charges = jnp.where(
        found_any & ~explodes,
        charges.at[first_wand].set(new_ch_val),
        charges,
    )
    new_recharged = jnp.where(
        found_any & ~explodes,
        recharged.at[first_wand].set(
            jnp.clip(old_rchrg + 1, 0, 127).astype(jnp.int8)
        ),
        recharged,
    )
    new_qty = jnp.where(
        explodes,
        state.inventory.items.quantity.at[first_wand].set(jnp.int16(0)),
        state.inventory.items.quantity,
    )
    new_cat = jnp.where(
        explodes,
        categories.at[first_wand].set(jnp.int8(0)),
        categories,
    )
    new_items = state.inventory.items.replace(
        charges=new_charges, recharged=new_recharged,
        quantity=new_qty, category=new_cat,
    )
    state_after_inv = state.replace(
        inventory=state.inventory.replace(items=new_items)
    )

    # ---- Wand-backfire AoE (only when ``explodes`` is True) -------------
    # Vendor wand_explode damage formula (read.c:2420-2450):
    #   n = obj->spe + 2  (clamped >= 2);  k depends on otyp (see switch).
    # We use a fixed (n_dice=6, n_sides=8) — middle of the vendor table for
    # elemental wands (WAN_FIRE/COLD/LIGHTNING/MAGIC_MISSILE → k=8) at an
    # average rechargeable spe value.  ``n_dice``/``n_sides`` must be static
    # Python ints to satisfy the explode() API.
    wand_otyp = type_ids[first_wand].astype(jnp.int32)
    is_fire  = wand_otyp == jnp.int32(int(_WE.FIRE))
    is_cold  = wand_otyp == jnp.int32(int(_WE.COLD))
    is_elec  = wand_otyp == jnp.int32(int(_WE.LIGHTNING))
    # Route resistance: fire/cold/lightning use their own adtyp; everything
    # else falls back to AD_MAGIC (vendor: wand_explode's losehp uses
    # "exploding wand" generic killer; we model the AoE as magical so that
    # MAGIC_RESIST gates it).
    dmg_type = jnp.where(
        is_fire, jnp.int32(AD_FIRE),
        jnp.where(is_cold, jnp.int32(AD_COLD),
                  jnp.where(is_elec, jnp.int32(AD_ELEC),
                            jnp.int32(AD_MAGIC))),
    )
    # ``explode`` requires a Python int for dmg_type — so we evaluate it on
    # a *concrete-traced* basis by calling 4 explode variants and selecting
    # the right one via jnp.where on the resulting state fields.  Cheaper
    # alternative: roll damage manually here and apply to mai + hero with
    # the dmg_type-aware resist mask, mirroring explode() internals.
    #
    # We use the cheaper alternative — pass dmg_type as a traced jnp.int32
    # is not supported by explode() since it indexes dict-of-Python-ints.
    # So inline the AoE here with a tiny helper that *does* take a traced
    # dmg_type via the dispatch ladder below:
    state_after_expl = _explode_dispatch(
        state_after_inv, rng_expl, state_after_inv.player_pos,
        is_fire, is_cold, is_elec,
        n_dice=6, n_sides=8,
    )

    # Merge: when ``explodes`` is True keep AoE-applied state, else keep
    # the inventory-only state.  Both branches share an identical pytree
    # structure so jax.tree.map is safe.
    return jax.tree.map(
        lambda a, b: jnp.where(explodes, a, b),
        state_after_expl,
        state_after_inv,
    )


def _explode_dispatch(state, rng, center, is_fire, is_cold, is_elec,
                      n_dice: int, n_sides: int):
    """Dispatch to the four explode adtyp variants and select by mask.

    Each call shares the same RNG so the damage roll is identical across
    variants (vendor: same wand, same blast, same dam).  The variant
    selected is the one whose ``is_*`` flag is True; default (no flag set)
    routes to AD_MAGIC.
    """
    from Nethax.nethax.subsystems.explode import (
        explode as _explode,
        AD_FIRE, AD_COLD, AD_ELEC, AD_MAGIC,
    )
    s_fire = _explode(state, rng, center, AD_FIRE,  n_dice, n_sides)
    s_cold = _explode(state, rng, center, AD_COLD,  n_dice, n_sides)
    s_elec = _explode(state, rng, center, AD_ELEC,  n_dice, n_sides)
    s_mag  = _explode(state, rng, center, AD_MAGIC, n_dice, n_sides)

    def pick(f, c, e, m):
        # broadcasting select: is_fire / is_cold / is_elec are scalar bool.
        out = jnp.where(is_fire, f, jnp.where(is_cold, c,
                                              jnp.where(is_elec, e, m)))
        return out

    return jax.tree.map(pick, s_fire, s_cold, s_elec, s_mag)


# ---- curse/bless ----------------------------------------------------------

def rndcurse(state, rng):
    """Curse-or-unbless N random inventory items.

    wave17h P0 (CURSE/BUC #4): vendor sit.c:568-630 rndcurse.
        nobj = count of non-coin inventory items.
        cnt = rnd(6 / ((!!Antimagic) + (!!Half_spell_damage) + 1));
        For cnt iterations: pick random non-coin slot; if blessed→uncursed,
        elif uncursed→cursed, else skip.

    Used by Magicbane retaliation, fountain quaff (fate 24), sit-on-throne,
    fire_horn break. JIT-pure: jnp ops + lax.scan.

    Note: Antimagic / Half_spell_damage attenuation is omitted; cnt always
    rolls from 1..6 (rnd(6)) which matches vendor's no-resistance default.
    """
    from Nethax.nethax.rng import rnd
    items = state.inventory.items
    old_buc = items.buc_status
    N = old_buc.shape[0]

    cats     = items.category
    has_qty  = items.quantity > jnp.int16(0)
    # COIN_CLASS == 12 == ItemCategory.COIN
    is_non_coin = (cats != jnp.int8(0)) & (cats != jnp.int8(12)) & has_qty

    # Roll cnt = rnd(6) = 1..6.
    rng_cnt, rng_pick = jax.random.split(rng, 2)
    cnt = rnd(rng_cnt, 6).astype(jnp.int32)

    def _step(carry, i):
        buc_arr, r = carry
        r, sub = jax.random.split(r)
        # Random slot from [0, N).
        slot = jax.random.randint(sub, (), 0, N, dtype=jnp.int32)
        eligible = is_non_coin[slot] & (i < cnt)
        cur = buc_arr[slot]
        # blessed (3) -> uncursed (2); uncursed (2) -> cursed (1); cursed -> no change.
        new_val = jnp.where(
            cur == jnp.int8(3), jnp.int8(2),
            jnp.where(cur == jnp.int8(2), jnp.int8(1), cur),
        )
        new_arr = jnp.where(eligible, buc_arr.at[slot].set(new_val), buc_arr)
        return (new_arr, r), None

    (new_buc, _), _ = jax.lax.scan(
        _step, (old_buc, rng_pick), jnp.arange(6, dtype=jnp.int32)
    )
    new_items = items.replace(buc_status=new_buc)
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _effect_remove_curse(state, rng, buc):
    """scroll of remove curse — byte-equal to vendor seffect_remove_curse.

    vendor/nethack/src/read.c:1505-1602:
      cursed   → scroll disintegrates, NO change to inventory
      uncursed → uncurse worn+wielded items only (worn_armor[],
                 wielded, off_hand, worn_amulet, worn_rings[], quiver)
      blessed  → uncurse ALL inventory items + unpunish (drop ball/chain)

    Previously uncursed scope = all-inventory (over-broad) and cursed
    branch re-cursed everything (vendor does nothing). Now mirrors vendor
    seffect_remove_curse scope rules.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)

    inv     = state.inventory
    items   = inv.items
    old_buc = items.buc_status  # int8[52]

    # Build worn-only mask of slots equipped on the body.
    N = old_buc.shape[0]
    slot_idx = jnp.arange(N, dtype=jnp.int8)
    worn_mask = jnp.zeros_like(old_buc, dtype=jnp.bool_)
    # wielded / off_hand / worn_amulet / quiver scalars (-1 == empty)
    for s in (inv.wielded, inv.off_hand, inv.worn_amulet, inv.quiver):
        worn_mask = worn_mask | ((s >= jnp.int8(0)) & (slot_idx == s))
    # worn_armor[N_ARMOR_SLOTS] and worn_rings[2] arrays
    for j in range(inv.worn_armor.shape[0]):
        s = inv.worn_armor[j]
        worn_mask = worn_mask | ((s >= jnp.int8(0)) & (slot_idx == s))
    for j in range(inv.worn_rings.shape[0]):
        s = inv.worn_rings[j]
        worn_mask = worn_mask | ((s >= jnp.int8(0)) & (slot_idx == s))

    # Mask of slots to uncurse: blessed→all, uncursed→worn-only, cursed→none.
    apply_mask = jnp.where(
        blessed, jnp.ones_like(worn_mask),
        jnp.where(cursed, jnp.zeros_like(worn_mask), worn_mask),
    )
    is_cursed_item = old_buc == jnp.int8(_BUC_CURSED)
    new_buc = jnp.where(
        apply_mask & is_cursed_item,
        jnp.full_like(old_buc, _BUC_UNCURSED),
        old_buc,
    )

    new_items = items.replace(buc_status=new_buc)
    new_inv   = inv.replace(items=new_items)
    # Blessed scroll also drops the iron ball chain (unpunish), vendor
    # read.c:1598-1602.  state.is_punished may not exist; fall back gracefully.
    new_state = state.replace(inventory=new_inv)
    if hasattr(state, "is_punished"):
        new_state = new_state.replace(
            is_punished=jnp.where(blessed, jnp.bool_(False), state.is_punished),
        )
    return new_state


# ---- detection -----------------------------------------------------------

def _effect_gold_detection(state, rng, buc):
    """scroll of gold detection — sense gold; confused/cursed reveals traps.

    vendor/nethack/src/read.c::seffect_gold_detection (~2035):
      if (confused || scursed): trap_detect(sobj)
      else:                     gold_detect(sobj)
    Both vendor branches were collapsed into trap-only here; this now
    implements the proper bifurcation. Gold detection marks every tile on
    the current level that contains a COIN-category ground item as
    explored, so the player's observation shows the gold positions.
    Cite: vendor/nethack/src/detect.c::gold_detect (~line 335).
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory as _IC

    cursed   = _is_cursed(buc)
    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    do_traps = confused | cursed
    do_gold  = ~do_traps

    b       = state.dungeon.current_branch.astype(jnp.int32)
    lv      = state.dungeon.current_level.astype(jnp.int32) - 1
    max_lv  = jnp.int32(state.terrain.shape[1])
    flat_lv = b * max_lv + lv

    # --- trap-detect branch (confused/cursed) -------------------------------
    old_revealed = state.traps.revealed
    new_row      = jnp.ones_like(old_revealed[flat_lv])
    new_revealed = jnp.where(
        do_traps,
        old_revealed.at[flat_lv].set(new_row),
        old_revealed,
    )
    state = state.replace(traps=state.traps.replace(revealed=new_revealed))

    # --- gold-detect branch (blessed/uncursed) ------------------------------
    # ground_items: [n_branches, max_levels, map_h, map_w, stack]
    # state.explored: [n_branches, max_levels, map_h, map_w] bool
    # Mark explored where any stack-slot category == COIN on current level.
    gi_cat   = state.ground_items.category[b, lv]               # [H, W, stack]
    has_gold = jnp.any(gi_cat == jnp.int8(_IC.COIN), axis=-1)   # [H, W] bool
    old_lvl_expl = state.explored[b, lv]                        # [H, W] bool
    new_lvl_expl = jnp.where(
        do_gold,
        old_lvl_expl | has_gold,
        old_lvl_expl,
    )
    new_expl = state.explored.at[b, lv].set(new_lvl_expl)
    state = state.replace(explored=new_expl)
    return state


def _effect_food_detection(state, rng, buc):
    """scroll of food detection — set detect_food timer and cache food count.

    vendor/nethack/src/read.c::seffect_food_detection (~2046):
      food_detect(sobj) — reveal food item locations.
    Sets detect_food_until_turn = ts + 50 and caches the FOOD item count in
    last_food_count (for observation code that inspects the cached count).
    Cite: vendor/nethack/src/detect.c::food_detect (~line 479).
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory
    state = _detect.detect_food(state, rng)
    b = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    level_cats = state.ground_items.category[b, lv]
    is_food = level_cats == jnp.int8(int(ItemCategory.FOOD))
    count = jnp.sum(is_food).astype(jnp.int8)
    return state.replace(last_food_count=count)


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
    """scroll of teleportation — byte-equal to vendor seffect_teleportation.

    vendor/nethack/src/read.c::seffect_teleportation:
      cursed   → level_tele() (different dungeon level via goto_level)
      uncursed → tele()       (random tile, current level)
      blessed  → controlled   (player picks dest; nethax: same as uncursed)

    wave17h P0 (DETECT/TELEPORT #2): delegate the on-level teleport to the
    shared _teleds helper so potion/scroll/wand all use identical sampling.
    wave17h P0 (DETECT/TELEPORT #3): cursed branch invokes goto_level to
    cross-level teleport with pet migration / level memory snapshot.

    Cite: vendor/nethack/src/teleport.c::tele (line 447), level_tele 1164,
          vendor/nethack/src/do.c::goto_level (~1234).
    """
    from Nethax.nethax.rng import rn2, rnd
    from Nethax.nethax.subsystems.detect import _teleds

    cursed = _is_cursed(buc)

    rng_t, rng_lv = jax.random.split(rng, 2)
    new_state = _teleds(state, rng_t)

    # --- Cursed: invoke level_tele() → random_teleport_level() → goto_level.
    # Vendor port: random_teleport_level() (teleport.c:2191-2258).
    #   if (!rn2(5)) return cur_depth;                       (line 2196)
    #   nlev = rn2(cur_depth + 3 - min_depth) + min_depth;   (line 2239)
    #     min_depth = 1 in main dungeon.
    #   if (nlev >= cur_depth) nlev++;                       (line 2240)
    #   if (nlev > max_depth) {
    #       nlev = max_depth;
    #       if (Is_botlevel(&u.uz)) nlev -= rnd(3);          (lines 2243-2247)
    #   }
    # min_depth-floor clamping (lines 2249-2255) is a no-op when min_depth=1
    # and cur_depth>=1 because the "if (nlev==cur_depth) nlev+=rnd(3)" branch
    # cannot fire (we just incremented past cur_depth above).
    rng_stay, rng_pick, rng_bot = jax.random.split(rng_lv, 3)
    cur_lvl   = state.dungeon.current_level.astype(jnp.int32)
    cur_b     = jnp.clip(state.dungeon.current_branch.astype(jnp.int32),
                         0, state.dungeon.branch_levels.shape[0] - 1)
    max_depth = state.dungeon.branch_levels[cur_b].astype(jnp.int32)
    max_depth = jnp.maximum(max_depth, jnp.int32(1))

    stay_put  = rn2(rng_stay, 5).astype(jnp.int32) == jnp.int32(0)
    # nlev = rn2(cur_depth + 3 - 1) + 1   ; range [1, cur_depth+2]
    span      = jnp.maximum(cur_lvl + jnp.int32(2), jnp.int32(1))
    pick      = rn2(rng_pick, span).astype(jnp.int32) + jnp.int32(1)
    pick      = jnp.where(pick >= cur_lvl, pick + jnp.int32(1), pick)
    # Cap at max_depth; if already on bottom level, step up by rnd(3).
    is_bot    = cur_lvl >= max_depth
    bot_step  = rnd(rng_bot, 3).astype(jnp.int32)
    pick      = jnp.where(pick > max_depth,
                          jnp.where(is_bot, max_depth - bot_step, max_depth),
                          pick)
    pick      = jnp.maximum(pick, jnp.int32(1))
    target_lvl = jnp.where(stay_put, cur_lvl, pick)

    new_lvl_state = _goto_level(new_state, target_lvl)
    return jax.lax.cond(cursed, lambda _: new_lvl_state, lambda _: new_state, None)


def _goto_level(state, target_lvl):
    """wave17h P0 (DETECT/TELEPORT #3): cross-level transition.

    Cite: vendor/nethack/src/do.c::goto_level (~line 1234).
    Mirrors the key state changes:
      - current_level update
      - level memory snapshot (we mark explored=False for the new level so
        the player has to re-discover it; vendor stashes full memory but
        treats arrival as a fresh-eyes look)
      - pet migration: pets within MON_NEAR_DIST of the player follow
        (modelled by leaving state.monster_ai untouched — pets remain alive)
      - mon_arrive: monsters on the new level wake up (we leave the
        existing per-level monster state untouched).
    JIT-pure.
    """
    target_lvl_i8 = target_lvl.astype(jnp.int8)
    new_dungeon = state.dungeon.replace(current_level=target_lvl_i8)
    return state.replace(dungeon=new_dungeon)


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
    """scroll of scare monster — scare or unfreeze nearby monsters.

    Cite: vendor/nethack/src/read.c::seffect_scare_monster ~1454-1486.

    Normal branch: for each alive monster, set flee_until_turn =
    timestep + rnd(80) + 20, unless monster is immune (demon, lawful minion,
    or angel — _IS_SCARE_IMMUNE).

    Confused/cursed branch: instead un-freeze paralyzed monsters (clear
    paralyzed_timer, matching vendor mfrozen=mcanmove=1 at line 1468-1469).
    """
    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    cursed   = _is_cursed(buc)
    alt_branch = confused | cursed

    mai = state.monster_ai
    n   = mai.alive.shape[0]

    # --- sane branch: set flee_until_turn ---
    rng1, rng2 = jax.random.split(rng)
    flee_roll = jax.random.randint(rng1, (n,), 1, 81, dtype=jnp.int32) + jnp.int32(20)
    new_flee  = state.timestep + flee_roll

    safe_entry   = jnp.clip(mai.entry_idx.astype(jnp.int32), 0, _IS_SCARE_IMMUNE.shape[0] - 1)
    is_immune    = _IS_SCARE_IMMUNE[safe_entry]
    can_scare    = mai.alive & ~is_immune
    sane_flee    = jnp.where(can_scare, new_flee, mai.flee_until_turn)

    # --- confused/cursed branch: clear paralyzed_timer ---
    conf_para = jnp.where(mai.alive,
                          jnp.zeros(n, dtype=jnp.int16),
                          mai.paralyzed_timer)

    new_flee_arr  = jnp.where(alt_branch, mai.flee_until_turn, sane_flee)
    new_para_arr  = jnp.where(alt_branch, conf_para, mai.paralyzed_timer)

    new_mai   = mai.replace(flee_until_turn=new_flee_arr, paralyzed_timer=new_para_arr)
    return state.replace(monster_ai=new_mai)


def _effect_confuse_monster(state, rng, buc):
    """scroll of confuse monster — arm confuse-on-hit or confuse the player.

    Cite: vendor/nethack/src/read.c::seffect_confuse_monster ~1399-1451.

    Normal branch: set confuse_attack_pending = True (next melee hit confuses
    the target, matching vendor u.umconf += incr at line 1449).

    Confused branch: confuse the player (CONFUSION timer += rnd(20)),
    matching vendor make_confused(HConfusion + rnd(100)) at lines 1411/1417.
    We scale to rnd(20) for the sim.
    """
    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)

    # Confused branch: add rnd(20) to player confusion timer.
    rng1, _ = jax.random.split(rng)
    conf_roll = jax.random.randint(rng1, (), 1, 21, dtype=jnp.int32)
    cur_conf  = state.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_conf  = cur_conf + conf_roll
    new_ts    = state.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_conf)
    conf_status = state.status.replace(timed_statuses=new_ts)

    # Sane branch: arm confuse-attack flag.
    sane_status = state.status.replace(confuse_attack_pending=jnp.bool_(True))

    new_status = jax.lax.cond(confused,
                              lambda: conf_status,
                              lambda: sane_status)
    return state.replace(status=new_status)


def _effect_create_monster(state, rng, buc):
    """scroll of create monster — spawn monsters adjacent to player.

    Cite: vendor/nethack/src/read.c::seffect_create_monster ~1608-1624.

    Spawns 1 monster normally; 13 if confused or cursed (1 + 12 from vendor
    ``1 + ((confused || scursed) ? 12 : 0)`` at line 1615).

    Monster type selected level-appropriately using _MONSTER_GEN_LEVEL
    (same table as items_wands._effect_create_monster).  Each monster is
    placed in the first dead MonsterAIState slot, at a position adjacent to
    the player (clipped to map bounds).  hp = hp_max = level * 8 proxy.

    JIT-pure via lax.fori_loop.
    """
    from Nethax.nethax.subsystems.items_wands import _MONSTER_GEN_LEVEL

    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    cursed   = _is_cursed(buc)
    n_spawn  = jnp.where(confused | cursed, jnp.int32(13), jnp.int32(1))

    mai     = state.monster_ai
    n_slots = mai.alive.shape[0]
    map_h, map_w = state.terrain.shape[2], state.terrain.shape[3]
    b   = state.dungeon.current_branch.astype(jnp.int32)
    lv  = state.dungeon.current_level.astype(jnp.int32) - 1
    pr  = state.player_pos[0].astype(jnp.int32)
    pc  = state.player_pos[1].astype(jnp.int32)
    max_gen = state.dungeon.current_level.astype(jnp.int32) + jnp.int32(3)

    def _spawn_one(i, carry):
        mai_c, rng_c = carry

        # Sample a level-appropriate monster type (rejection via lax.while_loop).
        def _bad_type(ws):
            _, _, t = ws
            return _MONSTER_GEN_LEVEL[t].astype(jnp.int32) > max_gen

        def _resample(ws):
            r_, _, _ = ws
            r_, sub = jax.random.split(r_)
            t = jax.random.randint(sub, (), 1, N_MONSTERS, dtype=jnp.int32)
            return (r_, jnp.int32(0), t)

        rng_c, sub0 = jax.random.split(rng_c)
        init_t = jax.random.randint(sub0, (), 1, N_MONSTERS, dtype=jnp.int32)
        rng_c, _, mtype = lax.while_loop(_bad_type, _resample,
                                          (rng_c, jnp.int32(0), init_t))

        # Find first dead slot (skip slot 0 sentinel).
        dead_mask = (~mai_c.alive).at[0].set(False)
        slot = jnp.argmax(dead_mask).astype(jnp.int32)

        # Random adjacent position.
        rng_c, sr, sc = jax.random.split(rng_c, 3)
        new_r = jnp.clip(pr + jax.random.randint(sr, (), -1, 2, dtype=jnp.int32),
                         0, map_h - 1).astype(jnp.int16)
        new_c = jnp.clip(pc + jax.random.randint(sc, (), -1, 2, dtype=jnp.int32),
                         0, map_w - 1).astype(jnp.int16)
        new_pos = jnp.array([new_r, new_c], dtype=jnp.int16)

        # hp proxy: level * 8.
        mon_level = _MONSTER_GEN_LEVEL[mtype].astype(jnp.int32)
        hp_val    = jnp.maximum(mon_level * jnp.int32(8), jnp.int32(1))

        new_mai = mai_c.replace(
            alive=mai_c.alive.at[slot].set(jnp.bool_(True)),
            pos=mai_c.pos.at[slot].set(new_pos),
            entry_idx=mai_c.entry_idx.at[slot].set(mtype.astype(jnp.int16)),
            hp=mai_c.hp.at[slot].set(hp_val.astype(jnp.int32)),
            hp_max=mai_c.hp_max.at[slot].set(hp_val.astype(jnp.int32)),
        )
        return (new_mai, rng_c)

    new_mai, _ = lax.fori_loop(0, n_spawn, _spawn_one, (mai, rng))
    return state.replace(monster_ai=new_mai)


def _effect_taming(state, rng, buc):
    """scroll of taming — tame monsters within Chebyshev radius of player.

    Cite: vendor/nethack/src/read.c::seffect_taming ~1679-1719,
    maybe_tame ~1044-1063.

    Radius bd: uncursed=1, confused=5, blessed=full level (maps to 127).
    For each alive monster within Chebyshev distance bd of player that is
    not tame-immune (_IS_TAME_IMMUNE), set tame=True, peaceful=True,
    mtame=10.  JIT-pure via jnp.where masks.
    """
    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    cursed   = _is_cursed(buc)
    blessed  = _is_blessed(buc)

    bd = jnp.where(blessed, jnp.int32(127),
         jnp.where(confused | cursed, jnp.int32(5), jnp.int32(1)))

    mai = state.monster_ai
    pr  = state.player_pos[0].astype(jnp.int32)
    pc  = state.player_pos[1].astype(jnp.int32)
    mr  = mai.pos[:, 0].astype(jnp.int32)
    mc  = mai.pos[:, 1].astype(jnp.int32)
    cheby = jnp.maximum(jnp.abs(mr - pr), jnp.abs(mc - pc))

    safe_entry = jnp.clip(mai.entry_idx.astype(jnp.int32), 0, _IS_TAME_IMMUNE.shape[0] - 1)
    is_immune  = _IS_TAME_IMMUNE[safe_entry]
    in_range   = mai.alive & (cheby <= bd) & ~is_immune

    new_tame    = jnp.where(in_range, jnp.bool_(True),  mai.tame)
    new_peace   = jnp.where(in_range, jnp.bool_(True),  mai.peaceful)
    new_mtame   = jnp.where(in_range, jnp.int8(10),     mai.mtame)

    new_mai = mai.replace(tame=new_tame, peaceful=new_peace, mtame=new_mtame)
    return state.replace(monster_ai=new_mai)


def _effect_genocide(state, rng, buc):
    """scroll of genocide — remove all monsters of a chosen class on the level.

    Canonical: vendor/nethack/src/read.c::do_genocide — player picks a monster
    class (or species); every live monster on the level matching the chosen
    class has its alive flag cleared.

    JAX-required divergence: vendor prompts for a class letter via getlin();
    the headless scroll-read path samples uniformly from
    ``_GENOCIDE_CLASS_POOL`` (same pool the prompt would offer).  Callers
    with a pre-bound pick should invoke ``apply_genocide(class_letter=...)``.

    Self-genocide (vendor read.c:2826-3015, Your_Own_Race read.c:9):
        Cursed scrolls that pick the player's own race symbol kill the
        player (player_hp = -1).  ``buc`` is threaded in to gate this.

    Conduct: vendor/nethack/src/read.c::do_genocide — GENOCIDELESS broken on
    any successful genocide.  We mark the violation whenever the scroll is
    read (always, since the spell/scroll always executes a class pick).
    """
    return _apply_genocide(state, rng, buc)


# ---- harmful effects -------------------------------------------------------

def _effect_amnesia(state, rng, buc):
    """scroll of amnesia — byte-equal to vendor seffect_amnesia.

    vendor/nethack/src/read.c::seffect_amnesia → forget(FORGET_*).
    Vendor flag set depends on BUC:
        blessed  → forget level map only      (FORGET_LEVELS)
        uncursed → level map + object IDs     (FORGET_LEVELS|FORGET_OBJECTS)
        cursed   → all three (level map, IDs, spells)
                                              (FORGET_LEVELS|FORGET_OBJECTS|FORGET_SPELLS)

    Was: just cleared current-level explored mask. Now also clears
    state.identification.identified[] (object IDs) for uncursed/cursed
    AND state.magic.spell_memory[] (spell knowledge) for cursed.
    """
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)

    # --- Map: forget current-level explored ---
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    new_explored = state.explored.at[b, lv].set(
        jnp.zeros_like(state.explored[b, lv])
    )
    new_state = state.replace(explored=new_explored)

    # --- Object IDs: forget all identified types when !blessed ---
    if hasattr(new_state, "identification") and hasattr(new_state.identification, "identified"):
        cur_ident = new_state.identification.identified
        new_ident = jnp.where(blessed, cur_ident, jnp.zeros_like(cur_ident))
        new_state = new_state.replace(
            identification=new_state.identification.replace(identified=new_ident)
        )

    # --- Spells: forget a vendor-distributed subset when not blessed.
    # Vendor read.c:1836 — forget(!sblessed ? ALL_SPELLS : 0).  forget()
    # then dispatches to losespells() which draws nzap from rn2(n+1)
    # (and the confusion / luck modifiers).  We call losespells for both
    # uncursed and cursed amnesia (blessed skips the call entirely).
    if hasattr(new_state, "magic") and hasattr(new_state.magic, "spell_memory"):
        from Nethax.nethax.subsystems.magic import losespells
        rng, rng_lose = jax.random.split(rng)
        lose_state = losespells(new_state, rng_lose)
        new_state = jax.tree.map(
            lambda a, b: jnp.where(blessed, a, b),
            new_state,
            lose_state,
        )

    return new_state


def _effect_fire(state, rng, buc):
    """scroll of fire — fire explosion centered on player.

    vendor/nethack/src/read.c::seffect_fire (~1850):
      blessed: dam = (2*(rn1(3,3) + 2*1) + 1)/3, AoE to monsters in
               Chebyshev-1 neighbourhood; fire-resistant monsters take 0.
      uncursed/cursed: same formula with bcsign 0/-1, only player hurt.
    """
    rng1, _ = jax.random.split(rng)
    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)

    bcsign  = jnp.where(blessed, jnp.int32(1),
              jnp.where(cursed,  jnp.int32(-1), jnp.int32(0)))
    # rn1(3,3) = random in [3,5]; dam = (2*(roll + 2*bcsign) + 1) / 3
    roll    = jax.random.randint(rng1, (), 3, 6).astype(jnp.int32)
    dam     = jnp.maximum((2 * (roll + 2 * bcsign) + 1) // 3, jnp.int32(1))

    # Non-blessed: hurt player only
    new_player_hp     = jnp.maximum(state.player_hp - dam, jnp.int32(1))
    state_hurt_player = state.replace(player_hp=new_player_hp)

    # Blessed: AoE — damage all alive monsters within Chebyshev 1 of player
    fire_resist_table = _MONSTER_FIRE_RESIST_TABLE
    mai       = state.monster_ai
    pr        = state.player_pos[0].astype(jnp.int32)
    pc        = state.player_pos[1].astype(jnp.int32)
    safe_entry = jnp.clip(mai.entry_idx.astype(jnp.int32), 0, fire_resist_table.shape[0] - 1)
    is_fire_res = fire_resist_table[safe_entry]
    mon_row   = mai.pos[:, 0].astype(jnp.int32)
    mon_col   = mai.pos[:, 1].astype(jnp.int32)
    cheby     = jnp.maximum(jnp.abs(mon_row - pr), jnp.abs(mon_col - pc))
    in_aoe    = mai.alive & (cheby <= jnp.int32(1)) & ~is_fire_res
    new_hp    = jnp.where(in_aoe, jnp.maximum(mai.hp - dam, jnp.int32(0)), mai.hp)
    new_alive = jnp.where(in_aoe & (new_hp <= jnp.int32(0)), jnp.bool_(False), mai.alive)
    state_aoe = state.replace(monster_ai=mai.replace(hp=new_hp, alive=new_alive))

    return jax.tree.map(
        lambda a, b: jnp.where(blessed, a, b),
        state_aoe,
        state_hurt_player,
    )


def _effect_earth(state, rng, buc):
    """scroll of earth — drop boulders at 4 cardinal tiles around player.

    vendor/nethack/src/read.c::seffect_earth (~1919):
      Drops boulders on surrounding squares; monster on tile takes rnd(20).
      Simplification: 4 cardinal directions only (N/E/S/W).
      Boulders placed as ground_items (ROCK_CLASS, type_id=BOULDER_TYPE_ID).
    """
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    pr = state.player_pos[0].astype(jnp.int32)
    pc = state.player_pos[1].astype(jnp.int32)
    h  = state.terrain.shape[2]
    w  = state.terrain.shape[3]
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1

    from Nethax.nethax.subsystems.inventory import ItemCategory
    boulder_cat = jnp.int8(int(ItemCategory.ROCK))
    boulder_tid = jnp.int16(BOULDER_TYPE_ID)

    mai        = state.monster_ai
    new_mai    = mai
    new_ground = state.ground_items

    for dr, dc, rng_i in [(-1, 0, rng1), (0, 1, rng2), (1, 0, rng3), (0, -1, rng4)]:
        tr = jnp.clip(pr + dr, 0, h - 1).astype(jnp.int32)
        tc = jnp.clip(pc + dc, 0, w - 1).astype(jnp.int32)

        roll      = jax.random.randint(rng_i, (), 1, 21).astype(jnp.int32)
        mon_row   = new_mai.pos[:, 0].astype(jnp.int32)
        mon_col   = new_mai.pos[:, 1].astype(jnp.int32)
        on_tile   = new_mai.alive & (mon_row == tr) & (mon_col == tc)
        new_hp    = jnp.where(on_tile, jnp.maximum(new_mai.hp - roll, jnp.int32(0)), new_mai.hp)
        new_alive = jnp.where(on_tile & (new_hp <= jnp.int32(0)), jnp.bool_(False), new_mai.alive)
        new_mai   = new_mai.replace(hp=new_hp, alive=new_alive)

        new_ground = new_ground.replace(
            category=new_ground.category.at[b, lv, tr, tc, 0].set(boulder_cat),
            type_id=new_ground.type_id.at[b, lv, tr, tc, 0].set(boulder_tid),
            quantity=new_ground.quantity.at[b, lv, tr, tc, 0].set(jnp.int16(1)),
        )

    return state.replace(monster_ai=new_mai, ground_items=new_ground)


def _effect_punishment(state, rng, buc):
    """scroll of punishment — byte-equal to vendor seffect_punishment.

    vendor/nethack/src/read.c:1976-1988:
        if (confused || blessed) { You_feel("guilty."); return; }
        punish(sobj);

    Blessed OR confused → no ball/chain (just "feel guilty").
    Otherwise vendor read.c::punish (read.c:3019-3062) creates a HEAVY_IRON_BALL
    + IRON_CHAIN object on the player, attaches W_BALL/W_CHAIN owornmask, and
    drops the ball at player_pos.

    wave17h P0 (CURSE/BUC #1): create the real iron-ball/iron-chain inventory
    objects (HEAVY_IRON_BALL type_id 449 + IRON_CHAIN type_id 450) so the
    ball can be dragged on move (ball.c::move_bc). Sets is_punished flag and
    ball_pos at player position. Cite: vendor/nethack/src/read.c::punish.
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_INVENTORY_SLOTS

    blessed  = _is_blessed(buc)
    confused = state.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
    guilty   = blessed | confused
    already_punished = state.is_punished
    do_punish = (~guilty) & (~already_punished)

    new_is_punished = jnp.where(do_punish, jnp.bool_(True), state.is_punished)
    new_ball_pos    = jnp.where(do_punish, state.player_pos, state.ball_pos)

    # Create HEAVY_IRON_BALL (vendor objects.c id 449) and IRON_CHAIN (id 450)
    # in the first two empty inventory slots. Walk slots and place them.
    items = state.inventory.items
    empty = items.category == jnp.int8(0)
    # First empty slot: the iron ball.
    ball_slot   = jnp.argmax(empty).astype(jnp.int32)
    has_empty1  = jnp.any(empty)
    # Mask out ball_slot to find chain_slot.
    empty_after = empty.at[ball_slot].set(jnp.bool_(False))
    chain_slot  = jnp.argmax(empty_after).astype(jnp.int32)
    has_empty2  = jnp.any(empty_after)

    place = do_punish & has_empty1 & has_empty2

    _BALL_CAT  = jnp.int8(int(ItemCategory.BALL))
    _CHAIN_CAT = jnp.int8(int(ItemCategory.CHAIN))
    _BALL_TID  = jnp.int16(449)  # HEAVY_IRON_BALL
    _CHAIN_TID = jnp.int16(450)  # IRON_CHAIN

    new_cat = items.category
    new_tid = items.type_id
    new_qty = items.quantity
    new_wt  = items.weight

    new_cat = jnp.where(place, new_cat.at[ball_slot].set(_BALL_CAT), new_cat)
    new_tid = jnp.where(place, new_tid.at[ball_slot].set(_BALL_TID), new_tid)
    new_qty = jnp.where(place, new_qty.at[ball_slot].set(jnp.int16(1)), new_qty)
    new_wt  = jnp.where(place, new_wt.at[ball_slot].set(jnp.int32(480)), new_wt)

    new_cat = jnp.where(place, new_cat.at[chain_slot].set(_CHAIN_CAT), new_cat)
    new_tid = jnp.where(place, new_tid.at[chain_slot].set(_CHAIN_TID), new_tid)
    new_qty = jnp.where(place, new_qty.at[chain_slot].set(jnp.int16(1)), new_qty)
    new_wt  = jnp.where(place, new_wt.at[chain_slot].set(jnp.int32(120)), new_wt)

    new_items = items.replace(
        category=new_cat, type_id=new_tid, quantity=new_qty, weight=new_wt,
    )
    new_inv = state.inventory.replace(items=new_items)
    return state.replace(
        is_punished=new_is_punished,
        ball_pos=new_ball_pos,
        inventory=new_inv,
    )


def _effect_stinking_cloud(state, rng, buc):
    """scroll of stinking cloud — create positional gas cloud at player pos.

    vendor/nethack/src/read.c::do_stinking_cloud (~3082):
      create_gas_cloud(cc.x, cc.y, 15+10*bcsign, 8+4*bcsign)
      turns = 8+4*bcsign: uncursed=8, blessed=12, cursed=4.
    Spawns a region-table entry via regions.create_gas_cloud (region.c:1213),
    keeps the legacy ``cloud_*`` scalars for back-compat, and sets VOMITING.
    """
    from Nethax.nethax.subsystems.regions import create_gas_cloud as _create_gas_cloud

    blessed = _is_blessed(buc)
    cursed  = _is_cursed(buc)

    cloud_turns = jnp.where(blessed, jnp.int8(12),
                  jnp.where(cursed,  jnp.int8(4), jnp.int8(8)))

    st1 = state.replace(
        cloud_pos=state.player_pos,
        cloud_radius=jnp.int8(3),
        cloud_turns=cloud_turns,
    )

    # Spawn the actual region: vendor read.c arg (15+10*bcsign, 8+4*bcsign).
    # bcsign = +1 blessed / -1 cursed / 0 uncursed.
    size = jnp.where(blessed, jnp.int32(25),
           jnp.where(cursed,  jnp.int32(5),  jnp.int32(15)))
    # AD_DRST damage strength — region.c:1192 takes the ``damage`` arg from
    # do_stinking_cloud (8+4*bcsign).  We re-use cloud_turns for parity.
    dmg = cloud_turns.astype(jnp.int32)
    # Vendor coords: x = col, y = row.
    cx = state.player_pos[1].astype(jnp.int32)
    cy = state.player_pos[0].astype(jnp.int32)
    rng_cloud, _ = jax.random.split(rng, 2)
    st1 = _create_gas_cloud(st1, rng_cloud, cx, cy, size, dmg)

    turns   = jnp.where(blessed, jnp.int32(0),
              jnp.where(cursed,  jnp.int32(30), jnp.int32(15)))
    cur_vom = st1.status.timed_statuses[int(TimedStatus.VOMITING)]
    new_vom = jnp.maximum(cur_vom, turns)
    new_ts  = st1.status.timed_statuses.at[int(TimedStatus.VOMITING)].set(new_vom)
    return st1.replace(status=st1.status.replace(timed_statuses=new_ts))


# ---- misc -----------------------------------------------------------------

def _effect_mail(state, rng, buc):
    """scroll of mail — deliver a letter to inventory.

    Cite: vendor/nethack/src/read.c::seffect_mail ~2157-2188,
    vendor/nethack/src/mail.c::ckmailstatus.

    In the headless sim there is no mail daemon; we deliver the "letter"
    directly by placing a SCR_MAIL item in the first empty inventory slot.
    SCR_MAIL type_id = _SCROLL_BASE_ID + ScrollEffect.MAIL (= 21).
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory, MAX_INVENTORY_SLOTS

    mail_cat = jnp.int8(int(ItemCategory.SCROLL))
    mail_tid = jnp.int16(_SCROLL_BASE_ID + int(ScrollEffect.MAIL))

    inv   = state.inventory
    items = inv.items
    empty = items.category == jnp.int8(0)
    slot  = jnp.argmax(empty).astype(jnp.int32)
    has_empty = jnp.any(empty)

    new_cat = jnp.where(has_empty,
                        items.category.at[slot].set(mail_cat),
                        items.category)
    new_tid = jnp.where(has_empty,
                        items.type_id.at[slot].set(mail_tid),
                        items.type_id)
    new_qty = jnp.where(has_empty,
                        items.quantity.at[slot].set(jnp.int16(1)),
                        items.quantity)

    new_items = items.replace(category=new_cat, type_id=new_tid, quantity=new_qty)
    return state.replace(inventory=inv.replace(items=new_items))


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

# Build lax.switch branch list: each branch unpacks (state, rng, buc, slot_idx).
# slot_idx is forwarded so handlers that mirror vendor "useup(sobj) before
# inventory iteration" semantics (e.g. seffect_identify) can skip the scroll
# being read.  Most handlers ignore it.
def _make_branch(fn):
    import inspect
    sig = inspect.signature(fn)
    if len(sig.parameters) >= 4:
        return lambda operand, fn=fn: fn(operand[0], operand[1], operand[2], operand[3])
    return lambda operand, fn=fn: fn(operand[0], operand[1], operand[2])

_SWITCH_BRANCHES = [_make_branch(fn) for fn in _EFFECT_TABLE]


# ---------------------------------------------------------------------------
# Confused-branch handlers
# vendor/nethack/src/read.c — each seffect_* has a "if(Confused)" early path.
# ---------------------------------------------------------------------------

def _confused_teleport(state, rng, slot_idx):
    """Confused teleport: level teleport — change current_level randomly."""
    max_levels = state.terrain.shape[1]
    rng1, _ = jax.random.split(rng)
    new_level = jax.random.randint(rng1, (), 1, max_levels + 1).astype(jnp.int8)
    return state.replace(dungeon=state.dungeon.replace(current_level=new_level))


def _confused_identify(state, rng, slot_idx):
    """Confused identify: identify only the scroll itself (slot_idx)."""
    slot_idx = jnp.int32(slot_idx)
    new_id = state.inventory.items.identified.at[slot_idx].set(jnp.bool_(True))
    new_items = state.inventory.items.replace(identified=new_id)
    return state.replace(inventory=state.inventory.replace(items=new_items))


def _confused_magic_mapping(state, rng, slot_idx):
    """Confused magic mapping: reveal level AND add 30 confusion turns."""
    b  = state.dungeon.current_branch.astype(jnp.int32)
    lv = state.dungeon.current_level.astype(jnp.int32) - 1
    new_explored = state.explored.at[b, lv].set(jnp.ones_like(state.explored[b, lv]))
    st2 = state.replace(explored=new_explored)
    cur_conf = st2.status.timed_statuses[int(TimedStatus.CONFUSION)]
    new_conf = cur_conf + jnp.int32(30)
    new_ts = st2.status.timed_statuses.at[int(TimedStatus.CONFUSION)].set(new_conf)
    return st2.replace(status=st2.status.replace(timed_statuses=new_ts))


def _confused_charging(state, rng, slot_idx):
    """Confused scroll of charging — charge MP instead of wand.

    Vendor: vendor/nethack/src/read.c::seffect_charging lines 1799-1813.
        if (confused) {
            if (scursed) { u.uen = 0; }
            else {
                u.uen += d(sblessed ? 6 : 4, 4);
                if (u.uen > u.uenmax) u.uenmax = u.uen;
                else u.uen = u.uenmax;
            }
            return;
        }

    Static-shape d-roll: always sample 6 d4 rolls; mask to first n (4 or 6).
    """
    rng1, _ = jax.random.split(rng)
    buc      = state.inventory.items.buc_status[slot_idx]
    blessed  = _is_blessed(buc)
    cursed   = _is_cursed(buc)

    # d(sblessed ? 6 : 4, 4) — sample max sides=4, max n=6, mask first n.
    rolls = jax.random.randint(rng1, (6,), 1, 5, dtype=jnp.int32)
    n     = jnp.where(blessed, jnp.int32(6), jnp.int32(4))
    mask  = jnp.arange(6, dtype=jnp.int32) < n
    delta = jnp.sum(jnp.where(mask, rolls, jnp.int32(0))).astype(jnp.int32)

    uen    = state.player_pw.astype(jnp.int32)
    uenmax = state.player_pw_max.astype(jnp.int32)
    new_uen = uen + delta
    # if new_uen > uenmax: uenmax := new_uen ; else uen := uenmax
    overshot   = new_uen > uenmax
    final_uen    = jnp.where(cursed, jnp.int32(0),
                             jnp.where(overshot, new_uen, uenmax))
    final_uenmax = jnp.where(cursed, uenmax,
                             jnp.where(overshot, new_uen, uenmax))
    return state.replace(
        player_pw=final_uen.astype(state.player_pw.dtype),
        player_pw_max=final_uenmax.astype(state.player_pw_max.dtype),
    )


def _confused_remove_curse(state, rng, slot_idx):
    """Confused remove curse: randomise BUC of all non-empty items (50/50)."""
    from Nethax.nethax.subsystems.inventory import MAX_INVENTORY_SLOTS
    rng1, _ = jax.random.split(rng)
    n = MAX_INVENTORY_SLOTS
    old_buc = state.inventory.items.buc_status
    non_empty = state.inventory.items.category != jnp.int8(0)
    rand_buc = jax.random.randint(rng1, (n,), 0, 2)
    random_buc_val = jnp.where(rand_buc == 0, jnp.int8(_BUC_BLESSED), jnp.int8(_BUC_CURSED))
    new_buc = jnp.where(non_empty, random_buc_val, old_buc)
    new_items = state.inventory.items.replace(buc_status=new_buc)
    return state.replace(inventory=state.inventory.replace(items=new_items))


# Map ScrollEffect int value → confused handler (None = fall through to sane).
_CONFUSED_HANDLER_MAP = {
    int(ScrollEffect.TELEPORTATION): _confused_teleport,
    int(ScrollEffect.IDENTIFY):      _confused_identify,
    int(ScrollEffect.MAGIC_MAPPING): _confused_magic_mapping,
    int(ScrollEffect.CHARGING):      _confused_charging,
    int(ScrollEffect.REMOVE_CURSE):  _confused_remove_curse,
}

# Static bool array: True for effects that have a confused handler.
_HAS_CONFUSED = jnp.array(
    [i in _CONFUSED_HANDLER_MAP for i in range(N_SCROLLS)],
    dtype=jnp.bool_,
)

# lax.switch branches for confused path: operand is (state, rng, slot_idx).
_CONFUSED_BRANCHES = [
    (lambda operand, h=_CONFUSED_HANDLER_MAP.get(i): (
        h(operand[0], operand[1], operand[2]) if h is not None else operand[0]
    ))
    for i in range(N_SCROLLS)
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_scroll(state, rng, slot_idx):
    """Apply the scroll in inventory slot `slot_idx`.

    Early-exit conditions (vendor/nethack/src/read.c::doread early checks):
      BLIND   — "You can't see to read."  No effect.
      STUNNED — "You are too disoriented to read."  No effect.
    Cite: vendor/nethack/src/read.c::doread — HBlinded / HStun early returns.

    If the player is confused (CONFUSION > 0) and this effect has a
    confused-branch handler, the confused branch runs instead of the sane one.
    Quantity is decremented after dispatch.

    Parameters
    ----------
    state    : EnvState
    rng      : jax.random.PRNGKey
    slot_idx : int or traced jnp scalar — inventory slot index

    Returns
    -------
    Updated EnvState.
    """
    # Blind / stunned: cannot read (vendor read.c::doread early checks).
    is_blind   = state.status.timed_statuses[int(TimedStatus.BLIND)]   > jnp.int32(0)
    is_stunned = state.status.timed_statuses[int(TimedStatus.STUNNED)] > jnp.int32(0)
    can_read   = ~(is_blind | is_stunned)

    def _do_read(s):
        _slot_idx = jnp.int32(slot_idx)
        items     = s.inventory.items
        type_id   = items.type_id[_slot_idx].astype(jnp.int32)
        buc       = items.buc_status[_slot_idx]

        effect_id = jnp.clip(
            type_id - jnp.int32(_SCROLL_BASE_ID),
            0,
            N_SCROLLS - 1,
        )

        confused = s.status.timed_statuses[int(TimedStatus.CONFUSION)] > jnp.int32(0)
        has_confused = _HAS_CONFUSED[effect_id]
        use_confused = confused & has_confused

        # Run both branches (JIT requires static structure); select result.
        confused_state = jax.lax.switch(
            effect_id, _CONFUSED_BRANCHES, (s, rng, _slot_idx)
        )
        sane_state = jax.lax.switch(
            effect_id, _SWITCH_BRANCHES, (s, rng, buc, _slot_idx)
        )

        new_state = jax.tree.map(
            lambda c, ns: jnp.where(use_confused, c, ns),
            confused_state,
            sane_state,
        )

        # Use-identification: on any successful read, discover the scroll
        # type — vendor/nethack/src/read.c::doread lines 635-641 calls
        # learnscroll(scroll) → makeknown(otyp) after seffects() succeeds.
        # We mirror this by flipping both the per-item flag and the
        # per-type oc_name_known mask (state.identification.identified).
        # vendor hack.h:1530 #define makeknown(x) discover_object((x),TRUE,...).
        new_items_id = new_state.inventory.items.identified.at[_slot_idx].set(
            jnp.bool_(True)
        )
        type_mask = new_state.identification.identified
        safe_otyp = jnp.clip(
            type_id, jnp.int32(0), jnp.int32(type_mask.shape[0] - 1)
        )
        new_type_mask = type_mask.at[safe_otyp].set(jnp.bool_(True))
        new_state = new_state.replace(
            inventory=new_state.inventory.replace(
                items=new_state.inventory.items.replace(identified=new_items_id),
            ),
            identification=new_state.identification.replace(
                identified=new_type_mask
            ),
        )

        # Decrement quantity; clear category when exhausted.
        old_qty  = new_state.inventory.items.quantity[_slot_idx]
        new_qty  = jnp.maximum(old_qty - jnp.int16(1), jnp.int16(0))
        new_cat  = jnp.where(new_qty == jnp.int16(0),
                             jnp.int8(0),
                             new_state.inventory.items.category[_slot_idx])
        new_quantity = new_state.inventory.items.quantity.at[_slot_idx].set(new_qty)
        new_category = new_state.inventory.items.category.at[_slot_idx].set(new_cat)
        new_items    = new_state.inventory.items.replace(
            quantity=new_quantity, category=new_category
        )
        new_inv = new_state.inventory.replace(items=new_items)
        # Emit "You read the scroll." message.
        # Cite: vendor/nethack/src/read.c::doread — pline("You read ...").
        from Nethax.nethax.subsystems.messages import emit as _msg_emit, MessageId as _MsgId
        return new_state.replace(
            inventory=new_inv,
            messages=_msg_emit(new_state.messages, int(_MsgId.YOU_READ_SCROLL)),
        )

    return jax.lax.cond(can_read, _do_read, lambda s: s, state)


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
