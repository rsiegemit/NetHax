"""Spellbook reading subsystem — study_book logic.

Canonical sources:
  vendor/nethack/src/spell.c::study_book  — learning chance, memory init
  vendor/nethack/include/spell.h          — KEEN = 20000, SPELL_LEV_PW
  vendor/nethack/include/objects.h        — SPELL() table (level, delay)

Wave 8d implementation (vendor-probabilistic formula):
  Vendor spell.c::study_book lines 582-599 (uncursed book path):
    read_ability = ACURR(A_INT) + 4 + u.ulevel/2 - 2 * book_level
    success if rnd(20) <= read_ability   (rnd(20) is 1..20)

  We use player_int for A_INT and player_xl for u.ulevel.
  For Wizard role (role_id=12) we apply an additional +2 bonus matching
  the lenses bonus range (wizard tends toward high INT giving naturally
  higher read_ability; the +2 models the Wizard's studied comprehension
  advantage — see vendor study_book:587-596 wizard-only early check).

  On success: spell_known[spell_id] = True, spell_memory = KEEN + 1
    (vendor incrnknow(i, 1); spell.c line 22 + lines 410/428)
  On failure: no change (side effects like confusion/paralysis are Wave 4+)

BUC handling (vendor spell.c::study_book / cursed_book lines 590-650):
  CURSED  (buc_status == 1): skips success roll; five-branch backfire via
    rnd(20) (1-4 explode, 5-8 paralyze, 9-12 poison, 13-16 amnesia,
    17-20 blank), spell never learned.
    Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
  UNCURSED (buc_status == 2): standard formula (see above).
  BLESSED  (buc_status == 3): +2 bonus to read_ability (vendor line ~555-560).

Wave 8d simplifications:
  - Blank-book and novel detection: treated as unknown spell_id (-1) → no-op
"""

import jax
import jax.numpy as jnp

from Nethax.nethax.rng import rnd, rn1
from Nethax.nethax.subsystems.magic import (
    KEEN,
    MagicState,
    MAX_SPELL_MEMORY,
    N_SPELLS,
    SpellId,
    _SPELL_LEVELS,
)


# Blank-book sentinel: slot_spell_id == -1 means no spell (blank or novel).
BLANK_SPELL_ID = -1

# Wizard role index (role.c: PM_WIZARD = 12).
_ROLE_WIZARD = 12

# Wizard comprehension bonus (models Wizard's higher studied INT advantage).
# Vendor: wizards get an early-out "too difficult?" check (spell.c line 587)
# plus lens bonus (+2).  We give +2 flat for Wizard role.
_WIZARD_STUDY_BONUS = 2

# BUC status constants (containers.BUCStatus — 1=cursed, 2=uncursed, 3=blessed).
_BUC_CURSED   = 1
_BUC_UNCURSED = 2
_BUC_BLESSED  = 3

# Blessed spellbook bonus to read_ability (vendor spell.c study_book ~lines 555-560).
_BLESSED_STUDY_BONUS = 2


# ---------------------------------------------------------------------------
# Study-time formula
# Cite: vendor/nethack/src/spell.c::study_book lines 537-559.
# Vendor switches on objects[booktype].oc_level:
#     level 1-2: delay = -oc_delay
#     level 3-4: delay = -(oc_level - 1) * oc_delay
#     level 5-6: delay = -oc_level * oc_delay
#     level 7:   delay = -8 * oc_delay
# The negative sign in svc.context.spbook.delay simply marks it as a study
# interruption; magnitude is the turn count.  Per-spell oc_delay values are
# the SPELL() macro's `delay` field (vendor objects.h line 1277:
# SPELL(name, desc, sub, prob, delay, level, mgc, dir, color, sn)).
# ---------------------------------------------------------------------------

# oc_delay per spell, indexed by SpellId (objects.h SPELL() entries).
# Source: vendor/nethack/include/objects.h lines 1293-1396.
_SPELL_OC_DELAYS: tuple = (
    6,  # DIG
    2,  # MAGIC_MISSILE
    4,  # FIREBALL
    7,  # CONE_OF_COLD
    1,  # SLEEP
    10, # FINGER_OF_DEATH
    1,  # LIGHT
    1,  # DETECT_MONSTERS
    5,  # HEALING
    1,  # KNOCK
    2,  # FORCE_BOLT
    2,  # CONFUSE_MONSTER
    2,  # CURE_BLINDNESS
    2,  # DRAIN_LIFE
    2,  # SLOW_MONSTER
    3,  # WIZARD_LOCK
    3,  # CREATE_MONSTER
    3,  # DETECT_FOOD
    3,  # CAUSE_FEAR
    3,  # CLAIRVOYANCE
    3,  # CURE_SICKNESS
    6,  # CHARM_MONSTER
    4,  # HASTE_SELF
    3,  # DETECT_UNSEEN
    4,  # LEVITATION
    5,  # EXTRA_HEALING
    5,  # RESTORE_ABILITY
    5,  # INVISIBILITY
    4,  # DETECT_TREASURE
    5,  # REMOVE_CURSE
    5,  # MAGIC_MAPPING
    3,  # IDENTIFY
    8,  # TURN_UNDEAD
    7,  # POLYMORPH
    6,  # TELEPORT_AWAY
    7,  # CREATE_FAMILIAR
    8,  # CANCELLATION
    1,  # PROTECTION
    3,  # JUMPING
    3,  # STONE_TO_FLESH
    2,  # CHAIN_LIGHTNING
    1,  # FLAME_SPHERE
    1,  # FREEZE_SPHERE
)


def study_book_delay(book_level: int, oc_delay: int) -> int:
    """Return the number of turns required to study a spellbook.

    Mirrors the vendor switch in spell.c::study_book (lines 537-559):
        level 1-2: delay = oc_delay
        level 3-4: delay = (level - 1) * oc_delay
        level 5-6: delay = level * oc_delay
        level 7:   delay = 8 * oc_delay

    Cite: vendor/nethack/src/spell.c::study_book lines 537-559.
    """
    if book_level <= 2:
        return oc_delay
    if book_level <= 4:
        return (book_level - 1) * oc_delay
    if book_level <= 6:
        return book_level * oc_delay
    return 8 * oc_delay


def _assign_letter(magic: MagicState, spell_id: int) -> MagicState:
    """Assign the first unused a-z / A-Z letter to this spell.

    Letters 0-25 map to 'a'-'z', 26-51 map to 'A'-'Z'.
    If all 52 are taken, leave spell_letter[spell_id] at -1.
    """
    used = magic.spell_letter  # [N_SPELLS] int8, -1 = unbound
    for letter_idx in range(52):
        taken = bool(jnp.any(used == jnp.int8(letter_idx)))
        if not taken:
            new_letters = magic.spell_letter.at[spell_id].set(jnp.int8(letter_idx))
            return magic.replace(spell_letter=new_letters)
    return magic


def study_success_chance(
    player_int: int,
    player_xl: int,
    book_level: int,
    role_id: int = 0,
    buc_status: int = 2,
) -> float:
    """Return success probability in [0.0, 1.0] for studying a spellbook.

    Vendor formula (spell.c::study_book lines 582-599, uncursed path):
        read_ability = INT + 4 + xl//2 - 2 * book_level
        success iff rnd(20) <= read_ability   (rnd(20) in 1..20)
        => success_chance = clamp(read_ability, 0, 20) / 20

    Blessed (+2 to read_ability) and Wizard (+_WIZARD_STUDY_BONUS) modifiers
    are applied before clamping.  Cursed books always return 0.0 (backfire).

    Cite: vendor/nethack/src/spell.c::study_book lines 590-650.
    This function is used outside JIT (e.g. for tests / host-side validation).
    """
    # Cursed books always backfire — never learn (vendor lines 590-650).
    if buc_status == _BUC_CURSED:
        return 0.0
    ra = player_int + 4 + player_xl // 2 - 2 * book_level
    if role_id == _ROLE_WIZARD:
        ra += _WIZARD_STUDY_BONUS
    # Blessed: +2 bonus (vendor spell.c study_book ~lines 555-560).
    if buc_status == _BUC_BLESSED:
        ra += _BLESSED_STUDY_BONUS
    ra = max(0, min(ra, 20))
    return ra / 20.0


def read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Read a spellbook from inventory slot `slot_idx`.

    Looks up the spell_id stored in inventory item at slot_idx via
    ``state.inventory.items[slot_idx].type_id``.

    Returns updated state.

    BUC handling (vendor spell.c::study_book / cursed_book lines 590-650):
      CURSED  (buc_status == 1): skips success roll; five-branch backfire
        selected by rnd(20): 1-4 explode (damage+destroy), 5-8 paralyze,
        9-12 poison (str-1), 13-16 amnesia (clear all spells), 17-20 blank.
        Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
      UNCURSED (buc_status == 2): standard formula (see below).
      BLESSED  (buc_status == 3): +_BLESSED_STUDY_BONUS to read_ability.

    Vendor study check (spell.c::study_book lines 582-599, uncursed path):
        read_ability = INT + 4 + xl//2 - 2 * book_level
        [+ _WIZARD_STUDY_BONUS for Wizard role]
        [+ _BLESSED_STUDY_BONUS for blessed book]
        roll = jax.random.randint in [1..20]
        success iff roll <= read_ability

    On success:
        spell_known[spell_id]  = True
        spell_memory[spell_id] = KEEN + 1  (vendor incrnknow(i, 1);
            spell.c line 22 + lines 410/428)
        assign inventory letter if not yet assigned

    On failure:
        state unchanged
    """
    from Nethax.nethax.subsystems.status_effects import TimedStatus

    # Blind / stunned: cannot read (vendor/nethack/src/read.c::doread early checks).
    is_blind   = state.status.timed_statuses[int(TimedStatus.BLIND)]   > jnp.int32(0)
    is_stunned = state.status.timed_statuses[int(TimedStatus.STUNNED)] > jnp.int32(0)
    if bool(is_blind) or bool(is_stunned):
        return state

    # --- resolve spell_id and buc_status from inventory slot ---
    spell_id   = int(state.inventory.items.type_id[slot_idx])
    buc_status = int(state.inventory.items.buc_status[slot_idx])

    # Blank / novel book: no-op
    if spell_id == BLANK_SPELL_ID or spell_id < 0 or spell_id >= N_SPELLS:
        return state

    book_level = int(_SPELL_LEVELS[spell_id])

    # --- CURSED path (vendor spell.c::cursed_book lines 130-185) ---
    # Five branches selected by rnd(20), grouped in blocks of 4:
    #   1-4  explode: rnd(20) hp damage, book destroyed (quantity=0)
    #   5-8  paralyze: FROZEN timer rn1(5,10) = 10..14 turns
    #   9-12 poison: ATTRIBUTE_AWAY timer set, player_str decremented (min 3)
    #   13-16 amnesia: all spell_known cleared
    #   17-20 blank: no effect (turn wasted)
    # Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
    if buc_status == _BUC_CURSED:
        rng, sub_b, sub_dmg, sub_par, _sub_pois, sub_burn, sub_drop = jax.random.split(rng, 7)

        # rnd(20) in [1,20]; branch index = (b-1)//4 in [0,4]:
        #   0=explode, 1=paralyze, 2=poison, 3=amnesia, 4=blank
        # Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
        b = rnd(sub_b, 20)
        branch = (b - jnp.int32(1)) // jnp.int32(4)  # [0,4]

        is_explode  = branch == jnp.int32(0)
        is_paralyze = branch == jnp.int32(1)
        is_poison   = branch == jnp.int32(2)
        is_amnesia  = branch == jnp.int32(3)

        # --- Branch 0: explode — rnd(20) hp damage, destroy book, burn hands ---
        # Vendor spell.c::cursed_book line 176: book explodes in face.
        # Additional: 1d4 damage to hands from burning (spell.c:590-650 full table).
        # Cite: vendor/nethack/src/spell.c::cursed_book lines 590-650.
        explode_dmg = rnd(sub_dmg, 20)
        burn_dmg    = rnd(sub_burn, 4)   # additional 1d4 hand burn
        total_explode_dmg = explode_dmg + burn_dmg
        new_hp = jnp.where(
            is_explode,
            jnp.maximum(state.player_hp - total_explode_dmg, jnp.int32(1)),
            state.player_hp,
        )
        new_qty = jnp.where(
            is_explode,
            jnp.int16(0),
            state.inventory.items.quantity[slot_idx],
        ).astype(jnp.int16)
        new_inventory_qty = state.inventory.items.quantity.at[slot_idx].set(new_qty)
        new_items_stage = state.inventory.items.replace(quantity=new_inventory_qty)

        # Explode also force-drops the wielded item (burned hands can't hold weapon).
        # Cite: vendor/nethack/src/spell.c::cursed_book — item drop on explode branch.
        cur_wielded = state.inventory.wielded.astype(jnp.int32)
        new_wielded = jnp.where(is_explode, jnp.int8(-1), state.inventory.wielded)
        new_inventory = state.inventory.replace(items=new_items_stage, wielded=new_wielded)

        # --- Branch 1: paralyze — FROZEN timer rn1(5,10) = 10..14 turns ---
        # Vendor spell.c::cursed_book (paralysis path).
        par_turns = rn1(sub_par, 5, 10)
        ts = state.status.timed_statuses
        cur_frozen = ts[int(TimedStatus.FROZEN)]
        new_frozen = jnp.where(is_paralyze, jnp.maximum(cur_frozen, par_turns), cur_frozen)

        # --- Branch 2: poison — ATTRIBUTE_AWAY set, str -1 (min 3) ---
        # Vendor spell.c::cursed_book line 164: poison_strdmg.
        cur_attr = ts[int(TimedStatus.ATTRIBUTE_AWAY)]
        new_attr = jnp.where(is_poison, jnp.int32(10), cur_attr)
        new_str = jnp.where(
            is_poison,
            jnp.maximum(state.player_str - jnp.int16(1), jnp.int16(3)),
            state.player_str,
        ).astype(jnp.int16)

        # --- Branch 3: amnesia — all spell_known cleared ---
        new_known = jnp.where(
            is_amnesia,
            jnp.zeros_like(state.magic.spell_known),
            state.magic.spell_known,
        )

        # Commit timed_statuses with all branch updates applied selectively.
        new_ts = (
            ts
            .at[int(TimedStatus.FROZEN)].set(new_frozen)
            .at[int(TimedStatus.ATTRIBUTE_AWAY)].set(new_attr)
        )
        new_status = state.status.replace(timed_statuses=new_ts)
        new_magic  = state.magic.replace(spell_known=new_known)

        return state.replace(
            player_hp=new_hp,
            player_str=new_str,
            inventory=new_inventory,
            status=new_status,
            magic=new_magic,
        )

    # --- UNCURSED / BLESSED path ---
    player_int = int(state.player_int)
    player_xl  = int(state.player_xl)
    role_id    = int(state.player_role)

    # Vendor formula: read_ability = INT + 4 + xl//2 - 2*book_level
    read_ability = player_int + 4 + player_xl // 2 - 2 * book_level
    if role_id == _ROLE_WIZARD:
        read_ability += _WIZARD_STUDY_BONUS
    # Blessed bonus (vendor spell.c study_book ~lines 555-560)
    if buc_status == _BUC_BLESSED:
        read_ability += _BLESSED_STUDY_BONUS
    read_ability = max(0, min(read_ability, 20))

    # Roll 1..20; success if roll <= read_ability.
    rng, sub = jax.random.split(rng)
    roll = int(jax.random.randint(sub, (), 1, 21))

    if roll > read_ability:
        return state

    # --- update MagicState ---
    # Vendor: study_book success calls ``incrnknow(i, 1)`` which sets
    # ``sp_know = KEEN + 1`` (vendor/nethack/src/spell.c lines 410, 428;
    # macro defined line 22).  Byte-equal: KEEN + 1 = 20001.
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(KEEN + 1))
    magic = magic.replace(spell_known=new_known, spell_memory=new_mem)

    # Assign letter if not yet bound
    if int(magic.spell_letter[spell_id]) == -1:
        magic = _assign_letter(magic, spell_id)

    # Advance timestep by study delay (vendor spell.c::study_book switch
    # at lines 537-559).  Cite via study_book_delay above.
    oc_delay = int(_SPELL_OC_DELAYS[spell_id]) if 0 <= spell_id < len(_SPELL_OC_DELAYS) else 1
    delay_turns = study_book_delay(book_level, oc_delay)
    new_timestep = (state.timestep.astype(jnp.int32) + jnp.int32(delay_turns))

    return state.replace(magic=magic, timestep=new_timestep)


def handle_read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Entry point called from items read-dispatch for SPBOOK_CLASS items.

    Delegates to read_spellbook.
    """
    new_state = read_spellbook(state, rng, slot_idx)
    # Conduct: vendor/nethack/src/read.c::study_book — ILLITERATE broken on
    # reading a spellbook (insight.c ~2147, u.uconduct.literate).
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated
    return mark_violated(new_state, int(Conduct.ILLITERATE))
