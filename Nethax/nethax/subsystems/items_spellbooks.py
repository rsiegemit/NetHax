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

  On success: spell_known[spell_id] = True, spell_memory = MAX_SPELL_MEMORY
  On failure: no change (side effects like confusion/paralysis are Wave 4+)

Wave 8d simplifications:
  - No confusion penalty, no sleep-inducing check, no multi-turn delay
  - Blessed/cursed book modifiers not modeled (Wave 4+)
  - Blank-book and novel detection: treated as unknown spell_id (-1) → no-op
"""

import jax
import jax.numpy as jnp

from Nethax.nethax.subsystems.magic import (
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
) -> float:
    """Return success probability in [0.0, 1.0] for studying a spellbook.

    Vendor formula (spell.c::study_book lines 582-599, uncursed path):
        read_ability = INT + 4 + xl//2 - 2 * book_level
        success iff rnd(20) <= read_ability   (rnd(20) in 1..20)
        => success_chance = clamp(read_ability, 0, 20) / 20

    Wizard (+_WIZARD_STUDY_BONUS) models the role's higher comprehension.
    This function is used outside JIT (e.g. for tests / host-side validation).
    """
    ra = player_int + 4 + player_xl // 2 - 2 * book_level
    if role_id == _ROLE_WIZARD:
        ra += _WIZARD_STUDY_BONUS
    ra = max(0, min(ra, 20))
    return ra / 20.0


def read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Read a spellbook from inventory slot `slot_idx`.

    Looks up the spell_id stored in inventory item at slot_idx via
    ``state.inventory.items[slot_idx].type_id``.

    Returns updated state.

    Vendor study check (spell.c::study_book lines 582-599, uncursed path):
        read_ability = INT + 4 + xl//2 - 2 * book_level
        [+ _WIZARD_STUDY_BONUS for Wizard role]
        roll = jax.random.randint in [1..20]
        success iff roll <= read_ability

    On success:
        spell_known[spell_id]  = True
        spell_memory[spell_id] = MAX_SPELL_MEMORY
        assign inventory letter if not yet assigned

    On failure:
        state unchanged (side effects are Wave 4+)
    """
    # --- resolve spell_id from inventory slot ---
    spell_id = int(state.inventory.items.type_id[slot_idx])

    # Blank / novel book: no-op
    if spell_id == BLANK_SPELL_ID or spell_id < 0 or spell_id >= N_SPELLS:
        return state

    book_level = int(_SPELL_LEVELS[spell_id])

    # Vendor formula: read_ability = INT + 4 + xl//2 - 2*book_level
    player_int = int(state.player_int)
    player_xl  = int(state.player_xl)
    role_id    = int(state.player_role)

    read_ability = player_int + 4 + player_xl // 2 - 2 * book_level
    if role_id == _ROLE_WIZARD:
        read_ability += _WIZARD_STUDY_BONUS
    read_ability = max(0, min(read_ability, 20))

    # Roll 1..20; success if roll <= read_ability.
    rng, sub = jax.random.split(rng)
    roll = int(jax.random.randint(sub, (), 1, 21))

    if roll > read_ability:
        return state

    # --- update MagicState ---
    magic = state.magic
    new_known = magic.spell_known.at[spell_id].set(True)
    new_mem   = magic.spell_memory.at[spell_id].set(jnp.int32(MAX_SPELL_MEMORY))
    magic = magic.replace(spell_known=new_known, spell_memory=new_mem)

    # Assign letter if not yet bound
    if int(magic.spell_letter[spell_id]) == -1:
        magic = _assign_letter(magic, spell_id)

    return state.replace(magic=magic)


def handle_read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Entry point called from items read-dispatch for SPBOOK_CLASS items.

    Delegates to read_spellbook.
    """
    new_state = read_spellbook(state, rng, slot_idx)
    # Conduct: vendor/nethack/src/read.c::study_book — ILLITERATE broken on
    # reading a spellbook (insight.c ~2147, u.uconduct.literate).
    from Nethax.nethax.subsystems.conduct import Conduct, mark_violated
    return mark_violated(new_state, int(Conduct.ILLITERATE))
