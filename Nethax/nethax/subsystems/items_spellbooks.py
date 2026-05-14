"""Spellbook reading subsystem — study_book logic.

Canonical sources:
  vendor/nethack/src/spell.c::study_book  — learning chance, memory init
  vendor/nethack/include/spell.h          — KEEN = 20000, SPELL_LEV_PW
  vendor/nethack/include/objects.h        — SPELL() table (level, delay)

Wave 3 implementation:
  - d20 study check: success if d20 + INT_bonus + book_level_modifier >= 10
  - On success: spell_known[spell_id] = True, spell_memory = MAX_SPELL_MEMORY,
    assign first free letter
  - On failure (roll < 5): blank the book (spell_memory stays 0)
  - Wired from handle_read via class dispatch

Wave 3 simplifications:
  - No confusion penalty
  - No sleep-inducing dull-book check
  - No multi-turn reading delay
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


# Minimum d20 roll (before bonuses) that succeeds study.
# Derived from study_book: difficulty scales with book level.
# We use threshold 10 to match a mid-difficulty check.
_STUDY_THRESHOLD = 10

# Blank-book sentinel: slot_spell_id == -1 means no spell (blank or novel).
BLANK_SPELL_ID = -1


def _int_bonus(player_int: jnp.ndarray) -> jnp.ndarray:
    """INT bonus for studying (≈ +(INT-10)//2, clamped to [-5, +5])."""
    raw = (player_int.astype(jnp.int32) - 10) // 2
    return jnp.clip(raw, -5, 5)


def _book_level_modifier(spell_id: int) -> int:
    """Level-based modifier: lower-level spells are easier to learn.

    Modifier = (4 - spell_level), so level-1 books give +3, level-7 give -3.
    """
    lv = int(_SPELL_LEVELS[spell_id])
    return 4 - lv


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


def read_spellbook(state, rng: jax.Array, slot_idx: int):
    """Read a spellbook from inventory slot `slot_idx`.

    Looks up the spell_id stored in inventory item at slot_idx via
    ``state.inventory.items[slot_idx].type_id``.

    Returns updated state.

    Study check (simplified from spell.c:study_book):
        roll = d20 + INT_bonus + book_level_modifier
        success  if roll >= _STUDY_THRESHOLD
        blank    if roll < 5  (critical failure)

    On success:
        spell_known[spell_id]  = True
        spell_memory[spell_id] = MAX_SPELL_MEMORY
        assign inventory letter if not yet assigned

    On blank (roll < 5):
        spell_memory stays 0; spell_known stays unchanged
        (blanking the book item is a Wave 4 concern)
    """
    # --- resolve spell_id from inventory slot ---
    # Item fields are arrays of shape [MAX_INVENTORY_SLOTS]; index each field.
    spell_id = int(state.inventory.items.type_id[slot_idx])

    # Blank / novel book: no-op
    if spell_id == BLANK_SPELL_ID or spell_id < 0 or spell_id >= N_SPELLS:
        return state

    # Roll d20
    rng, sub = jax.random.split(rng)
    d20 = jax.random.randint(sub, (), 1, 21).astype(jnp.int32)

    total = d20 + _int_bonus(state.player_int) + _book_level_modifier(spell_id)

    success = bool(total >= _STUDY_THRESHOLD)

    if not success:
        # Failure (including blank total < 5): no spell gained
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
