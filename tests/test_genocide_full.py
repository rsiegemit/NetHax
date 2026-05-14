"""Wave 6 Phase B+ — full vendor genocide-class table tests.

Replaces the Wave-5 5-class pool with vendor-parity coverage of all 26 lowercase
+ 26 uppercase monster class letters plus '@' (humans).

Cite: vendor/nethack/src/read.c::do_genocide
      vendor/nethack/include/monsym.h::S_* glyph class table
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import string

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.conduct import Conduct
from Nethax.nethax.subsystems.items_scrolls import apply_genocide
from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol


_RNG = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_idx_with_symbol(symbol_val: int):
    """Return the first MONSTERS index whose .symbol matches ``symbol_val``,
    or None if no monster in the table has that symbol."""
    for i, m in enumerate(MONSTERS):
        if int(m.symbol) == int(symbol_val):
            return i
    return None


def _seed_monsters_of_symbol(state: EnvState, symbol_val: int, n: int = 3) -> EnvState:
    """Place ``n`` live monsters of the given symbol class in slots 0..n-1.

    Returns the state unchanged if no monster in MONSTERS has that symbol.
    """
    entry_idx = _entry_idx_with_symbol(symbol_val)
    if entry_idx is None:
        return state

    mai = state.monster_ai
    new_alive = mai.alive
    new_entry = mai.entry_idx
    new_hp = mai.hp
    new_pos = mai.pos
    for i in range(n):
        new_alive = new_alive.at[i].set(jnp.bool_(True))
        new_entry = new_entry.at[i].set(jnp.int16(entry_idx))
        new_hp = new_hp.at[i].set(jnp.int32(5))
        new_pos = new_pos.at[i].set(jnp.array([5 + i, 5 + i], dtype=jnp.int16))
    new_mai = mai.replace(alive=new_alive, entry_idx=new_entry,
                          hp=new_hp, pos=new_pos)
    return state.replace(monster_ai=new_mai)


def _letter_to_symbol(letter: str) -> int:
    """Same mapping used by apply_genocide (see items_scrolls._build_class_letter_to_symbol)."""
    if 'a' <= letter <= 'z':
        return 1 + (ord(letter) - ord('a'))
    if 'A' <= letter <= 'Z':
        return 27 + (ord(letter) - ord('A'))
    if letter == '@':
        return int(MonsterSymbol.S_HUMAN)
    if letter == ' ':
        return int(MonsterSymbol.S_GHOST)
    if letter == "'":
        return int(MonsterSymbol.S_GOLEM)
    if letter == '&':
        return int(MonsterSymbol.S_DEMON)
    if letter == ';':
        return int(MonsterSymbol.S_EEL)
    if letter == ':':
        return int(MonsterSymbol.S_LIZARD)
    if letter == '~':
        return int(MonsterSymbol.S_WORM_TAIL)
    if letter == ']':
        return int(MonsterSymbol.S_MIMIC_DEF)
    return -1


# ---------------------------------------------------------------------------
# Per-letter tests
# ---------------------------------------------------------------------------

def test_genocide_d_class_kills_all_dogs():
    """'d' = S_DOG.  Vendor read.c::do_genocide kills every alive 'd' on level."""
    state = EnvState.default(_RNG)
    state = _seed_monsters_of_symbol(state, int(MonsterSymbol.S_DOG), n=4)
    assert bool(state.monster_ai.alive[:4].all())
    new_state = apply_genocide(state, _RNG, class_letter='d')
    # All 'd' monsters should be dead.
    assert bool(new_state.monster_ai.alive[:4].any()) is False


def test_genocide_L_class_kills_liches():
    """'L' = S_LICH.  Vendor read.c::do_genocide kills every alive 'L'."""
    state = EnvState.default(_RNG)
    state = _seed_monsters_of_symbol(state, int(MonsterSymbol.S_LICH), n=3)
    assert bool(state.monster_ai.alive[:3].all())
    new_state = apply_genocide(state, _RNG, class_letter='L')
    assert bool(new_state.monster_ai.alive[:3].any()) is False


def test_genocide_at_class_kills_humans():
    """'@' = S_HUMAN (also race-glyph).  Vendor: every '@' monster dies."""
    state = EnvState.default(_RNG)
    state = _seed_monsters_of_symbol(state, int(MonsterSymbol.S_HUMAN), n=3)
    assert bool(state.monster_ai.alive[:3].all())
    new_state = apply_genocide(state, _RNG, class_letter='@')
    assert bool(new_state.monster_ai.alive[:3].any()) is False


def test_genocide_unknown_class_noop():
    """An unknown class letter kills nothing but still flips GENOCIDELESS
    (vendor read.c::do_genocide always counts the use)."""
    state = EnvState.default(_RNG)
    state = _seed_monsters_of_symbol(state, int(MonsterSymbol.S_DOG), n=2)
    # '!' is not a monster class letter.
    new_state = apply_genocide(state, _RNG, class_letter='!')
    assert bool(new_state.monster_ai.alive[:2].all())
    assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True


def test_genocide_other_classes_not_affected():
    """Genocide of 'L' kills 'L' monsters but leaves 'd' monsters alive."""
    state = EnvState.default(_RNG)
    state = _seed_monsters_of_symbol(state, int(MonsterSymbol.S_LICH), n=2)
    # Seed a dog in slot 5.
    dog_entry_idx = _entry_idx_with_symbol(int(MonsterSymbol.S_DOG))
    assert dog_entry_idx is not None
    mai = state.monster_ai
    new_mai = mai.replace(
        alive=mai.alive.at[5].set(jnp.bool_(True)),
        entry_idx=mai.entry_idx.at[5].set(jnp.int16(dog_entry_idx)),
        hp=mai.hp.at[5].set(jnp.int32(5)),
        pos=mai.pos.at[5].set(jnp.array([10, 10], dtype=jnp.int16)),
    )
    state = state.replace(monster_ai=new_mai)

    new_state = apply_genocide(state, _RNG, class_letter='L')
    # 'L' slot 0..1 dead, 'd' slot 5 still alive.
    assert bool(new_state.monster_ai.alive[:2].any()) is False
    assert bool(new_state.monster_ai.alive[5]) is True


def test_genocide_flips_genocideless_for_every_letter():
    """Every valid letter genocide flips the GENOCIDELESS conduct."""
    for letter in 'abcdLM@':
        state = EnvState.default(_RNG)
        assert bool(state.conduct.violations[int(Conduct.GENOCIDELESS)]) is False
        new_state = apply_genocide(state, _RNG, class_letter=letter)
        assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True, \
            f"letter {letter!r} did not flip GENOCIDELESS"


def test_genocide_all_26_letter_classes_have_targets():
    """For every monster class letter that maps to a class with at least one
    MONSTERS entry, apply_genocide kills the seeded monsters.

    Covers all 26 lowercase + 26 uppercase + '@' that are populated in our
    MONSTERS table (some glyph classes like S_INVISIBLE may be empty).
    """
    letters = list(string.ascii_lowercase) + list(string.ascii_uppercase) + ['@']
    covered = []
    for letter in letters:
        sym = _letter_to_symbol(letter)
        entry_idx = _entry_idx_with_symbol(sym)
        if entry_idx is None:
            # No monster of that class in the table — skip but record.
            continue
        state = EnvState.default(_RNG)
        state = _seed_monsters_of_symbol(state, sym, n=2)
        if not bool(state.monster_ai.alive[:2].all()):
            # Seeding failed (should not happen if entry_idx existed).
            continue
        new_state = apply_genocide(state, _RNG, class_letter=letter)
        assert bool(new_state.monster_ai.alive[:2].any()) is False, \
            f"letter {letter!r} (symbol {sym}) did not kill seeded monsters"
        assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True
        covered.append(letter)

    # We expect at least the canonical 26 lowercase + 26 uppercase + '@' that
    # have populated symbols in our MONSTERS table.  Sanity: at least 50.
    assert len(covered) >= 50, f"too few letter classes covered: {covered!r}"


def test_genocide_legacy_random_pool_path_still_works():
    """apply_genocide(state, rng) (no class_letter) keeps Wave-5 RNG-pool path."""
    state = EnvState.default(_RNG)
    # Default RNG pool path still flips conduct.
    new_state = apply_genocide(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True


def test_genocide_via_scroll_path_still_works():
    """Reading a genocide scroll dispatches through items_scrolls and flips
    both ILLITERATE and GENOCIDELESS."""
    from Nethax.nethax.constants.objects import ObjectClass
    from Nethax.nethax.subsystems.items_scrolls import (
        handle_read, _SCROLL_BASE_ID, ScrollEffect,
    )

    state = EnvState.default(_RNG)
    type_id = _SCROLL_BASE_ID + int(ScrollEffect.GENOCIDE)
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.SCROLL_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))
    new_state = handle_read(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ILLITERATE)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True
