"""Full vendor-parity tests for cursed spellbook five-branch backfire.

Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.

Branch selection: rnd(20) in [1,20], branch = (b-1)//4:
  0 (b=1-4):   explode  — rnd(20) damage + book destroyed (quantity=0)
  1 (b=5-8):   paralyze — FROZEN timer rn1(5,10) = 10..14 turns
  2 (b=9-12):  poison   — ATTRIBUTE_AWAY=10 turns, player_str decremented (min 3)
  3 (b=13-16): amnesia  — all spell_known cleared
  4 (b=17-20): blank    — no effect (turn wasted)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import SpellId
from Nethax.nethax.subsystems.items_spellbooks import (
    read_spellbook,
    _BUC_CURSED,
)
from Nethax.nethax.subsystems.inventory import make_item, _items_from_list, ItemCategory
from Nethax.nethax.subsystems.status_effects import TimedStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_cursed_book(
    spell_id: int = int(SpellId.HEALING),
    player_hp: int = 50,
    player_str: int = 18,
) -> tuple:
    """Return (state, slot_idx) with a cursed spellbook in slot 0."""
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    state = state.replace(
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(player_hp),
        player_str=jnp.int16(player_str),
        player_int=jnp.int8(16),
        player_wis=jnp.int8(10),
        player_xl=jnp.int32(5),
        player_role=jnp.int8(0),
        player_pw=jnp.int32(50),
        player_pw_max=jnp.int32(50),
    )
    magic = state.magic.replace(
        spell_known=jnp.zeros_like(state.magic.spell_known)
    )
    state = state.replace(magic=magic)

    item = make_item(
        category=int(ItemCategory.SPBOOK),
        type_id=spell_id,
        quantity=1,
        buc_status=_BUC_CURSED,
    )
    inv = _items_from_list([item])
    state = state.replace(inventory=state.inventory.replace(items=inv))
    return state, 0


def _find_seed_for_branch(target_branch: int, max_tries: int = 200) -> int:
    """Find a seed whose rng split matches the target cursed branch.

    Mirrors read_spellbook exactly:
        rng, sub_b, sub_dmg, sub_par, _sub_pois = split(rng, 5)
        b = rnd(sub_b, 20)  # [1,20]
        branch = (b - 1) // 4  # [0,4]
    """
    from Nethax.nethax.rng import rnd as nethax_rnd
    for seed in range(max_tries):
        rng = jax.random.PRNGKey(seed)
        rng, sub_b, _sub_dmg, _sub_par, _sub_pois = jax.random.split(rng, 5)
        b = int(nethax_rnd(sub_b, 20))
        branch = (b - 1) // 4
        if branch == target_branch:
            return seed
    raise RuntimeError(f"No seed found for branch {target_branch} in {max_tries} tries")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cursed_explode_branch():
    """Branch 0 (rnd20 in 1-4): book destroyed and hp decreases.

    Cite: vendor/nethack/src/spell.c::cursed_book line 176 — book explodes
    in face; dmg = 2*rnd(10)+5; we model as rnd(20).
    """
    spell_id = int(SpellId.HEALING)
    state, slot = _state_with_cursed_book(spell_id, player_hp=60)
    seed = _find_seed_for_branch(0)

    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    qty = int(new_state.inventory.items.quantity[slot])
    assert qty == 0, f"Explode branch: book quantity {qty} should be 0"

    hp_diff = int(state.player_hp) - int(new_state.player_hp)
    assert hp_diff > 0, f"Explode branch: no hp loss (diff={hp_diff})"


def test_cursed_paralyze_branch():
    """Branch 1 (rnd20 in 5-8): FROZEN timer > 0.

    Cite: vendor/nethack/src/spell.c::cursed_book — paralysis via rn1(5,10).
    """
    spell_id = int(SpellId.HEALING)
    state, slot = _state_with_cursed_book(spell_id)
    seed = _find_seed_for_branch(1)

    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    frozen = int(new_state.status.timed_statuses[int(TimedStatus.FROZEN)])
    assert frozen > 0, f"Paralyze branch: FROZEN timer {frozen} should be > 0"


def test_cursed_poison_branch():
    """Branch 2 (rnd20 in 9-12): ATTRIBUTE_AWAY timer set, player_str decremented.

    Cite: vendor/nethack/src/spell.c::cursed_book line 164 — poison_strdmg.
    """
    spell_id = int(SpellId.HEALING)
    initial_str = 18
    state, slot = _state_with_cursed_book(spell_id, player_str=initial_str)
    seed = _find_seed_for_branch(2)

    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    attr_away = int(new_state.status.timed_statuses[int(TimedStatus.ATTRIBUTE_AWAY)])
    assert attr_away > 0, f"Poison branch: ATTRIBUTE_AWAY timer {attr_away} should be > 0"

    new_str = int(new_state.player_str)
    assert new_str < initial_str, (
        f"Poison branch: player_str {new_str} should be < {initial_str}"
    )


def test_cursed_amnesia_branch():
    """Branch 3 (rnd20 in 13-16): all spell_known cleared.

    Cite: vendor/nethack/src/spell.c::cursed_book — forget all spells.
    """
    spell_id = int(SpellId.HEALING)
    state, slot = _state_with_cursed_book(spell_id)

    # Pre-set all spells as known so amnesia is detectable
    magic = state.magic.replace(
        spell_known=jnp.ones_like(state.magic.spell_known)
    )
    state = state.replace(magic=magic)

    seed = _find_seed_for_branch(3)
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    all_false = bool(jnp.all(~new_state.magic.spell_known))
    assert all_false, "Amnesia branch: not all spell_known were cleared"


def test_cursed_blank_branch():
    """Branch 4 (rnd20 in 17-20): no meaningful state change (turn wasted).

    Cite: vendor/nethack/src/spell.c::cursed_book — blank, no effect.
    """
    spell_id = int(SpellId.HEALING)
    initial_str = 18
    initial_hp = 50
    state, slot = _state_with_cursed_book(
        spell_id, player_hp=initial_hp, player_str=initial_str
    )
    seed = _find_seed_for_branch(4)

    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    assert int(new_state.player_hp) == initial_hp, "Blank branch: hp should not change"
    assert int(new_state.player_str) == initial_str, "Blank branch: str should not change"
    assert int(new_state.inventory.items.quantity[slot]) == 1, (
        "Blank branch: book should not be destroyed"
    )
    frozen = int(new_state.status.timed_statuses[int(TimedStatus.FROZEN)])
    assert frozen == 0, f"Blank branch: FROZEN timer {frozen} should be 0"
    attr_away = int(new_state.status.timed_statuses[int(TimedStatus.ATTRIBUTE_AWAY)])
    assert attr_away == 0, f"Blank branch: ATTRIBUTE_AWAY timer {attr_away} should be 0"


def test_cursed_book_never_learns_spell():
    """Across all branches, the target spell is never learned.

    Cite: vendor/nethack/src/spell.c::cursed_book — cursed book never teaches
    the spell; spell.c study_book cursed path skips learn entirely.
    """
    spell_id = int(SpellId.HEALING)
    state, slot = _state_with_cursed_book(spell_id)

    for seed in range(50):
        rng = jax.random.PRNGKey(seed)
        new_state = read_spellbook(state, rng, slot)
        assert not bool(new_state.magic.spell_known[spell_id]), (
            f"Cursed book should never teach spell (seed={seed})"
        )
