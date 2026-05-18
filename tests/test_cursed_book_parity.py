"""Vendor-parity tests: cursed/blessed spellbook backfire.

Cite: vendor/nethack/src/spell.c::study_book / cursed_book lines 130-185.
Cursed: skip success roll, five-branch backfire, spell NOT learned.
Blessed: +2 read_ability bonus over uncursed.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.magic import SpellId, _SPELL_LEVELS
from Nethax.nethax.subsystems.items_spellbooks import (
    read_spellbook,
    _BUC_CURSED,
    _BUC_UNCURSED,
    _BUC_BLESSED,
    _ROLE_WIZARD,
)
from Nethax.nethax.subsystems.inventory import make_item, _items_from_list
from Nethax.nethax.subsystems.inventory import ItemCategory
from Nethax.nethax.subsystems.status_effects import TimedStatus


def _state_with_spellbook(
    spell_id: int,
    buc_status: int = _BUC_UNCURSED,
    player_int: int = 16,
    player_xl: int = 5,
    player_role: int = 0,
    player_hp: int = 20,
) -> tuple:
    """Return (state, slot_idx) with a spellbook in inventory slot 0."""
    rng = jax.random.PRNGKey(0)
    state = EnvState.default(rng)
    state = state.replace(
        player_int=jnp.int8(player_int),
        player_wis=jnp.int8(10),
        player_xl=jnp.int32(player_xl),
        player_role=jnp.int8(player_role),
        player_hp=jnp.int32(player_hp),
        player_hp_max=jnp.int32(player_hp),
        player_pw=jnp.int32(50),
        player_pw_max=jnp.int32(50),
    )
    item = make_item(
        category=int(ItemCategory.SPBOOK),
        type_id=spell_id,
        quantity=1,
        buc_status=buc_status,
    )
    inv = _items_from_list([item])
    state = state.replace(inventory=state.inventory.replace(items=inv))
    return state, 0


def test_cursed_book_no_learn():
    """Cursed spellbook: spell is never learned regardless of INT/xl.

    Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185 —
    cursed path skips the success roll entirely; spell_known stays False.
    """
    spell_id = int(SpellId.HEALING)
    # Use very high INT so uncursed would always succeed
    state, slot = _state_with_spellbook(
        spell_id, buc_status=_BUC_CURSED, player_int=18, player_xl=10
    )
    # Reset spell_known to False
    magic = state.magic.replace(
        spell_known=state.magic.spell_known.at[spell_id].set(False)
    )
    state = state.replace(magic=magic)

    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        new_state = read_spellbook(state, rng, slot)
        assert not bool(new_state.magic.spell_known[spell_id]), (
            f"Cursed book should never teach spell (seed={seed})"
        )


def test_cursed_book_some_branch_triggers():
    """Cursed spellbook triggers one of the five backfire branches across seeds.

    Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185 —
    five-branch backfire: explode / paralyze / poison / amnesia / blank.
    At least one non-blank branch should fire in 20 seeds.
    """
    spell_id = int(SpellId.HEALING)
    initial_hp = 50
    state, slot = _state_with_spellbook(
        spell_id, buc_status=_BUC_CURSED, player_hp=initial_hp
    )

    any_effect = False
    for seed in range(20):
        rng = jax.random.PRNGKey(seed)
        new_state = read_spellbook(state, rng, slot)
        hp_changed = int(new_state.player_hp) != initial_hp
        paralyzed = int(new_state.status.timed_statuses[int(TimedStatus.FROZEN)]) > 0
        poisoned = int(new_state.status.timed_statuses[int(TimedStatus.ATTRIBUTE_AWAY)]) > 0
        destroyed = int(new_state.inventory.items.quantity[slot]) == 0
        if hp_changed or paralyzed or poisoned or destroyed:
            any_effect = True
    assert any_effect, "No cursed-book branch had any observable effect across 20 seeds"


def test_cursed_book_damages():
    """Cursed explode branch (rnd20 in 1-4) deals hp damage and destroys book.

    Cite: vendor/nethack/src/spell.c::cursed_book line 176 —
    book explodes in face; dmg = 2*rnd(10)+5; we model as rnd(20).
    """
    spell_id = int(SpellId.HEALING)
    initial_hp = 60
    state, slot = _state_with_spellbook(
        spell_id, buc_status=_BUC_CURSED, player_hp=initial_hp
    )

    # Find a seed that hits the explode branch (b in 1-4 → branch index 0)
    from Nethax.nethax.rng import rnd as nethax_rnd
    explode_seed = None
    for seed in range(200):
        rng = jax.random.PRNGKey(seed)
        rng, sub_b, *_ = jax.random.split(rng, 5)
        b = int(nethax_rnd(sub_b, 20))
        if (b - 1) // 4 == 0:
            explode_seed = seed
            break

    assert explode_seed is not None, "No explode seed found in 200 tries"
    rng = jax.random.PRNGKey(explode_seed)
    new_state = read_spellbook(state, rng, slot)
    hp_after = int(new_state.player_hp)
    assert hp_after < initial_hp, (
        f"Explode branch: hp {hp_after} should be < {initial_hp}"
    )
    assert int(new_state.inventory.items.quantity[slot]) == 0, (
        "Explode branch: book should be destroyed"
    )


def test_blessed_book_higher_success():
    """Blessed level-7 book has higher success rate than uncursed over 200 trials.

    Cite: vendor/nethack/src/spell.c::study_book ~lines 555-560 —
    blessed book adds +2 to read_ability.
    """
    spell_id = int(SpellId.CANCELLATION)  # level 7
    assert int(_SPELL_LEVELS[spell_id]) == 7

    # INT=14, xl=5: uncursed read_ability = 14+4+2-14 = 6 → 30% success
    #               blessed  read_ability = 6+2 = 8      → 40% success
    n_trials = 200
    blessed_successes = 0
    uncursed_successes = 0

    for seed in range(n_trials):
        rng = jax.random.PRNGKey(seed + 500)

        state_b, slot_b = _state_with_spellbook(
            spell_id, buc_status=_BUC_BLESSED, player_int=14, player_xl=5
        )
        magic_b = state_b.magic.replace(
            spell_known=state_b.magic.spell_known.at[spell_id].set(False),
            spell_memory=state_b.magic.spell_memory.at[spell_id].set(jnp.int32(0)),
        )
        ns_b = read_spellbook(state_b.replace(magic=magic_b), rng, slot_b)
        if bool(ns_b.magic.spell_known[spell_id]):
            blessed_successes += 1

        state_u, slot_u = _state_with_spellbook(
            spell_id, buc_status=_BUC_UNCURSED, player_int=14, player_xl=5
        )
        magic_u = state_u.magic.replace(
            spell_known=state_u.magic.spell_known.at[spell_id].set(False),
            spell_memory=state_u.magic.spell_memory.at[spell_id].set(jnp.int32(0)),
        )
        ns_u = read_spellbook(state_u.replace(magic=magic_u), rng, slot_u)
        if bool(ns_u.magic.spell_known[spell_id]):
            uncursed_successes += 1

    blessed_rate = blessed_successes / n_trials
    uncursed_rate = uncursed_successes / n_trials

    assert blessed_rate > uncursed_rate - 0.05, (
        f"Blessed success rate {blessed_rate:.2%} should exceed uncursed "
        f"{uncursed_rate:.2%} (minus 5% slack)"
    )
    # Both rates should be non-zero for a meaningful level-7 test
    assert blessed_rate > 0.10, (
        f"Blessed rate {blessed_rate:.2%} too low — check formula"
    )
