"""Vendor-parity tests: cursed/blessed spellbook backfire.

Cite: vendor/nethack/src/spell.c::study_book lines 577-602 +
      vendor/nethack/src/spell.c::cursed_book lines 130-185.

  Cursed:   skip success roll; switch (rn2(oc_level)) over 7 vendor cases
            (tele / aggravate / blind / take_gold / confuse / poison / explode),
            spell NOT learned.
  Blessed:  vendor line 577 short-circuits the failure roll entirely
            (``if (!blessed && otyp != SPE_BOOK_OF_THE_DEAD)``).  Blessed books
            always succeed — there is no separate "+N" read_ability bonus.

REBALANCE NOTE: the pre-Wave-15 cursed model used a 5-branch
``(rnd(20)-1)//4`` selector with invented branches {explode, paralyze,
poison, amnesia, blank}.  Vendor has *seven* branches selected by
``rn2(lev)``: {tele, aggravate, blind, take_gold, confuse, poison, explode}
plus an unreachable rndcurse default.  Tests that depended on the old
branch list have been rewritten to assert vendor behaviour, with each
docstring noting the prior bug-pin.
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
    player_gold: int = 100,
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
        player_gold=jnp.int32(player_gold),
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
    """Cursed spellbook triggers some observable side-effect across seeds.

    REBALANCE: previously this test allowed {hp_changed, paralyzed, poisoned,
    destroyed} as proxies — those were the 5-branch model's observables.
    Vendor's 7-branch model has different observables, so we widen to cover
    the vendor branches that are reachable for a level-7 book (CANCELLATION):

      branch 0 (tele):      player_pos changes
      branch 1 (aggravate): monster_ai.asleep set decreases  (skipped — no
                                                              sleeping mons here)
      branch 2 (blind):     BLIND timer goes positive
      branch 3 (take_gold): player_gold drops
      branch 4 (confuse):   CONFUSION timer goes positive
      branch 5 (poison):    ATTRIBUTE_AWAY timer goes positive
      branch 6 (explode):   hp drops AND book destroyed

    Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.
    """
    # CANCELLATION = level 7 → rn2(7) ∈ [0,6] so every branch is hit.
    spell_id = int(SpellId.CANCELLATION)
    initial_hp = 60
    initial_gold = 200
    initial_pos_state, slot = _state_with_spellbook(
        spell_id, buc_status=_BUC_CURSED,
        player_hp=initial_hp, player_gold=initial_gold,
    )
    pos_before = tuple(int(x) for x in initial_pos_state.player_pos)

    any_effect = False
    for seed in range(30):
        rng = jax.random.PRNGKey(seed)
        new_state = read_spellbook(initial_pos_state, rng, slot)

        hp_changed   = int(new_state.player_hp) != initial_hp
        blind_set    = int(new_state.status.timed_statuses[int(TimedStatus.BLIND)]) > 0
        conf_set     = int(new_state.status.timed_statuses[int(TimedStatus.CONFUSION)]) > 0
        attr_set     = int(new_state.status.timed_statuses[int(TimedStatus.ATTRIBUTE_AWAY)]) > 0
        gold_dropped = int(new_state.player_gold) < initial_gold
        destroyed    = int(new_state.inventory.items.quantity[slot]) == 0
        pos_after    = tuple(int(x) for x in new_state.player_pos)
        pos_changed  = pos_after != pos_before

        if (hp_changed or blind_set or conf_set or attr_set
                or gold_dropped or destroyed or pos_changed):
            any_effect = True
            break
    assert any_effect, "No cursed-book vendor branch had any observable effect across 30 seeds"


def test_cursed_book_damages():
    """Cursed explode branch (vendor case 6) deals hp damage and destroys book.

    REBALANCE: explode used to be old-branch 0 (selected by rnd(20)∈[1,4]).
    Vendor explode is case 6 (selected by rn2(lev) == 6, needing lev≥7).
    Also: vendor damage is ``2*rnd(10)+5`` (= 7..25) without the old invented
    ``+rnd(4)`` hand-burn.  Test now uses a level-7 spell and asserts the
    vendor damage range.

    Cite: vendor/nethack/src/spell.c::cursed_book lines 169-179.
    """
    spell_id = int(SpellId.CANCELLATION)  # level 7
    assert int(_SPELL_LEVELS[spell_id]) == 7
    initial_hp = 60
    state, slot = _state_with_spellbook(
        spell_id, buc_status=_BUC_CURSED, player_hp=initial_hp
    )

    # Find a seed that hits the explode branch (rn2(7) == 6).
    from Nethax.nethax.rng import rn2 as nethax_rn2
    explode_seed = None
    for seed in range(1000):
        rng = jax.random.PRNGKey(seed)
        subs = jax.random.split(rng, 9)
        if int(nethax_rn2(subs[0], 7)) == 6:
            explode_seed = seed
            break

    assert explode_seed is not None, "No explode seed (rn2(7)==6) found in 1000 tries"
    rng = jax.random.PRNGKey(explode_seed)
    new_state = read_spellbook(state, rng, slot)
    hp_after = int(new_state.player_hp)
    hp_loss = initial_hp - hp_after
    assert 7 <= hp_loss <= 25, (
        f"Explode branch: hp loss {hp_loss} should be in [7, 25] (vendor 2*rnd(10)+5)"
    )
    assert int(new_state.inventory.items.quantity[slot]) == 0, (
        "Explode branch: book should be destroyed"
    )


def test_blessed_book_succeeds_more_than_uncursed():
    """Blessed level-7 book always succeeds while uncursed sometimes fails.

    REBALANCE: previously this test ran ``blessed_rate > uncursed_rate - 0.05``
    using the OLD "blessed = +2 read_ability" model.  Vendor blessed skips
    the failure roll entirely (spell.c line 577), so the blessed rate is
    ALWAYS 100% — much tighter than the old ±5% slack.

    Cite: vendor/nethack/src/spell.c::study_book line 577 ``if (!blessed && ...)``.
    """
    spell_id = int(SpellId.CANCELLATION)  # level 7
    assert int(_SPELL_LEVELS[spell_id]) == 7

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

    # Blessed should be exactly 100% (no failure roll).
    assert blessed_rate == 1.0, (
        f"Blessed level-7 book success rate {blessed_rate:.2%} should be 100% "
        f"(vendor spell.c:577 short-circuits the failure roll)."
    )
    # Uncursed at INT=14, xl=5, lev=7: read_ability = 14+4+2-14 = 6 → ~30%.
    assert uncursed_rate < blessed_rate, (
        f"Uncursed rate {uncursed_rate:.2%} should be < blessed {blessed_rate:.2%}"
    )
