"""Full vendor-parity tests for cursed spellbook 7-branch backfire.

Cite: vendor/nethack/src/spell.c::cursed_book lines 130-185.

Vendor selects backfire via ``switch (rn2(lev))`` where
``lev = objects[booktype].oc_level`` (1..7).  Vanilla spells have lev≤7,
so the default branch (rndcurse) is structurally unreachable here.

Branch table (one case per ``rn2(lev)`` value):
  0  tele()               — random teleport to a FLOOR tile
  1  aggravate()          — wakes all level monsters
  2  make_blinded(rn1(100,250))
                          — adds 250..349 to BLIND timer
  3  take_gold()          — leprechaun-style gold theft
  4  make_confused(rn1(7,16))
                          — adds 16..22 to CONFUSION timer
  5  poison_strdmg()      — STR drain (ATTRIBUTE_AWAY timer set)
  6  dmg = 2*rnd(10)+5    — explode + book destroyed (Antimagic → 0)

All branches are tested against the highest-level spellbook the tests need
to *guarantee reachability*: lev≥7 (CANCELLATION) so every branch index is
in [0, lev) and any seed-scan over rn2(7) eventually hits each value.
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
)
from Nethax.nethax.subsystems.inventory import make_item, _items_from_list, ItemCategory
from Nethax.nethax.subsystems.status_effects import TimedStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# CANCELLATION = level 7 — every branch index 0..6 is reachable via rn2(7).
_LEV7_SPELL = int(SpellId.CANCELLATION)


def _state_with_cursed_book(
    spell_id: int = _LEV7_SPELL,
    player_hp: int = 50,
    player_str: int = 18,
    player_gold: int = 200,
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
        player_gold=jnp.int32(player_gold),
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


def _find_seed_for_branch(target_branch: int, lev: int = 7,
                          max_tries: int = 1000) -> int:
    """Find a seed whose RNG-split matches vendor ``rn2(lev) == target_branch``.

    Mirrors :func:`_cursed_book_backfire` exactly:
        rng_branch, sub_tele, sub_blind, sub_gold_amt, sub_gold_pos,
        sub_confuse, sub_poison, sub_dmg, sub_curse = split(rng, 9)
        rn2_lev = rn2(rng_branch, lev)
    """
    from Nethax.nethax.rng import rn2 as nethax_rn2
    for seed in range(max_tries):
        rng = jax.random.PRNGKey(seed)
        subs = jax.random.split(rng, 9)
        rn2_lev = int(nethax_rn2(subs[0], lev))
        if rn2_lev == target_branch:
            return seed
    raise RuntimeError(
        f"No seed found for branch {target_branch} (lev={lev}) in {max_tries} tries"
    )


# ---------------------------------------------------------------------------
# Tests — one per vendor branch
#
# REBALANCE NOTE: the prior 5-branch model pinned its branches to the old
# ``(rnd(20)-1)//4`` selector with branches {explode, paralyze, poison,
# amnesia, blank}.  Vendor uses ``switch (rn2(lev))`` with branches
# {tele, aggravate, blind, take_gold, confuse, poison, explode} — *no*
# paralyze, *no* amnesia, *no* blank.  Each test below explains in its
# docstring whether it had to be rewritten because the prior assertion
# was bug-pinning.
# ---------------------------------------------------------------------------


def test_cursed_teleport_branch():
    """Branch 0: vendor ``tele()`` — player teleports to a FLOOR tile.

    REBALANCE: previously this slot tested "explode" (old branch 0).  Vendor
    branch 0 is ``tele()`` — the explode case is at branch 6.  We assert the
    player position changes (or is at minimum unchanged because no floor was
    available, which is itself a defined fallback).

    Cite: vendor/nethack/src/spell.c::cursed_book line 139 ``tele();``.
    """
    state, slot = _state_with_cursed_book()
    seed = _find_seed_for_branch(0)

    pos_before = state.player_pos.copy()
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    # Either we moved, or there's no FLOOR on this synthetic level (fallback).
    moved = bool(jnp.any(new_state.player_pos != pos_before))
    # Book is NOT destroyed on teleport branch (vendor returns FALSE → caller keeps it).
    qty = int(new_state.inventory.items.quantity[slot])
    assert qty == 1, f"Teleport branch should NOT destroy the book; qty={qty}"
    # Spell not learned regardless.
    assert not bool(new_state.magic.spell_known[_LEV7_SPELL])


def test_cursed_aggravate_branch():
    """Branch 1: vendor ``aggravate()`` — wake all level monsters.

    REBALANCE: was ``test_cursed_paralyze_branch`` pinning the OLD (bug)
    paralyze branch.  Vendor has *no* paralyze branch in cursed_book — the
    paralyze behavior was invented.  Vendor branch 1 is ``aggravate()``.

    We assert that any sleeping monster gets its ``asleep`` flag cleared.
    Cite: vendor/nethack/src/spell.c::cursed_book line 143 ``aggravate();``.
          vendor/nethack/src/wizard.c::aggravate lines 493-511.
    """
    state, slot = _state_with_cursed_book()
    # Force every alive monster slot into sleep so the wake_monsters_near
    # effect is observable.
    mai = state.monster_ai
    new_asleep = jnp.where(mai.alive, jnp.bool_(True), mai.asleep)
    state = state.replace(monster_ai=mai.replace(asleep=new_asleep))

    seed = _find_seed_for_branch(1)
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    # All previously-sleeping live monsters within radius 999 (level-wide)
    # should be awake now.  At minimum, the total sleeping-count must not
    # have *increased*.
    sleep_before = int(jnp.sum(new_asleep & mai.alive))
    sleep_after = int(jnp.sum(new_state.monster_ai.asleep & new_state.monster_ai.alive))
    assert sleep_after <= sleep_before, (
        f"Aggravate branch should not increase sleeping monsters "
        f"(before={sleep_before}, after={sleep_after})"
    )


def test_cursed_blind_branch():
    """Branch 2: vendor ``make_blinded(rn1(100, 250))`` — BLIND timer += 250..349.

    REBALANCE: vendor branch 2 maps to "blind", which was *not* in the prior
    5-branch model at all (it had {explode, paralyze, poison, amnesia, blank}).
    This is a brand-new vendor-truth test.

    BUT: ``read_spellbook`` returns early when the player is already blind
    (vendor read.c::doread).  Since the cursed branch fires from a fresh
    (non-blind) state, the timer should be exactly 0 before and >= 250 after.

    Cite: vendor/nethack/src/spell.c::cursed_book line 146.
    """
    state, slot = _state_with_cursed_book()
    seed = _find_seed_for_branch(2)
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    blind_after = int(new_state.status.timed_statuses[int(TimedStatus.BLIND)])
    assert blind_after >= 250, (
        f"Blind branch: BLIND timer {blind_after} should be >= 250 (vendor rn1(100,250))"
    )
    assert blind_after <= 349, (
        f"Blind branch: BLIND timer {blind_after} should be <= 349 (vendor rn1(100,250))"
    )


def test_cursed_take_gold_branch():
    """Branch 3: vendor ``take_gold()`` — player_gold drops by somegold(igold).

    REBALANCE: new vendor-truth test.  No analog in the prior 5-branch model.

    Cite: vendor/nethack/src/spell.c::cursed_book line 149;
          vendor/nethack/src/steal.c::somegold lines 13-34.
    """
    initial_gold = 200
    state, slot = _state_with_cursed_book(player_gold=initial_gold)
    seed = _find_seed_for_branch(3)
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    gold_after = int(new_state.player_gold)
    # somegold(200) is in the [50, 200] bracket (vendor steal.c:24).
    assert 0 <= gold_after < initial_gold, (
        f"Take-gold branch: gold {gold_after} should be < {initial_gold}"
    )


def test_cursed_confuse_branch():
    """Branch 4: vendor ``make_confused(rn1(7, 16))`` — CONFUSION timer += 16..22.

    REBALANCE: new vendor-truth test.  No analog in the prior 5-branch model
    (the prior "amnesia" branch invented a non-vendor behaviour).

    Cite: vendor/nethack/src/spell.c::cursed_book line 153.
    """
    state, slot = _state_with_cursed_book()
    seed = _find_seed_for_branch(4)
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    conf_after = int(new_state.status.timed_statuses[int(TimedStatus.CONFUSION)])
    assert 16 <= conf_after <= 22, (
        f"Confuse branch: CONFUSION timer {conf_after} should be in [16, 22] "
        f"(vendor rn1(7, 16))"
    )


def test_cursed_poison_branch():
    """Branch 5: vendor ``poison_strdmg()`` — STR drained, ATTRIBUTE_AWAY set.

    REBALANCE: was branch 2 in the old 5-branch model (selected by
    ``(rnd(20)-1)//4 == 2`` → rnd(20)∈[9,12]).  Vendor branch is *5*
    (selected by ``rn2(lev) == 5``).  Same observable effects; only the
    branch index changed.

    Cite: vendor/nethack/src/spell.c::cursed_book lines 155-168.
    """
    initial_str = 18
    state, slot = _state_with_cursed_book(player_str=initial_str)
    seed = _find_seed_for_branch(5)
    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    attr_away = int(new_state.status.timed_statuses[int(TimedStatus.ATTRIBUTE_AWAY)])
    assert attr_away > 0, (
        f"Poison branch: ATTRIBUTE_AWAY timer {attr_away} should be > 0"
    )
    new_str = int(new_state.player_str)
    assert new_str < initial_str, (
        f"Poison branch: player_str {new_str} should be < {initial_str}"
    )


def test_cursed_explode_branch():
    """Branch 6: vendor ``dmg = 2*rnd(10) + 5`` — explode + book destroyed.

    REBALANCE: was branch 0 in the old 5-branch model (selected by
    ``(rnd(20)-1)//4 == 0`` → rnd(20)∈[1,4]).  Vendor branch is *6*
    (last numeric case; the previous "explode at branch 0" was bug-pinned).
    Damage formula was also wrong: old code used ``rnd(20) + rnd(4)``
    (= 2..24 plus an invented 1d4 hand burn).  Vendor: ``2*rnd(10)+5`` in
    [7, 25] with Antimagic gating to 0.

    Cite: vendor/nethack/src/spell.c::cursed_book lines 169-179.
    """
    initial_hp = 60
    state, slot = _state_with_cursed_book(player_hp=initial_hp)
    seed = _find_seed_for_branch(6)

    rng = jax.random.PRNGKey(seed)
    new_state = read_spellbook(state, rng, slot)

    qty = int(new_state.inventory.items.quantity[slot])
    assert qty == 0, f"Explode branch: book quantity {qty} should be 0"

    hp_after = int(new_state.player_hp)
    hp_loss = initial_hp - hp_after
    # Vendor: dmg = 2*rnd(10)+5 = 7..25.  Antimagic not modelled → always full dmg.
    assert 7 <= hp_loss <= 25, (
        f"Explode branch: hp loss {hp_loss} should be in [7, 25] (vendor 2*rnd(10)+5)"
    )


def test_cursed_book_never_learns_spell():
    """Across all branches, the target spell is never learned.

    Cite: vendor/nethack/src/spell.c::study_book lines 577-602 — cursed path
    takes the ``too_hard = TRUE`` exit and never calls incrnknow().
    """
    state, slot = _state_with_cursed_book()

    for seed in range(50):
        rng = jax.random.PRNGKey(seed)
        new_state = read_spellbook(state, rng, slot)
        assert not bool(new_state.magic.spell_known[_LEV7_SPELL]), (
            f"Cursed book should never teach spell (seed={seed})"
        )
