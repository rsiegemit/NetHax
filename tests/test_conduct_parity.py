"""Conduct parity tests — verify each trigger site wires the correct conduct.

Canonical sources:
  vendor/nethack/src/insight.c::show_conduct  (~2126-2225) — per-conduct display
  vendor/nethack/src/conduct.c                — u.uconduct counter definitions

Covers:
  test_pacifist_violated_on_kill        — combat.melee_attack kill path
  test_foodless_violated_on_eat         — action_dispatch._handle_eat
  test_atheist_violated_on_pray         — prayer.handle_pray
  test_illiterate_violated_on_read      — items_scrolls.handle_read
  test_wish_free_violated_on_wand       — wish.grant_wish
  test_polyselfless_violated_on_poly    — polymorph.polymorph_player (potion path)
  test_kept_conducts_award_bonus        — scoring.compute_final_score (no violations)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.conduct import Conduct, N_CONDUCTS, mark_violated
from Nethax.nethax.constants.objects import OBJECTS, ObjectClass


_RNG = jax.random.PRNGKey(42)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# test_pacifist_violated_on_kill
# Cite: vendor/nethack/src/insight.c:2144  u.uconduct.killer
#       vendor/nethack/src/uhitm.c (xkilled path)
# ---------------------------------------------------------------------------

def test_pacifist_violated_on_kill():
    """Killing a monster via melee_attack flips PACIFIST.

    Mirror of insight.c:2144 u.uconduct.killer check.
    We set monster HP to 1 and AC extremely low so a hit is guaranteed.
    """
    from Nethax.nethax.subsystems.combat import melee_attack

    state = _fresh_state()
    mai = state.monster_ai
    state = state.replace(monster_ai=mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(1)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(1)),
        ac=mai.ac.at[0].set(jnp.int32(50)),
        is_large=mai.is_large.at[0].set(jnp.bool_(False)),
    ))

    assert not bool(state.conduct.violations[int(Conduct.PACIFIST)])

    killed_any = False
    for seed in range(10):
        ns, _dmg, _hit = melee_attack(state, jax.random.PRNGKey(seed), jnp.int32(0))
        if bool(ns.conduct.violations[int(Conduct.PACIFIST)]):
            killed_any = True
            break
    assert killed_any, "PACIFIST should be violated on at least one kill across 10 seeds"


# ---------------------------------------------------------------------------
# test_foodless_violated_on_eat
# Cite: vendor/nethack/src/insight.c:2126  u.uconduct.food
#       vendor/nethack/src/eat.c::eatfood
# ---------------------------------------------------------------------------

def _food_type_id(name: str) -> int:
    for i, o in enumerate(OBJECTS):
        if o.name == name and int(o.class_) == int(ObjectClass.FOOD_CLASS):
            return i
    raise AssertionError(f"food {name!r} not found in OBJECTS")


def _state_with_food(type_id: int) -> EnvState:
    state = _fresh_state()
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ObjectClass.FOOD_CLASS))),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(0)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


def test_foodless_violated_on_eat():
    """Eating any food item flips FOODLESS.

    Cite: vendor/nethack/src/insight.c:2126 u.uconduct.food.
    """
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat

    state = _state_with_food(_food_type_id("food ration"))
    assert not bool(state.conduct.violations[int(Conduct.FOODLESS)])
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.FOODLESS)])


# ---------------------------------------------------------------------------
# test_atheist_violated_on_pray
# Cite: vendor/nethack/src/insight.c:2134  u.uconduct.gnostic
#       vendor/nethack/src/pray.c::pray
# ---------------------------------------------------------------------------

def test_atheist_violated_on_pray():
    """handle_pray flips ATHEIST.

    Cite: vendor/nethack/src/insight.c:2134 u.uconduct.gnostic.
    """
    from Nethax.nethax.subsystems.prayer import handle_pray

    state = _fresh_state()
    assert not bool(state.conduct.violations[int(Conduct.ATHEIST)])
    new_state = handle_pray(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ATHEIST)])


# ---------------------------------------------------------------------------
# test_illiterate_violated_on_read
# Cite: vendor/nethack/src/insight.c:2147  u.uconduct.literate
#       vendor/nethack/src/read.c::doread
# ---------------------------------------------------------------------------

def test_illiterate_violated_on_read():
    """Reading a scroll flips ILLITERATE.

    Cite: vendor/nethack/src/insight.c:2147 u.uconduct.literate.
    """
    from Nethax.nethax.subsystems.items_scrolls import handle_read

    state = _fresh_state()
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(int(ObjectClass.SCROLL_CLASS))),
        type_id=items.type_id.at[0].set(jnp.int16(1)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))

    assert not bool(state.conduct.violations[int(Conduct.ILLITERATE)])
    new_state = handle_read(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ILLITERATE)])


# ---------------------------------------------------------------------------
# test_wish_free_violated_on_wand_of_wishing
# Cite: vendor/nethack/src/insight.c:2183  u.uconduct.wishes
#       vendor/nethack/src/wizard.c::makewish
# ---------------------------------------------------------------------------

def test_wish_free_violated_on_wand_of_wishing():
    """grant_wish flips WISHLESS regardless of what is wished for.

    Cite: vendor/nethack/src/insight.c:2183 u.uconduct.wishes.
    The function represents the wand-of-wishing / scroll-of-wishing grant
    path (wizard.c::makewish).
    """
    from Nethax.nethax.subsystems.wish import grant_wish

    state = _fresh_state()
    assert not bool(state.conduct.violations[int(Conduct.WISHLESS)])
    new_state = grant_wish(state, _RNG, b"long sword")
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)])
    # Non-artifact wish must NOT set ARTIWISHLESS.
    assert not bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)])


# ---------------------------------------------------------------------------
# test_polyselfless_violated_on_poly_potion
# Cite: vendor/nethack/src/insight.c:2175  u.uconduct.polyselfs
#       vendor/nethack/src/polyself.c::polyself
# ---------------------------------------------------------------------------

def test_polyselfless_violated_on_poly_potion():
    """polymorph_player flips POLYSELFLESS.

    Represents drinking a potion of polymorph (polyself.c::polyself path).
    Cite: vendor/nethack/src/insight.c:2175 u.uconduct.polyselfs.
    """
    from Nethax.nethax.subsystems.polymorph import polymorph_player
    from Nethax.nethax.constants.monsters import MONSTERS

    target_idx = 0
    for i, m in enumerate(MONSTERS):
        if m.attacks and m.attacks[0][0] != 0:
            target_idx = i
            break

    state = _fresh_state()
    state = state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(12),
        player_con=jnp.int8(14),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
        player_role=jnp.int8(0),
        player_ac=jnp.int32(10),
    )
    assert not bool(state.conduct.violations[int(Conduct.POLYSELFLESS)])
    new_state = polymorph_player(state, _RNG, target_idx, controlled=False)
    assert bool(new_state.conduct.violations[int(Conduct.POLYSELFLESS)])


# ---------------------------------------------------------------------------
# test_kept_conducts_award_bonus
# Cite: vendor/nethack/src/insight.c::show_conduct (conduct contribution)
#       Nethax scoring.py::compute_final_score
# ---------------------------------------------------------------------------

def test_kept_conducts_award_bonus():
    """A completely clean run awards CONDUCT_BONUS for all 13 kept conducts.

    All 13 conducts intact on a fresh state means compute_final_score
    includes the full conduct contribution.  We verify the conduct slice
    specifically via scoring.compute_conduct_bonus.

    Cite: vendor/nethack/src/insight.c::show_conduct — per-conduct bonus
    table; Nethax collapses to flat bonuses in _CONDUCT_BONUS.
    """
    from Nethax.nethax.subsystems.scoring import compute_conduct_bonus, _CONDUCT_BONUS

    state = _fresh_state()
    # Confirm no conducts are violated.
    assert not bool(state.conduct.violations.any())

    bonus = compute_conduct_bonus(state)
    expected = sum(_CONDUCT_BONUS[Conduct(i)] for i in range(N_CONDUCTS))
    assert int(bonus) == expected
    assert expected > 0, "sanity: non-zero conduct bonus expected"
