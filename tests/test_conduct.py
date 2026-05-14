"""Wave 4 conduct subsystem tests.

Covers:
  - Fresh state: all 13 conducts intact.
  - mark_violated / mark_violated_if helpers.
  - Per-action trigger wiring:
      FOODLESS / VEGETARIAN / VEGAN  via _handle_eat
      ATHEIST                        via handle_pray  (pre-existing wiring)
      WEAPONLESS                     via inventory.handle_wield
      PACIFIST                       via combat.melee_attack on a kill
      ILLITERATE                     via items_scrolls.handle_read
                                     and items_spellbooks.handle_read_spellbook
      POLYSELFLESS                   via polymorph.polymorph_player  (pre-existing wiring)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.conduct import (
    Conduct,
    ConductState,
    N_CONDUCTS,
    mark_violated,
    mark_violated_if,
)
from Nethax.nethax.constants.objects import ObjectClass


_RNG = jax.random.PRNGKey(0)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


# ---------------------------------------------------------------------------
# Schema / helper tests
# ---------------------------------------------------------------------------

def test_all_conducts_intact_on_fresh_state():
    """Fresh ConductState has 13 violations, all False."""
    cs = ConductState.default()
    assert int(N_CONDUCTS) == 13
    assert cs.violations.shape == (13,)
    assert bool(cs.violations.any()) is False


def test_mark_violated_helper():
    """mark_violated flips the indexed bit on EnvState.conduct."""
    state = _fresh_state()
    new_state = mark_violated(state, int(Conduct.WEAPONLESS))
    assert bool(new_state.conduct.violations[int(Conduct.WEAPONLESS)]) is True
    # other bits stay False
    assert int(new_state.conduct.violations.sum()) == 1


def test_mark_violated_if_helper():
    """mark_violated_if only flips when condition is True."""
    state = _fresh_state()
    no_op = mark_violated_if(state, int(Conduct.PACIFIST), jnp.bool_(False))
    assert bool(no_op.conduct.violations[int(Conduct.PACIFIST)]) is False
    flipped = mark_violated_if(state, int(Conduct.PACIFIST), jnp.bool_(True))
    assert bool(flipped.conduct.violations[int(Conduct.PACIFIST)]) is True


# ---------------------------------------------------------------------------
# FOODLESS / VEGAN / VEGETARIAN via _handle_eat
# ---------------------------------------------------------------------------

def _state_with_food(type_id: int) -> EnvState:
    state = _fresh_state()
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.FOOD_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(0)),
    )
    return state.replace(inventory=state.inventory.replace(items=new_items))


# Wave 6 parity-fix: updated to match vendor/nethack/src/eat.c VEGAN/VEGETARIAN
# tagging via objects.h oc_material (FLESH = meat + animal product; VEGGY = safe).
# Real OBJECTS indices replace the legacy synthetic type_ids 1/3/4/5; resolved
# by name to stay robust against OBJECTS aggregation reorderings.
from Nethax.nethax.constants.objects import OBJECTS as _OBJECTS


def _object_idx(name: str) -> int:
    for i, o in enumerate(_OBJECTS):
        if o.name == name and int(o.class_) == int(ObjectClass.FOOD_CLASS):
            return i
    raise ValueError(f"No FOOD entry named {name!r}")


_FOOD_RATION_TYPE_ID = _object_idx("food ration")
_APPLE_TYPE_ID = _object_idx("apple")
_CORPSE_TYPE_ID = _object_idx("corpse")
_EGG_TYPE_ID = _object_idx("egg")
_MEATBALL_TYPE_ID = _object_idx("meatball")


def test_foodless_violated_on_eat():
    """Eating any food flips FOODLESS (eat.c::eatfood)."""
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    state = _state_with_food(type_id=_FOOD_RATION_TYPE_ID)
    assert bool(state.conduct.violations[int(Conduct.FOODLESS)]) is False
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.FOODLESS)]) is True


def test_vegan_violated_on_meat():
    """Eating FLESH food (meatball) flips VEGAN."""
    # Wave 6 parity-fix: updated to match vendor/nethack/src/eat.c VEGAN logic
    # via objects.h MAT_FLESH (meatball is FLESH).
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    state = _state_with_food(type_id=_MEATBALL_TYPE_ID)
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.VEGAN)]) is True


def test_vegetarian_violated_on_corpse():
    """Eating a corpse (FLESH) flips VEGETARIAN and VEGAN."""
    # Wave 6 parity-fix: updated to match vendor/nethack/src/eat.c VEGAN/VEGETARIAN
    # tagging via objects.h MAT_FLESH.
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    state = _state_with_food(type_id=_CORPSE_TYPE_ID)
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.VEGETARIAN)]) is True
    # FLESH also violates VEGAN
    assert bool(new_state.conduct.violations[int(Conduct.VEGAN)]) is True


def test_vegan_and_vegetarian_violated_on_egg():
    """Per vendor objects.h, egg has oc_material == FLESH so eating egg flips
    BOTH VEGAN and VEGETARIAN (animal product *and* animal flesh).
    """
    # Wave 6 parity-fix: updated to match vendor/nethack/src/eat.c VEGAN/VEGETARIAN
    # tagging — vendor egg.material == FLESH (vendor/nethack/include/objects.h line 1052).
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    state = _state_with_food(type_id=_EGG_TYPE_ID)
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.VEGAN)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.VEGETARIAN)]) is True


def test_eat_veggy_food_does_not_flip_vegan():
    """Eating a food ration (VEGGY) leaves VEGAN and VEGETARIAN intact."""
    # Wave 6 parity-fix: updated to match vendor/nethack/src/eat.c via
    # objects.h MAT_VEGGY (food_ration material is VEGGY).
    from Nethax.nethax.subsystems.action_dispatch import _handle_eat
    state = _state_with_food(type_id=_FOOD_RATION_TYPE_ID)
    new_state = _handle_eat(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.FOODLESS)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.VEGAN)]) is False
    assert bool(new_state.conduct.violations[int(Conduct.VEGETARIAN)]) is False


# ---------------------------------------------------------------------------
# ATHEIST via prayer (pre-existing wiring — verify still works)
# ---------------------------------------------------------------------------

def test_atheist_violated_on_pray():
    """handle_pray flips ATHEIST (pre-existing wiring; insight.c ~2134)."""
    from Nethax.nethax.subsystems.prayer import handle_pray
    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.ATHEIST)]) is False
    new_state = handle_pray(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ATHEIST)]) is True


# ---------------------------------------------------------------------------
# WEAPONLESS via inventory.handle_wield
# ---------------------------------------------------------------------------

def test_weaponless_violated_on_wield():
    """Wielding a weapon flips WEAPONLESS (wield.c::wieldwep)."""
    from Nethax.nethax.subsystems.inventory import handle_wield, ItemCategory
    state = _fresh_state()
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ItemCategory.WEAPON)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))
    assert bool(state.conduct.violations[int(Conduct.WEAPONLESS)]) is False
    new_state = handle_wield(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.WEAPONLESS)]) is True


def test_weaponless_intact_when_no_weapon():
    """handle_wield with empty inventory does NOT flip WEAPONLESS."""
    from Nethax.nethax.subsystems.inventory import handle_wield
    state = _fresh_state()
    new_state = handle_wield(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.WEAPONLESS)]) is False


# ---------------------------------------------------------------------------
# PACIFIST via combat.melee_attack
# ---------------------------------------------------------------------------

def test_pacifist_violated_on_kill():
    """Killing a monster via melee_attack flips PACIFIST.

    Setup: place a fragile monster directly in monster_ai, hit it once;
    if the damage roll kills it, conduct should flip.  Because to_hit /
    damage rolls are stochastic, we set the monster to 1 hp and rely on
    1d4 base damage to land at least 1.  We also brute the attack with a
    near-zero target AC for stable hit rate.
    """
    from Nethax.nethax.subsystems.combat import melee_attack
    state = _fresh_state()
    mai = state.monster_ai
    # Configure monster 0 to be trivially killable.
    new_mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(1)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(1)),
        ac=mai.ac.at[0].set(jnp.int32(50)),  # extremely hittable (tmp = 1+abon+ac)
        is_large=mai.is_large.at[0].set(jnp.bool_(False)),
    )
    state = state.replace(monster_ai=new_mai)
    assert bool(state.conduct.violations[int(Conduct.PACIFIST)]) is False

    # Try several RNGs — at least one should kill (1d4 base, target AC -50).
    killed_any = False
    for seed in range(8):
        rng = jax.random.PRNGKey(seed)
        new_state, dmg, hit = melee_attack(state, rng, jnp.int32(0))
        if bool(new_state.conduct.violations[int(Conduct.PACIFIST)]):
            killed_any = True
            break
    assert killed_any, "expected at least one kill across attempts"


def test_pacifist_intact_on_miss():
    """melee_attack that does not kill the target does NOT flip PACIFIST."""
    from Nethax.nethax.subsystems.combat import melee_attack
    state = _fresh_state()
    mai = state.monster_ai
    # Monster fully alive and tanky — single 1d4 base hit cannot kill 100 hp.
    new_mai = mai.replace(
        alive=mai.alive.at[0].set(jnp.bool_(True)),
        hp=mai.hp.at[0].set(jnp.int32(100)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(100)),
        ac=mai.ac.at[0].set(jnp.int32(10)),
        is_large=mai.is_large.at[0].set(jnp.bool_(False)),
    )
    state = state.replace(monster_ai=new_mai)
    new_state, _, _ = melee_attack(state, _RNG, jnp.int32(0))
    assert bool(new_state.conduct.violations[int(Conduct.PACIFIST)]) is False


# ---------------------------------------------------------------------------
# ILLITERATE via scrolls and spellbooks
# ---------------------------------------------------------------------------

def test_illiterate_violated_on_read_scroll():
    """handle_read on a scroll flips ILLITERATE (read.c::doread)."""
    from Nethax.nethax.subsystems.items_scrolls import handle_read
    state = _fresh_state()
    items = state.inventory.items
    new_items = items.replace(
        category=items.category.at[0].set(jnp.int8(ObjectClass.SCROLL_CLASS)),
        type_id=items.type_id.at[0].set(jnp.int16(1)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
    )
    state = state.replace(inventory=state.inventory.replace(items=new_items))
    assert bool(state.conduct.violations[int(Conduct.ILLITERATE)]) is False
    new_state = handle_read(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ILLITERATE)]) is True


def test_illiterate_violated_on_read_spellbook():
    """handle_read_spellbook flips ILLITERATE (read.c::study_book)."""
    from Nethax.nethax.subsystems.items_spellbooks import handle_read_spellbook
    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.ILLITERATE)]) is False
    new_state = handle_read_spellbook(state, _RNG, slot_idx=0)
    assert bool(new_state.conduct.violations[int(Conduct.ILLITERATE)]) is True


# ---------------------------------------------------------------------------
# POLYSELFLESS via polymorph (pre-existing wiring — verify)
# ---------------------------------------------------------------------------

def test_polyselfless_violated_on_polymorph():
    """polymorph_player flips POLYSELFLESS (pre-existing wiring)."""
    from Nethax.nethax.subsystems.polymorph import polymorph_player
    from Nethax.nethax.constants.monsters import MONSTERS

    target = 0
    for i, m in enumerate(MONSTERS):
        if m.attacks and m.attacks[0][0] != 0:
            target = i
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
    assert bool(state.conduct.violations[int(Conduct.POLYSELFLESS)]) is False
    new_state = polymorph_player(state, _RNG, target, controlled=False)
    assert bool(new_state.conduct.violations[int(Conduct.POLYSELFLESS)]) is True


# ---------------------------------------------------------------------------
# Wave 5 Phase 4 — POLYPILELESS / GENOCIDELESS / ELBERETHLESS
# ---------------------------------------------------------------------------

def test_polypileless_violated_on_poly_trap_pile():
    """POLY_TRAP on a tile with items flips POLYPILELESS (trap.c::do_poly_pile)."""
    from Nethax.nethax.subsystems.traps import poly_pile_effect
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _fresh_state()
    state = state.replace(player_pos=jnp.array([5, 5], dtype=jnp.int16))

    # Place a single item on the floor at (5, 5), slot 0 of the stack.
    b = int(state.dungeon.current_branch)
    lv = int(state.dungeon.current_level) - 1
    g = state.ground_items
    new_g = g.replace(
        category=g.category.at[b, lv, 5, 5, 0].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=g.type_id.at[b, lv, 5, 5, 0].set(jnp.int16(7)),
        quantity=g.quantity.at[b, lv, 5, 5, 0].set(jnp.int16(1)),
    )
    state = state.replace(ground_items=new_g)
    assert bool(state.conduct.violations[int(Conduct.POLYPILELESS)]) is False

    new_state = poly_pile_effect(state, _RNG, 5, 5)
    assert bool(new_state.conduct.violations[int(Conduct.POLYPILELESS)]) is True


def test_polypileless_intact_on_empty_tile():
    """poly_pile_effect on an empty tile does NOT flip POLYPILELESS."""
    from Nethax.nethax.subsystems.traps import poly_pile_effect

    state = _fresh_state()
    new_state = poly_pile_effect(state, _RNG, 3, 3)
    assert bool(new_state.conduct.violations[int(Conduct.POLYPILELESS)]) is False


def test_genocideless_violated_on_genocide_scroll():
    """apply_genocide flips GENOCIDELESS (read.c::do_genocide)."""
    from Nethax.nethax.subsystems.items_scrolls import apply_genocide

    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.GENOCIDELESS)]) is False
    new_state = apply_genocide(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.GENOCIDELESS)]) is True


def test_elberethless_violated_on_engrave():
    """handle_engrave flips ELBERETHLESS (engrave.c::doengrave)."""
    from Nethax.nethax.subsystems.engrave import handle_engrave

    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.ELBERETHLESS)]) is False
    new_state = handle_engrave(state, _RNG)
    assert bool(new_state.conduct.violations[int(Conduct.ELBERETHLESS)]) is True


# ---------------------------------------------------------------------------
# Wave 6 Phase B — WISHLESS / ARTIWISHLESS via wish subsystem
# ---------------------------------------------------------------------------

def test_wishless_violated_on_wish():
    """grant_wish flips WISHLESS (wizard.c::makewish, insight.c ~2163)."""
    from Nethax.nethax.subsystems.wish import grant_wish

    state = _fresh_state()
    assert bool(state.conduct.violations[int(Conduct.WISHLESS)]) is False
    new_state = grant_wish(state, _RNG, b"long sword")
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True


def test_artiwishless_violated_on_artifact_wish():
    """grant_wish on an artifact flips ARTIWISHLESS (insight.c ~2166)."""
    from Nethax.nethax.subsystems.wish import grant_wish

    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"Excalibur")
    assert bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)]) is True
    assert bool(new_state.conduct.violations[int(Conduct.WISHLESS)]) is True


def test_artiwishless_not_violated_on_normal_wish():
    """A non-artifact wish does NOT flip ARTIWISHLESS."""
    from Nethax.nethax.subsystems.wish import grant_wish

    state = _fresh_state()
    new_state = grant_wish(state, _RNG, b"long sword")
    assert bool(new_state.conduct.violations[int(Conduct.ARTIWISHLESS)]) is False
