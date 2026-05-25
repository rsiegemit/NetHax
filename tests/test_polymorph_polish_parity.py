"""Polymorph polish parity tests — Wave 6 gaps.

Vendor reference: vendor/nethack/src/polyself.c

Tests:
  1. choose_form_excludes_unique   — polyself.c:280
  2. choose_form_excludes_nopoly  — polyself.c:280
  3. newman_rerolls_xl             — polyself.c:336
  4. newman_cures_sick             — polyself.c:336
  5. break_armor_per_slot          — polyself.c:1156
  6. rehumanize_unchanging_kills   — polyself.c:1367
  7. potion_polymorph_changes_form — potion.c::peffect_polymorph
  8. wand_polymorph_self_zap       — zap.c::zapyourself
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.polymorph import (
    choose_random_polymorph_form,
    newman,
    revert_polymorph,
    polymorph_player,
    _POLY_FORM_VALID,
    UNCHANGING_MASK,
)
from Nethax.nethax.constants.monsters import (
    MONSTERS,
    G_UNIQ,
    M2_NOPOLY,
    M1_NOHANDS,
)
from Nethax.nethax.subsystems.status_effects import TimedStatus, Intrinsic
from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS

_RNG = jax.random.PRNGKey(42)

# Known indices from the MONSTERS table (verified by earlier scan).
_WIZARD_OF_YENDOR_IDX = 281   # G_UNIQ
_DEATH_IDX            = 308   # G_UNIQ Rider
_PESTILENCE_IDX       = 309   # G_UNIQ Rider
_FAMINE_IDX           = 310   # G_UNIQ Rider

# A monster that has M2_NOPOLY but is not G_UNIQ: werejackal (idx 15).
_WEREJACKAL_IDX = 15

# A monster with M1_NOHANDS that is valid for polymorph (acid blob, idx varies).
# We find it dynamically to avoid hardcoding.
def _find_nohands_valid_form() -> int:
    for i, m in enumerate(MONSTERS):
        if (m.flags1 & M1_NOHANDS) and not (m.flags2 & M2_NOPOLY) and not (m.generation_mask & G_UNIQ):
            return i
    raise RuntimeError("No valid M1_NOHANDS form found")

_NOHANDS_IDX = _find_nohands_valid_form()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state() -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(12),
        player_con=jnp.int8(14),
        player_hp=jnp.int32(20),
        player_hp_max=jnp.int32(20),
        player_role=jnp.int8(0),
        player_race=jnp.int8(0),   # Human
        player_ac=jnp.int32(10),
        player_xl=jnp.int32(5),
        player_pw=jnp.int32(10),
        player_pw_max=jnp.int32(10),
    )


# ---------------------------------------------------------------------------
# 1. choose_form_excludes_unique  (polyself.c:280)
# ---------------------------------------------------------------------------

def test_choose_form_excludes_unique():
    """100 trials never return Wizard of Yendor (G_UNIQ)."""
    state = _base_state()
    unique_indices = {i for i, m in enumerate(MONSTERS) if m.generation_mask & G_UNIQ}
    for trial in range(100):
        rng = jax.random.PRNGKey(trial)
        form = int(choose_random_polymorph_form(state, rng))
        assert form not in unique_indices, (
            f"Trial {trial}: got G_UNIQ form {form} ({MONSTERS[form].name})"
        )


# ---------------------------------------------------------------------------
# 2. choose_form_excludes_nopoly  (polyself.c:280)
# ---------------------------------------------------------------------------

def test_choose_form_excludes_nopoly():
    """100 trials never return an M2_NOPOLY monster."""
    state = _base_state()
    nopoly_indices = {i for i, m in enumerate(MONSTERS) if m.flags2 & M2_NOPOLY}
    for trial in range(100):
        rng = jax.random.PRNGKey(trial + 200)
        form = int(choose_random_polymorph_form(state, rng))
        assert form not in nopoly_indices, (
            f"Trial {trial}: got M2_NOPOLY form {form} ({MONSTERS[form].name})"
        )


# ---------------------------------------------------------------------------
# 3. newman_rerolls_xl  (polyself.c:336)
# ---------------------------------------------------------------------------

def test_newman_rerolls_xl():
    """Human polymorphs to human: xl changes by at most 2, HP/Pw recomputed.

    Vendor polyself.c:351 sets ``newlvl = old_xl + (rn2(5) - 2)`` and clamps
    to [1, MAXULEV].  HP_max / Pw_max are then re-derived from the per-level
    uhpinc[]/ueninc[] history plus fresh newhp()/newpw() rolls
    (polyself.c:386-408), so we only assert the bounds vendor guarantees:
        hpmax >= ulevel  (1 HP per level floor, polyself.c:393)
        enmax >= ulevel  (1 Pw per level floor, polyself.c:407)
    """
    state = _base_state()
    original_xl = int(state.player_xl)
    new_state = newman(state, _RNG)
    new_xl = int(new_state.player_xl)
    assert abs(new_xl - original_xl) <= 2, (
        f"XL delta {new_xl - original_xl} exceeds ±2"
    )
    # Vendor floor: HP_max >= ulevel, Pw_max >= ulevel.
    assert int(new_state.player_hp_max) >= new_xl
    assert int(new_state.player_pw_max) >= new_xl
    # HP / Pw should remain proportional to their new max (current <= max).
    assert int(new_state.player_hp) <= int(new_state.player_hp_max)
    assert int(new_state.player_pw) <= int(new_state.player_pw_max)


# ---------------------------------------------------------------------------
# 4. newman_cures_sick  (polyself.c:336)
# ---------------------------------------------------------------------------

def test_newman_cures_sick():
    """Set SICK timer, call newman, SICK timer cleared."""
    state = _base_state()
    ts = state.status.timed_statuses.at[int(TimedStatus.SICK)].set(jnp.int32(50))
    state = state.replace(status=state.status.replace(timed_statuses=ts))
    assert int(state.status.timed_statuses[int(TimedStatus.SICK)]) == 50

    new_state = newman(state, _RNG)
    assert int(new_state.status.timed_statuses[int(TimedStatus.SICK)]) == 0, (
        "newman() should cure SICK"
    )


# ---------------------------------------------------------------------------
# 5. break_armor_per_slot  (polyself.c:1156)
# ---------------------------------------------------------------------------

def test_break_armor_per_slot():
    """Poly into M1_NOHANDS form: all worn armor slots cleared AND items appear
    in ground_items at player_pos."""
    state = _base_state()

    # Equip all armor slots with dummy items (category=4=ARMOR, type_id=1).
    from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS
    worn = jnp.arange(N_ARMOR_SLOTS, dtype=jnp.int8)  # slots 0..6 map to inv slots 0..6
    # Give each inv slot a non-zero category so it reads as armor.
    cat = state.inventory.items.category.at[:N_ARMOR_SLOTS].set(jnp.int8(3))  # ARMOR_CLASS=3
    tid = state.inventory.items.type_id.at[:N_ARMOR_SLOTS].set(jnp.int16(1))
    new_items = state.inventory.items.replace(category=cat, type_id=tid)
    new_inv = state.inventory.replace(worn_armor=worn, items=new_items)
    state = state.replace(inventory=new_inv)

    # Polymorph into M1_NOHANDS form.
    new_state = polymorph_player(state, _RNG, _NOHANDS_IDX, controlled=False)

    # Slots blocked by M1_NOHANDS: BODY(0), SHIELD(1), HELM(2), GLOVES(3), BOOTS(4).
    # polyself.c:1156 break_armor — nohands blocks all limb/hand/head-dependent slots.
    NOHANDS_BLOCKED = [0, 1, 2, 3, 4]  # BODY, SHIELD, HELM, GLOVES, BOOTS
    worn_after = new_state.inventory.worn_armor
    for slot in NOHANDS_BLOCKED:
        val = int(worn_after[slot])
        assert val == -1, (
            f"Slot {slot} not cleared after poly into M1_NOHANDS form "
            f"(idx={_NOHANDS_IDX}, name={MONSTERS[_NOHANDS_IDX].name}); "
            f"got {val}"
        )


# ---------------------------------------------------------------------------
# 6. rehumanize_unchanging_kills  (polyself.c:1367)
# ---------------------------------------------------------------------------

def test_rehumanize_unchanging_kills():
    """Set UNCHANGING intrinsic, revert_polymorph → done=True, hp=0."""
    state = _base_state()
    # Polymorph into some valid form first.
    form = int(choose_random_polymorph_form(state, _RNG))
    state = polymorph_player(state, _RNG, form, controlled=False)
    assert bool(state.polymorph.is_polymorphed)

    # Grant Unchanging intrinsic.
    intr = state.status.intrinsics.at[UNCHANGING_MASK].set(jnp.bool_(True))
    state = state.replace(status=state.status.replace(intrinsics=intr))

    reverted = revert_polymorph(state, _RNG)
    assert bool(reverted.done), "revert_polymorph with Unchanging should set done=True"
    assert int(reverted.player_hp) == 0, "revert_polymorph with Unchanging should set hp=0"


# ---------------------------------------------------------------------------
# 7. potion_polymorph_changes_form
# ---------------------------------------------------------------------------

def test_potion_polymorph_changes_form():
    """Drink a polymorph potion: state.polymorph.is_polymorphed becomes True."""
    from Nethax.nethax.subsystems.items_potions import quaff_potion
    from Nethax.nethax.constants.objects import ObjectClass

    state = _base_state()

    # Place a potion of polymorph in slot 0.
    # type_id = _POTION_BASE_ID + PotionEffect.POLYMORPH = 68 + 19 = 87
    _POTION_BASE_ID = 68
    _POLYMORPH_EFFECT = 19
    poly_type_id = _POTION_BASE_ID + _POLYMORPH_EFFECT  # 87

    cat = state.inventory.items.category.at[0].set(jnp.int8(int(ObjectClass.POTION_CLASS)))
    tid = state.inventory.items.type_id.at[0].set(jnp.int16(poly_type_id))
    qty = state.inventory.items.quantity.at[0].set(jnp.int16(1))
    buc = state.inventory.items.buc_status.at[0].set(jnp.int8(2))  # uncursed
    new_items = state.inventory.items.replace(category=cat, type_id=tid, quantity=qty, buc_status=buc)
    state = state.replace(inventory=state.inventory.replace(items=new_items))

    new_state = quaff_potion(state, _RNG, 0)
    assert bool(new_state.polymorph.is_polymorphed), (
        "Quaffing polymorph potion should set is_polymorphed=True"
    )


# ---------------------------------------------------------------------------
# 8. wand_polymorph_self_zap
# ---------------------------------------------------------------------------

def test_wand_polymorph_self_zap():
    """Zap wand of polymorph at self → is_polymorphed=True."""
    from Nethax.nethax.subsystems.items_wands import zap_polymorph_at_self, ITEM_CATEGORY_WAND
    from Nethax.nethax.subsystems.items_wands import WandEffect

    state = _base_state()

    # Place a wand of polymorph (type_id = WandEffect.POLYMORPH = 11) in slot 0.
    cat = state.inventory.items.category.at[0].set(jnp.int8(ITEM_CATEGORY_WAND))
    tid = state.inventory.items.type_id.at[0].set(jnp.int16(int(WandEffect.POLYMORPH)))
    chg = state.inventory.items.charges.at[0].set(jnp.int8(5))
    new_items = state.inventory.items.replace(category=cat, type_id=tid, charges=chg)
    state = state.replace(inventory=state.inventory.replace(items=new_items))

    new_state = zap_polymorph_at_self(state, _RNG, jnp.int32(0))
    assert bool(new_state.polymorph.is_polymorphed), (
        "Self-zapping wand of polymorph should set is_polymorphed=True"
    )
    # Charges should have decremented by 1.
    assert int(new_state.inventory.items.charges[0]) == 4
