"""Polymorph polish parity tests — Wave 6 gap fixes (batch 2).

Vendor reference: vendor/nethack/src/polyself.c

Tests:
  1. test_poly_while_riding_dismounts     — polyself.c:1412
  2. test_silver_armor_drops_on_vampire_poly — polyself.c::retouch_equipment
  3. test_genocide_self_kills_on_revert   — polyself.c::rehumanize (ugenocided)
  4. test_newman_resets_hunger            — polyself.c:414
  5. test_poly_clamps_pw                  — polyself.c (Pw rescale)
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
    _FORM_HATES_SILVER,
    _ITEM_IS_SILVER,
)
from Nethax.nethax.constants.monsters import (
    MONSTERS,
    G_UNIQ,
    M2_NOPOLY,
    M2_WERE,
    M2_DEMON,
    M2_UNDEAD,
    MonsterSymbol,
)
from Nethax.nethax.constants.objects import OBJECTS, Material, ObjectClass
from Nethax.nethax.subsystems.inventory import N_ARMOR_SLOTS, ArmorSlot

_RNG = jax.random.PRNGKey(99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state() -> EnvState:
    state = EnvState.default(_RNG)
    return state.replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(12),
        player_con=jnp.int8(14),
        player_hp=jnp.int32(40),
        player_hp_max=jnp.int32(40),
        player_role=jnp.int8(0),
        player_race=jnp.int8(0),   # Human
        player_ac=jnp.int32(10),
        player_xl=jnp.int32(5),
        player_pw=jnp.int32(10),
        player_pw_max=jnp.int32(10),
    )


def _find_vampire_idx() -> int:
    """Return MONSTERS index of "vampire" (silver-allergic via M2_UNDEAD + S_VAMPIRE)."""
    for i, m in enumerate(MONSTERS):
        if m.name == "vampire" and m.symbol == MonsterSymbol.S_VAMPIRE:
            return i
    raise RuntimeError("vampire not found in MONSTERS")


def _find_silver_armor_idx() -> int:
    """Return OBJECTS index of the first ARMOR_CLASS item made of SILVER."""
    for i, o in enumerate(OBJECTS):
        if o.class_ == ObjectClass.ARMOR_CLASS and o.material == Material.SILVER:
            return i
    raise RuntimeError("No SILVER ARMOR_CLASS object found in OBJECTS")


def _find_valid_non_silver_form() -> int:
    """Return a form valid for poly that does NOT hate silver (for control tests)."""
    from Nethax.nethax.constants.monsters import M2_UNDEAD
    for i, m in enumerate(MONSTERS):
        if not (m.generation_mask & G_UNIQ) and not (m.flags2 & M2_NOPOLY):
            hates = (
                bool(m.flags2 & M2_WERE)
                or bool(m.flags2 & M2_DEMON)
                or (bool(m.flags2 & M2_UNDEAD) and m.symbol == MonsterSymbol.S_VAMPIRE)
            )
            if not hates:
                return i
    raise RuntimeError("No non-silver-hating poly form found")


_VAMPIRE_IDX      = _find_vampire_idx()
_SILVER_ARMOR_IDX = _find_silver_armor_idx()


# ---------------------------------------------------------------------------
# 1. test_poly_while_riding_dismounts  (polyself.c:1412)
# ---------------------------------------------------------------------------

def test_poly_while_riding_dismounts():
    """Player riding steed polymorphs → steed_mid cleared, HP decreases (fall dmg).

    polyself.c:1412 — when can_ride() fails after poly, dismount_steed() is
    called.  We force-dismount on every poly for safety.
    """
    state = _base_state()
    # Simulate riding: set player_steed_mid to a non-zero sentinel.
    state = state.replace(player_steed_mid=jnp.uint32(5))
    assert int(state.player_steed_mid) == 5

    hp_before = int(state.player_hp)

    form = _find_valid_non_silver_form()
    new_state = polymorph_player(state, _RNG, form, controlled=False)

    assert int(new_state.player_steed_mid) == 0, (
        "polymorph_player should clear player_steed_mid when riding"
    )
    assert int(new_state.player_hp) < hp_before, (
        "polymorph_player should apply fall damage (1d6) when dismounting"
    )


# ---------------------------------------------------------------------------
# 2. test_silver_armor_drops_on_vampire_poly  (polyself.c::retouch_equipment)
# ---------------------------------------------------------------------------

def test_silver_armor_drops_on_vampire_poly():
    """Wearing a silver item and polying into a vampire: item drops to ground_items.

    polyself.c::retouch_equipment — silver-allergic forms (vampires, were,
    demons) have silver worn items dropped and take 1d6 burn damage per item.
    """
    state = _base_state()

    # Equip a silver armor item in the SHIELD slot (slot 1).
    # Set inventory slot 0 to have the silver object's type_id.
    silver_tid = jnp.int16(_SILVER_ARMOR_IDX)
    new_cat = state.inventory.items.category.at[0].set(jnp.int8(int(ObjectClass.ARMOR_CLASS)))
    new_tid = state.inventory.items.type_id.at[0].set(silver_tid)
    new_items = state.inventory.items.replace(category=new_cat, type_id=new_tid)
    # Wear it in slot 1 (SHIELD).
    new_worn = state.inventory.worn_armor.at[1].set(jnp.int8(0))
    new_inv  = state.inventory.replace(worn_armor=new_worn, items=new_items)
    state    = state.replace(inventory=new_inv)

    assert int(state.inventory.worn_armor[1]) == 0  # slot 1 → inv index 0

    hp_before = int(state.player_hp)

    new_state = polymorph_player(state, _RNG, _VAMPIRE_IDX, controlled=False)

    # Worn slot should be cleared.
    assert int(new_state.inventory.worn_armor[1]) == -1, (
        "Silver item in SHIELD slot should be dropped when polying into vampire"
    )

    # Item should appear in ground_items at player_pos.
    p_row = int(new_state.player_pos[0])
    p_col = int(new_state.player_pos[1])
    ground_tids = new_state.ground_items.type_id[0, 0, p_row, p_col]
    found = any(int(ground_tids[i]) == _SILVER_ARMOR_IDX for i in range(ground_tids.shape[0]))
    assert found, (
        f"Silver item (type_id={_SILVER_ARMOR_IDX}) should appear in ground_items; "
        f"ground tids: {[int(ground_tids[i]) for i in range(ground_tids.shape[0])]}"
    )

    # HP should have decreased (burn damage).
    assert int(new_state.player_hp) < hp_before, (
        "polymorph into vampire with silver item should deal burn damage"
    )


# ---------------------------------------------------------------------------
# 3. test_genocide_self_kills_on_revert  (polyself.c::rehumanize)
# ---------------------------------------------------------------------------

def test_genocide_self_kills_on_revert():
    """If player's own race is genocided, revert_polymorph kills the player.

    polyself.c::rehumanize → ugenocided() check (polyself.c:233).
    player_race=0 (Human) → genocided_species[0]=True → done=True on revert.
    """
    state = _base_state()

    # Polymorph into any valid form.
    form = _find_valid_non_silver_form()
    state = polymorph_player(state, _RNG, form, controlled=False)
    assert bool(state.polymorph.is_polymorphed)

    # Genocide human race (index 0 = player_race).
    new_genocided = state.genocided_species.at[0].set(jnp.bool_(True))
    state = state.replace(genocided_species=new_genocided)

    reverted = revert_polymorph(state, _RNG)

    assert bool(reverted.done), (
        "revert_polymorph should set done=True when player's race is genocided"
    )
    assert int(reverted.player_hp) == 0, (
        "revert_polymorph should set hp=0 when player's race is genocided"
    )


# ---------------------------------------------------------------------------
# 4. test_newman_resets_hunger  (polyself.c:336)
# ---------------------------------------------------------------------------

def test_newman_resets_hunger():
    """newman() resets nutrition to rn1(500, 500) ∈ [500, 999] per vendor.

    Wave 36e (commit eedac47) corrected newman to vendor polyself.c:414:
        u.uhunger = rn1(500, 500);
    where rn1(x, y) := rn2(x) + y, giving the range [500, 999] inclusive
    (uniform).  The previous assertion ``== 1000`` pinned the pre-wave-36e
    flat-1000 simplification, which was a documented Nethax-only divergence
    cited (incorrectly) at polyself.c:336 — the real vendor reference is
    line 414.
    """
    state = _base_state()

    # Set nutrition to a very low value (starving).
    new_status = state.status.replace(nutrition=jnp.int32(-100))
    state = state.replace(status=new_status)
    assert int(state.status.nutrition) == -100

    new_state = newman(state, _RNG)

    n = int(new_state.status.nutrition)
    assert 500 <= n < 1000, (
        f"newman() should reset nutrition to rn1(500, 500) ∈ [500, 999); got {n}"
    )


# ---------------------------------------------------------------------------
# 5. test_poly_clamps_pw  (polyself.c — Pw rescale)
# ---------------------------------------------------------------------------

def test_poly_clamps_pw():
    """After polymorph, player_pw is clamped to player_pw_max.

    polyself.c — HP and Pw are both clamped to the new form's max on poly.
    We test that if player_pw > player_pw_max after poly, it's clamped.
    """
    state = _base_state()
    # Set pw=50 but keep pw_max=10 (artificially high pw).
    state = state.replace(
        player_pw=jnp.int32(50),
        player_pw_max=jnp.int32(10),
    )
    assert int(state.player_pw) == 50

    form = _find_valid_non_silver_form()
    new_state = polymorph_player(state, _RNG, form, controlled=False)

    assert int(new_state.player_pw) <= int(new_state.player_pw_max), (
        f"player_pw={int(new_state.player_pw)} should be <= "
        f"player_pw_max={int(new_state.player_pw_max)} after poly"
    )
    assert int(new_state.player_pw) <= 10, (
        f"player_pw should have been clamped from 50 to pw_max=10; "
        f"got {int(new_state.player_pw)}"
    )
