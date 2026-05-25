"""Artifact powers parity tests — spec_dbon and arti_prop.

Tests cover Part A (damage bonus on hit) and Part B (wield intrinsic grants).

Canonical references:
  vendor/nethack/src/artifact.c::spec_dbon       lines 1091-1109
  vendor/nethack/src/artifact.c::spec_applies    lines 1009-1060
  vendor/nethack/src/artifact.c::arti_prop       lines 880-885
  vendor/nethack/include/artilist.h              lines 85-88, 109-112,
                                                 134-136, 138-140, 149-155
  vendor/nethack/include/prop.h                  FIRE_RES=1, COLD_RES=2,
                                                 DRAIN_RES=9

Artifact indices (wish.py _ARTIFACTS, 0-based):
    0  Excalibur    5  Sting    22  Frost Brand
    3  Mjollnir     6  Orcrist  23  Fire Brand
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    from Nethax.nethax.state import EnvState
    return EnvState.default(_RNG)


def _place_monster(state, entry_idx: int, monster_slot: int = 0):
    """Put a live monster (from MONSTERS[entry_idx]) into slot monster_slot
    adjacent to the player, with high HP so it survives many hits."""
    mai = state.monster_ai
    player_pos = state.player_pos
    mon_pos = jnp.array(
        [int(player_pos[0]), int(player_pos[1]) + 1], dtype=jnp.int16
    )
    mai = mai.replace(
        alive=mai.alive.at[monster_slot].set(True),
        hp=mai.hp.at[monster_slot].set(jnp.int32(100_000)),
        hp_max=mai.hp_max.at[monster_slot].set(jnp.int32(100_000)),
        pos=mai.pos.at[monster_slot].set(mon_pos),
        ac=mai.ac.at[monster_slot].set(jnp.int8(10)),
        entry_idx=mai.entry_idx.at[monster_slot].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


def _wield_artifact(state, artifact_idx: int, type_id: int):
    """Place a weapon with the given type_id + artifact_idx in slot 0 and wield it.

    Sets inventory.wielded_artifact_idx directly (bypasses handle_wield's
    action-dispatch signature which is fixed to (state, rng)).
    """
    from Nethax.nethax.subsystems.inventory import ItemCategory, wield
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_intrinsics

    items = state.inventory.items
    items = items.replace(
        category=items.category.at[0].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=items.type_id.at[0].set(jnp.int16(type_id)),
        enchantment=items.enchantment.at[0].set(jnp.int8(0)),
        quantity=items.quantity.at[0].set(jnp.int16(1)),
        weight=items.weight.at[0].set(jnp.int32(40)),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))
    # Wield slot 0.
    state = wield(state, 0)
    # Set the artifact idx directly (wield() doesn't know artifact identity).
    new_inv = state.inventory.replace(wielded_artifact_idx=jnp.int8(artifact_idx))
    state = state.replace(inventory=new_inv)
    # Apply intrinsics (JIT-safe).
    return apply_artifact_intrinsics(state)


def _first_undead_entry_idx() -> int:
    """Return the MONSTERS[] index of the first monster with M2_UNDEAD set."""
    from Nethax.nethax.constants.monsters import MONSTERS, M2_UNDEAD
    for i, m in enumerate(MONSTERS):
        if m.flags2 & M2_UNDEAD:
            return i
    pytest.skip("No undead monster found in MONSTERS table")


def _first_orc_entry_idx() -> int:
    """Return the MONSTERS[] index of the first monster with M2_ORC set."""
    from Nethax.nethax.constants.monsters import MONSTERS, M2_ORC
    for i, m in enumerate(MONSTERS):
        if m.flags2 & M2_ORC:
            return i
    pytest.skip("No orc monster found in MONSTERS table")


def _first_non_orc_entry_idx() -> int:
    """Return the MONSTERS[] index of the first non-orc, non-undead, non-demon monster."""
    from Nethax.nethax.constants.monsters import MONSTERS, M2_ORC, M2_UNDEAD, M2_DEMON
    for i, m in enumerate(MONSTERS):
        if not (m.flags2 & (M2_ORC | M2_UNDEAD | M2_DEMON)):
            return i
    pytest.skip("No non-orc non-undead monster found")


# ---------------------------------------------------------------------------
# Part A: damage bonus tests
# ---------------------------------------------------------------------------

def test_frost_brand_adds_cold_damage():
    """Frost Brand adds +1..6 cold damage on every hit against any target.

    Cite: artilist.h line 149-151 COLD(5,0); spec_dbon line 1091-1109;
          artifact_idx=22 in wish._ARTIFACTS.
    """
    from Nethax.nethax.subsystems.artifact_powers import artifact_bonus_damage

    # Type_id 37 = long sword; artifact_idx 22 = Frost Brand.
    # Target: entry_idx 0 (any — predicate is ALWAYS).
    rng = jax.random.PRNGKey(1)
    bonuses = set()
    for i in range(60):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        bonus = int(artifact_bonus_damage(jnp.int32(22), jnp.int32(0), sub))
        assert 1 <= bonus <= 6, f"Frost Brand bonus out of range: {bonus}"
        bonuses.add(bonus)
    # Over 60 rolls we should see at least 4 distinct values (d6 coverage).
    assert len(bonuses) >= 4, f"Expected d6 spread, got: {sorted(bonuses)}"


def test_excalibur_bonus_always_applies():
    """Excalibur adds +1..10 PHYS damage against any target.

    Vendor: Excalibur's spfx is (SPFX_NOGEN|SPFX_RESTR|SPFX_SEEK|SPFX_DEFN
    |SPFX_INTEL|SPFX_SEARCH) — none of {SPFX_DBONUS, SPFX_ATTK}, so the
    first branch of spec_applies returns ``weap->attk.adtyp == AD_PHYS``
    which is TRUE for any target.  The PHYS(5, 10) attack bonus
    therefore applies as +d10 against every monster.
    Cite: vendor/nethack/src/artifact.c::spec_applies lines 1014-1015;
          vendor/nethack/include/artilist.h:85-88 (PHYS(5, 10)).
    """
    from Nethax.nethax.subsystems.artifact_powers import artifact_bonus_damage

    undead_idx = _first_undead_entry_idx()
    non_undead_idx = _first_non_orc_entry_idx()
    rng = jax.random.PRNGKey(2)
    for target_idx, name in ((undead_idx, "undead"), (non_undead_idx, "non-undead")):
        bonuses = set()
        for i in range(60):
            sub = jax.random.fold_in(rng, jnp.uint32(i))
            bonus = int(artifact_bonus_damage(jnp.int32(0), jnp.int32(target_idx), sub))
            assert 1 <= bonus <= 10, (
                f"Excalibur vs {name} bonus out of d10 range: {bonus}"
            )
            bonuses.add(bonus)
        # d10 spread — at least 6 distinct outcomes over 60 rolls.
        assert len(bonuses) >= 6, (
            f"Expected d10 spread vs {name}, got: {sorted(bonuses)}"
        )


def test_sting_vs_orc():
    """Sting adds +1..5 damage against orc targets.

    Cite: artilist.h line 138-140 PHYS(5,0) SPFX_DFLAG2+M2_ORC;
          spec_applies SPFX_DFLAG2 path (artifact.c line 1026-1027);
          artifact_idx=5.
    """
    from Nethax.nethax.subsystems.artifact_powers import artifact_bonus_damage

    orc_idx = _first_orc_entry_idx()
    rng = jax.random.PRNGKey(4)
    bonuses = set()
    for i in range(60):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        bonus = int(artifact_bonus_damage(jnp.int32(5), jnp.int32(orc_idx), sub))
        assert 1 <= bonus <= 5, f"Sting vs orc bonus out of range: {bonus}"
        bonuses.add(bonus)
    assert len(bonuses) >= 3, f"Expected d5 spread, got: {sorted(bonuses)}"


def test_sting_vs_non_orc_no_bonus():
    """Sting gives no bonus against non-orc targets.

    Cite: spec_applies returns False when M2_ORC not set (artifact.c:1026).
    """
    from Nethax.nethax.subsystems.artifact_powers import artifact_bonus_damage

    non_orc_idx = _first_non_orc_entry_idx()
    rng = jax.random.PRNGKey(5)
    for i in range(20):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        bonus = int(artifact_bonus_damage(jnp.int32(5), jnp.int32(non_orc_idx), sub))
        assert bonus == 0, f"Sting should give 0 bonus vs non-orc, got {bonus}"


def test_no_artifact_no_bonus():
    """artifact_idx=-1 (bare hands or plain weapon) always returns 0."""
    from Nethax.nethax.subsystems.artifact_powers import artifact_bonus_damage

    rng = jax.random.PRNGKey(6)
    for i in range(20):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        bonus = int(artifact_bonus_damage(jnp.int32(-1), jnp.int32(0), sub))
        assert bonus == 0, f"No-artifact should give 0 bonus, got {bonus}"


# ---------------------------------------------------------------------------
# Part A: integration test through _single_melee_strike
# ---------------------------------------------------------------------------

def test_frost_brand_adds_cold_damage_via_melee():
    """Melee damage with Frost Brand exceeds base damage on average.

    Uses _single_melee_strike directly; verifies the artifact hook is wired.
    Frost Brand adds +1..6 to weapon damage so average total > base.
    """
    from Nethax.nethax.subsystems.combat import _single_melee_strike

    # Type_id 37 = long sword; artifact_idx 22 = Frost Brand.
    state = _fresh_state().replace(
        player_str=jnp.int16(18 + 100),
        player_dex=jnp.int8(18),
        player_xl=jnp.int32(10),
    )

    # Place a neutral monster (non-orc, non-undead) to isolate cold bonus.
    non_orc_idx = _first_non_orc_entry_idx()
    state = _place_monster(state, non_orc_idx, monster_slot=0)

    # Wield Frost Brand (artifact_idx=22, long sword type_id=37).
    state_frost = _wield_artifact(state, artifact_idx=22, type_id=37)
    # Plain long sword (no artifact).
    state_plain = _wield_artifact(state, artifact_idx=-1, type_id=37)

    rng = jax.random.PRNGKey(99)
    n = 80
    total_frost, total_plain = 0, 0
    for i in range(n):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        _, dmg_frost, hit_frost = _single_melee_strike(state_frost, sub, jnp.int32(0))
        _, dmg_plain, hit_plain = _single_melee_strike(state_plain, sub, jnp.int32(0))
        total_frost += int(dmg_frost)
        total_plain += int(dmg_plain)

    avg_frost = total_frost / n
    avg_plain = total_plain / n
    assert avg_frost > avg_plain, (
        f"Frost Brand avg dmg ({avg_frost:.2f}) should exceed plain long sword "
        f"({avg_plain:.2f})"
    )


# ---------------------------------------------------------------------------
# Part B: intrinsic grant tests
# ---------------------------------------------------------------------------

def test_frost_brand_grants_resist_cold():
    """Wielding Frost Brand sets RESIST_COLD intrinsic.

    Cite: artilist.h line 149-151 COLD(0,0) DFNS field (grants cold resistance
          while wielded); arti_prop artifact.c lines 880-885; COLD_RES=2 prop.h.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    assert not bool(state.status.intrinsics[int(Intrinsic.RESIST_COLD)]), \
        "RESIST_COLD should not be set before wielding"

    # Wield Frost Brand: artifact_idx=22, type_id=37 (long sword).
    state = _wield_artifact(state, artifact_idx=22, type_id=37)

    assert bool(state.status.intrinsics[int(Intrinsic.RESIST_COLD)]), \
        "RESIST_COLD should be set after wielding Frost Brand"


def test_fire_brand_grants_resist_fire():
    """Wielding Fire Brand sets RESIST_FIRE intrinsic.

    Cite: artilist.h line 153-155 FIRE(0,0) DFNS field; FIRE_RES=1 prop.h.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=23, type_id=37)

    assert bool(state.status.intrinsics[int(Intrinsic.RESIST_FIRE)]), \
        "RESIST_FIRE should be set after wielding Fire Brand"


def test_excalibur_grants_resist_drain():
    """Wielding Excalibur sets RESIST_DRAIN intrinsic.

    Cite: artilist.h line 85-88 DRLI(0,0) DFNS field; DRAIN_RES=9 prop.h.
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=0, type_id=37)

    assert bool(state.status.intrinsics[int(Intrinsic.RESIST_DRAIN)]), \
        "RESIST_DRAIN should be set after wielding Excalibur"


def test_unwield_frost_brand_clears_resist_cold():
    """Wielding then unwielding Frost Brand clears RESIST_COLD.

    Cite: artifact.c lines 880-885 (setworn W_ART clear path removes
          inv_prop extrinsic when weapon is unwielded).
    """
    from Nethax.nethax.subsystems.status_effects import Intrinsic
    from Nethax.nethax.subsystems.inventory import handle_wield

    state = _fresh_state()

    # Wield Frost Brand.
    state = _wield_artifact(state, artifact_idx=22, type_id=37)
    assert bool(state.status.intrinsics[int(Intrinsic.RESIST_COLD)]), \
        "RESIST_COLD should be set after wielding Frost Brand"

    # Unwield: place a plain (non-artifact) weapon in slot 1 and wield it.
    from Nethax.nethax.subsystems.inventory import ItemCategory
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[1].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=items.type_id.at[1].set(jnp.int16(14)),  # short sword
        quantity=items.quantity.at[1].set(jnp.int16(1)),
        weight=items.weight.at[1].set(jnp.int32(30)),
    )
    state = state.replace(inventory=state.inventory.replace(items=items))

    # Wield the plain weapon (artifact_idx=-1).
    # handle_wield finds the first weapon — that's still slot 0 (Frost Brand).
    # To force slot 1, directly call wield() then apply_artifact_intrinsics.
    from Nethax.nethax.subsystems.inventory import wield
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_intrinsics

    state = wield(state, 1)
    new_inv = state.inventory.replace(wielded_artifact_idx=jnp.int8(-1))
    state = state.replace(inventory=new_inv)
    state = apply_artifact_intrinsics(state)

    assert not bool(state.status.intrinsics[int(Intrinsic.RESIST_COLD)]), \
        "RESIST_COLD should be cleared after unwielding Frost Brand"


def test_frost_brand_does_not_grant_fire_resistance():
    """Frost Brand grants RESIST_COLD but not RESIST_FIRE."""
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=22, type_id=37)

    assert not bool(state.status.intrinsics[int(Intrinsic.RESIST_FIRE)]), \
        "Frost Brand must not grant RESIST_FIRE"
