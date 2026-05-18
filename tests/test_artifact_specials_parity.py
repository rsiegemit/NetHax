"""Artifact special-abilities parity tests.

Covers:
  1. Vorpal Blade beheading of lich (instant kill).
  2. Magicbane on-hit status effects (at least one status applied over trials).
  3. Eye of the Aethiopica #invoke grants +1 Pw.
  4. #invoke cooldown (second invoke in same turn is no-op).
  5. Excalibur wielded by non-lawful player deals 4d10 damage.

Canonical vendor references:
  vendor/nethack/src/artifact.c::artifact_hit lines 1220-1255 (Vorpal Blade)
  vendor/nethack/src/artifact.c::magicbane_hit lines 1090-1170 (Magicbane)
  vendor/nethack/src/artifact.c::arti_invoke line ~1480 (Eye / invoke dispatch)
  vendor/nethack/src/artifact.c::Wield_artifact_unaligned (Excalibur alignment)
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

_RNG = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fresh_state():
    from Nethax.nethax.state import EnvState
    return EnvState.default(_RNG)


def _place_monster(state, entry_idx: int, slot: int = 0, hp: int = 999_999):
    """Put a live monster adjacent to the player at the given slot."""
    mai = state.monster_ai
    mon_pos = jnp.array(
        [int(state.player_pos[0]), int(state.player_pos[1]) + 1], dtype=jnp.int16
    )
    mai = mai.replace(
        alive=mai.alive.at[slot].set(True),
        hp=mai.hp.at[slot].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[slot].set(jnp.int32(hp)),
        pos=mai.pos.at[slot].set(mon_pos),
        ac=mai.ac.at[slot].set(jnp.int8(10)),
        entry_idx=mai.entry_idx.at[slot].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


def _wield_artifact(state, artifact_idx: int, type_id: int = 37):
    """Set wielded_artifact_idx and place a matching weapon in slot 0."""
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
    state = wield(state, 0)
    new_inv = state.inventory.replace(wielded_artifact_idx=jnp.int8(artifact_idx))
    state = state.replace(inventory=new_inv)
    return apply_artifact_intrinsics(state)


def _first_lich_entry_idx() -> int:
    """Return the first MONSTERS[] index whose symbol == S_LICH."""
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
    for i, m in enumerate(MONSTERS):
        if m.symbol == MonsterSymbol.S_LICH:
            return i
    pytest.skip("No lich monster in MONSTERS table")


def _first_non_special_entry_idx() -> int:
    """Return a non-lich, non-undead, non-demon monster entry index."""
    from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol, M2_UNDEAD, M2_DEMON
    for i, m in enumerate(MONSTERS):
        if (m.symbol != MonsterSymbol.S_LICH
                and not (m.flags2 & M2_UNDEAD)
                and not (m.flags2 & M2_DEMON)):
            return i
    pytest.skip("No plain monster in MONSTERS table")


# ---------------------------------------------------------------------------
# 1. Vorpal Blade beheads lich instantly
# ---------------------------------------------------------------------------

def test_vorpal_blade_beheads_lich():
    """Vorpal Blade always kills a lich-class monster on hit.

    Cite: vendor/nethack/src/artifact.c::artifact_hit lines 1220-1255 —
    is_lich (S_LICH symbol) → instant kill regardless of rn2(23).
    artifact_idx=8 (Vorpal Blade, wish.py).
    """
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_hit_effects

    lich_idx = _first_lich_entry_idx()
    state = _fresh_state()
    state = _place_monster(state, lich_idx, slot=0, hp=999_999)
    state = _wield_artifact(state, artifact_idx=8)  # Vorpal Blade

    mon_slot = jnp.int32(0)
    rng = jax.random.PRNGKey(42)

    # Run several times — lich must always die.
    for i in range(10):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        new_state, killed = apply_artifact_hit_effects(state, mon_slot, sub)
        assert bool(killed), f"Lich should be killed by Vorpal Blade (trial {i})"
        assert not bool(new_state.monster_ai.alive[0]), (
            f"Monster slot 0 should be dead (trial {i})"
        )


def test_vorpal_blade_one_in_23_vs_non_lich():
    """Vorpal Blade kills a non-lich ~1/23 of the time.

    Over 230 trials we expect roughly 10 kills (1/23 ≈ 4.3%).
    Allow a generous range of 1-40 to avoid flakiness.
    Cite: artifact.c::artifact_hit line ~1240 — rn2(23)==0 for non-lich.
    """
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_hit_effects

    plain_idx = _first_non_special_entry_idx()
    state = _fresh_state()
    state = _place_monster(state, plain_idx, slot=0, hp=999_999)
    state = _wield_artifact(state, artifact_idx=8)

    mon_slot = jnp.int32(0)
    rng = jax.random.PRNGKey(99)
    kills = 0
    n = 230
    for i in range(n):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        # Reset hp each trial (state is reset each time).
        fresh = _place_monster(state, plain_idx, slot=0, hp=999_999)
        _, killed = apply_artifact_hit_effects(fresh, mon_slot, sub)
        kills += int(killed)

    assert 1 <= kills <= 40, (
        f"Expected ~10 Vorpal kills in {n} trials vs non-lich, got {kills}"
    )


# ---------------------------------------------------------------------------
# 2. Magicbane on-hit status effects
# ---------------------------------------------------------------------------

def test_magicbane_status_effect():
    """At least one Magicbane hit over many trials applies a status effect.

    Cite: vendor/nethack/src/artifact.c::magicbane_hit lines 1090-1170 —
    25% trigger (rn2(4)==0), then one of 4 effects chosen by rn2(4).
    artifact_idx=29 (synthetic Magicbane sentinel).
    """
    from Nethax.nethax.subsystems.artifact_powers import apply_artifact_hit_effects

    plain_idx = _first_non_special_entry_idx()
    state = _fresh_state()
    state = _place_monster(state, plain_idx, slot=0, hp=999_999)
    state = _wield_artifact(state, artifact_idx=29)  # Magicbane sentinel

    mon_slot = jnp.int32(0)
    rng = jax.random.PRNGKey(13)

    got_status = False
    for i in range(100):
        sub = jax.random.fold_in(rng, jnp.uint32(i))
        new_state, _ = apply_artifact_hit_effects(state, mon_slot, sub)
        mai = new_state.monster_ai
        # Status: either asleep or mstrategy changed to FLEE (4).
        asleep_changed = bool(mai.asleep[0]) and not bool(state.monster_ai.asleep[0])
        flee_changed = int(mai.mstrategy[0]) == 4 and int(state.monster_ai.mstrategy[0]) != 4
        if asleep_changed or flee_changed:
            got_status = True
            break

    assert got_status, (
        "Magicbane should apply at least one status effect in 100 trials"
    )


# ---------------------------------------------------------------------------
# 3. Eye of the Aethiopica #invoke grants +1 Pw
# ---------------------------------------------------------------------------

def test_invoke_eye_of_aethiopica_grants_pw():
    """#invoke Eye of the Aethiopica increases player_pw by 1.

    Cite: vendor/nethack/src/artifact.c::arti_invoke line ~1480 —
    Eye grants energy (Pw) when invoked.
    artifact_idx=21 (wish.py index 21).
    """
    from Nethax.nethax.subsystems.action_dispatch import _handle_invoke

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=21)  # Eye of the Aethiopica
    initial_pw = int(state.player_pw)

    new_state = _handle_invoke(state, _RNG)
    assert int(new_state.player_pw) == initial_pw + 1, (
        f"Eye invoke should add 1 Pw: {initial_pw} → {int(new_state.player_pw)}"
    )


# ---------------------------------------------------------------------------
# 4. #invoke cooldown — second invoke is a no-op
# ---------------------------------------------------------------------------

def test_invoke_cooldown():
    """The second consecutive #invoke is a no-op while cooldown > 0.

    After the first invoke, invoke_cooldown[21] is set to 100.
    The second invoke should not increment player_pw again.
    Cite: artifact.c::arti_invoke artiintrinsics_taught[] cooldown tracking.
    """
    from Nethax.nethax.subsystems.action_dispatch import _handle_invoke

    state = _fresh_state()
    state = _wield_artifact(state, artifact_idx=21)  # Eye of the Aethiopica

    # First invoke.
    state_after1 = _handle_invoke(state, _RNG)
    pw_after1 = int(state_after1.player_pw)
    cd_after1 = int(state_after1.invoke_cooldown[21])
    assert cd_after1 == 100, f"Cooldown should be 100 after first invoke, got {cd_after1}"

    # Second invoke — should be no-op.
    state_after2 = _handle_invoke(state_after1, _RNG)
    pw_after2 = int(state_after2.player_pw)
    assert pw_after2 == pw_after1, (
        f"Second invoke should be no-op (cooldown active): "
        f"pw went from {pw_after1} to {pw_after2}"
    )


# ---------------------------------------------------------------------------
# 5. Excalibur wielded by non-lawful player deals damage
# ---------------------------------------------------------------------------

def test_excalibur_wielded_by_non_lawful_damages():
    """Wielding Excalibur as a Chaotic player deals 4d10 damage.

    Cite: vendor/nethack/src/artifact.c::Wield_artifact_unaligned —
    non-lawful wielder of Excalibur takes 4d10 damage.
    player_align=0 = CHAOTIC (prayer.py Alignment enum).
    artifact_idx=0 (Excalibur, wish.py).
    """
    from Nethax.nethax.subsystems.artifact_powers import check_excalibur_alignment

    state = _fresh_state()
    state = state.replace(
        player_align=jnp.int8(0),   # CHAOTIC
        player_hp=jnp.int32(200),
        player_hp_max=jnp.int32(200),
    )
    state = _wield_artifact(state, artifact_idx=0)  # Excalibur

    new_state = check_excalibur_alignment(state, _RNG)

    assert int(new_state.player_hp) < 200, (
        f"Non-lawful Excalibur wielder should take damage, hp={int(new_state.player_hp)}"
    )
    # 4d10 minimum is 4, maximum is 40.
    damage_taken = 200 - int(new_state.player_hp)
    assert 4 <= damage_taken <= 40, (
        f"Expected 4-40 damage (4d10), got {damage_taken}"
    )


def test_excalibur_wielded_by_lawful_no_damage():
    """Lawful player wielding Excalibur takes no damage.

    Cite: artifact.c::Wield_artifact_unaligned — only fires when unaligned.
    player_align=2 = LAWFUL (prayer.py).
    """
    from Nethax.nethax.subsystems.artifact_powers import check_excalibur_alignment

    state = _fresh_state()
    state = state.replace(
        player_align=jnp.int8(2),   # LAWFUL
        player_hp=jnp.int32(200),
        player_hp_max=jnp.int32(200),
    )
    state = _wield_artifact(state, artifact_idx=0)  # Excalibur

    new_state = check_excalibur_alignment(state, _RNG)
    assert int(new_state.player_hp) == 200, (
        f"Lawful Excalibur wielder should take no damage, hp={int(new_state.player_hp)}"
    )
