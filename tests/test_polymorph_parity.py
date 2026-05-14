"""Wave 6 Phase B+ — polymorph attack-set parity tests.

Verifies that polymorph_player(target_form) copies the attack-set EXACTLY
from MONSTERS[target_form].attacks (NATTK=6 slots; unused slots zero-padded).

Sweeps ~20 sample forms across vendor monster classes (S_DOG, S_DRAGON,
S_HUMAN, S_LICH, S_DEMON, etc.).

Cite: vendor/nethack/src/polyself.c::polymon  (attack-set assignment)
      vendor/nethack/include/permonst.h::NATTK = 6
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.polymorph import (
    polymorph_player,
    revert_polymorph,
    NATTK,
)
from Nethax.nethax.constants.monsters import (
    MONSTERS, MonsterSymbol, AttackType, DamageType, NO_ATTK,
)


_RNG = jax.random.PRNGKey(11)


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
        player_ac=jnp.int32(10),
    )


def _find_entry_by_name(name: str) -> int:
    """Return the first MONSTERS index whose .name == ``name`` (exact match)."""
    for i, m in enumerate(MONSTERS):
        if m.name == name:
            return i
    raise ValueError(f"No monster named {name!r}")


def _padded_attacks(entry_idx: int):
    """Return tuple of 6 attack tuples (padded with NO_ATTK)."""
    m = MONSTERS[entry_idx]
    attks = list(m.attacks) + [NO_ATTK] * (NATTK - len(m.attacks))
    return tuple(attks[:NATTK])


def _assert_attacks_match(poly_state, entry_idx: int):
    """Assert that polymorph.attack_* arrays match MONSTERS[entry_idx].attacks."""
    expected = _padded_attacks(entry_idx)
    for i, (atyp, adtyp, ndice, nsides) in enumerate(expected):
        assert int(poly_state.attack_types[i]) == int(atyp), (
            f"attack_types[{i}] mismatch: got {int(poly_state.attack_types[i])} "
            f"expected {int(atyp)} (form {MONSTERS[entry_idx].name!r})"
        )
        assert int(poly_state.attack_damage_types[i]) == int(adtyp), (
            f"attack_damage_types[{i}] mismatch: got {int(poly_state.attack_damage_types[i])} "
            f"expected {int(adtyp)} (form {MONSTERS[entry_idx].name!r})"
        )
        assert int(poly_state.attack_n_dice[i]) == int(ndice), (
            f"attack_n_dice[{i}] mismatch: got {int(poly_state.attack_n_dice[i])} "
            f"expected {int(ndice)} (form {MONSTERS[entry_idx].name!r})"
        )
        assert int(poly_state.attack_n_sides[i]) == int(nsides), (
            f"attack_n_sides[{i}] mismatch: got {int(poly_state.attack_n_sides[i])} "
            f"expected {int(nsides)} (form {MONSTERS[entry_idx].name!r})"
        )


# ---------------------------------------------------------------------------
# Per-form parity tests
# ---------------------------------------------------------------------------

def test_polymorph_into_dragon_copies_breath_attack():
    """Red dragon has AT_BREA / AD_FIRE as primary attack."""
    target = _find_entry_by_name("red dragon")
    state = _base_state()
    new = polymorph_player(state, _RNG, target, controlled=False)
    # First attack: AT_BREA = 12, AD_FIRE = 2.
    assert int(new.polymorph.attack_types[0]) == int(AttackType.AT_BREA)
    assert int(new.polymorph.attack_damage_types[0]) == int(DamageType.AD_FIRE)
    _assert_attacks_match(new.polymorph, target)


def test_polymorph_into_cockatrice_copies_petrify_attack():
    """Cockatrice has AT_TUCH / AD_STON as second attack."""
    target = _find_entry_by_name("cockatrice")
    state = _base_state()
    new = polymorph_player(state, _RNG, target, controlled=False)
    # cockatrice attacks: (BITE, PHYS, 1, 3), (TUCH, STON, 0, 0), (NONE, STON, 0, 0)
    assert int(new.polymorph.attack_types[0]) == int(AttackType.AT_BITE)
    assert int(new.polymorph.attack_types[1]) == int(AttackType.AT_TUCH)
    assert int(new.polymorph.attack_damage_types[1]) == int(DamageType.AD_STON)
    _assert_attacks_match(new.polymorph, target)


def test_polymorph_into_human_copies_punch_attack():
    """Human has AT_WEAP / AD_PHYS as primary attack."""
    target = _find_entry_by_name("human")
    state = _base_state()
    new = polymorph_player(state, _RNG, target, controlled=False)
    # Human attacks: (WEAP, PHYS, 1, 6) then NO_ATTK fill.
    assert int(new.polymorph.attack_types[0]) == int(AttackType.AT_WEAP)
    assert int(new.polymorph.attack_damage_types[0]) == int(DamageType.AD_PHYS)
    assert int(new.polymorph.attack_n_sides[0]) == 6
    _assert_attacks_match(new.polymorph, target)


def test_polymorph_into_acid_blob_copies_acid_attack():
    """Acid blob has AT_NONE / AD_ACID passive attack."""
    target = _find_entry_by_name("acid blob")
    state = _base_state()
    new = polymorph_player(state, _RNG, target, controlled=False)
    # acid blob: (AT_NONE, AD_ACID, 1, 8)
    assert int(new.polymorph.attack_types[0]) == int(AttackType.AT_NONE)
    assert int(new.polymorph.attack_damage_types[0]) == int(DamageType.AD_ACID)
    assert int(new.polymorph.attack_n_dice[0]) == 1
    assert int(new.polymorph.attack_n_sides[0]) == 8
    _assert_attacks_match(new.polymorph, target)


def test_polymorph_into_lich_copies_magic_attack():
    """Lich uses AT_MAGC / AD_SPEL as second attack."""
    target = _find_entry_by_name("lich")
    state = _base_state()
    new = polymorph_player(state, _RNG, target, controlled=False)
    assert int(new.polymorph.attack_types[1]) == int(AttackType.AT_MAGC)
    _assert_attacks_match(new.polymorph, target)


def test_polymorph_attack_dice_match_vendor_table():
    """For 5+ specific forms, every (aatyp, adtyp, damn, damd) tuple matches
    vendor MONSTERS exactly."""
    sample_names = [
        "red dragon", "black dragon", "cockatrice", "human", "acid blob",
        "kitten", "lich", "vampire", "gnome",
    ]
    for name in sample_names:
        target = _find_entry_by_name(name)
        state = _base_state()
        new = polymorph_player(state, _RNG, target, controlled=False)
        _assert_attacks_match(new.polymorph, target)


def test_polymorph_resets_attacks_on_revert():
    """revert_polymorph restores the saved orig_attack_* arrays."""
    state = _base_state()
    target = _find_entry_by_name("red dragon")
    new = polymorph_player(state, _RNG, target, controlled=False)
    assert int(new.polymorph.attack_types[0]) == int(AttackType.AT_BREA)
    reverted = revert_polymorph(new, _RNG)
    # After revert, attack arrays should match the *original* (zero-filled)
    # baseline that polymorph_player saved as orig_*.
    for i in range(NATTK):
        assert int(reverted.polymorph.attack_types[i]) == int(
            new.polymorph.orig_attack_types[i]
        )
        assert int(reverted.polymorph.attack_damage_types[i]) == int(
            new.polymorph.orig_attack_damage_types[i]
        )
        assert int(reverted.polymorph.attack_n_dice[i]) == int(
            new.polymorph.orig_attack_n_dice[i]
        )
        assert int(reverted.polymorph.attack_n_sides[i]) == int(
            new.polymorph.orig_attack_n_sides[i]
        )


# ---------------------------------------------------------------------------
# Broad sweep — 20 sample forms across vendor classes
# ---------------------------------------------------------------------------

def _find_first_by_symbol(symbol: MonsterSymbol) -> int:
    """Return the first MONSTERS index whose symbol matches ``symbol``."""
    for i, m in enumerate(MONSTERS):
        if int(m.symbol) == int(symbol):
            return i
    return -1


def test_polymorph_into_20_sample_forms_attack_set_parity():
    """Sweep one form from each of 20+ monster classes and verify attack-set
    byte-for-byte parity with MONSTERS[idx].attacks."""
    sample_classes = [
        MonsterSymbol.S_ANT, MonsterSymbol.S_BLOB, MonsterSymbol.S_COCKATRICE,
        MonsterSymbol.S_DOG, MonsterSymbol.S_EYE, MonsterSymbol.S_FELINE,
        MonsterSymbol.S_HUMANOID, MonsterSymbol.S_KOBOLD, MonsterSymbol.S_MIMIC,
        MonsterSymbol.S_NYMPH, MonsterSymbol.S_ORC, MonsterSymbol.S_RODENT,
        MonsterSymbol.S_DRAGON, MonsterSymbol.S_GNOME, MonsterSymbol.S_LICH,
        MonsterSymbol.S_MUMMY, MonsterSymbol.S_TROLL, MonsterSymbol.S_VAMPIRE,
        MonsterSymbol.S_ZOMBIE, MonsterSymbol.S_HUMAN, MonsterSymbol.S_DEMON,
    ]
    covered = 0
    for sym in sample_classes:
        target = _find_first_by_symbol(sym)
        if target < 0:
            continue
        state = _base_state()
        new = polymorph_player(state, _RNG, target, controlled=False)
        _assert_attacks_match(new.polymorph, target)
        covered += 1
    # We must hit at least 20 forms.
    assert covered >= 20, f"only {covered} sample forms covered (need >=20)"


def test_polymorph_attack_set_size_is_nattk():
    """Sanity: the polymorph state always holds exactly NATTK=6 attack slots."""
    state = _base_state()
    target = _find_entry_by_name("red dragon")
    new = polymorph_player(state, _RNG, target, controlled=False)
    assert int(new.polymorph.attack_types.shape[0]) == NATTK
    assert int(new.polymorph.attack_damage_types.shape[0]) == NATTK
    assert int(new.polymorph.attack_n_dice.shape[0]) == NATTK
    assert int(new.polymorph.attack_n_sides.shape[0]) == NATTK
