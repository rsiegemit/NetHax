"""Wave 6 Phase B — role-specific combat bonus tests.

Covers:
  * Monk martial-arts bonus damage when bare-handed (vendor/nethack/src/
    uhitm.c::hmon_hitmon_barehands ~847 and mon_arms_table XL scaling).
  * Samurai bushido weapon bonus for katana / wakizashi / yumi
    (vendor/nethack/src/uhitm.c ~969, ~1051).
  * Knight chivalric to-hit bonus vs humanoid opponents
    (vendor/nethack/src/uhitm.c::check_caitiff invoked from
    find_roll_to_hit).

These bonuses are wired into ``_single_melee_strike`` via
``_monk_martial_arts_bonus``, ``_samurai_bushido_bonus``, and
``_knight_chivalric_bonus`` in ``combat.py``.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

from Nethax.nethax.constants.monsters import MONSTERS, MonsterSymbol
from Nethax.nethax.constants.roles import Role
from Nethax.nethax.state import EnvState
from Nethax.nethax.subsystems.combat import (
    WEAPON_TYPE_KATANA,
    WEAPON_TYPE_YUMI,
    _knight_chivalric_bonus,
    _monk_martial_arts_bonus,
    _samurai_bushido_bonus,
    melee_attack,
)
from Nethax.nethax.subsystems.inventory import ItemCategory


_RNG = jax.random.PRNGKey(2026)


def _fresh_state() -> EnvState:
    return EnvState.default(_RNG)


def _first_monster_with_symbol(symbol: MonsterSymbol) -> int:
    for i, entry in enumerate(MONSTERS):
        if entry.symbol == symbol:
            return i
    raise AssertionError(f"no monster with symbol {symbol}")


def _spawn_monster(state, entry_idx: int, pos=(5, 6), hp: int = 100, ac: int = 10):
    """Place a live monster at slot 0 with the given entry_idx."""
    mai = state.monster_ai
    mai = mai.replace(
        alive=mai.alive.at[0].set(True),
        hp=mai.hp.at[0].set(jnp.int32(hp)),
        hp_max=mai.hp_max.at[0].set(jnp.int32(hp)),
        pos=mai.pos.at[0].set(jnp.array(pos, dtype=jnp.int16)),
        ac=mai.ac.at[0].set(jnp.int8(ac)),
        entry_idx=mai.entry_idx.at[0].set(jnp.int16(entry_idx)),
    )
    return state.replace(monster_ai=mai)


def _wield_weapon(state, type_id: int, slot: int = 0):
    """Place a weapon with ``type_id`` in inventory slot ``slot`` and wield it."""
    items = state.inventory.items
    items = items.replace(
        category=items.category.at[slot].set(jnp.int8(ItemCategory.WEAPON)),
        type_id=items.type_id.at[slot].set(jnp.int16(type_id)),
        quantity=items.quantity.at[slot].set(jnp.int16(1)),
    )
    inv = state.inventory.replace(items=items, wielded=jnp.int8(slot))
    return state.replace(inventory=inv)


# ---------------------------------------------------------------------------
# Monk martial arts
# ---------------------------------------------------------------------------

def test_monk_bare_handed_xl5_extra_damage():
    """Bare-handed XL5 monk samples a strictly positive bonus on most rolls."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.MONK)),
        player_xl=jnp.int32(5),
    )
    keys = jax.random.split(jax.random.PRNGKey(11), 32)
    samples = jnp.array(
        [int(_monk_martial_arts_bonus(state, k)) for k in keys]
    )
    # At XL5 the formula rolls 2d4 → bonus is always >= 2.
    assert int(samples.min()) >= 2
    assert float(samples.mean()) > 3.0


def test_monk_with_weapon_no_bonus():
    """A monk wielding a katana gets no martial-arts bonus."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.MONK)),
        player_xl=jnp.int32(5),
    )
    state = _wield_weapon(state, WEAPON_TYPE_KATANA)
    keys = jax.random.split(jax.random.PRNGKey(13), 8)
    for k in keys:
        assert int(_monk_martial_arts_bonus(state, k)) == 0


def test_monk_xl9_double_die_bonus():
    """At XL9 the monk rolls more dice than at XL5 on average."""
    state_xl5 = _fresh_state().replace(
        player_role=jnp.int8(int(Role.MONK)),
        player_xl=jnp.int32(5),
    )
    state_xl9 = state_xl5.replace(player_xl=jnp.int32(9))

    keys = jax.random.split(jax.random.PRNGKey(17), 80)
    xl5 = jnp.array([int(_monk_martial_arts_bonus(state_xl5, k)) for k in keys])
    xl9 = jnp.array([int(_monk_martial_arts_bonus(state_xl9, k)) for k in keys])

    # XL5 rolls 2d4 (mean 5); XL9 rolls 3d4 (mean 7.5).  Use a comfortable gap
    # to avoid false negatives from sampling variance.
    assert float(xl9.mean()) > float(xl5.mean()) + 1.0
    # Minimum for XL9 is 3 dice → at least 3 damage.
    assert int(xl9.min()) >= 3


def test_non_monk_bare_handed_no_bonus():
    """A bare-handed Valkyrie sees zero martial-arts bonus."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.VALKYRIE)),
        player_xl=jnp.int32(9),
    )
    keys = jax.random.split(jax.random.PRNGKey(19), 8)
    for k in keys:
        assert int(_monk_martial_arts_bonus(state, k)) == 0


# ---------------------------------------------------------------------------
# Samurai bushido
# ---------------------------------------------------------------------------

def test_samurai_katana_dam_bonus():
    """A samurai wielding a katana rolls a positive 1d6 bonus."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.SAMURAI)),
    )
    state = _wield_weapon(state, WEAPON_TYPE_KATANA)
    keys = jax.random.split(jax.random.PRNGKey(23), 40)
    samples = jnp.array(
        [int(_samurai_bushido_bonus(state, k)) for k in keys]
    )
    # 1d6 → bonus is in [1, 6] and averages ~3.5.
    assert int(samples.min()) >= 1
    assert int(samples.max()) <= 6
    assert 2.5 <= float(samples.mean()) <= 4.5


def test_samurai_yumi_dam_bonus():
    """A samurai wielding a yumi rolls a positive 1d4 bonus (wakizashi/yumi)."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.SAMURAI)),
    )
    state = _wield_weapon(state, WEAPON_TYPE_YUMI)
    keys = jax.random.split(jax.random.PRNGKey(29), 40)
    samples = jnp.array(
        [int(_samurai_bushido_bonus(state, k)) for k in keys]
    )
    assert int(samples.min()) >= 1
    assert int(samples.max()) <= 4


def test_samurai_no_katana_no_bonus():
    """A samurai wielding a non-cultural weapon (e.g. long sword id 19) → 0."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.SAMURAI)),
    )
    state = _wield_weapon(state, type_id=19)  # long sword
    keys = jax.random.split(jax.random.PRNGKey(31), 8)
    for k in keys:
        assert int(_samurai_bushido_bonus(state, k)) == 0


def test_non_samurai_katana_no_bonus():
    """A non-samurai (Knight) wielding a katana sees no bushido bonus."""
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.KNIGHT)),
    )
    state = _wield_weapon(state, WEAPON_TYPE_KATANA)
    keys = jax.random.split(jax.random.PRNGKey(37), 8)
    for k in keys:
        assert int(_samurai_bushido_bonus(state, k)) == 0


# ---------------------------------------------------------------------------
# Knight chivalric morale
# ---------------------------------------------------------------------------

def test_knight_ac_bonus_vs_humanoid():
    """A Knight engaging a humanoid (hobbit) earns the chivalric to-hit bonus."""
    hobbit_idx = _first_monster_with_symbol(MonsterSymbol.S_HUMANOID)
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.KNIGHT)),
    )
    state = _spawn_monster(state, entry_idx=hobbit_idx)
    bonus = int(_knight_chivalric_bonus(state, jnp.int32(0)))
    assert bonus == 1


def test_knight_no_bonus_vs_dragon():
    """A Knight vs a dragon sees no chivalric bonus."""
    dragon_idx = _first_monster_with_symbol(MonsterSymbol.S_DRAGON)
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.KNIGHT)),
    )
    state = _spawn_monster(state, entry_idx=dragon_idx)
    bonus = int(_knight_chivalric_bonus(state, jnp.int32(0)))
    assert bonus == 0


def test_non_knight_no_chivalric_bonus():
    """A Valkyrie vs a humanoid sees no chivalric bonus."""
    hobbit_idx = _first_monster_with_symbol(MonsterSymbol.S_HUMANOID)
    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.VALKYRIE)),
    )
    state = _spawn_monster(state, entry_idx=hobbit_idx)
    bonus = int(_knight_chivalric_bonus(state, jnp.int32(0)))
    assert bonus == 0


# ---------------------------------------------------------------------------
# End-to-end: damage averages with vs. without role bonuses
# ---------------------------------------------------------------------------

@pytest.mark.timeout(900)
def test_monk_does_more_total_damage_than_non_monk_bare_handed():
    """Across many melee strikes, a bare-handed XL5 Monk out-damages a Tourist
    using the same stats and weapon (none).

    Timeout bumped to 900s: the first cold compile of melee_attack — which
    pulls in the full Role / monk-martial-arts / samurai / knight / artifact
    pipeline — can run past the default in heavily-loaded CI environments.
    Cite: vendor/nethack/src/uhitm.c::hmon_hitmon_barehands.
    """
    hobbit_idx = _first_monster_with_symbol(MonsterSymbol.S_HUMANOID)
    base = _fresh_state().replace(
        player_str=jnp.int16(18),
        player_dex=jnp.int8(14),
        player_xl=jnp.int32(5),
        player_pos=jnp.array([5, 5], dtype=jnp.int16),
    )

    def _total_damage(role):
        state = base.replace(player_role=jnp.int8(int(role)))
        state = _spawn_monster(state, entry_idx=hobbit_idx, hp=10_000)
        rng = jax.random.PRNGKey(101)
        total = 0
        cur = state
        for _ in range(60):
            rng, sub = jax.random.split(rng)
            cur, dmg, _hit = melee_attack(cur, sub, jnp.int32(0))
            total += int(dmg)
        return total

    monk_dmg = _total_damage(Role.MONK)
    tourist_dmg = _total_damage(Role.TOURIST)
    assert monk_dmg > tourist_dmg, (
        f"expected monk damage to exceed tourist; monk={monk_dmg}, "
        f"tourist={tourist_dmg}"
    )
