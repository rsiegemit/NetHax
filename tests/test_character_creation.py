"""Wave 3 integration tests — character creation by role.

Tests verify that each of the 13 roles initializes with the correct starting
equipment and known spells.

Wave 3 character-creation logic is implemented by the character agent.  Each
role-specific test is guarded with skipif when the feature is still a stub
(all roles produce the same zeroed default state in Wave 1/2).

All imports are lazy so collection never fails.
"""

import pytest


def _reset_with_role(role_value):
    """Helper: reset env with a specific role, return (state, rng)."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.env import NethaxEnv

    rng = jax.random.PRNGKey(role_value + 100)
    env = NethaxEnv()
    # Wave 3: reset will accept role kwarg; for now we inject via state.replace
    state, _ = env.reset(rng)
    state = state.replace(player_role=jnp.int8(role_value))
    return state, rng, env


def _role_produces_valid_state(role_value):
    """Return True if reset with role produces a structurally valid state."""
    import jax
    from Nethax.nethax.state import EnvState

    state, _rng, _env = _reset_with_role(role_value)
    if not isinstance(state, EnvState):
        return False
    leaves = jax.tree.leaves(state)
    return len(leaves) > 0 and all(isinstance(l, jax.Array) for l in leaves)


def test_all_roles_construct():
    """reset() for each of the 13 roles produces a valid EnvState pytree.

    This test does NOT require Wave 3 character-creation logic; it only
    verifies that the state is structurally valid (correct pytree shape,
    all-array leaves) for every role value.
    """
    from Nethax.nethax.constants.roles import N_ROLES

    failures = []
    for role_val in range(N_ROLES):
        try:
            ok = _role_produces_valid_state(role_val)
            if not ok:
                failures.append(f"role={role_val}: invalid state structure")
        except Exception as exc:
            failures.append(f"role={role_val}: exception {exc}")

    assert not failures, "Some roles failed to construct:\n" + "\n".join(failures)


def test_valkyrie_init():
    """reset(role=VALKYRIE) -> canonical starting kit (spear/dagger/shield)."""
    import jax
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    from Nethax.nethax.subsystems.inventory import ArmorSlot, ItemCategory
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    rng = jax.random.PRNGKey(int(Role.VALKYRIE) + 100)
    state, _ = env.reset(rng, role=Role.VALKYRIE, race=Race.HUMAN)

    # Player role should be VALKYRIE
    assert int(state.player_role) == int(Role.VALKYRIE), (
        f"Expected role=VALKYRIE ({int(Role.VALKYRIE)}), got {int(state.player_role)}"
    )

    # Wielded weapon: spear (canonical Valkyrie primary weapon)
    wielded_slot = int(state.inventory.wielded)
    assert wielded_slot != -1, (
        "Valkyrie should have a weapon wielded at start, got bare hands (-1)"
    )

    # Worn shield: small shield (ArmorSlot.SHIELD should be occupied)
    shield_slot = int(state.inventory.worn_armor[int(ArmorSlot.SHIELD)])
    assert shield_slot != -1, (
        "Valkyrie should be wearing a small shield at start (ArmorSlot.SHIELD not -1)"
    )

    # AC bonus from small shield >= 1
    cat = int(state.inventory.items.category[shield_slot])
    assert cat == int(ItemCategory.ARMOR), (
        f"Worn-shield slot should hold ARMOR, got category={cat}"
    )

    # Dagger should be in inventory: at least one slot has category=WEAPON.
    weapon_slots = [
        i for i in range(10)
        if int(state.inventory.items.category[i]) == int(ItemCategory.WEAPON)
    ]
    assert len(weapon_slots) >= 2, (
        "Valkyrie should have at least 2 weapons (spear + dagger), got "
        f"{len(weapon_slots)}"
    )


def test_wizard_init():
    """reset(role=WIZARD) -> quarterstaff wielded, force bolt spell known."""
    import jax
    import jax.numpy as jnp
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    from Nethax.nethax.subsystems.magic import SpellId
    from Nethax.nethax.env import NethaxEnv

    env = NethaxEnv()
    rng = jax.random.PRNGKey(int(Role.WIZARD) + 100)
    state, _ = env.reset(rng, role=Role.WIZARD, race=Race.HUMAN)

    assert int(state.player_role) == int(Role.WIZARD), (
        f"Expected role=WIZARD ({int(Role.WIZARD)}), got {int(state.player_role)}"
    )

    # Wielded weapon: quarterstaff (should not be bare-hand)
    wielded_slot = int(state.inventory.wielded)
    assert wielded_slot != -1, (
        "Wizard should have a quarterstaff wielded at start"
    )

    # Force bolt should be memorised in magic.spell_known and magic.spell_memory.
    spell_known = state.magic.spell_known
    assert bool(spell_known[int(SpellId.FORCE_BOLT)]), (
        "Wizard should know FORCE_BOLT at character creation"
    )
    assert bool(jnp.any(spell_known)), (
        "Wizard should know at least one spell at start (force bolt)"
    )
