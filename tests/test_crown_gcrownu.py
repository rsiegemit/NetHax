"""Tests for gcrownu — the crowning event (priest.py).

Vendor source: vendor/nethack/src/pray.c::gcrownu (lines 805-996).
Effects: SEE_INVIS + 5 resistances FROMOUTSIDE, role/alignment artifact gift.
"""

import pytest


def _fresh_state():
    import jax
    from Nethax.nethax.state import EnvState
    return EnvState.default(jax.random.PRNGKey(7))


# Vendor pray.c lines 813-818 — six intrinsics granted by gcrownu.
def test_gcrownu_grants_six_intrinsics():
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.subsystems.status_effects import Intrinsic

    state = _fresh_state()
    # Default role/align (0,0) — table fallback to STRANGE_OBJECT.
    new_state = gcrownu(state, jax.random.PRNGKey(11))
    for intr in (
        Intrinsic.RESIST_FIRE,
        Intrinsic.RESIST_COLD,
        Intrinsic.RESIST_SHOCK,
        Intrinsic.RESIST_SLEEP,
        Intrinsic.RESIST_POISON,
        Intrinsic.SEE_INVIS,
    ):
        assert bool(new_state.status.intrinsics[int(intr)]), (
            f"intrinsic {intr.name} not set"
        )


def test_gcrownu_knight_lawful_gets_excalibur():
    """Knight × Lawful → Excalibur (artifact_idx 0).

    Vendor pray.c lines 838-845 (A_LAWFUL → Excalibur via oname(...,
    ART_EXCALIBUR, ...) at line 907).
    """
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    state = _fresh_state()
    state = state.replace(
        player_role=jnp.int8(int(Role.KNIGHT)),
        player_align=jnp.int8(int(Alignment.LAWFUL)),
    )
    new_state = gcrownu(state, jax.random.PRNGKey(12))
    items = new_state.inventory.items
    assert int(items.artifact_idx[0]) == 0, (
        f"expected artifact_idx 0 (Excalibur), got {int(items.artifact_idx[0])}"
    )
    assert int(items.buc_status[0]) == 3  # BLESSED (pray.c:978 bless(obj))
    assert int(items.enchantment[0]) == 1  # pray.c:983 spe = 1
    assert bool(items.oerodeproof[0])      # pray.c:980 oerodeproof=TRUE


def test_gcrownu_wizard_gets_magicbane():
    """Wizard → Magicbane (artifact_idx 29 per wish._ARTIFACTS)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    state = _fresh_state()
    state = state.replace(
        player_role=jnp.int8(int(Role.WIZARD)),
        player_align=jnp.int8(int(Alignment.NEUTRAL)),
    )
    new_state = gcrownu(state, jax.random.PRNGKey(13))
    items = new_state.inventory.items
    assert int(items.artifact_idx[0]) == 29


def test_gcrownu_valkyrie_lawful_gets_mjollnir():
    """Valkyrie × Lawful → Mjollnir (artifact_idx 3)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    state = _fresh_state()
    state = state.replace(
        player_role=jnp.int8(int(Role.VALKYRIE)),
        player_align=jnp.int8(int(Alignment.LAWFUL)),
    )
    new_state = gcrownu(state, jax.random.PRNGKey(14))
    items = new_state.inventory.items
    assert int(items.artifact_idx[0]) == 3  # Mjollnir


def test_gcrownu_valkyrie_chaotic_gets_frost_brand():
    """Valkyrie × Chaotic → Frost Brand (artifact_idx 22)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    state = _fresh_state()
    state = state.replace(
        player_role=jnp.int8(int(Role.VALKYRIE)),
        player_align=jnp.int8(int(Alignment.CHAOTIC)),
    )
    new_state = gcrownu(state, jax.random.PRNGKey(15))
    items = new_state.inventory.items
    assert int(items.artifact_idx[0]) == 22  # Frost Brand


def test_gcrownu_priest_neutral_gets_sceptre():
    """Priest × Neutral → Sceptre of Might (artifact_idx 9)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    state = _fresh_state()
    state = state.replace(
        player_role=jnp.int8(int(Role.PRIEST)),
        player_align=jnp.int8(int(Alignment.NEUTRAL)),
    )
    new_state = gcrownu(state, jax.random.PRNGKey(16))
    items = new_state.inventory.items
    assert int(items.artifact_idx[0]) == 9


def test_gcrownu_priest_lawful_or_chaotic_gets_mjollnir():
    """Priest × {Lawful,Chaotic} → Mjollnir (artifact_idx 3)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    for align in (Alignment.LAWFUL, Alignment.CHAOTIC):
        state = _fresh_state().replace(
            player_role=jnp.int8(int(Role.PRIEST)),
            player_align=jnp.int8(int(align)),
        )
        new_state = gcrownu(state, jax.random.PRNGKey(17))
        items = new_state.inventory.items
        assert int(items.artifact_idx[0]) == 3, f"align={align.name}"


def test_gcrownu_samurai_gets_snickersnee():
    """Samurai → Snickersnee (artifact_idx 1)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    state = _fresh_state().replace(
        player_role=jnp.int8(int(Role.SAMURAI)),
        player_align=jnp.int8(int(Alignment.LAWFUL)),
    )
    new_state = gcrownu(state, jax.random.PRNGKey(18))
    items = new_state.inventory.items
    assert int(items.artifact_idx[0]) == 1


def test_gcrownu_other_role_artifacts():
    """Spot-check Barbarian/Caveman/Archeologist/Healer/Monk/Ranger/Rogue/Tourist."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.prayer import Alignment

    cases = [
        (Role.BARBARIAN,    Alignment.NEUTRAL, 4),   # Cleaver
        (Role.CAVEMAN,      Alignment.LAWFUL,  9),   # Sceptre of Might
        (Role.ARCHEOLOGIST, Alignment.NEUTRAL, 11),  # Magic Mirror of Merlin
        (Role.HEALER,       Alignment.NEUTRAL, 14),  # Staff of Aesculapius
        (Role.MONK,         Alignment.NEUTRAL, 15),  # Eyes of the Overworld
        (Role.RANGER,       Alignment.CHAOTIC, 17),  # Longbow of Diana
        (Role.ROGUE,        Alignment.CHAOTIC, 18),  # Master Key of Thievery
        (Role.TOURIST,      Alignment.NEUTRAL, 19),  # Yendorian Express Card
    ]
    for role, align, expected_idx in cases:
        state = _fresh_state().replace(
            player_role=jnp.int8(int(role)),
            player_align=jnp.int8(int(align)),
        )
        new_state = gcrownu(state, jax.random.PRNGKey(role.value + 100))
        got = int(new_state.inventory.items.artifact_idx[0])
        assert got == expected_idx, (
            f"role={role.name}/align={align.name}: expected artifact_idx "
            f"{expected_idx}, got {got}"
        )


def test_gcrownu_jit_safe():
    """gcrownu must be jax.jit-compatible (Threefry RNG, no host branches)."""
    import jax, jax.numpy as jnp
    from Nethax.nethax.subsystems.priest import gcrownu

    state = _fresh_state().replace(
        player_role=jnp.int8(4),   # KNIGHT
        player_align=jnp.int8(2),  # LAWFUL
    )
    jit_gcrownu = jax.jit(gcrownu)
    s2 = jit_gcrownu(state, jax.random.PRNGKey(99))
    assert int(s2.inventory.items.artifact_idx[0]) == 0
