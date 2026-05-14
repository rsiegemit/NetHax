"""Wave 3 tests for character creation.

Tests cover:
  - Reset with VALKYRIE → correct wielded weapon, worn shield
  - Reset with WIZARD   → quarterstaff wielded, cloak worn
  - Starting HP/PW within expected ranges for each role
  - Stats within min/max ranges for role
  - All 13 roles create without error
"""
import pytest


def _reset_as(role, race=None, alignment=0, seed=42):
    """Helper: reset env as a given role/race and return state."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.races import Race

    if race is None:
        race = Race.HUMAN

    rng = jax.random.PRNGKey(seed)
    env = NethaxEnv()
    state, _ = env.reset(rng, role=role, race=race, alignment=alignment)
    return state


# ---------------------------------------------------------------------------
# Valkyrie
# ---------------------------------------------------------------------------

def test_valkyrie_reset_has_spear_wielded():
    """Valkyrie should start with a spear (type_id=10) wielded."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.character import ObjType

    state = _reset_as(Role.VALKYRIE)
    wielded = int(state.inventory.wielded)
    assert wielded >= 0, "Valkyrie should have a wielded weapon"

    cat = int(state.inventory.items.category[wielded])
    assert cat == int(ItemCategory.WEAPON), (
        f"Wielded item should be WEAPON; got category={cat}"
    )
    # Valkyrie starts with a spear
    tid = int(state.inventory.items.type_id[wielded])
    assert tid == ObjType.SPEAR, f"Expected spear type_id={ObjType.SPEAR}; got {tid}"


def test_valkyrie_has_shield_worn():
    """Valkyrie should have a small shield worn in the SHIELD slot."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory, ArmorSlot

    state = _reset_as(Role.VALKYRIE)
    shield_idx = int(state.inventory.worn_armor[int(ArmorSlot.SHIELD)])
    assert shield_idx >= 0, "Valkyrie should have a shield equipped"

    cat = int(state.inventory.items.category[shield_idx])
    assert cat == int(ItemCategory.ARMOR), (
        f"Shield slot item should be ARMOR; got category={cat}"
    )


def test_valkyrie_ac_reduced_by_shield():
    """Valkyrie's AC should be < BASE_AC because of the starting shield."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import BASE_AC

    state = _reset_as(Role.VALKYRIE)
    assert int(state.player_ac) < BASE_AC, (
        f"Valkyrie AC should be less than {BASE_AC}; got {int(state.player_ac)}"
    )


def test_valkyrie_starting_hp():
    """Valkyrie HP should be at least hp_base=14."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import STARTING_HP_PW

    state = _reset_as(Role.VALKYRIE)
    hp_base = STARTING_HP_PW[Role.VALKYRIE][0]
    assert int(state.player_hp) >= hp_base, (
        f"Valkyrie HP {int(state.player_hp)} < expected base {hp_base}"
    )


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

def test_wizard_reset_has_quarterstaff_wielded():
    """Wizard should start with quarterstaff wielded."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.character import ObjType

    state = _reset_as(Role.WIZARD)
    wielded = int(state.inventory.wielded)
    assert wielded >= 0, "Wizard should have a wielded weapon"

    cat = int(state.inventory.items.category[wielded])
    assert cat == int(ItemCategory.WEAPON), f"Wielded item is not a weapon; cat={cat}"

    tid = int(state.inventory.items.type_id[wielded])
    assert tid == ObjType.QUARTERSTAFF, (
        f"Expected quarterstaff type_id={ObjType.QUARTERSTAFF}; got {tid}"
    )


def test_wizard_has_cloak_worn():
    """Wizard should have cloak of magic resistance in CLOAK slot."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory, ArmorSlot

    state = _reset_as(Role.WIZARD)
    cloak_idx = int(state.inventory.worn_armor[int(ArmorSlot.CLOAK)])
    assert cloak_idx >= 0, "Wizard should have a cloak equipped"

    cat = int(state.inventory.items.category[cloak_idx])
    assert cat == int(ItemCategory.ARMOR), (
        f"Cloak slot item should be ARMOR; got category={cat}"
    )


def test_wizard_has_spellbook_in_inventory():
    """Wizard should have at least one spellbook in inventory."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _reset_as(Role.WIZARD)
    items = state.inventory.items
    spellbook_found = any(
        int(items.category[i]) == int(ItemCategory.SPBOOK)
        for i in range(len(list(items.category)))
    )
    assert spellbook_found, "Wizard should have at least one spellbook"


def test_wizard_has_wand_in_inventory():
    """Wizard should have at least one wand in inventory."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory

    state = _reset_as(Role.WIZARD)
    items = state.inventory.items
    wand_found = any(
        int(items.category[i]) == int(ItemCategory.WAND)
        for i in range(len(list(items.category)))
    )
    assert wand_found, "Wizard should have at least one wand"


# ---------------------------------------------------------------------------
# Monk — fights barehanded
# ---------------------------------------------------------------------------

def test_monk_has_no_weapon_wielded():
    """Monk should start with wielded == -1 (fights barehanded)."""
    from Nethax.nethax.constants.roles import Role

    state = _reset_as(Role.MONK)
    assert int(state.inventory.wielded) == -1, (
        f"Monk should have no weapon wielded; got {int(state.inventory.wielded)}"
    )


# ---------------------------------------------------------------------------
# Stats within range
# ---------------------------------------------------------------------------

def test_valkyrie_stats_in_range():
    """Valkyrie stats must be within the canonical role min/max."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    from Nethax.nethax.subsystems.character import STARTING_STATS

    state  = _reset_as(Role.VALKYRIE, Race.HUMAN)
    ranges = STARTING_STATS[(Role.VALKYRIE, Race.HUMAN)]

    stat_values = {
        "str": int(state.player_str),
        "int": int(state.player_int),
        "wis": int(state.player_wis),
        "dex": int(state.player_dex),
        "con": int(state.player_con),
        "cha": int(state.player_cha),
    }
    for stat, val in stat_values.items():
        lo, hi = ranges[stat]
        assert lo <= val <= hi, (
            f"Valkyrie {stat}={val} out of range [{lo}, {hi}]"
        )


def test_wizard_stats_in_range():
    """Wizard stats must be within the canonical role min/max."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    from Nethax.nethax.subsystems.character import STARTING_STATS

    state  = _reset_as(Role.WIZARD, Race.HUMAN, seed=7)
    ranges = STARTING_STATS[(Role.WIZARD, Race.HUMAN)]

    stat_values = {
        "str": int(state.player_str),
        "int": int(state.player_int),
        "wis": int(state.player_wis),
        "dex": int(state.player_dex),
        "con": int(state.player_con),
        "cha": int(state.player_cha),
    }
    for stat, val in stat_values.items():
        lo, hi = ranges[stat]
        assert lo <= val <= hi, (
            f"Wizard {stat}={val} out of range [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# All 13 roles create without error
# ---------------------------------------------------------------------------

def test_all_13_roles_create():
    """All 13 roles must reset without raising an exception."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.roles import Role, N_ROLES
    from Nethax.nethax.constants.races import Race

    assert N_ROLES == 13, f"Expected 13 roles, got {N_ROLES}"

    env = NethaxEnv()
    errors = []
    for role in Role:
        try:
            rng = jax.random.PRNGKey(int(role))
            state, _ = env.reset(rng, role=role, race=Race.HUMAN)
            # Basic sanity: HP > 0, state is usable
            assert int(state.player_hp) > 0, f"{role.name} has HP <= 0"
        except Exception as exc:
            errors.append(f"{role.name}: {exc}")

    assert not errors, "Errors creating characters:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# HP/PW sanity for all roles
# ---------------------------------------------------------------------------

def test_all_roles_hp_pw_positive():
    """All roles must start with positive HP and non-negative PW."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race

    env = NethaxEnv()
    for role in Role:
        rng   = jax.random.PRNGKey(int(role) + 100)
        state, _ = env.reset(rng, role=role, race=Race.HUMAN)
        assert int(state.player_hp) > 0, f"{role.name}: HP={int(state.player_hp)} <= 0"
        assert int(state.player_pw) >= 0, f"{role.name}: PW={int(state.player_pw)} < 0"


# ---------------------------------------------------------------------------
# Inventory non-empty for roles that have starting items
# ---------------------------------------------------------------------------

def test_starting_inventory_non_empty_for_armed_roles():
    """Roles with weapons should have at least one item in inventory."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race
    from Nethax.nethax.subsystems.inventory import ItemCategory

    armed_roles = [
        Role.VALKYRIE, Role.WIZARD, Role.KNIGHT, Role.SAMURAI,
        Role.BARBARIAN, Role.ROGUE, Role.RANGER, Role.HEALER,
        Role.PRIEST, Role.CAVEMAN, Role.TOURIST, Role.ARCHEOLOGIST,
    ]

    env = NethaxEnv()
    for role in armed_roles:
        rng   = jax.random.PRNGKey(int(role) + 200)
        state, _ = env.reset(rng, role=role, race=Race.HUMAN)
        cats  = [int(state.inventory.items.category[i]) for i in range(10)]
        has_item = any(c != 0 for c in cats)
        assert has_item, f"{role.name}: no items in starting inventory"


# ---------------------------------------------------------------------------
# Role / race fields set correctly
# ---------------------------------------------------------------------------

def test_reset_sets_role_and_race_fields():
    """state.player_role and state.player_race must match the requested values."""
    import jax
    from Nethax.nethax.env import NethaxEnv
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.constants.races import Race

    env = NethaxEnv()
    rng = jax.random.PRNGKey(99)
    state, _ = env.reset(rng, role=Role.SAMURAI, race=Race.HUMAN)

    assert int(state.player_role) == int(Role.SAMURAI), (
        f"player_role={int(state.player_role)} expected {int(Role.SAMURAI)}"
    )
    assert int(state.player_race) == int(Race.HUMAN), (
        f"player_race={int(state.player_race)} expected {int(Race.HUMAN)}"
    )


# ---------------------------------------------------------------------------
# Wave 6 Phase B — starting kits, starting spells, starting pets
# ---------------------------------------------------------------------------

def test_valkyrie_wields_weapon_in_slot_0():
    """Valkyrie should wield slot 0 (primary weapon)."""
    from Nethax.nethax.constants.roles import Role

    state = _reset_as(Role.VALKYRIE)
    assert int(state.inventory.wielded) == 0, (
        f"Valkyrie should wield slot 0; got {int(state.inventory.wielded)}"
    )


def test_wizard_has_quarterstaff_and_force_bolt():
    """Wizard starts with quarterstaff wielded AND knows force-bolt spell."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import ObjType
    from Nethax.nethax.subsystems.magic import SpellId

    state = _reset_as(Role.WIZARD)

    # Quarterstaff wielded
    wielded = int(state.inventory.wielded)
    assert wielded >= 0
    tid = int(state.inventory.items.type_id[wielded])
    assert tid == ObjType.QUARTERSTAFF, (
        f"Wizard wielded type_id={tid}, expected QUARTERSTAFF={ObjType.QUARTERSTAFF}"
    )

    # Force bolt memorised
    assert bool(state.magic.spell_known[int(SpellId.FORCE_BOLT)]), (
        "Wizard should know FORCE_BOLT at start"
    )


def test_wizard_starting_spell_force_bolt_in_memory():
    """Wizard's spell_memory[FORCE_BOLT] should be positive (freshly memorised)."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.magic import SpellId

    state = _reset_as(Role.WIZARD)
    mem = int(state.magic.spell_memory[int(SpellId.FORCE_BOLT)])
    assert mem > 0, f"Force-bolt spell_memory should be > 0; got {mem}"


def test_priest_starts_with_protection_spell():
    """Priest starts with PROTECTION memorised."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.magic import SpellId

    state = _reset_as(Role.PRIEST)
    assert bool(state.magic.spell_known[int(SpellId.PROTECTION)]), (
        "Priest should know PROTECTION at start"
    )


def test_healer_starts_with_healing_spell():
    """Healer starts with HEALING memorised."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.magic import SpellId

    state = _reset_as(Role.HEALER)
    assert bool(state.magic.spell_known[int(SpellId.HEALING)]), (
        "Healer should know HEALING at start"
    )


def test_monk_starts_with_protection_spell():
    """Monk starts with PROTECTION memorised."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.magic import SpellId

    state = _reset_as(Role.MONK)
    assert bool(state.magic.spell_known[int(SpellId.PROTECTION)]), (
        "Monk should know PROTECTION at start"
    )


def test_non_caster_roles_have_no_starting_spells():
    """Valkyrie/Knight/Barbarian/etc. start with no spells memorised."""
    import jax.numpy as jnp
    from Nethax.nethax.constants.roles import Role

    for role in (Role.VALKYRIE, Role.KNIGHT, Role.BARBARIAN, Role.SAMURAI):
        state = _reset_as(role)
        assert not bool(jnp.any(state.magic.spell_known)), (
            f"{role.name} should start with no spells memorised"
        )


def test_healer_has_four_potions_of_healing():
    """Healer starts with exactly 4 potions of healing (canonical NetHack)."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.character import ObjType

    state = _reset_as(Role.HEALER)
    items = state.inventory.items
    healing_quantity = 0
    for i in range(20):
        cat = int(items.category[i])
        tid = int(items.type_id[i])
        qty = int(items.quantity[i])
        if cat == int(ItemCategory.POTION) and tid == ObjType.POT_HEALING:
            healing_quantity += qty
    assert healing_quantity == 4, (
        f"Healer should have 4 potions of healing; got {healing_quantity}"
    )


def test_samurai_has_yumi_and_arrows():
    """Samurai starts with yumi (bow) + ya (arrows)."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.character import ObjType

    state = _reset_as(Role.SAMURAI)
    items = state.inventory.items
    has_yumi = False
    arrow_qty = 0
    for i in range(20):
        cat = int(items.category[i])
        tid = int(items.type_id[i])
        if cat == int(ItemCategory.WEAPON):
            if tid == ObjType.YUMI:
                has_yumi = True
            elif tid == ObjType.YA:
                arrow_qty += int(items.quantity[i])
    assert has_yumi, "Samurai should have a yumi (bow)"
    assert arrow_qty >= 25, f"Samurai should have >= 25 ya arrows; got {arrow_qty}"


def test_knight_has_long_sword_and_lance():
    """Knight starts with long sword + lance."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.inventory import ItemCategory
    from Nethax.nethax.subsystems.character import ObjType

    state = _reset_as(Role.KNIGHT)
    items = state.inventory.items
    has_long_sword = False
    has_lance = False
    for i in range(20):
        cat = int(items.category[i])
        tid = int(items.type_id[i])
        if cat == int(ItemCategory.WEAPON):
            if tid == ObjType.LONG_SWORD:
                has_long_sword = True
            elif tid == ObjType.LANCE:
                has_lance = True
    assert has_long_sword, "Knight should start with a long sword"
    assert has_lance, "Knight should start with a lance"


def test_knight_starting_pet_is_pony():
    """Knight's starting pet is canonically a pony (role.c roles[].petnum)."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import get_starting_pet

    assert get_starting_pet(Role.KNIGHT) == "pony"


def test_wizard_starting_pet_is_kitten():
    """Wizard's starting pet is a kitten."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import get_starting_pet

    assert get_starting_pet(Role.WIZARD) == "kitten"


def test_samurai_starting_pet_is_little_dog():
    """Samurai's canonical pet is a little dog (vendor role.c)."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import get_starting_pet

    assert get_starting_pet(Role.SAMURAI) == "little dog"


def test_all_13_roles_have_starting_pet():
    """STARTING_PET table covers all 13 roles."""
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import STARTING_PET

    for role in Role:
        assert role in STARTING_PET, f"{role.name} missing from STARTING_PET"
        assert isinstance(STARTING_PET[role], str)
        assert len(STARTING_PET[role]) > 0


def test_each_role_creates_unique_kit():
    """At least 10 of the 13 role inventories should be pairwise unique.

    Canonical NetHack roles all have distinct starting kits, but two roles
    could share a single weapon if the kit is otherwise identical — so we
    require uniqueness via a (sorted type_id × category) fingerprint.
    """
    from Nethax.nethax.constants.roles import Role
    from Nethax.nethax.subsystems.character import STARTING_INVENTORY

    fingerprints = set()
    for role in Role:
        items = STARTING_INVENTORY[role]
        fp = tuple(sorted(
            (int(it.category), int(it.type_id), int(it.quantity))
            for it in items
        ))
        fingerprints.add(fp)
    assert len(fingerprints) == 13, (
        f"Expected 13 unique starting kits across 13 roles; got "
        f"{len(fingerprints)} distinct fingerprints"
    )
