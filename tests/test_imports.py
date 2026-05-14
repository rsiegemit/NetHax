"""Smoke-import tests — verifies every Wave 1 stub module is importable.

All imports are lazy (inside test functions) so that a missing module
causes a single test failure rather than a collection error.
"""

import pytest


# ---------------------------------------------------------------------------
# Constants sub-modules
# ---------------------------------------------------------------------------

def test_import_constants_actions():
    import Nethax.nethax.constants.actions  # noqa: F401


def test_import_constants_glyphs():
    import Nethax.nethax.constants.glyphs  # noqa: F401


def test_import_constants_blstats():
    import Nethax.nethax.constants.blstats  # noqa: F401


def test_import_constants_roles():
    import Nethax.nethax.constants.roles  # noqa: F401


def test_import_constants_races():
    import Nethax.nethax.constants.races  # noqa: F401


def test_import_constants_monsters():
    import Nethax.nethax.constants.monsters  # noqa: F401


def test_import_constants_objects():
    import Nethax.nethax.constants.objects  # noqa: F401


def test_import_constants_package():
    import Nethax.nethax.constants  # noqa: F401


# ---------------------------------------------------------------------------
# Subsystems
# ---------------------------------------------------------------------------

def test_import_subsystem_combat():
    import Nethax.nethax.subsystems.combat  # noqa: F401


def test_import_subsystem_magic():
    import Nethax.nethax.subsystems.magic  # noqa: F401


def test_import_subsystem_monster_ai():
    import Nethax.nethax.subsystems.monster_ai  # noqa: F401


def test_import_subsystem_polymorph():
    import Nethax.nethax.subsystems.polymorph  # noqa: F401


def test_import_subsystem_inventory():
    import Nethax.nethax.subsystems.inventory  # noqa: F401


def test_import_subsystem_items():
    import Nethax.nethax.subsystems.items  # noqa: F401


def test_import_subsystem_identification():
    import Nethax.nethax.subsystems.identification  # noqa: F401


def test_import_subsystem_traps():
    import Nethax.nethax.subsystems.traps  # noqa: F401


def test_import_subsystem_features():
    import Nethax.nethax.subsystems.features  # noqa: F401


def test_import_subsystem_prayer():
    import Nethax.nethax.subsystems.prayer  # noqa: F401


def test_import_subsystem_conduct():
    import Nethax.nethax.subsystems.conduct  # noqa: F401


def test_import_subsystem_shop():
    import Nethax.nethax.subsystems.shop  # noqa: F401


def test_import_subsystem_status_effects():
    import Nethax.nethax.subsystems.status_effects  # noqa: F401


# ---------------------------------------------------------------------------
# Dungeon package
# ---------------------------------------------------------------------------

def test_import_dungeon_package():
    import Nethax.nethax.dungeon  # noqa: F401


def test_import_dungeon_branches():
    import Nethax.nethax.dungeon.branches  # noqa: F401


def test_import_dungeon_rooms():
    import Nethax.nethax.dungeon.rooms  # noqa: F401


def test_import_dungeon_mazes():
    import Nethax.nethax.dungeon.mazes  # noqa: F401


def test_import_dungeon_corridors():
    import Nethax.nethax.dungeon.corridors  # noqa: F401


def test_import_dungeon_special_levels():
    import Nethax.nethax.dungeon.special_levels  # noqa: F401


def test_import_dungeon_level_memory():
    import Nethax.nethax.dungeon.level_memory  # noqa: F401


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def test_import_obs_nle_obs():
    import Nethax.nethax.obs.nle_obs  # noqa: F401


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

def test_import_fov():
    import Nethax.nethax.fov  # noqa: F401


def test_import_rng():
    import Nethax.nethax.rng  # noqa: F401


def test_import_save_load():
    import Nethax.nethax.save_load  # noqa: F401
