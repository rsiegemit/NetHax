"""Tests that each subsystem state class can be instantiated via its zero-init factory."""

import pytest

# Each tuple is (human_label, lazy_import_expr, factory_attr_or_fn).
# factory_attr_or_fn is the name of the classmethod/function on the module or
# on the class itself that produces a default instance.
_CASES = [
    # (label, module_path, class_name, factory)
    # factory is either a classmethod name on the class ("default", "empty", …)
    # or a module-level function name ("make_polymorph_state", …).
    ("CombatState",         "Nethax.nethax.subsystems.combat",          "CombatState",         "default"),
    ("MagicState",          "Nethax.nethax.subsystems.magic",           "MagicState",          "default"),
    ("ShopState",           "Nethax.nethax.subsystems.shop",            "ShopState",           "default"),
    ("PrayerState",         "Nethax.nethax.subsystems.prayer",          "PrayerState",         "default"),
    ("ConductState",        "Nethax.nethax.subsystems.conduct",         "ConductState",        "default"),
    ("StatusState",         "Nethax.nethax.subsystems.status_effects",  "StatusState",         "default"),
    ("InventoryState",      "Nethax.nethax.subsystems.inventory",       "InventoryState",      "empty"),
    ("IdentificationState", "Nethax.nethax.subsystems.identification",  "IdentificationState", "unshuffled"),
    ("ItemEffects",         "Nethax.nethax.subsystems.items",           "ItemEffects",         "default"),
    # PolymorphState: module-level factory function
    ("PolymorphState",      "Nethax.nethax.subsystems.polymorph",       None,                  "make_polymorph_state"),
    # MonsterAIState: module-level factory function
    ("MonsterAIState",      "Nethax.nethax.subsystems.monster_ai",      None,                  "make_monster_ai_state"),
]

# TrapState and FeaturesState require dimension args — test them separately.

@pytest.mark.parametrize("label,mod_path,cls_name,factory", _CASES, ids=[c[0] for c in _CASES])
def test_state_default(label, mod_path, cls_name, factory):
    import importlib
    mod = importlib.import_module(mod_path)
    if cls_name is not None:
        cls = getattr(mod, cls_name)
        instance = getattr(cls, factory)()
    else:
        fn = getattr(mod, factory)
        instance = fn()
    assert instance is not None, f"{label}: factory returned None"


def test_trap_state_default():
    from Nethax.nethax.subsystems.traps import TrapState
    instance = TrapState.default(num_levels=1, map_h=21, map_w=79)
    assert instance is not None


def test_features_state_default():
    from Nethax.nethax.subsystems.features import FeaturesState
    instance = FeaturesState.default(num_levels=1, map_h=21, map_w=79)
    assert instance is not None
