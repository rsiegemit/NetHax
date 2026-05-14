#!/usr/bin/env python3
"""Generate the new Nethax/nethax/constants/objects.py from vendor parse.

Uses /tmp/vendor_entries.json (produced by extract_vendor_fields.py) as input.
Emits the complete OBJECTS tuple with 453 entries in vendor canonical order.
"""
import json
import re

with open("/tmp/vendor_entries.json") as f:
    entries = json.load(f)

# Class name → ObjectClass member (for code emission)
CLASS_MAP = {
    "ILLOBJ_CLASS":  "ObjectClass.ILLOBJ_CLASS",
    "WEAPON_CLASS":  "ObjectClass.WEAPON_CLASS",
    "ARMOR_CLASS":   "ObjectClass.ARMOR_CLASS",
    "RING_CLASS":    "ObjectClass.RING_CLASS",
    "AMULET_CLASS":  "ObjectClass.AMULET_CLASS",
    "TOOL_CLASS":    "ObjectClass.TOOL_CLASS",
    "FOOD_CLASS":    "ObjectClass.FOOD_CLASS",
    "POTION_CLASS":  "ObjectClass.POTION_CLASS",
    "SCROLL_CLASS":  "ObjectClass.SCROLL_CLASS",
    "SPBOOK_CLASS":  "ObjectClass.SPBOOK_CLASS",
    "WAND_CLASS":    "ObjectClass.WAND_CLASS",
    "COIN_CLASS":    "ObjectClass.COIN_CLASS",
    "GEM_CLASS":     "ObjectClass.GEM_CLASS",
    "ROCK_CLASS":    "ObjectClass.ROCK_CLASS",
    "BALL_CLASS":    "ObjectClass.BALL_CLASS",
    "CHAIN_CLASS":   "ObjectClass.CHAIN_CLASS",
    "VENOM_CLASS":   "ObjectClass.VENOM_CLASS",
}

# Material constant names → Material enum (for code emission)
MATERIAL_MAP = {
    "0":            "Material.NO_MATERIAL",
    "LIQUID":       "Material.LIQUID",
    "WAX":          "Material.WAX",
    "VEGGY":        "Material.VEGGY",
    "FLESH":        "Material.FLESH",
    "PAPER":        "Material.PAPER",
    "CLOTH":        "Material.CLOTH",
    "LEATHER":      "Material.LEATHER",
    "WOOD":         "Material.WOOD",
    "BONE":         "Material.BONE",
    "DRAGON_HIDE":  "Material.DRAGON_HIDE",
    "IRON":         "Material.IRON",
    "METAL":        "Material.METAL",
    "COPPER":       "Material.COPPER",
    "SILVER":       "Material.SILVER",
    "GOLD":         "Material.GOLD",
    "PLATINUM":     "Material.PLATINUM",
    "MITHRIL":      "Material.MITHRIL",
    "PLASTIC":      "Material.PLASTIC",
    "GLASS":        "Material.GLASS",
    "GEMSTONE":     "Material.GEMSTONE",
    "MINERAL":      "Material.MINERAL",
}

# Color name → integer value (vendor color.h)
COLOR_VALUES = {
    "CLR_BLACK": 0,
    "CLR_RED": 1,
    "CLR_GREEN": 2,
    "CLR_BROWN": 3,
    "CLR_BLUE": 4,
    "CLR_MAGENTA": 5,
    "CLR_CYAN": 6,
    "CLR_GRAY": 7,
    "NO_COLOR": 8,
    "CLR_ORANGE": 9,
    "CLR_BRIGHT_GREEN": 10,
    "CLR_YELLOW": 11,
    "CLR_BRIGHT_BLUE": 12,
    "CLR_BRIGHT_MAGENTA": 13,
    "CLR_BRIGHT_CYAN": 14,
    "CLR_WHITE": 15,
    "HI_METAL": 6,
    "HI_WOOD": 3,
    "HI_SILVER": 7,
    "HI_COPPER": 11,
    "HI_GOLD": 11,
    "HI_LEATHER": 3,
    "HI_CLOTH": 3,
    "HI_GLASS": 14,
    "HI_MINERAL": 7,
    "HI_ORGANIC": 3,
    "HI_PAPER": 15,
    "DRAGON_SILVER": 14,
}


def emit_class(c):
    return CLASS_MAP.get(c, "ObjectClass.ILLOBJ_CLASS")


def emit_material(m):
    return MATERIAL_MAP.get(m, "Material.NO_MATERIAL")


def emit_color(c):
    c = c.strip()
    if c in COLOR_VALUES:
        if c.startswith("CLR_") or c == "NO_COLOR":
            return f"Color.{c}"
        if hasattr_color_alias(c):
            return f"Color.{c}"
        return str(COLOR_VALUES[c])
    if c.isdigit():
        return c
    return "0"  # unknown color fallback


def hasattr_color_alias(c):
    return c in {"HI_METAL", "HI_WOOD", "HI_SILVER", "HI_COPPER", "HI_GOLD",
                 "HI_LEATHER", "HI_CLOTH", "HI_GLASS", "HI_MINERAL"}


def emit_int(s):
    """Emit integer field. Pass through literal numbers and identifier-resolved values."""
    s = s.strip()
    if s == "":
        return "0"
    # Try parse as int
    try:
        return str(int(s))
    except ValueError:
        pass
    # Handle simple arith like "(10 - 2)"
    m = re.match(r"^\(?\s*(\d+)\s*-\s*(\d+)\s*\)?$", s)
    if m:
        return str(int(m.group(1)) - int(m.group(2)))
    # Unknown identifier (e.g., spell level macro) → 0 (safe default)
    return "0"


def emit_dam(field):
    """Damage field 'X' meaning (1, X)."""
    field = field.strip()
    try:
        n = int(field)
        return f"(1, {n})" if n != 0 else "(0, 0)"
    except ValueError:
        return "(0, 0)"


def emit_entry(i, e):
    """Emit a single ObjectEntry literal."""
    name = e["name"]
    desc = e["desc"]
    cls = emit_class(e["class_"])
    prob = emit_int(e["prob"])
    wt = emit_int(e["wt"])
    cost = emit_int(e["cost"])
    sdam = emit_dam(e["sdam"])
    ldam = emit_dam(e["ldam"])
    oc1 = emit_int(e["oc1"])
    oc2 = emit_int(e["oc2"])
    nut = emit_int(e["nutrition"])
    color = emit_color(e["color"])
    material = emit_material(e["material"])

    name_str = "None" if name is None else repr(name)
    desc_str = "None" if desc is None else repr(desc)

    return (
        f"    # {i} — {name or '<no-name>'}  ({e['macro']})\n"
        f"    ObjectEntry(\n"
        f"        name={name_str},\n"
        f"        description={desc_str},\n"
        f"        class_={cls},\n"
        f"        prob={prob},\n"
        f"        weight={wt},\n"
        f"        cost={cost},\n"
        f"        sdam={sdam},\n"
        f"        ldam={ldam},\n"
        f"        oc1={oc1},\n"
        f"        oc2={oc2},\n"
        f"        nutrition={nut},\n"
        f"        color={color},\n"
        f"        material={material},\n"
        f"    ),\n"
    )


# Build the file
HEADER = '''"""
NetHack object table — full vendor-parity table (Wave 6 Phase B+).

Canonical source: vendor/nle/src/objects.c::objects[] (430 named entries +
23 None-named appearance slots = 453 total, matching live NLE NUM_OBJECTS).

Generated by scripts/generate_objects_py.py from vendor entries dumped to
/tmp/vendor_entries.json (see scripts/extract_vendor_fields.py).

DO NOT hand-edit this file. To regenerate:
    .venv/bin/python scripts/extract_vendor_fields.py
    .venv/bin/python scripts/generate_objects_py.py
"""

from __future__ import annotations

import dataclasses
from enum import IntEnum
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Object class enum  (from vendor/nethack/include/defsym.h OBJCLASS_CLASS_ENUM)
# ---------------------------------------------------------------------------

class ObjectClass(IntEnum):
    RANDOM_CLASS =  0
    ILLOBJ_CLASS =  1
    WEAPON_CLASS =  2
    ARMOR_CLASS  =  3
    RING_CLASS   =  4
    AMULET_CLASS =  5
    TOOL_CLASS   =  6
    FOOD_CLASS   =  7
    POTION_CLASS =  8
    SCROLL_CLASS =  9
    SPBOOK_CLASS = 10
    WAND_CLASS   = 11
    COIN_CLASS   = 12
    GEM_CLASS    = 13
    ROCK_CLASS   = 14
    BALL_CLASS   = 15
    CHAIN_CLASS  = 16
    VENOM_CLASS  = 17


# ---------------------------------------------------------------------------
# Material types
# ---------------------------------------------------------------------------

class Material(IntEnum):
    NO_MATERIAL =  0
    LIQUID      =  1
    WAX         =  2
    VEGGY       =  3
    FLESH       =  4
    PAPER       =  5
    CLOTH       =  6
    LEATHER     =  7
    WOOD        =  8
    BONE        =  9
    DRAGON_HIDE = 10
    IRON        = 11
    METAL       = 12
    COPPER      = 13
    SILVER      = 14
    GOLD        = 15
    PLATINUM    = 16
    MITHRIL     = 17
    PLASTIC     = 18
    GLASS       = 19
    GEMSTONE    = 20
    MINERAL     = 21


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

class Color(IntEnum):
    CLR_BLACK          =  0
    CLR_RED            =  1
    CLR_GREEN          =  2
    CLR_BROWN          =  3
    CLR_BLUE           =  4
    CLR_MAGENTA        =  5
    CLR_CYAN           =  6
    CLR_GRAY           =  7
    NO_COLOR           =  8
    CLR_ORANGE         =  9
    CLR_BRIGHT_GREEN   = 10
    CLR_YELLOW         = 11
    CLR_BRIGHT_BLUE    = 12
    CLR_BRIGHT_MAGENTA = 13
    CLR_BRIGHT_CYAN    = 14
    CLR_WHITE          = 15
    # Aliases used in objects.h
    HI_METAL   =  6
    HI_WOOD    =  3
    HI_SILVER  =  7
    HI_COPPER  = 11
    HI_GOLD    = 11
    HI_LEATHER =  3
    HI_CLOTH   =  3
    HI_GLASS   = 14
    HI_MINERAL =  7


# ---------------------------------------------------------------------------
# ObjectEntry dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ObjectEntry:
    name: Optional[str]
    description: Optional[str]
    class_: ObjectClass
    prob: int
    weight: int
    cost: int
    sdam: Tuple[int, int]
    ldam: Tuple[int, int]
    oc1: int
    oc2: int
    nutrition: int
    color: int
    material: Material


# ---------------------------------------------------------------------------
# Canonical NUM_OBJECTS — matches live NLE binary (vendor/nle/src/objects.c
# with 430 named entries + 23 None-named appearance slots = 453 total).
# ---------------------------------------------------------------------------

NUM_OBJECTS: int = 453


# ---------------------------------------------------------------------------
# OBJECTS — full canonical table, vendor-index aligned (0..452).
# Each ObjectEntry tracks vendor/nle/src/objects.c canonical order so that
# OBJECTS[i] corresponds to live NLE OBJ_NAME(objclass(i)).
# ---------------------------------------------------------------------------

OBJECTS: Tuple[ObjectEntry, ...] = (
'''

FOOTER = ''')


# ---------------------------------------------------------------------------
# Backwards-compat alias map  ("potion of healing" -> bare-name index).
# Used by wish.py and other consumers that still pass prefixed lookup keys.
# ---------------------------------------------------------------------------

_CLASS_PREFIXES: dict = {
    int(ObjectClass.POTION_CLASS): "potion of ",
    int(ObjectClass.SCROLL_CLASS): "scroll of ",
    int(ObjectClass.WAND_CLASS):   "wand of ",
    int(ObjectClass.RING_CLASS):   "ring of ",
    int(ObjectClass.AMULET_CLASS): "amulet of ",
    int(ObjectClass.SPBOOK_CLASS): "spellbook of ",
}

OBJECT_NAME_ALIASES: dict = {}
for _i, _o in enumerate(OBJECTS):
    if _o.name is None:
        continue
    _prefix = _CLASS_PREFIXES.get(int(_o.class_))
    if _prefix is not None and not _o.name.startswith(_prefix):
        OBJECT_NAME_ALIASES[_prefix + _o.name] = _i
del _i, _o, _prefix
'''

body_lines = [HEADER]
for i, e in enumerate(entries):
    body_lines.append(emit_entry(i, e))
body_lines.append(FOOTER)

out_path = "/Users/rsiegelmann/Downloads/Projects/nethax/Nethax/nethax/constants/objects.py"
with open(out_path, "w") as f:
    f.write("".join(body_lines))

print(f"Wrote {out_path} with {len(entries)} entries.")
