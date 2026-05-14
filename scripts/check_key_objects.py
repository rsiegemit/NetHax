#!/usr/bin/env python3
"""Check key vendor parity outcomes."""
import sys
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")
from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS, ObjectClass, OBJECT_NAME_ALIASES

print("NUM_OBJECTS:", NUM_OBJECTS, "len(OBJECTS):", len(OBJECTS))
print("aliases sample:", list(OBJECT_NAME_ALIASES.items())[:3])
print()


def find(name):
    for i, o in enumerate(OBJECTS):
        if o.name == name:
            return i, o
    return None, None


# Long sword (vendor: WEAPON("long sword", None, 1, 1, 0, 50, 40, 15, 8, 12, 0, S, P_LONG_SWORD, IRON, HI_METAL))
i, o = find("long sword")
print(f"long sword: idx={i} wt={o.weight} cost={o.cost} sdam={o.sdam} ldam={o.ldam}")

# water (bare name; potion of healing alias should resolve)
i, o = find("water")
print(f"water: idx={i} class={o.class_.name}")

# healing
i, o = find("healing")
print(f"healing: idx={i} class={o.class_.name}")

# Amulet of Yendor
i, o = find("Amulet of Yendor")
print(f"Amulet of Yendor: idx={i} class={o.class_.name}")

# Dragon scales — count chromatic variants
print("\nDragon scale mail variants:")
for i, o in enumerate(OBJECTS):
    if o.name and "dragon scale mail" in o.name:
        print(f"  [{i}] {o.name}")
print("Dragon scales (non-mail):")
for i, o in enumerate(OBJECTS):
    if o.name and o.name.endswith("dragon scales"):
        print(f"  [{i}] {o.name}")

# Alias check
print("\n'potion of healing' alias:", OBJECT_NAME_ALIASES.get("potion of healing"))
print("'scroll of identify' alias:", OBJECT_NAME_ALIASES.get("scroll of identify"))
