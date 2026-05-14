#!/usr/bin/env python3
"""Verify the newly generated objects.py loads correctly."""
import sys
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")
from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS, ObjectClass

print("NUM_OBJECTS:", NUM_OBJECTS)
print("len(OBJECTS):", len(OBJECTS))

# Spot check vendor indices
for i in (0, 1, 3, 17, 38, 314, 451, 452):
    o = OBJECTS[i]
    print(f"  [{i}] {o.name!r:30s} desc={o.description!r:30s} cls={o.class_!r}")
print()

# Find specific items
def find(name):
    for i, o in enumerate(OBJECTS):
        if o.name == name:
            return i, o
    return None, None

print("Required items present:")
for nm in ["strange object", "long sword", "arrow", "silver arrow",
          "orcish arrow", "amulet of Yendor", "novel", "acid venom",
          "blinding venom", "gold piece"]:
    i, o = find(nm)
    if o is None:
        print(f"  MISSING: {nm!r}")
    else:
        print(f"  [{i:3d}] {nm!r}")

print()
print("Formerly-extra items should be absent:")
for nm in ["amulet of flying", "gold dragon scale mail", "shimmering dragon scales",
          "amulet of guarding", "generic strange", "generic weapon",
          "splash of acid venom"]:
    i, o = find(nm)
    if o is None:
        print(f"  OK absent: {nm!r}")
    else:
        print(f"  STILL PRESENT: {nm!r} at [{i}]")
