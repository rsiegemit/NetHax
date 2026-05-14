#!/usr/bin/env python3
"""Check appearance slot entries in the generated objects.py."""
import sys
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")
from Nethax.nethax.constants.objects import OBJECTS

# Check the appearance slots (should be around index 320-339 for scrolls)
for i in range(319, 342):
    if i < len(OBJECTS):
        o = OBJECTS[i]
        print(f"{i}: name={o.name!r:35s} desc={o.description!r}")

print()
print("None-named count:", sum(1 for o in OBJECTS if o.name is None))
print("Empty-named count:", sum(1 for o in OBJECTS if o.name == ""))
