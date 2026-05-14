#!/usr/bin/env python3
"""Dump our OBJECTS table for comparison."""
from Nethax.nethax.constants.objects import OBJECTS

with open("/tmp/our_objects.txt", "w") as f:
    for i, o in enumerate(OBJECTS):
        f.write(f"{i}\t{o.name}\t{o.description or ''}\t{int(o.class_)}\t{o.weight}\t{o.cost}\n")

# Print first 10
for i in range(min(15, len(OBJECTS))):
    o = OBJECTS[i]
    print(f"  {i:3d}  {o.name!r:30s} desc={o.description!r}")
print(f"Total: {len(OBJECTS)}")
