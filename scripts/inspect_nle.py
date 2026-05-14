#!/usr/bin/env python3
"""Inspect live NLE object attributes."""
import nle.nethack as n

attrs = sorted(dir(n))
print(f"NUM_OBJECTS = {n.NUM_OBJECTS}")
print()
print("Object-related attrs:")
for a in attrs:
    if "OBJ" in a.upper() and not a.startswith("_"):
        try:
            v = getattr(n, a)
            r = repr(v)
            if len(r) > 80:
                r = r[:80] + "..."
            print(f"  {a} = {r}")
        except Exception as e:
            print(f"  {a}: <{e}>")
