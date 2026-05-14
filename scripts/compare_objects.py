#!/usr/bin/env python3
"""Compare our OBJECTS table against vendor canonical list."""
import sys
sys.path.insert(0, "/Users/rsiegelmann/Downloads/Projects/nethax")

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass

# Load vendor list
vendor = []
with open("/tmp/vendor_objects_raw.txt") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        idx, macro, name = parts[0], parts[1], parts[2]
        desc = parts[3] if len(parts) > 3 else ""
        vendor.append((int(idx), macro, name, desc))

vendor_names = [v[2] for v in vendor]
vendor_set = set(vendor_names)

# Build our list: bare names (since prefixes were stripped)
ours = [(o.name, int(o.class_)) for o in OBJECTS]
our_names = [n for n, _ in ours]
our_set = set(our_names)

# Extras: in ours, not in vendor
extras = [n for n in our_names if n not in vendor_set]
# Missing: in vendor, not in ours
missing = [n for n in vendor_names if n not in our_set]

print(f"OURS:    {len(ours)}")
print(f"VENDOR:  {len(vendor)}")
print(f"EXTRAS (in ours, not vendor): {len(extras)}")
for e in extras:
    print(f"  - {e!r}")
print(f"MISSING (in vendor, not ours): {len(missing)}")
for m in missing:
    print(f"  - {m!r}")
