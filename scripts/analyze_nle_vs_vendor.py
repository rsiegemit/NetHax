#!/usr/bin/env python3
"""Analyze NLE-live vs vendor.c diff. Find the 23 extras in live NLE."""
import nle.nethack as n

live_names = []
for i in range(n.NUM_OBJECTS):
    try:
        oc = n.objclass(i)
        nm = n.OBJ_NAME(oc)
    except Exception:
        nm = None
    live_names.append(nm or "")

# Vendor names (treat literal "None" as Python None)
vendor_names = []
with open("/tmp/vendor_objects_raw.txt") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        nm = parts[2]
        if nm == "None":
            nm = ""
        vendor_names.append(nm)

print(f"Live NLE count:     {len(live_names)}")
print(f"Vendor.c count:     {len(vendor_names)}")

live_set = set(live_names)
vendor_set = set(vendor_names)

print(f"\nIn live, not in vendor.c ({len(live_set - vendor_set)}):")
for nm in sorted(live_set - vendor_set):
    print(f"  - {nm!r}")
print(f"\nIn vendor.c, not in live ({len(vendor_set - live_set)}):")
for nm in sorted(vendor_set - live_set):
    print(f"  - {nm!r}")

# Side-by-side index alignment for the differing portion
print("\n--- Side by side (first 30 + last 10) ---")
for i in range(min(30, len(live_names))):
    v = vendor_names[i] if i < len(vendor_names) else ""
    flag = "" if v == live_names[i] else "  <-- MISMATCH"
    print(f"  {i:3d}  live={live_names[i]:35s} vendor={v}{flag}")

# Compare full alignment, treating None==''
print("\n--- Index-by-index mismatches ---")
mlen = max(len(live_names), len(vendor_names))
mismatches = 0
for i in range(mlen):
    lv = live_names[i] if i < len(live_names) else "<END>"
    vv = vendor_names[i] if i < len(vendor_names) else "<END>"
    lv_n = lv or ""
    vv_n = vv or ""
    if lv_n != vv_n:
        mismatches += 1
        if mismatches <= 20:
            print(f"  {i:3d}  live={lv!r:35s} vendor={vv!r}")
print(f"Total mismatches: {mismatches}")
