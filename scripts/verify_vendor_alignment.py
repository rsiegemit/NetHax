#!/usr/bin/env python3
"""Verify parsed vendor entries align with live NLE OBJ_NAME / OBJ_DESCR."""
import json
import nle.nethack as n

with open("/tmp/vendor_entries.json") as f:
    entries = json.load(f)

print(f"Parsed: {len(entries)}  Live NLE: {n.NUM_OBJECTS}")

mismatches = 0
for i, e in enumerate(entries):
    try:
        oc = n.objclass(i)
        live_name = n.OBJ_NAME(oc) or ""
        live_desc = n.OBJ_DESCR(oc) or ""
    except Exception:
        live_name, live_desc = "<error>", ""
    vendor_name = e["name"] or ""
    vendor_desc = e["desc"] or ""
    if live_name != vendor_name or live_desc != vendor_desc:
        mismatches += 1
        if mismatches <= 12:
            print(f"  [{i}] live=({live_name!r}, {live_desc!r}) vendor=({vendor_name!r}, {vendor_desc!r})")
print(f"Total mismatches: {mismatches}")

# Verify some field values vs live NLE
print("\n--- Sample field checks ---")
for i in (1, 38, 47, 169, 230, 314, 318):
    if i >= n.NUM_OBJECTS:
        continue
    try:
        oc = n.objclass(i)
        live_wt = oc.oc_weight
        live_cost = oc.oc_cost
        live_cls = oc.oc_class
    except Exception as ex:
        live_wt = live_cost = live_cls = f"<err:{ex}>"
    e = entries[i]
    print(f"  [{i}] {e['name']!r:30s}  vendor wt={e['wt']} cost={e['cost']} | live wt={live_wt} cost={live_cost} class={live_cls}")
