#!/usr/bin/env python3
"""Dump live NLE objects[] table for ground truth."""
import nle.nethack as n

print(f"NUM_OBJECTS = {n.NUM_OBJECTS}")

with open("/tmp/nle_objects.txt", "w") as f:
    for i in range(n.NUM_OBJECTS):
        try:
            oc = n.objclass(i)
            name = n.OBJ_NAME(oc)
        except Exception as e:
            name = f"<err:{e}>"
        try:
            oc = n.objclass(i)
            desc = n.OBJ_DESCR(oc)
        except Exception:
            desc = ""
        # Also try to fetch objclass struct
        try:
            oc = n.objclass(i)
            cls = oc.oc_class
            wt = oc.oc_weight
            cost = oc.oc_cost
            color = oc.oc_color
            material = getattr(oc, 'oc_material', -1)
        except Exception as e:
            cls = wt = cost = color = material = -1
        f.write(f"{i}\t{name or ''}\t{desc or ''}\t{cls}\t{wt}\t{cost}\t{color}\t{material}\n")

print("Dumped to /tmp/nle_objects.txt")
# Print first/last to spot-check
with open("/tmp/nle_objects.txt") as f:
    lines = f.readlines()
for line in lines[:5]:
    print(line.rstrip())
print("...")
for line in lines[-5:]:
    print(line.rstrip())
