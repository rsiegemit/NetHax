#!/usr/bin/env python3
"""Count empty-name entries in live NLE."""
import nle.nethack as n

empty_indices = []
all_entries = []
for i in range(n.NUM_OBJECTS):
    try:
        oc = n.objclass(i)
        nm = n.OBJ_NAME(oc)
        ds = n.OBJ_DESCR(oc)
        cls = oc.oc_class
    except Exception:
        nm = None
        ds = None
        cls = -1
    all_entries.append((i, nm, ds, cls))
    if not nm:
        empty_indices.append(i)

print(f"Total: {len(all_entries)}")
print(f"Empty name entries: {len(empty_indices)}")
print(f"Indices: {empty_indices}")
print()
# Show context around each empty range
ranges = []
start = None
for i in empty_indices:
    if start is None:
        start = i
        end = i
    elif i == end + 1:
        end = i
    else:
        ranges.append((start, end))
        start = i
        end = i
if start is not None:
    ranges.append((start, end))
print("Empty ranges (with surrounding class info):")
for s, e in ranges:
    cls_before = all_entries[s-1][3] if s > 0 else "n/a"
    cls_after = all_entries[e+1][3] if e+1 < len(all_entries) else "n/a"
    # Class of empty entries
    cls_empty = set(all_entries[i][3] for i in range(s, e+1))
    name_before = all_entries[s-1][1] if s > 0 else "<begin>"
    name_after = all_entries[e+1][1] if e+1 < len(all_entries) else "<end>"
    print(f"  [{s}..{e}] count={e-s+1} cls={cls_empty} | before={name_before!r}(cls={cls_before}) | after={name_after!r}(cls={cls_after})")
