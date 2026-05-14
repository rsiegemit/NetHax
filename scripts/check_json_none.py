#!/usr/bin/env python3
import json
with open("/tmp/vendor_entries.json") as f:
    es = json.load(f)
e = es[319]
print("Entry 319:", e)
print("name is None:", e["name"] is None)
print("name == '':", e["name"] == "")
print()
# How many have name == None vs empty?
none_count = sum(1 for x in es if x["name"] is None)
empty_count = sum(1 for x in es if x["name"] == "")
print(f"None: {none_count}, empty: {empty_count}")
