#!/usr/bin/env python3
"""Parse vendor/nle/src/objects.c to extract canonical object list.

Outputs /tmp/vendor_objects_raw.txt:
    index\tmacro\tname\tdescription
"""
import re

VENDOR_C = "/Users/rsiegelmann/Downloads/Projects/nethax/vendor/nle/src/objects.c"

with open(VENDOR_C) as f:
    text = f.read()

MACROS = [
    "OBJECT", "WEAPON", "PROJECTILE", "BOW", "ARMOR", "CLOAK", "HELM",
    "SHIELD", "GLOVES", "BOOTS", "DRGN_ARMR", "FOOD", "TOOL", "CONTAINER",
    "WAND", "RING", "AMULET", "POTION", "SCROLL", "SPELL", "GEM", "ROCK",
    "ARTIFACT_GEM", "COIN", "WEPTOOL",
]

# Strip C comments
text_no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
text_no_comments = re.sub(r"//.*", "", text_no_comments)

# Strip #if 0 ... #endif blocks (DEFERRED entries)
def strip_if0(s):
    out = []
    i = 0
    while i < len(s):
        m = re.search(r"#if 0\b", s[i:])
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:i + m.start()])
        # find matching #endif (no #if nesting expected in this file but handle simply)
        depth = 1
        k = i + m.end()
        while k < len(s) and depth > 0:
            m2 = re.search(r"#if\b|#endif\b", s[k:])
            if not m2:
                break
            tok = m2.group(0)
            k = k + m2.end()
            if tok.startswith("#if"):
                depth += 1
            elif tok == "#endif":
                depth -= 1
        i = k
    return "".join(out)

text_no_comments = strip_if0(text_no_comments)

# Strip #ifdef MAIL ... #endif (NLE builds with MAIL undefined)
def strip_ifdef(s, names):
    pattern = r"#ifdef\s+(" + "|".join(names) + r")\b"
    out = []
    i = 0
    while i < len(s):
        m = re.search(pattern, s[i:])
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:i + m.start()])
        depth = 1
        k = i + m.end()
        while k < len(s) and depth > 0:
            m2 = re.search(r"#if\b|#ifdef\b|#ifndef\b|#endif\b", s[k:])
            if not m2:
                break
            tok = m2.group(0)
            k = k + m2.end()
            if tok in ("#if", "#ifdef", "#ifndef"):
                depth += 1
            elif tok == "#endif":
                depth -= 1
        i = k
    return "".join(out)

text_no_comments = strip_ifdef(text_no_comments, ["MAIL"])

# Remove #define lines (which may span continuation lines)
lines = text_no_comments.split("\n")
out_lines = []
skip_continuation = False
for line in lines:
    if skip_continuation:
        if not line.rstrip().endswith("\\"):
            skip_continuation = False
        continue
    if line.lstrip().startswith("#define"):
        if line.rstrip().endswith("\\"):
            skip_continuation = True
        continue
    out_lines.append(line)
src = "\n".join(out_lines)


def parse_macros(s):
    results = []
    i = 0
    n = len(s)
    while i < n:
        m = re.match(r"\b(" + "|".join(MACROS) + r")\s*\(", s[i:])
        if not m:
            i += 1
            continue
        macro = m.group(1)
        start = i + m.end()
        depth = 1
        j = start
        while j < n and depth > 0:
            c = s[j]
            if c == '"':
                j += 1
                while j < n and s[j] != '"':
                    if s[j] == "\\":
                        j += 2
                        continue
                    j += 1
                j += 1
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = s[start:j]
        results.append((macro, body))
        i = j + 1
    return results


invocations = parse_macros(src)

objects = []
for macro, body in invocations:
    s = body.strip()
    # OBJ("name"|None, "desc"|None) form (used by OBJECT(...))
    m = re.match(
        r'OBJ\s*\(\s*(None|"([^"]*)"|\(char\s*\*\)\s*0)\s*,\s*(None|"([^"]*)"|\(char\s*\*\)\s*0)',
        s,
    )
    if m:
        name = m.group(2) if m.group(2) is not None else None
        desc = m.group(4) if m.group(4) is not None else None
        # Skip terminator (both None)
        if name is None and desc is None:
            continue
        objects.append((macro, name, desc, body))
        continue
    # wrapper macros with "name", "desc"
    m2 = re.match(
        r'\s*(None|"([^"]*)"|\(char\s*\*\)\s*0)\s*,\s*(None|"([^"]*)"|\(char\s*\*\)\s*0)',
        s,
    )
    if m2:
        name = m2.group(2) if m2.group(2) is not None else None
        desc = m2.group(4) if m2.group(4) is not None else None
        if name is None and desc is None:
            continue
        objects.append((macro, name, desc, body))
        continue
    # Fallback: macro("name", <numeric or other>) — used by FOOD, DRGN_ARMR, ARTIFACT_GEM
    m3 = re.match(r'\s*(None|"([^"]*)")\s*,', s)
    if m3:
        name = m3.group(2) if m3.group(2) is not None else None
        objects.append((macro, name, None, body))
        continue

# Keep None-named entries (shuffled appearance slots). Only drop entries where
# BOTH name and desc are missing (would be the terminator only).
objects = [o for o in objects if (o[1] is not None) or (o[2] is not None)]

print(f"Total: {len(objects)} entries")

with open("/tmp/vendor_objects_raw.txt", "w") as f:
    for i, (macro, name, desc, body) in enumerate(objects):
        f.write(f"{i}\t{macro}\t{name}\t{desc or ''}\n")

for i, (macro, name, desc, body) in enumerate(objects[:8]):
    print(f"{i}: {macro:12s} {name!r:30s} desc={desc!r}")
print("...")
for i in range(max(0, len(objects)-8), len(objects)):
    macro, name, desc, body = objects[i]
    print(f"{i}: {macro:12s} {name!r:30s} desc={desc!r}")
