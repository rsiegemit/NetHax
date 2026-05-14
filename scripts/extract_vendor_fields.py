#!/usr/bin/env python3
"""Extract full field data from vendor objects.c for each macro invocation.

For each entry, output a JSON record with:
    index, macro, name, desc, class, prob, wt, cost, sdam, ldam, oc1, oc2,
    nutrition, color, material, sub_class (skill/armor-slot)

This is used to generate the canonical OBJECTS table.
"""
import json
import re

VENDOR_C = "/Users/rsiegelmann/Downloads/Projects/nethax/vendor/nle/src/objects.c"

with open(VENDOR_C) as f:
    text = f.read()

# Strip comments
text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
text = re.sub(r"//.*", "", text)

# Strip #if 0 ... #endif (DEFERRED)
def strip_if0(s):
    out, i = [], 0
    while i < len(s):
        m = re.search(r"#if 0\b", s[i:])
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:i + m.start()])
        depth, k = 1, i + m.end()
        while k < len(s) and depth > 0:
            m2 = re.search(r"#if\b|#endif\b", s[k:])
            if not m2:
                break
            tok = m2.group(0)
            k += m2.end()
            if tok.startswith("#if"):
                depth += 1
            elif tok == "#endif":
                depth -= 1
        i = k
    return "".join(out)


def strip_ifdef(s, names):
    pattern = r"#ifdef\s+(" + "|".join(names) + r")\b"
    out, i = [], 0
    while i < len(s):
        m = re.search(pattern, s[i:])
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:i + m.start()])
        depth, k = 1, i + m.end()
        while k < len(s) and depth > 0:
            m2 = re.search(r"#if\b|#ifdef\b|#ifndef\b|#endif\b", s[k:])
            if not m2:
                break
            tok = m2.group(0)
            k += m2.end()
            if tok in ("#if", "#ifdef", "#ifndef"):
                depth += 1
            elif tok == "#endif":
                depth -= 1
        i = k
    return "".join(out)


text = strip_if0(text)
text = strip_ifdef(text, ["MAIL"])

# Strip #define lines (multi-line via backslash)
lines = text.split("\n")
out_lines, skip = [], False
for line in lines:
    if skip:
        if not line.rstrip().endswith("\\"):
            skip = False
        continue
    if line.lstrip().startswith("#define"):
        if line.rstrip().endswith("\\"):
            skip = True
        continue
    out_lines.append(line)
text = "\n".join(out_lines)

MACROS = [
    "OBJECT", "WEAPON", "PROJECTILE", "BOW", "ARMOR", "CLOAK", "HELM",
    "SHIELD", "GLOVES", "BOOTS", "DRGN_ARMR", "FOOD", "TOOL", "CONTAINER",
    "WAND", "RING", "AMULET", "POTION", "SCROLL", "SPELL", "GEM", "ROCK",
    "ARTIFACT_GEM", "COIN", "WEPTOOL",
]


def split_args(body):
    """Split macro args at top-level commas, respecting nested parens/strings."""
    args = []
    cur = []
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == '"':
            cur.append(c)
            i += 1
            while i < n and body[i] != '"':
                if body[i] == '\\':
                    cur.append(body[i])
                    if i + 1 < n:
                        cur.append(body[i+1])
                        i += 2
                        continue
                cur.append(body[i])
                i += 1
            if i < n:
                cur.append(body[i])
                i += 1
            continue
        if c == '(':
            depth += 1
            cur.append(c)
        elif c == ')':
            depth -= 1
            cur.append(c)
        elif c == ',' and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        args.append("".join(cur).strip())
    return args


def parse_string(arg):
    """Parse string arg: 'None', '"abc"', or (char *) 0 -> None or str."""
    arg = arg.strip()
    if arg == "None" or "(char" in arg:
        return None
    m = re.match(r'^"((?:\\.|[^"\\])*)"$', arg)
    if m:
        return m.group(1)
    return arg  # numeric/identifier passthrough


def find_invocations(s):
    """Find all top-level macro invocations."""
    results = []
    i, n = 0, len(s)
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
                    if s[j] == '\\':
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
        results.append((macro, body, start))
        i = j + 1
    return results


invocations = find_invocations(text)

# Parse each entry
entries = []
for macro, body, pos in invocations:
    args = split_args(body)
    # Each macro has its own signature. Decode based on macro:
    entry = None
    if macro == "OBJECT":
        # OBJECT(OBJ(name,desc), BITS(...), prp, sym, prob, dly, wt, cost,
        #        sdam, ldam, oc1, oc2, nut, color)
        if len(args) < 14:
            continue
        # first arg is OBJ(name,desc)
        obj_match = re.match(r'OBJ\s*\((.*)\)\s*$', args[0], re.DOTALL)
        if not obj_match:
            continue
        obj_args = split_args(obj_match.group(1))
        if len(obj_args) < 2:
            continue
        name = parse_string(obj_args[0])
        desc = parse_string(obj_args[1])
        # Skip terminator
        if name is None and desc is None:
            continue
        # BITS args: BITS(nmkn,mrg,uskn,ctnr,mgc,chrg,uniq,nwsh,big,tuf,dir,sub,mtrl)
        bits_match = re.match(r'BITS\s*\((.*)\)\s*$', args[1], re.DOTALL)
        bits_args = split_args(bits_match.group(1)) if bits_match else []
        sub = bits_args[11] if len(bits_args) > 11 else "0"
        material = bits_args[12] if len(bits_args) > 12 else "0"
        prp = args[2]
        sym = args[3]   # class
        prob = args[4]
        dly = args[5]
        wt = args[6]
        cost = args[7]
        sdam = args[8]
        ldam = args[9]
        oc1 = args[10]
        oc2 = args[11]
        nut = args[12]
        color = args[13]
        entry = dict(
            macro=macro, name=name, desc=desc, class_=sym, prob=prob,
            wt=wt, cost=cost, sdam=sdam, ldam=ldam, oc1=oc1, oc2=oc2,
            nutrition=nut, color=color, material=material, sub=sub,
        )
    elif macro == "PROJECTILE":
        # PROJECTILE(name,desc,kn,prob,wt,cost,sdam,ldam,hitbon,metal,sub,color)
        if len(args) < 12:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="WEAPON_CLASS",
            prob=args[3], wt=args[4], cost=args[5], sdam=args[6], ldam=args[7],
            oc1=args[8], oc2="0", nutrition=args[4], color=args[11],
            material=args[9], sub=args[10],
        )
    elif macro == "WEAPON":
        # WEAPON(name,desc,kn,mg,bi,prob,wt,cost,sdam,ldam,hitbon,typ,sub,metal,color)
        if len(args) < 15:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="WEAPON_CLASS",
            prob=args[5], wt=args[6], cost=args[7], sdam=args[8], ldam=args[9],
            oc1=args[10], oc2="0", nutrition=args[6], color=args[14],
            material=args[13], sub=args[12],
        )
    elif macro == "BOW":
        # BOW(name,desc,kn,prob,wt,cost,hitbon,metal,sub,color)
        if len(args) < 10:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="WEAPON_CLASS",
            prob=args[3], wt=args[4], cost=args[5], sdam="2", ldam="2",
            oc1=args[6], oc2="0", nutrition=args[4], color=args[9],
            material=args[7], sub=args[8],
        )
    elif macro == "ARMOR":
        # ARMOR(name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,sub,metal,c)
        if len(args) < 15:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        # OC1 = 10 - ac
        try:
            oc1 = str(10 - int(args[10]))
        except ValueError:
            oc1 = f"(10 - {args[10]})"
        entry = dict(
            macro=macro, name=name, desc=desc, class_="ARMOR_CLASS",
            prob=args[6], wt=args[8], cost=args[9], sdam="0", ldam="0",
            oc1=oc1, oc2=args[11], nutrition=args[8], color=args[14],
            material=args[13], sub=args[12],
        )
    elif macro in ("HELM", "CLOAK", "GLOVES", "BOOTS"):
        # WRAPPER(name,desc,kn,mgc,power,prob,delay,wt,cost,ac,can,metal,c)
        # Equivalent to ARMOR(name, desc, kn, mgc, 0, power, prob, delay, wt,
        #                      cost, ac, can, ARM_HELM, metal, c)
        sub_map = {"HELM": "ARM_HELM", "CLOAK": "ARM_CLOAK",
                   "GLOVES": "ARM_GLOVES", "BOOTS": "ARM_BOOTS"}
        if len(args) < 13:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        try:
            oc1 = str(10 - int(args[9]))
        except ValueError:
            oc1 = f"(10 - {args[9]})"
        entry = dict(
            macro=macro, name=name, desc=desc, class_="ARMOR_CLASS",
            prob=args[5], wt=args[7], cost=args[8], sdam="0", ldam="0",
            oc1=oc1, oc2=args[10], nutrition=args[7], color=args[12],
            material=args[11], sub=sub_map[macro],
        )
    elif macro == "SHIELD":
        # SHIELD(name,desc,kn,mgc,blk,power,prob,delay,wt,cost,ac,can,metal,c)
        if len(args) < 14:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        try:
            oc1 = str(10 - int(args[10]))
        except ValueError:
            oc1 = f"(10 - {args[10]})"
        entry = dict(
            macro=macro, name=name, desc=desc, class_="ARMOR_CLASS",
            prob=args[6], wt=args[8], cost=args[9], sdam="0", ldam="0",
            oc1=oc1, oc2=args[11], nutrition=args[8], color=args[13],
            material=args[12], sub="ARM_SHIELD",
        )
    elif macro == "DRGN_ARMR":
        # DRGN_ARMR(name,mgc,power,cost,ac,color)
        # -> ARMOR(name, None, 1, mgc, 1, power, 0, 5, 40, cost, ac, 0,
        #          ARM_SUIT, DRAGON_HIDE, color)
        if len(args) < 6:
            continue
        name = parse_string(args[0])
        desc = None
        try:
            oc1 = str(10 - int(args[4]))
        except ValueError:
            oc1 = f"(10 - {args[4]})"
        entry = dict(
            macro=macro, name=name, desc=desc, class_="ARMOR_CLASS",
            prob="0", wt="40", cost=args[3], sdam="0", ldam="0",
            oc1=oc1, oc2="0", nutrition="40", color=args[5],
            material="DRAGON_HIDE", sub="ARM_SUIT",
        )
    elif macro == "FOOD":
        # FOOD(name,prob,wt,cost,tin,material,nutrition,color)
        if len(args) < 8:
            continue
        name = parse_string(args[0])
        desc = None
        entry = dict(
            macro=macro, name=name, desc=desc, class_="FOOD_CLASS",
            prob=args[1], wt=args[2], cost=args[3], sdam="0", ldam="0",
            oc1=args[4], oc2="0", nutrition=args[6], color=args[7],
            material=args[5], sub="0",
        )
    elif macro == "TOOL":
        # TOOL(name,desc,kn,mrg,mgc,chg,prob,wt,cost,mat,color)
        if len(args) < 11:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="TOOL_CLASS",
            prob=args[6], wt=args[7], cost=args[8], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition=args[7], color=args[10],
            material=args[9], sub="0",
        )
    elif macro == "WEPTOOL":
        # WEPTOOL(name,desc,kn,mgc,bi,prob,wt,cost,sdam,ldam,hitbon,sub,mat,clr) 14 args
        if len(args) < 14:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="TOOL_CLASS",
            prob=args[5], wt=args[6], cost=args[7], sdam=args[8], ldam=args[9],
            oc1=args[10], oc2="0", nutrition=args[6], color=args[13],
            material=args[12], sub=args[11],
        )
    elif macro == "CONTAINER":
        # CONTAINER(name,desc,kn,mgc,chg,prob,wt,cost,mat,color)
        if len(args) < 10:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="TOOL_CLASS",
            prob=args[5], wt=args[6], cost=args[7], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition=args[6], color=args[9],
            material=args[8], sub="0",
        )
    elif macro == "WAND":
        # WAND(name,typ,prob,cost,mgc,dir,metal,color)  -- 8 args
        if len(args) < 8:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="WAND_CLASS",
            prob=args[2], wt="7", cost=args[3], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="30", color=args[7],
            material=args[6], sub="0",
        )
    elif macro == "RING":
        # RING(name,stone,power,cost,mgc,spec,mohs,metal,color)  -- 9 args
        if len(args) < 9:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="RING_CLASS",
            prob="0", wt="3", cost=args[3], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="15", color=args[8],
            material=args[7], sub="0",
        )
    elif macro == "AMULET":
        # AMULET(name,desc,power,prob)
        if len(args) < 4:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="AMULET_CLASS",
            prob=args[3], wt="20", cost="150", sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="20", color="HI_METAL",
            material="IRON", sub="0",
        )
    elif macro == "POTION":
        # POTION(name,desc,mgc,power,prob,cost,color)
        if len(args) < 7:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="POTION_CLASS",
            prob=args[4], wt="20", cost=args[5], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="10", color=args[6],
            material="GLASS", sub="0",
        )
    elif macro == "SCROLL":
        # SCROLL(name,text,mgc,prob,cost)
        if len(args) < 5:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="SCROLL_CLASS",
            prob=args[3], wt="5", cost=args[4], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="6", color="HI_PAPER",
            material="PAPER", sub="0",
        )
    elif macro == "SPELL":
        # SPELL(name,desc,sub,prob,delay,level,mgc,dir,color)
        # Defined in objects.c around line 891:
        # SPELL(name,desc,sub,prob,delay,level,spec,dir,color)
        if len(args) < 9:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="SPBOOK_CLASS",
            prob=args[3], wt="50", cost="100", sdam="0", ldam="0",
            oc1="0", oc2=args[5], nutrition="50", color=args[8],
            material="PAPER", sub=args[2],
        )
    elif macro == "GEM":
        # GEM(name,desc,prob,wt,gval,nutr,mohs,glass,color)
        if len(args) < 9:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        # Determine material by glass flag
        glass = args[7]
        material = "GLASS" if glass != "0" else "GEMSTONE"
        entry = dict(
            macro=macro, name=name, desc=desc, class_="GEM_CLASS",
            prob=args[2], wt=args[3], cost=args[4], sdam="3", ldam="3",
            oc1="0", oc2="0", nutrition=args[5], color=args[8],
            material=material, sub="0",
        )
    elif macro == "ROCK":
        # ROCK(name,desc,kn,prob,wt,gval,sdmg,ldmg,mgc,mtrl,color)
        if len(args) < 11:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="GEM_CLASS",
            prob=args[3], wt=args[4], cost=args[5], sdam=args[6], ldam=args[7],
            oc1="0", oc2="0", nutrition=args[4], color=args[10],
            material=args[9], sub="0",
        )
    elif macro == "ARTIFACT_GEM":
        if len(args) < 4:
            continue
        name = parse_string(args[0])
        desc = parse_string(args[1])
        entry = dict(
            macro=macro, name=name, desc=desc, class_="GEM_CLASS",
            prob="0", wt="1", cost="0", sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="1", color=args[3] if len(args) > 3 else "CLR_WHITE",
            material="MINERAL", sub="0",
        )
    elif macro == "COIN":
        # COIN(name,prob,metal,worth)
        if len(args) < 4:
            continue
        name = parse_string(args[0])
        desc = None
        entry = dict(
            macro=macro, name=name, desc=desc, class_="COIN_CLASS",
            prob=args[1], wt="1", cost=args[3], sdam="0", ldam="0",
            oc1="0", oc2="0", nutrition="0", color="HI_GOLD",
            material=args[2], sub="0",
        )

    if entry is not None:
        if entry["name"] is None and entry["desc"] is None:
            continue  # terminator
        entries.append(entry)

print(f"Total parsed: {len(entries)}")

# Write JSON
with open("/tmp/vendor_entries.json", "w") as f:
    json.dump(entries, f, indent=2)
print("Saved to /tmp/vendor_entries.json")

# Quick spot check
for i in (0, 1, 3, 124, 130, 451, 452):
    if i < len(entries):
        e = entries[i]
        print(f"  {i:3d} {e['macro']:10s} {e['name']!r:30s} cls={e['class_']} wt={e['wt']} cost={e['cost']}")
