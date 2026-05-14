#!/usr/bin/env python3
import re

# Test on a sample line
test = ' SCROLL(None,      "FOOBIE BLETCH",  1,   0, 100)'
# Strip macro name
s = test.split("(", 1)[1].rsplit(")", 1)[0]
print("Body:", repr(s))

# Try regex
m2 = re.match(
    r'\s*(None|"([^"]*)"|\(char\s*\*\)\s*0)\s*,\s*(None|"([^"]*)"|\(char\s*\*\)\s*0)',
    s,
)
if m2:
    print("Matched m2")
    print(" g1 (name token):", m2.group(1))
    print(" g2 (name str):  ", m2.group(2))
    print(" g3 (desc token):", m2.group(3))
    print(" g4 (desc str):  ", m2.group(4))
else:
    print("No m2 match")
