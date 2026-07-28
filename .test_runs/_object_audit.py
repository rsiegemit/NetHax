"""Full object-catalog DATA audit: minihax OBJECTS table vs vendor NLE.

For every object index (0..NUM_OBJECTS-1) compares the canonical base ``name``
and random ``description`` (appearance) against vendor NLE's
``nethack.OBJ_NAME`` / ``nethack.OBJ_DESCR``.  This validates the object table
that drives every glyph + inv_strs rendering (independent of the per-game
descr_idx shuffle, which is a separate RNG concern the char sweep covers).

Usage:
    NETHAX_EAGER=1 JAX_PLATFORMS=cpu .venv/bin/python .test_runs/_object_audit.py
    ... [--class WAND] [--show N]
"""
from __future__ import annotations

import argparse

# Reuse the harness setup (gym shim, parity mode, sys.path).
import minihax_byteparity as H  # noqa: F401,E402
from nle import nethack  # noqa: E402

from Nethax.nethax.constants.objects import OBJECTS, NUM_OBJECTS, ObjectClass  # noqa: E402


def _norm(s):
    return s if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="oclass", default=None,
                    help="Filter to one ObjectClass name (e.g. WAND).")
    ap.add_argument("--show", type=int, default=80, help="Max mismatches to list.")
    args = ap.parse_args()

    want_cls = None
    if args.oclass:
        want_cls = int(getattr(ObjectClass, args.oclass.upper() + "_CLASS",
                               getattr(ObjectClass, args.oclass.upper(), -1)))

    name_mismatch = []
    descr_mismatch = []
    checked = 0
    for i in range(NUM_OBJECTS):
        m = OBJECTS[i]
        if want_cls is not None and int(m.class_) != want_cls:
            continue
        checked += 1
        try:
            oc = nethack.objclass(i)
            vname = nethack.OBJ_NAME(oc)
            vdescr = _norm(nethack.OBJ_DESCR(oc))
        except Exception as e:  # noqa: BLE001
            name_mismatch.append((i, f"<VENDOR ERR {type(e).__name__}>", m.name))
            continue
        if vname != m.name:
            name_mismatch.append((i, vname, m.name))
        if vdescr != _norm(m.description):
            descr_mismatch.append((i, vdescr, _norm(m.description)))

    print(f"[object-audit] checked={checked} of {NUM_OBJECTS}  "
          f"name_mismatch={len(name_mismatch)}  descr_mismatch={len(descr_mismatch)}")
    if name_mismatch:
        print("[object-audit] NAME mismatches:")
        for i, v, mm in name_mismatch[:args.show]:
            print(f"    obj {i:3d} vendor={v!r} minihax={mm!r}")
    if descr_mismatch:
        print("[object-audit] DESCR (appearance) mismatches:")
        for i, v, mm in descr_mismatch[:args.show]:
            print(f"    obj {i:3d} vendor={v!r} minihax={mm!r}")


if __name__ == "__main__":
    main()
