"""Object discovery menu (``\\`` command) — vendor parity.

Mirrors vendor/nethack/src/o_init.c::dodiscovered (lines 762-873): the
``#known`` / ``\\`` command lists all object types the player has identified
or encountered, grouped by ObjectClass.  Each line is the canonical type name
prefixed with "  " (or "* " for encountered-but-not-identified types).

Wave 8c scope:
  - Only handle the "identified" path (``state.identification.identified``).
  - Group by ObjectClass, using vendor invent.c::names[] for the header.
  - Sort within a class by type_id (canonical order) — matches vendor's loop.

Returns: ``uint8[N, 80]`` bytes, one row per menu line, zero-padded.

Citations:
  - vendor/nethack/src/o_init.c::dodiscovered (line 762)
  - vendor/nethack/src/invent.c::let_to_name (line 4800)
  - vendor/nethack/src/invent.c::names[] (line 4789)
"""
from __future__ import annotations

import numpy as np

from Nethax.nethax.constants.objects import OBJECTS, ObjectClass
from Nethax.nethax.obs.inv_strs import _CLASS_HEADERS, _DEFAULT_INV_ORDER


_MENU_WIDTH = 80


def _str_to_row(s: str) -> np.ndarray:
    """Pad / truncate ``s`` to exactly _MENU_WIDTH bytes (zero-filled)."""
    b = s.encode("ascii", errors="replace")[:_MENU_WIDTH]
    out = np.zeros((_MENU_WIDTH,), dtype=np.uint8)
    out[: len(b)] = np.frombuffer(b, dtype=np.uint8)
    return out


def _display_name(obj) -> str:
    """Vendor name shown in discovery: prefixed with the class noun.

    Vendor's dodiscovered uses ``disco_append_typename`` which produces
    something like "potion of healing", "scroll of magic mapping", "wooden
    wand of striking" etc.  For simplicity (and to match the inv_strs format),
    we render "<class prefix><name>" so an identified blue potion of healing
    becomes "potion of healing", and an identified +0 long sword becomes
    "long sword".
    """
    if obj.name is None:
        return obj.description or ""
    prefix_map = {
        ObjectClass.POTION_CLASS: "potion of ",
        ObjectClass.SCROLL_CLASS: "scroll of ",
        ObjectClass.SPBOOK_CLASS: "spellbook of ",
        ObjectClass.RING_CLASS:   "ring of ",
        ObjectClass.AMULET_CLASS: "amulet of ",
    }
    prefix = prefix_map.get(obj.class_, "")
    return prefix + obj.name


def build_discovery_text(state) -> np.ndarray:
    """Build the ``\\`` (Discoveries) menu text for the current run.

    Iterates ``state.identification.identified`` (bool array indexed by
    object type_id) and emits one section per ObjectClass that has any
    identified members.  Sections appear in vendor `flags.inv_order` order
    (see vendor/nethack/src/options.c::def_inv_order).

    Returns:
        uint8[N, 80] — N rows of zero-padded ASCII bytes.  When nothing has
        been discovered, N=1 and the row reads "You haven't discovered
        anything yet..." (mirrors o_init.c line 857).
    """
    id_state = state.identification
    identified = np.asarray(id_state.identified).astype(bool)

    lines: list[str] = ["Discoveries"]

    any_disc = False
    for class_id in _DEFAULT_INV_ORDER:
        # Collect identified type_ids whose ObjectClass matches this section.
        type_ids = [
            i
            for i, obj in enumerate(OBJECTS)
            if int(obj.class_) == class_id and i < len(identified) and identified[i]
        ]
        if not type_ids:
            continue
        if not any_disc:
            any_disc = True
        header = _CLASS_HEADERS.get(class_id, "Items")
        lines.append(header)
        for tid in type_ids:
            obj = OBJECTS[tid]
            # Vendor uses "  " for fully-identified, "* " for partial; we
            # only model the identified path here.
            lines.append("  " + _display_name(obj))

    if not any_disc:
        # Vendor message when player hasn't discovered anything yet
        # (o_init.c line 857: You("haven't discovered anything yet...");).
        lines = ["You haven't discovered anything yet..."]

    rows = np.stack([_str_to_row(line) for line in lines], axis=0)
    return rows
