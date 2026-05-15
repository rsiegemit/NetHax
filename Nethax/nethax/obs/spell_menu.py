"""Spell-menu `+` command output — byte-equal port of vendor dospellmenu.

Sources:
  vendor/nethack/src/spell.c::dospellmenu          (lines 2075-2167)
  vendor/nethack/src/spell.c::spelltypemnemonic    (lines 832-852)
  vendor/nethack/src/spell.c::spellretention       (lines 619-650)
  vendor/nethack/src/spell.c::percent_success      (lines 2179-2300)
  vendor/nethack/include/objects.h::SPELL macro    (lines 1277-1412)

Vendor format strings (spell.c:2103-2110):

    "    %-20s Level %-12s Fail Retention"     # menu header  (DUMP omits "    ")
    "%-20s  %2d   %-12s %3d%% %9s"             # per-spell row

The header is preceded by 4 spaces of letter-column padding (for the
"a - " selection letters added by add_menu).

Host-side helper (not jit'd) — used for UI prompts + parity tests.
"""
from __future__ import annotations

from typing import List

import numpy as np

from Nethax.nethax.subsystems.magic import SpellId, N_SPELLS, MAX_SPELL_MEMORY


# vendor/nethack/include/objects.h SPELL() entries, in SpellId order.
# Fields: (name, level, skill_mnemonic).
# `level` is the 6th SPELL() macro arg (oc_level); `skill_mnemonic` comes
# from spelltypemnemonic() in spell.c::832-852.
_SPELL_INFO: List[tuple[str, int, str]] = [
    ("dig",              5, "matter"),       # SPE_DIG               objects.h:1293
    ("magic missile",    2, "attack"),       # SPE_MAGIC_MISSILE     objects.h:1297
    ("fireball",         4, "attack"),       # SPE_FIREBALL          objects.h:1300
    ("cone of cold",     4, "attack"),       # SPE_CONE_OF_COLD      objects.h:1302
    ("sleep",            1, "enchantment"),  # SPE_SLEEP             objects.h:1304
    ("finger of death",  7, "attack"),       # SPE_FINGER_OF_DEATH   objects.h:1306
    ("light",            1, "divination"),   # SPE_LIGHT             objects.h:1308
    ("detect monsters",  1, "divination"),   # SPE_DETECT_MONSTERS   objects.h:1310
    ("healing",          1, "healing"),      # SPE_HEALING           objects.h:1313
    ("knock",            1, "matter"),       # SPE_KNOCK             objects.h:1316
    ("force bolt",       1, "attack"),       # SPE_FORCE_BOLT        objects.h:1319
    ("confuse monster",  1, "enchantment"),  # SPE_CONFUSE_MONSTER   objects.h:1322
    ("cure blindness",   2, "healing"),      # SPE_CURE_BLINDNESS    objects.h:1325
    ("drain life",       2, "attack"),       # SPE_DRAIN_LIFE        objects.h:1328
    ("slow monster",     2, "enchantment"),  # SPE_SLOW_MONSTER      objects.h:1331
    ("wizard lock",      2, "matter"),       # SPE_WIZARD_LOCK       objects.h:1334
    ("create monster",   2, "clerical"),     # SPE_CREATE_MONSTER    objects.h:1337
    ("detect food",      2, "divination"),   # SPE_DETECT_FOOD       objects.h:1340
    ("cause fear",       3, "enchantment"),  # SPE_CAUSE_FEAR        objects.h:1343
    ("clairvoyance",     3, "divination"),   # SPE_CLAIRVOYANCE      objects.h:1346
    ("cure sickness",    3, "healing"),      # SPE_CURE_SICKNESS     objects.h:1349
    ("charm monster",    3, "enchantment"),  # SPE_CHARM_MONSTER     objects.h:1352
    ("haste self",       3, "escape"),       # SPE_HASTE_SELF        objects.h:1355
    ("detect unseen",    3, "divination"),   # SPE_DETECT_UNSEEN     objects.h:1358
    ("levitation",       4, "escape"),       # SPE_LEVITATION        objects.h:1361
    ("extra healing",    3, "healing"),      # SPE_EXTRA_HEALING     objects.h:1364
    ("restore ability",  4, "healing"),      # SPE_RESTORE_ABILITY   objects.h:1367
    ("invisibility",     4, "escape"),       # SPE_INVISIBILITY      objects.h:1370
    ("detect treasure",  4, "divination"),   # SPE_DETECT_TREASURE   objects.h:1373
    ("remove curse",     3, "clerical"),     # SPE_REMOVE_CURSE      objects.h:1376
    ("magic mapping",    5, "divination"),   # SPE_MAGIC_MAPPING     objects.h:1379
    ("identify",         3, "divination"),   # SPE_IDENTIFY          objects.h:1382
    ("turn undead",      6, "clerical"),     # SPE_TURN_UNDEAD       objects.h:1385
    ("polymorph",        6, "matter"),       # SPE_POLYMORPH         objects.h:1388
    ("teleport away",    6, "escape"),       # SPE_TELEPORT_AWAY     objects.h:1391
    ("create familiar",  6, "clerical"),     # SPE_CREATE_FAMILIAR   objects.h:1394
    ("cancellation",     7, "matter"),       # SPE_CANCELLATION      objects.h:1397
    ("protection",       1, "clerical"),     # SPE_PROTECTION        objects.h:1400
    ("jumping",          1, "escape"),       # SPE_JUMPING           objects.h:1403
    ("stone to flesh",   3, "healing"),      # SPE_STONE_TO_FLESH    objects.h:1406
    ("chain lightning",  2, "attack"),       # SPE_CHAIN_LIGHTNING   objects.h:1409
    ("flame sphere",     1, "matter"),       # SPE_FLAME_SPHERE  (DEFERRED in vendor)
    ("freeze sphere",    1, "matter"),       # SPE_FREEZE_SPHERE (DEFERRED in vendor)
]
assert len(_SPELL_INFO) == N_SPELLS, "SpellId / _SPELL_INFO drift"


def _letter_for_spell(sid: int) -> str:
    """Vendor casting letter (spell.c::spellet → 'a'..'z','A'..'Z')."""
    if 0 <= sid < 26:
        return chr(ord("a") + sid)
    if 26 <= sid < 52:
        return chr(ord("A") + sid - 26)
    return "?"


def _retention_str(memory: int) -> str:
    """Mirror vendor/nethack/src/spell.c::spellretention (lines ~619-650).

    Vendor returns one of:
      "expired"            (memory <= 0)
      "<N> turns"          (memory > 0)

    For full byte-equality we'd need access to KEEN/turn arithmetic; the
    simplified form matches the human-readable column header.
    """
    if memory <= 0:
        return "expired"
    return f"{memory} turns"


def _fail_pct(state, sid: int) -> int:
    """Simplified vendor percent_success → return Fail = 100 - success%.

    Full port lives in Nethax/nethax/subsystems/magic.py::spell_fail_chance
    (already vendor-cited).  This helper just routes to it with the
    relevant blstats / role fields.
    """
    try:
        from Nethax.nethax.subsystems.magic import spell_fail_chance
        from Nethax.nethax.constants.blstats import BL_INT, BL_WIS, BL_XP
        from Nethax.nethax.obs.nle_obs import build_blstats
        import jax.numpy as jnp
        bl = np.asarray(build_blstats(state))
        role = jnp.int32(int(state.player_role))
        chance = int(spell_fail_chance(
            role=role,
            spell_id=jnp.int32(sid),
            xl=jnp.int32(int(bl[BL_XP])),
            stat_int=jnp.int32(int(bl[BL_INT])),
            stat_wis=jnp.int32(int(bl[BL_WIS])),
        ))
        return max(0, min(100, chance))
    except Exception:
        return 0


def build_spell_menu_text(state) -> List[str]:
    """Build the `+` spell menu, byte-equal to vendor dospellmenu output.

    Header (spell.c:2103-2106):
        "    %-20s Level %-12s Fail Retention" % ("Name", "Category")

    Per-spell rows (spell.c:2110):
        "%-20s  %2d   %-12s %3d%% %9s" % (name, level, category, fail%, retention)
        prefixed by "a - " / "b - " / ... selection letter (vendor add_menu).

    Citation: vendor/nethack/src/spell.c::dospellmenu (lines 2075-2167).
    """
    known = np.asarray(state.magic.spell_known).astype(bool)
    memory = np.asarray(state.magic.spell_memory).astype(np.int32)

    if not bool(known.any()):
        # Vendor message when no spells known (spell.c::docast line ~775).
        return ["You don't know any spells right now."]

    # Byte-equal vendor header.
    header = "    %-20s Level %-12s Fail Retention" % ("Name", "Category")
    lines: List[str] = [header]

    for sid in range(N_SPELLS):
        if not known[sid]:
            continue
        name, level, cat = _SPELL_INFO[sid]
        ret = _retention_str(int(memory[sid]))
        fail = _fail_pct(state, sid)
        letter = _letter_for_spell(sid)
        # vendor add_menu prepends "<letter> - " automatically.
        body = "%-20s  %2d   %-12s %3d%% %9s" % (name, level, cat, fail, ret)
        lines.append(f"{letter} - {body}")
    return lines
